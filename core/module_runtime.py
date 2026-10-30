"""Optional dependencies and model I/O. Importing this file needs only stdlib."""

from __future__ import annotations

import asyncio
import contextvars
import importlib
import importlib.metadata
import importlib.util
import logging
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

_ROOT = Path(__file__).resolve().parents[1]
_PIP_TASK = contextvars.ContextVar("kuang_pip_task", default=None)
_IMPORT_NAMES = {"Pillow": "PIL", "opencv-python-headless": "cv2"}
_CV_DISTRIBUTIONS = (
    "opencv-python-headless",
    "opencv-python",
    "opencv-contrib-python-headless",
    "opencv-contrib-python",
)


class RestartRequired(RuntimeError):
    pass


def requirements_for(module_id: str) -> tuple[str, ...]:
    profile = "gif" if module_id == "gif" else "runtime"
    return tuple(
        line.strip()
        for line in (_ROOT / "requirements" / f"{profile}.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    )


def restore_dependencies(module_id: str) -> None:
    """Restore desktop packages after restart using AstrBot's own import policy.

    Without a root requirements.txt, the host cannot preload these optional
    packages for us. This operation only loads already installed packages.
    """
    try:
        from astrbot.core import pip_installer
    except ImportError:
        return
    prefer = getattr(pip_installer, "prefer_installed_dependencies", None)
    if callable(prefer):
        profile = "gif" if module_id == "gif" else "runtime"
        prefer(str(_ROOT / "requirements" / f"{profile}.txt"))


def missing_requirements(requirements: tuple[str, ...]) -> list[str]:
    """Inspect metadata without importing NumPy, OpenCV or ONNX Runtime."""
    missing = []
    for spec in requirements:
        # packaging is supplied by AstrBot; defer it so the core can also be
        # imported by a completely empty Python environment.
        try:
            from packaging.requirements import Requirement
        except ImportError:
            return list(requirements)
        req = Requirement(spec)
        module = _IMPORT_NAMES.get(req.name, req.name)
        names = _CV_DISTRIBUTIONS if module == "cv2" else (req.name,)
        versions = []
        for name in names:
            try:
                versions.append(importlib.metadata.version(name))
            except importlib.metadata.PackageNotFoundError:
                pass
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if not found or not any(req.specifier.contains(v) for v in versions):
            missing.append(spec)
    return missing


async def install_requirements(requirements: tuple[str, ...], log) -> None:
    """Use AstrBot's configured mirror and installer, including desktop support."""
    from astrbot.core import pip_installer

    loop = asyncio.get_running_loop()
    marker = object()

    class InstallLog(logging.Handler):
        def emit(self, record):
            if _PIP_TASK.get() is marker:
                loop.call_soon_threadsafe(log, record.getMessage())

    handler = InstallLog()
    pip_logger = logging.getLogger("astrbot")
    token = _PIP_TASK.set(marker)
    pip_logger.addHandler(handler)
    try:
        await pip_installer.install(package_name=" ".join(requirements))
    finally:
        pip_logger.removeHandler(handler)
        _PIP_TASK.reset(token)
        importlib.invalidate_caches()


def check_loaded_versions(module_id: str) -> None:
    from packaging.requirements import Requirement
    from packaging.version import Version

    for spec in requirements_for(module_id):
        req = Requirement(spec)
        module = sys.modules.get(_IMPORT_NAMES.get(req.name, req.name))
        loaded = getattr(module, "__version__", None)
        if loaded and not req.specifier.contains(loaded):
            raise RestartRequired(
                f"{req.name} 的旧版本已加载，请重启 AstrBot 后重试启用。"
            )
        if loaded and req.name != "opencv-python-headless":
            installed = importlib.metadata.version(req.name)
            if Version(loaded) != Version(installed):
                raise RestartRequired(
                    f"{req.name} 已更新，但当前进程仍使用旧版本，请重启 AstrBot 后重试启用。"
                )


def validate_module(module_id: str, model_path: Path | None) -> None:
    check_loaded_versions(module_id)
    try:
        import numpy as np
        from PIL import Image

        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).convert("RGB")
        if module_id == "gif":
            import cv2

            frame = np.zeros((32, 32), dtype=np.uint8)
            cv2.calcOpticalFlowPyrLK(
                frame, frame, np.array([[[16, 16]]], dtype=np.float32), None
            )
            return
        # Import here before the backends translate ImportError into their
        # general inference error, so a native-loader failure isn't mistaken
        # for a damaged model and followed by an unnecessary model download.
        importlib.import_module("onnxruntime")
        from .saliency_backend import SaliencyDetectionBackend
        from .yolo_backend import AnimeYoloDetectionBackend, Yolo26nDetectionBackend

        backend_cls = {
            "general": Yolo26nDetectionBackend,
            "anime": AnimeYoloDetectionBackend,
            "saliency": SaliencyDetectionBackend,
        }[module_id]
        backend = backend_cls(
            model_path=str(model_path),
            auto_download_model=False,
            intra_op_num_threads=1,
        )
        session = backend._get_session()
        width, height = backend._input_size
        outputs = session.run(
            None,
            {backend._input_name: np.zeros((1, 3, height, width), dtype=np.float32)},
        )
        if not outputs or not all(np.isfinite(output).all() for output in outputs):
            raise RuntimeError("模型试运行返回了无效输出。")
    except (ImportError, OSError) as exc:
        raise RestartRequired(
            f"依赖已安装，但当前进程无法加载：{exc}。请重启 AstrBot 后重试。"
        ) from exc


def download_model(url: str, destination: Path, progress, stopped) -> None:
    """Write only to a staging file. The manager validates before publishing it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "astrbot-plugin-kuang/0.9"})
    started = time.monotonic()
    last = started
    downloaded = 0
    with urlopen(request, timeout=30) as response, destination.open("wb") as target:
        length = response.headers.get("Content-Length")
        total = int(length) if length and length.isdigit() else None
        progress(0, total, 0)
        while True:
            if stopped.is_set():
                raise InterruptedError("插件正在重载，下载已停止，可重新安装。")
            chunk = response.read(256 * 1024)
            if not chunk:
                break
            target.write(chunk)
            downloaded += len(chunk)
            now = time.monotonic()
            if now - last >= 0.2:
                progress(downloaded, total, int(downloaded / max(now - started, 0.001)))
                last = now
        if downloaded == 0 or (total is not None and downloaded != total):
            raise RuntimeError("模型下载不完整，请重试。")
        progress(
            downloaded, total, int(downloaded / max(time.monotonic() - started, 0.001))
        )
