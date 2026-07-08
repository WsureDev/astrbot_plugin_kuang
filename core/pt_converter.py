from __future__ import annotations

import shutil
from pathlib import Path
from urllib.request import urlopen

from .logger import get_logger

_logger = get_logger(__name__)


def _is_onnx_loadable(onnx_path: Path) -> bool:
    try:
        import onnxruntime as ort
    except ImportError:
        _logger.debug(
            "[pt_converter] onnxruntime unavailable; skip ONNX validation: %s",
            onnx_path,
        )
        return True
    try:
        ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        return True
    except Exception as exc:
        _logger.warning(
            "[pt_converter] existing ONNX is invalid and will be regenerated: %s (%s)",
            onnx_path,
            exc,
        )
        return False


def ensure_onnx_from_pt(
    *,
    onnx_path: str | Path,
    pt_url: str,
    auto_download: bool = True,
    model_input_size: int = 640,
) -> Path:
    """Ensure an ONNX model file exists, downloading and converting from .pt if needed.

    Workflow:
      1. If onnx_path already exists → return immediately.
      2. Download the .pt file to a sibling path (same dir, .pt suffix).
      3. Convert .pt → .onnx using ultralytics export.
      4. Remove the .pt file to save disk space.

    This function is intended to be called eagerly at plugin load time
    (not lazily), so the ONNX file is ready before any inference request.
    """
    onnx_path = Path(onnx_path)
    if onnx_path.exists():
        if _is_onnx_loadable(onnx_path):
            _logger.debug("[pt_converter] ONNX already exists: %s", onnx_path)
            return onnx_path
        try:
            onnx_path.unlink()
        except OSError as exc:
            raise RuntimeError(
                f"已检测到损坏的 ONNX 文件且无法删除: {onnx_path} ({exc})"
            ) from exc

    if not auto_download:
        raise RuntimeError(
            f"未找到模型文件: {onnx_path}. "
            "请手动放置 ONNX 文件，或开启 auto_download_model。"
        )
    if not pt_url:
        raise RuntimeError("未配置 .pt 模型下载地址。")

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    pt_path = onnx_path.with_suffix(".pt")

    # Step 1: Download .pt if not already cached
    if not pt_path.exists():
        _logger.info(
            "[pt_converter] downloading .pt model: %s -> %s", pt_url, pt_path
        )
        temp_path = pt_path.with_suffix(".pt.download")
        try:
            with urlopen(pt_url, timeout=120) as response, temp_path.open("wb") as f:
                shutil.copyfileobj(response, f)
            temp_path.replace(pt_path)
            _logger.info("[pt_converter] .pt download completed: %s", pt_path)
        except Exception as exc:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise RuntimeError(
                f".pt 模型下载失败，请检查网络或手动放置文件: {exc}"
            ) from exc
    else:
        _logger.debug("[pt_converter] .pt already cached: %s", pt_path)

    # Step 2: Convert .pt → .onnx
    _logger.info("[pt_converter] converting .pt -> .onnx: %s", pt_path)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "缺少 ultralytics 依赖，无法自动将 .pt 模型转换为 ONNX。"
            "请安装: pip install ultralytics"
        ) from exc

    try:
        model = YOLO(str(pt_path))
        export_path = model.export(format="onnx", imgsz=model_input_size, simplify=True)
        exported = Path(export_path)
        # ultralytics exports to same dir with .onnx suffix; move if needed
        if exported != onnx_path:
            exported.replace(onnx_path)
        _logger.info("[pt_converter] ONNX conversion completed: %s", onnx_path)
    except Exception as exc:
        raise RuntimeError(
            f".pt → ONNX 转换失败: {exc}"
        ) from exc

    # Step 3: Clean up .pt to save disk space
    try:
        pt_path.unlink(missing_ok=True)
        _logger.debug("[pt_converter] removed .pt after conversion: %s", pt_path)
    except OSError as exc:
        _logger.warning("[pt_converter] failed to remove .pt: %s", exc)

    return onnx_path
