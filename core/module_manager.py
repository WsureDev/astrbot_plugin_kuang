"""Four optional capabilities, one shared installer and persistent progress."""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from pathlib import Path

from . import module_runtime as runtime
from .saliency_backend import DEFAULT_SALIENCY_MODEL_URL
from .yolo_backend import DEFAULT_ANIME_YOLO_MODEL_URL, DEFAULT_YOLO26N_MODEL_URL

MODULES = {
    "general": {
        "name": "通用识别",
        "flag": "enable_general",
        "prefix": "",
        "file": "yolo26n.onnx",
        "url": DEFAULT_YOLO26N_MODEL_URL,
        "description": "识别人、动物和常见物体。",
        "model_bytes": 9941957,
    },
    "saliency": {
        "name": "前景识别",
        "flag": "enable_saliency",
        "prefix": "saliency_",
        "file": "u2netp.onnx",
        "url": DEFAULT_SALIENCY_MODEL_URL,
        "description": "提取不易归类的角色、玩偶和其他前景主体。",
        "model_bytes": 4574861,
    },
    "anime": {
        "name": "动漫识别",
        "flag": "enable_anime_fallback",
        "prefix": "anime_",
        "file": "anime_yolo.onnx",
        "url": DEFAULT_ANIME_YOLO_MODEL_URL,
        "description": "识别动漫头部和身体，支持大头特写。",
        "model_bytes": 80434971,
    },
    "gif": {
        "name": "GIF 加速",
        "flag": "enable_gif_acceleration",
        "prefix": "gif_",
        "file": None,
        "url": None,
        "description": "用光流跟踪减少重复识别；关闭后仍能逐帧处理 GIF。",
        "model_bytes": 0,
    },
}
_ACTIVE_PHASES = {"queued", "installing", "downloading", "validating"}
_ROOT = Path(__file__).resolve().parents[1]
_COMMON_SETTINGS = {
    "enable_auto_animation",
    "auto_animation_probability",
    "box_count",
    "line_width",
    "line_alpha",
    "avatar_url_template",
    "debug_mode",
    "fallback_policy",
    "inference_num_threads",
    "max_concurrent_images",
}
_GENERAL_SETTINGS = {
    "confidence_threshold",
    "detector_iou_threshold",
    "model_input_size",
    "model_path",
    "model_url",
    "auto_download_model",
}


class ModuleManager:
    def __init__(
        self,
        data_dir: Path,
        config: dict,
        on_change=lambda: None,
        *,
        installer=runtime.install_requirements,
        probe=runtime.missing_requirements,
        validator=runtime.validate_module,
        downloader=runtime.download_model,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.on_change = on_change
        self.installer, self.probe = installer, probe
        self.validator, self.downloader = validator, downloader
        self._state_path = self.data_dir / "modules.json"
        self._tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        self._stopped = threading.Event()
        self._last_save = 0.0
        self._missing: dict[str, list[str]] = {}
        self._schema = json.loads(
            (_ROOT / "_conf_schema.json").read_text(encoding="utf-8")
        )
        first_load = not self._state_path.exists()
        try:
            state = (
                json.loads(self._state_path.read_text(encoding="utf-8"))
                if not first_load
                else {}
            )
            if not isinstance(state, dict):
                raise ValueError("模块状态格式错误")
            self._records = state.get("modules", {})
            if not isinstance(self._records, dict):
                raise ValueError("模块状态格式错误")
        except (OSError, ValueError, TypeError):
            # Do not re-run migration and turn disabled capabilities back on.
            self._records = {}
        for module_id in MODULES:
            if not isinstance(self._records.get(module_id), dict):
                self._records[module_id] = {}
            record = self._records[module_id]
            if record.get("phase") in _ACTIVE_PHASES:
                record.update(
                    phase="interrupted", message="上次安装被中断，请点击重试。"
                )
        for module_id, spec in MODULES.items():
            path = self.model_path(module_id)
            if self.config.get(spec["flag"], False) or (
                first_load and path is not None and path.is_file()
            ):
                try:
                    runtime.restore_dependencies(module_id)
                except Exception as exc:
                    self._records[module_id].update(
                        phase="restart_required",
                        message=f"已安装依赖加载失败，请重试安装或重启 AstrBot：{exc}",
                    )
        self.refresh_packages()
        if first_load:
            self._adopt_legacy_installation()
        self._save()

    def _adopt_legacy_installation(self):
        existing = {
            key for key in MODULES if key != "gif" and self._assets_present(key)
        }
        if not existing:
            return
        backend = self.config.get("detector_backend", "yolo26n")
        self.config["enable_general"] = (
            "general" in existing and backend != "anime_yolo"
        )
        if backend in {"anime_yolo", "cascade_yolo26n_anime"}:
            self.config["enable_anime_fallback"] = True
        self.config["enable_gif_acceleration"] = (
            not self._missing["gif"]
            and int(self.config.get("gif_detection_interval", 4)) > 1
        )
        self._save_config()

    def refresh_packages(self):
        for key in MODULES:
            self._missing[key] = self.probe(runtime.requirements_for(key))

    def model_path(self, module_id: str) -> Path | None:
        spec = self._spec(module_id)
        if not spec["file"]:
            return None
        configured = str(self.config.get(f"{spec['prefix']}model_path", "")).strip()
        return (
            Path(configured).expanduser()
            if configured
            else self.data_dir / "models" / spec["file"]
        )

    def _assets_present(self, module_id):
        path = self.model_path(module_id)
        try:
            return not self._missing[module_id] and (
                path is None or (path.is_file() and path.stat().st_size > 0)
            )
        except OSError:
            return False

    def is_enabled(self, module_id: str) -> bool:
        return (
            bool(self.config.get(self._spec(module_id)["flag"], False))
            and self._assets_present(module_id)
            and self._records.get(module_id, {}).get("phase")
            not in (_ACTIVE_PHASES | {"error", "interrupted", "restart_required"})
        )

    def snapshot(self) -> dict:
        modules = []
        for key, spec in MODULES.items():
            record = self._records.get(key, {})
            installed = self._assets_present(key)
            enabled = self.is_enabled(key)
            phase = record.get("phase", "idle")
            if phase not in _ACTIVE_PHASES | {
                "error",
                "interrupted",
                "restart_required",
            }:
                phase = (
                    "enabled"
                    if enabled
                    else "installed"
                    if installed
                    else "not_installed"
                )
            modules.append(
                {
                    "id": key,
                    "name": spec["name"],
                    "description": spec["description"],
                    "model_bytes": spec["model_bytes"],
                    "installed": installed,
                    "enabled": enabled,
                    "phase": phase,
                    "message": record.get("message", ""),
                    "downloaded_bytes": record.get("downloaded_bytes", 0),
                    "total_bytes": record.get("total_bytes"),
                    "bytes_per_second": record.get("bytes_per_second", 0),
                    "updated_at_ms": record.get("updated_at_ms", 0),
                    "logs": list(record.get("logs", [])),
                    "missing_requirements": list(self._missing[key]),
                }
            )
        return {
            "modules": modules,
            "busy": bool(self._tasks),
            "updated_at_ms": time.time_ns() // 1_000_000,
        }

    def start_install(self, module_id: str, *, install: bool = True) -> dict:
        self._spec(module_id)
        if self._stopped.is_set():
            raise ValueError("插件正在重载，请稍后再试。")
        if module_id in self._tasks:
            return self.snapshot()
        self._update(
            module_id,
            phase="queued",
            message="等待安装" if install else "等待加载验证",
            logs=[],
            downloaded_bytes=0,
            total_bytes=None,
            bytes_per_second=0,
        )
        self._tasks[module_id] = asyncio.create_task(self._install(module_id, install))
        # An already enabled module may be under repair. Rebuild immediately
        # so other active modules do not keep invoking its previous backend.
        self.on_change()
        return self.snapshot()

    async def _install(self, module_id, allow_install):
        try:
            async with self._lock:
                self._check_running()
                try:
                    await asyncio.to_thread(runtime.restore_dependencies, module_id)
                except Exception as exc:
                    self._log(module_id, f"已有依赖加载未完成，将检查缺失依赖：{exc}")
                self.refresh_packages()
                missing = self._missing[module_id]
                if missing:
                    if not allow_install:
                        raise ValueError("模块依赖尚未安装，请使用“安装并启用”。")
                    self._update(
                        module_id,
                        phase="installing",
                        message="正在安装共享依赖，首次安装可能需要几分钟。",
                    )
                    self._log(module_id, "需要安装：" + ", ".join(missing))
                    await self.installer(
                        tuple(missing), lambda line: self._log(module_id, line)
                    )
                    self._check_running()
                    self.refresh_packages()
                    if self._missing[module_id]:
                        raise runtime.RestartRequired(
                            "安装已结束，但依赖尚不可用，请重启 AstrBot 后重试。"
                        )
                else:
                    self._log(module_id, "复用已安装的共享依赖。")
                path = self.model_path(module_id)
                self._update(module_id, phase="validating", message="正在加载验证")
                if path is None or path.is_file():
                    try:
                        await asyncio.to_thread(self.validator, module_id, path)
                    except runtime.RestartRequired:
                        raise
                    except Exception:
                        if path is None or not allow_install:
                            raise
                        self._log(module_id, "现有模型验证未通过，尝试重新下载。")
                        await self._download_and_validate(module_id, path)
                elif allow_install:
                    await self._download_and_validate(module_id, path)
                else:
                    raise ValueError("模型文件不存在，请使用“安装并启用”。")
                self._check_running()
                self._update(module_id, phase="ready", message="验证通过，已启用。")
                self._set_enabled(module_id, True)
        except runtime.RestartRequired as exc:
            self._update(module_id, phase="restart_required", message=str(exc))
        except (InterruptedError, asyncio.CancelledError):
            self._update(
                module_id, phase="interrupted", message="安装被中断，可点击重试。"
            )
        except Exception as exc:
            self._update(
                module_id, phase="error", message=str(exc) or type(exc).__name__
            )
            self._log(module_id, str(exc))
        finally:
            self._tasks.pop(module_id, None)

    async def _download_and_validate(self, module_id, path):
        spec = self._spec(module_id)
        if not self.config.get(f"{spec['prefix']}auto_download_model", True):
            raise ValueError(
                f"模型不存在或不可用，且自动下载已关闭。请手动放置模型：{path}"
            )
        url = (
            str(self.config.get(f"{spec['prefix']}model_url", "")).strip()
            or spec["url"]
        )
        staging = path.with_suffix(path.suffix + ".part")
        loop = asyncio.get_running_loop()

        def progress(done, total, speed):
            loop.call_soon_threadsafe(
                self._download_progress, module_id, done, total, speed
            )

        self._update(
            module_id,
            phase="downloading",
            message="正在下载模型",
            downloaded_bytes=0,
            total_bytes=None,
        )
        await asyncio.to_thread(self.downloader, url, staging, progress, self._stopped)
        self._check_running()
        self._update(module_id, phase="validating", message="正在验证模型和试运行")
        await asyncio.to_thread(self.validator, module_id, staging)
        self._check_running()
        staging.replace(path)

    def _download_progress(self, module_id, done, total, speed):
        self._update(
            module_id, downloaded_bytes=done, total_bytes=total, bytes_per_second=speed
        )

    def disable(self, module_id: str) -> dict:
        self._spec(module_id)
        if module_id in self._tasks:
            raise ValueError("模块正在安装，请等待安装结束后关闭。")
        self._set_enabled(module_id, False)
        self._update(module_id, phase="idle", message="已关闭，保留已安装资源。")
        return self.snapshot()

    def _set_enabled(self, module_id, enabled):
        self.config[self._spec(module_id)["flag"]] = enabled
        self._save_config()
        self.on_change()

    def settings(self, module_id: str) -> dict:
        if module_id == "common":
            keys = _COMMON_SETTINGS
        elif module_id == "general":
            keys = _GENERAL_SETTINGS
        else:
            spec = self._spec(module_id)
            keys = {key for key in self._schema if key.startswith(spec["prefix"])}
            if module_id == "gif":
                keys |= {key for key in self._schema if key.startswith("motion_")} | {
                    "enable_motion"
                }
        return {
            key: {
                **self._schema[key],
                "value": self.config.get(key, self._schema[key]["default"]),
            }
            for key in self._schema
            if key in keys
        }

    def save_settings(self, module_id: str, values: dict) -> dict:
        if self._tasks:
            raise ValueError("安装过程中请等待任务结束再修改设置。")
        if module_id != "common" and not self._assets_present(module_id):
            raise ValueError("请先安装该模块，再调整设置。")
        fields = self.settings(module_id)
        if not isinstance(values, dict) or not values.keys() <= fields.keys():
            raise ValueError("包含不支持的设置项。")
        for key, value in values.items():
            kind = fields[key]["type"]
            valid = (
                (kind == "bool" and type(value) is bool)
                or (kind == "int" and type(value) is int)
                or (
                    kind == "float"
                    and type(value) in (int, float)
                    and math.isfinite(value)
                )
                or (kind == "string" and isinstance(value, str))
            )
            if not valid:
                raise ValueError(f"{fields[key]['description']} 的值类型不正确。")
            if key == "fallback_policy" and value not in {"adaptive", "all"}:
                raise ValueError("补强策略必须是 adaptive 或 all。")
            if key == "auto_animation_probability" and not 0 <= value <= 100:
                raise ValueError("自动动图回复概率必须在 0–100% 之间。")
        previous = dict(self.config)
        try:
            self.config.update(values)
            if module_id != "common":
                spec = self._spec(module_id)
                path_key = f"{spec['prefix']}model_path"
                if path_key in values and values[path_key] != previous.get(
                    path_key, ""
                ):
                    # A different file must pass validation before it is used.
                    self.config[spec["flag"]] = False
            self.on_change()
            self._save_config()
        except Exception:
            self.config.clear()
            self.config.update(previous)
            self.on_change()
            raise
        if (
            module_id != "common"
            and path_key in values
            and values[path_key] != previous.get(path_key, "")
        ):
            self._update(
                module_id,
                phase="idle",
                message="模型路径已变更，请重新启用以加载验证。",
            )
        return self.settings(module_id)

    async def close(self):
        self._stopped.set()
        # Let an in-process pip operation finish before unloading its owner.
        # Downloads observe the stop flag between chunks; no orphan workers.
        if self._tasks:
            await asyncio.gather(*list(self._tasks.values()), return_exceptions=True)

    def _check_running(self):
        if self._stopped.is_set():
            raise InterruptedError()

    @staticmethod
    def _spec(module_id):
        if module_id not in MODULES:
            raise ValueError("未知模块，请选择通用、前景、动漫或 GIF。")
        return MODULES[module_id]

    def _log(self, module_id, message):
        logs = self._records.setdefault(module_id, {}).setdefault("logs", [])
        logs.extend(line[:1500] for line in str(message).splitlines() if line.strip())
        del logs[:-80]
        self._update(module_id)

    def _update(self, module_id, **changes):
        record = self._records.setdefault(module_id, {})
        changed_phase = "phase" in changes and record.get("phase") != changes["phase"]
        record.update(changes, updated_at_ms=time.time_ns() // 1_000_000)
        if changed_phase or time.monotonic() - self._last_save >= 1:
            self._save()

    def _save_config(self):
        save = getattr(self.config, "save_config", None)
        if callable(save):
            save()

    def _save(self):
        staging = self._state_path.with_suffix(".json.tmp")
        staging.write_text(
            json.dumps({"version": 1, "modules": self._records}, ensure_ascii=False),
            encoding="utf-8",
        )
        staging.replace(self._state_path)
        self._last_save = time.monotonic()
