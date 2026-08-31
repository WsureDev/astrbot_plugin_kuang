from __future__ import annotations

import datetime as _datetime
import shutil
from collections import deque
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from .logger import get_logger
from .models import DetectionBox

_logger = get_logger(__name__)

# rembg 项目已经把 U²-Net-p 转成通用 ONNX，用同一套 auto-download 流程即可。
DEFAULT_SALIENCY_MODEL_URL = (
    "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class SaliencyDetectionBackend:
    """基于 U²-Net-p 的显著性检测后端。

    产出 `semantic_role="salient"` 的候选框，代表"不认识但明显是前景"的主体。
    融合规则在 arranger 里统一处理，本 backend 只负责把热图切成 bbox。
    """

    source_name = "saliency"
    default_model_url = DEFAULT_SALIENCY_MODEL_URL

    def __init__(
        self,
        *,
        model_path: str,
        model_url: str = "",
        auto_download_model: bool = True,
        model_input_size: int = 320,
        binary_threshold: float = 0.5,
        min_area_ratio: float = 0.005,
        max_boxes: int = 8,
        edge_padding_ratio: float = 0.01,
    ) -> None:
        resolved_model_path = str(model_path or "").strip()
        if not resolved_model_path:
            raise ValueError(
                "saliency model_path 不能为空，请指定 U²-Net-p ONNX 模型文件路径。"
            )
        self.model_path = Path(resolved_model_path)
        resolved_model_url = str(model_url or "").strip()
        self.model_url = resolved_model_url or self.default_model_url
        self.auto_download_model = bool(auto_download_model)
        self.model_input_size = max(64, int(model_input_size))
        self.binary_threshold = max(0.0, min(1.0, float(binary_threshold)))
        self.min_area_ratio = max(0.0, float(min_area_ratio))
        self.max_boxes = max(1, int(max_boxes))
        self.edge_padding_ratio = max(0.0, float(edge_padding_ratio))
        self._np: Any | None = None
        self._ort: Any | None = None
        self._session: Any | None = None
        self._input_name: str | None = None
        self._input_size: tuple[int, int] = (
            self.model_input_size,
            self.model_input_size,
        )
        _logger.debug(
            "[%s] backend init: model_path=%s auto_download=%s model_url=%s"
            " threshold=%.3f min_area_ratio=%.4f max_boxes=%s input_size=%s",
            self.source_name,
            self.model_path,
            self.auto_download_model,
            self.model_url or "<empty>",
            self.binary_threshold,
            self.min_area_ratio,
            self.max_boxes,
            self._input_size,
        )

    # -- public API -------------------------------------------------------

    def detect(self, image_rgb) -> list[DetectionBox]:
        image_height, image_width = image_rgb.shape[:2]
        raw = self._infer(image_rgb)
        mask_probability = self._normalize_mask(raw)
        binary_mask = mask_probability >= self.binary_threshold
        components = self._extract_connected_components(binary_mask)
        _logger.debug(
            "[%s] mask stats: image=%sx%s mask_shape=%s max_prob=%.3f mean_prob=%.3f components=%s",
            self.source_name,
            image_width,
            image_height,
            binary_mask.shape,
            float(mask_probability.max()) if mask_probability.size else 0.0,
            float(mask_probability.mean()) if mask_probability.size else 0.0,
            len(components),
        )

        mask_height, mask_width = binary_mask.shape
        scale_x = image_width / max(1, mask_width)
        scale_y = image_height / max(1, mask_height)
        min_area_px = max(1, int(mask_height * mask_width * self.min_area_ratio))
        pad_x = int(round(image_width * self.edge_padding_ratio))
        pad_y = int(round(image_height * self.edge_padding_ratio))

        boxes: list[DetectionBox] = []
        for component in components:
            min_x, min_y, max_x, max_y, area = component
            if area < min_area_px:
                continue
            x1 = int(round(min_x * scale_x)) - pad_x
            y1 = int(round(min_y * scale_y)) - pad_y
            x2 = int(round((max_x + 1) * scale_x)) + pad_x
            y2 = int(round((max_y + 1) * scale_y)) + pad_y
            x1 = max(0, min(image_width, x1))
            y1 = max(0, min(image_height, y1))
            x2 = max(0, min(image_width, x2))
            y2 = max(0, min(image_height, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            score = float(area / max(1, mask_height * mask_width))
            boxes.append(
                DetectionBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    score=score,
                    category="salient",
                    priority=0,
                    source=self.source_name,
                    stage="stage3_salient",
                    part="salient",
                    semantic_role="salient",
                )
            )

        # 面积大者优先，只保留前 max_boxes 个
        boxes.sort(key=lambda item: -item.area)
        limited = boxes[: self.max_boxes]
        _logger.debug(
            "[%s] detect: raw_components=%s kept=%s dropped=%s",
            self.source_name,
            len(components),
            len(limited),
            len(boxes) - len(limited),
        )
        return limited

    def load_numpy(self):
        if self._np is None:
            try:
                import numpy as np
            except ImportError as exc:
                raise RuntimeError("缺少 NumPy 依赖，请先安装 requirements.txt。") from exc
            self._np = np
        return self._np

    # -- inference --------------------------------------------------------

    def _infer(self, image_rgb):
        np = self.load_numpy()
        session = self._get_session()
        tensor = self._prepare_input(image_rgb)
        _logger.debug(
            "[%s] infer: tensor_shape=%s input_name=%s",
            self.source_name,
            tuple(tensor.shape),
            self._input_name,
        )
        outputs = session.run(None, {self._input_name: tensor})
        if not outputs:
            raise RuntimeError(f"{self.source_name} ONNX 推理未返回任何输出。")
        raw = np.asarray(outputs[0])
        _logger.debug(
            "[%s] infer done: outputs=%s primary_shape=%s dtype=%s",
            self.source_name,
            len(outputs),
            tuple(raw.shape),
            raw.dtype,
        )
        return raw

    def _prepare_input(self, image_rgb):
        np = self.load_numpy()
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("缺少 Pillow 依赖，请先安装 requirements.txt。") from exc

        input_width, input_height = self._input_size
        resized = Image.fromarray(image_rgb).resize(
            (input_width, input_height),
            Image.BILINEAR,
        )
        tensor = np.asarray(resized).astype(np.float32) / 255.0
        mean = np.asarray(_IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(_IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
        tensor = (tensor - mean) / std
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        return tensor.astype(np.float32)

    def _normalize_mask(self, raw):
        np = self.load_numpy()
        mask = raw
        # 常见形状是 (1, 1, H, W)、(1, H, W) 或 (H, W)
        while mask.ndim > 2:
            mask = mask[0]
        # 部分导出的模型输出未经 sigmoid，取值范围可能超出 [0,1]
        if mask.max() > 1.0 or mask.min() < 0.0:
            mask = 1.0 / (1.0 + np.exp(-mask))
        mask_min = float(mask.min())
        mask_max = float(mask.max())
        if mask_max - mask_min > 1e-6:
            mask = (mask - mask_min) / (mask_max - mask_min)
        return mask

    def _extract_connected_components(
        self,
        binary_mask,
    ) -> list[tuple[int, int, int, int, int]]:
        """4-连通域标记，返回 (min_x, min_y, max_x, max_y, area) 列表。

        320×320 的显著性 mask 前景像素通常在 5%-30% 之间，纯 Python BFS 可控。
        为避免退化到"整图都是前景"的极端情况，加了一个 fg_ratio 上限守卫。
        """
        np = self.load_numpy()
        mask = np.ascontiguousarray(binary_mask, dtype=bool).copy()
        height, width = mask.shape
        total = height * width
        fg_count = int(mask.sum())
        if fg_count == 0:
            return []
        # 前景占比 > 92% 时直接把整块当成一个大目标返回，节省 BFS 成本
        if fg_count / max(1, total) > 0.92:
            return [(0, 0, width - 1, height - 1, fg_count)]

        components: list[tuple[int, int, int, int, int]] = []
        # 直接把 mask 做 list-of-list，避免 numpy 单像素读写在 Python 循环里的开销
        rows = [row.tolist() for row in mask]
        for seed_y in range(height):
            row = rows[seed_y]
            for seed_x in range(width):
                if not row[seed_x]:
                    continue
                queue: deque[tuple[int, int]] = deque()
                queue.append((seed_y, seed_x))
                rows[seed_y][seed_x] = False
                min_x = max_x = seed_x
                min_y = max_y = seed_y
                area = 0
                while queue:
                    y, x = queue.popleft()
                    area += 1
                    if x < min_x:
                        min_x = x
                    if x > max_x:
                        max_x = x
                    if y < min_y:
                        min_y = y
                    if y > max_y:
                        max_y = y
                    if y + 1 < height and rows[y + 1][x]:
                        rows[y + 1][x] = False
                        queue.append((y + 1, x))
                    if y - 1 >= 0 and rows[y - 1][x]:
                        rows[y - 1][x] = False
                        queue.append((y - 1, x))
                    if x + 1 < width and rows[y][x + 1]:
                        rows[y][x + 1] = False
                        queue.append((y, x + 1))
                    if x - 1 >= 0 and rows[y][x - 1]:
                        rows[y][x - 1] = False
                        queue.append((y, x - 1))
                components.append((min_x, min_y, max_x, max_y, area))
        return components

    # -- session / model download (对齐 yolo_backend 的行为) ---------------

    def _load_onnxruntime(self):
        if self._ort is None:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError(
                    "缺少 onnxruntime 依赖，请先安装 requirements.txt。"
                ) from exc
            self._ort = ort
        return self._ort

    def _get_session(self):
        if self._session is not None:
            return self._session

        ort = self._load_onnxruntime()
        model_path = self._ensure_model_file()
        session = self._create_session_with_recovery(ort, model_path)

        input_meta = session.get_inputs()
        if not input_meta:
            raise RuntimeError(f"{self.source_name} ONNX 模型未暴露输入张量。")
        self._session = session
        self._input_name = input_meta[0].name
        self._input_size = self._resolve_input_size(input_meta[0].shape)
        _logger.info(
            "[%s] model session ready: path=%s input_name=%s input_size=%s",
            self.source_name,
            model_path,
            self._input_name,
            self._input_size,
        )
        return self._session

    def _ensure_model_file(self) -> Path:
        if self.model_path.exists():
            if not self.model_path.is_file():
                raise RuntimeError(
                    f"{self.source_name} 模型路径不是文件: {self.model_path}"
                )
            _logger.debug(
                "[%s] using local model file: %s size=%s",
                self.source_name,
                self.model_path,
                self._safe_file_size(self.model_path),
            )
            return self.model_path

        if not self.auto_download_model:
            raise RuntimeError(
                f"未找到 {self.source_name} 模型文件: {self.model_path}. "
                "请在配置中指定 saliency_model_path，或开启 saliency_auto_download_model。"
            )
        if not self.model_url:
            raise RuntimeError(
                f"未配置 {self.source_name} 模型下载地址 saliency_model_url。"
            )

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.model_path.with_suffix(f"{self.model_path.suffix}.download")
        _logger.info(
            "[%s] model file missing, downloading from %s -> %s",
            self.source_name,
            self.model_url,
            self.model_path,
        )
        try:
            with urlopen(self.model_url, timeout=60) as response, temp_path.open(
                "wb"
            ) as file_obj:
                shutil.copyfileobj(response, file_obj)
            temp_path.replace(self.model_path)
            _logger.info(
                "[%s] model download completed: %s",
                self.source_name,
                self.model_path,
            )
        except Exception as exc:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            _logger.error(
                "[%s] model download failed: %s",
                self.source_name,
                exc,
            )
            raise RuntimeError(
                f"{self.source_name} 模型下载失败，请检查网络或手动放置模型文件: {exc}"
            ) from exc
        return self.model_path

    def _create_session_with_recovery(self, ort, model_path: Path):
        try:
            return ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            file_size = self._safe_file_size(model_path)
            _logger.warning(
                "[%s] model session creation failed: path=%s size=%s error=%s",
                self.source_name,
                model_path,
                file_size,
                exc,
            )
            if not self._should_retry_model_load(exc):
                raise RuntimeError(
                    f"{self.source_name} ONNX 模型加载失败: {exc}"
                ) from exc
            backup_path = self._backup_invalid_model_file(model_path)
            _logger.warning(
                "[%s] detected corrupted ONNX model, removed local file and retrying download once: backup=%s",
                self.source_name,
                backup_path or "<backup_failed>",
            )
            refreshed_model_path = self._ensure_model_file()
            try:
                return ort.InferenceSession(
                    str(refreshed_model_path),
                    providers=["CPUExecutionProvider"],
                )
            except Exception as retry_exc:
                retry_size = self._safe_file_size(refreshed_model_path)
                raise RuntimeError(
                    f"{self.source_name} ONNX 模型加载失败，已重下 1 次仍不可用: "
                    f"path={refreshed_model_path} size={retry_size} error={retry_exc}"
                ) from retry_exc

    def _should_retry_model_load(self, exc: Exception) -> bool:
        if not self.auto_download_model or not self.model_url:
            return False
        if not self.model_path.exists():
            return False
        text = str(exc).lower()
        return any(
            marker in text
            for marker in (
                "invalid_protobuf",
                "protobuf parsing failed",
                "modelproto",
                "load model",
            )
        )

    def _backup_invalid_model_file(self, model_path: Path) -> str:
        try:
            if model_path.exists():
                timestamp = _datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
                backup_path = model_path.with_suffix(
                    f"{model_path.suffix}.invalid.{timestamp}"
                )
                model_path.replace(backup_path)
                return str(backup_path)
        except Exception as exc:
            _logger.warning(
                "[%s] failed to back up invalid model file before retry: path=%s error=%s",
                self.source_name,
                model_path,
                exc,
            )
        return ""

    @staticmethod
    def _safe_file_size(model_path: Path) -> int:
        try:
            return model_path.stat().st_size
        except OSError:
            return -1

    def _resolve_input_size(
        self,
        shape: list[Any] | tuple[Any, ...],
    ) -> tuple[int, int]:
        height = self.model_input_size
        width = self.model_input_size
        if len(shape) >= 4:
            if isinstance(shape[2], int) and shape[2] > 0:
                height = int(shape[2])
            if isinstance(shape[3], int) and shape[3] > 0:
                width = int(shape[3])
        return width, height
