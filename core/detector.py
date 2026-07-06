from __future__ import annotations

from typing import Any

from .detection_backend import DetectionBackend
from .models import DetectionBox
from .random_layout import PerspectiveRandomBoxGenerator
from .yolo_backend import DEFAULT_YOLO26N_MODEL_URL, Yolo26nDetectionBackend

_DETECTION_BACKENDS = {
    "yolo26n": Yolo26nDetectionBackend,
}


class EspBoxDetector:
    def __init__(
        self,
        *,
        box_count: int = 5,
        model_path: str,
        model_url: str = DEFAULT_YOLO26N_MODEL_URL,
        auto_download_model: bool = True,
        confidence_threshold: float = 0.25,
        random_width_ratio_min: float = 0.28,
        random_width_ratio_max: float = 0.52,
        random_height_ratio_min: float = 0.09,
        random_height_ratio_max: float = 0.55,
        nms_iou_threshold: float = 0.45,
        model_input_size: int = 640,
        enable_random_boxes: bool = True,
        backend_name: str = "yolo26n",
        detection_backend: DetectionBackend | None = None,
    ) -> None:
        self.box_count = max(1, int(box_count))
        self.enable_random_boxes = bool(enable_random_boxes)
        self._backend = detection_backend or self._build_detection_backend(
            backend_name=backend_name,
            model_path=model_path,
            model_url=model_url,
            auto_download_model=auto_download_model,
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold,
            model_input_size=model_input_size,
        )
        self._random_box_generator = PerspectiveRandomBoxGenerator(
            box_count=self.box_count,
            random_width_ratio_min=random_width_ratio_min,
            random_width_ratio_max=random_width_ratio_max,
            random_height_ratio_min=random_height_ratio_min,
            random_height_ratio_max=random_height_ratio_max,
        )

    def load_image(self, image_path: str):
        np = self._load_numpy()
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("缺少 Pillow 依赖，请先安装 requirements.txt。") from exc

        with Image.open(image_path) as image:
            return np.array(image.convert("RGB"))

    def detect_from_path(self, image_path: str) -> list[DetectionBox]:
        image = self.load_image(image_path)
        return self.detect(image)

    def detect(self, image_rgb) -> list[DetectionBox]:
        image_height, image_width = image_rgb.shape[:2]
        recognized = list(self._backend.detect(image_rgb))[: self.box_count]
        if self.enable_random_boxes and len(recognized) < self.box_count:
            recognized.extend(
                self._random_box_generator.generate_missing_boxes(
                    image_width,
                    image_height,
                    self.box_count - len(recognized),
                    recognized,
                )
            )
        return recognized[: self.box_count]

    def generate_random_layout_preview(
        self,
        image_width: int,
        image_height: int,
        *,
        box_count: int | None = None,
        seed: int | None = None,
    ) -> tuple[list[DetectionBox], dict[str, Any]]:
        return self._random_box_generator.generate_random_layout_preview(
            image_width,
            image_height,
            box_count=box_count,
            seed=seed,
        )

    def _load_numpy(self):
        load_numpy = getattr(self._backend, "load_numpy", None)
        if callable(load_numpy):
            return load_numpy()
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("缺少 NumPy 依赖，请先安装 requirements.txt。") from exc
        return np

    @staticmethod
    def _build_detection_backend(
        *,
        backend_name: str,
        model_path: str,
        model_url: str,
        auto_download_model: bool,
        confidence_threshold: float,
        nms_iou_threshold: float,
        model_input_size: int,
    ) -> DetectionBackend:
        resolved_backend_name = str(backend_name or "yolo26n").strip().lower()
        backend_cls = _DETECTION_BACKENDS.get(resolved_backend_name)
        if backend_cls is None:
            supported = ", ".join(sorted(_DETECTION_BACKENDS))
            raise RuntimeError(
                f"不支持的识别器后端: {resolved_backend_name}。当前支持: {supported}"
            )
        return backend_cls(
            model_path=model_path,
            model_url=model_url or DEFAULT_YOLO26N_MODEL_URL,
            auto_download_model=auto_download_model,
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold,
            model_input_size=model_input_size,
        )
