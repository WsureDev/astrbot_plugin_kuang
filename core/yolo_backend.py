from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from .logger import get_logger
from .models import DetectionBox

_logger = get_logger(__name__)

DEFAULT_YOLO26N_MODEL_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.onnx"
)
DEFAULT_ANIME_YOLO_MODEL_URL = (
    "https://huggingface.co/laowanglaowang/yolov11m_anime_Image_segmentation/resolve/main/best.onnx"
)

_COCO_CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

_ANIMAL_CLASSES = {
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
}


class BaseYoloOnnxDetectionBackend:
    source_name = "yolo_onnx"
    default_model_url = ""
    class_names: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        model_path: str,
        model_url: str = "",
        auto_download_model: bool = True,
        confidence_threshold: float = 0.25,
        nms_iou_threshold: float = 0.45,
        model_input_size: int = 640,
    ) -> None:
        self.model_path = Path(model_path)
        resolved_model_url = str(model_url or "").strip()
        self.model_url = resolved_model_url or self.default_model_url
        self.auto_download_model = bool(auto_download_model)
        self.confidence_threshold = max(0.0, float(confidence_threshold))
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.model_input_size = max(64, int(model_input_size))
        self._np: Any | None = None
        self._ort: Any | None = None
        self._session: Any | None = None
        self._input_name: str | None = None
        self._input_size: tuple[int, int] = (
            self.model_input_size,
            self.model_input_size,
        )
        _logger.debug(
            "[%s] backend init: model_path=%s auto_download=%s model_url=%s conf=%.3f iou=%.3f input=%s",
            self.source_name,
            self.model_path,
            self.auto_download_model,
            self.model_url or "<empty>",
            self.confidence_threshold,
            self.nms_iou_threshold,
            self._input_size,
        )

    def detect(self, image_rgb) -> list[DetectionBox]:
        image_height, image_width = image_rgb.shape[:2]
        raw_output, scale, pad_x, pad_y = self._infer(image_rgb)
        detections = self._decode_predictions(
            raw_output,
            image_width=image_width,
            image_height=image_height,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
        )
        selected = self._select_recognized_boxes(detections)
        _logger.debug(
            "[%s] detect: image=%sx%s raw_candidates=%s selected=%s",
            self.source_name,
            image_width,
            image_height,
            len(detections),
            len(selected),
        )
        _logger.debug(
            "[%s] detect details: %s",
            self.source_name,
            self._summarize_detections(selected),
        )
        return selected

    def load_numpy(self):
        if self._np is None:
            try:
                import numpy as np
            except ImportError as exc:
                raise RuntimeError("缺少 NumPy 依赖，请先安装 requirements.txt。") from exc
            self._np = np
        return self._np

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

    def _ensure_model_file(self) -> Path:
        if self.model_path.exists():
            _logger.debug(
                "[%s] using local model file: %s",
                self.source_name,
                self.model_path,
            )
            return self.model_path

        if not self.auto_download_model:
            _logger.debug(
                "[%s] model file missing and auto download disabled: %s",
                self.source_name,
                self.model_path,
            )
            raise RuntimeError(
                f"未找到 {self.source_name} 模型文件: {self.model_path}. "
                "请在配置中指定 model_path，或开启 auto_download_model。"
            )
        if not self.model_url:
            _logger.debug(
                "[%s] model file missing and no download url configured",
                self.source_name,
            )
            raise RuntimeError(f"未配置 {self.source_name} 模型下载地址 model_url。")

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

    def _get_session(self):
        if self._session is not None:
            return self._session

        ort = self._load_onnxruntime()
        model_path = self._ensure_model_file()
        try:
            session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise RuntimeError(f"{self.source_name} ONNX 模型加载失败: {exc}") from exc

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

    def _resolve_input_size(self, shape: list[Any] | tuple[Any, ...]) -> tuple[int, int]:
        height = self.model_input_size
        width = self.model_input_size
        if len(shape) >= 4:
            if isinstance(shape[2], int) and shape[2] > 0:
                height = int(shape[2])
            if isinstance(shape[3], int) and shape[3] > 0:
                width = int(shape[3])
        return width, height

    def _infer(self, image_rgb):
        np = self.load_numpy()
        session = self._get_session()
        tensor, scale, pad_x, pad_y = self._prepare_input(image_rgb)
        _logger.debug(
            "[%s] infer: tensor_shape=%s scale=%.4f pad_x=%.1f pad_y=%.1f",
            self.source_name,
            tuple(tensor.shape),
            scale,
            pad_x,
            pad_y,
        )
        outputs = session.run(None, {self._input_name: tensor})
        if not outputs:
            raise RuntimeError(f"{self.source_name} ONNX 推理未返回任何输出。")
        _logger.debug(
            "[%s] infer outputs=%s first_shape=%s",
            self.source_name,
            len(outputs),
            tuple(getattr(outputs[0], "shape", ()) or ()),
        )
        return np.asarray(outputs[0]), scale, pad_x, pad_y

    def _prepare_input(self, image_rgb):
        np = self.load_numpy()
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("缺少 Pillow 依赖，请先安装 requirements.txt。") from exc

        image_height, image_width = image_rgb.shape[:2]
        input_width, input_height = self._input_size
        scale = min(input_width / image_width, input_height / image_height)
        resized_width = max(1, int(round(image_width * scale)))
        resized_height = max(1, int(round(image_height * scale)))
        pad_x = (input_width - resized_width) // 2
        pad_y = (input_height - resized_height) // 2

        resized = Image.fromarray(image_rgb).resize(
            (resized_width, resized_height),
            Image.BILINEAR,
        )
        canvas = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
        canvas[
            pad_y : pad_y + resized_height,
            pad_x : pad_x + resized_width,
        ] = np.asarray(resized)

        tensor = canvas.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        return tensor, scale, float(pad_x), float(pad_y)

    def _decode_predictions(
        self,
        raw_output,
        *,
        image_width: int,
        image_height: int,
        scale: float,
        pad_x: float,
        pad_y: float,
    ) -> list[DetectionBox]:
        np = self.load_numpy()
        predictions = np.asarray(raw_output)

        if predictions.ndim == 3 and predictions.shape[0] == 1:
            predictions = predictions[0]
        if predictions.ndim != 2:
            raise RuntimeError(
                f"{self.source_name} 输出形状不受支持: {tuple(predictions.shape)}"
            )
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        detections: list[DetectionBox] = []
        for row in predictions:
            row = np.asarray(row).astype(np.float32).reshape(-1)
            if row.size < 6:
                continue

            if row.size == 7 and abs(row[0]) < 1e-6:
                row = row[1:]

            if row.size == 6:
                x1, y1, x2, y2, score, class_id = row.tolist()
                class_id = int(class_id)
            else:
                class_scores = row[4:]
                class_id = int(np.argmax(class_scores))
                score = float(class_scores[class_id])
                if class_id < 0 or class_id >= len(class_scores):
                    continue
                center_x, center_y, box_width, box_height = row[:4].tolist()
                x1 = center_x - (box_width / 2.0)
                y1 = center_y - (box_height / 2.0)
                x2 = center_x + (box_width / 2.0)
                y2 = center_y + (box_height / 2.0)

            if score < self.confidence_threshold:
                continue

            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale

            x1 = max(0, min(image_width, int(round(x1))))
            y1 = max(0, min(image_height, int(round(y1))))
            x2 = max(0, min(image_width, int(round(x2))))
            y2 = max(0, min(image_height, int(round(y2))))
            if x2 <= x1 or y2 <= y1:
                continue

            label = self._label_for_class_id(class_id)
            detections.append(
                DetectionBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    score=float(score),
                    category=label,
                    priority=self._priority_for_label(label),
                    source=self.source_name,
                )
            )

        return detections

    def _label_for_class_id(self, class_id: int) -> str:
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"class_{class_id}"

    @staticmethod
    def _priority_for_label(label: str) -> int:
        del label
        return 0

    def _select_recognized_boxes(
        self, candidates: list[DetectionBox]
    ) -> list[DetectionBox]:
        ordered = sorted(
            candidates,
            key=lambda item: (item.priority, -item.score, -item.area),
        )

        selected: list[DetectionBox] = []
        for candidate in ordered:
            if candidate.area <= 0:
                continue
            if any(
                self._iou(candidate, other) >= self.nms_iou_threshold
                for other in selected
            ):
                continue
            selected.append(candidate)
        return selected

    @staticmethod
    def _summarize_detections(detections: list[DetectionBox]) -> str:
        if not detections:
            return "<none>"
        return ", ".join(
            (
                f"{item.category}"
                f"@{item.score:.2f}"
                f"[{item.x1},{item.y1},{item.x2},{item.y2}]"
            )
            for item in detections
        )

    @staticmethod
    def _iou(left: DetectionBox, right: DetectionBox) -> float:
        inter_x1 = max(left.x1, right.x1)
        inter_y1 = max(left.y1, right.y1)
        inter_x2 = min(left.x2, right.x2)
        inter_y2 = min(left.y2, right.y2)

        inter_width = max(0, inter_x2 - inter_x1)
        inter_height = max(0, inter_y2 - inter_y1)
        if inter_width == 0 or inter_height == 0:
            return 0.0

        inter_area = inter_width * inter_height
        union_area = left.area + right.area - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area


class Yolo26nDetectionBackend(BaseYoloOnnxDetectionBackend):
    source_name = "yolo26n"
    default_model_url = DEFAULT_YOLO26N_MODEL_URL
    class_names = tuple(_COCO_CLASS_NAMES)

    @staticmethod
    def _priority_for_label(label: str) -> int:
        if label == "person":
            return 0
        if label in _ANIMAL_CLASSES:
            return 1
        return 5


class AnimeYoloDetectionBackend(BaseYoloOnnxDetectionBackend):
    source_name = "anime_yolo"
    default_model_url = DEFAULT_ANIME_YOLO_MODEL_URL
    class_names = ("anime_face",)
