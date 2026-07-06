from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from .models import DetectionBox

_DEFAULT_MODEL_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.onnx"
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


class EspBoxDetector:
    def __init__(
        self,
        *,
        box_count: int = 5,
        model_path: str,
        model_url: str = _DEFAULT_MODEL_URL,
        auto_download_model: bool = True,
        confidence_threshold: float = 0.25,
        random_width_ratio_min: float = 0.28,
        random_width_ratio_max: float = 0.52,
        random_height_ratio_min: float = 0.09,
        random_height_ratio_max: float = 0.55,
        nms_iou_threshold: float = 0.45,
        model_input_size: int = 640,
    ) -> None:
        self.box_count = max(1, int(box_count))
        self.model_path = Path(model_path)
        resolved_model_url = str(model_url or "").strip()
        self.model_url = resolved_model_url or _DEFAULT_MODEL_URL
        self.auto_download_model = bool(auto_download_model)
        self.confidence_threshold = max(0.0, float(confidence_threshold))
        self.random_width_ratio_min = float(random_width_ratio_min)
        self.random_width_ratio_max = float(random_width_ratio_max)
        self.random_height_ratio_min = float(random_height_ratio_min)
        self.random_height_ratio_max = float(random_height_ratio_max)
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
        raw_output, scale, pad_x, pad_y = self._infer(image_rgb)
        candidates = self._decode_predictions(
            raw_output,
            image_width=image_width,
            image_height=image_height,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
        )

        recognized = self._select_recognized_boxes(candidates)
        if len(recognized) < self.box_count:
            recognized.extend(
                self._generate_random_boxes(
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
        target_count = self.box_count if box_count is None else max(1, int(box_count))
        rng = self._build_random_rng(image_width, image_height, target_count, seed)
        scene = self._build_perspective_scene(image_width, image_height, rng)
        boxes = self._generate_random_boxes(
            image_width,
            image_height,
            target_count,
            [],
            rng=rng,
            scene=scene,
        )
        return boxes[:target_count], scene

    def _load_numpy(self):
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
            return self.model_path

        if not self.auto_download_model:
            raise RuntimeError(
                f"未找到 YOLO26n 模型文件: {self.model_path}. "
                "请在配置中指定 model_path，或开启 auto_download_model。"
            )
        if not self.model_url:
            raise RuntimeError("未配置 YOLO26n 模型下载地址 model_url。")

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.model_path.with_suffix(f"{self.model_path.suffix}.download")
        try:
            with urlopen(self.model_url, timeout=60) as response, temp_path.open(
                "wb"
            ) as file_obj:
                shutil.copyfileobj(response, file_obj)
            temp_path.replace(self.model_path)
        except Exception as exc:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise RuntimeError(
                f"YOLO26n 模型下载失败，请检查网络或手动放置模型文件: {exc}"
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
            raise RuntimeError(f"YOLO26n ONNX 模型加载失败: {exc}") from exc

        input_meta = session.get_inputs()
        if not input_meta:
            raise RuntimeError("YOLO26n ONNX 模型未暴露输入张量。")

        self._session = session
        self._input_name = input_meta[0].name
        self._input_size = self._resolve_input_size(input_meta[0].shape)
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
        np = self._load_numpy()
        session = self._get_session()
        tensor, scale, pad_x, pad_y = self._prepare_input(image_rgb)
        outputs = session.run(None, {self._input_name: tensor})
        if not outputs:
            raise RuntimeError("YOLO26n ONNX 推理未返回任何输出。")
        return np.asarray(outputs[0]), scale, pad_x, pad_y

    def _prepare_input(self, image_rgb):
        np = self._load_numpy()
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
        np = self._load_numpy()
        predictions = np.asarray(raw_output)

        if predictions.ndim == 3 and predictions.shape[0] == 1:
            predictions = predictions[0]
        if predictions.ndim != 2:
            raise RuntimeError(
                f"YOLO26n 输出形状不受支持: {tuple(predictions.shape)}"
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
                    source="yolo26n",
                )
            )

        return detections

    @staticmethod
    def _build_random_rng(
        image_width: int,
        image_height: int,
        box_count: int,
        seed: int | None = None,
    ) -> random.Random:
        if seed is None:
            seed = (image_width << 16) ^ image_height ^ box_count
        return random.Random(seed)

    def _build_perspective_scene(
        self,
        image_width: int,
        image_height: int,
        rng: random.Random,
    ) -> dict[str, Any]:
        horizon_y = image_height / 2.0
        hfov_degrees = 92.0
        vfov_degrees = 68.0
        focal_x = (image_width / 2.0) / self._tan_deg(hfov_degrees / 2.0)
        focal_y = (image_height / 2.0) / self._tan_deg(vfov_degrees / 2.0)
        camera_height = 1.65
        platforms: list[dict[str, Any]] = []

        platform_count = 1 + rng.randint(0, 1)
        attempts = 0
        while len(platforms) < platform_count and attempts < 40:
            attempts += 1
            radius = rng.uniform(7.0, 18.0)
            angle_rad = self._deg_to_rad(rng.uniform(-70.0, 70.0))
            x = radius * self._sin(angle_rad)
            z = radius * self._cos(angle_rad)
            width = rng.uniform(1.8, 3.2)
            depth = rng.uniform(1.2, 2.2)
            height = rng.uniform(0.45, 1.10)
            if z <= (depth / 2.0) + 1.0:
                continue
            if any(
                abs(platform["x"] - x) < ((platform["width"] + width) * 0.55)
                and abs(platform["z"] - z) < ((platform["depth"] + depth) * 0.55)
                for platform in platforms
            ):
                continue

            platforms.append(
                {
                    "x": x,
                    "z": z,
                    "width": width,
                    "depth": depth,
                    "height": height,
                }
            )

        platforms.sort(key=lambda item: (item["z"], item["x"]))
        return {
            "horizon_y": horizon_y,
            "camera_height": camera_height,
            "hfov_degrees": hfov_degrees,
            "vfov_degrees": vfov_degrees,
            "focal_x": focal_x,
            "focal_y": focal_y,
            "platforms": platforms,
        }

    @staticmethod
    def _deg_to_rad(value: float) -> float:
        return value * 0.017453292519943295

    @staticmethod
    def _sin(value: float) -> float:
        import math

        return math.sin(value)

    def _project_person_height(self, image_height: int, depth: float) -> float:
        max_height = image_height * min(self.random_height_ratio_max, 0.40)
        min_ratio = max(
            0.01,
            min(self.random_height_ratio_min, 0.06),
        )
        min_height = image_height * min_ratio
        depth = max(0.0, min(1.0, depth))
        return min_height + ((max_height - min_height) * (depth**2.35))

    @staticmethod
    def _cos(value: float) -> float:
        import math

        return math.cos(value)

    @staticmethod
    def _tan_deg(value: float) -> float:
        import math

        return math.tan(math.radians(value))

    def _project_world_point(
        self,
        x: float,
        y: float,
        z: float,
        scene: dict[str, Any],
        image_width: int,
        image_height: int,
    ) -> tuple[float, float] | None:
        if z <= 0.2:
            return None
        focal_x = float(scene["focal_x"])
        focal_y = float(scene["focal_y"])
        camera_height = float(scene["camera_height"])
        screen_x = (image_width / 2.0) + ((x * focal_x) / z)
        screen_y = (image_height / 2.0) - (((y - camera_height) * focal_y) / z)
        return screen_x, screen_y

    def _make_random_candidate(
        self,
        image_width: int,
        image_height: int,
        rng: random.Random,
        scene: dict[str, Any],
    ) -> DetectionBox | None:
        platforms = scene["platforms"]
        use_platform = bool(platforms) and rng.random() < 0.24

        if use_platform:
            platform = rng.choice(platforms)
            local_x = rng.uniform(
                -(float(platform["width"]) * 0.32),
                float(platform["width"]) * 0.32,
            )
            world_x = float(platform["x"]) + local_x
            world_z = float(platform["z"]) + rng.uniform(
                -(float(platform["depth"]) * 0.08),
                float(platform["depth"]) * 0.08,
            )
            foot_world_y = float(platform["height"])
            source = "random_platform"
        else:
            radius = rng.triangular(5.0, 30.0, 12.0)
            angle_rad = self._deg_to_rad(rng.uniform(-175.0, 175.0))
            world_x = radius * self._sin(angle_rad)
            world_z = radius * self._cos(angle_rad)
            foot_world_y = 0.0
            source = "random_ground"

        if world_z <= 1.0:
            return None

        foot_point = self._project_world_point(
            world_x,
            foot_world_y,
            world_z,
            scene,
            image_width,
            image_height,
        )
        if foot_point is None:
            return None
        screen_center_x, screen_foot_y = foot_point

        top_world_y = foot_world_y + rng.uniform(1.58, 1.80)
        top_point = self._project_world_point(
            world_x,
            top_world_y,
            world_z,
            scene,
            image_width,
            image_height,
        )
        if top_point is None:
            return None

        _, screen_top_y = top_point
        box_height = int(round(screen_foot_y - screen_top_y))
        if box_height < max(10, int(image_height * 0.035)):
            return None

        box_width = int(
            round(
                box_height
                * rng.uniform(
                    self.random_width_ratio_min,
                    self.random_width_ratio_max,
                )
            )
        )

        x1 = int(round(screen_center_x - (box_width / 2.0)))
        x2 = x1 + box_width
        y2 = int(round(screen_foot_y))
        y1 = y2 - box_height

        if x2 <= 0 or x1 >= image_width:
            return None
        if y2 <= 0 or y1 >= image_height:
            return None

        x1 = max(0, x1)
        x2 = min(image_width, x2)
        y1 = max(0, y1)
        y2 = min(image_height, y2)
        if x2 - x1 < 12 or y2 - y1 < 16:
            return None

        if screen_foot_y < image_height * 0.50:
            return None
        if screen_center_x < image_width * 0.03 or screen_center_x > image_width * 0.97:
            return None

        return DetectionBox(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            score=0.0,
            category="random",
            priority=99,
            source=source,
        )

    @staticmethod
    def _label_for_class_id(class_id: int) -> str:
        if 0 <= class_id < len(_COCO_CLASS_NAMES):
            return _COCO_CLASS_NAMES[class_id]
        return f"class_{class_id}"

    @staticmethod
    def _priority_for_label(label: str) -> int:
        if label == "person":
            return 0
        if label in _ANIMAL_CLASSES:
            return 1
        return 5

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
            if len(selected) >= self.box_count:
                break
        return selected

    def _generate_random_boxes(
        self,
        image_width: int,
        image_height: int,
        missing_count: int,
        existing_boxes: list[DetectionBox],
        *,
        rng: random.Random | None = None,
        scene: dict[str, Any] | None = None,
    ) -> list[DetectionBox]:
        if rng is None:
            rng = self._build_random_rng(
                image_width,
                image_height,
                missing_count,
            )
        if scene is None:
            scene = self._build_perspective_scene(image_width, image_height, rng)

        generated: list[DetectionBox] = []
        max_attempts = max(80, missing_count * 80)
        for _ in range(max_attempts):
            if len(generated) >= missing_count:
                break

            candidate = self._make_random_candidate(
                image_width,
                image_height,
                rng,
                scene,
            )
            if candidate is None:
                continue

            all_boxes = existing_boxes + generated
            if any(self._overlaps(candidate, other) for other in all_boxes):
                continue

            generated.append(candidate)

        if len(generated) < missing_count:
            generated.extend(
                self._generate_perspective_fallback_boxes(
                    image_width,
                    image_height,
                    missing_count - len(generated),
                    existing_boxes + generated,
                    rng,
                    scene,
                )
            )
        return generated

    def _generate_perspective_fallback_boxes(
        self,
        image_width: int,
        image_height: int,
        missing_count: int,
        existing_boxes: list[DetectionBox],
        rng: random.Random,
        scene: dict[str, Any],
    ) -> list[DetectionBox]:
        fallback: list[DetectionBox] = []
        slots: list[tuple[str, Any, Any]] = []

        for platform in scene["platforms"]:
            slots.append(("platform", platform, 0.0))
            slots.append(("platform", platform, -0.26))
            slots.append(("platform", platform, 0.26))

        radius_slots = [5.0, 7.0, 9.0, 12.0, 16.0, 20.0, 25.0, 30.0]
        angle_slots = [-78.0, -60.0, -42.0, -26.0, -12.0, 0.0, 12.0, 26.0, 42.0, 60.0, 78.0]
        for radius in radius_slots:
            for angle in angle_slots:
                slots.append(("ground", radius, angle))

        rng.shuffle(slots)
        for slot_kind, slot_a, slot_b in slots:
            if len(fallback) >= missing_count:
                return fallback

            if slot_kind == "platform":
                platform = slot_a
                world_x = float(platform["x"]) + (float(slot_b) * float(platform["width"]) * 0.28)
                world_z = float(platform["z"])
                foot_world_y = float(platform["height"])
                source = "random_platform_fallback"
            else:
                radius = float(slot_a)
                angle_rad = self._deg_to_rad(float(slot_b))
                world_x = radius * self._sin(angle_rad)
                world_z = radius * self._cos(angle_rad)
                foot_world_y = 0.0
                source = "random_ground_fallback"

            if world_z <= 1.0:
                continue
            foot_point = self._project_world_point(
                world_x,
                foot_world_y,
                world_z,
                scene,
                image_width,
                image_height,
            )
            top_point = self._project_world_point(
                world_x,
                foot_world_y + 1.70,
                world_z,
                scene,
                image_width,
                image_height,
            )
            if foot_point is None or top_point is None:
                continue

            screen_center_x, screen_foot_y = foot_point
            _, screen_top_y = top_point
            box_height = int(round(screen_foot_y - screen_top_y))
            if box_height < max(10, int(image_height * 0.035)):
                continue

            box_width = int(
                round(
                    box_height
                    * ((self.random_width_ratio_min + self.random_width_ratio_max) / 2.0)
                )
            )

            x1 = int(round(screen_center_x - (box_width / 2.0)))
            x2 = x1 + box_width
            y2 = int(round(screen_foot_y))
            y1 = y2 - box_height
            if x2 <= 0 or x1 >= image_width or y2 <= 0 or y1 >= image_height:
                continue

            x1 = max(0, x1)
            x2 = min(image_width, x2)
            y1 = max(0, y1)
            y2 = min(image_height, y2)
            if x2 - x1 < 12 or y2 - y1 < 16:
                continue

            candidate = DetectionBox(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                score=0.0,
                category="random",
                priority=99,
                source=source,
            )
            all_boxes = existing_boxes + fallback
            if any(self._overlaps(candidate, other) for other in all_boxes):
                continue
            fallback.append(candidate)
        return fallback

    @staticmethod
    def _overlaps(left: DetectionBox, right: DetectionBox) -> bool:
        return (
            max(left.x1, right.x1) < min(left.x2, right.x2)
            and max(left.y1, right.y1) < min(left.y2, right.y2)
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
