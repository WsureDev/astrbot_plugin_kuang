from __future__ import annotations

import os
import random
from typing import Any

from .models import DetectionBox


class EspBoxDetector:
    def __init__(
        self,
        *,
        box_count: int = 5,
        random_width_ratio_min: float = 0.28,
        random_width_ratio_max: float = 0.52,
        random_height_ratio_min: float = 0.22,
        random_height_ratio_max: float = 0.55,
        nms_iou_threshold: float = 0.45,
    ) -> None:
        self.box_count = max(1, int(box_count))
        self.random_width_ratio_min = float(random_width_ratio_min)
        self.random_width_ratio_max = float(random_width_ratio_max)
        self.random_height_ratio_min = float(random_height_ratio_min)
        self.random_height_ratio_max = float(random_height_ratio_max)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self._cv2: Any | None = None
        self._np: Any | None = None
        self._hog: Any | None = None
        self._hog_unavailable = False
        self._cascade_cache: dict[str, Any] = {}

    def load_image(self, image_path: str):
        cv2, np = self._load_cv_stack()
        file_buffer = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(file_buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV 无法读取图片: {image_path}")
        return image

    def detect_from_path(self, image_path: str) -> list[DetectionBox]:
        image = self.load_image(image_path)
        return self.detect(image)

    def detect(self, image_bgr) -> list[DetectionBox]:
        cv2, _ = self._load_cv_stack()
        height, width = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        candidates: list[DetectionBox] = []
        candidates.extend(self._detect_hog_people(image_bgr))
        candidates.extend(
            self._detect_cascade(
                gray,
                "haarcascade_fullbody.xml",
                category="person_fullbody",
                priority=0,
                scale_factor=1.03,
                min_neighbors=4,
                min_size=(48, 96),
                expand_mode="body",
            )
        )
        candidates.extend(
            self._detect_cascade(
                gray,
                "haarcascade_upperbody.xml",
                category="person_upperbody",
                priority=1,
                scale_factor=1.03,
                min_neighbors=4,
                min_size=(40, 40),
                expand_mode="upper",
            )
        )
        candidates.extend(
            self._detect_cascade(
                gray,
                "haarcascade_frontalface_default.xml",
                category="head",
                priority=2,
                scale_factor=1.08,
                min_neighbors=5,
                min_size=(28, 28),
                expand_mode="head",
            )
        )
        candidates.extend(
            self._detect_cascade(
                gray,
                "haarcascade_profileface.xml",
                category="head_profile",
                priority=2,
                scale_factor=1.08,
                min_neighbors=5,
                min_size=(28, 28),
                expand_mode="head",
            )
        )
        candidates.extend(
            self._detect_cascade(
                gray,
                "haarcascade_frontalcatface.xml",
                category="animal_head",
                priority=2,
                scale_factor=1.08,
                min_neighbors=4,
                min_size=(24, 24),
                expand_mode="head",
            )
        )

        recognized = self._select_recognized_boxes(candidates)
        if len(recognized) < self.box_count:
            recognized.extend(
                self._generate_random_boxes(
                    width,
                    height,
                    self.box_count - len(recognized),
                    recognized,
                )
            )
        return recognized[: self.box_count]

    def _load_cv_stack(self):
        if self._cv2 is None or self._np is None:
            try:
                import cv2
                import numpy as np
            except ImportError as exc:
                raise RuntimeError(
                    "缺少 OpenCV / NumPy 依赖，请先安装 requirements.txt。"
                ) from exc
            self._cv2 = cv2
            self._np = np
        return self._cv2, self._np

    def _get_hog(self):
        cv2, _ = self._load_cv_stack()
        if self._hog is None and not self._hog_unavailable:
            hog_ctor = getattr(cv2, "HOGDescriptor", None)
            detector_factory = getattr(
                cv2,
                "HOGDescriptor_getDefaultPeopleDetector",
                None,
            )
            if not callable(hog_ctor) or not callable(detector_factory):
                self._hog_unavailable = True
                return None
            try:
                hog = hog_ctor()
                hog.setSVMDetector(detector_factory())
            except Exception:
                self._hog_unavailable = True
                return None
            self._hog = hog
        return self._hog

    def _get_cascade(self, file_name: str):
        cv2, _ = self._load_cv_stack()
        if file_name in self._cascade_cache:
            return self._cascade_cache[file_name]

        cascade_path = os.path.join(cv2.data.haarcascades, file_name)
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            raise RuntimeError(f"无法加载 Haar Cascade 模型: {file_name}")
        self._cascade_cache[file_name] = cascade
        return cascade

    def _detect_hog_people(self, image_bgr) -> list[DetectionBox]:
        hog = self._get_hog()
        if hog is None:
            return []
        try:
            rects, weights = hog.detectMultiScale(
                image_bgr,
                winStride=(4, 4),
                padding=(8, 8),
                scale=1.05,
            )
        except Exception:
            self._hog = None
            self._hog_unavailable = True
            return []

        detections: list[DetectionBox] = []
        for (x, y, w, h), weight in zip(rects, weights):
            detections.append(
                self._normalize_box(
                    x,
                    y,
                    w,
                    h,
                    image_bgr.shape[1],
                    image_bgr.shape[0],
                    score=float(weight),
                    category="person_fullbody",
                    priority=0,
                    source="hog",
                    expand_mode="body",
                )
            )
        return detections

    def _detect_cascade(
        self,
        gray_image,
        cascade_name: str,
        *,
        category: str,
        priority: int,
        scale_factor: float,
        min_neighbors: int,
        min_size: tuple[int, int],
        expand_mode: str,
    ) -> list[DetectionBox]:
        try:
            cascade = self._get_cascade(cascade_name)
        except RuntimeError:
            return []
        rects = cascade.detectMultiScale(
            gray_image,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
        )
        image_height, image_width = gray_image.shape[:2]
        detections: list[DetectionBox] = []
        for x, y, w, h in rects:
            detections.append(
                self._normalize_box(
                    x,
                    y,
                    w,
                    h,
                    image_width,
                    image_height,
                    score=1.0,
                    category=category,
                    priority=priority,
                    source=cascade_name,
                    expand_mode=expand_mode,
                )
            )
        return detections

    def _normalize_box(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        image_width: int,
        image_height: int,
        *,
        score: float,
        category: str,
        priority: int,
        source: str,
        expand_mode: str,
    ) -> DetectionBox:
        x1 = float(x)
        y1 = float(y)
        x2 = float(x + w)
        y2 = float(y + h)

        if expand_mode == "body":
            padding_x = w * 0.08
            padding_y = h * 0.05
            x1 -= padding_x
            x2 += padding_x
            y1 -= padding_y
            y2 += padding_y
        elif expand_mode == "upper":
            padding_x = w * 0.12
            top_padding = h * 0.10
            bottom_padding = h * 0.45
            x1 -= padding_x
            x2 += padding_x
            y1 -= top_padding
            y2 += bottom_padding
        elif expand_mode == "head":
            center_x = x + (w / 2.0)
            new_w = w * 1.45
            new_h = h * 1.90
            x1 = center_x - (new_w / 2.0)
            x2 = center_x + (new_w / 2.0)
            y1 = y - (h * 0.28)
            y2 = y1 + new_h

        x1 = max(0, int(round(x1)))
        y1 = max(0, int(round(y1)))
        x2 = min(image_width, int(round(x2)))
        y2 = min(image_height, int(round(y2)))

        return DetectionBox(
            x1=x1,
            y1=y1,
            x2=max(x1 + 1, x2),
            y2=max(y1 + 1, y2),
            score=score,
            category=category,
            priority=priority,
            source=source,
        )

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
    ) -> list[DetectionBox]:
        rng = random.Random((image_width << 16) ^ image_height ^ missing_count)
        generated: list[DetectionBox] = []

        max_attempts = max(40, missing_count * 40)
        for _ in range(max_attempts):
            if len(generated) >= missing_count:
                break

            height_ratio = rng.uniform(
                self.random_height_ratio_min,
                self.random_height_ratio_max,
            )
            box_height = max(24, int(image_height * height_ratio))
            box_width = max(
                16,
                int(
                    box_height
                    * rng.uniform(
                        self.random_width_ratio_min,
                        self.random_width_ratio_max,
                    )
                ),
            )

            if box_width >= image_width or box_height >= image_height:
                continue

            x1 = rng.randint(0, max(0, image_width - box_width))
            y1 = rng.randint(0, max(0, image_height - box_height))
            candidate = DetectionBox(
                x1=x1,
                y1=y1,
                x2=x1 + box_width,
                y2=y1 + box_height,
                score=0.0,
                category="random",
                priority=99,
                source="random",
            )

            all_boxes = existing_boxes + generated
            if any(self._iou(candidate, other) >= 0.35 for other in all_boxes):
                continue

            generated.append(candidate)

        if len(generated) < missing_count:
            generated.extend(
                self._generate_grid_fallback_boxes(
                    image_width,
                    image_height,
                    missing_count - len(generated),
                    existing_boxes + generated,
                )
            )
        return generated

    def _generate_grid_fallback_boxes(
        self,
        image_width: int,
        image_height: int,
        missing_count: int,
        existing_boxes: list[DetectionBox],
    ) -> list[DetectionBox]:
        fallback: list[DetectionBox] = []
        columns = 3
        rows = 2
        cell_width = max(1, image_width // columns)
        cell_height = max(1, image_height // rows)

        for row in range(rows):
            for column in range(columns):
                if len(fallback) >= missing_count:
                    return fallback

                box_height = max(24, int(cell_height * 0.72))
                box_width = max(16, int(box_height * 0.38))
                center_x = int((column + 0.5) * cell_width)
                x1 = max(0, center_x - (box_width // 2))
                y1 = max(0, row * cell_height + int(cell_height * 0.15))
                candidate = DetectionBox(
                    x1=x1,
                    y1=y1,
                    x2=min(image_width, x1 + box_width),
                    y2=min(image_height, y1 + box_height),
                    score=0.0,
                    category="random",
                    priority=99,
                    source="grid_fallback",
                )

                if any(self._iou(candidate, other) >= 0.35 for other in existing_boxes):
                    continue
                fallback.append(candidate)

        if len(fallback) < missing_count:
            for row in range(rows):
                for column in range(columns):
                    if len(fallback) >= missing_count:
                        return fallback
                    box_height = max(24, int(cell_height * 0.72))
                    box_width = max(16, int(box_height * 0.38))
                    center_x = int((column + 0.5) * cell_width)
                    x1 = max(0, center_x - (box_width // 2))
                    y1 = max(0, row * cell_height + int(cell_height * 0.15))
                    candidate = DetectionBox(
                        x1=x1,
                        y1=y1,
                        x2=min(image_width, x1 + box_width),
                        y2=min(image_height, y1 + box_height),
                        score=0.0,
                        category="random",
                        priority=99,
                        source="grid_forced",
                    )
                    if any(candidate.as_tuple() == item.as_tuple() for item in fallback):
                        continue
                    fallback.append(candidate)
        return fallback

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
