from __future__ import annotations

from .detection_backend import DetectionBackend
from .logger import get_logger
from .models import DetectionBox

_logger = get_logger(__name__)


class CascadeDetectionBackend:
    def __init__(
        self,
        *,
        primary_backend: DetectionBackend,
        secondary_backend: DetectionBackend | None,
        fallback_trigger_count: int = 2,
        merge_iou_threshold: float = 0.45,
        ignore_secondary_errors: bool = True,
    ) -> None:
        self.primary_backend = primary_backend
        self.secondary_backend = secondary_backend
        self.fallback_trigger_count = max(1, int(fallback_trigger_count))
        self.merge_iou_threshold = max(0.0, float(merge_iou_threshold))
        self.ignore_secondary_errors = bool(ignore_secondary_errors)

    def detect(self, image_rgb) -> list[DetectionBox]:
        primary = list(self.primary_backend.detect(image_rgb))
        _logger.debug(
            "[cascade] primary detections=%s trigger_count=%s secondary_enabled=%s",
            len(primary),
            self.fallback_trigger_count,
            self.secondary_backend is not None,
        )
        _logger.debug(
            "[cascade] primary details: %s",
            self._summarize_detections(primary),
        )
        if self.secondary_backend is None:
            return primary
        if len(primary) >= self.fallback_trigger_count:
            _logger.debug("[cascade] skip secondary because primary detections are enough")
            return primary

        try:
            secondary = list(self.secondary_backend.detect(image_rgb))
        except Exception:
            if not self.ignore_secondary_errors:
                raise
            _logger.exception("[cascade] secondary backend failed, using primary only")
            return primary

        _logger.debug(
            "[cascade] secondary detections=%s, merging with primary=%s",
            len(secondary),
            len(primary),
        )
        _logger.debug(
            "[cascade] secondary details: %s",
            self._summarize_detections(secondary),
        )
        merged = self._merge_detections(primary, secondary)
        _logger.debug(
            "[cascade] merged detections=%s details=%s",
            len(merged),
            self._summarize_detections(merged),
        )
        return merged

    def load_numpy(self):
        for backend in (self.primary_backend, self.secondary_backend):
            if backend is None:
                continue
            load_numpy = getattr(backend, "load_numpy", None)
            if callable(load_numpy):
                return load_numpy()

        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("缺少 NumPy 依赖，请先安装 requirements.txt。") from exc
        return np

    def _merge_detections(
        self,
        primary: list[DetectionBox],
        secondary: list[DetectionBox],
    ) -> list[DetectionBox]:
        merged = list(primary)
        ordered_secondary = sorted(
            secondary,
            key=lambda item: (item.priority, -item.score, -item.area),
        )
        for candidate in ordered_secondary:
            if candidate.area <= 0:
                continue
            if any(
                self._iou(candidate, other) >= self.merge_iou_threshold
                for other in merged
            ):
                continue
            merged.append(candidate)

        return sorted(
            merged,
            key=lambda item: (item.priority, -item.score, -item.area),
        )

    @staticmethod
    def _summarize_detections(detections: list[DetectionBox]) -> str:
        if not detections:
            return "<none>"
        return ", ".join(
            (
                f"{item.source}:{item.category}"
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
