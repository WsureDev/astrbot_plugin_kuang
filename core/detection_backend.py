from __future__ import annotations

from typing import Protocol

from .models import DetectionBox


class DetectionBackend(Protocol):
    def detect(self, image_rgb) -> list[DetectionBox]:
        """Detect candidate boxes from an RGB image array."""
