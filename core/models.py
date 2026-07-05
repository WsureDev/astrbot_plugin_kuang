from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DetectionBox:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    category: str
    priority: int
    source: str

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2
