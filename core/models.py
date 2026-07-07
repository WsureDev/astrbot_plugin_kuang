from __future__ import annotations

from dataclasses import dataclass, replace


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
    stage: str = ""
    part: str = ""
    semantic_role: str = ""
    composite: bool = False

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return self.x1 + (self.width / 2.0)

    @property
    def center_y(self) -> float:
        return self.y1 + (self.height / 2.0)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    def clone(self, **changes) -> "DetectionBox":
        return replace(self, **changes)

    def overlaps(self, other: "DetectionBox") -> bool:
        return (
            max(self.x1, other.x1) < min(self.x2, other.x2)
            and max(self.y1, other.y1) < min(self.y2, other.y2)
        )

    def intersection_area(self, other: "DetectionBox") -> int:
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)
        inter_width = max(0, inter_x2 - inter_x1)
        inter_height = max(0, inter_y2 - inter_y1)
        return inter_width * inter_height

    def iou(self, other: "DetectionBox") -> float:
        inter_area = self.intersection_area(other)
        if inter_area <= 0:
            return 0.0
        union_area = self.area + other.area - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def containment_ratio(self, other: "DetectionBox") -> float:
        inter_area = self.intersection_area(other)
        if inter_area <= 0:
            return 0.0
        smallest_area = min(self.area, other.area)
        if smallest_area <= 0:
            return 0.0
        return inter_area / smallest_area

    def union(self, *others: "DetectionBox") -> "DetectionBox":
        boxes = (self, *others)
        return DetectionBox(
            x1=min(item.x1 for item in boxes),
            y1=min(item.y1 for item in boxes),
            x2=max(item.x2 for item in boxes),
            y2=max(item.y2 for item in boxes),
            score=max(item.score for item in boxes),
            category=self.category,
            priority=self.priority,
            source=self.source,
            stage=self.stage,
            part=self.part,
            semantic_role=self.semantic_role,
            composite=self.composite,
        )

    def describe(self) -> str:
        return (
            f"{self.stage or '-'}|{self.source}:{self.category}"
            f"|role={self.semantic_role or '-'}"
            f"|part={self.part or '-'}"
            f"|score={self.score:.2f}"
            f"|prio={self.priority}"
            f"|box=[{self.x1},{self.y1},{self.x2},{self.y2}]"
        )


@dataclass(slots=True)
class DetectionPipelineResult:
    stage1_boxes: list[DetectionBox]
    stage2_boxes: list[DetectionBox]
    stage2_composite_boxes: list[DetectionBox]
    arranged_boxes: list[DetectionBox]
    random_boxes: list[DetectionBox]
    final_boxes: list[DetectionBox]
