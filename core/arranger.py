from __future__ import annotations

from .logger import get_logger
from .models import DetectionBox

_logger = get_logger(__name__)
_ANIME_HEAD = "Head"
_ANIME_TORSO = "Torso"
_ANIME_LEGS = "Legs"
_STAGE1_HUMANOID_GUESS = "stage1_humanoid_guess"

# booru_yolo part groupings for composite assembly
_BOORU_HEAD_CLASSES = frozenset({
    "head", "hdrago", "hpony", "hfox", "hrabb", "hcat", "hbear", "hhorse", "hbird",
})
_BOORU_UPPER_CLASSES = frozenset({"bust", "shld"})
_BOORU_LOWER_CLASSES = frozenset({"belly", "butt", "hip", "split", "vsplt"})
_HUMANOID_ROLES = {
    "person_full",
    _STAGE1_HUMANOID_GUESS,
    "anime_full_body",
    "anime_upper_body",
    "anime_torso",
    "anime_head",
    "anime_legs",
}
_ANIMAL_CATEGORIES = {
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


class DetectionArranger:
    def __init__(
        self,
        *,
        box_count: int,
        overlap_iou_threshold: float = 0.55,
        containment_threshold: float = 0.72,
        humanoid_containment_threshold: float = 0.55,
    ) -> None:
        self.box_count = max(1, int(box_count))
        self.overlap_iou_threshold = max(0.0, float(overlap_iou_threshold))
        self.containment_threshold = max(0.0, float(containment_threshold))
        self.humanoid_containment_threshold = max(
            0.0,
            float(humanoid_containment_threshold),
        )

    def compose_anime_boxes(
        self,
        boxes: list[DetectionBox],
    ) -> list[DetectionBox]:
        heads = [item for item in boxes if item.category == _ANIME_HEAD]
        torsos = [item for item in boxes if item.category == _ANIME_TORSO]
        legs = [item for item in boxes if item.category == _ANIME_LEGS]
        used_heads: set[int] = set()
        used_legs: set[int] = set()
        composites: list[DetectionBox] = []
        _logger.debug(
            "[arranger] compose anime start: heads=%s torsos=%s legs=%s",
            len(heads),
            len(torsos),
            len(legs),
        )
        for torso in sorted(torsos, key=self._selection_sort_key):
            head_index = self._pick_best_head_index(torso, heads, used_heads)
            leg_index = self._pick_best_leg_index(torso, legs, used_legs)
            parts = [torso]
            part_names = ["torso"]
            if head_index is not None:
                parts.append(heads[head_index])
                part_names.insert(0, "head")
                used_heads.add(head_index)
            if leg_index is not None:
                parts.append(legs[leg_index])
                part_names.append("legs")
                used_legs.add(leg_index)
            if len(parts) == 1:
                continue

            union_box = torso.union(*parts[1:]).clone(
                source="anime_yolo_composite",
                category="+".join(part_names),
                stage="stage2_composite",
                part="+".join(part_names),
                semantic_role=(
                    "anime_full_body"
                    if "head" in part_names and "legs" in part_names
                    else "anime_upper_body"
                ),
                priority=(
                    1
                    if "head" in part_names and "legs" in part_names
                    else 2
                ),
                composite=True,
            )
            composites.append(union_box)
            _logger.debug(
                "[arranger] anime composite built: torso=%s head=%s legs=%s composite=%s",
                torso.describe(),
                heads[head_index].describe() if head_index is not None else "<none>",
                legs[leg_index].describe() if leg_index is not None else "<none>",
                union_box.describe(),
            )
        _logger.debug(
            "[arranger] compose anime done: composites=%s details=%s",
            len(composites),
            self._summarize(composites),
        )
        return composites

    def compose_booru_boxes(
        self,
        boxes: list[DetectionBox],
    ) -> list[DetectionBox]:
        """Assemble booru_yolo fine-grained parts into full/upper-body composites.

        Strategy: use bust/shld (upper body core) as anchor, associate head above
        and belly/hip/butt below to form full_body or upper_body composites.
        """
        heads = [item for item in boxes if item.category in _BOORU_HEAD_CLASSES]
        uppers = [item for item in boxes if item.category in _BOORU_UPPER_CLASSES]
        lowers = [item for item in boxes if item.category in _BOORU_LOWER_CLASSES]
        used_heads: set[int] = set()
        used_lowers: set[int] = set()
        composites: list[DetectionBox] = []
        _logger.debug(
            "[arranger] compose booru start: heads=%s uppers=%s lowers=%s",
            len(heads),
            len(uppers),
            len(lowers),
        )
        for upper in sorted(uppers, key=self._selection_sort_key):
            head_index = self._pick_best_part_index(
                anchor=upper, parts=heads, used_indices=used_heads, part_name="head",
            )
            lower_index = self._pick_best_part_index(
                anchor=upper, parts=lowers, used_indices=used_lowers, part_name="legs",
            )
            parts = [upper]
            part_names = ["upper"]
            if head_index is not None:
                parts.append(heads[head_index])
                part_names.insert(0, "head")
                used_heads.add(head_index)
            if lower_index is not None:
                parts.append(lowers[lower_index])
                part_names.append("lower")
                used_lowers.add(lower_index)
            if len(parts) == 1:
                continue

            has_head = "head" in part_names
            has_lower = "lower" in part_names
            union_box = upper.union(*parts[1:]).clone(
                source="booru_yolo_composite",
                category="+".join(part_names),
                stage="stage2_composite",
                part="+".join(part_names),
                semantic_role=(
                    "anime_full_body" if has_head and has_lower
                    else "anime_upper_body"
                ),
                priority=(1 if has_head and has_lower else 2),
                composite=True,
            )
            composites.append(union_box)
            _logger.debug(
                "[arranger] booru composite built: upper=%s head=%s lower=%s composite=%s",
                upper.describe(),
                heads[head_index].describe() if head_index is not None else "<none>",
                lowers[lower_index].describe() if lower_index is not None else "<none>",
                union_box.describe(),
            )
        _logger.debug(
            "[arranger] compose booru done: composites=%s details=%s",
            len(composites),
            self._summarize(composites),
        )
        return composites

    def compose_stage2_boxes(
        self,
        boxes: list[DetectionBox],
    ) -> list[DetectionBox]:
        """Unified entry: dispatch to appropriate composition strategy by source."""
        booru_boxes = [b for b in boxes if b.source == "booru_yolo"]
        anime_boxes = [b for b in boxes if b.source != "booru_yolo"]
        composites: list[DetectionBox] = []
        if booru_boxes:
            composites.extend(self.compose_booru_boxes(booru_boxes))
        if anime_boxes:
            composites.extend(self.compose_anime_boxes(anime_boxes))
        return composites

    def arrange(
        self,
        *,
        stage1_boxes: list[DetectionBox],
        stage2_boxes: list[DetectionBox],
        stage2_composite_boxes: list[DetectionBox],
    ) -> list[DetectionBox]:
        candidates = self._build_candidates(
            stage1_boxes=stage1_boxes,
            stage2_boxes=stage2_boxes,
            stage2_composite_boxes=stage2_composite_boxes,
        )
        _logger.debug(
            "[arranger] arrange start: candidates=%s details=%s",
            len(candidates),
            self._summarize(candidates),
        )
        selected: list[DetectionBox] = []
        for candidate in sorted(candidates, key=self._selection_sort_key):
            blocked_by, replacement_indexes = self._resolve_conflicts(candidate, selected)
            if blocked_by is not None:
                _logger.debug(
                    "[arranger] drop candidate because of conflict: candidate=%s conflict=%s iou=%.3f containment=%.3f",
                    candidate.describe(),
                    blocked_by.describe(),
                    candidate.iou(blocked_by),
                    candidate.containment_ratio(blocked_by),
                )
                continue
            if replacement_indexes:
                removed = [selected[index].describe() for index in replacement_indexes]
                for index in sorted(replacement_indexes, reverse=True):
                    selected.pop(index)
                _logger.debug(
                    "[arranger] replace selected with candidate: candidate=%s removed=%s",
                    candidate.describe(),
                    removed,
                )
            selected.append(candidate)
            _logger.debug("[arranger] accept candidate: %s", candidate.describe())
        if len(selected) > self.box_count:
            _logger.debug(
                "[arranger] trim accepted candidates: kept=%s dropped=%s",
                self.box_count,
                len(selected) - self.box_count,
            )
            selected = selected[: self.box_count]
        _logger.debug(
            "[arranger] arrange done: selected=%s details=%s",
            len(selected),
            self._summarize(selected),
        )
        return selected

    def _build_candidates(
        self,
        *,
        stage1_boxes: list[DetectionBox],
        stage2_boxes: list[DetectionBox],
        stage2_composite_boxes: list[DetectionBox],
    ) -> list[DetectionBox]:
        normalized_stage2_composites: list[DetectionBox] = []
        for item in stage2_composite_boxes:
            semantic_role = item.semantic_role or "anime_upper_body"
            normalized_stage2_composites.append(
                item.clone(
                    stage=item.stage or "stage2_composite",
                    semantic_role=semantic_role,
                    part=item.part or "torso",
                    priority=self._arrangement_priority_for_role(semantic_role),
                )
            )

        normalized_stage2_boxes: list[DetectionBox] = []
        for item in stage2_boxes:
            semantic_role = item.semantic_role or self._semantic_role_for_stage2(item)
            normalized_stage2_boxes.append(
                item.clone(
                    stage=item.stage or "stage2_anime",
                    semantic_role=semantic_role,
                    part=item.part or item.category.lower(),
                    priority=self._arrangement_priority_for_role(semantic_role),
                )
            )

        stage2_humanoid_hints = [
            item
            for item in (*normalized_stage2_composites, *normalized_stage2_boxes)
            if self._is_stage2_humanoid(item)
        ]

        candidates: list[DetectionBox] = []
        for item in stage1_boxes:
            semantic_role = item.semantic_role
            if semantic_role in {"", "other"}:
                semantic_role = self._semantic_role_for_stage1(
                    item,
                    stage2_humanoid_hints,
                )
            if semantic_role != item.semantic_role:
                _logger.debug(
                    "[arranger] recomputed stage1 semantic role: before=%s after=%s box=%s",
                    item.semantic_role or "<empty>",
                    semantic_role,
                    item.describe(),
                )
            candidates.append(
                item.clone(
                    stage=item.stage or "stage1_primary",
                    semantic_role=semantic_role,
                    part=item.part or item.category.lower(),
                    priority=self._arrangement_priority_for_role(semantic_role),
                )
            )
        candidates.extend(normalized_stage2_composites)
        candidates.extend(normalized_stage2_boxes)
        return candidates

    def _resolve_conflicts(
        self,
        candidate: DetectionBox,
        selected: list[DetectionBox],
    ) -> tuple[DetectionBox | None, list[int]]:
        replacement_indexes: list[int] = []
        for index, other in enumerate(selected):
            decision = self._decide_conflict_action(candidate, other)
            if decision == "keep_both":
                continue
            if decision == "replace_selected":
                replacement_indexes.append(index)
                continue
            return other, []
        return None, replacement_indexes

    def _decide_conflict_action(
        self,
        candidate: DetectionBox,
        other: DetectionBox,
    ) -> str:
        if not candidate.overlaps(other):
            return "keep_both"
        if not self._has_arrangement_conflict(candidate, other):
            return "keep_both"

        if self._is_stage2_humanoid(candidate) and self._is_stage1_humanoid_candidate(other):
            if not self._is_same_target_pair(candidate, other):
                return "keep_both"
            if self._should_stage2_replace_stage1(candidate, other):
                return "replace_selected"
            return "drop_candidate"

        if self._is_stage1_humanoid_candidate(candidate) and self._is_stage2_humanoid(other):
            if not self._is_same_target_pair(candidate, other):
                return "keep_both"
            if self._should_stage2_replace_stage1(other, candidate):
                return "drop_candidate"
            return "replace_selected"

        if (
            self._semantic_family(candidate) == "humanoid"
            and self._semantic_family(other) == "humanoid"
        ):
            if not self._is_same_target_pair(candidate, other):
                return "keep_both"
            return "drop_candidate"

        return "drop_candidate"

    def _has_arrangement_conflict(
        self,
        candidate: DetectionBox,
        other: DetectionBox,
    ) -> bool:
        iou = candidate.iou(other)
        containment = candidate.containment_ratio(other)
        if iou >= self.overlap_iou_threshold:
            return True
        if containment >= self.containment_threshold:
            return True
        if (
            self._semantic_family(candidate) == "humanoid"
            and self._semantic_family(other) == "humanoid"
            and self._smaller_overlap_ratio(candidate, other)
            >= self.humanoid_containment_threshold
        ):
            return True
        return False

    def _is_same_target_pair(
        self,
        candidate: DetectionBox,
        other: DetectionBox,
    ) -> bool:
        if (
            self._semantic_family(candidate) != "humanoid"
            or self._semantic_family(other) != "humanoid"
        ):
            return False
        if (
            self._smaller_overlap_ratio(candidate, other)
            < self.humanoid_containment_threshold
        ):
            return False
        if self._center_offset_ratio(candidate, other) > 0.72:
            return False
        if not self._role_alignment_ok(candidate, other):
            return False
        return True

    def _should_stage2_replace_stage1(
        self,
        stage2_box: DetectionBox,
        stage1_box: DetectionBox,
    ) -> bool:
        if stage1_box.area >= stage2_box.area:
            return False
        if not stage2_box.composite:
            return False
        if stage2_box.semantic_role not in {"anime_full_body", "anime_upper_body"}:
            return False
        if (
            self._smaller_overlap_ratio(stage2_box, stage1_box)
            < self.containment_threshold
        ):
            return False
        if self._center_offset_ratio(stage2_box, stage1_box) > 0.45:
            return False
        return True

    def _pick_best_head_index(
        self,
        torso: DetectionBox,
        heads: list[DetectionBox],
        used_heads: set[int],
    ) -> int | None:
        return self._pick_best_part_index(
            anchor=torso,
            parts=heads,
            used_indices=used_heads,
            part_name="head",
        )

    def _pick_best_leg_index(
        self,
        torso: DetectionBox,
        legs: list[DetectionBox],
        used_legs: set[int],
    ) -> int | None:
        return self._pick_best_part_index(
            anchor=torso,
            parts=legs,
            used_indices=used_legs,
            part_name="legs",
        )

    def _pick_best_part_index(
        self,
        *,
        anchor: DetectionBox,
        parts: list[DetectionBox],
        used_indices: set[int],
        part_name: str,
    ) -> int | None:
        best_index: int | None = None
        best_score = float("-inf")
        for index, part in enumerate(parts):
            if index in used_indices:
                continue
            score = self._part_alignment_score(anchor, part, part_name)
            if score <= 0:
                continue
            if score > best_score:
                best_score = score
                best_index = index
        return best_index

    def _part_alignment_score(
        self,
        anchor: DetectionBox,
        part: DetectionBox,
        part_name: str,
    ) -> float:
        center_offset_ratio = abs(anchor.center_x - part.center_x) / max(
            1.0,
            max(anchor.width, part.width),
        )
        horizontal_overlap = max(0, min(anchor.x2, part.x2) - max(anchor.x1, part.x1))
        horizontal_overlap_ratio = horizontal_overlap / max(1.0, min(anchor.width, part.width))
        if center_offset_ratio > 0.95:
            return 0.0

        if part_name == "head":
            if part.center_y > anchor.center_y:
                return 0.0
            if part.y2 > anchor.y1 + int(anchor.height * 0.75):
                return 0.0
            vertical_gap = max(0, anchor.y1 - part.y2)
        else:
            if part.center_y < anchor.center_y:
                return 0.0
            if part.y1 < anchor.y1 + int(anchor.height * 0.15):
                return 0.0
            vertical_gap = max(0, part.y1 - anchor.y2)

        return (
            (horizontal_overlap_ratio * 3.0)
            + max(0.0, 1.1 - center_offset_ratio)
            + max(0.0, 1.0 - (vertical_gap / max(1.0, anchor.height)))
            + (part.score * 0.2)
        )

    def _semantic_role_for_stage1(
        self,
        box: DetectionBox,
        stage2_humanoid_hints: list[DetectionBox],
    ) -> str:
        if box.category == "person":
            return "person_full"
        if box.category in _ANIMAL_CATEGORIES:
            return "animal"
        if self._looks_like_stage1_humanoid_guess(box, stage2_humanoid_hints):
            return _STAGE1_HUMANOID_GUESS
        return "other"

    @staticmethod
    def _semantic_role_for_stage2(box: DetectionBox) -> str:
        if box.category == _ANIME_TORSO:
            return "anime_torso"
        if box.category == _ANIME_HEAD:
            return "anime_head"
        if box.category == _ANIME_LEGS:
            return "anime_legs"
        return "other"

    @staticmethod
    def _semantic_family(box: DetectionBox) -> str:
        if box.semantic_role in _HUMANOID_ROLES:
            return "humanoid"
        if box.semantic_role == "animal" or box.category in _ANIMAL_CATEGORIES:
            return "animal"
        if box.category == "random":
            return "random"
        return "other"

    @staticmethod
    def _smaller_overlap_ratio(left: DetectionBox, right: DetectionBox) -> float:
        smaller_area = min(left.area, right.area)
        if smaller_area <= 0:
            return 0.0
        return left.intersection_area(right) / smaller_area

    @staticmethod
    def _center_offset_ratio(left: DetectionBox, right: DetectionBox) -> float:
        reference_width = max(1.0, max(left.width, right.width))
        return abs(left.center_x - right.center_x) / reference_width

    def _role_alignment_ok(
        self,
        left: DetectionBox,
        right: DetectionBox,
    ) -> bool:
        return (
            self._single_role_alignment_ok(left, right)
            and self._single_role_alignment_ok(right, left)
        )

    def _single_role_alignment_ok(
        self,
        part_box: DetectionBox,
        other: DetectionBox,
    ) -> bool:
        role = part_box.semantic_role
        if role == "anime_head":
            return part_box.center_y <= other.y1 + (other.height * 0.68)
        if role == "anime_legs":
            return part_box.center_y >= other.y1 + (other.height * 0.32)
        if role == "anime_torso":
            return (
                part_box.center_y >= other.y1 + (other.height * 0.18)
                and part_box.center_y <= other.y1 + (other.height * 0.82)
            )
        return True

    def _looks_like_stage1_humanoid_guess(
        self,
        box: DetectionBox,
        stage2_humanoid_hints: list[DetectionBox],
    ) -> bool:
        for hint in stage2_humanoid_hints:
            if not box.overlaps(hint):
                continue
            if box.area <= hint.area:
                continue
            if (
                self._smaller_overlap_ratio(box, hint)
                < self.humanoid_containment_threshold
            ):
                continue
            if self._center_offset_ratio(box, hint) > 0.35:
                continue
            if not self._role_alignment_ok(hint, box):
                continue
            return True
        return False

    @staticmethod
    def _is_stage1_humanoid_candidate(box: DetectionBox) -> bool:
        return box.stage == "stage1_primary" and (
            box.semantic_role in {"person_full", _STAGE1_HUMANOID_GUESS}
        )

    @staticmethod
    def _is_stage2_humanoid(box: DetectionBox) -> bool:
        return box.stage in {"stage2_anime", "stage2_composite"} and (
            box.semantic_role in _HUMANOID_ROLES
        )

    @staticmethod
    def _selection_sort_key(box: DetectionBox) -> tuple[int, int, float]:
        return (box.priority, -int(round(box.score * 1000)), -float(box.area))

    @staticmethod
    def _arrangement_priority_for_role(role: str) -> int:
        order = {
            "person_full": 0,
            "anime_full_body": 1,
            _STAGE1_HUMANOID_GUESS: 2,
            "anime_upper_body": 2,
            "anime_torso": 3,
            "anime_head": 4,
            "anime_legs": 5,
            "animal": 6,
            "other": 7,
        }
        return order.get(role, 7)

    @staticmethod
    def _summarize(boxes: list[DetectionBox]) -> str:
        if not boxes:
            return "<none>"
        return ", ".join(item.describe() for item in boxes)
