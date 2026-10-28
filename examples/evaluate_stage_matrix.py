"""Evaluate single-stage and merged-stage detection strategies.

The detector is executed once with all optional stages enabled.  Every
strategy then reuses the same per-frame candidates and asks the arranger to
resolve only the selected stage set.  This makes comparisons deterministic and
avoids paying for one model inference per strategy.

Example::

    python examples/evaluate_stage_matrix.py \
        ~/Pictures/Image_1784702076126.gif

The output directory contains one rendered GIF per strategy, ``report.json``
and ``report.md``.
"""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import EspBoxDetector, EspBoxRenderer
from core.models import DetectionBox, DetectionPipelineResult


_STAGE_FIELDS = {
    "stage1": ("stage1_boxes",),
    "stage2": ("stage2_boxes", "stage2_composite_boxes"),
    "stage3": ("stage3_boxes",),
    "stage4": ("stage4_boxes",),
}

# Keep the matrix small enough to inspect while covering the useful product
# decisions: each stage alone, the existing cascade, and the main fallbacks.
_STRATEGIES: tuple[tuple[str, frozenset[str]], ...] = (
    ("stage1_only", frozenset({"stage1"})),
    ("stage2_only", frozenset({"stage2"})),
    ("stage3_only", frozenset({"stage3"})),
    ("stage4_only", frozenset({"stage4"})),
    ("stage1_stage2", frozenset({"stage1", "stage2"})),
    ("stage1_stage3", frozenset({"stage1", "stage3"})),
    ("stage3_stage4", frozenset({"stage3", "stage4"})),
    ("stage1_stage2_stage3", frozenset({"stage1", "stage2", "stage3"})),
    ("stage1_stage2_stage4", frozenset({"stage1", "stage2", "stage4"})),
    ("stage1_stage3_stage4", frozenset({"stage1", "stage3", "stage4"})),
    ("stage2_stage3_stage4", frozenset({"stage2", "stage3", "stage4"})),
    ("all_stages", frozenset({"stage1", "stage2", "stage3", "stage4"})),
)

_TEMPORAL_KIND_WEIGHTS = {
    "stage1": 3.5,
    "stage2_composite": 4.0,
    "stage2_part": 2.0,
    "stage3": 2.5,
    "stage4": 0.5,
}


@dataclass(frozen=True, slots=True)
class _TemporalCandidate:
    kind: str
    box: DetectionBox
    support: float = 0.0


def _pick_boxes(result: DetectionPipelineResult, field: str) -> list[DetectionBox]:
    return list(getattr(result, field, []) or [])


def _arrange_for_strategy(
    detector: EspBoxDetector,
    result: DetectionPipelineResult,
    stages: frozenset[str],
) -> list[DetectionBox]:
    fields: dict[str, list[DetectionBox]] = {
        "stage1_boxes": [],
        "stage2_boxes": [],
        "stage2_composite_boxes": [],
        "stage3_boxes": [],
        "stage4_boxes": [],
    }
    for stage in stages:
        for field in _STAGE_FIELDS[stage]:
            fields[field] = _pick_boxes(result, field)
    return detector._arranger.arrange(**fields)


def _temporal_candidate_frames(
    results: list[DetectionPipelineResult],
    *,
    width: int,
    height: int,
) -> list[list[_TemporalCandidate]]:
    """Build independent per-frame candidates for the temporal experiment."""
    canvas_area = max(1, width * height)
    frames: list[list[_TemporalCandidate]] = []
    for result in results:
        candidates: list[_TemporalCandidate] = []
        for box in _pick_boxes(result, "stage1_boxes"):
            candidates.append(_TemporalCandidate("stage1", box))
        for box in _pick_boxes(result, "stage2_composite_boxes"):
            candidates.append(_TemporalCandidate("stage2_composite", box))
        for box in _pick_boxes(result, "stage2_boxes"):
            if box.category in {"Head", "Torso", "Legs"}:
                candidates.append(_TemporalCandidate("stage2_part", box))
        for box in _pick_boxes(result, "stage3_boxes"):
            candidates.append(_TemporalCandidate("stage3", box))
        # Motion boxes are useful as a last-resort cue, but near-global frame
        # differences are usually camera/encoding movement rather than a
        # subject.  Keep them in the experiment only below this cap.
        for box in _pick_boxes(result, "stage4_boxes"):
            if box.area / canvas_area < 0.75:
                candidates.append(_TemporalCandidate("stage4", box))
        frames.append(candidates)

    # A candidate gets a small persistence bonus when a similar box appears in
    # neighboring frames.  This is deliberately independent of stage priority
    # so a stable lower-priority fallback can beat a flashing high-priority box.
    for frame_index, candidates in enumerate(frames):
        start = max(0, frame_index - 2)
        end = min(len(frames), frame_index + 3)
        neighbor_count = max(1, end - start - 1)
        updated: list[_TemporalCandidate] = []
        for candidate in candidates:
            hits = 0
            for neighbor_index in range(start, end):
                if neighbor_index == frame_index:
                    continue
                if any(
                    _temporal_boxes_similar(candidate.box, other.box)
                    for other in frames[neighbor_index]
                ):
                    hits += 1
            updated.append(
                _TemporalCandidate(
                    candidate.kind,
                    candidate.box,
                    support=hits / neighbor_count,
                )
            )
        frames[frame_index] = updated
    return frames


def _temporal_boxes_similar(left: DetectionBox, right: DetectionBox) -> bool:
    iou = left.iou(right)
    dx = left.center_x - right.center_x
    dy = left.center_y - right.center_y
    scale = max(1.0, left.width, left.height, right.width, right.height)
    center_distance = math.hypot(dx, dy) / scale
    area_ratio = min(left.area, right.area) / max(1, left.area, right.area)
    return iou >= 0.08 or (center_distance <= 0.9 and area_ratio >= 0.2)


def _temporal_transition_score(left: DetectionBox, right: DetectionBox) -> float:
    iou = left.iou(right)
    dx = left.center_x - right.center_x
    dy = left.center_y - right.center_y
    scale = max(1.0, left.width, left.height, right.width, right.height)
    center_distance = math.hypot(dx, dy) / scale
    area_ratio = min(left.area, right.area) / max(1, left.area, right.area)
    return (
        (2.8 * iou)
        + (0.8 * area_ratio)
        + (0.5 * max(0.0, 1.0 - center_distance))
        - (1.8 * max(0.0, center_distance - 0.7))
    )


def _temporal_emission(candidate: _TemporalCandidate, width: int, height: int) -> float:
    area_ratio = candidate.box.area / max(1, width * height)
    broad_penalty = max(0.0, area_ratio - 0.7) * 3.0
    return (
        _TEMPORAL_KIND_WEIGHTS[candidate.kind]
        + (0.8 * candidate.box.score)
        + (1.2 * candidate.support)
        - broad_penalty
    )


def _temporal_consensus_boxes(
    results: list[DetectionPipelineResult],
    *,
    width: int,
    height: int,
    smoothing_alpha: float = 0.75,
    fast_translation_threshold: float = 0.02,
    max_gap: int = 1,
) -> list[list[DetectionBox]]:
    """Choose one temporally coherent trajectory from independent stages.

    This is an offline research baseline, not the production arranger: it uses
    all frames (Viterbi-style dynamic programming), then smooths coordinates.
    ``max_gap`` controls how many missing frames may be bridged.
    """
    candidate_frames = _temporal_candidate_frames(
        results,
        width=width,
        height=height,
    )
    if not candidate_frames:
        return []

    states: list[list[_TemporalCandidate | None]] = [
        candidates + [None] for candidates in candidate_frames
    ]
    dynamic_scores: list[list[float]] = []
    back_indexes: list[list[int | None]] = []
    for frame_index, options in enumerate(states):
        if frame_index == 0:
            dynamic_scores.append(
                [
                    _temporal_emission(option, width, height)
                    if option is not None
                    else -2.5
                    for option in options
                ]
            )
            back_indexes.append([None] * len(options))
            continue

        previous_options = states[frame_index - 1]
        previous_scores = dynamic_scores[-1]
        current_scores: list[float] = []
        current_back: list[int] = []
        for option in options:
            emission = (
                _temporal_emission(option, width, height)
                if option is not None
                else -2.5
            )
            best_score = float("-inf")
            best_index = 0
            for previous_index, previous in enumerate(previous_options):
                if option is None or previous is None:
                    transition = -0.2
                else:
                    transition = _temporal_transition_score(
                        previous.box,
                        option.box,
                    )
                score = previous_scores[previous_index] + emission + transition
                if score > best_score:
                    best_score = score
                    best_index = previous_index
            current_scores.append(best_score)
            current_back.append(best_index)
        dynamic_scores.append(current_scores)
        back_indexes.append(current_back)

    option_index = max(
        range(len(dynamic_scores[-1])),
        key=dynamic_scores[-1].__getitem__,
    )
    path: list[_TemporalCandidate | None] = []
    for frame_index in range(len(states) - 1, -1, -1):
        path.append(states[frame_index][option_index])
        previous_index = back_indexes[frame_index][option_index]
        if previous_index is None:
            break
        option_index = previous_index
    path.reverse()

    # Fill only short gaps.  Long gaps remain empty instead of inventing a
    # trajectory for a transient object.
    for frame_index, option in enumerate(path):
        if option is not None:
            continue
        previous = next(
            (
                path[index]
                for index in range(frame_index - 1, max(-1, frame_index - max_gap - 1), -1)
                if path[index] is not None
            ),
            None,
        )
        following = next(
            (
                path[index]
                for index in range(frame_index + 1, min(len(path), frame_index + max_gap + 2))
                if path[index] is not None
            ),
            None,
        )
        if previous is not None and following is not None:
            path[frame_index] = _interpolate_candidate(
                previous,
                following,
                0.5,
            )
        elif previous is not None:
            path[frame_index] = previous
        elif following is not None:
            path[frame_index] = following

    path = _correct_single_frame_jumps(path)
    alpha = max(0.0, min(1.0, float(smoothing_alpha)))
    smoothed: list[list[DetectionBox]] = []
    previous_box: DetectionBox | None = None
    previous_raw: DetectionBox | None = None
    for option in path:
        if option is None:
            smoothed.append([])
            continue
        box = option.box
        if previous_box is not None:
            # Snap quickly when a real translation is detected; only smooth
            # small residual jitter.  A fixed low-alpha EMA visibly lags a
            # moving subject, which is worse than a one-pixel wobble here.
            reference = previous_raw or previous_box
            displacement = max(
                abs(box.x1 - reference.x1),
                abs(box.y1 - reference.y1),
                abs(box.x2 - reference.x2),
                abs(box.y2 - reference.y2),
            ) / max(1.0, previous_box.width, previous_box.height)
            update_alpha = (
                1.0 if displacement >= fast_translation_threshold else alpha
            )
            box = box.clone(
                x1=round(
                    (update_alpha * box.x1)
                    + ((1.0 - update_alpha) * previous_box.x1)
                ),
                y1=round(
                    (update_alpha * box.y1)
                    + ((1.0 - update_alpha) * previous_box.y1)
                ),
                x2=round(
                    (update_alpha * box.x2)
                    + ((1.0 - update_alpha) * previous_box.x2)
                ),
                y2=round(
                    (update_alpha * box.y2)
                    + ((1.0 - update_alpha) * previous_box.y2)
                ),
            )
        box = box.clone(stage=f"temporal_{option.kind}")
        smoothed.append([box])
        previous_box = box
        previous_raw = option.box
    return smoothed


def _correct_single_frame_jumps(
    path: list[_TemporalCandidate | None],
) -> list[_TemporalCandidate | None]:
    """Replace an isolated coordinate jump with a local interpolation.

    This is intentionally conservative: it only changes a point when the
    preceding and following observations look like the same box.  Sustained
    translations therefore pass through immediately instead of being slowed
    by a global smoothing factor.
    """
    corrected = list(path)
    for index in range(1, len(path) - 1):
        previous = path[index - 1]
        current = path[index]
        following = path[index + 1]
        if previous is None or current is None or following is None:
            continue
        if not _temporal_boxes_similar(previous.box, following.box):
            continue
        midpoint = _interpolate_candidate(previous, following, 0.5)
        reference_scale = max(
            1.0,
            previous.box.width,
            previous.box.height,
            following.box.width,
            following.box.height,
        )
        residual = max(
            abs(current.box.x1 - midpoint.box.x1),
            abs(current.box.y1 - midpoint.box.y1),
            abs(current.box.x2 - midpoint.box.x2),
            abs(current.box.y2 - midpoint.box.y2),
        ) / reference_scale
        if residual >= 0.25:
            corrected[index] = midpoint
    return corrected


def _interpolate_candidate(
    previous: _TemporalCandidate,
    following: _TemporalCandidate,
    ratio: float,
) -> _TemporalCandidate:
    left = previous.box
    right = following.box
    return _TemporalCandidate(
        kind=previous.kind,
        support=min(previous.support, following.support),
        box=left.clone(
            x1=round(left.x1 + ((right.x1 - left.x1) * ratio)),
            y1=round(left.y1 + ((right.y1 - left.y1) * ratio)),
            x2=round(left.x2 + ((right.x2 - left.x2) * ratio)),
            y2=round(left.y2 + ((right.y2 - left.y2) * ratio)),
            score=left.score + ((right.score - left.score) * ratio),
        ),
    )


def _box_area_ratio(boxes: Iterable[DetectionBox], width: int, height: int) -> float:
    canvas_area = max(1, width * height)
    return sum(box.area for box in boxes) / canvas_area


def _pairwise_overlap_count(boxes: list[DetectionBox]) -> int:
    return sum(
        1
        for index, left in enumerate(boxes)
        for right in boxes[index + 1 :]
        if left.overlaps(right)
    )


def _best_transition_iou(
    previous: list[DetectionBox],
    current: list[DetectionBox],
) -> tuple[float, float]:
    """Return mean matched IoU and matched-pair ratio for two frames."""
    if not previous and not current:
        return 1.0, 1.0
    if not previous or not current:
        return 0.0, 0.0

    available = set(range(len(previous)))
    matches: list[float] = []
    for candidate in current:
        if not available:
            break
        best_index = max(available, key=lambda index: candidate.iou(previous[index]))
        best_iou = candidate.iou(previous[best_index])
        if best_iou > 0.0:
            matches.append(best_iou)
            available.remove(best_index)
    denominator = max(len(previous), len(current))
    return (
        (statistics.mean(matches) if matches else 0.0),
        len(matches) / denominator,
    )


def _summarize_strategy(
    boxes_by_frame: list[list[DetectionBox]],
    *,
    width: int,
    height: int,
) -> dict[str, object]:
    counts = [len(boxes) for boxes in boxes_by_frame]
    area_ratios = [_box_area_ratio(boxes, width, height) for boxes in boxes_by_frame]
    scores = [box.score for boxes in boxes_by_frame for box in boxes]
    stage_counts = Counter(
        box.stage or box.semantic_role or "unknown"
        for boxes in boxes_by_frame
        for box in boxes
    )
    source_counts = Counter(
        box.source or "unknown"
        for boxes in boxes_by_frame
        for box in boxes
    )
    overlap_pairs = sum(_pairwise_overlap_count(boxes) for boxes in boxes_by_frame)

    transition_ious: list[float] = []
    transition_match_ratios: list[float] = []
    for previous, current in pairwise(boxes_by_frame):
        iou, match_ratio = _best_transition_iou(previous, current)
        transition_ious.append(iou)
        transition_match_ratios.append(match_ratio)

    def rounded_mean(values: list[float]) -> float:
        return round(statistics.mean(values), 4) if values else 0.0

    return {
        "frames": len(boxes_by_frame),
        "count_mean": rounded_mean([float(value) for value in counts]),
        "count_median": statistics.median(counts) if counts else 0,
        "count_max": max(counts, default=0),
        "empty_frames": sum(value == 0 for value in counts),
        "area_ratio_mean": rounded_mean(area_ratios),
        "area_ratio_median": round(statistics.median(area_ratios), 4)
        if area_ratios
        else 0.0,
        "score_mean": rounded_mean(scores),
        "overlap_pairs": overlap_pairs,
        "transition_iou_mean": rounded_mean(transition_ious),
        "transition_match_mean": rounded_mean(transition_match_ratios),
        "selected_stage_counts": dict(stage_counts),
        "selected_source_counts": dict(source_counts),
    }


def _save_gif(frames: list, durations: list[int], loop: int, output_path: Path) -> None:
    from PIL import Image

    prepared = [
        frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
        for frame in frames
    ]
    prepared[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=prepared[1:],
        duration=durations,
        loop=loop,
        disposal=2,
        optimize=True,
    )


def _render_strategy(
    frames: list,
    boxes_by_frame: list[list[DetectionBox]],
    durations: list[int],
    loop: int,
    output_path: Path,
    renderer: EspBoxRenderer,
    is_gif: bool,
) -> None:
    rendered = [
        renderer.draw_boxes(frame, boxes)
        for frame, boxes in zip(frames, boxes_by_frame)
    ]
    if is_gif:
        _save_gif(rendered, durations, loop, output_path)
    else:
        rendered[0].convert("RGB").save(output_path, format="PNG")


def _build_detector(model_dir: Path, box_count: int) -> EspBoxDetector:
    return EspBoxDetector(
        box_count=box_count,
        model_path=str(model_dir / "yolo26n.onnx"),
        auto_download_model=True,
        enable_random_boxes=False,
        enable_anime_fallback=True,
        anime_model_path=str(model_dir / "anime_yolo.onnx"),
        anime_auto_download_model=True,
        anime_fallback_trigger_count=999,
        enable_saliency=True,
        saliency_model_path=str(model_dir / "u2netp.onnx"),
        saliency_auto_download_model=True,
        enable_motion=True,
    )


def _is_animated_gif(path: Path) -> bool:
    from PIL import Image

    with Image.open(path) as image:
        return image.format == "GIF" and bool(getattr(image, "is_animated", False))


def _collect_frames(
    image_path: Path,
    detector: EspBoxDetector,
) -> tuple[list, list[DetectionPipelineResult], list[int], int, bool, tuple[int, int]]:
    import numpy as np
    from PIL import Image, ImageSequence

    is_gif = _is_animated_gif(image_path)
    frames: list = []
    results: list[DetectionPipelineResult] = []
    durations: list[int] = []
    loop = 0
    previous_frame_array = None

    with Image.open(image_path) as image:
        base_duration = int(image.info.get("duration", 100) or 100)
        loop = int(image.info.get("loop", 0) or 0)
        iterator = ImageSequence.Iterator(image) if is_gif else (image,)
        for frame in iterator:
            duration = int(frame.info.get("duration", base_duration) or base_duration)
            rgba_frame = frame.convert("RGBA")
            rgb_array = np.asarray(rgba_frame.convert("RGB")).copy()
            result = detector.analyze(
                rgb_array,
                previous_frame_rgb=previous_frame_array,
            )
            frames.append(rgba_frame.copy())
            results.append(result)
            durations.append(max(20, duration))
            previous_frame_array = rgb_array

    if not frames:
        raise RuntimeError("输入文件没有可处理的帧。")
    return frames, results, durations, loop, is_gif, frames[0].size


def _markdown_report(report: dict[str, object]) -> str:
    lines = [
        "# Stage 合并矩阵评估",
        "",
        f"- 输入：`{report['input']}`",
        f"- 尺寸：`{report['size'][0]}x{report['size'][1]}`",
        f"- 帧数：`{report['frames']}`",
        f"- 全阶段单次采集耗时：`{report['elapsed_seconds']:.2f}s`",
        (
            f"- 时间策略：`alpha={report['temporal_config']['alpha']}`，"
            f"`fast_threshold={report['temporal_config']['fast_translation_threshold']}`，"
            f"`max_gap={report['temporal_config']['max_gap']}`"
        ),
        "",
        "| 策略 | 平均框数 | 最大框数 | 空帧 | 平均覆盖率 | 相邻帧 IoU | 匹配率 | 选中来源 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, item in report["strategies"].items():
        stages = "+".join(item["active_stages"])
        stage_counts = ", ".join(
            f"{key}:{value}" for key, value in item["metrics"]["selected_stage_counts"].items()
        ) or "-"
        metrics = item["metrics"]
        lines.append(
            f"| `{name}` ({stages}) | {metrics['count_mean']} | {metrics['count_max']} | "
            f"{metrics['empty_frames']} | {metrics['area_ratio_mean']} | "
            f"{metrics['transition_iou_mean']} | {metrics['transition_match_mean']} | "
            f"{stage_counts} |"
        )
    lines.extend(
        [
            "",
            "指标说明：覆盖率为所有选中框面积之和/画布面积，可能因重叠而超过真实 union；相邻帧 IoU 使用贪心匹配，主要用于比较稳定性，不是检测准确率。",
        ]
    )
    return "\n".join(lines) + "\n"


def evaluate_one(
    image_path: Path,
    *,
    output_root: Path,
    model_dir: Path,
    box_count: int,
    render: bool,
    temporal_alpha: float,
    temporal_fast_threshold: float,
    temporal_max_gap: int,
) -> Path:
    detector = _build_detector(model_dir, box_count)
    renderer = EspBoxRenderer()
    output_dir = output_root / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    frames, results, durations, loop, is_gif, (width, height) = _collect_frames(
        image_path,
        detector,
    )
    elapsed = time.perf_counter() - started

    report: dict[str, object] = {
        "input": str(image_path),
        "size": [width, height],
        "frames": len(frames),
        "is_gif": is_gif,
        "elapsed_seconds": round(elapsed, 4),
        "temporal_config": {
            "alpha": temporal_alpha,
            "fast_translation_threshold": temporal_fast_threshold,
            "max_gap": temporal_max_gap,
        },
        "stage_candidates": {},
        "strategies": {},
    }
    for stage, fields in _STAGE_FIELDS.items():
        counts = [
            sum(len(_pick_boxes(result, field)) for field in fields)
            for result in results
        ]
        report["stage_candidates"][stage] = {
            "count_mean": round(statistics.mean(counts), 4) if counts else 0.0,
            "count_max": max(counts, default=0),
            "empty_frames": sum(value == 0 for value in counts),
        }

    temporal_boxes_by_frame = _temporal_consensus_boxes(
        results,
        width=width,
        height=height,
        smoothing_alpha=temporal_alpha,
        fast_translation_threshold=temporal_fast_threshold,
        max_gap=temporal_max_gap,
    )

    for name, stages in _STRATEGIES:
        boxes_by_frame = [
            _arrange_for_strategy(detector, result, stages)
            for result in results
        ]
        metrics = _summarize_strategy(
            boxes_by_frame,
            width=width,
            height=height,
        )
        strategy_report = {
            "active_stages": sorted(stages),
            "metrics": metrics,
        }
        if render:
            suffix = ".gif" if is_gif else ".png"
            output_path = output_dir / f"{name}{suffix}"
            _render_strategy(
                frames,
                boxes_by_frame,
                durations,
                loop,
                output_path,
                renderer,
                is_gif,
            )
            strategy_report["output"] = str(output_path)
        report["strategies"][name] = strategy_report

    temporal_metrics = _summarize_strategy(
        temporal_boxes_by_frame,
        width=width,
        height=height,
    )
    temporal_report: dict[str, object] = {
        "active_stages": ["temporal_consensus"],
        "metrics": temporal_metrics,
    }
    if render:
        suffix = ".gif" if is_gif else ".png"
        output_path = output_dir / f"temporal_consensus{suffix}"
        _render_strategy(
            frames,
            temporal_boxes_by_frame,
            durations,
            loop,
            output_path,
            renderer,
            is_gif,
        )
        temporal_report["output"] = str(output_path)
    report["strategies"]["temporal_consensus"] = temporal_report

    json_path = output_dir / "report.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        _markdown_report(report),
        encoding="utf-8",
    )
    print(f"✅ {image_path.name}: {elapsed:.2f}s -> {output_dir}")
    return json_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="评估单阶段与合并 Stage 策略。")
    parser.add_argument("images", type=Path, nargs="+", help="输入图片或 GIF。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "sample_output" / "stage_matrix",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "sample_output" / "models",
    )
    parser.add_argument("--box-count", type=int, default=5)
    parser.add_argument(
        "--temporal-alpha",
        type=float,
        default=0.75,
        help="仅用于小抖动的平滑系数；大于 fast-threshold 的平移会立即跟随。",
    )
    parser.add_argument(
        "--temporal-fast-threshold",
        type=float,
        default=0.02,
        help="相对框尺寸的位移阈值，超过后不做低通平滑。",
    )
    parser.add_argument(
        "--temporal-max-gap",
        type=int,
        default=1,
        help="轨迹最多允许跨越的缺失帧数。",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="只写指标，不输出各策略 GIF/PNG。",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_root = args.output_dir.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    for raw_path in args.images:
        image_path = raw_path.expanduser().resolve()
        if not image_path.exists():
            print(f"❌ 找不到输入文件: {image_path}", file=sys.stderr)
            continue
        evaluate_one(
            image_path,
            output_root=output_root,
            model_dir=model_dir,
            box_count=args.box_count,
            render=not args.no_render,
            temporal_alpha=args.temporal_alpha,
            temporal_fast_threshold=args.temporal_fast_threshold,
            temporal_max_gap=args.temporal_max_gap,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
