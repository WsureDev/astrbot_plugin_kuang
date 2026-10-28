from __future__ import annotations

from time import perf_counter

from .models import DetectionBox, DetectionPipelineResult


class GifDetectionStream:
    """One GIF owns one stream; model sessions remain shared by the detector.

    Track from the last *inferred* frame, not chained predictions. Full
    detection runs periodically and immediately when motion is unreliable.
    """

    def __init__(self, detector, *, interval: int = 4, max_age_ms: int = 240):
        self.detector = detector
        self.interval = max(1, int(interval))
        self.max_age_ms = max(20, int(max_age_ms))
        self._anchor = None
        self._previous = None
        self._boxes: list[DetectionBox] = []
        self._last_boxes: list[DetectionBox] = []
        self._age_frames = 0
        self._age_ms = 0

    def analyze(self, image_rgb, *, elapsed_ms: int = 0) -> DetectionPipelineResult:
        import numpy as np

        started = perf_counter()
        self._age_frames += 1
        self._age_ms += max(0, int(elapsed_ms))
        tracked = None
        reason = "initial"
        if self._anchor is not None and self._anchor.shape == image_rgb.shape:
            if self._age_frames >= self.interval or self._age_ms >= self.max_age_ms:
                reason = "periodic_refresh"
            elif np.array_equal(image_rgb, self._previous):
                tracked = list(self._last_boxes)
                reason = "identical_frame"
            elif self._boxes:
                tracked = self._track(self._anchor, image_rgb, self._boxes)
                reason = "optical_flow" if tracked is not None else "tracking_lost"
            else:
                reason = "empty_subjects_changed"
        if tracked is None:
            result = self.detector.analyze(image_rgb, previous_frame_rgb=self._previous)
            self._anchor = image_rgb.copy()
            self._boxes = list(result.final_boxes)
            self._age_frames = 0
            self._age_ms = 0
            result.decisions["temporal"] = reason
        else:
            tracked = [b.clone(stage="temporal_track") for b in tracked]
            result = DetectionPipelineResult(
                stage1_boxes=[],
                stage2_boxes=[],
                stage2_composite_boxes=[],
                arranged_boxes=tracked,
                final_boxes=list(tracked),
                tracked_boxes=tracked,
                decisions={"temporal": reason},
            )
        self._previous = image_rgb.copy()
        self._last_boxes = list(result.final_boxes)
        result.timings_ms["frame_total"] = (perf_counter() - started) * 1000
        return result

    @staticmethod
    def _track(
        previous, current, boxes: list[DetectionBox]
    ) -> list[DetectionBox] | None:
        import cv2
        import numpy as np

        height, width = current.shape[:2]
        scale = min(1.0, 320 / max(height, width))
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        old_rgb = cv2.resize(previous, size, interpolation=cv2.INTER_AREA)
        new_rgb = cv2.resize(current, size, interpolation=cv2.INTER_AREA)
        difference = np.abs(new_rgb.astype(np.int16) - old_rgb.astype(np.int16)).mean(
            axis=2
        )
        if float((difference > 60).mean()) > 0.55:
            return None
        old = cv2.cvtColor(old_rgb, cv2.COLOR_RGB2GRAY)
        new = cv2.cvtColor(new_rgb, cv2.COLOR_RGB2GRAY)
        occupied = np.zeros(old.shape, dtype=np.uint8)
        masks = []
        for box in boxes:
            mask = np.zeros(old.shape, dtype=np.uint8)
            x1, y1, x2, y2 = (round(v * scale) for v in box.as_tuple())
            mask[max(0, y1) : min(size[1], y2), max(0, x1) : min(size[0], x2)] = 255
            masks.append(mask)
            occupied |= mask
        # A newly appearing object outside existing subjects needs detection.
        if float(((difference > 30) & (occupied == 0)).mean()) > 0.12:
            return None
        tracked = []
        for box, mask in zip(boxes, masks):
            points = cv2.goodFeaturesToTrack(
                old,
                maxCorners=60,
                qualityLevel=0.02,
                minDistance=4,
                mask=mask,
                blockSize=3,
            )
            if points is None or len(points) < 4:
                return None
            moved, status, _ = cv2.calcOpticalFlowPyrLK(
                old, new, points, None, winSize=(21, 21), maxLevel=3
            )
            if moved is None:
                return None
            back, back_status, _ = cv2.calcOpticalFlowPyrLK(
                new, old, moved, None, winSize=(21, 21), maxLevel=3
            )
            if back is None:
                return None
            valid = (
                (status.ravel() == 1)
                & (back_status.ravel() == 1)
                & (np.linalg.norm((back - points).reshape(-1, 2), axis=1) < 1.5)
                & np.isfinite(moved.reshape(-1, 2)).all(axis=1)
            )
            if valid.sum() < max(4, len(points) * 0.5):
                return None
            shifts = (moved - points).reshape(-1, 2)[valid]
            median = np.median(shifts, axis=0)
            inliers = np.linalg.norm(shifts - median, axis=1) < 3
            if inliers.sum() < max(4, len(shifts) * 0.6):
                return None
            dx, dy = np.median(shifts[inliers], axis=0) / scale
            x1, x2 = max(0, round(box.x1 + dx)), min(width, round(box.x2 + dx))
            y1, y2 = max(0, round(box.y1 + dy)), min(height, round(box.y2 + dy))
            if (x2 - x1) * (y2 - y1) < box.area * 0.6 or x2 <= x1 or y2 <= y1:
                return None
            # Confirm appearance at the translated location to reject stale
            # features surviving an object disappearing or a scene change.
            x, y = round(box.x1 * scale), round(box.y1 * scale)
            w, h = max(1, round(box.width * scale)), max(1, round(box.height * scale))
            target = cv2.warpAffine(
                new, np.float32([[1, 0, -dx * scale], [0, 1, -dy * scale]]), size
            )
            patch = old[y : y + h, x : x + w].astype(np.float32)
            aligned = target[y : y + h, x : x + w].astype(np.float32)
            if not patch.size or float(np.abs(patch - aligned).mean()) > 35:
                return None
            tracked.append(box.clone(x1=x1, y1=y1, x2=x2, y2=y2))
        return tracked
