import unittest

import cv2
import numpy as np

from core.models import DetectionBox, DetectionPipelineResult
from core.temporal import GifDetectionStream


def make_frame(dx=0, dy=0):
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    for y in range(6):
        for x in range(6):
            color = (40 + (x * 31 + y * 17) % 210, 80 + (x * 13 + y * 29) % 170, 200)
            cv2.rectangle(
                image,
                (25 + dx + x * 7, 30 + dy + y * 7),
                (28 + dx + x * 7, 33 + dy + y * 7),
                color,
                -1,
            )
    return image


class DetectorStub:
    def __init__(self, boxes=None):
        self.calls = 0
        self.boxes = (
            boxes
            if boxes is not None
            else [DetectionBox(23, 28, 69, 74, 0.9, "Head", 0, "test")]
        )

    def analyze(self, image, *, previous_frame_rgb=None):
        self.calls += 1
        return DetectionPipelineResult([], [], [], list(self.boxes), list(self.boxes))


class TemporalTests(unittest.TestCase):
    def test_translation_tracks_from_inferred_frame_without_accumulating_drift(self):
        detector = DetectorStub()
        stream = GifDetectionStream(detector, interval=5, max_age_ms=500)
        stream.analyze(make_frame())
        for shift in (2, 4, 6):
            result = stream.analyze(make_frame(shift, shift), elapsed_ms=30)
            self.assertTrue(result.tracked_boxes)
            self.assertAlmostEqual(result.final_boxes[0].x1, 23 + shift, delta=1)
            self.assertAlmostEqual(result.final_boxes[0].y1, 28 + shift, delta=1)
        self.assertEqual(detector.calls, 1)

    def test_periodic_refresh_even_when_frames_are_identical(self):
        detector = DetectorStub()
        stream = GifDetectionStream(detector, interval=2, max_age_ms=1000)
        for _ in range(5):
            stream.analyze(make_frame(), elapsed_ms=20)
        self.assertEqual(detector.calls, 3)

    def test_duration_bound_prevents_slow_frame_reuse(self):
        detector = DetectorStub()
        stream = GifDetectionStream(detector, interval=20, max_age_ms=100)
        stream.analyze(make_frame())
        stream.analyze(make_frame(), elapsed_ms=100)
        self.assertEqual(detector.calls, 2)

    def test_cut_or_disappearance_triggers_immediate_detection(self):
        for replacement in (
            np.full((128, 128, 3), 255, np.uint8),
            np.zeros((128, 128, 3), np.uint8),
        ):
            detector = DetectorStub()
            stream = GifDetectionStream(detector)
            stream.analyze(make_frame())
            result = stream.analyze(replacement, elapsed_ms=20)
            self.assertEqual(detector.calls, 2)
            self.assertFalse(result.tracked_boxes)

    def test_independent_streams_share_no_temporal_state(self):
        detector = DetectorStub()
        first, second = GifDetectionStream(detector), GifDetectionStream(detector)
        first.analyze(make_frame())
        second.analyze(make_frame())
        first.analyze(make_frame(), elapsed_ms=20)
        second.analyze(make_frame(), elapsed_ms=20)
        self.assertEqual(detector.calls, 2)

    def test_empty_frame_does_not_hide_new_subjects(self):
        detector = DetectorStub([])
        stream = GifDetectionStream(detector)
        stream.analyze(np.zeros((128, 128, 3), np.uint8))
        stream.analyze(make_frame(), elapsed_ms=20)
        self.assertEqual(detector.calls, 2)

    def test_featureless_motion_falls_back_to_detection(self):
        detector = DetectorStub()
        stream = GifDetectionStream(detector)
        stream.analyze(np.zeros((128, 128, 3), np.uint8))
        stream.analyze(np.full((128, 128, 3), 10, np.uint8), elapsed_ms=20)
        self.assertEqual(detector.calls, 2)

    def test_two_subjects_are_tracked_independently(self):
        base = make_frame()
        large = np.zeros((128, 256, 3), np.uint8)
        large[:, :128] = base
        large[:, 128:] = base
        moved = np.zeros_like(large)
        moved[:, :128] = make_frame(4, 2)
        moved[:, 128:] = make_frame(-3, 1)
        boxes = [
            DetectionBox(23, 28, 69, 74, 0.9, "Head", 0, "test"),
            DetectionBox(151, 28, 197, 74, 0.9, "Head", 0, "test"),
        ]
        detector = DetectorStub(boxes)
        stream = GifDetectionStream(detector)
        stream.analyze(large)
        result = stream.analyze(moved, elapsed_ms=20)
        self.assertEqual(detector.calls, 1)
        self.assertEqual(len(result.final_boxes), 2)
        self.assertAlmostEqual(result.final_boxes[0].x1, 27, delta=1)
        self.assertAlmostEqual(result.final_boxes[1].x1, 148, delta=1)

    def test_head_mode_tracking_preserves_head_box_without_body_parts(self):
        from test_detector import box, make_detector

        detector = make_detector(
            anime=[box((23, 28, 69, 74), "Head"), box((10, 70, 90, 125), "Torso")],
            head_only=True,
        )
        stream = GifDetectionStream(detector)
        first = stream.analyze(make_frame())
        tracked = stream.analyze(make_frame(3, 2), elapsed_ms=20)
        self.assertEqual([item.category for item in first.final_boxes], ["Head"])
        self.assertEqual([item.category for item in tracked.final_boxes], ["Head"])
        self.assertTrue(tracked.tracked_boxes)
        self.assertEqual(tracked.final_boxes[0].semantic_role, "anime_head")
        self.assertAlmostEqual(tracked.final_boxes[0].x1, 26, delta=1)
        self.assertEqual(detector._backend_plan.secondary_backend.calls, 1)
