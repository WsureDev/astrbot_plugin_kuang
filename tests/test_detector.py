import unittest

import numpy as np

from core.detector import EspBoxDetector
from core.models import DetectionBox


class EmptyPrimaryBackend:
    def load_numpy(self):
        return np

    def detect(self, image_rgb):
        del image_rgb
        return []


class CountingSaliencyBackend:
    def __init__(self):
        self.calls = 0

    def detect(self, image_rgb):
        self.calls += 1
        height, width = image_rgb.shape[:2]
        return [
            DetectionBox(
                x1=0,
                y1=0,
                x2=max(1, width // 2),
                y2=max(1, height // 2),
                score=0.8,
                category="salient",
                priority=0,
                source="saliency",
                stage="stage3_salient",
                part="salient",
                semantic_role="salient",
            )
        ]


class EmptySaliencyBackend(CountingSaliencyBackend):
    def detect(self, image_rgb):
        del image_rgb
        self.calls += 1
        return []


class MotionStub:
    def __init__(self):
        self.calls = 0

    def detect(self, current, previous):
        self.calls += 1
        if np.array_equal(current, previous):
            return []
        return [
            DetectionBox(
                x1=2,
                y1=2,
                x2=6,
                y2=6,
                score=0.5,
                category="motion",
                priority=0,
                source="motion",
                stage="stage4_motion",
                part="motion",
                semantic_role="motion",
            )
        ]


class DetectorSaliencyReuseTests(unittest.TestCase):
    def make_detector(self):
        detector = EspBoxDetector(
            box_count=2,
            model_path="unused.onnx",
            detection_backend=EmptyPrimaryBackend(),
            enable_saliency=True,
            saliency_model_path="unused-saliency.onnx",
            enable_motion=True,
            enable_random_boxes=False,
        )
        saliency = CountingSaliencyBackend()
        motion = MotionStub()
        detector._saliency_backend = saliency
        detector._motion_backend = motion
        return detector, saliency, motion

    def make_non_reusing_detector(self):
        detector, saliency, motion = self.make_detector()
        detector.saliency_reuse_static_frames = False
        return detector, saliency, motion

    def test_identical_gif_frame_reuses_saliency_result(self):
        detector, saliency, motion = self.make_detector()
        first = np.zeros((8, 8, 3), dtype=np.uint8)
        second = first.copy()

        detector.analyze(first)
        detector.analyze(second, previous_frame_rgb=first)

        self.assertEqual(saliency.calls, 1)
        self.assertEqual(motion.calls, 1)

    def test_motion_invalidates_saliency_cache(self):
        detector, saliency, motion = self.make_detector()
        first = np.zeros((8, 8, 3), dtype=np.uint8)
        changed = first.copy()
        changed[3:5, 3:5] = 255

        detector.analyze(first)
        detector.analyze(changed, previous_frame_rgb=first)

        self.assertEqual(saliency.calls, 2)
        self.assertEqual(motion.calls, 1)

    def test_cache_does_not_cross_concurrent_streams(self):
        detector, saliency, _ = self.make_detector()
        stream_a = np.zeros((8, 8, 3), dtype=np.uint8)
        stream_b = np.full((8, 8, 3), 64, dtype=np.uint8)

        detector.analyze(stream_a)
        detector.analyze(stream_b)
        detector.analyze(stream_a.copy(), previous_frame_rgb=stream_a)

        self.assertEqual(saliency.calls, 3)

    def test_static_frame_reuse_can_be_disabled(self):
        detector, saliency, _ = self.make_non_reusing_detector()
        first = np.zeros((8, 8, 3), dtype=np.uint8)
        second = first.copy()

        detector.analyze(first)
        detector.analyze(second, previous_frame_rgb=first)

        self.assertEqual(saliency.calls, 2)

    def test_empty_saliency_result_is_cached(self):
        detector, _, motion = self.make_detector()
        empty_saliency = EmptySaliencyBackend()
        detector._saliency_backend = empty_saliency
        first = np.zeros((8, 8, 3), dtype=np.uint8)
        second = first.copy()

        detector.analyze(first)
        detector.analyze(second, previous_frame_rgb=first)

        self.assertEqual(empty_saliency.calls, 1)
        self.assertEqual(motion.calls, 1)


if __name__ == "__main__":
    unittest.main()
