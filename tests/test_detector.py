import unittest
from types import SimpleNamespace

import numpy as np

from core.detector import EspBoxDetector
from core.models import DetectionBox


class CountingBackend:
    def __init__(self, boxes=()):
        self.boxes = list(boxes)
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return list(self.boxes)


def box(coords, category="person", score=0.9):
    return DetectionBox(*coords, score, category, 0, "test")


def make_detector(primary=(), anime=(), salient=(), policy="adaptive"):
    d = EspBoxDetector(
        model_path="unused",
        detection_backend=CountingBackend(primary),
        enable_saliency=True,
        saliency_model_path="unused",
        fallback_policy=policy,
    )
    d._backend_plan = SimpleNamespace(
        mode="yolo_with_anime_fallback",
        primary_backend=CountingBackend(primary),
        secondary_backend=CountingBackend(anime),
        anime_fallback_trigger_count=2,
        ignore_secondary_errors=True,
    )
    d._saliency_backend = CountingBackend(salient)
    return d


class DetectorRoutingTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_empty_detection_stays_empty(self):
        self.assertEqual(make_detector().analyze(self.image).final_boxes, [])

    def test_reliable_person_skips_both_fallback_models(self):
        d = make_detector([box((10, 10, 80, 90))])
        result = d.analyze(self.image)
        self.assertEqual(len(result.final_boxes), 1)
        self.assertEqual(d._backend_plan.secondary_backend.calls, 0)
        self.assertEqual(d._saliency_backend.calls, 0)

    def test_closeup_head_is_a_complete_subject(self):
        d = make_detector(anime=[box((0, 0, 95, 95), "Head")])
        result = d.analyze(self.image)
        self.assertEqual(result.final_boxes[0].category, "Head")
        self.assertEqual(result.final_boxes[0].as_tuple(), (0, 0, 95, 95))
        self.assertEqual(d._saliency_backend.calls, 0)

    def test_small_head_does_not_prevent_foreground_completion(self):
        foreground = box((10, 0, 90, 100), "salient", 0.7).clone(
            stage="stage3_salient", semantic_role="salient"
        )
        d = make_detector(anime=[box((30, 10, 70, 50), "Head")], salient=[foreground])
        result = d.analyze(self.image)
        self.assertEqual(d._saliency_backend.calls, 1)
        self.assertEqual([b.as_tuple() for b in result.final_boxes], [(10, 0, 90, 100)])

    def test_all_policy_bypasses_quality_and_count_gates(self):
        d = make_detector(
            primary=[box((0, 0, 40, 100)), box((50, 0, 100, 100))], policy="all"
        )
        d.analyze(self.image)
        self.assertEqual(d._backend_plan.secondary_backend.calls, 1)
        self.assertEqual(d._saliency_backend.calls, 1)

    def test_separate_calls_have_no_frame_cache(self):
        d = make_detector()
        d.analyze(self.image)
        d.analyze(self.image.copy(), previous_frame_rgb=self.image)
        self.assertEqual(d._saliency_backend.calls, 2)
