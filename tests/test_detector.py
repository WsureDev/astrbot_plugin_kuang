import unittest
from types import SimpleNamespace
from unittest.mock import Mock

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


def make_detector(
    primary=(), anime=(), salient=(), policy="adaptive", head_only=False, box_count=5
):
    d = EspBoxDetector(
        model_path="unused",
        detection_backend=CountingBackend(primary),
        enable_saliency=True,
        saliency_model_path="unused",
        fallback_policy=policy,
        head_only=head_only,
        box_count=box_count,
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


class HeadLockTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.head = box((30, 5, 70, 35), "Head")
        self.torso = box((25, 30, 75, 70), "Torso")
        self.legs = box((25, 65, 75, 100), "Legs")

    def test_head_mode_bypasses_body_quality_count_and_all_policy_gates(self):
        for policy in ("adaptive", "all"):
            detector = make_detector(
                primary=[box((0, 0, 50, 100)), box((50, 0, 100, 100))],
                anime=[self.torso, self.head, self.legs],
                salient=[box((0, 0, 100, 100), "salient")],
                head_only=True,
                policy=policy,
            )
            detector.enable_motion = True
            detector._motion_backend = Mock()
            result = detector.analyze(self.image, previous_frame_rgb=self.image)
            self.assertEqual(
                [b.as_tuple() for b in result.final_boxes], [self.head.as_tuple()]
            )
            self.assertEqual([b.category for b in result.stage2_boxes], ["Head"])
            self.assertEqual(
                result.stage1_boxes
                + result.stage2_composite_boxes
                + result.stage3_boxes
                + result.stage4_boxes,
                [],
            )
            self.assertEqual(detector._backend_plan.primary_backend.calls, 0)
            self.assertEqual(detector._backend_plan.secondary_backend.calls, 1)
            self.assertEqual(detector._saliency_backend.calls, 0)
            detector._motion_backend.detect.assert_not_called()

    def test_no_head_never_falls_back_to_body_or_estimates_one(self):
        detector = make_detector(
            primary=[box((0, 0, 100, 100))],
            anime=[self.torso, self.legs],
            salient=[box((0, 0, 100, 100), "salient")],
            head_only=True,
        )
        self.assertEqual(detector.analyze(self.image).final_boxes, [])

    def test_body_parts_are_filtered_before_candidate_limit(self):
        detector = make_detector(anime=[self.torso] * 30 + [self.head], head_only=True)
        self.assertEqual(
            detector.analyze(self.image).final_boxes[0].as_tuple(), self.head.as_tuple()
        )

    def test_head_deduplication_limit_and_invalid_boxes(self):
        heads = [
            self.head,
            self.head.clone(score=0.7),
            box((0, 0, 20, 20), "Head", 0.8),
            box((80, 0, 100, 20), "Head", 0.7),
            box((0, 0, 0, 10), "Head"),
        ]
        detector = make_detector(anime=heads, head_only=True, box_count=2)
        result = detector.analyze(self.image)
        self.assertEqual(
            [b.as_tuple() for b in result.final_boxes],
            [self.head.as_tuple(), (0, 0, 20, 20)],
        )
        self.assertTrue(
            all(
                b.semantic_role == "anime_head" and not b.composite
                for b in result.final_boxes
            )
        )

    def test_anime_alone_uses_primary_as_head_model(self):
        detector = make_detector(head_only=True)
        detector._backend_plan.mode = "anime_only"
        detector._backend_plan.primary_backend = CountingBackend(
            [self.head, self.torso]
        )
        detector._backend_plan.secondary_backend = None
        self.assertEqual(
            detector.analyze(self.image).final_boxes[0].as_tuple(), self.head.as_tuple()
        )
        self.assertEqual(detector._backend_plan.primary_backend.calls, 1)

    def test_missing_or_failed_head_backend_never_returns_primary_body(self):
        detector = make_detector(primary=[box((0, 0, 100, 100))], head_only=True)
        detector._backend_plan.mode = "single_backend"
        self.assertEqual(detector.analyze(self.image).final_boxes, [])
        detector._backend_plan.mode = "yolo_with_anime_fallback"
        detector._backend_plan.secondary_backend = Mock()
        detector._backend_plan.secondary_backend.detect.side_effect = RuntimeError(
            "head model failed"
        )
        self.assertEqual(detector.analyze(self.image).final_boxes, [])

    def test_turning_head_mode_off_restores_body_composition(self):
        detector = make_detector(
            anime=[self.head, self.torso, self.legs], head_only=True
        )
        head = detector.analyze(self.image)
        detector.head_only = False
        body = detector.analyze(self.image)
        expected = make_detector(anime=[self.head, self.torso, self.legs]).analyze(
            self.image
        )
        self.assertEqual(head.final_boxes[0].as_tuple(), self.head.as_tuple())
        self.assertEqual(body.final_boxes, expected.final_boxes)
        self.assertTrue(body.final_boxes[0].composite)
        self.assertGreater(body.final_boxes[0].area, head.final_boxes[0].area)
