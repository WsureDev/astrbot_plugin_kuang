import unittest

from core.models import DetectionBox, DetectionPipelineResult
from examples.benchmark_pipeline import quality_metrics


class QualityMetricTests(unittest.TestCase):
    def result(self, coordinates):
        boxes = [DetectionBox(*c, 1, "subject", 0, "test") for c in coordinates]
        return DetectionPipelineResult([], [], [], boxes, boxes)

    def test_missing_annotations_do_not_create_accuracy(self):
        metrics = quality_metrics([self.result([(0, 0, 10, 10)])], {})
        self.assertEqual(metrics["annotated_frames"], 0)
        self.assertIsNone(metrics["recall_iou50"])
        self.assertIsNone(metrics["precision_iou50"])

    def test_missed_and_false_subjects_are_counted(self):
        results = [
            self.result([(0, 0, 10, 10), (30, 30, 40, 40)]),
            self.result([(0, 0, 5, 5)]),
        ]
        metrics = quality_metrics(
            results, {"0": [[0, 0, 10, 10], [60, 60, 70, 70]], "1": []}
        )
        self.assertEqual(metrics["matched_targets"], 1)
        self.assertEqual(metrics["recall_iou50"], 0.5)
        self.assertAlmostEqual(metrics["precision_iou50"], 1 / 3)

    def test_a_prediction_cannot_match_two_subjects(self):
        metrics = quality_metrics(
            [self.result([(0, 0, 10, 10)])], {"0": [[0, 0, 10, 10], [1, 0, 11, 10]]}
        )
        self.assertEqual(metrics["matched_targets"], 1)
