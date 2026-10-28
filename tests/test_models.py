import unittest

from core.models import DetectionBox, DetectionPipelineResult


class DetectionModelTests(unittest.TestCase):
    def test_pipeline_result_default_lists_are_independent(self):
        result = DetectionPipelineResult([], [], [], [], [])
        another = DetectionPipelineResult([], [], [], [], [])

        self.assertEqual(result.stage3_boxes, [])
        self.assertEqual(result.stage4_boxes, [])
        self.assertIsNot(result.stage3_boxes, another.stage3_boxes)
        self.assertIsNot(result.stage4_boxes, another.stage4_boxes)

    def test_box_geometry(self):
        left = DetectionBox(0, 0, 10, 10, 0.9, "person", 0, "test")
        right = DetectionBox(5, 5, 15, 15, 0.8, "object", 1, "test")

        self.assertEqual(left.intersection_area(right), 25)
        self.assertAlmostEqual(left.iou(right), 25 / 175)
        self.assertAlmostEqual(left.containment_ratio(right), 0.25)


if __name__ == "__main__":
    unittest.main()
