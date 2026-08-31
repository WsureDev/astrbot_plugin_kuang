import unittest

import numpy as np

from core.motion_backend import MotionDetectionBackend


class MotionBackendTests(unittest.TestCase):
    def setUp(self):
        self.backend = MotionDetectionBackend(
            diff_threshold=25,
            min_area_ratio=0.001,
            downscale_long_edge=240,
            edge_padding_ratio=0,
            dilate_radius=0,
        )
        self.base = np.zeros((100, 160, 3), dtype=np.uint8)

    def test_missing_previous_frame_is_empty(self):
        self.assertEqual(self.backend.detect(self.base, None), [])

    def test_identical_frames_are_empty(self):
        boxes = self.backend.detect(self.base, self.base.copy())
        self.assertEqual(boxes, [])

    def test_changed_rectangle_maps_to_expected_box(self):
        current = self.base.copy()
        current[30:50, 40:60] = 200

        boxes = self.backend.detect(current, self.base)

        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].as_tuple(), (40, 30, 60, 50))
        self.assertEqual(boxes[0].semantic_role, "motion")

    def test_small_global_brightness_change_is_ignored(self):
        current = np.full_like(self.base, 10)
        boxes = self.backend.detect(current, self.base)
        self.assertEqual(boxes, [])

    def test_shape_mismatch_is_ignored(self):
        boxes = self.backend.detect(self.base[:, :-1], self.base)
        self.assertEqual(boxes, [])

    def test_translation_reports_changed_regions(self):
        previous = self.base.copy()
        previous[30:60, 50:80] = 200
        current = self.base.copy()
        current[30:60, 60:90] = 200

        boxes = self.backend.detect(current, previous)

        self.assertGreaterEqual(len(boxes), 1)
        self.assertTrue(
            any(box.x1 <= 50 and box.x2 >= 60 for box in boxes),
            boxes,
        )


if __name__ == "__main__":
    unittest.main()
