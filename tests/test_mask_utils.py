import builtins
import unittest
from unittest.mock import patch

import numpy as np

from core.mask_utils import _scanline_components, connected_components


class MaskComponentsTests(unittest.TestCase):
    def test_four_connectivity_and_merging_runs(self):
        diagonal = np.eye(5, dtype=np.uint8)
        self.assertEqual(
            sorted(_scanline_components(diagonal)), [(i, i, i, i, 1) for i in range(5)]
        )
        bridge = np.array([[1, 0, 1, 0, 1], [1, 1, 1, 1, 1]], dtype=np.uint8)
        self.assertEqual(_scanline_components(bridge), [(0, 0, 4, 1, 8)])

    def test_matches_opencv_on_exhaustive_small_masks(self):
        for bits in range(512):
            mask = np.array(
                [(bits >> i) & 1 for i in range(9)], dtype=np.uint8
            ).reshape(3, 3)
            self.assertEqual(
                sorted(_scanline_components(mask)), sorted(connected_components(mask))
            )

    def test_foreground_path_runs_without_opencv(self):
        original = builtins.__import__

        def without_cv(name, *args, **kwargs):
            if name == "cv2":
                raise ImportError("OpenCV not installed")
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=without_cv):
            self.assertEqual(
                connected_components(np.ones((320, 320), dtype=np.uint8)),
                [(0, 0, 319, 319, 102400)],
            )
