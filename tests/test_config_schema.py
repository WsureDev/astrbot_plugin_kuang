import json
import re
import unittest
from pathlib import Path


class ConfigSchemaTests(unittest.TestCase):
    def test_stage3_and_stage4_options_are_wired(self):
        root = Path(__file__).resolve().parents[1]
        schema = json.loads((root / "_conf_schema.json").read_text())
        main_text = (root / "main.py").read_text()
        expected = {
            "enable_saliency",
            "saliency_model_path",
            "saliency_model_url",
            "saliency_auto_download_model",
            "saliency_model_input_size",
            "saliency_threshold",
            "saliency_min_area_ratio",
            "saliency_max_boxes",
            "saliency_max_foreground_ratio",
            "saliency_preserve_aspect_ratio",
            "saliency_intra_op_num_threads",
            "saliency_inter_op_num_threads",
            "saliency_minmax_normalize_output",
            "fallback_policy",
            "inference_num_threads",
            "max_concurrent_images",
            "gif_detection_interval",
            "gif_max_reuse_ms",
            "enable_motion",
            "motion_diff_threshold",
            "motion_min_area_ratio",
            "motion_max_boxes",
            "motion_downscale_long_edge",
            "motion_dilate_radius",
        }

        self.assertTrue(expected.issubset(schema))
        self.assertTrue(schema["enable_saliency"]["default"])
        self.assertFalse(schema["enable_motion"]["default"])
        self.assertFalse(schema["saliency_preserve_aspect_ratio"]["default"])
        self.assertEqual(schema["saliency_max_foreground_ratio"]["default"], 0.995)
        self.assertEqual(schema["saliency_intra_op_num_threads"]["default"], 0)
        self.assertTrue(schema["saliency_minmax_normalize_output"]["default"])
        for key in expected:
            self.assertRegex(main_text, rf'config\.get\("{re.escape(key)}"')


if __name__ == "__main__":
    unittest.main()
