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
            "enable_motion",
            "motion_diff_threshold",
            "motion_min_area_ratio",
            "motion_max_boxes",
            "motion_downscale_long_edge",
            "motion_dilate_radius",
        }

        self.assertTrue(expected.issubset(schema))
        self.assertFalse(schema["enable_saliency"]["default"])
        self.assertFalse(schema["enable_motion"]["default"])
        for key in expected:
            self.assertRegex(main_text, rf'config\.get\("{re.escape(key)}"')


if __name__ == "__main__":
    unittest.main()
