import tempfile
import unittest

import numpy as np

from core.saliency_backend import SaliencyDetectionBackend


class FakeSaliencyBackend(SaliencyDetectionBackend):
    def __init__(self, raw_output, **kwargs):
        super().__init__(model_path="fake-saliency.onnx", **kwargs)
        self.raw_output = raw_output

    def _infer(self, image_rgb):
        del image_rgb
        return self.raw_output


class SaliencyBackendTests(unittest.TestCase):
    def test_empty_model_path_is_rejected(self):
        with self.assertRaises(ValueError):
            SaliencyDetectionBackend(model_path="")

    def test_directory_model_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = SaliencyDetectionBackend(
                model_path=temp_dir,
                auto_download_model=False,
            )
            with self.assertRaises(RuntimeError):
                backend._ensure_model_file()

    def test_zero_mask_produces_no_boxes(self):
        backend = FakeSaliencyBackend(
            np.zeros((1, 1, 4, 4), dtype=np.float32),
            edge_padding_ratio=0,
        )
        image = np.zeros((8, 8, 3), dtype=np.uint8)

        self.assertEqual(backend.detect(image), [])

    def test_connected_component_maps_back_to_image_coordinates(self):
        raw = np.zeros((1, 1, 4, 4), dtype=np.float32)
        raw[0, 0, 1:3, 1:3] = 1.0
        backend = FakeSaliencyBackend(raw, edge_padding_ratio=0)
        image = np.zeros((8, 8, 3), dtype=np.uint8)

        boxes = backend.detect(image)

        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].as_tuple(), (2, 2, 6, 6))
        self.assertEqual(boxes[0].semantic_role, "salient")

    def test_logits_are_normalized_to_probability_range(self):
        backend = SaliencyDetectionBackend(model_path="fake-saliency.onnx")
        normalized = backend._normalize_mask(
            np.asarray([[[[-2.0, 0.0], [1.0, 3.0]]]], dtype=np.float32)
        )

        self.assertGreaterEqual(float(normalized.min()), 0.0)
        self.assertLessEqual(float(normalized.max()), 1.0)
        self.assertAlmostEqual(float(normalized.min()), 0.0)
        self.assertAlmostEqual(float(normalized.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
