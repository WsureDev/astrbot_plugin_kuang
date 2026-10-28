import tempfile
import threading
import time
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


class TransformSaliencyBackend(FakeSaliencyBackend):
    def _infer(self, image_rgb):
        self._prepare_input(image_rgb)
        return self.raw_output


class BlockingSession:
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def run(self, output_names, feed):
        del output_names, feed
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.02)
        with self._lock:
            self.active -= 1
        return [np.zeros((1, 1, 4, 4), dtype=np.float32)]


class FakeSessionOptions:
    def __init__(self):
        self.graph_optimization_level = None
        self.intra_op_num_threads = 0
        self.inter_op_num_threads = 0


class FakeGraphOptimizationLevel:
    ORT_ENABLE_ALL = "all"


class FakeOrt:
    SessionOptions = FakeSessionOptions
    GraphOptimizationLevel = FakeGraphOptimizationLevel


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
        self.assertLess(float(normalized[0, 0]), float(normalized[1, 1]))

    def test_probability_mask_is_not_min_max_stretched(self):
        raw = np.asarray([[[[0.2, 0.4], [0.3, 0.45]]]], dtype=np.float32)
        backend = FakeSaliencyBackend(
            raw,
            binary_threshold=0.5,
            min_area_ratio=0,
            edge_padding_ratio=0,
            minmax_normalize_output=False,
        )

        self.assertEqual(backend.detect(np.zeros((4, 4, 3), dtype=np.uint8)), [])

    def test_full_frame_mask_is_rejected(self):
        backend = FakeSaliencyBackend(
            np.ones((1, 1, 4, 4), dtype=np.float32),
            min_area_ratio=0,
            edge_padding_ratio=0,
        )

        self.assertEqual(backend.detect(np.zeros((8, 8, 3), dtype=np.uint8)), [])

    def test_empty_mask_is_rejected(self):
        backend = FakeSaliencyBackend(
            np.empty((1, 1, 0, 0), dtype=np.float32),
            min_area_ratio=0,
            edge_padding_ratio=0,
        )

        self.assertEqual(backend.detect(np.zeros((8, 8, 3), dtype=np.uint8)), [])

    def test_malformed_one_dimensional_mask_is_rejected(self):
        backend = FakeSaliencyBackend(
            np.asarray([0.2, 0.8], dtype=np.float32),
            min_area_ratio=0,
            edge_padding_ratio=0,
        )

        self.assertEqual(backend.detect(np.zeros((8, 8, 3), dtype=np.uint8)), [])

    def test_aspect_ratio_restore_removes_letterbox(self):
        raw = np.zeros((1, 1, 4, 4), dtype=np.float32)
        raw[0, 0, 1:3, 1:3] = 1.0
        backend = TransformSaliencyBackend(
            raw,
            preserve_aspect_ratio=True,
            min_area_ratio=0,
            edge_padding_ratio=0,
        )
        backend._input_size = (4, 4)

        boxes = backend.detect(np.zeros((8, 16, 3), dtype=np.uint8))

        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].as_tuple(), (4, 0, 12, 8))

    def test_inference_calls_are_serialized(self):
        backend = SaliencyDetectionBackend(model_path="fake-saliency.onnx")
        backend._session = BlockingSession()
        backend._input_name = "input"
        backend._input_size = (4, 4)
        image = np.zeros((8, 8, 3), dtype=np.uint8)

        threads = [threading.Thread(target=backend._infer, args=(image,)) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(backend._session.max_active, 1)

    def test_minmax_normalization_remains_the_default_contract(self):
        backend = SaliencyDetectionBackend(model_path="fake-saliency.onnx")
        normalized = backend._normalize_mask(
            np.asarray([[[[0.2, 0.4], [0.3, 0.45]]]], dtype=np.float32)
        )

        self.assertAlmostEqual(float(normalized.min()), 0.0)
        self.assertAlmostEqual(float(normalized.max()), 1.0)

    def test_session_options_are_explicit_and_tunable(self):
        backend = SaliencyDetectionBackend(
            model_path="fake-saliency.onnx",
            intra_op_num_threads=2,
            inter_op_num_threads=3,
        )

        options = backend._create_session_options(FakeOrt)

        self.assertEqual(options.graph_optimization_level, "all")
        self.assertEqual(options.intra_op_num_threads, 2)
        self.assertEqual(options.inter_op_num_threads, 3)


if __name__ == "__main__":
    unittest.main()
