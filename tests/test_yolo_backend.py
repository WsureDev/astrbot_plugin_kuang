import tempfile
import unittest

import numpy as np

from core.yolo_backend import Yolo26nDetectionBackend, AnimeYoloDetectionBackend


class YoloBackendPathTests(unittest.TestCase):
    def test_empty_model_path_is_rejected(self):
        with self.assertRaises(ValueError):
            Yolo26nDetectionBackend(model_path="")

    def test_directory_model_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = Yolo26nDetectionBackend(
                model_path=temp_dir,
                auto_download_model=False,
            )
            with self.assertRaises(RuntimeError):
                backend._ensure_model_file()


class YoloDecodeTests(unittest.TestCase):
    def decode(self, backend, output):
        return backend._decode_predictions(
            output, image_width=100, image_height=100, scale=1, pad_x=0, pad_y=0
        )

    def test_small_processed_outputs_are_not_transposed(self):
        backend = Yolo26nDetectionBackend(model_path="unused")
        for count in (0, 1, 2, 5, 6, 300):
            raw = np.tile(np.array([10, 20, 60, 90, 0.8, 0], np.float32), (1, count, 1))
            boxes = self.decode(backend, raw)
            self.assertEqual(len(boxes), count)
            for box in boxes:
                self.assertEqual(box.as_tuple(), (10, 20, 60, 90))

    def test_raw_anime_scores_are_filtered_before_box_creation(self):
        backend = AnimeYoloDetectionBackend(model_path="unused")
        raw = np.zeros((1, 7, 8400), np.float32)
        raw[0, :, 9] = [50, 50, 40, 60, 0.8, 0.1, 0.05]
        raw[0, :, 10] = [20, 20, 10, 10, 0.1, 0.2, 0.15]
        boxes = self.decode(backend, raw)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].category, "Head")
        self.assertEqual(boxes[0].as_tuple(), (30, 20, 70, 80))

    def test_invalid_values_are_rejected(self):
        backend = Yolo26nDetectionBackend(model_path="unused")
        raw = np.array(
            [
                [
                    [0, 0, 20, 20, 0.9, -1],
                    [0, 0, 20, 20, 0.9, 80],
                    [0, 0, 20, 20, float("nan"), 0],
                    [0, 0, 20, 20, 0.9, 0.5],
                ]
            ]
        )
        self.assertEqual(self.decode(backend, raw), [])

    def test_processed_detections_do_not_undergo_second_nms(self):
        backend = Yolo26nDetectionBackend(model_path="unused")
        backend._processed_output = True
        boxes = self.decode(
            backend, np.array([[[0, 0, 80, 80, 0.9, 0], [10, 10, 90, 90, 0.8, 0]]])
        )
        self.assertEqual(len(backend._select_recognized_boxes(boxes)), 2)


class YoloSessionConcurrencyTests(unittest.TestCase):
    def test_concurrent_first_use_initializes_one_complete_session(self):
        from concurrent.futures import ThreadPoolExecutor
        from types import SimpleNamespace
        from unittest.mock import Mock
        import time

        backend = Yolo26nDetectionBackend(model_path="unused")
        session = SimpleNamespace(
            get_inputs=lambda: [SimpleNamespace(name="images", shape=[1, 3, 640, 640])],
            get_outputs=lambda: [SimpleNamespace(shape=[1, 300, 6])],
        )
        backend._load_onnxruntime = lambda: None
        backend._ensure_model_file = lambda: "unused"
        def load(*args):
            time.sleep(.02)
            return session
        backend._create_session_with_recovery = Mock(side_effect=load)
        with ThreadPoolExecutor(max_workers=4) as pool:
            sessions = list(pool.map(lambda _: backend._get_session(), range(4)))
        self.assertTrue(all(item is session for item in sessions))
        self.assertEqual(backend._create_session_with_recovery.call_count, 1)
        self.assertEqual(backend._input_name, "images")
        self.assertTrue(backend._processed_output)


if __name__ == "__main__":
    unittest.main()
