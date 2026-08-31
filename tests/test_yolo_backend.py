import tempfile
import unittest

from core.yolo_backend import Yolo26nDetectionBackend


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


if __name__ == "__main__":
    unittest.main()
