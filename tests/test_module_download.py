import io
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from core.module_runtime import download_model


class Response(io.BytesIO):
    def __init__(self, payload, size=None):
        super().__init__(payload)
        self.headers = {"Content-Length": str(size)} if size is not None else {}


class ModuleDownloadTests(unittest.TestCase):
    def test_unknown_length_reports_bytes_without_inventing_percentage(self):
        self.check_download(b"abc", None, 3, None)

    def test_known_length_reports_exact_bytes(self):
        self.check_download(b"abcdef", 6, 6, 6)

    def check_download(self, payload, length, received, total):
        with (
            tempfile.TemporaryDirectory() as root,
            patch(
                "core.module_runtime.urlopen", return_value=Response(payload, length)
            ),
        ):
            progress = []
            path = Path(root) / "model.part"
            download_model(
                "https://example.test/model",
                path,
                lambda *args: progress.append(args),
                threading.Event(),
            )
            self.assertEqual(progress[-1][:2], (received, total))
            self.assertEqual(path.read_bytes(), payload)
            self.assertIsInstance(progress[-1][2], int)

    def test_short_response_fails(self):
        with (
            tempfile.TemporaryDirectory() as root,
            patch("core.module_runtime.urlopen", return_value=Response(b"abc", 100)),
        ):
            with self.assertRaisesRegex(RuntimeError, "不完整"):
                download_model(
                    "https://example.test/model",
                    Path(root) / "model.part",
                    lambda *args: None,
                    threading.Event(),
                )

    def test_reload_stop_signal_interrupts_download(self):
        stopped = threading.Event()
        stopped.set()
        with (
            tempfile.TemporaryDirectory() as root,
            patch("core.module_runtime.urlopen", return_value=Response(b"abc", 3)),
        ):
            with self.assertRaises(InterruptedError):
                download_model(
                    "https://example.test/model",
                    Path(root) / "model.part",
                    lambda *args: None,
                    stopped,
                )
