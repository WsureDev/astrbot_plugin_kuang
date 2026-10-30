import tempfile
import unittest
from pathlib import Path

from PIL import Image

from core.models import DetectionPipelineResult
from core.processor import EspBoxProcessor


class FakeRenderer:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.render_calls = []
        self.draw_calls = []
        self.gif_call = None

    def render(self, image_path, boxes, output_dir):
        self.render_calls.append((image_path, boxes, output_dir))
        return str(self.output_path)

    def draw_boxes(self, base, boxes):
        self.draw_calls.append((base.size, boxes))
        return base.copy()

    def save_gif(self, *, frames, durations, loop, output_dir):
        self.gif_call = (frames, durations, loop, output_dir)
        return str(self.output_path)


class OldResult:
    def __init__(self):
        self.stage1_boxes = []
        self.stage2_boxes = []
        self.stage2_composite_boxes = []
        self.arranged_boxes = []
        self.final_boxes = []


class OldDetector:
    def __init__(self):
        self.calls = []

    def analyze_from_path(self, image_path):
        self.calls.append(("path", image_path))
        return OldResult()

    def analyze(self, image):
        self.calls.append(("frame", image.shape))
        return OldResult()


class TemporalDetector:
    def __init__(self):
        self.previous_frames = []

    def analyze(self, image, *, previous_frame_rgb=None):
        self.previous_frames.append(previous_frame_rgb)
        del image
        return DetectionPipelineResult([], [], [], [], [])


class ProcessorCompatibilityTests(unittest.TestCase):
    def make_png(self, directory: Path) -> Path:
        path = directory / "input.png"
        Image.new("RGB", (8, 8), "red").save(path)
        return path

    def make_gif(self, directory: Path) -> Path:
        path = directory / "input.gif"
        first = Image.new("RGB", (8, 8), "red")
        second = Image.new("RGB", (8, 8), "blue")
        first.save(
            path,
            save_all=True,
            append_images=[second],
            duration=[40, 60],
            loop=0,
        )
        return path

    def test_static_path_accepts_legacy_result_shape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            detector = OldDetector()
            renderer = FakeRenderer(root / "output.png")
            processor = EspBoxProcessor(
                detector=detector,
                renderer=renderer,
                output_dir=root / "out",
            )

            output = processor.render_path(str(self.make_png(root)))

            self.assertEqual(output, str(root / "output.png"))
            self.assertEqual(len(renderer.render_calls), 1)

    def test_gif_uses_legacy_detector_without_temporal_keyword(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            detector = OldDetector()
            renderer = FakeRenderer(root / "output.gif")
            processor = EspBoxProcessor(
                detector=detector,
                renderer=renderer,
                output_dir=root / "out",
            )

            output = processor.render_path(str(self.make_gif(root)))

            self.assertEqual(output, str(root / "output.gif"))
            self.assertEqual([kind for kind, _ in detector.calls], ["frame", "frame"])
            self.assertEqual(renderer.gif_call[1], [40, 60])

    def test_gif_passes_previous_frame_to_current_detector_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            detector = TemporalDetector()
            renderer = FakeRenderer(root / "output.gif")
            processor = EspBoxProcessor(
                detector=detector,
                renderer=renderer,
                output_dir=root / "out",
            )

            processor.render_path(str(self.make_gif(root)))

            self.assertEqual(len(detector.previous_frames), 2)
            self.assertIsNone(detector.previous_frames[0])
            self.assertIsNotNone(detector.previous_frames[1])
            self.assertEqual(renderer.gif_call[1], [40, 60])

    def test_animated_webp_preserves_frames_durations_and_loop_as_gif(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "animation.bin"
            Image.new("RGB", (12, 10), "red").save(
                path,
                format="WEBP",
                save_all=True,
                append_images=[Image.new("RGB", (12, 10), "blue")],
                duration=[40, 90],
                loop=3,
                lossless=True,
            )
            detector = OldDetector()
            renderer = FakeRenderer(root / "output.gif")
            processor = EspBoxProcessor(
                detector=detector, renderer=renderer, output_dir=root / "out"
            )
            self.assertTrue(processor.is_supported_animation(str(path)))
            processor.render_path(str(path))
            self.assertEqual([kind for kind, _ in detector.calls], ["frame", "frame"])
            self.assertEqual(renderer.gif_call[1], [40, 90])
            self.assertEqual(renderer.gif_call[2], 3)

    def test_static_webp_and_single_frame_gif_are_not_automatic_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for format in ("PNG", "WEBP", "GIF"):
                path = root / format
                Image.new("RGB", (8, 8), "red").save(path, format=format)
                self.assertFalse(EspBoxProcessor.is_supported_animation(str(path)))


if __name__ == "__main__":
    unittest.main()
