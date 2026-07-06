from __future__ import annotations

from pathlib import Path

from .logger import get_logger

_logger = get_logger(__name__)


class EspBoxProcessor:
    def __init__(self, *, detector, renderer, output_dir: str | Path) -> None:
        self.detector = detector
        self.renderer = renderer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def render_path(self, image_path: str) -> str:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("缺少 Pillow 依赖，请先安装 requirements.txt。") from exc

        _logger.debug("[processor] render_path start: %s", image_path)
        with Image.open(image_path) as image:
            _logger.debug(
                "[processor] opened image: path=%s format=%s size=%sx%s animated=%s",
                image_path,
                image.format,
                image.width,
                image.height,
                bool(getattr(image, "is_animated", False)),
            )
            if image.format == "GIF" and getattr(image, "is_animated", False):
                _logger.info("[processor] GIF detected, switching to frame-by-frame render")
                return self._render_gif(image_path)

        boxes = self.detector.detect_from_path(image_path)
        _logger.debug("[processor] detector returned %s boxes", len(boxes))
        output_path = self.renderer.render(image_path, boxes, str(self.output_dir))
        _logger.debug("[processor] render_path done: output=%s", output_path)
        return output_path

    def _render_gif(self, image_path: str) -> str:
        try:
            import numpy as np
            from PIL import Image, ImageSequence
        except ImportError as exc:
            raise RuntimeError("缺少图片处理依赖，请先安装 requirements.txt。") from exc

        rendered_frames = []
        durations: list[int] = []

        with Image.open(image_path) as image:
            base_duration = int(image.info.get("duration", 100) or 100)
            loop = int(image.info.get("loop", 0) or 0)

            frame_index = 0
            for frame in ImageSequence.Iterator(image):
                frame_index += 1
                duration = int(frame.info.get("duration", base_duration) or base_duration)
                rgba_frame = frame.convert("RGBA")
                rgb_frame = rgba_frame.convert("RGB")
                current_boxes = self.detector.detect(np.asarray(rgb_frame))
                _logger.debug(
                    "[processor] GIF frame=%s duration=%sms boxes=%s",
                    frame_index,
                    duration,
                    len(current_boxes),
                )
                rendered_frames.append(self.renderer.draw_boxes(rgba_frame, current_boxes))
                durations.append(max(20, duration))

        if not rendered_frames:
            raise RuntimeError("GIF 中没有可处理的帧。")

        output_path = self.renderer.save_gif(
            frames=rendered_frames,
            durations=durations,
            loop=loop,
            output_dir=str(self.output_dir),
        )
        _logger.debug(
            "[processor] GIF render done: frames=%s output=%s",
            len(rendered_frames),
            output_path,
        )
        return output_path
