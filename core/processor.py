from __future__ import annotations

from pathlib import Path


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

        with Image.open(image_path) as image:
            if image.format == "GIF" and getattr(image, "is_animated", False):
                return self._render_gif(image_path)

        boxes = self.detector.detect_from_path(image_path)
        return self.renderer.render(image_path, boxes, str(self.output_dir))

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

            for frame in ImageSequence.Iterator(image):
                duration = int(frame.info.get("duration", base_duration) or base_duration)
                rgba_frame = frame.convert("RGBA")
                rgb_frame = rgba_frame.convert("RGB")
                current_boxes = self.detector.detect(np.asarray(rgb_frame))
                rendered_frames.append(self.renderer.draw_boxes(rgba_frame, current_boxes))
                durations.append(max(20, duration))

        if not rendered_frames:
            raise RuntimeError("GIF 中没有可处理的帧。")

        return self.renderer.save_gif(
            frames=rendered_frames,
            durations=durations,
            loop=loop,
            output_dir=str(self.output_dir),
        )
