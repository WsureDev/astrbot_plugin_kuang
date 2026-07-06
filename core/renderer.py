from __future__ import annotations

import os
import uuid

from .models import DetectionBox


class EspBoxRenderer:
    def __init__(self, *, line_width: int = 3, line_alpha: int = 220) -> None:
        self.line_width = max(1, int(line_width))
        self.line_alpha = max(0, min(255, int(line_alpha)))

    def render(
        self,
        image_path: str,
        boxes: list[DetectionBox],
        output_dir: str,
    ) -> str:
        try:
            from PIL import Image, ImageDraw
        except ImportError as exc:
            raise RuntimeError("缺少 Pillow 依赖，请先安装 requirements.txt。") from exc

        os.makedirs(output_dir, exist_ok=True)
        with Image.open(image_path) as image:
            base = image.convert("RGBA")

        composed = self.draw_boxes(base, boxes).convert("RGB")
        output_path = os.path.join(output_dir, f"kuang_{uuid.uuid4().hex}.png")
        composed.save(output_path, format="PNG")
        return output_path

    def render_gif(
        self,
        image_path: str,
        detector,
        output_dir: str,
    ) -> str:
        try:
            from PIL import Image, ImageSequence
        except ImportError as exc:
            raise RuntimeError("缺少 Pillow 依赖，请先安装 requirements.txt。") from exc

        os.makedirs(output_dir, exist_ok=True)

        rendered_frames = []
        durations: list[int] = []

        with Image.open(image_path) as image:
            base_duration = int(image.info.get("duration", 100) or 100)
            loop = int(image.info.get("loop", 0) or 0)

            for frame in ImageSequence.Iterator(image):
                duration = int(frame.info.get("duration", base_duration) or base_duration)
                rgba_frame = frame.convert("RGBA")
                rgb_frame = rgba_frame.convert("RGB")
                current_boxes = detector.detect(detector._load_numpy().array(rgb_frame))
                rendered_frames.append(self.draw_boxes(rgba_frame, current_boxes))
                durations.append(max(20, duration))

        if not rendered_frames:
            raise RuntimeError("GIF 中没有可处理的帧。")

        return self._save_gif(
            frames=rendered_frames,
            durations=durations,
            loop=loop,
            output_dir=output_dir,
        )

    def draw_boxes(self, base, boxes: list[DetectionBox]):
        from PIL import Image, ImageDraw

        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        dynamic_width = max(
            self.line_width,
            int(round(min(base.size) * 0.0022)),
        )
        outline = (255, 255, 255, self.line_alpha)
        shadow = (0, 0, 0, 255)

        for box in boxes:
            x1, y1, x2, y2 = box.as_tuple()
            draw.rectangle((x1 + 1, y1 + 1, x2 + 1, y2 + 1), outline=shadow, width=1)
            draw.rectangle((x1, y1, x2, y2), outline=outline, width=dynamic_width)

        return Image.alpha_composite(base, overlay)

    def _save_gif(
        self,
        *,
        frames,
        durations: list[int],
        loop: int,
        output_dir: str,
    ) -> str:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("缺少 Pillow 依赖，请先安装 requirements.txt。") from exc

        prepared_frames = [
            frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
            for frame in frames
        ]
        output_path = os.path.join(
            output_dir,
            f"kuang_{uuid.uuid4().hex}.gif",
        )
        prepared_frames[0].save(
            output_path,
            format="GIF",
            save_all=True,
            append_images=prepared_frames[1:],
            duration=durations,
            loop=loop,
            disposal=2,
            optimize=True,
        )
        return output_path
