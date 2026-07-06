from __future__ import annotations

import os
import uuid

from .logger import get_logger
from .models import DetectionBox

_logger = get_logger(__name__)


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
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("缺少 Pillow 依赖，请先安装 requirements.txt。") from exc

        os.makedirs(output_dir, exist_ok=True)
        _logger.debug(
            "[renderer] render start: image=%s boxes=%s output_dir=%s",
            image_path,
            len(boxes),
            output_dir,
        )
        with Image.open(image_path) as image:
            base = image.convert("RGBA")

        composed = self.draw_boxes(base, boxes).convert("RGB")
        output_path = os.path.join(output_dir, f"kuang_{uuid.uuid4().hex}.png")
        composed.save(output_path, format="PNG")
        _logger.debug("[renderer] render saved: %s", output_path)
        return output_path

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

        _logger.debug(
            "[renderer] draw_boxes: canvas=%sx%s line_width=%s dynamic_width=%s boxes=%s",
            base.size[0],
            base.size[1],
            self.line_width,
            dynamic_width,
            len(boxes),
        )
        for box in boxes:
            x1, y1, x2, y2 = box.as_tuple()
            draw.rectangle((x1 + 1, y1 + 1, x2 + 1, y2 + 1), outline=shadow, width=1)
            draw.rectangle((x1, y1, x2, y2), outline=outline, width=dynamic_width)

        return Image.alpha_composite(base, overlay)

    def save_gif(
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

        os.makedirs(output_dir, exist_ok=True)
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
        _logger.debug(
            "[renderer] save_gif: frames=%s output=%s loop=%s",
            len(prepared_frames),
            output_path,
            loop,
        )
        return output_path
