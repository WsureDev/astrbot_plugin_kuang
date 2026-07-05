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

        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        dynamic_width = max(
            self.line_width,
            int(round(min(base.size) * 0.0022)),
        )
        outline = (255, 255, 255, self.line_alpha)

        for box in boxes:
            x1, y1, x2, y2 = box.as_tuple()
            draw.rectangle((x1, y1, x2, y2), outline=outline, width=dynamic_width)

        composed = Image.alpha_composite(base, overlay).convert("RGB")
        output_path = os.path.join(output_dir, f"kuang_{uuid.uuid4().hex}.png")
        composed.save(output_path, format="PNG")
        return output_path
