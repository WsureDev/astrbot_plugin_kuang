from __future__ import annotations

import os
import uuid
from pathlib import Path

from .models import DetectionBox


class EspBoxRenderer:
    GIF_DETECT_INTERVAL_MS = 500

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

        composed = self._draw_boxes(base, boxes).convert("RGB")
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
        input_path = Path(image_path)
        input_size = input_path.stat().st_size

        rendered_frames = []
        durations: list[int] = []
        cumulative_ms = 0
        next_detect_at_ms = 0
        current_boxes: list[DetectionBox] = []

        with Image.open(image_path) as image:
            base_duration = int(image.info.get("duration", 100) or 100)
            loop = int(image.info.get("loop", 0) or 0)

            for index, frame in enumerate(ImageSequence.Iterator(image)):
                duration = int(frame.info.get("duration", base_duration) or base_duration)
                rgba_frame = frame.convert("RGBA")

                if index == 0 or cumulative_ms >= next_detect_at_ms:
                    rgb_frame = rgba_frame.convert("RGB")
                    current_boxes = detector.detect(detector._load_numpy().array(rgb_frame))
                    next_detect_at_ms = cumulative_ms + self.GIF_DETECT_INTERVAL_MS

                rendered_frames.append(self._draw_boxes(rgba_frame, current_boxes))
                durations.append(max(20, duration))
                cumulative_ms += max(20, duration)

        if not rendered_frames:
            raise RuntimeError("GIF 中没有可处理的帧。")

        return self._save_gif_with_size_cap(
            frames=rendered_frames,
            durations=durations,
            loop=loop,
            input_size=input_size,
            output_dir=output_dir,
        )

    def _draw_boxes(self, base, boxes: list[DetectionBox]):
        from PIL import Image, ImageDraw

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

        return Image.alpha_composite(base, overlay)

    def _save_gif_with_size_cap(
        self,
        *,
        frames,
        durations: list[int],
        loop: int,
        input_size: int,
        output_dir: str,
    ) -> str:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("缺少 Pillow 依赖，请先安装 requirements.txt。") from exc

        color_candidates = [256, 192, 160, 128, 96, 64, 48, 32, 24, 16, 12, 8, 6, 4, 2]
        best_path: str | None = None
        best_size: int | None = None

        for colors in color_candidates:
            prepared_frames = [
                frame.quantize(
                    colors=colors,
                    method=Image.Quantize.FASTOCTREE,
                    dither=Image.Dither.NONE,
                )
                for frame in frames
            ]
            candidate_path = os.path.join(
                output_dir,
                f"kuang_{uuid.uuid4().hex}_{colors}.gif",
            )
            prepared_frames[0].save(
                candidate_path,
                format="GIF",
                save_all=True,
                append_images=prepared_frames[1:],
                duration=durations,
                loop=loop,
                disposal=2,
                optimize=True,
            )
            candidate_size = Path(candidate_path).stat().st_size

            if best_size is None or candidate_size < best_size:
                if best_path and os.path.exists(best_path):
                    os.remove(best_path)
                best_path = candidate_path
                best_size = candidate_size
            else:
                if candidate_size <= input_size:
                    os.remove(candidate_path)
                    return best_path
                os.remove(candidate_path)

            if best_size is not None and best_size <= input_size:
                return best_path

        if best_path is None:
            raise RuntimeError("GIF 保存失败。")
        return best_path
