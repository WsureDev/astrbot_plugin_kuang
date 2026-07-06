from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import EspBoxDetector, EspBoxRenderer


def build_background(scene: dict, size: int):
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (size, size), (0, 0, 0, 255))
    draw = ImageDraw.Draw(image)
    horizon_y = int(scene["horizon_y"])
    ground_bottom_y = int(scene["ground_bottom_y"])
    vanishing_x = float(scene["vanishing_x"])

    for y in range(horizon_y):
        t = y / max(1, horizon_y)
        color = (
            int(28 + (38 * (1.0 - t))),
            int(42 + (52 * (1.0 - t))),
            int(52 + (68 * (1.0 - t))),
            255,
        )
        draw.line((0, y, size, y), fill=color)

    for y in range(horizon_y, ground_bottom_y + 1):
        t = (y - horizon_y) / max(1, ground_bottom_y - horizon_y)
        color = (
            int(30 + (24 * t)),
            int(34 + (28 * t)),
            int(36 + (30 * t)),
            255,
        )
        draw.line((0, y, size, y), fill=color)

    draw.line((0, horizon_y, size, horizon_y), fill=(140, 170, 190, 255), width=1)

    for lane in range(-4, 5):
        ground_x = int(size * (0.5 + lane * 0.18))
        draw.line(
            (vanishing_x, horizon_y, ground_x, ground_bottom_y),
            fill=(62, 74, 82, 255),
            width=1,
        )

    for depth_mark in (0.22, 0.38, 0.56, 0.74, 0.90):
        y = int(round(horizon_y + ((ground_bottom_y - horizon_y) * (depth_mark**0.92))))
        draw.line((0, y, size, y), fill=(48, 58, 65, 160), width=1)

    for platform in sorted(scene["platforms"], key=lambda item: item["ground_y"]):
        x1 = int(platform["x1"])
        x2 = int(platform["x2"])
        top_y = int(platform["top_y"])
        ground_y = int(platform["ground_y"])
        front_color = (92, 78, 62, 255)
        top_color = (120, 104, 82, 255)
        side_color = (74, 62, 49, 255)
        draw.rectangle((x1, top_y, x2, ground_y), fill=front_color, outline=(24, 24, 24, 255))
        top_face = [
            (x1, top_y),
            (x2, top_y),
            (min(size, x2 + 8), max(horizon_y, top_y - 6)),
            (max(0, x1 + 8), max(horizon_y, top_y - 6)),
        ]
        side_face = [
            (x2, top_y),
            (x2, ground_y),
            (min(size, x2 + 8), max(horizon_y, ground_y - 6)),
            (min(size, x2 + 8), max(horizon_y, top_y - 6)),
        ]
        draw.polygon(top_face, fill=top_color, outline=(24, 24, 24, 255))
        draw.polygon(side_face, fill=side_color, outline=(24, 24, 24, 255))

    vignette = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    vignette_draw = ImageDraw.Draw(vignette)
    vignette_draw.rectangle((0, 0, size - 1, size - 1), outline=(0, 0, 0, 90), width=6)
    image.alpha_composite(vignette)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate random FPS-style box samples.")
    parser.add_argument("--size", type=int, default=300, help="Canvas width/height.")
    parser.add_argument("--samples", type=int, default=20, help="Number of preview images.")
    parser.add_argument("--boxes", type=int, default=5, help="Boxes per preview image.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "sample_output" / "random_box_samples",
        help="Directory for generated previews.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = EspBoxDetector(
        box_count=args.boxes,
        model_path=str(ROOT / "sample_output" / "_unused_preview_model.onnx"),
        auto_download_model=False,
        random_height_ratio_min=0.09,
        random_height_ratio_max=0.55,
    )
    renderer = EspBoxRenderer(line_width=3, line_alpha=235)

    for index in range(args.samples):
        boxes, scene = detector.generate_random_layout_preview(
            args.size,
            args.size,
            box_count=args.boxes,
            seed=index + 1,
        )
        base = build_background(scene, args.size)
        composed = renderer.draw_boxes(base, boxes).convert("RGB")
        composed.save(output_dir / f"sample_{index + 1:02d}.png", format="PNG")

    print(f"Generated {args.samples} samples in {output_dir}")


if __name__ == "__main__":
    main()
