from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import EspBoxDetector, EspBoxRenderer


def project_point(scene: dict, size: int, x: float, y: float, z: float):
    if z <= 0.2:
        return None
    focal_x = float(scene["focal_x"])
    focal_y = float(scene["focal_y"])
    camera_height = float(scene["camera_height"])
    screen_x = (size / 2.0) + ((x * focal_x) / z)
    screen_y = (size / 2.0) - (((y - camera_height) * focal_y) / z)
    return screen_x, screen_y


def build_background(scene: dict, size: int):
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (size, size), (0, 0, 0, 255))
    draw = ImageDraw.Draw(image)
    horizon_y = int(scene["horizon_y"])
    ground_bottom_y = size - 1

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

    for world_x in (-18, -12, -7, -3, 0, 3, 7, 12, 18):
        near_point = project_point(scene, size, world_x, 0.0, 4.0)
        far_point = project_point(scene, size, world_x, 0.0, 55.0)
        if near_point and far_point:
            draw.line((*near_point, *far_point), fill=(62, 74, 82, 255), width=1)

    for world_z in (8.0, 12.0, 18.0, 26.0, 38.0, 52.0):
        left = project_point(scene, size, -20.0, 0.0, world_z)
        right = project_point(scene, size, 20.0, 0.0, world_z)
        if left and right:
            draw.line((*left, *right), fill=(48, 58, 65, 160), width=1)

    for platform in sorted(scene["platforms"], key=lambda item: item["z"], reverse=True):
        x = float(platform["x"])
        z = float(platform["z"])
        width = float(platform["width"])
        depth = float(platform["depth"])
        height = float(platform["height"])
        z_front = max(1.0, z - (depth / 2.0))
        front_bl = project_point(scene, size, x - (width / 2.0), 0.0, z_front)
        front_br = project_point(scene, size, x + (width / 2.0), 0.0, z_front)
        front_tl = project_point(scene, size, x - (width / 2.0), height, z_front)
        front_tr = project_point(scene, size, x + (width / 2.0), height, z_front)
        if not all((front_bl, front_br, front_tl, front_tr)):
            continue
        front_color = (92, 78, 62, 255)
        draw.polygon(
            [front_bl, front_br, front_tr, front_tl],
            fill=front_color,
            outline=(24, 24, 24, 255),
        )
        draw.line((*front_tl, *front_tr), fill=(132, 116, 92, 255), width=1)

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
