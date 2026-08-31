"""调试对比工具：一次输入产出 5 份结果。

- 01_stage1_only：只画 YOLO26n 阶段的候选框
- 02_stage2_only：只画二次元 Head/Torso/Legs（未合成）
- 03_stage3_only：只画显著性检测框
- 04_stage4_only：只画帧差运动框（仅 GIF 有效）
- 05_merged    ：走完整仲裁的最终结果（关闭随机补框，纯 stage 融合）

设计意图是"每帧只跑一次 detector，5 份产物共用同一批候选"，方便对比同一决策下各 stage 的贡献。

用法示例：
    uv run python examples/render_stages.py ~/Pictures/some.gif
    uv run python examples/render_stages.py ~/Pictures/a.png ~/Pictures/b.jpg
    uv run python examples/render_stages.py path/to.gif --debug
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import (  # noqa: E402
    EspBoxDetector,
    EspBoxRenderer,
    configure_logger,
    get_logger,
)
from core.models import DetectionBox, DetectionPipelineResult  # noqa: E402


_logger = get_logger("astrbot_plugin_kuang")


# 顺序即最终产物文件名前缀顺序
_VARIANTS: list[tuple[str, str]] = [
    ("01_stage1_only", "stage1_boxes"),
    ("02_stage2_only", "stage2_boxes"),
    ("03_stage3_only", "stage3_boxes"),
    ("04_stage4_only", "stage4_boxes"),
    ("05_merged", "final_boxes"),
]


def _install_console_logger(debug: bool) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    )
    root_logger.handlers[:] = [handler]

    configure_logger(
        logging.getLogger("astrbot_plugin_kuang_stdin"),
        plugin_name="astrbot_plugin_kuang",
        debug_enabled=debug,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按 stage 拆分渲染，产出 5 份对比图/GIF。",
    )
    parser.add_argument(
        "images",
        type=Path,
        nargs="+",
        help="一个或多个输入图片/GIF 路径。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "sample_output" / "render_stages",
        help="输出根目录，默认 sample_output/render_stages。",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "sample_output" / "models",
        help="模型缓存目录（首次运行会自动下载）。",
    )
    parser.add_argument("--box-count", type=int, default=5)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="打印每个 stage 的候选框细节。",
    )
    return parser


def _pick_boxes(result: DetectionPipelineResult, field: str) -> list[DetectionBox]:
    return list(getattr(result, field, []) or [])


def _save_png(base_rgba, boxes: list[DetectionBox], output_path: Path, renderer: EspBoxRenderer) -> None:
    composed = renderer.draw_boxes(base_rgba, boxes).convert("RGB")
    composed.save(output_path, format="PNG")


def _save_gif(
    frames: list,
    durations: list[int],
    loop: int,
    output_path: Path,
) -> None:
    from PIL import Image

    prepared = [
        frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
        for frame in frames
    ]
    prepared[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=prepared[1:],
        duration=durations,
        loop=loop,
        disposal=2,
        optimize=True,
    )


def _process_static(
    image_path: Path,
    output_dir: Path,
    detector: EspBoxDetector,
    renderer: EspBoxRenderer,
) -> dict[str, int]:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("缺少 Pillow/NumPy 依赖，请先安装 requirements.txt。") from exc

    with Image.open(image_path) as image:
        base_rgba = image.convert("RGBA")
        rgb_array = np.asarray(base_rgba.convert("RGB"))

    result = detector.analyze(rgb_array)
    counts: dict[str, int] = {}
    for prefix, field in _VARIANTS:
        boxes = _pick_boxes(result, field)
        counts[prefix] = len(boxes)
        output_path = output_dir / f"{prefix}.png"
        _save_png(base_rgba, boxes, output_path, renderer)
    return counts


def _process_gif(
    image_path: Path,
    output_dir: Path,
    detector: EspBoxDetector,
    renderer: EspBoxRenderer,
) -> dict[str, int]:
    try:
        import numpy as np
        from PIL import Image, ImageSequence
    except ImportError as exc:
        raise RuntimeError("缺少 Pillow/NumPy 依赖，请先安装 requirements.txt。") from exc

    frames_by_variant: dict[str, list] = {prefix: [] for prefix, _ in _VARIANTS}
    max_counts: dict[str, int] = {prefix: 0 for prefix, _ in _VARIANTS}
    durations: list[int] = []
    previous_frame_array = None

    with Image.open(image_path) as image:
        base_duration = int(image.info.get("duration", 100) or 100)
        loop = int(image.info.get("loop", 0) or 0)

        for frame in ImageSequence.Iterator(image):
            duration = int(frame.info.get("duration", base_duration) or base_duration)
            rgba_frame = frame.convert("RGBA")
            rgb_array = np.asarray(rgba_frame.convert("RGB"))

            result = detector.analyze(
                rgb_array,
                previous_frame_rgb=previous_frame_array,
            )
            for prefix, field in _VARIANTS:
                boxes = _pick_boxes(result, field)
                if len(boxes) > max_counts[prefix]:
                    max_counts[prefix] = len(boxes)
                frames_by_variant[prefix].append(
                    renderer.draw_boxes(rgba_frame, boxes)
                )
            durations.append(max(20, duration))
            previous_frame_array = rgb_array

    for prefix, _ in _VARIANTS:
        frames = frames_by_variant[prefix]
        if not frames:
            continue
        _save_gif(
            frames=frames,
            durations=durations,
            loop=loop,
            output_path=output_dir / f"{prefix}.gif",
        )
    return max_counts


def _is_animated_gif(image_path: Path) -> bool:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("缺少 Pillow 依赖，请先安装 requirements.txt。") from exc
    with Image.open(image_path) as image:
        return image.format == "GIF" and bool(getattr(image, "is_animated", False))


def _process_one(
    image_path: Path,
    output_root: Path,
    detector: EspBoxDetector,
    renderer: EspBoxRenderer,
) -> None:
    if not image_path.exists():
        print(f"❌ 找不到文件: {image_path}", file=sys.stderr)
        return

    stem = image_path.stem
    unique_suffix = uuid.uuid4().hex[:6]
    output_dir = output_root / f"{stem}_{unique_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    is_gif = _is_animated_gif(image_path)
    started = time.perf_counter()
    if is_gif:
        counts = _process_gif(image_path, output_dir, detector, renderer)
        kind = "GIF"
    else:
        counts = _process_static(image_path, output_dir, detector, renderer)
        kind = "IMG"
    elapsed = time.perf_counter() - started

    print(f"✅ [{kind}] {image_path.name}  ({elapsed:.2f}s)")
    for prefix, _ in _VARIANTS:
        label = "max_frame_count" if is_gif else "count"
        print(f"   {prefix:<18} {label}={counts[prefix]}")
    print(f"   → {output_dir}\n")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    _install_console_logger(args.debug)

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    model_dir = args.model_dir.resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    yolo_model_path = str(model_dir / "yolo26n.onnx")
    anime_model_path = str(model_dir / "anime_yolo.onnx")
    saliency_model_path = str(model_dir / "u2netp.onnx")

    _logger.info("output_root=%s", output_root)
    _logger.info("model_dir=%s", model_dir)
    _logger.info(
        "stage matrix always-on: stage1(yolo26n) + stage2(anime) + stage3(saliency) + stage4(motion), random off"
    )

    detector = EspBoxDetector(
        box_count=args.box_count,
        model_path=yolo_model_path,
        auto_download_model=True,
        # 关掉随机补框——测试要看的是各 stage 的真实贡献，不需要凑数
        enable_random_boxes=False,
        # 强制开启 anime，把触发阈值调很高保证每帧都跑一次
        enable_anime_fallback=True,
        anime_model_path=anime_model_path,
        anime_auto_download_model=True,
        anime_fallback_trigger_count=999,
        # 打开 saliency / motion
        enable_saliency=True,
        saliency_model_path=saliency_model_path,
        saliency_auto_download_model=True,
        enable_motion=True,
    )
    renderer = EspBoxRenderer()

    for image_path in args.images:
        _process_one(image_path.expanduser().resolve(), output_root, detector, renderer)

    return 0


if __name__ == "__main__":
    sys.exit(main())
