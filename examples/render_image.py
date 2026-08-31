"""CLI 完整流程测试脚本。

用法示例：
    uv run python examples/render_image.py ~/Pictures/some.gif
    uv run python examples/render_image.py path/to.png --disable-saliency
    uv run python examples/render_image.py path/to.gif --disable-motion --anime

跑一次 processor.render_path，等价于插件收到一张图后走的完整链路：
Stage1(yolo26n) → Stage2(anime, 可选) → Stage3(saliency, 可选)
 → Stage4(motion, GIF 帧差) → arranger → Stage5(随机补框, 可选)。
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import (  # noqa: E402
    EspBoxDetector,
    EspBoxProcessor,
    EspBoxRenderer,
    configure_logger,
    get_logger,
)


_logger = get_logger("astrbot_plugin_kuang")


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
        description="跑完整框图 pipeline，支持静态图和 GIF。",
    )
    parser.add_argument("image", type=Path, help="输入图片路径（支持 GIF）。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "sample_output" / "render_image",
        help="输出目录，默认 sample_output/render_image。",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "sample_output" / "models",
        help="模型缓存目录，默认 sample_output/models。首次运行会自动下载。",
    )
    parser.add_argument("--box-count", type=int, default=5)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="打印每个 stage 的候选框细节。",
    )

    # stage 开关（默认全开，方便一次性验证新 pipeline）
    parser.add_argument(
        "--anime",
        action="store_true",
        help="开启二次元兜底（stage2）。默认关，跟插件默认一致。",
    )
    parser.add_argument(
        "--disable-saliency",
        action="store_true",
        help="关闭显著性 stage3。",
    )
    parser.add_argument(
        "--disable-motion",
        action="store_true",
        help="关闭 GIF 运动 stage4。",
    )
    parser.add_argument(
        "--disable-random",
        action="store_true",
        help="关闭随机补框 stage5。",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    _install_console_logger(args.debug)

    image_path = args.image.expanduser().resolve()
    if not image_path.exists():
        print(f"❌ 找不到输入文件: {image_path}", file=sys.stderr)
        return 2

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.model_dir.resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    yolo_model_path = str(model_dir / "yolo26n.onnx")
    anime_model_path = str(model_dir / "anime_yolo.onnx")
    saliency_model_path = str(model_dir / "u2netp.onnx")

    _logger.info("image=%s", image_path)
    _logger.info("output_dir=%s", output_dir)
    _logger.info("model_dir=%s", model_dir)
    _logger.info(
        "stages: anime=%s saliency=%s motion=%s random=%s",
        args.anime,
        not args.disable_saliency,
        not args.disable_motion,
        not args.disable_random,
    )

    detector = EspBoxDetector(
        box_count=args.box_count,
        model_path=yolo_model_path,
        auto_download_model=True,
        enable_random_boxes=not args.disable_random,
        enable_anime_fallback=args.anime,
        anime_model_path=anime_model_path,
        anime_auto_download_model=True,
        enable_saliency=not args.disable_saliency,
        saliency_model_path=saliency_model_path,
        saliency_auto_download_model=True,
        enable_motion=not args.disable_motion,
    )
    renderer = EspBoxRenderer()
    processor = EspBoxProcessor(
        detector=detector,
        renderer=renderer,
        output_dir=output_dir,
    )

    started_at = time.perf_counter()
    output_path = processor.render_path(str(image_path))
    elapsed = time.perf_counter() - started_at
    print(f"✅ done in {elapsed:.2f}s")
    print(f"   input  : {image_path}")
    print(f"   output : {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
