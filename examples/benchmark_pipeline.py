"""Compare full inference, adaptive fallbacks and GIF tracking on local inputs.

Manifest: {"samples": [{"path": "input.gif", "boxes": {"0": [[x1,y1,x2,y2]]}}]}.
Frame numbers are zero based; only explicitly annotated frames affect quality
metrics. An empty annotation list marks a frame with no target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageSequence

from core.detector import EspBoxDetector
from core.models import DetectionBox
from core.renderer import EspBoxRenderer
from core.temporal import GifDetectionStream


def quality_metrics(results, annotations):
    """Maximum one-to-one matches at IoU >= 0.5, without category labels."""
    matched = targets = predictions = annotated = 0
    for key, coordinates in annotations.items():
        index = int(key)
        if index < 0 or index >= len(results):
            raise ValueError(f"标注帧越界: {index}")
        truth = [
            DetectionBox(*coords, 1, "subject", 0, "annotation")
            for coords in coordinates
        ]
        boxes = results[index].final_boxes
        edges = [
            [j for j, candidate in enumerate(boxes) if target.iou(candidate) >= 0.5]
            for target in truth
        ]
        used = {}

        def assign(i, seen):
            for j in edges[i]:
                if j in seen:
                    continue
                seen.add(j)
                if j not in used or assign(used[j], seen):
                    used[j] = i
                    return True
            return False

        matched += sum(assign(i, set()) for i in range(len(truth)))
        targets += len(truth)
        predictions += len(boxes)
        annotated += 1
    return {
        "annotated_frames": annotated,
        "matched_targets": matched,
        "targets": targets,
        "predictions": predictions,
        "recall_iou50": matched / targets if targets else None,
        "precision_iou50": matched / predictions if predictions else None,
    }


def build_detector(model_dir, anime, policy):
    return EspBoxDetector(
        model_path=str(model_dir / "yolo26n.onnx"),
        auto_download_model=False,
        enable_anime_fallback=anime,
        anime_model_path=str(model_dir / "anime_yolo.onnx"),
        anime_auto_download_model=False,
        enable_saliency=True,
        saliency_model_path=str(model_dir / "u2netp.onnx"),
        saliency_auto_download_model=False,
        fallback_policy=policy,
    )


def evaluate(path, annotations, args):
    with Image.open(path) as image:
        frames = [frame.convert("RGBA") for frame in ImageSequence.Iterator(image)]
        durations = []
        for i in range(len(frames)):
            image.seek(i)
            durations.append(max(20, int(image.info.get("duration", 100))))
        loop = int(image.info.get("loop", 0))
    arrays = [np.asarray(frame.convert("RGB")) for frame in frames]
    report = {
        "input": str(path),
        "frames": len(frames),
        "size": frames[0].size,
        "modes": {},
    }
    renderer = EspBoxRenderer()
    for mode in ("all", "adaptive", "tracked"):
        detector = build_detector(
            args.model_dir, args.anime, "all" if mode == "all" else "adaptive"
        )
        calls = Counter()
        backends = {
            "primary": detector._backend_plan.primary_backend,
            "anime": detector._backend_plan.secondary_backend,
            "saliency": detector._saliency_backend,
        }
        for name, backend in backends.items():
            if backend is None:
                continue
            backend.detect(arrays[0])  # Warm actual inference, outside measured region.
            original = backend.detect

            def counted(*pos, _name=name, _original=original, **kw):
                calls[_name] += 1
                return _original(*pos, **kw)

            backend.detect = counted
        elapsed_runs = []
        for _ in range(args.repeat):
            stream = GifDetectionStream(
                detector, interval=args.interval, max_age_ms=args.max_age_ms
            )
            calls.clear()
            results, latencies = [], []
            started = time.perf_counter()
            for index, array in enumerate(arrays):
                frame_started = time.perf_counter()
                if mode == "tracked":
                    result = stream.analyze(
                        array, elapsed_ms=durations[index - 1] if index else 0
                    )
                else:
                    result = detector.analyze(
                        array, previous_frame_rgb=arrays[index - 1] if index else None
                    )
                results.append(result)
                latencies.append((time.perf_counter() - frame_started) * 1000)
            elapsed_runs.append((time.perf_counter() - started) * 1000)
        details = {
            "median_detection_ms": round(statistics.median(elapsed_runs), 2),
            "runs_ms": [round(value, 2) for value in elapsed_runs],
            "last_run_frame_p95_ms": round(float(np.percentile(latencies, 95)), 2),
            "model_calls": dict(calls),
            "empty_frames": sum(not r.final_boxes for r in results),
            "frame_decisions": [r.decisions for r in results],
            "boxes": [
                [
                    {"xyxy": b.as_tuple(), "source": b.source, "role": b.semantic_role}
                    for b in r.final_boxes
                ]
                for r in results
            ],
            "quality": quality_metrics(results, annotations),
        }
        if args.render:
            rendered = [
                renderer.draw_boxes(frame, result.final_boxes)
                for frame, result in zip(frames, results)
            ]
            directory = args.output_dir / path.stem / mode
            directory.mkdir(parents=True, exist_ok=True)
            if len(frames) > 1:
                details["rendered"] = renderer.save_gif(
                    frames=rendered,
                    durations=durations,
                    loop=loop,
                    output_dir=str(directory),
                )
            else:
                output = directory / "result.png"
                rendered[0].save(output)
                details["rendered"] = str(output)
        report["modes"][mode] = details
        print(
            path.name,
            mode,
            details["median_detection_ms"],
            "ms",
            dict(calls),
            flush=True,
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "sample_output/models")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "sample_output/benchmark"
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--anime", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--interval", type=int, default=4)
    parser.add_argument("--max-age-ms", type=int, default=240)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat 必须大于 0")
    samples = [{"path": str(path.resolve())} for path in args.images]
    if args.manifest:
        for item in json.loads(args.manifest.read_text())["samples"]:
            samples.append(
                {**item, "path": str((args.manifest.parent / item["path"]).resolve())}
            )
    if not samples:
        parser.error("请提供图片或 --manifest")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = ["yolo26n.onnx", "u2netp.onnx"] + (
        ["anime_yolo.onnx"] if args.anime else []
    )
    hashes = {}
    for name in names:
        with (args.model_dir / name).open("rb") as source:
            digest = hashlib.sha256()
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
            hashes[name] = digest.hexdigest()
    import onnxruntime as ort

    report = {
        "created_at_ms": time.time_ns() // 1_000_000,
        "platform": platform.platform(),
        "python": sys.version,
        "onnxruntime": ort.__version__,
        "provider": "CPUExecutionProvider",
        "models_sha256": hashes,
        "anime": args.anime,
        "interval": args.interval,
        "max_age_ms": args.max_age_ms,
        "repeat": args.repeat,
        "samples": [
            evaluate(Path(item["path"]), item.get("boxes", {}), args)
            for item in samples
        ],
    }
    output = args.output_dir / "report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
