from __future__ import annotations

import inspect
from pathlib import Path

from .logger import get_logger

_logger = get_logger(__name__)


class EspBoxProcessor:
    @staticmethod
    def is_supported_animation(image_path: str) -> bool:
        from PIL import Image

        with Image.open(image_path) as image:
            return image.format in {"GIF", "WEBP"} and bool(
                getattr(image, "is_animated", False)
            )

    def __init__(
        self,
        *,
        detector,
        renderer,
        output_dir: str | Path,
        gif_detection_interval: int = 4,
        gif_max_reuse_ms: int = 240,
    ) -> None:
        self.detector = detector
        self.renderer = renderer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gif_detection_interval = max(1, int(gif_detection_interval))
        self.gif_max_reuse_ms = max(20, int(gif_max_reuse_ms))

    def render_path(self, image_path: str) -> str:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "缺少 Pillow 依赖，请在模块管理中安装识别模块（命令行开发使用 requirements/dev.txt）。"
            ) from exc

        _logger.debug("[processor] render_path start: %s", image_path)
        with Image.open(image_path) as image:
            _logger.debug(
                "[processor] opened image: path=%s format=%s size=%sx%s animated=%s",
                image_path,
                image.format,
                image.width,
                image.height,
                bool(getattr(image, "is_animated", False)),
            )
            if image.format in {"GIF", "WEBP"} and getattr(image, "is_animated", False):
                _logger.info(
                    "[processor] %s animation detected, switching to frame-by-frame render",
                    image.format,
                )
                return self._render_animation(image_path)

        detection_result = self.detector.analyze_from_path(image_path)
        stage3_boxes = getattr(detection_result, "stage3_boxes", []) or []
        stage4_boxes = getattr(detection_result, "stage4_boxes", []) or []
        _logger.debug(
            "[processor] detector pipeline: stage1=%s stage2=%s stage2_composite=%s"
            " stage3=%s stage4=%s arranged=%s final=%s",
            len(detection_result.stage1_boxes),
            len(detection_result.stage2_boxes),
            len(detection_result.stage2_composite_boxes),
            len(stage3_boxes),
            len(stage4_boxes),
            len(detection_result.arranged_boxes),
            len(detection_result.final_boxes),
        )
        output_path = self.renderer.render(
            image_path,
            detection_result.final_boxes,
            str(self.output_dir),
        )
        _logger.debug("[processor] render_path done: output=%s", output_path)
        return output_path

    def _render_animation(self, image_path: str) -> str:
        try:
            import numpy as np
            from PIL import Image, ImageSequence
        except ImportError as exc:
            raise RuntimeError(
                "缺少图片处理依赖，请在模块管理中安装识别模块（命令行开发使用 requirements/dev.txt）。"
            ) from exc

        rendered_frames = []
        durations: list[int] = []
        previous_frame_array = None
        analyze = self.detector.analyze
        accepts_previous_frame = self._supports_previous_frame(analyze)
        # The stream belongs to this render call, never to the shared detector.
        from .detector import EspBoxDetector
        from .temporal import GifDetectionStream

        stream = None
        if (
            isinstance(self.detector, EspBoxDetector)
            and self.gif_detection_interval > 1
        ):
            stream = GifDetectionStream(
                self.detector,
                interval=self.gif_detection_interval,
                max_age_ms=self.gif_max_reuse_ms,
            )
        previous_duration = 0

        with Image.open(image_path) as image:
            base_duration = int(image.info.get("duration", 100) or 100)
            loop = int(image.info.get("loop", 0) or 0)

            frame_index = 0
            for frame in ImageSequence.Iterator(image):
                frame_index += 1
                # WebP exposes this frame's duration only after decoding it.
                rgba_frame = frame.convert("RGBA")
                duration = int(
                    frame.info.get("duration", base_duration) or base_duration
                )
                rgb_frame = rgba_frame.convert("RGB")
                current_frame_array = np.asarray(rgb_frame)
                if stream is not None:
                    detection_result = stream.analyze(
                        current_frame_array, elapsed_ms=previous_duration
                    )
                elif accepts_previous_frame:
                    detection_result = analyze(
                        current_frame_array,
                        previous_frame_rgb=previous_frame_array,
                    )
                else:
                    # Preserve compatibility with injected detectors written
                    # before temporal detection was added.
                    detection_result = analyze(current_frame_array)
                stage3_boxes = getattr(detection_result, "stage3_boxes", []) or []
                stage4_boxes = getattr(detection_result, "stage4_boxes", []) or []
                _logger.debug(
                    "[processor] GIF frame=%s duration=%sms stage1=%s stage2=%s"
                    " stage2_composite=%s stage3=%s stage4=%s arranged=%s final=%s",
                    frame_index,
                    duration,
                    len(detection_result.stage1_boxes),
                    len(detection_result.stage2_boxes),
                    len(detection_result.stage2_composite_boxes),
                    len(stage3_boxes),
                    len(stage4_boxes),
                    len(detection_result.arranged_boxes),
                    len(detection_result.final_boxes),
                )
                _logger.debug(
                    "[processor] GIF frame=%s decisions=%s timings_ms=%s",
                    frame_index,
                    getattr(detection_result, "decisions", {}),
                    getattr(detection_result, "timings_ms", {}),
                )
                # Retain one byte per pixel instead of RGBA for every frame.
                rendered = self.renderer.draw_boxes(
                    rgba_frame, detection_result.final_boxes
                )
                rendered_frames.append(
                    rendered.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
                )
                durations.append(max(20, duration))
                previous_frame_array = current_frame_array
                previous_duration = duration

        if not rendered_frames:
            raise RuntimeError("GIF 中没有可处理的帧。")

        output_path = self.renderer.save_gif(
            frames=rendered_frames,
            durations=durations,
            loop=loop,
            output_dir=str(self.output_dir),
        )
        _logger.debug(
            "[processor] GIF render done: frames=%s output=%s",
            len(rendered_frames),
            output_path,
        )
        return output_path

    @staticmethod
    def _supports_previous_frame(analyze) -> bool:
        """Return whether a detector accepts the temporal keyword argument.

        The processor is also used with lightweight injected detectors in
        tests and integrations.  Signature inspection avoids catching an
        unrelated ``TypeError`` raised from inside detector inference.
        """
        try:
            signature = inspect.signature(analyze)
        except (TypeError, ValueError):
            # If a callable does not expose a signature, assume the current
            # detector contract; this keeps wrapped/new implementations useful.
            return True
        parameters = signature.parameters
        return "previous_frame_rgb" in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
