from __future__ import annotations

import asyncio

from astrbot.api import logger
from astrbot.api.all import Image, Reply
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools, register

from .core import EspBoxDetector, EspBoxRenderer

_PLUGIN_NAME = "astrbot_plugin_kuang"
_DEFAULT_QQ_AVATAR = "https://q1.qlogo.cn/g?b=qq&nk={user_id}&s=640"


@register(
    "kuang",
    "WsureDev",
    "为图片叠加 FPS 外挂透视风格白色线框",
    "0.2.0",
    "https://github.com/WsureDev/astrbot_plugin_kuang",
)
class KuangPlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        self.plugin_data_dir = StarTools.get_data_dir(_PLUGIN_NAME)
        self.output_dir = self.plugin_data_dir / "output"
        self.model_dir = self.plugin_data_dir / "models"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.debug_mode = bool(self.config.get("debug_mode", False))
        configured_model_path = str(self.config.get("model_path", "")).strip()
        model_path = (
            configured_model_path
            if configured_model_path
            else str(self.model_dir / "yolo26n.onnx")
        )
        self.detector = EspBoxDetector(
            box_count=int(self.config.get("box_count", 5)),
            model_path=model_path,
            model_url=str(self.config.get("model_url", "")).strip(),
            auto_download_model=bool(self.config.get("auto_download_model", True)),
            confidence_threshold=float(
                self.config.get("confidence_threshold", 0.25)
            ),
            nms_iou_threshold=float(self.config.get("detector_iou_threshold", 0.45)),
            model_input_size=int(self.config.get("model_input_size", 640)),
            random_width_ratio_min=float(
                self.config.get("random_width_ratio_min", 0.28)
            ),
            random_width_ratio_max=float(
                self.config.get("random_width_ratio_max", 0.52)
            ),
            random_height_ratio_min=float(
                self.config.get("random_height_ratio_min", 0.22)
            ),
            random_height_ratio_max=float(
                self.config.get("random_height_ratio_max", 0.55)
            ),
        )
        self.renderer = EspBoxRenderer(
            line_width=int(self.config.get("line_width", 3)),
            line_alpha=int(self.config.get("line_alpha", 220)),
        )

    @filter.regex(r"^\s*框\s*$")
    async def kuang(self, event: AstrMessageEvent):
        async for result in self._do_kuang(event):
            yield result

    async def _do_kuang(self, event: AstrMessageEvent):
        direct_images = self._collect_direct_images(event)
        reply_images = [] if direct_images else self._collect_reply_images(event)
        image_targets = direct_images or reply_images
        avatar_mode = False

        if not image_targets:
            avatar_target = self._build_avatar_target(event)
            if avatar_target is None:
                yield event.plain_result(
                    "❌ 没找到可处理的图片。请直接附图发送“框”，或回复带图消息发送“框”。"
                )
                return
            image_targets = [avatar_target]
            avatar_mode = True

        output_paths: list[str] = []
        last_error_message = ""
        for image_target in image_targets:
            try:
                input_path = await image_target.convert_to_file_path()
                output_path = await asyncio.to_thread(
                    self._render_image_with_boxes,
                    input_path,
                )
                event.track_temporary_local_file(output_path)
                output_paths.append(output_path)
            except Exception as exc:
                last_error_message = str(exc).strip()
                logger.warning(
                    f"[{_PLUGIN_NAME}] 图片处理失败: {exc}",
                    exc_info=self.debug_mode,
                )

        if not output_paths:
            fallback_message = "❌ 图片处理失败。"
            if last_error_message:
                fallback_message = f"❌ 图片处理失败：{last_error_message}"
            elif avatar_mode:
                fallback_message = "❌ 头像获取或处理失败，请直接附图再试。"
            yield event.plain_result(fallback_message)
            return

        if self.debug_mode:
            logger.info(
                f"[{_PLUGIN_NAME}] 完成处理: inputs={len(image_targets)}, "
                f"outputs={len(output_paths)}, avatar_mode={avatar_mode}"
            )

        for output_path in output_paths:
            yield event.image_result(output_path)

    def _render_image_with_boxes(self, image_path: str) -> str:
        boxes = self.detector.detect_from_path(image_path)
        return self.renderer.render(image_path, boxes, str(self.output_dir))

    def _collect_direct_images(self, event: AstrMessageEvent) -> list[Image]:
        images: list[Image] = []
        for component in event.get_messages():
            if isinstance(component, Image):
                images.append(component)
        return images

    def _collect_reply_images(self, event: AstrMessageEvent) -> list[Image]:
        images: list[Image] = []
        for component in event.get_messages():
            if not isinstance(component, Reply):
                continue
            if not getattr(component, "chain", None):
                continue
            for reply_component in component.chain:
                if isinstance(reply_component, Image):
                    images.append(reply_component)
        return images

    def _build_avatar_target(self, event: AstrMessageEvent) -> Image | None:
        sender_id = str(event.get_sender_id() or "").strip()
        sender = getattr(event.message_obj, "sender", None)
        for attr_name in ("avatar", "avatar_url", "face_url", "face"):
            avatar_value = str(getattr(sender, attr_name, "") or "").strip()
            if avatar_value.startswith("http://") or avatar_value.startswith("https://"):
                return Image.fromURL(avatar_value)

        if not sender_id:
            return None

        template = str(self.config.get("avatar_url_template", "")).strip()
        if template:
            avatar_url = template.format(
                user_id=sender_id,
                platform_name=event.get_platform_name(),
                platform_id=event.get_platform_id(),
            )
            return Image.fromURL(avatar_url)

        if sender_id.isdigit():
            return Image.fromURL(_DEFAULT_QQ_AVATAR.format(user_id=sender_id))

        return None
