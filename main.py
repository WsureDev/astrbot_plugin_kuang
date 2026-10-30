from __future__ import annotations

import asyncio
import math
import random
import re

from astrbot.api import logger
from astrbot.api.all import At, Image, Reply
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools, register

from .core import (
    EspBoxDetector,
    EspBoxProcessor,
    EspBoxRenderer,
    configure_logger,
    get_logger,
)
from .core.module_manager import ModuleManager
from .core.channel_settings import ChannelSettings

_PLUGIN_NAME = "astrbot_plugin_kuang"
_DEFAULT_QQ_AVATAR = "https://q1.qlogo.cn/g?b=qq&nk={user_id}&s=640"
_BIDI_CONTROL_RE = re.compile(r"[\u202A-\u202E\u2066-\u2069]")
_KUANG_TRIGGER_RE = re.compile(r"^(?:\s|\[At:[^\]]+\])*\s*框\s*(?:\s|\[At:[^\]]+\])*$")
_logger = get_logger(_PLUGIN_NAME)


@register(
    "kuang",
    "WsureDev",
    "为图片叠加 FPS 外挂透视风格白色线框",
    "0.9.0",
    "https://github.com/WsureDev/astrbot_plugin_kuang",
)
class KuangPlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        self.debug_mode = bool(self.config.get("debug_mode", False))
        configure_logger(
            logger, plugin_name=_PLUGIN_NAME, debug_enabled=self.debug_mode
        )
        if self.debug_mode:
            logger.info(f"[{_PLUGIN_NAME}] debug_mode=True，调试日志已开启")
        self.plugin_data_dir = StarTools.get_data_dir(_PLUGIN_NAME)
        self.output_dir = self.plugin_data_dir / "output"
        self.model_dir = self.plugin_data_dir / "models"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.channels = ChannelSettings(self.plugin_data_dir)

        self.modules = ModuleManager(
            self.plugin_data_dir, self.config, self._configure_pipeline
        )
        self._configure_pipeline()
        self._register_module_api()

    def _configure_pipeline(self):
        self.debug_mode = bool(self.config.get("debug_mode", False))
        configure_logger(
            logger, plugin_name=_PLUGIN_NAME, debug_enabled=self.debug_mode
        )
        model_path = str(self.modules.model_path("general"))
        anime_model_path = str(self.modules.model_path("anime"))
        saliency_model_path = str(self.modules.model_path("saliency"))
        general = self.modules.is_enabled("general")
        anime = self.modules.is_enabled("anime")
        detector_backend = "yolo26n" if general else "anime_yolo" if anime else "none"
        enable_anime_fallback = general and anime
        anime_model_url = str(self.config.get("anime_model_url", "")).strip()
        anime_auto_download_model = False
        _logger.debug(
            f"detector config: backend={detector_backend}, "
            f"enable_anime_fallback={enable_anime_fallback}, "
            f"model_path={model_path}, anime_model_path={anime_model_path}, "
            f"anime_auto_download_model={anime_auto_download_model}, "
            f"anime_model_url={anime_model_url or '<default>'}"
        )
        self.detector = EspBoxDetector(
            box_count=int(self.config.get("box_count", 5)),
            head_only=bool(self.config.get("head_only", False)),
            fallback_policy=str(self.config.get("fallback_policy", "adaptive")),
            inference_num_threads=int(self.config.get("inference_num_threads", 0)),
            model_path=model_path,
            model_url=str(self.config.get("model_url", "")).strip(),
            auto_download_model=False,
            confidence_threshold=float(self.config.get("confidence_threshold", 0.25)),
            nms_iou_threshold=float(self.config.get("detector_iou_threshold", 0.45)),
            model_input_size=int(self.config.get("model_input_size", 640)),
            backend_name=detector_backend,
            enable_anime_fallback=enable_anime_fallback,
            anime_model_path=anime_model_path,
            anime_model_url=anime_model_url,
            anime_auto_download_model=anime_auto_download_model,
            anime_confidence_threshold=float(
                self.config.get("anime_confidence_threshold", 0.25)
            ),
            anime_nms_iou_threshold=float(
                self.config.get("anime_detector_iou_threshold", 0.45)
            ),
            anime_model_input_size=int(self.config.get("anime_model_input_size", 640)),
            anime_fallback_trigger_count=int(
                self.config.get("anime_fallback_trigger_count", 2)
            ),
            anime_merge_iou_threshold=float(
                self.config.get("anime_merge_iou_threshold", 0.45)
            ),
            enable_saliency=self.modules.is_enabled("saliency"),
            saliency_model_path=saliency_model_path,
            saliency_model_url=str(self.config.get("saliency_model_url", "")).strip(),
            saliency_auto_download_model=False,
            saliency_model_input_size=int(
                self.config.get("saliency_model_input_size", 320)
            ),
            saliency_threshold=float(self.config.get("saliency_threshold", 0.5)),
            saliency_min_area_ratio=float(
                self.config.get("saliency_min_area_ratio", 0.005)
            ),
            saliency_max_boxes=int(self.config.get("saliency_max_boxes", 8)),
            saliency_max_foreground_ratio=float(
                self.config.get("saliency_max_foreground_ratio", 0.995)
            ),
            saliency_preserve_aspect_ratio=bool(
                self.config.get("saliency_preserve_aspect_ratio", False)
            ),
            saliency_intra_op_num_threads=int(
                self.config.get("saliency_intra_op_num_threads", 0)
            ),
            saliency_inter_op_num_threads=int(
                self.config.get("saliency_inter_op_num_threads", 0)
            ),
            saliency_minmax_normalize_output=bool(
                self.config.get("saliency_minmax_normalize_output", True)
            ),
            enable_motion=self.modules.is_enabled("gif")
            and bool(self.config.get("enable_motion", False)),
            motion_diff_threshold=int(self.config.get("motion_diff_threshold", 25)),
            motion_min_area_ratio=float(
                self.config.get("motion_min_area_ratio", 0.005)
            ),
            motion_max_boxes=int(self.config.get("motion_max_boxes", 8)),
            motion_downscale_long_edge=int(
                self.config.get("motion_downscale_long_edge", 240)
            ),
            motion_dilate_radius=int(self.config.get("motion_dilate_radius", 4)),
        )
        self.renderer = EspBoxRenderer(
            line_width=int(self.config.get("line_width", 3)),
            line_alpha=int(self.config.get("line_alpha", 220)),
        )
        self.processor = EspBoxProcessor(
            detector=self.detector,
            renderer=self.renderer,
            output_dir=self.output_dir,
            gif_detection_interval=(
                int(self.config.get("gif_detection_interval", 4))
                if self.modules.is_enabled("gif")
                else 1
            ),
            gif_max_reuse_ms=int(self.config.get("gif_max_reuse_ms", 240)),
        )

        limit = max(1, int(self.config.get("max_concurrent_images", 1)))
        if getattr(self, "_processing_limit", None) != limit:
            self._processing_slots = asyncio.Semaphore(limit)
            self._processing_limit = limit

    def _register_module_api(self):
        register_api = getattr(self.context, "register_web_api", None)
        if not callable(register_api):
            return
        for route, handler, methods in (
            ("modules", self._api_modules, ["GET"]),
            ("modules/action", self._api_module_action, ["POST"]),
            ("modules/settings", self._api_module_settings, ["POST"]),
        ):
            register_api(f"/{_PLUGIN_NAME}/{route}", handler, methods, "框模块管理")

    @staticmethod
    async def _request_json():
        try:
            from astrbot.api.web import request
        except ImportError:
            from quart import request

            return await request.get_json()
        return await request.json(default={})

    async def _api_modules(self):
        return self.modules.snapshot()

    async def _api_module_action(self):
        try:
            body = await self._request_json()
            if not isinstance(body, dict):
                raise ValueError("请求必须是 JSON 对象。")
            module_id, action = body.get("module"), body.get("action")
            if action == "disable":
                return self.modules.disable(module_id)
            if action in {"install", "enable"}:
                return self.modules.start_install(
                    module_id, install=action == "install"
                )
            raise ValueError("不支持的模块操作。")
        except (ValueError, TypeError) as exc:
            return {"status": "error", "message": str(exc)}, 400

    async def _api_module_settings(self):
        try:
            body = await self._request_json()
            if not isinstance(body, dict):
                raise ValueError("请求必须是 JSON 对象。")
            module_id = body.get("module")
            if "values" in body:
                return self.modules.save_settings(module_id, body["values"])
            return self.modules.settings(module_id)
        except (ValueError, TypeError) as exc:
            return {"status": "error", "message": str(exc)}, 400

    @filter.command("框模块")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def kuang_modules(
        self, event: AstrMessageEvent, action: str = "状态", module: str = ""
    ):
        aliases = {
            "通用": "general",
            "前景": "saliency",
            "动漫": "anime",
            "GIF": "gif",
            "gif": "gif",
        }
        module_id = aliases.get(module, module)
        try:
            if action == "安装":
                self.modules.start_install(module_id)
            elif action == "启用":
                self.modules.start_install(module_id, install=False)
            elif action == "关闭":
                self.modules.disable(module_id)
            elif action != "状态":
                raise ValueError(
                    "用法：框模块 状态 / 安装 / 启用 / 关闭 [通用|前景|动漫|GIF]"
                )
            yield event.plain_result(self._module_status_text())
        except (ValueError, TypeError) as exc:
            yield event.plain_result(str(exc))

    def _module_status_text(self):
        phases = {
            "not_installed": "未安装",
            "installed": "已安装，未启用",
            "enabled": "已启用",
            "queued": "排队中",
            "installing": "安装依赖",
            "downloading": "下载模型",
            "validating": "加载验证",
            "error": "安装失败",
            "interrupted": "安装中断",
            "restart_required": "需要重启 AstrBot",
        }
        lines = ["框 · 模块状态"]
        for item in self.modules.snapshot()["modules"]:
            line = f"{item['name']}：{phases.get(item['phase'], item['phase'])}"
            if item["phase"] == "downloading":
                total, done = item["total_bytes"], item["downloaded_bytes"]
                line += (
                    f" {done / total:.0%}" if total else f" {done / 1048576:.1f} MiB"
                )
            if item["phase"] in {"error", "interrupted", "restart_required"}:
                line += f"\n  {item['message']}"
            lines.append(line)
        lines.append(
            "安装进度可在“[框]模块管理”页面实时查看，或再次发送“框模块 状态”。"
        )
        return "\n".join(lines)

    async def terminate(self):
        await self.modules.close()

    def _auto_animation_probability(self) -> float:
        try:
            value = float(self.config.get("auto_animation_probability", 0))
        except (TypeError, ValueError):
            return 0
        return value if math.isfinite(value) and 0 <= value <= 100 else 0

    @filter.command("kuang")
    async def kuang_animation(self, event: AstrMessageEvent, action: str = ""):
        key = self.channels.key(event)
        if key is None:
            yield event.plain_result("无法确定当前渠道，请在群聊或私聊中使用此指令。")
            return
        action = action.strip().lower()
        if action not in {"on", "off", ""}:
            yield event.plain_result(
                "用法：/kuang on 开启，/kuang off 关闭；/kuang 查看状态。"
            )
            return
        if action:
            try:
                self.channels.set_enabled(key, action == "on")
            except OSError as exc:
                _logger.warning("保存自动动图渠道设置失败：%s", exc)
                yield event.plain_result("渠道设置保存失败，请稍后重试。")
                return
        enabled = self.channels.is_enabled(key)
        master = bool(self.config.get("enable_auto_animation", False))
        probability = self._auto_animation_probability()
        lines = [
            f"当前渠道自动框动图：{'已开启' if enabled else '已关闭'}",
            f"总开关：{'已开启' if master else '已关闭'}；回复概率：{probability:g}%",
            "同一渠道的新会话继承此设置。",
        ]
        if enabled and (not master or probability == 0):
            lines.append(
                "请在 AstrBot 的“框”插件配置中打开自动动图总开关，并设置大于 0% 的回复概率。"
            )
        if enabled and not self._has_recognition_module():
            lines.append(
                "锁头模式还需安装并启用动漫识别模块。"
                if self.config.get("head_only", False)
                else "还需安装并启用至少一个识别模块。"
            )
        yield event.plain_result("\n".join(lines))

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def auto_kuang(self, event: AstrMessageEvent):
        if not self.config.get("enable_auto_animation", False):
            return
        probability = self._auto_animation_probability()
        if probability <= 0 or not self._has_recognition_module():
            return
        if not self.channels.is_enabled(self.channels.key(event)):
            return
        if self._matches_kuang_trigger(event):
            return  # The explicit command owns this message, regardless of probability.
        if event.get_self_id() and event.get_sender_id() == event.get_self_id():
            return
        if not self._collect_direct_images(event):
            return
        # Sample once per message, before downloads and decoding. Never affects boxes.
        if probability < 100 and random.random() * 100 >= probability:
            return
        async for result in self._do_kuang(event, automatic=True):
            yield result

    def _has_recognition_module(self) -> bool:
        if self.config.get("head_only", False):
            return self.modules.is_enabled("anime")
        return any(
            self.modules.is_enabled(key) for key in ("general", "saliency", "anime")
        )

    @filter.regex(r"框")
    async def kuang(self, event: AstrMessageEvent):
        if not self._matches_kuang_trigger(event):
            return
        async for result in self._do_kuang(event):
            yield result

    async def _do_kuang(self, event: AstrMessageEvent, *, automatic: bool = False):
        if not self._has_recognition_module():
            if automatic:
                return
            if self.config.get("head_only", False):
                yield event.plain_result(
                    "锁头模式需要安装并启用动漫识别模块。"
                    "请在“[框]模块管理”中启用动漫识别，或在插件配置中关闭“锁头模式”恢复锁身。"
                    "管理员也可发送：框模块 安装 动漫。"
                )
                return
            yield event.plain_result(
                "尚未启用识别模块。请在插件的“[框]模块管理”页面安装并启用通用、前景或动漫识别。"
                "管理员也可发送：框模块 安装 前景；发送“框模块 状态”查看进度。"
            )
            return
        direct_images = self._collect_direct_images(event)
        reply_images = (
            [] if direct_images or automatic else self._collect_reply_images(event)
        )
        image_targets = direct_images or reply_images
        avatar_mode = False
        _logger.debug(
            f"image collection: direct={len(direct_images)}, "
            f"reply={len(reply_images)}, selected={len(image_targets)}"
        )

        if not image_targets:
            if automatic:
                return
            avatar_target = self._build_avatar_target(event)
            if avatar_target is None:
                yield event.plain_result(
                    "❌ 没找到可处理的图片。请直接附图发送“框”，或回复带图消息发送“框”。"
                )
                return
            image_targets = [avatar_target]
            avatar_mode = True
            _logger.debug("no image found, falling back to avatar mode")

        output_paths: list[str] = []
        last_error_message = ""
        for image_target in image_targets:
            try:
                async with self._processing_slots:
                    input_path = await image_target.convert_to_file_path()
                    if automatic and not await asyncio.to_thread(
                        EspBoxProcessor.is_supported_animation, input_path
                    ):
                        continue
                    _logger.debug(f"processing image target: {input_path}")
                    output_path = await asyncio.to_thread(
                        self.processor.render_path,
                        input_path,
                    )
                event.track_temporary_local_file(output_path)
                output_paths.append(output_path)
                _logger.debug(
                    f"processing succeeded: input={input_path}, output={output_path}"
                )
            except Exception as exc:
                last_error_message = str(exc).strip()
                _logger.warning(
                    f"[{_PLUGIN_NAME}] 图片处理失败: {exc}",
                    exc_info=self.debug_mode,
                )

        if not output_paths:
            if automatic:
                return
            fallback_message = "❌ 图片处理失败。"
            if last_error_message:
                fallback_message = f"❌ 图片处理失败：{last_error_message}"
            elif avatar_mode:
                fallback_message = "❌ 头像获取或处理失败，请直接附图再试。"
            yield event.plain_result(fallback_message)
            return

        _logger.debug(
            f"完成处理: inputs={len(image_targets)}, "
            f"outputs={len(output_paths)}, avatar_mode={avatar_mode}"
        )

        for output_path in output_paths:
            yield event.image_result(output_path)

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
        target_user_id = self._extract_at_target_id(event)
        sender_id = str(event.get_sender_id() or "").strip()
        avatar_user_id = target_user_id or sender_id
        sender = getattr(event.message_obj, "sender", None)
        if not target_user_id:
            for attr_name in ("avatar", "avatar_url", "face_url", "face"):
                avatar_value = str(getattr(sender, attr_name, "") or "").strip()
                if avatar_value.startswith("http://") or avatar_value.startswith(
                    "https://"
                ):
                    _logger.debug(
                        f"avatar target resolved from sender.{attr_name}: {avatar_value}"
                    )
                    return Image.fromURL(avatar_value)

        if not avatar_user_id:
            _logger.debug("avatar target unavailable: no target_user_id or sender_id")
            return None

        template = str(self.config.get("avatar_url_template", "")).strip()
        if template:
            avatar_url = template.format(
                user_id=avatar_user_id,
                platform_name=event.get_platform_name(),
                platform_id=event.get_platform_id(),
            )
            _logger.debug(
                f"avatar target resolved from template: user_id={avatar_user_id}, url={avatar_url}"
            )
            return Image.fromURL(avatar_url)

        if avatar_user_id.isdigit():
            avatar_url = _DEFAULT_QQ_AVATAR.format(user_id=avatar_user_id)
            _logger.debug(
                f"avatar target resolved from default QQ avatar: user_id={avatar_user_id}, url={avatar_url}"
            )
            return Image.fromURL(avatar_url)

        _logger.debug(
            f"avatar target unavailable: non-numeric user_id={avatar_user_id!r}"
        )
        return None

    def _extract_at_target_id(self, event: AstrMessageEvent) -> str:
        for component in event.get_messages():
            if not isinstance(component, At):
                continue
            target_id = str(getattr(component, "qq", "") or "").strip()
            if not target_id or target_id.lower() == "all":
                continue
            return target_id
        return ""

    def _matches_kuang_trigger(self, event: AstrMessageEvent) -> bool:
        message_str = str(event.get_message_str() or "")
        sanitized = _BIDI_CONTROL_RE.sub("", message_str).strip()
        matched = bool(_KUANG_TRIGGER_RE.search(sanitized))
        _logger.debug(
            f"trigger check: raw={message_str!r}, sanitized={sanitized!r}, matched={matched}"
        )
        return matched
