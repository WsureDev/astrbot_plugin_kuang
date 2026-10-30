import importlib.util
import json
import logging
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def load_plugin(data_dir):
    def decorator(*args, **kwargs):
        return lambda value: value

    class Star:
        def __init__(self, context):
            self.context = context

    def module(name, **attrs):
        value = types.ModuleType(name)
        value.__dict__.update(attrs)
        return value

    fake = {
        "astrbot": module("astrbot"),
        "astrbot.api": module("astrbot.api", logger=logging.getLogger("kuang_test")),
        "astrbot.api.all": module(
            "astrbot.api.all",
            At=type("At", (), {}),
            Image=type("Image", (), {}),
            Reply=type("Reply", (), {}),
        ),
        "astrbot.api.event": module(
            "astrbot.api.event",
            AstrMessageEvent=object,
            filter=types.SimpleNamespace(
                regex=decorator,
                command=decorator,
                permission_type=decorator,
                event_message_type=decorator,
                EventMessageType=types.SimpleNamespace(ALL="all"),
                PermissionType=types.SimpleNamespace(ADMIN="admin"),
            ),
        ),
        "astrbot.api.star": module(
            "astrbot.api.star",
            Context=object,
            Star=Star,
            register=decorator,
            StarTools=types.SimpleNamespace(get_data_dir=lambda name: data_dir),
        ),
        "kuang_test_plugin": module("kuang_test_plugin", __path__=[str(ROOT)]),
    }
    spec = importlib.util.spec_from_file_location(
        "kuang_test_plugin.main", ROOT / "main.py"
    )
    result = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, fake):
        spec.loader.exec_module(result)
        plugin_modules = {
            name: value
            for name, value in sys.modules.items()
            if name == "kuang_test_plugin" or name.startswith("kuang_test_plugin.")
        }
    sys.modules.update(plugin_modules)
    return result


class PluginModuleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.data = Path(self.temp.name)
        self.module = load_plugin(self.data)
        self.routes = {}
        self.context = types.SimpleNamespace(
            register_web_api=lambda path, handler, methods, description: (
                self.routes.update({path: handler})
            )
        )
        self.config = json.loads((ROOT / "_conf_schema.json").read_text())
        self.config = {key: field["default"] for key, field in self.config.items()}
        manager = self.module.ModuleManager
        self.patch = patch.object(
            self.module,
            "ModuleManager",
            side_effect=lambda data, config, callback: manager(
                data, config, callback, probe=lambda reqs: []
            ),
        )
        self.patch.start()
        self.plugin = self.module.KuangPlugin(self.context, self.config)

    async def asyncTearDown(self):
        await self.plugin.terminate()
        self.patch.stop()
        self.temp.cleanup()

    def add_model(self, key):
        path = self.plugin.modules.model_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"already installed")

    async def test_empty_core_loads_and_provides_setup_guidance(self):
        self.assertEqual(self.plugin.detector._backend_plan.mode, "disabled")
        self.assertEqual(self.plugin.processor.gif_detection_interval, 1)
        event = types.SimpleNamespace(plain_result=lambda message: message)
        results = [value async for value in self.plugin._do_kuang(event)]
        self.assertIn("尚未启用识别模块", results[0])
        self.assertIn("/astrbot_plugin_kuang/modules", self.routes)

    async def test_anime_alone_does_not_create_general_backend(self):
        self.add_model("anime")
        self.config["enable_anime_fallback"] = True
        self.plugin._configure_pipeline()
        plan = self.plugin.detector._backend_plan
        self.assertEqual(plan.mode, "anime_only")
        self.assertFalse(plan.primary_backend.auto_download_model)
        self.assertIsNone(plan.secondary_backend)
        self.assertFalse(self.plugin.detector.enable_saliency)

    async def test_saliency_alone_and_standard_gif_do_not_require_opencv(self):
        self.add_model("saliency")
        self.config["enable_saliency"] = True
        slots = self.plugin._processing_slots
        self.plugin._configure_pipeline()
        self.assertIs(self.plugin._processing_slots, slots)
        self.assertEqual(self.plugin.detector._backend_plan.mode, "disabled")
        self.assertTrue(self.plugin.detector.enable_saliency)
        self.assertFalse(self.plugin.detector._saliency_backend.auto_download_model)
        self.assertEqual(self.plugin.processor.gif_detection_interval, 1)

    async def test_enabling_gif_applies_tracking_and_disabling_restores_full_detection(
        self,
    ):
        self.config["enable_gif_acceleration"] = True
        self.plugin._configure_pipeline()
        self.assertEqual(self.plugin.processor.gif_detection_interval, 4)
        self.plugin.modules.disable("gif")
        self.assertEqual(self.plugin.processor.gif_detection_interval, 1)

    async def test_queued_repair_removes_backend_from_active_pipeline(self):
        self.add_model("general")
        self.add_model("saliency")
        self.config.update(enable_general=True, enable_saliency=True)
        self.plugin._configure_pipeline()
        self.plugin.modules.validator = lambda *args: None
        self.plugin.modules.start_install("general")
        self.assertIsNone(self.plugin.detector._backend_plan.primary_backend)
        self.assertTrue(self.plugin.detector.enable_saliency)
        await __import__("asyncio").gather(*self.plugin.modules._tasks.values())
        self.assertIsNotNone(self.plugin.detector._backend_plan.primary_backend)

    async def test_api_rejects_unknown_actions_and_returns_progress(self):
        async def request():
            return {"module": "general", "action": "unknown"}

        self.plugin._request_json = request
        body, status = await self.plugin._api_module_action()
        self.assertEqual(status, 400)
        self.assertEqual(body["status"], "error")
        self.assertEqual(len((await self.plugin._api_modules())["modules"]), 4)

    async def test_head_toggle_rebuilds_pipeline_without_changing_other_preferences(
        self,
    ):
        for key in ("general", "anime", "saliency"):
            self.add_model(key)
        self.config.update(
            enable_general=True,
            enable_anime_fallback=True,
            enable_saliency=True,
            enable_gif_acceleration=True,
            auto_animation_probability=37,
        )
        self.plugin._configure_pipeline()
        original_config = dict(self.config)
        original_processor = self.plugin.processor
        self.plugin.modules.save_settings("common", {"head_only": True})
        self.assertTrue(self.plugin.detector.head_only)
        self.assertTrue(self.plugin._has_recognition_module())
        self.assertEqual(self.plugin.processor.gif_detection_interval, 4)
        self.assertIsNot(self.plugin.processor, original_processor)
        self.assertFalse(original_processor.detector.head_only)
        self.assertEqual(self.config, {**original_config, "head_only": True})
        self.plugin.modules.save_settings("common", {"head_only": False})
        self.assertFalse(self.plugin.detector.head_only)
        self.assertEqual(self.config, original_config)

    async def test_head_mode_requires_anime_module_even_if_general_is_enabled(self):
        self.add_model("general")
        self.config.update(enable_general=True, head_only=True)
        self.plugin._configure_pipeline()
        self.assertFalse(self.plugin._has_recognition_module())
        event = types.SimpleNamespace(plain_result=lambda message: message)
        replies = [value async for value in self.plugin._do_kuang(event)]
        self.assertEqual(len(replies), 1)
        self.assertIn("锁头模式需要安装并启用动漫识别模块", replies[0])
        self.assertEqual(
            [value async for value in self.plugin._do_kuang(event, automatic=True)], []
        )
        self.assertFalse(
            self.plugin.detector._backend_plan.primary_backend.auto_download_model
        )


class AutomaticAnimationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await PluginModuleTests.asyncSetUp(self)
        PluginModuleTests.add_model(self, "saliency")
        self.config.update(
            enable_saliency=True,
            enable_auto_animation=True,
            auto_animation_probability=100,
        )
        self.plugin.processor.render_path = Mock(
            return_value=str(self.data / "out.gif")
        )

    async def asyncTearDown(self):
        await PluginModuleTests.asyncTearDown(self)

    def image(self, format="GIF", animated=True):
        from PIL import Image

        path = self.data / f"input-{format}-{animated}.bin"
        first = Image.new("RGB", (8, 8), "red")
        kwargs = {"format": format}
        if animated:
            kwargs.update(
                save_all=True,
                append_images=[Image.new("RGB", (8, 8), "blue")],
                duration=40,
            )
        first.save(path, **kwargs)
        component = self.module.Image()
        component.convert_to_file_path = AsyncMock(return_value=str(path))
        return component

    def event(self, images=(), text="", sender="alice", session="first", enabled=True):
        event = types.SimpleNamespace(
            get_messages=lambda: images,
            get_message_str=lambda: text,
            get_platform_id=lambda: "webchat",
            get_group_id=lambda: "",
            get_sender_id=lambda: sender,
            get_self_id=lambda: "bot",
            session_id=session,
            plain_result=lambda text: ("text", text),
            image_result=lambda path: ("image", path),
            track_temporary_local_file=Mock(),
        )
        if enabled:
            self.plugin.channels.set_enabled(self.plugin.channels.key(event), True)
        return event

    async def collect(self, handler):
        return [result async for result in handler]

    async def test_master_channel_probability_and_missing_module_skip_without_download(
        self,
    ):
        image = self.image()
        event = self.event([image], enabled=False)
        self.assertEqual(await self.collect(self.plugin.auto_kuang(event)), [])
        self.plugin.channels.set_enabled(self.plugin.channels.key(event), True)
        for config in (
            {"enable_auto_animation": False},
            {"enable_auto_animation": True, "auto_animation_probability": 0},
            {"auto_animation_probability": 100, "enable_saliency": False},
        ):
            self.config.update(config)
            self.assertEqual(await self.collect(self.plugin.auto_kuang(event)), [])
        image.convert_to_file_path.assert_not_awaited()

    async def test_only_actual_direct_animations_are_processed(self):
        images = [
            self.image("PNG", False),
            self.image("WEBP", False),
            self.image(),
            self.image("WEBP"),
        ]
        event = self.event(images, text="看这个动图")
        results = await self.collect(self.plugin.auto_kuang(event))
        self.assertEqual(len(results), 2)
        self.assertTrue(all(kind == "image" for kind, _ in results))
        self.assertEqual(self.plugin.processor.render_path.call_count, 2)
        self.assertEqual(event.track_temporary_local_file.call_count, 2)

    async def test_probability_is_sampled_once_per_message_with_exact_boundary(self):
        event = self.event([self.image(), self.image("WEBP")])
        self.config["auto_animation_probability"] = 25
        with patch.object(self.module.random, "random", return_value=0.25) as draw:
            self.assertEqual(await self.collect(self.plugin.auto_kuang(event)), [])
            draw.assert_called_once()
        with patch.object(self.module.random, "random", return_value=0.249) as draw:
            self.assertEqual(len(await self.collect(self.plugin.auto_kuang(event))), 2)
            draw.assert_called_once()

    async def test_manual_command_is_not_duplicated_and_bypasses_automatic_controls(
        self,
    ):
        event = self.event([self.image()], text="框")
        self.assertEqual(await self.collect(self.plugin.auto_kuang(event)), [])
        self.config.update(enable_auto_animation=False, auto_animation_probability=0)
        self.assertEqual(len(await self.collect(self.plugin.kuang(event))), 1)
        self.plugin.processor.render_path.assert_called_once()

    async def test_self_messages_and_quoted_images_are_ignored(self):
        image = self.image()
        self.assertEqual(
            await self.collect(
                self.plugin.auto_kuang(self.event([image], sender="bot"))
            ),
            [],
        )
        reply = self.module.Reply()
        reply.chain = [image]
        self.assertEqual(
            await self.collect(self.plugin.auto_kuang(self.event([reply]))), []
        )
        image.convert_to_file_path.assert_not_awaited()

    async def test_processing_errors_are_silent_and_do_not_discard_other_images(self):
        broken = self.module.Image()
        broken.convert_to_file_path = AsyncMock(side_effect=OSError("download failed"))
        event = self.event([broken, self.image()])
        results = await self.collect(self.plugin.auto_kuang(event))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "image")

    async def test_channel_command_persists_into_new_session_and_plugin_reload(self):
        event = self.event(enabled=False)
        result = await self.collect(self.plugin.kuang_animation(event, "开启"))
        self.assertIn("已开启", result[0][1])
        from core.channel_settings import ChannelSettings

        self.plugin.channels = ChannelSettings(self.data)
        newer = self.event([self.image()], session="new", enabled=False)
        self.assertEqual(len(await self.collect(self.plugin.auto_kuang(newer))), 1)
        await self.collect(self.plugin.kuang_animation(newer, "关闭"))
        self.assertEqual(await self.collect(self.plugin.auto_kuang(event)), [])
        self.assertFalse(
            self.plugin.channels.is_enabled(self.plugin.channels.key(newer))
        )

    async def test_channel_enable_explains_disabled_master_and_zero_probability(self):
        self.config.update(enable_auto_animation=False, auto_animation_probability=0)
        event = self.event(enabled=False)
        result = await self.collect(self.plugin.kuang_animation(event, "开启"))
        self.assertIn("总开关：已关闭", result[0][1])
        self.assertIn("回复概率：0%", result[0][1])
        self.assertIn("通用设置", result[0][1])

    async def test_invalid_probability_disables_automatic_trigger(self):
        event = self.event([self.image()])
        for value in (-1, 101, float("nan"), float("inf"), "bad", None):
            self.config["auto_animation_probability"] = value
            self.assertEqual(await self.collect(self.plugin.auto_kuang(event)), [])

    async def test_head_mode_automatic_trigger_respects_anime_module_availability(self):
        self.config["head_only"] = True
        image = self.image("WEBP")
        event = self.event([image])
        self.assertEqual(await self.collect(self.plugin.auto_kuang(event)), [])
        image.convert_to_file_path.assert_not_awaited()
        PluginModuleTests.add_model(self, "anime")
        self.config["enable_anime_fallback"] = True
        self.assertEqual(len(await self.collect(self.plugin.auto_kuang(event))), 1)
