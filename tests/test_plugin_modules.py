import importlib.util
import json
import logging
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

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
