import asyncio
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.module_manager import MODULES, ModuleManager
from core.module_runtime import RestartRequired, requirements_for


class Config(dict):
    def save_config(self):
        self.saves = getattr(self, "saves", 0) + 1


class ModuleManagerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.data = Path(self.temp.name)
        self.available = set()
        self.installs = []
        self.downloads = []
        self.validations = []
        self.config = Config()
        self.changes = []
        self.manager = self.make_manager()

    async def asyncTearDown(self):
        await self.manager.close()
        self.temp.cleanup()

    def make_manager(self):
        return ModuleManager(
            self.data,
            self.config,
            lambda: self.changes.append(
                [key for key in MODULES if self.manager.is_enabled(key)]
            ),
            probe=lambda reqs: [r for r in reqs if r not in self.available],
            installer=self.install,
            downloader=self.download,
            validator=self.validate,
        )

    async def install(self, requirements, log):
        self.installs.append(requirements)
        log("Collecting packages")
        await asyncio.sleep(0)
        self.available.update(requirements)

    def download(self, url, destination, progress, stopped):
        self.downloads.append(url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"valid model")
        progress(11, 11, 500)

    def validate(self, module_id, path):
        self.validations.append((module_id, path))
        if path is not None and path.read_bytes() != b"valid model":
            raise ValueError("invalid model")

    async def wait(self):
        await asyncio.gather(*list(self.manager._tasks.values()))

    def item(self, key):
        return next(m for m in self.manager.snapshot()["modules"] if m["id"] == key)

    async def test_empty_install_never_downloads_or_installs(self):
        self.assertFalse(self.manager.snapshot()["busy"])
        self.assertEqual(self.installs, [])
        self.assertEqual(self.downloads, [])
        self.assertTrue(
            all(
                m["phase"] == "not_installed"
                for m in self.manager.snapshot()["modules"]
            )
        )

    async def test_foreground_is_independent_and_enables_only_after_validation(self):
        self.manager.start_install("saliency")
        self.assertFalse(self.manager.is_enabled("saliency"))
        await self.wait()
        self.assertTrue(self.manager.is_enabled("saliency"))
        self.assertFalse(self.manager.is_enabled("general"))
        self.assertNotIn("opencv-python-headless>=4.13.0,<5", self.available)
        self.assertEqual(self.item("saliency")["downloaded_bytes"], 11)
        self.assertIsInstance(self.item("saliency")["updated_at_ms"], int)
        self.assertEqual(self.changes[-1], ["saliency"])

    async def test_queue_reuses_shared_dependencies_and_duplicate_clicks(self):
        for key in ("general", "anime", "saliency", "gif", "anime"):
            self.manager.start_install(key)
        await self.wait()
        self.assertEqual(len(self.downloads), 3)
        self.assertEqual(len(self.installs), 2)
        self.assertEqual(self.installs[1], ("opencv-python-headless>=4.13.0,<5",))
        self.assertTrue(all(self.manager.is_enabled(key) for key in MODULES))

    async def test_disabled_module_reuses_model_and_dependencies(self):
        self.manager.start_install("anime")
        await self.wait()
        path = self.manager.model_path("anime")
        self.manager.disable("anime")
        self.assertTrue(path.exists())
        self.assertFalse(self.manager.is_enabled("anime"))
        self.manager.start_install("anime", install=False)
        await self.wait()
        self.assertTrue(self.manager.is_enabled("anime"))
        self.assertEqual(len(self.installs), 1)
        self.assertEqual(len(self.downloads), 1)

    async def test_download_failure_can_retry_without_reinstalling_dependencies(self):
        def fail(*args):
            raise ConnectionError("network unavailable")

        self.manager.downloader = fail
        self.manager.start_install("general")
        await self.wait()
        self.assertEqual(self.item("general")["phase"], "error")
        self.assertFalse(self.manager.is_enabled("general"))
        self.manager.downloader = self.download
        self.manager.start_install("general")
        await self.wait()
        self.assertTrue(self.manager.is_enabled("general"))
        self.assertEqual(len(self.installs), 1)

    async def test_invalid_download_is_never_published_as_installed(self):
        self.manager.validator = lambda *args: (_ for _ in ()).throw(
            ValueError("bad onnx")
        )
        self.manager.start_install("anime")
        await self.wait()
        self.assertFalse(self.manager.model_path("anime").exists())
        self.assertFalse(self.item("anime")["installed"])
        self.assertEqual(self.item("anime")["phase"], "error")

    async def test_corrupt_existing_file_is_only_replaced_after_validation(self):
        path = self.manager.model_path("general")
        path.parent.mkdir(parents=True)
        path.write_bytes(b"corrupt")
        self.manager.start_install("general")
        await self.wait()
        self.assertEqual(path.read_bytes(), b"valid model")
        self.assertTrue(self.manager.is_enabled("general"))

    async def test_explicitly_disabled_download_is_respected(self):
        self.config["saliency_auto_download_model"] = False
        self.manager.start_install("saliency")
        await self.wait()
        self.assertEqual(self.downloads, [])
        self.assertEqual(self.item("saliency")["phase"], "error")

    async def test_enable_does_not_silently_install_missing_dependencies(self):
        self.manager.start_install("general", install=False)
        await self.wait()
        self.assertEqual(self.installs, [])
        self.assertEqual(self.item("general")["phase"], "error")

    async def test_restart_required_is_visible_and_does_not_enable(self):
        def needs_restart(*args):
            raise RestartRequired("restart needed")

        self.manager.validator = needs_restart
        self.manager.start_install("gif")
        await self.wait()
        self.assertEqual(self.item("gif")["phase"], "restart_required")
        self.assertFalse(self.manager.is_enabled("gif"))

    async def test_reload_recovers_interrupted_state_without_automatic_install(self):
        path = self.data / "modules.json"
        path.write_text(
            json.dumps({"version": 1, "modules": {"general": {"phase": "downloading"}}})
        )
        self.manager = self.make_manager()
        self.assertEqual(self.item("general")["phase"], "interrupted")
        self.assertEqual(self.installs, [])

    async def test_close_waits_for_dependency_step_and_does_not_enable_after_reload(
        self,
    ):
        started, finish = asyncio.Event(), asyncio.Event()

        async def installing(reqs, log):
            started.set()
            await finish.wait()
            self.available.update(reqs)

        self.manager.installer = installing
        self.manager.start_install("gif")
        await started.wait()
        closing = asyncio.create_task(self.manager.close())
        await asyncio.sleep(0)
        self.assertFalse(closing.done())
        finish.set()
        await closing
        self.assertFalse(self.manager.is_enabled("gif"))
        self.assertEqual(self.item("gif")["phase"], "interrupted")

    async def test_legacy_models_are_adopted_once_and_disabling_survives_reload(self):
        (self.data / "modules.json").unlink()
        self.available.update(requirements_for("general"))
        self.available.update(requirements_for("gif"))
        self.config.update(enable_saliency=True, enable_anime_fallback=False)
        for key in ("general", "saliency", "anime"):
            path = self.manager.model_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"valid model")
        self.manager = self.make_manager()
        self.assertTrue(self.manager.is_enabled("general"))
        self.assertTrue(self.manager.is_enabled("saliency"))
        self.assertTrue(self.manager.is_enabled("gif"))
        self.assertFalse(self.manager.is_enabled("anime"))
        self.manager.disable("general")
        self.manager = self.make_manager()
        self.assertFalse(self.manager.is_enabled("general"))
        self.assertEqual(self.installs, [])

    async def test_download_page_cannot_change_functional_settings_or_enable_modules(
        self,
    ):
        self.manager.start_install("general")
        await self.wait()
        before = dict(self.config)
        for module, values in (
            ("general", {"confidence_threshold": 0.4}),
            ("general", {"enable_general": False}),
            ("anime", {"anime_model_input_size": 320}),
            ("saliency", {"saliency_threshold": 0.3}),
            ("gif", {"gif_detection_interval": 8}),
            ("common", {"head_only": True}),
        ):
            with self.assertRaisesRegex(ValueError, "插件配置"):
                self.manager.save_settings(module, values)
        self.assertEqual(self.config, before)

    async def test_download_options_can_be_saved_before_first_install_and_are_used(
        self,
    ):
        self.config.update(head_only=True, auto_animation_probability=37)
        model = self.data / "custom" / "foreground.onnx"
        values = {
            "saliency_model_path": str(model),
            "saliency_model_url": "https://mirror.example/u2netp.onnx",
            "saliency_auto_download_model": True,
        }
        self.manager.save_settings("saliency", values)
        self.assertEqual(self.installs, [])
        self.assertEqual(self.downloads, [])
        self.manager = self.make_manager()
        self.manager.start_install("saliency")
        await self.wait()
        self.assertEqual(self.downloads, [values["saliency_model_url"]])
        self.assertEqual(model.read_bytes(), b"valid model")
        self.assertTrue(self.manager.is_enabled("saliency"))
        self.assertTrue(self.config["head_only"])
        self.assertEqual(self.config["auto_animation_probability"], 37)

    async def test_existing_model_path_and_install_state_survive_reload(self):
        self.manager.start_install("general")
        await self.wait()
        self.manager = self.make_manager()
        self.assertTrue(self.manager.is_enabled("general"))
        self.assertIsInstance(self.item("general")["updated_at_ms"], int)

    async def test_changing_model_path_requires_validation_before_reenable(self):
        self.manager.start_install("general")
        await self.wait()
        other = self.data / "different.onnx"
        other.write_bytes(b"valid model")
        self.manager.save_settings("general", {"model_path": str(other)})
        self.assertFalse(self.manager.is_enabled("general"))
        self.manager.start_install("general", install=False)
        await self.wait()
        self.assertTrue(self.manager.is_enabled("general"))

    async def test_bad_state_does_not_reenable_disabled_modules(self):
        (self.data / "modules.json").write_text("[]")
        self.manager = self.make_manager()
        self.assertTrue(all(not self.manager.is_enabled(key) for key in MODULES))

    async def test_download_options_reject_invalid_types_without_partial_save(self):
        before = dict(self.config)
        for values in (
            {"model_url": 3},
            {"model_path": None},
            {
                "model_url": "https://mirror.example/model.onnx",
                "auto_download_model": "false",
            },
        ):
            with self.assertRaises(ValueError):
                self.manager.save_settings("general", values)
            self.assertEqual(self.config, before)

    async def test_download_fields_do_not_appear_in_native_configuration(self):
        schema = self.manager._schema
        for module, spec in MODULES.items():
            self.assertTrue(schema[spec["flag"]]["invisible"])
            fields = self.manager.settings(module)
            expected = (
                {
                    spec["prefix"] + suffix
                    for suffix in ("model_path", "model_url", "auto_download_model")
                }
                if spec["file"]
                else set()
            )
            self.assertEqual(set(fields), expected)
            self.assertTrue(all(schema[key]["invisible"] for key in fields))
        for key in (
            "head_only",
            "auto_animation_probability",
            "saliency_threshold",
            "gif_detection_interval",
        ):
            self.assertFalse(schema[key].get("invisible", False))


class EmptyCoreTests(unittest.TestCase):
    def test_plugin_loads_without_optional_site_packages(self):
        root = Path(__file__).resolve().parents[1]
        script = """
import json, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, 'tests')
from test_plugin_modules import load_plugin
with tempfile.TemporaryDirectory() as directory:
    main = load_plugin(Path(directory))
    schema = json.loads(Path('_conf_schema.json').read_text(encoding='utf-8'))
    config = {key: value['default'] for key, value in schema.items()}
    plugin = main.KuangPlugin(SimpleNamespace(), config)
    assert not {'numpy', 'onnxruntime', 'PIL', 'cv2'} & sys.modules.keys()
    assert not plugin.modules.snapshot()['busy']
"""
        result = subprocess.run(
            [sys.executable, "-S", "-c", script],
            cwd=root,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_core_imports_without_any_site_packages(self):
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                "import sys; import core; import core.module_manager; "
                "assert not {'numpy', 'onnxruntime', 'PIL', 'cv2'} & sys.modules.keys()",
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_other_opencv_distributions_are_reused(self):
        from core.module_runtime import missing_requirements

        def version(name):
            if name == "opencv-python":
                return "4.13.0"
            from importlib.metadata import PackageNotFoundError

            raise PackageNotFoundError(name)

        with (
            patch(
                "core.module_runtime.importlib.metadata.version", side_effect=version
            ),
            patch(
                "core.module_runtime.importlib.util.find_spec", return_value=object()
            ),
        ):
            self.assertEqual(
                missing_requirements(("opencv-python-headless>=4.13.0,<5",)), []
            )
