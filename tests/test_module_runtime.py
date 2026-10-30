import asyncio
import logging
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from core.module_runtime import (
    RestartRequired,
    check_loaded_versions,
    install_requirements,
    restore_dependencies,
    validate_module,
)


class RuntimeValidationTests(unittest.TestCase):
    def test_loaded_numpy_needs_restart_even_when_old_version_meets_minimum(self):
        with (
            patch.dict(
                sys.modules, {"numpy": types.SimpleNamespace(__version__="1.26.4")}
            ),
            patch(
                "core.module_runtime.importlib.metadata.version", return_value="2.0.0"
            ),
        ):
            with self.assertRaises(RestartRequired):
                check_loaded_versions("gif")

    def test_native_loader_error_is_not_reported_as_damaged_model(self):
        with (
            patch("core.module_runtime.check_loaded_versions"),
            patch(
                "core.module_runtime.importlib.import_module",
                side_effect=ImportError("native library cannot load"),
            ),
        ):
            with self.assertRaisesRegex(RestartRequired, "重启"):
                validate_module("general", Path("not-a-model.onnx"))


class HostInstallerTests(unittest.IsolatedAsyncioTestCase):
    async def test_restore_uses_host_import_policy_without_installing(self):
        calls = []
        fake = types.ModuleType("astrbot.core")
        fake.pip_installer = types.SimpleNamespace(
            prefer_installed_dependencies=calls.append
        )
        with patch.dict(sys.modules, {"astrbot.core": fake}):
            restore_dependencies("anime")
        self.assertEqual(len(calls), 1)
        self.assertEqual(Path(calls[0]).name, "runtime.txt")

    async def test_host_installer_receives_specs_and_streams_logs(self):
        calls, logs = [], []
        host_logger = logging.getLogger("astrbot")
        old_level, old_handlers = host_logger.level, list(host_logger.handlers)

        async def install(**kwargs):
            calls.append(kwargs)
            host_logger.info("Installing numpy")
            await asyncio.to_thread(host_logger.info, "Successfully installed numpy")

        fake = types.ModuleType("astrbot.core")
        fake.pip_installer = types.SimpleNamespace(install=install)
        try:
            host_logger.setLevel(logging.INFO)
            with patch.dict(sys.modules, {"astrbot.core": fake}):
                await install_requirements(
                    ("numpy>=1.26.0", "Pillow>=10.0.0"), logs.append
                )
                await asyncio.sleep(0)
            self.assertEqual(calls, [{"package_name": "numpy>=1.26.0 Pillow>=10.0.0"}])
            self.assertEqual(logs, ["Installing numpy", "Successfully installed numpy"])
            self.assertEqual(host_logger.handlers, old_handlers)
        finally:
            host_logger.setLevel(old_level)

    async def test_host_failure_is_propagated_and_log_handler_is_removed(self):
        async def fail(**kwargs):
            raise RuntimeError("dependency conflict")

        fake = types.ModuleType("astrbot.core")
        fake.pip_installer = types.SimpleNamespace(install=fail)
        logger = logging.getLogger("astrbot")
        before = list(logger.handlers)
        with patch.dict(sys.modules, {"astrbot.core": fake}):
            with self.assertRaisesRegex(RuntimeError, "dependency conflict"):
                await install_requirements(("numpy>=1.26.0",), lambda message: None)
        self.assertEqual(logger.handlers, before)
