import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from core.channel_settings import ChannelSettings


def event(platform="qq-bot-a", group="group-a", sender="alice", session="old"):
    return SimpleNamespace(
        get_platform_id=lambda: platform,
        get_group_id=lambda: group,
        get_sender_id=lambda: sender,
        session_id=session,
    )


class ChannelSettingsTests(unittest.TestCase):
    def test_group_members_and_new_sessions_share_setting_after_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = ChannelSettings(Path(directory))
            first = settings.key(event())
            self.assertFalse(settings.is_enabled(first))
            settings.set_enabled(first, True)
            reloaded = ChannelSettings(Path(directory))
            second = reloaded.key(event(sender="bob", session="new-conversation"))
            self.assertEqual(first, second)
            self.assertTrue(reloaded.is_enabled(second))
            state = json.loads(settings.path.read_text())
            self.assertIsInstance(state["channels"][first]["updated_at_ms"], int)
            reloaded.set_enabled(second, False)
            self.assertFalse(ChannelSettings(Path(directory)).is_enabled(first))

    def test_webchat_new_conversations_share_user_but_not_other_users(self):
        first = ChannelSettings.key(event(platform="webchat", group="", session="a"))
        second = ChannelSettings.key(event(platform="webchat", group="", session="b"))
        other = ChannelSettings.key(event(platform="webchat", group="", sender="bob"))
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)

    def test_platform_group_and_private_channel_settings_are_isolated(self):
        keys = {
            ChannelSettings.key(event()),
            ChannelSettings.key(event(platform="qq-bot-b")),
            ChannelSettings.key(event(group="group-b")),
            ChannelSettings.key(event(group="", sender="group-a")),
        }
        self.assertEqual(len(keys), 4)
        self.assertIsNone(ChannelSettings.key(event(platform="")))
        self.assertIsNone(ChannelSettings.key(event(group="", sender="")))

    def test_failed_write_does_not_change_live_setting(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = ChannelSettings(Path(directory))
            key = settings.key(event())
            with patch.object(Path, "replace", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    settings.set_enabled(key, True)
            self.assertFalse(settings.is_enabled(key))

    def test_bad_state_does_not_enable_channels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "auto_animation_channels.json"
            for value in ("[]", '{"channels": []}', "not-json"):
                path.write_text(value)
                self.assertFalse(ChannelSettings(Path(directory)).is_enabled("x"))
