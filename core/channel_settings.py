"""Persistent channel preferences, independent of conversation sessions."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .logger import get_logger


class ChannelSettings:
    def __init__(self, data_dir: Path):
        self.path = Path(data_dir) / "auto_animation_channels.json"
        self._channels = {}
        if self.path.exists():
            try:
                state = json.loads(self.path.read_text(encoding="utf-8"))
                channels = state["channels"]
                if not isinstance(channels, dict):
                    raise ValueError("channels 必须是 JSON 对象")
                self._channels = channels
            except (OSError, ValueError, KeyError, TypeError) as exc:
                get_logger(__name__).warning("读取自动动图渠道设置失败：%s", exc)

    @staticmethod
    def key(event) -> str | None:
        platform = str(event.get_platform_id() or "").strip()
        group = str(event.get_group_id() or "").strip()
        sender = str(event.get_sender_id() or "").strip()
        target = group or sender
        if not platform or not target:
            return None
        # JSON arrays avoid separator collisions in adapter/channel identifiers.
        return json.dumps(
            [platform, "group" if group else "private", target],
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def is_enabled(self, key: str | None) -> bool:
        record = self._channels.get(key, {})
        return isinstance(record, dict) and record.get("enabled") is True

    def set_enabled(self, key: str, enabled: bool) -> None:
        channels = dict(self._channels)
        channels[key] = {
            "enabled": enabled,
            "updated_at_ms": time.time_ns() // 1_000_000,
        }
        staging = self.path.with_suffix(".json.tmp")
        staging.write_text(
            json.dumps({"version": 1, "channels": channels}, ensure_ascii=False),
            encoding="utf-8",
        )
        staging.replace(self.path)
        self._channels = channels
