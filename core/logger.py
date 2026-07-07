from __future__ import annotations

import logging
from typing import Optional, Protocol


class LoggerProtocol(Protocol):
    def debug(self, msg: object, *args: object, **kwargs: object) -> None: ...
    def info(self, msg: object, *args: object, **kwargs: object) -> None: ...
    def warning(self, msg: object, *args: object, **kwargs: object) -> None: ...
    def error(self, msg: object, *args: object, **kwargs: object) -> None: ...
    def critical(self, msg: object, *args: object, **kwargs: object) -> None: ...
    def exception(self, msg: object, *args: object, **kwargs: object) -> None: ...
    def isEnabledFor(self, level: int) -> bool: ...


_injected: Optional[LoggerProtocol] = None


def configure(logger: LoggerProtocol) -> None:
    global _injected
    _injected = logger


class _LoggerProxy:
    __slots__ = ("_fallback",)

    def __init__(self, name: str) -> None:
        self._fallback = logging.getLogger(name)

    def _target(self) -> LoggerProtocol:
        return _injected if _injected is not None else self._fallback  # type: ignore[return-value]

    def debug(self, msg: object, *args: object, **kwargs: object) -> None:
        self._target().debug(msg, *args, **kwargs)

    def info(self, msg: object, *args: object, **kwargs: object) -> None:
        self._target().info(msg, *args, **kwargs)

    def warning(self, msg: object, *args: object, **kwargs: object) -> None:
        self._target().warning(msg, *args, **kwargs)

    def error(self, msg: object, *args: object, **kwargs: object) -> None:
        self._target().error(msg, *args, **kwargs)

    def critical(self, msg: object, *args: object, **kwargs: object) -> None:
        self._target().critical(msg, *args, **kwargs)

    def exception(self, msg: object, *args: object, **kwargs: object) -> None:
        self._target().exception(msg, *args, **kwargs)

    def isEnabledFor(self, level: int) -> bool:
        t = self._target()
        if hasattr(t, "isEnabledFor"):
            return t.isEnabledFor(level)
        return True


def get_logger(name: str) -> "_LoggerProxy":
    return _LoggerProxy(name)
