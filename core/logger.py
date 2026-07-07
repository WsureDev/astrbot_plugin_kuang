from __future__ import annotations

import logging
from typing import Protocol


class LoggerProtocol(Protocol):
    def debug(self, msg: object, *args: object, **kwargs: object) -> None: ...

    def info(self, msg: object, *args: object, **kwargs: object) -> None: ...

    def warning(self, msg: object, *args: object, **kwargs: object) -> None: ...

    def error(self, msg: object, *args: object, **kwargs: object) -> None: ...

    def critical(self, msg: object, *args: object, **kwargs: object) -> None: ...

    def exception(self, msg: object, *args: object, **kwargs: object) -> None: ...

    def isEnabledFor(self, level: int) -> bool: ...


_injected: LoggerProtocol | None = None
_level_override: int | None = None
_astrbot_named_loggers: dict[str, logging.Logger] = {}


def configure(logger: LoggerProtocol) -> None:
    global _injected
    _injected = logger


def set_debug_mode(enabled: bool) -> None:
    global _level_override
    _level_override = logging.DEBUG if enabled else None


def _resolve_target_logger(name: str) -> logging.Logger:
    cached = _astrbot_named_loggers.get(name)
    if cached is not None:
        _sync_logger_level(cached)
        return cached

    target = _build_target_logger(name)
    _astrbot_named_loggers[name] = target
    _sync_logger_level(target)
    return target


def _build_target_logger(name: str) -> logging.Logger:
    if _injected is None:
        return logging.getLogger(name)

    try:
        from astrbot.core import LogManager
    except Exception:
        return logging.getLogger(name)

    return LogManager.GetLogger(log_name=name)


def _sync_logger_level(logger: logging.Logger) -> None:
    desired_level = _level_override
    if desired_level is None and isinstance(_injected, logging.Logger):
        desired_level = _injected.getEffectiveLevel()
    if desired_level is None:
        return
    if logger.level != desired_level:
        logger.setLevel(desired_level)


class _LoggerProxy:
    __slots__ = ("_name", "_fallback")

    def __init__(self, name: str) -> None:
        self._name = name
        self._fallback = logging.getLogger(name)

    def _target(self) -> LoggerProtocol:
        if _injected is None:
            return self._fallback
        return _resolve_target_logger(self._name)

    def _call(self, method_name: str, msg: object, *args: object, **kwargs: object) -> None:
        target = self._target()
        method = getattr(target, method_name)
        if isinstance(target, logging.Logger):
            stacklevel = int(kwargs.pop("stacklevel", 1)) + 2
            method(msg, *args, stacklevel=stacklevel, **kwargs)
            return
        method(msg, *args, **kwargs)

    def debug(self, msg: object, *args: object, **kwargs: object) -> None:
        self._call("debug", msg, *args, **kwargs)

    def info(self, msg: object, *args: object, **kwargs: object) -> None:
        self._call("info", msg, *args, **kwargs)

    def warning(self, msg: object, *args: object, **kwargs: object) -> None:
        self._call("warning", msg, *args, **kwargs)

    def error(self, msg: object, *args: object, **kwargs: object) -> None:
        self._call("error", msg, *args, **kwargs)

    def critical(self, msg: object, *args: object, **kwargs: object) -> None:
        self._call("critical", msg, *args, **kwargs)

    def exception(self, msg: object, *args: object, **kwargs: object) -> None:
        if "exc_info" not in kwargs:
            kwargs["exc_info"] = True
        self._call("exception", msg, *args, **kwargs)

    def isEnabledFor(self, level: int) -> bool:
        target = self._target()
        if hasattr(target, "isEnabledFor"):
            return target.isEnabledFor(level)
        return self._fallback.isEnabledFor(level)


def get_logger(name: str) -> _LoggerProxy:
    return _LoggerProxy(name)
