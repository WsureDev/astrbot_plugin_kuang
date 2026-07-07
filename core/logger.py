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

    def _call(self, method_name: str, msg: object, *args: object, **kwargs: object) -> None:
        target = self._target()
        method = getattr(target, method_name)
        # 目标为标准 logging.Logger 时注入 stacklevel，让日志来源跳过本代理层，
        # 指向真正的调用处（如 core/renderer.py:42），修复来源被塌缩成 core.logger 的问题。
        if isinstance(target, logging.Logger):
            stacklevel = int(kwargs.pop("stacklevel", 1)) + 2
            try:
                method(msg, *args, stacklevel=stacklevel, **kwargs)
                return
            except TypeError:
                kwargs.pop("stacklevel", None)
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
        t = self._target()
        if hasattr(t, "isEnabledFor"):
            return t.isEnabledFor(level)
        return True


def get_logger(name: str) -> "_LoggerProxy":
    return _LoggerProxy(name)
