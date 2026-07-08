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
_debug_enabled: bool = False
_plugin_name: str = ""


def configure(
    logger: LoggerProtocol,
    *,
    plugin_name: Optional[str] = None,
    debug_enabled: Optional[bool] = None,
) -> None:
    """注入运行时 logger，并可选配置插件名和 debug 开关。

    该函数是本模块唯一的外部接线点，保持零业务依赖，可跨插件复用。
    """
    global _injected, _plugin_name, _debug_enabled
    _injected = logger
    if plugin_name is not None:
        _plugin_name = str(plugin_name)
    if debug_enabled is not None:
        _debug_enabled = bool(debug_enabled)


def set_debug_enabled(enabled: bool) -> None:
    global _debug_enabled
    _debug_enabled = bool(enabled)


def set_plugin_name(name: str) -> None:
    global _plugin_name
    _plugin_name = str(name or "")


def _debug_native_ok(target: object) -> bool:
    """目标 logger 是否原生支持 DEBUG 级别落地（如本地测试环境或未来 #9186 落地后）。"""
    return hasattr(target, "isEnabledFor") and target.isEnabledFor(logging.DEBUG)


def _mark_debug(msg: object) -> object:
    """为 debug 消息加上 [DEBUG][插件名] 前缀，仅在 workaround 降级路径使用。"""
    prefix = f"[DEBUG][{_plugin_name}]" if _plugin_name else "[DEBUG]"
    if isinstance(msg, str):
        return f"{prefix} {msg}"
    return f"{prefix} {msg}"


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
        target = self._target()
        # 优先：目标原生支持 DEBUG 落地（本地标准 logging 配了 DEBUG，
        # 或未来 issue #9186 落地后插件有可独立调级别的 logger）→ 走真正的 debug
        if _debug_native_ok(target):
            self._call("debug", msg, *args, **kwargs)
            return
        # workaround：astrbot 全局无法单独开 DEBUG。
        # 开关开启时以 info 降级落地，前缀 [DEBUG][插件名]；开关关闭时直接 return，零输出。
        if _debug_enabled:
            self._call("info", _mark_debug(msg), *args, **kwargs)

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
        # 当 debug 开关开启时，对 DEBUG 级别也返回 True，以便业务层短路判断生效
        if level <= logging.DEBUG and _debug_enabled:
            return True
        t = self._target()
        if hasattr(t, "isEnabledFor"):
            return t.isEnabledFor(level)
        return True


def get_logger(name: str) -> "_LoggerProxy":
    return _LoggerProxy(name)
