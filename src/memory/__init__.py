"""按需导出记忆组件，避免导入包时创建全局管理器。"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ChatEvent": (".types", "ChatEvent"),
    "EventType": (".types", "EventType"),
    "GeneralMemorySystem": (".general_memory_system", "GeneralMemorySystem"),
    "Memory": (".types", "Memory"),
    "MemoryConfig": (".types", "MemoryConfig"),
    "MemoryContextManager": (".context", "MemoryContextManager"),
    "MemoryManager": (".server", "MemoryManager"),
    "memory_manager": (".server", "memory_manager"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """首次访问导出项时再加载对应模块。"""
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"模块 {__name__!r} 不包含属性 {name!r}") from error
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
