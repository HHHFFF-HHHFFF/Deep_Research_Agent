"""按需导出环境组件，避免导入包时初始化外部资源。"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "FaissEnvironment": (".faiss_environment", "FaissEnvironment"),
    "FileSystemEnvironment": (".file_system_environment", "FileSystemEnvironment"),
    "ecp": (".server", "ecp"),
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
