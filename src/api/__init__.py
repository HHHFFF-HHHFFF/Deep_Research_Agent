"""按需导出稳定 Web 后端入口，避免导入数据模型时创建应用。"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ApiSettings": (".app", "ApiSettings"),
    "app": (".app", "app"),
    "create_app": (".app", "create_app"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """首次访问公共入口时再导入 FastAPI 应用模块。"""
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"模块 {__name__!r} 不包含属性 {name!r}") from error
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
