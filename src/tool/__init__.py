"""按需导出工具组件，避免导入包时初始化全部工具依赖。"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "DeepAnalyzerTool": (".workflow_tools", "DeepAnalyzerTool"),
    "DeepResearcherTool": (".workflow_tools", "DeepResearcherTool"),
    "DoneTool": (".default_tools", "DoneTool"),
    "FileEditorTool": (".default_tools", "FileEditorTool"),
    "FileReaderTool": (".default_tools", "FileReaderTool"),
    "MdifyTool": (".default_tools", "MdifyTool"),
    "ReporterTool": (".workflow_tools", "ReporterTool"),
    "TCPServer": (".server", "TCPServer"),
    "TodoTool": (".workflow_tools", "TodoTool"),
    "Tool": (".types", "Tool"),
    "ToolContextManager": (".context", "ToolContextManager"),
    "ToolResponse": (".types", "ToolResponse"),
    "WebFetcherTool": (".default_tools", "WebFetcherTool"),
    "WebSearcherTool": (".default_tools", "WebSearcherTool"),
    "tcp": (".server", "tcp"),
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
