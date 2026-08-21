"""内置组件注册的离线回归测试。"""

from src.registry import (
    AGENT,
    ENVIRONMENT,
    MEMORY_SYSTEM,
    PROMPT,
    TOOL,
    load_builtin_components,
)


def test_all_builtin_component_modules_populate_registries() -> None:
    """按需导出不能阻止运行时发现研究所需的内置组件。"""
    for component_type in ("prompt", "memory", "tool", "environment", "agent"):
        load_builtin_components(component_type)

    assert "ToolCallingAgent" in AGENT.module_dict
    assert {"FaissEnvironment", "FileSystemEnvironment"} <= set(ENVIRONMENT.module_dict)
    assert "GeneralMemorySystem" in MEMORY_SYSTEM.module_dict
    assert {"ToolCallingAgentMessagePrompt", "ToolCallingSystemPrompt"} <= set(
        PROMPT.module_dict
    )
    assert {
        "DeepAnalyzerTool",
        "DeepResearcherTool",
        "DoneTool",
        "FileEditorTool",
        "FileReaderTool",
        "MdifyTool",
        "ReporterTool",
        "TodoTool",
        "WebSearcherTool",
    } <= set(TOOL.module_dict)
