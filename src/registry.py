from importlib import import_module

from mmengine.registry import Registry

MEMORY_SYSTEM = Registry("memory_system", locations=["src.memory"])
TOOL = Registry("tool", locations=["src.tool"])
ENVIRONMENT = Registry("environment", locations=["src.environment"])
AGENT = Registry("agent", locations=["src.agent"])
PROMPT = Registry("prompt", locations=["src.prompt"])
SKILL = Registry("skill", locations=["src.skill"])


_BUILTIN_COMPONENT_MODULES = {
    "agent": ("src.agent.tool_calling_agent",),
    "environment": (
        "src.environment.faiss_environment",
        "src.environment.file_system_environment",
    ),
    "memory": ("src.memory.general_memory_system",),
    "prompt": ("src.prompt.template",),
    "tool": (
        "src.tool.default_tools",
        "src.tool.workflow_tools",
    ),
}


def load_builtin_components(component_type: str) -> None:
    """加载内置组件模块，使装饰器在读取注册表前完成注册。"""
    try:
        module_names = _BUILTIN_COMPONENT_MODULES[component_type]
    except KeyError as error:
        raise ValueError(f"未知的组件类型：{component_type}") from error

    for module_name in module_names:
        import_module(module_name)
