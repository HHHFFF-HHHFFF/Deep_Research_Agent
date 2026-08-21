# ruff: noqa: F401
# MMEngine 会把 read_base 中导入的变量合并为场景配置字段。
from mmengine.config import read_base

with read_base():
    from .agents.tool_calling import tool_calling_agent
    from .base import (
        embedding_fallback_models,
        embedding_model_id,
        embedding_model_name,
        embedding_provider,
        fallback_models,
        model_id,
        model_name,
        model_provider,
    )
    from .environments.file_system import environment as file_system_environment
    from .memory.general_memory_system import memory_system as general_memory_system
    from .tools.deep_analyzer import deep_analyzer_tool
    from .tools.deep_researcher import deep_researcher_tool
    from .tools.mdify import mdify_tool
    from .tools.reporter import reporter_tool
    from .tools.todo import todo_tool
    from .tools.web_searcher import web_searcher_tool

tag = "tool_calling_agent"
workdir = f"workdir/{tag}"
log_path = "agent.log"

use_local_proxy = True
version = "0.1.0"
# `model_name` 由基础配置统一生成，不在场景配置中写死厂商。

env_names = ["file_system"]
memory_names = [
    "general_memory_system",
]
agent_names = ["tool_calling"]
tool_names = [
    "done",
    "todo",
    "read",
    "edit",
    "mdify",
    "web_searcher",
    "deep_analyzer",
    "deep_researcher",
    "reporter",
]
skill_names: list[str] = []

# 配置相关参数。
mdify_tool.update(
    base_dir="tool/mdify",
)
todo_tool.update(
    base_dir="tool/todo",
    require_grad=False,
)
# 配置相关参数。
deep_researcher_tool.update(
    model_name=model_name,
    base_dir="tool/deep_researcher",
)

web_searcher_tool.update(
    model_name=model_name,
)

# 配置相关参数。
deep_analyzer_tool.update(
    model_name=model_name,
    file_model_name=model_name,
    base_dir="tool/deep_analyzer",
    require_grad=False,
)

# 配置相关参数。
reporter_tool.update(
    base_dir="tool/reporter",
    model_name=model_name,
    require_grad=False,
)
# 配置相关参数。
general_memory_system.update(
    base_dir="memory/general_memory_system",
    model_name=model_name,
    max_summaries=10,
    max_insights=10,
    require_grad=False,
)

# 配置相关参数。
file_system_environment.update(
    base_dir="environment/file_system",
    require_grad=False,
)

# 配置相关参数。
tool_calling_agent.update(
    workdir=workdir,
    model_name=model_name,
    memory_name=memory_names[0],
    require_grad=False,
    use_memory=True,
)
