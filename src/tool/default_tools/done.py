"""提供done相关实现。"""

from typing import Any

from pydantic import Field

from src.registry import TOOL
from src.tool.types import Tool, ToolExtra, ToolResponse

_DONE_TOOL_DESCRIPTION = """Done tool for indicating that the task has been completed.
Use this tool to signal that a task or subtask has been finished.
Provide the `result` and `reasoning` of the task in the result and reasoning parameters.

Args:
- result (str): The result of the task completion.
- reasoning (str): The analysis or explanation of the task completion.

Example: {"name": "done", "args": {"reasoning": "The task has been completed successfully.","result": "The task has been completed."}}.
"""


@TOOL.register_module(force=True)
class DoneTool(Tool):
    """定义 `DoneTool`，封装相关数据与行为。"""

    name: str = "done"
    description: str = _DONE_TOOL_DESCRIPTION
    metadata: dict[str, Any] = Field(default={}, description="The metadata of the tool")
    require_grad: bool = Field(
        default=False, description="Whether the tool requires gradients"
    )

    def __init__(self, require_grad: bool = False, **kwargs):
        """初始化实例。"""
        super().__init__(require_grad=require_grad, **kwargs)

    async def __call__(self, reasoning: str, result: str, **kwargs) -> ToolResponse:
        """执行组件调用并返回结果。"""
        # 组装并返回结果。
        if reasoning is None or reasoning == "":
            reasoning = "No reasoning provided"
        else:
            reasoning = str(reasoning)
        if result is None or result == "":
            result = "No result provided"
        else:
            result = str(result)
        return ToolResponse(
            success=True,
            message=result,
            extra=ToolExtra(data={"reasoning": reasoning, "result": result}),
        )
