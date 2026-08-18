"""研究请求的数据模型与输入解析。"""

from collections.abc import Callable

from pydantic import BaseModel, Field, field_validator


class ResearchRequest(BaseModel):
    """描述一次可以由命令行或网页界面发起的研究任务。"""

    task: str = Field(
        min_length=1,
        max_length=4000,
        description="用户希望智能体研究的问题或方向。",
    )
    files: list[str] = Field(
        default_factory=list,
        description="本次研究允许读取的本地文档路径。",
    )

    @field_validator("task")
    @classmethod
    def normalize_task(cls, value: str) -> str:
        """清理研究主题，并拒绝只包含空白字符的输入。"""
        normalized = value.strip()
        if not normalized:
            raise ValueError("研究主题不能为空")
        return normalized


def resolve_research_task(
    task: str | None,
    prompt: Callable[[str], str] = input,
) -> str:
    """优先读取命令行主题，否则在交互式终端中询问用户。"""
    if task is not None:
        return task

    try:
        return prompt("请输入研究主题：")
    except EOFError as error:
        raise ValueError("未提供研究主题，请使用 --task 参数") from error
