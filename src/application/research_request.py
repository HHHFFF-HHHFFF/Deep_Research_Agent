"""研究请求的数据模型与输入解析。"""

from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


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
    model_provider: Literal["qwen", "deepseek"] | None = Field(
        default=None,
        description="聊天模型提供方；为空时使用项目配置。",
    )
    model_id: str | None = Field(
        default=None,
        min_length=1,
        description="聊天模型标识；为空时使用项目配置。",
    )
    fallback_models: list[str] | None = Field(
        default=None,
        description="按顺序尝试的备用聊天模型，格式为“提供方/模型”。",
    )
    embedding_provider: str | None = Field(
        default=None,
        min_length=1,
        description="向量模型提供方；为空时使用项目配置。",
    )
    embedding_model_id: str | None = Field(
        default=None,
        min_length=1,
        description="向量模型标识；为空时使用项目配置。",
    )

    @field_validator("task")
    @classmethod
    def normalize_task(cls, value: str) -> str:
        """清理研究主题，并拒绝只包含空白字符的输入。"""
        normalized = value.strip()
        if not normalized:
            raise ValueError("研究主题不能为空")
        return normalized

    @field_validator("files")
    @classmethod
    def normalize_files(cls, value: list[str]) -> list[str]:
        """清理文件路径，并保持首次出现的顺序。"""
        normalized_files: list[str] = []
        for file_path in value:
            normalized = file_path.strip()
            if normalized and normalized not in normalized_files:
                normalized_files.append(normalized)
        return normalized_files

    @field_validator("fallback_models")
    @classmethod
    def validate_fallback_models(cls, value: list[str] | None) -> list[str] | None:
        """确保备用模型使用统一的“提供方/模型”标识。"""
        if value is None:
            return None

        normalized_models: list[str] = []
        for model_name in value:
            normalized = model_name.strip()
            if "/" not in normalized:
                raise ValueError("备用模型必须使用“提供方/模型”格式")
            if normalized not in normalized_models:
                normalized_models.append(normalized)
        return normalized_models

    @model_validator(mode="after")
    def validate_model_pairs(self) -> "ResearchRequest":
        """提供方被显式覆盖时，必须同时指定对应模型。"""
        if self.model_provider and not self.model_id:
            raise ValueError("指定聊天模型提供方时必须同时指定模型标识")
        if self.embedding_provider and not self.embedding_model_id:
            raise ValueError("指定向量模型提供方时必须同时指定模型标识")
        return self


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
