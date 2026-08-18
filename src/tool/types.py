from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.dynamic import dynamic_manager


class ToolExtra(BaseModel):
    """定义 `ToolExtra`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    file_path: str | list[str] | None = Field(
        default=None, description="The file path of the extra data"
    )
    data: dict[str, Any] | None = Field(
        default=None, description="The data of the extra data"
    )
    parsed_model: BaseModel | None = Field(
        default=None, description="The parsed model of the extra data"
    )


class ToolResponse(BaseModel):
    """定义 `ToolResponse`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    success: bool = Field(description="Whether the tool call was successful")
    message: str = Field(description="The message from the tool call")
    extra: ToolExtra | None = Field(
        default=None, description="The extra data from the tool call"
    )


class Tool(BaseModel):
    """定义 `Tool`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(description="The name of the tool")
    description: str = Field(description="The description of the tool")
    metadata: dict[str, Any] | None = Field(
        default={}, description="The metadata of the tool"
    )
    require_grad: bool = Field(
        default=False, description="Whether the tool requires gradients"
    )


class ToolConfig(BaseModel):
    """定义 `ToolConfig`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(description="The name of the tool")
    description: str = Field(description="The description of the tool")
    metadata: dict[str, Any] | None = Field(
        default={}, description="The metadata of the tool"
    )
    require_grad: bool = Field(
        default=False, description="Whether the tool requires gradients"
    )
    version: str = Field(default="1.0.0", description="Version of the tool")

    cls: type[Tool] | None = Field(default=None, description="The class of the tool")
    config: dict[str, Any] | None = Field(
        default={}, description="The initialization configuration of the tool"
    )
    instance: Tool | None = Field(default=None, description="The instance of the tool")
    code: str | None = Field(
        default=None,
        description="Source code for dynamically generated tool classes (used when cls cannot be imported from a module)",
    )

    # 说明相关实现细节。
    function_calling: dict[str, Any] | None = Field(
        default=None, description="Default function calling representation"
    )
    text: str | None = Field(default=None, description="Default text representation")
    args_schema: type[BaseModel] | None = Field(
        default=None, description="Default args schema (BaseModel type)"
    )

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """实现 `model_dump` 的业务逻辑。"""

        result = {
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata,
            "require_grad": self.require_grad,
            "version": self.version,
            "cls": dynamic_manager.get_class_string(self.cls) if self.cls else None,
            "config": self.config,
            "instance": None,
            "code": self.code,
            "function_calling": self.function_calling,
            "text": self.text,
            "args_schema": dynamic_manager.serialize_args_schema(self.args_schema)
            if self.args_schema
            else None,
        }

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolConfig:
        """从持久化字典恢复工具配置，不执行其中携带的代码。"""
        payload = dict(data)
        serialized_schema = payload.get("args_schema")
        payload["args_schema"] = (
            dynamic_manager.deserialize_args_schema(serialized_schema)
            if isinstance(serialized_schema, dict)
            else None
        )
        payload["cls"] = None
        payload["instance"] = None
        return cls.model_validate(payload)


__all__ = [
    "Tool",
    "ToolConfig",
    "ToolResponse",
]
