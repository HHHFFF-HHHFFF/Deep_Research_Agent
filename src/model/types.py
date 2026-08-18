from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    """单个模型的能力、连接方式与降级关系。"""

    model_name: str = Field(description="项目内部使用的“提供方/模型”标识。")
    model_type: str = Field(description="模型类型，例如聊天或向量模型。")
    model_id: str = Field(description="传给模型服务的真实模型标识。")
    provider: str = Field(description="模型提供方名称。")
    adapter: str = Field(default="openai_compatible", description="使用的协议适配器。")
    api_base: str | None = Field(default=None, description="模型服务地址。")
    api_key: str | None = Field(default=None, description="模型密钥。", repr=False, exclude=True)
    temperature: float | None = Field(default=None, description="生成温度。")
    reasoning: dict[str, Any] | None = Field(default=None, description="推理参数。")
    plugins: list[dict[str, Any]] | None = Field(default=None, description="提供方扩展插件。")
    max_completion_tokens: int | None = Field(default=None, description="最大输出词元数。")
    max_output_tokens: int | None = Field(default=None, description="响应接口最大输出词元数。")
    timeout_seconds: float = Field(default=60.0, gt=0, description="请求超时秒数。")
    max_retries: int = Field(default=2, ge=0, description="客户端重试次数。")
    max_tokens_parameter: str = Field(default="max_tokens", description="兼容接口使用的词元参数名。")
    supports_streaming: bool = Field(default=True, description="是否支持流式输出。")
    supports_functions: bool = Field(default=False, description="是否支持工具调用。")
    supports_vision: bool = Field(default=False, description="是否支持视觉输入。")
    supports_structured_output: bool = Field(default=False, description="是否支持结构化输出。")
    output_version: str | None = Field(
        default=None,
        description="提供方需要时使用的输出格式版本。",
    )
    fallback_model: str | None = Field(
        default=None,
        description="当前模型失败时使用的下一个模型。",
    )


class LLMExtra(BaseModel):
    """模型响应附加数据。"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    file_path: str | None = Field(default=None, description="内容对应的文件路径。")
    data: dict[str, Any] | None = Field(default=None, description="调用用量、原始响应等附加数据。")
    parsed_model: BaseModel | None = Field(default=None, description="校验后的结构化响应。")

class LLMResponse(BaseModel):
    """统一的模型调用结果。"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    success: bool = Field(description="模型调用是否成功。")
    message: str = Field(description="模型正文或错误信息。")
    extra: LLMExtra | None = Field(default=None, description="模型调用附加数据。")

__all__ = ["LLMExtra", "LLMResponse", "ModelConfig"]
