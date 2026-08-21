"""模型提供方与运行时配置。"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ProviderSettings(BaseModel):
    """描述一个模型提供方的连接方式。"""

    name: str
    adapter: Literal["openai_compatible", "anthropic", "google"]
    api_key_env: str
    base_url_env: str | None = None
    default_base_url: str | None = None
    max_tokens_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"

    def api_key(self) -> str | None:
        """从环境变量读取密钥，避免把密钥写入源码。"""
        return os.getenv(self.api_key_env) or None

    def base_url(self) -> str | None:
        """优先使用用户配置的服务地址。"""
        if self.base_url_env:
            configured = os.getenv(self.base_url_env)
            if configured:
                normalized = configured.strip().rstrip("/")
                if not normalized.startswith(("http://", "https://")):
                    normalized = f"https://{normalized}"
                if self.name == "qwen" and not normalized.endswith(
                    "/compatible-mode/v1"
                ):
                    normalized = f"{normalized}/compatible-mode/v1"
                return normalized
        return self.default_base_url


PROVIDERS: dict[str, ProviderSettings] = {
    "qwen": ProviderSettings(
        name="qwen",
        adapter="openai_compatible",
        api_key_env="DASHSCOPE_API_KEY",
        base_url_env="QWEN_BASE_URL",
        default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
    "deepseek": ProviderSettings(
        name="deepseek",
        adapter="openai_compatible",
        api_key_env="DEEPSEEK_API_KEY",
        base_url_env="DEEPSEEK_API_BASE",
        default_base_url="https://api.deepseek.com",
    ),
    "openai": ProviderSettings(
        name="openai",
        adapter="openai_compatible",
        api_key_env="OPENAI_API_KEY",
        base_url_env="OPENAI_API_BASE",
        default_base_url="https://api.openai.com/v1",
        max_tokens_parameter="max_completion_tokens",
    ),
    "openrouter": ProviderSettings(
        name="openrouter",
        adapter="openai_compatible",
        api_key_env="OPENROUTER_API_KEY",
        base_url_env="OPENROUTER_API_BASE",
        default_base_url="https://openrouter.ai/api/v1",
    ),
    "anthropic": ProviderSettings(
        name="anthropic",
        adapter="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        base_url_env="ANTHROPIC_API_BASE",
    ),
    "google": ProviderSettings(
        name="google",
        adapter="google",
        api_key_env="GOOGLE_API_KEY",
    ),
}


def normalize_model_reference(model: str, provider: str) -> str:
    """将模型标识统一为 ``提供方/模型`` 格式。"""
    cleaned_model = model.strip()
    cleaned_provider = provider.strip().lower()
    if not cleaned_model:
        raise ValueError("模型名称不能为空")
    if "/" in cleaned_model:
        return cleaned_model
    return f"{cleaned_provider}/{cleaned_model}"


def split_model_reference(reference: str) -> tuple[str, str]:
    """拆分并校验 ``提供方/模型`` 标识。"""
    provider, separator, model_id = reference.partition("/")
    if not separator or not provider or not model_id:
        raise ValueError(f"模型标识必须使用“提供方/模型”格式：{reference}")
    normalized_provider = provider.lower()
    if normalized_provider not in PROVIDERS:
        supported = "、".join(sorted(PROVIDERS))
        raise ValueError(f"不支持的模型提供方 {provider}，可选值：{supported}")
    return normalized_provider, model_id


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


class ModelRuntimeSettings(BaseModel):
    """控制本次运行使用的聊天模型、备用模型和向量模型。"""

    primary_model: str = "qwen/qwen3-max"
    fallback_models: list[str] = Field(default_factory=list)
    embedding_model: str = "qwen/text-embedding-v4"
    embedding_fallback_models: list[str] = Field(default_factory=list)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=8192, ge=1)
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=10)

    @field_validator(
        "primary_model",
        "embedding_model",
        "fallback_models",
        "embedding_fallback_models",
    )
    @classmethod
    def validate_model_references(cls, value):
        """尽早发现拼写错误，避免运行到 API 调用时才失败。"""
        references = value if isinstance(value, list) else [value]
        for reference in references:
            split_model_reference(reference)
        return value

    @classmethod
    def from_env(
        cls,
        *,
        primary_model: str | None = None,
        fallback_models: list[str] | None = None,
        embedding_model: str | None = None,
        embedding_fallback_models: list[str] | None = None,
    ) -> ModelRuntimeSettings:
        """读取环境变量，并允许命令行或配置文件覆盖关键模型。"""
        provider = os.getenv("MODEL_PROVIDER", "qwen")
        model_id = os.getenv("MODEL_NAME", "qwen3-max")
        embedding_provider = os.getenv("EMBEDDING_PROVIDER", "qwen")
        embedding_model_id = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")

        resolved_primary = primary_model or normalize_model_reference(
            model_id, provider
        )
        resolved_embedding = embedding_model or normalize_model_reference(
            embedding_model_id,
            embedding_provider,
        )

        raw_fallbacks = fallback_models
        if raw_fallbacks is None:
            raw_fallbacks = _split_csv(os.getenv("MODEL_FALLBACKS"))

        raw_embedding_fallbacks = embedding_fallback_models
        if raw_embedding_fallbacks is None:
            raw_embedding_fallbacks = _split_csv(os.getenv("EMBEDDING_FALLBACKS"))

        return cls(
            primary_model=resolved_primary,
            fallback_models=raw_fallbacks,
            embedding_model=resolved_embedding,
            embedding_fallback_models=raw_embedding_fallbacks,
            temperature=float(os.getenv("MODEL_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("MODEL_MAX_TOKENS", "8192")),
            timeout_seconds=float(os.getenv("MODEL_TIMEOUT_SECONDS", "60")),
            max_retries=int(os.getenv("MODEL_MAX_RETRIES", "2")),
        )


__all__ = [
    "PROVIDERS",
    "ModelRuntimeSettings",
    "ProviderSettings",
    "normalize_model_reference",
    "split_model_reference",
]
