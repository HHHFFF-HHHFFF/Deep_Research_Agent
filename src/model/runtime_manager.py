"""配置驱动的统一模型管理器。"""

from __future__ import annotations

import builtins
import os
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from pydantic import BaseModel

from src.logger import logger
from src.message.types import Message
from src.model.openai.embedding import EmbeddingOpenAI
from src.model.openai_compatible import ChatOpenAICompatible
from src.model.settings import PROVIDERS, ModelRuntimeSettings, split_model_reference
from src.model.types import LLMExtra, LLMResponse, ModelConfig

if TYPE_CHECKING:
    from src.tool.types import Tool


class MissingCredentialsModel:
    """阻止 SDK 从其他提供方环境变量中自动读取密钥。"""

    def __init__(self, provider: str, api_key_env: str) -> None:
        self.provider = provider
        self.api_key_env = api_key_env

    async def __call__(self, **kwargs: Any) -> LLMResponse:
        return LLMResponse(
            success=False,
            message=(
                f"提供方 {self.provider} 未配置 {self.api_key_env}，"
                "已阻止调用以避免误用其他提供方密钥"
            ),
        )


class ModelManager:
    """注册模型、路由调用并在失败时按顺序降级。"""

    def __init__(self) -> None:
        self.models: dict[str, ModelConfig] = {}
        self.model_clients: dict[str, Any] = {}
        self.settings: ModelRuntimeSettings | None = None

    @property
    def default_model_name(self) -> str | None:
        """返回当前默认聊天模型。"""
        return self.settings.primary_model if self.settings else None

    @property
    def embedding_model_name(self) -> str | None:
        """返回当前默认向量模型。"""
        return self.settings.embedding_model if self.settings else None

    async def initialize(
        self,
        settings: ModelRuntimeSettings | None = None,
        *,
        primary_model: str | None = None,
        fallback_models: builtins.list[str] | None = None,
        embedding_model: str | None = None,
        embedding_fallback_models: builtins.list[str] | None = None,
    ) -> None:
        """根据环境变量和运行参数，仅初始化本次需要的模型。"""
        load_dotenv(verbose=False)
        self.settings = settings or ModelRuntimeSettings.from_env(
            primary_model=primary_model,
            fallback_models=fallback_models,
            embedding_model=embedding_model,
            embedding_fallback_models=embedding_fallback_models,
        )

        self.models.clear()
        self.model_clients.clear()

        await self._register_chain(
            [self.settings.primary_model, *self.settings.fallback_models],
            model_type="chat/completions",
        )
        await self._register_chain(
            [self.settings.embedding_model, *self.settings.embedding_fallback_models],
            model_type="embeddings",
        )

        logger.info(
            "| 模型管理器初始化完成：聊天模型=%s，向量模型=%s",
            self.settings.primary_model,
            self.settings.embedding_model,
        )

    async def _register_chain(self, references: builtins.list[str], model_type: str) -> None:
        """注册一条模型链，并把每个模型指向下一个备用模型。"""
        unique_references = list(dict.fromkeys(references))
        for index, reference in enumerate(unique_references):
            fallback = unique_references[index + 1] if index + 1 < len(unique_references) else None
            existing = self.models.get(reference)
            if existing:
                if existing.model_type != model_type:
                    raise ValueError(f"模型 {reference} 不能同时作为聊天模型和向量模型")
                existing.fallback_model = fallback
                continue

            config = self._build_model_config(reference, model_type, fallback)
            await self.register_model(config)

    def _build_model_config(
        self,
        reference: str,
        model_type: str,
        fallback_model: str | None,
    ) -> ModelConfig:
        if self.settings is None:
            raise RuntimeError("模型管理器尚未初始化")

        provider_name, model_id = split_model_reference(reference)
        provider = PROVIDERS[provider_name]
        is_chat = model_type == "chat/completions"

        return ModelConfig(
            model_name=reference,
            model_type=model_type,
            model_id=model_id,
            provider=provider_name,
            adapter=provider.adapter,
            api_base=provider.base_url(),
            api_key=provider.api_key(),
            temperature=self.settings.temperature if is_chat else None,
            max_completion_tokens=self.settings.max_tokens if is_chat else None,
            timeout_seconds=self.settings.timeout_seconds,
            max_retries=self.settings.max_retries,
            max_tokens_parameter=provider.max_tokens_parameter,
            supports_streaming=is_chat,
            supports_functions=is_chat,
            supports_vision=False,
            supports_structured_output=is_chat,
            fallback_model=fallback_model,
        )

    def _openrouter_headers(self, provider: str) -> dict[str, str] | None:
        if provider != "openrouter":
            return None

        headers: dict[str, str] = {}
        if referer := os.getenv("OPENROUTER_HTTP_REFERER"):
            headers["HTTP-Referer"] = referer
        if title := os.getenv("OPENROUTER_X_TITLE"):
            headers["X-Title"] = title
        return headers or None

    async def _create_client(self, config: ModelConfig) -> Any:
        """按照协议适配器创建客户端，不在这里发起网络请求。"""
        if not config.api_key:
            provider = PROVIDERS[config.provider]
            return MissingCredentialsModel(config.provider, provider.api_key_env)

        if config.model_type == "embeddings":
            if config.adapter != "openai_compatible":
                raise ValueError(f"提供方 {config.provider} 暂不支持统一向量接口")
            return EmbeddingOpenAI(
                model=config.model_id,
                api_key=config.api_key,
                base_url=config.api_base,
                timeout=config.timeout_seconds,
                max_retries=config.max_retries,
            )

        if config.adapter == "anthropic":
            # 可选提供方按需导入，避免默认路径加载无关依赖。
            from src.model.anthropic.chat import ChatAnthropic

            return ChatAnthropic(
                model=config.model_id,
                api_key=config.api_key,
                base_url=config.api_base,
                temperature=config.temperature,
                max_tokens=config.max_completion_tokens,
            )

        if config.adapter == "google":
            # 可选提供方按需导入，避免默认路径加载无关依赖。
            from src.model.google.chat import ChatGoogle

            return ChatGoogle(
                model=config.model_id,
                api_key=config.api_key,
                temperature=config.temperature,
                max_output_tokens=config.max_completion_tokens,
            )

        return ChatOpenAICompatible(
            model=config.model_id,
            provider_name=config.provider,
            api_key=config.api_key,
            base_url=config.api_base,
            temperature=config.temperature,
            frequency_penalty=None,
            max_completion_tokens=config.max_completion_tokens,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
            max_tokens_parameter=config.max_tokens_parameter,
            default_headers=self._openrouter_headers(config.provider),
        )

    async def register_model(self, config: ModelConfig, client: Any = None) -> None:
        """注册模型；测试可以注入不联网的模拟客户端。"""
        if config.provider not in PROVIDERS:
            raise ValueError(f"不支持的模型提供方：{config.provider}")

        self.models[config.model_name] = config
        self.model_clients[config.model_name] = client or await self._create_client(config)
        logger.info("| 已注册模型：%s", config.model_name)

    def _fallback_chain(self, model: str) -> builtins.list[str]:
        """展开备用模型链，并阻止循环配置。"""
        chain: list[str] = []
        visited = set()
        current: str | None = model

        while current:
            if current in visited:
                raise ValueError(f"检测到循环备用模型配置：{current}")
            visited.add(current)
            chain.append(current)
            config = self.models.get(current)
            current = config.fallback_model if config else None
        return chain

    async def _invoke_client(
        self,
        model_name: str,
        messages: builtins.list[Message],
        tools: builtins.list[Tool] | None,
        response_format: type[BaseModel] | BaseModel | dict[str, Any] | None,
        stream: bool,
        plugins: builtins.list[dict[str, Any]] | None,
        **kwargs: Any,
    ) -> LLMResponse:
        config = self.models[model_name]
        client = self.model_clients[model_name]

        if config.model_type == "embeddings":
            embedding_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key not in {"tools", "response_format", "stream", "plugins"}
            }
            return await client(messages=messages, **embedding_kwargs)

        call_kwargs: dict[str, Any] = {
            "messages": messages,
            "tools": tools,
            "response_format": response_format,
            "stream": stream,
            **kwargs,
        }
        if plugins is not None and config.provider == "openrouter":
            call_kwargs["plugins"] = plugins
        return await client(**call_kwargs)

    def _log_usage(self, model_name: str, result: LLMResponse) -> None:
        if not result.success or not result.extra or not result.extra.data:
            return
        usage = result.extra.data.get("usage")
        if not usage:
            return

        input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        output_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
        total_tokens = usage.get("total_tokens") or input_tokens + output_tokens
        logger.info(
            "| 模型用量：model=%s, input=%s, output=%s, total=%s",
            model_name,
            input_tokens,
            output_tokens,
            total_tokens,
        )

    async def __call__(
        self,
        model: str | None = None,
        messages: builtins.list[Message] | None = None,
        tools: builtins.list[Tool] | None = None,
        response_format: type[BaseModel] | BaseModel | dict[str, Any] | None = None,
        stream: bool = False,
        plugins: builtins.list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """调用指定模型；失败响应和异常都会触发备用模型。"""
        if tools and response_format:
            raise ValueError("工具调用和结构化响应不能同时启用")
        if messages is None:
            raise ValueError("messages 参数不能为空")
        if self.settings is None:
            raise RuntimeError("模型管理器尚未初始化")

        selected_model = model or self.settings.primary_model
        if selected_model not in self.models:
            available = "、".join(self.models)
            return LLMResponse(
                success=False,
                message=f"模型 {selected_model} 未注册。当前可用模型：{available}",
            )

        errors: list[str] = []
        for candidate in self._fallback_chain(selected_model):
            try:
                result = await self._invoke_client(
                    candidate,
                    messages,
                    tools,
                    response_format,
                    stream,
                    plugins,
                    **kwargs,
                )
                if result.success:
                    if candidate != selected_model:
                        logger.warning("| 已降级到备用模型：%s", candidate)
                    self._log_usage(candidate, result)
                    return result
                errors.append(f"{candidate}: {result.message}")
            # 提供方 SDK 的异常类型并不统一，路由层必须统一捕获后才能降级。
            except Exception as error:  # noqa: BLE001
                errors.append(f"{candidate}: {error}")

            logger.warning("| 模型调用失败，准备尝试备用模型：%s", candidate)

        return LLMResponse(
            success=False,
            message="所有候选模型均调用失败：" + " | ".join(errors),
            extra=LLMExtra(data={"attempted_models": self._fallback_chain(selected_model)}),
        )

    async def achat(
        self,
        messages: builtins.list[Message],
        model: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """使用默认聊天模型或显式指定的聊天模型。"""
        return await self(model=model, messages=messages, **kwargs)

    async def aembedding(
        self,
        messages: builtins.list[Message],
        model: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """使用与聊天模型独立配置的向量模型。"""
        if self.settings is None:
            raise RuntimeError("模型管理器尚未初始化")
        return await self(
            model=model or self.settings.embedding_model,
            messages=messages,
            **kwargs,
        )

    async def get_model_config(self, model: str) -> ModelConfig | None:
        return self.models.get(model)

    async def list(self) -> builtins.list[str]:
        return list(self.models)


model_manager = ModelManager()

__all__ = ["ModelManager", "model_manager"]
