"""Qwen、DeepSeek 等 OpenAI 兼容聊天接口的统一适配器。"""

from typing import Any, Literal

from src.model.openai.chat import ChatOpenAI


class ChatOpenAICompatible(ChatOpenAI):
    """复用 OpenAI SDK，同时处理兼容服务的参数差异。"""

    provider_name: str = "openai_compatible"
    max_tokens_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"

    @property
    def provider(self) -> str:
        return self.provider_name

    async def _build_params(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        payload = await super()._build_params(*args, **kwargs)
        params = payload["params"]

        if self.max_tokens_parameter == "max_tokens":
            max_tokens = params.pop("max_completion_tokens", None)
            if max_tokens is not None:
                params["max_tokens"] = max_tokens

        return payload


__all__ = ["ChatOpenAICompatible"]
