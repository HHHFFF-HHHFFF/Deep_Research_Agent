from collections.abc import Mapping
from typing import Any

import httpx

try:
    from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError
    from openai.types.shared.chat_model import ChatModel
except ImportError:
    # 执行回退或重试逻辑。
    AsyncOpenAI = None
    APIConnectionError = Exception
    APIStatusError = Exception
    RateLimitError = Exception
    ChatModel = str

from pydantic import BaseModel, ConfigDict

from src.logger import logger
from src.message.types import ContentPartText, HumanMessage, Message, SystemMessage
from src.model.types import LLMExtra, LLMResponse


class EmbeddingOpenAI(BaseModel):
    """定义 `EmbeddingOpenAI`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # 配置相关参数。
    model: ChatModel | str = "text-embedding-3-small"

    # 处理输入参数。
    api_key: str | None = None
    organization: str | None = None
    project: str | None = None
    base_url: str | httpx.URL | None = None
    websocket_base_url: str | httpx.URL | None = None
    timeout: float | httpx.Timeout | None = None
    max_retries: int = 5
    default_headers: Mapping[str, str] | None = None
    default_query: Mapping[str, object] | None = None
    http_client: httpx.AsyncClient | None = None
    _strict_response_validation: bool = False

    # 处理输入参数。
    dimensions: int | None = None  # 处理模型调用。
    encoding_format: str | None = None  # 说明相关实现细节。

    @property
    def provider(self) -> str:
        return "openai"

    def _get_client_params(self) -> dict[str, Any]:
        """实现 `_get_client_params` 的业务逻辑。"""
        base_params = {
            "api_key": self.api_key,
            "organization": self.organization,
            "project": self.project,
            "base_url": self.base_url,
            "websocket_base_url": self.websocket_base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
            "_strict_response_validation": self._strict_response_validation,
        }

        # 创建所需对象。
        client_params = {k: v for k, v in base_params.items() if v is not None}

        # 说明相关实现细节。
        if self.http_client is not None:
            client_params["http_client"] = self.http_client

        return client_params

    def get_client(self) -> AsyncOpenAI:
        """获取与 `get_client` 对应的数据或状态。"""
        if AsyncOpenAI is None:
            raise ImportError(
                "openai package is required. Install it with: pip install openai"
            )

        client_params = self._get_client_params()
        return AsyncOpenAI(**client_params)

    @property
    def name(self) -> str:
        return str(self.model)

    def _extract_text_from_messages(self, messages: list[Message]) -> list[str]:
        """实现 `_extract_text_from_messages` 的业务逻辑。"""
        texts = []

        for message in messages:
            if isinstance(message, (HumanMessage, SystemMessage)):
                if isinstance(message.content, str):
                    texts.append(message.content)
                elif isinstance(message.content, list):
                    for part in message.content:
                        if isinstance(part, ContentPartText):
                            texts.append(part.text)

        return texts

    async def _build_params(
        self,
        messages: list[Message],
        dimensions: int | None = None,
        encoding_format: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """实现 `_build_params` 的业务逻辑。"""
        # 说明相关实现细节。
        texts = self._extract_text_from_messages(messages)
        if not texts:
            raise ValueError("No text content found in messages")

        # 创建所需对象。
        params: dict[str, Any] = {
            "model": self.model,
            "input": texts if len(texts) > 1 else texts[0],  # 说明相关实现细节。
        }

        # 处理输入参数。
        if dimensions is not None:
            params["dimensions"] = dimensions
        elif self.dimensions is not None:
            params["dimensions"] = self.dimensions

        if encoding_format is not None:
            params["encoding_format"] = encoding_format
        elif self.encoding_format is not None:
            params["encoding_format"] = self.encoding_format

        # 说明相关实现细节。
        params.update(kwargs)

        return {
            "input": texts if len(texts) > 1 else texts[0],
            "params": params,
        }

    async def _call_model(
        self,
        input_text: str | list[str],
        **params: Any,
    ) -> Any:
        """实现 `_call_model` 的业务逻辑。"""
        client = self.get_client()
        # 创建所需对象。
        if "input" not in params:
            params["input"] = input_text
        response = await client.embeddings.create(**params)

        return response

    async def _format_response(
        self,
        response: Any,
    ) -> LLMResponse:
        """实现 `_format_response` 的业务逻辑。"""
        # 组装并返回结果。
        embeddings = []
        if hasattr(response, "data"):
            for item in response.data:
                if hasattr(item, "embedding"):
                    embeddings.append(item.embedding)
                elif isinstance(item, dict):
                    embeddings.append(item.get("embedding"))
        elif isinstance(response, dict):
            data = response.get("data", [])
            for item in data:
                if isinstance(item, dict):
                    embeddings.append(item.get("embedding"))

        # 组装并返回结果。
        if len(embeddings) == 1:
            message = f"Embedding vector with {len(embeddings[0])} dimensions"
        else:
            message = f"{len(embeddings)} embedding vectors"

        # 组装并返回结果。
        extra = LLMExtra(
            data={
                "raw_response": response.model_dump()
                if hasattr(response, "model_dump")
                else str(response),
                "embeddings": embeddings,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens
                    if hasattr(response, "usage")
                    and hasattr(response.usage, "prompt_tokens")
                    else None,
                    "total_tokens": response.usage.total_tokens
                    if hasattr(response, "usage")
                    and hasattr(response.usage, "total_tokens")
                    else None,
                }
                if hasattr(response, "usage")
                else None,
            }
        )

        return LLMResponse(success=True, message=message, extra=extra)

    async def __call__(
        self,
        messages: list[Message],
        dimensions: int | None = None,
        encoding_format: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """执行组件调用并返回结果。"""
        if AsyncOpenAI is None:
            raise ImportError(
                "openai package is required. Install it with: pip install openai"
            )

        try:
            params = await self._build_params(
                messages=messages,
                dimensions=dimensions,
                encoding_format=encoding_format,
                **kwargs,
            )

            response = await self._call_model(
                input_text=params["input"],
                **params["params"],
            )

            return await self._format_response(
                response=response,
            )

        except RateLimitError as e:
            logger.error(f"Rate limit error: {e}")
            return LLMResponse(
                success=False,
                message=f"Rate limit error: {e.message}",
                extra={"error": str(e), "model": self.name},
            )
        except APIConnectionError as e:
            logger.error(f"API connection error: {e}")
            return LLMResponse(
                success=False,
                message=f"API connection error: {e!s}",
                extra={"error": str(e), "model": self.name},
            )
        except APIStatusError as e:
            logger.error(f"API status error: {e}")
            return LLMResponse(
                success=False,
                message=f"API status error: {e.message}",
                extra={
                    "error": str(e),
                    "status_code": e.status_code,
                    "model": self.name,
                },
            )
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return LLMResponse(
                success=False,
                message=f"Unexpected error: {e!s}",
                extra={"error": str(e), "model": self.name},
            )
