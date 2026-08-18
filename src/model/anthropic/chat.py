from typing import Any, ClassVar

import httpx

try:
    from anthropic import APIConnectionError, APIError, AsyncAnthropic, RateLimitError

    try:
        from anthropic import transform_schema
    except ImportError:
        transform_schema = None
except ImportError:
    AsyncAnthropic = None
    APIError = Exception
    APIConnectionError = Exception
    RateLimitError = Exception
    transform_schema = None

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from src.logger import logger
from src.message.types import Message
from src.model.anthropic.serializer import AnthropicChatSerializer
from src.model.types import LLMExtra, LLMResponse

if TYPE_CHECKING:
    from src.tool.types import Tool


class ChatAnthropic(BaseModel):
    """定义 `ChatAnthropic`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # 组装并返回结果。
    OUTPUT_FORMAT_SUPPORTED_MODELS: ClassVar[list[str]] = [
        "claude-sonnet-4-5-20250929",
        "claude-opus-4-1-20250805",  # 说明相关实现细节。
        # 处理模型调用。
    ]

    # 配置相关参数。
    model: str

    # 处理模型调用。
    temperature: float | None = 0.7
    top_p: float | None = None
    max_tokens: int | None = 16384

    # 处理输入参数。
    api_key: str | None = None
    base_url: str | httpx.URL | None = None
    reasoning: dict[str, Any] | None = None
    timeout: float | httpx.Timeout | None = None
    max_retries: int = 5
    default_headers: dict[str, str] | None = None
    http_client: httpx.AsyncClient | None = None

    @property
    def provider(self) -> str:
        return "anthropic"

    def _get_client_params(self) -> dict[str, Any]:
        """实现 `_get_client_params` 的业务逻辑。"""
        # 说明相关实现细节。
        headers = dict(self.default_headers) if self.default_headers else {}

        # 组装并返回结果。
        if "anthropic-beta" not in headers:
            headers["anthropic-beta"] = "structured-outputs-2025-11-13"

        base_params = {
            "api_key": self.api_key,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "default_headers": headers if headers else None,
        }

        # 说明相关实现细节。
        if self.base_url:
            base_params["base_url"] = str(self.base_url)

        # 说明相关实现细节。
        if self.http_client is not None:
            base_params["http_client"] = self.http_client

        # 创建所需对象。
        client_params = {k: v for k, v in base_params.items() if v is not None}

        return client_params

    def get_client(self) -> AsyncAnthropic:
        """获取与 `get_client` 对应的数据或状态。"""
        if AsyncAnthropic is None:
            raise ImportError(
                "anthropic package is required. Install it with: pip install anthropic"
            )

        client_params = self._get_client_params()
        return AsyncAnthropic(**client_params)

    @property
    def name(self) -> str:
        return str(self.model)

    def _get_usage(self, response) -> dict[str, Any] | None:
        """实现 `_get_usage` 的业务逻辑。"""
        if hasattr(response, "usage") and response.usage is not None:
            return response.usage.model_dump()
        else:
            return None

    async def _build_params(
        self,
        messages: list[Message],
        tools: list["Tool"] | None = None,
        response_format: type[BaseModel] | BaseModel | dict | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """实现 `_build_params` 的业务逻辑。"""
        # 转换并规范化数据。
        system_message, anthropic_messages = AnthropicChatSerializer.serialize_messages(
            messages
        )

        # 创建所需对象。
        params: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
        }

        # 说明相关实现细节。
        if system_message:
            params["system"] = system_message

        # 处理输入参数。
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.reasoning is not None:
            params.update(self.reasoning)

        # 转换并规范化数据。
        if tools:
            formatted_tools = AnthropicChatSerializer.serialize_tools(tools)
            if formatted_tools:
                params["tools"] = formatted_tools

        # 组装并返回结果。
        # 组装并返回结果。
        use_beta_api = False
        if response_format:
            # 校验输入与当前状态。
            model_supports_output_format = any(
                supported_model in self.model
                for supported_model in ChatAnthropic.OUTPUT_FORMAT_SUPPORTED_MODELS
            )

            if not model_supports_output_format:
                logger.warning(
                    f"Model {self.model} does not support output_format. "
                    f"Supported models: {', '.join(ChatAnthropic.OUTPUT_FORMAT_SUPPORTED_MODELS)}. "
                    f"Skipping structured output."
                )
            else:
                try:
                    params["output_format"] = (
                        AnthropicChatSerializer.serialize_response_format(
                            response_format
                        )
                    )
                    use_beta_api = True
                except ValueError as e:
                    logger.warning(f"Failed to serialize response_format: {e}")

        # 组装并返回结果。
        if use_beta_api:
            params["betas"] = ["structured-outputs-2025-11-13"]

        # 说明相关实现细节。
        if stream:
            params["stream"] = True
            logger.warning("Streaming is not yet fully implemented for Anthropic API")

        # 说明相关实现细节。
        params.update(kwargs)

        return {
            "system": system_message,
            "messages": anthropic_messages,
            "params": params,
            "use_beta_api": use_beta_api,
        }

    async def _call_model(
        self,
        use_beta_api: bool = False,
        **params: Any,
    ) -> Any:
        """实现 `_call_model` 的业务逻辑。"""
        client = self.get_client()
        # 组装并返回结果。
        if use_beta_api:
            response = await client.beta.messages.create(**params)
        else:
            response = await client.messages.create(**params)

        return response

    async def __call__(
        self,
        messages: list[Message],
        tools: list["Tool"] | None = None,
        response_format: type[BaseModel] | BaseModel | dict | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        """执行组件调用并返回结果。"""
        if AsyncAnthropic is None:
            raise ImportError(
                "anthropic package is required. Install it with: pip install anthropic"
            )

        try:
            params = await self._build_params(
                messages=messages,
                tools=tools,
                response_format=response_format,
                stream=stream,
                **kwargs,
            )

            response = await self._call_model(
                use_beta_api=params.get("use_beta_api", False),
                **params["params"],
            )

            return await self._format_response(
                response=response,
                tools=tools,
                response_format=response_format,
            )

        except RateLimitError as e:
            logger.error(f"Rate limit error: {e}")
            return LLMResponse(
                success=False,
                message=f"Rate limit error: {e!s}",
                extra=LLMExtra(data={"error": str(e), "model": self.name}),
            )
        except APIConnectionError as e:
            logger.error(f"API connection error: {e}")
            return LLMResponse(
                success=False,
                message=f"API connection error: {e!s}",
                extra=LLMExtra(data={"error": str(e), "model": self.name}),
            )
        except APIError as e:
            logger.error(f"API error: {e}")
            return LLMResponse(
                success=False,
                message=f"API error: {e!s}",
                extra=LLMExtra(
                    data={
                        "error": str(e),
                        "status_code": getattr(e, "status_code", None),
                        "model": self.name,
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return LLMResponse(
                success=False,
                message=f"Unexpected error: {e!s}",
                extra=LLMExtra(data={"error": str(e), "model": self.name}),
            )

    async def _format_response(
        self,
        response: Any,
        tools: list["Tool"] | None = None,
        response_format: type[BaseModel] | BaseModel | dict | None = None,
    ) -> LLMResponse:
        """实现 `_format_response` 的业务逻辑。"""
        try:
            # 组装并返回结果。
            if hasattr(response, "content"):
                content = response.content
            elif isinstance(response, dict):
                content = response.get("content", [])
            else:
                content = []

            if not content:
                return LLMResponse(
                    success=False,
                    message="No content in response",
                    extra=LLMExtra(
                        data={
                            "raw_response": response.model_dump()
                            if hasattr(response, "model_dump")
                            else str(response)
                        }
                    ),
                )

            # 处理工具调用。
            text_parts = []
            tool_calls = []

            for item in content:
                if hasattr(item, "type"):
                    # 组装并返回结果。
                    if item.type == "text":
                        text_parts.append(item.text)
                    elif item.type == "tool_use":
                        tool_calls.append(
                            {
                                "id": item.id,
                                "name": item.name,
                                "input": item.input,
                            }
                        )
                elif isinstance(item, dict):
                    # 转换并规范化数据。
                    item_type = item.get("type")
                    if item_type == "text":
                        text_parts.append(item.get("text", ""))
                    elif item_type == "tool_use":
                        tool_calls.append(item)

            message_text = "\n".join(text_parts) if text_parts else ""

            usage = self._get_usage(response)
            stop_reason = (
                response.stop_reason
                if hasattr(response, "stop_reason")
                else response.get("stop_reason")
                if isinstance(response, dict)
                else None
            )

            # 说明相关实现细节。
            if tools and tool_calls:
                formatted_lines = []
                functions = []

                for tool_call in tool_calls:
                    name = tool_call.get("name", "")
                    tool_id = tool_call.get("id", "")
                    input_data = tool_call.get("input", {})

                    # 处理输入参数。
                    if input_data:
                        args_str = ", ".join(
                            [f"{k}={v!r}" for k, v in input_data.items()]
                        )
                        formatted_lines.append(f"Calling function {name}({args_str})")
                    else:
                        formatted_lines.append(f"Calling function {name}()")

                    functions.append({"id": tool_id, "name": name, "args": input_data})

                formatted_message = "\n".join(formatted_lines)

                extra = LLMExtra(
                    data={
                        "raw_response": response.model_dump()
                        if hasattr(response, "model_dump")
                        else response,
                        "functions": functions,
                        "usage": usage,
                        "stop_reason": stop_reason,
                    }
                )

                return LLMResponse(success=True, message=formatted_message, extra=extra)

            # 组装并返回结果。
            elif (
                response_format
                and isinstance(response_format, type)
                and issubclass(response_format, BaseModel)
            ):
                if not message_text:
                    return LLMResponse(
                        success=False,
                        message="Empty response content from model",
                        extra=LLMExtra(data={"raw_response": response}),
                    )

                # 转换并规范化数据。
                import json

                try:
                    data = json.loads(message_text)
                    parsed_model = response_format.model_validate(data)

                    # 转换并规范化数据。
                    model_name = response_format.__name__
                    model_dict = parsed_model.model_dump()

                    field_lines = []
                    for field_name, field_value in model_dict.items():
                        field_lines.append(f"{field_name}={field_value!r}")

                    formatted_message = f"Response result:\n\n{model_name}(\n"
                    formatted_message += ",\n".join(
                        f"    {line}" for line in field_lines
                    )
                    formatted_message += "\n)"

                    extra = LLMExtra(
                        parsed_model=parsed_model,
                        data={
                            "raw_response": response.model_dump()
                            if hasattr(response, "model_dump")
                            else response,
                            "usage": usage,
                            "stop_reason": stop_reason,
                        },
                    )

                    return LLMResponse(
                        success=True, message=formatted_message, extra=extra
                    )
                except json.JSONDecodeError as e:
                    return LLMResponse(
                        success=False,
                        message=f"Failed to parse JSON from response: {e}",
                        extra=LLMExtra(data={"error": str(e), "content": message_text}),
                    )
                except Exception as e:
                    return LLMResponse(
                        success=False,
                        message=f"Failed to validate response against schema: {e}",
                        extra=LLMExtra(data={"error": str(e), "content": message_text}),
                    )

            # 组装并返回结果。
            else:
                extra = LLMExtra(
                    data={
                        "raw_response": response.model_dump()
                        if hasattr(response, "model_dump")
                        else response,
                        "usage": usage,
                        "stop_reason": stop_reason,
                    }
                )

                return LLMResponse(success=True, message=message_text, extra=extra)

        except Exception as e:
            logger.error(f"Failed to format response: {e}")
            return LLMResponse(
                success=False,
                message=f"Failed to format response: {e}",
                extra=LLMExtra(data={"error": str(e)}),
            )
