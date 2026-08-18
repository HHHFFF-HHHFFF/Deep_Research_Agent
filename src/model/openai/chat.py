from collections.abc import Mapping
from typing import Any, Literal

import httpx

try:
    from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError
    from openai.types.chat import ChatCompletionContentPartTextParam
    from openai.types.chat.chat_completion import ChatCompletion
    from openai.types.shared.chat_model import ChatModel
    from openai.types.shared_params.reasoning_effort import ReasoningEffort
    from openai.types.shared_params.response_format_json_schema import (
        JSONSchema,
        ResponseFormatJSONSchema,
    )
except ImportError:
    # 执行回退或重试逻辑。
    AsyncOpenAI = None
    APIConnectionError = Exception
    APIStatusError = Exception
    RateLimitError = Exception
    ChatCompletion = dict
    ChatModel = str
    ReasoningEffort = str
    JSONSchema = dict
    ResponseFormatJSONSchema = dict
    ChatCompletionContentPartTextParam = dict

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from src.logger import logger
from src.message.types import Message
from src.model.openai.serializer import OpenAIChatSerializer
from src.model.types import LLMExtra, LLMResponse

if TYPE_CHECKING:
    from src.tool.types import Tool


class ChatOpenAI(BaseModel):
    """定义 `ChatOpenAI`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # 配置相关参数。
    model: ChatModel | str

    # 处理模型调用。
    temperature: float | None = 0.7
    frequency_penalty: float | None = 0.3
    reasoning: dict[str, Any] | None = None
    seed: int | None = None
    service_tier: Literal["auto", "default", "flex", "priority", "scale"] | None = None
    top_p: float | None = None
    max_completion_tokens: int | None = 16384
    plugins: list[dict[str, Any]] | None = None

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

    reasoning_models: list[ChatModel | str] | None = Field(
        default_factory=lambda: [
            "o3",
            "gpt-5",
            "gpt-5.1",
        ]
    )

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

    def _get_usage(self, response: ChatCompletion) -> dict[str, Any] | None:
        """实现 `_get_usage` 的业务逻辑。"""
        if response.usage is not None:
            usage = response.usage.model_dump()
            return usage
        else:
            return None

    def _get_reasoning(self, message) -> str | None:
        """实现 `_get_reasoning` 的业务逻辑。"""
        reasoning = None
        try:
            # 说明相关实现细节。
            if hasattr(message, "reasoning") and message.reasoning is not None:
                reasoning = message.reasoning
            elif (
                hasattr(message, "reasoning_details")
                and message.reasoning_details is not None
            ):
                # 说明相关实现细节。
                reasoning_details = message.reasoning_details
                if reasoning_details:
                    for detail in reasoning_details:
                        if hasattr(detail, "type"):
                            detail_type = detail.type
                            if detail_type == "reasoning.text" and hasattr(
                                detail, "text"
                            ):
                                reasoning = detail.text
                                break
                            elif detail_type == "reasoning.summary" and hasattr(
                                detail, "summary"
                            ):
                                reasoning = detail.summary
                                break
                        elif isinstance(detail, dict):
                            detail_type = detail.get("type")
                            if detail_type == "reasoning.text":
                                reasoning = detail.get("text")
                                break
                            elif detail_type == "reasoning.summary":
                                reasoning = detail.get("summary")
                                break
        except (AttributeError, KeyError, TypeError, IndexError):
            pass

        return reasoning

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
        openai_messages = OpenAIChatSerializer.serialize_messages(messages)

        # 创建所需对象。
        params: dict[str, Any] = {}

        # 处理输入参数。
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.frequency_penalty is not None:
            params["frequency_penalty"] = self.frequency_penalty
        if self.max_completion_tokens is not None:
            params["max_completion_tokens"] = self.max_completion_tokens
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.seed is not None:
            params["seed"] = self.seed
        if self.service_tier is not None:
            params["service_tier"] = self.service_tier
        if self.reasoning is not None:
            params.update(self.reasoning)

        # 处理模型调用。
        if self.reasoning_models and any(
            str(m).lower() in str(self.model).lower() for m in self.reasoning_models
        ):
            # 移除相关数据或组件。
            params.pop("temperature", None)
            params.pop("frequency_penalty", None)

        # 转换并规范化数据。
        if tools:
            formatted_tools = OpenAIChatSerializer.serialize_tools(tools)
            if formatted_tools:
                params["tools"] = formatted_tools

        # 组装并返回结果。
        if response_format:
            if (
                isinstance(response_format, type)
                and issubclass(response_format, BaseModel)
                or isinstance(response_format, BaseModel)
            ):
                # 转换并规范化数据。
                params["response_format"] = (
                    OpenAIChatSerializer.serialize_response_format(response_format)
                )
            elif isinstance(response_format, dict):
                # 转换并规范化数据。
                params["response_format"] = response_format
            else:
                logger.warning(
                    f"Unsupported response_format type: {type(response_format)}"
                )

        # 说明相关实现细节。
        if stream:
            params["stream"] = True

        # 说明相关实现细节。
        params.update(kwargs)

        return {
            "messages": openai_messages,
            "params": params,
        }

    async def _call_model(
        self,
        messages: list[dict[str, Any]],
        **params: Any,
    ) -> ChatCompletion:
        """实现 `_call_model` 的业务逻辑。"""
        client = self.get_client()
        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            **params,
        )

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
        if AsyncOpenAI is None:
            raise ImportError(
                "openai package is required. Install it with: pip install openai"
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
                messages=params["messages"],
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
                message=f"Rate limit error: {e.message}",
                extra=LLMExtra(data={"error": str(e), "model": self.name}),
            )
        except APIConnectionError as e:
            logger.error(f"API connection error: {e}")
            return LLMResponse(
                success=False,
                message=f"API connection error: {e!s}",
                extra=LLMExtra(data={"error": str(e), "model": self.name}),
            )
        except APIStatusError as e:
            logger.error(f"API status error: {e}")
            return LLMResponse(
                success=False,
                message=f"API status error: {e.message}",
                extra=LLMExtra(
                    data={
                        "error": str(e),
                        "status_code": e.status_code,
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
        response: ChatCompletion,
        tools: list["Tool"] | None = None,
        response_format: type[BaseModel] | BaseModel | dict | None = None,
    ) -> LLMResponse:
        """实现 `_format_response` 的业务逻辑。"""
        try:
            if not response.choices:
                return LLMResponse(
                    success=False,
                    message="No choices in response",
                    extra=LLMExtra(
                        data={
                            "raw_response": response.model_dump()
                            if hasattr(response, "model_dump")
                            else str(response)
                        }
                    ),
                )

            message = response.choices[0].message
            usage = self._get_usage(response)
            finish_reason = response.choices[0].finish_reason
            reasoning = self._get_reasoning(message)

            # 说明相关实现细节。
            if tools and message.tool_calls:
                # 转换并规范化数据。
                formatted_lines = []
                functions = []

                for tool_call in message.tool_calls:
                    function_info = tool_call.function
                    name = function_info.name
                    arguments_str = function_info.arguments

                    # 处理输入参数。
                    import json

                    try:
                        arguments = (
                            json.loads(arguments_str)
                            if isinstance(arguments_str, str)
                            else arguments_str
                        )
                    except json.JSONDecodeError:
                        arguments = {}

                    # 处理输入参数。
                    if arguments:
                        args_str = ", ".join(
                            [f"{k}={v!r}" for k, v in arguments.items()]
                        )
                        formatted_lines.append(f"Calling function {name}({args_str})")
                    else:
                        formatted_lines.append(f"Calling function {name}()")

                    functions.append({"name": name, "args": arguments})

                formatted_message = "\n".join(formatted_lines)

                extra = LLMExtra(
                    data={
                        "raw_response": response.model_dump()
                        if hasattr(response, "model_dump")
                        else str(response),
                        "functions": functions,
                        "usage": usage,
                        "finish_reason": finish_reason,
                        "reasoning": reasoning,
                    }
                )

                return LLMResponse(success=True, message=formatted_message, extra=extra)

            # 组装并返回结果。
            elif (
                response_format
                and isinstance(response_format, type)
                and issubclass(response_format, BaseModel)
            ):
                content = message.content or ""
                if not content:
                    return LLMResponse(
                        success=False,
                        message="Empty response content from model",
                        extra=LLMExtra(
                            data={
                                "raw_response": response.model_dump()
                                if hasattr(response, "model_dump")
                                else str(response)
                            }
                        ),
                    )

                # 转换并规范化数据。
                import json

                try:
                    data = json.loads(content)
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
                            else str(response),
                            "usage": usage,
                            "finish_reason": finish_reason,
                            "reasoning": reasoning,
                        },
                    )

                    return LLMResponse(
                        success=True, message=formatted_message, extra=extra
                    )
                except json.JSONDecodeError as e:
                    return LLMResponse(
                        success=False,
                        message=f"Failed to parse JSON from response: {e}",
                        extra=LLMExtra(data={"error": str(e), "content": content}),
                    )
                except Exception as e:
                    return LLMResponse(
                        success=False,
                        message=f"Failed to validate response against schema: {e}",
                        extra=LLMExtra(data={"error": str(e), "content": content}),
                    )

            # 组装并返回结果。
            else:
                content = message.content or ""

                extra = LLMExtra(
                    data={
                        "raw_response": response.model_dump()
                        if hasattr(response, "model_dump")
                        else str(response),
                        "usage": usage,
                        "finish_reason": finish_reason,
                        "reasoning": reasoning,
                    }
                )

                return LLMResponse(success=True, message=content, extra=extra)

        except Exception as e:
            logger.error(f"Failed to format response: {e}")
            return LLMResponse(
                success=False,
                message=f"Failed to format response: {e}",
                extra=LLMExtra(data={"error": str(e)}),
            )
