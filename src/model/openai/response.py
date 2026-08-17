from collections.abc import Iterable, Mapping
from typing import Any, Literal, Optional, Union, List, Dict, Type
import httpx

try:
    from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError
    from openai.types.shared.chat_model import ChatModel
    from openai.types.shared_params.reasoning_effort import ReasoningEffort
except ImportError:
    # 执行回退或重试逻辑。
    AsyncOpenAI = None
    APIConnectionError = Exception
    APIStatusError = Exception
    RateLimitError = Exception
    ChatModel = str
    ReasoningEffort = str

from pydantic import BaseModel, Field, ConfigDict

from src.message.types import Message
from src.model.openai.serializer import OpenAIResponseSerializer, OpenAIChatSerializer
from src.model.types import LLMResponse, LLMExtra
from src.logger import logger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.tool.types import Tool


class ResponseOpenAI(BaseModel):
    """定义 `ResponseOpenAI`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # 配置相关参数。
    model: Union[ChatModel, str]

    # 组装并返回结果。
    reasoning: Optional[Dict[str, Any]] = None
    max_output_tokens: Optional[int] = 16384
    temperature: Optional[float] = None  # 处理模型调用。

    # 处理输入参数。
    api_key: Optional[str] = None
    organization: Optional[str] = None
    project: Optional[str] = None
    base_url: Optional[Union[str, httpx.URL]] = None
    websocket_base_url: Optional[Union[str, httpx.URL]] = None
    timeout: Optional[Union[float, httpx.Timeout]] = None
    max_retries: int = 5
    default_headers: Optional[Mapping[str, str]] = None
    default_query: Optional[Mapping[str, object]] = None
    http_client: Optional[httpx.AsyncClient] = None
    _strict_response_validation: bool = False

    @property
    def provider(self) -> str:
        return 'openai'

    def _get_client_params(self) -> dict[str, Any]:
        """实现 `_get_client_params` 的业务逻辑。"""
        base_params = {
            'api_key': self.api_key,
            'organization': self.organization,
            'project': self.project,
            'base_url': self.base_url,
            'websocket_base_url': self.websocket_base_url,
            'timeout': self.timeout,
            'max_retries': self.max_retries,
            'default_headers': self.default_headers,
            'default_query': self.default_query,
            '_strict_response_validation': self._strict_response_validation,
        }

        # 创建所需对象。
        client_params = {k: v for k, v in base_params.items() if v is not None}

        # 说明相关实现细节。
        if self.http_client is not None:
            client_params['http_client'] = self.http_client

        return client_params

    def get_client(self) -> AsyncOpenAI:
        """获取与 `get_client` 对应的数据或状态。"""
        if AsyncOpenAI is None:
            raise ImportError("openai package is required. Install it with: pip install openai")

        client_params = self._get_client_params()
        return AsyncOpenAI(**client_params)

    @property
    def name(self) -> str:
        return str(self.model)

    def _get_usage(self, response: Any) -> Optional[Dict[str, Any]]:
        """实现 `_get_usage` 的业务逻辑。"""
        usage = None
        try:
            if hasattr(response, 'usage') and response.usage is not None:
                usage_obj = response.usage

                # 组装并返回结果。
                input_tokens = getattr(usage_obj, 'input_tokens', None)
                output_tokens = getattr(usage_obj, 'output_tokens', None)
                total_tokens = getattr(usage_obj, 'total_tokens', None)

                # 执行回退或重试逻辑。
                if input_tokens is None:
                    input_tokens = getattr(usage_obj, 'prompt_tokens', 0)
                if output_tokens is None:
                    output_tokens = getattr(usage_obj, 'completion_tokens', 0)
                if total_tokens is None:
                    total_tokens = getattr(usage_obj, 'total_tokens', 0)

                usage = {
                    'prompt_tokens': input_tokens,
                    'completion_tokens': output_tokens,
                    'total_tokens': total_tokens,
                }

                # 说明相关实现细节。
                if hasattr(usage_obj, 'output_tokens_details'):
                    details = usage_obj.output_tokens_details
                    if details and hasattr(details, 'reasoning_tokens'):
                        reasoning_tokens = details.reasoning_tokens
                        if reasoning_tokens is not None:
                            usage['reasoning_tokens'] = reasoning_tokens
                elif hasattr(usage_obj, 'completion_tokens_details'):
                    # 执行回退或重试逻辑。
                    details = usage_obj.completion_tokens_details
                    if details and hasattr(details, 'reasoning_tokens'):
                        reasoning_tokens = details.reasoning_tokens
                        if reasoning_tokens is not None:
                            usage['reasoning_tokens'] = reasoning_tokens

                # 处理记忆或缓存状态。
                if hasattr(usage_obj, 'input_tokens_details'):
                    prompt_details = usage_obj.input_tokens_details
                    if prompt_details and hasattr(prompt_details, 'cached_tokens'):
                        usage['prompt_cached_tokens'] = prompt_details.cached_tokens
                elif hasattr(usage_obj, 'prompt_tokens_details'):
                    # 执行回退或重试逻辑。
                    prompt_details = usage_obj.prompt_tokens_details
                    if prompt_details and hasattr(prompt_details, 'cached_tokens'):
                        usage['prompt_cached_tokens'] = prompt_details.cached_tokens
        except (AttributeError, TypeError) as e:
            logger.debug(f"Error extracting usage: {e}")
            pass

        return usage

    def _get_reasoning(self, response: Any) -> Optional[str]:
        """实现 `_get_reasoning` 的业务逻辑。"""
        reasoning = None
        try:
            # 组装并返回结果。
            if hasattr(response, 'output') and response.output is not None:
                # 校验输入与当前状态。
                output = response.output
                if isinstance(output, list):
                    for item in output:
                        # 说明相关实现细节。
                        item_type = None
                        if hasattr(item, 'type'):
                            item_type = item.type
                        elif isinstance(item, dict):
                            item_type = item.get('type')

                        if item_type == 'reasoning':
                            # 说明相关实现细节。
                            if hasattr(item, 'content') and item.content:
                                reasoning = item.content
                                break
                            elif isinstance(item, dict) and item.get('content'):
                                reasoning = item.get('content')
                                break
                            elif hasattr(item, 'text') and item.text:
                                reasoning = item.text
                                break
                            elif isinstance(item, dict) and item.get('text'):
                                reasoning = item.get('text')
                                break
                            elif hasattr(item, 'summary') and item.summary:
                                reasoning = item.summary
                                break
                            elif isinstance(item, dict) and item.get('summary'):
                                reasoning = item.get('summary')
                                break

                            # 更新相关状态。
                            if reasoning is None:
                                reasoning = None
                                break
                elif isinstance(output, dict):
                    if output.get('type') == 'reasoning':
                        reasoning = output.get('content') or output.get('text') or output.get('summary') or None

            # 校验输入与当前状态。
            if reasoning is None and hasattr(response, 'reasoning_details'):
                reasoning_details = response.reasoning_details
                if reasoning_details:
                    for detail in reasoning_details:
                        if hasattr(detail, 'type'):
                            detail_type = detail.type
                            if detail_type == "reasoning.text" and hasattr(detail, 'text'):
                                reasoning = detail.text
                                break
                            elif detail_type == "reasoning.summary" and hasattr(detail, 'summary'):
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
        except (AttributeError, KeyError, TypeError, IndexError) as e:
            logger.debug(f"Error extracting reasoning: {e}")
            pass

        return reasoning

    def _extract_output_text(self, response: Any) -> str:
        """实现 `_extract_output_text` 的业务逻辑。"""
        text = ""
        try:
            # 组装并返回结果。
            if hasattr(response, 'output_text') and response.output_text is not None:
                return response.output_text

            # 组装并返回结果。
            if hasattr(response, 'output') and response.output is not None:
                output = response.output
                if isinstance(output, list):
                    for item in output:
                        if hasattr(item, 'type'):
                            if item.type == 'message':
                                if hasattr(item, 'content'):
                                    content = item.content
                                    if isinstance(content, list):
                                        for content_item in content:
                                            if hasattr(content_item, 'type') and content_item.type == 'output_text':
                                                if hasattr(content_item, 'text'):
                                                    return content_item.text
                                            elif isinstance(content_item, dict) and content_item.get('type') == 'output_text':
                                                return content_item.get('text', '')
                        elif isinstance(item, dict):
                            if item.get('type') == 'message':
                                content = item.get('content', [])
                                for content_item in content:
                                    if content_item.get('type') == 'output_text':
                                        return content_item.get('text', '')
                elif isinstance(output, dict):
                    if output.get('type') == 'message':
                        content = output.get('content', [])
                        for content_item in content:
                            if content_item.get('type') == 'output_text':
                                return content_item.get('text', '')
        except (AttributeError, KeyError, TypeError, IndexError):
            pass

        return text

    async def _build_params(
        self,
        messages: List[Message],
        tools: Optional[List["Tool"]] = None,
        response_format: Optional[Union[Type[BaseModel], BaseModel, Dict]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """实现 `_build_params` 的业务逻辑。"""
        # 组装并返回结果。
        input_messages = OpenAIResponseSerializer.serialize_messages(messages)

        # 创建所需对象。
        params: Dict[str, Any] = {
            "model": self.model,
            "input": input_messages,
        }

        # 组装并返回结果。
        if self.reasoning is not None:
            params.update(self.reasoning)

        # 组装并返回结果。
        if self.max_output_tokens is not None:
            params["max_output_tokens"] = self.max_output_tokens

        # 组装并返回结果。
        if response_format:
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                # 组装并返回结果。
                # 组装并返回结果。
                # 说明相关实现细节。
                json_schema = response_format.model_json_schema()
                # 转换并规范化数据。
                optimized = OpenAIChatSerializer.serialize_response_format(response_format)
                # 转换并规范化数据。
                schema = optimized['json_schema']['schema']

                # 创建所需对象。
                params["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": response_format.__name__,
                        "strict": True,
                        "schema": schema
                    }
                }
            elif isinstance(response_format, BaseModel):
                # 组装并返回结果。
                model_class = type(response_format)
                optimized = OpenAIChatSerializer.serialize_response_format(model_class)
                schema = optimized['json_schema']['schema']

                params["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": model_class.__name__,
                        "strict": True,
                        "schema": schema
                    }
                }
            elif isinstance(response_format, dict):
                # 加载所需数据。
                if "text" in response_format:
                    # 加载所需数据。
                    params["text"] = response_format["text"]
                elif "type" in response_format and "name" in response_format and "schema" in response_format:
                    # 加载所需数据。
                    params["text"] = {
                        "format": response_format
                    }
                elif "type" in response_format and "json_schema" in response_format:
                    # 组装并返回结果。
                    json_schema_obj = response_format["json_schema"]
                    params["text"] = {
                        "format": {
                            "type": "json_schema",
                            "name": json_schema_obj.get("name", "response"),
                            "strict": json_schema_obj.get("strict", True),
                            "schema": json_schema_obj.get("schema", {})
                        }
                    }
                else:
                    # 说明相关实现细节。
                    params["text"] = {
                        "format": {
                            "type": "json_schema",
                            "name": "response",
                            "strict": True,
                            "schema": response_format
                        }
                    }
            else:
                logger.warning(f"Unsupported response_format type: {type(response_format)}")

        # 组装并返回结果。
        if tools:
            logger.warning("Tools may not be supported in responses API")
            # 处理工具调用。

        if stream:
            logger.warning("Streaming may not be supported in responses API")
            # 说明相关实现细节。

        # 说明相关实现细节。
        params.update(kwargs)

        return {
            "input": input_messages,
            "params": params,
        }

    async def _call_model(
        self,
        input_messages: List[Dict[str, Any]],
        **params: Any,
    ) -> Any:
        """实现 `_call_model` 的业务逻辑。"""
        client = self.get_client()
        response = await client.responses.create(**params)

        return response

    async def __call__(
        self,
        messages: List[Message],
        tools: Optional[List["Tool"]] = None,
        response_format: Optional[Union[Type[BaseModel], BaseModel, Dict]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        """执行组件调用并返回结果。"""
        if AsyncOpenAI is None:
            raise ImportError("openai package is required. Install it with: pip install openai")

        try:
            params = await self._build_params(
                messages=messages,
                tools=tools,
                response_format=response_format,
                stream=stream,
                **kwargs,
            )

            response = await self._call_model(
                input_messages=params["input"],
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
                extra=LLMExtra(data={"error": str(e), "model": self.name})
            )
        except APIConnectionError as e:
            logger.error(f"API connection error: {e}")
            return LLMResponse(
                success=False,
                message=f"API connection error: {str(e)}",
                extra=LLMExtra(data={"error": str(e), "model": self.name})
            )
        except APIStatusError as e:
            logger.error(f"API status error: {e}")
            return LLMResponse(
                success=False,
                message=f"API status error: {e.message}",
                extra=LLMExtra(data={"error": str(e), "status_code": e.status_code, "model": self.name})
            )
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return LLMResponse(
                success=False,
                message=f"Unexpected error: {str(e)}",
                extra=LLMExtra(data={"error": str(e), "model": self.name})
            )

    async def _format_response(
        self,
        response: Any,
        tools: Optional[List["Tool"]] = None,
        response_format: Optional[Union[Type[BaseModel], BaseModel, Dict]] = None,
    ) -> LLMResponse:
        """实现 `_format_response` 的业务逻辑。"""
        try:
            usage = self._get_usage(response)
            reasoning = self._get_reasoning(response)
            output_text = self._extract_output_text(response)

            # 组装并返回结果。
            if response_format and isinstance(response_format, type) and issubclass(response_format, BaseModel):
                if not output_text:
                    return LLMResponse(
                        success=False,
                        message="Empty response content from model",
                        extra=LLMExtra(data={"raw_response": response.model_dump() if hasattr(response, 'model_dump') else str(response)})
                    )

                # 转换并规范化数据。
                import json
                try:
                    data = json.loads(output_text)
                    parsed_model = response_format.model_validate(data)

                    # 转换并规范化数据。
                    model_name = response_format.__name__
                    model_dict = parsed_model.model_dump()

                    field_lines = []
                    for field_name, field_value in model_dict.items():
                        field_lines.append(f"{field_name}={field_value!r}")

                    formatted_message = f"Response result:\n\n{model_name}(\n"
                    formatted_message += ",\n".join(f"    {line}" for line in field_lines)
                    formatted_message += "\n)"

                    extra = LLMExtra(
                        parsed_model=parsed_model,
                        data={
                            "raw_response": response.model_dump() if hasattr(response, 'model_dump') else str(response),
                            "usage": usage,
                            "reasoning": reasoning,
                        }
                    )

                    return LLMResponse(
                        success=True,
                        message=formatted_message,
                        extra=extra
                    )
                except json.JSONDecodeError as e:
                    return LLMResponse(
                        success=False,
                        message=f"Failed to parse JSON from response: {e}",
                        extra=LLMExtra(data={"error": str(e), "content": output_text})
                    )
                except Exception as e:
                    return LLMResponse(
                        success=False,
                        message=f"Failed to validate response against schema: {e}",
                        extra=LLMExtra(data={"error": str(e), "content": output_text})
                    )

            # 组装并返回结果。
            else:
                extra = LLMExtra(
                    data={
                        "raw_response": response.model_dump() if hasattr(response, 'model_dump') else str(response),
                        "usage": usage,
                        "reasoning": reasoning,
                    }
                )

                return LLMResponse(
                    success=True,
                    message=output_text,
                    extra=extra
                )

        except Exception as e:
            logger.error(f"Failed to format response: {e}")
            return LLMResponse(
                success=False,
                message=f"Failed to format response: {e}",
                extra=LLMExtra(data={"error": str(e)})
            )
