from typing import Any, Optional, Union, List, Dict, ClassVar
import httpx

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    from google.api_core import exceptions as google_exceptions
except ImportError:
    genai = None
    HarmCategory = None
    HarmBlockThreshold = None
    google_exceptions = None

from pydantic import BaseModel, Field, ConfigDict

import json
from src.logger import logger
from src.model.types import LLMResponse, LLMExtra
from src.message.types import Message, HumanMessage, SystemMessage, AssistantMessage
from src.model.google.serializer import GoogleChatSerializer
from typing import Type, TYPE_CHECKING

if TYPE_CHECKING:
    from src.tool.types import Tool

class ChatGoogle(BaseModel):
    """定义 `ChatGoogle`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # 配置相关参数。
    model: str

    # 处理模型调用。
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_output_tokens: Optional[int] = 8192
    reasoning: Optional[Dict[str, Any]] = None

    # 处理输入参数。
    api_key: Optional[str] = None
    timeout: Optional[Union[float, httpx.Timeout]] = None
    max_retries: int = 5

    @property
    def provider(self) -> str:
        return 'google'

    def _get_client_params(self) -> Dict[str, Any]:
        """实现 `_get_client_params` 的业务逻辑。"""
        if genai is None:
            raise ImportError("google-generativeai package is required. Install it with: pip install google-generativeai")

        # 配置相关参数。
        if self.api_key:
            genai.configure(api_key=self.api_key)
        elif not genai.api_key:
            # 说明相关实现细节。
            import os
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
            else:
                raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")

        return {}

    def get_client(self, system_instruction: Optional[str] = None):
        """获取与 `get_client` 对应的数据或状态。"""
        if genai is None:
            raise ImportError("google-generativeai package is required. Install it with: pip install google-generativeai")

        self._get_client_params()

        # 处理模型调用。
        if system_instruction:
            return genai.GenerativeModel(
                self.model,
                system_instruction=system_instruction
            )
        else:
            return genai.GenerativeModel(self.model)

    @property
    def name(self) -> str:
        return str(self.model)

    def _get_usage(self, response) -> Optional[Dict[str, Any]]:
        """实现 `_get_usage` 的业务逻辑。"""
        if hasattr(response, 'usage_metadata') and response.usage_metadata is not None:
            return response.usage_metadata.model_dump()
        else:
            return None

    async def _build_params(
        self,
        messages: List[Message],
        tools: Optional[List["Tool"]] = None,
        response_format: Optional[Union[Type[BaseModel], BaseModel, Dict]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """实现 `_build_params` 的业务逻辑。"""
        # 转换并规范化数据。
        system_instruction, gemini_contents = GoogleChatSerializer.serialize_messages(messages)

        # 配置相关参数。
        generation_config: Dict[str, Any] = {}

        if self.temperature is not None:
            generation_config['temperature'] = self.temperature
        if self.top_p is not None:
            generation_config['top_p'] = self.top_p
        if self.top_k is not None:
            generation_config['top_k'] = self.top_k
        if self.max_output_tokens is not None:
            generation_config['max_output_tokens'] = self.max_output_tokens
        if self.reasoning is not None:
            generation_config.update(self.reasoning)
        # 组装并返回结果。
        if response_format:
            try:
                response_format_config = GoogleChatSerializer.serialize_response_format(response_format)
                generation_config.update(response_format_config)
            except ValueError as e:
                logger.warning(f"Failed to serialize response_format: {e}")

        # 转换并规范化数据。
        tools_config = None
        if tools:
            formatted_tools = GoogleChatSerializer.serialize_tools(tools)
            if formatted_tools:
                tools_config = formatted_tools

        # 配置相关参数。
        for key, value in kwargs.items():
            if key not in ['contents', 'system_instruction', 'tools', 'generation_config']:
                generation_config[key] = value

        return {
            "system_instruction": system_instruction,
            "contents": gemini_contents,
            "generation_config": generation_config,
            "tools": tools_config,
            "stream": stream,
        }

    async def _call_model(
        self,
        contents: List[Dict[str, Any]],
        system_instruction: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """实现 `_call_model` 的业务逻辑。"""
        # 处理模型调用。
        client = self.get_client(system_instruction=system_instruction)

        # 处理输入参数。
        call_kwargs: Dict[str, Any] = {}

        if contents:
            call_kwargs['contents'] = contents
        if generation_config:
            call_kwargs['generation_config'] = generation_config
        if tools:
            call_kwargs['tools'] = tools

        # 说明相关实现细节。
        if stream:
            # 说明相关实现细节。
            # 说明相关实现细节。
            logger.warning("Streaming is not yet fully implemented for Google Gemini API")
            call_kwargs['stream'] = True

        # 说明相关实现细节。
        # 处理输入参数。
        # 说明相关实现细节。
        response = await self._async_generate_content(client, **call_kwargs)

        return response

    async def _async_generate_content(self, client, **kwargs):
        """实现 `_async_generate_content` 的业务逻辑。"""
        import asyncio

        # 加载所需数据。
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: client.generate_content(**kwargs))

    async def __call__(
        self,
        messages: List[Message],
        tools: Optional[List["Tool"]] = None,
        response_format: Optional[Union[Type[BaseModel], BaseModel, Dict]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        """执行组件调用并返回结果。"""
        if genai is None:
            raise ImportError("google-generativeai package is required. Install it with: pip install google-generativeai")

        try:
            params = await self._build_params(
                messages=messages,
                tools=tools,
                response_format=response_format,
                stream=stream,
                **kwargs,
            )

            response = await self._call_model(
                contents=params["contents"],
                system_instruction=params.get("system_instruction"),
                generation_config=params.get("generation_config"),
                tools=params.get("tools"),
                stream=params.get("stream", False),
            )

            return await self._format_response(
                response=response,
                tools=tools,
                response_format=response_format,
            )

        except Exception as e:
            error_msg = str(e)
            status_code = None

            # 处理异常情况。
            if google_exceptions and isinstance(e, google_exceptions.GoogleAPIError):
                status_code = getattr(e, 'status_code', None)

            logger.error(f"API error: {e}")
            return LLMResponse(
                success=False,
                message=f"API error: {error_msg}",
                extra=LLMExtra(data={"error": error_msg, "status_code": status_code, "model": self.name})
            )

    async def _format_response(
        self,
        response: Any,
        tools: Optional[List["Tool"]] = None,
        response_format: Optional[Union[Type[BaseModel], BaseModel, Dict]] = None,
    ) -> LLMResponse:
        """实现 `_format_response` 的业务逻辑。"""
        try:
            # 组装并返回结果。
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
            elif isinstance(response, dict):
                candidates = response.get("candidates", [])
                candidate = candidates[0] if candidates else {}
            else:
                candidate = {}

            if not candidate:
                return LLMResponse(
                    success=False,
                    message="No candidates in response",
                    extra=LLMExtra(data={"raw_response": str(response)})
                )

            # 说明相关实现细节。
            text_parts = []
            function_calls = []

            if hasattr(candidate, 'content'):
                # 组装并返回结果。
                content = candidate.content
                if hasattr(content, 'parts'):
                    parts = content.parts
                else:
                    parts = []
            elif isinstance(candidate, dict):
                # 转换并规范化数据。
                content = candidate.get("content", {})
                parts = content.get("parts", [])
            else:
                parts = []

            for part in parts:
                if hasattr(part, 'text'):
                    # 组装并返回结果。
                    text_parts.append(part.text)
                elif hasattr(part, 'function_call'):
                    # 组装并返回结果。
                    func_call = part.function_call
                    function_calls.append({
                        "name": func_call.name if hasattr(func_call, 'name') else "",
                        "args": func_call.args if hasattr(func_call, 'args') else {},
                    })
                elif isinstance(part, dict):
                    # 转换并规范化数据。
                    if "text" in part:
                        text_parts.append(part.get("text", ""))
                    elif "function_call" in part:
                        func_call = part["function_call"]
                        function_calls.append({
                            "name": func_call.get("name", ""),
                            "args": func_call.get("args", {}),
                        })

            message_text = "\n".join(text_parts) if text_parts else ""

            usage = self._get_usage(response)
            finish_reason = None
            if hasattr(candidate, 'finish_reason'):
                finish_reason = candidate.finish_reason
            elif isinstance(candidate, dict):
                finish_reason = candidate.get("finish_reason")

            # 说明相关实现细节。
            if tools and function_calls:
                formatted_lines = []
                functions = []

                for func_call in function_calls:
                    name = func_call.get("name", "")
                    args_data = func_call.get("args", {})

                    # 处理输入参数。
                    if args_data:
                        args_str = ", ".join([f"{k}={v!r}" for k, v in args_data.items()])
                        formatted_lines.append(f"Calling function {name}({args_str})")
                    else:
                        formatted_lines.append(f"Calling function {name}()")

                    functions.append({
                        "name": name,
                        "args": args_data
                    })

                formatted_message = "\n".join(formatted_lines)

                extra = LLMExtra(
                    data={
                        "raw_response": str(response),
                        "functions": functions,
                        "usage": usage,
                        "finish_reason": finish_reason,
                    }
                )

                return LLMResponse(
                    success=True,
                    message=formatted_message,
                    extra=extra
                )

            # 组装并返回结果。
            elif response_format and isinstance(response_format, type) and issubclass(response_format, BaseModel):
                if not message_text:
                    return LLMResponse(
                        success=False,
                        message="Empty response content from model",
                        extra=LLMExtra(data={"raw_response": str(response)})
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
                    formatted_message += ",\n".join(f"    {line}" for line in field_lines)
                    formatted_message += "\n)"

                    extra = LLMExtra(
                        parsed_model=parsed_model,
                        data={
                            "raw_response": str(response),
                            "usage": usage,
                            "finish_reason": finish_reason,
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
                        extra=LLMExtra(data={"error": str(e), "content": message_text})
                    )
                except Exception as e:
                    return LLMResponse(
                        success=False,
                        message=f"Failed to validate response against schema: {e}",
                        extra=LLMExtra(data={"error": str(e), "content": message_text})
                    )

            # 组装并返回结果。
            else:
                extra = LLMExtra(
                    data={
                        "raw_response": str(response),
                        "usage": usage,
                        "finish_reason": finish_reason,
                    }
                )

                return LLMResponse(
                    success=True,
                    message=message_text,
                    extra=extra
                )

        except Exception as e:
            logger.error(f"Failed to format response: {e}")
            return LLMResponse(
                success=False,
                message=f"Failed to format response: {e}",
                extra=LLMExtra(data={"error": str(e)})
            )
