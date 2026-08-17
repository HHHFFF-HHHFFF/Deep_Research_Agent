import httpx
import json
from typing import Optional, Dict, Any, List, Union
from collections.abc import Mapping

try:
    from openai.types.chat.chat_completion import ChatCompletion, Choice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage
    from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
    from openai.types.chat.chat_completion_message_function_tool_call_param import Function as ChatCompletionFunction
    from openai.types.completion_usage import CompletionUsage
except ImportError:
    # 执行回退或重试逻辑。
    ChatCompletion = dict
    Choice = dict
    ChatCompletionMessage = dict
    ChatCompletionMessageToolCall = dict
    ChatCompletionFunction = dict
    CompletionUsage = dict

from src.logger import logger


class OpenRouterCompletions:
    """定义 `OpenRouterCompletions`，封装相关数据与行为。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = "https://openrouter.ai/api/v1",
        http_referer: Optional[str] = None,
        x_title: Optional[str] = None,
        default_headers: Optional[Mapping[str, str]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = 300.0,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/') if base_url else "https://openrouter.ai/api/v1"
        self.http_referer = http_referer
        self.x_title = x_title
        self.default_headers = default_headers
        self.timeout = timeout
        self._http_client = http_client
        self._endpoint = "/chat/completions"

    def _get_headers(self) -> Dict[str, str]:
        """实现 `_get_headers` 的业务逻辑。"""
        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # 说明相关实现细节。
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.x_title:
            headers["X-Title"] = self.x_title

        # 说明相关实现细节。
        if self.default_headers:
            headers.update(self.default_headers)

        return headers

    def _get_api_url(self) -> str:
        """实现 `_get_api_url` 的业务逻辑。"""
        return f"{self.base_url}{self._endpoint}"

    def _dict_to_chat_completion(self, data: Dict[str, Any]) -> ChatCompletion:
        """实现 `_dict_to_chat_completion` 的业务逻辑。"""
        if ChatCompletion == dict:
            # 组装并返回结果。
            return data

        # 说明相关实现细节。
        choices_data = data.get("choices", [])
        choices = []
        for choice_data in choices_data:
            message_data = choice_data.get("message", {})

            # 创建所需对象。
            tool_calls = None
            if message_data.get("tool_calls"):
                tool_calls = []
                for tc_data in message_data["tool_calls"]:
                    function_data = tc_data.get("function", {})
                    tool_call = ChatCompletionMessageToolCall(
                        id=tc_data.get("id", ""),
                        type="function",
                        function=ChatCompletionFunction(
                            name=function_data.get("name", ""),
                            arguments=function_data.get("arguments", "{}")
                        )
                    )
                    tool_calls.append(tool_call)

            # 创建所需对象。
            message = ChatCompletionMessage(
                role=message_data.get("role", "assistant"),
                content=message_data.get("content"),
                tool_calls=tool_calls,
            )

            # 创建所需对象。
            choice = Choice(
                finish_reason=choice_data.get("finish_reason"),
                index=choice_data.get("index", 0),
                message=message,
            )
            choices.append(choice)

        # 创建所需对象。
        usage_data = data.get("usage", {})
        usage = None
        if usage_data:
            usage = CompletionUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )

        # 创建所需对象。
        return ChatCompletion(
            id=data.get("id", ""),
            choices=choices,
            created=data.get("created", 0),
            model=data.get("model", ""),
            object="chat.completion",
            usage=usage,
        )

    async def create(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        plugins: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        """实现 `create` 的业务逻辑。"""
        # 加载所需数据。
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        # 说明相关实现细节。
        if plugins is not None:
            payload["plugins"] = plugins

        # 处理输入参数。
        payload["usage"] = {
            "include": True
        }

        # 处理输入参数。
        for key, value in kwargs.items():
            if value is None:
                continue
            if key == "max_completion_tokens":  # 说明相关实现细节。
                payload["max_tokens"] = value
            else:
                payload[key] = value

        # 说明相关实现细节。
        headers = self._get_headers()
        api_url = self._get_api_url()

        # 说明相关实现细节。
        timeout_obj = self.timeout
        if isinstance(timeout_obj, (int, float)):
            timeout_obj = httpx.Timeout(timeout_obj)

        # 处理输入参数。
        try:
            # 创建所需对象。
            if self._http_client:
                client = self._http_client
                should_close = False
            else:
                client = httpx.AsyncClient(timeout=timeout_obj)
                should_close = True

            try:
                response = await client.post(
                    url=api_url,
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                response_dict = response.json()

                # 转换并规范化数据。
                return self._dict_to_chat_completion(response_dict)
            finally:
                if should_close:
                    await client.aclose()
        except httpx.HTTPStatusError as e:
            logger.error(f"OpenRouter API HTTP error: {e}")
            try:
                error_detail = e.response.json()
                raise Exception(f"OpenRouter API request failed: {error_detail}")
            except:
                raise Exception(f"OpenRouter API request failed: {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"OpenRouter API request error: {e}")
            raise Exception(f"OpenRouter API request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in OpenRouter API request: {e}")
            raise

class OpenRouterChatNamespace:
    """定义 `OpenRouterChatNamespace`，封装相关数据与行为。"""

    def __init__(self, completions: OpenRouterCompletions):
        self.completions = completions


class OpenRouterClient:
    """定义 `OpenRouterClient`，封装相关数据与行为。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = "https://openrouter.ai/api/v1",
        http_referer: Optional[str] = None,
        x_title: Optional[str] = None,
        default_headers: Optional[Mapping[str, str]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = 300.0,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.http_referer = http_referer
        self.x_title = x_title
        self.default_headers = default_headers
        self.timeout = timeout
        self._http_client = http_client

        # 初始化相关状态。
        completions = OpenRouterCompletions(
            api_key=api_key,
            base_url=base_url,
            http_referer=http_referer,
            x_title=x_title,
            default_headers=default_headers,
            timeout=timeout,
            http_client=http_client
        )

        # 创建所需对象。
        self.chat = OpenRouterChatNamespace(completions=completions)
