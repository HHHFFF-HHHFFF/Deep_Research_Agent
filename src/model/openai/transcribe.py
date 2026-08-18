import base64
import os
from collections.abc import Mapping
from typing import Any, BinaryIO

import aiohttp
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
from src.message.types import ContentPartAudio, HumanMessage, Message
from src.model.types import LLMResponse
from src.utils import assemble_project_path, open_binary_file


class TranscribeOpenAI(BaseModel):
    """定义 `TranscribeOpenAI`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # 配置相关参数。
    model: ChatModel | str = "gpt-4o-transcribe"

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
    language: str | None = None
    prompt: str | None = None
    response_format: str | None = None  # 说明相关实现细节。
    temperature: float | None = None
    timestamp_granularities: list[str] | None = None  # 说明相关实现细节。

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

    def _extract_audio_from_messages(
        self, messages: list[Message]
    ) -> tuple[str | BinaryIO | None, str | None, str | None]:
        """实现 `_extract_audio_from_messages` 的业务逻辑。"""
        audio_file = None
        prompt_text = None
        filename = None

        for message in messages:
            if isinstance(message, HumanMessage):
                if isinstance(message.content, list):
                    for part in message.content:
                        if isinstance(part, ContentPartAudio):
                            audio_url = part.audio_url.url
                            # 说明相关实现细节。
                            if audio_url.startswith("data:"):
                                # 说明相关实现细节。
                                # 转换并规范化数据。
                                try:
                                    if "," in audio_url:
                                        header, data = audio_url.split(",", 1)
                                        audio_bytes = base64.b64decode(data)
                                        # 创建所需对象。
                                        from io import BytesIO

                                        audio_file = BytesIO(audio_bytes)

                                        # 处理文件与路径。
                                        if (
                                            "audio/mpeg" in header
                                            or "audio/mp3" in header
                                        ):
                                            filename = "audio.mp3"
                                        elif "audio/wav" in header:
                                            filename = "audio.wav"
                                        elif "audio/ogg" in header:
                                            filename = "audio.ogg"
                                        elif "audio/flac" in header:
                                            filename = "audio.flac"
                                        elif "audio/m4a" in header:
                                            filename = "audio.m4a"
                                        else:
                                            filename = "audio.mp3"  # 说明相关实现细节。
                                    else:
                                        logger.error(
                                            f"Invalid data URL format: {audio_url}"
                                        )
                                        return None, None, None
                                except Exception as e:
                                    logger.error(f"Failed to decode base64 audio: {e}")
                                    return None, None, None
                            # 处理文件与路径。
                            elif audio_url.startswith("file://"):
                                # 移除相关数据或组件。
                                file_path = audio_url[7:]
                                # 处理文件与路径。
                                if not os.path.isabs(file_path):
                                    file_path = assemble_project_path(file_path)
                                audio_file = file_path
                                filename = os.path.basename(file_path)
                            elif os.path.exists(audio_url):
                                # 处理文件与路径。
                                audio_file = audio_url
                                filename = os.path.basename(audio_url)
                            elif os.path.exists(assemble_project_path(audio_url)):
                                # 处理文件与路径。
                                audio_file = assemble_project_path(audio_url)
                                filename = os.path.basename(audio_file)
                            else:
                                # 加载所需数据。
                                logger.info(
                                    f"Audio URL detected, will download: {audio_url}"
                                )
                                audio_file = audio_url
                                # 处理文件与路径。
                                filename = (
                                    os.path.basename(audio_url.split("?")[0])
                                    or "audio.mp3"
                                )
                        elif hasattr(part, "type") and part.type == "text":
                            # 说明相关实现细节。
                            if prompt_text is None:
                                prompt_text = part.text
                            else:
                                prompt_text += " " + part.text
                elif isinstance(message.content, str):
                    # 说明相关实现细节。
                    if prompt_text is None:
                        prompt_text = message.content
                    else:
                        prompt_text += " " + message.content

        return audio_file, prompt_text, filename

    def _cleanup_file_resources(
        self, file_obj: Any, temp_file_path: str | None = None
    ) -> None:
        """实现 `_cleanup_file_resources` 的业务逻辑。"""
        # 清理并释放相关资源。
        if file_obj is not None:
            # 转换并规范化数据。
            actual_file = file_obj[1] if isinstance(file_obj, tuple) else file_obj

            if hasattr(actual_file, "close"):
                # 清理并释放相关资源。
                from io import BytesIO

                if not isinstance(actual_file, BytesIO):
                    try:
                        actual_file.close()
                    except Exception as e:
                        logger.debug(f"Error closing file: {e}")

        # 加载所需数据。
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_file_path}: {e}")

    async def _download_audio_file(self, url: str) -> str | None:
        """实现 `_download_audio_file` 的业务逻辑。"""
        try:
            import tempfile

            async with (
                aiohttp.ClientSession() as session,
                session.get(url) as response,
            ):
                if response.status == 200:
                    # 创建所需对象。
                    ext = os.path.splitext(url)[1] or ".mp3"
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=ext
                    ) as tmp_file:
                        async for chunk in response.content.iter_chunked(8192):
                            tmp_file.write(chunk)
                        return tmp_file.name
                else:
                    logger.error(f"Failed to download audio: HTTP {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error downloading audio file: {e}")
            return None

    async def _build_params(
        self,
        messages: list[Message],
        language: str | None = None,
        response_format: str | None = None,
        temperature: float | None = None,
        timestamp_granularities: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """实现 `_build_params` 的业务逻辑。"""
        # 处理文件与路径。
        audio_file, extracted_prompt, extracted_filename = (
            self._extract_audio_from_messages(messages)
        )
        if audio_file is None:
            raise ValueError("No audio file found in messages")

        # 说明相关实现细节。
        prompt = extracted_prompt
        filename = extracted_filename or "audio.mp3"  # 处理文件与路径。

        # 处理输入参数。
        file_obj = None
        temp_file_path = None

        # 校验输入与当前状态。
        from io import BytesIO

        if isinstance(audio_file, BytesIO) or (
            hasattr(audio_file, "read") and hasattr(audio_file, "seek")
        ):
            # 加载所需数据。
            # 更新相关状态。
            audio_file.seek(0)
            # 转换并规范化数据。
            # 转换并规范化数据。
            file_obj = (filename, audio_file)
        elif isinstance(audio_file, str):
            # 处理文件与路径。
            if audio_file.startswith(("http://", "https://")):
                # 加载所需数据。
                temp_file_path = await self._download_audio_file(audio_file)
                if temp_file_path is None:
                    raise ValueError(
                        f"Failed to download audio file from URL: {audio_file}"
                    )
                file_obj = await open_binary_file(temp_file_path)
            elif audio_file.startswith("file://"):
                # 移除相关数据或组件。
                file_path = audio_file[7:]
                if not os.path.isabs(file_path):
                    file_path = assemble_project_path(file_path)
                if not os.path.exists(file_path):
                    raise ValueError(f"Audio file not found: {file_path}")
                file_obj = await open_binary_file(file_path)
            else:
                # 处理文件与路径。
                if not os.path.exists(audio_file):
                    # 处理文件与路径。
                    file_path = assemble_project_path(audio_file)
                    if not os.path.exists(file_path):
                        raise ValueError(f"Audio file not found: {audio_file}")
                    audio_file = file_path
                file_obj = await open_binary_file(audio_file)
        else:
            raise ValueError(f"Unsupported audio file type: {type(audio_file)}")

        # 创建所需对象。
        params: dict[str, Any] = {
            "model": self.model,
            "file": file_obj,
        }

        # 处理输入参数。
        if language is not None:
            params["language"] = language
        elif self.language is not None:
            params["language"] = self.language

        # 说明相关实现细节。
        if prompt:
            params["prompt"] = prompt
        elif self.prompt is not None:
            params["prompt"] = self.prompt

        if response_format is not None:
            params["response_format"] = response_format
        elif self.response_format is not None:
            params["response_format"] = self.response_format

        if temperature is not None:
            params["temperature"] = temperature
        elif self.temperature is not None:
            params["temperature"] = self.temperature

        if timestamp_granularities is not None:
            params["timestamp_granularities"] = timestamp_granularities
        elif self.timestamp_granularities is not None:
            params["timestamp_granularities"] = self.timestamp_granularities

        # 说明相关实现细节。
        params.update(kwargs)

        return {
            "file": file_obj,
            "params": params,
            "temp_file_path": temp_file_path,
        }

    async def _call_model(
        self,
        file_obj: Any,
        **params: Any,
    ) -> Any:
        """实现 `_call_model` 的业务逻辑。"""
        client = self.get_client()
        response = await client.audio.transcriptions.create(**params)

        return response

    async def _format_response(
        self,
        transcription: Any,
        temp_file_path: str | None = None,
        file_obj: Any | None = None,
    ) -> LLMResponse:
        """实现 `_format_response` 的业务逻辑。"""
        # 组装并返回结果。
        text = ""
        if hasattr(transcription, "text"):
            text = transcription.text
        elif isinstance(transcription, dict):
            text = transcription.get("text", "")
        elif isinstance(transcription, str):
            text = transcription

        # 说明相关实现细节。
        self._cleanup_file_resources(file_obj, temp_file_path)

        # 组装并返回结果。
        extra = {
            "raw_response": transcription.model_dump()
            if hasattr(transcription, "model_dump")
            else str(transcription),
        }

        # 转换并规范化数据。
        if hasattr(transcription, "words"):
            extra["words"] = transcription.words
        if hasattr(transcription, "segments"):
            extra["segments"] = transcription.segments

        return LLMResponse(success=True, message=text, extra=extra)

    async def __call__(
        self,
        messages: list[Message],
        language: str | None = None,
        response_format: str | None = None,
        temperature: float | None = None,
        timestamp_granularities: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """执行组件调用并返回结果。"""
        if AsyncOpenAI is None:
            raise ImportError(
                "openai package is required. Install it with: pip install openai"
            )

        file_obj = None
        temp_file_path = None

        try:
            params = await self._build_params(
                messages=messages,
                language=language,
                response_format=response_format,
                temperature=temperature,
                timestamp_granularities=timestamp_granularities,
                **kwargs,
            )

            file_obj = params["file"]
            temp_file_path = params.get("temp_file_path")

            transcription = await self._call_model(
                file_obj=file_obj,
                **params["params"],
            )

            return await self._format_response(
                transcription=transcription,
                temp_file_path=temp_file_path,
                file_obj=file_obj,
            )

        except RateLimitError as e:
            logger.error(f"Rate limit error: {e}")
            self._cleanup_file_resources(file_obj, temp_file_path)
            return LLMResponse(
                success=False,
                message=f"Rate limit error: {e.message}",
                extra={"error": str(e), "model": self.name},
            )
        except APIConnectionError as e:
            logger.error(f"API connection error: {e}")
            self._cleanup_file_resources(file_obj, temp_file_path)
            return LLMResponse(
                success=False,
                message=f"API connection error: {e!s}",
                extra={"error": str(e), "model": self.name},
            )
        except APIStatusError as e:
            logger.error(f"API status error: {e}")
            self._cleanup_file_resources(file_obj, temp_file_path)
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
            # 处理异常情况。
            self._cleanup_file_resources(file_obj, temp_file_path)
            return LLMResponse(
                success=False,
                message=f"Unexpected error: {e!s}",
                extra={"error": str(e), "model": self.name},
            )
