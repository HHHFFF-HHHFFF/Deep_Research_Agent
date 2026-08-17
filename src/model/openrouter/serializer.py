from typing import overload, Any, Union, List, Dict, Type
import base64
import os

try:
    from openai.types.chat import (
        ChatCompletionAssistantMessageParam,
        ChatCompletionContentPartImageParam,
        ChatCompletionContentPartRefusalParam,
        ChatCompletionContentPartTextParam,
        ChatCompletionMessageFunctionToolCallParam,
        ChatCompletionMessageParam,
        ChatCompletionSystemMessageParam,
        ChatCompletionUserMessageParam,
    )
    from openai.types.chat.chat_completion_content_part_image_param import ImageURL as OpenAIImageURL
    from openai.types.chat.chat_completion_message_function_tool_call_param import Function as OpenAIFunction
except ImportError:
    # 执行回退或重试逻辑。
    ChatCompletionAssistantMessageParam = dict
    ChatCompletionContentPartImageParam = dict
    ChatCompletionContentPartRefusalParam = dict
    ChatCompletionContentPartTextParam = dict
    ChatCompletionMessageFunctionToolCallParam = dict
    ChatCompletionMessageParam = dict
    ChatCompletionSystemMessageParam = dict
    ChatCompletionUserMessageParam = dict
    OpenAIImageURL = dict
    OpenAIFunction = dict

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel

from src.message.types import (
    AssistantMessage,
    ContentPartAudio,
    ContentPartImage,
    ContentPartPdf,
    ContentPartRefusal,
    ContentPartText,
    ContentPartVideo,
    HumanMessage,
    Message,
    SystemMessage,
    ToolCall,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.tool.types import Tool

from src.utils import assemble_project_path, encode_file_base64


class OpenRouterChatSerializer:
    """定义 `OpenRouterChatSerializer`，封装相关数据与行为。"""

    @staticmethod
    def _serialize_content_part_text(part: ContentPartText) -> ChatCompletionContentPartTextParam:
        return ChatCompletionContentPartTextParam(text=part.text, type='text')

    @staticmethod
    def _serialize_content_part_image(part: ContentPartImage) -> ChatCompletionContentPartImageParam:
        return ChatCompletionContentPartImageParam(
            image_url=OpenAIImageURL(url=part.image_url.url, detail=part.image_url.detail),
            type='image_url',
        )

    @staticmethod
    def _serialize_content_part_refusal(part: ContentPartRefusal) -> ChatCompletionContentPartRefusalParam:
        return ChatCompletionContentPartRefusalParam(refusal=part.refusal, type='refusal')

    @staticmethod
    def _serialize_content_part_audio(part: ContentPartAudio) -> dict[str, Any]:
        """实现 `_serialize_content_part_audio` 的业务逻辑。"""
        audio_url = part.audio_url.url
        audio_format = part.audio_url.media_type.split('/')[-1]  # 转换并规范化数据。

        # 说明相关实现细节。
        if audio_url.startswith("data:"):
            # 说明相关实现细节。
            # 转换并规范化数据。
            if "," in audio_url:
                header, data = audio_url.split(",", 1)
                # 转换并规范化数据。
                if "audio/" in header:
                    format_part = header.split("audio/")[1].split(";")[0]
                    if format_part:
                        audio_format = format_part
                return {
                    "type": "input_audio",
                    "input_audio": {
                        "data": data,
                        "format": audio_format,
                    }
                }
        elif audio_url.startswith("file://"):
            # 加载所需数据。
            file_path = audio_url[7:]
            if not os.path.isabs(file_path):
                file_path = assemble_project_path(file_path)
            if os.path.exists(file_path):
                # 加载所需数据。
                with open(file_path, "rb") as f:
                    audio_data = f.read()
                base64_data = base64.b64encode(audio_data).decode("utf-8")
                return {
                    "type": "input_audio",
                    "input_audio": {
                        "data": base64_data,
                        "format": audio_format,
                    }
                }
        elif os.path.exists(audio_url):
            # 处理文件与路径。
            with open(audio_url, "rb") as f:
                audio_data = f.read()
            base64_data = base64.b64encode(audio_data).decode("utf-8")
            return {
                "type": "input_audio",
                "input_audio": {
                    "data": base64_data,
                    "format": audio_format,
                }
            }
        elif os.path.exists(assemble_project_path(audio_url)):
            # 处理文件与路径。
            file_path = assemble_project_path(audio_url)
            with open(file_path, "rb") as f:
                audio_data = f.read()
            base64_data = base64.b64encode(audio_data).decode("utf-8")
            return {
                "type": "input_audio",
                "input_audio": {
                    "data": base64_data,
                    "format": audio_format,
                }
            }
        else:
            # 说明相关实现细节。
            # 说明相关实现细节。
            # 说明相关实现细节。
            return {
                "type": "input_audio",
                "input_audio": {
                    "url": audio_url,
                    "format": audio_format,
                }
            }

    @staticmethod
    def _serialize_content_part_video(part: ContentPartVideo) -> dict[str, Any]:
        """实现 `_serialize_content_part_video` 的业务逻辑。"""
        return {
            "type": "video_url",
            "video_url": {
                "url": part.video_url.url,
            }
        }

    @staticmethod
    def _serialize_content_part_pdf(part: ContentPartPdf) -> dict[str, Any]:
        """实现 `_serialize_content_part_pdf` 的业务逻辑。"""
        pdf_url = part.pdf_url.url

        # 说明相关实现细节。
        if pdf_url.startswith("data:"):
            # 加载所需数据。
            filename = "document.pdf"
            # 转换并规范化数据。
            if "file://" in pdf_url or "filename=" in pdf_url:
                # 转换并规范化数据。
                pass
            return {
                "type": "file",
                "file": {
                    "filename": filename,
                    "file_data": pdf_url,
                }
            }
        elif pdf_url.startswith("file://"):
            # 加载所需数据。
            file_path = pdf_url[7:]
            if not os.path.isabs(file_path):
                file_path = assemble_project_path(file_path)
            if os.path.exists(file_path):
                # 加载所需数据。
                with open(file_path, "rb") as f:
                    pdf_data = f.read()
                base64_data = base64.b64encode(pdf_data).decode("utf-8")
                filename = os.path.basename(file_path)
                data_url = f"data:application/pdf;base64,{base64_data}"
                return {
                    "type": "file",
                    "file": {
                        "filename": filename,
                        "file_data": data_url,
                    }
                }
        elif os.path.exists(pdf_url):
            # 处理文件与路径。
            with open(pdf_url, "rb") as f:
                pdf_data = f.read()
            base64_data = base64.b64encode(pdf_data).decode("utf-8")
            filename = os.path.basename(pdf_url)
            data_url = f"data:application/pdf;base64,{base64_data}"
            return {
                "type": "file",
                "file": {
                    "filename": filename,
                    "file_data": data_url,
                }
            }
        elif os.path.exists(assemble_project_path(pdf_url)):
            # 处理文件与路径。
            file_path = assemble_project_path(pdf_url)
            with open(file_path, "rb") as f:
                pdf_data = f.read()
            base64_data = base64.b64encode(pdf_data).decode("utf-8")
            filename = os.path.basename(file_path)
            data_url = f"data:application/pdf;base64,{base64_data}"
            return {
                "type": "file",
                "file": {
                    "filename": filename,
                    "file_data": data_url,
                }
            }
        else:
            # 加载所需数据。
            # 加载所需数据。
            return {
                "type": "file",
                "file": {
                    "filename": "document.pdf",
                    "file_data": pdf_url if pdf_url.startswith("data:") else f"data:application/pdf;base64,{pdf_url}",
                }
            }

    @staticmethod
    def _serialize_user_content(
        content: str | list[ContentPartText | ContentPartImage | ContentPartAudio | ContentPartVideo | ContentPartPdf],
    ) -> str | list[Union[ChatCompletionContentPartTextParam, ChatCompletionContentPartImageParam, dict[str, Any]]]:
        """实现 `_serialize_user_content` 的业务逻辑。"""
        if isinstance(content, str):
            return content
        serialized_parts: list[Union[ChatCompletionContentPartTextParam, ChatCompletionContentPartImageParam, dict[str, Any]]] = []
        for part in content:
            if part.type == 'text':
                serialized_parts.append(OpenRouterChatSerializer._serialize_content_part_text(part))
            elif part.type == 'image_url':
                serialized_parts.append(OpenRouterChatSerializer._serialize_content_part_image(part))
            elif part.type == 'audio_url':
                serialized_parts.append(OpenRouterChatSerializer._serialize_content_part_audio(part))
            elif part.type == 'video_url':
                serialized_parts.append(OpenRouterChatSerializer._serialize_content_part_video(part))
            elif part.type == 'pdf_url':
                serialized_parts.append(OpenRouterChatSerializer._serialize_content_part_pdf(part))
        return serialized_parts

    @staticmethod
    def _serialize_system_content(
        content: str | list[ContentPartText],
    ) -> str | list[ChatCompletionContentPartTextParam]:
        """实现 `_serialize_system_content` 的业务逻辑。"""
        if isinstance(content, str):
            return content
        serialized_parts: list[ChatCompletionContentPartTextParam] = []
        for part in content:
            if part.type == 'text':
                serialized_parts.append(OpenRouterChatSerializer._serialize_content_part_text(part))
        return serialized_parts

    @staticmethod
    def _serialize_assistant_content(
        content: str | list[ContentPartText | ContentPartRefusal] | None,
    ) -> str | list[ChatCompletionContentPartTextParam | ChatCompletionContentPartRefusalParam] | None:
        """实现 `_serialize_assistant_content` 的业务逻辑。"""
        if content is None:
            return None
        if isinstance(content, str):
            return content
        serialized_parts: list[ChatCompletionContentPartTextParam | ChatCompletionContentPartRefusalParam] = []
        for part in content:
            if part.type == 'text':
                serialized_parts.append(OpenRouterChatSerializer._serialize_content_part_text(part))
            elif part.type == 'refusal':
                serialized_parts.append(OpenRouterChatSerializer._serialize_content_part_refusal(part))
        return serialized_parts

    @staticmethod
    def _serialize_tool_call(tool_call: ToolCall) -> ChatCompletionMessageFunctionToolCallParam:
        return ChatCompletionMessageFunctionToolCallParam(
            id=tool_call.id,
            function=OpenAIFunction(name=tool_call.function.name, arguments=tool_call.function.arguments),
            type='function',
        )

    # 加载所需数据。

    @overload
    @staticmethod
    def serialize_message(message: HumanMessage) -> ChatCompletionUserMessageParam: ...

    @overload
    @staticmethod
    def serialize_message(message: SystemMessage) -> ChatCompletionSystemMessageParam: ...

    @overload
    @staticmethod
    def serialize_message(message: AssistantMessage) -> ChatCompletionAssistantMessageParam: ...

    @staticmethod
    def serialize_message(message: Message) -> ChatCompletionMessageParam:
        """序列化与 `serialize_message` 对应的数据或状态。"""
        if isinstance(message, HumanMessage):
            user_result: ChatCompletionUserMessageParam = {
                'role': 'user',
                'content': OpenRouterChatSerializer._serialize_user_content(message.content),
            }
            if message.name is not None:
                user_result['name'] = message.name
            return user_result

        elif isinstance(message, SystemMessage):
            system_result: ChatCompletionSystemMessageParam = {
                'role': 'system',
                'content': [
                    {
                        'type': 'text',
                        'text': OpenRouterChatSerializer._serialize_system_content(message.content),
                        "cache_control": {
                            "type": "ephemeral"
                        }
                    }
                ],
            }
            if message.name is not None:
                system_result['name'] = message.name
            return system_result

        elif isinstance(message, AssistantMessage):
            # 说明相关实现细节。
            content = None
            if message.content is not None:
                content = OpenRouterChatSerializer._serialize_assistant_content(message.content)
            assistant_result: ChatCompletionAssistantMessageParam = {'role': 'assistant'}
            # 说明相关实现细节。
            if content is not None:
                assistant_result['content'] = content
            if message.name is not None:
                assistant_result['name'] = message.name
            if message.refusal is not None:
                assistant_result['refusal'] = message.refusal
            if message.tool_calls:
                assistant_result['tool_calls'] = [OpenRouterChatSerializer._serialize_tool_call(tc) for tc in message.tool_calls]
            return assistant_result

        else:
            raise ValueError(f'Unknown message type: {type(message)}')

    @staticmethod
    def serialize_messages(messages: list[Message]) -> list[ChatCompletionMessageParam]:
        return [OpenRouterChatSerializer.serialize_message(m) for m in messages]

    @staticmethod
    def serialize_tool(tool: "Tool") -> Dict[str, Any]:
        """序列化与 `serialize_tool` 对应的数据或状态。"""
        return tool.function_calling

    @staticmethod
    def serialize_tools(tools: List["Tool"]) -> List[Dict[str, Any]]:
        """序列化与 `serialize_tools` 对应的数据或状态。"""
        return [OpenRouterChatSerializer.serialize_tool(tool) for tool in tools]

    @staticmethod
    def serialize_response_format(
        response_format: Union[Type[BaseModel], BaseModel],
        model_name: str = "openrouter/gemini-3-flash-preview",
    ) -> Dict[str, Any]:
        """序列化与 `serialize_response_format` 对应的数据或状态。"""
        model_class = response_format if isinstance(response_format, type) else type(response_format)
        schema = model_class.model_json_schema()
        defs = schema.pop("$defs", {})  # 移除相关数据或组件。

        def transform(obj: Any) -> Any:
            if not isinstance(obj, dict):
                return obj

            # 说明相关实现细节。
            if "$ref" in obj:
                ref_path = obj["$ref"]
                if ref_path.startswith("#/$defs/"):
                    def_name = ref_path.split("/")[-1]
                    if def_name in defs:
                        return transform(defs[def_name])
                return {"type": "object", "additionalProperties": False}

            # 说明相关实现细节。
            for k in ["anyOf", "oneOf", "allOf"]:
                if k in obj:
                    items = obj[k]
                    non_null = [i for i in items if isinstance(i, dict) and i.get("type") != "null"]
                    if len(non_null) == 1:
                        # 说明相关实现细节。
                        result = transform(non_null[0])
                        if isinstance(result, dict):
                            if "description" in obj and "description" not in result:
                                result["description"] = obj["description"]
                            if "title" in obj and "title" not in result:
                                result["title"] = obj["title"]
                        return result
                    else:
                        return {
                            "type": "object",
                            "description": obj.get("description", "Simplified Object"),
                            "additionalProperties": False
                        }

            # 说明相关实现细节。
            if obj.get("type") == "object" or "properties" in obj:
                props = obj.get("properties", {})
                required = obj.get("required", [])
                new_props = {}
                new_required = []

                for k, v in props.items():
                    new_props[k] = transform(v)
                    if k in required:
                        new_required.append(k)

                # 说明相关实现细节。
                # 说明相关实现细节。
                # 说明相关实现细节。
                ap = obj.get("additionalProperties", None)
                if isinstance(ap, dict):
                    # 说明相关实现细节。
                    transformed_ap = transform(ap)
                    additional_props = transformed_ap
                else:
                    # 说明相关实现细节。
                    # 说明相关实现细节。
                    # 说明相关实现细节。
                    if not new_props and ap is True:
                        additional_props = True
                    else:
                        additional_props = False

                # 处理模型调用。
                if model_name in ["openai/gpt-4o", "openai/gpt-4.1", "openai/gpt-5", "openai/gpt-5.1", "openai/gpt-5.2", "openai/o3"]:
                    additional_props = False

                result = {
                    "type": "object",
                    "properties": new_props,
                    "required": new_required,
                    "additionalProperties": additional_props
                }
                # 说明相关实现细节。
                if "description" in obj:
                    result["description"] = obj["description"]
                if "title" in obj:
                    result["title"] = obj["title"]
                return result

            # 说明相关实现细节。
            if obj.get("type") == "array":
                result = {
                    "type": "array",
                    "items": transform(obj.get("items", {}))
                }
                # 说明相关实现细节。
                if "description" in obj:
                    result["description"] = obj["description"]
                if "title" in obj:
                    result["title"] = obj["title"]
                return result

            # 说明相关实现细节。
            return obj

        return {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "strict": True,
                "schema": transform(schema),
            },
        }
