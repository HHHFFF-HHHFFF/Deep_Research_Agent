import base64
import os
from typing import Any, overload

from pydantic import BaseModel

from src.message.types import (
    AssistantMessage,
    ContentPartImage,
    ContentPartText,
    HumanMessage,
    Message,
    SystemMessage,
    ToolCall,
)
from src.tool.types import Tool
from src.utils import assemble_project_path

try:
    from anthropic import transform_schema
except ImportError:
    transform_schema = None


class AnthropicChatSerializer:
    """定义 `AnthropicChatSerializer`，封装相关数据与行为。"""

    @staticmethod
    def _serialize_content_part_text(part: ContentPartText) -> dict[str, Any]:
        return {"type": "text", "text": part.text}

    @staticmethod
    def _normalize_media_type(media_type: str) -> str:
        """实现 `_normalize_media_type` 的业务逻辑。"""
        media_type = media_type.lower().strip()

        # 说明相关实现细节。
        if media_type in ["image/jpeg", "image/jpg"]:
            return "image/jpeg"
        elif media_type == "image/png":
            return "image/png"
        elif media_type == "image/gif":
            return "image/gif"
        elif media_type == "image/webp":
            return "image/webp"
        else:
            # 说明相关实现细节。
            return "image/jpeg"

    @staticmethod
    def _serialize_content_part_image(part: ContentPartImage) -> dict[str, Any]:
        """实现 `_serialize_content_part_image` 的业务逻辑。"""
        image_url = part.image_url.url

        # 说明相关实现细节。
        if image_url.startswith("data:"):
            # 说明相关实现细节。
            # 转换并规范化数据。
            header, data = image_url.split(",", 1)
            media_type = "image/jpeg"  # 说明相关实现细节。
            if "image/" in header:
                extracted_type = header.split("image/")[1].split(";")[0]
                media_type = AnthropicChatSerializer._normalize_media_type(
                    f"image/{extracted_type}"
                )
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data,
                },
            }
        elif image_url.startswith("file://"):
            # 加载所需数据。
            file_path = image_url[7:]
            if not os.path.isabs(file_path):
                file_path = assemble_project_path(file_path)
            if os.path.exists(file_path):
                # 加载所需数据。
                with open(file_path, "rb") as f:
                    image_data = f.read()
                base64_data = base64.b64encode(image_data).decode("utf-8")
                # 处理文件与路径。
                import mimetypes

                guessed_type, _ = mimetypes.guess_type(file_path)
                if not guessed_type or not guessed_type.startswith("image/"):
                    media_type = "image/jpeg"  # 说明相关实现细节。
                else:
                    media_type = AnthropicChatSerializer._normalize_media_type(
                        guessed_type
                    )
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_data,
                    },
                }
        elif os.path.exists(image_url):
            # 处理文件与路径。
            with open(image_url, "rb") as f:
                image_data = f.read()
            base64_data = base64.b64encode(image_data).decode("utf-8")
            import mimetypes

            guessed_type, _ = mimetypes.guess_type(image_url)
            if not guessed_type or not guessed_type.startswith("image/"):
                media_type = "image/jpeg"  # 说明相关实现细节。
            else:
                media_type = AnthropicChatSerializer._normalize_media_type(guessed_type)
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_data,
                },
            }
        elif os.path.exists(assemble_project_path(image_url)):
            # 处理文件与路径。
            file_path = assemble_project_path(image_url)
            with open(file_path, "rb") as f:
                image_data = f.read()
            base64_data = base64.b64encode(image_data).decode("utf-8")
            import mimetypes

            guessed_type, _ = mimetypes.guess_type(file_path)
            if not guessed_type or not guessed_type.startswith("image/"):
                media_type = "image/jpeg"  # 说明相关实现细节。
            else:
                media_type = AnthropicChatSerializer._normalize_media_type(guessed_type)
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_data,
                },
            }
        else:
            # 处理异常情况。
            # 说明相关实现细节。
            raise ValueError(
                f"Anthropic API only supports base64-encoded images or local files. Got: {image_url}"
            )

    @staticmethod
    def _serialize_user_content(
        content: str | list[ContentPartText | ContentPartImage],
    ) -> list[dict[str, Any]]:
        """实现 `_serialize_user_content` 的业务逻辑。"""
        serialized_parts: list[dict[str, Any]] = []

        if isinstance(content, str):
            # 转换并规范化数据。
            serialized_parts.append({"type": "text", "text": content})
        else:
            # 说明相关实现细节。
            for part in content:
                if part.type == "text":
                    serialized_parts.append(
                        AnthropicChatSerializer._serialize_content_part_text(part)
                    )
                elif part.type == "image_url":
                    serialized_parts.append(
                        AnthropicChatSerializer._serialize_content_part_image(part)
                    )

        return serialized_parts

    @staticmethod
    def _serialize_assistant_content(
        content: str | list[ContentPartText] | None,
    ) -> list[dict[str, Any]]:
        """实现 `_serialize_assistant_content` 的业务逻辑。"""
        serialized_parts: list[dict[str, Any]] = []

        if content is None:
            return serialized_parts

        if isinstance(content, str):
            # 转换并规范化数据。
            serialized_parts.append({"type": "text", "text": content})
        else:
            # 说明相关实现细节。
            for part in content:
                if part.type == "text":
                    serialized_parts.append(
                        AnthropicChatSerializer._serialize_content_part_text(part)
                    )

        return serialized_parts

    @staticmethod
    def _serialize_tool_call(tool_call: ToolCall) -> dict[str, Any]:
        """实现 `_serialize_tool_call` 的业务逻辑。"""
        import json

        try:
            input_data = (
                json.loads(tool_call.function.arguments)
                if isinstance(tool_call.function.arguments, str)
                else tool_call.function.arguments
            )
        except json.JSONDecodeError:
            input_data = {}

        return {
            "type": "tool_use",
            "id": tool_call.id,
            "name": tool_call.function.name,
            "input": input_data,
        }

    @overload
    @staticmethod
    def serialize(message: HumanMessage) -> dict[str, Any]: ...

    @overload
    @staticmethod
    def serialize(message: SystemMessage) -> dict[str, Any]: ...

    @overload
    @staticmethod
    def serialize(message: AssistantMessage) -> dict[str, Any]: ...

    @staticmethod
    def serialize(message: Message) -> dict[str, Any]:
        """实现 `serialize` 的业务逻辑。"""
        if isinstance(message, HumanMessage):
            content = AnthropicChatSerializer._serialize_user_content(message.content)
            result: dict[str, Any] = {
                "role": "user",
                "content": content,
            }
            return result

        elif isinstance(message, SystemMessage):
            # 说明相关实现细节。
            # 组装并返回结果。
            content = message.content
            if isinstance(content, str):
                return {"role": "system", "content": content}
            elif isinstance(content, list):
                # 说明相关实现细节。
                text_parts = []
                for part in content:
                    if isinstance(part, ContentPartText):
                        text_parts.append(part.text)
                return {"role": "system", "content": " ".join(text_parts)}
            else:
                return {"role": "system", "content": str(content)}

        elif isinstance(message, AssistantMessage):
            content_parts = AnthropicChatSerializer._serialize_assistant_content(
                message.content
            )
            result: dict[str, Any] = {"role": "assistant"}

            # 处理工具调用。
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    content_parts.append(
                        AnthropicChatSerializer._serialize_tool_call(tool_call)
                    )

            # 说明相关实现细节。
            result["content"] = content_parts

            return result

        else:
            raise TypeError(f"Unknown message type: {type(message)}")

    @staticmethod
    def serialize_messages(
        messages: list[Message],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """序列化与 `serialize_messages` 对应的数据或状态。"""
        system_message = None
        anthropic_messages: list[dict[str, Any]] = []

        for message in messages:
            if isinstance(message, SystemMessage):
                # 说明相关实现细节。
                serialized = AnthropicChatSerializer.serialize(message)
                if serialized.get("content"):
                    if system_message is None:
                        system_message = serialized["content"]
                    else:
                        system_message += "\n" + serialized["content"]
            else:
                # 转换并规范化数据。
                anthropic_messages.append(AnthropicChatSerializer.serialize(message))

        return system_message, anthropic_messages

    @staticmethod
    def serialize_tools(tools: list["Tool"]) -> list[dict[str, Any]]:
        """序列化与 `serialize_tools` 对应的数据或状态。"""
        formatted_tools = []
        for tool in tools:
            if isinstance(tool, Tool):
                # 转换并规范化数据。
                function_call = tool.function_calling
                function_def = function_call.get("function", {})

                # 处理输入参数。
                anthropic_tool = {
                    "name": function_def.get("name", tool.name),
                    "description": function_def.get("description", tool.description),
                    "input_schema": function_def.get("parameters", {}),
                }
                formatted_tools.append(anthropic_tool)

        return formatted_tools

    @staticmethod
    def serialize_response_format(
        response_format: type[BaseModel] | BaseModel | dict[str, Any],
    ) -> dict[str, Any]:
        """序列化与 `serialize_response_format` 对应的数据或状态。"""
        if isinstance(response_format, dict):
            # 加载所需数据。
            if "type" in response_format and "schema" in response_format:
                return response_format
            elif "type" in response_format and "json_schema" in response_format:
                json_schema_obj = response_format["json_schema"]
                schema = json_schema_obj.get("schema", {})
                return {"type": "json_schema", "schema": schema}
            else:
                return {"type": "json_schema", "schema": response_format}

        model_class = (
            response_format
            if isinstance(response_format, type)
            else type(response_format)
        )
        if not issubclass(model_class, BaseModel):
            raise TypeError(
                f"Unsupported response_format type: {type(response_format)}"
            )

        # 说明相关实现细节。
        # 说明相关实现细节。
        if transform_schema is not None:
            schema = transform_schema(model_class)
        else:
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
                return {"type": "object", "additionalProperties": True}

            # 说明相关实现细节。
            for k in ["anyOf", "oneOf", "allOf"]:
                if k in obj:
                    items = obj[k]
                    non_null = [
                        i
                        for i in items
                        if isinstance(i, dict) and i.get("type") != "null"
                    ]
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
                            "additionalProperties": True,
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
                # 更新相关状态。
                if not new_props and obj.get("additionalProperties") is True:
                    additional_props = True
                else:
                    additional_props = False

                result = {
                    "type": "object",
                    "properties": new_props,
                    "required": new_required,
                    "additionalProperties": additional_props,
                }
                # 说明相关实现细节。
                if "description" in obj:
                    result["description"] = obj["description"]
                if "title" in obj:
                    result["title"] = obj["title"]
                return result

            # 说明相关实现细节。
            if obj.get("type") == "array":
                result = {"type": "array", "items": transform(obj.get("items", {}))}
                # 说明相关实现细节。
                if "description" in obj:
                    result["description"] = obj["description"]
                if "title" in obj:
                    result["title"] = obj["title"]
                return result

            # 说明相关实现细节。
            return obj

        return {"type": "json_schema", "schema": transform(schema)}
