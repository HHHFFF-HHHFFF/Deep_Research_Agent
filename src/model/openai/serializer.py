from typing import TYPE_CHECKING, Any, overload

if TYPE_CHECKING:
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
    from openai.types.chat.chat_completion_content_part_image_param import (
        ImageURL as OpenAIImageURL,
    )
    from openai.types.chat.chat_completion_message_function_tool_call_param import (
        Function as OpenAIFunction,
    )
else:
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
        from openai.types.chat.chat_completion_content_part_image_param import (
            ImageURL as OpenAIImageURL,
        )
        from openai.types.chat.chat_completion_message_function_tool_call_param import (
            Function as OpenAIFunction,
        )
    except ImportError:
        # 仅在未安装 OpenAI SDK 时提供可导入的字典回退。
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

if TYPE_CHECKING:
    from src.tool.types import Tool

UserContent = (
    str
    | list[
        ContentPartText
        | ContentPartImage
        | ContentPartAudio
        | ContentPartVideo
        | ContentPartPdf
    ]
)
AssistantContent = (
    str
    | list[
        ContentPartText
        | ContentPartImage
        | ContentPartAudio
        | ContentPartVideo
        | ContentPartPdf
        | ContentPartRefusal
    ]
)


class OpenAIChatSerializer:
    """定义 `OpenAIChatSerializer`，封装相关数据与行为。"""

    @staticmethod
    def _serialize_content_part_text(
        part: ContentPartText,
    ) -> ChatCompletionContentPartTextParam:
        return ChatCompletionContentPartTextParam(text=part.text, type="text")

    @staticmethod
    def _serialize_content_part_image(
        part: ContentPartImage,
    ) -> ChatCompletionContentPartImageParam:
        return ChatCompletionContentPartImageParam(
            image_url=OpenAIImageURL(
                url=part.image_url.url, detail=part.image_url.detail
            ),
            type="image_url",
        )

    @staticmethod
    def _serialize_content_part_refusal(
        part: ContentPartRefusal,
    ) -> ChatCompletionContentPartRefusalParam:
        return ChatCompletionContentPartRefusalParam(
            refusal=part.refusal, type="refusal"
        )

    @staticmethod
    def _serialize_user_content(
        content: UserContent,
    ) -> (
        str
        | list[ChatCompletionContentPartTextParam | ChatCompletionContentPartImageParam]
    ):
        """实现 `_serialize_user_content` 的业务逻辑。"""
        if isinstance(content, str):
            return content
        serialized_parts: list[
            ChatCompletionContentPartTextParam | ChatCompletionContentPartImageParam
        ] = []
        for part in content:
            if isinstance(part, ContentPartText):
                serialized_parts.append(
                    OpenAIChatSerializer._serialize_content_part_text(part)
                )
            elif isinstance(part, ContentPartImage):
                serialized_parts.append(
                    OpenAIChatSerializer._serialize_content_part_image(part)
                )
            else:
                raise TypeError(f"OpenAI Chat 暂不支持内容类型：{part.type}")
        return serialized_parts

    @staticmethod
    def _serialize_system_content(
        content: UserContent,
    ) -> str | list[ChatCompletionContentPartTextParam]:
        """实现 `_serialize_system_content` 的业务逻辑。"""
        if isinstance(content, str):
            return content
        serialized_parts: list[ChatCompletionContentPartTextParam] = []
        for part in content:
            if isinstance(part, ContentPartText):
                serialized_parts.append(
                    OpenAIChatSerializer._serialize_content_part_text(part)
                )
            else:
                raise TypeError(f"系统消息只支持文本内容，收到：{part.type}")
        return serialized_parts

    @staticmethod
    def _serialize_assistant_content(
        content: AssistantContent | None,
    ) -> (
        str
        | list[
            ChatCompletionContentPartTextParam | ChatCompletionContentPartRefusalParam
        ]
        | None
    ):
        """实现 `_serialize_assistant_content` 的业务逻辑。"""
        if content is None:
            return None
        if isinstance(content, str):
            return content
        serialized_parts: list[
            ChatCompletionContentPartTextParam | ChatCompletionContentPartRefusalParam
        ] = []
        for part in content:
            if isinstance(part, ContentPartText):
                serialized_parts.append(
                    OpenAIChatSerializer._serialize_content_part_text(part)
                )
            elif isinstance(part, ContentPartRefusal):
                serialized_parts.append(
                    OpenAIChatSerializer._serialize_content_part_refusal(part)
                )
            else:
                raise TypeError(f"助手消息暂不支持内容类型：{part.type}")
        return serialized_parts

    @staticmethod
    def _serialize_tool_call(
        tool_call: ToolCall,
    ) -> ChatCompletionMessageFunctionToolCallParam:
        return ChatCompletionMessageFunctionToolCallParam(
            id=tool_call.id,
            function=OpenAIFunction(
                name=tool_call.function.name, arguments=tool_call.function.arguments
            ),
            type="function",
        )

    # 加载所需数据。

    @overload
    @staticmethod
    def serialize(message: HumanMessage) -> ChatCompletionUserMessageParam: ...

    @overload
    @staticmethod
    def serialize(message: SystemMessage) -> ChatCompletionSystemMessageParam: ...

    @overload
    @staticmethod
    def serialize(message: AssistantMessage) -> ChatCompletionAssistantMessageParam: ...

    @overload
    @staticmethod
    def serialize(message: Message) -> ChatCompletionMessageParam: ...

    @staticmethod
    def serialize(message: Message) -> ChatCompletionMessageParam:
        """实现 `serialize` 的业务逻辑。"""
        if isinstance(message, HumanMessage):
            user_result: ChatCompletionUserMessageParam = {
                "role": "user",
                "content": OpenAIChatSerializer._serialize_user_content(
                    message.content
                ),
            }
            if message.name is not None:
                user_result["name"] = message.name
            return user_result

        elif isinstance(message, SystemMessage):
            system_result: ChatCompletionSystemMessageParam = {
                "role": "system",
                "content": OpenAIChatSerializer._serialize_system_content(
                    message.content
                ),
            }
            if message.name is not None:
                system_result["name"] = message.name
            return system_result

        elif isinstance(message, AssistantMessage):
            # 说明相关实现细节。
            content = None
            if message.content is not None:
                content = OpenAIChatSerializer._serialize_assistant_content(
                    message.content
                )
            assistant_result: ChatCompletionAssistantMessageParam = {
                "role": "assistant"
            }
            # 说明相关实现细节。
            if content is not None:
                assistant_result["content"] = content
            if message.name is not None:
                assistant_result["name"] = message.name
            if message.refusal is not None:
                assistant_result["refusal"] = message.refusal
            if message.tool_calls:
                assistant_result["tool_calls"] = [
                    OpenAIChatSerializer._serialize_tool_call(tc)
                    for tc in message.tool_calls
                ]
            return assistant_result

        else:
            raise TypeError(f"Unknown message type: {type(message)}")

    @staticmethod
    def serialize_messages(messages: list[Message]) -> list[ChatCompletionMessageParam]:
        return [OpenAIChatSerializer.serialize(m) for m in messages]

    @staticmethod
    def serialize_tools(tools: list["Tool"]) -> list[dict[str, Any]]:
        """序列化与 `serialize_tools` 对应的数据或状态。"""
        return [
            tool.function_calling for tool in tools if tool.function_calling is not None
        ]

    @staticmethod
    def serialize_response_format(
        response_format: type[BaseModel] | BaseModel,
    ) -> dict[str, Any]:
        """序列化与 `serialize_response_format` 对应的数据或状态。"""
        model_class = (
            response_format
            if isinstance(response_format, type)
            else type(response_format)
        )
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

        return {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "strict": True,
                "schema": transform(schema),
            },
        }


class OpenAIResponseSerializer:
    """定义 `OpenAIResponseSerializer`，封装相关数据与行为。"""

    @staticmethod
    def _serialize_content_part_text(part: ContentPartText) -> dict[str, Any]:
        """实现 `_serialize_content_part_text` 的业务逻辑。"""
        return {
            "type": "input_text",
            "text": part.text,
        }

    @staticmethod
    def _serialize_content_part_image(part: ContentPartImage) -> dict[str, Any]:
        """实现 `_serialize_content_part_image` 的业务逻辑。"""
        return {
            "type": "input_image",
            "image_url": part.image_url.url,
        }

    @staticmethod
    def _serialize_content(
        content: UserContent | AssistantContent,
    ) -> list[dict[str, Any]]:
        """实现 `_serialize_content` 的业务逻辑。"""
        if isinstance(content, str):
            return [{"type": "input_text", "text": content}]

        serialized_parts: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, ContentPartText):
                serialized_parts.append(
                    OpenAIResponseSerializer._serialize_content_part_text(part)
                )
            elif isinstance(part, ContentPartImage):
                serialized_parts.append(
                    OpenAIResponseSerializer._serialize_content_part_image(part)
                )
            else:
                raise TypeError(f"OpenAI Responses 暂不支持内容类型：{part.type}")

        return serialized_parts

    @staticmethod
    def serialize(message: Message) -> dict[str, Any]:
        """实现 `serialize` 的业务逻辑。"""
        if isinstance(message, HumanMessage):
            user_result: dict[str, Any] = {
                "role": "user",
                "content": OpenAIResponseSerializer._serialize_content(message.content),
            }
            if message.name is not None:
                user_result["name"] = message.name
            return user_result

        elif isinstance(message, SystemMessage):
            # 说明相关实现细节。
            # 组装并返回结果。
            system_result: dict[str, Any] = {
                "role": "system",
                "content": OpenAIResponseSerializer._serialize_content(message.content),
            }
            if message.name is not None:
                system_result["name"] = message.name
            return system_result

        elif isinstance(message, AssistantMessage):
            # 处理输入参数。
            assistant_result: dict[str, Any] = {
                "role": "assistant",
            }
            if message.content is not None:
                assistant_result["content"] = (
                    OpenAIResponseSerializer._serialize_content(message.content)
                )
            if message.name is not None:
                assistant_result["name"] = message.name
            return assistant_result

        else:
            raise TypeError(f"Unknown message type: {type(message)}")

    @staticmethod
    def serialize_messages(messages: list[Message]) -> list[dict[str, Any]]:
        """序列化与 `serialize_messages` 对应的数据或状态。"""
        return [OpenAIResponseSerializer.serialize(m) for m in messages]
