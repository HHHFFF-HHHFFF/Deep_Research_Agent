import pytest

from src.message.types import (
    AssistantMessage,
    AudioURL,
    ContentPartAudio,
    HumanMessage,
)
from src.model.openai.serializer import OpenAIChatSerializer, OpenAIResponseSerializer
from src.tool.types import Tool


def test_chat_serializer_converts_text_message() -> None:
    serialized = OpenAIChatSerializer.serialize(HumanMessage(content="研究主题"))

    assert serialized == {"role": "user", "content": "研究主题"}


def test_chat_serializer_rejects_unsupported_content() -> None:
    message = HumanMessage(
        content=[ContentPartAudio(audio_url=AudioURL(url="audio.mp3"))]
    )

    with pytest.raises(TypeError, match="暂不支持内容类型"):
        OpenAIChatSerializer.serialize(message)


def test_chat_serializer_skips_tools_without_protocol() -> None:
    enabled = Tool(
        name="search",
        description="搜索",
        function_calling={"type": "function", "function": {"name": "search"}},
    )
    disabled = Tool(name="internal", description="内部工具")

    assert OpenAIChatSerializer.serialize_tools([enabled, disabled]) == [
        {"type": "function", "function": {"name": "search"}}
    ]


def test_response_serializer_uses_distinct_message_payloads() -> None:
    serialized = OpenAIResponseSerializer.serialize(
        AssistantMessage(content="研究完成")
    )

    assert serialized == {
        "role": "assistant",
        "content": [{"type": "input_text", "text": "研究完成"}],
    }
