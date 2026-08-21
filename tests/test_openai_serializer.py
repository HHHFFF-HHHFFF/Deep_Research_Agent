import asyncio

import pytest

from src.message.types import (
    AssistantMessage,
    AudioURL,
    ContentPartAudio,
    HumanMessage,
)
from src.model.openai.embedding import EmbeddingOpenAI
from src.model.openai.serializer import OpenAIChatSerializer, OpenAIResponseSerializer
from src.model.openai_compatible import ChatOpenAICompatible
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


def test_qwen_compatible_adapter_uses_max_tokens_parameter() -> None:
    model = ChatOpenAICompatible(
        model="qwen3-max",
        provider_name="qwen",
        max_completion_tokens=2048,
    )

    payload = asyncio.run(model._build_params([HumanMessage(content="研究主题")]))

    assert payload["messages"] == [{"role": "user", "content": "研究主题"}]
    assert payload["params"]["max_tokens"] == 2048
    assert "max_completion_tokens" not in payload["params"]


def test_embedding_adapter_filters_invalid_vectors() -> None:
    model = EmbeddingOpenAI(model="text-embedding-v4")

    response = asyncio.run(
        model._format_response(
            {
                "data": [
                    {"embedding": [0.1, 0.2]},
                    {"embedding": None},
                    {"unexpected": True},
                ]
            }
        )
    )

    assert response.success is True
    assert response.extra is not None
    assert response.extra.data is not None
    assert response.extra.data["embeddings"] == [[0.1, 0.2]]
