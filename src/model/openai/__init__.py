"""OpenAI 模型组件的按需导出。"""

from typing import Any

__all__ = [
    "ChatOpenAI",
    "EmbeddingOpenAI",
    "OpenAIChatSerializer",
    "OpenAIResponseSerializer",
    "ResponseOpenAI",
    "TranscribeOpenAI",
]


def __getattr__(name: str) -> Any:
    """仅在实际使用组件时导入，避免加载无关的音频或响应依赖。"""
    if name == "ChatOpenAI":
        from .chat import ChatOpenAI

        return ChatOpenAI
    if name == "ResponseOpenAI":
        from .response import ResponseOpenAI

        return ResponseOpenAI
    if name == "TranscribeOpenAI":
        from .transcribe import TranscribeOpenAI

        return TranscribeOpenAI
    if name == "EmbeddingOpenAI":
        from .embedding import EmbeddingOpenAI

        return EmbeddingOpenAI
    if name in {"OpenAIChatSerializer", "OpenAIResponseSerializer"}:
        from .serializer import OpenAIChatSerializer, OpenAIResponseSerializer

        return {
            "OpenAIChatSerializer": OpenAIChatSerializer,
            "OpenAIResponseSerializer": OpenAIResponseSerializer,
        }[name]
    raise AttributeError(name)
