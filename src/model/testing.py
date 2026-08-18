"""供离线测试使用的确定性模型。"""

from __future__ import annotations

from typing import Any

from src.model.types import LLMExtra, LLMResponse


class DeterministicChatModel:
    """始终返回固定文本，不访问任何外部服务。"""

    def __init__(self, response: str = "离线模型响应") -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        return LLMResponse(
            success=True,
            message=self.response,
            extra=LLMExtra(
                data={
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    }
                }
            ),
        )


class DeterministicEmbeddingModel:
    """根据输入条数返回固定维度向量。"""

    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        messages = kwargs.get("messages") or []
        embeddings = [
            [float(index + 1)] * self.dimension for index, _ in enumerate(messages)
        ]
        return LLMResponse(
            success=True,
            message=f"生成了 {len(embeddings)} 个测试向量",
            extra=LLMExtra(data={"embeddings": embeddings}),
        )


class FailingModel:
    """用于验证备用模型降级流程。"""

    def __init__(self, message: str = "模拟调用失败") -> None:
        self.message = message
        self.calls = 0

    async def __call__(self, **kwargs: Any) -> LLMResponse:
        self.calls += 1
        return LLMResponse(success=False, message=self.message)


__all__ = [
    "DeterministicChatModel",
    "DeterministicEmbeddingModel",
    "FailingModel",
]
