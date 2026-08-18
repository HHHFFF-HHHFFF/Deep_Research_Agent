"""统一模型管理器的离线测试。"""

import pytest

from src.message import HumanMessage
from src.model.openai.embedding import EmbeddingOpenAI
from src.model.openai_compatible import ChatOpenAICompatible
from src.model.runtime_manager import ModelManager
from src.model.settings import ModelRuntimeSettings
from src.model.testing import (
    DeterministicChatModel,
    DeterministicEmbeddingModel,
    FailingModel,
)


@pytest.fixture(autouse=True)
def isolate_provider_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """避免开发者本地 `.env` 改变离线测试结果。"""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "离线测试密钥")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "离线测试密钥")
    monkeypatch.setenv("QWEN_BASE_URL", "")
    monkeypatch.setenv("DEEPSEEK_API_BASE", "")


@pytest.fixture
def settings() -> ModelRuntimeSettings:
    return ModelRuntimeSettings(
        primary_model="qwen/qwen-plus",
        fallback_models=["deepseek/deepseek-v4-flash"],
        embedding_model="qwen/text-embedding-v4",
    )


@pytest.mark.asyncio
async def test_failed_response_uses_fallback(settings: ModelRuntimeSettings) -> None:
    manager = ModelManager()
    await manager.initialize(settings)

    primary = FailingModel()
    fallback = DeterministicChatModel("备用模型成功")
    manager.model_clients["qwen/qwen-plus"] = primary
    manager.model_clients["deepseek/deepseek-v4-flash"] = fallback

    result = await manager.achat([HumanMessage(content="测试")])

    assert result.success is True
    assert result.message == "备用模型成功"
    assert primary.calls == 1
    assert len(fallback.calls) == 1


@pytest.mark.asyncio
async def test_embedding_uses_independent_model(settings: ModelRuntimeSettings) -> None:
    manager = ModelManager()
    await manager.initialize(settings)

    embedding = DeterministicEmbeddingModel(dimension=3)
    manager.model_clients["qwen/text-embedding-v4"] = embedding

    result = await manager.aembedding(
        [HumanMessage(content="第一段"), HumanMessage(content="第二段")]
    )

    assert result.success is True
    assert result.extra is not None
    assert result.extra.data is not None
    assert result.extra.data["embeddings"] == [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]


@pytest.mark.asyncio
async def test_manager_only_registers_configured_models(
    settings: ModelRuntimeSettings,
) -> None:
    manager = ModelManager()
    await manager.initialize(settings)

    assert await manager.list() == [
        "qwen/qwen-plus",
        "deepseek/deepseek-v4-flash",
        "qwen/text-embedding-v4",
    ]
    qwen_config = await manager.get_model_config("qwen/qwen-plus")
    deepseek_config = await manager.get_model_config("deepseek/deepseek-v4-flash")
    assert qwen_config is not None
    assert deepseek_config is not None
    assert qwen_config.api_base == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert deepseek_config.api_base == "https://api.deepseek.com"
    assert isinstance(manager.model_clients["qwen/qwen-plus"], ChatOpenAICompatible)
    assert isinstance(
        manager.model_clients["deepseek/deepseek-v4-flash"],
        ChatOpenAICompatible,
    )


@pytest.mark.asyncio
async def test_compatible_adapter_uses_max_tokens() -> None:
    client = ChatOpenAICompatible(
        model="qwen-plus",
        provider_name="qwen",
        api_key="仅用于离线测试",
        max_completion_tokens=321,
        frequency_penalty=None,
    )

    payload = await client._build_params(messages=[HumanMessage(content="测试")])

    assert payload["params"]["max_tokens"] == 321
    assert "max_completion_tokens" not in payload["params"]


@pytest.mark.asyncio
async def test_missing_provider_key_never_reuses_openai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "不能发送给其他提供方")
    manager = ModelManager()
    settings = ModelRuntimeSettings(
        primary_model="deepseek/deepseek-v4-flash",
        embedding_model="qwen/text-embedding-v4",
    )
    await manager.initialize(settings)

    result = await manager.achat([HumanMessage(content="测试")])

    assert result.success is False
    assert "DEEPSEEK_API_KEY" in result.message
    assert "不能发送给其他提供方" not in result.message


@pytest.mark.asyncio
async def test_embedding_response_uses_unified_extra_data() -> None:
    class Item:
        def __init__(self) -> None:
            self.embedding = [0.1, 0.2, 0.3]

    class Usage:
        prompt_tokens = 2
        total_tokens = 2

    class Response:
        def __init__(self) -> None:
            self.data = [Item()]
            self.usage = Usage()

        def model_dump(self) -> dict:
            return {"data": [{"embedding": self.data[0].embedding}]}

    result = await EmbeddingOpenAI()._format_response(Response())

    assert result.extra is not None
    assert result.extra.data is not None
    assert result.extra.data["embeddings"] == [[0.1, 0.2, 0.3]]
