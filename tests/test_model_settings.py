"""模型运行配置的离线测试。"""

import pytest

from src.model.settings import PROVIDERS, ModelRuntimeSettings, split_model_reference


def test_from_env_separates_chat_and_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("MODEL_NAME", "deepseek-v4-flash")
    monkeypatch.setenv("MODEL_FALLBACKS", "qwen/qwen-plus")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "qwen")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-v4")

    settings = ModelRuntimeSettings.from_env()

    assert settings.primary_model == "deepseek/deepseek-v4-flash"
    assert settings.fallback_models == ["qwen/qwen-plus"]
    assert settings.embedding_model == "qwen/text-embedding-v4"


def test_invalid_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="不支持的模型提供方"):
        split_model_reference("unknown/example")


def test_qwen_workspace_host_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "QWEN_BASE_URL",
        "example.cn-beijing.maas.aliyuncs.com",
    )

    assert PROVIDERS["qwen"].base_url() == (
        "https://example.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    )
