"""研究场景配置加载的离线回归测试。"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pytest
from mmengine import Config as MMConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_CONFIG = PROJECT_ROOT / "configs" / "tool_calling_agent.py"


def test_scene_config_inherits_complete_model_defaults() -> None:
    """Web 只覆盖聊天模型时，场景仍应保留完整的向量模型配置。"""
    config = MMConfig.fromfile(SCENE_CONFIG)

    assert config.model_provider == "qwen"
    assert config.model_id == "qwen3-max"
    assert config.embedding_provider == "qwen"
    assert config.embedding_model_id == "text-embedding-v4"
    assert config.embedding_model_name == "qwen/text-embedding-v4"
    assert config.fallback_models == []
    assert config.embedding_fallback_models == []


def test_environment_options_load_dotenv_before_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FastAPI 入口应能读取项目根目录的本地模型配置。"""
    config_module = import_module("src.config.config")
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

    def fake_load_dotenv(*, verbose: bool) -> bool:
        assert verbose is False
        monkeypatch.setenv("EMBEDDING_MODEL", "web-embedding-model")
        return True

    monkeypatch.setattr(config_module, "load_dotenv", fake_load_dotenv)

    options = config_module.get_environment_model_options()

    assert options["embedding_model_id"] == "web-embedding-model"
