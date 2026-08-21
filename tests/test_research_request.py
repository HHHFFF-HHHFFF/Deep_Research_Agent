"""研究请求输入的离线测试。"""

import pytest
from pydantic import ValidationError

from src.application import ResearchRequest, resolve_research_task


def test_command_line_task_takes_priority() -> None:
    """命令行提供主题时不应调用交互式输入。"""
    task = resolve_research_task(
        "调研多文档 RAG",
        prompt=lambda _: pytest.fail("不应调用交互式输入"),
    )

    assert task == "调研多文档 RAG"


def test_missing_task_uses_interactive_prompt() -> None:
    """没有命令行主题时应读取终端输入。"""
    task = resolve_research_task(None, prompt=lambda _: "调研智能体评测")

    assert task == "调研智能体评测"


def test_non_interactive_environment_requires_task_argument() -> None:
    """非交互环境无法读取输入时应给出明确错误。"""

    def raise_eof(_: str) -> str:
        raise EOFError

    with pytest.raises(ValueError, match="--task"):
        resolve_research_task(None, prompt=raise_eof)


def test_research_request_normalizes_and_validates_task() -> None:
    """研究请求应清理空白并拒绝空主题。"""
    request = ResearchRequest(task="  调研提示注入防护  ")

    assert request.task == "调研提示注入防护"

    with pytest.raises(ValidationError, match="研究主题不能为空"):
        ResearchRequest(task="   ")


def test_research_request_normalizes_files_and_model_options() -> None:
    """研究请求应清理重复文件，并校验模型覆盖项。"""
    request = ResearchRequest(
        task="调研本地 RAG",
        files=[" 资料.pdf ", "资料.pdf", "  "],
        model_provider="deepseek",
        model_id="deepseek-chat",
        fallback_models=["qwen/qwen3-max", " qwen/qwen3-max "],
    )

    assert request.files == ["资料.pdf"]
    assert request.fallback_models == ["qwen/qwen3-max"]

    with pytest.raises(ValidationError, match="必须同时指定模型标识"):
        ResearchRequest(task="调研模型路由", model_provider="deepseek")

    with pytest.raises(ValidationError, match="提供方/模型"):
        ResearchRequest(task="调研模型路由", fallback_models=["qwen3-max"])
