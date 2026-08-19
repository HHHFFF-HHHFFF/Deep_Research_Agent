"""可复用研究运行入口的离线测试。"""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src import research_runner
from src.application import ResearchProgress, ResearchRequest, ResearchStage
from src.research_runner import (
    ResearchCancelledError,
    ResearchRunError,
    run_research,
)
from src.session.types import SessionContext


@pytest.fixture
def stub_runtime(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """替换真实组件初始化，并记录收到的运行参数。"""
    captured: dict[str, Any] = {}

    async def fake_initialize(
        request: ResearchRequest,
        config_path: str,
        cfg_options: dict[str, Any] | None,
    ) -> research_runner._RuntimeSettings:
        captured["request"] = request
        captured["config_path"] = config_path
        captured["cfg_options"] = cfg_options
        return research_runner._RuntimeSettings(
            model_name="qwen/qwen-plus",
            report_base_dir=None,
        )

    monkeypatch.setattr(research_runner, "_initialize_runtime", fake_initialize)
    return captured


@pytest.mark.asyncio
async def test_run_research_returns_report_and_emits_stages(
    monkeypatch: pytest.MonkeyPatch,
    stub_runtime: dict[str, Any],
) -> None:
    """成功运行应返回结构化报告，并按顺序通知真实阶段。"""
    request = ResearchRequest(
        task="  调研多文档 RAG  ",
        files=["资料一.pdf"],
        model_provider="qwen",
        model_id="qwen-plus",
    )
    stages: list[ResearchStage] = []

    async def fake_invoke(
        request: ResearchRequest,
        ctx: SessionContext,
    ) -> SimpleNamespace:
        assert request.task == "调研多文档 RAG"
        assert ctx is not None
        return SimpleNamespace(
            success=True,
            message="# 研究报告\n\n这是离线报告。",
        )

    monkeypatch.setattr(research_runner, "_invoke_agent", fake_invoke)

    result = await run_research(
        request,
        config_path="configs/test.py",
        cfg_options={"max_tokens": 1024},
        on_progress=lambda progress: stages.append(progress.stage),
    )

    assert result.task == "调研多文档 RAG"
    assert result.report.startswith("# 研究报告")
    assert result.model_name == "qwen/qwen-plus"
    assert result.files == ["资料一.pdf"]
    assert result.session_id
    assert result.report_path is None
    assert stages == [
        ResearchStage.INITIALIZING,
        ResearchStage.RESEARCHING,
        ResearchStage.COMPLETED,
    ]
    assert stub_runtime["config_path"] == "configs/test.py"
    assert stub_runtime["cfg_options"] == {"max_tokens": 1024}


@pytest.mark.asyncio
async def test_run_research_prefers_generated_markdown_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """报告工具已生成文件时，应返回文件正文而不是完成提示。"""

    async def fake_initialize(
        request: ResearchRequest,
        config_path: str,
        cfg_options: dict[str, Any] | None,
    ) -> research_runner._RuntimeSettings:
        return research_runner._RuntimeSettings(
            model_name="deepseek/deepseek-chat",
            report_base_dir=tmp_path,
        )

    async def fake_invoke(
        request: ResearchRequest,
        ctx: SessionContext,
    ) -> SimpleNamespace:
        session_id = str(ctx.id)
        (tmp_path / f"{session_id}.md").write_text(
            "# 文件中的最终报告\n\n完整内容。",
            encoding="utf-8",
        )
        return SimpleNamespace(success=True, message="报告已经生成")

    monkeypatch.setattr(research_runner, "_initialize_runtime", fake_initialize)
    monkeypatch.setattr(research_runner, "_invoke_agent", fake_invoke)

    result = await run_research(ResearchRequest(task="调研报告读取"))

    assert result.report.startswith("# 文件中的最终报告")
    assert result.report_path == str(tmp_path / f"{result.session_id}.md")
    assert result.model_name == "deepseek/deepseek-chat"


@pytest.mark.asyncio
async def test_run_research_converts_agent_failure_to_stable_error(
    monkeypatch: pytest.MonkeyPatch,
    stub_runtime: dict[str, Any],
) -> None:
    """智能体未完成时应抛出稳定错误并通知失败阶段。"""
    stages: list[ResearchStage] = []

    async def fake_invoke(
        request: ResearchRequest,
        ctx: SessionContext,
    ) -> SimpleNamespace:
        return SimpleNamespace(success=False, message="达到最大研究步数")

    monkeypatch.setattr(research_runner, "_invoke_agent", fake_invoke)

    with pytest.raises(ResearchRunError, match="达到最大研究步数"):
        await run_research(
            ResearchRequest(task="调研智能体失败处理"),
            on_progress=lambda progress: stages.append(progress.stage),
        )

    assert stages[-1] is ResearchStage.FAILED
    assert ResearchStage.COMPLETED not in stages
    assert stub_runtime["request"].task == "调研智能体失败处理"


@pytest.mark.asyncio
async def test_run_research_honors_cancel_event_before_agent_call(
    monkeypatch: pytest.MonkeyPatch,
    stub_runtime: dict[str, Any],
) -> None:
    """研究阶段收到取消信号后不应继续调用智能体。"""
    cancel_event = asyncio.Event()
    stages: list[ResearchStage] = []

    async def capture_progress(progress: ResearchProgress) -> None:
        stage = progress.stage
        stages.append(stage)
        if stage is ResearchStage.RESEARCHING:
            cancel_event.set()

    async def unexpected_invoke(
        request: ResearchRequest,
        ctx: SessionContext,
    ) -> SimpleNamespace:
        pytest.fail("取消后不应调用智能体")

    monkeypatch.setattr(research_runner, "_invoke_agent", unexpected_invoke)

    with pytest.raises(ResearchCancelledError, match="已取消"):
        await run_research(
            ResearchRequest(task="调研取消逻辑"),
            on_progress=capture_progress,
            cancel_event=cancel_event,
        )

    assert stages == [
        ResearchStage.INITIALIZING,
        ResearchStage.RESEARCHING,
        ResearchStage.CANCELLED,
    ]
    assert stub_runtime["request"].task == "调研取消逻辑"


@pytest.mark.asyncio
async def test_run_research_hides_unexpected_error_details(
    monkeypatch: pytest.MonkeyPatch,
    stub_runtime: dict[str, Any],
) -> None:
    """未知异常不应把底层敏感详情暴露给调用方。"""

    async def fake_invoke(
        request: ResearchRequest,
        ctx: SessionContext,
    ) -> SimpleNamespace:
        raise RuntimeError("Authorization: secret-token")

    monkeypatch.setattr(research_runner, "_invoke_agent", fake_invoke)

    with pytest.raises(ResearchRunError) as error_info:
        await run_research(ResearchRequest(task="调研异常处理"))

    assert "secret-token" not in str(error_info.value)
    assert stub_runtime["request"].task == "调研异常处理"
