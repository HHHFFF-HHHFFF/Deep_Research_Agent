"""统一研究应用服务的离线测试。"""

import asyncio
from types import SimpleNamespace
from typing import Any

from src.application import (
    AgentResearchRuntime,
    ProgressEvent,
    ResearchRequest,
    ResearchRunOutput,
    ResearchService,
    ResearchStatus,
)


class FakeResearchRuntime:
    """提供不访问网络和文件系统的确定性研究运行时。"""

    def __init__(
        self,
        *,
        output: ResearchRunOutput | None = None,
        initialize_error: Exception | None = None,
        execute_error: Exception | None = None,
        cleanup_error: Exception | None = None,
        wait_forever: bool = False,
    ) -> None:
        self.output = output or ResearchRunOutput(report="# 离线研究报告")
        self.initialize_error = initialize_error
        self.execute_error = execute_error
        self.cleanup_error = cleanup_error
        self.wait_forever = wait_forever
        self.initialized = False
        self.executed = False
        self.cleaned = False
        self.execute_started = asyncio.Event()

    async def initialize(self) -> None:
        self.initialized = True
        if self.initialize_error:
            raise self.initialize_error

    async def execute(
        self,
        request: ResearchRequest,
        task_id: str,
    ) -> ResearchRunOutput:
        self.executed = True
        self.execute_started.set()
        if self.execute_error:
            raise self.execute_error
        if self.wait_forever:
            await asyncio.Event().wait()
        return self.output

    async def cleanup(self) -> None:
        self.cleaned = True
        if self.cleanup_error:
            raise self.cleanup_error


class RecordedComponent:
    """记录现有管理器适配器的调用顺序和参数。"""

    def __init__(self, name: str, calls: list[str]) -> None:
        self.name = name
        self.calls = calls
        self.initialize_kwargs: dict[str, Any] = {}

    async def initialize(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(f"initialize:{self.name}")
        self.initialize_kwargs = dict(kwargs)
        if args:
            self.initialize_kwargs["args"] = args

    async def cleanup(self) -> None:
        self.calls.append(f"cleanup:{self.name}")


class RecordedAgentServer(RecordedComponent):
    async def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append("execute:agent")
        self.execute_kwargs = kwargs
        return SimpleNamespace(
            success=True,
            message="# 智能体研究报告",
            extra=SimpleNamespace(file_path="reports/agent.md"),
        )


def test_research_service_returns_completed_result_and_progress() -> None:
    runtime = FakeResearchRuntime(
        output=ResearchRunOutput(
            report="# RAG 调研",
            output_files=["reports/rag.md"],
            metrics={"source_count": 3},
        )
    )
    callback_events: list[ProgressEvent] = []
    service = ResearchService(runtime)

    result = asyncio.run(
        service.run(
            ResearchRequest(task="调研 RAG"),
            task_id="research-success",
            on_progress=callback_events.append,
        )
    )

    assert result.status == ResearchStatus.COMPLETED
    assert result.report == "# RAG 调研"
    assert result.output_files == ["reports/rag.md"]
    assert result.metrics["source_count"] == 3
    assert result.metrics["duration_seconds"] >= 0
    assert [event.status for event in result.events] == [
        ResearchStatus.PENDING,
        ResearchStatus.INITIALIZING,
        ResearchStatus.PLANNING,
        ResearchStatus.REPORTING,
        ResearchStatus.COMPLETED,
    ]
    assert callback_events == result.events
    assert runtime.initialized is True
    assert runtime.executed is True
    assert runtime.cleaned is True


def test_research_service_converts_runtime_failure_and_cleans_up() -> None:
    runtime = FakeResearchRuntime(execute_error=ValueError("模型返回无效数据"))

    result = asyncio.run(
        ResearchService(runtime).run(
            ResearchRequest(task="失败场景"),
            task_id="research-failed",
        )
    )

    assert result.status == ResearchStatus.FAILED
    assert result.error == "ValueError: 模型返回无效数据"
    assert result.report is None
    assert runtime.cleaned is True


def test_research_service_converts_initialization_failure() -> None:
    runtime = FakeResearchRuntime(initialize_error=RuntimeError("初始化模型失败"))

    result = asyncio.run(
        ResearchService(runtime).run(
            ResearchRequest(task="初始化失败场景"),
            task_id="research-init-failed",
        )
    )

    assert result.status == ResearchStatus.FAILED
    assert result.error == "RuntimeError: 初始化模型失败"
    assert runtime.executed is False
    assert runtime.cleaned is True


def test_research_service_converts_timeout_and_cleans_up() -> None:
    runtime = FakeResearchRuntime(wait_forever=True)

    result = asyncio.run(
        ResearchService(runtime, timeout_seconds=0.01).run(
            ResearchRequest(task="超时场景"),
            task_id="research-timeout",
        )
    )

    assert result.status == ResearchStatus.TIMED_OUT
    assert result.error == "研究任务超过 0.01 秒未完成"
    assert runtime.cleaned is True


def test_research_service_converts_cancellation_and_cleans_up() -> None:
    runtime = FakeResearchRuntime(wait_forever=True)
    service = ResearchService(runtime)

    async def cancel_running_research():
        task = asyncio.create_task(
            service.run(
                ResearchRequest(task="取消场景"),
                task_id="research-cancelled",
            )
        )
        await runtime.execute_started.wait()
        task.cancel()
        return await task

    result = asyncio.run(cancel_running_research())

    assert result.status == ResearchStatus.CANCELLED
    assert runtime.cleaned is True


def test_cleanup_failure_replaces_success_with_failed_result() -> None:
    runtime = FakeResearchRuntime(cleanup_error=RuntimeError("无法关闭工具"))

    result = asyncio.run(
        ResearchService(runtime).run(
            ResearchRequest(task="清理失败场景"),
            task_id="research-cleanup-failed",
        )
    )

    assert result.status == ResearchStatus.FAILED
    assert result.error == "资源释放失败：RuntimeError: 无法关闭工具"
    assert result.report is None


def test_progress_callback_failure_does_not_interrupt_research() -> None:
    runtime = FakeResearchRuntime()

    def broken_callback(_: ProgressEvent) -> None:
        raise RuntimeError("界面连接已断开")

    result = asyncio.run(
        ResearchService(runtime).run(
            ResearchRequest(task="进度回调失败场景"),
            task_id="research-callback-failed",
            on_progress=broken_callback,
        )
    )

    assert result.status == ResearchStatus.COMPLETED
    assert runtime.cleaned is True


def test_agent_runtime_initializes_executes_and_cleans_up_in_order() -> None:
    calls: list[str] = []
    model = RecordedComponent("model", calls)
    prompt = RecordedComponent("prompt", calls)
    memory = RecordedComponent("memory", calls)
    tool = RecordedComponent("tool", calls)
    skill = RecordedComponent("skill", calls)
    environment = RecordedComponent("environment", calls)
    agent = RecordedAgentServer("agent", calls)
    version = RecordedComponent("version", calls)
    config = SimpleNamespace(
        model_name="qwen/qwen-plus",
        fallback_models=["deepseek/deepseek-chat"],
        embedding_model_name="qwen/text-embedding-v4",
        embedding_fallback_models=[],
        memory_names=["general_memory_system"],
        tool_names=["deep_researcher", "reporter"],
        skill_names=[],
        env_names=["file_system"],
        agent_names=["tool_calling"],
    )
    runtime = AgentResearchRuntime(
        config=config,
        model_manager=model,
        prompt_manager=prompt,
        memory_manager=memory,
        tool_server=tool,
        skill_server=skill,
        environment_server=environment,
        agent_server=agent,
        version_manager=version,
    )

    async def run_runtime() -> ResearchRunOutput:
        await runtime.initialize()
        output = await runtime.execute(
            ResearchRequest(task="研究智能体"),
            "research-agent-runtime",
        )
        await runtime.cleanup()
        return output

    output = asyncio.run(run_runtime())

    assert output.report == "# 智能体研究报告"
    assert output.output_files == ["reports/agent.md"]
    assert calls == [
        "initialize:model",
        "initialize:prompt",
        "initialize:memory",
        "initialize:tool",
        "initialize:skill",
        "initialize:environment",
        "initialize:agent",
        "initialize:version",
        "execute:agent",
        "cleanup:agent",
        "cleanup:environment",
        "cleanup:skill",
        "cleanup:tool",
        "cleanup:memory",
        "cleanup:prompt",
    ]
    assert model.initialize_kwargs["primary_model"] == "qwen/qwen-plus"
    assert memory.initialize_kwargs["memory_names"] == ["general_memory_system"]
