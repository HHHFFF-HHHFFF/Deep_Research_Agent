"""统一编排一次深度研究任务的应用服务。"""

from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any, Protocol

from pydantic import BaseModel, Field

from .research_request import ResearchRequest
from .research_result import ProgressEvent, ResearchResult, ResearchStatus

ProgressCallback = Callable[[ProgressEvent], Awaitable[None] | None]


class ResearchRunOutput(BaseModel):
    """运行时成功完成研究后交给应用层的标准输出。"""

    report: str | None = Field(default=None, description="Markdown 研究报告。")
    output_files: list[str] = Field(
        default_factory=list,
        description="研究过程中生成的文件。",
    )
    metrics: dict[str, int | float] = Field(
        default_factory=dict,
        description="运行时收集的令牌、来源数量等指标。",
    )


class ResearchRuntime(Protocol):
    """定义 `ResearchService` 所需的最小运行时边界。"""

    async def initialize(self) -> None:
        """初始化本次研究需要的资源。"""

    async def execute(
        self,
        request: ResearchRequest,
        task_id: str,
    ) -> ResearchRunOutput:
        """执行研究并返回标准输出。"""

    async def cleanup(self) -> None:
        """释放本次研究占用的资源。"""


class ResearchService:
    """把运行时结果稳定转换为应用层终态和进度事件。"""

    def __init__(
        self,
        runtime: ResearchRuntime,
        *,
        timeout_seconds: float | None = None,
    ) -> None:
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("研究超时时间必须大于零")
        self._runtime = runtime
        self._timeout_seconds = timeout_seconds

    async def run(
        self,
        request: ResearchRequest,
        *,
        task_id: str | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> ResearchResult:
        """执行一次研究，并保证始终返回合法的终态结果。"""
        resolved_task_id = task_id or f"research-{uuid.uuid4().hex}"
        started_at = datetime.now(timezone.utc)
        started_clock = time.perf_counter()
        events: list[ProgressEvent] = []

        async def emit(
            status: ResearchStatus,
            message: str,
            progress: int,
            details: dict[str, Any] | None = None,
        ) -> None:
            event = ProgressEvent(
                task_id=resolved_task_id,
                status=status,
                message=message,
                progress=progress,
                sequence=len(events),
                details=details or {},
            )
            events.append(event)
            if on_progress is None:
                return
            try:
                callback_result = on_progress(event)
                if inspect.isawaitable(callback_result):
                    await callback_result
            except Exception:
                # 进度消费者故障不能中断研究主流程。
                return

        async def execute_runtime() -> ResearchRunOutput:
            await emit(
                ResearchStatus.INITIALIZING,
                "正在初始化研究运行时",
                5,
            )
            await self._runtime.initialize()
            await emit(
                ResearchStatus.PLANNING,
                "研究运行时已就绪，开始执行研究任务",
                15,
            )
            output = await self._runtime.execute(request, resolved_task_id)
            if not output.report and not output.output_files:
                raise RuntimeError("研究运行时未返回报告或输出文件")
            await emit(
                ResearchStatus.REPORTING,
                "研究执行完成，正在整理最终结果",
                90,
            )
            return output

        output: ResearchRunOutput | None = None
        status = ResearchStatus.COMPLETED
        error: str | None = None

        try:
            await emit(ResearchStatus.PENDING, "研究任务已创建", 0)
            if self._timeout_seconds is None:
                output = await execute_runtime()
            else:
                output = await asyncio.wait_for(
                    execute_runtime(),
                    timeout=self._timeout_seconds,
                )
        except asyncio.TimeoutError:
            status = ResearchStatus.TIMED_OUT
            error = f"研究任务超过 {self._timeout_seconds:g} 秒未完成"
        except asyncio.CancelledError:
            status = ResearchStatus.CANCELLED
        except Exception as runtime_error:
            status = ResearchStatus.FAILED
            error = f"{type(runtime_error).__name__}: {runtime_error}"

        cleanup_error = await self._cleanup_runtime()
        if cleanup_error:
            if status == ResearchStatus.COMPLETED:
                status = ResearchStatus.FAILED
                error = cleanup_error
                output = None
            elif error:
                error = f"{error}；{cleanup_error}"
            else:
                error = cleanup_error

        final_messages = {
            ResearchStatus.COMPLETED: "研究任务已完成",
            ResearchStatus.FAILED: "研究任务执行失败",
            ResearchStatus.CANCELLED: "研究任务已取消",
            ResearchStatus.TIMED_OUT: "研究任务执行超时",
        }
        final_progress = (
            100 if status == ResearchStatus.COMPLETED else events[-1].progress
        )
        await emit(status, final_messages[status], final_progress)

        metrics = dict(output.metrics) if output else {}
        metrics.setdefault("duration_seconds", time.perf_counter() - started_clock)

        return ResearchResult(
            task_id=resolved_task_id,
            status=status,
            report=output.report if output else None,
            error=error,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            events=events,
            output_files=list(output.output_files) if output else [],
            metrics=metrics,
        )

    async def _cleanup_runtime(self) -> str | None:
        """释放运行时资源，并把清理异常转换为稳定错误文本。"""
        try:
            await self._runtime.cleanup()
        except Exception as cleanup_error:
            return f"资源释放失败：{type(cleanup_error).__name__}: {cleanup_error}"
        return None


class AgentResearchRuntime:
    """把现有模型、工具和 Agent 管理器适配到应用层运行时协议。"""

    def __init__(
        self,
        *,
        config: Any,
        model_manager: Any,
        prompt_manager: Any,
        memory_manager: Any,
        tool_server: Any,
        skill_server: Any,
        environment_server: Any,
        agent_server: Any,
        version_manager: Any,
        agent_name: str = "tool_calling",
    ) -> None:
        self._config = config
        self._model_manager = model_manager
        self._prompt_manager = prompt_manager
        self._memory_manager = memory_manager
        self._tool_server = tool_server
        self._skill_server = skill_server
        self._environment_server = environment_server
        self._agent_server = agent_server
        self._version_manager = version_manager
        self._agent_name = agent_name

    async def initialize(self) -> None:
        """按照依赖顺序初始化现有研究组件。"""
        await self._model_manager.initialize(
            primary_model=self._config.model_name,
            fallback_models=self._config.fallback_models,
            embedding_model=self._config.embedding_model_name,
            embedding_fallback_models=self._config.embedding_fallback_models,
        )
        await self._prompt_manager.initialize()
        await self._memory_manager.initialize(
            memory_names=self._config.memory_names,
        )
        await self._tool_server.initialize(tool_names=self._config.tool_names)
        await self._skill_server.initialize(
            skill_names=getattr(self._config, "skill_names", None),
        )
        await self._environment_server.initialize(self._config.env_names)
        await self._agent_server.initialize(agent_names=self._config.agent_names)
        await self._version_manager.initialize()

    async def execute(
        self,
        request: ResearchRequest,
        task_id: str,
    ) -> ResearchRunOutput:
        """调用现有工具型智能体，并提取报告和输出文件。"""
        from src.session.types import SessionContext

        response = await self._agent_server(
            name=self._agent_name,
            input={"task": request.task, "files": request.files},
            ctx=SessionContext(id=task_id),
        )
        if not bool(getattr(response, "success", False)):
            message = str(getattr(response, "message", "智能体未返回错误说明"))
            raise RuntimeError(message)

        report = str(getattr(response, "message", "")).strip() or None
        extra = getattr(response, "extra", None)
        raw_file_path = getattr(extra, "file_path", None) if extra else None
        if isinstance(raw_file_path, str):
            output_files = [raw_file_path]
        elif isinstance(raw_file_path, list):
            output_files = [str(path) for path in raw_file_path]
        else:
            output_files = []

        return ResearchRunOutput(report=report, output_files=output_files)

    async def cleanup(self) -> None:
        """按照依赖的逆序释放已经创建的研究组件。"""
        cleanup_errors: list[str] = []
        components = (
            ("智能体", self._agent_server),
            ("环境", self._environment_server),
            ("技能", self._skill_server),
            ("工具", self._tool_server),
            ("记忆", self._memory_manager),
            ("提示词", self._prompt_manager),
        )
        for component_name, component in components:
            cleanup = getattr(component, "cleanup", None)
            if not callable(cleanup):
                continue
            try:
                cleanup_result = cleanup()
                if inspect.isawaitable(cleanup_result):
                    await cleanup_result
            except Exception as cleanup_error:
                cleanup_errors.append(f"{component_name}：{cleanup_error}")

        if cleanup_errors:
            raise RuntimeError("；".join(cleanup_errors))


def create_default_research_service(
    config: Any,
    *,
    timeout_seconds: float | None = None,
) -> ResearchService:
    """使用项目现有管理器创建默认研究应用服务。"""
    from src.agent.server import acp
    from src.environment.server import ecp
    from src.memory.server import memory_manager
    from src.model.runtime_manager import model_manager
    from src.prompt.server import prompt_manager
    from src.skill.server import scp
    from src.tool.server import tcp
    from src.version.server import version_manager

    agent_names = getattr(config, "agent_names", None) or ["tool_calling"]
    runtime = AgentResearchRuntime(
        config=config,
        model_manager=model_manager,
        prompt_manager=prompt_manager,
        memory_manager=memory_manager,
        tool_server=tcp,
        skill_server=scp,
        environment_server=ecp,
        agent_server=acp,
        version_manager=version_manager,
        agent_name=agent_names[0],
    )
    return ResearchService(runtime, timeout_seconds=timeout_seconds)
