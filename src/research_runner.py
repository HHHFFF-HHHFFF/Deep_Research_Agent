"""供命令行和后续 Web API 共用的研究运行入口。"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from argparse import Namespace
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

from src.application import (
    ResearchProgress,
    ResearchRequest,
    ResearchResult,
    ResearchStage,
)
from src.logger import logger
from src.session.types import SessionContext

DEFAULT_CONFIG_PATH = str(
    Path(__file__).resolve().parents[1] / "configs" / "tool_calling_agent.py"
)

ProgressCallback = Callable[[ResearchProgress], Awaitable[None] | None]
_T = TypeVar("_T")


class ResearchRunError(RuntimeError):
    """表示可以安全展示给调用方的研究运行错误。"""


class ResearchCancelledError(ResearchRunError):
    """表示调用方主动取消了研究任务。"""


class AgentExecutionResponse(Protocol):
    """研究入口实际使用的智能体响应字段。"""

    success: bool
    message: str


@dataclass(frozen=True)
class _RuntimeSettings:
    """研究运行完成初始化后需要保留的少量配置。"""

    model_name: str
    report_base_dir: Path | None


def _create_config_args(
    request: ResearchRequest,
    cfg_options: dict[str, Any] | None,
) -> Namespace:
    """把研究请求中的模型覆盖项转换为配置加载器参数。"""
    return Namespace(
        cfg_options=dict(cfg_options or {}),
        model_provider=request.model_provider,
        model_id=request.model_id,
        fallback_models=request.fallback_models,
        embedding_provider=request.embedding_provider,
        embedding_model_id=request.embedding_model_id,
    )


async def _emit_progress(
    callback: ProgressCallback | None,
    stage: ResearchStage,
    message: str,
) -> None:
    """兼容同步和异步阶段回调。"""
    if callback is None:
        return

    callback_result = callback(ResearchProgress(stage=stage, message=message))
    if inspect.isawaitable(callback_result):
        await cast(Awaitable[None], callback_result)


async def _await_with_cancellation(
    operation: Callable[[], Awaitable[_T]],
    cancel_event: asyncio.Event | None,
) -> _T:
    """在等待异步操作时同时监听协作式取消信号。"""
    if cancel_event is None:
        return await operation()
    if cancel_event.is_set():
        raise ResearchCancelledError("研究任务已取消")

    operation_task: asyncio.Future[_T] = asyncio.ensure_future(operation())
    cancel_task: asyncio.Task[bool] = asyncio.create_task(cancel_event.wait())
    try:
        pending_operations: set[asyncio.Future[Any]] = {
            operation_task,
            cancel_task,
        }
        done, _ = await asyncio.wait(
            pending_operations,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if operation_task in done:
            return await operation_task

        operation_task.cancel()
        await asyncio.gather(operation_task, return_exceptions=True)
        raise ResearchCancelledError("研究任务已取消")
    except asyncio.CancelledError:
        operation_task.cancel()
        cancel_task.cancel()
        await asyncio.gather(operation_task, cancel_task, return_exceptions=True)
        raise
    finally:
        if not cancel_task.done():
            cancel_task.cancel()
            await asyncio.gather(cancel_task, return_exceptions=True)


async def _initialize_runtime(
    request: ResearchRequest,
    config_path: str,
    cfg_options: dict[str, Any] | None,
) -> _RuntimeSettings:
    """按原命令行顺序初始化研究核心及其依赖。"""
    from src.agent import acp
    from src.config import config
    from src.environment import ecp
    from src.memory import memory_manager
    from src.model import model_manager
    from src.prompt import prompt_manager
    from src.skill import scp
    from src.tool import tcp
    from src.version import version_manager

    config.initialize(
        config_path=config_path,
        args=_create_config_args(request, cfg_options),
    )
    logger.initialize(config=config)
    logger.info(f"| Config: {config.pretty_text}")

    logger.info("| 🧠 正在初始化模型管理器……")
    await model_manager.initialize(
        primary_model=config.model_name,
        fallback_models=config.fallback_models,
        embedding_model=config.embedding_model_name,
        embedding_fallback_models=config.embedding_fallback_models,
    )
    logger.info(f"| ✅ 模型管理器初始化完成：{await model_manager.list()}")

    logger.info("| 📁 正在初始化提示词管理器……")
    await prompt_manager.initialize()
    logger.info(f"| ✅ 提示词管理器初始化完成：{await prompt_manager.list()}")

    logger.info("| 📁 正在初始化记忆管理器……")
    await memory_manager.initialize(memory_names=config.memory_names)
    logger.info(f"| ✅ 记忆管理器初始化完成：{await memory_manager.list()}")

    logger.info("| 🛠️ 正在初始化研究工具……")
    await tcp.initialize(tool_names=config.tool_names)
    logger.info(f"| ✅ 研究工具初始化完成：{await tcp.list()}")

    logger.info("| 🎯 正在初始化技能……")
    await scp.initialize(skill_names=getattr(config, "skill_names", None))
    logger.info(f"| ✅ 技能初始化完成：{await scp.list()}")

    logger.info("| 🎮 正在初始化运行环境……")
    await ecp.initialize(config.env_names)
    logger.info(f"| ✅ 运行环境初始化完成：{ecp.list()}")

    logger.info("| 🤖 正在初始化智能体……")
    await acp.initialize(agent_names=config.agent_names)
    logger.info(f"| ✅ 智能体初始化完成：{await acp.list()}")

    logger.info("| 📁 正在初始化版本管理器……")
    await version_manager.initialize()
    versions = json.dumps(await version_manager.list(), ensure_ascii=False, indent=4)
    logger.info(f"| ✅ 版本管理器初始化完成：{versions}")

    reporter_options = getattr(config, "reporter_tool", None)
    reporter_base_dir = (
        reporter_options.get("base_dir")
        if reporter_options is not None and hasattr(reporter_options, "get")
        else None
    )
    return _RuntimeSettings(
        model_name=str(config.model_name),
        report_base_dir=Path(str(reporter_base_dir)) if reporter_base_dir else None,
    )


async def _invoke_agent(
    request: ResearchRequest,
    ctx: SessionContext,
) -> AgentExecutionResponse:
    """调用现有工具型 Agent，不改写其执行循环。"""
    from src.agent import acp
    from src.agent.types import AgentResponse

    response = await acp(
        name="tool_calling",
        input={"task": request.task, "files": request.files},
        ctx=ctx,
    )
    if not isinstance(response, AgentResponse):
        raise ResearchRunError("智能体返回了无法识别的结果")
    return response


async def _load_generated_report(
    runtime: _RuntimeSettings,
    session_id: str,
    fallback: str,
) -> tuple[str, str | None]:
    """优先读取报告工具生成的 Markdown，缺失时使用智能体结果。"""
    if runtime.report_base_dir is None:
        return fallback, None

    safe_session_id = re.sub(r"[^\w\s-]", "", session_id).strip().replace(" ", "_")
    if not safe_session_id:
        return fallback, None

    report_root = runtime.report_base_dir.resolve()
    report_path = (report_root / f"{safe_session_id}.md").resolve()
    if report_path.parent != report_root or not report_path.is_file():
        return fallback, None

    report = await asyncio.to_thread(report_path.read_text, encoding="utf-8")
    normalized_report = report.strip()
    if not normalized_report:
        return fallback, None
    return normalized_report, str(report_path)


async def run_research(
    request: ResearchRequest,
    *,
    config_path: str = DEFAULT_CONFIG_PATH,
    cfg_options: dict[str, Any] | None = None,
    on_progress: ProgressCallback | None = None,
    cancel_event: asyncio.Event | None = None,
) -> ResearchResult:
    """执行一次研究，并返回可供命令行或 API 使用的结构化结果。"""
    try:
        await _emit_progress(
            on_progress,
            ResearchStage.INITIALIZING,
            "正在初始化研究组件",
        )
        runtime = await _await_with_cancellation(
            lambda: _initialize_runtime(request, config_path, cfg_options),
            cancel_event,
        )

        logger.info(f"| 📋 研究主题：{request.task}")
        logger.info(f"| 📂 本地文件：{request.files}")
        await _emit_progress(
            on_progress,
            ResearchStage.RESEARCHING,
            "正在执行网页研究和本地文档检索",
        )

        ctx = SessionContext()
        response = await _await_with_cancellation(
            lambda: _invoke_agent(request, ctx),
            cancel_event,
        )
        if not response.success:
            detail = response.message.strip() or "智能体未能完成研究"
            raise ResearchRunError(f"研究未完成：{detail}")

        report, report_path = await _load_generated_report(
            runtime,
            ctx.id,
            response.message.strip(),
        )
        if not report:
            raise ResearchRunError("研究已结束，但没有生成报告内容")

        result = ResearchResult(
            task=request.task,
            report=report,
            model_name=runtime.model_name,
            session_id=ctx.id,
            files=request.files,
            report_path=report_path,
        )
        await _emit_progress(
            on_progress,
            ResearchStage.COMPLETED,
            "研究报告已经生成",
        )
        return result
    except ResearchCancelledError:
        await _emit_progress(
            on_progress,
            ResearchStage.CANCELLED,
            "研究任务已取消",
        )
        raise
    except asyncio.CancelledError:
        await _emit_progress(
            on_progress,
            ResearchStage.CANCELLED,
            "研究任务已取消",
        )
        raise
    except ResearchRunError:
        await _emit_progress(
            on_progress,
            ResearchStage.FAILED,
            "研究任务运行失败",
        )
        raise
    except Exception as error:
        logger.error(f"| ❌ 研究运行异常：{type(error).__name__}")
        await _emit_progress(
            on_progress,
            ResearchStage.FAILED,
            "研究任务运行失败",
        )
        raise ResearchRunError("研究运行失败，请查看本地日志") from error


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "ProgressCallback",
    "ResearchCancelledError",
    "ResearchRunError",
    "run_research",
]
