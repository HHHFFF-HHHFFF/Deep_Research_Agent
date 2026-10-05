"""单进程、单活动任务的异步研究管理器。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from src.api.database import ResearchDatabase, StoredFileRecord, TaskRecord
from src.api.models import (
    TERMINAL_TASK_STATUSES,
    TaskCreateRequest,
    TaskPlanConfirmRequest,
    TaskStage,
    TaskStatus,
    build_default_research_plan,
)
from src.application import (
    ResearchActivityStatus,
    ResearchProgress,
    ResearchRequest,
    ResearchResult,
)
from src.document_parser import parsed_cache_path
from src.research_runner import (
    ResearchCancelledError,
    ResearchRunError,
    run_research,
)


class TaskBusyError(RuntimeError):
    """表示当前已有研究任务正在运行。"""


class TaskNotFoundError(RuntimeError):
    """表示请求的研究任务不存在。"""


class TaskStateError(RuntimeError):
    """表示任务当前状态不允许执行该操作。"""


class UnknownFileError(RuntimeError):
    """表示请求引用了不存在的上传文件。"""


class TaskFileLimitError(RuntimeError):
    """表示一次研究引用的文件总量超过个人项目限制。"""


class ResearchRunner(Protocol):
    """W1 研究运行入口在 W2 中使用的最小协议。"""

    def __call__(
        self,
        request: ResearchRequest,
        *,
        on_progress: Callable[[ResearchProgress], Awaitable[None] | None] | None,
        cancel_event: asyncio.Event | None,
    ) -> Awaitable[ResearchResult]: ...


class ResearchTaskManager:
    """在一个 FastAPI 进程中最多运行一个研究任务。"""

    def __init__(
        self,
        database: ResearchDatabase,
        upload_dir: Path,
        report_dir: Path,
        runner: ResearchRunner = run_research,
        max_task_file_bytes: int = 25 * 1024 * 1024,
    ):
        self.database = database
        self.upload_dir = upload_dir.resolve()
        self.report_dir = report_dir.resolve()
        self.runner = runner
        self.max_task_file_bytes = max_task_file_bytes
        self._lock = asyncio.Lock()
        self._active_task_id: str | None = None
        self._active_task: asyncio.Task[None] | None = None
        self._cancel_event: asyncio.Event | None = None
        self._shutting_down = False

    @property
    def active_task_id(self) -> str | None:
        """返回当前真正仍在运行的任务编号。"""
        if self._active_task is None or self._active_task.done():
            return None
        return self._active_task_id

    async def create_task(self, request: TaskCreateRequest) -> TaskRecord:
        """验证文件，并持久化一份等待用户确认的研究计划。"""
        async with self._lock:
            if self._shutting_down:
                raise TaskStateError("服务正在关闭，暂时不能创建研究任务")
            if self._active_task is not None and not self._active_task.done():
                raise TaskBusyError("已有研究任务正在运行，请等待完成或先取消")

            file_ids = [str(file_id) for file_id in request.file_ids]
            files = self.database.get_files(file_ids)
            if len(files) != len(file_ids):
                raise UnknownFileError("请求中包含不存在的上传文件")
            if sum(file.size for file in files) > self.max_task_file_bytes:
                limit_mb = self.max_task_file_bytes / (1024 * 1024)
                raise TaskFileLimitError(
                    f"单次研究使用的资料总量不能超过 {limit_mb:g} MB"
                )

            task_id = str(uuid4())
            record = self.database.create_task(
                task_id=task_id,
                task=request.task.strip(),
                model_provider=request.model_provider,
                model_id=request.model_id,
                file_ids=file_ids,
                research_plan=build_default_research_plan(has_files=bool(files)),
            )
            return record

    async def confirm_task(
        self,
        task_id: str,
        request: TaskPlanConfirmRequest,
    ) -> TaskRecord:
        """确认研究计划，并在没有其他活动任务时启动执行。"""
        async with self._lock:
            if self._shutting_down:
                raise TaskStateError("服务正在关闭，暂时不能启动研究任务")
            record = self.database.get_task(task_id)
            if record is None:
                raise TaskNotFoundError("研究任务不存在")
            if record.status is not TaskStatus.AWAITING_CONFIRMATION:
                raise TaskStateError("当前任务不处于计划确认阶段")
            if self._active_task is not None and not self._active_task.done():
                raise TaskBusyError("已有研究任务正在运行，请等待完成或先取消")

            files = self.database.get_files(record.file_ids)
            if len(files) != len(record.file_ids):
                raise UnknownFileError("研究任务引用的上传文件已不存在")
            if sum(file.size for file in files) > self.max_task_file_bytes:
                limit_mb = self.max_task_file_bytes / (1024 * 1024)
                raise TaskFileLimitError(
                    f"单次研究使用的资料总量不能超过 {limit_mb:g} MB"
                )

            confirmed = self.database.confirm_task_plan(task_id, request.steps)
            if confirmed is None:
                raise TaskStateError("研究计划状态已经变化，请刷新后重试")
            self._cancel_event = asyncio.Event()
            self._active_task_id = task_id
            self._active_task = asyncio.create_task(
                self._execute(confirmed, files, self._cancel_event),
                name=f"research-task-{task_id}",
            )
            return confirmed

    async def cancel_task(self, task_id: str) -> TaskRecord:
        """向当前活动任务发出协作式取消信号。"""
        async with self._lock:
            record = self.database.get_task(task_id)
            if record is None:
                raise TaskNotFoundError("研究任务不存在")
            if record.status is TaskStatus.AWAITING_CONFIRMATION:
                self.database.mark_cancelled(task_id)
                return self.database.get_task(task_id) or record
            if record.status not in {TaskStatus.WAITING, TaskStatus.RUNNING}:
                raise TaskStateError("当前任务状态不能取消")
            if (
                self._active_task_id != task_id
                or self._active_task is None
                or self._active_task.done()
                or self._cancel_event is None
            ):
                raise TaskStateError("研究任务已不在当前进程中运行")
            self._cancel_event.set()
            self.database.update_progress(
                task_id,
                TaskStage.CANCELLING,
                "正在取消研究任务",
            )

        return self.database.get_task(task_id) or record

    async def delete_task(self, task_id: str) -> None:
        """删除已结束任务，以及不再被其他任务引用的本地文件。"""
        async with self._lock:
            record = self.database.get_task(task_id)
            if record is None:
                raise TaskNotFoundError("研究任务不存在")
            if record.status not in TERMINAL_TASK_STATUSES:
                raise TaskStateError("运行中的研究任务不能删除，请先等待完成或取消")

            orphan_files = self.database.get_task_orphan_files(task_id)

            def remove_local_files() -> None:
                report_path = self.report_dir / f"{task_id}.md"
                temporary_path = self.report_dir / f"{task_id}.tmp"
                report_path.unlink(missing_ok=True)
                temporary_path.unlink(missing_ok=True)
                for file in orphan_files:
                    stored_path = Path(file.stored_path).resolve()
                    if stored_path.parent != self.upload_dir:
                        raise TaskStateError("上传资料的存储路径异常，任务未删除")
                    stored_path.unlink(missing_ok=True)
                    parsed_cache_path(stored_path).unlink(missing_ok=True)

            await asyncio.to_thread(remove_local_files)
            deleted = self.database.delete_task(
                task_id,
                [file.id for file in orphan_files],
            )
            if not deleted:
                raise TaskNotFoundError("研究任务不存在")

    async def _execute(
        self,
        record: TaskRecord,
        files: list[StoredFileRecord],
        cancel_event: asyncio.Event,
    ) -> None:
        """执行 W1 研究入口并把每个终态写入 SQLite。"""
        task_id = record.id

        async def on_progress(progress: ResearchProgress) -> None:
            stage = TaskStage(progress.stage.value)
            self.database.update_progress(task_id, stage, progress.message)
            if progress.activity is not None:
                self.database.upsert_activity(task_id, progress.activity)
                for evidence in progress.activity.evidence:
                    self.database.upsert_evidence(task_id, evidence)

        try:
            self.database.mark_running(task_id)
            plan = self.database.get_task_plan(task_id)
            result = await self.runner(
                ResearchRequest(
                    task=record.task,
                    files=[file.stored_path for file in files],
                    model_provider=record.model_provider,
                    model_id=record.model_id,
                    research_plan=plan.steps if plan is not None else [],
                ),
                on_progress=on_progress,
                cancel_event=cancel_event,
            )
            for evidence in result.evidence:
                self.database.upsert_evidence(task_id, evidence)
            if result.citation_validation is not None:
                self.database.save_citation_validation(
                    task_id,
                    result.citation_validation,
                )
            report_path = await self._save_report(task_id, result.report)
            self.database.mark_succeeded(
                task_id,
                actual_model_name=result.model_name,
                report_path=str(report_path),
            )
        except ResearchCancelledError:
            self.database.finalize_running_activities(
                task_id,
                ResearchActivityStatus.CANCELLED,
                "研究任务已取消，本活动未继续执行",
            )
            if self._shutting_down:
                self.database.mark_interrupted(task_id)
            else:
                self.database.mark_cancelled(task_id)
        except ResearchRunError as error:
            self.database.finalize_running_activities(
                task_id,
                ResearchActivityStatus.FAILED,
                "研究任务失败，本活动未能完成",
            )
            self.database.mark_failed(task_id, str(error))
        except asyncio.CancelledError:
            self.database.finalize_running_activities(
                task_id,
                ResearchActivityStatus.CANCELLED,
                "服务停止，本活动已中断",
            )
            self.database.mark_interrupted(task_id)
            raise
        except Exception:
            self.database.finalize_running_activities(
                task_id,
                ResearchActivityStatus.FAILED,
                "研究任务异常，本活动未能完成",
            )
            self.database.mark_failed(task_id, "研究运行失败，请查看本地日志")
        finally:
            async with self._lock:
                if self._active_task_id == task_id:
                    self._active_task_id = None
                    self._active_task = None
                    self._cancel_event = None

    async def _save_report(self, task_id: str, report: str) -> Path:
        """把研究结果写入由任务编号确定的稳定 Markdown 文件。"""
        self.report_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.report_dir / f"{task_id}.md"
        temporary_path = self.report_dir / f"{task_id}.tmp"

        def write_report() -> None:
            temporary_path.write_text(report, encoding="utf-8")
            temporary_path.replace(report_path)

        await asyncio.to_thread(write_report)
        return report_path.resolve()

    async def shutdown(self, timeout: float = 5.0) -> None:
        """关闭应用时中断活动任务并等待其收敛。"""
        async with self._lock:
            self._shutting_down = True
            active_task = self._active_task
            cancel_event = self._cancel_event
        if active_task is None or active_task.done():
            return

        if cancel_event is not None:
            cancel_event.set()
        try:
            await asyncio.wait_for(asyncio.shield(active_task), timeout=timeout)
        except asyncio.TimeoutError:
            active_task.cancel()
            await asyncio.gather(active_task, return_exceptions=True)


__all__ = [
    "ResearchRunner",
    "ResearchTaskManager",
    "TaskBusyError",
    "TaskFileLimitError",
    "TaskNotFoundError",
    "TaskStateError",
    "UnknownFileError",
]
