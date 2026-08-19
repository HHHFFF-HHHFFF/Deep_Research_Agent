"""单进程、单活动任务的异步研究管理器。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from src.api.database import ResearchDatabase, StoredFileRecord, TaskRecord
from src.api.models import TaskCreateRequest, TaskStage, TaskStatus
from src.application import ResearchProgress, ResearchRequest, ResearchResult
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
        report_dir: Path,
        runner: ResearchRunner = run_research,
    ):
        self.database = database
        self.report_dir = report_dir.resolve()
        self.runner = runner
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
        """验证文件、持久化任务并启动后台执行。"""
        async with self._lock:
            if self._shutting_down:
                raise TaskStateError("服务正在关闭，暂时不能创建研究任务")
            if self._active_task is not None and not self._active_task.done():
                raise TaskBusyError("已有研究任务正在运行，请等待完成或先取消")

            file_ids = [str(file_id) for file_id in request.file_ids]
            files = self.database.get_files(file_ids)
            if len(files) != len(file_ids):
                raise UnknownFileError("请求中包含不存在的上传文件")

            task_id = str(uuid4())
            record = self.database.create_task(
                task_id=task_id,
                task=request.task.strip(),
                model_provider=request.model_provider,
                model_id=request.model_id,
                file_ids=file_ids,
            )
            self._cancel_event = asyncio.Event()
            self._active_task_id = task_id
            self._active_task = asyncio.create_task(
                self._execute(record, files, self._cancel_event),
                name=f"research-task-{task_id}",
            )
            return record

    async def cancel_task(self, task_id: str) -> TaskRecord:
        """向当前活动任务发出协作式取消信号。"""
        record = self.database.get_task(task_id)
        if record is None:
            raise TaskNotFoundError("研究任务不存在")
        if record.status not in {TaskStatus.WAITING, TaskStatus.RUNNING}:
            raise TaskStateError("当前任务状态不能取消")

        async with self._lock:
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

        try:
            self.database.mark_running(task_id)
            result = await self.runner(
                ResearchRequest(
                    task=record.task,
                    files=[file.stored_path for file in files],
                    model_provider=record.model_provider,
                    model_id=record.model_id,
                ),
                on_progress=on_progress,
                cancel_event=cancel_event,
            )
            report_path = await self._save_report(task_id, result.report)
            self.database.mark_succeeded(
                task_id,
                actual_model_name=result.model_name,
                report_path=str(report_path),
            )
        except ResearchCancelledError:
            if self._shutting_down:
                self.database.mark_interrupted(task_id)
            else:
                self.database.mark_cancelled(task_id)
        except ResearchRunError as error:
            self.database.mark_failed(task_id, str(error))
        except asyncio.CancelledError:
            self.database.mark_interrupted(task_id)
            raise
        except Exception:
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
    "TaskNotFoundError",
    "TaskStateError",
    "UnknownFileError",
]
