"""使用 SQLAlchemy 2 和 SQLite 保存研究任务元数据。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import JSON, URL, DateTime, String, Text, create_engine, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from src.api.models import TERMINAL_TASK_STATUSES, TaskStage, TaskStatus


def utc_now() -> datetime:
    """返回带 UTC 时区的当前时间。"""
    return datetime.now(timezone.utc)


def _ensure_utc(value: datetime | None) -> datetime | None:
    """SQLite 丢失时区信息时按 UTC 恢复。"""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class Base(DeclarativeBase):
    """W2 元数据表的声明式基类。"""


class ResearchTaskRow(Base):
    """研究任务数据库记录。"""

    __tablename__ = "research_tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    task: Mapped[str] = mapped_column(Text)
    model_provider: Mapped[str] = mapped_column(String(32))
    model_id: Mapped[str] = mapped_column(String(200))
    actual_model_name: Mapped[str | None] = mapped_column(String(240), nullable=True)
    file_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(32), index=True)
    stage: Mapped[str] = mapped_column(String(32))
    message: Mapped[str] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    report_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class UploadedFileRow(Base):
    """上传文件数据库记录。"""

    __tablename__ = "uploaded_files"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    original_name: Mapped[str] = mapped_column(String(255))
    stored_path: Mapped[str] = mapped_column(Text, unique=True)
    size: Mapped[int]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


@dataclass(frozen=True)
class TaskRecord:
    """脱离数据库会话后仍可安全使用的任务记录。"""

    id: str
    task: str
    model_provider: str
    model_id: str
    actual_model_name: str | None
    file_ids: list[str]
    status: TaskStatus
    stage: TaskStage
    message: str
    error_message: str | None
    report_path: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


@dataclass(frozen=True)
class StoredFileRecord:
    """脱离数据库会话后仍可安全使用的文件记录。"""

    id: str
    original_name: str
    stored_path: str
    size: int
    created_at: datetime


def _task_record(row: ResearchTaskRow) -> TaskRecord:
    return TaskRecord(
        id=row.id,
        task=row.task,
        model_provider=row.model_provider,
        model_id=row.model_id,
        actual_model_name=row.actual_model_name,
        file_ids=list(row.file_ids or []),
        status=TaskStatus(row.status),
        stage=TaskStage(row.stage),
        message=row.message,
        error_message=row.error_message,
        report_path=row.report_path,
        created_at=_ensure_utc(row.created_at) or utc_now(),
        started_at=_ensure_utc(row.started_at),
        finished_at=_ensure_utc(row.finished_at),
    )


def _file_record(row: UploadedFileRow) -> StoredFileRecord:
    return StoredFileRecord(
        id=row.id,
        original_name=row.original_name,
        stored_path=row.stored_path,
        size=row.size,
        created_at=_ensure_utc(row.created_at) or utc_now(),
    )


class ResearchDatabase:
    """为单用户应用提供短会话 SQLite 操作。"""

    def __init__(self, database_path: Path):
        self.database_path = database_path.resolve()
        self.engine: Engine = create_engine(
            URL.create("sqlite", database=str(self.database_path)),
            connect_args={"check_same_thread": False},
        )
        self._session_factory = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
        )

    def initialize(self) -> None:
        """创建数据库目录与表。"""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        Base.metadata.create_all(self.engine)

    def ping(self) -> bool:
        """执行最小查询确认数据库连接可用。"""
        with self._session_factory() as session:
            session.execute(select(ResearchTaskRow.id).limit(1))
        return True

    def create_file(
        self,
        *,
        file_id: str,
        original_name: str,
        stored_path: str,
        size: int,
    ) -> StoredFileRecord:
        """保存一个上传文件的元数据。"""
        row = UploadedFileRow(
            id=file_id,
            original_name=original_name,
            stored_path=stored_path,
            size=size,
            created_at=utc_now(),
        )
        with self._session_factory.begin() as session:
            session.add(row)
        return _file_record(row)

    def get_files(self, file_ids: list[str]) -> list[StoredFileRecord]:
        """按请求顺序读取文件记录。"""
        if not file_ids:
            return []
        with self._session_factory() as session:
            rows = session.scalars(
                select(UploadedFileRow).where(UploadedFileRow.id.in_(file_ids))
            ).all()
        by_id = {row.id: _file_record(row) for row in rows}
        return [by_id[file_id] for file_id in file_ids if file_id in by_id]

    def create_task(
        self,
        *,
        task_id: str,
        task: str,
        model_provider: str,
        model_id: str,
        file_ids: list[str],
    ) -> TaskRecord:
        """创建等待执行的研究任务。"""
        row = ResearchTaskRow(
            id=task_id,
            task=task,
            model_provider=model_provider,
            model_id=model_id,
            file_ids=file_ids,
            status=TaskStatus.WAITING.value,
            stage=TaskStage.WAITING.value,
            message="研究任务正在等待执行",
            created_at=utc_now(),
        )
        with self._session_factory.begin() as session:
            session.add(row)
        return _task_record(row)

    def get_task(self, task_id: str) -> TaskRecord | None:
        """读取一个研究任务。"""
        with self._session_factory() as session:
            row = session.get(ResearchTaskRow, task_id)
            return _task_record(row) if row else None

    def list_tasks(self, limit: int) -> list[TaskRecord]:
        """按创建时间倒序返回最近任务。"""
        with self._session_factory() as session:
            rows = session.scalars(
                select(ResearchTaskRow)
                .order_by(ResearchTaskRow.created_at.desc())
                .limit(limit)
            ).all()
        return [_task_record(row) for row in rows]

    def mark_running(self, task_id: str) -> None:
        """把等待任务标记为正在运行。"""
        self._update_task(
            task_id,
            status=TaskStatus.RUNNING.value,
            stage=TaskStage.INITIALIZING.value,
            message="正在初始化研究组件",
            started_at=utc_now(),
        )

    def update_progress(self, task_id: str, stage: TaskStage, message: str) -> None:
        """更新非终态任务的真实阶段说明。"""
        with self._session_factory.begin() as session:
            row = session.get(ResearchTaskRow, task_id)
            if row is None or TaskStatus(row.status) in TERMINAL_TASK_STATUSES:
                return
            row.stage = stage.value
            row.message = message

    def mark_succeeded(
        self,
        task_id: str,
        *,
        actual_model_name: str,
        report_path: str,
    ) -> None:
        """保存成功状态与稳定报告路径。"""
        self._update_task(
            task_id,
            status=TaskStatus.SUCCEEDED.value,
            stage=TaskStage.COMPLETED.value,
            message="研究报告已经生成",
            actual_model_name=actual_model_name,
            report_path=report_path,
            error_message=None,
            finished_at=utc_now(),
        )

    def mark_failed(self, task_id: str, message: str) -> None:
        """保存对用户安全的失败原因。"""
        self._update_task(
            task_id,
            status=TaskStatus.FAILED.value,
            stage=TaskStage.FAILED.value,
            message="研究任务运行失败",
            error_message=message,
            finished_at=utc_now(),
        )

    def mark_cancelled(self, task_id: str) -> None:
        """保存用户主动取消状态。"""
        self._update_task(
            task_id,
            status=TaskStatus.CANCELLED.value,
            stage=TaskStage.CANCELLED.value,
            message="研究任务已取消",
            error_message=None,
            finished_at=utc_now(),
        )

    def mark_interrupted(self, task_id: str) -> None:
        """保存进程关闭导致的中断状态。"""
        self._update_task(
            task_id,
            status=TaskStatus.INTERRUPTED.value,
            stage=TaskStage.INTERRUPTED.value,
            message="研究任务因服务停止而中断",
            error_message=None,
            finished_at=utc_now(),
        )

    def mark_active_tasks_interrupted(self) -> int:
        """应用启动时收敛上次进程遗留的活动状态。"""
        with self._session_factory.begin() as session:
            result = session.execute(
                update(ResearchTaskRow)
                .where(
                    ResearchTaskRow.status.in_(
                        [TaskStatus.WAITING.value, TaskStatus.RUNNING.value]
                    )
                )
                .values(
                    status=TaskStatus.INTERRUPTED.value,
                    stage=TaskStage.INTERRUPTED.value,
                    message="研究任务因服务重启而中断",
                    error_message=None,
                    finished_at=utc_now(),
                )
            )
        return int(result.rowcount or 0)

    def _update_task(self, task_id: str, **values: object) -> None:
        """更新已存在的任务字段。"""
        with self._session_factory.begin() as session:
            row = session.get(ResearchTaskRow, task_id)
            if row is None:
                return
            for field_name, value in values.items():
                setattr(row, field_name, value)

    def dispose(self) -> None:
        """释放数据库连接池。"""
        self.engine.dispose()
