"""使用 SQLAlchemy 2 和 SQLite 保存研究任务元数据。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    JSON,
    URL,
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    delete,
    select,
    update,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from src.api.models import TERMINAL_TASK_STATUSES, TaskStage, TaskStatus
from src.application import (
    CitationIssue,
    CitationValidation,
    EvidenceSourceType,
    ResearchActivity,
    ResearchActivityStatus,
    ResearchActivityType,
    ResearchEvidence,
)


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


class TaskActivityRow(Base):
    """可供界面恢复的安全研究活动记录。"""

    __tablename__ = "task_activities"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    task_id: Mapped[str] = mapped_column(String(36), index=True)
    type: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32))
    title: Mapped[str] = mapped_column(String(200))
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    step_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tool_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class TaskEvidenceRow(Base):
    """任务采集到的网页或本地文档证据。"""

    __tablename__ = "task_evidence"

    row_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    task_id: Mapped[str] = mapped_column(String(36), index=True)
    evidence_id: Mapped[str] = mapped_column(String(64))
    source_type: Mapped[str] = mapped_column(String(16))
    title: Mapped[str] = mapped_column(String(300))
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    chunk_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    excerpt: Mapped[str | None] = mapped_column(Text, nullable=True)
    relevance_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    citation_labels: Mapped[list[str]] = mapped_column(JSON, default=list)


class CitationValidationRow(Base):
    """任务最终报告的引用核验摘要。"""

    __tablename__ = "citation_validations"

    task_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    passed: Mapped[bool] = mapped_column(Boolean)
    total_citations: Mapped[int] = mapped_column(Integer)
    valid_citations: Mapped[int] = mapped_column(Integer)
    invalid_citations: Mapped[int] = mapped_column(Integer)
    cited_evidence: Mapped[int] = mapped_column(Integer)
    total_evidence: Mapped[int] = mapped_column(Integer)
    coverage_rate: Mapped[float] = mapped_column(Float)
    issues: Mapped[list[dict[str, object]]] = mapped_column(JSON, default=list)


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


@dataclass(frozen=True)
class TaskActivityRecord:
    """脱离数据库会话后可安全返回的活动记录。"""

    id: str
    task_id: str
    type: ResearchActivityType
    status: ResearchActivityStatus
    title: str
    detail: str | None
    step_number: int | None
    tool_name: str | None
    duration_ms: int | None
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


def _activity_record(row: TaskActivityRow) -> TaskActivityRecord:
    return TaskActivityRecord(
        id=row.id,
        task_id=row.task_id,
        type=ResearchActivityType(row.type),
        status=ResearchActivityStatus(row.status),
        title=row.title,
        detail=row.detail,
        step_number=row.step_number,
        tool_name=row.tool_name,
        duration_ms=row.duration_ms,
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

    def upsert_activity(self, task_id: str, activity: ResearchActivity) -> None:
        """按活动编号新增或更新状态，避免开始和结束各占一行。"""
        with self._session_factory.begin() as session:
            if session.get(ResearchTaskRow, task_id) is None:
                return
            row = session.get(TaskActivityRow, activity.id)
            if row is None:
                row = TaskActivityRow(
                    id=activity.id,
                    task_id=task_id,
                    type=activity.type.value,
                    status=activity.status.value,
                    title=activity.title,
                    detail=activity.detail,
                    step_number=activity.step_number,
                    tool_name=activity.tool_name,
                    duration_ms=activity.duration_ms,
                    created_at=activity.created_at,
                )
                session.add(row)
                return
            if row.task_id != task_id:
                return
            row.status = activity.status.value
            row.title = activity.title
            row.detail = activity.detail
            row.duration_ms = activity.duration_ms

    def list_task_activities(self, task_id: str) -> list[TaskActivityRecord]:
        """按发生顺序返回指定任务的执行活动。"""
        with self._session_factory() as session:
            rows = session.scalars(
                select(TaskActivityRow)
                .where(TaskActivityRow.task_id == task_id)
                .order_by(TaskActivityRow.created_at.asc(), TaskActivityRow.id.asc())
            ).all()
        return [_activity_record(row) for row in rows]

    def finalize_running_activities(
        self,
        task_id: str,
        status: ResearchActivityStatus,
        detail: str,
    ) -> None:
        """任务终止时收敛仍处于运行中的活动，避免界面永久转圈。"""
        with self._session_factory.begin() as session:
            session.execute(
                update(TaskActivityRow)
                .where(
                    TaskActivityRow.task_id == task_id,
                    TaskActivityRow.status == ResearchActivityStatus.RUNNING.value,
                )
                .values(status=status.value, detail=detail)
            )

    def upsert_evidence(self, task_id: str, evidence: ResearchEvidence) -> None:
        """新增证据或更新最终引用标签。"""
        row_id = f"{task_id}:{evidence.id}"
        with self._session_factory.begin() as session:
            if session.get(ResearchTaskRow, task_id) is None:
                return
            row = session.get(TaskEvidenceRow, row_id)
            if row is None:
                row = TaskEvidenceRow(
                    row_id=row_id,
                    task_id=task_id,
                    evidence_id=evidence.id,
                    source_type=evidence.source_type.value,
                    title=evidence.title,
                    url=evidence.url,
                    file_name=evidence.file_name,
                    chunk_index=evidence.chunk_index,
                    excerpt=evidence.excerpt,
                    relevance_score=evidence.relevance_score,
                    citation_labels=evidence.citation_labels,
                )
                session.add(row)
                return
            row.title = evidence.title
            row.url = evidence.url
            row.file_name = evidence.file_name
            row.chunk_index = evidence.chunk_index
            row.excerpt = evidence.excerpt
            row.relevance_score = evidence.relevance_score
            row.citation_labels = evidence.citation_labels

    def list_task_evidence(self, task_id: str) -> list[ResearchEvidence]:
        """返回任务采集到的真实证据。"""
        with self._session_factory() as session:
            rows = session.scalars(
                select(TaskEvidenceRow)
                .where(TaskEvidenceRow.task_id == task_id)
                .order_by(
                    TaskEvidenceRow.source_type.asc(),
                    TaskEvidenceRow.title.asc(),
                )
            ).all()
        return [
            ResearchEvidence(
                id=row.evidence_id,
                source_type=EvidenceSourceType(row.source_type),
                title=row.title,
                url=row.url,
                file_name=row.file_name,
                chunk_index=row.chunk_index,
                excerpt=row.excerpt,
                relevance_score=row.relevance_score,
                citation_labels=list(row.citation_labels or []),
            )
            for row in rows
        ]

    def save_citation_validation(
        self,
        task_id: str,
        validation: CitationValidation,
    ) -> None:
        """保存或覆盖最终报告的引用核验摘要。"""
        with self._session_factory.begin() as session:
            if session.get(ResearchTaskRow, task_id) is None:
                return
            row = session.get(CitationValidationRow, task_id)
            values = validation.model_dump(mode="json")
            if row is None:
                session.add(CitationValidationRow(task_id=task_id, **values))
                return
            for field_name, value in values.items():
                setattr(row, field_name, value)

    def get_citation_validation(
        self,
        task_id: str,
    ) -> CitationValidation | None:
        """读取指定任务的引用核验结果。"""
        with self._session_factory() as session:
            row = session.get(CitationValidationRow, task_id)
            if row is None:
                return None
            return CitationValidation(
                passed=row.passed,
                total_citations=row.total_citations,
                valid_citations=row.valid_citations,
                invalid_citations=row.invalid_citations,
                cited_evidence=row.cited_evidence,
                total_evidence=row.total_evidence,
                coverage_rate=row.coverage_rate,
                issues=[
                    CitationIssue.model_validate(issue)
                    for issue in list(row.issues or [])
                ],
            )

    def get_task_orphan_files(self, task_id: str) -> list[StoredFileRecord]:
        """返回仅由指定任务引用、可随任务一同清理的上传文件。"""
        with self._session_factory() as session:
            task_row = session.get(ResearchTaskRow, task_id)
            if task_row is None:
                return []
            candidate_ids = set(task_row.file_ids or [])
            if not candidate_ids:
                return []
            other_file_ids: set[str] = set()
            for file_ids in session.scalars(
                select(ResearchTaskRow.file_ids).where(ResearchTaskRow.id != task_id)
            ):
                other_file_ids.update(file_ids or [])
            orphan_ids = candidate_ids - other_file_ids
            rows = session.scalars(
                select(UploadedFileRow).where(UploadedFileRow.id.in_(orphan_ids))
            ).all()
        return [_file_record(row) for row in rows]

    def delete_task(self, task_id: str, orphan_file_ids: list[str]) -> bool:
        """删除任务元数据及确认不再被引用的上传文件元数据。"""
        with self._session_factory.begin() as session:
            task_row = session.get(ResearchTaskRow, task_id)
            if task_row is None:
                return False
            session.execute(
                delete(TaskActivityRow).where(TaskActivityRow.task_id == task_id)
            )
            session.execute(
                delete(TaskEvidenceRow).where(TaskEvidenceRow.task_id == task_id)
            )
            session.execute(
                delete(CitationValidationRow).where(
                    CitationValidationRow.task_id == task_id
                )
            )
            session.delete(task_row)
            for file_id in orphan_file_ids:
                file_row = session.get(UploadedFileRow, file_id)
                if file_row is not None:
                    session.delete(file_row)
        return True

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
