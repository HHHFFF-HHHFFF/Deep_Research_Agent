"""稳定 Web 接口使用的请求、响应与状态模型。"""

from datetime import datetime
from enum import Enum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class TaskStatus(str, Enum):
    """研究任务可以持久化的生命周期状态。"""

    WAITING = "waiting"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class TaskStage(str, Enum):
    """前端可展示的任务阶段。"""

    WAITING = "waiting"
    INITIALIZING = "initializing"
    RESEARCHING = "researching"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


TERMINAL_TASK_STATUSES = {
    TaskStatus.SUCCEEDED,
    TaskStatus.FAILED,
    TaskStatus.CANCELLED,
    TaskStatus.INTERRUPTED,
}


class TaskCreateRequest(BaseModel):
    """创建一次研究任务所需的最小输入。"""

    task: str = Field(min_length=1, max_length=4000)
    model_provider: Literal["qwen", "deepseek"]
    model_id: str = Field(min_length=1, max_length=200, pattern=r"^[\w./-]+$")
    file_ids: list[UUID] = Field(default_factory=list, max_length=5)

    @field_validator("task", "model_id")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        """清理文本输入，并拒绝只包含空白字符的值。"""
        normalized = value.strip()
        if not normalized:
            raise ValueError("字段不能为空")
        return normalized

    @field_validator("file_ids")
    @classmethod
    def reject_duplicate_files(cls, value: list[UUID]) -> list[UUID]:
        """同一任务不能重复引用同一个上传文件。"""
        if len(set(value)) != len(value):
            raise ValueError("不能重复选择同一个文件")
        return value


class UploadedFileResponse(BaseModel):
    """上传文件对外可见的安全元数据。"""

    id: str
    name: str
    size: int
    created_at: datetime


class TaskResponse(BaseModel):
    """任务详情与轮询共用的响应。"""

    id: str
    task: str
    model_provider: str
    model_id: str
    actual_model_name: str | None = None
    status: TaskStatus
    stage: TaskStage
    message: str
    error_message: str | None = None
    files: list[UploadedFileResponse] = Field(default_factory=list)
    rag_enabled: bool = False
    report_available: bool = False
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


class TaskListResponse(BaseModel):
    """最近任务列表。"""

    items: list[TaskResponse]


class HealthResponse(BaseModel):
    """服务健康状态。"""

    status: Literal["ok"] = "ok"
    database: Literal["ok"] = "ok"
    active_task_id: str | None = None


class ErrorDetail(BaseModel):
    """统一错误详情。"""

    code: str
    message: str


class ErrorResponse(BaseModel):
    """统一错误响应。"""

    error: ErrorDetail
