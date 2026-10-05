"""稳定 Web 接口使用的请求、响应与状态模型。"""

from datetime import datetime
from enum import Enum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from src.application import (
    CitationValidation,
    ResearchActivityStatus,
    ResearchActivityType,
    ResearchEvidence,
)


class TaskStatus(str, Enum):
    """研究任务可以持久化的生命周期状态。"""

    AWAITING_CONFIRMATION = "awaiting_confirmation"
    WAITING = "waiting"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class TaskStage(str, Enum):
    """前端可展示的任务阶段。"""

    AWAITING_CONFIRMATION = "awaiting_confirmation"
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


class TaskPlanConfirmRequest(BaseModel):
    """用户确认或修改后的研究计划。"""

    steps: list[str] = Field(min_length=2, max_length=8)

    @field_validator("steps")
    @classmethod
    def normalize_steps(cls, value: list[str]) -> list[str]:
        """清理计划步骤，并限制单步长度与重复内容。"""
        normalized_steps: list[str] = []
        for step in value:
            normalized = step.strip()
            if len(normalized) < 3:
                raise ValueError("每个计划步骤至少需要 3 个字符")
            if len(normalized) > 300:
                raise ValueError("每个计划步骤不能超过 300 个字符")
            if normalized in normalized_steps:
                raise ValueError("研究计划不能包含重复步骤")
            normalized_steps.append(normalized)
        return normalized_steps


def build_default_research_plan(*, has_files: bool) -> list[str]:
    """根据是否包含本地资料生成无需额外模型调用的默认计划。"""
    steps = ["拆解研究主题，明确核心问题、研究范围和判断标准"]
    if has_files:
        steps.append("检索上传资料，提取与研究主题相关的本地证据")
    steps.extend(
        [
            "检索公开网页资料，记录可核验的来源与关键信息",
            "交叉核对不同来源，处理冲突信息并归纳主要发现",
            "按照证据组织结论，生成带来源引用的中文研究报告",
        ]
    )
    return steps


class UploadedFileResponse(BaseModel):
    """上传文件对外可见的安全元数据。"""

    id: str
    name: str
    size: int
    created_at: datetime


class TaskActivityResponse(BaseModel):
    """一次可安全展示给用户的智能体执行活动。"""

    id: str
    type: ResearchActivityType
    status: ResearchActivityStatus
    title: str
    detail: str | None = None
    step_number: int | None = None
    tool_name: str | None = None
    duration_ms: int | None = None
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
    research_plan: list[str] = Field(default_factory=list)
    plan_confirmed: bool = False
    files: list[UploadedFileResponse] = Field(default_factory=list)
    activities: list[TaskActivityResponse] = Field(default_factory=list)
    evidence: list[ResearchEvidence] = Field(default_factory=list)
    citation_validation: CitationValidation | None = None
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
