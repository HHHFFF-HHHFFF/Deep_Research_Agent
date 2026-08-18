"""研究任务状态、进度事件与最终结果的数据契约。"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class ResearchStatus(str, Enum):
    """描述研究任务从等待到结束的完整生命周期。"""

    PENDING = "pending"
    INITIALIZING = "initializing"
    PLANNING = "planning"
    COLLECTING = "collecting"
    ANALYZING = "analyzing"
    VERIFYING = "verifying"
    REPORTING = "reporting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


TERMINAL_RESEARCH_STATUSES = frozenset(
    {
        ResearchStatus.COMPLETED,
        ResearchStatus.FAILED,
        ResearchStatus.CANCELLED,
        ResearchStatus.TIMED_OUT,
    }
)


class ProgressEvent(BaseModel):
    """记录一次可供命令行、日志或流式接口消费的进度更新。"""

    task_id: str = Field(min_length=1, description="研究任务的唯一标识。")
    status: ResearchStatus = Field(description="事件发生时的研究任务状态。")
    message: str = Field(
        min_length=1,
        max_length=2000,
        description="面向用户的简短进度说明。",
    )
    progress: int = Field(
        ge=0,
        le=100,
        description="当前整体完成百分比。",
    )
    sequence: int = Field(
        default=0,
        ge=0,
        description="同一任务内用于稳定排序的事件序号。",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="使用 UTC 记录的事件创建时间。",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="供具体界面选择性展示的结构化补充信息。",
    )

    @field_validator("task_id", "message")
    @classmethod
    def normalize_required_text(cls, value: str) -> str:
        """清理必填文本，并拒绝只包含空白字符的内容。"""
        normalized = value.strip()
        if not normalized:
            raise ValueError("任务标识和进度说明不能为空")
        return normalized


class ResearchResult(BaseModel):
    """描述一次研究任务结束后返回给调用方的稳定结果。"""

    task_id: str = Field(min_length=1, description="研究任务的唯一标识。")
    status: ResearchStatus = Field(description="研究任务的最终状态。")
    report: str | None = Field(default=None, description="生成的 Markdown 研究报告。")
    error: str | None = Field(default=None, description="任务失败或超时时的错误说明。")
    started_at: datetime | None = Field(default=None, description="任务开始时间。")
    finished_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="任务结束时间。",
    )
    events: list[ProgressEvent] = Field(
        default_factory=list,
        description="本次任务产生的进度事件。",
    )
    output_files: list[str] = Field(
        default_factory=list,
        description="本次任务生成的报告或附件路径。",
    )
    metrics: dict[str, int | float] = Field(
        default_factory=dict,
        description="耗时、令牌数量等可选运行指标。",
    )

    @field_validator("task_id")
    @classmethod
    def normalize_task_id(cls, value: str) -> str:
        """清理任务标识，并拒绝空白标识。"""
        normalized = value.strip()
        if not normalized:
            raise ValueError("任务标识不能为空")
        return normalized

    @field_validator("report", "error")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        """把没有实际内容的可选文本统一转换为未提供。"""
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def validate_final_result(self) -> ResearchResult:
        """确保最终结果具有终态，并包含与终态匹配的必要信息。"""
        if self.status not in TERMINAL_RESEARCH_STATUSES:
            raise ValueError("研究结果只能使用已完成、失败、取消或超时状态")

        if self.status == ResearchStatus.COMPLETED and not (
            self.report or self.output_files
        ):
            raise ValueError("已完成的研究结果必须包含报告内容或输出文件")

        if self.status in {ResearchStatus.FAILED, ResearchStatus.TIMED_OUT} and not self.error:
            raise ValueError("失败或超时的研究结果必须包含错误说明")

        if self.started_at and self.finished_at < self.started_at:
            raise ValueError("任务结束时间不能早于开始时间")

        return self
