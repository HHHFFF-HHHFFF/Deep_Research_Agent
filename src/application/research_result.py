"""研究运行过程、可审阅活动与结果的数据模型。"""

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class ResearchStage(str, Enum):
    """前端需要关注的少量真实研究阶段。"""

    INITIALIZING = "initializing"
    RESEARCHING = "researching"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResearchActivityType(str, Enum):
    """用户可以审阅的研究活动类型。"""

    PLANNING = "planning"
    TOOL = "tool"
    RETRIEVAL = "retrieval"


class ResearchActivityStatus(str, Enum):
    """研究活动的执行状态。"""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResearchActivity(BaseModel):
    """不包含模型隐藏推理和工具原始输入输出的安全活动摘要。"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    type: ResearchActivityType
    status: ResearchActivityStatus
    title: str = Field(min_length=1, max_length=200)
    detail: str | None = Field(default=None, max_length=500)
    step_number: int | None = Field(default=None, ge=1)
    tool_name: str | None = Field(default=None, max_length=120)
    duration_ms: int | None = Field(default=None, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ResearchProgress(BaseModel):
    """描述一次阶段变化，不使用没有依据的百分比。"""

    stage: ResearchStage
    message: str = Field(min_length=1)
    activity: ResearchActivity | None = None


class ResearchResult(BaseModel):
    """一次成功研究返回给命令行或 API 的稳定结果。"""

    task: str
    report: str = Field(min_length=1)
    model_name: str
    session_id: str
    files: list[str] = Field(default_factory=list)
    report_path: str | None = Field(
        default=None,
        description="研究工具实际生成的 Markdown 报告路径。",
    )
