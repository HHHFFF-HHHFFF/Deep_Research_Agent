"""研究运行过程与结果的数据模型。"""

from enum import Enum

from pydantic import BaseModel, Field


class ResearchStage(str, Enum):
    """前端需要关注的少量真实研究阶段。"""

    INITIALIZING = "initializing"
    RESEARCHING = "researching"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResearchProgress(BaseModel):
    """描述一次阶段变化，不使用没有依据的百分比。"""

    stage: ResearchStage
    message: str = Field(min_length=1)


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
