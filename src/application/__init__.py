"""研究应用数据模型的公共接口。"""

from .research_request import ResearchRequest, resolve_research_task
from .research_result import ResearchProgress, ResearchResult, ResearchStage

__all__ = [
    "ResearchProgress",
    "ResearchRequest",
    "ResearchResult",
    "ResearchStage",
    "resolve_research_task",
]
