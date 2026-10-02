"""研究应用数据模型的公共接口。"""

from .research_request import ResearchRequest, resolve_research_task
from .research_result import (
    ResearchActivity,
    ResearchActivityStatus,
    ResearchActivityType,
    ResearchProgress,
    ResearchResult,
    ResearchStage,
)

__all__ = [
    "CitationIssue",
    "CitationValidation",
    "EvidenceSourceType",
    "ResearchActivity",
    "ResearchActivityStatus",
    "ResearchActivityType",
    "ResearchEvidence",
    "ResearchProgress",
    "ResearchRequest",
    "ResearchResult",
    "ResearchStage",
    "local_evidence_id",
    "normalize_web_url",
    "resolve_research_task",
    "validate_and_link_citations",
    "web_evidence_from_sources",
]
from .evidence import (
    CitationIssue,
    CitationValidation,
    EvidenceSourceType,
    ResearchEvidence,
    local_evidence_id,
    normalize_web_url,
    validate_and_link_citations,
    web_evidence_from_sources,
)
