"""研究证据、引用核验结果与确定性校验逻辑。"""

from __future__ import annotations

import hashlib
import re
from enum import Enum
from urllib.parse import unquote, urlsplit, urlunsplit

from pydantic import BaseModel, Field, field_validator


class EvidenceSourceType(str, Enum):
    """研究证据的来源类型。"""

    WEB = "web"
    LOCAL = "local"


class ResearchEvidence(BaseModel):
    """一条可供用户核验的网页或本地文档证据。"""

    id: str = Field(min_length=1, max_length=64)
    source_type: EvidenceSourceType
    title: str = Field(min_length=1, max_length=300)
    url: str | None = Field(default=None, max_length=2048)
    file_name: str | None = Field(default=None, max_length=255)
    chunk_index: int | None = Field(default=None, ge=1)
    excerpt: str | None = Field(default=None, max_length=800)
    relevance_score: float | None = None
    citation_labels: list[str] = Field(default_factory=list)

    @field_validator("url")
    @classmethod
    def validate_public_url(cls, value: str | None) -> str | None:
        """只允许浏览器安全打开的 HTTP(S) 来源。"""
        if value is None:
            return None
        normalized = value.strip()
        scheme = urlsplit(normalized).scheme.lower()
        if scheme not in {"http", "https"}:
            raise ValueError("网页证据只允许 HTTP 或 HTTPS 地址")
        return normalized


class CitationIssue(BaseModel):
    """一条不会阻断报告生成的引用核验问题。"""

    label: str
    target: str | None = None
    reason: str


class CitationValidation(BaseModel):
    """报告引用与本次真实证据集合的确定性核验摘要。"""

    passed: bool
    total_citations: int = Field(ge=0)
    valid_citations: int = Field(ge=0)
    invalid_citations: int = Field(ge=0)
    cited_evidence: int = Field(ge=0)
    total_evidence: int = Field(ge=0)
    coverage_rate: float = Field(ge=0.0, le=1.0)
    issues: list[CitationIssue] = Field(default_factory=list)


_MARKDOWN_CITATION_PATTERN = re.compile(r"\[(\d+)\]\(([^)]+)\)")
_LOCAL_MARKER_PATTERN = re.compile(r"\[本地资料：([^\]#]+)#片段(\d+)\]")


def _compact_text(value: object, limit: int) -> str:
    normalized = re.sub(r"\s+", " ", str(value or "")).strip()
    return normalized[:limit]


def normalize_web_url(value: str) -> str | None:
    """规范化 HTTP(S) URL，以便确定性比较引用和采集来源。"""
    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        return None
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or not parsed.netloc:
        return None
    hostname = (parsed.hostname or "").lower()
    port = f":{parsed.port}" if parsed.port else ""
    netloc = f"{hostname}{port}"
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit((scheme, netloc, path, parsed.query, ""))


def web_evidence_from_sources(
    sources: list[dict[str, object]],
) -> list[ResearchEvidence]:
    """把网页搜索工具的安全来源摘要转换为统一证据。"""
    evidence_by_id: dict[str, ResearchEvidence] = {}
    for source in sources:
        raw_url = str(source.get("url", "")).strip()
        normalized_url = normalize_web_url(raw_url)
        if normalized_url is None:
            continue
        evidence_id = hashlib.sha256(normalized_url.encode("utf-8")).hexdigest()[:32]
        title = _compact_text(source.get("title"), 300) or normalized_url
        excerpt = (
            _compact_text(source.get("summary"), 800)
            or _compact_text(source.get("description"), 800)
            or None
        )
        evidence_by_id[evidence_id] = ResearchEvidence(
            id=evidence_id,
            source_type=EvidenceSourceType.WEB,
            title=title,
            url=raw_url,
            excerpt=excerpt,
        )
    return list(evidence_by_id.values())


def local_evidence_id(file_name: str, chunk_index: int, content_hash: str) -> str:
    """根据文档、片段与内容哈希生成稳定证据编号。"""
    identity = f"{file_name}:{chunk_index}:{content_hash}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]


def _local_anchor(evidence_id: str) -> str:
    return f"#evidence-{evidence_id}"


def _file_name_from_file_url(target: str) -> str:
    raw_path = unquote(target.removeprefix("file://"))
    return re.split(r"[/\\]", raw_path)[-1]


def validate_and_link_citations(
    report: str,
    evidence: list[ResearchEvidence],
) -> tuple[str, list[ResearchEvidence], CitationValidation]:
    """核验 Markdown 引用，并把本地证据标记转换为页面内安全链接。"""
    evidence_by_id = {item.id: item for item in evidence}
    web_by_url = {
        normalized: item.id
        for item in evidence
        if item.source_type is EvidenceSourceType.WEB and item.url
        if (normalized := normalize_web_url(item.url)) is not None
    }
    local_by_key = {
        (item.file_name, item.chunk_index): item.id
        for item in evidence
        if item.source_type is EvidenceSourceType.LOCAL
        and item.file_name
        and item.chunk_index is not None
    }
    labels_by_evidence: dict[str, set[str]] = {
        item.id: set(item.citation_labels) for item in evidence
    }
    issues: list[CitationIssue] = []
    seen_citations: set[tuple[str, str]] = set()
    valid_citations = 0

    def replace_local_marker(match: re.Match[str]) -> str:
        nonlocal valid_citations
        file_name = match.group(1).strip()
        chunk_index = int(match.group(2))
        label = f"本地资料：{file_name}#片段{chunk_index}"
        key = (label, f"{file_name}:{chunk_index}")
        if key in seen_citations:
            evidence_id = local_by_key.get((file_name, chunk_index))
            return (
                f"[{label}]({_local_anchor(evidence_id)})"
                if evidence_id
                else match.group(0)
            )
        seen_citations.add(key)
        evidence_id = local_by_key.get((file_name, chunk_index))
        if evidence_id is None:
            issues.append(
                CitationIssue(
                    label=label,
                    reason="本地引用未匹配到本次检索片段",
                )
            )
            return match.group(0)
        labels_by_evidence[evidence_id].add(label)
        valid_citations += 1
        return f"[{label}]({_local_anchor(evidence_id)})"

    linked_report = _LOCAL_MARKER_PATTERN.sub(replace_local_marker, report)

    def replace_markdown_citation(match: re.Match[str]) -> str:
        nonlocal valid_citations
        label = match.group(1)
        target = match.group(2).strip()
        key = (label, target)
        if key in seen_citations:
            return match.group(0)
        seen_citations.add(key)

        evidence_id: str | None = None
        replacement_target = target
        normalized_url = normalize_web_url(target)
        if normalized_url is not None:
            evidence_id = web_by_url.get(normalized_url)
        elif target.startswith("#evidence-"):
            candidate_id = target.removeprefix("#evidence-")
            if candidate_id in evidence_by_id:
                evidence_id = candidate_id
        elif target.startswith("file://"):
            # 无论能否匹配到证据，都不能把服务器本地路径返回浏览器。
            replacement_target = "#local-evidence"
            file_name = _file_name_from_file_url(target)
            matches = [
                item.id
                for item in evidence
                if item.source_type is EvidenceSourceType.LOCAL
                and item.file_name == file_name
            ]
            if matches:
                evidence_id = matches[0]
                replacement_target = _local_anchor(evidence_id)

        if evidence_id is None:
            reason = (
                "引用链接不在本次真实采集来源中"
                if normalized_url is not None
                else "引用目标无法安全核验"
            )
            issues.append(CitationIssue(label=label, target=target, reason=reason))
            return f"[{label}]({replacement_target})"

        labels_by_evidence[evidence_id].add(label)
        valid_citations += 1
        return f"[{label}]({replacement_target})"

    linked_report = _MARKDOWN_CITATION_PATTERN.sub(
        replace_markdown_citation,
        linked_report,
    )

    updated_evidence = [
        item.model_copy(update={"citation_labels": sorted(labels_by_evidence[item.id])})
        for item in evidence
    ]
    total_citations = len(seen_citations)
    invalid_citations = len(issues)
    cited_evidence = sum(bool(item.citation_labels) for item in updated_evidence)
    total_evidence = len(updated_evidence)
    coverage_rate = cited_evidence / total_evidence if total_evidence else 0.0
    if total_citations == 0:
        issues.append(
            CitationIssue(
                label="报告",
                reason="报告没有包含可核验的网页引用或本地证据标记",
            )
        )
        invalid_citations = len(issues)

    validation = CitationValidation(
        passed=total_citations > 0 and invalid_citations == 0,
        total_citations=total_citations,
        valid_citations=valid_citations,
        invalid_citations=invalid_citations,
        cited_evidence=cited_evidence,
        total_evidence=total_evidence,
        coverage_rate=coverage_rate,
        issues=issues,
    )
    return linked_report, updated_evidence, validation


__all__ = [
    "CitationIssue",
    "CitationValidation",
    "EvidenceSourceType",
    "ResearchEvidence",
    "local_evidence_id",
    "normalize_web_url",
    "validate_and_link_citations",
    "web_evidence_from_sources",
]
