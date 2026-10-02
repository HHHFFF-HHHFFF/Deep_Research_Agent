"""统一证据模型与引用核验的确定性离线测试。"""

from src.application import (
    CitationValidation,
    EvidenceSourceType,
    ResearchEvidence,
    local_evidence_id,
    normalize_web_url,
    validate_and_link_citations,
    web_evidence_from_sources,
)


def test_web_sources_are_deduplicated_and_unsafe_urls_are_rejected() -> None:
    evidence = web_evidence_from_sources(
        [
            {
                "title": "官方文档",
                "url": "HTTPS://Example.com/docs/",
                "description": "第一条说明",
            },
            {
                "title": "重复来源",
                "url": "https://example.com/docs",
                "description": "第二条说明",
            },
            {
                "title": "不安全来源",
                "url": "javascript:alert(1)",
            },
        ]
    )

    assert len(evidence) == 1
    assert evidence[0].source_type is EvidenceSourceType.WEB
    assert normalize_web_url(evidence[0].url or "") == "https://example.com/docs"


def test_report_citations_link_web_and_local_evidence() -> None:
    local_id = local_evidence_id("资料.pdf", 2, "content-hash")
    evidence = [
        ResearchEvidence(
            id="web-source",
            source_type=EvidenceSourceType.WEB,
            title="网页来源",
            url="https://example.com/article/",
        ),
        ResearchEvidence(
            id=local_id,
            source_type=EvidenceSourceType.LOCAL,
            title="资料.pdf · 片段 2",
            file_name="资料.pdf",
            chunk_index=2,
            excerpt="本地证据正文",
            relevance_score=0.91,
        ),
    ]
    report = (
        "网页结论[1](https://example.com/article)。\n\n"
        "本地结论[本地资料：资料.pdf#片段2]。"
    )

    linked_report, updated_evidence, validation = validate_and_link_citations(
        report,
        evidence,
    )

    assert validation == CitationValidation(
        passed=True,
        total_citations=2,
        valid_citations=2,
        invalid_citations=0,
        cited_evidence=2,
        total_evidence=2,
        coverage_rate=1.0,
        issues=[],
    )
    assert f"[本地资料：资料.pdf#片段2](#evidence-{local_id})" in linked_report
    assert updated_evidence[0].citation_labels == ["1"]
    assert updated_evidence[1].citation_labels == ["本地资料：资料.pdf#片段2"]


def test_unknown_and_unresolved_citations_are_reported() -> None:
    report = "未知网页[3](https://unknown.example/path)。\n未解析引用[4](#ref4)。"

    _, _, validation = validate_and_link_citations(report, [])

    assert validation.passed is False
    assert validation.total_citations == 2
    assert validation.valid_citations == 0
    assert validation.invalid_citations == 2
    assert {issue.reason for issue in validation.issues} == {
        "引用链接不在本次真实采集来源中",
        "引用目标无法安全核验",
    }


def test_report_without_citations_returns_clear_issue() -> None:
    _, _, validation = validate_and_link_citations("只有报告正文。", [])

    assert validation.passed is False
    assert validation.total_citations == 0
    assert validation.invalid_citations == 1
    assert validation.issues[0].reason == "报告没有包含可核验的网页引用或本地证据标记"


def test_unmatched_file_url_is_removed_from_report() -> None:
    report = "本地结论[5](file:///F:/secret/private.pdf)。"

    linked_report, _, validation = validate_and_link_citations(report, [])

    assert "file://" not in linked_report
    assert linked_report == "本地结论[5](#local-evidence)。"
    assert validation.passed is False
    assert validation.issues[0].target == "file:///F:/secret/private.pdf"
