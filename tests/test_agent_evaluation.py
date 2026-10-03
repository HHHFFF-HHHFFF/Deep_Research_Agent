"""A3 Agent 执行结果离线评测测试。"""

from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from src.evaluation import (
    AgentEvaluationSuite,
    RatioMetric,
    evaluate_agent_suite,
    render_agent_report_markdown,
)
from src.evaluation.agent_results import _sum_ratios


def _successful_case() -> dict[str, object]:
    return {
        "id": "agent-001",
        "description": "研究任务正常完成",
        "expected": {
            "status": "succeeded",
            "required_activity_types": ["planning", "tool"],
            "required_tools": ["web_searcher", "reporter"],
            "minimum_tool_success_rate": 1.0,
            "minimum_evidence": 1,
            "required_evidence_types": ["web"],
            "minimum_citations": 1,
            "maximum_invalid_citations": 0,
            "minimum_evidence_coverage": 1.0,
        },
        "snapshot": {
            "status": "succeeded",
            "report_available": True,
            "activities": [
                {
                    "id": "plan",
                    "type": "planning",
                    "status": "succeeded",
                    "title": "规划",
                },
                {
                    "id": "search",
                    "type": "tool",
                    "status": "succeeded",
                    "title": "搜索",
                    "tool_name": "web_searcher",
                },
                {
                    "id": "report",
                    "type": "tool",
                    "status": "succeeded",
                    "title": "报告",
                    "tool_name": "reporter",
                },
            ],
            "evidence": [
                {
                    "id": "source-1",
                    "source_type": "web",
                    "title": "来源",
                    "url": "https://example.com/source",
                    "citation_labels": ["1"],
                }
            ],
            "citation_validation": {
                "passed": True,
                "total_citations": 1,
                "valid_citations": 1,
                "invalid_citations": 0,
                "cited_evidence": 1,
                "total_evidence": 1,
                "coverage_rate": 1.0,
                "issues": [],
            },
        },
    }


def _suite(cases: list[dict[str, object]]) -> AgentEvaluationSuite:
    return AgentEvaluationSuite.model_validate(
        {"schema_version": 1, "name": "测试套件", "cases": cases}
    )


def test_agent_case_passes_all_observable_requirements() -> None:
    report = evaluate_agent_suite(_suite([_successful_case()]))

    assert report.passed is True
    assert report.case_pass_rate.rate == 1.0
    assert report.required_tool_coverage.rate == 1.0
    assert report.citation_traceability_rate.rate == 1.0
    assert report.evidence_coverage_rate.rate == 1.0
    assert report.semantic_quality_evaluated is False


def test_agent_case_reports_running_activity_missing_tool_and_bad_citation() -> None:
    case = _successful_case()
    snapshot = case["snapshot"]
    assert isinstance(snapshot, dict)
    activities = snapshot["activities"]
    assert isinstance(activities, list)
    activities[0]["status"] = "running"
    activities[:] = [
        activity
        for activity in activities
        if activity.get("tool_name") != "web_searcher"
    ]
    snapshot["citation_validation"]["valid_citations"] = 0
    snapshot["citation_validation"]["invalid_citations"] = 1
    snapshot["citation_validation"]["cited_evidence"] = 0
    snapshot["citation_validation"]["coverage_rate"] = 0.0
    snapshot["citation_validation"]["passed"] = False
    snapshot["citation_validation"]["issues"] = [
        {"label": "1", "reason": "引用目标无法安全核验"}
    ]
    snapshot["evidence"][0]["citation_labels"] = []

    result = evaluate_agent_suite(_suite([case])).cases[0]

    assert result.passed is False
    assert any("运行中" in issue for issue in result.issues)
    assert any("web_searcher" in issue for issue in result.issues)
    assert any("异常引用" in issue for issue in result.issues)
    assert any("证据引用覆盖率" in issue for issue in result.issues)


def test_expected_cancelled_case_can_pass_with_na_metrics() -> None:
    case = {
        "id": "cancel-001",
        "description": "用户取消任务",
        "expected": {
            "status": "cancelled",
            "require_report": False,
            "required_activity_types": [],
            "minimum_evidence": 0,
            "minimum_citations": 0,
        },
        "snapshot": {
            "status": "cancelled",
            "report_available": False,
            "activities": [],
            "evidence": [],
            "citation_validation": None,
        },
    }

    report = evaluate_agent_suite(_suite([case]))

    assert report.passed is True
    assert report.tool_success_rate.rate is None
    assert report.citation_traceability_rate.rate is None
    assert report.successful_task_completion_rate.rate is None


def test_zero_denominator_with_zero_threshold_remains_valid_na() -> None:
    case = {
        "id": "empty-citations",
        "description": "没有证据的预期终止任务",
        "expected": {
            "status": "cancelled",
            "require_report": False,
            "required_activity_types": [],
            "minimum_evidence": 0,
            "minimum_citations": 0,
            "maximum_invalid_citations": 1,
            "minimum_evidence_coverage": 0.0,
        },
        "snapshot": {
            "status": "cancelled",
            "report_available": False,
            "activities": [],
            "evidence": [],
            "citation_validation": {
                "passed": False,
                "total_citations": 0,
                "valid_citations": 0,
                "invalid_citations": 1,
                "cited_evidence": 0,
                "total_evidence": 0,
                "coverage_rate": 0.0,
                "issues": [{"label": "报告", "reason": "报告没有可核验引用"}],
            },
        },
    }

    report = evaluate_agent_suite(_suite([case]))

    assert report.passed is True
    assert report.citation_traceability_rate.rate is None
    assert report.evidence_coverage_rate.rate is None


def test_agent_metrics_aggregate_raw_numerators_and_denominators() -> None:
    first = _successful_case()
    second = deepcopy(first)
    second["id"] = "agent-002"
    second_snapshot = second["snapshot"]
    assert isinstance(second_snapshot, dict)
    second_activities = second_snapshot["activities"]
    assert isinstance(second_activities, list)
    second_activities[-1]["status"] = "failed"

    report = evaluate_agent_suite(_suite([first, second]))

    assert report.tool_success_rate.numerator == 3
    assert report.tool_success_rate.denominator == 4
    assert report.tool_success_rate.rate == 0.75
    assert report.passed is False


def test_invalid_child_ratio_cannot_be_hidden_by_aggregation() -> None:
    metric = _sum_ratios(
        [
            RatioMetric(numerator=2, denominator=1, rate=None),
            RatioMetric(numerator=0, denominator=10, rate=0.0),
        ]
    )

    assert metric.numerator == 2
    assert metric.denominator == 11
    assert metric.rate is None


def test_agent_case_rejects_inconsistent_citation_counts() -> None:
    case = _successful_case()
    snapshot = case["snapshot"]
    assert isinstance(snapshot, dict)
    validation = snapshot["citation_validation"]
    assert isinstance(validation, dict)
    validation["valid_citations"] = 2
    validation["cited_evidence"] = 2

    with pytest.raises(ValidationError, match="已引用证据数与证据标签不一致"):
        _suite([case])


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("duplicate_activity", "活动编号不能重复"),
        ("duplicate_evidence", "证据编号不能重复"),
        ("evidence_total", "证据总数与快照证据数量不一致"),
        ("coverage", "证据覆盖率与计数不一致"),
        ("issue_total", "异常引用数与问题明细数量不一致"),
        ("citation_total", "有效引用数与异常引用数之和不等于引用总数"),
        ("passed", "通过状态与引用计数不一致"),
    ],
)
def test_agent_snapshot_rejects_inconsistent_or_duplicate_data(
    target: str,
    message: str,
) -> None:
    case = _successful_case()
    snapshot = case["snapshot"]
    assert isinstance(snapshot, dict)
    activities = snapshot["activities"]
    evidence = snapshot["evidence"]
    validation = snapshot["citation_validation"]
    assert isinstance(activities, list)
    assert isinstance(evidence, list)
    assert isinstance(validation, dict)

    if target == "duplicate_activity":
        activities.append(deepcopy(activities[0]))
    elif target == "duplicate_evidence":
        evidence.append(deepcopy(evidence[0]))
        validation["total_evidence"] = 2
        validation["cited_evidence"] = 2
    elif target == "evidence_total":
        validation["total_evidence"] = 2
    elif target == "coverage":
        validation["coverage_rate"] = 0.5
    elif target == "issue_total":
        validation["issues"] = [{"label": "1", "reason": "异常"}]
    elif target == "citation_total":
        validation["total_citations"] = 2
    else:
        validation["passed"] = False

    with pytest.raises(ValidationError, match=message):
        _suite([case])


@pytest.mark.parametrize("field", ["thinking", "tool_args", "raw_output"])
def test_agent_snapshot_rejects_extra_sensitive_fields(field: str) -> None:
    case = _successful_case()
    snapshot = case["snapshot"]
    assert isinstance(snapshot, dict)
    activities = snapshot["activities"]
    assert isinstance(activities, list)
    activities[0][field] = "不应进入评测的敏感内容"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _suite([case])


def test_agent_report_escapes_untrusted_markdown() -> None:
    case = _successful_case()
    case["description"] = "失败说明\n# 伪标题 ![图片](https://example.com/x.png)"
    expected = case["expected"]
    assert isinstance(expected, dict)
    expected["status"] = "cancelled"
    suite = AgentEvaluationSuite.model_validate(
        {
            "schema_version": 1,
            "name": "恶意套件\n# 注入标题",
            "cases": [case],
        }
    )

    markdown = render_agent_report_markdown(evaluate_agent_suite(suite))

    assert markdown.startswith("# Agent 执行结果离线评测\n")
    assert "`恶意套件 # 注入标题`" in markdown
    assert "`失败说明 # 伪标题 ![图片](https://example.com/x.png)`" in markdown
    assert "\n# 注入标题" not in markdown


def test_agent_required_tool_rejects_markdown_syntax() -> None:
    case = _successful_case()
    expected = case["expected"]
    assert isinstance(expected, dict)
    expected["required_tools"] = ["![泄露](https://example.com/x)"]

    with pytest.raises(ValidationError, match="必需工具名称"):
        _suite([case])


def test_agent_suite_rejects_duplicate_case_ids() -> None:
    case = _successful_case()

    with pytest.raises(ValidationError, match="用例编号不能重复"):
        _suite([case, deepcopy(case)])
