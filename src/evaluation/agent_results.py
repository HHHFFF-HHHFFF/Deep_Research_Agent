"""基于安全任务快照的 Agent 执行结果离线评测。"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.application import (
    EvidenceSourceType,
    ResearchActivityStatus,
    ResearchActivityType,
    ResearchEvidence,
)


class EvaluationTaskStatus(str, Enum):
    """允许进入离线结果评测的任务终态。"""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class _StrictInputModel(BaseModel):
    """拒绝评测输入中未声明的字段，避免误收隐藏推理或原始输出。"""

    model_config = ConfigDict(extra="forbid")


class AgentExpectedResult(_StrictInputModel):
    """人工声明的一条 Agent 用例验收条件。"""

    status: EvaluationTaskStatus = EvaluationTaskStatus.SUCCEEDED
    require_report: bool = True
    require_terminal_activities: bool = True
    required_activity_types: list[ResearchActivityType] = Field(
        default_factory=lambda: [
            ResearchActivityType.PLANNING,
            ResearchActivityType.TOOL,
        ]
    )
    required_tools: list[str] = Field(default_factory=list)
    minimum_tool_success_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    minimum_evidence: int = Field(default=1, ge=0)
    required_evidence_types: list[EvidenceSourceType] = Field(default_factory=list)
    minimum_citations: int = Field(default=1, ge=0)
    maximum_invalid_citations: int = Field(default=0, ge=0)
    minimum_evidence_coverage: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("required_tools")
    @classmethod
    def normalize_tools(cls, value: list[str]) -> list[str]:
        normalized = [tool.strip() for tool in value if tool.strip()]
        if any(
            not tool.replace("_", "").replace("-", "").replace(".", "").isalnum()
            or len(tool) > 120
            for tool in normalized
        ):
            raise ValueError("必需工具名称只能包含字母、数字、点、下划线和连字符")
        if len(normalized) != len(set(normalized)):
            raise ValueError("必需工具不能重复")
        return normalized

    @field_validator("required_activity_types", "required_evidence_types")
    @classmethod
    def reject_duplicate_enums(cls, value: list[Enum]) -> list[Enum]:
        if len(value) != len(set(value)):
            raise ValueError("必需类型不能重复")
        return value


class AgentActivitySnapshot(_StrictInputModel):
    """允许进入离线评测的安全活动字段。"""

    id: str = Field(min_length=1, max_length=80, pattern=r"^[\w.-]+$")
    type: ResearchActivityType
    status: ResearchActivityStatus
    title: str = Field(min_length=1, max_length=200)
    detail: str | None = Field(default=None, max_length=500)
    step_number: int | None = Field(default=None, ge=1)
    tool_name: str | None = Field(
        default=None,
        max_length=120,
        pattern=r"^[A-Za-z0-9_.-]+$",
    )
    duration_ms: int | None = Field(default=None, ge=0)
    created_at: datetime | None = None


class AgentEvidenceSnapshot(ResearchEvidence):
    """复用业务证据字段，同时拒绝任何额外敏感字段。"""

    model_config = ConfigDict(extra="forbid")


class AgentCitationIssueSnapshot(_StrictInputModel):
    """引用核验问题的安全摘要。"""

    label: str = Field(min_length=1, max_length=200)
    target: str | None = Field(default=None, max_length=2048)
    reason: str = Field(min_length=1, max_length=500)


class AgentCitationValidationSnapshot(_StrictInputModel):
    """不包含报告正文的引用核验计数。"""

    passed: bool
    total_citations: int = Field(ge=0)
    valid_citations: int = Field(ge=0)
    invalid_citations: int = Field(ge=0)
    cited_evidence: int = Field(ge=0)
    total_evidence: int = Field(ge=0)
    coverage_rate: float = Field(ge=0.0, le=1.0)
    issues: list[AgentCitationIssueSnapshot] = Field(default_factory=list)


class AgentResultSnapshot(_StrictInputModel):
    """不包含隐藏推理、工具参数和原始输出的任务快照。"""

    status: EvaluationTaskStatus
    report_available: bool
    activities: list[AgentActivitySnapshot] = Field(default_factory=list)
    evidence: list[AgentEvidenceSnapshot] = Field(default_factory=list)
    citation_validation: AgentCitationValidationSnapshot | None = None

    @model_validator(mode="after")
    def validate_consistency(self) -> AgentResultSnapshot:
        activity_ids = [activity.id for activity in self.activities]
        if len(activity_ids) != len(set(activity_ids)):
            raise ValueError("活动编号不能重复")
        evidence_ids = [evidence.id for evidence in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("证据编号不能重复")

        validation = self.citation_validation
        if validation is None:
            return self
        if validation.total_evidence != len(self.evidence):
            raise ValueError("引用核验的证据总数与快照证据数量不一致")
        observed_cited_evidence = sum(
            bool(evidence.citation_labels) for evidence in self.evidence
        )
        if validation.cited_evidence != observed_cited_evidence:
            raise ValueError("引用核验的已引用证据数与证据标签不一致")
        expected_coverage = (
            validation.cited_evidence / validation.total_evidence
            if validation.total_evidence
            else 0.0
        )
        if not math.isclose(
            validation.coverage_rate,
            expected_coverage,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("引用核验的证据覆盖率与计数不一致")
        if validation.invalid_citations != len(validation.issues):
            raise ValueError("异常引用数与问题明细数量不一致")
        if validation.total_citations > 0:
            if (
                validation.valid_citations + validation.invalid_citations
                != validation.total_citations
            ):
                raise ValueError("有效引用数与异常引用数之和不等于引用总数")
        elif validation.valid_citations != 0:
            raise ValueError("没有引用时有效引用数必须为零")
        expected_passed = (
            validation.total_citations > 0 and validation.invalid_citations == 0
        )
        if validation.passed is not expected_passed:
            raise ValueError("引用核验通过状态与引用计数不一致")
        return self


class AgentEvaluationCase(_StrictInputModel):
    """一条带预期条件的 Agent 结果评测用例。"""

    id: str = Field(min_length=1, max_length=80, pattern=r"^[\w.-]+$")
    description: str = Field(min_length=1, max_length=300)
    expected: AgentExpectedResult
    snapshot: AgentResultSnapshot


class AgentEvaluationSuite(_StrictInputModel):
    """可提交到版本库或由真实任务快照生成的评测集。"""

    schema_version: Literal[1] = 1
    name: str = Field(min_length=1, max_length=120)
    cases: list[AgentEvaluationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_duplicate_case_ids(self) -> AgentEvaluationSuite:
        case_ids = [case.id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Agent 评测用例编号不能重复")
        return self


class RatioMetric(BaseModel):
    """保留分子、分母和可空比例，避免零分母被伪装成 100%。"""

    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)
    rate: float | None = Field(default=None, ge=0.0, le=1.0)


class AgentCaseEvaluation(BaseModel):
    """单条 Agent 快照的确定性检查结果。"""

    id: str
    description: str
    passed: bool
    issues: list[str]
    status_matched: bool
    activity_terminal_rate: RatioMetric
    required_activity_coverage: RatioMetric
    required_tool_coverage: RatioMetric
    tool_success_rate: RatioMetric
    evidence_type_coverage: RatioMetric
    citation_traceability_rate: RatioMetric
    evidence_coverage_rate: RatioMetric


class AgentEvaluationReport(BaseModel):
    """Agent 执行结果的聚合评测报告。"""

    schema_version: Literal[1] = 1
    suite_name: str
    generated_at: datetime
    passed: bool
    total_cases: int = Field(ge=1)
    passed_cases: int = Field(ge=0)
    case_pass_rate: RatioMetric
    expected_status_match_rate: RatioMetric
    successful_task_completion_rate: RatioMetric
    activity_terminal_rate: RatioMetric
    required_activity_coverage: RatioMetric
    required_tool_coverage: RatioMetric
    tool_success_rate: RatioMetric
    citation_traceability_rate: RatioMetric
    evidence_coverage_rate: RatioMetric
    cases: list[AgentCaseEvaluation]
    semantic_quality_evaluated: Literal[False] = False
    limitations: list[str]


def _ratio(numerator: int, denominator: int) -> RatioMetric:
    rate = numerator / denominator if denominator and numerator <= denominator else None
    return RatioMetric(
        numerator=numerator,
        denominator=denominator,
        rate=rate,
    )


def _sum_ratios(metrics: list[RatioMetric]) -> RatioMetric:
    numerator = sum(metric.numerator for metric in metrics)
    denominator = sum(metric.denominator for metric in metrics)
    if any(metric.denominator > 0 and metric.rate is None for metric in metrics):
        return RatioMetric(
            numerator=numerator,
            denominator=denominator,
            rate=None,
        )
    return _ratio(numerator, denominator)


def evaluate_agent_case(case: AgentEvaluationCase) -> AgentCaseEvaluation:
    """检查一条 Agent 快照是否满足人工声明的结构化条件。"""
    expected = case.expected
    snapshot = case.snapshot
    issues: list[str] = []

    status_matched = snapshot.status is expected.status
    if not status_matched:
        issues.append(
            f"任务终态不符：期望 {expected.status.value}，实际 {snapshot.status.value}"
        )
    if expected.require_report and not snapshot.report_available:
        issues.append("任务没有可用报告")

    terminal_activities = sum(
        activity.status is not ResearchActivityStatus.RUNNING
        for activity in snapshot.activities
    )
    activity_terminal_rate = _ratio(terminal_activities, len(snapshot.activities))
    if (
        expected.require_terminal_activities
        and activity_terminal_rate.denominator > 0
        and activity_terminal_rate.numerator != activity_terminal_rate.denominator
    ):
        issues.append("仍有活动停留在运行中")

    observed_activity_types = {activity.type for activity in snapshot.activities}
    matched_activity_types = sum(
        activity_type in observed_activity_types
        for activity_type in expected.required_activity_types
    )
    required_activity_coverage = _ratio(
        matched_activity_types,
        len(expected.required_activity_types),
    )
    missing_activity_types = [
        activity_type.value
        for activity_type in expected.required_activity_types
        if activity_type not in observed_activity_types
    ]
    if missing_activity_types:
        issues.append(f"缺少必需活动：{'、'.join(missing_activity_types)}")

    tool_activities = [
        activity
        for activity in snapshot.activities
        if activity.type is ResearchActivityType.TOOL
    ]
    succeeded_tools = {
        activity.tool_name
        for activity in tool_activities
        if activity.status is ResearchActivityStatus.SUCCEEDED
        and activity.tool_name is not None
    }
    matched_tools = sum(tool in succeeded_tools for tool in expected.required_tools)
    required_tool_coverage = _ratio(matched_tools, len(expected.required_tools))
    missing_tools = [
        tool for tool in expected.required_tools if tool not in succeeded_tools
    ]
    if missing_tools:
        issues.append(f"缺少成功的必需工具：{'、'.join(missing_tools)}")

    succeeded_tool_calls = sum(
        activity.status is ResearchActivityStatus.SUCCEEDED
        for activity in tool_activities
    )
    tool_success_rate = _ratio(succeeded_tool_calls, len(tool_activities))
    if (
        expected.minimum_tool_success_rate is not None
        and expected.minimum_tool_success_rate > 0
        and (
            tool_success_rate.rate is None
            or tool_success_rate.rate < expected.minimum_tool_success_rate
        )
    ):
        issues.append(f"工具成功率低于 {expected.minimum_tool_success_rate:.0%}")

    if len(snapshot.evidence) < expected.minimum_evidence:
        issues.append(
            f"证据数量不足：至少 {expected.minimum_evidence} 条，"
            f"实际 {len(snapshot.evidence)} 条"
        )
    observed_evidence_types = {evidence.source_type for evidence in snapshot.evidence}
    matched_evidence_types = sum(
        source_type in observed_evidence_types
        for source_type in expected.required_evidence_types
    )
    evidence_type_coverage = _ratio(
        matched_evidence_types,
        len(expected.required_evidence_types),
    )
    missing_evidence_types = [
        source_type.value
        for source_type in expected.required_evidence_types
        if source_type not in observed_evidence_types
    ]
    if missing_evidence_types:
        issues.append(f"缺少证据类型：{'、'.join(missing_evidence_types)}")

    validation = snapshot.citation_validation
    if validation is None:
        citation_traceability_rate = _ratio(0, 0)
        evidence_coverage_rate = _ratio(0, 0)
        if expected.minimum_citations > 0 or expected.minimum_evidence_coverage > 0:
            issues.append("缺少引用核验结果")
    else:
        citation_traceability_rate = _ratio(
            validation.valid_citations,
            validation.total_citations,
        )
        evidence_coverage_rate = _ratio(
            validation.cited_evidence,
            validation.total_evidence,
        )
        if validation.total_citations < expected.minimum_citations:
            issues.append(
                f"引用数量不足：至少 {expected.minimum_citations} 条，"
                f"实际 {validation.total_citations} 条"
            )
        if validation.invalid_citations > expected.maximum_invalid_citations:
            issues.append(
                f"异常引用过多：最多 {expected.maximum_invalid_citations} 条，"
                f"实际 {validation.invalid_citations} 条"
            )
        if expected.minimum_evidence_coverage > 0 and (
            evidence_coverage_rate.rate is None
            or evidence_coverage_rate.rate < expected.minimum_evidence_coverage
        ):
            issues.append(
                f"证据引用覆盖率低于 {expected.minimum_evidence_coverage:.0%}"
            )

    return AgentCaseEvaluation(
        id=case.id,
        description=case.description,
        passed=not issues,
        issues=issues,
        status_matched=status_matched,
        activity_terminal_rate=activity_terminal_rate,
        required_activity_coverage=required_activity_coverage,
        required_tool_coverage=required_tool_coverage,
        tool_success_rate=tool_success_rate,
        evidence_type_coverage=evidence_type_coverage,
        citation_traceability_rate=citation_traceability_rate,
        evidence_coverage_rate=evidence_coverage_rate,
    )


def evaluate_agent_suite(suite: AgentEvaluationSuite) -> AgentEvaluationReport:
    """聚合多条 Agent 快照，保留所有指标分子与分母。"""
    results = [evaluate_agent_case(case) for case in suite.cases]
    passed_cases = sum(result.passed for result in results)
    successful_expected_cases = [
        case
        for case in suite.cases
        if case.expected.status is EvaluationTaskStatus.SUCCEEDED
    ]
    successful_actual_cases = sum(
        case.snapshot.status is EvaluationTaskStatus.SUCCEEDED
        for case in successful_expected_cases
    )
    return AgentEvaluationReport(
        suite_name=suite.name,
        generated_at=datetime.now(timezone.utc),
        passed=passed_cases == len(results),
        total_cases=len(results),
        passed_cases=passed_cases,
        case_pass_rate=_ratio(passed_cases, len(results)),
        expected_status_match_rate=_ratio(
            sum(result.status_matched for result in results),
            len(results),
        ),
        successful_task_completion_rate=_ratio(
            successful_actual_cases,
            len(successful_expected_cases),
        ),
        activity_terminal_rate=_sum_ratios(
            [result.activity_terminal_rate for result in results]
        ),
        required_activity_coverage=_sum_ratios(
            [result.required_activity_coverage for result in results]
        ),
        required_tool_coverage=_sum_ratios(
            [result.required_tool_coverage for result in results]
        ),
        tool_success_rate=_sum_ratios([result.tool_success_rate for result in results]),
        citation_traceability_rate=_sum_ratios(
            [result.citation_traceability_rate for result in results]
        ),
        evidence_coverage_rate=_sum_ratios(
            [result.evidence_coverage_rate for result in results]
        ),
        cases=results,
        limitations=[
            "只评估任务终态、活动收敛、工具、证据和引用等可观测结构。",
            "不评估隐藏推理、工具选择最优性、事实正确性或正文与证据的语义支持关系。",
        ],
    )


def _format_metric(metric: RatioMetric) -> str:
    if metric.rate is None:
        return f"{metric.numerator}/{metric.denominator}（N/A）"
    return f"{metric.numerator}/{metric.denominator}（{metric.rate:.2%}）"


def _markdown_code(value: str) -> str:
    normalized = " ".join(value.split()).replace("`", "'")
    return f"`{normalized}`"


def render_agent_report_markdown(report: AgentEvaluationReport) -> str:
    """把 Agent 评测结果渲染为中文 Markdown。"""
    status = "通过" if report.passed else "未通过"
    rows = [
        ("用例通过率", report.case_pass_rate),
        ("预期终态匹配率", report.expected_status_match_rate),
        ("成功任务完成率", report.successful_task_completion_rate),
        ("活动终态率", report.activity_terminal_rate),
        ("必需活动覆盖率", report.required_activity_coverage),
        ("必需工具覆盖率", report.required_tool_coverage),
        ("工具成功率", report.tool_success_rate),
        ("引用可追溯率", report.citation_traceability_rate),
        ("证据引用覆盖率", report.evidence_coverage_rate),
    ]
    lines = [
        "# Agent 执行结果离线评测",
        "",
        f"- 评测集：{_markdown_code(report.suite_name)}",
        f"- 总体门禁：**{status}**",
        f"- 用例：{report.passed_cases}/{report.total_cases} 通过",
        "",
        "## 指标",
        "",
    ]
    lines.extend(f"- {label}：{_format_metric(metric)}" for label, metric in rows)
    failed_cases = [case for case in report.cases if not case.passed]
    lines.extend(["", "## 失败用例", ""])
    if not failed_cases:
        lines.append("没有失败用例。")
    else:
        for case in failed_cases:
            lines.append(f"- `{case.id}` {_markdown_code(case.description)}")
            lines.extend(f"  - {_markdown_code(issue)}" for issue in case.issues)
    lines.extend(["", "## 能力边界", ""])
    lines.extend(f"- {limitation}" for limitation in report.limitations)
    return "\n".join(lines) + "\n"


__all__ = [
    "AgentActivitySnapshot",
    "AgentCitationValidationSnapshot",
    "AgentEvaluationCase",
    "AgentEvaluationReport",
    "AgentEvaluationSuite",
    "AgentEvidenceSnapshot",
    "AgentExpectedResult",
    "AgentResultSnapshot",
    "EvaluationTaskStatus",
    "RatioMetric",
    "evaluate_agent_case",
    "evaluate_agent_suite",
    "render_agent_report_markdown",
]
