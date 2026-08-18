"""研究状态、进度事件与最终结果的数据契约测试。"""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from src.application import ProgressEvent, ResearchResult, ResearchStatus


def test_progress_event_normalizes_text_and_uses_utc_time() -> None:
    """进度事件应清理文本，并默认使用带时区的 UTC 时间。"""
    event = ProgressEvent(
        task_id="  research-001  ",
        status=ResearchStatus.PLANNING,
        message="  正在制定研究计划  ",
        progress=20,
    )

    assert event.task_id == "research-001"
    assert event.message == "正在制定研究计划"
    assert event.created_at.tzinfo is not None
    assert event.model_dump(mode="json")["status"] == "planning"


@pytest.mark.parametrize("progress", [-1, 101])
def test_progress_event_rejects_invalid_percentage(progress: int) -> None:
    """进度百分比必须处于零到一百之间。"""
    with pytest.raises(ValidationError):
        ProgressEvent(
            task_id="research-001",
            status=ResearchStatus.ANALYZING,
            message="正在分析材料",
            progress=progress,
        )


@pytest.mark.parametrize(
    ("status", "data"),
    [
        (ResearchStatus.COMPLETED, {"report": "# 研究报告"}),
        (ResearchStatus.FAILED, {"error": "模型调用失败"}),
        (ResearchStatus.CANCELLED, {}),
        (ResearchStatus.TIMED_OUT, {"error": "研究任务超时"}),
    ],
)
def test_research_result_accepts_all_terminal_states(
    status: ResearchStatus,
    data: dict[str, str],
) -> None:
    """完成、失败、取消和超时都应成为可区分的最终结果。"""
    result = ResearchResult(task_id="research-001", status=status, **data)

    assert result.status == status


def test_research_result_rejects_non_terminal_state() -> None:
    """运行中的状态不能伪装成最终研究结果。"""
    with pytest.raises(ValidationError, match="研究结果只能使用"):
        ResearchResult(
            task_id="research-001",
            status=ResearchStatus.COLLECTING,
        )


def test_research_result_validates_required_terminal_content() -> None:
    """完成、失败和超时结果应携带与状态匹配的必要信息。"""
    with pytest.raises(ValidationError, match="必须包含报告内容"):
        ResearchResult(
            task_id="research-001",
            status=ResearchStatus.COMPLETED,
        )

    with pytest.raises(ValidationError, match="必须包含错误说明"):
        ResearchResult(
            task_id="research-001",
            status=ResearchStatus.FAILED,
        )


def test_research_result_rejects_reversed_time_range() -> None:
    """任务结束时间不能早于开始时间。"""
    started_at = datetime.now(timezone.utc)

    with pytest.raises(ValidationError, match="结束时间不能早于开始时间"):
        ResearchResult(
            task_id="research-001",
            status=ResearchStatus.CANCELLED,
            started_at=started_at,
            finished_at=started_at - timedelta(seconds=1),
        )


def test_research_result_collections_are_not_shared() -> None:
    """不同结果实例之间不能共享事件、文件和指标集合。"""
    first = ResearchResult(
        task_id="research-001",
        status=ResearchStatus.CANCELLED,
    )
    second = ResearchResult(
        task_id="research-002",
        status=ResearchStatus.CANCELLED,
    )

    first.output_files.append("reports/research-001.md")
    first.metrics["duration_seconds"] = 1.5

    assert second.output_files == []
    assert second.metrics == {}
