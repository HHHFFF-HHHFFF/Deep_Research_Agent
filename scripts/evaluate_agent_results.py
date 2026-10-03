"""评测不包含敏感原始数据的 Agent 执行结果快照。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (
    AgentEvaluationSuite,
    evaluate_agent_suite,
    render_agent_report_markdown,
)

DEFAULT_INPUT = PROJECT_ROOT / "evals" / "agent_results.example.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "workdir" / "evaluations" / "agent_report.json"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对 Agent 安全活动、证据和引用快照执行确定性离线评测",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="快照 JSON")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="机器可读 JSON 报告路径",
    )
    return parser.parse_args(argv)


def _write_reports(output: Path, report_json: str, report_markdown: str) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report_json, encoding="utf-8")
    markdown_path = output.with_suffix(".md")
    markdown_path.write_text(report_markdown, encoding="utf-8")
    return markdown_path


def main(argv: list[str] | None = None) -> int:
    """运行评测；通过返回 0，门禁失败返回 1，输入错误返回 2。"""
    args = _parse_args(argv)
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        suite = AgentEvaluationSuite.model_validate(payload)
        report = evaluate_agent_suite(suite)
        markdown_path = _write_reports(
            args.output,
            report.model_dump_json(indent=2),
            render_agent_report_markdown(report),
        )
    except ValidationError as error:
        error_count = len(error.errors(include_input=False, include_url=False))
        print(
            f"Agent 评测失败：输入结构不符合要求（{error_count} 处）", file=sys.stderr
        )
        return 2
    except json.JSONDecodeError as error:
        print(
            f"Agent 评测失败：JSON 格式无效（第 {error.lineno} 行，第 {error.colno} 列）",
            file=sys.stderr,
        )
        return 2
    except OSError:
        print("Agent 评测失败：无法读取输入或写入报告文件", file=sys.stderr)
        return 2

    print(f"Agent 执行结果离线评测：{'通过' if report.passed else '未通过'}")
    print(f"用例通过：{report.passed_cases}/{report.total_cases}")
    print(f"JSON 报告：{args.output}")
    print(f"Markdown 报告：{markdown_path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
