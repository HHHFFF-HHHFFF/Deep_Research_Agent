"""运行可复现的 RAG 离线回归评测。"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (
    RagEvaluationDataset,
    render_rag_report_markdown,
    run_offline_rag_evaluation,
)

DEFAULT_INPUT = PROJECT_ROOT / "evals" / "rag_cases.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "workdir" / "evaluations" / "rag_report.json"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用固定标注集、离线哈希向量和真实 FAISS 评测本地 RAG",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="评测集 JSON")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="机器可读 JSON 报告路径",
    )
    return parser.parse_args(argv)


def _load_dataset(path: Path) -> RagEvaluationDataset:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RagEvaluationDataset.model_validate(payload)


def _write_reports(output: Path, report_json: str, report_markdown: str) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report_json, encoding="utf-8")
    markdown_path = output.with_suffix(".md")
    markdown_path.write_text(report_markdown, encoding="utf-8")
    return markdown_path


async def _run(dataset: RagEvaluationDataset):
    with tempfile.TemporaryDirectory(prefix="deep-research-rag-eval-") as directory:
        return await run_offline_rag_evaluation(dataset, index_dir=directory)


def main(argv: list[str] | None = None) -> int:
    """运行评测；通过返回 0，门禁失败返回 1，输入错误返回 2。"""
    args = _parse_args(argv)
    try:
        dataset = _load_dataset(args.input)
        report = asyncio.run(_run(dataset))
        markdown_path = _write_reports(
            args.output,
            report.model_dump_json(indent=2),
            render_rag_report_markdown(report),
        )
    except ValidationError as error:
        error_count = len(error.errors(include_input=False, include_url=False))
        print(f"RAG 评测失败：输入结构不符合要求（{error_count} 处）", file=sys.stderr)
        return 2
    except json.JSONDecodeError as error:
        print(
            f"RAG 评测失败：JSON 格式无效（第 {error.lineno} 行，第 {error.colno} 列）",
            file=sys.stderr,
        )
        return 2
    except OSError:
        print("RAG 评测失败：无法读取输入或写入报告文件", file=sys.stderr)
        return 2
    except (RuntimeError, ValueError):
        print("RAG 评测失败：离线检索链路执行异常", file=sys.stderr)
        return 2

    threshold = next(
        metric for metric in report.metrics if metric.k == report.thresholds.k
    )
    print(f"RAG 离线评测：{'通过' if report.passed else '未通过'}")
    print(
        f"Recall@{threshold.k}={threshold.recall:.2%}，"
        f"MRR@{threshold.k}={threshold.mrr:.2%}，"
        f"HitRate@{threshold.k}={threshold.hit_rate:.2%}"
    )
    print(f"JSON 报告：{args.output}")
    print(f"Markdown 报告：{markdown_path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
