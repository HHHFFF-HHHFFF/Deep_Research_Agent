"""A3 RAG 离线评测的指标与完整链路测试。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from src.document_retriever import RetrievedChunk
from src.evaluation import (
    RagEvaluationDataset,
    evaluate_rag_rankings,
    offline_hash_embeddings,
    render_rag_report_markdown,
    run_offline_rag_evaluation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _dataset_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": "手工排名测试",
        "retriever": {
            "chunk_size": 200,
            "chunk_overlap": 20,
            "k_values": [1, 3],
        },
        "thresholds": {"k": 3, "minimum_recall": 1.0, "minimum_mrr": 0.5},
        "documents": [
            {"path": "苹果.md", "content": "苹果证据：苹果含有膳食纤维。"},
            {"path": "香蕉.md", "content": "香蕉证据：香蕉成熟后通常呈黄色。"},
        ],
        "cases": [
            {
                "id": "fruit-001",
                "query": "苹果有什么营养特点？",
                "relevant": [{"source": "苹果.md", "contains": "苹果含有膳食纤维"}],
            }
        ],
    }


def _chunk(text: str, source: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        source=source,
        chunk_index=0,
        score=score,
        content_hash=f"hash-{source}",
    )


def test_rag_metrics_use_ranked_relevant_evidence() -> None:
    dataset = RagEvaluationDataset.model_validate(_dataset_payload())
    report = evaluate_rag_rankings(
        dataset,
        {
            "fruit-001": [
                _chunk("香蕉成熟后通常呈黄色。", "香蕉.md", 0.9),
                _chunk("苹果证据：苹果含有膳食纤维。", "苹果.md", 0.8),
            ]
        },
        indexed_chunks=2,
    )

    metrics = {metric.k: metric for metric in report.metrics}
    assert metrics[1].recall == 0.0
    assert metrics[1].hit_rate == 0.0
    assert metrics[3].recall == 1.0
    assert metrics[3].hit_rate == 1.0
    assert metrics[3].mrr == 0.5
    assert report.cases[0].first_relevant_rank == 2
    assert report.passed is True
    assert report.semantic_embedding_evaluated is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate_case", "用例编号不能重复"),
        ("missing_source", "引用了不存在的文档"),
        ("missing_anchor", "证据锚点不在文档"),
        ("unsafe_path", "只能使用文件名"),
    ],
)
def test_rag_dataset_rejects_invalid_annotations(
    mutation: str,
    message: str,
) -> None:
    payload = _dataset_payload()
    cases = payload["cases"]
    assert isinstance(cases, list)
    if mutation == "duplicate_case":
        cases.append(dict(cases[0]))
    elif mutation == "unsafe_path":
        documents = payload["documents"]
        assert isinstance(documents, list)
        documents[0]["path"] = "目录\\苹果.md"
    else:
        relevant = cases[0]["relevant"]
        if mutation == "missing_source":
            relevant[0]["source"] = "不存在.md"
        else:
            relevant[0]["contains"] = "不存在的证据"

    with pytest.raises(ValidationError, match=message):
        RagEvaluationDataset.model_validate(payload)


def test_offline_hash_embeddings_are_deterministic() -> None:
    first = offline_hash_embeddings(["FAISS 本地检索", "引用核验"])
    second = offline_hash_embeddings(["FAISS 本地检索", "引用核验"])

    assert np.array_equal(first, second)
    assert first.shape == (2, 768)
    assert np.count_nonzero(first) > 0


@pytest.mark.parametrize("text", ["", "   ", "……！！"])
def test_offline_hash_embeddings_reject_text_without_tokens(text: str) -> None:
    with pytest.raises(ValueError, match="没有可生成向量的文字"):
        offline_hash_embeddings([text])


def test_rag_dataset_rejects_query_without_tokens() -> None:
    payload = _dataset_payload()
    cases = payload["cases"]
    assert isinstance(cases, list)
    cases[0]["query"] = "……！！"

    with pytest.raises(ValidationError, match="问题没有可生成向量的文字"):
        RagEvaluationDataset.model_validate(payload)


def test_full_rag_evaluation_uses_real_faiss_without_online_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.model import model_manager

    async def fail_online_embedding(**_: object) -> None:
        raise AssertionError("离线评测不应访问模型 API")

    monkeypatch.setattr(model_manager, "aembedding", fail_online_embedding)
    dataset = RagEvaluationDataset.model_validate_json(
        (PROJECT_ROOT / "evals" / "rag_cases.json").read_text(encoding="utf-8")
    )

    report = asyncio.run(
        run_offline_rag_evaluation(dataset, index_dir=tmp_path / "faiss")
    )

    threshold = next(
        metric for metric in report.metrics if metric.k == report.thresholds.k
    )
    assert report.total_cases == 50
    assert report.indexed_chunks > len(dataset.documents)
    assert report.passed is True
    assert threshold.recall >= dataset.thresholds.minimum_recall
    assert threshold.mrr >= dataset.thresholds.minimum_mrr
    cases_by_id = {case.id: case for case in report.cases}
    assert cases_by_id["rag-004"].first_relevant_rank == 1
    assert cases_by_id["rag-004"].retrieved[0].source == "本地RAG.md"
    assert cases_by_id["rag-004"].retrieved[0].chunk_index == 2
    assert cases_by_id["rag-006"].first_relevant_rank == 1
    assert cases_by_id["rag-006"].retrieved[0].source == "Agent执行.md"
    assert cases_by_id["rag-006"].retrieved[0].chunk_index == 2
    assert cases_by_id["rag-015"].recall_at_k[4] == 1.0


def test_rag_evaluation_rejects_nonempty_index_directory(tmp_path: Path) -> None:
    dataset = RagEvaluationDataset.model_validate(_dataset_payload())
    index_dir = tmp_path / "faiss"
    index_dir.mkdir()
    (index_dir / "old-index.bin").write_bytes(b"stale")

    with pytest.raises(ValueError, match="必须使用空索引目录"):
        asyncio.run(run_offline_rag_evaluation(dataset, index_dir=index_dir))


def test_rag_preamble_is_validated_split_and_indexed(tmp_path: Path) -> None:
    payload = _dataset_payload()
    documents = payload["documents"]
    cases = payload["cases"]
    assert isinstance(documents, list)
    assert isinstance(cases, list)
    documents[0]["preamble"] = "前言唯一证据：离线评测必须复用真实 FAISS。"
    cases[0]["query"] = "离线评测需要复用什么索引？"
    cases[0]["relevant"][0] = {
        "source": "苹果.md",
        "contains": "离线评测必须复用真实 FAISS",
    }
    dataset = RagEvaluationDataset.model_validate(payload)

    report = asyncio.run(
        run_offline_rag_evaluation(dataset, index_dir=tmp_path / "faiss")
    )

    assert report.passed is True
    assert report.cases[0].first_relevant_rank is not None


def test_rag_report_escapes_untrusted_markdown() -> None:
    payload = _dataset_payload()
    payload["name"] = "数据集\n# 伪标题"
    cases = payload["cases"]
    assert isinstance(cases, list)
    cases[0]["query"] = "找不到的内容 ![图片](https://example.com/x.png)"
    dataset = RagEvaluationDataset.model_validate(payload)
    report = evaluate_rag_rankings(dataset, {"fruit-001": []})

    markdown = render_rag_report_markdown(report)

    assert markdown.startswith("# RAG 离线评测\n")
    assert "`数据集 # 伪标题`" in markdown
    assert "`找不到的内容 ![图片](https://example.com/x.png)`" in markdown
    assert "\n# 伪标题" not in markdown


def test_rag_report_is_json_serializable() -> None:
    dataset = RagEvaluationDataset.model_validate(_dataset_payload())
    report = evaluate_rag_rankings(dataset, {"fruit-001": []})

    payload = json.loads(report.model_dump_json())

    assert payload["passed"] is False
    assert payload["cases"][0]["missing_evidence"]
