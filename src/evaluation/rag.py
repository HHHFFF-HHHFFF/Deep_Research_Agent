"""可复现的本地 RAG 离线回归评测。"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.document_retriever import (
    LocalDocumentRetriever,
    RetrievedChunk,
    split_document_text,
)
from src.environment.faiss.service import FaissService


class RagRetrieverConfig(BaseModel):
    """评测时复用的文档切分与检索参数。"""

    model_config = ConfigDict(extra="forbid")

    chunk_size: int = Field(default=500, ge=100, le=5000)
    chunk_overlap: int = Field(default=80, ge=0)
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 4], min_length=1)

    @field_validator("k_values")
    @classmethod
    def normalize_k_values(cls, value: list[int]) -> list[int]:
        normalized = sorted(set(value))
        if not normalized or normalized[0] < 1 or normalized[-1] > 20:
            raise ValueError("K 必须位于 1 到 20 之间")
        return normalized

    @model_validator(mode="after")
    def validate_overlap(self) -> RagRetrieverConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("片段重叠必须小于片段长度")
        return self


class RagThresholds(BaseModel):
    """RAG 回归门禁使用的最低指标。"""

    model_config = ConfigDict(extra="forbid")

    k: int = Field(default=4, ge=1, le=20)
    minimum_recall: float = Field(default=0.8, ge=0.0, le=1.0)
    minimum_mrr: float = Field(default=0.7, ge=0.0, le=1.0)


class RagEvaluationDocument(BaseModel):
    """评测集内嵌的一份无外部依赖文档。"""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=255)
    preamble: str = Field(default="", max_length=50_000)
    content: str = Field(min_length=1, max_length=200_000)

    @property
    def evaluation_content(self) -> str:
        """返回参与切分和索引的完整评测正文。"""
        return "\n\n".join(part for part in (self.preamble, self.content) if part)

    @field_validator("path")
    @classmethod
    def validate_file_name(cls, value: str) -> str:
        normalized = value.strip()
        if (
            not normalized
            or Path(normalized).name != normalized
            or any(separator in normalized for separator in ("/", "\\"))
            or normalized in {".", ".."}
        ):
            raise ValueError("评测文档只能使用文件名，不能包含目录")
        return normalized

    @field_validator("content")
    @classmethod
    def normalize_content(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("评测文档内容不能为空")
        return normalized

    @field_validator("preamble")
    @classmethod
    def normalize_preamble(cls, value: str) -> str:
        return value.strip()


class RagRelevantEvidence(BaseModel):
    """人工标注的一条相关证据。"""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1, max_length=255)
    contains: str = Field(min_length=1, max_length=500)

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or any(separator in normalized for separator in ("/", "\\")):
            raise ValueError("相关证据来源只能使用文件名")
        return normalized

    @field_validator("contains")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("相关证据字段不能为空")
        return normalized


class RagEvaluationCase(BaseModel):
    """一个查询及其人工标注的相关证据。"""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=80, pattern=r"^[\w.-]+$")
    query: str = Field(min_length=1, max_length=1000)
    relevant: list[RagRelevantEvidence] = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("评测问题不能为空")
        return normalized


class RagEvaluationDataset(BaseModel):
    """版本化、可自校验的 RAG 离线评测集。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    name: str = Field(min_length=1, max_length=120)
    retriever: RagRetrieverConfig = Field(default_factory=RagRetrieverConfig)
    thresholds: RagThresholds = Field(default_factory=RagThresholds)
    documents: list[RagEvaluationDocument] = Field(min_length=1)
    cases: list[RagEvaluationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_dataset(self) -> RagEvaluationDataset:
        document_names = [document.path.casefold() for document in self.documents]
        if len(document_names) != len(set(document_names)):
            raise ValueError("评测文档文件名不能重复")
        content_hashes = [
            hashlib.sha256(document.evaluation_content.encode("utf-8")).hexdigest()
            for document in self.documents
        ]
        if len(content_hashes) != len(set(content_hashes)):
            raise ValueError("评测文档内容不能完全重复")

        case_ids = [case.id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("评测用例编号不能重复")

        documents = {document.path.casefold(): document for document in self.documents}
        for document in self.documents:
            if len(document.evaluation_content) > 200_000:
                raise ValueError(f"评测文档 {document.path} 的完整内容超过限制")
            if not _offline_tokens(document.evaluation_content):
                raise ValueError(f"评测文档 {document.path} 没有可生成向量的文字")
        for case in self.cases:
            if not _offline_tokens(case.query):
                raise ValueError(f"用例 {case.id} 的问题没有可生成向量的文字")
            evidence_keys: set[tuple[str, str]] = set()
            for evidence in case.relevant:
                source_key = evidence.source.casefold()
                evidence_key = (source_key, evidence.contains)
                if evidence_key in evidence_keys:
                    raise ValueError(f"用例 {case.id} 包含重复的相关证据")
                evidence_keys.add(evidence_key)

                relevant_document = documents.get(source_key)
                if relevant_document is None:
                    raise ValueError(
                        f"用例 {case.id} 引用了不存在的文档 {evidence.source}"
                    )
                if evidence.contains not in relevant_document.evaluation_content:
                    raise ValueError(
                        f"用例 {case.id} 的证据锚点不在文档 {evidence.source} 中"
                    )
                chunks = split_document_text(
                    relevant_document.evaluation_content,
                    chunk_size=self.retriever.chunk_size,
                    chunk_overlap=self.retriever.chunk_overlap,
                )
                if not any(evidence.contains in chunk for chunk in chunks):
                    raise ValueError(
                        f"用例 {case.id} 的证据锚点跨越切分边界，无法稳定评测"
                    )
        return self


class RagRetrievedItem(BaseModel):
    """评测报告中保留的安全检索摘要。"""

    source: str
    chunk_index: int = Field(ge=1)
    score: float
    excerpt: str = Field(max_length=240)


class RagMetricsAtK(BaseModel):
    """指定 K 值的宏平均指标。"""

    k: int
    recall: float = Field(ge=0.0, le=1.0)
    hit_rate: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)


class RagCaseEvaluation(BaseModel):
    """单条查询的排名与命中详情。"""

    id: str
    query: str
    passed: bool
    first_relevant_rank: int | None
    recall_at_k: dict[int, float]
    reciprocal_rank_at_k: dict[int, float]
    missing_evidence: list[str]
    retrieved: list[RagRetrievedItem]


class RagEvaluationReport(BaseModel):
    """一次可保存、可比较的 RAG 离线评测报告。"""

    schema_version: Literal[1] = 1
    dataset_name: str
    generated_at: datetime
    indexed_chunks: int = Field(ge=0)
    total_cases: int = Field(ge=1)
    metrics: list[RagMetricsAtK]
    thresholds: RagThresholds
    passed: bool
    cases: list[RagCaseEvaluation]
    semantic_embedding_evaluated: Literal[False] = False
    limitations: list[str]


def _offline_tokens(text: str) -> list[tuple[str, float]]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens: list[tuple[str, float]] = []
    chinese = "".join(re.findall(r"[\u3400-\u9fff]", normalized))
    for width, weight in ((1, 0.25), (2, 1.0), (3, 1.25)):
        for index in range(max(len(chinese) - width + 1, 0)):
            tokens.append((f"zh:{chinese[index : index + width]}", weight))
    tokens.extend(
        (f"word:{word}", 1.0)
        for word in re.findall(r"[a-z0-9][a-z0-9_+.-]*", normalized)
    )
    return tokens


def offline_hash_embeddings(texts: Sequence[str], dimensions: int = 768) -> np.ndarray:
    """生成跨进程稳定的词面哈希向量，禁止访问在线 Embedding。"""
    vectors: np.ndarray = np.zeros((len(texts), dimensions), dtype=np.float32)
    for row, text in enumerate(texts):
        tokens = _offline_tokens(text)
        if not tokens:
            raise ValueError("离线评测文本没有可生成向量的文字")
        for token, weight in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            column = int.from_bytes(digest, "big") % dimensions
            vectors[row, column] += weight
    return vectors


class OfflineHashFaissService(FaissService):
    """仅替换向量生成、其余行为复用生产 FAISS 服务。"""

    def __init__(self, base_dir: str | Path, *, dimensions: int = 768) -> None:
        self._offline_dimensions = dimensions
        super().__init__(base_dir=base_dir)

    async def _get_embedding_dimension(self) -> int:
        self._embedding_dimension = self._offline_dimensions
        return self._offline_dimensions

    async def _get_embeddings(self, texts: list[str]) -> np.ndarray:
        return offline_hash_embeddings(texts, self._offline_dimensions)


def _matches(chunk: RetrievedChunk, evidence: RagRelevantEvidence) -> bool:
    return (
        chunk.source.casefold() == evidence.source.casefold()
        and evidence.contains in chunk.text
    )


def _safe_excerpt(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()[:240]


def evaluate_rag_rankings(
    dataset: RagEvaluationDataset,
    rankings: Mapping[str, Sequence[RetrievedChunk]],
    *,
    indexed_chunks: int = 0,
) -> RagEvaluationReport:
    """根据人工标注计算 Recall@K、HitRate@K 与 MRR@K。"""
    k_values = sorted({*dataset.retriever.k_values, dataset.thresholds.k})
    per_k_recalls: dict[int, list[float]] = {k: [] for k in k_values}
    per_k_hits: dict[int, list[float]] = {k: [] for k in k_values}
    per_k_rr: dict[int, list[float]] = {k: [] for k in k_values}
    case_results: list[RagCaseEvaluation] = []

    for case in dataset.cases:
        retrieved = list(rankings.get(case.id, []))
        relevant_ranks = [
            rank
            for rank, chunk in enumerate(retrieved, start=1)
            if any(_matches(chunk, evidence) for evidence in case.relevant)
        ]
        first_rank = min(relevant_ranks, default=None)
        recalls: dict[int, float] = {}
        reciprocal_ranks: dict[int, float] = {}
        for k in k_values:
            top_k = retrieved[:k]
            matched = sum(
                any(_matches(chunk, evidence) for chunk in top_k)
                for evidence in case.relevant
            )
            recall = matched / len(case.relevant)
            reciprocal_rank = (
                1.0 / first_rank if first_rank is not None and first_rank <= k else 0.0
            )
            recalls[k] = recall
            reciprocal_ranks[k] = reciprocal_rank
            per_k_recalls[k].append(recall)
            per_k_hits[k].append(float(matched > 0))
            per_k_rr[k].append(reciprocal_rank)

        threshold_k = dataset.thresholds.k
        missing = [
            f"{evidence.source}：{evidence.contains}"
            for evidence in case.relevant
            if not any(_matches(chunk, evidence) for chunk in retrieved[:threshold_k])
        ]
        case_results.append(
            RagCaseEvaluation(
                id=case.id,
                query=case.query,
                passed=not missing,
                first_relevant_rank=first_rank,
                recall_at_k=recalls,
                reciprocal_rank_at_k=reciprocal_ranks,
                missing_evidence=missing,
                retrieved=[
                    RagRetrievedItem(
                        source=chunk.source,
                        chunk_index=chunk.chunk_index + 1,
                        score=chunk.score,
                        excerpt=_safe_excerpt(chunk.text),
                    )
                    for chunk in retrieved[: max(k_values)]
                ],
            )
        )

    total_cases = len(dataset.cases)
    metrics = [
        RagMetricsAtK(
            k=k,
            recall=sum(per_k_recalls[k]) / total_cases,
            hit_rate=sum(per_k_hits[k]) / total_cases,
            mrr=sum(per_k_rr[k]) / total_cases,
        )
        for k in k_values
    ]
    threshold_metrics = next(
        metric for metric in metrics if metric.k == dataset.thresholds.k
    )
    passed = (
        threshold_metrics.recall >= dataset.thresholds.minimum_recall
        and threshold_metrics.mrr >= dataset.thresholds.minimum_mrr
    )
    return RagEvaluationReport(
        dataset_name=dataset.name,
        generated_at=datetime.now(timezone.utc),
        indexed_chunks=indexed_chunks,
        total_cases=total_cases,
        metrics=metrics,
        thresholds=dataset.thresholds,
        passed=passed,
        cases=case_results,
        limitations=[
            "离线哈希向量只验证词面检索回归，不代表线上 Qwen Embedding 的语义效果。",
            "指标验证相关证据能否被检索，不判断证据本身或最终报告的事实正确性。",
        ],
    )


async def run_offline_rag_evaluation(
    dataset: RagEvaluationDataset,
    *,
    index_dir: str | Path,
) -> RagEvaluationReport:
    """使用真实切分和 FAISS、离线向量运行完整评测链路。"""
    resolved_index_dir = Path(index_dir)
    if resolved_index_dir.exists() and any(resolved_index_dir.iterdir()):
        raise ValueError("RAG 离线评测必须使用空索引目录")
    vector_store = OfflineHashFaissService(resolved_index_dir)
    retriever = LocalDocumentRetriever(
        base_dir=resolved_index_dir,
        chunk_size=dataset.retriever.chunk_size,
        chunk_overlap=dataset.retriever.chunk_overlap,
        vector_store=vector_store,
    )

    try:
        indexed_chunks = await retriever.index_documents(
            [
                {"path": document.path, "content": document.evaluation_content}
                for document in dataset.documents
            ]
        )
        max_k = max(*dataset.retriever.k_values, dataset.thresholds.k)
        rankings: dict[str, list[RetrievedChunk]] = {}
        for case in dataset.cases:
            rankings[case.id] = await retriever.retrieve(case.query, k=max_k)
        return evaluate_rag_rankings(
            dataset,
            rankings,
            indexed_chunks=indexed_chunks,
        )
    finally:
        await retriever.close()


def _markdown_code(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value).strip().replace("`", "'")
    return f"`{normalized}`"


def render_rag_report_markdown(report: RagEvaluationReport) -> str:
    """把 RAG 评测结果渲染为便于人工审阅的中文 Markdown。"""
    status = "通过" if report.passed else "未通过"
    lines = [
        "# RAG 离线评测",
        "",
        f"- 评测集：{_markdown_code(report.dataset_name)}",
        f"- 总体门禁：**{status}**",
        f"- 用例数：{report.total_cases}",
        f"- 索引片段数：{report.indexed_chunks}",
        "",
        "## 指标",
        "",
        "| K | Recall@K | HitRate@K | MRR@K |",
        "| ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {metric.k} | {metric.recall:.2%} | "
        f"{metric.hit_rate:.2%} | {metric.mrr:.2%} |"
        for metric in report.metrics
    )
    failed_cases = [case for case in report.cases if not case.passed]
    lines.extend(["", "## 未命中用例", ""])
    if not failed_cases:
        lines.append("当前门禁 K 值下没有完全未命中的标注证据。")
    else:
        for case in failed_cases:
            lines.append(f"- `{case.id}` {_markdown_code(case.query)}")
            for missing in case.missing_evidence:
                lines.append(f"  - 缺失：{_markdown_code(missing)}")
    lines.extend(["", "## 能力边界", ""])
    lines.extend(f"- {limitation}" for limitation in report.limitations)
    return "\n".join(lines) + "\n"


__all__ = [
    "OfflineHashFaissService",
    "RagEvaluationCase",
    "RagEvaluationDataset",
    "RagEvaluationDocument",
    "RagEvaluationReport",
    "RagMetricsAtK",
    "RagRelevantEvidence",
    "evaluate_rag_rankings",
    "offline_hash_embeddings",
    "render_rag_report_markdown",
    "run_offline_rag_evaluation",
]
