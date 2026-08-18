"""提供service相关实现。"""

import asyncio
import pickle
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from src.environment.faiss.exceptions import (
    FaissConfigurationError,
    FaissEmbeddingError,
    FaissIndexError,
    FaissStorageError,
)
from src.environment.faiss.types import (
    FaissAddRequest,
    FaissConfig,
    FaissDeleteRequest,
    FaissIndexInfo,
    FaissSearchRequest,
)
from src.environment.types import ActionResult
from src.logger import logger
from src.message import HumanMessage
from src.model import model_manager
from src.utils import write_pickle_file


def dependable_faiss_import():
    """实现 `dependable_faiss_import` 的业务逻辑。"""
    try:
        import faiss

        return faiss
    except ImportError:
        raise ImportError(
            "Could not import faiss python package. "
            "Please install it with `pip install faiss-gpu` (for CUDA supported GPU) "
            "or `pip install faiss-cpu`."
        )


class Document:
    """定义 `Document`，封装相关数据与行为。"""

    def __init__(self, page_content: str, metadata: dict[str, Any] | None = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class FaissService:
    """定义 `FaissService`，封装相关数据与行为。"""

    def __init__(
        self,
        base_dir: str | Path,
        model_name: str | None = None,
        config: FaissConfig | None = None,
    ):
        """初始化实例。"""
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.config = config or FaissConfig(base_dir=str(base_dir))
        # 聊天模型与向量模型分开配置；未显式指定时由模型管理器路由。
        self.model_name = model_name or self.config.model_name

        # 说明相关实现细节。
        self.faiss = dependable_faiss_import()
        self.index: Any | None = None
        self.docstore: dict[str, Document] = {}  # 说明相关实现细节。
        self.id_to_index: dict[str, int] = {}  # 说明相关实现细节。
        self.index_to_id: dict[int, str] = {}  # 说明相关实现细节。

        self._operation_count = 0
        self._embedding_dimension: int | None = None
        self._index_creation_lock = asyncio.Lock()  # 执行异步任务。

        # 初始化相关状态。
        self._initialize_index()

    def _initialize_index(self) -> None:
        """实现 `_initialize_index` 的业务逻辑。"""
        try:
            self.index_path = self.base_dir / f"{self.config.index_name}.faiss"
            self.pkl_path = self.base_dir / f"{self.config.index_name}.pkl"

            if self.index_path.exists() and self.pkl_path.exists():
                # 加载所需数据。
                self._load_index()
            else:
                # 创建所需对象。
                self._create_index()

        except Exception as e:
            raise FaissIndexError(f"Failed to initialize FAISS index: {e}")

    async def _get_embedding_dimension(self) -> int:
        """实现 `_get_embedding_dimension` 的业务逻辑。"""
        if self._embedding_dimension is not None:
            return self._embedding_dimension

        try:
            test_message = [HumanMessage(content="test")]
            response = await model_manager.aembedding(
                model=self.model_name or model_manager.embedding_model_name,
                messages=test_message,
            )

            if not response.success:
                raise FaissConfigurationError(
                    f"Failed to get embedding dimension: {response.message}"
                )

            response_data = response.extra.data if response.extra else None
            if not response_data or "embeddings" not in response_data:
                raise FaissEmbeddingError("Embedding response is missing vectors")
            embedding = response_data["embeddings"]

            # 处理模型调用。
            # 组装并返回结果。
            # 组装并返回结果。
            if isinstance(embedding, np.ndarray):
                # 处理模型调用。
                if embedding.ndim == 1:
                    self._embedding_dimension = embedding.shape[0]
                else:
                    self._embedding_dimension = embedding.shape[-1]
            elif isinstance(embedding, list) and len(embedding) > 0:
                # 处理模型调用。
                first_emb = embedding[0]
                if isinstance(first_emb, np.ndarray):
                    if first_emb.ndim == 1:
                        self._embedding_dimension = first_emb.shape[0]
                    else:
                        self._embedding_dimension = first_emb.shape[-1]
                else:
                    # 执行回退或重试逻辑。
                    self._embedding_dimension = len(first_emb)
            else:
                raise FaissConfigurationError(
                    f"Invalid embedding response format: {type(embedding)}"
                )

            return self._embedding_dimension

        except Exception as e:
            raise FaissConfigurationError(f"Failed to get embedding dimension: {e}")

    def _create_index(self) -> None:
        """实现 `_create_index` 的业务逻辑。"""
        try:
            # 更新相关状态。
            # 创建所需对象。
            logger.info("| 🔍 FAISS index will be created on first document addition")

        except Exception as e:
            raise FaissIndexError(f"Failed to create FAISS index: {e}")

    async def _ensure_index_created(self) -> None:
        """实现 `_ensure_index_created` 的业务逻辑。"""
        # 校验输入与当前状态。
        if self.index is not None:
            return

        async with self._index_creation_lock:
            # 校验输入与当前状态。
            if self.index is not None:
                return

            dimension = await self._get_embedding_dimension()

            # 创建所需对象。
            if self.config.distance_strategy == "max_inner_product":
                self.index = self.faiss.IndexFlatIP(dimension)
            elif self.config.distance_strategy == "cosine":
                # 说明相关实现细节。
                self.index = self.faiss.IndexFlatL2(dimension)
            else:  # 说明相关实现细节。
                self.index = self.faiss.IndexFlatL2(dimension)

            logger.info(
                f"| 🔍 Created FAISS index with dimension {dimension}, strategy: {self.config.distance_strategy}"
            )

    def _load_index(self) -> None:
        """实现 `_load_index` 的业务逻辑。"""
        try:
            # 加载所需数据。
            self.index = self.faiss.read_index(str(self.index_path))

            # 加载所需数据。
            with open(self.pkl_path, "rb") as f:
                data = pickle.load(f)
                self.docstore = data.get("docstore", {})
                self.id_to_index = data.get("id_to_index", {})
                self.index_to_id = data.get("index_to_id", {})
                self._embedding_dimension = data.get("embedding_dimension")

            logger.info(
                f"| 🔍 Loaded existing FAISS index from {self.base_dir} with {len(self.docstore)} documents"
            )

        except Exception as e:
            raise FaissIndexError(f"Failed to load FAISS index: {e}")

    async def _get_embeddings(self, texts: list[str]) -> np.ndarray:
        """实现 `_get_embeddings` 的业务逻辑。"""
        try:
            messages = [HumanMessage(content=text) for text in texts]
            response = await model_manager.aembedding(
                model=self.model_name or model_manager.embedding_model_name,
                messages=messages,
            )

            if not response.success:
                raise FaissEmbeddingError(
                    f"Failed to get embeddings: {response.message}"
                )

            response_data = response.extra.data if response.extra else None
            if not response_data or "embeddings" not in response_data:
                raise FaissEmbeddingError("Embedding response is missing vectors")
            embeddings = response_data["embeddings"]

            # 处理模型调用。
            # 组装并返回结果。
            # 组装并返回结果。
            if isinstance(embeddings, np.ndarray):
                # 处理模型调用。
                if embeddings.ndim == 1:
                    embeddings = embeddings.reshape(1, -1)
                return embeddings.astype(np.float32)
            elif isinstance(embeddings, list):
                # 处理模型调用。
                # 说明相关实现细节。
                embeddings_list = []
                for emb in embeddings:
                    if isinstance(emb, np.ndarray):
                        if emb.ndim == 1:
                            embeddings_list.append(emb)
                        else:
                            embeddings_list.append(emb.flatten())
                    else:
                        # 执行回退或重试逻辑。
                        embeddings_list.append(np.array(emb))

                if embeddings_list:
                    embeddings_array = np.stack(embeddings_list, axis=0).astype(
                        np.float32
                    )
                    return embeddings_array
                else:
                    raise FaissEmbeddingError("Empty embeddings list")
            else:
                raise FaissEmbeddingError(
                    f"Unexpected embedding format: {type(embeddings)}"
                )

        except Exception as e:
            raise FaissEmbeddingError(f"Failed to get embeddings: {e}")

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """实现 `_normalize_vectors` 的业务逻辑。"""
        if self.config.distance_strategy == "cosine" or self.config.normalize_L2:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1  # 说明相关实现细节。
            return vectors / norms
        return vectors

    async def add_documents(self, request: FaissAddRequest) -> ActionResult:
        """添加与 `add_documents` 对应的数据或状态。"""
        try:
            # 创建所需对象。
            await self._ensure_index_created()

            # 说明相关实现细节。
            valid_texts = []
            valid_metadatas = []
            valid_indices = []
            for i, text in enumerate(request.texts):
                if text and text.strip():
                    valid_texts.append(text)
                    if request.metadatas and i < len(request.metadatas):
                        valid_metadatas.append(request.metadatas[i])
                    else:
                        valid_metadatas.append({})
                    valid_indices.append(i)

            if not valid_texts:
                logger.info("| ⚠️ No valid texts to add (all texts were empty)")
                return ActionResult(
                    success=True,
                    message="No valid texts to add (all texts were empty)",
                    extra={"ids": [], "count": 0, "total_input": len(request.texts)},
                )

            # 说明相关实现细节。
            ids = []
            if request.ids:
                # 说明相关实现细节。
                for idx in valid_indices:
                    if idx < len(request.ids):
                        ids.append(request.ids[idx])
                    else:
                        ids.append(str(uuid.uuid4()))
            else:
                ids = [str(uuid.uuid4()) for _ in valid_texts]

            # 处理模型调用。
            embeddings = await self._get_embeddings(valid_texts)
            embeddings = self._normalize_vectors(embeddings)

            # 说明相关实现细节。
            start_idx = self.index.ntotal
            self.index.add(embeddings)

            # 说明相关实现细节。
            for i, (text, metadata, doc_id) in enumerate(
                zip(valid_texts, valid_metadatas, ids)
            ):
                idx = start_idx + i
                self.docstore[doc_id] = Document(page_content=text, metadata=metadata)
                self.id_to_index[doc_id] = idx
                self.index_to_id[idx] = doc_id

            self._operation_count += 1
            await self._auto_save()

            logger.info(f"| ➕ Added {len(ids)} documents to FAISS index")
            return ActionResult(
                success=True,
                message=f"Added {len(ids)} documents to FAISS index",
                extra={
                    "ids": ids,
                    "count": len(ids),
                    "total_input": len(request.texts),
                    "valid_input": len(valid_texts),
                },
            )

        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Failed to add documents: {e!s}",
                extra={"error": str(e)},
            )

    async def search_similar(self, request: FaissSearchRequest) -> ActionResult:
        """搜索与 `search_similar` 对应的数据或状态。"""
        if self.index is None or self.index.ntotal == 0:
            return ActionResult(
                success=False,
                message="FAISS index not initialized or empty",
                extra={"error": "FAISS index not initialized or empty"},
            )

        try:
            # 检索所需信息。
            query_embeddings = await self._get_embeddings([request.query])
            query_embedding = self._normalize_vectors(query_embeddings)[
                0:1
            ]  # 说明相关实现细节。

            # 检索所需信息。
            k = min(request.k, self.index.ntotal)
            fetch_k = min(request.fetch_k or k, self.index.ntotal)

            distances, indices = self.index.search(query_embedding, fetch_k)

            # 转换并规范化数据。
            if self.config.distance_strategy == "cosine":
                # 归一化向量的平方欧氏距离满足 d=2-2*cosine。
                scores = np.clip(1 - distances[0] / 2, -1.0, 1.0)
            elif self.config.distance_strategy == "max_inner_product":
                # 加载所需数据。
                scores = distances[0]
            else:  # 说明相关实现细节。
                # 转换并规范化数据。
                # 说明相关实现细节。
                max_dist = np.max(distances[0]) if len(distances[0]) > 0 else 1.0
                scores = 1 - (distances[0] / (max_dist + 1e-8))

            # 说明相关实现细节。
            docs_and_scores = []
            for idx, score in zip(indices[0], scores):
                if idx < 0:  # 组装并返回结果。
                    continue
                doc_id = self.index_to_id.get(idx)
                if doc_id and doc_id in self.docstore:
                    doc = self.docstore[doc_id]
                    # 说明相关实现细节。
                    if request.filter:
                        if isinstance(request.filter, dict):
                            # 校验输入与当前状态。
                            if not all(
                                doc.metadata.get(k) == v
                                for k, v in request.filter.items()
                            ):
                                continue
                        elif callable(request.filter) and not request.filter(
                            doc.metadata
                        ):
                            continue
                    docs_and_scores.append((doc, float(score)))

            # 说明相关实现细节。
            docs_and_scores.sort(key=lambda x: x[1], reverse=True)

            # 说明相关实现细节。
            if request.score_threshold is not None:
                docs_and_scores = [
                    (doc, score)
                    for doc, score in docs_and_scores
                    if score >= request.score_threshold
                ]

            # 说明相关实现细节。
            docs_and_scores = docs_and_scores[:k]

            documents = [doc for doc, _ in docs_and_scores]
            scores_list = [score for _, score in docs_and_scores]

            # 转换并规范化数据。
            documents_dict = [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in documents
            ]

            logger.info(
                f"| 🔍 Found {len(documents)} similar documents for query: {request.query[:50]}..."
            )
            return ActionResult(
                success=True,
                message=f"Found {len(documents)} similar documents",
                extra={
                    "documents": documents_dict,
                    "scores": scores_list,
                    "total_found": len(documents),
                    "query": request.query,
                    "k": request.k,
                },
            )

        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Failed to search documents: {e!s}",
                extra={"error": str(e), "query": request.query},
            )

    async def delete_documents(self, request: FaissDeleteRequest) -> ActionResult:
        """删除与 `delete_documents` 对应的数据或状态。"""
        if self.index is None:
            return ActionResult(
                success=False,
                message="FAISS index not initialized",
                extra={"error": "FAISS index not initialized"},
            )

        try:
            deleted_count = 0
            for doc_id in request.ids:
                if doc_id in self.docstore:
                    # 移除相关数据或组件。
                    idx = self.id_to_index.pop(doc_id, None)
                    if idx is not None:
                        self.index_to_id.pop(idx, None)
                    del self.docstore[doc_id]
                    deleted_count += 1

            self._operation_count += 1
            await self._auto_save()

            logger.info(f"| 🗑️ Deleted {deleted_count} documents from FAISS index")
            return ActionResult(
                success=True,
                message=f"Deleted {deleted_count} documents from FAISS index",
                extra={
                    "deleted_count": deleted_count,
                    "requested_ids": request.ids,
                    "total_requested": len(request.ids),
                },
            )

        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Failed to delete documents: {e!s}",
                extra={"error": str(e), "requested_ids": request.ids},
            )

    async def get_index_info(self) -> ActionResult:
        """获取与 `get_index_info` 对应的数据或状态。"""
        if self.index is None:
            return ActionResult(
                success=False,
                message="FAISS index not initialized",
                extra={"error": "FAISS index not initialized"},
            )

        try:
            total_documents = len(self.docstore)
            embedding_dimension = (
                self.index.d
                if hasattr(self.index, "d")
                else self._embedding_dimension or 0
            )

            index_info = FaissIndexInfo(
                total_documents=total_documents,
                embedding_dimension=embedding_dimension,
                index_type=type(self.index).__name__,
                distance_strategy=self.config.distance_strategy,
            )

            return ActionResult(
                success=True,
                message="FAISS index information retrieved successfully",
                extra={
                    "index_info": index_info.model_dump(),
                    "total_documents": total_documents,
                    "embedding_dimension": embedding_dimension,
                    "index_type": type(self.index).__name__,
                    "distance_strategy": self.config.distance_strategy,
                    "index_name": self.config.index_name,
                },
            )

        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Failed to get index info: {e!s}",
                extra={"error": str(e)},
            )

    async def save_index(self) -> None:
        """保存与 `save_index` 对应的数据或状态。"""
        if self.index is None:
            raise FaissIndexError("FAISS index not initialized")

        try:
            # 持久化相关数据。
            self.faiss.write_index(self.index, str(self.index_path))

            # 持久化相关数据。
            data = {
                "docstore": self.docstore,
                "id_to_index": self.id_to_index,
                "index_to_id": self.index_to_id,
                "embedding_dimension": self._embedding_dimension,
            }
            await write_pickle_file(self.pkl_path, data)

            logger.info(f"| 💾 Saved FAISS index to {self.base_dir}")

        except Exception as e:
            raise FaissStorageError(f"Failed to save FAISS index: {e}")

    async def _auto_save(self) -> None:
        """实现 `_auto_save` 的业务逻辑。"""
        if (
            self.config.auto_save
            and self._operation_count % self.config.save_interval == 0
        ):
            await self.save_index()

    async def cleanup(self) -> None:
        """释放组件占用的资源。"""
        try:
            # 清理并释放相关资源。
            if self.index is not None:
                await self.save_index()
        except Exception as e:
            logger.warning(f"| ⚠️ Error during FAISS cleanup: {e}")
