"""提供faiss environment相关实现。"""

from typing import Any

from pydantic import ConfigDict, Field

from src.environment.faiss.service import FaissService
from src.environment.faiss.types import (
    FaissAddRequest,
    FaissConfig,
    FaissDeleteRequest,
    FaissSearchRequest,
)
from src.environment.server import ecp
from src.environment.types import Environment
from src.logger import logger
from src.registry import ENVIRONMENT
from src.utils import assemble_project_path, dedent


@ENVIRONMENT.register_module(force=True)
class FaissEnvironment(Environment):
    """定义 `FaissEnvironment`，封装相关数据与行为。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(default="faiss", description="The name of the FAISS environment.")
    description: str = Field(
        default="FAISS vector store environment for similarity search and document management",
        description="The description of the FAISS environment.",
    )
    metadata: dict[str, Any] = Field(
        default={
            "has_vision": False,
            "additional_rules": {
                "state": "The state of the FAISS vector store environment.",
            },
        },
        description="The metadata of the FAISS environment.",
    )
    require_grad: bool = Field(
        default=False, description="Whether the environment requires gradients"
    )

    def __init__(
        self,
        base_dir: str,
        model_name: str | None = None,
        config: FaissConfig | None = None,
        require_grad: bool = False,
        **kwargs,
    ):
        """初始化实例。"""
        super().__init__(**kwargs)

        self.base_dir = assemble_project_path(base_dir)
        self.model_name = model_name
        self.config = config or FaissConfig(base_dir=self.base_dir)

        # 初始化相关状态。
        self.faiss_service = FaissService(
            base_dir=self.base_dir,
            model_name=self.model_name or self.config.model_name,
            config=self.config,
        )

    async def initialize(self) -> None:
        """初始化组件及其依赖资源。"""
        logger.info(f"| 🔍 FAISS Environment initialized at: {self.base_dir}")

    async def cleanup(self) -> None:
        """释放组件占用的资源。"""
        await self.faiss_service.cleanup()
        logger.info("| 🧹 FAISS Environment cleanup completed")

    @ecp.action(
        name="add_documents",
        description="Add documents to the FAISS vector store",
    )
    async def add_documents(
        self, texts: list[str], metadatas: list[dict[str, Any]] | None = None, **kwargs
    ) -> dict[str, Any]:
        """添加与 `add_documents` 对应的数据或状态。"""
        try:
            request = FaissAddRequest(
                texts=texts,
                metadatas=metadatas,
            )

            result = await self.faiss_service.add_documents(request)

            extra = result.extra.copy() if result.extra else {}

            return {
                "success": result.success,
                "message": result.message,
                "extra": extra,
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to add documents: {e!s}",
                "extra": {"error": str(e)},
            }

    @ecp.action(
        name="search_similar",
        description="Search for similar documents in the FAISS vector store",
        metadata={},
    )
    async def search_similar(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        fetch_k: int = 20,
        score_threshold: float | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """搜索与 `search_similar` 对应的数据或状态。"""
        try:
            request = FaissSearchRequest(
                query=query,
                k=k,
                filter=filter,
                fetch_k=fetch_k,
                score_threshold=score_threshold,
            )

            result = await self.faiss_service.search_similar(request)

            extra = result.extra.copy() if result.extra else {}

            # 组装并返回结果。
            if result.success and "documents" in extra and extra["documents"]:
                documents_info = []
                documents = extra["documents"]
                scores = extra.get("scores", [])
                for i, (doc, score) in enumerate(zip(documents, scores)):
                    content = (
                        doc.get("page_content", "")[:200]
                        if isinstance(doc, dict)
                        else str(doc)[:200]
                    )
                    documents_info.append(
                        f"Document {i + 1} (Score: {score:.4f}):\n{content}..."
                    )
                message = (
                    f"Found {extra.get('total_found', 0)} similar documents for query '{query}':\n\n"
                    + "\n\n".join(documents_info)
                )
            else:
                message = result.message

            return {"success": result.success, "message": message, "extra": extra}
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to search documents: {e!s}",
                "extra": {"error": str(e), "query": query},
            }

    @ecp.action(
        name="delete_documents",
        description="Delete documents from the FAISS vector store",
    )
    async def delete_documents(self, ids: list[str], **kwargs) -> dict[str, Any]:
        """删除与 `delete_documents` 对应的数据或状态。"""
        try:
            request = FaissDeleteRequest(ids=ids)
            result = await self.faiss_service.delete_documents(request)

            extra = result.extra.copy() if result.extra else {}

            return {
                "success": result.success,
                "message": result.message,
                "extra": extra,
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to delete documents: {e!s}",
                "extra": {"error": str(e), "ids": ids},
            }

    @ecp.action(
        name="get_index_info", description="Get information about the FAISS index"
    )
    async def get_index_info(self, **kwargs) -> dict[str, Any]:
        """获取与 `get_index_info` 对应的数据或状态。"""
        try:
            result = await self.faiss_service.get_index_info()

            extra = result.extra.copy() if result.extra else {}

            if result.success:
                message = (
                    f"FAISS Index Information:\n"
                    f"Total Documents: {extra.get('total_documents', 0)}\n"
                    f"Embedding Dimension: {extra.get('embedding_dimension', 0)}\n"
                    f"Index Type: {extra.get('index_type', 'Unknown')}\n"
                    f"Distance Strategy: {extra.get('distance_strategy', 'Unknown')}"
                )
            else:
                message = result.message

            return {"success": result.success, "message": message, "extra": extra}
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to get index info: {e!s}",
                "extra": {"error": str(e)},
            }

    @ecp.action(name="save_index", description="Save the FAISS index to disk")
    async def save_index(self, **kwargs) -> dict[str, Any]:
        """保存与 `save_index` 对应的数据或状态。"""
        try:
            await self.faiss_service.save_index()
            return {
                "success": True,
                "message": f"FAISS index saved successfully to: {self.base_dir}",
                "extra": {"base_dir": str(self.base_dir)},
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to save index: {e!s}",
                "extra": {"error": str(e), "base_dir": str(self.base_dir)},
            }

    async def get_state(self, **kwargs) -> dict[str, Any]:
        """获取与 `get_state` 对应的数据或状态。"""
        try:
            index_result = await self.faiss_service.get_index_info()
            extra = index_result.extra if index_result.extra else {}

            state = dedent(f"""
                <info>
                Base Directory: {self.base_dir!s}
                Index Name: {self.config.index_name}
                Total Documents: {extra.get("total_documents", 0)}
                Embedding Dimension: {extra.get("embedding_dimension", 0)}
                Index Type: {extra.get("index_type", "Unknown")}
                Distance Strategy: {extra.get("distance_strategy", "Unknown")}
                Auto Save: {self.config.auto_save}
                Save Interval: {self.config.save_interval}
                </info>
            """)
            return {"state": state, "extra": extra}
        except Exception as e:
            logger.error(f"| ❌ Failed to get FAISS state: {e}")
            return {
                "state": f"Failed to get FAISS state: {e!s}",
                "extra": {"error": str(e)},
            }
