"""提供数据类型相关实现。"""

from typing import Any, Dict, List, Optional, Union, Callable
from pydantic import BaseModel, Field


class FaissSearchRequest(BaseModel):
    """定义 `FaissSearchRequest`，封装相关数据与行为。"""
    query: str = Field(..., description="Query text to search for")
    k: int = Field(4, ge=1, le=1000, description="Number of documents to return")
    filter: Optional[Union[Dict[str, Any], Callable]] = Field(None, description="Filter by metadata")
    fetch_k: int = Field(20, ge=1, le=10000, description="Number of documents to fetch before filtering")
    score_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity score")


class FaissAddRequest(BaseModel):
    """定义 `FaissAddRequest`，封装相关数据与行为。"""
    texts: List[str] = Field(..., description="List of texts to add")
    metadatas: Optional[List[Dict[str, Any]]] = Field(None, description="Metadata for each text")
    ids: Optional[List[str]] = Field(None, description="Custom IDs for each text")


class FaissDeleteRequest(BaseModel):
    """定义 `FaissDeleteRequest`，封装相关数据与行为。"""
    ids: List[str] = Field(..., description="IDs of documents to delete")


class FaissIndexInfo(BaseModel):
    """定义 `FaissIndexInfo`，封装相关数据与行为。"""
    total_documents: int = Field(..., description="Total number of documents in index")
    embedding_dimension: int = Field(..., description="Dimension of embeddings")
    index_type: str = Field(..., description="Type of FAISS index")
    distance_strategy: str = Field(..., description="Distance strategy used")


class FaissConfig(BaseModel):
    """定义 `FaissConfig`，封装相关数据与行为。"""
    base_dir: str = Field(..., description="Base directory for FAISS storage")
    index_name: str = Field("index", description="Name of the FAISS index")
    model_name: Optional[str] = Field(
        default=None,
        description="向量模型；为空时使用模型管理器的独立 Embedding 配置。",
    )
    distance_strategy: str = Field("cosine", description="Distance strategy (euclidean, cosine, max_inner_product)")
    normalize_L2: bool = Field(False, description="Whether to normalize L2")
    max_documents: int = Field(1000000, description="Maximum number of documents")
    auto_save: bool = Field(False, description="Whether to auto-save the index")
    save_interval: int = Field(100, description="Save interval in number of operations")
