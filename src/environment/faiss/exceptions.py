"""提供exceptions相关实现。"""

from typing import Any, Dict, Optional


class FaissError(Exception):
    """定义 `FaissError`，封装相关数据与行为。"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class FaissIndexError(FaissError):
    """定义 `FaissIndexError`，封装相关数据与行为。"""
    pass


class FaissEmbeddingError(FaissError):
    """定义 `FaissEmbeddingError`，封装相关数据与行为。"""
    pass


class FaissDocumentError(FaissError):
    """定义 `FaissDocumentError`，封装相关数据与行为。"""
    pass


class FaissSearchError(FaissError):
    """定义 `FaissSearchError`，封装相关数据与行为。"""
    pass


class FaissStorageError(FaissError):
    """定义 `FaissStorageError`，封装相关数据与行为。"""
    pass


class FaissConfigurationError(FaissError):
    """定义 `FaissConfigurationError`，封装相关数据与行为。"""
    pass
