"""提供exceptions相关实现。"""

from typing import Any


class FaissError(Exception):
    """定义 `FaissError`，封装相关数据与行为。"""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


class FaissIndexError(FaissError):
    """定义 `FaissIndexError`，封装相关数据与行为。"""


class FaissEmbeddingError(FaissError):
    """定义 `FaissEmbeddingError`，封装相关数据与行为。"""


class FaissDocumentError(FaissError):
    """定义 `FaissDocumentError`，封装相关数据与行为。"""


class FaissSearchError(FaissError):
    """定义 `FaissSearchError`，封装相关数据与行为。"""


class FaissStorageError(FaissError):
    """定义 `FaissStorageError`，封装相关数据与行为。"""


class FaissConfigurationError(FaissError):
    """定义 `FaissConfigurationError`，封装相关数据与行为。"""
