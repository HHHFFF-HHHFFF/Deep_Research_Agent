from __future__ import annotations

from pathlib import Path


class FileSystemError(Exception):
    """定义 `FileSystemError`，封装相关数据与行为。"""

    def __init__(
        self,
        message: str,
        path: str | Path | None = None,
        error_code: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.path = str(path) if path else None
        self.error_code = error_code

    def __str__(self) -> str:
        if self.path:
            return f"{self.message} (path: {self.path})"
        return self.message


class InvalidPathError(FileSystemError):
    """定义 `InvalidPathError`，封装相关数据与行为。"""

    def __init__(self, message: str, path: str | Path | None = None):
        super().__init__(message, path, "INVALID_PATH")


class PathTraversalError(FileSystemError):
    """定义 `PathTraversalError`，封装相关数据与行为。"""

    def __init__(
        self,
        message: str,
        path: str | Path | None = None,
        base_dir: str | Path | None = None,
    ):
        super().__init__(message, path, "PATH_TRAVERSAL")
        self.base_dir = str(base_dir) if base_dir else None


class NotFoundError(FileSystemError):
    """定义 `NotFoundError`，封装相关数据与行为。"""

    def __init__(self, message: str, path: str | Path | None = None):
        super().__init__(message, path, "NOT_FOUND")


class ConflictError(FileSystemError):
    """定义 `ConflictError`，封装相关数据与行为。"""

    def __init__(
        self,
        message: str,
        path: str | Path | None = None,
        conflict_type: str | None = None,
    ):
        super().__init__(message, path, "CONFLICT")
        self.conflict_type = conflict_type


class PermissionDeniedError(FileSystemError):
    """定义 `PermissionDeniedError`，封装相关数据与行为。"""

    def __init__(
        self, message: str, path: str | Path | None = None, operation: str | None = None
    ):
        super().__init__(message, path, "PERMISSION_DENIED")
        self.operation = operation


class UnsupportedTypeError(FileSystemError):
    """定义 `UnsupportedTypeError`，封装相关数据与行为。"""

    def __init__(
        self, message: str, file_type: str | None = None, path: str | Path | None = None
    ):
        super().__init__(message, path, "UNSUPPORTED_TYPE")
        self.file_type = file_type


class InvalidArgumentError(FileSystemError):
    """定义 `InvalidArgumentError`，封装相关数据与行为。"""

    def __init__(
        self, message: str, argument: str | None = None, value: str | None = None
    ):
        super().__init__(message, error_code="INVALID_ARGUMENT")
        self.argument = argument
        self.value = value


class CacheError(FileSystemError):
    """定义 `CacheError`，封装相关数据与行为。"""

    def __init__(self, message: str, operation: str | None = None):
        super().__init__(message, error_code="CACHE_ERROR")
        self.operation = operation


class StorageError(FileSystemError):
    """定义 `StorageError`，封装相关数据与行为。"""

    def __init__(
        self, message: str, path: str | Path | None = None, operation: str | None = None
    ):
        super().__init__(message, path, "STORAGE_ERROR")
        self.operation = operation


class HandlerError(FileSystemError):
    """定义 `HandlerError`，封装相关数据与行为。"""

    def __init__(
        self,
        message: str,
        handler_type: str | None = None,
        path: str | Path | None = None,
    ):
        super().__init__(message, path, "HANDLER_ERROR")
        self.handler_type = handler_type


class LockError(FileSystemError):
    """定义 `LockError`，封装相关数据与行为。"""

    def __init__(
        self, message: str, path: str | Path | None = None, operation: str | None = None
    ):
        super().__init__(message, path, "LOCK_ERROR")
        self.operation = operation
