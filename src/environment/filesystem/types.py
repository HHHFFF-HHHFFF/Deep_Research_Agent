from __future__ import annotations

from pathlib import Path
from typing import Literal, NamedTuple, Optional, Union, List, Dict, Any
from datetime import datetime

from pydantic import BaseModel, Field, validator


class FilePath(NamedTuple):
    """定义 `FilePath`，封装相关数据与行为。"""

    relative: Path
    absolute: Path


class FileReadRequest(BaseModel):
    """定义 `FileReadRequest`，封装相关数据与行为。"""
    path: Path
    as_text: bool = True
    encoding: str = "utf-8"
    start_line: Optional[int] = Field(None, ge=1, description="Start line number (1-based)")
    end_line: Optional[int] = Field(None, ge=1, description="End line number (1-based)")
    max_bytes: Optional[int] = Field(5 * 1024 * 1024, ge=0, description="Maximum bytes to read")

    @validator('end_line')
    def validate_line_range(cls, v, values):
        """校验与 `validate_line_range` 对应的数据或状态。"""
        if v is not None and 'start_line' in values and values['start_line'] is not None:
            if v <= values['start_line']:
                raise ValueError('end_line must be greater than start_line')
        return v


class FileReadResult(BaseModel):
    """定义 `FileReadResult`，封装相关数据与行为。"""
    path: Path
    source: Literal["cache", "disk", "remote"]
    content_bytes: Optional[bytes] = None
    content_text: Optional[str] = None
    total_lines: Optional[int] = Field(None, ge=0)
    preview: Optional[str] = None
    read_time: Optional[datetime] = None
    file_size: Optional[int] = Field(None, ge=0)


class SearchMatch(BaseModel):
    """定义 `SearchMatch`，封装相关数据与行为。"""
    line: int = Field(ge=1, description="Line number (1-based)")
    text: str
    column: Optional[int] = Field(None, ge=0, description="Column position of match")
    context_before: Optional[str] = None
    context_after: Optional[str] = None


class SearchResult(BaseModel):
    """定义 `SearchResult`，封装相关数据与行为。"""
    path: Path
    matches: List[SearchMatch]
    total_matches: Optional[int] = Field(None, ge=0)
    search_time: Optional[datetime] = None


class FileStats(BaseModel):
    """定义 `FileStats`，封装相关数据与行为。"""
    size: int = Field(ge=0)
    created: Optional[datetime] = None
    modified: Optional[datetime] = None
    accessed: Optional[datetime] = None
    permissions: Optional[str] = None
    is_directory: bool = False
    is_file: bool = False
    is_symlink: bool = False


class DirectoryInfo(BaseModel):
    """定义 `DirectoryInfo`，封装相关数据与行为。"""
    path: Path
    total_files: int = Field(ge=0)
    total_directories: int = Field(ge=0)
    total_size: int = Field(ge=0)
    file_types: Dict[str, int] = Field(default_factory=dict)
    last_modified: Optional[datetime] = None


class CacheStats(BaseModel):
    """定义 `CacheStats`，封装相关数据与行为。"""
    entries: int = Field(ge=0)
    total_bytes: int = Field(ge=0)
    max_entries: int = Field(ge=0)
    max_bytes: int = Field(ge=0)
    hits: int = Field(ge=0)
    misses: int = Field(ge=0)
    hit_rate: float = Field(ge=0.0, le=100.0)
    ttl_seconds: int = Field(ge=0)


# 处理输入参数。

class FileWriteRequest(BaseModel):
    """定义 `FileWriteRequest`，封装相关数据与行为。"""
    path: Path
    content: str
    mode: str = Field("w", description="Write mode: 'w' for overwrite, 'a' for append")
    encoding: str = Field("utf-8", description="Text encoding")


class FileReplaceRequest(BaseModel):
    """定义 `FileReplaceRequest`，封装相关数据与行为。"""
    path: Path
    old_string: str
    new_string: str
    start_line: Optional[int] = Field(None, ge=1, description="Start line number (1-based)")
    end_line: Optional[int] = Field(None, ge=1, description="End line number (1-based)")
    encoding: str = Field("utf-8", description="Text encoding")


class FileDeleteRequest(BaseModel):
    """定义 `FileDeleteRequest`，封装相关数据与行为。"""
    path: Path


class FileCopyRequest(BaseModel):
    """定义 `FileCopyRequest`，封装相关数据与行为。"""
    src_path: Path
    dst_path: Path
    overwrite: bool = Field(False, description="Whether to overwrite existing file")


class FileMoveRequest(BaseModel):
    """定义 `FileMoveRequest`，封装相关数据与行为。"""
    src_path: Path
    dst_path: Path
    overwrite: bool = Field(False, description="Whether to overwrite existing file")


class DirectoryCreateRequest(BaseModel):
    """定义 `DirectoryCreateRequest`，封装相关数据与行为。"""
    path: Path
    parents: bool = Field(True, description="Whether to create parent directories")


class DirectoryDeleteRequest(BaseModel):
    """定义 `DirectoryDeleteRequest`，封装相关数据与行为。"""
    path: Path
    recursive: bool = Field(False, description="Whether to delete recursively")


class FileListRequest(BaseModel):
    """定义 `FileListRequest`，封装相关数据与行为。"""
    path: Path
    show_hidden: bool = Field(False, description="Whether to show hidden files")
    file_types: Optional[List[str]] = Field(None, description="Filter by file extensions")


class FileTreeRequest(BaseModel):
    """定义 `FileTreeRequest`，封装相关数据与行为。"""
    path: Path
    max_depth: int = Field(3, ge=1, le=10, description="Maximum tree depth")
    show_hidden: bool = Field(False, description="Whether to show hidden files")
    exclude_patterns: Optional[List[str]] = Field(None, description="Patterns to exclude")
    file_types: Optional[List[str]] = Field(None, description="Filter by file extensions")


class FileSearchRequest(BaseModel):
    """定义 `FileSearchRequest`，封装相关数据与行为。"""
    path: Path
    query: str
    by: str = Field("name", description="Search by 'name' or 'content'")
    file_types: Optional[List[str]] = Field(None, description="Filter by file extensions")
    case_sensitive: bool = Field(False, description="Whether search is case sensitive")
    max_results: int = Field(100, ge=1, le=1000, description="Maximum number of results")


class FileStatRequest(BaseModel):
    """定义 `FileStatRequest`，封装相关数据与行为。"""
    path: Path

class FileChangePermissionsRequest(BaseModel):
    """定义 `FileChangePermissionsRequest`，封装相关数据与行为。"""
    path: Path
    permissions: str
