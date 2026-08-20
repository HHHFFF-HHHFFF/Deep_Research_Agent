"""为本地 RAG 提供受限、可缓存的文档解析入口。"""

from __future__ import annotations

import asyncio
import warnings
import zipfile
from pathlib import Path

SUPPORTED_DOCUMENT_EXTENSIONS = frozenset({".md", ".txt", ".pdf", ".docx"})
DOCUMENT_PARSE_TIMEOUT_SECONDS = 30.0
MAX_PARSED_DOCUMENT_CHARACTERS = 200_000
MAX_DOCX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024
PARSED_CACHE_SUFFIX = ".parsed.md"


class DocumentParseError(RuntimeError):
    """表示文档无法被安全转换为非空 Markdown。"""


class EmptyDocumentError(DocumentParseError):
    """表示文件存在，但解析后没有可用于研究的文字。"""


class DocumentParseTimeoutError(DocumentParseError):
    """表示文档解析时间超过当前个人项目的限制。"""


class DocumentContentTooLargeError(DocumentParseError):
    """表示解析文本过长，不适合进入当前本地 Embedding 流程。"""


def parsed_cache_path(source: str | Path) -> Path:
    """返回与原文件同目录、由服务器文件名确定的解析缓存路径。"""
    path = Path(source)
    return path.with_name(f"{path.name}{PARSED_CACHE_SUFFIX}")


def _read_cached_markdown(source: Path) -> str | None:
    cache_path = parsed_cache_path(source)
    if (
        not cache_path.is_file()
        or cache_path.stat().st_mtime_ns < source.stat().st_mtime_ns
    ):
        return None
    content = cache_path.read_text(encoding="utf-8").strip()
    return content or None


def _convert_document(source: Path) -> str:
    """使用 mdify 的底层 MarkItDown 转换器读取受支持文档。"""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Couldn't find ffmpeg or avconv*",
            category=RuntimeWarning,
        )
        from markitdown import MarkItDown

    result = MarkItDown(enable_plugins=False).convert(str(source))
    content = getattr(result, "markdown", "")
    if not isinstance(content, str):
        return ""
    return content.strip()


def _validate_document_signature(source: Path) -> None:
    """阻止纯文本伪装成 PDF 或 DOCX 后落入通用文本回退。"""
    extension = source.suffix.lower()
    if extension == ".pdf":
        with source.open("rb") as file:
            if file.read(5) != b"%PDF-":
                raise DocumentParseError("PDF 文件签名无效")
    elif extension == ".docx":
        if not zipfile.is_zipfile(source):
            raise DocumentParseError("DOCX 文件签名无效")
        try:
            with zipfile.ZipFile(source) as archive:
                members = archive.infolist()
                names = {member.filename for member in members}
        except zipfile.BadZipFile as error:
            raise DocumentParseError("DOCX 文件结构无效") from error
        if "[Content_Types].xml" not in names or "word/document.xml" not in names:
            raise DocumentParseError("DOCX 文件结构无效")
        if sum(member.file_size for member in members) > MAX_DOCX_UNCOMPRESSED_BYTES:
            raise DocumentParseError("DOCX 解压后内容超过限制")


def _validate_content_size(content: str, max_characters: int) -> None:
    if len(content) > max_characters:
        raise DocumentContentTooLargeError(
            f"文档解析内容不能超过 {max_characters} 个字符"
        )


def _write_cache(source: Path, content: str) -> None:
    cache_path = parsed_cache_path(source)
    temporary_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(cache_path)


async def parse_local_document(
    source: str | Path,
    *,
    timeout_seconds: float = DOCUMENT_PARSE_TIMEOUT_SECONDS,
    max_characters: int = MAX_PARSED_DOCUMENT_CHARACTERS,
    cache_result: bool = False,
) -> str:
    """解析本地文档，拒绝伪格式和空结果，并复用有效缓存。"""
    path = Path(source).resolve()
    if not path.is_file():
        raise DocumentParseError("本地文档不存在")
    if path.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise DocumentParseError("暂不支持该文件格式")
    await asyncio.to_thread(_validate_document_signature, path)

    cached = await asyncio.to_thread(_read_cached_markdown, path)
    if cached is not None:
        _validate_content_size(cached, max_characters)
        return cached

    try:
        content = await asyncio.wait_for(
            asyncio.to_thread(_convert_document, path),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError as error:
        raise DocumentParseTimeoutError("文档解析超时") from error
    except Exception as error:
        raise DocumentParseError("文档格式无效或内容无法解析") from error

    if not content:
        raise EmptyDocumentError("文档解析后没有可用文字")
    _validate_content_size(content, max_characters)
    if cache_result:
        await asyncio.to_thread(_write_cache, path, content)
    return content


__all__ = [
    "DOCUMENT_PARSE_TIMEOUT_SECONDS",
    "MAX_PARSED_DOCUMENT_CHARACTERS",
    "PARSED_CACHE_SUFFIX",
    "SUPPORTED_DOCUMENT_EXTENSIONS",
    "DocumentContentTooLargeError",
    "DocumentParseError",
    "DocumentParseTimeoutError",
    "EmptyDocumentError",
    "parse_local_document",
    "parsed_cache_path",
]
