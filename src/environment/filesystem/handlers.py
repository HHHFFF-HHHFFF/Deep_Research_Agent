from __future__ import annotations

import csv
import json
import os
import tempfile
from typing import ClassVar, Protocol

from markitdown import MarkItDown

from src.environment.filesystem.types import FileReadRequest, FileReadResult


class ContentHandler(Protocol):
    """定义 `ContentHandler`，封装相关数据与行为。"""

    extensions: set[str]

    async def decode(self, data: bytes, request: FileReadRequest) -> FileReadResult: ...

    async def encode(self, text: str | bytes, *, mode: str, encoding: str) -> bytes: ...


class TextHandler:
    """定义 `TextHandler`，封装相关数据与行为。"""

    extensions: ClassVar[set[str]] = {
        ".txt",
        ".md",
        ".py",
        ".log",
        ".cfg",
        ".ini",
        ".conf",
        ".yml",
        ".yaml",
        ".xml",
        ".html",
        ".css",
        ".js",
        ".ts",
        ".sh",
        ".bat",
        ".ps1",
    }

    async def decode(self, data: bytes, request: FileReadRequest) -> FileReadResult:
        """实现 `decode` 的业务逻辑。"""
        text = data.decode(request.encoding, errors="replace")
        total_lines: int | None = None
        content_text = text

        if request.start_line is not None or request.end_line is not None:
            lines = text.splitlines()
            total_lines = len(lines)
            start = (request.start_line - 1) if request.start_line else 0
            end = request.end_line if request.end_line else total_lines
            if start < 0 or end > total_lines or start >= end:
                # 组装并返回结果。
                content_text = ""
            else:
                content_text = "\n".join(lines[start:end])

        # 说明相关实现细节。
        preview = None
        if content_text:
            plines = content_text.splitlines()[:3]
            preview = "\n".join(plines)

        return FileReadResult(
            path=request.path,
            source="disk",
            content_bytes=None,
            content_text=content_text,
            total_lines=total_lines,
            preview=preview,
        )

    async def encode(self, text: str | bytes, *, mode: str, encoding: str) -> bytes:
        """实现 `encode` 的业务逻辑。"""
        if isinstance(text, bytes):
            return text
        return text.encode(encoding)


class JsonHandler(TextHandler):
    """定义 `JsonHandler`，封装相关数据与行为。"""

    extensions: ClassVar[set[str]] = {".json", ".jsonl"}

    async def decode(self, data: bytes, request: FileReadRequest) -> FileReadResult:
        """实现 `decode` 的业务逻辑。"""
        # 检索所需信息。
        base = await super().decode(data, request)
        try:
            text = data.decode(request.encoding, errors="replace")
            if request.path.suffix == ".jsonl":
                # 转换并规范化数据。
                lines = text.strip().split("\n")
                if lines and lines[0]:
                    first_obj = json.loads(lines[0])
                    base.preview = (
                        f"JSONL: {type(first_obj).__name__} with {len(lines)} lines"
                    )
            else:
                # 说明相关实现细节。
                obj = json.loads(text)
                if isinstance(obj, dict):
                    keys = list(obj.keys())[:5]
                    base.preview = f"JSON Object with keys: {', '.join(keys)}"
                elif isinstance(obj, list):
                    base.preview = f"JSON Array with {len(obj)} items"
                else:
                    base.preview = f"JSON {type(obj).__name__}"
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
            return base
        return base


class CsvHandler(TextHandler):
    """定义 `CsvHandler`，封装相关数据与行为。"""

    extensions: ClassVar[set[str]] = {".csv", ".tsv"}

    async def decode(self, data: bytes, request: FileReadRequest) -> FileReadResult:
        """实现 `decode` 的业务逻辑。"""
        base = await super().decode(data, request)
        try:
            text = data.decode(request.encoding, errors="replace")
            delimiter = "," if request.path.suffix == ".csv" else "\t"

            # 加载所需数据。
            lines = text.splitlines()[:3]
            if lines:
                reader = csv.reader(lines, delimiter=delimiter)
                rows = list(reader)
                if rows:
                    headers = rows[0] if len(rows) > 0 else []
                    row_count = len(text.splitlines())
                    base.preview = f"CSV: {len(headers)} columns, {row_count} rows. Headers: {', '.join(headers[:3])}"
        except (UnicodeDecodeError, csv.Error):
            return base
        return base


class BinaryHandler:
    """定义 `BinaryHandler`，封装相关数据与行为。"""

    extensions: ClassVar[set[str]] = {
        ".bin",
        ".dat",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".7z",
        ".rar",
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".tiff",
        ".ico",
        ".svg",
        ".mp3",
        ".mp4",
        ".avi",
        ".mov",
        ".wav",
        ".flac",
    }

    async def decode(self, data: bytes, request: FileReadRequest) -> FileReadResult:
        """实现 `decode` 的业务逻辑。"""
        # 说明相关实现细节。
        preview = f"Binary file ({len(data)} bytes): {data[:32].hex()}"
        return FileReadResult(
            path=request.path,
            source="disk",
            content_bytes=data,
            content_text=None,
            total_lines=None,
            preview=preview,
        )

    async def encode(self, text: str | bytes, *, mode: str, encoding: str) -> bytes:
        """实现 `encode` 的业务逻辑。"""
        if isinstance(text, bytes):
            return text
        return text.encode(encoding)


class MarkdownHandler(TextHandler):
    """定义 `MarkdownHandler`，封装相关数据与行为。"""

    extensions: ClassVar[set[str]] = {
        ".md",
        ".markdown",
        ".mdown",
        ".mkdn",
        ".mkd",
        ".mdwn",
        ".mdtxt",
        ".mdtext",
    }

    async def decode(self, data: bytes, request: FileReadRequest) -> FileReadResult:
        """实现 `decode` 的业务逻辑。"""
        base = await super().decode(data, request)
        try:
            text = data.decode(request.encoding, errors="replace")
            lines = text.splitlines()

            # 说明相关实现细节。
            headers = []
            for line in lines[:10]:  # 校验输入与当前状态。
                line = line.strip()
                if line.startswith("#"):
                    level = len(line) - len(line.lstrip("#"))
                    title = line.lstrip("#").strip()
                    headers.append(f"{'  ' * (level - 1)}- {title}")
                    if len(headers) >= 3:
                        break

            if headers:
                base.preview = "Markdown with headers:\n" + "\n".join(headers)
        except (UnicodeDecodeError, IndexError, ValueError):
            return base
        return base


class PythonHandler(TextHandler):
    """定义 `PythonHandler`，封装相关数据与行为。"""

    extensions: ClassVar[set[str]] = {".py", ".pyi", ".pyc", ".pyo"}

    async def decode(self, data: bytes, request: FileReadRequest) -> FileReadResult:
        """实现 `decode` 的业务逻辑。"""
        base = await super().decode(data, request)
        try:
            text = data.decode(request.encoding, errors="replace")
            lines = text.splitlines()

            # 说明相关实现细节。
            definitions = []
            for i, line in enumerate(lines[:20]):  # 校验输入与当前状态。
                line = line.strip()
                if line.startswith("class ") and ":" in line:
                    class_name = (
                        line.split("class ")[1].split("(")[0].split(":")[0].strip()
                    )
                    definitions.append(f"class {class_name}")
                elif line.startswith("def ") and ":" in line:
                    func_name = line.split("def ")[1].split("(")[0].strip()
                    definitions.append(f"def {func_name}")
                elif line.startswith("async def ") and ":" in line:
                    func_name = line.split("async def ")[1].split("(")[0].strip()
                    definitions.append(f"async def {func_name}")

                if len(definitions) >= 5:
                    break

            if definitions:
                base.preview = "Python code with:\n" + "\n".join(definitions[:5])
        except (UnicodeDecodeError, IndexError, ValueError):
            return base
        return base


class XlsxHandler:
    """定义 `XlsxHandler`，封装相关数据与行为。"""

    extensions: ClassVar[set[str]] = {".xlsx"}

    async def decode(self, data: bytes, request: FileReadRequest) -> FileReadResult:
        """实现 `decode` 的业务逻辑。"""
        temp_file_path = None
        try:
            # 持久化相关数据。
            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as temp_file:
                temp_file.write(data)
                temp_file_path = temp_file.name

            # 转换并规范化数据。
            md = MarkItDown()
            result = md.convert(temp_file_path)

            # 说明相关实现细节。
            markdown_content = result.text_content

            # 说明相关实现细节。
            preview_lines = markdown_content.splitlines()[:5]
            preview = "XLSX converted to Markdown:\n" + "\n".join(preview_lines)

            return FileReadResult(
                path=request.path,
                source="disk",
                content_bytes=None,
                content_text=markdown_content,
                total_lines=len(markdown_content.splitlines()),
                preview=preview,
            )
        except Exception as e:
            # 处理异常情况。
            preview = f"XLSX file ({len(data)} bytes) - conversion failed: {e!s}"
            return FileReadResult(
                path=request.path,
                source="disk",
                content_bytes=data,
                content_text=None,
                total_lines=None,
                preview=preview,
            )
        finally:
            # 处理文件与路径。
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    async def encode(self, text: str | bytes, *, mode: str, encoding: str) -> bytes:
        """实现 `encode` 的业务逻辑。"""
        raise NotImplementedError("XLSX encoding not supported")


class DocxHandler:
    """定义 `DocxHandler`，封装相关数据与行为。"""

    extensions: ClassVar[set[str]] = {".docx"}

    async def decode(self, data: bytes, request: FileReadRequest) -> FileReadResult:
        """实现 `decode` 的业务逻辑。"""
        temp_file_path = None
        try:
            # 持久化相关数据。
            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as temp_file:
                temp_file.write(data)
                temp_file_path = temp_file.name

            # 转换并规范化数据。
            md = MarkItDown()
            result = md.convert(temp_file_path)

            # 说明相关实现细节。
            markdown_content = result.text_content

            # 说明相关实现细节。
            preview_lines = markdown_content.splitlines()[:5]
            preview = "DOCX converted to Markdown:\n" + "\n".join(preview_lines)

            return FileReadResult(
                path=request.path,
                source="disk",
                content_bytes=None,
                content_text=markdown_content,
                total_lines=len(markdown_content.splitlines()),
                preview=preview,
            )
        except Exception as e:
            # 处理异常情况。
            preview = f"DOCX file ({len(data)} bytes) - conversion failed: {e!s}"
            return FileReadResult(
                path=request.path,
                source="disk",
                content_bytes=data,
                content_text=None,
                total_lines=None,
                preview=preview,
            )
        finally:
            # 处理文件与路径。
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    async def encode(self, text: str | bytes, *, mode: str, encoding: str) -> bytes:
        """实现 `encode` 的业务逻辑。"""
        raise NotImplementedError("DOCX encoding not supported")


class PdfHandler:
    """定义 `PdfHandler`，封装相关数据与行为。"""

    extensions: ClassVar[set[str]] = {".pdf"}

    async def decode(self, data: bytes, request: FileReadRequest) -> FileReadResult:
        """实现 `decode` 的业务逻辑。"""
        temp_file_path = None
        try:
            # 持久化相关数据。
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                temp_file.write(data)
                temp_file_path = temp_file.name

            # 转换并规范化数据。
            md = MarkItDown()
            result = md.convert(temp_file_path)

            # 说明相关实现细节。
            markdown_content = result.text_content

            # 说明相关实现细节。
            preview_lines = markdown_content.splitlines()[:5]
            preview = "PDF converted to Markdown:\n" + "\n".join(preview_lines)

            return FileReadResult(
                path=request.path,
                source="disk",
                content_bytes=None,
                content_text=markdown_content,
                total_lines=len(markdown_content.splitlines()),
                preview=preview,
            )
        except Exception as e:
            # 处理异常情况。
            preview = f"PDF file ({len(data)} bytes) - conversion failed: {e!s}"
            return FileReadResult(
                path=request.path,
                source="disk",
                content_bytes=data,
                content_text=None,
                total_lines=None,
                preview=preview,
            )
        finally:
            # 处理文件与路径。
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    async def encode(self, text: str | bytes, *, mode: str, encoding: str) -> bytes:
        """实现 `encode` 的业务逻辑。"""
        raise NotImplementedError("PDF encoding not supported")


class HandlerRegistry:
    """定义 `HandlerRegistry`，封装相关数据与行为。"""

    def __init__(self) -> None:
        self._handlers: list[ContentHandler] = []
        self._extension_map: dict[str, ContentHandler] = {}

    def register(self, handler: ContentHandler) -> None:
        """实现 `register` 的业务逻辑。"""
        self._handlers.append(handler)
        # 更新相关状态。
        for ext in handler.extensions:
            self._extension_map[ext.lower()] = handler

    def find_for_extension(self, suffix: str) -> ContentHandler | None:
        """实现 `find_for_extension` 的业务逻辑。"""
        return self._extension_map.get(suffix.lower())

    def get_all_handlers(self) -> list[ContentHandler]:
        """获取与 `get_all_handlers` 对应的数据或状态。"""
        return self._handlers.copy()

    def get_supported_extensions(self) -> set[str]:
        """获取与 `get_supported_extensions` 对应的数据或状态。"""
        return set(self._extension_map.keys())
