"""真实文档格式到本地 FAISS RAG 的离线验收。"""

from __future__ import annotations

import asyncio
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src import document_parser
from src.document_parser import (
    DocumentContentTooLargeError,
    DocumentParseError,
    EmptyDocumentError,
    parse_local_document,
    parsed_cache_path,
)
from src.document_retriever import LocalDocumentRetriever


def _write_docx(path: Path, text: str = "DOCX 本地 RAG 证据") -> None:
    """使用标准库生成仅包含一段文字的最小 DOCX。"""
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""
    relationships = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
    document = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p><w:sectPr/></w:body>
</w:document>"""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", relationships)
        archive.writestr("word/document.xml", document)


def _write_pdf(path: Path, text: str = "PDF local RAG evidence") -> None:
    """生成带有一行可提取文字的最小 PDF。"""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(stream)).encode()
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
    ]
    content = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(content))
        content.extend(f"{index} 0 obj\n".encode())
        content.extend(obj)
        content.extend(b"\nendobj\n")
    xref_offset = len(content)
    content.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    content.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode())
    trailer = (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    )
    content.extend(trailer.encode())
    path.write_bytes(content)


def _real_document_samples(root: Path) -> list[Path]:
    markdown = root / "证据.md"
    text = root / "证据.txt"
    document = root / "证据.docx"
    pdf = root / "证据.pdf"
    markdown.write_text("# Markdown 本地 RAG 证据", encoding="utf-8")
    text.write_text("TXT 本地 RAG 证据", encoding="utf-8")
    _write_docx(document)
    _write_pdf(pdf)
    return [markdown, text, document, pdf]


@pytest.mark.asyncio
async def test_supported_real_formats_produce_nonempty_markdown(tmp_path: Path) -> None:
    """四种白名单格式都必须经过真实转换并生成解析缓存。"""
    for source in _real_document_samples(tmp_path):
        content = await parse_local_document(source, cache_result=True)
        assert content
        assert "RAG" in content
        assert parsed_cache_path(source).is_file()


@pytest.mark.asyncio
async def test_empty_and_invalid_documents_are_rejected(tmp_path: Path) -> None:
    """非零字节的空白文档和伪造格式不能进入 RAG。"""
    blank = tmp_path / "空白.txt"
    invalid = tmp_path / "伪造.pdf"
    blank.write_text("  \n\t", encoding="utf-8")
    invalid.write_bytes(b"not a pdf")

    with pytest.raises(EmptyDocumentError):
        await parse_local_document(blank)
    with pytest.raises(DocumentParseError):
        await parse_local_document(invalid)


@pytest.mark.asyncio
async def test_valid_parsed_cache_avoids_duplicate_conversion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """上传阶段的解析结果应在研究阶段直接复用。"""
    source = tmp_path / "缓存.md"
    source.write_text("缓存中的本地 RAG 证据", encoding="utf-8")
    first = await parse_local_document(source, cache_result=True)

    def unexpected_convert(_: Path) -> str:
        pytest.fail("存在有效解析缓存时不应重复转换")

    monkeypatch.setattr(document_parser, "_convert_document", unexpected_convert)
    second = await parse_local_document(source)
    assert second == first


@pytest.mark.asyncio
async def test_parsed_content_length_is_limited(tmp_path: Path) -> None:
    """解析文本限制应在建立大量 Embedding 前生效。"""
    source = tmp_path / "过长.txt"
    source.write_text("本地证据" * 100, encoding="utf-8")

    with pytest.raises(DocumentContentTooLargeError):
        await parse_local_document(source, max_characters=100)


def test_real_formats_flow_into_faiss_with_offline_embeddings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """真实格式应完成解析、去重、虚拟 Embedding 和 FAISS 检索。"""
    from src.environment.faiss import service as faiss_service_module

    async def fake_embedding(**kwargs: Any) -> SimpleNamespace:
        vectors = []
        for message in kwargs["messages"]:
            content = str(message.content)
            vectors.append([0.0, 1.0] if "PDF" in content else [1.0, 0.0])
        return SimpleNamespace(
            success=True,
            message="离线向量生成成功",
            extra=SimpleNamespace(data={"embeddings": vectors}),
        )

    monkeypatch.setattr(
        faiss_service_module.model_manager,
        "aembedding",
        fake_embedding,
    )

    async def run_pipeline() -> tuple[int, int, str]:
        sources = _real_document_samples(tmp_path / "documents")
        documents = [
            {"path": source.name, "content": await parse_local_document(source)}
            for source in sources
        ]
        retriever = LocalDocumentRetriever(base_dir=tmp_path / "faiss")
        first_count = await retriever.index_documents(documents)
        duplicate_count = await retriever.index_documents(documents)
        chunks = await retriever.retrieve("PDF RAG evidence", k=1)
        await retriever.close()
        return first_count, duplicate_count, chunks[0].source

    (tmp_path / "documents").mkdir()
    first_count, duplicate_count, source = asyncio.run(run_pipeline())

    assert first_count == 4
    assert duplicate_count == 0
    assert source == "证据.pdf"
    assert (tmp_path / "faiss" / "index.faiss").is_file()
    assert (tmp_path / "faiss" / "index.pkl").is_file()
