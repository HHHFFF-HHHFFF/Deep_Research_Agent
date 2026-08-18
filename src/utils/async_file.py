"""提供不会阻塞异步事件循环的常用文件读写函数。"""

import asyncio
import json
import pickle
from os import PathLike
from typing import Any, BinaryIO

import aiofiles

FilePath = str | PathLike[str]


async def read_text_file(
    file_path: FilePath,
    *,
    encoding: str = "utf-8",
    errors: str | None = None,
) -> str:
    """异步读取文本文件的全部内容。"""
    async with aiofiles.open(
        file_path,
        "r",
        encoding=encoding,
        errors=errors,
    ) as file:
        return await file.read()


async def read_lines_file(
    file_path: FilePath,
    *,
    encoding: str = "utf-8",
    errors: str | None = None,
) -> list[str]:
    """异步读取文本文件并保留每一行的换行符。"""
    content = await read_text_file(file_path, encoding=encoding, errors=errors)
    return content.splitlines(keepends=True)


async def write_text_file(
    file_path: FilePath,
    content: str,
    *,
    encoding: str = "utf-8",
) -> None:
    """异步覆盖写入文本文件。"""
    async with aiofiles.open(file_path, "w", encoding=encoding) as file:
        await file.write(content)


async def write_lines_file(
    file_path: FilePath,
    lines: list[str],
    *,
    encoding: str = "utf-8",
) -> None:
    """异步覆盖写入一组文本行。"""
    await write_text_file(file_path, "".join(lines), encoding=encoding)


async def read_json_file(file_path: FilePath) -> Any:
    """异步读取并解析 UTF-8 JSON 文件。"""
    return json.loads(await read_text_file(file_path))


async def write_json_file(
    file_path: FilePath,
    data: Any,
    *,
    indent: int = 4,
    ensure_ascii: bool = False,
) -> None:
    """把数据序列化为 JSON 后异步写入文件。"""
    content = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii)
    await write_text_file(file_path, content)


async def write_pickle_file(file_path: FilePath, data: Any) -> None:
    """异步写入 Pickle 二进制数据。"""
    payload = pickle.dumps(data)
    async with aiofiles.open(file_path, "wb") as file:
        await file.write(payload)


def _open_binary_file(file_path: FilePath) -> BinaryIO:
    """在线程中执行同步文件打开，并把句柄交给调用方管理。"""
    return open(file_path, "rb")


async def open_binary_file(file_path: FilePath) -> BinaryIO:
    """在线程池中打开二进制文件，避免阻塞事件循环。"""
    return await asyncio.to_thread(_open_binary_file, file_path)
