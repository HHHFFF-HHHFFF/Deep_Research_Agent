import asyncio
import os
from datetime import datetime, timezone
from typing import Any

from src.utils.singleton import Singleton


def format_size(size_bytes: int) -> str:
    """格式化与 `format_size` 对应的数据或状态。"""
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1

    return f"{size_bytes:.1f} {size_names[i]}"


def get_file_info(file_path: str) -> dict[str, Any]:
    """获取与 `get_file_info` 对应的数据或状态。"""
    abs_path = os.path.abspath(file_path)

    info = {}
    file_stats = os.stat(abs_path)

    info["path"] = abs_path
    info["size"] = format_size(file_stats.st_size)
    info["created"] = datetime.fromtimestamp(
        file_stats.st_ctime, timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S")
    info["modified"] = datetime.fromtimestamp(
        file_stats.st_mtime, timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S")
    info["accessed"] = datetime.fromtimestamp(
        file_stats.st_atime, timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S")
    info["permissions"] = oct(file_stats.st_mode)[-3:]
    info["is_directory"] = os.path.isdir(abs_path)
    info["is_file"] = os.path.isfile(abs_path)
    info["is_symlink"] = os.path.islink(abs_path)

    return info


class FileLock(metaclass=Singleton):
    def __init__(self):
        self._locks = {}

    def get_lock(self, key: str) -> asyncio.Lock:
        # 转换并规范化数据。
        key_str = str(key) if not isinstance(key, str) else key
        if key_str not in self._locks:
            self._locks[key_str] = asyncio.Lock()
        return self._locks[key_str]

    def __call__(self, key):
        return _FileLockContext(self.get_lock(key))


class _FileLockContext:
    def __init__(self, lock):
        self._lock = lock

    async def __aenter__(self):
        await self._lock.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._lock.release()


file_lock = FileLock()
