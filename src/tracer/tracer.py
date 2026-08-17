"""提供执行轨迹相关实现。"""

import json
import asyncio
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from src.utils import file_lock
from src.session import SessionContext


class Record(BaseModel):
    """定义 `Record`，封装相关数据与行为。"""

    id: Optional[int] = Field(default=None, description="Unique identifier for the record")
    session_id: Optional[str] = Field(default=None, description="Session ID for this record")
    task_id: Optional[str] = Field(default=None, description="Task ID for this record")
    observation: Optional[Any] = Field(default=None, description="Observation data for this execution step")
    tool: Optional[Any] = Field(default=None, description="Tool calls taken in this execution step")
    timestamp: Optional[str] = Field(default=None, description="Timestamp of the record in ISO format")


class SessionRecords:
    """定义 `SessionRecords`，封装相关数据与行为。"""

    def __init__(self):
        self.records: List[Record] = []
        self._next_id: int = 1

    def add_record(
        self,
        observation: Any,
        tool: Any = None,
        task_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> Record:
        """添加与 `add_record` 对应的数据或状态。"""
        if timestamp is None:
            timestamp = datetime.now()

        record = Record(
            id=self._next_id,
            task_id=task_id,
            observation=observation,
            tool=tool,
            timestamp=timestamp.isoformat(),
        )
        self._next_id += 1
        self.records.append(record)
        return record

    def get_records(self) -> List[Record]:
        """获取与 `get_records` 对应的数据或状态。"""
        return self.records.copy()

    def get_record(self, index: int) -> Optional[Record]:
        """获取与 `get_record` 对应的数据或状态。"""
        if 0 <= index < len(self.records):
            return self.records[index]
        return None

    def get_last_record(self) -> Optional[Record]:
        """获取与 `get_last_record` 对应的数据或状态。"""
        if len(self.records) > 0:
            return self.records[-1]
        return None

    def get_record_by_id(self, record_id: int) -> Optional[Record]:
        """获取与 `get_record_by_id` 对应的数据或状态。"""
        for record in self.records:
            if record.id == record_id:
                return record
        return None

    def get_records_by_task_id(self, task_id: str) -> List[Record]:
        """获取与 `get_records_by_task_id` 对应的数据或状态。"""
        return [r for r in self.records if r.task_id == task_id]

    def clear(self) -> None:
        """实现 `clear` 的业务逻辑。"""
        self.records.clear()
        self._next_id = 1

    def __len__(self) -> int:
        return len(self.records)


class Tracer:
    """定义 `Tracer`，封装相关数据与行为。"""

    def __init__(self):
        """初始化实例。"""
        # 处理记忆或缓存状态。
        self._session_records_cache: Dict[str, SessionRecords] = {}
        # 执行异步任务。
        self._session_locks: Dict[str, asyncio.Lock] = {}
        # 处理记忆或缓存状态。
        self._cache_lock = asyncio.Lock()
        # 说明相关实现细节。
        self._current_session_id: Optional[str] = None

    def _get_id_from_ctx(self, ctx: Optional[SessionContext]) -> Optional[str]:
        """实现 `_get_id_from_ctx` 的业务逻辑。"""
        if ctx is None:
            return None
        return ctx.id

    async def _get_or_create_session_records(self, id: str) -> tuple[SessionRecords, asyncio.Lock]:
        """实现 `_get_or_create_session_records` 的业务逻辑。"""
        async with self._cache_lock:
            if id not in self._session_locks:
                self._session_locks[id] = asyncio.Lock()

            if id not in self._session_records_cache:
                self._session_records_cache[id] = SessionRecords()

            return self._session_records_cache[id], self._session_locks[id]

    async def _cleanup_session_records(self, id: str) -> None:
        """实现 `_cleanup_session_records` 的业务逻辑。"""
        async with self._cache_lock:
            if id in self._session_records_cache:
                del self._session_records_cache[id]
            if id in self._session_locks:
                del self._session_locks[id]

    async def add_record(
        self,
        observation: Any,
        tool: Any = None,
        task_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        ctx: Optional[SessionContext] = None,
    ) -> None:
        """添加与 `add_record` 对应的数据或状态。"""
        if ctx is None:
            ctx = SessionContext()
        id = ctx.id

        session_records, session_lock = await self._get_or_create_session_records(id)
        async with session_lock:
            record = session_records.add_record(
                observation=observation,
                tool=tool,
                task_id=task_id,
                timestamp=timestamp,
            )
            record.session_id = id

        self._current_session_id = id

    async def get_records(self, ctx: Optional[SessionContext] = None) -> List[Record]:
        """获取与 `get_records` 对应的数据或状态。"""
        id = self._get_id_from_ctx(ctx)
        if id:
            session_records, session_lock = await self._get_or_create_session_records(id)
            async with session_lock:
                return session_records.get_records()

        # 组装并返回结果。
        all_records = []
        async with self._cache_lock:
            for session_records in self._session_records_cache.values():
                all_records.extend(session_records.get_records())
        return all_records

    async def get_record(
        self, index: int, ctx: Optional[SessionContext] = None
    ) -> Optional[Record]:
        """获取与 `get_record` 对应的数据或状态。"""
        id = self._get_id_from_ctx(ctx) or self._current_session_id
        if id is None:
            return None

        session_records, session_lock = await self._get_or_create_session_records(id)
        async with session_lock:
            return session_records.get_record(index)

    async def get_last_record(self, ctx: Optional[SessionContext] = None) -> Optional[Record]:
        """获取与 `get_last_record` 对应的数据或状态。"""
        id = self._get_id_from_ctx(ctx) or self._current_session_id
        if id is None:
            return None

        session_records, session_lock = await self._get_or_create_session_records(id)
        async with session_lock:
            return session_records.get_last_record()

    async def get_record_by_id(
        self, record_id: int, ctx: Optional[SessionContext] = None
    ) -> Optional[Record]:
        """获取与 `get_record_by_id` 对应的数据或状态。"""
        id = self._get_id_from_ctx(ctx)
        if id:
            session_records, session_lock = await self._get_or_create_session_records(id)
            async with session_lock:
                return session_records.get_record_by_id(record_id)

        async with self._cache_lock:
            for session_records in self._session_records_cache.values():
                record = session_records.get_record_by_id(record_id)
                if record:
                    return record
        return None

    async def get_records_by_task_id(
        self, task_id: str, ctx: Optional[SessionContext] = None
    ) -> List[Record]:
        """获取与 `get_records_by_task_id` 对应的数据或状态。"""
        id = self._get_id_from_ctx(ctx)
        if id:
            session_records, session_lock = await self._get_or_create_session_records(id)
            async with session_lock:
                return session_records.get_records_by_task_id(task_id)

        all_records = []
        async with self._cache_lock:
            for session_records in self._session_records_cache.values():
                all_records.extend(session_records.get_records_by_task_id(task_id))
        return all_records

    async def clear(self, ctx: Optional[SessionContext] = None) -> None:
        """实现 `clear` 的业务逻辑。"""
        id = self._get_id_from_ctx(ctx)
        if id:
            await self._cleanup_session_records(id)
            if id == self._current_session_id:
                self._current_session_id = None
        else:
            async with self._cache_lock:
                self._session_records_cache.clear()
                self._session_locks.clear()
            self._current_session_id = None

    async def save_to_json(self, file_path: str) -> None:
        """保存与 `save_to_json` 对应的数据或状态。"""
        file_path = str(file_path)

        async with file_lock(file_path):
            async with self._cache_lock:
                metadata = {
                    "current_session_id": self._current_session_id,
                    "session_ids": list(self._session_records_cache.keys()),
                }

                sessions = {}
                for session_id, session_records in self._session_records_cache.items():
                    sessions[session_id] = []
                    for record in session_records.records:
                        json_record = {
                            "id": record.id,
                            "session_id": record.session_id,
                            "task_id": record.task_id,
                            "observation": self._serialize_for_json(record.observation),
                            "tool": self._serialize_for_json(record.tool),
                            "timestamp": record.timestamp,
                        }
                        sessions[session_id].append(json_record)

            save_data = {"metadata": metadata, "sessions": sessions}

            parent_dir = os.path.dirname(file_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)

    async def load_from_json(self, file_path: str) -> None:
        """加载与 `load_from_json` 对应的数据或状态。"""
        file_path = str(file_path)

        async with file_lock(file_path):
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"JSON file not found: {file_path}")

            with open(file_path, "r", encoding="utf-8") as f:
                load_data = json.load(f)

            if not isinstance(load_data, dict) or "metadata" not in load_data or "sessions" not in load_data:
                raise ValueError(
                    f"Invalid tracer format. Expected {{'metadata': {{...}}, 'sessions': {{...}}}}, "
                    f"got: {type(load_data).__name__}"
                )

            metadata = load_data.get("metadata", {})
            self._current_session_id = metadata.get("current_session_id")

            async with self._cache_lock:
                self._session_records_cache.clear()
                self._session_locks.clear()

                sessions_data = load_data.get("sessions", {})
                for session_id, records_data in sessions_data.items():
                    self._session_locks[session_id] = asyncio.Lock()
                    session_records = SessionRecords()
                    max_id = 0
                    for json_record in records_data:
                        record = Record(
                            id=json_record.get("id"),
                            session_id=session_id,
                            task_id=json_record.get("task_id"),
                            observation=json_record.get("observation"),
                            tool=json_record.get("tool"),
                            timestamp=json_record.get("timestamp"),
                        )
                        session_records.records.append(record)
                        if record.id is not None and record.id > max_id:
                            max_id = record.id
                    session_records._next_id = max_id + 1 if max_id > 0 else 1
                    self._session_records_cache[session_id] = session_records

    def _serialize_for_json(self, obj: Any) -> Any:
        """实现 `_serialize_for_json` 的业务逻辑。"""
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, dict):
            return {k: self._serialize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_for_json(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return str(obj)

    def __len__(self) -> int:
        """实现 `__len__` 的业务逻辑。"""
        return sum(len(sr.records) for sr in self._session_records_cache.values())

    async def get_count(self, ctx: Optional[SessionContext] = None) -> int:
        """获取与 `get_count` 对应的数据或状态。"""
        id = self._get_id_from_ctx(ctx)
        if id:
            session_records, _ = await self._get_or_create_session_records(id)
            return len(session_records)
        return sum(len(sr.records) for sr in self._session_records_cache.values())

    async def get_session_ids(self) -> List[str]:
        """获取与 `get_session_ids` 对应的数据或状态。"""
        return list(self._session_records_cache.keys())

    def __repr__(self) -> str:
        total_records = sum(len(sr.records) for sr in self._session_records_cache.values())
        return f"Tracer(records={total_records}, sessions={len(self._session_records_cache)})"

    def __str__(self) -> str:
        return self.__repr__()
