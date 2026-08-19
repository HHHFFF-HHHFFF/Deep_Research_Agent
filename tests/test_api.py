"""W2 FastAPI、SQLite 与单任务管理的离线测试。"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from src.api.app import ApiSettings, create_app
from src.api.database import ResearchDatabase
from src.api.models import TaskStatus
from src.application import (
    ResearchProgress,
    ResearchRequest,
    ResearchResult,
    ResearchStage,
)
from src.research_runner import (
    ProgressCallback,
    ResearchCancelledError,
    ResearchRunError,
)


async def _emit(
    callback: ProgressCallback | None,
    stage: ResearchStage,
    message: str,
) -> None:
    """让测试替身同时支持同步和异步进度回调。"""
    if callback is None:
        return
    result = callback(ResearchProgress(stage=stage, message=message))
    if inspect.isawaitable(result):
        await result


async def successful_runner(
    request: ResearchRequest,
    *,
    on_progress: ProgressCallback | None,
    cancel_event: asyncio.Event | None,
) -> ResearchResult:
    """不访问网络的成功研究替身。"""
    await _emit(on_progress, ResearchStage.RESEARCHING, "正在执行离线研究")
    return ResearchResult(
        task=request.task,
        report=f"# 离线研究报告\n\n主题：{request.task}",
        model_name=f"{request.model_provider}/{request.model_id}",
        session_id="session-offline",
        files=request.files,
    )


def _settings(tmp_path: Path, name: str = "web", max_bytes: int = 1024) -> ApiSettings:
    root = tmp_path / name
    return ApiSettings(
        database_path=root / "research.db",
        upload_dir=root / "uploads",
        report_dir=root / "reports",
        max_upload_bytes=max_bytes,
    )


def _task_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task": "调研本地 RAG 的稳定性",
        "model_provider": "qwen",
        "model_id": "qwen-plus",
        "file_ids": [],
    }
    payload.update(overrides)
    return payload


async def _wait_for_status(
    client: httpx.AsyncClient,
    task_id: str,
    expected: set[str],
    timeout: float = 3.0,
) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    for _ in range(max(1, int(timeout / 0.01))):
        response = await client.get(f"/api/tasks/{task_id}")
        assert response.status_code == 200
        latest = response.json()
        if latest["status"] in expected:
            return latest
        await asyncio.sleep(0.01)
    raise AssertionError(f"任务未在限定时间内进入状态 {expected}，最后状态：{latest}")


@asynccontextmanager
async def _app_client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    """显式运行 lifespan，并使用 HTTPX 异步访问 ASGI 应用。"""
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as async_client:
            yield async_client


@pytest_asyncio.fixture
async def client(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    """提供使用临时 SQLite 文件的成功 API。"""
    app = create_app(_settings(tmp_path), runner=successful_runner)
    async with _app_client(app) as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_health_docs_and_empty_task_list(client: httpx.AsyncClient) -> None:
    """服务启动后健康检查、接口文档和空列表应可访问。"""
    health = await client.get("/api/health")
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "database": "ok",
        "active_task_id": None,
    }
    assert (await client.get("/docs")).status_code == 200
    assert (await client.get("/api/tasks")).json() == {"items": []}


@pytest.mark.asyncio
async def test_successful_task_can_be_polled_and_downloaded(
    client: httpx.AsyncClient,
) -> None:
    """成功任务应持久化状态，并提供 Markdown 报告。"""
    created = await client.post("/api/tasks", json=_task_payload())
    assert created.status_code == 202
    task_id = created.json()["id"]

    completed = await _wait_for_status(client, task_id, {TaskStatus.SUCCEEDED.value})
    assert completed["stage"] == "completed"
    assert completed["actual_model_name"] == "qwen/qwen-plus"
    assert completed["report_available"] is True

    report = await client.get(f"/api/tasks/{task_id}/report")
    assert report.status_code == 200
    assert report.headers["content-type"].startswith("text/markdown")
    assert "# 离线研究报告" in report.text

    recent = (await client.get("/api/tasks", params={"limit": 1})).json()["items"]
    assert [item["id"] for item in recent] == [task_id]


@pytest.mark.asyncio
async def test_upload_is_validated_and_passed_as_server_path(
    tmp_path: Path,
) -> None:
    """合法上传只通过服务器生成的路径交给 W1 研究入口。"""
    captured: dict[str, Any] = {}
    received = asyncio.Event()

    async def capture_runner(
        request: ResearchRequest,
        *,
        on_progress: ProgressCallback | None,
        cancel_event: asyncio.Event | None,
    ) -> ResearchResult:
        captured["files"] = request.files
        received.set()
        return await successful_runner(
            request,
            on_progress=on_progress,
            cancel_event=cancel_event,
        )

    settings = _settings(tmp_path, name="upload")
    app = create_app(settings, runner=capture_runner)
    async with _app_client(app) as test_client:
        uploaded = await test_client.post(
            "/api/files",
            files={"file": ("资料.md", b"# local evidence", "text/markdown")},
        )
        assert uploaded.status_code == 201
        file_info = uploaded.json()
        assert file_info["name"] == "资料.md"
        assert "stored_path" not in file_info

        created = await test_client.post(
            "/api/tasks",
            json=_task_payload(file_ids=[file_info["id"]]),
        )
        assert created.status_code == 202
        await asyncio.wait_for(received.wait(), timeout=2)
        await _wait_for_status(
            test_client,
            created.json()["id"],
            {TaskStatus.SUCCEEDED.value},
        )

    server_path = Path(captured["files"][0]).resolve()
    assert server_path.parent == settings.upload_dir.resolve()
    assert server_path.name != "资料.md"
    assert server_path.read_text(encoding="utf-8") == "# local evidence"


@pytest.mark.parametrize(
    ("filename", "content", "error_code"),
    [
        ("../越界.pdf", b"content", "invalid_file_name"),
        ("脚本.exe", b"content", "unsupported_file"),
        ("空文件.md", b"", "empty_file"),
        ("超限.md", b"12345", "file_too_large"),
    ],
)
@pytest.mark.asyncio
async def test_invalid_uploads_return_stable_errors(
    tmp_path: Path,
    filename: str,
    content: bytes,
    error_code: str,
) -> None:
    """非法文件不能落盘，也不能泄露服务器路径。"""
    settings = _settings(tmp_path, name=error_code, max_bytes=4)
    app = create_app(settings, runner=successful_runner)
    async with _app_client(app) as test_client:
        response = await test_client.post(
            "/api/files",
            files={"file": (filename, content, "application/octet-stream")},
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == error_code
    assert list(settings.upload_dir.glob("*")) == []


@pytest.mark.asyncio
async def test_only_one_task_runs_and_active_task_can_be_cancelled(
    tmp_path: Path,
) -> None:
    """活动任务期间重复创建应冲突，取消后任务进入终态。"""
    started = asyncio.Event()

    async def blocking_runner(
        request: ResearchRequest,
        *,
        on_progress: ProgressCallback | None,
        cancel_event: asyncio.Event | None,
    ) -> ResearchResult:
        assert cancel_event is not None
        started.set()
        await cancel_event.wait()
        raise ResearchCancelledError("研究任务已取消")

    app = create_app(_settings(tmp_path, name="cancel"), runner=blocking_runner)
    async with _app_client(app) as test_client:
        first = await test_client.post("/api/tasks", json=_task_payload())
        assert first.status_code == 202
        task_id = first.json()["id"]
        await asyncio.wait_for(started.wait(), timeout=2)

        duplicate = await test_client.post("/api/tasks", json=_task_payload())
        assert duplicate.status_code == 409
        assert duplicate.json()["error"]["code"] == "task_busy"

        cancelled = await test_client.post(f"/api/tasks/{task_id}/cancel")
        assert cancelled.status_code == 200
        final = await _wait_for_status(
            test_client,
            task_id,
            {TaskStatus.CANCELLED.value},
        )
        assert final["stage"] == "cancelled"
        assert final["report_available"] is False


@pytest.mark.asyncio
async def test_runner_failure_is_persisted_without_report(tmp_path: Path) -> None:
    """W1 稳定错误应写入任务状态，失败任务不能下载报告。"""

    async def failing_runner(
        request: ResearchRequest,
        *,
        on_progress: ProgressCallback | None,
        cancel_event: asyncio.Event | None,
    ) -> ResearchResult:
        raise ResearchRunError("模型服务暂时不可用")

    app = create_app(_settings(tmp_path, name="failure"), runner=failing_runner)
    async with _app_client(app) as test_client:
        created = await test_client.post("/api/tasks", json=_task_payload())
        task_id = created.json()["id"]
        failed = await _wait_for_status(
            test_client,
            task_id,
            {TaskStatus.FAILED.value},
        )
        assert failed["error_message"] == "模型服务暂时不可用"
        report = await test_client.get(f"/api/tasks/{task_id}/report")
        assert report.status_code == 409
        assert report.json()["error"]["code"] == "report_unavailable"


@pytest.mark.asyncio
async def test_unexpected_runner_error_is_sanitized(tmp_path: Path) -> None:
    """未知运行异常不能把令牌或底层详情写入任务响应。"""

    async def unexpected_runner(
        request: ResearchRequest,
        *,
        on_progress: ProgressCallback | None,
        cancel_event: asyncio.Event | None,
    ) -> ResearchResult:
        raise RuntimeError("Authorization: Bearer secret-token")

    app = create_app(_settings(tmp_path, name="unexpected"), runner=unexpected_runner)
    async with _app_client(app) as test_client:
        created = await test_client.post("/api/tasks", json=_task_payload())
        failed = await _wait_for_status(
            test_client,
            created.json()["id"],
            {TaskStatus.FAILED.value},
        )
        assert failed["error_message"] == "研究运行失败，请查看本地日志"
        assert "secret-token" not in str(failed)


@pytest.mark.asyncio
async def test_unknown_file_and_validation_errors_are_unified(
    client: httpx.AsyncClient,
) -> None:
    """不存在的文件和非法请求应使用统一错误结构。"""
    unknown = await client.post(
        "/api/tasks",
        json=_task_payload(file_ids=["11111111-1111-1111-1111-111111111111"]),
    )
    assert unknown.status_code == 400
    assert unknown.json()["error"]["code"] == "unknown_file"

    invalid = await client.post("/api/tasks", json=_task_payload(task="   "))
    assert invalid.status_code == 422
    assert invalid.json() == {
        "error": {"code": "validation_error", "message": "请求参数无效"}
    }


@pytest.mark.asyncio
async def test_startup_marks_stale_running_task_interrupted(tmp_path: Path) -> None:
    """服务重启后不能让旧运行状态永久停留。"""
    settings = _settings(tmp_path, name="restart")
    database = ResearchDatabase(settings.database_path)
    database.initialize()
    stale = database.create_task(
        task_id="22222222-2222-2222-2222-222222222222",
        task="调研服务重启恢复",
        model_provider="qwen",
        model_id="qwen-plus",
        file_ids=[],
    )
    database.mark_running(stale.id)
    database.dispose()

    app = create_app(settings, runner=successful_runner)
    async with _app_client(app) as test_client:
        response = await test_client.get(f"/api/tasks/{stale.id}")
        assert response.status_code == 200
        recovered = response.json()
        assert recovered["status"] == TaskStatus.INTERRUPTED.value
    assert recovered["stage"] == "interrupted"
    assert recovered["finished_at"] is not None


@pytest.mark.asyncio
async def test_shutdown_marks_active_task_interrupted(tmp_path: Path) -> None:
    """应用正常关闭时，尚未结束的研究任务应持久化为已中断。"""
    started = asyncio.Event()

    async def blocking_runner(
        request: ResearchRequest,
        *,
        on_progress: ProgressCallback | None,
        cancel_event: asyncio.Event | None,
    ) -> ResearchResult:
        assert cancel_event is not None
        started.set()
        await cancel_event.wait()
        raise ResearchCancelledError("研究任务已取消")

    settings = _settings(tmp_path, name="shutdown")
    app = create_app(settings, runner=blocking_runner)
    async with _app_client(app) as test_client:
        created = await test_client.post("/api/tasks", json=_task_payload())
        task_id = created.json()["id"]
        await asyncio.wait_for(started.wait(), timeout=2)

    database = ResearchDatabase(settings.database_path)
    database.initialize()
    interrupted = database.get_task(task_id)
    database.dispose()
    assert interrupted is not None
    assert interrupted.status is TaskStatus.INTERRUPTED
