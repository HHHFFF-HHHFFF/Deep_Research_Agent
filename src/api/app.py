"""FastAPI 应用、文件上传和研究任务接口。"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import aiofiles
from fastapi import FastAPI, File, Query, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.database import ResearchDatabase, StoredFileRecord, TaskRecord
from src.api.models import (
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    TaskCreateRequest,
    TaskListResponse,
    TaskResponse,
    TaskStatus,
    UploadedFileResponse,
)
from src.api.task_manager import (
    ResearchRunner,
    ResearchTaskManager,
    TaskBusyError,
    TaskNotFoundError,
    TaskStateError,
    UnknownFileError,
)
from src.research_runner import run_research

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEB_ROOT = PROJECT_ROOT / "workdir" / "web"


@dataclass(frozen=True)
class ApiSettings:
    """W2 后端保持可测试所需的少量路径与上传限制。"""

    database_path: Path = DEFAULT_WEB_ROOT / "research.db"
    upload_dir: Path = DEFAULT_WEB_ROOT / "uploads"
    report_dir: Path = DEFAULT_WEB_ROOT / "reports"
    max_upload_bytes: int = 10 * 1024 * 1024
    upload_extensions: frozenset[str] = frozenset({".md", ".txt", ".pdf", ".docx"})


class ApiError(RuntimeError):
    """可以转换为统一 JSON 的安全接口错误。"""

    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    payload = ErrorResponse(error=ErrorDetail(code=code, message=message))
    return JSONResponse(status_code=status_code, content=payload.model_dump())


def _file_response(record: StoredFileRecord) -> UploadedFileResponse:
    return UploadedFileResponse(
        id=record.id,
        name=record.original_name,
        size=record.size,
        created_at=record.created_at,
    )


def _task_response(
    database: ResearchDatabase,
    record: TaskRecord,
) -> TaskResponse:
    files = database.get_files(record.file_ids)
    return TaskResponse(
        id=record.id,
        task=record.task,
        model_provider=record.model_provider,
        model_id=record.model_id,
        actual_model_name=record.actual_model_name,
        status=record.status,
        stage=record.stage,
        message=record.message,
        error_message=record.error_message,
        files=[_file_response(file) for file in files],
        report_available=(
            record.status is TaskStatus.SUCCEEDED and record.report_path is not None
        ),
        created_at=record.created_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
    )


def create_app(
    settings: ApiSettings | None = None,
    runner: ResearchRunner = run_research,
) -> FastAPI:
    """创建可在生产和离线测试中使用的 FastAPI 应用。"""
    app_settings = settings or ApiSettings()
    database = ResearchDatabase(app_settings.database_path)
    task_manager = ResearchTaskManager(
        database=database,
        report_dir=app_settings.report_dir,
        runner=runner,
    )
    application_logger = logging.getLogger(__name__)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        app_settings.upload_dir.mkdir(parents=True, exist_ok=True)
        app_settings.report_dir.mkdir(parents=True, exist_ok=True)
        database.initialize()
        interrupted_count = database.mark_active_tasks_interrupted()
        if interrupted_count:
            application_logger.warning(
                "应用启动时将 %s 个遗留任务标记为已中断",
                interrupted_count,
            )
        yield
        await task_manager.shutdown()
        database.dispose()

    app = FastAPI(
        title="深度研究智能体 API",
        description="本地单用户深度研究助手的任务与报告接口。",
        version="0.2.0",
        lifespan=lifespan,
    )
    app.state.settings = app_settings
    app.state.database = database
    app.state.task_manager = task_manager

    @app.exception_handler(ApiError)
    async def handle_api_error(_: Request, error: ApiError) -> JSONResponse:
        return _error_response(error.status_code, error.code, error.message)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _: Request,
        __: RequestValidationError,
    ) -> JSONResponse:
        return _error_response(422, "validation_error", "请求参数无效")

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(
        _: Request,
        error: StarletteHTTPException,
    ) -> JSONResponse:
        message = "请求的接口不存在" if error.status_code == 404 else "请求无法处理"
        return _error_response(error.status_code, "http_error", message)

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, error: Exception) -> JSONResponse:
        application_logger.error("接口发生未处理异常：%s", type(error).__name__)
        return _error_response(500, "internal_error", "服务内部错误")

    @app.get("/api/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        database.ping()
        return HealthResponse(active_task_id=task_manager.active_task_id)

    @app.post(
        "/api/files",
        response_model=UploadedFileResponse,
        status_code=status.HTTP_201_CREATED,
        responses={400: {"model": ErrorResponse}},
    )
    async def upload_file(
        file: Annotated[UploadFile, File(description="本次研究使用的本地资料")],
    ) -> UploadedFileResponse:
        original_name = (file.filename or "").strip()
        if (
            not original_name
            or "/" in original_name
            or "\\" in original_name
            or Path(original_name).name != original_name
        ):
            raise ApiError(400, "invalid_file_name", "文件名无效")

        extension = Path(original_name).suffix.lower()
        if extension not in app_settings.upload_extensions:
            raise ApiError(400, "unsupported_file", "暂不支持该文件格式")

        file_id = str(uuid4())
        stored_path = (app_settings.upload_dir / f"{file_id}{extension}").resolve()
        upload_root = app_settings.upload_dir.resolve()
        if stored_path.parent != upload_root:
            raise ApiError(400, "invalid_file_name", "文件名无效")

        size = 0
        try:
            async with aiofiles.open(stored_path, "wb") as output:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > app_settings.max_upload_bytes:
                        raise ApiError(400, "file_too_large", "上传文件超过大小限制")
                    await output.write(chunk)
            if size == 0:
                raise ApiError(400, "empty_file", "上传文件不能为空")
            record = database.create_file(
                file_id=file_id,
                original_name=original_name,
                stored_path=str(stored_path),
                size=size,
            )
        except Exception:
            stored_path.unlink(missing_ok=True)
            raise
        finally:
            await file.close()
        return _file_response(record)

    @app.post(
        "/api/tasks",
        response_model=TaskResponse,
        status_code=status.HTTP_202_ACCEPTED,
        responses={400: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    )
    async def create_task(request: TaskCreateRequest) -> TaskResponse:
        try:
            record = await task_manager.create_task(request)
        except TaskBusyError as error:
            raise ApiError(409, "task_busy", str(error)) from error
        except TaskStateError as error:
            raise ApiError(409, "invalid_task_state", str(error)) from error
        except UnknownFileError as error:
            raise ApiError(400, "unknown_file", str(error)) from error
        return _task_response(database, record)

    @app.get("/api/tasks", response_model=TaskListResponse)
    async def list_tasks(
        limit: Annotated[int, Query(ge=1, le=50)] = 20,
    ) -> TaskListResponse:
        records = database.list_tasks(limit)
        return TaskListResponse(
            items=[_task_response(database, record) for record in records]
        )

    @app.get(
        "/api/tasks/{task_id}",
        response_model=TaskResponse,
        responses={404: {"model": ErrorResponse}},
    )
    async def get_task(task_id: str) -> TaskResponse:
        record = database.get_task(task_id)
        if record is None:
            raise ApiError(404, "task_not_found", "研究任务不存在")
        return _task_response(database, record)

    @app.post(
        "/api/tasks/{task_id}/cancel",
        response_model=TaskResponse,
        responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    )
    async def cancel_task(task_id: str) -> TaskResponse:
        try:
            record = await task_manager.cancel_task(task_id)
        except TaskNotFoundError as error:
            raise ApiError(404, "task_not_found", str(error)) from error
        except TaskStateError as error:
            raise ApiError(409, "invalid_task_state", str(error)) from error
        return _task_response(database, record)

    @app.get(
        "/api/tasks/{task_id}/report",
        response_class=FileResponse,
        responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    )
    async def get_report(task_id: str) -> FileResponse:
        record = database.get_task(task_id)
        if record is None:
            raise ApiError(404, "task_not_found", "研究任务不存在")
        if record.status is not TaskStatus.SUCCEEDED or not record.report_path:
            raise ApiError(409, "report_unavailable", "研究报告尚不可用")

        report_root = app_settings.report_dir.resolve()
        report_path = Path(record.report_path).resolve()
        if report_path.parent != report_root or not report_path.is_file():
            raise ApiError(404, "report_not_found", "研究报告文件不存在")
        return FileResponse(
            path=report_path,
            media_type="text/markdown; charset=utf-8",
            filename=f"research-{task_id}.md",
        )

    return app


app = create_app()

__all__ = ["ApiSettings", "app", "create_app"]
