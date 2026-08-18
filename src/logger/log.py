import logging
import threading
from collections.abc import Mapping
from contextlib import ExitStack
from enum import IntEnum
from queue import Empty, Full, Queue
from types import TracebackType
from typing import Any

from rich.console import Console, Group
from rich.logging import RichHandler
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

YELLOW_HEX = "#d4b702"


class LogLevel(IntEnum):
    CRITICAL = logging.CRITICAL
    FATAL = logging.FATAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    WARN = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG


class Logger(logging.Logger):
    """定义 `Logger`，封装相关数据与行为。"""

    def __init__(self, name="logger", level=logging.INFO):
        # 初始化相关状态。
        super().__init__(name, level)

        # 转换并规范化数据。
        self.formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s:%(levelname)s - %(filename)s:%(lineno)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # 执行异步任务。
        self._log_queue: Queue[str | None] | None = None
        self._log_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._log_path: str | None = None
        self._file_stack = ExitStack()
        self._initialized = False

    def _log_writer_thread(self, log_path: str):
        """实现 `_log_writer_thread` 的业务逻辑。"""
        log_queue = self._log_queue
        if log_queue is None:
            return
        with open(log_path, "a", encoding="utf-8") as log_file:
            while not self._stop_event.is_set():
                try:
                    # 说明相关实现细节。
                    log_entry = log_queue.get(timeout=0.1)
                    if log_entry is None:  # 说明相关实现细节。
                        break

                    # 持久化相关数据。
                    log_file.write(log_entry)
                    log_file.flush()  # 持久化相关数据。
                    log_queue.task_done()

                except Empty:
                    continue
                except Exception as e:
                    # 持久化相关数据。
                    import sys

                    print(f"Logger write error: {e}", file=sys.stderr)

    def _enqueue_log(self, level: str, msg: str, *args, **kwargs):
        """实现 `_enqueue_log` 的业务逻辑。"""
        if not self._initialized or self._log_queue is None:
            # 初始化相关状态。
            return

        try:
            # 转换并规范化数据。
            record = self.makeRecord(
                self.name, getattr(logging, level.upper()), "", 0, msg, args, None
            )
            formatted = self.formatter.format(record)

            # 说明相关实现细节。
            try:
                self._log_queue.put_nowait(formatted + "\n")
            except Full:
                # 说明相关实现细节。
                try:
                    self._log_queue.get_nowait()
                    self._log_queue.put_nowait(formatted + "\n")
                except (Empty, Full):
                    return

        except (AttributeError, TypeError, ValueError):
            return

    def initialize(self, config, level: int = LogLevel.INFO):
        """初始化组件及其依赖资源。"""

        log_path = config.log_path
        self._log_path = log_path

        self.handlers.clear()

        # 组装并返回结果。
        self.console = Console(
            width=None, markup=True, color_system="truecolor", force_terminal=True
        )
        rich_handler = RichHandler(
            console=self.console,
            rich_tracebacks=True,
            show_time=False,
            show_level=False,
            show_path=False,
            markup=True,
            omit_repeated_times=False,
        )
        rich_handler.setLevel(level)
        rich_handler.setFormatter(self.formatter)
        self.addHandler(rich_handler)

        # 组装并返回结果。
        self._log_queue = Queue(maxsize=1000)  # 处理记忆或缓存状态。
        self._stop_event.clear()

        # 加载所需数据。
        self._log_thread = threading.Thread(
            target=self._log_writer_thread,
            args=(log_path,),
            daemon=True,
            name="Logger-Writer",
        )
        self._log_thread.start()

        # 处理文件与路径。
        self._file_stack.close()
        self._file_stack = ExitStack()
        log_file = self._file_stack.enter_context(
            open(log_path, "a", encoding="utf-8")  # noqa: SIM115 - 文件句柄由 ExitStack 统一关闭
        )
        self.file_console = Console(
            width=None,
            markup=True,
            color_system="truecolor",
            force_terminal=True,
            file=log_file,
        )
        rich_file_handler = RichHandler(
            console=self.file_console,
            rich_tracebacks=True,
            show_time=False,
            show_level=False,
            show_path=False,
            markup=True,
            omit_repeated_times=False,
        )
        rich_file_handler.setLevel(level)
        rich_file_handler.setFormatter(self.formatter)
        self.addHandler(rich_file_handler)

        self.propagate = False
        self._initialized = True

    def info(self, msg, *args, **kwargs):
        """实现 `info` 的业务逻辑。"""
        kwargs.setdefault("stacklevel", 2)

        if "style" in kwargs:
            kwargs.pop("style")
        if "level" in kwargs:
            kwargs.pop("level")

        # 组装并返回结果。
        super().info(msg, *args, **kwargs)

        # 组装并返回结果。
        self._enqueue_log("info", msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """实现 `warning` 的业务逻辑。"""
        kwargs.setdefault("stacklevel", 2)
        super().warning(msg, *args, **kwargs)
        self._enqueue_log("warning", msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """实现 `error` 的业务逻辑。"""
        kwargs.setdefault("stacklevel", 2)
        super().error(msg, *args, **kwargs)
        self._enqueue_log("error", msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """实现 `critical` 的业务逻辑。"""
        kwargs.setdefault("stacklevel", 2)
        super().critical(msg, *args, **kwargs)
        self._enqueue_log("critical", msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        """实现 `debug` 的业务逻辑。"""
        kwargs.setdefault("stacklevel", 2)
        super().debug(msg, *args, **kwargs)
        self._enqueue_log("debug", msg, *args, **kwargs)

    def log(
        self,
        level: int,
        msg: object,
        *args: object,
        exc_info: (
            bool
            | tuple[type[BaseException], BaseException, TracebackType | None]
            | tuple[None, None, None]
            | BaseException
            | None
        ) = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        """按照标准库日志接口记录普通消息。"""
        super().log(
            level,
            msg,
            *args,
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel,
            extra=extra,
        )

    def render(
        self, content: Group | Panel | Rule | Syntax | Table | Tree, **kwargs: Any
    ) -> None:
        """向终端和日志文件渲染 Rich 内容。"""
        if hasattr(self, "console"):
            self.console.print(content, **kwargs)
        if hasattr(self, "file_console"):
            self.file_console.print(content, **kwargs)

    def shutdown(self):
        """实现 `shutdown` 的业务逻辑。"""
        if self._log_thread and self._log_thread.is_alive():
            # 说明相关实现细节。
            if self._log_queue:
                self._log_queue.put(None)
            self._stop_event.set()
            self._log_thread.join(timeout=5)  # 说明相关实现细节。
        self._file_stack.close()


logger = Logger()
