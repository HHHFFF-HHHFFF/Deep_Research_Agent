from __future__ import annotations

from pathlib import Path

from src.environment.filesystem.exceptions import InvalidPathError, PathTraversalError


class PathPolicy:
    """定义 `PathPolicy`，封装相关数据与行为。"""

    def __init__(self, base_dir: Path) -> None:
        if not isinstance(base_dir, Path):
            raise InvalidPathError("base_dir must be a pathlib.Path instance")
        self._base_dir = base_dir.resolve()

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def to_relative(self, path: Path) -> Path:
        absolute = path.resolve()
        try:
            return absolute.relative_to(self._base_dir)
        except ValueError:
            # 组装并返回结果。
            # 处理文件与路径。
            return absolute

    def resolve_relative(self, relative: str | Path) -> Path:
        if isinstance(relative, str):
            relative = Path(relative)

        # 处理文件与路径。
        if relative.is_absolute():
            return relative.resolve()

        # 处理文件与路径。
        absolute = (self._base_dir / relative).resolve()
        if not str(absolute).startswith(str(self._base_dir)):
            # 校验输入与当前状态。
            raise PathTraversalError(
                f"Resolved path '{absolute}' escapes base_dir '{self._base_dir}'"
            )
        return absolute
