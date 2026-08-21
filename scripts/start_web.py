"""构建前端并以单端口启动本地 Web 应用。"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_ROOT = PROJECT_ROOT / "frontend"
FRONTEND_INDEX = FRONTEND_ROOT / "dist" / "index.html"


def _build_frontend(pnpm_command: str) -> None:
    """使用项目锁定的 pnpm 依赖生成生产前端。"""
    if not (FRONTEND_ROOT / "node_modules").is_dir():
        raise RuntimeError("前端依赖尚未安装，请先在 frontend 目录执行 pnpm install")

    completed = subprocess.run(
        [pnpm_command, "build"],
        cwd=FRONTEND_ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("前端生产构建失败，请根据上方提示修复后重试")


def _require_frontend_build() -> None:
    """确保 FastAPI 启动前已经存在可托管的页面。"""
    if not FRONTEND_INDEX.is_file():
        raise RuntimeError("未找到前端生产构建，请先运行本脚本完成构建")


def _prepare_frontend(skip_build: bool) -> None:
    """优先重新构建前端，工具不可见时复用已有生产构建。"""
    if skip_build:
        _require_frontend_build()
        return

    pnpm_command = shutil.which("pnpm")
    if pnpm_command is None:
        if FRONTEND_INDEX.is_file():
            print("当前环境未找到 pnpm，已复用现有前端生产构建。")
            return
        raise RuntimeError(
            "未找到 pnpm，且没有可复用的前端生产构建；请先安装 Node.js 和 pnpm"
        )

    try:
        _build_frontend(pnpm_command)
    except RuntimeError as error:
        if FRONTEND_INDEX.is_file():
            print(f"无法重新构建前端（{error}），已复用现有生产构建。")
            return
        raise
    _require_frontend_build()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动深度研究智能体 Web 应用")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="复用已有前端生产构建",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="只检查生产构建，不启动服务",
    )
    return parser.parse_args()


def main() -> int:
    """执行前端构建检查，并启动单进程 Uvicorn。"""
    args = _parse_args()
    try:
        _prepare_frontend(args.skip_build)
    except RuntimeError as error:
        print(f"启动准备失败：{error}", file=sys.stderr)
        return 1

    if args.check:
        print("生产前端已就绪，FastAPI 可以进行单端口托管。")
        return 0

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    import uvicorn

    print(f"应用地址：http://{args.host}:{args.port}")
    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        workers=1,
        app_dir=str(PROJECT_ROOT),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
