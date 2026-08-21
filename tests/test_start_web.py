"""统一 Web 启动脚本的离线测试。"""

from __future__ import annotations

import runpy
import shutil
from pathlib import Path
from typing import Any

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "start_web.py"


def _load_script() -> dict[str, Any]:
    """在不执行主函数的前提下加载启动脚本。"""
    return runpy.run_path(str(SCRIPT_PATH), run_name="start_web_test")


def test_existing_build_is_reused_when_pnpm_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """PyCharm 看不到 pnpm 时，已有生产构建仍应可以启动。"""
    namespace = _load_script()
    frontend_index = tmp_path / "dist" / "index.html"
    frontend_index.parent.mkdir()
    frontend_index.write_text("ready", encoding="utf-8")
    prepare_frontend = namespace["_prepare_frontend"]
    monkeypatch.setitem(prepare_frontend.__globals__, "FRONTEND_INDEX", frontend_index)
    monkeypatch.setattr(shutil, "which", lambda _: None)

    prepare_frontend(False)

    assert "已复用现有前端生产构建" in capsys.readouterr().out


def test_missing_pnpm_and_build_returns_actionable_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """全新环境缺少 pnpm 和构建时，应给出完整修复方向。"""
    namespace = _load_script()
    prepare_frontend = namespace["_prepare_frontend"]
    monkeypatch.setitem(
        prepare_frontend.__globals__,
        "FRONTEND_INDEX",
        tmp_path / "missing" / "index.html",
    )
    monkeypatch.setattr(shutil, "which", lambda _: None)

    with pytest.raises(RuntimeError, match="没有可复用的前端生产构建"):
        prepare_frontend(False)


def test_existing_build_is_reused_when_pnpm_cannot_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """pnpm 可见但无法调用 Node 时，也应复用已经验收的构建。"""
    namespace = _load_script()
    frontend_index = tmp_path / "dist" / "index.html"
    frontend_index.parent.mkdir()
    frontend_index.write_text("ready", encoding="utf-8")
    prepare_frontend = namespace["_prepare_frontend"]
    monkeypatch.setitem(prepare_frontend.__globals__, "FRONTEND_INDEX", frontend_index)
    monkeypatch.setattr(shutil, "which", lambda _: "pnpm.cmd")

    def fail_to_build(_: str) -> None:
        raise RuntimeError("Node.js 不可用")

    monkeypatch.setitem(
        prepare_frontend.__globals__,
        "_build_frontend",
        fail_to_build,
    )

    prepare_frontend(False)

    assert "无法重新构建前端" in capsys.readouterr().out
