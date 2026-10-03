"""A3 评测命令行入口的返回码与报告输出测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import evaluate_agent_results, evaluate_rag

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_agent_evaluation_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    output = tmp_path / "agent.json"

    exit_code = evaluate_agent_results.main(
        [
            "--input",
            str(PROJECT_ROOT / "evals" / "agent_results.example.json"),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True
    assert output.with_suffix(".md").is_file()


def test_rag_evaluation_cli_writes_reproducible_report(tmp_path: Path) -> None:
    output = tmp_path / "rag.json"

    exit_code = evaluate_rag.main(
        [
            "--input",
            str(PROJECT_ROOT / "evals" / "rag_cases.json"),
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["passed"] is True
    assert payload["semantic_embedding_evaluated"] is False
    assert output.with_suffix(".md").is_file()


def test_agent_evaluation_cli_returns_two_for_invalid_input(tmp_path: Path) -> None:
    source = tmp_path / "invalid.json"
    source.write_text("not-json", encoding="utf-8")

    exit_code = evaluate_agent_results.main(
        ["--input", str(source), "--output", str(tmp_path / "report.json")]
    )

    assert exit_code == 2


def test_agent_validation_error_does_not_echo_sensitive_input(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = json.loads(
        (PROJECT_ROOT / "evals" / "agent_results.example.json").read_text(
            encoding="utf-8"
        )
    )
    secret = "secret-api-token-must-not-be-printed"
    payload["cases"][0]["snapshot"]["activities"][0]["raw_output"] = secret
    source = tmp_path / "sensitive.json"
    source.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    exit_code = evaluate_agent_results.main(
        ["--input", str(source), "--output", str(tmp_path / "report.json")]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert secret not in captured.err
    assert "输入结构不符合要求" in captured.err


def test_agent_evaluation_cli_returns_one_when_gate_fails(tmp_path: Path) -> None:
    payload = json.loads(
        (PROJECT_ROOT / "evals" / "agent_results.example.json").read_text(
            encoding="utf-8"
        )
    )
    payload["cases"][0]["snapshot"]["status"] = "failed"
    source = tmp_path / "failed-gate.json"
    source.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    output = tmp_path / "failed-report.json"

    exit_code = evaluate_agent_results.main(
        ["--input", str(source), "--output", str(output)]
    )

    assert exit_code == 1
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is False
