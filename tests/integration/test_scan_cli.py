"""Integration tests for the Atoll scan command."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import cast

import pytest

from atoll.cli import main
from atoll.report import ScanReport

FIXTURE_ROOT = Path("tests/fixtures/simple_project")
EXPECTED_MODULE_COUNT = 3
REPORT_SCHEMA_VERSION = 2


def test_scan_command_writes_reports(tmp_path: Path) -> None:
    """`atoll scan` writes JSON and Markdown reports under `.atoll`."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)

    exit_code = main(["scan", str(project_root)])
    second_exit_code = main(["scan", str(project_root)])

    report_path = project_root / ".atoll" / "report.json"
    markdown_path = project_root / ".atoll" / "report.md"
    report = cast(ScanReport, json.loads(report_path.read_text(encoding="utf-8")))
    assert exit_code == 0
    assert second_exit_code == 0
    assert report["version"] == REPORT_SCHEMA_VERSION
    assert report["tool"] == "atoll"
    assert report["summary"]["modules_scanned"] == EXPECTED_MODULE_COUNT
    assert report["summary"]["island_candidates"] == 1
    assert report["summary"]["typed_regions"] >= 1
    assert report["summary"]["hard_blockers"] == 1
    assert report["modules"][1]["island_candidates"][0]["risk"] == "low"
    assert report["modules"][1]["typed_regions"]
    binding = report["modules"][1]["typed_regions"][0]["bindings"][0]
    assert binding["source"].startswith("app.ranking::")
    assert binding["kind"] == "module"
    assert binding["execution_kind"] == "sync"
    assert binding["required"] is True
    candidate = report["modules"][1]["island_candidates"][0]
    markdown = markdown_path.read_text(encoding="utf-8")
    assert candidate["score_summary"].endswith("very promising scan-only candidate")
    assert candidate["risk_summary"].startswith("low extraction risk")
    assert markdown.startswith("# Atoll Scan Report")
    assert "Score is a 0-100 heuristic" in markdown
    assert "low extraction risk" in markdown


def test_cli_without_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Calling the CLI without a subcommand returns a usage error."""
    exit_code = main([])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "usage: atoll" in captured.out
