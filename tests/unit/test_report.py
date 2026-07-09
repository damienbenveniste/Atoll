"""Tests for user-facing scan report wording."""

from __future__ import annotations

from pathlib import Path

import pytest

from atoll.models import (
    CompileAttempt,
    EnabledIslandConfig,
    IslandRisk,
    PytestRunResult,
    VerifyResult,
)
from atoll.report import (
    CompilationReportInput,
    build_compilation_report,
    render_compilation_markdown_report,
    risk_summary,
    score_label,
    score_summary,
)


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (90, "strong"),
        (80, "good"),
        (70, "possible"),
        (69, "weak"),
    ],
)
def test_score_label_describes_recommendation_band(score: int, expected: str) -> None:
    """Candidate scores are grouped into stable display bands."""
    assert score_label(score) == expected


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (90, "90/100, very promising scan-only candidate"),
        (80, "80/100, promising scan-only candidate"),
        (70, "70/100, worth trying, but less compelling"),
        (60, "60/100, below Atoll's normal recommendation threshold"),
    ],
)
def test_score_summary_explains_recommendation_band(score: int, expected: str) -> None:
    """Candidate score summaries include the numeric score and its meaning."""
    assert score_summary(score) == expected


@pytest.mark.parametrize(
    ("risk", "expected"),
    [
        ("low", "low extraction risk; only high-confidence internal dependencies were seen"),
        (
            "medium",
            "medium extraction risk; a low-confidence dependency needs trial validation",
        ),
        ("high", "high extraction risk; expect manual review before enabling"),
    ],
)
def test_risk_summary_explains_extraction_risk(risk: IslandRisk, expected: str) -> None:
    """Candidate risk summaries state what validation the risk implies."""
    assert risk_summary(risk) == expected


def test_compilation_markdown_hides_generated_input_paths(tmp_path: Path) -> None:
    """Successful compilation reports do not expose disposable generated input paths."""
    source_path = tmp_path / "src" / "app" / "ranking.py"
    generated_path = tmp_path / ".atoll" / "sidecars" / "_atoll_app_ranking.py"
    artifact_path = tmp_path / ".atoll" / "artifacts" / "_atoll_app_ranking.so"
    island = EnabledIslandConfig(
        source_module="app.ranking",
        source_path=source_path,
        sidecar_module="app._atoll_app_ranking",
        sidecar_path=generated_path,
        symbols=("score_user",),
    )
    report = build_compilation_report(
        CompilationReportInput(
            root=tmp_path,
            operation="compile",
            module_filter="app.ranking",
            islands=(island,),
            build=CompileAttempt(
                success=True,
                command=("mypyc", str(generated_path), "build_ext"),
                stdout="",
                stderr="",
                artifact_paths=(artifact_path,),
                duration_seconds=0.25,
            ),
            verification=(
                VerifyResult(
                    source_module="app.ranking",
                    sidecar_module="app._atoll_app_ranking",
                    active=True,
                    compiled=True,
                    origin=str(artifact_path),
                    symbols=(("score_user", True),),
                ),
            ),
            cleanup_removed=(generated_path, generated_path.parent, tmp_path / ".atoll" / "build"),
        )
    )

    markdown = render_compilation_markdown_report(report)

    assert report["build"]["command"] == [
        "mypyc",
        "<1 generated Python build input>",
        "build_ext",
    ]
    assert report["cleanup"]["removed"] == [
        "<2 generated Python build inputs>",
        ".atoll/build",
    ]
    assert report["islands"][0]["generated_module"] == "app._atoll_app_ranking"
    assert report["tests"] is None
    assert report["summary"]["semantic_tests_run"] is False
    assert report["summary"]["semantic_test_failures"] == 0
    assert "Sidecar" not in markdown
    assert ".atoll/sidecars" not in markdown
    assert "Generated module" in markdown
    assert "semantic equivalence" in markdown
    assert "Semantic tests: not run" in markdown


def test_compilation_report_includes_semantic_test_gate(tmp_path: Path) -> None:
    """Compilation reports distinguish routing success from target test success."""
    source_path = tmp_path / "src" / "app" / "ranking.py"
    generated_path = tmp_path / ".atoll" / "sidecars" / "_atoll_app_ranking.py"
    artifact_path = tmp_path / ".atoll" / "artifacts" / "_atoll_app_ranking.so"
    island = EnabledIslandConfig(
        source_module="app.ranking",
        source_path=source_path,
        sidecar_module="app._atoll_app_ranking",
        sidecar_path=generated_path,
        symbols=("score_user",),
    )
    report = build_compilation_report(
        CompilationReportInput(
            root=tmp_path,
            operation="compile",
            module_filter="app.ranking",
            islands=(island,),
            build=CompileAttempt(
                success=True,
                command=("mypyc", str(generated_path), "build_ext"),
                stdout="",
                stderr="",
                artifact_paths=(artifact_path,),
                duration_seconds=0.25,
            ),
            verification=(
                VerifyResult(
                    source_module="app.ranking",
                    sidecar_module="app._atoll_app_ranking",
                    active=True,
                    compiled=True,
                    origin=str(artifact_path),
                    symbols=(("score_user", True),),
                ),
            ),
            tests=PytestRunResult(
                command=("pytest", "tests"),
                exit_code=1,
                success=False,
            ),
        )
    )

    markdown = render_compilation_markdown_report(report)

    assert report["success"] is False
    assert report["tests"] == {
        "command": ["pytest", "tests"],
        "exit_code": 1,
        "success": False,
    }
    assert report["summary"]["semantic_tests_run"] is True
    assert report["summary"]["semantic_test_failures"] == 1
    assert "Semantic tests: failed (`pytest tests`, exit code 1)" in markdown
    assert "- Command: `pytest tests`" in markdown
