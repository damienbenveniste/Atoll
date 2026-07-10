"""Tests for user-facing scan report wording."""

from __future__ import annotations

from pathlib import Path

import pytest

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.clustering import enrich_island_analysis
from atoll.analysis.native_readiness import NativeReadiness
from atoll.models import (
    ArtifactRecord,
    BackendAssessment,
    CompileAttempt,
    CompilePhaseTiming,
    EnabledIslandConfig,
    IslandRisk,
    ModuleId,
    PytestRunResult,
    VerifyResult,
)
from atoll.report import (
    CompilationPreflightBlockerInput,
    CompilationReportInput,
    CompilationSkippedModuleInput,
    build_compilation_report,
    render_compilation_markdown_report,
    risk_summary,
    score_label,
    score_summary,
)

REPORT_SCHEMA_VERSION = 2


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
                phase_timings=(
                    CompilePhaseTiming(name="cache_lookup", duration_seconds=0.01, detail="miss"),
                    CompilePhaseTiming(name="mypycify", duration_seconds=0.20),
                ),
                cache_status="miss",
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
    assert report["build"]["cache_status"] == "miss"
    assert report["build"]["phase_timings"][0]["name"] == "cache_lookup"
    assert "Sidecar" not in markdown
    assert ".atoll/sidecars" not in markdown
    assert "Generated module" in markdown
    assert "Cache: miss" in markdown
    assert "mypycify: 0.200s" in markdown
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


def test_source_clean_compilation_report_explains_wheel_and_skips(tmp_path: Path) -> None:
    """Source-clean reports explain wheel output, cleanup, and skipped modules."""
    source_path = tmp_path / "src" / "app" / "ranking.py"
    wheel_path = tmp_path / ".atoll" / "dist" / "app-0+atoll-cp312.whl"
    island = EnabledIslandConfig(
        source_module="app.ranking",
        source_path=source_path,
        sidecar_module="app._atoll_app_ranking",
        sidecar_path=(
            tmp_path / ".atoll" / "dist" / "build" / ".atoll" / "sidecars" / "_atoll_app_ranking.py"
        ),
        symbols=("score_user",),
    )
    report = build_compilation_report(
        CompilationReportInput(
            root=tmp_path,
            operation="compile",
            mode="source-clean",
            module_filter=None,
            islands=(island,),
            build=CompileAttempt(
                success=True,
                command=("mypyc", str(island.sidecar_path), "build_ext"),
                stdout="",
                stderr="",
                artifact_paths=(),
                duration_seconds=0.25,
            ),
            wheel_path=wheel_path,
            cleanup_removed=(tmp_path / ".atoll" / "dist" / "build",),
            cleanup_kept=(tmp_path / ".atoll" / "dist" / "install",),
            skipped_modules=(
                CompilationSkippedModuleInput(
                    module="app.bad",
                    reason="MYPYC_TYPE_ERROR: bad failed",
                ),
            ),
            preflight_blockers=(
                CompilationPreflightBlockerInput(
                    module="app.blocked",
                    path=tmp_path / "src" / "app" / "blocked.py",
                    line=4,
                    code="MYPYC_UNSUPPORTED_TYPEVAR",
                    message="TypeVar keyword(s) default are rejected by mypyc",
                ),
            ),
            native_readiness=(
                NativeReadiness(
                    source_module="app.ranking",
                    symbol="normalize_features",
                    eligible=True,
                    score=100,
                    function_count=1,
                    any_typed_functions=(),
                    boxed_typed_functions=(),
                    dynamic_dependencies=(),
                    loop_count=1,
                    native_operation_count=2,
                    reasons=(),
                ),
                NativeReadiness(
                    source_module="app.ranking",
                    symbol="score_user",
                    eligible=False,
                    score=30,
                    function_count=1,
                    any_typed_functions=("score_user",),
                    boxed_typed_functions=(),
                    dynamic_dependencies=("Score",),
                    loop_count=0,
                    native_operation_count=2,
                    reasons=(
                        "Any annotations: score_user",
                        "dynamic getattr dependencies: Score",
                    ),
                ),
            ),
        )
    )

    markdown = render_compilation_markdown_report(report)

    assert report["mode"] == "source-clean"
    assert report["version"] == REPORT_SCHEMA_VERSION
    assert report["summary"]["typed_regions"] == 0
    assert report["typed_regions"] == []
    assert report["wheel_path"] == ".atoll/dist/app-0+atoll-cp312.whl"
    assert report["cleanup"]["removed"] == [".atoll/dist/build"]
    assert report["cleanup"]["kept"] == [".atoll/dist/install"]
    assert report["summary"]["skipped_modules"] == 1
    assert report["summary"]["preflight_blockers"] == 1
    assert report["summary"]["native_ready_symbols"] == 1
    assert report["summary"]["native_rejected_symbols"] == 1
    assert report["native_readiness"][1]["symbol"] == "score_user"
    assert report["native_readiness"][1]["eligible"] is False
    assert "normal PEP 517 wheel" in markdown
    assert "Scan scores estimate extraction safety" not in markdown
    assert "`app.ranking.score_user`: rejected (30/100)" not in markdown
    assert "- Wheel: `.atoll/dist/app-0+atoll-cp312.whl`" in markdown
    assert "## Skipped Modules" in markdown
    assert "`app.blocked` (src/app/blocked.py:4)" in markdown


def test_compilation_report_retains_legacy_compiled_region_evidence(tmp_path: Path) -> None:
    """Schema v2 still derives backend and artifacts from legacy region fields."""
    source_path = tmp_path / "report_regions.py"
    source_path.write_text(
        """def first(value: int) -> int:
    return value + 1

def second(value: int) -> int:
    return value + 2

def third(value: int) -> int:
    return value + 3
""",
        encoding="utf-8",
    )
    scan = enrich_island_analysis(scan_module(ModuleId(name="report_regions", path=source_path)))
    regions = tuple(
        next(
            region
            for region in scan.typed_regions
            if any(member.id.qualname == name for member in region.members)
        )
        for name in ("first", "second", "third")
    )
    first, second, third = regions
    assessment = BackendAssessment(
        region_id=first.id,
        backend="mypyc",
        status="supported",
        supported_members=tuple(member.id for member in first.members),
        unsupported_members=(),
        capabilities=("typed_function",),
        reasons=(),
    )
    artifact = ArtifactRecord(
        region_id=second.id,
        backend="cython",
        logical_module="_atoll_second",
        role="primary",
        install_relative_path=".atoll/artifacts/_atoll_second.so",
        digest="a" * 64,
        abi="cp312",
        platform_tag="macosx_11_0_arm64",
    )

    report = build_compilation_report(
        CompilationReportInput(
            root=tmp_path,
            operation="compile",
            module_filter="report_regions",
            islands=(),
            build=CompileAttempt(
                success=True,
                command=("atoll", "typed-region-build"),
                stdout="",
                stderr="",
                artifact_paths=(),
                duration_seconds=0.1,
            ),
            typed_regions=regions,
            compiled_regions=regions,
            compiled_bindings=tuple(binding for region in regions for binding in region.bindings),
            backend_assessments=(assessment,),
            artifact_records=(artifact,),
        )
    )

    compiled = {item["id"]: item for item in report["compiled_regions"]}
    assert compiled[first.id]["backend"] == "mypyc"
    assert compiled[second.id]["backend"] == "cython"
    assert compiled[second.id]["artifacts"] == [".atoll/artifacts/_atoll_second.so"]
    assert compiled[third.id]["backend"] is None
    assert all(item["variant_id"] == item["id"] for item in compiled.values())
