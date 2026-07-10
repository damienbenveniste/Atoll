"""Tests for user-facing scan report wording."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.clustering import enrich_island_analysis
from atoll.analysis.native_readiness import NativeReadiness
from atoll.analysis.typed_regions import build_directed_region_slice
from atoll.models import (
    ArtifactRecord,
    BackendAssessment,
    CompileAttempt,
    CompiledRegionVariant,
    CompilePhaseTiming,
    EnabledIslandConfig,
    IslandRisk,
    ModuleId,
    PytestRunResult,
    SymbolId,
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
from atoll.runtime.profiling import (
    CanonicalTypeObservation,
    LifecycleCounts,
    MappedCandidateDecision,
    ObservedSignature,
    ProfiledMember,
    ProfileResult,
    SubprocessPassEvidence,
)

REPORT_SCHEMA_VERSION = 3
PROFILE_MAPPED_COVERAGE = 0.75
PROFILE_SELECTED_HOT_COVERAGE = 0.8
PROFILE_SAMPLING_INTERVAL_MS = 2


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
    assert report["summary"]["profile_status"] == "unconfigured"
    assert report["summary"]["profile_mapped_coverage"] == 0.0
    assert report["summary"]["profile_selected_hot_coverage"] == 0.0
    assert report["profile"] == {
        "status": "unconfigured",
        "reason": "no benchmark command configured; static candidate evidence only",
        "launch_kind": "unconfigured",
        "sampling_policy": {
            "interval_ms": PROFILE_SAMPLING_INTERVAL_MS,
            "mode": "statistical leaf-frame sampling",
        },
        "total_samples": 0,
        "mapped_project_samples": 0,
        "mapped_coverage": 0.0,
        "selected_hot_samples": 0,
        "selected_hot_coverage": 0.0,
        "child_passes": [],
        "lifecycle": {"start": 0, "return_": 0, "yield_": 0, "resume": 0, "unwind": 0, "throw": 0},
        "members": [],
        "candidate_mapping_decisions": [],
        "selected_symbols": [],
    }
    assert report["candidate_trials"] == []
    assert report["suspension_plans"] == []
    assert report["backend_decisions"] == []
    assert report["accepted_variants"] == []
    assert report["rejected_variants"] == [
        {"module": "app.bad", "reason": "MYPYC_TYPE_ERROR: bad failed"}
    ]
    assert report["native_readiness"][1]["symbol"] == "score_user"
    assert report["native_readiness"][1]["eligible"] is False
    assert "normal PEP 517 wheel" in markdown
    assert "## Profile-Guided Selection" in markdown
    assert "- Status: unconfigured" in markdown
    assert "- Selected candidates: none" in markdown
    assert "Scan scores estimate extraction safety" not in markdown
    assert "`app.ranking.score_user`: rejected (30/100)" not in markdown
    assert "- Wheel: `.atoll/dist/app-0+atoll-cp312.whl`" in markdown
    assert "## Skipped Modules" in markdown
    assert "`app.blocked` (src/app/blocked.py:4)" in markdown


def test_compilation_report_serializes_profile_guided_selection_without_values(
    tmp_path: Path,
) -> None:
    """Schema v3 emits canonical profile evidence without object values or reprs."""
    profile = ProfileResult(
        status="profiled",
        reason="baseline profile collected",
        launch_kind="script",
        total_samples=200,
        mapped_project_samples=150,
        mapped_coverage=PROFILE_MAPPED_COVERAGE,
        selected_hot_samples=120,
        selected_hot_coverage=PROFILE_SELECTED_HOT_COVERAGE,
        runs=(
            SubprocessPassEvidence(
                "sampling",
                ("python", "-m", "atoll.runtime._profile_bootstrap", "config.json"),
                0,
                "SECRET_VALUE repr(payload)",
                "",
                0.1,
            ),
        ),
        lifecycle=LifecycleCounts(start=10, return_=9, yield_=1, resume=1, unwind=0, throw=0),
        members=(
            ProfiledMember(
                module="app.ranking",
                qualname="score_user",
                samples=120,
                coverage=0.6,
                call_count=12,
                lifecycle=LifecycleCounts(
                    start=12,
                    return_=12,
                    yield_=0,
                    resume=0,
                    unwind=0,
                    throw=0,
                ),
                signatures=(
                    ObservedSignature(
                        parameters=(
                            CanonicalTypeObservation(
                                parameter_name="payload",
                                type_path="app.models.SecretPayload",
                                count=12,
                            ),
                        ),
                        count=12,
                    ),
                ),
                polymorphic_overflow=True,
                observation_capped=True,
            ),
            ProfiledMember(
                module="app.ranking",
                qualname="cold_path",
                samples=30,
                coverage=0.15,
                call_count=3,
                lifecycle=LifecycleCounts(
                    start=3,
                    return_=3,
                    yield_=0,
                    resume=0,
                    unwind=0,
                    throw=0,
                ),
                signatures=(),
                polymorphic_overflow=False,
            ),
        ),
        candidates=(
            MappedCandidateDecision(
                symbol=SymbolId("app.ranking", "score_user"),
                module="app.ranking",
                qualname="score_user",
                samples=120,
                coverage=0.6,
                selected=True,
                reason="selected",
            ),
            MappedCandidateDecision(
                symbol=None,
                module="app.ranking",
                qualname="cold_path",
                samples=30,
                coverage=0.15,
                selected=False,
                reason="unmapped",
            ),
        ),
        selected_symbols=(SymbolId("app.ranking", "score_user"),),
    )

    report = build_compilation_report(
        CompilationReportInput(
            root=tmp_path,
            operation="compile",
            module_filter="app.ranking",
            islands=(),
            build=CompileAttempt(
                success=True,
                command=("mypyc", "build_ext"),
                stdout="",
                stderr="",
                artifact_paths=(),
                duration_seconds=0.25,
            ),
            profile=profile,
        )
    )
    markdown = render_compilation_markdown_report(report)
    serialized = json.dumps(report, sort_keys=True)

    assert report["version"] == REPORT_SCHEMA_VERSION
    assert report["summary"]["profile_status"] == "profiled"
    assert report["summary"]["profile_mapped_coverage"] == PROFILE_MAPPED_COVERAGE
    assert report["summary"]["profile_selected_hot_coverage"] == PROFILE_SELECTED_HOT_COVERAGE
    assert report["profile"]["sampling_policy"]["interval_ms"] == PROFILE_SAMPLING_INTERVAL_MS
    assert report["profile"]["child_passes"] == [
        {
            "pass_kind": "sampling",
            "command": ["python", "-m", "atoll.runtime._profile_bootstrap", "config.json"],
            "returncode": 0,
            "duration_seconds": 0.1,
        }
    ]
    assert report["profile"]["members"][0]["signatures"][0]["parameters"] == [
        {"parameter_name": "payload", "type_path": "app.models.SecretPayload", "count": 12}
    ]
    assert report["profile"]["members"][0]["polymorphic"] is True
    assert report["profile"]["members"][0]["observation_capped"] is True
    assert report["profile"]["candidate_mapping_decisions"][1]["reason"] == "unmapped"
    assert report["profile"]["selected_symbols"] == ["app.ranking::score_user"]
    assert "SECRET_VALUE" not in serialized
    assert "repr(payload)" not in serialized
    assert "- Status: profiled" in markdown
    assert "- Mapped coverage: 75.0%" in markdown
    assert "- Selected hot coverage: 80.0%" in markdown
    assert "- Selected candidates: `app.ranking::score_user`" in markdown
    assert "`app.ranking::cold_path` (unmapped)" in markdown
    assert "- Unmeasured profiling passes: sampling 0.100s" in markdown
    assert "- Bounded type observation reached: `app.ranking::score_user`" in markdown


def test_compilation_markdown_reports_unsupported_profile_static_fallback(
    tmp_path: Path,
) -> None:
    """Unsupported benchmark launchers are called out as static fallback evidence."""
    report = build_compilation_report(
        CompilationReportInput(
            root=tmp_path,
            operation="compile",
            module_filter=None,
            islands=(),
            build=CompileAttempt(
                success=True,
                command=("mypyc", "build_ext"),
                stdout="",
                stderr="",
                artifact_paths=(),
                duration_seconds=0.25,
            ),
            profile=ProfileResult(
                status="static-fallback",
                reason="unsupported benchmark launcher; static candidate fallback required",
                launch_kind="unsupported",
                total_samples=0,
                mapped_project_samples=0,
                mapped_coverage=0.0,
                selected_hot_samples=0,
                selected_hot_coverage=0.0,
                runs=(),
                lifecycle=LifecycleCounts(
                    start=0,
                    return_=0,
                    yield_=0,
                    resume=0,
                    unwind=0,
                    throw=0,
                ),
                members=(),
                candidates=(),
                selected_symbols=(),
            ),
        )
    )

    markdown = render_compilation_markdown_report(report)

    assert report["profile"]["status"] == "static-fallback"
    assert "unsupported benchmark launcher" in markdown
    assert "- Unsupported launcher: using static fallback candidate evidence" in markdown


def test_compilation_report_retains_legacy_compiled_region_evidence(tmp_path: Path) -> None:
    """Schema v3 still derives backend and artifacts from legacy region fields."""
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
    assert report["backend_decisions"] == [
        {
            "region_id": first.id,
            "backend": "mypyc",
            "status": "supported",
            "supported_members": [member.id.stable_id for member in first.members],
            "unsupported_members": [],
            "capabilities": ["typed_function"],
            "reasons": [],
            "deterministic": True,
        }
    ]
    assert report["accepted_variants"] == [
        {
            "region_id": first.id,
            "variant_id": first.id,
            "source_module": "report_regions",
            "backend": "mypyc",
            "cache_status": "disabled",
            "symbols": [binding.source.stable_id for binding in first.bindings],
            "artifacts": [],
        },
        {
            "region_id": second.id,
            "variant_id": second.id,
            "source_module": "report_regions",
            "backend": "cython",
            "cache_status": "disabled",
            "symbols": [binding.source.stable_id for binding in second.bindings],
            "artifacts": [".atoll/artifacts/_atoll_second.so"],
        },
        {
            "region_id": third.id,
            "variant_id": third.id,
            "source_module": "report_regions",
            "backend": None,
            "cache_status": "disabled",
            "symbols": [binding.source.stable_id for binding in third.bindings],
            "artifacts": [],
        },
    ]
    assert report["rejected_variants"] == []


def test_compilation_report_serializes_async_region_planning_evidence(tmp_path: Path) -> None:
    """Schema v3 exposes ordered suspension, call, import, and boundary facts."""
    source_path = tmp_path / "async_regions.py"
    source_path.write_text(
        """async def helper(value: int) -> int:
    await value
    return value + 1

async def hot(value: int) -> int:
    import math
    return await helper(value) + math.floor(0.5)
""",
        encoding="utf-8",
    )
    scan = enrich_island_analysis(scan_module(ModuleId(name="async_regions", path=source_path)))
    region = next(
        item
        for item in scan.typed_regions
        if {member.id.qualname for member in item.members} == {"helper", "hot"}
    )
    hot_id = next(member.id for member in region.members if member.id.qualname == "hot")
    helper_id = next(member.id for member in region.members if member.id.qualname == "helper")
    region = replace(
        region,
        dependencies=tuple(
            replace(dependency, requires_same_unit=True)
            if dependency.src == hot_id and dependency.dst == helper_id
            else dependency
            for dependency in region.dependencies
        ),
    )
    region = build_directed_region_slice(region, hot_id)
    hot_binding = next(binding for binding in region.bindings if binding.source == hot_id)

    report = build_compilation_report(
        CompilationReportInput(
            root=tmp_path,
            operation="compile",
            module_filter="async_regions",
            islands=(),
            build=CompileAttempt(
                success=True,
                command=("atoll", "typed-region-build"),
                stdout="",
                stderr="",
                artifact_paths=(),
                duration_seconds=0.1,
            ),
            typed_regions=(region,),
            compiled_regions=(region,),
            compiled_bindings=(hot_binding,),
        )
    )
    markdown = render_compilation_markdown_report(report)
    serialized_region = report["typed_regions"][0]
    hot = next(member for member in serialized_region["members"] if member["id"].endswith("::hot"))
    helper_call = next(call for call in hot["call_sites"] if call["target"] == "helper")
    helper_dependency = next(
        dependency
        for dependency in serialized_region["dependencies"]
        if dependency["dst"].endswith("::helper")
    )

    assert helper_call["invocation_mode"] == "awaited"
    assert helper_call["requires_same_unit"] is False
    assert hot["suspension_points"][0]["kind"] == "await"
    assert hot["runtime_imports"][0]["imported_names"] == ["math"]
    assert helper_dependency["invocation_mode"] == "awaited"
    assert helper_dependency["requires_same_unit"] is True
    plans = {plan["member"]: plan for plan in report["suspension_plans"]}
    assert plans["async_regions::hot"]["lowering_mode"] == "whole-callable"
    assert plans["async_regions::helper"]["lowering_mode"] == "whole-callable"
    assert "## Suspension Handling" in markdown
    assert "`async_regions::hot`: whole-callable" in markdown


def test_compilation_report_accepts_explicit_compiled_variants(tmp_path: Path) -> None:
    """Accepted variants normalize explicit backend-specific compiled subsets."""
    source_path = tmp_path / "variant_regions.py"
    source_path.write_text(
        """def score(value: int) -> int:
    return value + 1
""",
        encoding="utf-8",
    )
    scan = enrich_island_analysis(scan_module(ModuleId(name="variant_regions", path=source_path)))
    region = scan.typed_regions[0]
    variant_id = f"{region.id}:mypyc"
    variant = CompiledRegionVariant(
        id=variant_id,
        region=region,
        backend="mypyc",
        bindings=region.bindings,
        cache_status="hit",
    )
    artifact = ArtifactRecord(
        region_id=variant_id,
        backend="mypyc",
        logical_module="_atoll_variant_score",
        role="primary",
        install_relative_path=".atoll/artifacts/_atoll_variant_score.so",
        digest="b" * 64,
        abi="cp312",
        platform_tag="macosx_11_0_arm64",
    )

    report = build_compilation_report(
        CompilationReportInput(
            root=tmp_path,
            operation="compile",
            module_filter="variant_regions",
            islands=(),
            build=CompileAttempt(
                success=True,
                command=("atoll", "typed-region-build"),
                stdout="",
                stderr="",
                artifact_paths=(),
                duration_seconds=0.1,
            ),
            typed_regions=(region,),
            compiled_variants=(variant,),
            artifact_records=(artifact,),
        )
    )

    assert report["compiled_regions"] == [
        {
            "id": region.id,
            "variant_id": variant_id,
            "source_module": "variant_regions",
            "backend": "mypyc",
            "cache_status": "hit",
            "bindings": [
                {
                    "source": binding.source.stable_id,
                    "compiled_name": binding.compiled_name,
                    "kind": binding.kind,
                    "owner_class": binding.owner_class,
                    "target_owner_class": binding.target_owner_class,
                    "execution_kind": binding.execution_kind,
                    "required": binding.required,
                    "guards": [],
                }
                for binding in region.bindings
            ],
            "artifacts": [".atoll/artifacts/_atoll_variant_score.so"],
        }
    ]
    assert report["accepted_variants"] == [
        {
            "region_id": region.id,
            "variant_id": variant_id,
            "source_module": "variant_regions",
            "backend": "mypyc",
            "cache_status": "hit",
            "symbols": [binding.source.stable_id for binding in region.bindings],
            "artifacts": [".atoll/artifacts/_atoll_variant_score.so"],
        }
    ]
