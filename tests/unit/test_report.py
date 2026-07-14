"""Tests for user-facing scan report wording."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path, PurePosixPath

import pytest

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.clustering import enrich_island_analysis
from atoll.analysis.native_readiness import NativeReadiness
from atoll.analysis.task_fusion import FusionGateRejection, FusionPlan
from atoll.analysis.typed_regions import build_directed_region_slice
from atoll.execution_plans.models import (
    ChangedPayloadFile,
    ExecutionPlan,
    ExecutionPlanDiagnostic,
    ExecutionPlanTrial,
    PlanEdge,
    PlanGuard,
    PlanNode,
    PlanRejection,
)
from atoll.models import (
    ArtifactRecord,
    BackendAssessment,
    CandidateTrial,
    CompileAttempt,
    CompiledRegionVariant,
    CompilePhaseTiming,
    EnabledIslandConfig,
    IslandRisk,
    LoweringMode,
    ModuleId,
    PytestRunResult,
    RegionMember,
    SuspensionPoint,
    SymbolId,
    TypedRegion,
    VerifyResult,
)
from atoll.report import (
    COMPILE_REPORT_SCHEMA_VERSION,
    CompilationPreflightBlockerInput,
    CompilationReportInput,
    CompilationSkippedModuleInput,
    SourceOptimizationReportStatus,
    build_compilation_report,
    render_compilation_markdown_report,
    risk_summary,
    score_label,
    score_summary,
)
from atoll.runtime.fusion_performance import FusionTrial
from atoll.runtime.package_verify import PackageVerificationResult
from atoll.runtime.profiling import (
    CanonicalCallableCount,
    CanonicalTypeObservation,
    LifecycleCounts,
    MappedCandidateDecision,
    ObservedSignature,
    ProfiledMember,
    ProfiledSpawnSite,
    ProfileResult,
    ProfileSpawnSiteTarget,
    SubprocessPassEvidence,
    unconfigured_profile,
)
from atoll.source_optimization.models import (
    SourceAccessSite,
    SourceCallableEvidence,
    SourceEdit,
    SourceOptimizationApplicationStatus,
    SourceOptimizationAssessment,
    SourceOptimizationIdentity,
    SourceOptimizationPlan,
    SourceOptimizationTrial,
    SourceOptimizationTrialStatus,
    TransformationStep,
)

REPORT_SCHEMA_VERSION = COMPILE_REPORT_SCHEMA_VERSION
PROFILE_MAPPED_COVERAGE = 0.75
PROFILE_SELECTED_HOT_COVERAGE = 0.8
PROFILE_SAMPLING_INTERVAL_MS = 2
PROFILE_COMPLETED_CALLS = 11
PROFILE_MAX_ACTIVE_CALLS = 2
PROFILE_SPAWN_INVOCATIONS = 1_200
PROFILE_SCHEDULER_OVERHEAD_SAMPLES = 20
PROFILE_SCHEDULER_OVERHEAD_COVERAGE = 0.1
PROFILE_MEMBER_SCHEDULER_OVERHEAD_SAMPLES = 8
PROFILE_MEMBER_SCHEDULER_OVERHEAD_COVERAGE = 0.04
PROFILE_SELECTED_MEMBER_SAMPLES = 120
EXECUTION_PLAN_CANDIDATE_COUNT = 2
SOURCE_OPTIMIZATION_CURRENT_MEDIAN_SECONDS = 0.92
SOURCE_OPTIMIZATION_RESIDUAL_SAMPLES = 123


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


def test_source_clean_report_keeps_rejected_candidate_probes_diagnostic(
    tmp_path: Path,
) -> None:
    """Rejected selection probes do not override a successful final package."""
    wheel_path = tmp_path / ".atoll" / "dist" / "pkg-0+atoll.whl"
    failed_probe = PackageVerificationResult(
        stage="payload",
        target=tmp_path / ".atoll" / "dist" / "install",
        command=("python", "verify"),
        success=False,
        exit_code=1,
        stdout="",
        stderr="optional dependency unavailable",
        duration_seconds=0.1,
    )
    final_verification = PackageVerificationResult(
        stage="wheel",
        target=wheel_path,
        command=("python", "verify"),
        success=True,
        exit_code=0,
        stdout="",
        stderr="",
        duration_seconds=0.2,
    )
    report = build_compilation_report(
        CompilationReportInput(
            root=tmp_path,
            operation="compile",
            mode="source-clean",
            module_filter=None,
            islands=(),
            build=CompileAttempt(
                success=True,
                command=("mypyc",),
                stdout="",
                stderr="",
                artifact_paths=(),
                duration_seconds=0.3,
            ),
            operation_success=True,
            wheel_path=wheel_path,
            verification_steps=(failed_probe, final_verification),
        )
    )

    markdown = render_compilation_markdown_report(report)

    assert report["success"] is True
    assert report["summary"]["subprocess_verification_failures"] == 1
    assert "- Status: success" in markdown
    assert "rejected candidate-selection probes" in markdown
    assert "- payload: failed" in markdown


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
    assert report["summary"]["profile_accepted_hot_coverage"] == 0.0
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
        "scheduler_overhead_samples": 0,
        "scheduler_overhead_coverage": 0.0,
        "selected_hot_samples": 0,
        "selected_hot_coverage": 0.0,
        "child_passes": [],
        "lifecycle": {"start": 0, "return_": 0, "yield_": 0, "resume": 0, "unwind": 0, "throw": 0},
        "members": [],
        "spawn_sites": [],
        "candidate_mapping_decisions": [],
        "selected_symbols": [],
    }
    assert report["candidate_trials"] == []
    assert report["summary"]["source_optimization_status"] == "unbenchmarked"
    assert report["summary"]["source_optimization_plans"] == 0
    assert report["summary"]["source_optimization_trial_ready_assessments"] == 0
    assert report["summary"]["source_optimization_trials"] == 0
    assert report["source_optimization"] == {
        "status": "unbenchmarked",
        "minimum_speedup": 3.0,
        "headroom_speedup": None,
        "attributed_hot_share": 0.0,
        "plans": [],
        "assessments": [],
        "trials": [],
        "patch_path": None,
        "application_status": "not-applied",
    }
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
    assert "## Source Optimization" in markdown
    assert "- Status: unbenchmarked" in markdown
    assert "- Minimum speedup: 3.000x" in markdown
    assert "- No patch was emitted." in markdown
    assert "Scan scores estimate extraction safety" not in markdown
    assert "`app.ranking.score_user`: rejected (30/100)" not in markdown
    assert "- Wheel: `.atoll/dist/app-0+atoll-cp312.whl`" in markdown
    assert "## Skipped Modules" in markdown
    assert "`app.blocked` (src/app/blocked.py:4)" in markdown


def test_compilation_report_serializes_profile_guided_selection_without_values(
    tmp_path: Path,
) -> None:
    """Compile schema v6 emits canonical profile evidence without values or reprs."""
    profile = ProfileResult(
        status="profiled",
        reason="baseline profile collected",
        launch_kind="script",
        total_samples=200,
        mapped_project_samples=150,
        mapped_coverage=PROFILE_MAPPED_COVERAGE,
        scheduler_overhead_samples=PROFILE_SCHEDULER_OVERHEAD_SAMPLES,
        scheduler_overhead_coverage=PROFILE_SCHEDULER_OVERHEAD_COVERAGE,
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
                samples=PROFILE_SELECTED_MEMBER_SAMPLES,
                coverage=0.6,
                call_count=12,
                invocation_count=PROFILE_SPAWN_INVOCATIONS,
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
                scheduler_overhead_samples=PROFILE_MEMBER_SCHEDULER_OVERHEAD_SAMPLES,
                scheduler_overhead_coverage=PROFILE_MEMBER_SCHEDULER_OVERHEAD_COVERAGE,
                observation_capped=True,
                completed_calls=PROFILE_COMPLETED_CALLS,
                max_active_calls=PROFILE_MAX_ACTIVE_CALLS,
                pre_completion_suspensions=1,
            ),
            ProfiledMember(
                module="app.ranking",
                qualname="cold_path",
                samples=30,
                coverage=0.15,
                call_count=3,
                invocation_count=3,
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
                scheduler_overhead_samples=0,
                attributed_samples=PROFILE_SELECTED_MEMBER_SAMPLES,
                attributed_coverage=0.6,
                selected=True,
                reason="selected",
            ),
            MappedCandidateDecision(
                symbol=None,
                module="app.ranking",
                qualname="cold_path",
                samples=30,
                coverage=0.15,
                scheduler_overhead_samples=0,
                attributed_samples=30,
                attributed_coverage=0.15,
                selected=False,
                reason="unmapped",
            ),
        ),
        selected_symbols=(SymbolId("app.ranking", "score_user"),),
        spawn_sites=(
            ProfiledSpawnSite(
                target=ProfileSpawnSiteTarget(
                    id="spawn-site-report",
                    owner=SymbolId("app.ranking", "score_user"),
                    lineno=20,
                    col_offset=8,
                    scheduler_method="create_task",
                ),
                invocation_count=PROFILE_SPAWN_INVOCATIONS,
                callable_identities=(
                    CanonicalCallableCount(
                        identity="asyncio.taskgroups.TaskGroup.create_task",
                        count=PROFILE_SPAWN_INVOCATIONS,
                    ),
                ),
            ),
        ),
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
            candidate_trials=(
                CandidateTrial(
                    id="01:score-cython",
                    source_region_id="score-region",
                    variant_id="score-cython",
                    backend="cython",
                    lowering_mode="whole-callable",
                    symbols=("app.ranking::score_user",),
                    status="accepted",
                    reason="compiled median speedup 1.040 meets threshold 1.010",
                    marginal_speedup=1.04,
                    fallback_reason="mypyc rejected unresolved source TypeVars",
                    profile_samples=120,
                    profile_coverage=0.8,
                    accepted_hot_coverage=0.8,
                    baseline_variants=(),
                    trial_variants=("score-cython",),
                    semantic_test_exit_code=0,
                    semantic_test_duration_seconds=0.2,
                    benchmark_status="passed",
                    baseline_median_seconds=0.52,
                    candidate_median_seconds=0.5,
                    minimum_speedup=1.01,
                ),
            ),
            fusion_plans=(
                FusionPlan(
                    id="task-fusion:1234567890abcdef",
                    source_hash="a" * 64,
                    root="app.ranking::score_user",
                    caller="app.ranking::score_user",
                    callee="app.ranking::_run_task",
                    spawn_api="task_group.start_soon",
                    lineno=20,
                    end_lineno=20,
                    col_offset=8,
                    end_col_offset=52,
                    eligible=False,
                    observed_calls=12,
                    completed_calls=11,
                    max_active_calls=2,
                    pre_completion_suspensions=1,
                    observed_signatures=1,
                    observation_capped=True,
                    rejections=(
                        FusionGateRejection(
                            code="overlapping_calls",
                            reason="callee had overlapping active invocations",
                        ),
                    ),
                ),
            ),
            fusion_trials=(
                FusionTrial(
                    plan_id="task-fusion:test",
                    status="not-profitable",
                    reason="fused ratios missed thresholds",
                    semantic_runs=(),
                    baseline_median_seconds=1.0,
                    unfused_median_seconds=0.99,
                    fused_median_seconds=1.01,
                    baseline_over_unfused=1.01,
                    baseline_over_fused=0.99,
                    unfused_over_fused=0.98,
                    warmups=(),
                    samples=(),
                ),
            ),
        )
    )
    markdown = render_compilation_markdown_report(report)
    serialized = json.dumps(report, sort_keys=True)

    assert report["version"] == REPORT_SCHEMA_VERSION
    assert report["summary"]["profile_status"] == "profiled"
    assert report["summary"]["profile_mapped_coverage"] == PROFILE_MAPPED_COVERAGE
    assert report["summary"]["profile_selected_hot_coverage"] == PROFILE_SELECTED_HOT_COVERAGE
    assert report["summary"]["profile_accepted_hot_coverage"] == PROFILE_SELECTED_HOT_COVERAGE
    assert report["profile"]["scheduler_overhead_samples"] == PROFILE_SCHEDULER_OVERHEAD_SAMPLES
    assert report["profile"]["scheduler_overhead_coverage"] == PROFILE_SCHEDULER_OVERHEAD_COVERAGE
    assert report["candidate_trials"] == [
        {
            "id": "01:score-cython",
            "region_id": "score-cython",
            "source_region_id": "score-region",
            "variant_id": "score-cython",
            "backend": "cython",
            "lowering_mode": "whole-callable",
            "symbols": ["app.ranking::score_user"],
            "status": "accepted",
            "reason": "compiled median speedup 1.040 meets threshold 1.010",
            "marginal_speedup": 1.04,
            "fallback_reason": "mypyc rejected unresolved source TypeVars",
            "profile_samples": 120,
            "profile_coverage": 0.8,
            "accepted_hot_coverage": 0.8,
            "baseline_variants": [],
            "trial_variants": ["score-cython"],
            "semantic_test_exit_code": 0,
            "semantic_test_duration_seconds": 0.2,
            "benchmark_status": "passed",
            "baseline_median_seconds": 0.52,
            "candidate_median_seconds": 0.5,
            "minimum_speedup": 1.01,
        }
    ]
    assert report["optimization_policy"] == {
        "version": 1,
        "stability_floor_seconds": 0.25,
        "profile_guided_minimum_marginal_speedup": 1.01,
        "specialized_minimum_marginal_speedup": 1.05,
        "final_minimum_speedup": 1.1,
        "hard_benchmark_minimum_speedup": 3.0,
    }
    assert report["stage_medians"][0] == {
        "stage": "native:score-cython",
        "status": "passed",
        "baseline_median_seconds": 0.52,
        "candidate_median_seconds": 0.5,
        "speedup": 1.04,
        "minimum_speedup": 1.01,
    }
    assert "## Optimization Policy" in markdown
    assert "## Stage Medians" in markdown
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
    assert report["profile"]["members"][0]["completed_calls"] == PROFILE_COMPLETED_CALLS
    assert report["profile"]["members"][0]["max_active_calls"] == PROFILE_MAX_ACTIVE_CALLS
    assert report["profile"]["members"][0]["pre_completion_suspensions"] == 1
    assert report["profile"]["members"][0]["invocation_count"] == PROFILE_SPAWN_INVOCATIONS
    assert (
        report["profile"]["members"][0]["scheduler_overhead_samples"]
        == PROFILE_MEMBER_SCHEDULER_OVERHEAD_SAMPLES
    )
    assert (
        report["profile"]["members"][0]["scheduler_overhead_coverage"]
        == PROFILE_MEMBER_SCHEDULER_OVERHEAD_COVERAGE
    )
    assert report["profile"]["members"][0]["immediate_result_ratio"] == pytest.approx(10 / 11)
    assert report["profile"]["spawn_sites"] == [
        {
            "id": "spawn-site-report",
            "owner": "app.ranking::score_user",
            "lineno": 20,
            "col_offset": 8,
            "end_lineno": None,
            "end_col_offset": None,
            "scheduler_method": "create_task",
            "invocation_count": PROFILE_SPAWN_INVOCATIONS,
            "callable_identities": [
                {
                    "identity": "asyncio.taskgroups.TaskGroup.create_task",
                    "count": PROFILE_SPAWN_INVOCATIONS,
                }
            ],
        }
    ]
    assert report["profile"]["candidate_mapping_decisions"][1]["reason"] == "unmapped"
    assert (
        report["profile"]["candidate_mapping_decisions"][0]["attributed_samples"]
        == PROFILE_SELECTED_MEMBER_SAMPLES
    )
    assert report["profile"]["selected_symbols"] == ["app.ranking::score_user"]
    assert "120 leaf + 0 nested = 120" in markdown
    assert "## Candidate Profitability" in markdown
    assert "marginal speedup 1.040x" in markdown
    assert "fallback: mypyc rejected unresolved source TypeVars" in markdown
    assert report["summary"]["fusion_plans"] == 1
    assert report["summary"]["fusion_eligible_plans"] == 0
    assert report["summary"]["fusion_trials"] == 1
    assert report["fusion_plans"][0]["rejections"][0]["code"] == "overlapping_calls"
    assert report["fusion_trials"][0]["plan_id"] == "task-fusion:test"
    assert report["fusion_trials"][0]["status"] == "not-profitable"
    assert "## Experimental Task-Fusion Research" in markdown
    assert "rejected before trial" in markdown
    assert "### Three-Arm Trials" in markdown
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
            "lowering_mode": "whole-callable",
            "native_helpers": [],
            "symbols": [binding.source.stable_id for binding in first.bindings],
            "artifacts": [],
        },
        {
            "region_id": second.id,
            "variant_id": second.id,
            "source_module": "report_regions",
            "backend": "cython",
            "cache_status": "disabled",
            "lowering_mode": "whole-callable",
            "native_helpers": [],
            "symbols": [binding.source.stable_id for binding in second.bindings],
            "artifacts": [".atoll/artifacts/_atoll_second.so"],
        },
        {
            "region_id": third.id,
            "variant_id": third.id,
            "source_module": "report_regions",
            "backend": None,
            "cache_status": "disabled",
            "lowering_mode": "whole-callable",
            "native_helpers": [],
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
    start = value + 1
    doubled = start * 2
    total = doubled + math.floor(0.5)
    result = await helper(total)
    return result
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
    outlined_variant = CompiledRegionVariant(
        id=f"{region.id}@cython-outline",
        region=region,
        backend="cython",
        bindings=(hot_binding,),
        lowering_mode="outlined-block",
        native_helpers=("_hot__outlined_0_fixture",),
    )

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
            compiled_variants=(outlined_variant,),
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
    assert plans["async_regions::hot"]["lowering_mode"] == "outlined-block"
    assert plans["async_regions::hot"]["native_helpers"] == ["_hot__outlined_0_fixture"]
    assert plans["async_regions::hot"]["blocks"][0]["eligible"] is True
    assert plans["async_regions::hot"]["blocks"][0]["live_outs"] == ["total"]
    assert plans["async_regions::helper"]["lowering_mode"] == "interpreted"
    assert "## Suspension Handling" in markdown
    assert "`async_regions::hot`: outlined-block via `_hot__outlined_0_fixture`" in markdown


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
            "lowering_mode": "whole-callable",
            "native_helpers": [],
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
    assert report["cache_decisions"] == [
        {
            "variant_id": variant_id,
            "backend": "mypyc",
            "status": "hit",
            "batched": False,
        }
    ]
    assert report["final_composition"]["native_variant_ids"] == [variant_id]
    assert report["final_composition"]["artifacts"] == [".atoll/artifacts/_atoll_variant_score.so"]
    assert report["accepted_variants"] == [
        {
            "region_id": region.id,
            "variant_id": variant_id,
            "source_module": "variant_regions",
            "backend": "mypyc",
            "cache_status": "hit",
            "lowering_mode": "whole-callable",
            "native_helpers": [],
            "symbols": [binding.source.stable_id for binding in region.bindings],
            "artifacts": [".atoll/artifacts/_atoll_variant_score.so"],
        }
    ]


def test_compilation_report_serializes_execution_plan_compatibility_fields(tmp_path: Path) -> None:
    """Execution plans remain distinct from native regions and fusion research."""
    owner = SymbolId(module="app.scheduler", qualname="run")
    producer = SymbolId(module="app.scheduler", qualname="_produce")
    consumer = SymbolId(module="app.scheduler", qualname="_consume")
    plan = ExecutionPlan(
        id="exec-plan-selected",
        source_module="app.scheduler",
        owner=owner,
        dialect="asyncio",
        lowering_version="asyncio-v1",
        source_hash="a" * 64,
        callsite_fingerprint="b" * 64,
        topology_fingerprint="c" * 64,
        nodes=(
            PlanNode(owner.stable_id, owner, "orchestrator", 10),
            PlanNode(producer.stable_id, producer, "producer", 4),
            PlanNode(consumer.stable_id, consumer, "consumer", 7),
        ),
        edges=(
            PlanEdge(owner.stable_id, producer.stable_id, "spawns", "queue", 12),
            PlanEdge(producer.stable_id, consumer.stable_id, "passes_transport", "queue", 12),
        ),
        guards=(PlanGuard("scheduler", "asyncio", "scheduler must remain asyncio"),),
        hotness=2_000,
    )
    rejected = PlanRejection(
        id="exec-plan-rejected",
        source_module="app.scheduler",
        owner=SymbolId(module="app.scheduler", qualname="cold"),
        reason="low-hotness",
        message="site did not meet the invocation threshold",
        dialect="asyncio",
        lineno=20,
        hotness=12,
    )
    trial = ExecutionPlanTrial(
        plan_id=plan.id,
        status="accepted",
        command=("python", "benchmark.py"),
        exit_code=0,
        duration_seconds=0.5,
        diagnostics=(
            ExecutionPlanDiagnostic(
                code="verified",
                severity="note",
                message="semantic command passed",
            ),
        ),
        backend="task-preserving",
        reason="planned payload passed the marginal gate",
        benchmark_command=("python", "benchmark.py"),
        benchmark_status="passed",
        minimum_speedup=1.05,
        minimum_overall_speedup=1.10,
        baseline_median_seconds=1.2,
        unplanned_median_seconds=1.0,
        planned_median_seconds=0.8,
        marginal_speedup=1.25,
        overall_speedup=1.5,
        cache_status="hit",
        payload_files=(
            ChangedPayloadFile(
                install_path=PurePosixPath("app/scheduler.py"),
                before_hash="d" * 64,
                after_hash="e" * 64,
                role="source-overlay",
            ),
        ),
    )

    report = build_compilation_report(
        CompilationReportInput(
            root=tmp_path,
            operation="compile",
            module_filter="app.scheduler",
            islands=(),
            build=CompileAttempt(
                success=True,
                command=("atoll", "compile"),
                stdout="",
                stderr="",
                artifact_paths=(),
                duration_seconds=1.0,
            ),
            execution_plans=(plan, rejected),
            applied_execution_plans=(plan.id,),
            execution_plan_trials=(trial,),
        )
    )
    markdown = render_compilation_markdown_report(report)

    assert report["version"] == REPORT_SCHEMA_VERSION
    assert report["summary"]["execution_plans"] == EXECUTION_PLAN_CANDIDATE_COUNT
    assert report["summary"]["execution_selected_plans"] == 1
    assert report["summary"]["execution_applied_plans"] == 1
    assert report["summary"]["execution_plan_trials"] == 1
    assert report["execution_plans"][0]["status"] == "selected"
    assert report["execution_plans"][0]["callsite_fingerprint"] == "b" * 64
    assert report["execution_plans"][0]["topology_fingerprint"] == "c" * 64
    assert report["execution_plans"][0]["task_ownership"] == "structured"
    assert report["execution_plans"][1]["rejections"] == [
        {
            "code": "low-hotness",
            "reason": "site did not meet the invocation threshold",
        }
    ]
    assert report["applied_execution_plans"] == [plan.id]
    assert report["execution_plan_trials"][0]["diagnostics"][0]["code"] == "verified"
    assert report["execution_plan_trials"][0]["backend"] == "task-preserving"
    assert report["execution_plan_trials"][0]["marginal_speedup"] == pytest.approx(1.25)
    assert report["execution_plan_trials"][0]["overall_speedup"] == pytest.approx(1.5)
    assert report["execution_plan_trials"][0]["cache_status"] == "hit"
    assert report["execution_plan_trials"][0]["payload_files"] == [
        {
            "install_path": "app/scheduler.py",
            "before_hash": "d" * 64,
            "after_hash": "e" * 64,
            "role": "source-overlay",
        }
    ]
    assert "## Async Execution Plans" in markdown
    assert "Runtime status: report-only unless an applied plan" in markdown


def test_compilation_report_serializes_source_optimization_schema_v6(
    tmp_path: Path,
) -> None:
    """Schema v6 retains source plans, assessments, trials, and Markdown."""
    owner = SymbolId(module="app.scheduler", qualname="run")
    worker = SymbolId(module="app.scheduler", qualname="_produce")
    consumer = SymbolId(module="app.scheduler", qualname="_consume")
    access_site = SourceAccessSite(
        path=PurePosixPath("app/scheduler.py"),
        symbol=owner,
        kind="transport-drain",
        lineno=18,
        expression="queue",
        hazards=("suspension",),
    )
    step = TransformationStep(
        kind="private-transport-batch-drain",
        version="v1",
        source_symbol=worker,
        target_symbol=SymbolId(module="app.scheduler", qualname="_produce_batch"),
        access_sites=(access_site,),
        semantic_boundary="private-queue-ordering",
        description="batch private queue drains inside the owner",
    )
    identity = SourceOptimizationIdentity(
        execution_plan_id="exec-plan-selected",
        source_hashes=((PurePosixPath("app/scheduler.py"), "a" * 64),),
        topology_fingerprint="c" * 64,
        dialect="asyncio",
        lowering_version="source-v1",
        python_abi="cp312",
        transformation_versions=((step.kind, step.version),),
    )
    plan = SourceOptimizationPlan(
        id="source-opt-fixture",
        identity=identity,
        source=PurePosixPath("app/scheduler.py"),
        owner=owner,
        worker=worker,
        consumer=consumer,
        reducer=None,
        transport="queue",
        access_sites=(access_site,),
        entrypoint=owner,
        steps=(step,),
        semantic_boundaries=("private-queue-ordering",),
    )
    callable_evidence = SourceCallableEvidence(
        symbol=worker,
        static_role="worker",
        observed_invocations=400,
        completed_calls=390,
        static_suspension_points=1,
        observed_suspensions=0,
        immediate_result_ratio=1.0,
        median_seconds=0.002,
        hot_share=0.42,
        scheduler_overhead_samples=35,
        task_introspection=("asyncio.current_task",),
        cancellation=("CancelledError",),
        context_mutation=("contextvars.ContextVar.set",),
        unknown_dynamic_calls=("callback",),
        hazards=("suspension",),
    )
    assessment = SourceOptimizationAssessment(
        plan_id=plan.id,
        status="trial-ready",
        minimum_speedup=1.2,
        work_items=(worker,),
        observed_work_items=400,
        immediate_result_ratio=1.0,
        attributed_hot_share=0.46,
        scheduler_overhead_samples=35,
        scheduler_overhead_share=0.07,
        scheduler_overhead_evidence=("35 nested sample(s)",),
        callable_evidence=(callable_evidence,),
        headroom_speedup=1.8,
    )
    overlapping_assessment = SourceOptimizationAssessment(
        plan_id=f"{plan.id}:overlap",
        status="partial",
        minimum_speedup=1.1,
        work_items=(consumer,),
        observed_work_items=200,
        immediate_result_ratio=0.9,
        attributed_hot_share=0.31,
        scheduler_overhead_samples=20,
        scheduler_overhead_share=0.04,
        scheduler_overhead_evidence=("overlapping owner sample window",),
        callable_evidence=(),
        rejections=("overlapping hot share with source-opt-fixture",),
        headroom_speedup=1.3,
    )
    unbenchmarked_assessment = SourceOptimizationAssessment(
        plan_id=plan.id,
        status="unbenchmarked",
        minimum_speedup=3.0,
        work_items=(worker,),
        observed_work_items=0,
        immediate_result_ratio=0.0,
        attributed_hot_share=0.0,
        scheduler_overhead_samples=0,
        scheduler_overhead_share=0.0,
        scheduler_overhead_evidence=(),
        callable_evidence=(),
    )
    patch_path = tmp_path / ".atoll" / "patches" / "source-opt-fixture.patch"
    trial = SourceOptimizationTrial(
        plan_id=plan.id,
        status="accepted",
        semantic_command=("pytest", "tests/test_scheduler.py"),
        benchmark_command=("python", "bench_scheduler.py"),
        baseline_median_seconds=1.0,
        current_median_seconds=SOURCE_OPTIMIZATION_CURRENT_MEDIAN_SECONDS,
        source_median_seconds=0.7,
        wheel_median_seconds=0.65,
        source_speedup=1.43,
        wheel_speedup=1.54,
        patch_path=patch_path,
        source_edits=(
            SourceEdit(
                path=PurePosixPath("app/scheduler.py"),
                before_hash="b" * 64,
                after_hash="c" * 64,
                summary="batch private queue drains",
                touched_symbols=(owner, worker),
                transformation_id=step.stable_id,
                start_line=10,
                end_line=25,
            ),
        ),
        application_status="not-applied",
        diagnostics=("semantic command passed",),
        candidate_id="source-opt-fixture:batch",
        transformation_ids=(step.stable_id,),
        reason="source and wheel speedups exceeded the profitability gate",
        semantic_exit_code=0,
        semantic_duration_seconds=0.4,
        residual_profile=replace(
            unconfigured_profile(),
            status="profiled",
            reason="fresh accepted source profile",
            total_samples=SOURCE_OPTIMIZATION_RESIDUAL_SAMPLES,
        ),
    )
    report_only = build_compilation_report(
        CompilationReportInput(
            root=tmp_path,
            operation="compile",
            module_filter="app.scheduler",
            islands=(),
            build=CompileAttempt(
                success=True,
                command=("atoll", "compile"),
                stdout="",
                stderr="",
                artifact_paths=(),
                duration_seconds=1.0,
            ),
            source_optimization_plans=(plan,),
            source_optimization_assessments=(assessment, overlapping_assessment),
        )
    )
    unbenchmarked_report = build_compilation_report(
        CompilationReportInput(
            root=tmp_path,
            operation="compile",
            module_filter="app.scheduler",
            islands=(),
            build=CompileAttempt(
                success=True,
                command=("atoll", "compile"),
                stdout="",
                stderr="",
                artifact_paths=(),
                duration_seconds=1.0,
            ),
            source_optimization_plans=(plan,),
            source_optimization_assessments=(unbenchmarked_assessment,),
        )
    )
    rejected_report = build_compilation_report(
        CompilationReportInput(
            root=tmp_path,
            operation="compile",
            module_filter="app.scheduler",
            islands=(),
            build=CompileAttempt(
                success=True,
                command=("atoll", "compile"),
                stdout="",
                stderr="",
                artifact_paths=(),
                duration_seconds=1.0,
            ),
            source_optimization_plans=(plan,),
            source_optimization_assessments=(
                replace(assessment, status="unsupported", rejections=("unsafe",)),
            ),
        )
    )
    report = build_compilation_report(
        CompilationReportInput(
            root=tmp_path,
            operation="compile",
            module_filter="app.scheduler",
            islands=(),
            build=CompileAttempt(
                success=True,
                command=("atoll", "compile"),
                stdout="",
                stderr="",
                artifact_paths=(),
                duration_seconds=1.0,
            ),
            source_optimization_plans=(plan,),
            source_optimization_assessments=(assessment, overlapping_assessment),
            source_optimization_trials=(trial,),
        )
    )
    report_only_markdown = render_compilation_markdown_report(report_only)
    markdown = render_compilation_markdown_report(report)

    assert report_only["source_optimization"]["status"] == "report-only"
    assert report_only["source_optimization"]["patch_path"] is None
    assert unbenchmarked_report["source_optimization"]["status"] == "unbenchmarked"
    assert rejected_report["source_optimization"]["status"] == "rejected"
    assert "- No patch was emitted." in report_only_markdown
    assert "only an accepted trial contributes a transformed wheel or patch" in (
        report_only_markdown
    )
    assert report["version"] == REPORT_SCHEMA_VERSION
    assert report["execution_plans"] == []
    assert report["fusion_plans"] == []
    assert report["source_optimization"]["status"] == "accepted"
    assert report["source_optimization"]["minimum_speedup"] == pytest.approx(1.2)
    assert report["source_optimization"]["headroom_speedup"] == pytest.approx(1.8)
    assert report["source_optimization"]["attributed_hot_share"] == pytest.approx(0.46)
    assert report["source_optimization"]["patch_path"] == (
        ".atoll/patches/source-opt-fixture.patch"
    )
    assert report["source_optimization"]["application_status"] == "not-applied"
    assert report["summary"]["source_optimization_status"] == "accepted"
    assert report["summary"]["source_optimization_plans"] == 1
    assert report["summary"]["source_optimization_trial_ready_assessments"] == 1
    assert report["summary"]["source_optimization_trials"] == 1
    assert report["final_composition"]["source_plan_ids"] == [plan.id]
    assert report["final_composition"]["transformation_ids"] == [step.stable_id]
    assert report["final_composition"]["native_variant_ids"] == []
    assert "## Final Composition" in markdown
    assert report["source_optimization"]["plans"][0]["identity"]["source_hashes"] == {
        "app/scheduler.py": "a" * 64
    }
    assert report["source_optimization"]["plans"][0]["steps"][0]["access_sites"][0] == {
        "path": "app/scheduler.py",
        "symbol": "app.scheduler::run",
        "kind": "transport-drain",
        "lineno": 18,
        "expression": "queue",
        "hazards": ["suspension"],
    }
    assert report["source_optimization"]["assessments"][0]["callable_evidence"][0][
        "unknown_dynamic_calls"
    ] == ["callback"]
    assert report["source_optimization"]["trials"][0]["source_edits"] == [
        {
            "path": "app/scheduler.py",
            "before_hash": "b" * 64,
            "after_hash": "c" * 64,
            "summary": "batch private queue drains",
            "touched_symbols": ["app.scheduler::run", "app.scheduler::_produce"],
            "transformation_id": step.stable_id,
            "start_line": 10,
            "end_line": 25,
        }
    ]
    assert report["source_optimization"]["trials"][0]["semantic_command"] == [
        "pytest",
        "tests/test_scheduler.py",
    ]
    assert report["source_optimization"]["trials"][0]["baseline_median_seconds"] == 1.0
    assert (
        report["source_optimization"]["trials"][0]["current_median_seconds"]
        == SOURCE_OPTIMIZATION_CURRENT_MEDIAN_SECONDS
    )
    assert report["source_optimization"]["trials"][0]["source_speedup"] == pytest.approx(1.43)
    assert report["source_optimization"]["trials"][0]["wheel_speedup"] == pytest.approx(1.54)
    residual_profile = report["source_optimization"]["trials"][0]["residual_profile"]
    assert residual_profile is not None
    assert residual_profile["status"] == "profiled"
    assert residual_profile["total_samples"] == SOURCE_OPTIMIZATION_RESIDUAL_SAMPLES
    assert "## Source Optimization" in markdown
    assert "Plan `source-opt-fixture`" in markdown
    assert "Trial `source-opt-fixture`: accepted" in markdown
    assert "residual profile profiled with 123 samples" in markdown


@pytest.mark.parametrize(
    ("trial_status", "application_status", "expected_status"),
    [
        ("accepted", "applied", "applied"),
        ("rejected", "not-applied", "not-profitable"),
        ("failed-semantics", "not-applied", "unavailable"),
        ("not-run", "not-applied", "rejected"),
        ("accepted", "failed", "failed"),
        ("accepted", "conflicted", "conflicted"),
        ("accepted", "rolled-back", "rolled-back"),
        ("accepted", "stale-source", "stale-source"),
        ("accepted", "unavailable", "unavailable"),
    ],
)
def test_source_optimization_report_derives_trial_and_application_statuses(
    tmp_path: Path,
    trial_status: SourceOptimizationTrialStatus,
    application_status: SourceOptimizationApplicationStatus,
    expected_status: SourceOptimizationReportStatus,
) -> None:
    """Schema v5 distinguishes trial decisions from transactional application state."""
    trial = SourceOptimizationTrial(
        plan_id="source-opt-status",
        status=trial_status,
        semantic_command=("pytest", "-q"),
        benchmark_command=("python", "bench.py"),
        baseline_median_seconds=None,
        source_median_seconds=None,
        wheel_median_seconds=None,
        source_speedup=None,
        wheel_speedup=None,
        patch_path=None,
        source_edits=(),
        application_status=application_status,
    )

    report = build_compilation_report(
        CompilationReportInput(
            root=tmp_path,
            operation="compile",
            module_filter=None,
            islands=(),
            build=CompileAttempt(
                success=True,
                command=("atoll", "compile"),
                stdout="",
                stderr="",
                artifact_paths=(),
                duration_seconds=0.0,
            ),
            source_optimization_trials=(trial,),
        )
    )

    assert report["source_optimization"]["status"] == expected_status
    assert report["source_optimization"]["application_status"] == application_status


def test_suspension_report_applies_member_rejections_to_block_eligibility(
    tmp_path: Path,
) -> None:
    """Member-level scope rejection overrides an otherwise eligible local block."""
    member = RegionMember(
        id=SymbolId("blocked_async", "blocked"),
        kind="function",
        owner_class=None,
        binding_kind="module",
        execution_kind="coroutine",
        source_text="""async def blocked(values):
    callback = lambda value: value
    start = len(values) + 1
    doubled = start * 2
    total = callback(doubled)
    await checkpoint()
    return total
""",
        type_parameters=(),
        type_parameter_records=(),
        scope_type_parameters=(),
        scope_type_parameter_records=(),
        parameters=(),
        return_annotation=None,
        suspension_points=(
            SuspensionPoint(
                kind="await",
                lineno=6,
                end_lineno=6,
                col_offset=4,
                end_col_offset=22,
            ),
        ),
    )
    region = TypedRegion(
        id="blocked_async::blocked:fixture",
        source_module=ModuleId("blocked_async", tmp_path / "blocked_async.py"),
        members=(member,),
        dependencies=(),
        type_bindings=(),
        bindings=(),
        decisions=(),
        source_hash="a" * 64,
    )
    report = build_compilation_report(
        CompilationReportInput(
            root=tmp_path,
            operation="compile",
            module_filter="blocked_async",
            islands=(),
            build=CompileAttempt(
                success=False,
                command=("atoll", "typed-region-build"),
                stdout="",
                stderr="interpreted",
                artifact_paths=(),
                duration_seconds=0.0,
            ),
            typed_regions=(region,),
        )
    )
    plan = report["suspension_plans"][0]

    assert {rejection["code"] for rejection in plan["rejections"]} == {"nested_scope"}
    assert plan["blocks"][0]["eligible"] is False
    assert "0/1 synchronous blocks eligible" in render_compilation_markdown_report(report)


@pytest.mark.parametrize(
    ("mode", "helpers", "message"),
    [
        ("outlined-block", (), "require native helpers"),
        ("whole-callable", ("helper",), "cannot declare"),
        ("outlined-block", ("helper", "helper"), "must be unique"),
    ],
)
def test_compiled_variant_rejects_contradictory_lowering_metadata(
    tmp_path: Path,
    mode: LoweringMode,
    helpers: tuple[str, ...],
    message: str,
) -> None:
    """Compiled variant mode and helper metadata remain internally consistent."""
    source_path = tmp_path / "variant_validation.py"
    source_path.write_text(
        "def score(value: int) -> int:\n    return value + 1\n",
        encoding="utf-8",
    )
    scan = enrich_island_analysis(
        scan_module(ModuleId(name="variant_validation", path=source_path))
    )
    region = scan.typed_regions[0]

    with pytest.raises(ValueError, match=message):
        CompiledRegionVariant(
            id=f"{region.id}:fixture",
            region=region,
            backend="cython",
            bindings=region.bindings,
            lowering_mode=mode,
            native_helpers=helpers,
        )
