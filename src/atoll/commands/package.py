"""Build installable Atoll artifacts without modifying source files."""

from __future__ import annotations

import ast
import hashlib
import re
import shutil
import textwrap
import time
import tomllib
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path, PurePosixPath
from typing import cast

from packaging import tags

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.clustering import enrich_island_analysis
from atoll.analysis.execution_plans import (
    build_execution_plans,
    execution_plan_observation_targets,
    execution_plan_profile_targets,
)
from atoll.analysis.native_readiness import NativeReadiness
from atoll.analysis.task_fusion import (
    FusionPlan,
    build_fusion_plans,
    fusion_observation_targets,
)
from atoll.analysis.typed_regions import build_directed_region_slice
from atoll.backends.base import CompilerBackend
from atoll.backends.cython import CYTHON_BACKEND
from atoll.backends.mypyc import MYPYC_BACKEND
from atoll.execution_plans.models import ExecutionPlan, PlanRejection
from atoll.generation.outlined_region import (
    OUTLINED_REGION_GENERATOR_VERSION,
    generate_outlined_region,
)
from atoll.generation.region_shim import (
    RegionShimConfig,
    insert_or_replace_region_shim,
    remove_region_shim,
)
from atoll.generation.task_fusion import generate_eager_task_fusion
from atoll.generation.typed_region import (
    TYPED_METHOD_GENERATOR_VERSION,
    TypedRegionGeneration,
    TypedRegionGenerationOptions,
    generate_typed_method_region,
)
from atoll.models import (
    ArtifactRecord,
    Backend,
    BackendAssessment,
    BackendCompileContext,
    BackendCompileResult,
    BackendLoweringRequest,
    BindingTarget,
    Blocker,
    CandidateTrial,
    CandidateTrialStatus,
    CompilationUnit,
    CompileAttempt,
    CompileCacheStatus,
    CompiledRegionVariant,
    CompilePhaseTiming,
    EnabledIslandConfig,
    LoweringDecision,
    LoweringMode,
    ModuleId,
    ModuleScan,
    RegionSpecialization,
    SymbolId,
    TypedRegion,
)
from atoll.project import DiscoveredProject, discover_project
from atoll.region_cache import compile_with_region_cache
from atoll.runtime.fusion_performance import (
    FusionBenchmarkConfig,
    FusionTrial,
    run_fusion_trial,
    unavailable_fusion_trial,
)
from atoll.runtime.package_verify import (
    PackageVerificationPlan,
    PackageVerificationResult,
    VerificationArtifact,
    VerificationBinding,
    VerificationStage,
    verify_package_subprocess,
)
from atoll.runtime.performance import (
    BenchmarkGateConfig,
    BenchmarkGateResult,
    BenchmarkProgress,
    CommandRunEvidence,
    run_benchmark_gate,
    run_performance_command,
)
from atoll.runtime.profiling import (
    ProfileResult,
    run_baseline_profile,
    select_profile_candidates,
)
from atoll.wheel_overlay import (
    WheelBuildEvidence,
    WheelOverlayError,
    build_baseline_wheel,
    repack_overlaid_wheel,
    unpack_wheel_payload,
)

PackageProgress = Callable[[str], None]

_COMPILER_BACKENDS: dict[Backend, CompilerBackend] = {
    "mypyc": MYPYC_BACKEND,
    "cython": CYTHON_BACKEND,
}

_CANDIDATE_BENCHMARK_WARMUPS = 1
_CANDIDATE_BENCHMARK_SAMPLES = 3
_CANDIDATE_MINIMUM_SPEEDUP = 1.01

_GENERATED_DIR_NAMES = frozenset(
    {
        ".atoll",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "build",
        "dist",
        "site-packages",
    }
)
_PEP517_IGNORED_NAMES = frozenset(
    {
        ".atoll",
        ".git",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "venv",
    }
)


@dataclass(frozen=True, slots=True)
class PackageOptions:
    """Options for building an installable source-clean Atoll artifact.

    ``selected_members`` is an internal selection boundary used by trial mode.
    Empty selection compiles every backend-supported typed region in scope;
    explicit selections also retain their same-region runtime dependencies.
    ``cache_dir`` can isolate reusable state for a temporary caller, and
    ``run_quality_gates=False`` delegates semantic and benchmark commands to
    that caller without weakening wheel routing verification.

    Attributes:
        root: Root directory of the target Python project.
        module_name: Importable module name used to restrict the command.
        output_dir: Directory receiving source-clean wheel artifacts.
        keep_install_tree: Whether source-clean staging is retained after completion.
        progress: Optional progress callback used by long-running packaging work.
        selected_members: Explicit members to compile; empty selects every supported region.
        cache_dir: Optional cache override used by isolated callers such as trial mode.
        run_quality_gates: Whether package verification, tests, and benchmarks should run.
    """

    root: Path
    module_name: str | None = None
    output_dir: Path | None = None
    keep_install_tree: bool = False
    progress: PackageProgress | None = None
    selected_members: tuple[SymbolId, ...] = ()
    cache_dir: Path | None = None
    run_quality_gates: bool = True


@dataclass(frozen=True, slots=True)
class PackageCommandResult:
    """Result from building a source-clean Atoll package artifact.

    Attributes:
        success: Whether the represented operation completed successfully.
        project_root: Root directory of the target Python project.
        output_dir: Directory receiving source-clean wheel artifacts.
        install_root: Temporary source-clean installation tree.
        wheel_path: Source-clean wheel path, when produced.
        islands: Enabled islands included in the operation or report.
        build: Captured native build evidence.
        install_tree_kept: Whether the temporary installation tree remains for inspection.
        cleanup_removed: Generated paths removed after the operation.
        cleanup_kept: Generated paths intentionally retained for diagnostics.
        report_artifact_paths: Artifact paths exposed in user-facing reports.
        error: User-facing failure text, or `None` on success.
        skipped: Island builds that failed without producing usable artifacts.
        preflight_skipped: Modules skipped before compilation because of known blockers.
        native_readiness: Post-generation native-readiness evidence.
        typed_regions: Backend-neutral typed regions discovered or reported.
        compiled_regions: Typed regions successfully compiled into the wheel.
        compiled_bindings: Source bindings successfully provided by compiled regions.
        compiled_variants: Backend and specialization variants successfully compiled.
        backend_assessments: Capability assessments produced before lowering.
        artifact_records: Validated install metadata for produced native artifacts.
        region_skipped: Typed-region variants rejected or failed during compilation.
        verification_steps: Isolated wheel and payload verification evidence.
        test_results: Target-project command evidence used by quality gates.
        performance: Paired performance-gate evidence.
        profile: Unmeasured baseline profile and hot-candidate selection evidence.
        candidate_trials: Greedy marginal-profitability decisions in profile order.
        execution_plans: Selected and rejected scheduler execution-plan candidates.
        fusion_plans: Deterministic report-only task-fusion safety decisions.
        fusion_trials: Three-arm research trials run only for eligible generated variants.
    """

    success: bool
    project_root: Path
    output_dir: Path
    install_root: Path
    wheel_path: Path | None
    islands: tuple[EnabledIslandConfig, ...]
    build: CompileAttempt
    install_tree_kept: bool = False
    cleanup_removed: tuple[Path, ...] = ()
    cleanup_kept: tuple[Path, ...] = ()
    report_artifact_paths: tuple[Path, ...] = ()
    error: str | None = None
    skipped: tuple[PackageBuildFailure, ...] = ()
    preflight_skipped: tuple[PackagePreflightFailure, ...] = ()
    native_readiness: tuple[NativeReadiness, ...] = ()
    typed_regions: tuple[TypedRegion, ...] = ()
    compiled_regions: tuple[TypedRegion, ...] = ()
    compiled_bindings: tuple[BindingTarget, ...] = ()
    compiled_variants: tuple[CompiledRegionVariant, ...] = ()
    backend_assessments: tuple[BackendAssessment, ...] = ()
    artifact_records: tuple[ArtifactRecord, ...] = ()
    region_skipped: tuple[PackageRegionBuildFailure, ...] = ()
    verification_steps: tuple[PackageVerificationResult, ...] = ()
    test_results: tuple[CommandRunEvidence, ...] = ()
    performance: BenchmarkGateResult | None = None
    profile: ProfileResult | None = None
    candidate_trials: tuple[CandidateTrial, ...] = ()
    execution_plans: tuple[ExecutionPlan | PlanRejection, ...] = ()
    fusion_plans: tuple[FusionPlan, ...] = ()
    fusion_trials: tuple[FusionTrial, ...] = ()


@dataclass(frozen=True, slots=True)
class PackageBuildFailure:
    """A selected island that could not be compiled into the artifact package.

    Attributes:
        island: Enabled island affected by the command.
        build: Captured native build evidence.
    """

    island: EnabledIslandConfig
    build: CompileAttempt


@dataclass(frozen=True, slots=True)
class PackagePreflightFailure:
    """A selected module skipped before build because mypyc rejects module-level code.

    Attributes:
        scan: Cached module scan payload.
        blockers: Conservative blockers attached to this module or symbol.
    """

    scan: ModuleScan
    blockers: tuple[Blocker, ...]


@dataclass(frozen=True, slots=True)
class PackageRegionBuildFailure:
    """One typed region retained as interpreted after backend failure.

    Attributes:
        region: Backend-neutral typed region represented by this record.
        variant_id: Stable backend/specialization variant identifier.
        backend: Native compiler backend selected for this record.
        assessment: Backend capability assessment associated with the build failure.
        build: Captured native build evidence.
    """

    region: TypedRegion
    variant_id: str
    backend: Backend
    assessment: BackendAssessment
    build: CompileAttempt


@dataclass(frozen=True, slots=True)
class _ProjectMetadata:
    name: str
    version: str
    requires_python: str | None
    dependencies: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SelectedTypedRegion:
    """One typed region and member subset selected for a backend variant.

    Attributes:
        scan: Module scan containing retained source facts.
        region: Backend-neutral typed region represented by the state.
        variant_id: Stable backend and specialization variant ID.
        backend: Compiler backend selected for this state.
        assessment: Backend capability assessment for the selected region.
        members: Typed-region members included in this state.
        bound_members: Members already assigned to another region or specialization.
        specialization: Concrete guarded specialization for the variant.
        conditional_on_failure_of: Preferred variant that must fail before this fallback runs.
        source_region_id: Connected scan-region ID used to rebuild a directed staged slice.
        slice_root: Hot or explicitly requested binding at the root of a directed slice.
    """

    scan: ModuleScan
    region: TypedRegion
    variant_id: str
    backend: Backend
    assessment: BackendAssessment
    members: tuple[SymbolId, ...]
    bound_members: tuple[SymbolId, ...] | None = None
    specialization: RegionSpecialization | None = None
    conditional_on_failure_of: str | None = None
    source_region_id: str | None = None
    slice_root: SymbolId | None = None


@dataclass(frozen=True, slots=True)
class _RequestedCallableVariant:
    """Inputs for selecting one backend for a directed callable slice.

    Attributes:
        scan: Module scan containing retained source facts.
        region: Directed backend-neutral source slice.
        closure: Members required in the generated compilation unit.
        requested: Public bindings promised by this slice.
        backends: Backends considered in configured preference order.
        source_region_id: Connected scan-region ID owning the slice.
        slice_root: Public root binding promised by the slice.
    """

    scan: ModuleScan
    region: TypedRegion
    closure: tuple[SymbolId, ...]
    requested: frozenset[SymbolId]
    backends: tuple[Backend, ...]
    source_region_id: str
    slice_root: SymbolId


@dataclass(frozen=True, slots=True)
class _PreparedTypedRegion:
    """Generated unit plus its staged runtime binding contract.

    Attributes:
        generation: Generated typed-region source and binding metadata.
        assessment: Backend capability assessment for the selected region.
        unit: Prepared backend compilation unit.
        shim: Managed region shim edit for the staged source.
        fallback: Fallback variant attempted after the preferred backend fails.
        conditional_on_failure_of: Preferred variant that must fail before this fallback runs.
        lowering_mode: Whether the compiled target owns the whole callable or native blocks.
        native_helpers: Private native helper names used by an outlined Python shell.
        fallback_reason: Ordered deterministic backend failures preceding this successful variant.
    """

    generation: TypedRegionGeneration
    assessment: BackendAssessment
    unit: CompilationUnit
    shim: RegionShimConfig
    fallback: _PreparedTypedRegion | None = None
    conditional_on_failure_of: str | None = None
    lowering_mode: LoweringMode = "whole-callable"
    native_helpers: tuple[str, ...] = ()
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        """Require outlined variants to identify every native helper.

        Raises:
            ValueError: If lowering mode and helper metadata contradict each other.
        """
        if self.lowering_mode == "outlined-block" and not self.native_helpers:
            raise ValueError("outlined prepared regions require native helpers")
        if self.lowering_mode == "whole-callable" and self.native_helpers:
            raise ValueError("whole-callable prepared regions cannot declare native helpers")


@dataclass(frozen=True, slots=True)
class _TypedRegionBuildOutcome:
    """Per-region backend results aggregated for source-clean packaging.

    Attributes:
        successful: Successfully compiled typed-region variants.
        build: Captured native compilation attempt.
        artifacts: Validated native artifacts produced by successful variants.
        skipped: Region variants rejected or failed during compilation.
        cache_statuses: Cache outcomes collected across compiled variants.
    """

    successful: tuple[_PreparedTypedRegion, ...]
    build: CompileAttempt
    artifacts: tuple[ArtifactRecord, ...]
    skipped: tuple[PackageRegionBuildFailure, ...]
    cache_statuses: tuple[tuple[str, CompileCacheStatus], ...] = ()


@dataclass(frozen=True, slots=True)
class _TypedRegionBuildContext:
    """Filesystem, cache, and progress boundaries shared by region builds.

    Attributes:
        build_root: Root of the temporary source-clean build tree.
        staged_source_roots: All import roots inside source-clean staging.
        mypy_cache_dir: Mypy cache directory used by the native build.
        compile_cache_dir: Cache directory used for native region compilation.
        progress: Optional progress callback.
    """

    build_root: Path
    staged_source_roots: tuple[Path, ...]
    mypy_cache_dir: Path
    compile_cache_dir: Path
    progress: PackageProgress | None


@dataclass(frozen=True, slots=True)
class _StagedTypedRegionContext:
    """Copied source evidence shared by primary and fallback backend variants.

    Attributes:
        build_root: Root of the temporary source-clean build tree.
        staged_source_root: Primary import root inside source-clean staging.
        module: Module identity or syntax module associated with the state.
        scan: Module scan containing retained source facts.
        region: Backend-neutral typed region represented by the state.
    """

    build_root: Path
    staged_source_root: Path
    module: ModuleId
    scan: ModuleScan
    region: TypedRegion


@dataclass(frozen=True, slots=True)
class _TypedRegionPackageContext:
    """Selected analysis evidence carried into source-clean region packaging.

    Attributes:
        selected: Backend-supported region selections.
        typed_regions: Backend-neutral typed regions considered by packaging.
        preflight_skipped: Modules rejected before native compilation.
        native_readiness: Post-generation native-readiness evidence.
        execution_plans: Scheduler execution-plan candidates retained for reporting.
        fusion_plans: Deterministic task-fusion safety evidence for profiled scheduler sites.
    """

    selected: tuple[_SelectedTypedRegion, ...]
    typed_regions: tuple[TypedRegion, ...]
    preflight_skipped: tuple[PackagePreflightFailure, ...]
    native_readiness: tuple[NativeReadiness, ...]
    execution_plans: tuple[ExecutionPlan | PlanRejection, ...] = ()
    fusion_plans: tuple[FusionPlan, ...] = ()


@dataclass(frozen=True, slots=True)
class _BaselineWheelPayload:
    """Normal target wheel unpacked as the immutable source-clean base layer.

    Attributes:
        wheel_path: Final or intermediate wheel archive path.
        build: Captured native compilation attempt.
        baseline_install_root: Unpacked baseline payload used for interpreted quality gates.
        quality_project_root: Project root used by quality-gate subprocesses.
        semantic_test_result: Baseline semantic-test evidence collected before profiling.
    """

    wheel_path: Path | None
    build: CompileAttempt
    baseline_install_root: Path | None = None
    quality_project_root: Path | None = None
    semantic_test_result: CommandRunEvidence | None = None


@dataclass(frozen=True, slots=True)
class _QualityGateOutcome:
    """Configured semantic-test and benchmark evidence before wheel promotion.

    Attributes:
        success: Whether this operation completed successfully.
        tests: Optional semantic test result.
        performance: Paired performance-gate result.
        error: User-facing failure text, or `None` on success.
    """

    success: bool
    tests: tuple[CommandRunEvidence, ...]
    performance: BenchmarkGateResult
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _SourceCleanPromotionContext:
    """Shared inputs for verifying, gating, and promoting one staged payload.

    Attributes:
        options: Validated command or generation options.
        project: Discovered target project.
        output_dir: Directory receiving wheel artifacts.
        build_root: Root of the temporary source-clean build tree.
        install_root: Temporary source-clean payload root.
        baseline: Baseline wheel build evidence.
        verification_plan: Expected modules, regions, and artifacts for isolated verification.
        build: Captured native compilation attempt.
        requires_native_artifact: Whether profile-guided selection must retain at least one region.
    """

    options: PackageOptions
    project: DiscoveredProject
    output_dir: Path
    build_root: Path
    install_root: Path
    baseline: _BaselineWheelPayload
    verification_plan: PackageVerificationPlan
    build: CompileAttempt
    requires_native_artifact: bool = False


@dataclass(frozen=True, slots=True)
class _SourceCleanPromotionResult:
    """Final wheel, gate evidence, and cleanup state for a staged payload.

    Attributes:
        success: Whether this operation completed successfully.
        wheel_path: Final or intermediate wheel archive path.
        build: Captured native compilation attempt.
        verification_steps: Isolated payload and wheel verification evidence.
        test_results: Target-project command evidence.
        performance: Paired performance-gate result.
        cleanup_removed: Generated paths removed after completion.
        cleanup_kept: Generated paths retained for diagnostics.
        error: User-facing failure text, or `None` on success.
    """

    success: bool
    wheel_path: Path | None
    build: CompileAttempt
    verification_steps: tuple[PackageVerificationResult, ...]
    test_results: tuple[CommandRunEvidence, ...] = ()
    performance: BenchmarkGateResult | None = None
    cleanup_removed: tuple[Path, ...] = ()
    cleanup_kept: tuple[Path, ...] = ()
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _SourceCleanPromotionFailure:
    """Evidence needed to reject a staged source-clean wheel candidate.

    Attributes:
        build: Captured native compilation attempt.
        verification_steps: Isolated payload and wheel verification evidence.
        error: User-facing failure text, or `None` on success.
        wheel_path: Final or intermediate wheel archive path.
        quality_gate: Test and benchmark outcome for staged artifacts.
    """

    build: CompileAttempt
    verification_steps: tuple[PackageVerificationResult, ...]
    error: str | None
    wheel_path: Path | None = None
    quality_gate: _QualityGateOutcome | None = None


@dataclass(frozen=True, slots=True)
class _ProfilePreparation:
    """Baseline and profile state prepared before static region selection.

    Attributes:
        baseline: Built baseline wheel payload, when profile-guided selection is configured.
        profile: Unmeasured profile evidence, when profiling reached a supported launcher.
        failure: Early failure that must stop region scanning and native compilation.
        execution_plans: Scheduler execution-plan candidates formed from static or profile evidence.
        fusion_plans: Deterministic task-fusion safety evidence formed after profile selection.
    """

    baseline: _BaselineWheelPayload | None = None
    profile: ProfileResult | None = None
    failure: PackageCommandResult | None = None
    execution_plans: tuple[ExecutionPlan | PlanRejection, ...] = ()
    fusion_plans: tuple[FusionPlan, ...] = ()


@dataclass(frozen=True, slots=True)
class _ProfitabilityCandidate:
    """Compiled variant plus profile evidence used by greedy selection.

    Attributes:
        prepared: Successfully compiled variant available in the superset payload.
        symbols: Profiled public bindings represented by the candidate.
        profile_samples: Mapped project samples attributed to the candidate.
        profile_coverage: Fraction of mapped project samples attributed to the candidate.
        fallback_reason: Deterministic compiler rejection that selected a fallback variant.
    """

    prepared: _PreparedTypedRegion
    symbols: tuple[str, ...]
    profile_samples: int
    profile_coverage: float
    fallback_reason: str | None


@dataclass(frozen=True, slots=True)
class _ProfitabilitySelectionOutcome:
    """Accepted variants and diagnostic timings from greedy candidate trials.

    Attributes:
        accepted: Successful variants retained in the final payload.
        trials: Marginal semantic and benchmark decisions in profile order.
        timings: Command timings appended to the overall compile attempt.
    """

    accepted: tuple[_PreparedTypedRegion, ...]
    trials: tuple[CandidateTrial, ...] = ()
    timings: tuple[CompilePhaseTiming, ...] = ()


@dataclass(frozen=True, slots=True)
class _ProfitabilitySelectionContext:
    """Inputs required to evaluate compiled candidates without rebuilding them.

    Attributes:
        successful: Successfully compiled variants available in the superset payload.
        skipped: Backend failures that may explain fallback selection.
        profile: Baseline profile supplying candidate order and sample attribution.
        project: Discovered target project configuration and modules.
        baseline: Baseline wheel and source-stripped quality-project evidence.
        payload_root: Superset staged payload controlled by the internal allowlist.
        progress: Optional progress callback for long-running trials.
    """

    successful: tuple[_PreparedTypedRegion, ...]
    skipped: tuple[PackageRegionBuildFailure, ...]
    profile: ProfileResult
    project: DiscoveredProject
    baseline: _BaselineWheelPayload
    payload_root: Path
    progress: PackageProgress | None


@dataclass(frozen=True, slots=True)
class _TypedPayloadFinalizationContext:
    """Superset payload state needed to apply profile profitability decisions.

    Attributes:
        options: Validated command or generation options.
        project: Discovered target project configuration and modules.
        profile: Dynamic profile used for candidate order, when available.
        baseline: Immutable interpreted payload used to rebuild the final install tree.
        install_root: Superset payload evaluated by candidate trials.
        staged_source_roots: Copied source roots containing shims and artifacts.
        outcome: Native build results for every successfully compiled candidate.
        overlay_error: Failure from staging the candidate superset, when present.
    """

    options: PackageOptions
    project: DiscoveredProject
    profile: ProfileResult | None
    baseline: _BaselineWheelPayload
    install_root: Path
    staged_source_roots: tuple[Path, ...]
    outcome: _TypedRegionBuildOutcome
    overlay_error: str | None


@dataclass(frozen=True, slots=True)
class _TypedPayloadFinalizationResult:
    """Accepted payload subset and evidence passed to final promotion.

    Attributes:
        successful: Variants retained in the final staged payload.
        artifacts: Native artifact records reachable from retained variants.
        build: Build evidence including candidate semantic and benchmark timings.
        trials: Greedy marginal-profitability decisions.
        overlay_error: Failure rebuilding the accepted payload, when present.
        profitability_applied: Whether dynamic candidate selection owned the final subset.
    """

    successful: tuple[_PreparedTypedRegion, ...]
    artifacts: tuple[ArtifactRecord, ...]
    build: CompileAttempt
    trials: tuple[CandidateTrial, ...]
    overlay_error: str | None
    profitability_applied: bool


@dataclass(frozen=True, slots=True)
class _FusionResearchOutcome:
    """Conditional three-arm task-fusion evidence and measured phase timings.

    Attributes:
        trials: Plan-bound semantic and profitability decisions.
        timings: Subprocess timings appended to the compile report.
    """

    trials: tuple[FusionTrial, ...] = ()
    timings: tuple[CompilePhaseTiming, ...] = ()


@dataclass(frozen=True, slots=True)
class _FusionResearchContext:
    """Inputs for conditional eager-task trials after the safe gate misses.

    Attributes:
        options: Command options supplying progress and quality-gate ownership.
        project: Discovered target project and compile policy.
        baseline: Immutable interpreted payload and quality-project roots.
        build_root: Temporary build root owning disposable research copies.
        install_root: Final unfused compiled payload.
        plans: Safety plans emitted from the baseline profile.
        accepted: Native variants retained by greedy candidate selection.
        performance: Full safe-payload benchmark decision.
    """

    options: PackageOptions
    project: DiscoveredProject
    baseline: _BaselineWheelPayload
    build_root: Path
    install_root: Path
    plans: tuple[FusionPlan, ...]
    accepted: tuple[_PreparedTypedRegion, ...]
    performance: BenchmarkGateResult | None


def execute_package(options: PackageOptions) -> PackageCommandResult:
    """Build a source-clean wheel from backend-neutral typed regions.

    Generated sidecars and generated-code native-readiness analysis are
    deliberately excluded from this command. Explicit in-place workflows retain
    their sidecar implementation, while default compile uses scanner IR, backend
    assessments, and per-region fallback throughout.

    Args:
        options: Validated command options supplied by the CLI layer.

    Returns:
        PackageCommandResult: Complete source-clean wheel build, verification, and quality-gate
            evidence.
    """
    _progress(options.progress, f"discovering project at {options.root.resolve()}")
    project = discover_project(options.root)
    _progress(
        options.progress,
        f"discovered {len(project.modules)} module(s); scan scope: {options.module_name or 'all'}",
    )
    scan_started = time.perf_counter()
    scans = _selected_scans(project, options.module_name, options.selected_members)
    typed_regions = tuple(region for scan in scans for region in scan.typed_regions)
    execution_plans = build_execution_plans(scans, None)
    _progress(options.progress, f"scanned {len(scans)} module(s) in {_duration(scan_started)}")
    preflight_selected = _selected_typed_regions(
        scans,
        project.config.compile.backends,
        options.selected_members,
    )
    preflight_missing = _missing_requested_members(options.selected_members, preflight_selected)
    profile_candidates = (
        _profile_candidate_members(scans, project.config.compile.backends)
        if project.config.compile.benchmark_command is not None and not options.selected_members
        else ()
    )
    preflight_error = _region_selection_error(
        profile=None,
        selection_members=options.selected_members,
        profile_members=(),
        selected=preflight_selected,
        missing=preflight_missing,
    )
    if preflight_error is not None and not profile_candidates:
        _remove_failed_wheels(
            project,
            _resolve_output_dir(project.config.root, options.output_dir),
        )
        return _failed_result(
            project.config.root,
            options.output_dir,
            preflight_error,
            typed_regions=typed_regions,
            execution_plans=execution_plans,
        )
    preparation = _prepare_profile_guided_selection(options, project, scans)
    if preparation.failure is not None:
        return preparation.failure
    baseline = preparation.baseline
    profile = preparation.profile
    if profile is not None:
        profile = select_profile_candidates(
            profile,
            tuple(symbol for scan in scans for symbol in scan.symbols),
        )
        _progress(
            options.progress,
            (
                f"profile selected {len(profile.selected_symbols)} hot member(s) covering "
                f"{profile.selected_hot_coverage:.1%} of mapped project samples"
            ),
        )
    execution_plans = build_execution_plans(scans, profile)
    if execution_plans:
        selected_execution_plans = sum(isinstance(plan, ExecutionPlan) for plan in execution_plans)
        _progress(
            options.progress,
            (
                f"execution-plan discovery produced {len(execution_plans)} candidate(s); "
                f"{selected_execution_plans} selected for future backend assessment"
            ),
        )
    fusion_plans = build_fusion_plans(scans, profile) if profile is not None else ()
    if fusion_plans:
        eligible_fusion_plans = sum(plan.eligible for plan in fusion_plans)
        _progress(
            options.progress,
            (
                f"task-fusion planning produced {len(fusion_plans)} plan(s); "
                f"{eligible_fusion_plans} passed every safety gate"
            ),
        )
    preparation = replace(
        preparation,
        profile=profile,
        execution_plans=execution_plans,
        fusion_plans=fusion_plans,
    )
    profile_members = (
        profile.selected_symbols if profile is not None and profile.status == "profiled" else ()
    )
    selection_members = options.selected_members or profile_members
    selected_typed_regions = (
        _selected_typed_regions(
            scans,
            project.config.compile.backends,
            selection_members,
            hot_members=profile_members,
        )
        if profile_members and not options.selected_members
        else preflight_selected
    )
    _progress_compile_selection(
        options.progress,
        selected_typed_regions,
        requested_members=selection_members,
    )
    missing_members = _missing_requested_members(
        options.selected_members,
        selected_typed_regions,
    )
    selection_error = _region_selection_error(
        profile=profile,
        selection_members=selection_members,
        profile_members=profile_members,
        selected=selected_typed_regions,
        missing=missing_members,
    )
    if selection_error is not None:
        return _failed_region_selection(
            options=options,
            project=project,
            preparation=preparation,
            error=selection_error,
            typed_regions=typed_regions,
        )
    return _execute_typed_region_package(
        options=options,
        project=project,
        context=_TypedRegionPackageContext(
            selected=selected_typed_regions,
            typed_regions=typed_regions,
            preflight_skipped=(),
            native_readiness=(),
            execution_plans=execution_plans,
            fusion_plans=fusion_plans,
        ),
        prepared_baseline=baseline,
        profile=profile,
    )


def _execute_typed_region_package(
    *,
    options: PackageOptions,
    project: DiscoveredProject,
    context: _TypedRegionPackageContext,
    prepared_baseline: _BaselineWheelPayload | None = None,
    profile: ProfileResult | None = None,
) -> PackageCommandResult:
    """Build source-clean class and callable region variants.

    Generation and shims live only in copied build roots. Regions compile
    independently so one backend rejection leaves successful regions available
    in the wheel while preserving the original implementation as fallback.

    Args:
        options: Validated command or generation options.
        project: Discovered target project configuration and modules.
        context: Prepared state shared by this operation.
        prepared_baseline: Baseline wheel already tested and profiled before region selection.
        profile: Unmeasured baseline profile and selected hot-candidate evidence.

    Returns:
        PackageCommandResult: Complete source-clean package result for selected regions.
    """
    output_dir = _resolve_output_dir(project.config.root, options.output_dir)
    build_root = output_dir / "build"
    install_root = output_dir / "install"
    baseline = _package_baseline(options, project, prepared_baseline)
    if baseline.wheel_path is None:
        _remove_failed_wheels(project, output_dir)
        cleanup_removed = _remove_tree(install_root)
        return PackageCommandResult(
            success=False,
            project_root=project.config.root,
            output_dir=output_dir,
            install_root=install_root,
            wheel_path=None,
            islands=(),
            build=baseline.build,
            cleanup_removed=cleanup_removed,
            cleanup_kept=(build_root,),
            error=baseline.build.stderr,
            preflight_skipped=context.preflight_skipped,
            native_readiness=context.native_readiness,
            typed_regions=context.typed_regions,
            backend_assessments=tuple(selection.assessment for selection in context.selected),
            profile=profile,
            execution_plans=context.execution_plans,
            fusion_plans=context.fusion_plans,
        )

    copy_started = time.perf_counter()
    _progress(options.progress, "copying source roots into temporary build tree")
    staged_source_roots = _copy_source_roots(project, build_root)
    _progress(options.progress, f"copied source roots in {_duration(copy_started)}")

    generation_started = time.perf_counter()
    prepared: list[_PreparedTypedRegion] = []
    preparation_failures: list[PackageRegionBuildFailure] = []
    for selection in context.selected:
        try:
            prepared.append(
                _prepare_typed_region(
                    project=project,
                    build_root=build_root,
                    staged_source_roots=staged_source_roots,
                    selection=selection,
                )
            )
        except (SyntaxError, ValueError) as lowering_error:
            preparation_failures.append(
                PackageRegionBuildFailure(
                    region=selection.region,
                    variant_id=selection.variant_id,
                    backend=selection.backend,
                    assessment=selection.assessment,
                    build=_failed_region_attempt(f"typed-region lowering failed: {lowering_error}"),
                )
            )
    _progress(
        options.progress,
        (
            f"lowered {len(prepared)} typed region backend variant(s); "
            f"kept {len(preparation_failures)} as fallback in {_duration(generation_started)}"
        ),
    )
    if not prepared:
        result_error = "No selected typed regions could be lowered for a compiler backend."
        _remove_failed_wheels(project, output_dir)
        cleanup_removed = _remove_tree(install_root)
        return PackageCommandResult(
            success=False,
            project_root=project.config.root,
            output_dir=output_dir,
            install_root=install_root,
            wheel_path=None,
            islands=(),
            build=_failed_region_attempt(result_error),
            cleanup_removed=cleanup_removed,
            cleanup_kept=(build_root,),
            error=result_error,
            preflight_skipped=context.preflight_skipped,
            native_readiness=context.native_readiness,
            typed_regions=context.typed_regions,
            backend_assessments=tuple(selection.assessment for selection in context.selected),
            region_skipped=tuple(preparation_failures),
            profile=profile,
            execution_plans=context.execution_plans,
            fusion_plans=context.fusion_plans,
        )

    prepared_assessments = _prepared_backend_assessments(tuple(prepared))
    outcome = _build_typed_regions(
        prepared=tuple(prepared),
        context=_TypedRegionBuildContext(
            build_root=build_root,
            staged_source_roots=staged_source_roots,
            mypy_cache_dir=(options.cache_dir or project.config.cache_dir)
            / "mypy"
            / "source-clean",
            compile_cache_dir=(options.cache_dir or project.config.cache_dir)
            / "compile"
            / "regions",
            progress=options.progress,
        ),
        initial_failures=tuple(preparation_failures),
    )
    outcome = replace(outcome, build=_combine_baseline_and_native(baseline.build, outcome.build))
    if not outcome.successful:
        _progress(options.progress, "all typed-region builds failed; keeping diagnostics")
        _remove_failed_wheels(project, output_dir)
        cleanup_removed = _remove_tree(install_root)
        return PackageCommandResult(
            success=False,
            project_root=project.config.root,
            output_dir=output_dir,
            install_root=install_root,
            wheel_path=None,
            islands=(),
            build=outcome.build,
            cleanup_removed=cleanup_removed,
            cleanup_kept=(build_root,),
            error=outcome.build.stderr,
            preflight_skipped=context.preflight_skipped,
            native_readiness=context.native_readiness,
            typed_regions=context.typed_regions,
            backend_assessments=prepared_assessments,
            artifact_records=outcome.artifacts,
            region_skipped=outcome.skipped,
            profile=profile,
            execution_plans=context.execution_plans,
            fusion_plans=context.fusion_plans,
        )

    successful_bindings = tuple(
        binding for item in outcome.successful for binding in item.generation.bindings
    )
    compiled_sources = frozenset(binding.source for binding in successful_bindings)
    missing_compiled = tuple(
        member for member in options.selected_members if member not in compiled_sources
    )
    if missing_compiled:
        missing_text = ", ".join(member.stable_id for member in missing_compiled)
        error = f"requested member(s) did not compile successfully: {missing_text}"
        failed_build = replace(outcome.build, success=False, stderr=error)
        _remove_failed_wheels(project, output_dir)
        cleanup_removed = _remove_tree(install_root)
        return PackageCommandResult(
            success=False,
            project_root=project.config.root,
            output_dir=output_dir,
            install_root=install_root,
            wheel_path=None,
            islands=(),
            build=failed_build,
            cleanup_removed=cleanup_removed,
            cleanup_kept=(build_root,),
            error=error,
            preflight_skipped=context.preflight_skipped,
            native_readiness=context.native_readiness,
            typed_regions=context.typed_regions,
            compiled_regions=tuple(
                {
                    item.generation.region.id: item.generation.region for item in outcome.successful
                }.values()
            ),
            compiled_bindings=successful_bindings,
            backend_assessments=prepared_assessments,
            artifact_records=outcome.artifacts,
            region_skipped=outcome.skipped,
            profile=profile,
            execution_plans=context.execution_plans,
            fusion_plans=context.fusion_plans,
        )

    payload_started = time.perf_counter()
    _progress(options.progress, "binding compiled classes and callables in staged modules")
    overlay_error = _stage_compiled_superset(
        outcome,
        tuple(prepared),
        staged_source_roots,
        install_root,
    )
    finalized = _finalize_typed_payload(
        _TypedPayloadFinalizationContext(
            options=options,
            project=project,
            profile=profile,
            baseline=baseline,
            install_root=install_root,
            staged_source_roots=staged_source_roots,
            outcome=outcome,
            overlay_error=overlay_error,
        )
    )
    accepted_successful = finalized.successful
    accepted_shims = tuple(item.shim for item in accepted_successful)
    accepted_artifacts = finalized.artifacts
    report_artifact_paths = _source_clean_region_report_artifact_paths(
        project.config.root,
        accepted_artifacts,
    )
    verification_plan = _typed_verification_plan(accepted_shims, accepted_artifacts)
    promotion_context = _SourceCleanPromotionContext(
        options=options,
        project=project,
        output_dir=output_dir,
        build_root=build_root,
        install_root=install_root,
        baseline=baseline,
        verification_plan=verification_plan,
        build=finalized.build,
        requires_native_artifact=finalized.profitability_applied,
    )
    if finalized.overlay_error is None:
        _progress(options.progress, f"prepared install payload in {_duration(payload_started)}")
        promotion = _promote_source_clean_payload(promotion_context)
    else:
        promotion = _failed_promotion(
            promotion_context,
            _SourceCleanPromotionFailure(
                build=replace(
                    finalized.build,
                    success=False,
                    stderr=finalized.overlay_error,
                ),
                verification_steps=(),
                error=finalized.overlay_error,
            ),
        )
    promotion, fusion_research = _attach_conditional_task_fusion_research(
        promotion,
        _FusionResearchContext(
            options=options,
            project=project,
            baseline=baseline,
            build_root=build_root,
            install_root=install_root,
            plans=context.fusion_plans,
            accepted=accepted_successful,
            performance=promotion.performance,
        ),
    )
    cache_statuses = dict(outcome.cache_statuses)
    successful_regions = tuple(
        {item.generation.region.id: item.generation.region for item in accepted_successful}.values()
    )
    successful_bindings = tuple(
        binding for item in accepted_successful for binding in item.generation.bindings
    )
    successful_variants = tuple(
        CompiledRegionVariant(
            id=item.unit.region_id,
            region=item.generation.region,
            backend=item.generation.backend,
            bindings=item.generation.bindings,
            cache_status=cache_statuses.get(item.unit.region_id, "disabled"),
            lowering_mode=item.lowering_mode,
            native_helpers=item.native_helpers,
        )
        for item in accepted_successful
    )
    return PackageCommandResult(
        success=promotion.success,
        project_root=project.config.root,
        output_dir=output_dir,
        install_root=install_root,
        wheel_path=promotion.wheel_path,
        islands=(),
        build=promotion.build,
        install_tree_kept=options.keep_install_tree and promotion.success,
        cleanup_removed=promotion.cleanup_removed,
        cleanup_kept=promotion.cleanup_kept,
        report_artifact_paths=report_artifact_paths,
        error=promotion.error,
        preflight_skipped=context.preflight_skipped,
        native_readiness=context.native_readiness,
        typed_regions=context.typed_regions,
        compiled_regions=successful_regions,
        compiled_bindings=successful_bindings,
        compiled_variants=successful_variants,
        backend_assessments=prepared_assessments,
        artifact_records=accepted_artifacts,
        region_skipped=outcome.skipped,
        verification_steps=promotion.verification_steps,
        test_results=promotion.test_results,
        performance=promotion.performance,
        profile=profile,
        candidate_trials=finalized.trials,
        execution_plans=context.execution_plans,
        fusion_plans=context.fusion_plans,
        fusion_trials=fusion_research.trials,
    )


def _prepare_typed_region(
    *,
    project: DiscoveredProject,
    build_root: Path,
    staged_source_roots: tuple[Path, ...],
    selection: _SelectedTypedRegion,
) -> _PreparedTypedRegion:
    staged_module = _staged_module(selection.scan.module, project, staged_source_roots)
    staged_scan = enrich_island_analysis(scan_module(staged_module))
    staged_selection = _staged_typed_selection(staged_scan, selection)
    staged_region = staged_selection.region
    staged = _StagedTypedRegionContext(
        build_root=build_root,
        staged_source_root=_staged_source_root(
            selection.scan.module,
            project,
            staged_source_roots,
        ),
        module=staged_module,
        scan=staged_scan,
        region=staged_region,
    )
    try:
        prepared = _prepare_backend_variant(staged, staged_selection)
    except (SyntaxError, ValueError):
        if staged_selection.backend != "cython":
            raise
        return _prepare_outlined_backend_variant(staged, staged_selection)
    if staged_selection.backend != "mypyc":
        return _attach_outlined_fallback(staged, staged_selection, prepared)
    cython_assessment = CYTHON_BACKEND.assess(staged_region)
    if not set(staged_selection.members) <= set(cython_assessment.supported_members):
        return prepared
    fallback_selection = replace(
        staged_selection,
        variant_id=f"{staged_region.id}@cython-mypyc-fallback",
        backend="cython",
        assessment=cython_assessment,
    )
    try:
        fallback = _prepare_backend_variant(staged, fallback_selection)
    except (SyntaxError, ValueError):
        try:
            fallback = _prepare_outlined_backend_variant(staged, fallback_selection)
        except (SyntaxError, ValueError):
            return prepared
    else:
        fallback = _attach_outlined_fallback(staged, fallback_selection, fallback)
    return replace(prepared, fallback=fallback)


def _attach_outlined_fallback(
    staged: _StagedTypedRegionContext,
    selection: _SelectedTypedRegion,
    prepared: _PreparedTypedRegion,
) -> _PreparedTypedRegion:
    """Append an outlined Cython fallback when one suspension binding is precise.

    Args:
        staged: Staged source-clean package and native build context.
        selection: Whole-callable Cython selection being prepared.
        prepared: Successfully lowered whole-callable variant.

    Returns:
        _PreparedTypedRegion: Whole-callable variant with an optional outlined fallback.
    """
    try:
        outlined = _prepare_outlined_backend_variant(staged, selection)
    except (SyntaxError, ValueError):
        return prepared
    return replace(prepared, fallback=outlined)


def _prepare_outlined_backend_variant(
    staged: _StagedTypedRegionContext,
    selection: _SelectedTypedRegion,
) -> _PreparedTypedRegion:
    """Lower planner-approved synchronous blocks into one private Cython unit.

    Args:
        staged: Staged source-clean package and native build context.
        selection: Cython selection with exactly one promised suspension binding.

    Returns:
        _PreparedTypedRegion: Outlined native helpers and staged Python shell contract.

    Raises:
        ValueError: If specialization, binding selection, or suspension planning is ambiguous.
    """
    if selection.backend != "cython" or selection.specialization is not None:
        raise ValueError("outlined lowering requires an unspecialized Cython selection")
    binding = _outlined_binding(selection)
    variant_id = f"{staged.region.id}@cython-outline"
    logical_module = _typed_region_module_name(staged.region, "cython", variant_id)
    generated_path = staged.build_root / f"{logical_module}.py"
    outlined = generate_outlined_region(
        staged.region,
        binding.source,
        binding,
        output_path=generated_path,
    )
    generation = TypedRegionGeneration(
        region=staged.region,
        logical_module=logical_module,
        source_path=outlined.source_path,
        source_text=outlined.source_text,
        source_hash=outlined.source_hash,
        selected_members=(binding.source,),
        bindings=(binding,),
        backend="cython",
    )
    unit = CYTHON_BACKEND.lower(
        BackendLoweringRequest(
            region=staged.region,
            source_path=generated_path,
            logical_module=logical_module,
            install_relative_dir=_region_artifact_relative_dir(variant_id),
            members=(binding.source,),
            variant_id=variant_id,
        )
    )
    return _PreparedTypedRegion(
        generation=generation,
        assessment=CYTHON_BACKEND.assess(staged.region),
        unit=unit,
        shim=RegionShimConfig(
            source_module=staged.module.name,
            source_path=staged.module.path,
            region_id=variant_id,
            backend="cython",
            compiled_module=logical_module,
            artifact_dir=staged.staged_source_root / unit.install_relative_dir,
            bindings=(binding,),
            outlined_shell=outlined.shell,
        ),
        conditional_on_failure_of=selection.conditional_on_failure_of,
        lowering_mode="outlined-block",
        native_helpers=outlined.helper_names,
    )


def _outlined_binding(selection: _SelectedTypedRegion) -> BindingTarget:
    promised = (
        frozenset(selection.bound_members)
        if selection.bound_members is not None
        else frozenset(selection.members)
    )
    if selection.slice_root is not None:
        promised = frozenset((selection.slice_root,))
    bindings = tuple(
        binding
        for binding in selection.region.bindings
        if binding.source in promised
        and binding.execution_kind in {"coroutine", "generator", "async_generator"}
    )
    if len(bindings) != 1:
        raise ValueError("outlined lowering requires exactly one promised suspension binding")
    return bindings[0]


def _staged_typed_selection(
    staged_scan: ModuleScan,
    selection: _SelectedTypedRegion,
) -> _SelectedTypedRegion:
    """Rebind a deterministic selection to equivalent evidence in the copied tree.

    Args:
        staged_scan: Scan of the staged module copy.
        selection: One selected region and backend assessment.

    Returns:
        _SelectedTypedRegion: Prepared staged source, generation metadata, and shim edit.

    Raises:
        ValueError: If staged scanning changes a directed slice or omits its root evidence.
    """
    if selection.source_region_id is not None:
        if selection.slice_root is None:
            raise ValueError("directed staged selection requires a slice root")
        staged_source_region = next(
            region
            for region in staged_scan.typed_regions
            if region.id == selection.source_region_id
        )
        staged_region = build_directed_region_slice(
            staged_source_region,
            selection.slice_root,
        )
        if staged_region.id != selection.region.id:
            raise ValueError(
                "staged directed slice differs from checkout analysis: "
                f"{selection.region.id} != {staged_region.id}"
            )
        backend = _compiler_backend(selection.backend)
        return replace(
            selection,
            scan=staged_scan,
            region=staged_region,
            assessment=backend.assess(staged_region),
        )
    if selection.specialization is None:
        staged_region = next(
            region for region in staged_scan.typed_regions if region.id == selection.region.id
        )
        return replace(selection, scan=staged_scan, region=staged_region)
    staged_source_region = next(
        region
        for region in staged_scan.typed_regions
        if any(
            specialization.id == selection.specialization.id
            for specialization in region.specializations
        )
    )
    staged_specialization = next(
        specialization
        for specialization in staged_source_region.specializations
        if specialization.id == selection.specialization.id
    )
    staged_region = _specialized_region(staged_source_region, staged_specialization)
    backend = _compiler_backend(selection.backend)
    return replace(
        selection,
        scan=staged_scan,
        region=staged_region,
        assessment=backend.assess(staged_region),
        members=(staged_specialization.source_member,),
        specialization=staged_specialization,
    )


def _prepare_backend_variant(
    staged: _StagedTypedRegionContext,
    selection: _SelectedTypedRegion,
) -> _PreparedTypedRegion:
    """Lower one selected backend variant inside the copied build tree.

    Args:
        staged: Staged source-clean package and native build context.
        selection: One selected region and backend assessment.

    Returns:
        _PreparedTypedRegion: Prepared backend variant, or a structured skip failure.
    """
    logical_module = _typed_region_module_name(
        staged.region,
        selection.backend,
        selection.variant_id,
    )
    generated_path = staged.build_root / f"{logical_module}.py"
    generation = generate_typed_method_region(
        staged.scan,
        staged.region,
        selection.members,
        output_path=generated_path,
        options=TypedRegionGenerationOptions(
            backend=selection.backend,
            specialization=selection.specialization,
        ),
    )
    if selection.bound_members is not None:
        bound = frozenset(selection.bound_members)
        generation = replace(
            generation,
            bindings=tuple(binding for binding in generation.bindings if binding.source in bound),
        )
    unit = _compiler_backend(selection.backend).lower(
        BackendLoweringRequest(
            region=staged.region,
            source_path=generated_path,
            logical_module=logical_module,
            install_relative_dir=_region_artifact_relative_dir(selection.variant_id),
            members=selection.members,
            variant_id=selection.variant_id,
        )
    )
    return _PreparedTypedRegion(
        generation=generation,
        assessment=selection.assessment,
        unit=unit,
        shim=RegionShimConfig(
            source_module=staged.module.name,
            source_path=staged.module.path,
            region_id=selection.variant_id,
            backend=selection.backend,
            compiled_module=logical_module,
            artifact_dir=staged.staged_source_root / unit.install_relative_dir,
            bindings=generation.bindings,
        ),
        conditional_on_failure_of=selection.conditional_on_failure_of,
    )


def _prepared_source_paths(
    prepared: tuple[_PreparedTypedRegion, ...],
) -> tuple[Path, ...]:
    """Return primary and speculative fallback source paths for cleanup.

    Args:
        prepared: Prepared typed-region variant and generated source.

    Returns:
        tuple[Path, ...]: Source paths consumed by the backend compilation unit.
    """
    paths: list[Path] = []
    for item in prepared:
        candidate: _PreparedTypedRegion | None = item
        while candidate is not None:
            paths.append(candidate.generation.source_path)
            candidate = candidate.fallback
    return tuple(dict.fromkeys(paths))


def _prepared_backend_assessments(
    prepared: tuple[_PreparedTypedRegion, ...],
) -> tuple[BackendAssessment, ...]:
    """Return one assessment per semantic region and prepared backend.

    Args:
        prepared: Preferred variants and their deterministic fallback chains.

    Returns:
        tuple[BackendAssessment, ...]: Deduplicated assessments in preparation order.
    """
    assessments: dict[tuple[str, Backend], BackendAssessment] = {}
    for item in prepared:
        candidate: _PreparedTypedRegion | None = item
        while candidate is not None:
            assessment = candidate.assessment
            assessments.setdefault(
                (assessment.region_id, assessment.backend),
                assessment,
            )
            candidate = candidate.fallback
    return tuple(assessments.values())


def _build_typed_regions(
    *,
    prepared: tuple[_PreparedTypedRegion, ...],
    context: _TypedRegionBuildContext,
    initial_failures: tuple[PackageRegionBuildFailure, ...],
) -> _TypedRegionBuildOutcome:
    successful: list[_PreparedTypedRegion] = []
    skipped = list(initial_failures)
    attempts: list[CompileAttempt] = [failure.build for failure in initial_failures]
    artifacts: list[ArtifactRecord] = []
    cache_statuses: list[tuple[str, CompileCacheStatus]] = []
    successful_promises: set[str] = set()
    backend_context = BackendCompileContext(
        project_root=context.build_root,
        build_dir=context.build_root / ".atoll" / "build",
        source_roots=context.staged_source_roots,
        cache_dir=context.mypy_cache_dir,
        backend_options=(
            ("typed_region_generator", TYPED_METHOD_GENERATOR_VERSION),
            ("outlined_region_generator", OUTLINED_REGION_GENERATOR_VERSION),
        ),
    )
    for index, item in enumerate(prepared, start=1):
        if (
            item.conditional_on_failure_of is not None
            and item.conditional_on_failure_of in successful_promises
        ):
            _progress(
                context.progress,
                f"class variant succeeded; skipped method fallback {item.unit.region_id}",
            )
            continue
        candidate = item
        rejected_attempts: list[tuple[_PreparedTypedRegion, CompileAttempt]] = []
        while True:
            _progress(
                context.progress,
                (
                    f"compiling typed region variant {index}/{len(prepared)} with "
                    f"{candidate.generation.backend} ({candidate.lowering_mode}): "
                    f"{candidate.unit.region_id}"
                ),
            )
            result = _compile_typed_variant(
                candidate,
                backend_context,
                cache_root=context.compile_cache_dir,
            )
            if result.attempt.cache_status == "hit":
                _progress(
                    context.progress,
                    f"compile cache hit for typed region variant {candidate.unit.region_id}",
                )
            elif result.attempt.cache_status == "miss":
                _progress(
                    context.progress,
                    f"compile cache miss for typed region variant {candidate.unit.region_id}",
                )
            tagged_attempt = _tag_region_timings(result.attempt, candidate.unit.region_id)
            if result.attempt.success:
                attempts.extend(
                    _recovered_backend_attempt(attempt, candidate.unit.region_id)
                    for _rejected, attempt in rejected_attempts
                )
                attempts.append(tagged_attempt)
                successful.append(
                    replace(
                        candidate,
                        fallback_reason=_fallback_attempt_reason(tuple(rejected_attempts)),
                    )
                )
                artifacts.extend(result.artifacts)
                cache_statuses.append((candidate.unit.region_id, result.attempt.cache_status))
                successful_promises.add(item.unit.region_id)
                _progress(
                    context.progress,
                    (
                        f"compiled typed region variant {candidate.unit.region_id} "
                        f"as {candidate.lowering_mode}"
                    ),
                )
                break
            fallback = candidate.fallback
            if fallback is not None and _should_retry_with_fallback(candidate, result):
                rejected_attempts.append((candidate, tagged_attempt))
                _progress(
                    context.progress,
                    (
                        f"retrying deterministic {candidate.generation.backend} failure with "
                        f"{fallback.generation.backend} {fallback.lowering_mode}: "
                        f"{fallback.unit.region_id}"
                    ),
                )
                candidate = fallback
                continue
            attempts.extend(attempt for _rejected, attempt in rejected_attempts)
            attempts.append(tagged_attempt)
            skipped.append(
                PackageRegionBuildFailure(
                    region=candidate.generation.region,
                    variant_id=candidate.unit.region_id,
                    backend=candidate.generation.backend,
                    assessment=candidate.assessment,
                    build=result.attempt,
                )
            )
            _progress(
                context.progress,
                f"kept typed region variant {candidate.unit.region_id} as fallback",
            )
            break
    return _TypedRegionBuildOutcome(
        successful=tuple(successful),
        build=_aggregate_region_attempts(tuple(attempts), bool(successful)),
        artifacts=tuple(artifacts),
        skipped=tuple(skipped),
        cache_statuses=tuple(cache_statuses),
    )


def _compile_typed_variant(
    item: _PreparedTypedRegion,
    context: BackendCompileContext,
    *,
    cache_root: Path,
) -> BackendCompileResult:
    """Restore or invoke the adapter selected for one prepared backend variant.

    Args:
        item: Object being formatted for deterministic diagnostics.
        context: Prepared state shared by this operation.
        cache_root: Root directory for content-addressed cache entries.

    Returns:
        BackendCompileResult: Successful compiled variant or normalized build failure.
    """
    backend = _compiler_backend(item.generation.backend)
    return compile_with_region_cache(
        backend,
        item.unit,
        context,
        cache_root=cache_root,
    )


def _should_retry_with_fallback(
    item: _PreparedTypedRegion,
    result: BackendCompileResult,
) -> bool:
    """Return whether deterministic diagnostics permit the prepared fallback.

    Args:
        item: Object being formatted for deterministic diagnostics.
        result: Operation result being normalized or rendered.

    Returns:
        bool: Whether the failed variant qualifies for its next prepared fallback.
    """
    if result.attempt.success:
        return False
    diagnostic_prefix = (
        "MYPYC_TYPE_ERROR:" if item.generation.backend == "mypyc" else "CYTHON_COMPILE_ERROR:"
    )
    return result.attempt.stderr.startswith(diagnostic_prefix)


def _recovered_backend_attempt(
    attempt: CompileAttempt,
    fallback_variant_id: str,
) -> CompileAttempt:
    """Retain deterministic rejection evidence without failing the aggregate build.

    Args:
        attempt: Native compilation attempt being recovered or reported.
        fallback_variant_id: Variant ID used when no preferred backend succeeds.

    Returns:
        CompileAttempt: Rejection augmented with successful fallback evidence.
    """
    recovery = (
        f"mypyc rejected this variant; compiled {fallback_variant_id} with Cython"
        if attempt.stderr.startswith("MYPYC_TYPE_ERROR:")
        else (
            "whole-callable Cython rejected this variant; compiled "
            f"{fallback_variant_id} with outlined Cython"
        )
    )
    return replace(
        attempt,
        success=True,
        stdout="\n".join(
            part
            for part in (
                attempt.stdout,
                recovery,
                attempt.stderr,
            )
            if part
        ),
        stderr="",
        artifact_paths=(),
    )


def _tag_region_timings(attempt: CompileAttempt, region_id: str) -> CompileAttempt:
    return replace(
        attempt,
        phase_timings=tuple(
            replace(
                timing,
                detail=f"{region_id}; {timing.detail}" if timing.detail else region_id,
            )
            for timing in attempt.phase_timings
        ),
    )


def _aggregate_region_attempts(
    attempts: tuple[CompileAttempt, ...],
    success: bool,
) -> CompileAttempt:
    return CompileAttempt(
        success=success,
        command=("atoll", "typed-region-build"),
        stdout="\n".join(attempt.stdout for attempt in attempts if attempt.stdout),
        stderr="\n\n".join(attempt.stderr for attempt in attempts if not attempt.success),
        artifact_paths=tuple(
            dict.fromkeys(
                artifact
                for attempt in attempts
                if attempt.success
                for artifact in attempt.artifact_paths
            )
        ),
        duration_seconds=sum(attempt.duration_seconds for attempt in attempts),
        phase_timings=tuple(timing for attempt in attempts for timing in attempt.phase_timings),
        cache_status=_aggregate_cache_status(attempts),
    )


def _aggregate_cache_status(attempts: tuple[CompileAttempt, ...]) -> CompileCacheStatus:
    statuses = {attempt.cache_status for attempt in attempts if attempt.cache_status != "disabled"}
    if not statuses:
        return "disabled"
    if statuses == {"hit"}:
        return "hit"
    if statuses == {"miss"}:
        return "miss"
    return "partial"


def _failed_region_attempt(error: str) -> CompileAttempt:
    return CompileAttempt(
        success=False,
        command=(),
        stdout="",
        stderr=error,
        artifact_paths=(),
        duration_seconds=0.0,
    )


def _stage_compiled_superset(
    outcome: _TypedRegionBuildOutcome,
    prepared: tuple[_PreparedTypedRegion, ...],
    staged_source_roots: tuple[Path, ...],
    install_root: Path,
) -> str | None:
    """Overlay every successful candidate before allowlist-driven trials.

    Args:
        outcome: Successful native variants, artifacts, and build evidence.
        prepared: Generated units whose temporary source files must be removed.
        staged_source_roots: Copied source roots containing staged wheel inputs.
        install_root: Baseline payload receiving superset shims and artifacts.

    Returns:
        str | None: Normalized overlay failure text, or `None` on success.
    """
    shims = tuple(item.shim for item in outcome.successful)
    _insert_region_shims(shims)
    _place_region_artifacts(shims, outcome.build.artifact_paths, outcome.artifacts)
    for path in _prepared_source_paths(prepared):
        path.unlink(missing_ok=True)
    return _overlay_install_payload(
        staged_source_roots,
        install_root,
        tuple(config.source_path for config in shims),
    )


def _insert_region_shims(configs: tuple[RegionShimConfig, ...]) -> None:
    configs_by_path: dict[Path, list[RegionShimConfig]] = {}
    for config in configs:
        configs_by_path.setdefault(config.source_path, []).append(config)
    for source_path, module_configs in configs_by_path.items():
        source_text = source_path.read_text(encoding="utf-8")
        source_path.write_text(
            insert_or_replace_region_shim(source_text, tuple(module_configs)).new_text,
            encoding="utf-8",
        )


def _place_region_artifacts(
    configs: tuple[RegionShimConfig, ...],
    artifact_paths: tuple[Path, ...],
    artifact_records: tuple[ArtifactRecord, ...],
) -> None:
    source_by_digest = {_file_digest(path): path for path in artifact_paths}
    for config in configs:
        records = tuple(
            record
            for record in artifact_records
            if record.region_id == config.region_id
            or (
                record.region_id == "__shared__"
                and PurePosixPath(record.install_relative_path).parent.name
                == config.artifact_dir.name
            )
        )
        if not records:
            raise ValueError(f"compiled region has no artifact records: {config.region_id}")
        config.artifact_dir.mkdir(parents=True, exist_ok=True)
        for record in records:
            source = source_by_digest.get(record.digest)
            if source is None:
                raise ValueError(
                    "compiled region artifact digest is unavailable: "
                    f"{record.install_relative_path}"
                )
            destination = config.artifact_dir / PurePosixPath(record.install_relative_path).name
            _copy_if_different(source, destination)


def _source_clean_region_report_artifact_paths(
    root: Path,
    artifact_records: tuple[ArtifactRecord, ...],
) -> tuple[Path, ...]:
    """Map region-owned install paths to stable report paths under the target root.

    Args:
        root: Root directory of the target Python project.
        artifact_records: Validated native artifact metadata for successful variants.

    Returns:
        tuple[Path, ...]: Artifact paths exposed in source-clean reports.
    """
    return tuple(
        root / PurePosixPath(path)
        for path in dict.fromkeys(record.install_relative_path for record in artifact_records)
    )


def _artifact_records_for_prepared(
    prepared: tuple[_PreparedTypedRegion, ...],
    artifact_records: tuple[ArtifactRecord, ...],
) -> tuple[ArtifactRecord, ...]:
    """Return primary and colocated support artifacts for retained variants.

    Args:
        prepared: Compiled variants retained in the final staged payload.
        artifact_records: Artifact records produced by the superset native build.

    Returns:
        tuple[ArtifactRecord, ...]: Records reachable from the retained variant directories.
    """
    accepted_ids = frozenset(item.unit.region_id for item in prepared)
    accepted_directories = frozenset(item.shim.artifact_dir.name for item in prepared)
    return tuple(
        record
        for record in artifact_records
        if record.region_id in accepted_ids
        or (
            record.region_id == "__shared__"
            and PurePosixPath(record.install_relative_path).parent.name in accepted_directories
        )
    )


def _materialize_profitable_payload(
    *,
    baseline: _BaselineWheelPayload,
    staged_source_roots: tuple[Path, ...],
    install_root: Path,
    superset: tuple[_PreparedTypedRegion, ...],
    accepted: tuple[_PreparedTypedRegion, ...],
) -> str | None:
    """Rebuild the install payload from baseline with only profitable variants.

    Candidate trials run against a superset payload controlled by an internal
    allowlist. Before final verification, this function restores the immutable
    baseline wheel payload, rewrites staged shims to the accepted subset, removes
    rejected native directories, and overlays only the retained files.

    Args:
        baseline: Immutable interpreted payload prepared before profiling.
        staged_source_roots: Copied source roots containing generated shims and artifacts.
        install_root: Temporary payload rebuilt for final wheel promotion.
        superset: Every successfully compiled candidate present during trials.
        accepted: Marginally profitable variants retained for the final gate.

    Returns:
        str | None: Normalized materialization failure text, or `None` on success.
    """
    baseline_root = baseline.baseline_install_root
    if baseline_root is None:
        return "profile-guided payload cannot be rebuilt without a baseline payload"
    accepted_ids = frozenset(item.unit.region_id for item in accepted)
    try:
        _rewrite_region_shims(superset, accepted_ids)
        accepted_artifact_dirs = frozenset(item.shim.artifact_dir.resolve() for item in accepted)
        for item in superset:
            artifact_dir = item.shim.artifact_dir.resolve()
            if artifact_dir not in accepted_artifact_dirs:
                shutil.rmtree(artifact_dir, ignore_errors=True)
        _reset_dir(install_root)
        shutil.copytree(baseline_root, install_root, dirs_exist_ok=True)
        source_paths = tuple(dict.fromkeys(item.shim.source_path for item in accepted))
        _overlay_staged_sources(staged_source_roots, install_root, source_paths)
        _copy_atoll_artifacts(staged_source_roots, install_root)
    except (OSError, ValueError) as error:
        return f"profitable payload materialization failed: {error}"
    return None


def _rewrite_region_shims(
    superset: tuple[_PreparedTypedRegion, ...],
    accepted_ids: frozenset[str],
) -> None:
    """Replace superset staged shims with the accepted candidate subset.

    Args:
        superset: Every successfully compiled candidate present during trials.
        accepted_ids: Variant IDs retained by greedy selection.
    """
    by_path: dict[Path, list[RegionShimConfig]] = {}
    for item in superset:
        by_path.setdefault(item.shim.source_path, []).append(item.shim)
    for source_path, configs in by_path.items():
        source_text = source_path.read_text(encoding="utf-8")
        accepted = tuple(config for config in configs if config.region_id in accepted_ids)
        if accepted:
            updated = insert_or_replace_region_shim(source_text, accepted).new_text
        else:
            updated = remove_region_shim(
                source_text,
                source_module=configs[0].source_module,
                filename=source_path.name,
            ).new_text
        source_path.write_text(updated, encoding="utf-8")


def _typed_region_module_name(
    region: TypedRegion,
    backend: Backend,
    variant_id: str,
) -> str:
    module = re.sub(r"[^A-Za-z0-9_]", "_", region.source_module.name)
    variant_hash = hashlib.sha256(variant_id.encode()).hexdigest()[:8]
    return f"_atoll_region_{module}_{backend}_{region.source_hash[:12]}_{variant_hash}"


def _region_artifact_relative_dir(variant_id: str) -> str:
    """Return a stable collision-resistant install directory for one variant.

    Args:
        variant_id: Stable backend and specialization variant identifier.

    Returns:
        str: Install-relative directory that owns region artifacts.
    """
    readable = re.sub(r"[^A-Za-z0-9_.-]", "_", variant_id).strip("_.-")[:48]
    digest = hashlib.sha256(variant_id.encode()).hexdigest()[:12]
    label = readable or "region"
    return f".atoll/artifacts/{label}-{digest}"


def _compiler_backend(backend: Backend) -> CompilerBackend:
    """Return the configured compiler adapter for one automatic selection.

    Args:
        backend: Compiler backend selected for this operation.

    Returns:
        CompilerBackend: Compiler backend registered for the requested identifier.
    """
    return _COMPILER_BACKENDS[backend]


def _failed_result(
    root: Path,
    output_dir: Path | None,
    error: str,
    *,
    typed_regions: tuple[TypedRegion, ...] = (),
    execution_plans: tuple[ExecutionPlan | PlanRejection, ...] = (),
) -> PackageCommandResult:
    resolved_output_dir = _resolve_output_dir(root, output_dir)
    return PackageCommandResult(
        success=False,
        project_root=root,
        output_dir=resolved_output_dir,
        install_root=resolved_output_dir / "install",
        wheel_path=None,
        islands=(),
        build=CompileAttempt(
            success=False,
            command=(),
            stdout="",
            stderr=error,
            artifact_paths=(),
            duration_seconds=0.0,
        ),
        error=error,
        typed_regions=typed_regions,
        execution_plans=execution_plans,
    )


def _prepare_profile_guided_selection(
    options: PackageOptions,
    project: DiscoveredProject,
    scans: tuple[ModuleScan, ...],
) -> _ProfilePreparation:
    """Build, test, and profile the baseline before static region selection.

    Args:
        options: Validated command or generation options.
        project: Discovered target project configuration and modules.
        scans: Static scan facts used to include task-spawn callees in targeted observation.

    Returns:
        _ProfilePreparation: Prepared baseline/profile evidence or an early failure.
    """
    benchmark = project.config.compile.benchmark_command
    static_execution_plans = build_execution_plans(scans, None)
    if not options.run_quality_gates or benchmark is None:
        return _ProfilePreparation(execution_plans=static_execution_plans)
    output_dir = _resolve_output_dir(project.config.root, options.output_dir)
    build_root = output_dir / "build"
    install_root = output_dir / "install"
    _progress(options.progress, f"resetting temporary build roots in {output_dir}")
    _reset_dir(build_root)
    _reset_dir(install_root)
    baseline = _prepare_baseline_wheel_payload(
        project=project,
        build_root=build_root,
        install_root=install_root,
        progress=options.progress,
        run_quality_gates=options.run_quality_gates,
    )
    preparation = _ProfilePreparation(
        baseline=baseline,
        execution_plans=static_execution_plans,
    )
    if baseline.wheel_path is None:
        return _profile_preparation_failure(options, project, preparation, baseline.build.stderr)
    baseline = _run_baseline_semantic_test(
        project=project,
        baseline=baseline,
        progress=options.progress,
    )
    preparation = replace(preparation, baseline=baseline)
    semantic_test = baseline.semantic_test_result
    if semantic_test is None or not semantic_test.succeeded:
        error = (
            semantic_test.stderr
            if semantic_test is not None and semantic_test.stderr
            else "baseline semantic test command failed before profiling"
        )
        return _profile_preparation_failure(options, project, preparation, error)
    if baseline.baseline_install_root is None or baseline.quality_project_root is None:
        return _profile_preparation_failure(
            options,
            project,
            preparation,
            "baseline profiling payload is unavailable",
        )
    _progress(options.progress, "profiling baseline benchmark before region selection")
    profile = run_baseline_profile(
        benchmark,
        project_root=baseline.quality_project_root,
        payload_root=baseline.baseline_install_root,
        module_paths=_profile_module_paths(project),
        scratch_dir=build_root / "profile",
        observation_targets=_profile_observation_symbols(scans),
        spawn_targets=execution_plan_profile_targets(scans),
    )
    baseline = _append_profile_timings(baseline, profile)
    preparation = replace(preparation, baseline=baseline, profile=profile)
    _profile_progress(options.progress, profile)
    if profile.status == "invalid":
        return _profile_preparation_failure(options, project, preparation, profile.reason)
    return preparation


def _profile_preparation_failure(
    options: PackageOptions,
    project: DiscoveredProject,
    preparation: _ProfilePreparation,
    error: str,
) -> _ProfilePreparation:
    failure = _failed_before_region_selection(
        options=options,
        project=project,
        preparation=preparation,
        error=error,
    )
    return replace(preparation, failure=failure)


def _profile_progress(progress: PackageProgress | None, profile: ProfileResult) -> None:
    for run in profile.runs:
        _progress(
            progress,
            (
                f"profile {run.pass_kind} pass exited {run.returncode} "
                f"in {run.duration_seconds:.2f}s"
            ),
        )
    _progress(
        progress,
        (
            f"profile status {profile.status}: {profile.total_samples} sample(s), "
            f"{profile.mapped_project_samples} mapped to project code"
        ),
    )


def _region_selection_error(
    *,
    profile: ProfileResult | None,
    selection_members: tuple[SymbolId, ...],
    profile_members: tuple[SymbolId, ...],
    selected: tuple[_SelectedTypedRegion, ...],
    missing: tuple[SymbolId, ...],
) -> str | None:
    if profile is not None and profile.status == "profiled" and not selection_members:
        return "baseline profile found no credible hot project members in the compile scope"
    if missing:
        names = ", ".join(member.stable_id for member in missing)
        return f"requested member(s) are not backend-supported typed regions: {names}"
    if selected:
        return None
    if profile_members:
        return "profile-selected hot members are not supported by the configured compiler backends"
    return "scan found no backend-supported typed regions"


def _failed_region_selection(
    *,
    options: PackageOptions,
    project: DiscoveredProject,
    preparation: _ProfilePreparation,
    error: str,
    typed_regions: tuple[TypedRegion, ...],
) -> PackageCommandResult:
    _remove_failed_wheels(
        project,
        _resolve_output_dir(project.config.root, options.output_dir),
    )
    if preparation.baseline is not None:
        return _failed_before_region_selection(
            options=options,
            project=project,
            preparation=preparation,
            error=error,
            typed_regions=typed_regions,
        )
    return _failed_result(
        project.config.root,
        options.output_dir,
        error,
        typed_regions=typed_regions,
        execution_plans=preparation.execution_plans,
    )


def _failed_before_region_selection(
    *,
    options: PackageOptions,
    project: DiscoveredProject,
    preparation: _ProfilePreparation,
    error: str,
    typed_regions: tuple[TypedRegion, ...] = (),
) -> PackageCommandResult:
    """Return pre-compilation baseline, semantic, or profiling failure evidence.

    A successfully unpacked baseline is retained with its source-clean build
    tree because it contains the exact payload and profiling diagnostics that
    caused compilation to stop. A failed baseline build removes an empty install
    root while preserving backend diagnostics.

    Args:
        options: Validated command or generation options.
        project: Discovered target project configuration and modules.
        preparation: Baseline wheel and any early profile evidence.
        error: User-facing reason region compilation was not attempted.
        typed_regions: Backend-neutral regions scanned before a selection failure.

    Returns:
        PackageCommandResult: Failed result with no promoted wheel or native artifacts.

    Raises:
        ValueError: Baseline preparation evidence is absent despite this failure path.
    """
    baseline = preparation.baseline
    if baseline is None:
        raise ValueError("pre-selection failure requires baseline evidence")
    output_dir = _resolve_output_dir(project.config.root, options.output_dir)
    build_root = output_dir / "build"
    install_root = output_dir / "install"
    _remove_failed_wheels(project, output_dir)
    cleanup_removed = _remove_tree(install_root) if baseline.wheel_path is None else ()
    cleanup_kept = tuple(
        path for path in (build_root, install_root) if path.exists() and path not in cleanup_removed
    )
    semantic_test = baseline.semantic_test_result
    test_results = (semantic_test,) if semantic_test is not None else ()
    return PackageCommandResult(
        success=False,
        project_root=project.config.root,
        output_dir=output_dir,
        install_root=install_root,
        wheel_path=None,
        islands=(),
        build=replace(baseline.build, success=False, stderr=error),
        cleanup_removed=cleanup_removed,
        cleanup_kept=cleanup_kept,
        error=error,
        typed_regions=typed_regions,
        test_results=test_results,
        performance=BenchmarkGateResult(
            status="invalid",
            reason=error,
            minimum_speedup=project.config.compile.minimum_speedup,
            baseline_median_seconds=None,
            compiled_median_seconds=None,
            speedup=None,
            warmups=(),
            samples=(),
        ),
        profile=preparation.profile,
        execution_plans=preparation.execution_plans,
        fusion_plans=preparation.fusion_plans,
    )


def _append_profile_timings(
    baseline: _BaselineWheelPayload,
    profile: ProfileResult,
) -> _BaselineWheelPayload:
    """Attach unmeasured profiling duration to baseline phase evidence.

    Args:
        baseline: Built baseline wheel payload used by the profiling passes.
        profile: Captured baseline profiling evidence.

    Returns:
        _BaselineWheelPayload: Baseline payload with profiling phase timings appended.
    """
    timings = tuple(
        CompilePhaseTiming(
            name=f"profile_{run.pass_kind}",
            duration_seconds=run.duration_seconds,
            detail=f"exit {run.returncode}",
        )
        for run in profile.runs
    )
    if not timings:
        return baseline
    duration = sum(timing.duration_seconds for timing in timings)
    return replace(
        baseline,
        build=replace(
            baseline.build,
            duration_seconds=baseline.build.duration_seconds + duration,
            phase_timings=(*baseline.build.phase_timings, *timings),
        ),
    )


def _profile_module_paths(project: DiscoveredProject) -> tuple[tuple[str, str], ...]:
    """Map discovered modules to install-relative source suffixes for profiling.

    Args:
        project: Discovered target project configuration and modules.

    Returns:
        tuple[tuple[str, str], ...]: Module names and deterministic wheel payload paths.
    """
    return tuple(
        sorted(
            (
                module.name,
                _source_relative_path(module.path, project.config.source_roots).as_posix(),
            )
            for module in project.modules
        )
    )


def _profile_observation_symbols(scans: tuple[ModuleScan, ...]) -> tuple[SymbolId, ...]:
    """Convert scheduler planner targets into profiler symbol contracts.

    Args:
        scans: Static module scans inspected for recognized task-spawn callees.

    Returns:
        tuple[SymbolId, ...]: Deterministically ordered scheduler observation targets.

    Raises:
        ValueError: If the internal planner emits a malformed canonical identity.
    """
    symbols: list[SymbolId] = []
    targets = {
        *fusion_observation_targets(scans),
        *execution_plan_observation_targets(scans),
    }
    for target in sorted(targets):
        module, separator, qualname = target.partition("::")
        if not separator or not module or not qualname:
            raise ValueError(f"invalid scheduler observation target: {target}")
        symbols.append(SymbolId(module=module, qualname=qualname))
    return tuple(symbols)


def _remove_tree(path: Path) -> tuple[Path, ...]:
    if not path.exists():
        return ()
    shutil.rmtree(path)
    return (path,)


def _progress(progress: PackageProgress | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _progress_compile_selection(
    progress: PackageProgress | None,
    selected_regions: tuple[_SelectedTypedRegion, ...],
    *,
    requested_members: tuple[SymbolId, ...] = (),
) -> None:
    member_count = sum(len(region.members) for region in selected_regions)
    specialization_count = sum(region.specialization is not None for region in selected_regions)
    requested_text = (
        f" for {len(requested_members)} explicitly requested member(s)" if requested_members else ""
    )
    _progress(
        progress,
        (
            f"selected {len(selected_regions)} typed region backend variant(s), "
            f"{member_count} member binding(s), {specialization_count} specialization(s)"
            f"{requested_text}"
        ),
    )


def _duration(started: float) -> str:
    return f"{time.perf_counter() - started:.2f}s"


def _package_baseline(
    options: PackageOptions,
    project: DiscoveredProject,
    prepared: _BaselineWheelPayload | None,
) -> _BaselineWheelPayload:
    """Return an early profiled baseline or prepare the ordinary static baseline.

    Args:
        options: Validated command or generation options.
        project: Discovered target project configuration and modules.
        prepared: Baseline already built before profile-guided selection.

    Returns:
        _BaselineWheelPayload: Built and unpacked normal target wheel evidence.
    """
    if prepared is not None:
        return prepared
    output_dir = _resolve_output_dir(project.config.root, options.output_dir)
    build_root = output_dir / "build"
    install_root = output_dir / "install"
    _progress(options.progress, f"resetting temporary build roots in {output_dir}")
    _reset_dir(build_root)
    _reset_dir(install_root)
    return _prepare_baseline_wheel_payload(
        project=project,
        build_root=build_root,
        install_root=install_root,
        progress=options.progress,
        run_quality_gates=options.run_quality_gates,
    )


def _prepare_baseline_wheel_payload(
    *,
    project: DiscoveredProject,
    build_root: Path,
    install_root: Path,
    progress: PackageProgress | None,
    run_quality_gates: bool,
) -> _BaselineWheelPayload:
    """Build and unpack the target project's normal wheel from a clean copy.

    Args:
        project: Discovered target project configuration and modules.
        build_root: Root of the temporary source-clean build tree.
        install_root: Temporary install payload receiving compiled artifacts.
        progress: Optional progress callback for long-running work.
        run_quality_gates: Whether verification, tests, and benchmarks should run.

    Returns:
        _BaselineWheelPayload: Baseline wheel and unpacked payload evidence.
    """
    copied_project = build_root / "pep517-project"
    baseline_output = build_root / "pep517-dist"
    _progress(progress, "building target PEP 517 baseline wheel")
    copy_started = time.perf_counter()
    _copy_pep517_project(
        project.config.root,
        copied_project,
        excluded_output=build_root.parent,
    )
    copy_timing = CompilePhaseTiming(
        name="pep517_project_copy",
        duration_seconds=time.perf_counter() - copy_started,
        detail="source-clean project copy",
    )
    evidence = build_baseline_wheel(copied_project, baseline_output)
    build_timing = CompilePhaseTiming(
        name="pep517_wheel",
        duration_seconds=evidence.duration_seconds,
        detail=f"exit {evidence.returncode}",
    )
    if not evidence.succeeded or len(evidence.wheel_paths) != 1:
        error = _baseline_build_error(evidence)
        _progress(progress, f"PEP 517 baseline wheel failed in {evidence.duration_seconds:.2f}s")
        return _BaselineWheelPayload(
            wheel_path=None,
            build=CompileAttempt(
                success=False,
                command=evidence.command,
                stdout=evidence.stdout,
                stderr=error,
                artifact_paths=(),
                duration_seconds=copy_timing.duration_seconds + evidence.duration_seconds,
                phase_timings=(copy_timing, build_timing),
            ),
        )
    wheel_path = evidence.wheel_paths[0]
    unpack_started = time.perf_counter()
    try:
        unpack_wheel_payload(wheel_path, install_root)
    except (OSError, WheelOverlayError, zipfile.BadZipFile) as error:
        unpack_timing = CompilePhaseTiming(
            name="wheel_unpack",
            duration_seconds=time.perf_counter() - unpack_started,
            detail="failed",
        )
        return _BaselineWheelPayload(
            wheel_path=None,
            build=CompileAttempt(
                success=False,
                command=evidence.command,
                stdout=evidence.stdout,
                stderr=f"PEP 517 wheel unpack failed: {error}",
                artifact_paths=(),
                duration_seconds=(
                    copy_timing.duration_seconds
                    + evidence.duration_seconds
                    + unpack_timing.duration_seconds
                ),
                phase_timings=(copy_timing, build_timing, unpack_timing),
            ),
        )
    unpack_timing = CompilePhaseTiming(
        name="wheel_unpack",
        duration_seconds=time.perf_counter() - unpack_started,
        detail=wheel_path.name,
    )
    baseline_install_root: Path | None = None
    baseline_copy_timing: tuple[CompilePhaseTiming, ...] = ()
    if run_quality_gates and project.config.compile.benchmark_command is not None:
        baseline_started = time.perf_counter()
        baseline_install_root = build_root / "baseline-install"
        shutil.copytree(install_root, baseline_install_root)
        baseline_copy_timing = (
            CompilePhaseTiming(
                name="baseline_payload_copy",
                duration_seconds=time.perf_counter() - baseline_started,
                detail="benchmark baseline",
            ),
        )
    quality_project_root: Path | None = None
    if run_quality_gates and (
        project.config.compile.test_command is not None
        or project.config.compile.benchmark_command is not None
    ):
        _remove_quality_gate_sources(project, copied_project)
        quality_project_root = copied_project
    _progress(
        progress,
        f"built and unpacked PEP 517 baseline wheel in {evidence.duration_seconds:.2f}s",
    )
    return _BaselineWheelPayload(
        wheel_path=wheel_path,
        build=CompileAttempt(
            success=True,
            command=evidence.command,
            stdout=evidence.stdout,
            stderr=evidence.stderr,
            artifact_paths=(),
            duration_seconds=(
                copy_timing.duration_seconds
                + evidence.duration_seconds
                + unpack_timing.duration_seconds
                + sum(timing.duration_seconds for timing in baseline_copy_timing)
            ),
            phase_timings=(copy_timing, build_timing, unpack_timing, *baseline_copy_timing),
        ),
        baseline_install_root=baseline_install_root,
        quality_project_root=quality_project_root,
    )


def _baseline_build_error(evidence: WheelBuildEvidence) -> str:
    if not evidence.succeeded:
        return evidence.stderr or f"PEP 517 wheel build exited {evidence.returncode}"
    return f"PEP 517 build produced {len(evidence.wheel_paths)} wheel(s); expected exactly one"


def _combine_baseline_and_native(
    baseline: CompileAttempt,
    native: CompileAttempt,
) -> CompileAttempt:
    """Preserve normal-wheel and native-build evidence in one compatibility view.

    Args:
        baseline: Baseline wheel build evidence.
        native: Compiled payload command evidence.

    Returns:
        CompileAttempt: Final wheel path after overlaying native artifacts.
    """
    return CompileAttempt(
        success=baseline.success and native.success,
        command=("atoll", "source-clean-build"),
        stdout="\n".join(
            part
            for part in (
                baseline.stdout,
                baseline.stderr if baseline.success else "",
                native.stdout,
            )
            if part
        ),
        stderr="\n\n".join(
            part
            for part in (
                baseline.stderr if not baseline.success else "",
                native.stderr,
            )
            if part
        ),
        artifact_paths=native.artifact_paths,
        duration_seconds=baseline.duration_seconds + native.duration_seconds,
        phase_timings=(*baseline.phase_timings, *native.phase_timings),
        cache_status=native.cache_status,
    )


def _typed_verification_plan(
    configs: tuple[RegionShimConfig, ...],
    records: tuple[ArtifactRecord, ...],
) -> PackageVerificationPlan:
    regions_by_module: dict[str, list[str]] = {}
    for config in configs:
        regions_by_module.setdefault(config.source_module, []).append(config.region_id)
    artifacts = {
        record.install_relative_path: VerificationArtifact(
            path=record.install_relative_path,
            digest=record.digest,
        )
        for record in records
    }
    bindings = {
        (config.source_module, _verification_binding_qualname(binding)): VerificationBinding(
            module=config.source_module,
            qualname=_verification_binding_qualname(binding),
            kind=binding.kind,
            execution_kind=binding.execution_kind,
        )
        for config in configs
        for binding in config.bindings
        if binding.required
    }
    return PackageVerificationPlan(
        modules=tuple(sorted(regions_by_module)),
        regions=tuple(
            (module, tuple(region_ids)) for module, region_ids in sorted(regions_by_module.items())
        ),
        artifacts=tuple(artifacts[path] for path in sorted(artifacts)),
        bindings=tuple(bindings[key] for key in sorted(bindings)),
    )


def _verification_binding_qualname(binding: BindingTarget) -> str:
    """Return the public runtime path used by subprocess binding verification.

    Args:
        binding: Required source or specialized descriptor binding.

    Returns:
        str: Module-relative runtime path to the bound callable or class.
    """
    member_name = binding.source.qualname.rsplit(".", maxsplit=1)[-1]
    if binding.target_owner_class is not None:
        return f"{binding.target_owner_class}.{member_name}"
    return binding.source.qualname


def _verify_package_stage(
    *,
    stage: VerificationStage,
    target: Path,
    plan: PackageVerificationPlan,
    project_root: Path,
    progress: PackageProgress | None,
) -> PackageVerificationResult:
    result = verify_package_subprocess(
        stage=stage,
        target=target,
        plan=plan,
        project_root=project_root,
    )
    status = "passed" if result.success else f"failed with exit {result.exit_code}"
    _progress(progress, f"{stage} verification {status} in {result.duration_seconds:.2f}s")
    return result


def _append_verification_timing(
    attempt: CompileAttempt,
    result: PackageVerificationResult,
) -> CompileAttempt:
    return replace(
        attempt,
        duration_seconds=attempt.duration_seconds + result.duration_seconds,
        phase_timings=(
            *attempt.phase_timings,
            CompilePhaseTiming(
                name=f"{result.stage}_verification",
                duration_seconds=result.duration_seconds,
                detail="passed" if result.success else f"exit {result.exit_code}",
            ),
        ),
    )


def _append_phase_timing(
    attempt: CompileAttempt,
    *,
    name: str,
    duration_seconds: float,
    detail: str,
) -> CompileAttempt:
    return replace(
        attempt,
        duration_seconds=attempt.duration_seconds + duration_seconds,
        phase_timings=(
            *attempt.phase_timings,
            CompilePhaseTiming(
                name=name,
                duration_seconds=duration_seconds,
                detail=detail,
            ),
        ),
    )


def _append_phase_timings(
    attempt: CompileAttempt,
    timings: tuple[CompilePhaseTiming, ...],
) -> CompileAttempt:
    """Append a batch of measured phases to one compile attempt.

    Args:
        attempt: Existing native build and packaging evidence.
        timings: Additional semantic or benchmark phases to retain.

    Returns:
        CompileAttempt: Build evidence with cumulative duration and ordered phases.
    """
    return replace(
        attempt,
        duration_seconds=attempt.duration_seconds
        + sum(timing.duration_seconds for timing in timings),
        phase_timings=(*attempt.phase_timings, *timings),
    )


def _promote_source_clean_payload(
    context: _SourceCleanPromotionContext,
) -> _SourceCleanPromotionResult:
    payload_verification = _verify_package_stage(
        stage="payload",
        target=context.install_root,
        plan=context.verification_plan,
        project_root=context.project.config.root,
        progress=context.options.progress,
    )
    build = _append_verification_timing(context.build, payload_verification)
    if not payload_verification.success:
        return _failed_promotion(
            context,
            _SourceCleanPromotionFailure(
                build=build,
                verification_steps=(payload_verification,),
                error=payload_verification.stderr,
            ),
        )

    baseline_wheel_path = context.baseline.wheel_path
    if baseline_wheel_path is None:
        return _failed_promotion(
            context,
            _SourceCleanPromotionFailure(
                build=build,
                verification_steps=(payload_verification,),
                error="baseline wheel path is unavailable during final overlay",
            ),
        )

    wheel_started = time.perf_counter()
    candidate_output = context.build_root / "candidate-dist"
    _reset_dir(candidate_output)
    _progress(context.options.progress, "writing wheel candidate for verification")
    try:
        wheel_path = repack_overlaid_wheel(
            baseline_wheel_path=baseline_wheel_path,
            payload_dir=context.install_root,
            output_dir=candidate_output,
            platform_tag=_wheel_tag(),
        )
    except (OSError, WheelOverlayError, zipfile.BadZipFile) as error:
        return _failed_promotion(
            context,
            _SourceCleanPromotionFailure(
                build=build,
                verification_steps=(payload_verification,),
                error=f"final wheel overlay failed: {error}",
            ),
        )
    build = _append_phase_timing(
        build,
        name="wheel_repack",
        duration_seconds=time.perf_counter() - wheel_started,
        detail=wheel_path.name,
    )
    _progress(context.options.progress, f"wrote wheel in {_duration(wheel_started)}")

    wheel_verification = _verify_package_stage(
        stage="wheel",
        target=wheel_path,
        plan=context.verification_plan,
        project_root=context.project.config.root,
        progress=context.options.progress,
    )
    build = _append_verification_timing(build, wheel_verification)
    verification_steps = (payload_verification, wheel_verification)
    if not wheel_verification.success:
        return _failed_promotion(
            context,
            _SourceCleanPromotionFailure(
                build=build,
                verification_steps=verification_steps,
                error=wheel_verification.stderr,
                wheel_path=wheel_path,
            ),
        )

    quality_gate = (
        _run_configured_quality_gate(
            project=context.project,
            baseline=context.baseline,
            compiled_payload_root=context.install_root,
            progress=context.options.progress,
        )
        if context.options.run_quality_gates
        else _skipped_quality_gate(context.project.config.compile.minimum_speedup)
    )
    build = _append_quality_gate_timings(build, quality_gate)
    promotion_failure = _quality_gate_promotion_failure(
        context,
        build,
        verification_steps,
        wheel_path,
        quality_gate,
    )
    if promotion_failure is None:
        promotion_started = time.perf_counter()
        promoted_wheel = context.output_dir / wheel_path.name
        try:
            context.output_dir.mkdir(parents=True, exist_ok=True)
            wheel_path.replace(promoted_wheel)
        except OSError as error:
            promotion_failure = _SourceCleanPromotionFailure(
                build=build,
                verification_steps=verification_steps,
                error=f"verified wheel promotion failed: {error}",
                wheel_path=wheel_path,
                quality_gate=quality_gate,
            )
        else:
            build = _append_phase_timing(
                build,
                name="wheel_promote",
                duration_seconds=time.perf_counter() - promotion_started,
                detail=promoted_wheel.name,
            )
            wheel_verification = replace(wheel_verification, target=promoted_wheel)
            verification_steps = (payload_verification, wheel_verification)
            wheel_path = promoted_wheel
            _progress(context.options.progress, f"promoted verified wheel to {wheel_path}")
    if promotion_failure is not None:
        return _failed_promotion(
            context,
            promotion_failure,
        )

    cleanup_started = time.perf_counter()
    _progress(context.options.progress, "cleaning temporary build outputs")
    cleanup_removed = [context.build_root]
    shutil.rmtree(context.build_root)
    cleanup_kept: tuple[Path, ...] = ()
    if context.options.keep_install_tree:
        cleanup_kept = (context.install_root,)
    else:
        cleanup_removed.append(context.install_root)
        shutil.rmtree(context.install_root)
    _progress(
        context.options.progress,
        f"cleaned temporary outputs in {_duration(cleanup_started)}",
    )
    return _SourceCleanPromotionResult(
        success=True,
        wheel_path=wheel_path,
        build=build,
        verification_steps=verification_steps,
        test_results=quality_gate.tests,
        performance=quality_gate.performance,
        cleanup_removed=tuple(cleanup_removed),
        cleanup_kept=cleanup_kept,
    )


def _quality_gate_promotion_failure(
    context: _SourceCleanPromotionContext,
    build: CompileAttempt,
    verification_steps: tuple[PackageVerificationResult, ...],
    wheel_path: Path,
    quality_gate: _QualityGateOutcome,
) -> _SourceCleanPromotionFailure | None:
    """Normalize final-gate and empty-profile rejection before wheel publication.

    Args:
        context: Final source-clean promotion boundaries.
        build: Build evidence including final quality-gate timings.
        verification_steps: Successful payload and candidate-wheel verification.
        wheel_path: Verified candidate wheel still under temporary build storage.
        quality_gate: Final semantic and configured benchmark decision.

    Returns:
        _SourceCleanPromotionFailure | None: Promotion failure, or `None` when publication may run.
    """
    if not quality_gate.success:
        error = quality_gate.error
    elif context.requires_native_artifact and not context.verification_plan.artifacts:
        error = "no profile-guided candidate met the 1.01x marginal speedup threshold"
    else:
        return None
    return _SourceCleanPromotionFailure(
        build=build,
        verification_steps=verification_steps,
        error=error,
        wheel_path=wheel_path,
        quality_gate=quality_gate,
    )


def _failed_promotion(
    context: _SourceCleanPromotionContext,
    failure: _SourceCleanPromotionFailure,
) -> _SourceCleanPromotionResult:
    verification_steps = failure.verification_steps
    _remove_failed_wheels(context.project, context.output_dir)
    if failure.wheel_path is not None:
        retained_wheel = _retain_failed_wheel(context.build_root, failure.wheel_path)
        if retained_wheel is not None:
            verification_steps = tuple(
                replace(step, target=retained_wheel)
                if step.target.resolve() == failure.wheel_path.resolve()
                else step
                for step in verification_steps
            )
    return _SourceCleanPromotionResult(
        success=False,
        wheel_path=None,
        build=failure.build,
        verification_steps=verification_steps,
        test_results=failure.quality_gate.tests if failure.quality_gate is not None else (),
        performance=(
            failure.quality_gate.performance if failure.quality_gate is not None else None
        ),
        cleanup_removed=(),
        cleanup_kept=(context.build_root, context.install_root),
        error=failure.error,
    )


def _retain_failed_wheel(build_root: Path, wheel_path: Path) -> Path | None:
    """Move a rejected candidate under diagnostic scratch without exposing a wheel output.

    Args:
        build_root: Root of the temporary source-clean build tree.
        wheel_path: Wheel archive being retained, overlaid, or reported.

    Returns:
        Path | None: Retained diagnostic wheel path, when one can be preserved.
    """
    if not wheel_path.exists():
        return None
    diagnostic_root = build_root / "diagnostics"
    diagnostic_root.mkdir(parents=True, exist_ok=True)
    retained = diagnostic_root / wheel_path.name
    try:
        shutil.move(wheel_path, retained)
    except OSError:
        wheel_path.unlink(missing_ok=True)
        return None
    return retained


def _finalize_typed_payload(
    context: _TypedPayloadFinalizationContext,
) -> _TypedPayloadFinalizationResult:
    """Apply greedy selection and rebuild the accepted payload when configured.

    Args:
        context: Candidate superset, profile, baseline, and staging boundaries.

    Returns:
        _TypedPayloadFinalizationResult: Accepted variants and final overlay evidence.
    """
    profitability = _ProfitabilitySelectionOutcome(accepted=context.outcome.successful)
    overlay_error = context.overlay_error
    profile = context.profile
    build = context.outcome.build
    profitability_applied = profile is not None and _profile_profitability_enabled(
        context.options, context.project, profile
    )
    if overlay_error is None and profile is not None and profitability_applied:
        profitability = _select_profitable_candidates(
            _ProfitabilitySelectionContext(
                successful=context.outcome.successful,
                skipped=context.outcome.skipped,
                profile=profile,
                project=context.project,
                baseline=context.baseline,
                payload_root=context.install_root,
                progress=context.options.progress,
            )
        )
        build = _append_phase_timings(build, profitability.timings)
        overlay_error = _materialize_profitable_payload(
            baseline=context.baseline,
            staged_source_roots=context.staged_source_roots,
            install_root=context.install_root,
            superset=context.outcome.successful,
            accepted=profitability.accepted,
        )
    artifacts = _artifact_records_for_prepared(
        profitability.accepted,
        context.outcome.artifacts,
    )
    return _TypedPayloadFinalizationResult(
        successful=profitability.accepted,
        artifacts=artifacts,
        build=build,
        trials=profitability.trials,
        overlay_error=overlay_error,
        profitability_applied=profitability_applied,
    )


def _attach_conditional_task_fusion_research(
    promotion: _SourceCleanPromotionResult,
    context: _FusionResearchContext,
) -> tuple[_SourceCleanPromotionResult, _FusionResearchOutcome]:
    """Run conditional research and append its timings to promotion evidence.

    Args:
        promotion: Completed safe-payload promotion or rejection result.
        context: Inputs required by the conditional fusion workflow.

    Returns:
        tuple[_SourceCleanPromotionResult, _FusionResearchOutcome]: Promotion
            evidence with research timings plus the plan-bound trial results.
    """
    outcome = _run_conditional_task_fusion_research(context)
    return (
        replace(
            promotion,
            build=_append_phase_timings(promotion.build, outcome.timings),
        ),
        outcome,
    )


def _run_conditional_task_fusion_research(
    context: _FusionResearchContext,
) -> _FusionResearchOutcome:
    """Trial eligible eager-task plans only after the safe payload misses 1.10x.

    Every fused arm is a disposable copy of the final unfused payload. Passing
    evidence never changes the staged wheel or enables fusion by default; it
    only establishes whether a future explicit opt-in could be justified.

    Args:
        context: Safe benchmark result, eligible plans, payloads, and policy.

    Returns:
        _FusionResearchOutcome: Plan-bound trials and subprocess timings.

    Raises:
        ValueError: If validated compile policy and prepared baseline roots become inconsistent.
    """
    if context.performance is None or context.performance.status != "not-profitable":
        return _FusionResearchOutcome()
    eligible = tuple(plan for plan in context.plans if plan.eligible)
    if not eligible:
        return _FusionResearchOutcome()
    config = context.project.config.compile
    prerequisite_error = _fusion_research_prerequisite_error(context)
    if prerequisite_error is not None:
        return _FusionResearchOutcome(
            trials=tuple(unavailable_fusion_trial(plan.id, prerequisite_error) for plan in eligible)
        )
    if config.test_command is None or config.benchmark_command is None:
        raise ValueError("fusion research prerequisites were accepted without commands")
    if (
        context.baseline.quality_project_root is None
        or context.baseline.baseline_install_root is None
    ):
        raise ValueError("fusion research prerequisites were accepted without baseline roots")

    accepted_ids = frozenset(item.unit.region_id for item in context.accepted)
    scratch_root = context.build_root / "fusion-research"
    trials: list[FusionTrial] = []
    timings: list[CompilePhaseTiming] = []
    for index, plan in enumerate(eligible, start=1):
        fused_root = scratch_root / f"{index:02d}-{_wheel_safe_name(plan.id)}"
        stage_started = time.perf_counter()
        try:
            _reset_dir(fused_root)
            shutil.copytree(context.install_root, fused_root, dirs_exist_ok=True)
            staged_source = _task_fusion_source_path(context.project, fused_root, plan)
            generated = generate_eager_task_fusion(
                staged_source.read_text(encoding="utf-8"),
                plan,
            )
            staged_source.write_text(generated.new_text, encoding="utf-8")
            timings.append(
                CompilePhaseTiming(
                    name="fusion_stage",
                    duration_seconds=time.perf_counter() - stage_started,
                    detail=plan.id,
                )
            )
            _progress(
                context.options.progress,
                f"task-fusion trial {index}/{len(eligible)} testing {plan.id}",
            )
            trial = run_fusion_trial(
                FusionBenchmarkConfig(
                    plan_id=plan.id,
                    command=config.benchmark_command,
                    semantic_command=config.test_command,
                ),
                project_root=context.baseline.quality_project_root,
                baseline_payload_root=context.baseline.baseline_install_root,
                unfused_payload_root=context.install_root,
                fused_payload_root=fused_root,
                unfused_region_allowlist=accepted_ids,
                fused_region_allowlist=accepted_ids,
            )
        except (OSError, SyntaxError, ValueError) as error:
            trial = unavailable_fusion_trial(
                plan.id,
                f"task-fusion staged trial could not run: {error}",
            )
            if not any(
                timing.name == "fusion_stage" and timing.detail == plan.id for timing in timings
            ):
                timings.append(
                    CompilePhaseTiming(
                        name="fusion_stage",
                        duration_seconds=time.perf_counter() - stage_started,
                        detail=f"{plan.id}; failed",
                    )
                )
        finally:
            shutil.rmtree(fused_root, ignore_errors=True)
        trials.append(trial)
        timings.extend(_fusion_trial_timings(trial))
        _progress(
            context.options.progress,
            f"task-fusion trial {index}/{len(eligible)} {trial.status}: {plan.id}",
        )
    shutil.rmtree(scratch_root, ignore_errors=True)
    return _FusionResearchOutcome(trials=tuple(trials), timings=tuple(timings))


def _fusion_research_prerequisite_error(context: _FusionResearchContext) -> str | None:
    """Return the first missing research prerequisite in stable order.

    Args:
        context: Safe result, configured commands, and payload boundaries.

    Returns:
        str | None: Concrete unavailable reason, or `None` when trials can run.
    """
    config = context.project.config.compile
    checks = (
        (
            not context.options.run_quality_gates,
            "quality gates are delegated to the calling workflow",
        ),
        (config.test_command is None, "task-fusion research requires test_command"),
        (config.benchmark_command is None, "task-fusion research requires benchmark_command"),
        (
            context.baseline.quality_project_root is None,
            "task-fusion research quality-project root is unavailable",
        ),
        (
            context.baseline.baseline_install_root is None,
            "task-fusion research baseline payload is unavailable",
        ),
        (not context.install_root.is_dir(), "task-fusion research unfused payload is unavailable"),
    )
    return next((reason for failed, reason in checks if failed), None)


def _task_fusion_source_path(
    project: DiscoveredProject,
    payload_root: Path,
    plan: FusionPlan,
) -> Path:
    """Resolve one plan's installed module without consulting checkout imports.

    Args:
        project: Discovered module and source-root mapping.
        payload_root: Disposable fused payload copy.
        plan: Eligible plan identifying the caller module.

    Returns:
        Path: Existing installed Python source file to transform.

    Raises:
        ValueError: If the plan identity is malformed or its module/source is unavailable.
    """
    module_name, separator, _qualname = plan.caller.partition("::")
    if separator == "":
        raise ValueError(f"task-fusion caller has no module identity: {plan.caller}")
    module = next((item for item in project.modules if item.name == module_name), None)
    if module is None:
        raise ValueError(f"task-fusion module is not part of the project: {module_name}")
    relative_path = _source_relative_path(module.path, project.config.source_roots)
    source_path = payload_root / relative_path
    if not source_path.is_file():
        raise ValueError(f"task-fusion installed source is unavailable: {relative_path}")
    return source_path


def _fusion_trial_timings(trial: FusionTrial) -> tuple[CompilePhaseTiming, ...]:
    """Convert three-arm command evidence into compile-report phase timings.

    Args:
        trial: Completed, rejected, invalid, or unavailable plan-bound trial.

    Returns:
        tuple[CompilePhaseTiming, ...]: Ordered semantic, warmup, and sample timings.
    """
    timings = tuple(
        CompilePhaseTiming(
            name="fusion_semantic_test",
            duration_seconds=evidence.run.duration_seconds,
            detail=f"{trial.plan_id}; {evidence.arm}; exit {evidence.run.returncode}",
        )
        for evidence in trial.semantic_runs
    )
    timings += tuple(
        CompilePhaseTiming(
            name="fusion_benchmark_warmup",
            duration_seconds=evidence.run.duration_seconds,
            detail=f"{trial.plan_id}; {evidence.arm}",
        )
        for evidence in trial.warmups
    )
    return timings + tuple(
        CompilePhaseTiming(
            name="fusion_benchmark",
            duration_seconds=evidence.run.duration_seconds,
            detail=f"{trial.plan_id}; {evidence.arm}; {trial.status}",
        )
        for evidence in trial.samples
    )


def _profile_profitability_enabled(
    options: PackageOptions,
    project: DiscoveredProject,
    profile: ProfileResult | None,
) -> bool:
    """Return whether this invocation has an ordered dynamic candidate set.

    Explicit member requests and unsupported profiling launchers retain static
    all-at-once behavior. Their configured full benchmark still gates wheel
    promotion, but Atoll does not invent a greedy order without profile evidence.

    Args:
        options: Validated command or generation options.
        project: Discovered target project configuration and modules.
        profile: Dynamic baseline profile, when configured and supported.

    Returns:
        bool: Whether marginal candidate selection should run.
    """
    return (
        options.run_quality_gates
        and not options.selected_members
        and project.config.compile.benchmark_command is not None
        and profile is not None
        and profile.status == "profiled"
        and bool(profile.selected_symbols)
    )


def _select_profitable_candidates(
    context: _ProfitabilitySelectionContext,
) -> _ProfitabilitySelectionOutcome:
    """Greedily retain profile-ordered variants with measurable marginal value.

    Each trial runs the semantic command once with the candidate combination,
    then compares the currently accepted allowlist with that combination using
    one warmup and three alternating benchmark pairs. Candidate decisions never
    promote a wheel; the configured full benchmark remains the only performance
    promotion gate.

    Args:
        context: Compiled candidates, profile, baseline, and subprocess boundaries.

    Returns:
        _ProfitabilitySelectionOutcome: Accepted variants, decisions, and command timings.
    """
    candidates = _profiled_profitability_candidates(
        context.successful,
        context.skipped,
        context.profile,
    )
    test_command = context.project.config.compile.test_command
    benchmark_command = context.project.config.compile.benchmark_command
    quality_root = context.baseline.quality_project_root
    if test_command is None or benchmark_command is None or quality_root is None:
        reason = "candidate selection prerequisites are unavailable"
        return _unavailable_candidate_selection(candidates, reason)

    accepted: list[_PreparedTypedRegion] = []
    accepted_symbols: set[str] = set()
    profile_samples = {
        member.symbol.stable_id: member.samples for member in context.profile.members
    }
    trials: list[CandidateTrial] = []
    timings: list[CompilePhaseTiming] = []
    candidate_count = len(candidates)
    for index, candidate in enumerate(candidates, start=1):
        prepared = candidate.prepared
        variant_id = prepared.unit.region_id
        baseline_ids = tuple(item.unit.region_id for item in accepted)
        trial_ids = (*baseline_ids, variant_id)
        _progress(
            context.progress,
            f"candidate {index}/{candidate_count} testing {variant_id} semantics",
        )
        semantic = run_performance_command(
            test_command,
            project_root=quality_root,
            payload_root=context.payload_root,
            mode="compiled",
            region_allowlist=frozenset(trial_ids),
        )
        timings.append(
            CompilePhaseTiming(
                name="candidate_semantic_test",
                duration_seconds=semantic.duration_seconds,
                detail=f"{variant_id}; exit {semantic.returncode}",
            )
        )
        if not semantic.succeeded:
            status: CandidateTrialStatus = "failed-semantics"
            reason = _command_failure_summary(semantic, "candidate semantic test failed")
            benchmark_status = "not-run"
            marginal_speedup = None
        else:
            _progress(
                context.progress,
                (
                    f"candidate {index}/{candidate_count} benchmarking {variant_id} "
                    f"against {len(baseline_ids)} accepted variant(s)"
                ),
            )

            benchmark = run_benchmark_gate(
                BenchmarkGateConfig(
                    command=benchmark_command,
                    warmups=_CANDIDATE_BENCHMARK_WARMUPS,
                    samples=_CANDIDATE_BENCHMARK_SAMPLES,
                    minimum_speedup=_CANDIDATE_MINIMUM_SPEEDUP,
                ),
                project_root=quality_root,
                baseline_payload_root=context.payload_root,
                compiled_payload_root=context.payload_root,
                baseline_region_allowlist=frozenset(baseline_ids),
                compiled_region_allowlist=frozenset(trial_ids),
                progress=partial(_candidate_benchmark_progress, context.progress, variant_id),
            )
            timings.extend(_candidate_benchmark_timings(variant_id, benchmark))
            benchmark_status = benchmark.status
            marginal_speedup = benchmark.speedup
            reason = benchmark.reason
            if benchmark.status == "passed":
                status = "accepted"
                accepted.append(prepared)
                accepted_symbols.update(candidate.symbols)
            elif benchmark.status == "not-profitable":
                status = "rejected"
            else:
                status = "unavailable"
        accepted_coverage = _sample_coverage(
            sum(profile_samples.get(symbol, 0) for symbol in accepted_symbols),
            context.profile.mapped_project_samples,
        )
        trials.append(
            CandidateTrial(
                id=f"{index:02d}:{variant_id}",
                source_region_id=prepared.generation.region.id,
                variant_id=variant_id,
                backend=prepared.generation.backend,
                lowering_mode=prepared.lowering_mode,
                symbols=candidate.symbols,
                status=status,
                reason=reason,
                marginal_speedup=marginal_speedup,
                fallback_reason=candidate.fallback_reason,
                profile_samples=candidate.profile_samples,
                profile_coverage=candidate.profile_coverage,
                accepted_hot_coverage=accepted_coverage,
                baseline_variants=baseline_ids,
                trial_variants=trial_ids,
                semantic_test_exit_code=semantic.returncode,
                semantic_test_duration_seconds=semantic.duration_seconds,
                benchmark_status=benchmark_status,
            )
        )
        _progress(
            context.progress,
            (
                f"candidate {index}/{candidate_count} {status}: {variant_id}; "
                f"accepted hot coverage {accepted_coverage:.1%}"
            ),
        )
    return _ProfitabilitySelectionOutcome(
        accepted=tuple(accepted),
        trials=tuple(trials),
        timings=tuple(timings),
    )


def _profiled_profitability_candidates(
    successful: tuple[_PreparedTypedRegion, ...],
    skipped: tuple[PackageRegionBuildFailure, ...],
    profile: ProfileResult,
) -> tuple[_ProfitabilityCandidate, ...]:
    """Order successful variants by their first profile-selected binding.

    Args:
        successful: Successfully compiled variants available for trials.
        skipped: Backend failures retained as fallback explanations.
        profile: Dynamic profile with a descending-hotness selected-symbol order.

    Returns:
        tuple[_ProfitabilityCandidate, ...]: Deduplicated candidates in profile order.
    """
    profile_samples = {member.symbol: member.samples for member in profile.members}
    selected = frozenset(profile.selected_symbols)
    by_symbol: dict[SymbolId, list[_PreparedTypedRegion]] = {}
    for prepared in successful:
        for binding in prepared.generation.bindings:
            if binding.source in selected:
                by_symbol.setdefault(binding.source, []).append(prepared)
    ordered: list[_ProfitabilityCandidate] = []
    seen: set[str] = set()
    for symbol in profile.selected_symbols:
        for prepared in by_symbol.get(symbol, ()):
            variant_id = prepared.unit.region_id
            if variant_id in seen:
                continue
            seen.add(variant_id)
            represented = tuple(
                dict.fromkeys(
                    binding.source
                    for binding in prepared.generation.bindings
                    if binding.source in selected
                )
            )
            samples = sum(profile_samples.get(member, 0) for member in represented)
            ordered.append(
                _ProfitabilityCandidate(
                    prepared=prepared,
                    symbols=tuple(member.stable_id for member in represented),
                    profile_samples=samples,
                    profile_coverage=_sample_coverage(samples, profile.mapped_project_samples),
                    fallback_reason=_candidate_fallback_reason(prepared, skipped),
                )
            )
    return tuple(ordered)


def _candidate_fallback_reason(
    prepared: _PreparedTypedRegion,
    skipped: tuple[PackageRegionBuildFailure, ...],
) -> str | None:
    """Return the deterministic preferred-backend failure for a fallback variant.

    Args:
        prepared: Successful variant selected after backend retries.
        skipped: Failed preferred variants retained by the package build.

    Returns:
        str | None: First diagnostic line explaining fallback selection.
    """
    if prepared.fallback_reason is not None:
        return prepared.fallback_reason
    preferred_id = prepared.conditional_on_failure_of
    if preferred_id is None:
        return None
    failure = next((item for item in skipped if item.variant_id == preferred_id), None)
    if failure is None:
        return f"preferred variant {preferred_id} was unavailable"
    return (
        _first_diagnostic_line(failure.build.stderr) or f"preferred variant {preferred_id} failed"
    )


def _fallback_attempt_reason(
    rejected: tuple[tuple[_PreparedTypedRegion, CompileAttempt], ...],
) -> str | None:
    """Describe the ordered backend chain preceding a successful fallback.

    Args:
        rejected: Prepared variants and deterministic failed attempts in retry order.

    Returns:
        str | None: Concise ordered fallback provenance, when retries occurred.
    """
    if not rejected:
        return None
    return "; ".join(
        (
            f"{prepared.generation.backend} {prepared.lowering_mode}: "
            f"{_first_diagnostic_line(attempt.stderr) or 'compiler rejected variant'}"
        )
        for prepared, attempt in rejected
    )


def _unavailable_candidate_selection(
    candidates: tuple[_ProfitabilityCandidate, ...],
    reason: str,
) -> _ProfitabilitySelectionOutcome:
    """Represent a failed internal selection setup without accepting candidates.

    Args:
        candidates: Ordered compiled variants that could not be measured.
        reason: Concrete missing prerequisite.

    Returns:
        _ProfitabilitySelectionOutcome: Unavailable decisions and an empty accepted set.
    """
    trials = tuple(
        CandidateTrial(
            id=f"{index:02d}:{candidate.prepared.unit.region_id}",
            source_region_id=candidate.prepared.generation.region.id,
            variant_id=candidate.prepared.unit.region_id,
            backend=candidate.prepared.generation.backend,
            lowering_mode=candidate.prepared.lowering_mode,
            symbols=candidate.symbols,
            status="unavailable",
            reason=reason,
            marginal_speedup=None,
            fallback_reason=candidate.fallback_reason,
            profile_samples=candidate.profile_samples,
            profile_coverage=candidate.profile_coverage,
            accepted_hot_coverage=0.0,
            baseline_variants=(),
            trial_variants=(candidate.prepared.unit.region_id,),
            semantic_test_exit_code=None,
            semantic_test_duration_seconds=None,
            benchmark_status="not-run",
        )
        for index, candidate in enumerate(candidates, start=1)
    )
    return _ProfitabilitySelectionOutcome(accepted=(), trials=trials)


def _candidate_benchmark_progress(
    progress: PackageProgress | None,
    variant_id: str,
    event: BenchmarkProgress,
) -> None:
    """Render one marginal benchmark pair event with candidate context.

    Args:
        progress: Optional package progress callback.
        variant_id: Candidate variant measured by this event.
        event: Runtime benchmark phase notification.
    """
    _progress(
        progress,
        (
            f"candidate benchmark {variant_id} {event.phase} pair {event.pair_index} "
            f"{event.mode} completed in {event.duration_seconds:.2f}s"
        ),
    )


def _candidate_benchmark_timings(
    variant_id: str,
    result: BenchmarkGateResult,
) -> tuple[CompilePhaseTiming, ...]:
    """Convert one marginal benchmark into ordered compile phase evidence.

    Args:
        variant_id: Candidate variant measured by the benchmark.
        result: Marginal benchmark decision and child-process timings.

    Returns:
        tuple[CompilePhaseTiming, ...]: Warmup and sample timings for the compile report.
    """
    return tuple(
        CompilePhaseTiming(
            name="candidate_benchmark",
            duration_seconds=run.duration_seconds,
            detail=f"{variant_id}; {phase}; {run.mode}; {result.status}",
        )
        for phase, runs in (("warmup", result.warmups), ("sample", result.samples))
        for run in runs
    )


def _command_failure_summary(result: CommandRunEvidence, fallback: str) -> str:
    """Return a concise subprocess failure without discarding its exit status.

    Args:
        result: Failed candidate command evidence.
        fallback: Description used when stderr is empty.

    Returns:
        str: First stderr line or an exit-code fallback.
    """
    return _first_diagnostic_line(result.stderr) or f"{fallback} with exit {result.returncode}"


def _first_diagnostic_line(text: str) -> str | None:
    """Return the first non-empty diagnostic line.

    Args:
        text: Compiler or subprocess diagnostic text.

    Returns:
        str | None: First non-empty stripped line, when present.
    """
    return next((line.strip() for line in text.splitlines() if line.strip()), None)


def _sample_coverage(samples: int, total_samples: int) -> float:
    """Return bounded mapped-project coverage for candidate evidence.

    Args:
        samples: Candidate or accepted-set sample count.
        total_samples: Total samples mapped to project members.

    Returns:
        float: Coverage in the inclusive range zero through one.
    """
    if total_samples <= 0:
        return 0.0
    return min(samples / total_samples, 1.0)


def _run_configured_quality_gate(
    *,
    project: DiscoveredProject,
    baseline: _BaselineWheelPayload,
    compiled_payload_root: Path,
    progress: PackageProgress | None,
) -> _QualityGateOutcome:
    config = project.config.compile
    commands_configured = config.test_command is not None or config.benchmark_command is not None
    if commands_configured and baseline.quality_project_root is None:
        return _invalid_quality_gate(config.minimum_speedup, "quality-gate project is missing")
    command_root = baseline.quality_project_root or project.config.root
    tests: list[CommandRunEvidence] = []
    if baseline.semantic_test_result is not None:
        tests.append(baseline.semantic_test_result)
    if config.test_command is not None:
        pending_tests: list[CommandRunEvidence] = []
        if config.benchmark_command is not None and baseline.semantic_test_result is None:
            if baseline.baseline_install_root is None:
                return _invalid_quality_gate(config.minimum_speedup, "baseline payload is missing")
            pending_tests.append(
                run_performance_command(
                    config.test_command,
                    project_root=command_root,
                    payload_root=baseline.baseline_install_root,
                    mode="baseline",
                )
            )
        pending_tests.append(
            run_performance_command(
                config.test_command,
                project_root=command_root,
                payload_root=compiled_payload_root,
                mode="compiled",
            )
        )
        tests.extend(pending_tests)
        for result in pending_tests:
            status = "passed" if result.succeeded else f"failed with exit {result.returncode}"
            _progress(
                progress,
                f"{result.mode} semantic tests {status} in {result.duration_seconds:.2f}s",
            )
        failure = next((result for result in tests if not result.succeeded), None)
        if failure is not None:
            return _QualityGateOutcome(
                success=False,
                tests=tuple(tests),
                performance=BenchmarkGateResult(
                    status="invalid",
                    reason=f"{failure.mode} semantic test command failed",
                    minimum_speedup=config.minimum_speedup,
                    baseline_median_seconds=None,
                    compiled_median_seconds=None,
                    speedup=None,
                    warmups=(),
                    samples=(),
                ),
                error=(
                    failure.stderr
                    or f"{failure.mode} semantic test command exited {failure.returncode}"
                ),
            )
    if config.benchmark_command is not None and baseline.baseline_install_root is None:
        return _invalid_quality_gate(config.minimum_speedup, "baseline payload is missing")
    benchmark = run_benchmark_gate(
        BenchmarkGateConfig(
            command=config.benchmark_command,
            warmups=config.benchmark_warmups,
            samples=config.benchmark_samples,
            minimum_speedup=config.minimum_speedup,
        ),
        project_root=command_root,
        baseline_payload_root=baseline.baseline_install_root or compiled_payload_root,
        compiled_payload_root=compiled_payload_root,
        progress=lambda event: _benchmark_progress(progress, event),
    )
    accepted = benchmark.status in {"passed", "unbenchmarked"}
    _progress(progress, f"performance status {benchmark.status}: {benchmark.reason}")
    return _QualityGateOutcome(
        success=accepted,
        tests=tuple(tests),
        performance=benchmark,
        error=None if accepted else benchmark.reason,
    )


def _run_baseline_semantic_test(
    *,
    project: DiscoveredProject,
    baseline: _BaselineWheelPayload,
    progress: PackageProgress | None,
) -> _BaselineWheelPayload:
    """Run the benchmark baseline's semantic test exactly once before profiling.

    The early result is retained on the baseline payload and reused by the final
    quality gate. This prevents an invalid baseline from consuming compiler time
    and avoids running the same interpreted test twice.

    Args:
        project: Discovered target project configuration and modules.
        baseline: Built and unpacked baseline wheel payload.
        progress: Optional progress callback for long-running work.

    Returns:
        _BaselineWheelPayload: Baseline evidence with an optional semantic test result.
    """
    config = project.config.compile
    if config.benchmark_command is None or config.test_command is None:
        return baseline
    if baseline.baseline_install_root is None or baseline.quality_project_root is None:
        return baseline
    result = run_performance_command(
        config.test_command,
        project_root=baseline.quality_project_root,
        payload_root=baseline.baseline_install_root,
        mode="baseline",
    )
    status = "passed" if result.succeeded else f"failed with exit {result.returncode}"
    _progress(progress, f"baseline semantic tests {status} in {result.duration_seconds:.2f}s")
    return replace(baseline, semantic_test_result=result)


def _invalid_quality_gate(minimum_speedup: float, reason: str) -> _QualityGateOutcome:
    return _QualityGateOutcome(
        success=False,
        tests=(),
        performance=BenchmarkGateResult(
            status="invalid",
            reason=reason,
            minimum_speedup=minimum_speedup,
            baseline_median_seconds=None,
            compiled_median_seconds=None,
            speedup=None,
            warmups=(),
            samples=(),
        ),
        error=reason,
    )


def _skipped_quality_gate(minimum_speedup: float) -> _QualityGateOutcome:
    """Record that a caller owns semantic and performance gates for this build.

    Args:
        minimum_speedup: Configured profitability threshold for benchmark results.

    Returns:
        _QualityGateOutcome: Explicit skipped quality-gate result with its reason.
    """
    return _QualityGateOutcome(
        success=True,
        tests=(),
        performance=BenchmarkGateResult(
            status="unbenchmarked",
            reason="quality gates delegated to the calling workflow",
            minimum_speedup=minimum_speedup,
            baseline_median_seconds=None,
            compiled_median_seconds=None,
            speedup=None,
            warmups=(),
            samples=(),
        ),
        error=None,
    )


def _benchmark_progress(progress: PackageProgress | None, event: BenchmarkProgress) -> None:
    _progress(
        progress,
        (
            f"benchmark {event.phase} pair {event.pair_index} {event.mode} "
            f"completed in {event.duration_seconds:.2f}s"
        ),
    )


def _append_quality_gate_timings(
    attempt: CompileAttempt,
    outcome: _QualityGateOutcome,
) -> CompileAttempt:
    timings = tuple(
        CompilePhaseTiming(
            name="semantic_test",
            duration_seconds=result.duration_seconds,
            detail=f"{result.mode}; exit {result.returncode}",
        )
        for result in outcome.tests
    )
    benchmark_runs = (*outcome.performance.warmups, *outcome.performance.samples)
    timings += tuple(
        CompilePhaseTiming(
            name="benchmark",
            duration_seconds=result.duration_seconds,
            detail=f"{result.mode}; {outcome.performance.status}",
        )
        for result in benchmark_runs
    )
    return replace(
        attempt,
        duration_seconds=attempt.duration_seconds
        + sum(timing.duration_seconds for timing in timings),
        phase_timings=(*attempt.phase_timings, *timings),
    )


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_scans(
    project: DiscoveredProject,
    module_name: str | None,
    selected_members: tuple[SymbolId, ...] = (),
) -> tuple[ModuleScan, ...]:
    selected_module_names = tuple(dict.fromkeys(member.module for member in selected_members))
    if module_name is not None and any(name != module_name for name in selected_module_names):
        raise ValueError("selected members must belong to the requested module scope")
    if selected_module_names:
        modules = tuple(_find_module(project.modules, name) for name in selected_module_names)
    elif module_name:
        modules = (_find_module(project.modules, module_name),)
    else:
        modules = project.modules
    return tuple(enrich_island_analysis(scan_module(module)) for module in modules)


def _profile_candidate_members(
    scans: tuple[ModuleScan, ...],
    backends: tuple[Backend, ...],
) -> tuple[SymbolId, ...]:
    """Return callable roots a dynamic profile may select for some backend.

    This preflight does not select or compile a member. It prevents Atoll from
    rejecting a benchmark-configured project merely because all credible hot
    members require boxed Cython lowering.

    Args:
        scans: Selected module scans in deterministic order.
        backends: Backends considered in configured preference order.

    Returns:
        tuple[SymbolId, ...]: Potential profile roots in scan/source order.
    """
    candidates: list[SymbolId] = []
    for scan in scans:
        for region in scan.typed_regions:
            decisions = {decision.target: decision for decision in region.decisions}
            all_members = frozenset(member.id for member in region.members)
            eligible = _eligible_typed_callables(region, decisions, hot=all_members)
            for root in eligible:
                sliced = build_directed_region_slice(region, root)
                if any(
                    root in _compiler_backend(backend).assess(sliced).supported_members
                    for backend in backends
                ):
                    candidates.append(root)
    return tuple(candidates)


def _selected_typed_regions(
    scans: tuple[ModuleScan, ...],
    backends: tuple[Backend, ...] = ("mypyc", "cython"),
    requested_members: tuple[SymbolId, ...] = (),
    *,
    hot_members: tuple[SymbolId, ...] = (),
) -> tuple[_SelectedTypedRegion, ...]:
    """Select backend variants for typed callables and safe atomic classes.

    Explicit requests are expanded only along same-region runtime call edges.
    This keeps trial selection precise while ensuring copied callables retain
    every eligible helper required by their executable bodies.

    Args:
        scans: Selected module scans in deterministic order.
        backends: Backends considered in configured preference order.
        requested_members: Explicit source members requested by the caller.
        hot_members: Profile-selected members allowed to use boxed Cython lowering.

    Returns:
        tuple[_SelectedTypedRegion, ...]: Backend-supported region selections in deterministic
            order.
    """
    selected: list[_SelectedTypedRegion] = []
    requested = frozenset(requested_members)
    hot = frozenset(hot_members)
    for scan in scans:
        for region in scan.typed_regions:
            selected.extend(
                _selected_region_variants(
                    scan,
                    region,
                    backends,
                    requested,
                    hot,
                )
            )
    return tuple(selected)


def _selected_region_variants(
    scan: ModuleScan,
    region: TypedRegion,
    backends: tuple[Backend, ...],
    requested: frozenset[SymbolId],
    hot: frozenset[SymbolId],
) -> tuple[_SelectedTypedRegion, ...]:
    region_member_ids = frozenset(member.id for member in region.members)
    if requested and not requested.intersection(region_member_ids):
        return ()
    decisions = {decision.target: decision for decision in region.decisions}
    mypyc_assessment = MYPYC_BACKEND.assess(region) if "mypyc" in backends else None
    cython_assessment = CYTHON_BACKEND.assess(region) if "cython" in backends else None
    if requested:
        return _selected_requested_region_slices(
            scan=scan,
            source_region=region,
            requested=requested,
            hot=hot,
            backends=backends,
        )
    eligible = _eligible_typed_callables(region, decisions, hot=frozenset())
    variants: list[_SelectedTypedRegion] = []
    atomic_variant_id = _append_atomic_class_variant(
        variants,
        scan,
        region,
        cython_assessment,
        requested,
    )
    mypyc_members, mypyc_bound = _backend_callable_members(
        region,
        eligible,
        mypyc_assessment,
        excluded=(),
    )
    if mypyc_assessment is not None and mypyc_members:
        variants.append(
            _SelectedTypedRegion(
                scan=scan,
                region=region,
                variant_id=f"{region.id}@mypyc",
                backend="mypyc",
                assessment=mypyc_assessment,
                members=mypyc_members,
                bound_members=mypyc_bound,
                conditional_on_failure_of=atomic_variant_id,
            )
        )
    cython_members, cython_bound = _backend_callable_members(
        region,
        eligible,
        cython_assessment,
        excluded=mypyc_bound,
    )
    if cython_assessment is not None and cython_members:
        variants.append(
            _SelectedTypedRegion(
                scan=scan,
                region=region,
                variant_id=f"{region.id}@cython",
                backend="cython",
                assessment=cython_assessment,
                members=cython_members,
                bound_members=cython_bound,
                conditional_on_failure_of=atomic_variant_id,
            )
        )
    for specialization in region.specializations:
        if requested and specialization.source_member not in requested:
            continue
        variant = _selected_specialization_variant(scan, region, specialization, backends)
        if variant is not None:
            variants.append(variant)
    return tuple(variants)


def _selected_requested_region_slices(
    *,
    scan: ModuleScan,
    source_region: TypedRegion,
    requested: frozenset[SymbolId],
    hot: frozenset[SymbolId],
    backends: tuple[Backend, ...],
) -> tuple[_SelectedTypedRegion, ...]:
    """Select one deterministic directed backend slice per requested binding.

    Args:
        scan: Module scan containing retained source facts.
        source_region: Connected scan region that owns requested roots.
        requested: Explicit or profile-derived public binding roots.
        hot: Profile-selected roots allowed to use boxed Cython semantics.
        backends: Backends considered in configured preference order.

    Returns:
        tuple[_SelectedTypedRegion, ...]: Independent root slices in source order.

    Raises:
        ValueError: If a required same-unit dependency is absent from the source region.
    """
    variants: list[_SelectedTypedRegion] = []
    roots = tuple(member.id for member in source_region.members if member.id in requested)
    for root in roots:
        sliced = build_directed_region_slice(source_region, root)
        decisions = {decision.target: decision for decision in sliced.decisions}
        eligible = _eligible_typed_callables(sliced, decisions, hot=hot)
        closure = _runtime_member_closure(sliced, eligible, frozenset({root}))
        callable_variants = _selected_requested_callable_variant(
            _RequestedCallableVariant(
                scan=scan,
                region=sliced,
                closure=closure,
                requested=frozenset({root}),
                backends=backends,
                source_region_id=source_region.id,
                slice_root=root,
            )
        )
        variants.extend(callable_variants)
        if callable_variants:
            continue
        variants.extend(
            variant
            for specialization in source_region.specializations
            if specialization.source_member == root
            for variant in (
                _selected_specialization_variant(scan, source_region, specialization, backends),
            )
            if variant is not None
        )
    return tuple(variants)


def _selected_requested_callable_variant(
    inputs: _RequestedCallableVariant,
) -> tuple[_SelectedTypedRegion, ...]:
    """Choose one backend that supports a requested callable closure in full.

    Args:
        inputs: Directed slice, closure, backend order, and public binding evidence.

    Returns:
        tuple[_SelectedTypedRegion, ...]: Callable variant matching the explicit request, if any.
    """
    if not inputs.closure:
        return ()
    closure_set = frozenset(inputs.closure)
    bound_members = tuple(member for member in inputs.closure if member in inputs.requested)
    for backend in inputs.backends:
        assessment = _compiler_backend(backend).assess(inputs.region)
        if closure_set <= frozenset(assessment.supported_members):
            return (
                _SelectedTypedRegion(
                    scan=inputs.scan,
                    region=inputs.region,
                    variant_id=f"{inputs.region.id}@{backend}",
                    backend=backend,
                    assessment=assessment,
                    members=inputs.closure,
                    bound_members=bound_members,
                    source_region_id=inputs.source_region_id,
                    slice_root=inputs.slice_root,
                ),
            )
    return ()


def _backend_callable_members(
    region: TypedRegion,
    eligible: tuple[SymbolId, ...],
    assessment: BackendAssessment | None,
    *,
    excluded: tuple[SymbolId, ...],
) -> tuple[tuple[SymbolId, ...], tuple[SymbolId, ...]]:
    """Return complete generated closure members and public backend bindings.

    Args:
        region: Backend-neutral typed region being processed.
        eligible: Members accepted by backend and specialization checks.
        assessment: Backend capability assessment for the selected region.
        excluded: Paths or names that must not be copied.

    Returns:
        tuple[tuple[SymbolId, ...], tuple[SymbolId, ...]]: Generated callable members and
            source-class bindings supported by the selected backend.
    """
    if assessment is None:
        return (), ()
    supported = frozenset(assessment.supported_members)
    excluded_set = frozenset(excluded)
    generated: set[SymbolId] = set()
    bound: set[SymbolId] = set()
    for member in eligible:
        if member in excluded_set:
            continue
        closure = _runtime_member_closure(region, eligible, frozenset({member}))
        if closure and frozenset(closure) <= supported:
            generated.update(closure)
            bound.add(member)
    region_order = tuple(member.id for member in region.members)
    return (
        tuple(member for member in region_order if member in generated),
        tuple(member for member in region_order if member in bound),
    )


def _append_atomic_class_variant(
    variants: list[_SelectedTypedRegion],
    scan: ModuleScan,
    region: TypedRegion,
    assessment: BackendAssessment | None,
    requested: frozenset[SymbolId],
) -> str | None:
    region_members = frozenset(member.id for member in region.members)
    atomic_member = (
        _eligible_atomic_class(region, assessment)
        if assessment is not None and (not requested or region_members <= requested)
        else None
    )
    if atomic_member is None or assessment is None:
        return None
    variant_id = f"{region.id}@cython-class"
    variants.append(
        _SelectedTypedRegion(
            scan=scan,
            region=region,
            variant_id=variant_id,
            backend="cython",
            assessment=assessment,
            members=(atomic_member,),
        )
    )
    return variant_id


def _selected_specialization_variant(
    scan: ModuleScan,
    region: TypedRegion,
    specialization: RegionSpecialization,
    backends: tuple[Backend, ...],
) -> _SelectedTypedRegion | None:
    specialized_region = _specialized_region(region, specialization)
    for backend in backends:
        assessment = _compiler_backend(backend).assess(specialized_region)
        if specialization.source_member in assessment.supported_members:
            return _SelectedTypedRegion(
                scan=scan,
                region=specialized_region,
                variant_id=f"{specialization.id}@{backend}",
                backend=backend,
                assessment=assessment,
                members=(specialization.source_member,),
                specialization=specialization,
            )
    return None


def _eligible_atomic_class(
    region: TypedRegion,
    assessment: BackendAssessment,
) -> SymbolId | None:
    """Return the class binding only when Cython supports its complete region.

    Args:
        region: Backend-neutral typed region being processed.
        assessment: Backend capability assessment for the selected region.

    Returns:
        SymbolId | None: Atomic class region when every safety condition passes.
    """
    if not region.atomic_class or assessment.status != "supported":
        return None
    class_members = tuple(member for member in region.members if member.kind == "class")
    method_members = tuple(member for member in region.members if member.kind == "method")
    if len(class_members) != 1 or any(
        member.kind not in {"class", "method"} for member in region.members
    ):
        return None
    if not method_members:
        return None
    if any(member.execution_kind != "sync" for member in method_members):
        return None
    supported = set(assessment.supported_members)
    if any(member.id not in supported for member in region.members):
        return None
    return class_members[0].id


def _specialized_region(
    source_region: TypedRegion,
    specialization: RegionSpecialization,
) -> TypedRegion:
    """Materialize one backend-assessable view without changing generic source IR.

    Args:
        source_region: Unspecialized typed region used to derive the variant.
        specialization: Concrete guarded specialization applied to the region.

    Returns:
        TypedRegion: Region variant with concrete substitutions and runtime guards.
    """
    member = next(item for item in source_region.members if item.id == specialization.source_member)
    source_hash = hashlib.sha256(
        f"{source_region.source_hash}:{specialization.id}".encode()
    ).hexdigest()
    return TypedRegion(
        id=specialization.id,
        source_module=source_region.source_module,
        members=(member,),
        dependencies=tuple(
            dependency
            for dependency in source_region.dependencies
            if dependency.src == specialization.source_member
        ),
        type_bindings=specialization.type_bindings,
        bindings=(),
        decisions=(
            LoweringDecision(
                target=specialization.source_member.stable_id,
                action="specialize",
                reason=(
                    "all generic parameters resolved from "
                    + specialization.origin.replace("_", " ")
                ),
            ),
        ),
        source_hash=source_hash,
        atomic_class=False,
        specializations=(specialization,),
    )


def _eligible_typed_callables(
    region: TypedRegion,
    decisions: dict[str, LoweringDecision],
    *,
    hot: frozenset[SymbolId],
) -> tuple[SymbolId, ...]:
    return tuple(
        member.id
        for member in region.members
        if (
            (member.kind == "function" and member.binding_kind == "module")
            or (
                member.kind == "method"
                and member.binding_kind in {"instance_method", "staticmethod", "classmethod"}
                and not member.id.qualname.rsplit(".", 1)[-1].startswith("__")
                and not _owner_disallows_method_binding(
                    member.owner_class,
                    region.source_module.name,
                    decisions,
                )
                and not _member_requires_source_class(member.source_text)
            )
        )
        and (
            decisions[member.id.stable_id].action == "preserve"
            or (member.id in hot and decisions[member.id.stable_id].action in {"box", "fallback"})
        )
    )


def _runtime_member_closure(
    region: TypedRegion,
    eligible: tuple[SymbolId, ...],
    requested: frozenset[SymbolId],
) -> tuple[SymbolId, ...]:
    """Return a complete eligible callable closure, or empty when one edge is unsafe.

    Args:
        region: Backend-neutral typed region being processed.
        eligible: Members accepted by backend and specialization checks.
        requested: Explicit source members requested by the caller.

    Returns:
        tuple[SymbolId, ...]: Stable IDs required by the selected members at runtime.
    """
    eligible_set = frozenset(eligible)
    selected = set(requested.intersection(eligible_set))
    changed = True
    while changed:
        changed = False
        for dependency in region.dependencies:
            if (
                dependency.src not in selected
                or dependency.role != "runtime"
                or not dependency.requires_same_unit
            ):
                continue
            if not isinstance(dependency.dst, SymbolId):
                continue
            if (
                dependency.dst.module == region.source_module.name
                and dependency.dst not in eligible_set
            ):
                return ()
            if dependency.dst in eligible_set and dependency.dst not in selected:
                selected.add(dependency.dst)
                changed = True
    return tuple(member.id for member in region.members if member.id in selected)


def _missing_requested_members(
    requested: tuple[SymbolId, ...],
    selected: tuple[_SelectedTypedRegion, ...],
) -> tuple[SymbolId, ...]:
    """Return explicit requests not covered by any selected backend variant.

    Args:
        requested: Explicit source members requested by the caller.
        selected: Backend-supported region selections.

    Returns:
        tuple[SymbolId, ...]: Requested members absent from all selected backend variants.
    """
    covered: set[SymbolId] = set()
    for selection in selected:
        covered.update(selection.bound_members or selection.members)
        if selection.variant_id.endswith("@cython-class"):
            covered.update(member.id for member in selection.region.members)
    return tuple(member for member in requested if member not in covered)


def _owner_disallows_method_binding(
    owner_class: str | None,
    module_name: str,
    decisions: dict[str, LoweringDecision],
) -> bool:
    if owner_class is None:
        return True
    decision = decisions.get(f"{module_name}::{owner_class}")
    if decision is None:
        return False
    return any(
        reason in decision.reason
        for reason in (
            "decorators may register or replace",
            "dynamic behavior is blocked",
            "module binding is reassigned",
            "special method",
        )
    )


def _member_requires_source_class(source_text: str) -> bool:
    """Reject method extraction when Python's class compilation supplies semantics.

    Args:
        source_text: Original Python source text being analyzed or transformed.

    Returns:
        bool: Whether generated code must retain the source owner class.
    """
    tree = ast.parse(textwrap.dedent(source_text))
    return any(
        (isinstance(node, ast.Name) and node.id == "__class__")
        or (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "super"
            and not node.args
            and not node.keywords
        )
        or (
            isinstance(node, ast.Attribute)
            and node.attr.startswith("__")
            and not node.attr.endswith("__")
        )
        for node in ast.walk(tree)
    )


def _copy_source_roots(
    project: DiscoveredProject,
    build_root: Path,
) -> tuple[Path, ...]:
    staged_roots: list[Path] = []
    for source_root in project.config.source_roots:
        destination = build_root / _relative_source_root(project.config.root, source_root)
        if destination.resolve() == build_root.resolve():
            _copytree_contents(source_root, destination)
        else:
            shutil.copytree(source_root, destination, ignore=_copy_ignore)
        staged_roots.append(destination)
    return tuple(staged_roots)


def _copy_if_different(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    shutil.copy2(source, destination)


def _overlay_staged_sources(
    source_roots: tuple[Path, ...],
    install_root: Path,
    source_paths: tuple[Path, ...],
) -> None:
    """Overlay only shimmed modules that already exist in the backend wheel.

    Args:
        source_roots: Import roots visible to the target project.
        install_root: Temporary install payload receiving compiled artifacts.
        source_paths: Prepared source files included in the build.

    Raises:
        ValueError: If the baseline wheel omitted a source module that Atoll must shim.
    """
    for source_path in dict.fromkeys(source_paths):
        relative = _source_relative_path(source_path, source_roots)
        destination = install_root / relative
        if not destination.is_file():
            raise ValueError(
                f"target PEP 517 wheel omitted a compiled source module: {relative.as_posix()}"
            )
        shutil.copy2(source_path, destination)


def _overlay_install_payload(
    source_roots: tuple[Path, ...],
    install_root: Path,
    source_paths: tuple[Path, ...],
) -> str | None:
    """Overlay source modules and artifacts, normalizing backend omissions as failure text.

    Args:
        source_roots: Import roots visible to the target project.
        install_root: Temporary install payload receiving compiled artifacts.
        source_paths: Prepared source files included in the build.

    Returns:
        str | None: Paths installed into the staged wheel payload.
    """
    try:
        _overlay_staged_sources(source_roots, install_root, source_paths)
        _copy_atoll_artifacts(source_roots, install_root)
    except (OSError, ValueError) as error:
        return f"install payload overlay failed: {error}"
    return None


def _source_relative_path(path: Path, source_roots: tuple[Path, ...]) -> Path:
    for source_root in source_roots:
        try:
            return path.relative_to(source_root)
        except ValueError:
            continue
    raise ValueError(f"staged source is outside copied source roots: {path}")


def _copy_atoll_artifacts(source_roots: tuple[Path, ...], install_root: Path) -> None:
    for source_root in source_roots:
        artifact_root = source_root / ".atoll" / "artifacts"
        if not artifact_root.exists():
            continue
        for path in sorted(artifact_root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source_root)
            destination = install_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def _wheel_output_path(output_dir: Path, metadata: _ProjectMetadata) -> Path:
    distribution = _wheel_safe_name(metadata.name)
    version = _wheel_safe_version(metadata.version)
    return output_dir / f"{distribution}-{version}-{_wheel_tag()}.whl"


def _remove_failed_wheels(project: DiscoveredProject, output_dir: Path) -> None:
    """Remove wheel artifacts that could be mistaken for the failed attempt.

    Args:
        project: Discovered target project configuration and modules.
        output_dir: Directory receiving generated wheel artifacts.
    """
    metadata = _project_metadata(project.config.root)
    default_output = project.config.root / ".atoll" / "dist"
    if output_dir.resolve() != default_output.resolve():
        _wheel_output_path(output_dir, metadata).unlink(missing_ok=True)
        return
    distribution = _wheel_safe_name(metadata.name)
    version = _wheel_safe_version(metadata.version)
    for wheel_path in output_dir.glob(f"{distribution}-{version}-*.whl"):
        wheel_path.unlink()


def _project_metadata(root: Path) -> _ProjectMetadata:
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        return _ProjectMetadata(
            name=root.name,
            version="0+atoll",
            requires_python=None,
            dependencies=(),
        )
    data = cast(dict[str, object], tomllib.loads(pyproject.read_text(encoding="utf-8")))
    project = _mapping(data.get("project"))
    name = _string(project.get("name")) or root.name
    version = _string(project.get("version")) or "0+atoll"
    requires_python = _string(project.get("requires-python"))
    dependencies = tuple(
        dependency
        for item in _sequence(project.get("dependencies"))
        if (dependency := _string(item))
    )
    return _ProjectMetadata(
        name=name,
        version=version,
        requires_python=requires_python,
        dependencies=dependencies,
    )


def _wheel_tag() -> str:
    return str(next(tags.sys_tags()))


def _wheel_safe_name(value: str) -> str:
    return re.sub(r"[-_.]+", "_", value).strip("_").lower()


def _wheel_safe_version(value: str) -> str:
    return value.replace("-", "_")


def _resolve_output_dir(root: Path, output_dir: Path | None) -> Path:
    if output_dir is None:
        return root / ".atoll" / "dist"
    if output_dir.is_absolute():
        return output_dir.resolve()
    return (root / output_dir).resolve()


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _copytree_contents(source: Path, destination: Path) -> None:
    ignored_names = _copy_ignore(str(source), [item.name for item in source.iterdir()])
    for item in source.iterdir():
        if item.name in ignored_names:
            continue
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, ignore=_copy_ignore)
        else:
            shutil.copy2(item, target)


def _copy_pep517_project(
    source: Path,
    destination: Path,
    *,
    excluded_output: Path,
) -> None:
    """Copy complete build inputs while excluding Atoll state and native residue.

    Args:
        source: Source expression, declaration, or filesystem path being processed.
        destination: Filesystem destination receiving copied or overlaid content.
        excluded_output: Output directory excluded from the copied project tree.
    """
    source_root = source.resolve()
    excluded_root = excluded_output.resolve()

    def ignore(directory: str, names: list[str]) -> set[str]:
        directory_path = Path(directory).resolve()
        ignored = {
            name
            for name in names
            if name in _PEP517_IGNORED_NAMES or name.endswith((".so", ".pyd"))
        }
        if directory_path == source_root:
            ignored.update(name for name in names if name in {"build", "dist"})
        for name in names:
            if (directory_path / name).resolve() == excluded_root:
                ignored.add(name)
        return ignored

    shutil.copytree(source_root, destination, ignore=ignore)
    _write_gitdir_pointer(source_root, destination)


def _remove_quality_gate_sources(project: DiscoveredProject, copied_project: Path) -> None:
    """Remove importable checkout modules while preserving tests and benchmark files.

    Args:
        project: Discovered target project configuration and modules.
        copied_project: Temporary project copy used for a PEP 517 build.
    """
    for module in project.modules:
        try:
            relative = module.path.relative_to(project.config.root)
        except ValueError:
            continue
        (copied_project / relative).unlink(missing_ok=True)


def _write_gitdir_pointer(source: Path, destination: Path) -> None:
    """Expose read-only VCS metadata to dynamic-version PEP 517 backends.

    Args:
        source: Source expression, declaration, or filesystem path being processed.
        destination: Filesystem destination receiving copied or overlaid content.
    """
    source_git = source / ".git"
    if source_git.is_dir():
        git_dir = source_git.resolve()
    elif source_git.is_file():
        first_line = source_git.read_text(encoding="utf-8").splitlines()[0]
        prefix = "gitdir:"
        if not first_line.startswith(prefix):
            return
        value = first_line.removeprefix(prefix).strip()
        candidate = Path(value)
        git_dir = candidate.resolve() if candidate.is_absolute() else (source / candidate).resolve()
    else:
        return
    (destination / ".git").write_text(f"gitdir: {git_dir}\n", encoding="utf-8")


def _copy_ignore(_directory: str, names: list[str]) -> set[str]:
    return {
        name for name in names if name in _GENERATED_DIR_NAMES or name.endswith((".so", ".pyd"))
    }


def _staged_module(
    module: ModuleId,
    project: DiscoveredProject,
    staged_source_roots: tuple[Path, ...],
) -> ModuleId:
    staged_source_root = _staged_source_root(module, project, staged_source_roots)
    for source_root in project.config.source_roots:
        try:
            relative = module.path.relative_to(source_root)
        except ValueError:
            continue
        return ModuleId(name=module.name, path=staged_source_root / relative)
    raise ValueError(f"module is outside configured source roots: {module.name}")


def _staged_source_root(
    module: ModuleId,
    project: DiscoveredProject,
    staged_source_roots: tuple[Path, ...],
) -> Path:
    for index, source_root in enumerate(project.config.source_roots):
        try:
            module.path.relative_to(source_root)
        except ValueError:
            continue
        return staged_source_roots[index]
    raise ValueError(f"module is outside configured source roots: {module.name}")


def _relative_source_root(root: Path, source_root: Path) -> Path:
    try:
        return source_root.relative_to(root)
    except ValueError:
        return Path(f"source_{abs(hash(source_root))}")


def _find_module(modules: tuple[ModuleId, ...], module_name: str) -> ModuleId:
    for module in modules:
        if module.name == module_name:
            return module
    raise ValueError(f"module not found under configured source roots: {module_name}")


def _mapping(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        raw = cast(dict[object, object], value)
        return {str(key): item for key, item in raw.items()}
    return {}


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, list):
        return tuple(cast(list[object], value))
    return ()


def _string(value: object) -> str | None:
    return value if isinstance(value, str) else None
