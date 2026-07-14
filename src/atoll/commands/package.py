"""Build installable Atoll artifacts without modifying source files."""

from __future__ import annotations

import ast
import hashlib
import re
import shutil
import sys
import tempfile
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
from atoll.baseline_cache import restore_baseline_wheel, store_baseline_wheel
from atoll.execution_plans.anyio_task_preserving import ANYIO_TASK_PRESERVING_BACKEND
from atoll.execution_plans.base import ExecutionPlanBackend
from atoll.execution_plans.cache import (
    restore_execution_plan_cache,
    store_execution_plan_cache,
)
from atoll.execution_plans.callback_backed import CALLBACK_BACKED_BACKEND
from atoll.execution_plans.models import (
    ExecutionPlan,
    ExecutionPlanAssessmentContext,
    ExecutionPlanCacheStatus,
    ExecutionPlanDiagnostic,
    ExecutionPlanDiagnosticSeverity,
    ExecutionPlanStageContext,
    ExecutionPlanTrial,
    ExecutionPlanTrialStatus,
    PlanRejection,
    StagedExecutionPlan,
)
from atoll.execution_plans.task_preserving import TASK_PRESERVING_BACKEND
from atoll.generation.buffer_kernel import (
    BUFFER_KERNEL_GENERATOR_VERSION,
    BufferKernelGenerationRequest,
    generate_buffer_kernel,
)
from atoll.generation.call_chain import (
    CALL_CHAIN_GENERATOR_VERSION,
    CallChainGenerationRequest,
    generate_call_chain_kernel,
)
from atoll.generation.outlined_region import (
    OUTLINED_REGION_GENERATOR_VERSION,
    generate_outlined_region,
)
from atoll.generation.region_shim import (
    RegionShimConfig,
    insert_or_replace_region_shim,
    remove_region_shim,
)
from atoll.generation.run_guard import (
    RUN_GUARD_GENERATOR_VERSION,
    RunGuardGenerationRequest,
    generate_run_guard,
)
from atoll.generation.scalar_kernel import (
    SCALAR_KERNEL_GENERATOR_VERSION,
    ScalarKernelGenerationRequest,
    generate_scalar_kernel,
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
from atoll.native_optimization.buffer_analysis import (
    BufferAnalysisResult,
    BufferKernelPlan,
    analyze_buffer_scan,
)
from atoll.native_optimization.call_chains import (
    CallChainAnalysisResult,
    CallChainPlan,
    analyze_call_chain_scan,
    call_chain_runtime_guards,
)
from atoll.native_optimization.run_guard import RunGuardNativePlan, build_run_guard_region
from atoll.native_optimization.scalar_analysis import (
    ScalarAnalysisResult,
    ScalarKernelPlan,
    ScalarWidthProof,
    analyze_scalar_scan,
)
from atoll.optimization_policy import (
    DEFAULT_MINIMUM_MARGINAL_SPEEDUP,
    PROFILE_GUIDED_MINIMUM_MARGINAL_SPEEDUP,
)
from atoll.profile_plan_cache import ProfilePlanDecision, ProfilePlanIdentity, select_profile_plan
from atoll.project import DiscoveredProject, discover_project
from atoll.region_cache import (
    compile_many_with_region_cache,
    compile_with_region_cache,
    probe_region_cache,
)
from atoll.runtime.execution_plan_performance import (
    ExecutionPlanBenchmarkConfig,
    ExecutionPlanBenchmarkProgress,
    ExecutionPlanBenchmarkResult,
    run_execution_plan_benchmark,
)
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
    CandidateDecisionReason,
    MappedCandidateDecision,
    ProfileCallEdgeTarget,
    ProfileResult,
    run_baseline_profile,
    select_profile_candidates,
)
from atoll.source_optimization import (
    SourceOptimizationAssessment,
    SourceOptimizationPlan,
    SourceOptimizationPlanningOptions,
    SourceOptimizationPlanningResult,
    SourceOptimizationTrial,
    build_source_optimization_plans,
    materialize_transformed_files,
    validate_source_application_root,
)
from atoll.source_optimization.search import (
    SourceOptimizationSearchOptions,
    SourceOptimizationSearchResult,
    run_source_optimization_search,
)
from atoll.source_snapshot import copy_source_snapshot, symlink_target_bytes
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
_EXECUTION_PLAN_BACKENDS: tuple[ExecutionPlanBackend, ...] = (
    CALLBACK_BACKED_BACKEND,
    ANYIO_TASK_PRESERVING_BACKEND,
    TASK_PRESERVING_BACKEND,
)

_CANDIDATE_BENCHMARK_WARMUPS = 1
_CANDIDATE_BENCHMARK_SAMPLES = 3
_CANDIDATE_MINIMUM_SPEEDUP = PROFILE_GUIDED_MINIMUM_MARGINAL_SPEEDUP
_SPECIALIZED_VARIANT_MINIMUM_SPEEDUP = DEFAULT_MINIMUM_MARGINAL_SPEEDUP
_EXECUTION_PLAN_BENCHMARK_SAMPLES = 7
_EXECUTION_PLAN_MINIMUM_SPEEDUP = DEFAULT_MINIMUM_MARGINAL_SPEEDUP
_WHEEL_TAG_COMPONENT_COUNT = 3
_SCALAR_INT32_WIDTH = 32
_SCALAR_INT32_DISPATCH_RANK = 10
_SCALAR_INT64_DISPATCH_RANK = 20
_BUFFER_DISPATCH_RANK = 30
_RUN_GUARD_DISPATCH_RANK = 25
_MAX_PROFILED_CALL_CHAIN_ROOTS = 4
_MINIMUM_CYTHON_BATCH_SIZE = 2
_MINIMUM_INTERACTION_VARIANTS = 2
_PROFILE_PLAN_CACHE_FORMAT_VERSION = "1"
_PROFILE_PLAN_LOWERING_VERSION = "profile-native-selection-v2"
_BACKEND_POLICY_BYPASS_PREFIX = "BACKEND_POLICY_BYPASS:"

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
_STAGED_SOURCE_DIGEST_IGNORED_TOP_LEVEL = frozenset(
    {
        ".atoll",
        "accepted-source-baseline",
        "accepted-source-project",
        "baseline-install",
        "candidate-dist",
        "execution-plan-trials",
        "fusion-research",
        "pep517-dist",
        "pep517-project",
        "profile",
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
        apply_source: Whether an accepted source patch should be applied transactionally.
    """

    root: Path
    module_name: str | None = None
    output_dir: Path | None = None
    keep_install_tree: bool = False
    progress: PackageProgress | None = None
    selected_members: tuple[SymbolId, ...] = ()
    cache_dir: Path | None = None
    run_quality_gates: bool = True
    apply_source: bool = False


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
        scalar_analyses: Fixed-width scalar plans and explicit frontend fallbacks.
        call_chain_analyses: Direct native call-chain plans and explicit fallbacks.
        buffer_analyses: Zero-copy standard-buffer plans and explicit fallbacks.
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
        applied_execution_plans: Plan IDs retained in the promoted payload.
        execution_plan_trials: Semantic and marginal benchmark evidence for plan candidates.
        fusion_plans: Deterministic report-only task-fusion safety decisions.
        fusion_trials: Three-arm research trials run only for eligible generated variants.
        source_optimization_plans: Profile-ranked source rewrite plans retained for reporting.
        source_optimization_assessments: Static and dynamic 3x gate evidence for source plans.
        source_optimization_trials: Disposable source candidate trials, when implemented.
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
    scalar_analyses: tuple[ScalarAnalysisResult, ...] = ()
    call_chain_analyses: tuple[CallChainAnalysisResult, ...] = ()
    buffer_analyses: tuple[BufferAnalysisResult, ...] = ()
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
    applied_execution_plans: tuple[str, ...] = ()
    execution_plan_trials: tuple[ExecutionPlanTrial, ...] = ()
    fusion_plans: tuple[FusionPlan, ...] = ()
    fusion_trials: tuple[FusionTrial, ...] = ()
    source_optimization_plans: tuple[SourceOptimizationPlan, ...] = ()
    source_optimization_assessments: tuple[SourceOptimizationAssessment, ...] = ()
    source_optimization_trials: tuple[SourceOptimizationTrial, ...] = ()


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
class _ProfileCandidateRejection:
    """One statically known callable excluded from automatic profile selection.

    Attributes:
        symbol: Source callable that runtime profiling may observe.
        reason: Stable capability reason retained in profile and compile reports.
    """

    symbol: SymbolId
    reason: CandidateDecisionReason


@dataclass(frozen=True, slots=True)
class _ProfileCandidateSupport:
    """Backend-supported roots and explicit rejections for profile ranking.

    Automatic profiling may only consume ``supported`` roots. Rejections stay
    attached to their real static symbols so unsupported hot code remains
    visible without consuming candidate limits or aborting compilation.

    Attributes:
        supported: Callable roots with a complete backend-supported directed slice.
        rejected: Callable roots that cannot be independently bound or lowered.
    """

    supported: tuple[SymbolId, ...]
    rejected: tuple[_ProfileCandidateRejection, ...]


@dataclass(frozen=True, slots=True)
class _ProfileCompileSelectionScope:
    """Reusable support evidence and identity for strict selection caching.

    Attributes:
        identity: Stable baseline or transformed-source scope identity.
        support: Capability results already used for profile ranking. Reusing
            them prevents whole-project directed-slice analysis from running twice.
    """

    identity: str
    support: _ProfileCandidateSupport


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
        lowering_mode: Whether the target owns a callable, outlined blocks, or
            one helper introduced by an accepted source transform.
        native_helpers: Private helper names used by outlined or source-fused lowering.
        fallback_reason: Ordered backend rejection or policy-bypass evidence preceding success.
        minimum_marginal_speedup: Candidate-specific profitability floor, or
            ``None`` for the generic compiler policy.
        profitability_symbols: Public hot bindings whose workload is represented
            by a private compiled helper. Empty uses the generated public bindings.
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
    minimum_marginal_speedup: float | None = None
    profitability_symbols: tuple[SymbolId, ...] = ()

    def __post_init__(self) -> None:
        """Require outlined variants to identify every native helper.

        Raises:
            ValueError: If lowering mode and helper metadata contradict each other.
        """
        if self.lowering_mode != "whole-callable" and not self.native_helpers:
            raise ValueError("partial prepared regions require native helpers")
        if self.lowering_mode == "whole-callable" and self.native_helpers:
            raise ValueError("whole-callable prepared regions cannot declare native helpers")
        if self.minimum_marginal_speedup is not None and self.minimum_marginal_speedup <= 1.0:
            raise ValueError("native variant marginal speedup must be greater than 1.0")
        if len(set(self.profitability_symbols)) != len(self.profitability_symbols):
            raise ValueError("native variant profitability symbols must be unique")


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
        source_tree_digest: Exact copied-source identity for project-scoped decisions.
        enable_project_circuit: Whether automatic whole-project selection may bypass retries.
    """

    build_root: Path
    staged_source_roots: tuple[Path, ...]
    mypy_cache_dir: Path
    compile_cache_dir: Path
    progress: PackageProgress | None
    source_tree_digest: str = ""
    enable_project_circuit: bool = False


@dataclass(slots=True)
class _BackendCircuitBuildState:
    """Coordinate project-scoped backend rejections during one native build.

    Persistent decisions and artifacts remain owned by the region cache. This
    mutable state only routes later variants around a rejection that was already
    proven for their imported source package in the current invocation.

    Attributes:
        context: Backend filesystem and toolchain context.
        cache_root: Persistent region cache namespace.
        progress: Optional user-facing progress callback.
        batched_cython: Independently selected top-level Cython results by index.
        triggers: First project rejection by backend and top-level import package.
        primaries: Preferred backend cache results discovered before fallback batching.
        fallbacks: Precompiled Cython fallback results by variant ID.
        enabled: Whether this automatic whole-project build permits circuit bypasses.
    """

    context: BackendCompileContext
    cache_root: Path
    progress: PackageProgress | None
    batched_cython: dict[int, BackendCompileResult]
    triggers: dict[tuple[Backend, str], str]
    primaries: dict[str, BackendCompileResult]
    fallbacks: dict[str, BackendCompileResult]
    enabled: bool


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
class _ScalarVariantContext:
    """Staged filesystem and scan evidence shared by scalar width variants.

    Attributes:
        project: Discovered target project used to map copied source roots.
        build_root: Disposable root receiving generated scalar compilation units.
        staged_source_roots: Copied import roots used for source-clean compilation.
        scan: Revalidated scan of the staged module copy.
        region: Typed region containing the scalar member.
    """

    project: DiscoveredProject
    build_root: Path
    staged_source_roots: tuple[Path, ...]
    scan: ModuleScan
    region: TypedRegion


@dataclass(frozen=True, slots=True)
class _ScalarExtensionContext:
    """Compile configuration and staged roots for scalar variant extension.

    Attributes:
        project: Discovered target project and configured backend order.
        build_root: Disposable root receiving generated scalar compilation units.
        staged_source_roots: Copied import roots used for source-clean compilation.
        analyses: Scalar plans and rejection evidence for selected scans.
        progress: Optional callback receiving scalar preparation progress.
    """

    project: DiscoveredProject
    build_root: Path
    staged_source_roots: tuple[Path, ...]
    analyses: tuple[ScalarAnalysisResult, ...]
    progress: PackageProgress | None


@dataclass(frozen=True, slots=True)
class _CallChainExtensionContext:
    """Compile configuration and staged roots for direct call-chain variants.

    Attributes:
        project: Discovered target project and configured backend order.
        build_root: Disposable root receiving generated call-chain units.
        staged_source_roots: Copied import roots used for source-clean compilation.
        analyses: Direct call-chain plans and rejection evidence for selected scans.
        progress: Optional callback receiving call-chain preparation progress.
    """

    project: DiscoveredProject
    build_root: Path
    staged_source_roots: tuple[Path, ...]
    analyses: tuple[CallChainAnalysisResult, ...]
    progress: PackageProgress | None


@dataclass(frozen=True, slots=True)
class _BufferExtensionContext:
    """Compile configuration and staged roots for zero-copy buffer variants.

    Attributes:
        project: Discovered target project and configured backend order.
        build_root: Disposable root receiving generated buffer units.
        staged_source_roots: Copied import roots used for source-clean compilation.
        analyses: Buffer proof plans and rejection evidence for selected scans.
        progress: Optional callback receiving buffer preparation progress.
    """

    project: DiscoveredProject
    build_root: Path
    staged_source_roots: tuple[Path, ...]
    analyses: tuple[BufferAnalysisResult, ...]
    progress: PackageProgress | None


@dataclass(frozen=True, slots=True)
class _RunGuardExtensionContext:
    """Accepted-source run guards and staged roots for Cython composition.

    Attributes:
        project: Disposable transformed project containing the Python fallback.
        build_root: Disposable root receiving generated Cython helper units.
        staged_source_roots: Copied transformed import roots used for compilation.
        plans: Content-addressed run-guard plans carried by the accepted source arm.
        progress: Optional callback receiving preparation decisions.
    """

    project: DiscoveredProject
    build_root: Path
    staged_source_roots: tuple[Path, ...]
    plans: tuple[RunGuardNativePlan, ...]
    progress: PackageProgress | None


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
        scalar_analyses: Fixed-width scalar proofs and explicit fallbacks for selected scans.
        call_chain_analyses: Direct call-chain plans and explicit fallbacks for selected scans.
        buffer_analyses: Zero-copy buffer plans and explicit fallbacks for selected scans.
        run_guard_plans: Source-fused run guards carried by an accepted source patch.
    """

    selected: tuple[_SelectedTypedRegion, ...]
    typed_regions: tuple[TypedRegion, ...]
    preflight_skipped: tuple[PackagePreflightFailure, ...]
    native_readiness: tuple[NativeReadiness, ...]
    execution_plans: tuple[ExecutionPlan | PlanRejection, ...] = ()
    fusion_plans: tuple[FusionPlan, ...] = ()
    scalar_analyses: tuple[ScalarAnalysisResult, ...] = ()
    call_chain_analyses: tuple[CallChainAnalysisResult, ...] = ()
    buffer_analyses: tuple[BufferAnalysisResult, ...] = ()
    run_guard_plans: tuple[RunGuardNativePlan, ...] = ()


@dataclass(frozen=True, slots=True)
class _ProfileSelectedPackageContext:
    """Inputs required to route one automatic or explicit native selection.

    Attributes:
        options: Source-clean command and output policy.
        project: Active target project and compile configuration.
        scans: Selected source scans used for directed region selection.
        baseline: Prepared interpreted wheel payload, when profiling was configured.
        preparation: Baseline, profile, and planning evidence for failure reporting.
        typed_regions: Backend-neutral regions retained in the compile report.
        preflight_selected: Static selections prepared before profiling.
        profile: Current profile and supported candidate decisions.
        execution_plans: Scheduler plans and rejections retained for reporting.
        fusion_plans: Task-fusion research plans retained for reporting.
        scalar_analyses: Fixed-width scalar proof results.
        call_chain_analyses: Direct call-chain proof results.
        buffer_analyses: Zero-copy buffer proof results.
        automatic_benchmark: Whether selection came from configured automatic profiling.
    """

    options: PackageOptions
    project: DiscoveredProject
    scans: tuple[ModuleScan, ...]
    baseline: _BaselineWheelPayload | None
    preparation: _ProfilePreparation
    typed_regions: tuple[TypedRegion, ...]
    preflight_selected: tuple[_SelectedTypedRegion, ...]
    profile: ProfileResult | None
    execution_plans: tuple[ExecutionPlan | PlanRejection, ...]
    fusion_plans: tuple[FusionPlan, ...]
    scalar_analyses: tuple[ScalarAnalysisResult, ...]
    call_chain_analyses: tuple[CallChainAnalysisResult, ...]
    buffer_analyses: tuple[BufferAnalysisResult, ...]
    automatic_benchmark: bool


@dataclass(frozen=True, slots=True)
class _PreparationFailureContext:
    """Inputs used to report a source-clean native preparation failure.

    Attributes:
        options: Command options that determine output and cleanup locations.
        project: Discovered target project and source-clean configuration.
        package: Static region-selection evidence retained in the report.
        profile: Optional unmeasured profile attached to the failed attempt.
        failures: Backend lowering failures retained as interpreted fallbacks.
        wheel_omissions: Variants excluded because their modules are absent from the wheel.
    """

    options: PackageOptions
    project: DiscoveredProject
    package: _TypedRegionPackageContext
    profile: ProfileResult | None
    failures: tuple[PackageRegionBuildFailure, ...]
    wheel_omissions: tuple[PackageRegionBuildFailure, ...]


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
class _BaselineWheelPreparation:
    """Inputs for producing one reusable interpreted baseline wheel payload.

    Attributes:
        project: Discovered target project configuration and modules.
        build_root: Root of the temporary source-clean build tree.
        install_root: Temporary install payload receiving compiled artifacts.
        cache_root: Caller-owned cache root for reusable baseline wheel state.
        progress: Optional progress callback for long-running work.
        run_quality_gates: Whether verification, tests, and benchmarks should run.
    """

    project: DiscoveredProject
    build_root: Path
    install_root: Path
    cache_root: Path
    progress: PackageProgress | None
    run_quality_gates: bool


@dataclass(frozen=True, slots=True)
class OptimizationArm:
    """One accepted source or native composition available to later stages.

    The active project may be a disposable transformed copy, while
    ``report_project`` always identifies the immutable checkout shown in user
    reports. ``baseline`` is the accepted wheel/payload that the next stage
    must improve; a failed later stage returns to that baseline rather than
    invalidating earlier profitable work.

    Attributes:
        report_project: Original discovered checkout used for public paths.
        active_project: Source tree scanned and compiled by the next stage.
        baseline: Accepted wheel and unpacked payload for marginal trials.
        source_search: Accepted source-optimization evidence, when this arm
            materializes a transformed project.
    """

    report_project: DiscoveredProject
    active_project: DiscoveredProject
    baseline: _BaselineWheelPayload
    source_search: SourceOptimizationSearchResult | None = None


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
        requires_profitable_optimization: Whether profiling requires a retained optimization.
        profitable_optimization_applied: Whether a native region or execution plan was retained.
    """

    options: PackageOptions
    project: DiscoveredProject
    output_dir: Path
    build_root: Path
    install_root: Path
    baseline: _BaselineWheelPayload
    verification_plan: PackageVerificationPlan
    build: CompileAttempt
    requires_profitable_optimization: bool = False
    profitable_optimization_applied: bool = False


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
        staged_source_roots: Copied source roots containing the generated candidate
            shims and native artifacts.
        progress: Optional progress callback for long-running trials.
    """

    successful: tuple[_PreparedTypedRegion, ...]
    skipped: tuple[PackageRegionBuildFailure, ...]
    profile: ProfileResult
    project: DiscoveredProject
    baseline: _BaselineWheelPayload
    payload_root: Path
    staged_source_roots: tuple[Path, ...]
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
class _RuntimeSafetySelectionResult:
    """Compiled subset that survives isolated import and routing verification.

    Attributes:
        successful: Variants retained after deterministic failure isolation.
        artifacts: Native artifacts reachable from the retained variants.
        build: Build evidence extended with safety-verification timings.
        failures: Variants downgraded to interpreted fallback after verification.
        verification_steps: Child-process attempts used to isolate unsafe variants.
        overlay_error: Failure rebuilding the verified payload, when present.
    """

    successful: tuple[_PreparedTypedRegion, ...]
    artifacts: tuple[ArtifactRecord, ...]
    build: CompileAttempt
    failures: tuple[PackageRegionBuildFailure, ...] = ()
    verification_steps: tuple[PackageVerificationResult, ...] = ()
    overlay_error: str | None = None


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


@dataclass(frozen=True, slots=True)
class _ExecutionPlanApplicationContext:
    """Inputs for disposable task-preserving plan trials.

    Attributes:
        options: Command options controlling quality gates and progress output.
        project: Discovered target project and compile policy.
        baseline: Immutable baseline wheel and quality-project roots.
        build_root: Temporary build root that owns disposable trial payloads.
        install_root: Current accepted native or planned payload.
        plans: Profile-selected scheduler plans considered in hotness order.
        accepted_region_ids: Native region variants enabled during plan trials.
    """

    options: PackageOptions
    project: DiscoveredProject
    baseline: _BaselineWheelPayload
    build_root: Path
    install_root: Path
    plans: tuple[ExecutionPlan | PlanRejection, ...]
    accepted_region_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class _ExecutionPlanOnlyContext:
    """Baseline and profile state for a Python-overlay-only package.

    Attributes:
        options: Validated source-clean command options.
        project: Discovered target project and compile policy.
        typed_regions: Native regions retained for report compatibility only.
        execution_plans: Selected and rejected scheduler plans from profiling.
        prepared_baseline: Baseline wheel already built and profiled.
        profile: Dynamic evidence that selected the scheduler plan.
    """

    options: PackageOptions
    project: DiscoveredProject
    typed_regions: tuple[TypedRegion, ...]
    execution_plans: tuple[ExecutionPlan | PlanRejection, ...]
    prepared_baseline: _BaselineWheelPayload | None
    profile: ProfileResult | None


@dataclass(frozen=True, slots=True)
class _ExecutionPlanApplicationOutcome:
    """Applied plan IDs and trial evidence retained after disposable trials.

    Attributes:
        applied_plan_ids: Plans whose staged payload passed semantics and profitability.
        trials: Ordered staging, semantic, and benchmark decisions.
        timings: Trial command timings appended to compile evidence.
    """

    applied_plan_ids: tuple[str, ...] = ()
    trials: tuple[ExecutionPlanTrial, ...] = ()
    timings: tuple[CompilePhaseTiming, ...] = ()


@dataclass(frozen=True, slots=True)
class _ExecutionPlanTrialRecord:
    """Inputs normalized into one immutable plan-trial report record.

    Attributes:
        plan: Plan represented by the trial.
        backend: Backend that staged the disposable candidate.
        staged: Validated staged payload files.
        semantic: Configured semantic-command evidence.
        status: Accepted, rejected, failed-semantics, or unavailable outcome.
        reason: Plain-language decision evidence.
        diagnostics: Ordered backend and trial diagnostics.
        benchmark: Three-arm benchmark evidence, when semantics passed.
        cache_status: Whether staging restored or generated the payload changes.
    """

    plan: ExecutionPlan
    backend: ExecutionPlanBackend
    staged: StagedExecutionPlan
    semantic: CommandRunEvidence
    status: ExecutionPlanTrialStatus
    reason: str
    diagnostics: tuple[ExecutionPlanDiagnostic, ...]
    benchmark: ExecutionPlanBenchmarkResult | None
    cache_status: ExecutionPlanCacheStatus


@dataclass(frozen=True, slots=True)
class _ExecutionPlanStagingResult:
    """Staged payload plus cache evidence that must survive into reports.

    Attributes:
        staged: Backend payload changes restored or generated for the trial.
        cache_status: Strict cache lookup decision.
        diagnostics: Non-fatal cache corruption or write diagnostics.
    """

    staged: StagedExecutionPlan
    cache_status: ExecutionPlanCacheStatus
    diagnostics: tuple[ExecutionPlanDiagnostic, ...] = ()


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
    scalar_analyses = _scalar_analyses(scans, options.progress)
    call_chain_analyses = _call_chain_analyses(scans, options.progress)
    buffer_analyses = _buffer_analyses(scans, options.progress)
    execution_plans = build_execution_plans(scans, None)
    _progress(options.progress, f"scanned {len(scans)} module(s) in {_duration(scan_started)}")
    preflight_selected = _selected_typed_regions(
        scans,
        project.config.compile.backends,
        options.selected_members,
        hot_members=options.selected_members,
    )
    preflight_missing = _missing_requested_members(options.selected_members, preflight_selected)
    plan_profile_targets = execution_plan_profile_targets(scans)
    preflight_error = _region_selection_error(
        profile=None,
        selection_members=options.selected_members,
        profile_members=(),
        selected=preflight_selected,
        missing=preflight_missing,
    )
    automatic_benchmark = (
        project.config.compile.benchmark_command is not None and not options.selected_members
    )
    if preflight_error is not None and not plan_profile_targets and not automatic_benchmark:
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
    preparation = _prepare_profile_guided_selection(
        options,
        project,
        scans,
        call_chain_analyses,
    )
    if preparation.failure is not None:
        return preparation.failure
    baseline = preparation.baseline
    profile = preparation.profile
    profile_support = _ProfileCandidateSupport(supported=(), rejected=())
    if profile is not None:
        support_roots = (
            _profile_support_roots(profile, call_chain_analyses)
            if profile.status == "profiled"
            else None
        )
        profile_support = _profile_candidate_support(
            scans,
            project.config.compile.backends,
            roots=support_roots,
        )
        profile = _select_profile_with_call_chains(
            profile,
            scans,
            call_chain_analyses,
            project.config.compile.backends,
            support=profile_support,
        )
        _progress(
            options.progress,
            (
                f"profile selected {len(profile.selected_symbols)} hot member(s) covering "
                f"{profile.selected_hot_coverage:.1%} of mapped project samples"
            ),
        )
    execution_plans = build_execution_plans(scans, profile)
    selected_execution_plans = tuple(
        plan for plan in execution_plans if isinstance(plan, ExecutionPlan)
    )
    if execution_plans:
        _progress(
            options.progress,
            (
                f"execution-plan discovery produced {len(execution_plans)} candidate(s); "
                f"{len(selected_execution_plans)} selected for future backend assessment"
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
    source_optimization = (
        build_source_optimization_plans(
            scans,
            execution_plans,
            SourceOptimizationPlanningOptions(
                profile=profile,
                compile_config=project.config.compile,
                project_root=project.config.root,
            ),
        )
        if not options.selected_members
        else SourceOptimizationPlanningResult(plans=(), assessments=())
    )
    if source_optimization.plans:
        trial_ready = sum(
            assessment.status == "trial-ready" for assessment in source_optimization.assessments
        )
        _progress(
            options.progress,
            (
                f"source-optimization planning produced "
                f"{len(source_optimization.plans)} plan(s); {trial_ready} ready for bounded trials"
            ),
        )
    source_search = _run_source_optimization_trials(
        options=options,
        project=project,
        baseline=baseline,
        planning=source_optimization,
    )
    preparation = replace(
        preparation,
        profile=profile,
        execution_plans=execution_plans,
        fusion_plans=fusion_plans,
    )
    source_terminal = _source_optimization_continuation_result(
        options=options,
        project=project,
        preparation=preparation,
        planning=source_optimization,
        search=source_search,
    )
    if source_terminal is not None:
        return source_terminal
    profile = _stabilize_profile_compile_selection(
        _ProfileCompileSelectionScope(identity="baseline", support=profile_support),
        options=options,
        project=project,
        scans=scans,
        profile=profile,
    )
    preparation = replace(preparation, profile=profile)
    source_trials = source_search.trials if source_search is not None else ()
    package_result = _execute_profile_selected_package(
        _ProfileSelectedPackageContext(
            options=options,
            project=project,
            scans=scans,
            baseline=baseline,
            preparation=preparation,
            typed_regions=typed_regions,
            preflight_selected=preflight_selected,
            profile=profile,
            execution_plans=execution_plans,
            fusion_plans=fusion_plans,
            scalar_analyses=scalar_analyses,
            call_chain_analyses=call_chain_analyses,
            buffer_analyses=buffer_analyses,
            automatic_benchmark=automatic_benchmark,
        )
    )
    return _with_source_optimization(
        replace(
            package_result,
            scalar_analyses=scalar_analyses,
            call_chain_analyses=call_chain_analyses,
            buffer_analyses=buffer_analyses,
        ),
        source_optimization,
        source_trials,
    )


def _execute_profile_selected_package(
    context: _ProfileSelectedPackageContext,
) -> PackageCommandResult:
    """Route supported native selections or a measured automatic no-op.

    Explicit selections remain hard contracts: unsupported requested members
    fail before compilation. Automatic benchmark selection instead runs the
    complete semantic and performance gate when no backend-supported hot root
    remains, producing honest ``not-profitable`` evidence without a wheel.

    Args:
        context: Static, profile, baseline, and optimizer evidence for routing.

    Returns:
        PackageCommandResult: Native package result, explicit selection failure,
            or measured unoptimized result.
    """
    profile = context.profile
    options = context.options
    project = context.project
    profile_members = (
        profile.selected_symbols if profile is not None and profile.status == "profiled" else ()
    )
    selection_members = options.selected_members or profile_members
    if options.selected_members:
        selected = context.preflight_selected
    elif profile is not None and profile.status == "profiled":
        selected = (
            _selected_typed_regions(
                context.scans,
                project.config.compile.backends,
                selection_members,
                hot_members=profile_members,
            )
            if profile_members
            else ()
        )
    else:
        selected = context.preflight_selected
    _progress_compile_selection(
        options.progress,
        selected,
        requested_members=selection_members,
    )
    missing = _missing_requested_members(options.selected_members, selected)
    selection_error = _region_selection_error(
        profile=profile,
        selection_members=selection_members,
        profile_members=profile_members,
        selected=selected,
        missing=missing,
    )
    if context.automatic_benchmark and not selected:
        return _execute_execution_plan_only_package(
            _ExecutionPlanOnlyContext(
                options=options,
                project=project,
                typed_regions=context.typed_regions,
                execution_plans=context.execution_plans,
                prepared_baseline=context.baseline,
                profile=profile,
            )
        )
    if selection_error is not None:
        return _failed_region_selection(
            options=options,
            project=project,
            preparation=context.preparation,
            error=selection_error,
            typed_regions=context.typed_regions,
        )
    return _execute_typed_region_package(
        options=options,
        project=project,
        context=_TypedRegionPackageContext(
            selected=selected,
            typed_regions=context.typed_regions,
            preflight_skipped=(),
            native_readiness=(),
            execution_plans=context.execution_plans,
            fusion_plans=context.fusion_plans,
            scalar_analyses=context.scalar_analyses,
            call_chain_analyses=context.call_chain_analyses,
            buffer_analyses=context.buffer_analyses,
        ),
        prepared_baseline=context.baseline,
        profile=profile,
    )


def _run_source_optimization_trials(
    *,
    options: PackageOptions,
    project: DiscoveredProject,
    baseline: _BaselineWheelPayload | None,
    planning: SourceOptimizationPlanningResult,
) -> SourceOptimizationSearchResult | None:
    """Run source trials only when profiling prepared every required root.

    Args:
        options: Public package options and progress callback.
        project: Discovered target project and compile policy.
        baseline: Prepared baseline wheel payload from profile setup.
        planning: Ranked source plans and current-invocation assessments.

    Returns:
        SourceOptimizationSearchResult | None: Search evidence, or `None` when
        source optimization is not configured for this package invocation.
    """
    config = project.config.compile
    if (
        options.selected_members
        or not options.run_quality_gates
        or config.test_command is None
        or config.benchmark_command is None
    ):
        return None
    if (
        baseline is None
        or baseline.baseline_install_root is None
        or baseline.quality_project_root is None
    ):
        return None
    output_dir = _resolve_output_dir(project.config.root, options.output_dir)
    cache_root = options.cache_dir or project.config.cache_dir
    return run_source_optimization_search(
        planning.plans,
        planning.assessments,
        SourceOptimizationSearchOptions(
            project_root=project.config.root,
            source_roots=project.config.source_roots,
            module_paths=_project_relative_module_paths(project),
            output_dir=output_dir,
            scratch_root=output_dir / "build" / "source-optimization-search",
            cache_root=cache_root / "source-optimization",
            baseline_payload_root=baseline.baseline_install_root,
            quality_project_root=baseline.quality_project_root,
            compile_config=config,
            baseline_build=baseline.build,
            apply_source=options.apply_source,
            candidate_profiler=partial(_profile_source_candidate, options),
            progress=options.progress,
        ),
    )


def _profile_source_candidate(
    options: PackageOptions,
    source_project_root: Path,
    quality_project_root: Path,
    payload_root: Path,
    candidate_id: str,
) -> ProfileResult:
    """Collect a fresh optimized profile before later candidate selection.

    Args:
        options: Original source-clean command scope and progress callback.
        source_project_root: Disposable transformed project containing candidate sources.
        quality_project_root: Source-stripped project copy containing benchmark files.
        payload_root: Candidate import payload placed first on ``PYTHONPATH``.
        candidate_id: Stable source candidate identity used for scratch isolation.

    Returns:
        ProfileResult: Fresh candidate profile with ordinary and call-chain selections.

    Raises:
        ValueError: If the transformed project no longer has a benchmark command.
    """
    active_project = discover_project(source_project_root)
    benchmark = active_project.config.compile.benchmark_command
    if benchmark is None:
        raise ValueError("transformed source candidate lost its benchmark command")
    scans = _selected_scans(
        active_project,
        options.module_name,
        options.selected_members,
    )
    call_chains = _call_chain_analyses(scans, None)
    profile = run_baseline_profile(
        benchmark,
        project_root=quality_project_root,
        payload_root=payload_root,
        module_paths=_profile_module_paths(active_project),
        scratch_dir=source_project_root.parent / f"profile-{candidate_id}",
        observation_targets=_profile_observation_symbols(scans),
        spawn_targets=execution_plan_profile_targets(scans),
        call_edge_targets=_call_chain_profile_targets(call_chains),
        enable_atoll=True,
    )
    support = _profile_candidate_support(scans, active_project.config.compile.backends)
    selected = _select_profile_with_call_chains(
        profile,
        scans,
        call_chains,
        active_project.config.compile.backends,
        support=support,
    )
    _profile_progress(options.progress, selected)
    return selected


def _source_optimization_terminal_result(
    *,
    options: PackageOptions,
    project: DiscoveredProject,
    preparation: _ProfilePreparation,
    planning: SourceOptimizationPlanningResult,
    search: SourceOptimizationSearchResult | None,
) -> PackageCommandResult | None:
    """Return only source-application failures that must stop composition.

    Args:
        options: Public package and source-application options.
        project: Discovered target project.
        preparation: Baseline profile and scheduler planning evidence.
        planning: Source plans and assessments used by the search.
        search: Search result, or `None` when source trials were not configured.

    Returns:
        PackageCommandResult | None: Explicit source-application failure, or
            `None` to continue through source/native composition.
    """
    if search is None:
        return None
    if search.accepted:
        return None
    if search.error is None and not options.apply_source:
        return None
    error = search.error or "no source patch met the required 3.0x promotion floor"
    return _source_optimization_package_failure(
        options=options,
        project=project,
        preparation=preparation,
        planning=planning,
        search=replace(search, error=error),
    )


def _source_optimization_continuation_result(
    *,
    options: PackageOptions,
    project: DiscoveredProject,
    preparation: _ProfilePreparation,
    planning: SourceOptimizationPlanningResult,
    search: SourceOptimizationSearchResult | None,
) -> PackageCommandResult | None:
    """Return a source failure or execute an accepted composable source arm.

    Args:
        options: Original package options.
        project: Immutable checkout discovery.
        preparation: Baseline profile and semantic evidence.
        planning: Source plans and assessments.
        search: Optional source-search outcome.

    Returns:
        PackageCommandResult | None: Terminal failure, composed result, or
            `None` when native packaging should continue normally.
    """
    terminal = _source_optimization_terminal_result(
        options=options,
        project=project,
        preparation=preparation,
        planning=planning,
        search=search,
    )
    if terminal is not None:
        return terminal
    if search is None or not search.accepted:
        return None
    return _execute_composed_source_arm(
        options=options,
        project=project,
        preparation=preparation,
        planning=planning,
        search=search,
    )


def _execute_composed_source_arm(
    *,
    options: PackageOptions,
    project: DiscoveredProject,
    preparation: _ProfilePreparation,
    planning: SourceOptimizationPlanningResult,
    search: SourceOptimizationSearchResult,
) -> PackageCommandResult:
    """Continue an accepted source transform through native and plan stages.

    The accepted source wheel remains the fallback arm. The transformed project
    is recreated under disposable build storage, rescanned, and offered to the
    existing native/execution-plan pipeline. Any later rejection restores and
    returns the already profitable source wheel.

    Args:
        options: Original source-clean package options.
        project: Immutable checkout discovery shown in public reports.
        preparation: Baseline profile and semantic evidence.
        planning: Source plans whose accepted trial produced `search`.
        search: Accepted source search with wheel and materialization evidence.

    Returns:
        PackageCommandResult: The composed wheel when profitable, otherwise the
            accepted source-only wheel augmented with rejected-stage evidence.
    """
    if search.wheel_path is None or search.materialization_patch is None:
        return _source_optimization_package_result(
            options=options,
            project=project,
            preparation=preparation,
            planning=planning,
            search=search,
        )
    try:
        arm = _materialize_source_optimization_arm(
            options=options,
            project=project,
            preparation=preparation,
            search=search,
        )
    except (OSError, ValueError, WheelOverlayError, zipfile.BadZipFile) as error:
        _progress(options.progress, f"source composition skipped: {error}")
        source_result = _source_optimization_package_result(
            options=options,
            project=project,
            preparation=preparation,
            planning=planning,
            search=search,
        )
        return replace(
            source_result,
            build=_append_phase_timing(
                source_result.build,
                name="source_composition",
                duration_seconds=0.0,
                detail=f"fallback retained: {error}",
            ),
        )

    active_scans = _selected_scans(
        arm.active_project,
        options.module_name,
        options.selected_members,
    )
    scalar_analyses = _scalar_analyses(active_scans, options.progress)
    call_chain_analyses = _call_chain_analyses(active_scans, options.progress)
    buffer_analyses = _buffer_analyses(active_scans, options.progress)
    active_profile = _accepted_source_profile(search) or preparation.profile
    if active_profile is not None:
        support_roots = (
            _profile_support_roots(active_profile, call_chain_analyses)
            if active_profile.status == "profiled"
            else None
        )
        active_support = _profile_candidate_support(
            active_scans,
            arm.active_project.config.compile.backends,
            roots=support_roots,
        )
        active_profile = _select_profile_with_call_chains(
            active_profile,
            active_scans,
            call_chain_analyses,
            arm.active_project.config.compile.backends,
            support=active_support,
        )
    else:
        active_support = _ProfileCandidateSupport(supported=(), rejected=())
    active_execution_plans = build_execution_plans(active_scans, active_profile)
    active_fusion_plans = (
        build_fusion_plans(active_scans, active_profile) if active_profile is not None else ()
    )
    active_profile = _stabilize_profile_compile_selection(
        _ProfileCompileSelectionScope(
            identity=(
                "accepted-source:"
                + hashlib.sha256(
                    search.materialization_patch.patch_text.encode("utf-8")
                ).hexdigest()
            ),
            support=active_support,
        ),
        options=options,
        project=arm.active_project,
        scans=active_scans,
        profile=active_profile,
    )
    typed_regions = tuple(region for scan in active_scans for region in scan.typed_regions)
    profile_members = (
        active_profile.selected_symbols
        if active_profile is not None and active_profile.status == "profiled"
        else ()
    )
    selection_members = options.selected_members or profile_members
    run_guard_plans = search.native_plans
    selected = _selected_typed_regions(
        active_scans,
        arm.active_project.config.compile.backends,
        selection_members,
        hot_members=profile_members,
    )
    planned_helpers = frozenset(
        helper for plan in run_guard_plans for helper in _run_guard_member_ids(plan)
    )
    if planned_helpers:
        selected = tuple(
            selection
            for selection in selected
            if selection.slice_root not in planned_helpers
            and not planned_helpers.intersection(selection.bound_members or ())
        )
    selected_plans = tuple(
        plan for plan in active_execution_plans if isinstance(plan, ExecutionPlan)
    )
    _progress(
        options.progress,
        (
            f"rescanned accepted source arm: {len(active_scans)} module(s), "
            f"{len(selected)} native variant(s), {len(run_guard_plans)} source-fused "
            f"guard variant(s), {len(selected_plans)} execution plan(s)"
        ),
    )
    if not selected and not selected_plans and not run_guard_plans:
        return replace(
            _source_optimization_package_result(
                options=options,
                project=project,
                preparation=preparation,
                planning=planning,
                search=search,
            ),
            scalar_analyses=scalar_analyses,
            call_chain_analyses=call_chain_analyses,
            buffer_analyses=buffer_analyses,
        )

    output_dir = _resolve_output_dir(project.config.root, options.output_dir)
    composed_options = replace(
        options,
        output_dir=output_dir,
        cache_dir=options.cache_dir or project.config.cache_dir,
        apply_source=False,
    )
    with tempfile.TemporaryDirectory(prefix="atoll-source-arm-") as backup_dir_text:
        backup_path = Path(backup_dir_text) / search.wheel_path.name
        shutil.copy2(search.wheel_path, backup_path)
        if selected or run_guard_plans:
            candidate = _execute_typed_region_package(
                options=composed_options,
                project=arm.active_project,
                context=_TypedRegionPackageContext(
                    selected=selected,
                    typed_regions=typed_regions,
                    preflight_skipped=(),
                    native_readiness=(),
                    execution_plans=active_execution_plans,
                    fusion_plans=active_fusion_plans,
                    scalar_analyses=scalar_analyses,
                    call_chain_analyses=call_chain_analyses,
                    buffer_analyses=buffer_analyses,
                    run_guard_plans=run_guard_plans,
                ),
                prepared_baseline=arm.baseline,
                profile=active_profile,
            )
        else:
            candidate = _execute_execution_plan_only_package(
                _ExecutionPlanOnlyContext(
                    options=composed_options,
                    project=arm.active_project,
                    typed_regions=typed_regions,
                    execution_plans=active_execution_plans,
                    prepared_baseline=arm.baseline,
                    profile=active_profile,
                )
            )
        if candidate.success:
            if search.wheel_path != candidate.wheel_path and search.wheel_path.exists():
                search.wheel_path.unlink()
            rebased = _rebase_composed_result(candidate, project)
            return _with_source_optimization(
                replace(
                    rebased,
                    test_results=(*search.test_results, *rebased.test_results),
                    scalar_analyses=scalar_analyses,
                    call_chain_analyses=call_chain_analyses,
                    buffer_analyses=buffer_analyses,
                ),
                planning,
                search.trials,
            )

        search.wheel_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup_path, search.wheel_path)
        _progress(options.progress, "later optimization rejected; retained accepted source wheel")
        source_result = _source_optimization_package_result(
            options=options,
            project=project,
            preparation=preparation,
            planning=planning,
            search=search,
        )
        return replace(
            _source_result_with_composition_fallback(source_result, candidate, project),
            scalar_analyses=scalar_analyses,
            call_chain_analyses=call_chain_analyses,
            buffer_analyses=buffer_analyses,
        )


def _materialize_source_optimization_arm(
    *,
    options: PackageOptions,
    project: DiscoveredProject,
    preparation: _ProfilePreparation,
    search: SourceOptimizationSearchResult,
) -> OptimizationArm:
    """Recreate accepted transformed sources and unpack their wheel baseline.

    Args:
        options: Original package options and output override.
        project: Immutable project checkout used to form the accepted patch.
        preparation: Original baseline and quality-project evidence.
        search: Accepted source search with wheel and generated patch payload.

    Returns:
        OptimizationArm: Disposable transformed project plus accepted baseline.

    Raises:
        ValueError: If accepted source evidence is incomplete.
        OSError: If project materialization or wheel staging fails.
        WheelOverlayError: If the accepted source wheel cannot be unpacked.
    """
    if search.wheel_path is None or search.materialization_patch is None:
        raise ValueError("accepted source arm lacks wheel or materialization evidence")
    original_baseline = preparation.baseline
    if original_baseline is None:
        raise ValueError("accepted source arm lacks original baseline evidence")
    output_dir = _resolve_output_dir(project.config.root, options.output_dir)
    build_root = output_dir / "build"
    transformed_root = build_root / "accepted-source-project"
    shutil.rmtree(transformed_root, ignore_errors=True)
    _copy_pep517_project(
        project.config.root,
        transformed_root,
        excluded_output=output_dir,
    )
    materialize_transformed_files(
        project.config.root,
        transformed_root,
        search.materialization_patch,
    )
    active_project = discover_project(transformed_root)

    install_root = output_dir / "install"
    _reset_dir(install_root)
    unpack_wheel_payload(search.wheel_path, install_root)
    baseline_install_root = build_root / "accepted-source-baseline"
    _reset_dir(baseline_install_root)
    _copytree_contents(install_root, baseline_install_root)
    baseline = _BaselineWheelPayload(
        wheel_path=search.wheel_path,
        build=search.build,
        baseline_install_root=baseline_install_root,
        quality_project_root=original_baseline.quality_project_root,
        semantic_test_result=original_baseline.semantic_test_result,
    )
    return OptimizationArm(
        report_project=project,
        active_project=active_project,
        baseline=baseline,
        source_search=search,
    )


def _accepted_source_profile(search: SourceOptimizationSearchResult) -> ProfileResult | None:
    """Return the fresh transformed profile retained by the accepted source trial.

    Args:
        search: Completed source search whose winner may carry residual evidence.

    Returns:
        ProfileResult | None: Most recent accepted transformed profile, when available.
    """
    return next(
        (
            trial.residual_profile
            for trial in reversed(search.trials)
            if trial.status == "accepted" and trial.residual_profile is not None
        ),
        None,
    )


def _rebase_composed_result(
    result: PackageCommandResult,
    report_project: DiscoveredProject,
) -> PackageCommandResult:
    """Replace disposable module paths with immutable checkout identities.

    Args:
        result: Successful result produced from a transformed project copy.
        report_project: Original project whose paths belong in reports.

    Returns:
        PackageCommandResult: Result with typed-region source paths rebased.
    """
    modules = {module.name: module for module in report_project.modules}

    def rebase_region(region: TypedRegion) -> TypedRegion:
        module = modules.get(region.source_module.name)
        return replace(region, source_module=module) if module is not None else region

    return replace(
        result,
        project_root=report_project.config.root,
        typed_regions=tuple(rebase_region(region) for region in result.typed_regions),
        compiled_regions=tuple(rebase_region(region) for region in result.compiled_regions),
        compiled_variants=tuple(
            replace(variant, region=rebase_region(variant.region))
            for variant in result.compiled_variants
        ),
        region_skipped=tuple(
            replace(failure, region=rebase_region(failure.region))
            for failure in result.region_skipped
        ),
    )


def _source_result_with_composition_fallback(
    source_result: PackageCommandResult,
    rejected: PackageCommandResult,
    report_project: DiscoveredProject,
) -> PackageCommandResult:
    """Attach rejected later-stage evidence to a successful source fallback.

    Args:
        source_result: Accepted source-only result retained for promotion.
        rejected: Native or execution-plan result that failed later gates.
        report_project: Original project used to rebase transformed paths.

    Returns:
        PackageCommandResult: Successful source result with auditable rejection evidence.
    """
    rebased = _rebase_composed_result(rejected, report_project)
    rejected_detail = rejected.error or rejected.build.stderr or "later stage rejected"
    build = replace(
        source_result.build,
        stdout="\n".join(
            part
            for part in (
                source_result.build.stdout,
                f"composition fallback retained: {rejected_detail}",
                rejected.build.stdout,
            )
            if part
        ),
        duration_seconds=(source_result.build.duration_seconds + rejected.build.duration_seconds),
        phase_timings=(*source_result.build.phase_timings, *rejected.build.phase_timings),
    )
    return replace(
        source_result,
        build=build,
        typed_regions=rebased.typed_regions,
        backend_assessments=rebased.backend_assessments,
        region_skipped=rebased.region_skipped,
        candidate_trials=rebased.candidate_trials,
        execution_plans=rebased.execution_plans,
        execution_plan_trials=rebased.execution_plan_trials,
        fusion_plans=rebased.fusion_plans,
        fusion_trials=rebased.fusion_trials,
        test_results=(*source_result.test_results, *rebased.test_results),
    )


def _project_relative_module_paths(project: DiscoveredProject) -> tuple[Path, ...]:
    paths: list[Path] = []
    for module in project.modules:
        try:
            paths.append(module.path.relative_to(project.config.root))
        except ValueError:
            continue
    return tuple(paths)


def _source_optimization_package_result(
    *,
    options: PackageOptions,
    project: DiscoveredProject,
    preparation: _ProfilePreparation,
    planning: SourceOptimizationPlanningResult,
    search: SourceOptimizationSearchResult,
) -> PackageCommandResult:
    """Normalize an accepted transformed source wheel into package evidence.

    Args:
        options: Package output and debug retention policy.
        project: Discovered target project.
        preparation: Baseline profile, execution-plan, and fusion evidence.
        planning: Source plans and assessments used by the search.
        search: Accepted source and wheel gate result.

    Returns:
        PackageCommandResult: Successful pure or project-native normal wheel result.
    """
    output_dir, build_root, install_root = _source_clean_output_paths(
        project.config.root,
        options.output_dir,
    )
    cleanup_removed: list[Path] = []
    cleanup_kept: tuple[Path, ...] = ()
    if options.keep_install_tree and search.wheel_path is not None:
        _reset_dir(install_root)
        unpack_wheel_payload(search.wheel_path, install_root)
        cleanup_removed.extend(_remove_tree(build_root))
        cleanup_kept = (install_root,)
    else:
        cleanup_removed.extend(_remove_source_clean_scratch(build_root, install_root))
    return PackageCommandResult(
        success=True,
        project_root=project.config.root,
        output_dir=output_dir,
        install_root=install_root,
        wheel_path=search.wheel_path,
        islands=(),
        build=search.build,
        install_tree_kept=bool(cleanup_kept),
        cleanup_removed=tuple(cleanup_removed),
        cleanup_kept=cleanup_kept,
        test_results=search.test_results,
        performance=search.performance,
        profile=preparation.profile,
        execution_plans=preparation.execution_plans,
        fusion_plans=preparation.fusion_plans,
        source_optimization_plans=planning.plans,
        source_optimization_assessments=planning.assessments,
        source_optimization_trials=search.trials,
    )


def _source_optimization_package_failure(
    *,
    options: PackageOptions,
    project: DiscoveredProject,
    preparation: _ProfilePreparation,
    planning: SourceOptimizationPlanningResult,
    search: SourceOptimizationSearchResult,
) -> PackageCommandResult:
    """Return a failed requested application without leaving wheel scratch.

    Args:
        options: Package output policy.
        project: Discovered target project.
        preparation: Baseline profile and scheduler evidence.
        planning: Source plans and assessments used by the search.
        search: Rejected or transactionally rolled-back source search.

    Returns:
        PackageCommandResult: Failed result retaining source trial evidence.
    """
    error = search.error or "source optimization failed"
    output_dir = _resolve_output_dir(project.config.root, options.output_dir)
    _remove_failed_wheels(project, output_dir)
    cleanup_removed = _remove_source_clean_scratch(
        output_dir / "build",
        output_dir / "install",
    )
    return PackageCommandResult(
        success=False,
        project_root=project.config.root,
        output_dir=output_dir,
        install_root=output_dir / "install",
        wheel_path=None,
        islands=(),
        build=replace(search.build, success=False, stderr=error),
        cleanup_removed=cleanup_removed,
        error=error,
        test_results=search.test_results,
        performance=search.performance,
        profile=preparation.profile,
        execution_plans=preparation.execution_plans,
        fusion_plans=preparation.fusion_plans,
        source_optimization_plans=planning.plans,
        source_optimization_assessments=planning.assessments,
        source_optimization_trials=search.trials,
    )


def _execute_execution_plan_only_package(
    context: _ExecutionPlanOnlyContext,
) -> PackageCommandResult:
    """Promote a profitable Python-overlay plan without a native region.

    The normal PEP 517 wheel remains the base payload. Selected scheduler plans
    stage only against its temporary unpacked copy, then pass the same isolated
    payload/wheel verification and final semantic/performance gate as native
    variants. A profile-selected plan that fails every backend trial cannot
    publish an unchanged baseline wheel.

    Args:
        context: Baseline, profile, project, and selected plan state.

    Returns:
        PackageCommandResult: Plan trials, promotion evidence, and optional pure wheel.
    """
    options = context.options
    project = context.project
    output_dir = _resolve_output_dir(project.config.root, options.output_dir)
    build_root = output_dir / "build"
    install_root = output_dir / "install"
    baseline = _package_baseline(options, project, context.prepared_baseline)
    if baseline.wheel_path is None:
        _remove_failed_wheels(project, output_dir)
        cleanup_removed = _remove_source_clean_scratch(build_root, install_root)
        return PackageCommandResult(
            success=False,
            project_root=project.config.root,
            output_dir=output_dir,
            install_root=install_root,
            wheel_path=None,
            islands=(),
            build=baseline.build,
            cleanup_removed=cleanup_removed,
            cleanup_kept=(),
            error=baseline.build.stderr,
            typed_regions=context.typed_regions,
            profile=context.profile,
            execution_plans=context.execution_plans,
        )

    plan_application = _apply_execution_plan_trials(
        _ExecutionPlanApplicationContext(
            options=options,
            project=project,
            baseline=baseline,
            build_root=build_root,
            install_root=install_root,
            plans=context.execution_plans,
            accepted_region_ids=frozenset(),
        )
    )
    promotion = _promote_source_clean_payload(
        _SourceCleanPromotionContext(
            options=options,
            project=project,
            output_dir=output_dir,
            build_root=build_root,
            install_root=install_root,
            baseline=baseline,
            verification_plan=PackageVerificationPlan(
                modules=(),
                regions=(),
                artifacts=(),
            ),
            build=_append_phase_timings(baseline.build, plan_application.timings),
            requires_profitable_optimization=True,
            profitable_optimization_applied=bool(plan_application.applied_plan_ids),
        )
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
        error=promotion.error,
        typed_regions=context.typed_regions,
        verification_steps=promotion.verification_steps,
        test_results=promotion.test_results,
        performance=promotion.performance,
        profile=context.profile,
        execution_plans=context.execution_plans,
        applied_execution_plans=plan_application.applied_plan_ids,
        execution_plan_trials=plan_application.trials,
    )


def _with_source_optimization(
    result: PackageCommandResult,
    planning: SourceOptimizationPlanningResult,
    trials: tuple[SourceOptimizationTrial, ...] = (),
) -> PackageCommandResult:
    """Attach source plans and disposable trials without changing package success.

    Args:
        result: Existing native or execution-plan package result.
        planning: Source plans and 3x gate assessments derived before packaging.
        trials: Bounded source candidate search evidence.

    Returns:
        PackageCommandResult: Result augmented only with source-optimization report evidence.
    """
    return replace(
        result,
        source_optimization_plans=planning.plans,
        source_optimization_assessments=planning.assessments,
        source_optimization_trials=trials,
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
    output_dir, build_root, install_root = _source_clean_output_paths(
        project.config.root,
        options.output_dir,
    )
    baseline = _package_baseline(options, project, prepared_baseline)
    if baseline.wheel_path is None:
        _remove_failed_wheels(project, output_dir)
        cleanup_removed = _remove_source_clean_scratch(build_root, install_root)
        return PackageCommandResult(
            success=False,
            project_root=project.config.root,
            output_dir=output_dir,
            install_root=install_root,
            wheel_path=None,
            islands=(),
            build=baseline.build,
            cleanup_removed=cleanup_removed,
            cleanup_kept=(),
            error=baseline.build.stderr,
            preflight_skipped=context.preflight_skipped,
            native_readiness=context.native_readiness,
            typed_regions=context.typed_regions,
            backend_assessments=tuple(selection.assessment for selection in context.selected),
            profile=profile,
            execution_plans=context.execution_plans,
            fusion_plans=context.fusion_plans,
        )

    staged_source_roots, source_tree_digest = _stage_target_sources(
        project,
        build_root,
        options.progress,
    )
    generation_started = time.perf_counter()
    prepared, preparation_failures = _prepare_selected_variants(
        project=project,
        build_root=build_root,
        staged_source_roots=staged_source_roots,
        selected=context.selected,
    )
    prepared, preparation_failures, scalar_variant_count = _extend_with_scalar_variants(
        context=_ScalarExtensionContext(
            project=project,
            build_root=build_root,
            staged_source_roots=staged_source_roots,
            analyses=context.scalar_analyses,
            progress=options.progress,
        ),
        prepared=prepared,
        failures=preparation_failures,
    )
    prepared, preparation_failures, call_chain_variant_count = _extend_with_call_chain_variants(
        context=_CallChainExtensionContext(
            project=project,
            build_root=build_root,
            staged_source_roots=staged_source_roots,
            analyses=context.call_chain_analyses,
            progress=options.progress,
        ),
        prepared=prepared,
        failures=preparation_failures,
    )
    prepared, preparation_failures, buffer_variant_count = _extend_with_buffer_variants(
        context=_BufferExtensionContext(
            project=project,
            build_root=build_root,
            staged_source_roots=staged_source_roots,
            analyses=context.buffer_analyses,
            progress=options.progress,
        ),
        prepared=prepared,
        failures=preparation_failures,
    )
    prepared, preparation_failures, run_guard_variant_count = _extend_with_run_guard_variants(
        context=_RunGuardExtensionContext(
            project=project,
            build_root=build_root,
            staged_source_roots=staged_source_roots,
            plans=context.run_guard_plans,
            progress=options.progress,
        ),
        prepared=prepared,
        failures=preparation_failures,
    )
    wheel_owned_prepared, wheel_omissions = _partition_wheel_owned_variants(
        tuple(prepared),
        staged_source_roots=staged_source_roots,
        install_root=install_root,
    )
    preparation_failures.extend(wheel_omissions)
    if wheel_omissions:
        _progress(
            options.progress,
            (
                f"skipped {len(wheel_omissions)} native variant(s) whose source modules "
                "are not shipped by the target PEP 517 wheel"
            ),
        )
    _progress(
        options.progress,
        (
            f"lowered {len(wheel_owned_prepared)} native region variant(s), "
            f"including {scalar_variant_count} scalar and "
            f"{call_chain_variant_count} direct call-chain and "
            f"{buffer_variant_count} zero-copy buffer variant(s); "
            f"{run_guard_variant_count} source-fused guard variant(s); "
            f"kept {len(preparation_failures)} as fallback in {_duration(generation_started)}"
        ),
    )
    if not wheel_owned_prepared:
        return _failed_preparation_result(
            _PreparationFailureContext(
                options=options,
                project=project,
                package=context,
                profile=profile,
                failures=tuple(preparation_failures),
                wheel_omissions=wheel_omissions,
            )
        )

    prepared_assessments = _prepared_backend_assessments(wheel_owned_prepared)
    outcome = _build_typed_regions(
        prepared=wheel_owned_prepared,
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
            source_tree_digest=source_tree_digest,
            enable_project_circuit=(options.module_name is None and not options.selected_members),
        ),
        initial_failures=tuple(preparation_failures),
    )
    outcome = replace(outcome, build=_combine_baseline_and_native(baseline.build, outcome.build))
    if not outcome.successful:
        _progress(options.progress, "all typed-region builds failed; cleaning temporary outputs")
        _remove_failed_wheels(project, output_dir)
        cleanup_removed = _remove_source_clean_scratch(build_root, install_root)
        return PackageCommandResult(
            success=False,
            project_root=project.config.root,
            output_dir=output_dir,
            install_root=install_root,
            wheel_path=None,
            islands=(),
            build=outcome.build,
            cleanup_removed=cleanup_removed,
            cleanup_kept=(),
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

    successful_bindings = _deduplicated_public_bindings(outcome.successful)
    compiled_sources = frozenset(binding.source for binding in successful_bindings)
    missing_compiled = tuple(
        member for member in options.selected_members if member not in compiled_sources
    )
    if missing_compiled:
        missing_text = ", ".join(member.stable_id for member in missing_compiled)
        error = f"requested member(s) did not compile successfully: {missing_text}"
        failed_build = replace(outcome.build, success=False, stderr=error)
        _remove_failed_wheels(project, output_dir)
        cleanup_removed = _remove_source_clean_scratch(build_root, install_root)
        return PackageCommandResult(
            success=False,
            project_root=project.config.root,
            output_dir=output_dir,
            install_root=install_root,
            wheel_path=None,
            islands=(),
            build=failed_build,
            cleanup_removed=cleanup_removed,
            cleanup_kept=(),
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
        wheel_owned_prepared,
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
    finalized, outcome, safety = _apply_runtime_safety_selection(
        context=_TypedPayloadFinalizationContext(
            options=options,
            project=project,
            profile=profile,
            baseline=baseline,
            install_root=install_root,
            staged_source_roots=staged_source_roots,
            outcome=outcome,
            overlay_error=finalized.overlay_error,
        ),
        finalized=finalized,
        selected_members=options.selected_members,
    )
    accepted_successful, plan_application = (
        finalized.successful,
        _execution_plan_application_for_finalized_payload(
            _ExecutionPlanApplicationContext(
                options=options,
                project=project,
                baseline=baseline,
                build_root=build_root,
                install_root=install_root,
                plans=context.execution_plans,
                accepted_region_ids=frozenset(item.unit.region_id for item in finalized.successful),
            ),
            finalized=finalized,
        ),
    )
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
        build=_append_phase_timings(finalized.build, plan_application.timings),
        requires_profitable_optimization=finalized.profitability_applied,
        profitable_optimization_applied=bool(
            accepted_artifacts or plan_application.applied_plan_ids
        ),
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
    promotion = replace(
        promotion,
        verification_steps=(*safety.verification_steps, *promotion.verification_steps),
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
    successful_bindings = _deduplicated_public_bindings(accepted_successful)
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
        scalar_analyses=context.scalar_analyses,
        call_chain_analyses=context.call_chain_analyses,
        buffer_analyses=context.buffer_analyses,
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
        applied_execution_plans=plan_application.applied_plan_ids,
        execution_plan_trials=plan_application.trials,
        fusion_plans=context.fusion_plans,
        fusion_trials=fusion_research.trials,
    )


def _failed_preparation_result(
    context: _PreparationFailureContext,
) -> PackageCommandResult:
    """Report that every selected native variant remained interpreted.

    Args:
        context: Lowering failures and source-clean package boundaries.

    Returns:
        PackageCommandResult: Failed result with scratch output removed and
        every rejected variant retained as report evidence.
    """
    output_dir, build_root, install_root = _source_clean_output_paths(
        context.project.config.root,
        context.options.output_dir,
    )
    error = (
        context.wheel_omissions[0].build.stderr
        if context.wheel_omissions
        else "No selected typed regions could be lowered for a compiler backend."
    )
    _remove_failed_wheels(context.project, output_dir)
    cleanup_removed = _remove_source_clean_scratch(build_root, install_root)
    return PackageCommandResult(
        success=False,
        project_root=context.project.config.root,
        output_dir=output_dir,
        install_root=install_root,
        wheel_path=None,
        islands=(),
        build=_failed_region_attempt(error),
        cleanup_removed=cleanup_removed,
        cleanup_kept=(),
        error=error,
        preflight_skipped=context.package.preflight_skipped,
        native_readiness=context.package.native_readiness,
        typed_regions=context.package.typed_regions,
        backend_assessments=tuple(selection.assessment for selection in context.package.selected),
        region_skipped=context.failures,
        profile=context.profile,
        execution_plans=context.package.execution_plans,
        fusion_plans=context.package.fusion_plans,
    )


def _prepare_selected_variants(
    *,
    project: DiscoveredProject,
    build_root: Path,
    staged_source_roots: tuple[Path, ...],
    selected: tuple[_SelectedTypedRegion, ...],
) -> tuple[list[_PreparedTypedRegion], list[PackageRegionBuildFailure]]:
    """Lower generic backend selections while retaining explicit failures.

    Args:
        project: Target project discovery.
        build_root: Disposable native build root.
        staged_source_roots: Copied import roots.
        selected: Generic typed-region backend selections.

    Returns:
        tuple[list[_PreparedTypedRegion], list[PackageRegionBuildFailure]]:
        Prepared variants and deterministic lowering failures.
    """
    prepared: list[_PreparedTypedRegion] = []
    failures: list[PackageRegionBuildFailure] = []
    for selection in selected:
        try:
            prepared.append(
                _prepare_typed_region(
                    project=project,
                    build_root=build_root,
                    staged_source_roots=staged_source_roots,
                    selection=selection,
                )
            )
        except (SyntaxError, ValueError) as error:
            failures.append(
                PackageRegionBuildFailure(
                    region=selection.region,
                    variant_id=selection.variant_id,
                    backend=selection.backend,
                    assessment=selection.assessment,
                    build=_failed_region_attempt(f"typed-region lowering failed: {error}"),
                )
            )
    return prepared, failures


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


def _prepare_scalar_variants(
    *,
    project: DiscoveredProject,
    build_root: Path,
    staged_source_roots: tuple[Path, ...],
    analyses: tuple[ScalarAnalysisResult, ...],
) -> tuple[tuple[_PreparedTypedRegion, ...], tuple[PackageRegionBuildFailure, ...]]:
    """Revalidate and lower fixed-width plans inside copied source roots.

    Args:
        project: Original project discovery used to map staged module paths.
        build_root: Disposable source-clean build root.
        staged_source_roots: Copied import roots that remain source-clean.
        analyses: Checkout or accepted-source scalar proof evidence.

    Returns:
        tuple[tuple[_PreparedTypedRegion, ...], tuple[PackageRegionBuildFailure, ...]]:
        Prepared 32/64-bit variants followed by explicit lowering failures.
    """
    plans = tuple(plan for analysis in analyses for plan in analysis.plans)
    scans: dict[str, ModuleScan] = {}
    staged_analyses: dict[str, ScalarAnalysisResult] = {}
    prepared: list[_PreparedTypedRegion] = []
    failures: list[PackageRegionBuildFailure] = []
    for plan in plans:
        module = _find_module(project.modules, plan.member.module)
        staged_scan = scans.get(plan.member.module)
        if staged_scan is None:
            staged_module = _staged_module(module, project, staged_source_roots)
            staged_scan = enrich_island_analysis(scan_module(staged_module))
            scans[plan.member.module] = staged_scan
            staged_analyses[plan.member.module] = analyze_scalar_scan(staged_scan)
        staged_region = _scalar_region_for_plan(staged_scan, plan)
        staged_plan = next(
            (
                candidate
                for candidate in staged_analyses[plan.member.module].plans
                if candidate.id == plan.id and candidate.member == plan.member
            ),
            None,
        )
        if staged_plan is None:
            failures.append(
                PackageRegionBuildFailure(
                    region=staged_region,
                    variant_id=f"{plan.id}@cython-scalar",
                    backend="cython",
                    assessment=CYTHON_BACKEND.assess(staged_region),
                    build=_failed_region_attempt(
                        "scalar lowering failed: staged proof differs from checkout analysis"
                    ),
                )
            )
            continue
        for proof in staged_plan.width_proofs:
            variant_id = _scalar_variant_id(staged_plan, proof)
            try:
                prepared.append(
                    _prepare_scalar_variant(
                        context=_ScalarVariantContext(
                            project=project,
                            build_root=build_root,
                            staged_source_roots=staged_source_roots,
                            scan=staged_scan,
                            region=staged_region,
                        ),
                        plan=staged_plan,
                        variant_id=variant_id,
                        proof=proof,
                    )
                )
            except (SyntaxError, ValueError) as error:
                failures.append(
                    PackageRegionBuildFailure(
                        region=staged_region,
                        variant_id=variant_id,
                        backend="cython",
                        assessment=CYTHON_BACKEND.assess(staged_region),
                        build=_failed_region_attempt(f"scalar lowering failed: {error}"),
                    )
                )
    return tuple(prepared), tuple(failures)


def _extend_with_scalar_variants(
    *,
    context: _ScalarExtensionContext,
    prepared: list[_PreparedTypedRegion],
    failures: list[PackageRegionBuildFailure],
) -> tuple[list[_PreparedTypedRegion], list[PackageRegionBuildFailure], int]:
    """Prepend available scalar variants without changing generic fallback.

    Args:
        context: Target project, copied roots, analyses, and progress callback.
        prepared: Existing generic native variants.
        failures: Existing generic preparation failures.

    Returns:
        tuple[list[_PreparedTypedRegion], list[PackageRegionBuildFailure], int]:
        Updated variants, failures, and prepared scalar variant count.
    """
    if "cython" not in context.project.config.compile.backends:
        if any(analysis.plans for analysis in context.analyses):
            _progress(
                context.progress,
                "scalar native variants skipped because Cython is disabled",
            )
        return prepared, failures, 0
    selected_sources = {
        binding.source
        for item in prepared
        for binding in item.generation.bindings
        if binding.kind != "class"
    }
    selected_sources.update(
        member.id
        for failure in failures
        for member in failure.region.members
        if member.kind in {"function", "method"}
    )
    selected_analyses = tuple(
        replace(
            analysis,
            plans=tuple(plan for plan in analysis.plans if plan.member in selected_sources),
        )
        for analysis in context.analyses
    )
    scalar_prepared, scalar_failures = _prepare_scalar_variants(
        project=context.project,
        build_root=context.build_root,
        staged_source_roots=context.staged_source_roots,
        analyses=selected_analyses,
    )
    return (
        _merge_specialized_companions(prepared, scalar_prepared),
        [*failures, *scalar_failures],
        len(scalar_prepared),
    )


def _merge_specialized_companions(
    generic: list[_PreparedTypedRegion],
    specialized: tuple[_PreparedTypedRegion, ...],
) -> list[_PreparedTypedRegion]:
    """Place specialized variants beside the selected generic public binding.

    Atomic class variants retain their existing first position. Scalar method
    companions inherit the method fallback's class-failure condition, while
    module functions try int32, int64, then their generic target.

    Args:
        generic: Existing generic variants in selection priority order.
        specialized: Width or call-chain variants in frontend dispatch order.

    Returns:
        list[_PreparedTypedRegion]: Composable variant order used for build and trials.
    """
    by_source: dict[SymbolId, list[_PreparedTypedRegion]] = {}
    for item in specialized:
        source = item.generation.bindings[0].source
        by_source.setdefault(source, []).append(item)
    merged: list[_PreparedTypedRegion] = []
    for item in generic:
        sources = tuple(
            binding.source for binding in item.generation.bindings if binding.kind != "class"
        )
        for source in sources:
            merged.extend(
                replace(companion, conditional_on_failure_of=item.conditional_on_failure_of)
                for companion in by_source.pop(source, ())
            )
        merged.append(item)
    merged.extend(companion for companions in by_source.values() for companion in companions)
    return merged


def _extend_with_call_chain_variants(
    *,
    context: _CallChainExtensionContext,
    prepared: list[_PreparedTypedRegion],
    failures: list[PackageRegionBuildFailure],
) -> tuple[list[_PreparedTypedRegion], list[PackageRegionBuildFailure], int]:
    """Prepend direct call-chain variants without changing generic fallbacks.

    Args:
        context: Target project, staged roots, call-chain analyses, and progress callback.
        prepared: Existing generic and scalar native variants.
        failures: Existing preparation failures.

    Returns:
        tuple[list[_PreparedTypedRegion], list[PackageRegionBuildFailure], int]:
        Updated variants, failures, and prepared call-chain variant count.
    """
    if "cython" not in context.project.config.compile.backends:
        if any(analysis.plans for analysis in context.analyses):
            _progress(
                context.progress,
                "direct call-chain variants skipped because Cython is disabled",
            )
        return prepared, failures, 0
    selected_sources = {
        binding.source
        for item in prepared
        for binding in item.generation.bindings
        if binding.kind != "class"
    }
    selected_analyses = tuple(
        replace(
            analysis,
            plans=tuple(plan for plan in analysis.plans if plan.root in selected_sources),
        )
        for analysis in context.analyses
    )
    chain_prepared, chain_failures = _prepare_call_chain_variants(
        project=context.project,
        build_root=context.build_root,
        staged_source_roots=context.staged_source_roots,
        analyses=selected_analyses,
    )
    return (
        _merge_specialized_companions(prepared, chain_prepared),
        [*failures, *chain_failures],
        len(chain_prepared),
    )


def _prepare_call_chain_variants(
    *,
    project: DiscoveredProject,
    build_root: Path,
    staged_source_roots: tuple[Path, ...],
    analyses: tuple[CallChainAnalysisResult, ...],
) -> tuple[tuple[_PreparedTypedRegion, ...], tuple[PackageRegionBuildFailure, ...]]:
    """Revalidate and lower direct call-chain plans inside copied source roots.

    Args:
        project: Original project discovery used to map staged modules.
        build_root: Disposable source-clean native build root.
        staged_source_roots: Copied import roots used during lowering.
        analyses: Checkout or accepted-source call-chain proof evidence.

    Returns:
        tuple[tuple[_PreparedTypedRegion, ...], tuple[PackageRegionBuildFailure, ...]]:
        Prepared width variants and explicit lowering failures.
    """
    plans = tuple(plan for analysis in analyses for plan in analysis.plans)
    scans: dict[str, ModuleScan] = {}
    staged_analyses: dict[str, CallChainAnalysisResult] = {}
    prepared: list[_PreparedTypedRegion] = []
    failures: list[PackageRegionBuildFailure] = []
    for plan in plans:
        module = _find_module(project.modules, plan.root.module)
        staged_scan = scans.get(plan.root.module)
        if staged_scan is None:
            staged_module = _staged_module(module, project, staged_source_roots)
            staged_scan = enrich_island_analysis(scan_module(staged_module))
            scans[plan.root.module] = staged_scan
            staged_analyses[plan.root.module] = analyze_call_chain_scan(staged_scan)
        staged_region = _call_chain_region_for_plan(staged_scan, plan)
        staged_plan = next(
            (
                candidate
                for candidate in staged_analyses[plan.root.module].plans
                if candidate.id == plan.id and candidate.root == plan.root
            ),
            None,
        )
        if staged_plan is None:
            failures.append(
                PackageRegionBuildFailure(
                    region=staged_region,
                    variant_id=f"{plan.id}@cython-call-chain",
                    backend="cython",
                    assessment=CYTHON_BACKEND.assess(staged_region),
                    build=_failed_region_attempt(
                        "call-chain lowering failed: staged proof differs from checkout analysis"
                    ),
                )
            )
            continue
        for proof in staged_plan.scalar_plan.width_proofs:
            variant_id = _call_chain_variant_id(staged_plan, proof)
            try:
                prepared.append(
                    _prepare_call_chain_variant(
                        context=_ScalarVariantContext(
                            project=project,
                            build_root=build_root,
                            staged_source_roots=staged_source_roots,
                            scan=staged_scan,
                            region=staged_region,
                        ),
                        plan=staged_plan,
                        variant_id=variant_id,
                        proof=proof,
                    )
                )
            except (SyntaxError, ValueError) as error:
                failures.append(
                    PackageRegionBuildFailure(
                        region=staged_region,
                        variant_id=variant_id,
                        backend="cython",
                        assessment=CYTHON_BACKEND.assess(staged_region),
                        build=_failed_region_attempt(f"call-chain lowering failed: {error}"),
                    )
                )
    return tuple(prepared), tuple(failures)


def _extend_with_buffer_variants(
    *,
    context: _BufferExtensionContext,
    prepared: list[_PreparedTypedRegion],
    failures: list[PackageRegionBuildFailure],
) -> tuple[list[_PreparedTypedRegion], list[PackageRegionBuildFailure], int]:
    """Prepend guarded zero-copy buffer variants without changing fallbacks.

    Args:
        context: Target project, copied roots, buffer analyses, and progress callback.
        prepared: Existing generic and specialized native variants.
        failures: Existing preparation failures.

    Returns:
        tuple[list[_PreparedTypedRegion], list[PackageRegionBuildFailure], int]:
        Updated variants, failures, and prepared buffer variant count.
    """
    if "cython" not in context.project.config.compile.backends:
        if any(analysis.plans for analysis in context.analyses):
            _progress(
                context.progress,
                "zero-copy buffer variants skipped because Cython is disabled",
            )
        return prepared, failures, 0
    selected_sources = {
        binding.source
        for item in prepared
        for binding in item.generation.bindings
        if binding.kind != "class"
    }
    selected_sources.update(
        member.id
        for failure in failures
        for member in failure.region.members
        if member.kind in {"function", "method"}
    )
    selected_analyses = tuple(
        replace(
            analysis,
            plans=tuple(plan for plan in analysis.plans if plan.member in selected_sources),
        )
        for analysis in context.analyses
    )
    buffer_prepared, buffer_failures = _prepare_buffer_variants(
        project=context.project,
        build_root=context.build_root,
        staged_source_roots=context.staged_source_roots,
        analyses=selected_analyses,
    )
    return (
        _merge_specialized_companions(prepared, buffer_prepared),
        [*failures, *buffer_failures],
        len(buffer_prepared),
    )


def _prepare_buffer_variants(
    *,
    project: DiscoveredProject,
    build_root: Path,
    staged_source_roots: tuple[Path, ...],
    analyses: tuple[BufferAnalysisResult, ...],
) -> tuple[tuple[_PreparedTypedRegion, ...], tuple[PackageRegionBuildFailure, ...]]:
    """Revalidate and lower zero-copy plans inside copied source roots.

    Args:
        project: Original project discovery used to map staged modules.
        build_root: Disposable source-clean native build root.
        staged_source_roots: Copied import roots used during lowering.
        analyses: Checkout or accepted-source buffer proof evidence.

    Returns:
        tuple[tuple[_PreparedTypedRegion, ...], tuple[PackageRegionBuildFailure, ...]]:
        Prepared buffer variants and explicit lowering failures.
    """
    plans = tuple(plan for analysis in analyses for plan in analysis.plans)
    scans: dict[str, ModuleScan] = {}
    staged_analyses: dict[str, BufferAnalysisResult] = {}
    prepared: list[_PreparedTypedRegion] = []
    failures: list[PackageRegionBuildFailure] = []
    for plan in plans:
        module = _find_module(project.modules, plan.member.module)
        staged_scan = scans.get(plan.member.module)
        if staged_scan is None:
            staged_module = _staged_module(module, project, staged_source_roots)
            staged_scan = enrich_island_analysis(scan_module(staged_module))
            scans[plan.member.module] = staged_scan
            staged_analyses[plan.member.module] = analyze_buffer_scan(staged_scan)
        staged_region = _buffer_region_for_plan(staged_scan, plan)
        staged_plan = next(
            (
                candidate
                for candidate in staged_analyses[plan.member.module].plans
                if candidate.id == plan.id and candidate.member == plan.member
            ),
            None,
        )
        variant_id = _buffer_variant_id(plan)
        if staged_plan is None:
            failures.append(
                PackageRegionBuildFailure(
                    region=staged_region,
                    variant_id=variant_id,
                    backend="cython",
                    assessment=CYTHON_BACKEND.assess(staged_region),
                    build=_failed_region_attempt(
                        "buffer lowering failed: staged proof differs from checkout analysis"
                    ),
                )
            )
            continue
        try:
            prepared.append(
                _prepare_buffer_variant(
                    context=_ScalarVariantContext(
                        project=project,
                        build_root=build_root,
                        staged_source_roots=staged_source_roots,
                        scan=staged_scan,
                        region=staged_region,
                    ),
                    plan=staged_plan,
                    variant_id=variant_id,
                )
            )
        except (SyntaxError, ValueError) as error:
            failures.append(
                PackageRegionBuildFailure(
                    region=staged_region,
                    variant_id=variant_id,
                    backend="cython",
                    assessment=CYTHON_BACKEND.assess(staged_region),
                    build=_failed_region_attempt(f"buffer lowering failed: {error}"),
                )
            )
    return tuple(prepared), tuple(failures)


def _prepare_buffer_variant(
    *,
    context: _ScalarVariantContext,
    plan: BufferKernelPlan,
    variant_id: str,
) -> _PreparedTypedRegion:
    """Generate one guarded Cython zero-copy buffer variant.

    Args:
        context: Staged source, scan, region, and filesystem evidence.
        plan: Revalidated buffer proof plan.
        variant_id: Stable layout-specific variant identity.

    Returns:
        _PreparedTypedRegion: Compilable unit and transactional shim contract.
    """
    logical_module = _typed_region_module_name(context.region, "cython", variant_id)
    generated_path = context.build_root / f"{logical_module}.pyx"
    generated = generate_buffer_kernel(
        BufferKernelGenerationRequest(
            scan=context.scan,
            region=context.region,
            plan=plan,
            logical_module=logical_module,
            output_path=generated_path,
        )
    )
    unit = CYTHON_BACKEND.lower(
        BackendLoweringRequest(
            region=context.region,
            source_path=generated_path,
            logical_module=logical_module,
            install_relative_dir=_region_artifact_relative_dir(variant_id),
            members=(plan.member,),
            variant_id=variant_id,
        )
    )
    source_module = _find_module(context.project.modules, context.scan.module.name)
    staged_source_root = _staged_source_root(
        source_module,
        context.project,
        context.staged_source_roots,
    )
    return _PreparedTypedRegion(
        generation=generated.generation,
        assessment=CYTHON_BACKEND.assess(context.region),
        unit=unit,
        shim=RegionShimConfig(
            source_module=context.scan.module.name,
            source_path=context.scan.module.path,
            region_id=context.region.id,
            variant_id=variant_id,
            backend="cython",
            compiled_module=logical_module,
            artifact_dir=staged_source_root / unit.install_relative_dir,
            bindings=generated.generation.bindings,
            dispatch_rank=_BUFFER_DISPATCH_RANK,
            variant_guards=plan.guards,
        ),
        minimum_marginal_speedup=_SPECIALIZED_VARIANT_MINIMUM_SPEEDUP,
    )


def _buffer_region_for_plan(scan: ModuleScan, plan: BufferKernelPlan) -> TypedRegion:
    region = next(
        (
            candidate
            for candidate in scan.typed_regions
            if any(member.id == plan.member for member in candidate.members)
        ),
        None,
    )
    if region is None:
        raise ValueError(f"staged buffer member is absent: {plan.member.stable_id}")
    return region


def _buffer_variant_id(plan: BufferKernelPlan) -> str:
    layout = plan.buffers[0].layout
    return f"{plan.id}@cython-buffer-{layout.format}-{layout.itemsize}"


def _extend_with_run_guard_variants(
    *,
    context: _RunGuardExtensionContext,
    prepared: list[_PreparedTypedRegion],
    failures: list[PackageRegionBuildFailure],
) -> tuple[list[_PreparedTypedRegion], list[PackageRegionBuildFailure], int]:
    """Prepend source-fused guard variants carried by an accepted source arm.

    Args:
        context: Transformed project, copied roots, plans, and progress callback.
        prepared: Existing generic and specialized native variants.
        failures: Existing preparation failures retained for reports.

    Returns:
        tuple[list[_PreparedTypedRegion], list[PackageRegionBuildFailure], int]:
        Updated variants, failures, and prepared source-fused guard count.
    """
    if not context.plans:
        return prepared, failures, 0
    run_guards, run_guard_failures = _prepare_run_guard_variants(context)
    owned_members = frozenset(
        member for run_guard in run_guards for member in run_guard.generation.selected_members
    )
    retained = [
        candidate
        for candidate in prepared
        if owned_members.isdisjoint(candidate.generation.selected_members)
    ]
    redundant_count = len(prepared) - len(retained)
    if redundant_count:
        _progress(
            context.progress,
            f"removed {redundant_count} generic variant(s) already owned by "
            "transactional source-fused units",
        )
    return (
        [*run_guards, *retained],
        [*failures, *run_guard_failures],
        len(run_guards),
    )


def _run_guard_member_ids(plan: RunGuardNativePlan) -> tuple[SymbolId, ...]:
    """Return every private or public member owned by a source-fused unit.

    Args:
        plan: Accepted source-fused run guard and optional completion index.

    Returns:
        tuple[SymbolId, ...]: Members excluded from duplicate generic selection.
    """
    members = [plan.eligibility_helper, plan.helper]
    if plan.completion_index is not None:
        members.extend((plan.completion_index.snapshot, plan.completion_index.query))
    return tuple(members)


def _prepare_run_guard_variants(
    context: _RunGuardExtensionContext,
) -> tuple[tuple[_PreparedTypedRegion, ...], tuple[PackageRegionBuildFailure, ...]]:
    """Revalidate accepted transformed helpers and prepare their Cython units.

    Args:
        context: Source-fused plans and staged transformed source roots.

    Returns:
        tuple[tuple[_PreparedTypedRegion, ...], tuple[PackageRegionBuildFailure, ...]]:
        Prepared helpers followed by explicit disabled-backend or lowering failures.
    """
    cython_enabled = "cython" in context.project.config.compile.backends
    scans: dict[str, ModuleScan] = {}
    prepared: list[_PreparedTypedRegion] = []
    failures: list[PackageRegionBuildFailure] = []
    for plan in context.plans:
        source_module = _find_module(context.project.modules, plan.helper.module)
        scan = scans.get(plan.helper.module)
        if scan is None:
            staged_module = _staged_module(
                source_module,
                context.project,
                context.staged_source_roots,
            )
            scan = enrich_island_analysis(scan_module(staged_module))
            scans[plan.helper.module] = scan
        source_region = next(
            (
                region
                for region in scan.typed_regions
                if any(member.id == plan.helper for member in region.members)
            ),
            None,
        )
        variant_id = f"{plan.stable_id}@cython-source-fused"
        if source_region is None:
            owner_region = next(
                (
                    region
                    for region in scan.typed_regions
                    if any(member.id == plan.owner for member in region.members)
                ),
                None,
            )
            reason = f"source-fused guard helper disappeared: {plan.helper.stable_id}"
            _progress(context.progress, reason)
            if owner_region is not None:
                failures.append(
                    PackageRegionBuildFailure(
                        region=owner_region,
                        variant_id=variant_id,
                        backend="cython",
                        assessment=CYTHON_BACKEND.assess(owner_region),
                        build=_failed_region_attempt(reason),
                    )
                )
            continue
        try:
            region = build_run_guard_region(scan, plan)
        except (OSError, SyntaxError, ValueError) as error:
            failures.append(
                PackageRegionBuildFailure(
                    region=source_region,
                    variant_id=variant_id,
                    backend="cython",
                    assessment=CYTHON_BACKEND.assess(source_region),
                    build=_failed_region_attempt(f"source-fused guard lowering failed: {error}"),
                )
            )
            continue
        if not cython_enabled:
            reason = "source-fused guard skipped because Cython is disabled"
            _progress(context.progress, reason)
            failures.append(
                PackageRegionBuildFailure(
                    region=region,
                    variant_id=variant_id,
                    backend="cython",
                    assessment=CYTHON_BACKEND.assess(region),
                    build=_failed_region_attempt(reason),
                )
            )
            continue
        try:
            prepared.append(
                _prepare_run_guard_variant(
                    context=_ScalarVariantContext(
                        project=context.project,
                        build_root=context.build_root,
                        staged_source_roots=context.staged_source_roots,
                        scan=scan,
                        region=region,
                    ),
                    plan=plan,
                    variant_id=variant_id,
                )
            )
        except (OSError, SyntaxError, ValueError) as error:
            failures.append(
                PackageRegionBuildFailure(
                    region=region,
                    variant_id=variant_id,
                    backend="cython",
                    assessment=CYTHON_BACKEND.assess(region),
                    build=_failed_region_attempt(f"source-fused guard lowering failed: {error}"),
                )
            )
    return tuple(prepared), tuple(failures)


def _prepare_run_guard_variant(
    *,
    context: _ScalarVariantContext,
    plan: RunGuardNativePlan,
    variant_id: str,
) -> _PreparedTypedRegion:
    """Generate one transactional boxed Cython helper unit.

    Args:
        context: Staged transformed source, scan, region, and build roots.
        plan: Revalidated source-fused guard plan.
        variant_id: Stable backend variant identity.

    Returns:
        _PreparedTypedRegion: Compilable helper and transactional dispatcher contract.
    """
    logical_module = _typed_region_module_name(context.region, "cython", variant_id)
    generated_path = context.build_root / f"{logical_module}.py"
    selected_members = tuple(member.id for member in context.region.members)
    generation = generate_run_guard(
        RunGuardGenerationRequest(
            scan=context.scan,
            region=context.region,
            plan=plan,
            logical_module=logical_module,
            output_path=generated_path,
        )
    )
    unit = CYTHON_BACKEND.lower(
        BackendLoweringRequest(
            region=context.region,
            source_path=generated_path,
            logical_module=logical_module,
            install_relative_dir=_region_artifact_relative_dir(variant_id),
            members=selected_members,
            variant_id=variant_id,
        )
    )
    source_module = _find_module(context.project.modules, context.scan.module.name)
    staged_source_root = _staged_source_root(
        source_module,
        context.project,
        context.staged_source_roots,
    )
    return _PreparedTypedRegion(
        generation=generation,
        assessment=CYTHON_BACKEND.assess(context.region),
        unit=unit,
        shim=RegionShimConfig(
            source_module=context.scan.module.name,
            source_path=context.scan.module.path,
            region_id=context.region.id,
            variant_id=variant_id,
            backend="cython",
            compiled_module=logical_module,
            artifact_dir=staged_source_root / unit.install_relative_dir,
            bindings=generation.bindings,
            dispatch_rank=_RUN_GUARD_DISPATCH_RANK,
            variant_guards=(),
        ),
        lowering_mode="source-fused",
        native_helpers=tuple(member.qualname for member in selected_members),
        minimum_marginal_speedup=_SPECIALIZED_VARIANT_MINIMUM_SPEEDUP,
        profitability_symbols=(plan.owner,),
    )


def _prepare_scalar_variant(
    *,
    context: _ScalarVariantContext,
    plan: ScalarKernelPlan,
    variant_id: str,
    proof: ScalarWidthProof,
) -> _PreparedTypedRegion:
    """Generate one Cython fixed-width variant and staged dispatcher config.

    Args:
        context: Target project, copied roots, scan, and region evidence.
        plan: Revalidated scalar kernel plan.
        variant_id: Stable width-specific variant identity.
        proof: Width proof selected from ``plan``.

    Returns:
        _PreparedTypedRegion: Compilable unit and guarded runtime binding.

    """
    logical_module = _typed_region_module_name(context.region, "cython", variant_id)
    generated_path = context.build_root / f"{logical_module}.pyx"
    generated = generate_scalar_kernel(
        ScalarKernelGenerationRequest(
            scan=context.scan,
            region=context.region,
            plan=plan,
            width_proof=proof,
            logical_module=logical_module,
            output_path=generated_path,
        )
    )
    unit = CYTHON_BACKEND.lower(
        BackendLoweringRequest(
            region=context.region,
            source_path=generated_path,
            logical_module=logical_module,
            install_relative_dir=_region_artifact_relative_dir(variant_id),
            members=(plan.member,),
            variant_id=variant_id,
        )
    )
    source_module = _find_module(context.project.modules, context.scan.module.name)
    staged_source_root = _staged_source_root(
        source_module,
        context.project,
        context.staged_source_roots,
    )
    return _PreparedTypedRegion(
        generation=generated.generation,
        assessment=CYTHON_BACKEND.assess(context.region),
        unit=unit,
        shim=RegionShimConfig(
            source_module=context.scan.module.name,
            source_path=context.scan.module.path,
            region_id=context.region.id,
            variant_id=variant_id,
            backend="cython",
            compiled_module=logical_module,
            artifact_dir=staged_source_root / unit.install_relative_dir,
            bindings=generated.generation.bindings,
            dispatch_rank=(
                _SCALAR_INT32_DISPATCH_RANK
                if proof.native.width == _SCALAR_INT32_WIDTH
                else _SCALAR_INT64_DISPATCH_RANK
            ),
            variant_guards=proof.guards,
        ),
        minimum_marginal_speedup=_SPECIALIZED_VARIANT_MINIMUM_SPEEDUP,
    )


def _scalar_region_for_plan(scan: ModuleScan, plan: ScalarKernelPlan) -> TypedRegion:
    region = next(
        (
            candidate
            for candidate in scan.typed_regions
            if any(member.id == plan.member for member in candidate.members)
        ),
        None,
    )
    if region is None:
        raise ValueError(f"staged scalar member is absent: {plan.member.stable_id}")
    return region


def _scalar_variant_id(plan: ScalarKernelPlan, proof: ScalarWidthProof) -> str:
    signed = "i" if proof.native.signed else "u"
    return f"{plan.id}@cython-{signed}{proof.native.width}"


def _prepare_call_chain_variant(
    *,
    context: _ScalarVariantContext,
    plan: CallChainPlan,
    variant_id: str,
    proof: ScalarWidthProof,
) -> _PreparedTypedRegion:
    """Generate one guarded Cython call-chain variant with private helpers.

    Args:
        context: Staged source, scan, region, and filesystem evidence.
        plan: Revalidated direct call-chain plan.
        variant_id: Stable width-specific variant identity.
        proof: Fixed-width proof selected for lowering.

    Returns:
        _PreparedTypedRegion: Compilable unit and transactional shim contract.
    """
    logical_module = _typed_region_module_name(context.region, "cython", variant_id)
    generated_path = context.build_root / f"{logical_module}.pyx"
    generated = generate_call_chain_kernel(
        CallChainGenerationRequest(
            scan=context.scan,
            region=context.region,
            plan=plan,
            width_proof=proof,
            logical_module=logical_module,
            output_path=generated_path,
        )
    )
    members = (plan.root, *plan.helpers)
    unit = CYTHON_BACKEND.lower(
        BackendLoweringRequest(
            region=context.region,
            source_path=generated_path,
            logical_module=logical_module,
            install_relative_dir=_region_artifact_relative_dir(variant_id),
            members=members,
            variant_id=variant_id,
        )
    )
    source_module = _find_module(context.project.modules, context.scan.module.name)
    staged_source_root = _staged_source_root(
        source_module,
        context.project,
        context.staged_source_roots,
    )
    return _PreparedTypedRegion(
        generation=generated.generation,
        assessment=CYTHON_BACKEND.assess(context.region),
        unit=unit,
        shim=RegionShimConfig(
            source_module=context.scan.module.name,
            source_path=context.scan.module.path,
            region_id=context.region.id,
            variant_id=variant_id,
            backend="cython",
            compiled_module=logical_module,
            artifact_dir=staged_source_root / unit.install_relative_dir,
            bindings=generated.generation.bindings,
            dispatch_rank=(
                _SCALAR_INT32_DISPATCH_RANK
                if proof.native.width == _SCALAR_INT32_WIDTH
                else _SCALAR_INT64_DISPATCH_RANK
            ),
            variant_guards=call_chain_runtime_guards(plan, proof),
        ),
        minimum_marginal_speedup=_SPECIALIZED_VARIANT_MINIMUM_SPEEDUP,
    )


def _call_chain_region_for_plan(scan: ModuleScan, plan: CallChainPlan) -> TypedRegion:
    region = next((item for item in scan.typed_regions if item.id == plan.region_id), None)
    if region is None:
        region = next(
            (
                item
                for item in scan.typed_regions
                if any(member.id == plan.root for member in item.members)
            ),
            None,
        )
    if region is None:
        raise ValueError(f"staged call-chain root is absent: {plan.root.stable_id}")
    return region


def _call_chain_variant_id(plan: CallChainPlan, proof: ScalarWidthProof) -> str:
    signed = "i" if proof.native.signed else "u"
    return f"{plan.id}@cython-chain-{signed}{proof.native.width}"


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
            region_id=staged.region.id,
            variant_id=variant_id,
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
            region_id=staged.region.id,
            variant_id=selection.variant_id,
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


def _deduplicated_public_bindings(
    prepared: tuple[_PreparedTypedRegion, ...],
) -> tuple[BindingTarget, ...]:
    """Collapse native variants to one report-facing public binding promise.

    Args:
        prepared: Accepted scalar, generic, and specialization variants.

    Returns:
        tuple[BindingTarget, ...]: First binding for each public installation destination.
    """
    bindings: dict[tuple[object, ...], BindingTarget] = {}
    for item in prepared:
        for binding in item.generation.bindings:
            identity = (
                binding.source,
                binding.kind,
                binding.owner_class,
                binding.target_owner_class,
                binding.execution_kind,
            )
            bindings.setdefault(identity, binding)
    return tuple(bindings.values())


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
            ("scalar_kernel_generator", SCALAR_KERNEL_GENERATOR_VERSION),
            ("call_chain_generator", CALL_CHAIN_GENERATOR_VERSION),
            ("buffer_kernel_generator", BUFFER_KERNEL_GENERATOR_VERSION),
            ("run_guard_generator", RUN_GUARD_GENERATOR_VERSION),
        ),
        project_source_digest=context.source_tree_digest or None,
    )
    batched_cython = _compile_batched_cython_variants(
        prepared,
        backend_context,
        cache_root=context.compile_cache_dir,
        progress=context.progress,
    )
    circuit_state = _BackendCircuitBuildState(
        context=backend_context,
        cache_root=context.compile_cache_dir,
        progress=context.progress,
        batched_cython=batched_cython,
        triggers={},
        primaries={},
        fallbacks={},
        enabled=context.enable_project_circuit,
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
            result = _compile_or_skip_typed_variant(
                candidate=candidate,
                preferred=item,
                index=index - 1,
                total=len(prepared),
                state=circuit_state,
            )
            circuit_routed = _is_backend_circuit_routed(candidate, circuit_state)
            _progress_typed_variant_cache(
                result=result,
                candidate=candidate,
                circuit_routed=circuit_routed,
                progress=context.progress,
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
                if not circuit_routed:
                    _progress(
                        context.progress,
                        (
                            f"compiled typed region variant {candidate.unit.region_id} "
                            f"as {candidate.lowering_mode}"
                        ),
                    )
                break
            _open_project_backend_circuit(
                result=result,
                candidate=candidate,
                remaining=prepared[index - 1 :],
                state=circuit_state,
            )
            fallback = candidate.fallback
            if fallback is not None and _should_retry_with_fallback(candidate, result):
                rejected_attempts.append((candidate, tagged_attempt))
                if not circuit_routed:
                    _progress(
                        context.progress,
                        (
                            f"retrying deterministic {candidate.generation.backend} failure "
                            f"with {fallback.generation.backend} {fallback.lowering_mode}: "
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
            if not circuit_routed:
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


def _backend_circuit_key(item: _PreparedTypedRegion) -> tuple[Backend, str]:
    """Return the backend and import-package scope for repeated rejection control.

    Args:
        item: Prepared variant whose source binding defines the import package.

    Returns:
        tuple[Backend, str]: Backend and top-level import package identity.
    """
    source_package = item.shim.source_module.partition(".")[0]
    return item.generation.backend, source_package


def _compile_or_skip_typed_variant(
    *,
    candidate: _PreparedTypedRegion,
    preferred: _PreparedTypedRegion,
    index: int,
    total: int,
    state: _BackendCircuitBuildState,
) -> BackendCompileResult:
    """Resolve one variant from a circuit, batch, cache, or compiler backend.

    Args:
        candidate: Current member of a preferred/fallback chain.
        preferred: Top-level prepared variant owning the ordered build slot.
        index: Zero-based position of the preferred variant.
        total: Total number of preferred variants in this build.
        state: Per-build circuit and precompiled-result state.

    Returns:
        BackendCompileResult: Explicit skip, restored entry, or compiler result.
    """
    primary_result = state.primaries.get(candidate.unit.region_id)
    if primary_result is not None:
        return primary_result
    trigger = state.triggers.get(_backend_circuit_key(candidate))
    if trigger is not None:
        cached = probe_region_cache(
            _compiler_backend(candidate.generation.backend),
            candidate.unit,
            state.context,
            cache_root=state.cache_root,
        )
        if cached is not None:
            return cached
        return _project_circuit_bypass(candidate, trigger=trigger)
    fallback_result = state.fallbacks.get(candidate.unit.region_id)
    if fallback_result is not None:
        return fallback_result
    _progress(
        state.progress,
        (
            f"compiling typed region variant {index + 1}/{total} with "
            f"{candidate.generation.backend} ({candidate.lowering_mode}): "
            f"{candidate.unit.region_id}"
        ),
    )
    if candidate is preferred and index in state.batched_cython:
        return state.batched_cython[index]
    return _compile_typed_variant(
        candidate,
        state.context,
        cache_root=state.cache_root,
    )


def _is_backend_circuit_routed(
    item: _PreparedTypedRegion,
    state: _BackendCircuitBuildState,
) -> bool:
    """Return whether package-level circuit evidence already resolves a variant.

    Args:
        item: Current preferred or fallback variant.
        state: Per-build project-circuit and batched-result state.

    Returns:
        bool: Whether per-variant progress would duplicate a package-level summary.
    """
    return _backend_circuit_key(item) in state.triggers or item.unit.region_id in state.fallbacks


def _progress_typed_variant_cache(
    *,
    result: BackendCompileResult,
    candidate: _PreparedTypedRegion,
    circuit_routed: bool,
    progress: PackageProgress | None,
) -> None:
    """Report a direct cache result without repeating circuit batch details.

    Args:
        result: Backend result whose cache status may be user-visible.
        candidate: Variant owning the cache result.
        circuit_routed: Whether package-level progress already covered the result.
        progress: Optional user-facing compile progress callback.
    """
    if circuit_routed or result.attempt.cache_status not in {"hit", "miss"}:
        return
    _progress(
        progress,
        (
            f"compile cache {result.attempt.cache_status} for typed region variant "
            f"{candidate.unit.region_id}"
        ),
    )


def _project_circuit_bypass(
    item: _PreparedTypedRegion,
    *,
    trigger: str,
) -> BackendCompileResult:
    """Represent a policy skip after a project-scoped deterministic failure.

    Args:
        item: Preferred backend variant skipped before native execution.
        trigger: Earlier variant whose diagnostic opened the project circuit.

    Returns:
        BackendCompileResult: Honest bypass evidence with no compiler process.
    """
    backend = item.generation.backend
    return BackendCompileResult(
        attempt=CompileAttempt(
            success=False,
            command=("atoll", "backend-circuit", backend, item.unit.region_id),
            stdout="",
            stderr=(
                f"{_BACKEND_POLICY_BYPASS_PREFIX} skipped {backend} for "
                f"{item.unit.region_id}; imported project source was already rejected by "
                f"{trigger}"
            ),
            artifact_paths=(),
            duration_seconds=0.0,
            phase_timings=(
                CompilePhaseTiming(
                    name="backend_project_circuit",
                    duration_seconds=0.0,
                    detail=f"{backend}; trigger={trigger}; skipped={item.unit.region_id}",
                ),
            ),
        ),
        artifacts=(),
    )


def _open_project_backend_circuit(
    *,
    result: BackendCompileResult,
    candidate: _PreparedTypedRegion,
    remaining: tuple[_PreparedTypedRegion, ...],
    state: _BackendCircuitBuildState,
) -> None:
    """Open one project circuit and precompile independent Cython fallbacks.

    Conditional variants remain dormant until their preferred class variant
    fails. A deterministic failure inside the Cython batch is still isolated by
    the region cache's recursive bisection, so one fallback cannot discard peers.

    Args:
        result: Current backend result carrying structured diagnostic scope.
        candidate: Variant whose backend produced the result.
        remaining: Current and later preferred variants in build order.
        state: Per-build circuit and precompiled-result state to update.
    """
    if not state.enabled or result.diagnostic_scope != "project":
        return
    circuit_key = _backend_circuit_key(candidate)
    if circuit_key in state.triggers:
        return
    trigger = candidate.unit.region_id
    state.triggers[circuit_key] = trigger
    blocked_backend, source_package = circuit_key
    eligible: dict[str, _PreparedTypedRegion] = {}
    for item in remaining:
        if (
            item.generation.backend != blocked_backend
            or item.shim.source_module.partition(".")[0] != source_package
            or item.conditional_on_failure_of is not None
        ):
            continue
        cached_primary = probe_region_cache(
            _compiler_backend(blocked_backend),
            item.unit,
            state.context,
            cache_root=state.cache_root,
        )
        if cached_primary is not None:
            state.primaries[item.unit.region_id] = cached_primary
            if cached_primary.attempt.success:
                continue
        fallback = item.fallback
        if fallback is None or fallback.generation.backend != "cython":
            continue
        eligible.setdefault(fallback.unit.region_id, fallback)
    if len(eligible) < _MINIMUM_CYTHON_BATCH_SIZE:
        return
    fallbacks = tuple(eligible.values())
    _progress(
        state.progress,
        (
            f"{blocked_backend} project circuit opened for {source_package} after {trigger}; "
            f"probing {len(fallbacks)} Cython fallback(s) as one batch frontier"
        ),
    )
    results = compile_many_with_region_cache(
        _compiler_backend("cython"),
        tuple(item.unit for item in fallbacks),
        state.context,
        cache_root=state.cache_root,
    )
    batch_invocations = _cython_batch_invocation_count(results)
    if batch_invocations:
        _progress(
            state.progress,
            f"compiled project-circuit fallbacks in {batch_invocations} Cython batch(es)",
        )
    else:
        _progress(
            state.progress,
            "restored all project-circuit Cython fallbacks from cache",
        )
    state.fallbacks.update(
        {
            fallback.unit.region_id: fallback_result
            for fallback, fallback_result in zip(fallbacks, results, strict=True)
        }
    )


def _cython_batch_invocation_count(
    results: tuple[BackendCompileResult, ...],
) -> int:
    """Count successful, terminal-failure, and bisected Cython processes.

    Args:
        results: Per-unit cache results carrying cold physical-build timings.

    Returns:
        int: Exact number of native compiler invocations represented by results.
    """
    return sum(
        timing.name in {"cython_batch", "cython_batch_retry"}
        for result in results
        for timing in result.attempt.phase_timings
    )


def _compile_batched_cython_variants(
    prepared: tuple[_PreparedTypedRegion, ...],
    context: BackendCompileContext,
    *,
    cache_root: Path,
    progress: PackageProgress | None,
) -> dict[int, BackendCompileResult]:
    """Compile independent top-level Cython variants through one cache frontier.

    Fallback chains and conditional class/method variants retain sequential
    orchestration because their eligibility depends on an earlier result. Every
    other Cython variant is independent and can share cold compiler startup while
    preserving its own fingerprint, artifact manifest, and rejection decision.

    Args:
        prepared: Ordered native variants ready for backend execution.
        context: Shared typed-region backend context.
        cache_root: Persistent region cache root.
        progress: Optional user-facing compile progress callback.

    Returns:
        dict[int, BackendCompileResult]: Prepared indexes resolved by the batch frontier.
    """
    eligible = tuple(
        (index, item)
        for index, item in enumerate(prepared)
        if item.generation.backend == "cython"
        and item.fallback is None
        and item.conditional_on_failure_of is None
    )
    if len(eligible) < _MINIMUM_CYTHON_BATCH_SIZE:
        return {}
    _progress(progress, f"probing compile cache for {len(eligible)} Cython variant(s)")
    results = compile_many_with_region_cache(
        CYTHON_BACKEND,
        tuple(item.unit for _index, item in eligible),
        context,
        cache_root=cache_root,
    )
    batch_invocations = _cython_batch_invocation_count(results)
    if batch_invocations:
        _progress(
            progress,
            f"compiled Cython cache misses in {batch_invocations} physical batch(es)",
        )
    else:
        _progress(progress, "restored all Cython variants from compile cache")
    return {index: result for (index, _item), result in zip(eligible, results, strict=True)}


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
    """Return whether rejection or circuit policy permits the prepared fallback.

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
    return result.attempt.stderr.startswith((diagnostic_prefix, _BACKEND_POLICY_BYPASS_PREFIX))


def _recovered_backend_attempt(
    attempt: CompileAttempt,
    fallback_variant_id: str,
) -> CompileAttempt:
    """Retain rejected or bypassed backend evidence after fallback succeeds.

    Args:
        attempt: Native rejection or orchestration bypass being recovered.
        fallback_variant_id: Variant ID used when no preferred backend succeeds.

    Returns:
        CompileAttempt: Rejection augmented with successful fallback evidence.
    """
    if attempt.stderr.startswith(_BACKEND_POLICY_BYPASS_PREFIX):
        recovery = (
            f"project-scoped backend circuit bypassed this variant; selected {fallback_variant_id}"
        )
    elif attempt.stderr.startswith("MYPYC_TYPE_ERROR:"):
        recovery = f"mypyc rejected this variant; compiled {fallback_variant_id} with Cython"
    else:
        recovery = (
            "whole-callable Cython rejected this variant; compiled "
            f"{fallback_variant_id} with outlined Cython"
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


def _partition_wheel_owned_variants(
    prepared: tuple[_PreparedTypedRegion, ...],
    *,
    staged_source_roots: tuple[Path, ...],
    install_root: Path,
) -> tuple[tuple[_PreparedTypedRegion, ...], tuple[PackageRegionBuildFailure, ...]]:
    """Keep only variants whose source module is owned by the baseline wheel.

    Flat-layout repositories often contain documentation plugins, release
    scripts, examples, and other importable-looking Python files beside their
    distributable package. The PEP 517 wheel is the authoritative package
    boundary: automatic compilation may replace a shipped source module, but it
    must not introduce a module that the backend intentionally omitted.

    Args:
        prepared: Lowered native variants awaiting compiler invocation.
        staged_source_roots: Copied import roots used to derive install paths.
        install_root: Unpacked baseline wheel payload.

    Returns:
        tuple[tuple[_PreparedTypedRegion, ...], tuple[PackageRegionBuildFailure, ...]]:
        Wheel-owned variants and deterministic interpreted-fallback evidence for
        every omitted source module.
    """
    retained: list[_PreparedTypedRegion] = []
    omitted: list[PackageRegionBuildFailure] = []
    for item in prepared:
        relative = _source_relative_path(item.shim.source_path, staged_source_roots)
        if (install_root / relative).is_file():
            retained.append(item)
            continue
        error = f"target PEP 517 wheel omitted a compiled source module: {relative.as_posix()}"
        omitted.append(
            PackageRegionBuildFailure(
                region=item.generation.region,
                variant_id=item.unit.region_id,
                backend=item.generation.backend,
                assessment=item.assessment,
                build=_failed_region_attempt(error),
            )
        )
    return tuple(retained), tuple(omitted)


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
        variant_id = config.variant_id or config.region_id
        records = tuple(
            record
            for record in artifact_records
            if record.region_id == variant_id
            or (
                record.region_id == "__shared__"
                and PurePosixPath(record.install_relative_path).parent.name
                == config.artifact_dir.name
            )
        )
        if not records:
            raise ValueError(f"compiled variant has no artifact records: {variant_id}")
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
        accepted = tuple(
            config for config in configs if (config.variant_id or config.region_id) in accepted_ids
        )
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
    call_chain_analyses: tuple[CallChainAnalysisResult, ...],
) -> _ProfilePreparation:
    """Build, test, and profile the baseline before static region selection.

    Args:
        options: Validated command or generation options.
        project: Discovered target project configuration and modules.
        scans: Static scan facts used to include task-spawn callees in targeted observation.
        call_chain_analyses: Static direct edges counted by targeted profiling.

    Returns:
        _ProfilePreparation: Prepared baseline/profile evidence or an early failure.
    """
    benchmark = project.config.compile.benchmark_command
    test_command = project.config.compile.test_command
    static_execution_plans = build_execution_plans(scans, None)
    application_config_error = (
        "--apply-source requires configured test_command and benchmark_command"
        if options.apply_source
        and (not options.run_quality_gates or test_command is None or benchmark is None)
        else None
    )
    application_root_error = (
        validate_source_application_root(project.config.root)
        if options.apply_source and application_config_error is None
        else None
    )
    if (
        not options.run_quality_gates
        or benchmark is None
        or application_config_error is not None
        or application_root_error is not None
    ):
        preparation = _ProfilePreparation(execution_plans=static_execution_plans)
        if options.apply_source:
            preparation = replace(
                preparation,
                failure=_failed_result(
                    project.config.root,
                    options.output_dir,
                    application_config_error or application_root_error or "source apply failed",
                    execution_plans=static_execution_plans,
                ),
            )
        return preparation
    output_dir = _resolve_output_dir(project.config.root, options.output_dir)
    build_root = output_dir / "build"
    install_root = output_dir / "install"
    _progress(options.progress, f"resetting temporary build roots in {output_dir}")
    _reset_dir(build_root)
    _reset_dir(install_root)
    baseline = _prepare_baseline_wheel_payload(
        _BaselineWheelPreparation(
            project=project,
            build_root=build_root,
            install_root=install_root,
            cache_root=options.cache_dir or project.config.cache_dir,
            progress=options.progress,
            run_quality_gates=options.run_quality_gates,
        )
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
        call_edge_targets=_call_chain_profile_targets(call_chain_analyses),
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
            f"{profile.mapped_project_samples} mapped to project code, "
            f"{profile.scheduler_overhead_samples} attributed to nested scheduler/library work"
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

    Captured command, profile, and benchmark evidence remains available to the
    report writer, while copied payloads and build roots are always removed.

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
    cleanup_removed = _remove_source_clean_scratch(build_root, install_root)
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
        cleanup_kept=(),
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


def _remove_source_clean_scratch(build_root: Path, install_root: Path) -> tuple[Path, ...]:
    """Remove all disposable source-clean roots after a failed operation.

    Failure diagnostics are already captured in command and report evidence.
    Keeping copied payloads or rejected wheels would make a failed compile look
    partially successful and violates the source-clean persistence contract.

    Args:
        build_root: Disposable build, trial, and candidate-wheel root.
        install_root: Disposable unpacked install payload.

    Returns:
        Paths that existed and were removed, in build-then-install order.
    """
    return (*_remove_tree(build_root), *_remove_tree(install_root))


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
        _BaselineWheelPreparation(
            project=project,
            build_root=build_root,
            install_root=install_root,
            cache_root=options.cache_dir or project.config.cache_dir,
            progress=options.progress,
            run_quality_gates=options.run_quality_gates,
        )
    )


def _prepare_baseline_wheel_payload(
    preparation: _BaselineWheelPreparation,
) -> _BaselineWheelPayload:
    """Build and unpack the target project's normal wheel from a clean copy.

    Args:
        preparation: Project, paths, cache ownership, and quality-gate policy.

    Returns:
        _BaselineWheelPayload: Baseline wheel and unpacked payload evidence.
    """
    project = preparation.project
    build_root = preparation.build_root
    install_root = preparation.install_root
    copied_project = build_root / "pep517-project"
    baseline_output = build_root / "pep517-dist"
    _progress(preparation.progress, "building target PEP 517 baseline wheel")
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
    cache_probe = restore_baseline_wheel(
        project_root=copied_project,
        cache_root=preparation.cache_root / "baseline-wheel",
        output_dir=baseline_output,
    )
    phase_timings = [
        copy_timing,
        CompilePhaseTiming(
            name="pep517_cache_lookup",
            duration_seconds=cache_probe.lookup_duration_seconds,
            detail=f"{cache_probe.status}; {cache_probe.reason}",
        ),
    ]
    if cache_probe.wheel_path is not None:
        cache_key = cast(str, cache_probe.key)
        phase_timings.append(
            CompilePhaseTiming(
                name="pep517_cache_restore",
                duration_seconds=cache_probe.restore_duration_seconds,
                detail=cache_probe.wheel_path.name,
            )
        )
        evidence = WheelBuildEvidence(
            command=("atoll", "cache", "restore", "baseline-wheel", cache_key),
            project_root=copied_project.resolve(),
            outdir=baseline_output.resolve(),
            returncode=0,
            stdout="restored target PEP 517 baseline wheel from Atoll cache",
            stderr="",
            duration_seconds=(
                cache_probe.lookup_duration_seconds + cache_probe.restore_duration_seconds
            ),
            wheel_paths=(cache_probe.wheel_path,),
        )
        _progress(preparation.progress, "PEP 517 baseline wheel cache hit")
    else:
        cache_action = "miss" if cache_probe.status == "miss" else "bypass"
        _progress(
            preparation.progress,
            f"PEP 517 baseline wheel cache {cache_action}: {cache_probe.reason}",
        )
        evidence = build_baseline_wheel(copied_project, baseline_output)
        phase_timings.append(
            CompilePhaseTiming(
                name="pep517_wheel",
                duration_seconds=evidence.duration_seconds,
                detail=f"exit {evidence.returncode}",
            )
        )
    if not evidence.succeeded or len(evidence.wheel_paths) != 1:
        error = _baseline_build_error(evidence)
        _progress(
            preparation.progress,
            f"PEP 517 baseline wheel failed in {evidence.duration_seconds:.2f}s",
        )
        return _BaselineWheelPayload(
            wheel_path=None,
            build=CompileAttempt(
                success=False,
                command=evidence.command,
                stdout=evidence.stdout,
                stderr=error,
                artifact_paths=(),
                duration_seconds=sum(timing.duration_seconds for timing in phase_timings),
                phase_timings=tuple(phase_timings),
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
                    sum(timing.duration_seconds for timing in phase_timings)
                    + unpack_timing.duration_seconds
                ),
                phase_timings=(*phase_timings, unpack_timing),
            ),
        )
    unpack_timing = CompilePhaseTiming(
        name="wheel_unpack",
        duration_seconds=time.perf_counter() - unpack_started,
        detail=wheel_path.name,
    )
    phase_timings.append(unpack_timing)
    if cache_probe.status == "miss" and cache_probe.key is not None:
        cache_store = store_baseline_wheel(
            key=cache_probe.key,
            wheel_path=wheel_path,
            cache_root=preparation.cache_root / "baseline-wheel",
        )
        phase_timings.append(
            CompilePhaseTiming(
                name="pep517_cache_store",
                duration_seconds=cache_store.duration_seconds,
                detail="stored" if cache_store.stored else "store skipped",
            )
        )
    baseline_started = time.perf_counter()
    baseline_install_root = build_root / "baseline-install"
    shutil.copytree(install_root, baseline_install_root)
    baseline_copy_timing = (
        CompilePhaseTiming(
            name="baseline_payload_copy",
            duration_seconds=time.perf_counter() - baseline_started,
            detail="immutable interpreted baseline",
        ),
    )
    phase_timings.extend(baseline_copy_timing)
    quality_project_root: Path | None = None
    if preparation.run_quality_gates and (
        project.config.compile.test_command is not None
        or project.config.compile.benchmark_command is not None
    ):
        _remove_quality_gate_sources(project, copied_project)
        quality_project_root = copied_project
    action = "restored" if cache_probe.status == "hit" else "built"
    _progress(preparation.progress, f"{action} and unpacked PEP 517 baseline wheel")
    return _BaselineWheelPayload(
        wheel_path=wheel_path,
        build=CompileAttempt(
            success=True,
            command=evidence.command,
            stdout=evidence.stdout,
            stderr=evidence.stderr,
            artifact_paths=(),
            duration_seconds=sum(timing.duration_seconds for timing in phase_timings),
            phase_timings=tuple(phase_timings),
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
        variant_id = config.variant_id or config.region_id
        regions_by_module.setdefault(config.source_module, []).append(variant_id)
    artifacts = {
        record.install_relative_path: VerificationArtifact(
            path=record.install_relative_path,
            digest=record.digest,
        )
        for record in records
    }
    binding_configs: dict[tuple[str, str], list[tuple[RegionShimConfig, BindingTarget]]] = {}
    for config in configs:
        for binding in config.bindings:
            if binding.required:
                key = (config.source_module, _verification_binding_qualname(binding))
                binding_configs.setdefault(key, []).append((config, binding))
    bindings: dict[tuple[str, str], VerificationBinding] = {}
    for key, variants in binding_configs.items():
        first_config, first_binding = variants[0]
        bindings[key] = VerificationBinding(
            module=first_config.source_module,
            qualname=_verification_binding_qualname(first_binding),
            kind=first_binding.kind,
            execution_kind=first_binding.execution_kind,
            variant_ids=tuple(
                sorted(config.variant_id or config.region_id for config, _ in variants)
            ),
        )
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
    _verification_progress(result, progress)
    return result


def _verification_progress(
    result: PackageVerificationResult,
    progress: PackageProgress | None,
) -> None:
    """Emit one bounded child-verification status line.

    Args:
        result: Completed child-process verification attempt.
        progress: Optional callback receiving the normalized status line.
    """
    status = "passed" if result.success else f"failed with exit {result.exit_code}"
    _progress(
        progress,
        f"{result.stage} verification {status} in {result.duration_seconds:.2f}s",
    )


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
    _clear_payload_bytecode_with_progress(
        (context.install_root,),
        context.options.progress,
    )
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
            platform_tag=_promotion_wheel_tag(context, baseline_wheel_path),
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
    elif context.requires_profitable_optimization and not context.profitable_optimization_applied:
        error = "no profile-guided candidate met its marginal speedup threshold"
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
    _remove_failed_wheels(context.project, context.output_dir)
    cleanup_removed = _remove_source_clean_scratch(context.build_root, context.install_root)
    return _SourceCleanPromotionResult(
        success=False,
        wheel_path=None,
        build=failure.build,
        verification_steps=failure.verification_steps,
        test_results=failure.quality_gate.tests if failure.quality_gate is not None else (),
        performance=(
            failure.quality_gate.performance if failure.quality_gate is not None else None
        ),
        cleanup_removed=cleanup_removed,
        cleanup_kept=(),
        error=failure.error,
    )


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
                staged_source_roots=context.staged_source_roots,
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


def _apply_runtime_safety_selection(
    *,
    context: _TypedPayloadFinalizationContext,
    finalized: _TypedPayloadFinalizationResult,
    selected_members: tuple[SymbolId, ...],
) -> tuple[
    _TypedPayloadFinalizationResult,
    _TypedRegionBuildOutcome,
    _RuntimeSafetySelectionResult,
]:
    """Apply isolated safety selection and enforce explicit member promises.

    Args:
        context: Finalization boundaries and complete native build outcome.
        finalized: Candidate subset accepted by static and profitability gates.
        selected_members: Explicitly requested bindings that may not fall back silently.

    Returns:
        tuple[_TypedPayloadFinalizationResult, _TypedRegionBuildOutcome,
        _RuntimeSafetySelectionResult]: Verified finalization state, augmented
        build outcome, and isolated verification evidence.
    """
    safety = _select_runtime_safe_variants(context, finalized)
    finalized = replace(
        finalized,
        successful=safety.successful,
        artifacts=safety.artifacts,
        build=safety.build,
        overlay_error=safety.overlay_error,
    )
    outcome = replace(
        context.outcome,
        skipped=(*context.outcome.skipped, *safety.failures),
    )
    verified_bindings = _deduplicated_public_bindings(finalized.successful)
    verified_sources = frozenset(binding.source for binding in verified_bindings)
    missing = tuple(member for member in selected_members if member not in verified_sources)
    if missing:
        missing_text = ", ".join(member.stable_id for member in missing)
        finalized = replace(
            finalized,
            overlay_error=(
                f"requested member(s) failed isolated payload verification: {missing_text}"
            ),
        )
    return finalized, outcome, safety


def _select_runtime_safe_variants(
    context: _TypedPayloadFinalizationContext,
    finalized: _TypedPayloadFinalizationResult,
) -> _RuntimeSafetySelectionResult:
    """Isolate import or binding failures and retain the maximal verified subset.

    Native toolchains can successfully emit an extension that later crashes or
    raises while its source module is imported. Verification therefore starts
    with the complete accepted set and bisects failures in fresh interpreters.
    Variants that cannot verify alone remain interpreted; verified variants are
    rematerialized from the immutable baseline before wheel promotion.

    Args:
        context: Baseline payload, staged source roots, and build boundaries.
        finalized: Variants accepted by static or profitability selection.

    Returns:
        _RuntimeSafetySelectionResult: Verified variants, rejected evidence,
        subprocess attempts, and payload rematerialization status.
    """
    candidates = finalized.successful
    if finalized.overlay_error is not None or not candidates:
        return _RuntimeSafetySelectionResult(
            successful=candidates,
            artifacts=finalized.artifacts,
            build=finalized.build,
            overlay_error=finalized.overlay_error,
        )

    attempts: list[PackageVerificationResult] = []
    rejected_results: dict[str, PackageVerificationResult] = {}

    def verify(candidates_to_verify: tuple[_PreparedTypedRegion, ...]) -> PackageVerificationResult:
        artifacts = _artifact_records_for_prepared(
            candidates_to_verify,
            context.outcome.artifacts,
        )
        plan = _typed_verification_plan(
            tuple(item.shim for item in candidates_to_verify),
            artifacts,
        )
        allowlist = frozenset(item.unit.region_id for item in candidates_to_verify)
        result = verify_package_subprocess(
            stage="payload",
            target=context.install_root,
            plan=plan,
            project_root=context.project.config.root,
            variant_allowlist=allowlist,
        )
        _verification_progress(result, context.options.progress)
        attempts.append(result)
        return result

    complete_result = verify(candidates)
    if complete_result.success:
        return _RuntimeSafetySelectionResult(
            successful=candidates,
            artifacts=finalized.artifacts,
            build=_append_verification_timings(finalized.build, tuple(attempts)),
            verification_steps=tuple(attempts),
        )

    harness_result = verify(())
    if not harness_result.success:
        return _RuntimeSafetySelectionResult(
            successful=candidates,
            artifacts=finalized.artifacts,
            build=_append_verification_timings(finalized.build, tuple(attempts)),
            verification_steps=tuple(attempts),
            overlay_error=(
                "isolated payload verification failed without any native variants: "
                f"{_verification_failure_summary(harness_result)}"
            ),
        )

    retained = _bisect_runtime_safe_variants(
        candidates,
        verify=verify,
        failed_result=complete_result,
        rejected_results=rejected_results,
    )
    retained = _resolve_runtime_variant_interactions(
        retained,
        verify=verify,
        rejected_results=rejected_results,
    )

    retained_ids = frozenset(candidate.unit.region_id for candidate in retained)
    rejected = tuple(
        candidate for candidate in candidates if candidate.unit.region_id not in retained_ids
    )
    failures = tuple(
        _runtime_verification_failure(
            candidate,
            rejected_results.get(candidate.unit.region_id, complete_result),
        )
        for candidate in rejected
    )
    overlay_error = _materialize_profitable_payload(
        baseline=context.baseline,
        staged_source_roots=context.staged_source_roots,
        install_root=context.install_root,
        superset=candidates,
        accepted=retained,
    )
    artifacts = _artifact_records_for_prepared(retained, context.outcome.artifacts)
    _progress(
        context.options.progress,
        (f"runtime verification retained {len(retained)} of {len(candidates)} compiled variant(s)"),
    )
    return _RuntimeSafetySelectionResult(
        successful=retained,
        artifacts=artifacts,
        build=_append_verification_timings(finalized.build, tuple(attempts)),
        failures=failures,
        verification_steps=tuple(attempts),
        overlay_error=overlay_error,
    )


def _bisect_runtime_safe_variants(
    candidates: tuple[_PreparedTypedRegion, ...],
    *,
    verify: Callable[
        [tuple[_PreparedTypedRegion, ...]],
        PackageVerificationResult,
    ],
    failed_result: PackageVerificationResult,
    rejected_results: dict[str, PackageVerificationResult],
) -> tuple[_PreparedTypedRegion, ...]:
    """Recursively isolate variants whose activation fails in a fresh child.

    Args:
        candidates: Ordered native variants represented by ``failed_result``.
        verify: Fresh-process verifier for an arbitrary candidate subset.
        failed_result: Verification evidence for the complete candidate tuple.
        rejected_results: Mutable diagnostic sink keyed by rejected variant ID.

    Returns:
        tuple[_PreparedTypedRegion, ...]: Variants that verify alone or within
        their recursively isolated subset.
    """
    if failed_result.success:
        return candidates
    if len(candidates) == 1:
        rejected_results[candidates[0].unit.region_id] = failed_result
        return ()
    midpoint = len(candidates) // 2
    left = candidates[:midpoint]
    right = candidates[midpoint:]
    return (
        *_bisect_runtime_safe_variants(
            left,
            verify=verify,
            failed_result=verify(left),
            rejected_results=rejected_results,
        ),
        *_bisect_runtime_safe_variants(
            right,
            verify=verify,
            failed_result=verify(right),
            rejected_results=rejected_results,
        ),
    )


def _resolve_runtime_variant_interactions(
    candidates: tuple[_PreparedTypedRegion, ...],
    *,
    verify: Callable[
        [tuple[_PreparedTypedRegion, ...]],
        PackageVerificationResult,
    ],
    rejected_results: dict[str, PackageVerificationResult],
) -> tuple[_PreparedTypedRegion, ...]:
    """Greedily remove variants that fail only when combined with safe peers.

    Args:
        candidates: Individually verified variants that may interact unsafely.
        verify: Fresh-process verifier for an arbitrary candidate subset.
        rejected_results: Mutable diagnostic sink keyed by rejected variant ID.

    Returns:
        tuple[_PreparedTypedRegion, ...]: Deterministic maximal prefix-compatible
        subset retained for wheel promotion.
    """
    if len(candidates) < _MINIMUM_INTERACTION_VARIANTS or verify(candidates).success:
        return candidates
    compatible: list[_PreparedTypedRegion] = []
    for candidate in candidates:
        trial = (*compatible, candidate)
        result = verify(tuple(trial))
        if result.success:
            compatible.append(candidate)
        else:
            rejected_results[candidate.unit.region_id] = result
    return tuple(compatible)


def _runtime_verification_failure(
    candidate: _PreparedTypedRegion,
    result: PackageVerificationResult,
) -> PackageRegionBuildFailure:
    """Normalize one unsafe runtime variant into interpreted-fallback evidence.

    Args:
        candidate: Native variant rejected by isolated activation.
        result: Child-process evidence that explains the rejection.

    Returns:
        PackageRegionBuildFailure: Report evidence preserving the Python fallback.
    """
    return PackageRegionBuildFailure(
        region=candidate.generation.region,
        variant_id=candidate.unit.region_id,
        backend=candidate.generation.backend,
        assessment=candidate.assessment,
        build=_failed_region_attempt(
            "isolated payload verification rejected native variant: "
            f"{_verification_failure_summary(result)}"
        ),
    )


def _verification_failure_summary(result: PackageVerificationResult) -> str:
    """Return bounded deterministic diagnostics for a failed child verifier.

    Args:
        result: Failed child-process verification result.

    Returns:
        str: Last non-empty diagnostic line, or an exit-code fallback, capped
        for stable progress and report output.
    """
    lines = tuple(line.strip() for line in result.stderr.splitlines() if line.strip())
    if not lines:
        return f"child exited {result.exit_code} without diagnostics"
    return lines[-1][:500]


def _append_verification_timings(
    attempt: CompileAttempt,
    results: tuple[PackageVerificationResult, ...],
) -> CompileAttempt:
    """Append every failure-isolation subprocess duration to build evidence.

    Args:
        attempt: Native compilation evidence receiving verification timings.
        results: Ordered child-process attempts from safety selection.

    Returns:
        CompileAttempt: Build evidence with every verification phase appended.
    """
    for result in results:
        attempt = _append_verification_timing(attempt, result)
    return attempt


def _execution_plan_application_for_finalized_payload(
    context: _ExecutionPlanApplicationContext,
    *,
    finalized: _TypedPayloadFinalizationResult,
) -> _ExecutionPlanApplicationOutcome:
    """Build trial context only after native payload finalization succeeds.

    Args:
        context: Validated command, project, payload, selected plan, and native-region state.
        finalized: Native payload finalization result.

    Returns:
        _ExecutionPlanApplicationOutcome: Applied plan IDs and disposable trial evidence.
    """
    if finalized.overlay_error is not None:
        return _ExecutionPlanApplicationOutcome()
    return _apply_execution_plan_trials(context)


def _apply_execution_plan_trials(
    context: _ExecutionPlanApplicationContext,
) -> _ExecutionPlanApplicationOutcome:
    """Try execution-plan backends in priority order until one plan variant passes.

    A capability-supported callback variant remains speculative until its staged
    payload passes semantics and marginal profitability. Any failed callback
    attempt advances to the next backend so the conservative task-preserving
    lowering is not lost merely because the preferred optimization was unsafe or
    unprofitable for the configured workload.

    Args:
        context: Project policy, payload roots, native allowlist, and selected plans.

    Returns:
        _ExecutionPlanApplicationOutcome: Combined backend trials and applied plan IDs.
    """
    selected = tuple(plan for plan in context.plans if isinstance(plan, ExecutionPlan))
    applied: list[str] = []
    trials: list[ExecutionPlanTrial] = []
    timings: list[CompilePhaseTiming] = []
    for plan in selected:
        remaining_backends = _EXECUTION_PLAN_BACKENDS
        while remaining_backends:
            outcome = _apply_execution_plan_trials_once(
                replace(context, plans=(plan,)),
                backends=remaining_backends,
            )
            applied.extend(outcome.applied_plan_ids)
            trials.extend(outcome.trials)
            timings.extend(outcome.timings)
            if outcome.applied_plan_ids or not outcome.trials:
                break
            attempted_backend = outcome.trials[-1].backend
            if attempted_backend is None:
                break
            attempted_index = next(
                (
                    index
                    for index, backend in enumerate(remaining_backends)
                    if backend.name == attempted_backend
                ),
                None,
            )
            if attempted_index is None:
                break
            remaining_backends = remaining_backends[attempted_index + 1 :]
    return _ExecutionPlanApplicationOutcome(
        applied_plan_ids=tuple(applied),
        trials=tuple(trials),
        timings=tuple(timings),
    )


def _apply_execution_plan_trials_once(
    context: _ExecutionPlanApplicationContext,
    *,
    backends: tuple[ExecutionPlanBackend, ...],
) -> _ExecutionPlanApplicationOutcome:
    """Stage and trial selected scheduler plans without risking the accepted payload.

    Each backend writes to a disposable copy of the current payload. Atoll validates
    every reported file change, runs the configured semantic command once, and then
    compares the planned copy with the current accepted payload. Only a passing
    semantic result and at least 1.05x marginal speedup replace the accepted payload.
    The later final payload benchmark independently enforces the configured overall
    threshold before a wheel can be promoted.

    Args:
        context: Project policy, payload roots, native allowlist, and selected plans.
        backends: Remaining backend candidates in configured priority order.

    Returns:
        _ExecutionPlanApplicationOutcome: Applied plan IDs, trial decisions, and timings.
    """
    config = context.project.config.compile
    selected = tuple(plan for plan in context.plans if isinstance(plan, ExecutionPlan))
    quality_root = context.baseline.quality_project_root
    baseline_payload_root = context.baseline.baseline_install_root
    if (
        not selected
        or not context.options.run_quality_gates
        or config.test_command is None
        or config.benchmark_command is None
        or quality_root is None
        or baseline_payload_root is None
    ):
        return _ExecutionPlanApplicationOutcome()

    trials_root = context.build_root / "execution-plan-trials"
    _reset_dir(trials_root)
    applied: list[str] = []
    trials: list[ExecutionPlanTrial] = []
    timings: list[CompilePhaseTiming] = []
    for index, plan in enumerate(selected, start=1):
        backend, assessment_diagnostics = _select_execution_plan_backend(
            context,
            plan,
            backends=backends,
        )
        if backend is None:
            trials.append(
                ExecutionPlanTrial(
                    plan_id=plan.id,
                    status="unavailable",
                    command=(),
                    exit_code=None,
                    duration_seconds=None,
                    diagnostics=assessment_diagnostics,
                    reason="no execution-plan backend accepted the complete plan",
                )
            )
            continue

        trial_root = trials_root / f"{index:02d}-{plan.id}-{backend.name}"
        staging_started = time.perf_counter()
        try:
            shutil.copytree(context.install_root, trial_root)
            staging = _stage_execution_plan_candidate(
                backend=backend,
                plan=plan,
                context=ExecutionPlanStageContext(
                    project_root=context.project.config.root,
                    payload_root=trial_root,
                    cache_root=context.project.config.cache_dir / "execution-plans",
                ),
            )
            staged = staging.staged
            _validate_staged_execution_plan(
                staged=staged,
                backend=backend,
                plan=plan,
                baseline_root=context.install_root,
                trial_root=trial_root,
            )
        except Exception as error:
            diagnostic = backend.normalize_diagnostic(
                error,
                diagnostics=str(error),
                log_path=None,
            )
            duration = time.perf_counter() - staging_started
            timings.append(
                CompilePhaseTiming(
                    name="execution_plan_staging",
                    duration_seconds=duration,
                    detail=f"{plan.id}; {backend.name}; failed",
                )
            )
            trials.append(
                ExecutionPlanTrial(
                    plan_id=plan.id,
                    status="unavailable",
                    command=(),
                    exit_code=None,
                    duration_seconds=None,
                    diagnostics=(*assessment_diagnostics, diagnostic),
                    backend=backend.name,
                    reason=diagnostic.message,
                )
            )
            _remove_tree(trial_root)
            continue

        staging_duration = time.perf_counter() - staging_started
        timings.append(
            CompilePhaseTiming(
                name="execution_plan_staging",
                duration_seconds=staging_duration,
                detail=(
                    f"{plan.id}; {backend.name}; cache {staging.cache_status}; "
                    f"{len(staged.payload_files)} file(s)"
                ),
            )
        )
        _progress(
            context.options.progress,
            (
                f"execution plan {index}/{len(selected)} staged {plan.id} with "
                f"{backend.name} in {staging_duration:.2f}s"
            ),
        )
        semantic = run_performance_command(
            config.test_command,
            project_root=quality_root,
            payload_root=trial_root,
            mode="compiled",
            variant_allowlist=context.accepted_region_ids,
        )
        timings.append(
            CompilePhaseTiming(
                name="execution_plan_semantic_test",
                duration_seconds=semantic.duration_seconds,
                detail=f"{plan.id}; exit {semantic.returncode}",
            )
        )
        if not semantic.succeeded:
            reason = _command_failure_summary(semantic, "execution-plan semantic test failed")
            trials.append(
                _execution_plan_trial(
                    _ExecutionPlanTrialRecord(
                        plan=plan,
                        backend=backend,
                        staged=staged,
                        semantic=semantic,
                        status="failed-semantics",
                        reason=reason,
                        diagnostics=(
                            *assessment_diagnostics,
                            *staging.diagnostics,
                            ExecutionPlanDiagnostic(
                                code="semantic-test-failed",
                                severity="error",
                                message=reason,
                            ),
                        ),
                        benchmark=None,
                        cache_status=staging.cache_status,
                    )
                )
            )
            _remove_tree(trial_root)
            continue

        _progress(
            context.options.progress,
            f"execution plan {index}/{len(selected)} benchmarking {plan.id}",
        )
        benchmark = run_execution_plan_benchmark(
            ExecutionPlanBenchmarkConfig(
                plan_id=plan.id,
                command=config.benchmark_command,
                samples=_EXECUTION_PLAN_BENCHMARK_SAMPLES,
                minimum_marginal_speedup=_EXECUTION_PLAN_MINIMUM_SPEEDUP,
                minimum_overall_speedup=config.minimum_speedup,
            ),
            project_root=quality_root,
            baseline_payload_root=baseline_payload_root,
            unplanned_payload_root=context.install_root,
            planned_payload_root=trial_root,
            baseline_variant_allowlist=context.accepted_region_ids,
            unplanned_variant_allowlist=context.accepted_region_ids,
            planned_variant_allowlist=context.accepted_region_ids,
            progress=partial(_execution_plan_benchmark_progress, context.options.progress, plan.id),
        )
        timings.extend(_execution_plan_benchmark_timings(plan.id, benchmark))
        status: ExecutionPlanTrialStatus = (
            "accepted" if benchmark.status == "passed" else "rejected"
        )
        if benchmark.status in {"invalid", "unavailable"}:
            status = "unavailable"
        diagnostic_severity: ExecutionPlanDiagnosticSeverity = (
            "note" if status == "accepted" else "warning"
        )
        trial = _execution_plan_trial(
            _ExecutionPlanTrialRecord(
                plan=plan,
                backend=backend,
                staged=staged,
                semantic=semantic,
                status=status,
                reason=benchmark.reason,
                diagnostics=(
                    *assessment_diagnostics,
                    *staging.diagnostics,
                    ExecutionPlanDiagnostic(
                        code=f"execution-plan-benchmark-{benchmark.status}",
                        severity=diagnostic_severity,
                        message=benchmark.reason,
                    ),
                ),
                benchmark=benchmark,
                cache_status=staging.cache_status,
            )
        )
        if status == "accepted":
            try:
                _replace_payload_transactionally(
                    current_root=context.install_root,
                    candidate_root=trial_root,
                    backup_root=trials_root / ".accepted-backup",
                )
            except OSError as error:
                diagnostic = backend.normalize_diagnostic(
                    error,
                    diagnostics=str(error),
                    log_path=None,
                )
                trial = replace(
                    trial,
                    status="unavailable",
                    reason="accepted plan could not replace the current payload",
                    diagnostics=(*trial.diagnostics, diagnostic),
                )
            else:
                applied.append(plan.id)
        else:
            _remove_tree(trial_root)
        trials.append(trial)
        _progress(
            context.options.progress,
            f"execution plan {plan.id} {trial.status}: {trial.reason or 'no reason recorded'}",
        )
    return _ExecutionPlanApplicationOutcome(
        applied_plan_ids=tuple(applied),
        trials=tuple(trials),
        timings=tuple(timings),
    )


def _stage_execution_plan_candidate(
    *,
    backend: ExecutionPlanBackend,
    plan: ExecutionPlan,
    context: ExecutionPlanStageContext,
) -> _ExecutionPlanStagingResult:
    """Restore a strict cached helper or generate and cache a fresh one.

    Cache corruption never becomes runtime input: an invalid entry is ignored
    and the backend regenerates the payload from the untouched disposable copy.
    Cache write failures are diagnostic-only because the generated candidate is
    still complete and must proceed through semantic and profitability gates.

    Args:
        backend: Selected execution-plan lowering backend.
        plan: Complete source-hashed execution plan.
        context: Disposable payload and persistent cache boundaries.

    Returns:
        _ExecutionPlanStagingResult: Staged payload and cache evidence.

    Raises:
        Exception: Propagates fingerprinting and backend staging failures for
            normalization by the caller.
    """
    fingerprint = backend.fingerprint(plan, context)
    cached = restore_execution_plan_cache(
        context.cache_root,
        context,
        plan,
        backend=backend.name,
        fingerprint=fingerprint,
    )
    if cached.status == "hit":
        if cached.staged is None:
            raise ValueError("execution-plan cache hit did not restore staged payload metadata")
        return _ExecutionPlanStagingResult(
            staged=cached.staged,
            cache_status="hit",
        )

    diagnostics: list[ExecutionPlanDiagnostic] = []
    if cached.status == "invalid":
        diagnostics.append(
            ExecutionPlanDiagnostic(
                code="execution-plan-cache-invalid",
                severity="warning",
                message="invalid execution-plan cache entry was ignored and regenerated",
                details=((cached.reason,) if cached.reason else ()),
            )
        )
    staged = backend.stage(plan, context)
    try:
        store_execution_plan_cache(
            context.cache_root,
            context,
            staged,
            fingerprint=fingerprint,
        )
    except (OSError, TypeError, ValueError) as error:
        diagnostics.append(
            ExecutionPlanDiagnostic(
                code="execution-plan-cache-store-failed",
                severity="warning",
                message="generated execution-plan payload could not be cached",
                details=(str(error) or error.__class__.__name__,),
            )
        )
    return _ExecutionPlanStagingResult(
        staged=staged,
        cache_status=cached.status,
        diagnostics=tuple(diagnostics),
    )


def _select_execution_plan_backend(
    context: _ExecutionPlanApplicationContext,
    plan: ExecutionPlan,
    *,
    backends: tuple[ExecutionPlanBackend, ...] | None = None,
) -> tuple[ExecutionPlanBackend | None, tuple[ExecutionPlanDiagnostic, ...]]:
    """Return the first backend supporting the complete plan and its assessments.

    Args:
        context: Project and profile context for backend capability checks.
        plan: Complete scheduler plan requiring strict lowering.
        backends: Optional restricted backend suffix used after a failed trial.

    Returns:
        tuple[ExecutionPlanBackend | None, tuple[ExecutionPlanDiagnostic, ...]]: Selected backend
            and normalized assessment notes, or `None` when every backend rejects the plan.
    """
    module = next(
        (
            candidate
            for candidate in context.project.modules
            if candidate.name == plan.source_module
        ),
        None,
    )
    if module is None:
        return None, (
            ExecutionPlanDiagnostic(
                code="source-module-missing",
                severity="error",
                message=f"source module {plan.source_module} is unavailable",
            ),
        )
    source_root = next(
        (root for root in context.project.config.source_roots if module.path.is_relative_to(root)),
        None,
    )
    if source_root is None:
        return None, (
            ExecutionPlanDiagnostic(
                code="source-root-missing",
                severity="error",
                message=f"source root for {plan.source_module} is unavailable",
            ),
        )
    diagnostics: list[ExecutionPlanDiagnostic] = []
    assessment_context = ExecutionPlanAssessmentContext(
        project_root=context.project.config.root,
        source_root=source_root,
        profile_status="profiled",
    )
    for backend in backends or _EXECUTION_PLAN_BACKENDS:
        try:
            assessment = backend.assess(plan, assessment_context)
        except Exception as error:
            diagnostics.append(
                backend.normalize_diagnostic(
                    error,
                    diagnostics=str(error),
                    log_path=None,
                )
            )
            continue
        if assessment.status == "supported" and not assessment.unsupported_nodes:
            diagnostics.append(
                ExecutionPlanDiagnostic(
                    code="backend-supported",
                    severity="note",
                    message=f"{backend.name} accepted the complete execution plan",
                    details=assessment.reasons,
                )
            )
            return backend, tuple(diagnostics)
        diagnostics.append(
            ExecutionPlanDiagnostic(
                code="backend-rejected",
                severity="note",
                message=f"{backend.name} rejected the execution plan",
                details=assessment.reasons,
            )
        )
    return None, tuple(diagnostics)


def _validate_staged_execution_plan(
    *,
    staged: StagedExecutionPlan,
    backend: ExecutionPlanBackend,
    plan: ExecutionPlan,
    baseline_root: Path,
    trial_root: Path,
) -> None:
    """Verify backend identity, file boundaries, digests, and complete change reporting.

    Args:
        staged: Backend output to validate before subprocess execution.
        backend: Backend expected to own the staged output.
        plan: Plan expected to own the staged output.
        baseline_root: Current accepted payload before staging.
        trial_root: Disposable candidate payload after staging.

    Raises:
        ValueError: If identity, path, digest, or changed-file evidence is inconsistent.
    """
    if staged.plan.id != plan.id or staged.backend != backend.name:
        raise ValueError("execution-plan backend returned mismatched plan identity")
    if not staged.payload_files:
        raise ValueError("execution-plan backend did not change any payload files")
    reported: dict[PurePosixPath, tuple[str | None, str]] = {}
    for payload_file in staged.payload_files:
        install_path = payload_file.install_path
        if install_path.is_absolute() or ".." in install_path.parts:
            raise ValueError(f"execution-plan payload path escapes the wheel: {install_path}")
        before_path = baseline_root / install_path
        after_path = trial_root / install_path
        before_hash = _file_digest(before_path) if before_path.is_file() else None
        if not after_path.is_file():
            raise ValueError(f"staged execution-plan file is missing: {install_path}")
        after_hash = _file_digest(after_path)
        if payload_file.before_hash != before_hash or payload_file.after_hash != after_hash:
            raise ValueError(f"staged execution-plan digest mismatch: {install_path}")
        if install_path in reported:
            raise ValueError(f"staged execution-plan file was reported twice: {install_path}")
        reported[install_path] = (before_hash, after_hash)
    actual = _changed_payload_files(baseline_root, trial_root)
    if set(reported) != set(actual):
        missing = sorted(path.as_posix() for path in set(actual) - set(reported))
        extra = sorted(path.as_posix() for path in set(reported) - set(actual))
        raise ValueError(
            "execution-plan changed-file report is incomplete: "
            f"missing={missing or 'none'} extra={extra or 'none'}"
        )


def _changed_payload_files(
    baseline_root: Path,
    candidate_root: Path,
) -> dict[PurePosixPath, tuple[str | None, str | None]]:
    """Return digest pairs for files changed between two unpacked wheel payloads.

    Args:
        baseline_root: Current accepted payload root.
        candidate_root: Disposable staged payload root.

    Returns:
        dict[PurePosixPath, tuple[str | None, str | None]]: Changed paths and before/after digests.
    """
    baseline = _payload_file_digests(baseline_root)
    candidate = _payload_file_digests(candidate_root)
    return {
        path: (baseline.get(path), candidate.get(path))
        for path in sorted(set(baseline) | set(candidate), key=PurePosixPath.as_posix)
        if baseline.get(path) != candidate.get(path)
    }


def _payload_file_digests(root: Path) -> dict[PurePosixPath, str]:
    """Hash every regular payload file using install-relative POSIX paths.

    Args:
        root: Unpacked wheel payload root.

    Returns:
        dict[PurePosixPath, str]: File digests keyed by install-relative path.
    """
    return {
        PurePosixPath(path.relative_to(root).as_posix()): _file_digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _execution_plan_trial(record: _ExecutionPlanTrialRecord) -> ExecutionPlanTrial:
    """Build one immutable execution-plan trial from subprocess evidence.

    Args:
        record: Normalized backend, staging, subprocess, and decision evidence.

    Returns:
        ExecutionPlanTrial: Report-ready immutable trial evidence.
    """
    return ExecutionPlanTrial(
        plan_id=record.plan.id,
        status=record.status,
        command=record.semantic.command,
        exit_code=record.semantic.returncode,
        duration_seconds=record.semantic.duration_seconds,
        diagnostics=record.diagnostics,
        backend=record.backend.name,
        reason=record.reason,
        benchmark_command=(
            record.benchmark.samples[0].run.command
            if record.benchmark and record.benchmark.samples
            else ()
        ),
        benchmark_status=(record.benchmark.status if record.benchmark is not None else "not-run"),
        minimum_speedup=(
            record.benchmark.minimum_marginal_speedup if record.benchmark is not None else None
        ),
        minimum_overall_speedup=(
            record.benchmark.minimum_overall_speedup if record.benchmark is not None else None
        ),
        baseline_median_seconds=(
            record.benchmark.baseline_median_seconds if record.benchmark is not None else None
        ),
        unplanned_median_seconds=(
            record.benchmark.unplanned_median_seconds if record.benchmark is not None else None
        ),
        planned_median_seconds=(
            record.benchmark.planned_median_seconds if record.benchmark is not None else None
        ),
        marginal_speedup=(
            record.benchmark.marginal_speedup if record.benchmark is not None else None
        ),
        overall_speedup=(
            record.benchmark.overall_speedup if record.benchmark is not None else None
        ),
        cache_status=record.cache_status,
        payload_files=record.staged.payload_files,
    )


def _replace_payload_transactionally(
    *,
    current_root: Path,
    candidate_root: Path,
    backup_root: Path,
) -> None:
    """Atomically swap a passing candidate payload with rollback on rename failure.

    Args:
        current_root: Current accepted payload.
        candidate_root: Passing disposable candidate payload.
        backup_root: Temporary rollback location on the same filesystem.

    Raises:
        OSError: If the payload cannot be replaced or rolled back.
    """
    _remove_tree(backup_root)
    current_root.rename(backup_root)
    try:
        candidate_root.rename(current_root)
    except OSError:
        backup_root.rename(current_root)
        raise
    shutil.rmtree(backup_root, ignore_errors=True)


def _execution_plan_benchmark_progress(
    progress: PackageProgress | None,
    plan_id: str,
    event: ExecutionPlanBenchmarkProgress,
) -> None:
    """Render one marginal execution-plan benchmark event.

    Args:
        progress: Optional package progress callback.
        plan_id: Plan measured by the event.
        event: Three-arm runtime benchmark phase notification.
    """
    _progress(
        progress,
        (
            f"execution-plan benchmark {plan_id} {event.phase} trio {event.trio_index} "
            f"{event.arm} completed in {event.duration_seconds:.2f}s"
        ),
    )


def _execution_plan_benchmark_timings(
    plan_id: str,
    result: ExecutionPlanBenchmarkResult,
) -> tuple[CompilePhaseTiming, ...]:
    """Convert marginal execution-plan measurements into compile phase timings.

    Args:
        plan_id: Plan measured by the benchmark.
        result: Three-arm benchmark decision and child-process evidence.

    Returns:
        tuple[CompilePhaseTiming, ...]: Ordered warmup and sample timings.
    """
    return tuple(
        CompilePhaseTiming(
            name="execution_plan_benchmark",
            duration_seconds=sample.run.duration_seconds,
            detail=f"{plan_id}; {phase}; {sample.arm}; {result.status}",
        )
        for phase, samples in (("warmup", result.warmups), ("sample", result.samples))
        for sample in samples
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
                    minimum_over_unfused=DEFAULT_MINIMUM_MARGINAL_SPEEDUP,
                    minimum_overall=config.minimum_speedup,
                ),
                project_root=context.baseline.quality_project_root,
                baseline_payload_root=context.baseline.baseline_install_root,
                unfused_payload_root=context.install_root,
                fused_payload_root=fused_root,
                unfused_variant_allowlist=accepted_ids,
                fused_variant_allowlist=accepted_ids,
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


def _materialize_candidate_payload(
    context: _ProfitabilitySelectionContext,
    selected: tuple[_PreparedTypedRegion, ...],
    destination: Path,
) -> None:
    """Create the exact payload represented by one candidate allowlist.

    Candidate benchmarks must measure the same shim and artifact layout that a
    promoted wheel receives. Merely disabling superset variants at runtime
    leaves their generated shim declarations on the import path and can make a
    profitable candidate appear slower than its final materialized payload.

    Args:
        context: Baseline wheel and staged superset inputs.
        selected: Native variants that should exist in the disposable payload.
        destination: New temporary payload root to materialize.

    Raises:
        OSError: If baseline files, generated shims, or artifacts cannot be copied.
        ValueError: If a selected source or artifact is outside the staged roots,
            or the baseline wheel omitted a selected source module.
    """
    baseline_root = context.baseline.baseline_install_root
    if baseline_root is None:
        raise ValueError("candidate payload requires an unpacked baseline wheel")
    copy_source_snapshot(baseline_root, destination)

    configs_by_path: dict[Path, list[RegionShimConfig]] = {}
    for item in selected:
        configs_by_path.setdefault(item.shim.source_path, []).append(item.shim)
    for source_path, configs in configs_by_path.items():
        relative = _source_relative_path(source_path, context.staged_source_roots)
        target = destination / relative
        if not target.is_file():
            raise ValueError(
                f"target PEP 517 wheel omitted a compiled source module: {relative.as_posix()}"
            )
        source_text = source_path.read_text(encoding="utf-8")
        target.write_text(
            insert_or_replace_region_shim(source_text, tuple(configs)).new_text,
            encoding="utf-8",
        )

    artifact_dirs = tuple(dict.fromkeys(item.shim.artifact_dir for item in selected))
    for artifact_dir in artifact_dirs:
        relative = _source_relative_path(artifact_dir, context.staged_source_roots)
        copy_source_snapshot(artifact_dir, destination / relative)
    _clear_payload_bytecode((destination,))


def _run_exact_candidate_trial(
    context: _ProfitabilitySelectionContext,
    accepted: tuple[_PreparedTypedRegion, ...],
    candidate: _PreparedTypedRegion,
    minimum_speedup: float,
) -> tuple[CommandRunEvidence, BenchmarkGateResult | None]:
    """Test one candidate against exact accepted and proposed payloads.

    Native artifacts are reused from the superset build, but each arm receives
    only its selected shim declarations and artifact directories. This keeps
    marginal timing equivalent to final-wheel routing without another compiler
    invocation.

    Args:
        context: Profile-guided selection boundaries and staged build state.
        accepted: Variants retained before this trial.
        candidate: Proposed additional variant.
        minimum_speedup: Marginal speedup required for acceptance.

    Returns:
        tuple[CommandRunEvidence, BenchmarkGateResult | None]: Semantic evidence
        and paired timing evidence when semantics pass.

    Raises:
        ValueError: If required commands or payload roots are unavailable.
    """
    test_command = context.project.config.compile.test_command
    benchmark_command = context.project.config.compile.benchmark_command
    quality_root = context.baseline.quality_project_root
    if test_command is None or benchmark_command is None or quality_root is None:
        raise ValueError("candidate selection prerequisites are unavailable")

    selected = (*accepted, candidate)
    accepted_ids = frozenset(item.unit.region_id for item in accepted)
    variant_id = candidate.unit.region_id
    with tempfile.TemporaryDirectory(
        prefix="atoll-candidate-payloads-",
        dir=context.payload_root.parent,
    ) as workspace_text:
        workspace = Path(workspace_text)
        accepted_payload = workspace / "accepted"
        candidate_payload = workspace / "candidate"
        _materialize_candidate_payload(context, accepted, accepted_payload)
        _materialize_candidate_payload(context, selected, candidate_payload)
        semantic = run_performance_command(
            test_command,
            project_root=quality_root,
            payload_root=candidate_payload,
            mode="compiled",
        )
        if not semantic.succeeded:
            return semantic, None
        _progress(
            context.progress,
            (f"candidate benchmarking {variant_id} against {len(accepted)} accepted variant(s)"),
        )
        benchmark = run_benchmark_gate(
            BenchmarkGateConfig(
                command=benchmark_command,
                warmups=_CANDIDATE_BENCHMARK_WARMUPS,
                samples=_CANDIDATE_BENCHMARK_SAMPLES,
                minimum_speedup=minimum_speedup,
            ),
            project_root=quality_root,
            baseline_payload_root=accepted_payload,
            compiled_payload_root=candidate_payload,
            baseline_variant_allowlist=(accepted_ids if accepted else None),
            progress=partial(_candidate_benchmark_progress, context.progress, variant_id),
        )
        return semantic, benchmark


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

    Raises:
        AssertionError: If a successful semantic trial omits benchmark evidence.
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
        minimum_speedup = (
            prepared.minimum_marginal_speedup
            if prepared.minimum_marginal_speedup is not None
            else _CANDIDATE_MINIMUM_SPEEDUP
        )
        baseline_median_seconds: float | None = None
        candidate_median_seconds: float | None = None
        baseline_ids = tuple(item.unit.region_id for item in accepted)
        trial_ids = (*baseline_ids, variant_id)
        _progress(
            context.progress,
            f"candidate {index}/{candidate_count} testing {variant_id} semantics",
        )
        semantic, benchmark = _run_exact_candidate_trial(
            context,
            tuple(accepted),
            prepared,
            minimum_speedup,
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
            if benchmark is None:
                raise AssertionError("successful candidate semantics omitted benchmark evidence")
            timings.extend(_candidate_benchmark_timings(variant_id, benchmark))
            benchmark_status = benchmark.status
            marginal_speedup = benchmark.speedup
            baseline_median_seconds = benchmark.baseline_median_seconds
            candidate_median_seconds = benchmark.compiled_median_seconds
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
                baseline_median_seconds=baseline_median_seconds,
                candidate_median_seconds=candidate_median_seconds,
                minimum_speedup=minimum_speedup,
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
    """Order attributed private helpers before profile-selected public bindings.

    Args:
        successful: Successfully compiled variants available for trials.
        skipped: Backend failures retained as fallback explanations.
        profile: Dynamic profile with a descending-hotness selected-symbol order.

    Returns:
        tuple[_ProfitabilityCandidate, ...]: Deduplicated candidates with explicit
        orchestration attribution first, then ordinary profile order.
    """
    profile_samples = {member.symbol: member.samples for member in profile.members}
    selected = frozenset(profile.selected_symbols)
    by_symbol: dict[SymbolId, list[_PreparedTypedRegion]] = {}
    for prepared in successful:
        for symbol in dict.fromkeys(binding.source for binding in prepared.generation.bindings):
            if symbol in selected:
                by_symbol.setdefault(symbol, []).append(prepared)
    ordered: list[_ProfitabilityCandidate] = []
    seen: set[str] = set()

    def append_candidate(
        prepared: _PreparedTypedRegion,
        represented: tuple[SymbolId, ...],
    ) -> None:
        variant_id = prepared.unit.region_id
        if variant_id in seen:
            return
        seen.add(variant_id)
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

    for prepared in successful:
        if prepared.profitability_symbols:
            append_candidate(prepared, prepared.profitability_symbols)
    for symbol in profile.selected_symbols:
        for prepared in by_symbol.get(symbol, ()):
            represented = tuple(
                symbol
                for symbol in dict.fromkeys(
                    binding.source for binding in prepared.generation.bindings
                )
                if symbol in selected
            )
            append_candidate(prepared, represented)
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
    """Describe the ordered rejection or bypass chain preceding a fallback.

    Args:
        rejected: Prepared variants and failed or bypassed attempts in retry order.

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
    payload_roots: tuple[Path, ...] = (compiled_payload_root,)
    if baseline.baseline_install_root is not None:
        payload_roots = (baseline.baseline_install_root, *payload_roots)
    _clear_payload_bytecode_with_progress(payload_roots, progress)
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


def _scalar_analyses(
    scans: tuple[ModuleScan, ...],
    progress: PackageProgress | None,
) -> tuple[ScalarAnalysisResult, ...]:
    """Derive fixed-width proof candidates without changing generic selection.

    The proof frontend runs before backend selection and again after an accepted
    source patch is materialized. Milestone-specific native lowering consumes
    this evidence later; ordinary typed-region selection remains independent.

    Args:
        scans: Enriched module scans in compile order.
        progress: Optional CLI progress callback.

    Returns:
        tuple[ScalarAnalysisResult, ...]: Per-module scalar plans and explicit fallbacks.
    """
    analyses = tuple(analyze_scalar_scan(scan) for scan in scans)
    plan_count = sum(len(analysis.plans) for analysis in analyses)
    rejection_count = sum(len(analysis.rejections) for analysis in analyses)
    _progress(
        progress,
        (
            f"scalar analysis proved {plan_count} callable(s); "
            f"{rejection_count} callable(s) retained Python fallback"
        ),
    )
    return analyses


def _call_chain_analyses(
    scans: tuple[ModuleScan, ...],
    progress: PackageProgress | None,
) -> tuple[CallChainAnalysisResult, ...]:
    """Derive direct native call-chain candidates without changing selection.

    Args:
        scans: Enriched module scans in compile order.
        progress: Optional CLI progress callback.

    Returns:
        tuple[CallChainAnalysisResult, ...]: Per-module plans and explicit fallbacks.
    """
    analyses = tuple(analyze_call_chain_scan(scan) for scan in scans)
    plan_count = sum(len(analysis.plans) for analysis in analyses)
    rejection_count = sum(len(analysis.rejections) for analysis in analyses)
    _progress(
        progress,
        (
            f"call-chain analysis proved {plan_count} root(s); "
            f"{rejection_count} root(s) retained Python dispatch"
        ),
    )
    return analyses


def _buffer_analyses(
    scans: tuple[ModuleScan, ...],
    progress: PackageProgress | None,
) -> tuple[BufferAnalysisResult, ...]:
    """Derive zero-copy standard-buffer candidates without changing selection.

    Args:
        scans: Enriched module scans in compile order.
        progress: Optional CLI progress callback.

    Returns:
        tuple[BufferAnalysisResult, ...]: Per-module plans and explicit fallbacks.
    """
    analyses = tuple(analyze_buffer_scan(scan) for scan in scans)
    plan_count = sum(len(analysis.plans) for analysis in analyses)
    rejection_count = sum(len(analysis.rejections) for analysis in analyses)
    _progress(
        progress,
        (
            f"buffer analysis proved {plan_count} zero-copy kernel(s); "
            f"{rejection_count} callable(s) retained Python buffer semantics"
        ),
    )
    return analyses


def _call_chain_profile_targets(
    analyses: tuple[CallChainAnalysisResult, ...],
) -> tuple[ProfileCallEdgeTarget, ...]:
    """Return deduplicated exact source sites for direct call-edge counting.

    Args:
        analyses: Static call-chain plans whose edges may be hot at runtime.

    Returns:
        tuple[ProfileCallEdgeTarget, ...]: Stable full-span profiling targets.
    """
    targets: dict[str, ProfileCallEdgeTarget] = {}
    for analysis in analyses:
        for plan in analysis.plans:
            for edge in plan.edges:
                identity = (
                    f"{edge.caller.stable_id}>{edge.callee.stable_id}:"
                    f"{edge.lineno}:{edge.col_offset}:{edge.end_lineno}:{edge.end_col_offset}"
                )
                target_id = (
                    f"call-edge-{hashlib.blake2b(identity.encode(), digest_size=12).hexdigest()}"
                )
                targets.setdefault(
                    target_id,
                    ProfileCallEdgeTarget(
                        id=target_id,
                        owner=edge.caller,
                        callee=edge.callee,
                        lineno=edge.lineno,
                        col_offset=edge.col_offset,
                        end_lineno=edge.end_lineno,
                        end_col_offset=edge.end_col_offset,
                    ),
                )
    return tuple(targets.values())


def _profiled_call_chain_roots(
    analyses: tuple[CallChainAnalysisResult, ...],
    profile: ProfileResult | None,
) -> tuple[SymbolId, ...]:
    """Rank hot caller roots by exact direct-edge invocation counts.

    Args:
        analyses: Current static call-chain plans.
        profile: Current-invocation profile evidence, when configured.

    Returns:
        tuple[SymbolId, ...]: At most four hot public roots in descending count order.
    """
    if profile is None or profile.status != "profiled":
        return ()
    counts = {
        (
            item.target.owner,
            item.target.callee,
            item.target.lineno,
            item.target.col_offset,
            item.target.end_lineno,
            item.target.end_col_offset,
        ): item.invocation_count
        for item in profile.call_edges
    }
    root_counts: dict[SymbolId, int] = {}
    for analysis in analyses:
        for plan in analysis.plans:
            direct_count = sum(
                counts.get(
                    (
                        edge.caller,
                        edge.callee,
                        edge.lineno,
                        edge.col_offset,
                        edge.end_lineno,
                        edge.end_col_offset,
                    ),
                    0,
                )
                for edge in plan.edges
                if edge.caller == plan.root
            )
            if direct_count > 0:
                root_counts[plan.root] = max(root_counts.get(plan.root, 0), direct_count)
    ranked = sorted(root_counts, key=lambda item: (-root_counts[item], item.stable_id))
    return tuple(ranked[:_MAX_PROFILED_CALL_CHAIN_ROOTS])


def _profile_support_roots(
    profile: ProfileResult,
    analyses: tuple[CallChainAnalysisResult, ...],
) -> tuple[SymbolId, ...]:
    """Return only observed roots that can influence profile selection.

    Whole-project support assessment is expensive because each callable needs
    a directed dependency slice and backend capability check. A successful
    profile can select only members it observed or roots promoted by observed
    direct call edges, so unrelated cold callables cannot affect backfilling.

    Args:
        profile: Current successful baseline profile.
        analyses: Static call-chain evidence used for root promotion.

    Returns:
        tuple[SymbolId, ...]: Deduplicated observed members and hot call-chain
        roots in deterministic profile order.
    """
    return tuple(
        dict.fromkeys(
            (
                *(member.symbol for member in profile.members),
                *_profiled_call_chain_roots(analyses, profile),
            )
        )
    )


def _select_profile_with_call_chains(
    profile: ProfileResult,
    scans: tuple[ModuleScan, ...],
    analyses: tuple[CallChainAnalysisResult, ...],
    backends: tuple[Backend, ...],
    *,
    support: _ProfileCandidateSupport | None = None,
) -> ProfileResult:
    """Add exact-edge hot callers to ordinary leaf-sample profile selection.

    Args:
        profile: Current-invocation baseline profile.
        scans: Current static symbols used for ordinary candidate mapping.
        analyses: Current call-chain plans used for edge-root promotion.
        backends: Configured compiler backends in preference order.
        support: Reusable capability assessment for the current scans.

    Returns:
        ProfileResult: Candidate selection including hot public call-chain roots.
    """
    support = support or _profile_candidate_support(scans, backends)
    supported = frozenset(support.supported)
    rejected = {item.symbol: item.reason for item in support.rejected}
    call_chain_roots = tuple(
        root for root in _profiled_call_chain_roots(analyses, profile) if root in supported
    )
    selected = select_profile_candidates(
        profile,
        tuple(symbol for scan in scans for symbol in scan.symbols if symbol.id in supported),
    )
    decisions = tuple(
        replace(
            decision,
            symbol=observed,
            reason=rejected[observed],
            selected=False,
        )
        if decision.symbol is None
        and (observed := SymbolId(decision.module, decision.qualname)) in rejected
        else decision
        for decision in selected.candidates
    )
    if not call_chain_roots:
        return replace(selected, candidates=decisions)
    return replace(
        selected,
        status="profiled",
        reason="baseline profile collected with hot direct call-edge roots",
        candidates=decisions,
        selected_symbols=tuple(dict.fromkeys((*selected.selected_symbols, *call_chain_roots))),
    )


def _stabilize_profile_compile_selection(
    selection_scope: _ProfileCompileSelectionScope,
    *,
    options: PackageOptions,
    project: DiscoveredProject,
    scans: tuple[ModuleScan, ...],
    profile: ProfileResult | None,
) -> ProfileResult | None:
    """Replay the first strict native candidate plan for an unchanged scope.

    Profiling and source/execution-plan discovery run before this boundary on
    every invocation. The cache controls only which native bindings are offered
    to compilation, preventing statistical sample jitter from creating new
    cold artifact variants on an otherwise identical warm build. Semantic and
    profitability gates still evaluate the replayed selection from scratch.

    Args:
        selection_scope: Stable arm identity and previously computed backend support.
        options: Current package options, including an optional cache override.
        project: Active original or transformed project configuration.
        scans: Active source scans whose content and symbols bound replay.
        profile: Fresh current-invocation profile and candidate selection.

    Returns:
        ProfileResult | None: Fresh profile evidence with a strictly replayed native
            selection on a cache hit, otherwise the current selection.
    """
    benchmark = project.config.compile.benchmark_command
    if (
        profile is None
        or profile.status != "profiled"
        or not profile.selected_symbols
        or benchmark is None
    ):
        return profile
    available = frozenset(selection_scope.support.supported)
    try:
        identity = ProfilePlanIdentity(
            scope_identity=selection_scope.identity,
            candidate_identity=tuple(sorted(symbol.stable_id for symbol in available)),
            module_source_hashes=tuple(
                sorted(
                    (
                        scan.module.name,
                        hashlib.sha256(scan.module.path.read_bytes()).hexdigest(),
                    )
                    for scan in scans
                )
            ),
            benchmark_argv=benchmark,
            backend_order=tuple(project.config.compile.backends),
            module_scope=options.module_name,
            python_cache_tag=(
                sys.implementation.cache_tag
                or f"{sys.implementation.name}-{sys.version_info.major}{sys.version_info.minor}"
            ),
            python_platform=next(tags.sys_tags()).platform,
            cache_format_version=_PROFILE_PLAN_CACHE_FORMAT_VERSION,
            lowering_version=_PROFILE_PLAN_LOWERING_VERSION,
        )
        decision = select_profile_plan(
            options.cache_dir or project.config.cache_dir,
            identity,
            profile.selected_symbols,
            available,
        )
    except (OSError, ValueError) as error:
        _progress(options.progress, f"profile compile plan cache unavailable: {error}")
        return profile
    _progress(
        options.progress,
        (
            f"profile compile plan cache {decision.status}: "
            f"{len(decision.selection)} member(s); {decision.diagnostic}"
        ),
    )
    if decision.status != "hit":
        return profile
    return _profile_with_replayed_compile_selection(profile, decision)


def _profile_with_replayed_compile_selection(
    profile: ProfileResult,
    decision: ProfilePlanDecision,
) -> ProfileResult:
    """Make cached native selection explicit in otherwise fresh profile evidence.

    Args:
        profile: Current profile whose samples and lifecycle evidence remain authoritative.
        decision: Strict cache hit containing the first ordered native selection.

    Returns:
        ProfileResult: Profile with replayed selection, current-run coverage,
            and candidate reasons that distinguish replay from fresh ranking.
    """
    selected = frozenset(decision.selection)
    members = {member.symbol: member for member in profile.members}
    seen: set[SymbolId] = set()
    candidates: list[MappedCandidateDecision] = []
    for candidate in profile.candidates:
        if candidate.symbol in selected:
            seen.add(candidate.symbol)
            candidates.append(replace(candidate, selected=True, reason="cache-replayed"))
        elif candidate.selected:
            candidates.append(replace(candidate, selected=False, reason="cache-replay-excluded"))
        else:
            candidates.append(candidate)
    for symbol in decision.selection:
        if symbol in seen:
            continue
        member = members.get(symbol)
        samples = member.samples if member is not None else 0
        scheduler_samples = member.scheduler_overhead_samples if member is not None else 0
        attributed_samples = samples + scheduler_samples
        candidates.append(
            MappedCandidateDecision(
                symbol=symbol,
                module=symbol.module,
                qualname=symbol.qualname,
                samples=samples,
                coverage=member.coverage if member is not None else 0.0,
                scheduler_overhead_samples=scheduler_samples,
                attributed_samples=attributed_samples,
                attributed_coverage=_sample_coverage(
                    attributed_samples,
                    profile.total_samples,
                ),
                selected=True,
                reason="cache-replayed",
            )
        )
    selected_samples = sum(
        member.attributed_samples for member in profile.members if member.symbol in selected
    )
    return replace(
        profile,
        reason=(
            f"{profile.reason}; native candidate selection replayed from strict cache "
            f"{decision.identity_digest[:12]}"
        ),
        selected_hot_samples=selected_samples,
        selected_hot_coverage=_sample_coverage(
            selected_samples,
            profile.mapped_project_samples + profile.scheduler_overhead_samples,
        ),
        candidates=tuple(candidates),
        selected_symbols=decision.selection,
    )


def _profile_candidate_support(
    scans: tuple[ModuleScan, ...],
    backends: tuple[Backend, ...],
    *,
    roots: tuple[SymbolId, ...] | None = None,
) -> _ProfileCandidateSupport:
    """Assess callable roots that can participate in dynamic hotness ranking.

    The profile should rank only roots Atoll can independently bind through a
    complete directed closure. Unsupported roots remain explicit report
    evidence instead of appearing unmapped or causing a later selection crash.

    Args:
        scans: Selected module scans in deterministic source order.
        backends: Backends considered in configured preference order.
        roots: Profile-observed roots to assess. ``None`` retains exhaustive
            static behavior for unsupported profiling launchers.

    Returns:
        _ProfileCandidateSupport: Supported roots and stable capability rejections.
    """
    supported: list[SymbolId] = []
    rejected: list[_ProfileCandidateRejection] = []
    requested = None if roots is None else frozenset(roots)
    for scan in scans:
        for region in scan.typed_regions:
            decisions = {decision.target: decision for decision in region.decisions}
            all_members = frozenset(member.id for member in region.members)
            eligible = _eligible_typed_callables(region, decisions, hot=all_members)
            specialization_roots = frozenset(
                specialization.source_member for specialization in region.specializations
            )
            for member in region.members:
                if member.kind not in {"function", "method"}:
                    continue
                root = member.id
                if requested is not None and root not in requested:
                    continue
                variants = _selected_requested_root_variants(
                    scan=scan,
                    source_region=region,
                    root=root,
                    hot=all_members,
                    backends=backends,
                )
                if variants:
                    supported.append(root)
                    continue
                reason: CandidateDecisionReason = (
                    "backend-unsupported"
                    if root in eligible or root in specialization_roots
                    else "not-independently-bindable"
                )
                rejected.append(_ProfileCandidateRejection(symbol=root, reason=reason))
    return _ProfileCandidateSupport(
        supported=tuple(dict.fromkeys(supported)),
        rejected=tuple(rejected),
    )


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
        variants.extend(
            _selected_requested_root_variants(
                scan=scan,
                source_region=source_region,
                root=root,
                hot=hot,
                backends=backends,
            )
        )
    return tuple(variants)


def _selected_requested_root_variants(
    *,
    scan: ModuleScan,
    source_region: TypedRegion,
    root: SymbolId,
    hot: frozenset[SymbolId],
    backends: tuple[Backend, ...],
) -> tuple[_SelectedTypedRegion, ...]:
    """Select a callable slice or concrete specialization for one public root.

    Eligibility is checked against the connected source region before directed
    slicing. This keeps low-level slice invariants strict while turning ordinary
    capability exclusions, such as dunder methods, into reportable selection
    rejections rather than exceptions.

    Args:
        scan: Module scan retaining source facts for generated code.
        source_region: Connected source region that owns ``root``.
        root: Explicit or profile-derived public binding.
        hot: Profile-selected roots allowed to use boxed lowering.
        backends: Backends considered in configured preference order.

    Returns:
        tuple[_SelectedTypedRegion, ...]: At most one ordinary backend variant,
            followed by any supported concrete specialization when needed.
    """
    decisions = {decision.target: decision for decision in source_region.decisions}
    eligible = _eligible_typed_callables(source_region, decisions, hot=hot)
    if root in eligible:
        sliced = build_directed_region_slice(source_region, root)
        sliced_decisions = {decision.target: decision for decision in sliced.decisions}
        sliced_eligible = _eligible_typed_callables(sliced, sliced_decisions, hot=hot)
        closure = _runtime_member_closure(sliced, sliced_eligible, frozenset({root}))
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
        if callable_variants:
            return callable_variants
    return tuple(
        variant
        for specialization in source_region.specializations
        if specialization.source_member == root
        for variant in (
            _selected_specialization_variant(scan, source_region, specialization, backends),
        )
        if variant is not None
    )


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
    owner_class = class_members[0].id.qualname
    if any(
        member.owner_class != owner_class or member.execution_kind != "sync"
        for member in method_members
    ):
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
    try:
        tree = ast.parse(textwrap.dedent(source_text))
    except SyntaxError:
        return True
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


def _stage_target_sources(
    project: DiscoveredProject,
    build_root: Path,
    progress: PackageProgress | None,
) -> tuple[tuple[Path, ...], str]:
    """Copy target source roots and fingerprint their source-clean originals.

    Args:
        project: Discovered target whose original source roots remain source-clean.
        build_root: Disposable destination for native generation and staged shims.
        progress: Optional user-facing compile progress callback.

    Returns:
        tuple[tuple[Path, ...], str]: Copied roots and strict backend cache digest.
    """
    copy_started = time.perf_counter()
    _progress(progress, "copying source roots into temporary build tree")
    staged = _copy_source_roots(project, build_root)
    _progress(progress, f"copied source roots in {_duration(copy_started)}")
    fingerprint_started = time.perf_counter()
    digest = _source_roots_digest(
        staged,
        ignored_top_level=_STAGED_SOURCE_DIGEST_IGNORED_TOP_LEVEL,
    )
    _progress(
        progress,
        f"fingerprinted target sources in {_duration(fingerprint_started)}",
    )
    return staged, digest


def _source_roots_digest(
    source_roots: tuple[Path, ...],
    *,
    ignored_top_level: frozenset[str] = frozenset(),
) -> str:
    """Hash source-clean target inputs before Atoll generates private source.

    The digest covers every regular file path and byte sequence under each
    ordered source root. It is included in backend cache keys so a deterministic
    rejection caused by imported project source cannot survive a source edit.

    Args:
        source_roots: Ordered copied import roots before native generation.
        ignored_top_level: Disposable sibling names present only in a flat build root.

    Returns:
        str: Lowercase SHA-256 digest of copied target inputs.
    """
    digest = hashlib.sha256()
    for index, source_root in enumerate(source_roots):
        digest.update(f"root:{index}".encode("ascii"))
        digest.update(b"\0")
        for path in sorted(source_root.rglob("*")):
            relative = path.relative_to(source_root).as_posix()
            relative_parts = PurePosixPath(relative).parts
            if (
                (relative_parts and relative_parts[0] in ignored_top_level)
                or any(part in _GENERATED_DIR_NAMES for part in relative_parts)
                or path.suffix
                in {
                    ".pyd",
                    ".so",
                }
            ):
                continue
            if path.is_symlink():
                kind = b"symlink"
                content_digest = symlink_target_bytes(path)
            elif path.is_file():
                kind = b"file"
                content_digest = hashlib.sha256(path.read_bytes()).digest()
            else:
                continue
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(kind)
            digest.update(b"\0")
            digest.update(content_digest)
    return digest.hexdigest()


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
            copy_source_snapshot(source_root, destination, ignore=_copy_ignore)
        staged_roots.append(destination)
    return tuple(staged_roots)


def _copy_if_different(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    shutil.copy2(source, destination)


def _clear_payload_bytecode(roots: tuple[Path, ...]) -> tuple[Path, ...]:
    """Remove interpreter caches that would bias payload timing or wheel contents.

    Verification subprocesses and third-party build hooks may import staged
    modules before the performance gate. Baseline and compiled measurements
    must not differ merely because one tree already contains bytecode caches.
    Symlink entries are unlinked without following their targets.

    Args:
        roots: Owned unpacked wheel or disposable candidate payload roots.

    Returns:
        tuple[Path, ...]: Removed cache directories and standalone bytecode files.
    """
    removed: list[Path] = []
    for root in dict.fromkeys(path.resolve() for path in roots):
        if not root.is_dir():
            continue
        cache_dirs = sorted(
            root.rglob("__pycache__"),
            key=lambda path: len(path.parts),
            reverse=True,
        )
        for cache_dir in cache_dirs:
            if cache_dir.is_symlink():
                cache_dir.unlink(missing_ok=True)
            elif cache_dir.is_dir():
                shutil.rmtree(cache_dir)
            else:
                continue
            removed.append(cache_dir)
        for pattern in ("*.pyc", "*.pyo"):
            for bytecode in root.rglob(pattern):
                if bytecode.is_file() or bytecode.is_symlink():
                    bytecode.unlink(missing_ok=True)
                    removed.append(bytecode)
    return tuple(removed)


def _clear_payload_bytecode_with_progress(
    roots: tuple[Path, ...],
    progress: PackageProgress | None,
) -> None:
    """Clear timing-affecting caches and report only when stale paths existed.

    Args:
        roots: Owned payload roots scrubbed before verification or timing.
        progress: Optional compile progress callback.
    """
    removed = _clear_payload_bytecode(roots)
    if removed:
        _progress(
            progress,
            f"removed {len(removed)} pre-existing bytecode cache path(s)",
        )


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


def _promotion_wheel_tag(
    context: _SourceCleanPromotionContext,
    baseline_wheel_path: Path,
) -> str:
    """Choose a native tag only when the overlay contains native artifacts.

    Args:
        context: Promotion state containing verified artifact records.
        baseline_wheel_path: Normal target wheel whose existing tag is preserved.

    Returns:
        str: Native interpreter tag or the baseline wheel's existing three-part tag.

    Raises:
        WheelOverlayError: If the baseline filename does not contain a valid wheel tag.
    """
    if context.verification_plan.artifacts:
        return _wheel_tag()
    name = baseline_wheel_path.name
    if not name.endswith(".whl"):
        raise WheelOverlayError(f"baseline wheel has an invalid filename: {name}")
    parts = name.removesuffix(".whl").rsplit("-", maxsplit=3)
    if len(parts) != _WHEEL_TAG_COMPONENT_COUNT + 1:
        raise WheelOverlayError(f"baseline wheel tag is unavailable: {name}")
    return "-".join(parts[-_WHEEL_TAG_COMPONENT_COUNT:])


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


def _source_clean_output_paths(
    root: Path,
    output_dir: Path | None,
) -> tuple[Path, Path, Path]:
    """Resolve the persistent output and disposable source-clean roots.

    Args:
        root: Target project root used for the default Atoll directory.
        output_dir: Optional absolute or project-relative wheel destination.

    Returns:
        tuple[Path, Path, Path]: Output directory, build root, and install root.
    """
    resolved = _resolve_output_dir(root, output_dir)
    return resolved, resolved / "build", resolved / "install"


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
        if item.is_symlink():
            shutil.copy2(item, target, follow_symlinks=False)
        elif item.is_dir():
            copy_source_snapshot(item, target, ignore=_copy_ignore)
        else:
            shutil.copy2(item, target)


def _copy_pep517_project(
    source: Path,
    destination: Path,
    *,
    excluded_output: Path,
) -> None:
    """Copy complete build inputs while excluding Atoll state and native residue.

    The generated ``.git`` pointer is excluded from the baseline-wheel tree
    digest. Writing it still changes the destination root directory metadata,
    so the source root metadata is restored afterward to keep repeated copies
    content-addressable.

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

    copy_source_snapshot(source_root, destination, ignore=ignore)
    _write_gitdir_pointer(source_root, destination)
    shutil.copystat(source_root, destination)


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
