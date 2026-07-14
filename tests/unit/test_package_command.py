"""Unit tests for source-clean package artifact helpers."""

from __future__ import annotations

import ast
import hashlib
import importlib.machinery
import os
import shutil
import zipfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Protocol, cast

import pytest

from atoll import cli as cli_module
from atoll.analysis.task_fusion import FusionPlan
from atoll.baseline_cache import BASELINE_WHEEL_CACHE_CONTEXT_ENV, baseline_wheel_cache_key
from atoll.commands import package as package_command
from atoll.execution_plans.models import ExecutionPlanTrial
from atoll.generation.region_shim import RegionShimConfig
from atoll.models import (
    ArtifactRecord,
    ArtifactRole,
    Backend,
    BackendAssessment,
    BackendCompileContext,
    BackendCompileResult,
    BindingTarget,
    Blocker,
    CompilationUnit,
    CompileAttempt,
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
from atoll.native_optimization.buffer_analysis import BufferAnalysisResult
from atoll.native_optimization.call_chains import CallChainAnalysisResult
from atoll.native_optimization.run_guard import CompletionIndexNativePlan, RunGuardNativePlan
from atoll.profile_plan_cache import ProfilePlanDecision
from atoll.project import DiscoveredProject, discover_project
from atoll.report import CompilationReportInput, build_compilation_report
from atoll.runtime.fusion_performance import (
    FusionArmRunEvidence,
    FusionBenchmarkConfig,
    FusionTrial,
)
from atoll.runtime.package_verify import (
    PackageVerificationPlan,
    PackageVerificationResult,
    VerificationStage,
)
from atoll.runtime.performance import (
    BenchmarkGateConfig,
    BenchmarkGateResult,
    BenchmarkProgress,
    BenchmarkStatus,
    CommandRunEvidence,
    RuntimeMode,
)
from atoll.runtime.profiling import (
    LifecycleCounts,
    MappedCandidateDecision,
    ProfileCallEdgeTarget,
    ProfiledCallEdge,
    ProfiledMember,
    ProfileResult,
    select_profile_candidates,
    unconfigured_profile,
)
from atoll.source_optimization.analysis import SourceOptimizationPlanningResult
from atoll.source_optimization.models import SourceOptimizationTrial
from atoll.source_optimization.search import (
    SourceOptimizationSearchOptions,
    SourceOptimizationSearchResult,
)
from atoll.source_optimization.transforms import (
    GeneratedSourcePatch,
    TransformedSourceFile,
)
from atoll.wheel_overlay import WheelOverlayError

FIXTURE_ROOT = Path("tests/fixtures/simple_project")
TYPED_FIXTURE_ROOT = Path("tests/fixtures/typed_region_project")
NATIVE_FIXTURE_ROOT = Path("tests/fixtures/native_optimization_project")
EXPECTED_ATOMIC_SELECTION_COUNT = 2
TEST_FAILURE_RETURN_CODE = 9
RANKING_BINDING_COUNT = 3
OUTLINED_COMPILE_CALL_COUNT = 2
_CANDIDATE_SPEEDUP = 1.01
EXPECTED_FINAL_TEST_RESULTS = 2
EXPECTED_CALL_CHAIN_WIDTH_COUNT = 2
EXPECTED_SINGLE_FAILURE = 1
EXPECTED_SAFETY_VERIFICATION_STEPS = 2
OWNER_PROFILE_SAMPLES = 40
SECOND_CANDIDATE_REGION_COUNT = 2
PROFILE_REPLAY_RANK_SAMPLES = 120


@pytest.fixture(autouse=True)
def stub_native_subprocess_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    """Package orchestration tests stop at the separately tested verification boundary."""

    def verify(**kwargs: object) -> PackageVerificationResult:
        stage = cast(VerificationStage, kwargs["stage"])
        target = cast(Path, kwargs["target"])
        return PackageVerificationResult(
            stage=stage,
            target=target,
            command=("python", "verify"),
            success=True,
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=0.0,
        )

    monkeypatch.setattr(package_command, "verify_package_subprocess", verify)


class _Metadata(Protocol):
    name: str
    version: str
    requires_python: str | None
    dependencies: tuple[str, ...]


class _BaselinePayloadFactory(Protocol):
    def __call__(
        self,
        *,
        wheel_path: Path | None,
        build: CompileAttempt,
        baseline_install_root: Path | None = None,
        quality_project_root: Path | None = None,
        semantic_test_result: CommandRunEvidence | None = None,
    ) -> object: ...


class _Pep517ProjectCopy(Protocol):
    def __call__(
        self,
        source: Path,
        destination: Path,
        *,
        excluded_output: Path,
    ) -> None:
        """Copy one stable PEP 517 build-input tree."""


class _BaselinePayloadView(Protocol):
    baseline_install_root: Path | None


class _OptimizationArmView(Protocol):
    active_project: DiscoveredProject
    baseline: _BaselinePayloadView
    source_search: SourceOptimizationSearchResult | None


class _QualityGateOutcomeView(Protocol):
    success: bool
    tests: tuple[CommandRunEvidence, ...]
    performance: BenchmarkGateResult
    error: str | None


class _PromotionResultView(Protocol):
    success: bool
    wheel_path: Path | None
    cleanup_removed: tuple[Path, ...]
    error: str | None


class _PromotionContextView(Protocol):
    verification_plan: PackageVerificationPlan
    requires_profitable_optimization: bool
    profitable_optimization_applied: bool


class _RuntimeSafetySelectionView(Protocol):
    successful: tuple[_PreparedTypedRegion, ...]
    failures: tuple[_TypedRegionFailure, ...]
    verification_steps: tuple[PackageVerificationResult, ...]
    overlay_error: str | None


class _FusionResearchOutcomeView(Protocol):
    trials: tuple[FusionTrial, ...]
    timings: tuple[CompilePhaseTiming, ...]


class _SelectedScans(Protocol):
    def __call__(
        self,
        project: DiscoveredProject,
        module_name: str | None,
        selected_members: tuple[SymbolId, ...] = (),
    ) -> tuple[ModuleScan, ...]: ...


class _TypedSelection(Protocol):
    scan: ModuleScan
    backend: Backend
    variant_id: str
    region: TypedRegion
    assessment: BackendAssessment
    members: tuple[SymbolId, ...]
    bound_members: tuple[SymbolId, ...] | None
    specialization: RegionSpecialization | None
    conditional_on_failure_of: str | None
    source_region_id: str | None
    slice_root: SymbolId | None


class _TypedGeneration(Protocol):
    backend: Backend
    region: TypedRegion
    bindings: tuple[BindingTarget, ...]


class _PreparedTypedRegion(Protocol):
    generation: _TypedGeneration
    assessment: BackendAssessment
    unit: CompilationUnit
    fallback: _PreparedTypedRegion | None
    conditional_on_failure_of: str | None
    lowering_mode: LoweringMode
    native_helpers: tuple[str, ...]
    fallback_reason: str | None
    shim: RegionShimConfig
    profitability_symbols: tuple[SymbolId, ...]


class _ProfitabilityCandidate(Protocol):
    prepared: _PreparedTypedRegion
    symbols: tuple[str, ...]
    profile_samples: int


class _ProfitabilitySelectionOutcomeView(Protocol):
    accepted: tuple[_PreparedTypedRegion, ...]
    trials: tuple[object, ...]


class _ProfileCandidateRejectionView(Protocol):
    symbol: SymbolId


class _ProfileCandidateSupportView(Protocol):
    supported: tuple[SymbolId, ...]
    rejected: tuple[_ProfileCandidateRejectionView, ...]


class _TypedRegionOutcome(Protocol):
    successful: tuple[_PreparedTypedRegion, ...]
    build: CompileAttempt
    artifacts: tuple[ArtifactRecord, ...]
    skipped: tuple[_TypedRegionFailure, ...]


class _TypedRegionFailure(Protocol):
    variant_id: str
    build: CompileAttempt


class _FakeCompileBackend:
    """Backend stub used to force deterministic retry orchestration."""

    def __init__(self, result: BackendCompileResult) -> None:
        self.result = result
        self.calls: list[tuple[CompilationUnit, ...]] = []
        self.name = cast(Backend, result.attempt.command[0])

    def fingerprint(
        self,
        unit: CompilationUnit,
        context: BackendCompileContext,
    ) -> str:
        """Return a stable per-variant key for cache orchestration tests."""
        _ = context
        return hashlib.sha256(
            f"{self.name}:{unit.region_id}:{unit.source_hash}".encode()
        ).hexdigest()

    def compile(
        self,
        units: tuple[CompilationUnit, ...],
        context: BackendCompileContext,
    ) -> BackendCompileResult:
        """Record one invocation and return configured compiler evidence."""
        _ = context
        self.calls.append(units)
        return self.result


class _SequencedCompileBackend:
    """Backend stub returning one configured result per distinct fallback attempt."""

    def __init__(
        self,
        name: Backend,
        results: tuple[BackendCompileResult, ...],
    ) -> None:
        self.name = name
        self.results = results
        self.calls: list[tuple[CompilationUnit, ...]] = []

    def fingerprint(
        self,
        unit: CompilationUnit,
        context: BackendCompileContext,
    ) -> str:
        """Return a stable key that distinguishes whole and outlined units."""
        _ = context
        return hashlib.sha256(
            f"{self.name}:{unit.region_id}:{unit.source_hash}".encode()
        ).hexdigest()

    def compile(
        self,
        units: tuple[CompilationUnit, ...],
        context: BackendCompileContext,
    ) -> BackendCompileResult:
        """Return the next result and reject unexpected additional invocations."""
        _ = context
        index = len(self.calls)
        if index >= len(self.results):
            raise AssertionError("native backend received an unexpected compile invocation")
        self.calls.append(units)
        return self.results[index]


class _ArtifactBatchCompileBackend:
    """Native test double that emits one cacheable artifact per submitted unit."""

    def __init__(self, name: Backend = "cython") -> None:
        self.name: Backend = name
        self.calls: list[tuple[CompilationUnit, ...]] = []

    def fingerprint(
        self,
        unit: CompilationUnit,
        context: BackendCompileContext,
    ) -> str:
        """Return a deterministic per-unit key independent of batch membership."""
        _ = context
        return hashlib.sha256(
            f"{self.name}:{unit.region_id}:{unit.source_hash}".encode()
        ).hexdigest()

    def compile(
        self,
        units: tuple[CompilationUnit, ...],
        context: BackendCompileContext,
    ) -> BackendCompileResult:
        """Record one physical invocation and produce partitionable artifacts."""
        self.calls.append(units)
        artifact_root = context.build_dir.parent / "artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        records: list[ArtifactRecord] = []
        for unit in units:
            artifact = artifact_root / f"{unit.logical_module}.so"
            artifact.write_bytes(unit.region_id.encode())
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            paths.append(artifact)
            records.append(
                ArtifactRecord(
                    region_id=unit.region_id,
                    backend=self.name,
                    logical_module=unit.logical_module,
                    role="primary",
                    install_relative_path=f"{unit.install_relative_dir}/{artifact.name}",
                    digest=digest,
                    abi="cp312",
                    platform_tag="test-platform",
                )
            )
        return BackendCompileResult(
            attempt=CompileAttempt(
                success=True,
                command=(self.name, *(unit.region_id for unit in units)),
                stdout="",
                stderr="",
                artifact_paths=tuple(paths),
                duration_seconds=0.2,
            ),
            artifacts=tuple(records),
        )


def _package_attr(name: str) -> object:
    return vars(package_command)[name]


_copy_atoll_artifacts = cast(
    Callable[[tuple[Path, ...], Path], None],
    _package_attr("_copy_atoll_artifacts"),
)
_copy_if_different = cast(
    Callable[[Path, Path], None],
    _package_attr("_copy_if_different"),
)
_copy_source_roots = cast(
    Callable[[DiscoveredProject, Path], tuple[Path, ...]],
    _package_attr("_copy_source_roots"),
)
_copy_pep517_project = cast(
    _Pep517ProjectCopy,
    _package_attr("_copy_pep517_project"),
)
_source_roots_digest = cast(
    Callable[[tuple[Path, ...]], str],
    _package_attr("_source_roots_digest"),
)
_stage_target_sources = cast(
    Callable[
        [DiscoveredProject, Path, Callable[[str], None] | None],
        tuple[tuple[Path, ...], str],
    ],
    _package_attr("_stage_target_sources"),
)
_find_module = cast(
    Callable[[tuple[ModuleId, ...], str], ModuleId],
    _package_attr("_find_module"),
)
_mapping = cast(
    Callable[[object], dict[str, object]],
    _package_attr("_mapping"),
)
_project_metadata = cast(
    Callable[[Path], _Metadata],
    _package_attr("_project_metadata"),
)
_relative_source_root = cast(
    Callable[[Path, Path], Path],
    _package_attr("_relative_source_root"),
)
_reset_dir = cast(Callable[[Path], None], _package_attr("_reset_dir"))
_resolve_output_dir = cast(
    Callable[[Path, Path | None], Path],
    _package_attr("_resolve_output_dir"),
)
_sequence = cast(
    Callable[[object], tuple[object, ...]],
    _package_attr("_sequence"),
)
_staged_module = cast(
    Callable[[ModuleId, DiscoveredProject, tuple[Path, ...]], ModuleId],
    _package_attr("_staged_module"),
)
_string = cast(Callable[[object], str | None], _package_attr("_string"))
_selected_scans = cast(
    _SelectedScans,
    _package_attr("_selected_scans"),
)
_selected_typed_regions = cast(
    Callable[..., tuple[_TypedSelection, ...]],
    _package_attr("_selected_typed_regions"),
)
_eligible_atomic_class = cast(
    Callable[[TypedRegion, BackendAssessment], SymbolId | None],
    _package_attr("_eligible_atomic_class"),
)
_prepare_typed_region = cast(
    Callable[..., _PreparedTypedRegion], _package_attr("_prepare_typed_region")
)
_PreparedTypedRegionState = cast(
    Callable[..., _PreparedTypedRegion],
    _package_attr("_PreparedTypedRegion"),
)
_profiled_profitability_candidates = cast(
    Callable[
        [
            tuple[_PreparedTypedRegion, ...],
            tuple[_TypedRegionFailure, ...],
            ProfileResult,
        ],
        tuple[_ProfitabilityCandidate, ...],
    ],
    _package_attr("_profiled_profitability_candidates"),
)
_artifact_records_for_prepared = cast(
    Callable[
        [tuple[_PreparedTypedRegion, ...], tuple[ArtifactRecord, ...]],
        tuple[ArtifactRecord, ...],
    ],
    _package_attr("_artifact_records_for_prepared"),
)
_partition_wheel_owned_variants = cast(
    Callable[..., tuple[tuple[_PreparedTypedRegion, ...], tuple[_TypedRegionFailure, ...]]],
    _package_attr("_partition_wheel_owned_variants"),
)
_materialize_profitable_payload = cast(
    Callable[..., str | None],
    _package_attr("_materialize_profitable_payload"),
)
_materialize_candidate_payload = cast(
    Callable[..., None],
    _package_attr("_materialize_candidate_payload"),
)
_run_exact_candidate_trial = cast(
    Callable[..., tuple[CommandRunEvidence, BenchmarkGateResult | None]],
    _package_attr("_run_exact_candidate_trial"),
)
_select_profitable_candidates = cast(
    Callable[..., object],
    _package_attr("_select_profitable_candidates"),
)
_clear_payload_bytecode = cast(
    Callable[[tuple[Path, ...]], tuple[Path, ...]],
    _package_attr("_clear_payload_bytecode"),
)
_clear_payload_bytecode_with_progress = cast(
    Callable[[tuple[Path, ...], Callable[[str], None] | None], None],
    _package_attr("_clear_payload_bytecode_with_progress"),
)
_SelectedTypedRegion = cast(Callable[..., _TypedSelection], _package_attr("_SelectedTypedRegion"))
_RequestedCallableVariant = cast(Callable[..., object], _package_attr("_RequestedCallableVariant"))
_staged_typed_selection = cast(
    Callable[[ModuleScan, _TypedSelection], _TypedSelection],
    _package_attr("_staged_typed_selection"),
)
_ProfileCompileSelectionScope = cast(
    Callable[..., object],
    _package_attr("_ProfileCompileSelectionScope"),
)
_profile_candidate_support = cast(
    Callable[..., object],
    _package_attr("_profile_candidate_support"),
)
_stabilize_profile_compile_selection = cast(
    Callable[..., ProfileResult | None],
    _package_attr("_stabilize_profile_compile_selection"),
)
_profile_with_replayed_compile_selection = cast(
    Callable[[ProfileResult, ProfilePlanDecision], ProfileResult],
    _package_attr("_profile_with_replayed_compile_selection"),
)
_call_chain_analyses = cast(
    Callable[
        [tuple[ModuleScan, ...], Callable[[str], None] | None],
        tuple[CallChainAnalysisResult, ...],
    ],
    _package_attr("_call_chain_analyses"),
)
_call_chain_profile_targets = cast(
    Callable[
        [tuple[CallChainAnalysisResult, ...]],
        tuple[ProfileCallEdgeTarget, ...],
    ],
    _package_attr("_call_chain_profile_targets"),
)
_profiled_call_chain_roots = cast(
    Callable[
        [tuple[CallChainAnalysisResult, ...], ProfileResult | None],
        tuple[SymbolId, ...],
    ],
    _package_attr("_profiled_call_chain_roots"),
)
_select_profile_with_call_chains = cast(
    Callable[
        [
            ProfileResult,
            tuple[ModuleScan, ...],
            tuple[CallChainAnalysisResult, ...],
            tuple[Backend, ...],
        ],
        ProfileResult,
    ],
    _package_attr("_select_profile_with_call_chains"),
)
_CallChainExtensionContext = cast(
    Callable[..., object],
    _package_attr("_CallChainExtensionContext"),
)
_prepare_call_chain_variants = cast(
    Callable[..., tuple[tuple[_PreparedTypedRegion, ...], tuple[_TypedRegionFailure, ...]]],
    _package_attr("_prepare_call_chain_variants"),
)
_extend_with_call_chain_variants = cast(
    Callable[..., tuple[list[_PreparedTypedRegion], list[_TypedRegionFailure], int]],
    _package_attr("_extend_with_call_chain_variants"),
)
_buffer_analyses = cast(
    Callable[
        [tuple[ModuleScan, ...], Callable[[str], None] | None],
        tuple[BufferAnalysisResult, ...],
    ],
    _package_attr("_buffer_analyses"),
)
_BufferExtensionContext = cast(
    Callable[..., object],
    _package_attr("_BufferExtensionContext"),
)
_extend_with_buffer_variants = cast(
    Callable[..., tuple[list[_PreparedTypedRegion], list[_TypedRegionFailure], int]],
    _package_attr("_extend_with_buffer_variants"),
)
_run_guard_member_ids = cast(
    Callable[[RunGuardNativePlan], tuple[SymbolId, ...]],
    _package_attr("_run_guard_member_ids"),
)
_runtime_member_closure = cast(
    Callable[[TypedRegion, tuple[SymbolId, ...], frozenset[SymbolId]], tuple[SymbolId, ...]],
    _package_attr("_runtime_member_closure"),
)
_selected_requested_callable_variant = cast(
    Callable[[object], tuple[_TypedSelection, ...]],
    _package_attr("_selected_requested_callable_variant"),
)
_build_typed_regions = cast(
    Callable[..., _TypedRegionOutcome], _package_attr("_build_typed_regions")
)
_TypedRegionBuildContext = cast(Callable[..., object], _package_attr("_TypedRegionBuildContext"))
_TypedRegionBuildOutcome = cast(Callable[..., object], _package_attr("_TypedRegionBuildOutcome"))
_TypedPayloadFinalizationContext = cast(
    Callable[..., object],
    _package_attr("_TypedPayloadFinalizationContext"),
)
_TypedPayloadFinalizationResult = cast(
    Callable[..., object],
    _package_attr("_TypedPayloadFinalizationResult"),
)
_select_runtime_safe_variants = cast(
    Callable[[object, object], object],
    _package_attr("_select_runtime_safe_variants"),
)
_bisect_runtime_safe_variants = cast(
    Callable[..., tuple[_PreparedTypedRegion, ...]],
    _package_attr("_bisect_runtime_safe_variants"),
)
_resolve_runtime_variant_interactions = cast(
    Callable[..., tuple[_PreparedTypedRegion, ...]],
    _package_attr("_resolve_runtime_variant_interactions"),
)
_compiler_backends = cast(dict[Backend, object], _package_attr("_COMPILER_BACKENDS"))
_member_requires_source_class = cast(
    Callable[[str], bool],
    _package_attr("_member_requires_source_class"),
)
_owner_disallows_method_binding = cast(
    Callable[[str | None, str, dict[str, LoweringDecision]], bool],
    _package_attr("_owner_disallows_method_binding"),
)
_BaselineWheelPayload = cast(
    _BaselinePayloadFactory,
    _package_attr("_BaselineWheelPayload"),
)
_ProfitabilitySelectionContext = cast(
    Callable[..., object],
    _package_attr("_ProfitabilitySelectionContext"),
)
_ProfilePreparation = cast(Callable[..., object], _package_attr("_ProfilePreparation"))
_OptimizationArm = cast(Callable[..., object], _package_attr("OptimizationArm"))
_execute_composed_source_arm = cast(
    Callable[..., package_command.PackageCommandResult],
    _package_attr("_execute_composed_source_arm"),
)
_materialize_source_optimization_arm = cast(
    Callable[..., object],
    _package_attr("_materialize_source_optimization_arm"),
)
_profile_source_candidate = cast(
    Callable[[package_command.PackageOptions, Path, Path, Path, str], ProfileResult],
    _package_attr("_profile_source_candidate"),
)
_accepted_source_profile = cast(
    Callable[[SourceOptimizationSearchResult], ProfileResult | None],
    _package_attr("_accepted_source_profile"),
)
_source_result_with_composition_fallback = cast(
    Callable[
        [
            package_command.PackageCommandResult,
            package_command.PackageCommandResult,
            DiscoveredProject,
        ],
        package_command.PackageCommandResult,
    ],
    _package_attr("_source_result_with_composition_fallback"),
)
_run_configured_quality_gate = cast(
    Callable[..., _QualityGateOutcomeView],
    _package_attr("_run_configured_quality_gate"),
)
_QualityGateOutcome = cast(
    Callable[..., _QualityGateOutcomeView],
    _package_attr("_QualityGateOutcome"),
)
_SourceCleanPromotionContext = cast(
    Callable[..., object],
    _package_attr("_SourceCleanPromotionContext"),
)
_promote_source_clean_payload = cast(
    Callable[[object], _PromotionResultView],
    _package_attr("_promote_source_clean_payload"),
)
_FusionResearchContext = cast(Callable[..., object], _package_attr("_FusionResearchContext"))
_run_conditional_task_fusion_research = cast(
    Callable[[object], _FusionResearchOutcomeView],
    _package_attr("_run_conditional_task_fusion_research"),
)
_task_fusion_source_path = cast(
    Callable[[DiscoveredProject, Path, FusionPlan], Path],
    _package_attr("_task_fusion_source_path"),
)
_fusion_trial_timings = cast(
    Callable[[FusionTrial], tuple[CompilePhaseTiming, ...]],
    _package_attr("_fusion_trial_timings"),
)
_print_source_clean_success = cast(
    Callable[..., None],
    vars(cli_module)["_print_source_clean_success"],
)
_promotion_wheel_tag = cast(
    Callable[[object, Path], str],
    _package_attr("_promotion_wheel_tag"),
)
_ExecutionPlanOnlyContext = cast(
    Callable[..., object],
    _package_attr("_ExecutionPlanOnlyContext"),
)
_ExecutionPlanApplicationOutcome = cast(
    Callable[..., object],
    _package_attr("_ExecutionPlanApplicationOutcome"),
)
_SourceCleanPromotionResult = cast(
    Callable[..., object],
    _package_attr("_SourceCleanPromotionResult"),
)
_execute_execution_plan_only_package = cast(
    Callable[[object], package_command.PackageCommandResult],
    _package_attr("_execute_execution_plan_only_package"),
)


def test_typed_region_selection_prefers_mypyc_for_safe_specializations(
    tmp_path: Path,
) -> None:
    """Subclass and closed-call specializations enter the normal automatic routing path."""
    project_root = tmp_path / "typed_region_project"
    shutil.copytree(TYPED_FIXTURE_ROOT, project_root)
    project = discover_project(project_root)

    worker_selections = _selected_typed_regions(
        _selected_scans(project, "typed_region_project.worker")
    )
    worker_specializations = tuple(
        selection.specialization
        for selection in worker_selections
        if selection.specialization is not None
    )
    function_selections = _selected_typed_regions(
        _selected_scans(project, "typed_region_project.generic_functions")
    )

    assert {
        (specialization.origin, specialization.target_owner_class)
        for specialization in worker_specializations
    } == {
        ("concrete_subclass", "IntPairer"),
        ("concrete_subclass", "PayloadPairer"),
    }
    assert all(
        selection.backend == "mypyc"
        for selection in worker_selections
        if selection.specialization is not None
    )
    assert len(function_selections) == EXPECTED_ATOMIC_SELECTION_COUNT
    ordinary = next(
        selection for selection in function_selections if selection.specialization is None
    )
    assert ordinary.backend == "mypyc"
    assert tuple(member.qualname for member in ordinary.members) == ("pair_int",)
    function_specialization = next(
        selection for selection in function_selections if selection.specialization is not None
    )
    assert function_specialization.backend == "mypyc"
    assert function_specialization.specialization is not None
    assert function_specialization.specialization.origin == "closed_call"


def test_explicit_function_selection_creates_one_directed_slice_per_binding(
    tmp_path: Path,
) -> None:
    """Independent requested roots do not inherit an oversized connected component."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    requested = (
        SymbolId(module="app.ranking", qualname="score_user"),
        SymbolId(module="app.ranking", qualname="rank_candidates"),
    )

    selections = _selected_typed_regions(
        _selected_scans(project, "app.ranking"),
        ("mypyc", "cython"),
        requested,
    )

    assert len(selections) == EXPECTED_ATOMIC_SELECTION_COUNT
    assert all(selection.backend == "mypyc" for selection in selections)
    assert {tuple(member.qualname for member in selection.members) for selection in selections} == {
        ("score_user",),
        ("rank_candidates",),
    }
    assert {
        member for selection in selections for member in (selection.bound_members or ())
    } == set(requested)
    assert all(selection.slice_root is not None for selection in selections)
    assert all(selection.source_region_id is not None for selection in selections)


def test_profiled_call_edges_promote_hot_call_chain_root(tmp_path: Path) -> None:
    """Exact helper-edge counts select the public caller rather than only the leaf."""
    project_root = tmp_path / "native_project"
    shutil.copytree(Path("tests/fixtures/native_optimization_project"), project_root)
    project = discover_project(project_root)
    scans = _selected_scans(project, "native_optimization_fixture.kernels")
    analyses = _call_chain_analyses(scans, None)
    targets = _call_chain_profile_targets(analyses)
    target = next(item for item in targets if item.owner.qualname == "direct_chain_root")
    profile = replace(
        unconfigured_profile(),
        status="profiled",
        reason="test profile",
        call_edges=(ProfiledCallEdge(target=target, invocation_count=10_000),),
    )

    roots = _profiled_call_chain_roots(analyses, profile)
    selected = _select_profile_with_call_chains(
        profile,
        scans,
        analyses,
        project.config.compile.backends,
    )

    assert roots[0] == SymbolId(
        "native_optimization_fixture.kernels",
        "direct_chain_root",
    )
    assert roots[0] in selected.selected_symbols


def test_profile_selection_skips_unbindable_dunder_and_backfills_supported_member(
    tmp_path: Path,
) -> None:
    """An unsupported hottest root cannot consume automatic candidate capacity."""
    project_root = tmp_path / "typed_region_project"
    shutil.copytree(TYPED_FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    scans = _selected_scans(project, "typed_region_project.worker")
    lifecycle = LifecycleCounts(
        start=0,
        return_=0,
        yield_=0,
        resume=0,
        unwind=0,
        throw=0,
    )
    unsupported = ProfiledMember(
        module="typed_region_project.worker",
        qualname="Worker.__init__",
        samples=100,
        coverage=0.5,
        call_count=0,
        lifecycle=lifecycle,
        signatures=(),
        polymorphic_overflow=False,
    )
    supported = ProfiledMember(
        module="typed_region_project.worker",
        qualname="Worker.score",
        samples=80,
        coverage=0.4,
        call_count=0,
        lifecycle=lifecycle,
        signatures=(),
        polymorphic_overflow=False,
    )
    profile = replace(
        unconfigured_profile(),
        status="profiled",
        reason="test profile",
        total_samples=200,
        mapped_project_samples=180,
        mapped_coverage=0.9,
        members=(unsupported, supported),
    )

    selected = _select_profile_with_call_chains(
        profile,
        scans,
        (),
        project.config.compile.backends,
    )

    assert selected.selected_symbols == (supported.symbol,)
    assert [(item.symbol, item.reason) for item in selected.candidates] == [
        (unsupported.symbol, "not-independently-bindable"),
        (supported.symbol, "selected"),
    ]


def test_explicit_unbindable_member_remains_an_unsupported_request(
    tmp_path: Path,
) -> None:
    """Explicit roots return no variant without weakening directed-slice invariants."""
    project_root = tmp_path / "typed_region_project"
    shutil.copytree(TYPED_FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    scans = _selected_scans(project, "typed_region_project.worker")
    constructor = SymbolId("typed_region_project.worker", "Worker.__init__")

    selected = _selected_typed_regions(
        scans,
        project.config.compile.backends,
        (constructor,),
        hot_members=(constructor,),
    )

    assert selected == ()


def test_iterator_protocol_roots_are_selected_without_other_dunders(tmp_path: Path) -> None:
    """Package selection binds iterator slots while retaining other class-owned dunders."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    (project_root / "src" / "app" / "iterator.py").write_text(
        """class Cursor:
    def value(self) -> int:
        return 1

    def __iter__(self) -> object:
        return self

    def __next__(self) -> int:
        raise StopIteration

    def __get__(self, instance: object, owner: type[object] | None = None) -> object:
        return self

    def __add__(self, other: object) -> object:
        return self
""",
        encoding="utf-8",
    )
    project = discover_project(project_root)
    scans = _selected_scans(project, "app.iterator")
    iterator_roots = tuple(
        SymbolId("app.iterator", f"Cursor.{name}") for name in ("__iter__", "__next__")
    )
    ordinary_root = SymbolId("app.iterator", "Cursor.value")
    blocked_roots = tuple(
        SymbolId("app.iterator", f"Cursor.{name}") for name in ("__get__", "__add__")
    )

    selected = _selected_typed_regions(
        scans,
        project.config.compile.backends,
        iterator_roots,
        hot_members=iterator_roots,
    )
    blocked = _selected_typed_regions(
        scans,
        project.config.compile.backends,
        blocked_roots,
        hot_members=blocked_roots,
    )
    ordinary = _selected_typed_regions(
        scans,
        project.config.compile.backends,
        (ordinary_root,),
        hot_members=(ordinary_root,),
    )

    assert {selection.slice_root for selection in selected} == set(iterator_roots)
    assert {
        binding.source for selection in selected for binding in selection.region.bindings
    } == set(iterator_roots)
    assert all(
        binding.kind == "instance_method"
        for selection in selected
        for binding in selection.region.bindings
    )
    assert {selection.slice_root for selection in ordinary} == {ordinary_root}
    assert blocked == ()


def test_profitability_selection_rejects_missing_payload_prerequisites(tmp_path: Path) -> None:
    """Internal candidate trials fail before touching incomplete payload state."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    baseline = _BaselineWheelPayload(
        wheel_path=None,
        build=_successful_attempt(),
    )
    context = _ProfitabilitySelectionContext(
        successful=(),
        skipped=(),
        profile=unconfigured_profile(),
        project=project,
        baseline=baseline,
        payload_root=tmp_path / "payload",
        staged_source_roots=(),
        progress=None,
    )

    with pytest.raises(ValueError, match="unpacked baseline wheel"):
        _materialize_candidate_payload(context, (), tmp_path / "candidate")
    with pytest.raises(ValueError, match="selection prerequisites"):
        _run_exact_candidate_trial(context, (), object(), _CANDIDATE_SPEEDUP)

    outcome = cast(_ProfitabilitySelectionOutcomeView, _select_profitable_candidates(context))
    assert outcome.accepted == ()
    assert outcome.trials == ()


def test_profiled_zero_variant_package_runs_unoptimized_full_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Automatic capability rejection produces measured no-op evidence, not an error."""
    project_root = tmp_path / "typed_region_project"
    output_dir = tmp_path / "out"
    shutil.copytree(TYPED_FIXTURE_ROOT, project_root)
    pyproject = project_root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8")
        + """

[tool.atoll.compile]
test_command = ["python", "-c", "pass"]
benchmark_command = ["python", "bench.py"]
benchmark_warmups = 0
benchmark_samples = 1
minimum_speedup = 1.10
""",
        encoding="utf-8",
    )
    test_modes: list[RuntimeMode] = []

    def pass_semantics(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        mode: RuntimeMode,
        **_options: object,
    ) -> CommandRunEvidence:
        test_modes.append(mode)
        return CommandRunEvidence(
            command=command,
            project_root=project_root,
            payload_root=payload_root,
            mode=mode,
            returncode=0,
            stdout="",
            stderr="",
            duration_seconds=0.1,
        )

    def unsupported_profile(*_args: object, **_kwargs: object) -> ProfileResult:
        lifecycle = LifecycleCounts(
            start=10,
            return_=10,
            yield_=0,
            resume=0,
            unwind=0,
            throw=0,
        )
        return replace(
            unconfigured_profile(),
            status="profiled",
            reason="fixture profile",
            launch_kind="script",
            total_samples=200,
            mapped_project_samples=180,
            mapped_coverage=0.9,
            lifecycle=lifecycle,
            members=(
                ProfiledMember(
                    module="typed_region_project.worker",
                    qualname="Worker.__init__",
                    samples=180,
                    coverage=0.9,
                    call_count=10,
                    lifecycle=lifecycle,
                    signatures=(),
                    polymorphic_overflow=False,
                ),
            ),
        )

    def no_speedup(*_args: object, **_kwargs: object) -> BenchmarkGateResult:
        return BenchmarkGateResult(
            status="not-profitable",
            reason="fixture measured no speedup",
            minimum_speedup=1.1,
            baseline_median_seconds=1.0,
            compiled_median_seconds=1.0,
            speedup=1.0,
            warmups=(),
            samples=(),
        )

    def reject_native_execution(*_args: object, **_kwargs: object) -> object:
        pytest.fail("automatic unsupported roots must not enter native compilation")

    monkeypatch.setattr(package_command, "run_performance_command", pass_semantics)
    monkeypatch.setattr(package_command, "run_baseline_profile", unsupported_profile)
    monkeypatch.setattr(package_command, "run_benchmark_gate", no_speedup)
    monkeypatch.setattr(package_command, "_execute_typed_region_package", reject_native_execution)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="typed_region_project.worker",
            output_dir=output_dir,
        )
    )

    assert result.success is False
    assert result.error == "fixture measured no speedup"
    assert result.performance is not None
    assert result.performance.status == "not-profitable"
    assert result.profile is not None
    assert result.profile.selected_symbols == ()
    assert result.profile.candidates[0].reason == "not-independently-bindable"
    assert test_modes == ["baseline", "compiled"]
    assert not tuple(output_dir.glob("*.whl"))


def test_call_chain_extension_reports_cython_disabled(tmp_path: Path) -> None:
    """Configured backend order disables specialized chain variants explicitly."""
    project_root = tmp_path / "native_project"
    shutil.copytree(NATIVE_FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    project = replace(
        project,
        config=replace(
            project.config,
            compile=replace(project.config.compile, backends=("mypyc",)),
        ),
    )
    analyses = _call_chain_analyses(
        _selected_scans(project, "native_optimization_fixture.kernels"),
        None,
    )
    messages: list[str] = []
    context = _CallChainExtensionContext(
        project=project,
        build_root=tmp_path / "build",
        staged_source_roots=(),
        analyses=analyses,
        progress=messages.append,
    )

    prepared, failures, count = _extend_with_call_chain_variants(
        context=context,
        prepared=[],
        failures=[],
    )

    assert prepared == []
    assert failures == []
    assert count == 0
    assert messages == ["direct call-chain variants skipped because Cython is disabled"]


def test_call_chain_preparation_rejects_staged_source_drift(tmp_path: Path) -> None:
    """Copied-source rescanning must reproduce the exact call-chain plan identity."""
    project_root = tmp_path / "native_project"
    shutil.copytree(NATIVE_FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    analyses = _call_chain_analyses(
        _selected_scans(project, "native_optimization_fixture.kernels"),
        None,
    )
    selected_analyses = tuple(
        replace(
            analysis,
            plans=tuple(
                plan for plan in analysis.plans if plan.root.qualname == "direct_chain_root"
            ),
        )
        for analysis in analyses
    )
    build_root = tmp_path / "build"
    staged_roots = _copy_source_roots(project, build_root)
    module = _find_module(project.modules, "native_optimization_fixture.kernels")
    staged_module = _staged_module(module, project, staged_roots)
    source = staged_module.path.read_text(encoding="utf-8")
    staged_module.path.write_text(
        source.replace(
            "    total = 0\n    for offset in range(depth):",
            "    total = 1\n    for offset in range(depth):",
            1,
        )
    )

    prepared, failures = _prepare_call_chain_variants(
        project=project,
        build_root=build_root,
        staged_source_roots=staged_roots,
        analyses=selected_analyses,
    )

    assert prepared == ()
    assert len(failures) == EXPECTED_SINGLE_FAILURE
    assert failures[0].variant_id.endswith("@cython-call-chain")


def test_buffer_extension_reports_cython_disabled(tmp_path: Path) -> None:
    """Configured backend order disables zero-copy buffer variants explicitly."""
    project_root = tmp_path / "native_project"
    shutil.copytree(NATIVE_FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    project = replace(
        project,
        config=replace(
            project.config,
            compile=replace(project.config.compile, backends=("mypyc",)),
        ),
    )
    analyses = _buffer_analyses(
        _selected_scans(project, "native_optimization_fixture.kernels"),
        None,
    )
    messages: list[str] = []
    context = _BufferExtensionContext(
        project=project,
        build_root=tmp_path / "build",
        staged_source_roots=(),
        analyses=analyses,
        progress=messages.append,
    )

    prepared, failures, count = _extend_with_buffer_variants(
        context=context,
        prepared=[],
        failures=[],
    )

    assert prepared == []
    assert failures == []
    assert count == 0
    assert messages == ["zero-copy buffer variants skipped because Cython is disabled"]


def test_buffer_extension_keeps_candidates_after_generic_preparation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A generic backend failure cannot filter out an independent buffer proof."""
    project_root = tmp_path / "native_project"
    shutil.copytree(NATIVE_FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    scans = _selected_scans(project, "native_optimization_fixture.kernels")
    analyses = _buffer_analyses(scans, None)
    selected_plan = next(
        plan
        for analysis in analyses
        for plan in analysis.plans
        if plan.member.qualname == "bytes_checksum"
    )
    region = next(
        item
        for item in scans[0].typed_regions
        if any(member.id == selected_plan.member for member in item.members)
    )
    generic_failure = package_command.PackageRegionBuildFailure(
        region=region,
        variant_id="generic-failure",
        backend="mypyc",
        assessment=BackendAssessment(
            region_id=region.id,
            backend="mypyc",
            status="unsupported",
            supported_members=(),
            unsupported_members=(selected_plan.member,),
            capabilities=(),
            reasons=("fixture generic failure",),
        ),
        build=CompileAttempt(
            success=False,
            command=(),
            stdout="",
            stderr="fixture generic failure",
            artifact_paths=(),
            duration_seconds=0.0,
        ),
    )
    captured: list[object] = []

    def capture_plans(**kwargs: object) -> tuple[tuple[object, ...], tuple[object, ...]]:
        selected = cast(tuple[BufferAnalysisResult, ...], kwargs["analyses"])
        captured.extend(plan for analysis in selected for plan in analysis.plans)
        return (), ()

    monkeypatch.setattr(package_command, "_prepare_buffer_variants", capture_plans)
    context = _BufferExtensionContext(
        project=project,
        build_root=tmp_path / "build",
        staged_source_roots=(),
        analyses=analyses,
        progress=None,
    )

    prepared, failures, count = _extend_with_buffer_variants(
        context=context,
        prepared=[],
        failures=[generic_failure],
    )

    assert prepared == []
    assert len(failures) == 1
    assert failures[0].variant_id == generic_failure.variant_id
    assert count == 0
    assert captured == [selected_plan]


def test_call_chain_preparation_retains_each_width_lowering_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A generator rejection records both width variants without losing fallback."""
    project_root = tmp_path / "native_project"
    shutil.copytree(NATIVE_FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    analyses = _call_chain_analyses(
        _selected_scans(project, "native_optimization_fixture.kernels"),
        None,
    )
    selected_analyses = tuple(
        replace(
            analysis,
            plans=tuple(
                plan for plan in analysis.plans if plan.root.qualname == "direct_chain_root"
            ),
        )
        for analysis in analyses
    )
    build_root = tmp_path / "build"
    staged_roots = _copy_source_roots(project, build_root)

    def reject_lowering(**_kwargs: object) -> object:
        raise ValueError("deliberate call-chain lowering rejection")

    monkeypatch.setattr(package_command, "_prepare_call_chain_variant", reject_lowering)

    prepared, failures = _prepare_call_chain_variants(
        project=project,
        build_root=build_root,
        staged_source_roots=staged_roots,
        analyses=selected_analyses,
    )

    assert prepared == ()
    assert len(failures) == EXPECTED_CALL_CHAIN_WIDTH_COUNT
    assert all(failure.variant_id.startswith("call-chain-") for failure in failures)


def test_directed_selection_rejects_staged_drift_and_missing_root(tmp_path: Path) -> None:
    """Staged rescanning must reproduce the exact checkout-derived slice."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    scans = _selected_scans(project, "app.ranking")
    requested = (SymbolId("app.ranking", "score_user"),)
    selection = _selected_typed_regions(scans, requested_members=requested)[0]
    assert selection.source_region_id is not None
    assert selection.slice_root is not None

    missing_root = _SelectedTypedRegion(
        scan=selection.scan,
        region=selection.region,
        variant_id=selection.variant_id,
        backend=selection.backend,
        assessment=selection.assessment,
        members=selection.members,
        bound_members=selection.bound_members,
        specialization=selection.specialization,
        conditional_on_failure_of=selection.conditional_on_failure_of,
        source_region_id=selection.source_region_id,
        slice_root=None,
    )
    with pytest.raises(ValueError, match="requires a slice root"):
        _staged_typed_selection(scans[0], missing_root)

    drifted = _SelectedTypedRegion(
        scan=selection.scan,
        region=replace(selection.region, id=f"{selection.region.id}:drifted"),
        variant_id=selection.variant_id,
        backend=selection.backend,
        assessment=selection.assessment,
        members=selection.members,
        bound_members=selection.bound_members,
        specialization=selection.specialization,
        conditional_on_failure_of=selection.conditional_on_failure_of,
        source_region_id=selection.source_region_id,
        slice_root=selection.slice_root,
    )
    with pytest.raises(ValueError, match="staged directed slice differs"):
        _staged_typed_selection(scans[0], drifted)


def test_profile_and_directed_closure_helpers_cover_backend_boundaries(tmp_path: Path) -> None:
    """Profile preflight and required-edge closure remain conservative."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    scans = _selected_scans(project, "app.ranking")
    observed_root = next(
        symbol.id for symbol in scans[0].symbols if symbol.id.qualname == "rank_candidates"
    )
    support = cast(
        _ProfileCandidateSupportView,
        _profile_candidate_support(
            scans,
            project.config.compile.backends,
            roots=(observed_root,),
        ),
    )
    assessed = (*support.supported, *(item.symbol for item in support.rejected))
    assert assessed == (observed_root,)

    region = next(
        region
        for region in scans[0].typed_regions
        if any(
            isinstance(dependency.dst, SymbolId)
            and dependency.src
            in {member.id for member in region.members if member.kind == "function"}
            and dependency.dst
            in {member.id for member in region.members if member.kind == "function"}
            for dependency in region.dependencies
        )
    )
    member_ids = tuple(member.id for member in region.members if member.kind == "function")
    dependency = next(
        dependency
        for dependency in region.dependencies
        if dependency.src in member_ids
        and isinstance(dependency.dst, SymbolId)
        and dependency.dst in member_ids
    )
    assert isinstance(dependency.dst, SymbolId)
    required = replace(
        region,
        dependencies=tuple(
            replace(item, requires_same_unit=True) if item is dependency else item
            for item in region.dependencies
        ),
    )
    closure = _runtime_member_closure(required, member_ids, frozenset({dependency.src}))
    assert {dependency.src, dependency.dst} <= set(closure)

    blocked_closure = _runtime_member_closure(
        required,
        tuple(member for member in member_ids if member != dependency.dst),
        frozenset({dependency.src}),
    )
    assert blocked_closure == ()

    external = replace(
        required,
        dependencies=tuple(
            replace(item, dst="external.boundary")
            if item.src == dependency.src and item.dst == dependency.dst and item.requires_same_unit
            else item
            for item in required.dependencies
        ),
    )
    assert _runtime_member_closure(
        external,
        member_ids,
        frozenset({dependency.src}),
    ) == (dependency.src,)

    empty_inputs = _RequestedCallableVariant(
        scan=scans[0],
        region=region,
        closure=(),
        requested=frozenset({dependency.src}),
        backends=("mypyc", "cython"),
        source_region_id=region.id,
        slice_root=dependency.src,
    )
    unsupported_inputs = _RequestedCallableVariant(
        scan=scans[0],
        region=region,
        closure=(dependency.src,),
        requested=frozenset({dependency.src}),
        backends=(),
        source_region_id=region.id,
        slice_root=dependency.src,
    )
    assert _selected_requested_callable_variant(empty_inputs) == ()
    assert _selected_requested_callable_variant(unsupported_inputs) == ()


def test_selected_scans_reject_cross_module_member_scope(tmp_path: Path) -> None:
    """A module-filtered compile cannot smuggle in a member from another module."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    project = discover_project(project_root)

    with pytest.raises(ValueError, match="must belong to the requested module scope"):
        _selected_scans(
            project,
            "app.ranking",
            (SymbolId("app.models", "User"),),
        )


def test_explicit_package_fails_when_one_requested_region_does_not_compile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial region build cannot promote a wheel that promised explicit members."""
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    ranking_source = project_root / "src" / "app" / "ranking.py"
    (project_root / "src" / "app" / "extra.py").write_text(
        ranking_source.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    requested = (
        SymbolId(module="app.ranking", qualname="normalize_features"),
        SymbolId(module="app.extra", qualname="normalize_features"),
    )

    def partial_build(**kwargs: object) -> object:
        prepared = cast(tuple[_PreparedTypedRegion, ...], kwargs["prepared"])
        assert len(prepared) == EXPECTED_ATOMIC_SELECTION_COUNT
        return _TypedRegionBuildOutcome(
            successful=(prepared[0],),
            build=_successful_attempt(),
            artifacts=(),
            skipped=(),
        )

    monkeypatch.setattr(package_command, "_build_typed_regions", partial_build)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            output_dir=output_dir,
            selected_members=requested,
        )
    )

    assert result.success is False
    assert result.wheel_path is None
    assert result.error == (
        "requested member(s) did not compile successfully: app.extra::normalize_features"
    )
    assert result.cleanup_removed == (output_dir / "build", output_dir / "install")
    assert result.cleanup_kept == ()


def test_function_with_same_region_class_dependency_uses_runtime_boundary(
    tmp_path: Path,
) -> None:
    """A local class can remain interpreted while its caller compiles."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    module_path = project_root / "src" / "app" / "class_dependency.py"
    module_path.write_text(
        """class Payload:
    def __init__(self, value: int) -> None:
        self.value = value

def make_payload(value: int) -> Payload:
    return Payload(value)

def add_one(value: int) -> int:
    return value + 1
""",
        encoding="utf-8",
    )
    project = discover_project(project_root)

    selections = _selected_typed_regions(_selected_scans(project, "app.class_dependency"))

    bound = {
        member
        for selection in selections
        for member in (selection.bound_members or selection.members)
    }
    assert SymbolId(module="app.class_dependency", qualname="add_one") in bound
    assert SymbolId(module="app.class_dependency", qualname="make_payload") in bound


def test_atomic_class_selection_is_exclusive_and_partial_classes_split(
    tmp_path: Path,
) -> None:
    """A closed class has one class variant while mixed shapes remain per-member."""
    project_root = tmp_path / "typed_region_project"
    shutil.copytree(TYPED_FIXTURE_ROOT, project_root)
    project = discover_project(project_root)

    selections = _selected_typed_regions(_selected_scans(project, "typed_region_project.worker"))
    scale_selections = tuple(
        selection
        for selection in selections
        if any(member.qualname.startswith("ScaleModel") for member in selection.members)
    )
    worker_selections = tuple(
        selection
        for selection in selections
        if any(member.qualname.startswith("Worker") for member in selection.members)
    )

    class_selection = next(
        selection
        for selection in scale_selections
        if selection.variant_id.endswith("@cython-class")
    )
    method_fallback = next(
        selection for selection in scale_selections if selection is not class_selection
    )
    assert len(scale_selections) == EXPECTED_ATOMIC_SELECTION_COUNT
    assert class_selection.backend == "cython"
    assert tuple(member.qualname for member in class_selection.members) == ("ScaleModel",)
    assert method_fallback.backend == "mypyc"
    assert {member.qualname for member in method_fallback.members} == {
        "ScaleModel.apply",
        "ScaleModel.describe",
    }
    assert method_fallback.conditional_on_failure_of == class_selection.variant_id
    assert {member.qualname for selection in worker_selections for member in selection.members} == {
        "Worker.adjust",
        "Worker.exchange",
        "Worker.parse",
        "Worker.scale",
        "Worker.score",
        "Worker.values",
    }
    assert {
        selection.backend
        for selection in worker_selections
        if any(member.qualname == "Worker.exchange" for member in selection.members)
    } == {"cython"}
    assert all(
        member.qualname not in {"Worker", "Worker.__init__"}
        for selection in worker_selections
        for member in selection.members
    )


def test_atomic_class_selection_rejects_methods_from_another_owner(tmp_path: Path) -> None:
    """A foreign method cannot authorize replacement of an unrelated class."""
    project_root = tmp_path / "typed_region_project"
    shutil.copytree(TYPED_FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    scan = _selected_scans(project, "typed_region_project.worker")[0]
    region = next(item for item in scan.typed_regions if item.atomic_class)
    class_member = next(member for member in region.members if member.kind == "class")
    method_member = next(member for member in region.members if member.kind == "method")
    malformed = replace(
        region,
        members=(class_member, replace(method_member, owner_class="ForeignOwner")),
    )
    assessment = BackendAssessment(
        region_id=malformed.id,
        backend="cython",
        status="supported",
        supported_members=tuple(member.id for member in malformed.members),
        unsupported_members=(),
        capabilities=("native_class", "instance_method"),
        reasons=("test assessment",),
    )

    assert _eligible_atomic_class(malformed, assessment) is None


def test_method_selection_rejects_class_cell_and_private_name_semantics() -> None:
    """Top-level extraction never guesses class-cell or name-mangling behavior."""
    assert _member_requires_source_class("def value(self) -> int:\n    return super().value()\n")
    assert _member_requires_source_class("def owner(self) -> type[object]:\n    return __class__\n")
    assert _member_requires_source_class("def secret(self) -> int:\n    return self.__secret\n")
    assert not _member_requires_source_class(
        "def regular(self) -> int:\n    return len(self.__dict__)\n"
    )


def test_method_selection_keeps_unparseable_normalized_source_interpreted() -> None:
    """Column-zero string contents cannot crash whole-project method selection."""
    source = (
        "    @staticmethod\n"
        "    def render(self) -> str:\n"
        '        return """first line\n'
        "column-zero content\n"
        '        last line"""\n'
    )

    assert _member_requires_source_class(source)
    assert _member_requires_source_class("    def malformed(:\n")


def test_method_selection_preserves_registered_and_dynamic_owner_classes() -> None:
    """Method mutation rejects owner-wide hazards, not atomic-class limitations."""
    registered = LoweringDecision(
        target="module::Registered",
        action="fallback",
        reason="class remains interpreted because decorators may register or replace it",
    )
    eager = LoweringDecision(
        target="module::Eager",
        action="fallback",
        reason="class remains interpreted because module-time code retains its original identity",
    )
    special = LoweringDecision(
        target="module::Iterable",
        action="fallback",
        reason=(
            "class remains interpreted because special method __iter__ "
            "requires interpreted class semantics"
        ),
    )
    dynamic_special = LoweringDecision(
        target="module::Dynamic",
        action="fallback",
        reason=(
            "class remains interpreted because special method __getattr__ "
            "requires interpreted class semantics"
        ),
    )

    assert _owner_disallows_method_binding(
        "Registered",
        "module",
        {registered.target: registered},
    )
    assert not _owner_disallows_method_binding(
        "Eager",
        "module",
        {eager.target: eager},
    )
    assert not _owner_disallows_method_binding(
        "Iterable",
        "module",
        {special.target: special},
    )
    assert _owner_disallows_method_binding(
        "Dynamic",
        "module",
        {dynamic_special.target: dynamic_special},
    )


def test_profile_selected_dataclass_methods_use_boxed_cython_slices(
    tmp_path: Path,
) -> None:
    """Hot boxed methods rebind safely on the original recognized dataclass."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    module_path = project_root / "src" / "app" / "boxed_runner.py"
    module_path.write_text(
        """from __future__ import annotations

import typing
from dataclasses import dataclass


@dataclass
class Runner:
    bias: int

    def dynamic(self, value: typing.Any) -> typing.Any:
        return value

    def incomplete(self, value):
        return value + self.bias

    def identity[T](self, value: T) -> T:
        return value
""",
        encoding="utf-8",
    )
    project = discover_project(project_root)
    scans = _selected_scans(project, "app.boxed_runner")
    hot = tuple(
        SymbolId("app.boxed_runner", qualname)
        for qualname in ("Runner.dynamic", "Runner.incomplete", "Runner.identity")
    )

    static = _selected_typed_regions(scans)
    selected = _selected_typed_regions(
        scans,
        ("mypyc", "cython"),
        hot,
        hot_members=hot,
    )

    assert static == ()
    assert len(selected) == len(hot)
    assert all(selection.backend == "cython" for selection in selected)
    assert {selection.slice_root for selection in selected} == set(hot)
    assert all(len(selection.members) == 1 for selection in selected)
    assert all(selection.region.atomic_class is False for selection in selected)


def test_package_default_does_not_call_legacy_sidecar_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Source-clean package compilation bypasses the legacy sidecar facade."""
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    source_path = project_root / "src" / "app" / "ranking.py"
    original_source = source_path.read_text(encoding="utf-8")

    def failing_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        assert args
        assert kwargs
        raise AssertionError("legacy sidecar backend was invoked")

    monkeypatch.setattr(package_command, "build_sidecars", failing_build_sidecars, raising=False)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
        )
    )

    assert result.success is True
    assert result.error is None
    assert result.wheel_path is not None
    assert source_path.read_text(encoding="utf-8") == original_source
    assert not (output_dir / "build").exists()
    assert not (output_dir / "install").exists()
    assert result.islands == ()
    assert len(result.compiled_bindings) == RANKING_BINDING_COUNT


@pytest.mark.parametrize("failed_stage", ["payload", "wheel"])
def test_package_cleans_payload_after_subprocess_verification_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_stage: VerificationStage,
) -> None:
    """Routing failures retain report evidence without persistent scratch payloads."""
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]

    def successful_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        assert kwargs
        path = cast(tuple[Path, ...], args[0])[0]
        artifact = tmp_path / f"{path.stem}{suffix}"
        artifact.write_text("binary", encoding="utf-8")
        return CompileAttempt(
            success=True,
            command=("mypyc",),
            stdout="",
            stderr="",
            artifact_paths=(artifact,),
            duration_seconds=0.1,
        )

    def verify(**kwargs: object) -> PackageVerificationResult:
        stage = cast(VerificationStage, kwargs["stage"])
        target = cast(Path, kwargs["target"])
        success = stage != failed_stage
        return PackageVerificationResult(
            stage=stage,
            target=target,
            command=("python", "verify"),
            success=success,
            exit_code=0 if success else 1,
            stdout="",
            stderr="routing failed" if not success else "",
            duration_seconds=0.1,
        )

    monkeypatch.setattr(package_command, "build_sidecars", successful_build_sidecars, raising=False)
    monkeypatch.setattr(package_command, "verify_package_subprocess", verify)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
        )
    )

    assert result.success is False
    if failed_stage == "payload":
        assert result.error == (
            "isolated payload verification failed without any native variants: routing failed"
        )
    else:
        assert result.error == "routing failed"
    assert result.cleanup_removed == (output_dir / "build", output_dir / "install")
    assert result.cleanup_kept == ()
    assert not (output_dir / "build").exists()
    assert not (output_dir / "install").exists()
    assert not tuple(output_dir.glob("*.whl"))
    assert result.verification_steps[-1].stage == failed_stage
    assert not result.verification_steps[-1].target.exists()


def _prepare_outlined_coroutine_fixture(
    tmp_path: Path,
) -> tuple[_PreparedTypedRegion, Path, tuple[Path, ...]]:
    project_root = tmp_path / "typed_region_project"
    build_root = tmp_path / "build"
    shutil.copytree(TYPED_FIXTURE_ROOT, project_root)
    module_path = project_root / "src" / "typed_region_project" / "outline_worker.py"
    module_path.write_text(
        """async def checkpoint() -> None:
    return None


async def hot(values: list[int]) -> int:
    start = len(values) + 1
    doubled = start * 2
    total = doubled + 3
    await checkpoint()
    return total
""",
        encoding="utf-8",
    )
    project = discover_project(project_root)
    scans = _selected_scans(project, "typed_region_project.outline_worker")
    hot = SymbolId("typed_region_project.outline_worker", "hot")
    selections = _selected_typed_regions(
        scans,
        ("mypyc", "cython"),
        (hot,),
        hot_members=(hot,),
    )
    assert len(selections) == 1
    staged_source_roots = _copy_source_roots(project, build_root)
    prepared = _prepare_typed_region(
        project=project,
        build_root=build_root,
        staged_source_roots=staged_source_roots,
        selection=selections[0],
    )
    return prepared, build_root, staged_source_roots


def test_prepare_typed_region_appends_outlined_cython_fallback(tmp_path: Path) -> None:
    """A precise async root prepares whole-callable and outlined backend variants."""
    prepared, _build_root, _staged_source_roots = _prepare_outlined_coroutine_fixture(tmp_path)

    assert prepared.generation.backend == "mypyc"
    assert prepared.fallback is not None
    assert prepared.fallback.lowering_mode == "whole-callable"
    assert prepared.fallback.fallback is not None
    outlined = prepared.fallback.fallback
    assert outlined.generation.backend == "cython"
    assert outlined.lowering_mode == "outlined-block"
    assert outlined.native_helpers
    assert outlined.unit.region_id.endswith("@cython-outline")


def test_wheel_owned_variant_partition_rejects_backend_omissions(tmp_path: Path) -> None:
    """Only source modules already shipped by the baseline wheel reach a compiler."""
    prepared, _build_root, staged_source_roots = _prepare_outlined_coroutine_fixture(tmp_path)
    install_root = tmp_path / "install"
    relative = prepared.shim.source_path.relative_to(staged_source_roots[0])
    destination = install_root / relative
    destination.parent.mkdir(parents=True)
    destination.write_text("baseline\n", encoding="utf-8")

    retained, omitted = _partition_wheel_owned_variants(
        (prepared,),
        staged_source_roots=staged_source_roots,
        install_root=install_root,
    )

    assert retained == (prepared,)
    assert omitted == ()

    destination.unlink()
    retained, omitted = _partition_wheel_owned_variants(
        (prepared,),
        staged_source_roots=staged_source_roots,
        install_root=install_root,
    )

    assert retained == ()
    assert len(omitted) == 1
    assert omitted[0].variant_id == prepared.unit.region_id
    assert "target PEP 517 wheel omitted" in omitted[0].build.stderr


def test_runtime_safety_selection_drops_variant_that_fails_in_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crashing compiled variant is removed while the baseline payload survives."""
    prepared, _build_root, staged_source_roots = _prepare_outlined_coroutine_fixture(tmp_path)
    project_root = tmp_path / "project"
    project = discover_project(project_root)
    baseline_root = tmp_path / "baseline"
    install_root = tmp_path / "install"
    shutil.copytree(staged_source_roots[0], baseline_root)
    shutil.copytree(baseline_root, install_root)
    outcome = _TypedRegionBuildOutcome(
        successful=(prepared,),
        build=_successful_attempt(),
        artifacts=(),
        skipped=(),
    )
    baseline = _BaselineWheelPayload(
        wheel_path=tmp_path / "baseline.whl",
        build=_successful_attempt(),
        baseline_install_root=baseline_root,
    )

    def verify(**kwargs: object) -> PackageVerificationResult:
        allowlist = cast(frozenset[str] | None, kwargs["variant_allowlist"])
        success = allowlist == frozenset()
        return PackageVerificationResult(
            stage=cast(VerificationStage, kwargs["stage"]),
            target=cast(Path, kwargs["target"]),
            command=("python", "verify"),
            success=success,
            exit_code=0 if success else -11,
            stdout="",
            stderr="",
            duration_seconds=0.01,
        )

    monkeypatch.setattr(package_command, "verify_package_subprocess", verify)
    context = _TypedPayloadFinalizationContext(
        options=package_command.PackageOptions(root=project_root),
        project=project,
        profile=None,
        baseline=baseline,
        install_root=install_root,
        staged_source_roots=staged_source_roots,
        outcome=outcome,
        overlay_error=None,
    )
    finalized = _TypedPayloadFinalizationResult(
        successful=(prepared,),
        artifacts=(),
        build=_successful_attempt(),
        trials=(),
        overlay_error=None,
        profitability_applied=False,
    )

    result = cast(
        _RuntimeSafetySelectionView,
        _select_runtime_safe_variants(context, finalized),
    )

    assert result.successful == ()
    assert len(result.failures) == 1
    assert result.failures[0].variant_id == prepared.unit.region_id
    assert len(result.verification_steps) == EXPECTED_SAFETY_VERIFICATION_STEPS
    assert result.overlay_error is None


def test_runtime_safety_selection_resolves_interaction_only_failure(tmp_path: Path) -> None:
    """Individually safe variants are combined only while the aggregate still imports."""
    first, _build_root, _staged_source_roots = _prepare_outlined_coroutine_fixture(tmp_path)
    second = _PreparedTypedRegionState(
        generation=first.generation,
        assessment=first.assessment,
        unit=replace(first.unit, region_id=f"{first.unit.region_id}-second"),
        shim=first.shim,
        lowering_mode=first.lowering_mode,
        native_helpers=first.native_helpers,
    )
    failed = PackageVerificationResult(
        stage="payload",
        target=tmp_path,
        command=("python", "verify"),
        success=False,
        exit_code=-11,
        stdout="",
        stderr="combined activation failed",
        duration_seconds=0.01,
    )
    passed = replace(failed, success=True, exit_code=0, stderr="")

    def verify(candidates: tuple[_PreparedTypedRegion, ...]) -> PackageVerificationResult:
        return passed if len(candidates) <= 1 else failed

    rejected: dict[str, PackageVerificationResult] = {}
    individually_safe = _bisect_runtime_safe_variants(
        (first, second),
        verify=verify,
        failed_result=failed,
        rejected_results=rejected,
    )
    retained = _resolve_runtime_variant_interactions(
        individually_safe,
        verify=verify,
        rejected_results=rejected,
    )

    assert individually_safe == (first, second)
    assert retained == (first,)
    assert rejected == {second.unit.region_id: failed}


def test_private_native_helper_uses_public_owner_for_profile_profitability(
    tmp_path: Path,
) -> None:
    """A source-fused helper is trialed when its owner is outside leaf selection."""
    prepared, _build_root, _staged_source_roots = _prepare_outlined_coroutine_fixture(tmp_path)
    owner = SymbolId("typed_region_project.outline_worker", "Owner.submit")
    selected_leaf = SymbolId("typed_region_project.outline_worker", "hot")
    attributed = _PreparedTypedRegionState(
        generation=prepared.generation,
        assessment=prepared.assessment,
        unit=prepared.unit,
        shim=prepared.shim,
        fallback=prepared.fallback,
        conditional_on_failure_of=prepared.conditional_on_failure_of,
        lowering_mode=prepared.lowering_mode,
        native_helpers=prepared.native_helpers,
        fallback_reason=prepared.fallback_reason,
        minimum_marginal_speedup=1.05,
        profitability_symbols=(owner,),
    )
    lifecycle = LifecycleCounts(start=0, return_=0, yield_=0, resume=0, unwind=0, throw=0)
    profile = replace(
        unconfigured_profile(),
        status="profiled",
        reason="public owner is hot",
        launch_kind="script",
        total_samples=60,
        mapped_project_samples=50,
        mapped_coverage=5 / 6,
        selected_hot_samples=10,
        selected_hot_coverage=0.2,
        lifecycle=lifecycle,
        members=(
            ProfiledMember(
                module=owner.module,
                qualname=owner.qualname,
                samples=OWNER_PROFILE_SAMPLES,
                coverage=0.8,
                call_count=10,
                lifecycle=lifecycle,
                signatures=(),
                polymorphic_overflow=False,
            ),
            ProfiledMember(
                module=selected_leaf.module,
                qualname=selected_leaf.qualname,
                samples=10,
                coverage=1 / 6,
                call_count=10,
                lifecycle=lifecycle,
                signatures=(),
                polymorphic_overflow=False,
            ),
        ),
        selected_symbols=(selected_leaf,),
    )

    candidates = _profiled_profitability_candidates((attributed,), (), profile)

    assert len(candidates) == 1
    assert candidates[0].prepared is attributed
    assert candidates[0].symbols == (owner.stable_id,)
    assert candidates[0].profile_samples == OWNER_PROFILE_SAMPLES


def test_source_fused_member_ownership_includes_completion_helpers() -> None:
    """Generic selection cannot duplicate members compiled transactionally."""
    module = "app.pipeline"
    completion = CompletionIndexNativePlan(
        snapshot=SymbolId(module, "_snapshot"),
        query=SymbolId(module, "_query"),
        index_attribute="_index",
        count_attribute="_count",
        active_attribute="active",
        fallback_predicate_method="_is_complete",
        graph_attribute="graph",
        parent_lookup_method="get_parent",
        intermediate_nodes_attribute="intermediate_nodes",
    )
    plan = RunGuardNativePlan(
        source_plan_id="source-plan",
        source=PurePosixPath("app/pipeline.py"),
        owner=SymbolId(module, "Owner.submit"),
        helper=SymbolId(module, "_guard"),
        source_guard=SymbolId(module, "_source_guard"),
        eligibility_helper=SymbolId(module, "_eligible"),
        protocol_context=SymbolId(module, "_protocol_context"),
        disable_module=SymbolId(module, "_os"),
        clear_helper=SymbolId(module, "_clear"),
        protocol_await_helper=SymbolId(module, "_protocol_await"),
        fallback_attribute="_fallback",
        state_attribute="_passed",
        run_identity_attribute="_run_identity",
        completion_index=completion,
    )

    assert _run_guard_member_ids(plan) == (
        plan.eligibility_helper,
        plan.helper,
        completion.snapshot,
        completion.query,
    )


def test_artifact_filter_keeps_only_accepted_region_support_files(tmp_path: Path) -> None:
    """Shared support records follow their collision-resistant variant directory."""
    accepted, _build_root, _staged_source_roots = _prepare_outlined_coroutine_fixture(tmp_path)
    assert accepted.fallback is not None
    rejected = accepted.fallback
    digest = "0" * 64

    def record(
        prepared: _PreparedTypedRegion,
        *,
        region_id: str,
        role: ArtifactRole,
        filename: str,
    ) -> ArtifactRecord:
        return ArtifactRecord(
            region_id=region_id,
            backend=prepared.generation.backend,
            logical_module=prepared.unit.logical_module,
            role=role,
            install_relative_path=f"{prepared.unit.install_relative_dir}/{filename}",
            digest=digest,
            abi="cp312",
            platform_tag="test-platform",
        )

    accepted_primary = record(
        accepted,
        region_id=accepted.unit.region_id,
        role="primary",
        filename="accepted.so",
    )
    accepted_support = record(
        accepted,
        region_id="__shared__",
        role="support",
        filename="accepted-support.so",
    )
    rejected_primary = record(
        rejected,
        region_id=rejected.unit.region_id,
        role="primary",
        filename="rejected.so",
    )
    rejected_support = record(
        rejected,
        region_id="__shared__",
        role="support",
        filename="rejected-support.so",
    )

    filtered = _artifact_records_for_prepared(
        (accepted,),
        (accepted_primary, accepted_support, rejected_primary, rejected_support),
    )

    assert filtered == (accepted_primary, accepted_support)


def test_rejected_module_keeps_baseline_wheel_source_bytes(tmp_path: Path) -> None:
    """A module with no accepted candidate is not overlaid from the copied checkout."""
    rejected, build_root, staged_source_roots = _prepare_outlined_coroutine_fixture(tmp_path)
    rejected.shim.source_path.write_text("staged checkout bytes\n", encoding="utf-8")
    baseline_root = build_root / "baseline-install"
    baseline_module = baseline_root / "typed_region_project" / "outline_worker.py"
    baseline_module.parent.mkdir(parents=True)
    baseline_module.write_text("baseline wheel bytes\n", encoding="utf-8")
    install_root = tmp_path / "install"

    error = _materialize_profitable_payload(
        baseline=_BaselineWheelPayload(
            wheel_path=build_root / "baseline.whl",
            build=_successful_attempt(),
            baseline_install_root=baseline_root,
        ),
        staged_source_roots=staged_source_roots,
        install_root=install_root,
        superset=(rejected,),
        accepted=(),
    )

    assert error is None
    assert (install_root / "typed_region_project" / "outline_worker.py").read_text(
        encoding="utf-8"
    ) == "baseline wheel bytes\n"


def test_typed_region_build_retries_whole_callable_failure_with_outline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deterministic backend failures continue through the outlined Cython variant."""
    prepared, build_root, staged_source_roots = _prepare_outlined_coroutine_fixture(tmp_path)
    mypyc = _FakeCompileBackend(
        BackendCompileResult(
            attempt=CompileAttempt(
                success=False,
                command=("mypyc",),
                stdout="",
                stderr="MYPYC_TYPE_ERROR: fixture",
                artifact_paths=(),
                duration_seconds=0.1,
            ),
            artifacts=(),
        )
    )
    cython = _SequencedCompileBackend(
        "cython",
        (
            BackendCompileResult(
                attempt=CompileAttempt(
                    success=False,
                    command=("cython",),
                    stdout="",
                    stderr="CYTHON_COMPILE_ERROR: whole callable fixture",
                    artifact_paths=(),
                    duration_seconds=0.2,
                ),
                artifacts=(),
            ),
            BackendCompileResult(
                attempt=CompileAttempt(
                    success=True,
                    command=("cython",),
                    stdout="",
                    stderr="",
                    artifact_paths=(),
                    duration_seconds=0.2,
                ),
                artifacts=(),
            ),
        ),
    )
    monkeypatch.setitem(_compiler_backends, "mypyc", mypyc)
    monkeypatch.setitem(_compiler_backends, "cython", cython)

    outcome = _build_typed_regions(
        prepared=(prepared,),
        context=_TypedRegionBuildContext(
            build_root=build_root,
            staged_source_roots=staged_source_roots,
            mypy_cache_dir=tmp_path / "mypy-cache",
            compile_cache_dir=tmp_path / "compile-cache",
            progress=None,
        ),
        initial_failures=(),
    )

    assert outcome.build.success is True
    assert outcome.skipped == ()
    assert len(mypyc.calls) == 1
    assert len(cython.calls) == OUTLINED_COMPILE_CALL_COUNT
    assert len(outcome.successful) == 1
    assert outcome.successful[0].lowering_mode == "outlined-block"
    assert outcome.successful[0].native_helpers
    assert outcome.successful[0].fallback_reason == (
        "mypyc whole-callable: MYPYC_TYPE_ERROR: fixture; "
        "cython whole-callable: CYTHON_COMPILE_ERROR: whole callable fixture"
    )
    assert "outlined Cython" in outcome.build.stdout


def test_outlined_fallback_chain_restores_all_warm_cache_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A warm outline build invokes neither backend and restores its native artifact."""
    prepared, build_root, staged_source_roots = _prepare_outlined_coroutine_fixture(tmp_path)
    assert prepared.fallback is not None
    assert prepared.fallback.fallback is not None
    outlined = prepared.fallback.fallback
    artifact = (
        build_root / ".atoll" / "artifacts" / outlined.unit.install_relative_dir / "native.so"
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"outlined-native-artifact")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    mypyc = _FakeCompileBackend(
        BackendCompileResult(
            attempt=CompileAttempt(
                success=False,
                command=("mypyc",),
                stdout="",
                stderr="MYPYC_TYPE_ERROR: deterministic fixture rejection",
                artifact_paths=(),
                duration_seconds=0.1,
            ),
            artifacts=(),
        )
    )
    cython = _SequencedCompileBackend(
        "cython",
        (
            BackendCompileResult(
                attempt=CompileAttempt(
                    success=False,
                    command=("cython",),
                    stdout="",
                    stderr="CYTHON_COMPILE_ERROR: deterministic whole-callable rejection",
                    artifact_paths=(),
                    duration_seconds=0.2,
                ),
                artifacts=(),
            ),
            BackendCompileResult(
                attempt=CompileAttempt(
                    success=True,
                    command=("cython",),
                    stdout="",
                    stderr="",
                    artifact_paths=(artifact,),
                    duration_seconds=0.2,
                ),
                artifacts=(
                    ArtifactRecord(
                        region_id=outlined.unit.region_id,
                        backend="cython",
                        logical_module=outlined.unit.logical_module,
                        role="primary",
                        install_relative_path=(f"{outlined.unit.install_relative_dir}/native.so"),
                        digest=digest,
                        abi="cp312",
                        platform_tag="test-platform",
                    ),
                ),
            ),
        ),
    )
    monkeypatch.setitem(_compiler_backends, "mypyc", mypyc)
    monkeypatch.setitem(_compiler_backends, "cython", cython)
    context = _TypedRegionBuildContext(
        build_root=build_root,
        staged_source_roots=staged_source_roots,
        mypy_cache_dir=tmp_path / "mypy-cache",
        compile_cache_dir=tmp_path / "compile-cache",
        progress=None,
    )

    first = _build_typed_regions(
        prepared=(prepared,),
        context=context,
        initial_failures=(),
    )
    artifact.unlink()
    second = _build_typed_regions(
        prepared=(prepared,),
        context=context,
        initial_failures=(),
    )

    assert first.build.success is True
    assert second.build.success is True
    assert len(mypyc.calls) == 1
    assert len(cython.calls) == OUTLINED_COMPILE_CALL_COUNT
    assert second.successful[0].lowering_mode == "outlined-block"
    assert second.build.artifact_paths[0].read_bytes() == b"outlined-native-artifact"
    timing_names = {timing.name for timing in second.build.phase_timings}
    assert "backend_decision_cache" in timing_names
    assert "cache_restore" in timing_names


def test_typed_region_build_retries_deterministic_mypyc_failure_with_cython(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mypyc type rejection uses the prepared Cython variant before fallback."""
    project_root = tmp_path / "typed_region_project"
    build_root = tmp_path / "build"
    shutil.copytree(TYPED_FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    scans = _selected_scans(project, "typed_region_project.worker")
    selections = _selected_typed_regions(scans)
    mypyc_selection = next(selection for selection in selections if selection.backend == "mypyc")
    staged_source_roots = _copy_source_roots(project, build_root)
    prepared = _prepare_typed_region(
        project=project,
        build_root=build_root,
        staged_source_roots=staged_source_roots,
        selection=mypyc_selection,
    )
    assert prepared.fallback is not None
    mypyc = _FakeCompileBackend(
        BackendCompileResult(
            attempt=CompileAttempt(
                success=False,
                command=("mypyc",),
                stdout="",
                stderr="MYPYC_TYPE_ERROR: fixture",
                artifact_paths=(),
                duration_seconds=0.1,
            ),
            artifacts=(),
        )
    )
    cython = _FakeCompileBackend(
        BackendCompileResult(
            attempt=CompileAttempt(
                success=True,
                command=("cython",),
                stdout="",
                stderr="",
                artifact_paths=(),
                duration_seconds=0.2,
            ),
            artifacts=(),
        )
    )
    monkeypatch.setitem(_compiler_backends, "mypyc", mypyc)
    monkeypatch.setitem(_compiler_backends, "cython", cython)

    outcome = _build_typed_regions(
        prepared=(prepared,),
        context=_TypedRegionBuildContext(
            build_root=build_root,
            staged_source_roots=staged_source_roots,
            mypy_cache_dir=tmp_path / "mypy-cache",
            compile_cache_dir=tmp_path / "compile-cache",
            progress=None,
        ),
        initial_failures=(),
    )

    assert outcome.build.success is True
    assert outcome.build.stderr == ""
    assert "compiled" in outcome.build.stdout
    assert [item.generation.backend for item in outcome.successful] == ["cython"]
    assert outcome.successful[0].fallback_reason == (
        "mypyc whole-callable: MYPYC_TYPE_ERROR: fixture"
    )
    assert outcome.skipped == ()
    assert len(mypyc.calls) == 1
    assert len(cython.calls) == 1


def test_typed_region_build_restores_rejection_and_cython_artifact_on_second_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unchanged retry path invokes neither native compiler a second time."""
    project_root = tmp_path / "typed_region_project"
    build_root = tmp_path / "build"
    shutil.copytree(TYPED_FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    scans = _selected_scans(project, "typed_region_project.worker")
    selections = _selected_typed_regions(scans)
    mypyc_selection = next(selection for selection in selections if selection.backend == "mypyc")
    staged_source_roots = _copy_source_roots(project, build_root)
    prepared = _prepare_typed_region(
        project=project,
        build_root=build_root,
        staged_source_roots=staged_source_roots,
        selection=mypyc_selection,
    )
    assert prepared.fallback is not None
    fallback = prepared.fallback
    artifact = (
        build_root / ".atoll" / "artifacts" / fallback.unit.install_relative_dir / "native.so"
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"native-cython-artifact")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    mypyc = _FakeCompileBackend(
        BackendCompileResult(
            attempt=CompileAttempt(
                success=False,
                command=("mypyc",),
                stdout="",
                stderr="MYPYC_TYPE_ERROR: deterministic fixture rejection",
                artifact_paths=(),
                duration_seconds=0.1,
            ),
            artifacts=(),
        )
    )
    cython = _FakeCompileBackend(
        BackendCompileResult(
            attempt=CompileAttempt(
                success=True,
                command=("cython",),
                stdout="",
                stderr="",
                artifact_paths=(artifact,),
                duration_seconds=0.2,
            ),
            artifacts=(
                ArtifactRecord(
                    region_id=fallback.unit.region_id,
                    backend="cython",
                    logical_module=fallback.unit.logical_module,
                    role="primary",
                    install_relative_path=(f"{fallback.unit.install_relative_dir}/native.so"),
                    digest=digest,
                    abi="cp312",
                    platform_tag="test-platform",
                ),
            ),
        )
    )
    monkeypatch.setitem(_compiler_backends, "mypyc", mypyc)
    monkeypatch.setitem(_compiler_backends, "cython", cython)
    context = _TypedRegionBuildContext(
        build_root=build_root,
        staged_source_roots=staged_source_roots,
        mypy_cache_dir=tmp_path / "mypy-cache",
        compile_cache_dir=tmp_path / "compile-cache",
        progress=None,
    )

    first = _build_typed_regions(
        prepared=(prepared,),
        context=context,
        initial_failures=(),
    )
    artifact.unlink()
    second = _build_typed_regions(
        prepared=(prepared,),
        context=context,
        initial_failures=(),
    )

    assert first.build.success is True
    assert second.build.success is True
    assert len(mypyc.calls) == 1
    assert len(cython.calls) == 1
    assert second.successful[0].generation.backend == "cython"
    assert second.build.artifact_paths[0].read_bytes() == b"native-cython-artifact"
    assert "backend_decision_cache" in {timing.name for timing in second.build.phase_timings}
    assert "cache_restore" in {timing.name for timing in second.build.phase_timings}


def test_typed_region_build_circuits_project_mypyc_failure_and_batches_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One imported-source rejection routes package peers through cached Cython."""
    project_root = tmp_path / "typed_region_project"
    build_root = tmp_path / "build"
    shutil.copytree(TYPED_FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    selections = tuple(
        selection
        for selection in _selected_typed_regions(_selected_scans(project, None))
        if selection.backend == "mypyc" and selection.conditional_on_failure_of is None
    )[:3]
    staged_source_roots = _copy_source_roots(project, build_root)
    prepared = tuple(
        _prepare_typed_region(
            project=project,
            build_root=build_root,
            staged_source_roots=staged_source_roots,
            selection=selection,
        )
        for selection in selections
    )
    assert all(item.fallback is not None for item in prepared)
    mypyc = _FakeCompileBackend(
        BackendCompileResult(
            attempt=CompileAttempt(
                success=False,
                command=("mypyc",),
                stdout="",
                stderr=(
                    "MYPYC_TYPE_ERROR: SystemExit(1)\n"
                    "typed_region_project/async_runner.py:1: error: "
                    "project graph rejection  [misc]"
                ),
                artifact_paths=(),
                duration_seconds=0.1,
            ),
            artifacts=(),
        )
    )
    cython = _ArtifactBatchCompileBackend()
    monkeypatch.setitem(_compiler_backends, "mypyc", mypyc)
    monkeypatch.setitem(_compiler_backends, "cython", cython)
    progress: list[str] = []
    context = _TypedRegionBuildContext(
        build_root=build_root,
        staged_source_roots=staged_source_roots,
        mypy_cache_dir=tmp_path / "mypy-cache",
        compile_cache_dir=tmp_path / "compile-cache",
        progress=progress.append,
        source_tree_digest="fixture-tree",
        enable_project_circuit=True,
    )

    cold = _build_typed_regions(
        prepared=prepared,
        context=context,
        initial_failures=(),
    )
    cold_call_counts = (len(mypyc.calls), len(cython.calls))

    def forbidden_compile(
        units: tuple[CompilationUnit, ...],
        backend_context: BackendCompileContext,
    ) -> BackendCompileResult:
        _ = (units, backend_context)
        raise AssertionError("warm project circuit reached a native compiler")

    monkeypatch.setattr(mypyc, "compile", forbidden_compile)
    monkeypatch.setattr(cython, "compile", forbidden_compile)
    warm = _build_typed_regions(
        prepared=prepared,
        context=context,
        initial_failures=(),
    )

    assert cold.build.success is True
    assert warm.build.success is True
    assert cold_call_counts == (1, 1)
    assert (len(mypyc.calls), len(cython.calls)) == cold_call_counts
    assert len(cython.calls[0]) == len(prepared)
    assert all(item.generation.backend == "cython" for item in cold.successful)
    assert all(item.generation.backend == "cython" for item in warm.successful)
    assert "BACKEND_POLICY_BYPASS:" in cold.build.stdout
    assert (
        sum(timing.name == "backend_project_circuit" for timing in cold.build.phase_timings)
        == len(prepared) - 1
    )
    assert any("project circuit opened" in message for message in progress)
    assert any("restored all project-circuit" in message for message in progress)


def test_project_backend_circuit_honors_cached_mypyc_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prior verified preferred artifact wins over a later package circuit."""
    project_root = tmp_path / "typed_region_project"
    build_root = tmp_path / "build"
    shutil.copytree(TYPED_FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    selections = tuple(
        selection
        for selection in _selected_typed_regions(_selected_scans(project, None))
        if selection.backend == "mypyc" and selection.conditional_on_failure_of is None
    )[:2]
    staged_source_roots = _copy_source_roots(project, build_root)
    prepared = tuple(
        _prepare_typed_region(
            project=project,
            build_root=build_root,
            staged_source_roots=staged_source_roots,
            selection=selection,
        )
        for selection in selections
    )
    assert all(item.fallback is not None for item in prepared)
    first_fallback = prepared[0].fallback
    assert first_fallback is not None
    context = _TypedRegionBuildContext(
        build_root=build_root,
        staged_source_roots=staged_source_roots,
        mypy_cache_dir=tmp_path / "mypy-cache",
        compile_cache_dir=tmp_path / "compile-cache",
        progress=None,
        source_tree_digest="fixture-tree",
        enable_project_circuit=True,
    )
    seed_mypyc = _ArtifactBatchCompileBackend("mypyc")
    dormant_cython = _ArtifactBatchCompileBackend()
    monkeypatch.setitem(_compiler_backends, "mypyc", seed_mypyc)
    monkeypatch.setitem(_compiler_backends, "cython", dormant_cython)
    seeded = _build_typed_regions(
        prepared=(prepared[1],),
        context=context,
        initial_failures=(),
    )
    assert seeded.build.success is True
    assert dormant_cython.calls == []

    rejecting_mypyc = _FakeCompileBackend(
        BackendCompileResult(
            attempt=CompileAttempt(
                success=False,
                command=("mypyc",),
                stdout="",
                stderr=(
                    "MYPYC_TYPE_ERROR: SystemExit(1)\n"
                    "typed_region_project/async_runner.py:1: error: "
                    "project graph rejection  [misc]"
                ),
                artifact_paths=(),
                duration_seconds=0.1,
            ),
            artifacts=(),
        )
    )
    cython = _ArtifactBatchCompileBackend()
    monkeypatch.setitem(_compiler_backends, "mypyc", rejecting_mypyc)
    monkeypatch.setitem(_compiler_backends, "cython", cython)

    outcome = _build_typed_regions(
        prepared=prepared,
        context=context,
        initial_failures=(),
    )

    assert outcome.build.success is True
    assert len(rejecting_mypyc.calls) == 1
    assert len(cython.calls) == 1
    assert len(cython.calls[0]) == 1
    assert cython.calls[0][0].region_id == first_fallback.unit.region_id
    assert [item.generation.backend for item in outcome.successful] == ["cython", "mypyc"]
    assert outcome.successful[1].unit.region_id == prepared[1].unit.region_id
    assert outcome.successful[1].fallback_reason is None


def test_typed_region_build_does_not_compile_speculative_cython_after_mypyc_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prepared Cython fallback remains dormant when the preferred backend succeeds."""
    project_root = tmp_path / "typed_region_project"
    build_root = tmp_path / "build"
    shutil.copytree(TYPED_FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    scans = _selected_scans(project, "typed_region_project.worker")
    selections = _selected_typed_regions(scans)
    mypyc_selection = next(selection for selection in selections if selection.backend == "mypyc")
    staged_source_roots = _copy_source_roots(project, build_root)
    prepared = _prepare_typed_region(
        project=project,
        build_root=build_root,
        staged_source_roots=staged_source_roots,
        selection=mypyc_selection,
    )
    assert prepared.fallback is not None
    mypyc = _FakeCompileBackend(
        BackendCompileResult(
            attempt=CompileAttempt(
                success=True,
                command=("mypyc",),
                stdout="",
                stderr="",
                artifact_paths=(),
                duration_seconds=0.1,
            ),
            artifacts=(),
        )
    )
    cython = _FakeCompileBackend(
        BackendCompileResult(
            attempt=CompileAttempt(
                success=True,
                command=("cython",),
                stdout="",
                stderr="",
                artifact_paths=(),
                duration_seconds=0.2,
            ),
            artifacts=(),
        )
    )
    monkeypatch.setitem(_compiler_backends, "mypyc", mypyc)
    monkeypatch.setitem(_compiler_backends, "cython", cython)

    outcome = _build_typed_regions(
        prepared=(prepared,),
        context=_TypedRegionBuildContext(
            build_root=build_root,
            staged_source_roots=staged_source_roots,
            mypy_cache_dir=tmp_path / "mypy-cache",
            compile_cache_dir=tmp_path / "compile-cache",
            progress=None,
        ),
        initial_failures=(),
    )

    assert outcome.build.success is True
    assert [item.generation.backend for item in outcome.successful] == ["mypyc"]
    assert outcome.skipped == ()
    assert len(mypyc.calls) == 1
    assert cython.calls == []


@pytest.mark.parametrize("class_succeeds", [True, False])
def test_atomic_class_build_conditionally_uses_method_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    class_succeeds: bool,
) -> None:
    """Method variants stay dormant unless the selected atomic class fails."""
    project_root = tmp_path / "typed_region_project"
    build_root = tmp_path / "build"
    shutil.copytree(TYPED_FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    selections = _selected_typed_regions(_selected_scans(project, "typed_region_project.worker"))
    class_selection = next(
        selection for selection in selections if selection.variant_id.endswith("@cython-class")
    )
    method_selection = next(
        selection
        for selection in selections
        if selection.conditional_on_failure_of == class_selection.variant_id
    )
    staged_source_roots = _copy_source_roots(project, build_root)
    prepared = tuple(
        _prepare_typed_region(
            project=project,
            build_root=build_root,
            staged_source_roots=staged_source_roots,
            selection=selection,
        )
        for selection in (class_selection, method_selection)
    )
    cython = _FakeCompileBackend(
        BackendCompileResult(
            attempt=CompileAttempt(
                success=class_succeeds,
                command=("cython",),
                stdout="",
                stderr="" if class_succeeds else "CYTHON_COMPILE_ERROR: fixture",
                artifact_paths=(),
                duration_seconds=0.1,
            ),
            artifacts=(),
        )
    )
    mypyc = _FakeCompileBackend(
        BackendCompileResult(
            attempt=CompileAttempt(
                success=True,
                command=("mypyc",),
                stdout="",
                stderr="",
                artifact_paths=(),
                duration_seconds=0.1,
            ),
            artifacts=(),
        )
    )
    monkeypatch.setitem(_compiler_backends, "cython", cython)
    monkeypatch.setitem(_compiler_backends, "mypyc", mypyc)

    outcome = _build_typed_regions(
        prepared=prepared,
        context=_TypedRegionBuildContext(
            build_root=build_root,
            staged_source_roots=staged_source_roots,
            mypy_cache_dir=tmp_path / "mypy-cache",
            compile_cache_dir=tmp_path / "compile-cache",
            progress=None,
        ),
        initial_failures=(),
    )

    assert outcome.build.success is True
    if class_succeeds:
        assert [item.unit.region_id for item in outcome.successful] == [class_selection.variant_id]
        assert outcome.skipped == ()
        assert mypyc.calls == []
    else:
        assert [item.unit.region_id for item in outcome.successful] == [method_selection.variant_id]
        assert [failure.variant_id for failure in outcome.skipped] == [class_selection.variant_id]
        assert len(mypyc.calls) == 1
    assert len(cython.calls) == 1


def test_typed_region_build_records_real_cython_artifacts_after_mypyc_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deterministic retry compiles a real Cython artifact owned by its variant."""
    project_root = tmp_path / "typed_region_project"
    build_root = tmp_path / "build"
    shutil.copytree(TYPED_FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    scans = _selected_scans(project, "typed_region_project.worker")
    selections = _selected_typed_regions(scans)
    mypyc_selection = next(selection for selection in selections if selection.backend == "mypyc")
    staged_source_roots = _copy_source_roots(project, build_root)
    prepared = _prepare_typed_region(
        project=project,
        build_root=build_root,
        staged_source_roots=staged_source_roots,
        selection=mypyc_selection,
    )
    assert prepared.fallback is not None
    mypyc = _FakeCompileBackend(
        BackendCompileResult(
            attempt=CompileAttempt(
                success=False,
                command=("mypyc",),
                stdout="",
                stderr="MYPYC_TYPE_ERROR: fixture",
                artifact_paths=(),
                duration_seconds=0.1,
            ),
            artifacts=(),
        )
    )
    monkeypatch.setitem(_compiler_backends, "mypyc", mypyc)

    outcome = _build_typed_regions(
        prepared=(prepared,),
        context=_TypedRegionBuildContext(
            build_root=build_root,
            staged_source_roots=staged_source_roots,
            mypy_cache_dir=tmp_path / "mypy-cache",
            compile_cache_dir=tmp_path / "compile-cache",
            progress=None,
        ),
        initial_failures=(),
    )

    successful = outcome.successful[0]
    variant = CompiledRegionVariant(
        id=successful.unit.region_id,
        region=successful.generation.region,
        backend=successful.generation.backend,
        bindings=successful.generation.bindings,
    )
    report = build_compilation_report(
        CompilationReportInput(
            root=project_root,
            operation="compile",
            module_filter="typed_region_project.worker",
            islands=(),
            build=outcome.build,
            typed_regions=(variant.region,),
            compiled_regions=(variant.region,),
            compiled_bindings=variant.bindings,
            compiled_variants=(variant,),
            backend_assessments=(successful.assessment,),
            artifact_records=outcome.artifacts,
        )
    )

    assert outcome.build.success is True
    assert successful.generation.backend == "cython"
    assert successful.unit.region_id.endswith("@cython-mypyc-fallback")
    assert outcome.artifacts
    assert all(path.is_file() for path in outcome.build.artifact_paths)
    assert {artifact.region_id for artifact in outcome.artifacts} == {variant.id}
    assert report["compiled_regions"][0]["backend"] == "cython"
    assert report["compiled_regions"][0]["variant_id"] == variant.id
    assert report["compiled_regions"][0]["artifacts"]


def test_package_whole_project_compiles_regions_without_legacy_batch_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whole-project package mode compiles each typed region without sidecar batching."""
    project_root = tmp_path / "project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    package_dir = project_root / "src" / "app"
    ranking_source = package_dir / "ranking.py"
    (package_dir / "good.py").write_text(ranking_source.read_text(encoding="utf-8"))
    (package_dir / "bad.py").write_text(ranking_source.read_text(encoding="utf-8"))
    ranking_source.unlink()
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]

    def mixed_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        assert kwargs
        assert args
        paths = cast(tuple[Path, ...], args[0])
        assert paths
        if len(paths) > 1:
            return CompileAttempt(
                success=False,
                command=("mypyc", "batch"),
                stdout="",
                stderr="MYPYC_TYPE_ERROR: batch failed",
                artifact_paths=(),
                duration_seconds=0.1,
            )
        path = next(iter(paths))
        if path.stem.endswith("_good"):
            artifact = tmp_path / f"{path.stem}{suffix}"
            artifact.write_text("binary", encoding="utf-8")
            return CompileAttempt(
                success=True,
                command=("mypyc", str(path)),
                stdout="",
                stderr="",
                artifact_paths=(artifact,),
                duration_seconds=0.1,
            )
        return CompileAttempt(
            success=False,
            command=("mypyc", str(path)),
            stdout="",
            stderr="MYPYC_TYPE_ERROR: bad failed",
            artifact_paths=(),
            duration_seconds=0.1,
        )

    monkeypatch.setattr(package_command, "build_sidecars", mixed_build_sidecars, raising=False)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            output_dir=output_dir,
            keep_install_tree=True,
        )
    )

    good_text = (output_dir / "install" / "app" / "good.py").read_text(encoding="utf-8")
    bad_text = (output_dir / "install" / "app" / "bad.py").read_text(encoding="utf-8")
    assert result.success is True
    assert result.install_tree_kept is True
    assert result.cleanup_removed == (output_dir / "build",)
    assert result.cleanup_kept == (output_dir / "install",)
    assert result.islands == ()
    assert result.skipped == ()
    assert {binding.source.module for binding in result.compiled_bindings} == {
        "app.good",
        "app.bad",
    }
    assert "Initial batch build failed" not in result.build.stdout
    assert "# BEGIN ATOLL TYPED REGIONS: app.good" in good_text
    assert "# BEGIN ATOLL TYPED REGIONS: app.bad" in bad_text
    assert tuple((output_dir / "install" / ".atoll" / "artifacts").rglob(f"*{suffix}"))
    assert result.wheel_path is not None


def test_package_reports_progress_for_expensive_phases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Source-clean package builds expose phase progress to the CLI."""
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    messages: list[str] = []

    def successful_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        assert kwargs
        paths = cast(tuple[Path, ...], args[0])
        artifacts: list[Path] = []
        for path in paths:
            artifact = tmp_path / f"{path.stem}{suffix}"
            artifact.write_text("binary", encoding="utf-8")
            artifacts.append(artifact)
        return CompileAttempt(
            success=True,
            command=("mypyc",),
            stdout="",
            stderr="",
            artifact_paths=tuple(artifacts),
            duration_seconds=0.1,
        )

    monkeypatch.setattr(package_command, "build_sidecars", successful_build_sidecars, raising=False)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
            progress=messages.append,
        )
    )

    assert result.success is True
    assert result.scalar_analyses
    assert any(analysis.plans or analysis.rejections for analysis in result.scalar_analyses)
    assert result.call_chain_analyses
    assert any(analysis.plans or analysis.rejections for analysis in result.call_chain_analyses)
    assert any(message.startswith("discovered ") for message in messages)
    assert any(message.startswith("scanned ") for message in messages)
    assert any(message.startswith("scalar analysis proved ") for message in messages)
    assert any(message.startswith("call-chain analysis proved ") for message in messages)
    assert any(message.startswith("compiling typed region variant") for message in messages)
    assert any(message.startswith("compile cache miss") for message in messages)
    assert any(message.startswith("writing wheel") for message in messages)


def test_quality_gate_rejects_missing_source_stripped_project(tmp_path: Path) -> None:
    """Configured commands cannot silently fall back to the target checkout."""
    project = _quality_gate_project(
        tmp_path,
        ('test_command = ["python", "-c", "pass"]',),
    )
    baseline = _BaselineWheelPayload(
        wheel_path=tmp_path / "baseline.whl",
        build=_successful_attempt(),
    )

    outcome = _run_configured_quality_gate(
        project=project,
        baseline=baseline,
        compiled_payload_root=tmp_path / "compiled",
        progress=None,
    )

    assert outcome.success is False
    assert outcome.error == "quality-gate project is missing"
    assert outcome.performance.status == "invalid"


def test_quality_gate_rejects_missing_benchmark_baseline(tmp_path: Path) -> None:
    """Benchmarking requires a distinct unpacked baseline payload."""
    project = _quality_gate_project(
        tmp_path,
        (
            'test_command = ["python", "-c", "pass"]',
            'benchmark_command = ["python", "bench.py"]',
        ),
    )
    baseline = _BaselineWheelPayload(
        wheel_path=tmp_path / "baseline.whl",
        build=_successful_attempt(),
        quality_project_root=tmp_path / "quality-project",
    )

    outcome = _run_configured_quality_gate(
        project=project,
        baseline=baseline,
        compiled_payload_root=tmp_path / "compiled",
        progress=None,
    )

    assert outcome.success is False
    assert outcome.error == "baseline payload is missing"


def test_quality_gate_reuses_early_baseline_semantic_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Promotion runs only the compiled test after the baseline passed before profiling."""
    project = _quality_gate_project(
        tmp_path,
        (
            'test_command = ["python", "-c", "pass"]',
            'benchmark_command = ["python", "bench.py"]',
        ),
    )
    quality_root = tmp_path / "quality-project"
    baseline_root = tmp_path / "baseline"
    compiled_root = tmp_path / "compiled"
    quality_root.mkdir()
    baseline_root.mkdir()
    compiled_root.mkdir()
    baseline_result = CommandRunEvidence(
        command=("python", "-c", "pass"),
        project_root=quality_root,
        payload_root=baseline_root,
        mode="baseline",
        returncode=0,
        stdout="",
        stderr="",
        duration_seconds=0.2,
    )
    baseline = _BaselineWheelPayload(
        wheel_path=tmp_path / "baseline.whl",
        build=_successful_attempt(),
        baseline_install_root=baseline_root,
        quality_project_root=quality_root,
        semantic_test_result=baseline_result,
    )
    executed_modes: list[RuntimeMode] = []

    def run_test(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        mode: RuntimeMode,
    ) -> CommandRunEvidence:
        executed_modes.append(mode)
        return CommandRunEvidence(
            command=command,
            project_root=project_root,
            payload_root=payload_root,
            mode=mode,
            returncode=0,
            stdout="",
            stderr="",
            duration_seconds=0.2,
        )

    def pass_benchmark(*args: object, **kwargs: object) -> BenchmarkGateResult:
        assert len(args) == 1
        assert kwargs
        return BenchmarkGateResult(
            status="passed",
            reason="fixture passed",
            minimum_speedup=1.1,
            baseline_median_seconds=1.1,
            compiled_median_seconds=1.0,
            speedup=1.1,
            warmups=(),
            samples=(),
        )

    monkeypatch.setattr(package_command, "run_performance_command", run_test)
    monkeypatch.setattr(package_command, "run_benchmark_gate", pass_benchmark)

    outcome = _run_configured_quality_gate(
        project=project,
        baseline=baseline,
        compiled_payload_root=compiled_root,
        progress=None,
    )

    assert outcome.success is True
    assert outcome.tests[0] is baseline_result
    assert [result.mode for result in outcome.tests] == ["baseline", "compiled"]
    assert executed_modes == ["compiled"]


def test_quality_gate_reports_semantic_test_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed configured test stops before benchmarking and preserves exit evidence."""
    project = _quality_gate_project(
        tmp_path,
        ('test_command = ["python", "-c", "raise SystemExit(9)"]',),
    )
    quality_root = tmp_path / "quality-project"
    quality_root.mkdir()
    baseline = _BaselineWheelPayload(
        wheel_path=tmp_path / "baseline.whl",
        build=_successful_attempt(),
        quality_project_root=quality_root,
    )
    messages: list[str] = []

    def failing_test(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        mode: RuntimeMode,
    ) -> CommandRunEvidence:
        return CommandRunEvidence(
            command=command,
            project_root=project_root,
            payload_root=payload_root,
            mode=mode,
            returncode=9,
            stdout="",
            stderr="",
            duration_seconds=0.5,
        )

    monkeypatch.setattr(package_command, "run_performance_command", failing_test)

    outcome = _run_configured_quality_gate(
        project=project,
        baseline=baseline,
        compiled_payload_root=tmp_path / "compiled",
        progress=messages.append,
    )

    assert outcome.success is False
    assert outcome.error == "compiled semantic test command exited 9"
    assert outcome.tests[0].returncode == TEST_FAILURE_RETURN_CODE
    assert outcome.performance.reason == "compiled semantic test command failed"
    assert messages == ["compiled semantic tests failed with exit 9 in 0.50s"]


def test_source_clean_success_summary_lists_every_fallback_kind(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI success output distinguishes build, preflight, and typed-region fallbacks."""
    project_root = tmp_path / "typed_region_project"
    shutil.copytree(TYPED_FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    scan = _selected_scans(project, "typed_region_project.worker")[0]
    selection = next(item for item in _selected_typed_regions((scan,)) if item.region.bindings)
    island = EnabledIslandConfig(
        source_module=scan.module.name,
        source_path=scan.module.path,
        sidecar_module="_atoll_fixture",
        sidecar_path=tmp_path / "_atoll_fixture.py",
        symbols=("passthrough",),
    )
    failed_attempt = CompileAttempt(
        success=False,
        command=("compiler",),
        stdout="",
        stderr="",
        artifact_paths=(),
        duration_seconds=0.1,
    )
    result = package_command.PackageCommandResult(
        success=True,
        project_root=project_root,
        output_dir=tmp_path / "dist",
        install_root=tmp_path / "dist" / "install",
        wheel_path=tmp_path / "dist" / "fixture.whl",
        islands=(island,),
        build=_successful_attempt(),
        install_tree_kept=True,
        skipped=(package_command.PackageBuildFailure(island=island, build=failed_attempt),),
        preflight_skipped=(
            package_command.PackagePreflightFailure(
                scan=scan,
                blockers=(Blocker(severity="hard", code="module", message="module blocker"),),
            ),
            package_command.PackagePreflightFailure(
                scan=scan,
                blockers=(
                    Blocker(
                        severity="hard",
                        code="line",
                        message="line blocker",
                        lineno=7,
                    ),
                ),
            ),
        ),
        region_skipped=(
            package_command.PackageRegionBuildFailure(
                region=selection.region,
                variant_id=selection.variant_id,
                backend=selection.backend,
                assessment=selection.assessment,
                build=failed_attempt,
            ),
        ),
        performance=BenchmarkGateResult(
            status="passed",
            reason="fixture",
            minimum_speedup=1.1,
            baseline_median_seconds=1.2,
            compiled_median_seconds=1.0,
            speedup=1.2,
            warmups=(),
            samples=(),
        ),
    )

    _print_source_clean_success(
        result,
        label="source-clean compile",
        report_paths=(tmp_path / "report.json", tmp_path / "report.md"),
    )

    output = capsys.readouterr().out
    assert "Skipped 1 module(s) that mypyc could not build." in output
    assert f"- {scan.module.name}: failed" in output
    assert f"- {scan.module.name}: module: module blocker" in output
    assert f"- {scan.module.name}: line 7: line blocker" in output
    assert "Kept 1 typed region(s) as interpreted fallback." in output
    assert f"- {selection.variant_id} [{selection.backend}]: failed" in output
    assert "Install tree:" in output
    assert "Performance: 1.200x median speedup (passed)." in output


def test_plan_only_success_output_names_applied_plan_and_cache_hit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A pure Python plan overlay is not described as zero compiled work."""
    result = package_command.PackageCommandResult(
        success=True,
        project_root=tmp_path,
        output_dir=tmp_path / ".atoll" / "dist",
        install_root=tmp_path / ".atoll" / "dist" / "install",
        wheel_path=tmp_path / ".atoll" / "dist" / "fixture-0.1-py3-none-any.whl",
        islands=(),
        build=_successful_attempt(),
        applied_execution_plans=("plan-a",),
        execution_plan_trials=(
            ExecutionPlanTrial(
                plan_id="plan-a",
                status="accepted",
                command=("python", "verify.py"),
                exit_code=0,
                duration_seconds=0.1,
                cache_status="hit",
            ),
        ),
    )

    _print_source_clean_success(
        result,
        label="source-clean compile",
        report_paths=(tmp_path / "report.json", tmp_path / "report.md"),
    )

    output = capsys.readouterr().out
    assert "applied 1 async execution plan(s); no native regions were retained" in output
    assert "Execution-plan trials: 1/1 accepted; 1 staging cache hit(s)." in output
    assert (
        "Composition: 0 source optimization(s), 0 native variant(s), 1 execution plan(s)." in output
    )


def test_plan_only_promotion_preserves_the_baseline_pure_wheel_tag(tmp_path: Path) -> None:
    """A Python-only overlay cannot acquire the current interpreter platform tag."""
    project = _quality_gate_project(tmp_path, ())
    context = _SourceCleanPromotionContext(
        options=package_command.PackageOptions(root=project.config.root),
        project=project,
        output_dir=tmp_path / "dist",
        build_root=tmp_path / "dist" / "build",
        install_root=tmp_path / "dist" / "install",
        baseline=_BaselineWheelPayload(
            wheel_path=tmp_path / "fixture-0.1-py3-none-any.whl",
            build=_successful_attempt(),
        ),
        verification_plan=PackageVerificationPlan(modules=(), regions=(), artifacts=()),
        build=_successful_attempt(),
        profitable_optimization_applied=True,
    )

    assert _promotion_wheel_tag(context, tmp_path / "fixture-0.1-py3-none-any.whl") == (
        "py3-none-any"
    )
    with pytest.raises(WheelOverlayError, match="invalid filename"):
        _promotion_wheel_tag(context, tmp_path / "fixture.txt")
    with pytest.raises(WheelOverlayError, match="tag is unavailable"):
        _promotion_wheel_tag(context, tmp_path / "fixture.whl")


def test_plan_only_baseline_failure_cleans_scratch_and_returns_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed baseline leaves no plan-only scratch tree or misleading wheel."""
    project = _quality_gate_project(tmp_path, ())
    output_dir = project.config.root / ".atoll" / "dist"
    build_root = output_dir / "build"
    install_root = output_dir / "install"
    build_root.mkdir(parents=True)
    install_root.mkdir(parents=True)
    failed_wheel = output_dir / "simple_project-0.1.0-py3-none-any.whl"
    failed_wheel.write_text("stale", encoding="utf-8")
    attempt = CompileAttempt(
        success=False,
        command=("python", "-m", "build"),
        stdout="",
        stderr="baseline failed",
        artifact_paths=(),
        duration_seconds=0.1,
    )
    baseline = _BaselineWheelPayload(wheel_path=None, build=attempt)

    def package_baseline(*_args: object) -> object:
        return baseline

    monkeypatch.setattr(package_command, "_package_baseline", package_baseline)
    context = _ExecutionPlanOnlyContext(
        options=package_command.PackageOptions(root=project.config.root),
        project=project,
        typed_regions=(),
        execution_plans=(),
        prepared_baseline=None,
        profile=None,
    )

    result = _execute_execution_plan_only_package(context)

    assert result.success is False
    assert result.error == "baseline failed"
    assert result.build == attempt
    assert result.cleanup_removed == (build_root, install_root)
    assert not failed_wheel.exists()
    assert not build_root.exists()
    assert not install_root.exists()


def test_plan_only_success_forwards_trial_and_promotion_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A profitable plan-only payload reaches promotion with no native verification promises."""
    project = _quality_gate_project(tmp_path, ())
    output_dir = tmp_path / "plan-output"
    wheel_path = output_dir / "fixture-0.1-py3-none-any.whl"
    attempt = _successful_attempt()
    baseline = _BaselineWheelPayload(wheel_path=wheel_path, build=attempt)
    trial = ExecutionPlanTrial(
        plan_id="plan-a",
        status="accepted",
        command=("python", "verify.py"),
        exit_code=0,
        duration_seconds=0.1,
    )
    application = _ExecutionPlanApplicationOutcome(
        applied_plan_ids=("plan-a",),
        trials=(trial,),
        timings=(CompilePhaseTiming("execution_plan_staging", 0.1),),
    )
    promotion = _SourceCleanPromotionResult(
        success=True,
        wheel_path=wheel_path,
        build=attempt,
        verification_steps=(),
    )

    def package_baseline(*_args: object) -> object:
        return baseline

    def apply_execution_plan_trials(_context: object) -> object:
        return application

    monkeypatch.setattr(package_command, "_package_baseline", package_baseline)
    monkeypatch.setattr(
        package_command, "_apply_execution_plan_trials", apply_execution_plan_trials
    )

    def promote(context: object) -> object:
        view = cast(_PromotionContextView, context)
        assert view.verification_plan.modules == ()
        assert view.verification_plan.regions == ()
        assert view.verification_plan.artifacts == ()
        assert view.requires_profitable_optimization is True
        assert view.profitable_optimization_applied is True
        return promotion

    monkeypatch.setattr(package_command, "_promote_source_clean_payload", promote)
    context = _ExecutionPlanOnlyContext(
        options=package_command.PackageOptions(root=project.config.root, output_dir=output_dir),
        project=project,
        typed_regions=(),
        execution_plans=(),
        prepared_baseline=baseline,
        profile=None,
    )

    result = _execute_execution_plan_only_package(context)

    assert result.success is True
    assert result.wheel_path == wheel_path
    assert result.applied_execution_plans == ("plan-a",)
    assert result.execution_plan_trials == (trial,)


def test_package_rejects_not_profitable_wheel_after_semantic_tests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured benchmark below threshold removes the candidate wheel."""
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    pyproject = project_root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8")
        + "\n".join(
            (
                "",
                "[tool.atoll.compile]",
                'test_command = ["python", "-m", "pytest", "-q"]',
                'benchmark_command = ["python", "bench.py"]',
                "benchmark_warmups = 0",
                "benchmark_samples = 1",
                "minimum_speedup = 1.10",
                "",
            )
        ),
        encoding="utf-8",
    )
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    test_modes: list[RuntimeMode] = []
    target_project_root = project_root

    def successful_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        assert kwargs
        path = cast(tuple[Path, ...], args[0])[0]
        artifact = tmp_path / f"{path.stem}{suffix}"
        artifact.write_text("binary", encoding="utf-8")
        return CompileAttempt(
            success=True,
            command=("mypyc",),
            stdout="",
            stderr="",
            artifact_paths=(artifact,),
            duration_seconds=0.1,
        )

    def passing_test_command(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        mode: RuntimeMode,
    ) -> CommandRunEvidence:
        test_modes.append(mode)
        assert project_root != target_project_root
        assert not tuple((project_root / "src").rglob("*.py"))
        return CommandRunEvidence(
            command=command,
            project_root=project_root,
            payload_root=payload_root,
            mode=mode,
            returncode=0,
            stdout="passed",
            stderr="",
            duration_seconds=0.5,
        )

    def rejecting_benchmark(
        config: BenchmarkGateConfig,
        *,
        project_root: Path,
        baseline_payload_root: Path,
        compiled_payload_root: Path,
        progress: Callable[[BenchmarkProgress], None] | None = None,
    ) -> BenchmarkGateResult:
        assert config.command == ("python", "bench.py")
        assert project_root == project_root.resolve()
        assert baseline_payload_root != compiled_payload_root
        assert progress is not None
        return BenchmarkGateResult(
            status="not-profitable",
            reason="compiled median speedup 1.020 is below threshold 1.100",
            minimum_speedup=1.1,
            baseline_median_seconds=1.02,
            compiled_median_seconds=1.0,
            speedup=1.02,
            warmups=(),
            samples=(),
        )

    def insufficient_profile(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        module_paths: tuple[tuple[str, str], ...],
        **options: object,
    ) -> ProfileResult:
        scratch_dir = cast(Path, options["scratch_dir"])
        observation_targets = cast(tuple[SymbolId, ...], options["observation_targets"])
        assert command == ("python", "bench.py")
        assert project_root != target_project_root
        assert payload_root.is_dir()
        assert module_paths
        assert scratch_dir.name == "profile"
        assert observation_targets == ()
        return replace(
            unconfigured_profile(),
            status="static-fallback",
            reason="insufficient baseline profile samples: observed 90, required 100",
            launch_kind="script",
            total_samples=90,
        )

    monkeypatch.setattr(package_command, "build_sidecars", successful_build_sidecars, raising=False)
    monkeypatch.setattr(package_command, "run_performance_command", passing_test_command)
    monkeypatch.setattr(package_command, "run_baseline_profile", insufficient_profile)
    monkeypatch.setattr(package_command, "run_benchmark_gate", rejecting_benchmark)

    def reject_static_bulk_compile(**_kwargs: object) -> object:
        raise AssertionError("insufficient profile evidence must not compile every static region")

    monkeypatch.setattr(
        package_command,
        "_execute_typed_region_package",
        reject_static_bulk_compile,
    )

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
        )
    )

    assert result.success is False
    assert result.wheel_path is None
    assert result.performance is not None
    assert result.performance.status == "not-profitable"
    assert result.error == result.performance.reason
    assert test_modes == ["baseline", "compiled"]
    assert not tuple(output_dir.glob("*.whl"))
    assert not (output_dir / "install").exists()
    assert not (output_dir / "build").exists()
    assert result.cleanup_removed == (output_dir / "build", output_dir / "install")
    assert not result.verification_steps[-1].target.exists()


def test_profiled_promotion_rejects_a_wheel_without_profitable_regions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full gate still runs, but an all-rejected profile cannot publish a no-op wheel."""
    project = _quality_gate_project(tmp_path, ())
    output_dir = tmp_path / "out"
    build_root = output_dir / "build"
    install_root = output_dir / "install"
    build_root.mkdir(parents=True)
    install_root.mkdir(parents=True)
    baseline_wheel = build_root / "fixture-0.1-py3-none-any.whl"
    baseline_wheel.write_bytes(b"baseline")
    gate_calls = 0

    def repack(**kwargs: object) -> Path:
        candidate_output = cast(Path, kwargs["output_dir"])
        candidate_output.mkdir(parents=True, exist_ok=True)
        candidate = candidate_output / "candidate.whl"
        candidate.write_bytes(b"candidate")
        return candidate

    def pass_full_gate(**kwargs: object) -> object:
        nonlocal gate_calls
        assert kwargs
        gate_calls += 1
        return _QualityGateOutcome(
            success=True,
            tests=(),
            performance=BenchmarkGateResult(
                status="passed",
                reason="fixture full gate passed",
                minimum_speedup=0.5,
                baseline_median_seconds=1.0,
                compiled_median_seconds=1.0,
                speedup=1.0,
                warmups=(),
                samples=(),
            ),
        )

    monkeypatch.setattr(package_command, "repack_overlaid_wheel", repack)
    monkeypatch.setattr(package_command, "_run_configured_quality_gate", pass_full_gate)

    result = _promote_source_clean_payload(
        _SourceCleanPromotionContext(
            options=package_command.PackageOptions(root=project.config.root),
            project=project,
            output_dir=output_dir,
            build_root=build_root,
            install_root=install_root,
            baseline=_BaselineWheelPayload(
                wheel_path=baseline_wheel,
                build=_successful_attempt(),
            ),
            verification_plan=PackageVerificationPlan(
                modules=(),
                regions=(),
                artifacts=(),
            ),
            build=_successful_attempt(),
            requires_profitable_optimization=True,
        )
    )

    assert gate_calls == 1
    assert result.success is False
    assert result.wheel_path is None
    assert result.error == "no profile-guided candidate met its marginal speedup threshold"
    assert not tuple(output_dir.glob("*.whl"))
    assert not build_root.exists()
    assert not install_root.exists()
    assert result.cleanup_removed == (build_root, install_root)


def test_package_profiles_before_backend_selection_and_scopes_hot_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured benchmark selects its hot member before backend assessment."""
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    pyproject = project_root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8")
        + """

[tool.atoll.compile]
test_command = ["python", "-c", "pass"]
benchmark_command = ["python", "bench.py"]
benchmark_warmups = 0
benchmark_samples = 1
minimum_speedup = 1.10
""",
        encoding="utf-8",
    )
    target_project_root = project_root
    events: list[str] = []
    progress_messages: list[str] = []
    original_selection = _selected_typed_regions

    def run_test(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        mode: RuntimeMode,
        **options: object,
    ) -> CommandRunEvidence:
        del options
        events.append(f"test:{mode}")
        return CommandRunEvidence(
            command=command,
            project_root=project_root,
            payload_root=payload_root,
            mode=mode,
            returncode=0,
            stdout="",
            stderr="",
            duration_seconds=0.2,
        )

    def profile_baseline(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        module_paths: tuple[tuple[str, str], ...],
        **options: object,
    ) -> ProfileResult:
        scratch_dir = cast(Path, options["scratch_dir"])
        observation_targets = cast(tuple[SymbolId, ...], options["observation_targets"])
        events.append("profile")
        assert command == ("python", "bench.py")
        assert project_root != target_project_root
        assert payload_root.is_dir()
        assert ("app.ranking", "app/ranking.py") in module_paths
        assert scratch_dir.name == "profile"
        assert observation_targets == (SymbolId("app.ranking", "normalize_features"),)
        return replace(
            unconfigured_profile(),
            status="profiled",
            reason="fixture profile",
            launch_kind="script",
            total_samples=200,
            mapped_project_samples=180,
            mapped_coverage=0.9,
            lifecycle=LifecycleCounts(
                start=10,
                return_=10,
                yield_=0,
                resume=0,
                unwind=0,
                throw=0,
            ),
            members=(
                ProfiledMember(
                    module="app.ranking",
                    qualname="rank_candidates",
                    samples=180,
                    coverage=0.9,
                    call_count=10,
                    lifecycle=LifecycleCounts(
                        start=10,
                        return_=10,
                        yield_=0,
                        resume=0,
                        unwind=0,
                        throw=0,
                    ),
                    signatures=(),
                    polymorphic_overflow=False,
                ),
            ),
        )

    def record_selection(*args: object, **kwargs: object) -> tuple[_TypedSelection, ...]:
        events.append("selection")
        return original_selection(*args, **kwargs)

    def pass_benchmark(*args: object, **kwargs: object) -> BenchmarkGateResult:
        assert len(args) == 1
        assert kwargs
        progress = cast(Callable[[BenchmarkProgress], None], kwargs["progress"])
        progress(
            BenchmarkProgress(
                phase="sample",
                pair_index=1,
                sample_index=1,
                mode="baseline",
                duration_seconds=0.125,
            )
        )
        return BenchmarkGateResult(
            status="passed",
            reason="fixture passed",
            minimum_speedup=1.1,
            baseline_median_seconds=1.1,
            compiled_median_seconds=1.0,
            speedup=1.1,
            warmups=(),
            samples=(),
        )

    def fusion_targets(scans: tuple[ModuleScan, ...]) -> tuple[str, ...]:
        assert scans
        return ("app.ranking::normalize_features",)

    monkeypatch.setattr(package_command, "run_performance_command", run_test)
    monkeypatch.setattr(package_command, "run_baseline_profile", profile_baseline)
    monkeypatch.setattr(package_command, "fusion_observation_targets", fusion_targets)
    monkeypatch.setattr(package_command, "_selected_typed_regions", record_selection)
    monkeypatch.setattr(package_command, "run_benchmark_gate", pass_benchmark)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
            progress=progress_messages.append,
        )
    )

    assert result.success is True
    assert result.profile is not None
    assert result.profile.selected_symbols == (SymbolId("app.ranking", "rank_candidates"),)
    assert result.profile.selected_hot_coverage == 1.0
    assert events == [
        "selection",
        "test:baseline",
        "profile",
        "selection",
        "test:compiled",
        "test:compiled",
    ]
    assert "benchmark sample pair 1 baseline completed in 0.12s" in progress_messages
    assert {binding.source.qualname for binding in result.compiled_bindings} == {"rank_candidates"}


def test_profile_compile_plan_replays_first_selection_and_invalidates_on_source_change(
    tmp_path: Path,
) -> None:
    """Fresh profile jitter cannot create a new warm native artifact plan."""
    project = _quality_gate_project(
        tmp_path,
        (
            'test_command = ["python", "-c", "pass"]',
            'benchmark_command = ["python", "bench.py"]',
        ),
    )
    scans = _selected_scans(project, "app.ranking")
    rank = SymbolId("app.ranking", "rank_candidates")
    score = SymbolId("app.ranking", "score_user")
    profile_symbols = tuple(
        symbol for scan in scans for symbol in scan.symbols if symbol.id in {rank, score}
    )
    warm_rank_samples = 20
    warm_score_samples = 140
    expected_cache_misses = 2
    cache_root = tmp_path / "compile-cache"
    progress: list[str] = []
    options = package_command.PackageOptions(
        root=project.config.root,
        module_name="app.ranking",
        cache_dir=cache_root,
        progress=progress.append,
    )

    def profile(selected: SymbolId, *, rank_samples: int, score_samples: int) -> ProfileResult:
        members = tuple(
            ProfiledMember(
                module=symbol.module,
                qualname=symbol.qualname,
                samples=samples,
                coverage=samples / 200,
                call_count=10,
                lifecycle=LifecycleCounts(
                    start=10,
                    return_=10,
                    yield_=0,
                    resume=0,
                    unwind=0,
                    throw=0,
                ),
                signatures=(),
                polymorphic_overflow=False,
            )
            for symbol, samples in ((rank, rank_samples), (score, score_samples))
        )
        fresh = replace(
            unconfigured_profile(),
            status="profiled",
            reason="fresh fixture profile",
            launch_kind="script",
            total_samples=200,
            mapped_project_samples=rank_samples + score_samples,
            mapped_coverage=(rank_samples + score_samples) / 200,
            members=members,
        )
        ranked = select_profile_candidates(fresh, profile_symbols)
        assert ranked.selected_symbols == (selected,)
        return ranked

    full_support = _profile_candidate_support(scans, project.config.compile.backends)
    cold_scope = _ProfileCompileSelectionScope(
        identity="baseline",
        support=full_support,
    )
    warm_scope = _ProfileCompileSelectionScope(
        identity="baseline",
        support=full_support,
    )
    cold = _stabilize_profile_compile_selection(
        cold_scope,
        options=options,
        project=project,
        scans=scans,
        profile=profile(rank, rank_samples=140, score_samples=20),
    )
    warm = _stabilize_profile_compile_selection(
        warm_scope,
        options=options,
        project=project,
        scans=scans,
        profile=profile(
            score,
            rank_samples=warm_rank_samples,
            score_samples=warm_score_samples,
        ),
    )
    empty_fresh_profile = profile(
        score,
        rank_samples=warm_rank_samples,
        score_samples=warm_score_samples,
    )
    empty_warm = _stabilize_profile_compile_selection(
        warm_scope,
        options=options,
        project=project,
        scans=scans,
        profile=replace(
            empty_fresh_profile,
            candidates=tuple(
                replace(candidate, selected=False, reason="below-threshold")
                for candidate in empty_fresh_profile.candidates
            ),
            selected_symbols=(),
            selected_hot_samples=0,
            selected_hot_coverage=0.0,
        ),
    )

    assert cold is not None
    assert warm is not None
    assert empty_warm is not None
    assert cold.selected_symbols == (rank,)
    assert warm.selected_symbols == (rank,)
    assert empty_warm.selected_symbols == (rank,)
    assert empty_warm.selected_hot_samples == warm_rank_samples
    assert warm.selected_hot_samples == warm_rank_samples
    assert warm.selected_hot_coverage == pytest.approx(0.125)
    assert "native candidate selection replayed from strict cache" in warm.reason
    assert [(candidate.symbol, candidate.reason) for candidate in warm.candidates] == [
        (score, "cache-replay-excluded"),
        (rank, "cache-replayed"),
    ]
    assert any("profile compile plan cache miss" in message for message in progress)
    assert any("profile compile plan cache hit" in message for message in progress)

    module_path = next(scan.module.path for scan in scans if scan.module.name == "app.ranking")
    module_path.write_text(module_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    changed_project = discover_project(project.config.root)
    changed_scans = _selected_scans(changed_project, "app.ranking")
    changed = _stabilize_profile_compile_selection(
        _ProfileCompileSelectionScope(
            identity="baseline",
            support=_profile_candidate_support(
                changed_scans,
                changed_project.config.compile.backends,
            ),
        ),
        options=options,
        project=changed_project,
        scans=changed_scans,
        profile=profile(
            score,
            rank_samples=warm_rank_samples,
            score_samples=warm_score_samples,
        ),
    )

    assert changed is not None
    assert changed.selected_symbols == (score,)
    assert (
        sum("profile compile plan cache miss" in message for message in progress)
        == expected_cache_misses
    )


def test_profile_compile_replay_restores_candidates_missing_from_fresh_ranking(
    tmp_path: Path,
) -> None:
    """A strict cache hit reconstructs observed and zero-sample cached members."""
    rank = SymbolId("app.ranking", "rank_candidates")
    score = SymbolId("app.ranking", "score_user")
    missing = SymbolId("app.ranking", "cached_but_unobserved")
    lifecycle = LifecycleCounts(
        start=10,
        return_=10,
        yield_=0,
        resume=0,
        unwind=0,
        throw=0,
    )
    profile = replace(
        unconfigured_profile(),
        status="profiled",
        reason="fresh fixture profile",
        total_samples=200,
        mapped_project_samples=120,
        members=(
            ProfiledMember(
                module=rank.module,
                qualname=rank.qualname,
                samples=PROFILE_REPLAY_RANK_SAMPLES,
                coverage=0.6,
                call_count=10,
                lifecycle=lifecycle,
                signatures=(),
                polymorphic_overflow=False,
            ),
        ),
        candidates=(
            MappedCandidateDecision(
                symbol=score,
                module=score.module,
                qualname=score.qualname,
                samples=80,
                coverage=0.4,
                scheduler_overhead_samples=0,
                attributed_samples=80,
                attributed_coverage=0.4,
                selected=True,
                reason="selected",
            ),
        ),
        selected_symbols=(score,),
    )
    decision = ProfilePlanDecision(
        status="hit",
        selection=(rank, missing),
        diagnostic="fixture replay",
        cache_path=tmp_path / "plan.json",
        identity_digest="a" * 64,
    )

    replayed = _profile_with_replayed_compile_selection(profile, decision)

    assert replayed.selected_symbols == (rank, missing)
    assert replayed.selected_hot_samples == PROFILE_REPLAY_RANK_SAMPLES
    assert [(item.symbol, item.samples, item.reason) for item in replayed.candidates] == [
        (score, 80, "cache-replay-excluded"),
        (rank, PROFILE_REPLAY_RANK_SAMPLES, "cache-replayed"),
        (missing, 0, "cache-replayed"),
    ]


def test_conditional_task_fusion_trials_disposable_payload_after_safe_miss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _quality_gate_project(
        tmp_path,
        (
            'test_command = ["python", "verify.py"]',
            'benchmark_command = ["python", "bench.py"]',
        ),
    )
    source_path = project.config.source_roots[0] / "app" / "fusion_case.py"
    source_text = (
        "import asyncio\n\n"
        "async def worker(value: int) -> int:\n"
        "    return value + 1\n\n"
        "async def root(value: int):\n"
        "    return asyncio.create_task(worker(value))\n"
    )
    source_path.write_text(source_text, encoding="utf-8")
    project = discover_project(project.config.root)
    spawn = next(
        node
        for node in ast.walk(ast.parse(source_text))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "create_task"
    )
    plan = FusionPlan(
        id="task-fusion:fixture",
        source_hash="source-hash",
        root="app.fusion_case::root",
        caller="app.fusion_case::root",
        callee="app.fusion_case::worker",
        spawn_api="asyncio.create_task",
        lineno=spawn.lineno,
        end_lineno=spawn.end_lineno or spawn.lineno,
        col_offset=spawn.col_offset,
        end_col_offset=spawn.end_col_offset,
        eligible=True,
        observed_calls=25,
        completed_calls=25,
        max_active_calls=1,
        pre_completion_suspensions=0,
        observed_signatures=1,
        observation_capped=False,
        rejections=(),
        spawn_source=ast.get_source_segment(source_text, spawn) or "",
    )
    install_root = tmp_path / "install"
    baseline_root = tmp_path / "baseline"
    installed_source = install_root / "app" / "fusion_case.py"
    installed_source.parent.mkdir(parents=True)
    installed_source.write_text(source_text, encoding="utf-8")
    shutil.copytree(install_root, baseline_root)
    build_root = tmp_path / "build"
    build_root.mkdir()
    calls: list[str] = []

    def run_trial(
        config: FusionBenchmarkConfig,
        **kwargs: object,
    ) -> FusionTrial:
        calls.append(config.plan_id)
        assert config.minimum_over_unfused == pytest.approx(1.05)
        assert config.minimum_overall == pytest.approx(project.config.compile.minimum_speedup)
        fused_root = cast(Path, kwargs["fused_payload_root"])
        transformed = (fused_root / "app" / "fusion_case.py").read_text(encoding="utf-8")
        assert "_atoll_eager_spawn_fixture" in transformed
        assert installed_source.read_text(encoding="utf-8") == source_text
        assert source_path.read_text(encoding="utf-8") == source_text
        return FusionTrial(
            plan_id=config.plan_id,
            status="not-profitable",
            reason="fixture ratios missed thresholds",
            semantic_runs=(),
            baseline_median_seconds=1.0,
            unfused_median_seconds=1.0,
            fused_median_seconds=1.0,
            baseline_over_unfused=1.0,
            baseline_over_fused=1.0,
            unfused_over_fused=1.0,
            warmups=(),
            samples=(),
        )

    monkeypatch.setattr(package_command, "run_fusion_trial", run_trial)
    performance = BenchmarkGateResult(
        status="not-profitable",
        reason="safe payload below threshold",
        minimum_speedup=1.1,
        baseline_median_seconds=1.0,
        compiled_median_seconds=1.0,
        speedup=1.0,
        warmups=(),
        samples=(),
    )

    outcome = _run_conditional_task_fusion_research(
        _FusionResearchContext(
            options=package_command.PackageOptions(root=project.config.root),
            project=project,
            baseline=_BaselineWheelPayload(
                wheel_path=tmp_path / "baseline.whl",
                build=_successful_attempt(),
                baseline_install_root=baseline_root,
                quality_project_root=project.config.root,
            ),
            build_root=build_root,
            install_root=install_root,
            plans=(plan,),
            accepted=(),
            performance=performance,
        )
    )

    assert calls == [plan.id]
    assert [trial.plan_id for trial in outcome.trials] == [plan.id]
    assert [timing.name for timing in outcome.timings] == ["fusion_stage"]
    assert not (build_root / "fusion-research").exists()
    assert installed_source.read_text(encoding="utf-8") == source_text
    assert source_path.read_text(encoding="utf-8") == source_text


def test_conditional_task_fusion_does_not_run_after_safe_gate_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _quality_gate_project(tmp_path, ())

    def unexpected_trial(*args: object, **kwargs: object) -> FusionTrial:
        raise AssertionError(f"unexpected task-fusion trial: {args!r} {kwargs!r}")

    monkeypatch.setattr(package_command, "run_fusion_trial", unexpected_trial)
    outcome = _run_conditional_task_fusion_research(
        _FusionResearchContext(
            options=package_command.PackageOptions(root=project.config.root),
            project=project,
            baseline=_BaselineWheelPayload(
                wheel_path=None,
                build=_successful_attempt(),
            ),
            build_root=tmp_path / "build",
            install_root=tmp_path / "install",
            plans=(),
            accepted=(),
            performance=BenchmarkGateResult(
                status="passed",
                reason="safe payload passed",
                minimum_speedup=1.1,
                baseline_median_seconds=1.1,
                compiled_median_seconds=1.0,
                speedup=1.1,
                warmups=(),
                samples=(),
            ),
        )
    )

    assert outcome.trials == ()
    assert outcome.timings == ()


def test_conditional_task_fusion_records_unavailable_delegated_gate(tmp_path: Path) -> None:
    project = _quality_gate_project(tmp_path, ())
    outcome = _run_conditional_task_fusion_research(
        _FusionResearchContext(
            options=package_command.PackageOptions(
                root=project.config.root,
                run_quality_gates=False,
            ),
            project=project,
            baseline=_BaselineWheelPayload(
                wheel_path=None,
                build=_successful_attempt(),
            ),
            build_root=tmp_path / "build",
            install_root=tmp_path / "install",
            plans=(_eligible_fusion_plan(),),
            accepted=(),
            performance=_benchmark_result("not-profitable"),
        )
    )

    assert len(outcome.trials) == 1
    assert outcome.trials[0].plan_id == "task-fusion:fixture"
    assert outcome.trials[0].status == "unavailable"
    assert outcome.trials[0].reason == "quality gates are delegated to the calling workflow"


def test_conditional_task_fusion_reports_stale_staged_source(tmp_path: Path) -> None:
    project = _quality_gate_project(
        tmp_path,
        (
            'test_command = ["python", "verify.py"]',
            'benchmark_command = ["python", "bench.py"]',
        ),
    )
    source_path = project.config.source_roots[0] / "app" / "fusion_case.py"
    source_text = (
        "import asyncio\n\n"
        "async def worker(value: int) -> int:\n"
        "    return value + 1\n\n"
        "async def root(value: int):\n"
        "    return asyncio.create_task(worker(value))\n"
    )
    source_path.write_text(source_text, encoding="utf-8")
    project = discover_project(project.config.root)
    spawn = next(
        node
        for node in ast.walk(ast.parse(source_text))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "create_task"
    )
    plan = replace(
        _eligible_fusion_plan(
            caller="app.fusion_case::root",
            callee="app.fusion_case::worker",
        ),
        lineno=spawn.lineno,
        end_lineno=spawn.end_lineno or spawn.lineno,
        col_offset=spawn.col_offset,
        end_col_offset=spawn.end_col_offset,
        spawn_source="stale source",
    )
    install_root = tmp_path / "install"
    installed_source = install_root / "app" / "fusion_case.py"
    installed_source.parent.mkdir(parents=True)
    installed_source.write_text(source_text, encoding="utf-8")
    baseline_root = tmp_path / "baseline"
    shutil.copytree(install_root, baseline_root)
    build_root = tmp_path / "build"
    build_root.mkdir()

    outcome = _run_conditional_task_fusion_research(
        _FusionResearchContext(
            options=package_command.PackageOptions(root=project.config.root),
            project=project,
            baseline=_BaselineWheelPayload(
                wheel_path=None,
                build=_successful_attempt(),
                baseline_install_root=baseline_root,
                quality_project_root=project.config.root,
            ),
            build_root=build_root,
            install_root=install_root,
            plans=(plan,),
            accepted=(),
            performance=_benchmark_result("not-profitable"),
        )
    )

    assert outcome.trials[0].status == "unavailable"
    assert "spawn source changed" in outcome.trials[0].reason
    assert outcome.timings[0].detail == "task-fusion:fixture; failed"
    assert not (build_root / "fusion-research").exists()


def test_task_fusion_source_resolution_rejects_invalid_boundaries(tmp_path: Path) -> None:
    project = _quality_gate_project(tmp_path, ())
    payload_root = tmp_path / "payload"

    with pytest.raises(ValueError, match="no module identity"):
        _task_fusion_source_path(
            project,
            payload_root,
            _eligible_fusion_plan(caller="malformed"),
        )
    with pytest.raises(ValueError, match="not part of the project"):
        _task_fusion_source_path(
            project,
            payload_root,
            _eligible_fusion_plan(caller="missing.module::root"),
        )
    with pytest.raises(ValueError, match="installed source is unavailable"):
        _task_fusion_source_path(
            project,
            payload_root,
            _eligible_fusion_plan(caller="app.ranking::root"),
        )


def test_fusion_trial_timings_preserve_arm_and_phase() -> None:
    run = CommandRunEvidence(
        command=("python", "bench.py"),
        project_root=Path("project"),
        payload_root=Path("payload"),
        mode="compiled",
        returncode=0,
        stdout="",
        stderr="",
        duration_seconds=0.5,
    )
    trial = FusionTrial(
        plan_id="task-fusion:fixture",
        status="passed",
        reason="fixture passed",
        semantic_runs=(FusionArmRunEvidence(arm="baseline", run=run),),
        baseline_median_seconds=1.2,
        unfused_median_seconds=1.1,
        fused_median_seconds=1.0,
        baseline_over_unfused=1.09,
        baseline_over_fused=1.2,
        unfused_over_fused=1.1,
        warmups=(FusionArmRunEvidence(arm="unfused", run=run),),
        samples=(FusionArmRunEvidence(arm="fused", run=run),),
    )

    timings = _fusion_trial_timings(trial)

    assert [timing.name for timing in timings] == [
        "fusion_semantic_test",
        "fusion_benchmark_warmup",
        "fusion_benchmark",
    ]
    assert [timing.detail for timing in timings] == [
        "task-fusion:fixture; baseline; exit 0",
        "task-fusion:fixture; unfused",
        "task-fusion:fixture; fused; passed",
    ]


@pytest.mark.parametrize("second_candidate_semantics_pass", [True, False])
def test_package_greedily_keeps_only_profitable_profile_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    second_candidate_semantics_pass: bool,
) -> None:
    """Profile order retains only candidates passing semantics and marginal timing.

    Args:
        tmp_path: Isolated source-clean target and wheel output.
        monkeypatch: Deterministic semantic, profile, and benchmark boundaries.
        second_candidate_semantics_pass: Whether the second candidate reaches timing.
    """
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    pyproject = project_root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8")
        + """

[tool.atoll.compile]
test_command = ["python", "-c", "pass"]
benchmark_command = ["python", "bench.py"]
benchmark_warmups = 1
benchmark_samples = 7
minimum_speedup = 1.10
""",
        encoding="utf-8",
    )
    zero_lifecycle = LifecycleCounts(
        start=0,
        return_=0,
        yield_=0,
        resume=0,
        unwind=0,
        throw=0,
    )
    observed_allowlists: list[frozenset[str] | None] = []
    benchmark_thresholds: list[float] = []

    def run_test(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        mode: RuntimeMode,
        **options: object,
    ) -> CommandRunEvidence:
        variant_allowlist = cast(frozenset[str] | None, options.get("variant_allowlist"))
        region_allowlist = cast(frozenset[str] | None, options.get("region_allowlist"))
        observed_allowlists.append(variant_allowlist or region_allowlist)
        compiled_count = _compiled_region_marker_count(payload_root)
        failed = (
            mode == "compiled"
            and compiled_count == SECOND_CANDIDATE_REGION_COUNT
            and not second_candidate_semantics_pass
        )
        return CommandRunEvidence(
            command=command,
            project_root=project_root,
            payload_root=payload_root,
            mode=mode,
            returncode=9 if failed else 0,
            stdout="",
            stderr="second candidate failed semantics" if failed else "",
            duration_seconds=0.5,
        )

    def profile_baseline(*args: object, **kwargs: object) -> ProfileResult:
        assert args
        assert kwargs
        return replace(
            unconfigured_profile(),
            status="profiled",
            reason="two hot fixture members",
            launch_kind="script",
            total_samples=200,
            mapped_project_samples=180,
            mapped_coverage=0.9,
            lifecycle=zero_lifecycle,
            members=(
                ProfiledMember(
                    module="app.ranking",
                    qualname="rank_candidates",
                    samples=120,
                    coverage=0.6,
                    call_count=10,
                    lifecycle=zero_lifecycle,
                    signatures=(),
                    polymorphic_overflow=False,
                ),
                ProfiledMember(
                    module="app.ranking",
                    qualname="normalize_features",
                    samples=60,
                    coverage=0.3,
                    call_count=10,
                    lifecycle=zero_lifecycle,
                    signatures=(),
                    polymorphic_overflow=False,
                ),
            ),
        )

    def benchmark(
        config: BenchmarkGateConfig,
        **kwargs: object,
    ) -> BenchmarkGateResult:
        benchmark_thresholds.append(config.minimum_speedup)
        if config.minimum_speedup == _CANDIDATE_SPEEDUP:
            candidate_index = benchmark_thresholds.count(_CANDIDATE_SPEEDUP)
            speedup = 1.02 if candidate_index == 1 else 1.005
            status: BenchmarkStatus = "passed" if candidate_index == 1 else "not-profitable"
            baseline_payload_root = cast(Path, kwargs["baseline_payload_root"])
            compiled_payload_root = cast(Path, kwargs["compiled_payload_root"])
            assert baseline_payload_root != compiled_payload_root
            baseline_source = (baseline_payload_root / "app" / "ranking.py").read_text(
                encoding="utf-8"
            )
            compiled_source = (compiled_payload_root / "app" / "ranking.py").read_text(
                encoding="utf-8"
            )
            assert baseline_source.count("'compiled_module':") == candidate_index - 1
            assert compiled_source.count("'compiled_module':") == candidate_index
            assert "baseline_variant_allowlist" in kwargs
            assert "compiled_variant_allowlist" not in kwargs
        else:
            speedup = 1.12
            status = "passed"
            assert kwargs["baseline_payload_root"] != kwargs["compiled_payload_root"]
        return BenchmarkGateResult(
            status=status,
            reason=f"fixture speedup {speedup:.3f}",
            minimum_speedup=config.minimum_speedup,
            baseline_median_seconds=speedup,
            compiled_median_seconds=1.0,
            speedup=speedup,
            warmups=(),
            samples=(),
        )

    monkeypatch.setattr(package_command, "run_performance_command", run_test)
    monkeypatch.setattr(package_command, "run_baseline_profile", profile_baseline)
    monkeypatch.setattr(package_command, "run_benchmark_gate", benchmark)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
        )
    )

    assert result.success is True
    assert result.wheel_path is not None
    assert [trial.status for trial in result.candidate_trials] == [
        "accepted",
        "rejected" if second_candidate_semantics_pass else "failed-semantics",
    ]
    assert result.candidate_trials[0].symbols == ("app.ranking::rank_candidates",)
    assert result.candidate_trials[1].symbols == ("app.ranking::normalize_features",)
    assert result.candidate_trials[0].accepted_hot_coverage == pytest.approx(2 / 3)
    assert result.candidate_trials[1].accepted_hot_coverage == pytest.approx(2 / 3)
    assert result.candidate_trials[0].marginal_speedup == pytest.approx(1.02)
    assert result.performance is not None
    assert result.performance.speedup == pytest.approx(1.12)
    assert {binding.source.qualname for binding in result.compiled_bindings} == {"rank_candidates"}
    expected_thresholds = (
        [_CANDIDATE_SPEEDUP, _CANDIDATE_SPEEDUP, 1.1]
        if second_candidate_semantics_pass
        else [_CANDIDATE_SPEEDUP, 1.1]
    )
    assert benchmark_thresholds == expected_thresholds
    assert observed_allowlists[0] is None
    assert observed_allowlists[1:] == [None, None, None]
    assert len(result.test_results) == EXPECTED_FINAL_TEST_RESULTS
    with zipfile.ZipFile(result.wheel_path) as wheel:
        native_entries = {
            name
            for name in wheel.namelist()
            if any(name.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES)
        }
    assert native_entries == {record.install_relative_path for record in result.artifact_records}


def test_payload_bytecode_cleanup_removes_caches_without_following_symlinks(
    tmp_path: Path,
) -> None:
    """Pre-gate cleanup removes timing bias without escaping the owned payload."""
    payload = tmp_path / "payload"
    cache = payload / "pkg" / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "module.cpython-312.pyc").write_bytes(b"cached")
    standalone = payload / "legacy.pyo"
    standalone.write_bytes(b"legacy")
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_marker = outside / "keep.pyc"
    outside_marker.write_bytes(b"keep")
    linked_cache = payload / "linked" / "__pycache__"
    linked_cache.parent.mkdir()
    linked_cache.symlink_to(outside, target_is_directory=True)

    removed = _clear_payload_bytecode((payload,))

    assert cache in removed
    assert standalone in removed
    assert linked_cache in removed
    assert not cache.exists()
    assert not standalone.exists()
    assert not linked_cache.exists()
    assert outside_marker.read_bytes() == b"keep"

    progress_cache = payload / "pkg" / "__pycache__"
    progress_cache.mkdir(parents=True)
    messages: list[str] = []
    _clear_payload_bytecode_with_progress((tmp_path / "missing", payload), messages.append)

    assert messages == ["removed 1 pre-existing bytecode cache path(s)"]
    assert not progress_cache.exists()


def test_package_stops_before_profiling_when_baseline_semantics_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed interpreted baseline test prevents profile and native work."""
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    pyproject = project_root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8")
        + """

[tool.atoll.compile]
test_command = ["python", "-c", "raise SystemExit(9)"]
benchmark_command = ["python", "bench.py"]
""",
        encoding="utf-8",
    )

    def fail_baseline(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        mode: RuntimeMode,
    ) -> CommandRunEvidence:
        assert mode == "baseline"
        return CommandRunEvidence(
            command=command,
            project_root=project_root,
            payload_root=payload_root,
            mode=mode,
            returncode=9,
            stdout="",
            stderr="baseline fixture failed",
            duration_seconds=0.2,
        )

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("work continued after the baseline semantic failure")

    monkeypatch.setattr(package_command, "run_performance_command", fail_baseline)
    monkeypatch.setattr(package_command, "run_baseline_profile", forbidden)
    monkeypatch.setattr(package_command, "_build_typed_regions", forbidden)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
        )
    )

    assert result.success is False
    assert result.error == "baseline fixture failed"
    assert result.profile is None
    assert [run.mode for run in result.test_results] == ["baseline"]
    assert result.performance is not None
    assert result.performance.status == "invalid"
    assert not tuple(output_dir.glob("*.whl"))


def test_apply_source_rejects_non_git_root_before_baseline_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit source application rejects a non-Git root before expensive setup."""
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    pyproject = project_root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8")
        + """

[tool.atoll.compile]
test_command = ["python", "-c", "pass"]
benchmark_command = ["python", "bench.py"]
""",
        encoding="utf-8",
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("baseline setup ran before source application preflight")

    monkeypatch.setattr(package_command, "_prepare_baseline_wheel_payload", forbidden)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
            apply_source=True,
        )
    )

    assert result.success is False
    assert result.error == f"source application root is not a Git work tree: {project_root}"
    assert not output_dir.exists()


def test_apply_source_rejects_missing_quality_commands_before_git_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing quality commands cannot authorize source mutation or baseline setup."""
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("baseline setup ran without a configured semantic command")

    monkeypatch.setattr(package_command, "_prepare_baseline_wheel_payload", forbidden)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
            apply_source=True,
        )
    )

    assert result.success is False
    assert result.error == "--apply-source requires configured test_command and benchmark_command"
    assert not output_dir.exists()


def test_package_routes_accepted_source_wheel_into_composition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accepted source materialization continues into later optimization stages."""
    project_root, output_dir, preparation = _prepared_source_search_project(tmp_path)
    composed = False

    def prepare(*_args: object, **_kwargs: object) -> object:
        return preparation

    def source_search(
        _plans: tuple[object, ...],
        _assessments: tuple[object, ...],
        options: SourceOptimizationSearchOptions,
    ) -> SourceOptimizationSearchResult:
        options.output_dir.mkdir(parents=True)
        wheel_path = options.output_dir / "fixture-0.1.0-py3-none-any.whl"
        wheel_path.write_bytes(b"wheel")
        return SourceOptimizationSearchResult(
            attempted=True,
            accepted=True,
            wheel_path=wheel_path,
            patch_path=project_root / ".atoll" / "patches" / "accepted.patch",
            trials=(),
            test_results=(),
            performance=_benchmark_result("passed"),
            build=_successful_attempt(),
            materialization_patch=GeneratedSourcePatch(
                patch_text="",
                source_edits=(),
                files=(),
            ),
        )

    def compose(**kwargs: object) -> package_command.PackageCommandResult:
        nonlocal composed
        composed = True
        search = cast(SourceOptimizationSearchResult, kwargs["search"])
        assert search.materialization_patch is not None
        return package_command.PackageCommandResult(
            success=True,
            project_root=project_root,
            output_dir=output_dir,
            install_root=output_dir / "install",
            wheel_path=search.wheel_path,
            islands=(),
            build=search.build,
            performance=search.performance,
        )

    monkeypatch.setattr(package_command, "_prepare_profile_guided_selection", prepare)
    monkeypatch.setattr(package_command, "run_source_optimization_search", source_search)
    monkeypatch.setattr(package_command, "_execute_composed_source_arm", compose)

    result = package_command.execute_package(
        package_command.PackageOptions(root=project_root, output_dir=output_dir)
    )

    assert result.success is True
    assert composed is True
    assert result.wheel_path == output_dir / "fixture-0.1.0-py3-none-any.whl"
    assert result.compiled_bindings == ()
    assert result.performance is not None
    assert result.performance.status == "passed"


def test_transformed_source_candidate_is_profiled_with_optimized_routing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidate selection receives a fresh profile from the staged optimized payload."""
    project_root, output_dir, _preparation = _prepared_source_search_project(tmp_path)
    quality_root = tmp_path / "candidate-quality"
    payload_root = tmp_path / "candidate-payload"
    quality_root.mkdir()
    payload_root.mkdir()
    candidate_id = "source-candidate-profiled"
    observed_options: dict[str, object] = {}
    progress: list[str] = []
    expected = replace(
        unconfigured_profile(),
        status="profiled",
        reason="fresh optimized profile",
        total_samples=100,
    )

    def selected_scans(*_args: object, **_kwargs: object) -> tuple[ModuleScan, ...]:
        return ()

    def call_chains(*_args: object) -> tuple[CallChainAnalysisResult, ...]:
        return ()

    def select_profile(
        profile: ProfileResult,
        _scans: tuple[ModuleScan, ...],
        _chains: tuple[CallChainAnalysisResult, ...],
        _backends: tuple[Backend, ...],
        **_options: object,
    ) -> ProfileResult:
        return profile

    monkeypatch.setattr(package_command, "_selected_scans", selected_scans)
    monkeypatch.setattr(package_command, "_call_chain_analyses", call_chains)
    monkeypatch.setattr(package_command, "_select_profile_with_call_chains", select_profile)

    def profile(
        _command: tuple[str, ...],
        **kwargs: object,
    ) -> ProfileResult:
        observed_options.update(kwargs)
        return expected

    monkeypatch.setattr(package_command, "run_baseline_profile", profile)

    result = _profile_source_candidate(
        package_command.PackageOptions(
            root=project_root,
            output_dir=output_dir,
            progress=progress.append,
        ),
        project_root,
        quality_root,
        payload_root,
        candidate_id,
    )

    assert result is expected
    assert observed_options["project_root"] == quality_root
    assert observed_options["payload_root"] == payload_root
    assert observed_options["enable_atoll"] is True
    assert observed_options["scratch_dir"] == project_root.parent / f"profile-{candidate_id}"
    assert progress


def test_transformed_source_candidate_requires_retained_benchmark(tmp_path: Path) -> None:
    """A transformed project that loses benchmark configuration cannot seed selection."""
    project_root = tmp_path / "project"
    shutil.copytree(FIXTURE_ROOT, project_root)

    with pytest.raises(ValueError, match="lost its benchmark command"):
        _profile_source_candidate(
            package_command.PackageOptions(root=project_root),
            project_root,
            tmp_path / "quality",
            tmp_path / "payload",
            "missing-benchmark",
        )


def test_composed_arm_prefers_profile_from_accepted_source_trial(tmp_path: Path) -> None:
    """Only an accepted trial supplies residual evidence to later native selection."""
    stale = replace(unconfigured_profile(), reason="baseline profile")
    fresh = replace(
        unconfigured_profile(),
        status="profiled",
        reason="fresh source profile",
        total_samples=100,
    )
    rejected = SourceOptimizationTrial(
        plan_id="rejected",
        status="not-profitable",
        semantic_command=(),
        benchmark_command=(),
        baseline_median_seconds=None,
        source_median_seconds=None,
        wheel_median_seconds=None,
        source_speedup=None,
        wheel_speedup=None,
        patch_path=None,
        source_edits=(),
        application_status="not-applied",
        residual_profile=stale,
    )
    accepted = replace(rejected, plan_id="accepted", status="accepted", residual_profile=fresh)
    search = SourceOptimizationSearchResult(
        attempted=True,
        accepted=True,
        wheel_path=tmp_path / "source.whl",
        patch_path=tmp_path / "source.patch",
        trials=(rejected, accepted),
        test_results=(),
        performance=None,
        build=_successful_attempt(),
    )

    assert _accepted_source_profile(search) is fresh
    assert _accepted_source_profile(replace(search, trials=(rejected,))) is None


def test_composition_fallback_preserves_source_success_and_rejection_evidence(
    tmp_path: Path,
) -> None:
    """A rejected later arm cannot invalidate an accepted source optimization."""
    project_root = tmp_path / "project"
    shutil.copytree(TYPED_FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    scan = _selected_scans(project, "typed_region_project.worker")[0]
    selection = next(item for item in _selected_typed_regions((scan,)) if item.region.bindings)
    variant = CompiledRegionVariant(
        id=selection.variant_id,
        region=selection.region,
        backend=selection.backend,
        bindings=selection.region.bindings,
    )
    artifact = ArtifactRecord(
        region_id=variant.id,
        backend=variant.backend,
        logical_module="_atoll_fixture",
        role="primary",
        install_relative_path=".atoll/artifacts/_atoll_fixture.so",
        digest="0" * 64,
        abi="cp312",
        platform_tag="test-platform",
    )
    source_result = package_command.PackageCommandResult(
        success=True,
        project_root=project_root,
        output_dir=tmp_path / "out",
        install_root=tmp_path / "install",
        wheel_path=tmp_path / "source.whl",
        islands=(),
        build=_successful_attempt(),
        performance=_benchmark_result("passed"),
    )
    rejected_attempt = CompileAttempt(
        success=False,
        command=("native",),
        stdout="native output",
        stderr="native benchmark rejected",
        artifact_paths=(),
        duration_seconds=2.5,
        phase_timings=(
            CompilePhaseTiming(
                name="native_trial",
                duration_seconds=2.5,
                detail="not profitable",
            ),
        ),
    )
    rejected = replace(
        source_result,
        success=False,
        wheel_path=None,
        build=rejected_attempt,
        error="native benchmark rejected",
        compiled_regions=(variant.region,),
        compiled_bindings=variant.bindings,
        compiled_variants=(variant,),
        artifact_records=(artifact,),
        applied_execution_plans=("rejected-plan",),
    )

    recovered = _source_result_with_composition_fallback(
        source_result,
        rejected,
        project,
    )

    assert recovered.success is True
    assert recovered.wheel_path == source_result.wheel_path
    assert recovered.performance == source_result.performance
    assert recovered.compiled_regions == ()
    assert recovered.compiled_bindings == ()
    assert recovered.compiled_variants == ()
    assert recovered.artifact_records == ()
    assert recovered.applied_execution_plans == ()
    assert "composition fallback retained: native benchmark rejected" in recovered.build.stdout
    assert recovered.build.duration_seconds == pytest.approx(2.5)
    assert recovered.build.phase_timings[-1].name == "native_trial"


def test_materialize_source_optimization_arm_recreates_disposable_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accepted source evidence recreates a transformed copy and wheel baseline."""
    project_root = tmp_path / "project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    source_path = project_root / "src" / "app" / "ranking.py"
    before_source = source_path.read_text(encoding="utf-8")
    after_source = before_source.replace("DEFAULT_WEIGHT = 1.5", "DEFAULT_WEIGHT = 2.0")
    patch = GeneratedSourcePatch(
        patch_text="fixture patch",
        source_edits=(),
        files=(
            TransformedSourceFile(
                path=PurePosixPath("src/app/ranking.py"),
                before_source=before_source,
                after_source=after_source,
            ),
        ),
    )
    output_dir.mkdir()
    wheel_path = output_dir / "simple_project-0.1.0-py3-none-any.whl"
    wheel_path.write_bytes(b"accepted wheel")

    def unpack(_wheel_path: Path, install_root: Path) -> None:
        package_root = install_root / "app"
        package_root.mkdir(parents=True)
        (package_root / "__init__.py").write_text("", encoding="utf-8")
        (package_root / "ranking.py").write_text("WHEEL_VALUE = 1\n", encoding="utf-8")

    monkeypatch.setattr(package_command, "unpack_wheel_payload", unpack)
    baseline_payload = tmp_path / "baseline-payload"
    quality_project = tmp_path / "quality-project"
    baseline_payload.mkdir()
    quality_project.mkdir()
    preparation = _ProfilePreparation(
        baseline=_BaselineWheelPayload(
            wheel_path=tmp_path / "baseline.whl",
            build=_successful_attempt(),
            baseline_install_root=baseline_payload,
            quality_project_root=quality_project,
        )
    )
    search = SourceOptimizationSearchResult(
        attempted=True,
        accepted=True,
        wheel_path=wheel_path,
        patch_path=tmp_path / "accepted.patch",
        trials=(),
        test_results=(),
        performance=_benchmark_result("passed"),
        build=_successful_attempt(),
        materialization_patch=patch,
    )

    arm = cast(
        _OptimizationArmView,
        _materialize_source_optimization_arm(
            options=package_command.PackageOptions(root=project_root, output_dir=output_dir),
            project=project,
            preparation=preparation,
            search=search,
        ),
    )
    active_project = arm.active_project
    baseline = arm.baseline

    assert source_path.read_text(encoding="utf-8") == before_source
    assert (active_project.config.root / "src" / "app" / "ranking.py").read_text(
        encoding="utf-8"
    ) == after_source
    assert (output_dir / "install" / "app" / "ranking.py").read_text(
        encoding="utf-8"
    ) == "WHEEL_VALUE = 1\n"
    assert baseline.baseline_install_root is not None
    assert (baseline.baseline_install_root / "app" / "ranking.py").exists()
    assert arm.source_search is search


def test_composed_source_arm_keeps_successful_native_overlay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful later native stage replaces the source-only wheel arm."""
    project_root = tmp_path / "project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    output_dir.mkdir()
    project = discover_project(project_root)
    wheel_path = output_dir / "source.whl"
    wheel_path.write_bytes(b"source-wheel")
    baseline = _BaselineWheelPayload(
        wheel_path=wheel_path,
        build=_successful_attempt(),
        baseline_install_root=tmp_path / "payload",
        quality_project_root=tmp_path / "quality",
    )
    preparation = _ProfilePreparation(baseline=baseline)
    search = SourceOptimizationSearchResult(
        attempted=True,
        accepted=True,
        wheel_path=wheel_path,
        patch_path=tmp_path / "accepted.patch",
        trials=(),
        test_results=(),
        performance=_benchmark_result("passed"),
        build=_successful_attempt(),
        materialization_patch=GeneratedSourcePatch("", (), ()),
    )
    planning = SourceOptimizationPlanningResult(plans=(), assessments=())
    arm = _OptimizationArm(
        report_project=project,
        active_project=project,
        baseline=baseline,
        source_search=search,
    )

    def materialize(**_kwargs: object) -> object:
        return arm

    def selected_scans(*_args: object) -> tuple[ModuleScan, ...]:
        return ()

    def selected_regions(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        return (object(),)

    monkeypatch.setattr(package_command, "_materialize_source_optimization_arm", materialize)
    monkeypatch.setattr(package_command, "_selected_scans", selected_scans)
    monkeypatch.setattr(package_command, "_selected_typed_regions", selected_regions)

    def execute_native(**kwargs: object) -> package_command.PackageCommandResult:
        assert kwargs["prepared_baseline"] is baseline
        return package_command.PackageCommandResult(
            success=True,
            project_root=project_root,
            output_dir=output_dir,
            install_root=output_dir / "install",
            wheel_path=wheel_path,
            islands=(),
            build=_successful_attempt(),
        )

    monkeypatch.setattr(package_command, "_execute_typed_region_package", execute_native)

    result = _execute_composed_source_arm(
        options=package_command.PackageOptions(root=project_root, output_dir=output_dir),
        project=project,
        preparation=preparation,
        planning=planning,
        search=search,
    )

    assert result.success is True
    assert result.wheel_path == wheel_path
    assert result.source_optimization_trials == ()


def test_composed_source_arm_restores_source_wheel_after_native_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed later stage restores the previously accepted source wheel bytes."""
    project_root = tmp_path / "project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    output_dir.mkdir()
    project = discover_project(project_root)
    wheel_path = output_dir / "source.whl"
    wheel_path.write_bytes(b"accepted-source-wheel")
    baseline = _BaselineWheelPayload(
        wheel_path=wheel_path,
        build=_successful_attempt(),
        baseline_install_root=tmp_path / "payload",
        quality_project_root=tmp_path / "quality",
    )
    preparation = _ProfilePreparation(baseline=baseline)
    search = SourceOptimizationSearchResult(
        attempted=True,
        accepted=True,
        wheel_path=wheel_path,
        patch_path=tmp_path / "accepted.patch",
        trials=(),
        test_results=(),
        performance=_benchmark_result("passed"),
        build=_successful_attempt(),
        materialization_patch=GeneratedSourcePatch("", (), ()),
    )
    planning = SourceOptimizationPlanningResult(plans=(), assessments=())
    arm = _OptimizationArm(
        report_project=project,
        active_project=project,
        baseline=baseline,
        source_search=search,
    )

    def materialize(**_kwargs: object) -> object:
        return arm

    def selected_scans(*_args: object) -> tuple[ModuleScan, ...]:
        return ()

    def selected_regions(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        return (object(),)

    monkeypatch.setattr(package_command, "_materialize_source_optimization_arm", materialize)
    monkeypatch.setattr(package_command, "_selected_scans", selected_scans)
    monkeypatch.setattr(package_command, "_selected_typed_regions", selected_regions)

    def reject_native(**_kwargs: object) -> package_command.PackageCommandResult:
        wheel_path.write_bytes(b"rejected-native-wheel")
        return package_command.PackageCommandResult(
            success=False,
            project_root=project_root,
            output_dir=output_dir,
            install_root=output_dir / "install",
            wheel_path=None,
            islands=(),
            build=replace(_successful_attempt(), success=False, stderr="not profitable"),
            error="not profitable",
            applied_execution_plans=("rejected-plan",),
        )

    monkeypatch.setattr(package_command, "_execute_typed_region_package", reject_native)

    result = _execute_composed_source_arm(
        options=package_command.PackageOptions(root=project_root, output_dir=output_dir),
        project=project,
        preparation=preparation,
        planning=planning,
        search=search,
    )

    assert result.success is True
    assert result.wheel_path == wheel_path
    assert wheel_path.read_bytes() == b"accepted-source-wheel"
    assert "composition fallback retained: not profitable" in result.build.stdout
    assert result.compiled_regions == ()
    assert result.compiled_bindings == ()
    assert result.compiled_variants == ()
    assert result.artifact_records == ()
    assert result.applied_execution_plans == ()


def test_package_returns_source_application_failure_before_native_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit source application failure cannot fall through to native packaging."""
    project_root, output_dir, preparation = _prepared_source_search_project(tmp_path)

    def prepare(*_args: object, **_kwargs: object) -> object:
        return preparation

    def source_search(
        _plans: tuple[object, ...],
        _assessments: tuple[object, ...],
        _options: SourceOptimizationSearchOptions,
    ) -> SourceOptimizationSearchResult:
        return SourceOptimizationSearchResult(
            attempted=True,
            accepted=False,
            wheel_path=None,
            patch_path=project_root / ".atoll" / "patches" / "accepted.patch",
            trials=(),
            test_results=(),
            performance=_benchmark_result("passed"),
            build=_successful_attempt(),
            error="forced source application failure",
        )

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("native compilation ran after explicit source application failure")

    monkeypatch.setattr(package_command, "_prepare_profile_guided_selection", prepare)
    monkeypatch.setattr(package_command, "run_source_optimization_search", source_search)
    monkeypatch.setattr(package_command, "_build_typed_regions", forbidden)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            output_dir=output_dir,
            apply_source=True,
        )
    )

    assert result.success is False
    assert result.wheel_path is None
    assert result.error == "forced source application failure"
    assert result.build.success is False


def test_package_rejects_invalid_member_before_baseline_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Static request preflight prevents side effects for an invalid member."""
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    pyproject = project_root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8")
        + """

[tool.atoll.compile]
test_command = ["python", "-c", "pass"]
benchmark_command = ["python", "bench.py"]
""",
        encoding="utf-8",
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("configured target command ran before request validation")

    monkeypatch.setattr(package_command, "_prepare_baseline_wheel_payload", forbidden)
    monkeypatch.setattr(package_command, "run_performance_command", forbidden)
    monkeypatch.setattr(package_command, "run_baseline_profile", forbidden)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
            selected_members=(SymbolId("app.ranking", "missing_member"),),
        )
    )

    assert result.success is False
    assert result.error == (
        "requested member(s) are not backend-supported typed regions: app.ranking::missing_member"
    )
    assert result.profile is None
    assert result.test_results == ()
    assert not output_dir.exists()


def test_package_reuses_region_cache_for_unchanged_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unchanged second source-clean package build restores region artifacts."""
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    calls = 0

    def successful_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        nonlocal calls
        assert kwargs
        calls += 1
        if calls > 1:
            raise AssertionError("compile cache did not skip mypyc")
        path = cast(tuple[Path, ...], args[0])[0]
        artifact = tmp_path / f"{path.stem}{suffix}"
        artifact.write_text("binary", encoding="utf-8")
        return CompileAttempt(
            success=True,
            command=("mypyc",),
            stdout="",
            stderr="",
            artifact_paths=(artifact,),
            duration_seconds=0.1,
            phase_timings=(
                CompilePhaseTiming(name="mypycify", duration_seconds=0.08),
                CompilePhaseTiming(name="build_ext", duration_seconds=0.02),
            ),
        )

    monkeypatch.setattr(package_command, "build_sidecars", successful_build_sidecars, raising=False)

    first = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
        )
    )
    second = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
        )
    )

    assert first.success is True
    assert first.build.cache_status == "miss"
    assert second.success is True
    assert second.build.cache_status == "hit"
    assert calls == 0
    cache_timings = tuple(
        timing.name for timing in second.build.phase_timings if timing.name.startswith("cache_")
    )
    assert cache_timings == ("cache_lookup", "cache_restore")
    assert second.wheel_path is not None
    with zipfile.ZipFile(second.wheel_path) as wheel:
        assert any(name.startswith(".atoll/artifacts/") for name in wheel.namelist())


def test_package_caches_multiple_regions_independently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whole-project source-clean builds restore every unchanged region independently."""
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    ranking_source = project_root / "src" / "app" / "ranking.py"
    extra_source = project_root / "src" / "app" / "extra.py"
    extra_source.write_text(ranking_source.read_text(encoding="utf-8"), encoding="utf-8")
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    calls = 0

    def partial_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        nonlocal calls
        assert kwargs
        calls += 1
        paths = cast(tuple[Path, ...], args[0])
        if len(paths) > 1:
            return CompileAttempt(
                success=False,
                command=("mypyc",),
                stdout="",
                stderr="batch failed",
                artifact_paths=(),
                duration_seconds=0.1,
            )
        assert paths
        path = paths[0]
        if "extra" in path.stem:
            return CompileAttempt(
                success=False,
                command=("mypyc",),
                stdout="",
                stderr="extra failed",
                artifact_paths=(),
                duration_seconds=0.1,
            )
        artifact = tmp_path / f"{path.stem}{suffix}"
        artifact.write_text("binary", encoding="utf-8")
        return CompileAttempt(
            success=True,
            command=("mypyc",),
            stdout="",
            stderr="",
            artifact_paths=(artifact,),
            duration_seconds=0.1,
        )

    monkeypatch.setattr(package_command, "build_sidecars", partial_build_sidecars, raising=False)

    first = package_command.execute_package(
        package_command.PackageOptions(root=project_root, output_dir=output_dir)
    )
    second = package_command.execute_package(
        package_command.PackageOptions(root=project_root, output_dir=output_dir)
    )

    assert first.success is True
    assert first.build.cache_status == "miss"
    assert first.skipped == ()
    assert second.success is True
    assert second.build.cache_status == "hit"
    assert second.skipped == ()
    assert calls == 0
    cache_timings = tuple(
        timing.name for timing in second.build.phase_timings if timing.name.startswith("cache_")
    )
    assert cache_timings.count("cache_lookup") == EXPECTED_ATOMIC_SELECTION_COUNT
    assert cache_timings.count("cache_restore") == EXPECTED_ATOMIC_SELECTION_COUNT
    assert second.wheel_path is not None
    with zipfile.ZipFile(second.wheel_path) as wheel:
        names = set(wheel.namelist())
    assert any(name.startswith(".atoll/artifacts/") for name in names)
    assert "app/extra.py" in names
    with zipfile.ZipFile(second.wheel_path) as wheel:
        extra_text = wheel.read("app/extra.py").decode()
    assert "BEGIN ATOLL TYPED REGIONS: app.extra" in extra_text


def test_package_cache_invalidates_when_source_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Region cache keys change when retained function source changes."""
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    calls = 0

    def successful_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        nonlocal calls
        assert kwargs
        calls += 1
        path = cast(tuple[Path, ...], args[0])[0]
        artifact = tmp_path / f"{path.stem}-{calls}{suffix}"
        artifact.write_text("binary", encoding="utf-8")
        return CompileAttempt(
            success=True,
            command=("mypyc",),
            stdout="",
            stderr="",
            artifact_paths=(artifact,),
            duration_seconds=0.1,
        )

    monkeypatch.setattr(package_command, "build_sidecars", successful_build_sidecars, raising=False)

    first = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
        )
    )
    ranking_source = project_root / "src" / "app" / "ranking.py"
    ranking_source.write_text(
        ranking_source.read_text(encoding="utf-8").replace(
            "DEFAULT_WEIGHT = 1.5",
            "DEFAULT_WEIGHT = 1.75",
        ),
        encoding="utf-8",
    )
    second = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
        )
    )

    assert first.build.cache_status == "miss"
    assert second.build.cache_status == "miss"
    assert calls == 0


def test_package_cache_invalidates_when_generator_version_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Region cache keys include the typed-region generator version."""
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    calls = 0

    def successful_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        nonlocal calls
        assert kwargs
        calls += 1
        path = cast(tuple[Path, ...], args[0])[0]
        artifact = tmp_path / f"{path.stem}-{calls}{suffix}"
        artifact.write_text("binary", encoding="utf-8")
        return CompileAttempt(
            success=True,
            command=("mypyc",),
            stdout="",
            stderr="",
            artifact_paths=(artifact,),
            duration_seconds=0.1,
        )

    monkeypatch.setattr(package_command, "build_sidecars", successful_build_sidecars, raising=False)

    first = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
        )
    )
    monkeypatch.setattr(package_command, "TYPED_METHOD_GENERATOR_VERSION", "changed")
    second = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
        )
    )

    assert first.build.cache_status == "miss"
    assert second.build.cache_status == "miss"
    assert calls == 0


def test_package_whole_project_never_enters_legacy_retry_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Typed-region whole-project compilation never calls the sidecar retry loop."""
    project_root = tmp_path / "project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    package_dir = project_root / "src" / "app"
    ranking_source = package_dir / "ranking.py"
    (package_dir / "first.py").write_text(ranking_source.read_text(encoding="utf-8"))
    (package_dir / "second.py").write_text(ranking_source.read_text(encoding="utf-8"))
    ranking_source.unlink()

    def failing_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        assert kwargs
        assert args
        raise AssertionError("legacy sidecar retry loop was invoked")

    monkeypatch.setattr(package_command, "build_sidecars", failing_build_sidecars, raising=False)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            output_dir=output_dir,
            keep_install_tree=True,
        )
    )

    assert result.success is True
    assert result.error is None
    assert result.islands == ()
    assert result.skipped == ()
    assert {binding.source.module for binding in result.compiled_bindings} == {
        "app.first",
        "app.second",
    }


def test_package_compiles_typed_functions_despite_unrelated_typevar_syntax(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Typed functions are assessed independently from unrelated module TypeVars."""
    project_root = tmp_path / "project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    bad_module = project_root / "src" / "app" / "typing_features.py"
    bad_module.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "from typing_extensions import TypeVar",
                "",
                "T = TypeVar('T', infer_variance=True)",
                "",
                "def helper(value: int) -> int:",
                "    return value + 1",
                "",
                "def candidate(value: int) -> int:",
                "    adjusted = helper(value)",
                "    return adjusted",
                "",
            ]
        ),
        encoding="utf-8",
    )

    build_calls: list[tuple[Path, ...]] = []

    def failing_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        assert kwargs
        paths = cast(tuple[Path, ...], args[0])
        build_calls.append(paths)
        return CompileAttempt(
            success=False,
            command=("mypyc", *(str(path) for path in paths)),
            stdout="",
            stderr="MYPYC_TYPE_ERROR: generated sidecar failed",
            artifact_paths=(),
            duration_seconds=0.1,
        )

    monkeypatch.setattr(package_command, "build_sidecars", failing_build_sidecars, raising=False)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.typing_features",
            output_dir=output_dir,
        )
    )

    assert result.success is True
    assert build_calls == []
    assert result.error is None
    assert result.preflight_skipped == ()
    assert not (output_dir / "build").exists()
    assert result.cleanup_removed == (output_dir / "build", output_dir / "install")
    assert result.cleanup_kept == ()
    assert result.native_readiness == ()
    assert {binding.source.qualname for binding in result.compiled_bindings} == {
        "helper",
        "candidate",
    }


def test_package_whole_project_uses_region_assessments_in_typing_heavy_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whole-project mode compiles safe regions without module-level readiness gates."""
    project_root = tmp_path / "project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    package_dir = project_root / "src" / "app"
    clean_source = package_dir / "ranking.py"
    original_clean_source = clean_source.read_text(encoding="utf-8")
    (package_dir / "blocked.py").write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "from typing_extensions import TypeVar",
                "",
                "T = TypeVar('T', default=str)",
                "",
                "def helper(value: int) -> int:",
                "    return value + 1",
                "",
                "def candidate(value: int) -> int:",
                "    adjusted = helper(value)",
                "    return adjusted",
                "",
            ]
        ),
        encoding="utf-8",
    )
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]

    def successful_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        assert kwargs
        paths = cast(tuple[Path, ...], args[0])
        artifacts: list[Path] = []
        for path in paths:
            artifact = tmp_path / f"{path.stem}{suffix}"
            artifact.write_text("binary", encoding="utf-8")
            artifacts.append(artifact)
        return CompileAttempt(
            success=True,
            command=("mypyc",),
            stdout="",
            stderr="",
            artifact_paths=tuple(artifacts),
            duration_seconds=0.1,
        )

    monkeypatch.setattr(package_command, "build_sidecars", successful_build_sidecars, raising=False)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            output_dir=output_dir,
            keep_install_tree=True,
        )
    )

    assert result.success is True
    assert result.install_tree_kept is True
    assert result.cleanup_removed == (output_dir / "build",)
    assert result.cleanup_kept == (output_dir / "install",)
    assert result.islands == ()
    assert {binding.source.module for binding in result.compiled_bindings} == {
        "app.ranking",
        "app.blocked",
    }
    assert result.preflight_skipped == ()
    assert "# BEGIN ATOLL TYPED REGIONS: app.ranking" in (
        output_dir / "install" / "app" / "ranking.py"
    ).read_text(encoding="utf-8")
    assert "# BEGIN ATOLL TYPED REGIONS: app.blocked" in (
        output_dir / "install" / "app" / "blocked.py"
    ).read_text(encoding="utf-8")
    assert result.native_readiness == ()
    assert clean_source.read_text(encoding="utf-8") == original_clean_source


def test_package_helpers_handle_flat_source_roots(tmp_path: Path) -> None:
    """Flat source roots copy their contents into the build root."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "current.py").symlink_to("pkg/mod.py")
    project = discover_project(tmp_path)
    build_root = tmp_path / "build"
    build_root.mkdir()

    staged_roots = _copy_source_roots(project, build_root)

    assert staged_roots == (build_root,)
    assert (build_root / "pkg" / "mod.py").exists()
    assert (build_root / "current.py").is_symlink()
    assert (build_root / "current.py").readlink() == Path("pkg/mod.py")


def test_source_roots_digest_invalidates_on_imported_source_change(tmp_path: Path) -> None:
    """The backend cache identity covers files outside a generated native unit."""
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    source = first_root / "pkg" / "dependency.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n", encoding="utf-8")
    (second_root / "config.ini").write_text("strict = true\n", encoding="utf-8")

    original = _source_roots_digest((first_root, second_root))
    source.write_text("VALUE = 2\n", encoding="utf-8")
    changed_source = _source_roots_digest((first_root, second_root))
    reversed_roots = _source_roots_digest((second_root, first_root))

    assert original != changed_source
    assert changed_source != reversed_roots
    assert changed_source == _source_roots_digest((first_root, second_root))


def test_staged_source_digest_identifies_the_copied_snapshot(tmp_path: Path) -> None:
    """A post-copy checkout edit cannot relabel staged bytes in the backend cache."""
    project_root = tmp_path / "project"
    source = project_root / "pkg" / "dependency.py"
    source.parent.mkdir(parents=True)
    (project_root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    source.write_text("VALUE = 1\n", encoding="utf-8")
    project = discover_project(project_root)

    staged_roots, staged_digest = _stage_target_sources(
        project,
        tmp_path / "build",
        None,
    )
    source.write_text("VALUE = 2\n", encoding="utf-8")

    assert staged_digest == _source_roots_digest(staged_roots)
    assert staged_digest != _source_roots_digest(project.config.source_roots)


def test_pep517_project_copy_preserves_reproducible_root_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The generated Git pointer cannot give identical copies different keys."""
    source = tmp_path / "project"
    package = source / "src" / "demo"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    git_dir = source / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    fixed_timestamp = 1_700_000_000_000_000_000
    os.utime(source, ns=(fixed_timestamp, fixed_timestamp))
    monkeypatch.setenv(BASELINE_WHEEL_CACHE_CONTEXT_ENV, "pep517-copy-test")
    first = tmp_path / "first"
    second = tmp_path / "second"

    _copy_pep517_project(
        source,
        first,
        excluded_output=source / ".atoll" / "dist",
    )
    _copy_pep517_project(
        source,
        second,
        excluded_output=source / ".atoll" / "dist",
    )

    assert first.stat().st_mtime_ns == fixed_timestamp
    assert second.stat().st_mtime_ns == fixed_timestamp
    assert baseline_wheel_cache_key(first) == baseline_wheel_cache_key(second)


def test_pep517_project_copy_resolves_worktree_git_pointer(tmp_path: Path) -> None:
    """A linked-worktree checkout exposes its real Git directory to the build copy."""
    source = tmp_path / "project"
    package = source / "src" / "demo"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    git_dir = tmp_path / "worktree-git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (source / ".git").write_text("gitdir: ../worktree-git\n", encoding="utf-8")
    destination = tmp_path / "copy"

    _copy_pep517_project(
        source,
        destination,
        excluded_output=source / ".atoll" / "dist",
    )

    assert (destination / ".git").read_text(encoding="utf-8") == f"gitdir: {git_dir.resolve()}\n"


def test_pep517_project_copy_ignores_invalid_git_pointer(tmp_path: Path) -> None:
    """Malformed copied-worktree metadata is not propagated into a build tree."""
    source = tmp_path / "project"
    source.mkdir()
    (source / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
    (source / ".git").write_text("not a git pointer\n", encoding="utf-8")
    destination = tmp_path / "copy"

    _copy_pep517_project(
        source,
        destination,
        excluded_output=source / ".atoll" / "dist",
    )

    assert not (destination / ".git").exists()


def test_atoll_artifact_helpers_copy_artifacts_and_skip_same_file(tmp_path: Path) -> None:
    """Source-clean artifact copies tolerate missing roots and an identical target."""
    source_root = tmp_path / "source"
    artifact_dir = source_root / ".atoll" / "artifacts"
    install_root = tmp_path / "install"
    artifact_dir.mkdir(parents=True)
    native = artifact_dir / f"_sidecar{importlib.machinery.EXTENSION_SUFFIXES[0]}"
    native.write_text("binary", encoding="utf-8")

    _copy_atoll_artifacts((tmp_path / "missing", source_root), install_root)
    copied = install_root / ".atoll" / "artifacts" / native.name
    _copy_if_different(copied, copied)

    assert copied.read_text(encoding="utf-8") == "binary"


def test_project_metadata_falls_back_for_missing_or_dynamic_version(tmp_path: Path) -> None:
    """Project metadata falls back to stable Atoll values when version is dynamic."""
    fallback = _project_metadata(tmp_path)
    assert fallback.name == tmp_path.name
    assert fallback.version == "0+atoll"

    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "dynamic-project"',
                'dynamic = ["version"]',
                'requires-python = ">=3.12"',
                'dependencies = ["pydantic>=2"]',
            ]
        ),
        encoding="utf-8",
    )

    metadata = _project_metadata(project_root)
    assert metadata.name == "dynamic-project"
    assert metadata.version == "0+atoll"
    assert metadata.requires_python == ">=3.12"
    assert metadata.dependencies == ("pydantic>=2",)


def test_package_small_helpers_cover_fallbacks(tmp_path: Path) -> None:
    """Small helper fallbacks stay deterministic."""
    path = tmp_path / "existing"
    path.mkdir()
    (path / "old.txt").write_text("old", encoding="utf-8")
    _reset_dir(path)
    assert path.exists()
    assert not (path / "old.txt").exists()

    assert _relative_source_root(tmp_path, tmp_path / "src") == Path("src")
    outside_root = tmp_path.parent / "not-under-root"
    assert _relative_source_root(tmp_path, outside_root) != outside_root
    assert _mapping(None) == {}
    assert _mapping({1: "value"}) == {"1": "value"}
    assert _sequence(None) == ()
    assert _sequence([1, "two"]) == (1, "two")
    assert _string(1) is None
    assert _string("value") == "value"
    assert _resolve_output_dir(tmp_path, None) == tmp_path / ".atoll" / "dist"
    assert (
        _resolve_output_dir(tmp_path, Path("custom-dist")) == (tmp_path / "custom-dist").resolve()
    )


def test_package_helpers_report_missing_modules(tmp_path: Path) -> None:
    """Module lookup helpers fail clearly for impossible paths."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    project = DiscoveredProject(
        config=discover_project(project_root).config,
        modules=(),
    )

    with pytest.raises(ValueError, match="module not found"):
        _find_module((), "missing")
    with pytest.raises(ValueError, match="outside configured source roots"):
        _staged_module(
            ModuleId(name="missing", path=tmp_path / "outside.py"),
            project,
            (tmp_path / "stage",),
        )


def _quality_gate_project(
    tmp_path: Path,
    compile_lines: tuple[str, ...],
) -> DiscoveredProject:
    project_root = tmp_path / "quality-gate-project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    pyproject = project_root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8")
        + "\n".join(("", "[tool.atoll.compile]", *compile_lines, "")),
        encoding="utf-8",
    )
    return discover_project(project_root)


def _prepared_source_search_project(tmp_path: Path) -> tuple[Path, Path, object]:
    project_root = tmp_path / "source-search-project"
    output_dir = tmp_path / "source-search-output"
    shutil.copytree(FIXTURE_ROOT, project_root)
    pyproject = project_root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8")
        + """

[tool.atoll.compile]
test_command = ["python", "-c", "pass"]
benchmark_command = ["python", "bench.py"]
""",
        encoding="utf-8",
    )
    baseline_payload = tmp_path / "baseline-payload"
    quality_project = tmp_path / "quality-project"
    baseline_payload.mkdir()
    quality_project.mkdir()
    baseline_wheel = tmp_path / "baseline-py3-none-any.whl"
    baseline_wheel.write_bytes(b"wheel")
    preparation = _ProfilePreparation(
        baseline=_BaselineWheelPayload(
            wheel_path=baseline_wheel,
            build=_successful_attempt(),
            baseline_install_root=baseline_payload,
            quality_project_root=quality_project,
        )
    )
    return project_root, output_dir, preparation


def _successful_attempt() -> CompileAttempt:
    return CompileAttempt(
        success=True,
        command=("fixture",),
        stdout="",
        stderr="",
        artifact_paths=(),
        duration_seconds=0.0,
    )


def _compiled_region_marker_count(payload_root: Path) -> int:
    """Count generated native binding declarations in a candidate payload.

    Args:
        payload_root: Exact unpacked wheel payload measured by a candidate trial.

    Returns:
        int: Number of compiled-module declarations in the fixture source shim.
    """
    source_path = payload_root / "app" / "ranking.py"
    if not source_path.is_file():
        return 0
    return source_path.read_text(encoding="utf-8").count("'compiled_module':")


def _benchmark_result(status: BenchmarkStatus) -> BenchmarkGateResult:
    return BenchmarkGateResult(
        status=status,
        reason=f"fixture {status}",
        minimum_speedup=1.1,
        baseline_median_seconds=1.0,
        compiled_median_seconds=1.0,
        speedup=1.0,
        warmups=(),
        samples=(),
    )


def _eligible_fusion_plan(
    *,
    caller: str = "sample::root",
    callee: str = "sample::worker",
) -> FusionPlan:
    return FusionPlan(
        id="task-fusion:fixture",
        source_hash="source-hash",
        root=caller,
        caller=caller,
        callee=callee,
        spawn_api="asyncio.create_task",
        lineno=1,
        end_lineno=1,
        col_offset=0,
        end_col_offset=1,
        eligible=True,
        observed_calls=25,
        completed_calls=25,
        max_active_calls=1,
        pre_completion_suspensions=0,
        observed_signatures=1,
        observation_capped=False,
        rejections=(),
        spawn_source="asyncio.create_task(worker())",
    )
