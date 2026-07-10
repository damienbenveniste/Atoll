"""Report schema conversion and Markdown rendering for Atoll commands.

This module is the boundary between internal dataclasses and user-visible JSON
or Markdown artifacts. TypedDict classes define stable report shapes, while the
rendering functions keep paths relative where possible and avoid importing or
executing target-project code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

from atoll.analysis.native_readiness import NativeReadiness
from atoll.models import (
    ArtifactRecord,
    Backend,
    BackendAssessment,
    BindingKind,
    BindingTarget,
    Blocker,
    BlockerSeverity,
    CompileAttempt,
    CompileCacheStatus,
    CompiledRegionVariant,
    Confidence,
    ConstantKind,
    DependencyKind,
    DependencyRole,
    DiagnosticSeverity,
    EnabledIslandConfig,
    ExecutionKind,
    IslandRisk,
    LossAction,
    ModuleScan,
    MypyDiagnostic,
    ParameterKind,
    PytestRunResult,
    RuntimeTypeGuard,
    ScanResult,
    SpecializationOrigin,
    SymbolId,
    SymbolKind,
    TypedRegion,
    TypeParameterKind,
    TypeParameterRecord,
    VerifyResult,
    Visibility,
)
from atoll.runtime.package_verify import PackageVerificationResult
from atoll.runtime.performance import BenchmarkGateResult, CommandRunEvidence

_STRONG_SCORE = 90
_GOOD_SCORE = 80
_POSSIBLE_SCORE = 70
_ATOLL_PART_INDEX = 0
_ATOLL_GENERATED_INPUT_DIR_INDEX = 1


class BlockerReport(TypedDict):
    """Serialized blocker shown in scan and symbol reports."""

    severity: BlockerSeverity
    code: str
    message: str
    lineno: int | None
    symbol: str | None


class ImportReport(TypedDict):
    """Serialized top-level import with original source text preserved."""

    source_text: str
    imported_names: list[str]
    module: str | None
    level: int
    lineno: int
    end_lineno: int


class ConstantReport(TypedDict):
    """Serialized top-level assignment and its extraction safety classification."""

    name: str
    kind: ConstantKind
    source_text: str
    lineno: int
    end_lineno: int


class SymbolReport(TypedDict):
    """Serialized AST, blocker, and type-checker facts for one symbol."""

    id: str
    qualname: str
    kind: SymbolKind
    visibility: Visibility
    lineno: int
    end_lineno: int
    decorators: list[str]
    arg_count: int
    annotated_arg_count: int
    has_return_annotation: bool
    has_any_annotation: bool
    called_names: list[str]
    uses_globals: list[str]
    local_names: list[str]
    referenced_names: list[str]
    owner_class: str | None
    binding_kind: BindingKind
    execution_kind: ExecutionKind
    type_parameters: list[str]
    parameters: list[ParameterReport]
    return_annotation: str | None
    annotation_names: list[str]
    called_paths: list[str]
    base_names: list[str]
    fields: list[FieldReport]
    declaration_start_lineno: int | None
    scope_type_parameters: list[str]
    type_parameter_records: list[TypeParameterReport]
    scope_type_parameter_records: list[TypeParameterReport]
    any_annotation_sources: list[str]
    blockers: list[BlockerReport]
    mypy_diagnostics: list[MypyDiagnosticReport]


class ParameterReport(TypedDict):
    """Exact source parameter evidence retained by typed-region analysis."""

    name: str
    kind: ParameterKind
    annotation: str | None
    default_source: str | None


class FieldReport(TypedDict):
    """Typed class field evidence retained for class-region planning."""

    name: str
    annotation: str
    default_source: str | None
    class_variable: bool


class TypeParameterReport(TypedDict):
    """Exact type-parameter declaration retained for backend assessment."""

    name: str
    kind: TypeParameterKind
    declaration: str


class MypyDiagnosticReport(TypedDict):
    """Serialized mypy diagnostic after optional symbol range mapping."""

    path: str
    line: int
    column: int | None
    severity: DiagnosticSeverity
    code: str | None
    message: str
    symbol: str | None


class DependencyEdgeReport(TypedDict):
    """Serialized same-module dependency or external boundary edge evidence."""

    src: str
    dst: str
    kind: DependencyKind
    confidence: Confidence
    lineno: int | None


class IslandCandidateReport(TypedDict):
    """Serialized island recommendation with score, risk, and dependency context."""

    symbols: list[str]
    required_imports: list[str]
    required_constants: list[str]
    required_local_symbols: list[str]
    rejected_symbols: list[str]
    score: int
    score_label: str
    score_summary: str
    risk: IslandRisk
    risk_summary: str
    reasons: list[str]


class PoisonRadiusReport(TypedDict):
    """Serialized explanation of a rejected symbol's impact on candidates."""

    poison: str
    impacted: list[str]
    reason: str


class RegionMemberReport(TypedDict):
    """One unlowered declaration included in a typed region."""

    id: str
    kind: SymbolKind
    owner_class: str | None
    binding_kind: BindingKind
    execution_kind: ExecutionKind
    type_parameters: list[str]
    scope_type_parameters: list[str]
    type_parameter_records: list[TypeParameterReport]
    scope_type_parameter_records: list[TypeParameterReport]
    parameters: list[ParameterReport]
    return_annotation: str | None
    fields: list[FieldReport]


class RegionDependencyReport(TypedDict):
    """Dependency retained with runtime versus type-only intent."""

    src: str
    dst: str
    kind: DependencyKind
    confidence: Confidence
    role: DependencyRole
    type_only: bool


class TypeBindingReport(TypedDict):
    """Source type evidence retained before backend lowering."""

    name: str
    annotation: str
    source: str
    concrete: bool
    substitutions: list[list[str]]


class LoweringDecisionReport(TypedDict):
    """Auditable region-level preservation or fallback decision."""

    target: str
    action: LossAction
    reason: str


class BindingTargetReport(TypedDict):
    """Descriptor-aware source binding promised by a typed region."""

    source: str
    compiled_name: str
    kind: BindingKind
    owner_class: str | None
    target_owner_class: str | None
    execution_kind: ExecutionKind
    required: bool
    guards: list[RuntimeTypeGuardReport]


class RuntimeTypeGuardReport(TypedDict):
    """Constant-time input check required before specialized native routing."""

    parameter_name: str
    positional_index: int | None
    annotation: str
    nominal_type_paths: list[str]
    allow_none: bool


class RegionSpecializationReport(TypedDict):
    """Concrete TypeVar binding layered on an unchanged generic declaration."""

    id: str
    source_member: str
    source_owner_class: str | None
    target_owner_class: str | None
    origin: SpecializationOrigin
    substitutions: list[list[str]]
    guards: list[RuntimeTypeGuardReport]
    type_bindings: list[TypeBindingReport]


class TypedRegionReport(TypedDict):
    """Backend-neutral typed region serialized in scan reports."""

    id: str
    source_hash: str
    atomic_class: bool
    members: list[RegionMemberReport]
    dependencies: list[RegionDependencyReport]
    type_bindings: list[TypeBindingReport]
    bindings: list[BindingTargetReport]
    decisions: list[LoweringDecisionReport]
    specializations: list[RegionSpecializationReport]


class ModuleReport(TypedDict):
    """Complete scan report section for one discovered Python module."""

    module: str
    path: str
    imports: list[ImportReport]
    constants: list[ConstantReport]
    symbols: list[SymbolReport]
    blockers: list[BlockerReport]
    top_level_statement_lines: list[int]
    mypy_diagnostics: list[MypyDiagnosticReport]
    dependency_edges: list[DependencyEdgeReport]
    island_candidates: list[IslandCandidateReport]
    poison_radii: list[PoisonRadiusReport]
    typed_regions: list[TypedRegionReport]


class SummaryReport(TypedDict):
    """Aggregate scan counts used by JSON and Markdown summaries."""

    modules_scanned: int
    symbols_scanned: int
    island_candidates: int
    typed_regions: int
    hard_blockers: int
    soft_blockers: int


class ScanReport(TypedDict):
    """Top-level stable JSON report emitted by `atoll scan`."""

    version: int
    tool: str
    project_root: str
    source_roots: list[str]
    summary: SummaryReport
    modules: list[ModuleReport]


CompilationOperation = Literal["build", "compile"]
CompilationMode = Literal["in-place", "source-clean"]


class CompilationNativeReadinessReport(TypedDict):
    """Post-generation evidence that a selected symbol can benefit from mypyc."""

    source_module: str
    symbol: str
    eligible: bool
    score: int
    function_count: int
    any_typed_functions: list[str]
    boxed_typed_functions: list[str]
    dynamic_dependencies: list[str]
    loop_count: int
    native_operation_count: int
    reasons: list[str]


class CompilationSummaryReport(TypedDict):
    """Aggregate build, verification, test, and cleanup counts for compilation."""

    islands: int
    typed_regions: int
    compiled_regions: int
    symbols: int
    native_ready_symbols: int
    native_rejected_symbols: int
    artifacts: int
    support_artifacts: int
    skipped_modules: int
    preflight_blockers: int
    verified: int
    verify_failures: int
    semantic_tests_run: bool
    semantic_test_failures: int
    subprocess_verifications: int
    subprocess_verification_failures: int
    performance_status: str
    duration_seconds: float


class CompilationBuildReport(TypedDict):
    """Serialized mypyc build command, diagnostics, and produced artifacts."""

    success: bool
    command: list[str]
    duration_seconds: float
    stdout: str
    stderr: str
    cache_status: CompileCacheStatus
    phase_timings: list[CompilationPhaseTimingReport]
    artifacts: list[str]
    support_artifacts: list[str]


class CompilationPhaseTimingReport(TypedDict):
    name: str
    duration_seconds: float
    detail: str | None


class CompilationCleanupReport(TypedDict):
    """Paths removed or intentionally kept after a build or package operation."""

    removed: list[str]
    kept: list[str]


class CompilationSkippedModuleReport(TypedDict):
    """Source-clean module skipped because its island failed compilation."""

    module: str
    reason: str


class CompilationPreflightBlockerReport(TypedDict):
    """Module-level blocker that prevented a source-clean module build attempt."""

    module: str
    path: str
    line: int | None
    code: str
    message: str


class CompilationTestReport(TypedDict):
    """Target-project semantic test command and process exit status."""

    command: list[str]
    exit_code: int
    success: bool


class CompilationVerifySymbolReport(TypedDict):
    """Runtime verification result for one exported symbol rebound by a shim."""

    symbol: str
    rebound: bool


class CompilationVerifyReport(TypedDict):
    """Runtime routing state for one compiled or pure-Python sidecar."""

    active: bool
    compiled: bool
    origin: str | None
    symbols: list[CompilationVerifySymbolReport]
    error: str | None


class CompilationIslandReport(TypedDict):
    """Compilation report section for one enabled source island."""

    source_module: str
    source_path: str
    generated_module: str
    symbols: list[str]
    artifacts: list[str]
    verification: CompilationVerifyReport | None


class CompilationCompiledBindingReport(TypedDict):
    """One guarded function or descriptor promised by a compiled region."""

    source: str
    compiled_name: str
    kind: BindingKind
    owner_class: str | None
    target_owner_class: str | None
    execution_kind: ExecutionKind
    required: bool
    guards: list[RuntimeTypeGuardReport]


class CompilationCompiledRegionReport(TypedDict):
    """Backend, binding, and artifact evidence for one successful region."""

    id: str
    variant_id: str
    source_module: str
    backend: Backend | None
    cache_status: CompileCacheStatus
    bindings: list[CompilationCompiledBindingReport]
    artifacts: list[str]


class CompilationVerificationStepReport(TypedDict):
    """Fresh-interpreter verification evidence for a payload or final wheel."""

    stage: str
    target: str
    command: list[str]
    success: bool
    exit_code: int
    duration_seconds: float
    stdout: str
    stderr: str


class CompilationCommandRunReport(TypedDict):
    """One baseline or compiled semantic-test or benchmark subprocess run."""

    command: list[str]
    mode: str
    payload_root: str
    returncode: int
    success: bool
    duration_seconds: float
    stdout: str
    stderr: str


class CompilationPerformanceReport(TypedDict):
    """Measured profitability evidence or an explicit unbenchmarked status."""

    status: str
    reason: str
    minimum_speedup: float
    baseline_median_seconds: float | None
    compiled_median_seconds: float | None
    speedup: float | None
    warmups: list[CompilationCommandRunReport]
    samples: list[CompilationCommandRunReport]


class CompilationReport(TypedDict):
    """Top-level stable JSON report for build and source-clean compile commands."""

    version: int
    tool: str
    operation: CompilationOperation
    mode: CompilationMode
    project_root: str
    module_filter: str | None
    success: bool
    wheel_path: str | None
    summary: CompilationSummaryReport
    build: CompilationBuildReport
    tests: CompilationTestReport | None
    test_results: list[CompilationCommandRunReport]
    verification_steps: list[CompilationVerificationStepReport]
    performance: CompilationPerformanceReport
    cleanup: CompilationCleanupReport
    skipped_modules: list[CompilationSkippedModuleReport]
    preflight_blockers: list[CompilationPreflightBlockerReport]
    native_readiness: list[CompilationNativeReadinessReport]
    typed_regions: list[TypedRegionReport]
    compiled_regions: list[CompilationCompiledRegionReport]
    islands: list[CompilationIslandReport]


@dataclass(frozen=True, slots=True)
class CompilationSkippedModuleInput:
    """Internal input for a selected module skipped after mypyc rejection.

    This keeps source-clean packaging failures separate from preflight blockers:
    the module reached compilation, but no usable artifact was produced.
    """

    module: str
    reason: str


@dataclass(frozen=True, slots=True)
class CompilationPreflightBlockerInput:
    """Internal input for a known module-level mypyc blocker.

    Preflight blockers are emitted before running mypyc so the report can explain
    why a module was not attempted at all.
    """

    module: str
    path: Path
    line: int | None
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class _CompiledVariantReportInput:
    variant_id: str
    region: TypedRegion
    backend: Backend | None
    cache_status: CompileCacheStatus


@dataclass(frozen=True, slots=True)
class CompilationReportInput:
    """All command evidence needed to render one compilation report.

    The renderer derives success from build, verification, and optional semantic
    test evidence instead of trusting a caller-supplied status. Paths are kept as
    `Path` objects until rendering so they can be normalized relative to `root`.
    """

    root: Path
    operation: CompilationOperation
    module_filter: str | None
    islands: tuple[EnabledIslandConfig, ...]
    build: CompileAttempt
    mode: CompilationMode = "in-place"
    wheel_path: Path | None = None
    verification: tuple[VerifyResult, ...] = ()
    tests: PytestRunResult | None = None
    cleanup_removed: tuple[Path, ...] = ()
    cleanup_kept: tuple[Path, ...] = ()
    skipped_modules: tuple[CompilationSkippedModuleInput, ...] = ()
    preflight_blockers: tuple[CompilationPreflightBlockerInput, ...] = ()
    native_readiness: tuple[NativeReadiness, ...] = ()
    typed_regions: tuple[TypedRegion, ...] = ()
    compiled_regions: tuple[TypedRegion, ...] = ()
    compiled_bindings: tuple[BindingTarget, ...] = ()
    compiled_variants: tuple[CompiledRegionVariant, ...] = ()
    backend_assessments: tuple[BackendAssessment, ...] = ()
    artifact_records: tuple[ArtifactRecord, ...] = ()
    verification_steps: tuple[PackageVerificationResult, ...] = ()
    test_results: tuple[CommandRunEvidence, ...] = ()
    performance: BenchmarkGateResult | None = None


def build_scan_report(result: ScanResult) -> ScanReport:
    """Convert enriched scan dataclasses into the stable scan JSON shape."""
    module_reports = [_module_report(module) for module in result.modules]
    all_blockers = [
        blocker
        for module in result.modules
        for blocker in (
            *module.blockers,
            *(blocker for symbol in module.symbols for blocker in symbol.blockers),
        )
    ]
    hard_count = sum(blocker.severity == "hard" for blocker in all_blockers)
    soft_count = sum(blocker.severity == "soft" for blocker in all_blockers)
    return {
        "version": 2,
        "tool": "atoll",
        "project_root": str(result.config.root),
        "source_roots": [str(path) for path in result.config.source_roots],
        "summary": {
            "modules_scanned": len(result.modules),
            "symbols_scanned": sum(len(module.symbols) for module in result.modules),
            "island_candidates": sum(len(module.island_candidates) for module in result.modules),
            "typed_regions": sum(len(module.typed_regions) for module in result.modules),
            "hard_blockers": hard_count,
            "soft_blockers": soft_count,
        },
        "modules": module_reports,
    }


def write_json_report(path: Path, report: ScanReport) -> None:
    """Write a scan report as sorted, formatted JSON.

    Parent directories are created automatically. The function performs no schema
    validation beyond the `ScanReport` type shape used by callers.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8")


def write_markdown_report(path: Path, report: ScanReport) -> None:
    """Write the human-readable scan report next to JSON artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown_report(report), encoding="utf-8")


def render_markdown_report(report: ScanReport) -> str:
    """Render a concise Markdown scan report for users reviewing candidates."""
    lines = [
        "# Atoll Scan Report",
        "",
        "## Summary",
        "",
        f"- Modules scanned: {report['summary']['modules_scanned']}",
        f"- Symbols scanned: {report['summary']['symbols_scanned']}",
        f"- Island candidates: {report['summary']['island_candidates']}",
        f"- Typed regions: {report['summary']['typed_regions']}",
        f"- Hard blockers: {report['summary']['hard_blockers']}",
        f"- Soft blockers: {report['summary']['soft_blockers']}",
        "",
        "## How To Read Candidates",
        "",
        "- Score is a 0-100 heuristic for how promising the island looks before compilation.",
        "- Risk is extraction risk: `low` means Atoll saw only high-confidence dependencies.",
        (
            "- Candidates are predictions; `atoll build` and `atoll verify` prove "
            "compiled routing, while target tests prove exercised semantics."
        ),
        "",
    ]
    for module in report["modules"]:
        lines.extend(_markdown_module(module))
    return "\n".join(lines).rstrip() + "\n"


def build_compilation_report(report_input: CompilationReportInput) -> CompilationReport:
    """Convert build, verification, and cleanup evidence into a stable report."""
    compiled_regions = _compiled_region_reports(report_input)
    verify_by_module = {result.source_module: result for result in report_input.verification}
    artifact_paths = tuple(report_input.build.artifact_paths)
    island_artifacts = {
        island.source_module: _island_artifacts(island, artifact_paths)
        for island in report_input.islands
    }
    mapped_artifacts = {
        artifact for artifacts in island_artifacts.values() for artifact in artifacts
    }
    compiled_artifact_names = {
        Path(record.install_relative_path).name
        for record in report_input.artifact_records
        if record.region_id != "__shared__"
    }
    mapped_artifacts.update(
        artifact for artifact in artifact_paths if artifact.name in compiled_artifact_names
    )
    support_artifacts = tuple(path for path in artifact_paths if path not in mapped_artifacts)
    verify_failures = sum(result.error is not None for result in report_input.verification)
    subprocess_verify_failures = sum(
        not result.success for result in report_input.verification_steps
    )
    test_failures = int(report_input.tests is not None and not report_input.tests.success) + sum(
        not result.succeeded for result in report_input.test_results
    )
    performance = _compilation_performance_report(report_input.root, report_input.performance)
    performance_failed = performance["status"] not in {"passed", "unbenchmarked"}
    wheel_missing = report_input.mode == "source-clean" and report_input.wheel_path is None
    success = (
        report_input.build.success
        and verify_failures == 0
        and subprocess_verify_failures == 0
        and test_failures == 0
        and not performance_failed
        and not wheel_missing
    )
    return {
        "version": 2,
        "tool": "atoll",
        "operation": report_input.operation,
        "mode": report_input.mode,
        "project_root": str(report_input.root.resolve()),
        "module_filter": report_input.module_filter,
        "success": success,
        "wheel_path": (
            _path_text(report_input.root, report_input.wheel_path)
            if report_input.wheel_path is not None
            else None
        ),
        "summary": {
            "islands": len(report_input.islands),
            "typed_regions": len(report_input.typed_regions),
            "compiled_regions": len(compiled_regions),
            "symbols": sum(len(island.symbols) for island in report_input.islands)
            + len(report_input.compiled_bindings),
            "native_ready_symbols": sum(
                readiness.eligible for readiness in report_input.native_readiness
            ),
            "native_rejected_symbols": sum(
                not readiness.eligible for readiness in report_input.native_readiness
            ),
            "artifacts": len(artifact_paths),
            "support_artifacts": len(support_artifacts),
            "skipped_modules": len(report_input.skipped_modules),
            "preflight_blockers": len(report_input.preflight_blockers),
            "verified": len(report_input.verification),
            "verify_failures": verify_failures,
            "semantic_tests_run": bool(report_input.tests or report_input.test_results),
            "semantic_test_failures": test_failures,
            "subprocess_verifications": len(report_input.verification_steps),
            "subprocess_verification_failures": subprocess_verify_failures,
            "performance_status": performance["status"],
            "duration_seconds": report_input.build.duration_seconds,
        },
        "build": {
            "success": report_input.build.success,
            "command": _build_command_report(report_input.root, report_input.build.command),
            "duration_seconds": report_input.build.duration_seconds,
            "stdout": report_input.build.stdout,
            "stderr": report_input.build.stderr,
            "cache_status": report_input.build.cache_status,
            "phase_timings": [
                {
                    "name": timing.name,
                    "duration_seconds": timing.duration_seconds,
                    "detail": timing.detail,
                }
                for timing in report_input.build.phase_timings
            ],
            "artifacts": [_path_text(report_input.root, path) for path in artifact_paths],
            "support_artifacts": [
                _path_text(report_input.root, path) for path in support_artifacts
            ],
        },
        "tests": _compilation_test_report(report_input.tests),
        "test_results": [
            _compilation_command_run_report(report_input.root, result)
            for result in report_input.test_results
        ],
        "verification_steps": [
            _compilation_verification_step_report(report_input.root, result)
            for result in report_input.verification_steps
        ],
        "performance": performance,
        "cleanup": {
            "removed": [
                *_generated_input_cleanup_reports(report_input.root, report_input.cleanup_removed),
                *[
                    _path_text(report_input.root, path)
                    for path in report_input.cleanup_removed
                    if not _is_generated_input_path(report_input.root, path)
                ],
            ],
            "kept": [_path_text(report_input.root, path) for path in report_input.cleanup_kept],
        },
        "skipped_modules": [
            {"module": skipped.module, "reason": skipped.reason}
            for skipped in report_input.skipped_modules
        ],
        "preflight_blockers": [
            {
                "module": blocker.module,
                "path": _path_text(report_input.root, blocker.path),
                "line": blocker.line,
                "code": blocker.code,
                "message": blocker.message,
            }
            for blocker in report_input.preflight_blockers
        ],
        "native_readiness": [
            {
                "source_module": readiness.source_module,
                "symbol": readiness.symbol,
                "eligible": readiness.eligible,
                "score": readiness.score,
                "function_count": readiness.function_count,
                "any_typed_functions": list(readiness.any_typed_functions),
                "boxed_typed_functions": list(readiness.boxed_typed_functions),
                "dynamic_dependencies": list(readiness.dynamic_dependencies),
                "loop_count": readiness.loop_count,
                "native_operation_count": readiness.native_operation_count,
                "reasons": list(readiness.reasons),
            }
            for readiness in report_input.native_readiness
        ],
        "typed_regions": [_typed_region_report(region) for region in report_input.typed_regions],
        "compiled_regions": compiled_regions,
        "islands": [
            {
                "source_module": island.source_module,
                "source_path": _path_text(report_input.root, island.source_path),
                "generated_module": island.sidecar_module,
                "symbols": list(island.symbols),
                "artifacts": [
                    _path_text(report_input.root, path)
                    for path in island_artifacts[island.source_module]
                ],
                "verification": _compilation_verify_report(
                    verify_by_module.get(island.source_module)
                ),
            }
            for island in report_input.islands
        ],
    }


def _compiled_region_reports(
    report_input: CompilationReportInput,
) -> list[CompilationCompiledRegionReport]:
    if report_input.compiled_variants:
        return [
            _compiled_region_variant_report(
                identity=_CompiledVariantReportInput(
                    variant_id=variant.id,
                    region=variant.region,
                    backend=variant.backend,
                    cache_status=variant.cache_status,
                ),
                bindings=variant.bindings,
                artifact_records=report_input.artifact_records,
            )
            for variant in report_input.compiled_variants
        ]
    assessments = {
        assessment.region_id: assessment for assessment in report_input.backend_assessments
    }
    reports: list[CompilationCompiledRegionReport] = []
    for region in report_input.compiled_regions:
        member_ids = {member.id for member in region.members}
        bindings = tuple(
            binding for binding in report_input.compiled_bindings if binding.source in member_ids
        )
        records = tuple(
            record for record in report_input.artifact_records if record.region_id == region.id
        )
        assessment = assessments.get(region.id)
        backend: Backend | None
        if assessment is not None:
            backend = assessment.backend
        elif records:
            backend = records[0].backend
        else:
            backend = None
        reports.append(
            _compiled_region_variant_report(
                identity=_CompiledVariantReportInput(
                    variant_id=region.id,
                    region=region,
                    backend=backend,
                    cache_status="disabled",
                ),
                bindings=bindings,
                artifact_records=report_input.artifact_records,
            )
        )
    return reports


def _compiled_region_variant_report(
    *,
    identity: _CompiledVariantReportInput,
    bindings: tuple[BindingTarget, ...],
    artifact_records: tuple[ArtifactRecord, ...],
) -> CompilationCompiledRegionReport:
    records = tuple(
        record for record in artifact_records if record.region_id == identity.variant_id
    )
    return {
        "id": identity.region.id,
        "variant_id": identity.variant_id,
        "source_module": identity.region.source_module.name,
        "backend": identity.backend,
        "cache_status": identity.cache_status,
        "bindings": [
            {
                "source": binding.source.stable_id,
                "compiled_name": binding.compiled_name,
                "kind": binding.kind,
                "owner_class": binding.owner_class,
                "target_owner_class": binding.target_owner_class,
                "execution_kind": binding.execution_kind,
                "required": binding.required,
                "guards": [_runtime_guard_report(guard) for guard in binding.guards],
            }
            for binding in bindings
        ],
        "artifacts": sorted(record.install_relative_path for record in records),
    }


def write_compilation_json_report(path: Path, report: CompilationReport) -> None:
    """Write a machine-readable compilation report as sorted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8")


def write_compilation_markdown_report(path: Path, report: CompilationReport) -> None:
    """Write the human-readable compilation report for CLI workflows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_compilation_markdown_report(report), encoding="utf-8")


def render_compilation_markdown_report(report: CompilationReport) -> str:
    """Render a concise Markdown report for a build or compile attempt."""
    status = "success" if report["success"] else "failed"
    module_filter = report["module_filter"] or "all enabled modules"
    lines = [
        "# Atoll Compilation Report",
        "",
        "## Summary",
        "",
        f"- Operation: {report['operation']}",
        f"- Mode: {report['mode']}",
        f"- Status: {status}",
        f"- Module filter: {module_filter}",
        f"- Wheel: {_optional_path(report['wheel_path'])}",
        f"- Islands: {report['summary']['islands']}",
        f"- Typed regions: {report['summary']['typed_regions']}",
        f"- Compiled regions: {report['summary']['compiled_regions']}",
        f"- Symbols: {report['summary']['symbols']}",
        f"- Native-ready scan candidates: {report['summary']['native_ready_symbols']}",
        f"- Rejected scan candidates: {report['summary']['native_rejected_symbols']}",
        f"- Artifacts: {report['summary']['artifacts']}",
        f"- Support artifacts: {report['summary']['support_artifacts']}",
        f"- Skipped modules: {report['summary']['skipped_modules']}",
        f"- Preflight blockers: {report['summary']['preflight_blockers']}",
        f"- Verified islands: {report['summary']['verified']}",
        f"- Verification failures: {report['summary']['verify_failures']}",
        f"- Subprocess verifications: {report['summary']['subprocess_verifications']}",
        (
            "- Subprocess verification failures: "
            f"{report['summary']['subprocess_verification_failures']}"
        ),
        f"- Semantic tests: {_semantic_test_summary(report['tests'])}",
        f"- Performance: {report['summary']['performance_status']}",
        f"- Build duration: {report['summary']['duration_seconds']:.3f}s",
        "",
        "## Verification Scope",
        "",
        _verification_scope_text(report["mode"]),
        "",
        "## Native Readiness",
        "",
        (
            "Scan scores estimate extraction safety. Source-clean compile separately checks the "
            "generated code for concrete native types and repeated primitive work before mypyc."
        ),
        "",
    ]
    if report["native_readiness"]:
        lines.extend(
            (
                f"- `{readiness['source_module']}.{readiness['symbol']}`: "
                f"{'ready' if readiness['eligible'] else 'rejected'} "
                f"({readiness['score']}/100)"
                + (f"; {'; '.join(readiness['reasons'])}" if readiness["reasons"] else "")
            )
            for readiness in report["native_readiness"]
        )
    else:
        lines.append("- Not evaluated for this compilation mode.")
    if report["typed_regions"]:
        lines.extend(["", "### Planned Regions", ""])
        lines.extend(
            f"- `{region['id']}`: " + ", ".join(member["id"] for member in region["members"])
            for region in report["typed_regions"]
        )
    _append_compiled_regions_markdown(lines, report["compiled_regions"])
    lines.extend(
        [
            "",
            "## Build",
            "",
            f"- Success: {_yes_no(report['build']['success'])}",
            f"- Command: `{' '.join(report['build']['command'])}`",
            f"- Cache: {report['build']['cache_status']}",
        ]
    )
    if report["build"]["stderr"]:
        lines.append(f"- Error: `{_first_line(report['build']['stderr'])}`")
    if report["build"]["phase_timings"]:
        lines.extend(["", "### Phase Timings", ""])
        lines.extend(
            f"- {timing['name']}: {timing['duration_seconds']:.3f}s"
            + (f" ({timing['detail']})" if timing["detail"] else "")
            for timing in report["build"]["phase_timings"]
        )
    if report["build"]["artifacts"]:
        lines.extend(["", "### Artifacts", ""])
        lines.extend(f"- `{artifact}`" for artifact in report["build"]["artifacts"])
    if report["build"]["support_artifacts"]:
        lines.extend(["", "### Support Artifacts", ""])
        lines.extend(f"- `{artifact}`" for artifact in report["build"]["support_artifacts"])
    _append_test_gate_markdown(lines, report)
    _append_package_verification_markdown(lines, report["verification_steps"])
    _append_performance_markdown(lines, report["performance"])
    _append_cleanup_markdown(lines, report["cleanup"])
    _append_source_clean_skip_markdown(lines, report)
    lines.extend(["", "## Islands", ""])
    if not report["islands"]:
        lines.append("- None")
    for island in report["islands"]:
        lines.extend(_compilation_markdown_island(island))
    return "\n".join(lines).rstrip() + "\n"


def _append_test_gate_markdown(lines: list[str], report: CompilationReport) -> None:
    lines.extend(["", "## Test Gate", ""])
    if report["tests"] is None and not report["test_results"]:
        lines.append("- Not run")
    elif report["tests"] is not None:
        lines.extend(
            [
                f"- Command: `{' '.join(report['tests']['command'])}`",
                f"- Exit code: {report['tests']['exit_code']}",
                f"- Success: {_yes_no(report['tests']['success'])}",
            ]
        )
    for result in report["test_results"]:
        lines.extend(
            [
                f"- {result['mode']}: `{' '.join(result['command'])}`",
                (
                    f"  exit {result['returncode']}, "
                    f"{result['duration_seconds']:.3f}s, "
                    f"{'passed' if result['success'] else 'failed'}"
                ),
            ]
        )


def _append_package_verification_markdown(
    lines: list[str],
    results: list[CompilationVerificationStepReport],
) -> None:
    lines.extend(["", "## Package Verification", ""])
    if not results:
        lines.append("- Not run")
        return
    lines.extend(
        (
            f"- {result['stage']}: {'passed' if result['success'] else 'failed'} "
            f"in {result['duration_seconds']:.3f}s"
        )
        for result in results
    )


def _append_performance_markdown(
    lines: list[str],
    performance: CompilationPerformanceReport,
) -> None:
    lines.extend(
        [
            "",
            "## Performance",
            "",
            f"- Status: {performance['status']}",
            f"- Reason: {performance['reason']}",
            f"- Minimum speedup: {performance['minimum_speedup']:.3f}x",
        ]
    )
    if performance["speedup"] is None:
        return
    lines.extend(
        [
            f"- Baseline median: {_optional_seconds(performance['baseline_median_seconds'])}",
            f"- Compiled median: {_optional_seconds(performance['compiled_median_seconds'])}",
            f"- Speedup: {performance['speedup']:.3f}x",
        ]
    )


def _compiled_region_markdown(region: CompilationCompiledRegionReport) -> str:
    backend = region["backend"] or "unknown backend"
    bindings = ", ".join(_compiled_binding_markdown(binding) for binding in region["bindings"])
    artifacts = ", ".join(region["artifacts"]) or "no recorded artifacts"
    return (
        f"- `{region['variant_id']}` [{backend}] for region `{region['id']}`: "
        f"{bindings}; cache: {region['cache_status']}; artifacts: {artifacts}"
    )


def _compiled_binding_markdown(binding: CompilationCompiledBindingReport) -> str:
    target = binding["target_owner_class"]
    target_text = f" -> {target}" if target is not None else ""
    guard_text = f", {len(binding['guards'])} guard(s)" if binding["guards"] else ""
    return f"{binding['source']}{target_text} ({binding['kind']}{guard_text})"


def _append_compiled_regions_markdown(
    lines: list[str],
    regions: list[CompilationCompiledRegionReport],
) -> None:
    if not regions:
        return
    lines.extend(["", "### Compiled Regions", ""])
    lines.extend(_compiled_region_markdown(region) for region in regions)


def _append_cleanup_markdown(lines: list[str], cleanup: CompilationCleanupReport) -> None:
    lines.extend(["", "## Cleanup", ""])
    lines.extend(
        [f"- Removed `{path}`" for path in cleanup["removed"]]
        if cleanup["removed"]
        else ["- Removed: none"]
    )
    lines.extend(
        [f"- Kept `{path}`" for path in cleanup["kept"]] if cleanup["kept"] else ["- Kept: none"]
    )


def _append_source_clean_skip_markdown(lines: list[str], report: CompilationReport) -> None:
    if report["skipped_modules"]:
        lines.extend(["", "## Skipped Modules", ""])
        lines.extend(
            f"- `{skipped['module']}`: {_first_line(skipped['reason'])}"
            for skipped in report["skipped_modules"]
        )
    if report["preflight_blockers"]:
        lines.extend(["", "## Preflight Blockers", ""])
        lines.extend(
            (
                f"- `{blocker['module']}` ({blocker['path']}"
                f"{_line_suffix(blocker['line'])}): {blocker['message']}"
            )
            for blocker in report["preflight_blockers"]
        )


def _module_report(module: ModuleScan) -> ModuleReport:
    return {
        "module": module.module.name,
        "path": str(module.module.path),
        "imports": [
            {
                "source_text": record.source_text,
                "imported_names": list(record.imported_names),
                "module": record.module,
                "level": record.level,
                "lineno": record.lineno,
                "end_lineno": record.end_lineno,
            }
            for record in module.imports
        ],
        "constants": [
            {
                "name": record.name,
                "kind": record.kind,
                "source_text": record.source_text,
                "lineno": record.lineno,
                "end_lineno": record.end_lineno,
            }
            for record in module.constants
        ],
        "symbols": [
            {
                "id": symbol.id.stable_id,
                "qualname": symbol.id.qualname,
                "kind": symbol.kind,
                "visibility": symbol.visibility,
                "lineno": symbol.lineno,
                "end_lineno": symbol.end_lineno,
                "decorators": list(symbol.decorators),
                "arg_count": symbol.arg_count,
                "annotated_arg_count": symbol.annotated_arg_count,
                "has_return_annotation": symbol.has_return_annotation,
                "has_any_annotation": symbol.has_any_annotation,
                "called_names": list(symbol.called_names),
                "uses_globals": list(symbol.uses_globals),
                "local_names": list(symbol.local_names),
                "referenced_names": list(symbol.referenced_names),
                "owner_class": symbol.owner_class,
                "binding_kind": symbol.binding_kind,
                "execution_kind": symbol.execution_kind,
                "type_parameters": list(symbol.type_parameters),
                "parameters": [
                    {
                        "name": parameter.name,
                        "kind": parameter.kind,
                        "annotation": parameter.annotation,
                        "default_source": parameter.default_source,
                    }
                    for parameter in symbol.parameters
                ],
                "return_annotation": symbol.return_annotation,
                "annotation_names": list(symbol.annotation_names),
                "called_paths": list(symbol.called_paths),
                "base_names": list(symbol.base_names),
                "fields": [
                    {
                        "name": field.name,
                        "annotation": field.annotation,
                        "default_source": field.default_source,
                        "class_variable": field.class_variable,
                    }
                    for field in symbol.fields
                ],
                "declaration_start_lineno": symbol.declaration_start_lineno,
                "scope_type_parameters": list(symbol.scope_type_parameters),
                "type_parameter_records": [
                    _type_parameter_report(record) for record in symbol.type_parameter_records
                ],
                "scope_type_parameter_records": [
                    _type_parameter_report(record) for record in symbol.scope_type_parameter_records
                ],
                "any_annotation_sources": list(symbol.any_annotation_sources),
                "blockers": [_blocker_report(blocker) for blocker in symbol.blockers],
                "mypy_diagnostics": [
                    _mypy_diagnostic_report(diagnostic) for diagnostic in symbol.mypy_diagnostics
                ],
            }
            for symbol in module.symbols
        ],
        "blockers": [_blocker_report(blocker) for blocker in module.blockers],
        "top_level_statement_lines": list(module.top_level_statement_lines),
        "mypy_diagnostics": [
            _mypy_diagnostic_report(diagnostic) for diagnostic in module.mypy_diagnostics
        ],
        "dependency_edges": [
            {
                "src": edge.src.stable_id,
                "dst": _edge_dst_text(edge.dst),
                "kind": edge.kind,
                "confidence": edge.confidence,
                "lineno": edge.lineno,
            }
            for edge in module.dependency_edges
        ],
        "island_candidates": [
            {
                "symbols": [symbol.stable_id for symbol in candidate.symbols],
                "required_imports": list(candidate.required_imports),
                "required_constants": list(candidate.required_constants),
                "required_local_symbols": [
                    symbol.stable_id for symbol in candidate.required_local_symbols
                ],
                "rejected_symbols": [symbol.stable_id for symbol in candidate.rejected_symbols],
                "score": candidate.score,
                "score_label": score_label(candidate.score),
                "score_summary": score_summary(candidate.score),
                "risk": candidate.risk,
                "risk_summary": risk_summary(candidate.risk),
                "reasons": list(candidate.reasons),
            }
            for candidate in module.island_candidates
        ],
        "poison_radii": [
            {
                "poison": radius.poison.stable_id,
                "impacted": [symbol.stable_id for symbol in radius.impacted],
                "reason": radius.reason,
            }
            for radius in module.poison_radii
        ],
        "typed_regions": [_typed_region_report(region) for region in module.typed_regions],
    }


def _typed_region_report(region: TypedRegion) -> TypedRegionReport:
    return {
        "id": region.id,
        "source_hash": region.source_hash,
        "atomic_class": region.atomic_class,
        "members": [
            {
                "id": member.id.stable_id,
                "kind": member.kind,
                "owner_class": member.owner_class,
                "binding_kind": member.binding_kind,
                "execution_kind": member.execution_kind,
                "type_parameters": list(member.type_parameters),
                "scope_type_parameters": list(member.scope_type_parameters),
                "type_parameter_records": [
                    _type_parameter_report(record) for record in member.type_parameter_records
                ],
                "scope_type_parameter_records": [
                    _type_parameter_report(record) for record in member.scope_type_parameter_records
                ],
                "parameters": [
                    {
                        "name": parameter.name,
                        "kind": parameter.kind,
                        "annotation": parameter.annotation,
                        "default_source": parameter.default_source,
                    }
                    for parameter in member.parameters
                ],
                "return_annotation": member.return_annotation,
                "fields": [
                    {
                        "name": field.name,
                        "annotation": field.annotation,
                        "default_source": field.default_source,
                        "class_variable": field.class_variable,
                    }
                    for field in member.fields
                ],
            }
            for member in region.members
        ],
        "dependencies": [
            {
                "src": dependency.src.stable_id,
                "dst": _edge_dst_text(dependency.dst),
                "kind": dependency.kind,
                "confidence": dependency.confidence,
                "role": dependency.role,
                "type_only": dependency.type_only,
            }
            for dependency in region.dependencies
        ],
        "type_bindings": [
            {
                "name": binding.name,
                "annotation": binding.annotation,
                "source": binding.source,
                "concrete": binding.concrete,
                "substitutions": [list(item) for item in binding.substitutions],
            }
            for binding in region.type_bindings
        ],
        "bindings": [
            {
                "source": binding.source.stable_id,
                "compiled_name": binding.compiled_name,
                "kind": binding.kind,
                "owner_class": binding.owner_class,
                "target_owner_class": binding.target_owner_class,
                "execution_kind": binding.execution_kind,
                "required": binding.required,
                "guards": [_runtime_guard_report(guard) for guard in binding.guards],
            }
            for binding in region.bindings
        ],
        "decisions": [
            {
                "target": decision.target,
                "action": decision.action,
                "reason": decision.reason,
            }
            for decision in region.decisions
        ],
        "specializations": [
            {
                "id": specialization.id,
                "source_member": specialization.source_member.stable_id,
                "source_owner_class": specialization.source_owner_class,
                "target_owner_class": specialization.target_owner_class,
                "origin": specialization.origin,
                "substitutions": [list(item) for item in specialization.substitutions],
                "guards": [_runtime_guard_report(guard) for guard in specialization.guards],
                "type_bindings": [
                    {
                        "name": binding.name,
                        "annotation": binding.annotation,
                        "source": binding.source,
                        "concrete": binding.concrete,
                        "substitutions": [list(item) for item in binding.substitutions],
                    }
                    for binding in specialization.type_bindings
                ],
            }
            for specialization in region.specializations
        ],
    }


def _runtime_guard_report(guard: RuntimeTypeGuard) -> RuntimeTypeGuardReport:
    """Serialize one runtime guard without exposing target code objects."""
    return {
        "parameter_name": guard.parameter_name,
        "positional_index": guard.positional_index,
        "annotation": guard.annotation,
        "nominal_type_paths": list(guard.nominal_type_paths),
        "allow_none": guard.allow_none,
    }


def _type_parameter_report(record: TypeParameterRecord) -> TypeParameterReport:
    return {
        "name": record.name,
        "kind": record.kind,
        "declaration": record.declaration,
    }


def _blocker_report(blocker: Blocker) -> BlockerReport:
    return {
        "severity": blocker.severity,
        "code": blocker.code,
        "message": blocker.message,
        "lineno": blocker.lineno,
        "symbol": blocker.symbol.stable_id if blocker.symbol is not None else None,
    }


def _mypy_diagnostic_report(diagnostic: MypyDiagnostic) -> MypyDiagnosticReport:
    return {
        "path": str(diagnostic.path),
        "line": diagnostic.line,
        "column": diagnostic.column,
        "severity": diagnostic.severity,
        "code": diagnostic.code,
        "message": diagnostic.message,
        "symbol": diagnostic.symbol.stable_id if diagnostic.symbol is not None else None,
    }


def _edge_dst_text(dst: SymbolId | str) -> str:
    if isinstance(dst, SymbolId):
        return dst.stable_id
    return dst


def score_label(score: int) -> str:
    """Return a short label for a candidate score."""
    if score >= _STRONG_SCORE:
        return "strong"
    if score >= _GOOD_SCORE:
        return "good"
    if score >= _POSSIBLE_SCORE:
        return "possible"
    return "weak"


def score_summary(score: int) -> str:
    """Explain a scan-only candidate score in user-facing language."""
    label = score_label(score)
    if label == "strong":
        detail = "very promising scan-only candidate"
    elif label == "good":
        detail = "promising scan-only candidate"
    elif label == "possible":
        detail = "worth trying, but less compelling"
    else:
        detail = "below Atoll's normal recommendation threshold"
    return f"{score}/100, {detail}"


def risk_summary(risk: IslandRisk) -> str:
    """Explain candidate extraction risk in user-facing scan report language."""
    if risk == "low":
        return "low extraction risk; only high-confidence internal dependencies were seen"
    if risk == "medium":
        return "medium extraction risk; a low-confidence dependency needs trial validation"
    return "high extraction risk; expect manual review before enabling"


def _markdown_module(module: ModuleReport) -> list[str]:
    lines = [f"## {module['module']}", ""]
    if not module["symbols"]:
        lines.extend(["No top-level symbols found.", ""])
        return lines
    for symbol in module["symbols"]:
        blocker_codes = ", ".join(blocker["code"] for blocker in symbol["blockers"])
        status = blocker_codes or "no blockers"
        lines.append(f"- `{symbol['qualname']}` ({symbol['kind']}): {status}")
    if module["island_candidates"]:
        lines.extend(["", "Candidates:"])
        for candidate in module["island_candidates"]:
            symbols = ", ".join(f"`{symbol}`" for symbol in candidate["symbols"])
            lines.append(f"- {candidate['score_summary']}; {candidate['risk_summary']}: {symbols}")
    if module["poison_radii"]:
        lines.extend(["", "Poison residue:"])
        for radius in module["poison_radii"]:
            impacted = ", ".join(radius["impacted"]) or "none"
            lines.append(f"- `{radius['poison']}` impacts: {impacted}")
    if module["typed_regions"]:
        lines.extend(["", "Typed regions:"])
        for region in module["typed_regions"]:
            members = ", ".join(f"`{member['id']}`" for member in region["members"])
            mode = "atomic class" if region["atomic_class"] else "member region"
            lines.append(f"- `{region['id']}` ({mode}): {members}")
    lines.append("")
    return lines


def _island_artifacts(
    island: EnabledIslandConfig,
    artifact_paths: tuple[Path, ...],
) -> tuple[Path, ...]:
    return tuple(
        path for path in artifact_paths if path.name.startswith(f"{island.sidecar_path.stem}.")
    )


def _compilation_verify_report(result: VerifyResult | None) -> CompilationVerifyReport | None:
    if result is None:
        return None
    return {
        "active": result.active,
        "compiled": result.compiled,
        "origin": result.origin,
        "symbols": [{"symbol": symbol, "rebound": rebound} for symbol, rebound in result.symbols],
        "error": result.error,
    }


def _compilation_test_report(result: PytestRunResult | None) -> CompilationTestReport | None:
    if result is None:
        return None
    return {
        "command": list(result.command),
        "exit_code": result.exit_code,
        "success": result.success,
    }


def _compilation_command_run_report(
    root: Path,
    result: CommandRunEvidence,
) -> CompilationCommandRunReport:
    return {
        "command": list(result.command),
        "mode": result.mode,
        "payload_root": _path_text(root, result.payload_root),
        "returncode": result.returncode,
        "success": result.succeeded,
        "duration_seconds": result.duration_seconds,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _compilation_verification_step_report(
    root: Path,
    result: PackageVerificationResult,
) -> CompilationVerificationStepReport:
    return {
        "stage": result.stage,
        "target": _path_text(root, result.target),
        "command": _build_command_report(root, result.command),
        "success": result.success,
        "exit_code": result.exit_code,
        "duration_seconds": result.duration_seconds,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _compilation_performance_report(
    root: Path,
    result: BenchmarkGateResult | None,
) -> CompilationPerformanceReport:
    if result is None:
        return {
            "status": "unbenchmarked",
            "reason": "no benchmark command configured",
            "minimum_speedup": 1.10,
            "baseline_median_seconds": None,
            "compiled_median_seconds": None,
            "speedup": None,
            "warmups": [],
            "samples": [],
        }
    return {
        "status": result.status,
        "reason": result.reason,
        "minimum_speedup": result.minimum_speedup,
        "baseline_median_seconds": result.baseline_median_seconds,
        "compiled_median_seconds": result.compiled_median_seconds,
        "speedup": result.speedup,
        "warmups": [_compilation_command_run_report(root, run) for run in result.warmups],
        "samples": [_compilation_command_run_report(root, run) for run in result.samples],
    }


def _semantic_test_summary(result: CompilationTestReport | None) -> str:
    if result is None:
        return "not run"
    if result["success"]:
        return f"passed (`{' '.join(result['command'])}`)"
    return f"failed (`{' '.join(result['command'])}`, exit code {result['exit_code']})"


def _verification_scope_text(mode: CompilationMode) -> str:
    if mode == "source-clean":
        return (
            "Source-clean compile overlays native artifacts and staged shims onto the target "
            "project's normal PEP 517 wheel. Fresh child interpreters verify both the unpacked "
            "payload and final wheel. Semantic equivalence and speedup are claimed only when "
            "the configured test and benchmark gates pass."
        )
    return (
        "Atoll runtime verification proves managed shims import compiled extensions "
        "and rebound configured symbols. It does not prove semantic equivalence "
        "unless the semantic test gate below passed."
    )


def _compilation_markdown_island(island: CompilationIslandReport) -> list[str]:
    verification = island["verification"]
    if verification is None:
        verify_status = "not run"
    elif verification["error"] is None:
        verify_status = "ok"
    else:
        verify_status = f"failed: {verification['error']}"
    lines = [
        f"### {island['source_module']}",
        "",
        f"- Source: `{island['source_path']}`",
        f"- Generated module: `{island['generated_module']}`",
        f"- Symbols: {', '.join(island['symbols'])}",
        f"- Verification: {verify_status}",
    ]
    if island["artifacts"]:
        lines.append("- Artifacts:")
        lines.extend(f"  - `{artifact}`" for artifact in island["artifacts"])
    else:
        lines.append("- Artifacts: none")
    lines.append("")
    return lines


def _path_text(root: Path, path: Path) -> str:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        return str(resolved_path)


def _optional_path(path: str | None) -> str:
    return f"`{path}`" if path is not None else "none"


def _optional_seconds(value: float | None) -> str:
    return f"{value:.3f}s" if value is not None else "unknown"


def _line_suffix(line: int | None) -> str:
    return f":{line}" if line is not None else ""


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _first_line(value: str) -> str:
    return next((line for line in value.splitlines() if line.strip()), "")


def _build_command_report(root: Path, command: tuple[str, ...]) -> list[str]:
    reported: list[str] = []
    generated_inputs = 0
    for item in command:
        if _is_generated_input_text(root, item):
            generated_inputs += 1
            continue
        if generated_inputs:
            reported.append(_generated_inputs_label(generated_inputs))
            generated_inputs = 0
        reported.append(item)
    if generated_inputs:
        reported.append(_generated_inputs_label(generated_inputs))
    return reported


def _generated_input_cleanup_reports(
    root: Path,
    paths: tuple[Path, ...],
) -> list[str]:
    count = sum(_is_generated_input_path(root, path) for path in paths)
    return [_generated_inputs_label(count)] if count else []


def _generated_inputs_label(count: int) -> str:
    suffix = "" if count == 1 else "s"
    return f"<{count} generated Python build input{suffix}>"


def _is_generated_input_text(root: Path, value: str) -> bool:
    if ".atoll/sidecars/" in value or value == ".atoll/sidecars":
        return True
    try:
        return _is_generated_input_path(root, Path(value))
    except OSError:
        return False


def _is_generated_input_path(root: Path, path: Path) -> bool:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    parts = relative.parts
    return (
        len(parts) > _ATOLL_GENERATED_INPUT_DIR_INDEX
        and parts[_ATOLL_PART_INDEX] == ".atoll"
        and parts[_ATOLL_GENERATED_INPUT_DIR_INDEX] == "sidecars"
    )
