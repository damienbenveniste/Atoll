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
from atoll.analysis.suspension_planner import (
    RejectionEvidence,
    SuspensionBlock,
    plan_suspension_blocks,
)
from atoll.analysis.task_fusion import FusionPlan
from atoll.execution_plans.models import (
    ExecutionPlan,
    ExecutionPlanDiagnostic,
    ExecutionPlanTrial,
    PlanRejection,
)
from atoll.models import (
    ArtifactRecord,
    Backend,
    BackendAssessment,
    BindingKind,
    BindingTarget,
    Blocker,
    BlockerSeverity,
    CallSiteFact,
    CandidateTrial,
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
    ImportRecord,
    InvocationMode,
    IslandRisk,
    LossAction,
    LoweringMode,
    ModuleScan,
    MypyDiagnostic,
    ParameterKind,
    PytestRunResult,
    RuntimeTypeGuard,
    ScanResult,
    SpecializationOrigin,
    SuspensionKind,
    SuspensionPoint,
    SymbolId,
    SymbolKind,
    TypedRegion,
    TypeParameterKind,
    TypeParameterRecord,
    VerifyResult,
    Visibility,
)
from atoll.optimization_policy import (
    DEFAULT_MINIMUM_FINAL_SPEEDUP,
    DEFAULT_MINIMUM_MARGINAL_SPEEDUP,
    HARD_BENCHMARK_MINIMUM_SPEEDUP,
    MINIMUM_STABLE_MEDIAN_SECONDS,
    PROFILE_GUIDED_MINIMUM_MARGINAL_SPEEDUP,
)
from atoll.runtime.fusion_performance import FusionArmRunEvidence, FusionTrial
from atoll.runtime.package_verify import PackageVerificationResult
from atoll.runtime.performance import BenchmarkGateResult, CommandRunEvidence
from atoll.runtime.profiling import LifecycleCounts, ProfileResult
from atoll.source_optimization.models import (
    SourceAccessSite,
    SourceCallableEvidence,
    SourceEdit,
    SourceOptimizationApplicationStatus,
    SourceOptimizationAssessment,
    SourceOptimizationPlan,
    SourceOptimizationTrial,
    SourceTransformationKind,
    TransformationStep,
)

_STRONG_SCORE = 90
_GOOD_SCORE = 80
_POSSIBLE_SCORE = 70
_ATOLL_PART_INDEX = 0
_ATOLL_GENERATED_INPUT_DIR_INDEX = 1

SCAN_REPORT_SCHEMA_VERSION = 3
COMPILE_REPORT_SCHEMA_VERSION = 6
OPTIMIZATION_POLICY_VERSION = 1


class BlockerReport(TypedDict):
    """Serialized blocker shown in scan and symbol reports.

    Attributes:
        severity: Diagnostic severity used for filtering and reporting.
        code: Stable machine-readable diagnostic or blocker code.
        message: Human-readable diagnostic or blocker explanation.
        lineno: One-based first source line covered by the record.
        symbol: Stable symbol identifier associated with this record.
    """

    severity: BlockerSeverity
    code: str
    message: str
    lineno: int | None
    symbol: str | None


class ImportReport(TypedDict):
    """Serialized top-level import with original source text preserved.

    Attributes:
        source_text: Exact source text retained for analysis or generation.
        imported_names: Names introduced into the module namespace by the import.
        module: Imported module path, or `None` for imports without one.
        level: Relative import level; zero denotes an absolute import.
        lineno: One-based first source line covered by the record.
        end_lineno: One-based final source line covered by the record.
    """

    source_text: str
    imported_names: list[str]
    module: str | None
    level: int
    lineno: int
    end_lineno: int


class ConstantReport(TypedDict):
    """Serialized top-level assignment and its extraction safety classification.

    Attributes:
        name: Top-level assignment name.
        kind: Literal, runtime-dynamic, or unknown safety classification.
        source_text: Exact source text retained for analysis or generation.
        lineno: One-based first source line covered by the record.
        end_lineno: One-based final source line covered by the record.
    """

    name: str
    kind: ConstantKind
    source_text: str
    lineno: int
    end_lineno: int


class SymbolReport(TypedDict):
    """Serialized AST, blocker, and type-checker facts for one symbol.

    Attributes:
        id: Stable `module::qualname` symbol identifier.
        qualname: Module-local qualified symbol name.
        kind: Function, class, or method declaration kind.
        visibility: Public or private source visibility.
        lineno: One-based first source line covered by the record.
        end_lineno: One-based final source line covered by the record.
        decorators: Source text for decorators applied to the symbol.
        arg_count: Total caller-visible parameter count.
        annotated_arg_count: Number of parameters with explicit annotations.
        has_return_annotation: Whether the callable declares a return annotation.
        has_any_annotation: Whether any visible annotation contains `Any`.
        called_names: Simple names observed in call position.
        uses_globals: Module globals read by the symbol body.
        local_names: Names bound locally within the symbol body.
        referenced_names: All names read by the symbol body or annotations.
        owner_class: Source owner class for a method binding, when applicable.
        binding_kind: Runtime descriptor or module binding classification.
        execution_kind: Synchronous, generator, coroutine, async-generator, or class shape.
        type_parameters: Type parameter names declared directly by the symbol.
        parameters: Exact source parameter declarations in call order.
        return_annotation: Exact source return annotation, when present.
        annotation_names: Names referenced by source annotations.
        called_paths: Dotted call targets recovered from source syntax.
        call_sites: Ordered source call facts retained for region planning.
        suspension_points: Ordered coroutine and generator suspension boundaries.
        runtime_imports: Function-local imports executed by the declaration.
        base_names: Base-class expressions referenced by the declaration.
        fields: Typed class fields retained for region planning.
        declaration_start_lineno: First line of decorators or declaration syntax.
        scope_type_parameters: Type parameter names inherited from enclosing scopes.
        type_parameter_records: Structured type parameters declared directly by the symbol.
        scope_type_parameter_records: Structured type parameters inherited from enclosing scopes.
        any_annotation_sources: Locations where `Any` enters the symbol's type surface.
        blockers: Conservative blockers attached to this module or symbol.
        mypy_diagnostics: Mypy diagnostics mapped to this module or symbol.
    """

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
    call_sites: list[CallSiteReport]
    suspension_points: list[SuspensionPointReport]
    runtime_imports: list[ImportReport]
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
    """Exact source parameter evidence retained by typed-region analysis.

    Attributes:
        name: Source parameter name without `*` or `**` prefixes.
        kind: Positional, variadic, keyword-only, or keyword variadic kind.
        annotation: Exact source annotation text.
        default_source: Exact default-value source text, or `None` when required.
    """

    name: str
    kind: ParameterKind
    annotation: str | None
    default_source: str | None


class FieldReport(TypedDict):
    """Typed class field evidence retained for class-region planning.

    Attributes:
        name: Class field name.
        annotation: Exact source annotation text.
        default_source: Exact default-value source text, or `None` when required.
        class_variable: Whether the field is declared as a class variable.
    """

    name: str
    annotation: str
    default_source: str | None
    class_variable: bool


class TypeParameterReport(TypedDict):
    """Exact type-parameter declaration retained for backend assessment.

    Attributes:
        name: Type parameter name visible in source.
        kind: `TypeVar`, `ParamSpec`, or `TypeVarTuple` classification.
        declaration: Exact source declaration for the type parameter.
    """

    name: str
    kind: TypeParameterKind
    declaration: str


class CallSiteReport(TypedDict):
    """Ordered syntax evidence for one call inside a source declaration.

    Attributes:
        target: Source-level call target expression.
        root_name: First lexical name in the target expression.
        invocation_mode: Ordinary, awaited, or async-iteration call mode.
        lineno: One-based first source line covered by the call.
        end_lineno: One-based final source line covered by the call.
        col_offset: Zero-based first source column covered by the call.
        end_col_offset: Zero-based final source column, when available.
        requires_same_unit: Whether the call must share a native compilation unit.
    """

    target: str
    root_name: str
    invocation_mode: InvocationMode
    lineno: int
    end_lineno: int
    col_offset: int
    end_col_offset: int | None
    requires_same_unit: bool


class SuspensionPointReport(TypedDict):
    """Source location and syntax kind for one suspension boundary.

    Attributes:
        kind: Await, yield, async-loop, or async-context suspension kind.
        lineno: One-based first source line covered by the suspension.
        end_lineno: One-based final source line covered by the suspension.
        col_offset: Zero-based first source column covered by the suspension.
        end_col_offset: Zero-based final source column, when available.
    """

    kind: SuspensionKind
    lineno: int
    end_lineno: int
    col_offset: int
    end_col_offset: int | None


class MypyDiagnosticReport(TypedDict):
    """Serialized mypy diagnostic after optional symbol range mapping.

    Attributes:
        path: Absolute source path reported by mypy.
        line: One-based diagnostic line.
        column: One-based diagnostic column, when mypy reports one.
        severity: Diagnostic severity used for filtering and reporting.
        code: Stable machine-readable diagnostic or blocker code.
        message: Human-readable diagnostic or blocker explanation.
        symbol: Stable symbol identifier associated with this record.
    """

    path: str
    line: int
    column: int | None
    severity: DiagnosticSeverity
    code: str | None
    message: str
    symbol: str | None


class DependencyEdgeReport(TypedDict):
    """Serialized same-module dependency or external boundary edge evidence.

    Attributes:
        src: Source symbol for the dependency edge.
        dst: Dependency destination symbol or external boundary.
        kind: Calls, global use, inheritance, decoration, import, or annotation edge kind.
        confidence: Confidence assigned to the dependency evidence.
        lineno: One-based first source line covered by the record.
        invocation_mode: Call execution mode when the edge comes from a call site.
        requires_same_unit: Whether the dependency must share a compilation unit.
    """

    src: str
    dst: str
    kind: DependencyKind
    confidence: Confidence
    lineno: int | None
    invocation_mode: InvocationMode | None
    requires_same_unit: bool


class IslandCandidateReport(TypedDict):
    """Serialized island recommendation with score, risk, and dependency context.

    Attributes:
        symbols: Stable IDs of symbols recommended as one compilation candidate.
        required_imports: Imports required by the candidate.
        required_constants: Literal constants required by the candidate.
        required_local_symbols: Local symbols required by dependency closure.
        rejected_symbols: Symbols excluded from the candidate dependency closure.
        score: Scan-only extraction-safety or native-readiness score.
        score_label: Short qualitative score label.
        score_summary: User-facing explanation of the score.
        risk: Conservative extraction risk classification.
        risk_summary: User-facing explanation of extraction risk.
        reasons: Deterministically ordered evidence supporting the decision.
    """

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
    """Serialized explanation of a rejected symbol's impact on candidates.

    Attributes:
        poison: Rejected symbol whose dependencies affect nearby candidates.
        impacted: Otherwise viable symbols affected by the rejected symbol.
        reason: Concrete blocker or dependency evidence causing the impact.
    """

    poison: str
    impacted: list[str]
    reason: str


class RegionMemberReport(TypedDict):
    """One unlowered declaration included in a typed region.

    Attributes:
        id: Stable ID of the retained source declaration.
        kind: Function, class, or method declaration kind.
        owner_class: Source owner class for a method binding, when applicable.
        binding_kind: Runtime descriptor or module binding classification.
        execution_kind: Synchronous, generator, coroutine, async-generator, or class shape.
        type_parameters: Type parameter names declared directly by the symbol.
        scope_type_parameters: Type parameter names inherited from enclosing scopes.
        type_parameter_records: Structured type parameters declared directly by the symbol.
        scope_type_parameter_records: Structured type parameters inherited from enclosing scopes.
        parameters: Exact source parameter declarations in call order.
        return_annotation: Exact source return annotation, when present.
        fields: Typed class fields retained for region planning.
        call_sites: Ordered source call facts retained for region planning.
        suspension_points: Ordered coroutine and generator suspension boundaries.
        runtime_imports: Function-local imports executed by the declaration.
    """

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
    call_sites: list[CallSiteReport]
    suspension_points: list[SuspensionPointReport]
    runtime_imports: list[ImportReport]
    fields: list[FieldReport]


class RegionDependencyReport(TypedDict):
    """Dependency retained with runtime versus type-only intent.

    Attributes:
        src: Source symbol for the dependency edge.
        dst: Dependency destination symbol or external boundary.
        kind: Calls, global use, inheritance, decoration, import, or annotation edge kind.
        confidence: Confidence assigned to the dependency evidence.
        role: Whether the dependency is required at runtime, for typing, or by the facade.
        type_only: Whether the dependency is used exclusively for typing.
        lineno: One-based source line for the dependency evidence, when available.
        invocation_mode: Call execution mode when the dependency comes from a call site.
        requires_same_unit: Whether the dependency must share a compilation unit.
    """

    src: str
    dst: str
    kind: DependencyKind
    confidence: Confidence
    role: DependencyRole
    type_only: bool
    lineno: int | None
    invocation_mode: InvocationMode | None
    requires_same_unit: bool


class TypeBindingReport(TypedDict):
    """Source type evidence retained before backend lowering.

    Attributes:
        name: Parameter, return, field, base, type-parameter, or import name.
        annotation: Exact source annotation text.
        source: Category from which the type evidence was obtained.
        concrete: Whether the type evidence is fully concrete for lowering.
        substitutions: Concrete type substitutions applied to generic parameters.
    """

    name: str
    annotation: str
    source: str
    concrete: bool
    substitutions: list[list[str]]


class LoweringDecisionReport(TypedDict):
    """Auditable region-level preservation or fallback decision.

    Attributes:
        target: Stable symbol or region fact affected by the decision.
        action: Lowering action chosen for the target.
        reason: Concrete evidence supporting the lowering action.
    """

    target: str
    action: LossAction
    reason: str


class BindingTargetReport(TypedDict):
    """Descriptor-aware source binding promised by a typed region.

    Attributes:
        source: Stable ID of the public source binding.
        compiled_name: Backend-generated attribute name containing the compiled callable.
        kind: Module, class, or descriptor-aware binding kind.
        owner_class: Source owner class for a method binding, when applicable.
        target_owner_class: Concrete runtime owner class for a specialized binding.
        execution_kind: Synchronous, generator, coroutine, async-generator, or class shape.
        required: Whether absence of the compiled binding is a verification failure.
        guards: Runtime type guards required before selecting this binding or specialization.
    """

    source: str
    compiled_name: str
    kind: BindingKind
    owner_class: str | None
    target_owner_class: str | None
    execution_kind: ExecutionKind
    required: bool
    guards: list[RuntimeTypeGuardReport]


class RuntimeTypeGuardReport(TypedDict):
    """Constant-time input check required before specialized native routing.

    Attributes:
        parameter_name: Source parameter guarded at runtime.
        positional_index: Zero-based positional argument index, when applicable.
        annotation: Exact source annotation text.
        nominal_type_paths: Importable runtime types accepted by the guard.
        allow_none: Whether `None` satisfies this runtime type guard.
    """

    parameter_name: str
    positional_index: int | None
    annotation: str
    nominal_type_paths: list[str]
    allow_none: bool


class RegionSpecializationReport(TypedDict):
    """Concrete TypeVar binding layered on an unchanged generic declaration.

    Attributes:
        id: Deterministic specialization ID.
        source_member: Generic source member from which a specialization was derived.
        source_owner_class: Owner class declared by the generic source member.
        target_owner_class: Concrete runtime owner class for a specialized binding.
        origin: Resolved module origin or specialization evidence source.
        substitutions: Concrete type substitutions applied to generic parameters.
        guards: Runtime type guards required before selecting this binding or specialization.
        type_bindings: Preserved or concretized type evidence for the region.
    """

    id: str
    source_member: str
    source_owner_class: str | None
    target_owner_class: str | None
    origin: SpecializationOrigin
    substitutions: list[list[str]]
    guards: list[RuntimeTypeGuardReport]
    type_bindings: list[TypeBindingReport]


class TypedRegionReport(TypedDict):
    """Backend-neutral typed region serialized in scan reports.

    Attributes:
        id: Deterministic typed-region ID.
        source_hash: Deterministic digest of generated or retained source.
        atomic_class: Whether the owner class must be lowered as one indivisible region.
        members: Source declarations owned by the typed region or compilation unit.
        dependencies: Runtime and typing dependencies retained by the region.
        type_bindings: Preserved or concretized type evidence for the region.
        bindings: Source bindings promised by the compiled region or variant.
        decisions: Auditable preservation, specialization, boxing, fallback, or rejection decisions.
        specializations: Guarded concrete variants available for the typed region.
    """

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
    """Complete scan report section for one discovered Python module.

    Attributes:
        module: Importable dotted name of the scanned module.
        path: Absolute source path of the scanned module.
        imports: Top-level imports retained for analysis or generation.
        constants: Top-level constants retained for analysis or sidecar generation.
        symbols: Serialized declarations discovered in the module.
        blockers: Module-level blockers not owned by one declaration.
        top_level_statement_lines: Executable module-level statements that may affect extraction.
        mypy_diagnostics: Mypy diagnostics mapped to this module or symbol.
        dependency_edges: Conservative dependency edges derived from syntax.
        island_candidates: Conservative compilation candidates for the module.
        poison_radii: Report-only impact records for rejected symbols.
        typed_regions: Backend-neutral typed regions discovered or reported.
    """

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
    """Aggregate scan counts used by JSON and Markdown summaries.

    Attributes:
        modules_scanned: Number of modules included in the scan.
        symbols_scanned: Number of source symbols included in the scan.
        island_candidates: Number of conservative compilation candidates discovered.
        typed_regions: Number of backend-neutral typed regions discovered.
        hard_blockers: Number of blockers that prevent extraction.
        soft_blockers: Number of blockers that increase extraction risk.
    """

    modules_scanned: int
    symbols_scanned: int
    island_candidates: int
    typed_regions: int
    hard_blockers: int
    soft_blockers: int


class ScanReport(TypedDict):
    """Top-level stable JSON report emitted by `atoll scan`.

    Attributes:
        version: Schema or cache format version.
        tool: Tool identifier that produced the report.
        project_root: Root directory of the target Python project.
        source_roots: Absolute import roots discovered for the target project.
        summary: Aggregate counts and status for the report.
        modules: Discovered or reported modules in deterministic order.
    """

    version: int
    tool: str
    project_root: str
    source_roots: list[str]
    summary: SummaryReport
    modules: list[ModuleReport]


CompilationOperation = Literal["build", "compile"]
CompilationMode = Literal["in-place", "source-clean"]


class CompilationNativeReadinessReport(TypedDict):
    """Post-generation evidence that a selected symbol can benefit from mypyc.

    Attributes:
        source_module: Importable source module name.
        symbol: Stable symbol identifier associated with this record.
        eligible: Whether generated code passes the native-readiness gate.
        score: Scan-only extraction-safety or native-readiness score.
        function_count: Number of generated functions inspected.
        any_typed_functions: Generated functions whose annotations contain `Any`.
        boxed_typed_functions: Generated functions whose useful values remain boxed Python objects.
        dynamic_dependencies: Generated dependencies requiring dynamic runtime lookup.
        loop_count: Number of loops found in generated functions.
        native_operation_count: Count of operations likely to benefit from native lowering.
        reasons: Deterministically ordered evidence supporting the decision.
    """

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
    """Aggregate build, verification, test, and cleanup counts for compilation.

    Attributes:
        islands: Number of legacy managed islands represented in the report.
        typed_regions: Number of backend-neutral typed regions considered.
        compiled_regions: Number of typed-region variants compiled successfully.
        symbols: Number of source symbols represented by islands and compiled bindings.
        native_ready_symbols: Number of symbols accepted by the native-readiness gate.
        native_rejected_symbols: Number of symbols rejected by the native-readiness gate.
        artifacts: Number of native artifact paths produced by the build.
        support_artifacts: Number of native support files not owned by one region.
        skipped_modules: Number of modules skipped after native compiler rejection.
        preflight_blockers: Number of module blockers detected before compilation.
        verified: Number of islands subjected to runtime routing verification.
        verify_failures: Number of runtime routing verification failures.
        semantic_tests_run: Whether target-project semantic tests were executed.
        semantic_test_failures: Number of failed target-project test gates.
        subprocess_verifications: Number of isolated package verification stages run.
        subprocess_verification_failures: Number of failed isolated package verifications.
        performance_status: Passed, failed, skipped, or unavailable benchmark status.
        profile_status: Whether profile evidence was collected or static fallback was used.
        profile_mapped_coverage: Fraction of samples mapped to configured project modules.
        profile_selected_hot_coverage: Fraction of mapped samples covered by selected candidates.
        profile_accepted_hot_coverage: Fraction of mapped samples covered by profitable candidates.
        execution_plans: Number of selected and rejected scheduler execution-plan candidates.
        execution_selected_plans: Candidates selected by profile-guided plan policy.
        execution_applied_plans: Plans staged into the promoted payload.
        execution_plan_trials: Semantic or performance trials run for staged plans.
        fusion_plans: Number of deterministic task-fusion plans emitted.
        fusion_eligible_plans: Plans that passed every static and dynamic safety gate.
        fusion_trials: Number of three-arm fusion profitability trials run.
        source_optimization_status: Overall source-optimization milestone status.
        source_optimization_plans: Number of source-level optimization plans discovered.
        source_optimization_trial_ready_assessments: Assessments ready for trial execution.
        source_optimization_trials: Number of source-optimization trials run.
        duration_seconds: Elapsed wall-clock duration in seconds.
    """

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
    profile_status: str
    profile_mapped_coverage: float
    profile_selected_hot_coverage: float
    profile_accepted_hot_coverage: float
    execution_plans: int
    execution_selected_plans: int
    execution_applied_plans: int
    execution_plan_trials: int
    fusion_plans: int
    fusion_eligible_plans: int
    fusion_trials: int
    source_optimization_status: str
    source_optimization_plans: int
    source_optimization_trial_ready_assessments: int
    source_optimization_trials: int
    duration_seconds: float


class CompilationProfileSamplingPolicyReport(TypedDict):
    """Stable profile sampling policy included in compile reports.

    Attributes:
        interval_ms: Sampling interval used by the baseline profiler.
        mode: Sampling mode described without target-project runtime values.
    """

    interval_ms: int
    mode: str


class CompilationProfilePassReport(TypedDict):
    """Child profiling pass evidence without benchmark stdout or stderr.

    Attributes:
        pass_kind: Profiling pass represented by the child process.
        command: Exact child argv used to launch the profiling bootstrap.
        returncode: Child process exit status.
        duration_seconds: Parent-observed elapsed duration.
    """

    pass_kind: str
    command: list[str]
    returncode: int
    duration_seconds: float


class CompilationProfileLifecycleReport(TypedDict):
    """Python lifecycle event counts serialized for profile evidence.

    Attributes:
        start: Python frame start events.
        return_: Python frame return events.
        yield_: Python yield events.
        resume: Python resume events.
        unwind: Python unwind events.
        throw: Python throw events.
    """

    start: int
    return_: int
    yield_: int
    resume: int
    unwind: int
    throw: int


class CompilationProfileTypeObservationReport(TypedDict):
    """Canonical argument-type count for one observed parameter.

    Attributes:
        parameter_name: Source parameter name observed in frame locals.
        type_path: Canonical runtime type identity as `module.qualname`.
        count: Number of calls whose parameter had this canonical type.
    """

    parameter_name: str
    type_path: str
    count: int


class CompilationProfileSignatureReport(TypedDict):
    """Canonical observed signature evidence for a profiled member.

    Attributes:
        parameters: Canonical parameter type identities and counts.
        count: Number of calls with this exact canonical signature.
    """

    parameters: list[CompilationProfileTypeObservationReport]
    count: int


class CompilationProfileMemberReport(TypedDict):
    """Profiled member evidence safe for persisted reports.

    Attributes:
        module: Importable module name resolved from profiling path evidence.
        qualname: Runtime qualified name observed in the profile.
        symbol: Stable static symbol implied by the member.
        samples: Statistical samples mapped to this member.
        coverage: Fraction of total workload samples represented by this member.
        call_count: Targeted type-observation calls observed for this member.
        invocation_count: Total target invocations, including calls after type observation capped.
        lifecycle: Python lifecycle event counts for this member.
        signatures: Canonical argument-type signatures observed for this member.
        polymorphic: Whether the member exceeded the retained signature budget.
        observation_capped: Whether targeted observation reached its call budget.
        completed_calls: Target invocations observed through return or unwind.
        max_active_calls: Maximum simultaneous active invocations for this member.
        pre_completion_suspensions: Yield events observed while an invocation was active.
        scheduler_overhead_samples: Nested scheduler samples attributed to active calls.
        scheduler_overhead_coverage: Fraction of workload samples attributed to scheduler work.
        immediate_result_ratio: Conservative fraction of completed calls without suspension.
        invocation_lower_bound: Conservative count from bounded project-wide observation.
        invocation_upper_bound: Upper count estimate from bounded project-wide observation.
    """

    module: str
    qualname: str
    symbol: str
    samples: int
    coverage: float
    call_count: int
    invocation_count: int
    lifecycle: CompilationProfileLifecycleReport
    signatures: list[CompilationProfileSignatureReport]
    polymorphic: bool
    observation_capped: bool
    completed_calls: int
    max_active_calls: int
    pre_completion_suspensions: int
    scheduler_overhead_samples: int
    scheduler_overhead_coverage: float
    immediate_result_ratio: float
    invocation_lower_bound: int
    invocation_upper_bound: int


class CompilationProfileCandidateDecisionReport(TypedDict):
    """Profile-to-static mapping and candidate policy decision.

    Attributes:
        symbol: Stable static symbol when mapping succeeded.
        module: Runtime module name observed in the profile.
        qualname: Runtime qualified name observed in the profile.
        samples: Statistical samples mapped to this member.
        coverage: Fraction of total workload samples represented by this member.
        scheduler_overhead_samples: Nested scheduler or library samples owned by the member.
        attributed_samples: Leaf plus nested samples used by candidate selection.
        attributed_coverage: Fraction of workload samples used by candidate selection.
        selected: Whether the member passed the candidate policy.
        reason: Deterministic selection or rejection reason.
        invocation_lower_bound: Conservative project-wide call-count bound.
        invocation_upper_bound: Project-wide call-count upper estimate.
        invocation_coverage: Lower-bound share of bounded mapped invocation events.
        selection_basis: Evidence that selected this member, or ``none`` when rejected.
    """

    symbol: str | None
    module: str
    qualname: str
    samples: int
    coverage: float
    scheduler_overhead_samples: int
    attributed_samples: int
    attributed_coverage: float
    selected: bool
    reason: str
    invocation_lower_bound: int
    invocation_upper_bound: int
    invocation_coverage: float
    selection_basis: str


class CompilationProfileInvocationSummaryReport(TypedDict):
    """Fixed-budget project-wide invocation summary metadata.

    Attributes:
        observed_events: Mapped project starts processed by the monitoring pass.
        event_limit: Maximum starts the pass may process.
        member_limit: Maximum callable identities retained as heavy hitters.
        capped: Whether the monitoring pass exhausted its event budget.
    """

    observed_events: int
    event_limit: int
    member_limit: int
    capped: bool


class CompilationProfileCallableIdentityReport(TypedDict):
    """Canonical scheduler callable identity observed at one spawn site.

    Attributes:
        identity: Runtime callable type path without values or representations.
        count: Calls attributed to this identity at the spawn site.
    """

    identity: str
    count: int


class CompilationProfileSpawnSiteReport(TypedDict):
    """Exact scheduler call-site invocation evidence.

    Attributes:
        id: Stable source-derived spawn-site identity.
        owner: Module-qualified callable containing the scheduler call.
        lineno: First source line of the scheduler call.
        col_offset: First source column of the scheduler call.
        end_lineno: Final source line of the scheduler call.
        end_col_offset: Final source column of the scheduler call.
        scheduler_method: Recognized scheduling method such as `create_task`.
        invocation_count: Calls observed at this exact source span.
        callable_identities: Canonical runtime scheduler identities and counts.
    """

    id: str
    owner: str
    lineno: int
    col_offset: int
    end_lineno: int | None
    end_col_offset: int | None
    scheduler_method: str
    invocation_count: int
    callable_identities: list[CompilationProfileCallableIdentityReport]


class CompilationProfileReport(TypedDict):
    """Profile-guided selection evidence for compile report schema v6.

    Attributes:
        status: Profile status describing dynamic evidence or static fallback.
        reason: Human-readable explanation for the current profile status.
        launch_kind: Supported launch shape used for child execution.
        sampling_policy: Stable sampling interval and sampling mode.
        total_samples: Statistical samples collected across the benchmark.
        mapped_project_samples: Samples mapped to configured project modules.
        mapped_coverage: Fraction of samples mapped to configured project modules.
        scheduler_overhead_samples: Nested scheduler samples attributed to project callers.
        scheduler_overhead_coverage: Fraction of samples represented by scheduler overhead.
        selected_hot_samples: Samples covered by selected candidates.
        selected_hot_coverage: Fraction of mapped samples covered by selected candidates.
        child_passes: Child-process profiling pass evidence.
        lifecycle: Aggregate Python lifecycle event counts.
        members: Profiled project members with sample and type evidence.
        spawn_sites: Exact scheduler call-site invocation evidence.
        broad_invocations: Fixed-budget project-wide invocation summary metadata.
        candidate_mapping_decisions: Static mapping and hotness policy decisions.
        selected_symbols: Static symbols accepted by the profile candidate policy.
    """

    status: str
    reason: str
    launch_kind: str
    sampling_policy: CompilationProfileSamplingPolicyReport
    total_samples: int
    mapped_project_samples: int
    mapped_coverage: float
    scheduler_overhead_samples: int
    scheduler_overhead_coverage: float
    selected_hot_samples: int
    selected_hot_coverage: float
    child_passes: list[CompilationProfilePassReport]
    lifecycle: CompilationProfileLifecycleReport
    members: list[CompilationProfileMemberReport]
    spawn_sites: list[CompilationProfileSpawnSiteReport]
    broad_invocations: CompilationProfileInvocationSummaryReport
    candidate_mapping_decisions: list[CompilationProfileCandidateDecisionReport]
    selected_symbols: list[str]


class CompilationBackendDecisionReport(TypedDict):
    """Normalized backend capability assessment retained in schema v3.

    Attributes:
        region_id: Stable typed-region identifier assessed by the backend.
        backend: Native compiler backend selected for this record.
        status: Supported, partial, or unsupported capability status.
        supported_members: Region members accepted by the backend.
        unsupported_members: Region members rejected by the backend.
        capabilities: Backend capabilities exercised by supported members.
        reasons: Deterministically ordered evidence supporting the decision.
        deterministic: Whether identical inputs must produce this assessment.
    """

    region_id: str
    backend: Backend
    status: str
    supported_members: list[str]
    unsupported_members: list[str]
    capabilities: list[str]
    reasons: list[str]
    deterministic: bool


class CompilationAcceptedVariantReport(TypedDict):
    """Compatibility view of compiled variants and legacy compiled regions.

    Attributes:
        region_id: Stable typed-region identifier.
        variant_id: Stable backend or specialization variant identifier.
        source_module: Importable source module name.
        backend: Native compiler backend selected for this record.
        cache_status: Whether compilation used, missed, or restored cache state.
        lowering_mode: Whether compilation owns the whole callable or native blocks.
        native_helpers: Private native helper names used by an outlined shell.
        symbols: Source bindings promised by the compiled variant.
        artifacts: Install-relative native artifact paths owned by the variant.
    """

    region_id: str
    variant_id: str
    source_module: str
    backend: Backend | None
    cache_status: CompileCacheStatus
    lowering_mode: LoweringMode
    native_helpers: list[str]
    symbols: list[str]
    artifacts: list[str]


class CompilationRejectedVariantReport(TypedDict):
    """Compatibility view of rejected source-clean module variants.

    Attributes:
        module: Importable source module skipped after native compiler rejection.
        reason: Representative native compiler diagnostic explaining the rejection.
    """

    module: str
    reason: str


class CompilationCandidateTrialReport(TypedDict):
    """One measured candidate-variant decision made during profitability selection.

    Attributes:
        id: Stable candidate trial identifier.
        region_id: Runtime allowlist variant evaluated by the trial.
        source_region_id: Backend-neutral source region represented by the trial.
        variant_id: Compatibility alias of `region_id`.
        backend: Native backend used by the candidate.
        lowering_mode: Whole-callable or outlined-block lowering mode.
        symbols: Profiled source bindings represented by the candidate.
        status: Accepted, rejected, failed-semantics, or unavailable status.
        reason: Evidence supporting the candidate decision.
        marginal_speedup: Speedup over the previously accepted set, when measured.
        fallback_reason: Preferred-backend failure that selected a fallback, when relevant.
        profile_samples: Mapped project samples attributed to this candidate.
        profile_coverage: Fraction of mapped project samples attributed to this candidate.
        accepted_hot_coverage: Mapped hot-path coverage retained after this decision.
        baseline_variants: Previously accepted variants used as the reference arm.
        trial_variants: Candidate combination used as the measured compiled arm.
        semantic_test_exit_code: Exit code from the candidate semantic command.
        semantic_test_duration_seconds: Candidate semantic-test wall-clock duration.
        benchmark_status: Marginal benchmark status, or not-run after semantic failure.
        baseline_median_seconds: Current accepted-arm median used by the marginal gate.
        candidate_median_seconds: Candidate-composition median used by the marginal gate.
        minimum_speedup: Marginal threshold applied to this candidate.
    """

    id: str
    region_id: str
    source_region_id: str
    variant_id: str
    backend: Backend
    lowering_mode: LoweringMode
    symbols: list[str]
    status: str
    reason: str
    marginal_speedup: float | None
    fallback_reason: str | None
    profile_samples: int
    profile_coverage: float
    accepted_hot_coverage: float
    baseline_variants: list[str]
    trial_variants: list[str]
    semantic_test_exit_code: int | None
    semantic_test_duration_seconds: float | None
    benchmark_status: str
    baseline_median_seconds: float | None
    candidate_median_seconds: float | None
    minimum_speedup: float | None


class CompilationFusionGateRejectionReport(TypedDict):
    """One coded reason a task-fusion plan cannot be trialed safely.

    Attributes:
        code: Stable machine-readable safety-gate identifier.
        reason: Plain-language explanation of the rejected condition.
    """

    code: str
    reason: str


class CompilationExecutionPlanNodeReport(TypedDict):
    """One callable or transport role in a scheduler execution plan.

    Attributes:
        id: Stable node identity within the plan topology.
        symbol: Module-qualified source symbol, when the node represents a callable.
        role: Orchestrator, producer, consumer, reducer, transport, or support role.
        lineno: Relevant one-based source line.
    """

    id: str
    symbol: str | None
    role: str
    lineno: int


class CompilationExecutionPlanEdgeReport(TypedDict):
    """One directed scheduler or transport relation in an execution plan.

    Attributes:
        src: Source plan-node identity.
        dst: Destination plan-node identity.
        kind: Spawn, production, delivery, reduction, or report relation.
        transport: Private transport identity carried by the relation.
        lineno: One-based source line establishing the relation.
    """

    src: str
    dst: str
    kind: str
    transport: str | None
    lineno: int


class CompilationExecutionPlanGuardReport(TypedDict):
    """One runtime or lowering invariant required by an execution plan.

    Attributes:
        kind: Scheduler, transport, topology, or semantic guard category.
        expression: Stable predicate or invariant identifier.
        message: Plain-language explanation of the required invariant.
    """

    kind: str
    expression: str
    message: str


class CompilationExecutionPlanRejectionReport(TypedDict):
    """One coded reason a scheduler plan remains report-only.

    Attributes:
        code: Stable machine-readable rejection category.
        reason: Plain-language rejection explanation.
    """

    code: str
    reason: str


class CompilationExecutionPlanReport(TypedDict):
    """Selected or rejected scheduler execution-plan evidence.

    Attributes:
        id: Content-derived plan or rejection identity.
        status: Selected or rejected discovery status.
        source_module: Importable module containing the orchestration site.
        owner: Module-qualified orchestration callable.
        dialect: Recognized scheduler dialect, when available.
        lowering_version: Dialect lowering version included in the plan identity.
        source_hash: Digest of every source member required by the plan.
        source_hashes: Complete per-module source digests covered by source_hash.
        source_members: Exact module-qualified declarations covered by source_hash.
        callsite_fingerprint: Digest of scheduler coordinates and callees.
        topology_fingerprint: Digest of plan nodes, edges, and transport relations.
        completion_transport: Private result-delivery transport identity.
        consumer: Module-qualified result consumer.
        reducer: Module-qualified reduction owner.
        transport_capacity: Statically known dialect-defined delivery capacity.
        ordering_policy: Result ordering policy a lowering must preserve.
        task_ownership: Static proof category for task-handle ownership and joining.
        observed_invocations: Maximum exact invocation count among plan spawn sites.
        lifecycle_starts: Child coroutine starts attributed to the plan.
        lifecycle_share: Fraction of mapped hot spawn activity represented by the plan.
        guarded_callable_identities: Canonical scheduler callables required at runtime.
        hotness: Dynamic ranking value excluded from plan identity.
        nodes: Deterministic plan topology nodes.
        edges: Deterministic directed topology relations.
        guards: Runtime and lowering invariants required by the plan.
        rejections: Coded reasons preventing selection or application.
    """

    id: str
    status: str
    source_module: str
    owner: str
    dialect: str | None
    lowering_version: str | None
    source_hash: str | None
    source_hashes: dict[str, str]
    source_members: list[str]
    callsite_fingerprint: str | None
    topology_fingerprint: str | None
    completion_transport: str | None
    consumer: str | None
    reducer: str | None
    transport_capacity: int | None
    ordering_policy: str | None
    task_ownership: str | None
    observed_invocations: int
    lifecycle_starts: int
    lifecycle_share: float
    guarded_callable_identities: list[str]
    hotness: int
    nodes: list[CompilationExecutionPlanNodeReport]
    edges: list[CompilationExecutionPlanEdgeReport]
    guards: list[CompilationExecutionPlanGuardReport]
    rejections: list[CompilationExecutionPlanRejectionReport]


class CompilationExecutionPlanDiagnosticReport(TypedDict):
    """Normalized execution-plan trial diagnostic.

    Attributes:
        code: Stable diagnostic category.
        severity: Error, warning, or note severity.
        message: Plain-language diagnostic summary.
        details: Deterministic supporting lines without target values.
    """

    code: str
    severity: str
    message: str
    details: list[str]


class CompilationExecutionPlanPayloadFileReport(TypedDict):
    """One payload change made by a disposable execution-plan candidate.

    Attributes:
        install_path: POSIX path below the unpacked wheel payload root.
        before_hash: Digest before staging, or `None` for a generated file.
        after_hash: Digest after staging.
        role: Backend-defined purpose of the changed file.
    """

    install_path: str
    before_hash: str | None
    after_hash: str
    role: str


class CompilationExecutionPlanTrialReport(TypedDict):
    """Semantic or performance evidence for one staged execution plan.

    Attributes:
        plan_id: Content-derived plan identity under trial.
        status: Accepted, rejected, failed-semantics, or unavailable result.
        command: Exact argv executed with `shell=False`.
        exit_code: Child exit status, when a command ran.
        duration_seconds: Parent-observed command duration.
        diagnostics: Normalized failure or decision evidence.
        backend: Execution-plan backend used for staging.
        reason: Plain-language acceptance or rejection reason.
        benchmark_command: Exact argv used for marginal measurement.
        benchmark_status: Marginal benchmark result, or `not-run`.
        minimum_speedup: Required speedup over the current accepted payload.
        minimum_overall_speedup: Required speedup over the interpreted baseline.
        baseline_median_seconds: Interpreted-baseline median duration.
        unplanned_median_seconds: Current accepted-payload median duration.
        planned_median_seconds: Planned-payload median duration.
        marginal_speedup: Unplanned-payload median divided by planned-payload median.
        overall_speedup: Baseline-payload median divided by planned-payload median.
        cache_status: Whether staging restored or generated the plan payload.
        payload_files: Validated staged payload changes.
    """

    plan_id: str
    status: str
    command: list[str]
    exit_code: int | None
    duration_seconds: float | None
    diagnostics: list[CompilationExecutionPlanDiagnosticReport]
    backend: str | None
    reason: str | None
    benchmark_command: list[str]
    benchmark_status: str
    minimum_speedup: float | None
    minimum_overall_speedup: float | None
    baseline_median_seconds: float | None
    unplanned_median_seconds: float | None
    planned_median_seconds: float | None
    marginal_speedup: float | None
    overall_speedup: float | None
    cache_status: str
    payload_files: list[CompilationExecutionPlanPayloadFileReport]


class CompilationFusionPlanReport(TypedDict):
    """Deterministic task-fusion plan and its static/dynamic gate evidence.

    Attributes:
        id: Content-derived plan identity stable across profile count changes.
        source_hash: Digest of the spawn site and spawned coroutine closure.
        root: Profile-selected hot root that reaches the spawn site.
        caller: Callable containing the recognized task-spawn expression.
        callee: Same-module spawned coroutine when resolution succeeded.
        spawn_api: Source-level scheduling API expression.
        lineno: One-based first line of the spawn call.
        end_lineno: One-based final line of the spawn call.
        col_offset: Zero-based first column of the spawn call.
        end_col_offset: Zero-based final column of the spawn call, when available.
        eligible: Whether every conservative fusion gate passed.
        observed_calls: Targeted callee invocations represented by profile evidence.
        completed_calls: Invocations observed through return or unwind.
        max_active_calls: Maximum overlapping active invocations.
        pre_completion_suspensions: Yield events observed before invocation completion.
        observed_signatures: Number of retained canonical call signatures.
        observation_capped: Whether targeted type observation reached its budget.
        rejections: Ordered coded reasons preventing a fusion trial.
    """

    id: str
    source_hash: str
    root: str
    caller: str
    callee: str | None
    spawn_api: str
    lineno: int
    end_lineno: int
    col_offset: int
    end_col_offset: int | None
    eligible: bool
    observed_calls: int
    completed_calls: int
    max_active_calls: int
    pre_completion_suspensions: int
    observed_signatures: int
    observation_capped: bool
    rejections: list[CompilationFusionGateRejectionReport]


SourceOptimizationReportStatus = Literal[
    "unbenchmarked",
    "report-only",
    "rejected",
    "accepted",
    "not-profitable",
    "not-applied",
    "applied",
    "conflicted",
    "rolled-back",
    "stale-source",
    "failed",
    "unavailable",
]


class CompilationSourceOptimizationAccessSiteReport(TypedDict):
    """Static source access evidence used by a source-optimization plan.

    Attributes:
        path: POSIX source path containing the access site.
        symbol: Stable owner symbol, when resolution succeeded.
        kind: Access operation observed at the site.
        lineno: One-based source line for the access.
        expression: Stable expression, attribute, or transport name.
        hazards: Conservative hazards attached to the site.
    """

    path: str
    symbol: str | None
    kind: str
    lineno: int
    expression: str
    hazards: list[str]


class CompilationSourceOptimizationIdentityReport(TypedDict):
    """Static source-optimization identity inputs retained for cache compatibility.

    Attributes:
        execution_plan_id: Stable execution-plan identifier that produced the plan.
        source_hashes: Per-source-file hashes covered by the plan.
        topology_fingerprint: Stable execution-plan topology digest.
        dialect: Scheduler dialect whose semantics the transformation preserves.
        lowering_version: Source-lowering version that changes patch semantics.
        python_abi: Python ABI compatibility boundary.
        transformation_versions: Per-transformation versions included in identity.
    """

    execution_plan_id: str
    source_hashes: dict[str, str]
    topology_fingerprint: str
    dialect: str
    lowering_version: str
    python_abi: str
    transformation_versions: dict[SourceTransformationKind, str]


class CompilationSourceOptimizationStepReport(TypedDict):
    """One ordered source transformation in a source-optimization plan.

    Attributes:
        id: Stable step identity derived from kind, version, and source symbol.
        kind: Transformation family applied by this step.
        version: Step-specific transformation version.
        source_symbol: Primary symbol read or rewritten by the step.
        target_symbol: Generated or rewritten symbol produced by the step.
        access_sites: Static access evidence used by the step.
        semantic_boundary: Invariant preserved by the step.
        description: Human-readable report text for the transformation.
    """

    id: str
    kind: SourceTransformationKind
    version: str
    source_symbol: str
    target_symbol: str | None
    access_sites: list[CompilationSourceOptimizationAccessSiteReport]
    semantic_boundary: str
    description: str


class CompilationSourceOptimizationPlanReport(TypedDict):
    """Source-level optimization plan derived from a scheduler execution plan.

    Attributes:
        id: Stable source-optimization plan identifier.
        identity: Static identity inputs that define patch compatibility.
        source: POSIX source path owning the entrypoint orchestration site.
        owner: Orchestration owner symbol.
        worker: Worker callable transformed by the plan.
        consumer: Result consumer callable, when distinct.
        reducer: Reduction callable, when relevant.
        transport: Private transport expression or name used by the plan.
        access_sites: Static source accesses used to prove privacy and semantics.
        entrypoint: Callable used to enter the optimized path.
        steps: Ordered transformations that make up the source patch.
        semantic_boundaries: Named invariants the plan preserves.
        transport_capacity: Statically known private transport capacity, when available.
    """

    id: str
    identity: CompilationSourceOptimizationIdentityReport
    source: str
    owner: str
    worker: str
    consumer: str | None
    reducer: str | None
    transport: str
    access_sites: list[CompilationSourceOptimizationAccessSiteReport]
    entrypoint: str
    steps: list[CompilationSourceOptimizationStepReport]
    semantic_boundaries: list[str]
    transport_capacity: int | None


class CompilationSourceOptimizationCallableEvidenceReport(TypedDict):
    """Static and runtime callable evidence for source optimization assessment.

    Attributes:
        symbol: Callable represented by this evidence.
        static_role: Plan-facing callable role.
        observed_invocations: Runtime invocation count used for assessment.
        completed_calls: Invocations observed through normal return or unwind.
        static_suspension_points: Suspension syntax found in the declaration.
        observed_suspensions: Runtime pre-completion suspension events.
        immediate_result_ratio: Conservative fraction of completed calls without suspension.
        median_seconds: Optional median runtime attributed to the callable.
        hot_share: Fraction of mapped runtime attributed to the callable.
        scheduler_overhead_samples: Nested scheduler samples attributed to active calls.
        task_introspection: Static task identity or metadata reads.
        cancellation: Static cancellation API references.
        context_mutation: Static context-local mutation evidence.
        unknown_dynamic_calls: Calls whose runtime target was not proven statically.
        hazards: Conservative static hazards associated with the callable.
    """

    symbol: str
    static_role: str
    observed_invocations: int
    completed_calls: int
    static_suspension_points: int
    observed_suspensions: int
    immediate_result_ratio: float
    median_seconds: float | None
    hot_share: float
    scheduler_overhead_samples: int
    task_introspection: list[str]
    cancellation: list[str]
    context_mutation: list[str]
    unknown_dynamic_calls: list[str]
    hazards: list[str]


class CompilationSourceOptimizationAssessmentReport(TypedDict):
    """Capability and profitability assessment for one source-optimization plan.

    Attributes:
        plan_id: Stable source-optimization plan identifier.
        status: Assessment outcome before trial execution.
        minimum_speedup: Required speedup for profitability.
        work_items: Static callable identities expected to benefit.
        observed_work_items: Runtime work-item count represented by the plan.
        immediate_result_ratio: Conservative fraction of work that did not suspend.
        attributed_hot_share: Fraction of observed time attributed to the plan.
        scheduler_overhead_samples: Nested scheduler samples attributed to the plan.
        scheduler_overhead_share: Fraction of total samples represented by overhead.
        scheduler_overhead_evidence: Normalized scheduler overhead evidence.
        callable_evidence: Per-callable assessment evidence.
        rejections: Deterministic rejection reasons or guarded caveats.
        headroom_speedup: Optional measured ceiling speedup.
    """

    plan_id: str
    status: str
    minimum_speedup: float
    work_items: list[str]
    observed_work_items: int
    immediate_result_ratio: float
    attributed_hot_share: float
    scheduler_overhead_samples: int
    scheduler_overhead_share: float
    scheduler_overhead_evidence: list[str]
    callable_evidence: list[CompilationSourceOptimizationCallableEvidenceReport]
    rejections: list[str]
    headroom_speedup: float | None


class CompilationSourceOptimizationEditReport(TypedDict):
    """One source edit represented by a source-optimization trial patch.

    Attributes:
        path: POSIX source path changed by the edit.
        before_hash: Digest of source before the edit, or `None` for additions.
        after_hash: Digest of source after the edit.
        summary: Stable summary of the generated change.
        touched_symbols: Symbols whose definitions or call sites were edited.
        transformation_id: Stable transformation step that produced the edit.
        start_line: One-based first changed source line, when known.
        end_line: One-based final changed source line, when known.
    """

    path: str
    before_hash: str | None
    after_hash: str
    summary: str
    touched_symbols: list[str]
    transformation_id: str | None
    start_line: int | None
    end_line: int | None


class CompilationSourceOptimizationTrialReport(TypedDict):
    """Semantic and benchmark evidence for one source-optimization trial.

    Attributes:
        plan_id: Stable source-optimization plan identifier.
        status: Trial outcome after commands and benchmarks run.
        semantic_command: Command used to validate patched source behavior.
        benchmark_command: Command used to measure source and wheel performance.
        baseline_median_seconds: Median unoptimized source runtime.
        current_median_seconds: Current accepted-candidate median before this trial.
        source_median_seconds: Median optimized source runtime.
        wheel_median_seconds: Median optimized wheel runtime.
        source_speedup: Baseline median divided by optimized source median.
        wheel_speedup: Baseline median divided by optimized wheel median.
        patch_path: Filesystem path to the generated patch, when one exists.
        source_edits: Source edits represented by the patch.
        application_status: Whether the patch was applied to the working tree.
        diagnostics: Semantic, benchmark, or application diagnostics.
        candidate_id: Stable candidate-combination identity.
        transformation_ids: Ordered transformation steps enabled for this candidate.
        reason: Plain-language acceptance or rejection reason.
        semantic_exit_code: Semantic command exit status, when executed.
        semantic_duration_seconds: Parent-observed semantic command duration.
        residual_profile: Fresh transformed-candidate profile collected before later selection.
    """

    plan_id: str
    status: str
    semantic_command: list[str]
    benchmark_command: list[str]
    baseline_median_seconds: float | None
    current_median_seconds: float | None
    source_median_seconds: float | None
    wheel_median_seconds: float | None
    source_speedup: float | None
    wheel_speedup: float | None
    patch_path: str | None
    source_edits: list[CompilationSourceOptimizationEditReport]
    application_status: SourceOptimizationApplicationStatus
    diagnostics: list[str]
    candidate_id: str
    transformation_ids: list[str]
    reason: str
    semantic_exit_code: int | None
    semantic_duration_seconds: float | None
    residual_profile: CompilationProfileReport | None


class CompilationSourceOptimizationReport(TypedDict):
    """Top-level source-optimization section retained by compile schema v6.

    Attributes:
        status: Overall source-optimization milestone status.
        minimum_speedup: Maximum required speedup among assessments and trials.
        headroom_speedup: Best reported ceiling speedup, when available.
        attributed_hot_share: Total assessed hot share capped to the full workload.
        plans: Source-optimization plans discovered for report-only review.
        assessments: Capability and profitability assessments for plans.
        trials: Semantic and benchmark source-optimization trials.
        patch_path: First emitted patch path, when a trial created one.
        application_status: Aggregate patch application status.
    """

    status: SourceOptimizationReportStatus
    minimum_speedup: float
    headroom_speedup: float | None
    attributed_hot_share: float
    plans: list[CompilationSourceOptimizationPlanReport]
    assessments: list[CompilationSourceOptimizationAssessmentReport]
    trials: list[CompilationSourceOptimizationTrialReport]
    patch_path: str | None
    application_status: SourceOptimizationApplicationStatus


class CompilationSuspensionPlanReport(TypedDict):
    """Suspension evidence and lowering mode selected for one callable.

    Attributes:
        id: Stable plan identifier derived from region and member identities.
        region_id: Source region containing the callable.
        member: Stable source callable identity.
        execution_kind: Coroutine, generator, or async-generator shape.
        lowering_mode: Whole-callable or interpreted mode used by this compile.
        native_helpers: Private native helper names for outlined-block lowering.
        points: Ordered syntax-level suspension boundaries.
        blocks: Synchronous block plans with liveness and rejection evidence.
        rejections: Member-level reasons preventing safe block extraction.
        reason: Evidence supporting the selected lowering mode.
    """

    id: str
    region_id: str
    member: str
    execution_kind: ExecutionKind
    lowering_mode: str
    native_helpers: list[str]
    points: list[SuspensionPointReport]
    blocks: list[CompilationSuspensionBlockReport]
    rejections: list[CompilationSuspensionRejectionReport]
    reason: str


class CompilationSuspensionRejectionReport(TypedDict):
    """One stable reason a callable or synchronous block remains interpreted.

    Attributes:
        code: Stable machine-readable rejection category.
        message: Human-readable conservative planning explanation.
        lineno: One-based source line for local evidence, when available.
    """

    code: str
    message: str
    lineno: int | None


class CompilationSuspensionBlockReport(TypedDict):
    """One synchronous block considered between suspension boundaries.

    Attributes:
        id: Stable content-derived block identifier.
        start_lineno: One-based first source line.
        start_col_offset: Zero-based first source byte offset.
        end_lineno: One-based final source line.
        end_col_offset: Zero-based final source byte offset.
        live_ins: Values passed explicitly from the Python shell.
        live_outs: Values restored into the Python shell.
        late_bound_globals: Runtime global dependencies passed explicitly.
        receiver_dependencies: Receiver attributes read by the block.
        loop_count: Synchronous loops retained in the block.
        operation_count: Conservative native-work signal.
        eligible: Whether the block passed every extraction gate.
        rejections: Block-local reasons for retaining interpreted execution.
    """

    id: str
    start_lineno: int
    start_col_offset: int
    end_lineno: int
    end_col_offset: int
    live_ins: list[str]
    live_outs: list[str]
    late_bound_globals: list[str]
    receiver_dependencies: list[str]
    loop_count: int
    operation_count: int
    eligible: bool
    rejections: list[CompilationSuspensionRejectionReport]


class CompilationBuildReport(TypedDict):
    """Serialized mypyc build command, diagnostics, and produced artifacts.

    Attributes:
        success: Whether the represented operation completed successfully.
        command: Normalized command argument vector.
        duration_seconds: Elapsed wall-clock duration in seconds.
        stdout: Captured child process standard output.
        stderr: Captured child process standard error.
        cache_status: Whether compilation used, missed, or partially restored cache state.
        phase_timings: Measured native compiler subphases.
        artifacts: Project-relative paths to primary native artifacts.
        support_artifacts: Native support files not owned by one compiled region.
    """

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
    """Serialized timing evidence for one native compilation phase.

    Attributes:
        name: Stable compiler phase name.
        duration_seconds: Elapsed wall-clock duration in seconds.
        detail: Optional human-readable context for the measured phase.
    """

    name: str
    duration_seconds: float
    detail: str | None


class CompilationCleanupReport(TypedDict):
    """Paths removed or intentionally kept after a build or package operation.

    Attributes:
        removed: Paths removed during cleanup.
        kept: Paths intentionally retained after cleanup.
    """

    removed: list[str]
    kept: list[str]


class CompilationSkippedModuleReport(TypedDict):
    """Source-clean module skipped because its island failed compilation.

    Attributes:
        module: Importable source module skipped after native compilation.
        reason: Representative native compiler diagnostic explaining the skip.
    """

    module: str
    reason: str


class CompilationPreflightBlockerReport(TypedDict):
    """Module-level blocker that prevented a source-clean module build attempt.

    Attributes:
        module: Importable source module blocked before native compilation.
        path: Project-relative path of the blocked source module.
        line: One-based diagnostic or blocker line.
        code: Stable machine-readable diagnostic or blocker code.
        message: Human-readable diagnostic or blocker explanation.
    """

    module: str
    path: str
    line: int | None
    code: str
    message: str


class CompilationTestReport(TypedDict):
    """Target-project semantic test command and process exit status.

    Attributes:
        command: Normalized command argument vector.
        exit_code: Child process exit code.
        success: Whether the represented operation completed successfully.
    """

    command: list[str]
    exit_code: int
    success: bool


class CompilationVerifySymbolReport(TypedDict):
    """Runtime verification result for one exported symbol rebound by a shim.

    Attributes:
        symbol: Stable symbol identifier associated with this record.
        rebound: Whether the source symbol resolves to the promised compiled callable.
    """

    symbol: str
    rebound: bool


class CompilationVerifyReport(TypedDict):
    """Runtime routing state for one compiled or pure-Python sidecar.

    Attributes:
        active: Whether the managed runtime shim is active.
        compiled: Whether routing resolved to a native extension.
        origin: Resolved module origin or specialization evidence source.
        symbols: Per-symbol routing and rebinding results.
        error: User-facing failure text, or `None` on success.
    """

    active: bool
    compiled: bool
    origin: str | None
    symbols: list[CompilationVerifySymbolReport]
    error: str | None


class CompilationIslandReport(TypedDict):
    """Compilation report section for one enabled source island.

    Attributes:
        source_module: Importable source module name.
        source_path: Filesystem path of the source module or prepared source.
        generated_module: Importable module containing generated code.
        symbols: Exported symbol names promised by the legacy island.
        artifacts: Project-relative native artifacts mapped to the island.
        verification: Runtime routing result for the island, when run.
    """

    source_module: str
    source_path: str
    generated_module: str
    symbols: list[str]
    artifacts: list[str]
    verification: CompilationVerifyReport | None


class CompilationCompiledBindingReport(TypedDict):
    """One guarded function or descriptor promised by a compiled region.

    Attributes:
        source: Stable ID of the public source binding.
        compiled_name: Backend-generated attribute name containing the compiled callable.
        kind: Module, class, or descriptor-aware binding kind.
        owner_class: Source owner class for a method binding, when applicable.
        target_owner_class: Concrete runtime owner class for a specialized binding.
        execution_kind: Synchronous, generator, coroutine, async-generator, or class shape.
        required: Whether absence of the compiled binding is a verification failure.
        guards: Runtime type guards required before selecting this binding or specialization.
    """

    source: str
    compiled_name: str
    kind: BindingKind
    owner_class: str | None
    target_owner_class: str | None
    execution_kind: ExecutionKind
    required: bool
    guards: list[RuntimeTypeGuardReport]


class CompilationCompiledRegionReport(TypedDict):
    """Backend, binding, and artifact evidence for one successful region.

    Attributes:
        id: Stable source typed-region ID.
        variant_id: Stable backend/specialization variant identifier.
        source_module: Importable source module name.
        backend: Native compiler backend selected for this record.
        cache_status: Whether compilation used, missed, or partially restored cache state.
        lowering_mode: Whether compilation owns the whole callable or native blocks.
        native_helpers: Private native helper names used by an outlined shell.
        bindings: Source bindings promised by the compiled region or variant.
        artifacts: Install-relative native artifact paths owned by the variant.
    """

    id: str
    variant_id: str
    source_module: str
    backend: Backend | None
    cache_status: CompileCacheStatus
    lowering_mode: LoweringMode
    native_helpers: list[str]
    bindings: list[CompilationCompiledBindingReport]
    artifacts: list[str]


class CompilationVerificationStepReport(TypedDict):
    """Fresh-interpreter verification evidence for a payload or final wheel.

    Attributes:
        stage: Package verification stage represented by the result.
        target: Lowering target or package verification target.
        command: Normalized command argument vector.
        success: Whether the represented operation completed successfully.
        exit_code: Child process exit code.
        duration_seconds: Elapsed wall-clock duration in seconds.
        stdout: Captured child process standard output.
        stderr: Captured child process standard error.
    """

    stage: str
    target: str
    command: list[str]
    success: bool
    exit_code: int
    duration_seconds: float
    stdout: str
    stderr: str


class CompilationCommandRunReport(TypedDict):
    """One baseline or compiled semantic-test or benchmark subprocess run.

    Attributes:
        command: Normalized command argument vector.
        mode: Compilation or runtime mode represented by this record.
        payload_root: Unpacked wheel payload used for the command.
        returncode: Child process return code.
        success: Whether the represented operation completed successfully.
        duration_seconds: Elapsed wall-clock duration in seconds.
        stdout: Captured child process standard output.
        stderr: Captured child process standard error.
    """

    command: list[str]
    mode: str
    payload_root: str
    returncode: int
    success: bool
    duration_seconds: float
    stdout: str
    stderr: str


class CompilationFusionArmRunReport(TypedDict):
    """One semantic or benchmark subprocess tagged with its fusion arm.

    Attributes:
        arm: Baseline, unfused, or fused research role.
        run: Ordinary command evidence captured for that role.
    """

    arm: str
    run: CompilationCommandRunReport


class CompilationFusionTrialReport(TypedDict):
    """Three-arm task-fusion semantic and profitability evidence.

    Attributes:
        plan_id: Stable fusion plan identity represented by the trial.
        status: Passed, not-profitable, invalid, or unavailable decision.
        reason: Concrete semantic, timing, or prerequisite explanation.
        baseline_median_seconds: Median interpreted duration, when stable.
        unfused_median_seconds: Median safe compiled duration, when stable.
        fused_median_seconds: Median experimental fused duration, when stable.
        baseline_over_unfused: Baseline divided by unfused median.
        baseline_over_fused: Baseline divided by fused median.
        unfused_over_fused: Unfused divided by fused median.
        semantic_runs: One semantic command result per arm.
        warmups: Unmeasured rotated three-arm benchmark runs.
        samples: Measured rotated three-arm benchmark runs.
    """

    plan_id: str
    status: str
    reason: str
    baseline_median_seconds: float | None
    unfused_median_seconds: float | None
    fused_median_seconds: float | None
    baseline_over_unfused: float | None
    baseline_over_fused: float | None
    unfused_over_fused: float | None
    semantic_runs: list[CompilationFusionArmRunReport]
    warmups: list[CompilationFusionArmRunReport]
    samples: list[CompilationFusionArmRunReport]


class CompilationPerformanceReport(TypedDict):
    """Measured profitability evidence or an explicit unbenchmarked status.

    Attributes:
        status: Supported, partial, unsupported, or quality-gate status.
        reason: Concrete measurement or execution evidence supporting the status.
        minimum_speedup: Smallest acceptable compiled-to-baseline speedup ratio.
        baseline_median_seconds: Median elapsed time for baseline samples, when measured.
        compiled_median_seconds: Median elapsed time for compiled samples, when measured.
        speedup: Baseline-to-compiled median ratio, when measured.
        warmups: Unmeasured benchmark warmup command evidence.
        samples: Measured benchmark command evidence.
    """

    status: str
    reason: str
    minimum_speedup: float
    baseline_median_seconds: float | None
    compiled_median_seconds: float | None
    speedup: float | None
    warmups: list[CompilationCommandRunReport]
    samples: list[CompilationCommandRunReport]


class CompilationOptimizationPolicyReport(TypedDict):
    """Numerical policy snapshot used for every compile profitability decision.

    Attributes:
        version: Policy schema version independent of report schema versions.
        stability_floor_seconds: Minimum credible median for each compared arm.
        profile_guided_minimum_marginal_speedup: Hot-region incremental gate.
        specialized_minimum_marginal_speedup: Scalar, call-chain, buffer, and plan gate.
        final_minimum_speedup: Configured final wheel promotion threshold.
        hard_benchmark_minimum_speedup: Representative family and source-patch floor.
    """

    version: int
    stability_floor_seconds: float
    profile_guided_minimum_marginal_speedup: float
    specialized_minimum_marginal_speedup: float
    final_minimum_speedup: float
    hard_benchmark_minimum_speedup: float


class CompilationStageMedianReport(TypedDict):
    """Normalized median comparison for one optimization stage or final gate.

    Attributes:
        stage: Stable stage label including the variant or plan identity when applicable.
        status: Profitability or execution outcome reported by the owning optimizer.
        baseline_median_seconds: Median duration before applying this stage.
        candidate_median_seconds: Median duration after applying this stage.
        speedup: Baseline median divided by candidate median.
        minimum_speedup: Threshold this stage had to meet for promotion.
    """

    stage: str
    status: str
    baseline_median_seconds: float
    candidate_median_seconds: float
    speedup: float
    minimum_speedup: float


class CompilationCacheDecisionReport(TypedDict):
    """Per-variant cache outcome and whether it shared physical Cython startup.

    Attributes:
        variant_id: Stable compiled variant or typed-region identity.
        backend: Backend selected for the variant, when compilation was attempted.
        status: Success-cache, decision-cache, or cold-compile outcome.
        batched: Whether the variant shared one physical Cython invocation.
    """

    variant_id: str
    backend: Backend | None
    status: CompileCacheStatus
    batched: bool


class CompilationFinalCompositionReport(TypedDict):
    """Authoritative optimization layers present in the promoted wheel.

    Attributes:
        source_plan_ids: Accepted source-optimization plans materialized in the payload.
        transformation_ids: Individual accepted source transformation identities.
        native_variant_ids: Native dispatch variants present in the final payload.
        execution_plan_ids: Accepted async execution plans present in the payload.
        artifacts: Install-relative native artifact paths included in the wheel.
        wheel_path: Promoted wheel path, or `None` when no payload passed all gates.
        retained_previous_arm: Whether a rejected later stage preserved an earlier arm.
    """

    source_plan_ids: list[str]
    transformation_ids: list[str]
    native_variant_ids: list[str]
    execution_plan_ids: list[str]
    artifacts: list[str]
    wheel_path: str | None
    retained_previous_arm: bool


@dataclass(frozen=True, slots=True)
class _StageMedianInput:
    stage: str
    status: str
    baseline: float | None
    candidate: float | None
    speedup: float | None
    minimum: float | None


@dataclass(frozen=True, slots=True)
class _StageMedianEvidence:
    candidate_trials: list[CompilationCandidateTrialReport]
    execution_plan_trials: list[CompilationExecutionPlanTrialReport]
    fusion_trials: list[CompilationFusionTrialReport]
    source_optimization: CompilationSourceOptimizationReport
    composition_performance: CompilationPerformanceReport | None
    performance: CompilationPerformanceReport


@dataclass(frozen=True, slots=True)
class _FinalCompositionInput:
    root: Path
    wheel_path: Path | None
    accepted_variants: list[CompilationAcceptedVariantReport]
    applied_execution_plans: tuple[str, ...]
    source_optimization: CompilationSourceOptimizationReport
    artifact_records: tuple[ArtifactRecord, ...]
    build: CompileAttempt


class CompilationReport(TypedDict):
    """Top-level stable JSON report for build and source-clean compile commands.

    Attributes:
        version: Schema or cache format version.
        tool: Tool identifier that produced the report.
        operation: Build or compile operation represented by the report.
        mode: Compilation or runtime mode represented by this record.
        project_root: Root directory of the target Python project.
        module_filter: Optional module restriction applied to compilation.
        success: Whether the represented operation completed successfully.
        wheel_path: Source-clean wheel path, when produced.
        summary: Aggregate counts and status for the report.
        build: Captured native build evidence.
        tests: Optional target-project semantic test result.
        test_results: Target-project command evidence used by quality gates.
        verification_steps: Isolated wheel and payload verification evidence.
        performance: Paired performance-gate evidence.
        composition_performance: Direct accepted-arm versus composed-payload evidence.
        optimization_policy: Numerical policy applied by every profitability gate.
        stage_medians: Normalized ordered comparisons across optimization stages.
        cache_decisions: Per-native-variant cache and batching outcomes.
        final_composition: Optimization layers actually present in the promoted wheel.
        profile: Profile-guided selection evidence or explicit static fallback.
        candidate_trials: Candidate variants evaluated by profitability selection.
        execution_plans: Selected and rejected scheduler execution-plan candidates.
        applied_execution_plans: Plan IDs staged into the promoted payload.
        execution_plan_trials: Semantic and performance evidence for staged plans.
        fusion_plans: Report-only task-fusion safety plans rooted in profiled hot paths.
        fusion_trials: Three-arm fusion trials run for generated eligible variants.
        source_optimization: Report-only source-optimization plans and trial evidence.
        suspension_plans: Per-callable suspension evidence and lowering decisions.
        backend_decisions: Normalized backend capability assessments.
        accepted_variants: Compatibility view of accepted compiled variants.
        rejected_variants: Compatibility view of rejected source-clean module variants.
        cleanup: Paths removed or retained after compilation.
        skipped_modules: Source modules skipped after native compiler rejection.
        preflight_blockers: Module blockers detected before native compilation.
        native_readiness: Post-generation native-readiness evidence.
        typed_regions: Backend-neutral typed regions discovered or reported.
        compiled_regions: Typed regions successfully compiled into the wheel.
        islands: Enabled islands included in the operation or report.
    """

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
    composition_performance: CompilationPerformanceReport | None
    optimization_policy: CompilationOptimizationPolicyReport
    stage_medians: list[CompilationStageMedianReport]
    cache_decisions: list[CompilationCacheDecisionReport]
    final_composition: CompilationFinalCompositionReport
    profile: CompilationProfileReport
    candidate_trials: list[CompilationCandidateTrialReport]
    execution_plans: list[CompilationExecutionPlanReport]
    applied_execution_plans: list[str]
    execution_plan_trials: list[CompilationExecutionPlanTrialReport]
    fusion_plans: list[CompilationFusionPlanReport]
    fusion_trials: list[CompilationFusionTrialReport]
    source_optimization: CompilationSourceOptimizationReport
    suspension_plans: list[CompilationSuspensionPlanReport]
    backend_decisions: list[CompilationBackendDecisionReport]
    accepted_variants: list[CompilationAcceptedVariantReport]
    rejected_variants: list[CompilationRejectedVariantReport]
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

    Attributes:
        module: Importable source module skipped after mypyc rejection.
        reason: Representative mypyc diagnostic explaining the skip.
    """

    module: str
    reason: str


@dataclass(frozen=True, slots=True)
class CompilationPreflightBlockerInput:
    """Internal input for a known module-level mypyc blocker.

    Preflight blockers are emitted before running mypyc so the report can explain
    why a module was not attempted at all.

    Attributes:
        module: Importable source module blocked before mypyc invocation.
        path: Absolute path of the blocked source module before report normalization.
        line: One-based diagnostic or blocker line.
        code: Stable machine-readable diagnostic or blocker code.
        message: Human-readable diagnostic or blocker explanation.
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
    lowering_mode: LoweringMode = "whole-callable"
    native_helpers: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CompilationReportInput:
    """All command evidence needed to render one compilation report.

    The renderer derives success from build, verification, and optional semantic
    test evidence by default. Source-clean orchestration may also supply its
    authoritative operation status because its verification history can include
    failed candidate-selection probes followed by a successful final package.
    Paths are kept as `Path` objects until rendering so they can be normalized
    relative to `root`.

    Attributes:
        root: Root directory of the target Python project.
        operation: Build or compile operation represented by the report.
        module_filter: Optional module restriction applied to compilation.
        islands: Enabled islands included in the operation or report.
        build: Captured native build evidence.
        operation_success: Authoritative command outcome when candidate-selection
            diagnostics must not determine the final package status.
        mode: Compilation or runtime mode represented by this record.
        wheel_path: Source-clean wheel path, when produced.
        verification: Runtime routing result for the island, when run.
        tests: Optional target-project semantic test result.
        cleanup_removed: Generated paths removed after the operation.
        cleanup_kept: Generated paths intentionally retained for diagnostics.
        skipped_modules: Source modules skipped after native compiler rejection.
        preflight_blockers: Module blockers detected before native compilation.
        native_readiness: Post-generation native-readiness evidence.
        typed_regions: Backend-neutral typed regions discovered or reported.
        compiled_regions: Typed regions successfully compiled into the wheel.
        compiled_bindings: Source bindings successfully provided by compiled regions.
        compiled_variants: Backend and specialization variants successfully compiled.
        backend_assessments: Capability assessments produced before lowering.
        artifact_records: Validated install metadata for produced native artifacts.
        verification_steps: Isolated wheel and payload verification evidence.
        test_results: Target-project command evidence used by quality gates.
        performance: Paired performance-gate evidence.
        composition_performance: Direct accepted-arm versus composed-payload evidence.
        profile: Profile-guided candidate evidence, or `None` for explicit static fallback.
        candidate_trials: Greedy marginal-profitability decisions in profile order.
        execution_plans: Selected and rejected scheduler execution-plan candidates.
        applied_execution_plans: Plan IDs staged into the promoted payload.
        execution_plan_trials: Semantic or performance evidence for staged plans.
        fusion_plans: Deterministic task-fusion safety plans.
        fusion_trials: Three-arm task-fusion research evidence.
        source_optimization_plans: Source-level optimization plans discovered for review.
        source_optimization_assessments: Trial-readiness assessments for source plans.
        source_optimization_trials: Semantic and benchmark trials for source patches.
    """

    root: Path
    operation: CompilationOperation
    module_filter: str | None
    islands: tuple[EnabledIslandConfig, ...]
    build: CompileAttempt
    operation_success: bool | None = None
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
    composition_performance: BenchmarkGateResult | None = None
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


def build_scan_report(result: ScanResult) -> ScanReport:
    """Convert enriched scan dataclasses into the stable scan JSON shape.

    Args:
        result: Enriched project scan to convert into a stable report.

    Returns:
        ScanReport: Versioned JSON-compatible scan report.
    """
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
        "version": SCAN_REPORT_SCHEMA_VERSION,
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

    Args:
        path: Filesystem path consumed or produced by the operation.
        report: Stable report mapping to serialize or render.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8")


def write_markdown_report(path: Path, report: ScanReport) -> None:
    """Write the human-readable scan report next to JSON artifacts.

    Args:
        path: Filesystem path consumed or produced by the operation.
        report: Stable report mapping to serialize or render.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown_report(report), encoding="utf-8")


def render_markdown_report(report: ScanReport) -> str:
    """Render a concise Markdown scan report for users reviewing candidates.

    Args:
        report: Stable report mapping to serialize or render.

    Returns:
        str: Human-readable Markdown scan report ending with one newline.
    """
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
    """Convert build, verification, and cleanup evidence into a stable report.

    Args:
        report_input: Build, verification, test, and cleanup evidence for one report.

    Returns:
        CompilationReport: Versioned compilation report with derived overall success.
    """
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
    composition_performance = (
        _compilation_performance_report(
            report_input.root,
            report_input.composition_performance,
        )
        if report_input.composition_performance is not None
        else None
    )
    profile = _compilation_profile_report(report_input.profile)
    candidate_trials = _candidate_trial_reports(report_input.candidate_trials)
    execution_plans = _execution_plan_reports(report_input.execution_plans)
    execution_plan_trials = _execution_plan_trial_reports(report_input.execution_plan_trials)
    fusion_plans = _fusion_plan_reports(report_input.fusion_plans)
    fusion_trials = _fusion_trial_reports(report_input.root, report_input.fusion_trials)
    source_optimization = _source_optimization_report(
        report_input.root,
        report_input.source_optimization_plans,
        report_input.source_optimization_assessments,
        report_input.source_optimization_trials,
    )
    accepted_hot_coverage = (
        candidate_trials[-1]["accepted_hot_coverage"] if candidate_trials else 0.0
    )
    backend_decisions = _backend_decision_reports(report_input.backend_assessments)
    suspension_plans = _suspension_plan_reports(
        report_input.typed_regions,
        report_input.compiled_regions,
        report_input.compiled_variants,
        report_input.compiled_bindings,
    )
    accepted_variants = _accepted_variant_reports(compiled_regions)
    rejected_variants = _rejected_variant_reports(report_input.skipped_modules)
    optimization_policy = _optimization_policy_report(performance)
    stage_medians = _stage_median_reports(
        _StageMedianEvidence(
            candidate_trials=candidate_trials,
            execution_plan_trials=execution_plan_trials,
            fusion_trials=fusion_trials,
            source_optimization=source_optimization,
            composition_performance=composition_performance,
            performance=performance,
        )
    )
    cache_decisions = _cache_decision_reports(compiled_regions, report_input.build)
    final_composition = _final_composition_report(
        _FinalCompositionInput(
            root=report_input.root,
            wheel_path=report_input.wheel_path,
            accepted_variants=accepted_variants,
            applied_execution_plans=report_input.applied_execution_plans,
            source_optimization=source_optimization,
            artifact_records=report_input.artifact_records,
            build=report_input.build,
        )
    )
    performance_failed = performance["status"] not in {"passed", "unbenchmarked"}
    wheel_missing = report_input.mode == "source-clean" and report_input.wheel_path is None
    required_evidence_succeeded = (
        report_input.build.success
        and verify_failures == 0
        and test_failures == 0
        and not performance_failed
        and not wheel_missing
    )
    verification_history_succeeded = (
        subprocess_verify_failures == 0
        if report_input.operation_success is None
        else report_input.operation_success
    )
    success = required_evidence_succeeded and verification_history_succeeded
    return {
        "version": COMPILE_REPORT_SCHEMA_VERSION,
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
            "profile_status": profile["status"],
            "profile_mapped_coverage": profile["mapped_coverage"],
            "profile_selected_hot_coverage": profile["selected_hot_coverage"],
            "profile_accepted_hot_coverage": accepted_hot_coverage,
            "execution_plans": len(execution_plans),
            "execution_selected_plans": sum(
                plan["status"] == "selected" for plan in execution_plans
            ),
            "execution_applied_plans": len(report_input.applied_execution_plans),
            "execution_plan_trials": len(execution_plan_trials),
            "fusion_plans": len(fusion_plans),
            "fusion_eligible_plans": sum(plan["eligible"] for plan in fusion_plans),
            "fusion_trials": len(fusion_trials),
            "source_optimization_status": source_optimization["status"],
            "source_optimization_plans": len(source_optimization["plans"]),
            "source_optimization_trial_ready_assessments": sum(
                assessment["status"] == "trial-ready"
                for assessment in source_optimization["assessments"]
            ),
            "source_optimization_trials": len(source_optimization["trials"]),
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
        "composition_performance": composition_performance,
        "optimization_policy": optimization_policy,
        "stage_medians": stage_medians,
        "cache_decisions": cache_decisions,
        "final_composition": final_composition,
        "profile": profile,
        "candidate_trials": candidate_trials,
        "execution_plans": execution_plans,
        "applied_execution_plans": list(report_input.applied_execution_plans),
        "execution_plan_trials": execution_plan_trials,
        "fusion_plans": fusion_plans,
        "fusion_trials": fusion_trials,
        "source_optimization": source_optimization,
        "suspension_plans": suspension_plans,
        "backend_decisions": backend_decisions,
        "accepted_variants": accepted_variants,
        "rejected_variants": rejected_variants,
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
                    lowering_mode=variant.lowering_mode,
                    native_helpers=variant.native_helpers,
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
        "lowering_mode": identity.lowering_mode,
        "native_helpers": list(identity.native_helpers),
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
    """Write a machine-readable compilation report as sorted JSON.

    Args:
        path: Filesystem path consumed or produced by the operation.
        report: Stable report mapping to serialize or render.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8")


def write_compilation_markdown_report(path: Path, report: CompilationReport) -> None:
    """Write the human-readable compilation report for CLI workflows.

    Args:
        path: Filesystem path consumed or produced by the operation.
        report: Stable report mapping to serialize or render.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_compilation_markdown_report(report), encoding="utf-8")


def render_compilation_markdown_report(report: CompilationReport) -> str:
    """Render a concise Markdown report for a build or compile attempt.

    Args:
        report: Stable report mapping to serialize or render.

    Returns:
        str: Human-readable Markdown compilation report ending with one newline.
    """
    status = "success" if report["success"] else "failed"
    module_filter = report["module_filter"] or "all discovered modules"
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
        f"- Legacy islands: {report['summary']['islands']}",
        f"- Typed regions: {report['summary']['typed_regions']}",
        f"- Compiled regions: {report['summary']['compiled_regions']}",
        f"- Symbols: {report['summary']['symbols']}",
        f"- Artifacts: {report['summary']['artifacts']}",
        f"- Support artifacts: {report['summary']['support_artifacts']}",
        f"- Skipped modules: {report['summary']['skipped_modules']}",
        f"- Preflight blockers: {report['summary']['preflight_blockers']}",
        f"- Legacy island verifications: {report['summary']['verified']}",
        f"- Legacy verification failures: {report['summary']['verify_failures']}",
        f"- Subprocess verifications: {report['summary']['subprocess_verifications']}",
        (
            "- Subprocess verification failures: "
            f"{report['summary']['subprocess_verification_failures']}"
        ),
        f"- Semantic tests: {_semantic_test_summary(report['tests'])}",
        f"- Performance: {report['summary']['performance_status']}",
        (
            "- Profitable hot-path coverage: "
            f"{report['summary']['profile_accepted_hot_coverage']:.1%}"
        ),
        (
            "- Execution plans: "
            f"{report['summary']['execution_plans']} candidates, "
            f"{report['summary']['execution_selected_plans']} selected, "
            f"{report['summary']['execution_applied_plans']} applied"
        ),
        f"- Execution-plan trials: {report['summary']['execution_plan_trials']}",
        (
            "- Task-fusion plans: "
            f"{report['summary']['fusion_plans']} total, "
            f"{report['summary']['fusion_eligible_plans']} eligible"
        ),
        f"- Task-fusion trials: {report['summary']['fusion_trials']}",
        (
            "- Source optimization: "
            f"{report['summary']['source_optimization_status']}, "
            f"{report['summary']['source_optimization_plans']} plan(s), "
            f"{report['summary']['source_optimization_trial_ready_assessments']} "
            "trial-ready"
        ),
        (f"- Source-optimization trials: {report['summary']['source_optimization_trials']}"),
        f"- Build duration: {report['summary']['duration_seconds']:.3f}s",
        "",
        "## Verification Scope",
        "",
        _verification_scope_text(report["mode"]),
        "",
    ]
    _append_final_composition_markdown(lines, report["final_composition"])
    _append_optimization_policy_markdown(lines, report["optimization_policy"])
    _append_stage_medians_markdown(lines, report["stage_medians"])
    _append_profile_guided_selection_markdown(lines, report["profile"])
    _append_candidate_trials_markdown(lines, report["candidate_trials"])
    _append_execution_plans_markdown(
        lines,
        report["execution_plans"],
        report["applied_execution_plans"],
        report["execution_plan_trials"],
    )
    _append_fusion_plans_markdown(lines, report["fusion_plans"])
    _append_fusion_trials_markdown(lines, report["fusion_trials"])
    _append_source_optimization_markdown(lines, report["source_optimization"])
    _append_suspension_plans_markdown(lines, report["suspension_plans"])
    if report["typed_regions"]:
        lines.extend(["## Planned Regions", ""])
        lines.extend(
            f"- `{region['id']}`: " + ", ".join(member["id"] for member in region["members"])
            for region in report["typed_regions"]
        )
    _append_compiled_regions_markdown(lines, report["compiled_regions"])
    _append_cache_decisions_markdown(lines, report["cache_decisions"])
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
    composition_performance = report.get("composition_performance")
    if composition_performance is not None:
        _append_performance_markdown(
            lines,
            composition_performance,
            heading="Composition Performance",
        )
    _append_performance_markdown(lines, report["performance"])
    _append_cleanup_markdown(lines, report["cleanup"])
    _append_source_clean_skip_markdown(lines, report)
    lines.extend(["", "## Islands", ""])
    if not report["islands"]:
        lines.append("- None")
    for island in report["islands"]:
        lines.extend(_compilation_markdown_island(island))
    return "\n".join(lines).rstrip() + "\n"


def _append_final_composition_markdown(
    lines: list[str],
    composition: CompilationFinalCompositionReport,
) -> None:
    """Explain which accepted layers and artifacts are present in the wheel.

    Args:
        lines: Mutable Markdown output receiving the rendered section.
        composition: Accepted source, native, and execution-plan layers.
    """
    lines.extend(
        [
            "## Final Composition",
            "",
            f"- Wheel: {_optional_path(composition['wheel_path'])}",
            f"- Source plans: {_inline_code_list(composition['source_plan_ids'])}",
            f"- Source transformations: {_inline_code_list(composition['transformation_ids'])}",
            f"- Native variants: {_inline_code_list(composition['native_variant_ids'])}",
            f"- Execution plans: {_inline_code_list(composition['execution_plan_ids'])}",
            f"- Native artifacts: {_inline_code_list(composition['artifacts'])}",
            f"- Retained previous accepted arm: {_yes_no(composition['retained_previous_arm'])}",
            "",
        ]
    )


def _append_optimization_policy_markdown(
    lines: list[str],
    policy: CompilationOptimizationPolicyReport,
) -> None:
    """Render the centralized stability and promotion thresholds.

    Args:
        lines: Mutable Markdown output receiving the rendered section.
        policy: Numerical thresholds used for every profitability decision.
    """
    lines.extend(
        [
            "## Optimization Policy",
            "",
            f"- Policy version: {policy['version']}",
            f"- Stability floor: {policy['stability_floor_seconds']:.3f}s per median",
            (
                "- Profile-guided marginal floor: "
                f"{policy['profile_guided_minimum_marginal_speedup']:.3f}x"
            ),
            (
                "- Specialized marginal floor: "
                f"{policy['specialized_minimum_marginal_speedup']:.3f}x"
            ),
            f"- Final payload floor: {policy['final_minimum_speedup']:.3f}x",
            f"- Hard benchmark floor: {policy['hard_benchmark_minimum_speedup']:.3f}x",
            "",
        ]
    )


def _append_stage_medians_markdown(
    lines: list[str],
    medians: list[CompilationStageMedianReport],
) -> None:
    """Render normalized stage medians in optimization order.

    Args:
        lines: Mutable Markdown output receiving the rendered section.
        medians: Ordered credible median comparisons from optimizer stages.
    """
    if not medians:
        return
    lines.extend(["## Stage Medians", ""])
    lines.extend(
        (
            f"- `{item['stage']}`: {item['baseline_median_seconds']:.3f}s -> "
            f"{item['candidate_median_seconds']:.3f}s, {item['speedup']:.3f}x "
            f"against {item['minimum_speedup']:.3f}x ({item['status']})"
        )
        for item in medians
    )
    lines.append("")


def _append_cache_decisions_markdown(
    lines: list[str],
    decisions: list[CompilationCacheDecisionReport],
) -> None:
    """Render per-variant cache restoration and physical batching evidence.

    Args:
        lines: Mutable Markdown output receiving the rendered section.
        decisions: Ordered cache outcomes for compiled variants.
    """
    if not decisions:
        return
    lines.extend(["", "## Cache Decisions", ""])
    lines.extend(
        f"- `{item['variant_id']}` [{item['backend']}]: {item['status']}; "
        f"batched {_yes_no(item['batched'])}"
        for item in decisions
    )


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
    lines.append(
        "- Evidence can include rejected candidate-selection probes as well as final validation."
    )
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
    *,
    heading: str = "Performance",
) -> None:
    lines.extend(
        [
            "",
            f"## {heading}",
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


def _append_profile_guided_selection_markdown(
    lines: list[str],
    profile: CompilationProfileReport,
) -> None:
    broad_invocations = profile["broad_invocations"]
    observed_invocations = broad_invocations["observed_events"]
    invocation_cap_reached = broad_invocations["capped"]
    lines.extend(
        [
            "## Profile-Guided Selection",
            "",
            f"- Status: {profile['status']}",
            f"- Reason: {profile['reason']}",
            (
                "- Sample sufficiency: "
                f"{profile['total_samples']} total sample(s), "
                f"{profile['mapped_project_samples']} mapped to project code"
            ),
            f"- Mapped coverage: {profile['mapped_coverage']:.1%}",
            f"- Selected leaf-sample coverage: {profile['selected_hot_coverage']:.1%}",
            (
                "- Bounded invocation evidence: "
                f"{observed_invocations} event(s), "
                f"cap reached: {invocation_cap_reached}"
            ),
        ]
    )
    if profile["launch_kind"] == "unsupported":
        lines.append("- Unsupported launcher: using static fallback candidate evidence")
    passes = profile["child_passes"]
    if passes:
        pass_text = ", ".join(
            f"{child['pass_kind']} {child['duration_seconds']:.3f}s" for child in passes
        )
        lines.append(f"- Unmeasured profiling passes: {pass_text}")
    selected = profile["selected_symbols"]
    selected_text = ", ".join(f"`{symbol}`" for symbol in selected) if selected else "none"
    selected_activity = [
        candidate for candidate in profile["candidate_mapping_decisions"] if candidate["selected"]
    ]
    selected_activity_text = (
        ", ".join(
            (
                f"`{candidate['module']}::{candidate['qualname']}` "
                f"{candidate['samples']} leaf + "
                f"{candidate['scheduler_overhead_samples']} nested = "
                f"{candidate['attributed_samples']}; "
                f"{candidate.get('invocation_lower_bound', 0)}.."
                f"{candidate.get('invocation_upper_bound', 0)} invocation(s); "
                f"basis {candidate.get('selection_basis', 'legacy')}"
            )
            for candidate in selected_activity
        )
        if selected_activity
        else "none"
    )
    rejected = [
        candidate
        for candidate in profile["candidate_mapping_decisions"]
        if not candidate["selected"]
    ]
    rejected_text = (
        ", ".join(
            f"`{candidate['module']}::{candidate['qualname']}` ({candidate['reason']})"
            for candidate in rejected
        )
        if rejected
        else "none"
    )
    capped = [
        f"{member['module']}::{member['qualname']}"
        for member in profile["members"]
        if member["observation_capped"]
    ]
    capped_text = ", ".join(f"`{symbol}`" for symbol in capped) if capped else "none"
    lines.extend(
        [
            f"- Selected candidates: {selected_text}",
            f"- Selected candidate activity: {selected_activity_text}",
            f"- Rejected candidates: {rejected_text}",
            f"- Bounded type observation reached: {capped_text}",
            "",
        ]
    )


def _append_candidate_trials_markdown(
    lines: list[str],
    trials: list[CompilationCandidateTrialReport],
) -> None:
    """Render marginal candidate decisions separately from the final benchmark gate.

    Args:
        lines: Mutable Markdown line buffer receiving the section.
        trials: Ordered candidate semantic and benchmark decisions.
    """
    if not trials:
        return
    lines.extend(["## Candidate Profitability", ""])
    for trial in trials:
        speedup = (
            f"{trial['marginal_speedup']:.3f}x"
            if trial["marginal_speedup"] is not None
            else "unavailable"
        )
        fallback = f"; fallback: {trial['fallback_reason']}" if trial["fallback_reason"] else ""
        lines.append(
            f"- `{trial['variant_id']}` [{trial['backend']}, {trial['lowering_mode']}]: "
            f"{trial['status']}; marginal speedup {speedup}; "
            f"candidate coverage {trial['profile_coverage']:.1%}; "
            f"accepted coverage {trial['accepted_hot_coverage']:.1%}; "
            f"{trial['reason']}{fallback}"
        )
    lines.append("")


def _append_execution_plans_markdown(
    lines: list[str],
    plans: list[CompilationExecutionPlanReport],
    applied: list[str],
    trials: list[CompilationExecutionPlanTrialReport],
) -> None:
    """Render scheduler execution-plan discovery separately from native regions.

    Args:
        lines: Mutable Markdown line buffer receiving the section.
        plans: Selected and rejected execution-plan candidates.
        applied: Plan IDs staged into the promoted payload.
        trials: Semantic or performance evidence for staged plans.
    """
    if not plans and not applied and not trials:
        return
    lines.extend(["## Async Execution Plans", ""])
    for plan in plans:
        rejection_text = "; ".join(
            f"{rejection['code']}: {rejection['reason']}" for rejection in plan["rejections"]
        )
        detail = rejection_text or "selected for backend assessment"
        lines.append(
            f"- `{plan['id']}` [{plan['status']}]: `{plan['owner']}`; "
            f"dialect `{plan['dialect'] or 'unresolved'}`; "
            f"observed invocations {plan['observed_invocations']}; "
            f"lifecycle starts {plan['lifecycle_starts']} "
            f"({plan['lifecycle_share']:.1%} mapped async activity); {detail}"
        )
    lines.append("- Applied plans: " + (", ".join(f"`{plan_id}`" for plan_id in applied) or "none"))
    lines.extend(
        (
            f"- Trial `{trial['plan_id']}`: {trial['status']}; "
            f"backend `{trial['backend'] or 'unavailable'}`; semantic exit "
            f"{trial['exit_code'] if trial['exit_code'] is not None else 'not run'}; "
            f"marginal benchmark {trial['benchmark_status']}"
            + (
                f" at {trial['marginal_speedup']:.3f}x marginal"
                if trial["marginal_speedup"] is not None
                else ""
            )
            + (
                f", {trial['overall_speedup']:.3f}x overall"
                if trial["overall_speedup"] is not None
                else ""
            )
            + f"; staging cache {trial['cache_status']}"
            + (f"; {trial['reason']}" if trial["reason"] else "")
        )
        for trial in trials
    )
    lines.extend(
        [
            "- Runtime status: report-only unless an applied plan and passing trial are listed.",
            "",
        ]
    )


def _append_fusion_plans_markdown(
    lines: list[str],
    plans: list[CompilationFusionPlanReport],
) -> None:
    """Render task-fusion safety evidence without implying runtime activation.

    Args:
        lines: Mutable Markdown line buffer receiving the section.
        plans: Ordered deterministic fusion plans.
    """
    if not plans:
        return
    lines.extend(["## Experimental Task-Fusion Research", ""])
    for plan in plans:
        status = "eligible for a trial" if plan["eligible"] else "rejected before trial"
        rejection_text = (
            "; ".join(
                f"{rejection['code']}: {rejection['reason']}" for rejection in plan["rejections"]
            )
            or "all safety gates passed"
        )
        callee = plan["callee"] or "unresolved callee"
        lines.append(
            f"- `{plan['id']}`: {status}; `{plan['caller']}` -> `{callee}` via "
            f"`{plan['spawn_api']}`; calls {plan['completed_calls']}/"
            f"{plan['observed_calls']} complete; max overlap {plan['max_active_calls']}; "
            f"pre-completion suspensions {plan['pre_completion_suspensions']}; "
            f"{rejection_text}"
        )
    lines.extend(
        [
            "- Runtime status: research evidence only; task fusion is not enabled by default.",
            "",
        ]
    )


def _append_fusion_trials_markdown(
    lines: list[str],
    trials: list[CompilationFusionTrialReport],
) -> None:
    """Render measured three-arm fusion trials when an eligible variant exists.

    Args:
        lines: Mutable Markdown line buffer receiving the section.
        trials: Three-arm semantic and profitability evidence.
    """
    if not trials:
        return
    lines.extend(["### Three-Arm Trials", ""])
    for trial in trials:
        overall = (
            f"{trial['baseline_over_fused']:.3f}x"
            if trial["baseline_over_fused"] is not None
            else "unavailable"
        )
        marginal = (
            f"{trial['unfused_over_fused']:.3f}x"
            if trial["unfused_over_fused"] is not None
            else "unavailable"
        )
        lines.append(
            f"- `{trial['plan_id']}` {trial['status']}: overall {overall}; "
            f"over unfused {marginal}; {trial['reason']}"
        )
    lines.append("")


def _append_source_optimization_markdown(
    lines: list[str],
    source_optimization: CompilationSourceOptimizationReport,
) -> None:
    """Render source optimization planning, trial, patch, and application evidence.

    Args:
        lines: Mutable Markdown line buffer receiving the section.
        source_optimization: Serialized source-optimization report section.
    """
    lines.extend(["## Source Optimization", ""])
    lines.extend(
        [
            f"- Status: {source_optimization['status']}",
            f"- Minimum speedup: {source_optimization['minimum_speedup']:.3f}x",
            f"- Headroom speedup: {_optional_speedup(source_optimization['headroom_speedup'])}",
            f"- Attributed hot share: {source_optimization['attributed_hot_share']:.1%}",
            f"- Patch: {_optional_path(source_optimization['patch_path'])}",
            f"- Application status: {source_optimization['application_status']}",
            "- Runtime status: only an accepted trial contributes a transformed wheel or patch.",
        ]
    )
    if source_optimization["patch_path"] is None:
        lines.append("- No patch was emitted.")
    lines.extend(
        (
            f"- Plan `{plan['id']}`: `{plan['owner']}` -> `{plan['worker']}` via "
            f"`{plan['transport']}`; {len(plan['steps'])} step(s); boundaries "
            f"{', '.join(plan['semantic_boundaries']) or 'none'}"
        )
        for plan in source_optimization["plans"]
    )
    for assessment in source_optimization["assessments"]:
        rejection_text = "; ".join(assessment["rejections"]) or "no rejection evidence"
        lines.append(
            f"- Assessment `{assessment['plan_id']}`: {assessment['status']}; "
            f"minimum {assessment['minimum_speedup']:.3f}x; "
            f"hot share {assessment['attributed_hot_share']:.1%}; {rejection_text}"
        )
    lines.extend(
        (
            f"- Trial `{trial['plan_id']}`: {trial['status']}; source "
            f"{_optional_speedup(trial['source_speedup'])}; wheel "
            f"{_optional_speedup(trial['wheel_speedup'])}; "
            f"application {trial['application_status']}"
            + (
                f"; residual profile {trial['residual_profile']['status']} with "
                f"{trial['residual_profile']['total_samples']} samples"
                if trial["residual_profile"] is not None
                else ""
            )
            + (f"; {trial['reason']}" if trial["reason"] else "")
        )
        for trial in source_optimization["trials"]
    )
    lines.append("")


def _compiled_region_markdown(region: CompilationCompiledRegionReport) -> str:
    backend = region["backend"] or "unknown backend"
    bindings = ", ".join(_compiled_binding_markdown(binding) for binding in region["bindings"])
    artifacts = ", ".join(region["artifacts"]) or "no recorded artifacts"
    helpers = ", ".join(f"`{name}`" for name in region["native_helpers"])
    lowering = region["lowering_mode"]
    lowering_text = f"; lowering: {lowering}"
    if helpers:
        lowering_text += f" via {helpers}"
    return (
        f"- `{region['variant_id']}` [{backend}] for region `{region['id']}`: "
        f"{bindings}; cache: {region['cache_status']}{lowering_text}; artifacts: {artifacts}"
    )


def _append_suspension_plans_markdown(
    lines: list[str],
    plans: list[CompilationSuspensionPlanReport],
) -> None:
    if not plans:
        return
    lines.extend(["## Suspension Handling", ""])
    for plan in plans:
        points = ", ".join(f"{point['kind']} at line {point['lineno']}" for point in plan["points"])
        helpers = ", ".join(f"`{name}`" for name in plan["native_helpers"])
        helper_text = f" via {helpers}" if helpers else ""
        eligible_blocks = sum(block["eligible"] for block in plan["blocks"])
        block_text = f"{eligible_blocks}/{len(plan['blocks'])} synchronous blocks eligible"
        lines.append(
            f"- `{plan['member']}`: {plan['lowering_mode']}{helper_text} "
            f"({points}); {block_text}; {plan['reason']}"
        )
    lines.append("")


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
                "call_sites": [_call_site_report(call) for call in symbol.call_sites],
                "suspension_points": [
                    _suspension_point_report(point) for point in symbol.suspension_points
                ],
                "runtime_imports": [_import_report(record) for record in symbol.runtime_imports],
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
                "invocation_mode": edge.invocation_mode,
                "requires_same_unit": edge.requires_same_unit,
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
                "call_sites": [_call_site_report(call) for call in member.call_sites],
                "suspension_points": [
                    _suspension_point_report(point) for point in member.suspension_points
                ],
                "runtime_imports": [_import_report(record) for record in member.runtime_imports],
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
                "lineno": dependency.lineno,
                "invocation_mode": dependency.invocation_mode,
                "requires_same_unit": dependency.requires_same_unit,
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
    """Serialize one runtime guard without exposing target code objects.

    Args:
        guard: Runtime type guard being rendered or evaluated.

    Returns:
        RuntimeTypeGuardReport: Serialized runtime type guard.
    """
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


def _call_site_report(call: CallSiteFact) -> CallSiteReport:
    return {
        "target": call.target,
        "root_name": call.root_name,
        "invocation_mode": call.invocation_mode,
        "lineno": call.lineno,
        "end_lineno": call.end_lineno,
        "col_offset": call.col_offset,
        "end_col_offset": call.end_col_offset,
        "requires_same_unit": call.requires_same_unit,
    }


def _suspension_point_report(point: SuspensionPoint) -> SuspensionPointReport:
    return {
        "kind": point.kind,
        "lineno": point.lineno,
        "end_lineno": point.end_lineno,
        "col_offset": point.col_offset,
        "end_col_offset": point.end_col_offset,
    }


def _import_report(record: ImportRecord) -> ImportReport:
    return {
        "source_text": record.source_text,
        "imported_names": list(record.imported_names),
        "module": record.module,
        "level": record.level,
        "lineno": record.lineno,
        "end_lineno": record.end_lineno,
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
    """Return a short label for a candidate score.

    Args:
        score: Scan-only candidate score in the inclusive 0-100 range.

    Returns:
        str: Short qualitative label for the candidate score.
    """
    if score >= _STRONG_SCORE:
        return "strong"
    if score >= _GOOD_SCORE:
        return "good"
    if score >= _POSSIBLE_SCORE:
        return "possible"
    return "weak"


def score_summary(score: int) -> str:
    """Explain a scan-only candidate score in user-facing language.

    Args:
        score: Scan-only candidate score in the inclusive 0-100 range.

    Returns:
        str: User-facing score text that does not imply measured performance.
    """
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
    """Explain candidate extraction risk in user-facing scan report language.

    Args:
        risk: Candidate extraction risk classification to explain.

    Returns:
        str: User-facing explanation of the extraction risk classification.
    """
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
            "minimum_speedup": DEFAULT_MINIMUM_FINAL_SPEEDUP,
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


def _optimization_policy_report(
    performance: CompilationPerformanceReport,
) -> CompilationOptimizationPolicyReport:
    """Serialize the single numerical policy applied across optimizer families.

    Args:
        performance: Final gate evidence carrying the configured payload threshold.

    Returns:
        Stable policy fields embedded in JSON and Markdown compilation reports.
    """
    return {
        "version": OPTIMIZATION_POLICY_VERSION,
        "stability_floor_seconds": MINIMUM_STABLE_MEDIAN_SECONDS,
        "profile_guided_minimum_marginal_speedup": (PROFILE_GUIDED_MINIMUM_MARGINAL_SPEEDUP),
        "specialized_minimum_marginal_speedup": DEFAULT_MINIMUM_MARGINAL_SPEEDUP,
        "final_minimum_speedup": performance["minimum_speedup"],
        "hard_benchmark_minimum_speedup": HARD_BENCHMARK_MINIMUM_SPEEDUP,
    }


def _stage_median_reports(evidence: _StageMedianEvidence) -> list[CompilationStageMedianReport]:
    """Normalize family-specific timing evidence into one ordered progression.

    Args:
        evidence: Native, execution-plan, source, composition, and final timing evidence.

    Returns:
        Credible timing comparisons in optimizer application order.
    """
    reports: list[CompilationStageMedianReport] = []
    for candidate_trial in evidence.candidate_trials:
        _append_stage_median(
            reports,
            _StageMedianInput(
                stage=f"native:{candidate_trial['variant_id']}",
                status=candidate_trial["benchmark_status"],
                baseline=candidate_trial["baseline_median_seconds"],
                candidate=candidate_trial["candidate_median_seconds"],
                speedup=candidate_trial["marginal_speedup"],
                minimum=candidate_trial["minimum_speedup"],
            ),
        )
    for execution_plan_trial in evidence.execution_plan_trials:
        _append_stage_median(
            reports,
            _StageMedianInput(
                stage=f"execution-plan:{execution_plan_trial['plan_id']}",
                status=execution_plan_trial["benchmark_status"],
                baseline=execution_plan_trial["unplanned_median_seconds"],
                candidate=execution_plan_trial["planned_median_seconds"],
                speedup=execution_plan_trial["marginal_speedup"],
                minimum=execution_plan_trial["minimum_speedup"],
            ),
        )
    for source_trial in evidence.source_optimization["trials"]:
        current_median = source_trial["current_median_seconds"]
        source_median = source_trial["source_median_seconds"]
        _append_stage_median(
            reports,
            _StageMedianInput(
                stage=f"source-search:{source_trial['candidate_id']}",
                status=source_trial["status"],
                baseline=current_median,
                candidate=source_median,
                speedup=_median_speedup_ratio(current_median, source_median),
                minimum=DEFAULT_MINIMUM_MARGINAL_SPEEDUP,
            ),
        )
        _append_stage_median(
            reports,
            _StageMedianInput(
                stage=f"source-final:{source_trial['candidate_id']}",
                status=source_trial["status"],
                baseline=source_trial["baseline_median_seconds"],
                candidate=source_trial["wheel_median_seconds"],
                speedup=source_trial["wheel_speedup"],
                minimum=evidence.source_optimization["minimum_speedup"],
            ),
        )
    for fusion_trial in evidence.fusion_trials:
        _append_stage_median(
            reports,
            _StageMedianInput(
                stage=f"fusion:{fusion_trial['plan_id']}",
                status=fusion_trial["status"],
                baseline=fusion_trial["unfused_median_seconds"],
                candidate=fusion_trial["fused_median_seconds"],
                speedup=fusion_trial["unfused_over_fused"],
                minimum=DEFAULT_MINIMUM_MARGINAL_SPEEDUP,
            ),
        )
    if evidence.composition_performance is not None:
        _append_stage_median(
            reports,
            _StageMedianInput(
                stage="native-composition",
                status=evidence.composition_performance["status"],
                baseline=evidence.composition_performance["baseline_median_seconds"],
                candidate=evidence.composition_performance["compiled_median_seconds"],
                speedup=evidence.composition_performance["speedup"],
                minimum=evidence.composition_performance["minimum_speedup"],
            ),
        )
    _append_stage_median(
        reports,
        _StageMedianInput(
            stage="final-payload",
            status=evidence.performance["status"],
            baseline=evidence.performance["baseline_median_seconds"],
            candidate=evidence.performance["compiled_median_seconds"],
            speedup=evidence.performance["speedup"],
            minimum=evidence.performance["minimum_speedup"],
        ),
    )
    return reports


def _median_speedup_ratio(baseline: float | None, candidate: float | None) -> float | None:
    """Return a report ratio only when both medians permit division.

    Args:
        baseline: Median duration before the candidate transformation.
        candidate: Median duration after the candidate transformation.

    Returns:
        Baseline divided by candidate, or `None` for incomplete or zero evidence.
    """
    if baseline is None or candidate is None or candidate == 0.0:
        return None
    return baseline / candidate


def _append_stage_median(
    reports: list[CompilationStageMedianReport],
    item: _StageMedianInput,
) -> None:
    if (
        item.baseline is None
        or item.candidate is None
        or item.speedup is None
        or item.minimum is None
    ):
        return
    reports.append(
        {
            "stage": item.stage,
            "status": item.status,
            "baseline_median_seconds": item.baseline,
            "candidate_median_seconds": item.candidate,
            "speedup": item.speedup,
            "minimum_speedup": item.minimum,
        }
    )


def _cache_decision_reports(
    compiled_regions: list[CompilationCompiledRegionReport],
    build: CompileAttempt,
) -> list[CompilationCacheDecisionReport]:
    return [
        {
            "variant_id": region["variant_id"],
            "backend": region["backend"],
            "status": region["cache_status"],
            "batched": any(
                timing.name in {"cython_batch", "cython_batch_member"}
                and timing.detail is not None
                and region["variant_id"] in timing.detail
                for timing in build.phase_timings
            ),
        }
        for region in compiled_regions
    ]


def _final_composition_report(
    inputs: _FinalCompositionInput,
) -> CompilationFinalCompositionReport:
    accepted_source_trials = tuple(
        trial for trial in inputs.source_optimization["trials"] if trial["status"] == "accepted"
    )
    return {
        "source_plan_ids": list(
            dict.fromkeys(trial["plan_id"] for trial in accepted_source_trials)
        ),
        "transformation_ids": list(
            dict.fromkeys(
                transformation_id
                for trial in accepted_source_trials
                for transformation_id in trial["transformation_ids"]
            )
        ),
        "native_variant_ids": [variant["variant_id"] for variant in inputs.accepted_variants],
        "execution_plan_ids": list(inputs.applied_execution_plans),
        "artifacts": sorted(record.install_relative_path for record in inputs.artifact_records),
        "wheel_path": (
            _path_text(inputs.root, inputs.wheel_path) if inputs.wheel_path is not None else None
        ),
        "retained_previous_arm": "composition fallback retained:" in inputs.build.stdout,
    }


def _candidate_trial_reports(
    trials: tuple[CandidateTrial, ...],
) -> list[CompilationCandidateTrialReport]:
    """Serialize greedy candidate decisions without treating rejections as failures.

    Args:
        trials: Ordered semantic and marginal benchmark decisions.

    Returns:
        list[CompilationCandidateTrialReport]: JSON-compatible trial evidence in profile order.
    """
    return [
        {
            "id": trial.id,
            "region_id": trial.variant_id,
            "source_region_id": trial.source_region_id,
            "variant_id": trial.variant_id,
            "backend": trial.backend,
            "lowering_mode": trial.lowering_mode,
            "symbols": list(trial.symbols),
            "status": trial.status,
            "reason": trial.reason,
            "marginal_speedup": trial.marginal_speedup,
            "fallback_reason": trial.fallback_reason,
            "profile_samples": trial.profile_samples,
            "profile_coverage": trial.profile_coverage,
            "accepted_hot_coverage": trial.accepted_hot_coverage,
            "baseline_variants": list(trial.baseline_variants),
            "trial_variants": list(trial.trial_variants),
            "semantic_test_exit_code": trial.semantic_test_exit_code,
            "semantic_test_duration_seconds": trial.semantic_test_duration_seconds,
            "benchmark_status": trial.benchmark_status,
            "baseline_median_seconds": trial.baseline_median_seconds,
            "candidate_median_seconds": trial.candidate_median_seconds,
            "minimum_speedup": trial.minimum_speedup,
        }
        for trial in trials
    ]


def _execution_plan_reports(
    plans: tuple[ExecutionPlan | PlanRejection, ...],
) -> list[CompilationExecutionPlanReport]:
    """Serialize selected and rejected scheduler execution-plan candidates.

    Args:
        plans: Profile-ranked execution plans and report-only rejections.

    Returns:
        list[CompilationExecutionPlanReport]: Stable JSON-compatible plan evidence.
    """
    reports: list[CompilationExecutionPlanReport] = []
    for plan in plans:
        if isinstance(plan, PlanRejection):
            reports.append(
                {
                    "id": plan.id,
                    "status": "rejected",
                    "source_module": plan.source_module,
                    "owner": plan.owner.stable_id,
                    "dialect": plan.dialect,
                    "lowering_version": None,
                    "source_hash": None,
                    "source_hashes": {},
                    "source_members": [],
                    "callsite_fingerprint": None,
                    "topology_fingerprint": None,
                    "completion_transport": None,
                    "consumer": None,
                    "reducer": None,
                    "transport_capacity": None,
                    "ordering_policy": None,
                    "task_ownership": None,
                    "observed_invocations": plan.hotness,
                    "lifecycle_starts": 0,
                    "lifecycle_share": 0.0,
                    "guarded_callable_identities": [],
                    "hotness": plan.hotness,
                    "nodes": [],
                    "edges": [],
                    "guards": [],
                    "rejections": [{"code": plan.reason, "reason": plan.message}],
                }
            )
            continue
        reports.append(
            {
                "id": plan.id,
                "status": "selected",
                "source_module": plan.source_module,
                "owner": plan.owner.stable_id,
                "dialect": plan.dialect,
                "lowering_version": plan.lowering_version,
                "source_hash": plan.source_hash,
                "source_hashes": dict(plan.source_hashes),
                "source_members": [member.stable_id for member in plan.source_members],
                "callsite_fingerprint": plan.callsite_fingerprint,
                "topology_fingerprint": plan.topology_fingerprint,
                "completion_transport": plan.completion_transport,
                "consumer": plan.consumer.stable_id if plan.consumer is not None else None,
                "reducer": plan.reducer.stable_id if plan.reducer is not None else None,
                "transport_capacity": plan.transport_capacity,
                "ordering_policy": plan.ordering_policy,
                "task_ownership": plan.task_ownership,
                "observed_invocations": plan.observed_invocations,
                "lifecycle_starts": plan.lifecycle_starts,
                "lifecycle_share": plan.lifecycle_share,
                "guarded_callable_identities": list(plan.guarded_callable_identities),
                "hotness": plan.hotness,
                "nodes": [
                    {
                        "id": node.id,
                        "symbol": node.symbol.stable_id if node.symbol is not None else None,
                        "role": node.role,
                        "lineno": node.lineno,
                    }
                    for node in plan.nodes
                ],
                "edges": [
                    {
                        "src": edge.src,
                        "dst": edge.dst,
                        "kind": edge.kind,
                        "transport": edge.transport,
                        "lineno": edge.lineno,
                    }
                    for edge in plan.edges
                ],
                "guards": [
                    {
                        "kind": guard.kind,
                        "expression": guard.expression,
                        "message": guard.message,
                    }
                    for guard in plan.guards
                ],
                "rejections": [
                    {"code": rejection.reason, "reason": rejection.message}
                    for rejection in plan.rejections
                ],
            }
        )
    return reports


def _execution_plan_trial_reports(
    trials: tuple[ExecutionPlanTrial, ...],
) -> list[CompilationExecutionPlanTrialReport]:
    """Serialize execution-plan semantic and performance trial evidence.

    Args:
        trials: Ordered staged-plan trials.

    Returns:
        list[CompilationExecutionPlanTrialReport]: Stable JSON-compatible trial evidence.
    """
    return [
        {
            "plan_id": trial.plan_id,
            "status": trial.status,
            "command": list(trial.command),
            "exit_code": trial.exit_code,
            "duration_seconds": trial.duration_seconds,
            "diagnostics": [
                _execution_plan_diagnostic_report(diagnostic) for diagnostic in trial.diagnostics
            ],
            "backend": trial.backend,
            "reason": trial.reason,
            "benchmark_command": list(trial.benchmark_command),
            "benchmark_status": trial.benchmark_status,
            "minimum_speedup": trial.minimum_speedup,
            "minimum_overall_speedup": trial.minimum_overall_speedup,
            "baseline_median_seconds": trial.baseline_median_seconds,
            "unplanned_median_seconds": trial.unplanned_median_seconds,
            "planned_median_seconds": trial.planned_median_seconds,
            "marginal_speedup": trial.marginal_speedup,
            "overall_speedup": trial.overall_speedup,
            "cache_status": trial.cache_status,
            "payload_files": [
                {
                    "install_path": payload_file.install_path.as_posix(),
                    "before_hash": payload_file.before_hash,
                    "after_hash": payload_file.after_hash,
                    "role": payload_file.role,
                }
                for payload_file in trial.payload_files
            ],
        }
        for trial in trials
    ]


def _execution_plan_diagnostic_report(
    diagnostic: ExecutionPlanDiagnostic,
) -> CompilationExecutionPlanDiagnosticReport:
    return {
        "code": diagnostic.code,
        "severity": diagnostic.severity,
        "message": diagnostic.message,
        "details": list(diagnostic.details),
    }


def _fusion_plan_reports(plans: tuple[FusionPlan, ...]) -> list[CompilationFusionPlanReport]:
    """Serialize deterministic fusion plans without changing compile success.

    Args:
        plans: Ordered report-only plans and their conservative gate rejections.

    Returns:
        list[CompilationFusionPlanReport]: JSON-compatible fusion safety evidence.
    """
    return [
        {
            "id": plan.id,
            "source_hash": plan.source_hash,
            "root": plan.root,
            "caller": plan.caller,
            "callee": plan.callee,
            "spawn_api": plan.spawn_api,
            "lineno": plan.lineno,
            "end_lineno": plan.end_lineno,
            "col_offset": plan.col_offset,
            "end_col_offset": plan.end_col_offset,
            "eligible": plan.eligible,
            "observed_calls": plan.observed_calls,
            "completed_calls": plan.completed_calls,
            "max_active_calls": plan.max_active_calls,
            "pre_completion_suspensions": plan.pre_completion_suspensions,
            "observed_signatures": plan.observed_signatures,
            "observation_capped": plan.observation_capped,
            "rejections": [
                {"code": rejection.code, "reason": rejection.reason}
                for rejection in plan.rejections
            ],
        }
        for plan in plans
    ]


def _fusion_trial_reports(
    root: Path,
    trials: tuple[FusionTrial, ...],
) -> list[CompilationFusionTrialReport]:
    """Serialize three-arm research evidence separately from wheel promotion.

    Args:
        root: Root directory used to normalize payload paths.
        trials: Fusion semantic and profitability trials.

    Returns:
        list[CompilationFusionTrialReport]: JSON-compatible three-arm evidence.
    """
    return [
        {
            "plan_id": trial.plan_id,
            "status": trial.status,
            "reason": trial.reason,
            "baseline_median_seconds": trial.baseline_median_seconds,
            "unfused_median_seconds": trial.unfused_median_seconds,
            "fused_median_seconds": trial.fused_median_seconds,
            "baseline_over_unfused": trial.baseline_over_unfused,
            "baseline_over_fused": trial.baseline_over_fused,
            "unfused_over_fused": trial.unfused_over_fused,
            "semantic_runs": [_fusion_arm_run_report(root, run) for run in trial.semantic_runs],
            "warmups": [_fusion_arm_run_report(root, run) for run in trial.warmups],
            "samples": [_fusion_arm_run_report(root, run) for run in trial.samples],
        }
        for trial in trials
    ]


def _fusion_arm_run_report(
    root: Path,
    evidence: FusionArmRunEvidence,
) -> CompilationFusionArmRunReport:
    """Tag ordinary command evidence with its baseline, unfused, or fused arm.

    Args:
        root: Root directory used to normalize payload paths.
        evidence: Arm-tagged child-process evidence.

    Returns:
        CompilationFusionArmRunReport: JSON-compatible arm and command evidence.
    """
    return {
        "arm": evidence.arm,
        "run": _compilation_command_run_report(root, evidence.run),
    }


def _source_optimization_report(
    root: Path,
    plans: tuple[SourceOptimizationPlan, ...],
    assessments: tuple[SourceOptimizationAssessment, ...],
    trials: tuple[SourceOptimizationTrial, ...],
) -> CompilationSourceOptimizationReport:
    """Serialize source-optimization evidence retained by compile schema v6.

    The aggregate status is derived from planning, trial, and application
    evidence. Only an accepted trial has a patch path, and command success
    remains governed by compile, semantic, source-performance, and wheel gates.

    Args:
        root: Root directory used to normalize patch paths.
        plans: Source-level optimization plans discovered for review.
        assessments: Trial-readiness and profitability assessments for plans.
        trials: Semantic, benchmark, and application evidence for patches.

    Returns:
        CompilationSourceOptimizationReport: Stable nested source-optimization report.
    """
    plan_reports = [_source_optimization_plan_report(plan) for plan in plans]
    assessment_reports = [
        _source_optimization_assessment_report(assessment)
        for assessment in sorted(assessments, key=lambda item: item.plan_id)
    ]
    trial_reports = [
        _source_optimization_trial_report(root, trial)
        for trial in sorted(trials, key=lambda item: (item.plan_id, item.candidate_id))
    ]
    minimum_speedups = [assessment["minimum_speedup"] for assessment in assessment_reports]
    headroom_speedups = [
        assessment["headroom_speedup"]
        for assessment in assessment_reports
        if assessment["headroom_speedup"] is not None
    ]
    patch_path = next(
        (trial["patch_path"] for trial in trial_reports if trial["patch_path"] is not None),
        None,
    )
    application_status = _source_optimization_application_status(trial_reports)
    return {
        "status": _source_optimization_status(plan_reports, assessment_reports, trial_reports),
        "minimum_speedup": max(
            minimum_speedups,
            default=HARD_BENCHMARK_MINIMUM_SPEEDUP,
        ),
        "headroom_speedup": max(headroom_speedups, default=None),
        "attributed_hot_share": max(
            (assessment["attributed_hot_share"] for assessment in assessment_reports),
            default=0.0,
        ),
        "plans": plan_reports,
        "assessments": assessment_reports,
        "trials": trial_reports,
        "patch_path": patch_path,
        "application_status": application_status,
    }


def _source_optimization_status(
    plans: list[CompilationSourceOptimizationPlanReport],
    assessments: list[CompilationSourceOptimizationAssessmentReport],
    trials: list[CompilationSourceOptimizationTrialReport],
) -> SourceOptimizationReportStatus:
    """Derive the aggregate source-optimization status from nested evidence.

    Args:
        plans: Serialized source optimization plans.
        assessments: Serialized source optimization assessments.
        trials: Serialized source optimization trials.

    Returns:
        SourceOptimizationReportStatus: Conservative overall status for report consumers.
    """
    status: SourceOptimizationReportStatus = "unbenchmarked"
    if trials:
        application_status = _source_optimization_application_status(trials)
        if application_status != "not-applied":
            status = application_status
        elif any(trial["status"] == "accepted" for trial in trials):
            status = "accepted"
        elif all(trial["status"] in {"rejected", "not-profitable"} for trial in trials):
            status = "not-profitable"
        elif any(trial["status"] in {"failed-semantics", "unavailable"} for trial in trials):
            status = "unavailable"
        else:
            status = "rejected"
    elif any(assessment["status"] == "trial-ready" for assessment in assessments):
        status = "report-only"
    elif assessments and all(assessment["status"] == "unbenchmarked" for assessment in assessments):
        status = "unbenchmarked"
    elif plans:
        status = "rejected"
    return status


def _source_optimization_application_status(
    trials: list[CompilationSourceOptimizationTrialReport],
) -> SourceOptimizationApplicationStatus:
    """Return the aggregate patch application status for source optimization.

    Args:
        trials: Serialized source-optimization trial evidence.

    Returns:
        SourceOptimizationApplicationStatus: Highest-priority non-default application status.
    """
    priority: tuple[SourceOptimizationApplicationStatus, ...] = (
        "failed",
        "conflicted",
        "rolled-back",
        "stale-source",
        "applied",
        "unavailable",
    )
    statuses = {trial["application_status"] for trial in trials}
    for status in priority:
        if status in statuses:
            return status
    return "not-applied"


def _source_optimization_plan_report(
    plan: SourceOptimizationPlan,
) -> CompilationSourceOptimizationPlanReport:
    """Serialize one source-optimization plan and static identity inputs.

    Args:
        plan: Source-level optimization plan derived from an execution plan.

    Returns:
        CompilationSourceOptimizationPlanReport: JSON-compatible plan evidence.
    """
    return {
        "id": plan.id,
        "identity": {
            "execution_plan_id": plan.identity.execution_plan_id,
            "source_hashes": {
                path.as_posix(): source_hash
                for path, source_hash in sorted(
                    plan.identity.source_hashes,
                    key=lambda item: item[0].as_posix(),
                )
            },
            "topology_fingerprint": plan.identity.topology_fingerprint,
            "dialect": plan.identity.dialect,
            "lowering_version": plan.identity.lowering_version,
            "python_abi": plan.identity.python_abi,
            "transformation_versions": dict(sorted(plan.identity.transformation_versions)),
        },
        "source": plan.source.as_posix(),
        "owner": plan.owner.stable_id,
        "worker": plan.worker.stable_id,
        "consumer": plan.consumer.stable_id if plan.consumer is not None else None,
        "reducer": plan.reducer.stable_id if plan.reducer is not None else None,
        "transport": plan.transport,
        "access_sites": [_source_access_site_report(site) for site in plan.access_sites],
        "entrypoint": plan.entrypoint.stable_id,
        "steps": [_source_optimization_step_report(step) for step in plan.steps],
        "semantic_boundaries": list(plan.semantic_boundaries),
        "transport_capacity": plan.transport_capacity,
    }


def _source_optimization_step_report(
    step: TransformationStep,
) -> CompilationSourceOptimizationStepReport:
    """Serialize one source transformation step.

    Args:
        step: Ordered transformation step from a source-optimization plan.

    Returns:
        CompilationSourceOptimizationStepReport: JSON-compatible step evidence.
    """
    return {
        "id": step.stable_id,
        "kind": step.kind,
        "version": step.version,
        "source_symbol": step.source_symbol.stable_id,
        "target_symbol": step.target_symbol.stable_id if step.target_symbol is not None else None,
        "access_sites": [_source_access_site_report(site) for site in step.access_sites],
        "semantic_boundary": step.semantic_boundary,
        "description": step.description,
    }


def _source_access_site_report(
    site: SourceAccessSite,
) -> CompilationSourceOptimizationAccessSiteReport:
    """Serialize one source access site without source object references.

    Args:
        site: Static access evidence attached to a source optimization.

    Returns:
        CompilationSourceOptimizationAccessSiteReport: JSON-compatible access evidence.
    """
    return {
        "path": site.path.as_posix(),
        "symbol": site.symbol.stable_id if site.symbol is not None else None,
        "kind": site.kind,
        "lineno": site.lineno,
        "expression": site.expression,
        "hazards": list(site.hazards),
    }


def _source_optimization_assessment_report(
    assessment: SourceOptimizationAssessment,
) -> CompilationSourceOptimizationAssessmentReport:
    """Serialize one source-optimization assessment.

    Args:
        assessment: Capability and profitability assessment for one source plan.

    Returns:
        CompilationSourceOptimizationAssessmentReport: JSON-compatible assessment evidence.
    """
    return {
        "plan_id": assessment.plan_id,
        "status": assessment.status,
        "minimum_speedup": assessment.minimum_speedup,
        "work_items": [symbol.stable_id for symbol in assessment.work_items],
        "observed_work_items": assessment.observed_work_items,
        "immediate_result_ratio": assessment.immediate_result_ratio,
        "attributed_hot_share": assessment.attributed_hot_share,
        "scheduler_overhead_samples": assessment.scheduler_overhead_samples,
        "scheduler_overhead_share": assessment.scheduler_overhead_share,
        "scheduler_overhead_evidence": list(assessment.scheduler_overhead_evidence),
        "callable_evidence": [
            _source_callable_evidence_report(evidence) for evidence in assessment.callable_evidence
        ],
        "rejections": list(assessment.rejections),
        "headroom_speedup": assessment.headroom_speedup,
    }


def _source_callable_evidence_report(
    evidence: SourceCallableEvidence,
) -> CompilationSourceOptimizationCallableEvidenceReport:
    """Serialize callable-level static and runtime source optimization evidence.

    Args:
        evidence: Callable evidence attached to an assessment.

    Returns:
        CompilationSourceOptimizationCallableEvidenceReport: JSON-compatible evidence.
    """
    return {
        "symbol": evidence.symbol.stable_id,
        "static_role": evidence.static_role,
        "observed_invocations": evidence.observed_invocations,
        "completed_calls": evidence.completed_calls,
        "static_suspension_points": evidence.static_suspension_points,
        "observed_suspensions": evidence.observed_suspensions,
        "immediate_result_ratio": evidence.immediate_result_ratio,
        "median_seconds": evidence.median_seconds,
        "hot_share": evidence.hot_share,
        "scheduler_overhead_samples": evidence.scheduler_overhead_samples,
        "task_introspection": list(evidence.task_introspection),
        "cancellation": list(evidence.cancellation),
        "context_mutation": list(evidence.context_mutation),
        "unknown_dynamic_calls": list(evidence.unknown_dynamic_calls),
        "hazards": list(evidence.hazards),
    }


def _source_optimization_trial_report(
    root: Path,
    trial: SourceOptimizationTrial,
) -> CompilationSourceOptimizationTrialReport:
    """Serialize one source-optimization semantic and benchmark trial.

    Args:
        root: Root directory used to normalize patch paths.
        trial: Trial evidence for a source-optimization candidate.

    Returns:
        CompilationSourceOptimizationTrialReport: JSON-compatible trial evidence.
    """
    return {
        "plan_id": trial.plan_id,
        "status": trial.status,
        "semantic_command": list(trial.semantic_command),
        "benchmark_command": list(trial.benchmark_command),
        "baseline_median_seconds": trial.baseline_median_seconds,
        "current_median_seconds": trial.current_median_seconds,
        "source_median_seconds": trial.source_median_seconds,
        "wheel_median_seconds": trial.wheel_median_seconds,
        "source_speedup": trial.source_speedup,
        "wheel_speedup": trial.wheel_speedup,
        "patch_path": _path_text(root, trial.patch_path) if trial.patch_path is not None else None,
        "source_edits": [_source_edit_report(edit) for edit in trial.source_edits],
        "application_status": trial.application_status,
        "diagnostics": list(trial.diagnostics),
        "candidate_id": trial.candidate_id,
        "transformation_ids": list(trial.transformation_ids),
        "reason": trial.reason,
        "semantic_exit_code": trial.semantic_exit_code,
        "semantic_duration_seconds": trial.semantic_duration_seconds,
        "residual_profile": (
            _compilation_profile_report(trial.residual_profile)
            if trial.residual_profile is not None
            else None
        ),
    }


def _source_edit_report(edit: SourceEdit) -> CompilationSourceOptimizationEditReport:
    """Serialize one generated source edit.

    Args:
        edit: Source edit generated for a source-optimization trial.

    Returns:
        CompilationSourceOptimizationEditReport: JSON-compatible edit evidence.
    """
    return {
        "path": edit.path.as_posix(),
        "before_hash": edit.before_hash,
        "after_hash": edit.after_hash,
        "summary": edit.summary,
        "touched_symbols": [symbol.stable_id for symbol in edit.touched_symbols],
        "transformation_id": edit.transformation_id,
        "start_line": edit.start_line,
        "end_line": edit.end_line,
    }


def _compilation_profile_report(result: ProfileResult | None) -> CompilationProfileReport:
    """Serialize profile evidence without retaining runtime values or object reprs.

    Args:
        result: Profile evidence collected by the runtime profiler, or `None` when
            no benchmark command was configured for candidate selection.

    Returns:
        CompilationProfileReport: Stable profile-guided selection section.
    """
    if result is None:
        return {
            "status": "unconfigured",
            "reason": "no benchmark command configured; static candidate evidence only",
            "launch_kind": "unconfigured",
            "sampling_policy": _profile_sampling_policy_report(),
            "total_samples": 0,
            "mapped_project_samples": 0,
            "mapped_coverage": 0.0,
            "scheduler_overhead_samples": 0,
            "scheduler_overhead_coverage": 0.0,
            "selected_hot_samples": 0,
            "selected_hot_coverage": 0.0,
            "child_passes": [],
            "lifecycle": _empty_profile_lifecycle_report(),
            "members": [],
            "spawn_sites": [],
            "broad_invocations": {
                "observed_events": 0,
                "event_limit": 0,
                "member_limit": 0,
                "capped": False,
            },
            "candidate_mapping_decisions": [],
            "selected_symbols": [],
        }
    return {
        "status": result.status,
        "reason": result.reason,
        "launch_kind": result.launch_kind,
        "sampling_policy": _profile_sampling_policy_report(),
        "total_samples": result.total_samples,
        "mapped_project_samples": result.mapped_project_samples,
        "mapped_coverage": result.mapped_coverage,
        "scheduler_overhead_samples": result.scheduler_overhead_samples,
        "scheduler_overhead_coverage": result.scheduler_overhead_coverage,
        "selected_hot_samples": result.selected_hot_samples,
        "selected_hot_coverage": result.selected_hot_coverage,
        "child_passes": [
            {
                "pass_kind": run.pass_kind,
                "command": list(run.command),
                "returncode": run.returncode,
                "duration_seconds": run.duration_seconds,
            }
            for run in result.runs
        ],
        "lifecycle": _profile_lifecycle_report(result.lifecycle),
        "members": [
            {
                "module": member.module,
                "qualname": member.qualname,
                "symbol": member.symbol.stable_id,
                "samples": member.samples,
                "coverage": member.coverage,
                "call_count": member.call_count,
                "invocation_count": member.invocation_count,
                "lifecycle": _profile_lifecycle_report(member.lifecycle),
                "signatures": [
                    {
                        "parameters": [
                            {
                                "parameter_name": parameter.parameter_name,
                                "type_path": parameter.type_path,
                                "count": parameter.count,
                            }
                            for parameter in signature.parameters
                        ],
                        "count": signature.count,
                    }
                    for signature in member.signatures
                ],
                "polymorphic": member.polymorphic_overflow,
                "observation_capped": member.observation_capped,
                "completed_calls": member.completed_calls,
                "max_active_calls": member.max_active_calls,
                "pre_completion_suspensions": member.pre_completion_suspensions,
                "scheduler_overhead_samples": member.scheduler_overhead_samples,
                "scheduler_overhead_coverage": member.scheduler_overhead_coverage,
                "immediate_result_ratio": member.immediate_result_ratio,
                "invocation_lower_bound": member.invocation_lower_bound,
                "invocation_upper_bound": member.invocation_upper_bound,
            }
            for member in result.members
        ],
        "spawn_sites": [
            {
                "id": site.target.id,
                "owner": site.target.owner.stable_id,
                "lineno": site.target.lineno,
                "col_offset": site.target.col_offset,
                "end_lineno": site.target.end_lineno,
                "end_col_offset": site.target.end_col_offset,
                "scheduler_method": site.target.scheduler_method,
                "invocation_count": site.invocation_count,
                "callable_identities": [
                    {"identity": callable_count.identity, "count": callable_count.count}
                    for callable_count in site.callable_identities
                ],
            }
            for site in result.spawn_sites
        ],
        "broad_invocations": {
            "observed_events": (
                result.broad_invocations.observed_events
                if result.broad_invocations is not None
                else 0
            ),
            "event_limit": (
                result.broad_invocations.event_limit if result.broad_invocations is not None else 0
            ),
            "member_limit": (
                result.broad_invocations.member_limit if result.broad_invocations is not None else 0
            ),
            "capped": (
                result.broad_invocations.capped if result.broad_invocations is not None else False
            ),
        },
        "candidate_mapping_decisions": [
            {
                "symbol": candidate.symbol.stable_id if candidate.symbol is not None else None,
                "module": candidate.module,
                "qualname": candidate.qualname,
                "samples": candidate.samples,
                "coverage": candidate.coverage,
                "scheduler_overhead_samples": candidate.scheduler_overhead_samples,
                "attributed_samples": candidate.attributed_samples,
                "attributed_coverage": candidate.attributed_coverage,
                "selected": candidate.selected,
                "reason": candidate.reason,
                "invocation_lower_bound": candidate.invocation_lower_bound,
                "invocation_upper_bound": candidate.invocation_upper_bound,
                "invocation_coverage": candidate.invocation_coverage,
                "selection_basis": candidate.selection_basis,
            }
            for candidate in result.candidates
        ],
        "selected_symbols": [symbol.stable_id for symbol in result.selected_symbols],
    }


def _profile_sampling_policy_report() -> CompilationProfileSamplingPolicyReport:
    return {"interval_ms": 2, "mode": "statistical leaf-frame sampling"}


def _empty_profile_lifecycle_report() -> CompilationProfileLifecycleReport:
    return {"start": 0, "return_": 0, "yield_": 0, "resume": 0, "unwind": 0, "throw": 0}


def _profile_lifecycle_report(lifecycle: LifecycleCounts) -> CompilationProfileLifecycleReport:
    return {
        "start": lifecycle.start,
        "return_": lifecycle.return_,
        "yield_": lifecycle.yield_,
        "resume": lifecycle.resume,
        "unwind": lifecycle.unwind,
        "throw": lifecycle.throw,
    }


def _backend_decision_reports(
    assessments: tuple[BackendAssessment, ...],
) -> list[CompilationBackendDecisionReport]:
    return [
        {
            "region_id": assessment.region_id,
            "backend": assessment.backend,
            "status": assessment.status,
            "supported_members": [symbol.stable_id for symbol in assessment.supported_members],
            "unsupported_members": [symbol.stable_id for symbol in assessment.unsupported_members],
            "capabilities": list(assessment.capabilities),
            "reasons": list(assessment.reasons),
            "deterministic": assessment.deterministic,
        }
        for assessment in sorted(
            assessments,
            key=lambda item: (item.region_id, item.backend, item.status),
        )
    ]


def _suspension_plan_reports(
    regions: tuple[TypedRegion, ...],
    compiled_regions: tuple[TypedRegion, ...],
    compiled_variants: tuple[CompiledRegionVariant, ...],
    compiled_bindings: tuple[BindingTarget, ...],
) -> list[CompilationSuspensionPlanReport]:
    compiled_by_source = {
        binding.source: variant
        for variant in sorted(compiled_variants, key=lambda item: item.id)
        for binding in variant.bindings
    }
    legacy_compiled_sources: set[SymbolId] = (
        {member.id for region in compiled_regions for member in region.members}
        if not compiled_variants
        else set()
    )
    legacy_compiled_sources.update(binding.source for binding in compiled_bindings)
    reports: list[CompilationSuspensionPlanReport] = []
    for region in sorted(regions, key=lambda item: item.id):
        for member in region.members:
            if not member.suspension_points:
                continue
            variant = compiled_by_source.get(member.id)
            compiled = variant is not None or member.id in legacy_compiled_sources
            lowering_mode = (
                variant.lowering_mode
                if variant is not None
                else "whole-callable"
                if compiled
                else "interpreted"
            )
            native_helpers = list(variant.native_helpers) if variant is not None else []
            suspension_plan = plan_suspension_blocks(member)
            reports.append(
                {
                    "id": f"{region.id}:{member.id.stable_id}",
                    "region_id": region.id,
                    "member": member.id.stable_id,
                    "execution_kind": member.execution_kind,
                    "lowering_mode": lowering_mode,
                    "native_helpers": native_helpers,
                    "points": [
                        _suspension_point_report(point) for point in member.suspension_points
                    ],
                    "blocks": [
                        _suspension_block_report(
                            block,
                            effective_eligible=block.id in suspension_plan.eligible_block_ids,
                        )
                        for block in suspension_plan.blocks
                    ],
                    "rejections": [
                        _suspension_rejection_report(rejection)
                        for rejection in suspension_plan.rejections
                    ],
                    "reason": (
                        "the Python shell retains suspension and delegates synchronous blocks "
                        "to verified native helpers"
                        if lowering_mode == "outlined-block"
                        else (
                            "the selected backend compiled the complete callable while "
                            "preserving its suspension protocol"
                            if compiled
                            else "no accepted compiled binding replaced this callable"
                        )
                    ),
                }
            )
    return reports


def _suspension_block_report(
    block: SuspensionBlock,
    *,
    effective_eligible: bool,
) -> CompilationSuspensionBlockReport:
    return {
        "id": block.id,
        "start_lineno": block.start_lineno,
        "start_col_offset": block.start_col_offset,
        "end_lineno": block.end_lineno,
        "end_col_offset": block.end_col_offset,
        "live_ins": list(block.live_ins),
        "live_outs": list(block.live_outs),
        "late_bound_globals": list(block.late_bound_globals),
        "receiver_dependencies": list(block.receiver_dependencies),
        "loop_count": block.loop_count,
        "operation_count": block.operation_count,
        "eligible": effective_eligible,
        "rejections": [_suspension_rejection_report(rejection) for rejection in block.rejections],
    }


def _suspension_rejection_report(
    rejection: RejectionEvidence,
) -> CompilationSuspensionRejectionReport:
    return {
        "code": rejection.code,
        "message": rejection.message,
        "lineno": rejection.lineno,
    }


def _accepted_variant_reports(
    compiled_regions: list[CompilationCompiledRegionReport],
) -> list[CompilationAcceptedVariantReport]:
    return [
        {
            "region_id": region["id"],
            "variant_id": region["variant_id"],
            "source_module": region["source_module"],
            "backend": region["backend"],
            "cache_status": region["cache_status"],
            "lowering_mode": region["lowering_mode"],
            "native_helpers": list(region["native_helpers"]),
            "symbols": sorted(binding["source"] for binding in region["bindings"]),
            "artifacts": list(region["artifacts"]),
        }
        for region in sorted(
            compiled_regions,
            key=lambda item: (item["id"], item["variant_id"], item["source_module"]),
        )
    ]


def _rejected_variant_reports(
    skipped_modules: tuple[CompilationSkippedModuleInput, ...],
) -> list[CompilationRejectedVariantReport]:
    return [
        {"module": skipped.module, "reason": skipped.reason}
        for skipped in sorted(skipped_modules, key=lambda item: (item.module, item.reason))
    ]


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


def _inline_code_list(values: list[str]) -> str:
    """Render a compact list of stable IDs or report paths.

    Args:
        values: Stable identifiers or paths to format as inline code.

    Returns:
        Comma-separated inline-code values, or `none` for an empty input.
    """
    return ", ".join(f"`{value}`" for value in values) if values else "none"


def _optional_seconds(value: float | None) -> str:
    return f"{value:.3f}s" if value is not None else "unknown"


def _optional_speedup(value: float | None) -> str:
    return f"{value:.3f}x" if value is not None else "unavailable"


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
