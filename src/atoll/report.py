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
from atoll.runtime.package_verify import PackageVerificationResult
from atoll.runtime.performance import BenchmarkGateResult, CommandRunEvidence
from atoll.runtime.profiling import LifecycleCounts, ProfileResult

_STRONG_SCORE = 90
_GOOD_SCORE = 80
_POSSIBLE_SCORE = 70
_ATOLL_PART_INDEX = 0
_ATOLL_GENERATED_INPUT_DIR_INDEX = 1


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
        lifecycle: Python lifecycle event counts for this member.
        signatures: Canonical argument-type signatures observed for this member.
        polymorphic: Whether the member exceeded the retained signature budget.
        observation_capped: Whether targeted observation reached its call budget.
    """

    module: str
    qualname: str
    symbol: str
    samples: int
    coverage: float
    call_count: int
    lifecycle: CompilationProfileLifecycleReport
    signatures: list[CompilationProfileSignatureReport]
    polymorphic: bool
    observation_capped: bool


class CompilationProfileCandidateDecisionReport(TypedDict):
    """Profile-to-static mapping and candidate policy decision.

    Attributes:
        symbol: Stable static symbol when mapping succeeded.
        module: Runtime module name observed in the profile.
        qualname: Runtime qualified name observed in the profile.
        samples: Statistical samples mapped to this member.
        coverage: Fraction of total workload samples represented by this member.
        selected: Whether the member passed the candidate policy.
        reason: Deterministic selection or rejection reason.
    """

    symbol: str | None
    module: str
    qualname: str
    samples: int
    coverage: float
    selected: bool
    reason: str


class CompilationProfileReport(TypedDict):
    """Profile-guided selection evidence for compile report schema v3.

    Attributes:
        status: Profile status describing dynamic evidence or static fallback.
        reason: Human-readable explanation for the current profile status.
        launch_kind: Supported launch shape used for child execution.
        sampling_policy: Stable sampling interval and sampling mode.
        total_samples: Statistical samples collected across the benchmark.
        mapped_project_samples: Samples mapped to configured project modules.
        mapped_coverage: Fraction of samples mapped to configured project modules.
        selected_hot_samples: Samples covered by selected candidates.
        selected_hot_coverage: Fraction of mapped samples covered by selected candidates.
        child_passes: Child-process profiling pass evidence.
        lifecycle: Aggregate Python lifecycle event counts.
        members: Profiled project members with sample and type evidence.
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
    selected_hot_samples: int
    selected_hot_coverage: float
    child_passes: list[CompilationProfilePassReport]
    lifecycle: CompilationProfileLifecycleReport
    members: list[CompilationProfileMemberReport]
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
        profile: Profile-guided selection evidence or explicit static fallback.
        candidate_trials: Candidate variants evaluated by profitability selection.
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
    profile: CompilationProfileReport
    candidate_trials: list[CompilationCandidateTrialReport]
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
    test evidence instead of trusting a caller-supplied status. Paths are kept as
    `Path` objects until rendering so they can be normalized relative to `root`.

    Attributes:
        root: Root directory of the target Python project.
        operation: Build or compile operation represented by the report.
        module_filter: Optional module restriction applied to compilation.
        islands: Enabled islands included in the operation or report.
        build: Captured native build evidence.
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
        profile: Profile-guided candidate evidence, or `None` for explicit static fallback.
        candidate_trials: Greedy marginal-profitability decisions in profile order.
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
    profile: ProfileResult | None = None
    candidate_trials: tuple[CandidateTrial, ...] = ()


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
    profile = _compilation_profile_report(report_input.profile)
    candidate_trials = _candidate_trial_reports(report_input.candidate_trials)
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
        "version": 3,
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
        "profile": profile,
        "candidate_trials": candidate_trials,
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
        f"- Build duration: {report['summary']['duration_seconds']:.3f}s",
        "",
        "## Verification Scope",
        "",
        _verification_scope_text(report["mode"]),
        "",
    ]
    _append_profile_guided_selection_markdown(lines, report["profile"])
    _append_candidate_trials_markdown(lines, report["candidate_trials"])
    _append_suspension_plans_markdown(lines, report["suspension_plans"])
    if report["typed_regions"]:
        lines.extend(["## Planned Regions", ""])
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


def _append_profile_guided_selection_markdown(
    lines: list[str],
    profile: CompilationProfileReport,
) -> None:
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
            f"- Selected hot coverage: {profile['selected_hot_coverage']:.1%}",
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
        }
        for trial in trials
    ]


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
            "selected_hot_samples": 0,
            "selected_hot_coverage": 0.0,
            "child_passes": [],
            "lifecycle": _empty_profile_lifecycle_report(),
            "members": [],
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
            }
            for member in result.members
        ],
        "candidate_mapping_decisions": [
            {
                "symbol": candidate.symbol.stable_id if candidate.symbol is not None else None,
                "module": candidate.module,
                "qualname": candidate.qualname,
                "samples": candidate.samples,
                "coverage": candidate.coverage,
                "selected": candidate.selected,
                "reason": candidate.reason,
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
