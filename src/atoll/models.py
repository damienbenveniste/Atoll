"""Typed contracts shared by Atoll analysis, generation, build, and reporting.

The dataclasses in this module are immutable handoff objects. They keep raw
source facts, conservative analysis decisions, and runtime/build evidence
separate so command handlers can enrich data without mutating earlier phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Literal

ArtifactRole = Literal["primary", "support"]
Backend = Literal["mypyc", "cython"]
BackendAssessmentStatus = Literal["supported", "partial", "unsupported"]
BackendCapability = Literal[
    "typed_function",
    "instance_method",
    "staticmethod",
    "classmethod",
    "native_class",
    "generator",
    "coroutine",
    "async_generator",
]
BackendDiagnosticCode = Literal[
    "MYPYC_TYPE_ERROR",
    "CYTHON_COMPILE_ERROR",
    "NATIVE_BUILD_ENV_ERROR",
    "IMPORT_PATH_ERROR",
    "UNKNOWN_BUILD_ERROR",
]
BindingKind = Literal["module", "class", "instance_method", "staticmethod", "classmethod"]
BlockerSeverity = Literal["hard", "soft", "info"]
CompileCacheStatus = Literal["disabled", "hit", "miss", "partial"]
Confidence = Literal["high", "medium", "low"]
ConstantKind = Literal["literal_constant", "runtime_dynamic", "unknown"]
DependencyKind = Literal[
    "calls",
    "calls_method",
    "uses_global",
    "inherits",
    "decorated_by",
    "imports",
    "annotation",
    "unknown",
]
DependencyRole = Literal["runtime", "typing", "facade"]
DiagnosticSeverity = Literal["error", "note"]
ExecutionKind = Literal["sync", "generator", "coroutine", "async_generator", "class"]
InvocationMode = Literal["ordinary", "awaited", "async_iteration"]
IslandRisk = Literal["low", "medium", "high"]
LossAction = Literal["preserve", "specialize", "box", "fallback", "reject"]
LoweringMode = Literal["whole-callable", "outlined-block"]
ParameterKind = Literal[
    "positional_only",
    "positional",
    "vararg",
    "keyword_only",
    "kwarg",
]
SymbolKind = Literal["function", "class", "method"]
TypeParameterKind = Literal["type_var", "param_spec", "type_var_tuple"]
TypeBindingSource = Literal[
    "parameter",
    "return",
    "field",
    "base",
    "type_parameter",
    "import",
]
SpecializationOrigin = Literal["concrete_subclass", "closed_call"]
SuspensionKind = Literal["await", "yield", "yield_from", "async_for", "async_with"]
Visibility = Literal["public", "private"]

_SHA256_HEX_LENGTH = 64
_LOWERCASE_HEX_DIGITS = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class CompileConfig:
    """Validated source-clean compiler, test, and benchmark configuration.

    Commands remain argv tuples and are never interpreted by a shell. A
    benchmark requires a semantic test command because Atoll must prove both
    baseline and compiled behavior before measuring profitability.

    Attributes:
        backends: Native backends attempted in configured preference order.
        test_command: Optional target-project semantic test command.
        benchmark_command: Optional command used for paired performance measurements.
        benchmark_warmups: Number of unmeasured warmup pairs run before sampling.
        benchmark_samples: Number of measured baseline/compiled sample pairs.
        minimum_speedup: Smallest acceptable compiled-to-baseline speedup ratio.
    """

    backends: tuple[Backend, ...] = ("mypyc", "cython")
    test_command: tuple[str, ...] | None = None
    benchmark_command: tuple[str, ...] | None = None
    benchmark_warmups: int = 1
    benchmark_samples: int = 7
    minimum_speedup: float = 1.10

    def __post_init__(self) -> None:
        """Reject incomplete or ambiguous compile policy at discovery time.

        Raises:
            ValueError: If backends are empty or duplicated, commands are malformed, benchmark
                policy is incomplete, counts are invalid, or the speedup threshold is not positive.
        """
        if not self.backends or len(set(self.backends)) != len(self.backends):
            raise ValueError("tool.atoll.compile.backends must be a non-empty unique list")
        if any(backend not in {"mypyc", "cython"} for backend in self.backends):
            raise ValueError("tool.atoll.compile.backends supports only mypyc and cython")
        for field_name, command in (
            ("test_command", self.test_command),
            ("benchmark_command", self.benchmark_command),
        ):
            if command is not None and (not command or any(not part.strip() for part in command)):
                raise ValueError(f"tool.atoll.compile.{field_name} must be a non-empty argv list")
        if self.benchmark_command is not None and self.test_command is None:
            raise ValueError("tool.atoll.compile.benchmark_command requires test_command")
        if self.benchmark_warmups < 0:
            raise ValueError("tool.atoll.compile.benchmark_warmups must be at least 0")
        if self.benchmark_samples < 1:
            raise ValueError("tool.atoll.compile.benchmark_samples must be at least 1")
        if self.minimum_speedup <= 0:
            raise ValueError("tool.atoll.compile.minimum_speedup must be greater than 0")


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Resolved project settings for one Atoll invocation.

    Paths are absolute after discovery. `source_roots` controls importable module
    discovery, while `cache_dir` and `report_dir` are Atoll-owned output
    locations under the project root unless configuration explicitly overrides
    them.

    Attributes:
        root: Root directory of the target Python project.
        source_roots: Absolute import roots discovered for the target project.
        backend: Compatibility default backend used by legacy in-place workflows.
        cache_dir: Directory containing reusable Atoll cache entries.
        report_dir: Directory containing Atoll JSON and Markdown reports.
        islands: Persisted managed islands used by legacy in-place workflows.
        compile: Compile and quality-gate configuration for the project.
    """

    root: Path
    source_roots: tuple[Path, ...]
    backend: Backend
    cache_dir: Path
    report_dir: Path
    islands: tuple[EnabledIslandConfig, ...] = ()
    compile: CompileConfig = field(default_factory=CompileConfig)


@dataclass(frozen=True, slots=True)
class EnabledIslandConfig:
    """Persistent configuration for one Atoll-managed source module.

    The source module keeps the managed shim, the sidecar module contains copied
    symbols, and `symbols` names the exported top-level functions that should be
    rebound at runtime. Disabled islands remain in configuration for auditability
    but are skipped by generation, build, and verification commands.

    Attributes:
        source_module: Importable source module name.
        source_path: Filesystem path of the source module or prepared source.
        sidecar_module: Importable generated sidecar module name.
        sidecar_path: Filesystem path of the generated sidecar.
        symbols: Exported top-level symbols rebound through the managed sidecar.
        enabled: Whether this island participates in generation, build, and verification.
    """

    source_module: str
    source_path: Path
    sidecar_module: str
    sidecar_path: Path
    symbols: tuple[str, ...]
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class ModuleId:
    """Stable import name and filesystem path for a discovered Python module.

    The `name` is the importable dotted module name relative to a configured
    source root. The `path` points at the source file scanned for AST facts.

    Attributes:
        name: Importable dotted module name relative to its source root.
        path: Absolute path to the Python source file.
    """

    name: str
    path: Path


@dataclass(frozen=True, slots=True)
class SymbolId:
    """Stable symbol identity within a scanned module.

    `qualname` is limited to top-level functions/classes and simple
    `Class.method` names because Atoll V1 does not extract nested symbols. The
    `stable_id` property is the report-facing identifier used across JSON and
    Markdown output.

    Attributes:
        module: Importable dotted module containing the symbol.
        qualname: Module-local qualified symbol name.
    """

    module: str
    qualname: str

    @property
    def stable_id(self) -> str:
        """Return the stable report-facing `module::qualname` identifier text used in reports.

        Returns:
            str: Stable `module::qualname` identifier used in reports and cache records.
        """
        return f"{self.module}::{self.qualname}"


@dataclass(frozen=True, slots=True)
class Blocker:
    """Conservative reason a module or symbol should not be compiled blindly.

    Hard blockers prevent candidate extraction, while soft and info blockers
    explain risk that may still be useful in reports. `symbol` is absent only for
    module-level conditions that cannot be tied to a specific symbol.

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
    lineno: int | None = None
    symbol: SymbolId | None = None


@dataclass(frozen=True, slots=True)
class MypyDiagnostic:
    """One parsed mypy diagnostic, optionally mapped back to a scanned symbol.

    Paths are resolved before storage so diagnostics can be grouped without
    depending on the current working directory. A mapped error also becomes a
    hard blocker on the owning symbol during type-readiness enrichment.

    Attributes:
        path: Absolute path to the source file reported by mypy.
        line: One-based diagnostic line.
        column: One-based diagnostic column, when mypy reports one.
        severity: Diagnostic severity used for filtering and reporting.
        code: Optional mypy error code, excluding brackets.
        message: Diagnostic message with location and code removed.
        symbol: Scanned symbol owning the diagnostic, when range mapping succeeds.
    """

    path: Path
    line: int
    column: int | None
    severity: DiagnosticSeverity
    code: str | None
    message: str
    symbol: SymbolId | None = None


@dataclass(frozen=True, slots=True)
class ImportRecord:
    """Top-level import statement captured exactly enough for sidecar copying.

    `source_text` preserves the original statement for generated sidecars, while
    `imported_names` records the names made available in the module namespace for
    dependency analysis.

    Attributes:
        source_text: Exact source text retained for analysis or generation.
        imported_names: Names introduced into the module namespace by the import.
        module: Imported module path, or `None` for imports without one.
        level: Relative import level; zero denotes an absolute import.
        lineno: One-based first source line covered by the record.
        end_lineno: One-based final source line covered by the record.
    """

    source_text: str
    imported_names: tuple[str, ...]
    module: str | None
    level: int
    lineno: int
    end_lineno: int


@dataclass(frozen=True, slots=True)
class ConstantRecord:
    """Top-level assignment classified for safe reuse in generated sidecars.

    Literal constants can be copied into sidecars. Dynamic or unknown constants
    remain visible to analysis so users can see why a global dependency blocked
    or raised the risk of extraction.

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


@dataclass(frozen=True, slots=True)
class ParameterRecord:
    """Exact source-level parameter facts retained for typed-region planning.

    Annotation and default text are preserved rather than interpreted so a
    backend can make its own semantic decision without inheriting an earlier
    lossy rewrite. `default_source` is absent for required parameters.

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


@dataclass(frozen=True, slots=True)
class FieldRecord:
    """Typed class field declaration retained for class-region planning.

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


@dataclass(frozen=True, slots=True)
class TypeParameterRecord:
    """Exact declaration of one PEP 695 or legacy typing parameter.

    `declaration` retains the complete source expression, including bounds,
    constraints, variance keywords, and defaults. `name` and `kind` provide the
    structured identity needed for scope and specialization decisions.

    Attributes:
        name: Type parameter name visible in source.
        kind: `TypeVar`, `ParamSpec`, or `TypeVarTuple` classification.
        declaration: Exact source declaration for the type parameter.
    """

    name: str
    kind: TypeParameterKind
    declaration: str


@dataclass(frozen=True, slots=True)
class CallSiteFact:
    """One syntactic call expression observed inside a scanned symbol body.

    The scanner records the call target expression and source position without
    evaluating arguments, receivers, descriptors, or runtime values.
    `invocation_mode` separates ordinary calls from calls that are directly
    awaited or used as an async-iteration source. `requires_same_unit` is
    reserved for syntax that proves a call cannot be split across compilation
    units; it defaults to `False` because most local calls can remain normal
    runtime boundaries.

    Attributes:
        target: Source-level call target path, such as `helper` or `self.step`.
        root_name: First lexical name in the target path used for boundary lookup.
        invocation_mode: How the call expression participates in control flow.
        lineno: One-based first source line covered by the call.
        end_lineno: One-based final source line covered by the call.
        col_offset: Zero-based first source column covered by the call.
        end_col_offset: Zero-based final source column covered by the call, when available.
        requires_same_unit: Whether syntax proves this call must share a compiled unit.
    """

    target: str
    root_name: str
    invocation_mode: InvocationMode
    lineno: int
    end_lineno: int
    col_offset: int
    end_col_offset: int | None
    requires_same_unit: bool = False


@dataclass(frozen=True, slots=True)
class SuspensionPoint:
    """One syntax-level suspension boundary inside a scanned symbol body.

    Suspension points describe control-flow shape, not the values yielded,
    delegated to, or awaited through protocol methods. Source positions are
    retained so later reports can explain why a function cannot be treated as a
    purely synchronous body.

    Attributes:
        kind: Syntax form that can suspend execution.
        lineno: One-based first source line covered by the suspension point.
        end_lineno: One-based final source line covered by the suspension point.
        col_offset: Zero-based first source column covered by the suspension point.
        end_col_offset: Zero-based final source column covered by the suspension point, when
            available.
    """

    kind: SuspensionKind
    lineno: int
    end_lineno: int
    col_offset: int
    end_col_offset: int | None


@dataclass(frozen=True, slots=True)
class SymbolRecord:
    """AST-derived facts for one function, class, or simple method.

    The scanner records source locations, type-readiness signals, referenced
    names, and local blockers without executing project code. Later phases attach
    mypy diagnostics, dependency edges, and candidate decisions by replacing this
    immutable record.

    Attributes:
        id: Module and qualified-name identity for the declaration.
        kind: Function, class, or method declaration kind.
        visibility: Public or private source visibility.
        lineno: One-based first source line covered by the record.
        end_lineno: One-based final source line covered by the record.
        col_offset: Zero-based source column where the declaration starts.
        end_col_offset: Zero-based source column where the declaration ends, when available.
        decorators: Source text for decorators applied to the symbol.
        arg_count: Total caller-visible parameter count.
        annotated_arg_count: Number of parameters with explicit annotations.
        has_return_annotation: Whether the callable declares a return annotation.
        has_any_annotation: Whether any visible annotation contains `Any`.
        called_names: Simple names observed in call position.
        uses_globals: Module globals read by the symbol body.
        local_names: Names bound locally within the symbol body.
        referenced_names: All names read by the symbol body or annotations.
        blockers: Conservative blockers attached to this module or symbol.
        mypy_diagnostics: Mypy diagnostics mapped to this module or symbol.
        owner_class: Source owner class for a method binding, when applicable.
        binding_kind: Runtime descriptor or module binding classification.
        execution_kind: Synchronous, generator, coroutine, async-generator, or class shape.
        type_parameters: Type parameter names declared directly by the symbol.
        parameters: Exact source parameter declarations in call order.
        return_annotation: Exact source return annotation, when present.
        annotation_names: Names referenced by source annotations.
        called_paths: Dotted call targets recovered from source syntax.
        call_sites: Ordered source call facts observed in the symbol body.
        suspension_points: Ordered syntax-level suspension boundaries in the symbol body.
        runtime_imports: Ordered function-local imports executed at runtime.
        base_names: Base-class expressions referenced by the declaration.
        fields: Typed class fields retained for region planning.
        declaration_start_lineno: First line of decorators or declaration syntax.
        scope_type_parameters: Type parameter names inherited from enclosing scopes.
        type_parameter_records: Structured type parameters declared directly by the symbol.
        scope_type_parameter_records: Structured type parameters inherited from enclosing scopes.
        any_annotation_sources: Locations where `Any` enters the symbol's type surface.
    """

    id: SymbolId
    kind: SymbolKind
    visibility: Visibility
    lineno: int
    end_lineno: int
    col_offset: int
    end_col_offset: int | None
    decorators: tuple[str, ...]
    arg_count: int
    annotated_arg_count: int
    has_return_annotation: bool
    has_any_annotation: bool
    called_names: tuple[str, ...]
    uses_globals: tuple[str, ...]
    local_names: tuple[str, ...]
    referenced_names: tuple[str, ...]
    blockers: tuple[Blocker, ...]
    mypy_diagnostics: tuple[MypyDiagnostic, ...] = field(default_factory=tuple)
    owner_class: str | None = None
    binding_kind: BindingKind = "module"
    execution_kind: ExecutionKind = "sync"
    type_parameters: tuple[str, ...] = ()
    parameters: tuple[ParameterRecord, ...] = ()
    return_annotation: str | None = None
    annotation_names: tuple[str, ...] = ()
    called_paths: tuple[str, ...] = ()
    call_sites: tuple[CallSiteFact, ...] = ()
    suspension_points: tuple[SuspensionPoint, ...] = ()
    runtime_imports: tuple[ImportRecord, ...] = ()
    base_names: tuple[str, ...] = ()
    fields: tuple[FieldRecord, ...] = ()
    declaration_start_lineno: int | None = None
    scope_type_parameters: tuple[str, ...] = ()
    type_parameter_records: tuple[TypeParameterRecord, ...] = ()
    scope_type_parameter_records: tuple[TypeParameterRecord, ...] = ()
    any_annotation_sources: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RuntimeTypeGuard:
    """Constant-time runtime type evidence for one specialized input.

    The guard describes only checks that can be performed from the object
    already passed by the caller. `positional_index` counts the original Python
    parameter position, including `self` or `cls` for methods; keyword-only
    parameters use `None` because there is no stable positional slot. The
    nominal paths are syntactic type names such as `int` or `models.Payload`,
    and `allow_none` records a separate `None` acceptance branch.

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
    nominal_type_paths: tuple[str, ...]
    allow_none: bool

    def __post_init__(self) -> None:
        """Reject guards that cannot describe a constant-time nominal check.

        Raises:
            ValueError: If the parameter identity, positional index, annotation, or nominal type
                paths cannot define a constant-time guard.
        """
        if not self.parameter_name.strip():
            raise ValueError("runtime type guard parameter_name must be non-empty")
        if self.positional_index is not None and self.positional_index < 0:
            raise ValueError("runtime type guard positional_index must be non-negative")
        if not self.annotation.strip():
            raise ValueError("runtime type guard annotation must be non-empty")
        if not self.allow_none and not self.nominal_type_paths:
            raise ValueError("runtime type guard must name a type or allow None")
        if any(not path.strip() for path in self.nominal_type_paths):
            raise ValueError("runtime type guard nominal paths must be non-empty")


@dataclass(frozen=True, slots=True)
class RegionSpecialization:
    """Concrete specialization evidence layered on an unchanged typed region.

    Specializations never rewrite the source member, original type bindings, or
    generic fallback decisions. They record a target owner or closed call that
    proves a concrete binding, the substitutions that made all emitted
    `type_bindings` concrete, and runtime guards required to dispatch to the
    specialized binding in constant time.

    Attributes:
        id: Deterministic specialization ID included in cache and report identities.
        source_member: Generic source member from which a specialization was derived.
        source_owner_class: Owner class declared by the generic source member.
        target_owner_class: Concrete runtime owner class for a specialized binding.
        origin: Resolved module origin or specialization evidence source.
        substitutions: Concrete type substitutions applied to generic parameters.
        guards: Runtime type guards required before selecting this binding or specialization.
        type_bindings: Preserved or concretized type evidence for the region.
    """

    id: str
    source_member: SymbolId
    source_owner_class: str | None
    target_owner_class: str | None
    origin: SpecializationOrigin
    substitutions: tuple[tuple[str, str], ...]
    guards: tuple[RuntimeTypeGuard, ...]
    type_bindings: tuple[TypeBinding, ...]

    def __post_init__(self) -> None:
        """Validate that specialization evidence is concrete and self-contained.

        Raises:
            ValueError: If the ID, substitutions, guards, bindings, or owner relationship is
                incomplete or contradictory.
        """
        if not self.id.strip():
            raise ValueError("region specialization id must be non-empty")
        if not self.substitutions:
            raise ValueError("region specialization requires at least one substitution")
        if any(
            not type_var.strip() or not annotation.strip()
            for type_var, annotation in self.substitutions
        ):
            raise ValueError("region specialization substitutions must be non-empty")
        names = tuple(type_var for type_var, _ in self.substitutions)
        if len(names) != len(set(names)):
            raise ValueError("region specialization substitutions must use unique TypeVars")
        if any(not binding.concrete for binding in self.type_bindings):
            raise ValueError("region specialization type bindings must be concrete")
        if self.origin == "concrete_subclass" and (
            self.source_owner_class is None
            or self.target_owner_class is None
            or self.source_owner_class == self.target_owner_class
        ):
            raise ValueError("concrete subclass specialization requires distinct source and target")
        if self.origin == "closed_call" and self.target_owner_class != self.source_owner_class:
            raise ValueError("closed-call specialization must retain its source owner")


@dataclass(frozen=True, slots=True)
class BindingTarget:
    """One source binding that a compiled region promises to replace.

    `source` is always the public source identity. `compiled_name` is private
    backend output and may differ for lowered methods or specializations.
    `target_owner_class` and `guards` are empty for ordinary bindings and are
    reserved for bindings that install a concrete specialization.

    Attributes:
        source: Public source symbol promised by the binding.
        compiled_name: Backend-generated attribute name containing the compiled callable.
        kind: Module, class, or descriptor-aware binding kind.
        owner_class: Source owner class for a method binding, when applicable.
        execution_kind: Synchronous, generator, coroutine, async-generator, or class shape.
        required: Whether absence of the compiled binding is a verification failure.
        target_owner_class: Concrete runtime owner class for a specialized binding.
        guards: Runtime type guards required before selecting this binding or specialization.
    """

    source: SymbolId
    compiled_name: str
    kind: BindingKind
    owner_class: str | None
    execution_kind: ExecutionKind
    required: bool = True
    target_owner_class: str | None = None
    guards: tuple[RuntimeTypeGuard, ...] = ()


@dataclass(frozen=True, slots=True)
class TypeBinding:
    """Preserved or concretized type evidence used by a typed region.

    Attributes:
        name: Parameter, return, field, base, type-parameter, or import name.
        annotation: Exact source annotation text.
        source: Category from which the type evidence was obtained.
        concrete: Whether the type evidence is fully concrete for lowering.
        substitutions: Concrete type substitutions applied to generic parameters.
    """

    name: str
    annotation: str
    source: TypeBindingSource
    concrete: bool
    substitutions: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class LoweringDecision:
    """Auditable decision about how one typed-region fact will be lowered.

    Attributes:
        target: Stable symbol or region fact affected by the decision.
        action: Lowering action chosen for the target.
        reason: Concrete evidence requiring preservation, specialization, boxing, fallback, or
            rejection.
    """

    target: str
    action: LossAction
    reason: str


@dataclass(frozen=True, slots=True)
class RegionMember:
    """Unlowered source declaration owned by a backend-neutral typed region.

    Attributes:
        id: Module and qualified-name identity for the retained declaration.
        kind: Function, class, or method declaration kind.
        owner_class: Source owner class for a method binding, when applicable.
        binding_kind: Runtime descriptor or module binding classification.
        execution_kind: Synchronous, generator, coroutine, async-generator, or class shape.
        source_text: Exact source text retained for analysis or generation.
        type_parameters: Type parameter names declared directly by the symbol.
        type_parameter_records: Structured type parameters declared directly by the symbol.
        scope_type_parameters: Type parameter names inherited from enclosing scopes.
        scope_type_parameter_records: Structured type parameters inherited from enclosing scopes.
        parameters: Exact source parameter declarations in call order.
        return_annotation: Exact source return annotation, when present.
        call_sites: Ordered source call facts retained for region planning.
        suspension_points: Ordered syntax-level suspension boundaries retained for backend
            planning.
        runtime_imports: Ordered function-local imports that execute at runtime.
        fields: Typed class fields retained for region planning.
    """

    id: SymbolId
    kind: SymbolKind
    owner_class: str | None
    binding_kind: BindingKind
    execution_kind: ExecutionKind
    source_text: str
    type_parameters: tuple[str, ...]
    type_parameter_records: tuple[TypeParameterRecord, ...]
    scope_type_parameters: tuple[str, ...]
    scope_type_parameter_records: tuple[TypeParameterRecord, ...]
    parameters: tuple[ParameterRecord, ...]
    return_annotation: str | None
    call_sites: tuple[CallSiteFact, ...] = ()
    suspension_points: tuple[SuspensionPoint, ...] = ()
    runtime_imports: tuple[ImportRecord, ...] = ()
    fields: tuple[FieldRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class RegionDependency:
    """Typed-region dependency retaining runtime versus annotation intent.

    Attributes:
        src: Source symbol for the dependency edge.
        dst: Dependency destination symbol or external boundary.
        kind: Calls, global use, inheritance, decoration, import, or annotation edge kind.
        confidence: Confidence assigned to the dependency evidence.
        role: Whether the dependency is required at runtime, for typing, or by the facade.
        type_only: Whether the dependency is used exclusively for typing.
        lineno: One-based source line for the dependency evidence, when available.
        invocation_mode: How the source call is invoked when this is a call edge.
        requires_same_unit: Whether syntax proves the dependency must share a compiled unit.
    """

    src: SymbolId
    dst: SymbolId | str
    kind: DependencyKind
    confidence: Confidence
    role: DependencyRole
    type_only: bool
    lineno: int | None = None
    invocation_mode: InvocationMode | None = None
    requires_same_unit: bool = False


@dataclass(frozen=True, slots=True)
class TypedRegion:
    """Backend-neutral connected source region before any type-losing rewrite.

    Region IDs and hashes are deterministic for the retained source and
    dependency evidence. Instances are safe to cache, compare, and pass to
    backend assessment without reading or importing target-project code.

    Attributes:
        id: Deterministic region ID derived from retained members and dependencies.
        source_module: Importable source module name.
        members: Source declarations owned by the typed region or compilation unit.
        dependencies: Runtime and typing dependencies retained by the region.
        type_bindings: Preserved or concretized type evidence for the region.
        bindings: Source bindings promised by the compiled region or variant.
        decisions: Auditable preservation, specialization, boxing, fallback, or rejection decisions.
        source_hash: Deterministic digest of generated or retained source.
        atomic_class: Whether the owner class must be lowered as one indivisible region.
        specializations: Guarded concrete variants available for the typed region.
    """

    id: str
    source_module: ModuleId
    members: tuple[RegionMember, ...]
    dependencies: tuple[RegionDependency, ...]
    type_bindings: tuple[TypeBinding, ...]
    bindings: tuple[BindingTarget, ...]
    decisions: tuple[LoweringDecision, ...]
    source_hash: str
    atomic_class: bool = False
    specializations: tuple[RegionSpecialization, ...] = ()


@dataclass(frozen=True, slots=True)
class BackendAssessment:
    """Deterministic capability decision for one backend and typed region.

    Partial assessments keep supported and unsupported members separate so one
    unsupported execution shape cannot poison independent members. Support
    means the backend can compile a correctly prepared member unit; it does not
    promise that runtime binding for that member has been implemented. Reasons
    are stable diagnostic text suitable for reports and backend selection.

    Attributes:
        region_id: Stable typed-region identifier owning the artifact or unit.
        backend: Native compiler backend selected for this record.
        status: Supported, partial, unsupported, or quality-gate status.
        supported_members: Region members accepted by the backend.
        unsupported_members: Region members rejected by the backend.
        capabilities: Backend capabilities exercised by supported members.
        reasons: Deterministically ordered evidence supporting the decision.
        deterministic: Whether identical inputs must produce this assessment.
    """

    region_id: str
    backend: Backend
    status: BackendAssessmentStatus
    supported_members: tuple[SymbolId, ...]
    unsupported_members: tuple[SymbolId, ...]
    capabilities: tuple[BackendCapability, ...]
    reasons: tuple[str, ...]
    deterministic: bool = True


@dataclass(frozen=True, slots=True)
class BackendLoweringRequest:
    """A prepared source file and member selection offered to one backend.

    Source generation remains separate from backend registration. An empty
    `members` tuple asks the backend to include every member it assessed as
    supported; explicit members must be a subset of that assessment.
    `variant_id` distinguishes backend-specific subsets of the same region and
    defaults to the semantic region ID for compatibility.

    Attributes:
        region: Backend-neutral typed region represented by this record.
        source_path: Filesystem path of the source module or prepared source.
        logical_module: Importable module name represented by the compilation unit.
        install_relative_dir: POSIX directory below the staged wheel payload root.
        members: Source declarations owned by the typed region or compilation unit.
        variant_id: Stable backend/specialization variant identifier.
    """

    region: TypedRegion
    source_path: Path
    logical_module: str
    install_relative_dir: str = ""
    members: tuple[SymbolId, ...] = ()
    variant_id: str | None = None


@dataclass(frozen=True, slots=True)
class CompilationUnit:
    """Backend-specific, content-addressable source unit ready to compile.

    Attributes:
        region_id: Stable typed-region identifier owning the artifact or unit.
        backend: Native compiler backend selected for this record.
        logical_module: Importable module name represented by the compilation unit.
        source_paths: Prepared source files compiled as one unit.
        source_hash: Deterministic digest of generated or retained source.
        members: Source declarations owned by the typed region or compilation unit.
        install_relative_dir: POSIX directory below the staged wheel payload root.
    """

    region_id: str
    backend: Backend
    logical_module: str
    source_paths: tuple[Path, ...]
    source_hash: str
    members: tuple[SymbolId, ...]
    install_relative_dir: str = ""


@dataclass(frozen=True, slots=True)
class BackendCompileContext:
    """Filesystem and import-path boundaries for a native backend invocation.

    `record_artifacts=False` is reserved for the legacy `build_sidecars` facade,
    whose callers consume only `CompileAttempt`. Typed-region compilation keeps
    the default strict artifact recording and validation. `backend_options` is
    normalized key/value evidence included in backend fingerprints.

    Attributes:
        project_root: Root directory of the target Python project.
        build_dir: Directory containing disposable native build inputs.
        source_roots: Absolute import roots discovered for the target project.
        cache_dir: Directory containing reusable Atoll cache entries.
        record_artifacts: Whether compilation must emit and validate artifact records.
        backend_options: Normalized backend options included in cache fingerprints.
    """

    project_root: Path
    build_dir: Path
    source_roots: tuple[Path, ...] = ()
    cache_dir: Path | None = None
    record_artifacts: bool = True
    backend_options: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """Install-facing identity and integrity metadata for a native artifact.

    `install_relative_path` is always POSIX-style and relative to the staged
    payload root. Support artifacts shared by multiple regions use the reserved
    region ID `__shared__` instead of being assigned to an arbitrary member.

    Attributes:
        region_id: Stable typed-region identifier owning the artifact or unit.
        backend: Native compiler backend selected for this record.
        logical_module: Importable module name represented by the compilation unit.
        role: Runtime, typing, facade, primary, or support role.
        install_relative_path: POSIX artifact path relative to the staged payload root.
        digest: Lowercase SHA-256 digest of artifact content.
        abi: Native ABI tag recorded for the artifact.
        platform_tag: Wheel platform tag for the native artifact.
    """

    region_id: str
    backend: Backend
    logical_module: str
    role: ArtifactRole
    install_relative_path: str
    digest: str
    abi: str
    platform_tag: str

    def __post_init__(self) -> None:
        """Reject absolute, parent-traversing, or non-POSIX install paths.

        Raises:
            ValueError: If install paths, digests, identifiers, ABI, or platform metadata are
                empty or unsafe.
        """
        path = PurePosixPath(self.install_relative_path)
        if (
            not self.install_relative_path
            or "\\" in self.install_relative_path
            or path.is_absolute()
            or ".." in path.parts
        ):
            raise ValueError("artifact install_relative_path must be a relative POSIX path")
        if len(self.digest) != _SHA256_HEX_LENGTH or any(
            character not in _LOWERCASE_HEX_DIGITS for character in self.digest
        ):
            raise ValueError("artifact digest must be a lowercase SHA-256 hex digest")
        if not self.region_id.strip() or not self.logical_module.strip():
            raise ValueError("artifact region_id and logical_module must be non-empty")
        if not self.abi.strip() or not self.platform_tag.strip():
            raise ValueError("artifact ABI and platform tag must be non-empty")


@dataclass(frozen=True, slots=True)
class CompiledRegionVariant:
    """One backend-specific compiled subset of a semantic typed region.

    A region can produce multiple variants when different members require
    different backends. ``id`` is unique across those variants and is the
    artifact/runtime status key; ``region.id`` remains the scanner identity.

    Attributes:
        id: Deterministic backend and specialization variant ID.
        region: Backend-neutral typed region represented by this record.
        backend: Native compiler backend selected for this record.
        bindings: Source bindings promised by the compiled region or variant.
        cache_status: Whether compilation used, missed, or partially restored cache state.
        lowering_mode: Whether the backend owns the complete callable or synchronous blocks.
        native_helpers: Private native helper names used by an outlined Python shell.
    """

    id: str
    region: TypedRegion
    backend: Backend
    bindings: tuple[BindingTarget, ...]
    cache_status: CompileCacheStatus = "disabled"
    lowering_mode: LoweringMode = "whole-callable"
    native_helpers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Require a non-empty variant ID and bindings owned by the region.

        Raises:
            ValueError: If the variant ID is empty or a promised binding is absent from the region.
        """
        if not self.id.strip():
            raise ValueError("compiled region variant ID must be non-empty")
        member_ids = {member.id for member in self.region.members}
        if not self.bindings or any(binding.source not in member_ids for binding in self.bindings):
            raise ValueError("compiled region variant bindings must belong to the region")
        if self.lowering_mode == "outlined-block" and not self.native_helpers:
            raise ValueError("outlined compiled region variants require native helpers")
        if self.lowering_mode == "whole-callable" and self.native_helpers:
            raise ValueError("whole-callable variants cannot declare outlined native helpers")
        if len(set(self.native_helpers)) != len(self.native_helpers):
            raise ValueError("compiled region variant native helpers must be unique")


@dataclass(frozen=True, slots=True)
class BackendDiagnostic:
    """Normalized compiler failure independent of backend exception types.

    Attributes:
        code: Stable machine-readable diagnostic or blocker code.
        message: Human-readable diagnostic or blocker explanation.
        details: Additional normalized diagnostic lines.
        log_path: Path to complete backend diagnostics, when retained.
        transient: Whether retrying in a corrected environment may succeed.
    """

    code: BackendDiagnosticCode
    message: str
    details: tuple[str, ...]
    log_path: Path | None
    transient: bool


@dataclass(frozen=True, slots=True)
class DependencyEdge:
    """Conservative dependency edge from one symbol to another boundary.

    Destinations are either another `SymbolId` in the same module or a string for
    imports, constants, and unresolved globals. `confidence` documents whether
    the edge is a direct AST fact or a lower-confidence boundary inference.

    Attributes:
        src: Source symbol for the dependency edge.
        dst: Dependency destination symbol or external boundary.
        kind: Calls, global use, inheritance, decoration, import, or annotation edge kind.
        confidence: Confidence assigned to the dependency evidence.
        lineno: One-based first source line covered by the record.
        invocation_mode: How the source call is invoked when this is a call edge.
        requires_same_unit: Whether syntax proves the dependency must share a compiled unit.
    """

    src: SymbolId
    dst: SymbolId | str
    kind: DependencyKind
    confidence: Confidence
    lineno: int | None = None
    invocation_mode: InvocationMode | None = None
    requires_same_unit: bool = False


@dataclass(frozen=True, slots=True)
class IslandCandidate:
    """Connected function cluster that appears useful to compile together.

    Candidates are heuristic recommendations, not proof of native compatibility.
    Build, verification, and target tests are still required before treating a
    candidate as deployable.

    Attributes:
        source_module: Importable source module name.
        symbols: Stable IDs of symbols recommended as one compilation candidate.
        required_imports: Imports required by the candidate.
        required_constants: Literal constants required by the candidate.
        required_local_symbols: Local symbols required by dependency closure.
        rejected_symbols: Symbols excluded from the candidate dependency closure.
        score: Scan-only extraction-safety or native-readiness score.
        risk: Conservative extraction risk classification.
        reasons: Deterministically ordered evidence supporting the decision.
    """

    source_module: ModuleId
    symbols: tuple[SymbolId, ...]
    required_imports: tuple[str, ...]
    required_constants: tuple[str, ...]
    required_local_symbols: tuple[SymbolId, ...]
    rejected_symbols: tuple[SymbolId, ...]
    score: int
    risk: IslandRisk
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PoisonRadius:
    """Explain how a rejected symbol contaminates otherwise clean candidates.

    The radius is report-only evidence: it helps users understand why a nearby
    candidate is risky, but it does not mutate the original blockers or scores.

    Attributes:
        poison: Rejected symbol whose dependencies affect nearby candidates.
        impacted: Otherwise viable symbols affected by the rejected symbol.
        reason: Concrete blocker or dependency evidence causing the impact.
    """

    poison: SymbolId
    impacted: tuple[SymbolId, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class SidecarGeneration:
    """Generated sidecar source and metadata for one enabled island.

    The source text is deterministic for the current generator version and input
    scan. `source_hash` is embedded into the generated file so stale sidecars can
    be detected without importing project code.

    Attributes:
        config: Enabled island that owns the generated sidecar.
        included_symbols: Public symbols copied into generated source.
        source_hash: Deterministic digest of generated or retained source.
        source_text: Exact source text retained for analysis or generation.
    """

    config: EnabledIslandConfig
    included_symbols: tuple[str, ...]
    source_hash: str
    source_text: str


@dataclass(frozen=True, slots=True)
class CompilePhaseTiming:
    """One measured subphase from a native compilation attempt.

    Attributes:
        name: Stable native compiler subphase name.
        duration_seconds: Elapsed wall-clock duration in seconds.
        detail: Optional human-readable timing context.
    """

    name: str
    duration_seconds: float
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class CompileAttempt:
    """Compatibility evidence from one native backend build attempt.

    Commands capture the invoked build shape, stdout/stderr preserve diagnostics,
    and `artifact_paths` lists extension outputs that were newly produced or
    modified. A failed attempt keeps enough detail for report rendering and CLI
    error messages without raising through command handlers.

    Attributes:
        success: Whether the represented operation completed successfully.
        command: Normalized command argument vector.
        stdout: Captured child process standard output.
        stderr: Captured child process standard error.
        artifact_paths: Native artifacts produced or changed by the build.
        duration_seconds: Elapsed wall-clock duration in seconds.
        phase_timings: Measured native compiler subphases.
        cache_status: Whether compilation used, missed, or partially restored cache state.
    """

    success: bool
    command: tuple[str, ...]
    stdout: str
    stderr: str
    artifact_paths: tuple[Path, ...]
    duration_seconds: float
    phase_timings: tuple[CompilePhaseTiming, ...] = ()
    cache_status: CompileCacheStatus = "disabled"


@dataclass(frozen=True, slots=True)
class BackendCompileResult:
    """Structured backend output plus the legacy attempt compatibility view.

    Attributes:
        attempt: Compatibility build evidence for the backend invocation.
        artifacts: Validated install-facing records for produced native files.
    """

    attempt: CompileAttempt
    artifacts: tuple[ArtifactRecord, ...]


@dataclass(frozen=True, slots=True)
class VerifyResult:
    """Runtime routing verification for an enabled Atoll island.

    Verification imports the source module, inspects the managed shim status, and
    records whether exported symbols were rebound to the sidecar. `error` is
    populated when the island is inactive, not compiled when required, or
    otherwise fails the routing contract.

    Attributes:
        source_module: Importable source module name.
        sidecar_module: Importable generated sidecar module name.
        active: Whether the managed runtime shim is active.
        compiled: Whether routing resolved to a native extension.
        origin: Resolved module origin or specialization evidence source.
        symbols: Pairs of exported symbol name and successful-rebinding status.
        error: User-facing failure text, or `None` on success.
    """

    source_module: str
    sidecar_module: str
    active: bool
    compiled: bool
    origin: str | None
    symbols: tuple[tuple[str, bool], ...]
    error: str | None = None


@dataclass(frozen=True, slots=True)
class PytestRunResult:
    """Result from a target-project pytest gate.

    The command is normalized before execution and the success flag is derived
    only from the child process exit code. Captured test output remains owned by
    pytest and is not stored in this lightweight result.

    Attributes:
        command: Normalized command argument vector.
        exit_code: Child process exit code.
        success: Whether the represented operation completed successfully.
    """

    command: tuple[str, ...]
    exit_code: int
    success: bool


@dataclass(frozen=True, slots=True)
class ModuleScan:
    """Complete analysis state for one Python module.

    Initial scans contain AST facts and module blockers. Later enrichment adds
    type diagnostics, dependency edges, candidates, and poison-radius records
    while preserving the same immutable shape for reports and commands.

    Attributes:
        module: Stable import name and source path for the scanned module.
        imports: Top-level imports retained for analysis or generation.
        constants: Top-level constants retained for analysis or sidecar generation.
        symbols: AST-derived declaration facts in source order.
        blockers: Conservative blockers attached to this module or symbol.
        top_level_statement_lines: Executable module-level statements that may affect extraction.
        mypy_diagnostics: Mypy diagnostics mapped to this module or symbol.
        dependency_edges: Conservative dependency edges derived from syntax.
        island_candidates: Conservative compilation candidates for the module.
        poison_radii: Report-only impact records for rejected symbols.
        typed_regions: Backend-neutral typed regions discovered or reported.
    """

    module: ModuleId
    imports: tuple[ImportRecord, ...]
    constants: tuple[ConstantRecord, ...]
    symbols: tuple[SymbolRecord, ...]
    blockers: tuple[Blocker, ...]
    top_level_statement_lines: tuple[int, ...]
    mypy_diagnostics: tuple[MypyDiagnostic, ...] = field(default_factory=tuple)
    dependency_edges: tuple[DependencyEdge, ...] = field(default_factory=tuple)
    island_candidates: tuple[IslandCandidate, ...] = field(default_factory=tuple)
    poison_radii: tuple[PoisonRadius, ...] = field(default_factory=tuple)
    typed_regions: tuple[TypedRegion, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class ScanResult:
    """Project-wide result produced by `atoll scan`.

    The result binds the resolved configuration to the enriched module scans used
    for JSON and Markdown reports. It intentionally excludes generated sidecar or
    build artifacts, which are owned by separate command results.

    Attributes:
        config: Resolved absolute project paths and persisted Atoll policy.
        modules: Discovered or reported modules in deterministic order.
    """

    config: ProjectConfig
    modules: tuple[ModuleScan, ...]
