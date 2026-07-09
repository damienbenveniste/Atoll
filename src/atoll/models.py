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
IslandRisk = Literal["low", "medium", "high"]
LossAction = Literal["preserve", "specialize", "box", "fallback", "reject"]
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
Visibility = Literal["public", "private"]

_SHA256_HEX_LENGTH = 64
_LOWERCASE_HEX_DIGITS = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Resolved project settings for one Atoll invocation.

    Paths are absolute after discovery. `source_roots` controls importable module
    discovery, while `cache_dir` and `report_dir` are Atoll-owned output
    locations under the project root unless configuration explicitly overrides
    them.
    """

    root: Path
    source_roots: tuple[Path, ...]
    backend: Backend
    cache_dir: Path
    report_dir: Path
    islands: tuple[EnabledIslandConfig, ...] = ()


@dataclass(frozen=True, slots=True)
class EnabledIslandConfig:
    """Persistent configuration for one Atoll-managed source module.

    The source module keeps the managed shim, the sidecar module contains copied
    symbols, and `symbols` names the exported top-level functions that should be
    rebound at runtime. Disabled islands remain in configuration for auditability
    but are skipped by generation, build, and verification commands.
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
    """

    module: str
    qualname: str

    @property
    def stable_id(self) -> str:
        """Return the stable report-facing `module::qualname` identifier text used in reports."""
        return f"{self.module}::{self.qualname}"


@dataclass(frozen=True, slots=True)
class Blocker:
    """Conservative reason a module or symbol should not be compiled blindly.

    Hard blockers prevent candidate extraction, while soft and info blockers
    explain risk that may still be useful in reports. `symbol` is absent only for
    module-level conditions that cannot be tied to a specific symbol.
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
    """

    name: str
    kind: ParameterKind
    annotation: str | None
    default_source: str | None


@dataclass(frozen=True, slots=True)
class FieldRecord:
    """Typed class field declaration retained for class-region planning."""

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
    """

    name: str
    kind: TypeParameterKind
    declaration: str


@dataclass(frozen=True, slots=True)
class SymbolRecord:
    """AST-derived facts for one function, class, or simple method.

    The scanner records source locations, type-readiness signals, referenced
    names, and local blockers without executing project code. Later phases attach
    mypy diagnostics, dependency edges, and candidate decisions by replacing this
    immutable record.
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
    base_names: tuple[str, ...] = ()
    fields: tuple[FieldRecord, ...] = ()
    declaration_start_lineno: int | None = None
    scope_type_parameters: tuple[str, ...] = ()
    type_parameter_records: tuple[TypeParameterRecord, ...] = ()
    scope_type_parameter_records: tuple[TypeParameterRecord, ...] = ()
    any_annotation_sources: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BindingTarget:
    """One source binding that a compiled region promises to replace.

    `source` is always the public source identity. `compiled_name` is private
    backend output and may differ for lowered methods or specializations.
    """

    source: SymbolId
    compiled_name: str
    kind: BindingKind
    owner_class: str | None
    execution_kind: ExecutionKind
    required: bool = True


@dataclass(frozen=True, slots=True)
class TypeBinding:
    """Preserved or concretized type evidence used by a typed region."""

    name: str
    annotation: str
    source: TypeBindingSource
    concrete: bool
    substitutions: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class LoweringDecision:
    """Auditable decision about how one typed-region fact will be lowered."""

    target: str
    action: LossAction
    reason: str


@dataclass(frozen=True, slots=True)
class RegionMember:
    """Unlowered source declaration owned by a backend-neutral typed region."""

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
    fields: tuple[FieldRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class RegionDependency:
    """Typed-region dependency retaining runtime versus annotation intent."""

    src: SymbolId
    dst: SymbolId | str
    kind: DependencyKind
    confidence: Confidence
    role: DependencyRole
    type_only: bool


@dataclass(frozen=True, slots=True)
class TypedRegion:
    """Backend-neutral connected source region before any type-losing rewrite.

    Region IDs and hashes are deterministic for the retained source and
    dependency evidence. Instances are safe to cache, compare, and pass to
    backend assessment without reading or importing target-project code.
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


@dataclass(frozen=True, slots=True)
class BackendAssessment:
    """Deterministic capability decision for one backend and typed region.

    Partial assessments keep supported and unsupported members separate so one
    unsupported execution shape cannot poison independent members. Support
    means the backend can compile a correctly prepared member unit; it does not
    promise that runtime binding for that member has been implemented. Reasons
    are stable diagnostic text suitable for reports and backend selection.
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
    """

    region: TypedRegion
    source_path: Path
    logical_module: str
    install_relative_dir: str = ""
    members: tuple[SymbolId, ...] = ()


@dataclass(frozen=True, slots=True)
class CompilationUnit:
    """Backend-specific, content-addressable source unit ready to compile."""

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
        """Reject absolute, parent-traversing, or non-POSIX install paths."""
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
class BackendDiagnostic:
    """Normalized compiler failure independent of backend exception types."""

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
    """

    src: SymbolId
    dst: SymbolId | str
    kind: DependencyKind
    confidence: Confidence
    lineno: int | None = None


@dataclass(frozen=True, slots=True)
class IslandCandidate:
    """Connected function cluster that appears useful to compile together.

    Candidates are heuristic recommendations, not proof of native compatibility.
    Build, verification, and target tests are still required before treating a
    candidate as deployable.
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
    """

    config: EnabledIslandConfig
    included_symbols: tuple[str, ...]
    source_hash: str
    source_text: str


@dataclass(frozen=True, slots=True)
class CompilePhaseTiming:
    """One measured subphase from a native compilation attempt."""

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
    """Structured backend output plus the legacy attempt compatibility view."""

    attempt: CompileAttempt
    artifacts: tuple[ArtifactRecord, ...]


@dataclass(frozen=True, slots=True)
class VerifyResult:
    """Runtime routing verification for an enabled Atoll island.

    Verification imports the source module, inspects the managed shim status, and
    records whether exported symbols were rebound to the sidecar. `error` is
    populated when the island is inactive, not compiled when required, or
    otherwise fails the routing contract.
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
    """

    config: ProjectConfig
    modules: tuple[ModuleScan, ...]
