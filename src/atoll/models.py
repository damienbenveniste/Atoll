"""Typed contracts shared by Atoll analysis, generation, build, and reporting.

The dataclasses in this module are immutable handoff objects. They keep raw
source facts, conservative analysis decisions, and runtime/build evidence
separate so command handlers can enrich data without mutating earlier phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Backend = Literal["mypyc"]
BlockerSeverity = Literal["hard", "soft", "info"]
Confidence = Literal["high", "medium", "low"]
ConstantKind = Literal["literal_constant", "runtime_dynamic", "unknown"]
DependencyKind = Literal["calls", "uses_global", "inherits", "decorated_by", "imports", "unknown"]
DiagnosticSeverity = Literal["error", "note"]
IslandRisk = Literal["low", "medium", "high"]
SymbolKind = Literal["function", "class", "method"]
Visibility = Literal["public", "private"]


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
class CompileAttempt:
    """Evidence from one mypyc build attempt.

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


@dataclass(frozen=True, slots=True)
class ScanResult:
    """Project-wide result produced by `atoll scan`.

    The result binds the resolved configuration to the enriched module scans used
    for JSON and Markdown reports. It intentionally excludes generated sidecar or
    build artifacts, which are owned by separate command results.
    """

    config: ProjectConfig
    modules: tuple[ModuleScan, ...]
