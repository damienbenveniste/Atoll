"""Typed data structures for Atoll's source analysis pipeline."""

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
    """Resolved project settings used by scan commands."""

    root: Path
    source_roots: tuple[Path, ...]
    backend: Backend
    cache_dir: Path
    report_dir: Path
    islands: tuple[EnabledIslandConfig, ...] = ()


@dataclass(frozen=True, slots=True)
class EnabledIslandConfig:
    """A configured Atoll island that can be generated, built, and verified."""

    source_module: str
    source_path: Path
    sidecar_module: str
    sidecar_path: Path
    symbols: tuple[str, ...]
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class ModuleId:
    """Stable identifier for a Python module in a source root."""

    name: str
    path: Path


@dataclass(frozen=True, slots=True)
class SymbolId:
    """Stable identifier for a top-level symbol or simple method."""

    module: str
    qualname: str

    @property
    def stable_id(self) -> str:
        """Return the report-facing `module::qualname` identifier."""
        return f"{self.module}::{self.qualname}"


@dataclass(frozen=True, slots=True)
class Blocker:
    """A reason a symbol or module is risky for native sidecar extraction."""

    severity: BlockerSeverity
    code: str
    message: str
    lineno: int | None = None
    symbol: SymbolId | None = None


@dataclass(frozen=True, slots=True)
class MypyDiagnostic:
    """One parsed mypy diagnostic, optionally mapped back to a symbol."""

    path: Path
    line: int
    column: int | None
    severity: DiagnosticSeverity
    code: str | None
    message: str
    symbol: SymbolId | None = None


@dataclass(frozen=True, slots=True)
class ImportRecord:
    """Top-level import statement and the names it binds in the module."""

    source_text: str
    imported_names: tuple[str, ...]
    module: str | None
    level: int
    lineno: int
    end_lineno: int


@dataclass(frozen=True, slots=True)
class ConstantRecord:
    """Top-level assignment classification used by later island extraction."""

    name: str
    kind: ConstantKind
    source_text: str
    lineno: int
    end_lineno: int


@dataclass(frozen=True, slots=True)
class SymbolRecord:
    """AST-derived facts for one function, class, or simple method."""

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
    """A conservative dependency edge from one symbol to another boundary."""

    src: SymbolId
    dst: SymbolId | str
    kind: DependencyKind
    confidence: Confidence
    lineno: int | None = None


@dataclass(frozen=True, slots=True)
class IslandCandidate:
    """A connected cluster that appears safe and useful to compile together."""

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
    """Report which clean candidates are affected by a rejected symbol."""

    poison: SymbolId
    impacted: tuple[SymbolId, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class SidecarGeneration:
    """Generated sidecar source and metadata for one enabled island."""

    config: EnabledIslandConfig
    included_symbols: tuple[str, ...]
    source_hash: str
    source_text: str


@dataclass(frozen=True, slots=True)
class CompileAttempt:
    """Result from compiling one or more generated sidecar modules."""

    success: bool
    command: tuple[str, ...]
    stdout: str
    stderr: str
    artifact_paths: tuple[Path, ...]
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class VerifyResult:
    """Runtime routing verification for an enabled Atoll island."""

    source_module: str
    sidecar_module: str
    active: bool
    compiled: bool
    origin: str | None
    symbols: tuple[tuple[str, bool], ...]
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ModuleScan:
    """Complete first-pass scan result for one Python module."""

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
    """Project-wide result produced by `atoll scan`."""

    config: ProjectConfig
    modules: tuple[ModuleScan, ...]
