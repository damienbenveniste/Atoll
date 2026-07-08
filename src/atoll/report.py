"""Report rendering for Atoll scan results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

from atoll.models import (
    Blocker,
    BlockerSeverity,
    Confidence,
    ConstantKind,
    DependencyKind,
    DiagnosticSeverity,
    IslandRisk,
    ModuleScan,
    MypyDiagnostic,
    ScanResult,
    SymbolId,
    SymbolKind,
    Visibility,
)


class BlockerReport(TypedDict):
    severity: BlockerSeverity
    code: str
    message: str
    lineno: int | None
    symbol: str | None


class ImportReport(TypedDict):
    source_text: str
    imported_names: list[str]
    module: str | None
    level: int
    lineno: int
    end_lineno: int


class ConstantReport(TypedDict):
    name: str
    kind: ConstantKind
    source_text: str
    lineno: int
    end_lineno: int


class SymbolReport(TypedDict):
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
    blockers: list[BlockerReport]
    mypy_diagnostics: list[MypyDiagnosticReport]


class MypyDiagnosticReport(TypedDict):
    path: str
    line: int
    column: int | None
    severity: DiagnosticSeverity
    code: str | None
    message: str
    symbol: str | None


class DependencyEdgeReport(TypedDict):
    src: str
    dst: str
    kind: DependencyKind
    confidence: Confidence
    lineno: int | None


class IslandCandidateReport(TypedDict):
    symbols: list[str]
    required_imports: list[str]
    required_constants: list[str]
    required_local_symbols: list[str]
    rejected_symbols: list[str]
    score: int
    risk: IslandRisk
    reasons: list[str]


class PoisonRadiusReport(TypedDict):
    poison: str
    impacted: list[str]
    reason: str


class ModuleReport(TypedDict):
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


class SummaryReport(TypedDict):
    modules_scanned: int
    symbols_scanned: int
    island_candidates: int
    hard_blockers: int
    soft_blockers: int


class ScanReport(TypedDict):
    version: int
    tool: str
    project_root: str
    source_roots: list[str]
    summary: SummaryReport
    modules: list[ModuleReport]


def build_scan_report(result: ScanResult) -> ScanReport:
    """Convert scan dataclasses into a stable JSON report shape."""
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
        "version": 1,
        "tool": "atoll",
        "project_root": str(result.config.root),
        "source_roots": [str(path) for path in result.config.source_roots],
        "summary": {
            "modules_scanned": len(result.modules),
            "symbols_scanned": sum(len(module.symbols) for module in result.modules),
            "island_candidates": sum(len(module.island_candidates) for module in result.modules),
            "hard_blockers": hard_count,
            "soft_blockers": soft_count,
        },
        "modules": module_reports,
    }


def write_json_report(path: Path, report: ScanReport) -> None:
    """Write `report` as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8")


def write_markdown_report(path: Path, report: ScanReport) -> None:
    """Write a human-readable scan report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown_report(report), encoding="utf-8")


def render_markdown_report(report: ScanReport) -> str:
    """Render a concise Markdown scan report."""
    lines = [
        "# Atoll Scan Report",
        "",
        "## Summary",
        "",
        f"- Modules scanned: {report['summary']['modules_scanned']}",
        f"- Symbols scanned: {report['summary']['symbols_scanned']}",
        f"- Island candidates: {report['summary']['island_candidates']}",
        f"- Hard blockers: {report['summary']['hard_blockers']}",
        f"- Soft blockers: {report['summary']['soft_blockers']}",
        "",
    ]
    for module in report["modules"]:
        lines.extend(_markdown_module(module))
    return "\n".join(lines).rstrip() + "\n"


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
                "risk": candidate.risk,
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
            lines.append(f"- score {candidate['score']} ({candidate['risk']}): {symbols}")
    if module["poison_radii"]:
        lines.extend(["", "Poison residue:"])
        for radius in module["poison_radii"]:
            impacted = ", ".join(radius["impacted"]) or "none"
            lines.append(f"- `{radius['poison']}` impacts: {impacted}")
    lines.append("")
    return lines
