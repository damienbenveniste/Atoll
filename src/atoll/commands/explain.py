"""Implementation of the `atoll explain` command."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.clustering import enrich_island_analysis
from atoll.analysis.type_readiness import attach_mypy_diagnostics
from atoll.backends.mypy import run_mypy
from atoll.models import ModuleScan, SymbolId, SymbolRecord
from atoll.project import discover_project
from atoll.report import risk_summary, score_summary


@dataclass(frozen=True, slots=True)
class ExplainOptions:
    """User-facing options for explanation output."""

    root: Path
    target: str
    mypy_enabled: bool = True


def execute_explain(options: ExplainOptions) -> str:
    """Explain a module or symbol using current Atoll analysis."""
    module_name, symbol_name = _split_target(options.target)
    module = _analyze_module(options.root, module_name, mypy_enabled=options.mypy_enabled)
    if symbol_name is not None:
        return _symbol_explanation(module, symbol_name)
    return _module_explanation(module)


def _analyze_module(root: Path, module_name: str, *, mypy_enabled: bool) -> ModuleScan:
    project = discover_project(root)
    modules = tuple(scan_module(module) for module in project.modules if module.name == module_name)
    if not modules:
        raise ValueError(f"module not found under configured source roots: {module_name}")
    if mypy_enabled:
        mypy_run = run_mypy(project.config)
        modules = attach_mypy_diagnostics(modules, mypy_run.diagnostics)
    return enrich_island_analysis(modules[0])


def _module_explanation(module: ModuleScan) -> str:
    lines = [
        f"# {module.module.name}",
        "",
        f"Path: {module.module.path}",
        f"Symbols: {len(module.symbols)}",
        f"Candidate islands: {len(module.island_candidates)}",
        "",
    ]
    if module.island_candidates:
        lines.append("Candidates:")
        for candidate in module.island_candidates:
            symbols = ", ".join(symbol.qualname for symbol in candidate.symbols)
            lines.append(f"- {score_summary(candidate.score)}; {risk_summary(candidate.risk)}")
            lines.append(f"  symbols: {symbols}")
            lines.append(f"  reasons: {', '.join(candidate.reasons)}")
    if module.blockers:
        lines.extend(["", "Module blockers:"])
        for blocker in module.blockers:
            location = f"line {blocker.lineno}" if blocker.lineno is not None else "module"
            lines.append(f"- {blocker.code} ({blocker.severity}, {location}): {blocker.message}")
    if module.poison_radii:
        lines.extend(["", "Poison residue:"])
        for radius in module.poison_radii:
            impacted = ", ".join(symbol.qualname for symbol in radius.impacted) or "none"
            lines.append(f"- {radius.poison.qualname}: {radius.reason}; impacts {impacted}")
    return "\n".join(lines).rstrip() + "\n"


def _symbol_explanation(module: ModuleScan, symbol_name: str) -> str:
    symbol = _find_symbol(module.symbols, symbol_name)
    blockers = tuple(blocker.code for blocker in symbol.blockers)
    edges = tuple(edge for edge in module.dependency_edges if edge.src == symbol.id)
    lines = [
        f"# {symbol.id.stable_id}",
        "",
        f"Kind: {symbol.kind}",
        f"Lines: {symbol.lineno}-{symbol.end_lineno}",
        f"Typed args: {symbol.annotated_arg_count}/{symbol.arg_count}",
        f"Return annotation: {symbol.has_return_annotation}",
        f"Blockers: {', '.join(blockers) if blockers else 'none'}",
        "",
        "Dependencies:",
    ]
    if edges:
        for edge in edges:
            destination = edge.dst.stable_id if isinstance(edge.dst, SymbolId) else edge.dst
            lines.append(f"- {edge.kind}/{edge.confidence}: {destination}")
    else:
        lines.append("- none")
    diagnostics = tuple(diagnostic for diagnostic in symbol.mypy_diagnostics)
    if diagnostics:
        lines.extend(["", "Mypy diagnostics:"])
        for diagnostic in diagnostics:
            code = f" [{diagnostic.code}]" if diagnostic.code is not None else ""
            lines.append(f"- line {diagnostic.line}: {diagnostic.message}{code}")
    return "\n".join(lines).rstrip() + "\n"


def _find_symbol(symbols: tuple[SymbolRecord, ...], symbol_name: str) -> SymbolRecord:
    for symbol in symbols:
        if symbol.id.qualname == symbol_name:
            return symbol
    raise ValueError(f"symbol not found: {symbol_name}")


def _split_target(target: str) -> tuple[str, str | None]:
    module_name, separator, symbol_name = target.partition("::")
    if not module_name:
        raise ValueError("explain target must include a module name")
    return module_name, symbol_name if separator else None
