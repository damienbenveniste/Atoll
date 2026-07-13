"""Implementation of the `atoll explain` command.

Explain runs the same analysis stack as scan for one module, then renders a
targeted Markdown explanation for either the module or a single symbol. It is a
read-only command and does not write reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.clustering import enrich_island_analysis
from atoll.analysis.type_readiness import attach_mypy_diagnostics
from atoll.backends.mypy import run_mypy
from atoll.models import ModuleScan, SymbolId, SymbolRecord
from atoll.native_optimization.buffer_analysis import analyze_buffer_scan
from atoll.native_optimization.call_chains import analyze_call_chain_scan
from atoll.native_optimization.scalar_analysis import analyze_scalar_scan
from atoll.optimization_policy import DEFAULT_MINIMUM_MARGINAL_SPEEDUP
from atoll.project import discover_project
from atoll.report import risk_summary, score_summary


@dataclass(frozen=True, slots=True)
class ExplainOptions:
    """User-facing options for module or symbol explanation output.

    Attributes:
        root: Root directory of the target Python project.
        target: Lowering target or package verification target.
        mypy_enabled: Whether the command runs mypy diagnostic mapping.
    """

    root: Path
    target: str
    mypy_enabled: bool = True


def execute_explain(options: ExplainOptions) -> str:
    """Explain a module or symbol using current Atoll analysis.

    The target is `module` or `module::symbol`. When mypy is enabled, diagnostics
    are attached before scoring so the explanation matches scan behavior.

    Args:
        options: Validated command options supplied by the CLI layer.

    Returns:
        str: Human-readable explanation of the selected module or symbol.
    """
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
    lines.extend(_native_module_explanation(module))
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
    lines.extend(_native_symbol_explanation(module, symbol.id))
    return "\n".join(lines).rstrip() + "\n"


def _native_module_explanation(module: ModuleScan) -> list[str]:
    scalar = analyze_scalar_scan(module)
    chains = analyze_call_chain_scan(module)
    buffers = analyze_buffer_scan(module)
    return [
        "",
        "Native optimization:",
        (f"- Fixed-width scalar: {len(scalar.plans)} proven, {len(scalar.rejections)} fallback"),
        (f"- Direct call chains: {len(chains.plans)} proven, {len(chains.rejections)} fallback"),
        (f"- Zero-copy buffers: {len(buffers.plans)} proven, {len(buffers.rejections)} fallback"),
        (
            "- Specialized variants require a stable "
            f"{DEFAULT_MINIMUM_MARGINAL_SPEEDUP:.3f}x marginal benchmark when configured."
        ),
    ]


def _native_symbol_explanation(module: ModuleScan, symbol: SymbolId) -> list[str]:
    scalar = analyze_scalar_scan(module)
    chains = analyze_call_chain_scan(module)
    buffers = analyze_buffer_scan(module)
    lines = ["", "Native optimization:"]
    scalar_plans = tuple(item for item in scalar.plans if item.member == symbol)
    for scalar_plan in scalar_plans:
        for proof in scalar_plan.width_proofs:
            domains = ", ".join(
                f"{parameter.name}={parameter.interval.minimum}..{parameter.interval.maximum}"
                for parameter in proof.parameters
            )
            lines.append(
                f"- Fixed-width i{proof.native.width}: {domains}; "
                f"return={proof.return_interval.minimum}..{proof.return_interval.maximum}"
            )
    for chain_plan in chains.plans:
        if chain_plan.root == symbol:
            helpers = ", ".join(helper.qualname for helper in chain_plan.helpers) or "none"
            lines.append(f"- Direct native call chain: helpers {helpers}")
    for buffer_plan in buffers.plans:
        if buffer_plan.member == symbol:
            parameters = ", ".join(buffer.name for buffer in buffer_plan.buffers)
            lines.append(f"- Zero-copy {buffer_plan.reduction} reduction: buffers {parameters}")
    rejections = (
        *(
            f"scalar/{item.code}: {item.message}"
            for item in scalar.rejections
            if item.member == symbol
        ),
        *(
            f"call-chain/{item.code}: {item.message}"
            for item in chains.rejections
            if item.root == symbol
        ),
        *(
            f"buffer/{item.code}: {item.message}"
            for item in buffers.rejections
            if item.member == symbol
        ),
    )
    lines.extend(f"- Fallback: {reason}" for reason in rejections)
    if lines == ["", "Native optimization:"]:
        lines.append("- No native specialization candidate was formed.")
    return lines


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
