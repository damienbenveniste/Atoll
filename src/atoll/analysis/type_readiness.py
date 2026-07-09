"""Attach type-checker diagnostics to scanned modules and symbols.

Atoll treats mypy errors inside a symbol as extraction blockers because mypyc
depends on the same type information. Diagnostics outside symbol line ranges
stay on the module so reports can surface them without falsely blaming a
candidate.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from atoll.models import Blocker, ModuleScan, MypyDiagnostic, SymbolRecord


def attach_mypy_diagnostics(
    modules: tuple[ModuleScan, ...],
    diagnostics: tuple[MypyDiagnostic, ...],
) -> tuple[ModuleScan, ...]:
    """Map mypy diagnostics to symbols by source line range.

    The mapping is intentionally line-based and conservative. It does not try to
    interpret mypy context notes beyond the scanner's symbol spans, so unmapped
    diagnostics remain module-level evidence.
    """
    diagnostics_by_path = _group_diagnostics_by_path(diagnostics)
    return tuple(
        _attach_module_diagnostics(module, diagnostics_by_path.get(module.module.path, ()))
        for module in modules
    )


def _attach_module_diagnostics(
    module: ModuleScan,
    diagnostics: tuple[MypyDiagnostic, ...],
) -> ModuleScan:
    mapped_symbols = tuple(
        _attach_symbol_diagnostics(symbol, diagnostics) for symbol in module.symbols
    )
    mapped_symbol_ids = {
        diagnostic.symbol
        for symbol in mapped_symbols
        for diagnostic in symbol.mypy_diagnostics
        if diagnostic.symbol is not None
    }
    module_diagnostics = tuple(
        diagnostic for diagnostic in diagnostics if diagnostic.symbol not in mapped_symbol_ids
    )
    return replace(
        module,
        symbols=mapped_symbols,
        mypy_diagnostics=module_diagnostics,
    )


def _attach_symbol_diagnostics(
    symbol: SymbolRecord,
    diagnostics: tuple[MypyDiagnostic, ...],
) -> SymbolRecord:
    matched = tuple(
        replace(diagnostic, symbol=symbol.id)
        for diagnostic in diagnostics
        if symbol.lineno <= diagnostic.line <= symbol.end_lineno
    )
    if not matched:
        return symbol
    mypy_blockers = tuple(
        Blocker(
            severity="hard",
            code="MYPY_ERROR",
            message=diagnostic.message,
            lineno=diagnostic.line,
            symbol=symbol.id,
        )
        for diagnostic in matched
        if diagnostic.severity == "error"
    )
    return replace(
        symbol,
        blockers=(*symbol.blockers, *mypy_blockers),
        mypy_diagnostics=matched,
    )


def _group_diagnostics_by_path(
    diagnostics: tuple[MypyDiagnostic, ...],
) -> dict[Path, tuple[MypyDiagnostic, ...]]:
    grouped: dict[Path, list[MypyDiagnostic]] = {}
    for diagnostic in diagnostics:
        grouped.setdefault(diagnostic.path, []).append(diagnostic)
    return {path: tuple(path_diagnostics) for path, path_diagnostics in grouped.items()}
