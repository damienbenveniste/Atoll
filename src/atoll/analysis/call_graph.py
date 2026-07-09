"""Conservative same-module dependency graph construction.

Edges model what copied sidecar code may need at runtime. The graph links direct
same-module calls with high confidence, imported boundaries with medium
confidence, and unresolved globals with low confidence so candidate scoring can
prefer explainable, local clusters.
"""

from __future__ import annotations

from atoll.models import ConstantRecord, DependencyEdge, ImportRecord, ModuleScan, SymbolId


def build_dependency_edges(module: ModuleScan) -> tuple[DependencyEdge, ...]:
    """Build dependency edges for top-level function symbols in a module.

    The function intentionally ignores classes and methods as edge sources for
    V1 candidate clustering. It uses scanner-provided name facts rather than AST
    re-traversal so dependency analysis stays aligned with cached scan output.
    """
    symbol_by_name = {symbol.id.qualname: symbol.id for symbol in module.symbols}
    imported_names = _imported_name_map(module.imports)
    constants = {constant.name: constant for constant in module.constants}
    edges: list[DependencyEdge] = []
    for symbol in module.symbols:
        if symbol.kind != "function":
            continue
        edges.extend(_call_edges(symbol.id, symbol.called_names, symbol_by_name, imported_names))
        edges.extend(
            _global_edges(symbol.id, symbol.uses_globals, constants, imported_names, symbol_by_name)
        )
    return tuple(edges)


def _call_edges(
    source: SymbolId,
    called_names: tuple[str, ...],
    symbol_by_name: dict[str, SymbolId],
    imported_names: dict[str, str],
) -> tuple[DependencyEdge, ...]:
    edges: list[DependencyEdge] = []
    for called_name in called_names:
        if called_name in symbol_by_name:
            edges.append(
                DependencyEdge(
                    src=source,
                    dst=symbol_by_name[called_name],
                    kind="calls",
                    confidence="high",
                )
            )
        elif called_name in imported_names:
            edges.append(
                DependencyEdge(
                    src=source,
                    dst=imported_names[called_name],
                    kind="imports",
                    confidence="medium",
                )
            )
    return tuple(edges)


def _global_edges(
    source: SymbolId,
    global_names: tuple[str, ...],
    constants: dict[str, ConstantRecord],
    imported_names: dict[str, str],
    symbol_by_name: dict[str, SymbolId],
) -> tuple[DependencyEdge, ...]:
    edges: list[DependencyEdge] = []
    for global_name in global_names:
        if global_name in symbol_by_name:
            edges.append(
                DependencyEdge(
                    src=source,
                    dst=symbol_by_name[global_name],
                    kind="uses_global",
                    confidence="high",
                )
            )
        elif global_name in constants:
            edges.append(
                DependencyEdge(
                    src=source,
                    dst=global_name,
                    kind="uses_global",
                    confidence="high",
                )
            )
        elif global_name in imported_names:
            edges.append(
                DependencyEdge(
                    src=source,
                    dst=imported_names[global_name],
                    kind="imports",
                    confidence="medium",
                )
            )
        else:
            edges.append(
                DependencyEdge(
                    src=source,
                    dst=global_name,
                    kind="unknown",
                    confidence="low",
                )
            )
    return tuple(edges)


def _imported_name_map(imports: tuple[ImportRecord, ...]) -> dict[str, str]:
    imported: dict[str, str] = {}
    for record in imports:
        for name in record.imported_names:
            imported[name] = _import_target(record, name)
    return imported


def _import_target(record: ImportRecord, name: str) -> str:
    if record.module is None:
        return name
    prefix = "." * record.level
    return f"{prefix}{record.module}.{name}" if record.module else f"{prefix}{name}"
