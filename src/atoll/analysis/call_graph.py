"""Conservative same-module dependency graph construction.

Edges model what copied sidecar code may need at runtime or in preserved type
surface. The graph links local symbols with high confidence, imported
boundaries with medium confidence, and unresolved globals with low confidence
so candidate scoring can prefer explainable, local clusters while richer report
consumers can see class, method, decorator, and annotation dependencies.
"""

from __future__ import annotations

import builtins
from collections.abc import Iterable
from dataclasses import dataclass

from atoll.models import (
    ConstantRecord,
    DependencyEdge,
    DependencyKind,
    ImportRecord,
    ModuleScan,
    SymbolId,
    SymbolRecord,
)

_BUILTIN_NAMES = frozenset(dir(builtins))


@dataclass(frozen=True, slots=True)
class _BoundaryIndex:
    symbols: dict[str, SymbolId]
    imports: dict[str, str]
    constants: dict[str, ConstantRecord]


def build_dependency_edges(module: ModuleScan) -> tuple[DependencyEdge, ...]:
    """Build dependency edges for functions, methods, and classes in a module.

    The graph uses scanner-provided name facts rather than AST re-traversal so
    dependency analysis stays aligned with cached scan output. Scanner-version
    invalidation ensures cached records always carry the typed-region fields.

    Args:
        module: Module scan or module identity being analyzed.

    Returns:
        tuple[DependencyEdge, ...]: Deterministically ordered dependency edges for the module.
    """
    symbol_by_name = _symbol_name_map(module.symbols)
    imported_names = _imported_name_map(module.imports)
    constants = {constant.name: constant for constant in module.constants}
    boundary_index = _BoundaryIndex(
        symbols=symbol_by_name,
        imports=imported_names,
        constants=constants,
    )
    edges: list[DependencyEdge] = []
    for symbol in module.symbols:
        edges.extend(
            _call_edges(
                symbol=symbol,
                symbol_by_name=symbol_by_name,
                imported_names=imported_names,
            )
        )
        edges.extend(
            _global_edges(symbol.id, symbol.uses_globals, constants, imported_names, symbol_by_name)
        )
        edges.extend(
            _named_boundary_edges(
                source=symbol.id,
                names=symbol.base_names,
                kind="inherits",
                index=boundary_index,
            )
        )
        edges.extend(
            _named_boundary_edges(
                source=symbol.id,
                names=_decorator_names(symbol),
                kind="decorated_by",
                index=boundary_index,
            )
        )
        edges.extend(
            _named_boundary_edges(
                source=symbol.id,
                names=symbol.annotation_names,
                kind="annotation",
                index=boundary_index,
            )
        )
    return _dedupe_edges(edges)


def _symbol_name_map(symbols: tuple[SymbolRecord, ...]) -> dict[str, SymbolId]:
    symbol_by_name = {symbol.id.qualname: symbol.id for symbol in symbols}
    short_names: dict[str, SymbolId | None] = {}
    for symbol in symbols:
        if "." in symbol.id.qualname:
            continue
        name = symbol.id.qualname.rsplit(".", maxsplit=1)[-1]
        short_names[name] = symbol.id if name not in short_names else None
    for name, symbol_id in short_names.items():
        if symbol_id is not None:
            symbol_by_name.setdefault(name, symbol_id)
    return symbol_by_name


def _dedupe_edges(edges: Iterable[DependencyEdge]) -> tuple[DependencyEdge, ...]:
    deduped: list[DependencyEdge] = []
    seen: set[tuple[SymbolId, SymbolId | str, DependencyKind, str, int | None]] = set()
    for edge in edges:
        key = (edge.src, edge.dst, edge.kind, edge.confidence, edge.lineno)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(edge)
    return tuple(deduped)


def _call_edges(
    symbol: SymbolRecord,
    symbol_by_name: dict[str, SymbolId],
    imported_names: dict[str, str],
) -> tuple[DependencyEdge, ...]:
    edges: list[DependencyEdge] = []
    for called_name in symbol.called_names:
        if called_name in symbol_by_name:
            edges.append(
                DependencyEdge(
                    src=symbol.id,
                    dst=symbol_by_name[called_name],
                    kind="calls",
                    confidence="high",
                    lineno=symbol.lineno,
                )
            )
        elif called_name in imported_names:
            edges.append(
                DependencyEdge(
                    src=symbol.id,
                    dst=imported_names[called_name],
                    kind="imports",
                    confidence="medium",
                    lineno=symbol.lineno,
                )
            )
    owner_class = _owner_class(symbol)
    for called_path in symbol.called_paths:
        target = _method_path_target(called_path, owner_class)
        if target in symbol_by_name:
            edges.append(
                DependencyEdge(
                    src=symbol.id,
                    dst=symbol_by_name[target],
                    kind="calls_method",
                    confidence="high",
                    lineno=symbol.lineno,
                )
            )
        else:
            root_name = target.split(".", maxsplit=1)[0]
            if root_name in imported_names:
                edges.append(
                    DependencyEdge(
                        src=symbol.id,
                        dst=imported_names[root_name],
                        kind="imports",
                        confidence="medium",
                        lineno=symbol.lineno,
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


def _named_boundary_edges(
    source: SymbolId,
    names: tuple[str, ...],
    kind: DependencyKind,
    index: _BoundaryIndex,
) -> tuple[DependencyEdge, ...]:
    edges: list[DependencyEdge] = []
    for name in names:
        lookup_name = _boundary_lookup_name(name)
        if kind == "annotation" and lookup_name in _BUILTIN_NAMES:
            continue
        if lookup_name in index.symbols:
            edges.append(
                DependencyEdge(
                    src=source,
                    dst=index.symbols[lookup_name],
                    kind=kind,
                    confidence="high",
                )
            )
        elif lookup_name in index.constants:
            edges.append(
                DependencyEdge(
                    src=source,
                    dst=lookup_name,
                    kind=kind,
                    confidence="high",
                )
            )
        elif lookup_name in index.imports:
            edges.append(
                DependencyEdge(
                    src=source,
                    dst=index.imports[lookup_name],
                    kind=kind,
                    confidence="medium",
                )
            )
        else:
            edges.append(
                DependencyEdge(
                    src=source,
                    dst=name,
                    kind=kind,
                    confidence="low",
                )
            )
    return tuple(edges)


def _boundary_lookup_name(expression: str) -> str:
    unsubscripted = expression.split("[", maxsplit=1)[0]
    return unsubscripted.split(".", maxsplit=1)[0]


def _owner_class(symbol: SymbolRecord) -> str | None:
    owner_class = symbol.owner_class
    if isinstance(owner_class, str) and owner_class:
        return owner_class
    if symbol.kind == "method" and "." in symbol.id.qualname:
        return symbol.id.qualname.rsplit(".", maxsplit=1)[0]
    return None


def _method_path_target(called_path: str, owner_class: str | None) -> str:
    if owner_class is not None and "." in called_path:
        root, member = called_path.split(".", maxsplit=1)
        if root in {"self", "cls"}:
            return f"{owner_class}.{member}"
    return called_path


def _decorator_names(symbol: SymbolRecord) -> tuple[str, ...]:
    return tuple(
        name for decorator in symbol.decorators if (name := _dependency_name(decorator)) is not None
    )


def _dependency_name(expression: str) -> str | None:
    name = expression.split("(", maxsplit=1)[0].strip()
    if not name:
        return None
    if "." in name:
        return name.split(".", maxsplit=1)[0]
    return name


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
