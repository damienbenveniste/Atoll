"""Focused tests for dependency-edge construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from atoll.analysis.call_graph import build_dependency_edges
from atoll.models import (
    ConstantRecord,
    DependencyEdge,
    ImportRecord,
    ModuleId,
    ModuleScan,
    SymbolId,
    SymbolRecord,
)


@dataclass(frozen=True, slots=True)
class FutureSymbolRecord:
    """Synthetic scanner record with upcoming dependency fields."""

    id: SymbolId
    kind: str
    called_names: tuple[str, ...] = ()
    uses_globals: tuple[str, ...] = ()
    decorators: tuple[str, ...] = ()
    lineno: int = 1
    owner_class: str | None = None
    called_paths: tuple[str, ...] = ()
    annotation_names: tuple[str, ...] = ()
    base_names: tuple[str, ...] = ()


def test_dependency_edges_resolve_methods_classes_and_typed_boundaries(tmp_path: Path) -> None:
    """New scanner facts produce specific dependency edges without AST re-traversal."""
    module = _module_scan(
        tmp_path,
        symbols=(
            FutureSymbolRecord(
                id=SymbolId("pkg.mod", "Worker"),
                kind="class",
                decorators=("dataclass",),
                annotation_names=("Helper", "ExternalType", "Alias", "MissingType"),
                base_names=("Base", "RemoteBase"),
            ),
            FutureSymbolRecord(id=SymbolId("pkg.mod", "Base"), kind="class"),
            FutureSymbolRecord(id=SymbolId("pkg.mod", "Helper"), kind="class"),
            FutureSymbolRecord(id=SymbolId("pkg.mod", "Worker.normalize"), kind="method"),
            FutureSymbolRecord(id=SymbolId("pkg.mod", "Worker.build"), kind="method"),
            FutureSymbolRecord(
                id=SymbolId("pkg.mod", "Worker.run"),
                kind="method",
                called_names=("Helper", "imported_factory"),
                called_paths=("self.normalize", "cls.build", "client.fetch"),
                owner_class="Worker",
                uses_globals=("Alias",),
            ),
        ),
    )

    edges = build_dependency_edges(module)

    assert _edge_summary(edges) == [
        ("Worker", "pkg.mod::Base", "inherits", "high"),
        ("Worker", "pkg.api.RemoteBase", "inherits", "medium"),
        ("Worker", "dataclasses.dataclass", "decorated_by", "medium"),
        ("Worker", "pkg.mod::Helper", "annotation", "high"),
        ("Worker", "pkg.api.ExternalType", "annotation", "medium"),
        ("Worker", "Alias", "annotation", "high"),
        ("Worker", "MissingType", "annotation", "low"),
        ("Worker.run", "pkg.mod::Helper", "calls", "high"),
        ("Worker.run", "pkg.api.imported_factory", "imports", "medium"),
        ("Worker.run", "pkg.mod::Worker.normalize", "calls_method", "high"),
        ("Worker.run", "pkg.mod::Worker.build", "calls_method", "high"),
        ("Worker.run", "pkg.api.client", "imports", "medium"),
        ("Worker.run", "Alias", "uses_global", "high"),
    ]


def test_dependency_edges_preserve_function_runtime_behavior(tmp_path: Path) -> None:
    """Existing function call, import, global, and unknown edges still resolve."""
    module = _module_scan(
        tmp_path,
        symbols=(
            FutureSymbolRecord(id=SymbolId("pkg.mod", "helper"), kind="function"),
            FutureSymbolRecord(
                id=SymbolId("pkg.mod", "score"),
                kind="function",
                called_names=("helper", "sqrt"),
                uses_globals=("RATE", "dynamic_name"),
            ),
        ),
    )

    edges = build_dependency_edges(module)

    assert _edge_summary(edges) == [
        ("score", "pkg.mod::helper", "calls", "high"),
        ("score", "math.sqrt", "imports", "medium"),
        ("score", "RATE", "uses_global", "high"),
        ("score", "dynamic_name", "unknown", "low"),
    ]


def _module_scan(
    tmp_path: Path,
    *,
    symbols: tuple[FutureSymbolRecord, ...],
) -> ModuleScan:
    return ModuleScan(
        module=ModuleId(name="pkg.mod", path=tmp_path / "mod.py"),
        imports=(
            ImportRecord(
                source_text="from dataclasses import dataclass",
                imported_names=("dataclass",),
                module="dataclasses",
                level=0,
                lineno=1,
                end_lineno=1,
            ),
            ImportRecord(
                source_text=(
                    "from pkg.api import RemoteBase, ExternalType, imported_factory, client"
                ),
                imported_names=("RemoteBase", "ExternalType", "imported_factory", "client"),
                module="pkg.api",
                level=0,
                lineno=2,
                end_lineno=2,
            ),
            ImportRecord(
                source_text="from math import sqrt",
                imported_names=("sqrt",),
                module="math",
                level=0,
                lineno=3,
                end_lineno=3,
            ),
        ),
        constants=(
            ConstantRecord(
                name="Alias",
                kind="literal_constant",
                source_text="Alias = int",
                lineno=4,
                end_lineno=4,
            ),
            ConstantRecord(
                name="RATE",
                kind="literal_constant",
                source_text="RATE = 1",
                lineno=5,
                end_lineno=5,
            ),
        ),
        symbols=cast("tuple[SymbolRecord, ...]", symbols),
        blockers=(),
        top_level_statement_lines=(),
    )


def _edge_summary(edges: tuple[DependencyEdge, ...]) -> list[tuple[str, str, str, str]]:
    return [
        (
            edge.src.qualname,
            edge.dst.stable_id if isinstance(edge.dst, SymbolId) else edge.dst,
            edge.kind,
            edge.confidence,
        )
        for edge in edges
    ]
