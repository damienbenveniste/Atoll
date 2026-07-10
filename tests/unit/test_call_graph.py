"""Focused tests for dependency-edge construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from atoll.analysis.call_graph import build_dependency_edges
from atoll.models import (
    CallSiteFact,
    ConstantRecord,
    DependencyEdge,
    ImportRecord,
    InvocationMode,
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
    call_sites: tuple[CallSiteFact, ...] = ()
    local_names: tuple[str, ...] = ()
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


def test_dependency_edges_preserve_invocation_mode_and_same_unit_evidence(
    tmp_path: Path,
) -> None:
    """Call-site facts enrich call edges without making ordinary calls atomic."""
    module = _module_scan(
        tmp_path,
        symbols=(
            FutureSymbolRecord(id=SymbolId("pkg.mod", "helper"), kind="function"),
            FutureSymbolRecord(id=SymbolId("pkg.mod", "Worker"), kind="class"),
            FutureSymbolRecord(id=SymbolId("pkg.mod", "Worker.fetch"), kind="method"),
            FutureSymbolRecord(
                id=SymbolId("pkg.mod", "Worker.run"),
                kind="method",
                owner_class="Worker",
                call_sites=(
                    _call_site("helper", "helper", "ordinary", 10),
                    _call_site("Worker", "Worker", "awaited", 11, requires_same_unit=True),
                    _call_site("self.fetch", "self", "awaited", 12),
                    _call_site("client.stream", "client", "async_iteration", 13),
                ),
            ),
        ),
    )

    edges = build_dependency_edges(module)

    assert _call_edge_details(edges) == [
        ("pkg.mod::helper", "calls", "ordinary", False, 10),
        ("pkg.mod::Worker", "calls", "awaited", True, 11),
        ("pkg.mod::Worker.fetch", "calls_method", "awaited", False, 12),
        ("pkg.api.client", "imports", "async_iteration", False, 13),
    ]


def test_dependency_edges_do_not_resolve_lexically_local_call_roots(
    tmp_path: Path,
) -> None:
    """Assignments and local imports shadow same-named module dependencies."""
    module = _module_scan(
        tmp_path,
        symbols=(
            FutureSymbolRecord(id=SymbolId("pkg.mod", "helper"), kind="function"),
            FutureSymbolRecord(
                id=SymbolId("pkg.mod", "hot"),
                kind="function",
                local_names=("helper", "client"),
                call_sites=(
                    _call_site("helper", "helper", "ordinary", 10),
                    _call_site("client.stream", "client", "ordinary", 11),
                ),
            ),
        ),
    )

    edges = build_dependency_edges(module)

    assert not any(
        edge.src.qualname == "hot" and edge.kind in {"calls", "imports"} for edge in edges
    )


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


def _call_edge_details(
    edges: tuple[DependencyEdge, ...],
) -> list[tuple[str, str, str | None, bool, int | None]]:
    return [
        (
            edge.dst.stable_id if isinstance(edge.dst, SymbolId) else edge.dst,
            edge.kind,
            edge.invocation_mode,
            edge.requires_same_unit,
            edge.lineno,
        )
        for edge in edges
        if edge.kind in {"calls", "calls_method", "imports"}
        and edge.src == SymbolId("pkg.mod", "Worker.run")
    ]


def _call_site(
    target: str,
    root_name: str,
    invocation_mode: InvocationMode,
    lineno: int,
    *,
    requires_same_unit: bool = False,
) -> CallSiteFact:
    return CallSiteFact(
        target=target,
        root_name=root_name,
        invocation_mode=invocation_mode,
        lineno=lineno,
        end_lineno=lineno,
        col_offset=4,
        end_col_offset=12,
        requires_same_unit=requires_same_unit,
    )
