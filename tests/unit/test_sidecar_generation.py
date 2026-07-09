"""Tests for Atoll sidecar generation."""

import ast
from pathlib import Path

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.clustering import enrich_island_analysis
from atoll.generation.sidecar import (
    build_sidecar_plan,
    default_sidecar_module,
    expected_sidecar_path,
    generate_sidecar,
)
from atoll.models import EnabledIslandConfig, ModuleId, ModuleScan

FIXTURE_ROOT = Path("tests/fixtures/simple_project")


def _ranking_scan() -> tuple[ModuleId, ModuleScan]:
    module = ModuleId(
        name="app.ranking",
        path=(FIXTURE_ROOT / "src" / "app" / "ranking.py").resolve(),
    )
    return module, enrich_island_analysis(scan_module(module))


def _imports_from_typing(source: str, name: str) -> bool:
    tree = ast.parse(source)
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "typing"
        and any(alias.name == name for alias in node.names)
        for node in tree.body
    )


def test_sidecar_plan_expands_same_module_helpers() -> None:
    """Selected exported symbols include clean same-module helper functions."""
    _, scan = _ranking_scan()

    plan = build_sidecar_plan(scan, ("score_user", "rank_candidates"))

    assert plan.included_symbol_names == (
        "normalize_features",
        "score_user",
        "rank_candidates",
    )
    assert [constant.name for constant in plan.constants] == ["DEFAULT_WEIGHT"]


def test_generate_sidecar_renders_atoll_metadata() -> None:
    """Generated sidecars use Atoll naming and include copied source slices."""
    module, scan = _ranking_scan()
    sidecar_module = default_sidecar_module(module.name)
    island = EnabledIslandConfig(
        source_module=module.name,
        source_path=module.path,
        sidecar_module=sidecar_module,
        sidecar_path=expected_sidecar_path(FIXTURE_ROOT, sidecar_module),
        symbols=("score_user", "rank_candidates"),
    )

    generation = generate_sidecar(scan, island)

    assert sidecar_module == "app._atoll_app_ranking"
    assert island.sidecar_path == (
        FIXTURE_ROOT.resolve() / ".atoll" / "sidecars" / "_atoll_app_ranking.py"
    )
    assert '"source_module": "app.ranking"' in generation.source_text
    assert "def normalize_features" in generation.source_text
    assert "def debug_dump" not in generation.source_text


def test_generate_sidecar_imports_same_module_annotation_classes(
    tmp_path: Path,
) -> None:
    """Type-only same-module class references are preserved for mypyc checking."""
    module_path = tmp_path / "sample.py"
    module_path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "class Row:",
                "    pass",
                "",
                "def wrap(row: Row) -> list[Row]:",
                "    return [row]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    module = ModuleId(name="sample", path=module_path)
    scan = enrich_island_analysis(scan_module(module))
    island = EnabledIslandConfig(
        source_module=module.name,
        source_path=module.path,
        sidecar_module="sample_atoll",
        sidecar_path=tmp_path / "sample_atoll.py",
        symbols=("wrap",),
    )

    generation = generate_sidecar(scan, island)

    assert "from typing import TYPE_CHECKING" in generation.source_text
    assert "if TYPE_CHECKING:\n    from sample import Row" in generation.source_text
    assert "class Row" not in generation.source_text


def test_generate_sidecar_simplifies_runtime_import_generics_for_mypyc(
    tmp_path: Path,
) -> None:
    """Imported runtime classes are not subscripted in generated sidecars."""
    module_path = tmp_path / "sample.py"
    module_path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "from package.nodes import Box",
                "",
                "def make_box(value: int):",
                "    return Box[int](value)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    module = ModuleId(name="sample", path=module_path)
    scan = enrich_island_analysis(scan_module(module))
    island = EnabledIslandConfig(
        source_module=module.name,
        source_path=module.path,
        sidecar_module="sample_atoll",
        sidecar_path=tmp_path / "sample_atoll.py",
        symbols=("make_box",),
    )

    generation = generate_sidecar(scan, island)

    tree = ast.parse(generation.source_text)
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "typing"
        and any(alias.name == "Any" for alias in node.names)
        for node in tree.body
    )
    assert "def make_box(value: int) -> Any:" in generation.source_text
    assert "return Box(value)" in generation.source_text
    assert "Box[int]" not in generation.source_text


def test_generate_sidecar_erases_typevar_annotations_for_mypyc(
    tmp_path: Path,
) -> None:
    """Generated sidecars erase TypeVar-only annotations that mypyc rejects."""
    module_path = tmp_path / "sample.py"
    module_path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "from typing_extensions import TypeVar as TV",
                "",
                "T = TV('T', infer_variance=True, default=str)",
                "",
                "def helper(value: T) -> T:",
                "    return value",
                "",
                "def identity(value: T) -> T:",
                "    return helper(value)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    module = ModuleId(name="sample", path=module_path)
    scan = enrich_island_analysis(scan_module(module))
    island = EnabledIslandConfig(
        source_module=module.name,
        source_path=module.path,
        sidecar_module="sample_atoll",
        sidecar_path=tmp_path / "sample_atoll.py",
        symbols=("identity",),
    )

    generation = generate_sidecar(scan, island)

    assert _imports_from_typing(generation.source_text, "Any")
    assert "def helper(value: Any) -> Any:" in generation.source_text
    assert "def identity(value: Any) -> Any:" in generation.source_text
    assert "TV(" not in generation.source_text
    assert "TypeVar" not in generation.source_text


def test_generate_sidecar_erases_local_type_alias_annotations_for_mypyc(
    tmp_path: Path,
) -> None:
    """Generated sidecars erase local runtime type aliases from annotations."""
    module_path = tmp_path / "sample.py"
    module_path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "from typing import TypeAlias",
                "",
                "NodeID: TypeAlias = str",
                "",
                "def normalize(value: NodeID) -> NodeID:",
                "    return value",
                "",
            ]
        ),
        encoding="utf-8",
    )
    module = ModuleId(name="sample", path=module_path)
    scan = enrich_island_analysis(scan_module(module))
    island = EnabledIslandConfig(
        source_module=module.name,
        source_path=module.path,
        sidecar_module="sample_atoll",
        sidecar_path=tmp_path / "sample_atoll.py",
        symbols=("normalize",),
    )

    generation = generate_sidecar(scan, island)

    assert _imports_from_typing(generation.source_text, "Any")
    assert "def normalize(value: Any) -> Any:" in generation.source_text
    assert "NodeID" not in generation.source_text


def test_generate_sidecar_prunes_annotation_only_imports_for_mypyc(
    tmp_path: Path,
) -> None:
    """Annotation-only imported types are erased so mypyc does not analyze them."""
    module_path = tmp_path / "sample.py"
    module_path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "from heavy.typing_graph import DecisionBranchBuilder",
                "from runtime.boxes import Box",
                "",
                "def wrap(value: DecisionBranchBuilder[int]) -> DecisionBranchBuilder[int]:",
                "    return Box(value).value",
                "",
            ]
        ),
        encoding="utf-8",
    )
    module = ModuleId(name="sample", path=module_path)
    scan = enrich_island_analysis(scan_module(module))
    island = EnabledIslandConfig(
        source_module=module.name,
        source_path=module.path,
        sidecar_module="sample_atoll",
        sidecar_path=tmp_path / "sample_atoll.py",
        symbols=("wrap",),
    )

    generation = generate_sidecar(scan, island)

    assert "from heavy.typing_graph import DecisionBranchBuilder" not in generation.source_text
    assert "from runtime.boxes import Box" in generation.source_text
    assert _imports_from_typing(generation.source_text, "Any")
    assert "def wrap(value: Any) -> Any:" in generation.source_text


def test_generate_sidecar_erases_typing_module_typevar_annotations(
    tmp_path: Path,
) -> None:
    """TypeVars created through a typing module alias are also erased in annotations."""
    module_path = tmp_path / "sample.py"
    module_path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import typing as t",
                "",
                "T: object = t.TypeVar('T', infer_variance=True)",
                "",
                "def choose(*items: T, **mapping: T) -> T | None:",
                "    if items:",
                "        return items[0]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    module = ModuleId(name="sample", path=module_path)
    scan = enrich_island_analysis(scan_module(module))
    island = EnabledIslandConfig(
        source_module=module.name,
        source_path=module.path,
        sidecar_module="sample_atoll",
        sidecar_path=tmp_path / "sample_atoll.py",
        symbols=("choose",),
    )

    generation = generate_sidecar(scan, island)

    assert _imports_from_typing(generation.source_text, "Any")
    assert "def choose(*items: Any, **mapping: Any) -> Any | None:" in generation.source_text
    assert generation.source_text.count("return None") == 1
    assert "t.TypeVar" not in generation.source_text


def test_generate_sidecar_makes_optional_fallthrough_explicit_for_mypyc(
    tmp_path: Path,
) -> None:
    """Optional-return functions get explicit `None` fallthroughs for mypyc."""
    module_path = tmp_path / "sample.py"
    module_path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "def parent(flag: bool) -> dict[str, object] | None:",
                "    if flag:",
                "        return {'ok': True}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    module = ModuleId(name="sample", path=module_path)
    scan = enrich_island_analysis(scan_module(module))
    island = EnabledIslandConfig(
        source_module=module.name,
        source_path=module.path,
        sidecar_module="sample_atoll",
        sidecar_path=tmp_path / "sample_atoll.py",
        symbols=("parent",),
    )

    generation = generate_sidecar(scan, island)

    assert "def parent(flag: bool) -> dict[str, object] | None:" in generation.source_text
    assert generation.source_text.count("return None") == 1


def test_generate_sidecar_makes_typing_optional_fallthroughs_explicit(
    tmp_path: Path,
) -> None:
    """Optional and Union annotations also get explicit fallthrough returns."""
    module_path = tmp_path / "sample.py"
    module_path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import typing",
                "from typing import Optional, Union",
                "",
                "async def from_optional(flag: bool) -> Optional[int]:",
                "    if flag:",
                "        return 1",
                "",
                "def from_union(flag: bool) -> Union[int, None]:",
                "    if flag:",
                "        return 2",
                "",
                "def from_attribute(flag: bool) -> typing.Optional[int]:",
                "    if flag:",
                "        return 3",
                "",
            ]
        ),
        encoding="utf-8",
    )
    module = ModuleId(name="sample", path=module_path)
    scan = enrich_island_analysis(scan_module(module))
    island = EnabledIslandConfig(
        source_module=module.name,
        source_path=module.path,
        sidecar_module="sample_atoll",
        sidecar_path=tmp_path / "sample_atoll.py",
        symbols=("from_optional", "from_union", "from_attribute"),
    )

    generation = generate_sidecar(scan, island)
    expected_none_returns = 3

    assert "async def from_optional" in generation.source_text
    assert generation.source_text.count("return None") == expected_none_returns
