"""Tests for Atoll sidecar generation."""

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
        sidecar_path=expected_sidecar_path(scan, sidecar_module),
        symbols=("score_user", "rank_candidates"),
    )

    generation = generate_sidecar(scan, island)

    assert sidecar_module == "app._ranking_atoll"
    assert island.sidecar_path.name == "_ranking_atoll.py"
    assert '"source_module": "app.ranking"' in generation.source_text
    assert "def normalize_features" in generation.source_text
    assert "def debug_dump" not in generation.source_text
