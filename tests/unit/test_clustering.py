"""Tests for dependency graph and island clustering."""

from pathlib import Path

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.clustering import enrich_island_analysis
from atoll.models import ModuleId, SymbolId

FIXTURE_ROOT = Path("tests/fixtures/simple_project")
HIGH_CONFIDENCE_SCORE = 90


def test_ranking_fixture_produces_one_maximal_clean_island() -> None:
    """The plan's clean ranking chain is reported as one candidate island."""
    module = ModuleId(
        name="app.ranking",
        path=(FIXTURE_ROOT / "src" / "app" / "ranking.py").resolve(),
    )
    scan = enrich_island_analysis(scan_module(module))

    assert sorted(
        (edge.src.qualname, edge.dst.qualname if isinstance(edge.dst, SymbolId) else edge.dst)
        for edge in scan.dependency_edges
        if edge.kind == "calls"
    ) == [
        ("rank_candidates", "score_user"),
        ("score_user", "normalize_features"),
    ]
    assert [
        [symbol.qualname for symbol in candidate.symbols] for candidate in scan.island_candidates
    ] == [["normalize_features", "rank_candidates", "score_user"]]
    assert scan.island_candidates[0].score >= HIGH_CONFIDENCE_SCORE
    assert scan.poison_radii[0].poison.qualname == "debug_dump"
    assert scan.poison_radii[0].impacted == ()


def test_dynamic_global_dependency_rejects_candidate(tmp_path: Path) -> None:
    """A function depending on a dynamic global is rejected with a local blocker."""
    module_path = tmp_path / "dynamic_global.py"
    module_path.write_text(
        "RATE = load_rate()\n\ndef score(x: float) -> float:\n    return x * RATE\n",
        encoding="utf-8",
    )

    scan = enrich_island_analysis(scan_module(ModuleId(name="dynamic_global", path=module_path)))

    assert scan.island_candidates == ()
    assert {blocker.code for blocker in scan.symbols[0].blockers} == {"DYNAMIC_GLOBAL_DEP"}
