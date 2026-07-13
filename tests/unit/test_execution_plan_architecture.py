"""Architecture boundaries for generic scheduler execution-plan support."""

from pathlib import Path


def test_product_code_contains_no_benchmark_project_or_symbol_special_cases() -> None:
    """Benchmark fixtures and pinned targets never become compiler product logic."""
    source_root = Path(__file__).resolve().parents[2] / "src" / "atoll"
    forbidden = (
        "pydantic_graph",
        "_graphiterator",
        "graphtask",
        "_run_tracked_task",
        "native_optimization_fixture",
        "scalar_polynomial",
        "direct_chain_root",
        "residual_async_profile",
        "source_optimization_fixture",
    )
    matches = [
        f"{path.relative_to(source_root)}: {term}"
        for path in sorted(source_root.rglob("*.py"))
        for term in forbidden
        if term in path.read_text(encoding="utf-8").lower()
    ]

    assert matches == []
