"""Architecture boundaries for generic scheduler execution-plan support."""

from pathlib import Path


def test_product_execution_plan_code_contains_no_pydantic_graph_special_cases() -> None:
    """Pydantic Graph remains benchmark evidence, never compiler product logic."""
    source_root = Path(__file__).resolve().parents[2] / "src" / "atoll"
    forbidden = ("pydantic_graph", "_graphiterator", "graphtask")
    matches = [
        f"{path.relative_to(source_root)}: {term}"
        for path in sorted(source_root.rglob("*.py"))
        for term in forbidden
        if term in path.read_text(encoding="utf-8").lower()
    ]

    assert matches == []
