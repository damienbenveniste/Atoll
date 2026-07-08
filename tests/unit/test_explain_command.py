"""Focused tests for Atoll explain output."""

from pathlib import Path

import pytest

from atoll.commands.explain import ExplainOptions, execute_explain

FIXTURE_ROOT = Path("tests/fixtures/simple_project")


def test_explain_symbol_with_dependency_edges() -> None:
    """Symbol explanation includes same-module dependency edges."""
    output = execute_explain(
        ExplainOptions(
            root=FIXTURE_ROOT,
            target="app.ranking::score_user",
            mypy_enabled=False,
        )
    )

    assert "calls/high: app.ranking::normalize_features" in output


def test_explain_missing_module_and_symbol_fail() -> None:
    """Invalid explain targets fail with useful errors."""
    with pytest.raises(ValueError, match="module not found"):
        execute_explain(ExplainOptions(root=FIXTURE_ROOT, target="app.missing", mypy_enabled=False))
    with pytest.raises(ValueError, match="symbol not found"):
        execute_explain(
            ExplainOptions(
                root=FIXTURE_ROOT,
                target="app.ranking::missing",
                mypy_enabled=False,
            )
        )
    with pytest.raises(ValueError, match="module name"):
        execute_explain(ExplainOptions(root=FIXTURE_ROOT, target="::missing", mypy_enabled=False))
