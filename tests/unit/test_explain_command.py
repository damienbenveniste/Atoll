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


def test_explain_module_describes_candidate_score_and_risk() -> None:
    """Module explanations spell out score and risk meaning."""
    output = execute_explain(
        ExplainOptions(root=FIXTURE_ROOT, target="app.ranking", mypy_enabled=False)
    )

    assert "90/100, very promising scan-only candidate" in output
    assert "low extraction risk" in output
    assert "symbols:" in output
    assert "normalize_features" in output
    assert "score_user" in output
    assert "rank_candidates" in output


def test_explain_module_lists_module_blockers(tmp_path: Path) -> None:
    """Module-level blockers are visible when they suppress candidate islands."""
    package = tmp_path / "src" / "pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "mod.py").write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "from typing_extensions import TypeVar",
                "",
                "T = TypeVar('T', infer_variance=True)",
                "",
                "def candidate(value: int) -> int:",
                "    return value + 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    output = execute_explain(ExplainOptions(root=tmp_path, target="pkg.mod", mypy_enabled=False))

    assert "Candidate islands: 0" in output
    assert "Module blockers:" in output
    assert "MYPYC_UNSUPPORTED_TYPEVAR" in output
    assert "infer_variance" in output


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
