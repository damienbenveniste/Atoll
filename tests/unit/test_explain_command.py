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


def test_explain_module_lists_module_blockers_without_suppressing_candidates(
    tmp_path: Path,
) -> None:
    """Module-level typing blockers are visible without hiding clean candidates."""
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
                "def helper(value: int) -> int:",
                "    return value + 1",
                "",
                "def candidate(value: int) -> int:",
                "    adjusted = helper(value)",
                "    return adjusted",
                "",
            ]
        ),
        encoding="utf-8",
    )

    output = execute_explain(ExplainOptions(root=tmp_path, target="pkg.mod", mypy_enabled=False))

    assert "Candidate islands: 1" in output
    assert "candidate" in output
    assert "Module blockers:" in output
    assert "MYPYC_UNSUPPORTED_TYPEVAR" in output
    assert "infer_variance" in output


def test_explain_symbol_reports_native_proofs_and_fallback_boundaries(tmp_path: Path) -> None:
    """Symbol output exposes fixed-width domains instead of only island safety scores."""
    package = tmp_path / "src" / "pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "kernels.py").write_text(
        "def polynomial(value: int) -> int:\n    return value * value + 1\n",
        encoding="utf-8",
    )

    output = execute_explain(
        ExplainOptions(
            root=tmp_path,
            target="pkg.kernels::polynomial",
            mypy_enabled=False,
        )
    )

    assert "Native optimization:" in output
    assert "Fixed-width i32:" in output
    assert "return=" in output
    assert "Fallback: buffer/unsupported-annotation" in output


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
