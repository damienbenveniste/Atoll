"""Tests for Atoll managed shim edits."""

from pathlib import Path

import pytest

from atoll.generation.shim import insert_or_replace_shim, remove_shim
from atoll.models import EnabledIslandConfig


def _island(tmp_path: Path) -> EnabledIslandConfig:
    return EnabledIslandConfig(
        source_module="app.ranking",
        source_path=tmp_path / "src" / "app" / "ranking.py",
        sidecar_module="app._atoll_app_ranking",
        sidecar_path=tmp_path / ".atoll" / "sidecars" / "_atoll_app_ranking.py",
        symbols=("score_user", "rank_candidates"),
    )


def test_insert_replace_and_remove_shim(tmp_path: Path) -> None:
    """Managed shims are marker-delimited, replaceable, and removable."""
    island = _island(tmp_path)
    source = "def score_user() -> int:\n    return 1\n"

    inserted = insert_or_replace_shim(source, island)
    replaced = insert_or_replace_shim(inserted.new_text, island)
    removed = remove_shim(replaced.new_text, island)

    assert "# BEGIN ATOLL MANAGED: app.ranking" in inserted.new_text
    assert inserted.new_text.count("# BEGIN ATOLL MANAGED") == 1
    assert replaced.new_text.count("# BEGIN ATOLL MANAGED") == 1
    assert "# BEGIN ATOLL MANAGED" not in removed.new_text
    assert "def score_user" in removed.new_text


def test_remove_shim_without_block_is_noop(tmp_path: Path) -> None:
    """Removing a missing shim leaves source unchanged."""
    source = "VALUE = 1\n"

    edit = remove_shim(source, _island(tmp_path))

    assert edit.new_text == source
    assert edit.diff == ""


def test_shim_rejects_unbalanced_and_duplicate_markers(tmp_path: Path) -> None:
    """Invalid managed marker layout fails loudly."""
    island = _island(tmp_path)

    with pytest.raises(ValueError, match="unbalanced"):
        insert_or_replace_shim("# BEGIN ATOLL MANAGED: app.ranking\n", island)

    duplicate = (
        "# BEGIN ATOLL MANAGED: app.ranking\n"
        "# END ATOLL MANAGED: app.ranking\n"
        "# BEGIN ATOLL MANAGED: app.ranking\n"
        "# END ATOLL MANAGED: app.ranking\n"
    )
    with pytest.raises(ValueError, match="multiple"):
        insert_or_replace_shim(duplicate, island)
