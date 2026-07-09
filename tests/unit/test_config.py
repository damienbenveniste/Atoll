"""Tests for Atoll configuration files."""

from pathlib import Path

from atoll.config import disable_island, load_enabled_islands, write_atoll_config
from atoll.models import EnabledIslandConfig


def test_write_and_load_atoll_config(tmp_path: Path) -> None:
    """Atoll writes and reads enabled island configuration."""
    island = EnabledIslandConfig(
        source_module="app.ranking",
        source_path=tmp_path / "src" / "app" / "ranking.py",
        sidecar_module="app._atoll_app_ranking",
        sidecar_path=tmp_path / ".atoll" / "sidecars" / "_atoll_app_ranking.py",
        symbols=("score_user", "rank_candidates"),
    )

    config_path = write_atoll_config(tmp_path, (island,))
    loaded = load_enabled_islands(tmp_path)

    assert config_path == tmp_path / ".atoll.toml"
    assert loaded == (island,)


def test_disable_island_marks_config_entry_disabled(tmp_path: Path) -> None:
    """Disabling an island preserves the entry but marks it inactive."""
    island = EnabledIslandConfig(
        source_module="app.ranking",
        source_path=tmp_path / "src" / "app" / "ranking.py",
        sidecar_module="app._atoll_app_ranking",
        sidecar_path=tmp_path / ".atoll" / "sidecars" / "_atoll_app_ranking.py",
        symbols=("score_user",),
    )
    write_atoll_config(tmp_path, (island,))

    disabled = disable_island(tmp_path, "app.ranking")

    assert disabled[0].enabled is False
    assert load_enabled_islands(tmp_path)[0].enabled is False


def test_load_enabled_islands_skips_incomplete_entries(tmp_path: Path) -> None:
    """Incomplete island entries are ignored instead of crashing config loading."""
    (tmp_path / ".atoll.toml").write_text(
        "[tool.atoll]\n[[tool.atoll.island]]\nsource_module = 'app.ranking'\n",
        encoding="utf-8",
    )

    assert load_enabled_islands(tmp_path) == ()


def test_write_atoll_config_handles_absolute_paths_outside_root(tmp_path: Path) -> None:
    """Absolute paths outside the project root are preserved in config output."""
    outside = tmp_path.parent / "outside.py"
    island = EnabledIslandConfig(
        source_module="outside",
        source_path=outside,
        sidecar_module="_outside_atoll",
        sidecar_path=tmp_path / "src" / "_outside_atoll.py",
        symbols=("score",),
    )

    path = write_atoll_config(tmp_path, (island,))

    assert outside.as_posix() in path.read_text(encoding="utf-8")
