"""Tests for Atoll configuration files."""

from pathlib import Path

import pytest

from atoll.config import (
    disable_island,
    load_compile_config,
    load_enabled_islands,
    write_atoll_config,
)
from atoll.models import CompileConfig, EnabledIslandConfig


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


def test_load_compile_config_reads_argv_and_performance_policy(tmp_path: Path) -> None:
    """Compile policy preserves argv arrays and validates numeric gates."""
    (tmp_path / "pyproject.toml").write_text(
        "\n".join(
            [
                "[tool.atoll.compile]",
                'backends = ["cython", "mypyc"]',
                'test_command = ["pytest", "-q"]',
                'benchmark_command = ["python", "benchmarks/workload.py"]',
                "benchmark_warmups = 2",
                "benchmark_samples = 9",
                "minimum_speedup = 1.25",
                "",
            ]
        ),
        encoding="utf-8",
    )

    config = load_compile_config(tmp_path)

    assert config == CompileConfig(
        backends=("cython", "mypyc"),
        test_command=("pytest", "-q"),
        benchmark_command=("python", "benchmarks/workload.py"),
        benchmark_warmups=2,
        benchmark_samples=9,
        minimum_speedup=1.25,
    )


def test_load_compile_config_defaults_when_section_is_absent(tmp_path: Path) -> None:
    """Projects need no Atoll configuration for source-clean compilation."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'sample'\n", encoding="utf-8")

    assert load_compile_config(tmp_path) == CompileConfig()


@pytest.mark.parametrize(
    ("compile_lines", "message"),
    [
        (["backends = []"], "backends"),
        (["test_command = 'pytest'"], "test_command"),
        (
            ['benchmark_command = ["python", "bench.py"]'],
            "requires test_command",
        ),
        (["benchmark_warmups = -1"], "benchmark_warmups"),
        (["benchmark_samples = 0"], "benchmark_samples"),
        (["minimum_speedup = 0"], "minimum_speedup"),
    ],
)
def test_load_compile_config_rejects_invalid_policy(
    tmp_path: Path,
    compile_lines: list[str],
    message: str,
) -> None:
    """Invalid compile settings fail discovery instead of weakening a gate."""
    (tmp_path / "pyproject.toml").write_text(
        "\n".join(["[tool.atoll.compile]", *compile_lines, ""]),
        encoding="utf-8",
    )

    with pytest.raises((TypeError, ValueError), match=message):
        load_compile_config(tmp_path)
