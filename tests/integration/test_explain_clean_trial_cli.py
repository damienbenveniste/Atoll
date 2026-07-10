"""Integration tests for explain, clean, and trial commands."""

from __future__ import annotations

import importlib.machinery
import json
import shutil
import sys
from pathlib import Path

import pytest

from atoll.cli import main
from atoll.config import write_atoll_config
from atoll.models import EnabledIslandConfig

FIXTURE_ROOT = Path("tests/fixtures/simple_project")


def test_explain_command_reports_module_and_symbol(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`atoll explain` reports module candidates and symbol blockers."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)

    module_exit = main(["explain", "app.ranking", "--root", str(project_root), "--no-mypy"])
    symbol_exit = main(
        ["explain", "app.ranking::debug_dump", "--root", str(project_root), "--no-mypy"]
    )

    captured = capsys.readouterr()
    assert module_exit == 0
    assert symbol_exit == 0
    assert "Candidate islands: 1" in captured.out
    assert "DYN_GETATTR_DYNAMIC" in captured.out


def test_clean_command_removes_cache_and_compiled_artifacts(tmp_path: Path) -> None:
    """`atoll clean --all` removes cache/build dirs and compiled sidecar artifacts."""
    sidecar_path = tmp_path / ".atoll" / "sidecars" / "_atoll_app_ranking.py"
    sidecar_path.parent.mkdir(parents=True)
    sidecar_path.write_text("", encoding="utf-8")
    artifact = (tmp_path / ".atoll" / "artifacts").joinpath(
        f"{sidecar_path.stem}{importlib.machinery.EXTENSION_SUFFIXES[0]}"
    )
    artifact.parent.mkdir(parents=True)
    artifact.write_text("", encoding="utf-8")
    cache_dir = tmp_path / ".atoll" / "cache"
    compile_cache_dir = cache_dir / "compile"
    mypy_cache_dir = cache_dir / "mypy" / "source-clean"
    build_dir = tmp_path / ".atoll" / "build"
    compile_cache_dir.mkdir(parents=True)
    mypy_cache_dir.mkdir(parents=True)
    build_dir.mkdir(parents=True)
    write_atoll_config(
        tmp_path,
        (
            EnabledIslandConfig(
                source_module="app.ranking",
                source_path=tmp_path / "src" / "app" / "ranking.py",
                sidecar_module="app._atoll_app_ranking",
                sidecar_path=sidecar_path,
                symbols=("score_user",),
            ),
        ),
    )

    exit_code = main(["clean", "--root", str(tmp_path), "--all"])

    assert exit_code == 0
    assert not artifact.exists()
    assert not cache_dir.exists()
    assert not compile_cache_dir.exists()
    assert not mypy_cache_dir.exists()
    assert not build_dir.exists()


def test_trial_command_builds_typed_region_wheel_and_runs_pytest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`atoll trial` compiles selected regions and runs fixture tests/benchmarks."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    source_path = project_root / "src" / "app" / "ranking.py"
    original_source = source_path.read_bytes()

    exit_code = main(
        [
            "trial",
            "--root",
            str(project_root),
            "--candidate",
            "app.ranking::score_user,rank_candidates",
            "--test",
            "python -m pytest tests",
            "--benchmark",
            "python -m pytest tests",
            "--require-compiled",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert source_path.read_bytes() == original_source
    assert (
        "Atoll trial selected 1 candidate(s), 2 member(s), and compiled 2 public binding(s)."
        in captured.out
    )
    assert "Atoll trial temporary artifacts cleaned." in captured.out
    assert "Atoll trial benchmark exit code: 0" in captured.out
    assert not (project_root / ".atoll" / "cache").exists()


def test_trial_owns_its_gates_instead_of_running_compile_policy(
    tmp_path: Path,
) -> None:
    """Trial flags run once even when normal compile policy would reject the wheel."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    pyproject = project_root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8")
        + "\n".join(
            (
                "",
                "[tool.atoll.compile]",
                (f'test_command = [{json.dumps(sys.executable)}, "-c", "raise SystemExit(9)"]'),
                "",
            )
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "trial",
            "--root",
            str(project_root),
            "--candidate",
            "app.ranking::score_user",
            "--test",
            "python -m pytest tests",
        ]
    )

    assert exit_code == 0
