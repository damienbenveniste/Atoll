"""Integration tests for Atoll enable, generate, verify, and disable commands."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from atoll.cli import main
from atoll.commands.build import BuildOptions
from atoll.models import CompileAttempt

FIXTURE_ROOT = Path("tests/fixtures/simple_project")


def test_enable_generate_verify_and_disable_workflow(tmp_path: Path) -> None:
    """Atoll can route enabled symbols through a generated pure-Python sidecar."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)

    assert (
        main(
            [
                "enable",
                "app.ranking",
                "--root",
                str(project_root),
                "--symbols",
                "score_user,rank_candidates",
                "--yes",
            ]
        )
        == 0
    )

    sidecar_path = project_root / "src" / "app" / "_ranking_atoll.py"
    source_path = project_root / "src" / "app" / "ranking.py"
    assert (project_root / ".atoll.toml").exists()
    assert sidecar_path.exists()
    assert "# BEGIN ATOLL MANAGED: app.ranking" in source_path.read_text(encoding="utf-8")
    assert main(["verify", "--root", str(project_root)]) == 0
    assert main(["verify", "--root", str(project_root), "--require-compiled"]) == 1
    assert main(["generate", "--root", str(project_root), "--check"]) == 0

    sidecar_path.write_text(
        f"{sidecar_path.read_text(encoding='utf-8')}\n# stale\n",
        encoding="utf-8",
    )
    assert main(["generate", "--root", str(project_root), "--check"]) == 1
    assert main(["generate", "--root", str(project_root)]) == 0
    build_marker = project_root / ".atoll" / "build" / "stale.txt"
    build_marker.parent.mkdir(parents=True)
    build_marker.write_text("stale", encoding="utf-8")
    assert main(["build", "--root", str(project_root), "--clean-first", "--inplace"]) == 0
    assert not build_marker.exists()
    assert main(["verify", "--root", str(project_root), "--require-compiled"]) == 0

    assert (
        main(
            [
                "disable",
                "app.ranking",
                "--root",
                str(project_root),
                "--delete-sidecar",
            ]
        )
        == 0
    )
    assert not sidecar_path.exists()
    assert "# BEGIN ATOLL MANAGED" not in source_path.read_text(encoding="utf-8")


def test_build_command_succeeds_without_enabled_sidecars(tmp_path: Path) -> None:
    """The build command is a no-op success when nothing is enabled."""
    project_root = tmp_path / "empty_project"
    (project_root / "src").mkdir(parents=True)

    assert main(["build", "--root", str(project_root)]) == 0


def test_enable_build_option_runs_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`atoll enable --build` invokes the build command for the enabled module."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    calls: list[BuildOptions] = []

    def fake_execute_build(options: BuildOptions) -> CompileAttempt:
        calls.append(options)
        return CompileAttempt(
            success=True,
            command=("mypyc",),
            stdout="",
            stderr="",
            artifact_paths=(tmp_path / "fixture.so",),
            duration_seconds=0.0,
        )

    monkeypatch.setattr("atoll.cli.execute_build", fake_execute_build)

    exit_code = main(
        [
            "enable",
            "app.ranking",
            "--root",
            str(project_root),
            "--symbols",
            "score_user",
            "--build",
            "--yes",
        ]
    )

    assert exit_code == 0
    assert calls == [BuildOptions(root=project_root, module_name="app.ranking")]
