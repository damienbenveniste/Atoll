"""Integration tests for Atoll enable, generate, verify, and disable commands."""

from __future__ import annotations

import importlib.machinery
import json
import shutil
from pathlib import Path
from typing import cast

import pytest

from atoll.cli import main
from atoll.commands.build import BuildOptions
from atoll.config import load_enabled_islands
from atoll.models import CompileAttempt, PytestRunResult, VerifyResult
from atoll.report import CompilationReport

FIXTURE_ROOT = Path("tests/fixtures/simple_project")
EXIT_USAGE = 2


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

    sidecar_path = project_root / ".atoll" / "sidecars" / "_atoll_app_ranking.py"
    old_sidecar_path = project_root / "src" / "app" / "_ranking_atoll.py"
    source_path = project_root / "src" / "app" / "ranking.py"
    assert (project_root / ".atoll.toml").exists()
    assert sidecar_path.exists()
    assert not old_sidecar_path.exists()
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
    sidecar_path.unlink()
    assert main(["build", "--root", str(project_root), "--clean-first", "--inplace"]) == 0
    assert sidecar_path.exists()
    assert not build_marker.exists()
    artifact_paths = tuple(
        path
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
        for path in (project_root / ".atoll" / "artifacts").glob(f"{sidecar_path.stem}*{suffix}")
    )
    assert artifact_paths
    assert not (project_root / "build").exists()
    assert not (project_root / ".atoll" / "build").exists()
    build_report = _compilation_report(project_root)
    assert build_report["operation"] == "build"
    assert build_report["success"] is True
    assert build_report["summary"]["artifacts"] >= 1
    assert build_report["cleanup"]["removed"] == [".atoll/build"]
    assert build_report["islands"][0]["verification"] is None
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
    assert not old_sidecar_path.exists()
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


def test_compile_command_enables_builds_and_verifies_candidates(tmp_path: Path) -> None:
    """`atoll compile` is the one-step path for discovered candidate islands."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)

    exit_code = main(
        [
            "compile",
            "app.ranking",
            "--root",
            str(project_root),
            "--in-place",
            "--test",
            "pytest tests -q",
        ]
    )

    sidecar_path = project_root / ".atoll" / "sidecars" / "_atoll_app_ranking.py"
    source_path = project_root / "src" / "app" / "ranking.py"
    islands = load_enabled_islands(project_root)
    artifact_paths = tuple(
        path
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
        for path in (project_root / ".atoll" / "artifacts").glob(f"{sidecar_path.stem}*{suffix}")
    )
    assert exit_code == 0
    assert islands[0].sidecar_module == "app._atoll_app_ranking"
    assert islands[0].sidecar_path == sidecar_path.resolve()
    assert islands[0].symbols == ("normalize_features", "score_user", "rank_candidates")
    assert not sidecar_path.exists()
    assert artifact_paths
    assert not (project_root / "src" / "app" / "_ranking_atoll.py").exists()
    assert not (project_root / "build").exists()
    assert not (project_root / ".atoll" / "build").exists()
    assert not (project_root / ".atoll" / "sidecars").exists()
    assert "# BEGIN ATOLL MANAGED: app.ranking" in source_path.read_text(encoding="utf-8")
    report = _compilation_report(project_root)
    assert report["operation"] == "compile"
    assert report["success"] is True
    assert report["summary"]["islands"] == 1
    assert report["summary"]["symbols"] == len(islands[0].symbols)
    assert report["summary"]["verified"] == 1
    assert report["summary"]["verify_failures"] == 0
    assert report["summary"]["semantic_tests_run"] is True
    assert report["summary"]["semantic_test_failures"] == 0
    assert report["tests"] == {
        "command": ["pytest", "tests", "-q"],
        "exit_code": 0,
        "success": True,
    }
    assert report["islands"][0]["source_module"] == "app.ranking"
    assert report["islands"][0]["verification"] is not None
    assert report["cleanup"]["removed"] == [
        "<2 generated Python build inputs>",
        ".atoll/build",
    ]
    assert (project_root / ".atoll" / "compilation-report.md").exists()


def test_compile_command_reports_test_gate_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`atoll compile --test` fails the compile report when target tests fail."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)

    def fake_execute_build(options: BuildOptions) -> CompileAttempt:
        assert options.module_name == "app.ranking"
        return CompileAttempt(
            success=True,
            command=("mypyc",),
            stdout="",
            stderr="",
            artifact_paths=(tmp_path / "fixture.so",),
            duration_seconds=0.0,
        )

    def fake_execute_verify(*args: object, **kwargs: object) -> tuple[VerifyResult, ...]:
        assert args
        assert kwargs == {}
        return (
            VerifyResult(
                source_module="app.ranking",
                sidecar_module="app._atoll_app_ranking",
                active=True,
                compiled=True,
                origin=str(tmp_path / "fixture.so"),
                symbols=(("score_user", True),),
            ),
        )

    def fake_run_pytest_command(*args: object, **kwargs: object) -> PytestRunResult:
        assert args == ("pytest tests",)
        assert kwargs["require_compiled"] is True
        return PytestRunResult(command=("pytest", "tests"), exit_code=1, success=False)

    monkeypatch.setattr("atoll.cli.execute_build", fake_execute_build)
    monkeypatch.setattr("atoll.cli.execute_verify", fake_execute_verify)
    monkeypatch.setattr("atoll.cli.run_pytest_command", fake_run_pytest_command)

    exit_code = main(
        [
            "compile",
            "app.ranking",
            "--root",
            str(project_root),
            "--in-place",
            "--test",
            "pytest tests",
        ]
    )

    assert exit_code == 1
    report = _compilation_report(project_root)
    assert report["success"] is False
    assert report["summary"]["semantic_tests_run"] is True
    assert report["summary"]["semantic_test_failures"] == 1
    assert report["tests"] == {
        "command": ["pytest", "tests"],
        "exit_code": 1,
        "success": False,
    }
    assert report["cleanup"]["removed"] == []


def test_compile_command_reports_no_candidates(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The one-step compile command reports when scan finds nothing to compile."""
    (tmp_path / "src" / "pkg").mkdir(parents=True)

    exit_code = main(["compile", "--root", str(tmp_path), "--in-place"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "scan found no candidate islands to enable" in captured.out


def test_compile_command_reports_build_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The one-step compile command stops when mypyc build fails."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)

    def fake_execute_build(options: BuildOptions) -> CompileAttempt:
        assert options.module_name == "app.ranking"
        assert options.clean_first is True
        return CompileAttempt(
            success=False,
            command=("mypyc",),
            stdout="",
            stderr="MYPYC_TYPE_ERROR: fixture",
            artifact_paths=(),
            duration_seconds=0.0,
        )

    monkeypatch.setattr("atoll.cli.execute_build", fake_execute_build)

    exit_code = main(["compile", "app.ranking", "--root", str(project_root), "--in-place"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "MYPYC_TYPE_ERROR: fixture" in captured.out
    report = _compilation_report(project_root)
    assert report["operation"] == "compile"
    assert report["success"] is False
    assert report["build"]["stderr"] == "MYPYC_TYPE_ERROR: fixture"
    assert report["summary"]["verified"] == 0


def test_compile_command_reports_verify_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one-step compile command requires compiled runtime routing."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)

    def fake_execute_build(options: BuildOptions) -> CompileAttempt:
        assert options.module_name == "app.ranking"
        return CompileAttempt(
            success=True,
            command=("mypyc",),
            stdout="",
            stderr="",
            artifact_paths=(tmp_path / "fixture.so",),
            duration_seconds=0.0,
        )

    def fake_execute_verify(*args: object, **kwargs: object) -> tuple[VerifyResult, ...]:
        assert args
        assert kwargs == {}
        return (
            VerifyResult(
                source_module="app.ranking",
                sidecar_module="app._atoll_app_ranking",
                active=True,
                compiled=False,
                origin=None,
                symbols=(("score_user", False),),
                error="sidecar is active but is not a compiled extension",
            ),
        )

    monkeypatch.setattr("atoll.cli.execute_build", fake_execute_build)
    monkeypatch.setattr("atoll.cli.execute_verify", fake_execute_verify)

    exit_code = main(["compile", "app.ranking", "--root", str(project_root), "--in-place"])

    assert exit_code == 1
    report = _compilation_report(project_root)
    assert report["operation"] == "compile"
    assert report["success"] is False
    assert report["summary"]["verify_failures"] == 1
    assert report["islands"][0]["verification"] is not None


def test_enable_requires_module_and_symbols_without_all_candidates(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Manual enable mode requires both a target module and explicit symbols."""
    exit_code = main(["enable", "--root", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == EXIT_USAGE
    assert "enable requires MODULE and --symbols" in captured.out


def test_enable_all_candidates_rejects_manual_symbols(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """All-candidates mode owns symbol selection."""
    exit_code = main(
        [
            "enable",
            "--root",
            str(tmp_path),
            "--all-candidates",
            "--symbols",
            "score_user",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == EXIT_USAGE
    assert "--symbols cannot be used with --all-candidates" in captured.out


def test_enable_all_candidates_rejects_manual_sidecar(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """All-candidates mode derives one sidecar name per source module."""
    exit_code = main(
        [
            "enable",
            "--root",
            str(tmp_path),
            "--all-candidates",
            "--sidecar",
            "app.custom_sidecar",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == EXIT_USAGE
    assert "--sidecar can only be used with manual single-module enable" in captured.out


def test_enable_all_candidates_enables_scan_candidates(tmp_path: Path) -> None:
    """`atoll enable --all-candidates` enables candidate unions by source module."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)

    exit_code = main(["enable", "--root", str(project_root), "--all-candidates", "--yes"])

    sidecar_path = project_root / ".atoll" / "sidecars" / "_atoll_app_ranking.py"
    source_path = project_root / "src" / "app" / "ranking.py"
    islands = load_enabled_islands(project_root)
    assert exit_code == 0
    assert tuple(island.source_module for island in islands) == ("app.ranking",)
    assert islands[0].symbols == ("normalize_features", "score_user", "rank_candidates")
    assert sidecar_path.exists()
    assert "# BEGIN ATOLL MANAGED: app.ranking" in source_path.read_text(encoding="utf-8")
    assert main(["verify", "--root", str(project_root)]) == 0


def test_enable_all_candidates_can_be_limited_to_module(tmp_path: Path) -> None:
    """All-candidates mode can enable candidates from one named source module."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)

    exit_code = main(
        [
            "enable",
            "app.ranking",
            "--root",
            str(project_root),
            "--all-candidates",
            "--yes",
        ]
    )

    islands = load_enabled_islands(project_root)
    assert exit_code == 0
    assert tuple(island.source_module for island in islands) == ("app.ranking",)
    assert islands[0].symbols == ("normalize_features", "score_user", "rank_candidates")


def test_enable_all_candidates_build_option_runs_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`atoll enable --all-candidates --build` compiles after enabling candidates."""
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
            "--all-candidates",
            "--build",
            "--yes",
        ]
    )

    assert exit_code == 0
    assert calls == [BuildOptions(root=project_root, module_name="app.ranking")]


def test_enable_all_candidates_dry_run_does_not_write(tmp_path: Path) -> None:
    """Dry-run all-candidates mode reports candidates without writing files."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)

    exit_code = main(
        ["enable", "--root", str(project_root), "--all-candidates", "--dry-run", "--yes"]
    )

    assert exit_code == 0
    assert not (project_root / ".atoll.toml").exists()
    assert not (project_root / ".atoll" / "sidecars" / "_atoll_app_ranking.py").exists()


def test_enable_all_candidates_reports_no_project_candidates(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """All-candidates mode reports when no discovered module has candidates."""
    module_path = tmp_path / "src" / "app" / "settings.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("VALUE = 1\n", encoding="utf-8")

    exit_code = main(["enable", "--root", str(tmp_path), "--all-candidates"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "scan found no candidate islands to enable" in captured.out


def test_enable_all_candidates_reports_module_without_candidates(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Module-limited all-candidates mode reports when that module has no candidates."""
    module_path = tmp_path / "src" / "app" / "settings.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("VALUE = 1\n", encoding="utf-8")

    exit_code = main(["enable", "app.settings", "--root", str(tmp_path), "--all-candidates"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "module has no candidate islands: app.settings" in captured.out


def _compilation_report(project_root: Path) -> CompilationReport:
    path = project_root / ".atoll" / "compilation-report.json"
    return cast(CompilationReport, json.loads(path.read_text(encoding="utf-8")))
