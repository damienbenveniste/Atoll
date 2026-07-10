"""Tests for source-clean typed-region trial behavior."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from atoll.commands import trial as trial_command
from atoll.commands.package import PackageCommandResult, PackageOptions
from atoll.commands.trial import TrialOptions, execute_trial
from atoll.models import CompileAttempt, PytestRunResult, SymbolId

FIXTURE_ROOT = Path("tests/fixtures/simple_project")
EXPECTED_COMMAND_COUNT = 2


def test_trial_reports_no_candidates(tmp_path: Path) -> None:
    """Trial fails clearly when scan finds nothing to try."""
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "__init__.py").write_text("", encoding="utf-8")

    result = execute_trial(TrialOptions(root=tmp_path, top=1))

    assert result.success is False
    assert result.error == "no trial candidates selected"


def test_trial_delegates_candidate_union_to_source_clean_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidate members reach one package build without legacy sidecar generation."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    captured: list[PackageOptions] = []

    def failing_package(options: PackageOptions) -> PackageCommandResult:
        captured.append(options)
        return _package_result(options, success=False, error="MYPYC_TYPE_ERROR: fixture")

    monkeypatch.setattr(trial_command, "execute_package", failing_package)

    result = execute_trial(
        TrialOptions(root=project_root, candidate="app.ranking::score_user,rank_candidates")
    )

    assert result.success is False
    assert result.error == "MYPYC_TYPE_ERROR: fixture"
    assert captured[0].keep_install_tree is True
    assert captured[0].run_quality_gates is False
    assert captured[0].cache_dir is not None
    assert captured[0].cache_dir.parent.name.startswith("atoll-trial-")
    assert captured[0].selected_members == (
        SymbolId(module="app.ranking", qualname="score_user"),
        SymbolId(module="app.ranking", qualname="rank_candidates"),
    )
    assert not result.artifact_root.exists()


def test_trial_runs_tests_against_compiled_install_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test and benchmark commands receive the retained package install root."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    commands: list[tuple[str, Path, tuple[Path, ...], bool]] = []

    def successful_package(options: PackageOptions) -> PackageCommandResult:
        result = _package_result(options, success=True)
        result.install_root.mkdir(parents=True)
        return result

    def successful_pytest(
        command: str,
        *,
        root: Path,
        source_roots: tuple[Path, ...],
        require_compiled: bool,
    ) -> PytestRunResult:
        commands.append((command, root, source_roots, require_compiled))
        return PytestRunResult(command=("pytest",), exit_code=0, success=True)

    monkeypatch.setattr(trial_command, "execute_package", successful_package)
    monkeypatch.setattr(trial_command, "run_pytest_command", successful_pytest)

    result = execute_trial(
        TrialOptions(
            root=project_root,
            candidate="app.ranking::score_user,rank_candidates",
            test_command="pytest tests",
            benchmark_command="pytest benchmarks",
        )
    )

    assert result.success is True
    assert result.package_result is not None
    assert result.test_result is not None
    assert result.benchmark_result is not None
    assert result.test_exit_code == 0
    assert result.benchmark_exit_code == 0
    assert len(commands) == EXPECTED_COMMAND_COUNT
    assert all(command[1] == project_root.resolve() for command in commands)
    assert all(command[2][0].name == "install" for command in commands)
    assert all(command[3] is True for command in commands)
    assert not result.artifact_root.exists()


def test_trial_stops_before_benchmark_when_tests_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed semantic test prevents the optional benchmark command."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    calls = 0

    def successful_package(options: PackageOptions) -> PackageCommandResult:
        return _package_result(options, success=True)

    def failing_pytest(*args: object, **kwargs: object) -> PytestRunResult:
        nonlocal calls
        assert args
        assert kwargs
        calls += 1
        return PytestRunResult(command=("pytest",), exit_code=1, success=False)

    monkeypatch.setattr(trial_command, "execute_package", successful_package)
    monkeypatch.setattr(trial_command, "run_pytest_command", failing_pytest)

    result = execute_trial(
        TrialOptions(
            root=project_root,
            candidate="app.ranking::score_user",
            test_command="pytest tests",
            benchmark_command="pytest benchmarks",
        )
    )

    assert result.success is False
    assert result.test_exit_code == 1
    assert result.benchmark_exit_code is None
    assert calls == 1


def test_trial_reports_bad_candidate_input(tmp_path: Path) -> None:
    """Bad candidate syntax is reported as a trial error."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)

    result = execute_trial(TrialOptions(root=project_root, candidate="app.ranking"))

    assert result.success is False
    assert "candidate must look like" in str(result.error)


def test_trial_rejects_non_function_explicit_symbols(tmp_path: Path) -> None:
    """Explicit trial candidates retain the legacy top-level function boundary."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)

    result = execute_trial(TrialOptions(root=project_root, candidate="app.ranking::Missing.method"))

    assert result.success is False
    assert "must be top-level functions" in str(result.error)


def test_trial_reports_unsupported_test_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compatibility command strings remain restricted to pytest invocations."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)

    def successful_package(options: PackageOptions) -> PackageCommandResult:
        return _package_result(options, success=True)

    monkeypatch.setattr(
        trial_command,
        "execute_package",
        successful_package,
    )

    result = execute_trial(
        TrialOptions(
            root=project_root,
            candidate="app.ranking::score_user",
            test_command="tox",
        )
    )

    assert result.success is False
    assert "pytest commands only" in str(result.error)


def test_trial_keeps_artifacts_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Debug retention keeps the source-clean wheel and install payload."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)

    def successful_package(options: PackageOptions) -> PackageCommandResult:
        result = _package_result(options, success=True)
        result.install_root.mkdir(parents=True)
        assert result.wheel_path is not None
        result.wheel_path.write_text("wheel", encoding="utf-8")
        return result

    monkeypatch.setattr(trial_command, "execute_package", successful_package)

    result = execute_trial(
        TrialOptions(
            root=project_root,
            candidate="app.ranking::score_user",
            keep_temp=True,
        )
    )

    assert result.success is True
    assert result.artifact_root.exists()
    assert result.overlay_root == result.artifact_root
    assert result.enabled == ()
    assert result.wheel_path is not None
    assert result.wheel_path.exists()
    shutil.rmtree(result.artifact_root)


def _package_result(
    options: PackageOptions,
    *,
    success: bool,
    error: str | None = None,
) -> PackageCommandResult:
    assert options.output_dir is not None
    install_root = options.output_dir / "install"
    wheel_path = options.output_dir / "fixture.whl" if success else None
    return PackageCommandResult(
        success=success,
        project_root=options.root.resolve(),
        output_dir=options.output_dir,
        install_root=install_root,
        wheel_path=wheel_path,
        islands=(),
        build=CompileAttempt(
            success=success,
            command=("typed-region",),
            stdout="",
            stderr=error or "",
            artifact_paths=(),
            duration_seconds=0.0,
        ),
        install_tree_kept=success,
        error=error,
    )
