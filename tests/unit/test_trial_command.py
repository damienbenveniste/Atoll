"""Tests for Atoll trial-mode branch behavior."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from atoll.commands import trial as trial_command
from atoll.commands.trial import TrialOptions, execute_trial
from atoll.models import CompileAttempt, VerifyResult

FIXTURE_ROOT = Path("tests/fixtures/simple_project")


def test_trial_reports_no_candidates(tmp_path: Path) -> None:
    """Trial fails clearly when scan finds nothing to try."""
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "__init__.py").write_text("", encoding="utf-8")

    result = execute_trial(TrialOptions(root=tmp_path, top=1))

    assert result.success is False
    assert result.error == "no trial candidates selected"


def test_trial_reports_build_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compile failures are returned from trial mode."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)

    def failing_build_sidecars(
        paths: tuple[Path, ...],
        *,
        project_root: Path,
        build_dir: Path,
        source_roots: tuple[Path, ...] = (),
    ) -> CompileAttempt:
        assert paths
        assert project_root
        assert build_dir
        assert source_roots
        return CompileAttempt(
            success=False,
            command=("mypyc",),
            stdout="",
            stderr="MYPYC_TYPE_ERROR: fixture",
            artifact_paths=(),
            duration_seconds=0.0,
        )

    monkeypatch.setattr(trial_command, "build_sidecars", failing_build_sidecars)

    result = execute_trial(
        TrialOptions(root=project_root, candidate="app.ranking::score_user,rank_candidates")
    )

    assert result.success is False
    assert result.error == "MYPYC_TYPE_ERROR: fixture"


def test_trial_reports_verify_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verification failures stop trial before tests run."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)

    def fake_build_sidecars(
        paths: tuple[Path, ...],
        *,
        project_root: Path,
        build_dir: Path,
        source_roots: tuple[Path, ...] = (),
    ) -> CompileAttempt:
        assert paths
        assert project_root
        assert build_dir
        assert source_roots
        return CompileAttempt(
            success=True,
            command=("mypyc",),
            stdout="",
            stderr="",
            artifact_paths=(),
            duration_seconds=0.0,
        )

    def fake_verify_islands(*args: object, **kwargs: object) -> tuple[VerifyResult, ...]:
        assert args
        assert kwargs
        return (
            VerifyResult(
                source_module="app.ranking",
                sidecar_module="app._ranking_atoll",
                active=False,
                compiled=False,
                origin=None,
                symbols=(("score_user", False),),
                error="sidecar missing",
            ),
        )

    monkeypatch.setattr(trial_command, "build_sidecars", fake_build_sidecars)
    monkeypatch.setattr(trial_command, "verify_islands", fake_verify_islands)

    result = execute_trial(
        TrialOptions(root=project_root, candidate="app.ranking::score_user,rank_candidates")
    )

    assert result.success is False
    assert result.error == "sidecar missing"


def test_trial_reports_bad_candidate_input(tmp_path: Path) -> None:
    """Bad candidate syntax is reported as a trial error."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)

    result = execute_trial(TrialOptions(root=project_root, candidate="app.ranking"))

    assert result.success is False
    assert "candidate must look like" in str(result.error)


def test_trial_reports_unsupported_test_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trial mode currently accepts pytest test commands only."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)

    def fake_build_sidecars(
        paths: tuple[Path, ...],
        *,
        project_root: Path,
        build_dir: Path,
        source_roots: tuple[Path, ...] = (),
    ) -> CompileAttempt:
        assert paths
        assert project_root
        assert build_dir
        assert source_roots
        return CompileAttempt(
            success=True,
            command=("mypyc",),
            stdout="",
            stderr="",
            artifact_paths=(),
            duration_seconds=0.0,
        )

    def fake_verify_islands(*args: object, **kwargs: object) -> tuple[VerifyResult, ...]:
        assert args
        assert kwargs
        return ()

    monkeypatch.setattr(trial_command, "build_sidecars", fake_build_sidecars)
    monkeypatch.setattr(trial_command, "verify_islands", fake_verify_islands)

    result = execute_trial(
        TrialOptions(
            root=project_root,
            candidate="app.ranking::score_user,rank_candidates",
            test_command="tox",
        )
    )

    assert result.success is False
    assert "pytest commands only" in str(result.error)


def test_trial_keeps_temp_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trial cleanup honors keep-temp mode."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)

    def fake_build_sidecars(
        paths: tuple[Path, ...],
        *,
        project_root: Path,
        build_dir: Path,
        source_roots: tuple[Path, ...] = (),
    ) -> CompileAttempt:
        assert paths
        assert project_root
        assert build_dir
        assert source_roots
        return CompileAttempt(
            success=True,
            command=("mypyc",),
            stdout="",
            stderr="",
            artifact_paths=(),
            duration_seconds=0.0,
        )

    def fake_verify_islands(*args: object, **kwargs: object) -> tuple[VerifyResult, ...]:
        assert args
        assert kwargs
        return ()

    monkeypatch.setattr(trial_command, "build_sidecars", fake_build_sidecars)
    monkeypatch.setattr(trial_command, "verify_islands", fake_verify_islands)

    result = execute_trial(
        TrialOptions(
            root=project_root,
            candidate="app.ranking::score_user,rank_candidates",
            keep_temp=True,
        )
    )

    assert result.success is True
    assert result.overlay_root.exists()
    shutil.rmtree(result.overlay_root)
