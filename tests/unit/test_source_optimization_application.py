"""Tests for transactional source-optimization patch application."""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

import pytest

import atoll.source_optimization.application as source_application
from atoll.source_optimization.application import (
    apply_source_patch_transactionally,
    validate_source_application_root,
)
from atoll.source_optimization.models import SourceEdit
from atoll.source_optimization.transforms import GeneratedSourcePatch, TransformedSourceFile

_run_subprocess = subprocess.run


def _sha256(source: str) -> str:
    """Return the SHA-256 digest used by source patch records.

    Args:
        source: Source text to hash as UTF-8.

    Returns:
        str: Hex SHA-256 digest.
    """
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run Git with argv-only subprocess invocation for temp repositories.

    Args:
        root: Git working tree root.
        *args: Git arguments after `git -C root`.

    Returns:
        subprocess.CompletedProcess[str]: Captured Git result.
    """
    return _run_subprocess(
        ("git", "-C", str(root), *args),
        input=None,
        text=True,
        capture_output=True,
        shell=False,
        check=False,
    )


def _init_repo(root: Path) -> None:
    """Create a temporary Git repository for application tests.

    Args:
        root: Directory to initialize as a Git repository.
    """
    root.mkdir(parents=True, exist_ok=True)
    assert _git(root, "init").returncode == 0


def _write(root: Path, relative: str, source: str) -> PurePosixPath:
    """Write a source fixture under the temporary project root.

    Args:
        root: Project root for the fixture.
        relative: POSIX relative file path.
        source: Source text to write.

    Returns:
        PurePosixPath: Relative POSIX path used by patch metadata.
    """
    path = PurePosixPath(relative)
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")
    return path


def _patch_text(relative: PurePosixPath, before_line: str, after_line: str) -> str:
    """Build a small Git-compatible patch for a one-line function body change.

    Args:
        relative: POSIX source path changed by the patch.
        before_line: Existing source line to remove.
        after_line: Replacement source line to add.

    Returns:
        str: Unified Git patch text.
    """
    path = relative.as_posix()
    return (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        "@@ -1,2 +1,2 @@\n"
        " def run(value: int) -> int:\n"
        f"-{before_line}"
        f"+{after_line}"
    )


def _generated_patch(
    *,
    path: PurePosixPath,
    before_source: str,
    after_source: str,
    patch_text: str,
) -> GeneratedSourcePatch:
    """Build a generated source patch fixture.

    Args:
        path: POSIX source path changed by the patch.
        before_source: Source content before the patch.
        after_source: Source content after the patch.
        patch_text: Persisted Git patch text.

    Returns:
        GeneratedSourcePatch: Patch metadata matching production contracts.
    """
    return GeneratedSourcePatch(
        patch_text=patch_text,
        source_edits=(
            SourceEdit(
                path=path,
                before_hash=_sha256(before_source),
                after_hash=_sha256(after_source),
                summary="rewrite run",
                transformation_id="step:run",
                start_line=2,
                end_line=2,
            ),
        ),
        files=(
            TransformedSourceFile(
                path=path,
                before_source=before_source,
                after_source=after_source,
            ),
        ),
    )


def _write_patch_file(root: Path, patch: GeneratedSourcePatch) -> Path:
    """Persist a generated patch beside the temporary repository.

    Args:
        root: Temporary project root.
        patch: Patch whose text should be written.

    Returns:
        Path: Path to the persisted patch file.
    """
    patch_path = root.parent / "candidate.patch"
    patch_path.write_text(patch.patch_text, encoding="utf-8")
    return patch_path


def _standard_patch(root: Path) -> tuple[PurePosixPath, str, str, GeneratedSourcePatch, Path]:
    """Create the common one-file source patch fixture.

    Args:
        root: Git project root to populate.

    Returns:
        tuple[PurePosixPath, str, str, GeneratedSourcePatch, Path]: Relative path,
        before source, after source, patch object, and persisted patch path.
    """
    before_source = "def run(value: int) -> int:\n    return value + 1\n"
    after_source = "def run(value: int) -> int:\n    return value * 3\n"
    relative = _write(root, "pkg/mod.py", before_source)
    patch = _generated_patch(
        path=relative,
        before_source=before_source,
        after_source=after_source,
        patch_text=_patch_text(relative, "    return value + 1\n", "    return value * 3\n"),
    )
    return relative, before_source, after_source, patch, _write_patch_file(root, patch)


def _accept_validation() -> tuple[bool, tuple[str, ...]]:
    """Return a successful validation result.

    Returns:
        tuple[bool, tuple[str, ...]]: Accepted validation marker.
    """
    return True, ("validated project",)


def test_validate_source_application_root_rejects_non_git_root(tmp_path: Path) -> None:
    """A plain directory is not eligible for transactional patch application."""
    result = apply_source_patch_transactionally(
        tmp_path,
        tmp_path / "missing.patch",
        GeneratedSourcePatch(patch_text="", source_edits=(), files=()),
        _accept_validation,
    )

    assert result.status == "failed"
    assert result.diagnostics == (
        f"source application root is not a Git work tree: {tmp_path.resolve()}",
    )


def test_validate_source_application_root_rejects_missing_directory(tmp_path: Path) -> None:
    """A missing path is rejected before Git root discovery runs."""
    missing_root = tmp_path / "missing"

    error = validate_source_application_root(missing_root)

    assert error == f"source application root is not a directory: {missing_root.resolve()}"


def test_validate_source_application_root_rejects_nested_git_directory(tmp_path: Path) -> None:
    """A Git work tree subdirectory is not accepted as the application root."""
    root = tmp_path / "project"
    _init_repo(root)
    nested = root / "pkg"
    nested.mkdir()

    error = validate_source_application_root(nested)

    assert error == f"source application root must be Git top-level: {nested.resolve()}"


def test_apply_source_patch_transactionally_rejects_missing_persisted_patch(
    tmp_path: Path,
) -> None:
    """Patch metadata is rejected when the reviewed patch file is absent."""
    root = tmp_path / "project"
    _init_repo(root)
    _, _, _, patch, patch_path = _standard_patch(root)
    patch_path.unlink()

    result = apply_source_patch_transactionally(root, patch_path, patch, _accept_validation)

    assert result.status == "failed"
    assert result.diagnostics == (f"persisted source patch does not exist: {patch_path.resolve()}",)


def test_apply_source_patch_transactionally_rejects_changed_persisted_patch(
    tmp_path: Path,
) -> None:
    """The persisted patch bytes must match the accepted generated patch."""
    root = tmp_path / "project"
    _init_repo(root)
    _, _, _, patch, patch_path = _standard_patch(root)
    patch_path.write_text(f"{patch.patch_text}\n", encoding="utf-8")

    result = apply_source_patch_transactionally(root, patch_path, patch, _accept_validation)

    assert result.status == "failed"
    assert result.diagnostics == (
        f"persisted source patch differs from generated patch: {patch_path.resolve()}",
    )


def test_apply_source_patch_transactionally_rejects_stale_source(tmp_path: Path) -> None:
    """Current source digests are checked before Git conflict checks run."""
    root = tmp_path / "project"
    _init_repo(root)
    _, _, _, patch, patch_path = _standard_patch(root)
    (root / "pkg/mod.py").write_text(
        "def run(value: int) -> int:\n    return value + 2\n",
        encoding="utf-8",
    )

    result = apply_source_patch_transactionally(root, patch_path, patch, _accept_validation)

    assert result.status == "stale-source"
    assert result.diagnostics[0].startswith("stale source for pkg/mod.py: expected ")


def test_apply_source_patch_transactionally_rejects_unexpected_existing_source(
    tmp_path: Path,
) -> None:
    """Creation patches are stale when the target file already exists."""
    root = tmp_path / "project"
    _init_repo(root)
    relative = _write(root, "pkg/new.py", "value = 1\n")
    patch = GeneratedSourcePatch(
        patch_text="",
        source_edits=(
            SourceEdit(
                path=relative,
                before_hash=None,
                after_hash=_sha256("value = 2\n"),
                summary="create source",
            ),
        ),
        files=(),
    )
    patch_path = _write_patch_file(root, patch)

    result = apply_source_patch_transactionally(root, patch_path, patch, _accept_validation)

    assert result.status == "stale-source"
    assert result.diagnostics == ("stale source for pkg/new.py: expected missing file",)


def test_apply_source_patch_transactionally_accepts_missing_creation_target_preflight(
    tmp_path: Path,
) -> None:
    """A creation patch reaches Git when the target file is still absent."""
    root = tmp_path / "project"
    _init_repo(root)
    patch = GeneratedSourcePatch(
        patch_text="",
        source_edits=(
            SourceEdit(
                path=PurePosixPath("pkg/new.py"),
                before_hash=None,
                after_hash=_sha256("value = 2\n"),
                summary="create source",
            ),
        ),
        files=(),
    )
    patch_path = _write_patch_file(root, patch)

    result = apply_source_patch_transactionally(root, patch_path, patch, _accept_validation)

    assert result.status == "conflicted"
    assert result.diagnostics[0] == "git apply --check exited 128"


def test_apply_source_patch_transactionally_rejects_missing_expected_source(
    tmp_path: Path,
) -> None:
    """Update patches are stale when the expected source file is missing."""
    root = tmp_path / "project"
    _init_repo(root)
    before_source = "value = 1\n"
    patch = _generated_patch(
        path=PurePosixPath("pkg/missing.py"),
        before_source=before_source,
        after_source="value = 2\n",
        patch_text="",
    )
    patch_path = _write_patch_file(root, patch)

    result = apply_source_patch_transactionally(root, patch_path, patch, _accept_validation)

    assert result.status == "stale-source"
    assert result.diagnostics == (
        f"stale source for pkg/missing.py: expected {_sha256(before_source)}, found missing file",
    )


def test_apply_source_patch_transactionally_rejects_source_path_with_parent_escape(
    tmp_path: Path,
) -> None:
    """Patch source paths cannot escape the project root with parent segments."""
    root = tmp_path / "project"
    _init_repo(root)
    patch = _generated_patch(
        path=PurePosixPath("../outside.py"),
        before_source="value = 1\n",
        after_source="value = 2\n",
        patch_text="",
    )
    patch_path = _write_patch_file(root, patch)

    result = apply_source_patch_transactionally(root, patch_path, patch, _accept_validation)

    assert result.status == "failed"
    assert result.diagnostics == ("source patch path escapes project root: ../outside.py",)


def test_apply_source_patch_transactionally_rejects_source_path_symlink_escape(
    tmp_path: Path,
) -> None:
    """Resolved source paths must remain below the repository root."""
    root = tmp_path / "project"
    outside = tmp_path / "outside"
    _init_repo(root)
    outside.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)
    patch = _generated_patch(
        path=PurePosixPath("link/outside.py"),
        before_source="value = 1\n",
        after_source="value = 2\n",
        patch_text="",
    )
    patch_path = _write_patch_file(root, patch)

    result = apply_source_patch_transactionally(root, patch_path, patch, _accept_validation)

    assert result.status == "failed"
    assert result.diagnostics == ("source patch path escapes project root: link/outside.py",)


def test_apply_source_patch_transactionally_applies_valid_patch(tmp_path: Path) -> None:
    """A clean patch and accepting callback leave the transformed source in place."""
    root = tmp_path / "project"
    _init_repo(root)
    _, _, after_source, patch, patch_path = _standard_patch(root)

    result = apply_source_patch_transactionally(
        root,
        patch_path,
        patch,
        _accept_validation,
    )

    assert result.status == "applied"
    assert result.diagnostics == ("validated project",)
    assert (root / "pkg/mod.py").read_text(encoding="utf-8") == after_source


def test_apply_source_patch_transactionally_reports_apply_conflict(tmp_path: Path) -> None:
    """A patch whose context does not match clean source is reported as conflicted."""
    root = tmp_path / "project"
    _init_repo(root)
    relative = _write(root, "pkg/mod.py", "def run(value: int) -> int:\n    return value + 1\n")
    before_source = "def run(value: int) -> int:\n    return value + 1\n"
    after_source = "def run(value: int) -> int:\n    return value * 3\n"
    patch = _generated_patch(
        path=relative,
        before_source=before_source,
        after_source=after_source,
        patch_text=_patch_text(relative, "    return missing_name\n", "    return value * 3\n"),
    )
    patch_path = _write_patch_file(root, patch)

    result = apply_source_patch_transactionally(root, patch_path, patch, _accept_validation)

    assert result.status == "conflicted"
    assert result.diagnostics[0] == "git apply --check exited 1"
    assert (root / "pkg/mod.py").read_text(encoding="utf-8") == before_source


def test_apply_source_patch_transactionally_reports_apply_command_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed second `git apply` command is reported with stdout and stderr."""
    patch_path = tmp_path / "candidate.patch"
    patch = GeneratedSourcePatch(patch_text="", source_edits=(), files=())
    patch_path.write_text("", encoding="utf-8")
    commands: list[tuple[str, ...]] = []

    def fake_run(command: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        """Return deterministic Git results for the apply failure sequence.

        Args:
            command: Git argv supplied by the application helper.
            **kwargs: Subprocess options supplied by the Git runner.

        Returns:
            subprocess.CompletedProcess[str]: Fake Git command outcome.
        """
        del kwargs
        argv = tuple(command)
        commands.append(argv)
        if argv[-2:] == ("rev-parse", "--show-toplevel"):
            return subprocess.CompletedProcess(argv, 0, stdout=f"{tmp_path}\n", stderr="")
        if argv[-3:-1] == ("apply", "--check"):
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        if argv[-2] == "apply":
            return subprocess.CompletedProcess(
                argv,
                2,
                stdout="partial stdout\n",
                stderr="",
            )
        raise AssertionError(f"unexpected command: {argv}")

    monkeypatch.setattr(source_application, "_run_subprocess", fake_run)

    result = apply_source_patch_transactionally(tmp_path, patch_path, patch, _accept_validation)

    assert result.status == "failed"
    assert result.diagnostics == ("git apply exited 2", "partial stdout")
    assert [command[3] for command in commands] == ["rev-parse", "apply", "apply"]


def test_apply_source_patch_transactionally_rolls_back_callback_failure(
    tmp_path: Path,
) -> None:
    """A rejecting validation callback reverses the exact applied patch."""
    root = tmp_path / "project"
    _init_repo(root)
    _, before_source, _, patch, patch_path = _standard_patch(root)

    result = apply_source_patch_transactionally(
        root,
        patch_path,
        patch,
        lambda: (False, ("semantic validation failed in project",)),
    )

    assert result.status == "rolled-back"
    assert result.diagnostics == (
        "semantic validation failed in project",
        "validation failed; patch rolled back",
    )
    assert (root / "pkg/mod.py").read_text(encoding="utf-8") == before_source


def test_apply_source_patch_transactionally_rolls_back_callback_exception(
    tmp_path: Path,
) -> None:
    """A raising validation callback still leaves the checkout restored."""
    root = tmp_path / "project"
    _init_repo(root)
    _, before_source, _, patch, patch_path = _standard_patch(root)

    def raise_validation_error() -> tuple[bool, tuple[str, ...]]:
        """Raise a deterministic validation error after the patch is applied.

        Returns:
            tuple[bool, tuple[str, ...]]: Unreachable validation result.

        Raises:
            RuntimeError: Always raised to exercise rollback behavior.
        """
        raise RuntimeError("validation crashed in project")

    result = apply_source_patch_transactionally(
        root,
        patch_path,
        patch,
        raise_validation_error,
    )

    assert result.status == "rolled-back"
    assert result.diagnostics == (
        "validation callback raised RuntimeError: validation crashed in project",
        "validation failed; patch rolled back",
    )
    assert (root / "pkg/mod.py").read_text(encoding="utf-8") == before_source


def test_apply_source_patch_transactionally_reports_reverse_check_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rollback reports a failed reverse check without hiding validation output."""
    patch_path = tmp_path / "candidate.patch"
    patch = GeneratedSourcePatch(patch_text="", source_edits=(), files=())
    patch_path.write_text("", encoding="utf-8")

    def fake_run(command: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        """Return deterministic Git results through a reverse-check failure.

        Args:
            command: Git argv supplied by the application helper.
            **kwargs: Subprocess options supplied by the Git runner.

        Returns:
            subprocess.CompletedProcess[str]: Fake Git command outcome.
        """
        del kwargs
        argv = tuple(command)
        if argv[-2:] == ("rev-parse", "--show-toplevel"):
            return subprocess.CompletedProcess(argv, 0, stdout=f"{tmp_path}\n", stderr="")
        if argv[-3:-1] == ("apply", "--check") or argv[-2] == "apply":
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        if argv[-4:-1] == ("apply", "--reverse", "--check"):
            return subprocess.CompletedProcess(
                argv,
                1,
                stdout="reverse stdout\n",
                stderr="reverse stderr\n",
            )
        raise AssertionError(f"unexpected command: {argv}")

    monkeypatch.setattr(source_application, "_run_subprocess", fake_run)

    result = apply_source_patch_transactionally(
        tmp_path,
        patch_path,
        patch,
        lambda: (False, ("validation rejected",)),
    )

    assert result.status == "failed"
    assert result.diagnostics == (
        "validation rejected",
        "git apply --reverse --check exited 1",
        "reverse stdout",
        "reverse stderr",
    )


def test_apply_source_patch_transactionally_reports_reverse_apply_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rollback reports a failed reverse apply after a successful reverse check."""
    patch_path = tmp_path / "candidate.patch"
    patch = GeneratedSourcePatch(patch_text="", source_edits=(), files=())
    patch_path.write_text("", encoding="utf-8")

    def fake_run(command: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        """Return deterministic Git results through a reverse-apply failure.

        Args:
            command: Git argv supplied by the application helper.
            **kwargs: Subprocess options supplied by the Git runner.

        Returns:
            subprocess.CompletedProcess[str]: Fake Git command outcome.
        """
        del kwargs
        argv = tuple(command)
        if argv[-2:] == ("rev-parse", "--show-toplevel"):
            return subprocess.CompletedProcess(argv, 0, stdout=f"{tmp_path}\n", stderr="")
        if argv[-3:-1] == ("apply", "--check") or argv[-2] == "apply":
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        if argv[-4:-1] == ("apply", "--reverse", "--check"):
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        if argv[-3:-1] == ("apply", "--reverse"):
            return subprocess.CompletedProcess(argv, 2, stdout="", stderr="reverse failed\n")
        raise AssertionError(f"unexpected command: {argv}")

    monkeypatch.setattr(source_application, "_run_subprocess", fake_run)

    result = apply_source_patch_transactionally(
        tmp_path,
        patch_path,
        patch,
        lambda: (False, ("validation rejected",)),
    )

    assert result.status == "failed"
    assert result.diagnostics == (
        "validation rejected",
        "git apply --reverse exited 2",
        "reverse failed",
    )


def test_validate_source_application_root_uses_safe_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Git subprocesses receive argv tuples with `shell=False`."""
    commands: list[tuple[str, ...]] = []

    def fake_run(command: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        """Record subprocess invocation shape and return a Git top-level path.

        Args:
            command: Argv sequence supplied to subprocess.
            **kwargs: Keyword-only subprocess options supplied by the Git runner.

        Returns:
            subprocess.CompletedProcess[str]: Successful fake Git result.
        """
        assert kwargs == {
            "input": None,
            "text": True,
            "capture_output": True,
            "shell": False,
            "check": False,
        }
        assert not isinstance(command, str)
        commands.append(tuple(command))
        return subprocess.CompletedProcess(command, 0, stdout=f"{tmp_path}\n", stderr="")

    monkeypatch.setattr(source_application, "_run_subprocess", fake_run)

    assert validate_source_application_root(tmp_path) is None
    assert commands == [("git", "-C", str(tmp_path.resolve()), "rev-parse", "--show-toplevel")]
