"""Transactional Git application for accepted source-optimization patches.

This module owns the final mutation step for generated source patches. It
requires callers to pass the exact Git repository root, verifies the recorded
source hashes before Git sees the patch, applies the persisted patch file with
`git apply`, and rolls the same patch back if post-apply validation does not
accept the checkout state.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from atoll.source_optimization.models import SourceOptimizationApplicationStatus
from atoll.source_optimization.transforms import GeneratedSourcePatch

SourcePatchValidationCallback = Callable[[], tuple[bool, tuple[str, ...]]]
_run_subprocess = subprocess.run


@dataclass(frozen=True, slots=True)
class SourcePatchApplicationResult:
    """Outcome of one transactional source patch application attempt.

    Attributes:
        status: Terminal application status reported to source-optimization trials.
        diagnostics: Deterministic human-readable diagnostics suitable for reports
            and logs. Messages describe the gate that stopped the transaction.
    """

    status: SourceOptimizationApplicationStatus
    diagnostics: tuple[str, ...]


def validate_source_application_root(project_root: Path) -> str | None:
    """Validate that a path is exactly the top level of a Git work tree.

    Args:
        project_root: Candidate repository root supplied by the caller.

    Returns:
        str | None: Deterministic error text when the path is not a directory,
        not inside Git, or resolves differently from Git's top-level path;
        otherwise `None`.
    """
    root = project_root.resolve()
    if not root.is_dir():
        return f"source application root is not a directory: {root}"

    completed = _run_git(root, ("rev-parse", "--show-toplevel"), input_text=None)
    if completed.returncode != 0:
        return f"source application root is not a Git work tree: {root}"

    git_root = Path(completed.stdout.strip()).resolve()
    if git_root != root:
        return f"source application root must be Git top-level: {root}"
    return None


def apply_source_patch_transactionally(
    project_root: Path,
    patch_path: Path,
    patch: GeneratedSourcePatch,
    validate_callback: SourcePatchValidationCallback,
) -> SourcePatchApplicationResult:
    """Apply a generated source patch and roll it back on validation failure.

    The transaction checks the repository root, confirms every stored source
    file still has its recorded pre-apply SHA-256 digest, verifies the persisted
    patch with `git apply --check`, applies that same persisted patch path, and
    runs the callback against the mutated root. A callback rejection or exception
    triggers a reverse check and reverse apply using the same patch file.

    Args:
        project_root: Exact Git top-level directory to mutate.
        patch_path: Path to the persisted reviewed patch file.
        patch: Accepted generated patch metadata and diff text.
        validate_callback: Post-apply validation hook. It closes over any command
            or path context it needs and returns success plus diagnostics, or
            raises when validation cannot complete.

    Returns:
        SourcePatchApplicationResult: Final transaction status and diagnostics.
    """
    root, persisted_patch, preflight = _prepare_transaction(project_root, patch_path, patch)
    if preflight is not None:
        return preflight

    apply_result = _apply_persisted_patch(root, persisted_patch)
    if apply_result is not None:
        return apply_result

    try:
        validation_ok, validation_diagnostics = validate_callback()
    except Exception as exc:
        rollback = _reverse_patch(root, persisted_patch)
        return SourcePatchApplicationResult(
            rollback.status,
            (
                f"validation callback raised {type(exc).__name__}: {exc}",
                *rollback.diagnostics,
            ),
        )

    if not validation_ok:
        rollback = _reverse_patch(root, persisted_patch)
        return SourcePatchApplicationResult(
            rollback.status,
            (*validation_diagnostics, *rollback.diagnostics),
        )

    return SourcePatchApplicationResult("applied", validation_diagnostics)


def _prepare_transaction(
    project_root: Path,
    patch_path: Path,
    patch: GeneratedSourcePatch,
) -> tuple[Path, Path, SourcePatchApplicationResult | None]:
    root = project_root.resolve()
    root_error = validate_source_application_root(root)
    if root_error is not None:
        return root, patch_path.resolve(), SourcePatchApplicationResult("failed", (root_error,))

    persisted_error = _validate_persisted_patch(patch_path, patch)
    if persisted_error is not None:
        return (
            root,
            patch_path.resolve(),
            SourcePatchApplicationResult(
                "failed",
                (persisted_error,),
            ),
        )

    try:
        stale_diagnostics = _stale_source_diagnostics(root, patch)
    except ValueError as exc:
        return root, patch_path.resolve(), SourcePatchApplicationResult("failed", (str(exc),))
    if stale_diagnostics:
        return (
            root,
            patch_path.resolve(),
            SourcePatchApplicationResult(
                "stale-source",
                stale_diagnostics,
            ),
        )

    return root, patch_path.resolve(), None


def _apply_persisted_patch(
    root: Path,
    persisted_patch: Path,
) -> SourcePatchApplicationResult | None:
    check = _run_git(root, ("apply", "--check", str(persisted_patch)), input_text=None)
    if check.returncode != 0:
        return SourcePatchApplicationResult(
            "conflicted",
            _git_diagnostics("git apply --check", check),
        )

    applied = _run_git(root, ("apply", str(persisted_patch)), input_text=None)
    if applied.returncode != 0:
        return SourcePatchApplicationResult("failed", _git_diagnostics("git apply", applied))
    return None


def _validate_persisted_patch(patch_path: Path, patch: GeneratedSourcePatch) -> str | None:
    path = patch_path.resolve()
    if not path.is_file():
        return f"persisted source patch does not exist: {path}"
    persisted_text = path.read_text(encoding="utf-8")
    if persisted_text != patch.patch_text:
        return f"persisted source patch differs from generated patch: {path}"
    return None


def _stale_source_diagnostics(
    root: Path,
    patch: GeneratedSourcePatch,
) -> tuple[str, ...]:
    diagnostics: list[str] = []
    for relative_path, expected_hash in _expected_source_hashes(patch):
        source_path = _safe_project_path(root, relative_path)
        if expected_hash is None:
            if source_path.exists():
                diagnostics.append(
                    f"stale source for {relative_path.as_posix()}: expected missing file"
                )
            continue
        if not source_path.is_file():
            diagnostics.append(
                f"stale source for {relative_path.as_posix()}: expected "
                f"{expected_hash}, found missing file"
            )
            continue
        current_hash = _sha256(source_path.read_bytes())
        if current_hash != expected_hash:
            diagnostics.append(
                f"stale source for {relative_path.as_posix()}: expected "
                f"{expected_hash}, found {current_hash}"
            )
    return tuple(diagnostics)


def _expected_source_hashes(
    patch: GeneratedSourcePatch,
) -> tuple[tuple[PurePosixPath, str | None], ...]:
    hashes: dict[PurePosixPath, str | None] = {
        file.path: _sha256(file.before_source.encode("utf-8")) for file in patch.files
    }
    for edit in patch.source_edits:
        hashes.setdefault(edit.path, edit.before_hash)
    return tuple(sorted(hashes.items(), key=lambda item: item[0].as_posix()))


def _reverse_patch(root: Path, patch_path: Path) -> SourcePatchApplicationResult:
    check = _run_git(root, ("apply", "--reverse", "--check", str(patch_path)), input_text=None)
    if check.returncode != 0:
        return SourcePatchApplicationResult(
            "failed",
            _git_diagnostics("git apply --reverse --check", check),
        )

    reversed_patch = _run_git(root, ("apply", "--reverse", str(patch_path)), input_text=None)
    if reversed_patch.returncode != 0:
        return SourcePatchApplicationResult(
            "failed",
            _git_diagnostics("git apply --reverse", reversed_patch),
        )
    return SourcePatchApplicationResult("rolled-back", ("validation failed; patch rolled back",))


def _git_diagnostics(
    operation: str,
    completed: subprocess.CompletedProcess[str],
) -> tuple[str, ...]:
    diagnostics = [f"{operation} exited {completed.returncode}"]
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if stdout:
        diagnostics.append(stdout)
    if stderr:
        diagnostics.append(stderr)
    return tuple(diagnostics)


def _run_git(
    cwd: Path,
    args: Sequence[str],
    *,
    input_text: str | None,
) -> subprocess.CompletedProcess[str]:
    command = ("git", "-C", str(cwd), *args)
    return _run_subprocess(
        command,
        input=input_text,
        text=True,
        capture_output=True,
        shell=False,
        check=False,
    )


def _safe_project_path(root: Path, relative_path: PurePosixPath) -> Path:
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"source patch path escapes project root: {relative_path}")
    resolved = (root / Path(relative_path.as_posix())).resolve()
    if not _is_relative_to(resolved, root):
        raise ValueError(f"source patch path escapes project root: {relative_path}")
    return resolved


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
