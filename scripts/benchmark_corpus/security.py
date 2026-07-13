"""Read-only security validation for pinned benchmark checkouts.

The helpers in this module inspect a detached checkout before any dependency
installation or Atoll configuration is applied.  Git owns tracked-file
enumeration; this module does not clone, fetch, checkout, clean, or otherwise
mutate the repository it validates.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol, cast

FindingCode = Literal[
    "attached-head",
    "compile-policy",
    "git-command",
    "gitmodules",
    "invalid-pyproject",
    "lfs-pointer",
    "missing-pyproject",
    "revision-mismatch",
    "submodule",
    "symlink-escape",
    "tracked-file",
    "unsafe-path",
]

_FULL_SHA = re.compile(r"[0-9a-f]{40}\Z")
_LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"
_ROOT_SUBROOT = PurePosixPath(".")


class _Digest(Protocol):
    """Minimal hash interface used by canonical manifest encoding."""

    def update(self, value: bytes, /) -> None:
        """Add bytes to the digest state."""


@dataclass(frozen=True, slots=True)
class SecurityFinding:
    """One reason a checkout is unsafe to use as benchmark input.

    Attributes:
        code: Stable machine-readable finding category.
        message: Maintainer-facing explanation of the rejected condition.
        path: Normalized tracked path involved in the finding, when applicable.
    """

    code: FindingCode
    message: str
    path: PurePosixPath | None = None


@dataclass(frozen=True, slots=True)
class TrackedFileDigest:
    """Stable content identity for one tracked checkout entry.

    Attributes:
        path: Normalized checkout-relative POSIX path.
        sha256: SHA-256 of regular-file bytes or a symlink's target text.
        size: Number of bytes included in ``sha256``.
    """

    path: PurePosixPath
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class TrackedSourceManifest:
    """Canonical content manifest for all tracked files in a checkout.

    Attributes:
        files: Per-file records sorted by normalized path.
        manifest_digest: SHA-256 over an unambiguous length-prefixed encoding of
            every path, content digest, and byte size in ``files`` order.
    """

    files: tuple[TrackedFileDigest, ...]
    manifest_digest: str


@dataclass(frozen=True, slots=True)
class CheckoutValidation:
    """Evidence produced after a checkout passes every security boundary.

    Attributes:
        revision: Full commit SHA observed at ``HEAD``.
        source_manifest: Stable digest of the tracked working-tree contents.
    """

    revision: str
    source_manifest: TrackedSourceManifest


class CheckoutSecurityError(ValueError):
    """Raised with immutable findings when checkout validation fails.

    Attributes:
        findings: Findings in deterministic code, path, and message order.
    """

    findings: tuple[SecurityFinding, ...]

    def __init__(self, findings: tuple[SecurityFinding, ...]) -> None:
        """Build an error from one or more deterministic findings.

        Args:
            findings: Conditions that prevent safe use of the checkout.

        Raises:
            ValueError: If called without any findings.
        """
        if not findings:
            raise ValueError("CheckoutSecurityError requires at least one finding")
        self.findings = tuple(sorted(findings, key=_finding_sort_key))
        super().__init__("; ".join(finding.message for finding in self.findings))


@dataclass(frozen=True, slots=True)
class _TrackedEntry:
    """One stage-zero Git index entry used during security inspection."""

    mode: str
    path: PurePosixPath


def validate_checkout(
    checkout: Path,
    expected_revision: str,
    project_subroot: PurePosixPath = _ROOT_SUBROOT,
    *,
    git_executable: str = "git",
) -> CheckoutValidation:
    """Validate a pinned detached checkout without changing repository state.

    The validator rejects attached or moving ``HEAD`` values, Git submodule
    metadata and gitlinks, unresolved Git LFS pointers, unsafe tracked paths,
    escaping symlinks, and a target project that already owns Atoll's injected
    ``[tool.atoll.compile]`` policy table.  It reads tracked working-tree bytes,
    so callers should validate before executing any code from the checkout.

    Args:
        checkout: Root of the existing Git working tree.
        expected_revision: Required full lowercase 40-character commit SHA.
        project_subroot: Checkout-relative directory containing ``pyproject.toml``.
        git_executable: Git executable name or absolute path injected by the caller.

    Returns:
        CheckoutValidation: Observed revision and stable tracked-source evidence.

    Raises:
        CheckoutSecurityError: If Git inspection fails or any unsafe condition is
            present. Callers should surface the findings and discard the checkout.
    """
    root = checkout.resolve()
    findings: list[SecurityFinding] = []
    if _FULL_SHA.fullmatch(expected_revision) is None:
        findings.append(
            SecurityFinding(
                "revision-mismatch",
                "expected revision must be a full lowercase 40-character SHA",
            )
        )

    revision = _head_revision(root, git_executable, findings)
    _check_detached_head(root, git_executable, findings)
    if revision is not None and revision != expected_revision:
        findings.append(
            SecurityFinding(
                "revision-mismatch",
                f"checkout revision {revision} does not equal expected {expected_revision}",
            )
        )

    entries = _tracked_entries(root, git_executable, findings)
    _inspect_entries(root, entries, findings)
    _inspect_pyproject(root, project_subroot, findings)
    if findings:
        raise CheckoutSecurityError(tuple(findings))

    source_manifest = _manifest_from_entries(root, entries)
    return CheckoutValidation(
        revision=cast(str, revision),
        source_manifest=source_manifest,
    )


def tracked_source_manifest(
    checkout: Path,
    *,
    git_executable: str = "git",
) -> TrackedSourceManifest:
    """Hash every stage-zero tracked file using normalized relative paths.

    Regular files are hashed from the working tree.  Symlinks are hashed from
    their link-target text rather than dereferencing them.  This preserves Git's
    content semantics and prevents a manifest read from following an unsafe
    target.  Unmerged entries and unreadable files are rejected rather than
    omitted, so partial manifests cannot be mistaken for complete evidence.

    Args:
        checkout: Root of the existing Git working tree.
        git_executable: Git executable name or absolute path injected by the caller.

    Returns:
        TrackedSourceManifest: Sorted file hashes and canonical aggregate digest.

    Raises:
        CheckoutSecurityError: If Git enumeration fails, a path is unsafe, an
            index entry is unmerged, or tracked content cannot be read.
    """
    root = checkout.resolve()
    findings: list[SecurityFinding] = []
    entries = _tracked_entries(root, git_executable, findings)
    _inspect_paths_and_gitlinks(entries, findings)
    if findings:
        raise CheckoutSecurityError(tuple(findings))
    try:
        return _manifest_from_entries(root, entries)
    except OSError as error:
        raise CheckoutSecurityError(
            (SecurityFinding("tracked-file", f"cannot read tracked content: {error}"),)
        ) from error


def _run_git(
    checkout: Path,
    git_executable: str,
    arguments: tuple[str, ...],
) -> subprocess.CompletedProcess[bytes]:
    """Run one read-only Git query with an explicit argv and no shell."""
    argv = (git_executable, "-C", os.fspath(checkout), *arguments)
    return subprocess.run(
        argv,
        shell=False,
        check=False,
        capture_output=True,
    )


def _head_revision(
    root: Path,
    git_executable: str,
    findings: list[SecurityFinding],
) -> str | None:
    result = _run_git(root, git_executable, ("rev-parse", "--verify", "HEAD^{commit}"))
    if result.returncode != 0:
        findings.append(SecurityFinding("git-command", _git_failure("resolve HEAD", result)))
        return None
    revision = result.stdout.decode("ascii", errors="replace").strip()
    if _FULL_SHA.fullmatch(revision) is None:
        findings.append(
            SecurityFinding("git-command", f"Git returned invalid HEAD SHA {revision!r}")
        )
        return None
    return revision


def _check_detached_head(
    root: Path,
    git_executable: str,
    findings: list[SecurityFinding],
) -> None:
    result = _run_git(root, git_executable, ("symbolic-ref", "-q", "HEAD"))
    if result.returncode == 0:
        reference = result.stdout.decode("utf-8", errors="replace").strip()
        findings.append(
            SecurityFinding("attached-head", f"checkout HEAD is attached to {reference}")
        )
    elif result.returncode != 1:
        findings.append(SecurityFinding("git-command", _git_failure("inspect HEAD", result)))


def _tracked_entries(
    root: Path,
    git_executable: str,
    findings: list[SecurityFinding],
) -> tuple[_TrackedEntry, ...]:
    result = _run_git(root, git_executable, ("ls-files", "--stage", "-z"))
    if result.returncode != 0:
        findings.append(
            SecurityFinding("git-command", _git_failure("enumerate tracked files", result))
        )
        return ()
    entries: list[_TrackedEntry] = []
    for record in result.stdout.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, _object_id, stage = metadata.decode("ascii").split(" ")
        except ValueError:
            findings.append(SecurityFinding("git-command", "Git returned malformed index data"))
            continue
        path_text = os.fsdecode(raw_path)
        normalized = PurePosixPath(path_text)
        if stage != "0":
            findings.append(
                SecurityFinding(
                    "tracked-file",
                    f"tracked path {path_text!r} has unresolved index stage {stage}",
                    normalized,
                )
            )
            continue
        entries.append(_TrackedEntry(mode=mode, path=normalized))
    return tuple(sorted(entries, key=lambda entry: entry.path.as_posix()))


def _inspect_entries(
    root: Path,
    entries: tuple[_TrackedEntry, ...],
    findings: list[SecurityFinding],
) -> None:
    _inspect_paths_and_gitlinks(entries, findings)
    for entry in entries:
        if not _is_safe_path(entry.path):
            continue
        path = root.joinpath(*entry.path.parts)
        try:
            content = _tracked_content(path, entry.mode)
        except OSError as error:
            findings.append(
                SecurityFinding(
                    "tracked-file",
                    f"cannot read tracked path {entry.path.as_posix()!r}: {error}",
                    entry.path,
                )
            )
            continue
        if content.splitlines()[:1] == [_LFS_HEADER]:
            findings.append(
                SecurityFinding(
                    "lfs-pointer",
                    f"tracked path {entry.path.as_posix()!r} is an unresolved Git LFS pointer",
                    entry.path,
                )
            )
        if entry.mode == "120000":
            _inspect_symlink(root, path, entry.path, findings)


def _inspect_paths_and_gitlinks(
    entries: tuple[_TrackedEntry, ...],
    findings: list[SecurityFinding],
) -> None:
    for entry in entries:
        if not _is_safe_path(entry.path):
            findings.append(
                SecurityFinding(
                    "unsafe-path",
                    f"tracked path {entry.path.as_posix()!r} is not a safe "
                    "normalized relative path",
                    entry.path,
                )
            )
        if entry.path.name == ".gitmodules":
            findings.append(
                SecurityFinding(
                    "gitmodules",
                    f"tracked Git submodule metadata is forbidden: {entry.path.as_posix()!r}",
                    entry.path,
                )
            )
        if entry.mode == "160000":
            findings.append(
                SecurityFinding(
                    "submodule",
                    f"tracked gitlink is forbidden: {entry.path.as_posix()!r}",
                    entry.path,
                )
            )


def _inspect_symlink(
    root: Path,
    path: Path,
    relative_path: PurePosixPath,
    findings: list[SecurityFinding],
) -> None:
    try:
        target = path.resolve(strict=False)
        contained = target.is_relative_to(root)
    except (OSError, RuntimeError) as error:
        findings.append(
            SecurityFinding(
                "symlink-escape",
                f"cannot safely resolve symlink {relative_path.as_posix()!r}: {error}",
                relative_path,
            )
        )
        return
    if not contained:
        findings.append(
            SecurityFinding(
                "symlink-escape",
                f"symlink {relative_path.as_posix()!r} resolves outside the checkout",
                relative_path,
            )
        )


def _inspect_pyproject(
    root: Path,
    project_subroot: PurePosixPath,
    findings: list[SecurityFinding],
) -> None:
    if not _is_safe_project_subroot(project_subroot):
        findings.append(
            SecurityFinding(
                "unsafe-path",
                f"project subroot {project_subroot.as_posix()!r} is not a safe relative path",
                project_subroot,
            )
        )
        return
    pyproject = root.joinpath(*project_subroot.parts, "pyproject.toml")
    relative_pyproject = project_subroot / "pyproject.toml"
    try:
        parsed = cast(dict[str, object], tomllib.loads(pyproject.read_text(encoding="utf-8")))
    except FileNotFoundError:
        findings.append(
            SecurityFinding(
                "missing-pyproject",
                f"target project has no {relative_pyproject.as_posix()}",
                relative_pyproject,
            )
        )
        return
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        findings.append(
            SecurityFinding(
                "invalid-pyproject",
                f"cannot inspect {relative_pyproject.as_posix()}: {error}",
                relative_pyproject,
            )
        )
        return
    raw_tool = parsed.get("tool")
    tool = cast(dict[str, object], raw_tool) if isinstance(raw_tool, dict) else {}
    raw_atoll = tool.get("atoll")
    atoll = cast(dict[str, object], raw_atoll) if isinstance(raw_atoll, dict) else {}
    if "compile" in atoll:
        findings.append(
            SecurityFinding(
                "compile-policy",
                f"target {relative_pyproject.as_posix()} already defines [tool.atoll.compile]",
                relative_pyproject,
            )
        )


def _manifest_from_entries(
    root: Path,
    entries: tuple[_TrackedEntry, ...],
) -> TrackedSourceManifest:
    files: list[TrackedFileDigest] = []
    aggregate = hashlib.sha256()
    for entry in entries:
        if entry.mode == "160000":
            continue
        path_text = entry.path.as_posix()
        content = _tracked_content(root.joinpath(*entry.path.parts), entry.mode)
        digest = hashlib.sha256(content).hexdigest()
        record = TrackedFileDigest(path=entry.path, sha256=digest, size=len(content))
        files.append(record)
        _update_length_prefixed(aggregate, path_text.encode("utf-8", errors="surrogateescape"))
        _update_length_prefixed(aggregate, digest.encode("ascii"))
        _update_length_prefixed(aggregate, str(len(content)).encode("ascii"))
    return TrackedSourceManifest(files=tuple(files), manifest_digest=aggregate.hexdigest())


def _tracked_content(path: Path, mode: str) -> bytes:
    if mode == "120000":
        return os.fsencode(path.readlink())
    return path.read_bytes()


def _update_length_prefixed(digest: _Digest, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, byteorder="big"))
    digest.update(value)


def _is_safe_path(path: PurePosixPath) -> bool:
    text = path.as_posix()
    return (
        bool(text)
        and text != "."
        and not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in text
        and PurePosixPath(text).as_posix() == text
    )


def _is_safe_project_subroot(path: PurePosixPath) -> bool:
    return path == _ROOT_SUBROOT or _is_safe_path(path)


def _git_failure(action: str, result: subprocess.CompletedProcess[bytes]) -> str:
    detail = result.stderr.decode("utf-8", errors="replace").strip()
    suffix = f": {detail}" if detail else ""
    return f"cannot {action} with Git (exit {result.returncode}){suffix}"


def _finding_sort_key(finding: SecurityFinding) -> tuple[str, str, str]:
    path = "" if finding.path is None else finding.path.as_posix()
    return (finding.code, path, finding.message)
