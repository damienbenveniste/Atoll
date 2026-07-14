"""Security validation and materialization for pinned benchmark sources.

Git helpers inspect detached checkouts without mutating repository state and
use the index as the tracked-file boundary. Archive helpers authenticate a
reviewed sdist before parsing it, then populate only a fresh caller-owned
destination with bounded regular files. This module does not execute project
code, clone repositories, install dependencies, or inject Atoll policy.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tarfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Literal, Never, Protocol, cast

from scripts.benchmark_corpus.models import SdistSource

FindingCode = Literal[
    "attached-head",
    "compile-policy",
    "git-command",
    "gitmodules",
    "invalid-pyproject",
    "lfs-pointer",
    "revision-mismatch",
    "submodule",
    "symlink-escape",
    "tracked-file",
    "unsafe-path",
    "archive-digest",
    "archive-member",
    "archive-size",
    "tree-digest",
]

_FULL_SHA = re.compile(r"[0-9a-f]{40}\Z")
_LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"
_ROOT_SUBROOT = PurePosixPath(".")
MAX_ARCHIVE_BYTES = 100 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 100_000
MAX_EXTRACTED_BYTES = 1024 * 1024 * 1024
MAX_EXTRACTED_FILE_BYTES = 100 * 1024 * 1024


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


@dataclass(frozen=True, slots=True)
class SdistValidation:
    """Evidence produced by one verified, freshly extracted sdist.

    Attributes:
        archive_sha256: Digest computed over the archive before tar parsing.
        source_manifest: Normalized regular-file tree identity after root stripping.
    """

    archive_sha256: str
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


def extract_sdist_archive(
    archive_path: Path,
    destination: Path,
    source: SdistSource,
    *,
    project_subroot: PurePosixPath = _ROOT_SUBROOT,
) -> SdistValidation:
    """Verify and safely extract one content-addressed tar sdist.

    Archive bytes are sized and hashed before ``tarfile`` sees them. The tar
    must have exactly one normalized top-level directory and may contain only
    directories and bounded regular files. Links, devices, FIFOs, traversal,
    duplicate normalized paths, Git submodule metadata, and path collisions are
    rejected before the destination is created. A root ``.git`` pointer file is
    deliberately omitted from both extraction and tree identity.

    Args:
        archive_path: Existing regular archive file.
        destination: Fresh path that will become the stripped source root.
        source: Locked archive byte and normalized tree identity.
        project_subroot: Extracted project directory containing ``pyproject.toml``.

    Returns:
        SdistValidation: Verified archive and extracted source-tree evidence.

    Raises:
        CheckoutSecurityError: If archive identity, structure, contents, tree
            identity, or project policy violates the corpus boundary.
    """
    findings: list[SecurityFinding] = []
    if archive_path.is_symlink() or not archive_path.is_file():
        raise CheckoutSecurityError(
            (
                SecurityFinding(
                    "archive-member", f"sdist archive is not a regular file: {archive_path}"
                ),
            )
        )
    size = archive_path.stat().st_size
    if source.archive_size <= 0 or source.archive_size > MAX_ARCHIVE_BYTES:
        findings.append(
            SecurityFinding("archive-size", "locked sdist archive size exceeds safety bounds")
        )
    if size != source.archive_size or size > MAX_ARCHIVE_BYTES:
        findings.append(
            SecurityFinding(
                "archive-size",
                f"sdist archive size {size} does not equal locked size {source.archive_size}",
            )
        )
    archive_sha256 = _sha256_regular_file(archive_path)
    if archive_sha256 != source.archive_sha256:
        findings.append(
            SecurityFinding(
                "archive-digest",
                f"sdist archive SHA-256 {archive_sha256} does not equal locked digest",
            )
        )
    if findings:
        raise CheckoutSecurityError(tuple(findings))

    members = _validated_sdist_members(archive_path)
    if destination.exists() or destination.is_symlink():
        raise CheckoutSecurityError(
            (SecurityFinding("archive-member", f"sdist destination is not fresh: {destination}"),)
        )
    destination.mkdir(parents=True)
    try:
        source_manifest = _populate_sdist_destination(
            archive_path,
            destination,
            members,
            source.tree_sha256,
            project_subroot,
        )
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return SdistValidation(
        archive_sha256=archive_sha256,
        source_manifest=source_manifest,
    )


def regular_source_manifest(
    root: Path,
    *,
    paths: tuple[PurePosixPath, ...] | None = None,
) -> TrackedSourceManifest:
    """Hash a link-free extracted source tree using normalized regular paths.

    Args:
        root: Existing extracted sdist root.
        paths: Optional locked regular-file allowlist. When provided, generated
            files are ignored while every named source file remains mandatory.

    Returns:
        TrackedSourceManifest: Sorted regular-file identity.

    Raises:
        CheckoutSecurityError: If a path escapes, is linked or special, or
            exceeds the extraction bounds.
    """
    resolved = root.resolve(strict=True)
    if paths is not None:
        return _regular_manifest_for_paths(resolved, paths)
    records: list[TrackedFileDigest] = []
    total = 0
    for directory, directory_names, file_names in os.walk(resolved, followlinks=False):
        directory_path = Path(directory)
        for name in tuple(directory_names):
            path = directory_path / name
            if path.is_symlink():
                raise CheckoutSecurityError(
                    (SecurityFinding("archive-member", f"extracted source contains link {path}"),)
                )
        for name in file_names:
            path = directory_path / name
            relative = PurePosixPath(path.relative_to(resolved).as_posix())
            if relative == PurePosixPath(".git"):
                continue
            stat_result = path.lstat()
            if not path.is_file() or path.is_symlink():
                raise CheckoutSecurityError(
                    (
                        SecurityFinding(
                            "archive-member",
                            f"extracted source contains non-regular file {relative.as_posix()!r}",
                            relative,
                        ),
                    )
                )
            if stat_result.st_size > MAX_EXTRACTED_FILE_BYTES:
                raise CheckoutSecurityError(
                    (SecurityFinding("archive-size", f"extracted file is too large: {relative}"),)
                )
            total += stat_result.st_size
            if total > MAX_EXTRACTED_BYTES or len(records) >= MAX_ARCHIVE_MEMBERS:
                raise CheckoutSecurityError(
                    (SecurityFinding("archive-size", "extracted source exceeds safety bounds"),)
                )
            content = path.read_bytes()
            records.append(
                TrackedFileDigest(relative, hashlib.sha256(content).hexdigest(), len(content))
            )
    return _manifest_from_records(tuple(sorted(records, key=lambda item: item.path.as_posix())))


def _regular_manifest_for_paths(
    root: Path,
    paths: tuple[PurePosixPath, ...],
) -> TrackedSourceManifest:
    records: list[TrackedFileDigest] = []
    total = 0
    for relative in paths:
        if not _is_safe_path(relative):
            _raise_archive_error("unsafe-path", f"locked source path is unsafe: {relative}")
        path = root.joinpath(*relative.parts)
        try:
            stat_result = path.lstat()
        except OSError as error:
            _raise_archive_error(
                "archive-member", f"cannot read locked source file {relative}: {error}", relative
            )
        if not path.is_file() or path.is_symlink():
            _raise_archive_error(
                "archive-member", f"locked source path is not regular: {relative}", relative
            )
        total += stat_result.st_size
        if stat_result.st_size > MAX_EXTRACTED_FILE_BYTES or total > MAX_EXTRACTED_BYTES:
            _raise_archive_error("archive-size", "locked source files exceed safety bounds")
        content = path.read_bytes()
        records.append(
            TrackedFileDigest(relative, hashlib.sha256(content).hexdigest(), len(content))
        )
    return _manifest_from_records(tuple(records))


@dataclass(frozen=True, slots=True)
class _SdistMember:
    """Validated archive member with its stripped normalized output path."""

    tar_name: str
    path: PurePosixPath | None
    is_directory: bool
    size: int


@dataclass(slots=True)
class _SdistScan:
    """Bounded state accumulated while validating tar headers."""

    members: list[_SdistMember] = field(default_factory=list)
    normalized_names: set[str] = field(default_factory=set)
    roots: set[str] = field(default_factory=set)
    regular_paths: set[PurePosixPath] = field(default_factory=set)
    all_paths: set[PurePosixPath] = field(default_factory=set)
    total_size: int = 0


def _validated_sdist_members(archive_path: Path) -> tuple[_SdistMember, ...]:
    scan = _SdistScan()
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for index, member in enumerate(archive):
                _scan_sdist_member(member, index, scan)
    except (OSError, tarfile.TarError) as error:
        _raise_archive_error("archive-member", f"cannot parse sdist tar archive: {error}")
    if len(scan.roots) != 1:
        _raise_archive_error("archive-member", "sdist must contain exactly one top-level directory")
    if not scan.regular_paths:
        _raise_archive_error("archive-member", "sdist contains no regular source files")
    for path in scan.all_paths:
        for parent in path.parents:
            if parent == PurePosixPath("."):
                break
            if parent in scan.regular_paths:
                _raise_archive_error(
                    "archive-member",
                    f"sdist path descends through regular file {parent.as_posix()!r}",
                    path,
                )
    return tuple(scan.members)


def _scan_sdist_member(member: tarfile.TarInfo, index: int, scan: _SdistScan) -> None:
    if index >= MAX_ARCHIVE_MEMBERS:
        _raise_archive_error("archive-size", "sdist contains too many members")
    path = _normalized_tar_path(member.name)
    if path is None:
        _raise_archive_error(
            "unsafe-path",
            f"sdist member path is not normalized and relative: {member.name!r}",
        )
    path_text = path.as_posix()
    if path_text in scan.normalized_names:
        _raise_archive_error(
            "archive-member", f"sdist contains duplicate normalized path {path_text!r}", path
        )
    scan.normalized_names.add(path_text)
    scan.roots.add(path.parts[0])
    if not member.isdir() and not member.isreg():
        _raise_archive_error(
            "archive-member",
            f"sdist member is not a regular file or directory: {path_text!r}",
            path,
        )
    stripped = PurePosixPath(*path.parts[1:]) if len(path.parts) > 1 else None
    _scan_sdist_payload(member, path, stripped, scan)


def _scan_sdist_payload(
    member: tarfile.TarInfo,
    archive_path: PurePosixPath,
    stripped: PurePosixPath | None,
    scan: _SdistScan,
) -> None:
    if stripped is not None and stripped.name == ".gitmodules":
        _raise_archive_error(
            "gitmodules",
            f"sdist contains forbidden Git submodule metadata: {stripped!s}",
            stripped,
        )
    if stripped is None and member.isreg():
        _raise_archive_error("archive-member", "sdist top-level root is a regular file")
    if stripped is not None and stripped.name == ".git" and member.isreg():
        scan.members.append(_SdistMember(member.name, None, False, 0))
        return
    if stripped is not None and ".git" in stripped.parts:
        _raise_archive_error(
            "archive-member", f"sdist contains VCS administration path {stripped!s}", stripped
        )
    if member.isreg():
        _record_regular_sdist_member(member, archive_path, stripped, scan)
    if stripped is not None:
        scan.all_paths.add(stripped)
    scan.members.append(_SdistMember(member.name, stripped, member.isdir(), member.size))


def _record_regular_sdist_member(
    member: tarfile.TarInfo,
    archive_path: PurePosixPath,
    stripped: PurePosixPath | None,
    scan: _SdistScan,
) -> None:
    if member.size < 0 or member.size > MAX_EXTRACTED_FILE_BYTES:
        _raise_archive_error(
            "archive-size", f"sdist member exceeds file bound: {archive_path!s}", archive_path
        )
    scan.total_size += member.size
    if scan.total_size > MAX_EXTRACTED_BYTES:
        _raise_archive_error("archive-size", "sdist extracted bytes exceed safety bound")
    if stripped is not None:
        scan.regular_paths.add(stripped)


def _raise_archive_error(
    code: FindingCode,
    message: str,
    path: PurePosixPath | None = None,
) -> Never:
    raise CheckoutSecurityError((SecurityFinding(code, message, path),))


def _normalized_tar_path(value: str) -> PurePosixPath | None:
    if not value or "\\" in value or "\x00" in value:
        return None
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value.rstrip("/")
    ):
        return None
    return path


def _extract_sdist_members(
    archive_path: Path,
    destination: Path,
    members: tuple[_SdistMember, ...],
) -> TrackedSourceManifest:
    records: list[TrackedFileDigest] = []
    expected = {member.tar_name: member for member in members}
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for tar_member in archive:
            member = expected.get(tar_member.name)
            if member is None or member.path is None:
                continue
            target = destination.joinpath(*member.path.parts)
            resolved_parent = target.parent.resolve(strict=False)
            if not resolved_parent.is_relative_to(destination.resolve()):
                raise CheckoutSecurityError(
                    (
                        SecurityFinding(
                            "unsafe-path", f"sdist member escapes destination: {member.path}"
                        ),
                    )
                )
            if member.is_directory:
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(tar_member)
            if source is None:
                raise CheckoutSecurityError(
                    (SecurityFinding("archive-member", f"cannot read sdist member {member.path}"),)
                )
            digest = hashlib.sha256()
            written = 0
            with source, target.open("xb") as output:
                while chunk := source.read(1024 * 1024):
                    written += len(chunk)
                    if written > member.size:
                        raise CheckoutSecurityError(
                            (
                                SecurityFinding(
                                    "archive-size",
                                    f"sdist member expanded beyond size: {member.path}",
                                ),
                            )
                        )
                    digest.update(chunk)
                    output.write(chunk)
            if written != member.size:
                raise CheckoutSecurityError(
                    (SecurityFinding("archive-size", f"sdist member size changed: {member.path}"),)
                )
            records.append(TrackedFileDigest(member.path, digest.hexdigest(), written))
    return _manifest_from_records(tuple(sorted(records, key=lambda item: item.path.as_posix())))


def _populate_sdist_destination(
    archive_path: Path,
    destination: Path,
    members: tuple[_SdistMember, ...],
    expected_tree_sha256: str,
    project_subroot: PurePosixPath,
) -> TrackedSourceManifest:
    source_manifest = _extract_sdist_members(archive_path, destination, members)
    policy_findings: list[SecurityFinding] = []
    _inspect_pyproject(destination.resolve(), project_subroot, policy_findings)
    if policy_findings:
        raise CheckoutSecurityError(tuple(policy_findings))
    if source_manifest.manifest_digest != expected_tree_sha256:
        _raise_archive_error(
            "tree-digest", "extracted sdist regular-file tree does not equal locked digest"
        )
    return source_manifest


def _manifest_from_records(files: tuple[TrackedFileDigest, ...]) -> TrackedSourceManifest:
    aggregate = hashlib.sha256()
    for record in files:
        _update_length_prefixed(aggregate, record.path.as_posix().encode("utf-8"))
        _update_length_prefixed(aggregate, record.sha256.encode("ascii"))
        _update_length_prefixed(aggregate, str(record.size).encode("ascii"))
    return TrackedSourceManifest(files=files, manifest_digest=aggregate.hexdigest())


def _sha256_regular_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
        # Legacy setup.py projects are valid corpus inputs.  The lifecycle
        # creates a disposable PEP 517 policy file after baseline qualification.
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
