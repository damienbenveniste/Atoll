"""Security tests for detached local benchmark checkouts."""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

import pytest
from scripts.benchmark_corpus.models import SdistSource
from scripts.benchmark_corpus.security import (
    CheckoutSecurityError,
    extract_sdist_archive,
    tracked_source_manifest,
    validate_checkout,
)


def _git_executable() -> str:
    """Return the local Git executable required by repository-only tests."""
    executable = shutil.which("git")
    if executable is None:
        pytest.skip("Git is required for checkout security tests")
    return executable


def _git(repository: Path, *arguments: str) -> str:
    """Run a deterministic local Git command without network access."""
    result = subprocess.run(
        (_git_executable(), "-C", os.fspath(repository), *arguments),
        shell=False,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str]:
    """Create a detached local repository with a minimal target project."""
    repository = tmp_path / "checkout"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.email", "benchmark@example.invalid")
    _git(repository, "config", "user.name", "Benchmark Corpus")
    (repository / "pyproject.toml").write_text(
        '[project]\nname = "fixture"\nversion = "1.0.0"\n',
        encoding="utf-8",
    )
    (repository / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repository, "add", "pyproject.toml", "source.py")
    _git(repository, "commit", "--quiet", "-m", "initial")
    revision = _git(repository, "rev-parse", "HEAD")
    _git(repository, "checkout", "--quiet", "--detach", revision)
    return repository, revision


def _commit_detached(repository: Path, message: str) -> str:
    """Commit currently staged changes while keeping ``HEAD`` detached."""
    _git(repository, "commit", "--quiet", "-m", message)
    return _git(repository, "rev-parse", "HEAD")


def _finding_codes(error: CheckoutSecurityError) -> set[str]:
    """Return stable finding codes from a rejected checkout."""
    return {finding.code for finding in error.findings}


def _tar_archive(
    path: Path,
    members: tuple[tuple[str, bytes, bytes, str], ...],
) -> Path:
    """Write a small tar.gz with exact member types for security tests."""
    with tarfile.open(path, mode="w:gz") as archive:
        for name, content, member_type, linkname in members:
            member = tarfile.TarInfo(name)
            member.type = member_type
            member.linkname = linkname
            member.size = len(content) if member_type == tarfile.REGTYPE else 0
            archive.addfile(member, io.BytesIO(content) if member.size else None)
    return path


def _tree_digest(files: tuple[tuple[str, bytes], ...]) -> str:
    """Encode regular files using the corpus tree-manifest contract."""
    aggregate = hashlib.sha256()
    for name, content in sorted(files):
        content_digest = hashlib.sha256(content).hexdigest()
        for value in (name.encode(), content_digest.encode(), str(len(content)).encode()):
            aggregate.update(len(value).to_bytes(8, byteorder="big"))
            aggregate.update(value)
    return aggregate.hexdigest()


def _sdist_source(archive: Path, files: tuple[tuple[str, bytes], ...]) -> SdistSource:
    """Lock one local test archive and its expected stripped tree."""
    content = archive.read_bytes()
    return SdistSource(
        url="https://example.invalid/source.tar.gz",
        archive_sha256=hashlib.sha256(content).hexdigest(),
        archive_size=len(content),
        tree_sha256=_tree_digest(files),
    )


def test_validate_checkout_rejects_head_that_moved_from_pin(tmp_path: Path) -> None:
    """A new detached commit cannot silently replace the reviewed revision."""
    repository, expected_revision = _repository(tmp_path)
    (repository / "source.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(repository, "add", "source.py")
    observed_revision = _commit_detached(repository, "move head")

    with pytest.raises(CheckoutSecurityError) as raised:
        validate_checkout(
            repository,
            expected_revision,
            git_executable=_git_executable(),
        )

    assert observed_revision != expected_revision
    assert "revision-mismatch" in _finding_codes(raised.value)


def test_validate_checkout_rejects_unresolved_lfs_pointer(tmp_path: Path) -> None:
    """Tracked LFS pointer metadata is not accepted as source content."""
    repository, _revision = _repository(tmp_path)
    (repository / "model.bin").write_text(
        "\n".join(
            (
                "version https://git-lfs.github.com/spec/v1",
                f"oid sha256:{'a' * 64}",
                "size 123",
                "",
            )
        ),
        encoding="utf-8",
    )
    _git(repository, "add", "model.bin")
    revision = _commit_detached(repository, "add unresolved LFS pointer")

    with pytest.raises(CheckoutSecurityError) as raised:
        validate_checkout(repository, revision, git_executable=_git_executable())

    assert "lfs-pointer" in _finding_codes(raised.value)
    assert raised.value.findings[0].path == PurePosixPath("model.bin")


def test_validate_checkout_rejects_symlink_escape(tmp_path: Path) -> None:
    """A tracked symlink may not resolve to a target outside the checkout."""
    repository, _revision = _repository(tmp_path)
    (tmp_path / "outside.py").write_text("SECRET = True\n", encoding="utf-8")
    (repository / "escape.py").symlink_to(tmp_path / "outside.py")
    _git(repository, "add", "escape.py")
    revision = _commit_detached(repository, "add escaping symlink")

    with pytest.raises(CheckoutSecurityError) as raised:
        validate_checkout(repository, revision, git_executable=_git_executable())

    assert "symlink-escape" in _finding_codes(raised.value)


def test_git_source_still_rejects_submodule_metadata_and_gitlinks(tmp_path: Path) -> None:
    """Adding sdist support does not relax the strict Git source boundary."""
    repository, revision = _repository(tmp_path)
    (repository / ".gitmodules").write_text(
        '[submodule "vendor"]\npath = vendor\nurl = https://example.invalid/vendor.git\n',
        encoding="utf-8",
    )
    _git(repository, "add", ".gitmodules")
    _git(repository, "update-index", "--add", "--cacheinfo", f"160000,{revision},vendor")
    pinned_revision = _commit_detached(repository, "add forbidden submodule")

    with pytest.raises(CheckoutSecurityError) as raised:
        validate_checkout(repository, pinned_revision, git_executable=_git_executable())

    assert {"gitmodules", "submodule"}.issubset(_finding_codes(raised.value))


def test_validate_checkout_rejects_existing_compile_policy(tmp_path: Path) -> None:
    """Corpus policy injection cannot override an upstream Atoll policy."""
    repository, _revision = _repository(tmp_path)
    (repository / "pyproject.toml").write_text(
        "\n".join(
            (
                "[project]",
                'name = "fixture"',
                'version = "1.0.0"',
                "",
                "[tool.atoll.compile]",
                'backends = ["mypyc"]',
                "",
            )
        ),
        encoding="utf-8",
    )
    _git(repository, "add", "pyproject.toml")
    revision = _commit_detached(repository, "add compile policy")

    with pytest.raises(CheckoutSecurityError) as raised:
        validate_checkout(repository, revision, git_executable=_git_executable())

    assert "compile-policy" in _finding_codes(raised.value)


def test_validate_checkout_accepts_legacy_setup_project_without_pyproject(
    tmp_path: Path,
) -> None:
    """A setup.py-only project reaches the disposable policy lifecycle."""
    repository, _revision = _repository(tmp_path)
    (repository / "pyproject.toml").unlink()
    (repository / "setup.py").write_text(
        "from setuptools import setup\nsetup()\n",
        encoding="utf-8",
    )
    _git(repository, "add", "--all")
    revision = _commit_detached(repository, "use legacy build metadata")

    validated = validate_checkout(repository, revision, git_executable=_git_executable())

    assert validated.revision == revision
    assert PurePosixPath("setup.py") in {record.path for record in validated.source_manifest.files}


def test_tracked_source_manifest_has_stable_sorted_hashes(tmp_path: Path) -> None:
    """Manifest identity is stable and changes only when tracked bytes change."""
    repository, revision = _repository(tmp_path)
    first = tracked_source_manifest(repository, git_executable=_git_executable())
    second = tracked_source_manifest(repository, git_executable=_git_executable())

    assert first == second
    assert tuple(record.path.as_posix() for record in first.files) == (
        "pyproject.toml",
        "source.py",
    )
    source = next(record for record in first.files if record.path == PurePosixPath("source.py"))
    assert source.sha256 == hashlib.sha256(b"VALUE = 1\n").hexdigest()
    assert source.size == len(b"VALUE = 1\n")

    validated = validate_checkout(repository, revision, git_executable=_git_executable())
    assert validated.source_manifest == first

    (repository / "ignored.txt").write_text("not tracked\n", encoding="utf-8")
    assert tracked_source_manifest(repository, git_executable=_git_executable()) == first

    (repository / "source.py").write_text("VALUE = 9\n", encoding="utf-8")
    changed = tracked_source_manifest(repository, git_executable=_git_executable())
    assert changed.manifest_digest != first.manifest_digest
    assert changed.files != first.files


def test_sdist_hash_is_rejected_before_malformed_archive_is_parsed(tmp_path: Path) -> None:
    """Untrusted bytes never reach tar parsing before content authentication."""
    archive = tmp_path / "source.tar.gz"
    archive.write_bytes(b"not a tar archive")
    source = SdistSource(
        url="https://example.invalid/source.tar.gz",
        archive_sha256="0" * 64,
        archive_size=archive.stat().st_size,
        tree_sha256="0" * 64,
    )

    with pytest.raises(CheckoutSecurityError) as raised:
        extract_sdist_archive(archive, tmp_path / "checkout", source)

    assert _finding_codes(raised.value) == {"archive-digest"}
    assert not (tmp_path / "checkout").exists()


def test_sdist_extracts_one_root_and_omits_git_pointer_files(tmp_path: Path) -> None:
    """Verified regular files are root-stripped while VCS pointers are omitted."""
    pyproject = b'[project]\nname = "fixture"\nversion = "1.0"\n'
    source_bytes = b"VALUE = 1\n"
    files = (("pyproject.toml", pyproject), ("src/source.py", source_bytes))
    archive = _tar_archive(
        tmp_path / "source.tar.gz",
        (
            ("fixture-1.0/pyproject.toml", pyproject, tarfile.REGTYPE, ""),
            ("fixture-1.0/src/source.py", source_bytes, tarfile.REGTYPE, ""),
            ("fixture-1.0/tests/data/.git", b"gitdir: ../admin\n", tarfile.REGTYPE, ""),
        ),
    )

    validation = extract_sdist_archive(
        archive,
        tmp_path / "checkout",
        _sdist_source(archive, files),
    )

    assert validation.source_manifest.manifest_digest == _tree_digest(files)
    assert (tmp_path / "checkout/src/source.py").read_bytes() == source_bytes
    assert not (tmp_path / "checkout/tests/data/.git").exists()


@pytest.mark.parametrize(
    ("members", "code"),
    [
        ((("root/../escape.py", b"x", tarfile.REGTYPE, ""),), "unsafe-path"),
        ((("/root/escape.py", b"x", tarfile.REGTYPE, ""),), "unsafe-path"),
        ((("root\\escape.py", b"x", tarfile.REGTYPE, ""),), "unsafe-path"),
        (
            (
                ("root/source.py", b"x", tarfile.REGTYPE, ""),
                ("root/source.py", b"y", tarfile.REGTYPE, ""),
            ),
            "archive-member",
        ),
        (
            (
                ("root/source.py", b"x", tarfile.REGTYPE, ""),
                ("other/source.py", b"y", tarfile.REGTYPE, ""),
            ),
            "archive-member",
        ),
        ((("root/link", b"", tarfile.SYMTYPE, "outside"),), "archive-member"),
        ((("root/link", b"", tarfile.LNKTYPE, "root/source.py"),), "archive-member"),
        ((("root/device", b"", tarfile.CHRTYPE, ""),), "archive-member"),
        ((("root/.gitmodules", b"[submodule]\n", tarfile.REGTYPE, ""),), "gitmodules"),
    ],
)
def test_sdist_rejects_unsafe_member_graphs(
    tmp_path: Path,
    members: tuple[tuple[str, bytes, bytes, str], ...],
    code: str,
) -> None:
    """Tar metadata cannot introduce aliases, external paths, or submodules."""
    archive = _tar_archive(tmp_path / "source.tar.gz", members)
    source = _sdist_source(archive, (("placeholder", b"placeholder"),))

    with pytest.raises(CheckoutSecurityError) as raised:
        extract_sdist_archive(archive, tmp_path / "checkout", source)

    assert code in _finding_codes(raised.value)
    assert not (tmp_path / "checkout").exists()


def test_sdist_rejects_declared_file_over_size_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Declared expansion size is bounded before extraction writes a file."""
    archive = _tar_archive(
        tmp_path / "source.tar.gz",
        (("root/source.py", b"four", tarfile.REGTYPE, ""),),
    )
    monkeypatch.setattr("scripts.benchmark_corpus.security.MAX_EXTRACTED_FILE_BYTES", 3)

    with pytest.raises(CheckoutSecurityError) as raised:
        extract_sdist_archive(
            archive,
            tmp_path / "checkout",
            _sdist_source(archive, (("source.py", b"four"),)),
        )

    assert _finding_codes(raised.value) == {"archive-size"}
    assert not (tmp_path / "checkout").exists()
