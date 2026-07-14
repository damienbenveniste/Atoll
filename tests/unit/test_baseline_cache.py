"""Tests for content-addressed target-wheel cache behavior."""

import json
import os
import shutil
import sys
from pathlib import Path
from typing import cast

import pytest

from atoll.baseline_cache import (
    BASELINE_WHEEL_CACHE_CONTEXT_ENV,
    baseline_wheel_cache_key,
    restore_baseline_wheel,
    store_baseline_wheel,
)


@pytest.fixture(autouse=True)
def reproducible_build_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give ordinary cache tests a stable caller-owned dependency identity."""
    monkeypatch.setenv(BASELINE_WHEEL_CACHE_CONTEXT_ENV, "baseline-cache-unit-tests")


def test_baseline_wheel_cache_restores_verified_artifact(tmp_path: Path) -> None:
    """A stored wheel is copied into a fresh output directory on a warm lookup."""
    project_root = _project(tmp_path)
    cache_root = tmp_path / "cache"
    output_dir = tmp_path / "output"
    wheel_path = tmp_path / "demo-1.0-py3-none-any.whl"
    wheel_path.write_bytes(b"verified-wheel")

    cold = restore_baseline_wheel(
        project_root=project_root,
        cache_root=cache_root,
        output_dir=output_dir,
    )
    assert cold.key is not None
    stored = store_baseline_wheel(
        key=cold.key,
        wheel_path=wheel_path,
        cache_root=cache_root,
    )
    (output_dir / "stale.txt").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "stale.txt").write_text("stale", encoding="utf-8")
    warm = restore_baseline_wheel(
        project_root=project_root,
        cache_root=cache_root,
        output_dir=output_dir,
    )

    assert cold.status == "miss"
    assert stored.stored is True
    assert warm.status == "hit"
    assert warm.key == cold.key
    assert warm.wheel_path is not None
    assert warm.wheel_path.read_bytes() == b"verified-wheel"
    assert not (output_dir / "stale.txt").exists()
    assert store_baseline_wheel(
        key=cold.key,
        wheel_path=wheel_path,
        cache_root=cache_root,
    ).stored


def test_baseline_wheel_cache_invalidates_source_and_build_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Source bytes and relevant compiler flags participate in the strict key."""
    project_root = _project(tmp_path)
    original = baseline_wheel_cache_key(project_root)

    (project_root / "src" / "demo.py").write_text("VALUE = 2\n", encoding="utf-8")
    changed_source = baseline_wheel_cache_key(project_root)
    monkeypatch.setenv("CFLAGS", "-O2")
    changed_environment = baseline_wheel_cache_key(project_root)

    assert changed_source != original
    assert changed_environment != changed_source


def test_baseline_wheel_cache_fingerprints_symlink_target_text(tmp_path: Path) -> None:
    """Equivalent and dangling link spellings remain distinct build inputs."""
    project_root = _project(tmp_path)
    link = project_root / "src" / "current.py"
    link.symlink_to("demo.py")
    original = baseline_wheel_cache_key(project_root)

    link.unlink()
    link.symlink_to("./demo.py")
    equivalent_referent = baseline_wheel_cache_key(project_root)
    link.unlink()
    link.symlink_to("missing.py")
    dangling = baseline_wheel_cache_key(project_root)

    assert equivalent_referent != original
    assert dangling != equivalent_referent


def test_baseline_wheel_cache_invalidates_arbitrary_environment_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Custom backend variables and build-visible mtimes participate in the key."""
    project_root = _project(tmp_path)
    source_path = project_root / "src" / "demo.py"
    original = baseline_wheel_cache_key(project_root)

    monkeypatch.setenv("WHEEL_FLAVOR", "debug")
    changed_environment = baseline_wheel_cache_key(project_root)
    source_stat = source_path.stat()
    os.utime(
        source_path,
        ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns + 1_000_000_000),
    )
    changed_metadata = baseline_wheel_cache_key(project_root)

    assert changed_environment != original
    assert changed_metadata != changed_environment


def test_baseline_wheel_cache_bypasses_unlocked_online_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutable package indexes never produce a reusable baseline-wheel identity."""
    monkeypatch.delenv(BASELINE_WHEEL_CACHE_CONTEXT_ENV)
    monkeypatch.delenv("PIP_NO_INDEX", raising=False)
    monkeypatch.delenv("PIP_FIND_LINKS", raising=False)
    cache_root = tmp_path / "cache"

    probe = restore_baseline_wheel(
        project_root=_project(tmp_path),
        cache_root=cache_root,
        output_dir=tmp_path / "output",
    )

    assert probe.status == "bypass"
    assert probe.key is None
    assert "reproducible" in probe.reason
    assert not cache_root.exists()


def test_baseline_wheel_cache_keys_complete_offline_wheelhouse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing an offline build artifact invalidates dependency resolution."""
    monkeypatch.delenv(BASELINE_WHEEL_CACHE_CONTEXT_ENV)
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    backend = wheelhouse / "setuptools-1-py3-none-any.whl"
    backend.write_bytes(b"backend-v1")
    (wheelhouse / "current.whl").symlink_to(backend.name)
    monkeypatch.setenv("PIP_NO_INDEX", "1")
    monkeypatch.setenv("PIP_FIND_LINKS", str(wheelhouse))
    project_root = _project(tmp_path)

    original = baseline_wheel_cache_key(project_root)
    first_probe = restore_baseline_wheel(
        project_root=project_root,
        cache_root=tmp_path / "cache",
        output_dir=tmp_path / "output",
    )
    backend.write_bytes(b"backend-v2")
    changed = baseline_wheel_cache_key(project_root)
    (wheelhouse / "current.whl").unlink()
    (wheelhouse / "current.whl").symlink_to(f"./{backend.name}")
    changed_link_text = baseline_wheel_cache_key(project_root)

    assert first_probe.status == "miss"
    assert changed != original
    assert changed_link_text != changed


@pytest.mark.parametrize(
    "unsafe_kind",
    ["html-index", "direct-requirement", "constraint-file"],
)
def test_baseline_wheel_cache_bypasses_mutable_offline_references(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_kind: str,
) -> None:
    """Local references that can redirect dependency bytes disable automatic reuse.

    Args:
        tmp_path: Isolated project and build-input directory.
        monkeypatch: Environment isolation for the offline resolver.
        unsafe_kind: Mutable reference shape exercised by this case.
    """
    monkeypatch.delenv(BASELINE_WHEEL_CACHE_CONTEXT_ENV)
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    project_root = _project(tmp_path)
    find_links = wheelhouse
    if unsafe_kind == "html-index":
        find_links = tmp_path / "packages.html"
        find_links.write_text(
            '<a href="https://example.invalid/backend.whl">x</a>', encoding="utf-8"
        )
    elif unsafe_kind == "direct-requirement":
        (project_root / "pyproject.toml").write_text(
            "[build-system]\n"
            "requires = ['backend @ file:///tmp/backend.whl']\n"
            "build-backend = 'backend'\n",
            encoding="utf-8",
        )
    else:
        constraint = tmp_path / "build-constraints.txt"
        constraint.write_text("backend @ https://example.invalid/backend.whl\n", encoding="utf-8")
        monkeypatch.setenv("PIP_BUILD_CONSTRAINT", str(constraint))
    monkeypatch.setenv("PIP_NO_INDEX", "1")
    monkeypatch.setenv("PIP_FIND_LINKS", str(find_links))

    probe = restore_baseline_wheel(
        project_root=project_root,
        cache_root=tmp_path / "cache",
        output_dir=tmp_path / "output",
    )

    assert probe.status == "bypass"


def test_baseline_wheel_cache_rejects_corrupt_artifact(tmp_path: Path) -> None:
    """Digest mismatch turns a persisted entry into an ordinary cache miss."""
    project_root = _project(tmp_path)
    cache_root = tmp_path / "cache"
    wheel_path = tmp_path / "demo-1.0-py3-none-any.whl"
    wheel_path.write_bytes(b"original")
    key = baseline_wheel_cache_key(project_root)
    store_baseline_wheel(key=key, wheel_path=wheel_path, cache_root=cache_root)
    (cache_root / key / wheel_path.name).write_bytes(b"corrupt")

    restored = restore_baseline_wheel(
        project_root=project_root,
        cache_root=cache_root,
        output_dir=tmp_path / "output",
    )

    assert restored.status == "miss"
    assert restored.wheel_path is None
    assert store_baseline_wheel(
        key=key,
        wheel_path=wheel_path,
        cache_root=cache_root,
    ).stored


def test_baseline_wheel_cache_tracks_worktree_git_metadata(tmp_path: Path) -> None:
    """A copied worktree pointer includes HEAD, staged entries, and common refs."""
    project_root = _project(tmp_path)
    git_dir = tmp_path / "worktree-git"
    common_dir = tmp_path / "common-git"
    tag = common_dir / "refs" / "tags" / "v1"
    tag.parent.mkdir(parents=True)
    git_dir.mkdir()
    (project_root / ".git").write_text(f"gitdir: {git_dir}\n", encoding="utf-8")
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git_dir / "index").write_bytes(b"index-v1")
    (git_dir / "commondir").write_text(str(common_dir), encoding="utf-8")
    (common_dir / "packed-refs").write_text("# pack-refs\n", encoding="utf-8")
    tag.write_text("a" * 40 + "\n", encoding="utf-8")

    first = baseline_wheel_cache_key(project_root)
    tag.write_text("b" * 40 + "\n", encoding="utf-8")
    second = baseline_wheel_cache_key(project_root)

    assert first != second


def test_baseline_wheel_cache_normalizes_git_index_stat_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Index housekeeping is stable while staged-entry changes invalidate the key."""
    project_root = _project(tmp_path)
    git_dir = tmp_path / "git"
    git_dir.mkdir()
    index = git_dir / "index"
    index.write_bytes(b"raw-index-stat-cache-v1")
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (project_root / ".git").write_text(f"gitdir: {git_dir}\n", encoding="utf-8")
    semantic_index = tmp_path / "semantic-index"
    semantic_index.write_bytes(b"100644 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa 0\tsrc/demo.py\0")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text(
        f"#!{sys.executable}\n"
        "from pathlib import Path\n"
        "import sys\n"
        f"sys.stdout.buffer.write(Path({str(semantic_index)!r}).read_bytes())\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")

    first = baseline_wheel_cache_key(project_root)
    index.write_bytes(b"raw-index-stat-cache-v2")
    housekeeping_only = baseline_wheel_cache_key(project_root)
    semantic_index.write_bytes(b"100644 bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb 0\tsrc/demo.py\0")
    staged_change = baseline_wheel_cache_key(project_root)

    assert housekeeping_only == first
    assert staged_change != housekeeping_only


@pytest.mark.parametrize(
    "manifest",
    [
        [],
        {"version": 0, "key": "key", "wheel_name": "demo.whl", "wheel_sha256": "x"},
        {"version": 1, "key": "key", "wheel_name": "demo.whl"},
        {"version": 1, "key": "wrong", "wheel_name": "demo.whl", "wheel_sha256": "x"},
        {"version": 1, "key": "key", "wheel_name": "../demo.whl", "wheel_sha256": "x"},
        {"version": 1, "key": "key", "wheel_name": "missing.whl", "wheel_sha256": "x"},
    ],
)
def test_baseline_wheel_cache_rejects_invalid_manifest(
    tmp_path: Path,
    manifest: object,
) -> None:
    """Unknown, unsafe, and incomplete manifests remain cache misses.

    Args:
        tmp_path: Isolated project and cache directory.
        manifest: Invalid JSON-compatible cache metadata.
    """
    project_root = _project(tmp_path)
    cache_root = tmp_path / "cache"
    key = baseline_wheel_cache_key(project_root)
    normalized: object = manifest
    if isinstance(manifest, dict):
        payload = cast(dict[object, object], manifest)
        if payload.get("key") == "key":
            normalized = {**payload, "key": key}
    entry = cache_root / key
    entry.mkdir(parents=True)
    (entry / "manifest.json").write_text(json.dumps(normalized), encoding="utf-8")

    restored = restore_baseline_wheel(
        project_root=project_root,
        cache_root=cache_root,
        output_dir=tmp_path / "output",
    )

    assert restored.status == "miss"


def test_baseline_wheel_cache_copy_failures_are_nonfatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restore and store I/O failures degrade to misses instead of failing compile."""
    project_root = _project(tmp_path)
    cache_root = tmp_path / "cache"
    wheel_path = tmp_path / "demo-1.0-py3-none-any.whl"
    wheel_path.write_bytes(b"wheel")
    key = baseline_wheel_cache_key(project_root)
    assert store_baseline_wheel(key=key, wheel_path=wheel_path, cache_root=cache_root).stored

    def fail_copy(_source: Path, _destination: Path) -> None:
        raise OSError("read-only cache")

    monkeypatch.setattr("atoll.baseline_cache.shutil.copy2", fail_copy)
    restored = restore_baseline_wheel(
        project_root=project_root,
        cache_root=cache_root,
        output_dir=tmp_path / "output",
    )
    failed_store = store_baseline_wheel(
        key="f" * 64,
        wheel_path=wheel_path,
        cache_root=tmp_path / "unwritable-cache",
    )

    assert restored.status == "miss"
    assert failed_store.stored is False


def test_baseline_wheel_cache_rehashes_restored_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A verify-copy race cannot promote bytes that differ from the manifest."""
    project_root = _project(tmp_path)
    cache_root = tmp_path / "cache"
    wheel_path = tmp_path / "demo-1.0-py3-none-any.whl"
    wheel_path.write_bytes(b"wheel")
    key = baseline_wheel_cache_key(project_root)
    assert store_baseline_wheel(key=key, wheel_path=wheel_path, cache_root=cache_root).stored
    copy2 = shutil.copy2

    def corrupt_after_copy(source: Path, destination: Path) -> str:
        result = copy2(source, destination)
        Path(destination).write_bytes(b"changed-after-verification")
        return str(result)

    monkeypatch.setattr("atoll.baseline_cache.shutil.copy2", corrupt_after_copy)

    restored = restore_baseline_wheel(
        project_root=project_root,
        cache_root=cache_root,
        output_dir=tmp_path / "output",
    )

    assert restored.status == "miss"
    assert restored.wheel_path is None
    assert "digest" in restored.reason


def test_baseline_wheel_cache_rejects_unsafe_store_key(tmp_path: Path) -> None:
    """A malformed internal key cannot delete or replace paths outside the cache."""
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "marker.txt"
    marker.write_text("keep", encoding="utf-8")
    wheel_path = tmp_path / "demo-1.0-py3-none-any.whl"
    wheel_path.write_bytes(b"wheel")

    stored = store_baseline_wheel(
        key="../outside",
        wheel_path=wheel_path,
        cache_root=tmp_path / "cache",
    )

    assert stored.stored is False
    assert marker.read_text(encoding="utf-8") == "keep"


def _project(tmp_path: Path) -> Path:
    project_root = tmp_path / "project"
    source_root = project_root / "src"
    source_root.mkdir(parents=True)
    (source_root / "demo.py").write_text("VALUE = 1\n", encoding="utf-8")
    (project_root / "pyproject.toml").write_text(
        "[build-system]\nrequires = ['setuptools']\nbuild-backend = 'setuptools.build_meta'\n",
        encoding="utf-8",
    )
    return project_root
