"""Regression tests for symlink-preserving disposable source snapshots."""

from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from atoll.commands import package as package_command
from atoll.source_optimization import search as source_search
from atoll.source_snapshot import copy_source_snapshot


def test_copy_source_snapshot_preserves_valid_and_dangling_symlinks(tmp_path: Path) -> None:
    """Snapshots retain link identity and exact target text without dereferencing."""
    source = tmp_path / "source"
    package = source / "pkg"
    package.mkdir(parents=True)
    (package / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "alias.py").symlink_to("module.py")
    (package / "missing.py").symlink_to("../generated/missing.py")
    destination = tmp_path / "snapshot"

    copy_source_snapshot(source, destination)

    copied_alias = destination / "pkg" / "alias.py"
    copied_missing = destination / "pkg" / "missing.py"
    assert copied_alias.is_symlink()
    assert copied_alias.readlink() == Path("module.py")
    assert copied_missing.is_symlink()
    assert not copied_missing.exists()
    assert copied_missing.readlink() == Path("../generated/missing.py")


@pytest.mark.parametrize("snapshot_kind", ["pep517", "source-search"])
def test_project_snapshot_call_sites_preserve_dangling_internal_symlinks(
    tmp_path: Path,
    snapshot_kind: str,
) -> None:
    """Build and source-search project copies both use link-preserving snapshots.

    Args:
        tmp_path: Isolated source and destination roots.
        snapshot_kind: Production project-copy call site under test.
    """
    source = tmp_path / "source"
    package = source / "pkg"
    package.mkdir(parents=True)
    dangling = package / "generated.py"
    dangling.symlink_to("../generated/pkg.py")
    destination = tmp_path / "snapshot"
    if snapshot_kind == "pep517":
        copy_project = cast(
            Callable[..., None],
            vars(package_command)["_copy_pep517_project"],
        )
    else:
        copy_project = cast(
            Callable[..., None],
            vars(source_search)["_copy_project"],
        )

    copy_project(source, destination, excluded_output=source / "dist")

    copied = destination / "pkg" / "generated.py"
    assert copied.is_symlink()
    assert not copied.exists()
    assert copied.readlink() == Path("../generated/pkg.py")


def test_staged_source_fingerprint_includes_symlink_target_text(tmp_path: Path) -> None:
    """Backend cache identity changes when only a link's spelling changes."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    link = source / "alias.py"
    link.symlink_to("module.py")
    source_roots_digest = cast(
        Callable[[tuple[Path, ...]], str],
        vars(package_command)["_source_roots_digest"],
    )
    original = source_roots_digest((source,))

    link.unlink()
    link.symlink_to("./module.py")

    assert source_roots_digest((source,)) != original
