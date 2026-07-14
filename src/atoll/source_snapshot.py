"""Create disposable source snapshots without changing filesystem links.

The helper in this module copies project and source trees for build and search
workspaces. It deliberately does not resolve, validate, or rewrite symlink
targets; snapshot policy such as ignored paths remains owned by each caller.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path

type SourceSnapshotIgnore = Callable[[str, list[str]], Iterable[str]]

_readlink = os.readlink


def copy_source_snapshot(
    source: Path,
    destination: Path,
    *,
    ignore: SourceSnapshotIgnore | None = None,
) -> None:
    """Copy a source tree while preserving every symlink as a symlink.

    Valid and dangling links are recreated with their original target text, so
    disposable builds observe the same filesystem structure as the checkout.

    Args:
        source: Existing directory whose contents form the snapshot.
        destination: New directory receiving the complete snapshot.
        ignore: Optional per-directory callback returning names to omit.

    Raises:
        OSError: If the source cannot be read or the snapshot cannot be created.
        shutil.Error: If one or more entries cannot be copied.
    """
    shutil.copytree(source, destination, symlinks=True, ignore=ignore)


def symlink_target_bytes(path: Path) -> bytes:
    """Return a symlink's exact, unnormalized target text as filesystem bytes.

    ``Path.readlink()`` constructs another ``Path`` and normalizes spellings
    such as ``./module.py`` to ``module.py``. Cache identities must distinguish
    those link payloads even when they currently resolve to the same referent.

    Args:
        path: Symlink whose stored target text is required.

    Returns:
        bytes: Target text encoded with the platform filesystem encoding.

    Raises:
        OSError: If ``path`` is not a readable symlink.
    """
    return os.fsencode(_readlink(path))
