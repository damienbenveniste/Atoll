"""Implementation of the `atoll clean` command."""

from __future__ import annotations

import importlib.machinery
import shutil
from dataclasses import dataclass
from pathlib import Path

from atoll.config import load_enabled_islands


@dataclass(frozen=True, slots=True)
class CleanOptions:
    """User-facing options for removing Atoll-generated outputs."""

    root: Path
    cache: bool = False
    artifacts: bool = False
    all_outputs: bool = False


@dataclass(frozen=True, slots=True)
class CleanCommandResult:
    """Paths removed by a clean operation."""

    removed: tuple[Path, ...]


def execute_clean(options: CleanOptions) -> CleanCommandResult:
    """Remove Atoll cache/build outputs and compiled sidecar artifacts."""
    root = options.root.resolve()
    remove_cache = options.cache or options.all_outputs or not options.artifacts
    remove_artifacts = options.artifacts or options.all_outputs
    removed: list[Path] = []
    if remove_cache:
        removed.extend(
            _remove_dir(path) for path in (root / ".atoll" / "cache", root / ".atoll" / "build")
        )
    if remove_artifacts:
        for path in _compiled_artifacts(root):
            if path.exists():
                path.unlink()
                removed.append(path)
    return CleanCommandResult(removed=tuple(path for path in removed if path.exists() is False))


def _remove_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    return path


def _compiled_artifacts(root: Path) -> tuple[Path, ...]:
    artifacts: set[Path] = set()
    for island in load_enabled_islands(root):
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            artifacts.update(
                island.sidecar_path.parent.glob(f"{island.sidecar_path.stem}*{suffix}")
            )
    return tuple(sorted(artifacts))
