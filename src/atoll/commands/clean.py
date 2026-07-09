"""Implementation of the `atoll clean` command.

Cleaning is limited to Atoll-owned cache, build, and compiled sidecar artifacts.
The command discovers configured islands before artifact deletion so it does not
remove unrelated extension modules in the project tree.
"""

from __future__ import annotations

import importlib.machinery
import shutil
from dataclasses import dataclass
from pathlib import Path

from atoll.config import load_enabled_islands


@dataclass(frozen=True, slots=True)
class CleanOptions:
    """User-facing options for selecting which Atoll outputs to remove."""

    root: Path
    cache: bool = False
    artifacts: bool = False
    all_outputs: bool = False


@dataclass(frozen=True, slots=True)
class CleanCommandResult:
    """Paths that no longer exist after a clean operation completes."""

    removed: tuple[Path, ...]


def execute_clean(options: CleanOptions) -> CleanCommandResult:
    """Remove selected Atoll cache/build outputs and compiled artifacts.

    With no specific flag, the default is to clear cache/build state. Artifact
    removal is opt-in through `artifacts` or `all_outputs` because compiled files
    may be useful for later inspection.
    """
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
    artifact_dir = root / ".atoll" / "artifacts"
    for island in load_enabled_islands(root):
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            artifacts.update(artifact_dir.rglob(f"{island.sidecar_path.stem}*{suffix}"))
            artifacts.update(
                island.sidecar_path.parent.glob(f"{island.sidecar_path.stem}*{suffix}")
            )
    return tuple(sorted(artifacts))
