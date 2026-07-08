"""Implementation of the `atoll disable` command."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from atoll.config import disable_island, load_enabled_islands
from atoll.generation.shim import ShimEdit, remove_shim
from atoll.models import EnabledIslandConfig


@dataclass(frozen=True, slots=True)
class DisableOptions:
    """User-facing options for disabling an Atoll island."""

    root: Path
    module_name: str
    delete_sidecar: bool = False
    dry_run: bool = False


@dataclass(frozen=True, slots=True)
class DisableCommandResult:
    """Files and generated text from a disable operation."""

    island: EnabledIslandConfig
    shim_edit: ShimEdit
    applied: bool


def execute_disable(options: DisableOptions) -> DisableCommandResult:
    """Remove a managed shim and mark an Atoll island disabled."""
    root = options.root.resolve()
    island = _find_island(load_enabled_islands(root), options.module_name)
    source_text = island.source_path.read_text(encoding="utf-8")
    shim_edit = remove_shim(source_text, island)
    if not options.dry_run:
        island.source_path.write_text(shim_edit.new_text, encoding="utf-8")
        disable_island(root, options.module_name)
        if options.delete_sidecar and island.sidecar_path.exists():
            island.sidecar_path.unlink()
    return DisableCommandResult(island=island, shim_edit=shim_edit, applied=not options.dry_run)


def _find_island(
    islands: tuple[EnabledIslandConfig, ...],
    module_name: str,
) -> EnabledIslandConfig:
    for island in islands:
        if island.source_module == module_name:
            return island
    raise ValueError(f"Atoll island is not configured: {module_name}")
