"""Implementation of the `atoll disable` command.

Disabling removes the managed shim from the source module and marks the island
disabled in configuration. Generated sidecar source is retained unless the caller
explicitly requests deletion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from atoll.config import disable_island, load_enabled_islands
from atoll.generation.shim import ShimEdit, remove_shim
from atoll.models import EnabledIslandConfig


@dataclass(frozen=True, slots=True)
class DisableOptions:
    """User-facing options for disabling one configured Atoll island.

    Attributes:
        root: Root directory of the target Python project.
        module_name: Importable module name used to restrict the command.
        delete_sidecar: Whether disabling also removes the generated sidecar file.
        dry_run: Whether changes are planned and reported without filesystem writes.
    """

    root: Path
    module_name: str
    delete_sidecar: bool = False
    dry_run: bool = False


@dataclass(frozen=True, slots=True)
class DisableCommandResult:
    """Prepared or applied edits from disabling one island.

    Attributes:
        island: Enabled island affected by the command.
        shim_edit: Managed source transformation planned or applied by the command.
        applied: Whether the planned filesystem changes were written.
    """

    island: EnabledIslandConfig
    shim_edit: ShimEdit
    applied: bool


def execute_disable(options: DisableOptions) -> DisableCommandResult:
    """Remove a managed shim and mark an Atoll island disabled.

    In dry-run mode the source text diff is prepared but no files or
    configuration are changed. A missing configured island raises `ValueError`.

    Args:
        options: Validated command options supplied by the CLI layer.

    Returns:
        DisableCommandResult: Planned or applied island, shim, and configuration changes.
    """
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
