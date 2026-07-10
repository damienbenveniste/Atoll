"""Implementation of the `atoll generate` command.

Generate refreshes sidecar source for configured islands. In check mode it only
reports missing or stale sidecars, allowing CI to verify generated files without
changing the working tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.clustering import enrich_island_analysis
from atoll.generation.sidecar import generate_sidecar, write_sidecar
from atoll.models import EnabledIslandConfig, ModuleId, SidecarGeneration
from atoll.project import discover_project


@dataclass(frozen=True, slots=True)
class GenerateOptions:
    """User-facing options for sidecar generation or stale checks.

    Attributes:
        root: Root directory of the target Python project.
        module_name: Importable module name used to restrict the command.
        check: Whether generation should report drift without writing files.
    """

    root: Path
    module_name: str | None = None
    check: bool = False


@dataclass(frozen=True, slots=True)
class GenerateCommandResult:
    """Generated sidecars plus stale or missing paths found in check mode.

    Attributes:
        generated: Sidecars generated for the selected islands.
        stale_paths: Generated sidecar paths whose current content differs.
    """

    generated: tuple[SidecarGeneration, ...]
    stale_paths: tuple[Path, ...]


def execute_generate(options: GenerateOptions) -> GenerateCommandResult:
    """Generate or check all enabled sidecars.

    `module_name` narrows work to one configured source module. When `check` is
    false, generated source is written to disk and `stale_paths` is empty because
    the command has already reconciled the files.

    Args:
        options: Validated command options supplied by the CLI layer.

    Returns:
        GenerateCommandResult: Generated sidecars and paths that differ from expected output.
    """
    project = discover_project(options.root)
    generations = tuple(
        _generate_for_island(island)
        for island in project.config.islands
        if island.enabled
        and (options.module_name is None or island.source_module == options.module_name)
    )
    stale_paths = tuple(
        generation.config.sidecar_path
        for generation in generations
        if generation.config.sidecar_path.exists()
        and generation.config.sidecar_path.read_text(encoding="utf-8") != generation.source_text
    )
    missing_paths = tuple(
        generation.config.sidecar_path
        for generation in generations
        if not generation.config.sidecar_path.exists()
    )
    if not options.check:
        for generation in generations:
            write_sidecar(generation)
    return GenerateCommandResult(
        generated=generations,
        stale_paths=(*stale_paths, *missing_paths) if options.check else (),
    )


def _generate_for_island(island: EnabledIslandConfig) -> SidecarGeneration:
    module = ModuleId(name=island.source_module, path=island.source_path)
    scan = enrich_island_analysis(scan_module(module))
    return generate_sidecar(scan, island)
