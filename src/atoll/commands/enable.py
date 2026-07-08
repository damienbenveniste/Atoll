"""Implementation of the `atoll enable` command."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.clustering import enrich_island_analysis
from atoll.config import upsert_enabled_island
from atoll.generation.shim import ShimEdit, insert_or_replace_shim
from atoll.generation.sidecar import default_sidecar_module, expected_sidecar_path, generate_sidecar
from atoll.models import EnabledIslandConfig, ModuleId, SidecarGeneration
from atoll.project import discover_project


@dataclass(frozen=True, slots=True)
class EnableOptions:
    """User-facing options for enabling an Atoll island."""

    root: Path
    module_name: str
    symbols: tuple[str, ...]
    sidecar_module: str | None = None
    dry_run: bool = False
    yes: bool = False


@dataclass(frozen=True, slots=True)
class EnableCommandResult:
    """Files and generated text from an enable operation."""

    island: EnabledIslandConfig
    sidecar: SidecarGeneration
    shim_edit: ShimEdit
    config_path: Path
    applied: bool


def execute_enable(options: EnableOptions) -> EnableCommandResult:
    """Generate a sidecar, insert a managed shim, and update `.atoll.toml`."""
    project = discover_project(options.root)
    module = _find_module(project.modules, options.module_name)
    scan = enrich_island_analysis(scan_module(module))
    sidecar_module = options.sidecar_module or default_sidecar_module(options.module_name)
    island = EnabledIslandConfig(
        source_module=options.module_name,
        source_path=module.path,
        sidecar_module=sidecar_module,
        sidecar_path=expected_sidecar_path(scan, sidecar_module),
        symbols=options.symbols,
        enabled=True,
    )
    sidecar = generate_sidecar(scan, island)
    source_text = module.path.read_text(encoding="utf-8")
    shim_edit = insert_or_replace_shim(source_text, island)
    config_path = project.config.root / ".atoll.toml"
    if not options.dry_run:
        upsert_enabled_island(project.config.root, island)
        island.sidecar_path.write_text(sidecar.source_text, encoding="utf-8")
        island.source_path.write_text(shim_edit.new_text, encoding="utf-8")
    return EnableCommandResult(
        island=island,
        sidecar=sidecar,
        shim_edit=shim_edit,
        config_path=config_path,
        applied=not options.dry_run,
    )


def _find_module(modules: tuple[ModuleId, ...], module_name: str) -> ModuleId:
    for module in modules:
        if module.name == module_name:
            return module
    raise ValueError(f"module not found under configured source roots: {module_name}")
