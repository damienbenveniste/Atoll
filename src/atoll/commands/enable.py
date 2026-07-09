"""Implementation of the `atoll enable` command.

Enable prepares an Atoll island by generating a sidecar, inserting or replacing
the managed source shim, and recording the island in configuration. The module
also supports enabling every current scan candidate in one operation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.clustering import enrich_island_analysis
from atoll.config import load_enabled_islands, upsert_enabled_island, write_atoll_config
from atoll.generation.shim import ShimEdit, insert_or_replace_shim
from atoll.generation.sidecar import default_sidecar_module, expected_sidecar_path, generate_sidecar
from atoll.models import EnabledIslandConfig, ModuleId, ModuleScan, SidecarGeneration
from atoll.project import DiscoveredProject, discover_project


@dataclass(frozen=True, slots=True)
class EnableOptions:
    """User-facing options for enabling one explicit Atoll island."""

    root: Path
    module_name: str
    symbols: tuple[str, ...]
    sidecar_module: str | None = None
    dry_run: bool = False
    yes: bool = False


@dataclass(frozen=True, slots=True)
class EnableAllOptions:
    """User-facing options for enabling all current scan candidates."""

    root: Path
    module_name: str | None = None
    dry_run: bool = False
    yes: bool = False


@dataclass(frozen=True, slots=True)
class EnableCommandResult:
    """Prepared or applied files from enabling one island.

    The result includes both generated sidecar source and the managed shim edit
    so dry-run callers can show exactly what would change before writing.
    """

    island: EnabledIslandConfig
    sidecar: SidecarGeneration
    shim_edit: ShimEdit
    config_path: Path
    applied: bool


@dataclass(frozen=True, slots=True)
class EnableAllCommandResult:
    """Prepared or applied results from enabling all selected candidates."""

    enabled: tuple[EnableCommandResult, ...]
    applied: bool

    @property
    def symbol_count(self) -> int:
        """Return the number of exported symbols enabled across all modules."""
        return sum(len(result.island.symbols) for result in self.enabled)


def execute_enable(options: EnableOptions) -> EnableCommandResult:
    """Generate a sidecar, insert a managed shim, and update `.atoll.toml`.

    Dry-run mode performs discovery, scanning, sidecar rendering, and shim diff
    generation without mutating the source module, sidecar path, or config file.
    """
    project = discover_project(options.root)
    module = _find_module(project.modules, options.module_name)
    scan = enrich_island_analysis(scan_module(module))
    result = _prepare_enable(
        root=project.config.root,
        scan=scan,
        symbols=options.symbols,
        sidecar_module=options.sidecar_module,
    )
    if not options.dry_run:
        upsert_enabled_island(project.config.root, result.island)
        _write_enable_result(result)
        return replace(result, applied=True)
    return result


def execute_enable_all(options: EnableAllOptions) -> EnableAllCommandResult:
    """Enable every scan candidate, grouped into one sidecar per source module.

    Existing islands for selected modules are replaced atomically in the written
    config, while islands for other modules are preserved.
    """
    project = discover_project(options.root)
    scans = _candidate_scans(project, options.module_name)
    results = tuple(
        _prepare_enable(
            root=project.config.root,
            scan=scan,
            symbols=_candidate_symbols(scan),
            sidecar_module=None,
        )
        for scan in scans
        if scan.island_candidates
    )
    if options.module_name is not None and not results:
        raise ValueError(f"module has no candidate islands: {options.module_name}")
    if not results:
        raise ValueError("scan found no candidate islands to enable")
    if not options.dry_run:
        _write_enable_all(project.config.root, results)
        return EnableAllCommandResult(
            enabled=tuple(replace(result, applied=True) for result in results),
            applied=True,
        )
    return EnableAllCommandResult(enabled=results, applied=False)


def _prepare_enable(
    *,
    root: Path,
    scan: ModuleScan,
    symbols: tuple[str, ...],
    sidecar_module: str | None,
) -> EnableCommandResult:
    resolved_sidecar_module = sidecar_module or default_sidecar_module(scan.module.name)
    island = EnabledIslandConfig(
        source_module=scan.module.name,
        source_path=scan.module.path,
        sidecar_module=resolved_sidecar_module,
        sidecar_path=expected_sidecar_path(root, resolved_sidecar_module),
        symbols=symbols,
        enabled=True,
    )
    sidecar = generate_sidecar(scan, island)
    source_text = scan.module.path.read_text(encoding="utf-8")
    shim_edit = insert_or_replace_shim(source_text, island)
    return EnableCommandResult(
        island=island,
        sidecar=sidecar,
        shim_edit=shim_edit,
        config_path=root / ".atoll.toml",
        applied=False,
    )


def _write_enable_result(result: EnableCommandResult) -> None:
    result.island.sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    result.island.sidecar_path.write_text(result.sidecar.source_text, encoding="utf-8")
    result.island.source_path.write_text(result.shim_edit.new_text, encoding="utf-8")


def _write_enable_all(root: Path, results: tuple[EnableCommandResult, ...]) -> None:
    enabled_modules = {result.island.source_module for result in results}
    islands = [
        island
        for island in load_enabled_islands(root)
        if island.source_module not in enabled_modules
    ]
    islands.extend(result.island for result in results)
    write_atoll_config(root, tuple(sorted(islands, key=lambda island: island.source_module)))
    for result in results:
        _write_enable_result(result)


def _candidate_scans(
    project: DiscoveredProject,
    module_name: str | None,
) -> tuple[ModuleScan, ...]:
    modules = (
        (_find_module(project.modules, module_name),)
        if module_name is not None
        else project.modules
    )
    return tuple(enrich_island_analysis(scan_module(module)) for module in modules)


def _candidate_symbols(scan: ModuleScan) -> tuple[str, ...]:
    candidates = {symbol for candidate in scan.island_candidates for symbol in candidate.symbols}
    return tuple(
        symbol.id.qualname
        for symbol in scan.symbols
        if symbol.id in candidates and symbol.kind == "function"
    )


def _find_module(modules: tuple[ModuleId, ...], module_name: str) -> ModuleId:
    for module in modules:
        if module.name == module_name:
            return module
    raise ValueError(f"module not found under configured source roots: {module_name}")
