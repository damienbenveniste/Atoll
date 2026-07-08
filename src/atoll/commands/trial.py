"""Implementation of the `atoll trial` command."""

from __future__ import annotations

import importlib
import os
import shlex
import shutil
import sys
import tempfile
from contextlib import chdir
from dataclasses import dataclass
from pathlib import Path

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.clustering import enrich_island_analysis
from atoll.backends.mypyc import build_sidecars
from atoll.config import write_atoll_config
from atoll.generation.shim import insert_or_replace_shim
from atoll.generation.sidecar import default_sidecar_module, expected_sidecar_path, generate_sidecar
from atoll.models import (
    EnabledIslandConfig,
    IslandCandidate,
    ModuleId,
    ModuleScan,
    ProjectConfig,
    VerifyResult,
)
from atoll.project import DiscoveredProject, discover_project
from atoll.runtime.verify import verify_islands


@dataclass(frozen=True, slots=True)
class TrialOptions:
    """User-facing options for overlay trial mode."""

    root: Path
    candidate: str | None = None
    top: int | None = None
    test_command: str | None = None
    benchmark_command: str | None = None
    keep_temp: bool = False
    require_compiled: bool = True


@dataclass(frozen=True, slots=True)
class TrialCommandResult:
    """Outcome of a trial run."""

    success: bool
    overlay_root: Path
    enabled: tuple[EnabledIslandConfig, ...]
    test_exit_code: int | None
    benchmark_exit_code: int | None
    error: str | None = None


def execute_trial(options: TrialOptions) -> TrialCommandResult:
    """Run selected Atoll islands in a temporary compiled overlay."""
    try:
        project = discover_project(options.root)
        selections = _select_islands(project, options)
    except Exception as error:
        return TrialCommandResult(
            success=False,
            overlay_root=Path(),
            enabled=(),
            test_exit_code=None,
            benchmark_exit_code=None,
            error=repr(error),
        )
    if not selections:
        return TrialCommandResult(
            success=False,
            overlay_root=Path(),
            enabled=(),
            test_exit_code=None,
            benchmark_exit_code=None,
            error="no trial candidates selected",
        )
    overlay_root = Path(tempfile.mkdtemp(prefix="atoll-trial-"))
    try:
        overlay_source_roots = _copy_source_roots(project, overlay_root)
        overlay_islands = tuple(
            _prepare_overlay_island(selection, project, overlay_source_roots)
            for selection in selections
        )
        write_atoll_config(overlay_root, overlay_islands)
        compile_attempt = build_sidecars(
            tuple(island.sidecar_path for island in overlay_islands),
            project_root=overlay_root,
            build_dir=overlay_root / ".atoll" / "build",
            source_roots=overlay_source_roots,
        )
        if not compile_attempt.success:
            return _trial_result(
                TrialCommandResult(
                    success=False,
                    overlay_root=overlay_root,
                    enabled=overlay_islands,
                    test_exit_code=None,
                    benchmark_exit_code=None,
                    error=compile_attempt.stderr,
                ),
                options,
            )
        verify_config = ProjectConfig(
            root=overlay_root,
            source_roots=overlay_source_roots,
            backend="mypyc",
            cache_dir=overlay_root / ".atoll" / "cache",
            report_dir=overlay_root / ".atoll",
            islands=overlay_islands,
        )
        verify_results = verify_islands(verify_config, require_compiled=options.require_compiled)
        verify_error = _verify_error(verify_results)
        if verify_error is not None:
            return _trial_result(
                TrialCommandResult(
                    success=False,
                    overlay_root=overlay_root,
                    enabled=overlay_islands,
                    test_exit_code=None,
                    benchmark_exit_code=None,
                    error=verify_error,
                ),
                options,
            )
        test_exit = _run_test_command(options.test_command, options.root, overlay_source_roots)
        benchmark_exit = None
        if test_exit in (None, 0):
            benchmark_exit = _run_test_command(
                options.benchmark_command,
                options.root,
                overlay_source_roots,
            )
        return _trial_result(
            TrialCommandResult(
                success=(test_exit is None or test_exit == 0)
                and (benchmark_exit is None or benchmark_exit == 0),
                overlay_root=overlay_root,
                enabled=overlay_islands,
                test_exit_code=test_exit,
                benchmark_exit_code=benchmark_exit,
            ),
            options,
        )
    except Exception as error:
        return _trial_result(
            TrialCommandResult(
                success=False,
                overlay_root=overlay_root,
                enabled=(),
                test_exit_code=None,
                benchmark_exit_code=None,
                error=repr(error),
            ),
            options,
        )


@dataclass(frozen=True, slots=True)
class _Selection:
    module: ModuleId
    symbols: tuple[str, ...]


def _select_islands(project: DiscoveredProject, options: TrialOptions) -> tuple[_Selection, ...]:
    if options.candidate is not None:
        module_name, symbols = _parse_candidate(options.candidate)
        return (_Selection(module=_find_module(project.modules, module_name), symbols=symbols),)
    limit = options.top if options.top is not None else 1
    candidates = sorted(
        (
            (scan.module, candidate)
            for scan in (_scan_candidate_module(module) for module in project.modules)
            for candidate in scan.island_candidates
        ),
        key=lambda item: item[1].score,
        reverse=True,
    )
    return tuple(
        _selection_from_candidate(module, candidate) for module, candidate in candidates[:limit]
    )


def _copy_source_roots(
    project: DiscoveredProject,
    overlay_root: Path,
) -> tuple[Path, ...]:
    overlay_roots: list[Path] = []
    for source_root in project.config.source_roots:
        destination = overlay_root / _relative_source_root(project.config.root, source_root)
        shutil.copytree(source_root, destination, ignore=_copy_ignore)
        overlay_roots.append(destination)
    return tuple(overlay_roots)


def _prepare_overlay_island(
    selection: _Selection,
    project: DiscoveredProject,
    overlay_source_roots: tuple[Path, ...],
) -> EnabledIslandConfig:
    overlay_module = _overlay_module(selection.module, project, overlay_source_roots)
    scan = enrich_island_analysis(scan_module(overlay_module))
    sidecar_module = default_sidecar_module(selection.module.name)
    island = EnabledIslandConfig(
        source_module=selection.module.name,
        source_path=overlay_module.path,
        sidecar_module=sidecar_module,
        sidecar_path=expected_sidecar_path(scan, sidecar_module),
        symbols=selection.symbols,
    )
    generation = generate_sidecar(scan, island)
    island.sidecar_path.write_text(generation.source_text, encoding="utf-8")
    source_text = island.source_path.read_text(encoding="utf-8")
    island.source_path.write_text(
        insert_or_replace_shim(source_text, island).new_text, encoding="utf-8"
    )
    return island


def _run_test_command(
    command: str | None,
    root: Path,
    overlay_source_roots: tuple[Path, ...],
) -> int | None:
    if command is None:
        return None
    args = _pytest_args(command)
    original_path = list(sys.path)
    original_pythonpath = os.environ.get("PYTHONPATH")
    sys.path[:0] = [str(path) for path in overlay_source_roots]
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [
            *(str(path) for path in overlay_source_roots),
            *(original_pythonpath or "").split(os.pathsep),
        ]
    )
    try:
        with chdir(root):
            pytest = importlib.import_module("pytest")
            return int(pytest.main(args))
    finally:
        sys.path[:] = original_path
        if original_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = original_pythonpath


def _pytest_args(command: str) -> list[str]:
    parts = shlex.split(command)
    if parts[:1] == ["pytest"]:
        return parts[1:]
    if parts[:3] == ["python", "-m", "pytest"]:
        return parts[3:]
    raise ValueError("trial currently supports pytest commands only")


def _trial_result(result: TrialCommandResult, options: TrialOptions) -> TrialCommandResult:
    if not options.keep_temp and result.overlay_root.exists():
        shutil.rmtree(result.overlay_root)
    return result


def _scan_candidate_module(module: ModuleId) -> ModuleScan:
    return enrich_island_analysis(scan_module(module))


def _selection_from_candidate(module: ModuleId, candidate: IslandCandidate) -> _Selection:
    return _Selection(module=module, symbols=tuple(symbol.qualname for symbol in candidate.symbols))


def _parse_candidate(value: str) -> tuple[str, tuple[str, ...]]:
    module_name, separator, symbols_text = value.partition("::")
    symbols = tuple(symbol.strip() for symbol in symbols_text.split(",") if symbol.strip())
    if not separator or not module_name or not symbols:
        raise ValueError("candidate must look like app.module::symbol,helper")
    return module_name, symbols


def _find_module(modules: tuple[ModuleId, ...], module_name: str) -> ModuleId:
    for module in modules:
        if module.name == module_name:
            return module
    raise ValueError(f"module not found under configured source roots: {module_name}")


def _overlay_module(
    module: ModuleId,
    project: DiscoveredProject,
    overlay_source_roots: tuple[Path, ...],
) -> ModuleId:
    for index, source_root in enumerate(project.config.source_roots):
        try:
            relative = module.path.relative_to(source_root)
        except ValueError:
            continue
        return ModuleId(name=module.name, path=overlay_source_roots[index] / relative)
    raise ValueError(f"module is outside configured source roots: {module.name}")


def _relative_source_root(root: Path, source_root: Path) -> Path:
    try:
        return source_root.relative_to(root)
    except ValueError:
        return Path(f"source_{abs(hash(source_root))}")


def _copy_ignore(_directory: str, names: list[str]) -> set[str]:
    ignored = {
        ".atoll",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "build",
        "dist",
    }
    return {name for name in names if name in ignored or name.endswith((".so", ".pyd"))}


def _verify_error(results: tuple[VerifyResult, ...]) -> str | None:
    for result in results:
        error = getattr(result, "error", None)
        if isinstance(error, str):
            return error
    return None
