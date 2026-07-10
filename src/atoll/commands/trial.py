"""Compile and validate selected typed regions without modifying checkout sources.

Trial keeps the scan-candidate selection UX but delegates compilation, backend
selection, wheel overlay, native artifact verification, and caching to the same
source-clean package pipeline used by ``atoll compile``. Its temporary install
payload exists only to run optional pytest commands and is removed by default.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.clustering import enrich_island_analysis
from atoll.commands.package import (
    PackageCommandResult,
    PackageOptions,
    PackageProgress,
    execute_package,
)
from atoll.models import (
    BindingTarget,
    EnabledIslandConfig,
    IslandCandidate,
    ModuleId,
    ModuleScan,
    PytestRunResult,
    SymbolId,
)
from atoll.project import DiscoveredProject, discover_project
from atoll.runtime.test_runner import run_pytest_command


@dataclass(frozen=True, slots=True)
class TrialOptions:
    """User-facing options for selecting and validating compiled regions.

    Attributes:
        root: Root directory of the target Python project.
        candidate: Explicit candidate selector supplied by the user.
        top: Maximum number of highest-scoring candidates to select.
        test_command: Optional target-project semantic test command.
        benchmark_command: Optional command used for paired performance measurements.
        keep_temp: Whether trial artifacts remain after completion.
        require_compiled: Whether interpreted fallback fails verification.
        progress: Optional progress callback used by long-running packaging work.
    """

    root: Path
    candidate: str | None = None
    top: int | None = None
    test_command: str | None = None
    benchmark_command: str | None = None
    keep_temp: bool = False
    require_compiled: bool = True
    progress: PackageProgress | None = None


@dataclass(frozen=True, slots=True)
class TrialCommandResult:
    """Outcome of a source-clean typed-region trial.

    ``artifact_root`` contains the temporary wheel and install payload only when
    ``keep_temp`` is enabled. ``overlay_root`` and ``enabled`` remain read-only
    compatibility views for callers written against the legacy overlay trial.

    Attributes:
        success: Whether the represented operation completed successfully.
        artifact_root: Retained trial directory containing generated artifacts and reports.
        selected: Stable symbol identities selected for the trial.
        compiled_bindings: Source bindings successfully provided by compiled regions.
        wheel_path: Source-clean wheel path, when produced.
        test_exit_code: Compatibility semantic-test exit code, when run.
        benchmark_exit_code: Compatibility benchmark process exit code, when run.
        error: User-facing failure text, or `None` on success.
        selections: Module and symbol selections made by the trial.
        package_result: Underlying source-clean package result, when compilation was attempted.
        test_result: Compatibility semantic-test result, when run.
        benchmark_result: Compatibility benchmark process result, when run.
    """

    success: bool
    artifact_root: Path
    selected: tuple[SymbolId, ...]
    compiled_bindings: tuple[BindingTarget, ...]
    wheel_path: Path | None
    test_exit_code: int | None
    benchmark_exit_code: int | None
    error: str | None = None
    selections: tuple[TrialSelection, ...] = ()
    package_result: PackageCommandResult | None = None
    test_result: PytestRunResult | None = None
    benchmark_result: PytestRunResult | None = None

    @property
    def overlay_root(self) -> Path:
        """Return the temporary artifact root under the legacy property name.

        Returns:
            Path: Compatibility alias for the trial artifact root.
        """
        return self.artifact_root

    @property
    def enabled(self) -> tuple[EnabledIslandConfig, ...]:
        """Return an empty legacy sidecar view; trial no longer enables islands.

        Returns:
            tuple[EnabledIslandConfig, ...]: Always empty; source-clean trials no longer enable
                persistent islands.
        """
        return ()


def execute_trial(options: TrialOptions) -> TrialCommandResult:
    """Compile selected typed members and run optional pytest gates.

    All compilation and routing verification is delegated to
    :func:`atoll.commands.package.execute_package`. Errors are normalized into a
    result so the CLI can report the temporary artifact location when retained.

    Args:
        options: Validated command options supplied by the CLI layer.

    Returns:
        TrialCommandResult: Compilation, test, benchmark, and retained-artifact evidence for the
            trial.
    """
    try:
        project = discover_project(options.root)
        selections = _select_islands(project, options)
    except Exception as error:
        return _failed_result(error=repr(error))
    if not selections:
        return _failed_result(error="no trial candidates selected")

    selected = _selected_member_ids(selections)
    artifact_root = Path(tempfile.mkdtemp(prefix="atoll-trial-"))
    package: PackageCommandResult | None = None
    try:
        package = execute_package(
            PackageOptions(
                root=project.config.root,
                output_dir=artifact_root / "dist",
                keep_install_tree=True,
                progress=options.progress,
                selected_members=selected,
                cache_dir=artifact_root / "cache",
                run_quality_gates=False,
            )
        )
        if not package.success:
            return _finalize(
                TrialCommandResult(
                    success=False,
                    artifact_root=artifact_root,
                    selected=selected,
                    compiled_bindings=package.compiled_bindings,
                    wheel_path=None,
                    test_exit_code=None,
                    benchmark_exit_code=None,
                    error=package.error or package.build.stderr,
                    selections=selections,
                    package_result=package,
                ),
                options,
            )
        test_result = _run_test_command(
            options.test_command,
            project.config.root,
            package.install_root,
            require_compiled=options.require_compiled,
        )
        benchmark_result = None
        if test_result is None or test_result.success:
            benchmark_result = _run_test_command(
                options.benchmark_command,
                project.config.root,
                package.install_root,
                require_compiled=options.require_compiled,
            )
        return _finalize(
            TrialCommandResult(
                success=(test_result is None or test_result.success)
                and (benchmark_result is None or benchmark_result.success),
                artifact_root=artifact_root,
                selected=selected,
                compiled_bindings=package.compiled_bindings,
                wheel_path=package.wheel_path,
                test_exit_code=(test_result.exit_code if test_result is not None else None),
                benchmark_exit_code=(
                    benchmark_result.exit_code if benchmark_result is not None else None
                ),
                selections=selections,
                package_result=package,
                test_result=test_result,
                benchmark_result=benchmark_result,
            ),
            options,
        )
    except Exception as error:
        return _finalize(
            TrialCommandResult(
                success=False,
                artifact_root=artifact_root,
                selected=selected,
                compiled_bindings=(package.compiled_bindings if package is not None else ()),
                wheel_path=(package.wheel_path if package is not None else None),
                test_exit_code=None,
                benchmark_exit_code=None,
                error=repr(error),
                selections=selections,
                package_result=package,
            ),
            options,
        )


@dataclass(frozen=True, slots=True)
class TrialSelection:
    """One scan candidate retained as a typed-region trial selection.

    Attributes:
        module: Discovered module containing the selected candidate symbols.
        symbols: Qualified symbol names selected from that module.
    """

    module: ModuleId
    symbols: tuple[str, ...]


def _select_islands(
    project: DiscoveredProject,
    options: TrialOptions,
) -> tuple[TrialSelection, ...]:
    if options.candidate is not None:
        module_name, symbols = _parse_candidate(options.candidate)
        module = _find_module(project.modules, module_name)
        _validate_explicit_symbols(module, symbols)
        return (TrialSelection(module=module, symbols=symbols),)
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


def _selected_member_ids(selections: tuple[TrialSelection, ...]) -> tuple[SymbolId, ...]:
    """Return stable, de-duplicated member identities in candidate order.

    Args:
        selections: Selected regions and backend assessments.

    Returns:
        tuple[SymbolId, ...]: Stable IDs selected across all trial candidates.
    """
    members = (
        SymbolId(module=selection.module.name, qualname=symbol)
        for selection in selections
        for symbol in selection.symbols
    )
    return tuple(dict.fromkeys(members))


def _run_test_command(
    command: str | None,
    root: Path,
    install_root: Path,
    *,
    require_compiled: bool,
) -> PytestRunResult | None:
    if command is None:
        return None
    return run_pytest_command(
        command,
        root=root,
        source_roots=(install_root,),
        require_compiled=require_compiled,
    )


def _finalize(result: TrialCommandResult, options: TrialOptions) -> TrialCommandResult:
    if not options.keep_temp and result.artifact_root.exists():
        shutil.rmtree(result.artifact_root)
    return result


def _failed_result(*, error: str) -> TrialCommandResult:
    return TrialCommandResult(
        success=False,
        artifact_root=Path(),
        selected=(),
        compiled_bindings=(),
        wheel_path=None,
        test_exit_code=None,
        benchmark_exit_code=None,
        error=error,
    )


def _scan_candidate_module(module: ModuleId) -> ModuleScan:
    return enrich_island_analysis(scan_module(module))


def _selection_from_candidate(module: ModuleId, candidate: IslandCandidate) -> TrialSelection:
    return TrialSelection(
        module=module,
        symbols=tuple(symbol.qualname for symbol in candidate.symbols),
    )


def _parse_candidate(value: str) -> tuple[str, tuple[str, ...]]:
    module_name, separator, symbols_text = value.partition("::")
    symbols = tuple(symbol.strip() for symbol in symbols_text.split(",") if symbol.strip())
    if not separator or not module_name or not symbols:
        raise ValueError("candidate must look like app.module::symbol,helper")
    return module_name, symbols


def _validate_explicit_symbols(module: ModuleId, symbols: tuple[str, ...]) -> None:
    """Keep the legacy candidate contract limited to top-level functions.

    Args:
        module: Scanned module or syntax module being processed.
        symbols: Source symbols processed in deterministic order.

    Raises:
        ValueError: If a requested symbol is not a top-level function in the selected module.
    """
    scan = _scan_candidate_module(module)
    functions = frozenset(
        symbol.id.qualname
        for symbol in scan.symbols
        if symbol.kind == "function" and "." not in symbol.id.qualname
    )
    unsupported = tuple(symbol for symbol in symbols if symbol not in functions)
    if unsupported:
        raise ValueError(
            "trial candidate symbols must be top-level functions in "
            f"{module.name}: {', '.join(unsupported)}"
        )


def _find_module(modules: tuple[ModuleId, ...], module_name: str) -> ModuleId:
    for module in modules:
        if module.name == module_name:
            return module
    raise ValueError(f"module not found under configured source roots: {module_name}")
