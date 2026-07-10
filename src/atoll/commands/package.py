"""Build installable Atoll artifacts without modifying source files."""

from __future__ import annotations

import ast
import base64
import csv
import hashlib
import importlib.machinery
import io
import json
import re
import shutil
import sys
import textwrap
import time
import tomllib
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import cast

from packaging import tags

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.clustering import enrich_island_analysis
from atoll.analysis.native_readiness import NativeReadiness, analyze_native_readiness
from atoll.backends.base import CompilerBackend
from atoll.backends.cython import CYTHON_BACKEND
from atoll.backends.mypyc import MYPYC_BACKEND, build_sidecars
from atoll.generation.region_shim import RegionShimConfig, insert_or_replace_region_shim
from atoll.generation.shim import insert_or_replace_shim, remove_shim
from atoll.generation.sidecar import (
    SIDECAR_GENERATOR_VERSION,
    default_sidecar_module,
    expected_sidecar_path,
    generate_sidecar,
)
from atoll.generation.typed_region import (
    TypedRegionGeneration,
    TypedRegionGenerationOptions,
    generate_typed_method_region,
)
from atoll.models import (
    ArtifactRecord,
    Backend,
    BackendAssessment,
    BackendCompileContext,
    BackendCompileResult,
    BackendLoweringRequest,
    BindingTarget,
    Blocker,
    CompilationUnit,
    CompileAttempt,
    CompiledRegionVariant,
    CompilePhaseTiming,
    EnabledIslandConfig,
    LoweringDecision,
    ModuleId,
    ModuleScan,
    RegionSpecialization,
    SymbolId,
    TypedRegion,
)
from atoll.project import DiscoveredProject, discover_project

PackageProgress = Callable[[str], None]

_COMPILER_BACKENDS: dict[Backend, CompilerBackend] = {
    "mypyc": MYPYC_BACKEND,
    "cython": CYTHON_BACKEND,
}

_GENERATED_DIR_NAMES = frozenset(
    {
        ".atoll",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "build",
        "dist",
        "site-packages",
    }
)
_COMPILE_CACHE_VERSION = 2
_CACHE_INPUT_SUFFIXES = frozenset({".py", ".pyi", ".toml"})
_CACHE_INPUT_NAMES = frozenset({"py.typed"})


@dataclass(frozen=True, slots=True)
class PackageOptions:
    """User-facing options for building an installable Atoll artifact."""

    root: Path
    module_name: str | None = None
    output_dir: Path | None = None
    keep_install_tree: bool = False
    progress: PackageProgress | None = None


@dataclass(frozen=True, slots=True)
class PackageCommandResult:
    """Result from building a source-clean Atoll package artifact."""

    success: bool
    project_root: Path
    output_dir: Path
    install_root: Path
    wheel_path: Path | None
    islands: tuple[EnabledIslandConfig, ...]
    build: CompileAttempt
    install_tree_kept: bool = False
    cleanup_removed: tuple[Path, ...] = ()
    cleanup_kept: tuple[Path, ...] = ()
    report_artifact_paths: tuple[Path, ...] = ()
    error: str | None = None
    skipped: tuple[PackageBuildFailure, ...] = ()
    preflight_skipped: tuple[PackagePreflightFailure, ...] = ()
    native_readiness: tuple[NativeReadiness, ...] = ()
    typed_regions: tuple[TypedRegion, ...] = ()
    compiled_regions: tuple[TypedRegion, ...] = ()
    compiled_bindings: tuple[BindingTarget, ...] = ()
    compiled_variants: tuple[CompiledRegionVariant, ...] = ()
    backend_assessments: tuple[BackendAssessment, ...] = ()
    artifact_records: tuple[ArtifactRecord, ...] = ()
    region_skipped: tuple[PackageRegionBuildFailure, ...] = ()


@dataclass(frozen=True, slots=True)
class PackageBuildFailure:
    """A selected island that could not be compiled into the artifact package."""

    island: EnabledIslandConfig
    build: CompileAttempt


@dataclass(frozen=True, slots=True)
class PackagePreflightFailure:
    """A selected module skipped before build because mypyc rejects module-level code."""

    scan: ModuleScan
    blockers: tuple[Blocker, ...]


@dataclass(frozen=True, slots=True)
class PackageRegionBuildFailure:
    """One typed region retained as interpreted after backend failure."""

    region: TypedRegion
    variant_id: str
    backend: Backend
    assessment: BackendAssessment
    build: CompileAttempt


@dataclass(frozen=True, slots=True)
class _ProjectMetadata:
    name: str
    version: str
    requires_python: str | None
    dependencies: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SelectedModule:
    scan: ModuleScan
    symbols: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SelectedTypedRegion:
    """One typed region and member subset selected for a backend variant."""

    scan: ModuleScan
    region: TypedRegion
    variant_id: str
    backend: Backend
    assessment: BackendAssessment
    members: tuple[SymbolId, ...]
    specialization: RegionSpecialization | None = None
    conditional_on_failure_of: str | None = None


@dataclass(frozen=True, slots=True)
class _PreparedTypedRegion:
    """Generated unit plus its staged runtime binding contract."""

    generation: TypedRegionGeneration
    assessment: BackendAssessment
    unit: CompilationUnit
    shim: RegionShimConfig
    fallback: _PreparedTypedRegion | None = None
    conditional_on_failure_of: str | None = None


@dataclass(frozen=True, slots=True)
class _TypedRegionBuildOutcome:
    """Per-region backend results aggregated for source-clean packaging."""

    successful: tuple[_PreparedTypedRegion, ...]
    build: CompileAttempt
    artifacts: tuple[ArtifactRecord, ...]
    skipped: tuple[PackageRegionBuildFailure, ...]


@dataclass(frozen=True, slots=True)
class _TypedRegionBuildContext:
    """Filesystem, cache, and progress boundaries shared by region builds."""

    build_root: Path
    staged_source_roots: tuple[Path, ...]
    cache_dir: Path
    progress: PackageProgress | None


@dataclass(frozen=True, slots=True)
class _StagedTypedRegionContext:
    """Copied source evidence shared by primary and fallback backend variants."""

    build_root: Path
    staged_source_root: Path
    module: ModuleId
    scan: ModuleScan
    region: TypedRegion


@dataclass(frozen=True, slots=True)
class _TypedRegionPackageContext:
    """Selected analysis evidence carried into source-clean region packaging."""

    selected: tuple[_SelectedTypedRegion, ...]
    typed_regions: tuple[TypedRegion, ...]
    preflight_skipped: tuple[PackagePreflightFailure, ...]
    native_readiness: tuple[NativeReadiness, ...]


@dataclass(frozen=True, slots=True)
class _PreparedModule:
    """Generated module after performance-worthiness filtering."""

    island: EnabledIslandConfig | None
    native_readiness: tuple[NativeReadiness, ...]


@dataclass(frozen=True, slots=True)
class _PackageBuildOutcome:
    successful: tuple[EnabledIslandConfig, ...]
    build: CompileAttempt
    skipped: tuple[PackageBuildFailure, ...]


@dataclass(frozen=True, slots=True)
class _PackageBuildContext:
    target_project: DiscoveredProject
    module_name: str | None
    project_root: Path
    source_roots: tuple[Path, ...]
    allow_partial: bool
    progress: PackageProgress | None


@dataclass(frozen=True, slots=True)
class _CompileCacheLookup:
    key: str
    hit: bool
    artifact_paths: tuple[Path, ...]
    successful_modules: tuple[str, ...]
    skipped_modules: tuple[str, ...]
    phase_timings: tuple[CompilePhaseTiming, ...]


def execute_package(options: PackageOptions) -> PackageCommandResult:
    """Build an install tree and wheel containing Atoll compiled islands."""
    _progress(options.progress, f"discovering project at {options.root.resolve()}")
    project = discover_project(options.root)
    _progress(
        options.progress,
        f"discovered {len(project.modules)} module(s); scan scope: {options.module_name or 'all'}",
    )
    scan_started = time.perf_counter()
    scans = _selected_scans(project, options.module_name)
    typed_regions = tuple(region for scan in scans for region in scan.typed_regions)
    _progress(options.progress, f"scanned {len(scans)} module(s) in {_duration(scan_started)}")
    selected = _selected_modules(scans)
    selected_typed_regions = _selected_typed_method_regions(scans)
    _progress_compile_selection(options.progress, selected, selected_typed_regions)
    if not selected and not selected_typed_regions:
        return _failed_result(
            project.config.root,
            options.output_dir,
            "scan found no candidate islands",
            typed_regions=typed_regions,
        )
    if not selected:
        return _execute_typed_region_package(
            options=options,
            project=project,
            context=_TypedRegionPackageContext(
                selected=selected_typed_regions,
                typed_regions=typed_regions,
                preflight_skipped=(),
                native_readiness=(),
            ),
        )

    output_dir = _resolve_output_dir(project.config.root, options.output_dir)
    build_root = output_dir / "build"
    install_root = output_dir / "install"
    _progress(options.progress, f"resetting temporary build roots in {output_dir}")
    _reset_dir(build_root)
    _reset_dir(install_root)

    copy_started = time.perf_counter()
    _progress(options.progress, "copying source roots into temporary build tree")
    staged_source_roots = _copy_source_roots(project, build_root)
    _progress(options.progress, f"copied source roots in {_duration(copy_started)}")
    sidecar_started = time.perf_counter()
    _progress(
        options.progress,
        (
            f"analyzing {sum(len(module.symbols) for module in selected)} generated "
            "candidate symbol(s) for native readiness"
        ),
    )
    prepared_modules = tuple(
        _prepare_staged_island(
            project=project,
            staged_source_roots=staged_source_roots,
            selected_module=selected_module,
        )
        for selected_module in selected
    )
    native_readiness = tuple(
        readiness
        for prepared_module in prepared_modules
        for readiness in prepared_module.native_readiness
    )
    islands = tuple(
        prepared_module.island
        for prepared_module in prepared_modules
        if prepared_module.island is not None
    )
    native_ready_symbols = sum(readiness.eligible for readiness in native_readiness)
    _progress(
        options.progress,
        (
            f"native readiness accepted {native_ready_symbols}/{len(native_readiness)} "
            f"symbol(s) across {len(islands)} module(s) in {_duration(sidecar_started)}"
        ),
    )
    if not islands:
        return _handle_no_native_islands(
            options=options,
            project=project,
            build_root=build_root,
            context=_TypedRegionPackageContext(
                selected=selected_typed_regions,
                typed_regions=typed_regions,
                preflight_skipped=(),
                native_readiness=native_readiness,
            ),
        )
    outcome = _build_package_islands(
        islands,
        _PackageBuildContext(
            target_project=project,
            module_name=options.module_name,
            project_root=build_root,
            source_roots=staged_source_roots,
            allow_partial=options.module_name is None,
            progress=options.progress,
        ),
    )
    if not outcome.build.success:
        _progress(options.progress, "build failed; keeping build tree for diagnostics")
        _remove_failed_wheels(project, output_dir)
        cleanup_removed = _remove_tree(install_root)
        return PackageCommandResult(
            success=False,
            project_root=project.config.root,
            output_dir=output_dir,
            install_root=install_root,
            wheel_path=None,
            islands=outcome.successful,
            build=outcome.build,
            error=outcome.build.stderr,
            cleanup_removed=cleanup_removed,
            cleanup_kept=(build_root,),
            skipped=outcome.skipped,
            preflight_skipped=(),
            native_readiness=native_readiness,
            typed_regions=typed_regions,
        )

    payload_started = time.perf_counter()
    _progress(options.progress, "placing compiled artifacts into install payload")
    _place_compiled_artifacts(outcome.successful, outcome.build.artifact_paths)
    report_artifact_paths = _source_clean_report_artifact_paths(
        project.config.root,
        outcome.build.artifact_paths,
    )
    _remove_generated_sidecar_sources(outcome.successful)
    _copy_install_payload(staged_source_roots, install_root)
    _copy_atoll_artifacts(staged_source_roots, install_root)
    _progress(options.progress, f"prepared install payload in {_duration(payload_started)}")
    metadata = _project_metadata(project.config.root)
    wheel_started = time.perf_counter()
    _progress(options.progress, f"writing wheel to {output_dir}")
    wheel_path = _write_wheel(
        install_root=install_root,
        output_dir=output_dir,
        metadata=metadata,
    )
    _progress(options.progress, f"wrote wheel in {_duration(wheel_started)}")
    cleanup_started = time.perf_counter()
    _progress(options.progress, "cleaning temporary build outputs")
    cleanup_removed_paths = [build_root]
    shutil.rmtree(build_root)
    cleanup_kept: tuple[Path, ...] = ()
    if not options.keep_install_tree:
        cleanup_removed_paths.append(install_root)
        shutil.rmtree(install_root)
    else:
        cleanup_kept = (install_root,)
    _progress(options.progress, f"cleaned temporary outputs in {_duration(cleanup_started)}")
    return PackageCommandResult(
        success=True,
        project_root=project.config.root,
        output_dir=output_dir,
        install_root=install_root,
        wheel_path=wheel_path,
        islands=outcome.successful,
        build=outcome.build,
        install_tree_kept=options.keep_install_tree,
        cleanup_removed=tuple(cleanup_removed_paths),
        cleanup_kept=cleanup_kept,
        report_artifact_paths=report_artifact_paths,
        skipped=outcome.skipped,
        preflight_skipped=(),
        native_readiness=native_readiness,
        typed_regions=typed_regions,
    )


def _handle_no_native_islands(
    *,
    options: PackageOptions,
    project: DiscoveredProject,
    build_root: Path,
    context: _TypedRegionPackageContext,
) -> PackageCommandResult:
    """Try selected method regions before reporting no native-ready work."""
    if context.selected:
        _progress(
            options.progress,
            "no function islands passed native readiness; trying typed regions",
        )
        return _execute_typed_region_package(
            options=options,
            project=project,
            context=context,
        )
    return _no_native_ready_result(
        project=project,
        build_root=build_root,
        native_readiness=context.native_readiness,
        preflight_skipped=context.preflight_skipped,
        typed_regions=context.typed_regions,
    )


def _execute_typed_region_package(
    *,
    options: PackageOptions,
    project: DiscoveredProject,
    context: _TypedRegionPackageContext,
) -> PackageCommandResult:
    """Build source-clean class and callable region variants.

    Generation and shims live only in copied build roots. Regions compile
    independently so one backend rejection leaves successful regions available
    in the wheel while preserving the original implementation as fallback.
    """
    output_dir = _resolve_output_dir(project.config.root, options.output_dir)
    build_root = output_dir / "build"
    install_root = output_dir / "install"
    _progress(options.progress, f"resetting temporary build roots in {output_dir}")
    _reset_dir(build_root)
    _reset_dir(install_root)

    copy_started = time.perf_counter()
    _progress(options.progress, "copying source roots into temporary build tree")
    staged_source_roots = _copy_source_roots(project, build_root)
    _progress(options.progress, f"copied source roots in {_duration(copy_started)}")

    generation_started = time.perf_counter()
    prepared: list[_PreparedTypedRegion] = []
    preparation_failures: list[PackageRegionBuildFailure] = []
    for selection in context.selected:
        try:
            prepared.append(
                _prepare_typed_region(
                    project=project,
                    build_root=build_root,
                    staged_source_roots=staged_source_roots,
                    selection=selection,
                )
            )
        except (SyntaxError, ValueError) as lowering_error:
            preparation_failures.append(
                PackageRegionBuildFailure(
                    region=selection.region,
                    variant_id=selection.variant_id,
                    backend=selection.backend,
                    assessment=selection.assessment,
                    build=_failed_region_attempt(f"typed-region lowering failed: {lowering_error}"),
                )
            )
    _progress(
        options.progress,
        (
            f"lowered {len(prepared)} typed region backend variant(s); "
            f"kept {len(preparation_failures)} as fallback in {_duration(generation_started)}"
        ),
    )
    if not prepared:
        result_error = "No selected typed regions could be lowered for a compiler backend."
        _remove_failed_wheels(project, output_dir)
        cleanup_removed = _remove_tree(install_root)
        return PackageCommandResult(
            success=False,
            project_root=project.config.root,
            output_dir=output_dir,
            install_root=install_root,
            wheel_path=None,
            islands=(),
            build=_failed_region_attempt(result_error),
            cleanup_removed=cleanup_removed,
            cleanup_kept=(build_root,),
            error=result_error,
            preflight_skipped=context.preflight_skipped,
            native_readiness=context.native_readiness,
            typed_regions=context.typed_regions,
            backend_assessments=tuple(selection.assessment for selection in context.selected),
            region_skipped=tuple(preparation_failures),
        )

    outcome = _build_typed_regions(
        prepared=tuple(prepared),
        context=_TypedRegionBuildContext(
            build_root=build_root,
            staged_source_roots=staged_source_roots,
            cache_dir=project.config.cache_dir / "mypy" / "source-clean",
            progress=options.progress,
        ),
        initial_failures=tuple(preparation_failures),
    )
    if not outcome.successful:
        _progress(options.progress, "all typed-region builds failed; keeping diagnostics")
        _remove_failed_wheels(project, output_dir)
        cleanup_removed = _remove_tree(install_root)
        return PackageCommandResult(
            success=False,
            project_root=project.config.root,
            output_dir=output_dir,
            install_root=install_root,
            wheel_path=None,
            islands=(),
            build=outcome.build,
            cleanup_removed=cleanup_removed,
            cleanup_kept=(build_root,),
            error=outcome.build.stderr,
            preflight_skipped=context.preflight_skipped,
            native_readiness=context.native_readiness,
            typed_regions=context.typed_regions,
            backend_assessments=tuple(selection.assessment for selection in context.selected),
            artifact_records=outcome.artifacts,
            region_skipped=outcome.skipped,
        )

    payload_started = time.perf_counter()
    _progress(options.progress, "binding compiled classes and callables in staged modules")
    successful_shims = tuple(item.shim for item in outcome.successful)
    _insert_region_shims(successful_shims)
    _place_region_artifacts(successful_shims, outcome.build.artifact_paths)
    report_artifact_paths = _source_clean_report_artifact_paths(
        project.config.root,
        outcome.build.artifact_paths,
    )
    for path in _prepared_source_paths(tuple(prepared)):
        path.unlink(missing_ok=True)
    _copy_install_payload(staged_source_roots, install_root)
    _copy_atoll_artifacts(staged_source_roots, install_root)
    _progress(options.progress, f"prepared install payload in {_duration(payload_started)}")

    metadata = _project_metadata(project.config.root)
    wheel_started = time.perf_counter()
    _progress(options.progress, f"writing wheel to {output_dir}")
    wheel_path = _write_wheel(
        install_root=install_root,
        output_dir=output_dir,
        metadata=metadata,
    )
    _progress(options.progress, f"wrote wheel in {_duration(wheel_started)}")

    cleanup_started = time.perf_counter()
    _progress(options.progress, "cleaning temporary build outputs")
    cleanup_removed_paths = [build_root]
    shutil.rmtree(build_root)
    cleanup_kept: tuple[Path, ...] = ()
    if not options.keep_install_tree:
        cleanup_removed_paths.append(install_root)
        shutil.rmtree(install_root)
    else:
        cleanup_kept = (install_root,)
    _progress(options.progress, f"cleaned temporary outputs in {_duration(cleanup_started)}")

    successful_regions = tuple(
        {item.generation.region.id: item.generation.region for item in outcome.successful}.values()
    )
    successful_bindings = tuple(
        binding for item in outcome.successful for binding in item.generation.bindings
    )
    successful_variants = tuple(
        CompiledRegionVariant(
            id=item.unit.region_id,
            region=item.generation.region,
            backend=item.generation.backend,
            bindings=item.generation.bindings,
        )
        for item in outcome.successful
    )
    return PackageCommandResult(
        success=True,
        project_root=project.config.root,
        output_dir=output_dir,
        install_root=install_root,
        wheel_path=wheel_path,
        islands=(),
        build=outcome.build,
        install_tree_kept=options.keep_install_tree,
        cleanup_removed=tuple(cleanup_removed_paths),
        cleanup_kept=cleanup_kept,
        report_artifact_paths=report_artifact_paths,
        preflight_skipped=context.preflight_skipped,
        native_readiness=context.native_readiness,
        typed_regions=context.typed_regions,
        compiled_regions=successful_regions,
        compiled_bindings=successful_bindings,
        compiled_variants=successful_variants,
        backend_assessments=tuple(selection.assessment for selection in context.selected),
        artifact_records=outcome.artifacts,
        region_skipped=outcome.skipped,
    )


def _prepare_typed_region(
    *,
    project: DiscoveredProject,
    build_root: Path,
    staged_source_roots: tuple[Path, ...],
    selection: _SelectedTypedRegion,
) -> _PreparedTypedRegion:
    staged_module = _staged_module(selection.scan.module, project, staged_source_roots)
    staged_scan = enrich_island_analysis(scan_module(staged_module))
    staged_selection = _staged_typed_selection(staged_scan, selection)
    staged_region = staged_selection.region
    staged = _StagedTypedRegionContext(
        build_root=build_root,
        staged_source_root=_staged_source_root(
            selection.scan.module,
            project,
            staged_source_roots,
        ),
        module=staged_module,
        scan=staged_scan,
        region=staged_region,
    )
    prepared = _prepare_backend_variant(staged, staged_selection)
    if staged_selection.backend != "mypyc":
        return prepared
    cython_assessment = CYTHON_BACKEND.assess(staged_region)
    if not set(staged_selection.members) <= set(cython_assessment.supported_members):
        return prepared
    fallback_selection = replace(
        staged_selection,
        variant_id=f"{staged_region.id}@cython-mypyc-fallback",
        backend="cython",
        assessment=cython_assessment,
    )
    try:
        fallback = _prepare_backend_variant(staged, fallback_selection)
    except (SyntaxError, ValueError):
        return prepared
    return replace(prepared, fallback=fallback)


def _staged_typed_selection(
    staged_scan: ModuleScan,
    selection: _SelectedTypedRegion,
) -> _SelectedTypedRegion:
    """Rebind a deterministic selection to equivalent evidence in the copied tree."""
    if selection.specialization is None:
        staged_region = next(
            region for region in staged_scan.typed_regions if region.id == selection.region.id
        )
        return replace(selection, scan=staged_scan, region=staged_region)
    staged_source_region = next(
        region
        for region in staged_scan.typed_regions
        if any(
            specialization.id == selection.specialization.id
            for specialization in region.specializations
        )
    )
    staged_specialization = next(
        specialization
        for specialization in staged_source_region.specializations
        if specialization.id == selection.specialization.id
    )
    staged_region = _specialized_region(staged_source_region, staged_specialization)
    backend = _compiler_backend(selection.backend)
    return replace(
        selection,
        scan=staged_scan,
        region=staged_region,
        assessment=backend.assess(staged_region),
        members=(staged_specialization.source_member,),
        specialization=staged_specialization,
    )


def _prepare_backend_variant(
    staged: _StagedTypedRegionContext,
    selection: _SelectedTypedRegion,
) -> _PreparedTypedRegion:
    """Lower one selected backend variant inside the copied build tree."""
    logical_module = _typed_region_module_name(
        staged.region,
        selection.backend,
        selection.variant_id,
    )
    generated_path = staged.build_root / f"{logical_module}.py"
    generation = generate_typed_method_region(
        staged.scan,
        staged.region,
        selection.members,
        output_path=generated_path,
        options=TypedRegionGenerationOptions(
            backend=selection.backend,
            specialization=selection.specialization,
        ),
    )
    unit = _compiler_backend(selection.backend).lower(
        BackendLoweringRequest(
            region=staged.region,
            source_path=generated_path,
            logical_module=logical_module,
            install_relative_dir=".atoll/artifacts",
            members=selection.members,
            variant_id=selection.variant_id,
        )
    )
    return _PreparedTypedRegion(
        generation=generation,
        assessment=selection.assessment,
        unit=unit,
        shim=RegionShimConfig(
            source_module=staged.module.name,
            source_path=staged.module.path,
            region_id=selection.variant_id,
            backend=selection.backend,
            compiled_module=logical_module,
            artifact_dir=staged.staged_source_root / ".atoll" / "artifacts",
            bindings=generation.bindings,
        ),
        conditional_on_failure_of=selection.conditional_on_failure_of,
    )


def _prepared_source_paths(
    prepared: tuple[_PreparedTypedRegion, ...],
) -> tuple[Path, ...]:
    """Return primary and speculative fallback source paths for cleanup."""
    return tuple(
        path
        for item in prepared
        for path in (
            item.generation.source_path,
            *((item.fallback.generation.source_path,) if item.fallback is not None else ()),
        )
    )


def _build_typed_regions(
    *,
    prepared: tuple[_PreparedTypedRegion, ...],
    context: _TypedRegionBuildContext,
    initial_failures: tuple[PackageRegionBuildFailure, ...],
) -> _TypedRegionBuildOutcome:
    successful: list[_PreparedTypedRegion] = []
    skipped = list(initial_failures)
    attempts: list[CompileAttempt] = [failure.build for failure in initial_failures]
    artifacts: list[ArtifactRecord] = []
    successful_promises: set[str] = set()
    backend_context = BackendCompileContext(
        project_root=context.build_root,
        build_dir=context.build_root / ".atoll" / "build",
        source_roots=context.staged_source_roots,
        cache_dir=context.cache_dir,
    )
    for index, item in enumerate(prepared, start=1):
        if (
            item.conditional_on_failure_of is not None
            and item.conditional_on_failure_of in successful_promises
        ):
            _progress(
                context.progress,
                f"class variant succeeded; skipped method fallback {item.unit.region_id}",
            )
            continue
        _progress(
            context.progress,
            (
                f"compiling typed region variant {index}/{len(prepared)} with "
                f"{item.generation.backend}: {item.unit.region_id}"
            ),
        )
        result = _compile_typed_variant(item, backend_context)
        tagged_attempt = _tag_region_timings(result.attempt, item.unit.region_id)
        if result.attempt.success:
            attempts.append(tagged_attempt)
            successful.append(item)
            artifacts.extend(result.artifacts)
            successful_promises.add(item.unit.region_id)
            _progress(context.progress, f"compiled typed region variant {item.unit.region_id}")
            continue
        failure_item = item
        failure_result = result
        if _should_retry_with_cython(item, result) and item.fallback is not None:
            fallback = item.fallback
            _progress(
                context.progress,
                f"retrying deterministic mypyc failure with Cython: {fallback.unit.region_id}",
            )
            fallback_result = _compile_typed_variant(fallback, backend_context)
            fallback_attempt = _tag_region_timings(
                fallback_result.attempt,
                fallback.unit.region_id,
            )
            if fallback_result.attempt.success:
                attempts.extend(
                    (
                        _recovered_mypyc_attempt(tagged_attempt, fallback.unit.region_id),
                        fallback_attempt,
                    )
                )
                successful.append(fallback)
                artifacts.extend(fallback_result.artifacts)
                successful_promises.add(item.unit.region_id)
                _progress(
                    context.progress,
                    f"compiled Cython fallback variant {fallback.unit.region_id}",
                )
                continue
            attempts.extend((tagged_attempt, fallback_attempt))
            failure_item = fallback
            failure_result = fallback_result
        else:
            attempts.append(tagged_attempt)
        skipped.append(
            PackageRegionBuildFailure(
                region=failure_item.generation.region,
                variant_id=failure_item.unit.region_id,
                backend=failure_item.generation.backend,
                assessment=failure_item.assessment,
                build=failure_result.attempt,
            )
        )
        _progress(
            context.progress,
            f"kept typed region variant {failure_item.unit.region_id} as fallback",
        )
    return _TypedRegionBuildOutcome(
        successful=tuple(successful),
        build=_aggregate_region_attempts(tuple(attempts), bool(successful)),
        artifacts=tuple(artifacts),
        skipped=tuple(skipped),
    )


def _compile_typed_variant(
    item: _PreparedTypedRegion,
    context: BackendCompileContext,
) -> BackendCompileResult:
    """Invoke the adapter selected for one prepared backend variant."""
    return _compiler_backend(item.generation.backend).compile((item.unit,), context)


def _should_retry_with_cython(
    item: _PreparedTypedRegion,
    result: BackendCompileResult,
) -> bool:
    """Return whether a deterministic mypyc rejection permits Cython retry."""
    return (
        item.generation.backend == "mypyc"
        and not result.attempt.success
        and result.attempt.stderr.startswith("MYPYC_TYPE_ERROR:")
    )


def _recovered_mypyc_attempt(
    attempt: CompileAttempt,
    fallback_variant_id: str,
) -> CompileAttempt:
    """Retain deterministic rejection evidence without failing the aggregate build."""
    return replace(
        attempt,
        success=True,
        stdout="\n".join(
            part
            for part in (
                attempt.stdout,
                f"mypyc rejected this variant; compiled {fallback_variant_id} with Cython",
                attempt.stderr,
            )
            if part
        ),
        stderr="",
        artifact_paths=(),
    )


def _tag_region_timings(attempt: CompileAttempt, region_id: str) -> CompileAttempt:
    return replace(
        attempt,
        phase_timings=tuple(
            replace(
                timing,
                detail=f"{region_id}; {timing.detail}" if timing.detail else region_id,
            )
            for timing in attempt.phase_timings
        ),
    )


def _aggregate_region_attempts(
    attempts: tuple[CompileAttempt, ...],
    success: bool,
) -> CompileAttempt:
    return CompileAttempt(
        success=success,
        command=("atoll", "typed-region-build"),
        stdout="\n".join(attempt.stdout for attempt in attempts if attempt.stdout),
        stderr="\n\n".join(attempt.stderr for attempt in attempts if not attempt.success),
        artifact_paths=tuple(
            dict.fromkeys(
                artifact
                for attempt in attempts
                if attempt.success
                for artifact in attempt.artifact_paths
            )
        ),
        duration_seconds=sum(attempt.duration_seconds for attempt in attempts),
        phase_timings=tuple(timing for attempt in attempts for timing in attempt.phase_timings),
    )


def _failed_region_attempt(error: str) -> CompileAttempt:
    return CompileAttempt(
        success=False,
        command=(),
        stdout="",
        stderr=error,
        artifact_paths=(),
        duration_seconds=0.0,
    )


def _insert_region_shims(configs: tuple[RegionShimConfig, ...]) -> None:
    configs_by_path: dict[Path, list[RegionShimConfig]] = {}
    for config in configs:
        configs_by_path.setdefault(config.source_path, []).append(config)
    for source_path, module_configs in configs_by_path.items():
        source_text = source_path.read_text(encoding="utf-8")
        source_path.write_text(
            insert_or_replace_region_shim(source_text, tuple(module_configs)).new_text,
            encoding="utf-8",
        )


def _place_region_artifacts(
    configs: tuple[RegionShimConfig, ...],
    artifact_paths: tuple[Path, ...],
) -> None:
    for artifact_dir in sorted({config.artifact_dir for config in configs}):
        artifact_dir.mkdir(parents=True, exist_ok=True)
        for artifact in artifact_paths:
            _copy_if_different(artifact, artifact_dir / artifact.name)


def _typed_region_module_name(
    region: TypedRegion,
    backend: Backend,
    variant_id: str,
) -> str:
    module = re.sub(r"[^A-Za-z0-9_]", "_", region.source_module.name)
    variant_hash = hashlib.sha256(variant_id.encode()).hexdigest()[:8]
    return f"_atoll_region_{module}_{backend}_{region.source_hash[:12]}_{variant_hash}"


def _compiler_backend(backend: Backend) -> CompilerBackend:
    """Return the configured compiler adapter for one automatic selection."""
    return _COMPILER_BACKENDS[backend]


def _no_native_ready_result(
    *,
    project: DiscoveredProject,
    build_root: Path,
    native_readiness: tuple[NativeReadiness, ...],
    preflight_skipped: tuple[PackagePreflightFailure, ...],
    typed_regions: tuple[TypedRegion, ...],
) -> PackageCommandResult:
    """Return a clean failure without invoking mypyc or retaining a stale wheel."""
    output_dir = build_root.parent
    install_root = output_dir / "install"
    _remove_failed_wheels(project, output_dir)
    cleanup_removed = (*_remove_tree(build_root), *_remove_tree(install_root))
    error = (
        "No performance-worthy native islands remain after generated-code analysis. "
        f"Rejected {len(native_readiness)} scan candidate symbol(s); mypyc was not invoked. "
        "See the compile report for native-readiness reasons."
    )
    return PackageCommandResult(
        success=False,
        project_root=project.config.root,
        output_dir=output_dir,
        install_root=install_root,
        wheel_path=None,
        islands=(),
        build=CompileAttempt(
            success=False,
            command=(),
            stdout="",
            stderr=error,
            artifact_paths=(),
            duration_seconds=0.0,
        ),
        cleanup_removed=cleanup_removed,
        error=error,
        preflight_skipped=preflight_skipped,
        native_readiness=native_readiness,
        typed_regions=typed_regions,
    )


def _failed_result(
    root: Path,
    output_dir: Path | None,
    error: str,
    *,
    preflight_skipped: tuple[PackagePreflightFailure, ...] = (),
    typed_regions: tuple[TypedRegion, ...] = (),
) -> PackageCommandResult:
    resolved_output_dir = _resolve_output_dir(root, output_dir)
    return PackageCommandResult(
        success=False,
        project_root=root,
        output_dir=resolved_output_dir,
        install_root=resolved_output_dir / "install",
        wheel_path=None,
        islands=(),
        build=CompileAttempt(
            success=False,
            command=(),
            stdout="",
            stderr=error,
            artifact_paths=(),
            duration_seconds=0.0,
        ),
        error=error,
        preflight_skipped=preflight_skipped,
        typed_regions=typed_regions,
    )


def _remove_tree(path: Path) -> tuple[Path, ...]:
    if not path.exists():
        return ()
    shutil.rmtree(path)
    return (path,)


def _progress(progress: PackageProgress | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _progress_compile_selection(
    progress: PackageProgress | None,
    selected_modules: tuple[_SelectedModule, ...],
    selected_regions: tuple[_SelectedTypedRegion, ...],
) -> None:
    function_count = sum(len(module.symbols) for module in selected_modules)
    member_count = sum(len(region.members) for region in selected_regions)
    specialization_count = sum(region.specialization is not None for region in selected_regions)
    _progress(
        progress,
        (
            f"selected {len(selected_modules)} candidate module(s), {function_count} "
            f"function(s), and {len(selected_regions)} typed region backend variant(s), "
            f"{member_count} member(s), {specialization_count} specialization(s)"
        ),
    )


def _duration(started: float) -> str:
    return f"{time.perf_counter() - started:.2f}s"


def _progress_phase_timings(
    progress: PackageProgress | None,
    timings: tuple[CompilePhaseTiming, ...],
) -> None:
    for timing in timings:
        detail = f" ({timing.detail})" if timing.detail else ""
        _progress(progress, f"{timing.name} completed in {timing.duration_seconds:.2f}s{detail}")


def _lookup_compile_cache(
    *,
    target_project: DiscoveredProject,
    module_name: str | None,
    islands: tuple[EnabledIslandConfig, ...],
) -> _CompileCacheLookup:
    key = _compile_cache_key(
        target_project=target_project,
        module_name=module_name,
        islands=islands,
    )
    cache_root = target_project.config.cache_dir / "compile"
    lookup_started = time.perf_counter()
    entry_root = cache_root / key
    manifest = _read_cache_manifest(entry_root / "manifest.json")
    if manifest is None or manifest.get("version") != _COMPILE_CACHE_VERSION:
        return _compile_cache_miss(key, lookup_started, "miss")
    if manifest.get("key") != key:
        return _compile_cache_miss(key, lookup_started, "key mismatch")
    artifacts = _cached_artifact_paths(entry_root, manifest)
    if artifacts is None:
        return _compile_cache_miss(key, lookup_started, "stale")
    cached_modules = _cached_manifest_modules(manifest)
    if cached_modules is None:
        return _compile_cache_miss(key, lookup_started, "stale")
    successful_modules, skipped_modules = cached_modules
    current_modules = {island.source_module for island in islands}
    if set(successful_modules) | set(skipped_modules) != current_modules:
        return _compile_cache_miss(key, lookup_started, "selection mismatch")
    return _compile_cache_hit(
        key=key,
        lookup_started=lookup_started,
        artifact_paths=artifacts,
        successful_modules=successful_modules,
        skipped_modules=skipped_modules,
    )


def _compile_cache_hit(
    *,
    key: str,
    lookup_started: float,
    artifact_paths: tuple[Path, ...],
    successful_modules: tuple[str, ...],
    skipped_modules: tuple[str, ...],
) -> _CompileCacheLookup:
    lookup_timing = CompilePhaseTiming(
        name="cache_lookup",
        duration_seconds=time.perf_counter() - lookup_started,
        detail="hit" if not skipped_modules else "partial hit",
    )
    restore_started = time.perf_counter()
    restored = tuple(path for path in artifact_paths if path.exists())
    restore_timing = CompilePhaseTiming(
        name="cache_restore",
        duration_seconds=time.perf_counter() - restore_started,
        detail=f"{len(restored)} artifact(s)",
    )
    if len(restored) != len(artifact_paths):
        return _CompileCacheLookup(
            key=key,
            hit=False,
            artifact_paths=(),
            successful_modules=(),
            skipped_modules=(),
            phase_timings=(
                lookup_timing,
                CompilePhaseTiming(
                    name="cache_restore",
                    duration_seconds=restore_timing.duration_seconds,
                    detail="stale",
                ),
            ),
        )
    return _CompileCacheLookup(
        key=key,
        hit=True,
        artifact_paths=restored,
        successful_modules=successful_modules,
        skipped_modules=skipped_modules,
        phase_timings=(lookup_timing, restore_timing),
    )


def _compile_cache_miss(
    key: str,
    lookup_started: float,
    detail: str,
) -> _CompileCacheLookup:
    return _CompileCacheLookup(
        key=key,
        hit=False,
        artifact_paths=(),
        successful_modules=(),
        skipped_modules=(),
        phase_timings=(
            CompilePhaseTiming(
                name="cache_lookup",
                duration_seconds=time.perf_counter() - lookup_started,
                detail=detail,
            ),
        ),
    )


def _read_cache_manifest(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    raw = cast(dict[object, object], data)
    return {str(key): value for key, value in raw.items()}


def _cached_artifact_paths(
    entry_root: Path,
    manifest: dict[str, object],
) -> tuple[Path, ...] | None:
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list):
        return None
    paths: list[Path] = []
    for raw_artifact in cast(list[object], raw_artifacts):
        if not isinstance(raw_artifact, dict):
            return None
        artifact = cast(dict[object, object], raw_artifact)
        name = artifact.get("name")
        digest = artifact.get("sha256")
        if not isinstance(name, str) or not isinstance(digest, str):
            return None
        path = entry_root / "artifacts" / name
        if not path.exists() or _file_digest(path) != digest:
            return None
        paths.append(path)
    return tuple(paths)


def _cached_manifest_modules(
    manifest: dict[str, object],
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    successful = _string_tuple_manifest_field(manifest, "successful_modules")
    skipped = _string_tuple_manifest_field(manifest, "skipped_modules")
    if successful is None or skipped is None:
        return None
    if set(successful) & set(skipped):
        return None
    return successful, skipped


def _string_tuple_manifest_field(
    manifest: dict[str, object],
    field: str,
) -> tuple[str, ...] | None:
    raw = manifest.get(field)
    if not isinstance(raw, list):
        return None
    values: list[str] = []
    for item in cast(list[object], raw):
        if not isinstance(item, str):
            return None
        values.append(item)
    return tuple(values)


def _store_compile_cache(
    *,
    cache_root: Path,
    key: str,
    artifact_paths: tuple[Path, ...],
    successful_modules: tuple[str, ...],
    skipped_modules: tuple[str, ...],
) -> None:
    if not artifact_paths:
        return
    cache_root.mkdir(parents=True, exist_ok=True)
    entry_root = cache_root / key
    temp_root = cache_root / f"{key}.tmp"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    artifact_root = temp_root / "artifacts"
    artifact_root.mkdir(parents=True)
    manifest_artifacts: list[dict[str, str]] = []
    for artifact in artifact_paths:
        destination = artifact_root / artifact.name
        shutil.copy2(artifact, destination)
        manifest_artifacts.append({"name": artifact.name, "sha256": _file_digest(destination)})
    manifest = {
        "version": _COMPILE_CACHE_VERSION,
        "key": key,
        "artifacts": manifest_artifacts,
        "successful_modules": list(successful_modules),
        "skipped_modules": list(skipped_modules),
    }
    (temp_root / "manifest.json").write_text(
        f"{json.dumps(manifest, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    if entry_root.exists():
        shutil.rmtree(entry_root)
    temp_root.rename(entry_root)


def _compile_cache_key(
    *,
    target_project: DiscoveredProject,
    module_name: str | None,
    islands: tuple[EnabledIslandConfig, ...],
) -> str:
    payload = {
        "version": _COMPILE_CACHE_VERSION,
        "python_tag": _python_tag(),
        "wheel_tag": _wheel_tag(),
        "extension_suffixes": list(importlib.machinery.EXTENSION_SUFFIXES),
        "atoll_version": _installed_version("atoll"),
        "mypy_version": _installed_version("mypy"),
        "setuptools_version": _installed_version("setuptools"),
        "sidecar_generator_version": SIDECAR_GENERATOR_VERSION,
        "module_filter": module_name,
        "source_tree_digest": _source_tree_digest(target_project),
        "source_roots": [
            _path_text(target_project.config.root, source_root)
            for source_root in target_project.config.source_roots
        ],
        "islands": [
            {
                "source_module": island.source_module,
                "sidecar_module": island.sidecar_module,
                "symbols": list(island.symbols),
                "sidecar_sha256": _file_digest(island.sidecar_path),
            }
            for island in islands
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_tree_digest(project: DiscoveredProject) -> str:
    digest = hashlib.sha256()
    for path in _cache_input_paths(project):
        digest.update(_path_text(project.config.root, path).encode())
        digest.update(b"\0")
        digest.update(_file_digest(path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _cache_input_paths(project: DiscoveredProject) -> tuple[Path, ...]:
    paths: set[Path] = set()
    pyproject = project.config.root / "pyproject.toml"
    if pyproject.exists():
        paths.add(pyproject)
    for source_root in project.config.source_roots:
        for path in source_root.rglob("*"):
            if not path.is_file() or _is_ignored_cache_input(path, source_root):
                continue
            if path.suffix in _CACHE_INPUT_SUFFIXES or path.name in _CACHE_INPUT_NAMES:
                paths.add(path)
    return tuple(sorted(paths))


def _is_ignored_cache_input(path: Path, source_root: Path) -> bool:
    relative_parts = path.relative_to(source_root).parts
    return any(
        part in _GENERATED_DIR_NAMES
        or part in {".nox", ".tox", ".venv", "venv"}
        or part.endswith((".egg-info", ".dist-info"))
        for part in relative_parts
    )


def _installed_version(package: str) -> str:
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_package_islands(
    islands: tuple[EnabledIslandConfig, ...],
    context: _PackageBuildContext,
) -> _PackageBuildOutcome:
    cache_lookup = _lookup_compile_cache(
        target_project=context.target_project,
        module_name=context.module_name,
        islands=islands,
    )
    if cache_lookup.hit:
        _progress(context.progress, f"compile cache hit: {cache_lookup.key[:12]}")
        _progress_phase_timings(context.progress, cache_lookup.phase_timings)
        successful_modules = set(cache_lookup.successful_modules)
        skipped_modules = set(cache_lookup.skipped_modules)
        successful_islands = tuple(
            island for island in islands if island.source_module in successful_modules
        )
        skipped = tuple(
            _cached_skipped_failure(island, cache_lookup.key)
            for island in islands
            if island.source_module in skipped_modules
        )
        for failure in skipped:
            _remove_staged_island(failure.island)
        return _PackageBuildOutcome(
            successful=successful_islands,
            build=CompileAttempt(
                success=True,
                command=("atoll", "compile-cache", "restore", cache_lookup.key[:12]),
                stdout=(
                    "compile cache hit"
                    if not skipped
                    else f"compile cache hit; restored {len(successful_islands)} island(s), "
                    f"kept {len(skipped)} cached skip(s)"
                ),
                stderr="",
                artifact_paths=cache_lookup.artifact_paths,
                duration_seconds=sum(
                    timing.duration_seconds for timing in cache_lookup.phase_timings
                ),
                phase_timings=cache_lookup.phase_timings,
                cache_status="hit",
            ),
            skipped=skipped,
        )
    _progress(context.progress, f"compile cache miss: {cache_lookup.key[:12]}")
    batch_started = time.perf_counter()
    _progress(context.progress, f"running mypyc batch for {len(islands)} island(s)")
    batch = build_sidecars(
        tuple(island.sidecar_path for island in islands),
        project_root=context.project_root,
        build_dir=context.project_root / ".atoll" / "build",
        source_roots=context.source_roots,
        cache_dir=context.target_project.config.cache_dir / "mypy" / "source-clean",
    )
    batch = replace(
        batch,
        phase_timings=(*cache_lookup.phase_timings, *batch.phase_timings),
        cache_status="miss",
    )
    if batch.success:
        cache_store_started = time.perf_counter()
        _store_compile_cache(
            cache_root=context.target_project.config.cache_dir / "compile",
            key=cache_lookup.key,
            artifact_paths=batch.artifact_paths,
            successful_modules=tuple(island.source_module for island in islands),
            skipped_modules=(),
        )
        batch = replace(
            batch,
            phase_timings=(
                *batch.phase_timings,
                CompilePhaseTiming(
                    name="cache_store",
                    duration_seconds=time.perf_counter() - cache_store_started,
                    detail="stored",
                ),
            ),
        )
        _progress_phase_timings(context.progress, batch.phase_timings)
        _progress(context.progress, f"mypyc batch succeeded in {_duration(batch_started)}")
        return _PackageBuildOutcome(successful=islands, build=batch, skipped=())
    if not context.allow_partial or len(islands) <= 1:
        _progress(context.progress, f"mypyc batch failed in {_duration(batch_started)}")
        return _PackageBuildOutcome(successful=(), build=batch, skipped=())
    _progress(
        context.progress,
        f"mypyc batch failed in {_duration(batch_started)}; retrying islands individually",
    )
    outcome = _build_package_islands_individually(
        islands,
        context,
        batch_failure=batch,
    )
    if not outcome.build.success:
        return outcome
    cache_store_started = time.perf_counter()
    _store_compile_cache(
        cache_root=context.target_project.config.cache_dir / "compile",
        key=cache_lookup.key,
        artifact_paths=outcome.build.artifact_paths,
        successful_modules=tuple(island.source_module for island in outcome.successful),
        skipped_modules=tuple(failure.island.source_module for failure in outcome.skipped),
    )
    return _PackageBuildOutcome(
        successful=outcome.successful,
        build=replace(
            outcome.build,
            phase_timings=(
                *outcome.build.phase_timings,
                CompilePhaseTiming(
                    name="cache_store",
                    duration_seconds=time.perf_counter() - cache_store_started,
                    detail="stored partial",
                ),
            ),
        ),
        skipped=outcome.skipped,
    )


def _cached_skipped_failure(island: EnabledIslandConfig, key: str) -> PackageBuildFailure:
    return PackageBuildFailure(
        island=island,
        build=CompileAttempt(
            success=False,
            command=("atoll", "compile-cache", "skip", key[:12]),
            stdout="",
            stderr=f"cached skip: previous mypyc build failed for {island.source_module}",
            artifact_paths=(),
            duration_seconds=0.0,
            cache_status="hit",
        ),
    )


def _build_package_islands_individually(
    islands: tuple[EnabledIslandConfig, ...],
    context: _PackageBuildContext,
    *,
    batch_failure: CompileAttempt,
) -> _PackageBuildOutcome:
    successful: list[EnabledIslandConfig] = []
    skipped: list[PackageBuildFailure] = []
    attempts: list[CompileAttempt] = []
    for index, island in enumerate(islands, start=1):
        retry_started = time.perf_counter()
        _progress(context.progress, f"retrying {island.source_module} ({index}/{len(islands)})")
        attempt = build_sidecars(
            (island.sidecar_path,),
            project_root=context.project_root,
            build_dir=context.project_root / ".atoll" / "retry-builds" / island.sidecar_path.stem,
            source_roots=context.source_roots,
            cache_dir=context.target_project.config.cache_dir / "mypy" / "source-clean",
        )
        attempts.append(attempt)
        if attempt.success:
            successful.append(island)
            _progress(
                context.progress,
                f"compiled {island.source_module} in {_duration(retry_started)}",
            )
            continue
        skipped.append(PackageBuildFailure(island=island, build=attempt))
        _remove_staged_island(island)
        _progress(context.progress, f"skipped {island.source_module} in {_duration(retry_started)}")
    combined = _combine_package_attempts(
        batch_failure=batch_failure,
        attempts=tuple(attempts),
        successful_count=len(successful),
        skipped_count=len(skipped),
    )
    return _PackageBuildOutcome(
        successful=tuple(successful),
        build=combined,
        skipped=tuple(skipped),
    )


def _combine_package_attempts(
    *,
    batch_failure: CompileAttempt,
    attempts: tuple[CompileAttempt, ...],
    successful_count: int,
    skipped_count: int,
) -> CompileAttempt:
    artifact_paths = tuple(path for attempt in attempts for path in attempt.artifact_paths)
    stdout_parts = [
        (
            "Initial batch build failed; retried islands individually. "
            f"Compiled {successful_count}, skipped {skipped_count}."
        )
    ]
    failed_attempts = tuple(attempt for attempt in attempts if not attempt.success)
    stderr_parts = (
        [_no_successful_retry_error(failed_attempts, batch_failure)]
        if successful_count == 0
        else [batch_failure.stderr, *(attempt.stderr for attempt in failed_attempts)]
    )
    return CompileAttempt(
        success=successful_count > 0,
        command=("mypyc", "partial-package-build"),
        stdout="\n".join(part for part in stdout_parts if part),
        stderr="\n\n".join(part for part in stderr_parts if part),
        artifact_paths=artifact_paths,
        duration_seconds=batch_failure.duration_seconds
        + sum(attempt.duration_seconds for attempt in attempts),
        phase_timings=(
            *batch_failure.phase_timings,
            *(timing for attempt in attempts for timing in attempt.phase_timings),
        ),
        cache_status="partial",
    )


def _no_successful_retry_error(
    attempts: tuple[CompileAttempt, ...],
    batch_failure: CompileAttempt,
) -> str:
    first_failure = next((attempt.stderr for attempt in attempts if attempt.stderr), "")
    if not first_failure:
        first_failure = batch_failure.stderr
    return "\n".join(
        part
        for part in (
            "No selected islands compiled after retrying them individually.",
            first_failure,
        )
        if part
    )


def _selected_scans(
    project: DiscoveredProject,
    module_name: str | None,
) -> tuple[ModuleScan, ...]:
    modules = (_find_module(project.modules, module_name),) if module_name else project.modules
    return tuple(enrich_island_analysis(scan_module(module)) for module in modules)


def _selected_modules(
    scans: tuple[ModuleScan, ...],
) -> tuple[_SelectedModule, ...]:
    selected: list[_SelectedModule] = []
    for scan in scans:
        symbols = _candidate_symbols(scan)
        if symbols:
            selected.append(_SelectedModule(scan=scan, symbols=symbols))
    return tuple(selected)


def _selected_typed_method_regions(
    scans: tuple[ModuleScan, ...],
) -> tuple[_SelectedTypedRegion, ...]:
    selected: list[_SelectedTypedRegion] = []
    for scan in scans:
        for region in scan.typed_regions:
            decisions = {decision.target: decision for decision in region.decisions}
            mypyc_assessment = MYPYC_BACKEND.assess(region)
            cython_assessment = CYTHON_BACKEND.assess(region)
            atomic_class_member = _eligible_atomic_class(region, cython_assessment)
            atomic_variant_id: str | None = None
            if atomic_class_member is not None:
                atomic_variant_id = f"{region.id}@cython-class"
                selected.append(
                    _SelectedTypedRegion(
                        scan=scan,
                        region=region,
                        variant_id=atomic_variant_id,
                        backend="cython",
                        assessment=cython_assessment,
                        members=(atomic_class_member,),
                    )
                )
            eligible = _eligible_typed_methods(region, decisions)
            mypyc_supported = set(mypyc_assessment.supported_members)
            mypyc_members = tuple(member for member in eligible if member in mypyc_supported)
            if mypyc_members:
                selected.append(
                    _SelectedTypedRegion(
                        scan=scan,
                        region=region,
                        variant_id=f"{region.id}@mypyc",
                        backend="mypyc",
                        assessment=mypyc_assessment,
                        members=mypyc_members,
                        conditional_on_failure_of=atomic_variant_id,
                    )
                )
            cython_supported = set(cython_assessment.supported_members)
            cython_members = tuple(
                member
                for member in eligible
                if member not in mypyc_supported and member in cython_supported
            )
            if cython_members:
                selected.append(
                    _SelectedTypedRegion(
                        scan=scan,
                        region=region,
                        variant_id=f"{region.id}@cython",
                        backend="cython",
                        assessment=cython_assessment,
                        members=cython_members,
                        conditional_on_failure_of=atomic_variant_id,
                    )
                )
            for specialization in region.specializations:
                specialized_region = _specialized_region(region, specialization)
                mypyc_assessment = MYPYC_BACKEND.assess(specialized_region)
                if specialization.source_member in mypyc_assessment.supported_members:
                    selected.append(
                        _SelectedTypedRegion(
                            scan=scan,
                            region=specialized_region,
                            variant_id=f"{specialization.id}@mypyc",
                            backend="mypyc",
                            assessment=mypyc_assessment,
                            members=(specialization.source_member,),
                            specialization=specialization,
                        )
                    )
                    continue
                cython_assessment = CYTHON_BACKEND.assess(specialized_region)
                if specialization.source_member in cython_assessment.supported_members:
                    selected.append(
                        _SelectedTypedRegion(
                            scan=scan,
                            region=specialized_region,
                            variant_id=f"{specialization.id}@cython",
                            backend="cython",
                            assessment=cython_assessment,
                            members=(specialization.source_member,),
                            specialization=specialization,
                        )
                    )
    return tuple(selected)


def _eligible_atomic_class(
    region: TypedRegion,
    assessment: BackendAssessment,
) -> SymbolId | None:
    """Return the class binding only when mypyc supports its complete region."""
    if not region.atomic_class or assessment.status != "supported":
        return None
    class_members = tuple(member for member in region.members if member.kind == "class")
    method_members = tuple(member for member in region.members if member.kind == "method")
    if len(class_members) != 1 or any(
        member.kind not in {"class", "method"} for member in region.members
    ):
        return None
    if not method_members:
        return None
    if any(member.execution_kind != "sync" for member in method_members):
        return None
    supported = set(assessment.supported_members)
    if any(member.id not in supported for member in region.members):
        return None
    return class_members[0].id


def _specialized_region(
    source_region: TypedRegion,
    specialization: RegionSpecialization,
) -> TypedRegion:
    """Materialize one backend-assessable view without changing generic source IR."""
    member = next(item for item in source_region.members if item.id == specialization.source_member)
    source_hash = hashlib.sha256(
        f"{source_region.source_hash}:{specialization.id}".encode()
    ).hexdigest()
    return TypedRegion(
        id=specialization.id,
        source_module=source_region.source_module,
        members=(member,),
        dependencies=tuple(
            dependency
            for dependency in source_region.dependencies
            if dependency.src == specialization.source_member
        ),
        type_bindings=specialization.type_bindings,
        bindings=(),
        decisions=(
            LoweringDecision(
                target=specialization.source_member.stable_id,
                action="specialize",
                reason=(
                    "all generic parameters resolved from "
                    + specialization.origin.replace("_", " ")
                ),
            ),
        ),
        source_hash=source_hash,
        atomic_class=False,
        specializations=(specialization,),
    )


def _eligible_typed_methods(
    region: TypedRegion,
    decisions: dict[str, LoweringDecision],
) -> tuple[SymbolId, ...]:
    return tuple(
        member.id
        for member in region.members
        if member.kind == "method"
        and member.binding_kind in {"instance_method", "staticmethod", "classmethod"}
        and not member.id.qualname.rsplit(".", 1)[-1].startswith("__")
        and decisions[member.id.stable_id].action == "preserve"
        and not _owner_disallows_method_binding(
            member.owner_class,
            region.source_module.name,
            decisions,
        )
        and not _member_requires_source_class(member.source_text)
    )


def _owner_disallows_method_binding(
    owner_class: str | None,
    module_name: str,
    decisions: dict[str, LoweringDecision],
) -> bool:
    if owner_class is None:
        return True
    decision = decisions.get(f"{module_name}::{owner_class}")
    if decision is None:
        return False
    return any(
        reason in decision.reason
        for reason in (
            "decorators may register or replace",
            "dynamic behavior is blocked",
            "module binding is reassigned",
            "special method",
        )
    )


def _member_requires_source_class(source_text: str) -> bool:
    """Reject method extraction when Python's class compilation supplies semantics."""
    tree = ast.parse(textwrap.dedent(source_text))
    return any(
        (isinstance(node, ast.Name) and node.id == "__class__")
        or (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "super"
            and not node.args
            and not node.keywords
        )
        or (
            isinstance(node, ast.Attribute)
            and node.attr.startswith("__")
            and not node.attr.endswith("__")
        )
        for node in ast.walk(tree)
    )


def _candidate_symbols(scan: ModuleScan) -> tuple[str, ...]:
    candidates = {symbol for candidate in scan.island_candidates for symbol in candidate.symbols}
    return tuple(
        symbol.id.qualname
        for symbol in scan.symbols
        if symbol.id in candidates and symbol.kind == "function"
    )


def _prepare_staged_island(
    *,
    project: DiscoveredProject,
    staged_source_roots: tuple[Path, ...],
    selected_module: _SelectedModule,
) -> _PreparedModule:
    staged_module = _staged_module(selected_module.scan.module, project, staged_source_roots)
    staged_source_root = _staged_source_root(
        selected_module.scan.module,
        project,
        staged_source_roots,
    )
    staged_scan = enrich_island_analysis(scan_module(staged_module))
    sidecar_module = default_sidecar_module(staged_module.name)
    sidecar_path = expected_sidecar_path(staged_source_root, sidecar_module)
    native_readiness: list[NativeReadiness] = []
    for symbol in selected_module.symbols:
        probe = EnabledIslandConfig(
            source_module=staged_module.name,
            source_path=staged_module.path,
            sidecar_module=sidecar_module,
            sidecar_path=sidecar_path,
            symbols=(symbol,),
        )
        generation = generate_sidecar(staged_scan, probe)
        native_readiness.append(
            analyze_native_readiness(
                source_module=staged_module.name,
                exported_symbol=symbol,
                generated_source=generation.source_text,
            )
        )
    eligible_symbols = tuple(
        readiness.symbol for readiness in native_readiness if readiness.eligible
    )
    if not eligible_symbols:
        return _PreparedModule(island=None, native_readiness=tuple(native_readiness))
    island = EnabledIslandConfig(
        source_module=staged_module.name,
        source_path=staged_module.path,
        sidecar_module=sidecar_module,
        sidecar_path=sidecar_path,
        symbols=eligible_symbols,
    )
    sidecar = generate_sidecar(staged_scan, island)
    island.sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    island.sidecar_path.write_text(sidecar.source_text, encoding="utf-8")
    source_text = island.source_path.read_text(encoding="utf-8")
    island.source_path.write_text(
        insert_or_replace_shim(source_text, island).new_text,
        encoding="utf-8",
    )
    return _PreparedModule(island=island, native_readiness=tuple(native_readiness))


def _copy_source_roots(
    project: DiscoveredProject,
    build_root: Path,
) -> tuple[Path, ...]:
    staged_roots: list[Path] = []
    for source_root in project.config.source_roots:
        destination = build_root / _relative_source_root(project.config.root, source_root)
        if destination.resolve() == build_root.resolve():
            _copytree_contents(source_root, destination)
        else:
            shutil.copytree(source_root, destination, ignore=_copy_ignore)
        staged_roots.append(destination)
    return tuple(staged_roots)


def _place_compiled_artifacts(
    islands: tuple[EnabledIslandConfig, ...],
    artifact_paths: tuple[Path, ...],
) -> None:
    island_artifacts = {
        artifact
        for island in islands
        for artifact in artifact_paths
        if artifact.name.startswith(f"{island.sidecar_path.stem}.")
    }
    support_artifacts = tuple(
        artifact for artifact in artifact_paths if artifact not in island_artifacts
    )
    target_dirs = tuple(sorted({_artifact_dir(island) for island in islands}))
    for island in islands:
        target_dir = _artifact_dir(island)
        target_dir.mkdir(parents=True, exist_ok=True)
        for artifact in artifact_paths:
            if artifact.name.startswith(f"{island.sidecar_path.stem}."):
                _copy_if_different(artifact, target_dir / artifact.name)
    for target_dir in target_dirs:
        target_dir.mkdir(parents=True, exist_ok=True)
        for artifact in support_artifacts:
            _copy_if_different(artifact, target_dir / artifact.name)


def _source_clean_report_artifact_paths(
    root: Path,
    artifact_paths: tuple[Path, ...],
) -> tuple[Path, ...]:
    return tuple(root / ".atoll" / "artifacts" / artifact.name for artifact in artifact_paths)


def _artifact_dir(island: EnabledIslandConfig) -> Path:
    if island.sidecar_path.parent.name == "sidecars":
        return island.sidecar_path.parent.parent / "artifacts"
    return island.sidecar_path.parent


def _copy_if_different(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    shutil.copy2(source, destination)


def _remove_generated_sidecar_sources(islands: tuple[EnabledIslandConfig, ...]) -> None:
    for island in islands:
        island.sidecar_path.unlink(missing_ok=True)


def _remove_staged_island(island: EnabledIslandConfig) -> None:
    source_text = island.source_path.read_text(encoding="utf-8")
    island.source_path.write_text(remove_shim(source_text, island).new_text, encoding="utf-8")
    island.sidecar_path.unlink(missing_ok=True)


def _copy_install_payload(source_roots: tuple[Path, ...], install_root: Path) -> None:
    for source_root in source_roots:
        for path in sorted(source_root.rglob("*")):
            if not path.is_file() or _is_ignored_payload(path, source_root):
                continue
            if not _is_package_payload(path, source_root):
                continue
            relative = path.relative_to(source_root)
            destination = install_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def _copy_atoll_artifacts(source_roots: tuple[Path, ...], install_root: Path) -> None:
    for source_root in source_roots:
        artifact_root = source_root / ".atoll" / "artifacts"
        if not artifact_root.exists():
            continue
        for path in sorted(artifact_root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source_root)
            destination = install_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def _write_wheel(
    *,
    install_root: Path,
    output_dir: Path,
    metadata: _ProjectMetadata,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    wheel_tag = _wheel_tag()
    distribution = _wheel_safe_name(metadata.name)
    version = _wheel_safe_version(metadata.version)
    dist_info = f"{distribution}-{version}.dist-info"
    wheel_path = _wheel_output_path(output_dir, metadata)
    if wheel_path.exists():
        wheel_path.unlink()
    payload = _wheel_payload(install_root)
    metadata_files = {
        f"{dist_info}/METADATA": _metadata_text(metadata),
        f"{dist_info}/WHEEL": _wheel_text(wheel_tag),
    }
    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        records: list[tuple[str, str, str]] = []
        for relative, path in payload:
            data = path.read_bytes()
            wheel.writestr(relative, data)
            records.append((relative, _record_hash(data), str(len(data))))
        for relative, text in metadata_files.items():
            data = text.encode()
            wheel.writestr(relative, data)
            records.append((relative, _record_hash(data), str(len(data))))
        record_path = f"{dist_info}/RECORD"
        records.append((record_path, "", ""))
        wheel.writestr(record_path, _record_text(records))
    return wheel_path


def _wheel_output_path(output_dir: Path, metadata: _ProjectMetadata) -> Path:
    distribution = _wheel_safe_name(metadata.name)
    version = _wheel_safe_version(metadata.version)
    return output_dir / f"{distribution}-{version}-{_wheel_tag()}.whl"


def _remove_failed_wheels(project: DiscoveredProject, output_dir: Path) -> None:
    """Remove wheel artifacts that could be mistaken for the failed attempt."""
    metadata = _project_metadata(project.config.root)
    default_output = project.config.root / ".atoll" / "dist"
    if output_dir.resolve() != default_output.resolve():
        _wheel_output_path(output_dir, metadata).unlink(missing_ok=True)
        return
    distribution = _wheel_safe_name(metadata.name)
    version = _wheel_safe_version(metadata.version)
    for wheel_path in output_dir.glob(f"{distribution}-{version}-*.whl"):
        wheel_path.unlink()


def _wheel_payload(install_root: Path) -> tuple[tuple[str, Path], ...]:
    return tuple(
        (path.relative_to(install_root).as_posix(), path)
        for path in sorted(install_root.rglob("*"))
        if path.is_file()
    )


def _metadata_text(metadata: _ProjectMetadata) -> str:
    lines = [
        "Metadata-Version: 2.3",
        f"Name: {metadata.name}",
        f"Version: {metadata.version}",
        "Summary: Atoll compiled artifact",
    ]
    if metadata.requires_python is not None:
        lines.append(f"Requires-Python: {metadata.requires_python}")
    lines.extend(f"Requires-Dist: {dependency}" for dependency in metadata.dependencies)
    return "\n".join(lines) + "\n"


def _wheel_text(wheel_tag: str) -> str:
    return "\n".join(
        [
            "Wheel-Version: 1.0",
            "Generator: atoll",
            "Root-Is-Purelib: false",
            f"Tag: {wheel_tag}",
            "",
        ]
    )


def _record_text(records: list[tuple[str, str, str]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(records)
    return output.getvalue()


def _record_hash(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"sha256={encoded}"


def _project_metadata(root: Path) -> _ProjectMetadata:
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        return _ProjectMetadata(
            name=root.name,
            version="0+atoll",
            requires_python=None,
            dependencies=(),
        )
    data = cast(dict[str, object], tomllib.loads(pyproject.read_text(encoding="utf-8")))
    project = _mapping(data.get("project"))
    name = _string(project.get("name")) or root.name
    version = _string(project.get("version")) or "0+atoll"
    requires_python = _string(project.get("requires-python"))
    dependencies = tuple(
        dependency
        for item in _sequence(project.get("dependencies"))
        if (dependency := _string(item))
    )
    return _ProjectMetadata(
        name=name,
        version=version,
        requires_python=requires_python,
        dependencies=dependencies,
    )


def _wheel_tag() -> str:
    return str(next(tags.sys_tags()))


def _python_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def _wheel_safe_name(value: str) -> str:
    return re.sub(r"[-_.]+", "_", value).strip("_").lower()


def _wheel_safe_version(value: str) -> str:
    return value.replace("-", "_")


def _resolve_output_dir(root: Path, output_dir: Path | None) -> Path:
    if output_dir is None:
        return root / ".atoll" / "dist"
    if output_dir.is_absolute():
        return output_dir.resolve()
    return (root / output_dir).resolve()


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _copytree_contents(source: Path, destination: Path) -> None:
    ignored_names = _copy_ignore(str(source), [item.name for item in source.iterdir()])
    for item in source.iterdir():
        if item.name in ignored_names:
            continue
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, ignore=_copy_ignore)
        else:
            shutil.copy2(item, target)


def _copy_ignore(_directory: str, names: list[str]) -> set[str]:
    return {
        name for name in names if name in _GENERATED_DIR_NAMES or name.endswith((".so", ".pyd"))
    }


def _is_ignored_payload(path: Path, source_root: Path) -> bool:
    relative_parts = path.relative_to(source_root).parts
    return any(part in _GENERATED_DIR_NAMES or part == "tests" for part in relative_parts)


def _is_package_payload(path: Path, source_root: Path) -> bool:
    if path.name in {"py.typed", "__init__.py"}:
        return True
    if any(path.name.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES):
        return True
    if path.suffix not in {".py", ".pyi"}:
        return False
    relative = path.relative_to(source_root)
    if len(relative.parts) == 1:
        return True
    return any(
        (parent / "__init__.py").exists()
        for parent in path.parents
        if source_root in parent.parents
    )


def _staged_module(
    module: ModuleId,
    project: DiscoveredProject,
    staged_source_roots: tuple[Path, ...],
) -> ModuleId:
    staged_source_root = _staged_source_root(module, project, staged_source_roots)
    for source_root in project.config.source_roots:
        try:
            relative = module.path.relative_to(source_root)
        except ValueError:
            continue
        return ModuleId(name=module.name, path=staged_source_root / relative)
    raise ValueError(f"module is outside configured source roots: {module.name}")


def _staged_source_root(
    module: ModuleId,
    project: DiscoveredProject,
    staged_source_roots: tuple[Path, ...],
) -> Path:
    for index, source_root in enumerate(project.config.source_roots):
        try:
            module.path.relative_to(source_root)
        except ValueError:
            continue
        return staged_source_roots[index]
    raise ValueError(f"module is outside configured source roots: {module.name}")


def _relative_source_root(root: Path, source_root: Path) -> Path:
    try:
        return source_root.relative_to(root)
    except ValueError:
        return Path(f"source_{abs(hash(source_root))}")


def _find_module(modules: tuple[ModuleId, ...], module_name: str) -> ModuleId:
    for module in modules:
        if module.name == module_name:
            return module
    raise ValueError(f"module not found under configured source roots: {module_name}")


def _path_text(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _mapping(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        raw = cast(dict[object, object], value)
        return {str(key): item for key, item in raw.items()}
    return {}


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, list):
        return tuple(cast(list[object], value))
    return ()


def _string(value: object) -> str | None:
    return value if isinstance(value, str) else None
