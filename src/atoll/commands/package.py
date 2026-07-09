"""Build installable Atoll artifacts without modifying source files."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.machinery
import io
import json
import re
import shutil
import sys
import sysconfig
import time
import tomllib
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import cast

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.clustering import enrich_island_analysis
from atoll.backends.mypyc import build_sidecars
from atoll.generation.shim import insert_or_replace_shim, remove_shim
from atoll.generation.sidecar import (
    SIDECAR_GENERATOR_VERSION,
    default_sidecar_module,
    expected_sidecar_path,
    generate_sidecar,
)
from atoll.models import (
    Blocker,
    CompileAttempt,
    CompilePhaseTiming,
    EnabledIslandConfig,
    ModuleId,
    ModuleScan,
)
from atoll.project import DiscoveredProject, discover_project

PackageProgress = Callable[[str], None]

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
_COMPILE_CACHE_VERSION = 1
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
    _progress(options.progress, f"scanned {len(scans)} module(s) in {_duration(scan_started)}")
    preflight_skipped: tuple[PackagePreflightFailure, ...] = ()
    selected = _selected_modules(scans)
    selected_symbols = sum(len(selected_module.symbols) for selected_module in selected)
    _progress(
        options.progress,
        f"selected {len(selected)} candidate module(s), {selected_symbols} symbol(s)",
    )
    if not selected:
        return _failed_result(
            project.config.root, options.output_dir, "scan found no candidate islands"
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
    _progress(options.progress, f"generating {len(selected)} sidecar module(s)")
    islands = tuple(
        _prepare_staged_island(
            project=project,
            staged_source_roots=staged_source_roots,
            selected_module=selected_module,
        )
        for selected_module in selected
    )
    _progress(options.progress, f"generated sidecars in {_duration(sidecar_started)}")
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
            preflight_skipped=preflight_skipped,
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
        preflight_skipped=preflight_skipped,
    )


def _failed_result(
    root: Path,
    output_dir: Path | None,
    error: str,
    *,
    preflight_skipped: tuple[PackagePreflightFailure, ...] = (),
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
    )


def _remove_tree(path: Path) -> tuple[Path, ...]:
    if not path.exists():
        return ()
    shutil.rmtree(path)
    return (path,)


def _progress(progress: PackageProgress | None, message: str) -> None:
    if progress is not None:
        progress(message)


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
    lookup_timing = CompilePhaseTiming(
        name="cache_lookup",
        duration_seconds=time.perf_counter() - lookup_started,
        detail="hit",
    )
    restore_started = time.perf_counter()
    restored = tuple(path for path in artifacts if path.exists())
    restore_timing = CompilePhaseTiming(
        name="cache_restore",
        duration_seconds=time.perf_counter() - restore_started,
        detail=f"{len(restored)} artifact(s)",
    )
    if len(restored) != len(artifacts):
        return _CompileCacheLookup(
            key=key,
            hit=False,
            artifact_paths=(),
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


def _store_compile_cache(
    *,
    cache_root: Path,
    key: str,
    artifact_paths: tuple[Path, ...],
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
        return _PackageBuildOutcome(
            successful=islands,
            build=CompileAttempt(
                success=True,
                command=("atoll", "compile-cache", "restore", cache_lookup.key[:12]),
                stdout="compile cache hit",
                stderr="",
                artifact_paths=cache_lookup.artifact_paths,
                duration_seconds=sum(
                    timing.duration_seconds for timing in cache_lookup.phase_timings
                ),
                phase_timings=cache_lookup.phase_timings,
                cache_status="hit",
            ),
            skipped=(),
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
    return _build_package_islands_individually(
        islands,
        context,
        batch_failure=batch,
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
) -> EnabledIslandConfig:
    staged_module = _staged_module(selected_module.scan.module, project, staged_source_roots)
    staged_source_root = _staged_source_root(
        selected_module.scan.module,
        project,
        staged_source_roots,
    )
    staged_scan = enrich_island_analysis(scan_module(staged_module))
    sidecar_module = default_sidecar_module(staged_module.name)
    island = EnabledIslandConfig(
        source_module=staged_module.name,
        source_path=staged_module.path,
        sidecar_module=sidecar_module,
        sidecar_path=expected_sidecar_path(staged_source_root, sidecar_module),
        symbols=selected_module.symbols,
    )
    sidecar = generate_sidecar(staged_scan, island)
    island.sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    island.sidecar_path.write_text(sidecar.source_text, encoding="utf-8")
    source_text = island.source_path.read_text(encoding="utf-8")
    island.source_path.write_text(
        insert_or_replace_shim(source_text, island).new_text,
        encoding="utf-8",
    )
    return island


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
    wheel_path = output_dir / f"{distribution}-{version}-{wheel_tag}.whl"
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
    python_tag = _python_tag()
    platform_tag = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    return f"{python_tag}-{python_tag}-{platform_tag}"


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
