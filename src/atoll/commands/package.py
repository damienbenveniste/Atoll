"""Build installable Atoll artifacts without modifying source files."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.machinery
import io
import re
import shutil
import sys
import sysconfig
import tomllib
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.clustering import enrich_island_analysis
from atoll.backends.mypyc import build_sidecars
from atoll.generation.shim import insert_or_replace_shim, remove_shim
from atoll.generation.sidecar import default_sidecar_module, expected_sidecar_path, generate_sidecar
from atoll.models import Blocker, CompileAttempt, EnabledIslandConfig, ModuleId, ModuleScan
from atoll.project import DiscoveredProject, discover_project

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
_MYPYC_PREFLIGHT_BLOCKERS = frozenset({"MYPYC_UNSUPPORTED_TYPEVAR"})
_MAX_PREFLIGHT_BLOCKERS = 12


@dataclass(frozen=True, slots=True)
class PackageOptions:
    """User-facing options for building an installable Atoll artifact."""

    root: Path
    module_name: str | None = None
    output_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class PackageCommandResult:
    """Result from building a source-clean Atoll package artifact."""

    success: bool
    output_dir: Path
    install_root: Path
    wheel_path: Path | None
    islands: tuple[EnabledIslandConfig, ...]
    build: CompileAttempt
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
class _ModuleBlocker:
    scan: ModuleScan
    blocker: Blocker


@dataclass(frozen=True, slots=True)
class _PackageBuildOutcome:
    successful: tuple[EnabledIslandConfig, ...]
    build: CompileAttempt
    skipped: tuple[PackageBuildFailure, ...]


def execute_package(options: PackageOptions) -> PackageCommandResult:
    """Build an install tree and wheel containing Atoll compiled islands."""
    project = discover_project(options.root)
    scans = _selected_scans(project, options.module_name)
    preflight_skipped = _mypyc_preflight_failures(scans)
    buildable_scans = _buildable_scans(scans, preflight_skipped)
    if options.module_name is not None and preflight_skipped:
        return _failed_result(
            project.config.root,
            options.output_dir,
            _format_mypyc_preflight_error(_module_blockers(preflight_skipped)),
            preflight_skipped=preflight_skipped,
        )
    selected = _selected_modules(buildable_scans)
    if not selected:
        if preflight_skipped:
            return _failed_result(
                project.config.root,
                options.output_dir,
                _format_mypyc_preflight_error(_module_blockers(preflight_skipped)),
                preflight_skipped=preflight_skipped,
            )
        return _failed_result(
            project.config.root, options.output_dir, "scan found no candidate islands"
        )

    output_dir = _resolve_output_dir(project.config.root, options.output_dir)
    build_root = output_dir / "build"
    install_root = output_dir / "install"
    _reset_dir(build_root)
    _reset_dir(install_root)

    staged_source_roots = _copy_source_roots(project, build_root)
    islands = tuple(
        _prepare_staged_island(
            project=project,
            staged_source_roots=staged_source_roots,
            selected_module=selected_module,
        )
        for selected_module in selected
    )
    outcome = _build_package_islands(
        islands,
        project_root=build_root,
        source_roots=staged_source_roots,
        allow_partial=options.module_name is None,
    )
    if not outcome.build.success:
        return PackageCommandResult(
            success=False,
            output_dir=output_dir,
            install_root=install_root,
            wheel_path=None,
            islands=outcome.successful,
            build=outcome.build,
            error=outcome.build.stderr,
            skipped=outcome.skipped,
            preflight_skipped=preflight_skipped,
        )

    _place_compiled_artifacts(outcome.successful, outcome.build.artifact_paths)
    _remove_generated_sidecar_sources(outcome.successful)
    _copy_install_payload(staged_source_roots, install_root)
    _copy_atoll_artifacts(staged_source_roots, install_root)
    metadata = _project_metadata(project.config.root)
    wheel_path = _write_wheel(
        install_root=install_root,
        output_dir=output_dir,
        metadata=metadata,
    )
    shutil.rmtree(build_root)
    return PackageCommandResult(
        success=True,
        output_dir=output_dir,
        install_root=install_root,
        wheel_path=wheel_path,
        islands=outcome.successful,
        build=outcome.build,
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


def _build_package_islands(
    islands: tuple[EnabledIslandConfig, ...],
    *,
    project_root: Path,
    source_roots: tuple[Path, ...],
    allow_partial: bool,
) -> _PackageBuildOutcome:
    batch = build_sidecars(
        tuple(island.sidecar_path for island in islands),
        project_root=project_root,
        build_dir=project_root / ".atoll" / "build",
        source_roots=source_roots,
    )
    if batch.success:
        return _PackageBuildOutcome(successful=islands, build=batch, skipped=())
    if not allow_partial or len(islands) <= 1:
        return _PackageBuildOutcome(successful=(), build=batch, skipped=())
    return _build_package_islands_individually(
        islands,
        project_root=project_root,
        source_roots=source_roots,
        batch_failure=batch,
    )


def _build_package_islands_individually(
    islands: tuple[EnabledIslandConfig, ...],
    *,
    project_root: Path,
    source_roots: tuple[Path, ...],
    batch_failure: CompileAttempt,
) -> _PackageBuildOutcome:
    successful: list[EnabledIslandConfig] = []
    skipped: list[PackageBuildFailure] = []
    attempts: list[CompileAttempt] = []
    for island in islands:
        attempt = build_sidecars(
            (island.sidecar_path,),
            project_root=project_root,
            build_dir=project_root / ".atoll" / "retry-builds" / island.sidecar_path.stem,
            source_roots=source_roots,
        )
        attempts.append(attempt)
        if attempt.success:
            successful.append(island)
            continue
        skipped.append(PackageBuildFailure(island=island, build=attempt))
        _remove_staged_island(island)
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


def _mypyc_preflight_failures(scans: tuple[ModuleScan, ...]) -> tuple[PackagePreflightFailure, ...]:
    return tuple(
        PackagePreflightFailure(scan=scan, blockers=blockers)
        for scan in scans
        if (
            blockers := tuple(
                blocker
                for blocker in scan.blockers
                if blocker.severity == "hard" and blocker.code in _MYPYC_PREFLIGHT_BLOCKERS
            )
        )
    )


def _buildable_scans(
    scans: tuple[ModuleScan, ...],
    skipped: tuple[PackagePreflightFailure, ...],
) -> tuple[ModuleScan, ...]:
    skipped_modules = {failure.scan.module.name for failure in skipped}
    return tuple(scan for scan in scans if scan.module.name not in skipped_modules)


def _module_blockers(skipped: tuple[PackagePreflightFailure, ...]) -> tuple[_ModuleBlocker, ...]:
    return tuple(
        _ModuleBlocker(scan=failure.scan, blocker=blocker)
        for failure in skipped
        for blocker in failure.blockers
    )


def _format_mypyc_preflight_error(blockers: tuple[_ModuleBlocker, ...]) -> str:
    shown = blockers[:_MAX_PREFLIGHT_BLOCKERS]
    lines = [
        "Atoll cannot compile the selected project because mypy/mypyc rejects typing constructs "
        "used before generated islands can be built.",
        "Unsupported typing blockers:",
    ]
    lines.extend(_format_mypyc_preflight_blocker(item) for item in shown)
    remaining = len(blockers) - len(shown)
    if remaining > 0:
        lines.append(f"... {remaining} more blocker(s)")
    lines.append(
        "This is a target-project typing compatibility issue, not an Atoll source-edit issue."
    )
    return "\n".join(lines)


def _format_mypyc_preflight_blocker(item: _ModuleBlocker) -> str:
    location = str(item.scan.module.path)
    if item.blocker.lineno is not None:
        location = f"{location}:{item.blocker.lineno}"
    return f"- {location}: {item.blocker.message}"


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
    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform_tag = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    return f"{python_tag}-{python_tag}-{platform_tag}"


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
