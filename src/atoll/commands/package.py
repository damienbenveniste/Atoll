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
from atoll.generation.shim import insert_or_replace_shim
from atoll.generation.sidecar import default_sidecar_module, generate_sidecar
from atoll.models import CompileAttempt, EnabledIslandConfig, ModuleId, ModuleScan
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


def execute_package(options: PackageOptions) -> PackageCommandResult:
    """Build an install tree and wheel containing Atoll compiled islands."""
    project = discover_project(options.root)
    selected = _selected_modules(project, options.module_name)
    if not selected:
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
    build = build_sidecars(
        tuple(island.sidecar_path for island in islands),
        project_root=build_root,
        build_dir=build_root / ".atoll" / "build",
        source_roots=staged_source_roots,
    )
    if not build.success:
        return PackageCommandResult(
            success=False,
            output_dir=output_dir,
            install_root=install_root,
            wheel_path=None,
            islands=islands,
            build=build,
            error=build.stderr,
        )

    _place_compiled_artifacts(islands, build.artifact_paths)
    _remove_generated_sidecar_sources(islands)
    _copy_install_payload(staged_source_roots, install_root)
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
        islands=islands,
        build=build,
    )


def _failed_result(
    root: Path,
    output_dir: Path | None,
    error: str,
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
    )


def _selected_modules(
    project: DiscoveredProject,
    module_name: str | None,
) -> tuple[_SelectedModule, ...]:
    modules = (_find_module(project.modules, module_name),) if module_name else project.modules
    selected: list[_SelectedModule] = []
    for module in modules:
        scan = enrich_island_analysis(scan_module(module))
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
    staged_scan = enrich_island_analysis(scan_module(staged_module))
    sidecar_module = default_sidecar_module(staged_module.name)
    sidecar_name = sidecar_module.rsplit(".", maxsplit=1)[-1]
    island = EnabledIslandConfig(
        source_module=staged_module.name,
        source_path=staged_module.path,
        sidecar_module=sidecar_module,
        sidecar_path=staged_module.path.parent / f"{sidecar_name}.py",
        symbols=selected_module.symbols,
    )
    sidecar = generate_sidecar(staged_scan, island)
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
    package_dirs = tuple(sorted({island.source_path.parent for island in islands}))
    for island in islands:
        for artifact in artifact_paths:
            if artifact.name.startswith(f"{island.sidecar_path.stem}."):
                shutil.copy2(artifact, island.source_path.parent / artifact.name)
    for package_dir in package_dirs:
        for artifact in support_artifacts:
            shutil.copy2(artifact, package_dir / artifact.name)


def _remove_generated_sidecar_sources(islands: tuple[EnabledIslandConfig, ...]) -> None:
    for island in islands:
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
    for index, source_root in enumerate(project.config.source_roots):
        try:
            relative = module.path.relative_to(source_root)
        except ValueError:
            continue
        return ModuleId(name=module.name, path=staged_source_roots[index] / relative)
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
