"""Project discovery and module-name resolution for Atoll scans.

Discovery resolves source roots, filters generated/test directories, and returns
importable module names without importing target-project code. The resulting
configuration is shared by scan, generation, build, and verification commands.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from atoll.config import load_compile_config, load_enabled_islands
from atoll.models import ModuleId, ProjectConfig

IGNORED_DIR_NAMES = frozenset(
    {
        ".atoll",
        ".mypy_cache",
        ".nox",
        ".pyislands",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "site-packages",
        "venv",
    }
)


@dataclass(frozen=True, slots=True)
class DiscoveredProject:
    """Resolved project configuration plus importable modules.

    `modules` is sorted by import name for deterministic scans and reports. The
    object is immutable so command handlers can pass it between phases safely.

    Attributes:
        config: Absolute project paths and persisted Atoll policy.
        modules: Importable Python modules sorted by dotted name.
    """

    config: ProjectConfig
    modules: tuple[ModuleId, ...]


def resolve_project_config(root: Path, source_roots: Sequence[Path] = ()) -> ProjectConfig:
    """Resolve source roots and Atoll output directories for `root`.

    Args:
        root: Root directory of the target Python project.
        source_roots: Import roots made visible to analysis or child processes.

    Returns:
        ProjectConfig: Normalized project configuration with absolute paths.
    """
    project_root = root.resolve()
    roots = _resolve_source_roots(project_root, source_roots)
    return ProjectConfig(
        root=project_root,
        source_roots=roots,
        backend="mypyc",
        cache_dir=project_root / ".atoll" / "cache",
        report_dir=project_root / ".atoll",
        islands=load_enabled_islands(project_root),
        compile=load_compile_config(project_root),
    )


def discover_project(
    root: Path,
    *,
    source_roots: Sequence[Path] = (),
    max_files: int | None = None,
) -> DiscoveredProject:
    """Discover Python modules under Atoll source roots.

    Generated caches, build outputs, virtual environments, and tests are skipped.
    `max_files` provides a deterministic early stop for exploratory scans.

    Args:
        root: Root directory of the target Python project.
        source_roots: Import roots made visible to analysis or child processes.
        max_files: Optional upper bound on discovered Python files.

    Returns:
        DiscoveredProject: Resolved configuration and deterministically discovered Python modules.
    """
    config = resolve_project_config(root, source_roots)
    modules = tuple(_iter_modules(config.source_roots, max_files=max_files))
    return DiscoveredProject(config=config, modules=modules)


def module_name_for_path(path: Path, source_root: Path) -> str:
    """Return the import name for `path` relative to `source_root`.

    Args:
        path: Filesystem path consumed or produced by the operation.
        source_root: Import root used to derive a dotted module name.

    Returns:
        str: Importable dotted module name relative to the source root.

    Raises:
        ValueError: If `path` is outside `source_root`.
    """
    relative = path.resolve().relative_to(source_root.resolve())
    if relative.name == "__init__.py":
        parts = relative.parts[:-1]
    else:
        parts = (*relative.parts[:-1], relative.stem)
    return ".".join(parts)


def _resolve_source_roots(root: Path, source_roots: Sequence[Path]) -> tuple[Path, ...]:
    if source_roots:
        return tuple(_resolve_path(root, source_root) for source_root in source_roots)
    src = root / "src"
    if src.is_dir():
        return (src.resolve(),)
    return (root.resolve(),)


def _resolve_path(root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def _iter_modules(source_roots: tuple[Path, ...], *, max_files: int | None) -> tuple[ModuleId, ...]:
    modules: list[ModuleId] = []
    for source_root in source_roots:
        for path in sorted(source_root.rglob("*.py")):
            if _is_ignored_path(path, source_root):
                continue
            module_name = module_name_for_path(path, source_root)
            if module_name:
                modules.append(ModuleId(name=module_name, path=path.resolve()))
            if max_files is not None and len(modules) >= max_files:
                return tuple(sorted(modules, key=lambda module: module.name))
    return tuple(sorted(modules, key=lambda module: module.name))


def _is_ignored_path(path: Path, source_root: Path) -> bool:
    relative_parts = path.relative_to(source_root).parts
    if any(part in IGNORED_DIR_NAMES for part in relative_parts):
        return True
    return "tests" in relative_parts or path.name.startswith("test_")
