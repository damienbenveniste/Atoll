"""Project discovery and module-name resolution for Atoll scans.

Discovery resolves source roots, filters generated/test directories, and returns
importable module names without importing target-project code. The resulting
configuration is shared by scan, generation, build, and verification commands.
"""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

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
    configured = _configured_source_roots(root)
    if configured:
        return configured
    return (root.resolve(),)


def _configured_source_roots(root: Path) -> tuple[Path, ...]:
    """Resolve import roots declared by common PEP 517 build backends.

    Atoll does not guess from arbitrary directory names. It accepts only
    relative, existing directories explicitly named by packaging metadata and
    ignores paths that escape the project. Invalid TOML remains the compile
    configuration loader's responsibility, so discovery falls back without
    hiding the later user-facing parse error.

    Args:
        root: Absolute target-project root containing optional packaging metadata.

    Returns:
        tuple[Path, ...]: Existing project-contained import roots in declaration order.
    """
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return ()
    try:
        data = cast(dict[str, object], tomllib.loads(pyproject.read_text(encoding="utf-8")))
    except tomllib.TOMLDecodeError:
        return ()
    tool = _mapping(data.get("tool"))
    specifications = (
        *_setuptools_source_root_specs(_mapping(tool.get("setuptools"))),
        *_hatch_source_root_specs(_mapping(tool.get("hatch"))),
        *_poetry_source_root_specs(_mapping(tool.get("poetry"))),
        *_single_source_root_spec(_mapping(_mapping(tool.get("pdm")).get("build")), "package-dir"),
        *_single_source_root_spec(_mapping(tool.get("maturin")), "python-source"),
    )
    roots: list[Path] = []
    for specification in specifications:
        candidate = _packaging_source_root(root, specification)
        if candidate is not None and candidate not in roots:
            roots.append(candidate)
    return tuple(roots)


def _setuptools_source_root_specs(setuptools: dict[str, object]) -> tuple[str, ...]:
    """Extract Setuptools package discovery roots from parsed TOML.

    Args:
        setuptools: Parsed ``[tool.setuptools]`` table.

    Returns:
        tuple[str, ...]: Relative import-root declarations.
    """
    find = _mapping(_mapping(setuptools.get("packages")).get("find"))
    where = _string_sequence(find.get("where"))
    package_dir = _mapping(setuptools.get("package-dir"))
    if not package_dir:
        package_dir = _mapping(setuptools.get("package_dir"))
    default_root = package_dir.get("")
    return (*where, *((default_root,) if isinstance(default_root, str) else ()))


def _hatch_source_root_specs(hatch: dict[str, object]) -> tuple[str, ...]:
    """Extract Hatch wheel source roots without interpreting build hooks.

    Args:
        hatch: Parsed ``[tool.hatch]`` table.

    Returns:
        tuple[str, ...]: Relative import-root declarations.
    """
    build = _mapping(hatch.get("build"))
    wheel = _mapping(_mapping(_mapping(build.get("targets")).get("wheel")))
    source_configuration = wheel.get("sources", build.get("sources"))
    sources = _hatch_source_roots(source_configuration)
    package_parents = tuple(
        str(Path(package).parent)
        for package in _string_sequence(wheel.get("packages"))
        if not any(marker in package for marker in "*?[")
    )
    return (*sources, *package_parents)


def _hatch_source_roots(value: object) -> tuple[str, ...]:
    """Extract source roots from Hatch prefix-removal or mapping syntax.

    Mapping entries are accepted only when the destination suffix matches the
    source suffix, because only those rewrites can be represented by Atoll's
    import-root model without renaming a package.

    Args:
        value: Parsed Hatch ``sources`` list or mapping.

    Returns:
        tuple[str, ...]: Relative roots that preserve installed import names.
    """
    sequence = _string_sequence(value)
    if sequence:
        return sequence
    roots: list[str] = []
    for source, destination in _mapping(value).items():
        if not isinstance(destination, str):
            continue
        source_parts = Path(source).parts
        destination_parts = () if destination in {"", "/"} else Path(destination).parts
        if destination_parts and source_parts[-len(destination_parts) :] != destination_parts:
            continue
        root_parts = source_parts[: -len(destination_parts)] if destination_parts else source_parts
        root = str(Path(*root_parts)) if root_parts else "."
        if root not in roots:
            roots.append(root)
    return tuple(roots)


def _poetry_source_root_specs(poetry: dict[str, object]) -> tuple[str, ...]:
    """Extract explicit Poetry ``from`` roots for included packages.

    Args:
        poetry: Parsed ``[tool.poetry]`` table.

    Returns:
        tuple[str, ...]: Relative import-root declarations.
    """
    roots: list[str] = []
    for package in _object_sequence(poetry.get("packages")):
        package_root = _mapping(package).get("from")
        if isinstance(package_root, str):
            roots.append(package_root)
    return tuple(roots)


def _single_source_root_spec(table: dict[str, object], key: str) -> tuple[str, ...]:
    value = table.get(key)
    return (value,) if isinstance(value, str) else ()


def _packaging_source_root(root: Path, specification: str) -> Path | None:
    path = Path(specification)
    if path.is_absolute():
        return None
    candidate = (root / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate if candidate.is_dir() else None


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return cast(dict[str, object], value)


def _object_sequence(value: object) -> tuple[object, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(cast(Sequence[object], value))


def _string_sequence(value: object) -> tuple[str, ...]:
    return tuple(item for item in _object_sequence(value) if isinstance(item, str))


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
