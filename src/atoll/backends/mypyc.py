"""Programmatic mypyc build backend for generated Atoll sidecars."""

from __future__ import annotations

import importlib.machinery
import time
from contextlib import chdir
from pathlib import Path

from mypyc.build import mypycify
from setuptools import Distribution
from setuptools.command.build_ext import build_ext

from atoll.models import CompileAttempt


def build_sidecars(
    paths: tuple[Path, ...],
    *,
    project_root: Path,
    build_dir: Path,
    source_roots: tuple[Path, ...] = (),
) -> CompileAttempt:
    """Compile generated sidecar source files in place using mypyc."""
    start = time.perf_counter()
    if not paths:
        return CompileAttempt(
            success=True,
            command=("mypyc", "build_ext", "--inplace"),
            stdout="no enabled Atoll sidecars to build",
            stderr="",
            artifact_paths=(),
            duration_seconds=time.perf_counter() - start,
        )
    build_dir.mkdir(parents=True, exist_ok=True)
    command = ("mypyc", *tuple(str(path) for path in paths), "build_ext", "--inplace")
    try:
        with chdir(project_root):
            ext_modules = mypycify([_source_arg(path, project_root) for path in paths])
            distribution = Distribution(
                _distribution_attrs(
                    project_root=project_root,
                    source_roots=source_roots,
                    ext_modules=ext_modules,
                )
            )
            command_obj = build_ext(distribution)
            command_obj.inplace = True
            command_obj.build_temp = str(build_dir / "temp")
            command_obj.ensure_finalized()
            command_obj.run()
    except Exception as error:
        return CompileAttempt(
            success=False,
            command=command,
            stdout="",
            stderr=_classify_build_error(error),
            artifact_paths=(),
            duration_seconds=time.perf_counter() - start,
        )
    artifacts = _artifact_paths(paths)
    return CompileAttempt(
        success=bool(artifacts),
        command=command,
        stdout="",
        stderr="" if artifacts else "mypyc build completed but no extension artifacts were found",
        artifact_paths=artifacts,
        duration_seconds=time.perf_counter() - start,
    )


def _artifact_paths(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    artifacts: set[Path] = set()
    for path in paths:
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            artifacts.update(path.parent.glob(f"{path.stem}*{suffix}"))
    return tuple(sorted(artifacts))


def _source_arg(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _distribution_attrs(
    *,
    project_root: Path,
    source_roots: tuple[Path, ...],
    ext_modules: object,
) -> dict[str, object]:
    attrs: dict[str, object] = {"name": "atoll_generated", "ext_modules": ext_modules}
    package_root = _package_root(project_root, source_roots)
    if package_root is not None:
        attrs["package_dir"] = {"": package_root}
    return attrs


def _package_root(project_root: Path, source_roots: tuple[Path, ...]) -> str | None:
    if not source_roots:
        return None
    first = source_roots[0].resolve()
    try:
        relative = first.relative_to(project_root.resolve())
    except ValueError:
        return None
    text = relative.as_posix()
    return text if text != "." else None


def _classify_build_error(error: Exception) -> str:
    message = repr(error)
    lowered = message.lower()
    if "no such file" in lowered or "compiler" in lowered or "clang" in lowered or "gcc" in lowered:
        return f"NATIVE_BUILD_ENV_ERROR: {message}"
    if "mypy" in lowered or "type" in lowered:
        return f"MYPYC_TYPE_ERROR: {message}"
    if "import" in lowered or "module" in lowered:
        return f"IMPORT_PATH_ERROR: {message}"
    return f"UNKNOWN_BUILD_ERROR: {message}"
