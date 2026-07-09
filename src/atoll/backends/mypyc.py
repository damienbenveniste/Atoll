"""Programmatic mypyc build backend for generated Atoll sidecars.

The backend invokes mypyc and setuptools in-process so command handlers can
capture structured build evidence. It isolates mypy cache and import-path state
around each build and records native stderr because mypyc can write diagnostics
outside Python's normal `sys.stderr` object.
"""

from __future__ import annotations

import importlib.machinery
import io
import os
import sys
import tempfile
import time
from collections.abc import Generator
from contextlib import chdir, contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import BinaryIO

from mypyc.build import mypycify
from setuptools import Distribution
from setuptools.command.build_ext import build_ext

from atoll.models import CompileAttempt

_MAX_DIAGNOSTIC_LINES = 20


def build_sidecars(
    paths: tuple[Path, ...],
    *,
    project_root: Path,
    build_dir: Path,
    source_roots: tuple[Path, ...] = (),
) -> CompileAttempt:
    """Compile generated sidecar source files into Atoll's artifact directory.

    Empty input is a successful no-op. Build failures are converted into
    `CompileAttempt` values with classified diagnostics instead of escaping as
    exceptions through CLI command handlers.
    """
    start = time.perf_counter()
    if not paths:
        return CompileAttempt(
            success=True,
            command=("mypyc", "build_ext"),
            stdout="no enabled Atoll sidecars to build",
            stderr="",
            artifact_paths=(),
            duration_seconds=time.perf_counter() - start,
        )
    build_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = build_dir.parent / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    previous_artifacts = {
        path: path.stat().st_mtime_ns for path in _all_extension_artifacts(artifact_dir)
    }
    command = ("mypyc", *tuple(str(path) for path in paths), "build_ext")
    stdout = io.StringIO()
    stderr = io.StringIO()
    native_stderr = _NativeStderrCapture()
    try:
        with (
            chdir(project_root),
            _mypy_environment(source_roots, build_dir / "mypy_cache"),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
            native_stderr,
        ):
            ext_modules = mypycify(
                [_source_arg(path, project_root) for path in paths],
                target_dir=str(build_dir / "generated"),
            )
            distribution = Distribution(
                _distribution_attrs(
                    project_root=project_root,
                    source_roots=source_roots,
                    ext_modules=ext_modules,
                )
            )
            command_obj = build_ext(distribution)
            command_obj.inplace = False
            command_obj.build_lib = str(artifact_dir)
            command_obj.build_temp = str(build_dir / "temp")
            command_obj.ensure_finalized()
            command_obj.run()
    except SystemExit as error:
        diagnostics = _captured_output(stdout, stderr, native_stderr.output())
        log_path = _write_build_log(build_dir, diagnostics, error)
        return CompileAttempt(
            success=False,
            command=command,
            stdout="",
            stderr=_classify_build_error(error, diagnostics=diagnostics, log_path=log_path),
            artifact_paths=(),
            duration_seconds=time.perf_counter() - start,
        )
    except Exception as error:
        diagnostics = _captured_output(stdout, stderr, native_stderr.output())
        log_path = _write_build_log(build_dir, diagnostics, error)
        return CompileAttempt(
            success=False,
            command=command,
            stdout="",
            stderr=_classify_build_error(error, diagnostics=diagnostics, log_path=log_path),
            artifact_paths=(),
            duration_seconds=time.perf_counter() - start,
        )
    artifacts = _artifact_paths(paths, artifact_dir, previous_artifacts)
    diagnostics = _captured_output(stdout, stderr, native_stderr.output())
    if diagnostics:
        _write_build_log(build_dir, diagnostics, None)
    return CompileAttempt(
        success=bool(artifacts),
        command=command,
        stdout=diagnostics,
        stderr="" if artifacts else "mypyc build completed but no extension artifacts were found",
        artifact_paths=artifacts,
        duration_seconds=time.perf_counter() - start,
    )


class _NativeStderrCapture:
    """Capture writes to file descriptor 2 during in-process native builds."""

    def __init__(self) -> None:
        """Initialize the saved descriptor, temporary file, and captured buffer."""
        self._saved_fd: int | None = None
        self._file: BinaryIO | None = None
        self._captured = ""

    def __enter__(self) -> _NativeStderrCapture:
        """Redirect native stderr to a temporary file for the build duration."""
        sys.stderr.flush()
        self._file = tempfile.TemporaryFile(mode="w+b")
        self._saved_fd = os.dup(2)
        os.dup2(self._file.fileno(), 2)
        return self

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        """Restore native stderr and load captured bytes as replacement text."""
        _ = (exc_type, exc, traceback)
        sys.stderr.flush()
        if self._saved_fd is not None:
            os.dup2(self._saved_fd, 2)
            os.close(self._saved_fd)
            self._saved_fd = None
        if self._file is not None:
            self._file.flush()
            self._file.seek(0)
            self._captured = self._file.read().decode("utf-8", errors="replace")
            self._file.close()
            self._file = None

    def output(self) -> str:
        """Return captured native stderr after the context has exited."""
        return self._captured


def _artifact_paths(
    paths: tuple[Path, ...],
    artifact_dir: Path,
    previous_artifacts: dict[Path, int],
) -> tuple[Path, ...]:
    artifacts: set[Path] = set()
    for path in paths:
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            artifacts.update(artifact_dir.rglob(f"{path.stem}*{suffix}"))
    for path in _all_extension_artifacts(artifact_dir):
        previous_mtime = previous_artifacts.get(path)
        if previous_mtime is None or path.stat().st_mtime_ns != previous_mtime:
            artifacts.add(path)
    return tuple(sorted(artifacts))


def _all_extension_artifacts(artifact_dir: Path) -> tuple[Path, ...]:
    artifacts: set[Path] = set()
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        artifacts.update(artifact_dir.rglob(f"*{suffix}"))
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


@contextmanager
def _mypy_environment(
    source_roots: tuple[Path, ...],
    cache_dir: Path,
) -> Generator[None]:
    paths = tuple(os.fspath(path.resolve()) for path in source_roots)
    original_mypy_path = os.environ.get("MYPYPATH")
    original_cache_dir = os.environ.get("MYPY_CACHE_DIR")
    if paths:
        values = (*paths, original_mypy_path) if original_mypy_path else paths
        os.environ["MYPYPATH"] = os.pathsep.join(values)
    os.environ["MYPY_CACHE_DIR"] = os.fspath(cache_dir)
    try:
        yield
    finally:
        if original_mypy_path is None:
            os.environ.pop("MYPYPATH", None)
        else:
            os.environ["MYPYPATH"] = original_mypy_path
        if original_cache_dir is None:
            os.environ.pop("MYPY_CACHE_DIR", None)
        else:
            os.environ["MYPY_CACHE_DIR"] = original_cache_dir


def _captured_output(stdout: io.StringIO, stderr: io.StringIO, native_stderr: str = "") -> str:
    return "\n".join(
        line
        for value in (stdout.getvalue(), stderr.getvalue(), native_stderr)
        for line in _diagnostic_lines(value)
    )


def _diagnostic_lines(value: str) -> tuple[str, ...]:
    return tuple(
        line.rstrip()
        for line in value.splitlines()
        if line.strip() and not _is_ignored_diagnostic(line.strip())
    )


def _is_ignored_diagnostic(line: str) -> bool:
    return line.startswith("ld: warning: duplicate -rpath ") and line.endswith(" ignored")


def _write_build_log(
    build_dir: Path,
    diagnostics: str,
    error: BaseException | None,
) -> Path | None:
    if not diagnostics and error is None:
        return None
    log_path = build_dir / "mypyc.log"
    lines = ["# Atoll mypyc build log", ""]
    if error is not None:
        lines.extend([f"exception: {type(error).__name__}: {error!r}", ""])
    if diagnostics:
        lines.extend(["diagnostics:", diagnostics, ""])
    log_path.write_text("\n".join(lines), encoding="utf-8")
    return log_path


def _classify_build_error(
    error: BaseException,
    *,
    diagnostics: str,
    log_path: Path | None,
) -> str:
    message = repr(error)
    lowered = f"{message}\n{diagnostics}".lower()
    detail = _diagnostic_summary(diagnostics, log_path)
    if "no such file" in lowered or "compiler" in lowered or "clang" in lowered or "gcc" in lowered:
        return f"NATIVE_BUILD_ENV_ERROR: {message}{detail}"
    if "mypy" in lowered or "type" in lowered or ": error:" in lowered:
        return f"MYPYC_TYPE_ERROR: {message}{detail}"
    if "import" in lowered or "module" in lowered:
        return f"IMPORT_PATH_ERROR: {message}{detail}"
    return f"UNKNOWN_BUILD_ERROR: {message}{detail}"


def _diagnostic_summary(diagnostics: str, log_path: Path | None) -> str:
    lines = [line for line in diagnostics.splitlines() if line.strip()]
    if not lines:
        return ""
    error_lines = [line for line in lines if ": error:" in line]
    selected = error_lines[:_MAX_DIAGNOSTIC_LINES] or lines[:_MAX_DIAGNOSTIC_LINES]
    omitted = max((len(error_lines) or len(lines)) - len(selected), 0)
    parts = [
        "",
        f"Captured {len(error_lines)} mypyc error line(s).",
    ]
    if log_path is not None:
        parts.append(f"Full diagnostics: {log_path}")
    if "[import-not-found]" in diagnostics:
        parts.append(
            "Hint: run `atoll build` from the target project's environment so mypyc "
            "can import the target package dependencies."
        )
    parts.extend(selected)
    if omitted:
        parts.append(f"... {omitted} more line(s) in the build log")
    return "\n".join(parts)
