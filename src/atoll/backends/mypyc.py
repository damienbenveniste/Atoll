"""Programmatic mypyc build backend for generated Atoll sidecars.

The backend invokes mypyc and setuptools in-process so command handlers can
capture structured build evidence. It isolates mypy cache and import-path state
around each build and records native stderr because mypyc can write diagnostics
outside Python's normal `sys.stderr` object.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import io
import json
import os
import sys
import sysconfig
import tempfile
import time
from collections.abc import Generator
from contextlib import chdir, contextmanager, redirect_stderr, redirect_stdout
from importlib import metadata as importlib_metadata
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from mypyc.build import mypycify
from packaging import tags
from setuptools import Distribution
from setuptools.command.build_ext import build_ext

from atoll.backends.base import UnsupportedBackendRegionError
from atoll.models import (
    ArtifactRecord,
    Backend,
    BackendAssessment,
    BackendAssessmentStatus,
    BackendCapability,
    BackendCompileContext,
    BackendCompileResult,
    BackendDiagnostic,
    BackendDiagnosticCode,
    BackendLoweringRequest,
    CompilationUnit,
    CompileAttempt,
    CompilePhaseTiming,
    LoweringDecision,
    RegionMember,
    SymbolId,
    TypedRegion,
)

_MAX_DIAGNOSTIC_LINES = 20
_MYPYC_ADAPTER_VERSION = "2"
_MYPYC_MYPY_OPTIONS = ("--no-warn-unused-configs",)
_SHARED_REGION_ID = "__shared__"


class MypycBackend:
    """CompilerBackend adapter around mypyc's in-process setuptools build.

    Capability assessment is member-specific so unsupported async generators or
    unresolved generic members remain available to another backend. Compilation
    delegates to the proven legacy build implementation and adds structured
    artifact metadata without changing command-facing `CompileAttempt` output.
    """

    @property
    def name(self) -> Backend:
        """Return the stable backend name used by reports and cache keys.

        Returns:
            Backend: The `mypyc` backend identifier.
        """
        return "mypyc"

    def assess(self, region: TypedRegion) -> BackendAssessment:
        """Assess typed functions, methods, classes, generators, and coroutines.

        Args:
            region: Backend-neutral typed region being assessed or generated.

        Returns:
            BackendAssessment: Mypyc capability assessment for each region member.
        """
        decisions = {decision.target: decision for decision in region.decisions}
        supported: list[SymbolId] = []
        unsupported: list[SymbolId] = []
        reasons: list[str] = []
        for member in region.members:
            reason = _unsupported_member_reason(member, decisions.get(member.id.stable_id))
            if reason is None:
                supported.append(member.id)
            else:
                unsupported.append(member.id)
                reasons.append(f"{member.id.stable_id}: {reason}")

        class_member = next((member for member in region.members if member.kind == "class"), None)
        if region.atomic_class and class_member is not None and unsupported:
            if class_member.id in supported:
                supported.remove(class_member.id)
                unsupported.append(class_member.id)
            reasons.append(
                f"{class_member.id.stable_id}: native class requires every class member to pass"
            )

        status = _assessment_status(supported, unsupported)
        return BackendAssessment(
            region_id=region.id,
            backend="mypyc",
            status=status,
            supported_members=tuple(supported),
            unsupported_members=tuple(unsupported),
            capabilities=_member_capabilities(region, tuple(supported)),
            reasons=tuple(reasons),
        )

    def lower(self, request: BackendLoweringRequest) -> CompilationUnit:
        """Validate prepared source and register supported members as one unit.

        The adapter deliberately does not generate source. The typed-region
        lowerer supplies a preserved source file, and this method records the
        exact member selection and content hash consumed by mypyc.

        Args:
            request: Prepared source and member selection offered to the backend.

        Returns:
            CompilationUnit: Mypyc compilation unit for the selected region members.

        Raises:
            UnsupportedBackendRegionError: If selected members violate the backend assessment
                contract.
        """
        assessment = self.assess(request.region)
        selected = request.members or assessment.supported_members
        unsupported = set(selected) - set(assessment.supported_members)
        if not selected or unsupported:
            names = (
                ", ".join(
                    symbol.stable_id
                    for symbol in sorted(unsupported, key=lambda item: item.stable_id)
                )
                or "none"
            )
            raise UnsupportedBackendRegionError(
                f"mypyc cannot lower requested members for {request.region.id}: {names}"
            )
        return CompilationUnit(
            region_id=request.variant_id or request.region.id,
            backend="mypyc",
            logical_module=request.logical_module,
            source_paths=(request.source_path,),
            source_hash=_file_digest(request.source_path),
            members=selected,
            install_relative_dir=request.install_relative_dir,
        )

    def compile(
        self,
        units: tuple[CompilationUnit, ...],
        context: BackendCompileContext,
    ) -> BackendCompileResult:
        """Compile prepared units and attach install-facing artifact records.

        Args:
            units: Backend compilation units submitted as one build request.
            context: Filesystem, cache, and artifact-recording boundaries for compilation.

        Returns:
            BackendCompileResult: Mypyc build evidence and validated artifact records.
        """
        _validate_units(units, expected_backend="mypyc")
        paths = tuple(path for unit in units for path in unit.source_paths)
        attempt = _build_paths(paths, context=context, backend=self)
        artifact_dir = context.build_dir.parent / "artifacts"
        artifacts = (
            _artifact_records(units, attempt.artifact_paths, artifact_dir)
            if context.record_artifacts
            else ()
        )
        return BackendCompileResult(attempt=attempt, artifacts=artifacts)

    def fingerprint(
        self,
        unit: CompilationUnit,
        context: BackendCompileContext,
    ) -> str:
        """Hash unit content together with the active mypyc toolchain and ABI.

        Args:
            unit: Content-addressable backend compilation unit.
            context: Filesystem, cache, and artifact-recording boundaries for compilation.

        Returns:
            str: Stable mypyc cache fingerprint for the unit and context.
        """
        _validate_units((unit,), expected_backend="mypyc")
        payload = {
            "adapter_version": _MYPYC_ADAPTER_VERSION,
            "backend": self.name,
            "mypy_version": importlib_metadata.version("mypy"),
            "mypy_options": _MYPYC_MYPY_OPTIONS,
            "setuptools_version": importlib_metadata.version("setuptools"),
            "python_cache_tag": sys.implementation.cache_tag,
            "platform": sysconfig.get_platform(),
            "soabi": sysconfig.get_config_var("SOABI"),
            "ext_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
            "compiler": sysconfig.get_config_var("CC"),
            "compiler_flags": sysconfig.get_config_var("CFLAGS"),
            "extension_suffixes": importlib.machinery.EXTENSION_SUFFIXES,
            "region_id": unit.region_id,
            "logical_module": unit.logical_module,
            "install_relative_dir": unit.install_relative_dir,
            "source_hash": unit.source_hash,
            "source_digests": [_file_digest(path) for path in unit.source_paths],
            "members": [member.stable_id for member in unit.members],
            "project_root": str(context.project_root.resolve()),
            "source_roots": [str(path.resolve()) for path in context.source_roots],
            "package_root": _package_root(context.project_root, context.source_roots),
            "backend_options": [list(option) for option in context.backend_options],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def normalize_diagnostic(
        self,
        error: BaseException,
        *,
        diagnostics: str,
        log_path: Path | None,
    ) -> BackendDiagnostic:
        """Normalize mypyc, import-path, and native toolchain failures.

        Args:
            error: Backend exception that caused compilation to fail.
            diagnostics: Captured backend diagnostic text to normalize.
            log_path: Optional path to the complete backend build log.

        Returns:
            BackendDiagnostic: Normalized mypyc or build-environment diagnostic.
        """
        message = repr(error)
        lowered = f"{message}\n{diagnostics}".lower()
        code = _diagnostic_code(lowered)
        detail = _diagnostic_summary(diagnostics, log_path)
        return BackendDiagnostic(
            code=code,
            message=message,
            details=tuple(line for line in detail.splitlines() if line),
            log_path=log_path,
            transient=code != "MYPYC_TYPE_ERROR",
        )


MYPYC_BACKEND = MypycBackend()


def _unsupported_member_reason(
    member: RegionMember,
    decision: LoweringDecision | None,
) -> str | None:
    if member.execution_kind == "async_generator":
        return "mypyc does not preserve async-generator execution semantics"
    if decision is None or decision.action in {"preserve", "specialize"}:
        return None
    if decision.action == "box":
        return "mypyc preference requires concrete typing; source uses Any"
    if decision.action == "fallback":
        return "member requires interpreted fallback before specialization"
    return "member was rejected by typed-region analysis"


def _assessment_status(
    supported: list[SymbolId],
    unsupported: list[SymbolId],
) -> BackendAssessmentStatus:
    if supported and unsupported:
        return "partial"
    if supported:
        return "supported"
    return "unsupported"


def _member_capabilities(
    region: TypedRegion,
    supported: tuple[SymbolId, ...],
) -> tuple[BackendCapability, ...]:
    supported_ids = set(supported)
    capabilities: list[BackendCapability] = []
    for member in region.members:
        if member.id not in supported_ids:
            continue
        if member.kind == "class":
            capabilities.append("native_class")
        elif member.kind == "function":
            capabilities.append("typed_function")
        elif member.binding_kind == "instance_method":
            capabilities.append("instance_method")
        elif member.binding_kind == "staticmethod":
            capabilities.append("staticmethod")
        elif member.binding_kind == "classmethod":
            capabilities.append("classmethod")
        if member.execution_kind == "generator":
            capabilities.append("generator")
        elif member.execution_kind == "coroutine":
            capabilities.append("coroutine")
    return tuple(dict.fromkeys(capabilities))


def _legacy_unit(path: Path) -> CompilationUnit:
    digest = _path_digest(path)
    return CompilationUnit(
        region_id=f"legacy:{path.stem}:{digest[:12]}",
        backend="mypyc",
        logical_module=path.stem,
        source_paths=(path,),
        source_hash=digest,
        members=(),
        install_relative_dir=".atoll/artifacts",
    )


def _validate_units(
    units: tuple[CompilationUnit, ...],
    *,
    expected_backend: Backend,
) -> None:
    mismatched = tuple(unit for unit in units if unit.backend != expected_backend)
    if mismatched:
        names = ", ".join(unit.region_id for unit in mismatched)
        raise ValueError(f"{expected_backend} backend received incompatible unit(s): {names}")
    empty = tuple(unit.region_id for unit in units if not unit.source_paths)
    if empty:
        raise ValueError(
            f"{expected_backend} backend received source-less unit(s): {', '.join(empty)}"
        )


def _artifact_records(
    units: tuple[CompilationUnit, ...],
    artifact_paths: tuple[Path, ...],
    artifact_dir: Path,
) -> tuple[ArtifactRecord, ...]:
    if not artifact_paths:
        return ()
    system_tag = next(tags.sys_tags())
    records: list[ArtifactRecord] = []
    for artifact in artifact_paths:
        unit = _artifact_unit(artifact, units, artifact_dir)
        install_dirs: tuple[str, ...]
        if unit is not None:
            install_dirs = (unit.install_relative_dir,)
        else:
            install_dirs = tuple(
                dict.fromkeys(candidate.install_relative_dir for candidate in units)
            ) or ("",)
        records.extend(
            ArtifactRecord(
                region_id=unit.region_id if unit is not None else _SHARED_REGION_ID,
                backend="mypyc",
                logical_module=(
                    unit.logical_module if unit is not None else _extension_module_stem(artifact)
                ),
                role="primary" if unit is not None else "support",
                install_relative_path=_install_relative_path(
                    artifact,
                    artifact_dir,
                    install_relative_dir,
                ),
                digest=_file_digest(artifact),
                abi=system_tag.abi,
                platform_tag=system_tag.platform,
            )
            for install_relative_dir in install_dirs
        )
    return tuple(records)


def _artifact_unit(
    artifact: Path,
    units: tuple[CompilationUnit, ...],
    artifact_dir: Path,
) -> CompilationUnit | None:
    module_stem = _extension_module_stem(artifact)
    try:
        relative = artifact.resolve().relative_to(artifact_dir.resolve())
    except ValueError:
        return None
    artifact_module = ".".join((*relative.parent.parts, module_stem))
    logical_matches = tuple(unit for unit in units if unit.logical_module == artifact_module)
    if len(logical_matches) == 1:
        return logical_matches[0]
    stem_matches = tuple(
        unit for unit in units if any(module_stem == path.stem for path in unit.source_paths)
    )
    return stem_matches[0] if len(stem_matches) == 1 else None


def _extension_module_stem(artifact: Path) -> str:
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        if artifact.name.endswith(suffix):
            return artifact.name[: -len(suffix)]
    return artifact.stem


def _install_relative_path(
    artifact: Path,
    artifact_dir: Path,
    install_relative_dir: str,
) -> str:
    try:
        relative_output = artifact.resolve().relative_to(artifact_dir.resolve())
    except ValueError as error:
        raise ValueError(f"artifact is outside backend output root: {artifact}") from error
    return (PurePosixPath(install_relative_dir) / relative_output.as_posix()).as_posix()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_digest(path: Path) -> str:
    if path.is_file():
        return _file_digest(path)
    return hashlib.sha256(f"missing:{path.resolve()}".encode()).hexdigest()


def build_sidecars(
    paths: tuple[Path, ...],
    *,
    project_root: Path,
    build_dir: Path,
    source_roots: tuple[Path, ...] = (),
    cache_dir: Path | None = None,
) -> CompileAttempt:
    """Compile generated sidecar source files into Atoll's artifact directory.

    Empty input is a successful no-op. Build failures are converted into
    `CompileAttempt` values with classified diagnostics instead of escaping as
    exceptions through CLI command handlers.

    Args:
        paths: Generated source paths submitted to the native compiler.
        project_root: Root directory of the target Python project.
        build_dir: Directory for disposable native compiler inputs and outputs.
        source_roots: Import roots made visible to analysis or child processes.
        cache_dir: Optional directory for reusable compiler cache entries.

    Returns:
        CompileAttempt: Compatibility build attempt containing command, diagnostics, timings, and
            artifacts.
    """
    units = tuple(_legacy_unit(path) for path in paths)
    return MYPYC_BACKEND.compile(
        units,
        BackendCompileContext(
            project_root=project_root,
            build_dir=build_dir,
            source_roots=source_roots,
            cache_dir=cache_dir,
            record_artifacts=False,
        ),
    ).attempt


def _build_paths(
    paths: tuple[Path, ...],
    *,
    context: BackendCompileContext,
    backend: MypycBackend,
) -> CompileAttempt:
    """Run the legacy mypyc build and preserve its CompileAttempt contract.

    Args:
        paths: Filesystem paths processed in deterministic order.
        context: Prepared state shared by this operation.
        backend: Compiler backend selected for this operation.

    Returns:
        CompileAttempt: Normalized build and cache directories for the native compiler.
    """
    project_root = context.project_root
    build_dir = context.build_dir
    source_roots = context.source_roots
    cache_dir = context.cache_dir
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
    command = (
        "mypyc",
        *_MYPYC_MYPY_OPTIONS,
        *tuple(str(path) for path in paths),
        "build_ext",
    )
    stdout = io.StringIO()
    stderr = io.StringIO()
    native_stderr = _NativeStderrCapture()
    phase_timings: list[CompilePhaseTiming] = []
    active_phase: tuple[str, float] | None = None
    try:
        with (
            chdir(project_root),
            _mypy_environment(source_roots, cache_dir or build_dir / "mypy_cache"),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
            native_stderr,
        ):
            active_phase = ("mypycify", time.perf_counter())
            generated_target = _source_arg(build_dir / "generated", project_root)
            ext_modules = mypycify(
                [
                    *_MYPYC_MYPY_OPTIONS,
                    *(_source_arg(path, project_root) for path in paths),
                ],
                target_dir=generated_target,
            )
            phase_timings.append(_phase_timing(active_phase))
            active_phase = ("build_ext", time.perf_counter())
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
            _prepare_build_temp_source_dir(Path(command_obj.build_temp), generated_target)
            command_obj.ensure_finalized()
            command_obj.run()
            phase_timings.append(_phase_timing(active_phase))
            active_phase = None
    except SystemExit as error:
        if active_phase is not None:
            phase_timings.append(_phase_timing(active_phase))
        diagnostics = _captured_output(stdout, stderr, native_stderr.output())
        log_path = _write_build_log(build_dir, diagnostics, error)
        return CompileAttempt(
            success=False,
            command=command,
            stdout="",
            stderr=_diagnostic_text(
                backend.normalize_diagnostic(
                    error,
                    diagnostics=diagnostics,
                    log_path=log_path,
                )
            ),
            artifact_paths=(),
            duration_seconds=time.perf_counter() - start,
            phase_timings=tuple(phase_timings),
        )
    except Exception as error:
        if active_phase is not None:
            phase_timings.append(_phase_timing(active_phase))
        diagnostics = _captured_output(stdout, stderr, native_stderr.output())
        log_path = _write_build_log(build_dir, diagnostics, error)
        return CompileAttempt(
            success=False,
            command=command,
            stdout="",
            stderr=_diagnostic_text(
                backend.normalize_diagnostic(
                    error,
                    diagnostics=diagnostics,
                    log_path=log_path,
                )
            ),
            artifact_paths=(),
            duration_seconds=time.perf_counter() - start,
            phase_timings=tuple(phase_timings),
        )
    artifact_started = time.perf_counter()
    artifacts = _artifact_paths(paths, artifact_dir, previous_artifacts)
    phase_timings.append(_phase_timing(("artifact_discovery", artifact_started)))
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
        phase_timings=tuple(phase_timings),
    )


def _phase_timing(active_phase: tuple[str, float]) -> CompilePhaseTiming:
    name, started = active_phase
    return CompilePhaseTiming(name=name, duration_seconds=time.perf_counter() - started)


class _NativeStderrCapture:
    """Capture writes to file descriptor 2 during in-process native builds."""

    def __init__(self) -> None:
        """Initialize the saved descriptor, temporary file, and captured buffer."""
        self._saved_fd: int | None = None
        self._file: BinaryIO | None = None
        self._captured = ""

    def __enter__(self) -> _NativeStderrCapture:
        """Redirect native stderr to a temporary file for the build duration.

        Returns:
            _NativeStderrCapture: Active context manager instance.
        """
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
        """Restore native stderr and load captured bytes as replacement text.

        Args:
            exc_type: Active exception type, when context exit follows a failure.
            exc: Active exception instance, when context exit follows a failure.
            traceback: Active traceback, when context exit follows a failure.
        """
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
        """Return captured native stderr after the context has exited.

        Returns:
            str: Captured and normalized compiler output.
        """
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


def _prepare_build_temp_source_dir(build_temp: Path, source_dir: str) -> None:
    """Create the object-file parent mirrored by setuptools for generated C sources.

    Args:
        build_temp: Temporary build directory whose output should be filtered.
        source_dir: Directory copied into source-clean staging.
    """
    source_path = Path(source_dir)
    relative_parts = source_path.parts[1:] if source_path.anchor else source_path.parts
    build_temp.joinpath(*relative_parts).mkdir(parents=True, exist_ok=True)


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


def _diagnostic_code(lowered: str) -> BackendDiagnosticCode:
    if "no such file" in lowered or "compiler" in lowered or "clang" in lowered or "gcc" in lowered:
        return "NATIVE_BUILD_ENV_ERROR"
    if "mypy" in lowered or "type" in lowered or ": error:" in lowered:
        return "MYPYC_TYPE_ERROR"
    if "import" in lowered or "module" in lowered:
        return "IMPORT_PATH_ERROR"
    return "UNKNOWN_BUILD_ERROR"


def _diagnostic_text(diagnostic: BackendDiagnostic) -> str:
    return "\n".join((f"{diagnostic.code}: {diagnostic.message}", *diagnostic.details))


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
