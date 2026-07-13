"""Programmatic Cython build backend for source-clean native regions.

The backend compiles preserved ``.py`` regions and proof-authorized ``.pyx``
specializations. It owns capability assessment, content/toolchain fingerprinting,
portable compiler optimization, in-process extension builds, and stable diagnostic
normalization; source generation and runtime binding remain outside this module.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import io
import json
import os
import sys
import sysconfig
import tempfile
import time
from collections.abc import Generator, Mapping, Sequence
from contextlib import chdir, contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Protocol, cast

from packaging import tags
from setuptools import Distribution, Extension
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

_CYTHON_ADAPTER_VERSION = "4"
_DIRECTIVES: dict[str, object] = {
    "language_level": 3,
    "annotation_typing": False,
    "infer_types": False,
    "binding": True,
    "embedsignature": True,
}
_MAX_DIAGNOSTIC_LINES = 20
_SHARED_REGION_ID = "__shared__"
_SUPPORTED_SOURCE_SUFFIXES = frozenset({".py", ".pyx"})
_PROOF_GENERATED_PYX_MARKERS = (
    "# atoll scalar proof ",
    "# atoll buffer proof ",
)
_PORTABLE_OPTIMIZATION_FLAGS = ("/O2",) if os.name == "nt" else ("-O3",)


class _CythonizeFunction(Protocol):
    """Typed view of ``Cython.Build.cythonize`` used by this adapter."""

    def __call__(
        self,
        module_list: Sequence[Extension],
        *,
        compiler_directives: Mapping[str, object],
        quiet: bool,
        build_dir: str,
        nthreads: int,
    ) -> list[Extension]:
        """Return setuptools extension modules generated from Cython inputs.

        Args:
            module_list: Mutable output list receiving discovered modules.
            compiler_directives: Cython compiler directives applied to the build.
            quiet: Whether progress output should be suppressed.
            build_dir: Directory containing disposable native build inputs.
            nthreads: Number of independent Cython translation workers.

        Returns:
            list[Extension]: Filtered compiler output with duplicate path noise removed.
        """
        ...


@dataclass(slots=True)
class _FailureState:
    """Mutable build state needed to normalize an interrupted compiler attempt.

    Attributes:
        backend: Compiler backend selected for this state.
        build_dir: Directory containing disposable native build inputs.
        command: Normalized command argument vector.
        stdout: Captured child process standard output.
        stderr: Captured child process standard error.
        native_stderr: Native compiler standard error retained for reporting.
        phase_timings: Measured compiler subphase timings.
        active_phase: Compiler phase currently contributing captured output.
        started: Monotonic start time used for elapsed-duration reporting.
    """

    backend: CythonBackend
    build_dir: Path
    command: tuple[str, ...]
    stdout: io.StringIO
    stderr: io.StringIO
    native_stderr: _NativeStderrCapture
    phase_timings: list[CompilePhaseTiming]
    active_phase: tuple[str, float] | None
    started: float


class CythonBackend:
    """CompilerBackend adapter around Cython's in-process setuptools build.

    Cython is selected for typed and boxed Python callables, including async
    generators, and for closed atomic classes whose method reflection must stay
    Python-compatible. Boxed callables preserve Python object semantics and
    source annotations; rejected class behavior remains unsupported.
    """

    @property
    def name(self) -> Backend:
        """Return the stable backend name used by reports and cache keys.

        Returns:
            Backend: The `cython` backend identifier.
        """
        return "cython"

    def assess(self, region: TypedRegion) -> BackendAssessment:
        """Assess typed functions, methods, async generators, and closed classes.

        Args:
            region: Backend-neutral typed region being assessed or generated.

        Returns:
            BackendAssessment: Cython capability assessment for each region member.
        """
        decisions = {decision.target: decision for decision in region.decisions}
        supported: list[SymbolId] = []
        unsupported: list[SymbolId] = []
        reasons: list[str] = []
        for member in region.members:
            reason = _unsupported_member_reason(
                member,
                decisions.get(member.id.stable_id),
                allow_boxed=not region.atomic_class,
            )
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
                f"{class_member.id.stable_id}: atomic class requires every class member to pass"
            )

        return BackendAssessment(
            region_id=region.id,
            backend="cython",
            status=_assessment_status(supported, unsupported),
            supported_members=tuple(supported),
            unsupported_members=tuple(unsupported),
            capabilities=_member_capabilities(region, tuple(supported)),
            reasons=tuple(reasons),
        )

    def lower(self, request: BackendLoweringRequest) -> CompilationUnit:
        """Validate a supported member subset and record the prepared source file.

        Args:
            request: Prepared source and member selection offered to the backend.

        Returns:
            CompilationUnit: Cython compilation unit for the selected region members.

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
                f"cython cannot lower requested members for {request.region.id}: {names}"
            )
        if request.source_path.suffix not in _SUPPORTED_SOURCE_SUFFIXES:
            raise UnsupportedBackendRegionError(
                f"cython lowers only .py and proof-generated .pyx units: {request.source_path}"
            )
        if request.source_path.suffix == ".pyx" and not _is_proof_generated_pyx(
            request.source_path
        ):
            raise UnsupportedBackendRegionError(
                f"cython .pyx lowering requires Atoll proof provenance: {request.source_path}"
            )
        return CompilationUnit(
            region_id=request.variant_id or request.region.id,
            backend="cython",
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
        """Compile pure-Python units and attach install-facing artifact records.

        Args:
            units: Backend compilation units submitted as one build request.
            context: Filesystem, cache, and artifact-recording boundaries for compilation.

        Returns:
            BackendCompileResult: Cython build evidence and validated artifact records.
        """
        _validate_units(units, expected_backend="cython")
        attempt = _build_units(units, context=context, backend=self)
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
        """Hash unit content together with Cython, ABI, platform, and options.

        Args:
            unit: Content-addressable backend compilation unit.
            context: Filesystem, cache, and artifact-recording boundaries for compilation.

        Returns:
            str: Stable Cython cache fingerprint for the unit and context.
        """
        _validate_units((unit,), expected_backend="cython")
        payload = {
            "adapter_version": _CYTHON_ADAPTER_VERSION,
            "backend": self.name,
            "cython_version": importlib_metadata.version("Cython"),
            "setuptools_version": importlib_metadata.version("setuptools"),
            "directives": _DIRECTIVES,
            "portable_optimization_flags": _PORTABLE_OPTIMIZATION_FLAGS,
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
            "source_suffixes": [path.suffix for path in unit.source_paths],
            "members": [member.stable_id for member in unit.members],
            "project_root": str(context.project_root.resolve()),
            "source_roots": [str(path.resolve()) for path in context.source_roots],
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
        """Normalize Cython, import-path, native toolchain, and unknown failures.

        Args:
            error: Backend exception that caused compilation to fail.
            diagnostics: Captured backend diagnostic text to normalize.
            log_path: Optional path to the complete backend build log.

        Returns:
            BackendDiagnostic: Normalized Cython or build-environment diagnostic.
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
            transient=code != "CYTHON_COMPILE_ERROR",
        )


CYTHON_BACKEND = CythonBackend()


def _unsupported_member_reason(
    member: RegionMember,
    decision: LoweringDecision | None,
    *,
    allow_boxed: bool,
) -> str | None:
    if decision is None or decision.action in {"preserve", "specialize"}:
        return None
    if (
        allow_boxed
        and member.kind in {"function", "method"}
        and decision.action in {"box", "fallback"}
    ):
        return None
    if member.kind in {"function", "method"} and decision.action in {"box", "fallback"}:
        return "atomic class lowering requires concrete callable typing"
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
        binding_capability = _binding_capability(member)
        execution_capability = _execution_capability(member)
        if binding_capability is not None:
            capabilities.append(binding_capability)
        if execution_capability is not None:
            capabilities.append(execution_capability)
    return tuple(dict.fromkeys(capabilities))


def _binding_capability(member: RegionMember) -> BackendCapability | None:
    if member.kind == "class":
        return "native_class"
    if member.kind == "function":
        return "typed_function"
    if member.binding_kind == "instance_method":
        return "instance_method"
    if member.binding_kind == "staticmethod":
        return "staticmethod"
    if member.binding_kind == "classmethod":
        return "classmethod"
    return None


def _execution_capability(member: RegionMember) -> BackendCapability | None:
    if member.execution_kind == "generator":
        return "generator"
    if member.execution_kind == "coroutine":
        return "coroutine"
    if member.execution_kind == "async_generator":
        return "async_generator"
    return None


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
    unsupported_sources = tuple(
        str(path)
        for unit in units
        for path in unit.source_paths
        if path.suffix not in _SUPPORTED_SOURCE_SUFFIXES
    )
    if unsupported_sources:
        raise ValueError(
            f"{expected_backend} backend received unsupported source unit(s): "
            f"{', '.join(unsupported_sources)}"
        )
    unproven_pyx = tuple(
        str(path)
        for unit in units
        for path in unit.source_paths
        if path.suffix == ".pyx" and not _is_proof_generated_pyx(path)
    )
    if unproven_pyx:
        raise ValueError(
            f"{expected_backend} backend received .pyx unit(s) without Atoll proof provenance: "
            f"{', '.join(unproven_pyx)}"
        )


def _is_proof_generated_pyx(path: Path) -> bool:
    """Return whether a Cython source carries Atoll's proof-generation marker.

    Args:
        path: Candidate `.pyx` source supplied to the backend.

    Returns:
        bool: Whether the file was emitted by a recognized Atoll proof generator.
    """
    source = path.read_text(encoding="utf-8")
    return any(marker in source for marker in _PROOF_GENERATED_PYX_MARKERS)


def _build_units(
    units: tuple[CompilationUnit, ...],
    *,
    context: BackendCompileContext,
    backend: CythonBackend,
) -> CompileAttempt:
    start = time.perf_counter()
    if not units:
        return CompileAttempt(
            success=True,
            command=("cython", "build_ext"),
            stdout="no Cython units to build",
            stderr="",
            artifact_paths=(),
            duration_seconds=time.perf_counter() - start,
        )
    build_dir = context.build_dir
    build_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = build_dir.parent / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    previous_artifacts = {
        path: path.stat().st_mtime_ns for path in _all_extension_artifacts(artifact_dir)
    }
    paths = tuple(path for unit in units for path in unit.source_paths)
    command = ("cython", *tuple(str(path) for path in paths), "build_ext")
    stdout = io.StringIO()
    stderr = io.StringIO()
    native_stderr = _NativeStderrCapture()
    phase_timings: list[CompilePhaseTiming] = []
    active_phase: tuple[str, float] | None = None
    try:
        with (
            chdir(context.project_root),
            _python_path(context.source_roots),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
            native_stderr,
        ):
            active_phase = ("cythonize", time.perf_counter())
            workers = _parallel_worker_count(len(units))
            extensions = _cythonize_extensions(
                units,
                build_dir,
                project_root=context.project_root,
                workers=workers,
            )
            phase_timings.append(_phase_timing(active_phase))
            active_phase = ("build_ext", time.perf_counter())
            distribution = Distribution(
                {"name": "atoll_cython_generated", "ext_modules": extensions}
            )
            command_obj = build_ext(distribution)
            command_obj.inplace = False
            command_obj.build_lib = str(artifact_dir)
            command_obj.build_temp = str(build_dir / "temp")
            command_obj.parallel = workers
            _prepare_build_temp_source_dir(
                Path(command_obj.build_temp),
                _project_path(build_dir / "cythonized", context.project_root),
            )
            command_obj.ensure_finalized()
            command_obj.run()
            build_ext_timing = _phase_timing(active_phase)
            phase_timings.append(
                CompilePhaseTiming(
                    name=build_ext_timing.name,
                    duration_seconds=build_ext_timing.duration_seconds,
                    detail=f"{workers} worker(s); {len(units)} extension(s)",
                )
            )
            active_phase = None
    except SystemExit as error:
        return _failed_attempt(
            error,
            _FailureState(
                backend=backend,
                build_dir=build_dir,
                command=command,
                stdout=stdout,
                stderr=stderr,
                native_stderr=native_stderr,
                phase_timings=phase_timings,
                active_phase=active_phase,
                started=start,
            ),
        )
    except Exception as error:
        return _failed_attempt(
            error,
            _FailureState(
                backend=backend,
                build_dir=build_dir,
                command=command,
                stdout=stdout,
                stderr=stderr,
                native_stderr=native_stderr,
                phase_timings=phase_timings,
                active_phase=active_phase,
                started=start,
            ),
        )
    artifact_started = time.perf_counter()
    artifacts = _artifact_paths(units, artifact_dir, previous_artifacts)
    phase_timings.append(_phase_timing(("artifact_discovery", artifact_started)))
    diagnostics = _captured_output(stdout, stderr, native_stderr.output())
    if diagnostics:
        _write_build_log(build_dir, diagnostics, None)
    return CompileAttempt(
        success=bool(artifacts),
        command=command,
        stdout=diagnostics,
        stderr="" if artifacts else "cython build completed but no extension artifacts were found",
        artifact_paths=artifacts,
        duration_seconds=time.perf_counter() - start,
        phase_timings=tuple(phase_timings),
    )


def _failed_attempt(
    error: BaseException,
    state: _FailureState,
) -> CompileAttempt:
    if state.active_phase is not None:
        state.phase_timings.append(_phase_timing(state.active_phase))
    diagnostics = _captured_output(state.stdout, state.stderr, state.native_stderr.output())
    log_path = _write_build_log(state.build_dir, diagnostics, error)
    return CompileAttempt(
        success=False,
        command=state.command,
        stdout="",
        stderr=_diagnostic_text(
            state.backend.normalize_diagnostic(error, diagnostics=diagnostics, log_path=log_path)
        ),
        artifact_paths=(),
        duration_seconds=time.perf_counter() - state.started,
        phase_timings=tuple(state.phase_timings),
    )


def _cythonize_extensions(
    units: tuple[CompilationUnit, ...],
    build_dir: Path,
    *,
    project_root: Path,
    workers: int,
) -> list[Extension]:
    cython_build = importlib.import_module("Cython.Build")
    cythonize = cast(_CythonizeFunction, cython_build.cythonize)
    cythonized_dir = build_dir / "cythonized"
    cythonized_dir.mkdir(parents=True, exist_ok=True)
    extensions = [
        Extension(
            unit.logical_module,
            [_project_path(unit.source_paths[0], project_root)],
            extra_compile_args=list(_PORTABLE_OPTIMIZATION_FLAGS),
        )
        for unit in units
    ]
    return cythonize(
        extensions,
        compiler_directives=_DIRECTIVES,
        quiet=True,
        build_dir=_project_path(cythonized_dir, project_root),
        nthreads=workers,
    )


def _parallel_worker_count(unit_count: int) -> int:
    """Bound translation and object-build workers by units and host CPUs.

    Args:
        unit_count: Number of independent extension units in one physical batch.

    Returns:
        int: Positive worker count suitable for Cython and setuptools.
    """
    return max(1, min(unit_count, os.cpu_count() or 1))


def _project_path(path: Path, project_root: Path) -> str:
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


@contextmanager
def _python_path(source_roots: tuple[Path, ...]) -> Generator[None]:
    paths = tuple(os.fspath(path.resolve()) for path in source_roots)
    original_path = sys.path.copy()
    try:
        if paths:
            sys.path[:0] = list(paths)
        yield
    finally:
        sys.path[:] = original_path


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
                backend="cython",
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


def _artifact_paths(
    units: tuple[CompilationUnit, ...],
    artifact_dir: Path,
    previous_artifacts: dict[Path, int],
) -> tuple[Path, ...]:
    artifacts: set[Path] = set()
    module_stems = {unit.logical_module.rsplit(".", maxsplit=1)[-1] for unit in units} | {
        path.stem for unit in units for path in unit.source_paths
    }
    for stem in module_stems:
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            artifacts.update(artifact_dir.rglob(f"{stem}*{suffix}"))
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


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    log_path = build_dir / "cython.log"
    lines = ["# Atoll Cython build log", ""]
    if error is not None:
        lines.extend([f"exception: {type(error).__name__}: {error!r}", ""])
    if diagnostics:
        lines.extend(["diagnostics:", diagnostics, ""])
    log_path.write_text("\n".join(lines), encoding="utf-8")
    return log_path


def _diagnostic_code(lowered: str) -> BackendDiagnosticCode:
    if "no such file" in lowered or "compiler" in lowered or "clang" in lowered or "gcc" in lowered:
        return "NATIVE_BUILD_ENV_ERROR"
    if "cython" in lowered or ": error:" in lowered or "compileerror" in lowered:
        return "CYTHON_COMPILE_ERROR"
    if "import" in lowered or "module" in lowered:
        return "IMPORT_PATH_ERROR"
    return "UNKNOWN_BUILD_ERROR"


def _diagnostic_text(diagnostic: BackendDiagnostic) -> str:
    return "\n".join((f"{diagnostic.code}: {diagnostic.message}", *diagnostic.details))


def _diagnostic_summary(diagnostics: str, log_path: Path | None) -> str:
    lines = [line for line in diagnostics.splitlines() if line.strip()]
    if not lines:
        return ""
    error_lines = [line for line in lines if ": error:" in line or "Error compiling" in line]
    selected = error_lines[:_MAX_DIAGNOSTIC_LINES] or lines[:_MAX_DIAGNOSTIC_LINES]
    omitted = max((len(error_lines) or len(lines)) - len(selected), 0)
    parts = ["", f"Captured {len(error_lines)} Cython error line(s)."]
    if log_path is not None:
        parts.append(f"Full diagnostics: {log_path}")
    parts.extend(selected)
    if omitted:
        parts.append(f"... {omitted} more line(s) in the build log")
    return "\n".join(parts)
