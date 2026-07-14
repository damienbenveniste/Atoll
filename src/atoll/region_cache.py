"""Persistent artifact and deterministic-rejection cache for typed regions.

The cache delegates fingerprint construction to the selected compiler backend,
then stores native files together with their install-facing `ArtifactRecord`
evidence. It also records normalized non-transient compiler rejections under a
separate decision namespace. Toolchain and environment failures are never
persisted as backend decisions.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace
from itertools import count
from pathlib import Path, PurePosixPath
from typing import cast

from atoll.backends.base import CompilerBackend
from atoll.models import (
    ArtifactRecord,
    ArtifactRole,
    Backend,
    BackendCompileContext,
    BackendCompileResult,
    BackendDiagnosticScope,
    CompilationUnit,
    CompileAttempt,
    CompilePhaseTiming,
)

REGION_CACHE_VERSION = 1
BACKEND_DECISION_CACHE_VERSION = 2
_DETERMINISTIC_DIAGNOSTIC_CODES = (
    "MYPYC_TYPE_ERROR",
    "CYTHON_COMPILE_ERROR",
)


@dataclass(frozen=True, slots=True)
class _CacheIdentity:
    entry_root: Path
    key: str
    backend: Backend
    region_id: str


@dataclass(frozen=True, slots=True)
class _CacheProbe:
    """One independently fingerprinted cache lookup before backend execution.

    Attributes:
        identity: Stable cache location and backend identity for the unit.
        lookup_duration_seconds: Time spent fingerprinting and reading cache evidence.
        cached_result: Restored result, or `None` when compilation is required.
    """

    identity: _CacheIdentity
    lookup_duration_seconds: float
    cached_result: BackendCompileResult | None


@dataclass(frozen=True, slots=True)
class _BatchMiss:
    """One ordered cache miss eligible for a shared backend invocation.

    Attributes:
        index: Original unit position used to restore deterministic result order.
        unit: Compilation unit that was absent from the success cache.
        probe: Lookup evidence retained for the eventual per-unit result.
    """

    index: int
    unit: CompilationUnit
    probe: _CacheProbe


@dataclass(frozen=True, slots=True)
class _BatchInvocation:
    """Shared physical invocation identity for per-unit attempt views.

    Attributes:
        number: One-based invocation number within recursive failure isolation.
        unit_count: Number of logical units sharing the physical compiler process.
    """

    number: int
    unit_count: int


def compile_with_region_cache(
    backend: CompilerBackend,
    unit: CompilationUnit,
    context: BackendCompileContext,
    *,
    cache_root: Path,
) -> BackendCompileResult:
    """Restore or compile one backend variant using a strict success-only key.

    Corrupt, stale, or unreadable entries are ordinary misses. Cache writes are
    atomic at the entry-directory level and never replace the backend result if
    storage itself fails.

    Args:
        backend: Compiler backend used to assess, lower, or compile the region.
        unit: Content-addressable backend compilation unit.
        context: Filesystem, cache, and artifact-recording boundaries for compilation.
        cache_root: Root directory for content-addressed typed-region cache entries.

    Returns:
        BackendCompileResult: Cached or newly compiled result with an accurate cache status.
    """
    probe = _probe_cache(backend, unit, context, cache_root=cache_root)
    if probe.cached_result is not None:
        return probe.cached_result
    result = backend.compile((unit,), context)
    return _finalize_miss(
        probe,
        result,
        unit=unit,
        context=context,
        cache_root=cache_root,
    )


def probe_region_cache(
    backend: CompilerBackend,
    unit: CompilationUnit,
    context: BackendCompileContext,
    *,
    cache_root: Path,
) -> BackendCompileResult | None:
    """Restore one cached result without invoking the compiler on a miss.

    Successful probes restore validated artifact bytes into the active backend
    output root. Deterministic rejection probes restore their structured scope.
    A miss returns ``None`` and leaves backend execution to the caller.

    Args:
        backend: Compiler backend that owns the fingerprint and cache namespace.
        unit: Content-addressable backend compilation unit.
        context: Filesystem, cache, and artifact restoration boundaries.
        cache_root: Root directory for typed-region artifacts and decisions.

    Returns:
        BackendCompileResult | None: Restored success or rejection, otherwise ``None``.
    """
    return _probe_cache(backend, unit, context, cache_root=cache_root).cached_result


def compile_many_with_region_cache(
    backend: CompilerBackend,
    units: tuple[CompilationUnit, ...],
    context: BackendCompileContext,
    *,
    cache_root: Path,
) -> tuple[BackendCompileResult, ...]:
    """Restore independent entries and batch compatible cold Cython misses.

    A fully warm call never reaches the backend. Cold Cython misses share one
    compiler invocation; deterministic batch failures are recursively bisected
    in isolated output roots until successful subsets and failing singletons can
    be cached independently. Other backends retain their one-unit behavior.

    Args:
        backend: Compiler backend selected for every supplied unit.
        units: Ordered backend units with independent fingerprints and cache entries.
        context: Shared source and toolchain context for compilation.
        cache_root: Root directory for typed-region artifact and decision entries.

    Returns:
        tuple[BackendCompileResult, ...]: One result per input unit in matching order.

    Raises:
        ValueError: If units mix backends or duplicate identities needed for partitioning.
    """
    if not units:
        return ()
    if any(unit.backend != backend.name for unit in units):
        raise ValueError("batched cache units must use the selected backend")
    if backend.name != "cython" or len(units) == 1:
        return tuple(
            compile_with_region_cache(
                backend,
                unit,
                context,
                cache_root=cache_root,
            )
            for unit in units
        )
    _validate_batch_identities(units)
    results: list[BackendCompileResult | None] = [None] * len(units)
    misses: list[_BatchMiss] = []
    for index, unit in enumerate(units):
        probe = _probe_cache(backend, unit, context, cache_root=cache_root)
        if probe.cached_result is not None:
            results[index] = probe.cached_result
        else:
            misses.append(_BatchMiss(index=index, unit=unit, probe=probe))
    if misses:
        for miss, result in _compile_cython_misses(
            backend,
            tuple(misses),
            context,
            cache_root=cache_root,
            sequence=count(1),
        ):
            results[miss.index] = result
    if any(result is None for result in results):
        raise AssertionError("batched region cache left an input without a result")
    return tuple(cast(BackendCompileResult, result) for result in results)


def _probe_cache(
    backend: CompilerBackend,
    unit: CompilationUnit,
    context: BackendCompileContext,
    *,
    cache_root: Path,
) -> _CacheProbe:
    lookup_started = time.perf_counter()
    key = backend.fingerprint(unit, context)
    identity = _CacheIdentity(
        entry_root=cache_root / backend.name / key,
        key=key,
        backend=backend.name,
        region_id=unit.region_id,
    )
    restored = _restore_entry(identity=identity, context=context)
    lookup_duration = time.perf_counter() - lookup_started
    if restored is not None:
        restore_duration, artifact_paths, artifact_records = restored
        return _CacheProbe(
            identity=identity,
            lookup_duration_seconds=lookup_duration,
            cached_result=BackendCompileResult(
                attempt=CompileAttempt(
                    success=True,
                    command=("atoll", "cache", "restore", backend.name, unit.region_id),
                    stdout="",
                    stderr="",
                    artifact_paths=artifact_paths,
                    duration_seconds=lookup_duration + restore_duration,
                    phase_timings=(
                        CompilePhaseTiming(
                            name="cache_lookup",
                            duration_seconds=lookup_duration,
                            detail=f"hit; {unit.region_id}",
                        ),
                        CompilePhaseTiming(
                            name="cache_restore",
                            duration_seconds=restore_duration,
                            detail=f"{len(artifact_paths)} artifact(s); {unit.region_id}",
                        ),
                    ),
                    cache_status="hit",
                ),
                artifacts=artifact_records,
            ),
        )
    cached_rejection = _restore_rejection(
        path=_decision_path(cache_root, identity),
        identity=identity,
        context=context,
        duration_seconds=lookup_duration,
    )
    return _CacheProbe(
        identity=identity,
        lookup_duration_seconds=lookup_duration,
        cached_result=cached_rejection,
    )


def _finalize_miss(
    probe: _CacheProbe,
    result: BackendCompileResult,
    *,
    unit: CompilationUnit,
    context: BackendCompileContext,
    cache_root: Path,
) -> BackendCompileResult:
    identity = probe.identity
    lookup_duration = probe.lookup_duration_seconds
    decision_path = _decision_path(cache_root, identity)
    miss_timing = CompilePhaseTiming(
        name="cache_lookup",
        duration_seconds=lookup_duration,
        detail=f"miss; {identity.region_id}",
    )
    attempt = replace(
        result.attempt,
        cache_status="miss",
        phase_timings=(miss_timing, *result.attempt.phase_timings),
    )
    if not result.attempt.success:
        code = _deterministic_diagnostic_code(result.attempt.stderr)
        if code is None:
            return replace(result, attempt=attempt)
        diagnostic_scope = _diagnostic_scope(
            code=code,
            stderr=result.attempt.stderr,
            unit=unit,
            context=context,
        )
        store_started = time.perf_counter()
        store_detail = _store_rejection(
            path=decision_path,
            identity=identity,
            diagnostic_code=code,
            diagnostic_scope=diagnostic_scope,
            project_source_digest=context.project_source_digest,
        )
        store_timing = CompilePhaseTiming(
            name="backend_decision_store",
            duration_seconds=time.perf_counter() - store_started,
            detail=f"{store_detail}; {identity.region_id}",
        )
        return replace(
            result,
            attempt=replace(
                attempt,
                duration_seconds=attempt.duration_seconds + store_timing.duration_seconds,
                phase_timings=(*attempt.phase_timings, store_timing),
            ),
            diagnostic_scope=diagnostic_scope,
        )
    if not result.attempt.artifact_paths:
        return replace(result, attempt=attempt)

    store_started = time.perf_counter()
    store_detail = _store_entry(
        identity=identity,
        context=context,
        result=result,
    )
    store_timing = CompilePhaseTiming(
        name="cache_store",
        duration_seconds=time.perf_counter() - store_started,
        detail=f"{store_detail}; {identity.region_id}",
    )
    return replace(
        result,
        attempt=replace(
            attempt,
            duration_seconds=attempt.duration_seconds + store_timing.duration_seconds,
            phase_timings=(*attempt.phase_timings, store_timing),
        ),
    )


def _compile_cython_misses(
    backend: CompilerBackend,
    misses: tuple[_BatchMiss, ...],
    context: BackendCompileContext,
    *,
    cache_root: Path,
    sequence: Iterator[int],
) -> tuple[tuple[_BatchMiss, BackendCompileResult], ...]:
    """Compile one miss subset and bisect deterministic aggregate failures.

    Args:
        backend: Cython adapter used for the physical build.
        misses: Ordered cache misses submitted together.
        context: Parent compile context whose output tree remains disposable.
        cache_root: Independent per-unit cache namespace.
        sequence: Monotonic invocation IDs used to isolate recursive output roots.

    Returns:
        tuple[tuple[_BatchMiss, BackendCompileResult], ...]: Per-unit results in miss order.
    """
    invocation = next(sequence)
    batch_context = _batch_context(context, invocation=invocation, unit_count=len(misses))
    physical = backend.compile(tuple(miss.unit for miss in misses), batch_context)
    if physical.attempt.success:
        try:
            split_results = _split_successful_batch(
                misses,
                physical,
                invocation=invocation,
            )
        except ValueError as error:
            physical = BackendCompileResult(
                attempt=replace(
                    physical.attempt,
                    success=False,
                    stderr=f"CYTHON_COMPILE_ERROR: invalid batch artifact parity: {error}",
                    artifact_paths=(),
                ),
                artifacts=(),
            )
        else:
            return tuple(
                (
                    miss,
                    _finalize_miss(
                        miss.probe,
                        result,
                        unit=miss.unit,
                        context=batch_context,
                        cache_root=cache_root,
                    ),
                )
                for miss, result in zip(misses, split_results, strict=True)
            )
    deterministic = _deterministic_diagnostic_code(physical.attempt.stderr) is not None
    if deterministic and len(misses) > 1:
        midpoint = len(misses) // 2
        isolated = (
            *_compile_cython_misses(
                backend,
                misses[:midpoint],
                context,
                cache_root=cache_root,
                sequence=sequence,
            ),
            *_compile_cython_misses(
                backend,
                misses[midpoint:],
                context,
                cache_root=cache_root,
                sequence=sequence,
            ),
        )
        return _retain_failed_batch_invocation(
            isolated,
            physical=physical,
            invocation=invocation,
            unit_count=len(misses),
        )
    failures = _split_failed_batch(misses, physical, invocation=invocation)
    return tuple(
        (
            miss,
            _finalize_miss(
                miss.probe,
                result,
                unit=miss.unit,
                context=batch_context,
                cache_root=cache_root,
            ),
        )
        for miss, result in zip(misses, failures, strict=True)
    )


def _retain_failed_batch_invocation(
    results: tuple[tuple[_BatchMiss, BackendCompileResult], ...],
    *,
    physical: BackendCompileResult,
    invocation: int,
    unit_count: int,
) -> tuple[tuple[_BatchMiss, BackendCompileResult], ...]:
    """Attach one failed parent process to the first isolated child result.

    Recursive bisection replaces a failed aggregate result with independently
    cacheable child results. This helper preserves the parent's duration and one
    explicit timing marker without persisting that transient aggregate evidence
    into any per-unit cache entry.

    Args:
        results: Ordered isolated child results from both recursive halves.
        physical: Failed aggregate Cython invocation being represented.
        invocation: Monotonic physical process identifier.
        unit_count: Number of units submitted to the failed process.

    Returns:
        tuple[tuple[_BatchMiss, BackendCompileResult], ...]: Results with exact
        cold-process timing retained once.
    """
    if not results:
        return results
    first_miss, first_result = results[0]
    retry_timing = CompilePhaseTiming(
        name="cython_batch_retry",
        duration_seconds=0.0,
        detail=f"failed invocation {invocation}; {unit_count} unit(s); recursively isolated",
    )
    first_attempt = replace(
        first_result.attempt,
        duration_seconds=(
            first_result.attempt.duration_seconds + physical.attempt.duration_seconds
        ),
        phase_timings=(
            *physical.attempt.phase_timings,
            retry_timing,
            *first_result.attempt.phase_timings,
        ),
    )
    return (
        (first_miss, replace(first_result, attempt=first_attempt)),
        *results[1:],
    )


def _split_successful_batch(
    misses: tuple[_BatchMiss, ...],
    physical: BackendCompileResult,
    *,
    invocation: int,
) -> tuple[BackendCompileResult, ...]:
    artifact_by_digest = {_file_digest(path): path for path in physical.attempt.artifact_paths}
    results: list[BackendCompileResult] = []
    batch = _BatchInvocation(number=invocation, unit_count=len(misses))
    for index, miss in enumerate(misses):
        unit = miss.unit
        records = tuple(
            record
            for record in physical.artifacts
            if record.region_id == unit.region_id
            or (
                record.region_id == "__shared__"
                and _record_belongs_to_install_dir(record, unit.install_relative_dir)
            )
        )
        if not any(
            record.region_id == unit.region_id and record.role == "primary" for record in records
        ):
            raise ValueError(f"{unit.region_id} has no primary artifact record")
        try:
            artifact_paths = tuple(
                dict.fromkeys(artifact_by_digest[record.digest] for record in records)
            )
        except KeyError as error:
            raise ValueError(
                f"{unit.region_id} references an unavailable artifact digest"
            ) from error
        attempt = _unit_batch_attempt(
            physical.attempt,
            artifact_paths=artifact_paths,
            batch=batch,
            unit_index=index,
            region_id=unit.region_id,
        )
        results.append(BackendCompileResult(attempt=attempt, artifacts=records))
    return tuple(results)


def _split_failed_batch(
    misses: tuple[_BatchMiss, ...],
    physical: BackendCompileResult,
    *,
    invocation: int,
) -> tuple[BackendCompileResult, ...]:
    batch = _BatchInvocation(number=invocation, unit_count=len(misses))
    return tuple(
        BackendCompileResult(
            attempt=_unit_batch_attempt(
                physical.attempt,
                artifact_paths=(),
                batch=batch,
                unit_index=index,
                region_id=miss.unit.region_id,
            ),
            artifacts=(),
        )
        for index, miss in enumerate(misses)
    )


def _unit_batch_attempt(
    physical: CompileAttempt,
    *,
    artifact_paths: tuple[Path, ...],
    batch: _BatchInvocation,
    unit_index: int,
    region_id: str,
) -> CompileAttempt:
    owner = unit_index == 0
    batch_timing = CompilePhaseTiming(
        name="cython_batch" if owner else "cython_batch_member",
        duration_seconds=0.0,
        detail=(
            f"invocation {batch.number}; {batch.unit_count} unit(s); {region_id}; "
            f"physical_timing={'owner' if owner else 'reported-on-first-unit'}"
        ),
    )
    return replace(
        physical,
        stdout=physical.stdout if owner else "",
        artifact_paths=artifact_paths,
        duration_seconds=physical.duration_seconds if owner else 0.0,
        phase_timings=(*physical.phase_timings, batch_timing) if owner else (batch_timing,),
    )


def _batch_context(
    context: BackendCompileContext,
    *,
    invocation: int,
    unit_count: int,
) -> BackendCompileContext:
    root = context.build_dir.parent / "cython-batches" / f"{invocation:04d}-{unit_count}"
    shutil.rmtree(root, ignore_errors=True)
    return replace(context, build_dir=root / "build")


def _record_belongs_to_install_dir(record: ArtifactRecord, install_relative_dir: str) -> bool:
    record_parts = PurePosixPath(record.install_relative_path).parts
    install_parts = PurePosixPath(install_relative_dir).parts
    return not install_parts or record_parts[: len(install_parts)] == install_parts


def _validate_batch_identities(units: tuple[CompilationUnit, ...]) -> None:
    for label, values in (
        ("region IDs", tuple(unit.region_id for unit in units)),
        ("logical modules", tuple(unit.logical_module for unit in units)),
        ("install directories", tuple(unit.install_relative_dir for unit in units)),
    ):
        if len(set(values)) != len(values):
            raise ValueError(f"batched Cython units require unique {label}")


def _decision_path(cache_root: Path, identity: _CacheIdentity) -> Path:
    decision_root = (
        cache_root.parent / "decisions"
        if cache_root.name == "regions"
        else cache_root / "decisions"
    )
    return decision_root / identity.backend / f"{identity.key}.json"


def _restore_rejection(
    *,
    path: Path,
    identity: _CacheIdentity,
    context: BackendCompileContext,
    duration_seconds: float,
) -> BackendCompileResult | None:
    manifest = _read_manifest(path)
    if (
        manifest is None
        or manifest.get("version") != BACKEND_DECISION_CACHE_VERSION
        or manifest.get("key") != identity.key
        or manifest.get("backend") != identity.backend
        or manifest.get("region_id") != identity.region_id
    ):
        return None
    code = manifest.get("diagnostic_code")
    if not isinstance(code, str) or code not in _DETERMINISTIC_DIAGNOSTIC_CODES:
        return None
    diagnostic_scope = manifest.get("diagnostic_scope")
    if diagnostic_scope not in {"unit", "project"}:
        return None
    restored_scope: BackendDiagnosticScope = "project" if diagnostic_scope == "project" else "unit"
    project_source_digest = manifest.get("project_source_digest")
    if restored_scope == "project":
        if (
            not isinstance(project_source_digest, str)
            or project_source_digest != context.project_source_digest
        ):
            return None
    elif project_source_digest is not None:
        return None
    return BackendCompileResult(
        attempt=CompileAttempt(
            success=False,
            command=("atoll", "cache", "reject", identity.backend, identity.region_id),
            stdout="",
            stderr=f"{code}: cached deterministic rejection for {identity.region_id}",
            artifact_paths=(),
            duration_seconds=duration_seconds,
            phase_timings=(
                CompilePhaseTiming(
                    name="backend_decision_cache",
                    duration_seconds=duration_seconds,
                    detail=f"hit; {code}; {identity.region_id}",
                ),
            ),
            cache_status="hit",
        ),
        artifacts=(),
        diagnostic_scope=restored_scope,
    )


def _store_rejection(
    *,
    path: Path,
    identity: _CacheIdentity,
    diagnostic_code: str,
    diagnostic_scope: BackendDiagnosticScope,
    project_source_digest: str | None,
) -> str:
    if diagnostic_scope == "project" and not project_source_digest:
        return "project-scoped rejection not stored because source digest is unavailable"
    temp_path = path.with_suffix(".tmp")
    manifest = {
        "version": BACKEND_DECISION_CACHE_VERSION,
        "key": identity.key,
        "backend": identity.backend,
        "region_id": identity.region_id,
        "diagnostic_code": diagnostic_code,
        "diagnostic_scope": diagnostic_scope,
        "project_source_digest": (project_source_digest if diagnostic_scope == "project" else None),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_text(
            f"{json.dumps(manifest, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
        temp_path.replace(path)
    except OSError as error:
        temp_path.unlink(missing_ok=True)
        return f"decision store failed: {error}"
    return f"stored deterministic {diagnostic_code} {diagnostic_scope}-scoped rejection"


def _diagnostic_scope(
    *,
    code: str,
    stderr: str,
    unit: CompilationUnit,
    context: BackendCompileContext,
) -> BackendDiagnosticScope:
    """Classify mypyc diagnostics that originate in imported project source.

    Mypyc type-checks the import graph around each generated unit. When its
    normalized diagnostics point at copied target-project files but not at the
    generated unit itself, retrying another unit from the same package repeats
    the same deterministic project rejection. Other backends and ambiguous
    paths remain unit-scoped.

    Args:
        code: Normalized deterministic backend diagnostic code.
        stderr: Backend diagnostic summary retained by the compile attempt.
        unit: Generated compilation unit submitted to the backend.
        context: Copied project and source-root boundaries used by the backend.

    Returns:
        BackendDiagnosticScope: ``project`` only for verified copied-source errors.
    """
    if unit.backend != "mypyc" or code != "MYPYC_TYPE_ERROR":
        return "unit"
    generated_paths = frozenset(path.resolve() for path in unit.source_paths)
    source_roots = tuple(path.resolve() for path in context.source_roots)
    generated_error = False
    project_error = False
    for match in re.finditer(r"(?m)^(.+?\.py):\d+(?::\d+)?: error:", stderr):
        raw_path = Path(match.group(1))
        candidates = (
            (raw_path.resolve(),)
            if raw_path.is_absolute()
            else (
                (context.project_root / raw_path).resolve(),
                *((root / raw_path).resolve() for root in source_roots),
            )
        )
        for candidate in candidates:
            if candidate in generated_paths:
                generated_error = True
                break
            if not candidate.is_file():
                continue
            if any(candidate.is_relative_to(root) for root in source_roots):
                project_error = True
                break
    return "project" if project_error and not generated_error else "unit"


def _deterministic_diagnostic_code(stderr: str) -> str | None:
    return next(
        (code for code in _DETERMINISTIC_DIAGNOSTIC_CODES if stderr.startswith(f"{code}:")),
        None,
    )


def _restore_entry(
    *,
    identity: _CacheIdentity,
    context: BackendCompileContext,
) -> tuple[float, tuple[Path, ...], tuple[ArtifactRecord, ...]] | None:
    manifest = _read_manifest(identity.entry_root / "manifest.json")
    if (
        manifest is None
        or manifest.get("version") != REGION_CACHE_VERSION
        or manifest.get("key") != identity.key
        or manifest.get("backend") != identity.backend
        or manifest.get("region_id") != identity.region_id
    ):
        return None
    files = _manifest_files(manifest)
    records = _manifest_records(manifest)
    if files is None or records is None or not files:
        return None
    restore_started = time.perf_counter()
    output_root = _artifact_output_root(context)
    restored: list[Path] = []
    for relative, digest in files:
        cached = identity.entry_root / "artifacts" / relative
        if not cached.is_file() or _file_digest(cached) != digest:
            return None
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, destination)
        if _file_digest(destination) != digest:
            return None
        restored.append(destination)
    return time.perf_counter() - restore_started, tuple(restored), records


def _store_entry(
    *,
    identity: _CacheIdentity,
    context: BackendCompileContext,
    result: BackendCompileResult,
) -> str:
    output_root = _artifact_output_root(context)
    files: list[dict[str, str]] = []
    try:
        relative_artifacts = tuple(
            (artifact, _safe_relative(artifact.resolve().relative_to(output_root.resolve())))
            for artifact in result.attempt.artifact_paths
        )
    except (TypeError, ValueError):
        return "not stored because an artifact is outside the backend output root"
    temp_root = identity.entry_root.with_name(f"{identity.entry_root.name}.tmp")
    try:
        if temp_root.exists():
            shutil.rmtree(temp_root)
        artifact_root = temp_root / "artifacts"
        artifact_root.mkdir(parents=True)
        for artifact, relative in relative_artifacts:
            destination = artifact_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(artifact, destination)
            files.append({"path": relative.as_posix(), "digest": _file_digest(destination)})
        manifest = {
            "version": REGION_CACHE_VERSION,
            "key": identity.key,
            "backend": identity.backend,
            "region_id": identity.region_id,
            "files": files,
            "artifacts": [_artifact_record_data(record) for record in result.artifacts],
        }
        (temp_root / "manifest.json").write_text(
            f"{json.dumps(manifest, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
        identity.entry_root.parent.mkdir(parents=True, exist_ok=True)
        if identity.entry_root.exists():
            shutil.rmtree(identity.entry_root)
        temp_root.rename(identity.entry_root)
    except OSError as error:
        shutil.rmtree(temp_root, ignore_errors=True)
        return f"store failed: {error}"
    return f"stored {len(files)} artifact(s)"


def _read_manifest(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    raw = cast(dict[object, object], value)
    return {str(key): item for key, item in raw.items()}


def _manifest_files(manifest: dict[str, object]) -> tuple[tuple[Path, str], ...] | None:
    value = manifest.get("files")
    if not isinstance(value, list):
        return None
    files: list[tuple[Path, str]] = []
    for item in cast(list[object], value):
        try:
            data = _object_mapping(item)
        except TypeError:
            return None
        path_text = data.get("path")
        digest = data.get("digest")
        if not isinstance(path_text, str) or not isinstance(digest, str):
            return None
        try:
            path = _safe_relative(Path(path_text))
        except ValueError:
            return None
        files.append((path, digest))
    return tuple(files)


def _manifest_records(manifest: dict[str, object]) -> tuple[ArtifactRecord, ...] | None:
    value = manifest.get("artifacts")
    if not isinstance(value, list):
        return None
    records: list[ArtifactRecord] = []
    try:
        for item in cast(list[object], value):
            data = _object_mapping(item)
            records.append(
                ArtifactRecord(
                    region_id=_required_string(data, "region_id"),
                    backend=_backend(data.get("backend")),
                    logical_module=_required_string(data, "logical_module"),
                    role=_artifact_role(data.get("role")),
                    install_relative_path=_required_string(data, "install_relative_path"),
                    digest=_required_string(data, "digest"),
                    abi=_required_string(data, "abi"),
                    platform_tag=_required_string(data, "platform_tag"),
                )
            )
    except (TypeError, ValueError):
        return None
    return tuple(records)


def _artifact_record_data(record: ArtifactRecord) -> dict[str, str]:
    return {
        "region_id": record.region_id,
        "backend": record.backend,
        "logical_module": record.logical_module,
        "role": record.role,
        "install_relative_path": record.install_relative_path,
        "digest": record.digest,
        "abi": record.abi,
        "platform_tag": record.platform_tag,
    }


def _artifact_output_root(context: BackendCompileContext) -> Path:
    return context.build_dir.parent / "artifacts"


def _safe_relative(path: Path) -> Path:
    pure = PurePosixPath(path.as_posix())
    if path.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError("cache artifact path must be relative and traversal-free")
    return Path(*pure.parts)


def _object_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("cache manifest entry must be an object")
    raw = cast(dict[object, object], value)
    return {str(key): item for key, item in raw.items()}


def _required_string(data: dict[str, object], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str):
        raise TypeError(f"cache artifact {field} must be a string")
    return value


def _backend(value: object) -> Backend:
    if value == "mypyc":
        return "mypyc"
    if value == "cython":
        return "cython"
    raise ValueError("cache artifact backend is invalid")


def _artifact_role(value: object) -> ArtifactRole:
    if value == "primary":
        return "primary"
    if value == "support":
        return "support"
    raise ValueError("cache artifact role is invalid")


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
