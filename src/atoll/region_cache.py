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
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import cast

from atoll.backends.base import CompilerBackend
from atoll.models import (
    ArtifactRecord,
    ArtifactRole,
    Backend,
    BackendCompileContext,
    BackendCompileResult,
    CompilationUnit,
    CompileAttempt,
    CompilePhaseTiming,
)

REGION_CACHE_VERSION = 1
BACKEND_DECISION_CACHE_VERSION = 1
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
    lookup_started = time.perf_counter()
    key = backend.fingerprint(unit, context)
    entry_root = cache_root / backend.name / key
    identity = _CacheIdentity(
        entry_root=entry_root,
        key=key,
        backend=backend.name,
        region_id=unit.region_id,
    )
    restored = _restore_entry(
        identity=identity,
        context=context,
    )
    lookup_duration = time.perf_counter() - lookup_started
    if restored is not None:
        restore_duration, artifact_paths, artifact_records = restored
        return BackendCompileResult(
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
        )

    decision_path = _decision_path(cache_root, identity)
    cached_rejection = _restore_rejection(
        path=decision_path,
        identity=identity,
        duration_seconds=lookup_duration,
    )
    if cached_rejection is not None:
        return cached_rejection

    result = backend.compile((unit,), context)
    miss_timing = CompilePhaseTiming(
        name="cache_lookup",
        duration_seconds=lookup_duration,
        detail=f"miss; {unit.region_id}",
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
        store_started = time.perf_counter()
        store_detail = _store_rejection(
            path=decision_path,
            identity=identity,
            diagnostic_code=code,
        )
        store_timing = CompilePhaseTiming(
            name="backend_decision_store",
            duration_seconds=time.perf_counter() - store_started,
            detail=f"{store_detail}; {unit.region_id}",
        )
        return replace(
            result,
            attempt=replace(
                attempt,
                duration_seconds=attempt.duration_seconds + store_timing.duration_seconds,
                phase_timings=(*attempt.phase_timings, store_timing),
            ),
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
        detail=f"{store_detail}; {unit.region_id}",
    )
    return replace(
        result,
        attempt=replace(
            attempt,
            duration_seconds=attempt.duration_seconds + store_timing.duration_seconds,
            phase_timings=(*attempt.phase_timings, store_timing),
        ),
    )


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
    )


def _store_rejection(
    *,
    path: Path,
    identity: _CacheIdentity,
    diagnostic_code: str,
) -> str:
    temp_path = path.with_suffix(".tmp")
    manifest = {
        "version": BACKEND_DECISION_CACHE_VERSION,
        "key": identity.key,
        "backend": identity.backend,
        "region_id": identity.region_id,
        "diagnostic_code": diagnostic_code,
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
    return f"stored deterministic {diagnostic_code} rejection"


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
