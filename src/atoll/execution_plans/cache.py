"""Persistent cache for staged execution-plan payload files.

The cache stores only successful backend staging outputs. It is rooted at an
explicit caller-owned path so restore never needs to write into the checkout,
and every entry is keyed by the backend-supplied fingerprint plus plan, backend,
and Python identity metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, TypedDict, cast

from atoll.execution_plans.models import (
    ChangedPayloadFile,
    ExecutionPlan,
    ExecutionPlanCacheStatus,
    ExecutionPlanStageContext,
    PlanGuard,
    StagedExecutionPlan,
)

_CACHE_VERSION = "1"
_SHA256_HEX_LENGTH = 64
_METADATA_FILE = "metadata.json"
_PAYLOAD_DIR = "payload"


class _PlanMetadata(TypedDict):
    id: str
    source_module: str
    source_hash: str
    callsite_fingerprint: str
    topology_fingerprint: str
    dialect: str
    lowering_version: str


class _PayloadFileMetadata(TypedDict):
    install_path: str
    before_hash: str | None
    after_hash: str
    role: str


class _CacheMetadata(TypedDict):
    cache_version: str
    key: str
    fingerprint: str
    backend: str
    python_identity: str
    plan: _PlanMetadata
    required_imports: list[str]
    guards: list[dict[str, str]]
    payload_files: list[_PayloadFileMetadata]


@dataclass(frozen=True, slots=True)
class ExecutionPlanCacheState:
    """Resolved immutable cache entry state.

    Attributes:
        cache_root: Caller-owned root directory that contains execution-plan cache entries.
        entry_root: Directory for the strict key derived from plan, backend,
            Python, and fingerprint.
        metadata_path: JSON metadata path for the entry.
        key: Strict cache key used for lookup.
        fingerprint: Backend-supplied fingerprint recorded in the entry.
        plan_id: Stable execution-plan identifier recorded in the entry.
        backend: Backend identifier recorded in the entry.
        python_identity: Python implementation and ABI identity recorded in the entry.
    """

    cache_root: Path
    entry_root: Path
    metadata_path: Path
    key: str
    fingerprint: str
    plan_id: str
    backend: str
    python_identity: str


@dataclass(frozen=True, slots=True)
class ExecutionPlanCacheResult:
    """Outcome of an execution-plan cache restore attempt.

    Attributes:
        status: `hit`, `miss`, or `invalid` restore classification.
        state: Immutable cache entry state that was looked up.
        staged: Restored staged execution plan for cache hits, otherwise `None`.
        reason: Short reason for misses and invalid entries.
    """

    status: ExecutionPlanCacheStatus
    state: ExecutionPlanCacheState
    staged: StagedExecutionPlan | None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class _RestorablePayloadFile:
    """Validated cached bytes and destination state, before any payload write.

    Attributes:
        changed_file: Staged payload file metadata recorded by the backend.
        target: Destination path inside the unpacked payload tree.
        content: Cached bytes that should be restored to the destination.
        before_content: Original destination bytes used for rollback, or `None`
            when the restore creates a new file.
    """

    changed_file: ChangedPayloadFile
    target: Path
    content: bytes
    before_content: bytes | None


def restore_execution_plan_cache(
    caller_path: Path,
    context: ExecutionPlanStageContext,
    plan: ExecutionPlan,
    *,
    backend: str,
    fingerprint: str,
) -> ExecutionPlanCacheResult:
    """Restore generated payload files from the execution-plan cache.

    Args:
        caller_path: Caller-owned cache root. The cache never derives this from
            the project checkout, which keeps restore writes confined to the
            staged payload root.
        context: Staging context whose payload root receives restored files.
        plan: Execution plan whose identity must match the cache metadata.
        backend: Stable execution-plan backend identifier.
        fingerprint: Strict backend-supplied fingerprint for the staged output.

    Returns:
        ExecutionPlanCacheResult: Hit, miss, or invalid result with immutable
        lookup state. Cache hits include a restored staged plan.
    """
    state = _cache_state(caller_path, plan, backend, fingerprint)
    if not state.metadata_path.exists():
        return ExecutionPlanCacheResult(status="miss", state=state, staged=None, reason="absent")
    try:
        metadata = _validate_metadata(
            _read_metadata(state.metadata_path),
            state,
            plan,
            backend,
            fingerprint,
        )
        files = tuple(_changed_file(file_metadata) for file_metadata in metadata["payload_files"])
        restorable_files = tuple(
            _restorable_payload_file(state, context.payload_root, file_metadata)
            for file_metadata in metadata["payload_files"]
        )
        _restore_payload_files(restorable_files)
        restored_files = tuple(item.changed_file for item in restorable_files)
        staged = StagedExecutionPlan(
            plan=plan,
            backend=backend,
            payload_files=restored_files,
            required_imports=tuple(metadata["required_imports"]),
            guards=tuple(
                PlanGuard(
                    kind=cast(
                        Literal["scheduler", "transport", "topology", "semantics"],
                        guard["kind"],
                    ),
                    expression=guard["expression"],
                    message=guard["message"],
                )
                for guard in metadata["guards"]
            ),
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return ExecutionPlanCacheResult(
            status="invalid",
            state=state,
            staged=None,
            reason=str(error) or error.__class__.__name__,
        )
    if files != staged.payload_files:
        return ExecutionPlanCacheResult(
            status="invalid",
            state=state,
            staged=None,
            reason="restored payload metadata changed",
        )
    return ExecutionPlanCacheResult(status="hit", state=state, staged=staged)


def store_execution_plan_cache(
    caller_path: Path,
    context: ExecutionPlanStageContext,
    staged: StagedExecutionPlan,
    *,
    fingerprint: str,
) -> ExecutionPlanCacheState:
    """Store a successful staged execution plan in the caller-owned cache.

    Args:
        caller_path: Caller-owned cache root for persistent execution-plan entries.
        context: Staging context containing the generated payload files.
        staged: Successful backend staging output to cache. Failed staging and
            trial diagnostics are intentionally not representable here.
        fingerprint: Strict backend-supplied fingerprint for the staged output.

    Returns:
        ExecutionPlanCacheState: Immutable cache entry state written to disk.

    Raises:
        ValueError: If staged payload paths are unsafe, missing, symlinks, or
            have digests that do not match the staged metadata.
        OSError: If the cache entry cannot be written atomically.
    """
    state = _cache_state(caller_path, staged.plan, staged.backend, fingerprint)
    payload_metadata = [
        _payload_file_metadata(context.payload_root, changed_file)
        for changed_file in staged.payload_files
    ]
    metadata = _metadata(state, staged, payload_metadata)
    state.entry_root.mkdir(parents=True, exist_ok=True)
    payload_root = state.entry_root / _PAYLOAD_DIR
    payload_root.mkdir(exist_ok=True)
    for file_metadata in payload_metadata:
        source = _safe_child(context.payload_root, file_metadata["install_path"])
        target = _safe_child(payload_root, file_metadata["install_path"])
        _write_file_atomic(target, source.read_bytes())
    _write_json_atomic(state.metadata_path, metadata)
    return state


def _cache_state(
    caller_path: Path,
    plan: ExecutionPlan,
    backend: str,
    fingerprint: str,
) -> ExecutionPlanCacheState:
    if not backend:
        raise ValueError("execution-plan cache backend is empty")
    if not fingerprint:
        raise ValueError("execution-plan cache fingerprint is empty")
    python_identity = _python_identity()
    key = _cache_key(plan, backend, fingerprint, python_identity)
    cache_root = caller_path.expanduser()
    entry_root = cache_root / key
    return ExecutionPlanCacheState(
        cache_root=cache_root,
        entry_root=entry_root,
        metadata_path=entry_root / _METADATA_FILE,
        key=key,
        fingerprint=fingerprint,
        plan_id=plan.id,
        backend=backend,
        python_identity=python_identity,
    )


def _cache_key(
    plan: ExecutionPlan,
    backend: str,
    fingerprint: str,
    python_identity: str,
) -> str:
    digest = hashlib.sha256()
    for part in (
        _CACHE_VERSION,
        fingerprint,
        backend,
        python_identity,
        plan.id,
        plan.source_module,
        plan.source_hash,
        plan.callsite_fingerprint,
        plan.topology_fingerprint,
        plan.dialect,
        plan.lowering_version,
    ):
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _python_identity() -> str:
    return (
        f"{sys.implementation.name}:"
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}:"
        f"{sys.abiflags}"
    )


def _metadata(
    state: ExecutionPlanCacheState,
    staged: StagedExecutionPlan,
    payload_files: list[_PayloadFileMetadata],
) -> _CacheMetadata:
    return {
        "cache_version": _CACHE_VERSION,
        "key": state.key,
        "fingerprint": state.fingerprint,
        "backend": state.backend,
        "python_identity": state.python_identity,
        "plan": _plan_metadata(staged.plan),
        "required_imports": list(staged.required_imports),
        "guards": [
            {"kind": guard.kind, "expression": guard.expression, "message": guard.message}
            for guard in staged.guards
        ],
        "payload_files": payload_files,
    }


def _plan_metadata(plan: ExecutionPlan) -> _PlanMetadata:
    return {
        "id": plan.id,
        "source_module": plan.source_module,
        "source_hash": plan.source_hash,
        "callsite_fingerprint": plan.callsite_fingerprint,
        "topology_fingerprint": plan.topology_fingerprint,
        "dialect": plan.dialect,
        "lowering_version": plan.lowering_version,
    }


def _read_metadata(path: Path) -> dict[str, object]:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("cache metadata is not an object")
    return cast(dict[str, object], raw)


def _validate_metadata(
    metadata: dict[str, object],
    state: ExecutionPlanCacheState,
    plan: ExecutionPlan,
    backend: str,
    fingerprint: str,
) -> _CacheMetadata:
    expected = {
        "cache_version": _CACHE_VERSION,
        "key": state.key,
        "fingerprint": fingerprint,
        "backend": backend,
        "python_identity": state.python_identity,
        "plan": _plan_metadata(plan),
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(f"cache metadata {field} mismatch")
    _validate_required_imports(_required_list(metadata, "required_imports"))
    _validate_guard_metadata(_required_list(metadata, "guards"))
    _validate_payload_file_metadata(_required_list(metadata, "payload_files"))
    return cast(_CacheMetadata, metadata)


def _required_list(metadata: dict[str, object], field: str) -> list[object]:
    value = metadata.get(field)
    if not isinstance(value, list):
        raise TypeError(f"cache metadata {field} is invalid")
    return cast(list[object], value)


def _validate_required_imports(required_imports: list[object]) -> None:
    for required_import in required_imports:
        if not isinstance(required_import, str):
            raise TypeError("cache metadata required import is invalid")


def _validate_guard_metadata(guards: list[object]) -> None:
    for guard in guards:
        if not isinstance(guard, dict):
            raise TypeError("cache metadata guard is invalid")
        guard_map = cast(dict[object, object], guard)
        if set(guard_map) != {"kind", "expression", "message"}:
            raise ValueError("cache metadata guard is invalid")
        kind = guard_map["kind"]
        expression = guard_map["expression"]
        message = guard_map["message"]
        if (
            kind not in {"scheduler", "transport", "topology", "semantics"}
            or not isinstance(expression, str)
            or not isinstance(message, str)
        ):
            raise ValueError("cache metadata guard is invalid")


def _validate_payload_file_metadata(payload_files: list[object]) -> None:
    for file_metadata in payload_files:
        if not isinstance(file_metadata, dict):
            raise TypeError("cache metadata payload file is invalid")
        _changed_file(cast(_PayloadFileMetadata, file_metadata))


def _payload_file_metadata(
    payload_root: Path,
    changed_file: ChangedPayloadFile,
) -> _PayloadFileMetadata:
    install_path = _install_path(changed_file.install_path)
    payload_path = _safe_child(payload_root, install_path)
    if not payload_path.exists() or not payload_path.is_file():
        raise ValueError(f"staged payload file is missing: {install_path}")
    if payload_path.is_symlink():
        raise ValueError(f"staged payload file is a symlink: {install_path}")
    after_hash = _sha256_bytes(payload_path.read_bytes())
    if after_hash != changed_file.after_hash:
        raise ValueError(f"staged payload digest mismatch: {install_path}")
    return {
        "install_path": install_path,
        "before_hash": changed_file.before_hash,
        "after_hash": changed_file.after_hash,
        "role": changed_file.role,
    }


def _changed_file(file_metadata: _PayloadFileMetadata) -> ChangedPayloadFile:
    install_path = _install_path(file_metadata["install_path"])
    before_hash = file_metadata["before_hash"]
    after_hash = file_metadata["after_hash"]
    role = file_metadata["role"]
    if before_hash is not None and not _is_hex_digest(before_hash):
        raise ValueError(f"cache metadata before digest is invalid: {install_path}")
    if not _is_hex_digest(after_hash):
        raise ValueError(f"cache metadata after digest is invalid: {install_path}")
    if not role:
        raise ValueError(f"cache metadata role is empty: {install_path}")
    return ChangedPayloadFile(
        install_path=PurePosixPath(install_path),
        before_hash=before_hash,
        after_hash=after_hash,
        role=role,
    )


def _restorable_payload_file(
    state: ExecutionPlanCacheState,
    payload_root: Path,
    file_metadata: _PayloadFileMetadata,
) -> _RestorablePayloadFile:
    changed_file = _changed_file(file_metadata)
    install_path = changed_file.install_path.as_posix()
    source = _safe_child(state.entry_root / _PAYLOAD_DIR, install_path)
    target = _safe_child(payload_root, install_path)
    if source.is_symlink():
        raise ValueError(f"cached payload file is a symlink: {install_path}")
    if not source.exists() or not source.is_file():
        raise ValueError(f"cached payload file is missing: {install_path}")
    content = source.read_bytes()
    if _sha256_bytes(content) != changed_file.after_hash:
        raise ValueError(f"cached payload digest mismatch: {install_path}")
    if target.is_symlink():
        raise ValueError(f"staged payload destination is a symlink: {install_path}")
    if changed_file.before_hash is None:
        if target.exists():
            raise ValueError(f"generated payload destination already exists: {install_path}")
        before_content = None
    elif not target.exists() or not target.is_file():
        raise ValueError(f"staged payload destination is missing: {install_path}")
    else:
        before_content = target.read_bytes()
        if _sha256_bytes(before_content) != changed_file.before_hash:
            raise ValueError(f"staged payload before digest mismatch: {install_path}")
    return _RestorablePayloadFile(
        changed_file=changed_file,
        target=target,
        content=content,
        before_content=before_content,
    )


def _restore_payload_files(files: tuple[_RestorablePayloadFile, ...]) -> None:
    """Write a fully validated cache entry and roll back partial I/O failures.

    Args:
        files: Validated cache payload files to write atomically in order.

    Raises:
        OSError: If a payload write fails or rollback cannot restore every
            previously written file.
        ValueError: If an atomic write rejects an unsafe target path.
    """
    written: list[_RestorablePayloadFile] = []
    try:
        for file in files:
            _write_file_atomic(file.target, file.content)
            written.append(file)
    except (OSError, ValueError) as error:
        rollback_error: OSError | ValueError | None = None
        for file in reversed(written):
            try:
                if file.before_content is None:
                    file.target.unlink(missing_ok=True)
                else:
                    _write_file_atomic(file.target, file.before_content)
            except (OSError, ValueError) as current_error:
                rollback_error = current_error
        if rollback_error is not None:
            raise OSError(
                "execution-plan cache restore failed and payload rollback was incomplete"
            ) from error
        raise


def _install_path(path: PurePosixPath | str) -> str:
    install_path = PurePosixPath(str(path))
    if (
        not install_path.parts
        or install_path.is_absolute()
        or ".." in install_path.parts
        or any(part in {"", "."} for part in install_path.parts)
    ):
        raise ValueError(f"unsafe cache install path: {path}")
    return install_path.as_posix()


def _safe_child(root: Path, install_path: str) -> Path:
    child = root.joinpath(*PurePosixPath(install_path).parts)
    resolved_root = root.resolve(strict=False)
    resolved_parent = child.parent.resolve(strict=False)
    if resolved_parent != resolved_root and resolved_root not in resolved_parent.parents:
        raise ValueError(f"cache path escapes root: {install_path}")
    return child


def _write_json_atomic(path: Path, metadata: _CacheMetadata) -> None:
    data = json.dumps(metadata, sort_keys=True, indent=2).encode("utf-8")
    _write_file_atomic(path, data)


def _write_file_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.is_symlink():
        raise ValueError(f"refusing to replace symlink: {path}")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_hex_digest(value: str) -> bool:
    return len(value) == _SHA256_HEX_LENGTH and all(
        character in "0123456789abcdef" for character in value
    )
