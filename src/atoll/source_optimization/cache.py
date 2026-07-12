"""Strict persistent cache for generated source-optimization patches.

The cache stores only successful `GeneratedSourcePatch` values under an
explicit caller-owned root. Entries are keyed by stable candidate and request
identity, validated against a manifest before use, and rebuilt whenever a
manifest or serialized patch payload is incomplete, corrupt, or stale. The cache
writes only below its caller-owned cache root; project source files are read only
by the transformation builder on misses and invalid entries.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TypedDict, cast

from atoll.models import SymbolId
from atoll.source_optimization.models import SourceEdit
from atoll.source_optimization.transforms import (
    GeneratedSourcePatch,
    SourceTransformationRequest,
    TransformedSourceFile,
    build_source_transformation_patch,
)

_SCHEMA_VERSION = "1"
_MANIFEST_FILE = "manifest.json"
_SHA256_HEX_LENGTH = 64
_RESTORE_ARGUMENT_COUNT = 4


class _SymbolMetadata(TypedDict):
    module: str
    qualname: str


class _CallableReplacementMetadata(TypedDict):
    target: _SymbolMetadata
    declaration_kind: str
    replacement_body: str


class _RequestMetadata(TypedDict):
    path: str
    expected_sha256: str
    target: _SymbolMetadata
    declaration_kind: str
    replacement_body: str
    helper_statements: list[str]
    trailing_statements: list[str]
    additional_replacements: list[_CallableReplacementMetadata]
    summary: str
    transformation_id: str | None


class _SourceEditMetadata(TypedDict):
    path: str
    before_hash: str | None
    after_hash: str
    summary: str
    touched_symbols: list[_SymbolMetadata]
    transformation_id: str | None
    start_line: int | None
    end_line: int | None


class _TransformedFileMetadata(TypedDict):
    path: str
    before_source: str
    after_source: str


class _PatchPayload(TypedDict):
    patch_text: str
    source_edits: list[_SourceEditMetadata]
    files: list[_TransformedFileMetadata]


class _CacheManifest(TypedDict):
    schema_version: str
    key: str
    candidate_id: str
    plan_ids: list[str]
    transformation_ids: list[str]
    requests: list[_RequestMetadata]
    patch_sha256: str
    patch: _PatchPayload


@dataclass(frozen=True, slots=True)
class _CacheLookup:
    """Validated lookup identity and manifest path for one cache entry.

    Attributes:
        manifest_path: Cache manifest path under the caller-owned cache root.
        key: Strict digest key derived from pre-generation identity.
        candidate_id: Stable candidate identity.
        plan_ids: Ordered plan IDs.
        transformation_ids: Ordered transformation IDs.
        requests: Sorted request path and source-hash fingerprints.
    """

    manifest_path: Path
    key: str
    candidate_id: str
    plan_ids: tuple[str, ...]
    transformation_ids: tuple[str, ...]
    requests: list[_RequestMetadata]


@dataclass(frozen=True, slots=True)
class _SourceEditScalarPayload:
    """Unvalidated scalar fields from a serialized source edit.

    Attributes:
        before_hash: Serialized pre-edit hash.
        after_hash: Serialized post-edit hash.
        summary: Serialized edit summary.
        touched_symbols: Serialized touched-symbol list.
        transformation_id: Serialized transformation ID.
        start_line: Serialized first changed line.
        end_line: Serialized final changed line.
    """

    before_hash: object
    after_hash: object
    summary: object
    touched_symbols: object
    transformation_id: object
    start_line: object
    end_line: object


@dataclass(frozen=True, slots=True)
class SourcePatchCacheResult:
    """Outcome of restoring or building a generated source patch.

    Attributes:
        patch: Generated source patch restored from the cache or built from the
            current project source.
        hit: Whether `patch` came from a valid existing cache entry.
        diagnostics: Human-readable cache diagnostics. Invalid cache entries are
            reported here after a successful rebuild instead of being treated as
            hits.
    """

    patch: GeneratedSourcePatch
    hit: bool
    diagnostics: tuple[str, ...] = ()


def restore_or_build_source_patch(
    cache_root: Path,
    candidate_id: str,
    *args: tuple[str, ...] | Path | tuple[SourceTransformationRequest, ...],
) -> SourcePatchCacheResult:
    """Restore a generated source patch from cache or build and store it.

    Args:
        cache_root: Caller-supplied persistent cache root. All cache writes stay
            under this path.
        candidate_id: Stable candidate identity for the generated patch.
        *args: Four positional values in this order: ordered source-optimization
            plan IDs, ordered transformation step IDs, project root, and
            transformation requests. The runtime call shape remains
            `restore_or_build_source_patch(cache_root, candidate_id, plan_ids,
            transformation_ids, project_root, requests)` while keeping this
            wrapper within the repository's argument-count limit.

    Returns:
        SourcePatchCacheResult: Frozen slots result containing the patch, cache
        hit flag, and diagnostics.

    Raises:
        ValueError: If cache identity strings or request fingerprints are unsafe
            or malformed, or if patch generation rejects the source/request
            combination.
        TypeError: If the positional argument payload does not match the public
            call shape.
        OSError: If source reads or atomic cache writes fail.
    """
    plan_ids, transformation_ids, project_root, requests = _restore_arguments(args)
    lookup = _cache_lookup(
        cache_root,
        candidate_id,
        plan_ids,
        transformation_ids,
        requests,
    )
    diagnostics: tuple[str, ...] = ()
    if lookup.manifest_path.exists():
        try:
            manifest = _validate_manifest(
                _read_manifest(lookup.manifest_path),
                lookup,
            )
            return SourcePatchCacheResult(
                patch=_patch_from_payload(manifest["patch"]),
                hit=True,
                diagnostics=("cache hit",),
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            diagnostics = (f"rebuilt invalid cache entry: {error}",)
    else:
        diagnostics = ("cache miss",)

    patch = build_source_transformation_patch(project_root, requests)
    manifest = _manifest(
        lookup,
        patch=patch,
    )
    _write_manifest_atomic(lookup.manifest_path, manifest)
    return SourcePatchCacheResult(patch=patch, hit=False, diagnostics=diagnostics)


def _restore_arguments(
    args: tuple[tuple[str, ...] | Path | tuple[SourceTransformationRequest, ...], ...],
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    Path,
    tuple[SourceTransformationRequest, ...],
]:
    if len(args) != _RESTORE_ARGUMENT_COUNT:
        raise TypeError("restore_or_build_source_patch expects four identity arguments")
    plan_ids = _string_tuple(args[0], "plan_ids")
    transformation_ids = _string_tuple(args[1], "transformation_ids")
    project_root = args[2]
    requests = args[3]
    if not isinstance(project_root, Path):
        raise TypeError("source patch cache project_root must be a Path")
    if not isinstance(requests, tuple) or not all(
        isinstance(request, SourceTransformationRequest) for request in requests
    ):
        raise TypeError("source patch cache requests must be SourceTransformationRequest tuple")
    return (
        plan_ids,
        transformation_ids,
        project_root,
        cast(tuple[SourceTransformationRequest, ...], requests),
    )


def _string_tuple(
    value: tuple[str, ...] | Path | tuple[SourceTransformationRequest, ...],
    label: str,
) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"source patch cache {label} must be a string tuple")
    return cast(tuple[str, ...], value)


def _cache_lookup(
    cache_root: Path,
    candidate_id: str,
    plan_ids: tuple[str, ...],
    transformation_ids: tuple[str, ...],
    requests: tuple[SourceTransformationRequest, ...],
) -> _CacheLookup:
    _validate_identity(candidate_id, plan_ids, transformation_ids)
    request_metadata = _request_metadata(requests)
    key = _cache_key(candidate_id, plan_ids, transformation_ids, request_metadata)
    return _CacheLookup(
        manifest_path=cache_root.expanduser() / key / _MANIFEST_FILE,
        key=key,
        candidate_id=candidate_id,
        plan_ids=plan_ids,
        transformation_ids=transformation_ids,
        requests=request_metadata,
    )


def _validate_identity(
    candidate_id: str,
    plan_ids: tuple[str, ...],
    transformation_ids: tuple[str, ...],
) -> None:
    _validate_cache_id(candidate_id, "candidate ID")
    if not plan_ids:
        raise ValueError("source patch cache plan IDs are empty")
    if not transformation_ids:
        raise ValueError("source patch cache transformation IDs are empty")
    for plan_id in plan_ids:
        _validate_cache_id(plan_id, "plan ID")
    for transformation_id in transformation_ids:
        _validate_cache_id(transformation_id, "transformation ID")


def _validate_cache_id(value: str, label: str) -> None:
    if not value:
        raise ValueError(f"source patch cache {label} is empty")
    if "\0" in value or "\\" in value:
        raise ValueError(f"unsafe source patch cache {label}: {value}")
    path = PurePosixPath(value)
    if path.is_absolute() or path.parts != (value,) or value in {".", ".."}:
        raise ValueError(f"unsafe source patch cache {label}: {value}")


def _request_metadata(
    requests: tuple[SourceTransformationRequest, ...],
) -> list[_RequestMetadata]:
    if not requests:
        raise ValueError("source patch cache requests are empty")
    metadata: list[_RequestMetadata] = []
    for request in sorted(requests, key=lambda item: item.path.as_posix()):
        path = _safe_posix_path(request.path)
        if not _is_hex_digest(request.expected_sha256):
            raise ValueError(f"source patch request digest is invalid: {path}")
        metadata.append(
            {
                "path": path,
                "expected_sha256": request.expected_sha256,
                "target": _symbol_payload(request.target),
                "declaration_kind": request.declaration_kind,
                "replacement_body": request.replacement_body,
                "helper_statements": list(request.helper_statements),
                "trailing_statements": list(request.trailing_statements),
                "additional_replacements": [
                    {
                        "target": _symbol_payload(replacement.target),
                        "declaration_kind": replacement.declaration_kind,
                        "replacement_body": replacement.replacement_body,
                    }
                    for replacement in request.additional_replacements
                ],
                "summary": request.summary,
                "transformation_id": request.transformation_id,
            }
        )
    return metadata


def _safe_posix_path(path: PurePosixPath | str) -> str:
    posix_path = PurePosixPath(str(path))
    if (
        not posix_path.parts
        or posix_path.is_absolute()
        or ".." in posix_path.parts
        or any(part in {"", "."} for part in posix_path.parts)
    ):
        raise ValueError(f"unsafe source patch cache path: {path}")
    return posix_path.as_posix()


def _cache_key(
    candidate_id: str,
    plan_ids: tuple[str, ...],
    transformation_ids: tuple[str, ...],
    requests: list[_RequestMetadata],
) -> str:
    fingerprint = {
        "schema_version": _SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "plan_ids": list(plan_ids),
        "transformation_ids": list(transformation_ids),
        "requests": requests,
    }
    return _sha256_bytes(_canonical_json(fingerprint))


def _manifest(lookup: _CacheLookup, patch: GeneratedSourcePatch) -> _CacheManifest:
    payload = _patch_payload(patch)
    return {
        "schema_version": _SCHEMA_VERSION,
        "key": lookup.key,
        "candidate_id": lookup.candidate_id,
        "plan_ids": list(lookup.plan_ids),
        "transformation_ids": list(lookup.transformation_ids),
        "requests": lookup.requests,
        "patch_sha256": _sha256_bytes(_canonical_json(payload)),
        "patch": payload,
    }


def _patch_payload(patch: GeneratedSourcePatch) -> _PatchPayload:
    return {
        "patch_text": patch.patch_text,
        "source_edits": [_source_edit_payload(edit) for edit in patch.source_edits],
        "files": [_transformed_file_payload(file) for file in patch.files],
    }


def _source_edit_payload(edit: SourceEdit) -> _SourceEditMetadata:
    return {
        "path": _safe_posix_path(edit.path),
        "before_hash": edit.before_hash,
        "after_hash": edit.after_hash,
        "summary": edit.summary,
        "touched_symbols": [_symbol_payload(symbol) for symbol in edit.touched_symbols],
        "transformation_id": edit.transformation_id,
        "start_line": edit.start_line,
        "end_line": edit.end_line,
    }


def _symbol_payload(symbol: SymbolId) -> _SymbolMetadata:
    return {"module": symbol.module, "qualname": symbol.qualname}


def _transformed_file_payload(file: TransformedSourceFile) -> _TransformedFileMetadata:
    return {
        "path": _safe_posix_path(file.path),
        "before_source": file.before_source,
        "after_source": file.after_source,
    }


def _read_manifest(path: Path) -> dict[str, object]:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("source patch cache manifest is not an object")
    return cast(dict[str, object], raw)


def _validate_manifest(
    manifest: dict[str, object],
    lookup: _CacheLookup,
) -> _CacheManifest:
    expected = {
        "schema_version": _SCHEMA_VERSION,
        "key": lookup.key,
        "candidate_id": lookup.candidate_id,
        "plan_ids": list(lookup.plan_ids),
        "transformation_ids": list(lookup.transformation_ids),
        "requests": lookup.requests,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"source patch cache manifest {field} mismatch")
    patch_sha256 = manifest.get("patch_sha256")
    if not isinstance(patch_sha256, str) or not _is_hex_digest(patch_sha256):
        raise TypeError("source patch cache manifest patch_sha256 is invalid")
    patch = manifest.get("patch")
    if not isinstance(patch, dict):
        raise TypeError("source patch cache manifest patch is invalid")
    payload = _validate_patch_payload(cast(dict[str, object], patch))
    if _sha256_bytes(_canonical_json(payload)) != patch_sha256:
        raise ValueError("source patch cache payload digest mismatch")
    return cast(_CacheManifest, {**manifest, "patch": payload})


def _validate_patch_payload(payload: dict[str, object]) -> _PatchPayload:
    patch_text = payload.get("patch_text")
    if not isinstance(patch_text, str):
        raise TypeError("source patch cache patch_text is invalid")
    source_edits = _required_list(payload, "source_edits")
    files = _required_list(payload, "files")
    return {
        "patch_text": patch_text,
        "source_edits": [_validate_source_edit_payload(item) for item in source_edits],
        "files": [_validate_transformed_file_payload(item) for item in files],
    }


def _required_list(payload: dict[str, object], field: str) -> list[object]:
    value = payload.get(field)
    if not isinstance(value, list):
        raise TypeError(f"source patch cache {field} is invalid")
    return cast(list[object], value)


def _validate_source_edit_payload(item: object) -> _SourceEditMetadata:
    if not isinstance(item, dict):
        raise TypeError("source patch cache source edit is invalid")
    metadata = cast(dict[object, object], item)
    if set(metadata) != {
        "path",
        "before_hash",
        "after_hash",
        "summary",
        "touched_symbols",
        "transformation_id",
        "start_line",
        "end_line",
    }:
        raise ValueError("source patch cache source edit fields are invalid")
    path = metadata["path"]
    before_hash = metadata["before_hash"]
    after_hash = metadata["after_hash"]
    summary = metadata["summary"]
    touched_symbols = metadata["touched_symbols"]
    transformation_id = metadata["transformation_id"]
    start_line = metadata["start_line"]
    end_line = metadata["end_line"]
    if not isinstance(path, str):
        raise TypeError("source patch cache source edit path is invalid")
    _validate_source_edit_scalar_fields(
        _SourceEditScalarPayload(
            before_hash=before_hash,
            after_hash=after_hash,
            summary=summary,
            touched_symbols=touched_symbols,
            transformation_id=transformation_id,
            start_line=start_line,
            end_line=end_line,
        )
    )
    return {
        "path": _safe_posix_path(path),
        "before_hash": cast(str | None, before_hash),
        "after_hash": cast(str, after_hash),
        "summary": cast(str, summary),
        "touched_symbols": [
            _validate_symbol_payload(symbol) for symbol in cast(list[object], touched_symbols)
        ],
        "transformation_id": cast(str | None, transformation_id),
        "start_line": cast(int | None, start_line),
        "end_line": cast(int | None, end_line),
    }


def _validate_source_edit_scalar_fields(payload: _SourceEditScalarPayload) -> None:
    if payload.before_hash is not None and (
        not isinstance(payload.before_hash, str) or not _is_hex_digest(payload.before_hash)
    ):
        raise ValueError("source patch cache source edit before_hash is invalid")
    if not isinstance(payload.after_hash, str) or not _is_hex_digest(payload.after_hash):
        raise ValueError("source patch cache source edit after_hash is invalid")
    if not isinstance(payload.summary, str):
        raise TypeError("source patch cache source edit summary is invalid")
    if not isinstance(payload.touched_symbols, list):
        raise TypeError("source patch cache source edit symbols are invalid")
    if payload.transformation_id is not None and not isinstance(
        payload.transformation_id,
        str,
    ):
        raise TypeError("source patch cache source edit transformation_id is invalid")
    if payload.start_line is not None and not isinstance(payload.start_line, int):
        raise TypeError("source patch cache source edit start_line is invalid")
    if payload.end_line is not None and not isinstance(payload.end_line, int):
        raise TypeError("source patch cache source edit end_line is invalid")


def _validate_symbol_payload(item: object) -> _SymbolMetadata:
    if not isinstance(item, dict):
        raise TypeError("source patch cache symbol is invalid")
    metadata = cast(dict[object, object], item)
    if set(metadata) != {"module", "qualname"}:
        raise ValueError("source patch cache symbol fields are invalid")
    module = metadata["module"]
    qualname = metadata["qualname"]
    if not isinstance(module, str) or not isinstance(qualname, str):
        raise TypeError("source patch cache symbol is invalid")
    return {"module": module, "qualname": qualname}


def _validate_transformed_file_payload(item: object) -> _TransformedFileMetadata:
    if not isinstance(item, dict):
        raise TypeError("source patch cache transformed file is invalid")
    metadata = cast(dict[object, object], item)
    if set(metadata) != {"path", "before_source", "after_source"}:
        raise ValueError("source patch cache transformed file fields are invalid")
    path = metadata["path"]
    before_source = metadata["before_source"]
    after_source = metadata["after_source"]
    if (
        not isinstance(path, str)
        or not isinstance(before_source, str)
        or not isinstance(after_source, str)
    ):
        raise TypeError("source patch cache transformed file is invalid")
    return {
        "path": _safe_posix_path(path),
        "before_source": before_source,
        "after_source": after_source,
    }


def _patch_from_payload(payload: _PatchPayload) -> GeneratedSourcePatch:
    return GeneratedSourcePatch(
        patch_text=payload["patch_text"],
        source_edits=tuple(_source_edit_from_payload(edit) for edit in payload["source_edits"]),
        files=tuple(_transformed_file_from_payload(file) for file in payload["files"]),
    )


def _source_edit_from_payload(payload: _SourceEditMetadata) -> SourceEdit:
    return SourceEdit(
        path=PurePosixPath(payload["path"]),
        before_hash=payload["before_hash"],
        after_hash=payload["after_hash"],
        summary=payload["summary"],
        touched_symbols=tuple(
            SymbolId(module=symbol["module"], qualname=symbol["qualname"])
            for symbol in payload["touched_symbols"]
        ),
        transformation_id=payload["transformation_id"],
        start_line=payload["start_line"],
        end_line=payload["end_line"],
    )


def _transformed_file_from_payload(
    payload: _TransformedFileMetadata,
) -> TransformedSourceFile:
    return TransformedSourceFile(
        path=PurePosixPath(payload["path"]),
        before_source=payload["before_source"],
        after_source=payload["after_source"],
    )


def _write_manifest_atomic(path: Path, manifest: _CacheManifest) -> None:
    data = json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8")
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


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_hex_digest(value: str) -> bool:
    return len(value) == _SHA256_HEX_LENGTH and all(
        character in "0123456789abcdef" for character in value
    )
