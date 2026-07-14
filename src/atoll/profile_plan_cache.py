"""Strict persistent selection cache for profile candidate plans.

This module owns only replay of a previously accepted ordered symbol selection.
It does not rank candidates, run benchmarks, import cached content, or determine
whether a symbol is otherwise safe to lower. Callers provide the complete
identity and the currently valid selection and availability boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict, cast

from atoll.models import SymbolId

ProfilePlanCacheStatus = Literal["disabled", "hit", "miss"]

_ENTRY_KEYS = {"digest", "identity", "selection"}
_IDENTITY_KEYS = {
    "backend_order",
    "benchmark_argv",
    "cache_format_version",
    "candidate_identity",
    "lowering_version",
    "module_scope",
    "module_source_hashes",
    "python_cache_tag",
    "python_platform",
    "scope_identity",
}
_SYMBOL_KEYS = {"module", "qualname"}


class _SymbolPayload(TypedDict):
    module: str
    qualname: str


class _SourceHashPayload(TypedDict):
    module: str
    source_hash: str


class _IdentityPayload(TypedDict):
    scope_identity: str
    candidate_identity: list[str]
    module_source_hashes: list[_SourceHashPayload]
    benchmark_argv: list[str]
    backend_order: list[str]
    module_scope: str | None
    python_cache_tag: str
    python_platform: str
    cache_format_version: str
    lowering_version: str


class _UnsignedEntryPayload(TypedDict):
    identity: _IdentityPayload
    selection: list[_SymbolPayload]


class _EntryPayload(_UnsignedEntryPayload):
    digest: str


@dataclass(frozen=True, slots=True)
class ProfilePlanIdentity:
    """Complete identity boundary for one replayable profile plan selection.

    The identity deliberately includes environmental and lowering inputs that
    can change whether a prior candidate plan remains valid. Source hashes must
    be ordered by module name so equivalent caller inputs serialize identically.

    Attributes:
        scope_identity: Stable identity for the compilation or profiling scope.
        candidate_identity: Ordered stable identities of the candidate plans.
        module_source_hashes: Module names and content hashes sorted by module name.
        benchmark_argv: Exact benchmark argv, with no shell interpretation.
        backend_order: Configured backend preference order.
        module_scope: Optional configured module restriction.
        python_cache_tag: Python implementation cache tag, such as `cpython-312`.
        python_platform: Stable interpreter platform string.
        cache_format_version: Version of the on-disk cache schema.
        lowering_version: Version of the candidate lowering contract.
    """

    scope_identity: str
    candidate_identity: tuple[str, ...]
    module_source_hashes: tuple[tuple[str, str], ...]
    benchmark_argv: tuple[str, ...]
    backend_order: tuple[str, ...]
    module_scope: str | None
    python_cache_tag: str
    python_platform: str
    cache_format_version: str
    lowering_version: str

    def __post_init__(self) -> None:
        """Reject ambiguous identities before they can become cache keys.

        Raises:
            ValueError: If a required identity component is empty, an ordered
                identity contains duplicates, or source hashes are not sorted.
        """
        scalar_fields = (
            ("scope_identity", self.scope_identity),
            ("python_cache_tag", self.python_cache_tag),
            ("python_platform", self.python_platform),
            ("cache_format_version", self.cache_format_version),
            ("lowering_version", self.lowering_version),
        )
        for field, value in scalar_fields:
            if not value:
                raise ValueError(f"profile plan identity {field} is empty")
        if self.module_scope == "":
            raise ValueError("profile plan identity module_scope is empty")
        _validate_non_empty_unique_strings("candidate_identity", self.candidate_identity)
        if not self.benchmark_argv or any(not value for value in self.benchmark_argv):
            raise ValueError("profile plan identity benchmark_argv is empty")
        _validate_non_empty_unique_strings("backend_order", self.backend_order)
        if not self.module_source_hashes:
            raise ValueError("profile plan identity module_source_hashes is empty")
        if any(not module or not source_hash for module, source_hash in self.module_source_hashes):
            raise ValueError("profile plan identity module source hash is empty")
        modules = tuple(module for module, _ in self.module_source_hashes)
        if modules != tuple(sorted(modules)) or len(modules) != len(set(modules)):
            raise ValueError("profile plan identity module_source_hashes must be sorted and unique")


@dataclass(frozen=True, slots=True)
class ProfilePlanDecision:
    """Deterministic evidence for a profile candidate-plan cache decision.

    Attributes:
        status: `disabled`, `miss`, or strict `hit` classification.
        selection: Ordered current or replayed symbol selection.
        diagnostic: Stable machine-readable explanation of the decision.
        cache_path: Exact entry path considered without requiring it to exist.
        identity_digest: SHA-256 digest naming the complete strict identity.
    """

    status: ProfilePlanCacheStatus
    selection: tuple[SymbolId, ...]
    diagnostic: str
    cache_path: Path
    identity_digest: str


def select_profile_plan(
    cache_root: Path,
    identity: ProfilePlanIdentity,
    current: tuple[SymbolId, ...],
    available: frozenset[SymbolId],
) -> ProfilePlanDecision:
    """Return a current selection or replay a strictly matching cached selection.

    A non-empty, unique, available current selection is the authoritative value
    written on a miss or after any invalid entry. Existing entries are data only:
    JSON parsing and explicit shape checks are the sole interpretation performed.
    An empty current selection may replay a valid existing entry, but it never
    creates or replaces cache state. This lets an unchanged warm compile survive
    statistical sampling jitter without allowing an empty profile to establish a
    new native-compilation promise.

    Args:
        cache_root: Caller-owned root beneath which the content-addressed entry lives.
        identity: Complete strict identity for the candidate-plan decision.
        current: Ordered selection produced by the current profiling run.
        available: Symbols the caller can currently lower or otherwise select.

    Returns:
        ProfilePlanDecision: Stable selection plus status, diagnostic, digest,
        and path evidence.

    Raises:
        ValueError: If a non-empty current selection contains duplicates or a
            symbol outside `available`.
        OSError: If an entry needs to be written but cannot be replaced atomically.
    """
    identity_payload = _identity_payload(identity)
    identity_digest = _sha256(_canonical_bytes(identity_payload))
    cache_path = cache_root.expanduser() / "profile-plans" / f"{identity_digest}.json"
    if current:
        _validate_current(current, available)
    if not cache_path.exists() and not cache_path.is_symlink():
        if not current:
            return ProfilePlanDecision(
                status="disabled",
                selection=(),
                diagnostic="empty-current-selection:no-entry",
                cache_path=cache_path,
                identity_digest=identity_digest,
            )
        _write_entry_atomic(cache_path, identity_payload, current)
        return _decision("miss", current, "absent", cache_path, identity_digest)
    invalid_reason: str | None = None
    stored: tuple[SymbolId, ...] = ()
    if cache_path.is_symlink():
        invalid_reason = "symlink"
    else:
        try:
            stored = _read_selection(cache_path, identity_payload, identity)
            invalid_reason = _stored_selection_invalid_reason(stored, available)
        except json.JSONDecodeError:
            invalid_reason = "malformed-json"
            stored = ()
        except OSError:
            invalid_reason = "io-error"
            stored = ()
        except (TypeError, ValueError) as error:
            invalid_reason = str(error)
            stored = ()
    if invalid_reason is None:
        return _decision("hit", stored, "strict-hit", cache_path, identity_digest)
    if not current:
        return _decision(
            "disabled",
            (),
            f"empty-current-selection:{invalid_reason}",
            cache_path,
            identity_digest,
        )
    _write_entry_atomic(cache_path, identity_payload, current)
    return _decision(
        "miss",
        current,
        f"replaced:{invalid_reason}",
        cache_path,
        identity_digest,
    )


def _validate_non_empty_unique_strings(field: str, values: tuple[str, ...]) -> None:
    if not values or any(not value for value in values):
        raise ValueError(f"profile plan identity {field} is empty")
    if len(values) != len(set(values)):
        raise ValueError(f"profile plan identity {field} contains duplicates")


def _validate_current(current: tuple[SymbolId, ...], available: frozenset[SymbolId]) -> None:
    if len(current) != len(set(current)):
        raise ValueError("current profile plan selection contains duplicates")
    if any(symbol not in available for symbol in current):
        raise ValueError("current profile plan selection contains an unavailable symbol")


def _decision(
    status: ProfilePlanCacheStatus,
    selection: tuple[SymbolId, ...],
    diagnostic: str,
    cache_path: Path,
    identity_digest: str,
) -> ProfilePlanDecision:
    return ProfilePlanDecision(status, selection, diagnostic, cache_path, identity_digest)


def _identity_payload(identity: ProfilePlanIdentity) -> _IdentityPayload:
    return {
        "scope_identity": identity.scope_identity,
        "candidate_identity": list(identity.candidate_identity),
        "module_source_hashes": [
            {"module": module, "source_hash": source_hash}
            for module, source_hash in identity.module_source_hashes
        ],
        "benchmark_argv": list(identity.benchmark_argv),
        "backend_order": list(identity.backend_order),
        "module_scope": identity.module_scope,
        "python_cache_tag": identity.python_cache_tag,
        "python_platform": identity.python_platform,
        "cache_format_version": identity.cache_format_version,
        "lowering_version": identity.lowering_version,
    }


def _symbol_payload(symbol: SymbolId) -> _SymbolPayload:
    return {"module": symbol.module, "qualname": symbol.qualname}


def _write_entry_atomic(
    path: Path,
    identity: _IdentityPayload,
    selection: tuple[SymbolId, ...],
) -> None:
    unsigned: _UnsignedEntryPayload = {
        "identity": identity,
        "selection": [_symbol_payload(symbol) for symbol in selection],
    }
    entry: _EntryPayload = {**unsigned, "digest": _sha256(_canonical_bytes(unsigned))}
    _write_atomic(path, _canonical_bytes(entry) + b"\n")


def _read_selection(
    path: Path,
    expected_identity: _IdentityPayload,
    identity: ProfilePlanIdentity,
) -> tuple[SymbolId, ...]:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("malformed-entry")
    entry = cast(dict[str, object], raw)
    if set(entry) != _ENTRY_KEYS:
        raise TypeError("malformed-entry")
    raw_identity = entry["identity"]
    raw_selection = entry["selection"]
    digest = entry["digest"]
    if not isinstance(raw_identity, dict):
        raise TypeError("malformed-entry")
    identity_map = cast(dict[str, object], raw_identity)
    if set(identity_map) != _IDENTITY_KEYS:
        raise TypeError("malformed-entry")
    if not isinstance(raw_selection, list) or not isinstance(digest, str):
        raise TypeError("malformed-entry")
    selection_payload = cast(list[object], raw_selection)
    unsigned: dict[str, object] = {
        "identity": identity_map,
        "selection": selection_payload,
    }
    if digest != _sha256(_canonical_bytes(unsigned)):
        raise ValueError("digest-mismatch")
    if identity_map.get("cache_format_version") != identity.cache_format_version:
        raise ValueError("version-mismatch")
    if identity_map.get("lowering_version") != identity.lowering_version:
        raise ValueError("version-mismatch")
    if raw_identity != expected_identity:
        raise ValueError("identity-mismatch")
    return tuple(_parse_symbol(item) for item in selection_payload)


def _parse_symbol(raw: object) -> SymbolId:
    if not isinstance(raw, dict):
        raise TypeError("malformed-entry")
    symbol = cast(dict[str, object], raw)
    if set(symbol) != _SYMBOL_KEYS:
        raise TypeError("malformed-entry")
    module = symbol["module"]
    qualname = symbol["qualname"]
    if not isinstance(module, str) or not module or not isinstance(qualname, str) or not qualname:
        raise TypeError("malformed-entry")
    return SymbolId(module=module, qualname=qualname)


def _stored_selection_invalid_reason(
    selection: tuple[SymbolId, ...], available: frozenset[SymbolId]
) -> str | None:
    if not selection:
        return "empty-selection"
    if len(selection) != len(set(selection)):
        return "duplicate-selection"
    if any(symbol not in available for symbol in selection):
        return "unavailable-symbol"
    return None


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.is_symlink():
        raise ValueError(f"refusing to replace symlink: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
