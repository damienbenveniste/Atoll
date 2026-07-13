"""Strict parsing and deterministic matrix projection for corpus manifests.

Unknown fields are rejected instead of ignored because every manifest field
affects reproducibility, external-code execution, or historical comparison.
The parser performs no network access and does not inspect a checkout.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path, PurePosixPath
from typing import cast
from urllib.parse import urlparse

from scripts.benchmark_corpus.models import (
    CorpusBackend,
    CorpusCase,
    CorpusDefaults,
    CorpusManifest,
    CorpusPlatform,
    CorpusTier,
    MatrixEntry,
    WorkloadProvenance,
    WorkloadSource,
)

MANIFEST_SCHEMA_VERSION = 1
INITIAL_PYTHON_VERSION = "3.12"
REQUIRED_BACKENDS: tuple[CorpusBackend, ...] = ("mypyc", "cython")
DEFAULT_TEST_TIMEOUT_SECONDS = 300
DEFAULT_COMPILE_TIMEOUT_SECONDS = 45 * 60
DEFAULT_PERFORMANCE_TIMEOUT_SECONDS = 90 * 60
DEFAULT_MAX_LOG_BYTES = 10 * 1024 * 1024

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "python_version",
        "backends",
        "test_timeout_seconds",
        "compile_timeout_seconds",
        "performance_timeout_seconds",
        "max_log_bytes",
        "case",
    }
)
_CASE_FIELDS = frozenset(
    {
        "id",
        "name",
        "repository",
        "revision",
        "project_subroot",
        "dependency_lock",
        "focused_test_command",
        "oracle_adapter",
        "tiers",
        "platforms",
        "workload",
        "test_timeout_seconds",
        "compile_timeout_seconds",
        "performance_timeout_seconds",
    }
)
_WORKLOAD_FIELDS = frozenset({"source", "repository", "revision", "path", "sha256", "notice"})
_TIERS: tuple[CorpusTier, ...] = (
    "compatibility",
    "performance",
    "calibration",
    "negative-control",
)
_PLATFORMS: tuple[CorpusPlatform, ...] = ("ubuntu-24.04", "macos-14")
_WORKLOAD_SOURCES: tuple[WorkloadSource, ...] = (
    "upstream",
    "pyperformance",
    "mypyc-benchmarks",
    "atoll",
)
_ID_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_ADAPTER_PATTERN = re.compile(r"[a-z0-9_]+(?:\.[a-z0-9_]+)*\Z")
_SHA_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class ManifestError(ValueError):
    """Raised when corpus metadata is incomplete, unsafe, or ambiguous."""


def load_manifest(path: Path) -> CorpusManifest:
    """Parse and validate a schema-v1 corpus manifest without side effects.

    Args:
        path: TOML manifest to parse.

    Returns:
        CorpusManifest: Immutable normalized configuration.

    Raises:
        ManifestError: If TOML syntax, types, values, paths, or identities are invalid.
    """
    try:
        payload = cast(dict[str, object], tomllib.loads(path.read_text(encoding="utf-8")))
        return _parse_manifest(path.resolve(), payload)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ManifestError(f"cannot read corpus manifest {path}: {error}") from error


def manifest_matrix(
    manifest: CorpusManifest,
    *,
    tier: CorpusTier | None = None,
    platform: CorpusPlatform | None = None,
    case_ids: tuple[str, ...] = (),
) -> tuple[MatrixEntry, ...]:
    """Project selected cases into stable case, tier, and platform rows.

    Args:
        manifest: Validated corpus configuration.
        tier: Optional tier filter.
        platform: Optional platform filter.
        case_ids: Optional exact case allowlist.

    Returns:
        tuple[MatrixEntry, ...]: Rows sorted by case, tier, and platform.

    Raises:
        ManifestError: If an allowlisted case does not exist.
    """
    selected = frozenset(case_ids)
    known = frozenset(case.id for case in manifest.cases)
    unknown = sorted(selected - known)
    if unknown:
        raise ManifestError(f"unknown corpus case(s): {', '.join(unknown)}")
    rows = (
        MatrixEntry(case_id=case.id, tier=case_tier, platform=case_platform)
        for case in manifest.cases
        if not selected or case.id in selected
        for case_tier in case.tiers
        if tier is None or case_tier == tier
        for case_platform in case.platforms
        if platform is None or case_platform == platform
    )
    return tuple(sorted(rows, key=lambda row: (row.case_id, row.tier, row.platform)))


def _parse_manifest(path: Path, payload: dict[str, object]) -> CorpusManifest:
    _reject_unknown(payload, _TOP_LEVEL_FIELDS, "manifest")
    version = _required_integer(payload, "schema_version", "manifest")
    if version != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            f"manifest.schema_version must be {MANIFEST_SCHEMA_VERSION}, got {version}"
        )
    python_version = _required_string(payload, "python_version", "manifest")
    if python_version != INITIAL_PYTHON_VERSION:
        raise ManifestError(
            f"manifest.python_version must be {INITIAL_PYTHON_VERSION}, got {python_version}"
        )
    backends = _literal_sequence(
        payload,
        "backends",
        "manifest",
        REQUIRED_BACKENDS,
    )
    if set(backends) != set(REQUIRED_BACKENDS) or len(backends) != len(REQUIRED_BACKENDS):
        raise ManifestError("manifest.backends must contain mypyc and cython exactly once")
    defaults = CorpusDefaults(
        test_timeout_seconds=_positive_integer(
            payload.get("test_timeout_seconds", DEFAULT_TEST_TIMEOUT_SECONDS),
            "manifest.test_timeout_seconds",
        ),
        compile_timeout_seconds=_positive_integer(
            payload.get("compile_timeout_seconds", DEFAULT_COMPILE_TIMEOUT_SECONDS),
            "manifest.compile_timeout_seconds",
        ),
        performance_timeout_seconds=_positive_integer(
            payload.get("performance_timeout_seconds", DEFAULT_PERFORMANCE_TIMEOUT_SECONDS),
            "manifest.performance_timeout_seconds",
        ),
        max_log_bytes=_positive_integer(
            payload.get("max_log_bytes", DEFAULT_MAX_LOG_BYTES),
            "manifest.max_log_bytes",
        ),
    )
    raw_cases = _optional_table_sequence(payload.get("case"), "manifest.case")
    cases = tuple(_parse_case(index, raw) for index, raw in enumerate(raw_cases))
    ids = tuple(case.id for case in cases)
    duplicate_ids = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicate_ids:
        raise ManifestError(f"duplicate corpus case id(s): {', '.join(duplicate_ids)}")
    return CorpusManifest(
        path=path,
        schema_version=version,
        python_version=python_version,
        backends=cast(tuple[CorpusBackend, ...], backends),
        defaults=defaults,
        cases=tuple(sorted(cases, key=lambda case: case.id)),
    )


def _parse_case(index: int, payload: dict[str, object]) -> CorpusCase:
    label = f"manifest.case[{index}]"
    _reject_unknown(payload, _CASE_FIELDS, label)
    case_id = _required_string(payload, "id", label)
    if _ID_PATTERN.fullmatch(case_id) is None:
        raise ManifestError(f"{label}.id must be a lowercase slug")
    repository = _repository_url(_required_string(payload, "repository", label), label)
    revision = _full_sha(_required_string(payload, "revision", label), f"{label}.revision")
    project_subroot = _safe_path(
        _required_string(payload, "project_subroot", label), f"{label}.project_subroot"
    )
    dependency_lock = _safe_path(
        _required_string(payload, "dependency_lock", label), f"{label}.dependency_lock"
    )
    command = _command(payload.get("focused_test_command"), f"{label}.focused_test_command")
    adapter = _required_string(payload, "oracle_adapter", label)
    if _ADAPTER_PATTERN.fullmatch(adapter) is None:
        raise ManifestError(f"{label}.oracle_adapter must be a dotted lowercase identifier")
    tiers = cast(
        tuple[CorpusTier, ...],
        _literal_sequence(payload, "tiers", label, _TIERS),
    )
    platforms = cast(
        tuple[CorpusPlatform, ...],
        _literal_sequence(payload, "platforms", label, _PLATFORMS),
    )
    workload_payload = payload.get("workload")
    workload = (
        None
        if workload_payload is None
        else _parse_workload(_mapping(workload_payload, f"{label}.workload"), label)
    )
    if "performance" in tiers and workload is None:
        raise ManifestError(f"{label} is a performance case but has no workload provenance")
    return CorpusCase(
        id=case_id,
        name=_required_string(payload, "name", label),
        repository=repository,
        revision=revision,
        project_subroot=project_subroot,
        dependency_lock=dependency_lock,
        focused_test_command=command,
        oracle_adapter=adapter,
        tiers=tiers,
        platforms=platforms,
        workload=workload,
        test_timeout_seconds=_optional_positive_integer(payload, "test_timeout_seconds", label),
        compile_timeout_seconds=_optional_positive_integer(
            payload, "compile_timeout_seconds", label
        ),
        performance_timeout_seconds=_optional_positive_integer(
            payload, "performance_timeout_seconds", label
        ),
    )


def _parse_workload(payload: dict[str, object], case_label: str) -> WorkloadProvenance:
    label = f"{case_label}.workload"
    _reject_unknown(payload, _WORKLOAD_FIELDS, label)
    source = _required_string(payload, "source", label)
    if source not in _WORKLOAD_SOURCES:
        raise ManifestError(f"{label}.source has unsupported value {source!r}")
    digest = _required_string(payload, "sha256", label)
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise ManifestError(f"{label}.sha256 must be 64 lowercase hexadecimal characters")
    return WorkloadProvenance(
        source=source,
        repository=_repository_url(_required_string(payload, "repository", label), label),
        revision=_full_sha(_required_string(payload, "revision", label), f"{label}.revision"),
        path=_safe_path(_required_string(payload, "path", label), f"{label}.path"),
        sha256=digest,
        notice=_safe_path(_required_string(payload, "notice", label), f"{label}.notice"),
    )


def _reject_unknown(payload: dict[str, object], allowed: frozenset[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ManifestError(f"{label} contains unknown field(s): {', '.join(unknown)}")


def _required_string(payload: dict[str, object], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{label}.{key} must be a non-empty string")
    return value


def _required_integer(payload: dict[str, object], key: str, label: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ManifestError(f"{label}.{key} must be an integer")
    return value


def _positive_integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ManifestError(f"{label} must be a positive integer")
    return value


def _optional_positive_integer(payload: dict[str, object], key: str, label: str) -> int | None:
    value = payload.get(key)
    return None if value is None else _positive_integer(value, f"{label}.{key}")


def _literal_sequence(
    payload: dict[str, object],
    key: str,
    label: str,
    allowed: tuple[str, ...],
) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise ManifestError(f"{label}.{key} must be a non-empty array")
    items = tuple(cast(list[object], value))
    if any(not isinstance(item, str) or item not in allowed for item in items):
        raise ManifestError(f"{label}.{key} values must be selected from {', '.join(allowed)}")
    strings = tuple(cast(str, item) for item in items)
    if len(strings) != len(set(strings)):
        raise ManifestError(f"{label}.{key} must not contain duplicates")
    return strings


def _command(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ManifestError(f"{label} must be a non-empty argv array")
    items = tuple(cast(list[object], value))
    if any(not isinstance(item, str) or not item or "\x00" in item for item in items):
        raise ManifestError(f"{label} must contain non-empty strings without NUL bytes")
    return tuple(cast(str, item) for item in items)


def _optional_table_sequence(value: object, label: str) -> tuple[dict[str, object], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ManifestError(f"{label} must be an array of tables")
    return tuple(
        _mapping(item, f"{label}[{index}]") for index, item in enumerate(cast(list[object], value))
    )


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ManifestError(f"{label} must be a table")
    raw = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in raw):
        raise ManifestError(f"{label} contains a non-string key")
    return {cast(str, key): item for key, item in raw.items()}


def _full_sha(value: str, label: str) -> str:
    if _SHA_PATTERN.fullmatch(value) is None:
        raise ManifestError(f"{label} must be 40 lowercase hexadecimal characters")
    return value


def _repository_url(value: str, label: str) -> str:
    parsed = urlparse(value)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.params
        or parsed.query
        or parsed.fragment
        or not parsed.path.endswith(".git")
    ):
        raise ManifestError(f"{label}.repository must be a canonical HTTPS .git URL")
    return value


def _safe_path(value: str, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        if value == ".":
            return path
        raise ManifestError(f"{label} must be a normalized repository-relative path")
    return path
