"""Persist accepted source-search winner identity without performance evidence.

The cache records only which candidate most recently passed every promotion
gate for an exact, caller-computed search identity. Benchmark measurements and
semantic results are deliberately excluded so every invocation must reproduce
all acceptance evidence. Missing, stale, and corrupt entries are ordinary cache
misses and never prevent a full source search.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

_SCHEMA_VERSION = "1"
_WINNER_DIRECTORY = "accepted-winners"


@dataclass(frozen=True, slots=True)
class SourceWinnerIdentity:
    """Static compatibility boundary for replaying one accepted candidate.

    Attributes:
        plan_sources: Sorted source-plan IDs and their sorted source path/hash pairs.
        candidate_ids: Complete deterministic formed-candidate universe.
        test_command: Exact semantic-test argv.
        benchmark_command: Exact benchmark argv.
        quality_project_digest: Content digest for tests, benchmarks, and project metadata.
        baseline_payload_digest: Content digest for the unpacked baseline wheel.
        environment_digest: Digest of installed distributions and process environment.
        configuration: Relevant compile and source-search policy values.
        python_abi: Interpreter ABI compatibility tag.
        platform: Interpreter platform compatibility tag.
        versions: Search, lowering, and cache contract versions.
    """

    plan_sources: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
    candidate_ids: tuple[str, ...]
    test_command: tuple[str, ...]
    benchmark_command: tuple[str, ...]
    quality_project_digest: str
    baseline_payload_digest: str
    environment_digest: str
    configuration: tuple[tuple[str, str], ...]
    python_abi: str
    platform: str
    versions: tuple[tuple[str, str], ...]

    @property
    def key(self) -> str:
        """Return a deterministic digest for every replay compatibility input.

        Returns:
            str: SHA-256 identity used only as an Atoll-owned cache filename.
        """
        payload = {
            "benchmark_command": self.benchmark_command,
            "candidate_ids": self.candidate_ids,
            "configuration": self.configuration,
            "baseline_payload_digest": self.baseline_payload_digest,
            "environment_digest": self.environment_digest,
            "plan_sources": self.plan_sources,
            "platform": self.platform,
            "python_abi": self.python_abi,
            "quality_project_digest": self.quality_project_digest,
            "test_command": self.test_command,
            "versions": self.versions,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SourceWinnerLookup:
    """Safe accepted-winner cache lookup result.

    Attributes:
        candidate_id: Accepted candidate ID on a valid hit, otherwise ``None``.
        diagnostic: Stable explanation suitable for progress output.
    """

    candidate_id: str | None
    diagnostic: str


class _WinnerManifest(TypedDict):
    schema_version: str
    identity_key: str
    candidate_id: str


def load_source_winner(cache_root: Path, identity: SourceWinnerIdentity) -> SourceWinnerLookup:
    """Load a winner for an exact identity, treating invalid data as a miss.

    Args:
        cache_root: Caller-owned source-optimization cache directory.
        identity: Complete static replay compatibility identity.

    Returns:
        SourceWinnerLookup: Candidate ID only when the manifest is exact and valid.
    """
    path = _winner_path(cache_root, identity)
    if path.is_symlink():
        return SourceWinnerLookup(
            candidate_id=None,
            diagnostic="ignored invalid accepted winner cache: manifest is a symlink",
        )
    if not path.is_file():
        return SourceWinnerLookup(candidate_id=None, diagnostic="accepted winner cache miss")
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
        candidate_id = _validated_candidate_id(payload, identity.key)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return SourceWinnerLookup(
            candidate_id=None,
            diagnostic=f"ignored invalid accepted winner cache: {error}",
        )
    return SourceWinnerLookup(
        candidate_id=candidate_id,
        diagnostic=f"accepted winner cache hit: {candidate_id}",
    )


def store_source_winner(
    cache_root: Path,
    identity: SourceWinnerIdentity,
    candidate_id: str,
) -> None:
    """Atomically store a candidate that passed all final promotion gates.

    Args:
        cache_root: Caller-owned source-optimization cache directory.
        identity: Complete static replay compatibility identity.
        candidate_id: Formed candidate that passed final semantic and 3x gates.

    Raises:
        OSError: If the cache directory or atomic manifest write fails.
        ValueError: If ``candidate_id`` is not in the formed candidate universe.
    """
    if candidate_id not in identity.candidate_ids:
        raise ValueError("accepted winner is outside the formed candidate universe")
    path = _winner_path(cache_root, identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.is_symlink():
        raise ValueError(f"refusing to replace symlink: {path}")
    manifest: _WinnerManifest = {
        "schema_version": _SCHEMA_VERSION,
        "identity_key": identity.key,
        "candidate_id": candidate_id,
    }
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def winner_manifest_path(cache_root: Path, identity: SourceWinnerIdentity) -> Path:
    """Return the Atoll-owned manifest path for focused diagnostics and tests.

    Args:
        cache_root: Caller-owned source-optimization cache directory.
        identity: Complete static replay compatibility identity.

    Returns:
        Path: Content-addressed accepted-winner manifest path.
    """
    return _winner_path(cache_root, identity)


def _winner_path(cache_root: Path, identity: SourceWinnerIdentity) -> Path:
    return cache_root / _WINNER_DIRECTORY / f"{identity.key}.json"


def _validated_candidate_id(payload: object, identity_key: str) -> str:
    if not isinstance(payload, dict):
        raise TypeError("winner manifest must be an object")
    manifest = cast(dict[str, object], payload)
    if manifest.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("winner manifest schema is stale")
    if manifest.get("identity_key") != identity_key:
        raise ValueError("winner manifest identity is stale")
    candidate_id = manifest.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id.startswith("source-candidate-"):
        raise ValueError("winner manifest candidate ID is invalid")
    return candidate_id
