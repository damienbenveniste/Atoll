"""Immutable contracts for corpus manifests and workflow matrix entries.

These models describe benchmark intent only.  External checkout lifecycle,
subprocess execution, and result aggregation are separate collaborators so a
manifest can be validated without network or toolchain access.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

CorpusBackend = Literal["mypyc", "cython"]
CorpusTier = Literal["compatibility", "performance", "calibration", "negative-control"]
CorpusPlatform = Literal["ubuntu-24.04", "macos-14"]
WorkloadSource = Literal["upstream", "pyperformance", "mypyc-benchmarks", "atoll"]


@dataclass(frozen=True, slots=True)
class CorpusDefaults:
    """Resource boundaries applied when a case does not override a timeout.

    Attributes:
        test_timeout_seconds: Maximum focused-test or oracle duration.
        compile_timeout_seconds: Maximum ordinary compatibility compile duration.
        performance_timeout_seconds: Maximum performance-case compile duration.
        max_log_bytes: Maximum retained bytes for each subprocess log.
    """

    test_timeout_seconds: int
    compile_timeout_seconds: int
    performance_timeout_seconds: int
    max_log_bytes: int


@dataclass(frozen=True, slots=True)
class WorkloadProvenance:
    """Immutable origin and digest for one performance workload body.

    Attributes:
        source: Upstream suite or Atoll adapter family supplying the workload.
        repository: Canonical repository containing the original workload.
        revision: Full immutable commit SHA for the workload source.
        path: Source-relative path of the original workload.
        sha256: Digest of the reviewed workload body or minimal adapter.
        notice: Repository-relative third-party notice retained by Atoll.
    """

    source: WorkloadSource
    repository: str
    revision: str
    path: PurePosixPath
    sha256: str
    notice: PurePosixPath


@dataclass(frozen=True, slots=True)
class CorpusCase:
    """One immutable external project and its qualification contract.

    Attributes:
        id: Stable lowercase case identifier.
        name: Human-readable project or workload name.
        repository: Canonical HTTPS Git repository URL.
        revision: Full detached commit SHA.
        project_subroot: Python project root inside the checkout.
        dependency_lock: Repository-relative reviewed constraint file.
        focused_test_command: Upstream test argv run before Atoll compilation.
        oracle_adapter: Repository-local adapter name producing canonical JSON.
        tiers: Compatibility, performance, calibration, or negative-control roles.
        platforms: Runner images on which this case is supported.
        workload: Performance workload provenance when applicable.
        test_timeout_seconds: Optional per-case focused-test timeout override.
        compile_timeout_seconds: Optional per-case normal compile timeout override.
        performance_timeout_seconds: Optional performance compile timeout override.
    """

    id: str
    name: str
    repository: str
    revision: str
    project_subroot: PurePosixPath
    dependency_lock: PurePosixPath
    focused_test_command: tuple[str, ...]
    oracle_adapter: str
    tiers: tuple[CorpusTier, ...]
    platforms: tuple[CorpusPlatform, ...]
    workload: WorkloadProvenance | None = None
    test_timeout_seconds: int | None = None
    compile_timeout_seconds: int | None = None
    performance_timeout_seconds: int | None = None


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    """Validated benchmark corpus configuration safe to pass across commands.

    Attributes:
        path: Source manifest path used to resolve repository-relative files.
        schema_version: Manifest schema version, currently exactly one.
        python_version: Python minor version used by the initial corpus.
        backends: Required Atoll backend order.
        defaults: Shared timeout and log-retention boundaries.
        cases: Cases sorted by stable identifier.
    """

    path: Path
    schema_version: int
    python_version: str
    backends: tuple[CorpusBackend, ...]
    defaults: CorpusDefaults
    cases: tuple[CorpusCase, ...]


@dataclass(frozen=True, slots=True)
class MatrixEntry:
    """One deterministic GitHub Actions matrix row.

    Attributes:
        case_id: Manifest case identifier.
        tier: Selected benchmark role.
        platform: Concrete workflow runner label.
    """

    case_id: str
    tier: CorpusTier
    platform: CorpusPlatform

    def as_json(self) -> dict[str, str]:
        """Return the stable mapping expected by GitHub Actions."""
        return {"case": self.case_id, "tier": self.tier, "platform": self.platform}
