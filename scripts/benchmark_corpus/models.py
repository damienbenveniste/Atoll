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
CaseStatus = Literal[
    "accelerated",
    "compiled-unbenchmarked",
    "supported-no-op",
    "not-profitable",
    "unsupported",
    "upstream-broken",
    "compile-error",
    "compatibility-regression",
    "unstable",
    "timeout",
    "infrastructure-error",
    "security-violation",
]


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
        sha256: Digest of the reviewed workload, case adapter, shared runner,
            golden result, and third-party notice bundle.
        notice: Repository-relative third-party notice retained by Atoll.
    """

    source: WorkloadSource
    repository: str
    revision: str
    path: PurePosixPath
    sha256: str
    notice: PurePosixPath


@dataclass(frozen=True, slots=True)
class SdistSource:
    """Content-addressed source-distribution identity.

    Attributes:
        url: Canonical HTTPS URL for the immutable archive bytes.
        archive_sha256: SHA-256 of the complete archive before parsing.
        archive_size: Exact expected archive byte count.
        tree_sha256: Digest of normalized extracted regular-file paths and bytes.
    """

    url: str
    archive_sha256: str
    archive_size: int
    tree_sha256: str


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
        oracle_arguments: Reviewed arguments passed to the oracle adapter.
        tiers: Compatibility, performance, calibration, or negative-control roles.
        platforms: Runner images on which this case is supported.
        workload: Performance workload provenance when applicable.
        test_timeout_seconds: Optional per-case focused-test timeout override.
        compile_timeout_seconds: Optional per-case normal compile timeout override.
        performance_timeout_seconds: Optional performance compile timeout override.
        sdist: Content-addressed sdist source, or ``None`` for strict Git checkout.
    """

    id: str
    name: str
    repository: str
    revision: str
    project_subroot: PurePosixPath
    dependency_lock: PurePosixPath
    focused_test_command: tuple[str, ...]
    oracle_adapter: str
    oracle_arguments: tuple[str, ...]
    tiers: tuple[CorpusTier, ...]
    platforms: tuple[CorpusPlatform, ...]
    workload: WorkloadProvenance | None = None
    test_timeout_seconds: int | None = None
    compile_timeout_seconds: int | None = None
    performance_timeout_seconds: int | None = None
    sdist: SdistSource | None = None


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


@dataclass(frozen=True, slots=True)
class CompilePolicy:
    """Exact Atoll policy appended only to a disposable checkout.

    Attributes:
        backends: Compiler backends in manifest preference order.
        test_command: Optional semantic command run by source-clean compile.
        benchmark_command: Optional workload command used for profitability.
        benchmark_warmups: Number of unmeasured benchmark pairs.
        benchmark_samples: Number of measured benchmark pairs.
        minimum_speedup: Final-wheel promotion threshold.
    """

    backends: tuple[CorpusBackend, ...]
    test_command: tuple[str, ...] | None = None
    benchmark_command: tuple[str, ...] | None = None
    benchmark_warmups: int = 1
    benchmark_samples: int = 7
    minimum_speedup: float = 1.10


@dataclass(frozen=True, slots=True)
class PolicyEvidence:
    """Reviewable source change introducing disposable compile configuration.

    Attributes:
        digest: SHA-256 digest of the exact retained unified policy patch.
        patch_path: Evidence-relative path containing a unified source patch.
        source_path: Checkout-relative pyproject path changed by the policy.
    """

    digest: str
    patch_path: PurePosixPath
    source_path: PurePosixPath


@dataclass(frozen=True, slots=True)
class PhaseEvidence:
    """Bounded subprocess evidence for one lifecycle phase.

    Attributes:
        name: Stable lifecycle phase name.
        argv: Exact command without shell interpretation.
        exit_code: Process return code, or ``None`` when timeout prevented one.
        timed_out: Whether the process exceeded its configured deadline.
        duration_seconds: Wall-clock phase duration.
        log_path: Evidence-relative bounded combined-output log.
        log_truncated: Whether output exceeded the retained byte limit.
    """

    name: str
    argv: tuple[str, ...]
    exit_code: int | None
    timed_out: bool
    duration_seconds: float
    log_path: PurePosixPath
    log_truncated: bool


@dataclass(frozen=True, slots=True)
class EnvironmentEvidence:
    """Comparison-critical interpreter, compiler, and runner identity.

    Attributes:
        python: Full Python implementation and patch version.
        atoll_revision: Atoll Git revision under evaluation.
        uv: uv version string.
        mypy: mypy/mypyc version string.
        cython: Cython version string.
        compiler: Native compiler identity.
        operating_system: Normalized OS and release.
        architecture: Machine architecture.
        runner_image: CI image or explicit local marker.
        hardware_class: Stable reviewed machine-class label.
        dependency_lock_digest: SHA-256 of the reviewed case constraints.
    """

    python: str
    atoll_revision: str
    uv: str
    mypy: str
    cython: str
    compiler: str
    operating_system: str
    architecture: str
    runner_image: str
    hardware_class: str
    dependency_lock_digest: str


@dataclass(frozen=True, slots=True)
class RatioEvidence:
    """Unambiguous end-to-end and composition speed ratios.

    Attributes:
        python_rewrite_vs_original: Original Python median divided by rewritten-source median.
        final_wheel_vs_original: Original Python median divided by final-wheel median.
        native_vs_source_only: Accepted source-only median divided by composed-wheel median.
        baseline_samples_seconds: Raw original-project wall-clock samples.
        source_only_samples_seconds: Raw source-transformed samples when present.
        final_wheel_samples_seconds: Raw final-wheel wall-clock samples.
    """

    python_rewrite_vs_original: float | None = None
    final_wheel_vs_original: float | None = None
    native_vs_source_only: float | None = None
    baseline_samples_seconds: tuple[float, ...] = ()
    source_only_samples_seconds: tuple[float, ...] = ()
    final_wheel_samples_seconds: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class CaseResult:
    """Schema-v1 evidence envelope emitted for every expected corpus case.

    The object remains valid for setup failures: unavailable fields are
    represented by ``None`` while status and diagnostics still explain the
    attribution.  Source and wheel payloads are referred to only by digest.

    Attributes:
        schema_version: Corpus result schema, currently exactly one.
        case_id: Stable manifest case identifier.
        tier: Executed corpus tier.
        platform: Workflow runner label.
        status: Explicit compatibility or acceleration outcome.
        repository: Pinned upstream repository URL.
        revision: Pinned upstream commit SHA.
        manifest_digest: Digest of the complete corpus manifest.
        case_digest: Digest of normalized case configuration.
        comparison_key: Environment/workload key excluding Atoll revision.
        diagnostics: Stable human-readable attribution details.
        source_digest_before: Tracked checkout digest after policy injection.
        source_digest_after: Tracked checkout digest after all Atoll phases.
        source_unchanged: Whether Atoll changed any tracked source after injection.
        policy: Appended disposable-policy evidence.
        environment: Toolchain and runner identity when probing completed.
        phases: Ordered bounded subprocess records.
        baseline_wheel_digest: Normal project wheel SHA-256.
        compiled_wheel_digest: Final Atoll wheel SHA-256.
        cold_report_path: Evidence-relative copied cold compile report.
        warm_report_path: Evidence-relative copied warm compile report.
        baseline_oracle_digest: Canonical baseline-oracle JSON digest.
        compiled_oracle_digest: Canonical final-wheel oracle JSON digest.
        cold_compiler_invocations: Compiler processes observed during the cold compile.
        warm_compiler_invocations: Compiler processes observed during the warm compile.
        ratios: Raw timing samples and clearly labeled ratios.
    """

    schema_version: int
    case_id: str
    tier: CorpusTier
    platform: CorpusPlatform
    status: CaseStatus
    repository: str
    revision: str
    manifest_digest: str
    case_digest: str
    comparison_key: str | None
    diagnostics: tuple[str, ...]
    source_digest_before: str | None
    source_digest_after: str | None
    source_unchanged: bool | None
    policy: PolicyEvidence | None
    environment: EnvironmentEvidence | None
    phases: tuple[PhaseEvidence, ...]
    baseline_wheel_digest: str | None
    compiled_wheel_digest: str | None
    cold_report_path: PurePosixPath | None
    warm_report_path: PurePosixPath | None
    baseline_oracle_digest: str | None
    compiled_oracle_digest: str | None
    cold_compiler_invocations: int | None
    warm_compiler_invocations: int | None
    ratios: RatioEvidence
