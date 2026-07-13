"""Strict loading, validation, and aggregation of corpus case results.

Aggregation owns no execution or lifecycle behavior.  It consumes complete
schema-v1 result envelopes, verifies them against one manifest matrix slice,
and emits deterministic summaries without combining tiers or runner platforms.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, cast

from scripts.benchmark_corpus.identity import case_digest, comparison_key
from scripts.benchmark_corpus.manifest import manifest_matrix
from scripts.benchmark_corpus.models import (
    CaseResult,
    CaseStatus,
    CorpusCase,
    CorpusManifest,
    CorpusPlatform,
    CorpusTier,
    EnvironmentEvidence,
    PhaseEvidence,
    PolicyEvidence,
    RatioEvidence,
)

RESULT_SCHEMA_VERSION = 1
AGGREGATE_SCHEMA_VERSION = 1

ResultValidity = Literal["valid", "infrastructure-invalid", "semantic-invalid"]

_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "case_id",
        "tier",
        "platform",
        "status",
        "repository",
        "revision",
        "manifest_digest",
        "case_digest",
        "comparison_key",
        "diagnostics",
        "source_digest_before",
        "source_digest_after",
        "source_unchanged",
        "policy",
        "environment",
        "phases",
        "baseline_wheel_digest",
        "compiled_wheel_digest",
        "cold_report_path",
        "warm_report_path",
        "baseline_oracle_digest",
        "compiled_oracle_digest",
        "cold_compiler_invocations",
        "warm_compiler_invocations",
        "ratios",
    }
)
_POLICY_FIELDS = frozenset({"digest", "patch_path", "source_path"})
_ENVIRONMENT_FIELDS = frozenset(
    {
        "python",
        "atoll_revision",
        "uv",
        "mypy",
        "cython",
        "compiler",
        "operating_system",
        "architecture",
        "runner_image",
        "hardware_class",
        "dependency_lock_digest",
    }
)
_PHASE_FIELDS = frozenset(
    {
        "name",
        "argv",
        "exit_code",
        "timed_out",
        "duration_seconds",
        "log_path",
        "log_truncated",
    }
)
_RATIO_FIELDS = frozenset(
    {
        "python_rewrite_vs_original",
        "final_wheel_vs_original",
        "native_vs_source_only",
        "baseline_samples_seconds",
        "source_only_samples_seconds",
        "final_wheel_samples_seconds",
    }
)
_TIERS: tuple[CorpusTier, ...] = (
    "compatibility",
    "performance",
    "calibration",
    "negative-control",
)
_PLATFORMS: tuple[CorpusPlatform, ...] = ("ubuntu-24.04", "macos-14")
_STATUSES: tuple[CaseStatus, ...] = (
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
)
_NEUTRAL_STATUSES = frozenset({"supported-no-op", "unsupported", "not-profitable"})
_INFRASTRUCTURE_INVALID = frozenset(
    {
        "upstream-broken",
        "compile-error",
        "timeout",
        "infrastructure-error",
        "security-violation",
    }
)
_SEMANTIC_INVALID = frozenset({"compatibility-regression", "unstable"})


class AggregationError(ValueError):
    """Raised when result evidence cannot form one trustworthy matrix slice."""


@dataclass(frozen=True, slots=True)
class AggregateCase:
    """Deterministic aggregate projection of one complete case result.

    Attributes:
        case_id: Stable manifest case identifier.
        status: Lifecycle outcome retained without collapsing failure categories.
        validity: Whether the outcome is valid, infrastructure-invalid, or semantic-invalid.
        comparison_key: Environment and workload identity used for historical comparison.
        accepted_speedup: End-to-end ratio only for accepted accelerated outcomes.
        effective_speedup: Accepted ratio, or 1.0 for valid neutral outcomes.
    """

    case_id: str
    status: CaseStatus
    validity: ResultValidity
    comparison_key: str | None
    accepted_speedup: float | None
    effective_speedup: float | None

    def as_json(self) -> dict[str, object]:
        """Return a JSON-ready case summary with stable field names."""
        return {
            "accepted_speedup": self.accepted_speedup,
            "case_id": self.case_id,
            "comparison_key": self.comparison_key,
            "effective_speedup": self.effective_speedup,
            "status": self.status,
            "validity": self.validity,
        }


@dataclass(frozen=True, slots=True)
class StatusCount:
    """One non-empty status bucket in deterministic status order.

    Attributes:
        status: Exact case status represented by the bucket.
        case_ids: Sorted case identifiers with that status.
    """

    status: CaseStatus
    case_ids: tuple[str, ...]

    def as_json(self) -> dict[str, object]:
        """Return a JSON-ready status bucket."""
        return {"case_ids": list(self.case_ids), "count": len(self.case_ids)}


@dataclass(frozen=True, slots=True)
class CorpusAggregate:
    """Immutable summary for exactly one tier and one runner platform.

    Attributes:
        schema_version: Aggregate output schema, currently exactly one.
        tier: The sole tier represented by this summary.
        platform: The sole platform represented by this summary.
        manifest_digest: Digest shared by the selected result files.
        cases: Case summaries sorted by manifest identifier.
        statuses: Non-empty exact-status buckets in stable status order.
        accelerated_coverage: Accelerated cases divided by all expected cases.
        accepted_geometric_mean: Geometric mean of accepted acceleration ratios only.
        effective_corpus_speedup: Geometric mean across measurable valid outcomes;
            neutral outcomes contribute 1.0 and invalid or unmeasured outcomes are excluded.
        infrastructure_invalid_case_ids: Sorted setup, toolchain, or execution failures.
        semantic_invalid_case_ids: Sorted semantic regression or unstable-result failures.
        unmeasured_valid_case_ids: Valid cases without an effective speedup observation.
    """

    schema_version: int
    tier: CorpusTier
    platform: CorpusPlatform
    manifest_digest: str
    cases: tuple[AggregateCase, ...]
    statuses: tuple[StatusCount, ...]
    accelerated_coverage: float
    accepted_geometric_mean: float | None
    effective_corpus_speedup: float | None
    infrastructure_invalid_case_ids: tuple[str, ...]
    semantic_invalid_case_ids: tuple[str, ...]
    unmeasured_valid_case_ids: tuple[str, ...]

    def as_json(self) -> dict[str, object]:
        """Return deterministic JSON-ready aggregate data."""
        return {
            "accepted_geometric_mean": self.accepted_geometric_mean,
            "accelerated_coverage": self.accelerated_coverage,
            "cases": [case.as_json() for case in self.cases],
            "effective_corpus_speedup": self.effective_corpus_speedup,
            "infrastructure_invalid_case_ids": list(self.infrastructure_invalid_case_ids),
            "manifest_digest": self.manifest_digest,
            "platform": self.platform,
            "schema_version": self.schema_version,
            "semantic_invalid_case_ids": list(self.semantic_invalid_case_ids),
            "statuses": {bucket.status: bucket.as_json() for bucket in self.statuses},
            "tier": self.tier,
            "unmeasured_valid_case_ids": list(self.unmeasured_valid_case_ids),
        }


@dataclass(frozen=True, slots=True)
class HistoricalCaseComparison:
    """Like-for-like historical speedup comparison for one case.

    Attributes:
        case_id: Stable manifest case identifier.
        comparison_key: Matching environment and workload identity.
        current_speedup: Current effective speedup when measurable.
        historical_speedup: Historical effective speedup when measurable.
        current_vs_historical: Current divided by historical when both are measurable.
    """

    case_id: str
    comparison_key: str
    current_speedup: float | None
    historical_speedup: float | None
    current_vs_historical: float | None


def load_case_result(path: Path) -> CaseResult:
    """Load one complete schema-v1 case result with strict field validation.

    Args:
        path: JSON result file produced by the corpus lifecycle.

    Returns:
        CaseResult: Fully typed immutable evidence.

    Raises:
        AggregationError: If the file is unreadable, malformed, incomplete, or not schema v1.
    """
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AggregationError(f"cannot read case result {path}: {error}") from error
    payload = _mapping(raw, str(path))
    return _parse_result(payload, str(path))


def load_case_results(result_paths: tuple[Path, ...]) -> tuple[CaseResult, ...]:
    """Load each result path exactly once for a validation transaction.

    Callers that both aggregate and retain case-level evidence must carry these
    immutable objects across that boundary instead of reopening mutable files.

    Args:
        result_paths: Case-result JSON files in arbitrary order.

    Returns:
        tuple[CaseResult, ...]: Parsed evidence in input order.
    """
    return tuple(load_case_result(path) for path in result_paths)


def aggregate_case_results(
    manifest: CorpusManifest,
    result_paths: tuple[Path, ...],
    *,
    tier: CorpusTier,
    platform: CorpusPlatform,
) -> CorpusAggregate:
    """Aggregate exactly one complete manifest matrix tier/platform slice.

    Input files are never filtered.  A file for another tier, another platform,
    or a case outside the selected matrix slice is rejected so accidental
    cross-platform or cross-purpose aggregation cannot look successful.

    Args:
        manifest: Validated manifest defining the expected matrix identities.
        result_paths: Case-result JSON files in arbitrary order.
        tier: Exact tier to aggregate.
        platform: Exact runner platform to aggregate.

    Returns:
        CorpusAggregate: Deterministic immutable summary.

    Raises:
        AggregationError: If identities, digests, result schemas, or coverage are invalid.
    """
    return aggregate_loaded_results(
        manifest,
        load_case_results(result_paths),
        tier=tier,
        platform=platform,
    )


def aggregate_loaded_results(
    manifest: CorpusManifest,
    results: tuple[CaseResult, ...],
    *,
    tier: CorpusTier,
    platform: CorpusPlatform,
) -> CorpusAggregate:
    """Aggregate one already-loaded result transaction.

    This variant is the promotion boundary: aggregate metrics and retained
    per-case fields are derived from the same immutable reads.

    Args:
        manifest: Validated manifest defining the expected matrix identities.
        results: Parsed case evidence loaded exactly once.
        tier: Exact tier to aggregate.
        platform: Exact runner platform to aggregate.

    Returns:
        CorpusAggregate: Deterministic immutable summary.

    Raises:
        AggregationError: If the selected matrix is empty or its evidence is
            incomplete, duplicated, stale, or otherwise invalid.
    """
    expected_rows = manifest_matrix(manifest, tier=tier, platform=platform)
    if not expected_rows:
        raise AggregationError(
            f"selected matrix slice has no expected cases: tier={tier}, platform={platform}"
        )
    expected_ids = tuple(row.case_id for row in expected_rows)
    expected = frozenset(expected_ids)
    by_id = _index_selected_results(results, expected, tier, platform)
    missing = sorted(expected - set(by_id))
    if missing:
        raise AggregationError(f"missing case result(s): {', '.join(missing)}")

    manifest_digest = hashlib.sha256(manifest.path.read_bytes()).hexdigest()
    ordered = tuple(by_id[case_id] for case_id in expected_ids)
    manifest_cases = {case.id: case for case in manifest.cases}
    _validate_manifest_identities(ordered, manifest_cases, manifest_digest)

    cases = tuple(_aggregate_case(result) for result in ordered)
    statuses = tuple(
        StatusCount(
            status=status,
            case_ids=tuple(case.case_id for case in cases if case.status == status),
        )
        for status in _STATUSES
        if any(case.status == status for case in cases)
    )
    accepted = tuple(case.accepted_speedup for case in cases if case.accepted_speedup is not None)
    effective = tuple(
        case.effective_speedup for case in cases if case.effective_speedup is not None
    )
    count = len(cases)
    return CorpusAggregate(
        schema_version=AGGREGATE_SCHEMA_VERSION,
        tier=tier,
        platform=platform,
        manifest_digest=manifest_digest,
        cases=cases,
        statuses=statuses,
        accelerated_coverage=(sum(case.status == "accelerated" for case in cases) / count)
        if count
        else 0.0,
        accepted_geometric_mean=_geometric_mean(accepted),
        effective_corpus_speedup=_geometric_mean(effective),
        infrastructure_invalid_case_ids=tuple(
            case.case_id for case in cases if case.validity == "infrastructure-invalid"
        ),
        semantic_invalid_case_ids=tuple(
            case.case_id for case in cases if case.validity == "semantic-invalid"
        ),
        unmeasured_valid_case_ids=tuple(
            case.case_id
            for case in cases
            if case.validity == "valid" and case.effective_speedup is None
        ),
    )


def _index_selected_results(
    results: tuple[CaseResult, ...],
    expected: frozenset[str],
    tier: CorpusTier,
    platform: CorpusPlatform,
) -> dict[str, CaseResult]:
    by_id: dict[str, CaseResult] = {}
    duplicates: set[str] = set()
    for result in results:
        if result.tier != tier:
            raise AggregationError(
                f"result {result.case_id} has tier {result.tier}, expected selected tier {tier}"
            )
        if result.platform != platform:
            raise AggregationError(
                f"result {result.case_id} has platform {result.platform}, "
                f"expected selected platform {platform}"
            )
        if result.case_id not in expected:
            raise AggregationError(
                f"result {result.case_id} is not in the selected {tier}/{platform} matrix"
            )
        if result.case_id in by_id:
            duplicates.add(result.case_id)
        by_id[result.case_id] = result
    if duplicates:
        raise AggregationError(f"duplicate case result(s): {', '.join(sorted(duplicates))}")
    return by_id


def _validate_manifest_identities(
    results: tuple[CaseResult, ...],
    manifest_cases: dict[str, CorpusCase],
    manifest_digest: str,
) -> None:
    for result in results:
        case = manifest_cases[result.case_id]
        if result.repository != case.repository or result.revision != case.revision:
            raise AggregationError(
                f"result {result.case_id} does not match its manifest repository and revision"
            )
        if result.manifest_digest != manifest_digest:
            raise AggregationError(f"result {result.case_id} has a stale manifest digest")
        expected_case_digest = case_digest(case)
        if result.case_digest != expected_case_digest:
            raise AggregationError(f"result {result.case_id} has a stale case digest")
        expected_comparison_key = (
            comparison_key(case, result.environment, result.policy)
            if result.environment is not None and result.policy is not None
            else None
        )
        if result.comparison_key != expected_comparison_key:
            raise AggregationError(
                f"result {result.case_id} has a stale or unverifiable comparison key"
            )


def compare_aggregates(
    current: CorpusAggregate, historical: CorpusAggregate
) -> tuple[HistoricalCaseComparison, ...]:
    """Compare aggregates only when every case has an identical comparison key.

    Args:
        current: Current aggregate.
        historical: Candidate historical aggregate.

    Returns:
        tuple[HistoricalCaseComparison, ...]: Case comparisons in stable case order.

    Raises:
        AggregationError: If tier, platform, case coverage, or comparison keys differ.
    """
    if current.tier != historical.tier or current.platform != historical.platform:
        raise AggregationError("historical comparison requires the same tier and platform")
    current_cases = {case.case_id: case for case in current.cases}
    historical_cases = {case.case_id: case for case in historical.cases}
    if current_cases.keys() != historical_cases.keys():
        raise AggregationError("historical comparison requires identical case coverage")
    comparisons: list[HistoricalCaseComparison] = []
    for case_id in sorted(current_cases):
        current_case = current_cases[case_id]
        historical_case = historical_cases[case_id]
        if (
            current_case.comparison_key is None
            or current_case.comparison_key != historical_case.comparison_key
        ):
            raise AggregationError(
                f"historical comparison key differs or is unavailable for case {case_id}"
            )
        ratio = (
            None
            if current_case.effective_speedup is None or historical_case.effective_speedup is None
            else current_case.effective_speedup / historical_case.effective_speedup
        )
        comparisons.append(
            HistoricalCaseComparison(
                case_id=case_id,
                comparison_key=current_case.comparison_key,
                current_speedup=current_case.effective_speedup,
                historical_speedup=historical_case.effective_speedup,
                current_vs_historical=ratio,
            )
        )
    return tuple(comparisons)


def render_aggregate_json(aggregate: CorpusAggregate) -> str:
    """Render canonical aggregate JSON ending in one newline.

    Args:
        aggregate: One strict tier/platform aggregate.

    Returns:
        str: Stable pretty-printed JSON.
    """
    return f"{json.dumps(aggregate.as_json(), indent=2, sort_keys=True)}\n"


def render_aggregate_markdown(aggregate: CorpusAggregate) -> str:
    """Render a deterministic human summary without changing metric semantics.

    Args:
        aggregate: One strict tier/platform aggregate.

    Returns:
        str: Markdown ending in one newline.
    """
    lines = [
        f"# Corpus aggregate: {aggregate.tier} on {aggregate.platform}",
        "",
        f"- Expected cases: {len(aggregate.cases)}",
        f"- Accelerated coverage: {aggregate.accelerated_coverage:.3%}",
        f"- Accepted-only geometric mean: {_ratio(aggregate.accepted_geometric_mean)}",
        f"- Effective corpus speedup: {_ratio(aggregate.effective_corpus_speedup)}",
        "",
        "## Statuses",
        "",
    ]
    lines.extend(
        f"- `{bucket.status}`: {len(bucket.case_ids)} ({', '.join(bucket.case_ids)})"
        for bucket in aggregate.statuses
    )
    if not aggregate.statuses:
        lines.append("- None")
    lines.extend(("", "## Cases", ""))
    lines.extend(
        (
            f"- `{case.case_id}`: `{case.status}`, `{case.validity}`, "
            f"effective {_ratio(case.effective_speedup)}"
        )
        for case in aggregate.cases
    )
    if not aggregate.cases:
        lines.append("- None")
    return f"{'\n'.join(lines)}\n"


def _parse_result(payload: dict[str, object], label: str) -> CaseResult:
    _require_exact_fields(payload, _RESULT_FIELDS, label)
    version = _integer(payload, "schema_version", label)
    if version != RESULT_SCHEMA_VERSION:
        raise AggregationError(
            f"{label}.schema_version must be {RESULT_SCHEMA_VERSION}, got {version}"
        )
    tier = _literal(payload, "tier", label, _TIERS)
    platform = _literal(payload, "platform", label, _PLATFORMS)
    status = _literal(payload, "status", label, _STATUSES)
    raw_policy = payload.get("policy")
    raw_environment = payload.get("environment")
    raw_phases = _sequence(payload, "phases", label)
    raw_ratios = _mapping(payload.get("ratios"), f"{label}.ratios")
    return CaseResult(
        schema_version=version,
        case_id=_string(payload, "case_id", label),
        tier=cast(CorpusTier, tier),
        platform=cast(CorpusPlatform, platform),
        status=cast(CaseStatus, status),
        repository=_string(payload, "repository", label),
        revision=_string(payload, "revision", label),
        manifest_digest=_string(payload, "manifest_digest", label),
        case_digest=_string(payload, "case_digest", label),
        comparison_key=_optional_string(payload, "comparison_key", label),
        diagnostics=_string_tuple(payload, "diagnostics", label),
        source_digest_before=_optional_string(payload, "source_digest_before", label),
        source_digest_after=_optional_string(payload, "source_digest_after", label),
        source_unchanged=_optional_bool(payload, "source_unchanged", label),
        policy=None
        if raw_policy is None
        else _parse_policy(_mapping(raw_policy, f"{label}.policy"), f"{label}.policy"),
        environment=None
        if raw_environment is None
        else _parse_environment(
            _mapping(raw_environment, f"{label}.environment"), f"{label}.environment"
        ),
        phases=tuple(
            _parse_phase(_mapping(raw, f"{label}.phases[{index}]"), f"{label}.phases[{index}]")
            for index, raw in enumerate(raw_phases)
        ),
        baseline_wheel_digest=_optional_string(payload, "baseline_wheel_digest", label),
        compiled_wheel_digest=_optional_string(payload, "compiled_wheel_digest", label),
        cold_report_path=_optional_path(payload, "cold_report_path", label),
        warm_report_path=_optional_path(payload, "warm_report_path", label),
        baseline_oracle_digest=_optional_string(payload, "baseline_oracle_digest", label),
        compiled_oracle_digest=_optional_string(payload, "compiled_oracle_digest", label),
        cold_compiler_invocations=_optional_nonnegative_integer(
            payload, "cold_compiler_invocations", label
        ),
        warm_compiler_invocations=_optional_nonnegative_integer(
            payload, "warm_compiler_invocations", label
        ),
        ratios=_parse_ratios(raw_ratios, f"{label}.ratios"),
    )


def _parse_policy(payload: dict[str, object], label: str) -> PolicyEvidence:
    _require_exact_fields(payload, _POLICY_FIELDS, label)
    return PolicyEvidence(
        digest=_string(payload, "digest", label),
        patch_path=_path(payload, "patch_path", label),
        source_path=_path(payload, "source_path", label),
    )


def _parse_environment(payload: dict[str, object], label: str) -> EnvironmentEvidence:
    _require_exact_fields(payload, _ENVIRONMENT_FIELDS, label)
    return EnvironmentEvidence(
        python=_string(payload, "python", label),
        atoll_revision=_string(payload, "atoll_revision", label),
        uv=_string(payload, "uv", label),
        mypy=_string(payload, "mypy", label),
        cython=_string(payload, "cython", label),
        compiler=_string(payload, "compiler", label),
        operating_system=_string(payload, "operating_system", label),
        architecture=_string(payload, "architecture", label),
        runner_image=_string(payload, "runner_image", label),
        hardware_class=_string(payload, "hardware_class", label),
        dependency_lock_digest=_string(payload, "dependency_lock_digest", label),
    )


def _parse_phase(payload: dict[str, object], label: str) -> PhaseEvidence:
    _require_exact_fields(payload, _PHASE_FIELDS, label)
    duration = _number(payload, "duration_seconds", label, allow_zero=True)
    return PhaseEvidence(
        name=_string(payload, "name", label),
        argv=_string_tuple(payload, "argv", label),
        exit_code=_optional_integer(payload, "exit_code", label),
        timed_out=_bool(payload, "timed_out", label),
        duration_seconds=duration,
        log_path=_path(payload, "log_path", label),
        log_truncated=_bool(payload, "log_truncated", label),
    )


def _parse_ratios(payload: dict[str, object], label: str) -> RatioEvidence:
    _require_exact_fields(payload, _RATIO_FIELDS, label)
    return RatioEvidence(
        python_rewrite_vs_original=_optional_number(payload, "python_rewrite_vs_original", label),
        final_wheel_vs_original=_optional_number(payload, "final_wheel_vs_original", label),
        native_vs_source_only=_optional_number(payload, "native_vs_source_only", label),
        baseline_samples_seconds=_number_tuple(payload, "baseline_samples_seconds", label),
        source_only_samples_seconds=_number_tuple(payload, "source_only_samples_seconds", label),
        final_wheel_samples_seconds=_number_tuple(payload, "final_wheel_samples_seconds", label),
    )


def _aggregate_case(result: CaseResult) -> AggregateCase:
    validity: ResultValidity
    if result.status in _INFRASTRUCTURE_INVALID:
        validity = "infrastructure-invalid"
    elif result.status in _SEMANTIC_INVALID:
        validity = "semantic-invalid"
    else:
        validity = "valid"
    accepted = result.ratios.final_wheel_vs_original if result.status == "accelerated" else None
    if result.status == "accelerated" and accepted is None:
        raise AggregationError(
            f"accelerated result {result.case_id} has no final-wheel versus original ratio"
        )
    effective = 1.0 if result.status in _NEUTRAL_STATUSES else accepted
    return AggregateCase(
        case_id=result.case_id,
        status=result.status,
        validity=validity,
        comparison_key=result.comparison_key,
        accepted_speedup=accepted,
        effective_speedup=effective,
    )


def _geometric_mean(values: tuple[float, ...]) -> float | None:
    if not values:
        return None
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AggregationError(f"{label} must be a JSON object with string keys")
    untyped = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in untyped):
        raise AggregationError(f"{label} must be a JSON object with string keys")
    return cast(dict[str, object], value)


def _require_exact_fields(
    payload: dict[str, object],
    expected: frozenset[str],
    label: str,
) -> None:
    """Require complete schema objects, including explicitly nullable fields."""
    missing = sorted(expected - set(payload))
    if missing:
        raise AggregationError(f"{label} is missing field(s): {', '.join(missing)}")
    unknown = sorted(set(payload) - expected)
    if unknown:
        raise AggregationError(f"{label} contains unknown field(s): {', '.join(unknown)}")


def _string(payload: dict[str, object], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise AggregationError(f"{label}.{key} must be a non-empty string")
    return value


def _optional_string(payload: dict[str, object], key: str, label: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise AggregationError(f"{label}.{key} must be null or a non-empty string")
    return value


def _integer(payload: dict[str, object], key: str, label: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise AggregationError(f"{label}.{key} must be an integer")
    return value


def _optional_integer(payload: dict[str, object], key: str, label: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise AggregationError(f"{label}.{key} must be null or an integer")
    return value


def _optional_nonnegative_integer(payload: dict[str, object], key: str, label: str) -> int | None:
    value = _optional_integer(payload, key, label)
    if value is not None and value < 0:
        raise AggregationError(f"{label}.{key} must be null or non-negative")
    return value


def _bool(payload: dict[str, object], key: str, label: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise AggregationError(f"{label}.{key} must be a boolean")
    return value


def _optional_bool(payload: dict[str, object], key: str, label: str) -> bool | None:
    value = payload.get(key)
    if value is not None and not isinstance(value, bool):
        raise AggregationError(f"{label}.{key} must be null or a boolean")
    return value


def _literal(payload: dict[str, object], key: str, label: str, allowed: tuple[str, ...]) -> str:
    value = _string(payload, key, label)
    if value not in allowed:
        raise AggregationError(f"{label}.{key} has unsupported value {value!r}")
    return value


def _sequence(payload: dict[str, object], key: str, label: str) -> list[object]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise AggregationError(f"{label}.{key} must be an array")
    return cast(list[object], value)


def _string_tuple(payload: dict[str, object], key: str, label: str) -> tuple[str, ...]:
    values = _sequence(payload, key, label)
    if any(not isinstance(value, str) for value in values):
        raise AggregationError(f"{label}.{key} must contain only strings")
    return cast(tuple[str, ...], tuple(values))


def _number(payload: dict[str, object], key: str, label: str, *, allow_zero: bool) -> float:
    value = payload.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise AggregationError(f"{label}.{key} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (not allow_zero and number == 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise AggregationError(f"{label}.{key} must be a finite {qualifier} number")
    return number


def _optional_number(payload: dict[str, object], key: str, label: str) -> float | None:
    return None if payload.get(key) is None else _number(payload, key, label, allow_zero=False)


def _number_tuple(payload: dict[str, object], key: str, label: str) -> tuple[float, ...]:
    values = _sequence(payload, key, label)
    parsed: list[float] = []
    for index, value in enumerate(values):
        parsed.append(
            _number(
                {"value": value},
                "value",
                f"{label}.{key}[{index}]",
                allow_zero=False,
            )
        )
    return tuple(parsed)


def _path(payload: dict[str, object], key: str, label: str) -> PurePosixPath:
    value = PurePosixPath(_string(payload, key, label))
    if value.is_absolute() or ".." in value.parts:
        raise AggregationError(f"{label}.{key} must be a safe relative path")
    return value


def _optional_path(payload: dict[str, object], key: str, label: str) -> PurePosixPath | None:
    return None if payload.get(key) is None else _path(payload, key, label)


def _ratio(value: float | None) -> str:
    return "not measured" if value is None else f"{value:.3f}x"
