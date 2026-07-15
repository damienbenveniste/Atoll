"""Manual promotion and documentation rendering for corpus evidence.

Promotion consumes a complete, already validated tier/platform result slice and
writes a compact repository snapshot.  It deliberately omits raw logs, wheels,
and timing samples; those remain workflow artifacts.  The module also owns the
generated history block in ``docs/benchmarks.md`` while preserving the authored
benchmark contract around that block.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from scripts.benchmark_corpus.aggregation import aggregate_loaded_results, load_case_results
from scripts.benchmark_corpus.models import (
    CaseStatus,
    CorpusManifest,
    CorpusPlatform,
    CorpusTier,
)

SNAPSHOT_SCHEMA_VERSION = 1
EXPERIMENT_SCHEMA_VERSION = 1
HISTORY_START = "<!-- corpus-history:start -->"
HISTORY_END = "<!-- corpus-history:end -->"
_LABEL = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
_HEX_SHA = re.compile(r"[0-9a-f]{40}\Z")
_MAX_REVIEWER_LENGTH = 128
_MAX_EXPERIMENT_BYTES = 4096
_MAX_WORKFLOW_REF_LENGTH = 512
_REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
_SNAPSHOT_FIELDS = frozenset(
    {
        "accepted_geometric_mean",
        "accelerated_coverage",
        "atoll_revision",
        "cases",
        "effective_corpus_speedup",
        "experiment",
        "infrastructure_invalid_case_ids",
        "label",
        "manifest_digest",
        "platform",
        "reviewed_by",
        "schema_version",
        "semantic_invalid_case_ids",
        "tier",
        "unmeasured_valid_case_ids",
    }
)
_EXPERIMENT_FIELDS = frozenset(
    {
        "head_sha",
        "label",
        "repository",
        "run_attempt",
        "run_id",
        "schema_version",
        "workflow_ref",
    }
)
_CASE_FIELDS = frozenset(
    {
        "case_digest",
        "case_id",
        "comparison_key",
        "final_wheel_vs_original",
        "native_layer_vs_source_only_wheel",
        "python_rewrite_vs_original",
        "status",
        "upstream_revision",
    }
)
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
_TIERS: tuple[CorpusTier, ...] = (
    "compatibility",
    "performance",
    "calibration",
    "negative-control",
)
_PLATFORMS: tuple[CorpusPlatform, ...] = ("ubuntu-24.04", "macos-14")


class HistoryError(ValueError):
    """Raised when evidence cannot be promoted without losing reviewability."""


@dataclass(frozen=True, slots=True)
class PromotionOptions:
    """Reviewed destination and matrix identity for one promotion operation."""

    tier: CorpusTier
    platform: CorpusPlatform
    label: str
    reviewed_by: str
    history_root: Path
    docs_path: Path


@dataclass(frozen=True, slots=True)
class ExperimentIdentity:
    """Immutable GitHub Actions batch identity retained with performance evidence."""

    schema_version: int
    label: str
    repository: str
    workflow_ref: str
    run_id: int
    run_attempt: int
    head_sha: str

    def as_json(self) -> dict[str, object]:
        """Return canonical workflow evidence for a reviewed snapshot."""
        return {
            "head_sha": self.head_sha,
            "label": self.label,
            "repository": self.repository,
            "run_attempt": self.run_attempt,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "workflow_ref": self.workflow_ref,
        }


@dataclass(frozen=True, slots=True)
class SnapshotCase:
    """Compact retained evidence for one expected case.

    Attributes:
        case_id: Stable manifest case identifier.
        status: Exact lifecycle classification.
        upstream_revision: Immutable repository commit tested by the case.
        case_digest: Digest of the complete normalized case contract.
        comparison_key: Like-for-like environment and workload key, excluding Atoll.
        python_rewrite_vs_original: Rewritten-source speedup when measured.
        final_wheel_vs_original: End-to-end final-wheel speedup when measured.
        native_layer_vs_source_only_wheel: Marginal native composition speedup.
    """

    case_id: str
    status: CaseStatus
    upstream_revision: str
    case_digest: str
    comparison_key: str | None
    python_rewrite_vs_original: float | None
    final_wheel_vs_original: float | None
    native_layer_vs_source_only_wheel: float | None

    def as_json(self) -> dict[str, object]:
        """Return the stable compact snapshot mapping."""
        return {
            "case_digest": self.case_digest,
            "case_id": self.case_id,
            "comparison_key": self.comparison_key,
            "final_wheel_vs_original": self.final_wheel_vs_original,
            "native_layer_vs_source_only_wheel": self.native_layer_vs_source_only_wheel,
            "python_rewrite_vs_original": self.python_rewrite_vs_original,
            "status": self.status,
            "upstream_revision": self.upstream_revision,
        }


@dataclass(frozen=True, slots=True)
class CorpusSnapshot:
    """Manually reviewed compact history for one tier and platform.

    Raw timing samples are intentionally excluded.  ``comparison_key`` values
    retained by each case are the boundary for like-for-like historical claims.
    Instances are deterministic and safe to compare or serialize as JSON.
    """

    schema_version: int
    label: str
    reviewed_by: str
    tier: CorpusTier
    platform: CorpusPlatform
    manifest_digest: str
    atoll_revision: str | None
    experiment: ExperimentIdentity | None
    cases: tuple[SnapshotCase, ...]
    accelerated_coverage: float
    accepted_geometric_mean: float | None
    effective_corpus_speedup: float | None
    infrastructure_invalid_case_ids: tuple[str, ...]
    semantic_invalid_case_ids: tuple[str, ...]
    unmeasured_valid_case_ids: tuple[str, ...]

    def as_json(self) -> dict[str, object]:
        """Return canonical JSON-ready snapshot data."""
        return {
            "accepted_geometric_mean": self.accepted_geometric_mean,
            "accelerated_coverage": self.accelerated_coverage,
            "atoll_revision": self.atoll_revision,
            "cases": [case.as_json() for case in self.cases],
            "effective_corpus_speedup": self.effective_corpus_speedup,
            "experiment": None if self.experiment is None else self.experiment.as_json(),
            "infrastructure_invalid_case_ids": list(self.infrastructure_invalid_case_ids),
            "label": self.label,
            "manifest_digest": self.manifest_digest,
            "platform": self.platform,
            "reviewed_by": self.reviewed_by,
            "schema_version": self.schema_version,
            "semantic_invalid_case_ids": list(self.semantic_invalid_case_ids),
            "tier": self.tier,
            "unmeasured_valid_case_ids": list(self.unmeasured_valid_case_ids),
        }


def promote_results(
    manifest: CorpusManifest,
    result_paths: tuple[Path, ...],
    options: PromotionOptions,
) -> tuple[Path, Path]:
    """Promote a complete result slice and regenerate the docs history block.

    Existing snapshots are immutable.  Repeating an identical promotion is
    idempotent, while trying to reuse its label for different evidence fails.
    No checkout, wheel, report, or raw timing artifact is copied.

    Args:
        manifest: Manifest defining the exact expected matrix slice.
        result_paths: Complete set of schema-v1 case result files.
        options: Reviewed matrix identity and repository destinations.

    Returns:
        tuple[Path, Path]: Snapshot and regenerated documentation paths.

    Raises:
        HistoryError: If labels, reviewers, existing history, or docs markers
            cannot support a deterministic reviewed promotion.
        AggregationError: If the result slice is incomplete or invalidly formed.
    """
    normalized_label = _validate_label(options.label)
    reviewer = _validate_reviewer(options.reviewed_by)
    experiment = _load_experiment_identity(
        result_paths,
        normalized_label,
        required=options.tier == "performance",
    )
    results = load_case_results(result_paths)
    aggregate = aggregate_loaded_results(
        manifest,
        results,
        tier=options.tier,
        platform=options.platform,
    )
    by_id = {result.case_id: result for result in results}
    revisions = {
        result.environment.atoll_revision for result in results if result.environment is not None
    }
    invalid_revisions = sorted(
        revision for revision in revisions if _HEX_SHA.fullmatch(revision) is None
    )
    if invalid_revisions:
        raise HistoryError("promotion requires full lowercase Atoll revision SHAs")
    if len(revisions) > 1:
        raise HistoryError("promotion requires one Atoll revision across all case results")
    atoll_revision = next(iter(revisions), None)
    if (
        experiment is not None
        and atoll_revision is not None
        and experiment.head_sha != atoll_revision
    ):
        raise HistoryError("workflow head SHA does not match case Atoll revision")
    snapshot = CorpusSnapshot(
        schema_version=SNAPSHOT_SCHEMA_VERSION,
        label=normalized_label,
        reviewed_by=reviewer,
        tier=options.tier,
        platform=options.platform,
        manifest_digest=aggregate.manifest_digest,
        atoll_revision=atoll_revision,
        experiment=experiment,
        cases=tuple(
            SnapshotCase(
                case_id=case.case_id,
                status=by_id[case.case_id].status,
                upstream_revision=by_id[case.case_id].revision,
                case_digest=by_id[case.case_id].case_digest,
                comparison_key=by_id[case.case_id].comparison_key,
                python_rewrite_vs_original=(by_id[case.case_id].ratios.python_rewrite_vs_original),
                final_wheel_vs_original=(by_id[case.case_id].ratios.final_wheel_vs_original),
                native_layer_vs_source_only_wheel=(
                    by_id[case.case_id].ratios.native_vs_source_only
                ),
            )
            for case in aggregate.cases
        ),
        accelerated_coverage=aggregate.accelerated_coverage,
        accepted_geometric_mean=aggregate.accepted_geometric_mean,
        effective_corpus_speedup=aggregate.effective_corpus_speedup,
        infrastructure_invalid_case_ids=aggregate.infrastructure_invalid_case_ids,
        semantic_invalid_case_ids=aggregate.semantic_invalid_case_ids,
        unmeasured_valid_case_ids=aggregate.unmeasured_valid_case_ids,
    )
    _validate_docs(options.docs_path)
    history = _safe_directory(options.history_root)
    snapshot_path = history / f"{normalized_label}--{options.tier}--{options.platform}.json"
    content = _render_snapshot_json(snapshot)
    with _promotion_lock():
        _validate_docs(options.docs_path)
        _write_immutable(snapshot_path, content)
        snapshots = tuple(load_snapshot(path) for path in sorted(history.glob("*.json")))
        _update_docs(options.docs_path, render_history_markdown(snapshots))
    return snapshot_path, options.docs_path


def load_snapshot(path: Path) -> CorpusSnapshot:
    """Strictly load one manually promoted schema-v1 snapshot."""
    if path.is_symlink():
        raise HistoryError(f"history snapshot is a symlink: {path}")
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HistoryError(f"cannot read history snapshot {path}: {error}") from error
    payload = _mapping(raw, str(path))
    _exact_fields(payload, _SNAPSHOT_FIELDS, str(path))
    version = _integer(payload, "schema_version", str(path))
    if version != SNAPSHOT_SCHEMA_VERSION:
        raise HistoryError(
            f"{path}.schema_version must be {SNAPSHOT_SCHEMA_VERSION}, got {version}"
        )
    tier = _literal(payload, "tier", str(path), _TIERS)
    platform = _literal(payload, "platform", str(path), _PLATFORMS)
    raw_cases = _sequence(payload, "cases", str(path))
    cases = tuple(
        _parse_case(_mapping(raw_case, f"{path}.cases[{index}]"), f"{path}.cases[{index}]")
        for index, raw_case in enumerate(raw_cases)
    )
    if tuple(case.case_id for case in cases) != tuple(sorted(case.case_id for case in cases)):
        raise HistoryError(f"{path}.cases must be sorted by case_id")
    experiment = _optional_experiment(payload, "experiment", str(path))
    atoll_revision = _optional_sha(payload, "atoll_revision", str(path))
    if tier == "performance" and experiment is None:
        raise HistoryError(f"{path}.experiment is required for performance history")
    if (
        experiment is not None
        and atoll_revision is not None
        and experiment.head_sha != atoll_revision
    ):
        raise HistoryError(f"{path}.experiment head SHA does not match atoll_revision")
    return CorpusSnapshot(
        schema_version=version,
        label=_validate_label(_string(payload, "label", str(path))),
        reviewed_by=_validate_reviewer(_string(payload, "reviewed_by", str(path))),
        tier=cast(CorpusTier, tier),
        platform=cast(CorpusPlatform, platform),
        manifest_digest=_digest(payload, "manifest_digest", str(path), 64),
        atoll_revision=atoll_revision,
        experiment=experiment,
        cases=cases,
        accelerated_coverage=_bounded_number(
            payload, "accelerated_coverage", str(path), minimum=0.0, maximum=1.0
        ),
        accepted_geometric_mean=_optional_positive_number(
            payload, "accepted_geometric_mean", str(path)
        ),
        effective_corpus_speedup=_optional_positive_number(
            payload, "effective_corpus_speedup", str(path)
        ),
        infrastructure_invalid_case_ids=_sorted_strings(
            payload, "infrastructure_invalid_case_ids", str(path)
        ),
        semantic_invalid_case_ids=_sorted_strings(payload, "semantic_invalid_case_ids", str(path)),
        unmeasured_valid_case_ids=_sorted_strings(payload, "unmeasured_valid_case_ids", str(path)),
    )


def render_history_markdown(snapshots: tuple[CorpusSnapshot, ...]) -> str:
    """Render aggregate and per-case history without combining corpus groups.

    Every promoted case remains visible in the generated documentation.  The
    summary table supports cross-snapshot comparison, while the detail tables
    prevent a single accelerated case from obscuring no-op, rejected, or
    unmeasured members of the same corpus run.

    Args:
        snapshots: Reviewed snapshots to render.

    Returns:
        Deterministic Markdown containing snapshot summaries and every case.
    """
    ordered = tuple(sorted(snapshots, key=lambda item: (item.platform, item.tier, item.label)))
    if not ordered:
        return "No reviewed corpus snapshots have been promoted."
    lines = [
        "| Label | Group | Platform | Cases | Accelerated | Accepted only | Effective corpus |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        (
            "| "
            f"`{snapshot.label}` | `{snapshot.tier}` | `{snapshot.platform}` | "
            f"{len(snapshot.cases)} | {snapshot.accelerated_coverage:.1%} | "
            f"{_ratio(snapshot.accepted_geometric_mean)} | "
            f"{_ratio(snapshot.effective_corpus_speedup)} |"
        )
        for snapshot in ordered
    )
    lines.extend(("", "Snapshots are grouped by tier and platform; their ratios are never pooled."))
    for snapshot in ordered:
        lines.extend(
            (
                "",
                f"### `{snapshot.label}`: {snapshot.tier} on {snapshot.platform}",
                "",
                "| Case | Status | Python rewrite versus original | "
                "Final wheel versus original | Native layer versus source-only wheel |",
                "| --- | --- | ---: | ---: | ---: |",
            )
        )
        lines.extend(
            "| "
            f"`{case.case_id}` | `{case.status}` | "
            f"{_ratio(case.python_rewrite_vs_original)} | "
            f"{_ratio(case.final_wheel_vs_original)} | "
            f"{_ratio(case.native_layer_vs_source_only_wheel)} |"
            for case in snapshot.cases
        )
    return "\n".join(lines)


def _render_snapshot_json(snapshot: CorpusSnapshot) -> str:
    return f"{json.dumps(snapshot.as_json(), indent=2, sort_keys=True)}\n"


def _safe_directory(path: Path) -> Path:
    if path.is_symlink():
        raise HistoryError(f"history root is a symlink: {path}")
    path.mkdir(parents=True, exist_ok=True)
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise HistoryError(f"history root is not a directory: {path}")
    return resolved


@contextmanager
def _promotion_lock() -> Generator[None, None, None]:
    """Serialize immutable snapshot creation and generated-doc regeneration."""
    lock_root = Path(tempfile.gettempdir()) / f"atoll-corpus-promotion-{os.getuid()}"
    if lock_root.is_symlink():
        raise HistoryError(f"promotion lock root is a symlink: {lock_root}")
    try:
        lock_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if lock_root.is_symlink():
            raise HistoryError(f"promotion lock root is a symlink: {lock_root}")
        resolved_lock_root = lock_root.resolve(strict=True)
        lock_path = resolved_lock_root / "promotion.lock"
        if lock_path.is_symlink():
            raise HistoryError(f"promotion lock is a symlink: {lock_path}")
        with lock_path.open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    except OSError as error:
        raise HistoryError(f"cannot lock corpus history promotion: {error}") from error


def _write_immutable(path: Path, content: str) -> None:
    if path.is_symlink():
        raise HistoryError(f"history snapshot is a symlink: {path}")
    created = False
    try:
        stream = path.open("x", encoding="utf-8")
        created = True
        with stream:
            stream.write(content)
    except FileExistsError:
        if path.is_symlink():
            raise HistoryError(f"history snapshot is a symlink: {path}") from None
        try:
            existing = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise HistoryError(f"cannot read existing history snapshot: {path}") from error
        if existing != content:
            raise HistoryError(
                f"history snapshot already exists with different evidence: {path}"
            ) from None
        return
    except OSError as error:
        if created:
            path.unlink(missing_ok=True)
        raise HistoryError(f"cannot create immutable history snapshot: {path}") from error
    else:
        return


def _load_experiment_identity(
    result_paths: tuple[Path, ...],
    expected: str,
    *,
    required: bool,
) -> ExperimentIdentity | None:
    """Bind every promoted result to one immutable workflow run and attempt."""
    parents = tuple(sorted({path.parent for path in result_paths}))
    identities: list[ExperimentIdentity] = []
    missing: list[Path] = []
    for parent in parents:
        marker = parent / "experiment.json"
        if not marker.exists():
            missing.append(marker)
            continue
        if marker.is_symlink() or not marker.is_file():
            raise HistoryError(f"experiment identity is not a regular file: {marker}")
        try:
            if marker.stat().st_size > _MAX_EXPERIMENT_BYTES:
                raise HistoryError(f"experiment identity exceeds {_MAX_EXPERIMENT_BYTES} bytes")
            raw: object = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise HistoryError(f"cannot read experiment identity: {marker}") from error
        identity = _parse_experiment(_mapping(raw, str(marker)), str(marker))
        if identity.label != expected:
            raise HistoryError(
                f"experiment label {identity.label!r} does not match promotion label {expected!r}"
            )
        identities.append(identity)
    if missing and (required or identities):
        raise HistoryError(f"performance evidence is missing experiment identity: {missing[0]}")
    if not identities:
        return None
    first = identities[0]
    if any(identity != first for identity in identities[1:]):
        raise HistoryError("performance evidence contains mixed workflow run identities")
    return first


def _parse_experiment(payload: dict[str, object], label: str) -> ExperimentIdentity:
    _exact_fields(payload, _EXPERIMENT_FIELDS, label)
    version = _integer(payload, "schema_version", label)
    if version != EXPERIMENT_SCHEMA_VERSION:
        raise HistoryError(
            f"{label}.schema_version must be {EXPERIMENT_SCHEMA_VERSION}, got {version}"
        )
    experiment_label = _validate_label(_string(payload, "label", label))
    repository = _string(payload, "repository", label)
    if _REPOSITORY.fullmatch(repository) is None:
        raise HistoryError(f"{label}.repository must be an owner/repository identity")
    workflow_ref = _string(payload, "workflow_ref", label)
    required_prefix = f"{repository}/.github/workflows/"
    if (
        len(workflow_ref) > _MAX_WORKFLOW_REF_LENGTH
        or "\0" in workflow_ref
        or not workflow_ref.startswith(required_prefix)
        or not workflow_ref.endswith("@refs/heads/main")
    ):
        raise HistoryError(f"{label}.workflow_ref must identify a default-branch workflow")
    return ExperimentIdentity(
        schema_version=version,
        label=experiment_label,
        repository=repository,
        workflow_ref=workflow_ref,
        run_id=_positive_integer(payload, "run_id", label),
        run_attempt=_positive_integer(payload, "run_attempt", label),
        head_sha=_digest(payload, "head_sha", label, 40),
    )


def _optional_experiment(
    payload: dict[str, object], key: str, label: str
) -> ExperimentIdentity | None:
    value = payload.get(key)
    if value is None:
        return None
    return _parse_experiment(_mapping(value, f"{label}.{key}"), f"{label}.{key}")


def _update_docs(path: Path, history: str) -> None:
    content = _validate_docs(path)
    prefix, remainder = content.split(HISTORY_START, maxsplit=1)
    _old, suffix = remainder.split(HISTORY_END, maxsplit=1)
    replacement = f"{prefix}{HISTORY_START}\n{history}\n{HISTORY_END}{suffix}"
    _atomic_write(path, replacement)


def _validate_docs(path: Path) -> str:
    """Read the authored benchmark page and require one generated block."""
    if path.is_symlink():
        raise HistoryError(f"benchmark documentation is a symlink: {path}")
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise HistoryError(f"cannot read benchmark documentation: {path}") from error
    if content.count(HISTORY_START) != 1 or content.count(HISTORY_END) != 1:
        raise HistoryError("benchmark documentation must contain exactly one history marker pair")
    if content.index(HISTORY_START) > content.index(HISTORY_END):
        raise HistoryError("benchmark documentation history markers are reversed")
    return content


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.is_symlink():
        raise HistoryError(f"temporary promotion path is a symlink: {temporary}")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise HistoryError(f"cannot write promoted history: {path}") from error


def _parse_case(payload: dict[str, object], label: str) -> SnapshotCase:
    _exact_fields(payload, _CASE_FIELDS, label)
    status = _literal(payload, "status", label, _STATUSES)
    return SnapshotCase(
        case_id=_string(payload, "case_id", label),
        status=cast(CaseStatus, status),
        upstream_revision=_digest(payload, "upstream_revision", label, 40),
        case_digest=_digest(payload, "case_digest", label, 64),
        comparison_key=_optional_string(payload, "comparison_key", label),
        python_rewrite_vs_original=_optional_positive_number(
            payload, "python_rewrite_vs_original", label
        ),
        final_wheel_vs_original=_optional_positive_number(
            payload, "final_wheel_vs_original", label
        ),
        native_layer_vs_source_only_wheel=_optional_positive_number(
            payload, "native_layer_vs_source_only_wheel", label
        ),
    )


def _validate_label(value: str) -> str:
    if _LABEL.fullmatch(value) is None:
        raise HistoryError(
            "promotion label must be 1-64 lowercase letters, digits, dots, dashes, or underscores"
        )
    return value


def _validate_reviewer(value: str) -> str:
    reviewer = value.strip()
    if (
        not reviewer
        or len(reviewer) > _MAX_REVIEWER_LENGTH
        or any(character in reviewer for character in "\r\n")
    ):
        raise HistoryError("reviewed-by must be one non-empty line of at most 128 characters")
    return reviewer


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise HistoryError(f"{label} must be a JSON object")
    raw = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in raw):
        raise HistoryError(f"{label} must have only string keys")
    return cast(dict[str, object], value)


def _exact_fields(payload: dict[str, object], expected: frozenset[str], label: str) -> None:
    missing = sorted(expected - set(payload))
    unknown = sorted(set(payload) - expected)
    if missing:
        raise HistoryError(f"{label} is missing field(s): {', '.join(missing)}")
    if unknown:
        raise HistoryError(f"{label} contains unknown field(s): {', '.join(unknown)}")


def _string(payload: dict[str, object], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise HistoryError(f"{label}.{key} must be a non-empty string")
    return value


def _optional_string(payload: dict[str, object], key: str, label: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise HistoryError(f"{label}.{key} must be null or a non-empty string")
    return value


def _integer(payload: dict[str, object], key: str, label: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise HistoryError(f"{label}.{key} must be an integer")
    return value


def _positive_integer(payload: dict[str, object], key: str, label: str) -> int:
    value = _integer(payload, key, label)
    if value <= 0:
        raise HistoryError(f"{label}.{key} must be positive")
    return value


def _literal(payload: dict[str, object], key: str, label: str, allowed: tuple[str, ...]) -> str:
    value = _string(payload, key, label)
    if value not in allowed:
        raise HistoryError(f"{label}.{key} has unsupported value {value!r}")
    return value


def _sequence(payload: dict[str, object], key: str, label: str) -> list[object]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise HistoryError(f"{label}.{key} must be an array")
    return cast(list[object], value)


def _sorted_strings(payload: dict[str, object], key: str, label: str) -> tuple[str, ...]:
    values = _sequence(payload, key, label)
    if any(not isinstance(value, str) or not value for value in values):
        raise HistoryError(f"{label}.{key} must contain non-empty strings")
    parsed = cast(tuple[str, ...], tuple(values))
    if parsed != tuple(sorted(set(parsed))):
        raise HistoryError(f"{label}.{key} must be sorted and unique")
    return parsed


def _number(payload: dict[str, object], key: str, label: str) -> float:
    value = payload.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise HistoryError(f"{label}.{key} must be a number")
    number = float(value)
    if not number > 0.0 or number == float("inf"):
        raise HistoryError(f"{label}.{key} must be a finite positive number")
    return number


def _optional_positive_number(payload: dict[str, object], key: str, label: str) -> float | None:
    return None if payload.get(key) is None else _number(payload, key, label)


def _bounded_number(
    payload: dict[str, object],
    key: str,
    label: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    value = payload.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise HistoryError(f"{label}.{key} must be a number")
    number = float(value)
    if not minimum <= number <= maximum:
        raise HistoryError(f"{label}.{key} must be between {minimum} and {maximum}")
    return number


def _digest(payload: dict[str, object], key: str, label: str, length: int) -> str:
    value = _string(payload, key, label)
    if len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise HistoryError(f"{label}.{key} must be a {length}-character lowercase hex digest")
    return value


def _optional_sha(payload: dict[str, object], key: str, label: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or _HEX_SHA.fullmatch(value) is None:
        raise HistoryError(f"{label}.{key} must be null or a full lowercase Git SHA")
    return value


def _ratio(value: float | None) -> str:
    return "not measured" if value is None else f"{value:.3f}x"
