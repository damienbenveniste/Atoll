"""Tests for strict, platform-isolated corpus aggregation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path, PurePosixPath
from typing import cast

import pytest
from scripts.benchmark_corpus.aggregation import (
    AggregationError,
    aggregate_case_results,
    compare_aggregates,
    load_case_result,
    render_aggregate_json,
    render_aggregate_markdown,
)
from scripts.benchmark_corpus.cli import main as corpus_main
from scripts.benchmark_corpus.identity import case_digest, comparison_key
from scripts.benchmark_corpus.models import (
    CaseResult,
    CaseStatus,
    CorpusCase,
    CorpusDefaults,
    CorpusManifest,
    CorpusPlatform,
    CorpusTier,
    EnvironmentEvidence,
    PolicyEvidence,
    RatioEvidence,
)

_REVISION = "a" * 40
_EXPECTED_ACCEPTED_MEAN = 6.0


def test_aggregation_is_stable_when_result_files_are_shuffled(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path,
        (
            "accepted-a",
            "accepted-b",
            "noop",
            "unsupported",
            "unprofitable",
            "infra",
            "semantic",
        ),
    )
    specifications: tuple[tuple[str, CaseStatus, float | None], ...] = (
        ("accepted-a", "accelerated", 4.0),
        ("accepted-b", "accelerated", 9.0),
        ("noop", "supported-no-op", None),
        ("unsupported", "unsupported", None),
        ("unprofitable", "not-profitable", None),
        ("infra", "infrastructure-error", None),
        ("semantic", "compatibility-regression", None),
    )
    paths = tuple(
        _write_result(
            tmp_path / f"{case_id}.json",
            _result(manifest, case_id, status=status, speedup=speedup),
        )
        for case_id, status, speedup in specifications
    )

    aggregate = aggregate_case_results(
        manifest,
        tuple(reversed(paths)),
        tier="performance",
        platform="ubuntu-24.04",
    )
    ordered = aggregate_case_results(
        manifest,
        paths,
        tier="performance",
        platform="ubuntu-24.04",
    )

    assert aggregate == ordered
    assert tuple(case.case_id for case in aggregate.cases) == tuple(
        sorted(manifest_case.id for manifest_case in manifest.cases)
    )
    assert aggregate.accelerated_coverage == pytest.approx(2 / 7)
    assert aggregate.accepted_geometric_mean == pytest.approx(_EXPECTED_ACCEPTED_MEAN)
    assert aggregate.effective_corpus_speedup == pytest.approx(math.pow(36.0, 1 / 5))
    assert aggregate.infrastructure_invalid_case_ids == ("infra",)
    assert aggregate.semantic_invalid_case_ids == ("semantic",)
    assert aggregate.unmeasured_valid_case_ids == ()
    assert render_aggregate_json(aggregate) == render_aggregate_json(ordered)
    assert render_aggregate_markdown(aggregate) == render_aggregate_markdown(ordered)
    assert '"platform": "ubuntu-24.04"' in render_aggregate_json(aggregate)
    assert "Accepted-only geometric mean: 6.000x" in render_aggregate_markdown(aggregate)


def test_compiled_unbenchmarked_is_valid_but_not_fabricated_as_neutral(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path, ("compiled",))
    path = _write_result(
        tmp_path / "compiled.json",
        _result(manifest, "compiled", status="compiled-unbenchmarked"),
    )

    aggregate = aggregate_case_results(
        manifest, (path,), tier="performance", platform="ubuntu-24.04"
    )

    assert aggregate.effective_corpus_speedup is None
    assert aggregate.accepted_geometric_mean is None
    assert aggregate.unmeasured_valid_case_ids == ("compiled",)


def test_aggregate_cli_writes_one_platform_and_tier_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = _manifest(tmp_path, ("alpha", "beta"))
    results = tmp_path / "results"
    for case_id in ("alpha", "beta"):
        directory = results / case_id
        directory.mkdir(parents=True)
        _write_result(
            directory / "case-result.json",
            _result(manifest, case_id, status="supported-no-op"),
        )
    output = tmp_path / "aggregate"

    def load_selected_manifest(_path: Path) -> CorpusManifest:
        return manifest

    monkeypatch.setattr(
        "scripts.benchmark_corpus.cli.load_manifest",
        load_selected_manifest,
    )

    exit_code = corpus_main(
        (
            "--manifest",
            str(manifest.path),
            "aggregate",
            "--tier",
            "performance",
            "--platform",
            "ubuntu-24.04",
            "--results-root",
            str(results),
            "--output-root",
            str(output),
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["tier"] == "performance"
    assert payload["platform"] == "ubuntu-24.04"
    assert (output / "aggregate-performance-ubuntu-24.04.json").is_file()
    assert (output / "aggregate-performance-ubuntu-24.04.md").is_file()


def test_aggregate_cli_writes_invalid_summary_and_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(tmp_path, ("alpha",))
    results = tmp_path / "results"
    results.mkdir()
    _write_result(
        results / "case-result.json",
        _result(manifest, "alpha", status="compatibility-regression"),
    )

    def load_selected_manifest(_path: Path) -> CorpusManifest:
        return manifest

    monkeypatch.setattr(
        "scripts.benchmark_corpus.cli.load_manifest",
        load_selected_manifest,
    )

    exit_code = corpus_main(
        (
            "--manifest",
            str(manifest.path),
            "aggregate",
            "--tier",
            "performance",
            "--platform",
            "ubuntu-24.04",
            "--results-root",
            str(results),
        )
    )

    assert exit_code == 1
    assert (results / "aggregate-performance-ubuntu-24.04.json").is_file()


def test_aggregation_rejects_missing_expected_matrix_case(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, ("alpha", "beta"))
    alpha = _write_result(tmp_path / "alpha.json", _result(manifest, "alpha"))

    with pytest.raises(AggregationError, match=r"missing case result\(s\): beta"):
        aggregate_case_results(manifest, (alpha,), tier="performance", platform="ubuntu-24.04")


def test_aggregation_rejects_empty_matrix_slice(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, ("alpha",))

    with pytest.raises(AggregationError, match="no expected cases"):
        aggregate_case_results(
            manifest,
            (),
            tier="calibration",
            platform="ubuntu-24.04",
        )


def test_aggregation_rejects_duplicate_matrix_identity(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, ("alpha",))
    alpha = _write_result(tmp_path / "alpha.json", _result(manifest, "alpha"))

    with pytest.raises(AggregationError, match=r"duplicate case result\(s\): alpha"):
        aggregate_case_results(
            manifest, (alpha, alpha), tier="performance", platform="ubuntu-24.04"
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("schema", "schema_version"),
        ("missing", "ratios"),
        ("missing-nullable", "baseline_wheel_digest"),
        ("unknown", "unknown field"),
        ("wrong-type", "diagnostics must be an array"),
    ],
)
def test_load_case_result_rejects_malformed_or_unknown_schema_payloads(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    manifest = _manifest(tmp_path, ("alpha",))
    payload = _json_payload(_result(manifest, "alpha"))
    if mutation == "schema":
        payload["schema_version"] = 2
    elif mutation == "missing":
        payload.pop("ratios")
    elif mutation == "missing-nullable":
        payload.pop("baseline_wheel_digest")
    elif mutation == "unknown":
        payload["unexpected"] = True
    else:
        payload["diagnostics"] = "not-an-array"
    path = tmp_path / "malformed.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AggregationError, match=message):
        load_case_result(path)


def test_load_case_result_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "case-result.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(AggregationError, match="cannot read case result"):
        load_case_result(path)


@pytest.mark.parametrize(
    ("tier", "platform", "message"),
    [
        ("compatibility", "ubuntu-24.04", "has tier compatibility"),
        ("performance", "macos-14", "has platform macos-14"),
    ],
)
def test_aggregation_rejects_tier_or_platform_incompatible_result(
    tmp_path: Path,
    tier: CorpusTier,
    platform: CorpusPlatform,
    message: str,
) -> None:
    manifest = _manifest(tmp_path, ("alpha",))
    incompatible = replace(
        _result(manifest, "alpha"),
        tier=tier,
        platform=platform,
    )
    path = _write_result(tmp_path / "incompatible.json", incompatible)

    with pytest.raises(AggregationError, match=message):
        aggregate_case_results(manifest, (path,), tier="performance", platform="ubuntu-24.04")


def test_aggregation_rejects_unknown_case_and_stale_manifest_identity(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, ("alpha",))
    unknown = replace(_result(manifest, "alpha"), case_id="unknown")
    unknown_path = _write_result(tmp_path / "unknown.json", unknown)

    with pytest.raises(AggregationError, match="not in the selected"):
        aggregate_case_results(
            manifest, (unknown_path,), tier="performance", platform="ubuntu-24.04"
        )

    stale = replace(_result(manifest, "alpha"), manifest_digest="0" * 64)
    stale_path = _write_result(tmp_path / "stale.json", stale)
    with pytest.raises(AggregationError, match="stale manifest digest"):
        aggregate_case_results(manifest, (stale_path,), tier="performance", platform="ubuntu-24.04")


def test_accelerated_result_requires_an_accepted_end_to_end_ratio(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, ("alpha",))
    path = _write_result(
        tmp_path / "alpha.json",
        _result(manifest, "alpha", status="accelerated", speedup=None),
    )

    with pytest.raises(AggregationError, match="has no final-wheel"):
        aggregate_case_results(manifest, (path,), tier="performance", platform="ubuntu-24.04")


def test_historical_comparison_rejects_unlike_keys_and_platforms(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, ("alpha",))
    current_path = _write_result(
        tmp_path / "current.json",
        _result(manifest, "alpha", status="accelerated", speedup=2.0),
    )
    current = aggregate_case_results(
        manifest, (current_path,), tier="performance", platform="ubuntu-24.04"
    )
    unlike_path = _write_result(
        tmp_path / "unlike.json",
        _result(
            manifest,
            "alpha",
            status="accelerated",
            speedup=1.5,
            hardware_class="different-hardware",
        ),
    )
    unlike = aggregate_case_results(
        manifest, (unlike_path,), tier="performance", platform="ubuntu-24.04"
    )

    with pytest.raises(AggregationError, match="comparison key"):
        compare_aggregates(current, unlike)

    other_platform = replace(unlike, platform="macos-14")
    with pytest.raises(AggregationError, match="same tier and platform"):
        compare_aggregates(current, other_platform)


def test_historical_comparison_uses_only_matching_keys(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, ("alpha",))
    current_path = _write_result(
        tmp_path / "current.json",
        _result(manifest, "alpha", status="accelerated", speedup=3.0),
    )
    historical_path = _write_result(
        tmp_path / "historical.json",
        _result(manifest, "alpha", status="accelerated", speedup=2.0),
    )
    current = aggregate_case_results(
        manifest, (current_path,), tier="performance", platform="ubuntu-24.04"
    )
    historical = aggregate_case_results(
        manifest, (historical_path,), tier="performance", platform="ubuntu-24.04"
    )

    comparison = compare_aggregates(current, historical)

    assert len(comparison) == 1
    assert comparison[0].current_vs_historical == pytest.approx(1.5)


def test_aggregation_recomputes_comparison_key(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, ("alpha",))
    forged = replace(_result(manifest, "alpha"), comparison_key="forged")
    path = _write_result(tmp_path / "forged.json", forged)

    with pytest.raises(AggregationError, match="unverifiable comparison key"):
        aggregate_case_results(
            manifest,
            (path,),
            tier="performance",
            platform="ubuntu-24.04",
        )


def test_comparison_key_excludes_atoll_revision_but_includes_hardware(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, ("alpha",))
    result = _result(manifest, "alpha")
    assert result.environment is not None
    assert result.policy is not None
    case = manifest.cases[0]
    changed_atoll = replace(result.environment, atoll_revision="0" * 40)
    changed_hardware = replace(result.environment, hardware_class="other-hardware")

    assert comparison_key(case, changed_atoll, result.policy) == result.comparison_key
    assert comparison_key(case, changed_hardware, result.policy) != result.comparison_key


def _manifest(tmp_path: Path, case_ids: tuple[str, ...]) -> CorpusManifest:
    path = tmp_path / "manifest.toml"
    path.write_text("schema_version = 1\n", encoding="utf-8")
    cases = tuple(
        CorpusCase(
            id=case_id,
            name=case_id,
            repository=f"https://example.invalid/{case_id}.git",
            revision=_REVISION,
            project_subroot=PurePosixPath("."),
            dependency_lock=PurePosixPath(f"locks/{case_id}.txt"),
            focused_test_command=("python", "-m", "pytest"),
            oracle_adapter="compatibility",
            oracle_arguments=(),
            tiers=("performance",),
            platforms=("ubuntu-24.04",),
        )
        for case_id in sorted(case_ids)
    )
    return CorpusManifest(
        path=path,
        schema_version=1,
        python_version="3.12",
        backends=("mypyc", "cython"),
        defaults=CorpusDefaults(
            test_timeout_seconds=1,
            compile_timeout_seconds=1,
            performance_timeout_seconds=1,
            max_log_bytes=1,
        ),
        cases=cases,
    )


def _result(
    manifest: CorpusManifest,
    case_id: str,
    *,
    status: CaseStatus = "supported-no-op",
    speedup: float | None = None,
    hardware_class: str = "same-hardware",
) -> CaseResult:
    case = next(item for item in manifest.cases if item.id == case_id)
    environment = EnvironmentEvidence(
        python="cpython 3.12.11",
        atoll_revision="f" * 40,
        uv="uv 0.8.0",
        mypy="1.18.2",
        cython="3.1.4",
        compiler="clang 17",
        operating_system="Linux 6.8",
        architecture="x86_64",
        runner_image="ubuntu-24.04",
        hardware_class=hardware_class,
        dependency_lock_digest="d" * 64,
    )
    policy = PolicyEvidence(
        digest="e" * 64,
        patch_path=PurePosixPath("compile-policy.patch"),
        source_path=PurePosixPath("pyproject.toml"),
    )
    return CaseResult(
        schema_version=1,
        case_id=case.id,
        tier="performance",
        platform="ubuntu-24.04",
        status=status,
        repository=case.repository,
        revision=case.revision,
        manifest_digest=hashlib.sha256(manifest.path.read_bytes()).hexdigest(),
        case_digest=case_digest(case),
        comparison_key=comparison_key(case, environment, policy),
        diagnostics=(),
        source_digest_before=None,
        source_digest_after=None,
        source_unchanged=None,
        policy=policy,
        environment=environment,
        phases=(),
        baseline_wheel_digest=None,
        compiled_wheel_digest=None,
        cold_report_path=None,
        warm_report_path=None,
        baseline_oracle_digest=None,
        compiled_oracle_digest=None,
        cold_compiler_invocations=None,
        warm_compiler_invocations=None,
        ratios=RatioEvidence(final_wheel_vs_original=speedup),
    )


def _write_result(path: Path, result: CaseResult) -> Path:
    path.write_text(json.dumps(_json_payload(result)), encoding="utf-8")
    return path


def _json_payload(result: CaseResult) -> dict[str, object]:
    raw: object = json.loads(json.dumps(asdict(result), default=_json_default))
    serialized = cast(dict[object, object], raw)
    assert isinstance(serialized, dict)
    assert all(isinstance(key, str) for key in serialized)
    return cast(dict[str, object], serialized)


def _json_default(value: object) -> str:
    if isinstance(value, PurePosixPath):
        return value.as_posix()
    raise TypeError(type(value).__name__)
