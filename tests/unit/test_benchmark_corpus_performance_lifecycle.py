"""Focused contracts for measured corpus compile outcomes and evidence."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import cast

import pytest
from scripts.benchmark_corpus.lifecycle import (
    LifecycleError,
    build_performance_compile_policy,
    calibrated_benchmark_repetitions,
    classify_compile_process,
    performance_asset_digest,
    ratio_evidence_from_report,
    validate_performance_assets,
)
from scripts.benchmark_corpus.models import CorpusCase, WorkloadProvenance
from scripts.benchmark_corpus.process import ProcessResult

_EXPECTED_SAMPLE_COUNT = 7


def test_performance_policy_uses_reviewed_adapters_and_isolated_python(tmp_path: Path) -> None:
    tools_python = tmp_path / "tools" / "bin" / "python"
    project_root = tmp_path / "workspace" / "checkout"
    semantic_adapter = tmp_path / "adapters" / "compatibility.py"
    performance_adapter = tmp_path / "adapters" / "fixture_case.py"

    policy = build_performance_compile_policy(
        backends=("mypyc", "cython"),
        tools_python=tools_python,
        adapters=(semantic_adapter, performance_adapter),
        project_root=project_root,
        oracle_arguments=("--case", "fixture-case"),
    )

    assert policy.test_command == (
        str(tools_python),
        str(semantic_adapter),
        "--project-root",
        str(project_root),
        "--case",
        "fixture-case",
    )
    assert policy.benchmark_command == (
        str(tools_python),
        str(performance_adapter),
        "--project-root",
        str(project_root),
        "--repetitions",
        "1",
    )
    assert policy.benchmark_warmups == 1
    assert policy.benchmark_samples == _EXPECTED_SAMPLE_COUNT
    assert policy.minimum_speedup == pytest.approx(1.10)


@pytest.mark.parametrize(
    ("current", "duration_seconds", "expected"),
    [
        (1, 0.50, 1),
        (1, 0.24, 3),
        (2, 0.40, 4),
        (1, 0.001, 128),
        (128, 0.10, 128),
    ],
)
def test_benchmark_calibration_scales_above_the_noise_floor(
    current: int,
    duration_seconds: float,
    expected: int,
) -> None:
    assert calibrated_benchmark_repetitions(current, duration_seconds) == expected


@pytest.mark.parametrize(("current", "duration_seconds"), [(0, 0.5), (1, 0.0), (1, float("inf"))])
def test_benchmark_calibration_rejects_invalid_measurements(
    current: int,
    duration_seconds: float,
) -> None:
    with pytest.raises(ValueError, match=r"positive|finite"):
        calibrated_benchmark_repetitions(current, duration_seconds)


def test_performance_assets_are_hash_checked_at_runtime(tmp_path: Path) -> None:
    workload = tmp_path / "benchmarks" / "corpus" / "workloads" / "fixture_case.py"
    notice = tmp_path / "benchmarks" / "corpus" / "notices" / "fixture-case.txt"
    adapter_root = tmp_path / "benchmarks" / "corpus" / "adapters"
    for path, contents in (
        (workload, "VALUE = 1\n"),
        (notice, "Fixture notice\n"),
        (adapter_root / "fixture_case.py", "# adapter\n"),
        (adapter_root / "_performance.py", "# shared runner\n"),
        (workload.parent / "golden.json", "{}\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    case = _performance_case("0" * 64)
    assert case.workload is not None
    digest = performance_asset_digest(tmp_path, case, adapter_root)
    case = replace(case, workload=replace(case.workload, sha256=digest))

    validate_performance_assets(tmp_path, case, adapter_root)
    workload.write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(LifecycleError, match="digest does not match") as error:
        validate_performance_assets(tmp_path, case, adapter_root)

    assert error.value.status == "security-violation"


@pytest.mark.parametrize(
    ("exit_code", "status", "reason", "expected"),
    [
        (0, "passed", "threshold met", "accelerated"),
        (1, "not-profitable", "below threshold", "not-profitable"),
        (1, "invalid", "benchmark medians are too noisy", "unstable"),
    ],
)
def test_performance_compile_status_preserves_report_backed_outcomes(
    tmp_path: Path,
    exit_code: int,
    status: str,
    reason: str,
    expected: str,
) -> None:
    report = _performance_report(tmp_path, status=status, reason=reason)
    report["success"] = exit_code == 0

    observed = classify_compile_process(
        _process(exit_code),
        report,
        "performance",
        "cold",
        tmp_path,
    )

    assert observed == expected


def test_compiled_semantic_failure_is_a_compatibility_regression() -> None:
    report: dict[str, object] = {
        "version": 6,
        "success": False,
        "build": {},
        "performance": {
            "status": "invalid",
            "reason": "compiled semantic test command failed",
        },
    }

    with pytest.raises(LifecycleError, match="compiled semantic") as error:
        classify_compile_process(_process(1), report, "performance", "warm", Path.cwd())

    assert error.value.status == "compatibility-regression"


def test_rejected_compile_does_not_accept_a_crashed_process(tmp_path: Path) -> None:
    report = _performance_report(tmp_path, status="not-profitable", reason="below threshold")

    with pytest.raises(LifecycleError, match="inconsistent process") as error:
        classify_compile_process(_process(137), report, "performance", "warm", tmp_path)

    assert error.value.status == "compile-error"


def test_performance_samples_must_match_and_import_from_each_payload(tmp_path: Path) -> None:
    report = _performance_report(tmp_path, status="passed", reason="threshold met")
    performance = cast(dict[str, object], report["performance"])
    samples = cast(list[dict[str, object]], performance["samples"])
    compiled = next(sample for sample in samples if sample["mode"] == "compiled")
    compiled["stdout"] = json.dumps(
        {
            "canonical": {"checksum": 99},
            "imports": [str(tmp_path / "compiled" / "package.py")],
        }
    )

    with pytest.raises(LifecycleError, match="different canonical") as error:
        classify_compile_process(_process(0), report, "performance", "warm", tmp_path)

    assert error.value.status == "compatibility-regression"


def test_passed_performance_report_requires_canonical_output_and_speedup(
    tmp_path: Path,
) -> None:
    report = _performance_report(tmp_path, status="passed", reason="threshold met")
    performance = cast(dict[str, object], report["performance"])
    samples = cast(list[dict[str, object]], performance["samples"])
    first = samples[0]
    first["stdout"] = json.dumps({"imports": [str(tmp_path / "baseline" / "package.py")]})

    with pytest.raises(LifecycleError, match="canonical object"):
        classify_compile_process(_process(0), report, "performance", "warm", tmp_path)

    first["stdout"] = json.dumps(
        {
            "canonical": {"checksum": 42},
            "imports": [str(tmp_path / "baseline" / "package.py")],
        }
    )
    performance["speedup"] = None
    with pytest.raises(LifecycleError, match="speedup is inconsistent"):
        classify_compile_process(_process(0), report, "performance", "warm", tmp_path)


def test_ratio_evidence_labels_composed_source_and_native_layers() -> None:
    report: dict[str, object] = {
        "performance": {
            "speedup": 3.0,
            "samples": [
                {"mode": "baseline", "success": True, "duration_seconds": 0.9},
                {"mode": "compiled", "success": True, "duration_seconds": 0.3},
                {"mode": "baseline", "success": False, "duration_seconds": 99.0},
            ],
        },
        "final_composition": {
            "source_plan_ids": ["source-plan"],
            "native_variant_ids": ["native-variant"],
        },
        "source_optimization": {
            "trials": [
                {
                    "plan_id": "source-plan",
                    "status": "accepted",
                    "source_speedup": 2.0,
                }
            ]
        },
    }

    evidence = ratio_evidence_from_report(report)

    assert evidence.python_rewrite_vs_original == pytest.approx(2.0)
    assert evidence.final_wheel_vs_original == pytest.approx(3.0)
    assert evidence.native_vs_source_only == pytest.approx(1.5)
    assert evidence.baseline_samples_seconds == (0.9,)
    assert evidence.source_only_samples_seconds == ()
    assert evidence.final_wheel_samples_seconds == (0.3,)


def _process(exit_code: int) -> ProcessResult:
    return ProcessResult(
        exit_code=exit_code,
        timed_out=False,
        duration_seconds=1.0,
        log_truncated=False,
        argv=("python", "-m", "atoll", "compile"),
    )


def _performance_case(workload_digest: str) -> CorpusCase:
    return CorpusCase(
        id="fixture-case",
        name="Fixture",
        repository="https://example.invalid/fixture.git",
        revision="a" * 40,
        project_subroot=PurePosixPath("."),
        dependency_lock=PurePosixPath("locks/fixture.txt"),
        focused_test_command=("python", "-m", "pytest"),
        oracle_adapter="compatibility",
        oracle_arguments=("--case", "fixture-case"),
        tiers=("performance",),
        platforms=("ubuntu-24.04",),
        workload=WorkloadProvenance(
            source="atoll",
            repository="https://example.invalid/atoll.git",
            revision="b" * 40,
            path=PurePosixPath("benchmarks/corpus/workloads/fixture_case.py"),
            sha256=workload_digest,
            notice=PurePosixPath("benchmarks/corpus/notices/fixture-case.txt"),
        ),
    )


def _performance_report(
    project_root: Path,
    *,
    status: str,
    reason: str,
) -> dict[str, object]:
    samples: list[dict[str, object]] = []
    for mode in ("baseline", "compiled"):
        payload_root = project_root / mode
        stdout = json.dumps(
            {
                "canonical": {"checksum": 42},
                "imports": [str(payload_root / "package.py")],
            }
        )
        samples.extend(
            {
                "mode": mode,
                "success": True,
                "returncode": 0,
                "duration_seconds": 0.5,
                "payload_root": str(payload_root),
                "stdout": stdout,
            }
            for _ in range(_EXPECTED_SAMPLE_COUNT)
        )
    baseline_median = 0.1 if status == "invalid" else 1.0
    compiled_median = 0.1 if status == "invalid" else (1.0 if status == "not-profitable" else 0.5)
    speedup = None if status == "invalid" else baseline_median / compiled_median
    return {
        "version": 6,
        "success": status == "passed",
        "build": {},
        "performance": {
            "status": status,
            "reason": reason,
            "minimum_speedup": 1.1,
            "baseline_median_seconds": baseline_median,
            "compiled_median_seconds": compiled_median,
            "speedup": speedup,
            "samples": samples,
        },
    }
