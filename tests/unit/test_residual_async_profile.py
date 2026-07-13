"""Tests for the generic residual async profile fixture and benchmark."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Protocol, cast

import pytest
from scripts import run_residual_async_profile_benchmark as benchmark

FIXTURE_ROOT = Path("tests/fixtures/residual_async_profile")
SOURCE_ROOT = FIXTURE_ROOT / "src"
EXPECTED_CHECKSUM = 266732
EXPECTED_DOUBLE_CHECKSUM = EXPECTED_CHECKSUM * 2
EXPECTED_FINAL_SPEEDUP = 3.0
EXPECTED_HEADROOM_ITERATIONS = 4


class FixtureModule(Protocol):
    """Loaded residual profile fixture surface used by these tests."""

    STAGE_NAMES: tuple[str, ...]

    def compare_semantics(
        self,
        repetitions: int,
    ) -> Coroutine[object, object, tuple[dict[str, object], bool]]:
        """Return an awaitable semantic comparison result."""
        ...

    def context_sensitive_fallback_snapshot(
        self,
    ) -> Coroutine[object, object, dict[str, object]]:
        """Return an awaitable fallback snapshot."""
        ...

    def residual_checksum(
        self,
        iterations: int,
    ) -> Coroutine[object, object, tuple[int, StageCountersView]]:
        """Return an awaitable checksum and counter tuple."""
        ...


class StageCountersView(Protocol):
    """Counter methods exposed by the dynamically loaded fixture."""

    def all_nonzero(self) -> bool:
        """Return whether every residual transformation executed."""
        ...

    def as_json(self) -> dict[str, int]:
        """Return residual counters keyed by stable stage name."""
        ...


def test_residual_semantics_match_baseline_repeatedly() -> None:
    module = _fixture_module()
    snapshot, matched = asyncio.run(module.compare_semantics(4))

    assert matched
    assert snapshot == {
        "workflow": "generic-residual-async-profile",
        "work_count": 192,
        "checksum": EXPECTED_CHECKSUM,
        "completed": 192,
        "first_label": "item-0000",
        "last_label": "item-0191",
        "parent_context": "owner",
        "fallback_context": "owner",
        "fallback_label": "item-0096",
        "stage_counter_total": 0,
    }


def test_context_sensitive_work_uses_fallback_and_preserves_parent_context() -> None:
    module = _fixture_module()
    snapshot = asyncio.run(module.context_sensitive_fallback_snapshot())

    assert snapshot == {
        "parent_before": "owner",
        "child_observed": "owner",
        "child_mutated": "child:context-sensitive",
        "parent_after": "owner",
    }


def test_residual_stage_counters_are_all_nonzero() -> None:
    module = _fixture_module()
    checksum, counters = asyncio.run(module.residual_checksum(2))

    assert checksum == EXPECTED_DOUBLE_CHECKSUM
    assert counters.all_nonzero()
    assert set(counters.as_json()) == set(module.STAGE_NAMES)


def test_benchmark_policy_rejects_weakened_sampling_or_speed_floor() -> None:
    with pytest.raises(benchmark.BenchmarkError, match="exactly seven samples"):
        benchmark.run_benchmark(benchmark.BenchmarkOptions(samples=6))

    with pytest.raises(benchmark.BenchmarkError, match=r"at least 0\.25"):
        benchmark.run_benchmark(benchmark.BenchmarkOptions(minimum_seconds=0.249))

    with pytest.raises(benchmark.BenchmarkError, match=r"at least 3\.0x"):
        benchmark.run_benchmark(benchmark.BenchmarkOptions(minimum_speedup=2.99))


def test_calibration_requires_headroom_above_stability_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One noisy sample just above 0.25 seconds cannot end calibration."""

    def fake_time_arm(
        _arm: benchmark.ArmName,
        iterations: int,
    ) -> tuple[float, dict[str, int] | None]:
        return (0.26 if iterations == benchmark.DEFAULT_ITERATIONS else 0.52), None

    monkeypatch.setattr(benchmark, "_time_arm", fake_time_arm)

    calibrate = cast(
        Callable[[benchmark.BenchmarkOptions], int],
        vars(benchmark)["_calibrate_iterations"],
    )
    assert calibrate(benchmark.BenchmarkOptions()) == EXPECTED_HEADROOM_ITERATIONS


def test_benchmark_json_output_reports_hard_policy_and_stage_counters(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = benchmark.BenchmarkReport(
        gate_passed=True,
        semantics_match=True,
        iterations=64,
        policy=benchmark.BenchmarkOptions(),
        summaries=(
            benchmark.SampleSummary(
                arm="baseline",
                samples=(0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87),
                median_seconds=0.84,
                speedup_over_baseline=1.0,
            ),
            benchmark.SampleSummary(
                arm="residual",
                samples=(0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31),
                median_seconds=0.28,
                speedup_over_baseline=3.0,
            ),
        ),
        final_speedup=EXPECTED_FINAL_SPEEDUP,
        stage_counters={
            "run_scoped_guard_amortization": 1,
            "quiescent_await_chain_collapse": 1,
            "context_copy_elision": 1,
            "incremental_completion_accounting": 1,
            "result_record_elision": 1,
        },
        semantic_snapshot={"workflow": "generic-residual-async-profile"},
    )

    def fake_run_benchmark(
        _options: benchmark.BenchmarkOptions | None = None,
    ) -> benchmark.BenchmarkReport:
        return report

    monkeypatch.setattr(benchmark, "run_benchmark", fake_run_benchmark)

    assert benchmark.main(()) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["gate_passed"] is True
    assert payload["policy"] == {
        "minimum_seconds": 0.25,
        "minimum_speedup": 3.0,
        "samples": 7,
        "semantic_repetitions": 8,
        "warmups": 1,
    }
    assert payload["stage_counters_nonzero"] is True
    assert payload["final_speedup"] == EXPECTED_FINAL_SPEEDUP
    assert [len(summary["samples"]) for summary in payload["summaries"]] == [7, 7]


def _fixture_module() -> FixtureModule:
    source_root = str(SOURCE_ROOT.resolve())
    sys.path.insert(0, source_root)
    try:
        return cast(FixtureModule, importlib.import_module("residual_async_profile.profile"))
    finally:
        sys.path.remove(source_root)
