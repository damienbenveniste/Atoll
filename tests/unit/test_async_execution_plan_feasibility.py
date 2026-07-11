"""Tests for the generic async execution-plan feasibility harness."""

from __future__ import annotations

import asyncio
import json
from typing import cast

import pytest
from scripts import async_execution_plan_feasibility as feasibility
from scripts.async_execution_plan_feasibility import (
    ARMS,
    BenchmarkSummary,
    FeasibilityOptions,
    FeasibilityReport,
    JsonValue,
    build_workload,
    canonical_json,
    compare_semantics,
    execute_arm,
    main,
    report_as_json,
    rotated_arms,
    run_semantic_probes,
    semantic_snapshot,
    summarize_samples,
)

BASELINE_MEDIAN_SECONDS = 4.0
CALLBACK_MEDIAN_SECONDS = 2.0
EXPECTED_SPEEDUP = 2.0
SEMANTIC_REPETITIONS = 32


def test_semantic_snapshots_match_across_all_scheduler_arms() -> None:
    workload = build_workload()
    executions = {arm: asyncio.run(execute_arm(arm, workload)) for arm in ARMS}
    snapshots = {arm: semantic_snapshot(execution) for arm, execution in executions.items()}

    assert snapshots["baseline"] == snapshots["task_preserving"]
    assert snapshots["baseline"] == snapshots["callback_backed"]
    snapshot = snapshots["baseline"]
    assert snapshot["capacity_one"] is True
    assert cast(int, snapshot["blocked_publications"]) > 0
    assert snapshot["context_isolated"] is True
    assert snapshot["cold_decoys"] == ["cold-decoy-a", "cold-decoy-b"]
    assert snapshot["active_after_cleanup"] == 0


def test_callback_arm_schedules_callbacks_and_falls_back_before_suspension() -> None:
    execution = asyncio.run(execute_arm("callback_backed", build_workload()))

    assert dict(execution.paths) == {
        "work-000": "callback",
        "work-001": "task_fallback",
        "work-002": "callback",
        "work-003": "callback",
        "work-004": "task_fallback",
        "work-005": "callback",
    }
    failure = next(record for record in execution.records if record.name == "work-003")
    observer = next(record for record in execution.records if record.name == "work-004")
    assert failure.error is not None
    assert failure.error.type_name == "_ControlledWorkError"
    assert failure.error.message == "failure:work-003"
    assert failure.error.cause_type == "LookupError"
    assert failure.error.notes == ("note:work-003",)
    assert failure.error.work_frame_present is True
    assert observer.task_identity_observed is True


def test_semantic_probes_preserve_backpressure_cancellation_and_custom_factory() -> None:
    baseline = asyncio.run(run_semantic_probes("baseline"))
    callback = asyncio.run(run_semantic_probes("callback_backed"))

    assert callback == baseline
    assert callback["blocked_publication"] == {
        "cancelled_blocked_send": True,
        "first_completion_immediate": True,
        "first_received": "first",
        "retained_after_cancel": "first",
        "second_received": "second",
        "second_released_after_drain": True,
        "second_was_blocked": True,
    }
    assert callback["cancellation"] == {
        "cancelled": True,
        "cleanup_count": 1,
        "task_done": True,
    }
    assert callback["custom_factory"] == {
        "factory_calls": 1,
        "real_task_path": True,
        "result": 2,
    }


def test_repeated_semantic_comparison_is_deterministic() -> None:
    snapshot, matched = compare_semantics(2)
    execution = cast(dict[str, JsonValue], snapshot["execution"])
    probes = cast(dict[str, JsonValue], snapshot["probes"])
    custom_factory = cast(dict[str, JsonValue], probes["custom_factory"])

    assert matched is True
    assert execution["context_isolated"] is True
    assert custom_factory["real_task_path"] is True


def test_wall_clock_summary_uses_measured_samples_and_rotating_order() -> None:
    summaries = summarize_samples(
        {
            "baseline": (3.0, 4.0, 5.0),
            "task_preserving": (2.0, 2.5, 3.0),
            "callback_backed": (1.0, 2.0, 2.0),
        }
    )

    by_arm = {summary.arm: summary for summary in summaries}
    assert by_arm["baseline"].median_seconds == BASELINE_MEDIAN_SECONDS
    assert by_arm["callback_backed"].median_seconds == CALLBACK_MEDIAN_SECONDS
    assert by_arm["callback_backed"].speedup_over_baseline == EXPECTED_SPEEDUP
    assert rotated_arms(0) == ("baseline", "task_preserving", "callback_backed")
    assert rotated_arms(1) == ("task_preserving", "callback_backed", "baseline")
    assert rotated_arms(2) == ("callback_backed", "baseline", "task_preserving")


def test_report_json_is_canonical_and_contains_elapsed_seconds() -> None:
    report = _report(gate_passed=True)
    payload = report_as_json(report)
    encoded = canonical_json(payload)
    decoded = json.loads(encoded)

    assert decoded["benchmark"]["arms"][0]["sample_seconds"] == [0.5]
    assert decoded["semantic_repetitions"] == SEMANTIC_REPETITIONS
    assert decoded["gate_passed"] is True
    assert "cost_units" not in encoded
    assert "0x" not in encoded
    assert canonical_json(payload) == encoded


def test_cli_exit_status_follows_real_report_gate(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def failed_report(_options: FeasibilityOptions) -> FeasibilityReport:
        return _report(gate_passed=False)

    monkeypatch.setattr(
        feasibility,
        "run_feasibility",
        failed_report,
    )

    exit_code = main(())

    captured = capsys.readouterr()
    assert exit_code == 1
    assert '"gate_passed":false' in captured.out
    assert "feasibility gate failed" in captured.err


def _report(*, gate_passed: bool) -> FeasibilityReport:
    summaries = tuple(
        BenchmarkSummary(
            arm=arm,
            sample_seconds=(0.5,),
            median_seconds=0.5,
            speedup_over_baseline=2.0 if arm == "callback_backed" else 1.0,
        )
        for arm in ARMS
    )
    return FeasibilityReport(
        semantic_repetitions=SEMANTIC_REPETITIONS,
        semantic_snapshot={"execution": {"context_isolated": True}},
        semantics_match=True,
        benchmark_width=5_000,
        benchmark_rounds=32,
        benchmark_summaries=summaries,
        callback_speedup=2.0,
        minimum_callback_speedup=1.5,
        stable_timings=True,
        gate_passed=gate_passed,
    )
