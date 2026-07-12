"""Tests for the generic async execution-plan feasibility harness."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine
from typing import Protocol, cast

import pytest
from scripts import async_execution_plan_feasibility as feasibility
from scripts.async_execution_plan_feasibility import (
    ARMS,
    BENCHMARK_ARMS,
    BenchmarkArmName,
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
UNSAFE_MEDIAN_SECONDS = 1.0
GUARDED_MEDIAN_SECONDS = 1.25
EXPECTED_UNSAFE_SPEEDUP = 4.0
EXPECTED_GUARDED_SPEEDUP = 3.2
MINIMUM_UNSAFE_FUSED_SPEEDUP = 3.3
MINIMUM_GUARDED_FUSED_SPEEDUP = 3.0
SEMANTIC_REPETITIONS = 32
BENCHMARK_PROBE_WIDTH = 4


class _BenchmarkQueueView(Protocol):
    """Expose only the queue state inspected by the benchmark regression test."""

    @property
    def depth(self) -> int:
        """Return the number of queued results."""

        ...

    @property
    def is_full(self) -> bool:
        """Return whether the queue reached its configured capacity."""

        ...


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


def test_guarded_fused_arm_drives_immediate_work_and_falls_back_for_tasks() -> None:
    execution = asyncio.run(execute_arm("guarded_fused_state_machine", build_workload()))

    assert dict(execution.paths) == {
        "work-000": "guarded_fused",
        "work-001": "task_fallback",
        "work-002": "guarded_fused",
        "work-003": "guarded_fused",
        "work-004": "task_fallback",
        "work-005": "guarded_fused",
    }
    snapshot = semantic_snapshot(execution)
    assert snapshot["context_isolated"] is True
    assert snapshot["capacity_one"] is True
    assert snapshot["active_after_cleanup"] == 0


def test_semantic_probes_preserve_backpressure_cancellation_and_custom_factory() -> None:
    baseline = asyncio.run(run_semantic_probes("baseline"))
    callback = asyncio.run(run_semantic_probes("callback_backed"))
    guarded = asyncio.run(run_semantic_probes("guarded_fused_state_machine"))

    assert callback == baseline
    assert guarded == baseline
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
    assert guarded["guarded_direct"] == {
        "direct_mutated_context": "child:guarded-direct",
        "direct_path": "guarded_fused",
        "direct_starting_context": "scheduled:guarded-direct",
        "fallback_paths": ["task_fallback", "task_fallback"],
        "parent_context_after": "parent",
        "task_observer_saw_task": True,
    }


def test_guarded_direct_probe_would_catch_parent_context_leak() -> None:
    probe = asyncio.run(run_semantic_probes("guarded_fused_state_machine"))
    guarded = cast(dict[str, JsonValue], probe["guarded_direct"])

    assert guarded["direct_starting_context"] == "scheduled:guarded-direct"
    assert guarded["direct_mutated_context"] == "child:guarded-direct"
    assert guarded["parent_context_after"] == "parent"


def test_fused_benchmark_arms_publish_full_fifo_queue_before_batch_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drained: list[tuple[str, int, bool]] = []
    original_drain = cast(
        Callable[[_BenchmarkQueueView], int],
        vars(feasibility)["_benchmark_batch_drain"],
    )
    benchmark_round = cast(
        Callable[[BenchmarkArmName, int], Coroutine[object, object, int]],
        vars(feasibility)["_benchmark_round"],
    )

    def spy_drain(queue: _BenchmarkQueueView) -> int:
        drained.append(
            (
                type(queue).__name__,
                queue.depth,
                queue.is_full,
            )
        )
        return original_drain(queue)

    monkeypatch.setattr(feasibility, "_benchmark_batch_drain", spy_drain)

    unsafe = asyncio.run(benchmark_round("unsafe_fused_state_machine", BENCHMARK_PROBE_WIDTH))
    guarded = asyncio.run(benchmark_round("guarded_fused_state_machine", BENCHMARK_PROBE_WIDTH))

    assert unsafe == sum(range(1, BENCHMARK_PROBE_WIDTH + 1))
    assert guarded == unsafe
    assert drained == [
        ("_BenchmarkBatchQueue", BENCHMARK_PROBE_WIDTH, True),
        ("_BenchmarkBatchQueue", BENCHMARK_PROBE_WIDTH, True),
    ]


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
            "unsafe_fused_state_machine": (0.9, 1.0, 1.1),
            "guarded_fused_state_machine": (1.0, 1.25, 1.5),
        }
    )

    by_arm = {summary.arm: summary for summary in summaries}
    assert by_arm["baseline"].median_seconds == BASELINE_MEDIAN_SECONDS
    assert by_arm["callback_backed"].median_seconds == CALLBACK_MEDIAN_SECONDS
    assert by_arm["callback_backed"].speedup_over_baseline == EXPECTED_SPEEDUP
    assert by_arm["unsafe_fused_state_machine"].median_seconds == UNSAFE_MEDIAN_SECONDS
    assert by_arm["guarded_fused_state_machine"].median_seconds == GUARDED_MEDIAN_SECONDS
    assert by_arm["unsafe_fused_state_machine"].speedup_over_baseline == EXPECTED_UNSAFE_SPEEDUP
    assert by_arm["guarded_fused_state_machine"].speedup_over_baseline == EXPECTED_GUARDED_SPEEDUP
    assert rotated_arms(0) == BENCHMARK_ARMS
    assert rotated_arms(1) == BENCHMARK_ARMS[1:] + BENCHMARK_ARMS[:1]
    assert rotated_arms(2) == BENCHMARK_ARMS[2:] + BENCHMARK_ARMS[:2]


def test_report_json_is_canonical_and_contains_elapsed_seconds() -> None:
    report = _report(gate_passed=True)
    payload = report_as_json(report)
    encoded = canonical_json(payload)
    decoded = json.loads(encoded)

    assert decoded["benchmark"]["arms"][0]["sample_seconds"] == [0.5]
    assert decoded["benchmark"]["unsafe_fused_speedup"] == EXPECTED_UNSAFE_SPEEDUP
    assert decoded["benchmark"]["guarded_fused_speedup"] == EXPECTED_GUARDED_SPEEDUP
    assert decoded["minimum_unsafe_fused_speedup"] == MINIMUM_UNSAFE_FUSED_SPEEDUP
    assert decoded["minimum_guarded_fused_speedup"] == MINIMUM_GUARDED_FUSED_SPEEDUP
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
            speedup_over_baseline=(
                EXPECTED_UNSAFE_SPEEDUP
                if arm == "unsafe_fused_state_machine"
                else EXPECTED_GUARDED_SPEEDUP
                if arm == "guarded_fused_state_machine"
                else EXPECTED_SPEEDUP
                if arm == "callback_backed"
                else 1.0
            ),
        )
        for arm in BENCHMARK_ARMS
    )
    return FeasibilityReport(
        semantic_repetitions=SEMANTIC_REPETITIONS,
        semantic_snapshot={"execution": {"context_isolated": True}},
        semantics_match=True,
        benchmark_width=5_000,
        benchmark_rounds=32,
        benchmark_summaries=summaries,
        callback_speedup=EXPECTED_SPEEDUP,
        unsafe_fused_speedup=EXPECTED_UNSAFE_SPEEDUP,
        guarded_fused_speedup=EXPECTED_GUARDED_SPEEDUP,
        minimum_callback_speedup=1.5,
        minimum_unsafe_fused_speedup=MINIMUM_UNSAFE_FUSED_SPEEDUP,
        minimum_guarded_fused_speedup=MINIMUM_GUARDED_FUSED_SPEEDUP,
        stable_timings=True,
        gate_passed=gate_passed,
    )
