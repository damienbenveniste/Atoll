"""Unit tests for generic execution-plan benchmark execution."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import pytest

from atoll.runtime import execution_plan_performance
from atoll.runtime.execution_plan_performance import (
    ExecutionPlanBenchmarkConfig,
    ExecutionPlanBenchmarkProgress,
    run_execution_plan_benchmark,
    unavailable_execution_plan_benchmark,
)
from atoll.runtime.performance import CommandRunEvidence, RuntimeMode

PLAN_ID = "execution-plan:test"


class PerformanceCommandRunner(Protocol):
    def __call__(
        self,
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        mode: RuntimeMode,
        **options: object,
    ) -> CommandRunEvidence: ...


def test_execution_plan_config_rejects_invalid_policy() -> None:
    with pytest.raises(ValueError, match="plan ID"):
        ExecutionPlanBenchmarkConfig(plan_id=" ", command=None)
    with pytest.raises(ValueError, match="command"):
        ExecutionPlanBenchmarkConfig(plan_id=PLAN_ID, command=())
    with pytest.raises(ValueError, match="command"):
        ExecutionPlanBenchmarkConfig(plan_id=PLAN_ID, command=("python", " "))
    with pytest.raises(ValueError, match="samples"):
        ExecutionPlanBenchmarkConfig(plan_id=PLAN_ID, command=None, samples=0)
    with pytest.raises(ValueError, match="marginal"):
        ExecutionPlanBenchmarkConfig(
            plan_id=PLAN_ID,
            command=None,
            minimum_marginal_speedup=0.0,
        )
    with pytest.raises(ValueError, match="overall"):
        ExecutionPlanBenchmarkConfig(
            plan_id=PLAN_ID,
            command=None,
            minimum_overall_speedup=0.0,
        )


def test_execution_plan_benchmark_runs_one_warmup_then_all_six_rotated_orders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[CallRecord] = []
    progress_events: list[ExecutionPlanBenchmarkProgress] = []
    durations = [
        9.0,
        8.0,
        7.0,
        3.0,
        2.0,
        1.0,
        4.0,
        1.0,
        2.0,
        2.0,
        5.0,
        1.0,
        2.0,
        1.0,
        6.0,
        1.0,
        2.0,
        1.0,
        1.0,
        1.0,
        2.0,
    ]
    monkeypatch.setattr(
        execution_plan_performance,
        "run_performance_command",
        _fake_runner(durations, calls),
    )

    result = run_execution_plan_benchmark(
        ExecutionPlanBenchmarkConfig(
            plan_id=PLAN_ID,
            command=("python", "bench.py"),
            samples=6,
        ),
        project_root=tmp_path,
        baseline_payload_root=tmp_path / "baseline",
        unplanned_payload_root=tmp_path / "unplanned",
        planned_payload_root=tmp_path / "planned",
        progress=progress_events.append,
        baseline_region_allowlist=frozenset(("baseline-b", "baseline-a")),
        unplanned_region_allowlist=frozenset(("unplanned-a",)),
        planned_region_allowlist=frozenset(),
    )

    assert [call.arm for call in calls] == [
        "baseline",
        "unplanned",
        "planned",
        "baseline",
        "unplanned",
        "planned",
        "baseline",
        "planned",
        "unplanned",
        "unplanned",
        "baseline",
        "planned",
        "unplanned",
        "planned",
        "baseline",
        "planned",
        "baseline",
        "unplanned",
        "planned",
        "unplanned",
        "baseline",
    ]
    assert [sample.arm for sample in result.warmups] == ["baseline", "unplanned", "planned"]
    assert [sample.arm for sample in result.samples] == [
        "baseline",
        "unplanned",
        "planned",
        "baseline",
        "planned",
        "unplanned",
        "unplanned",
        "baseline",
        "planned",
        "unplanned",
        "planned",
        "baseline",
        "planned",
        "baseline",
        "unplanned",
        "planned",
        "unplanned",
        "baseline",
    ]
    assert [(call.mode, call.payload_name, call.allowlist) for call in calls[:3]] == [
        ("baseline", "baseline", frozenset(("baseline-a", "baseline-b"))),
        ("compiled", "unplanned", frozenset(("unplanned-a",))),
        ("compiled", "planned", frozenset()),
    ]
    assert {call.command_name for call in calls} == {"bench.py"}
    assert [(event.phase, event.sample_index, event.arm) for event in progress_events[:6]] == [
        ("warmup", None, "baseline"),
        ("warmup", None, "unplanned"),
        ("warmup", None, "planned"),
        ("sample", 1, "baseline"),
        ("sample", 1, "unplanned"),
        ("sample", 1, "planned"),
    ]
    assert result.status == "passed"
    assert result.succeeded is True
    assert result.baseline_median_seconds == pytest.approx(3.5)
    assert result.unplanned_median_seconds == pytest.approx(2.0)
    assert result.planned_median_seconds == pytest.approx(1.0)
    assert result.marginal_speedup == pytest.approx(2.0)
    assert result.overall_speedup == pytest.approx(3.5)


def test_execution_plan_benchmark_forwards_variant_allowlists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Composed plan arms retain independently selected native variants."""
    calls: list[CallRecord] = []
    monkeypatch.setattr(
        execution_plan_performance,
        "run_performance_command",
        _fake_runner([3.0, 2.0, 1.0, 3.0, 2.0, 1.0], calls),
    )

    run_execution_plan_benchmark(
        ExecutionPlanBenchmarkConfig(
            plan_id=PLAN_ID,
            command=("python", "bench.py"),
            samples=1,
            minimum_marginal_speedup=1.01,
            minimum_overall_speedup=1.01,
        ),
        project_root=tmp_path,
        baseline_payload_root=tmp_path / "baseline",
        unplanned_payload_root=tmp_path / "unplanned",
        planned_payload_root=tmp_path / "planned",
        baseline_variant_allowlist=frozenset(("base",)),
        unplanned_variant_allowlist=frozenset(("native",)),
        planned_variant_allowlist=frozenset(("native", "planned")),
        baseline_require_optimized=True,
        unplanned_require_optimized=False,
        planned_require_optimized=True,
    )

    assert [(call.arm, call.allowlist) for call in calls[:3]] == [
        ("baseline", frozenset(("base",))),
        ("unplanned", frozenset(("native",))),
        ("planned", frozenset(("native", "planned"))),
    ]
    assert [(call.arm, call.require_optimized) for call in calls[:3]] == [
        ("baseline", True),
        ("unplanned", False),
        ("planned", True),
    ]


def test_execution_plan_benchmark_marks_unavailable_without_command(tmp_path: Path) -> None:
    result = run_execution_plan_benchmark(
        ExecutionPlanBenchmarkConfig(plan_id=PLAN_ID, command=None),
        project_root=tmp_path,
        baseline_payload_root=tmp_path / "baseline",
        unplanned_payload_root=tmp_path / "unplanned",
        planned_payload_root=tmp_path / "planned",
    )

    assert result.status == "unavailable"
    assert result.plan_id == PLAN_ID
    assert result.reason == "no execution-plan benchmark command configured"
    assert result.warmups == ()
    assert result.samples == ()
    assert result.marginal_speedup is None
    assert result.overall_speedup is None


def test_unavailable_execution_plan_benchmark_requires_and_retains_plan_identity() -> None:
    result = unavailable_execution_plan_benchmark(PLAN_ID, "staged source changed")

    assert result.plan_id == PLAN_ID
    assert result.status == "unavailable"
    assert result.reason == "staged source changed"
    with pytest.raises(ValueError, match="plan ID"):
        unavailable_execution_plan_benchmark(" ", "reason")
    with pytest.raises(ValueError, match="reason"):
        unavailable_execution_plan_benchmark(PLAN_ID, " ")


def test_execution_plan_benchmark_rejects_nonprofitable_thresholds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[CallRecord] = []
    monkeypatch.setattr(
        execution_plan_performance,
        "run_performance_command",
        _fake_runner([0.5, 0.5, 0.5, 1.08, 1.04, 1.0], calls),
    )

    result = run_execution_plan_benchmark(
        ExecutionPlanBenchmarkConfig(
            plan_id=PLAN_ID,
            command=("python", "bench.py"),
            samples=1,
        ),
        project_root=tmp_path,
        baseline_payload_root=tmp_path / "baseline",
        unplanned_payload_root=tmp_path / "unplanned",
        planned_payload_root=tmp_path / "planned",
    )

    assert result.status == "not-profitable"
    assert result.overall_speedup == pytest.approx(1.08)
    assert result.marginal_speedup == pytest.approx(1.04)
    assert "marginal ratio missed threshold" in result.reason


def test_execution_plan_benchmark_defers_overall_ratio_to_final_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[CallRecord] = []
    monkeypatch.setattr(
        execution_plan_performance,
        "run_performance_command",
        _fake_runner([0.5, 0.5, 0.5, 1.095, 1.079, 1.0], calls),
    )

    result = run_execution_plan_benchmark(
        ExecutionPlanBenchmarkConfig(
            plan_id=PLAN_ID,
            command=("python", "bench.py"),
            samples=1,
        ),
        project_root=tmp_path,
        baseline_payload_root=tmp_path / "baseline",
        unplanned_payload_root=tmp_path / "unplanned",
        planned_payload_root=tmp_path / "planned",
    )

    assert result.status == "passed"
    assert result.marginal_speedup == pytest.approx(1.079)
    assert result.overall_speedup == pytest.approx(1.095)
    assert "final payload gate decides promotion" in result.reason


def test_execution_plan_benchmark_accepts_exact_thresholds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[CallRecord] = []
    monkeypatch.setattr(
        execution_plan_performance,
        "run_performance_command",
        _fake_runner([0.5, 0.5, 0.5, 1.10, 1.05, 1.0], calls),
    )

    result = run_execution_plan_benchmark(
        ExecutionPlanBenchmarkConfig(
            plan_id=PLAN_ID,
            command=("python", "bench.py"),
            samples=1,
        ),
        project_root=tmp_path,
        baseline_payload_root=tmp_path / "baseline",
        unplanned_payload_root=tmp_path / "unplanned",
        planned_payload_root=tmp_path / "planned",
    )

    assert result.status == "passed"
    assert result.overall_speedup == pytest.approx(1.10)
    assert result.marginal_speedup == pytest.approx(1.05)


def test_execution_plan_benchmark_rejects_noisy_medians(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[CallRecord] = []
    monkeypatch.setattr(
        execution_plan_performance,
        "run_performance_command",
        _fake_runner([0.5, 0.5, 0.5, 0.30, 0.26, 0.20], calls),
    )

    result = run_execution_plan_benchmark(
        ExecutionPlanBenchmarkConfig(
            plan_id=PLAN_ID,
            command=("python", "bench.py"),
            samples=1,
        ),
        project_root=tmp_path,
        baseline_payload_root=tmp_path / "baseline",
        unplanned_payload_root=tmp_path / "unplanned",
        planned_payload_root=tmp_path / "planned",
    )

    assert result.status == "invalid"
    assert "too noisy" in result.reason
    assert result.baseline_median_seconds == pytest.approx(0.30)
    assert result.unplanned_median_seconds == pytest.approx(0.26)
    assert result.planned_median_seconds == pytest.approx(0.20)
    assert result.marginal_speedup is None
    assert result.overall_speedup is None


def test_execution_plan_benchmark_stops_after_failed_warmup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[CallRecord] = []
    monkeypatch.setattr(
        execution_plan_performance,
        "run_performance_command",
        _fake_runner([0.5, 0.5], calls, failing_call=2),
    )

    result = run_execution_plan_benchmark(
        ExecutionPlanBenchmarkConfig(
            plan_id=PLAN_ID,
            command=("python", "bench.py"),
            samples=1,
        ),
        project_root=tmp_path,
        baseline_payload_root=tmp_path / "baseline",
        unplanned_payload_root=tmp_path / "unplanned",
        planned_payload_root=tmp_path / "planned",
    )

    assert result.status == "invalid"
    assert result.reason == (
        "warmup execution-plan benchmark command exited with status 2 in unplanned arm"
    )
    assert [sample.arm for sample in result.warmups] == ["baseline", "unplanned"]
    assert result.samples == ()
    assert result.baseline_median_seconds is None
    assert result.marginal_speedup is None


def test_execution_plan_benchmark_stops_after_failed_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[CallRecord] = []
    monkeypatch.setattr(
        execution_plan_performance,
        "run_performance_command",
        _fake_runner([0.5, 0.5, 0.5, 1.0], calls, failing_call=4),
    )

    result = run_execution_plan_benchmark(
        ExecutionPlanBenchmarkConfig(
            plan_id=PLAN_ID,
            command=("python", "bench.py"),
            samples=2,
        ),
        project_root=tmp_path,
        baseline_payload_root=tmp_path / "baseline",
        unplanned_payload_root=tmp_path / "unplanned",
        planned_payload_root=tmp_path / "planned",
    )

    assert result.status == "invalid"
    assert result.reason == (
        "sample execution-plan benchmark command exited with status 2 in baseline arm"
    )
    assert [sample.arm for sample in result.warmups] == ["baseline", "unplanned", "planned"]
    assert [sample.arm for sample in result.samples] == ["baseline"]


def test_execution_plan_benchmark_rejects_unexpected_options(tmp_path: Path) -> None:
    runner = cast(
        Callable[..., object],
        execution_plan_performance.run_execution_plan_benchmark,
    )
    with pytest.raises(TypeError, match="unexpected"):
        runner(
            ExecutionPlanBenchmarkConfig(plan_id=PLAN_ID, command=("python", "bench.py")),
            project_root=tmp_path,
            baseline_payload_root=tmp_path / "baseline",
            unplanned_payload_root=tmp_path / "unplanned",
            planned_payload_root=tmp_path / "planned",
            unknown_allowlist=frozenset[str](),
        )


@pytest.mark.parametrize(
    ("payload_name", "arm"),
    [
        ("baseline", "baseline"),
        ("unplanned", "unplanned"),
        ("planned", "planned"),
    ],
)
def test_fake_runner_maps_payload_names(payload_name: str, arm: str, tmp_path: Path) -> None:
    assert _arm_from_payload(tmp_path / payload_name) == arm


@dataclass(frozen=True, slots=True)
class CallRecord:
    arm: str
    command_name: str
    mode: RuntimeMode
    payload_name: str
    allowlist: frozenset[str] | None
    require_optimized: bool


def _fake_runner(
    durations: Iterable[float],
    calls: list[CallRecord],
    *,
    failing_call: int | None = None,
) -> PerformanceCommandRunner:
    duration_iter = iter(durations)

    def run(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        mode: RuntimeMode,
        **options: object,
    ) -> CommandRunEvidence:
        arm = _arm_from_payload(payload_root)
        variant_allowlist = cast(frozenset[str] | None, options.get("variant_allowlist"))
        region_allowlist = cast(frozenset[str] | None, options.get("region_allowlist"))
        calls.append(
            CallRecord(
                arm=arm,
                command_name=command[-1],
                mode=mode,
                payload_name=payload_root.name,
                allowlist=variant_allowlist or region_allowlist,
                require_optimized=cast(bool, options.get("require_optimized", False)),
            )
        )
        return CommandRunEvidence(
            command=command,
            project_root=project_root,
            payload_root=payload_root,
            mode=mode,
            returncode=2 if failing_call == len(calls) else 0,
            stdout="",
            stderr="boom" if failing_call == len(calls) else "",
            duration_seconds=next(duration_iter),
        )

    return run


def _arm_from_payload(payload_root: Path) -> str:
    if payload_root.name not in {"baseline", "unplanned", "planned"}:
        raise AssertionError(f"unexpected payload root: {payload_root}")
    return payload_root.relative_to(payload_root.parent).as_posix()
