"""Unit tests for three-arm fusion research benchmark execution."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pytest

from atoll.runtime import fusion_performance
from atoll.runtime.fusion_performance import (
    FusionBenchmarkConfig,
    run_fusion_trial,
    unavailable_fusion_trial,
)
from atoll.runtime.performance import CommandRunEvidence, RuntimeMode

SEMANTIC_ARM_COUNT = 3
PLAN_ID = "task-fusion:test"


class PerformanceCommandRunner(Protocol):
    def __call__(
        self,
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        mode: RuntimeMode,
        region_allowlist: frozenset[str] | None = None,
    ) -> CommandRunEvidence: ...


def test_fusion_config_rejects_invalid_policy() -> None:
    with pytest.raises(ValueError, match="plan ID"):
        FusionBenchmarkConfig(plan_id=" ", command=None)
    with pytest.raises(ValueError, match="command"):
        FusionBenchmarkConfig(plan_id=PLAN_ID, command=())
    with pytest.raises(ValueError, match="command"):
        FusionBenchmarkConfig(plan_id=PLAN_ID, command=("python", " "))
    with pytest.raises(ValueError, match="semantic"):
        FusionBenchmarkConfig(plan_id=PLAN_ID, command=("python", "bench.py"))
    with pytest.raises(ValueError, match="semantic"):
        FusionBenchmarkConfig(plan_id=PLAN_ID, command=None, semantic_command=())
    with pytest.raises(ValueError, match="warmups"):
        FusionBenchmarkConfig(plan_id=PLAN_ID, command=None, warmups=-1)
    with pytest.raises(ValueError, match="samples"):
        FusionBenchmarkConfig(plan_id=PLAN_ID, command=None, samples=0)
    with pytest.raises(ValueError, match="unfused"):
        FusionBenchmarkConfig(plan_id=PLAN_ID, command=None, minimum_over_unfused=0.0)
    with pytest.raises(ValueError, match="overall"):
        FusionBenchmarkConfig(plan_id=PLAN_ID, command=None, minimum_overall=0.0)


def test_fusion_trial_runs_semantic_first_then_rotated_warmups_and_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[CallRecord] = []
    durations = [
        0.8,
        0.7,
        0.6,
        9.0,
        8.0,
        7.0,
        3.0,
        2.0,
        1.0,
        2.0,
        1.0,
        2.0,
        0.5,
        4.0,
        1.0,
    ]
    monkeypatch.setattr(
        fusion_performance,
        "run_performance_command",
        _fake_runner(durations, calls),
    )

    result = run_fusion_trial(
        FusionBenchmarkConfig(
            plan_id=PLAN_ID,
            command=("python", "bench.py"),
            semantic_command=("python", "verify.py"),
            warmups=1,
            samples=3,
        ),
        project_root=tmp_path,
        baseline_payload_root=tmp_path / "baseline",
        unfused_payload_root=tmp_path / "unfused",
        fused_payload_root=tmp_path / "fused",
        baseline_region_allowlist=frozenset(("baseline-b", "baseline-a")),
        unfused_region_allowlist=frozenset(("unfused-a",)),
        fused_region_allowlist=frozenset(),
    )

    assert [call.arm for call in calls] == [
        "baseline",
        "unfused",
        "fused",
        "baseline",
        "unfused",
        "fused",
        "baseline",
        "unfused",
        "fused",
        "unfused",
        "fused",
        "baseline",
        "fused",
        "baseline",
        "unfused",
    ]
    assert [run.arm for run in result.semantic_runs] == ["baseline", "unfused", "fused"]
    assert [run.arm for run in result.warmups] == ["baseline", "unfused", "fused"]
    assert [run.arm for run in result.samples] == [
        "baseline",
        "unfused",
        "fused",
        "unfused",
        "fused",
        "baseline",
        "fused",
        "baseline",
        "unfused",
    ]
    assert [(call.mode, call.payload_name, call.allowlist) for call in calls[:3]] == [
        ("baseline", "baseline", frozenset(("baseline-a", "baseline-b"))),
        ("compiled", "unfused", frozenset(("unfused-a",))),
        ("compiled", "fused", frozenset()),
    ]
    assert [call.command_name for call in calls[:3]] == ["verify.py"] * SEMANTIC_ARM_COUNT
    assert {call.command_name for call in calls[3:]} == {"bench.py"}
    assert result.status == "passed"
    assert result.plan_id == PLAN_ID
    assert result.baseline_median_seconds == pytest.approx(3.0)
    assert result.unfused_median_seconds == pytest.approx(2.0)
    assert result.fused_median_seconds == pytest.approx(1.0)
    assert result.baseline_over_unfused == pytest.approx(1.5)
    assert result.baseline_over_fused == pytest.approx(3.0)
    assert result.unfused_over_fused == pytest.approx(2.0)


def test_fusion_trial_marks_unavailable_without_command(tmp_path: Path) -> None:
    result = run_fusion_trial(
        FusionBenchmarkConfig(plan_id=PLAN_ID, command=None),
        project_root=tmp_path,
        baseline_payload_root=tmp_path / "baseline",
        unfused_payload_root=tmp_path / "unfused",
        fused_payload_root=tmp_path / "fused",
    )

    assert result.status == "unavailable"
    assert result.plan_id == PLAN_ID
    assert result.reason == "no fusion benchmark command configured"
    assert result.semantic_runs == ()
    assert result.warmups == ()
    assert result.samples == ()


def test_unavailable_fusion_trial_requires_and_retains_plan_identity() -> None:
    result = unavailable_fusion_trial(PLAN_ID, "staged source no longer matches")

    assert result.plan_id == PLAN_ID
    assert result.status == "unavailable"
    assert result.reason == "staged source no longer matches"
    with pytest.raises(ValueError, match="plan ID"):
        unavailable_fusion_trial(" ", "reason")
    with pytest.raises(ValueError, match="reason"):
        unavailable_fusion_trial(PLAN_ID, " ")


def test_fusion_trial_rejects_nonprofitable_thresholds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[CallRecord] = []
    monkeypatch.setattr(
        fusion_performance,
        "run_performance_command",
        _fake_runner(
            [
                0.5,
                0.5,
                0.5,
                1.08,
                1.04,
                1.0,
            ],
            calls,
        ),
    )

    result = run_fusion_trial(
        FusionBenchmarkConfig(
            plan_id=PLAN_ID,
            command=("python", "bench.py"),
            semantic_command=("python", "verify.py"),
            warmups=0,
            samples=1,
        ),
        project_root=tmp_path,
        baseline_payload_root=tmp_path / "baseline",
        unfused_payload_root=tmp_path / "unfused",
        fused_payload_root=tmp_path / "fused",
    )

    assert result.status == "not-profitable"
    assert result.baseline_over_fused == pytest.approx(1.08)
    assert result.unfused_over_fused == pytest.approx(1.04)
    assert "missed thresholds" in result.reason


def test_fusion_trial_accepts_exact_thresholds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[CallRecord] = []
    monkeypatch.setattr(
        fusion_performance,
        "run_performance_command",
        _fake_runner(
            [
                0.5,
                0.5,
                0.5,
                1.10,
                1.05,
                1.0,
            ],
            calls,
        ),
    )

    result = run_fusion_trial(
        FusionBenchmarkConfig(
            plan_id=PLAN_ID,
            command=("python", "bench.py"),
            semantic_command=("python", "verify.py"),
            warmups=0,
            samples=1,
        ),
        project_root=tmp_path,
        baseline_payload_root=tmp_path / "baseline",
        unfused_payload_root=tmp_path / "unfused",
        fused_payload_root=tmp_path / "fused",
    )

    assert result.status == "passed"
    assert result.baseline_over_fused == pytest.approx(1.10)
    assert result.unfused_over_fused == pytest.approx(1.05)


def test_fusion_trial_rejects_noisy_medians(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[CallRecord] = []
    monkeypatch.setattr(
        fusion_performance,
        "run_performance_command",
        _fake_runner(
            [
                0.5,
                0.5,
                0.5,
                0.30,
                0.26,
                0.20,
            ],
            calls,
        ),
    )

    result = run_fusion_trial(
        FusionBenchmarkConfig(
            plan_id=PLAN_ID,
            command=("python", "bench.py"),
            semantic_command=("python", "verify.py"),
            warmups=0,
            samples=1,
        ),
        project_root=tmp_path,
        baseline_payload_root=tmp_path / "baseline",
        unfused_payload_root=tmp_path / "unfused",
        fused_payload_root=tmp_path / "fused",
    )

    assert result.status == "invalid"
    assert "too noisy" in result.reason
    assert result.baseline_median_seconds == pytest.approx(0.30)
    assert result.unfused_median_seconds == pytest.approx(0.26)
    assert result.fused_median_seconds == pytest.approx(0.20)
    assert result.baseline_over_fused is None


def test_fusion_trial_stops_after_failed_semantic_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[CallRecord] = []
    monkeypatch.setattr(
        fusion_performance,
        "run_performance_command",
        _fake_runner([0.5, 0.5], calls, failing_call=2),
    )

    result = run_fusion_trial(
        FusionBenchmarkConfig(
            plan_id=PLAN_ID,
            command=("python", "bench.py"),
            semantic_command=("python", "verify.py"),
            warmups=1,
            samples=1,
        ),
        project_root=tmp_path,
        baseline_payload_root=tmp_path / "baseline",
        unfused_payload_root=tmp_path / "unfused",
        fused_payload_root=tmp_path / "fused",
    )

    assert result.status == "invalid"
    assert result.reason == "semantic fusion benchmark command exited with status 2 in unfused arm"
    assert [run.arm for run in result.semantic_runs] == ["baseline", "unfused"]
    assert result.warmups == ()
    assert result.samples == ()


def test_fusion_trial_stops_after_failed_sample_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[CallRecord] = []
    monkeypatch.setattr(
        fusion_performance,
        "run_performance_command",
        _fake_runner([0.5, 0.5, 0.5, 1.0], calls, failing_call=4),
    )

    result = run_fusion_trial(
        FusionBenchmarkConfig(
            plan_id=PLAN_ID,
            command=("python", "bench.py"),
            semantic_command=("python", "verify.py"),
            warmups=0,
            samples=2,
        ),
        project_root=tmp_path,
        baseline_payload_root=tmp_path / "baseline",
        unfused_payload_root=tmp_path / "unfused",
        fused_payload_root=tmp_path / "fused",
    )

    assert result.status == "invalid"
    assert result.reason == "sample fusion benchmark command exited with status 2 in baseline arm"
    assert len(result.semantic_runs) == SEMANTIC_ARM_COUNT
    assert [run.arm for run in result.samples] == ["baseline"]


def test_fusion_trial_rejects_unexpected_options(tmp_path: Path) -> None:
    options = {"unknown_allowlist": frozenset[str]()}
    with pytest.raises(TypeError, match="unexpected"):
        fusion_performance.run_fusion_trial(
            FusionBenchmarkConfig(
                plan_id=PLAN_ID,
                command=("python", "bench.py"),
                semantic_command=("python", "verify.py"),
            ),
            project_root=tmp_path,
            baseline_payload_root=tmp_path / "baseline",
            unfused_payload_root=tmp_path / "unfused",
            fused_payload_root=tmp_path / "fused",
            **options,
        )


@pytest.mark.parametrize(
    ("payload_name", "arm"),
    [
        ("baseline", "baseline"),
        ("unfused", "unfused"),
        ("fused", "fused"),
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
        region_allowlist: frozenset[str] | None = None,
    ) -> CommandRunEvidence:
        arm = _arm_from_payload(payload_root)
        calls.append(
            CallRecord(
                arm=arm,
                command_name=command[-1],
                mode=mode,
                payload_name=payload_root.name,
                allowlist=region_allowlist,
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
    if payload_root.name not in {"baseline", "unfused", "fused"}:
        raise AssertionError(f"unexpected payload root: {payload_root}")
    return payload_root.relative_to(payload_root.parent).as_posix()
