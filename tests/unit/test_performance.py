"""Unit tests for Milestone 7 performance benchmark execution."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, cast

import pytest

from atoll.runtime import performance
from atoll.runtime.performance import (
    BenchmarkGateConfig,
    BenchmarkProgress,
    run_benchmark_gate,
    run_performance_command,
)

FAILED_CALL_INDEX = 3
EXPECTED_WARMUP_RUNS = 2


class SubprocessInvocationView(Protocol):
    command: tuple[str, ...]
    cwd: Path
    env: dict[str, str]
    check: bool
    shell: bool
    capture_output: bool
    text: bool


def test_benchmark_config_rejects_invalid_policy() -> None:
    """Invalid benchmark settings fail before any subprocess can launch."""
    with pytest.raises(ValueError, match="command"):
        BenchmarkGateConfig(command=(), warmups=1, samples=1, minimum_speedup=1.1)
    with pytest.raises(ValueError, match="command"):
        BenchmarkGateConfig(command=("python", ""), warmups=1, samples=1, minimum_speedup=1.1)
    with pytest.raises(ValueError, match="warmups"):
        BenchmarkGateConfig(command=None, warmups=-1, samples=1, minimum_speedup=1.1)
    with pytest.raises(ValueError, match="samples"):
        BenchmarkGateConfig(command=None, warmups=0, samples=0, minimum_speedup=1.1)
    with pytest.raises(ValueError, match="speedup"):
        BenchmarkGateConfig(command=None, warmups=0, samples=1, minimum_speedup=0.0)


def test_run_performance_command_sets_child_env_without_mutating_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner uses argv subprocess execution and isolates Atoll flags."""
    captured: dict[str, object] = {}
    monkeypatch.setenv("PYTHONPATH", "existing")
    monkeypatch.setenv("ATOLL_DISABLE", "parent-disable")
    monkeypatch.setenv("ATOLL_REQUIRE_COMPILED", "parent-require")
    monkeypatch.setattr(performance, "_perf_counter", _clock([1.5]))

    def fake_run(
        invocation: SubprocessInvocationView,
    ) -> subprocess.CompletedProcess[str]:
        captured.update(
            {
                "command": invocation.command,
                "cwd": invocation.cwd,
                "env": invocation.env,
                "check": invocation.check,
                "shell": invocation.shell,
                "capture_output": invocation.capture_output,
                "text": invocation.text,
            }
        )
        return subprocess.CompletedProcess(invocation.command, 0, "out", "err")

    monkeypatch.setattr(performance, "_run_subprocess", fake_run)

    evidence = run_performance_command(
        ("python", "bench.py"),
        project_root=tmp_path,
        payload_root=tmp_path / "payload",
        mode="compiled",
    )

    child_env = cast(dict[str, str], captured["env"])
    assert captured == {
        "command": ("python", "bench.py"),
        "cwd": tmp_path.resolve(),
        "env": child_env,
        "check": False,
        "shell": False,
        "capture_output": True,
        "text": True,
    }
    assert child_env["PYTHONPATH"] == f"{(tmp_path / 'payload').resolve()}{os.pathsep}existing"
    assert child_env["ATOLL_REQUIRE_COMPILED"] == "1"
    assert "ATOLL_DISABLE" not in child_env
    assert os.environ["ATOLL_DISABLE"] == "parent-disable"
    assert os.environ["ATOLL_REQUIRE_COMPILED"] == "parent-require"
    assert evidence.returncode == 0
    assert evidence.stdout == "out"
    assert evidence.stderr == "err"
    assert evidence.duration_seconds == pytest.approx(1.5)


def test_baseline_command_sets_disable_and_clears_require_compiled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_env: dict[str, str] = {}
    monkeypatch.setenv("ATOLL_REQUIRE_COMPILED", "parent")
    monkeypatch.setattr(performance, "_perf_counter", _clock([0.5]))

    def fake_run(
        invocation: SubprocessInvocationView,
    ) -> subprocess.CompletedProcess[str]:
        captured_env.update(invocation.env)
        return subprocess.CompletedProcess(invocation.command, 0, "", "")

    monkeypatch.setattr(performance, "_run_subprocess", fake_run)

    run_performance_command(
        ("python", "bench.py"),
        project_root=tmp_path,
        payload_root=tmp_path / "payload",
        mode="baseline",
    )

    assert captured_env["ATOLL_DISABLE"] == "1"
    assert "ATOLL_REQUIRE_COMPILED" not in captured_env


def test_benchmark_gate_runs_warmups_then_alternating_sample_pairs_and_medians(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    durations = [9.0, 8.0, 2.0, 1.0, 1.0, 4.0, 6.0, 3.0]
    calls: list[str] = []
    progress_events: list[BenchmarkProgress] = []
    monkeypatch.setattr(performance, "_perf_counter", _clock(durations))

    def fake_run(
        invocation: SubprocessInvocationView,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(_mode_from_env(invocation.env))
        return subprocess.CompletedProcess(invocation.command, 0, "", "")

    monkeypatch.setattr(performance, "_run_subprocess", fake_run)

    result = run_benchmark_gate(
        BenchmarkGateConfig(
            command=("python", "bench.py"),
            warmups=1,
            samples=3,
            minimum_speedup=1.5,
        ),
        project_root=tmp_path,
        baseline_payload_root=tmp_path / "baseline",
        compiled_payload_root=tmp_path / "compiled",
        progress=progress_events.append,
    )

    assert calls == [
        "baseline",
        "compiled",
        "baseline",
        "compiled",
        "compiled",
        "baseline",
        "baseline",
        "compiled",
    ]
    assert result.status == "passed"
    assert result.baseline_median_seconds == pytest.approx(4.0)
    assert result.compiled_median_seconds == pytest.approx(1.0)
    assert result.speedup == pytest.approx(4.0)
    assert [event.phase for event in progress_events] == [
        "warmup",
        "warmup",
        "sample",
        "sample",
        "sample",
        "sample",
        "sample",
        "sample",
    ]
    assert [(event.sample_index, event.mode) for event in progress_events[2:]] == [
        (1, "baseline"),
        (1, "compiled"),
        (2, "compiled"),
        (2, "baseline"),
        (3, "baseline"),
        (3, "compiled"),
    ]


def test_benchmark_gate_rejects_noisy_medians(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(performance, "_perf_counter", _clock([0.30, 0.10, 0.20, 0.40]))
    monkeypatch.setattr(performance, "_run_subprocess", _successful_run)

    result = run_benchmark_gate(
        BenchmarkGateConfig(
            command=("python", "bench.py"),
            warmups=0,
            samples=2,
            minimum_speedup=1.1,
        ),
        project_root=tmp_path,
        baseline_payload_root=tmp_path / "baseline",
        compiled_payload_root=tmp_path / "compiled",
    )

    assert result.status == "invalid"
    assert "too noisy" in result.reason
    assert result.baseline_median_seconds == pytest.approx(0.35)
    assert result.compiled_median_seconds == pytest.approx(0.15)
    assert result.speedup is None


def test_benchmark_gate_rejects_speedup_below_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(performance, "_perf_counter", _clock([1.0, 0.8, 0.9, 1.1]))
    monkeypatch.setattr(performance, "_run_subprocess", _successful_run)

    result = run_benchmark_gate(
        BenchmarkGateConfig(
            command=("python", "bench.py"),
            warmups=0,
            samples=2,
            minimum_speedup=1.5,
        ),
        project_root=tmp_path,
        baseline_payload_root=tmp_path / "baseline",
        compiled_payload_root=tmp_path / "compiled",
    )

    assert result.status == "not-profitable"
    assert result.baseline_median_seconds == pytest.approx(1.05)
    assert result.compiled_median_seconds == pytest.approx(0.85)
    assert result.speedup == pytest.approx(1.05 / 0.85)
    assert "below threshold" in result.reason


def test_benchmark_gate_marks_nonzero_subprocess_exit_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    monkeypatch.setattr(performance, "_perf_counter", _clock([1.0, 1.0, 1.0]))

    def fake_run(
        invocation: SubprocessInvocationView,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(
            invocation.command,
            2 if calls == FAILED_CALL_INDEX else 0,
            "",
            "boom",
        )

    monkeypatch.setattr(performance, "_run_subprocess", fake_run)

    result = run_benchmark_gate(
        BenchmarkGateConfig(
            command=("python", "bench.py"),
            warmups=1,
            samples=2,
            minimum_speedup=1.1,
        ),
        project_root=tmp_path,
        baseline_payload_root=tmp_path / "baseline",
        compiled_payload_root=tmp_path / "compiled",
    )

    assert result.status == "invalid"
    assert result.reason == "sample benchmark command exited with status 2 in baseline mode"
    assert len(result.warmups) == EXPECTED_WARMUP_RUNS
    assert len(result.samples) == 1
    assert result.samples[0].stderr == "boom"


def test_benchmark_gate_stops_after_failed_warmup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(performance, "_perf_counter", _clock([1.0]))

    def failing_run(
        invocation: SubprocessInvocationView,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(invocation.command, 3, "", "warmup failed")

    monkeypatch.setattr(performance, "_run_subprocess", failing_run)

    result = run_benchmark_gate(
        BenchmarkGateConfig(
            command=("python", "bench.py"),
            warmups=1,
            samples=2,
            minimum_speedup=1.1,
        ),
        project_root=tmp_path,
        baseline_payload_root=tmp_path / "baseline",
        compiled_payload_root=tmp_path / "compiled",
    )

    assert result.status == "invalid"
    assert result.succeeded is False
    assert result.reason == "warmup benchmark command exited with status 3 in baseline mode"
    assert len(result.warmups) == 1
    assert result.samples == ()


def test_benchmark_gate_returns_unbenchmarked_evidence_when_command_is_absent(
    tmp_path: Path,
) -> None:
    result = run_benchmark_gate(
        BenchmarkGateConfig(command=None, warmups=1, samples=3, minimum_speedup=1.2),
        project_root=tmp_path,
        baseline_payload_root=tmp_path / "baseline",
        compiled_payload_root=tmp_path / "compiled",
    )

    assert result.status == "unbenchmarked"
    assert result.reason == "no benchmark command configured"
    assert result.warmups == ()
    assert result.samples == ()
    assert result.baseline_median_seconds is None
    assert result.compiled_median_seconds is None
    assert result.speedup is None


def _successful_run(
    invocation: SubprocessInvocationView,
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(invocation.command, 0, "", "")


def _clock(durations: Iterable[float]) -> object:
    duration_iter = iter(durations)
    current = 0.0
    pending_end: float | None = None

    def perf_counter() -> float:
        nonlocal current, pending_end
        if pending_end is None:
            pending_end = current + next(duration_iter)
            return current
        current = pending_end
        pending_end = None
        return current

    return perf_counter


def _mode_from_env(env: dict[str, str]) -> str:
    if env.get("ATOLL_DISABLE") == "1":
        return "baseline"
    if env.get("ATOLL_REQUIRE_COMPILED") == "1":
        return "compiled"
    raise AssertionError("missing Atoll runtime mode")
