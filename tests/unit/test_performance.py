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
    monkeypatch.setenv("ATOLL_REQUIRE_OPTIMIZED", "parent-optimized")
    monkeypatch.setenv("ATOLL_REGION_ALLOWLIST", "parent-region")
    monkeypatch.setenv("ATOLL_VARIANT_ALLOWLIST", "parent-variant")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "parent-bytecode")
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
    assert child_env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert "ATOLL_DISABLE" not in child_env
    assert "ATOLL_REQUIRE_OPTIMIZED" not in child_env
    assert "ATOLL_REGION_ALLOWLIST" not in child_env
    assert "ATOLL_VARIANT_ALLOWLIST" not in child_env
    assert os.environ["ATOLL_DISABLE"] == "parent-disable"
    assert os.environ["ATOLL_REQUIRE_COMPILED"] == "parent-require"
    assert os.environ["ATOLL_REQUIRE_OPTIMIZED"] == "parent-optimized"
    assert os.environ["ATOLL_REGION_ALLOWLIST"] == "parent-region"
    assert os.environ["ATOLL_VARIANT_ALLOWLIST"] == "parent-variant"
    assert os.environ["PYTHONDONTWRITEBYTECODE"] == "parent-bytecode"
    assert evidence.returncode == 0
    assert evidence.stdout == "out"
    assert evidence.stderr == "err"
    assert evidence.duration_seconds == pytest.approx(1.5)


def test_run_performance_command_transports_sorted_region_allowlist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Region allowlists force compiled transport without mutating parent state."""
    captured_env: dict[str, str] = {}
    monkeypatch.setenv("ATOLL_DISABLE", "parent-disable")
    monkeypatch.setenv("ATOLL_REQUIRE_COMPILED", "parent-require")
    monkeypatch.setenv("ATOLL_REGION_ALLOWLIST", "parent-region")
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
        region_allowlist=frozenset(("region-b", "region-a")),
    )

    assert "ATOLL_DISABLE" not in captured_env
    assert captured_env["ATOLL_REQUIRE_COMPILED"] == "1"
    assert captured_env["ATOLL_REGION_ALLOWLIST"] == "region-a\nregion-b"
    assert os.environ["ATOLL_DISABLE"] == "parent-disable"
    assert os.environ["ATOLL_REQUIRE_COMPILED"] == "parent-require"
    assert os.environ["ATOLL_REGION_ALLOWLIST"] == "parent-region"


def test_run_performance_command_transports_variant_allowlist_independently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Variant trials route dispatch candidates without inventing region selection."""
    captured_env: dict[str, str] = {}
    monkeypatch.setenv("ATOLL_DISABLE", "parent-disable")
    monkeypatch.setenv("ATOLL_REGION_ALLOWLIST", "parent-region")
    monkeypatch.setenv("ATOLL_VARIANT_ALLOWLIST", "parent-variant")
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
        variant_allowlist=frozenset(("variant-b", "variant-a")),
    )

    assert "ATOLL_DISABLE" not in captured_env
    assert captured_env["ATOLL_REQUIRE_COMPILED"] == "1"
    assert "ATOLL_REGION_ALLOWLIST" not in captured_env
    assert captured_env["ATOLL_VARIANT_ALLOWLIST"] == "variant-a\nvariant-b"
    assert os.environ["ATOLL_VARIANT_ALLOWLIST"] == "parent-variant"


def test_run_performance_command_requires_source_optimization_without_parent_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Source trials require their fast path only inside the child process."""
    captured_env: dict[str, str] = {}
    monkeypatch.setenv("ATOLL_DISABLE", "parent-disable")
    monkeypatch.setenv("ATOLL_REQUIRE_OPTIMIZED", "parent-optimized")
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
        mode="compiled",
        require_optimized=True,
    )

    assert "ATOLL_DISABLE" not in captured_env
    assert captured_env["ATOLL_REQUIRE_COMPILED"] == "1"
    assert captured_env["ATOLL_REQUIRE_OPTIMIZED"] == "1"
    assert os.environ["ATOLL_DISABLE"] == "parent-disable"
    assert os.environ["ATOLL_REQUIRE_OPTIMIZED"] == "parent-optimized"


def test_run_performance_command_enables_source_without_regions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Source activation does not require a compiled-region allowlist."""
    captured_env: dict[str, str] = {}
    monkeypatch.setenv("ATOLL_DISABLE", "parent-disable")
    monkeypatch.setenv("ATOLL_REQUIRE_COMPILED", "parent-compiled")
    monkeypatch.setenv("ATOLL_REGION_ALLOWLIST", "parent-region")
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
        require_optimized=True,
    )

    assert captured_env["ATOLL_REQUIRE_OPTIMIZED"] == "1"
    assert "ATOLL_REGION_ALLOWLIST" not in captured_env
    assert "ATOLL_REQUIRE_COMPILED" not in captured_env
    assert "ATOLL_DISABLE" not in captured_env


def test_run_performance_command_transports_empty_region_allowlist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty allowlist is explicit and distinct from omitting the allowlist."""
    captured_env: dict[str, str] = {}
    monkeypatch.setenv("ATOLL_DISABLE", "parent-disable")
    monkeypatch.setenv("ATOLL_REGION_ALLOWLIST", "parent-region")
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
        mode="compiled",
        region_allowlist=frozenset(),
    )

    assert "ATOLL_DISABLE" not in captured_env
    assert captured_env["ATOLL_REQUIRE_COMPILED"] == "1"
    assert captured_env["ATOLL_REGION_ALLOWLIST"] == ""
    assert "ATOLL_REQUIRE_OPTIMIZED" not in captured_env


def test_baseline_command_sets_disable_and_clears_require_compiled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_env: dict[str, str] = {}
    monkeypatch.setenv("ATOLL_REQUIRE_COMPILED", "parent")
    monkeypatch.setenv("ATOLL_REQUIRE_OPTIMIZED", "parent-optimized")
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
    assert "ATOLL_REQUIRE_OPTIMIZED" not in captured_env


def test_run_performance_command_preserves_baseline_compiled_compatibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normal baseline and compiled modes keep their legacy environment flags."""
    captured_envs: list[dict[str, str]] = []
    monkeypatch.setenv("ATOLL_DISABLE", "parent-disable")
    monkeypatch.setenv("ATOLL_REQUIRE_COMPILED", "parent-compiled")
    monkeypatch.setenv("ATOLL_REQUIRE_OPTIMIZED", "parent-optimized")
    monkeypatch.setenv("ATOLL_REGION_ALLOWLIST", "parent-region")
    monkeypatch.setattr(performance, "_perf_counter", _clock([0.5, 0.5]))

    def fake_run(
        invocation: SubprocessInvocationView,
    ) -> subprocess.CompletedProcess[str]:
        captured_envs.append(dict(invocation.env))
        return subprocess.CompletedProcess(invocation.command, 0, "", "")

    monkeypatch.setattr(performance, "_run_subprocess", fake_run)

    for mode in ("baseline", "compiled"):
        run_performance_command(
            ("python", "bench.py"),
            project_root=tmp_path,
            payload_root=tmp_path / "payload",
            mode=mode,
        )

    baseline_env, compiled_env = captured_envs
    assert baseline_env["ATOLL_DISABLE"] == "1"
    assert "ATOLL_REQUIRE_COMPILED" not in baseline_env
    assert "ATOLL_REQUIRE_OPTIMIZED" not in baseline_env
    assert "ATOLL_REGION_ALLOWLIST" not in baseline_env
    assert "ATOLL_DISABLE" not in compiled_env
    assert compiled_env["ATOLL_REQUIRE_COMPILED"] == "1"
    assert "ATOLL_REQUIRE_OPTIMIZED" not in compiled_env
    assert "ATOLL_REGION_ALLOWLIST" not in compiled_env


def test_benchmark_gate_passes_distinct_region_allowlists_to_each_side(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[str, str, str]] = []
    monkeypatch.setattr(performance, "_perf_counter", _clock([2.0, 1.0, 1.0, 2.0]))

    def fake_run(
        invocation: SubprocessInvocationView,
    ) -> subprocess.CompletedProcess[str]:
        allowlist = invocation.env["ATOLL_REGION_ALLOWLIST"]
        require_compiled = invocation.env["ATOLL_REQUIRE_COMPILED"]
        captured.append((str(invocation.cwd), require_compiled, allowlist))
        return subprocess.CompletedProcess(invocation.command, 0, "", "")

    monkeypatch.setattr(performance, "_run_subprocess", fake_run)

    result = run_benchmark_gate(
        BenchmarkGateConfig(
            command=("python", "bench.py"),
            warmups=0,
            samples=2,
            minimum_speedup=1.01,
        ),
        project_root=tmp_path,
        baseline_payload_root=tmp_path / "baseline",
        compiled_payload_root=tmp_path / "compiled",
        baseline_region_allowlist=frozenset(("baseline-b", "baseline-a")),
        compiled_region_allowlist=frozenset(("compiled-a",)),
    )

    assert result.status == "passed"
    assert captured == [
        (str(tmp_path.resolve()), "1", "baseline-a\nbaseline-b"),
        (str(tmp_path.resolve()), "1", "compiled-a"),
        (str(tmp_path.resolve()), "1", "compiled-a"),
        (str(tmp_path.resolve()), "1", "baseline-a\nbaseline-b"),
    ]


def test_benchmark_gate_passes_distinct_variant_allowlists_to_each_side(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Marginal gates independently select dispatcher variants on each arm."""
    captured: list[str] = []
    monkeypatch.setattr(performance, "_perf_counter", _clock([2.0, 1.0]))

    def fake_run(
        invocation: SubprocessInvocationView,
    ) -> subprocess.CompletedProcess[str]:
        captured.append(invocation.env["ATOLL_VARIANT_ALLOWLIST"])
        assert "ATOLL_REGION_ALLOWLIST" not in invocation.env
        return subprocess.CompletedProcess(invocation.command, 0, "", "")

    monkeypatch.setattr(performance, "_run_subprocess", fake_run)

    result = run_benchmark_gate(
        BenchmarkGateConfig(
            command=("python", "bench.py"),
            warmups=0,
            samples=1,
            minimum_speedup=1.01,
        ),
        project_root=tmp_path,
        baseline_payload_root=tmp_path / "baseline",
        compiled_payload_root=tmp_path / "compiled",
        baseline_variant_allowlist=frozenset(("baseline-b", "baseline-a")),
        compiled_variant_allowlist=frozenset(("compiled-a",)),
    )

    assert result.status == "passed"
    assert captured == ["baseline-a\nbaseline-b", "compiled-a"]


def test_benchmark_gate_requires_source_optimization_independently_per_arm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source-only arm stays active without pretending to be native transport."""
    captured: list[dict[str, str]] = []
    monkeypatch.setattr(performance, "_perf_counter", _clock([2.0, 1.0]))

    def fake_run(
        invocation: SubprocessInvocationView,
    ) -> subprocess.CompletedProcess[str]:
        captured.append(dict(invocation.env))
        return subprocess.CompletedProcess(invocation.command, 0, "", "")

    monkeypatch.setattr(performance, "_run_subprocess", fake_run)

    result = run_benchmark_gate(
        BenchmarkGateConfig(
            command=("python", "bench.py"),
            warmups=0,
            samples=1,
            minimum_speedup=1.01,
        ),
        project_root=tmp_path,
        baseline_payload_root=tmp_path / "source-only",
        compiled_payload_root=tmp_path / "composed",
        baseline_require_optimized=True,
        compiled_require_optimized=True,
    )

    assert result.status == "passed"
    baseline_env, compiled_env = captured
    assert baseline_env["ATOLL_REQUIRE_OPTIMIZED"] == "1"
    assert "ATOLL_DISABLE" not in baseline_env
    assert "ATOLL_REQUIRE_COMPILED" not in baseline_env
    assert compiled_env["ATOLL_REQUIRE_OPTIMIZED"] == "1"
    assert compiled_env["ATOLL_REQUIRE_COMPILED"] == "1"


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


def test_benchmark_gate_accepts_exact_marginal_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A candidate meeting the 1.01 marginal threshold is retained."""
    monkeypatch.setattr(performance, "_perf_counter", _clock([1.01, 1.0]))
    monkeypatch.setattr(performance, "_run_subprocess", _successful_run)

    result = run_benchmark_gate(
        BenchmarkGateConfig(
            command=("python", "bench.py"),
            warmups=0,
            samples=1,
            minimum_speedup=1.01,
        ),
        project_root=tmp_path,
        baseline_payload_root=tmp_path / "payload",
        compiled_payload_root=tmp_path / "payload",
        baseline_region_allowlist=frozenset(),
        compiled_region_allowlist=frozenset({"candidate"}),
    )

    assert result.status == "passed"
    assert result.speedup == pytest.approx(1.01)


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
