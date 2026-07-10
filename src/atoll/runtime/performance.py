"""Run target-project performance gates under Atoll runtime modes.

This module owns Milestone 7 benchmark execution only. It does not parse shell
commands, build native artifacts, or decide which benchmark a project should
use. Callers provide argv tuples, the target project root, and the prepared
payload root; the runner returns immutable evidence from child processes and a
conservative profitability decision.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Literal

RuntimeMode = Literal["baseline", "compiled"]
BenchmarkPhase = Literal["warmup", "sample"]
BenchmarkStatus = Literal["passed", "not-profitable", "invalid", "unbenchmarked"]

_MINIMUM_STABLE_MEDIAN_SECONDS = 0.25


@dataclass(frozen=True, slots=True)
class _SubprocessInvocation:
    command: tuple[str, ...]
    cwd: Path
    env: dict[str, str]
    check: bool
    shell: bool
    capture_output: bool
    text: bool


_perf_counter: Callable[[], float] = time.perf_counter


@dataclass(frozen=True, slots=True)
class CommandRunEvidence:
    """Observed result from one benchmark command invocation.

    `command` is the exact argv tuple passed to `subprocess.run` with
    `shell=False`. `project_root` is used as the child process working directory,
    while `payload_root` is prepended to the child `PYTHONPATH`. The parent
    process environment is never mutated; runtime mode flags are applied only to
    the child environment represented by this run.

    Attributes:
        command: Normalized command argument vector.
        project_root: Root directory of the target Python project.
        payload_root: Unpacked wheel payload used for the command.
        mode: Compilation or runtime mode represented by this record.
        returncode: Child process return code.
        stdout: Captured child process standard output.
        stderr: Captured child process standard error.
        duration_seconds: Elapsed wall-clock duration in seconds.
    """

    command: tuple[str, ...]
    project_root: Path
    payload_root: Path
    mode: RuntimeMode
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float

    @property
    def succeeded(self) -> bool:
        """Return whether the benchmark process exited successfully.

        Returns:
            bool: Whether the child process exited successfully.
        """
        return self.returncode == 0


@dataclass(frozen=True, slots=True)
class BenchmarkProgress:
    """Timing notification for one warmup or measured sample invocation.

    `pair_index` identifies the baseline/compiled pair within the current phase.
    `sample_index` is `None` for warmups and equals `pair_index` for measured
    samples, so callbacks can present human-oriented sample progress without
    inferring it from runtime mode.

    Attributes:
        phase: Warmup or measured benchmark phase.
        pair_index: One-based benchmark pair number.
        sample_index: One-based sample number, or `None` for unnumbered phases.
        mode: Compilation or runtime mode represented by this record.
        duration_seconds: Elapsed wall-clock duration in seconds.
    """

    phase: BenchmarkPhase
    pair_index: int
    sample_index: int | None
    mode: RuntimeMode
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class BenchmarkGateConfig:
    """Configuration for one benchmark gate invocation.

    `command` is an argv tuple or `None` when the caller has no benchmark to
    run. Warmup pairs are excluded from medians, sample pairs are measured, and
    `minimum_speedup` is the required baseline-median / compiled-median ratio.

    Attributes:
        command: Normalized command argument vector.
        warmups: Number of unmeasured baseline/compiled pairs.
        samples: Number of measured baseline/compiled pairs.
        minimum_speedup: Smallest acceptable compiled-to-baseline speedup ratio.
    """

    command: tuple[str, ...] | None
    warmups: int
    samples: int
    minimum_speedup: float

    def __post_init__(self) -> None:
        """Reject invalid benchmark policy before launching subprocesses.

        Raises:
            ValueError: If the command is empty, counts are invalid, or the speedup threshold is
                not positive.
        """
        if self.command is not None and (
            not self.command or any(not part.strip() for part in self.command)
        ):
            raise ValueError("benchmark command must be a non-empty argv tuple")
        if self.warmups < 0:
            raise ValueError("benchmark warmups must be at least 0")
        if self.samples < 1:
            raise ValueError("benchmark samples must be at least 1")
        if self.minimum_speedup <= 0:
            raise ValueError("minimum speedup must be greater than 0")


@dataclass(frozen=True, slots=True)
class BenchmarkGateResult:
    """Profitability decision and process evidence for one benchmark gate.

    `status` is `passed` when the median baseline duration divided by the median
    compiled duration meets `minimum_speedup`, `not-profitable` when it is stable
    but too small, `invalid` when execution failed or timings are too noisy, and
    `unbenchmarked` when no command was configured. Medians and speedup are
    omitted when no reliable measured sample set exists.

    Attributes:
        status: Passed, not-profitable, invalid, or unbenchmarked gate status.
        reason: Concrete measurement or execution evidence supporting the status.
        minimum_speedup: Smallest acceptable compiled-to-baseline speedup ratio.
        baseline_median_seconds: Median elapsed time for baseline samples, when measured.
        compiled_median_seconds: Median elapsed time for compiled samples, when measured.
        speedup: Baseline-to-compiled median ratio, when measured.
        warmups: Unmeasured benchmark warmup command evidence.
        samples: Measured benchmark command evidence.
    """

    status: BenchmarkStatus
    reason: str
    minimum_speedup: float
    baseline_median_seconds: float | None
    compiled_median_seconds: float | None
    speedup: float | None
    warmups: tuple[CommandRunEvidence, ...]
    samples: tuple[CommandRunEvidence, ...]

    @property
    def succeeded(self) -> bool:
        """Return whether the benchmark gate accepted the compiled payload.

        Returns:
            bool: Whether the benchmark gate status is exactly `passed`.
        """
        return self.status == "passed"


ProgressCallback = Callable[[BenchmarkProgress], None]


@dataclass(frozen=True, slots=True)
class _BenchmarkExecutionContext:
    command: tuple[str, ...]
    project_root: Path
    baseline_payload_root: Path
    compiled_payload_root: Path
    progress: ProgressCallback | None


def run_performance_command(
    command: tuple[str, ...],
    *,
    project_root: Path,
    payload_root: Path,
    mode: RuntimeMode,
) -> CommandRunEvidence:
    """Execute one argv command under a controlled Atoll runtime mode.

    The child process runs in `project_root`, captures stdout and stderr as text,
    and receives a copy of the current environment with `payload_root` prepended
    to `PYTHONPATH`. Baseline mode sets `ATOLL_DISABLE=1`; compiled mode sets
    `ATOLL_REQUIRE_COMPILED=1` and clears any inherited disable flag so the
    compiled routing path is exercised.

    Args:
        command: Command to validate, execute, or benchmark.
        project_root: Root directory of the target Python project.
        payload_root: Wheel payload root placed first on the child process import path.
        mode: Runtime mode selecting interpreted or compiled import behavior.

    Returns:
        CommandRunEvidence: Captured process output, status, mode, and elapsed duration.
    """
    resolved_project_root = project_root.resolve()
    resolved_payload_root = payload_root.resolve()
    child_env = _runtime_environment(payload_root=resolved_payload_root, mode=mode)
    started = _perf_counter()
    completed = _run_subprocess(
        _SubprocessInvocation(
            command=command,
            cwd=resolved_project_root,
            env=child_env,
            check=False,
            shell=False,
            capture_output=True,
            text=True,
        )
    )
    duration = _perf_counter() - started
    return CommandRunEvidence(
        command=command,
        project_root=resolved_project_root,
        payload_root=resolved_payload_root,
        mode=mode,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        duration_seconds=duration,
    )


def run_benchmark_gate(
    config: BenchmarkGateConfig,
    *,
    project_root: Path,
    baseline_payload_root: Path,
    compiled_payload_root: Path,
    progress: ProgressCallback | None = None,
) -> BenchmarkGateResult:
    """Run warmups and measured sample pairs, then decide profitability.

    Warmups and measured samples each execute baseline/compiled pairs. Pair
    order alternates by pair index to reduce ordering bias, and only measured
    sample durations contribute to medians. Any non-zero subprocess exit makes
    the gate invalid. Very short medians are rejected as noisy because Atoll
    cannot make a credible performance decision from sub-quarter-second samples.

    Args:
        config: Resolved configuration governing the requested operation.
        project_root: Root directory of the target Python project.
        baseline_payload_root: Unpacked baseline wheel payload used for interpreted measurements.
        compiled_payload_root: Unpacked compiled wheel payload used for native measurements.
        progress: Optional callback notified after each benchmark phase.

    Returns:
        BenchmarkGateResult: Paired baseline and compiled samples plus the derived speedup decision.
    """
    if config.command is None:
        return BenchmarkGateResult(
            status="unbenchmarked",
            reason="no benchmark command configured",
            minimum_speedup=config.minimum_speedup,
            baseline_median_seconds=None,
            compiled_median_seconds=None,
            speedup=None,
            warmups=(),
            samples=(),
        )
    context = _BenchmarkExecutionContext(
        command=config.command,
        project_root=project_root,
        baseline_payload_root=baseline_payload_root,
        compiled_payload_root=compiled_payload_root,
        progress=progress,
    )

    warmup_runs, failure = _run_phase(
        context,
        phase="warmup",
        pairs=config.warmups,
    )
    if failure is not None:
        return _invalid_result(
            reason=_failure_reason(failure, phase="warmup"),
            minimum_speedup=config.minimum_speedup,
            warmups=warmup_runs,
            samples=(),
        )

    sample_runs, failure = _run_phase(
        context,
        phase="sample",
        pairs=config.samples,
    )
    if failure is not None:
        return _invalid_result(
            reason=_failure_reason(failure, phase="sample"),
            minimum_speedup=config.minimum_speedup,
            warmups=warmup_runs,
            samples=sample_runs,
        )

    baseline_median = median(run.duration_seconds for run in sample_runs if run.mode == "baseline")
    compiled_median = median(run.duration_seconds for run in sample_runs if run.mode == "compiled")
    if (
        baseline_median < _MINIMUM_STABLE_MEDIAN_SECONDS
        or compiled_median < _MINIMUM_STABLE_MEDIAN_SECONDS
    ):
        return BenchmarkGateResult(
            status="invalid",
            reason=(
                "benchmark medians are too noisy: "
                f"baseline={baseline_median:.3f}s compiled={compiled_median:.3f}s"
            ),
            minimum_speedup=config.minimum_speedup,
            baseline_median_seconds=baseline_median,
            compiled_median_seconds=compiled_median,
            speedup=None,
            warmups=warmup_runs,
            samples=sample_runs,
        )

    speedup = baseline_median / compiled_median
    if speedup >= config.minimum_speedup:
        return BenchmarkGateResult(
            status="passed",
            reason=(
                f"compiled median speedup {speedup:.3f} "
                f"meets threshold {config.minimum_speedup:.3f}"
            ),
            minimum_speedup=config.minimum_speedup,
            baseline_median_seconds=baseline_median,
            compiled_median_seconds=compiled_median,
            speedup=speedup,
            warmups=warmup_runs,
            samples=sample_runs,
        )
    return BenchmarkGateResult(
        status="not-profitable",
        reason=(
            f"compiled median speedup {speedup:.3f} is below threshold {config.minimum_speedup:.3f}"
        ),
        minimum_speedup=config.minimum_speedup,
        baseline_median_seconds=baseline_median,
        compiled_median_seconds=compiled_median,
        speedup=speedup,
        warmups=warmup_runs,
        samples=sample_runs,
    )


def _runtime_environment(*, payload_root: Path, mode: RuntimeMode) -> dict[str, str]:
    child_env = dict(os.environ)
    existing_pythonpath = tuple(
        path for path in child_env.get("PYTHONPATH", "").split(os.pathsep) if path
    )
    child_env["PYTHONPATH"] = os.pathsep.join((str(payload_root), *existing_pythonpath))
    if mode == "baseline":
        child_env["ATOLL_DISABLE"] = "1"
        child_env.pop("ATOLL_REQUIRE_COMPILED", None)
    else:
        child_env.pop("ATOLL_DISABLE", None)
        child_env["ATOLL_REQUIRE_COMPILED"] = "1"
    return child_env


def _run_subprocess(invocation: _SubprocessInvocation) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        invocation.command,
        cwd=invocation.cwd,
        env=invocation.env,
        check=invocation.check,
        shell=invocation.shell,
        capture_output=invocation.capture_output,
        text=invocation.text,
    )


def _run_phase(
    context: _BenchmarkExecutionContext,
    *,
    phase: BenchmarkPhase,
    pairs: int,
) -> tuple[tuple[CommandRunEvidence, ...], CommandRunEvidence | None]:
    runs: list[CommandRunEvidence] = []
    for pair_index in range(pairs):
        for mode in _pair_order(pair_index):
            run = run_performance_command(
                context.command,
                project_root=context.project_root,
                payload_root=(
                    context.baseline_payload_root
                    if mode == "baseline"
                    else context.compiled_payload_root
                ),
                mode=mode,
            )
            runs.append(run)
            if context.progress is not None:
                context.progress(
                    BenchmarkProgress(
                        phase=phase,
                        pair_index=pair_index,
                        sample_index=pair_index if phase == "sample" else None,
                        mode=mode,
                        duration_seconds=run.duration_seconds,
                    )
                )
            if not run.succeeded:
                return tuple(runs), run
    return tuple(runs), None


def _pair_order(pair_index: int) -> tuple[RuntimeMode, RuntimeMode]:
    if pair_index % 2 == 0:
        return ("baseline", "compiled")
    return ("compiled", "baseline")


def _invalid_result(
    *,
    reason: str,
    minimum_speedup: float,
    warmups: tuple[CommandRunEvidence, ...],
    samples: tuple[CommandRunEvidence, ...],
) -> BenchmarkGateResult:
    return BenchmarkGateResult(
        status="invalid",
        reason=reason,
        minimum_speedup=minimum_speedup,
        baseline_median_seconds=None,
        compiled_median_seconds=None,
        speedup=None,
        warmups=warmups,
        samples=samples,
    )


def _failure_reason(run: CommandRunEvidence, *, phase: BenchmarkPhase) -> str:
    return f"{phase} benchmark command exited with status {run.returncode} in {run.mode} mode"
