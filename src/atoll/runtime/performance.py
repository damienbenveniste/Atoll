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
from typing import Literal, TypedDict, Unpack

from atoll.optimization_policy import assess_speedup, validate_acceleration_threshold

RuntimeMode = Literal["baseline", "compiled"]
BenchmarkPhase = Literal["warmup", "sample"]
BenchmarkStatus = Literal["passed", "not-profitable", "invalid", "unbenchmarked"]


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
            ValueError: If the command is empty, counts are invalid, or the speedup threshold does
                not require acceleration.
        """
        if self.command is not None and (
            not self.command or any(not part.strip() for part in self.command)
        ):
            raise ValueError("benchmark command must be a non-empty argv tuple")
        if self.warmups < 0:
            raise ValueError("benchmark warmups must be at least 0")
        if self.samples < 1:
            raise ValueError("benchmark samples must be at least 1")
        validate_acceleration_threshold(self.minimum_speedup, field="minimum speedup")


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


class _BenchmarkGateOptions(TypedDict, total=False):
    progress: ProgressCallback | None
    baseline_region_allowlist: frozenset[str] | None
    compiled_region_allowlist: frozenset[str] | None
    baseline_variant_allowlist: frozenset[str] | None
    compiled_variant_allowlist: frozenset[str] | None


class _PerformanceCommandOptions(TypedDict, total=False):
    region_allowlist: frozenset[str] | None
    variant_allowlist: frozenset[str] | None
    require_optimized: bool


@dataclass(frozen=True, slots=True)
class _RuntimeActivation:
    """Resolved Atoll runtime switches for one child process.

    The transport mode and source optimization requirement are intentionally
    independent. A compiled-region allowlist always selects compiled transport,
    but source optimization can be required with or without a region allowlist.

    Attributes:
        mode: Caller-visible runtime mode recorded in command evidence.
        compiled_region_allowlist: Region IDs enabled for compiled transport, or
            `None` when the child should not receive an allowlist.
        compiled_variant_allowlist: Native variant IDs enabled for dispatcher routing, or
            `None` when all loaded variants may participate.
        source_optimization_required: Whether the child must use generated source
            optimization instead of silently falling back.
    """

    mode: RuntimeMode
    compiled_region_allowlist: frozenset[str] | None
    compiled_variant_allowlist: frozenset[str] | None
    source_optimization_required: bool

    @property
    def uses_compiled_transport(self) -> bool:
        """Return whether the child should exercise compiled runtime routing.

        Returns:
            bool: Whether compiled transport is required by either the runtime
                mode or the presence of an explicit region allowlist.
        """
        return (
            self.mode == "compiled"
            or self.compiled_region_allowlist is not None
            or self.compiled_variant_allowlist is not None
        )

    @classmethod
    def from_command_options(
        cls,
        *,
        mode: RuntimeMode,
        region_allowlist: frozenset[str] | None,
        variant_allowlist: frozenset[str] | None,
        require_optimized: bool,
    ) -> _RuntimeActivation:
        """Create an immutable activation from legacy command options.

        Args:
            mode: Baseline or compiled mode requested by the existing caller.
            region_allowlist: Optional compiled-region allowlist for the child.
            variant_allowlist: Optional native dispatcher-variant allowlist for the child.
            require_optimized: Whether generated source optimization is required.

        Returns:
            _RuntimeActivation: Resolved runtime switches for environment assembly.
        """
        return cls(
            mode=mode,
            compiled_region_allowlist=region_allowlist,
            compiled_variant_allowlist=variant_allowlist,
            source_optimization_required=require_optimized,
        )


@dataclass(frozen=True, slots=True)
class _BenchmarkExecutionContext:
    command: tuple[str, ...]
    project_root: Path
    baseline_payload_root: Path
    compiled_payload_root: Path
    baseline_region_allowlist: frozenset[str] | None
    compiled_region_allowlist: frozenset[str] | None
    baseline_variant_allowlist: frozenset[str] | None
    compiled_variant_allowlist: frozenset[str] | None
    progress: ProgressCallback | None


def run_performance_command(
    command: tuple[str, ...],
    *,
    project_root: Path,
    payload_root: Path,
    mode: RuntimeMode,
    **options: Unpack[_PerformanceCommandOptions],
) -> CommandRunEvidence:
    """Execute one argv command under a controlled Atoll runtime mode.

    The child process runs in `project_root`, captures stdout and stderr as text,
    and receives a copy of the current environment with `payload_root` prepended
    to `PYTHONPATH`. Baseline mode sets `ATOLL_DISABLE=1`; compiled mode sets
    `ATOLL_REQUIRE_COMPILED=1` and clears any inherited disable flag so the
    compiled routing path is exercised. When `region_allowlist` is provided, the
    child uses the compiled transport path regardless of `mode`, receives a
    deterministic newline-separated `ATOLL_REGION_ALLOWLIST`, and treats an
    empty allowlist as an explicit request to allow no regions. When no
    allowlist is provided, inherited region allowlist state is cleared.
    `require_optimized=True` independently requires a generated source fast
    path and clears inherited source-optimization state for all other runs.

    Args:
        command: Command to validate, execute, or benchmark.
        project_root: Root directory of the target Python project.
        payload_root: Wheel payload root placed first on the child process import path.
        mode: Runtime mode selecting interpreted or compiled import behavior.
        **options: Optional compiled-region allowlist and source fast-path requirement.

    Returns:
        CommandRunEvidence: Captured process output, status, mode, and elapsed duration.
    """
    resolved_project_root = project_root.resolve()
    resolved_payload_root = payload_root.resolve()
    region_allowlist = options.get("region_allowlist")
    variant_allowlist = options.get("variant_allowlist")
    require_optimized = options.get("require_optimized", False)
    child_env = _runtime_environment(
        payload_root=resolved_payload_root,
        activation=_RuntimeActivation.from_command_options(
            mode=mode,
            region_allowlist=region_allowlist,
            variant_allowlist=variant_allowlist,
            require_optimized=require_optimized,
        ),
    )
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
    **options: Unpack[_BenchmarkGateOptions],
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
        **options: Optional benchmark controls. `progress` receives phase notifications,
            `baseline_region_allowlist` exposes region IDs to baseline-side child runs,
            and `compiled_region_allowlist` exposes region IDs to compiled-side child runs.

    Returns:
        BenchmarkGateResult: Paired baseline and compiled samples plus the derived speedup decision.

    Raises:
        AssertionError: If a stable policy assessment omits its required speedup ratio.
    """
    _reject_unexpected_benchmark_options(options)
    progress = options.get("progress")
    baseline_region_allowlist = options.get("baseline_region_allowlist")
    compiled_region_allowlist = options.get("compiled_region_allowlist")
    baseline_variant_allowlist = options.get("baseline_variant_allowlist")
    compiled_variant_allowlist = options.get("compiled_variant_allowlist")
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
        baseline_region_allowlist=baseline_region_allowlist,
        compiled_region_allowlist=compiled_region_allowlist,
        baseline_variant_allowlist=baseline_variant_allowlist,
        compiled_variant_allowlist=compiled_variant_allowlist,
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
    assessment = assess_speedup(
        baseline_median,
        compiled_median,
        minimum_speedup=config.minimum_speedup,
    )
    if not assessment.stable:
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

    speedup = assessment.speedup
    if speedup is None:
        raise AssertionError("stable benchmark assessment must include speedup")
    if assessment.passed:
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


def _reject_unexpected_benchmark_options(options: _BenchmarkGateOptions) -> None:
    allowed_options = {
        "baseline_region_allowlist",
        "compiled_region_allowlist",
        "baseline_variant_allowlist",
        "compiled_variant_allowlist",
        "progress",
    }
    unexpected_options = set(options) - allowed_options
    if unexpected_options:
        unexpected_option = sorted(unexpected_options)[0]
        raise TypeError(
            f"run_benchmark_gate() got an unexpected keyword argument {unexpected_option!r}"
        )


def _runtime_environment(
    *,
    payload_root: Path,
    activation: _RuntimeActivation,
) -> dict[str, str]:
    child_env = dict(os.environ)
    child_env["PYTHONDONTWRITEBYTECODE"] = "1"
    existing_pythonpath = tuple(
        path for path in child_env.get("PYTHONPATH", "").split(os.pathsep) if path
    )
    child_env["PYTHONPATH"] = os.pathsep.join((str(payload_root), *existing_pythonpath))
    child_env.pop("ATOLL_REGION_ALLOWLIST", None)
    child_env.pop("ATOLL_VARIANT_ALLOWLIST", None)
    child_env.pop("ATOLL_REQUIRE_OPTIMIZED", None)
    if activation.uses_compiled_transport:
        child_env.pop("ATOLL_DISABLE", None)
        child_env["ATOLL_REQUIRE_COMPILED"] = "1"
        if activation.compiled_region_allowlist is not None:
            child_env["ATOLL_REGION_ALLOWLIST"] = "\n".join(
                sorted(activation.compiled_region_allowlist)
            )
        if activation.compiled_variant_allowlist is not None:
            child_env["ATOLL_VARIANT_ALLOWLIST"] = "\n".join(
                sorted(activation.compiled_variant_allowlist)
            )
    elif activation.mode == "baseline":
        child_env["ATOLL_DISABLE"] = "1"
        child_env.pop("ATOLL_REQUIRE_COMPILED", None)
    else:
        raise AssertionError(f"unhandled runtime activation mode: {activation.mode}")
    if activation.source_optimization_required:
        child_env.pop("ATOLL_DISABLE", None)
        child_env["ATOLL_REQUIRE_OPTIMIZED"] = "1"
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
                region_allowlist=(
                    context.baseline_region_allowlist
                    if mode == "baseline"
                    else context.compiled_region_allowlist
                ),
                variant_allowlist=(
                    context.baseline_variant_allowlist
                    if mode == "baseline"
                    else context.compiled_variant_allowlist
                ),
            )
            runs.append(run)
            if context.progress is not None:
                pair_number = pair_index + 1
                context.progress(
                    BenchmarkProgress(
                        phase=phase,
                        pair_index=pair_number,
                        sample_index=pair_number if phase == "sample" else None,
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
