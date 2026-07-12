"""Run generic three-arm execution-plan benchmark trials.

Execution-plan trials compare an interpreted baseline payload, a compiled
payload without a proposed plan, and a compiled payload with the proposed plan
staged. The runner only owns subprocess evidence collection and conservative
performance decisions; callers own plan staging, semantic validation, and report
integration.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Literal, TypedDict, Unpack

from atoll.runtime.performance import CommandRunEvidence, run_performance_command

ExecutionPlanBenchmarkArm = Literal["baseline", "unplanned", "planned"]
ExecutionPlanBenchmarkPhase = Literal["warmup", "sample"]
ExecutionPlanBenchmarkStatus = Literal["passed", "not-profitable", "invalid", "unavailable"]

_MINIMUM_STABLE_MEDIAN_SECONDS = 0.25


@dataclass(frozen=True, slots=True)
class ExecutionPlanBenchmarkConfig:
    """Configuration for one generic execution-plan benchmark trial.

    The benchmark command is an argv tuple executed with `shell=False` through
    the shared runtime command machinery. Exactly one warmup trio runs before
    the measured samples. `minimum_marginal_speedup` gates whether the plan is
    retained for the final payload benchmark. `minimum_overall_speedup` records
    the final promotion target so the trial can explain its provisional overall
    ratio without duplicating the authoritative final gate.

    Attributes:
        plan_id: Stable execution-plan identity represented by the benchmark.
        command: Normalized benchmark command argument vector, or `None` when no
            benchmark is available.
        samples: Number of measured three-arm sample trios.
        minimum_marginal_speedup: Required unplanned-median / planned-median ratio.
        minimum_overall_speedup: Final baseline-median / payload-median promotion target.
    """

    plan_id: str
    command: tuple[str, ...] | None
    samples: int = 7
    minimum_marginal_speedup: float = 1.05
    minimum_overall_speedup: float = 1.10

    def __post_init__(self) -> None:
        """Reject invalid benchmark policy before launching child commands.

        Raises:
            ValueError: If the plan identity, command, sample count, or speedup
                thresholds are invalid.
        """
        if not self.plan_id.strip():
            raise ValueError("execution-plan benchmark plan ID must be non-empty")
        if self.command is not None and (
            not self.command or any(not part.strip() for part in self.command)
        ):
            raise ValueError("execution-plan benchmark command must be a non-empty argv tuple")
        if self.samples < 1:
            raise ValueError("execution-plan benchmark samples must be at least 1")
        if self.minimum_marginal_speedup <= 0:
            raise ValueError("minimum marginal speedup must be greater than 0")
        if self.minimum_overall_speedup <= 0:
            raise ValueError("minimum overall speedup must be greater than 0")


@dataclass(frozen=True, slots=True)
class ExecutionPlanBenchmarkSample:
    """Observed command evidence tagged with its execution-plan benchmark arm.

    Baseline samples execute in baseline runtime mode. Unplanned and planned
    samples both execute in compiled runtime mode and are distinguished by this
    wrapper's arm and payload root.

    Attributes:
        arm: Benchmark arm represented by this command invocation.
        run: Captured subprocess evidence from the shared performance runner.
    """

    arm: ExecutionPlanBenchmarkArm
    run: CommandRunEvidence

    @property
    def succeeded(self) -> bool:
        """Return whether the wrapped command exited successfully.

        Returns:
            bool: Whether the underlying command returned status code zero.
        """
        return self.run.succeeded


@dataclass(frozen=True, slots=True)
class ExecutionPlanBenchmarkProgress:
    """Timing notification for one warmup or measured sample invocation.

    `trio_index` is one-based within the phase. `sample_index` is `None` for the
    single warmup trio and one-based for measured sample trios.

    Attributes:
        phase: Warmup or measured benchmark phase.
        trio_index: One-based trio number within the phase.
        sample_index: One-based sample trio number, or `None` for warmup.
        arm: Benchmark arm represented by the completed command.
        duration_seconds: Elapsed wall-clock duration in seconds.
    """

    phase: ExecutionPlanBenchmarkPhase
    trio_index: int
    sample_index: int | None
    arm: ExecutionPlanBenchmarkArm
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class ExecutionPlanBenchmarkResult:
    """Three-arm execution-plan benchmark decision and command evidence.

    `status` is `passed` only when every command succeeds, measured medians are
    stable, and the planned payload meets the marginal profitability threshold.
    The overall ratio is provisional evidence because the separate final payload
    gate is authoritative for wheel promotion. `not-profitable` means timings
    were stable but the marginal threshold was missed. `invalid` means execution
    failed or timings were too short to trust. `unavailable` means the caller did
    not configure a benchmark command.

    Attributes:
        plan_id: Stable execution-plan identity evaluated by this benchmark.
        status: Benchmark decision.
        reason: Concrete execution or timing evidence supporting the decision.
        minimum_marginal_speedup: Required unplanned-median / planned-median ratio.
        minimum_overall_speedup: Final baseline-median / payload-median promotion target.
        baseline_median_seconds: Median elapsed time for baseline measured samples.
        unplanned_median_seconds: Median elapsed time for unplanned measured samples.
        planned_median_seconds: Median elapsed time for planned measured samples.
        marginal_speedup: Unplanned-median / planned-median ratio.
        overall_speedup: Baseline-median / planned-median ratio.
        warmups: Unmeasured warmup command evidence.
        samples: Measured sample command evidence.
    """

    plan_id: str
    status: ExecutionPlanBenchmarkStatus
    reason: str
    minimum_marginal_speedup: float
    minimum_overall_speedup: float
    baseline_median_seconds: float | None
    unplanned_median_seconds: float | None
    planned_median_seconds: float | None
    marginal_speedup: float | None
    overall_speedup: float | None
    warmups: tuple[ExecutionPlanBenchmarkSample, ...]
    samples: tuple[ExecutionPlanBenchmarkSample, ...]

    @property
    def succeeded(self) -> bool:
        """Return whether the benchmark accepted the planned payload.

        Returns:
            bool: Whether the benchmark status is exactly `passed`.
        """
        return self.status == "passed"


ProgressCallback = Callable[[ExecutionPlanBenchmarkProgress], None]


class _ExecutionPlanBenchmarkOptions(TypedDict, total=False):
    progress: ProgressCallback | None
    baseline_region_allowlist: frozenset[str] | None
    unplanned_region_allowlist: frozenset[str] | None
    planned_region_allowlist: frozenset[str] | None
    baseline_variant_allowlist: frozenset[str] | None
    unplanned_variant_allowlist: frozenset[str] | None
    planned_variant_allowlist: frozenset[str] | None


@dataclass(frozen=True, slots=True)
class _ExecutionPlanBenchmarkContext:
    command: tuple[str, ...]
    project_root: Path
    baseline_payload_root: Path
    unplanned_payload_root: Path
    planned_payload_root: Path
    progress: ProgressCallback | None
    baseline_region_allowlist: frozenset[str] | None
    unplanned_region_allowlist: frozenset[str] | None
    planned_region_allowlist: frozenset[str] | None
    baseline_variant_allowlist: frozenset[str] | None
    unplanned_variant_allowlist: frozenset[str] | None
    planned_variant_allowlist: frozenset[str] | None


def run_execution_plan_benchmark(
    config: ExecutionPlanBenchmarkConfig,
    *,
    project_root: Path,
    baseline_payload_root: Path,
    unplanned_payload_root: Path,
    planned_payload_root: Path,
    **options: Unpack[_ExecutionPlanBenchmarkOptions],
) -> ExecutionPlanBenchmarkResult:
    """Run warmup and measured trios for a generic execution-plan benchmark.

    The benchmark uses one warmup trio followed by `config.samples` measured
    trios. Trio order rotates through all six permutations of baseline,
    unplanned, and planned arms to reduce positional bias. Every child command
    must exit successfully; the first failure returns invalid evidence without
    deriving medians or speedups.

    Args:
        config: Resolved configuration and profitability thresholds.
        project_root: Root directory of the target Python project.
        baseline_payload_root: Payload root for interpreted baseline measurements.
        unplanned_payload_root: Payload root for compiled measurements without the plan.
        planned_payload_root: Payload root for compiled measurements with the plan staged.
        **options: Optional progress callback and per-arm region allowlists.

    Returns:
        ExecutionPlanBenchmarkResult: Three-arm run evidence, medians, speedups, and decision.

    Raises:
        TypeError: If an unsupported per-arm option is supplied.
    """
    _reject_unexpected_options(options)
    if config.command is None:
        return _unavailable_result(config, "no execution-plan benchmark command configured")

    context = _ExecutionPlanBenchmarkContext(
        command=config.command,
        project_root=project_root,
        baseline_payload_root=baseline_payload_root,
        unplanned_payload_root=unplanned_payload_root,
        planned_payload_root=planned_payload_root,
        progress=options.get("progress"),
        baseline_region_allowlist=options.get("baseline_region_allowlist"),
        unplanned_region_allowlist=options.get("unplanned_region_allowlist"),
        planned_region_allowlist=options.get("planned_region_allowlist"),
        baseline_variant_allowlist=options.get("baseline_variant_allowlist"),
        unplanned_variant_allowlist=options.get("unplanned_variant_allowlist"),
        planned_variant_allowlist=options.get("planned_variant_allowlist"),
    )

    warmups, failure = _run_trios(context, phase="warmup", count=1)
    if failure is not None:
        return _invalid_result(
            config=config,
            reason=_failure_reason(failure, phase="warmup"),
            warmups=warmups,
            samples=(),
        )

    samples, failure = _run_trios(context, phase="sample", count=config.samples)
    if failure is not None:
        return _invalid_result(
            config=config,
            reason=_failure_reason(failure, phase="sample"),
            warmups=warmups,
            samples=samples,
        )

    return _decision_result(config=config, warmups=warmups, samples=samples)


def unavailable_execution_plan_benchmark(
    plan_id: str,
    reason: str,
) -> ExecutionPlanBenchmarkResult:
    """Return explicit non-executed evidence for one execution plan.

    Args:
        plan_id: Stable execution-plan identity that could not be benchmarked.
        reason: Concrete missing prerequisite or staging failure.

    Returns:
        ExecutionPlanBenchmarkResult: Unavailable benchmark tied to the requested plan.

    Raises:
        ValueError: If the plan identity or reason is empty.
    """
    if not plan_id.strip():
        raise ValueError("execution-plan benchmark plan ID must be non-empty")
    if not reason.strip():
        raise ValueError("execution-plan benchmark unavailable reason must be non-empty")
    return _unavailable_result(
        ExecutionPlanBenchmarkConfig(plan_id=plan_id, command=None),
        reason,
    )


def _decision_result(
    *,
    config: ExecutionPlanBenchmarkConfig,
    warmups: tuple[ExecutionPlanBenchmarkSample, ...],
    samples: tuple[ExecutionPlanBenchmarkSample, ...],
) -> ExecutionPlanBenchmarkResult:
    baseline_median = _arm_median(samples, "baseline")
    unplanned_median = _arm_median(samples, "unplanned")
    planned_median = _arm_median(samples, "planned")
    if min(baseline_median, unplanned_median, planned_median) < _MINIMUM_STABLE_MEDIAN_SECONDS:
        return ExecutionPlanBenchmarkResult(
            plan_id=config.plan_id,
            status="invalid",
            reason=(
                "execution-plan benchmark medians are too noisy: "
                f"baseline={baseline_median:.3f}s "
                f"unplanned={unplanned_median:.3f}s "
                f"planned={planned_median:.3f}s"
            ),
            minimum_marginal_speedup=config.minimum_marginal_speedup,
            minimum_overall_speedup=config.minimum_overall_speedup,
            baseline_median_seconds=baseline_median,
            unplanned_median_seconds=unplanned_median,
            planned_median_seconds=planned_median,
            marginal_speedup=None,
            overall_speedup=None,
            warmups=warmups,
            samples=samples,
        )

    marginal_speedup = unplanned_median / planned_median
    overall_speedup = baseline_median / planned_median
    if marginal_speedup >= config.minimum_marginal_speedup:
        overall_reason = (
            f"overall_speedup={overall_speedup:.3f}"
            if overall_speedup >= config.minimum_overall_speedup
            else (
                f"provisional overall_speedup={overall_speedup:.3f} is below final target "
                f"{config.minimum_overall_speedup:.3f}; final payload gate decides promotion"
            )
        )
        return ExecutionPlanBenchmarkResult(
            plan_id=config.plan_id,
            status="passed",
            reason=(
                "planned marginal ratio meets threshold: "
                f"marginal_speedup={marginal_speedup:.3f} "
                f"{overall_reason}"
            ),
            minimum_marginal_speedup=config.minimum_marginal_speedup,
            minimum_overall_speedup=config.minimum_overall_speedup,
            baseline_median_seconds=baseline_median,
            unplanned_median_seconds=unplanned_median,
            planned_median_seconds=planned_median,
            marginal_speedup=marginal_speedup,
            overall_speedup=overall_speedup,
            warmups=warmups,
            samples=samples,
        )
    return ExecutionPlanBenchmarkResult(
        plan_id=config.plan_id,
        status="not-profitable",
        reason=(
            "planned marginal ratio missed threshold: "
            f"marginal_speedup={marginal_speedup:.3f} "
            f"required={config.minimum_marginal_speedup:.3f}; "
            f"provisional overall_speedup={overall_speedup:.3f} "
            f"final_target={config.minimum_overall_speedup:.3f}"
        ),
        minimum_marginal_speedup=config.minimum_marginal_speedup,
        minimum_overall_speedup=config.minimum_overall_speedup,
        baseline_median_seconds=baseline_median,
        unplanned_median_seconds=unplanned_median,
        planned_median_seconds=planned_median,
        marginal_speedup=marginal_speedup,
        overall_speedup=overall_speedup,
        warmups=warmups,
        samples=samples,
    )


def _reject_unexpected_options(options: _ExecutionPlanBenchmarkOptions) -> None:
    allowed_options = {
        "progress",
        "baseline_region_allowlist",
        "unplanned_region_allowlist",
        "planned_region_allowlist",
        "baseline_variant_allowlist",
        "unplanned_variant_allowlist",
        "planned_variant_allowlist",
    }
    unexpected_options = set(options) - allowed_options
    if unexpected_options:
        unexpected_option = sorted(unexpected_options)[0]
        raise TypeError(
            "run_execution_plan_benchmark() got an unexpected keyword argument "
            f"{unexpected_option!r}"
        )


def _run_trios(
    context: _ExecutionPlanBenchmarkContext,
    *,
    phase: ExecutionPlanBenchmarkPhase,
    count: int,
) -> tuple[tuple[ExecutionPlanBenchmarkSample, ...], ExecutionPlanBenchmarkSample | None]:
    runs: list[ExecutionPlanBenchmarkSample] = []
    for trio_index in range(count):
        trio_runs, failure = _run_arm_sequence(
            context,
            _trio_order(trio_index),
            phase=phase,
            trio_index=trio_index,
        )
        runs.extend(trio_runs)
        if failure is not None:
            return tuple(runs), failure
    return tuple(runs), None


def _run_arm_sequence(
    context: _ExecutionPlanBenchmarkContext,
    arms: tuple[ExecutionPlanBenchmarkArm, ...],
    *,
    phase: ExecutionPlanBenchmarkPhase,
    trio_index: int,
) -> tuple[tuple[ExecutionPlanBenchmarkSample, ...], ExecutionPlanBenchmarkSample | None]:
    runs: list[ExecutionPlanBenchmarkSample] = []
    for arm in arms:
        wrapped = ExecutionPlanBenchmarkSample(
            arm=arm,
            run=run_performance_command(
                context.command,
                project_root=context.project_root,
                payload_root=_payload_root(context, arm),
                mode="baseline" if arm == "baseline" else "compiled",
                region_allowlist=_region_allowlist(context, arm),
                variant_allowlist=_variant_allowlist(context, arm),
            ),
        )
        runs.append(wrapped)
        if context.progress is not None:
            context.progress(
                ExecutionPlanBenchmarkProgress(
                    phase=phase,
                    trio_index=trio_index + 1,
                    sample_index=trio_index + 1 if phase == "sample" else None,
                    arm=arm,
                    duration_seconds=wrapped.run.duration_seconds,
                )
            )
        if not wrapped.succeeded:
            return tuple(runs), wrapped
    return tuple(runs), None


def _trio_order(
    trio_index: int,
) -> tuple[ExecutionPlanBenchmarkArm, ExecutionPlanBenchmarkArm, ExecutionPlanBenchmarkArm]:
    orders: tuple[
        tuple[ExecutionPlanBenchmarkArm, ExecutionPlanBenchmarkArm, ExecutionPlanBenchmarkArm],
        tuple[ExecutionPlanBenchmarkArm, ExecutionPlanBenchmarkArm, ExecutionPlanBenchmarkArm],
        tuple[ExecutionPlanBenchmarkArm, ExecutionPlanBenchmarkArm, ExecutionPlanBenchmarkArm],
        tuple[ExecutionPlanBenchmarkArm, ExecutionPlanBenchmarkArm, ExecutionPlanBenchmarkArm],
        tuple[ExecutionPlanBenchmarkArm, ExecutionPlanBenchmarkArm, ExecutionPlanBenchmarkArm],
        tuple[ExecutionPlanBenchmarkArm, ExecutionPlanBenchmarkArm, ExecutionPlanBenchmarkArm],
    ] = (
        ("baseline", "unplanned", "planned"),
        ("baseline", "planned", "unplanned"),
        ("unplanned", "baseline", "planned"),
        ("unplanned", "planned", "baseline"),
        ("planned", "baseline", "unplanned"),
        ("planned", "unplanned", "baseline"),
    )
    return orders[trio_index % len(orders)]


def _payload_root(context: _ExecutionPlanBenchmarkContext, arm: ExecutionPlanBenchmarkArm) -> Path:
    if arm == "baseline":
        return context.baseline_payload_root
    if arm == "unplanned":
        return context.unplanned_payload_root
    return context.planned_payload_root


def _region_allowlist(
    context: _ExecutionPlanBenchmarkContext,
    arm: ExecutionPlanBenchmarkArm,
) -> frozenset[str] | None:
    if arm == "baseline":
        return context.baseline_region_allowlist
    if arm == "unplanned":
        return context.unplanned_region_allowlist
    return context.planned_region_allowlist


def _variant_allowlist(
    context: _ExecutionPlanBenchmarkContext,
    arm: ExecutionPlanBenchmarkArm,
) -> frozenset[str] | None:
    if arm == "baseline":
        return context.baseline_variant_allowlist
    if arm == "unplanned":
        return context.unplanned_variant_allowlist
    return context.planned_variant_allowlist


def _arm_median(
    samples: tuple[ExecutionPlanBenchmarkSample, ...],
    arm: ExecutionPlanBenchmarkArm,
) -> float:
    return median(sample.run.duration_seconds for sample in samples if sample.arm == arm)


def _failure_reason(
    sample: ExecutionPlanBenchmarkSample,
    *,
    phase: ExecutionPlanBenchmarkPhase,
) -> str:
    return (
        f"{phase} execution-plan benchmark command exited with status "
        f"{sample.run.returncode} in {sample.arm} arm"
    )


def _unavailable_result(
    config: ExecutionPlanBenchmarkConfig,
    reason: str,
) -> ExecutionPlanBenchmarkResult:
    return ExecutionPlanBenchmarkResult(
        plan_id=config.plan_id,
        status="unavailable",
        reason=reason,
        minimum_marginal_speedup=config.minimum_marginal_speedup,
        minimum_overall_speedup=config.minimum_overall_speedup,
        baseline_median_seconds=None,
        unplanned_median_seconds=None,
        planned_median_seconds=None,
        marginal_speedup=None,
        overall_speedup=None,
        warmups=(),
        samples=(),
    )


def _invalid_result(
    *,
    config: ExecutionPlanBenchmarkConfig,
    reason: str,
    warmups: tuple[ExecutionPlanBenchmarkSample, ...],
    samples: tuple[ExecutionPlanBenchmarkSample, ...],
) -> ExecutionPlanBenchmarkResult:
    return ExecutionPlanBenchmarkResult(
        plan_id=config.plan_id,
        status="invalid",
        reason=reason,
        minimum_marginal_speedup=config.minimum_marginal_speedup,
        minimum_overall_speedup=config.minimum_overall_speedup,
        baseline_median_seconds=None,
        unplanned_median_seconds=None,
        planned_median_seconds=None,
        marginal_speedup=None,
        overall_speedup=None,
        warmups=warmups,
        samples=samples,
    )
