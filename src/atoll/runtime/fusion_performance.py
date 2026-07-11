"""Run three-arm fusion research performance trials.

This module is a sibling to the two-arm benchmark gate and deliberately keeps
its result contracts separate. It compares a baseline payload, an unfused
compiled payload, and a fused compiled payload with deterministic trio rotation
so research callers can inspect command evidence without changing package gate
behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Literal, TypedDict, Unpack

from atoll.runtime.performance import CommandRunEvidence, run_performance_command

FusionArm = Literal["baseline", "unfused", "fused"]
FusionStatus = Literal["passed", "not-profitable", "invalid", "unavailable"]

_MINIMUM_STABLE_MEDIAN_SECONDS = 0.25


@dataclass(frozen=True, slots=True)
class FusionBenchmarkConfig:
    """Configuration for one fusion research benchmark trial.

    `command` is the benchmark argv tuple shared by all three arms, or `None`
    when the caller has no research benchmark available. `semantic_command`
    validates each arm before timing. Warmups are captured as evidence but
    excluded from medians. Sample medians must show that fused
    code is at least `minimum_over_unfused` faster than unfused compiled code
    and at least `minimum_overall` faster than baseline interpreted code.

    Attributes:
        plan_id: Stable task-fusion plan identity represented by the trial.
        command: Normalized benchmark command argument vector, or `None` for unavailable trials.
        semantic_command: Target-project command proving equivalent behavior in all three arms.
        warmups: Number of unmeasured three-arm warmup trios.
        samples: Number of measured three-arm sample trios.
        minimum_over_unfused: Required unfused-median / fused-median ratio.
        minimum_overall: Required baseline-median / fused-median ratio.
    """

    plan_id: str
    command: tuple[str, ...] | None
    semantic_command: tuple[str, ...] | None = None
    warmups: int = 1
    samples: int = 3
    minimum_over_unfused: float = 1.05
    minimum_overall: float = 1.10

    def __post_init__(self) -> None:
        """Reject invalid trial policy before launching child commands.

        Raises:
            ValueError: If the command is empty, counts are invalid, or thresholds are not
                positive.
        """
        if not self.plan_id.strip():
            raise ValueError("fusion benchmark plan ID must be non-empty")
        if self.command is not None and (
            not self.command or any(not part.strip() for part in self.command)
        ):
            raise ValueError("fusion benchmark command must be a non-empty argv tuple")
        if self.semantic_command is not None and (
            not self.semantic_command or any(not part.strip() for part in self.semantic_command)
        ):
            raise ValueError("fusion semantic command must be a non-empty argv tuple")
        if self.command is not None and self.semantic_command is None:
            raise ValueError("fusion benchmark command requires a semantic command")
        if self.warmups < 0:
            raise ValueError("fusion benchmark warmups must be at least 0")
        if self.samples < 1:
            raise ValueError("fusion benchmark samples must be at least 1")
        if self.minimum_over_unfused <= 0:
            raise ValueError("minimum over unfused must be greater than 0")
        if self.minimum_overall <= 0:
            raise ValueError("minimum overall must be greater than 0")


@dataclass(frozen=True, slots=True)
class FusionArmRunEvidence:
    """Observed command evidence tagged with its fusion benchmark arm.

    The underlying `run` keeps the Atoll runtime mode and subprocess details.
    `arm` retains the research role because both unfused and fused arms execute
    through compiled runtime mode and would otherwise be indistinguishable.

    Attributes:
        arm: Research arm represented by this command invocation.
        run: Captured subprocess evidence from the underlying performance runner.
    """

    arm: FusionArm
    run: CommandRunEvidence

    @property
    def succeeded(self) -> bool:
        """Return whether the wrapped command exited successfully.

        Returns:
            bool: Whether the underlying command returned status code zero.
        """
        return self.run.succeeded


@dataclass(frozen=True, slots=True)
class FusionTrial:
    """Three-arm fusion benchmark decision and command evidence.

    `status` is `passed` only when every arm succeeds, measured medians are
    stable, and the fused arm beats both configured profitability thresholds.
    `not-profitable` means timings were stable but at least one profitability
    threshold was missed. `invalid` means a command failed or timings were too
    short to support a useful decision. `unavailable` means no command was
    configured.

    Attributes:
        plan_id: Stable task-fusion plan identity evaluated by this trial.
        status: Trial decision.
        reason: Concrete execution or timing evidence supporting the decision.
        semantic_runs: Initial one-per-arm command evidence used for semantic validation.
        baseline_median_seconds: Median elapsed time for baseline measured samples.
        unfused_median_seconds: Median elapsed time for unfused measured samples.
        fused_median_seconds: Median elapsed time for fused measured samples.
        baseline_over_unfused: Baseline-median / unfused-median ratio.
        baseline_over_fused: Baseline-median / fused-median ratio.
        unfused_over_fused: Unfused-median / fused-median ratio.
        warmups: Unmeasured warmup command evidence.
        samples: Measured sample command evidence.
    """

    plan_id: str
    status: FusionStatus
    reason: str
    semantic_runs: tuple[FusionArmRunEvidence, ...]
    baseline_median_seconds: float | None
    unfused_median_seconds: float | None
    fused_median_seconds: float | None
    baseline_over_unfused: float | None
    baseline_over_fused: float | None
    unfused_over_fused: float | None
    warmups: tuple[FusionArmRunEvidence, ...]
    samples: tuple[FusionArmRunEvidence, ...]

    @property
    def succeeded(self) -> bool:
        """Return whether the fusion trial accepted the fused payload.

        Returns:
            bool: Whether the trial status is exactly `passed`.
        """
        return self.status == "passed"


class _FusionTrialOptions(TypedDict, total=False):
    baseline_region_allowlist: frozenset[str] | None
    unfused_region_allowlist: frozenset[str] | None
    fused_region_allowlist: frozenset[str] | None


@dataclass(frozen=True, slots=True)
class _FusionExecutionContext:
    command: tuple[str, ...]
    semantic_command: tuple[str, ...]
    project_root: Path
    baseline_payload_root: Path
    unfused_payload_root: Path
    fused_payload_root: Path
    baseline_region_allowlist: frozenset[str] | None
    unfused_region_allowlist: frozenset[str] | None
    fused_region_allowlist: frozenset[str] | None


def run_fusion_trial(
    config: FusionBenchmarkConfig,
    *,
    project_root: Path,
    baseline_payload_root: Path,
    unfused_payload_root: Path,
    fused_payload_root: Path,
    **options: Unpack[_FusionTrialOptions],
) -> FusionTrial:
    """Run semantic checks, warmups, and samples for a three-arm fusion trial.

    The semantic argv is first executed once for each arm. The benchmark argv
    then runs in deterministic rotated trios:
    baseline/unfused/fused, unfused/fused/baseline, then fused/baseline/unfused.
    Baseline runs use baseline runtime mode, while unfused and fused runs use
    compiled mode with their distinct payload roots and optional region
    allowlists.

    Args:
        config: Resolved trial configuration and profitability thresholds.
        project_root: Root directory of the target Python project.
        baseline_payload_root: Payload root for interpreted baseline measurements.
        unfused_payload_root: Payload root for compiled unfused measurements.
        fused_payload_root: Payload root for compiled fused measurements.
        **options: Optional per-arm region allowlists.

    Returns:
        FusionTrial: Three-arm semantic evidence, measured samples, ratios, and decision.

    Raises:
        TypeError: If an unsupported per-arm option is supplied.
        ValueError: If a configured benchmark lacks its required semantic command.
    """
    _reject_unexpected_trial_options(options)
    if config.command is None:
        return _unavailable_trial(config.plan_id, "no fusion benchmark command configured")
    if config.semantic_command is None:
        raise ValueError("fusion benchmark command requires a semantic command")

    context = _FusionExecutionContext(
        command=config.command,
        semantic_command=config.semantic_command,
        project_root=project_root,
        baseline_payload_root=baseline_payload_root,
        unfused_payload_root=unfused_payload_root,
        fused_payload_root=fused_payload_root,
        baseline_region_allowlist=options.get("baseline_region_allowlist"),
        unfused_region_allowlist=options.get("unfused_region_allowlist"),
        fused_region_allowlist=options.get("fused_region_allowlist"),
    )

    semantic_runs, failure = _run_semantic_checks(context)
    if failure is not None:
        return _invalid_trial(
            plan_id=config.plan_id,
            reason=_failure_reason(failure, phase="semantic"),
            semantic_runs=semantic_runs,
            warmups=(),
            samples=(),
        )

    warmups, failure = _run_trios(context, count=config.warmups)
    if failure is not None:
        return _invalid_trial(
            plan_id=config.plan_id,
            reason=_failure_reason(failure, phase="warmup"),
            semantic_runs=semantic_runs,
            warmups=warmups,
            samples=(),
        )

    samples, failure = _run_trios(context, count=config.samples)
    if failure is not None:
        return _invalid_trial(
            plan_id=config.plan_id,
            reason=_failure_reason(failure, phase="sample"),
            semantic_runs=semantic_runs,
            warmups=warmups,
            samples=samples,
        )

    return _decision_trial(
        config=config,
        semantic_runs=semantic_runs,
        warmups=warmups,
        samples=samples,
    )


def unavailable_fusion_trial(plan_id: str, reason: str) -> FusionTrial:
    """Return explicit non-executed evidence for one eligible plan.

    Args:
        plan_id: Stable task-fusion plan identity that could not be evaluated.
        reason: Concrete missing prerequisite or staging failure.

    Returns:
        FusionTrial: Unavailable trial tied to the requested plan.

    Raises:
        ValueError: If the plan identity or reason is empty.
    """
    if not plan_id.strip():
        raise ValueError("fusion trial plan ID must be non-empty")
    if not reason.strip():
        raise ValueError("fusion trial unavailable reason must be non-empty")
    return _unavailable_trial(plan_id, reason)


def _decision_trial(
    *,
    config: FusionBenchmarkConfig,
    semantic_runs: tuple[FusionArmRunEvidence, ...],
    warmups: tuple[FusionArmRunEvidence, ...],
    samples: tuple[FusionArmRunEvidence, ...],
) -> FusionTrial:
    baseline_median = _arm_median(samples, "baseline")
    unfused_median = _arm_median(samples, "unfused")
    fused_median = _arm_median(samples, "fused")
    if min(baseline_median, unfused_median, fused_median) < _MINIMUM_STABLE_MEDIAN_SECONDS:
        return FusionTrial(
            plan_id=config.plan_id,
            status="invalid",
            reason=(
                "fusion benchmark medians are too noisy: "
                f"baseline={baseline_median:.3f}s "
                f"unfused={unfused_median:.3f}s "
                f"fused={fused_median:.3f}s"
            ),
            semantic_runs=semantic_runs,
            baseline_median_seconds=baseline_median,
            unfused_median_seconds=unfused_median,
            fused_median_seconds=fused_median,
            baseline_over_unfused=None,
            baseline_over_fused=None,
            unfused_over_fused=None,
            warmups=warmups,
            samples=samples,
        )

    baseline_over_unfused = baseline_median / unfused_median
    baseline_over_fused = baseline_median / fused_median
    unfused_over_fused = unfused_median / fused_median
    if (
        unfused_over_fused >= config.minimum_over_unfused
        and baseline_over_fused >= config.minimum_overall
    ):
        return FusionTrial(
            plan_id=config.plan_id,
            status="passed",
            reason=(
                f"fused ratios meet thresholds: "
                f"unfused_over_fused={unfused_over_fused:.3f} "
                f"baseline_over_fused={baseline_over_fused:.3f}"
            ),
            semantic_runs=semantic_runs,
            baseline_median_seconds=baseline_median,
            unfused_median_seconds=unfused_median,
            fused_median_seconds=fused_median,
            baseline_over_unfused=baseline_over_unfused,
            baseline_over_fused=baseline_over_fused,
            unfused_over_fused=unfused_over_fused,
            warmups=warmups,
            samples=samples,
        )
    return FusionTrial(
        plan_id=config.plan_id,
        status="not-profitable",
        reason=(
            f"fused ratios missed thresholds: "
            f"unfused_over_fused={unfused_over_fused:.3f} "
            f"required={config.minimum_over_unfused:.3f}; "
            f"baseline_over_fused={baseline_over_fused:.3f} "
            f"required={config.minimum_overall:.3f}"
        ),
        semantic_runs=semantic_runs,
        baseline_median_seconds=baseline_median,
        unfused_median_seconds=unfused_median,
        fused_median_seconds=fused_median,
        baseline_over_unfused=baseline_over_unfused,
        baseline_over_fused=baseline_over_fused,
        unfused_over_fused=unfused_over_fused,
        warmups=warmups,
        samples=samples,
    )


def _reject_unexpected_trial_options(options: _FusionTrialOptions) -> None:
    allowed_options = {
        "baseline_region_allowlist",
        "unfused_region_allowlist",
        "fused_region_allowlist",
    }
    unexpected_options = set(options) - allowed_options
    if unexpected_options:
        unexpected_option = sorted(unexpected_options)[0]
        raise TypeError(
            f"run_fusion_trial() got an unexpected keyword argument {unexpected_option!r}"
        )


def _run_semantic_checks(
    context: _FusionExecutionContext,
) -> tuple[tuple[FusionArmRunEvidence, ...], FusionArmRunEvidence | None]:
    return _run_arm_sequence(
        context,
        ("baseline", "unfused", "fused"),
        command=context.semantic_command,
    )


def _run_trios(
    context: _FusionExecutionContext,
    *,
    count: int,
) -> tuple[tuple[FusionArmRunEvidence, ...], FusionArmRunEvidence | None]:
    runs: list[FusionArmRunEvidence] = []
    for trio_index in range(count):
        trio_runs, failure = _run_arm_sequence(
            context,
            _trio_order(trio_index),
            command=context.command,
        )
        runs.extend(trio_runs)
        if failure is not None:
            return tuple(runs), failure
    return tuple(runs), None


def _run_arm_sequence(
    context: _FusionExecutionContext,
    arms: tuple[FusionArm, ...],
    *,
    command: tuple[str, ...],
) -> tuple[tuple[FusionArmRunEvidence, ...], FusionArmRunEvidence | None]:
    runs: list[FusionArmRunEvidence] = []
    for arm in arms:
        wrapped = FusionArmRunEvidence(
            arm=arm,
            run=run_performance_command(
                command,
                project_root=context.project_root,
                payload_root=_payload_root(context, arm),
                mode="baseline" if arm == "baseline" else "compiled",
                region_allowlist=_region_allowlist(context, arm),
            ),
        )
        runs.append(wrapped)
        if not wrapped.succeeded:
            return tuple(runs), wrapped
    return tuple(runs), None


def _trio_order(trio_index: int) -> tuple[FusionArm, FusionArm, FusionArm]:
    orders: tuple[
        tuple[FusionArm, FusionArm, FusionArm],
        tuple[FusionArm, FusionArm, FusionArm],
        tuple[FusionArm, FusionArm, FusionArm],
    ] = (
        ("baseline", "unfused", "fused"),
        ("unfused", "fused", "baseline"),
        ("fused", "baseline", "unfused"),
    )
    return orders[trio_index % len(orders)]


def _payload_root(context: _FusionExecutionContext, arm: FusionArm) -> Path:
    if arm == "baseline":
        return context.baseline_payload_root
    if arm == "unfused":
        return context.unfused_payload_root
    return context.fused_payload_root


def _region_allowlist(
    context: _FusionExecutionContext,
    arm: FusionArm,
) -> frozenset[str] | None:
    if arm == "baseline":
        return context.baseline_region_allowlist
    if arm == "unfused":
        return context.unfused_region_allowlist
    return context.fused_region_allowlist


def _arm_median(samples: tuple[FusionArmRunEvidence, ...], arm: FusionArm) -> float:
    return median(run.run.duration_seconds for run in samples if run.arm == arm)


def _failure_reason(run: FusionArmRunEvidence, *, phase: str) -> str:
    return (
        f"{phase} fusion benchmark command exited with status {run.run.returncode} in {run.arm} arm"
    )


def _unavailable_trial(plan_id: str, reason: str) -> FusionTrial:
    return FusionTrial(
        plan_id=plan_id,
        status="unavailable",
        reason=reason,
        semantic_runs=(),
        baseline_median_seconds=None,
        unfused_median_seconds=None,
        fused_median_seconds=None,
        baseline_over_unfused=None,
        baseline_over_fused=None,
        unfused_over_fused=None,
        warmups=(),
        samples=(),
    )


def _invalid_trial(
    *,
    plan_id: str,
    reason: str,
    semantic_runs: tuple[FusionArmRunEvidence, ...],
    warmups: tuple[FusionArmRunEvidence, ...],
    samples: tuple[FusionArmRunEvidence, ...],
) -> FusionTrial:
    return FusionTrial(
        plan_id=plan_id,
        status="invalid",
        reason=reason,
        semantic_runs=semantic_runs,
        baseline_median_seconds=None,
        unfused_median_seconds=None,
        fused_median_seconds=None,
        baseline_over_unfused=None,
        baseline_over_fused=None,
        unfused_over_fused=None,
        warmups=warmups,
        samples=samples,
    )
