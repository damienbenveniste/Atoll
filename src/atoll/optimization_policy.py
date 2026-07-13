"""Central acceleration policy shared by every optimization family.

This module owns numerical promotion thresholds and the conservative comparison
used by benchmark runners. It does not execute benchmarks, choose candidates,
or format family-specific diagnostics. Keeping those responsibilities separate
lets native, source, execution-plan, and research gates apply identical stability
and acceleration rules without coupling their evidence models.
"""

from __future__ import annotations

from dataclasses import dataclass

MINIMUM_STABLE_MEDIAN_SECONDS = 0.25
PROFILE_GUIDED_MINIMUM_MARGINAL_SPEEDUP = 1.01
DEFAULT_MINIMUM_MARGINAL_SPEEDUP = 1.05
DEFAULT_MINIMUM_FINAL_SPEEDUP = 1.10
HARD_BENCHMARK_MINIMUM_SPEEDUP = 3.0


@dataclass(frozen=True, slots=True)
class SpeedupAssessment:
    """One stable baseline-to-candidate acceleration decision.

    ``speedup`` is omitted when either median is below the stability floor.
    This prevents callers from accidentally promoting a ratio derived from a
    sub-quarter-second arm. Instances are immutable and safe to include in
    report-facing structured evidence.

    Attributes:
        baseline_median_seconds: Median duration for the arm being improved.
        candidate_median_seconds: Median duration for the proposed faster arm.
        minimum_speedup: Required baseline-to-candidate ratio.
        stable: Whether both medians meet the shared stability floor.
        speedup: Stable baseline/candidate ratio, or ``None`` when too noisy.
        passed: Whether stable evidence meets the required ratio.
    """

    baseline_median_seconds: float
    candidate_median_seconds: float
    minimum_speedup: float
    stable: bool
    speedup: float | None
    passed: bool


def validate_acceleration_threshold(value: float, *, field: str) -> None:
    """Reject thresholds that can promote unchanged or slower code.

    Args:
        value: Configured baseline-to-candidate speedup ratio.
        field: User-facing field name included in validation errors.

    Raises:
        ValueError: If ``value`` is not strictly greater than one.
    """
    if value <= 1.0:
        raise ValueError(f"{field} must be greater than 1.0")


def assess_speedup(
    baseline_median_seconds: float,
    candidate_median_seconds: float,
    *,
    minimum_speedup: float,
) -> SpeedupAssessment:
    """Apply the shared stability floor and acceleration threshold.

    Args:
        baseline_median_seconds: Median duration of the current accepted arm.
        candidate_median_seconds: Median duration of the candidate arm.
        minimum_speedup: Required baseline-to-candidate ratio.

    Returns:
        SpeedupAssessment: Stable ratio and promotion decision.

    Raises:
        ValueError: If a duration is negative or the threshold is not acceleration.
    """
    validate_acceleration_threshold(minimum_speedup, field="minimum speedup")
    if baseline_median_seconds < 0.0 or candidate_median_seconds < 0.0:
        raise ValueError("benchmark medians must be non-negative")
    stable = (
        baseline_median_seconds >= MINIMUM_STABLE_MEDIAN_SECONDS
        and candidate_median_seconds >= MINIMUM_STABLE_MEDIAN_SECONDS
    )
    speedup = baseline_median_seconds / candidate_median_seconds if stable else None
    return SpeedupAssessment(
        baseline_median_seconds=baseline_median_seconds,
        candidate_median_seconds=candidate_median_seconds,
        minimum_speedup=minimum_speedup,
        stable=stable,
        speedup=speedup,
        passed=stable and speedup is not None and speedup >= minimum_speedup,
    )
