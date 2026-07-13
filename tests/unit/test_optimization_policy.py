"""Tests for shared acceleration thresholds and stability decisions."""

from __future__ import annotations

import pytest

from atoll.optimization_policy import (
    DEFAULT_MINIMUM_FINAL_SPEEDUP,
    DEFAULT_MINIMUM_MARGINAL_SPEEDUP,
    MINIMUM_STABLE_MEDIAN_SECONDS,
    assess_paired_speedup,
    assess_speedup,
    validate_acceleration_threshold,
)


@pytest.mark.parametrize("threshold", [0.0, 0.99, 1.0])
def test_acceleration_threshold_rejects_non_acceleration(threshold: float) -> None:
    """A gate can never be configured to accept unchanged or slower code."""
    with pytest.raises(ValueError, match=r"must be greater than 1\.0"):
        validate_acceleration_threshold(threshold, field="minimum speedup")


def test_speedup_assessment_requires_both_stable_medians() -> None:
    """A fast but sub-floor candidate remains too noisy for promotion."""
    assessment = assess_speedup(
        1.0,
        MINIMUM_STABLE_MEDIAN_SECONDS - 0.001,
        minimum_speedup=DEFAULT_MINIMUM_FINAL_SPEEDUP,
    )

    assert assessment.stable is False
    assert assessment.speedup is None
    assert assessment.passed is False


def test_speedup_assessment_accepts_stable_acceleration() -> None:
    """Stable medians meeting the threshold produce reusable decision evidence."""
    assessment = assess_speedup(1.1, 1.0, minimum_speedup=DEFAULT_MINIMUM_FINAL_SPEEDUP)

    assert assessment.stable is True
    assert assessment.speedup == pytest.approx(1.1)
    assert assessment.passed is True


def test_speedup_assessment_rejects_negative_duration() -> None:
    """Invalid timing evidence cannot reach a division or promotion decision."""
    with pytest.raises(ValueError, match="must be non-negative"):
        assess_speedup(-1.0, 1.0, minimum_speedup=DEFAULT_MINIMUM_FINAL_SPEEDUP)


def test_paired_speedup_rejects_order_biased_independent_medians() -> None:
    """One slow current sample cannot manufacture a marginal promotion."""
    assessment = assess_paired_speedup(
        (1.00, 1.07, 1.08),
        (0.96, 1.04, 0.98),
        minimum_speedup=DEFAULT_MINIMUM_MARGINAL_SPEEDUP,
    )

    assert (
        assessment.current_median_seconds / assessment.candidate_median_seconds
        > DEFAULT_MINIMUM_MARGINAL_SPEEDUP
    )
    assert assessment.median_pair_speedup == pytest.approx(1.04, rel=0.02)
    assert assessment.passed is False


def test_paired_speedup_accepts_consistent_marginal_improvement() -> None:
    """Corresponding stable samples must improve by the configured ratio."""
    assessment = assess_paired_speedup(
        (1.08, 1.10, 1.12),
        (1.00, 1.00, 1.00),
        minimum_speedup=DEFAULT_MINIMUM_MARGINAL_SPEEDUP,
    )

    assert assessment.stable is True
    assert assessment.median_pair_speedup == pytest.approx(1.10)
    assert assessment.passed is True


def test_paired_speedup_rejects_invalid_sample_sets() -> None:
    """Pairwise evidence requires corresponding non-empty sample groups."""
    with pytest.raises(ValueError, match="non-empty and equal"):
        assess_paired_speedup(
            (1.0,),
            (),
            minimum_speedup=DEFAULT_MINIMUM_MARGINAL_SPEEDUP,
        )
