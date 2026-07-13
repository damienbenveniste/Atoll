"""Tests for representative Cython cold-batch benchmark policy."""

import pytest
from scripts.run_cython_batch_benchmark import BatchEvidenceInputs, evaluate_evidence


def test_batch_evidence_requires_twenty_percent_reduction_and_warm_hits() -> None:
    evidence = evaluate_evidence(
        BatchEvidenceInputs(
            sequential_samples=(10.0, 11.0, 9.0),
            batch_samples=(7.0, 8.0, 7.5),
            artifact_parity=True,
            warm_cache_hits=8,
            warm_native_phase_count=0,
            unit_count=8,
            minimum_reduction=0.20,
        )
    )

    assert evidence.passed is True
    assert evidence.sequential_median_seconds == pytest.approx(10.0)
    assert evidence.batch_median_seconds == pytest.approx(7.5)
    assert evidence.cold_reduction == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("artifact_parity", "warm_cache_hits", "warm_native_phase_count"),
    [(False, 8, 0), (True, 7, 0), (True, 8, 1)],
)
def test_batch_evidence_rejects_parity_or_warm_cache_failures(
    artifact_parity: bool,
    warm_cache_hits: int,
    warm_native_phase_count: int,
) -> None:
    evidence = evaluate_evidence(
        BatchEvidenceInputs(
            sequential_samples=(10.0,),
            batch_samples=(7.0,),
            artifact_parity=artifact_parity,
            warm_cache_hits=warm_cache_hits,
            warm_native_phase_count=warm_native_phase_count,
            unit_count=8,
            minimum_reduction=0.20,
        )
    )

    assert evidence.passed is False


def test_batch_evidence_rejects_insufficient_cold_reduction() -> None:
    evidence = evaluate_evidence(
        BatchEvidenceInputs(
            sequential_samples=(10.0,),
            batch_samples=(8.5,),
            artifact_parity=True,
            warm_cache_hits=8,
            warm_native_phase_count=0,
            unit_count=8,
            minimum_reduction=0.20,
        )
    )

    assert evidence.passed is False
