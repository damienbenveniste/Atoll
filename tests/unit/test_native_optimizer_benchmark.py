"""Tests for generic native optimizer hard-gate evaluation."""

from scripts.run_native_optimizer_benchmark import FamilyEvaluationInputs, evaluate_family

from atoll.optimization_policy import HARD_BENCHMARK_MINIMUM_SPEEDUP

EXPECTED_DIAGNOSTIC_COUNT = 3


def test_native_family_accepts_stable_semantic_three_x_speedup() -> None:
    evidence = evaluate_family(
        FamilyEvaluationInputs(
            name="fixture",
            calls=100,
            baseline_samples=(1.5, 1.6, 1.4),
            compiled_samples=(0.4, 0.5, 0.45),
            semantic_match=True,
            active_bindings=("kernel",),
            expected_bindings=("kernel",),
        )
    )

    assert evidence.passed is True
    assert evidence.speedup is not None
    assert evidence.speedup > HARD_BENCHMARK_MINIMUM_SPEEDUP
    assert evidence.diagnostics == ()


def test_native_family_rejects_unstable_missing_or_unprofitable_routing() -> None:
    evidence = evaluate_family(
        FamilyEvaluationInputs(
            name="fixture",
            calls=100,
            baseline_samples=(0.2,),
            compiled_samples=(0.1,),
            semantic_match=False,
            active_bindings=(),
            expected_bindings=("kernel",),
        )
    )

    assert evidence.passed is False
    assert len(evidence.diagnostics) == EXPECTED_DIAGNOSTIC_COUNT
    assert "stability" in evidence.diagnostics[0]
    assert "differ" in evidence.diagnostics[1]
    assert "kernel" in evidence.diagnostics[2]
