"""Native-optimization fixture API for semantic and benchmark tests."""

from __future__ import annotations

from native_optimization_fixture.kernels import (
    FALLBACK_LIMIT,
    BranchArithmetic,
    FallbackProbe,
    ScalarArithmetic,
    WorkloadSnapshot,
    branch_checksum,
    keyword_polynomial_window,
    polynomial_checksum,
    run_baseline_workload,
    scalar_polynomial,
)

__all__ = [
    "FALLBACK_LIMIT",
    "BranchArithmetic",
    "FallbackProbe",
    "ScalarArithmetic",
    "WorkloadSnapshot",
    "branch_checksum",
    "keyword_polynomial_window",
    "polynomial_checksum",
    "run_baseline_workload",
    "scalar_polynomial",
]
