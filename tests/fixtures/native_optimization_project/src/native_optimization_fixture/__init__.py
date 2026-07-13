"""Native-optimization fixture API for semantic and benchmark tests."""

from __future__ import annotations

from native_optimization_fixture.kernels import (
    FALLBACK_LIMIT,
    BranchArithmetic,
    ChainAccumulator,
    FallbackProbe,
    ScalarArithmetic,
    WorkloadSnapshot,
    branch_checksum,
    call_chain_hard_checksum,
    direct_chain_leaf,
    direct_chain_middle,
    direct_chain_root,
    direct_chain_route,
    keyword_polynomial_window,
    polynomial_checksum,
    run_baseline_workload,
    scalar_polynomial,
)

__all__ = [
    "FALLBACK_LIMIT",
    "BranchArithmetic",
    "ChainAccumulator",
    "FallbackProbe",
    "ScalarArithmetic",
    "WorkloadSnapshot",
    "branch_checksum",
    "call_chain_hard_checksum",
    "direct_chain_leaf",
    "direct_chain_middle",
    "direct_chain_root",
    "direct_chain_route",
    "keyword_polynomial_window",
    "polynomial_checksum",
    "run_baseline_workload",
    "scalar_polynomial",
]
