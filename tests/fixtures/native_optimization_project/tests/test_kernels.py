"""Semantic contract for the native-optimization fixture package."""

from __future__ import annotations

from native_optimization_fixture import (
    FALLBACK_LIMIT,
    BranchArithmetic,
    FallbackProbe,
    ScalarArithmetic,
    branch_checksum,
    keyword_polynomial_window,
    polynomial_checksum,
    scalar_polynomial,
)

EXPECTED_BRANCH_CHECKSUM = 173
EXPECTED_EVEN_MIXED = 35
EXPECTED_ODD_MIXED = 128
EXPECTED_POLYNOMIAL_CHECKSUM = 19944


class StrictInt(int):
    """Integer subclass used to assert exact-type fallback routing."""


def test_typed_hot_kernels_are_deterministic() -> None:
    assert polynomial_checksum(width=8, repetitions=3) == EXPECTED_POLYNOMIAL_CHECKSUM
    assert branch_checksum((-3, -1, 0, 2, 7, 9, 12)) == EXPECTED_BRANCH_CHECKSUM
    assert BranchArithmetic.mixed(8, scale=4, bias=3) == EXPECTED_EVEN_MIXED
    assert BranchArithmetic.mixed(5, scale=4, bias=3) == EXPECTED_ODD_MIXED
    assert keyword_polynomial_window(2, scale=4, bias=1) == (
        13,
        22,
        33,
        46,
        61,
        78,
        97,
    )
    expected_polynomial = 725
    assert scalar_polynomial(8, rounds=2, bias=5) == expected_polynomial
    expected_weighted_sum = 145
    assert ScalarArithmetic.weighted_sum(10, factor=3) == expected_weighted_sum


def test_unsafe_values_use_exact_python_fallback_route() -> None:
    huge = FALLBACK_LIMIT + 10

    assert FallbackProbe.square_route(17) == ("native", 289)
    assert FallbackProbe.square_route(True) == ("python", 1)
    assert FallbackProbe.square_route(StrictInt(19)) == ("python", 361)
    assert FallbackProbe.square_route(huge) == ("python", huge * huge)
