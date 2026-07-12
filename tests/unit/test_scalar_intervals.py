"""Tests for scalar interval proof arithmetic."""

from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from atoll.native_optimization.intervals import (
    ClosedIntInterval,
    NativeBitWidth,
    NativeInteger,
    OperationProof,
    OperationReason,
    RangeLoopInduction,
    SymbolicBound,
    accumulate_additive,
    add,
    bitwise,
    build_range_induction,
    compare,
    every_intermediate_fits,
    floor_divide,
    join,
    modulo,
    multiply,
    power,
    shift,
    subtract,
)

_THREE_ITERATIONS = 3


def _interval_tuple(interval: ClosedIntInterval | None) -> tuple[int, int]:
    assert interval is not None
    return (interval.minimum, interval.maximum)


def test_interval_models_are_immutable_slots_with_decimal_bounds() -> None:
    interval = ClosedIntInterval.closed(-3, 7)

    assert hasattr(interval, "__slots__")
    assert interval.lower == SymbolicBound(value=-3, decimal="-3")
    assert interval.upper.decimal == "7"
    with pytest.raises(FrozenInstanceError):
        interval.__setattr__("minimum", 0)
    with pytest.raises(ValueError, match="decimal text"):
        SymbolicBound(value=10, decimal="+10")
    with pytest.raises(ValueError, match="minimum"):
        ClosedIntInterval.closed(5, 4)
    with pytest.raises(ValueError, match="symbolic bounds"):
        ClosedIntInterval(
            minimum=0,
            maximum=1,
            lower=SymbolicBound.exact(1),
            upper=SymbolicBound.exact(1),
        )


def test_proof_and_induction_models_reject_inconsistent_state() -> None:
    """Persistable proof records enforce their own construction invariants."""
    point = ClosedIntInterval.point(1)
    reason = OperationReason(code="unsupported-operation", message="unsupported")

    with pytest.raises(ValueError, match="32 or 64"):
        NativeInteger(width=cast(NativeBitWidth, 16))
    with pytest.raises(ValueError, match="non-empty"):
        OperationReason(code="unsupported-operation", message=" ")
    with pytest.raises(ValueError, match="operation name"):
        OperationProof("", (point,), "proved", point, (reason,))
    with pytest.raises(ValueError, match="at least one reason"):
        OperationProof("add", (point,), "proved", point, ())
    with pytest.raises(ValueError, match="require a result"):
        OperationProof("add", (point,), "proved", None, (reason,))
    with pytest.raises(ValueError, match="cannot carry"):
        OperationProof("add", (point,), "rejected", point, (reason,))
    with pytest.raises(ValueError, match="non-zero"):
        RangeLoopInduction(0, 1, 0, 1, point)
    with pytest.raises(ValueError, match="non-negative"):
        RangeLoopInduction(0, 1, 1, -1, point)
    with pytest.raises(ValueError, match="first and last"):
        RangeLoopInduction(0, 3, 1, 3, point)

    valid = RangeLoopInduction(
        0,
        _THREE_ITERATIONS,
        1,
        _THREE_ITERATIONS,
        ClosedIntInterval.closed(0, 2),
    )
    assert valid.iterations == _THREE_ITERATIONS
    assert NativeInteger(width=64).interval == ClosedIntInterval.closed(-(2**63), 2**63 - 1)


def test_native_widths_signedness_and_intermediate_fit_checks() -> None:
    signed32 = NativeInteger(width=32)
    signed64 = NativeInteger(width=64)
    unsigned32 = NativeInteger(width=32, signed=False)
    wide = add(ClosedIntInterval.point(2**31 - 1), ClosedIntInterval.point(1))
    small = multiply(ClosedIntInterval.closed(-2, 3), ClosedIntInterval.closed(4, 5))

    assert _interval_tuple(wide.result) == (2**31, 2**31)
    assert _interval_tuple(small.result) == (-10, 15)
    assert not every_intermediate_fits(signed32, (small, wide))
    assert every_intermediate_fits(signed64, (small, wide))
    assert ClosedIntInterval.closed(0, 2**32 - 1).fits_native(unsigned32)
    assert not ClosedIntInterval.closed(-1, 1).fits_native(unsigned32)


def test_native_fit_rejects_unrepresentable_operands_with_small_results() -> None:
    """A small remainder or boolean cannot hide a huge Python operand."""
    native = NativeInteger(width=64)
    huge = ClosedIntInterval.point(2**100)
    remainder = modulo(huge, ClosedIntInterval.point(3))
    comparison = compare(huge, ">", ClosedIntInterval.point(0))

    assert remainder.result == ClosedIntInterval.closed(0, 2)
    assert comparison.result == ClosedIntInterval.point(1)
    assert remainder.fits_native(native) is False
    assert comparison.fits_native(native) is False


def test_add_subtract_multiply_and_join_compute_conservative_exact_bounds() -> None:
    left = ClosedIntInterval.closed(-4, 6)
    right = ClosedIntInterval.closed(2, 5)

    assert _interval_tuple(add(left, right).result) == (-2, 11)
    assert _interval_tuple(subtract(left, right).result) == (-9, 4)
    assert _interval_tuple(multiply(left, right).result) == (-20, 30)
    assert _interval_tuple(join(ClosedIntInterval.closed(7, 9), left, right).result) == (-4, 9)
    rejected = join()
    assert not rejected.proved
    assert rejected.reasons[0].code == "unsupported-operation"


def test_comparisons_return_boolean_intervals_for_definite_and_maybe_relations() -> None:
    low = ClosedIntInterval.closed(0, 3)
    high = ClosedIntInterval.closed(5, 7)
    overlap = ClosedIntInterval.closed(3, 6)

    assert _interval_tuple(compare(low, "<", high).result) == (1, 1)
    assert _interval_tuple(compare(high, "<=", low).result) == (0, 0)
    assert _interval_tuple(compare(low, "==", overlap).result) == (0, 1)
    assert _interval_tuple(compare(low, "!=", high).result) == (1, 1)


def test_nonnegative_division_and_modulo_reject_zero_and_signed_domains() -> None:
    dividend = ClosedIntInterval.closed(10, 25)
    divisor = ClosedIntInterval.closed(3, 5)

    assert _interval_tuple(floor_divide(dividend, divisor).result) == (2, 8)
    assert _interval_tuple(modulo(dividend, divisor).result) == (0, 4)

    zero = floor_divide(dividend, ClosedIntInterval.closed(0, 4))
    signed_division = floor_divide(ClosedIntInterval.closed(-2, 4), divisor)
    zero_modulo = modulo(dividend, ClosedIntInterval.closed(0, 4))
    negative = modulo(ClosedIntInterval.closed(-1, 4), divisor)
    assert zero.reasons[0].code == "division-by-possible-zero"
    assert signed_division.reasons[0].code == "negative-divmod-domain"
    assert zero_modulo.reasons[0].code == "division-by-possible-zero"
    assert negative.reasons[0].code == "negative-divmod-domain"


def test_bitwise_operations_are_exact_for_small_nonnegative_domains() -> None:
    left = ClosedIntInterval.closed(2, 3)
    right = ClosedIntInterval.closed(4, 5)

    assert _interval_tuple(bitwise(left, "&", right).result) == (0, 1)
    assert _interval_tuple(bitwise(left, "|", right).result) == (6, 7)
    assert _interval_tuple(bitwise(left, "^", right).result) == (6, 7)

    negative = bitwise(ClosedIntInterval.closed(-1, 1), "&", right)
    too_large = bitwise(ClosedIntInterval.closed(0, 4096), "|", ClosedIntInterval.closed(0, 1))
    assert negative.reasons[0].code == "negative-bitwise-domain"
    assert too_large.reasons[0].code == "unproven-operation"


def test_guarded_shifts_reject_negative_and_unbounded_shift_counts() -> None:
    value = ClosedIntInterval.closed(4, 8)

    assert _interval_tuple(shift(value, "<<", ClosedIntInterval.closed(1, 3)).result) == (8, 64)
    assert _interval_tuple(shift(value, ">>", ClosedIntInterval.closed(1, 2)).result) == (1, 4)
    assert shift(value, "<<", ClosedIntInterval.closed(-1, 1)).reasons[0].code == (
        "negative-shift-count"
    )
    assert shift(value, "<<", ClosedIntInterval.closed(0, 64)).reasons[0].code == (
        "unbounded-shift-count"
    )
    assert (
        shift(ClosedIntInterval.closed(-1, 1), ">>", ClosedIntInterval.point(1)).reasons[0].code
        == "negative-bitwise-domain"
    )


def test_power_supports_small_constant_exponents_and_rejects_unsafe_domains() -> None:
    assert _interval_tuple(
        power(ClosedIntInterval.closed(2, 4), ClosedIntInterval.point(3)).result
    ) == (
        8,
        64,
    )
    assert _interval_tuple(
        power(ClosedIntInterval.closed(-2, 3), ClosedIntInterval.point(2)).result
    ) == (
        0,
        9,
    )
    assert power(ClosedIntInterval.closed(2, 4), ClosedIntInterval.closed(2, 3)).reasons[
        0
    ].code == ("unsafe-power")
    assert power(ClosedIntInterval.closed(2, 4), ClosedIntInterval.point(9)).reasons[0].code == (
        "unsafe-power"
    )
    assert power(ClosedIntInterval.closed(-5000, 1), ClosedIntInterval.point(2)).reasons[
        0
    ].code == ("unsafe-power")


def test_range_loop_induction_and_additive_accumulation_helpers_are_bounded() -> None:
    induction = build_range_induction(2, 11, 3)
    reverse = build_range_induction(10, 1, -4)
    empty = build_range_induction(5, 5)
    total = accumulate_additive(
        ClosedIntInterval.closed(1, 2),
        ClosedIntInterval.closed(3, 5),
        iterations=4,
    )
    rejected = accumulate_additive(ClosedIntInterval.point(0), ClosedIntInterval.point(1), -1)

    assert _interval_tuple(induction.result) == (2, 8)
    assert _interval_tuple(reverse.result) == (2, 10)
    assert empty.reasons[0].code == "empty-range"
    assert _interval_tuple(total.result) == (13, 22)
    assert rejected.reasons[0].code == "unsupported-operation"
    with pytest.raises(ValueError, match="step"):
        build_range_induction(0, 10, 0)
