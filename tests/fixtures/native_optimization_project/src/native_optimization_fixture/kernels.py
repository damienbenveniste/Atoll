"""Typed CPU kernels used by the native-optimization fixture.

The fixture intentionally stays independent of Atoll compilation. It exposes
small deterministic loops and fallback probes that future native backends can
exercise while preserving exact Python semantics at unsafe boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

FALLBACK_LIMIT = (1 << 53) - 1
_MASK = (1 << 61) - 1
_DEFAULT_WIDTH = 96
_DEFAULT_REPETITIONS = 24


def scalar_polynomial(limit: int, rounds: int = 1, *, bias: int = 3) -> int:
    """Run one fixed-width-friendly polynomial reduction.

    Args:
        limit: Exclusive upper bound for the reduction.
        rounds: Scalar multiplier applied after the loop.
        bias: Keyword-only additive result bias.

    Returns:
        Exact polynomial reduction for nonnegative guarded inputs.
    """
    total = 0
    for value in range(limit):
        total += value * value + 3 * value + 17
    return total * rounds + bias


class ScalarArithmetic:
    """Stateless scalar methods whose source class must remain unchanged."""

    @staticmethod
    def weighted_sum(limit: int, factor: int = 2) -> int:
        """Return a fixed-width-friendly weighted reduction.

        Args:
            limit: Exclusive upper bound for the reduction.
            factor: Multiplier applied to each loop value.

        Returns:
            Exact weighted reduction for nonnegative guarded inputs.
        """
        total = 0
        for value in range(limit):
            total += value * factor + 1
        return total


@dataclass(frozen=True, slots=True)
class WorkloadSnapshot:
    """Stable summary returned by the CPU benchmark workload.

    Attributes:
        checksum: Accumulated deterministic integer checksum.
        iterations: Number of workload iterations executed.
        logical_items: Number of logical loop items processed.
    """

    checksum: int
    iterations: int
    logical_items: int


def polynomial_checksum(
    width: int = _DEFAULT_WIDTH,
    repetitions: int = _DEFAULT_REPETITIONS,
) -> int:
    """Evaluate a deterministic integer polynomial loop.

    Args:
        width: Number of logical input points per repetition.
        repetitions: Number of times the loop should traverse the point range.

    Returns:
        Deterministic checksum with arithmetic constrained to a fixed mask.

    Raises:
        ValueError: If either dimension is less than one.
    """

    if width < 1 or repetitions < 1:
        raise ValueError("width and repetitions must be positive")

    total = 0
    for repetition in range(repetitions):
        seed = repetition + 3
        for value in range(width):
            x = value + seed
            polynomial = (((x * x + 3 * x + 17) * x) - (7 * x)) + 11
            total = (total + polynomial + (value ^ seed)) & _MASK
    return total


def branch_checksum(values: tuple[int, ...], pivot: int = 7) -> int:
    """Accumulate branch-heavy arithmetic over integer values.

    Args:
        values: Input values to classify.
        pivot: Divisibility pivot used by one branch.

    Returns:
        Deterministic checksum that changes branch routes for negative, even,
        divisible, and fallback values.

    Raises:
        ValueError: If `pivot` is zero.
    """

    return BranchArithmetic.accumulate(values, pivot=pivot)


def keyword_polynomial_window(
    start: int,
    stop: int = 9,
    *,
    scale: int = 3,
    bias: int = 5,
) -> tuple[int, ...]:
    """Return polynomial values while exercising defaults and keyword-only args.

    Args:
        start: Inclusive first value.
        stop: Exclusive final value.
        scale: Keyword-only multiplier.
        bias: Keyword-only additive term.

    Returns:
        Tuple of deterministic polynomial values for the requested window.
    """

    return tuple((index * index + scale * index + bias) for index in range(start, stop))


def run_baseline_workload(iterations: int) -> WorkloadSnapshot:
    """Run enough typed CPU work for a baseline benchmark command.

    Args:
        iterations: Number of workload rounds to execute.

    Returns:
        WorkloadSnapshot: Stable checksum and logical item counts.

    Raises:
        ValueError: If `iterations` is less than one.
    """

    if iterations < 1:
        raise ValueError("iterations must be positive")

    branch_values = (-9, -2, 0, 1, 4, 7, 11, 14, 23, 28)
    checksum = 0
    for index in range(iterations):
        width = 92 + (index & 7)
        repetitions = 22 + (index & 3)
        checksum = (checksum + polynomial_checksum(width, repetitions)) & _MASK
        checksum = (checksum + BranchArithmetic.accumulate(branch_values, pivot=7)) & _MASK
        checksum = (checksum + BranchArithmetic.mixed(index, scale=3, bias=11)) & _MASK
        window_total = sum(keyword_polynomial_window(1, 8, scale=2, bias=index & 5))
        checksum = (checksum + window_total) & _MASK
        checksum = (checksum + scalar_polynomial(96 + (index & 7), bias=index & 3)) & _MASK

    return WorkloadSnapshot(
        checksum=checksum,
        iterations=iterations,
        logical_items=iterations * len(branch_values),
    )


class BranchArithmetic:
    """Static branch-arithmetic methods used as native lowering candidates."""

    @staticmethod
    def accumulate(values: tuple[int, ...], *, pivot: int = 7) -> int:
        """Classify values and accumulate a deterministic integer checksum.

        Args:
            values: Values to classify.
            pivot: Divisibility pivot for the high-magnitude positive branch.

        Returns:
            Integer checksum for the branch decisions.

        Raises:
            ValueError: If `pivot` is zero.
        """

        if pivot == 0:
            raise ValueError("pivot must be nonzero")

        total = 0
        for value in values:
            if value < 0:
                total -= value * value - 3 * value
            elif value % 2 == 0:
                total += value // 2 + value * 5
            elif value % pivot == 0:
                total += value * value * 2
            else:
                total += value + 11
        return total

    @staticmethod
    def mixed(value: int, *, scale: int = 2, bias: int = 1) -> int:
        """Combine arithmetic branches with keyword-only configuration.

        Args:
            value: Input value to classify.
            scale: Keyword-only multiplier for the even branch.
            bias: Keyword-only additive term used by all branches.

        Returns:
            Branch-specific arithmetic result.
        """

        if value < 0:
            return bias - value * value
        if value & 1:
            return value * value * value + bias
        return value * scale + bias


class FallbackProbe:
    """Exact Python fallback boundary for values unsafe for native integers."""

    @staticmethod
    def square_route(value: int) -> tuple[str, int]:
        """Return the execution route and exact square for an integer-like value.

        Args:
            value: Candidate value for exact-int specialization.

        Returns:
            Pair whose first item is `native` for exact safe ints and `python`
            for bools, int subclasses, or huge integers. The second item always
            preserves Python's exact integer square.
        """

        exact_value = int(value)
        if type(value) is not int or abs(exact_value) > FALLBACK_LIMIT:
            return ("python", exact_value * exact_value)
        return ("native", exact_value * exact_value)
