"""Closed integer interval arithmetic contracts for native scalar analysis.

This module gives AST analyzers a conservative arithmetic substrate for proving
that scalar expressions stay inside native integer domains. It deliberately
returns structured proof records for operation failures so callers can report
why an expression fell back to Python instead of decoding exception text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

NativeBitWidth = Literal[32, 64]
ComparisonOperator = Literal["<", "<=", ">", ">=", "==", "!="]
BitwiseOperator = Literal["&", "|", "^"]
ShiftOperator = Literal["<<", ">>"]
ProofStatus = Literal["proved", "rejected"]
ReasonCode = Literal[
    "operation-proved",
    "division-by-possible-zero",
    "negative-divmod-domain",
    "negative-bitwise-domain",
    "negative-shift-count",
    "unbounded-shift-count",
    "unsafe-power",
    "unsupported-operation",
    "unproven-operation",
    "empty-range",
    "invalid-native-width",
]

_INT32_MIN = -(2**31)
_INT32_MAX = 2**31 - 1
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_UINT32_MAX = 2**32 - 1
_UINT64_MAX = 2**64 - 1
_INT32_WIDTH: NativeBitWidth = 32
_INT64_WIDTH: NativeBitWidth = 64
_MAX_ENUMERATED_VALUES = 4096
_MAX_SHIFT_BITS = 63
_MAX_POWER_EXPONENT = 8


def _decimal_text(value: int) -> str:
    return str(value)


def _reason(code: ReasonCode, message: str) -> OperationReason:
    return OperationReason(code=code, message=message)


def _reject(
    operation: str,
    operands: tuple[ClosedIntInterval, ...],
    reason: OperationReason,
) -> OperationProof:
    return OperationProof(
        operation=operation,
        operands=operands,
        status="rejected",
        result=None,
        reasons=(reason,),
    )


def _proved(
    operation: str,
    operands: tuple[ClosedIntInterval, ...],
    result: ClosedIntInterval,
    *,
    note: str,
) -> OperationProof:
    return OperationProof(
        operation=operation,
        operands=operands,
        status="proved",
        result=result,
        reasons=(OperationReason(code="operation-proved", message=note),),
    )


def _enumerable(interval: ClosedIntInterval) -> bool:
    return interval.cardinality <= _MAX_ENUMERATED_VALUES


def _binary_enumerable(left: ClosedIntInterval, right: ClosedIntInterval) -> bool:
    return left.cardinality * right.cardinality <= _MAX_ENUMERATED_VALUES


def _values(interval: ClosedIntInterval) -> range:
    return range(interval.minimum, interval.maximum + 1)


@dataclass(frozen=True, slots=True)
class SymbolicBound:
    """Exact integer bound with report-facing decimal text.

    The `decimal` field is intentionally plain text so future report renderers
    and guard builders can emit stable domain predicates without reparsing
    arbitrary source expressions. Instances are safe to compare, persist, and
    pass between analyzer phases because they contain only immutable scalar
    data.

    Attributes:
        value: Exact integer value used by interval arithmetic.
        decimal: Canonical decimal-string spelling of `value` for diagnostics
            and later constant-time guard rendering.
    """

    value: int
    decimal: str

    def __post_init__(self) -> None:
        """Reject decimal spellings that do not match the exact value.

        Raises:
            ValueError: If `decimal` is not the base-10 spelling of `value`.
        """
        if self.decimal != _decimal_text(self.value):
            raise ValueError("symbolic bound decimal text must match its integer value")

    @classmethod
    def exact(cls, value: int) -> SymbolicBound:
        """Build a symbolic bound from an exact integer.

        Args:
            value: Exact integer represented by the bound.

        Returns:
            SymbolicBound: Bound with canonical decimal report text.
        """
        return cls(value=value, decimal=_decimal_text(value))


@dataclass(frozen=True, slots=True)
class ClosedIntInterval:
    """Inclusive integer interval used by scalar proof arithmetic.

    Intervals are always finite and closed. The model stores symbolic endpoint
    wrappers rather than raw strings so native-width checks and later predicate
    generation can share the same exact endpoint values without trusting text.

    Attributes:
        minimum: Inclusive lower integer bound.
        maximum: Inclusive upper integer bound.
        lower: Symbolic lower endpoint used for decimal-string reporting.
        upper: Symbolic upper endpoint used for decimal-string reporting.
    """

    minimum: int
    maximum: int
    lower: SymbolicBound
    upper: SymbolicBound

    def __post_init__(self) -> None:
        """Reject inverted intervals or mismatched symbolic endpoints.

        Raises:
            ValueError: If the bounds are inverted or symbolic endpoints do not
                match `minimum` and `maximum`.
        """
        if self.minimum > self.maximum:
            raise ValueError("closed integer interval minimum must be <= maximum")
        if self.lower.value != self.minimum or self.upper.value != self.maximum:
            raise ValueError("symbolic bounds must match interval endpoints")

    @classmethod
    def closed(cls, minimum: int, maximum: int) -> ClosedIntInterval:
        """Build a closed interval from exact integer endpoints.

        Args:
            minimum: Inclusive lower bound.
            maximum: Inclusive upper bound.

        Returns:
            ClosedIntInterval: Finite closed interval with canonical endpoint text.
        """
        return cls(
            minimum=minimum,
            maximum=maximum,
            lower=SymbolicBound.exact(minimum),
            upper=SymbolicBound.exact(maximum),
        )

    @classmethod
    def point(cls, value: int) -> ClosedIntInterval:
        """Build an interval containing one exact integer.

        Args:
            value: Exact integer value.

        Returns:
            ClosedIntInterval: Singleton closed interval.
        """
        return cls.closed(value, value)

    @property
    def cardinality(self) -> int:
        """Return the number of integer values inside the interval.

        Returns:
            int: Inclusive endpoint count.
        """
        return self.maximum - self.minimum + 1

    @property
    def is_singleton(self) -> bool:
        """Return whether the interval contains exactly one value.

        Returns:
            bool: `True` when `minimum == maximum`.
        """
        return self.minimum == self.maximum

    @property
    def is_nonnegative(self) -> bool:
        """Return whether all values in the interval are at least zero.

        Returns:
            bool: `True` when no negative integer is represented.
        """
        return self.minimum >= 0

    def contains(self, value: int) -> bool:
        """Return whether `value` is inside the closed interval.

        Args:
            value: Integer to test.

        Returns:
            bool: Whether `minimum <= value <= maximum`.
        """
        return self.minimum <= value <= self.maximum

    def fits_native(self, native: NativeInteger) -> bool:
        """Return whether both endpoints fit a native integer representation.

        Args:
            native: Width and signedness domain being considered.

        Returns:
            bool: `True` when every interval member is representable.
        """
        return native.minimum <= self.minimum and self.maximum <= native.maximum


@dataclass(frozen=True, slots=True)
class NativeInteger:
    """Native integer width and signedness domain.

    Attributes:
        width: Integer storage width in bits. Only 32-bit and 64-bit scalar
            domains are supported by milestone 3.
        signed: Whether the native representation is signed.
    """

    width: NativeBitWidth
    signed: bool = True

    def __post_init__(self) -> None:
        """Reject unsupported widths.

        Raises:
            ValueError: If `width` is not 32 or 64.
        """
        if self.width not in (_INT32_WIDTH, _INT64_WIDTH):
            raise ValueError("native integer width must be 32 or 64 bits")

    @property
    def minimum(self) -> int:
        """Return the smallest representable integer.

        Returns:
            int: Signed lower bound or zero for unsigned domains.
        """
        if not self.signed:
            return 0
        return _INT32_MIN if self.width == _INT32_WIDTH else _INT64_MIN

    @property
    def maximum(self) -> int:
        """Return the largest representable integer.

        Returns:
            int: Inclusive upper bound for the configured native domain.
        """
        if self.width == _INT32_WIDTH:
            return _INT32_MAX if self.signed else _UINT32_MAX
        return _INT64_MAX if self.signed else _UINT64_MAX

    @property
    def interval(self) -> ClosedIntInterval:
        """Return the exact interval represented by this native domain.

        Returns:
            ClosedIntInterval: Closed interval covering all native values.
        """
        return ClosedIntInterval.closed(self.minimum, self.maximum)


@dataclass(frozen=True, slots=True)
class OperationReason:
    """Structured reason attached to an arithmetic proof.

    Attributes:
        code: Stable machine-readable failure or proof-note code.
        message: Human-readable diagnostic suitable for reports.
    """

    code: ReasonCode
    message: str

    def __post_init__(self) -> None:
        """Reject blank report messages.

        Raises:
            ValueError: If `message` is empty or only whitespace.
        """
        if not self.message.strip():
            raise ValueError("operation reason message must be non-empty")


@dataclass(frozen=True, slots=True)
class OperationProof:
    """Proof record for one scalar operation.

    A proved operation carries an exact conservative result interval. A rejected
    operation carries no result and explains the rejected condition through
    stable reason codes instead of raising analyzer-facing exceptions.

    Attributes:
        operation: Stable operation name such as `add`, `floordiv`, or `lshift`.
        operands: Exact input intervals used for the proof.
        status: Whether the operation was proven or rejected.
        result: Exact result interval when `status` is `proved`.
        reasons: Non-empty proof notes or rejection reasons.
    """

    operation: str
    operands: tuple[ClosedIntInterval, ...]
    status: ProofStatus
    result: ClosedIntInterval | None
    reasons: tuple[OperationReason, ...]

    def __post_init__(self) -> None:
        """Reject inconsistent proof records.

        Raises:
            ValueError: If result presence does not match `status` or reasons
                are missing.
        """
        if not self.operation.strip():
            raise ValueError("operation name must be non-empty")
        if not self.reasons:
            raise ValueError("operation proofs require at least one reason")
        if self.status == "proved" and self.result is None:
            raise ValueError("proved operation proofs require a result interval")
        if self.status == "rejected" and self.result is not None:
            raise ValueError("rejected operation proofs cannot carry a result interval")

    @property
    def proved(self) -> bool:
        """Return whether the operation has a result interval.

        Returns:
            bool: `True` when `status == "proved"`.
        """
        return self.status == "proved"

    def fits_native(self, native: NativeInteger) -> bool:
        """Return whether every proven operand and result fits `native`.

        Args:
            native: Native integer representation to test.

        Returns:
            bool: `False` for rejected operations or when an operand or result
                exceeds the native domain. Operand checks prevent a small result
                from hiding an unrepresentable Python literal or module constant.
        """
        return (
            self.result is not None
            and self.result.fits_native(native)
            and all(operand.fits_native(native) for operand in self.operands)
        )


@dataclass(frozen=True, slots=True)
class RangeLoopInduction:
    """Closed induction summary for a bounded Python `range` loop.

    Attributes:
        start: Exact range start value.
        stop: Exact exclusive range stop value.
        step: Exact non-zero range step value.
        iterations: Number of loop iterations.
        index_interval: Closed interval containing every yielded loop index.
    """

    start: int
    stop: int
    step: int
    iterations: int
    index_interval: ClosedIntInterval

    def __post_init__(self) -> None:
        """Reject inconsistent bounded range summaries.

        Raises:
            ValueError: If `step` is zero, iteration count is negative, or a
                non-empty range has an index interval that misses an endpoint.
        """
        if self.step == 0:
            raise ValueError("range-loop induction step must be non-zero")
        if self.iterations < 0:
            raise ValueError("range-loop induction iterations must be non-negative")
        if self.iterations > 0:
            last = self.start + ((self.iterations - 1) * self.step)
            if not self.index_interval.contains(self.start) or not self.index_interval.contains(
                last
            ):
                raise ValueError("range-loop induction interval must include first and last index")


def add(left: ClosedIntInterval, right: ClosedIntInterval) -> OperationProof:
    """Prove exact conservative bounds for interval addition.

    Args:
        left: Left operand interval.
        right: Right operand interval.

    Returns:
        OperationProof: Proved result interval for `left + right`.
    """
    return _proved(
        "add",
        (left, right),
        ClosedIntInterval.closed(left.minimum + right.minimum, left.maximum + right.maximum),
        note="addition is monotonic over closed integer intervals",
    )


def subtract(left: ClosedIntInterval, right: ClosedIntInterval) -> OperationProof:
    """Prove exact conservative bounds for interval subtraction.

    Args:
        left: Left operand interval.
        right: Right operand interval.

    Returns:
        OperationProof: Proved result interval for `left - right`.
    """
    return _proved(
        "subtract",
        (left, right),
        ClosedIntInterval.closed(left.minimum - right.maximum, left.maximum - right.minimum),
        note="subtraction extrema are formed from opposite endpoints",
    )


def multiply(left: ClosedIntInterval, right: ClosedIntInterval) -> OperationProof:
    """Prove exact conservative bounds for interval multiplication.

    Args:
        left: Left operand interval.
        right: Right operand interval.

    Returns:
        OperationProof: Proved result interval for `left * right`.
    """
    candidates = (
        left.minimum * right.minimum,
        left.minimum * right.maximum,
        left.maximum * right.minimum,
        left.maximum * right.maximum,
    )
    return _proved(
        "multiply",
        (left, right),
        ClosedIntInterval.closed(min(candidates), max(candidates)),
        note="multiplication extrema occur at interval endpoint products",
    )


def compare(
    left: ClosedIntInterval,
    operator: ComparisonOperator,
    right: ClosedIntInterval,
) -> OperationProof:
    """Prove conservative boolean bounds for an integer comparison.

    Args:
        left: Left comparison operand.
        operator: Comparison operator.
        right: Right comparison operand.

    Returns:
        OperationProof: Boolean-as-integer interval `[1, 1]`, `[0, 0]`, or
        `[0, 1]` when the relation depends on runtime values.
    """
    true_for_all = _comparison_always_true(left, operator, right)
    false_for_all = _comparison_always_false(left, operator, right)
    if true_for_all:
        result = ClosedIntInterval.point(1)
    elif false_for_all:
        result = ClosedIntInterval.point(0)
    else:
        result = ClosedIntInterval.closed(0, 1)
    return _proved(
        f"compare-{operator}",
        (left, right),
        result,
        note="comparison result is represented as a boolean integer interval",
    )


def floor_divide(left: ClosedIntInterval, right: ClosedIntInterval) -> OperationProof:
    """Prove bounds for nonnegative floor division.

    Args:
        left: Dividend interval. Every value must be nonnegative.
        right: Divisor interval. Every value must be positive.

    Returns:
        OperationProof: Proved quotient interval or a structured rejection for
        possible zero divisors or unsupported signed domains.
    """
    if right.contains(0):
        return _reject(
            "floordiv",
            (left, right),
            _reason("division-by-possible-zero", "floor division divisor may be zero"),
        )
    if not left.is_nonnegative or right.minimum <= 0:
        return _reject(
            "floordiv",
            (left, right),
            _reason("negative-divmod-domain", "floor division proof requires nonnegative inputs"),
        )
    return _proved(
        "floordiv",
        (left, right),
        ClosedIntInterval.closed(left.minimum // right.maximum, left.maximum // right.minimum),
        note="nonnegative floor division is monotonic in dividend and inverse-monotonic in divisor",
    )


def modulo(left: ClosedIntInterval, right: ClosedIntInterval) -> OperationProof:
    """Prove bounds for nonnegative modulo.

    Args:
        left: Dividend interval. Every value must be nonnegative.
        right: Divisor interval. Every value must be positive.

    Returns:
        OperationProof: Proved remainder interval or a structured rejection for
        possible zero divisors or unsupported signed domains.
    """
    if right.contains(0):
        return _reject(
            "modulo",
            (left, right),
            _reason("division-by-possible-zero", "modulo divisor may be zero"),
        )
    if not left.is_nonnegative or right.minimum <= 0:
        return _reject(
            "modulo",
            (left, right),
            _reason("negative-divmod-domain", "modulo proof requires nonnegative inputs"),
        )
    return _proved(
        "modulo",
        (left, right),
        ClosedIntInterval.closed(0, min(left.maximum, right.maximum - 1)),
        note="nonnegative modulo remainder is bounded by dividend and divisor minus one",
    )


def bitwise(
    left: ClosedIntInterval,
    operator: BitwiseOperator,
    right: ClosedIntInterval,
) -> OperationProof:
    """Prove exact bounds for small nonnegative bitwise domains.

    Args:
        left: Left bitwise operand.
        operator: Bitwise operator.
        right: Right bitwise operand.

    Returns:
        OperationProof: Proved result for enumerably small domains, or a
        structured rejection when signed or too-large domains make the result
        unproven.
    """
    if not left.is_nonnegative or not right.is_nonnegative:
        return _reject(
            f"bitwise-{operator}",
            (left, right),
            _reason("negative-bitwise-domain", "bitwise proof requires nonnegative operands"),
        )
    if not _binary_enumerable(left, right):
        return _reject(
            f"bitwise-{operator}",
            (left, right),
            _reason("unproven-operation", "bitwise domain is too large for exact proof"),
        )
    outputs = tuple(
        _apply_bitwise(value, operator, mask) for value in _values(left) for mask in _values(right)
    )
    return _proved(
        f"bitwise-{operator}",
        (left, right),
        ClosedIntInterval.closed(min(outputs), max(outputs)),
        note="bitwise bounds were proven by exhaustive finite-domain evaluation",
    )


def shift(
    value: ClosedIntInterval,
    operator: ShiftOperator,
    count: ClosedIntInterval,
) -> OperationProof:
    """Prove guarded nonnegative shift bounds.

    Args:
        value: Integer value being shifted. Every value must be nonnegative.
        operator: Shift direction.
        count: Shift-count interval. Every value must be between zero and the
            configured milestone shift limit.

    Returns:
        OperationProof: Proved shift interval or a structured rejection for
        negative, unbounded, or unsupported shift domains.
    """
    if not value.is_nonnegative:
        return _reject(
            f"shift-{operator}",
            (value, count),
            _reason("negative-bitwise-domain", "shift proof requires nonnegative values"),
        )
    if count.minimum < 0:
        return _reject(
            f"shift-{operator}",
            (value, count),
            _reason("negative-shift-count", "shift count may be negative"),
        )
    if count.maximum > _MAX_SHIFT_BITS:
        return _reject(
            f"shift-{operator}",
            (value, count),
            _reason("unbounded-shift-count", "shift count exceeds the guarded shift limit"),
        )
    if operator == "<<":
        result = ClosedIntInterval.closed(
            value.minimum << count.minimum, value.maximum << count.maximum
        )
    else:
        result = ClosedIntInterval.closed(
            value.minimum >> count.maximum, value.maximum >> count.minimum
        )
    return _proved(
        f"shift-{operator}",
        (value, count),
        result,
        note="nonnegative guarded shifts are monotonic over value and shift count",
    )


def power(base: ClosedIntInterval, exponent: ClosedIntInterval) -> OperationProof:
    """Prove exact bounds for small constant integer powers.

    Args:
        base: Base interval.
        exponent: Exponent interval, which must be a singleton in the supported
            nonnegative milestone range.

    Returns:
        OperationProof: Proved power interval or a structured rejection for
        unsupported exponent domains.
    """
    if not exponent.is_singleton or exponent.minimum < 0 or exponent.maximum > _MAX_POWER_EXPONENT:
        return _reject(
            "power",
            (base, exponent),
            _reason("unsafe-power", "power proof requires a small nonnegative constant exponent"),
        )
    if base.minimum < 0 and not _enumerable(base):
        return _reject(
            "power",
            (base, exponent),
            _reason("unsafe-power", "negative-base power proof requires a small enumerable domain"),
        )
    exp = exponent.minimum
    if base.minimum >= 0:
        result = ClosedIntInterval.closed(base.minimum**exp, base.maximum**exp)
    else:
        outputs = tuple(value**exp for value in _values(base))
        result = ClosedIntInterval.closed(min(outputs), max(outputs))
    return _proved(
        "power",
        (base, exponent),
        result,
        note="small constant power bounds were proven exactly",
    )


def join(*intervals: ClosedIntInterval) -> OperationProof:
    """Join branch intervals into one conservative closed interval.

    Args:
        *intervals: Branch result intervals to merge.

    Returns:
        OperationProof: Proved interval spanning every branch, or a structured
        rejection when no branch interval is supplied.
    """
    if not intervals:
        return OperationProof(
            operation="join",
            operands=(),
            status="rejected",
            result=None,
            reasons=(_reason("unsupported-operation", "join requires at least one interval"),),
        )
    return _proved(
        "join",
        intervals,
        ClosedIntInterval.closed(
            min(interval.minimum for interval in intervals),
            max(interval.maximum for interval in intervals),
        ),
        note="branch join spans the minimum and maximum reachable endpoints",
    )


def build_range_induction(start: int, stop: int, step: int = 1) -> OperationProof:
    """Build a bounded induction interval for a Python `range`.

    Args:
        start: Exact range start argument.
        stop: Exact exclusive range stop argument.
        step: Exact non-zero range step argument.

    Returns:
        OperationProof: Proved loop-index interval, or a structured empty-range
        rejection when the loop has no iterations.

    Raises:
        ValueError: If `step` is zero, matching Python `range` construction.
    """
    if step == 0:
        raise ValueError("range step cannot be zero")
    iterations = len(range(start, stop, step))
    if iterations == 0:
        return OperationProof(
            operation="range-induction",
            operands=(),
            status="rejected",
            result=None,
            reasons=(_reason("empty-range", "range loop has no induction values"),),
        )
    last = start + ((iterations - 1) * step)
    result = ClosedIntInterval.closed(min(start, last), max(start, last))
    return OperationProof(
        operation="range-induction",
        operands=(),
        status="proved",
        result=result,
        reasons=(
            OperationReason(
                code="operation-proved",
                message=f"range loop has {iterations} bounded induction values",
            ),
        ),
    )


def accumulate_additive(
    initial: ClosedIntInterval,
    per_iteration: ClosedIntInterval,
    iterations: int,
) -> OperationProof:
    """Accumulate a bounded additive loop update.

    Args:
        initial: Interval before the loop starts.
        per_iteration: Interval added once per loop iteration.
        iterations: Exact nonnegative iteration count.

    Returns:
        OperationProof: Proved accumulator interval after all iterations, or a
        structured rejection for a negative iteration count.
    """
    if iterations < 0:
        return _reject(
            "accumulate-additive",
            (initial, per_iteration),
            _reason("unsupported-operation", "accumulation iterations must be nonnegative"),
        )
    scaled = ClosedIntInterval.closed(
        per_iteration.minimum * iterations,
        per_iteration.maximum * iterations,
    )
    result = cast(ClosedIntInterval, add(initial, scaled).result)
    return _proved(
        "accumulate-additive",
        (initial, per_iteration),
        result,
        note="bounded additive accumulation scales the per-iteration interval by trip count",
    )


def every_intermediate_fits(
    native: NativeInteger,
    proofs: tuple[OperationProof, ...],
) -> bool:
    """Return whether every proof result fits a native integer domain.

    Args:
        native: Native integer domain to test.
        proofs: Operation proofs in evaluation order.

    Returns:
        bool: `True` only when every proof is proved and every result interval
        is representable by `native`.
    """
    return all(proof.fits_native(native) for proof in proofs)


def _comparison_always_true(
    left: ClosedIntInterval,
    operator: ComparisonOperator,
    right: ClosedIntInterval,
) -> bool:
    if operator == "<":
        return left.maximum < right.minimum
    if operator == "<=":
        return left.maximum <= right.minimum
    if operator == ">":
        return left.minimum > right.maximum
    if operator == ">=":
        return left.minimum >= right.maximum
    if operator == "==":
        return left.is_singleton and right.is_singleton and left.minimum == right.minimum
    return left.maximum < right.minimum or right.maximum < left.minimum


def _comparison_always_false(
    left: ClosedIntInterval,
    operator: ComparisonOperator,
    right: ClosedIntInterval,
) -> bool:
    if operator == "<":
        return left.minimum >= right.maximum
    if operator == "<=":
        return left.minimum > right.maximum
    if operator == ">":
        return left.maximum <= right.minimum
    if operator == ">=":
        return left.maximum < right.minimum
    if operator == "==":
        return left.maximum < right.minimum or right.maximum < left.minimum
    return left.is_singleton and right.is_singleton and left.minimum == right.minimum


def _apply_bitwise(left: int, operator: BitwiseOperator, right: int) -> int:
    if operator == "&":
        return left & right
    if operator == "|":
        return left | right
    return left ^ right
