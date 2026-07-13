"""Tests for conservative fixed-width scalar candidate analysis."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from textwrap import dedent, indent

import pytest

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.clustering import enrich_island_analysis
from atoll.models import Blocker, ModuleId, ModuleScan, RegionMember, SymbolId, TypedRegion
from atoll.native_optimization.models import (
    ExactTypeGuardPayload,
    IntegerDomainGuardPayload,
)
from atoll.native_optimization.scalar_analysis import (
    ScalarKernelPlan,
    ScalarRejection,
    analyze_scalar_member,
    analyze_scalar_scan,
)

_POLYNOMIAL_START_LINE = 3
_POLYNOMIAL_END_LINE = 7
_POLYNOMIAL_OPERATION_START_LINE = 5
_MOVED_DECLARATION_LINE = 3
_FIRST_OPERATION_LINE = 2
_MOVED_OPERATION_LINE = 4


def _scan(tmp_path: Path, source: str) -> ModuleScan:
    path = tmp_path / "scalar_subject.py"
    path.write_text(dedent(source).strip() + "\n", encoding="utf-8")
    return enrich_island_analysis(scan_module(ModuleId(name="scalar_subject", path=path)))


def _member(scan: ModuleScan, qualname: str) -> tuple[TypedRegion, RegionMember]:
    return next(
        (region, member)
        for region in scan.typed_regions
        for member in region.members
        if member.id.qualname == qualname
    )


def _plan(scan: ModuleScan, qualname: str) -> ScalarKernelPlan:
    return next(
        plan for plan in analyze_scalar_scan(scan).plans if plan.member.qualname == qualname
    )


def _direct_source_outcome(tmp_path: Path, body: str) -> ScalarKernelPlan | ScalarRejection:
    scan = _scan(
        tmp_path,
        """
        def subject(value: int) -> int:
            return value
        """,
    )
    region, member = _member(scan, "subject")
    source = "def subject(value: int) -> int:\n" + indent(dedent(body).strip(), "    ") + "\n"
    return analyze_scalar_member(region, replace(member, source_text=source))


def test_scalar_analysis_proves_polynomial_loop_and_stable_guards(tmp_path: Path) -> None:
    """A bounded additive polynomial loop receives safe 32-bit and 64-bit variants."""
    scan = _scan(
        tmp_path,
        """
        OFFSET = 3

        def polynomial(limit: int, bias: int = 1) -> int:
            total = 0
            for value in range(limit):
                total += value * value + OFFSET
            return total + bias
        """,
    )

    plan = _plan(scan, "polynomial")
    repeated = _plan(scan, "polynomial")

    assert plan.id == repeated.id
    assert plan.source_hash == repeated.source_hash
    assert tuple(proof.native.width for proof in plan.width_proofs) == (32, 64)
    assert plan.declaration_start_lineno == _POLYNOMIAL_START_LINE
    assert plan.end_lineno == _POLYNOMIAL_END_LINE
    for width_proof in plan.width_proofs:
        assert width_proof.return_interval.fits_native(width_proof.native)
        assert all(
            record.proof.fits_native(width_proof.native) for record in width_proof.operations
        )
        assert [domain.name for domain in width_proof.parameters] == ["limit", "bias"]
        assert all(domain.interval.minimum == 0 for domain in width_proof.parameters)
        assert all(domain.interval.maximum > 0 for domain in width_proof.parameters)
        assert (
            min(record.lineno for record in width_proof.operations)
            >= _POLYNOMIAL_OPERATION_START_LINE
        )
        exact_guards = [
            guard.payload
            for guard in width_proof.guards
            if isinstance(guard.payload, ExactTypeGuardPayload)
        ]
        domain_guards = [
            guard.payload
            for guard in width_proof.guards
            if isinstance(guard.payload, IntegerDomainGuardPayload)
        ]
        assert [guard.subject for guard in exact_guards] == ["limit", "bias"]
        assert all(guard.type_module == "builtins" for guard in exact_guards)
        assert all(guard.type_qualname == "int" for guard in exact_guards)
        assert [guard.subject for guard in domain_guards] == ["limit", "bias"]
        assert all(guard.bit_width == width_proof.native.width for guard in domain_guards)


def test_scalar_analysis_accepts_staticmethod_branches_keywords_and_bit_ops(
    tmp_path: Path,
) -> None:
    """Static methods retain keyword-only defaults and conservative branch joins."""
    scan = _scan(
        tmp_path,
        """
        class Kernels:
            @staticmethod
            def branch(value: int, *, offset: int = 2) -> int:
                if value < 16:
                    result = (value * value) & 255
                else:
                    result = (value >> 2) + offset
                return result
        """,
    )

    plan = _plan(scan, "Kernels.branch")

    assert tuple(proof.native.width for proof in plan.width_proofs) == (32, 64)
    assert (
        any(
            operation.proof.operation == "join"
            for proof in plan.width_proofs
            for operation in proof.operations
        )
        is False
    )
    assert any(
        operation.proof.operation == "bitwise-&"
        for proof in plan.width_proofs
        for operation in proof.operations
    )


def test_scalar_analysis_requires_positive_divisor_domain(tmp_path: Path) -> None:
    """A divisor parameter guard excludes zero before native execution starts."""
    scan = _scan(
        tmp_path,
        """
        def quotient(value: int, divisor: int) -> int:
            return value // divisor
        """,
    )

    plan = _plan(scan, "quotient")

    for proof in plan.width_proofs:
        domains = {domain.name: domain.interval for domain in proof.parameters}
        assert domains["value"].minimum == 0
        assert domains["divisor"].minimum == 1


@pytest.mark.parametrize(
    "expression",
    [
        "HUGE % (value + 1)",
        "1 if value < HUGE else 0",
        "value & HUGE",
    ],
)
def test_scalar_analysis_rejects_huge_operands_even_when_result_is_small(
    tmp_path: Path,
    expression: str,
) -> None:
    """Every native operand must fit; checking only the final result is unsafe."""
    scan = _scan(
        tmp_path,
        f"""
        HUGE = {2**100}

        def narrowed(value: int) -> int:
            return {expression}
        """,
    )

    result = analyze_scalar_scan(scan)
    rejection = next(item for item in result.rejections if item.member.qualname == "narrowed")

    assert rejection.code == "unproven-arithmetic"


@pytest.mark.parametrize(
    "source",
    [
        """
        def shadowed(range: int) -> int:
            total = 0
            for value in range(4):
                total += value
            return total
        """,
        """
        range = 4
        def shadowed(value: int) -> int:
            total = 0
            for item in range(value):
                total += item
            return total
        """,
        """
        def range(value: int) -> int:
            return value
        def shadowed(value: int) -> int:
            total = 0
            for item in range(value):
                total += item
            return total
        """,
    ],
)
def test_scalar_analysis_rejects_shadowed_range(tmp_path: Path, source: str) -> None:
    """Bounded-loop proof requires the actual builtins.range callable."""
    scan = _scan(tmp_path, source)
    result = analyze_scalar_scan(scan)
    rejection = next(item for item in result.rejections if item.member.qualname == "shadowed")

    assert rejection.code in {"opaque-call", "unsupported-scope"}


def test_scalar_analysis_rejects_unsafe_callable_shapes(tmp_path: Path) -> None:
    """Instance methods, async code, opaque calls, variadics, and recurrence fall back."""
    scan = _scan(
        tmp_path,
        """
        def opaque(value: int) -> int:
            return abs(value)

        def variadic(*values: int) -> int:
            return 0

        def recurrence(limit: int) -> int:
            total = 1
            for value in range(limit):
                total = total * 3 + value
            return total

        async def asynchronous(value: int) -> int:
            return value

        class Worker:
            def instance(self, value: int) -> int:
                return value + 1
        """,
    )
    result = analyze_scalar_scan(scan)
    rejected = {rejection.member.qualname: rejection for rejection in result.rejections}

    assert rejected["opaque"].code in {"unsupported-scope", "opaque-call"}
    assert rejected["variadic"].code == "unsupported-signature"
    assert rejected["recurrence"].code == "unproven-arithmetic"
    assert rejected["asynchronous"].code == "unsupported-execution"
    assert rejected["Worker.instance"].code == "unsupported-binding"


def test_scalar_analysis_rejects_nonliteral_power_exponents(tmp_path: Path) -> None:
    """Proof and code generation agree that power exponents are AST literals."""
    scan = _scan(
        tmp_path,
        """
        EXPONENT = 2

        def named_power(value: int) -> int:
            return value ** EXPONENT

        def local_power(value: int) -> int:
            exponent = 2
            return value ** exponent
        """,
    )

    result = analyze_scalar_scan(scan)
    rejected = {rejection.member.qualname: rejection for rejection in result.rejections}

    assert rejected["named_power"].code == "unproven-arithmetic"
    assert rejected["local_power"].code == "unproven-arithmetic"


def test_scalar_member_rejects_annotations_defaults_and_external_mutation(tmp_path: Path) -> None:
    """Scalar proof never invents integer semantics for incomplete declarations."""
    cases = (
        (
            "any_value",
            """
            import typing
            def any_value(value: typing.Any) -> int:
                return 1
            """,
            "unsupported-annotation",
        ),
        (
            "bad_default",
            """
            def bad_default(value: int = True) -> int:
                return value
            """,
            "unsupported-signature",
        ),
        (
            "mutate",
            """
            state = 0
            def mutate(value: int) -> int:
                global state
                state = value
                return state
            """,
            "unsupported-scope",
        ),
    )
    for qualname, source, expected in cases:
        scan = _scan(tmp_path, source)
        region, member = _member(scan, qualname)
        outcome = analyze_scalar_member(region, member)
        assert isinstance(outcome, ScalarRejection)
        assert outcome.code == expected


def test_scalar_analysis_plan_identity_ignores_absolute_source_location(tmp_path: Path) -> None:
    """Moving unchanged source changes report spans but not proof-plan identity."""
    first = _scan(
        tmp_path,
        """
        def square(value: int) -> int:
            return value * value
        """,
    )
    first_plan = _plan(first, "square")
    moved = _scan(
        tmp_path,
        """
        SENTINEL = 1

        def square(value: int) -> int:
            return value * value
        """,
    )
    moved_plan = _plan(moved, "square")

    assert first_plan.id == moved_plan.id
    assert first_plan.declaration_start_lineno == 1
    assert moved_plan.declaration_start_lineno == _MOVED_DECLARATION_LINE
    assert first_plan.width_proofs[0].operations[0].lineno == _FIRST_OPERATION_LINE
    assert moved_plan.width_proofs[0].operations[0].lineno == _MOVED_OPERATION_LINE


def test_scalar_proof_models_are_immutable(tmp_path: Path) -> None:
    """Analysis results can be persisted and shared without mutable proof state."""
    scan = _scan(
        tmp_path,
        """
        def increment(value: int) -> int:
            return value + 1
        """,
    )
    plan = _plan(scan, "increment")

    with pytest.raises(FrozenInstanceError):
        plan.__setattr__("id", "changed")


def test_scalar_rejection_requires_a_message(tmp_path: Path) -> None:
    """Structured fallback evidence cannot carry an empty explanation."""
    scan = _scan(
        tmp_path,
        """
        def subject(value: int) -> int:
            return value
        """,
    )
    _, member = _member(scan, "subject")

    with pytest.raises(ValueError, match="non-empty"):
        ScalarRejection(member=member.id, code="unsupported-scope", message=" ")


def test_scalar_member_contract_rejects_generic_empty_and_malformed_declarations(
    tmp_path: Path,
) -> None:
    """Direct frontend entry points retain all declaration-shape safety checks."""
    scan = _scan(
        tmp_path,
        """
        def subject(value: int) -> int:
            return value
        """,
    )
    region, member = _member(scan, "subject")
    cases = (
        replace(member, type_parameters=("T",)),
        replace(member, parameters=()),
        replace(
            member,
            source_text=(
                member.source_text + "\ndef extra(value: int) -> int:\n    return value\n"
            ),
        ),
        replace(member, source_text="@decorator\n" + member.source_text),
        replace(
            member,
            parameters=(replace(member.parameters[0], default_source="factory()"),),
        ),
    )

    outcomes = tuple(analyze_scalar_member(region, case) for case in cases)

    assert all(isinstance(outcome, ScalarRejection) for outcome in outcomes)


@pytest.mark.parametrize(
    ("body", "accepted"),
    [
        ("    value += 1\n    return value", True),
        ("    pass\n    return +(-value)", True),
        ("    value.attribute = value\n    return value", False),
        ("    value.attribute += 1\n    return value", False),
        ("    local: str = value\n    return local", False),
        ("    local: int\n    return value", False),
        ("    pass", False),
        ("    while value:\n        value -= 1\n    return value", False),
        ("    return", False),
        ("    return int(value)", False),
        ("    return value / 2", False),
        ("    return [value]", False),
        ("    return 1 if value and 1 else 0", False),
        ("    return 1 if value is 1 else 0", False),
        ("    nested = lambda: value\n    return value", False),
    ],
)
def test_scalar_statement_and_expression_matrix(
    tmp_path: Path,
    body: str,
    accepted: bool,
) -> None:
    """Local syntax either receives a complete proof or an explicit fallback."""
    outcome = _direct_source_outcome(tmp_path, body)

    assert isinstance(outcome, ScalarKernelPlan) is accepted


@pytest.mark.parametrize(
    ("body", "accepted"),
    [
        (
            "    total = 0\n"
            "    for item in range(1, value, 2):\n"
            "        total = item + total\n"
            "    return total",
            True,
        ),
        (
            "    total = 0\n    for item in range(0):\n        total += item\n    return total",
            True,
        ),
        (
            "    total = 0\n"
            "    for left, right in range(value):\n"
            "        total += left\n"
            "    return total",
            False,
        ),
        (
            "    total = 0\n"
            "    for item in range(value):\n"
            "        total += item\n"
            "    else:\n"
            "        total += 1\n"
            "    return total",
            False,
        ),
        (
            "    total = 1\n    for item in range(value):\n        total *= item\n    return total",
            False,
        ),
        (
            "    total = 0\n"
            "    for item in range(value, 10):\n"
            "        total += item\n"
            "    return total",
            True,
        ),
        (
            "    total = 0\n"
            "    for item in range(0, value, -1):\n"
            "        total += item\n"
            "    return total",
            False,
        ),
        (
            "    total = 0\n"
            "    for item in range(stop=value):\n"
            "        total += item\n"
            "    return total",
            False,
        ),
        (
            "    total = 0\n    for item in range(-1):\n        total += item\n    return total",
            False,
        ),
        (
            "    total = 0\n    for item in [1]:\n        total += item\n    return total",
            False,
        ),
    ],
)
def test_scalar_range_loop_matrix(tmp_path: Path, body: str, accepted: bool) -> None:
    """Only bounded builtin ranges and additive reductions enter native plans."""
    outcome = _direct_source_outcome(tmp_path, body)

    assert isinstance(outcome, ScalarKernelPlan) is accepted


@pytest.mark.parametrize(
    "expression",
    [
        "value - 1",
        "value ** 2",
        "value << 1",
        "value >> 1",
        "value | 1",
        "value ^ 1",
        "value // 2",
        "value % 2",
        "1 if value <= 2 else 0",
        "1 if value > 2 else 0",
        "1 if value >= 2 else 0",
        "1 if value == 2 else 0",
        "1 if value != 2 else 0",
    ],
)
def test_scalar_supported_operator_matrix(tmp_path: Path, expression: str) -> None:
    """Every advertised scalar operator produces at least one guarded width."""
    outcome = _direct_source_outcome(tmp_path, f"    return {expression}")

    assert isinstance(outcome, ScalarKernelPlan)


def test_scalar_scan_reports_missing_symbols_and_module_blockers(tmp_path: Path) -> None:
    """Scan-level drift and hard module blockers become explicit fallback evidence."""
    scan = _scan(
        tmp_path,
        """
        def subject(value: int) -> int:
            return value
        """,
    )
    region, member = _member(scan, "subject")
    drifted_member = replace(member, id=SymbolId(module="scalar_subject", qualname="missing"))
    drifted = replace(scan, typed_regions=(replace(region, members=(drifted_member,)),))
    blocked = replace(
        scan,
        blockers=(
            Blocker(
                severity="hard",
                code="MODULE_BLOCKED",
                message="module binding is dynamic",
                lineno=1,
            ),
        ),
    )

    drifted_result = analyze_scalar_scan(drifted)
    blocked_result = analyze_scalar_scan(blocked)

    assert drifted_result.rejections[0].member.qualname == "missing"
    assert "no matching scanner symbol" in drifted_result.rejections[0].message
    assert blocked_result.rejections[0].lineno == 1
    assert "MODULE_BLOCKED" in blocked_result.rejections[0].message
