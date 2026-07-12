"""Tests for immutable native-variant model contracts."""

from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from typing import cast

import pytest

from atoll.models import Backend, BindingTarget, SymbolId
from atoll.native_optimization import (
    BindingDispatchPlan,
    BufferLayoutGuardPayload,
    CallableCodeIdentityGuardPayload,
    CompiledBindingVariant,
    CompiledVariantKind,
    DirectFieldGuardPayload,
    ExactTypeGuardPayload,
    GuardExpression,
    GuardKind,
    GuardPayload,
    IntegerBitWidth,
    IntegerDomainGuardPayload,
    NativeVariantPlan,
    stable_native_variant_plan_id,
)

_INT32_WIDTH: IntegerBitWidth = 32
_INT64_WIDTH: IntegerBitWidth = 64


def _symbol(qualname: str) -> SymbolId:
    return SymbolId(module="app.worker", qualname=qualname)


def _binding(
    qualname: str = "run",
    *,
    owner_class: str | None = None,
    target_owner_class: str | None = None,
) -> BindingTarget:
    return BindingTarget(
        source=_symbol(qualname),
        compiled_name=f"_{qualname.replace('.', '_')}",
        kind="module" if owner_class is None else "instance_method",
        owner_class=owner_class,
        target_owner_class=target_owner_class,
        execution_kind="sync",
    )


def _int_guard(bit_width: IntegerBitWidth) -> GuardExpression:
    maximum = 2**31 - 1 if bit_width == _INT32_WIDTH else 2**63 - 1
    return GuardExpression(
        kind="integer-domain",
        payload=IntegerDomainGuardPayload(
            subject="count",
            minimum=0,
            maximum=maximum,
            bit_width=bit_width,
        ),
        message=f"count fits signed {bit_width}-bit native integer",
    )


def _generic_guards() -> tuple[GuardExpression, ...]:
    return (
        GuardExpression(
            kind="exact-type",
            payload=ExactTypeGuardPayload(
                subject="worker",
                type_module="app.worker",
                type_qualname="Worker",
            ),
            message="worker has the profiled concrete type",
        ),
        GuardExpression(
            kind="direct-field",
            payload=DirectFieldGuardPayload(
                owner_subject="worker",
                owner_type_module="app.worker",
                owner_type_qualname="Worker",
                field_name="factor",
                field_type="builtins.int",
            ),
            message="worker.factor is a direct typed field",
        ),
        GuardExpression(
            kind="callable-code-identity",
            payload=CallableCodeIdentityGuardPayload(
                subject="callback",
                callable_module="app.worker",
                callable_qualname="scale",
                code_fingerprint="code-v1",
            ),
            message="callback still points at the profiled code object",
        ),
        GuardExpression(
            kind="buffer-layout",
            payload=BufferLayoutGuardPayload(
                subject="values",
                format="i",
                itemsize=4,
                ndim=1,
                c_contiguous=True,
            ),
            message="values exposes the expected contiguous int buffer",
        ),
    )


def _variant(
    kind: CompiledVariantKind,
    backend: Backend,
    compiled_qualname: str,
) -> CompiledBindingVariant:
    guards = (_int_guard(32),) if kind == "safe-int32" else _generic_guards()
    if kind == "safe-int64":
        guards = (_int_guard(64),)
    return CompiledBindingVariant(
        backend=backend,
        kind=kind,
        binding=_binding(),
        compiled_symbol=_symbol(compiled_qualname),
        artifact_fingerprint=f"artifact-{compiled_qualname}",
        guards=guards,
        payload_fingerprint=f"payload-{compiled_qualname}",
    )


def _dispatch_plan() -> BindingDispatchPlan:
    return BindingDispatchPlan(
        binding=_binding(),
        fallback_symbol=_symbol("run"),
        variants=(
            _variant("generic", "cython", "_run_cython"),
            _variant("safe-int64", "cython", "_run_i64"),
            _variant("safe-int32", "cython", "_run_i32_cython"),
            _variant("generic", "mypyc", "_run_mypyc"),
            _variant("safe-int32", "cython", "_run_i32_second"),
        ),
        dispatch_fingerprint="dispatch-v1",
    )


def test_native_variant_models_are_frozen_slots_dataclasses() -> None:
    plan = NativeVariantPlan(
        owner=_symbol("run"),
        source_fingerprint="source-v1",
        planner_version="native-v1",
        dispatch_plans=(_dispatch_plan(),),
    )

    assert hasattr(plan, "__slots__")
    assert hasattr(plan.dispatch_plans[0], "__slots__")
    assert hasattr(plan.dispatch_plans[0].variants[0], "__slots__")
    assert plan.stable_id.startswith("native-plan-")
    with pytest.raises(FrozenInstanceError):
        plan.__setattr__("planner_version", "native-v2")


def test_dispatch_order_is_canonical_and_ends_with_python_fallback() -> None:
    dispatch = _dispatch_plan()

    assert tuple(variant.kind for variant in dispatch.variants) == (
        "safe-int32",
        "safe-int32",
        "safe-int64",
        "generic",
        "generic",
    )
    assert tuple(variant.backend for variant in dispatch.variants) == (
        "cython",
        "cython",
        "cython",
        "mypyc",
        "cython",
    )
    assert dispatch.ordered_targets[-1] == ("python-fallback", "app.worker::run")


def test_native_plan_id_is_content_derived_and_canonicalizes_dispatch_order() -> None:
    dispatch = _dispatch_plan()
    plan = NativeVariantPlan(
        owner=_symbol("run"),
        source_fingerprint="source-v1",
        planner_version="native-v1",
        dispatch_plans=(dispatch,),
    )
    reordered_plan = NativeVariantPlan(
        owner=_symbol("run"),
        source_fingerprint="source-v1",
        planner_version="native-v1",
        dispatch_plans=(replace(dispatch, variants=tuple(reversed(dispatch.variants))),),
    )
    changed_plan = replace(plan, source_fingerprint="source-v2")

    assert stable_native_variant_plan_id(plan) == stable_native_variant_plan_id(reordered_plan)
    assert stable_native_variant_plan_id(plan) != stable_native_variant_plan_id(changed_plan)
    assert plan.canonical_serialization == reordered_plan.canonical_serialization


@pytest.mark.parametrize(
    ("kind", "payload"),
    [
        (
            "exact-type",
            IntegerDomainGuardPayload(subject="value", minimum=0, maximum=10, bit_width=32),
        ),
        (
            "integer-domain",
            ExactTypeGuardPayload(
                subject="value",
                type_module="builtins",
                type_qualname="int",
            ),
        ),
    ],
)
def test_guard_expression_rejects_mismatched_payloads(kind: str, payload: object) -> None:
    with pytest.raises(TypeError, match="guard requires"):
        GuardExpression(
            kind=cast(GuardKind, kind),
            payload=cast(GuardPayload, payload),
            message="malformed guard",
        )


@pytest.mark.parametrize(
    "payload_factory",
    [
        lambda: IntegerDomainGuardPayload(subject="n", minimum=5, maximum=4, bit_width=32),
        lambda: IntegerDomainGuardPayload(
            subject="n",
            minimum=0,
            maximum=2**40,
            bit_width=32,
        ),
        lambda: DirectFieldGuardPayload(
            owner_subject="worker",
            owner_type_module="app.worker",
            owner_type_qualname="Worker",
            field_name="state.value",
        ),
        lambda: BufferLayoutGuardPayload(
            subject="values",
            format="i",
            itemsize=0,
            ndim=1,
            c_contiguous=True,
        ),
        lambda: BufferLayoutGuardPayload(
            subject="values",
            format="i",
            itemsize=4,
            ndim=0,
            c_contiguous=True,
        ),
    ],
)
def test_guard_payloads_reject_malformed_combinations(
    payload_factory: Callable[[], object],
) -> None:
    with pytest.raises(ValueError, match=r".+"):
        payload_factory()


def test_safe_integer_variants_require_matching_integer_domain_guard() -> None:
    with pytest.raises(ValueError, match="32-bit integer guard"):
        CompiledBindingVariant(
            backend="cython",
            kind="safe-int32",
            binding=_binding(),
            compiled_symbol=_symbol("_run_i32"),
            artifact_fingerprint="artifact",
            guards=(_int_guard(64),),
        )

    with pytest.raises(ValueError, match="Cython backend"):
        CompiledBindingVariant(
            backend="mypyc",
            kind="safe-int32",
            binding=_binding(),
            compiled_symbol=_symbol("_run_i32"),
            artifact_fingerprint="artifact",
            guards=(_int_guard(32),),
        )


def test_dispatch_plan_rejects_wrong_binding_and_duplicate_variants() -> None:
    wrong_binding = replace(
        _variant("generic", "mypyc", "_run_mypyc"),
        binding=_binding("other"),
    )
    with pytest.raises(ValueError, match="target the dispatch binding"):
        BindingDispatchPlan(
            binding=_binding(),
            fallback_symbol=_symbol("run"),
            variants=(wrong_binding,),
            dispatch_fingerprint="dispatch-v1",
        )

    int_owner = _binding(
        "Pairer.pair",
        owner_class="Pairer",
        target_owner_class="IntPairer",
    )
    other_owner = replace(int_owner, target_owner_class="PayloadPairer")
    owner_variant = replace(
        _variant("generic", "mypyc", "_pair"),
        binding=other_owner,
    )
    with pytest.raises(ValueError, match="target the dispatch binding"):
        BindingDispatchPlan(
            binding=int_owner,
            fallback_symbol=int_owner.source,
            variants=(owner_variant,),
            dispatch_fingerprint="dispatch-owner-v1",
        )

    duplicate = _variant("generic", "mypyc", "_run_mypyc")
    with pytest.raises(ValueError, match="unique"):
        BindingDispatchPlan(
            binding=_binding(),
            fallback_symbol=_symbol("run"),
            variants=(duplicate, duplicate),
            dispatch_fingerprint="dispatch-v1",
        )
