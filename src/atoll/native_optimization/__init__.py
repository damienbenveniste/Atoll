"""Native optimization model contracts for guarded binding dispatch."""

from atoll.native_optimization.models import (
    BindingDispatchPlan,
    BufferLayoutGuardPayload,
    CallableCodeIdentityGuardPayload,
    CompiledBindingVariant,
    CompiledVariantKind,
    DirectFieldGuardPayload,
    DispatchTargetKind,
    ExactTypeGuardPayload,
    GuardExpression,
    GuardKind,
    GuardPayload,
    IntegerBitWidth,
    IntegerDomainGuardPayload,
    NativeVariantPlan,
    stable_native_variant_plan_id,
)

__all__ = (
    "BindingDispatchPlan",
    "BufferLayoutGuardPayload",
    "CallableCodeIdentityGuardPayload",
    "CompiledBindingVariant",
    "CompiledVariantKind",
    "DirectFieldGuardPayload",
    "DispatchTargetKind",
    "ExactTypeGuardPayload",
    "GuardExpression",
    "GuardKind",
    "GuardPayload",
    "IntegerBitWidth",
    "IntegerDomainGuardPayload",
    "NativeVariantPlan",
    "stable_native_variant_plan_id",
)
