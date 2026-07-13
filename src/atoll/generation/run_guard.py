"""Generate one transactional source-fused Cython helper unit.

The generated unit preserves the source optimizer's eligibility body so Cython
can remove its Python frame and bytecode dispatch overhead. The public cached
guard and optional indexed-completion helpers are bound back together;
eligibility remains private and is guarded by the original source callable and
code identities.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

from atoll.generation.typed_region import (
    TypedRegionGeneration,
    TypedRegionGenerationOptions,
    generate_typed_method_region,
)
from atoll.models import ModuleScan, SymbolId, TypedRegion
from atoll.native_optimization.run_guard import (
    EXPECTED_COMPLETION_OWNER_NAME,
    EXPECTED_COMPLETION_PREDICATE_CODE_NAME,
    EXPECTED_COMPLETION_PREDICATE_NAME,
    EXPECTED_ELIGIBILITY_CODE_NAME,
    EXPECTED_ELIGIBILITY_NAME,
    RunGuardNativePlan,
)

RUN_GUARD_GENERATOR_VERSION = "run-guard-generator-v3"


@dataclass(frozen=True, slots=True)
class RunGuardGenerationRequest:
    """Inputs for one staged source-fused run-guard compilation unit.

    Attributes:
        scan: Fresh staged scan containing the Python fallback binding.
        region: Revalidated source-fused helper region.
        plan: Structured run-guard names and source-plan identity.
        logical_module: Private extension module name selected by packaging.
        output_path: Disposable generated ``.py`` path consumed by Cython.
    """

    scan: ModuleScan
    region: TypedRegion
    plan: RunGuardNativePlan
    logical_module: str
    output_path: Path


def generate_run_guard(request: RunGuardGenerationRequest) -> TypedRegionGeneration:
    """Write and describe a narrow source-fused Cython unit.

    Args:
        request: Revalidated staged scan, region, plan, and output location.

    Returns:
        TypedRegionGeneration: Generated source, binding metadata, and digest.

    Raises:
        ValueError: If the region does not own exactly the planned members and
            public helper bindings, or the logical module name is empty.
    """
    if not request.logical_module.strip():
        raise ValueError("run-guard logical module must be non-empty")
    selected_members = _selected_members(request.plan)
    member_ids = tuple(member.id for member in request.region.members)
    if member_ids != selected_members:
        raise ValueError("run-guard region must contain exactly its planned helper members")
    bound_members = frozenset(selected_members) - {request.plan.eligibility_helper}
    helper_bindings = tuple(
        binding for binding in request.region.bindings if binding.source in bound_members
    )
    if frozenset(binding.source for binding in helper_bindings) != bound_members:
        raise ValueError("run-guard generation requires every public helper binding")
    generated = generate_typed_method_region(
        request.scan,
        request.region,
        selected_members,
        output_path=request.output_path,
        options=TypedRegionGenerationOptions(backend="cython"),
    )
    source = _insert_source_identity_capture(generated.source_text, request)
    request.output_path.parent.mkdir(parents=True, exist_ok=True)
    request.output_path.write_text(source, encoding="utf-8")
    return replace(
        generated,
        logical_module=request.logical_module,
        source_text=source,
        source_hash=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        bindings=helper_bindings,
        backend="cython",
    )


def _selected_members(plan: RunGuardNativePlan) -> tuple[SymbolId, ...]:
    """Return deterministic private and public members for one native unit.

    Args:
        plan: Structured source-fused optimization plan.

    Returns:
        tuple[SymbolId, ...]: Eligibility, guard, then optional completion helpers.
    """
    members = [plan.eligibility_helper, plan.helper]
    if plan.completion_index is not None:
        members.extend((plan.completion_index.snapshot, plan.completion_index.query))
    return tuple(members)


def _insert_source_identity_capture(
    source: str,
    request: RunGuardGenerationRequest,
) -> str:
    """Capture the source eligibility callable before runtime shim installation.

    Args:
        source: Typed-region source containing both compiled helper bodies.
        request: Source module and structured eligibility identity.

    Returns:
        str: Source with one transformed-module import and immutable identity captures.

    Raises:
        ValueError: If the generated future import cannot be found.
    """
    source_import = f"import {request.scan.module.name} as _atoll_source"
    if source_import not in source:
        future = "from __future__ import annotations\n"
        if future not in source:
            raise ValueError("run-guard generation lost its future import")
        source = source.replace(future, f"{future}\n{source_import}\n", 1)
    capture = (
        f"{EXPECTED_ELIGIBILITY_NAME} = "
        f"_atoll_source.{request.plan.eligibility_helper.qualname}\n"
        f"{EXPECTED_ELIGIBILITY_CODE_NAME} = "
        f"getattr({EXPECTED_ELIGIBILITY_NAME}, '__code__', None)\n"
    )
    if request.plan.completion_index is not None:
        owner_class = request.plan.owner.qualname.split(".", maxsplit=1)[0]
        predicate = request.plan.completion_index.fallback_predicate_method
        capture += (
            f"{EXPECTED_COMPLETION_OWNER_NAME} = _atoll_source.{owner_class}\n"
            f"{EXPECTED_COMPLETION_PREDICATE_NAME} = "
            f"vars({EXPECTED_COMPLETION_OWNER_NAME})[{predicate!r}]\n"
            f"{EXPECTED_COMPLETION_PREDICATE_CODE_NAME} = "
            f"getattr({EXPECTED_COMPLETION_PREDICATE_NAME}, '__code__', None)\n"
        )
    captured = source.replace(source_import, f"{source_import}\n\n{capture}", 1)
    return f"# atoll run-guard generator {RUN_GUARD_GENERATOR_VERSION}\n{captured}"
