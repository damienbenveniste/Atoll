"""Tests for proof-authorized fixed-width Cython source generation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from textwrap import dedent

import pytest

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.clustering import enrich_island_analysis
from atoll.generation.scalar_kernel import (
    SCALAR_KERNEL_GENERATOR_VERSION,
    ScalarKernelGenerationRequest,
    generate_scalar_kernel,
)
from atoll.models import ModuleId, ModuleScan
from atoll.native_optimization.intervals import NativeInteger
from atoll.native_optimization.scalar_analysis import analyze_scalar_scan


def _scan(tmp_path: Path, source: str) -> ModuleScan:
    path = tmp_path / "scalar_generation.py"
    path.write_text(dedent(source).strip() + "\n", encoding="utf-8")
    return enrich_island_analysis(scan_module(ModuleId("scalar_generation", path)))


def test_scalar_kernel_generates_fixed_width_cython_with_proof_metadata(tmp_path: Path) -> None:
    """A safe polynomial loop becomes a typed Cython callable with no Python power."""
    scan = _scan(
        tmp_path,
        """
        OFFSET = 3

        def polynomial(limit: int, *, bias: int = 1) -> int:
            \"\"\"Return a bounded polynomial reduction.\"\"\"
            total = 0
            for value in range(limit):
                total += value ** 2 + OFFSET
            return total + bias
        """,
    )
    analysis = analyze_scalar_scan(scan)
    plan = next(item for item in analysis.plans if item.member.qualname == "polynomial")
    region = next(item for item in scan.typed_regions if item.id == plan.region_id)
    proof = plan.width_proofs[0]
    output = tmp_path / "generated.pyx"

    generated = generate_scalar_kernel(
        ScalarKernelGenerationRequest(
            scan=scan,
            region=region,
            plan=plan,
            width_proof=proof,
            logical_module="_atoll_scalar_test",
            output_path=output,
        )
    )

    assert generated.c_type == "int32_t"
    assert generated.generation.source_path == output
    assert generated.generation.bindings[0].source == plan.member
    assert generated.generation.bindings[0].compiled_name == generated.compiled_name
    assert (
        f"# atoll scalar proof {SCALAR_KERNEL_GENERATOR_VERSION}"
        in generated.generation.source_text
    )
    assert "cdef int32_t OFFSET = 3" in generated.generation.source_text
    assert "cdef int32_t total, value" in generated.generation.source_text
    assert (
        f"cdef int32_t {generated.compiled_name}_native(int32_t limit, int32_t bias):"
        in generated.generation.source_text
    )
    assert f"def {generated.compiled_name}(limit, *, bias):" in generated.generation.source_text
    assert "value * value" in generated.generation.source_text
    assert "**" not in generated.generation.source_text
    assert output.read_text(encoding="utf-8") == generated.generation.source_text


def test_scalar_kernel_rejects_stale_plan_and_non_pyx_destination(tmp_path: Path) -> None:
    """Generation stops before writing when staged identity or destination drifts."""
    scan = _scan(
        tmp_path,
        """
        def increment(value: int) -> int:
            return value + 1
        """,
    )
    plan = analyze_scalar_scan(scan).plans[0]
    region = next(item for item in scan.typed_regions if item.id == plan.region_id)
    request = ScalarKernelGenerationRequest(
        scan=scan,
        region=region,
        plan=plan,
        width_proof=plan.width_proofs[0],
        logical_module="_atoll_scalar_test",
        output_path=tmp_path / "generated.py",
    )

    with pytest.raises(ValueError, match=r"\.pyx"):
        generate_scalar_kernel(request)

    pyx_request = replace(request, output_path=tmp_path / "generated.pyx")
    with pytest.raises(ValueError, match="source differs"):
        generate_scalar_kernel(
            replace(
                pyx_request,
                plan=replace(plan, source_hash="stale"),
            )
        )

    with pytest.raises(ValueError, match="absent from staged region"):
        generate_scalar_kernel(
            replace(
                pyx_request,
                plan=replace(plan, member=replace(plan.member, qualname="missing")),
            )
        )

    unsigned = replace(
        plan.width_proofs[0],
        native=NativeInteger(width=32, signed=False),
    )
    with pytest.raises(ValueError, match="unsigned scalar lowering"):
        generate_scalar_kernel(replace(pyx_request, width_proof=unsigned))


def test_scalar_kernel_expands_zero_power_without_native_pow(tmp_path: Path) -> None:
    """A literal zero exponent lowers to the exact integer identity constant."""
    scan = _scan(
        tmp_path,
        """
        def identity_power(value: int) -> int:
            return value ** 0
        """,
    )
    plan = analyze_scalar_scan(scan).plans[0]
    region = next(item for item in scan.typed_regions if item.id == plan.region_id)

    generated = generate_scalar_kernel(
        ScalarKernelGenerationRequest(
            scan=scan,
            region=region,
            plan=plan,
            width_proof=plan.width_proofs[0],
            logical_module="_atoll_scalar_zero_power",
            output_path=tmp_path / "zero_power.pyx",
        )
    )

    assert "return 1" in generated.generation.source_text
    assert "**" not in generated.generation.source_text
