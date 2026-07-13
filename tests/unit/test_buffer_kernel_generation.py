"""Tests for proof-authorized zero-copy Cython buffer generation."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.clustering import enrich_island_analysis
from atoll.generation.buffer_kernel import (
    BUFFER_KERNEL_GENERATOR_VERSION,
    BufferKernelGenerationRequest,
    generate_buffer_kernel,
)
from atoll.models import ModuleId, ModuleScan, TypedRegion
from atoll.native_optimization.buffer_analysis import (
    BufferKernelPlan,
    BufferReductionKind,
    analyze_buffer_scan,
)


def _evidence(
    tmp_path: Path, source: str, qualname: str
) -> tuple[ModuleScan, BufferKernelPlan, TypedRegion]:
    path = tmp_path / "buffer_fixture.py"
    path.write_text(source, encoding="utf-8")
    scan = enrich_island_analysis(scan_module(ModuleId("buffer_fixture", path)))
    plan = next(
        item for item in analyze_buffer_scan(scan).plans if item.member.qualname == qualname
    )
    region = next(
        item
        for item in scan.typed_regions
        if any(member.id == plan.member for member in item.members)
    )
    return scan, plan, region


def test_generate_buffer_kernel_emits_typed_zero_copy_sum(tmp_path: Path) -> None:
    """A proven byte sum uses a const contiguous view and native loop values."""
    scan, plan, region = _evidence(
        tmp_path,
        """def checksum(data: bytes) -> int:
    total = 0
    for value in data:
        total += value
    return total
""",
        "checksum",
    )
    output = tmp_path / "_buffer_sum.pyx"

    generated = generate_buffer_kernel(
        BufferKernelGenerationRequest(
            scan=scan,
            region=region,
            plan=plan,
            logical_module="_buffer_sum",
            output_path=output,
        )
    )

    assert (
        f"# atoll buffer proof {BUFFER_KERNEL_GENERATOR_VERSION}"
        in generated.generation.source_text
    )
    assert "const _atoll_uint8_t[::1] _atoll_view_data = data" in generated.generation.source_text
    assert "cdef _atoll_uint64_t total" in generated.generation.source_text
    assert "cdef _atoll_uint8_t value" in generated.generation.source_text
    assert "for value in _atoll_view_data:" in generated.generation.source_text
    assert "boundscheck=False" in generated.generation.source_text
    assert ".tobytes(" not in generated.generation.source_text
    assert generated.generation.bindings[0].source == plan.member


def test_generate_buffer_kernel_preserves_staticmethod_binding_and_index_loop(
    tmp_path: Path,
) -> None:
    """Indexed static reductions retain descriptor metadata and proven bounds."""
    scan, plan, region = _evidence(
        tmp_path,
        """class Checksums:
    @staticmethod
    def xor(data: bytearray) -> int:
        total = 0
        for index in range(len(data)):
            total ^= data[index]
        return total
""",
        "Checksums.xor",
    )

    generated = generate_buffer_kernel(
        BufferKernelGenerationRequest(
            scan=scan,
            region=region,
            plan=plan,
            logical_module="_buffer_xor",
            output_path=tmp_path / "_buffer_xor.pyx",
        )
    )

    assert "cdef _atoll_size_t index" in generated.generation.source_text
    assert "range(len(_atoll_view_data))" in generated.generation.source_text
    binding = generated.generation.bindings[0]
    assert binding.kind == "staticmethod"
    assert binding.owner_class == "Checksums"


def test_generate_buffer_kernel_rejects_stale_or_unproven_inputs(tmp_path: Path) -> None:
    """Generation fails before writing when source identity or proof shape changed."""
    scan, plan, region = _evidence(
        tmp_path,
        """def checksum(data: memoryview) -> int:
    total = 0
    for value in data:
        total += value
    return total
""",
        "checksum",
    )
    with pytest.raises(ValueError, match="source differs"):
        generate_buffer_kernel(
            BufferKernelGenerationRequest(
                scan=scan,
                region=region,
                plan=replace(plan, source_hash="stale"),
                logical_module="_stale",
                output_path=tmp_path / "_stale.pyx",
            )
        )


def test_generate_buffer_kernel_rejects_malformed_proof_evidence(tmp_path: Path) -> None:
    """Proof consumers reject absent members, layouts, reductions, and return evidence."""
    scan, plan, region = _evidence(
        tmp_path,
        """def checksum(data: bytes) -> int:
    total = 0
    for value in data:
        total += value
    return total
""",
        "checksum",
    )

    def generate(
        selected_plan: BufferKernelPlan,
        selected_region: TypedRegion = region,
    ) -> None:
        generate_buffer_kernel(
            BufferKernelGenerationRequest(
                scan=scan,
                region=selected_region,
                plan=selected_plan,
                logical_module="_malformed",
                output_path=tmp_path / "_malformed.pyx",
            )
        )

    with pytest.raises(ValueError, match="member is absent"):
        generate(plan, replace(region, members=()))
    with pytest.raises(ValueError, match="exactly one buffer"):
        generate(replace(plan, buffers=()))
    bad_layout = replace(
        plan.buffers[0],
        layout=replace(plan.buffers[0].layout, format="I", itemsize=4),
    )
    with pytest.raises(ValueError, match="unsigned-byte layout"):
        generate(replace(plan, buffers=(bad_layout,)))
    with pytest.raises(ValueError, match="unsupported buffer reduction"):
        generate(replace(plan, reduction=cast(BufferReductionKind, "product")))
    missing_accumulator = replace(plan.returns[0], accumulator=None)
    with pytest.raises(ValueError, match="no direct accumulator"):
        generate(replace(plan, returns=(missing_accumulator,)))


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            """for value in data:
    pass
return value
""",
            "requires one accumulator initializer",
        ),
        (
            """total = 1
for value in data:
    total += value
return total
""",
            "initialize to exact zero",
        ),
        (
            """total = 0
for value in data:
    total += value
return total + 1
""",
            "return its accumulator directly",
        ),
        (
            """total = 0
for value in data:
    total += value
other = total
return other
""",
            "return its accumulator directly",
        ),
    ],
)
def test_generate_buffer_kernel_revalidates_source_shape(
    tmp_path: Path,
    body: str,
    message: str,
) -> None:
    """A source-hash match alone cannot bypass reduction-shape validation."""
    scan, plan, region = _evidence(
        tmp_path,
        """def checksum(data: bytes) -> int:
    total = 0
    for value in data:
        total += value
    return total
""",
        "checksum",
    )
    source = "def checksum(data: bytes) -> int:\n" + "\n".join(
        f"    {line}" if line else "" for line in body.splitlines()
    )
    member = next(item for item in region.members if item.id == plan.member)
    changed_member = replace(member, source_text=source)
    changed_region = replace(
        region,
        members=tuple(
            changed_member if item.id == plan.member else item for item in region.members
        ),
    )
    changed_plan = replace(
        plan,
        source_hash=hashlib.sha256(source.encode("utf-8")).hexdigest(),
    )

    with pytest.raises(ValueError, match=message):
        generate_buffer_kernel(
            BufferKernelGenerationRequest(
                scan=scan,
                region=changed_region,
                plan=changed_plan,
                logical_module="_shape",
                output_path=tmp_path / "_shape.pyx",
            )
        )


def test_generate_buffer_kernel_renders_annotated_accumulator(tmp_path: Path) -> None:
    """An analysis-proven local annotation becomes one native assignment."""
    scan, plan, region = _evidence(
        tmp_path,
        """def checksum(data: bytes) -> int:
    total: int = 0
    for value in data:
        total ^= value
    return total
""",
        "checksum",
    )
    generated = generate_buffer_kernel(
        BufferKernelGenerationRequest(
            scan=scan,
            region=region,
            plan=plan,
            logical_module="_annotated",
            output_path=tmp_path / "_annotated.pyx",
        )
    )

    assert "total = 0" in generated.generation.source_text
    assert "total: int" not in generated.generation.source_text


def test_generate_buffer_kernel_uses_plan_accumulator_and_fresh_view_name(
    tmp_path: Path,
) -> None:
    """Cached lengths and source-name collisions retain compilable private names."""
    scan, plan, region = _evidence(
        tmp_path,
        """def checksum(data: bytes) -> int:
    length = len(data)
    uint8_t = 0
    for _atoll_view_data in range(length):
        uint8_t += data[_atoll_view_data]
    return uint8_t
""",
        "checksum",
    )

    generated = generate_buffer_kernel(
        BufferKernelGenerationRequest(
            scan=scan,
            region=region,
            plan=plan,
            logical_module="_fresh_name",
            output_path=tmp_path / "_fresh_name.pyx",
        )
    )

    source = generated.generation.source_text
    assert "const _atoll_uint8_t[::1] _atoll_view_data_1 = data" in source
    assert "cimport uint8_t as _atoll_uint8_t" in source
    assert "cimport uint64_t as _atoll_uint64_t" in source
    assert "cdef _atoll_uint64_t uint8_t" in source
    assert "length = len(_atoll_view_data_1)" in source
    assert "uint8_t += _atoll_view_data_1[_atoll_view_data]" in source
    with pytest.raises(ValueError, match=r"require a \.pyx"):
        generate_buffer_kernel(
            BufferKernelGenerationRequest(
                scan=scan,
                region=region,
                plan=plan,
                logical_module="_wrong_suffix",
                output_path=tmp_path / "_wrong_suffix.py",
            )
        )
