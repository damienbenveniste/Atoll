"""Tests for conservative zero-copy buffer candidate analysis."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from textwrap import dedent, indent

import pytest

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.clustering import enrich_island_analysis
from atoll.models import ModuleId, ModuleScan, RegionMember, TypedRegion
from atoll.native_optimization.buffer_analysis import (
    BufferKernelPlan,
    BufferRejection,
    analyze_buffer_member,
    analyze_buffer_scan,
)
from atoll.native_optimization.models import (
    BufferLayoutGuardPayload,
    ExactTypeGuardPayload,
)

_CHECKSUM_START_LINE = 3
_CHECKSUM_END_LINE = 7
_FIRST_UPDATE_LINE = 5
_MOVED_DECLARATION_LINE = 3
_DIRECT_ITERATION_LINE = 5
_FIRST_MOVED_ACCESS_LINE = 3
_SECOND_MOVED_ACCESS_LINE = 5
_BYTE_ADDITIVE_MAX_LENGTH = (2**64 - 1) // 255
_COUNT_MAX_LENGTH = 2**64 - 1


def _scan(tmp_path: Path, source: str) -> ModuleScan:
    path = tmp_path / "buffer_subject.py"
    path.write_text(dedent(source).strip() + "\n", encoding="utf-8")
    return enrich_island_analysis(scan_module(ModuleId(name="buffer_subject", path=path)))


def _member(scan: ModuleScan, qualname: str) -> tuple[TypedRegion, RegionMember]:
    return next(
        (region, member)
        for region in scan.typed_regions
        for member in region.members
        if member.id.qualname == qualname
    )


def _plan(scan: ModuleScan, qualname: str) -> BufferKernelPlan:
    return next(
        plan for plan in analyze_buffer_scan(scan).plans if plan.member.qualname == qualname
    )


def _direct_source_outcome(tmp_path: Path, body: str) -> BufferKernelPlan | BufferRejection:
    scan = _scan(
        tmp_path,
        """
        def subject(data: bytes) -> int:
            return 0
        """,
    )
    region, member = _member(scan, "subject")
    source = "def subject(data: bytes) -> int:\n" + indent(dedent(body).strip(), "    ") + "\n"
    return analyze_buffer_member(region, replace(member, source_text=source))


def test_buffer_analysis_proves_direct_iteration_and_layout_guards(tmp_path: Path) -> None:
    """A direct byte iteration checksum receives stable exact-type and layout guards."""
    scan = _scan(
        tmp_path,
        """
        import array

        def checksum(data: bytes) -> int:
            total = 0
            for value in data:
                total += value
            return total
        """,
    )

    plan = _plan(scan, "checksum")
    repeated = _plan(scan, "checksum")

    assert plan.id == repeated.id
    assert plan.source_hash == repeated.source_hash
    assert plan.declaration_start_lineno == _CHECKSUM_START_LINE
    assert plan.end_lineno == _CHECKSUM_END_LINE
    assert [buffer.name for buffer in plan.buffers] == ["data"]
    assert [buffer.annotation for buffer in plan.buffers] == ["bytes"]
    assert plan.reduction == "add"
    assert [access.kind for access in plan.accesses] == ["iteration", "iteration"]
    assert plan.accesses[0].buffer == "data"
    assert plan.accesses[0].span.lineno == _DIRECT_ITERATION_LINE
    assert [item.kind for item in plan.accumulators] == ["initialize", "update"]
    assert plan.accumulators[1].span.lineno == _FIRST_UPDATE_LINE + 1
    assert plan.returns[0].accumulator == "total"
    assert plan.buffers[0].max_length == _BYTE_ADDITIVE_MAX_LENGTH

    exact_guards = [
        guard.payload for guard in plan.guards if isinstance(guard.payload, ExactTypeGuardPayload)
    ]
    layout_guards = [
        guard.payload
        for guard in plan.guards
        if isinstance(guard.payload, BufferLayoutGuardPayload)
    ]
    assert [(guard.subject, guard.type_module, guard.type_qualname) for guard in exact_guards] == [
        ("data", "builtins", "bytes")
    ]
    assert [
        (
            guard.subject,
            guard.format,
            guard.itemsize,
            guard.ndim,
            guard.c_contiguous,
            guard.f_contiguous,
            guard.readonly,
            guard.minimum_length,
            guard.maximum_length,
        )
        for guard in layout_guards
    ] == [("data", "B", 1, 1, True, True, True, 0, _BYTE_ADDITIVE_MAX_LENGTH)]


def test_buffer_analysis_accepts_indexed_bytearray_and_static_array_methods(
    tmp_path: Path,
) -> None:
    """Indexed loops are accepted only through a guarded range(len(buffer)) shape."""
    scan = _scan(
        tmp_path,
        """
        import array

        def indexed(data: bytearray) -> int:
            total = 0
            for index in range(len(data)):
                total += data[index]
            return total

        class Kernels:
            @staticmethod
            def array_checksum(values: array.array) -> int:
                checksum: int = 0
                length = len(values)
                for position in range(length):
                    checksum ^= values[position]
                return checksum
        """,
    )

    indexed = _plan(scan, "indexed")
    static = _plan(scan, "Kernels.array_checksum")

    assert [access.kind for access in indexed.accesses] == ["indexed"]
    assert indexed.accesses[0].index_name == "index"
    assert isinstance(indexed.guards[0].payload, ExactTypeGuardPayload)
    assert indexed.guards[0].payload.type_qualname == "bytearray"
    assert isinstance(indexed.guards[1].payload, BufferLayoutGuardPayload)
    assert indexed.guards[1].payload.readonly is False

    assert static.member.qualname == "Kernels.array_checksum"
    assert [access.kind for access in static.accesses] == ["len", "indexed"]
    assert static.accesses[1].index_name == "position"
    assert isinstance(static.guards[0].payload, ExactTypeGuardPayload)
    assert static.guards[0].payload.type_module == "array"
    assert static.guards[0].payload.type_qualname == "array"
    assert isinstance(static.guards[1].payload, BufferLayoutGuardPayload)
    assert static.guards[1].payload.format == "B"
    assert static.guards[1].payload.readonly is False
    assert static.guards[1].payload.minimum_length is None
    assert static.guards[1].payload.maximum_length is None
    assert static.reduction == "xor"
    assert static.buffers[0].max_length is None


def test_buffer_analysis_accepts_memoryview_branch_checks(tmp_path: Path) -> None:
    """Scalar conditions may read buffer elements without calls or materialization."""
    scan = _scan(
        tmp_path,
        """
        def count_nonzero(view: memoryview) -> int:
            total = 0
            for index in range(len(view)):
                if view[index] != 0:
                    total += 1
            return total
        """,
    )

    plan = _plan(scan, "count_nonzero")

    assert [access.kind for access in plan.accesses] == ["indexed"]
    assert [item.name for item in plan.accumulators] == ["total", "total"]
    assert plan.reduction == "count"
    assert plan.buffers[0].max_length == _COUNT_MAX_LENGTH
    assert plan.buffers[0].layout.readonly is None
    assert plan.buffers[0].layout.minimum_length == 0
    assert plan.buffers[0].layout.maximum_length == _COUNT_MAX_LENGTH
    assert plan.returns[0].expression == "total"


@pytest.mark.parametrize(
    ("qualname", "source", "expected"),
    [
        (
            "nonzero_seed",
            """
            def nonzero_seed(data: bytes) -> int:
                total = 1
                for value in data:
                    total += value
                return total
            """,
            "unsupported-expression",
        ),
        (
            "bare_count",
            """
            def bare_count(data: bytes) -> int:
                total = 0
                for value in data:
                    total += 1
                return total
            """,
            "unsupported-expression",
        ),
        (
            "arbitrary_binop",
            """
            def arbitrary_binop(data: bytes) -> int:
                total = 0
                for value in data:
                    total = total + value
                return total
            """,
            "unsupported-expression",
        ),
        (
            "mutate",
            """
            def mutate(data: bytearray) -> int:
                data[0] = 1
                return 0
            """,
            "external-mutation",
        ),
        (
            "opaque",
            """
            def opaque(data: bytes) -> int:
                return sum(data)
            """,
            "opaque-call",
        ),
        (
            "materialize",
            """
            def materialize(data: bytes) -> int:
                values = list(data)
                return len(data)
            """,
            "opaque-call",
        ),
        (
            "bad_index",
            """
            def bad_index(data: bytes) -> int:
                total = 0
                total += data[0]
                return total
            """,
            "unsupported-indexing",
        ),
        (
            "index_arithmetic",
            """
            def index_arithmetic(data: bytes) -> int:
                total = 0
                for index in range(len(data)):
                    total += data[index + 1]
                return total
            """,
            "unsupported-indexing",
        ),
        (
            "return_expression",
            """
            def return_expression(data: bytes) -> int:
                total = 0
                for value in data:
                    total += value
                return total + 1
            """,
            "unsupported-expression",
        ),
        (
            "mixed_reduction",
            """
            def mixed_reduction(data: bytes) -> int:
                total = 0
                for value in data:
                    total += value
                    total ^= value
                return total
            """,
            "unsupported-expression",
        ),
        (
            "two_buffers",
            """
            def two_buffers(left: bytes, right: bytes) -> int:
                total = 0
                for value in left:
                    total += value
                return total
            """,
            "unsupported-signature",
        ),
        (
            "scalar_parameter",
            """
            def scalar_parameter(data: bytes, scale: int) -> int:
                total = 0
                for value in data:
                    total += value
                return total
            """,
            "unsupported-signature",
        ),
        (
            "generic",
            """
            from collections.abc import Buffer
            def generic(data: Buffer) -> int:
                return len(data)
            """,
            "unsupported-annotation",
        ),
        (
            "no_parameters",
            """
            def no_parameters() -> int:
                return 0
            """,
            "unsupported-signature",
        ),
        (
            "defaulted",
            """
            def defaulted(data: bytes = b"") -> int:
                return 0
            """,
            "unsupported-signature",
        ),
        (
            "nested_scope",
            """
            def nested_scope(data: bytes) -> int:
                total = 0
                values = [value for value in data]
                for value in data:
                    total += value
                return total
            """,
            "unsupported-expression",
        ),
        (
            "declared_global",
            """
            STATE = 0
            def declared_global(data: bytes) -> int:
                global STATE
                total = 0
                for value in data:
                    total += value
                return total
            """,
            "unsupported-scope",
        ),
        (
            "unsupported_while",
            """
            def unsupported_while(data: bytes) -> int:
                total = 0
                while False:
                    pass
                return total
            """,
            "unsupported-statement",
        ),
        (
            "annotation_without_value",
            """
            def annotation_without_value(data: bytes) -> int:
                total: int
                for value in data:
                    total += value
                return total
            """,
            "external-mutation",
        ),
        (
            "wrong_local_annotation",
            """
            def wrong_local_annotation(data: bytes) -> int:
                total: bool = False
                for value in data:
                    total += value
                return total
            """,
            "unsupported-annotation",
        ),
        (
            "second_accumulator",
            """
            def second_accumulator(data: bytes) -> int:
                total = 0
                other = 0
                for value in data:
                    total += value
                return total
            """,
            "unsupported-statement",
        ),
        (
            "unsupported_operator",
            """
            def unsupported_operator(data: bytes) -> int:
                total = 0
                for value in data:
                    total *= value
                return total
            """,
            "unsupported-expression",
        ),
        (
            "count_else",
            """
            def count_else(data: bytes) -> int:
                total = 0
                for value in data:
                    if value != 0:
                        total += 1
                    else:
                        pass
                return total
            """,
            "unsupported-statement",
        ),
        (
            "bad_count_body",
            """
            def bad_count_body(data: bytes) -> int:
                total = 0
                for value in data:
                    if value != 0:
                        total ^= value
                return total
            """,
            "unsupported-statement",
        ),
        (
            "none_return",
            """
            def none_return(data: bytes) -> int:
                total = 0
                for value in data:
                    total += value
                return
            """,
            "unsupported-annotation",
        ),
        (
            "loop_else",
            """
            def loop_else(data: bytes) -> int:
                total = 0
                for value in data:
                    total += value
                else:
                    pass
                return total
            """,
            "unsupported-statement",
        ),
        (
            "tuple_target",
            """
            def tuple_target(data: bytes) -> int:
                total = 0
                for left, right in data:
                    total += left
                return total
            """,
            "external-mutation",
        ),
        (
            "bad_loop_source",
            """
            def bad_loop_source(data: bytes) -> int:
                total = 0
                for value in range(3):
                    total += value
                return total
            """,
            "unsupported-indexing",
        ),
        (
            "bad_count_condition",
            """
            def bad_count_condition(data: bytes) -> int:
                total = 0
                for value in data:
                    if value:
                        total += 1
                return total
            """,
            "unsupported-expression",
        ),
        (
            "two_element_condition",
            """
            def two_element_condition(data: bytes) -> int:
                total = 0
                for value in data:
                    if value == value:
                        total += 1
                return total
            """,
            "unsupported-expression",
        ),
        (
            "non_literal_condition",
            """
            def non_literal_condition(data: bytes) -> int:
                total = 0
                for value in data:
                    if value == total:
                        total += 1
                return total
            """,
            "unsupported-expression",
        ),
        (
            "range_arguments",
            """
            def range_arguments(data: bytes) -> int:
                total = 0
                for index in range(0, len(data)):
                    total += data[index]
                return total
            """,
            "unsupported-indexing",
        ),
        (
            "wrong_accumulator",
            """
            def wrong_accumulator(data: bytes) -> int:
                total = 0
                for value in data:
                    other += value
                return total
            """,
            "unsupported-expression",
        ),
        (
            "missing_return",
            """
            def missing_return(data: bytes) -> int:
                total = 0
                for value in data:
                    total += value
            """,
            "unsupported-statement",
        ),
        (
            "missing_update",
            """
            def missing_update(data: bytes) -> int:
                total = 0
                for value in data:
                    pass
                return total
            """,
            "unsupported-statement",
        ),
        (
            "repeated_update",
            """
            def repeated_update(data: bytes) -> int:
                total = 0
                for value in data:
                    total += value
                    total += value
                return total
            """,
            "unsupported-statement",
        ),
        (
            "repeated_loop",
            """
            def repeated_loop(data: bytes) -> int:
                total = 0
                for value in data:
                    total += value
                for value in data:
                    total += value
                return total
            """,
            "unsupported-statement",
        ),
        (
            "nested_loop",
            """
            def nested_loop(data: bytes) -> int:
                total = 0
                for value in data:
                    for other in data:
                        total += other
                return total
            """,
            "unsupported-statement",
        ),
        (
            "buffer_rebind",
            """
            def buffer_rebind(data: bytes) -> int:
                total = 0
                data = len(data)
                for value in data:
                    total += value
                return total
            """,
            "external-mutation",
        ),
        (
            "accumulator_rebind",
            """
            def accumulator_rebind(data: bytes) -> int:
                total = 0
                total = len(data)
                for value in data:
                    total += value
                return total
            """,
            "unsupported-statement",
        ),
        (
            "loop_initializer",
            """
            def loop_initializer(data: bytes) -> int:
                for value in data:
                    total = 0
                    total += value
                return total
            """,
            "unsupported-statement",
        ),
        (
            "buffer_loop_target",
            """
            def buffer_loop_target(data: bytes) -> int:
                total = 0
                for data in data:
                    total += data
                return total
            """,
            "external-mutation",
        ),
        (
            "accumulator_loop_target",
            """
            def accumulator_loop_target(data: bytes) -> int:
                total = 0
                for total in data:
                    total += total
                return total
            """,
            "external-mutation",
        ),
        (
            "loop_return",
            """
            def loop_return(data: bytes) -> int:
                total = 0
                for value in data:
                    total += value
                    return total
            """,
            "unsupported-statement",
        ),
    ],
)
def test_buffer_analysis_rejects_unsafe_buffer_shapes(
    tmp_path: Path,
    qualname: str,
    source: str,
    expected: str,
) -> None:
    """Mutation, opaque calls, materialization, and unsupported exporters fall back."""
    scan = _scan(tmp_path, source)
    result = analyze_buffer_scan(scan)
    rejection = next(item for item in result.rejections if item.member.qualname == qualname)

    assert rejection.code == expected


def test_buffer_analysis_rejects_unsafe_callable_shapes(tmp_path: Path) -> None:
    """Async code, instance methods, variadics, and shadowed builtins are rejected."""
    scan = _scan(
        tmp_path,
        """
        len = 3

        def shadowed(data: bytes) -> int:
            return len(data)

        def variadic(*data: bytes) -> int:
            return 0

        async def asynchronous(data: bytes) -> int:
            return 0

        class Worker:
            def instance(self, data: bytes) -> int:
                return len(data)
        """,
    )
    result = analyze_buffer_scan(scan)
    rejected = {rejection.member.qualname: rejection for rejection in result.rejections}

    assert rejected["shadowed"].code in {"opaque-call", "unsupported-scope"}
    assert rejected["variadic"].code == "unsupported-signature"
    assert rejected["asynchronous"].code == "unsupported-execution"
    assert rejected["Worker.instance"].code == "unsupported-binding"


def test_buffer_member_rejects_unread_returns_and_is_immutable(tmp_path: Path) -> None:
    """Plans require a buffer read, and returned plan dataclasses are frozen."""
    unread = _direct_source_outcome(
        tmp_path,
        """
        total = 0
        for value in data:
            pass
        return total
        """,
    )
    assert isinstance(unread, BufferRejection)
    assert unread.code == "unsupported-statement"

    accepted = _direct_source_outcome(
        tmp_path,
        """
        total = 0
        for value in data:
            total += value
        return total
        """,
    )
    assert isinstance(accepted, BufferKernelPlan)
    field_name = "id"
    with pytest.raises(FrozenInstanceError):
        setattr(accepted, field_name, "changed")


def test_buffer_analysis_plan_identity_ignores_absolute_source_location(tmp_path: Path) -> None:
    """Moving unchanged source changes report spans but not proof-plan identity."""
    first = _scan(
        tmp_path,
        """
        def checksum(data: bytes) -> int:
            total = 0
            for value in data:
                total += value
            return total
        """,
    )
    first_plan = _plan(first, "checksum")
    moved = _scan(
        tmp_path,
        """
        SENTINEL = 1

        def checksum(data: bytes) -> int:
            total = 0
            for value in data:
                total += value
            return total
        """,
    )
    moved_plan = _plan(moved, "checksum")

    assert first_plan.id == moved_plan.id
    assert first_plan.declaration_start_lineno == 1
    assert moved_plan.declaration_start_lineno == _MOVED_DECLARATION_LINE
    assert first_plan.accesses[0].span.lineno == _FIRST_MOVED_ACCESS_LINE
    assert moved_plan.accesses[0].span.lineno == _SECOND_MOVED_ACCESS_LINE
