"""Tests for conservative suspension-aware block planning."""

from __future__ import annotations

from textwrap import dedent

import pytest

from atoll.analysis.suspension_planner import plan_suspension_blocks
from atoll.models import RegionMember, SymbolId


def _member(source_text: str, qualname: str = "worker") -> RegionMember:
    return RegionMember(
        id=SymbolId(module="sample", qualname=qualname),
        kind="function",
        owner_class=None,
        binding_kind="module",
        execution_kind="coroutine" if source_text.lstrip().startswith("async ") else "sync",
        source_text=dedent(source_text),
        type_parameters=(),
        type_parameter_records=(),
        scope_type_parameters=(),
        scope_type_parameter_records=(),
        parameters=(),
        return_annotation=None,
    )


def test_blocks_split_at_await_and_keep_returns_in_shell() -> None:
    plan = plan_suspension_blocks(
        _member(
            """
            async def worker(values):
                scratch = 0
                for value in values:
                    scratch += value + 1
                sink(scratch)
                await checkpoint()
                current = seed + 1
                result = current * 2
                return result
            """
        )
    )

    assert [block.start_lineno for block in plan.blocks] == [2, 7]
    assert "await checkpoint()" not in "\n".join(block.source_text for block in plan.blocks)
    assert "return result" not in "\n".join(block.source_text for block in plan.blocks)
    assert plan.blocks[0].eligible is True
    assert plan.blocks[1].live_ins == ("seed",)
    assert plan.blocks[1].live_outs == ("result",)


def test_liveness_receiver_and_global_dependencies_are_reported() -> None:
    plan = plan_suspension_blocks(
        _member(
            """
            def worker(self, values):
                total = BASE + self.bias
                for value in values:
                    total += transform(value)
                self.record(total)
                return total
            """
        )
    )

    block = plan.blocks[0]
    assert block.live_ins == ("BASE", "self", "transform", "values")
    assert block.live_outs == ("total",)
    assert block.late_bound_globals == ("BASE", "transform")
    assert block.receiver_dependencies == ("self.bias", "self.record")


def test_exception_finally_and_async_contexts_stay_in_shell() -> None:
    plan = plan_suspension_blocks(
        _member(
            """
            async def worker(values, lock):
                before = len(values) + 1
                try:
                    risky()
                finally:
                    cleanup()
                async with lock:
                    await checkpoint()
                async for value in values:
                    sink(value)
                after = before + 2
                result = after * 3
                return result
            """
        )
    )

    assert [block.source_text for block in plan.blocks] == [
        "before = len(values) + 1",
        "after = before + 2\nresult = after * 3",
    ]
    assert plan.blocks[0].late_bound_globals == ("len",)
    assert all("cleanup" not in block.source_text for block in plan.blocks)
    assert all("async " not in block.source_text for block in plan.blocks)


def test_exception_group_handlers_stay_in_shell() -> None:
    plan = plan_suspension_blocks(
        _member(
            """
            async def worker(values):
                before = len(values) + 1
                try:
                    risky()
                except* ValueError:
                    recover()
                after = before + 2
                result = after * 3
                await checkpoint()
            """
        )
    )

    assert [block.source_text for block in plan.blocks] == [
        "before = len(values) + 1",
        "after = before + 2\nresult = after * 3",
    ]


@pytest.mark.parametrize(
    ("source_text", "expected_code"),
    [
        (
            """
            def worker():
                value = 1
                def inner():
                    return value
                return inner()
            """,
            "cell_variable",
        ),
        (
            """
            def worker():
                def inner():
                    return 1
                return inner()
            """,
            "nested_scope",
        ),
        (
            """
            def worker(values):
                yield from values
            """,
            "yield_from",
        ),
        (
            """
            def worker():
                global TOTAL
                TOTAL = 1
            """,
            "global_declaration",
        ),
        (
            """
            def worker():
                global TOTAL
                value = TOTAL + 1
                return value
            """,
            "global_declaration",
        ),
        (
            """
            def worker():
                try:
                    risky()
                except ValueError:
                    raise
            """,
            "bare_raise",
        ),
    ],
)
def test_member_level_rejections_are_reported(source_text: str, expected_code: str) -> None:
    plan = plan_suspension_blocks(_member(source_text))

    assert expected_code in {rejection.code for rejection in plan.rejections}
    assert plan.eligible_block_ids == ()


def test_unsafe_liveness_across_suspension_rejects_control_block() -> None:
    plan = plan_suspension_blocks(
        _member(
            """
            async def worker(flag, value):
                if flag:
                    score = value + 1
                else:
                    score = value - 1
                await checkpoint()
                result = score * 2
                return result
            """
        )
    )

    assert plan.blocks[0].source_text.startswith("if flag:")
    assert plan.blocks[0].live_outs == ("score",)
    assert "unsafe_liveness" in {rejection.code for rejection in plan.blocks[0].rejections}


def test_nested_safe_control_flow_can_be_eligible() -> None:
    plan = plan_suspension_blocks(
        _member(
            """
            async def worker(flag, values):
                if flag:
                    for value in values:
                        sink(value + 1)
                else:
                    sink(0)
                await checkpoint()
                done = True
                return done
            """
        )
    )

    assert plan.blocks[0].eligible is True
    assert plan.blocks[0].loop_count == 1
    assert plan.blocks[0].late_bound_globals == ("sink",)


def test_control_flow_assignment_retains_incoming_parameter() -> None:
    plan = plan_suspension_blocks(
        _member(
            """
                async def worker(x, flag):
                    if flag:
                        x = 10
                    first = x + 1
                    second = first * 2
                    audit(second)
                    await checkpoint()
            """
        )
    )

    block = plan.blocks[0]
    assert block.live_ins == ("audit", "flag", "x")
    assert block.eligible is True


def test_conditionally_assigned_prior_local_rejects_later_block() -> None:
    plan = plan_suspension_blocks(
        _member(
            """
            async def worker(flag):
                if flag:
                    x = 10
                await checkpoint()
                first = x + 1
                second = first * 2
                audit(second)
                await checkpoint()
            """
        )
    )

    later = plan.blocks[1]
    assert "x" in later.live_ins
    assert "unsafe_live_in" in {rejection.code for rejection in later.rejections}
    assert later.eligible is False


def test_augmented_assignment_is_both_live_in_and_live_out() -> None:
    plan = plan_suspension_blocks(
        _member(
            """
            async def worker(value):
                total = value
                await checkpoint()
                total += 2
                result = total * 3
                audit(result)
                await checkpoint()
                return result
            """
        )
    )

    second = plan.blocks[1]
    assert "total" in second.live_ins
    assert second.live_outs == ("result",)
    assert second.eligible is True


def test_self_referential_assignment_reads_shell_value_before_store() -> None:
    plan = plan_suspension_blocks(
        _member(
            """
            async def worker(total):
                await checkpoint()
                total = total + 2
                result = total * 3
                audit(result)
                await checkpoint()
                return result
            """
        )
    )

    block = plan.blocks[0]
    assert block.live_ins == ("audit", "total")
    assert block.live_outs == ("result",)
    assert block.eligible is True


def test_same_line_suspension_uses_column_aware_liveness() -> None:
    plan = plan_suspension_blocks(
        _member(
            "async def worker(values): start = len(values) + 1; "
            "doubled = start * 2; total = doubled + 3; await checkpoint(); "
            "result = total * 2; return result"
        )
    )

    first = plan.blocks[0]
    second = plan.blocks[1]
    assert first.start_lineno == first.end_lineno == 1
    assert first.start_col_offset < first.end_col_offset < second.start_col_offset
    assert first.live_outs == ("total",)
    assert first.eligible is True


def test_local_import_names_are_bound_inside_the_block() -> None:
    plan = plan_suspension_blocks(
        _member(
            """
            async def worker(values):
                await checkpoint()
                import math
                start = math.floor(len(values)) + 1
                doubled = start * 2
                audit(doubled)
                return doubled
            """
        )
    )

    block = plan.blocks[0]
    assert "math" in block.assigned_names
    assert "math" not in block.live_ins
    assert block.eligible is True


def test_later_delete_requires_helper_value_as_live_out() -> None:
    plan = plan_suspension_blocks(
        _member(
            """
            async def worker():
                x = 1
                first = x + 1
                x = first + 2
                audit(x)
                await checkpoint()
                del x
                return "done"
            """
        )
    )

    assert plan.blocks[0].live_outs == ("x",)
    assert plan.blocks[0].eligible is True
    assert "local_delete" in {rejection.code for rejection in plan.blocks[1].rejections}


@pytest.mark.parametrize(
    ("statement", "expected_code"),
    [
        ("mapped = [transform(value) for value in values]", "comprehension_scope"),
        ("del values[0]", "local_delete"),
        ("candidate: int", "annotation_only_local"),
        ("type Candidate = int", "local_type_alias"),
        ("candidate = (value := len(values))", "named_expression"),
    ],
)
def test_scope_and_deletion_effects_stay_interpreted(
    statement: str,
    expected_code: str,
) -> None:
    plan = plan_suspension_blocks(
        _member(
            f"""
            async def worker(values):
                {statement}
                first = len(values) + 1
                second = first * 2
                audit(second)
                await checkpoint()
            """
        )
    )

    assert expected_code in {rejection.code for rejection in plan.blocks[0].rejections}
    assert plan.blocks[0].eligible is False


def test_stable_ids_are_content_derived_and_deterministic() -> None:
    source_text = """
    async def worker(values):
        total = 0
        for value in values:
            total += value
        sink(total)
        await checkpoint()
    """

    first = plan_suspension_blocks(_member(source_text))
    second = plan_suspension_blocks(_member(source_text))
    changed = plan_suspension_blocks(_member(source_text.replace("value", "item"), "worker"))

    assert first.blocks[0].id == second.blocks[0].id
    assert first.blocks[0].id.startswith("susp-")
    assert first.blocks[0].id != changed.blocks[0].id
