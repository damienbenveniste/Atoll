"""Tests for disposable eager-task fusion source generation."""

from __future__ import annotations

import ast
import runpy
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from textwrap import dedent
from typing import cast

import pytest

from atoll.analysis.task_fusion import FusionPlan
from atoll.generation import task_fusion as task_fusion_generation
from atoll.generation.task_fusion import generate_eager_task_fusion

EXPECTED_TASK_RESULT = 5


def test_wraps_exact_spawn_and_preserves_future_import_order() -> None:
    source = dedent(
        '''\
        #!/usr/bin/env python
        """Example module."""
        from __future__ import annotations

        async def worker(value: int) -> int:
            return value + 1

        async def root(value: int):
            return asyncio.create_task(worker(value))
        '''
    )
    spawn = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "create_task"
    )
    plan = _plan(spawn)

    generated = generate_eager_task_fusion(source, plan)

    compile(generated.new_text, "sample.py", "exec")
    assert generated.plan_id == plan.id
    assert generated.helper_name in generated.new_text
    assert generated.new_text.index(
        "from __future__ import annotations"
    ) < generated.new_text.index(f"def {generated.helper_name}")
    assert (
        f"return {generated.helper_name}(asyncio.create_task, worker(value))" in generated.new_text
    )


def test_rejects_ineligible_or_stale_plan() -> None:
    source = "async def root():\n    asyncio.create_task(worker())\n"
    spawn = next(node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call))
    plan = _plan(spawn)

    with pytest.raises(ValueError, match="not eligible"):
        generate_eager_task_fusion(source, replace(plan, eligible=False))
    with pytest.raises(ValueError, match="resolved to 0"):
        generate_eager_task_fusion(source, replace(plan, lineno=plan.lineno + 1))
    with pytest.raises(ValueError, match="spawn API changed"):
        generate_eager_task_fusion(source, replace(plan, spawn_api="loop.create_task"))
    with pytest.raises(ValueError, match="spawn source changed"):
        generate_eager_task_fusion(source, replace(plan, spawn_source="changed()"))


def test_generated_wrapper_preserves_task_result_and_restores_factory(tmp_path: Path) -> None:
    source = (
        "import asyncio\n\n"
        "async def worker(value: int) -> int:\n"
        "    return value + 1\n\n"
        "async def root(value: int):\n"
        "    return asyncio.create_task(worker(value))\n"
    )
    spawn = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "create_task"
    )
    generated = generate_eager_task_fusion(source, _plan(spawn))
    module_path = tmp_path / "sample.py"
    module_path.write_text(
        generated.new_text
        + "\nasync def _verify():\n"
        + "    loop = asyncio.get_running_loop()\n"
        + "    previous_factory = loop.get_task_factory()\n"
        + "    task = await root(4)\n"
        + "    assert loop.get_task_factory() is previous_factory\n"
        + "    return await task\n\n"
        + "RESULT = asyncio.run(_verify())\n",
        encoding="utf-8",
    )

    namespace = runpy.run_path(str(module_path))

    assert cast(int, namespace["RESULT"]) == EXPECTED_TASK_RESULT


def test_rejects_generated_helper_name_collision() -> None:
    source = (
        "def _atoll_eager_spawn_test():\n"
        "    return None\n\n"
        "async def root():\n"
        "    asyncio.create_task(worker())\n"
    )
    calls = tuple(node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call))
    spawn = next(node for node in calls if isinstance(node.func, ast.Attribute))

    with pytest.raises(ValueError, match="already exists"):
        generate_eager_task_fusion(source, _plan(spawn))


def test_preserves_encoding_cookie_before_inserted_helper() -> None:
    source = (
        "# coding: utf-8\nimport asyncio\n\nasync def root():\n    asyncio.create_task(worker())\n"
    )
    spawn = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    )

    generated = generate_eager_task_fusion(source, _plan(spawn))

    assert generated.new_text.startswith("# coding: utf-8\n")
    assert generated.new_text.index(generated.helper_name) > generated.new_text.index("coding")


def test_source_edit_helpers_reject_out_of_bounds_offsets() -> None:
    line_start = cast(
        Callable[[bytes, int], int],
        vars(task_fusion_generation)["_line_start_offset"],
    )
    apply_edits = cast(
        Callable[[bytes, tuple[tuple[int, int, bytes], ...]], bytes],
        vars(task_fusion_generation)["_apply_edits"],
    )

    with pytest.raises(ValueError, match="outside module"):
        line_start(b"one\n", 3)
    with pytest.raises(ValueError, match="outside module bounds"):
        apply_edits(b"one", ((4, 4, b""),))


def _plan(spawn: ast.Call) -> FusionPlan:
    return FusionPlan(
        id="task-fusion:test",
        source_hash="source-hash",
        root="sample::root",
        caller="sample::root",
        callee="sample::worker",
        spawn_api="asyncio.create_task",
        lineno=spawn.lineno,
        end_lineno=spawn.end_lineno or spawn.lineno,
        col_offset=spawn.col_offset,
        end_col_offset=spawn.end_col_offset,
        eligible=True,
        observed_calls=25,
        completed_calls=25,
        max_active_calls=1,
        pre_completion_suspensions=0,
        observed_signatures=1,
        observation_capped=False,
        rejections=(),
        spawn_source=ast.unparse(spawn),
    )
