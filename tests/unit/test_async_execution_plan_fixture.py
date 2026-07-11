"""Tests for the persistent generic async execution-plan acceptance fixture."""

from __future__ import annotations

import ast
import asyncio
import importlib
import sys
import tomllib
from collections.abc import Coroutine, Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast

FIXTURE_ROOT = Path("tests/fixtures/async_execution_plan_project")
SOURCE_ROOT = FIXTURE_ROOT / "src"
WORKFLOW_SOURCE = SOURCE_ROOT / "execution_plan_fixture" / "workflow.py"
EXPECTED_REPETITIONS = 32
EXPECTED_TOTAL = 10


class FixtureModule(Protocol):
    """Loaded fixture module interface used by these tests."""

    MATRIX_REPETITIONS: int

    def repeat_baseline_semantic_matrix(
        self,
    ) -> Coroutine[object, object, tuple[Mapping[str, object], ...]]: ...


def test_fixture_project_has_configured_command_arrays() -> None:
    config = tomllib.loads((FIXTURE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    project = cast(Mapping[str, object], config["project"])
    compile_config = cast(
        Mapping[str, object],
        cast(Mapping[str, object], cast(Mapping[str, object], config["tool"])["atoll"])["compile"],
    )

    assert project["name"] == "async-execution-plan-project"
    assert compile_config["test_command"] == ["python", "-m", "pytest", "tests", "-q"]
    assert compile_config["benchmark_command"] == [
        "python",
        "benchmarks/run_workload.py",
        "--minimum-seconds",
        "0.25",
    ]


def test_repeated_baseline_semantic_matrix_is_canonical_and_stable() -> None:
    module = _fixture_module()
    matrix_repetitions = module.MATRIX_REPETITIONS
    repeat_matrix = module.repeat_baseline_semantic_matrix
    matrix = asyncio.run(repeat_matrix())

    assert len(matrix) == matrix_repetitions == EXPECTED_REPETITIONS
    assert all(snapshot == matrix[0] for snapshot in matrix)
    assert matrix[0] == {
        "workflow": "taskgroup-queue-reduction",
        "capacity": 4,
        "total": EXPECTED_TOTAL,
        "count": 3,
        "first": "alpha",
        "last": "gamma",
        "exception_type": "ControlledImmediateError",
        "exception_message": "controlled:failure",
        "exception_frame_present": True,
        "context_parent": "parent",
        "context_child": "child",
        "context_sibling": "parent",
        "cold_decoy_count": 7,
    }
    assert _only_canonical_primitives(matrix)


def test_source_contains_supported_taskgroup_queue_candidate_shape() -> None:
    tree = ast.parse(WORKFLOW_SOURCE.read_text(encoding="utf-8"))
    functions = _async_functions(tree)

    producer = functions["publish_immediate"]
    queue_calls = [
        node
        for node in ast.walk(producer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "put_nowait"
    ]
    assert len(queue_calls) == 1
    assert not any(isinstance(node, ast.Await) for node in ast.walk(producer))

    owner = functions["run_supported_workflow"]
    assert any(isinstance(node, ast.AsyncWith) for node in ast.walk(owner))
    assert any(
        isinstance(node, ast.Attribute) and node.attr == "TaskGroup" for node in ast.walk(owner)
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "publish_immediate"
        for node in ast.walk(owner)
    )
    assert any(
        isinstance(node, ast.Await)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "get"
        for node in ast.walk(owner)
    )
    assert _queue_capacity_is_positive_constant(tree)


def test_source_contains_rejected_decoy_shapes() -> None:
    source = WORKFLOW_SOURCE.read_text(encoding="utf-8")

    expected_functions = {
        "cold_suspension_workflow",
        "cold_task_introspection_workflow",
        "cold_custom_task_factory_workflow",
        "cold_cancellation_workflow",
        "cold_context_isolation_workflow",
        "cold_debug_mode_workflow",
        "cold_side_effecting_iterable_workflow",
    }
    functions = _async_functions(ast.parse(source))

    assert expected_functions <= functions.keys()
    assert "await asyncio.sleep(0)" in source
    assert "asyncio.current_task()" in source
    assert "loop.set_task_factory" in source
    assert "task.cancel()" in source
    assert "ContextVar" in source
    assert "loop.set_debug(True)" in source
    assert "_side_effecting_values(items)" in source
    assert "pydantic" not in source.lower()


def _async_functions(tree: ast.AST) -> dict[str, ast.AsyncFunctionDef]:
    return {node.name: node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)}


def _queue_capacity_is_positive_constant(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "QUEUE_CAPACITY"
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, int)
        ):
            return node.value.value > 0
    return False


def _fixture_module() -> FixtureModule:
    source_root = str(SOURCE_ROOT.resolve())
    sys.path.insert(0, source_root)
    try:
        return cast(FixtureModule, importlib.import_module("execution_plan_fixture"))
    finally:
        sys.path.remove(source_root)


def _only_canonical_primitives(value: object) -> bool:
    if isinstance(value, str | int | bool):
        return True
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        items = cast(Sequence[object], value)
        return all(_only_canonical_primitives(item) for item in items)
    if isinstance(value, Mapping):
        entries = cast(Mapping[object, object], value)
        return all(
            isinstance(key, str) and _only_canonical_primitives(item)
            for key, item in entries.items()
        )
    return False
