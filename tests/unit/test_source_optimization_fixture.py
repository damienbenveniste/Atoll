"""Tests for the generic source-optimization acceptance fixture."""

from __future__ import annotations

import ast
import asyncio
import importlib
import importlib.util
import json
import sys
import tomllib
from collections.abc import Coroutine, Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast

import pytest

FIXTURE_ROOT = Path("tests/fixtures/source_optimization_project")
SOURCE_ROOT = FIXTURE_ROOT / "src"
WORKFLOW_SOURCE = SOURCE_ROOT / "source_optimization_fixture" / "workflow.py"
BENCHMARK_SCRIPT = FIXTURE_ROOT / "benchmarks" / "run_workload.py"
EXPECTED_WORK_ITEM_COUNT = 256
EXPECTED_CHECKSUM = 300136
BENCHMARK_TEST_ITERATIONS = 40
MINIMUM_SOURCE_SPEEDUP = 3.0


class FixtureModule(Protocol):
    """Loaded fixture module interface used by these tests."""

    MATRIX_REPETITIONS: int

    def repeat_baseline_semantic_matrix(
        self,
    ) -> Coroutine[object, object, tuple[Mapping[str, object], ...]]:
        """Return the deterministic semantic matrix."""
        ...


class BenchmarkModule(Protocol):
    """Loaded benchmark module interface used by these tests."""

    def main(self) -> int:
        """Run the benchmark entry point."""
        ...


def test_fixture_project_has_configured_commands() -> None:
    config = tomllib.loads((FIXTURE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    project = cast(Mapping[str, object], config["project"])
    compile_config = cast(
        Mapping[str, object],
        cast(Mapping[str, object], cast(Mapping[str, object], config["tool"])["atoll"])["compile"],
    )

    assert project["name"] == "source-optimization-project"
    assert compile_config["test_command"] == ["python", "-m", "pytest", "tests", "-q"]
    assert compile_config["benchmark_command"] == [
        "python",
        "benchmarks/run_workload.py",
        "--iterations",
        "1024",
    ]
    assert compile_config["minimum_speedup"] == MINIMUM_SOURCE_SPEEDUP


def test_repeated_semantic_matrix_is_canonical_and_stable() -> None:
    module = _fixture_module()
    matrix = asyncio.run(module.repeat_baseline_semantic_matrix())

    assert len(matrix) == module.MATRIX_REPETITIONS
    assert all(snapshot == matrix[0] for snapshot in matrix)
    assert matrix[0] == {
        "workflow": "private-taskgroup-queue-reduction",
        "work_count": EXPECTED_WORK_ITEM_COUNT,
        "queue_capacity": EXPECTED_WORK_ITEM_COUNT,
        "result_count": EXPECTED_WORK_ITEM_COUNT,
        "checksum": EXPECTED_CHECKSUM,
        "first_label": "work-0000",
        "last_label": "work-0255",
        "parent_context": "parent",
        "child_context": "worker:child",
        "sibling_context": "parent",
        "worker_context": "worker:work-0000",
        "exception_type": "ControlledWorkflowError",
        "exception_message": "controlled failure: source-optimization",
        "cancellation_cancelled": True,
        "cancellation_cleanup_count": 1,
        "cancellation_cleanup_marker": "cleanup-complete",
        "introspection_current_task_seen": True,
        "introspection_task_name": "source-optimization-introspection",
        "iterator_values": (2, 3, 5, 8),
        "iterator_snapshots": (2, 5, 10, 18),
        "iterator_final_total": 18,
        "unsupported_context_parent": "outer",
        "unsupported_context_child": "unsupported-child",
    }
    assert _only_canonical_primitives(matrix)


def test_source_contains_private_hot_pipeline_shape() -> None:
    tree = ast.parse(WORKFLOW_SOURCE.read_text(encoding="utf-8"))
    functions = _async_functions(tree)

    hot = functions["_run_hot_private_pipeline"]
    assert _local_queue_capacity_matches_work_count(hot)
    assert any(isinstance(node, ast.AsyncWith) for node in ast.walk(hot))
    assert any(
        isinstance(node, ast.Attribute) and node.attr == "TaskGroup" for node in ast.walk(hot)
    )
    assert any(
        isinstance(node, ast.For)
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "create_task"
            and child.args
            and isinstance(child.args[0], ast.Call)
            and isinstance(child.args[0].func, ast.Name)
            and child.args[0].func.id == "_immediate_worker"
            for child in ast.walk(node)
        )
        for node in ast.walk(hot)
    )
    assert any(
        isinstance(node, ast.Await)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "get"
        for node in ast.walk(hot)
    )


def test_immediate_worker_mutates_context_directly_and_never_suspends() -> None:
    worker = _async_functions(ast.parse(WORKFLOW_SOURCE.read_text(encoding="utf-8")))[
        "_immediate_worker"
    ]

    assert not any(isinstance(node, ast.Await) for node in ast.walk(worker))
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "set"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "_WORKER_CONTEXT"
        for node in ast.walk(worker)
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_unsupported_context_mutator"
        for node in ast.walk(worker)
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "put_nowait"
        for node in ast.walk(worker)
    )


def test_source_contains_required_cold_paths_without_pydantic_identifiers() -> None:
    source = WORKFLOW_SOURCE.read_text(encoding="utf-8")
    functions = _async_functions(ast.parse(source))

    assert {
        "cold_suspending_worker",
        "cold_controlled_exception_path",
        "cold_cancellation_cleanup_path",
        "cold_task_introspection_path",
        "public_incremental_inspection",
        "cold_unsupported_indirect_context_mutation",
    } <= functions.keys()
    assert "await asyncio.sleep(0)" in source
    assert "asyncio.current_task()" in source
    assert "task.cancel()" in source
    assert "_unsupported_context_mutator()" in source
    assert "ContextVar" in source
    assert "pydantic" not in source.lower()


def test_benchmark_entry_point_prints_stable_json_checksum(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark = _benchmark_module()
    monkeypatch.chdir(FIXTURE_ROOT)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_workload.py", "--iterations", str(BENCHMARK_TEST_ITERATIONS)],
    )

    assert benchmark.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "checksum": EXPECTED_CHECKSUM * BENCHMARK_TEST_ITERATIONS,
        "iterations": BENCHMARK_TEST_ITERATIONS,
        "logical_items": EXPECTED_WORK_ITEM_COUNT * BENCHMARK_TEST_ITERATIONS,
    }


def _async_functions(tree: ast.AST) -> dict[str, ast.AsyncFunctionDef]:
    return {node.name: node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)}


def _local_queue_capacity_matches_work_count(node: ast.AsyncFunctionDef) -> bool:
    for child in ast.walk(node):
        if not (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "Queue"
        ):
            continue
        return bool(
            child.keywords
            and isinstance(child.keywords[0].value, ast.Call)
            and isinstance(child.keywords[0].value.func, ast.Name)
            and child.keywords[0].value.func.id == "len"
            and child.keywords[0].value.args
            and isinstance(child.keywords[0].value.args[0], ast.Name)
            and child.keywords[0].value.args[0].id == "items"
        )
    return False


def _fixture_module() -> FixtureModule:
    source_root = str(SOURCE_ROOT.resolve())
    sys.path.insert(0, source_root)
    try:
        return cast(FixtureModule, importlib.import_module("source_optimization_fixture"))
    finally:
        sys.path.remove(source_root)


def _benchmark_module() -> BenchmarkModule:
    spec = importlib.util.spec_from_file_location("source_optimization_benchmark", BENCHMARK_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("benchmark module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(BenchmarkModule, module)


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
