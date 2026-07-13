"""Tests for guarded source lowering of class-owned AnyIO result streams."""

from __future__ import annotations

import ast
import asyncio
import functools
import hashlib
import importlib.util
import inspect
import shutil
import sys
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from dataclasses import replace
from pathlib import Path, PurePosixPath
from types import ModuleType, SimpleNamespace
from typing import Protocol, cast

import libcst as cst
import pytest

import atoll.source_optimization.anyio_stream_lowering as anyio_lowering
import atoll.source_optimization.search as source_search
from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.clustering import enrich_island_analysis
from atoll.generation.run_guard import RunGuardGenerationRequest, generate_run_guard
from atoll.models import ModuleId, SymbolId
from atoll.native_optimization.run_guard import build_run_guard_region
from atoll.source_optimization import (
    SourceCallableEvidence,
    SourceOptimizationAssessment,
    SourceOptimizationIdentity,
    SourceOptimizationPlan,
    TransformationStep,
    stable_source_optimization_plan_id,
)
from atoll.source_optimization.anyio_stream_lowering import lower_anyio_stream_plan
from atoll.source_optimization.lowering import (
    SourceLoweringResult,
    lower_residual_state_machine_plan,
    lower_state_machine_plan,
)
from atoll.source_optimization.models import SourceTransformationKind
from atoll.source_optimization.transforms import (
    build_source_transformation_patch,
    materialize_transformed_files,
)

FIXTURE_ROOT = Path("tests/fixtures/source_optimization_project")
SOURCE_PATH = PurePosixPath("src/source_optimization_fixture/anyio_workflow.py")
MODULE = "source_optimization_fixture.anyio_workflow"
OWNER = SymbolId(MODULE, "PipelineRunner.submit")
WORKER = SymbolId(MODULE, "PipelineRunner._worker")
CONSUMER = SymbolId(MODULE, "PipelineRunner.results")
RESIDUAL_STEPS: tuple[SourceTransformationKind, ...] = (
    "run-scoped-guard-amortization",
    "transparent-quiescent-await-chain-collapse",
    "context-copy-elision",
    "incremental-private-completion-accounting",
    "private-result-record-elision",
)
EXPECTED_SKIPPED_STEP_REJECTIONS = 2
_ANYIO_PRIVATE = vars(anyio_lowering)


class _SourceVariantView(Protocol):
    transformation_ids: tuple[str, ...]


INDEXED_COMPLETION_SOURCE = """class StackItem:
    def __init__(self, run_id):
        self.run_id = run_id


class Task:
    def __init__(self, key, node_id, stack):
        self.key = key
        self.node_id = node_id
        self.stack = stack


class Parent:
    intermediate_nodes = {"middle"}


class Graph:
    def get_parent(self, join_id):
        return Parent()


class _IndexedRunner:
    active: dict[object, Task]

    def __post_init__(self):
        self.active = {}
        self.graph = Graph()

    async def consume(self, task):
        self.active[task.key] = task
        active_values = list(self.active.values())
        for completed in self.completed(task, active_values):
            del completed

    async def finish(self, key):
        self.active.pop(key, None)

    def submit(self, task):
        self.active[task.key] = task

    def completed(self, task, active_tasks):
        join_id = task.node_id
        fork_run_id = task.stack[0].run_id
        if self.is_complete(active_tasks, join_id, fork_run_id):
            return [(join_id, fork_run_id)]
        return []

    def is_complete(self, tasks, join_id, fork_run_id):
        parent = self.graph.get_parent(join_id)
        for task in tasks:
            if fork_run_id in {item.run_id for item in task.stack}:
                if task.node_id in parent.intermediate_nodes or task.node_id == join_id:
                    return False
            else:
                pass
        return True
"""


def _indexed_anyio_fixture_source() -> str:
    """Return the generic AnyIO fixture with an exact private completion scan."""
    source = (FIXTURE_ROOT / SOURCE_PATH).read_text(encoding="utf-8")
    support = """
@dataclass(frozen=True)
class StackItem:
    run_id: int


@dataclass(frozen=True)
class ParentGroup:
    intermediate_nodes: set[str]


class Graph:
    def get_parent(self, join_id: str) -> ParentGroup:
        del join_id
        return ParentGroup({"middle"})


"""
    source = source.replace(
        "@dataclass\nclass PipelineRunner:", f"{support}@dataclass\nclass PipelineRunner:"
    )
    source = source.replace(
        "    value: int\n\n\n@dataclass(frozen=True)\nclass StepNode:",
        "    value: int\n"
        "    stack: tuple[StackItem, ...]\n\n\n"
        "@dataclass(frozen=True)\n"
        "class StepNode:",
    )
    source = source.replace("    active: dict[int, WorkItem]", "    _active: dict[int, WorkItem]")
    source = source.replace("self.active", "self._active")
    source = source.replace(
        "self._active.pop(record.source.item_id)",
        "self._active.pop(record.source.item_id, None)",
    )
    source = source.replace(
        "        self.reducers = {}\n",
        "        self.reducers = {}\n        self.graph = Graph()\n",
    )
    source = source.replace(
        """        del source, active
        return ()
""",
        """        join_id = source.node_id
        fork_run_id = source.stack[0].run_id
        if self._is_complete(active, join_id, fork_run_id):
            return (0,)
        return ()

    def _is_complete(
        self,
        tasks: list[WorkItem],
        join_id: str,
        fork_run_id: int,
    ) -> bool:
        parent = self.graph.get_parent(join_id)
        for task in tasks:
            if fork_run_id in {item.run_id for item in task.stack}:
                if task.node_id in parent.intermediate_nodes or task.node_id == join_id:
                    return False
            else:
                pass
        return True
""",
    )
    return source.replace(
        'WorkItem(item_id=index, node_id="step", value=value)',
        'WorkItem(item_id=index, node_id="step", value=value, stack=(StackItem(1),))',
    )


def test_anyio_lowering_builds_reproducible_libcst_patch() -> None:
    """The structural method pipeline produces a stable source-clean patch."""
    plan, assessment = _plan_and_assessment()
    source_path = FIXTURE_ROOT / SOURCE_PATH
    before = source_path.read_bytes()

    first = lower_state_machine_plan(FIXTURE_ROOT, plan, assessment)
    second = lower_state_machine_plan(FIXTURE_ROOT, plan, assessment)

    assert first == second
    assert first.status == "lowered"
    assert first.request is not None
    patch = build_source_transformation_patch(FIXTURE_ROOT, (first.request,))
    assert (
        patch.patch_text
        == build_source_transformation_patch(FIXTURE_ROOT, (first.request,)).patch_text
    )
    transformed = patch.files[0].after_source
    assert "copy_context" in transformed
    assert "receive_nowait" in transformed
    assert "if self.reducers:" in transformed
    assert "ATOLL_REQUIRE_OPTIMIZED" in transformed
    assert source_path.read_bytes() == before


def test_indexed_completion_frontend_and_fallback_helpers_are_structural(
    tmp_path: Path,
) -> None:
    """A private exact-dict scan is indexed without benchmark-specific identities."""
    tree = ast.parse(INDEXED_COMPLETION_SOURCE)
    class_node = _ANYIO_PRIVATE["_class_node"](tree, "_IndexedRunner")
    consumer = cast(
        ast.AsyncFunctionDef,
        _ANYIO_PRIVATE["_named_method"](
            class_node,
            "consume",
            ast.AsyncFunctionDef,
            role="consumer",
        ),
    )
    shape = _ANYIO_PRIVATE["_analyze_native_completion_shape"](class_node, consumer)
    names = _ANYIO_PRIVATE["_names"]("indexed-completion-test")

    assert shape.active_attribute == "active"
    assert shape.predicate_method == "is_complete"
    assert shape.task_stack_attribute == "stack"
    assert shape.stack_run_attribute == "run_id"
    assert shape.task_node_attribute == "node_id"
    assert {
        method.name: (method.assignments, method.pops, method.snapshots, method.queries)
        for method in shape.methods
    } == {
        "consume": (1, 0, 1, 0),
        "finish": (0, 1, 0, 0),
        "submit": (1, 0, 0, 0),
        "completed": (0, 0, 0, 1),
    }

    support = _ANYIO_PRIVATE["_completion_index_support"](shape, names)
    module_path = tmp_path / "indexed_completion_runtime.py"
    module_path.write_text(f"{INDEXED_COMPLETION_SOURCE}\n{support}", encoding="utf-8")
    runtime = _load_module(module_path, "indexed_completion_runtime")
    namespace = vars(runtime)
    runner_type = cast(type[object], namespace["_IndexedRunner"])
    task_type = cast(Callable[..., object], namespace["Task"])
    stack_type = cast(Callable[..., object], namespace["StackItem"])
    runner = runner_type()
    initializer = cast(Callable[[object], None], vars(runner_type)["__post_init__"])
    initializer(runner)
    setattr(runner, names.completion_index_attribute, {})
    setattr(runner, names.completion_count_attribute, 0)
    task = task_type("task", "join", [stack_type("run")])
    indexed_set = cast(Callable[[object, object, object], None], namespace[names.completion_set])
    indexed_pop = cast(
        Callable[[object, object, object], object],
        namespace[names.completion_pop],
    )
    snapshot = cast(Callable[[object], list[object]], namespace[names.completion_snapshot])
    query = cast(
        Callable[[object, list[object], object, object], bool],
        namespace[names.completion_query],
    )

    indexed_set(runner, "task", task)
    assert snapshot(runner) == [task]
    assert query(runner, snapshot(runner), "join", "run") is False
    assert getattr(runner, names.completion_count_attribute) == 1
    assert indexed_pop(runner, "task", None) is task
    assert snapshot(runner) == []
    assert getattr(runner, names.completion_count_attribute) == 0


def test_indexed_completion_cst_edits_match_every_proven_site() -> None:
    """Formatting-aware edits replace every mutation, snapshot, and predicate call."""
    tree = ast.parse(INDEXED_COMPLETION_SOURCE)
    class_node = _ANYIO_PRIVATE["_class_node"](tree, "_IndexedRunner")
    consumer = cast(
        ast.AsyncFunctionDef,
        _ANYIO_PRIVATE["_named_method"](
            class_node,
            "consume",
            ast.AsyncFunctionDef,
            role="consumer",
        ),
    )
    shape = _ANYIO_PRIVATE["_analyze_native_completion_shape"](class_node, consumer)
    names = _ANYIO_PRIVATE["_names"]("indexed-completion-cst")
    module = cst.parse_module(INDEXED_COMPLETION_SOURCE)

    rendered: dict[str, str] = {}
    for method in shape.methods:
        declaration = _ANYIO_PRIVATE["_cst_method"](module, f"_IndexedRunner.{method.name}")
        body = cast(cst.IndentedBlock, declaration.body)
        transformer = _ANYIO_PRIVATE["_CompletionIndexTransformer"](shape, method, names)
        updated = cast(cst.IndentedBlock, body.visit(transformer))
        transformer.validate()
        rendered[method.name] = _ANYIO_PRIVATE["_body_source"](module, updated)

    assert names.completion_set in rendered["consume"]
    assert names.completion_snapshot in rendered["consume"]
    assert names.completion_pop in rendered["finish"]
    assert names.completion_set in rendered["submit"]
    assert names.completion_query in rendered["completed"]


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("class _IndexedRunner:", "class IndexedRunner:", "private owner or mapping"),
        ("active: dict[object, Task]", "active: object", "exact dict field"),
        (
            "active_values = list(self.active.values())",
            "active_values = []",
            "one active-value snapshot",
        ),
        (
            "self.completed(task, active_values)",
            "self.completed(task, [])",
            "snapshot must feed one",
        ),
        (
            "self.is_complete(active_tasks, join_id, fork_run_id)",
            "True",
            "must call one completion predicate",
        ),
        (
            "def is_complete(self, tasks, join_id, fork_run_id):",
            "def is_complete(self, tasks, join_id, fork_run_id, extra):",
            "four positional parameters",
        ),
        (
            "self.completed(task, active_values)",
            "self.completed(active_values, active_values)",
            "argument binding is ambiguous",
        ),
        ("return True\n", "return None\n", "unsupported control flow"),
        (
            "self.graph.get_parent(join_id)",
            "self.graph.parents[join_id]",
            "parent lookup is not direct",
        ),
        (
            "fork_run_id in {item.run_id for item in task.stack}",
            "fork_run_id == 1",
            "run membership is unsupported",
        ),
        (
            "{item.run_id for item in task.stack}",
            "{item.run_id for item in task.stack if item.run_id}",
            "stack projection is unsupported",
        ),
        ("for task in tasks:", "for task in ():", "one direct task scan"),
        (
            "                if task.node_id in parent.intermediate_nodes",
            "                if False",
            "node comparisons are unsupported",
        ),
        (
            "            else:\n                pass",
            "            else:\n                return False",
            "run miss branch must be inert",
        ),
        (
            "            if fork_run_id in {item.run_id for item in task.stack}:\n"
            "                if task.node_id",
            "            if fork_run_id in {item.run_id for item in task.stack}:\n"
            "                pass\n"
            "                if task.node_id",
            "run membership must guard one node test",
        ),
        ("return False\n", "return True\n", "node match must return False"),
        (
            "task.node_id in parent.intermediate_nodes or task.node_id == join_id",
            "task.node_id in parent.intermediate_nodes and task.node_id == join_id",
            "node test must be one disjunction",
        ),
        (
            "task.node_id in parent.intermediate_nodes",
            "task.node_id not in parent.intermediate_nodes",
            "node comparisons are unsupported",
        ),
        (
            "task.node_id in parent.intermediate_nodes",
            "other.node_id in parent.intermediate_nodes",
            "node projection is unsupported",
        ),
        (
            "task.node_id == join_id",
            "task.node_id == join_id == fork_run_id",
            "node comparisons are unsupported",
        ),
        ("active: dict[object, Task]", "active: list[Task]", "exact dict field"),
        ("self.active = {}", "self.active = dict()", "exact dict field"),
        (
            "self.active[task.key] = task",
            "self.active[get_key(task)] = task",
            "assignment is not directly indexable",
        ),
        (
            "self.active[task.key] = task",
            "self.active[task.key] = other = task",
            "assignment is not directly indexable",
        ),
        (
            "    def submit(self, task):\n        self.active[task.key] = task",
            "    def submit(self, task):\n"
            "        self.active = {}\n"
            "        self.active[task.key] = task",
            "rebound after initialization",
        ),
        (
            "self.active.pop(key, None)",
            "self.active.clear()",
            "unsupported mutator",
        ),
        (
            "self.active.pop(key, None)",
            "del self.active[key]",
            "direct deletion",
        ),
        (
            "self.active.pop(key, None)",
            "self.active.pop(key)",
            "ignored pop with None default",
        ),
        (
            "self.active.pop(key, None)",
            "self.active.pop(key, object())",
            "ignored pop with None default",
        ),
        (
            "self.active.pop(key, None)",
            "pass",
            "owned assignments and one removal",
        ),
    ],
)
def test_indexed_completion_frontend_rejects_unsafe_shapes(
    old: str,
    new: str,
    message: str,
) -> None:
    """Each ambiguous ownership or predicate shape remains interpreted."""
    source = INDEXED_COMPLETION_SOURCE.replace(old, new, 1)
    tree = ast.parse(source)
    class_name = "IndexedRunner" if "class IndexedRunner:" in source else "_IndexedRunner"
    class_node = _ANYIO_PRIVATE["_class_node"](tree, class_name)
    consumer = _ANYIO_PRIVATE["_named_method"](
        class_node,
        "consume",
        ast.AsyncFunctionDef,
        role="consumer",
    )

    with pytest.raises((TypeError, ValueError), match=message):
        _ANYIO_PRIVATE["_analyze_native_completion_shape"](class_node, consumer)


def test_indexed_completion_internal_fallback_branches_are_conservative() -> None:
    """Read-only accesses are ignored while mismatched CST and method shapes fail."""
    source = INDEXED_COMPLETION_SOURCE.replace(
        "    def is_complete(self, tasks, join_id, fork_run_id):",
        "    def lookup(self, key):\n"
        "        return self.active[key]\n\n"
        "    def is_complete(self, tasks, join_id, fork_run_id):",
    ).replace(
        "        active_values = list(self.active.values())",
        "        ignored = list(self.active.values(1))\n"
        "        active_values = list(self.active.values())",
    )
    tree = ast.parse(source)
    class_node = _ANYIO_PRIVATE["_class_node"](tree, "_IndexedRunner")
    consumer = _ANYIO_PRIVATE["_named_method"](
        class_node,
        "consume",
        ast.AsyncFunctionDef,
        role="consumer",
    )
    shape = _ANYIO_PRIVATE["_analyze_native_completion_shape"](class_node, consumer)
    method = next(item for item in shape.methods if item.name == "consume")
    with pytest.raises(ValueError, match="execution kind changed"):
        _ANYIO_PRIVATE["_merge_completion_method_shape"](
            (method,),
            replace(method, asynchronous=not method.asynchronous),
        )

    bad_tree = ast.parse(INDEXED_COMPLETION_SOURCE.replace("self.active = {}", "self.active = []"))
    bad_class = _ANYIO_PRIVATE["_class_node"](bad_tree, "_IndexedRunner")
    bad_consumer = _ANYIO_PRIVATE["_named_method"](
        bad_class,
        "consume",
        ast.AsyncFunctionDef,
        role="consumer",
    )
    assert _ANYIO_PRIVATE["_native_completion_shape"](bad_class, bad_consumer) is None

    names = _ANYIO_PRIVATE["_names"]("indexed-completion-mismatch")
    transformer = _ANYIO_PRIVATE["_CompletionIndexTransformer"](shape, method, names)
    empty_body = cst.IndentedBlock(body=(cst.parse_statement("pass\n"),))
    empty_body.visit(transformer)
    with pytest.raises(ValueError, match="rewrite counts"):
        transformer.validate()
    assert _ANYIO_PRIVATE["_completion_snapshot_call"](cst.Name("value"), "active") is False

    invalid_assignment = cst.parse_statement("self.active[first, second] = task\n")
    assert isinstance(invalid_assignment, cst.SimpleStatementLine)
    invalid_transformer = _ANYIO_PRIVATE["_CompletionIndexTransformer"](shape, method, names)
    with pytest.raises(ValueError, match="lost its direct key"):
        invalid_assignment.visit(invalid_transformer)

    unchanged = cst.IndentedBlock(body=(cst.parse_statement("pass\n"),))
    fake_shape = SimpleNamespace(native_completion=shape)
    assert (
        _ANYIO_PRIVATE["_apply_completion_index_body"](
            unchanged,
            fake_shape,
            "unknown_method",
            names,
        )
        is unchanged
    )
    multi_statement = cst.parse_statement("first = 1; second = 2\n")
    multi_transformer = _ANYIO_PRIVATE["_CompletionIndexTransformer"](shape, method, names)
    visited_multi = cast(cst.SimpleStatementLine, multi_statement.visit(multi_transformer))
    assert visited_multi.deep_equals(multi_statement)
    other_assignment = cst.parse_statement("self.other[key] = task\n")
    visited_other = cast(cst.SimpleStatementLine, other_assignment.visit(multi_transformer))
    assert visited_other.deep_equals(other_assignment)

    snapshot_assignment = _ANYIO_PRIVATE["_snapshot_assignment"]
    assert snapshot_assignment(cst.parse_statement("value = 1\n")) is None
    assert snapshot_assignment(cst.parse_statement("value = list(items)\n")) is None
    assert snapshot_assignment(cst.parse_statement("value = list(self.active.items())\n")) is None
    assert (
        _ANYIO_PRIVATE["_empty_list_assignment"](cst.parse_statement("first = []; second = []\n"))
        is False
    )

    comparison = ast.parse("left == right == extra", mode="eval").body
    node = ast.parse("task.node_id", mode="eval").body
    assert isinstance(comparison, ast.Compare)
    assert isinstance(node, ast.Attribute)
    assert _ANYIO_PRIVATE["_same_equality_operands"](comparison, node, "right") is False


@pytest.mark.parametrize(
    ("source", "constructor", "message"),
    [
        (
            "async def consume(self):\n    pass\n",
            "records.Result",
            "same-module nominal record",
        ),
        (
            "async def consume(self):\n    pass\n",
            "MissingResult",
            "one same-module record declaration",
        ),
        (
            "class Result:\n"
            "    source: object\n"
            "    value: object\n\n"
            "async def consume(self):\n"
            "    pass\n",
            "Result",
            "positional source/value and one error field",
        ),
        (
            "class Result:\n"
            "    source: object\n"
            "    value: object\n"
            "    error: object\n\n"
            "async def consume(self):\n"
            "    pass\n",
            "Result",
            "one named private receive target",
        ),
        (
            "class Result:\n"
            "    source: object\n"
            "    value: object\n"
            "    error: object\n\n"
            "async def consume(self):\n"
            "    async for record in self.receiver:\n"
            "        observed = record\n",
            "Result",
            "observes private result identity",
        ),
    ],
)
def test_result_record_proof_rejects_ambiguous_private_observation(
    source: str,
    constructor: str,
    message: str,
) -> None:
    """Result elision rejects nonlocal declarations and identity observations."""
    tree = ast.parse(source)
    consumer = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef))

    with pytest.raises(ValueError, match=message):
        _ANYIO_PRIVATE["_result_record_fields"](
            tree,
            consumer,
            constructor,
            "error",
            "self.receiver",
        )


@pytest.mark.parametrize(
    ("consumer_body", "message"),
    [
        ("async def consume(self):\n    pass\n", "shared request parameter"),
        (
            "async def consume(self, requests):\n    pass\n",
            "one owned submission call",
        ),
        (
            "async def consume(self, requests):\n"
            "    self.submit(requests)\n"
            "    async for record in self.receiver:\n"
            "        pass\n",
            "one active completion loop",
        ),
        (
            "async def consume(self, requests):\n"
            "    self.submit(requests)\n"
            "    while other.active:\n"
            "        async for record in self.receiver:\n"
            "            pass\n",
            "one private active mapping",
        ),
    ],
)
def test_completion_accounting_rejects_ambiguous_ownership(
    consumer_body: str,
    message: str,
) -> None:
    """Incremental accounting requires one owned request and completion loop."""
    indented_consumer = "\n".join(f"    {line}" for line in consumer_body.splitlines())
    tree = ast.parse(
        f"class Runner:\n    def submit(self, requests):\n        pass\n\n{indented_consumer}\n"
    )
    class_node = cast(ast.ClassDef, tree.body[0])
    consumer = next(node for node in class_node.body if isinstance(node, ast.AsyncFunctionDef))

    with pytest.raises(ValueError, match=message):
        _ANYIO_PRIVATE["_completion_shape"](
            class_node,
            consumer,
            "submit",
            "requests",
            "self.receiver",
        )


@pytest.mark.parametrize("async_result_type", [None, "AsyncResult"])
def test_worker_wrapper_rejects_unproven_await_shapes(
    async_result_type: str | None,
) -> None:
    """A producer cannot bypass an opaque await or omit async-result fallback."""
    source = (
        "async def worker(self, task):\n    await token\n"
        if async_result_type is None
        else "async def worker(self, task):\n    return task\n"
    )
    worker = cast(ast.AsyncFunctionDef, ast.parse(source).body[0])
    message = (
        "awaits behavior outside" if async_result_type is None else "no matching async-for fallback"
    )

    with pytest.raises(ValueError, match=message):
        _ANYIO_PRIVATE["_validate_worker_wrapper"](
            worker,
            run_method_name="run",
            sender_expression="self.sender",
            result_constructor="Result",
            async_result_type=async_result_type,
        )


def test_anyio_lowering_emits_transactional_indexed_native_plan(tmp_path: Path) -> None:
    """The public source lowerer carries indexed helpers through fresh staged scanning."""
    project_root = tmp_path / "project"
    source_path = project_root / SOURCE_PATH
    source_path.parent.mkdir(parents=True)
    source = _indexed_anyio_fixture_source()
    source_path.write_text(source, encoding="utf-8")
    base_plan, _assessment = _plan_and_assessment()
    identity = replace(
        base_plan.identity,
        source_hashes=((SOURCE_PATH, hashlib.sha256(source.encode("utf-8")).hexdigest()),),
    )
    plan = replace(
        base_plan,
        id=stable_source_optimization_plan_id(identity),
        identity=identity,
    )

    lowering = lower_anyio_stream_plan(project_root, plan)

    assert len(lowering.native_plans) == 1
    native_plan = lowering.native_plans[0]
    assert native_plan.completion_index is not None
    replacement_names = {
        replacement.target.qualname for replacement in lowering.request.additional_replacements
    }
    assert replacement_names.issuperset(
        {
            "PipelineRunner._completed_reducers",
            "PipelineRunner.results",
            "PipelineRunner.__post_init__",
        }
    )
    patch = build_source_transformation_patch(project_root, (lowering.request,))
    transformed = patch.files[0].after_source
    assert native_plan.completion_index.snapshot.qualname in transformed
    assert native_plan.completion_index.query.qualname in transformed

    source_path.write_text(transformed, encoding="utf-8")
    scan = enrich_island_analysis(scan_module(ModuleId(name=MODULE, path=source_path)))
    region = build_run_guard_region(scan, native_plan)
    generated = generate_run_guard(
        RunGuardGenerationRequest(
            scan=scan,
            region=region,
            plan=native_plan,
            logical_module="_atoll_indexed_anyio",
            output_path=tmp_path / "_atoll_indexed_anyio.py",
        )
    )

    assert generated.selected_members == tuple(member.id for member in region.members)
    assert {binding.source for binding in generated.bindings} == {
        native_plan.helper,
        native_plan.completion_index.snapshot,
        native_plan.completion_index.query,
    }


def test_anyio_lowering_rejects_stale_source_identity() -> None:
    """A source hash mismatch stops before shape analysis or patch generation."""
    plan, _ = _plan_and_assessment()
    stale = replace(
        plan,
        identity=replace(
            plan.identity,
            source_hashes=((SOURCE_PATH, "0" * 64),),
        ),
    )

    with pytest.raises(ValueError, match="stale source"):
        lower_anyio_stream_plan(FIXTURE_ROOT, stale)


def test_anyio_residual_lowering_composes_safe_cumulative_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Context, accounting, and record residuals share one guarded request."""
    plan, assessment = _plan_and_assessment()
    enabled = RESIDUAL_STEPS
    lowering = lower_residual_state_machine_plan(FIXTURE_ROOT, plan, assessment, enabled)

    assert lowering.status == "lowered"
    assert lowering.request is not None
    assert lowering.native_plans == ()
    patch = build_source_transformation_patch(FIXTURE_ROOT, (lowering.request,))
    transformed_source = patch.files[0].after_source
    assert "namedtuple" in transformed_source
    assert "_remaining = len(self.active)" in transformed_source
    assert "child_context =" not in transformed_source

    copied_root = tmp_path / "residual-project"
    shutil.copytree(FIXTURE_ROOT, copied_root)
    materialize_transformed_files(FIXTURE_ROOT, copied_root, patch)
    module = _load_module(
        copied_root / SOURCE_PATH,
        f"atoll_anyio_residual_{tmp_path.name.replace('-', '_')}",
    )
    run = _pipeline_callable(module)
    worker = _worker_callable(module, "immediate_double")
    _permit_optimized_runtime(module, monkeypatch)
    monkeypatch.setenv("ATOLL_REQUIRE_OPTIMIZED", "1")

    trace = sys.gettrace()
    profile = sys.getprofile()
    try:
        sys.settrace(None)
        sys.setprofile(None)
        observed = asyncio.run(run((2, 3, 5), worker))
    finally:
        sys.settrace(trace)
        sys.setprofile(profile)

    assert observed == (4, 6, 10)
    expected_route_hits = 3
    assert vars(module)[lowering.helper_names[-1]] == expected_route_hits


def test_anyio_search_forms_every_cumulative_residual_variant() -> None:
    """Bounded search receives one honest variant for each implemented composition."""
    plan, assessment = _plan_and_assessment()
    plan_variants = cast(
        Callable[
            [Path, SourceOptimizationPlan, SourceOptimizationAssessment],
            tuple[tuple[_SourceVariantView, ...], tuple[object, ...]],
        ],
        vars(source_search)["_plan_variants"],
    )

    variants, rejections = plan_variants(FIXTURE_ROOT, plan, assessment)
    residual_variants = tuple(
        variant
        for variant in variants
        if any(
            identifier.startswith(f"{kind}:")
            for kind in RESIDUAL_STEPS
            for identifier in variant.transformation_ids
        )
    )

    assert len(variants) == len(RESIDUAL_STEPS) + 1
    assert len(residual_variants) == len(RESIDUAL_STEPS)
    assert len(rejections) == 1


def test_anyio_search_skips_rejected_residual_and_continues_composition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One unsafe residual step does not poison independently provable later steps."""
    plan, assessment = _plan_and_assessment()
    plan_variants = cast(
        Callable[
            [Path, SourceOptimizationPlan, SourceOptimizationAssessment],
            tuple[tuple[_SourceVariantView, ...], tuple[object, ...]],
        ],
        vars(source_search)["_plan_variants"],
    )
    original = lower_residual_state_machine_plan

    def reject_context_only(
        root: Path,
        source_plan: SourceOptimizationPlan,
        source_assessment: SourceOptimizationAssessment,
        enabled: tuple[SourceTransformationKind, ...],
    ) -> SourceLoweringResult:
        if enabled[-1] == "context-copy-elision":
            return SourceLoweringResult(
                plan_id=source_plan.id,
                status="unsupported",
                request=None,
                rejections=("context-copy elision is unsafe",),
                mode="residual-state-machine",
            )
        return original(root, source_plan, source_assessment, enabled)

    monkeypatch.setattr(
        source_search,
        "lower_residual_state_machine_plan",
        reject_context_only,
    )

    variants, rejections = plan_variants(FIXTURE_ROOT, plan, assessment)
    residual_steps = tuple(
        tuple(
            kind
            for kind in RESIDUAL_STEPS
            if any(identifier.startswith(f"{kind}:") for identifier in variant.transformation_ids)
        )
        for variant in variants[1:]
    )

    assert residual_steps == (
        RESIDUAL_STEPS[:1],
        RESIDUAL_STEPS[:2],
        (*RESIDUAL_STEPS[:2], RESIDUAL_STEPS[3]),
        (*RESIDUAL_STEPS[:2], *RESIDUAL_STEPS[3:]),
    )
    assert len(rejections) == EXPECTED_SKIPPED_STEP_REJECTIONS


def test_anyio_search_retains_unavailable_residual_variant_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected residual prefix remains a trial rejection without poisoning the base."""
    plan, assessment = _plan_and_assessment()
    plan_variants = cast(
        Callable[
            [Path, SourceOptimizationPlan, SourceOptimizationAssessment],
            tuple[tuple[_SourceVariantView, ...], tuple[object, ...]],
        ],
        vars(source_search)["_plan_variants"],
    )

    def reject_residual(
        _root: Path,
        source_plan: SourceOptimizationPlan,
        _assessment: SourceOptimizationAssessment,
        _enabled: tuple[SourceTransformationKind, ...],
    ) -> SourceLoweringResult:
        return SourceLoweringResult(
            plan_id=source_plan.id,
            status="unsupported",
            request=None,
            rejections=("residual prefix unavailable",),
            mode="residual-state-machine",
        )

    monkeypatch.setattr(
        source_search,
        "lower_residual_state_machine_plan",
        reject_residual,
    )

    variants, rejections = plan_variants(FIXTURE_ROOT, plan, assessment)

    assert len(variants) == 1
    assert len(rejections) == len(RESIDUAL_STEPS) + 1


def test_anyio_residual_context_elision_rejects_context_mutation() -> None:
    """Context-copy elision cannot proceed with mutation evidence in the slice."""
    plan, assessment = _plan_and_assessment()
    evidence = tuple(
        replace(item, context_mutation=("context variable mutation",))
        if item.symbol == OWNER
        else item
        for item in assessment.callable_evidence
    )
    enabled = RESIDUAL_STEPS[:3]

    lowering = lower_residual_state_machine_plan(
        FIXTURE_ROOT,
        plan,
        replace(assessment, callable_evidence=evidence),
        enabled,
    )

    assert lowering.status == "unsupported"
    assert lowering.request is None
    assert "context-independent callable evidence" in " ".join(lowering.rejections)


@pytest.mark.parametrize(
    ("needle", "replacement", "expected_rejection"),
    [
        (
            "                    self.active.pop(record.source.item_id)\n",
            "                    self.active.pop(record.source.item_id)\n"
            "                    self.active.clear()\n",
            "active mapping",
        ),
        (
            "                    self.active.pop(record.source.item_id)\n",
            "                    self.active.pop(record.source.item_id)\n"
            "                    self.active[record.source.item_id] = record.source\n",
            "active mapping",
        ),
        (
            "        del source, active\n        return ()\n",
            "        del source, active\n        self.active.clear()\n        return ()\n",
            "active mapping",
        ),
        (
            "                    if not self.active:\n                        break\n",
            "                    if record.source.item_id < 0:\n                        break\n",
            "one pop and one terminal check",
        ),
    ],
)
def test_anyio_residual_completion_accounting_rejects_extra_active_mutation(
    tmp_path: Path,
    needle: str,
    replacement: str,
    expected_rejection: str,
) -> None:
    """The local completion counter requires one exclusive active-map pop."""
    plan, assessment = _plan_and_assessment()
    copied_root = tmp_path / "mutation-project"
    shutil.copytree(FIXTURE_ROOT, copied_root)
    source_path = copied_root / SOURCE_PATH
    source = source_path.read_text(encoding="utf-8")
    changed_source = source.replace(needle, replacement)
    assert changed_source != source
    source_path.write_text(changed_source, encoding="utf-8")
    changed_plan = replace(
        plan,
        identity=replace(
            plan.identity,
            source_hashes=(
                (SOURCE_PATH, hashlib.sha256(changed_source.encode("utf-8")).hexdigest()),
            ),
        ),
    )
    enabled = RESIDUAL_STEPS[:4]

    lowering = lower_residual_state_machine_plan(
        copied_root,
        changed_plan,
        replace(assessment, plan_id=changed_plan.id),
        enabled,
    )

    assert lowering.status == "unsupported"
    assert lowering.request is None
    assert expected_rejection in " ".join(lowering.rejections)


def test_anyio_lowering_rejects_stale_capacity_and_run_arity(tmp_path: Path) -> None:
    """Planned capacity and the one-item run signature remain exact source facts."""
    plan, _ = _plan_and_assessment()
    with pytest.raises(ValueError, match="capacity does not match"):
        lower_anyio_stream_plan(FIXTURE_ROOT, replace(plan, transport_capacity=1))

    copied_root = tmp_path / "project"
    shutil.copytree(FIXTURE_ROOT, copied_root)
    source_path = copied_root / SOURCE_PATH
    source = source_path.read_text(encoding="utf-8").replace(
        "async def _run_item(self, item: WorkItem)",
        "async def _run_item(self, item: WorkItem, extra: int = 0)",
    )
    source_path.write_text(source, encoding="utf-8")
    changed = replace(
        plan,
        identity=replace(
            plan.identity,
            source_hashes=((SOURCE_PATH, hashlib.sha256(source.encode()).hexdigest()),),
        ),
    )
    with pytest.raises(ValueError, match="run coroutine must accept only"):
        lower_anyio_stream_plan(copied_root, changed)


def test_strict_anyio_fast_path_matches_baseline_and_preserves_reflection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Immediate work uses copied-context routing without rebinding the class."""
    baseline = _load_module(FIXTURE_ROOT / SOURCE_PATH, "atoll_anyio_source_baseline")
    transformed, helpers = _transformed_module(tmp_path)
    baseline_run = _pipeline_callable(baseline)
    transformed_run = _pipeline_callable(transformed)
    baseline_worker = _worker_callable(baseline, "immediate_double")
    transformed_worker = _worker_callable(transformed, "immediate_double")
    values = tuple(range(32))
    monkeypatch.setenv("ATOLL_REQUIRE_OPTIMIZED", "1")
    _permit_optimized_runtime(transformed, monkeypatch)

    expected = asyncio.run(baseline_run(values, baseline_worker))
    trace = sys.gettrace()
    profile = sys.getprofile()
    try:
        sys.settrace(None)
        sys.setprofile(None)
        observed = asyncio.run(transformed_run(values, transformed_worker))
    finally:
        sys.settrace(trace)
        sys.setprofile(profile)

    assert observed == expected
    assert vars(transformed)[helpers[-1]] == len(values)
    assert tuple(inspect.signature(transformed_run).parameters) == tuple(
        inspect.signature(baseline_run).parameters
    )
    assert (
        inspect.signature(transformed_run).return_annotation
        == inspect.signature(baseline_run).return_annotation
    )
    assert transformed_run.__annotations__ == baseline_run.__annotations__
    runner = cast(type[object], vars(transformed)["PipelineRunner"])
    assert runner.__module__ == transformed.__name__
    assert runner.__qualname__ == "PipelineRunner"


def test_active_monitoring_falls_back_before_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime monitoring preserves the original scheduler before fast-path entry."""
    transformed, helpers = _transformed_module(tmp_path)
    run = _pipeline_callable(transformed)
    worker = _worker_callable(transformed, "immediate_double")
    no_monitoring_name = next(name for name in vars(transformed) if name.endswith("_no_monitoring"))

    def monitoring_active(_sys: object) -> bool:
        return False

    monkeypatch.setattr(transformed, no_monitoring_name, monitoring_active)
    trace = sys.gettrace()
    profile = sys.getprofile()
    try:
        sys.settrace(None)
        sys.setprofile(None)
        assert asyncio.run(run((2, 3, 5), worker)) == (4, 6, 10)
        assert vars(transformed)[helpers[-1]] == 0
        monkeypatch.setenv("ATOLL_REQUIRE_OPTIMIZED", "1")
        with pytest.raises(BaseExceptionGroup, match="unhandled errors in a TaskGroup") as raised:
            asyncio.run(run((7,), worker))
    finally:
        sys.settrace(trace)
        sys.setprofile(profile)
    assert any(
        isinstance(error, RuntimeError) and "AnyIO source guards failed" in str(error)
        for error in raised.value.exceptions
    )


def test_suspending_callable_falls_back_before_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A callable containing suspension stays on the original AnyIO task path."""
    transformed, helpers = _transformed_module(tmp_path)
    run = _pipeline_callable(transformed)
    suspending = _worker_callable(transformed, "suspending_double")

    observed = asyncio.run(run((2, 3, 5), suspending))

    assert observed == (4, 6, 10)
    assert vars(transformed)[helpers[-1]] == 0
    monkeypatch.setenv("ATOLL_REQUIRE_OPTIMIZED", "1")
    with pytest.raises(BaseExceptionGroup, match="unhandled errors in a TaskGroup") as raised:
        asyncio.run(run((7,), suspending))
    assert any(
        isinstance(error, RuntimeError) and "AnyIO source guards failed" in str(error)
        for error in raised.value.exceptions
    )


def test_indirect_context_mutation_is_rejected_before_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recursive callable inspection catches a context-mutating global helper."""
    transformed, helpers = _transformed_module(tmp_path)
    run = _pipeline_callable(transformed)
    mutating = _worker_callable(transformed, "indirect_context_mutation")
    monkeypatch.setenv("ATOLL_REQUIRE_OPTIMIZED", "1")

    with pytest.raises(BaseExceptionGroup, match="unhandled errors in a TaskGroup") as raised:
        asyncio.run(run((11,), mutating))

    assert any(
        isinstance(error, RuntimeError) and "AnyIO source guards failed" in str(error)
        for error in raised.value.exceptions
    )
    assert vars(transformed)[helpers[-1]] == 0


def test_disabled_anyio_optimization_uses_original_scheduler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ATOLL_DISABLE retains the staged source implementation unchanged."""
    transformed, helpers = _transformed_module(tmp_path)
    run = _pipeline_callable(transformed)
    worker = _worker_callable(transformed, "immediate_double")
    monkeypatch.setenv("ATOLL_DISABLE", "1")

    assert asyncio.run(run((1, 4, 9), worker)) == (2, 8, 18)
    assert vars(transformed)[helpers[-1]] == 0


def test_verified_atoll_dispatcher_composes_with_source_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only an Atoll wrapper retaining the exact fallback may pass identity guards."""
    transformed, helpers = _transformed_module(tmp_path)
    run = _pipeline_callable(transformed)
    worker = _worker_callable(transformed, "immediate_double")
    runner = cast(type[object], vars(transformed)["PipelineRunner"])
    original = cast(
        Callable[..., Coroutine[object, object, list[int]]],
        vars(runner)["results"],
    )

    @functools.wraps(original)
    async def managed(self: object, request: object) -> list[int]:
        return await original(self, request)

    managed.__dict__["__atoll_python_fallback__"] = original
    managed.__dict__["__atoll_compiled_targets__"] = ()
    monkeypatch.setattr(runner, "results", managed)
    _permit_optimized_runtime(transformed, monkeypatch)

    values = (2, 3, 5)
    assert asyncio.run(run(values, worker)) == (4, 6, 10)
    assert vars(transformed)[helpers[-1]] == 0

    managed.__dict__["__atoll_compiled_targets__"] = (object(),)
    monkeypatch.setenv("ATOLL_REQUIRE_OPTIMIZED", "1")
    trace = sys.gettrace()
    profile = sys.getprofile()
    try:
        sys.settrace(None)
        sys.setprofile(None)
        assert asyncio.run(run(values, worker)) == (4, 6, 10)
    finally:
        sys.settrace(trace)
        sys.setprofile(profile)
    assert vars(transformed)[helpers[-1]] == len(values)


def test_replaced_anyio_scheduler_falls_back_before_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replaced TaskGroup scheduler cannot pass the captured identity guard."""
    transformed, helpers = _transformed_module(tmp_path)
    run = _pipeline_callable(transformed)
    worker = _worker_callable(transformed, "immediate_double")
    task_group_class = cast(
        type[object],
        next(
            value
            for name, value in vars(transformed).items()
            if name.endswith("_expected_task_group_class")
        ),
    )
    original_start = cast(
        Callable[..., None],
        next(
            value
            for name, value in vars(transformed).items()
            if name.endswith("_expected_start_soon")
        ),
    )
    scheduled = 0

    def replacement_start(
        self: object,
        function: Callable[..., object],
        *args: object,
        name: object = None,
    ) -> None:
        nonlocal scheduled
        scheduled += 1
        original_start(self, function, *args, name=name)

    monkeypatch.setattr(task_group_class, "start_soon", replacement_start)

    values = (2, 3, 5)
    assert asyncio.run(run(values, worker)) == (4, 6, 10)
    assert scheduled == len(values)
    assert vars(transformed)[helpers[-1]] == 0


def _plan_and_assessment() -> tuple[
    SourceOptimizationPlan,
    SourceOptimizationAssessment,
]:
    source = (FIXTURE_ROOT / SOURCE_PATH).read_text(encoding="utf-8")
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    identity = SourceOptimizationIdentity(
        execution_plan_id="exec-plan-generic-anyio-source",
        source_hashes=((SOURCE_PATH, source_hash),),
        topology_fingerprint="generic-anyio-private-stream-v1",
        dialect="anyio-on-asyncio",
        lowering_version="source-optimization-analysis-v1",
        python_abi="cp312",
        transformation_versions=(
            ("private-transport-batch-drain", "batch-drain-v1"),
            ("quiescent-callable-execution", "quiescent-callable-v1"),
            ("local-state-machine-fusion", "state-machine-v1"),
            ("run-scoped-guard-amortization", "run-guard-v1"),
            ("transparent-quiescent-await-chain-collapse", "await-collapse-v1"),
            ("context-copy-elision", "context-elision-v1"),
            (
                "incremental-private-completion-accounting",
                "completion-accounting-v1",
            ),
            ("private-result-record-elision", "result-record-elision-v1"),
        ),
    )
    plan_id = stable_source_optimization_plan_id(identity)
    steps = tuple(
        TransformationStep(
            kind=kind,
            version=version,
            source_symbol=CONSUMER if kind == "private-transport-batch-drain" else OWNER,
            target_symbol=None,
            access_sites=(),
            semantic_boundary="private stream, copied context, and fallback before entry",
            description="Exercise one cumulative generic AnyIO source fast path.",
        )
        for kind, version in identity.transformation_versions
    )
    plan = SourceOptimizationPlan(
        id=plan_id,
        identity=identity,
        source=SOURCE_PATH,
        owner=OWNER,
        worker=WORKER,
        consumer=CONSUMER,
        reducer=None,
        transport="self.send_stream|self.receive_stream",
        access_sites=(),
        entrypoint=OWNER,
        steps=steps,
        semantic_boundaries=("FIFO", "copied Context", "fallback before entry"),
    )
    evidence = (
        SourceCallableEvidence(
            symbol=OWNER,
            static_role="owner",
            observed_invocations=20_000,
            completed_calls=20_000,
            immediate_result_ratio=1.0,
            hot_share=0.4,
        ),
        SourceCallableEvidence(
            symbol=WORKER,
            static_role="producer",
            observed_invocations=20_000,
            completed_calls=20_000,
            static_suspension_points=2,
            observed_suspensions=40_000,
            immediate_result_ratio=0.0,
            hot_share=0.4,
            cancellation=("CancelScope",),
        ),
    )
    assessment = SourceOptimizationAssessment(
        plan_id=plan_id,
        status="trial-ready",
        minimum_speedup=3.0,
        work_items=(WORKER,),
        observed_work_items=20_000,
        immediate_result_ratio=0.0,
        attributed_hot_share=0.8,
        scheduler_overhead_samples=10_000,
        scheduler_overhead_share=0.5,
        scheduler_overhead_evidence=("AnyIO task scheduling",),
        callable_evidence=evidence,
    )
    return plan, assessment


def _transformed_module(tmp_path: Path) -> tuple[ModuleType, tuple[str, ...]]:
    plan, assessment = _plan_and_assessment()
    lowering = lower_state_machine_plan(FIXTURE_ROOT, plan, assessment)
    assert lowering.request is not None
    patch = build_source_transformation_patch(FIXTURE_ROOT, (lowering.request,))
    copied_root = tmp_path / "project"
    shutil.copytree(FIXTURE_ROOT, copied_root)
    materialize_transformed_files(FIXTURE_ROOT, copied_root, patch)
    module = _load_module(
        copied_root / SOURCE_PATH,
        f"atoll_anyio_source_transformed_{tmp_path.name.replace('-', '_')}",
    )
    return module, lowering.helper_names


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load AnyIO source fixture: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _pipeline_callable(
    module: ModuleType,
) -> Callable[
    [Sequence[int], Callable[[int], Awaitable[int]]],
    Coroutine[object, object, tuple[int, ...]],
]:
    return cast(
        Callable[
            [Sequence[int], Callable[[int], Awaitable[int]]],
            Coroutine[object, object, tuple[int, ...]],
        ],
        vars(module)["run_pipeline"],
    )


def _worker_callable(
    module: ModuleType,
    name: str,
) -> Callable[[int], Awaitable[int]]:
    return cast(Callable[[int], Awaitable[int]], vars(module)[name])


def _permit_optimized_runtime(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ignore only the test runner's monitoring while exercising strict routing."""
    no_monitoring_name = next(name for name in vars(module) if name.endswith("_no_monitoring"))

    def monitoring_disabled(_sys: object) -> bool:
        return True

    monkeypatch.setattr(module, no_monitoring_name, monitoring_disabled)
