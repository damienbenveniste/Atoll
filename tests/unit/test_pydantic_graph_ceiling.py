"""Tests for the disposable Pydantic Graph optimization-ceiling experiment."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest
from scripts import pydantic_graph_ceiling
from scripts.pydantic_graph_ceiling import (
    ARMS,
    CeilingExperimentError,
    apply_batch_drain,
    apply_context_isolated_immediate,
    apply_fused_map_reduce,
    apply_immediate_batching,
    apply_lazy_reducer_scan,
    apply_reflection_hoist,
    apply_result_buffering,
)


def test_reflection_hoist_caches_arity_and_guards_reducer_replacement() -> None:
    source = (
        "class Join:\n"
        "    _reducer: ReducerFunction[StateT, DepsT, InputT, OutputT]\n"
        "    _initial_factory: object\n\n"
        "    def __init__(self, reducer, initial_factory):\n"
        "        self._reducer = reducer\n"
        "        self._initial_factory = initial_factory\n\n"
        "    def reduce(self, ctx, current, inputs):\n"
        "        n_parameters = len(inspect.signature(self.reducer).parameters)\n"
        "        if n_parameters == 2:\n"
        "            return cast(PlainReducerFunction[InputT, OutputT], "
        "self.reducer)(current, inputs)\n"
        "        else:\n"
        "            return cast(ContextReducerFunction[StateT, DepsT, InputT, OutputT], "
        "self.reducer)(ctx, current, inputs)\n"
    )

    transformed = apply_reflection_hoist(source)

    assert "\n    _atoll_reducer_identity:" not in transformed
    assert "self._atoll_reducer_identity = None" in transformed
    assert "self._atoll_reducer_code = None" in transformed
    assert "inspect.isfunction(reducer)" in transformed
    assert "not hasattr(reducer, '__signature__')" in transformed
    assert "not hasattr(reducer, '__wrapped__')" in transformed
    assert "and code is self._atoll_reducer_code" in transformed
    assert "n_parameters = len(inspect.signature(self.reducer).parameters)" not in transformed


def test_immediate_batching_keeps_forks_and_suspending_steps_on_task_path() -> None:
    source = (
        "import inspect\n"
        "import sys\n\n"
        "class Iterator:\n"
        "    iter_stream_receiver: MemoryObjectReceiveStream[_GraphTaskResult] = "
        "field(init=False)\n\n"
        "    def initialize(self):\n"
        "        self.iter_stream_sender, self.iter_stream_receiver = "
        "create_memory_object_stream[_GraphTaskResult]()\n"
        "        self._next_node_run_id = 1\n\n"
        "    def _handle_execution_request(self, request: Sequence[GraphTask]) -> None:\n"
        "        for new_task in request:\n"
        "            self.active_tasks[new_task.task_id] = new_task\n"
        "        for new_task in request:\n"
        "            self.task_group.start_soon(self._run_tracked_task, new_task)\n\n"
    )

    buffered = apply_result_buffering(source)
    transformed = apply_immediate_batching(buffered)

    assert "import dis" not in buffered
    assert "self._atoll_run_immediately" not in buffered
    assert transformed.startswith("import dis\nimport inspect\n")
    assert "create_memory_object_stream[_GraphTaskResult](\n            sys.maxsize" in transformed
    assert "_atoll_immediate_nodes:" not in transformed
    assert "if isinstance(node, Fork):\n            eligible = False" in transformed
    assert "'SEND'" in transformed
    assert "coroutine.close()" in transformed
    assert "self.task_group.start_soon" in transformed


def test_batch_drain_awaits_only_when_private_receiver_is_empty() -> None:
    source = (
        "from anyio import BrokenResourceError, CancelScope, ClosedResourceError, "
        "create_memory_object_stream, create_task_group\n\n"
        "async def consume(self):\n"
        "                    while active:\n"
        "                        async for task_result in "
        "self.iter_stream_receiver:  # pragma: no branch\n"
        "                            process(task_result)\n"
    )

    transformed = apply_batch_drain(source)

    assert "ClosedResourceError, WouldBlock, create_memory_object_stream" in transformed
    assert "task_result = self.iter_stream_receiver.receive_nowait()" in transformed
    assert "except WouldBlock:" in transformed
    assert "task_result = await self.iter_stream_receiver.receive()" in transformed


def test_absolute_fused_map_reduce_guards_before_entry() -> None:
    source = (
        "import inspect\n"
        "import sys\n\n"
        "T = TypeVar('T', infer_variance=True)\n\n\n"
        "# === Graph runner ===\n\n"
        "class Graph:\n"
        "    async def run(\n"
        "        self, *, state=None, deps=None, inputs=None, span=None, infer_name=True\n"
        "    ):\n"
        "        if infer_name and self.name is None:\n"
        "            inferred_name = infer_obj_name(self, depth=2)\n"
        "            if inferred_name is not None:  # pragma: no branch\n"
        "                self.name = inferred_name\n\n"
        "        async with self.iter(state=state, deps=deps, inputs=inputs, "
        "span=span, infer_name=False) as graph_run:\n"
        "            return graph_run\n"
    )

    transformed = apply_fused_map_reduce(source, copied_context=True)

    assert "context = contextvars.copy_context()" in transformed
    assert "context.run(coroutine.send, None)" in transformed
    assert "loop.get_task_factory() is not None" in transformed
    assert "sys.gettrace() is not None" in transformed
    assert "_ATOLL_FAST_HITS += 1" in transformed
    assert "if fast_result is not _ATOLL_FAST_MISS:" in transformed

    unsafe = apply_fused_map_reduce(source, copied_context=False)
    assert "context = contextvars.copy_context()" not in unsafe
    assert "coroutine.send(None)" in unsafe


def test_guarded_immediate_execution_copies_context_and_records_route() -> None:
    source = (
        "import dis\n"
        "import inspect\n"
        "import sys\n\n"
        "T = TypeVar('T', infer_variance=True)\n\n\n"
        "# === Graph runner ===\n\n"
        "class Iterator:\n"
        "    def _atoll_run_immediately(self, task: GraphTask) -> None:\n"
        "        coroutine = self._run_task(task)\n"
        "        try:\n"
        "            coroutine.send(None)\n"
        "        except StopIteration as completed:\n"
        "            result = _GraphTaskResult(task, completed.value)\n"
        "        except BaseException as exc:\n"
        "            result = _GraphTaskResult(task, [], error=exc)\n"
        "        else:\n"
        "            coroutine.close()\n"
        "            result = _GraphTaskResult(\n"
        "                task, [], error=RuntimeError('immediate graph task suspended "
        "unexpectedly')\n"
        "            )\n"
        "        try:\n"
        "            self.iter_stream_sender.send_nowait(result)\n"
        "        except (BrokenResourceError, ClosedResourceError):\n"
        "            pass\n"
    )

    transformed = apply_context_isolated_immediate(source)

    assert "import contextvars" in transformed
    assert "_ATOLL_FAST_HITS = 0" in transformed
    assert "child_context = contextvars.copy_context()" in transformed
    assert "child_context.run(coroutine.send, None)" in transformed
    assert "child_context.run(coroutine.close)" in transformed
    assert "_ATOLL_FAST_HITS += 1" in transformed


def test_lazy_reducer_scan_avoids_empty_active_task_snapshot() -> None:
    source = (
        "                            tasks_by_id_values = list(self.active_tasks.values())\n"
        "                            join_tasks: list[GraphTask] = []\n\n"
        "                            for join_id, fork_run_id in self._get_completed_fork_runs(\n"
        "                                task_result.source, tasks_by_id_values\n"
        "                            ):\n"
        "                                join_state = self.active_reducers.pop("
        "(join_id, fork_run_id))\n"
        "                                join_node = self.graph.nodes[join_id]\n"
        "                                assert isinstance(join_node, Join), "
        "f'Expected a `Join` but got {join_node}'\n"
        "                                new_tasks = self._handle_non_fork_edges(\n"
        "                                    join_node, join_state.current, "
        "join_state.downstream_fork_stack\n"
        "                                )\n"
        "                                join_tasks.extend(new_tasks)\n"
    )

    transformed = apply_lazy_reducer_scan(source)

    assert "if self.active_reducers:" in transformed
    assert transformed.index("if self.active_reducers:") < transformed.index(
        "list(self.active_tasks.values())"
    )


def test_source_replacement_rejects_drift_or_ambiguity() -> None:
    replace_once = cast(
        Callable[[str, str, str, str], str],
        vars(pydantic_graph_ceiling)["_replace_once"],
    )
    with pytest.raises(CeilingExperimentError, match="occurred 0 times"):
        replace_once("original", "missing", "new", "anchor")
    with pytest.raises(CeilingExperimentError, match="occurred 2 times"):
        replace_once("old old", "old", "new", "anchor")


def test_arm_rotation_balances_process_order() -> None:
    rotate = cast(
        Callable[[int], tuple[str, ...]],
        vars(pydantic_graph_ceiling)["_rotated_arms"],
    )
    assert rotate(0) == ARMS
    assert rotate(1) == ARMS[1:] + ARMS[:1]
    assert rotate(2) == ARMS[2:] + ARMS[:2]
    assert rotate(len(ARMS)) == rotate(0)
