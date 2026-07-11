"""Tests for the disposable Pydantic Graph optimization-ceiling experiment."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest
from scripts import pydantic_graph_ceiling
from scripts.pydantic_graph_ceiling import (
    CeilingExperimentError,
    apply_immediate_batching,
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
    assert rotate(0) == ("baseline", "reflection", "buffered", "unsafe_ceiling")
    assert rotate(1) == ("reflection", "buffered", "unsafe_ceiling", "baseline")
    assert rotate(2) == ("buffered", "unsafe_ceiling", "baseline", "reflection")
    assert rotate(3) == ("unsafe_ceiling", "baseline", "reflection", "buffered")
    assert rotate(4) == rotate(0)
