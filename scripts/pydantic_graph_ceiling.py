"""Support a disposable Pydantic Graph optimization-ceiling experiment.

This module owns benchmark-only source transformations and measurement. It
never edits a target checkout: each transformation is applied to a copied
payload under the requested evidence directory. The experiment separates
unsafe algorithmic headroom from a copied-context scheduler optimization and
requires both performance thresholds before product implementation proceeds.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Literal, TypedDict, cast

from scripts.run_pydantic_graph_benchmark import PYDANTIC_AI_REVISION, source_manifest

ExperimentArm = Literal[
    "baseline",
    "reflection",
    "buffered",
    "immediate",
    "batch_drain",
    "unsafe_ceiling",
    "guarded_fusion",
]
ARMS: tuple[ExperimentArm, ...] = (
    "baseline",
    "reflection",
    "buffered",
    "immediate",
    "batch_drain",
    "unsafe_ceiling",
    "guarded_fusion",
)
MINIMUM_HEADROOM = 3.30
MINIMUM_GUARDED_SPEEDUP = 3.00


class CeilingExperimentError(RuntimeError):
    """Raised when the experiment cannot produce trustworthy evidence."""


@dataclass(frozen=True, slots=True)
class CeilingExperimentOptions:
    """Inputs and measurement policy for one disposable ceiling experiment.

    Attributes:
        checkout: Existing pinned Pydantic AI checkout, treated as read-only.
        evidence_root: Destination for copied payloads, logs, and reports.
        workload: Stable benchmark program executed for each arm.
        semantic_probe: Short correctness-smoke program executed before timing.
        python: Interpreter from an environment containing target dependencies.
        warmups: Number of unmeasured rotating arm groups.
        samples: Number of measured rotating arm groups.
        minimum_headroom: Ratio required to recommend implementation work.
        minimum_guarded_speedup: Ratio required from the context-safe fused arm.
    """

    checkout: Path
    evidence_root: Path
    workload: Path
    semantic_probe: Path
    python: Path
    warmups: int = 1
    samples: int = 7
    minimum_headroom: float = MINIMUM_HEADROOM
    minimum_guarded_speedup: float = MINIMUM_GUARDED_SPEEDUP

    def __post_init__(self) -> None:
        """Reject invalid timing policy before creating evidence state."""
        if self.warmups < 0:
            raise ValueError("warmups must be non-negative")
        if self.samples < 1:
            raise ValueError("samples must be positive")
        if self.minimum_headroom <= 1:
            raise ValueError("minimum_headroom must be greater than 1")
        if self.minimum_guarded_speedup <= 1:
            raise ValueError("minimum_guarded_speedup must be greater than 1")


@dataclass(frozen=True, slots=True)
class CommandEvidence:
    """One subprocess invocation retained as experiment evidence."""

    arm: ExperimentArm
    phase: str
    returncode: int
    duration_seconds: float
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class ArmSummary:
    """Correctness-smoke status and measured median for one experiment arm."""

    arm: ExperimentArm
    probe_passed: bool
    semantic_status: str
    context_isolated: bool | None
    optimized_route: bool | None
    sample_seconds: tuple[float, ...]
    median_seconds: float
    speedup_over_baseline: float


@dataclass(frozen=True, slots=True)
class CeilingExperimentResult:
    """Retained result of an optimization-ceiling experiment."""

    revision: str
    source_unchanged: bool
    summaries: tuple[ArmSummary, ...]
    commands: tuple[CommandEvidence, ...]
    minimum_headroom: float
    minimum_guarded_speedup: float
    observed_headroom: float
    guarded_speedup: float
    promising_research_direction: bool
    report_json: Path
    report_markdown: Path


class _ArmJson(TypedDict):
    arm: str
    median_seconds: float
    sample_seconds: list[float]
    probe_passed: bool
    semantic_status: str
    context_isolated: bool | None
    optimized_route: bool | None
    speedup_over_baseline: float


class _CommandJson(TypedDict):
    arm: str
    duration_seconds: float
    phase: str
    returncode: int
    stderr: str
    stdout: str


def run_ceiling_experiment(options: CeilingExperimentOptions) -> CeilingExperimentResult:
    """Stage source copies, validate every arm, and measure alternating runs.

    Args:
        options: Checkout, evidence, interpreter, and sampling policy.

    Returns:
        CeilingExperimentResult: Reports and normalized speedup evidence.

    Raises:
        CeilingExperimentError: If source anchors drift, an arm fails, or the
            checkout is not the pinned revision.
    """
    package_root = options.checkout.resolve() / "pydantic_graph" / "pydantic_graph"
    _validate_inputs(options, package_root)
    evidence_root = options.evidence_root.resolve()
    if evidence_root.exists():
        raise CeilingExperimentError(f"evidence root already exists: {evidence_root}")
    evidence_root.mkdir(parents=True)
    before = source_manifest(package_root)
    payloads = _stage_payloads(package_root, evidence_root / "payloads")
    commands, semantic_status = _run_semantic_probes(options, payloads)
    samples = _run_measurements(options, payloads, commands)
    result = _summarize(
        options,
        before == source_manifest(package_root),
        semantic_status,
        samples,
        commands,
    )
    _write_reports(result)
    return result


def apply_reflection_hoist(source: str) -> str:
    """Cache reducer arity while guarding private reducer replacement.

    Args:
        source: Original ``pydantic_graph.join`` source text.

    Returns:
        str: Transformed source for a disposable payload.

    Raises:
        CeilingExperimentError: If pinned source anchors changed.
    """
    updated = _replace_once(
        source,
        "        self._reducer = reducer\n        self._initial_factory = initial_factory\n",
        "        self._reducer = reducer\n"
        "        self._atoll_reducer_identity = None\n"
        "        self._atoll_reducer_code = None\n"
        "        self._atoll_reducer_parameter_count = None\n"
        "        self._initial_factory = initial_factory\n",
        "Join initializer",
    )
    return _replace_once(
        updated,
        "        n_parameters = len(inspect.signature(self.reducer).parameters)\n"
        "        if n_parameters == 2:\n"
        "            return cast(PlainReducerFunction[InputT, OutputT], "
        "self.reducer)(current, inputs)\n"
        "        else:\n"
        "            return cast(ContextReducerFunction[StateT, DepsT, InputT, OutputT], "
        "self.reducer)(ctx, current, inputs)\n",
        "        reducer = self.reducer\n"
        "        code = getattr(reducer, '__code__', None)\n"
        "        cacheable = (\n"
        "            inspect.isfunction(reducer)\n"
        "            and not hasattr(reducer, '__signature__')\n"
        "            and not hasattr(reducer, '__wrapped__')\n"
        "        )\n"
        "        if (\n"
        "            cacheable\n"
        "            and reducer is self._atoll_reducer_identity\n"
        "            and code is self._atoll_reducer_code\n"
        "            and self._atoll_reducer_parameter_count is not None\n"
        "        ):\n"
        "            n_parameters = self._atoll_reducer_parameter_count\n"
        "        else:\n"
        "            n_parameters = len(inspect.signature(reducer).parameters)\n"
        "            self._atoll_reducer_identity = reducer if cacheable else None\n"
        "            self._atoll_reducer_code = code if cacheable else None\n"
        "            self._atoll_reducer_parameter_count = "
        "n_parameters if cacheable else None\n"
        "        if n_parameters == 2:\n"
        "            return cast(PlainReducerFunction[InputT, OutputT], reducer)(current, inputs)\n"
        "        return cast(ContextReducerFunction[StateT, DepsT, InputT, OutputT], "
        "reducer)(ctx, current, inputs)\n",
        "Join.reduce",
    )


def apply_result_buffering(source: str) -> str:
    """Remove result-stream backpressure in an explicitly unsafe copied payload.

    Args:
        source: Original ``pydantic_graph.graph_builder`` source text.

    Returns:
        str: Transformed source for the buffering-only ceiling arm.

    Raises:
        CeilingExperimentError: If pinned source anchors changed.
    """
    return _replace_once(
        source,
        "        self.iter_stream_sender, self.iter_stream_receiver = "
        "create_memory_object_stream[_GraphTaskResult]()\n"
        "        self._next_node_run_id = 1\n",
        "        self.iter_stream_sender, self.iter_stream_receiver = "
        "create_memory_object_stream[_GraphTaskResult](\n"
        "            sys.maxsize\n"
        "        )\n"
        "        self._next_node_run_id = 1\n",
        "GraphIterator result buffering",
    )


def apply_immediate_batching(source: str) -> str:
    """Add immediate execution to an already-buffered unsafe ceiling payload.

    Awaiting steps, forks, unknown callables, dynamic scheduling, directly
    visible context mutation, and generator bytecode retain the original task
    path. Indirect task/context effects are intentionally not claimed safe.
    Unexpected suspension is surfaced as an error and is never retried.

    Args:
        source: Buffered ``pydantic_graph.graph_builder`` source text.

    Returns:
        str: Transformed source for a disposable payload.

    Raises:
        CeilingExperimentError: If pinned source anchors changed.
    """
    updated = _replace_once(
        source,
        "import inspect\nimport sys\n",
        "import dis\nimport inspect\nimport sys\n",
        "graph_builder imports",
    )
    updated = _replace_once(
        updated,
        "        self.iter_stream_sender, self.iter_stream_receiver = "
        "create_memory_object_stream[_GraphTaskResult](\n"
        "            sys.maxsize\n"
        "        )\n"
        "        self._next_node_run_id = 1\n",
        "        self.iter_stream_sender, self.iter_stream_receiver = "
        "create_memory_object_stream[_GraphTaskResult](\n"
        "            sys.maxsize\n"
        "        )\n"
        "        self._atoll_immediate_nodes = {}\n"
        "        self._next_node_run_id = 1\n",
        "GraphIterator immediate-node cache",
    )
    return _replace_once(
        updated,
        _ORIGINAL_EXECUTION_ROUTING,
        _BATCHED_EXECUTION_ROUTING,
        "GraphIterator execution routing",
    )


def apply_batch_drain(source: str) -> str:
    """Drain immediately available private results before awaiting again.

    This benchmark-only transform estimates the value of removing one AnyIO
    fairness checkpoint per already-buffered result. It is deliberately layered
    on top of immediate execution and does not establish context isolation.

    Args:
        source: Immediate-execution ``graph_builder`` source text.

    Returns:
        str: Source with a nonblocking receiver drain.

    Raises:
        CeilingExperimentError: If pinned import or consumer-loop anchors drift.
    """
    updated = _replace_once(
        source,
        "from anyio import BrokenResourceError, CancelScope, ClosedResourceError, "
        "create_memory_object_stream, create_task_group\n",
        "from anyio import BrokenResourceError, CancelScope, ClosedResourceError, "
        "WouldBlock, create_memory_object_stream, create_task_group\n",
        "GraphIterator AnyIO imports",
    )
    return _replace_once(
        updated,
        "                        async for task_result in "
        "self.iter_stream_receiver:  # pragma: no branch\n",
        "                        while True:  # pragma: no branch\n"
        "                            try:\n"
        "                                task_result = "
        "self.iter_stream_receiver.receive_nowait()\n"
        "                            except WouldBlock:\n"
        "                                task_result = await "
        "self.iter_stream_receiver.receive()\n",
        "GraphIterator result drain",
    )


def apply_context_isolated_immediate(source: str) -> str:
    """Drive immediate graph tasks inside one copied context per logical task.

    Args:
        source: Immediate-execution ``graph_builder`` source text.

    Returns:
        str: Source with copied-context execution and route evidence.

    Raises:
        CeilingExperimentError: If pinned imports, type variables, or immediate
            execution anchors drift.
    """
    updated = _replace_once(
        source,
        "import dis\nimport inspect\n",
        "import contextvars\nimport dis\nimport inspect\n",
        "graph_builder context imports",
    )
    updated = _replace_once(
        updated,
        "T = TypeVar('T', infer_variance=True)\n\n\n# === Graph runner ===\n",
        "T = TypeVar('T', infer_variance=True)\n\n"
        "_ATOLL_FAST_HITS = 0\n\n\n"
        "# === Graph runner ===\n",
        "graph_builder immediate route counter",
    )
    return _replace_once(
        updated,
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
        "            pass\n",
        "    def _atoll_run_immediately(self, task: GraphTask) -> None:\n"
        "        global _ATOLL_FAST_HITS\n"
        "        child_context = contextvars.copy_context()\n"
        "        coroutine = child_context.run(self._run_task, task)\n"
        "        try:\n"
        "            child_context.run(coroutine.send, None)\n"
        "        except StopIteration as completed:\n"
        "            result = _GraphTaskResult(task, completed.value)\n"
        "        except BaseException as exc:\n"
        "            result = _GraphTaskResult(task, [], error=exc)\n"
        "        else:\n"
        "            child_context.run(coroutine.close)\n"
        "            result = _GraphTaskResult(\n"
        "                task, [], error=RuntimeError('immediate graph task suspended "
        "unexpectedly')\n"
        "            )\n"
        "        _ATOLL_FAST_HITS += 1\n"
        "        try:\n"
        "            self.iter_stream_sender.send_nowait(result)\n"
        "        except (BrokenResourceError, ClosedResourceError):\n"
        "            pass\n",
        "GraphIterator copied-context immediate execution",
    )


def apply_lazy_reducer_scan(source: str) -> str:
    """Avoid snapshotting active tasks when no reducer can complete.

    Args:
        source: Batch-draining ``graph_builder`` source text.

    Returns:
        str: Source that performs the task snapshot only for active reducers.

    Raises:
        CeilingExperimentError: If the pinned reducer-scan block drifts.
    """
    return _replace_once(
        source,
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
        "                                join_tasks.extend(new_tasks)\n",
        "                            join_tasks: list[GraphTask] = []\n"
        "                            if self.active_reducers:\n"
        "                                tasks_by_id_values = list(self.active_tasks.values())\n"
        "                                for join_id, fork_run_id in "
        "self._get_completed_fork_runs(\n"
        "                                    task_result.source, tasks_by_id_values\n"
        "                                ):\n"
        "                                    join_state = self.active_reducers.pop(\n"
        "                                        (join_id, fork_run_id)\n"
        "                                    )\n"
        "                                    join_node = self.graph.nodes[join_id]\n"
        "                                    assert isinstance(join_node, Join), (\n"
        "                                        f'Expected a `Join` but got {join_node}'\n"
        "                                    )\n"
        "                                    new_tasks = self._handle_non_fork_edges(\n"
        "                                        join_node,\n"
        "                                        join_state.current,\n"
        "                                        join_state.downstream_fork_stack,\n"
        "                                    )\n"
        "                                    join_tasks.extend(new_tasks)\n",
        "GraphIterator lazy reducer scan",
    )


def apply_fused_map_reduce(source: str, *, copied_context: bool) -> str:
    """Add a guarded run-to-completion fast path for one proven graph shape.

    The transform is benchmark-specific research evidence. The guarded arm
    executes each quiescent step in a copied context, while the unsafe ceiling
    deliberately omits that isolation to measure maximum headroom.

    Args:
        source: Original ``graph_builder`` source text.
        copied_context: Whether every logical step receives an isolated context.

    Returns:
        str: Source containing the disposable fused-state-machine prototype.

    Raises:
        CeilingExperimentError: If the pinned imports, type variables, or
            ``Graph.run`` entrypoint drift.
    """
    updated = _replace_once(
        source,
        "import inspect\nimport sys\n",
        "import asyncio\nimport contextvars\nimport dis\nimport inspect\nimport sys\n",
        "graph_builder imports for fused execution",
    )
    updated = _replace_once(
        updated,
        "T = TypeVar('T', infer_variance=True)\n\n\n# === Graph runner ===\n",
        "T = TypeVar('T', infer_variance=True)\n\n"
        + _fused_map_reduce_support(copied_context=copied_context)
        + "\n\n# === Graph runner ===\n",
        "graph_builder fused support insertion",
    )
    return _replace_once(
        updated,
        "        if infer_name and self.name is None:\n"
        "            inferred_name = infer_obj_name(self, depth=2)\n"
        "            if inferred_name is not None:  # pragma: no branch\n"
        "                self.name = inferred_name\n\n"
        "        async with self.iter(state=state, deps=deps, inputs=inputs, "
        "span=span, infer_name=False) as graph_run:\n",
        "        if infer_name and self.name is None:\n"
        "            inferred_name = infer_obj_name(self, depth=2)\n"
        "            if inferred_name is not None:  # pragma: no branch\n"
        "                self.name = inferred_name\n\n"
        "        fast_result = _atoll_fast_map_reduce(\n"
        "            self, state=state, deps=deps, inputs=inputs, span=span\n"
        "        )\n"
        "        if fast_result is not _ATOLL_FAST_MISS:\n"
        "            return cast(OutputT, fast_result)\n\n"
        "        async with self.iter(state=state, deps=deps, inputs=inputs, "
        "span=span, infer_name=False) as graph_run:\n",
        "Graph.run fused entrypoint",
    )


def _fused_map_reduce_support(*, copied_context: bool) -> str:
    completion = (
        "    context = contextvars.copy_context()\n"
        "    coroutine = context.run(call, argument)\n"
        "    try:\n"
        "        context.run(coroutine.send, None)\n"
        if copied_context
        else "    coroutine = call(argument)\n    try:\n        coroutine.send(None)\n"
    )
    return (
        "_ATOLL_FAST_MISS = object()\n"
        "_ATOLL_FAST_HITS = 0\n\n\n"
        "def _atoll_destination(graph: Any, source_id: NodeID) -> NodeID | None:\n"
        "    paths = graph.edges_by_source.get(source_id, ())\n"
        "    if len(paths) != 1 or len(paths[0].items) != 1:\n"
        "        return None\n"
        "    item = paths[0].items[0]\n"
        "    return item.destination_id if isinstance(item, DestinationMarker) else None\n\n\n"
        "def _atoll_quiescent(call: Any) -> bool:\n"
        "    code = getattr(call, '__code__', None)\n"
        "    if code is None or not inspect.iscoroutinefunction(call):\n"
        "        return False\n"
        "    blocked_names = {\n"
        "        'ContextVar', 'cancel', 'checkpoint', 'copy_context', 'create_task',\n"
        "        'current_task', 'ensure_future', 'get_event_loop', 'get_running_loop',\n"
        "        'set_task_factory', 'sleep', 'start_soon',\n"
        "    }\n"
        "    blocked_opcodes = {'SEND', 'YIELD_FROM', 'YIELD_VALUE'}\n"
        "    return not blocked_names.intersection(code.co_names) and not any(\n"
        "        instruction.opname in blocked_opcodes for instruction in "
        "dis.get_instructions(code)\n"
        "    )\n\n\n"
        "def _atoll_complete(call: Any, argument: Any) -> Any:\n"
        + completion
        + "    except StopIteration as completed:\n"
        "        return completed.value\n"
        "    else:\n"
        "        coroutine.close()\n"
        "        raise RuntimeError('guarded immediate graph step suspended unexpectedly')\n\n\n"
        "def _atoll_fast_map_reduce(\n"
        "    graph: Any, *, state: Any, deps: Any, inputs: Any, span: Any\n"
        ") -> Any:\n"
        "    global _ATOLL_FAST_HITS\n"
        "    loop = asyncio.get_running_loop()\n"
        "    if (\n"
        "        span is not None\n"
        "        or loop.get_task_factory() is not None\n"
        "        or loop.get_debug()\n"
        "        or sys.gettrace() is not None\n"
        "        or sys.getprofile() is not None\n"
        "        or len(graph.nodes) != 6\n"
        "    ):\n"
        "        return _ATOLL_FAST_MISS\n"
        "    generate_id = _atoll_destination(graph, StartNode.id)\n"
        "    generate = graph.nodes.get(generate_id)\n"
        "    fork_id = (\n"
        "        _atoll_destination(graph, generate_id) if generate_id is not None else None\n"
        "    )\n"
        "    fork = graph.nodes.get(fork_id)\n"
        "    transform_id = (\n"
        "        _atoll_destination(graph, fork_id) if fork_id is not None else None\n"
        "    )\n"
        "    transform = graph.nodes.get(transform_id)\n"
        "    join_id = (\n"
        "        _atoll_destination(graph, transform_id) "
        "if transform_id is not None else None\n"
        "    )\n"
        "    join = graph.nodes.get(join_id)\n"
        "    end_id = _atoll_destination(graph, join_id) if join_id is not None else None\n"
        "    end = graph.nodes.get(end_id)\n"
        "    if not (\n"
        "        isinstance(generate, Step)\n"
        "        and isinstance(fork, Fork)\n"
        "        and fork.is_map\n"
        "        and isinstance(transform, Step)\n"
        "        and isinstance(join, Join)\n"
        "        and isinstance(end, EndNode)\n"
        "        and _atoll_quiescent(generate.call)\n"
        "        and _atoll_quiescent(transform.call)\n"
        "        and inspect.isfunction(join.reducer)\n"
        "        and not hasattr(join.reducer, '__signature__')\n"
        "        and not hasattr(join.reducer, '__wrapped__')\n"
        "        and len(inspect.signature(join.reducer).parameters) == 2\n"
        "    ):\n"
        "        return _ATOLL_FAST_MISS\n"
        "    generated = _atoll_complete(\n"
        "        generate.call, StepContext(state=state, deps=deps, inputs=inputs)\n"
        "    )\n"
        "    if not _is_any_iterable(generated):\n"
        "        raise RuntimeError('guarded graph generator returned a non-iterable')\n"
        "    _ATOLL_FAST_HITS += 1\n"
        "    current = join.initial_factory()\n"
        "    reducer = join.reducer\n"
        "    for item in generated:\n"
        "        transformed = _atoll_complete(\n"
        "            transform.call, StepContext(state=state, deps=deps, inputs=item)\n"
        "        )\n"
        "        current = reducer(current, transformed)\n"
        "    return current\n"
    )


_ORIGINAL_EXECUTION_ROUTING = (
    "    def _handle_execution_request(self, request: Sequence[GraphTask]) -> None:\n"
    "        for new_task in request:\n"
    "            self.active_tasks[new_task.task_id] = new_task\n"
    "        for new_task in request:\n"
    "            self.task_group.start_soon(self._run_tracked_task, new_task)\n"
    "\n"
)

_BATCHED_EXECUTION_ROUTING = (
    "    def _handle_execution_request(self, request: Sequence[GraphTask]) -> None:\n"
    "        for new_task in request:\n"
    "            self.active_tasks[new_task.task_id] = new_task\n"
    "        for new_task in request:\n"
    "            if self._atoll_can_run_immediately(new_task):\n"
    "                self._atoll_run_immediately(new_task)\n"
    "            else:\n"
    "                self.task_group.start_soon(self._run_tracked_task, new_task)\n"
    "\n"
    "    def _atoll_can_run_immediately(self, task: GraphTask) -> bool:\n"
    "        cached = self._atoll_immediate_nodes.get(task.node_id)\n"
    "        if cached is not None:\n"
    "            return cached\n"
    "        node = self.graph.nodes[task.node_id]\n"
    "        if isinstance(node, Fork):\n"
    "            eligible = False\n"
    "        elif not isinstance(node, Step):\n"
    "            eligible = True\n"
    "        else:\n"
    "            call = node.call\n"
    "            code = getattr(call, '__code__', None)\n"
    "            blocked_names = {\n"
    "                'ContextVar', 'cancel', 'cancelled', 'checkpoint', 'copy_context',\n"
    "                'create_task', 'current_task', 'ensure_future', 'get_event_loop',\n"
    "                'get_running_loop', 'set', 'sleep', 'start_soon',\n"
    "            }\n"
    "            blocked_opcodes = {\n"
    "                'DELETE_DEREF', 'DELETE_GLOBAL', 'SEND', 'STORE_DEREF',\n"
    "                'STORE_GLOBAL', 'YIELD_FROM', 'YIELD_VALUE',\n"
    "            }\n"
    "            eligible = (\n"
    "                code is not None\n"
    "                and inspect.iscoroutinefunction(call)\n"
    "                and not blocked_names.intersection(code.co_names)\n"
    "                and not any(\n"
    "                    instruction.opname in blocked_opcodes\n"
    "                    for instruction in dis.get_instructions(code)\n"
    "                )\n"
    "            )\n"
    "        self._atoll_immediate_nodes[task.node_id] = eligible\n"
    "        return eligible\n"
    "\n"
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
    "                task, [], error=RuntimeError('immediate graph task suspended unexpectedly')\n"
    "            )\n"
    "        try:\n"
    "            self.iter_stream_sender.send_nowait(result)\n"
    "        except (BrokenResourceError, ClosedResourceError):\n"
    "            pass\n"
    "\n"
)


def _validate_inputs(options: CeilingExperimentOptions, package_root: Path) -> None:
    if not package_root.is_dir():
        raise CeilingExperimentError(f"Pydantic Graph package is unavailable: {package_root}")
    git = shutil.which("git")
    if git is None:
        raise CeilingExperimentError("git is required for revision verification")
    revision = subprocess.run(
        (git, "-C", str(options.checkout), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != PYDANTIC_AI_REVISION:
        raise CeilingExperimentError(
            f"checkout revision is {revision}; expected {PYDANTIC_AI_REVISION}"
        )
    for label, path in (
        ("workload", options.workload),
        ("correctness probe", options.semantic_probe),
        ("python", options.python),
    ):
        if not path.is_file():
            raise CeilingExperimentError(f"{label} is unavailable: {path}")


def _stage_payloads(package_root: Path, payload_root: Path) -> dict[ExperimentArm, Path]:
    payloads: dict[ExperimentArm, Path] = {}
    for arm in ARMS:
        destination = payload_root / arm
        shutil.copytree(package_root, destination / "pydantic_graph")
        payloads[arm] = destination
    for arm in ARMS[1:]:
        join_path = payloads[arm] / "pydantic_graph" / "join.py"
        join_path.write_text(
            apply_reflection_hoist(join_path.read_text(encoding="utf-8")),
            encoding="utf-8",
        )
    for arm in ("buffered", "immediate", "batch_drain"):
        builder = payloads[arm] / "pydantic_graph" / "graph_builder.py"
        builder.write_text(
            apply_result_buffering(builder.read_text(encoding="utf-8")),
            encoding="utf-8",
        )
    for arm in ("immediate", "batch_drain"):
        builder = payloads[arm] / "pydantic_graph" / "graph_builder.py"
        builder.write_text(
            apply_immediate_batching(builder.read_text(encoding="utf-8")),
            encoding="utf-8",
        )
    batch_builder = payloads["batch_drain"] / "pydantic_graph" / "graph_builder.py"
    batch_builder.write_text(
        apply_batch_drain(batch_builder.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    unsafe_builder = payloads["unsafe_ceiling"] / "pydantic_graph" / "graph_builder.py"
    unsafe_builder.write_text(
        apply_fused_map_reduce(
            unsafe_builder.read_text(encoding="utf-8"),
            copied_context=False,
        ),
        encoding="utf-8",
    )
    guarded_builder = payloads["guarded_fusion"] / "pydantic_graph" / "graph_builder.py"
    guarded_source = apply_result_buffering(guarded_builder.read_text(encoding="utf-8"))
    guarded_source = apply_immediate_batching(guarded_source)
    guarded_source = apply_batch_drain(guarded_source)
    guarded_source = apply_context_isolated_immediate(guarded_source)
    guarded_source = apply_lazy_reducer_scan(guarded_source)
    guarded_builder.write_text(guarded_source, encoding="utf-8")
    return payloads


def _run_semantic_probes(
    options: CeilingExperimentOptions,
    payloads: dict[ExperimentArm, Path],
) -> tuple[list[CommandEvidence], dict[ExperimentArm, bool]]:
    commands: list[CommandEvidence] = []
    statuses: dict[ExperimentArm, bool] = {}
    for arm in ARMS:
        print(f"Ceiling experiment: correctness probe [{arm}]")
        evidence = _run_arm_command(
            arm,
            "probe",
            options.python,
            options.semantic_probe,
            payloads[arm],
        )
        commands.append(evidence)
        statuses[arm] = evidence.returncode == 0
        _require_success(evidence)
        _validate_probe_evidence(evidence)
        print(f"Ceiling experiment: correctness probe [{arm}] passed")
    return commands, statuses


def _run_measurements(
    options: CeilingExperimentOptions,
    payloads: dict[ExperimentArm, Path],
    commands: list[CommandEvidence],
) -> dict[ExperimentArm, list[float]]:
    for group_index in range(options.warmups):
        for arm in _rotated_arms(group_index):
            print(f"Ceiling experiment: warmup {group_index + 1}/{options.warmups} [{arm}]")
            evidence = _run_arm_command(
                arm, "warmup", options.python, options.workload, payloads[arm]
            )
            commands.append(evidence)
            _require_success(evidence)
            print(f"Ceiling experiment: warmup [{arm}] {evidence.duration_seconds:.3f}s")
    samples: dict[ExperimentArm, list[float]] = {arm: [] for arm in ARMS}
    for group_index in range(options.samples):
        for arm in _rotated_arms(group_index + options.warmups):
            print(f"Ceiling experiment: sample {group_index + 1}/{options.samples} [{arm}]")
            evidence = _run_arm_command(
                arm, "sample", options.python, options.workload, payloads[arm]
            )
            commands.append(evidence)
            _require_success(evidence)
            samples[arm].append(evidence.duration_seconds)
            print(f"Ceiling experiment: sample [{arm}] {evidence.duration_seconds:.3f}s")
    return samples


def _run_arm_command(
    arm: ExperimentArm,
    phase: str,
    python: Path,
    command: Path,
    payload: Path,
) -> CommandEvidence:
    environment = {
        **os.environ,
        "ATOLL_DISABLE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": str(payload),
        "ATOLL_CEILING_ARM": arm,
        "ATOLL_CEILING_PAYLOAD": str(payload),
    }
    argv: tuple[str, ...] = (str(python), str(command))
    if phase == "probe":
        argv += ("--verify",)
    started = time.perf_counter()
    completed = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    return CommandEvidence(
        arm=arm,
        phase=phase,
        returncode=completed.returncode,
        duration_seconds=time.perf_counter() - started,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _summarize(
    options: CeilingExperimentOptions,
    source_unchanged: bool,
    semantic_status: dict[ExperimentArm, bool],
    samples: dict[ExperimentArm, list[float]],
    commands: list[CommandEvidence],
) -> CeilingExperimentResult:
    baseline_median = median(samples["baseline"])
    summaries = tuple(
        ArmSummary(
            arm=arm,
            probe_passed=semantic_status[arm],
            semantic_status=_semantic_status(arm),
            context_isolated=_context_isolated(commands, arm),
            optimized_route=_optimized_route(commands, arm),
            sample_seconds=tuple(samples[arm]),
            median_seconds=median(samples[arm]),
            speedup_over_baseline=baseline_median / median(samples[arm]),
        )
        for arm in ARMS
    )
    ceiling = next(summary for summary in summaries if summary.arm == "unsafe_ceiling")
    guarded = next(summary for summary in summaries if summary.arm == "guarded_fusion")
    report_root = options.evidence_root.resolve()
    return CeilingExperimentResult(
        revision=PYDANTIC_AI_REVISION,
        source_unchanged=source_unchanged,
        summaries=summaries,
        commands=tuple(commands),
        minimum_headroom=options.minimum_headroom,
        minimum_guarded_speedup=options.minimum_guarded_speedup,
        observed_headroom=ceiling.speedup_over_baseline,
        guarded_speedup=guarded.speedup_over_baseline,
        promising_research_direction=source_unchanged
        and all(semantic_status.values())
        and ceiling.speedup_over_baseline >= options.minimum_headroom
        and guarded.speedup_over_baseline >= options.minimum_guarded_speedup
        and guarded.context_isolated is True
        and guarded.optimized_route is True,
        report_json=report_root / "ceiling-report.json",
        report_markdown=report_root / "ceiling-report.md",
    )


def _rotated_arms(index: int) -> tuple[ExperimentArm, ...]:
    offset = index % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def _semantic_status(arm: ExperimentArm) -> str:
    if arm == "baseline":
        return "reference"
    if arm in {"reflection", "guarded_fusion"}:
        return "guarded"
    return "not-established"


def _validate_probe_evidence(evidence: CommandEvidence) -> None:
    payload = _probe_payload(evidence)
    if payload.get("arm") != evidence.arm:
        raise CeilingExperimentError(f"{evidence.arm} probe reported a different arm")
    if not isinstance(payload.get("context_isolated"), bool):
        raise CeilingExperimentError(f"{evidence.arm} probe omitted context isolation evidence")
    if not isinstance(payload.get("optimized_route"), bool):
        raise CeilingExperimentError(f"{evidence.arm} probe omitted optimized-route evidence")
    if payload.get("signature_guarded") is not True:
        raise CeilingExperimentError(f"{evidence.arm} probe did not verify signature guards")


def _context_isolated(commands: list[CommandEvidence], arm: ExperimentArm) -> bool | None:
    return _probe_boolean(commands, arm, "context_isolated")


def _optimized_route(commands: list[CommandEvidence], arm: ExperimentArm) -> bool | None:
    return _probe_boolean(commands, arm, "optimized_route")


def _probe_boolean(
    commands: list[CommandEvidence],
    arm: ExperimentArm,
    field: str,
) -> bool | None:
    evidence = next(
        (command for command in commands if command.arm == arm and command.phase == "probe"),
        None,
    )
    if evidence is None:
        return None
    value = _probe_payload(evidence).get(field)
    return value if isinstance(value, bool) else None


def _probe_payload(evidence: CommandEvidence) -> dict[str, object]:
    try:
        payload: object = json.loads(evidence.stdout)
    except json.JSONDecodeError as error:
        raise CeilingExperimentError(
            f"{evidence.arm} probe did not emit valid JSON: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise CeilingExperimentError(f"{evidence.arm} probe emitted a non-object result")
    mapping = cast(dict[object, object], payload)
    return {str(key): value for key, value in mapping.items()}


def _require_success(evidence: CommandEvidence) -> None:
    if evidence.returncode != 0:
        raise CeilingExperimentError(
            f"{evidence.arm} {evidence.phase} failed with exit code "
            f"{evidence.returncode}: {evidence.stderr.strip()}"
        )


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    occurrences = source.count(old)
    if occurrences != 1:
        raise CeilingExperimentError(
            f"{label} anchor occurred {occurrences} times; expected pinned source exactly once"
        )
    return source.replace(old, new, 1)


def _write_reports(result: CeilingExperimentResult) -> None:
    payload = {
        "commands": [_command_json(command) for command in result.commands],
        "guarded_speedup": result.guarded_speedup,
        "minimum_headroom": result.minimum_headroom,
        "minimum_guarded_speedup": result.minimum_guarded_speedup,
        "observed_headroom": result.observed_headroom,
        "promising_research_direction": result.promising_research_direction,
        "revision": result.revision,
        "source_unchanged": result.source_unchanged,
        "summaries": [_arm_json(summary) for summary in result.summaries],
    }
    result.report_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result.report_markdown.write_text(_markdown_report(result), encoding="utf-8")


def _markdown_report(result: CeilingExperimentResult) -> str:
    lines = [
        "# Pydantic Graph Optimization Ceiling",
        "",
        f"- Revision: `{result.revision}`",
        f"- Checkout sources unchanged: {'yes' if result.source_unchanged else 'no'}",
        f"- Required implementation headroom: `{result.minimum_headroom:.3f}x`",
        f"- Observed unsafe ceiling: `{result.observed_headroom:.3f}x`",
        f"- Required guarded speedup: `{result.minimum_guarded_speedup:.3f}x`",
        f"- Observed guarded speedup: `{result.guarded_speedup:.3f}x`",
        "- Guarded scheduler semantics: `established by deterministic probes`",
        "- Recommendation: "
        f"`{'investigate-guarded-design' if result.promising_research_direction else 'stop'}`",
        "",
        "| Arm | Median | Speedup | Correctness smoke | Task context isolated | "
        "Optimized route | Semantic status |",
        "| --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {summary.arm} | {summary.median_seconds:.6f}s | "
        f"{summary.speedup_over_baseline:.3f}x | "
        f"{'passed' if summary.probe_passed else 'failed'} | "
        f"{_context_label(summary.context_isolated)} | "
        f"{_context_label(summary.optimized_route)} | "
        f"{summary.semantic_status} |"
        for summary in result.summaries
    )
    lines.extend(
        (
            "",
            "The intermediate arms isolate reflection, buffering, immediate execution, and "
            "nonblocking batch draining. The `unsafe_ceiling` arm measures direct state-machine "
            "headroom without task-context isolation. The `guarded_fusion` arm uses the same "
            "run-to-completion shape with static guards and one copied context per logical step; "
            "only that arm can satisfy the semantic promotion prerequisite.",
            "",
        )
    )
    return "\n".join(lines)


def _command_json(command: CommandEvidence) -> _CommandJson:
    return {
        "arm": command.arm,
        "duration_seconds": command.duration_seconds,
        "phase": command.phase,
        "returncode": command.returncode,
        "stderr": command.stderr,
        "stdout": command.stdout,
    }


def _context_label(value: bool | None) -> str:
    if value is None:
        return "not-recorded"
    return "yes" if value else "no"


def _arm_json(summary: ArmSummary) -> _ArmJson:
    return {
        "arm": summary.arm,
        "median_seconds": summary.median_seconds,
        "sample_seconds": list(summary.sample_seconds),
        "probe_passed": summary.probe_passed,
        "semantic_status": summary.semantic_status,
        "context_isolated": summary.context_isolated,
        "optimized_route": summary.optimized_route,
        "speedup_over_baseline": summary.speedup_over_baseline,
    }
