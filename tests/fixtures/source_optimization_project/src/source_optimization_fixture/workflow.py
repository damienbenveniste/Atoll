"""Generic asyncio workflows for source-optimization acceptance.

The hot path is a deliberately small fan-out/fan-in pipeline. It owns queue
construction, task creation, and reduction in one coroutine so source
optimization can prove that the owner task is the only consumer of queued
records. Cold helpers exercise semantics that should stay outside that hot
optimization: suspension, cancellation, task identity, controlled exceptions,
public async iteration, and indirect context mutation.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextvars import ContextVar
from typing import Literal, Never, TypedDict

MATRIX_REPETITIONS = 24
WORK_ITEM_COUNT = 256

_WORKER_CONTEXT: ContextVar[str] = ContextVar("_WORKER_CONTEXT", default="parent")
_UNSUPPORTED_CONTEXT: ContextVar[str] = ContextVar("_UNSUPPORTED_CONTEXT", default="stable")


class WorkItem(TypedDict):
    """Input record scheduled by the hot fan-out pipeline.

    Attributes:
        ordinal: Stable zero-based input position.
        label: Stable input label used to verify order.
        weight: Deterministic numeric contribution to the checksum.
    """

    ordinal: int
    label: str
    weight: int


class WorkerRecord(TypedDict):
    """Queue payload emitted by one immediate worker.

    Attributes:
        ordinal: Stable zero-based input position.
        label: Stable input label copied from the work item.
        value: Deterministic value reduced by the owner task.
        context_value: Context value observed after helper-mediated mutation.
    """

    ordinal: int
    label: str
    value: int
    context_value: str


class ReductionResult(TypedDict):
    """Owner-side reduction over records from the hot pipeline.

    Attributes:
        count: Number of records reduced by the owner.
        checksum: Stable checksum derived from all worker values.
        ordered_labels: Labels sorted back into input order.
        worker_contexts: Context values observed in child worker tasks.
    """

    count: int
    checksum: int
    ordered_labels: tuple[str, ...]
    worker_contexts: tuple[str, ...]


class ContextIsolation(TypedDict):
    """Context observations used to prove task-local mutation behavior.

    Attributes:
        parent: Value visible to the parent task after child work completes.
        child: Value visible inside the mutating child task.
        sibling: Value visible to a sibling task that never mutates the context.
        worker_parent: Value visible to the hot-pipeline owner after workers.
        worker_child: Representative value visible inside immediate workers.
    """

    parent: str
    child: str
    sibling: str
    worker_parent: str
    worker_child: str


class ExceptionSummary(TypedDict):
    """Stable facts from a controlled exception path.

    Attributes:
        type: Exception class name without module details.
        message: Deterministic exception message.
    """

    type: str
    message: str


class CancellationSummary(TypedDict):
    """Stable cleanup facts from the cancellation path.

    Attributes:
        cancelled: Whether the task surfaced `asyncio.CancelledError`.
        cleanup_count: Number of cleanup callbacks observed.
        cleanup_marker: Stable marker appended by the cancelled worker.
    """

    cancelled: bool
    cleanup_count: int
    cleanup_marker: str


class TaskIntrospectionSummary(TypedDict):
    """Stable task identity facts from the introspection path.

    Attributes:
        current_task_seen: Whether `asyncio.current_task()` returned a task.
        task_name: Deterministic name assigned to the introspected task.
    """

    current_task_seen: bool
    task_name: str


class IteratorSummary(TypedDict):
    """Stable observations from public async iteration.

    Attributes:
        values: Values yielded by the public iterator.
        snapshots: Incremental totals observed after each yielded value.
        final_total: Final total retained by the iterator object.
    """

    values: tuple[int, ...]
    snapshots: tuple[int, ...]
    final_total: int


class SemanticSnapshot(TypedDict):
    """Canonical behavior summary using only stable primitive containers.

    Attributes:
        workflow: Name of the accepted hot-path shape.
        work_count: Number of logical work items in one hot run.
        queue_capacity: Capacity passed to the owner-created queue.
        result_count: Number of reduced records.
        checksum: Stable reduction checksum.
        first_label: First input label after owner-side ordering.
        last_label: Last input label after owner-side ordering.
        parent_context: Parent context after child and worker mutation.
        child_context: Context observed in a mutating child task.
        sibling_context: Context observed in a sibling task.
        worker_context: Representative worker-local context mutation.
        exception_type: Stable controlled exception class name.
        exception_message: Stable controlled exception message.
        cancellation_cancelled: Whether cancellation propagated.
        cancellation_cleanup_count: Number of cleanup actions observed.
        cancellation_cleanup_marker: Stable cleanup marker.
        introspection_current_task_seen: Whether task identity was observable.
        introspection_task_name: Deterministic task name.
        iterator_values: Values yielded by the public async iterator.
        iterator_snapshots: Incremental totals from iterator inspection.
        iterator_final_total: Final public iterator total.
        unsupported_context_parent: Parent value after unsupported mutation path.
        unsupported_context_child: Child value from unsupported mutation path.
    """

    workflow: str
    work_count: int
    queue_capacity: int
    result_count: int
    checksum: int
    first_label: str
    last_label: str
    parent_context: str
    child_context: str
    sibling_context: str
    worker_context: str
    exception_type: str
    exception_message: str
    cancellation_cancelled: bool
    cancellation_cleanup_count: int
    cancellation_cleanup_marker: str
    introspection_current_task_seen: bool
    introspection_task_name: str
    iterator_values: tuple[int, ...]
    iterator_snapshots: tuple[int, ...]
    iterator_final_total: int
    unsupported_context_parent: str
    unsupported_context_child: str


class ControlledWorkflowError(RuntimeError):
    """Exception raised by the deterministic cold failure path."""


class IncrementalInspector:
    """Public async iterator exposing incremental inspection state.

    The iterator yields caller-provided values after a semantic suspension and
    stores each running total for later inspection. It is intentionally public
    because source optimization must preserve user-visible async iteration
    behavior outside the private hot pipeline.

    Attributes:
        values: Values yielded in their original order.
        snapshots: Running totals observed after each yielded value.
    """

    def __init__(self, values: tuple[int, ...]) -> None:
        """Initialize the iterator with stable values.

        Args:
            values: Values yielded one at a time by the async iterator.
        """

        self.values = values
        self.snapshots: list[int] = []
        self._index = 0
        self._total = 0

    def __aiter__(self) -> AsyncIterator[int]:
        """Return the iterator itself for `async for` consumption.

        Returns:
            AsyncIterator[int]: The public asynchronous iterator.
        """

        return self

    async def __anext__(self) -> int:
        """Yield the next value after a semantic suspension.

        Returns:
            int: Next value in the configured sequence.

        Raises:
            StopAsyncIteration: Raised after every value has been yielded.
        """

        if self._index >= len(self.values):
            raise StopAsyncIteration
        await asyncio.sleep(0)
        value = self.values[self._index]
        self._index += 1
        self._total += value
        self.snapshots.append(self._total)
        return value

    @property
    def total(self) -> int:
        """Return the current inspected total.

        Returns:
            int: Running total after consumed values.
        """

        return self._total


_WORK_ITEMS: tuple[WorkItem, ...] = tuple(
    WorkItem(ordinal=index, label=f"work-{index:04d}", weight=(index % 17) + 1)
    for index in range(WORK_ITEM_COUNT)
)


async def repeat_baseline_semantic_matrix() -> tuple[SemanticSnapshot, ...]:
    """Run the canonical semantic matrix repeatedly.

    Returns:
        tuple[SemanticSnapshot, ...]: Identical primitive snapshots used by
        acceptance tests to catch nondeterministic behavior.
    """

    return tuple([await canonical_semantic_snapshot() for _ in range(MATRIX_REPETITIONS)])


async def canonical_semantic_snapshot() -> SemanticSnapshot:
    """Build the canonical primitive output for source optimization.

    Returns:
        SemanticSnapshot: Stable facts covering hot-path results and cold
        semantic boundaries.
    """

    reduction = await _run_hot_private_pipeline()
    context = await _context_isolation_probe(reduction)
    exception = await cold_controlled_exception_path()
    cancellation = await cold_cancellation_cleanup_path()
    introspection = await cold_task_introspection_path()
    iterator = await public_incremental_inspection()
    unsupported_context = await cold_unsupported_indirect_context_mutation()
    return SemanticSnapshot(
        workflow="private-taskgroup-queue-reduction",
        work_count=WORK_ITEM_COUNT,
        queue_capacity=WORK_ITEM_COUNT,
        result_count=reduction["count"],
        checksum=reduction["checksum"],
        first_label=reduction["ordered_labels"][0],
        last_label=reduction["ordered_labels"][-1],
        parent_context=context["parent"],
        child_context=context["child"],
        sibling_context=context["sibling"],
        worker_context=context["worker_child"],
        exception_type=exception["type"],
        exception_message=exception["message"],
        cancellation_cancelled=cancellation["cancelled"],
        cancellation_cleanup_count=cancellation["cleanup_count"],
        cancellation_cleanup_marker=cancellation["cleanup_marker"],
        introspection_current_task_seen=introspection["current_task_seen"],
        introspection_task_name=introspection["task_name"],
        iterator_values=iterator["values"],
        iterator_snapshots=iterator["snapshots"],
        iterator_final_total=iterator["final_total"],
        unsupported_context_parent=unsupported_context["parent"],
        unsupported_context_child=unsupported_context["child"],
    )


async def benchmark_checksum(iterations: int) -> int:
    """Run the hot private pipeline enough times for benchmark measurement.

    Args:
        iterations: Number of hot pipeline executions to perform.

    Returns:
        int: Stable checksum accumulated across all iterations.
    """

    checksum = 0
    for _ in range(iterations):
        checksum += (await _run_hot_private_pipeline())["checksum"]
    return checksum


async def public_incremental_inspection() -> IteratorSummary:
    """Exercise public async iteration and incremental state inspection.

    Returns:
        IteratorSummary: Values yielded, snapshots observed during iteration,
        and the final retained total.
    """

    inspector = IncrementalInspector((2, 3, 5, 8))
    values = [value async for value in inspector]
    return IteratorSummary(
        values=tuple(values),
        snapshots=tuple(inspector.snapshots),
        final_total=inspector.total,
    )


async def cold_suspending_worker(queue: asyncio.Queue[WorkerRecord], item: WorkItem) -> None:
    """Unsupported worker because it suspends before publishing.

    Args:
        queue: Queue receiving the delayed record.
        item: Work item converted into a delayed queue record.
    """

    await asyncio.sleep(0)
    queue.put_nowait(_make_record(item))


async def cold_controlled_exception_path() -> ExceptionSummary:
    """Capture deterministic exception type and message.

    Returns:
        ExceptionSummary: Stable class name and message from the controlled
        exception path.
    """

    try:
        _raise_controlled_exception()
    except ControlledWorkflowError as exc:
        return ExceptionSummary(type=type(exc).__name__, message=str(exc))


async def cold_cancellation_cleanup_path() -> CancellationSummary:
    """Verify cancellation cleanup without timing assumptions.

    Returns:
        CancellationSummary: Stable facts proving that cancellation propagated
        and the worker cleanup callback ran once.
    """

    cleanup: list[str] = []
    task = asyncio.create_task(_cancelled_worker(cleanup))
    await asyncio.sleep(0)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        return CancellationSummary(
            cancelled=True,
            cleanup_count=len(cleanup),
            cleanup_marker=cleanup[0],
        )
    return CancellationSummary(
        cancelled=False,
        cleanup_count=len(cleanup),
        cleanup_marker="missing",
    )


async def cold_task_introspection_path() -> TaskIntrospectionSummary:
    """Expose task identity as an unsupported optimization boundary.

    Returns:
        TaskIntrospectionSummary: Stable facts from `asyncio.current_task()`.
    """

    task = asyncio.current_task()
    if task is None:
        return TaskIntrospectionSummary(current_task_seen=False, task_name="missing")
    task.set_name("source-optimization-introspection")
    return TaskIntrospectionSummary(current_task_seen=True, task_name=task.get_name())


async def cold_unsupported_indirect_context_mutation() -> dict[Literal["parent", "child"], str]:
    """Mutate a ContextVar through an unsupported indirect call path.

    Returns:
        dict[Literal["parent", "child"], str]: Parent and child context values
        proving that the indirect mutation stays task-local.
    """

    _UNSUPPORTED_CONTEXT.set("outer")

    async def child() -> str:
        mutator = _unsupported_context_mutator()
        return mutator("unsupported-child")

    child_value = await asyncio.create_task(child())
    return {"parent": _UNSUPPORTED_CONTEXT.get(), "child": child_value}


async def _run_hot_private_pipeline() -> ReductionResult:
    items = _WORK_ITEMS
    queue: asyncio.Queue[WorkerRecord] = asyncio.Queue(maxsize=len(items))
    async with asyncio.TaskGroup() as group:
        for item in items:
            group.create_task(_immediate_worker(queue, item))
        records = [await queue.get() for _ in range(len(items))]
    ordered = tuple(sorted(records, key=lambda record: record["ordinal"]))
    return ReductionResult(
        count=len(ordered),
        checksum=sum(record["value"] for record in ordered),
        ordered_labels=tuple(record["label"] for record in ordered),
        worker_contexts=tuple(record["context_value"] for record in ordered),
    )


async def _immediate_worker(queue: asyncio.Queue[WorkerRecord], item: WorkItem) -> None:
    _WORKER_CONTEXT.set(f"worker:{item['label']}")
    queue.put_nowait(_make_record(item))


def _make_record(item: WorkItem) -> WorkerRecord:
    value = (item["ordinal"] + 1) * item["weight"]
    return WorkerRecord(
        ordinal=item["ordinal"],
        label=item["label"],
        value=value,
        context_value=_WORKER_CONTEXT.get(),
    )


async def _context_isolation_probe(reduction: ReductionResult) -> ContextIsolation:
    _WORKER_CONTEXT.set("parent")

    async def child() -> str:
        _WORKER_CONTEXT.set("worker:child")
        return _WORKER_CONTEXT.get()

    async def sibling() -> str:
        return _WORKER_CONTEXT.get()

    child_result, sibling_result = await asyncio.gather(child(), sibling())
    return ContextIsolation(
        parent=_WORKER_CONTEXT.get(),
        child=child_result,
        sibling=sibling_result,
        worker_parent=_WORKER_CONTEXT.get(),
        worker_child=reduction["worker_contexts"][0],
    )


def _raise_controlled_exception() -> Never:
    raise ControlledWorkflowError("controlled failure: source-optimization")


async def _cancelled_worker(cleanup: list[str]) -> None:
    try:
        await asyncio.Future[None]()
    finally:
        cleanup.append("cleanup-complete")


def _unsupported_context_mutator() -> _ContextMutator:
    return _ContextMutator()


class _ContextMutator:
    def __call__(self, label: str) -> str:
        _UNSUPPORTED_CONTEXT.set(label)
        return _UNSUPPORTED_CONTEXT.get()
