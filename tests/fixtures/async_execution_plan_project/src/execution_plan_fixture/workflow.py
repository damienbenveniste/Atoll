"""Deterministic async workflows for generic execution-plan acceptance.

The supported workflow deliberately stays narrow: module-level producers either
finish immediately or raise immediately, and the owner task performs all
queue-driven coordination. The remaining helpers are cold unsupported shapes so
static scanners can prove they reject suspension, task identity, loop policy,
cancellation, context propagation, debug mode, and iterable side effects without
running those paths.
"""

from __future__ import annotations

import asyncio
import traceback
from collections.abc import Coroutine, Iterable
from contextvars import ContextVar
from typing import Literal, TypedDict

MATRIX_REPETITIONS = 32
QUEUE_CAPACITY = 4

_CONTEXT_LABEL: ContextVar[str] = ContextVar("_CONTEXT_LABEL", default="parent")


class TracebackEvidence(TypedDict):
    """Stable traceback facts for the immediate exception variant.

    Attributes:
        type: Exception class name without module or address details.
        message: Deterministic message raised by the fixture producer.
        producer_frame: Whether the traceback still contains the producer frame.
    """

    type: str
    message: str
    producer_frame: bool


class SemanticSnapshot(TypedDict):
    """Canonical behavior summary using only stable primitive values.

    Attributes:
        workflow: Name of the supported workflow shape.
        capacity: Positive queue capacity known at construction time.
        total: Deterministic reduction produced by the owner after fan-in.
        count: Number of successful producer records reduced by the owner.
        first: First reduced label.
        last: Last reduced label.
        exception_type: Stable exception class name from the immediate failure.
        exception_message: Stable exception message from the immediate failure.
        exception_frame_present: Whether stable traceback evidence names the
            immediate failure producer.
        context_parent: Context value observed by the parent after child work.
        context_child: Context value observed inside a child task.
        context_sibling: Context value observed by a sibling task.
        cold_decoy_count: Number of intentionally cold decoy functions.
    """

    workflow: str
    capacity: int
    total: int
    count: int
    first: str
    last: str
    exception_type: str
    exception_message: str
    exception_frame_present: bool
    context_parent: str
    context_child: str
    context_sibling: str
    cold_decoy_count: int


class WorkItem(TypedDict):
    """Queue payload reduced by the owner task.

    Attributes:
        label: Stable work identifier.
        value: Integer contribution to the deterministic reduction.
    """

    label: str
    value: int


class ControlledImmediateError(RuntimeError):
    """Exception raised before the producer reaches any suspension point."""


async def publish_immediate(queue: asyncio.Queue[WorkItem], item: WorkItem) -> None:
    """Publish one item synchronously from an async producer.

    Args:
        queue: Owner-created queue with a statically known positive capacity.
        item: Stable work item published for owner-side reduction.
    """

    queue.put_nowait(item)


async def fail_immediate(label: str) -> None:
    """Raise a deterministic exception before any suspension can occur.

    Args:
        label: Stable work identifier included in the exception message.

    Raises:
        ControlledImmediateError: Always raised before the coroutine awaits.
    """

    raise ControlledImmediateError(f"controlled:{label}")


async def run_supported_workflow() -> tuple[tuple[WorkItem, ...], TracebackEvidence]:
    """Run the supported TaskGroup fan-out/fan-in workflow.

    Returns:
        Successful queue records and stable traceback evidence for the
        controlled immediate exception variant.
    """

    queue: asyncio.Queue[WorkItem] = asyncio.Queue(maxsize=QUEUE_CAPACITY)
    traceback_evidence = TracebackEvidence(
        type="missing",
        message="missing",
        producer_frame=False,
    )
    async with asyncio.TaskGroup() as group:
        for item in _successful_items():
            group.create_task(publish_immediate(queue, item))
        records = [await queue.get() for _ in range(len(_successful_items()))]

    await _capture_immediate_failure("failure", traceback_evidence)
    return tuple(records), traceback_evidence


async def _capture_immediate_failure(
    label: str,
    evidence: TracebackEvidence,
) -> None:
    try:
        await fail_immediate(label)
    except ControlledImmediateError as exc:
        summary = traceback.TracebackException.from_exception(exc)
        evidence["type"] = type(exc).__name__
        evidence["message"] = str(exc)
        evidence["producer_frame"] = any(frame.name == "fail_immediate" for frame in summary.stack)


async def canonical_semantic_snapshot() -> SemanticSnapshot:
    """Build the canonical semantic snapshot for one baseline execution."""

    records, failure = await run_supported_workflow()
    ordered = sorted(records, key=lambda item: item["label"])
    context = await _context_isolation_probe()
    return SemanticSnapshot(
        workflow="taskgroup-queue-reduction",
        capacity=QUEUE_CAPACITY,
        total=sum(item["value"] for item in ordered),
        count=len(ordered),
        first=ordered[0]["label"],
        last=ordered[-1]["label"],
        exception_type=failure["type"],
        exception_message=failure["message"],
        exception_frame_present=failure["producer_frame"],
        context_parent=context["parent"],
        context_child=context["child"],
        context_sibling=context["sibling"],
        cold_decoy_count=len(_cold_decoy_names()),
    )


async def repeat_baseline_semantic_matrix() -> tuple[SemanticSnapshot, ...]:
    """Repeat the baseline matrix deterministically for acceptance checks."""

    return tuple([await canonical_semantic_snapshot() for _ in range(MATRIX_REPETITIONS)])


async def _context_isolation_probe() -> dict[Literal["parent", "child", "sibling"], str]:
    _CONTEXT_LABEL.set("parent")

    async def child() -> str:
        _CONTEXT_LABEL.set("child")
        return _CONTEXT_LABEL.get()

    async def sibling() -> str:
        return _CONTEXT_LABEL.get()

    child_result, sibling_result = await asyncio.gather(child(), sibling())
    return {"parent": _CONTEXT_LABEL.get(), "child": child_result, "sibling": sibling_result}


def _successful_items() -> tuple[WorkItem, ...]:
    return (
        WorkItem(label="alpha", value=2),
        WorkItem(label="beta", value=3),
        WorkItem(label="gamma", value=5),
    )


def _cold_decoy_names() -> tuple[str, ...]:
    return (
        "suspension",
        "task-introspection",
        "custom-task-factory",
        "cancellation",
        "context-isolation",
        "debug-mode",
        "side-effecting-iterable",
    )


async def cold_suspension_workflow() -> str:
    """Unsupported decoy because it suspends before producing a value."""

    await asyncio.sleep(0)
    return "suspended"


async def cold_task_introspection_workflow() -> bool:
    """Unsupported decoy because task identity is observable."""

    return asyncio.current_task() is not None


async def cold_custom_task_factory_workflow() -> str:
    """Unsupported decoy because task construction policy is user-controlled."""

    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    loop.set_task_factory(_custom_task_factory)
    try:
        task = asyncio.create_task(asyncio.sleep(0, result="factory"))
        return await task
    finally:
        loop.set_task_factory(previous_factory)


async def cold_cancellation_workflow() -> bool:
    """Unsupported decoy because cancellation changes coroutine cleanup semantics."""

    task = asyncio.create_task(asyncio.sleep(10))
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        return True
    return False


async def cold_context_isolation_workflow() -> tuple[str, str, str]:
    """Unsupported decoy focused on parent, child, and sibling ContextVar state."""

    result = await _context_isolation_probe()
    return result["parent"], result["child"], result["sibling"]


async def cold_debug_mode_workflow() -> bool:
    """Unsupported decoy because event-loop debug mode changes runtime behavior."""

    loop = asyncio.get_running_loop()
    loop.set_debug(True)
    return loop.get_debug()


async def cold_side_effecting_iterable_workflow(items: Iterable[int]) -> int:
    """Unsupported decoy because iteration itself can mutate external state."""

    return sum(_side_effecting_values(items))


def _side_effecting_values(items: Iterable[int]) -> Iterable[int]:
    for item in items:
        yield item + 1


def _custom_task_factory[T](
    loop: asyncio.AbstractEventLoop,
    coro: Coroutine[object, object, T],
    /,
) -> asyncio.Future[T]:
    return asyncio.Task(coro, loop=loop)
