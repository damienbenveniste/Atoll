"""Generic class-owned AnyIO pipeline used by source-lowering tests."""

from __future__ import annotations

import contextvars
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

import anyio
from anyio.abc import TaskGroup
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

_ITEM_CONTEXT = contextvars.ContextVar("source_optimizer_item", default="parent")


@dataclass(frozen=True)
class WorkItem:
    """One nominal work item routed to a node by identifier."""

    item_id: int
    node_id: str
    value: int


@dataclass(frozen=True)
class StepNode:
    """One asynchronous step callable stored in the runner node mapping."""

    call: Callable[[int], Awaitable[int]]


@dataclass(frozen=True)
class ConstantNode:
    """One synchronous node shape handled without awaiting a callable."""

    offset: int


@dataclass(frozen=True)
class WorkResult:
    """Private result record delivered from producer to consumer."""

    source: WorkItem
    result: int | Sequence[WorkItem]
    error: BaseException | None = None


@dataclass
class PipelineRunner:
    """Own a private task group, zero-capacity stream, and result consumer."""

    nodes: dict[str, StepNode | ConstantNode]
    task_group: TaskGroup
    active: dict[int, WorkItem] = field(init=False)
    reducers: dict[int, int] = field(init=False)
    send_stream: MemoryObjectSendStream[WorkResult] = field(init=False)
    receive_stream: MemoryObjectReceiveStream[WorkResult] = field(init=False)

    def __post_init__(self) -> None:
        self.active = {}
        self.reducers = {}
        self.send_stream, self.receive_stream = anyio.create_memory_object_stream[WorkResult](0)

    def submit(self, request: Sequence[WorkItem]) -> None:
        for item in request:
            self.active[item.item_id] = item
        for item in request:
            self.task_group.start_soon(self._worker, item)

    async def _worker(self, item: WorkItem) -> None:
        try:
            result = await self._run_item(item)
        except BaseException as exc:
            await self.send_stream.send(WorkResult(item, (), error=exc))
            return
        await self.send_stream.send(WorkResult(item, result))

    async def _run_item(self, item: WorkItem) -> int | Sequence[WorkItem]:
        node_id = item.node_id
        node = self.nodes[node_id]
        if isinstance(node, StepNode):
            return await node.call(item.value)
        return item.value + node.offset

    async def results(self, request: Sequence[WorkItem]) -> list[int]:
        output: list[int] = []
        self.submit(request)
        async with self.receive_stream:
            while self.active:
                async for record in self.receive_stream:
                    if record.error is not None:
                        raise record.error
                    if not isinstance(record.result, int):
                        raise TypeError("fixture consumer expects a scalar result")
                    output.append(record.result)
                    self.active.pop(record.source.item_id)

                    active_snapshot = list(self.active.values())
                    completed_reducers: list[int] = []
                    for reducer_id in self._completed_reducers(record.source, active_snapshot):
                        self.reducers.pop(reducer_id)
                        completed_reducers.append(reducer_id)

                    if not self.active:
                        break
        return output

    def _completed_reducers(
        self,
        source: WorkItem,
        active: list[WorkItem],
    ) -> tuple[int, ...]:
        del source, active
        return ()


async def immediate_double(value: int) -> int:
    """Return without suspension while observing the current context."""
    if _ITEM_CONTEXT.get() not in {"parent", "child"}:
        raise RuntimeError("unexpected item context")
    return value * 2


async def suspending_double(value: int) -> int:
    """Suspend once so guarded lowering must retain the task path."""
    await anyio.sleep(0)
    return value * 2


def mutate_context(value: int) -> int:
    """Mutate context indirectly so recursive code inspection rejects it."""
    _ITEM_CONTEXT.set("child")
    return value


async def indirect_context_mutation(value: int) -> int:
    """Call a context-mutating helper without spelling the API locally."""
    return mutate_context(value) * 2


async def run_pipeline(
    values: Sequence[int],
    worker: Callable[[int], Awaitable[int]] = immediate_double,
) -> tuple[int, ...]:
    """Run one private AnyIO pipeline to completion."""
    async with anyio.create_task_group() as task_group:
        runner = PipelineRunner({"step": StepNode(worker)}, task_group)
        request = tuple(
            WorkItem(item_id=index, node_id="step", value=value)
            for index, value in enumerate(values)
        )
        result = await runner.results(request)
    return tuple(result)
