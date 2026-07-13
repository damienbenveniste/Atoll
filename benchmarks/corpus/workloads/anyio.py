"""Deterministic AnyIO memory-stream fan-out workload."""

from __future__ import annotations

from types import ModuleType

import anyio

_ROUNDS = 12
_WIDTH = 2_000
_CONSUMERS = 8


def run(*, repetitions: int, seed: int) -> tuple[dict[str, object], tuple[ModuleType, ...]]:
    """Distribute seeded integers over cloned receive streams and reduce them."""
    width = _WIDTH + seed % 101

    async def one_round() -> int:
        send, receive = anyio.create_memory_object_stream[int](64)
        partials: list[int] = []

        async def producer() -> None:
            async with send:
                for index in range(width):
                    await send.send(seed + index)

        async def consumer(stream: object) -> None:
            subtotal = 0
            async with stream:
                async for value in stream:
                    subtotal += value
            partials.append(subtotal)

        async with receive, anyio.create_task_group() as tasks:
            tasks.start_soon(producer)
            for _ in range(_CONSUMERS):
                tasks.start_soon(consumer, receive.clone())
        return sum(partials)

    async def execute() -> int:
        checksum = 0
        for _ in range(_ROUNDS * repetitions):
            checksum += await one_round()
        return checksum

    return {
        "checksum": anyio.run(execute),
        "consumers": _CONSUMERS,
        "items": width * _ROUNDS * repetitions,
        "rounds": _ROUNDS * repetitions,
    }, (anyio,)
