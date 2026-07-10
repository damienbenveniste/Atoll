"""Deterministic async and generator semantics for typed-region acceptance."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Generator
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Self

RUN_CONTEXT: ContextVar[str] = ContextVar("typed_region_run_context", default="unset")
DEFAULT_TOKEN = object()


@dataclass
class ProtocolRunner:
    """Stateful runner whose methods exercise Python suspension protocols."""

    bias: int = 2
    sync_started: int = field(init=False, default=0)
    sync_finalized: int = field(init=False, default=0)
    async_started: int = field(init=False, default=0)
    async_finalized: int = field(init=False, default=0)
    cancelled: bool = field(init=False, default=False)
    context_seen: str = field(init=False, default="")

    async def compute(
        self,
        limit: int,
        token: object = DEFAULT_TOKEN,
    ) -> tuple[int, object, str]:
        """Compute before one suspension and preserve the source default object."""
        total = 0
        for value in range(limit):
            total += value + self.bias
        await asyncio.sleep(0)
        return total, token, RUN_CONTEXT.get()

    def exchange(self, start: int = 0) -> Generator[int, int | None, None]:
        """Exercise lazy generator start plus send, throw, close, and cleanup."""
        self.sync_started += 1
        value = start
        try:
            while True:
                try:
                    received = yield value + self.bias
                except ValueError:
                    value = -1
                else:
                    value = value + 1 if received is None else received
        finally:
            self.sync_finalized += 1

    async def async_exchange(
        self,
        start: int = 0,
    ) -> AsyncGenerator[int, int | None]:
        """Exercise lazy async-generator start and its complete control protocol."""
        self.async_started += 1
        value = start
        try:
            while True:
                try:
                    received = yield value + self.bias
                except ValueError:
                    value = -1
                else:
                    value = value + 1 if received is None else received
        finally:
            self.async_finalized += 1

    async def wait_until_cancelled(self, started: asyncio.Event) -> None:
        """Record task context and cleanup when cancellation crosses a suspension."""
        self.context_seen = RUN_CONTEXT.get()
        started.set()
        blocker = asyncio.Event()
        try:
            await blocker.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            self.context_seen = RUN_CONTEXT.get()
            raise

    async def fail_after_suspend(self, message: str) -> None:
        """Raise one source exception after a suspension boundary."""
        await asyncio.sleep(0)
        raise LookupError(message)

    @staticmethod
    def parse(value: str = "7") -> int:
        """Parse a defaulted value through a staticmethod descriptor."""
        return int(value)

    @classmethod
    def with_bias(cls, bias: int = 3) -> Self:
        """Construct through a classmethod descriptor with a source default."""
        return cls(bias=bias)

    def cold_decoy(self, limit: int) -> int:
        """Provide a typed but deliberately cold profiling candidate."""
        total = 0
        for value in range(limit):
            total ^= value
        return total


@dataclass
class GenericRunner[T]:
    """Generic dataclass whose values must retain boxed Python identity."""

    payload: T

    def identity(self, value: T) -> T:
        """Return the exact boxed input without narrowing its representation."""
        return value


class IntRunner(GenericRunner[int]):
    """Concrete generic specialization target."""


class DynamicRunner:
    """Dynamic owner whose methods must remain interpreted."""

    def __getattr__(self, name: str) -> int:
        """Resolve unknown attributes dynamically."""
        return len(name)

    def calculate(self, value: int) -> int:
        """Use dynamic attribute lookup from an otherwise typed method."""
        return value + self.missing
