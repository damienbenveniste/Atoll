"""Typed methods used by Atoll's source-clean wheel acceptance tests."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Iterator
from typing import Self

from .helpers import Payload, twice


def passthrough(value: int) -> int:
    """Remain below the native-readiness threshold for fallback coverage."""
    return value


class Worker:
    """Small interpreted class whose eligible methods can be rebound."""

    bias: int
    closed: bool

    def __init__(self, bias: int) -> None:
        """Store the additive bias used by the fixture workload."""
        self.bias = bias
        self.closed = False

    def scale(self: Self, value: int) -> int:
        """Run a typed integer loop and apply this worker's bias."""
        total = 0
        for item in range(value):
            total += twice(item)
        return total + self.bias

    @staticmethod
    def parse(value: str) -> int:
        """Parse one integer without depending on instance state."""
        return int(value)

    @classmethod
    def adjust(cls: type[Self], value: int) -> int:
        """Exercise explicit classmethod descriptor rebinding."""
        _ = cls
        return value + 2

    def values(self, limit: int) -> Iterator[int]:
        """Yield scaled values while preserving generator behavior."""
        for value in range(limit):
            yield self.scale(value)

    async def score(self, value: int) -> int:
        """Return a scaled value while preserving coroutine behavior."""
        return self.scale(value)

    async def exchange(self, start: int) -> AsyncGenerator[int, int | None]:
        """Exercise the complete async-generator delegation protocol."""
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
            self.closed = True


class Pairer[T]:
    """Generic base retained as Python while concrete subclasses may specialize."""

    def pair(self, value: T) -> tuple[T, T]:
        """Return two references to the input without narrowing the public generic API."""
        return value, value


class IntPairer(Pairer[int]):
    """Concrete specialization target for the inherited ``pair`` method."""


class OptionalPairer[T]:
    """Generic base whose specialization accepts a nominal value or None."""

    def maybe_pair(self, value: T | None) -> tuple[T | None, T | None]:
        """Return the optional input twice while preserving generic fallback."""
        return value, value


class PayloadPairer(OptionalPairer[Payload]):
    """Nominal union specialization target used by wheel acceptance tests."""


class DynamicWorker:
    """Dynamic class that must remain entirely interpreted."""

    def __getattr__(self, name: str) -> int:
        """Resolve unknown attributes dynamically."""
        return len(name)

    def calculate(self, value: int) -> int:
        """Expose an otherwise typed method on an unsafe owner class."""
        return value + self.missing


class UnsafeIdentityWorker:
    """Eagerly retained owner downgraded so source identity survives."""

    value: int

    def __init__(self, value: int) -> None:
        """Store one value for module-time identity checks."""
        self.value = value

    def square(self, value: int) -> int:
        """Expose a safe method on an unsafe class owner."""
        return value * value


UNSAFE_IDENTITY_CLASS = UnsafeIdentityWorker
UNSAFE_IDENTITY_INSTANCE = UnsafeIdentityWorker(5)


class ScaleModel:
    """Safe non-generic class used to verify atomic class compilation."""

    name: str
    factor: int

    def __init__(self, name: str, factor: int = 2) -> None:
        """Store a named integer scale factor."""
        self.name = name
        self.factor = factor

    def apply(self, value: int) -> int:
        """Scale one integer value."""
        return value * self.factor

    def describe(self) -> str:
        """Return a compact source-visible description."""
        return f"{self.name}:{self.factor}"
