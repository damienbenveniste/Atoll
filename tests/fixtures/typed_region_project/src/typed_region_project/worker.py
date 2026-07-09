"""Typed methods used by Atoll's source-clean wheel acceptance tests."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Self

from .helpers import twice


def passthrough(value: int) -> int:
    """Remain below the native-readiness threshold for fallback coverage."""
    return value


class Worker:
    """Small interpreted class whose eligible methods can be rebound."""

    bias: int

    def __init__(self, bias: int) -> None:
        """Store the additive bias used by the fixture workload."""
        self.bias = bias

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


class DynamicWorker:
    """Dynamic class that must remain entirely interpreted."""

    def __getattr__(self, name: str) -> int:
        """Resolve unknown attributes dynamically."""
        return len(name)

    def calculate(self, value: int) -> int:
        """Expose an otherwise typed method on an unsafe owner class."""
        return value + self.missing
