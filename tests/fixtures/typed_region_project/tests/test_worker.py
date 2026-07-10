"""Behavior tests shipped with the typed-region fixture."""

from __future__ import annotations

import asyncio
import inspect
from typing import cast

from typed_region_project.generic_functions import pair_int, pair_value
from typed_region_project.helpers import Payload
from typed_region_project.worker import (
    DynamicWorker,
    IntPairer,
    OptionalPairer,
    Pairer,
    PayloadPairer,
    Worker,
)

SCALE_RESULT = 23
PARSED_RESULT = 7
ADJUSTED_RESULT = 6
DYNAMIC_RESULT = 11
ASYNC_FIRST_RESULT = 4
ASYNC_SENT_RESULT = 8
ASYNC_THROWN_RESULT = 2


def test_worker_callable_shapes_and_results() -> None:
    """The fixture defines stable behavior for compiled and fallback routing."""
    worker = Worker(3)

    assert worker.scale(5) == SCALE_RESULT
    assert Worker.parse("7") == PARSED_RESULT
    assert Worker.adjust(4) == ADJUSTED_RESULT
    assert list(worker.values(3)) == [3, 3, 5]
    assert asyncio.run(worker.score(5)) == SCALE_RESULT
    asyncio.run(_assert_async_generator_protocol(worker))
    assert inspect.isgeneratorfunction(Worker.values)
    assert inspect.iscoroutinefunction(Worker.score)
    assert inspect.isasyncgenfunction(Worker.exchange)


def test_dynamic_worker_remains_interpreted() -> None:
    """Custom attribute behavior stays authoritative in the source class."""
    worker = DynamicWorker()

    assert worker.calculate(4) == DYNAMIC_RESULT


def test_generic_fallbacks_cover_subclass_and_closed_call_specializations() -> None:
    """The fixture accepts both specialized and deliberately incompatible values."""
    assert Pairer[str]().pair("base") == ("base", "base")
    assert IntPairer().pair(4) == (4, 4)
    assert cast(Pairer[object], IntPairer()).pair("fallback") == (
        "fallback",
        "fallback",
    )
    payload = Payload(3)
    assert PayloadPairer().maybe_pair(payload) == (payload, payload)
    assert PayloadPairer().maybe_pair(None) == (None, None)
    assert cast(OptionalPairer[object], PayloadPairer()).maybe_pair(3) == (3, 3)
    assert pair_int(5) == (5, 5)
    assert pair_value("fallback") == ("fallback", "fallback")


async def _assert_async_generator_protocol(worker: Worker) -> None:
    generator = worker.exchange(1)

    assert await anext(generator) == ASYNC_FIRST_RESULT
    assert await generator.asend(5) == ASYNC_SENT_RESULT
    assert await generator.athrow(ValueError("fixture")) == ASYNC_THROWN_RESULT
    await generator.aclose()
    assert worker.closed is True
