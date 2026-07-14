"""Semantic contract for the deterministic async typed-region fixture."""

from __future__ import annotations

import asyncio
import gc
import inspect
import sys
from collections.abc import AsyncGenerator
from typing import cast

import pytest
from typed_region_project import async_runner
from typed_region_project.async_runner import (
    DEFAULT_TOKEN,
    RUN_CONTEXT,
    DynamicRunner,
    GenericRunner,
    IntRunner,
    ProtocolRunner,
)

PARSED_DEFAULT = 7
DEFAULT_BIAS = 3
BOXED_VALUE = 5
DYNAMIC_RESULT = 11
FIRST_EXCHANGE_RESULT = 3
SENT_EXCHANGE_RESULT = 7
THROWN_EXCHANGE_RESULT = 1
FINALIZER_FIRST_RESULT = 4
FINALIZER_COUNT = 2


def test_runner_shapes_defaults_descriptors_and_boxed_values() -> None:
    runner = ProtocolRunner()
    token = RUN_CONTEXT.set("fixture-context")
    try:
        assert asyncio.run(runner.compute(4)) == (14, DEFAULT_TOKEN, "fixture-context")
    finally:
        RUN_CONTEXT.reset(token)

    assert inspect.iscoroutinefunction(ProtocolRunner.compute)
    assert inspect.isgeneratorfunction(ProtocolRunner.exchange)
    assert inspect.isasyncgenfunction(ProtocolRunner.async_exchange)
    assert inspect.signature(ProtocolRunner.compute).parameters["token"].default is DEFAULT_TOKEN
    assert ProtocolRunner.compute.__module__ == "typed_region_project.async_runner"
    assert ProtocolRunner.compute.__qualname__ == "ProtocolRunner.compute"
    assert isinstance(vars(ProtocolRunner)["parse"], staticmethod)
    assert isinstance(vars(ProtocolRunner)["with_bias"], classmethod)
    assert ProtocolRunner.parse() == PARSED_DEFAULT
    assert ProtocolRunner.with_bias().bias == DEFAULT_BIAS
    assert GenericRunner[str]("boxed").identity("value") == "value"
    assert IntRunner(4).identity(BOXED_VALUE) == BOXED_VALUE
    assert DynamicRunner().calculate(4) == DYNAMIC_RESULT


def test_runner_compiled_and_interpreted_routes_are_explicit() -> None:
    compiled = bool(getattr(async_runner, "__atoll_status__", {}).get("compiled", False))
    compiled_members = (
        ProtocolRunner.cold_decoy,
        ProtocolRunner.compute,
        ProtocolRunner.exchange,
        ProtocolRunner.fail_after_suspend,
        ProtocolRunner.parse,
        ProtocolRunner.wait_until_cancelled,
        ProtocolRunner.with_bias,
        ProtocolRunner.async_exchange,
    )
    interpreted_members = (
        GenericRunner.identity,
        DynamicRunner.calculate,
    )

    assert all(
        hasattr(member, "__atoll_compiled_target__") is compiled for member in compiled_members
    )
    assert all(not hasattr(member, "__atoll_compiled_target__") for member in interpreted_members)


def test_runner_generator_protocols() -> None:
    runner = ProtocolRunner()
    generator = runner.exchange(1)
    assert runner.sync_started == 0
    assert next(generator) == FIRST_EXCHANGE_RESULT
    assert generator.send(5) == SENT_EXCHANGE_RESULT
    assert generator.throw(ValueError("fixture")) == THROWN_EXCHANGE_RESULT
    generator.close()
    assert runner.sync_finalized == 1

    asyncio.run(_exercise_async_generator(runner))
    asyncio.run(_exercise_async_generator_finalizer_hook(runner))


def test_runner_cancellation_context_and_exception_paths() -> None:
    runner = ProtocolRunner()
    asyncio.run(_exercise_cancellation(runner))
    assert runner.cancelled is True
    assert runner.context_seen == "cancel-context"

    with pytest.raises(LookupError, match="fixture failure"):
        asyncio.run(runner.fail_after_suspend("fixture failure"))


async def _exercise_async_generator(runner: ProtocolRunner) -> None:
    stream = runner.async_exchange(1)
    assert runner.async_started == 0
    assert await anext(stream) == FIRST_EXCHANGE_RESULT
    assert await stream.asend(5) == SENT_EXCHANGE_RESULT
    assert await stream.athrow(ValueError("fixture")) == THROWN_EXCHANGE_RESULT
    await stream.aclose()
    assert runner.async_finalized == 1


async def _exercise_cancellation(runner: ProtocolRunner) -> None:
    started = asyncio.Event()
    token = RUN_CONTEXT.set("cancel-context")
    try:
        task = asyncio.create_task(runner.wait_until_cancelled(started))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        RUN_CONTEXT.reset(token)


async def _exercise_async_generator_finalizer_hook(runner: ProtocolRunner) -> None:
    loop = asyncio.get_running_loop()
    previous_hooks = sys.get_asyncgen_hooks()
    finalizer_calls: list[AsyncGenerator[object, object]] = []
    finalizer_tasks: list[asyncio.Task[None]] = []

    def finalize(generator: object) -> None:
        stream = cast(AsyncGenerator[object, object], generator)
        finalizer_calls.append(stream)
        finalizer_tasks.append(loop.create_task(stream.aclose()))

    sys.set_asyncgen_hooks(firstiter=previous_hooks.firstiter, finalizer=finalize)
    try:
        stream = runner.async_exchange(2)
        assert await anext(stream) == FINALIZER_FIRST_RESULT
        del stream
        gc.collect()
        assert len(finalizer_calls) == 1
        await finalizer_tasks[0]
        assert runner.async_finalized == FINALIZER_COUNT
    finally:
        sys.set_asyncgen_hooks(
            firstiter=previous_hooks.firstiter,
            finalizer=previous_hooks.finalizer,
        )
