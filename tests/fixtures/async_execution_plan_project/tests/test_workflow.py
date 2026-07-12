"""Fixture-local behavior checks for the generic async workflow."""

from __future__ import annotations

import asyncio
import inspect

import pytest
from execution_plan_fixture import (
    MATRIX_REPETITIONS,
    repeat_baseline_semantic_matrix,
    run_supported_fanout,
)

EXPECTED_TOTAL = 32896


def test_repeated_matrix_is_stable() -> None:
    matrix = asyncio.run(repeat_baseline_semantic_matrix())

    assert len(matrix) == MATRIX_REPETITIONS
    assert len({tuple(snapshot.items()) for snapshot in matrix}) == 1
    assert matrix[0]["total"] == EXPECTED_TOTAL
    assert matrix[0]["exception_type"] == "ControlledImmediateError"


def test_public_fanout_preserves_native_coroutine_protocol() -> None:
    assert inspect.iscoroutinefunction(run_supported_fanout)
    assert run_supported_fanout.__module__ == "execution_plan_fixture.workflow"
    assert run_supported_fanout.__qualname__ == "run_supported_fanout"
    assert not inspect.signature(run_supported_fanout).parameters

    coroutine = run_supported_fanout()
    assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CREATED

    async def drive() -> tuple[object, ...]:
        task = asyncio.create_task(coroutine)
        assert task.get_coro() is coroutine
        return await task

    records = asyncio.run(drive())
    assert len(records) > 0
    assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED

    async def reuse() -> None:
        await coroutine

    with pytest.raises(RuntimeError, match="cannot reuse already awaited coroutine"):
        asyncio.run(reuse())
