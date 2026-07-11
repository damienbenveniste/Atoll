"""Fixture-local behavior checks for the generic async workflow."""

from __future__ import annotations

import asyncio

from execution_plan_fixture import MATRIX_REPETITIONS, repeat_baseline_semantic_matrix

EXPECTED_TOTAL = 10


def test_repeated_matrix_is_stable() -> None:
    matrix = asyncio.run(repeat_baseline_semantic_matrix())

    assert len(matrix) == MATRIX_REPETITIONS
    assert len({tuple(snapshot.items()) for snapshot in matrix}) == 1
    assert matrix[0]["total"] == EXPECTED_TOTAL
    assert matrix[0]["exception_type"] == "ControlledImmediateError"
