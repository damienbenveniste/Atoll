"""Fixture-local behavior checks for source-optimization semantics."""

from __future__ import annotations

import asyncio

from source_optimization_fixture import (
    MATRIX_REPETITIONS,
    WORK_ITEM_COUNT,
    public_incremental_inspection,
    repeat_baseline_semantic_matrix,
)
from source_optimization_fixture.workflow import benchmark_checksum

EXPECTED_CHECKSUM = 300136


def test_repeated_semantic_matrix_is_stable() -> None:
    matrix = asyncio.run(repeat_baseline_semantic_matrix())

    assert len(matrix) == MATRIX_REPETITIONS
    assert all(snapshot == matrix[0] for snapshot in matrix)
    assert matrix[0]["work_count"] == WORK_ITEM_COUNT
    assert matrix[0]["checksum"] == EXPECTED_CHECKSUM
    assert matrix[0]["first_label"] == "work-0000"
    assert matrix[0]["last_label"] == "work-0255"
    assert matrix[0]["parent_context"] == "parent"
    assert matrix[0]["child_context"] == "worker:child"
    assert matrix[0]["sibling_context"] == "parent"
    assert matrix[0]["exception_type"] == "ControlledWorkflowError"
    assert matrix[0]["exception_message"] == "controlled failure: source-optimization"
    assert matrix[0]["cancellation_cancelled"] is True
    assert matrix[0]["cancellation_cleanup_count"] == 1
    assert matrix[0]["iterator_values"] == (2, 3, 5, 8)


def test_public_async_iterator_supports_incremental_inspection() -> None:
    result = asyncio.run(public_incremental_inspection())

    assert result == {
        "values": (2, 3, 5, 8),
        "snapshots": (2, 5, 10, 18),
        "final_total": 18,
    }


def test_benchmark_checksum_matches_one_hot_pipeline_run() -> None:
    assert asyncio.run(benchmark_checksum(1)) == EXPECTED_CHECKSUM
