"""Benchmark entry point for the source-optimization fixture."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
from collections.abc import Coroutine
from pathlib import Path
from typing import Protocol, TypedDict, cast

DEFAULT_ITERATIONS = 1024
MINIMUM_ITERATIONS = 40
EXPECTED_WORK_ITEM_COUNT = 256
EXPECTED_CHECKSUM_PER_ITERATION = 300136


class BenchmarkPayload(TypedDict):
    """Stable JSON payload printed by the benchmark.

    Attributes:
        checksum: Accumulated checksum across every hot pipeline run.
        iterations: Number of hot pipeline runs performed.
        logical_items: Number of logical work items processed.
    """

    checksum: int
    iterations: int
    logical_items: int


class FixtureModule(Protocol):
    """Loaded fixture module interface used by the benchmark."""

    WORK_ITEM_COUNT: int

    def benchmark_checksum(self, iterations: int) -> Coroutine[object, object, int]:
        """Run the fixture hot path for benchmark measurement.

        Args:
            iterations: Number of hot path executions.

        Returns:
            Coroutine[object, object, int]: Awaitable checksum result.
        """


def main() -> int:
    """Run deterministic hot-path work and print stable JSON.

    Returns:
        int: Process exit status. Non-zero status means invalid arguments or an
        unexpected checksum.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    args = parser.parse_args()

    if args.iterations < MINIMUM_ITERATIONS:
        return 2
    module = _fixture_module()
    checksum = asyncio.run(module.benchmark_checksum(args.iterations))
    logical_items = args.iterations * module.WORK_ITEM_COUNT
    expected_checksum = EXPECTED_CHECKSUM_PER_ITERATION * args.iterations
    if module.WORK_ITEM_COUNT != EXPECTED_WORK_ITEM_COUNT or checksum != expected_checksum:
        return 1
    payload = BenchmarkPayload(
        checksum=checksum,
        iterations=args.iterations,
        logical_items=logical_items,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


def _fixture_module() -> FixtureModule:
    fixture_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(fixture_root / "src"))
    return cast(FixtureModule, importlib.import_module("source_optimization_fixture.workflow"))


if __name__ == "__main__":
    raise SystemExit(main())
