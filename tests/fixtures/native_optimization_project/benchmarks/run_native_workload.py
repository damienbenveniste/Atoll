"""Benchmark entry point for the native-optimization fixture."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Protocol, TypedDict, cast

DEFAULT_ITERATIONS = 6000


class BenchmarkPayload(TypedDict):
    """Stable JSON payload printed by the benchmark.

    Attributes:
        checksum: Accumulated workload checksum.
        iterations: Number of workload iterations.
        logical_items: Number of logical branch items processed.
    """

    checksum: int
    iterations: int
    logical_items: int


class WorkloadSnapshot(Protocol):
    """Snapshot returned by the loaded fixture module."""

    checksum: int
    iterations: int
    logical_items: int


class FixtureModule(Protocol):
    """Loaded fixture module interface used by the benchmark."""

    def run_baseline_workload(self, iterations: int) -> WorkloadSnapshot:
        """Run the baseline workload for measurement.

        Args:
            iterations: Number of workload rounds.

        Returns:
            WorkloadSnapshot: Deterministic benchmark result.
        """


def main() -> int:
    """Run deterministic CPU work and print stable JSON.

    Returns:
        int: Process exit status. Non-zero status means invalid arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    args = parser.parse_args()

    if args.iterations < 1:
        return 2

    snapshot = _fixture_module().run_baseline_workload(args.iterations)
    payload = BenchmarkPayload(
        checksum=snapshot.checksum,
        iterations=snapshot.iterations,
        logical_items=snapshot.logical_items,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


def _fixture_module() -> FixtureModule:
    try:
        return cast(FixtureModule, importlib.import_module("native_optimization_fixture"))
    except ModuleNotFoundError as error:
        if error.name != "native_optimization_fixture":
            raise
        fixture_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(fixture_root / "src"))
        return cast(FixtureModule, importlib.import_module("native_optimization_fixture"))


if __name__ == "__main__":
    raise SystemExit(main())
