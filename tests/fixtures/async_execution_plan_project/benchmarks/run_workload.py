"""Hot scheduler workload for async execution-plan acceptance."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import sys
from collections.abc import Coroutine, Mapping
from pathlib import Path
from typing import Protocol, cast

EXPECTED_TOTAL = 32896
DEFAULT_ITERATIONS = 1024


class FixtureModule(Protocol):
    """Loaded fixture module interface used by the benchmark script."""

    def run_supported_fanout(
        self,
    ) -> Coroutine[object, object, tuple[Mapping[str, object], ...]]: ...


def main() -> int:
    """Run a fixed amount of deterministic work for throughput measurement."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    args = parser.parse_args()

    if args.iterations < 1:
        return 2
    checksum = asyncio.run(_run_hot_workload(_fixture_module(), args.iterations))
    if checksum != EXPECTED_TOTAL * args.iterations:
        return 1
    print(f"iterations={args.iterations} checksum={checksum}")
    return 0


def _fixture_module() -> FixtureModule:
    fixture_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(fixture_root / "src"))
    return cast(FixtureModule, importlib.import_module("execution_plan_fixture"))


async def _run_hot_workload(module: FixtureModule, iterations: int) -> int:
    checksum = 0
    for _ in range(iterations):
        records = await module.run_supported_fanout()
        checksum += sum(cast(int, item["value"]) for item in records)
    return checksum


if __name__ == "__main__":
    raise SystemExit(main())
