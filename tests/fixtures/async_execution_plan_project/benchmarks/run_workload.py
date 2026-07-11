"""CPU-light manual benchmark for async execution-plan acceptance."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import sys
import time
from collections.abc import Callable, Coroutine, Mapping
from pathlib import Path
from typing import Protocol, cast

EXPECTED_TOTAL = 10


class FixtureModule(Protocol):
    """Loaded fixture module interface used by the benchmark script."""

    def repeat_baseline_semantic_matrix(
        self,
    ) -> Coroutine[object, object, tuple[Mapping[str, object], ...]]: ...


def main() -> int:
    """Run deterministic work for at least the requested duration."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum-seconds", type=float, default=0.25)
    args = parser.parse_args()

    repeat_matrix = _repeat_matrix()
    deadline = time.perf_counter() + args.minimum_seconds
    iterations = 0
    while iterations == 0 or time.perf_counter() < deadline:
        matrix = asyncio.run(repeat_matrix())
        if matrix[0]["total"] != EXPECTED_TOTAL:
            return 1
        iterations += 1
    print(f"iterations={iterations}")
    return 0


def _repeat_matrix() -> Callable[[], Coroutine[object, object, tuple[Mapping[str, object], ...]]]:
    fixture_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(fixture_root / "src"))
    module = cast(FixtureModule, importlib.import_module("execution_plan_fixture"))
    return module.repeat_baseline_semantic_matrix


if __name__ == "__main__":
    raise SystemExit(main())
