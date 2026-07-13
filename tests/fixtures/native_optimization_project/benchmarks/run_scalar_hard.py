"""Calibrated hard benchmark for guarded fixed-width scalar lowering."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Protocol, TypedDict, cast

DEFAULT_CALLS = 400_000


class ScalarBenchmarkPayload(TypedDict):
    """Stable scalar benchmark output."""

    calls: int
    checksum: int


class FixtureModule(Protocol):
    """Fixture callable surface used by the hard benchmark."""

    def scalar_polynomial(self, limit: int, rounds: int = ..., *, bias: int = ...) -> int:
        """Return one guarded scalar polynomial reduction."""
        ...


def main() -> int:
    """Run a stable scalar workload long enough for paired subprocess timing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=int, default=DEFAULT_CALLS)
    args = parser.parse_args()
    if args.calls < 1:
        return 2

    scalar_polynomial = _fixture_module().scalar_polynomial
    checksum = 0
    for index in range(args.calls):
        checksum ^= scalar_polynomial(96 + (index & 7), bias=index & 3)
    print(json.dumps(ScalarBenchmarkPayload(calls=args.calls, checksum=checksum), sort_keys=True))
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
