"""Calibrated hard benchmark for direct typed call-chain lowering."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Protocol, TypedDict, cast

DEFAULT_CALLS = 300_000
DEFAULT_DEPTH = 512


class CallChainBenchmarkPayload(TypedDict):
    """Stable direct-call-chain benchmark output.

    Attributes:
        calls: Number of direct-chain root calls.
        checksum: Accumulated direct-chain checksum.
        depth: Helper-chain depth for each root call.
    """

    calls: int
    checksum: int
    depth: int


class FixtureModule(Protocol):
    """Fixture callable surface used by the direct-call-chain benchmark."""

    def call_chain_hard_checksum(self, calls: int, *, depth: int = ...) -> int:
        """Return the deterministic direct-call-chain checksum.

        Args:
            calls: Number of direct-chain root calls.
            depth: Helper-chain depth for each root call.

        Returns:
            Deterministic integer checksum.
        """
        ...


def main() -> int:
    """Run a stable direct-call-chain workload for subprocess timing."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=int, default=DEFAULT_CALLS)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    args = parser.parse_args()
    if args.calls < 1 or args.depth < 0:
        return 2

    checksum = _fixture_module().call_chain_hard_checksum(args.calls, depth=args.depth)
    payload = CallChainBenchmarkPayload(calls=args.calls, checksum=checksum, depth=args.depth)
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
