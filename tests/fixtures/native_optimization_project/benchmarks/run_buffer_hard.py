"""Calibrated hard benchmark for exact buffer-kernel bindings."""

from __future__ import annotations

import argparse
import array
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypedDict, cast

DEFAULT_CALLS = 1_000
DEFAULT_WIDTH = 2_048
DEFAULT_MEASUREMENTS = 1


class BufferBenchmarkPayload(TypedDict):
    """Stable exact-buffer benchmark output.

    Attributes:
        active_bindings: Names whose imported binding exposes Atoll variants.
        calls: Number of direct calls to each checksum binding per measurement.
        checksums: Deterministic checksum from each measurement.
        measurements: Number of repeated measurements executed in-process.
        width: Number of logical items in each prebuilt benchmark exporter.
    """

    active_bindings: list[str]
    calls: int
    checksums: list[int]
    measurements: int
    width: int


class FixtureModule(Protocol):
    """Fixture callable surface used by the hard buffer benchmark."""

    def array_checksum(self, data: array.array[int]) -> int:
        """Return a deterministic checksum over exact `array.array` input."""
        ...

    def bytearray_checksum(self, data: bytearray) -> int:
        """Return a deterministic checksum over exact `bytearray` input."""
        ...

    def bytes_checksum(self, data: bytes) -> int:
        """Return a deterministic checksum over exact `bytes` input."""
        ...

    def memoryview_checksum(self, data: memoryview) -> int:
        """Return a deterministic checksum over exact `memoryview` input."""
        ...


@dataclass(frozen=True, slots=True)
class WorkloadBuffers:
    """Prebuilt exporters shared by every benchmark measurement."""

    payload: bytes
    mutable: bytearray
    view: memoryview
    values: array.array[int]


def main() -> int:
    """Run direct exact-buffer workloads for subprocess timing."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=int, default=DEFAULT_CALLS)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--measurements", type=int, default=DEFAULT_MEASUREMENTS)
    args = parser.parse_args()
    if args.calls < 1 or args.width < 1 or args.measurements < 1:
        return 2

    fixture = _fixture_module()
    payload = bytes((index * 37 + 11) & 0xFF for index in range(args.width))
    buffers = WorkloadBuffers(
        payload=payload,
        mutable=bytearray(payload),
        view=memoryview(bytearray(payload)),
        values=array.array("B", ((index * 97 + 13) & 0xFF for index in range(args.width))),
    )

    checksums = [
        _run_measurement(
            fixture,
            calls=args.calls,
            buffers=buffers,
        )
        for _ in range(args.measurements)
    ]
    result = BufferBenchmarkPayload(
        active_bindings=_active_bindings(fixture),
        calls=args.calls,
        checksums=checksums,
        measurements=args.measurements,
        width=args.width,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def _run_measurement(
    fixture: FixtureModule,
    *,
    calls: int,
    buffers: WorkloadBuffers,
) -> int:
    mask = (1 << 61) - 1
    checksum = 0
    for index in range(calls):
        checksum += fixture.bytes_checksum(buffers.payload) + index
        checksum ^= fixture.bytearray_checksum(buffers.mutable) << (index & 3)
        checksum += fixture.memoryview_checksum(buffers.view) + (index << 1)
        checksum ^= fixture.array_checksum(buffers.values) << ((index + 1) & 3)
        checksum &= mask
    return checksum


def _active_bindings(fixture: FixtureModule) -> list[str]:
    names = [
        "array_checksum",
        "bytearray_checksum",
        "bytes_checksum",
        "memoryview_checksum",
    ]
    return [
        name for name in names if getattr(getattr(fixture, name), "__atoll_binding_variants__", ())
    ]


def _fixture_module() -> FixtureModule:
    try:
        return cast(FixtureModule, importlib.import_module("native_optimization_fixture.kernels"))
    except ModuleNotFoundError as error:
        if error.name != "native_optimization_fixture":
            raise
        fixture_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(fixture_root / "src"))
        return cast(FixtureModule, importlib.import_module("native_optimization_fixture.kernels"))


if __name__ == "__main__":
    raise SystemExit(main())
