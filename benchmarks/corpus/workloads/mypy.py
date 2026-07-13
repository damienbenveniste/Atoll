"""Deterministic mypy in-process self-check workload."""

from __future__ import annotations

import os
from io import StringIO
from types import ModuleType

import mypy
from mypy.build import build
from mypy.main import process_options

_BATCH = 1


def run(*, repetitions: int, seed: int) -> tuple[dict[str, object], tuple[ModuleType, ...]]:
    """Type-check a generated generic module through mypy's build API."""
    members = "\n".join(f"    ({seed + index}, 'item-{index}')," for index in range(80))
    source = (
        "from collections.abc import Iterable\n"
        "from typing import TypeVar\n\n"
        "T = TypeVar('T')\n\n"
        "def first(values: Iterable[T]) -> T:\n"
        "    return next(iter(values))\n\n"
        "records: list[tuple[int, str]] = [\n"
        f"{members}\n"
        "]\n"
        "answer: tuple[int, str] = first(records)\n"
    )
    sources, options = process_options(
        [
            "--no-incremental",
            "--cache-dir",
            os.devnull,
            "--config-file",
            os.devnull,
            "-c",
            source,
        ],
        stdout=StringIO(),
        stderr=StringIO(),
    )
    checks = _BATCH * repetitions
    files = 0
    for _ in range(checks):
        result = build(sources=sources, options=options)
        if result.errors:
            raise RuntimeError("mypy workload unexpectedly reported type errors")
        files += len(result.files)
    return {"checks": checks, "modules": files, "source_bytes": len(source.encode())}, (mypy,)
