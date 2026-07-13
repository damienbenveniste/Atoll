"""Deterministic Tomli document-parsing workload."""

from __future__ import annotations

from types import ModuleType

import tomli

_BATCH = 10_000


def run(*, repetitions: int, seed: int) -> tuple[dict[str, object], tuple[ModuleType, ...]]:
    """Parse a typed TOML document and reduce stable scalar values."""
    document = f"""
title = "corpus-{seed}"
enabled = true
ports = [8000, 8001, 8002, 8003]

[database]
host = "localhost"
retries = {3 + seed % 4}

[[workers]]
name = "alpha"
weight = 2

[[workers]]
name = "beta"
weight = 5
"""
    parses = _BATCH * repetitions
    checksum = 0
    for _ in range(parses):
        payload = tomli.loads(document)
        checksum += (
            sum(payload["ports"])
            + payload["database"]["retries"]
            + sum(worker["weight"] for worker in payload["workers"])
        )
    return {"checksum": checksum, "parses": parses}, (tomli,)
