"""Deterministic Rich table-rendering workload."""

from __future__ import annotations

from io import StringIO
from types import ModuleType

import rich
from rich.console import Console
from rich.table import Table

_BATCH = 200


def run(*, repetitions: int, seed: int) -> tuple[dict[str, object], tuple[ModuleType, ...]]:
    """Render a fixed-width table to text and summarize exact output bytes."""
    table = Table(title=f"Corpus {seed}", box=None, padding=(0, 1))
    table.add_column("Index", justify="right")
    table.add_column("Label")
    table.add_column("Value", justify="right")
    for index in range(24):
        table.add_row(str(index), f"item-{(seed + index) % 97:02d}", str((seed + index) ** 2))
    renders = _BATCH * repetitions
    checksum = 0
    final_length = 0
    for _ in range(renders):
        target = StringIO()
        console = Console(
            file=target,
            force_terminal=False,
            color_system=None,
            width=72,
            legacy_windows=False,
        )
        console.print(table)
        final_length = len(target.getvalue())
        checksum += final_length
    return {"checksum": checksum, "last_length": final_length, "renders": renders}, (rich,)
