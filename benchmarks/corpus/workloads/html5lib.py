"""Deterministic html5lib document-parsing workload."""

from __future__ import annotations

from types import ModuleType

import html5lib

_BATCH = 1_000


def run(*, repetitions: int, seed: int) -> tuple[dict[str, object], tuple[ModuleType, ...]]:
    """Parse malformed-but-fixed HTML and count the normalized element tree."""
    rows = "".join(
        f"<tr><td>{seed + index}<td data-k='{index % 7}'>value-{index % 13}" for index in range(24)
    )
    document = (
        "<!doctype html><title>corpus</title><main><table>"
        f"{rows}</table><p>tail &amp; marker</main>"
    )
    elements = 0
    parses = _BATCH * repetitions
    for _ in range(parses):
        tree = html5lib.parse(document, namespaceHTMLElements=False)
        elements += sum(1 for _element in tree.iter())
    return {"element_visits": elements, "parses": parses}, (html5lib,)
