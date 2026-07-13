"""Deterministic SymPy symbolic-expansion workload."""

from __future__ import annotations

from types import ModuleType

import sympy

_BATCH = 200


def run(*, repetitions: int, seed: int) -> tuple[dict[str, object], tuple[ModuleType, ...]]:
    """Expand seeded multivariate polynomials and count canonical operations."""
    x, y, z = sympy.symbols("x y z")
    expansions = _BATCH * repetitions
    checksum = 0
    final_terms = 0
    for index in range(expansions):
        shift = (seed + index) % 11 + 1
        expression = ((x + shift * y - z) ** 7 * (x - y + shift * z) ** 3).expand()
        final_terms = len(expression.as_ordered_terms())
        checksum += final_terms + int(sympy.count_ops(expression))
    return {"checksum": checksum, "expansions": expansions, "last_terms": final_terms}, (sympy,)
