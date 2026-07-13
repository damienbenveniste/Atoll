"""Deterministic SQLGlot parsing and optimization workload."""

from __future__ import annotations

from types import ModuleType

import sqlglot
from sqlglot.optimizer import optimize

_BATCH = 300


def run(*, repetitions: int, seed: int) -> tuple[dict[str, object], tuple[ModuleType, ...]]:
    """Parse and optimize a seeded aggregate query into canonical SQL."""
    query = """
SELECT customer_id, SUM(amount) AS total
FROM orders
WHERE status = 'paid' AND amount > 11
GROUP BY customer_id
HAVING SUM(amount) > 100
ORDER BY total DESC
"""
    schema = {"orders": {"customer_id": "INT", "amount": "DECIMAL", "status": "TEXT"}}
    operations = _BATCH * repetitions
    checksum = 0
    canonical = ""
    for _ in range(operations):
        expression = sqlglot.parse_one(query, read="sqlite")
        canonical = optimize(expression, schema=schema).sql(dialect="sqlite")
        checksum += len(canonical) + seed % 17
    return {"checksum": checksum, "operations": operations, "sql": canonical}, (sqlglot,)
