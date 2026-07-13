"""Deterministic NetworkX connected-components workload."""

from __future__ import annotations

from types import ModuleType

import networkx

_BATCH = 240
_COMPONENT_SIZE = 80


def run(*, repetitions: int, seed: int) -> tuple[dict[str, object], tuple[ModuleType, ...]]:
    """Build seeded disjoint paths and repeatedly enumerate their components."""
    graph = networkx.Graph()
    component_count = 7 + seed % 5
    for component in range(component_count):
        offset = component * _COMPONENT_SIZE
        graph.add_edges_from(
            (offset + index, offset + index + 1) for index in range(_COMPONENT_SIZE - 1)
        )
    checksum = 0
    scans = _BATCH * repetitions
    for _ in range(scans):
        sizes = sorted(len(nodes) for nodes in networkx.connected_components(graph))
        checksum += sum((index + 1) * size for index, size in enumerate(sizes))
    return {
        "checksum": checksum,
        "components": component_count,
        "nodes": graph.number_of_nodes(),
        "scans": scans,
    }, (networkx,)
