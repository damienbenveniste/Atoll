"""Deterministic Pydantic Graph fan-out and normalization workload."""

from __future__ import annotations

import asyncio
from itertools import pairwise
from types import ModuleType

import pydantic_graph
from pydantic_graph import GraphBuilder, StepContext
from pydantic_graph.join import reduce_sum

_RUNS = 4
_WIDTH = 3_000
_BUILD_REPETITIONS = 80
_CHAIN_LENGTH = 300


def run(*, repetitions: int, seed: int) -> tuple[dict[str, object], tuple[ModuleType, ...]]:
    """Execute a fan-out graph and repeatedly normalize a typed chain."""
    width = _WIDTH + seed % 101
    builder = GraphBuilder(output_type=int)

    @builder.step
    async def generate(ctx: StepContext[None, None, None]) -> list[int]:
        del ctx
        return list(range(width))

    @builder.step
    async def transform(ctx: StepContext[None, None, int]) -> int:
        return (ctx.inputs + seed) % 17

    collect = builder.join(reduce_sum, initial=0)
    builder.add(
        builder.edge_from(builder.start_node).to(generate),
        builder.edge_from(generate).map().to(transform),
        builder.edge_from(transform).to(collect),
        builder.edge_from(collect).to(builder.end_node),
    )
    graph = builder.build()

    async def execute() -> int:
        checksum = 0
        for _ in range(_RUNS * repetitions):
            checksum += await graph.run(infer_name=False)
        return checksum

    chain = GraphBuilder(output_type=int)

    async def passthrough(ctx: StepContext[None, None, int]) -> int:
        return ctx.inputs

    steps = [chain.step(passthrough, node_id=f"chain_{index}") for index in range(_CHAIN_LENGTH)]
    edges = [chain.edge_from(chain.start_node).to(steps[0])]
    edges.extend(chain.edge_from(source).to(target) for source, target in pairwise(steps))
    edges.append(chain.edge_from(steps[-1]).to(chain.end_node))
    chain.add(*edges)
    builds = _BUILD_REPETITIONS * repetitions
    shape_checksum = 0
    for _ in range(builds):
        built = chain.build(validate_graph_structure=False)
        shape_checksum += len(built.nodes) + sum(
            len(paths) for paths in built.edges_by_source.values()
        )
    return {
        "builds": builds,
        "checksum": asyncio.run(execute()) + shape_checksum,
        "fan_out": width,
        "runs": _RUNS * repetitions,
    }, (pydantic_graph,)
