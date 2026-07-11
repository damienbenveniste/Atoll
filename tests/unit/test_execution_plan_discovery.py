"""Tests for AST-only scheduler execution-plan discovery."""

import ast
import sys
from dataclasses import replace
from pathlib import Path

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.execution_plans import (
    build_execution_plans,
    execution_plan_observation_targets,
    execution_plan_profile_targets,
)
from atoll.execution_plans.dialects import AnyioOnAsyncioDialect, AsyncioDialect
from atoll.execution_plans.models import ExecutionPlan, PlanRejection
from atoll.models import ModuleId, ModuleScan, SymbolId
from atoll.runtime.profiling import (
    CanonicalCallableCount,
    LifecycleCounts,
    ProfiledMember,
    ProfiledSpawnSite,
    ProfileResult,
    run_baseline_profile,
)

_HOT_PLAN_COUNT = 18_000
_SELECTION_LIMIT = 4
_LIVE_SPAWN_COUNT = 1_200
_QUEUE_CAPACITY = 4


def test_dialects_recognize_asyncio_and_anyio_spawn_shapes(tmp_path: Path) -> None:
    """Built-in dialects recognize spawn calls without Pydantic-specific identifiers."""
    module = _scan(
        tmp_path / "dialects.py",
        [
            "async def _produce(q):",
            "    await q.put(1)",
            "async def _consume(q):",
            "    await q.get()",
            "async def _send(send):",
            "    await send.send(1)",
            "async def _receive(receive):",
            "    await receive.receive()",
            "async def run_asyncio():",
            "    q = asyncio.Queue(maxsize=1)",
            "    asyncio.create_task(_produce(q))",
            "    async with asyncio.TaskGroup() as tg:",
            "        tg.create_task(_consume(q))",
            "async def run_anyio():",
            "    send, receive = anyio.create_memory_object_stream(10)",
            "    async with anyio.create_task_group() as tg:",
            "        tg.start_soon(_send, send)",
            "        tg.start_soon(_receive, receive)",
        ],
    )
    targets = execution_plan_observation_targets((module,))
    source_text = Path(module.module.path).read_text(encoding="utf-8")

    assert targets == (
        "dialects::_consume",
        "dialects::_produce",
        "dialects::_receive",
        "dialects::_send",
        "dialects::run_anyio",
        "dialects::run_asyncio",
    )
    assert AsyncioDialect().name == "asyncio"
    assert AnyioOnAsyncioDialect().name == "anyio-on-asyncio"
    assert "pydantic" not in source_text.lower()


def test_build_execution_plans_selects_by_threshold_order_and_coverage(tmp_path: Path) -> None:
    """Discovery ranks hot sites, applies thresholds, and stops at coverage target."""
    scan = _scan(
        tmp_path / "ranking.py",
        [
            "async def _produce(q):",
            "    await q.put(1)",
            "async def _consume(q):",
            "    await q.get()",
            "async def _produce_two(q):",
            "    await q.put(2)",
            "async def _consume_two(q):",
            "    await q.get()",
            "async def _produce_three(q):",
            "    await q.put(3)",
            "async def _consume_three(q):",
            "    await q.get()",
            "async def run_hot():",
            "    q = asyncio.Queue(maxsize=1)",
            "    async with asyncio.TaskGroup() as tg:",
            "        tg.create_task(_produce(q))",
            "        tg.create_task(_consume(q))",
            "async def run_warm():",
            "    q = asyncio.Queue(maxsize=1)",
            "    async with asyncio.TaskGroup() as tg:",
            "        tg.create_task(_produce_two(q))",
            "        tg.create_task(_consume_two(q))",
            "async def run_low():",
            "    q = asyncio.Queue(maxsize=1)",
            "    async with asyncio.TaskGroup() as tg:",
            "        tg.create_task(_produce_three(q))",
            "        tg.create_task(_consume_three(q))",
        ],
    )
    profile = _profile(
        scan,
        (
            ("ranking", "run_hot", 9_000, 9_000),
            ("ranking", "run_warm", 2_000, 2_000),
            ("ranking", "run_low", 999, 999),
            ("ranking", "_produce", 9_000, 9_000),
            ("ranking", "_consume", 9_000, 9_000),
            ("ranking", "_produce_two", 2_000, 2_000),
            ("ranking", "_consume_two", 2_000, 2_000),
            ("ranking", "_produce_three", 999, 999),
            ("ranking", "_consume_three", 999, 999),
        ),
    )

    results = build_execution_plans((scan,), profile)
    plans = tuple(result for result in results if isinstance(result, ExecutionPlan))
    rejections = tuple(result for result in results if isinstance(result, PlanRejection))

    assert [plan.owner.qualname for plan in plans] == ["run_hot"]
    assert plans[0].hotness == _HOT_PLAN_COUNT
    assert {rejection.owner.qualname: rejection.reason for rejection in rejections} == {
        "run_low": "low-hotness",
        "run_warm": "coverage-reached",
    }


def test_execution_plan_ids_ignore_profile_counts(tmp_path: Path) -> None:
    """Selected plan identity is stable when only dynamic profile counts change."""
    scan = _scan(
        tmp_path / "stable.py",
        [
            "async def _produce(q):",
            "    await q.put(1)",
            "async def _consume(q):",
            "    await q.get()",
            "async def run():",
            "    q = asyncio.Queue(maxsize=1)",
            "    async with asyncio.TaskGroup() as tg:",
            "        tg.create_task(_produce(q))",
            "        tg.create_task(_consume(q))",
        ],
    )

    first = build_execution_plans(
        (scan,),
        _profile(
            scan,
            (
                ("stable", "run", 2_000, 2_000),
                ("stable", "_produce", 2_000, 2_000),
                ("stable", "_consume", 2_000, 2_000),
            ),
        ),
    )
    second = build_execution_plans(
        (scan,),
        _profile(
            scan,
            (
                ("stable", "run", 4_000, 4_000),
                ("stable", "_produce", 4_000, 4_000),
                ("stable", "_consume", 4_000, 4_000),
            ),
        ),
    )
    first_plan = next(result for result in first if isinstance(result, ExecutionPlan))
    second_plan = next(result for result in second if isinstance(result, ExecutionPlan))

    assert first_plan.id == second_plan.id
    assert first_plan.source_hash == second_plan.source_hash


def test_hot_shared_worker_does_not_promote_a_cold_spawn_site(tmp_path: Path) -> None:
    """Exact spawn-site counts prevent cross-site worker heat from leaking."""
    scan = _scan(
        tmp_path / "shared.py",
        [
            "async def _produce(q):",
            "    await q.put(1)",
            "async def _consume(q):",
            "    await q.get()",
            "async def run_hot():",
            "    q = asyncio.Queue(maxsize=1)",
            "    async with asyncio.TaskGroup() as tg:",
            "        tg.create_task(_produce(q))",
            "        tg.create_task(_consume(q))",
            "async def run_cold():",
            "    q = asyncio.Queue(maxsize=1)",
            "    async with asyncio.TaskGroup() as tg:",
            "        tg.create_task(_produce(q))",
            "        tg.create_task(_consume(q))",
        ],
    )
    profile = _profile(
        scan,
        (
            ("shared", "run_hot", 2_000, 2_000),
            ("shared", "run_cold", 1, 1),
            ("shared", "_produce", 2_001, 2_001),
            ("shared", "_consume", 2_001, 2_001),
        ),
    )

    results = build_execution_plans((scan,), profile)

    assert [result.owner.qualname for result in results if isinstance(result, ExecutionPlan)] == [
        "run_hot"
    ]
    assert [
        (result.owner.qualname, result.reason)
        for result in results
        if isinstance(result, PlanRejection)
    ] == [("run_cold", "low-hotness")]


def test_owner_consumer_forms_bounded_fan_out_reduction_plan(tmp_path: Path) -> None:
    """An orchestrator may own result delivery and reduction without a consumer task."""
    scan = _scan(
        tmp_path / "fanout.py",
        [
            "async def _worker(q, value):",
            "    await q.put(value)",
            "async def run(values):",
            "    q = asyncio.Queue(maxsize=1)",
            "    total = 0",
            "    async with asyncio.TaskGroup() as tg:",
            "        for value in values:",
            "            tg.create_task(_worker(q, value))",
            "        for _ in values:",
            "            total += await q.get()",
            "    return total",
        ],
    )
    profile = _profile(
        scan,
        (
            ("fanout", "run", 2_000, 2_000),
            ("fanout", "_worker", 2_000, 2_000),
        ),
    )

    plan = next(
        result
        for result in build_execution_plans((scan,), profile)
        if isinstance(result, ExecutionPlan)
    )

    assert plan.consumer == SymbolId("fanout", "run")
    assert plan.reducer == SymbolId("fanout", "run")
    assert plan.transport_capacity == 1
    assert plan.ordering_policy == "completion-order"
    assert any(node.role == "reducer" for node in plan.nodes)


def test_module_integer_constant_resolves_private_queue_capacity(tmp_path: Path) -> None:
    """A direct module literal remains a statically known queue capacity."""
    scan = _scan(
        tmp_path / "constant_capacity.py",
        [
            "QUEUE_CAPACITY = 4",
            "async def _worker(q, value):",
            "    q.put_nowait(value)",
            "async def run(values):",
            "    q: asyncio.Queue[int] = asyncio.Queue(maxsize=QUEUE_CAPACITY)",
            "    async with asyncio.TaskGroup() as tg:",
            "        for value in values:",
            "            tg.create_task(_worker(q, value))",
            "        for _ in values:",
            "            await q.get()",
        ],
    )
    profile = _profile(
        scan,
        (
            ("constant_capacity", "run", 2_000, 2_000),
            ("constant_capacity", "_worker", 2_000, 2_000),
        ),
    )

    plan = next(
        result
        for result in build_execution_plans((scan,), profile)
        if isinstance(result, ExecutionPlan)
    )

    assert plan.transport_capacity == _QUEUE_CAPACITY


def test_unjoined_create_task_site_is_rejected_as_unstructured(tmp_path: Path) -> None:
    """Bare task creation cannot claim task-handle ownership or joined scope."""
    scan = _scan(
        tmp_path / "unjoined.py",
        [
            "async def _produce(q):",
            "    await q.put(1)",
            "async def _consume(q):",
            "    await q.get()",
            "async def run():",
            "    q = asyncio.Queue(maxsize=1)",
            "    asyncio.create_task(_produce(q))",
            "    asyncio.create_task(_consume(q))",
        ],
    )

    results = build_execution_plans(
        (scan,),
        _profile(
            scan,
            (
                ("unjoined", "run", 2_000, 2_000),
                ("unjoined", "_produce", 2_000, 2_000),
                ("unjoined", "_consume", 2_000, 2_000),
            ),
        ),
    )

    rejection = next(result for result in results if isinstance(result, PlanRejection))
    assert rejection.reason == "unstructured-task"


def test_unbounded_result_transport_is_rejected(tmp_path: Path) -> None:
    """Execution plans require a statically known positive delivery capacity."""
    scan = _scan(
        tmp_path / "unbounded.py",
        [
            "async def _worker(q):",
            "    await q.put(1)",
            "async def run():",
            "    q = asyncio.Queue()",
            "    async with asyncio.TaskGroup() as group:",
            "        group.create_task(_worker(q))",
            "        await q.get()",
        ],
    )

    rejection = next(
        result
        for result in build_execution_plans((scan,), None)
        if isinstance(result, PlanRejection)
    )
    assert rejection.reason == "unknown-capacity"


def test_discovery_rejects_escaping_handles_and_incomplete_transport_roles(
    tmp_path: Path,
) -> None:
    """Ownership and role proofs reject several distinct unsafe transport shapes."""
    scan = _scan(
        tmp_path / "ownership.py",
        [
            "class Holder:",
            "    pass",
            "holder = Holder()",
            "async def _produce(q):",
            "    await q.put(1)",
            "async def _consume(q):",
            "    await q.get()",
            "async def _unknown(q):",
            "    await q.join()",
            "async def escaping_handle():",
            "    q = asyncio.Queue(maxsize=1)",
            "    async with asyncio.TaskGroup() as group:",
            "        task = group.create_task(_produce(q))",
            "        await q.get()",
            "    return task",
            "async def attribute_escape():",
            "    q = asyncio.Queue(maxsize=1)",
            "    holder.queue = q",
            "    async with asyncio.TaskGroup() as group:",
            "        group.create_task(_produce(q))",
            "        await q.get()",
            "async def annotated_attribute_escape():",
            "    q = asyncio.Queue(maxsize=1)",
            "    holder.queue: object = q",
            "    async with asyncio.TaskGroup() as group:",
            "        group.create_task(_produce(q))",
            "        await q.get()",
            "async def global_escape():",
            "    global q",
            "    q = asyncio.Queue(maxsize=1)",
            "    async with asyncio.TaskGroup() as group:",
            "        group.create_task(_produce(q))",
            "        await q.get()",
            "async def container_escape():",
            "    q = asyncio.Queue(maxsize=1)",
            "    async with asyncio.TaskGroup() as group:",
            "        group.create_task(_produce(q))",
            "        await q.get()",
            "    return {'queue': q}",
            "async def starred_escape():",
            "    q = asyncio.Queue(maxsize=1)",
            "    async with asyncio.TaskGroup() as group:",
            "        group.create_task(_produce(q))",
            "        await q.get()",
            "    return (*[q],)",
            "async def no_producer():",
            "    q = asyncio.Queue(maxsize=1)",
            "    async with asyncio.TaskGroup() as group:",
            "        group.create_task(_consume(q))",
            "async def unknown_role():",
            "    q = asyncio.Queue(maxsize=1)",
            "    async with asyncio.TaskGroup() as group:",
            "        group.create_task(_produce(q))",
            "        group.create_task(_unknown(q))",
            "        await q.get()",
        ],
    )

    rejections = {
        result.owner.qualname: result.reason
        for result in build_execution_plans((scan,), None)
        if isinstance(result, PlanRejection)
    }

    assert rejections == {
        "attribute_escape": "public-transport",
        "annotated_attribute_escape": "public-transport",
        "container_escape": "public-transport",
        "escaping_handle": "escaping-handle",
        "global_escape": "public-transport",
        "no_producer": "unknown-transport",
        "starred_escape": "public-transport",
        "unknown_role": "unknown-transport",
    }


def test_dynamic_scheduler_identity_rejects_profiled_site(tmp_path: Path) -> None:
    """A monkey-patched scheduler callable cannot pass dialect identity guards."""
    scan = _scan(
        tmp_path / "dynamic.py",
        [
            "async def _worker(q):",
            "    await q.put(1)",
            "async def run():",
            "    q = asyncio.Queue(maxsize=1)",
            "    async with asyncio.TaskGroup() as group:",
            "        group.create_task(_worker(q))",
            "        await q.get()",
        ],
    )
    profile = _profile(
        scan,
        (
            ("dynamic", "run", 2_000, 2_000),
            ("dynamic", "_worker", 2_000, 2_000),
        ),
    )
    spawn_site = profile.spawn_sites[0]
    profile = replace(
        profile,
        spawn_sites=(
            replace(
                spawn_site,
                callable_identities=(
                    CanonicalCallableCount("project.custom_scheduler", spawn_site.invocation_count),
                ),
            ),
        ),
    )

    rejection = next(
        result
        for result in build_execution_plans((scan,), profile)
        if isinstance(result, PlanRejection)
    )
    assert rejection.reason == "dynamic-scheduler"


def test_real_profile_selects_a_hot_bounded_fan_out_site(tmp_path: Path) -> None:
    """Subprocess spawn counters feed discovery without synthetic counts."""
    scan = _scan(
        tmp_path / "live_scheduler.py",
        [
            "import asyncio",
            "async def _worker(q, value):",
            "    await q.put(value)",
            "async def run():",
            "    q = asyncio.Queue(maxsize=1)",
            "    async with asyncio.TaskGroup() as group:",
            f"        for value in range({_LIVE_SPAWN_COUNT}):",
            (
                "            group.create_task(_worker(q, value)); "
                "group.create_task(_worker(q, value))"
            ),
            f"        for _ in range({_LIVE_SPAWN_COUNT * 2}):",
            "            await q.get()",
        ],
    )
    (tmp_path / "bench.py").write_text(
        "import asyncio\nfrom live_scheduler import run\nasyncio.run(run())\n",
        encoding="utf-8",
    )
    observed_symbols = tuple(
        SymbolId(module=target.partition("::")[0], qualname=target.partition("::")[2])
        for target in execution_plan_observation_targets((scan,))
    )
    profile = run_baseline_profile(
        (sys.executable, "bench.py"),
        project_root=tmp_path,
        payload_root=tmp_path,
        module_paths=(("live_scheduler", "live_scheduler.py"),),
        scratch_dir=tmp_path / "scratch",
        observation_targets=observed_symbols,
        spawn_targets=execution_plan_profile_targets((scan,)),
    )

    plans = tuple(
        result
        for result in build_execution_plans((scan,), profile)
        if isinstance(result, ExecutionPlan)
    )

    assert [site.invocation_count for site in profile.spawn_sites] == [
        _LIVE_SPAWN_COUNT,
        _LIVE_SPAWN_COUNT,
    ]
    assert plans[0].observed_invocations == _LIVE_SPAWN_COUNT
    assert plans[0].lifecycle_starts == _LIVE_SPAWN_COUNT * 2


def test_rejected_report_plans_cover_unsafe_sites(tmp_path: Path) -> None:
    """Ambiguous, public, multi-consumer, unknown, and low-hotness sites are reported."""
    scan = _scan(
        tmp_path / "unsafe.py",
        [
            "async def public_consume(q):",
            "    await q.get()",
            "async def _produce(q):",
            "    await q.put(1)",
            "async def _consume(q):",
            "    await q.get()",
            "async def _consume_two(q):",
            "    await q.get()",
            "async def public_transport():",
            "    q = asyncio.Queue(maxsize=1)",
            "    async with asyncio.TaskGroup() as tg:",
            "        tg.create_task(_produce(q))",
            "        tg.create_task(public_consume(q))",
            "    return q",
            "async def multiple_consumer():",
            "    q = asyncio.Queue(maxsize=1)",
            "    async with asyncio.TaskGroup() as tg:",
            "        tg.create_task(_produce(q))",
            "        tg.create_task(_consume(q))",
            "        tg.create_task(_consume_two(q))",
            "async def unknown_transport():",
            "    async with asyncio.TaskGroup() as tg:",
            "        tg.create_task(_produce(q))",
            "        tg.create_task(_consume(q))",
            "async def ambiguous():",
            "    q = asyncio.Queue(maxsize=1)",
            "    async with asyncio.TaskGroup() as tg:",
            "        tg.create_task(factory(q))",
            "async def low_hotness():",
            "    q = asyncio.Queue(maxsize=1)",
            "    async with asyncio.TaskGroup() as tg:",
            "        tg.create_task(_produce(q))",
            "        tg.create_task(_consume(q))",
        ],
    )
    profile = _profile(
        scan,
        (
            ("unsafe", "public_transport", 2_000, 2_000),
            ("unsafe", "multiple_consumer", 2_000, 2_000),
            ("unsafe", "unknown_transport", 2_000, 2_000),
            ("unsafe", "ambiguous", 2_000, 2_000),
            ("unsafe", "low_hotness", 900, 900),
        ),
    )

    results = build_execution_plans((scan,), profile)
    rejections = {
        result.owner.qualname: result.reason
        for result in results
        if isinstance(result, PlanRejection)
    }

    assert rejections == {
        "ambiguous": "ambiguous-spawn",
        "low_hotness": "low-hotness",
        "multiple_consumer": "multiple-consumer",
        "public_transport": "public-transport",
        "unknown_transport": "unknown-transport",
    }


def test_discovery_reports_mixed_dialect_and_transport_argument_mismatches(
    tmp_path: Path,
) -> None:
    """Mixed schedulers and transports not passed to workers produce report rejections."""
    scan = _scan(
        tmp_path / "mixed.py",
        [
            "class Holder:",
            "    async def method(self):",
            "        return None",
            "async def _produce(q):",
            "    await q.put(1)",
            "async def _consume(q):",
            "    await q.get()",
            "async def _consume_receive(send):",
            "    await send.receive()",
            "async def _send(send):",
            "    await send.send(1)",
            "async def mixed_dialects():",
            "    q = asyncio.Queue(maxsize=1)",
            "    asyncio.create_task(_produce(q))",
            "    async with anyio.create_task_group() as tg:",
            "        tg.start_soon(_consume, q)",
            "async def unpassed_transport():",
            "    q = asyncio.Queue(maxsize=1)",
            "    async with asyncio.TaskGroup() as tg:",
            "        tg.create_task(_produce(other))",
            "        tg.create_task(_consume(other))",
            "async def annotated_queue():",
            "    q: asyncio.Queue = asyncio.Queue(maxsize=1)",
            "    async with asyncio.TaskGroup() as tg:",
            "        tg.create_task(_produce(q))",
            "        tg.create_task(_consume(q))",
            "async def memory_stream_receive():",
            "    send, receive = anyio.create_memory_object_stream(10)",
            "    async with anyio.create_task_group() as tg:",
            "        tg.start_soon(_send, send)",
            "        tg.start_soon(_consume_receive, receive)",
        ],
    )
    profile = _profile(
        scan,
        (
            ("mixed", "mixed_dialects", 2_000, 2_000),
            ("mixed", "unpassed_transport", 2_000, 2_000),
            ("mixed", "annotated_queue", 2_000, 2_000),
            ("mixed", "memory_stream_receive", 2_000, 2_000),
        ),
    )

    results = build_execution_plans((scan,), profile)
    plans = tuple(result for result in results if isinstance(result, ExecutionPlan))
    rejections = {
        result.owner.qualname: result.reason
        for result in results
        if isinstance(result, PlanRejection)
    }

    assert [plan.owner.qualname for plan in plans] == [
        "annotated_queue",
        "memory_stream_receive",
    ]
    assert rejections == {
        "mixed_dialects": "ambiguous-spawn",
        "unpassed_transport": "unknown-transport",
    }


def test_selection_limit_reports_remaining_hot_sites(tmp_path: Path) -> None:
    """Execution-plan selection emits report rejections after the four-plan limit."""
    lines = [
        "async def _produce(q):",
        "    await q.put(1)",
        "async def _consume(q):",
        "    await q.get()",
    ]
    members: list[tuple[str, str, int, int]] = []
    for index in range(_SELECTION_LIMIT + 1):
        name = f"run_{index}"
        lines.extend(
            [
                f"async def {name}():",
                "    q = asyncio.Queue(maxsize=1)",
                "    async with asyncio.TaskGroup() as tg:",
                "        tg.create_task(_produce(q))",
                "        tg.create_task(_consume(q))",
            ]
        )
        members.append(("limit", name, 2_000, 2_000))
    scan = _scan(tmp_path / "limit.py", lines)

    results = build_execution_plans((scan,), _profile(scan, tuple(members)))
    plans = tuple(result for result in results if isinstance(result, ExecutionPlan))
    rejections = tuple(result for result in results if isinstance(result, PlanRejection))

    assert len(plans) == _SELECTION_LIMIT
    assert [plan.owner.qualname for plan in plans] == ["run_0", "run_1", "run_2", "run_3"]
    assert [(rejection.owner.qualname, rejection.reason) for rejection in rejections] == [
        ("run_4", "selection-limit")
    ]


def test_bound_method_spawns_resolve_relative_to_the_owner_class(tmp_path: Path) -> None:
    """Generic scheduler discovery resolves `self` and `cls` task targets."""
    scan = _scan(
        tmp_path / "bound.py",
        [
            "class Runner:",
            "    async def _produce(self, q):",
            "        await q.put(1)",
            "    async def _consume(self, q):",
            "        await q.get()",
            "    async def run(self):",
            "        q = asyncio.Queue(maxsize=1)",
            "        async with asyncio.TaskGroup() as tg:",
            "            tg.create_task(self._produce(q))",
            "            tg.create_task(self._consume(q))",
        ],
    )
    profile = _profile(
        scan,
        (
            ("bound", "Runner.run", 2_000, 2_000),
            ("bound", "Runner._produce", 2_000, 2_000),
            ("bound", "Runner._consume", 2_000, 2_000),
        ),
    )

    results = build_execution_plans((scan,), profile)
    plan = next(result for result in results if isinstance(result, ExecutionPlan))

    assert plan.owner.qualname == "Runner.run"
    assert {node.symbol.qualname for node in plan.nodes if node.symbol is not None} == {
        "Runner.run",
        "Runner._produce",
        "Runner._consume",
    }


def test_unprofiled_sites_remain_report_only_rejections(tmp_path: Path) -> None:
    """Static compile reports candidates without treating them as eligible plans."""
    scan = _scan(
        tmp_path / "static.py",
        [
            "async def _produce(q):",
            "    await q.put(1)",
            "async def _consume(q):",
            "    await q.get()",
            "async def run():",
            "    q = asyncio.Queue(maxsize=1)",
            "    async with asyncio.TaskGroup() as tg:",
            "        tg.create_task(_produce(q))",
            "        tg.create_task(_consume(q))",
        ],
    )

    results = build_execution_plans((scan,), None)

    assert [
        (result.owner.qualname, result.reason)
        for result in results
        if isinstance(result, PlanRejection)
    ] == [("run", "low-hotness")]


def test_dialect_recognizers_reject_unknown_or_incomplete_calls() -> None:
    """Scheduler dialects ignore calls that do not carry supported spawn syntax."""
    expressions = ast.parse(
        "\n".join(
            [
                "asyncio.create_task(coro)",
                "other.create_task(_worker(q))",
                "factory[0](_worker(q))",
                "tg.start_later(_worker, q)",
                "value",
            ]
        )
    ).body
    asyncio_dialect = AsyncioDialect()
    anyio_dialect = AnyioOnAsyncioDialect()

    assert asyncio_dialect.recognize_spawn(_call_expression(expressions[0])) is None
    assert asyncio_dialect.recognize_spawn(_call_expression(expressions[1])) is not None
    assert asyncio_dialect.recognize_spawn(_call_expression(expressions[2])) is None
    assert anyio_dialect.recognize_spawn(_call_expression(expressions[3])) is None


def _scan(path: Path, lines: list[str]) -> ModuleScan:
    resolved_path = path if path.is_absolute() else Path.cwd() / path
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text("\n".join(lines), encoding="utf-8")
    return scan_module(ModuleId(name=resolved_path.stem, path=resolved_path))


def _call_expression(node: ast.stmt) -> ast.Call:
    assert isinstance(node, ast.Expr)
    assert isinstance(node.value, ast.Call)
    return node.value


def _profile(
    scan: ModuleScan,
    members: tuple[tuple[str, str, int, int], ...],
) -> ProfileResult:
    profiled = tuple(
        ProfiledMember(
            module=module,
            qualname=qualname,
            samples=0,
            coverage=0.0,
            call_count=call_count,
            invocation_count=call_count,
            lifecycle=LifecycleCounts(
                start=starts,
                return_=starts,
                yield_=0,
                resume=0,
                unwind=0,
                throw=0,
            ),
            signatures=(),
            polymorphic_overflow=False,
        )
        for module, qualname, call_count, starts in members
    )
    total_starts = sum(member.lifecycle.start for member in profiled)
    invocations_by_owner = {member.symbol: member.invocation_count for member in profiled}
    return ProfileResult(
        status="profiled",
        reason="test",
        launch_kind="script",
        total_samples=0,
        mapped_project_samples=0,
        mapped_coverage=0.0,
        selected_hot_samples=0,
        selected_hot_coverage=0.0,
        runs=(),
        lifecycle=LifecycleCounts(
            start=total_starts,
            return_=total_starts,
            yield_=0,
            resume=0,
            unwind=0,
            throw=0,
        ),
        members=profiled,
        candidates=(),
        selected_symbols=(),
        spawn_sites=tuple(
            ProfiledSpawnSite(
                target=target,
                invocation_count=invocations_by_owner.get(target.owner, 0),
                callable_identities=(
                    CanonicalCallableCount(
                        identity=(
                            f"anyio._backends._asyncio.TaskGroup.{target.scheduler_method}"
                            if target.scheduler_method == "start_soon"
                            else f"asyncio.taskgroups.TaskGroup.{target.scheduler_method}"
                        ),
                        count=invocations_by_owner.get(target.owner, 0),
                    ),
                ),
            )
            for target in execution_plan_profile_targets((scan,))
        ),
    )
