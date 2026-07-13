"""Tests for report-only profile-guided source-optimization planning."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Literal

from atoll.analysis.ast_scanner import scan_module
from atoll.execution_plans.models import ExecutionPlan, PlanEdge, PlanNode, PlanRejection
from atoll.models import CompileConfig, ModuleId, ModuleScan, SymbolId
from atoll.runtime.profiling import LifecycleCounts, ProfiledMember, ProfileResult
from atoll.source_optimization.analysis import (
    SourceOptimizationPlanningOptions,
    build_source_optimization_plans,
)

OBSERVED_WORK_ITEMS = 12_000
LOW_OBSERVED_WORK_ITEMS = 9_999
TOTAL_SAMPLES = 100
MINIMUM_SPEEDUP = 3.0
EXPECTED_HOT_SHARE = 1.0
EXPECTED_OVERHEAD_SAMPLES = 30
EXPECTED_PLAN_LIMIT = 2


def test_source_planner_forms_trial_ready_private_pipeline(tmp_path: Path) -> None:
    scan = _scan(tmp_path, introspection=False, forwarding=False)
    execution_plan = _execution_plan(scan, observed_work_items=OBSERVED_WORK_ITEMS)

    result = build_source_optimization_plans(
        (scan,),
        (execution_plan,),
        _planning_options(tmp_path, profile=_profile(immediate=True)),
    )

    assert len(result.plans) == 1
    plan = result.plans[0]
    assessment = result.assessments[0]
    assert plan.id.startswith("source-opt-")
    assert plan.identity.execution_plan_id == execution_plan.id
    assert plan.source.as_posix() == "pipeline.py"
    assert [step.kind for step in plan.steps] == [
        "private-transport-batch-drain",
        "quiescent-callable-execution",
        "local-state-machine-fusion",
    ]
    assert {site.kind for site in plan.access_sites} == {
        "transport-drain",
        "transport-receive",
        "transport-send",
    }
    assert assessment.status == "trial-ready"
    assert assessment.minimum_speedup == MINIMUM_SPEEDUP
    assert assessment.observed_work_items == OBSERVED_WORK_ITEMS
    assert assessment.immediate_result_ratio == 1.0
    assert assessment.attributed_hot_share == EXPECTED_HOT_SHARE
    assert assessment.scheduler_overhead_samples == EXPECTED_OVERHEAD_SAMPLES
    assert assessment.rejections == ()


def test_source_planner_reports_unbenchmarked_without_quality_commands(tmp_path: Path) -> None:
    scan = _scan(tmp_path, introspection=False, forwarding=False)

    result = build_source_optimization_plans(
        (scan,),
        (_execution_plan(scan, observed_work_items=OBSERVED_WORK_ITEMS),),
        _planning_options(
            tmp_path,
            profile=_profile(immediate=True),
            compile_config=CompileConfig(),
        ),
    )

    assessment = result.assessments[0]
    assert assessment.status == "unbenchmarked"
    assert assessment.rejections == (
        "source optimization requires configured test and benchmark commands",
    )


def test_source_planner_rejects_low_volume_share_and_suspension(tmp_path: Path) -> None:
    scan = _scan(tmp_path, introspection=False, forwarding=False)

    result = build_source_optimization_plans(
        (scan,),
        (_execution_plan(scan, observed_work_items=LOW_OBSERVED_WORK_ITEMS),),
        _planning_options(
            tmp_path,
            profile=_profile(immediate=False, attributed_samples=60, background_samples=40),
        ),
    )

    assessment = result.assessments[0]
    assert assessment.status == "unsupported"
    assert any("work items" in reason for reason in assessment.rejections)
    assert any("attributed hot share" in reason for reason in assessment.rejections)
    assert any("immediate-result ratio" in reason for reason in assessment.rejections)


def test_anyio_source_planner_defers_transport_suspension_to_runtime_guards(
    tmp_path: Path,
) -> None:
    """AnyIO producer send suspension does not poison a guarded source trial."""
    scan = _scan(tmp_path, introspection=False, forwarding=False)
    execution_plan = replace(
        _execution_plan(scan, observed_work_items=OBSERVED_WORK_ITEMS),
        dialect="anyio-on-asyncio",
    )

    result = build_source_optimization_plans(
        (scan,),
        (execution_plan,),
        _planning_options(tmp_path, profile=_profile(immediate=False)),
    )

    assessment = result.assessments[0]
    assert assessment.immediate_result_ratio == 0.0
    assert assessment.status == "trial-ready"
    assert not any("zero observed suspension" in reason for reason in assessment.rejections)
    assert [step.kind for step in result.plans[0].steps[-5:]] == [
        "run-scoped-guard-amortization",
        "transparent-quiescent-await-chain-collapse",
        "context-copy-elision",
        "incremental-private-completion-accounting",
        "private-result-record-elision",
    ]


def test_source_planner_rejects_task_introspection(tmp_path: Path) -> None:
    scan = _scan(tmp_path, introspection=True, forwarding=False)

    result = build_source_optimization_plans(
        (scan,),
        (_execution_plan(scan, observed_work_items=OBSERVED_WORK_ITEMS),),
        _planning_options(tmp_path, profile=_profile(immediate=True)),
    )

    assessment = result.assessments[0]
    producer = next(item for item in assessment.callable_evidence if item.static_role == "producer")
    assert assessment.status == "unsupported"
    assert producer.task_introspection == ("asyncio.current_task",)
    assert "task-introspection" in producer.hazards
    assert any("task introspection" in reason for reason in assessment.rejections)


def test_source_planner_rejects_indirect_context_mutation(tmp_path: Path) -> None:
    scan = _scan(
        tmp_path,
        introspection=False,
        forwarding=False,
        context_mutation="indirect",
    )

    result = build_source_optimization_plans(
        (scan,),
        (_execution_plan(scan, observed_work_items=OBSERVED_WORK_ITEMS),),
        _planning_options(tmp_path, profile=_profile(immediate=True)),
    )

    assessment = result.assessments[0]
    dependency = next(
        item for item in assessment.callable_evidence if item.symbol.qualname == "_mutate_context"
    )
    assert assessment.status == "unsupported"
    assert dependency.static_role == "dependency"
    assert dependency.context_mutation == ("_CONTEXT_LABEL.set",)
    assert any("context mutation" in reason for reason in assessment.rejections)


def test_source_planner_allows_direct_worker_context_mutation(tmp_path: Path) -> None:
    """Copied-context lowering can isolate mutation performed directly by the worker."""
    scan = _scan(
        tmp_path,
        introspection=False,
        forwarding=False,
        context_mutation="direct",
    )

    result = build_source_optimization_plans(
        (scan,),
        (_execution_plan(scan, observed_work_items=OBSERVED_WORK_ITEMS),),
        _planning_options(tmp_path, profile=_profile(immediate=True)),
    )

    assessment = result.assessments[0]
    producer = next(item for item in assessment.callable_evidence if item.static_role == "producer")
    assert producer.context_mutation == ("_CONTEXT_LABEL.set",)
    assert assessment.status == "trial-ready"


def test_source_planner_reports_missing_profile_and_ignores_unusable_plans(tmp_path: Path) -> None:
    scan = _scan(tmp_path, introspection=False, forwarding=False)
    plan = _execution_plan(scan, observed_work_items=OBSERVED_WORK_ITEMS)
    rejection = PlanRejection(
        id="rejected-plan",
        source_module="pipeline",
        owner=plan.owner,
        reason="low-hotness",
        message="not hot",
        dialect="asyncio",
        lineno=1,
        hotness=0,
    )
    missing_scan_plan = replace(plan, id="missing-plan", source_module="missing")

    ignored = build_source_optimization_plans(
        (scan,),
        (rejection, missing_scan_plan),
        SourceOptimizationPlanningOptions(
            profile=None,
            compile_config=_compile_config(),
            project_root=tmp_path,
            python_abi="cp312",
        ),
    )
    unprofiled = build_source_optimization_plans(
        (scan,),
        (plan,),
        SourceOptimizationPlanningOptions(
            profile=None,
            compile_config=_compile_config(),
            project_root=tmp_path,
            python_abi="cp312",
        ),
    )
    owner_only_plan = replace(
        plan,
        id="owner-only-plan",
        nodes=tuple(node for node in plan.nodes if node.role in {"orchestrator", "transport"}),
        consumer=None,
        reducer=plan.owner,
        source_members=(plan.owner,),
    )
    scan.module.path.unlink()
    missing_source = build_source_optimization_plans(
        (scan,),
        (owner_only_plan,),
        SourceOptimizationPlanningOptions(
            profile=None,
            compile_config=_compile_config(),
            project_root=tmp_path / "different-root",
            python_abi="cp312",
        ),
    )

    assert ignored.plans == ()
    assert unprofiled.assessments[0].status == "unbenchmarked"
    assert any(
        "current-invocation profile" in reason for reason in unprofiled.assessments[0].rejections
    )
    assert missing_source.plans[0].worker == plan.owner
    assert missing_source.plans[0].source.as_posix() == "pipeline.py"


def test_source_planner_rejects_missing_transport_and_unsafe_worker(tmp_path: Path) -> None:
    scan = _scan(
        tmp_path,
        introspection=False,
        forwarding=False,
        unsafe_worker=True,
    )
    original = _execution_plan(scan, observed_work_items=OBSERVED_WORK_ITEMS)
    ghost = SymbolId("pipeline", "missing_helper")
    plan = replace(
        original,
        nodes=(
            *(node for node in original.nodes if node.role != "consumer"),
            PlanNode(id=ghost.stable_id, symbol=ghost, role="worker", lineno=1),
        ),
        completion_transport=None,
        consumer=None,
        source_members=(original.owner, SymbolId("pipeline", "_producer")),
    )

    result = build_source_optimization_plans(
        (scan,),
        (plan,),
        _planning_options(tmp_path, profile=_profile(immediate=True)),
    )

    assessment = result.assessments[0]
    producer = next(
        item for item in assessment.callable_evidence if item.symbol.qualname == "_producer"
    )
    assert assessment.status == "unsupported"
    assert producer.static_suspension_points == 1
    assert producer.cancellation == ("task.cancel",)
    assert producer.unknown_dynamic_calls == ("eval",)
    assert any("no statically owned private" in reason for reason in assessment.rejections)
    assert any("no private consumer receive" in reason for reason in assessment.rejections)
    assert any("non-transport suspension" in reason for reason in assessment.rejections)
    assert any("cancellation" in reason for reason in assessment.rejections)
    assert any("dynamic calls" in reason for reason in assessment.rejections)


def test_source_planner_detects_simple_private_protocol_forwarder(tmp_path: Path) -> None:
    scan = _scan(tmp_path, introspection=False, forwarding=True)

    result = build_source_optimization_plans(
        (scan,),
        (_execution_plan(scan, observed_work_items=OBSERVED_WORK_ITEMS),),
        _planning_options(tmp_path, profile=_profile(immediate=True)),
    )

    plan = result.plans[0]
    assert plan.entrypoint.qualname == "stream"
    assert any(step.kind == "private-protocol-auto-forwarding" for step in plan.steps)
    assert plan.steps[-1].kind == "private-protocol-auto-forwarding"


def test_source_planner_ranks_and_caps_candidates(tmp_path: Path) -> None:
    scan = _scan(tmp_path, introspection=False, forwarding=False)
    plans = tuple(
        replace(
            _execution_plan(scan, observed_work_items=OBSERVED_WORK_ITEMS - index),
            id=f"exec-plan-{index}",
            lifecycle_share=0.9 - index / 10,
        )
        for index in range(3)
    )

    result = build_source_optimization_plans(
        (scan,),
        plans,
        _planning_options(tmp_path, profile=_profile(immediate=True)),
    )

    assert len(result.plans) == EXPECTED_PLAN_LIMIT
    assert [plan.identity.execution_plan_id for plan in result.plans] == [
        "exec-plan-0",
        "exec-plan-1",
    ]


def _scan(
    tmp_path: Path,
    *,
    introspection: bool,
    forwarding: bool,
    context_mutation: Literal["none", "direct", "indirect"] = "none",
    unsafe_worker: bool = False,
) -> ModuleScan:
    source = [
        "import asyncio",
    ]
    if context_mutation != "none":
        source.extend(
            [
                "from contextvars import ContextVar",
                "_CONTEXT_LABEL = ContextVar('_CONTEXT_LABEL', default='parent')",
            ]
        )
    if context_mutation == "indirect":
        source.extend(
            [
                "",
                "def _mutate_context():",
                "    _CONTEXT_LABEL.set('child')",
            ]
        )
    source.extend(["", "async def _producer(queue, value):"])
    if introspection:
        source.append("    asyncio.current_task()")
    if context_mutation == "indirect":
        source.append("    _mutate_context()")
    if context_mutation == "direct":
        source.append("    _CONTEXT_LABEL.set('child')")
    if unsafe_worker:
        source.extend(
            [
                "    await asyncio.sleep(0)",
                "    task = asyncio.current_task()",
                "    task.cancel()",
                "    eval('value')",
            ]
        )
    source.extend(
        [
            "    result = value + 1",
            "    await queue.put(result)",
            "",
            "async def _consume(queue):",
            "    try:",
            "        return queue.get_nowait()",
            "    except asyncio.QueueEmpty:",
            "        return await queue.get()",
            "",
            "async def run(values):",
            "    queue = asyncio.Queue(maxsize=1)",
            "    async with asyncio.TaskGroup() as group:",
            "        for value in values:",
            "            group.create_task(_producer(queue, value))",
            "        return await _consume(queue)",
        ]
    )
    if forwarding:
        source.extend(
            [
                "",
                "async def stream(values):",
                "    async for event in run(values):",
                "        yield event",
            ]
        )
    path = tmp_path / "pipeline.py"
    path.write_text("\n".join(source) + "\n", encoding="utf-8")
    return scan_module(ModuleId(name="pipeline", path=path))


def _execution_plan(scan: ModuleScan, *, observed_work_items: int) -> ExecutionPlan:
    owner = SymbolId("pipeline", "run")
    producer = SymbolId("pipeline", "_producer")
    consumer = SymbolId("pipeline", "_consume")
    helper = SymbolId("pipeline", "_mutate_context")
    source_hash = hashlib.sha256(scan.module.path.read_bytes()).hexdigest()
    return ExecutionPlan(
        id="exec-plan-pipeline",
        source_module="pipeline",
        owner=owner,
        dialect="asyncio",
        lowering_version="asyncio-v1",
        source_hash=source_hash,
        callsite_fingerprint="callsite",
        topology_fingerprint="topology",
        nodes=(
            PlanNode(id=owner.stable_id, symbol=owner, role="orchestrator", lineno=14),
            PlanNode(id=producer.stable_id, symbol=producer, role="producer", lineno=3),
            PlanNode(id="transport", symbol=None, role="transport", lineno=15),
            PlanNode(id=consumer.stable_id, symbol=consumer, role="consumer", lineno=8),
        ),
        edges=(
            PlanEdge(
                src=owner.stable_id,
                dst=producer.stable_id,
                kind="spawns",
                transport="queue",
                lineno=18,
            ),
            PlanEdge(
                src=producer.stable_id,
                dst="transport",
                kind="produces",
                transport="queue",
                lineno=6,
            ),
            PlanEdge(
                src="transport",
                dst=consumer.stable_id,
                kind="delivers",
                transport="queue",
                lineno=11,
            ),
        ),
        guards=(),
        completion_transport="queue",
        consumer=consumer,
        observed_invocations=observed_work_items,
        lifecycle_starts=observed_work_items,
        lifecycle_share=0.9,
        source_members=(
            owner,
            producer,
            consumer,
            *(symbol.id for symbol in scan.symbols if symbol.id == helper),
        ),
        source_hashes=(("pipeline", source_hash),),
        hotness=observed_work_items,
    )


def _profile(
    *,
    immediate: bool,
    attributed_samples: int = 90,
    background_samples: int = 0,
) -> ProfileResult:
    owner_samples = attributed_samples * 4 // 9
    producer_samples = attributed_samples // 3
    consumer_samples = attributed_samples - owner_samples - producer_samples
    plan_members = (
        _member("run", owner_samples, 0, 0),
        _member(
            "_producer",
            producer_samples,
            OBSERVED_WORK_ITEMS,
            0 if immediate else OBSERVED_WORK_ITEMS,
        ),
        _member("_consume", consumer_samples, OBSERVED_WORK_ITEMS, 0),
    )
    members = (
        *plan_members,
        *((_member("cold_background", background_samples, 0, 0),) if background_samples else ()),
    )
    return ProfileResult(
        status="profiled",
        reason="synthetic source-optimization profile",
        launch_kind="script",
        total_samples=TOTAL_SAMPLES,
        mapped_project_samples=sum(member.samples for member in members),
        mapped_coverage=attributed_samples / TOTAL_SAMPLES,
        selected_hot_samples=attributed_samples,
        selected_hot_coverage=attributed_samples / TOTAL_SAMPLES,
        runs=(),
        lifecycle=LifecycleCounts(
            start=OBSERVED_WORK_ITEMS,
            return_=OBSERVED_WORK_ITEMS,
            yield_=0 if immediate else OBSERVED_WORK_ITEMS,
            resume=0,
            unwind=0,
            throw=0,
        ),
        members=members,
        candidates=(),
        selected_symbols=(SymbolId("pipeline", "run"),),
        scheduler_overhead_samples=sum(member.scheduler_overhead_samples for member in members),
        scheduler_overhead_coverage=0.4,
    )


def _member(
    qualname: str,
    attributed_samples: int,
    invocations: int,
    suspensions: int,
) -> ProfiledMember:
    leaf_samples = max(0, attributed_samples - 10)
    overhead_samples = attributed_samples - leaf_samples
    return ProfiledMember(
        module="pipeline",
        qualname=qualname,
        samples=leaf_samples,
        coverage=leaf_samples / TOTAL_SAMPLES,
        scheduler_overhead_samples=overhead_samples,
        scheduler_overhead_coverage=overhead_samples / TOTAL_SAMPLES,
        call_count=invocations,
        invocation_count=invocations,
        lifecycle=LifecycleCounts(
            start=invocations,
            return_=invocations,
            yield_=suspensions,
            resume=suspensions,
            unwind=0,
            throw=0,
        ),
        signatures=(),
        polymorphic_overflow=False,
        completed_calls=invocations,
        max_active_calls=1 if invocations else 0,
        pre_completion_suspensions=suspensions,
    )


def _compile_config() -> CompileConfig:
    return CompileConfig(
        test_command=("pytest", "-q"),
        benchmark_command=("python", "bench.py"),
    )


def _planning_options(
    root: Path,
    *,
    profile: ProfileResult,
    compile_config: CompileConfig | None = None,
) -> SourceOptimizationPlanningOptions:
    return SourceOptimizationPlanningOptions(
        profile=profile,
        compile_config=compile_config or _compile_config(),
        project_root=root,
        python_abi="cp312",
    )
