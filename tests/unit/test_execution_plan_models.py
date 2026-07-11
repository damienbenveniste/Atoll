"""Tests for immutable execution-plan contracts."""

from pathlib import Path, PurePosixPath

from atoll.execution_plans import (
    ChangedPayloadFile,
    ExecutionPlan,
    ExecutionPlanAssessmentContext,
    ExecutionPlanIdentity,
    ExecutionPlanStageContext,
    ExecutionPlanTrial,
    PlanEdge,
    PlanGuard,
    PlanNode,
    stable_execution_plan_id,
)
from atoll.models import SymbolId


def test_execution_plan_id_ignores_dynamic_profile_counts() -> None:
    """Stable plan IDs depend on source and topology inputs, not profile counts."""
    first = stable_execution_plan_id(
        ExecutionPlanIdentity(
            source_module="app.worker",
            source_hash="source",
            callsite_fingerprint="callsite",
            topology_fingerprint="topology",
            dialect="asyncio",
            lowering_version="asyncio-v1",
        )
    )
    second = stable_execution_plan_id(
        ExecutionPlanIdentity(
            source_module="app.worker",
            source_hash="source",
            callsite_fingerprint="callsite",
            topology_fingerprint="topology",
            dialect="asyncio",
            lowering_version="asyncio-v1",
        )
    )
    changed_topology = stable_execution_plan_id(
        ExecutionPlanIdentity(
            source_module="app.worker",
            source_hash="source",
            callsite_fingerprint="callsite",
            topology_fingerprint="topology-v2",
            dialect="asyncio",
            lowering_version="asyncio-v1",
        )
    )
    changed_callable = stable_execution_plan_id(
        ExecutionPlanIdentity(
            source_module="app.worker",
            source_hash="source",
            callsite_fingerprint="callsite",
            topology_fingerprint="topology",
            dialect="asyncio",
            lowering_version="asyncio-v1",
            guarded_callable_identities=("custom.TaskGroup.create_task",),
        )
    )

    assert first == second
    assert first != changed_topology
    assert first != changed_callable


def test_execution_plan_models_are_frozen_slots_dataclasses(tmp_path: Path) -> None:
    """Execution-plan handoff objects are immutable and slot-backed."""
    owner = SymbolId(module="app.worker", qualname="run")
    producer = SymbolId(module="app.worker", qualname="_produce")
    consumer = SymbolId(module="app.worker", qualname="_consume")
    plan = ExecutionPlan(
        id="exec-plan-1",
        source_module="app.worker",
        owner=owner,
        dialect="asyncio",
        lowering_version="asyncio-v1",
        source_hash="abc",
        callsite_fingerprint="def",
        topology_fingerprint="ghi",
        nodes=(
            PlanNode(id=owner.stable_id, symbol=owner, role="orchestrator", lineno=10),
            PlanNode(id=producer.stable_id, symbol=producer, role="producer", lineno=2),
            PlanNode(id=consumer.stable_id, symbol=consumer, role="consumer", lineno=6),
        ),
        edges=(
            PlanEdge(
                src=owner.stable_id,
                dst=producer.stable_id,
                kind="spawns",
                transport="queue",
                lineno=12,
            ),
        ),
        guards=(PlanGuard(kind="scheduler", expression="asyncio", message="asyncio semantics"),),
    )
    staged_file = ChangedPayloadFile(
        install_path=PurePosixPath("app/worker.py"),
        before_hash="before",
        after_hash="after",
        role="source",
    )
    trial = ExecutionPlanTrial(
        plan_id=plan.id,
        status="accepted",
        command=("pytest",),
        exit_code=0,
        duration_seconds=0.1,
    )
    assessment_context = ExecutionPlanAssessmentContext(
        project_root=tmp_path,
        source_root=tmp_path / "src",
        profile_status="profiled",
    )
    stage_context = ExecutionPlanStageContext(
        project_root=tmp_path,
        payload_root=tmp_path / ".atoll" / "payload",
        cache_root=tmp_path / ".atoll" / "cache",
    )

    assert hasattr(plan, "__slots__")
    assert hasattr(staged_file, "__slots__")
    assert trial.status == "accepted"
    assert assessment_context.source_root.name == "src"
    assert stage_context.cache_root.name == "cache"
