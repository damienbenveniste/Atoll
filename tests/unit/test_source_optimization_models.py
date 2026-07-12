"""Tests for immutable source-optimization model contracts."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path, PurePosixPath

import pytest

from atoll.models import SymbolId
from atoll.source_optimization import (
    SourceAccessSite,
    SourceCallableEvidence,
    SourceEdit,
    SourceOptimizationAssessment,
    SourceOptimizationIdentity,
    SourceOptimizationPlan,
    SourceOptimizationTrial,
    TransformationStep,
    stable_source_optimization_plan_id,
)


def _identity() -> SourceOptimizationIdentity:
    """Build a representative static identity for ID tests.

    Returns:
        SourceOptimizationIdentity: Static source-optimization identity inputs.
    """
    return SourceOptimizationIdentity(
        execution_plan_id="exec-plan-123",
        source_hashes=(
            (PurePosixPath("app/worker.py"), "hash-worker"),
            (PurePosixPath("app/reducer.py"), "hash-reducer"),
        ),
        topology_fingerprint="topology-v1",
        dialect="asyncio",
        lowering_version="source-lowering-v1",
        python_abi="cp312",
        transformation_versions=(
            ("private-transport-batch-drain", "batch-v1"),
            ("quiescent-callable-execution", "quiescent-v1"),
        ),
    )


def test_source_optimization_id_canonicalizes_reordered_source_hashes() -> None:
    """Stable IDs do not depend on caller ordering for covered source hashes."""
    identity = _identity()
    reordered = replace(identity, source_hashes=tuple(reversed(identity.source_hashes)))

    assert stable_source_optimization_plan_id(identity) == stable_source_optimization_plan_id(
        reordered
    )


def _identity_variants() -> tuple[SourceOptimizationIdentity, ...]:
    """Build identities that each alter one static ID input.

    Returns:
        SourceOptimizationIdentity: Static identity variants with one changed field each.
    """
    identity = _identity()
    return (
        replace(identity, execution_plan_id="exec-plan-456"),
        replace(identity, source_hashes=((PurePosixPath("app/worker.py"), "hash-worker-v2"),)),
        replace(identity, topology_fingerprint="topology-v2"),
        replace(identity, dialect="anyio"),
        replace(identity, lowering_version="source-lowering-v2"),
        replace(identity, python_abi="cp313"),
        replace(identity, transformation_versions=(("local-state-machine-fusion", "fusion-v1"),)),
    )


@pytest.mark.parametrize("changed", _identity_variants())
def test_source_optimization_id_changes_for_each_static_identity_input(
    changed: SourceOptimizationIdentity,
) -> None:
    """Every static identity field contributes to the stable source-plan ID."""
    identity = _identity()
    assert stable_source_optimization_plan_id(identity) != stable_source_optimization_plan_id(
        changed
    )


def test_source_optimization_models_are_frozen_slots_dataclasses(tmp_path: Path) -> None:
    """Source-optimization handoff objects are immutable and slot-backed."""
    owner = SymbolId(module="app.worker", qualname="run")
    worker = SymbolId(module="app.worker", qualname="_work")
    consumer = SymbolId(module="app.worker", qualname="_consume")
    access = SourceAccessSite(
        path=PurePosixPath("app/worker.py"),
        symbol=owner,
        kind="transport-receive",
        lineno=12,
        expression="self._queue",
        hazards=("observable-ordering",),
    )
    callable_evidence = SourceCallableEvidence(
        symbol=worker,
        static_role="worker",
        observed_invocations=50,
        median_seconds=0.02,
        hot_share=0.75,
    )
    step = TransformationStep(
        kind="private-transport-batch-drain",
        version="batch-v1",
        source_symbol=worker,
        target_symbol=None,
        access_sites=(access,),
        semantic_boundary="completion-order",
        description="Batch private queue drains while preserving completion order.",
    )
    identity = _identity()
    plan_id = stable_source_optimization_plan_id(identity)
    plan = SourceOptimizationPlan(
        id=plan_id,
        identity=identity,
        source=PurePosixPath("app/worker.py"),
        owner=owner,
        worker=worker,
        consumer=consumer,
        reducer=None,
        transport="self._queue",
        access_sites=(access,),
        entrypoint=owner,
        steps=(step,),
        semantic_boundaries=("completion-order",),
    )
    edit = SourceEdit(
        path=PurePosixPath("app/worker.py"),
        before_hash="before",
        after_hash="after",
        summary="rewrote private queue drain",
        touched_symbols=(worker,),
    )
    assessment = SourceOptimizationAssessment(
        plan_id=plan.id,
        status="trial-ready",
        minimum_speedup=3.0,
        work_items=(worker,),
        observed_work_items=50,
        immediate_result_ratio=0.8,
        attributed_hot_share=0.75,
        scheduler_overhead_samples=20,
        scheduler_overhead_share=0.25,
        scheduler_overhead_evidence=("queue get dominated runtime",),
        callable_evidence=(callable_evidence,),
        headroom_speedup=3.5,
    )
    trial = SourceOptimizationTrial(
        plan_id=plan.id,
        status="accepted",
        semantic_command=("pytest", "tests/unit"),
        benchmark_command=("python", "-m", "pytest", "benchmarks"),
        baseline_median_seconds=3.0,
        source_median_seconds=1.0,
        wheel_median_seconds=0.9,
        source_speedup=3.0,
        wheel_speedup=3.33,
        patch_path=tmp_path / "source-opt.patch",
        source_edits=(edit,),
        application_status="applied",
    )

    assert hasattr(plan, "__slots__")
    assert hasattr(assessment, "__slots__")
    assert trial.source_edits == (edit,)
    frozen_field = "transport"
    with pytest.raises(FrozenInstanceError):
        setattr(plan, frozen_field, "other")


def test_runtime_assessment_counts_do_not_affect_plan_identity() -> None:
    """Runtime assessment evidence can change without changing source-plan identity."""
    worker = SymbolId(module="app.worker", qualname="_work")
    plan_id = stable_source_optimization_plan_id(_identity())
    cold = SourceOptimizationAssessment(
        plan_id=plan_id,
        status="partial",
        minimum_speedup=3.0,
        work_items=(worker,),
        observed_work_items=2,
        immediate_result_ratio=0.1,
        attributed_hot_share=0.2,
        scheduler_overhead_samples=1,
        scheduler_overhead_share=0.1,
        scheduler_overhead_evidence=("few calls",),
        callable_evidence=(
            SourceCallableEvidence(
                symbol=worker,
                static_role="worker",
                observed_invocations=2,
                hot_share=0.2,
            ),
        ),
    )
    hot = replace(
        cold,
        observed_work_items=2_000,
        immediate_result_ratio=0.9,
        attributed_hot_share=0.95,
        scheduler_overhead_samples=1_000,
        scheduler_overhead_share=0.5,
        scheduler_overhead_evidence=("many calls",),
        callable_evidence=(
            SourceCallableEvidence(
                symbol=worker,
                static_role="worker",
                observed_invocations=2_000,
                hot_share=0.95,
            ),
        ),
    )

    assert cold != hot
    assert cold.plan_id == hot.plan_id == plan_id
