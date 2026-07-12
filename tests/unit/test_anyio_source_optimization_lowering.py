"""Tests for guarded source lowering of class-owned AnyIO result streams."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import inspect
import shutil
import sys
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from dataclasses import replace
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import cast

import pytest

from atoll.models import SymbolId
from atoll.source_optimization import (
    SourceCallableEvidence,
    SourceOptimizationAssessment,
    SourceOptimizationIdentity,
    SourceOptimizationPlan,
    TransformationStep,
    stable_source_optimization_plan_id,
)
from atoll.source_optimization.anyio_stream_lowering import lower_anyio_stream_plan
from atoll.source_optimization.lowering import lower_state_machine_plan
from atoll.source_optimization.transforms import (
    build_source_transformation_patch,
    materialize_transformed_files,
)

FIXTURE_ROOT = Path("tests/fixtures/source_optimization_project")
SOURCE_PATH = PurePosixPath("src/source_optimization_fixture/anyio_workflow.py")
MODULE = "source_optimization_fixture.anyio_workflow"
OWNER = SymbolId(MODULE, "PipelineRunner.submit")
WORKER = SymbolId(MODULE, "PipelineRunner._worker")
CONSUMER = SymbolId(MODULE, "PipelineRunner.results")


def test_anyio_lowering_builds_reproducible_libcst_patch() -> None:
    """The structural method pipeline produces a stable source-clean patch."""
    plan, assessment = _plan_and_assessment()
    source_path = FIXTURE_ROOT / SOURCE_PATH
    before = source_path.read_bytes()

    first = lower_state_machine_plan(FIXTURE_ROOT, plan, assessment)
    second = lower_state_machine_plan(FIXTURE_ROOT, plan, assessment)

    assert first == second
    assert first.status == "lowered"
    assert first.request is not None
    patch = build_source_transformation_patch(FIXTURE_ROOT, (first.request,))
    assert (
        patch.patch_text
        == build_source_transformation_patch(FIXTURE_ROOT, (first.request,)).patch_text
    )
    transformed = patch.files[0].after_source
    assert "copy_context" in transformed
    assert "receive_nowait" in transformed
    assert "if self.reducers:" in transformed
    assert "ATOLL_REQUIRE_OPTIMIZED" in transformed
    assert source_path.read_bytes() == before


def test_anyio_lowering_rejects_stale_source_identity() -> None:
    """A source hash mismatch stops before shape analysis or patch generation."""
    plan, _ = _plan_and_assessment()
    stale = replace(
        plan,
        identity=replace(
            plan.identity,
            source_hashes=((SOURCE_PATH, "0" * 64),),
        ),
    )

    with pytest.raises(ValueError, match="stale source"):
        lower_anyio_stream_plan(FIXTURE_ROOT, stale)


def test_anyio_lowering_rejects_stale_capacity_and_run_arity(tmp_path: Path) -> None:
    """Planned capacity and the one-item run signature remain exact source facts."""
    plan, _ = _plan_and_assessment()
    with pytest.raises(ValueError, match="capacity does not match"):
        lower_anyio_stream_plan(FIXTURE_ROOT, replace(plan, transport_capacity=1))

    copied_root = tmp_path / "project"
    shutil.copytree(FIXTURE_ROOT, copied_root)
    source_path = copied_root / SOURCE_PATH
    source = source_path.read_text(encoding="utf-8").replace(
        "async def _run_item(self, item: WorkItem)",
        "async def _run_item(self, item: WorkItem, extra: int = 0)",
    )
    source_path.write_text(source, encoding="utf-8")
    changed = replace(
        plan,
        identity=replace(
            plan.identity,
            source_hashes=((SOURCE_PATH, hashlib.sha256(source.encode()).hexdigest()),),
        ),
    )
    with pytest.raises(ValueError, match="run coroutine must accept only"):
        lower_anyio_stream_plan(copied_root, changed)


def test_strict_anyio_fast_path_matches_baseline_and_preserves_reflection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Immediate work uses copied-context routing without rebinding the class."""
    baseline = _load_module(FIXTURE_ROOT / SOURCE_PATH, "atoll_anyio_source_baseline")
    transformed, helpers = _transformed_module(tmp_path)
    baseline_run = _pipeline_callable(baseline)
    transformed_run = _pipeline_callable(transformed)
    baseline_worker = _worker_callable(baseline, "immediate_double")
    transformed_worker = _worker_callable(transformed, "immediate_double")
    values = tuple(range(32))
    monkeypatch.setenv("ATOLL_REQUIRE_OPTIMIZED", "1")

    expected = asyncio.run(baseline_run(values, baseline_worker))
    trace = sys.gettrace()
    profile = sys.getprofile()
    try:
        sys.settrace(None)
        sys.setprofile(None)
        observed = asyncio.run(transformed_run(values, transformed_worker))
    finally:
        sys.settrace(trace)
        sys.setprofile(profile)

    assert observed == expected
    assert vars(transformed)[helpers[-1]] == len(values)
    assert tuple(inspect.signature(transformed_run).parameters) == tuple(
        inspect.signature(baseline_run).parameters
    )
    assert (
        inspect.signature(transformed_run).return_annotation
        == inspect.signature(baseline_run).return_annotation
    )
    assert transformed_run.__annotations__ == baseline_run.__annotations__
    runner = cast(type[object], vars(transformed)["PipelineRunner"])
    assert runner.__module__ == transformed.__name__
    assert runner.__qualname__ == "PipelineRunner"


def test_suspending_callable_falls_back_before_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A callable containing suspension stays on the original AnyIO task path."""
    transformed, helpers = _transformed_module(tmp_path)
    run = _pipeline_callable(transformed)
    suspending = _worker_callable(transformed, "suspending_double")

    observed = asyncio.run(run((2, 3, 5), suspending))

    assert observed == (4, 6, 10)
    assert vars(transformed)[helpers[-1]] == 0
    monkeypatch.setenv("ATOLL_REQUIRE_OPTIMIZED", "1")
    with pytest.raises(BaseExceptionGroup, match="unhandled errors in a TaskGroup") as raised:
        asyncio.run(run((7,), suspending))
    assert any(
        isinstance(error, RuntimeError) and "AnyIO source guards failed" in str(error)
        for error in raised.value.exceptions
    )


def test_indirect_context_mutation_is_rejected_before_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recursive callable inspection catches a context-mutating global helper."""
    transformed, helpers = _transformed_module(tmp_path)
    run = _pipeline_callable(transformed)
    mutating = _worker_callable(transformed, "indirect_context_mutation")
    monkeypatch.setenv("ATOLL_REQUIRE_OPTIMIZED", "1")

    with pytest.raises(BaseExceptionGroup, match="unhandled errors in a TaskGroup") as raised:
        asyncio.run(run((11,), mutating))

    assert any(
        isinstance(error, RuntimeError) and "AnyIO source guards failed" in str(error)
        for error in raised.value.exceptions
    )
    assert vars(transformed)[helpers[-1]] == 0


def test_disabled_anyio_optimization_uses_original_scheduler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ATOLL_DISABLE retains the staged source implementation unchanged."""
    transformed, helpers = _transformed_module(tmp_path)
    run = _pipeline_callable(transformed)
    worker = _worker_callable(transformed, "immediate_double")
    monkeypatch.setenv("ATOLL_DISABLE", "1")

    assert asyncio.run(run((1, 4, 9), worker)) == (2, 8, 18)
    assert vars(transformed)[helpers[-1]] == 0


def test_replaced_anyio_scheduler_falls_back_before_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replaced TaskGroup scheduler cannot pass the captured identity guard."""
    transformed, helpers = _transformed_module(tmp_path)
    run = _pipeline_callable(transformed)
    worker = _worker_callable(transformed, "immediate_double")
    task_group_class = cast(
        type[object],
        next(
            value
            for name, value in vars(transformed).items()
            if name.endswith("_expected_task_group_class")
        ),
    )
    original_start = cast(
        Callable[..., None],
        next(
            value
            for name, value in vars(transformed).items()
            if name.endswith("_expected_start_soon")
        ),
    )
    scheduled = 0

    def replacement_start(
        self: object,
        function: Callable[..., object],
        *args: object,
        name: object = None,
    ) -> None:
        nonlocal scheduled
        scheduled += 1
        original_start(self, function, *args, name=name)

    monkeypatch.setattr(task_group_class, "start_soon", replacement_start)

    values = (2, 3, 5)
    assert asyncio.run(run(values, worker)) == (4, 6, 10)
    assert scheduled == len(values)
    assert vars(transformed)[helpers[-1]] == 0


def _plan_and_assessment() -> tuple[
    SourceOptimizationPlan,
    SourceOptimizationAssessment,
]:
    source = (FIXTURE_ROOT / SOURCE_PATH).read_text(encoding="utf-8")
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    identity = SourceOptimizationIdentity(
        execution_plan_id="exec-plan-generic-anyio-source",
        source_hashes=((SOURCE_PATH, source_hash),),
        topology_fingerprint="generic-anyio-private-stream-v1",
        dialect="anyio-on-asyncio",
        lowering_version="source-optimization-analysis-v1",
        python_abi="cp312",
        transformation_versions=(
            ("private-transport-batch-drain", "batch-drain-v1"),
            ("quiescent-callable-execution", "quiescent-callable-v1"),
            ("local-state-machine-fusion", "state-machine-v1"),
        ),
    )
    plan_id = stable_source_optimization_plan_id(identity)
    steps = tuple(
        TransformationStep(
            kind=kind,
            version=version,
            source_symbol=CONSUMER if kind == "private-transport-batch-drain" else OWNER,
            target_symbol=None,
            access_sites=(),
            semantic_boundary="private stream, copied context, and fallback before entry",
            description="Exercise one cumulative generic AnyIO source fast path.",
        )
        for kind, version in identity.transformation_versions
    )
    plan = SourceOptimizationPlan(
        id=plan_id,
        identity=identity,
        source=SOURCE_PATH,
        owner=OWNER,
        worker=WORKER,
        consumer=CONSUMER,
        reducer=None,
        transport="self.send_stream|self.receive_stream",
        access_sites=(),
        entrypoint=OWNER,
        steps=steps,
        semantic_boundaries=("FIFO", "copied Context", "fallback before entry"),
    )
    evidence = (
        SourceCallableEvidence(
            symbol=OWNER,
            static_role="owner",
            observed_invocations=20_000,
            completed_calls=20_000,
            immediate_result_ratio=1.0,
            hot_share=0.4,
        ),
        SourceCallableEvidence(
            symbol=WORKER,
            static_role="producer",
            observed_invocations=20_000,
            completed_calls=20_000,
            static_suspension_points=2,
            observed_suspensions=40_000,
            immediate_result_ratio=0.0,
            hot_share=0.4,
            cancellation=("CancelScope",),
        ),
    )
    assessment = SourceOptimizationAssessment(
        plan_id=plan_id,
        status="trial-ready",
        minimum_speedup=3.0,
        work_items=(WORKER,),
        observed_work_items=20_000,
        immediate_result_ratio=0.0,
        attributed_hot_share=0.8,
        scheduler_overhead_samples=10_000,
        scheduler_overhead_share=0.5,
        scheduler_overhead_evidence=("AnyIO task scheduling",),
        callable_evidence=evidence,
    )
    return plan, assessment


def _transformed_module(tmp_path: Path) -> tuple[ModuleType, tuple[str, ...]]:
    plan, assessment = _plan_and_assessment()
    lowering = lower_state_machine_plan(FIXTURE_ROOT, plan, assessment)
    assert lowering.request is not None
    patch = build_source_transformation_patch(FIXTURE_ROOT, (lowering.request,))
    copied_root = tmp_path / "project"
    shutil.copytree(FIXTURE_ROOT, copied_root)
    materialize_transformed_files(FIXTURE_ROOT, copied_root, patch)
    module = _load_module(
        copied_root / SOURCE_PATH,
        f"atoll_anyio_source_transformed_{tmp_path.name.replace('-', '_')}",
    )
    return module, lowering.helper_names


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load AnyIO source fixture: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _pipeline_callable(
    module: ModuleType,
) -> Callable[
    [Sequence[int], Callable[[int], Awaitable[int]]],
    Coroutine[object, object, tuple[int, ...]],
]:
    return cast(
        Callable[
            [Sequence[int], Callable[[int], Awaitable[int]]],
            Coroutine[object, object, tuple[int, ...]],
        ],
        vars(module)["run_pipeline"],
    )


def _worker_callable(
    module: ModuleType,
    name: str,
) -> Callable[[int], Awaitable[int]]:
    return cast(Callable[[int], Awaitable[int]], vars(module)[name])
