"""Tests for guarded batch-drain and copied-context source lowering."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import inspect
import shutil
import sys
from collections.abc import Callable, Coroutine, Mapping
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
from atoll.source_optimization.lowering import SourceLoweringResult, lower_batch_quiescent_plan
from atoll.source_optimization.transforms import (
    build_source_transformation_patch,
    materialize_transformed_files,
)

FIXTURE_ROOT = Path("tests/fixtures/source_optimization_project")
SOURCE_PATH = PurePosixPath("src/source_optimization_fixture/workflow.py")
OWNER = SymbolId("source_optimization_fixture.workflow", "_run_hot_private_pipeline")
WORKER = SymbolId("source_optimization_fixture.workflow", "_immediate_worker")
DEPENDENCY = SymbolId("source_optimization_fixture.workflow", "_make_record")


def test_lowering_builds_reproducible_patch_without_mutating_checkout() -> None:
    """A supported plan produces stable LibCST output and leaves source untouched."""
    plan, assessment = _plan_and_assessment()
    source_file = FIXTURE_ROOT / SOURCE_PATH
    before = source_file.read_bytes()

    first = lower_batch_quiescent_plan(FIXTURE_ROOT, plan, assessment)
    second = lower_batch_quiescent_plan(FIXTURE_ROOT, plan, assessment)

    assert first == second
    assert first.status == "lowered"
    assert first.request is not None
    patch = build_source_transformation_patch(FIXTURE_ROOT, (first.request,))
    assert (
        patch.patch_text
        == build_source_transformation_patch(FIXTURE_ROOT, (first.request,)).patch_text
    )
    assert "contextvars.copy_context" in patch.files[0].after_source
    assert "ATOLL_REQUIRE_OPTIMIZED" in patch.files[0].after_source
    assert source_file.read_bytes() == before


def test_strict_transformed_pipeline_matches_baseline_and_reflection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The copied-context fast path preserves results, metadata, and parent context."""
    baseline = _load_workflow(FIXTURE_ROOT / SOURCE_PATH, "atoll_source_baseline")
    transformed, helper_names = _transformed_workflow(tmp_path)
    baseline_callable = _hot_pipeline(baseline)
    transformed_callable = _hot_pipeline(transformed)
    expected = asyncio.run(baseline_callable())
    monkeypatch.setenv("ATOLL_REQUIRE_OPTIMIZED", "1")

    observed = asyncio.run(transformed_callable())
    baseline_snapshot = asyncio.run(_async_mapping(baseline, "canonical_semantic_snapshot")())
    transformed_snapshot = asyncio.run(_async_mapping(transformed, "canonical_semantic_snapshot")())

    assert observed == expected
    assert transformed_snapshot == baseline_snapshot
    assert inspect.signature(transformed_callable) == inspect.signature(baseline_callable)
    assert transformed_callable.__annotations__ == baseline_callable.__annotations__
    assert transformed_callable.__doc__ == baseline_callable.__doc__
    assert all(name in vars(transformed) for name in helper_names)


def test_disable_uses_original_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`ATOLL_DISABLE=1` bypasses the generated fast helper."""
    baseline = _load_workflow(FIXTURE_ROOT / SOURCE_PATH, "atoll_source_disable_baseline")
    transformed, helper_names = _transformed_workflow(tmp_path)
    fast_name = helper_names[1]

    def fail_fast() -> object:
        raise AssertionError("disabled source optimization entered fast helper")

    monkeypatch.setitem(vars(transformed), fast_name, fail_fast)
    monkeypatch.setenv("ATOLL_DISABLE", "1")

    assert asyncio.run(_hot_pipeline(transformed)()) == asyncio.run(_hot_pipeline(baseline)())


def test_strict_guard_failure_happens_before_worker_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changed worker identity rejects strict routing before pipeline entry."""
    transformed, _helper_names = _transformed_workflow(tmp_path)
    calls = 0

    async def replacement(queue: object, item: object) -> None:
        nonlocal calls
        del queue, item
        calls += 1

    monkeypatch.setitem(vars(transformed), "_immediate_worker", replacement)
    monkeypatch.setenv("ATOLL_REQUIRE_OPTIMIZED", "1")

    with pytest.raises(RuntimeError, match="source guards failed"):
        asyncio.run(_hot_pipeline(transformed)())

    assert calls == 0


def test_fast_path_failure_never_retries_original_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure after optimized entry surfaces without invoking fallback."""
    transformed, helper_names = _transformed_workflow(tmp_path)
    original_name, _fast_name, _guard_name, _drive_name, batch_name = helper_names
    fallback_calls = 0

    async def counted_original() -> object:
        nonlocal fallback_calls
        fallback_calls += 1
        return object()

    def failed_batch(queue: object, count: object) -> object:
        del queue, count
        raise RuntimeError("forced batch failure")

    namespace = vars(transformed)
    monkeypatch.setitem(namespace, original_name, counted_original)
    monkeypatch.setitem(namespace, batch_name, failed_batch)
    monkeypatch.setenv("ATOLL_REQUIRE_OPTIMIZED", "1")

    with pytest.raises(RuntimeError, match="forced batch failure"):
        asyncio.run(_hot_pipeline(transformed)())

    assert fallback_calls == 0


def test_runtime_guard_rejects_custom_task_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observable task-construction policy keeps the original task path active."""
    transformed, helper_names = _transformed_workflow(tmp_path)
    guard = cast(Callable[[], bool], vars(transformed)[helper_names[2]])

    def task_factory_marker() -> object:
        return object()

    async def evaluate_guard() -> bool:
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(loop, "get_task_factory", task_factory_marker)
        return guard()

    assert asyncio.run(evaluate_guard()) is False


@pytest.mark.parametrize(
    ("changed_field", "reason"),
    [
        ("status", "not trial-ready"),
        ("immediate-result-ratio", "100% immediate-result ratio"),
    ],
)
def test_lowering_rejects_unproven_assessments(
    changed_field: str,
    reason: str,
) -> None:
    """Static lowering never weakens profile or capability gates."""
    plan, assessment = _plan_and_assessment()
    changed = (
        replace(assessment, status="unsupported")
        if changed_field == "status"
        else replace(assessment, immediate_result_ratio=0.99)
    )

    result = lower_batch_quiescent_plan(FIXTURE_ROOT, plan, changed)

    assert result.status == "unsupported"
    assert result.request is None
    assert any(reason in rejection for rejection in result.rejections)


def test_lowering_rejects_indirect_context_mutation_dependency() -> None:
    """Copied context permits direct worker mutation but not opaque dependency mutation."""
    plan, assessment = _plan_and_assessment()
    indirect = replace(
        assessment.callable_evidence[1],
        context_mutation=("_UNSUPPORTED_CONTEXT.set",),
    )
    changed = replace(
        assessment,
        callable_evidence=(assessment.callable_evidence[0], indirect),
    )

    result = lower_batch_quiescent_plan(FIXTURE_ROOT, plan, changed)

    assert result.status == "unsupported"
    assert any("mutates context indirectly" in reason for reason in result.rejections)


def test_preflight_reports_plan_and_worker_evidence_mismatches() -> None:
    """Backend selection rejects mismatched plans, dialects, steps, and missing workers."""
    plan, assessment = _plan_and_assessment()
    changed_identity = replace(plan.identity, dialect="anyio")
    changed_plan = replace(plan, identity=changed_identity, steps=())
    changed_assessment = replace(
        assessment,
        plan_id="different-plan",
        callable_evidence=(assessment.callable_evidence[1],),
    )

    result = lower_batch_quiescent_plan(FIXTURE_ROOT, changed_plan, changed_assessment)

    assert result.status == "unsupported"
    assert any("different plan" in reason for reason in result.rejections)
    assert any("does not support dialect" in reason for reason in result.rejections)
    assert any("lacks batch-drain" in reason for reason in result.rejections)
    assert any("no worker evidence" in reason for reason in result.rejections)


def test_preflight_reports_worker_runtime_hazards() -> None:
    """Suspension, task, cancellation, and dynamic-call evidence each block lowering."""
    plan, assessment = _plan_and_assessment()
    hazardous_worker = replace(
        assessment.callable_evidence[0],
        static_suspension_points=1,
        observed_suspensions=1,
        task_introspection=("asyncio.current_task",),
        cancellation=("task.cancel",),
        unknown_dynamic_calls=("eval",),
    )
    owner_evidence = SourceCallableEvidence(symbol=OWNER, static_role="owner")
    changed = replace(
        assessment,
        callable_evidence=(hazardous_worker, owner_evidence),
    )

    result = lower_batch_quiescent_plan(FIXTURE_ROOT, plan, changed)

    assert any("can suspend" in reason for reason in result.rejections)
    assert any("observes task state" in reason for reason in result.rejections)
    assert any("uses cancellation APIs" in reason for reason in result.rejections)
    assert any("unknown dynamic calls" in reason for reason in result.rejections)


@pytest.mark.parametrize(
    ("old", "new", "reason"),
    [
        (
            "async with asyncio.TaskGroup() as group:",
            "async with asyncio.timeout(1):",
            "exactly one top-level asyncio.TaskGroup",
        ),
        (
            "async with asyncio.TaskGroup() as group:",
            "async with asyncio.TaskGroup() as (group,):",
            "bind one local group name",
        ),
        (
            "        for item in items:\n",
            "        pass\n        for item in items:\n",
            "one spawn loop and one receive",
        ),
        (
            "            group.create_task(_immediate_worker(queue, item))\n        records =",
            "            group.create_task(_immediate_worker(queue, item))\n"
            "        else:\n"
            "            pass\n"
            "        records =",
            "no else branch",
        ),
        (
            "            group.create_task(_immediate_worker(queue, item))",
            "            pending = _immediate_worker(queue, item)",
            "one create_task call",
        ),
        (
            "        for item in items:\n"
            "            group.create_task(_immediate_worker(queue, item))\n"
            "        records = [await queue.get() for _ in range(len(items))]",
            "        records = [await queue.get() for _ in range(len(items))]\n"
            "        for item in items:\n"
            "            group.create_task(_immediate_worker(queue, item))",
            "spawn before receiving",
        ),
        (
            "group.create_task(_immediate_worker(queue, item))",
            "group.create_task(_immediate_worker(queue, item), name='work')",
            "without options",
        ),
        (
            "group.create_task(_immediate_worker(queue, item))",
            "group.create_task(cold_suspending_worker(queue, item))",
            "exact two-argument module callable",
        ),
        (
            "group.create_task(_immediate_worker(queue, item))",
            "group.create_task(_immediate_worker(queue, items))",
            "task loop target",
        ),
        (
            "        records = [await queue.get() for _ in range(len(items))]",
            "        records, = [await queue.get() for _ in range(len(items))]",
            "assign one local result list",
        ),
        (
            "[await queue.get() for _ in range(len(items))]",
            "tuple(await queue.get() for _ in range(len(items)))",
            "one queue list comprehension",
        ),
        (
            "for _ in range(len(items))]",
            "for _ in range(len(items)) if items]",
            "cannot be async or filtered",
        ),
        (
            "await queue.get()",
            "queue.get_nowait()",
            "must await the owned queue",
        ),
        (
            "for _ in range(len(items))]",
            "for _ in enumerate(items)]",
            "must use range(count)",
        ),
        (
            "queue: asyncio.Queue[WorkerRecord] = asyncio.Queue(maxsize=len(items))",
            "queue: asyncio.Queue[WorkerRecord] = asyncio.LifoQueue(maxsize=len(items))",
            "exact Queue type",
        ),
        (
            "queue: asyncio.Queue[WorkerRecord] = asyncio.Queue(maxsize=len(items))",
            "transport: asyncio.Queue[WorkerRecord] = asyncio.Queue(maxsize=len(items))",
            "one local constructor assignment",
        ),
        (
            "asyncio.Queue(maxsize=len(items))",
            "asyncio.Queue(maxsize=len(items) + 1)",
            "same expression",
        ),
        (
            "len(items)",
            "WORK_ITEM_COUNT",
            "must be len(task_iterable)",
        ),
        (
            "asyncio.Queue(maxsize=len(items))",
            "asyncio.Queue(len(items), loop=None)",
            "one explicit maxsize",
        ),
        (
            "    async with asyncio.TaskGroup() as group:",
            "    await asyncio.sleep(0)\n    async with asyncio.TaskGroup() as group:",
            "may await only its private queue receive",
        ),
        (
            "    async with asyncio.TaskGroup() as group:",
            "    yield None\n    async with asyncio.TaskGroup() as group:",
            "cannot be a generator",
        ),
        (
            "async def _immediate_worker(queue: asyncio.Queue[WorkerRecord], item: WorkItem)",
            "async def _immediate_worker(\n"
            "    queue: asyncio.Queue[WorkerRecord], item: WorkItem, flag: bool = False\n"
            ")",
            "exactly two positional parameters",
        ),
        (
            "async def _immediate_worker(queue: asyncio.Queue[WorkerRecord], item: WorkItem)",
            "async def _immediate_worker(q: asyncio.Queue[WorkerRecord], item: WorkItem)",
            "parameters must match",
        ),
        (
            "    _WORKER_CONTEXT.set(f\"worker:{item['label']}\")",
            "    await asyncio.sleep(0)\n    _WORKER_CONTEXT.set(f\"worker:{item['label']}\")",
            "suspension or nonlocal state",
        ),
        (
            "    queue.put_nowait(_make_record(item))",
            "    record = _make_record(item)",
            "publish exactly one",
        ),
        (
            "    queue.put_nowait(_make_record(item))",
            "    queue.put_nowait(_make_record(item))\n    return None",
            "publication must be its final",
        ),
        (
            "    _WORKER_CONTEXT.set(f\"worker:{item['label']}\")",
            "    item['weight'] = 0\n    _WORKER_CONTEXT.set(f\"worker:{item['label']}\")",
            "mutates attribute or container state",
        ),
        (
            "    _WORKER_CONTEXT.set(f\"worker:{item['label']}\")",
            "    item['weight'] += 1\n    _WORKER_CONTEXT.set(f\"worker:{item['label']}\")",
            "mutates attribute or container state",
        ),
    ],
)
def test_lowering_rejects_unsafe_source_shapes(
    tmp_path: Path,
    old: str,
    new: str,
    reason: str,
) -> None:
    """Every unsupported topology shape stays on the original implementation."""
    source = (FIXTURE_ROOT / SOURCE_PATH).read_text(encoding="utf-8")
    assert old in source

    result = _lower_source(tmp_path, source.replace(old, new))

    assert result.status == "unsupported"
    assert any(reason in rejection for rejection in result.rejections)


def test_lowering_supports_positional_queue_capacity_and_owner_docstring(
    tmp_path: Path,
) -> None:
    """Equivalent queue syntax and source documentation survive lowering."""
    source = (FIXTURE_ROOT / SOURCE_PATH).read_text(encoding="utf-8")
    source = source.replace(
        "async def _run_hot_private_pipeline() -> ReductionResult:\n",
        "async def _run_hot_private_pipeline() -> ReductionResult:\n"
        '    """Hot source documentation."""\n',
    ).replace("asyncio.Queue(maxsize=len(items))", "asyncio.Queue(len(items))")

    result = _lower_source(tmp_path, source)

    assert result.status == "lowered"
    assert result.request is not None
    assert "Hot source documentation" in result.request.replacement_body


def test_lowering_supports_owner_argument_forwarding_and_unrelated_with(
    tmp_path: Path,
) -> None:
    """Generated fallback forwards every parameter and preserves unrelated contexts."""
    source = (FIXTURE_ROOT / SOURCE_PATH).read_text(encoding="utf-8")
    source = source.replace(
        "async def _run_hot_private_pipeline() -> ReductionResult:\n",
        "async def _run_hot_private_pipeline(\n"
        "    marker: object = None, *args: object, option: int = 1, **kwargs: object\n"
        ") -> ReductionResult:\n"
        "    with contextlib.nullcontext():\n"
        "        pass\n",
    )

    result = _lower_source(tmp_path, source)

    assert result.status == "lowered"
    assert result.request is not None
    assert "*args" in result.request.replacement_body
    assert "option=option" in result.request.replacement_body
    assert "**kwargs" in result.request.replacement_body


def test_lowering_rejects_method_and_missing_owner_symbols() -> None:
    """This vertical slice remains module-function-only and requires exact source symbols."""
    plan, assessment = _plan_and_assessment()
    method_plan = replace(plan, owner=SymbolId(plan.owner.module, "Runner.run"))
    missing_plan = replace(plan, owner=SymbolId(plan.owner.module, "missing_owner"))

    method = lower_batch_quiescent_plan(FIXTURE_ROOT, method_plan, assessment)
    missing = lower_batch_quiescent_plan(FIXTURE_ROOT, missing_plan, assessment)

    assert any("method lowering is not supported" in reason for reason in method.rejections)
    assert any("resolve to one top-level async function" in reason for reason in missing.rejections)


def test_lowering_rejects_unsafe_and_missing_source_paths(tmp_path: Path) -> None:
    """Plan paths cannot traverse the root or resolve to missing source files."""
    plan, assessment = _plan_and_assessment()
    unsafe_source = PurePosixPath("..", "workflow.py")
    unsafe_identity = replace(
        plan.identity,
        source_hashes=((unsafe_source, "missing"),),
    )
    unsafe_plan = replace(
        plan,
        source=unsafe_source,
        identity=unsafe_identity,
    )
    missing_source = PurePosixPath("missing.py")
    missing_identity = replace(
        plan.identity,
        source_hashes=((missing_source, "missing"),),
    )
    missing_plan = replace(
        plan,
        source=missing_source,
        identity=missing_identity,
    )

    unsafe = lower_batch_quiescent_plan(tmp_path, unsafe_plan, assessment)
    missing = lower_batch_quiescent_plan(tmp_path, missing_plan, assessment)

    assert any("unsafe source path" in reason for reason in unsafe.rejections)
    assert any("does not exist" in reason for reason in missing.rejections)


def test_lowering_rejects_symlink_escape(tmp_path: Path) -> None:
    """Resolving a relative source symlink cannot cross the project boundary."""
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("pass\n", encoding="utf-8")
    link = root / "workflow.py"
    link.symlink_to(outside)
    plan, assessment = _plan_and_assessment()
    source = PurePosixPath("workflow.py")
    identity = replace(plan.identity, source_hashes=((source, "unused"),))
    changed = replace(plan, source=source, identity=identity)

    result = lower_batch_quiescent_plan(root, changed, assessment)

    assert any("escapes project root" in reason for reason in result.rejections)


def test_lowering_rejects_missing_plan_source_hash() -> None:
    """A request is not emitted when its static identity omits the owner source hash."""
    plan, assessment = _plan_and_assessment()
    changed = replace(plan, identity=replace(plan.identity, source_hashes=()))

    result = lower_batch_quiescent_plan(FIXTURE_ROOT, changed, assessment)

    assert any("does not contain one hash" in reason for reason in result.rejections)


def _plan_and_assessment() -> tuple[SourceOptimizationPlan, SourceOptimizationAssessment]:
    source = (FIXTURE_ROOT / SOURCE_PATH).read_text(encoding="utf-8")
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    identity = SourceOptimizationIdentity(
        execution_plan_id="exec-plan-source-fixture",
        source_hashes=((SOURCE_PATH, source_hash),),
        topology_fingerprint="taskgroup-private-queue-v1",
        dialect="asyncio",
        lowering_version="source-lowering-v1",
        python_abi="cp312",
        transformation_versions=(
            ("private-transport-batch-drain", "batch-drain-v1"),
            ("quiescent-callable-execution", "quiescent-callable-v1"),
        ),
    )
    plan_id = stable_source_optimization_plan_id(identity)
    steps = (
        TransformationStep(
            kind="private-transport-batch-drain",
            version="batch-drain-v1",
            source_symbol=OWNER,
            target_symbol=None,
            access_sites=(),
            semantic_boundary="private FIFO completion ordering",
            description="Drain private records after quiescent publication.",
        ),
        TransformationStep(
            kind="quiescent-callable-execution",
            version="quiescent-callable-v1",
            source_symbol=WORKER,
            target_symbol=None,
            access_sites=(),
            semantic_boundary="copied Context and fallback before entry",
            description="Drive exact non-suspending workers in copied contexts.",
        ),
    )
    plan = SourceOptimizationPlan(
        id=plan_id,
        identity=identity,
        source=SOURCE_PATH,
        owner=OWNER,
        worker=WORKER,
        consumer=OWNER,
        reducer=None,
        transport="queue",
        access_sites=(),
        entrypoint=OWNER,
        steps=steps,
        semantic_boundaries=("FIFO", "copied Context", "fallback before entry"),
    )
    evidence = (
        SourceCallableEvidence(
            symbol=WORKER,
            static_role="worker",
            observed_invocations=20_000,
            completed_calls=20_000,
            immediate_result_ratio=1.0,
            hot_share=0.8,
            context_mutation=("_WORKER_CONTEXT.set",),
        ),
        SourceCallableEvidence(
            symbol=DEPENDENCY,
            static_role="dependency",
            observed_invocations=20_000,
            completed_calls=20_000,
            immediate_result_ratio=1.0,
            hot_share=0.1,
        ),
    )
    assessment = SourceOptimizationAssessment(
        plan_id=plan.id,
        status="trial-ready",
        minimum_speedup=3.0,
        work_items=(WORKER,),
        observed_work_items=20_000,
        immediate_result_ratio=1.0,
        attributed_hot_share=0.9,
        scheduler_overhead_samples=10_000,
        scheduler_overhead_share=0.5,
        scheduler_overhead_evidence=("asyncio task scheduling",),
        callable_evidence=evidence,
    )
    return plan, assessment


def _lower_source(tmp_path: Path, source: str) -> SourceLoweringResult:
    root = tmp_path / "source-project"
    source_path = root / SOURCE_PATH
    source_path.parent.mkdir(parents=True)
    source_path.write_text(source, encoding="utf-8")
    plan, assessment = _plan_and_assessment()
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    identity = replace(
        plan.identity,
        source_hashes=((SOURCE_PATH, source_hash),),
    )
    plan_id = stable_source_optimization_plan_id(identity)
    changed_plan = replace(plan, id=plan_id, identity=identity)
    changed_assessment = replace(assessment, plan_id=plan_id)
    return lower_batch_quiescent_plan(root, changed_plan, changed_assessment)


def _transformed_workflow(tmp_path: Path) -> tuple[ModuleType, tuple[str, ...]]:
    plan, assessment = _plan_and_assessment()
    lowering = lower_batch_quiescent_plan(FIXTURE_ROOT, plan, assessment)
    assert lowering.request is not None
    patch = build_source_transformation_patch(FIXTURE_ROOT, (lowering.request,))
    copied_root = tmp_path / "project"
    shutil.copytree(FIXTURE_ROOT, copied_root)
    materialize_transformed_files(FIXTURE_ROOT, copied_root, patch)
    module = _load_workflow(
        copied_root / SOURCE_PATH,
        f"atoll_source_transformed_{tmp_path.name.replace('-', '_')}",
    )
    return module, lowering.helper_names


def _load_workflow(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load source optimization fixture: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _hot_pipeline(
    module: ModuleType,
) -> Callable[[], Coroutine[object, object, Mapping[str, object]]]:
    return _async_mapping(module, "_run_hot_private_pipeline")


def _async_mapping(
    module: ModuleType,
    name: str,
) -> Callable[[], Coroutine[object, object, Mapping[str, object]]]:
    return cast(
        Callable[[], Coroutine[object, object, Mapping[str, object]]],
        vars(module)[name],
    )
