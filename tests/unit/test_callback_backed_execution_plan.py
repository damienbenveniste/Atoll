"""Tests for the guarded callback-backed execution-plan backend."""

from __future__ import annotations

import ast
import asyncio
import contextvars
import hashlib
import importlib.util
import sys
import traceback
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import pytest

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.execution_plans import build_execution_plans, execution_plan_profile_targets
from atoll.execution_plans import (
    CALLBACK_BACKED_BACKEND,
    CallbackBackedExecutionPlanBackend,
    ExecutionPlan,
    ExecutionPlanAssessmentContext,
    ExecutionPlanStageContext,
    PlanEdge,
    PlanGuard,
    PlanNode,
)
from atoll.models import ModuleId, ModuleScan, SymbolId
from atoll.runtime.profiling import (
    CanonicalCallableCount,
    LifecycleCounts,
    ProfiledMember,
    ProfiledSpawnSite,
    ProfileResult,
)

_EXPECTED_VALUES = [1, 2, 3]
_EXPECTED_TOTAL = 6
_STATIC_REJECTION_COUNT = 6


class _PayloadModule(Protocol):
    EVENTS: list[str]
    run: Callable[[list[int]], Coroutine[object, object, object]]


@dataclass(frozen=True, slots=True)
class _SourceShape:
    decorator: str | None = None
    producer_prefix: tuple[str, ...] = ()
    producer_line: str = "    q.put_nowait(value)"
    queue_line: str = "    q = asyncio.Queue(maxsize=3)"
    spawn_line: str = "            tg.create_task(_worker(q, value))"
    receive_line: str = "            total += await q.get()"


_REJECTION_SHAPES = {
    "await-put": _SourceShape(producer_line="    await q.put(value)"),
    "await-put-nowait": _SourceShape(producer_line="    await q.put_nowait(value)"),
    "queue-put": _SourceShape(producer_line="    q.put(value)"),
    "nested": _SourceShape(producer_prefix=("    async def nested():", "        return None")),
    "global": _SourceShape(producer_prefix=("    global SEEN",)),
    "local-mutation": _SourceShape(producer_prefix=("    local = value",)),
    "empty-statement": _SourceShape(producer_prefix=("    pass",)),
    "double-publish": _SourceShape(producer_prefix=("    q.put_nowait(value)",)),
    "container-publish": _SourceShape(producer_line="    q.put_nowait([value])"),
    "dynamic-call": _SourceShape(producer_line="    getattr(q, 'put_nowait')(value)"),
    "binop": _SourceShape(producer_line="    q.put_nowait(value + 1)"),
    "compare": _SourceShape(producer_line="    q.put_nowait(value == 1)"),
    "attribute-store": _SourceShape(
        producer_prefix=("    value.result = 1",),
    ),
    "subscript-store": _SourceShape(
        producer_prefix=("    value[0] = 1",),
    ),
    "decorator": _SourceShape(decorator="@decorator"),
    "method-spawn": _SourceShape(spawn_line="            tg.create_task(self._worker(q, value))"),
    "unknown-capacity": _SourceShape(queue_line="    q = asyncio.Queue()"),
    "unsupported-receive": _SourceShape(receive_line="            await other.get()"),
}


def test_backend_assesses_and_stages_supported_payload(tmp_path: Path) -> None:
    """Supported callback-backed plans rewrite only the copied payload source."""
    scan = _write_and_scan(tmp_path / "src" / "app" / "worker.py")
    plan = _plan(scan)
    payload_root = _payload_from_source(tmp_path, scan)
    payload_file = payload_root / "app" / "worker.py"
    original_payload = payload_file.read_text(encoding="utf-8")
    backend = CallbackBackedExecutionPlanBackend()

    assessment = backend.assess(
        plan,
        ExecutionPlanAssessmentContext(
            project_root=tmp_path,
            source_root=tmp_path / "src",
            profile_status="profiled",
        ),
    )
    staged = backend.stage(
        plan,
        ExecutionPlanStageContext(
            project_root=tmp_path,
            payload_root=payload_root,
            cache_root=tmp_path / ".cache",
        ),
    )

    staged_text = payload_file.read_text(encoding="utf-8")
    assert assessment.status == "supported"
    assert staged.backend == CALLBACK_BACKED_BACKEND.name
    assert staged.payload_files[0].install_path.as_posix() == "app/worker.py"
    assert staged.payload_files[0].before_hash == _sha256(original_payload)
    assert staged.payload_files[0].after_hash == _sha256(staged_text)
    assert staged.payload_files[0].before_hash != staged.payload_files[0].after_hash
    assert "loop.call_soon" in staged_text
    assert "_atoll_callback_" in staged_text
    assert "await _atoll_callback_" in staged_text
    assert Path(scan.module.path).read_text(encoding="utf-8") == _source_text()


def test_producer_docstring_is_metadata_not_executable_work(tmp_path: Path) -> None:
    """A leading producer docstring does not invalidate a strict publish body."""
    shape = _SourceShape(producer_prefix=('    """Publish one item."""',))
    source = _source_text(shape)
    scan = _scan(tmp_path / "src" / "app" / "worker.py", source.splitlines())
    plan = _manual_plan(scan, source)

    assessment = CallbackBackedExecutionPlanBackend().assess(
        plan,
        ExecutionPlanAssessmentContext(tmp_path, tmp_path / "src", "profiled"),
    )

    assert assessment.status == "supported"


def test_optimized_path_calls_soon_once_per_item_and_preserves_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The callback path schedules one callback per item and returns queue order."""
    module = _stage_and_import(tmp_path, _source_text(), "payload_callback_order")
    _permit_optimized_runtime(module, monkeypatch)
    call_soon_count = 0

    async def run_with_counter() -> object:
        nonlocal call_soon_count
        loop = asyncio.get_running_loop()
        original = loop.call_soon

        def counted_call_soon(
            callback: Callable[..., object],
            *args: object,
            context: contextvars.Context | None = None,
        ) -> asyncio.Handle:
            nonlocal call_soon_count
            if getattr(callback, "__name__", "") == "_atoll_callback_drive_once":
                call_soon_count += 1
            return original(callback, *args, context=context)

        monkeypatch.setattr(loop, "call_soon", counted_call_soon)
        return await module.run(_EXPECTED_VALUES)

    result = asyncio.run(run_with_counter())

    assert result == _EXPECTED_TOTAL
    assert call_soon_count == len(_EXPECTED_VALUES)
    assert _last_mode(module) == "optimized"


def test_optimized_path_uses_separate_copied_context_per_item(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each scheduled logical item receives its own copied context object."""
    module = _stage_and_import(tmp_path, _source_text(), "payload_callback_context")
    _permit_optimized_runtime(module, monkeypatch)
    contexts: list[contextvars.Context] = []
    used_contexts: list[contextvars.Context] = []
    original_copy_context = contextvars.copy_context

    def fake_copy_context() -> contextvars.Context:
        context = original_copy_context()
        contexts.append(context)
        return context

    async def run_with_context_capture() -> object:
        loop = asyncio.get_running_loop()
        original_call_soon = loop.call_soon

        def capture_call_soon(
            callback: Callable[..., object],
            *args: object,
            context: contextvars.Context | None = None,
        ) -> asyncio.Handle:
            if getattr(callback, "__name__", "") == "_atoll_callback_drive_once":
                assert context is not None
                used_contexts.append(context)
            return original_call_soon(callback, *args, context=context)

        monkeypatch.setattr(contextvars, "copy_context", fake_copy_context)
        monkeypatch.setattr(loop, "call_soon", capture_call_soon)
        return await module.run(_EXPECTED_VALUES)

    result = asyncio.run(run_with_context_capture())

    assert result == _EXPECTED_TOTAL
    assert len(used_contexts) == len(_EXPECTED_VALUES)
    assert len({id(context) for context in used_contexts}) == len(_EXPECTED_VALUES)
    assert all(context in contexts for context in used_contexts)


def test_immediate_exception_uses_original_taskgroup_path_with_producer_traceback(
    tmp_path: Path,
) -> None:
    """Producer exceptions are rejected so the original TaskGroup owns aggregation."""
    source = "\n".join(
        [
            "import asyncio",
            "async def _worker(q, value):",
            "    raise RuntimeError",
            "    q.put_nowait(value)",
            "async def run(values):",
            "    q = asyncio.Queue(maxsize=3)",
            "    total = 0",
            "    async with asyncio.TaskGroup() as tg:",
            "        for value in values:",
            "            tg.create_task(_worker(q, value))",
            "        for _ in values:",
            "            total += await q.get()",
            "    return total",
        ],
    )
    source_path = tmp_path / "payload_callback_exception" / "app" / "worker.py"
    scan = _scan(source_path, source.splitlines())
    plan = _manual_plan(scan, source)
    assessment = CallbackBackedExecutionPlanBackend().assess(
        plan,
        ExecutionPlanAssessmentContext(
            project_root=tmp_path,
            source_root=source_path.parents[1],
            profile_status="profiled",
        ),
    )
    module = cast(
        _PayloadModule,
        _import_payload_module(source_path, "payload_callback_exception"),
    )

    with pytest.raises(ExceptionGroup) as raised:
        asyncio.run(module.run(_EXPECTED_VALUES))

    assert assessment.status == "unsupported"
    assert any("task-preserving" in reason for reason in assessment.reasons)
    exception = _leaf_exception(raised.value)
    rendered = "".join(traceback.format_exception(exception))
    assert isinstance(exception, RuntimeError)
    assert "in _worker" in rendered


def test_repeated_optimized_invocations_release_per_queue_state(tmp_path: Path) -> None:
    """Successful delivery removes retained queue and TaskGroup references."""
    module = _stage_and_import(tmp_path, _source_text(), "payload_callback_cleanup")

    assert asyncio.run(module.run(_EXPECTED_VALUES)) == _EXPECTED_TOTAL
    assert asyncio.run(module.run(_EXPECTED_VALUES)) == _EXPECTED_TOTAL

    states = [
        value
        for name, value in vars(cast(ModuleType, module)).items()
        if name.startswith("_atoll_callback_") and name.endswith("_state")
    ]
    assert states == [{}]


@pytest.mark.parametrize(
    "configure",
    [
        "task-factory",
        "debug",
        "extra-task",
        "scheduled-work",
    ],
)
def test_runtime_guard_failures_use_original_real_task_fallback(
    tmp_path: Path,
    configure: str,
) -> None:
    """Runtime scheduler guard failures execute the original task path."""
    module = _stage_and_import(tmp_path, _source_text(), f"payload_callback_{configure}")

    async def run_guard_case() -> object:
        loop = asyncio.get_running_loop()
        cleanup: list[Callable[[], None]] = []
        if configure == "task-factory":
            loop.set_task_factory(asyncio.eager_task_factory)
            cleanup.append(lambda: loop.set_task_factory(None))
        elif configure == "debug":
            loop.set_debug(True)
            cleanup.append(lambda: loop.set_debug(False))
        elif configure == "extra-task":
            event = asyncio.Event()
            task = asyncio.create_task(event.wait())

            def cancel_task() -> None:
                task.cancel()

            cleanup.append(cancel_task)
            cleanup.append(event.set)
        elif configure == "scheduled-work":
            handle = loop.call_later(60, lambda: None)
            cleanup.append(handle.cancel)
        try:
            return await module.run(_EXPECTED_VALUES)
        finally:
            for callback in reversed(cleanup):
                callback()

    result = asyncio.run(run_guard_case())

    assert result == _EXPECTED_TOTAL
    assert _last_mode(module) == "fallback"


def test_unexpected_producer_suspension_is_hard_execution_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A producer that suspends after staging fails instead of retrying."""
    module = _stage_and_import(tmp_path, _source_text(), "payload_callback_suspend")
    _permit_optimized_runtime(module, monkeypatch)

    async def suspending_worker(q: asyncio.Queue[int], value: int) -> None:
        await asyncio.sleep(0)
        q.put_nowait(value)

    original_name = _original_name(module)
    worker_name = "_worker"
    setattr(module, worker_name, suspending_worker)
    setattr(module, original_name, suspending_worker)

    with pytest.raises(ExceptionGroup) as raised:
        asyncio.run(module.run(_EXPECTED_VALUES))

    exception = _leaf_exception(raised.value)
    assert isinstance(exception, RuntimeError)
    assert "unexpectedly suspended" in str(exception)
    assert _last_mode(module) == "optimized"


def test_fingerprint_is_stable_source_sensitive_and_support_sensitive(tmp_path: Path) -> None:
    """Fingerprints include plan identity, payload source, Python ABI, and support."""
    scan = _write_and_scan(tmp_path / "src" / "app" / "worker.py")
    plan = _plan(scan)
    payload_root = _payload_from_source(tmp_path, scan)
    context = ExecutionPlanStageContext(
        project_root=tmp_path,
        payload_root=payload_root,
        cache_root=tmp_path / ".cache",
    )
    backend = CallbackBackedExecutionPlanBackend()

    first = backend.fingerprint(plan, context)
    second = backend.fingerprint(plan, context)
    source_sensitive = backend.fingerprint(replace(plan, topology_fingerprint="changed"), context)
    payload_file = payload_root / "app" / "worker.py"
    payload_file.write_text(
        f"{payload_file.read_text(encoding='utf-8')}\n# payload-only change\n",
        encoding="utf-8",
    )

    assert first == second
    assert source_sensitive != first
    assert backend.fingerprint(plan, context) != first


def test_backend_reports_static_and_missing_payload_failures(tmp_path: Path) -> None:
    """Capability and filesystem failures stay explicit before staging."""
    scan = _write_and_scan(tmp_path / "src" / "app" / "worker.py")
    plan = _plan(scan)
    backend = CallbackBackedExecutionPlanBackend()
    missing_root = tmp_path / "missing"

    missing_assessment = backend.assess(
        plan,
        ExecutionPlanAssessmentContext(tmp_path, missing_root, "profiled"),
    )
    broken = replace(
        plan,
        dialect="unsupported",
        task_ownership="escaping",
        transport_capacity=0,
        completion_transport=None,
        edges=(),
        nodes=(),
    )
    broken_assessment = backend.assess(
        broken,
        ExecutionPlanAssessmentContext(tmp_path, tmp_path / "src", "profiled"),
    )

    assert "not present" in missing_assessment.reasons[0]
    assert broken_assessment.status == "unsupported"
    assert len(broken_assessment.reasons) == _STATIC_REJECTION_COUNT
    missing_context = ExecutionPlanStageContext(tmp_path, missing_root, tmp_path / ".cache")
    with pytest.raises(ValueError, match="unsupported callback-backed"):
        backend.stage(broken, missing_context)
    with pytest.raises(ValueError, match="payload module is not present"):
        backend.stage(plan, missing_context)
    with pytest.raises(ValueError, match="payload module is not present"):
        backend.fingerprint(plan, missing_context)

    diagnostic = backend.normalize_diagnostic(
        RuntimeError("stage failed"),
        diagnostics="first line\nsecond line",
        log_path=tmp_path / "callback.log",
    )
    assert diagnostic.details == (
        "first line",
        "second line",
        f"log: {tmp_path / 'callback.log'}",
    )


@pytest.mark.parametrize(
    ("prefix", "message"),
    [
        ("_AtollCallbackFailure = object()", "generated name already exists"),
        ("len = object()", "requires unshadowed builtin len"),
    ],
)
def test_generated_support_rejects_payload_name_collisions(
    tmp_path: Path,
    prefix: str,
    message: str,
) -> None:
    """Generated helpers never overwrite source-owned names or builtins."""
    source = f"{prefix}\n{_source_text()}"
    scan = _scan(tmp_path / "src" / "app" / "worker.py", source.splitlines())
    plan = _manual_plan(scan, source)

    assessment = CallbackBackedExecutionPlanBackend().assess(
        plan,
        ExecutionPlanAssessmentContext(tmp_path, tmp_path / "src", "profiled"),
    )

    assert assessment.status == "unsupported"
    assert any(message in reason for reason in assessment.reasons)


@pytest.mark.parametrize(
    ("loop_source", "message"),
    [
        ("async for value in values:", "async"),
        ("for value, other in values:", "single local name"),
        ("for value in list(values):", "side-effect-free local name"),
    ],
)
def test_fanout_loop_shape_rejections_are_explicit(
    tmp_path: Path,
    loop_source: str,
    message: str,
) -> None:
    """Async, destructuring, and expression iterables remain interpreted."""
    source = _source_text().replace("for value in values:", loop_source)
    scan = _scan(tmp_path / "src" / "app" / "worker.py", source.splitlines())
    plan = _manual_plan(scan, source)

    assessment = CallbackBackedExecutionPlanBackend().assess(
        plan,
        ExecutionPlanAssessmentContext(tmp_path, tmp_path / "src", "profiled"),
    )

    assert assessment.status == "unsupported"
    assert any(message in reason for reason in assessment.reasons)


@pytest.mark.parametrize("constant_declaration", ["CAPACITY = 3", "CAPACITY: int = 3"])
def test_callback_backend_resolves_module_constant_queue_capacity(
    tmp_path: Path,
    constant_declaration: str,
) -> None:
    """Literal module constants can prove an annotated queue's capacity."""
    shape = _SourceShape(
        queue_line="    q: asyncio.Queue[int] = asyncio.Queue(maxsize=CAPACITY)",
        spawn_line=("            pass\n            tg.create_task(_worker(q, value))"),
    )
    source = f"{constant_declaration}\n{_source_text(shape)}"
    scan = _scan(tmp_path / "src" / "app" / "worker.py", source.splitlines())
    plan = _manual_plan(scan, source)

    assessment = CallbackBackedExecutionPlanBackend().assess(
        plan,
        ExecutionPlanAssessmentContext(tmp_path, tmp_path / "src", "profiled"),
    )

    assert assessment.status == "supported"


@pytest.mark.parametrize(
    ("edge_kind", "message"),
    [("produces", "producer transport edge"), ("delivers", "delivery edge")],
)
def test_callback_backend_rejects_transport_edge_drift(
    tmp_path: Path,
    edge_kind: str,
    message: str,
) -> None:
    """A staged source cannot be paired with stale producer or delivery edges."""
    scan = _write_and_scan(tmp_path / "src" / "app" / "worker.py")
    plan = _plan(scan)
    edges = tuple(
        replace(edge, transport="changed") if edge.kind == edge_kind else edge
        for edge in plan.edges
    )

    assessment = CallbackBackedExecutionPlanBackend().assess(
        replace(plan, edges=edges),
        ExecutionPlanAssessmentContext(tmp_path, tmp_path / "src", "profiled"),
    )

    assert assessment.status == "unsupported"
    assert any(message in reason for reason in assessment.reasons)


def test_staging_rejects_stale_source_and_callsite_identity(tmp_path: Path) -> None:
    """Source and call-site drift fail before a payload rewrite is applied."""
    scan = _write_and_scan(tmp_path / "src" / "app" / "worker.py")
    plan = _plan(scan)
    payload_root = _payload_from_source(tmp_path, scan)
    payload_file = payload_root / "app" / "worker.py"
    context = ExecutionPlanStageContext(tmp_path, payload_root, tmp_path / ".cache")
    payload_file.write_text(
        payload_file.read_text(encoding="utf-8").replace("return total", "return total + 0"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source hash"):
        CallbackBackedExecutionPlanBackend().stage(plan, context)

    payload_file.write_text(Path(scan.module.path).read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="call-site fingerprint"):
        CallbackBackedExecutionPlanBackend().stage(
            replace(plan, callsite_fingerprint="stale"),
            context,
        )


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            "\n".join(
                [
                    "import asyncio",
                    "async def _worker(q, value):",
                    "    q.put_nowait(value)",
                    "async def run(values):",
                    "    q = asyncio.Queue(maxsize=3)",
                    "    async with asyncio.TaskGroup() as tg:",
                    "        for value in values:",
                    "            tg.create_task(_worker(q, value))",
                    "    for _ in values:",
                    "        await q.get()",
                ]
            ),
            "inside the spawning TaskGroup",
        ),
        (
            "\n".join(
                [
                    "import asyncio",
                    "async def _worker(q, value):",
                    "    q.put_nowait(value)",
                    "async def _other():",
                    "    return None",
                    "async def run(values):",
                    "    q = asyncio.Queue(maxsize=3)",
                    "    async with asyncio.TaskGroup() as tg:",
                    "        for value in values:",
                    "            tg.create_task(_worker(q, value))",
                    "        tg.create_task(_other())",
                    "        for _ in values:",
                    "            await q.get()",
                ]
            ),
            "exactly one planned spawn site",
        ),
    ],
)
def test_owner_topology_rejects_shifted_receive_or_additional_spawn(
    tmp_path: Path,
    source: str,
    message: str,
) -> None:
    """Callback work cannot escape its original structured scheduler boundary."""
    scan = _scan(tmp_path / "src" / "app" / "worker.py", source.splitlines())
    plan = _manual_plan(scan, source)

    assessment = CallbackBackedExecutionPlanBackend().assess(
        plan,
        ExecutionPlanAssessmentContext(tmp_path, tmp_path / "src", "profiled"),
    )

    assert assessment.status == "unsupported"
    assert any(message in reason for reason in assessment.reasons)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("await-put", "must not suspend"),
        ("await-put-nowait", "must not suspend"),
        ("queue-put", "opaque calls"),
        ("nested", "nested"),
        ("global", "global or nonlocal"),
        ("local-mutation", "local mutation"),
        ("empty-statement", "exact queue.put_nowait"),
        ("double-publish", "publish exactly once"),
        ("container-publish", "parameter or constant"),
        ("dynamic-call", "opaque calls"),
        ("binop", "dynamic operations"),
        ("compare", "dynamic operations"),
        ("attribute-store", "attribute or subscript mutation"),
        ("subscript-store", "attribute or subscript mutation"),
        ("decorator", "decorators"),
        ("method-spawn", "module-level"),
        ("unknown-capacity", "known positive capacity"),
        ("unsupported-receive", "await q.get"),
    ],
)
def test_conservative_rejection_matrix(tmp_path: Path, case: str, message: str) -> None:
    """Unsupported AnyIO, dynamic, suspending, and consumer shapes are explicit."""
    source = _rejection_source(case)
    scan = _scan(tmp_path / "src" / "app" / "worker.py", source.splitlines())
    plan = _manual_plan(scan, source)
    backend = CallbackBackedExecutionPlanBackend()

    assessment = backend.assess(
        plan,
        ExecutionPlanAssessmentContext(
            project_root=tmp_path,
            source_root=tmp_path / "src",
            profile_status="profiled",
        ),
    )

    assert assessment.status == "unsupported"
    assert any(message in reason for reason in assessment.reasons)


def test_anyio_and_method_plans_are_rejected_before_source_rewrite(tmp_path: Path) -> None:
    """Non-asyncio dialects and method producers remain outside this backend."""
    scan = _write_and_scan(tmp_path / "src" / "app" / "worker.py")
    plan = _plan(scan)
    method_symbol = SymbolId(scan.module.name, "Runner._worker")
    method_plan = replace(
        plan,
        nodes=(
            PlanNode(plan.owner.stable_id, plan.owner, "orchestrator", 4),
            PlanNode(method_symbol.stable_id, method_symbol, "producer", 2),
        ),
    )

    anyio = CallbackBackedExecutionPlanBackend().assess(
        replace(plan, dialect="anyio-on-asyncio"),
        ExecutionPlanAssessmentContext(tmp_path, tmp_path / "src", "profiled"),
    )
    method = CallbackBackedExecutionPlanBackend().assess(
        method_plan,
        ExecutionPlanAssessmentContext(tmp_path, tmp_path / "src", "profiled"),
    )

    assert anyio.status == "unsupported"
    assert any("unsupported scheduler dialect" in reason for reason in anyio.reasons)
    assert method.status == "unsupported"
    assert any("module-level" in reason for reason in method.reasons)


def _stage_and_import(tmp_path: Path, source: str, module_name: str) -> _PayloadModule:
    scan = _scan(tmp_path / module_name / "src" / "app" / "worker.py", source.splitlines())
    plan = _manual_plan(scan, source)
    payload_root = tmp_path / module_name / "payload"
    payload_path = payload_root / "app" / "worker.py"
    payload_path.parent.mkdir(parents=True)
    payload_path.write_text(source, encoding="utf-8")
    CallbackBackedExecutionPlanBackend().stage(
        plan,
        ExecutionPlanStageContext(
            project_root=tmp_path,
            payload_root=payload_root,
            cache_root=tmp_path / ".cache",
        ),
    )
    return cast(_PayloadModule, _import_payload_module(payload_path, module_name))


def _write_and_scan(path: Path) -> ModuleScan:
    return _scan(path, _source_text().splitlines())


def _scan(path: Path, lines: Sequence[str]) -> ModuleScan:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return scan_module(ModuleId(name="app.worker", path=path))


def _source_text(shape: _SourceShape | None = None) -> str:
    source_shape = shape or _SourceShape()
    lines = [
        "import asyncio",
        "EVENTS = []",
        *(
            line
            for line in ((source_shape.decorator,) if source_shape.decorator is not None else ())
            if line
        ),
        "async def _worker(q, value):",
        *source_shape.producer_prefix,
        source_shape.producer_line,
        "async def run(values):",
        source_shape.queue_line,
        "    total = 0",
        "    async with asyncio.TaskGroup() as tg:",
        "        for value in values:",
        source_shape.spawn_line,
        "        for _ in values:",
        source_shape.receive_line,
        "    return total",
    ]
    return "\n".join(lines)


def _rejection_source(case: str) -> str:
    try:
        shape = _REJECTION_SHAPES[case]
    except KeyError as error:
        raise AssertionError(f"unknown rejection source case: {case}") from error
    return _source_text(shape)


def _plan(scan: ModuleScan) -> ExecutionPlan:
    profiled = tuple(
        ProfiledMember(
            module=scan.module.name,
            qualname=qualname,
            samples=0,
            coverage=0.0,
            call_count=count,
            invocation_count=count,
            lifecycle=LifecycleCounts(
                start=count,
                return_=count,
                yield_=0,
                resume=0,
                unwind=0,
                throw=0,
            ),
            signatures=(),
            polymorphic_overflow=False,
        )
        for qualname, count in (("run", 2_000), ("_worker", 2_000))
    )
    invocations_by_owner = {member.symbol: member.invocation_count for member in profiled}
    profile = ProfileResult(
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
            start=sum(member.lifecycle.start for member in profiled),
            return_=sum(member.lifecycle.return_ for member in profiled),
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
                        identity=f"asyncio.taskgroups.TaskGroup.{target.scheduler_method}",
                        count=invocations_by_owner.get(target.owner, 0),
                    ),
                ),
            )
            for target in execution_plan_profile_targets((scan,))
        ),
    )
    return next(
        result
        for result in build_execution_plans((scan,), profile)
        if isinstance(result, ExecutionPlan)
    )


def _payload_from_source(tmp_path: Path, scan: ModuleScan) -> Path:
    payload_root = tmp_path / "payload"
    payload_path = payload_root / "app" / "worker.py"
    payload_path.parent.mkdir(parents=True)
    payload_path.write_text(Path(scan.module.path).read_text(encoding="utf-8"), encoding="utf-8")
    return payload_root


def _manual_plan(scan: ModuleScan, source: str) -> ExecutionPlan:
    owner = SymbolId(module=scan.module.name, qualname="run")
    worker = SymbolId(module=scan.module.name, qualname="_worker")
    callsite = _callsite(source)
    callsite_lineno = int(callsite.split(":")[1])
    transport_node = f"{owner.stable_id}::transport::q"
    reducer_node = f"{owner.stable_id}::reducer"
    return ExecutionPlan(
        id="exec-plan-callback",
        source_module=scan.module.name,
        owner=owner,
        dialect="asyncio",
        lowering_version="asyncio-v1",
        source_hash=_source_hash_for_text(scan.module.name, source, (worker, owner)),
        callsite_fingerprint=_callsite_fingerprint(source),
        topology_fingerprint="topology",
        nodes=(
            PlanNode(id=owner.stable_id, symbol=owner, role="orchestrator", lineno=6),
            PlanNode(id=worker.stable_id, symbol=worker, role="producer", lineno=3),
            PlanNode(id=transport_node, symbol=None, role="transport", lineno=7),
            PlanNode(id=reducer_node, symbol=owner, role="reducer", lineno=6),
        ),
        edges=(
            PlanEdge(
                src=owner.stable_id,
                dst=worker.stable_id,
                kind="spawns",
                transport="q",
                lineno=callsite_lineno,
            ),
            PlanEdge(
                src=worker.stable_id,
                dst=transport_node,
                kind="produces",
                transport="q",
                lineno=callsite_lineno,
            ),
            PlanEdge(
                src=transport_node,
                dst=reducer_node,
                kind="delivers",
                transport="q",
                lineno=callsite_lineno,
            ),
        ),
        guards=(PlanGuard(kind="scheduler", expression="asyncio", message="asyncio semantics"),),
        completion_transport="q",
        consumer=owner,
        reducer=owner,
        transport_capacity=3,
    )


def _callsite(source: str) -> str:
    tree = ast.parse(source)
    owner = next(
        node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "run"
    )
    call = next(
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "create_task"
    )
    assert isinstance(call.args[0], ast.Call)
    callee = _attribute_path(call.args[0].func)
    return f"asyncio:{call.lineno}:{call.col_offset}:{callee}"


def _callsite_fingerprint(source: str) -> str:
    tree = ast.parse(source)
    owner = next(
        node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "run"
    )
    parts: list[str] = []
    for call in (
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "create_task"
    ):
        assert isinstance(call.args[0], ast.Call)
        parts.append(
            f"asyncio:{call.lineno}:{call.col_offset}:{_attribute_path(call.args[0].func)}"
        )
    return _digest_parts(tuple(parts))


def _attribute_path(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_attribute_path(node.value)}.{node.attr}"
    raise AssertionError("unsupported callsite attribute path")


def _leaf_exception(group: ExceptionGroup[Exception]) -> Exception:
    exception = group.exceptions[0]
    while isinstance(exception, ExceptionGroup):
        exception = cast(Exception | ExceptionGroup[Exception], exception.exceptions[0])
    return exception


def _permit_optimized_runtime(
    module: _PayloadModule,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Make optimized-path tests independent from the test runner's own monitoring."""

    def monitoring_disabled(_sys: object) -> bool:
        return True

    monkeypatch.setattr(sys, "gettrace", lambda: None)
    monkeypatch.setattr(sys, "getprofile", lambda: None)
    monkeypatch.setattr(
        cast(ModuleType, module),
        "_atoll_callback_no_monitoring_hooks",
        monitoring_disabled,
    )


def _source_hash_for_text(
    module_name: str,
    text: str,
    symbols: tuple[SymbolId, ...],
) -> str:
    tree = compile(text, module_name, "exec", flags=ast.PyCF_ONLY_AST)
    assert isinstance(tree, ast.Module)
    lines = text.splitlines()
    digest = hashlib.sha256()
    for symbol in sorted(set(symbols), key=lambda item: item.qualname):
        node = next(
            child
            for child in tree.body
            if isinstance(child, ast.AsyncFunctionDef) and child.name == symbol.qualname
        )
        start = min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])
        digest.update(f"{module_name}::{symbol.qualname}".encode())
        digest.update(b"\0")
        digest.update("\n".join(lines[start - 1 : node.end_lineno]).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _digest_parts(parts: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _import_payload_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _last_mode(module: object) -> str | None:
    names = [name for name in dir(module) if name.endswith("_last_mode")]
    assert len(names) == 1
    value = getattr(module, names[0])
    assert value is None or isinstance(value, str)
    return value


def _original_name(module: object) -> str:
    names = [name for name in dir(module) if name.endswith("_original")]
    assert len(names) == 1
    return names[0]
