"""Tests for the task-preserving execution-plan backend."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import importlib.util
import sys
from collections.abc import Callable, Coroutine, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from types import TracebackType
from typing import Protocol, cast

import pytest

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.execution_plans import build_execution_plans, execution_plan_profile_targets
from atoll.execution_plans import (
    ExecutionPlan,
    ExecutionPlanAssessmentContext,
    ExecutionPlanStageContext,
    PlanEdge,
    PlanGuard,
    PlanNode,
    TaskPreservingExecutionPlanBackend,
)
from atoll.models import ModuleId, ModuleScan, SymbolId
from atoll.runtime.profiling import (
    CanonicalCallableCount,
    LifecycleCounts,
    ProfiledMember,
    ProfiledSpawnSite,
    ProfileResult,
)

_EXPECTED_TOTAL = 6
_EXPECTED_TASK_COUNT = 3
_STATIC_REJECTION_COUNT = 5
_CLASS_METHOD_PARTS = 2


@dataclass(frozen=True, slots=True)
class _UnsafeIterableCase:
    """One unsafe fan-out iteration fixture and its expected rejection.

    Attributes:
        module_name: Unique fixture module name.
        module_prefix: Source lines inserted before imports.
        fanout_loop: Fan-out loop header under test.
        error_type: Exception class expected from staging.
        message: Diagnostic substring expected from staging.
    """

    module_name: str
    module_prefix: tuple[str, ...]
    fanout_loop: str
    error_type: type[Exception]
    message: str


def _backend_attr(name: str) -> object:
    return getattr(sys.modules[TaskPreservingExecutionPlanBackend.__module__], name)


_reject_dynamic_loop_shape = cast(
    Callable[[ast.For | ast.AsyncFor], None],
    _backend_attr("_reject_dynamic_loop_shape"),
)
_reject_scheduler_reassignment = cast(
    Callable[[ast.For, str], None],
    _backend_attr("_reject_scheduler_reassignment"),
)
_splice_expressions = cast(
    Callable[[str, tuple[tuple[ast.expr, str], ...]], str],
    _backend_attr("_splice_expressions"),
)
_byte_offset = cast(Callable[[bytes, int, int], int], _backend_attr("_byte_offset"))
_module_path = cast(Callable[[Path, str], Path | None], _backend_attr("_module_path"))
_create_task_scheduler = cast(
    Callable[[ast.Call], str | None],
    _backend_attr("_create_task_scheduler"),
)
_spawn_callee = cast(Callable[[ast.Call], str | None], _backend_attr("_spawn_callee"))
_scope_binds_name = cast(
    Callable[[list[ast.stmt], str], bool],
    _backend_attr("_scope_binds_name"),
)
_validate_source_hash = cast(
    Callable[[str, ast.Module, ExecutionPlan], None],
    _backend_attr("_validate_source_hash"),
)
_validate_callsite_fingerprint = cast(
    Callable[[ast.Module, ExecutionPlan], None],
    _backend_attr("_validate_callsite_fingerprint"),
)
_rewrite_target = cast(
    Callable[[ast.Module, ExecutionPlan], object],
    _backend_attr("_rewrite_target"),
)
_validated_rewrite = cast(
    Callable[[str, ExecutionPlan], object],
    _backend_attr("_validated_rewrite"),
)


class _AsyncioModule(Protocol):
    TaskGroup: object


class _PayloadModule(Protocol):
    asyncio: _AsyncioModule
    run: Callable[[list[int]], Coroutine[object, object, int]]


def test_backend_assesses_and_stages_supported_payload(tmp_path: Path) -> None:
    """Supported plans rewrite only the copied payload source."""
    scan = _write_and_scan(tmp_path / "src" / "app" / "worker.py")
    plan = _plan(scan)
    payload_root = _payload_from_source(tmp_path, scan)
    backend = TaskPreservingExecutionPlanBackend()
    original_payload = (payload_root / "app" / "worker.py").read_text(encoding="utf-8")

    assessment = backend.assess(
        plan,
        ExecutionPlanAssessmentContext(
            project_root=tmp_path,
            source_root=tmp_path / "src",
            profile_status="ok",
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

    staged_text = (payload_root / "app" / "worker.py").read_text(encoding="utf-8")
    source_text = Path(scan.module.path).read_text(encoding="utf-8")
    assert assessment.status == "supported"
    assert assessment.unsupported_nodes == ()
    assert staged.payload_files[0].install_path.as_posix() == "app/worker.py"
    assert staged.payload_files[0].before_hash != staged.payload_files[0].after_hash
    assert "_atoll_create_task_" in staged_text
    assert "getattr(_atoll_create_task_" in staged_text
    assert staged_text.count("\n") == original_payload.count("\n")
    assert source_text == Path(scan.module.path).read_text(encoding="utf-8")


def test_backend_validates_and_stages_method_callsite_identity(tmp_path: Path) -> None:
    """Dotted `self.worker` callsites retain the discovery fingerprint exactly."""
    scan = _scan(
        tmp_path / "src" / "app" / "runner.py",
        [
            "import asyncio",
            "# payload comment must survive staging",
            "class Runner:",
            "    async def _worker(self, q, value):",
            "        await q.put(value)",
            "    async def run(self, values):",
            "        q = asyncio.Queue(maxsize=1)",
            "        total = 0",
            "        async with asyncio.TaskGroup() as tg:",
            "            for value in values:",
            "                tg.create_task(self._worker(q, value))",
            "            for _ in values:",
            "                total += await q.get()",
            "        return total",
        ],
    )
    plan = _plan(scan, owner_qualname="Runner.run", worker_qualname="Runner._worker")
    payload_root = tmp_path / "payload"
    payload_path = payload_root / "app" / "runner.py"
    payload_path.parent.mkdir(parents=True)
    payload_path.write_text(Path(scan.module.path).read_text(encoding="utf-8"), encoding="utf-8")
    backend = TaskPreservingExecutionPlanBackend()

    assessment = backend.assess(
        plan,
        ExecutionPlanAssessmentContext(
            project_root=tmp_path,
            source_root=tmp_path / "src",
            profile_status="profiled",
        ),
    )
    backend.stage(
        plan,
        ExecutionPlanStageContext(
            project_root=tmp_path,
            payload_root=payload_root,
            cache_root=tmp_path / ".cache",
        ),
    )

    assert assessment.status == "supported"
    staged_text = payload_path.read_text(encoding="utf-8")
    assert "self._worker(q, value)" in staged_text
    assert "# payload comment must survive staging" in staged_text


def test_stage_rejects_stale_payload_source(tmp_path: Path) -> None:
    """Payload staging fails when the selected source digest no longer matches."""
    scan = _write_and_scan(tmp_path / "src" / "app" / "worker.py")
    plan = _plan(scan)
    payload_root = _payload_from_source(tmp_path, scan)
    payload_file = payload_root / "app" / "worker.py"
    changed_payload = payload_file.read_text(encoding="utf-8").replace(
        "await q.put(value)",
        "await q.put(value + 1)",
    )
    payload_file.write_text(
        changed_payload,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source hash"):
        TaskPreservingExecutionPlanBackend().stage(
            plan,
            ExecutionPlanStageContext(
                project_root=tmp_path,
                payload_root=payload_root,
                cache_root=tmp_path / ".cache",
            ),
        )


def test_stage_rejects_dynamic_or_unsafe_scheduler_shape(tmp_path: Path) -> None:
    """Assigned task handles are refused instead of weakening ownership semantics."""
    scan = _scan(
        tmp_path / "src" / "app" / "unsafe.py",
        [
            "import asyncio",
            "async def _worker(q, value):",
            "    await q.put(value)",
            "async def run(values):",
            "    q = asyncio.Queue(maxsize=1)",
            "    async with asyncio.TaskGroup() as tg:",
            "        for value in values:",
            "            task = tg.create_task(_worker(q, value))",
            "        for _ in values:",
            "            await q.get()",
        ],
    )
    payload_root = tmp_path / "payload"
    (payload_root / "app").mkdir(parents=True)
    payload_file = payload_root / "app" / "unsafe.py"
    payload_file.write_text(Path(scan.module.path).read_text(encoding="utf-8"), encoding="utf-8")
    plan = _manual_loop_plan(scan)

    with pytest.raises(ValueError, match="assigned or captured"):
        TaskPreservingExecutionPlanBackend().stage(
            plan,
            ExecutionPlanStageContext(
                project_root=tmp_path,
                payload_root=payload_root,
                cache_root=tmp_path / ".cache",
            ),
        )


def test_fingerprint_is_stable_and_payload_sensitive(tmp_path: Path) -> None:
    """Fingerprints are deterministic and invalidate on payload content changes."""
    scan = _write_and_scan(tmp_path / "src" / "app" / "worker.py")
    plan = _plan(scan)
    payload_root = _payload_from_source(tmp_path, scan)
    context = ExecutionPlanStageContext(
        project_root=tmp_path,
        payload_root=payload_root,
        cache_root=tmp_path / ".cache",
    )
    backend = TaskPreservingExecutionPlanBackend()

    first = backend.fingerprint(plan, context)
    second = backend.fingerprint(plan, context)
    payload_file = payload_root / "app" / "worker.py"
    payload_file.write_text(
        f"{payload_file.read_text(encoding='utf-8')}\n# payload-only change\n",
        encoding="utf-8",
    )

    assert first == second
    assert backend.fingerprint(plan, context) != first


def test_backend_reports_static_rejections_and_missing_payloads(tmp_path: Path) -> None:
    """Every static capability gate and missing-module path remains diagnostic."""
    scan = _write_and_scan(tmp_path / "src" / "app" / "worker.py")
    plan = _plan(scan)
    backend = TaskPreservingExecutionPlanBackend()
    missing_root = tmp_path / "missing"
    missing_root.mkdir()
    missing_assessment = backend.assess(
        plan,
        ExecutionPlanAssessmentContext(
            project_root=tmp_path,
            source_root=missing_root,
            profile_status="profiled",
        ),
    )
    unsupported = replace(
        plan,
        dialect="anyio-on-asyncio",
        task_ownership="escaping",
        transport_capacity=None,
        completion_transport=None,
        edges=(),
    )
    unsupported_assessment = backend.assess(
        unsupported,
        ExecutionPlanAssessmentContext(
            project_root=tmp_path,
            source_root=tmp_path / "src",
            profile_status="profiled",
        ),
    )
    missing_context = ExecutionPlanStageContext(
        project_root=tmp_path,
        payload_root=missing_root,
        cache_root=tmp_path / ".cache",
    )

    assert missing_assessment.status == "unsupported"
    assert "source module is not present" in missing_assessment.reasons[0]
    assert unsupported_assessment.status == "unsupported"
    assert len(unsupported_assessment.reasons) == _STATIC_REJECTION_COUNT
    with pytest.raises(ValueError, match="unsupported task-preserving"):
        backend.stage(unsupported, missing_context)
    with pytest.raises(ValueError, match="payload module is not present"):
        backend.stage(plan, missing_context)
    with pytest.raises(ValueError, match="payload module is not present"):
        backend.fingerprint(plan, missing_context)


def test_backend_normalizes_multiline_diagnostics_with_log_path(tmp_path: Path) -> None:
    """Backend errors retain deterministic non-empty detail lines and log identity."""
    diagnostic = TaskPreservingExecutionPlanBackend().normalize_diagnostic(
        RuntimeError(),
        diagnostics=" first line \n\n second line ",
        log_path=tmp_path / "plan.log",
    )

    assert diagnostic.message == "RuntimeError"
    assert diagnostic.details == (
        "first line",
        "second line",
        f"log: {tmp_path / 'plan.log'}",
    )


@pytest.mark.parametrize(
    "case",
    [
        _UnsafeIterableCase(
            module_name="non_name_iterable",
            module_prefix=(),
            fanout_loop="        for value in list(values):",
            error_type=TypeError,
            message="side-effect-free local name",
        ),
        _UnsafeIterableCase(
            module_name="async_iterable",
            module_prefix=(),
            fanout_loop="        async for value in values:",
            error_type=TypeError,
            message="async fan-out iteration",
        ),
        _UnsafeIterableCase(
            module_name="shadowed_builtin",
            module_prefix=("type = str",),
            fanout_loop="        for value in values:",
            error_type=ValueError,
            message="unshadowed builtin type",
        ),
    ],
)
def test_backend_rejects_unsafe_iterable_lowering_shapes(
    tmp_path: Path,
    case: _UnsafeIterableCase,
) -> None:
    """Unsafe iteration or shadowed runtime guards remain interpreted."""
    scan = _scan(
        tmp_path / "src" / "app" / f"{case.module_name}.py",
        [
            *case.module_prefix,
            "import asyncio",
            "async def _worker(q, value):",
            "    await q.put(value)",
            "async def run(values):",
            "    q = asyncio.Queue(maxsize=1)",
            "    total = 0",
            "    async with asyncio.TaskGroup() as tg:",
            case.fanout_loop,
            "            tg.create_task(_worker(q, value))",
            "        for _ in range(3):",
            "            total += await q.get()",
            "    return total",
        ],
    )
    plan = _plan(scan)
    payload_root = tmp_path / "payload" / case.module_name
    payload_path = payload_root / "app" / f"{case.module_name}.py"
    payload_path.parent.mkdir(parents=True)
    payload_path.write_text(Path(scan.module.path).read_text(encoding="utf-8"), encoding="utf-8")
    backend = TaskPreservingExecutionPlanBackend()
    assessment = backend.assess(
        plan,
        ExecutionPlanAssessmentContext(
            project_root=tmp_path,
            source_root=tmp_path / "src",
            profile_status="profiled",
        ),
    )

    assert assessment.status == "unsupported"
    with pytest.raises(case.error_type, match=case.message):
        backend.stage(
            plan,
            ExecutionPlanStageContext(
                project_root=tmp_path,
                payload_root=payload_root,
                cache_root=tmp_path / ".cache",
            ),
        )


@pytest.mark.parametrize(
    ("loop_source", "message"),
    [
        ("for value in values:\n    tg = other", "scheduler binding"),
        ("for value in values:\n    tg: object = other", "scheduler binding"),
        ("for value in values:\n    tg += other", "scheduler binding"),
        ("for value in values:\n    if (tg := other):\n        pass", "scheduler binding"),
        ("for value in values:\n    del tg", "scheduler binding"),
        ("for value in values:\n    tg, other = other, tg", "scheduler binding"),
    ],
)
def test_scheduler_reassignment_forms_are_rejected(loop_source: str, message: str) -> None:
    """All lexical assignment forms keep the cached receiver stable."""
    loop = ast.parse(loop_source).body[0]
    assert isinstance(loop, ast.For)

    with pytest.raises(ValueError, match=message):
        _reject_scheduler_reassignment(loop, "tg")


def test_dynamic_coroutine_expression_is_rejected() -> None:
    """Task-preserving lowering requires a direct coroutine-call argument."""
    loop = ast.parse("for value in values:\n    tg.create_task(coroutine)").body[0]
    assert isinstance(loop, ast.For)

    with pytest.raises(ValueError, match="direct coroutine call"):
        _reject_dynamic_loop_shape(loop)


def test_source_splice_and_coordinate_guards_reject_invalid_ranges() -> None:
    """Malformed compiler coordinates fail before touching staged source."""
    expression = cast(ast.Expr, ast.parse("value").body[0]).value
    expression.end_lineno = 2

    with pytest.raises(ValueError, match="same-line expression"):
        _splice_expressions("value\n", ((expression, "replacement"),))
    with pytest.raises(ValueError, match="outside the payload"):
        _byte_offset(b"value\n", 2, 0)


def test_lexical_helper_guards_cover_packages_dynamic_calls_and_bindings(tmp_path: Path) -> None:
    """AST helper boundaries distinguish package modules and dynamic expressions."""
    package_path = tmp_path / "app" / "worker" / "__init__.py"
    package_path.parent.mkdir(parents=True)
    package_path.write_text("", encoding="utf-8")
    nested_scheduler = cast(ast.Expr, ast.parse("a.b.create_task(worker())").body[0]).value
    dynamic_callee = cast(ast.Expr, ast.parse("tg.create_task(factory)").body[0]).value
    assert isinstance(nested_scheduler, ast.Call)
    assert isinstance(dynamic_callee, ast.Call)

    assert _module_path(tmp_path, "app.worker") == package_path
    assert _create_task_scheduler(nested_scheduler) is None
    assert _spawn_callee(dynamic_callee) is None
    assert _scope_binds_name(ast.parse("import type").body, "type") is True
    assert _scope_binds_name(ast.parse("from x import list").body, "list") is True
    assert _scope_binds_name(ast.parse("def tuple():\n    pass").body, "tuple") is True
    assert _scope_binds_name(ast.parse("async def range():\n    pass").body, "range") is True
    assert _scope_binds_name(ast.parse("class type:\n    pass").body, "type") is True
    assert _scope_binds_name(ast.parse("value = lambda type: type").body, "type") is False


def test_plan_identity_validators_reject_missing_or_changed_targets(tmp_path: Path) -> None:
    """Source symbols, owner identity, callsites, and planned lines fail independently."""
    scan = _write_and_scan(tmp_path / "src" / "app" / "worker.py")
    plan = _plan(scan)
    source_text = Path(scan.module.path).read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    missing = SymbolId(scan.module.name, "missing")
    missing_node = PlanNode(missing.stable_id, missing, "worker", 1)

    with pytest.raises(ValueError, match="planned symbol is missing"):
        _validate_source_hash(
            source_text,
            tree,
            replace(
                plan,
                nodes=(*plan.nodes, missing_node),
                source_members=(*plan.source_members, missing),
            ),
        )
    with pytest.raises(ValueError, match="plan owner is missing"):
        _validate_callsite_fingerprint(tree, replace(plan, owner=missing))
    with pytest.raises(ValueError, match="plan owner is missing"):
        _rewrite_target(tree, replace(plan, owner=missing))
    with pytest.raises(ValueError, match="call-site fingerprint"):
        _validate_callsite_fingerprint(tree, replace(plan, callsite_fingerprint="stale"))
    shifted_edges = tuple(replace(edge, lineno=edge.lineno + 100) for edge in plan.edges)
    with pytest.raises(ValueError, match="expected exactly one loop create_task call"):
        _rewrite_target(tree, replace(plan, edges=shifted_edges))


def test_generated_binding_collision_is_rejected(tmp_path: Path) -> None:
    """Generated local names cannot overwrite a source binding."""
    scan = _write_and_scan(tmp_path / "src" / "app" / "worker.py")
    plan = _plan(scan)
    source_text = Path(scan.module.path).read_text(encoding="utf-8")
    suffix = plan.id.rsplit("-", maxsplit=1)[-1]
    binding_name = f"_atoll_create_task_{suffix}"
    changed_source = source_text.replace("    total = 0", f"    total = 0; {binding_name} = None")
    changed_plan = replace(
        plan,
        source_hash=_source_hash_for_text(
            scan.module.name,
            changed_source,
            tuple(
                node.symbol
                for node in plan.nodes
                if node.symbol is not None and node.symbol.module == scan.module.name
            ),
        ),
        source_hashes=(
            (scan.module.name, hashlib.sha256(changed_source.encode("utf-8")).hexdigest()),
        ),
    )

    with pytest.raises(ValueError, match="binding name already exists"):
        _validated_rewrite(changed_source, changed_plan)


def test_staged_payload_preserves_one_task_per_item_and_coroutine_identity(
    tmp_path: Path,
) -> None:
    """The optimized branch still sends one original worker coroutine per item."""
    scan = _write_and_scan(tmp_path / "src" / "app" / "worker.py")
    plan = _plan(scan)
    payload_root = _payload_from_source(tmp_path, scan)
    TaskPreservingExecutionPlanBackend().stage(
        plan,
        ExecutionPlanStageContext(
            project_root=tmp_path,
            payload_root=payload_root,
            cache_root=tmp_path / ".cache",
        ),
    )
    module = cast(
        _PayloadModule,
        _import_payload_module(payload_root / "app" / "worker.py", "payload_worker"),
    )
    created: list[str] = []
    original_task_group = cast(type[asyncio.TaskGroup], module.asyncio.TaskGroup)

    class RecordingTaskGroup:
        def __init__(self) -> None:
            self._inner = original_task_group()

        async def __aenter__(self) -> RecordingTaskGroup:
            await self._inner.__aenter__()
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            await self._inner.__aexit__(exc_type, exc, traceback)

        def create_task(
            self,
            coroutine: Coroutine[object, object, object],
        ) -> asyncio.Task[object]:
            created.append(getattr(coroutine, "__qualname__", ""))
            return self._inner.create_task(coroutine)

    module.asyncio.TaskGroup = RecordingTaskGroup

    result = asyncio.run(module.run([1, 2, 3]))

    assert result == _EXPECTED_TOTAL
    assert created == ["_worker", "_worker", "_worker"]


def test_staged_payload_falls_back_before_side_effecting_iterable_starts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A custom iterator can replace the descriptor before any task is scheduled."""
    scan = _write_and_scan(tmp_path / "src" / "app" / "worker.py")
    plan = _plan(scan)
    payload_root = _payload_from_source(tmp_path, scan)
    TaskPreservingExecutionPlanBackend().stage(
        plan,
        ExecutionPlanStageContext(
            project_root=tmp_path,
            payload_root=payload_root,
            cache_root=tmp_path / ".cache",
        ),
    )
    module = cast(
        _PayloadModule,
        _import_payload_module(payload_root / "app" / "worker.py", "payload_worker_fallback"),
    )
    original_task_group = cast(type[asyncio.TaskGroup], module.asyncio.TaskGroup)
    patched_calls = 0

    class RecordingTaskGroup:
        def __init__(self) -> None:
            self._inner = original_task_group()

        async def __aenter__(self) -> RecordingTaskGroup:
            await self._inner.__aenter__()
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            await self._inner.__aexit__(exc_type, exc, traceback)

        def create_task(
            self,
            coroutine: Coroutine[object, object, object],
        ) -> asyncio.Task[object]:
            return self._inner.create_task(coroutine)

    original_create_task = RecordingTaskGroup.create_task

    def replacement_create_task(
        task_group: RecordingTaskGroup,
        coroutine: Coroutine[object, object, object],
    ) -> asyncio.Task[object]:
        nonlocal patched_calls
        patched_calls += 1
        return original_create_task(task_group, coroutine)

    class SideEffectingValues(list[int]):
        def __iter__(self) -> Iterator[int]:
            monkeypatch.setattr(RecordingTaskGroup, "create_task", replacement_create_task)
            return iter((1, 2, 3))

    module.asyncio.TaskGroup = RecordingTaskGroup

    result = asyncio.run(module.run(SideEffectingValues()))

    assert result == _EXPECTED_TOTAL
    assert patched_calls == _EXPECTED_TASK_COUNT


def _write_and_scan(path: Path) -> ModuleScan:
    return _scan(
        path,
        [
            "import asyncio",
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


def _scan(path: Path, lines: list[str]) -> ModuleScan:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    module = "app.worker" if path.stem == "worker" else f"app.{path.stem}"
    return scan_module(ModuleId(name=module, path=path))


def _plan(
    scan: ModuleScan,
    *,
    owner_qualname: str = "run",
    worker_qualname: str = "_worker",
) -> ExecutionPlan:
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
        for qualname, count in ((owner_qualname, 2_000), (worker_qualname, 2_000))
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


def _manual_loop_plan(scan: ModuleScan) -> ExecutionPlan:
    owner = SymbolId(module=scan.module.name, qualname="run")
    worker = SymbolId(module=scan.module.name, qualname="_worker")
    source_hash = _source_hash_for(scan, (worker, owner))
    return ExecutionPlan(
        id="exec-plan-unsafe",
        source_module=scan.module.name,
        owner=owner,
        dialect="asyncio",
        lowering_version="asyncio-v1",
        source_hash=source_hash,
        callsite_fingerprint=_digest_parts(("asyncio:8:19:_worker",)),
        topology_fingerprint="topology",
        nodes=(
            PlanNode(id=owner.stable_id, symbol=owner, role="orchestrator", lineno=4),
            PlanNode(id=worker.stable_id, symbol=worker, role="producer", lineno=2),
            PlanNode(
                id=f"{owner.stable_id}::transport::q",
                symbol=None,
                role="transport",
                lineno=8,
            ),
            PlanNode(id=f"{owner.stable_id}::reducer", symbol=owner, role="reducer", lineno=4),
        ),
        edges=(
            PlanEdge(
                src=owner.stable_id,
                dst=worker.stable_id,
                kind="spawns",
                transport="q",
                lineno=8,
            ),
        ),
        guards=(PlanGuard(kind="scheduler", expression="asyncio", message="asyncio semantics"),),
        completion_transport="q",
        consumer=owner,
        reducer=owner,
        transport_capacity=1,
    )


def _source_hash_for(scan: ModuleScan, symbols: tuple[SymbolId, ...]) -> str:
    text = Path(scan.module.path).read_text(encoding="utf-8")
    return _source_hash_for_text(scan.module.name, text, symbols)


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
        parts = symbol.qualname.split(".")
        owner: list[ast.stmt] = tree.body
        if len(parts) == _CLASS_METHOD_PARTS:
            class_node = next(
                child
                for child in tree.body
                if isinstance(child, ast.ClassDef) and child.name == parts[0]
            )
            owner = class_node.body
        node = next(
            child
            for child in owner
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef) and child.name == parts[-1]
        )
        start = min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])
        digest.update(symbol.stable_id.encode("utf-8"))
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


def _import_payload_module(path: Path, name: str) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
