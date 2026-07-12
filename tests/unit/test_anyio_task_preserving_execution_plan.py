"""Tests for AnyIO task-preserving execution-plan staging."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import importlib.util
import inspect
import sys
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from types import FrameType, ModuleType
from typing import Protocol, cast

import pytest

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.execution_plans import (
    build_execution_plans,
    execution_plan_observation_targets,
    execution_plan_profile_targets,
)
from atoll.execution_plans import (
    AnyioTaskPreservingExecutionPlanBackend,
    ExecutionPlan,
    ExecutionPlanAssessmentContext,
    ExecutionPlanStageContext,
    PlanRejection,
)
from atoll.execution_plans import anyio_task_preserving as anyio_lowering
from atoll.models import DependencyEdge, ModuleId, ModuleScan, SymbolId
from atoll.runtime.profiling import (
    CanonicalCallableCount,
    LifecycleCounts,
    ProfiledMember,
    ProfiledSpawnSite,
    ProfileResult,
)

_HOT_COUNT = 2_000
_STAGED_FILE_COUNT = 2
_TERMINAL_SEND_COUNT = 2
_VARIED_ARITY = 5
type TraceFunction = Callable[[FrameType, str, object], TraceFunction | None]


def _anyio_attr(name: str) -> object:
    return vars(anyio_lowering)[name]


def _anyio_callable(name: str) -> Callable[..., object]:
    return cast(Callable[..., object], _anyio_attr(name))


_direct_reducer = cast(Callable[[str, SymbolId], object | None], _anyio_attr("_direct_reducer"))
_is_pure_context_class = cast(Callable[[ast.ClassDef], bool], _anyio_attr("_is_pure_context_class"))
_statement_blocks = cast(
    Callable[[ast.stmt], tuple[list[ast.stmt], ...]], _anyio_attr("_statement_blocks")
)
_static_rejection_reasons = cast(
    Callable[[ExecutionPlan], tuple[str, ...]], _anyio_attr("_static_rejection_reasons")
)
_receiver_name = cast(Callable[[ast.FunctionDef], str], _anyio_attr("_receiver_name"))
_matching_loop_names = cast(
    Callable[[ast.For, ast.For], tuple[str, str]], _anyio_attr("_matching_loop_names")
)
_spawn_fields = cast(Callable[[ast.Call, str, str], tuple[str, str]], _anyio_attr("_spawn_fields"))
_registration_fields = cast(
    Callable[[ast.For, str, str], tuple[str, str]], _anyio_attr("_registration_fields")
)
_module_expression = cast(Callable[[str, str], str], _anyio_attr("_module_expression"))
_apply_line_edits = cast(
    Callable[[str, tuple[tuple[int, int, str], ...]], str],
    _anyio_attr("_apply_line_edits"),
)
_source_lines = cast(Callable[[str, int, int], str], _anyio_attr("_source_lines"))
_newline = cast(Callable[[str], str], _anyio_attr("_newline"))


class _PayloadModule(Protocol):
    """Runtime surface exposed by the generated runner fixture."""

    async def execute(
        self,
        values: list[int],
        *,
        fail: int | None = None,
        use_task_factory: bool = False,
    ) -> dict[str, object]: ...


class _ReflectionNamesView(Protocol):
    """Generated reflection binding needed by collision tests."""

    helper: str


class _SupportNamesView(Protocol):
    """Generated dispatch binding needed by collision tests."""

    owner_class: str


def test_discovery_links_anyio_plan_to_imported_reducer(tmp_path: Path) -> None:
    """Cross-module discovery observes and selects the narrowed reducer call."""
    scans = _fixture_scans(tmp_path)

    targets = execution_plan_observation_targets(scans)
    first = _selected_plan(scans, hot_count=_HOT_COUNT)
    second = _selected_plan(scans, hot_count=_HOT_COUNT * 2)

    assert "workflow.reducer::Reducer.reduce" in targets
    assert first.id == second.id
    assert first.source_hash == second.source_hash
    assert first.source_hashes == second.source_hashes
    assert all(module_hash for _, module_hash in first.source_hashes)
    assert first.reducer == SymbolId("workflow.reducer", "Reducer.reduce")
    assert any(node.role == "reducer" and node.symbol == first.reducer for node in first.nodes)
    assert first.completion_transport == "self.send_stream|self.receive_stream"
    assert first.transport_capacity == 0


def test_backend_stages_only_payload_overlays_and_preserves_source(
    tmp_path: Path,
) -> None:
    """Assessment and staging rewrite copied payload files without changing checkout source."""
    scans = _fixture_scans(tmp_path)
    plan = _selected_plan(scans)
    payload_root = _payload_from_scans(tmp_path, scans)
    source_hashes = _file_hashes(tuple(Path(scan.module.path) for scan in scans))
    payload_hashes = _file_hashes(tuple(payload_root / _module_file(scan) for scan in scans))
    backend = AnyioTaskPreservingExecutionPlanBackend()

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

    changed_paths = {file.install_path.as_posix() for file in staged.payload_files}
    assert assessment.status == "supported"
    assert changed_paths == {"workflow/runner.py", "workflow/reducer.py"}
    assert _file_hashes(tuple(Path(scan.module.path) for scan in scans)) == source_hashes
    assert (
        _file_hashes(tuple(payload_root / _module_file(scan) for scan in scans)) != payload_hashes
    )
    runner_source = (payload_root / "workflow" / "runner.py").read_text(encoding="utf-8")
    reducer_source = (payload_root / "workflow" / "reducer.py").read_text(encoding="utf-8")
    assert "self.task_group.start_soon(self._worker, item)" in runner_source
    assert runner_source.count("_atoll_anyio_dispatch_") > 0
    assert "if _atoll_anyio_dispatch_" in runner_source
    assert "_dispatch = _atoll_anyio_dispatch_" in runner_source
    assert "_scheduler_call(" in runner_source
    assert ".create_task" in runner_source
    assert "._spawn" not in runner_source
    assert "else:\n            for item in request:" in runner_source
    assert "if scope is not None:\n            scope.cancel()" in runner_source
    assert "_all_events" in runner_source
    assert "_reducer_callable =" not in runner_source
    assert "callable_value.__code__" in reducer_source
    assert "return len(signature(callable_value).parameters)" in reducer_source


def test_backend_composes_with_managed_native_region_overlay(tmp_path: Path) -> None:
    """A verified Atoll native shim does not look like checkout source drift."""
    scans = _fixture_scans(tmp_path)
    plan = _selected_plan(scans)
    payload_root = _payload_from_scans(tmp_path, scans)
    runner = payload_root / "workflow" / "runner.py"
    source = runner.read_text(encoding="utf-8")
    runner.write_text(
        source.rstrip()
        + "\n\n"
        + "# BEGIN ATOLL TYPED REGIONS: workflow.runner\n"
        + "_atoll_native_binding = True\n"
        + "# END ATOLL TYPED REGIONS: workflow.runner\n",
        encoding="utf-8",
    )

    staged = AnyioTaskPreservingExecutionPlanBackend().stage(
        plan,
        ExecutionPlanStageContext(tmp_path, payload_root, tmp_path / ".cache"),
    )

    staged_source = runner.read_text(encoding="utf-8")
    assert len(staged.payload_files) == _STAGED_FILE_COUNT
    assert "# BEGIN ATOLL TYPED REGIONS: workflow.runner" in staged_source
    assert "_atoll_native_binding = True" in staged_source
    assert "# AnyIO terminal-handoff support appended by Atoll." in staged_source


def test_stage_marks_only_terminal_sends_and_rejects_unsafe_shapes(tmp_path: Path) -> None:
    """Only tail sends get scope sentinels, while stale or changed shapes fail."""
    scans = _fixture_scans(tmp_path)
    plan = _selected_plan(scans)
    payload_root = _payload_from_scans(tmp_path, scans)
    backend = AnyioTaskPreservingExecutionPlanBackend()

    backend.stage(
        plan,
        ExecutionPlanStageContext(tmp_path, payload_root, tmp_path / ".cache"),
    )

    runner_source = (payload_root / "workflow" / "runner.py").read_text(encoding="utf-8")
    assert runner_source.count("self.scopes[item.key] = None") == _TERMINAL_SEND_COUNT
    assert (
        "self.scopes[item.key] = None\n                await self.send_stream.send" in runner_source
    )
    assert "if item.value < 0:\n                await self.send_stream.send" in runner_source

    stale_payload = _payload_from_scans(tmp_path, scans, name="stale")
    stale_runner = stale_payload / "workflow" / "runner.py"
    stale_runner.write_text(
        stale_runner.read_text(encoding="utf-8").replace(
            "self.registry[item.key] = item",
            "self.registry[item.key] = WorkItem(item.key, item.value)",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source hash"):
        backend.stage(plan, ExecutionPlanStageContext(tmp_path, stale_payload, tmp_path / ".cache"))

    unsafe_payload = _payload_from_scans(tmp_path, scans, name="unsafe")
    unsafe_runner = unsafe_payload / "workflow" / "runner.py"
    unsafe_runner.write_text(
        unsafe_runner.read_text(encoding="utf-8").replace(
            (
                "        for item in request:\n"
                "            self.task_group.start_soon(self._worker, item)"
            ),
            "        if request:\n            self.task_group.start_soon(self._worker, request[0])",
        ),
        encoding="utf-8",
    )
    unsafe_scans = _scans_from_payload(unsafe_payload)
    unsafe_hashes = _source_hashes_for(unsafe_scans, plan.source_members)
    unsafe_plan = replace(
        plan,
        source_hashes=unsafe_hashes,
        source_hash=_digest_parts(f"{module}:{digest}" for module, digest in unsafe_hashes),
    )
    with pytest.raises(ValueError, match=r"call-site fingerprint|start_soon loop"):
        backend.stage(
            unsafe_plan,
            ExecutionPlanStageContext(tmp_path, unsafe_payload, tmp_path / ".cache"),
        )


def test_dispatch_only_reducer_is_inlined_with_guarded_method_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pure arity dispatcher skips its context and observes later method replacement."""
    pytest.importorskip("anyio")
    scans = _fixture_scans(
        tmp_path,
        runner_source=_direct_runner_source(),
        reducer_source=_direct_reducer_source(),
    )
    plan = _selected_plan(scans)
    payload_root = _payload_from_scans(tmp_path, scans)
    AnyioTaskPreservingExecutionPlanBackend().stage(
        plan,
        ExecutionPlanStageContext(tmp_path, payload_root, tmp_path / ".cache"),
    )
    runner_source = (payload_root / "workflow" / "runner.py").read_text(encoding="utf-8")
    module = cast(_PayloadModule, _import_module(payload_root / "workflow" / "runner.py"))

    optimized = asyncio.run(module.execute([1, 2, 3]))

    assert optimized["values"] == [("outer", 2), ("outer", 4), ("outer", 6)]
    assert "_reducer_callable = candidate.reducer" in runner_source
    assert "_reducer_arity == 2" in runner_source
    assert runner_source.index("_reducer_arity == 2") < runner_source.index(
        "context = ReducerContext"
    )

    reducer_module = sys.modules["workflow.reducer"]
    reducer_class = cast(type[object], reducer_module.Reducer)

    def replacement(
        self: object,
        context: object,
        current: int,
        item: object,
    ) -> int:
        del self, context, current, item
        return 99

    monkeypatch.setattr(reducer_class, "reduce", replacement)
    fallback = asyncio.run(module.execute([4]))
    assert fallback["values"] == [("outer", 99)]


@pytest.mark.parametrize(
    "shape",
    [
        "wrong-qualname",
        "async-method",
        "varargs",
        "keyword-only",
        "extra-statement",
        "bad-reflection",
        "bad-branch",
        "missing-else",
        "different-field",
        "different-cast",
        "wrong-short-count",
        "literal-argument",
    ],
)
def test_direct_reducer_analysis_rejects_lossy_dispatch_shapes(shape: str) -> None:
    """Only a complete two-arm arity dispatcher is eligible for inlining."""
    source, qualname = _lossy_direct_reducer(shape)

    assert _direct_reducer(source, SymbolId("workflow.reducer", qualname)) is None


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "@dataclass(init=False)\n"
            "class Context(Generic[T]):\n"
            "    def __init__(self, *, state, deps):\n"
            "        self.state = state\n"
            "        self.deps = deps\n",
            True,
        ),
        (
            "@dataclass(init=False)\n"
            "class Context(Generic[T], metaclass=Meta):\n"
            "    def __init__(self, state):\n"
            "        self.state = state\n",
            False,
        ),
        (
            "@other\nclass Context:\n    def __init__(self, state):\n        self.state = state\n",
            False,
        ),
        (
            "@dataclass\n"
            "class Context(Base):\n"
            "    def __init__(self, state):\n"
            "        self.state = state\n",
            False,
        ),
        (
            "@dataclass\n"
            "class Context:\n"
            "    def __init__(self, state):\n"
            "        self.state = state\n"
            "    def __setattr__(self, name, value):\n"
            "        object.__setattr__(self, name, value)\n",
            False,
        ),
        (
            "@dataclass\n"
            "class Context:\n"
            "    async def __init__(self, state):\n"
            "        self.state = state\n",
            False,
        ),
        (
            "@dataclass\n"
            "class Context:\n"
            "    @decorator\n"
            "    def __init__(self, state):\n"
            "        self.state = state\n",
            False,
        ),
        (
            "@dataclass\n"
            "class Context:\n"
            "    def __init__(self, state):\n"
            "        self.state = transform(state)\n",
            False,
        ),
    ],
)
def test_pure_context_analysis_requires_assignment_only_construction(
    source: str,
    expected: bool,
) -> None:
    """Context allocation is lazy only when construction has no hidden behavior."""
    tree = ast.parse(source)
    class_node = tree.body[0]
    assert isinstance(class_node, ast.ClassDef)

    assert _is_pure_context_class(class_node) is expected


def test_statement_block_walker_covers_nested_control_flow() -> None:
    """Reducer-call lookup descends through every supported statement block."""
    tree = ast.parse(
        """
async def owner(items):
    if items:
        pass
    for item in items:
        pass
    async for item in items:
        pass
    while items:
        break
    try:
        pass
    except ValueError:
        pass
    else:
        pass
    finally:
        pass
    with manager():
        pass
    async with manager():
        pass
    match items:
        case []:
            pass
    return None
"""
    )
    function = tree.body[0]
    assert isinstance(function, ast.AsyncFunctionDef)

    block_counts = [len(_statement_blocks(statement)) for statement in function.body]

    assert block_counts == [2, 2, 2, 2, 4, 1, 1, 1, 0]


def test_fingerprint_tracks_payload_source_not_profile_counts(tmp_path: Path) -> None:
    """Backend fingerprints are stable for the same payload and change with source."""
    scans = _fixture_scans(tmp_path)
    first = _selected_plan(scans, hot_count=_HOT_COUNT)
    second = _selected_plan(scans, hot_count=_HOT_COUNT * 3)
    payload_root = _payload_from_scans(tmp_path, scans)
    context = ExecutionPlanStageContext(tmp_path, payload_root, tmp_path / ".cache")
    backend = AnyioTaskPreservingExecutionPlanBackend()

    first_fingerprint = backend.fingerprint(first, context)

    assert backend.fingerprint(second, context) == first_fingerprint
    runner = payload_root / "workflow" / "runner.py"
    runner.write_text(
        runner.read_text(encoding="utf-8").replace("self.values.append", "self.values.append"),
        encoding="utf-8",
    )
    assert backend.fingerprint(first, context) == first_fingerprint
    runner.write_text(
        runner.read_text(encoding="utf-8").replace(
            "self.registry[item.key] = item",
            "self.registry[item.key] = WorkItem(item.key, item.value)",
        ),
        encoding="utf-8",
    )
    assert backend.fingerprint(first, context) != first_fingerprint


def test_static_rejections_cover_stale_source_and_unsupported_plan(tmp_path: Path) -> None:
    """Assessment rejects stale source and static plan fields before staging."""
    scans = _fixture_scans(tmp_path)
    plan = _selected_plan(scans)
    runner_path = tmp_path / "src" / "workflow" / "runner.py"
    runner_path.write_text(
        runner_path.read_text(encoding="utf-8").replace(
            "self.registry[item.key] = item",
            "self.registry[item.key] = WorkItem(item.key, item.value)",
        ),
        encoding="utf-8",
    )
    backend = AnyioTaskPreservingExecutionPlanBackend()

    stale = backend.assess(
        plan,
        ExecutionPlanAssessmentContext(tmp_path, tmp_path / "src", "profiled"),
    )
    unsupported = backend.assess(
        replace(plan, dialect="asyncio"),
        ExecutionPlanAssessmentContext(tmp_path, tmp_path / "src", "profiled"),
    )

    assert stale.status == "unsupported"
    assert any("source hash" in reason for reason in stale.reasons)
    assert unsupported.status == "unsupported"
    assert unsupported.reasons == ("unsupported scheduler dialect: asyncio",)


def test_backend_reports_missing_payloads_and_normalizes_diagnostics(tmp_path: Path) -> None:
    """Backend setup failures retain explicit reasons and stable report diagnostics."""
    scans = _fixture_scans(tmp_path)
    plan = _selected_plan(scans)
    backend = AnyioTaskPreservingExecutionPlanBackend()
    missing_root = tmp_path / "missing"

    assessment = backend.assess(
        plan,
        ExecutionPlanAssessmentContext(tmp_path, missing_root, "profiled"),
    )
    with pytest.raises(ValueError, match="unsupported AnyIO"):
        backend.stage(
            replace(plan, dialect="asyncio"),
            ExecutionPlanStageContext(tmp_path, missing_root, tmp_path / ".cache"),
        )
    with pytest.raises(ValueError, match="payload module is not present"):
        backend.stage(
            plan,
            ExecutionPlanStageContext(tmp_path, missing_root, tmp_path / ".cache"),
        )
    payload_root = tmp_path / "partial"
    runner = payload_root / "workflow" / "runner.py"
    runner.parent.mkdir(parents=True)
    runner.write_text(_runner_source(), encoding="utf-8")
    with pytest.raises(ValueError, match="linked reducer payload module"):
        backend.stage(
            plan,
            ExecutionPlanStageContext(tmp_path, payload_root, tmp_path / ".cache"),
        )
    with pytest.raises(ValueError, match="payload module is not present"):
        backend.fingerprint(
            plan,
            ExecutionPlanStageContext(tmp_path, missing_root, tmp_path / ".cache"),
        )
    diagnostic = backend.normalize_diagnostic(
        ValueError(),
        diagnostics=" first \n\n second ",
        log_path=tmp_path / "backend.log",
    )

    assert assessment.status == "unsupported"
    assert assessment.reasons == ("source module is not present below the assessment source root",)
    assert diagnostic.message == "ValueError"
    assert diagnostic.details == ("first", "second", f"log: {tmp_path / 'backend.log'}")


def test_static_rejection_reasons_accumulate_independent_plan_failures(tmp_path: Path) -> None:
    """All backend-independent plan contract violations are reported together."""
    plan = _selected_plan(_fixture_scans(tmp_path))
    invalid = replace(
        plan,
        dialect="asyncio",
        task_ownership="escaping",
        transport_capacity=None,
        completion_transport=None,
        edges=(),
        owner=SymbolId(plan.owner.module, "schedule"),
    )

    assert _static_rejection_reasons(invalid) == (
        "unsupported scheduler dialect: asyncio",
        "unsupported task ownership: escaping",
        "transport capacity must be statically known",
        "completion transport must be statically known",
        "plan must contain exactly one spawn edge",
        "plan owner must be a direct class method",
    )


def test_anyio_lowering_utility_guards_reject_ambiguous_syntax() -> None:
    """Low-level source utilities reject coordinates and dynamic call shapes."""
    no_receiver = ast.parse("def owner():\n    pass\n").body[0]
    assert isinstance(no_receiver, ast.FunctionDef)
    with pytest.raises(ValueError, match="no receiver"):
        _receiver_name(no_receiver)

    loops = ast.parse(
        "def owner(first, second):\n"
        "    for item in first:\n"
        "        pass\n"
        "    for other in second:\n"
        "        pass\n"
    ).body[0]
    assert isinstance(loops, ast.FunctionDef)
    registration, spawn = loops.body
    assert isinstance(registration, ast.For)
    assert isinstance(spawn, ast.For)
    with pytest.raises(ValueError, match="share local request"):
        _matching_loop_names(registration, spawn)

    invalid_spawn = ast.parse("self.group.start_soon(worker, item, extra)", mode="eval").body
    assert isinstance(invalid_spawn, ast.Call)
    with pytest.raises(ValueError, match="one receiver method"):
        _spawn_fields(invalid_spawn, "self", "item")
    with pytest.raises(ValueError, match="one assignment"):
        _registration_fields(registration, "self", "item")
    with pytest.raises(ValueError, match="stable attribute path"):
        _module_expression("module", "lambda: None")
    with pytest.raises(ValueError, match="edit coordinates"):
        _apply_line_edits("line\n", ((3, 3, "replacement\n"),))
    with pytest.raises(ValueError, match="source coordinates"):
        _source_lines("line\n", 2, 1)
    assert _newline("first\r\nsecond\r\n") == "\r\n"


def test_dispatch_and_terminal_handoff_guards_reject_changed_shapes(tmp_path: Path) -> None:
    """Every guarded orchestration coordinate fails before a staged edit can run."""
    scans = _fixture_scans(tmp_path)
    plan = _selected_plan(scans)
    source = _runner_source()
    tree = ast.parse(source)
    dispatch_target = _anyio_callable("_dispatch_target")
    terminal_target = _anyio_callable("_terminal_handoff_target")
    validate_fingerprint = _anyio_callable("_validate_callsite_fingerprint")
    transport_fields = _anyio_callable("_transport_fields")
    cleanup_target = _anyio_callable("_cleanup_target")
    valid_dispatch = dispatch_target(tree, plan)

    with pytest.raises(ValueError, match="plan owner is missing"):
        validate_fingerprint(ast.parse("class Runner:\n    pass\n"), plan)
    for changed_call in (
        "self.task_group.start_soon()",
        "self.task_group.start_soon(lambda: None, item)",
    ):
        changed = source.replace(
            "self.task_group.start_soon(self._worker, item)",
            changed_call,
        )
        with pytest.raises(ValueError, match="fingerprint"):
            validate_fingerprint(ast.parse(changed), plan)

    async_owner = source.replace(
        "    def schedule(self, request):", "    async def schedule(self, request):"
    )
    with pytest.raises(TypeError, match="owner must be synchronous"):
        dispatch_target(ast.parse(async_owner), plan)
    trailing_statement = source.replace(
        "            self.task_group.start_soon(self._worker, item)\n\n    async def _worker",
        "            self.task_group.start_soon(self._worker, item)\n        marker = 1\n\n"
        "    async def _worker",
    )
    with pytest.raises(ValueError, match="end the owner"):
        dispatch_target(ast.parse(trailing_statement), plan)
    missing_registration = source.replace(
        "        for item in request:\n            self.registry[item.key] = item",
        "        marker = len(request)\n        pass",
    )
    with pytest.raises(TypeError, match="registration loop"):
        dispatch_target(ast.parse(missing_registration), plan)
    changed_worker = source.replace(
        "start_soon(self._worker, item)", "start_soon(self.other, item)"
    )
    with pytest.raises(ValueError, match="planned producer"):
        dispatch_target(ast.parse(changed_worker), plan)
    generator_worker = source.replace(
        "    async def _worker(self, item):\n",
        "    async def _worker(self, item):\n        if False:\n            yield item\n",
    )
    with pytest.raises(ValueError, match="ordinary coroutine"):
        dispatch_target(ast.parse(generator_worker), plan)

    sync_worker = source.replace(
        "    async def _worker(self, item):", "    def _worker(self, item):"
    )
    with pytest.raises(TypeError, match="not an ordinary coroutine"):
        terminal_target(ast.parse(sync_worker), plan, valid_dispatch)
    missing_item = source.replace("_worker(self, item)", "_worker(self)")
    with pytest.raises(ValueError, match="work-item parameter"):
        terminal_target(ast.parse(missing_item), plan, valid_dispatch)
    missing_send = source.replace("await self.send_stream.send", "await self.other.send")
    with pytest.raises(ValueError, match="no statically terminal"):
        terminal_target(ast.parse(missing_send), plan, valid_dispatch)

    with pytest.raises(ValueError, match="no private completion"):
        transport_fields(replace(plan, completion_transport=None), "self")
    with pytest.raises(ValueError, match="two direct instance"):
        transport_fields(replace(plan, completion_transport="send|receive"), "self")
    with pytest.raises(ValueError, match="stable instance field"):
        transport_fields(
            replace(plan, completion_transport="self.stream.send|self.receive"),
            "self",
        )

    with pytest.raises(ValueError, match="owner class is missing"):
        cleanup_target(ast.parse("class Other:\n    pass\n"), "Runner", "scopes", plan)
    cleanup_without_key = source.replace("cleanup(self, key)", "cleanup(self)")
    with pytest.raises(ValueError, match="private scope cleanup"):
        cleanup_target(ast.parse(cleanup_without_key), "Runner", "scopes", plan)
    cleanup_without_cancel = source.replace("            scope.cancel()", "            pass")
    with pytest.raises(ValueError, match="private scope cleanup"):
        cleanup_target(ast.parse(cleanup_without_cancel), "Runner", "scopes", plan)

    valid_handoff = terminal_target(tree, plan, valid_dispatch)
    names = _anyio_callable("_support_names")(plan.id)
    with pytest.raises(ValueError, match="known stream capacity"):
        _anyio_callable("_support_source")(
            names,
            valid_dispatch,
            valid_handoff,
            replace(plan, transport_capacity=None),
            None,
        )


def test_terminal_and_reducer_ast_guards_cover_ambiguous_forms(tmp_path: Path) -> None:
    """Nested control flow and reducer shortcuts remain conservative."""
    scope_registration = _anyio_callable("_scope_registration")
    terminal_child_blocks = _anyio_callable("_terminal_child_blocks")
    assignment_only_initializer = _anyio_callable("_is_assignment_only_initializer")
    dispatch_return = _anyio_callable("_dispatch_return")
    asserted_class_expression = _anyio_callable("_asserted_class_expression")
    previous_statement = _anyio_callable("_previous_statement")
    reflection_target = _anyio_callable("_reflection_target")
    module_binds_name = _anyio_callable("_module_binds_name")

    producer = ast.parse(
        "async def produce(self, item):\n    scope = object()\n    self.scopes[item.key] = scope\n"
    ).body[0]
    assert isinstance(producer, ast.AsyncFunctionDef)
    with pytest.raises(ValueError, match="cancellation-scope registration"):
        scope_registration(producer, "self", "item")

    try_statement = ast.parse("try:\n    pass\nfinally:\n    pass\n").body[0]
    match_statement = ast.parse("match value:\n    case _:\n        pass\n").body[0]
    assert terminal_child_blocks(try_statement) == ()
    assert len(cast(tuple[object, ...], terminal_child_blocks(match_statement))) == 1

    no_state = ast.parse("class Context:\n    def __init__(self):\n        pass\n").body[0]
    documented = ast.parse(
        "class Context:\n"
        "    def __init__(self, state):\n"
        "        'initialize'\n"
        "        self.state = state\n"
    ).body[0]
    assert isinstance(no_state, ast.ClassDef)
    assert isinstance(documented, ast.ClassDef)
    no_state_init = no_state.body[0]
    documented_init = documented.body[0]
    assert isinstance(no_state_init, ast.FunctionDef)
    assert isinstance(documented_init, ast.FunctionDef)
    assert assignment_only_initializer(no_state_init) is False
    assert assignment_only_initializer(documented_init) is True

    reducer_function = ast.parse(
        "def reduce(self, value):\n    return self.callable(value)\n"
    ).body[0]
    assert isinstance(reducer_function, ast.FunctionDef)
    malformed_returns = (
        ast.Pass(),
        ast.parse("return self.callable(value=1)").body[0],
        ast.parse("return wrong(self.callable)(value)").body[0],
        ast.parse("return callable(value)").body[0],
    )
    for statement in malformed_returns:
        assert dispatch_return(statement, reducer_function) is None

    asserted = ast.parse(
        "async def consume(candidate, value):\n"
        "    assert candidate\n"
        "    assert isinstance(candidate, Other)\n"
        "    result = candidate.reduce(value)\n"
    ).body[0]
    assert isinstance(asserted, ast.AsyncFunctionDef)
    assert asserted_class_expression(asserted, "candidate", 4, "Reducer") is None
    absent = ast.Pass()
    assert previous_statement(asserted.body, absent) is None

    reflection_function = ast.parse("def reduce(self):\n    pass\n").body[0]
    assert isinstance(reflection_function, ast.FunctionDef)
    bad_signature_calls = (
        "arity = len(inspect.signature().parameters)",
        "arity = len(inspect.other(self.reducer).parameters)",
    )
    for expression in bad_signature_calls:
        assignment = ast.parse(expression).body[0]
        assert reflection_target("workflow.reducer", reflection_function, assignment) is None

    for source in (
        "def len():\n    pass\n",
        "from helpers import size as len\n",
        "len = custom\n",
    ):
        assert module_binds_name(ast.parse(source), "len") is True

    one_parameter_source = (
        "import inspect\n"
        "class Reducer:\n"
        "    def reduce(self):\n"
        "        arity = len(inspect.signature(self.reducer).parameters)\n"
        "        if arity == 0:\n"
        "            return self.reducer()\n"
        "        else:\n"
        "            return self.reducer()\n"
    )
    assert (
        _direct_reducer(
            one_parameter_source,
            SymbolId("workflow.reducer", "Reducer.reduce"),
        )
        is None
    )
    documented_reducer = _direct_reducer_source().replace(
        "    def reduce(self, context, current, item):\n",
        "    def reduce(self, context, current, item):\n        'dispatch by arity'\n",
    )
    assert (
        _direct_reducer(
            documented_reducer,
            SymbolId("workflow.reducer", "Reducer.reduce"),
        )
        is not None
    )

    scans = _fixture_scans(
        tmp_path,
        runner_source=_direct_runner_source(),
        reducer_source=_direct_reducer_source(),
    )
    plan = _selected_plan(scans)
    reducer = _direct_reducer(
        _direct_reducer_source(),
        SymbolId("workflow.reducer", "Reducer.reduce"),
    )
    assert reducer is not None
    consumer_reducer_call = _anyio_callable("_consumer_reducer_call")
    runner_tree = ast.parse(_direct_runner_source())
    assert consumer_reducer_call(runner_tree, replace(plan, consumer=None), (reducer,)) is None
    assert (
        consumer_reducer_call(
            runner_tree,
            replace(plan, consumer=SymbolId("workflow.runner", "Runner.missing")),
            (reducer,),
        )
        is None
    )
    no_assert = _direct_runner_source().replace(
        "            assert isinstance(candidate, Reducer)",
        "            pass",
    )
    assert consumer_reducer_call(ast.parse(no_assert), plan, (reducer,)) is None


def test_reflection_rewrite_rejects_shadowing_missing_targets_and_collisions(
    tmp_path: Path,
) -> None:
    """Reflection specialization requires one unshadowed and collision-free lookup."""
    scans = _fixture_scans(tmp_path)
    plan = _selected_plan(scans)
    module_name = "workflow.reducer"
    qualname = "Reducer.reduce"
    rewrite = _anyio_callable("_validated_reflection_rewrite")

    shadowed = "len = lambda value: 0\n" + _reducer_source()
    with pytest.raises(ValueError, match="shadows the required len"):
        rewrite(
            shadowed, _plan_with_module_source(plan, module_name, shadowed), module_name, qualname
        )

    missing = "class Reducer:\n    pass\n"
    missing_plan = _plan_with_module_source(plan, module_name, missing)
    missing_plan = replace(
        missing_plan,
        source_members=tuple(
            member for member in missing_plan.source_members if member.module != module_name
        ),
    )
    with pytest.raises(ValueError, match="linked reducer is missing"):
        rewrite(missing, missing_plan, module_name, qualname)

    no_lookup = _reducer_source().replace(
        "        arity = len(inspect.signature(self.reducer).parameters)",
        "        arity = 1",
    )
    with pytest.raises(ValueError, match="expected one guarded signature lookup"):
        rewrite(
            no_lookup, _plan_with_module_source(plan, module_name, no_lookup), module_name, qualname
        )

    names = _anyio_callable("_reflection_names")(plan.id, module_name, qualname)
    helper_name = cast(_ReflectionNamesView, names).helper
    collision = _reducer_source() + f"\n{helper_name} = None\n"
    with pytest.raises(ValueError, match="support name already exists"):
        rewrite(
            collision,
            _plan_with_module_source(plan, module_name, collision),
            module_name,
            qualname,
        )

    source_path = Path(scans[0].module.path)
    with pytest.raises(ValueError, match="linked reducer module is not present"):
        _anyio_callable("_validate_assessment_sources")(
            source_path,
            tmp_path / "missing-source-root",
            plan,
        )

    direct_root = tmp_path / "direct"
    direct_scans = _fixture_scans(
        direct_root,
        runner_source=_direct_runner_source(),
        reducer_source=_direct_reducer_source(),
    )
    direct_plan = _selected_plan(direct_scans)
    _anyio_callable("_validate_assessment_sources")(
        Path(direct_scans[0].module.path),
        direct_root / "src",
        direct_plan,
    )

    support_names = _anyio_callable("_support_names")(plan.id)
    collision_name = cast(_SupportNamesView, support_names).owner_class
    runner_collision = _runner_source() + f"\n{collision_name} = None\n"
    collision_plan = _plan_with_module_source(plan, plan.source_module, runner_collision)
    with pytest.raises(ValueError, match="support name already exists"):
        _anyio_callable("_validated_rewrite")(runner_collision, collision_plan)


def test_staged_runtime_preserves_anyio_task_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The staged source keeps real AnyIO tasks, context isolation, and fallback behavior."""
    pytest.importorskip("anyio")
    scans = _fixture_scans(tmp_path)
    plan = _selected_plan(scans)
    payload_root = _payload_from_scans(tmp_path, scans)
    AnyioTaskPreservingExecutionPlanBackend().stage(
        plan,
        ExecutionPlanStageContext(tmp_path, payload_root, tmp_path / ".cache"),
    )
    module = cast(_PayloadModule, _import_module(payload_root / "workflow" / "runner.py"))
    expected_mode = "fallback" if _runtime_instrumentation_active() else "optimized"

    normal = asyncio.run(module.execute([1, 2, 3]))
    factory_run = asyncio.run(module.execute([4, 5], use_task_factory=True))
    debug_fallback = asyncio.run(module.execute([6]), debug=True)

    assert normal["values"] == [("outer", 2), ("outer", 4), ("outer", 6)]
    assert normal["inner_context"] == ["worker-1", "worker-2", "worker-3"]
    assert normal["outer_context"] == "outer"
    assert normal["cancelled"] == []
    assert normal["scopes_empty"] is True
    assert normal["task_names"] == ["workflow.runner.Runner._worker"] * 3
    assert normal["mode"] == expected_mode
    assert factory_run["values"] == [("outer", 8), ("outer", 10)]
    assert factory_run["mode"] == "fallback"
    assert cast(list[object], factory_run["factory_task_types"])
    assert factory_run["factory_registry_sizes"] == [2, 2]
    assert debug_fallback["mode"] == "fallback"

    def trace(frame: FrameType, event: str, argument: object) -> TraceFunction:
        del frame, event, argument
        return trace

    previous_trace = sys.gettrace()
    sys.settrace(trace)
    try:
        traced_fallback = asyncio.run(module.execute([9]))
    finally:
        sys.settrace(previous_trace)
    assert traced_fallback["mode"] == "fallback"

    reducer_module = sys.modules["workflow.reducer"]
    arity = cast(
        Callable[[object], int],
        next(
            value
            for name, value in vars(reducer_module).items()
            if name.startswith("_atoll_signature_arity_")
            and callable(value)
            and not name.endswith("_no_monitoring")
            and getattr(value, "__module__", None) == "workflow.reducer"
            and getattr(value, "__name__", None) == name
        ),
    )

    def varied(first: int, /, second: int = 0, *items: int, flag: bool, **options: int) -> int:
        del items, flag, options
        return first + second

    assert arity(varied) == len(inspect.signature(varied).parameters) == _VARIED_ARITY
    original_code = varied.__code__

    def unary(first: int) -> int:
        return first

    varied.__code__ = unary.__code__
    try:
        assert arity(varied) == 1
    finally:
        varied.__code__ = original_code
    assert arity(varied) == _VARIED_ARITY
    original_signature = inspect.signature
    reflected: list[object] = []

    def observed_signature(value: object) -> inspect.Signature:
        reflected.append(value)
        return original_signature(cast(Callable[..., object], value))

    monkeypatch.setattr(inspect, "signature", observed_signature)
    assert arity(varied) == _VARIED_ARITY
    assert reflected == [varied]

    with pytest.raises(ExceptionGroup) as raised:
        asyncio.run(module.execute([7, 8], fail=16, use_task_factory=True))
    assert any(str(error) == "boom" for error in raised.value.exceptions)


def test_staged_runtime_falls_back_for_replaced_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker descriptor replacement routes the complete request through source."""
    pytest.importorskip("anyio")
    scans = _fixture_scans(tmp_path)
    plan = _selected_plan(scans)
    payload_root = _payload_from_scans(tmp_path, scans)
    AnyioTaskPreservingExecutionPlanBackend().stage(
        plan,
        ExecutionPlanStageContext(tmp_path, payload_root, tmp_path / ".cache"),
    )
    module = cast(_PayloadModule, _import_module(payload_root / "workflow" / "runner.py"))
    runner_class = cast(type[object], vars(module)["Runner"])
    original_worker = cast(
        Callable[[object, object], Awaitable[None]],
        vars(runner_class)["_worker"],
    )

    async def replacement_worker(owner: object, item: object) -> None:
        await original_worker(owner, item)

    monkeypatch.setattr(runner_class, "_worker", replacement_worker)
    replaced_worker = asyncio.run(module.execute([10]))

    assert replaced_worker["values"] == [("outer", 20)]
    assert replaced_worker["mode"] == "fallback"


def _fixture_scans(
    tmp_path: Path,
    *,
    runner_source: str | None = None,
    reducer_source: str | None = None,
) -> tuple[ModuleScan, ModuleScan]:
    source_root = tmp_path / "src"
    package = source_root / "workflow"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    runner = package / "runner.py"
    reducer = package / "reducer.py"
    runner.write_text(runner_source or _runner_source(), encoding="utf-8")
    reducer.write_text(reducer_source or _reducer_source(), encoding="utf-8")
    runner_scan = scan_module(ModuleId("workflow.runner", runner))
    runner_scan = replace(
        runner_scan,
        dependency_edges=(
            *runner_scan.dependency_edges,
            DependencyEdge(
                src=SymbolId("workflow.runner", "Runner.results"),
                dst=SymbolId("workflow.runner", "Runner.cleanup"),
                kind="calls_method",
                confidence="high",
                lineno=64,
                invocation_mode="awaited",
                requires_same_unit=True,
            ),
        ),
    )
    return (
        runner_scan,
        scan_module(ModuleId("workflow.reducer", reducer)),
    )


def _runner_source() -> str:
    return "\n".join(
        [
            "from __future__ import annotations",
            "",
            "from contextvars import ContextVar",
            "from dataclasses import dataclass",
            "",
            "import asyncio",
            "import anyio",
            "from anyio import CancelScope",
            "from anyio.abc import TaskGroup",
            "",
            "from .reducer import Reducer",
            "",
            "TRACE = ContextVar('trace', default='unset')",
            "",
            "@dataclass(frozen=True, slots=True)",
            "class WorkItem:",
            "    key: int",
            "    value: int",
            "",
            "class Runner:",
            "    def __init__(self, task_group: TaskGroup, reducer: object):",
            "        self.task_group = task_group",
            "        self.reducer = reducer",
            "        self.send_stream, self.receive_stream = anyio.create_memory_object_stream(0)",
            "        self.registry = {}",
            "        self.scopes = {}",
            "        self.values = []",
            "        self.cancelled = []",
            "        self.context_values = []",
            "        self.task_names = []",
            "",
            "    def schedule(self, request):",
            "        for item in request:",
            "            self.registry[item.key] = item",
            "        for item in request:",
            "            self.task_group.start_soon(self._worker, item)",
            "",
            "    async def _worker(self, item):",
            "        with CancelScope() as scope:",
            "            self.scopes[item.key] = scope",
            "            self.task_names.append(asyncio.current_task().get_name())",
            "            TRACE.set(f'worker-{item.key}')",
            "            self.context_values.append(TRACE.get())",
            "            if item.value < 0:",
            "                await self.send_stream.send(item)",
            "            if item.value % 2:",
            "                await self.send_stream.send(WorkItem(item.key, item.value * 2))",
            "            else:",
            "                await self.send_stream.send(WorkItem(item.key, item.value * 2))",
            "",
            "    async def cleanup(self, key):",
            "        scope = self.scopes.pop(key, None)",
            "        if scope is not None:",
            "            scope.cancel()",
            "",
            "    async def results(self, expected):",
            "        for _ in range(expected):",
            "            item = await self.receive_stream.receive()",
            "            candidate = self.reducer",
            "            assert isinstance(candidate, Reducer)",
            "            self.values.append((TRACE.get(), candidate.reduce(item)))",
            "            await self.cleanup(item.key)",
            "        return list(self.values)",
            "",
            "async def execute(values, *, fail=None, use_task_factory=False):",
            "    reducer = Reducer(fail=fail)",
            "    factory_task_types = []",
            "    factory_registry_sizes = []",
            "    runner_ref = []",
            "    loop = asyncio.get_running_loop()",
            "    previous_factory = loop.get_task_factory()",
            "    def factory(loop, coro, **kwargs):",
            "        task = asyncio.Task(coro, loop=loop, **kwargs)",
            "        factory_task_types.append(type(task).__name__)",
            "        factory_registry_sizes.append(len(runner_ref[0].registry))",
            "        return task",
            "    if use_task_factory:",
            "        loop.set_task_factory(factory)",
            "    try:",
            "        async with anyio.create_task_group() as group:",
            "            runner = Runner(group, reducer)",
            "            runner_ref.append(runner)",
            (
                "            request = [WorkItem(index + 1, value) "
                "for index, value in enumerate(values)]"
            ),
            "            token = TRACE.set('outer')",
            "            try:",
            "                runner.schedule(request)",
            "                await runner.results(len(request))",
            "                return {",
            "                    'values': list(runner.values),",
            "                    'inner_context': list(runner.context_values),",
            "                    'outer_context': TRACE.get(),",
            "                    'cancelled': list(runner.cancelled),",
            "                    'scopes_empty': not runner.scopes,",
            "                    'task_names': list(runner.task_names),",
            "                    'factory_task_types': factory_task_types,",
            "                    'factory_registry_sizes': factory_registry_sizes,",
            "                    'mode': next((",
            "                        value",
            "                        for name, value in globals().items()",
            "                        if name.startswith('_atoll_anyio_dispatch_')",
            "                        and name.endswith('_last_mode')",
            "                    ), 'source'),",
            "                }",
            "            finally:",
            "                TRACE.reset(token)",
            "    finally:",
            "        if use_task_factory:",
            "            loop.set_task_factory(previous_factory)",
            "",
        ]
    )


def _reducer_source() -> str:
    return "\n".join(
        [
            "from __future__ import annotations",
            "",
            "import inspect",
            "",
            "class Reducer:",
            "    def __init__(self, *, fail=None):",
            "        self.fail = fail",
            "        self.reducer = lambda item: item.value",
            "",
            "    def reduce(self, item):",
            "        arity = len(inspect.signature(self.reducer).parameters)",
            "        if self.fail == item.value:",
            "            raise RuntimeError('boom')",
            "        return self.reducer(item) * arity",
            "",
        ]
    )


def _direct_runner_source() -> str:
    return (
        _runner_source()
        .replace(
            "from .reducer import Reducer",
            "from .reducer import Reducer, ReducerContext",
        )
        .replace(
            "            self.values.append((TRACE.get(), candidate.reduce(item)))",
            "\n".join(
                [
                    "            current = 0",
                    "            context = ReducerContext(state=self, deps=None, current=current)",
                    "            current = candidate.reduce(context, current, item)",
                    "            self.values.append((TRACE.get(), current))",
                ]
            ),
        )
    )


def _direct_reducer_source() -> str:
    return "\n".join(
        [
            "from __future__ import annotations",
            "",
            "import inspect",
            "from collections.abc import Callable",
            "from dataclasses import dataclass",
            "from typing import cast",
            "",
            "@dataclass(init=False)",
            "class ReducerContext:",
            "    state: object",
            "    deps: object",
            "    current: int",
            "",
            "    def __init__(self, *, state, deps, current):",
            "        self.state = state",
            "        self.deps = deps",
            "        self.current = current",
            "",
            "class Reducer:",
            "    def __init__(self, *, fail=None):",
            "        self.fail = fail",
            "        self.reducer = lambda current, item: current + item.value",
            "",
            "    def reduce(self, context, current, item):",
            "        arity = len(inspect.signature(self.reducer).parameters)",
            "        if arity == 2:",
            "            return cast(Callable, self.reducer)(current, item)",
            "        else:",
            "            return cast(Callable, self.reducer)(context, current, item)",
            "",
        ]
    )


def _lossy_direct_reducer(shape: str) -> tuple[str, str]:
    source = _direct_reducer_source()
    if shape == "wrong-qualname":
        return source, "Reducer.reduce.extra"
    replacements = {
        "async-method": ("    def reduce(", "    async def reduce("),
        "varargs": (
            "def reduce(self, context, current, item):",
            "def reduce(self, context, current, item, *items):",
        ),
        "keyword-only": (
            "def reduce(self, context, current, item):",
            "def reduce(self, context, current, *, item):",
        ),
        "extra-statement": (
            "        if arity == 2:",
            "        current = current\n        if arity == 2:",
        ),
        "bad-reflection": (
            "len(inspect.signature(self.reducer).parameters)",
            "inspect.signature(self.reducer)",
        ),
        "bad-branch": ("if arity == 2:", "if arity > 2:"),
        "missing-else": (
            "        else:\n"
            "            return cast(Callable, self.reducer)(context, current, item)",
            "        return cast(Callable, self.reducer)(context, current, item)",
        ),
        "different-field": (
            "return cast(Callable, self.reducer)(context, current, item)",
            "return cast(Callable, self.other)(context, current, item)",
        ),
        "different-cast": (
            "return cast(Callable, self.reducer)(context, current, item)",
            "return self.reducer(context, current, item)",
        ),
        "wrong-short-count": ("if arity == 2:", "if arity == 1:"),
        "literal-argument": (
            "return cast(Callable, self.reducer)(current, item)",
            "return cast(Callable, self.reducer)(0, item)",
        ),
    }
    replacement = replacements.get(shape)
    if replacement is None:
        raise AssertionError(f"unknown direct reducer shape: {shape}")
    return source.replace(*replacement), "Reducer.reduce"


def _selected_plan(scans: tuple[ModuleScan, ...], *, hot_count: int = _HOT_COUNT) -> ExecutionPlan:
    results = build_execution_plans(scans, _profile(scans, hot_count=hot_count))
    rejections = tuple(result for result in results if isinstance(result, PlanRejection))
    plans = tuple(result for result in results if isinstance(result, ExecutionPlan))
    assert rejections == ()
    assert len(plans) == 1
    return plans[0]


def _profile(scans: tuple[ModuleScan, ...], *, hot_count: int) -> ProfileResult:
    members = (
        ("workflow.runner", "Runner.schedule", hot_count, 0),
        ("workflow.runner", "Runner._worker", hot_count, hot_count),
        ("workflow.runner", "Runner.results", hot_count, 0),
        ("workflow.runner", "Runner.cleanup", hot_count, 0),
        ("workflow.reducer", "Reducer.reduce", hot_count, 0),
    )
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
    invocations_by_owner = {member.symbol: member.invocation_count for member in profiled}
    total_starts = max(hot_count, sum(member.lifecycle.start for member in profiled))
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
                        f"anyio._backends._asyncio.TaskGroup.{target.scheduler_method}",
                        invocations_by_owner.get(target.owner, 0),
                    ),
                ),
            )
            for target in execution_plan_profile_targets(scans)
        ),
    )


def _payload_from_scans(
    tmp_path: Path,
    scans: tuple[ModuleScan, ...],
    *,
    name: str = "payload",
) -> Path:
    payload_root = tmp_path / name
    for scan in scans:
        destination = payload_root / _module_file(scan)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(Path(scan.module.path).read_text(encoding="utf-8"), encoding="utf-8")
    init_path = payload_root / "workflow" / "__init__.py"
    init_path.parent.mkdir(parents=True, exist_ok=True)
    init_path.write_text("", encoding="utf-8")
    return payload_root


def _scans_from_payload(payload_root: Path) -> tuple[ModuleScan, ModuleScan]:
    runner = scan_module(ModuleId("workflow.runner", payload_root / "workflow" / "runner.py"))
    runner = replace(
        runner,
        dependency_edges=(
            *runner.dependency_edges,
            DependencyEdge(
                src=SymbolId("workflow.runner", "Runner.results"),
                dst=SymbolId("workflow.runner", "Runner.cleanup"),
                kind="calls_method",
                confidence="high",
                lineno=64,
                invocation_mode="awaited",
                requires_same_unit=True,
            ),
        ),
    )
    return (
        runner,
        scan_module(ModuleId("workflow.reducer", payload_root / "workflow" / "reducer.py")),
    )


def _module_file(scan: ModuleScan) -> Path:
    return Path(*scan.module.name.split(".")).with_suffix(".py")


def _source_hashes_for(
    scans: tuple[ModuleScan, ...],
    member_ids: tuple[SymbolId, ...],
) -> tuple[tuple[str, str], ...]:
    modules = sorted({member.module for member in member_ids})
    scans_by_module = {scan.module.name: scan for scan in scans}
    return tuple(
        (
            module,
            hashlib.sha256(
                scans_by_module[module].module.path.read_text(encoding="utf-8").encode("utf-8")
            ).hexdigest(),
        )
        for module in modules
    )


def _plan_with_module_source(
    plan: ExecutionPlan,
    module_name: str,
    source_text: str,
) -> ExecutionPlan:
    source_hashes = dict(plan.source_hashes)
    source_hashes[module_name] = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    ordered = tuple(sorted(source_hashes.items()))
    return replace(
        plan,
        source_hashes=ordered,
        source_hash=_digest_parts(f"{module}:{digest}" for module, digest in ordered),
    )


def _digest_parts(parts: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _file_hashes(paths: Sequence[Path]) -> dict[Path, str]:
    return {
        path: hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
        for path in paths
    }


def _import_module(path: Path) -> ModuleType:
    package_root = path.parent.parent
    sys.path.insert(0, str(package_root))
    try:
        spec = importlib.util.spec_from_file_location("workflow.runner", path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules.pop("workflow", None)
        sys.modules.pop("workflow.runner", None)
        sys.modules.pop("workflow.reducer", None)
        sys.modules["workflow.runner"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(package_root))


def _runtime_instrumentation_active() -> bool:
    if sys.gettrace() is not None or sys.getprofile() is not None:
        return True
    monitoring = getattr(sys, "monitoring", None)
    get_events = getattr(monitoring, "get_events", None)
    if get_events is None:
        return False
    for tool_id in range(6):
        try:
            if get_events(tool_id):
                return True
        except ValueError:
            continue
    return False
