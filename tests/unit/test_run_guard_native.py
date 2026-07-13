"""Tests for source-fused run-guard planning and generation."""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Protocol, cast

import pytest

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.clustering import enrich_island_analysis
from atoll.generation.run_guard import RunGuardGenerationRequest, generate_run_guard
from atoll.models import ModuleId, ModuleScan, SymbolId, TypedRegion
from atoll.native_optimization.run_guard import (
    CompletionIndexNativePlan,
    RunGuardNativePlan,
    build_run_guard_region,
    run_guard_function_source,
)
from atoll.source_optimization import anyio_stream_lowering

MODULE = "guard_subject"
IMMEDIATE_RESULT = 7
SUSPENDING_RESULT = 9


class _GuardOwner(Protocol):
    """Owner surface used to observe complete-guard and cached-helper routing."""

    _fallback: _GuardCallable
    _passed: bool
    _run_identity: object | None


class _GuardCallable(Protocol):
    def __call__(self, owner: _GuardOwner, request: object) -> bool:
        """Return whether the request may enter optimized execution."""

        ...


class _AwaitCallable(Protocol):
    async def __call__(self, owner: object, awaitable: object) -> object:
        """Await through the generated suspension-aware protocol helper."""

        ...


def _plan() -> RunGuardNativePlan:
    return RunGuardNativePlan(
        source_plan_id="source-plan",
        source=PurePosixPath("guard_subject.py"),
        owner=SymbolId(MODULE, "Owner.submit"),
        helper=SymbolId(MODULE, "_cached_guard"),
        source_guard=SymbolId(MODULE, "_source_guard"),
        eligibility_helper=SymbolId(MODULE, "_eligible"),
        protocol_context=SymbolId(MODULE, "_protocol_context"),
        disable_module=SymbolId(MODULE, "_os"),
        clear_helper=SymbolId(MODULE, "_clear_guard"),
        protocol_await_helper=SymbolId(MODULE, "_protocol_await"),
        fallback_attribute="_fallback",
        state_attribute="_passed",
        run_identity_attribute="_run_identity",
    )


def _completion_plan() -> RunGuardNativePlan:
    """Return one run guard composed with a source-maintained completion index."""
    return replace(
        _plan(),
        completion_index=CompletionIndexNativePlan(
            snapshot=SymbolId(MODULE, "_completion_snapshot"),
            query=SymbolId(MODULE, "_completion_query"),
            index_attribute="_completion_index",
            count_attribute="_completion_count",
            active_attribute="active",
            fallback_predicate_method="_is_complete",
            graph_attribute="graph",
            parent_lookup_method="get_parent",
            intermediate_nodes_attribute="intermediate_nodes",
        ),
    )


def _scan_transformed_source(tmp_path: Path) -> tuple[ModuleScan, RunGuardNativePlan]:
    source_path = tmp_path / "guard_subject.py"
    source_path.write_text(
        """from __future__ import annotations

import contextvars as _contextvars
import os as _os

_protocol_context = _contextvars.ContextVar("guard", default=None)


def _complete_guard(self: object, request: object) -> bool:
    return type(request) in (list, tuple)


_source_guard = _complete_guard


def _eligible(self: object, item: object) -> bool:
    return type(item) is int


def _clear_guard(owner: object) -> None:
    setattr(owner, "_passed", False)


async def _protocol_await(owner: object, awaitable: object) -> object:
    return await awaitable


def _cached_guard(self: object, request: object) -> bool:
    return self._fallback(self, request)


class Owner:
    def __post_init__(self) -> None:
        self._fallback = _source_guard
        self._passed = False
        self._run_identity = None

    def submit(self, request: object) -> bool:
        return _cached_guard(self, request)
""",
        encoding="utf-8",
    )
    return (
        enrich_island_analysis(scan_module(ModuleId(name=MODULE, path=source_path))),
        _plan(),
    )


def _region(tmp_path: Path) -> tuple[ModuleScan, RunGuardNativePlan, TypedRegion]:
    scan, plan = _scan_transformed_source(tmp_path)
    return scan, plan, build_run_guard_region(scan, plan)


def _completion_region(tmp_path: Path) -> tuple[ModuleScan, RunGuardNativePlan, TypedRegion]:
    source_path = tmp_path / "guard_subject.py"
    source_path.write_text(
        """from __future__ import annotations

import contextvars as _contextvars
import os as _os

_protocol_context = _contextvars.ContextVar("guard", default=None)


def _complete_guard(self: object, request: object) -> bool:
    return type(request) in (list, tuple)


_source_guard = _complete_guard


def _eligible(self: object, item: object) -> bool:
    return type(item) is int


def _clear_guard(owner: object) -> None:
    setattr(owner, "_passed", False)


async def _protocol_await(owner: object, awaitable: object) -> object:
    return await awaitable


def _cached_guard(self: object, request: object) -> bool:
    return self._fallback(self, request)


def _completion_snapshot(owner: object) -> list[object]:
    return list(owner.active.values())


def _completion_query(
    owner: object,
    active_tasks: object,
    join_id: object,
    fork_run_id: object,
) -> bool:
    return owner._is_complete(active_tasks, join_id, fork_run_id)


class Parent:
    intermediate_nodes = {"middle"}


class Graph:
    def get_parent(self, join_id: object) -> Parent:
        del join_id
        return Parent()


class Owner:
    def __post_init__(self) -> None:
        self._fallback = _source_guard
        self._passed = False
        self._run_identity = None
        self._completion_index = {}
        self._completion_count = 0
        self.active = {}
        self.graph = Graph()

    def submit(self, request: object) -> bool:
        return _cached_guard(self, request)

    def _is_complete(
        self,
        active_tasks: object,
        join_id: object,
        fork_run_id: object,
    ) -> bool:
        del join_id, fork_run_id
        return not list(active_tasks)
""",
        encoding="utf-8",
    )
    scan = enrich_island_analysis(scan_module(ModuleId(name=MODULE, path=source_path)))
    plan = _completion_plan()
    return scan, plan, build_run_guard_region(scan, plan)


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_run_guard_plan_identity_covers_run_scope_and_rejects_expressions() -> None:
    """Stable IDs cover run identity while every rendered name stays structural."""
    plan = _plan()

    assert plan.stable_id == _plan().stable_id
    assert replace(plan, run_identity_attribute="_other_run").stable_id != plan.stable_id
    with pytest.raises(ValueError, match="direct attributes"):
        replace(plan, run_identity_attribute="owner.run")

    indexed = _completion_plan()
    assert indexed.stable_id != plan.stable_id
    assert indexed.completion_index is not None
    changed_index = replace(indexed.completion_index, active_attribute="other_active")
    assert replace(indexed, completion_index=changed_index).stable_id != indexed.stable_id


def test_completion_and_run_guard_plans_reject_unrenderable_identities() -> None:
    """Structured plans reject cross-module paths and expression-shaped names."""
    plan = _completion_plan()
    assert plan.completion_index is not None
    completion = plan.completion_index
    invalid_completions = (
        {"query": SymbolId("other", "_completion_query")},
        {"snapshot": SymbolId(MODULE, "Owner.snapshot")},
        {"index_attribute": "owner.index"},
    )
    for completion_changes in invalid_completions:
        with pytest.raises(ValueError, match="completion-index"):
            replace(completion, **completion_changes)

    other_completion = replace(
        completion,
        snapshot=SymbolId("other", "_completion_snapshot"),
        query=SymbolId("other", "_completion_query"),
    )
    filesystem_root = "/"
    invalid_plans = (
        {"source_plan_id": ""},
        {"source": PurePosixPath(filesystem_root, "absolute.py")},
        {"source": PurePosixPath("../escape.py")},
        {"helper": SymbolId("other", "_cached_guard")},
        {"completion_index": other_completion},
        {"helper": SymbolId(MODULE, "Owner.guard")},
        {"owner": SymbolId(MODULE, "submit")},
        {"fallback_attribute": "owner.fallback"},
    )
    for plan_changes in invalid_plans:
        with pytest.raises(ValueError, match=r"run-guard|completion-index"):
            replace(plan, **plan_changes)


def test_run_guard_helper_reuses_only_top_level_guard_inside_private_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cached calls retain per-item identity checks and exact complete fallback."""
    plan = _plan()
    module_path = tmp_path / "guard_runtime.py"
    module_path.write_text(
        f"""import contextvars
import os as _os
import sys

_protocol_context = contextvars.ContextVar("guard", default=None)
fallback_calls = []
eligibility_calls = []


def _source_guard(owner, request):
    fallback_calls.append(request)
    return type(request) in (list, tuple) and all(type(item) is int for item in request)


def _eligible(owner, item):
    eligibility_calls.append(item)
    return type(item) is int


class Owner:
    def __init__(self):
        self._fallback = _source_guard
        self._passed = False
        self._run_identity = None


_atoll_source = sys.modules[__name__]

{run_guard_function_source(plan)}
""",
        encoding="utf-8",
    )
    source_module = _load_module(module_path, "guard_runtime")
    helper = cast(_GuardCallable, vars(source_module)[plan.helper.qualname])
    owner_factory = cast(Callable[[], _GuardOwner], vars(source_module)["Owner"])
    owner = owner_factory()
    protocol_context = vars(source_module)[plan.protocol_context.qualname]
    token = protocol_context.set(owner)
    try:
        assert helper(owner, [1, 2]) is True
        assert helper(owner, [3, 4]) is True
        fallback_calls = cast(list[object], vars(source_module)["fallback_calls"])
        eligibility_calls = cast(list[object], vars(source_module)["eligibility_calls"])
        assert fallback_calls == [[1, 2]]
        assert eligibility_calls == [3, 4]

        assert helper(owner, [5, object()]) is False
        assert fallback_calls[-1] != fallback_calls[0]

        monkeypatch.setenv("ATOLL_DISABLE", "1")
        assert helper(owner, [6]) is True
        assert fallback_calls[-1] == [6]
    finally:
        protocol_context.reset(token)

    monkeypatch.delenv("ATOLL_DISABLE")
    assert helper(owner, [7]) is True
    assert fallback_calls[-1] == [7]


def test_run_guard_region_revalidates_exact_fallback_and_initialized_state(
    tmp_path: Path,
) -> None:
    """A fresh transformed-source scan owns one helper and all fallback dependencies."""
    scan, plan, region = _region(tmp_path)

    members = {member.id: member for member in region.members}
    assert set(members) == {plan.helper, plan.eligibility_helper}
    assert "source_eligibility is not" in members[plan.helper].source_text
    assert {dependency.dst for dependency in region.dependencies}.issuperset(
        {
            plan.source_guard,
            plan.eligibility_helper,
            plan.protocol_context,
            plan.disable_module,
        }
    )
    assert build_run_guard_region(scan, plan) == region

    source = scan.module.path.read_text(encoding="utf-8")
    scan.module.path.write_text(
        source.replace("        self._run_identity = None\n", ""),
        encoding="utf-8",
    )
    stale_scan = enrich_island_analysis(scan_module(scan.module))
    with pytest.raises(ValueError, match="initialize complete run-guard state"):
        build_run_guard_region(stale_scan, plan)


def test_run_guard_generation_imports_only_transformed_source(tmp_path: Path) -> None:
    """The Cython unit does not copy target annotations or unrelated imports."""
    scan, plan, region = _region(tmp_path)
    output_path = tmp_path / "_atoll_guard.py"

    generated = generate_run_guard(
        RunGuardGenerationRequest(
            scan=scan,
            region=region,
            plan=plan,
            logical_module="_atoll_guard",
            output_path=output_path,
        )
    )

    assert generated.source_path == output_path
    assert f"import {MODULE} as _atoll_source" in generated.source_text
    assert "def _eligible(" in generated.source_text
    assert "__atoll_run_guard_expected_eligibility" in generated.source_text
    assert generated.bindings[0].source == plan.helper
    assert generated.selected_members == (plan.eligibility_helper, plan.helper)
    assert "Any" not in generated.source_text
    assert (
        generate_run_guard(
            RunGuardGenerationRequest(
                scan=scan,
                region=region,
                plan=plan,
                logical_module="_atoll_guard",
                output_path=output_path,
            )
        ).source_hash
        == generated.source_hash
    )


def test_run_guard_generation_composes_indexed_completion_helpers(tmp_path: Path) -> None:
    """One Cython unit binds the guard and indexed helpers transactionally."""
    scan, plan, region = _completion_region(tmp_path)
    assert plan.completion_index is not None
    output_path = tmp_path / "_atoll_indexed_guard.py"

    generated = generate_run_guard(
        RunGuardGenerationRequest(
            scan=scan,
            region=region,
            plan=plan,
            logical_module="_atoll_indexed_guard",
            output_path=output_path,
        )
    )

    selected = (
        plan.eligibility_helper,
        plan.helper,
        plan.completion_index.snapshot,
        plan.completion_index.query,
    )
    assert tuple(member.id for member in region.members) == selected
    assert generated.selected_members == selected
    assert {binding.source for binding in generated.bindings} == set(selected[1:])
    snapshot = next(
        member for member in region.members if member.id == plan.completion_index.snapshot
    )
    query = next(member for member in region.members if member.id == plan.completion_index.query)
    assert "return ()" in snapshot.source_text
    assert "by_node" in query.source_text


def test_run_guard_region_rejects_wrong_module_missing_member_and_binding(
    tmp_path: Path,
) -> None:
    """Fresh scan identity, members, and public bindings are mandatory."""
    scan, plan, region = _completion_region(tmp_path)
    wrong_module = replace(scan, module=replace(scan.module, name="other.module"))
    with pytest.raises(ValueError, match="different staged module"):
        build_run_guard_region(wrong_module, plan)

    assert plan.completion_index is not None
    missing_symbol = plan.completion_index.snapshot
    missing_regions = tuple(
        replace(
            item,
            members=tuple(member for member in item.members if member.id != missing_symbol),
        )
        for item in scan.typed_regions
    )
    with pytest.raises(ValueError, match="absent or ambiguous"):
        build_run_guard_region(replace(scan, typed_regions=missing_regions), plan)

    missing_bindings = tuple(
        replace(
            item,
            bindings=tuple(
                binding for binding in item.bindings if binding.source != missing_symbol
            ),
        )
        for item in scan.typed_regions
    )
    with pytest.raises(ValueError, match="requires every selected helper binding"):
        build_run_guard_region(replace(scan, typed_regions=missing_bindings), plan)

    no_decisions = replace(
        scan,
        typed_regions=tuple(replace(item, decisions=()) for item in scan.typed_regions),
    )
    fallback_region = build_run_guard_region(no_decisions, plan)
    assert fallback_region.decisions[0].action == "box"
    assert fallback_region.id == region.id


def test_run_guard_generation_rejects_incomplete_requests(tmp_path: Path) -> None:
    """Generation refuses empty module names, missing members, and missing bindings."""
    scan, plan, region = _completion_region(tmp_path)
    request = RunGuardGenerationRequest(
        scan=scan,
        region=region,
        plan=plan,
        logical_module="_atoll_guard",
        output_path=tmp_path / "_atoll_guard.py",
    )
    with pytest.raises(ValueError, match="logical module"):
        generate_run_guard(replace(request, logical_module=""))
    with pytest.raises(ValueError, match="planned helper members"):
        generate_run_guard(replace(request, region=replace(region, members=region.members[:-1])))
    with pytest.raises(ValueError, match="every public helper binding"):
        generate_run_guard(replace(request, region=replace(region, bindings=region.bindings[:-1])))

    private = vars(sys.modules[generate_run_guard.__module__])
    with pytest.raises(ValueError, match="future import"):
        private["_insert_source_identity_capture"]("def helper():\n    pass\n", request)
    inserted = private["_insert_source_identity_capture"](
        "from __future__ import annotations\n\ndef helper():\n    pass\n",
        request,
    )
    assert f"import {MODULE} as _atoll_source" in inserted


def test_run_guard_ast_helpers_cover_annotated_and_absent_initializers() -> None:
    """Validation helpers classify annotated assignments and missing initializers."""
    private = vars(sys.modules[build_run_guard_region.__module__])
    tree = ast.parse(
        "from package import value\nname: int = 1\nasync def operation():\n    return None\n"
    )
    assert private["_module_bound_names"](tree) == {"value", "name", "operation"}
    assignment = ast.parse("self.state: dict = {}\n").body[0]
    assert private["_assigned_attribute"](assignment) == "state"
    assert private["_initializes_empty_mapping"](None, "state") is False
    assert private["_initializes_zero"](None, "count") is False
    initializer = ast.parse(
        "def initialize(self):\n    self.state: dict = {}\n    self.count: int = 0\n"
    ).body[0]
    assert isinstance(initializer, ast.FunctionDef)
    assert private["_initializes_empty_mapping"](initializer, "state") is True
    assert private["_initializes_zero"](initializer, "count") is True


def test_run_guard_rejects_stale_completion_fallback_or_index(tmp_path: Path) -> None:
    """A changed scan delegate or nonempty initial index invalidates the native plan."""
    scan, plan, _region = _completion_region(tmp_path)
    source = scan.module.path.read_text(encoding="utf-8")
    scan.module.path.write_text(
        source.replace(
            "    return list(owner.active.values())\n",
            "    return []\n",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="completion fallback must remain one synchronous return"):
        build_run_guard_region(enrich_island_analysis(scan_module(scan.module)), plan)

    scan.module.path.write_text(
        source.replace(
            "        self._completion_index = {}\n", "        self._completion_index = {1: {}}\n"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="empty completion index"):
        build_run_guard_region(enrich_island_analysis(scan_module(scan.module)), plan)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("def _clear_guard", "def _missing_clear_guard", "support is incomplete"),
        ("import os as _os", "import os as _other_os", "dependency is absent"),
        ("class Owner:", "class MissingOwner:", "owner class is absent"),
        (
            "return _cached_guard(self, request)",
            "return True",
            "no longer calls the planned",
        ),
        ("def _is_complete(", "def _other_complete(", "lost the original"),
        ("self._completion_count = 0", "self._completion_count = 1", "zero completion"),
        (
            "def _completion_snapshot(owner: object)",
            "def _completion_snapshot(owner: object, extra: object)",
            "snapshot fallback signature",
        ),
        ("owner.active.values()", "owner.other.values()", "no longer scans"),
        (
            "fork_run_id: object,\n) -> bool:\n    return owner._is_complete",
            "fork_run_id: object,\n    extra: object,\n) -> bool:\n    return owner._is_complete",
            "query fallback signature",
        ),
        ("owner._is_complete(", "owner._other_complete(", "original predicate"),
        ("def _cached_guard", "async def _cached_guard", "one synchronous return"),
        (
            "def _cached_guard(self: object, request: object)",
            "def _cached_guard(self: object, request: object, extra: object)",
            "fallback signature changed",
        ),
        (
            "return self._fallback(self, request)",
            "return True",
            "no longer delegates directly",
        ),
        ("self._fallback(self, request)", "self._other(self, request)", "retained source guard"),
    ],
)
def test_run_guard_revalidation_rejects_stale_source_contracts(
    tmp_path: Path,
    old: str,
    new: str,
    message: str,
) -> None:
    """Every changed source route invalidates the transactional native unit."""
    scan, plan, _region = _completion_region(tmp_path)
    source = scan.module.path.read_text(encoding="utf-8")
    assert old in source
    scan.module.path.write_text(source.replace(old, new, 1), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=message):
        build_run_guard_region(enrich_island_analysis(scan_module(scan.module)), plan)


def test_protocol_await_invalidates_only_after_real_suspension(tmp_path: Path) -> None:
    """Immediate protocol steps retain the run guard; scheduler handoff clears it."""
    private = vars(anyio_stream_lowering)
    names = private["_names"]("run-guard-test")
    module_path = tmp_path / "protocol_await_runtime.py"
    module_path.write_text(
        f"""import types as {names.types}


def {names.guard}(owner, request):
    return True


{private["_run_guard_support"](names)}
""",
        encoding="utf-8",
    )
    module = _load_module(module_path, "protocol_await_runtime")
    protocol_await = cast(_AwaitCallable, vars(module)[names.protocol_await])

    class Owner:
        pass

    owner = Owner()
    setattr(owner, names.guard_state_attribute, True)

    async def immediate() -> int:
        return IMMEDIATE_RESULT

    assert asyncio.run(protocol_await(owner, immediate())) == IMMEDIATE_RESULT
    assert getattr(owner, names.guard_state_attribute) is True

    async def suspending() -> int:
        await asyncio.sleep(0)
        return SUSPENDING_RESULT

    assert asyncio.run(protocol_await(owner, suspending())) == SUSPENDING_RESULT
    assert getattr(owner, names.guard_state_attribute) is False
