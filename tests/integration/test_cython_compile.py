"""Native Cython compile and import smoke tests."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import sys
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath
from typing import Protocol, cast

import pytest

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.clustering import enrich_island_analysis
from atoll.backends.cython import CythonBackend
from atoll.generation.outlined_region import generate_outlined_region
from atoll.generation.region_shim import (
    RegionShimConfig,
    insert_or_replace_region_shim,
)
from atoll.generation.run_guard import RunGuardGenerationRequest, generate_run_guard
from atoll.generation.scalar_kernel import (
    ScalarKernelGenerationRequest,
    generate_scalar_kernel,
)
from atoll.generation.typed_region import (
    TypedRegionGenerationOptions,
    generate_typed_method_region,
)
from atoll.models import (
    BackendCompileContext,
    BackendLoweringRequest,
    CompilationUnit,
    ModuleId,
    SymbolId,
)
from atoll.native_optimization.run_guard import (
    CompletionIndexNativePlan,
    RunGuardNativePlan,
    build_run_guard_region,
)
from atoll.native_optimization.scalar_analysis import analyze_scalar_scan

SENT_VALUE = 3
LARGE_INTEGER = 10**40
SCALAR_VALUE = 12
SCALAR_BIAS = 3
SCALAR_EXPECTED = 147
SCALAR_BOOL_EXPECTED = 4
BATCH_SECOND_VALUE = 2


class _RunGuardOwner(Protocol):
    _passed: bool
    _completion_index: dict[object, dict[object, int]]
    _completion_count: int
    active: dict[object, object]
    eligibility_calls: list[object]

    def __post_init__(self) -> None:
        """Initialize the transformed helper state."""


class _RunGuardCallable(Protocol):
    def __call__(self, owner: _RunGuardOwner, request: object) -> bool:
        """Return whether the request may enter the accepted source fast path."""

        ...


class _CompletionSnapshotCallable(Protocol):
    def __call__(self, owner: _RunGuardOwner) -> Sequence[object]:
        """Return the native empty snapshot replacing a proven indexed scan."""

        ...


class _CompletionQueryCallable(Protocol):
    def __call__(
        self,
        owner: _RunGuardOwner,
        active_tasks: list[object],
        join_id: object,
        fork_run_id: object,
    ) -> bool:
        """Return whether the maintained index proves one fork run complete."""

        ...


class _ContextVariable(Protocol):
    def set(self, value: object) -> object:
        """Set the private protocol owner and return its reset token."""

    def reset(self, token: object) -> None:
        """Restore the previous private protocol owner."""


def _assert_indexed_completion_routing(
    owner: _RunGuardOwner,
    snapshot: _CompletionSnapshotCallable,
    query: _CompletionQueryCallable,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise indexed routing, stale-count fallback, and predicate replacement."""
    owner.active["active"] = object()
    state = vars(owner)
    completion_index = cast(dict[object, dict[object, int]], state["_completion_index"])
    completion_index["run"] = {"join": 1}
    state["_completion_count"] = 1
    assert snapshot(owner) == ()
    assert query(owner, list(owner.active.values()), "join", "run") is False
    completion_index["run"] = {"other": 1}
    assert query(owner, list(owner.active.values()), "join", "run") is True
    state["_completion_count"] = 0
    assert snapshot(owner) == list(owner.active.values())
    state["_completion_count"] = 1

    def replaced_predicate(
        _owner: object,
        _active_tasks: list[object],
        _join_id: object,
        _fork_run_id: object,
    ) -> bool:
        return False

    monkeypatch.setattr(type(owner), "_is_complete", replaced_predicate)
    assert query(owner, [], "join", "missing") is False


def test_cython_batches_two_units_with_owned_artifacts(tmp_path: Path) -> None:
    """One adapter invocation builds two independently owned importable extensions."""
    backend = CythonBackend()
    units: list[CompilationUnit] = []
    logical_modules = ("_atoll_batch_one", "_atoll_batch_two")
    for index, logical_module in enumerate(logical_modules, start=1):
        source_path = tmp_path / f"batch_source_{index}.py"
        source_path.write_text(
            f"def value() -> int:\n    return {index}\n",
            encoding="utf-8",
        )
        scan = enrich_island_analysis(
            scan_module(ModuleId(name=f"batch_source_{index}", path=source_path))
        )
        region = next(
            region
            for region in scan.typed_regions
            if any(member.id.qualname == "value" for member in region.members)
        )
        units.append(
            backend.lower(
                BackendLoweringRequest(
                    region=region,
                    source_path=source_path,
                    logical_module=logical_module,
                    install_relative_dir=f"compiled/{index}",
                    members=tuple(member.id for member in region.members),
                    variant_id=f"{region.id}@cython-batch-{index}",
                )
            )
        )

    result = backend.compile(
        tuple(units),
        BackendCompileContext(
            project_root=tmp_path,
            build_dir=tmp_path / ".atoll" / "build",
            source_roots=(tmp_path,),
        ),
    )

    assert result.attempt.success is True, result.attempt.stderr
    assert result.attempt.command[0] == "cython"
    assert result.attempt.command[-1] == "build_ext"
    build_ext_timing = next(
        timing for timing in result.attempt.phase_timings if timing.name == "build_ext"
    )
    assert build_ext_timing.detail is not None
    assert build_ext_timing.detail.endswith("; 2 extension(s)")
    assert len(result.attempt.artifact_paths) == len(logical_modules)
    primary = tuple(record for record in result.artifacts if record.role == "primary")
    assert len(primary) == len(logical_modules)
    assert {record.region_id for record in primary} == {unit.region_id for unit in units}
    assert {record.install_relative_path.split("/", maxsplit=2)[1] for record in primary} == {
        "1",
        "2",
    }
    artifact_root = tmp_path / ".atoll" / "artifacts"
    sys.path.insert(0, str(artifact_root))
    try:
        assert importlib.import_module(logical_modules[0]).value() == 1
        assert importlib.import_module(logical_modules[1]).value() == BATCH_SECOND_VALUE
    finally:
        sys.path.remove(str(artifact_root))
        for logical_module in logical_modules:
            sys.modules.pop(logical_module, None)


def test_cython_compiles_source_fused_run_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A boxed helper amortizes one guard while retaining per-item and disable checks."""
    module_name = "cython_run_guard_source"
    source_path = tmp_path / f"{module_name}.py"
    source_path.write_text(
        """from __future__ import annotations

import contextvars as _contextvars
import os as _os

_protocol_context = _contextvars.ContextVar("guard", default=None)
fallback_calls = []


def _complete_guard(self: object, request: object) -> bool:
    fallback_calls.append(request)
    return type(request) in (list, tuple) and all(_eligible(self, item) for item in request)


_source_guard = _complete_guard


def _eligible(self: object, item: object) -> bool:
    self.eligibility_calls.append(item)
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
    active_tasks: list[object],
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
        self.eligibility_calls = []

    def submit(self, request: object) -> bool:
        return _cached_guard(self, request)

    def _is_complete(
        self,
        active_tasks: list[object],
        join_id: object,
        fork_run_id: object,
    ) -> bool:
        del join_id, fork_run_id
        return not active_tasks
""",
        encoding="utf-8",
    )
    scan = enrich_island_analysis(scan_module(ModuleId(name=module_name, path=source_path)))
    plan = RunGuardNativePlan(
        source_plan_id="source-plan",
        source=PurePosixPath(source_path.name),
        owner=SymbolId(module_name, "Owner.submit"),
        helper=SymbolId(module_name, "_cached_guard"),
        source_guard=SymbolId(module_name, "_source_guard"),
        eligibility_helper=SymbolId(module_name, "_eligible"),
        protocol_context=SymbolId(module_name, "_protocol_context"),
        disable_module=SymbolId(module_name, "_os"),
        clear_helper=SymbolId(module_name, "_clear_guard"),
        protocol_await_helper=SymbolId(module_name, "_protocol_await"),
        fallback_attribute="_fallback",
        state_attribute="_passed",
        run_identity_attribute="_run_identity",
        completion_index=CompletionIndexNativePlan(
            snapshot=SymbolId(module_name, "_completion_snapshot"),
            query=SymbolId(module_name, "_completion_query"),
            index_attribute="_completion_index",
            count_attribute="_completion_count",
            active_attribute="active",
            fallback_predicate_method="_is_complete",
            graph_attribute="graph",
            parent_lookup_method="get_parent",
            intermediate_nodes_attribute="intermediate_nodes",
        ),
    )
    assert plan.completion_index is not None
    region = build_run_guard_region(scan, plan)
    logical_module = "_atoll_cython_run_guard"
    variant_id = f"{plan.stable_id}@cython-source-fused"
    generated = generate_run_guard(
        RunGuardGenerationRequest(
            scan=scan,
            region=region,
            plan=plan,
            logical_module=logical_module,
            output_path=tmp_path / f"{logical_module}.py",
        )
    )
    backend = CythonBackend()
    unit = backend.lower(
        BackendLoweringRequest(
            region=region,
            source_path=generated.source_path,
            logical_module=logical_module,
            install_relative_dir="compiled",
            members=generated.selected_members,
            variant_id=variant_id,
        )
    )

    assert generated.source_path.suffix == ".py"
    assert unit.members == generated.selected_members
    result = backend.compile(
        (unit,),
        BackendCompileContext(
            project_root=tmp_path,
            build_dir=tmp_path / ".atoll" / "build",
            source_roots=(tmp_path,),
        ),
    )

    assert result.attempt.success is True, result.attempt.stderr
    artifact_root = tmp_path / ".atoll" / "artifacts"
    monkeypatch.setattr(sys, "path", [str(artifact_root), str(tmp_path), *sys.path])
    sys.modules.pop(module_name, None)
    sys.modules.pop(logical_module, None)
    source_module = importlib.import_module(module_name)
    native_module = importlib.import_module(logical_module)
    owner_factory = cast(Callable[[], _RunGuardOwner], vars(source_module)["Owner"])
    helper = cast(_RunGuardCallable, vars(native_module)[plan.helper.qualname])
    snapshot = cast(
        _CompletionSnapshotCallable,
        vars(native_module)[plan.completion_index.snapshot.qualname],
    )
    query = cast(
        _CompletionQueryCallable,
        vars(native_module)[plan.completion_index.query.qualname],
    )
    owner = owner_factory()
    owner.__post_init__()
    protocol_context = cast(
        _ContextVariable,
        vars(source_module)[plan.protocol_context.qualname],
    )
    token = protocol_context.set(owner)
    try:
        assert helper(owner, [1, 2]) is True
        assert helper(owner, [3, 4]) is True
        fallback_calls = cast(list[object], vars(source_module)["fallback_calls"])
        assert fallback_calls == [[1, 2]]
        assert owner.eligibility_calls == [1, 2, 3, 4]

        _assert_indexed_completion_routing(owner, snapshot, query, monkeypatch)

        replacement_calls: list[object] = []

        def replacement(_owner: object, item: object) -> bool:
            replacement_calls.append(item)
            return False

        original_eligibility = vars(source_module)[plan.eligibility_helper.qualname]
        setattr(source_module, plan.eligibility_helper.qualname, replacement)
        assert helper(owner, [6]) is False
        assert fallback_calls[-1] == [6]
        assert replacement_calls == [6]
        setattr(source_module, plan.eligibility_helper.qualname, original_eligibility)

        monkeypatch.setenv("ATOLL_DISABLE", "1")
        assert helper(owner, [5]) is True
        assert fallback_calls[-1] == [5]
    finally:
        protocol_context.reset(token)


def test_cython_compiles_guarded_fixed_width_scalar_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A generated .pyx kernel routes exact safe ints and falls back before entry."""
    module_name = "cython_scalar_runtime"
    source_path = tmp_path / f"{module_name}.py"
    source = """def polynomial(value: int, *, bias: int = 1) -> int:
    return value * value + bias
"""
    source_path.write_text(source, encoding="utf-8")
    scan = enrich_island_analysis(scan_module(ModuleId(name=module_name, path=source_path)))
    plan = analyze_scalar_scan(scan).plans[0]
    region = next(item for item in scan.typed_regions if item.id == plan.region_id)
    proof = plan.width_proofs[0]
    logical_module = "_atoll_cython_scalar_runtime"
    variant_id = f"{plan.id}@cython-i32"
    generated = generate_scalar_kernel(
        ScalarKernelGenerationRequest(
            scan=scan,
            region=region,
            plan=plan,
            width_proof=proof,
            logical_module=logical_module,
            output_path=tmp_path / f"{logical_module}.pyx",
        )
    )
    backend = CythonBackend()
    unit = backend.lower(
        BackendLoweringRequest(
            region=region,
            source_path=generated.generation.source_path,
            logical_module=logical_module,
            install_relative_dir="compiled",
            members=(plan.member,),
            variant_id=variant_id,
        )
    )
    result = backend.compile(
        (unit,),
        BackendCompileContext(
            project_root=tmp_path,
            build_dir=tmp_path / ".atoll" / "build",
            source_roots=(tmp_path,),
        ),
    )

    assert result.attempt.success is True, result.attempt.stderr
    config = RegionShimConfig(
        source_module=module_name,
        source_path=source_path,
        region_id=region.id,
        variant_id=variant_id,
        backend="cython",
        compiled_module=logical_module,
        artifact_dir=tmp_path / ".atoll" / "artifacts",
        bindings=generated.generation.bindings,
        dispatch_rank=10,
        variant_guards=proof.guards,
    )
    source_path.write_text(
        insert_or_replace_region_shim(source, (config,)).new_text,
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)

    assert module.polynomial(SCALAR_VALUE, bias=SCALAR_BIAS) == SCALAR_EXPECTED
    assert module.polynomial(True, bias=SCALAR_BIAS) == SCALAR_BOOL_EXPECTED
    assert module.polynomial(LARGE_INTEGER, bias=SCALAR_BIAS) == (
        LARGE_INTEGER * LARGE_INTEGER + SCALAR_BIAS
    )
    candidate = module.polynomial.__atoll_binding_variants__[0]
    compiled_target = candidate["target"]

    def native_probe(*_args: object, **_kwargs: object) -> int:
        raise RuntimeError("native route")

    candidate["target"] = native_probe
    with pytest.raises(RuntimeError, match="native route"):
        module.polynomial(SCALAR_VALUE, bias=SCALAR_BIAS)
    assert module.polynomial(True, bias=SCALAR_BIAS) == SCALAR_BOOL_EXPECTED
    assert module.polynomial(LARGE_INTEGER, bias=SCALAR_BIAS) == (
        LARGE_INTEGER * LARGE_INTEGER + SCALAR_BIAS
    )
    candidate["target"] = compiled_target


def test_cython_outlined_coroutine_routes_through_native_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A staged Python coroutine shell owns await while native code owns its hot block."""
    module_name = "cython_outlined_runtime"
    source_path = tmp_path / f"{module_name}.py"
    source = """import asyncio

DEFAULT = (1, 2)


async def checkpoint() -> None:
    await asyncio.sleep(0)


async def workload(
    values: list[int], token: object = DEFAULT
) -> tuple[int, object]:
    start = len(values) + 1
    doubled = start * 2
    total = doubled + 3
    await checkpoint()
    return total, token
"""
    source_path.write_text(source, encoding="utf-8")
    scan = enrich_island_analysis(scan_module(ModuleId(name=module_name, path=source_path)))
    region = next(
        item
        for item in scan.typed_regions
        if any(member.id.qualname == "workload" for member in item.members)
    )
    member = next(member for member in region.members if member.id.qualname == "workload")
    binding = next(binding for binding in region.bindings if binding.source == member.id)
    logical_module = "_atoll_cython_outlined_runtime"
    variant_id = f"{region.id}@cython-outline"
    generation = generate_outlined_region(
        region,
        member.id,
        binding,
        output_path=tmp_path / f"{logical_module}.py",
    )
    backend = CythonBackend()
    unit = backend.lower(
        BackendLoweringRequest(
            region=region,
            source_path=generation.source_path,
            logical_module=logical_module,
            install_relative_dir="compiled",
            members=(member.id,),
            variant_id=variant_id,
        )
    )
    result = backend.compile(
        (unit,),
        BackendCompileContext(
            project_root=tmp_path,
            build_dir=tmp_path / ".atoll" / "build",
            source_roots=(tmp_path,),
        ),
    )

    assert result.attempt.success is True, result.attempt.stderr
    assert result.attempt.artifact_paths
    config = RegionShimConfig(
        source_module=module_name,
        source_path=source_path,
        region_id=variant_id,
        backend="cython",
        compiled_module=logical_module,
        artifact_dir=tmp_path / ".atoll" / "artifacts",
        bindings=(binding,),
        outlined_shell=generation.shell,
    )
    source_path.write_text(
        insert_or_replace_region_shim(source, (config,)).new_text,
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])
    monkeypatch.setenv("ATOLL_DISABLE", "1")
    sys.modules.pop(module_name, None)
    disabled = importlib.import_module(module_name)
    assert not hasattr(disabled.workload, "__atoll_compiled_target__")

    monkeypatch.delenv("ATOLL_DISABLE")
    sys.modules.pop(module_name, None)
    enabled = importlib.import_module(module_name)
    compiled_target = enabled.workload.__atoll_compiled_target__

    assert inspect.iscoroutinefunction(enabled.workload)
    assert inspect.iscoroutinefunction(compiled_target)
    assert inspect.signature(enabled.workload).parameters["token"].default is enabled.DEFAULT
    assert asyncio.run(enabled.workload([1, 2])) == (9, enabled.DEFAULT)
    assert len(compiled_target.__atoll_native_helpers__) == 1
    assert enabled.__atoll_status__["regions"][variant_id]["active"] is True


def test_cython_compiles_and_imports_async_generator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bundled Cython preserves direct async-generator protocol operations."""
    source_path = tmp_path / "cython_asyncgen.py"
    source_path.write_text(
        """from __future__ import annotations

from collections.abc import AsyncIterator


async def count(limit: int) -> AsyncIterator[int]:
    value = 0
    while value < limit:
        received = yield value
        if received is None:
            value += 1
        else:
            value = received
""",
        encoding="utf-8",
    )
    scan = enrich_island_analysis(scan_module(ModuleId(name="cython_asyncgen", path=source_path)))
    region = next(
        region
        for region in scan.typed_regions
        if any(member.id.qualname == "count" for member in region.members)
    )
    backend = CythonBackend()
    assessment = backend.assess(region)
    unit = backend.lower(
        BackendLoweringRequest(
            region=region,
            source_path=source_path,
            logical_module="cython_asyncgen",
            install_relative_dir="compiled",
        )
    )

    result = backend.compile(
        (unit,),
        BackendCompileContext(
            project_root=tmp_path,
            build_dir=tmp_path / ".atoll" / "build",
            source_roots=(tmp_path,),
        ),
    )

    assert assessment.status == "supported"
    assert "async_generator" in assessment.capabilities
    assert result.attempt.success is True, result.attempt.stderr
    assert result.attempt.artifact_paths
    assert {timing.name for timing in result.attempt.phase_timings} == {
        "cythonize",
        "build_ext",
        "artifact_discovery",
    }
    assert any(record.role == "primary" for record in result.artifacts)
    assert all(record.install_relative_path.startswith("compiled/") for record in result.artifacts)

    artifact_dir = tmp_path / ".atoll" / "artifacts"
    monkeypatch.setattr(sys, "path", [str(artifact_dir), *sys.path])
    sys.modules.pop("cython_asyncgen", None)
    module = importlib.import_module("cython_asyncgen")

    async def exercise_protocol() -> None:
        stream = module.count(5)
        assert await stream.__anext__() == 0
        assert await stream.asend(None) == 1
        assert await stream.asend(SENT_VALUE) == SENT_VALUE
        with pytest.raises(ValueError, match="stop"):
            await stream.athrow(ValueError("stop"))

        closing_stream = module.count(2)
        assert await closing_stream.__anext__() == 0
        await closing_stream.aclose()
        with pytest.raises(StopAsyncIteration):
            await closing_stream.__anext__()

    asyncio.run(exercise_protocol())


def test_cython_compiles_boxed_any_incomplete_and_typevar_callables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whole-callable lowering preserves Python objects without inventing annotations."""
    source_path = tmp_path / "cython_boxed.py"
    source_path.write_text(
        """from __future__ import annotations

import typing
from typing import TypeVar

T = TypeVar("T")


def dynamic(value: typing.Any) -> typing.Any:
    return value


def incomplete(value):
    return value + 1


def identity(value: T) -> T:
    return value


def workload(value: Any) -> Any:
    return identity(incomplete(dynamic(value)))
""",
        encoding="utf-8",
    )
    scan = enrich_island_analysis(scan_module(ModuleId(name="cython_boxed", path=source_path)))
    backend = CythonBackend()
    region = next(
        region
        for region in scan.typed_regions
        if any(member.id.qualname == "workload" for member in region.members)
    )
    unit = backend.lower(
        BackendLoweringRequest(
            region=region,
            source_path=source_path,
            logical_module="cython_boxed",
            install_relative_dir="compiled",
        )
    )

    result = backend.compile(
        (unit,),
        BackendCompileContext(
            project_root=tmp_path,
            build_dir=tmp_path / ".atoll" / "build",
            source_roots=(tmp_path,),
        ),
    )

    assert backend.assess(region).status == "supported"
    assert result.attempt.success is True, result.attempt.stderr
    artifact_dir = tmp_path / ".atoll" / "artifacts"
    monkeypatch.setattr(sys, "path", [str(artifact_dir), *sys.path])
    sys.modules.pop("cython_boxed", None)
    module = importlib.import_module("cython_boxed")
    marker = object()
    assert module.dynamic(marker) is marker
    assert module.incomplete(LARGE_INTEGER) == LARGE_INTEGER + 1
    assert module.identity(marker) is marker


def test_cython_compiles_generated_pep695_callable_without_type_erasure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cython-private PEP 695 lowering executes while retaining `T` annotations."""
    source_path = tmp_path / "pep695_source.py"
    source_path.write_text(
        """from __future__ import annotations

def identity[T](value: T) -> T:
    return value
""",
        encoding="utf-8",
    )
    scan = enrich_island_analysis(scan_module(ModuleId(name="pep695_source", path=source_path)))
    region = scan.typed_regions[0]
    member = region.members[0].id
    generated_path = tmp_path / "pep695_generated.py"
    generation = generate_typed_method_region(
        scan,
        region,
        (member,),
        output_path=generated_path,
        options=TypedRegionGenerationOptions(backend="cython"),
    )
    backend = CythonBackend()
    unit = backend.lower(
        BackendLoweringRequest(
            region=region,
            source_path=generated_path,
            logical_module="pep695_generated",
            install_relative_dir="compiled",
            members=(member,),
        )
    )

    result = backend.compile(
        (unit,),
        BackendCompileContext(
            project_root=tmp_path,
            build_dir=tmp_path / ".atoll" / "build",
            source_roots=(tmp_path,),
        ),
    )

    assert "identity[T]" not in generation.source_text
    assert "def identity(value: T) -> T:" in generation.source_text
    assert "Any" not in generation.source_text
    assert result.attempt.success is True, result.attempt.stderr
    artifact_dir = tmp_path / ".atoll" / "artifacts"
    monkeypatch.setattr(sys, "path", [str(artifact_dir), *sys.path])
    sys.modules.pop("pep695_generated", None)
    module = importlib.import_module("pep695_generated")
    marker = object()
    assert module.identity(marker) is marker
