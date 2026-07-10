"""Native Cython compile and import smoke tests."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import sys
from pathlib import Path

import pytest

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.clustering import enrich_island_analysis
from atoll.backends.cython import CythonBackend
from atoll.generation.outlined_region import generate_outlined_region
from atoll.generation.region_shim import (
    RegionShimConfig,
    insert_or_replace_region_shim,
)
from atoll.generation.typed_region import (
    TypedRegionGenerationOptions,
    generate_typed_method_region,
)
from atoll.models import BackendCompileContext, BackendLoweringRequest, ModuleId

SENT_VALUE = 3
LARGE_INTEGER = 10**40


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
