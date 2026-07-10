"""Source-clean wheel acceptance for typed method regions."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import shutil
import sys
import zipfile
from collections.abc import AsyncGenerator
from pathlib import Path
from types import ModuleType
from typing import Protocol, TypedDict, cast

import pytest

from atoll.cli import main

FIXTURE_ROOT = Path("tests/fixtures/typed_region_project")
COMPILED_METHOD_COUNT = 8
COMPILED_REGION_COUNT = 4
SCALE_RESULT = 23
PARSED_RESULT = 7
ADJUSTED_RESULT = 6
DYNAMIC_RESULT = 11
ASYNC_FIRST_RESULT = 4
ASYNC_SENT_RESULT = 8
ASYNC_THROWN_RESULT = 2


class _ExchangeWorker(Protocol):
    """Typed runtime view of the fixture worker's async generator."""

    closed: bool

    def exchange(self, start: int) -> AsyncGenerator[int, int | None]:
        """Return the fixture async generator."""
        ...


class _GuardReport(TypedDict):
    parameter_name: str
    positional_index: int | None
    annotation: str
    nominal_type_paths: list[str]
    allow_none: bool


class _BindingReport(TypedDict):
    source: str
    kind: str
    owner_class: str | None
    target_owner_class: str | None
    execution_kind: str
    guards: list[_GuardReport]


class _CompiledRegionReport(TypedDict):
    backend: str | None
    variant_id: str
    bindings: list[_BindingReport]
    artifacts: list[str]


class _SpecializedTypeBindingReport(TypedDict):
    concrete: bool


class _SpecializationReport(TypedDict):
    source_member: str
    origin: str
    substitutions: list[list[str]]
    type_bindings: list[_SpecializedTypeBindingReport]


class _TypedRegionReport(TypedDict):
    specializations: list[_SpecializationReport]


class _CompileReport(TypedDict):
    summary: dict[str, int]
    build: dict[str, object]
    compiled_regions: list[_CompiledRegionReport]
    typed_regions: list[_TypedRegionReport]


def test_compile_builds_and_routes_typed_methods_without_source_edits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A staged wheel binds methods while dynamic owners remain interpreted."""
    project_root = tmp_path / "typed_region_project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    original_hashes = _source_contents(project_root)

    exit_code = main(
        [
            "compile",
            "typed_region_project.worker",
            "--root",
            str(project_root),
            "--keep-install-tree",
        ]
    )

    output_dir = project_root / ".atoll" / "dist"
    install_root = output_dir / "install"
    wheel_path = next(output_dir.glob("*.whl"))
    report = cast(
        _CompileReport,
        json.loads((project_root / ".atoll" / "compile-report.json").read_text()),
    )
    markdown_report = (project_root / ".atoll" / "compile-report.md").read_text()
    assert exit_code == 0
    assert _source_contents(project_root) == original_hashes
    _assert_worker_compile_report(report)
    assert "-> IntPairer (instance_method, 1 guard(s))" in markdown_report
    assert "-> PayloadPairer (instance_method, 1 guard(s))" in markdown_report
    assert not (output_dir / "build").exists()
    assert not tuple(install_root.rglob("_atoll_region_*.py"))
    assert tuple((install_root / ".atoll" / "artifacts").glob("*.so"))
    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())
    assert "typed_region_project/worker.py" in names
    assert not any(name.endswith(".py") and "_atoll_region_" in name for name in names)
    assert any(name.startswith(".atoll/artifacts/_atoll_region_") for name in names)

    monkeypatch.setattr(sys, "path", [str(install_root), *sys.path])
    monkeypatch.setenv("ATOLL_REQUIRE_COMPILED", "1")
    _clear_fixture_modules()
    worker_module = importlib.import_module("typed_region_project.worker")
    _assert_compiled_worker_module(worker_module)

    monkeypatch.setenv("ATOLL_DISABLE", "1")
    _clear_fixture_modules()
    disabled_module = importlib.import_module("typed_region_project.worker")
    assert disabled_module.Worker(3).scale(5) == SCALE_RESULT
    assert disabled_module.__atoll_status__["compiled"] is False
    assert not hasattr(disabled_module.Worker.scale, "__atoll_compiled_target__")
    assert not hasattr(disabled_module.Worker.exchange, "__atoll_compiled_target__")
    assert "pair" not in vars(disabled_module.IntPairer)
    assert "maybe_pair" not in vars(disabled_module.PayloadPairer)
    assert not hasattr(disabled_module.Pairer.pair, "__atoll_compiled_target__")


def test_compile_routes_closed_generic_function_with_guarded_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A closed internal call specializes one public generic function safely."""
    project_root = tmp_path / "typed_region_project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    original_hashes = _source_contents(project_root)

    exit_code = main(
        [
            "compile",
            "typed_region_project.generic_functions",
            "--root",
            str(project_root),
            "--keep-install-tree",
        ]
    )

    output_dir = project_root / ".atoll" / "dist"
    install_root = output_dir / "install"
    report = cast(
        _CompileReport,
        json.loads((project_root / ".atoll" / "compile-report.json").read_text()),
    )
    markdown_report = (project_root / ".atoll" / "compile-report.md").read_text()
    compiled = next(
        region
        for region in report["compiled_regions"]
        if any(
            binding["source"] == "typed_region_project.generic_functions::pair_value"
            for binding in region["bindings"]
        )
    )
    specialization = next(
        item
        for region in report["typed_regions"]
        for item in region["specializations"]
        if item["source_member"] == "typed_region_project.generic_functions::pair_value"
    )
    assert exit_code == 0
    assert _source_contents(project_root) == original_hashes
    assert compiled["backend"] == "mypyc"
    assert compiled["bindings"][0]["kind"] == "module"
    assert compiled["bindings"][0]["target_owner_class"] is None
    assert compiled["bindings"][0]["guards"][0]["nominal_type_paths"] == ["int"]
    assert specialization["origin"] == "closed_call"
    assert specialization["substitutions"] == [["T", "int"]]
    assert all(binding["concrete"] for binding in specialization["type_bindings"])
    assert "pair_value (module, 1 guard(s))" in markdown_report

    monkeypatch.setattr(sys, "path", [str(install_root), *sys.path])
    monkeypatch.setenv("ATOLL_REQUIRE_COMPILED", "1")
    _clear_fixture_modules()
    module = importlib.import_module("typed_region_project.generic_functions")
    assert module.pair_int(5) == (5, 5)
    assert module.pair_value(4) == (4, 4)
    assert module.pair_value(value=4) == (4, 4)
    assert module.pair_value("fallback") == ("fallback", "fallback")
    assert hasattr(module.pair_value, "__atoll_compiled_target__")
    assert module.pair_value.__atoll_runtime_guards__[0]["types"] == (int,)
    assert inspect.signature(module.pair_value).return_annotation == "tuple[T, T]"

    monkeypatch.setenv("ATOLL_DISABLE", "1")
    _clear_fixture_modules()
    disabled_module = importlib.import_module("typed_region_project.generic_functions")
    assert disabled_module.pair_value(4) == (4, 4)
    assert not hasattr(disabled_module.pair_value, "__atoll_compiled_target__")


def _assert_worker_compile_report(report: _CompileReport) -> None:
    summary = report["summary"]
    assert summary["symbols"] == COMPILED_METHOD_COUNT
    assert summary["islands"] == 0
    assert summary["compiled_regions"] == COMPILED_REGION_COUNT
    assert summary["native_rejected_symbols"] == 1
    assert report["build"]["command"] == ["atoll", "typed-region-build"]
    compiled_regions = report["compiled_regions"]
    worker_mypyc = next(
        region
        for region in compiled_regions
        if any(
            binding["source"] == "typed_region_project.worker::Worker.scale"
            for binding in region["bindings"]
        )
    )
    pairer_mypyc = next(
        region
        for region in compiled_regions
        if any(binding["target_owner_class"] == "IntPairer" for binding in region["bindings"])
    )
    payload_mypyc = next(
        region
        for region in compiled_regions
        if any(binding["target_owner_class"] == "PayloadPairer" for binding in region["bindings"])
    )
    cython_region = next(region for region in compiled_regions if region["backend"] == "cython")
    assert worker_mypyc["backend"] == "mypyc"
    assert pairer_mypyc["backend"] == "mypyc"
    assert payload_mypyc["backend"] == "mypyc"
    assert {binding["source"] for binding in worker_mypyc["bindings"]} == {
        "typed_region_project.worker::Worker.adjust",
        "typed_region_project.worker::Worker.parse",
        "typed_region_project.worker::Worker.scale",
        "typed_region_project.worker::Worker.score",
        "typed_region_project.worker::Worker.values",
    }
    assert {binding["kind"] for binding in worker_mypyc["bindings"]} == {
        "instance_method",
        "staticmethod",
        "classmethod",
    }
    assert [binding["source"] for binding in cython_region["bindings"]] == [
        "typed_region_project.worker::Worker.exchange"
    ]
    assert (
        cython_region["bindings"][0]["execution_kind"],
        worker_mypyc["variant_id"].endswith("@mypyc"),
        cython_region["variant_id"].endswith("@cython"),
    ) == ("async_generator", True, True)
    pairer_binding = pairer_mypyc["bindings"][0]
    assert pairer_binding["source"] == "typed_region_project.worker::Pairer.pair"
    assert pairer_binding["owner_class"] == "Pairer"
    assert pairer_binding["target_owner_class"] == "IntPairer"
    assert pairer_binding["guards"] == [
        {
            "parameter_name": "value",
            "positional_index": 1,
            "annotation": "int",
            "nominal_type_paths": ["int"],
            "allow_none": False,
        }
    ]
    payload_binding = payload_mypyc["bindings"][0]
    assert payload_binding["guards"] == [
        {
            "parameter_name": "value",
            "positional_index": 1,
            "annotation": "Payload | None",
            "nominal_type_paths": ["Payload"],
            "allow_none": True,
        }
    ]
    assert all(region["artifacts"] for region in compiled_regions)
    assert all(
        path.startswith(".atoll/artifacts/")
        for region in compiled_regions
        for path in region["artifacts"]
    )


def _assert_compiled_worker_module(worker_module: ModuleType) -> None:
    worker = worker_module.Worker(3)
    assert worker.scale(5) == SCALE_RESULT
    assert worker_module.Worker.parse("7") == PARSED_RESULT
    assert worker_module.Worker.adjust(4) == ADJUSTED_RESULT
    assert list(worker.values(3)) == [3, 3, 5]
    assert asyncio.run(worker.score(5)) == SCALE_RESULT
    asyncio.run(_assert_async_generator_protocol(worker))
    assert worker_module.Pairer[str]().pair("base") == ("base", "base")
    assert worker_module.IntPairer().pair(4) == (4, 4)
    assert worker_module.IntPairer().pair(value=4) == (4, 4)
    assert worker_module.IntPairer().pair("fallback") == ("fallback", "fallback")
    assert worker_module.IntPairer().pair(value="fallback") == ("fallback", "fallback")
    payload = worker_module.Payload(3)
    assert worker_module.PayloadPairer().maybe_pair(payload) == (payload, payload)
    assert worker_module.PayloadPairer().maybe_pair(None) == (None, None)
    assert worker_module.PayloadPairer().maybe_pair(value=None) == (None, None)
    assert worker_module.PayloadPairer().maybe_pair(3) == (3, 3)
    assert inspect.isgeneratorfunction(worker_module.Worker.values)
    assert inspect.iscoroutinefunction(worker_module.Worker.score)
    assert inspect.isasyncgenfunction(worker_module.Worker.exchange)
    assert worker_module.Worker.scale.__qualname__ == "Worker.scale"
    assert worker_module.Worker.scale.__module__ == "typed_region_project.worker"
    assert worker_module.__atoll_status__["compiled"] is True
    assert hasattr(worker_module.Worker.scale, "__atoll_compiled_target__")
    assert hasattr(worker_module.Worker.exchange, "__atoll_compiled_target__")
    assert hasattr(worker_module.IntPairer.pair, "__atoll_compiled_target__")
    assert not hasattr(worker_module.Pairer.pair, "__atoll_compiled_target__")
    assert hasattr(worker_module.PayloadPairer.maybe_pair, "__atoll_compiled_target__")
    assert not hasattr(worker_module.OptionalPairer.maybe_pair, "__atoll_compiled_target__")
    assert worker_module.IntPairer.pair.__atoll_runtime_guards__[0]["types"] == (int,)
    assert not hasattr(worker_module.DynamicWorker.calculate, "__atoll_compiled_target__")
    assert worker_module.DynamicWorker().calculate(4) == DYNAMIC_RESULT


def _source_contents(root: Path) -> dict[Path, str]:
    return {
        path.relative_to(root): path.read_text(encoding="utf-8")
        for path in sorted((root / "src").rglob("*.py"))
    }


def _clear_fixture_modules() -> None:
    for name in tuple(sys.modules):
        if name == "typed_region_project" or name.startswith(
            ("typed_region_project.", "_atoll_region_typed_region_project_")
        ):
            sys.modules.pop(name, None)


async def _assert_async_generator_protocol(worker: _ExchangeWorker) -> None:
    generator = worker.exchange(1)

    assert await anext(generator) == ASYNC_FIRST_RESULT
    assert await generator.asend(5) == ASYNC_SENT_RESULT
    assert await generator.athrow(ValueError("fixture")) == ASYNC_THROWN_RESULT
    await generator.aclose()
    assert worker.closed is True

    worker.closed = False
    failing_generator = worker.exchange(2)
    assert await anext(failing_generator) == ASYNC_FIRST_RESULT + 1
    with pytest.raises(RuntimeError, match="unhandled"):
        await failing_generator.athrow(RuntimeError("unhandled"))
    assert worker.closed is True
    await failing_generator.aclose()
    with pytest.raises(StopAsyncIteration):
        await anext(failing_generator)
