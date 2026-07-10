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
from typing import Protocol

import pytest

from atoll.cli import main

FIXTURE_ROOT = Path("tests/fixtures/typed_region_project")
COMPILED_METHOD_COUNT = 6
COMPILED_REGION_COUNT = 2
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
    report = json.loads((project_root / ".atoll" / "compile-report.json").read_text())
    assert exit_code == 0
    assert _source_contents(project_root) == original_hashes
    assert report["summary"]["symbols"] == COMPILED_METHOD_COUNT
    assert report["summary"]["islands"] == 0
    assert report["summary"]["compiled_regions"] == COMPILED_REGION_COUNT
    assert report["summary"]["native_rejected_symbols"] == 1
    assert report["build"]["command"] == ["atoll", "typed-region-build"]
    compiled_by_backend = {region["backend"]: region for region in report["compiled_regions"]}
    assert set(compiled_by_backend) == {"mypyc", "cython"}
    assert {binding["source"] for binding in compiled_by_backend["mypyc"]["bindings"]} == {
        "typed_region_project.worker::Worker.adjust",
        "typed_region_project.worker::Worker.parse",
        "typed_region_project.worker::Worker.scale",
        "typed_region_project.worker::Worker.score",
        "typed_region_project.worker::Worker.values",
    }
    assert {binding["kind"] for binding in compiled_by_backend["mypyc"]["bindings"]} == {
        "instance_method",
        "staticmethod",
        "classmethod",
    }
    assert [binding["source"] for binding in compiled_by_backend["cython"]["bindings"]] == [
        "typed_region_project.worker::Worker.exchange"
    ]
    assert (
        compiled_by_backend["cython"]["bindings"][0]["execution_kind"],
        compiled_by_backend["mypyc"]["variant_id"].endswith("@mypyc"),
        compiled_by_backend["cython"]["variant_id"].endswith("@cython"),
    ) == ("async_generator", True, True)
    assert all(region["artifacts"] for region in compiled_by_backend.values())
    assert all(
        path.startswith(".atoll/artifacts/")
        for region in compiled_by_backend.values()
        for path in region["artifacts"]
    )
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
    worker = worker_module.Worker(3)

    assert worker.scale(5) == SCALE_RESULT
    assert worker_module.Worker.parse("7") == PARSED_RESULT
    assert worker_module.Worker.adjust(4) == ADJUSTED_RESULT
    assert list(worker.values(3)) == [3, 3, 5]
    assert asyncio.run(worker.score(5)) == SCALE_RESULT
    asyncio.run(_assert_async_generator_protocol(worker))
    assert inspect.isgeneratorfunction(worker_module.Worker.values)
    assert inspect.iscoroutinefunction(worker_module.Worker.score)
    assert inspect.isasyncgenfunction(worker_module.Worker.exchange)
    assert worker_module.Worker.scale.__qualname__ == "Worker.scale"
    assert worker_module.Worker.scale.__module__ == "typed_region_project.worker"
    assert worker_module.__atoll_status__["compiled"] is True
    assert hasattr(worker_module.Worker.scale, "__atoll_compiled_target__")
    assert hasattr(worker_module.Worker.exchange, "__atoll_compiled_target__")
    assert not hasattr(worker_module.DynamicWorker.calculate, "__atoll_compiled_target__")
    assert worker_module.DynamicWorker().calculate(4) == DYNAMIC_RESULT

    monkeypatch.setenv("ATOLL_DISABLE", "1")
    _clear_fixture_modules()
    disabled_module = importlib.import_module("typed_region_project.worker")
    assert disabled_module.Worker(3).scale(5) == SCALE_RESULT
    assert disabled_module.__atoll_status__["compiled"] is False
    assert not hasattr(disabled_module.Worker.scale, "__atoll_compiled_target__")
    assert not hasattr(disabled_module.Worker.exchange, "__atoll_compiled_target__")


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
