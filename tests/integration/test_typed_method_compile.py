"""Native vertical-slice tests for typed method lowering and binding."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import sys
from pathlib import Path

import pytest

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.clustering import enrich_island_analysis
from atoll.backends.mypyc import MypycBackend
from atoll.generation.region_shim import RegionShimConfig, insert_or_replace_region_shim
from atoll.generation.typed_region import generate_typed_method_region
from atoll.models import BackendCompileContext, BackendLoweringRequest, ModuleId

SCALE_RESULT = 23
PARSED_RESULT = 7
ADJUSTED_RESULT = 6


def test_mypyc_methods_bind_to_the_original_python_class(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Structural owner typing accepts source instances without class replacement."""
    source_path = tmp_path / "worker.py"
    source_path.write_text(_WORKER_SOURCE, encoding="utf-8")
    scan = enrich_island_analysis(scan_module(ModuleId(name="worker", path=source_path)))
    region = next(region for region in scan.typed_regions if region.atomic_class)
    selected = tuple(
        member.id
        for member in region.members
        if member.id.qualname
        in {
            "Worker.scale",
            "Worker.parse",
            "Worker.adjust",
            "Worker.values",
            "Worker.score",
        }
    )
    logical_module = f"_atoll_worker_{region.source_hash[:10]}"
    generated_path = tmp_path / f"{logical_module}.py"
    generation = generate_typed_method_region(
        scan,
        region,
        selected,
        output_path=generated_path,
    )
    backend = MypycBackend()
    unit = backend.lower(
        BackendLoweringRequest(
            region=region,
            source_path=generated_path,
            logical_module=logical_module,
            install_relative_dir=".atoll/artifacts",
            members=selected,
        )
    )
    artifact_dir = tmp_path / ".atoll" / "artifacts"

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
        source_module="worker",
        source_path=source_path,
        region_id=region.id,
        backend="mypyc",
        compiled_module=logical_module,
        artifact_dir=artifact_dir,
        bindings=generation.bindings,
    )
    source_path.write_text(
        insert_or_replace_region_shim(_WORKER_SOURCE, (config,)).new_text,
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])
    sys.modules.pop("worker", None)
    sys.modules.pop(logical_module, None)

    worker_module = importlib.import_module("worker")
    worker = worker_module.Worker(3)

    assert worker.scale(5) == SCALE_RESULT
    assert worker_module.Worker.parse("7") == PARSED_RESULT
    assert worker_module.Worker.adjust(4) == ADJUSTED_RESULT
    assert list(worker.values(3)) == [3, 3, 5]
    assert asyncio.run(worker.score(5)) == SCALE_RESULT
    assert inspect.isgeneratorfunction(worker_module.Worker.values)
    assert inspect.iscoroutinefunction(worker_module.Worker.score)
    assert inspect.signature(worker_module.Worker.scale) == inspect.Signature(
        parameters=(
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter(
                "value",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation="int",
            ),
        ),
        return_annotation="int",
    )
    assert worker_module.Worker.scale.__module__ == "worker"
    assert worker_module.Worker.scale.__qualname__ == "Worker.scale"
    assert worker_module.Worker.scale.__doc__ == "Scale one integer workload."
    assert hasattr(
        worker_module.Worker.scale,
        "__atoll_compiled_target__",
    ), worker_module.__atoll_status__
    assert worker_module.__atoll_status__["compiled"] is True
    binding_statuses = next(iter(worker_module.__atoll_region_status__.values()))["bindings"]
    assert all(status["compiled"] for status in binding_statuses.values())
    assert not any(name.startswith("_atoll_") for name in vars(worker_module))


_WORKER_SOURCE = """from __future__ import annotations

from collections.abc import Iterator


class Worker:
    bias: int

    def __init__(self, bias: int) -> None:
        self.bias = bias

    def scale(self, value: int) -> int:
        \"\"\"Scale one integer workload.\"\"\"
        total = 0
        for item in range(value):
            total += item * 2
        return total + self.bias

    @staticmethod
    def parse(value: str) -> int:
        return int(value)

    @classmethod
    def adjust(cls, value: int) -> int:
        _ = cls
        return value + 2

    def values(self, limit: int) -> Iterator[int]:
        for value in range(limit):
            yield self.scale(value)

    async def score(self, value: int) -> int:
        return self.scale(value)
"""
