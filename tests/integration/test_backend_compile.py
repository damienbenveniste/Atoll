"""Native capability smoke tests for compiler backend adapters."""

from __future__ import annotations

from pathlib import Path

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.clustering import enrich_island_analysis
from atoll.backends.mypyc import MypycBackend
from atoll.models import BackendCompileContext, BackendLoweringRequest, ModuleId


def test_mypyc_compiles_native_class_generator_and_coroutine_region(tmp_path: Path) -> None:
    """The adapter produces a real extension for every capability it advertises."""
    source_path = tmp_path / "shapes.py"
    source_path.write_text(
        """from __future__ import annotations

from collections.abc import Iterator


class Worker:
    def __init__(self, bias: int) -> None:
        self.bias = bias

    def scale(self, value: int) -> int:
        return value * self.bias

    @staticmethod
    def parse(value: str) -> int:
        return int(value)

    @classmethod
    def create(cls, bias: int) -> Worker:
        return cls(bias)

    def values(self, limit: int) -> Iterator[int]:
        for value in range(limit):
            yield self.scale(value)

    async def score(self, value: int) -> int:
        return self.scale(value)
""",
        encoding="utf-8",
    )
    scan = enrich_island_analysis(scan_module(ModuleId(name="shapes", path=source_path)))
    region = next(region for region in scan.typed_regions if region.atomic_class)
    backend = MypycBackend()
    assessment = backend.assess(region)
    unit = backend.lower(
        BackendLoweringRequest(
            region=region,
            source_path=source_path,
            logical_module="shapes",
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
    assert set(assessment.capabilities) >= {
        "native_class",
        "instance_method",
        "staticmethod",
        "classmethod",
        "generator",
        "coroutine",
    }
    assert result.attempt.success is True, result.attempt.stderr
    assert result.attempt.artifact_paths
    assert any(record.role == "primary" for record in result.artifacts)
    assert all(record.install_relative_path.startswith("compiled/") for record in result.artifacts)
