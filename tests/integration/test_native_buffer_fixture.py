"""Source-clean package acceptance for exact buffer native variants."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Literal, TypedDict, cast

from atoll.commands.package import PackageOptions, execute_package
from atoll.models import SymbolId
from atoll.runtime.performance import run_performance_command

FIXTURE_ROOT = Path("tests/fixtures/native_optimization_project")
BUFFER_MEMBERS = (
    SymbolId("native_optimization_fixture.kernels", "bytes_checksum"),
    SymbolId("native_optimization_fixture.kernels", "buffer_name_collision"),
    SymbolId("native_optimization_fixture.kernels", "bytearray_checksum"),
    SymbolId("native_optimization_fixture.kernels", "memoryview_checksum"),
    SymbolId("native_optimization_fixture.kernels", "array_checksum"),
)
EXPECTED_BINDINGS = {
    "array_checksum",
    "buffer_name_collision",
    "bytearray_checksum",
    "bytes_checksum",
    "memoryview_checksum",
}
BENCHMARK_BINDINGS = EXPECTED_BINDINGS - {"buffer_name_collision"}


class _RuntimeEvidence(TypedDict):
    array_empty: int
    array_values: int
    bytearray_empty: int
    bytearray_values: int
    bytes_empty: int
    bytes_values: int
    collision_values: int
    contiguous_view: int
    fallback_candidate: str
    mutable_view: int
    mutation_bytes: list[int]
    mutation_probe: list[object]
    noncontiguous_view: int
    readonly_probe: list[object]
    readonly_view: int
    unsupported_exporter: str
    variants: dict[str, list[int]]


class _BenchmarkEvidence(TypedDict):
    active_bindings: list[str]
    calls: int
    checksums: list[int]
    measurements: int
    width: int


class _RoutingEvidence(TypedDict):
    """Observed dispatcher choices after replacing only rank-30 targets."""

    direct_targets: dict[str, int]
    supported: dict[str, int]
    unsupported: dict[str, int]


def test_exact_buffer_variants_compile_and_preserve_python_semantics(tmp_path: Path) -> None:
    """Exact buffer kernels bind natively while fallback probes retain Python behavior."""

    project_root = tmp_path / "project"
    output_dir = project_root / ".atoll" / "dist"
    shutil.copytree(
        FIXTURE_ROOT,
        project_root,
        ignore=shutil.ignore_patterns(
            ".pytest_cache",
            "__pycache__",
            ".mypy_cache",
            ".ruff_cache",
            "*.pyc",
        ),
    )
    source_path = project_root / "src" / "native_optimization_fixture" / "kernels.py"
    source_digest = _digest(source_path)
    options = PackageOptions(
        root=project_root,
        output_dir=output_dir,
        keep_install_tree=True,
        selected_members=BUFFER_MEMBERS,
        run_quality_gates=False,
    )

    cold = execute_package(options)
    warm = execute_package(options)

    assert cold.success is True, cold.error
    assert warm.success is True, warm.error
    assert len(cold.compiled_variants) == 2 * len(BUFFER_MEMBERS)
    cold_buffers = tuple(
        variant for variant in cold.compiled_variants if "@cython-buffer-" in variant.id
    )
    warm_buffers = tuple(
        variant for variant in warm.compiled_variants if "@cython-buffer-" in variant.id
    )
    assert len(cold_buffers) == len(BUFFER_MEMBERS)
    assert [variant.cache_status for variant in cold_buffers] == ["miss"] * len(BUFFER_MEMBERS)
    assert [variant.cache_status for variant in warm_buffers] == ["hit"] * len(BUFFER_MEMBERS)
    assert {binding.source for binding in warm.compiled_bindings} == set(BUFFER_MEMBERS)
    assert {binding.source.qualname for binding in warm.compiled_bindings} == EXPECTED_BINDINGS
    assert warm.wheel_path is not None
    assert warm.wheel_path.exists()
    assert _digest(source_path) == source_digest
    assert not (output_dir / "build").exists()
    assert not tuple(project_root.rglob("*.pyx"))

    assert _runtime_evidence(warm.install_root) == {
        "array_empty": 0,
        "array_values": 65811,
        "bytearray_empty": 0,
        "bytearray_values": 6,
        "bytes_empty": 0,
        "bytes_values": 134,
        "collision_values": 134,
        "contiguous_view": 6,
        "fallback_candidate": "TypeError",
        "mutable_view": 6,
        "mutation_bytes": [90, 98, 99],
        "mutation_probe": ["mutated", 22, 694],
        "noncontiguous_view": 42,
        "readonly_probe": ["readonly", 22, 22],
        "readonly_view": 6,
        "unsupported_exporter": "TypeError",
        "variants": {
            "array_checksum": [30, 200],
            "bytearray_checksum": [30, 210],
            "bytes_checksum": [30, 200],
            "buffer_name_collision": [30, 200],
            "memoryview_checksum": [30, 200],
        },
    }
    assert _routing_evidence(warm.install_root) == {
        "direct_targets": {
            "array_checksum": 3,
            "bytearray_checksum": 3,
            "bytes_checksum": 3,
            "buffer_name_collision": 3,
            "memoryview_checksum": 3,
        },
        "supported": {
            "array_checksum": 1_000_003,
            "bytearray_checksum": 1_000_003,
            "bytes_checksum": 1_000_003,
            "buffer_name_collision": 1_000_003,
            "memoryview_checksum": 1_000_003,
        },
        "unsupported": {
            "array_non_byte_format": 3,
            "memoryview_strided": 2,
        },
    }
    baseline_benchmark = _benchmark_evidence(project_root, warm.install_root, mode="baseline")
    compiled_benchmark = _benchmark_evidence(project_root, warm.install_root, mode="compiled")
    assert baseline_benchmark == {
        "active_bindings": [],
        "calls": 12,
        "checksums": compiled_benchmark["checksums"],
        "measurements": 3,
        "width": 64,
    }
    assert compiled_benchmark == {
        "active_bindings": sorted(BENCHMARK_BINDINGS),
        "calls": 12,
        "checksums": [62438, 62438, 62438],
        "measurements": 3,
        "width": 64,
    }


def _runtime_evidence(install_root: Path) -> _RuntimeEvidence:
    script = """
import array
import json
from native_optimization_fixture import kernels

class UnsupportedExporter:
    pass

class FallbackCandidate:
    def __bytes__(self):
        return b"abc"

data = bytes([3, 8, 13, 21, 34, 55])
mutable = bytearray(data)
readonly_view = memoryview(data)
mutable_view = memoryview(mutable)
noncontiguous_view = mutable_view[1::2]
values = array.array("I", [1, 258, 65535, 17])
mutation_target = bytearray(b"abc")

def error_name(callable):
    try:
        callable()
    except Exception as error:
        return type(error).__name__
    return "none"

evidence = {
    "array_empty": kernels.array_checksum(array.array("I")),
    "array_values": kernels.array_checksum(values),
    "bytearray_empty": kernels.bytearray_checksum(bytearray()),
    "bytearray_values": kernels.bytearray_checksum(mutable),
    "bytes_empty": kernels.bytes_checksum(b""),
    "bytes_values": kernels.bytes_checksum(data),
    "collision_values": kernels.buffer_name_collision(data),
    "contiguous_view": kernels.memoryview_checksum(mutable_view),
    "fallback_candidate": error_name(
        lambda: kernels.buffer_weighted_checksum(FallbackCandidate())
    ),
    "mutable_view": kernels.memoryview_checksum(mutable_view),
    "mutation_probe": list(kernels.buffer_mutation_probe(mutation_target, 90)),
    "mutation_bytes": list(mutation_target),
    "noncontiguous_view": kernels.memoryview_checksum(noncontiguous_view),
    "readonly_probe": list(kernels.buffer_mutation_probe(b"abc", 90)),
    "readonly_view": kernels.memoryview_checksum(readonly_view),
    "unsupported_exporter": error_name(
        lambda: kernels.buffer_weighted_checksum(UnsupportedExporter())
    ),
    "variants": {
        name: [
            item["dispatch_rank"]
            for item in getattr(getattr(kernels, name), "__atoll_binding_variants__", ())
        ]
        for name in (
            "array_checksum",
            "bytearray_checksum",
            "bytes_checksum",
            "buffer_name_collision",
            "memoryview_checksum",
        )
    },
}
print(json.dumps(evidence, sort_keys=True))
"""
    completed = run_performance_command(
        (sys.executable, "-c", script),
        project_root=install_root,
        payload_root=install_root,
        mode="compiled",
    )
    assert completed.succeeded, completed.stderr
    return cast(_RuntimeEvidence, json.loads(completed.stdout))


def _benchmark_evidence(
    project_root: Path,
    install_root: Path,
    *,
    mode: Literal["baseline", "compiled"],
) -> _BenchmarkEvidence:
    completed = run_performance_command(
        (
            sys.executable,
            "benchmarks/run_buffer_hard.py",
            "--calls",
            "12",
            "--width",
            "64",
            "--measurements",
            "3",
        ),
        project_root=project_root,
        payload_root=install_root,
        mode=mode,
    )
    assert completed.succeeded, completed.stderr
    return cast(_BenchmarkEvidence, json.loads(completed.stdout))


def _routing_evidence(install_root: Path) -> _RoutingEvidence:
    script = """
import array
import json
from native_optimization_fixture import kernels

supported = {
    "array_checksum": array.array("B", [1, 2]),
    "bytearray_checksum": bytearray([1, 2]),
    "bytes_checksum": bytes([1, 2]),
    "buffer_name_collision": bytes([1, 2]),
    "memoryview_checksum": memoryview(bytearray([1, 2])),
}
direct_targets = {}
for name, value in supported.items():
    binding = getattr(kernels, name)
    variant = next(
        item
        for item in binding.__atoll_binding_variants__
        if item["dispatch_rank"] == 30
    )
    target = variant["target"]
    direct_targets[name] = target(value)
    variant["target"] = lambda argument, target=target: 1_000_000 + target(argument)

evidence = {
    "direct_targets": direct_targets,
    "supported": {name: getattr(kernels, name)(value) for name, value in supported.items()},
    "unsupported": {
        "array_non_byte_format": kernels.array_checksum(array.array("I", [1, 2])),
        "memoryview_strided": kernels.memoryview_checksum(memoryview(bytearray([1, 2, 3]))[::2]),
    },
}
print(json.dumps(evidence, sort_keys=True))
"""
    completed = run_performance_command(
        (sys.executable, "-c", script),
        project_root=install_root,
        payload_root=install_root,
        mode="compiled",
    )
    assert completed.succeeded, completed.stderr
    return cast(_RoutingEvidence, json.loads(completed.stdout))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
