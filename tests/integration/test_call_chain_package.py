"""Source-clean package acceptance for direct native call-chain variants."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import TypedDict, cast

from atoll.commands.package import PackageOptions, execute_package
from atoll.models import SymbolId
from atoll.runtime.performance import run_performance_command

FIXTURE_ROOT = Path("tests/fixtures/native_optimization_project")
CHAIN_ROOT = SymbolId("native_optimization_fixture.kernels", "direct_chain_root")
INSTANCE_CHAIN_ROOT = SymbolId("native_optimization_fixture.kernels", "ChainAccumulator.run")
EXPECTED_VARIANT_COUNT = 6


class _RuntimeEvidence(TypedDict):
    safe: int
    ranks: list[int]
    binding_count: int
    patched: int
    instance_safe: int
    instance_ranks: list[int]
    instance_boolean_field: int
    instance_huge_field: int
    instance_subclass: int
    instance_patched: int
    instance_shadowed: int


def test_call_chain_variants_compile_route_fallback_and_cache(tmp_path: Path) -> None:
    """Private inline helpers compose with generic fallback in an installed wheel."""
    project_root = tmp_path / "project"
    output_dir = project_root / ".atoll" / "dist"
    shutil.copytree(FIXTURE_ROOT, project_root)
    source_path = project_root / "src" / "native_optimization_fixture" / "kernels.py"
    source_digest = _digest(source_path)
    options = PackageOptions(
        root=project_root,
        output_dir=output_dir,
        keep_install_tree=True,
        selected_members=(CHAIN_ROOT, INSTANCE_CHAIN_ROOT),
        run_quality_gates=False,
    )

    cold = execute_package(options)
    warm = execute_package(options)

    assert cold.success is True, cold.error
    assert warm.success is True, warm.error
    assert len(cold.compiled_variants) == EXPECTED_VARIANT_COUNT
    assert [variant.cache_status for variant in cold.compiled_variants] == [
        "miss"
    ] * EXPECTED_VARIANT_COUNT
    assert [variant.cache_status for variant in warm.compiled_variants] == [
        "hit"
    ] * EXPECTED_VARIANT_COUNT
    assert {binding.source for binding in warm.compiled_bindings} == {
        CHAIN_ROOT,
        INSTANCE_CHAIN_ROOT,
    }
    assert warm.call_chain_analyses
    assert warm.wheel_path is not None
    assert warm.wheel_path.exists()
    assert _digest(source_path) == source_digest
    assert not (output_dir / "build").exists()
    assert not tuple(project_root.rglob("*.pyx"))
    assert _runtime_evidence(warm.install_root) == {
        "safe": 205,
        "ranks": [10, 20, 200],
        "binding_count": 3,
        "patched": 235,
        "instance_safe": 95,
        "instance_ranks": [10, 20, 200],
        "instance_boolean_field": 35,
        "instance_huge_field": 2 * 10**40 + 1,
        "instance_subclass": 229,
        "instance_patched": 145,
        "instance_shadowed": 999,
    }


def _runtime_evidence(install_root: Path) -> _RuntimeEvidence:
    script = """
import json
from native_optimization_fixture import kernels

safe = kernels.direct_chain_root(4, scale=3, bias=7)
variants = kernels.direct_chain_root.__atoll_binding_variants__
instance = kernels.ChainAccumulator(3)
instance_safe = instance.run(4)
instance_variants = kernels.ChainAccumulator.run.__atoll_binding_variants__

class SubAccumulator(kernels.ChainAccumulator):
    def step(self, value: int, bias: int = 1) -> int:
        return super().step(value, bias) + 100

instance_boolean_field = kernels.ChainAccumulator(True).run(4)
instance_huge_field = kernels.ChainAccumulator(10**40).run(2, rounds=1)
instance_subclass = SubAccumulator(3).run(4, rounds=2)
shadowed = kernels.ChainAccumulator(3)
shadowed.step = lambda value, bias=1: 999
instance_shadowed = shadowed.run(4, rounds=1)

def replacement(value: int, increment: int = 3, *, scale: int = 2) -> int:
    return (value + increment + 2) * scale

kernels.direct_chain_leaf = replacement
patched = kernels.direct_chain_root(4, scale=3, bias=7)

def replacement_step(self, value: int, bias: int = 1) -> int:
    return value * self.factor + bias + 10

kernels.ChainAccumulator.step = replacement_step
instance_patched = instance.run(4)
print(json.dumps({
    "safe": safe,
    "ranks": [item["dispatch_rank"] for item in variants],
    "binding_count": len(variants),
    "patched": patched,
    "instance_safe": instance_safe,
    "instance_ranks": [item["dispatch_rank"] for item in instance_variants],
    "instance_boolean_field": instance_boolean_field,
    "instance_huge_field": instance_huge_field,
    "instance_subclass": instance_subclass,
    "instance_patched": instance_patched,
    "instance_shadowed": instance_shadowed,
}, sort_keys=True))
"""
    completed = run_performance_command(
        (sys.executable, "-c", script),
        project_root=install_root,
        payload_root=install_root,
        mode="compiled",
    )
    assert completed.succeeded, completed.stderr
    return cast(_RuntimeEvidence, json.loads(completed.stdout))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
