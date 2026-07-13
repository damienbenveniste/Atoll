"""Source-clean package acceptance for guarded scalar Cython variants."""

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
SCALAR_MEMBERS = (
    SymbolId("native_optimization_fixture.kernels", "scalar_polynomial"),
    SymbolId("native_optimization_fixture.kernels", "ScalarArithmetic.weighted_sum"),
)
EXPECTED_VARIANT_COUNT = 6


class _RuntimeEvidence(TypedDict):
    safe: int
    boolean: int
    huge: int
    ranks: list[int]
    signature: str
    defaults: list[int]
    kwdefaults: dict[str, int]
    static_safe: int
    static_boolean: int
    static_huge: int
    static_ranks: list[int]
    static_descriptor: bool


def test_scalar_variants_compose_in_source_clean_wheel_and_cache(tmp_path: Path) -> None:
    """Package compilation installs ordered widths and restores them on a warm run."""
    project_root = tmp_path / "project"
    output_dir = project_root / ".atoll" / "dist"
    shutil.copytree(FIXTURE_ROOT, project_root)
    source_path = project_root / "src" / "native_optimization_fixture" / "kernels.py"
    source_digest = _digest(source_path)
    options = PackageOptions(
        root=project_root,
        output_dir=output_dir,
        keep_install_tree=True,
        selected_members=SCALAR_MEMBERS,
        run_quality_gates=False,
    )

    cold = execute_package(options)
    warm = execute_package(options)

    assert cold.success is True, cold.error
    assert warm.success is True, warm.error
    assert len(cold.compiled_variants) == EXPECTED_VARIANT_COUNT
    assert [variant.backend for variant in cold.compiled_variants] == [
        "cython",
        "cython",
        "mypyc",
        "cython",
        "cython",
        "mypyc",
    ]
    assert [variant.cache_status for variant in cold.compiled_variants] == ["miss"] * (
        EXPECTED_VARIANT_COUNT
    )
    assert [variant.cache_status for variant in warm.compiled_variants] == ["hit"] * (
        EXPECTED_VARIANT_COUNT
    )
    assert {binding.source for binding in warm.compiled_bindings} == set(SCALAR_MEMBERS)
    assert warm.wheel_path is not None
    assert warm.wheel_path.exists()
    assert _digest(source_path) == source_digest
    assert not (output_dir / "build").exists()
    assert not tuple(project_root.rglob("*.pyx"))

    evidence = _runtime_evidence(warm.install_root)
    assert evidence == {
        "safe": 305635,
        "boolean": 20,
        "huge": 3,
        "ranks": [10, 20, 200],
        "signature": "(limit: 'int', rounds: 'int' = 1, *, bias: 'int' = 3) -> 'int'",
        "defaults": [1],
        "kwdefaults": {"bias": 3},
        "static_safe": 145,
        "static_boolean": 1,
        "static_huge": 0,
        "static_ranks": [10, 20, 200],
        "static_descriptor": True,
    }


def _runtime_evidence(install_root: Path) -> _RuntimeEvidence:
    script = """
import inspect
import json
from native_optimization_fixture import ScalarArithmetic, scalar_polynomial

huge = 10**40
print(json.dumps({
    "safe": scalar_polynomial(96),
    "boolean": scalar_polynomial(True),
    "huge": scalar_polynomial(0, rounds=huge),
    "ranks": [item["dispatch_rank"] for item in scalar_polynomial.__atoll_binding_variants__],
    "signature": str(inspect.signature(scalar_polynomial)),
    "defaults": list(scalar_polynomial.__defaults__ or ()),
    "kwdefaults": dict(scalar_polynomial.__kwdefaults__ or {}),
    "static_safe": ScalarArithmetic.weighted_sum(10, factor=3),
    "static_boolean": ScalarArithmetic.weighted_sum(True),
    "static_huge": ScalarArithmetic.weighted_sum(0, factor=huge),
    "static_ranks": [
        item["dispatch_rank"]
        for item in ScalarArithmetic.weighted_sum.__atoll_binding_variants__
    ],
    "static_descriptor": isinstance(vars(ScalarArithmetic)["weighted_sum"], staticmethod),
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
