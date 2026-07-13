"""Tests for the generic native-optimization acceptance fixture."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import sys
import time
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, cast

import pytest

FIXTURE_ROOT = Path("tests/fixtures/native_optimization_project")
SOURCE_ROOT = FIXTURE_ROOT / "src"
KERNEL_SOURCE = SOURCE_ROOT / "native_optimization_fixture" / "kernels.py"
BENCHMARK_SCRIPT = FIXTURE_ROOT / "benchmarks" / "run_native_workload.py"
SCALAR_BENCHMARK_SCRIPT = FIXTURE_ROOT / "benchmarks" / "run_scalar_hard.py"
BENCHMARK_ITERATIONS = 6000
EXPECTED_BRANCH_CHECKSUM = 173
EXPECTED_EVEN_MIXED = 35
EXPECTED_NEGATIVE_MIXED = -7
EXPECTED_ODD_MIXED = 128
EXPECTED_SCALAR_POLYNOMIAL = 725
EXPECTED_SCALAR_BENCHMARK_CHECKSUM = 314234
EXPECTED_WEIGHTED_SUM = 145
EXPECTED_POLYNOMIAL_CHECKSUM = 19944
MINIMUM_NATIVE_SPEEDUP = 1.25
MINIMUM_BENCHMARK_SECONDS = 0.25


class BranchArithmeticType(Protocol):
    """Static branch-arithmetic class surface exported by the fixture."""

    def accumulate(self, values: tuple[int, ...], *, pivot: int = ...) -> int:
        """Return a deterministic branch checksum."""
        ...

    def mixed(self, value: int, *, scale: int = ..., bias: int = ...) -> int:
        """Return branch-specific arithmetic."""
        ...


class FallbackProbeType(Protocol):
    """Static fallback-probe class surface exported by the fixture."""

    def square_route(self, value: int) -> tuple[str, int]:
        """Return the execution route and exact square."""
        ...


class ScalarArithmeticType(Protocol):
    """Static scalar class surface exported by the fixture."""

    def weighted_sum(self, limit: int, factor: int = ...) -> int:
        """Return the weighted scalar reduction."""
        ...


class BenchmarkModule(Protocol):
    """Loaded benchmark module interface used by these tests."""

    def main(self) -> int:
        """Run the benchmark entry point."""
        ...


class FixtureModule(Protocol):
    """Loaded fixture module interface used by these tests."""

    BranchArithmetic: BranchArithmeticType
    FALLBACK_LIMIT: int
    FallbackProbe: FallbackProbeType
    ScalarArithmetic: ScalarArithmeticType

    def scalar_polynomial(self, limit: int, rounds: int = ..., *, bias: int = ...) -> int:
        """Return the fixed-width-friendly polynomial reduction."""
        ...

    def polynomial_checksum(self, width: int = ..., repetitions: int = ...) -> int:
        """Return a deterministic polynomial checksum."""
        ...

    def branch_checksum(self, values: tuple[int, ...], pivot: int = ...) -> int:
        """Return a deterministic branch checksum."""
        ...

    def keyword_polynomial_window(
        self,
        start: int,
        stop: int = ...,
        *,
        scale: int = ...,
        bias: int = ...,
    ) -> tuple[int, ...]:
        """Return deterministic polynomial window values."""
        ...


def test_fixture_project_has_pep517_and_configured_baseline_command() -> None:
    config = tomllib.loads((FIXTURE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    project = cast(Mapping[str, object], config["project"])
    build_system = cast(Mapping[str, object], config["build-system"])
    compile_config = cast(
        Mapping[str, object],
        cast(Mapping[str, object], cast(Mapping[str, object], config["tool"])["atoll"])["compile"],
    )

    assert project["name"] == "native-optimization-project"
    assert build_system["build-backend"] == "setuptools.build_meta"
    assert compile_config["test_command"] == ["python", "-m", "pytest", "tests", "-q"]
    assert compile_config["benchmark_command"] == [
        "python",
        "benchmarks/run_native_workload.py",
        "--iterations",
        str(BENCHMARK_ITERATIONS),
    ]
    assert compile_config["minimum_speedup"] == MINIMUM_NATIVE_SPEEDUP


def test_fixture_semantics_are_canonical_and_stable() -> None:
    module = _fixture_module()

    assert module.scalar_polynomial(8, rounds=2, bias=5) == EXPECTED_SCALAR_POLYNOMIAL
    assert module.ScalarArithmetic.weighted_sum(10, factor=3) == EXPECTED_WEIGHTED_SUM
    assert module.polynomial_checksum(width=8, repetitions=3) == EXPECTED_POLYNOMIAL_CHECKSUM
    assert module.polynomial_checksum(width=8, repetitions=3) == EXPECTED_POLYNOMIAL_CHECKSUM
    assert module.branch_checksum((-3, -1, 0, 2, 7, 9, 12)) == EXPECTED_BRANCH_CHECKSUM
    assert module.keyword_polynomial_window(2, scale=4, bias=1) == (
        13,
        22,
        33,
        46,
        61,
        78,
        97,
    )


def test_static_methods_cover_branch_and_fallback_semantics() -> None:
    module = _fixture_module()
    branch_arithmetic = module.BranchArithmetic
    fallback_probe = module.FallbackProbe

    class StrictInt(int):
        """Integer subclass used to assert exact-type fallback routing."""

    huge = module.FALLBACK_LIMIT + 10

    assert (
        branch_arithmetic.accumulate(
            (-3, -1, 0, 2, 7, 9, 12),
            pivot=7,
        )
        == EXPECTED_BRANCH_CHECKSUM
    )
    assert branch_arithmetic.mixed(-4, scale=5, bias=9) == EXPECTED_NEGATIVE_MIXED
    assert branch_arithmetic.mixed(8, scale=4, bias=3) == EXPECTED_EVEN_MIXED
    assert branch_arithmetic.mixed(5, scale=4, bias=3) == EXPECTED_ODD_MIXED
    assert fallback_probe.square_route(17) == ("native", 289)
    assert fallback_probe.square_route(True) == ("python", 1)
    assert fallback_probe.square_route(StrictInt(19)) == ("python", 361)
    assert fallback_probe.square_route(huge) == ("python", huge * huge)


def test_source_shape_contains_typed_functions_and_static_methods() -> None:
    tree = ast.parse(KERNEL_SOURCE.read_text(encoding="utf-8"))
    functions = _functions(tree)
    classes = _classes(tree)

    required_functions = {
        "scalar_polynomial",
        "polynomial_checksum",
        "branch_checksum",
        "keyword_polynomial_window",
    }
    assert required_functions <= functions.keys()
    assert _has_for_loop(functions["scalar_polynomial"])
    assert _has_for_loop(functions["polynomial_checksum"])
    assert _has_keyword_only_defaults(functions["keyword_polynomial_window"], {"scale", "bias"})
    assert _annotated_returns(functions["polynomial_checksum"], "int")
    assert _annotated_returns(functions["keyword_polynomial_window"], "tuple")

    branch_methods = _static_methods(classes["BranchArithmetic"])
    fallback_methods = _static_methods(classes["FallbackProbe"])
    scalar_methods = _static_methods(classes["ScalarArithmetic"])
    assert {"accumulate", "mixed"} <= branch_methods
    assert fallback_methods == {"square_route"}
    assert scalar_methods == {"weighted_sum"}
    assert any(isinstance(node, ast.If) for node in ast.walk(classes["BranchArithmetic"]))


def test_benchmark_entry_point_prints_stable_json_checksum(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark = _benchmark_module()
    monkeypatch.chdir(FIXTURE_ROOT)
    monkeypatch.setattr(sys, "argv", ["run_native_workload.py", "--iterations", "5"])

    assert benchmark.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "checksum": 4142789730,
        "iterations": 5,
        "logical_items": 50,
    }


def test_scalar_hard_benchmark_prints_stable_json_checksum(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark = _benchmark_module(SCALAR_BENCHMARK_SCRIPT)
    monkeypatch.chdir(FIXTURE_ROOT)
    monkeypatch.setattr(sys, "argv", ["run_scalar_hard.py", "--calls", "5"])

    assert benchmark.main() == 0
    assert json.loads(capsys.readouterr().out) == {
        "calls": 5,
        "checksum": EXPECTED_SCALAR_BENCHMARK_CHECKSUM,
    }


def test_configured_baseline_benchmark_command_runs_long_enough(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark = _benchmark_module()
    monkeypatch.chdir(FIXTURE_ROOT)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_workload.py", "--iterations", str(BENCHMARK_ITERATIONS)],
    )

    started = time.perf_counter()
    assert benchmark.main() == 0
    elapsed = time.perf_counter() - started

    payload = json.loads(capsys.readouterr().out)
    assert payload["iterations"] == BENCHMARK_ITERATIONS
    assert payload["logical_items"] == BENCHMARK_ITERATIONS * 10
    assert isinstance(payload["checksum"], int)
    assert elapsed > MINIMUM_BENCHMARK_SECONDS


def _functions(tree: ast.AST) -> dict[str, ast.FunctionDef]:
    return {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}


def _classes(tree: ast.AST) -> dict[str, ast.ClassDef]:
    return {node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}


def _fixture_module() -> FixtureModule:
    source_root = str(SOURCE_ROOT.resolve())
    sys.path.insert(0, source_root)
    try:
        return cast(FixtureModule, importlib.import_module("native_optimization_fixture"))
    finally:
        sys.path.remove(source_root)


def _benchmark_module(path: Path = BENCHMARK_SCRIPT) -> BenchmarkModule:
    spec = importlib.util.spec_from_file_location("native_optimization_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("benchmark module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(BenchmarkModule, module)


def _static_methods(node: ast.ClassDef) -> set[str]:
    methods: set[str] = set()
    for child in node.body:
        if not isinstance(child, ast.FunctionDef):
            continue
        if any(
            isinstance(decorator, ast.Name) and decorator.id == "staticmethod"
            for decorator in child.decorator_list
        ):
            methods.add(child.name)
    return methods


def _has_for_loop(node: ast.FunctionDef) -> bool:
    return any(isinstance(child, ast.For) for child in ast.walk(node))


def _has_keyword_only_defaults(node: ast.FunctionDef, names: set[str]) -> bool:
    keyword_names = {argument.arg for argument in node.args.kwonlyargs}
    return names <= keyword_names and all(default is not None for default in node.args.kw_defaults)


def _annotated_returns(node: ast.FunctionDef, expected: str) -> bool:
    annotation = node.returns
    if isinstance(annotation, ast.Name):
        return annotation.id == expected
    if isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name):
        return annotation.value.id == expected
    return False
