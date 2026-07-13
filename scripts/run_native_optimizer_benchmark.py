"""Run generic hard benchmarks for Atoll native optimization families."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import tomllib
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import cast

from atoll.optimization_policy import (
    HARD_BENCHMARK_MINIMUM_SPEEDUP,
    MINIMUM_STABLE_MEDIAN_SECONDS,
    assess_speedup,
)
from atoll.report import COMPILE_REPORT_SCHEMA_VERSION

FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1] / "tests" / "fixtures" / ("native_optimization_project")
)
DEFAULT_WARMUPS = 1
DEFAULT_SAMPLES = 7
DEFAULT_TARGET_FAST_SECONDS = 0.35
MAX_CALIBRATION_PASSES = 6
_NATIVE_PHASES = frozenset({"cythonize", "mypycify", "build_ext"})


class BenchmarkError(RuntimeError):
    """Raised when hard-benchmark setup or evidence is incomplete."""


@dataclass(frozen=True, slots=True)
class FamilySpec:
    """One generic native family and the bindings it must exercise."""

    name: str
    script: str
    initial_calls: int
    extra_args: tuple[str, ...]
    module: str
    bindings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FamilyEvaluationInputs:
    """Raw paired samples and semantic routing evidence for one family."""

    name: str
    calls: int
    baseline_samples: tuple[float, ...]
    compiled_samples: tuple[float, ...]
    semantic_match: bool
    active_bindings: tuple[str, ...]
    expected_bindings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FamilyEvidence:
    """Normalized 3x gate evidence for one native optimization family."""

    name: str
    calls: int
    baseline_median_seconds: float
    baseline_samples_seconds: tuple[float, ...]
    compiled_median_seconds: float
    compiled_samples_seconds: tuple[float, ...]
    speedup: float | None
    semantic_match: bool
    active_bindings: tuple[str, ...]
    expected_bindings: tuple[str, ...]
    diagnostics: tuple[str, ...]
    passed: bool


@dataclass(frozen=True, slots=True)
class NativeBenchmarkEvidence:
    """Cold/warm compile and per-family promotion evidence."""

    compile_report_version: int
    cold_compile_exit_code: int
    cold_compiler_invocations: int
    warm_compile_exit_code: int
    warm_compiler_invocations: int
    warm_native_phase_count: int
    sources_unchanged: bool
    artifact_count: int
    native_variant_count: int
    families: tuple[FamilyEvidence, ...]
    errors: tuple[str, ...]
    passed: bool


@dataclass(frozen=True, slots=True)
class _ArmResult:
    elapsed_seconds: float
    payload: dict[str, object]


@dataclass(frozen=True, slots=True)
class _GlobalEvaluationInputs:
    """Compile, cache, source, and family evidence for the final verdict."""

    report: dict[str, object]
    cold_exit: int
    warm_exit: int
    cold_invocations: int
    warm_invocations: int
    sources_unchanged: bool
    families: tuple[FamilyEvidence, ...]


_FAMILIES = (
    FamilySpec(
        name="scalar",
        script="benchmarks/run_scalar_hard.py",
        initial_calls=150_000,
        extra_args=(),
        module="native_optimization_fixture.kernels",
        bindings=(
            "BranchArithmetic.mixed",
            "ScalarArithmetic.weighted_sum",
            "scalar_polynomial",
        ),
    ),
    FamilySpec(
        name="call-chain",
        script="benchmarks/run_call_chain_hard.py",
        initial_calls=130_000,
        extra_args=("--depth", "512"),
        module="native_optimization_fixture.kernels",
        bindings=("direct_chain_root",),
    ),
    FamilySpec(
        name="buffer",
        script="benchmarks/run_buffer_hard.py",
        initial_calls=60_000,
        extra_args=("--width", "2048"),
        module="native_optimization_fixture.kernels",
        bindings=(
            "array_checksum",
            "bytearray_checksum",
            "bytes_checksum",
            "memoryview_checksum",
        ),
    ),
)


def main(argv: tuple[str, ...] | None = None) -> int:
    """Compile the generic fixture cold/warm and enforce every family gate.

    Args:
        argv: Optional arguments replacing ``sys.argv``.

    Returns:
        int: Zero only when compilation, caching, semantics, and all 3x gates pass.
    """
    args = _parse_args(tuple(sys.argv[1:] if argv is None else argv))
    try:
        evidence = run_benchmark(args.workspace, args.evidence_root)
    except (BenchmarkError, OSError, subprocess.SubprocessError, ValueError) as error:
        print(f"native optimizer benchmark failed: {error}", file=sys.stderr)
        return 1
    payload = json.dumps(asdict(evidence), sort_keys=True, separators=(",", ":"))
    print(payload)
    (args.evidence_root / "summary.json").write_text(f"{payload}\n", encoding="utf-8")
    return 0 if evidence.passed else 1


def run_benchmark(workspace: Path, evidence_root: Path) -> NativeBenchmarkEvidence:
    """Create a source-clean fixture copy and measure its promoted native wheel.

    Args:
        workspace: Nonexistent disposable project path.
        evidence_root: Directory receiving logs, reports, and normalized evidence.

    Returns:
        NativeBenchmarkEvidence: Full hard-gate verdict.

    Raises:
        BenchmarkError: If paths, compiler tools, wheel output, or reports are unavailable.
    """
    if workspace.exists():
        raise BenchmarkError(f"workspace already exists: {workspace}")
    evidence_root.mkdir(parents=True, exist_ok=True)
    _copy_fixture(workspace)
    _remove_configured_benchmark(workspace / "pyproject.toml")
    before_sources = _source_manifest(workspace / "src")
    compile_environment, probe_log = _compiler_probe_environment(evidence_root)
    compile_command = (sys.executable, "-m", "atoll", "compile", "--root", str(workspace))

    cold_exit = _run_compile(
        compile_command,
        workspace,
        compile_environment,
        evidence_root / "cold.compile.log",
    )
    _copy_compile_reports(workspace, evidence_root, "cold")
    cold_invocations = _line_count(probe_log)
    warm_exit = _run_compile(
        compile_command,
        workspace,
        compile_environment,
        evidence_root / "warm.compile.log",
    )
    _copy_compile_reports(workspace, evidence_root, "warm")
    warm_invocations = _line_count(probe_log) - cold_invocations

    report = _read_json_object(evidence_root / "warm.compile-report.json")
    wheel = _single_wheel(workspace / ".atoll" / "dist")
    payload_root = evidence_root / "compiled-payload"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(payload_root)
    _run_semantic_tests(workspace, payload_root, evidence_root)
    families = tuple(_measure_family(spec, workspace, payload_root) for spec in _FAMILIES)
    after_sources = _source_manifest(workspace / "src")
    errors = _global_errors(
        _GlobalEvaluationInputs(
            report=report,
            cold_exit=cold_exit,
            warm_exit=warm_exit,
            cold_invocations=cold_invocations,
            warm_invocations=warm_invocations,
            sources_unchanged=before_sources == after_sources,
            families=families,
        )
    )
    composition = _mapping_field(report, "final_composition")
    artifacts = _list_field(composition, "artifacts")
    variants = _list_field(composition, "native_variant_ids")
    return NativeBenchmarkEvidence(
        compile_report_version=_integer_field(report, "version"),
        cold_compile_exit_code=cold_exit,
        cold_compiler_invocations=cold_invocations,
        warm_compile_exit_code=warm_exit,
        warm_compiler_invocations=warm_invocations,
        warm_native_phase_count=_native_phase_count(report),
        sources_unchanged=before_sources == after_sources,
        artifact_count=len(artifacts),
        native_variant_count=len(variants),
        families=families,
        errors=errors,
        passed=not errors,
    )


def evaluate_family(inputs: FamilyEvaluationInputs) -> FamilyEvidence:
    """Apply stability, semantics, routing, and 3x policy to paired samples.

    Args:
        inputs: Raw family measurement and expected runtime bindings.

    Returns:
        FamilyEvidence: Normalized per-family verdict and diagnostics.
    """
    if not inputs.baseline_samples or len(inputs.baseline_samples) != len(inputs.compiled_samples):
        raise ValueError("native hard benchmark requires paired non-empty family samples")
    baseline_median = median(inputs.baseline_samples)
    compiled_median = median(inputs.compiled_samples)
    assessment = assess_speedup(
        baseline_median,
        compiled_median,
        minimum_speedup=HARD_BENCHMARK_MINIMUM_SPEEDUP,
    )
    diagnostics: list[str] = []
    if not assessment.stable:
        diagnostics.append(
            "both benchmark medians must meet the "
            f"{MINIMUM_STABLE_MEDIAN_SECONDS:.2f}s stability floor"
        )
    elif not assessment.passed:
        diagnostics.append(
            f"measured {assessment.speedup:.3f}x, below the "
            f"{HARD_BENCHMARK_MINIMUM_SPEEDUP:.3f}x hard floor"
        )
    if not inputs.semantic_match:
        diagnostics.append("baseline and compiled benchmark results differ")
    missing = tuple(sorted(set(inputs.expected_bindings) - set(inputs.active_bindings)))
    if missing:
        diagnostics.append(f"compiled payload did not route binding(s): {', '.join(missing)}")
    return FamilyEvidence(
        name=inputs.name,
        calls=inputs.calls,
        baseline_median_seconds=baseline_median,
        baseline_samples_seconds=inputs.baseline_samples,
        compiled_median_seconds=compiled_median,
        compiled_samples_seconds=inputs.compiled_samples,
        speedup=assessment.speedup,
        semantic_match=inputs.semantic_match,
        active_bindings=inputs.active_bindings,
        expected_bindings=inputs.expected_bindings,
        diagnostics=tuple(diagnostics),
        passed=not diagnostics,
    )


def _measure_family(spec: FamilySpec, workspace: Path, payload_root: Path) -> FamilyEvidence:
    baseline_environment = _payload_environment(workspace / "src", compiled=False)
    compiled_environment = _payload_environment(payload_root, compiled=True)
    calls, baseline_probe, compiled_probe = _calibrate_family(
        spec,
        workspace,
        baseline_environment,
        compiled_environment,
    )
    semantic_match = _comparable_payload(spec, baseline_probe.payload) == _comparable_payload(
        spec, compiled_probe.payload
    )
    active_bindings = _probe_active_bindings(spec, workspace, compiled_environment)
    baseline_samples: list[float] = []
    compiled_samples: list[float] = []
    for sample in range(DEFAULT_WARMUPS + DEFAULT_SAMPLES):
        arms = ("baseline", "compiled") if sample % 2 == 0 else ("compiled", "baseline")
        for arm in arms:
            environment = baseline_environment if arm == "baseline" else compiled_environment
            result = _run_family_arm(spec, calls, workspace, environment)
            expected = baseline_probe.payload if arm == "baseline" else compiled_probe.payload
            if _comparable_payload(spec, result.payload) != _comparable_payload(spec, expected):
                raise BenchmarkError(f"{spec.name} produced unstable semantic output")
            if sample >= DEFAULT_WARMUPS:
                target = baseline_samples if arm == "baseline" else compiled_samples
                target.append(result.elapsed_seconds)
    return evaluate_family(
        FamilyEvaluationInputs(
            name=spec.name,
            calls=calls,
            baseline_samples=tuple(baseline_samples),
            compiled_samples=tuple(compiled_samples),
            semantic_match=semantic_match,
            active_bindings=active_bindings,
            expected_bindings=spec.bindings,
        )
    )


def _calibrate_family(
    spec: FamilySpec,
    workspace: Path,
    baseline_environment: dict[str, str],
    compiled_environment: dict[str, str],
) -> tuple[int, _ArmResult, _ArmResult]:
    calls = spec.initial_calls
    baseline = _run_family_arm(spec, calls, workspace, baseline_environment)
    compiled = _run_family_arm(spec, calls, workspace, compiled_environment)
    for _attempt in range(MAX_CALIBRATION_PASSES):
        fastest = min(baseline.elapsed_seconds, compiled.elapsed_seconds)
        if fastest >= DEFAULT_TARGET_FAST_SECONDS:
            return calls, baseline, compiled
        scale = max(2.0, DEFAULT_TARGET_FAST_SECONDS / max(fastest, 0.001))
        calls = max(calls + 1, int(calls * scale))
        baseline = _run_family_arm(spec, calls, workspace, baseline_environment)
        compiled = _run_family_arm(spec, calls, workspace, compiled_environment)
    if min(baseline.elapsed_seconds, compiled.elapsed_seconds) >= DEFAULT_TARGET_FAST_SECONDS:
        return calls, baseline, compiled
    raise BenchmarkError(f"{spec.name} calibration could not reach the stability floor")


def _run_family_arm(
    spec: FamilySpec,
    calls: int,
    workspace: Path,
    environment: dict[str, str],
) -> _ArmResult:
    command = (
        sys.executable,
        spec.script,
        "--calls",
        str(calls),
        *spec.extra_args,
    )
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=workspace,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise BenchmarkError(
            f"{spec.name} benchmark exited {completed.returncode}: {completed.stderr.strip()}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise BenchmarkError(f"{spec.name} benchmark emitted invalid JSON") from error
    if not isinstance(payload, dict):
        raise BenchmarkError(f"{spec.name} benchmark JSON is not an object")
    return _ArmResult(elapsed_seconds=elapsed, payload=cast(dict[str, object], payload))


def _probe_active_bindings(
    spec: FamilySpec,
    workspace: Path,
    environment: dict[str, str],
) -> tuple[str, ...]:
    names = repr(spec.bindings)
    probe = (
        "import importlib, json; "
        f"module = importlib.import_module({spec.module!r}); "
        f"names = {names}; "
        "resolve = lambda name: __import__('functools').reduce(getattr, name.split('.'), module); "
        "print(json.dumps(sorted(name for name in names if "
        "getattr(resolve(name), '__atoll_binding_variants__', ()))))"
    )
    completed = subprocess.run(
        (sys.executable, "-c", probe),
        cwd=workspace,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise BenchmarkError(f"{spec.name} binding probe failed: {completed.stderr.strip()}")
    payload = json.loads(completed.stdout)
    if not isinstance(payload, list) or any(not isinstance(item, str) for item in payload):
        raise BenchmarkError(f"{spec.name} binding probe emitted invalid evidence")
    return tuple(cast(list[str], payload))


def _run_semantic_tests(workspace: Path, payload_root: Path, evidence_root: Path) -> None:
    config_path = evidence_root / "empty-pytest.ini"
    config_path.write_text("[pytest]\n", encoding="utf-8")
    command = (sys.executable, "-m", "pytest", "-c", str(config_path), "tests", "-q")
    for arm, payload, compiled in (
        ("baseline", workspace / "src", False),
        ("compiled", payload_root, True),
    ):
        completed = subprocess.run(
            command,
            cwd=workspace,
            env=_payload_environment(payload, compiled=compiled),
            check=False,
            capture_output=True,
            text=True,
        )
        (evidence_root / f"{arm}.tests.log").write_text(
            f"{completed.stdout}{completed.stderr}",
            encoding="utf-8",
        )
        if completed.returncode != 0:
            raise BenchmarkError(f"{arm} semantic tests exited {completed.returncode}")


def _global_errors(inputs: _GlobalEvaluationInputs) -> tuple[str, ...]:
    errors: list[str] = []
    report = inputs.report
    if _integer_field(report, "version") != COMPILE_REPORT_SCHEMA_VERSION:
        errors.append(f"warm compile report is not schema v{COMPILE_REPORT_SCHEMA_VERSION}")
    if inputs.cold_exit != 0 or inputs.warm_exit != 0:
        errors.append("cold or warm compile failed")
    if inputs.cold_invocations < 1:
        errors.append("cold compile did not invoke the native compiler")
    if inputs.warm_invocations:
        errors.append(f"warm compile invoked the native compiler {inputs.warm_invocations} time(s)")
    if _native_phase_count(report):
        errors.append("warm report contains native compiler phases")
    if not inputs.sources_unchanged:
        errors.append("fixture source hashes changed during source-clean compilation")
    composition = _mapping_field(report, "final_composition")
    if not _list_field(composition, "native_variant_ids"):
        errors.append("final composition contains no native variants")
    if not _list_field(composition, "artifacts"):
        errors.append("final composition contains no native artifacts")
    errors.extend(
        f"{family.name}: {diagnostic}"
        for family in inputs.families
        for diagnostic in family.diagnostics
    )
    return tuple(errors)


def _copy_fixture(workspace: Path) -> None:
    shutil.copytree(
        FIXTURE_ROOT,
        workspace,
        ignore=shutil.ignore_patterns(".atoll", ".pytest_cache", "__pycache__", "*.egg-info"),
    )


def _remove_configured_benchmark(pyproject_path: Path) -> None:
    lines = pyproject_path.read_text(encoding="utf-8").splitlines()
    retained = [line for line in lines if not line.startswith("benchmark_command =")]
    pyproject_path.write_text(f"{'\n'.join(retained)}\n", encoding="utf-8")
    parsed = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    compile_config = cast(dict[str, object], parsed["tool"]["atoll"]["compile"])
    if "benchmark_command" in compile_config:
        raise BenchmarkError("failed to remove fixture benchmark command")


def _compiler_probe_environment(evidence_root: Path) -> tuple[dict[str, str], Path]:
    real_cc = shutil.which("cc")
    real_cxx = shutil.which("c++")
    if real_cc is None or real_cxx is None:
        raise BenchmarkError("native benchmark requires cc and c++")
    probe_log = evidence_root / "compiler-probe.log"
    probe_log.write_text("", encoding="utf-8")
    environment = {
        **os.environ,
        "ATOLL_COMPILER_PROBE_LOG": str(probe_log),
        "ATOLL_REAL_CC": real_cc,
        "ATOLL_REAL_CXX": real_cxx,
    }
    probes = (
        ("atoll-cc-probe", "ATOLL_REAL_CC"),
        ("atoll-cxx-probe", "ATOLL_REAL_CXX"),
    )
    for name, target in probes:
        probe = evidence_root / name
        probe.write_text(
            "#!/bin/sh\n"
            'printf "%s\\n" "$0" >> "$ATOLL_COMPILER_PROBE_LOG"\n'
            f'exec "${target}" "$@"\n',
            encoding="utf-8",
        )
        probe.chmod(0o755)
        environment["CC" if target == "ATOLL_REAL_CC" else "CXX"] = str(probe)
    return environment, probe_log


def _run_compile(
    command: tuple[str, ...],
    workspace: Path,
    environment: dict[str, str],
    log_path: Path,
) -> int:
    completed = subprocess.run(
        command,
        cwd=workspace,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    print(completed.stdout, end="")
    return completed.returncode


def _copy_compile_reports(workspace: Path, evidence_root: Path, label: str) -> None:
    report_root = workspace / ".atoll"
    for suffix in ("json", "md"):
        source = report_root / f"compile-report.{suffix}"
        if not source.is_file():
            raise BenchmarkError(f"{label} compile did not produce {source.name}")
        shutil.copyfile(source, evidence_root / f"{label}.compile-report.{suffix}")


def _single_wheel(dist_root: Path) -> Path:
    wheels = tuple(dist_root.glob("*.whl"))
    if len(wheels) != 1:
        raise BenchmarkError(f"expected one promoted wheel, found {len(wheels)}")
    return wheels[0]


def _payload_environment(payload_root: Path, *, compiled: bool) -> dict[str, str]:
    environment = {**os.environ, "PYTHONPATH": str(payload_root)}
    if compiled:
        environment["ATOLL_REQUIRE_COMPILED"] = "1"
    else:
        environment["ATOLL_DISABLE"] = "1"
    return environment


def _comparable_payload(spec: FamilySpec, payload: dict[str, object]) -> dict[str, object]:
    comparable = dict(payload)
    if spec.name == "buffer":
        comparable.pop("active_bindings", None)
    return comparable


def _source_manifest(source_root: Path) -> dict[str, str]:
    return {
        path.relative_to(source_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source_root.rglob("*.py"))
    }


def _native_phase_count(report: dict[str, object]) -> int:
    build = _mapping_field(report, "build")
    return sum(
        isinstance(item, dict) and item.get("name") in _NATIVE_PHASES
        for item in _list_field(build, "phase_timings")
    )


def _line_count(path: Path) -> int:
    return sum(1 for _line in path.read_text(encoding="utf-8").splitlines())


def _read_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BenchmarkError(f"JSON payload is not an object: {path}")
    return cast(dict[str, object], payload)


def _mapping_field(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise BenchmarkError(f"report field {key} is not an object")
    return cast(dict[str, object], value)


def _list_field(payload: dict[str, object], key: str) -> list[object]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise BenchmarkError(f"report field {key} is not a list")
    return value


def _integer_field(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise BenchmarkError(f"report field {key} is not an integer")
    return value


def _parse_args(argv: tuple[str, ...]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
