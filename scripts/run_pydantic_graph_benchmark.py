"""Run Atoll's pinned, profile-guided Pydantic Graph hard benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

PYDANTIC_AI_REPOSITORY = "https://github.com/pydantic/pydantic-ai.git"
PYDANTIC_AI_REVISION = "e6ff64409f74124de581068be644a3dbf8999e7d"
COLD_MYPYC_BASELINE_SECONDS = 192.70191520900698
COLD_MYPYC_TARGET_SECONDS = COLD_MYPYC_BASELINE_SECONDS / 2
MINIMUM_MARGINAL_SPEEDUP = 1.05
MINIMUM_FINAL_SPEEDUP = 1.10
BENCHMARK_SAMPLES = 7
COMPILE_REPORT_VERSION = 4
NATIVE_PHASES = frozenset({"mypycify", "cythonize", "build_ext"})


class BenchmarkError(RuntimeError):
    """Raised when the manual benchmark cannot produce trustworthy evidence."""


@dataclass(frozen=True, slots=True)
class BenchmarkOptions:
    """Filesystem and interpreter boundaries for one disposable benchmark run."""

    atoll_root: Path
    workspace: Path
    evidence_root: Path
    python_version: str = "3.12"


@dataclass(frozen=True, slots=True)
class BenchmarkEvaluation:
    """Normalized hard-gate metrics and every failed acceptance condition."""

    cold_mypyc_seconds: float
    cold_native_phase_count: int
    cold_compiler_probe_count: int
    warm_native_phase_count: int
    warm_compiler_probe_count: int
    warm_cache_status: str
    final_speedup: float | None
    accepted_candidates: int
    execution_plan_count: int
    applied_execution_plan_count: int
    execution_plan_trial_count: int
    accepted_execution_plan_trials: int
    fusion_plan_count: int
    eligible_fusion_plan_count: int
    fusion_trial_count: int
    errors: tuple[str, ...]

    @property
    def succeeded(self) -> bool:
        """Return whether all hard-benchmark acceptance conditions passed."""
        return not self.errors

    def as_json(self) -> dict[str, object]:
        """Return stable JSON evidence for workflow artifacts."""
        return {
            "accepted_candidates": self.accepted_candidates,
            "cold_mypyc_seconds": self.cold_mypyc_seconds,
            "cold_mypyc_target_seconds": COLD_MYPYC_TARGET_SECONDS,
            "cold_native_phase_count": self.cold_native_phase_count,
            "cold_compiler_probe_count": self.cold_compiler_probe_count,
            "errors": list(self.errors),
            "execution_plan_count": self.execution_plan_count,
            "applied_execution_plan_count": self.applied_execution_plan_count,
            "execution_plan_trial_count": self.execution_plan_trial_count,
            "accepted_execution_plan_trials": self.accepted_execution_plan_trials,
            "final_speedup": self.final_speedup,
            "fusion_plan_count": self.fusion_plan_count,
            "eligible_fusion_plan_count": self.eligible_fusion_plan_count,
            "fusion_trial_count": self.fusion_trial_count,
            "minimum_final_speedup": MINIMUM_FINAL_SPEEDUP,
            "minimum_marginal_speedup": MINIMUM_MARGINAL_SPEEDUP,
            "succeeded": self.succeeded,
            "warm_cache_status": self.warm_cache_status,
            "warm_compiler_probe_count": self.warm_compiler_probe_count,
            "warm_native_phase_count": self.warm_native_phase_count,
        }


def main(argv: tuple[str, ...] | None = None) -> int:
    """Clone the pinned target, compile it twice, and enforce all hard gates."""
    options = _parse_options(tuple(sys.argv[1:] if argv is None else argv))
    try:
        evaluation = run_benchmark(options)
    except (BenchmarkError, OSError, subprocess.SubprocessError) as error:
        print(f"Pydantic Graph benchmark could not complete: {error}", file=sys.stderr)
        return 1
    if evaluation.succeeded:
        final_speedup = evaluation.final_speedup
        if final_speedup is None:
            raise BenchmarkError("successful evaluation did not retain final speedup")
        print(
            "Pydantic Graph hard benchmark passed: "
            f"{final_speedup:.3f}x final speedup, "
            f"{evaluation.cold_mypyc_seconds:.3f}s cold mypyc."
        )
        return 0
    print("Pydantic Graph hard benchmark failed:", file=sys.stderr)
    for failure in evaluation.errors:
        print(f"- {failure}", file=sys.stderr)
    return 1


def run_benchmark(options: BenchmarkOptions) -> BenchmarkEvaluation:
    """Create a disposable checkout and retain cold and warm acceptance evidence."""
    paths = _prepare_benchmark_paths(options)
    _clone_pinned_checkout(options.workspace)
    target_root = options.workspace / "pydantic_graph"
    workload_path, probe_environment, probe_log = _materialize_benchmark_assets(
        options.atoll_root,
        options.evidence_root,
    )
    append_compile_policy(target_root / "pyproject.toml", workload_path)
    before_sources = source_manifest(target_root / "pydantic_graph")
    _write_json(options.evidence_root / "source-hashes-before.json", before_sources)
    command = _compile_command(options, target_root)
    environment = {**os.environ, **probe_environment, "UV_DYNAMIC_VERSIONING_BYPASS": "0.0.0"}

    cold_exit = _run_streamed(command, target_root, environment, paths.cold_log)
    _copy_report(target_root, options.evidence_root, "cold")
    cold_probe_count = _line_count(probe_log)
    warm_exit = _run_streamed(command, target_root, environment, paths.warm_log)
    _copy_report(target_root, options.evidence_root, "warm")
    warm_probe_count = _line_count(probe_log) - cold_probe_count

    after_sources = source_manifest(target_root / "pydantic_graph")
    _write_json(options.evidence_root / "source-hashes-after.json", after_sources)
    cold_report = _read_json_object(options.evidence_root / "cold.compile-report.json")
    warm_report = _read_json_object(options.evidence_root / "warm.compile-report.json")
    wheel_present = any((target_root / ".atoll" / "dist").glob("*.whl"))
    evaluation = evaluate_reports(
        BenchmarkEvidenceInputs(
            cold_report=cold_report,
            warm_report=warm_report,
            sources_unchanged=before_sources == after_sources,
            cold_exit_code=cold_exit,
            warm_exit_code=warm_exit,
            wheel_present=wheel_present,
            cold_compiler_probe_count=cold_probe_count,
            warm_compiler_probe_count=warm_probe_count,
        )
    )
    _write_summary(options, command, evaluation)
    return evaluation


@dataclass(frozen=True, slots=True)
class _BenchmarkPaths:
    cold_log: Path
    warm_log: Path


def _prepare_benchmark_paths(options: BenchmarkOptions) -> _BenchmarkPaths:
    atoll_root = options.atoll_root.resolve()
    if not (atoll_root / "pyproject.toml").is_file():
        raise BenchmarkError(f"Atoll root is not a Python project: {atoll_root}")
    if options.workspace.exists():
        raise BenchmarkError(f"benchmark workspace already exists: {options.workspace}")
    options.workspace.parent.mkdir(parents=True, exist_ok=True)
    options.evidence_root.mkdir(parents=True, exist_ok=True)
    return _BenchmarkPaths(
        cold_log=options.evidence_root / "cold.compile.log",
        warm_log=options.evidence_root / "warm.compile.log",
    )


def _clone_pinned_checkout(workspace: Path) -> None:
    git = _required_executable("git")
    _run_checked(
        (
            git,
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            PYDANTIC_AI_REPOSITORY,
            str(workspace),
        ),
        cwd=workspace.parent,
    )
    _run_checked(
        (git, "-C", str(workspace), "checkout", "--detach", PYDANTIC_AI_REVISION),
        cwd=workspace.parent,
    )
    result = subprocess.run(
        (git, "-C", str(workspace), "rev-parse", "HEAD"),
        cwd=workspace.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip() != PYDANTIC_AI_REVISION:
        raise BenchmarkError(f"checkout resolved unexpected revision {result.stdout.strip()}")


def _materialize_benchmark_assets(
    atoll_root: Path,
    evidence_root: Path,
) -> tuple[Path, dict[str, str], Path]:
    template_root = atoll_root / "benchmarks" / "pydantic_graph"
    workload_path = evidence_root / "pydantic_graph_workload.py"
    shutil.copyfile(template_root / "workload.py.in", workload_path)
    probe_template = template_root / "compiler_probe.sh.in"
    cc_probe = evidence_root / "atoll-cc-probe"
    cxx_probe = evidence_root / "atoll-cxx-probe"
    for probe in (cc_probe, cxx_probe):
        shutil.copyfile(probe_template, probe)
        probe.chmod(0o755)
    probe_log = evidence_root / "compiler-probe.log"
    probe_log.touch()
    return (
        workload_path,
        {
            "ATOLL_COMPILER_PROBE_LOG": str(probe_log),
            "ATOLL_REAL_CC": _required_executable("cc"),
            "ATOLL_REAL_CXX": _required_executable("c++"),
            "CC": str(cc_probe),
            "CXX": str(cxx_probe),
        },
        probe_log,
    )


def append_compile_policy(pyproject: Path, workload_path: Path) -> None:
    """Append deterministic profiling policy to a disposable target project."""
    source = pyproject.read_text(encoding="utf-8")
    parsed = cast(dict[str, object], tomllib.loads(source))
    tool = _mapping(parsed.get("tool", {}), "tool")
    atoll = _mapping(tool.get("atoll", {}), "tool.atoll")
    if "compile" in atoll:
        raise BenchmarkError("pinned target already defines [tool.atoll.compile]")
    workload = json.dumps(str(workload_path.resolve()))
    policy = "\n".join(
        (
            "[tool.atoll.compile]",
            'backends = ["mypyc", "cython"]',
            f'test_command = ["python", {workload}, "--verify"]',
            f'benchmark_command = ["python", {workload}]',
            "benchmark_warmups = 1",
            f"benchmark_samples = {BENCHMARK_SAMPLES}",
            f"minimum_speedup = {MINIMUM_FINAL_SPEEDUP:.2f}",
            "",
        )
    )
    pyproject.write_text(f"{source.rstrip()}\n\n{policy}", encoding="utf-8")


def _compile_command(options: BenchmarkOptions, target_root: Path) -> tuple[str, ...]:
    return (
        _required_executable("uv"),
        "run",
        "--no-project",
        "--python",
        options.python_version,
        "--with-editable",
        str(options.atoll_root.resolve()),
        "--with-editable",
        str(target_root.resolve()),
        "atoll",
        "compile",
        "--root",
        str(target_root.resolve()),
    )


def _run_streamed(
    command: tuple[str, ...],
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
) -> int:
    print(f"$ {' '.join(command)}")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise BenchmarkError("compile subprocess did not expose stdout")
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return process.wait()


def _copy_report(target_root: Path, evidence_root: Path, label: str) -> None:
    report_root = target_root / ".atoll"
    json_report = report_root / "compile-report.json"
    markdown_report = report_root / "compile-report.md"
    if not json_report.is_file() or not markdown_report.is_file():
        raise BenchmarkError(f"{label} compile did not produce both reports")
    shutil.copyfile(json_report, evidence_root / f"{label}.compile-report.json")
    shutil.copyfile(markdown_report, evidence_root / f"{label}.compile-report.md")


@dataclass(frozen=True, slots=True)
class BenchmarkEvidenceInputs:
    """Cold and warm report evidence supplied to the hard-gate evaluator."""

    cold_report: dict[str, object]
    warm_report: dict[str, object]
    sources_unchanged: bool
    cold_exit_code: int
    warm_exit_code: int
    wheel_present: bool
    cold_compiler_probe_count: int
    warm_compiler_probe_count: int


def evaluate_reports(inputs: BenchmarkEvidenceInputs) -> BenchmarkEvaluation:
    """Evaluate fixed cold-build, cache, semantics, and profitability gates."""
    cold_mypyc_seconds = _phase_duration(inputs.cold_report, "mypycify")
    cold_native_phase_count = sum(
        _phase_count(inputs.cold_report, phase) for phase in NATIVE_PHASES
    )
    warm_native_phase_count = sum(
        _phase_count(inputs.warm_report, phase) for phase in NATIVE_PHASES
    )
    warm_cache_status = _string_field(_mapping_field(inputs.warm_report, "build"), "cache_status")
    final_speedup = _optional_number_field(
        _mapping_field(inputs.warm_report, "performance"), "speedup"
    )
    accepted_speedups = _accepted_candidate_speedups(inputs.warm_report)
    execution_plans = _execution_plans(inputs.warm_report)
    applied_execution_plan_ids = _applied_execution_plan_ids(inputs.warm_report)
    execution_plan_trials = _execution_plan_trials(inputs.warm_report)
    fusion_plans = _fusion_plans(inputs.warm_report)
    fusion_trials = _fusion_trials(inputs.warm_report)
    accepted_execution_plan_trials = tuple(
        trial for trial in execution_plan_trials if _string_field(trial, "status") == "accepted"
    )
    errors = _evaluation_errors(
        _EvaluationInputs(
            cold_report=inputs.cold_report,
            warm_report=inputs.warm_report,
            sources_unchanged=inputs.sources_unchanged,
            cold_exit_code=inputs.cold_exit_code,
            warm_exit_code=inputs.warm_exit_code,
            wheel_present=inputs.wheel_present,
            cold_mypyc_seconds=cold_mypyc_seconds,
            cold_native_phase_count=cold_native_phase_count,
            cold_compiler_probe_count=inputs.cold_compiler_probe_count,
            warm_native_phase_count=warm_native_phase_count,
            warm_compiler_probe_count=inputs.warm_compiler_probe_count,
            warm_cache_status=warm_cache_status,
            final_speedup=final_speedup,
            accepted_speedups=accepted_speedups,
            native_evidence_present=_native_evidence_present(
                inputs.cold_report, inputs.warm_report
            ),
            execution_plan_count=len(execution_plans),
            applied_execution_plan_ids=applied_execution_plan_ids,
            execution_plan_trial_count=len(execution_plan_trials),
            accepted_execution_plan_trials=accepted_execution_plan_trials,
            fusion_plan_count=len(fusion_plans),
            eligible_fusion_plan_count=sum(
                _boolean_field(plan, "eligible") for plan in fusion_plans
            ),
            fusion_trial_count=len(fusion_trials),
        )
    )
    return BenchmarkEvaluation(
        cold_mypyc_seconds=cold_mypyc_seconds,
        cold_native_phase_count=cold_native_phase_count,
        cold_compiler_probe_count=inputs.cold_compiler_probe_count,
        warm_native_phase_count=warm_native_phase_count,
        warm_compiler_probe_count=inputs.warm_compiler_probe_count,
        warm_cache_status=warm_cache_status,
        final_speedup=final_speedup,
        accepted_candidates=len(accepted_speedups),
        execution_plan_count=len(execution_plans),
        applied_execution_plan_count=len(applied_execution_plan_ids),
        execution_plan_trial_count=len(execution_plan_trials),
        accepted_execution_plan_trials=len(accepted_execution_plan_trials),
        fusion_plan_count=len(fusion_plans),
        eligible_fusion_plan_count=sum(_boolean_field(plan, "eligible") for plan in fusion_plans),
        fusion_trial_count=len(fusion_trials),
        errors=errors,
    )


@dataclass(frozen=True, slots=True)
class _EvaluationInputs:
    cold_report: dict[str, object]
    warm_report: dict[str, object]
    sources_unchanged: bool
    cold_exit_code: int
    warm_exit_code: int
    wheel_present: bool
    cold_mypyc_seconds: float
    cold_native_phase_count: int
    cold_compiler_probe_count: int
    warm_native_phase_count: int
    warm_compiler_probe_count: int
    warm_cache_status: str
    final_speedup: float | None
    accepted_speedups: tuple[float | None, ...]
    native_evidence_present: bool
    execution_plan_count: int
    applied_execution_plan_ids: tuple[str, ...]
    execution_plan_trial_count: int
    accepted_execution_plan_trials: tuple[dict[str, object], ...]
    fusion_plan_count: int
    eligible_fusion_plan_count: int
    fusion_trial_count: int


def _evaluation_errors(inputs: _EvaluationInputs) -> tuple[str, ...]:
    return (
        *_report_errors(inputs),
        *_cache_errors(inputs),
        *_source_errors(inputs),
        *_execution_plan_errors(inputs),
        *_profitability_errors(inputs),
    )


def _report_errors(inputs: _EvaluationInputs) -> tuple[str, ...]:
    errors: list[str] = []
    for label, report, exit_code in (
        ("cold", inputs.cold_report, inputs.cold_exit_code),
        ("warm", inputs.warm_report, inputs.warm_exit_code),
    ):
        if _integer_field(report, "version") != COMPILE_REPORT_VERSION:
            errors.append(f"{label} report is not schema version {COMPILE_REPORT_VERSION}")
        if exit_code != 0 or not _boolean_field(report, "success"):
            errors.append(f"{label} compile did not succeed")
        profile_status = _string_field(_mapping_field(report, "profile"), "status")
        if profile_status != "profiled":
            errors.append(f"{label} profile status is {profile_status}, expected profiled")
    return tuple(errors)


def _cache_errors(inputs: _EvaluationInputs) -> tuple[str, ...]:
    errors: list[str] = []
    cold_cache = _string_field(_mapping_field(inputs.cold_report, "build"), "cache_status")
    if cold_cache != "miss":
        errors.append(f"cold cache status is {cold_cache}, expected miss")
    if inputs.warm_cache_status != "hit":
        errors.append(f"warm cache status is {inputs.warm_cache_status}, expected hit")
    if inputs.native_evidence_present:
        if inputs.cold_mypyc_seconds > COLD_MYPYC_TARGET_SECONDS:
            errors.append(
                f"cold mypyc took {inputs.cold_mypyc_seconds:.3f}s, above "
                f"{COLD_MYPYC_TARGET_SECONDS:.3f}s"
            )
        if inputs.cold_native_phase_count == 0:
            errors.append("cold report contains no native compiler phase")
        if inputs.cold_compiler_probe_count == 0:
            errors.append("compiler probe observed no cold native invocation")
        if any(status != "hit" for status in _compiled_cache_statuses(inputs.warm_report)):
            errors.append("one or more warm compiled regions were not cache hits")
    if inputs.warm_native_phase_count:
        errors.append(
            f"warm report contains {inputs.warm_native_phase_count} native compiler phase(s)"
        )
    if inputs.warm_compiler_probe_count:
        errors.append(
            f"compiler probe observed {inputs.warm_compiler_probe_count} warm invocation(s)"
        )
    return tuple(errors)


def _source_errors(inputs: _EvaluationInputs) -> tuple[str, ...]:
    errors: list[str] = []
    if not inputs.sources_unchanged:
        errors.append("pydantic_graph source hashes changed during compilation")
    cold_typed_hashes = _typed_region_hashes(inputs.cold_report)
    warm_typed_hashes = _typed_region_hashes(inputs.warm_report)
    if (cold_typed_hashes or warm_typed_hashes) and cold_typed_hashes != warm_typed_hashes:
        errors.append("typed-region source hashes changed between cold and warm reports")
    errors.extend(_execution_plan_source_hash_errors("cold", inputs.cold_report))
    errors.extend(_execution_plan_source_hash_errors("warm", inputs.warm_report))
    if _execution_plan_hashes(inputs.cold_report) != _execution_plan_hashes(inputs.warm_report):
        errors.append("execution-plan identities or source hashes changed between runs")
    return tuple(errors)


def _execution_plan_source_hash_errors(
    label: str,
    report: dict[str, object],
) -> tuple[str, ...]:
    errors: list[str] = []
    for plan in _execution_plans(report):
        if _string_field(plan, "status") != "selected":
            continue
        plan_id = _string_field(plan, "id")
        source_hash = plan.get("source_hash")
        if not isinstance(source_hash, str) or not source_hash:
            errors.append(f"{label} selected execution plan {plan_id} has no source hash")
        source_hashes = plan.get("source_hashes")
        source_hash_items = (
            cast(dict[object, object], source_hashes).items()
            if isinstance(source_hashes, dict)
            else ()
        )
        if (
            not isinstance(source_hashes, dict)
            or not source_hashes
            or any(
                not isinstance(module, str)
                or not module
                or not isinstance(digest, str)
                or not digest
                for module, digest in source_hash_items
            )
        ):
            errors.append(
                f"{label} selected execution plan {plan_id} has no per-module source hashes"
            )
    return tuple(errors)


def _execution_plan_errors(inputs: _EvaluationInputs) -> tuple[str, ...]:
    return (
        *_execution_plan_discovery_errors(inputs),
        *_accepted_execution_plan_trial_errors(inputs),
        *_cold_execution_plan_cache_errors(inputs),
    )


def _execution_plan_discovery_errors(inputs: _EvaluationInputs) -> tuple[str, ...]:
    errors: list[str] = []
    if inputs.execution_plan_count == 0:
        errors.append("warm report contains no discovered execution plan")
    if not inputs.applied_execution_plan_ids:
        errors.append("warm report contains no applied execution plan")
    selected_plan_ids = {
        _string_field(plan, "id")
        for plan in _execution_plans(inputs.warm_report)
        if _string_field(plan, "status") == "selected"
    }
    errors.extend(
        f"applied execution plan {plan_id} was not discovered as selected"
        for plan_id in inputs.applied_execution_plan_ids
        if plan_id not in selected_plan_ids
    )
    return tuple(errors)


def _accepted_execution_plan_trial_errors(inputs: _EvaluationInputs) -> tuple[str, ...]:
    errors: list[str] = []
    accepted_by_plan = {
        _string_field(trial, "plan_id"): trial for trial in inputs.accepted_execution_plan_trials
    }
    for plan_id in inputs.applied_execution_plan_ids:
        trial = accepted_by_plan.get(plan_id)
        if trial is None:
            errors.append(f"applied execution plan {plan_id} has no accepted trial")
            continue
        marginal = _optional_number_field(trial, "marginal_speedup")
        overall = _optional_number_field(trial, "overall_speedup")
        cache_status = _string_field(trial, "cache_status")
        if marginal is None or marginal < MINIMUM_MARGINAL_SPEEDUP:
            errors.append("an accepted execution-plan trial is below 1.05x marginal speedup")
        if overall is None or overall < MINIMUM_FINAL_SPEEDUP:
            errors.append("an accepted execution-plan trial is below 1.10x overall speedup")
        if cache_status != "hit":
            errors.append(f"warm execution-plan trial cache status is {cache_status}, expected hit")
    return tuple(errors)


def _cold_execution_plan_cache_errors(inputs: _EvaluationInputs) -> tuple[str, ...]:
    errors: list[str] = []
    cold_trial_statuses = {
        _string_field(trial, "plan_id"): _string_field(trial, "cache_status")
        for trial in _execution_plan_trials(inputs.cold_report)
        if _string_field(trial, "status") == "accepted"
    }
    for plan_id in inputs.applied_execution_plan_ids:
        cache_status = cold_trial_statuses.get(plan_id)
        if cache_status != "miss":
            errors.append(f"cold execution-plan trial for {plan_id} was not a cache miss")
    return tuple(errors)


def _profitability_errors(inputs: _EvaluationInputs) -> tuple[str, ...]:
    errors: list[str] = []
    if not inputs.wheel_present:
        errors.append("warm compile did not leave a promoted wheel")
    if any(
        speedup is None or speedup < MINIMUM_MARGINAL_SPEEDUP
        for speedup in inputs.accepted_speedups
    ):
        errors.append("an accepted candidate is missing the required 1.05x marginal speedup")
    if inputs.final_speedup is None or inputs.final_speedup < MINIMUM_FINAL_SPEEDUP:
        errors.append("warm final speedup is below 1.10x")
    return tuple(errors)


def source_manifest(source_root: Path) -> dict[str, str]:
    """Hash every Python source file without importing or rewriting the target."""
    return {
        path.relative_to(source_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source_root.rglob("*.py"))
    }


def _phase_duration(report: dict[str, object], name: str) -> float:
    return sum(
        _number_field(timing, "duration_seconds")
        for timing in _phase_timings(report)
        if _string_field(timing, "name") == name
    )


def _phase_count(report: dict[str, object], name: str) -> int:
    return sum(_string_field(timing, "name") == name for timing in _phase_timings(report))


def _phase_timings(report: dict[str, object]) -> tuple[dict[str, object], ...]:
    build = _mapping_field(report, "build")
    return tuple(
        _mapping(item, "build.phase_timings[]") for item in _list_field(build, "phase_timings")
    )


def _accepted_candidate_speedups(report: dict[str, object]) -> tuple[float | None, ...]:
    trials = _list_field(report, "candidate_trials")
    return tuple(
        _optional_number_field(trial, "marginal_speedup")
        for item in trials
        if _string_field((trial := _mapping(item, "candidate_trials[]")), "status") == "accepted"
    )


def _compiled_cache_statuses(report: dict[str, object]) -> tuple[str, ...]:
    return tuple(
        _string_field(_mapping(item, "compiled_regions[]"), "cache_status")
        for item in _optional_list_field(report, "compiled_regions")
    )


def _typed_region_hashes(report: dict[str, object]) -> dict[str, str]:
    return {
        _string_field(region, "id"): _string_field(region, "source_hash")
        for item in _optional_list_field(report, "typed_regions")
        for region in (_mapping(item, "typed_regions[]"),)
    }


def _execution_plans(report: dict[str, object]) -> tuple[dict[str, object], ...]:
    return tuple(
        _mapping(item, "execution_plans[]")
        for item in _optional_list_field(report, "execution_plans")
    )


def _applied_execution_plan_ids(report: dict[str, object]) -> tuple[str, ...]:
    return tuple(
        str(item) if isinstance(item, str) else _raise_field_error("applied_execution_plans[]")
        for item in _optional_list_field(report, "applied_execution_plans")
    )


def _execution_plan_trials(report: dict[str, object]) -> tuple[dict[str, object], ...]:
    return tuple(
        _mapping(item, "execution_plan_trials[]")
        for item in _optional_list_field(report, "execution_plan_trials")
    )


def _execution_plan_hashes(report: dict[str, object]) -> dict[str, str]:
    return {
        _string_field(plan, "id"): source_hash
        for plan in _execution_plans(report)
        if _string_field(plan, "status") == "selected"
        and isinstance((source_hash := plan.get("source_hash")), str)
    }


def _native_evidence_present(
    cold_report: dict[str, object],
    warm_report: dict[str, object],
) -> bool:
    return bool(
        _typed_region_hashes(cold_report)
        or _typed_region_hashes(warm_report)
        or _compiled_cache_statuses(cold_report)
        or _compiled_cache_statuses(warm_report)
        or _accepted_candidate_speedups(warm_report)
    )


def _fusion_plans(report: dict[str, object]) -> tuple[dict[str, object], ...]:
    return tuple(
        _mapping(item, "fusion_plans[]") for item in _optional_list_field(report, "fusion_plans")
    )


def _fusion_trials(report: dict[str, object]) -> tuple[dict[str, object], ...]:
    return tuple(
        _mapping(item, "fusion_trials[]") for item in _optional_list_field(report, "fusion_trials")
    )


def _write_summary(
    options: BenchmarkOptions,
    command: tuple[str, ...],
    evaluation: BenchmarkEvaluation,
) -> None:
    summary = {
        "atoll_root": str(options.atoll_root.resolve()),
        "command": list(command),
        "evaluation": evaluation.as_json(),
        "pydantic_ai_repository": PYDANTIC_AI_REPOSITORY,
        "pydantic_ai_revision": PYDANTIC_AI_REVISION,
        "python_version": options.python_version,
        "workspace": str(options.workspace.resolve()),
    }
    _write_json(options.evidence_root / "acceptance-summary.json", summary)


def _parse_options(argv: tuple[str, ...]) -> BenchmarkOptions:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--atoll-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", default="3.12")
    args = parser.parse_args(argv)
    workspace = cast(Path, args.workspace).resolve()
    evidence = cast(Path | None, args.evidence_root)
    return BenchmarkOptions(
        atoll_root=cast(Path, args.atoll_root).resolve(),
        workspace=workspace,
        evidence_root=(
            evidence.resolve()
            if evidence is not None
            else workspace.parent / "pydantic-graph-evidence"
        ),
        python_version=cast(str, args.python),
    )


def _run_checked(command: tuple[str, ...], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _required_executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise BenchmarkError(f"required executable is unavailable: {name}")
    return executable


def _line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)


def _read_json_object(path: Path) -> dict[str, object]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise BenchmarkError(f"{field} must be an object")
    return cast(dict[str, object], value)


def _mapping_field(payload: dict[str, object], field: str) -> dict[str, object]:
    return _mapping(payload.get(field), field)


def _list_field(payload: dict[str, object], field: str) -> list[object]:
    value = payload.get(field)
    if not isinstance(value, list):
        raise BenchmarkError(f"{field} must be an array")
    return cast(list[object], value)


def _optional_list_field(payload: dict[str, object], field: str) -> list[object]:
    value = payload.get(field, [])
    if not isinstance(value, list):
        raise BenchmarkError(f"{field} must be an array")
    return cast(list[object], value)


def _string_field(payload: dict[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str):
        raise BenchmarkError(f"{field} must be a string")
    return value


def _boolean_field(payload: dict[str, object], field: str) -> bool:
    value = payload.get(field)
    if not isinstance(value, bool):
        raise BenchmarkError(f"{field} must be a boolean")
    return value


def _integer_field(payload: dict[str, object], field: str) -> int:
    value = payload.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise BenchmarkError(f"{field} must be an integer")
    return value


def _number_field(payload: dict[str, object], field: str) -> float:
    value = payload.get(field)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise BenchmarkError(f"{field} must be a number")
    return float(value)


def _optional_number_field(payload: dict[str, object], field: str) -> float | None:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise BenchmarkError(f"{field} must be a number or null")
    return float(value)


def _raise_field_error(field: str) -> str:
    raise BenchmarkError(f"{field} must be a string")


if __name__ == "__main__":
    raise SystemExit(main())
