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
from pathlib import Path, PurePosixPath
from typing import cast

PYDANTIC_AI_REPOSITORY = "https://github.com/pydantic/pydantic-ai.git"
PYDANTIC_AI_REVISION = "e6ff64409f74124de581068be644a3dbf8999e7d"
COLD_MYPYC_BASELINE_SECONDS = 192.70191520900698
COLD_MYPYC_TARGET_SECONDS = COLD_MYPYC_BASELINE_SECONDS / 2
MINIMUM_SOURCE_SPEEDUP = 3.0
MINIMUM_FINAL_SPEEDUP = 3.0
BENCHMARK_SAMPLES = 7
COMPILE_REPORT_VERSION = 5
MINIMUM_OBSERVED_WORK_ITEMS = 10_000
MINIMUM_ATTRIBUTED_HOT_SHARE = 0.70
ATOLL_PATCH_PATH_PARTS = 3
NATIVE_PHASES = frozenset({"mypycify", "cythonize", "build_ext"})
REQUIRED_SOURCE_TRANSFORMATION_KINDS = frozenset(
    {
        "local-state-machine-fusion",
        "private-protocol-auto-forwarding",
        "private-transport-batch-drain",
        "quiescent-callable-execution",
    }
)


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

    cold_native_phase_count: int
    cold_compiler_probe_count: int
    warm_native_phase_count: int
    warm_compiler_probe_count: int
    cold_patch_cache_status: str
    warm_patch_cache_status: str
    final_speedup: float | None
    source_speedup: float | None
    wheel_speedup: float | None
    source_plan_count: int
    trial_ready_source_plan_count: int
    source_trial_count: int
    accepted_source_trials: int
    patch_path: str | None
    application_status: str
    errors: tuple[str, ...]

    @property
    def succeeded(self) -> bool:
        """Return whether all hard-benchmark acceptance conditions passed."""
        return not self.errors

    def as_json(self) -> dict[str, object]:
        """Return stable JSON evidence for workflow artifacts."""
        return {
            "accepted_source_trials": self.accepted_source_trials,
            "cold_native_phase_count": self.cold_native_phase_count,
            "cold_compiler_probe_count": self.cold_compiler_probe_count,
            "cold_patch_cache_status": self.cold_patch_cache_status,
            "errors": list(self.errors),
            "final_speedup": self.final_speedup,
            "source_speedup": self.source_speedup,
            "wheel_speedup": self.wheel_speedup,
            "source_plan_count": self.source_plan_count,
            "trial_ready_source_plan_count": self.trial_ready_source_plan_count,
            "source_trial_count": self.source_trial_count,
            "patch_path": self.patch_path,
            "application_status": self.application_status,
            "minimum_final_speedup": MINIMUM_FINAL_SPEEDUP,
            "minimum_source_speedup": MINIMUM_SOURCE_SPEEDUP,
            "succeeded": self.succeeded,
            "warm_patch_cache_status": self.warm_patch_cache_status,
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
        source_speedup = evaluation.source_speedup
        if final_speedup is None or source_speedup is None:
            raise BenchmarkError("successful evaluation did not retain source and wheel speedups")
        print(
            "Pydantic Graph hard benchmark passed: "
            f"{source_speedup:.3f}x transformed source and "
            f"{final_speedup:.3f}x final wheel speedup."
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
    reported_patch = _reported_patch_path(warm_report)
    patch_present = reported_patch is not None and (target_root / reported_patch).is_file()
    evaluation = evaluate_reports(
        BenchmarkEvidenceInputs(
            cold_report=cold_report,
            warm_report=warm_report,
            sources_unchanged=before_sources == after_sources,
            cold_exit_code=cold_exit,
            warm_exit_code=warm_exit,
            wheel_present=wheel_present,
            patch_present=patch_present,
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
            f'benchmark_command = ["python", {workload}, "--build-repetitions", "0"]',
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
    patch_present: bool
    cold_compiler_probe_count: int
    warm_compiler_probe_count: int


def evaluate_reports(inputs: BenchmarkEvidenceInputs) -> BenchmarkEvaluation:
    """Evaluate schema-v5 source-patch reproducibility and 3x profitability gates."""
    cold_native_phase_count = sum(
        _phase_count(inputs.cold_report, phase) for phase in NATIVE_PHASES
    )
    warm_native_phase_count = sum(
        _phase_count(inputs.warm_report, phase) for phase in NATIVE_PHASES
    )
    cold_source = _source_optimization(inputs.cold_report)
    warm_source = _source_optimization(inputs.warm_report)
    warm_trials = _source_optimization_trials(inputs.warm_report)
    cold_accepted = _accepted_source_optimization_trials(inputs.cold_report)
    warm_accepted = _accepted_source_optimization_trials(inputs.warm_report)
    accepted_trial = _final_accepted_source_trial(inputs.warm_report)
    final_speedup = _optional_number_field(
        _mapping_field(inputs.warm_report, "performance"), "speedup"
    )
    source_speedup = (
        _optional_number_field(accepted_trial, "source_speedup")
        if accepted_trial is not None
        else None
    )
    wheel_speedup = (
        _optional_number_field(accepted_trial, "wheel_speedup")
        if accepted_trial is not None
        else None
    )
    patch_path = _reported_patch_path(inputs.warm_report)
    application_status = _string_field(warm_source, "application_status")
    errors = _evaluation_errors(
        _EvaluationInputs(
            cold_report=inputs.cold_report,
            warm_report=inputs.warm_report,
            sources_unchanged=inputs.sources_unchanged,
            cold_exit_code=inputs.cold_exit_code,
            warm_exit_code=inputs.warm_exit_code,
            wheel_present=inputs.wheel_present,
            patch_present=inputs.patch_present,
            cold_native_phase_count=cold_native_phase_count,
            cold_compiler_probe_count=inputs.cold_compiler_probe_count,
            warm_native_phase_count=warm_native_phase_count,
            warm_compiler_probe_count=inputs.warm_compiler_probe_count,
            final_speedup=final_speedup,
            source_speedup=source_speedup,
            wheel_speedup=wheel_speedup,
            cold_source=cold_source,
            warm_source=warm_source,
            cold_accepted_trials=cold_accepted,
            warm_accepted_trials=warm_accepted,
        )
    )
    return BenchmarkEvaluation(
        cold_native_phase_count=cold_native_phase_count,
        cold_compiler_probe_count=inputs.cold_compiler_probe_count,
        warm_native_phase_count=warm_native_phase_count,
        warm_compiler_probe_count=inputs.warm_compiler_probe_count,
        cold_patch_cache_status=_source_patch_cache_status(cold_accepted),
        warm_patch_cache_status=_source_patch_cache_status(warm_accepted),
        final_speedup=final_speedup,
        source_speedup=source_speedup,
        wheel_speedup=wheel_speedup,
        source_plan_count=len(_source_optimization_plans(inputs.warm_report)),
        trial_ready_source_plan_count=sum(
            _string_field(assessment, "status") == "trial-ready"
            for assessment in _source_optimization_assessments(inputs.warm_report)
        ),
        source_trial_count=len(warm_trials),
        accepted_source_trials=len(warm_accepted),
        patch_path=patch_path,
        application_status=application_status,
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
    patch_present: bool
    cold_native_phase_count: int
    cold_compiler_probe_count: int
    warm_native_phase_count: int
    warm_compiler_probe_count: int
    final_speedup: float | None
    source_speedup: float | None
    wheel_speedup: float | None
    cold_source: dict[str, object]
    warm_source: dict[str, object]
    cold_accepted_trials: tuple[dict[str, object], ...]
    warm_accepted_trials: tuple[dict[str, object], ...]


def _evaluation_errors(inputs: _EvaluationInputs) -> tuple[str, ...]:
    return (
        *_report_errors(inputs),
        *_cache_errors(inputs),
        *_source_errors(inputs),
        *_source_optimization_errors(inputs),
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
        performance_status = _string_field(_mapping_field(report, "performance"), "status")
        if performance_status != "passed":
            errors.append(f"{label} performance status is {performance_status}, expected passed")
        source = _source_optimization(report)
        source_status = _string_field(source, "status")
        if source_status != "accepted":
            errors.append(
                f"{label} source-optimization status is {source_status}, expected accepted"
            )
        minimum_speedup = _number_field(source, "minimum_speedup")
        if minimum_speedup < MINIMUM_SOURCE_SPEEDUP:
            errors.append(
                f"{label} source-optimization floor is {minimum_speedup:.3f}x, "
                f"below {MINIMUM_SOURCE_SPEEDUP:.3f}x"
            )
    return tuple(errors)


def _cache_errors(inputs: _EvaluationInputs) -> tuple[str, ...]:
    errors: list[str] = []
    cold_patch_cache = _source_patch_cache_status(inputs.cold_accepted_trials)
    warm_patch_cache = _source_patch_cache_status(inputs.warm_accepted_trials)
    if cold_patch_cache != "miss":
        errors.append(f"cold source-patch cache status is {cold_patch_cache}, expected miss")
    if warm_patch_cache != "hit":
        errors.append(f"warm source-patch cache status is {warm_patch_cache}, expected hit")
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
    errors.extend(_source_plan_identity_errors("cold", inputs.cold_report))
    errors.extend(_source_plan_identity_errors("warm", inputs.warm_report))
    if _source_plan_identities(inputs.cold_report) != _source_plan_identities(inputs.warm_report):
        errors.append("source-plan identities or source hashes changed between runs")
    if _accepted_source_trial_identities(inputs.cold_report) != _accepted_source_trial_identities(
        inputs.warm_report
    ):
        errors.append("accepted source candidate or patch identity changed between runs")
    cold_patch = _reported_patch_path(inputs.cold_report)
    warm_patch = _reported_patch_path(inputs.warm_report)
    if cold_patch != warm_patch:
        errors.append("reported source patch path changed between cold and warm runs")
    if warm_patch is not None and not _is_atoll_patch_path(warm_patch):
        errors.append(f"source patch is outside .atoll/patches: {warm_patch}")
    for label, source in (("cold", inputs.cold_source), ("warm", inputs.warm_source)):
        application_status = _string_field(source, "application_status")
        if application_status != "not-applied":
            errors.append(
                f"{label} source patch application status is {application_status}, "
                "expected not-applied"
            )
    return tuple(errors)


def _source_plan_identity_errors(
    label: str,
    report: dict[str, object],
) -> tuple[str, ...]:
    errors: list[str] = []
    for plan in _source_optimization_plans(report):
        plan_id = _string_field(plan, "id")
        identity = _mapping_field(plan, "identity")
        source_hashes = identity.get("source_hashes")
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
            errors.append(f"{label} source plan {plan_id} has no valid per-file source hashes")
    return tuple(errors)


def _source_optimization_errors(inputs: _EvaluationInputs) -> tuple[str, ...]:
    return (
        *_source_optimization_report_errors("cold", inputs.cold_report),
        *_source_optimization_report_errors("warm", inputs.warm_report),
    )


def _source_optimization_report_errors(
    label: str,
    report: dict[str, object],
) -> tuple[str, ...]:
    errors: list[str] = []
    plans = _source_optimization_plans(report)
    assessments = _source_optimization_assessments(report)
    accepted = _accepted_source_optimization_trials(report)
    if not plans:
        errors.append(f"{label} report contains no source-optimization plan")
    if not any(_string_field(assessment, "status") == "trial-ready" for assessment in assessments):
        errors.append(f"{label} report contains no trial-ready source plan")
    if len(accepted) != 1:
        errors.append(f"{label} report contains {len(accepted)} accepted source trials, expected 1")
        return tuple(errors)
    trial = accepted[0]
    errors.extend(_accepted_source_trial_errors(label, report, trial))
    errors.extend(_source_assessment_errors(label, assessments, trial))
    baseline_samples, compiled_samples = _performance_sample_counts(report)
    if baseline_samples < BENCHMARK_SAMPLES or compiled_samples < BENCHMARK_SAMPLES:
        errors.append(
            f"{label} profitability gate did not rerun {BENCHMARK_SAMPLES} "
            "baseline/compiled timing pairs"
        )
    return tuple(errors)


def _accepted_source_trial_errors(
    label: str,
    report: dict[str, object],
    trial: dict[str, object],
) -> tuple[str, ...]:
    errors: list[str] = []
    patch_path = trial.get("patch_path")
    if not isinstance(patch_path, str) or not _is_atoll_patch_path(patch_path):
        errors.append(f"{label} accepted source trial has no safe .atoll patch path")
    if patch_path != _reported_patch_path(report):
        errors.append(f"{label} accepted trial patch differs from aggregate patch path")
    if not _optional_list_field(trial, "source_edits"):
        errors.append(f"{label} accepted source trial contains no source edits")
    if trial.get("semantic_exit_code") != 0:
        errors.append(f"{label} accepted source trial did not pass semantic tests")
    transformation_kinds = {
        _transformation_kind(value)
        for value in _optional_list_field(trial, "transformation_ids")
        if isinstance(value, str)
    }
    missing = REQUIRED_SOURCE_TRANSFORMATION_KINDS - transformation_kinds
    if missing:
        errors.append(
            f"{label} accepted source trial is missing transformation(s): "
            f"{', '.join(sorted(missing))}"
        )
    return tuple(errors)


def _source_assessment_errors(
    label: str,
    assessments: tuple[dict[str, object], ...],
    trial: dict[str, object],
) -> tuple[str, ...]:
    plan_id = _string_field(trial, "plan_id")
    assessment = next(
        (
            item
            for item in assessments
            if _string_field(item, "plan_id") == plan_id
            and _string_field(item, "status") == "trial-ready"
        ),
        None,
    )
    if assessment is None:
        return (f"{label} accepted source trial has no trial-ready assessment",)
    errors: list[str] = []
    if _integer_field(assessment, "observed_work_items") < MINIMUM_OBSERVED_WORK_ITEMS:
        errors.append(f"{label} source plan observed fewer than 10,000 work items")
    if _number_field(assessment, "attributed_hot_share") < MINIMUM_ATTRIBUTED_HOT_SHARE:
        errors.append(f"{label} source plan covers less than 70% of the hot path")
    return tuple(errors)


def _profitability_errors(inputs: _EvaluationInputs) -> tuple[str, ...]:
    errors: list[str] = []
    if not inputs.wheel_present:
        errors.append("warm compile did not leave a promoted wheel")
    if not inputs.patch_present:
        errors.append("warm compile did not leave the accepted source patch")
    for label, trials in (
        ("cold", inputs.cold_accepted_trials),
        ("warm", inputs.warm_accepted_trials),
    ):
        for trial in trials:
            source_speedup = _optional_number_field(trial, "source_speedup")
            wheel_speedup = _optional_number_field(trial, "wheel_speedup")
            if source_speedup is None or source_speedup < MINIMUM_SOURCE_SPEEDUP:
                errors.append(f"{label} transformed source speedup is below 3.00x")
            if wheel_speedup is None or wheel_speedup < MINIMUM_FINAL_SPEEDUP:
                errors.append(f"{label} normal wheel speedup is below 3.00x")
    if inputs.final_speedup is None or inputs.final_speedup < MINIMUM_FINAL_SPEEDUP:
        errors.append("warm final speedup is below 3.00x")
    if inputs.source_speedup is None or inputs.source_speedup < MINIMUM_SOURCE_SPEEDUP:
        errors.append("warm accepted source speedup is below 3.00x")
    if inputs.wheel_speedup is None or inputs.wheel_speedup < MINIMUM_FINAL_SPEEDUP:
        errors.append("warm accepted wheel speedup is below 3.00x")
    return tuple(errors)


def source_manifest(source_root: Path) -> dict[str, str]:
    """Hash every Python source file without importing or rewriting the target."""
    return {
        path.relative_to(source_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source_root.rglob("*.py"))
    }


def _phase_count(report: dict[str, object], name: str) -> int:
    return sum(_string_field(timing, "name") == name for timing in _phase_timings(report))


def _phase_timings(report: dict[str, object]) -> tuple[dict[str, object], ...]:
    build = _mapping_field(report, "build")
    return tuple(
        _mapping(item, "build.phase_timings[]") for item in _list_field(build, "phase_timings")
    )


def _source_optimization(report: dict[str, object]) -> dict[str, object]:
    return _mapping_field(report, "source_optimization")


def _source_optimization_plans(
    report: dict[str, object],
) -> tuple[dict[str, object], ...]:
    return tuple(
        _mapping(item, "source_optimization.plans[]")
        for item in _optional_list_field(_source_optimization(report), "plans")
    )


def _source_optimization_assessments(
    report: dict[str, object],
) -> tuple[dict[str, object], ...]:
    return tuple(
        _mapping(item, "source_optimization.assessments[]")
        for item in _optional_list_field(_source_optimization(report), "assessments")
    )


def _source_optimization_trials(
    report: dict[str, object],
) -> tuple[dict[str, object], ...]:
    return tuple(
        _mapping(item, "source_optimization.trials[]")
        for item in _optional_list_field(_source_optimization(report), "trials")
    )


def _accepted_source_optimization_trials(
    report: dict[str, object],
) -> tuple[dict[str, object], ...]:
    return tuple(
        trial
        for trial in _source_optimization_trials(report)
        if _string_field(trial, "status") == "accepted"
    )


def _final_accepted_source_trial(
    report: dict[str, object],
) -> dict[str, object] | None:
    patch_path = _reported_patch_path(report)
    return next(
        (
            trial
            for trial in _accepted_source_optimization_trials(report)
            if trial.get("patch_path") == patch_path
        ),
        None,
    )


def _source_plan_identities(report: dict[str, object]) -> dict[str, str]:
    return {
        _string_field(plan, "id"): json.dumps(
            _mapping_field(plan, "identity"),
            sort_keys=True,
            separators=(",", ":"),
        )
        for plan in _source_optimization_plans(report)
    }


def _accepted_source_trial_identities(
    report: dict[str, object],
) -> tuple[tuple[str, str, str, tuple[str, ...]], ...]:
    identities = (
        (
            _string_field(trial, "plan_id"),
            _string_field(trial, "candidate_id"),
            _string_field(trial, "patch_path"),
            tuple(
                value
                if isinstance(value, str)
                else _raise_field_error("source_optimization.trials[].transformation_ids[]")
                for value in _optional_list_field(trial, "transformation_ids")
            ),
        )
        for trial in _accepted_source_optimization_trials(report)
    )
    return tuple(sorted(identities))


def _source_patch_cache_status(trials: tuple[dict[str, object], ...]) -> str:
    statuses: set[str] = set()
    for trial in trials:
        diagnostics = _optional_list_field(trial, "diagnostics")
        statuses.update(
            value.removeprefix("cache ")
            for value in diagnostics
            if isinstance(value, str) and value in {"cache hit", "cache miss"}
        )
    if len(statuses) == 1:
        return next(iter(statuses))
    return "mixed" if statuses else "unknown"


def _reported_patch_path(report: dict[str, object]) -> str | None:
    patch_path = _source_optimization(report).get("patch_path")
    if patch_path is None:
        return None
    if not isinstance(patch_path, str):
        _raise_field_error("source_optimization.patch_path")
    return cast(str, patch_path)


def _is_atoll_patch_path(path: str) -> bool:
    candidate = PurePosixPath(path)
    return (
        not candidate.is_absolute()
        and len(candidate.parts) == ATOLL_PATCH_PATH_PARTS
        and candidate.parts[:2] == (".atoll", "patches")
        and candidate.suffix == ".patch"
        and ".." not in candidate.parts
    )


def _performance_sample_counts(report: dict[str, object]) -> tuple[int, int]:
    samples = _optional_list_field(_mapping_field(report, "performance"), "samples")
    modes = tuple(
        _string_field(_mapping(sample, "performance.samples[]"), "mode") for sample in samples
    )
    return modes.count("baseline"), modes.count("compiled")


def _transformation_kind(transformation_id: str) -> str:
    return transformation_id.partition(":")[0]


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
