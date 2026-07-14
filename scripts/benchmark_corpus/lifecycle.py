"""Isolated checkout, compile, oracle, and evidence lifecycle for one case.

The lifecycle treats upstream code as untrusted input.  It clones an immutable
revision, qualifies the normal project wheel, appends policy only in the clone,
runs cold and warm source-clean compiles, verifies installed-wheel routing, and
removes the disposable workspace.  Every failure is converted into a schema-v1
case result before cleanup.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import stat
import sys
import sysconfig
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Literal, cast

from scripts.benchmark_corpus.identity import case_digest, comparison_key
from scripts.benchmark_corpus.manifest import ManifestError
from scripts.benchmark_corpus.models import (
    CaseResult,
    CaseStatus,
    CompilePolicy,
    CorpusBackend,
    CorpusCase,
    CorpusManifest,
    CorpusPlatform,
    CorpusTier,
    EnvironmentEvidence,
    PhaseEvidence,
    PolicyEvidence,
    RatioEvidence,
)
from scripts.benchmark_corpus.policy import append_compile_policy
from scripts.benchmark_corpus.process import (
    ProcessLimits,
    ProcessRequest,
    ProcessResult,
    Sandbox,
    detect_sandbox,
    run_process,
    sanitized_environment,
)
from scripts.benchmark_corpus.results import classify_compile_report, write_case_result
from scripts.benchmark_corpus.security import (
    CheckoutSecurityError,
    TrackedSourceManifest,
    tracked_source_manifest,
    validate_checkout,
)

EnvironmentMode = Literal["isolated", "current-test"]
PerformanceMode = Literal["baseline", "compiled"]
CASE_RESULT_SCHEMA_VERSION = 1
_NATIVE_PHASES = frozenset({"mypycify", "cythonize", "build_ext"})
_COMPILE_REPORT_SCHEMA_VERSION = 6
_COMPILER_WRAPPER_NAME = "compiler-probe.py"
_BOOTSTRAP_ATTEMPTS = 2
_BENCHMARK_WARMUPS = 1
_BENCHMARK_SAMPLES = 7


@dataclass(frozen=True, slots=True)
class LifecycleOptions:
    """Filesystem and isolation choices for one corpus case invocation.

    ``repository_mirror``, ``adapter_root``, ``sandbox_override``, and
    ``current-test`` mode exist for deterministic local integration tests.  The
    CLI never exposes those bypasses; external runs use the pinned repository,
    reviewed adapter directory, isolated environments, and detected sandbox.

    Attributes:
        atoll_root: Atoll repository containing locks and reviewed adapters.
        workspace_root: Parent receiving a disposable case workspace.
        evidence_root: Persistent case-specific evidence directory.
        tier: Manifest tier being executed.
        platform: Workflow runner label.
        allow_unsandboxed: Explicit opt-in to use the disposable host itself as
            the isolation boundary instead of a platform sandbox launcher.
        keep_workspace: Debug-only retention of the disposable workspace.
        repository_mirror: Test-only local Git remote replacing the manifest URL.
        adapter_root: Test-only reviewed adapter directory override.
        sandbox_override: Test-only explicit sandbox selection.
        environment_mode: Isolated production environments or current-process test mode.
    """

    atoll_root: Path
    workspace_root: Path
    evidence_root: Path
    tier: CorpusTier
    platform: CorpusPlatform
    allow_unsandboxed: bool = False
    keep_workspace: bool = False
    repository_mirror: Path | None = None
    adapter_root: Path | None = None
    sandbox_override: Sandbox | None = None
    environment_mode: EnvironmentMode = "isolated"


@dataclass(frozen=True, slots=True)
class CaseRunSummary:
    """Completed result plus the two canonical report paths.

    Attributes:
        result: Immutable schema-v1 case evidence.
        json_path: Authoritative case-result JSON path.
        markdown_path: Human-readable rendering derived from JSON evidence.
    """

    result: CaseResult
    json_path: Path
    markdown_path: Path


class LifecycleError(RuntimeError):
    """Expected case failure with explicit aggregate attribution.

    Attributes:
        status: Corpus status assigned to this failure.
        diagnostic: Stable maintainer-facing explanation.
    """

    status: CaseStatus
    diagnostic: str

    def __init__(self, status: CaseStatus, diagnostic: str) -> None:
        """Create a classified lifecycle failure.

        Args:
            status: Result category retained by aggregation.
            diagnostic: Explanation written to case evidence.
        """
        self.status = status
        self.diagnostic = diagnostic
        super().__init__(diagnostic)


@dataclass(frozen=True, slots=True)
class _Paths:
    workspace: Path
    checkout: Path
    project: Path
    evidence: Path
    logs: Path
    home: Path
    temporary: Path
    wheelhouse: Path
    tools_environment: Path


@dataclass(frozen=True, slots=True)
class _Arm:
    python: Path
    import_root: Path
    environment: dict[str, str]


@dataclass(slots=True)
class _State:
    phases: list[PhaseEvidence] = field(default_factory=list[PhaseEvidence])
    diagnostics: list[str] = field(default_factory=list[str])
    status: CaseStatus = "infrastructure-error"
    policy: PolicyEvidence | None = None
    environment: EnvironmentEvidence | None = None
    source_before: TrackedSourceManifest | None = None
    source_after: TrackedSourceManifest | None = None
    baseline_wheel_digest: str | None = None
    compiled_wheel_digest: str | None = None
    baseline_oracle_digest: str | None = None
    compiled_oracle_digest: str | None = None
    cold_report_path: PurePosixPath | None = None
    warm_report_path: PurePosixPath | None = None
    cold_compiler_invocations: int | None = None
    warm_compiler_invocations: int | None = None
    comparison_key: str | None = None
    ratios: RatioEvidence = field(default_factory=RatioEvidence)


@dataclass(frozen=True, slots=True)
class _RunContext:
    manifest: CorpusManifest
    case: CorpusCase
    options: LifecycleOptions
    paths: _Paths
    sandbox: Sandbox
    state: _State


@dataclass(frozen=True, slots=True)
class _CompileOutcome:
    """Cold/warm compile agreement and any promoted wheel.

    Attributes:
        report: Warm schema-v6 report retained as the canonical compile evidence.
        status: Corpus classification agreed by cold and warm compiles.
        wheel: Promoted wheel, baseline wheel for a clean no-op, or ``None``
            when profitability or stability prevented wheel promotion.
    """

    report: dict[str, object]
    status: CaseStatus
    wheel: Path | None


@dataclass(frozen=True, slots=True)
class _PhaseRequest:
    name: str
    argv: tuple[str, ...]
    cwd: Path
    environment: dict[str, str]
    timeout_seconds: float
    network_allowed: bool = False


@dataclass(frozen=True, slots=True)
class DependencyBootstrapCommands:
    """Hash-enforcing commands used after the isolated tools venv exists.

    Attributes:
        ensure_pip: Offline standard-library bootstrap for the venv-local pip.
        download: Network-enabled wheelhouse population from the reviewed lock.
        sync: Offline tools-environment synchronization from that wheelhouse.
    """

    ensure_pip: tuple[str, ...]
    download: tuple[str, ...]
    sync: tuple[str, ...]


def run_case(
    manifest: CorpusManifest,
    case_id: str,
    options: LifecycleOptions,
) -> CaseRunSummary:
    """Execute one pinned case and always write a schema-v1 result.

    Args:
        manifest: Validated corpus manifest.
        case_id: Exact case selected from the manifest.
        options: Workspace, evidence, tier, platform, and isolation choices.

    Returns:
        CaseRunSummary: Classified evidence and canonical report paths.

    Raises:
        ManifestError: If case selection or caller-owned paths are invalid before
            a safe case-specific evidence directory can be established.
    """
    case = _select_case(manifest, case_id, options.tier, options.platform)
    paths = _prepare_paths(case, options)
    state = _State()
    manifest_digest = _sha256_file(manifest.path)
    case_identity = case_digest(case)
    try:
        sandbox = options.sandbox_override or detect_sandbox(options.allow_unsandboxed)
        _execute_case(
            _RunContext(
                manifest=manifest,
                case=case,
                options=options,
                paths=paths,
                sandbox=sandbox,
                state=state,
            )
        )
    except CheckoutSecurityError as error:
        state.status = "security-violation"
        state.diagnostics.extend(f"{finding.code}: {finding.message}" for finding in error.findings)
    except LifecycleError as error:
        state.status = error.status
        state.diagnostics.append(error.diagnostic)
    except (ManifestError, OSError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
        state.status = "infrastructure-error"
        state.diagnostics.append(str(error))
    finally:
        _capture_final_source(paths, state)
        if _source_changed(state):
            state.status = "compatibility-regression"
            state.diagnostics.append("Atoll changed tracked checkout files after policy injection")
        result = _case_result(case, options, state, manifest_digest, case_identity)
        json_path, markdown_path = write_case_result(result, paths.evidence)
        if not options.keep_workspace:
            shutil.rmtree(paths.workspace, ignore_errors=True)
    return CaseRunSummary(result=result, json_path=json_path, markdown_path=markdown_path)


def _execute_case(context: _RunContext) -> None:
    case = context.case
    options = context.options
    paths = context.paths
    state = context.state
    online_environment = _environment(paths, offline=False)
    offline_environment = _environment(paths, offline=True)
    if options.environment_mode == "current-test":
        current_paths = tuple(path for path in sys.path if path and Path(path).exists())
        python_path = os.pathsep.join(current_paths)
        online_environment["PYTHONPATH"] = python_path
        offline_environment["PYTHONPATH"] = python_path
    repository = str(options.repository_mirror or case.repository)
    _clone_checkout(context, repository, online_environment)
    validation = validate_checkout(paths.checkout, case.revision, case.project_subroot)
    _write_source_manifest(
        validation.source_manifest,
        paths.evidence / "source-manifest-pristine.json",
    )
    lock_path = _dependency_lock(options.atoll_root, case)
    lock_digest = _sha256_file(lock_path)
    tools_python = _prepare_tools_environment(
        context, online_environment, offline_environment, lock_path
    )
    compiler_environment, compiler_log = _compiler_probe_environment(
        tools_python,
        paths,
        offline_environment,
    )
    state.environment = _probe_environment(context, tools_python, compiler_environment, lock_digest)
    baseline_wheel = _build_baseline_wheel(context, tools_python, compiler_environment)
    state.baseline_wheel_digest = _sha256_file(baseline_wheel)
    _require_pristine_source(
        paths,
        validation.source_manifest,
        "normal PEP 517 baseline wheel build",
    )
    baseline_arm = _prepare_arm(
        context, "baseline", baseline_wheel, compiler_environment, lock_path
    )
    _run_focused_tests(context, baseline_arm)
    _require_pristine_source(paths, validation.source_manifest, "focused upstream tests")
    state.baseline_oracle_digest = _run_oracle(context, "baseline-oracle", baseline_arm)
    _require_pristine_source(paths, validation.source_manifest, "baseline oracle")
    _clean_generated_checkout(context, offline_environment)
    cleaned = validate_checkout(paths.checkout, case.revision, case.project_subroot)
    if cleaned.source_manifest != validation.source_manifest:
        raise LifecycleError(
            "upstream-broken",
            "disposable checkout did not return to its pinned source identity",
        )
    _validate_workload_assets(context)
    state.policy = append_compile_policy(
        paths.project / "pyproject.toml",
        _compile_policy(context, tools_python),
        paths.evidence,
        paths.checkout,
    )
    state.comparison_key = comparison_key(case, state.environment, state.policy)
    state.source_before = tracked_source_manifest(paths.checkout)
    _write_source_manifest(state.source_before, paths.evidence / "source-manifest-before.json")
    outcome = _compile_cold_and_warm(
        context,
        tools_python,
        compiler_environment,
        compiler_log,
        baseline_wheel,
    )
    state.ratios = ratio_evidence_from_report(outcome.report)
    if outcome.wheel is not None:
        _verify_compiled_wheel(
            context,
            outcome.wheel,
            compiler_environment,
            lock_path,
            outcome.report,
        )
    state.status = outcome.status


def _compile_policy(context: _RunContext, tools_python: Path) -> CompilePolicy:
    """Build the disposable static or measured policy for one corpus tier.

    Performance commands use the isolated case interpreter and reviewed Atoll
    adapters by absolute path. The target checkout is supplied only as input;
    imports are routed by Atoll to each baseline or candidate wheel payload.

    Args:
        context: Selected manifest case and trusted repository paths.
        tools_python: Isolated interpreter containing locked case dependencies.

    Returns:
        CompilePolicy: Backend-only compatibility policy or fully configured
            semantic and benchmark commands for a performance case.
    """
    if context.options.tier != "performance":
        return CompilePolicy(backends=context.manifest.backends)
    semantic_adapter = _reviewed_adapter_path(context, context.case.oracle_adapter)
    performance_adapter = _reviewed_adapter_path(
        context,
        context.case.id.replace("-", "_"),
    )
    return build_performance_compile_policy(
        backends=context.manifest.backends,
        tools_python=tools_python,
        adapters=(semantic_adapter, performance_adapter),
        project_root=context.paths.project,
        oracle_arguments=context.case.oracle_arguments,
    )


def build_performance_compile_policy(
    *,
    backends: tuple[CorpusBackend, ...],
    tools_python: Path,
    adapters: tuple[Path, Path],
    project_root: Path,
    oracle_arguments: tuple[str, ...],
) -> CompilePolicy:
    """Construct the exact measured policy after trusted paths are resolved.

    Args:
        backends: Manifest backend preference order.
        tools_python: Isolated interpreter containing target dependencies.
        adapters: Reviewed semantic and performance command sources, in that order.
        project_root: Disposable target checkout root.
        oracle_arguments: Case-specific semantic adapter arguments.

    Returns:
        CompilePolicy: One warmup, seven measured pairs, and the default 1.10 gate.
    """
    semantic_adapter, performance_adapter = adapters
    return CompilePolicy(
        backends=backends,
        test_command=(
            str(tools_python),
            str(semantic_adapter),
            "--project-root",
            str(project_root),
            *oracle_arguments,
        ),
        benchmark_command=(
            str(tools_python),
            str(performance_adapter),
            "--project-root",
            str(project_root),
        ),
        benchmark_warmups=_BENCHMARK_WARMUPS,
        benchmark_samples=_BENCHMARK_SAMPLES,
    )


def _validate_workload_assets(context: _RunContext) -> None:
    """Enforce manifest workload identity before external performance runs.

    The case-local workload digest is manifest data. Shared runner and golden
    files are tied to the recorded Atoll revision, but are still required to be
    regular repository files so a local symlink cannot redirect execution.

    Args:
        context: Selected case and trusted Atoll repository root.

    Raises:
        LifecycleError: If a performance workload, notice, adapter, shared
            runner, or golden oracle is missing, redirected, or digest-mismatched.
    """
    if context.options.tier != "performance":
        return
    adapter_root = context.options.adapter_root or (
        context.options.atoll_root / "benchmarks" / "corpus" / "adapters"
    )
    validate_performance_assets(
        context.options.atoll_root,
        context.case,
        adapter_root,
    )


def validate_performance_assets(
    atoll_root: Path,
    case: CorpusCase,
    adapter_root: Path,
) -> None:
    """Validate one performance case's reviewed workload and harness files.

    Args:
        atoll_root: Repository root containing manifest-relative workload assets.
        case: Performance case whose digest and stable ID select the workload.
        adapter_root: Reviewed adapter directory used by the lifecycle.

    Raises:
        LifecycleError: If any required asset is unavailable or fails identity checks.
    """
    workload = case.workload
    if workload is None:
        raise LifecycleError("infrastructure-error", "performance case has no workload identity")
    observed_digest = performance_asset_digest(atoll_root, case, adapter_root)
    if observed_digest != workload.sha256:
        raise LifecycleError(
            "security-violation",
            f"manifest workload bundle digest does not match {workload.path}",
        )


def performance_asset_digest(atoll_root: Path, case: CorpusCase, adapter_root: Path) -> str:
    """Digest every reviewed file that defines one performance measurement.

    Args:
        atoll_root: Repository root containing workload and notice paths.
        case: Performance case selecting the case-specific workload and adapter.
        adapter_root: Reviewed adapter directory used by execution.

    Returns:
        str: Stable SHA-256 over logical path labels and exact file bytes.

    Raises:
        LifecycleError: If the case has no workload or any asset is unsafe.
    """
    workload = case.workload
    if workload is None:
        raise LifecycleError("infrastructure-error", "performance case has no workload identity")
    root = atoll_root.resolve(strict=True)
    workload_path = _trusted_atoll_asset(root, workload.path, "workload")
    expected = root / "benchmarks" / "corpus" / "workloads" / (f"{case.id.replace('-', '_')}.py")
    if workload_path != expected.resolve(strict=True):
        raise LifecycleError(
            "security-violation",
            f"manifest workload path does not match case {case.id}",
        )
    notice = _trusted_atoll_asset(root, workload.notice, "workload notice")
    if not notice.read_text(encoding="utf-8").strip():
        raise LifecycleError("security-violation", f"workload notice is empty: {workload.notice}")
    adapters = adapter_root.resolve(strict=True)
    adapter = _trusted_atoll_asset(
        adapters,
        PurePosixPath(f"{case.id.replace('-', '_')}.py"),
        "performance adapter",
    )
    shared = _trusted_atoll_asset(
        root,
        PurePosixPath("benchmarks/corpus/adapters/_performance.py"),
        "performance harness",
    )
    golden = _trusted_atoll_asset(
        root,
        PurePosixPath("benchmarks/corpus/workloads/golden.json"),
        "performance harness",
    )
    return _asset_bundle_digest(
        (
            (f"adapters/{adapter.name}", adapter),
            ("adapters/_performance.py", shared),
            ("workloads/golden.json", golden),
            (f"workloads/{workload_path.name}", workload_path),
            (f"notices/{notice.name}", notice),
        )
    )


def _asset_bundle_digest(assets: tuple[tuple[str, Path], ...]) -> str:
    """Hash labeled file boundaries so concatenation cannot create collisions."""
    digest = hashlib.sha256()
    for label, path in sorted(assets):
        contents = path.read_bytes()
        digest.update(label.encode())
        digest.update(b"\0")
        digest.update(str(len(contents)).encode())
        digest.update(b"\0")
        digest.update(contents)
    return digest.hexdigest()


def _trusted_atoll_asset(root: Path, relative: PurePosixPath, label: str) -> Path:
    """Resolve one regular repository asset without following a final symlink."""
    candidate = root.joinpath(*relative.parts)
    if candidate.is_symlink():
        raise LifecycleError("security-violation", f"{label} is a symlink: {relative}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise LifecycleError(
            "infrastructure-error",
            f"{label} is unavailable: {relative}: {error}",
        ) from error
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise LifecycleError("security-violation", f"{label} escapes Atoll: {relative}")
    return resolved


def _compile_timeout_seconds(context: _RunContext) -> int:
    """Return the case override or tier default for each Atoll invocation."""
    case = context.case
    defaults = context.manifest.defaults
    if context.options.tier == "performance":
        return case.performance_timeout_seconds or defaults.performance_timeout_seconds
    return case.compile_timeout_seconds or defaults.compile_timeout_seconds


def classify_compile_process(
    result: ProcessResult,
    report: dict[str, object],
    tier: CorpusTier,
    label: str,
    project_root: Path,
) -> CaseStatus:
    """Normalize one compile process and report without hiding rejections.

    Args:
        result: Bounded process outcome for one cold or warm invocation.
        report: Copied schema-v6 report emitted by that invocation.
        tier: Corpus role determining whether performance evidence is required.
        label: ``cold`` or ``warm`` for diagnostics.
        project_root: Disposable target root used to resolve report-relative payloads.

    Returns:
        CaseStatus: Successful, no-op, unprofitable, or unstable outcome.

    Raises:
        LifecycleError: If the process timed out, the report is malformed, a
            semantic gate regressed, or a non-performance compile failed.
    """
    _validate_compile_report(report)
    if result.timed_out:
        raise LifecycleError("timeout", f"{label} Atoll compile exceeded its timeout")
    if _is_clean_noop_report(report):
        _require_rejected_compile_process(result, report, label)
        return "supported-no-op"
    if tier != "performance":
        _require_success(result, "compile-error", f"{label} Atoll compile")
        _require_successful_compile_report(report, label)
        return classify_compile_report(report, tier)

    performance = _mapping_field(report, "performance")
    performance_status = performance.get("status")
    if performance_status == "passed":
        _require_success(result, "compile-error", f"{label} Atoll compile")
        _require_successful_compile_report(report, label)
        _validate_performance_summary(performance, "passed")
        _validate_performance_samples(report, project_root)
        return "accelerated"
    if performance_status == "not-profitable":
        _require_rejected_compile_process(result, report, label)
        _validate_performance_summary(performance, "not-profitable")
        _validate_performance_samples(report, project_root)
        return "not-profitable"
    if performance_status == "invalid":
        _require_rejected_compile_process(result, report, label)
        reason = performance.get("reason")
        reason_text = reason if isinstance(reason, str) else "invalid performance gate"
        if "compiled semantic test command failed" in reason_text:
            raise LifecycleError("compatibility-regression", reason_text)
        if "baseline semantic test command failed" in reason_text:
            raise LifecycleError("upstream-broken", reason_text)
        if "too noisy" in reason_text:
            _validate_performance_summary(performance, "invalid")
            _validate_performance_samples(report, project_root)
            return "unstable"
        raise LifecycleError("compile-error", f"{label} performance gate: {reason_text}")
    _require_success(result, "compile-error", f"{label} Atoll compile")
    raise LifecycleError(
        "compile-error",
        f"{label} performance compile reported unexpected status {performance_status!r}",
    )


def _agreed_compile_status(cold: CaseStatus, warm: CaseStatus) -> CaseStatus:
    """Require deterministic compile shape while tolerating timing variance."""
    if cold == warm:
        return warm
    measured = {"accelerated", "not-profitable", "unstable"}
    if cold in measured and warm in measured:
        return "unstable"
    raise LifecycleError(
        "compile-error",
        f"cold and warm compile classifications differed: {cold} versus {warm}",
    )


def _compiled_wheel_for_status(
    paths: _Paths,
    baseline_wheel: Path,
    status: CaseStatus,
) -> Path | None:
    """Locate only wheels that the status contract permits corpus verification."""
    if status == "supported-no-op":
        return baseline_wheel
    if status in {"accelerated", "compiled-unbenchmarked"}:
        return _single_wheel(paths.project / ".atoll" / "dist")
    return None


def _compile_cold_and_warm(
    context: _RunContext,
    tools_python: Path,
    compiler_environment: dict[str, str],
    compiler_log: Path,
    baseline_wheel: Path,
) -> _CompileOutcome:
    """Run identical cold and warm compiles and prove native cache reuse.

    A schema-backed no-op, profitability rejection, or noisy benchmark is a
    completed corpus outcome even though ``atoll compile`` intentionally exits
    nonzero and does not publish a wheel. Compiler, semantic, and malformed
    report failures still abort with their precise lifecycle classification.
    """
    options = context.options
    paths = context.paths
    state = context.state
    if (paths.project / ".atoll").exists():
        raise LifecycleError("infrastructure-error", "cold compile cache was not empty")
    compile_command = (
        str(tools_python),
        "-m",
        "atoll",
        "compile",
        "--root",
        str(paths.project),
    )
    compile_timeout = _compile_timeout_seconds(context)
    _remove_compile_reports(paths)
    cold = _phase(
        context,
        _PhaseRequest(
            name="compile-cold",
            argv=compile_command,
            cwd=paths.project,
            environment=compiler_environment,
            timeout_seconds=compile_timeout,
        ),
    )
    _retain_compiler_probe(paths, compiler_log)
    state.cold_report_path = _copy_compile_report(paths, "cold")
    cold_report = _read_json_object(paths.evidence / "cold.compile-report.json")
    cold_status = classify_compile_process(
        cold,
        cold_report,
        options.tier,
        "cold",
        paths.project,
    )
    state.cold_compiler_invocations = _line_count(compiler_log)
    _remove_compile_reports(paths)
    warm = _phase(
        context,
        _PhaseRequest(
            name="compile-warm",
            argv=compile_command,
            cwd=paths.project,
            environment=compiler_environment,
            timeout_seconds=compile_timeout,
        ),
    )
    _retain_compiler_probe(paths, compiler_log)
    state.warm_report_path = _copy_compile_report(paths, "warm")
    warm_report = _read_json_object(paths.evidence / "warm.compile-report.json")
    warm_status = classify_compile_process(
        warm,
        warm_report,
        options.tier,
        "warm",
        paths.project,
    )
    state.warm_compiler_invocations = _line_count(compiler_log) - state.cold_compiler_invocations
    if state.warm_compiler_invocations:
        raise LifecycleError(
            "compatibility-regression",
            f"warm compile invoked native compiler {state.warm_compiler_invocations} time(s)",
        )
    _validate_warm_report(warm_report)
    state.source_after = tracked_source_manifest(paths.checkout)
    _write_source_manifest(state.source_after, paths.evidence / "source-manifest-after.json")
    status = _agreed_compile_status(cold_status, warm_status)
    if status == "unstable" and cold_status != warm_status:
        state.diagnostics.append(
            f"cold and warm performance outcomes differed: {cold_status} versus {warm_status}"
        )
    wheel = _compiled_wheel_for_status(paths, baseline_wheel, status)
    if wheel is not None:
        state.compiled_wheel_digest = _sha256_file(wheel)
    return _CompileOutcome(report=warm_report, status=status, wheel=wheel)


def _verify_compiled_wheel(
    context: _RunContext,
    compiled_wheel: Path,
    compiler_environment: dict[str, str],
    lock_path: Path,
    warm_report: dict[str, object],
) -> None:
    """Install the final wheel and compare normal and strict routing oracles."""
    state = context.state
    compiled_arm = _prepare_arm(
        context, "compiled", compiled_wheel, compiler_environment, lock_path
    )
    state.compiled_oracle_digest = _run_oracle(context, "compiled-oracle", compiled_arm)
    if state.compiled_oracle_digest != state.baseline_oracle_digest:
        raise LifecycleError(
            "compatibility-regression",
            "baseline and final-wheel canonical oracle outputs differ",
        )
    strict_environment = _strict_routing_environment(warm_report)
    if strict_environment:
        strict_arm = _Arm(
            python=compiled_arm.python,
            import_root=compiled_arm.import_root,
            environment={**compiled_arm.environment, **strict_environment},
        )
        strict_digest = _run_oracle(context, "compiled-oracle-strict", strict_arm)
        if strict_digest != state.baseline_oracle_digest:
            raise LifecycleError(
                "compatibility-regression",
                "strict compiled routing changed canonical oracle output",
            )


def _select_case(
    manifest: CorpusManifest,
    case_id: str,
    tier: CorpusTier,
    platform_name: CorpusPlatform,
) -> CorpusCase:
    matches = tuple(case for case in manifest.cases if case.id == case_id)
    if len(matches) != 1:
        raise ManifestError(f"manifest contains no unique case {case_id!r}")
    case = matches[0]
    if tier not in case.tiers:
        raise ManifestError(f"case {case_id} is not configured for tier {tier}")
    if platform_name not in case.platforms:
        raise ManifestError(f"case {case_id} is not configured for platform {platform_name}")
    return case


def _prepare_paths(case: CorpusCase, options: LifecycleOptions) -> _Paths:
    atoll_root = options.atoll_root.resolve(strict=True)
    if not (atoll_root / "pyproject.toml").is_file():
        raise ManifestError(f"Atoll root is not a Python project: {atoll_root}")
    workspace = options.workspace_root.resolve() / f"{case.id}-{options.tier}-{options.platform}"
    evidence = options.evidence_root.resolve()
    if workspace.exists():
        raise ManifestError(f"case workspace already exists: {workspace}")
    if evidence.exists() and any(evidence.iterdir()):
        raise ManifestError(f"case evidence directory is not empty: {evidence}")
    if evidence.is_relative_to(workspace) or workspace.is_relative_to(evidence):
        raise ManifestError("case evidence and disposable workspace must not contain each other")
    workspace.mkdir(parents=True)
    evidence.mkdir(parents=True, exist_ok=True)
    checkout = workspace / "checkout"
    project = checkout.joinpath(*case.project_subroot.parts)
    home = workspace / "home"
    temporary = workspace / "tmp"
    logs = evidence / "logs"
    for directory in (home, temporary, logs):
        directory.mkdir(parents=True, exist_ok=True)
    return _Paths(
        workspace=workspace,
        checkout=checkout,
        project=project,
        evidence=evidence,
        logs=logs,
        home=home,
        temporary=temporary,
        wheelhouse=workspace / "wheelhouse",
        tools_environment=workspace / "envs" / "tools",
    )


def _environment(paths: _Paths, *, offline: bool) -> dict[str, str]:
    environment = sanitized_environment(paths.home, paths.temporary, offline=offline)
    if offline and paths.wheelhouse.exists():
        environment["PIP_FIND_LINKS"] = str(paths.wheelhouse)
    return environment


def _clone_checkout(
    context: _RunContext,
    repository: str,
    environment: dict[str, str],
) -> None:
    paths = context.paths
    git = shutil.which("git")
    if git is None:
        raise LifecycleError("infrastructure-error", "git executable is unavailable")
    for attempt in range(1, _BOOTSTRAP_ATTEMPTS + 1):
        if paths.checkout.exists():
            shutil.rmtree(paths.checkout)
        result = _phase(
            context,
            _PhaseRequest(
                name=f"clone-{attempt}",
                argv=(
                    git,
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    repository,
                    str(paths.checkout),
                ),
                cwd=paths.workspace,
                environment=environment,
                timeout_seconds=300,
                network_allowed=True,
            ),
        )
        if result.exit_code == 0 and not result.timed_out:
            break
        if attempt == _BOOTSTRAP_ATTEMPTS:
            _require_success(result, "infrastructure-error", "pinned repository clone")
    checkout = _phase(
        context,
        _PhaseRequest(
            name="checkout-detached",
            argv=(
                git,
                "-C",
                str(paths.checkout),
                "checkout",
                "--detach",
                context.case.revision,
            ),
            cwd=paths.workspace,
            environment=environment,
            timeout_seconds=300,
            network_allowed=True,
        ),
    )
    _require_success(checkout, "infrastructure-error", "detached revision checkout")


def _require_pristine_source(
    paths: _Paths,
    expected: TrackedSourceManifest,
    phase: str,
) -> None:
    observed = tracked_source_manifest(paths.checkout)
    if observed.manifest_digest != expected.manifest_digest:
        raise LifecycleError("upstream-broken", f"{phase} changed tracked project files")


def _clean_generated_checkout(
    context: _RunContext,
    environment: dict[str, str],
) -> None:
    git = shutil.which("git")
    if git is None:
        raise LifecycleError("infrastructure-error", "git executable is unavailable")
    _required_phase(
        context,
        _PhaseRequest(
            name="clean-generated-checkout",
            argv=(git, "-C", str(context.paths.checkout), "clean", "-ffdx"),
            cwd=context.paths.workspace,
            environment=environment,
            timeout_seconds=300,
        ),
        "disposable checkout cleanup",
    )


def _dependency_lock(atoll_root: Path, case: CorpusCase) -> Path:
    root = atoll_root.resolve(strict=True)
    candidate = root.joinpath(*case.dependency_lock.parts)
    if candidate.is_symlink():
        raise LifecycleError(
            "security-violation",
            f"dependency lock is a symlink: {case.dependency_lock}",
        )
    lock = candidate.resolve(strict=True)
    if not lock.is_relative_to(root) or not lock.is_file():
        raise LifecycleError(
            "security-violation",
            f"dependency lock is not a regular file inside Atoll: {case.dependency_lock}",
        )
    return lock


def _prepare_tools_environment(
    context: _RunContext,
    online_environment: dict[str, str],
    offline_environment: dict[str, str],
    lock_path: Path,
) -> Path:
    manifest = context.manifest
    options = context.options
    paths = context.paths
    if options.environment_mode == "current-test":
        return Path(sys.executable)
    uv = shutil.which("uv")
    if uv is None:
        raise LifecycleError("infrastructure-error", "uv executable is unavailable")
    _required_phase(
        context,
        _PhaseRequest(
            name="tools-venv",
            argv=tools_venv_command(uv, manifest.python_version, paths.tools_environment),
            cwd=paths.workspace,
            environment=online_environment,
            timeout_seconds=300,
            network_allowed=True,
        ),
        "isolated tools environment creation",
    )
    tools_python = venv_python(paths.tools_environment)
    bootstrap = dependency_bootstrap_commands(
        uv,
        tools_python,
        lock_path,
        paths.wheelhouse,
    )
    _required_phase(
        context,
        _PhaseRequest(
            name="tools-ensurepip",
            argv=bootstrap.ensure_pip,
            cwd=paths.workspace,
            environment=offline_environment,
            timeout_seconds=300,
        ),
        "bundled pip bootstrap",
    )
    paths.wheelhouse.mkdir()
    for attempt in range(1, _BOOTSTRAP_ATTEMPTS + 1):
        result = _phase(
            context,
            _PhaseRequest(
                name=f"dependency-download-{attempt}",
                argv=bootstrap.download,
                cwd=paths.workspace,
                environment=online_environment,
                timeout_seconds=900,
                network_allowed=True,
            ),
        )
        if result.exit_code == 0 and not result.timed_out:
            break
        if attempt == _BOOTSTRAP_ATTEMPTS:
            _require_success(result, "infrastructure-error", "dependency wheelhouse bootstrap")
    sync_environment = {**offline_environment, "PIP_FIND_LINKS": str(paths.wheelhouse)}
    _required_phase(
        context,
        _PhaseRequest(
            name="tools-sync",
            argv=bootstrap.sync,
            cwd=paths.workspace,
            environment=sync_environment,
            timeout_seconds=900,
        ),
        "locked tools dependency installation",
    )
    _required_phase(
        context,
        _PhaseRequest(
            name="install-atoll",
            argv=(
                uv,
                "pip",
                "install",
                "--python",
                str(tools_python),
                "--offline",
                "--find-links",
                str(paths.wheelhouse),
                "--no-deps",
                str(options.atoll_root.resolve()),
            ),
            cwd=paths.workspace,
            environment=sync_environment,
            timeout_seconds=900,
        ),
        "Atoll installation into case tools environment",
    )
    offline_environment["PIP_FIND_LINKS"] = str(paths.wheelhouse)
    return tools_python


def _build_baseline_wheel(
    context: _RunContext,
    python: Path,
    environment: dict[str, str],
) -> Path:
    paths = context.paths
    destination = paths.workspace / "baseline-dist"
    destination.mkdir()
    command = [str(python), "-m", "build", "--wheel", "--outdir", str(destination)]
    if context.options.environment_mode == "current-test":
        command.append("--no-isolation")
    result = _phase(
        context,
        _PhaseRequest(
            name="baseline-wheel",
            argv=tuple(command),
            cwd=paths.project,
            environment=environment,
            timeout_seconds=context.case.compile_timeout_seconds or 900,
        ),
    )
    _require_success(result, "upstream-broken", "normal PEP 517 baseline wheel build")
    return _single_wheel(destination)


def _prepare_arm(
    context: _RunContext,
    label: str,
    wheel: Path,
    environment: dict[str, str],
    lock_path: Path,
) -> _Arm:
    manifest = context.manifest
    options = context.options
    paths = context.paths
    if options.environment_mode == "current-test":
        payload = paths.workspace / f"{label}-payload"
        _extract_wheel(wheel, payload)
        inherited = environment.get("PYTHONPATH")
        python_path = (
            str(payload) if inherited is None else os.pathsep.join((str(payload), inherited))
        )
        arm_environment = {**environment, "PYTHONPATH": python_path}
        return _Arm(Path(sys.executable), payload.resolve(), arm_environment)
    uv = shutil.which("uv")
    if uv is None:
        raise LifecycleError("infrastructure-error", "uv executable is unavailable")
    environment_root = paths.workspace / "envs" / label
    _required_phase(
        context,
        _PhaseRequest(
            name=f"{label}-venv",
            argv=(uv, "venv", "--python", manifest.python_version, str(environment_root)),
            cwd=paths.workspace,
            environment=environment,
            timeout_seconds=300,
        ),
        f"{label} environment creation",
    )
    python = venv_python(environment_root)
    _required_phase(
        context,
        _PhaseRequest(
            name=f"{label}-sync",
            argv=(
                uv,
                "pip",
                "sync",
                "--python",
                str(python),
                "--require-hashes",
                "--offline",
                "--find-links",
                str(paths.wheelhouse),
                "--link-mode",
                "copy",
                str(lock_path),
            ),
            cwd=paths.workspace,
            environment=environment,
            timeout_seconds=900,
        ),
        f"{label} locked dependency installation",
    )
    _required_phase(
        context,
        _PhaseRequest(
            name=f"{label}-install-wheel",
            argv=(
                uv,
                "pip",
                "install",
                "--python",
                str(python),
                "--offline",
                "--find-links",
                str(paths.wheelhouse),
                "--no-deps",
                "--force-reinstall",
                str(wheel),
            ),
            cwd=paths.workspace,
            environment=environment,
            timeout_seconds=900,
        ),
        f"{label} wheel installation",
    )
    import_root = _probe_site_root(context, label, python, environment)
    return _Arm(python=python, import_root=import_root, environment=environment)


def _run_focused_tests(
    context: _RunContext,
    arm: _Arm,
) -> None:
    case = context.case
    environment = dict(arm.environment)
    if arm.import_root != Path(sysconfig.get_paths()["purelib"]).resolve():
        environment["PYTHONPATH"] = str(arm.import_root)
    command = _replace_python(case.focused_test_command, arm.python)
    result = _phase(
        context,
        _PhaseRequest(
            name="focused-upstream-tests",
            argv=command,
            cwd=context.paths.project,
            environment=environment,
            timeout_seconds=case.test_timeout_seconds or 300,
        ),
    )
    _require_success(result, "upstream-broken", "focused upstream baseline tests")


def _run_oracle(
    context: _RunContext,
    label: str,
    arm: _Arm,
) -> str:
    case = context.case
    paths = context.paths
    state = context.state
    adapter = _reviewed_adapter_path(context, case.oracle_adapter)
    oracle_cwd = paths.workspace / "oracle-cwd"
    oracle_cwd.mkdir(exist_ok=True)
    command = (
        str(arm.python),
        str(adapter),
        "--project-root",
        str(paths.project),
        *case.oracle_arguments,
    )
    result = _phase(
        context,
        _PhaseRequest(
            name=label,
            argv=command,
            cwd=oracle_cwd,
            environment=arm.environment,
            timeout_seconds=case.test_timeout_seconds or 300,
        ),
    )
    _require_success(result, "compatibility-regression", f"{label} execution")
    payload = _read_json_object(paths.evidence / state.phases[-1].log_path)
    canonical = payload.get("canonical")
    imports = payload.get("imports")
    if not isinstance(imports, list) or not imports:
        raise LifecycleError(
            "compatibility-regression",
            f"{label} did not report imported project payload paths",
        )
    for raw_path in cast(list[object], imports):
        if not isinstance(raw_path, str):
            raise LifecycleError("compatibility-regression", f"{label} import path is not text")
        imported = Path(raw_path).resolve(strict=True)
        if not imported.is_relative_to(arm.import_root.resolve(strict=True)):
            raise LifecycleError(
                "compatibility-regression",
                f"{label} imported project code outside installed payload: {imported}",
            )
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _reviewed_adapter_path(context: _RunContext, adapter_name: str) -> Path:
    """Resolve one adapter while rejecting symlinks and directory escape.

    Args:
        context: Case paths and optional test adapter-root override.
        adapter_name: Dotted adapter stem from reviewed corpus configuration.

    Returns:
        Path: Absolute regular Python source inside the adapter root.

    Raises:
        LifecycleError: If the adapter is missing, linked, or resolves outside
            the reviewed root.
    """
    options = context.options
    adapter_root = options.adapter_root or options.atoll_root / "benchmarks" / "corpus" / "adapters"
    root = adapter_root.resolve(strict=True)
    unresolved = root.joinpath(*adapter_name.split(".")).with_suffix(".py")
    if unresolved.is_symlink():
        raise LifecycleError(
            "security-violation",
            f"adapter is a symlink: {adapter_name}",
        )
    try:
        adapter = unresolved.resolve(strict=True)
    except OSError as error:
        raise LifecycleError(
            "infrastructure-error",
            f"reviewed adapter is unavailable: {adapter_name}: {error}",
        ) from error
    if not adapter.is_relative_to(root) or not adapter.is_file():
        raise LifecycleError(
            "security-violation",
            f"adapter is not a reviewed regular file: {adapter_name}",
        )
    return adapter


def _probe_environment(
    context: _RunContext,
    python: Path,
    environment: dict[str, str],
    lock_digest: str,
) -> EnvironmentEvidence:
    script = (
        "import Cython,json,mypy.version,platform,sys;"
        "print(json.dumps({'python':sys.implementation.name+' '+platform.python_version(),"
        "'mypy':mypy.version.__version__,"
        "'cython':Cython.__version__}))"
    )
    result = _phase(
        context,
        _PhaseRequest(
            name="environment-python",
            argv=(str(python), "-c", script),
            cwd=context.paths.workspace,
            environment=environment,
            timeout_seconds=60,
        ),
    )
    _require_success(result, "infrastructure-error", "Python toolchain probe")
    payload = _read_json_object(context.paths.evidence / context.state.phases[-1].log_path)
    uv = shutil.which("uv")
    compiler = shutil.which("cc")
    uv_version = _probe_first_line(
        context,
        "environment-uv",
        (uv, "--version") if uv is not None else (),
        environment,
    )
    compiler_version = _probe_first_line(
        context,
        "environment-compiler",
        (compiler, "--version") if compiler is not None else (),
        environment,
    )
    atoll_revision = _probe_first_line(
        context,
        "environment-atoll-revision",
        ("git", "-C", str(context.options.atoll_root.resolve()), "rev-parse", "HEAD"),
        environment,
    )
    runner_image = os.environ.get("ATOLL_RUNNER_IMAGE", "local")
    hardware = "|".join(
        (platform.machine(), platform.processor() or "unknown", str(os.cpu_count() or 0))
    )
    return EnvironmentEvidence(
        python=_string_field(payload, "python"),
        atoll_revision=atoll_revision,
        uv=uv_version,
        mypy=_string_field(payload, "mypy"),
        cython=_string_field(payload, "cython"),
        compiler=compiler_version,
        operating_system=f"{platform.system()} {platform.release()}",
        architecture=platform.machine(),
        runner_image=runner_image,
        hardware_class=hardware,
        dependency_lock_digest=lock_digest,
    )


def _probe_first_line(
    context: _RunContext,
    name: str,
    command: tuple[str, ...],
    environment: dict[str, str],
) -> str:
    if not command:
        raise LifecycleError("infrastructure-error", f"{name} executable is unavailable")
    result = _phase(
        context,
        _PhaseRequest(
            name=name,
            argv=command,
            cwd=context.paths.workspace,
            environment=environment,
            timeout_seconds=60,
        ),
    )
    _require_success(result, "infrastructure-error", name)
    log = context.paths.evidence / context.state.phases[-1].log_path
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        raise LifecycleError("infrastructure-error", f"{name} emitted no identity")
    return lines[0].strip()


def _compiler_probe_environment(
    python: Path,
    paths: _Paths,
    base: dict[str, str],
) -> tuple[dict[str, str], Path]:
    cc = shutil.which("cc")
    cxx = shutil.which("c++")
    if cc is None or cxx is None:
        raise LifecycleError("infrastructure-error", "cc and c++ are required")
    probe_log = paths.workspace / "compiler-probe.log"
    probe_log.write_text("", encoding="utf-8")
    wrappers = paths.workspace / "compiler-probes"
    wrappers.mkdir()
    environment = dict(base)
    for name, executable, variable in (("cc", cc, "CC"), ("cxx", cxx, "CXX")):
        wrapper = wrappers / f"{name}-{_COMPILER_WRAPPER_NAME}"
        wrapper.write_text(
            f"#!{python}\n"
            "import os, sys\n"
            f"with open({str(probe_log)!r}, 'a', encoding='utf-8') as stream:\n"
            f"    stream.write({name!r} + '\\n')\n"
            f"os.execv({executable!r}, [{executable!r}, *sys.argv[1:]])\n",
            encoding="utf-8",
        )
        wrapper.chmod(0o755)
        environment[variable] = str(wrapper)
    return environment, probe_log


def _phase(
    context: _RunContext,
    request: _PhaseRequest,
) -> ProcessResult:
    paths = context.paths
    state = context.state
    index = len(state.phases) + 1
    log_path = paths.logs / f"{index:02d}-{request.name}.log"
    readable_paths = [context.options.atoll_root.resolve()]
    if context.options.repository_mirror is not None:
        readable_paths.append(context.options.repository_mirror.resolve())
    if context.options.adapter_root is not None:
        readable_paths.append(context.options.adapter_root.resolve())
    result = run_process(
        ProcessRequest(
            argv=request.argv,
            cwd=request.cwd,
            environment=request.environment,
            log_path=log_path,
            readable_paths=tuple(readable_paths),
            writable_paths=(paths.workspace,),
            network_allowed=request.network_allowed,
        ),
        ProcessLimits(
            timeout_seconds=request.timeout_seconds,
            max_log_bytes=context.manifest.defaults.max_log_bytes,
        ),
        context.sandbox,
    )
    state.phases.append(
        PhaseEvidence(
            name=request.name,
            argv=result.argv,
            exit_code=result.exit_code,
            timed_out=result.timed_out,
            duration_seconds=result.duration_seconds,
            log_path=PurePosixPath(log_path.relative_to(paths.evidence).as_posix()),
            log_truncated=result.log_truncated,
        )
    )
    return result


def _required_phase(
    context: _RunContext,
    request: _PhaseRequest,
    description: str,
) -> None:
    result = _phase(context, request)
    _require_success(result, "infrastructure-error", description)


def _require_success(result: ProcessResult, status: CaseStatus, description: str) -> None:
    if result.timed_out:
        raise LifecycleError("timeout", f"{description} exceeded its timeout")
    if result.exit_code != 0:
        raise LifecycleError(status, f"{description} exited {result.exit_code}")


def _require_successful_compile_report(report: dict[str, object], label: str) -> None:
    """Match a zero process exit to an explicitly successful report."""
    if report.get("success") is not True:
        raise LifecycleError(
            "compile-error",
            f"{label} Atoll compile exited successfully but its report did not",
        )


def _require_rejected_compile_process(
    result: ProcessResult,
    report: dict[str, object],
    label: str,
) -> None:
    """Accept only Atoll's intentional exit-one, unsuccessful-report contract."""
    if result.timed_out:
        raise LifecycleError("timeout", f"{label} Atoll compile exceeded its timeout")
    if result.exit_code != 1 or report.get("success") is not False:
        raise LifecycleError(
            "compile-error",
            f"{label} rejected compile has inconsistent process/report status",
        )


def _remove_compile_reports(paths: _Paths) -> None:
    """Remove the prior invocation's reports so a crashed warm run cannot reuse them."""
    report_root = paths.project / ".atoll"
    for suffix in ("json", "md"):
        (report_root / f"compile-report.{suffix}").unlink(missing_ok=True)


def _copy_compile_report(paths: _Paths, label: str) -> PurePosixPath:
    report_root = paths.project / ".atoll"
    for suffix in ("json", "md"):
        source = report_root / f"compile-report.{suffix}"
        if not source.is_file():
            raise LifecycleError(
                "compile-error",
                f"{label} compile did not produce compile-report.{suffix}",
            )
        shutil.copyfile(source, paths.evidence / f"{label}.compile-report.{suffix}")
    return PurePosixPath(f"{label}.compile-report.json")


def _retain_compiler_probe(paths: _Paths, source: Path) -> None:
    """Copy trusted compiler-wrapper evidence out of disposable workspace."""
    shutil.copyfile(source, paths.evidence / "compiler-probe.log")


def _capture_final_source(paths: _Paths, state: _State) -> None:
    if state.source_before is None or not paths.checkout.exists():
        return
    try:
        state.source_after = tracked_source_manifest(paths.checkout)
        _write_source_manifest(state.source_after, paths.evidence / "source-manifest-after.json")
    except (CheckoutSecurityError, OSError) as error:
        state.diagnostics.append(f"cannot capture final tracked source manifest: {error}")


def _source_changed(state: _State) -> bool:
    return (
        state.source_before is not None
        and state.source_after is not None
        and state.source_before.manifest_digest != state.source_after.manifest_digest
    )


def _case_result(
    case: CorpusCase,
    options: LifecycleOptions,
    state: _State,
    manifest_digest: str,
    case_digest: str,
) -> CaseResult:
    return CaseResult(
        schema_version=CASE_RESULT_SCHEMA_VERSION,
        case_id=case.id,
        tier=options.tier,
        platform=options.platform,
        status=state.status,
        repository=case.repository,
        revision=case.revision,
        manifest_digest=manifest_digest,
        case_digest=case_digest,
        comparison_key=state.comparison_key,
        diagnostics=tuple(state.diagnostics),
        source_digest_before=(
            None if state.source_before is None else state.source_before.manifest_digest
        ),
        source_digest_after=(
            None if state.source_after is None else state.source_after.manifest_digest
        ),
        source_unchanged=(
            None
            if state.source_before is None or state.source_after is None
            else not _source_changed(state)
        ),
        policy=state.policy,
        environment=state.environment,
        phases=tuple(state.phases),
        baseline_wheel_digest=state.baseline_wheel_digest,
        compiled_wheel_digest=state.compiled_wheel_digest,
        cold_report_path=state.cold_report_path,
        warm_report_path=state.warm_report_path,
        baseline_oracle_digest=state.baseline_oracle_digest,
        compiled_oracle_digest=state.compiled_oracle_digest,
        cold_compiler_invocations=state.cold_compiler_invocations,
        warm_compiler_invocations=state.warm_compiler_invocations,
        ratios=state.ratios,
    )


def _validate_compile_report(report: dict[str, object]) -> None:
    """Require the report schema consumed by the corpus runner."""
    version = report.get("version")
    if version != _COMPILE_REPORT_SCHEMA_VERSION:
        raise LifecycleError(
            "compile-error",
            f"compile report schema is {version!r}, expected {_COMPILE_REPORT_SCHEMA_VERSION}",
        )


def _validate_warm_report(report: dict[str, object]) -> None:
    """Reject cache evidence that still contains a native compiler phase."""
    _validate_compile_report(report)
    build = _mapping_field(report, "build")
    timings = build.get("phase_timings")
    if isinstance(timings, list) and any(
        _is_native_timing(item) for item in cast(list[object], timings)
    ):
        raise LifecycleError(
            "compatibility-regression",
            "warm compile report contains a native compiler phase",
        )


def _validate_performance_summary(
    performance: dict[str, object],
    status: Literal["passed", "not-profitable", "invalid"],
) -> None:
    """Require internally consistent final-gate medians, ratio, and threshold."""
    minimum = _positive_finite_float(performance.get("minimum_speedup"))
    baseline = _positive_finite_float(performance.get("baseline_median_seconds"))
    compiled = _positive_finite_float(performance.get("compiled_median_seconds"))
    if minimum is None or minimum <= 1.0 or baseline is None or compiled is None:
        raise LifecycleError("compile-error", "performance summary has invalid gate medians")
    speedup = _positive_finite_float(performance.get("speedup"))
    if status == "invalid":
        if performance.get("speedup") is not None:
            raise LifecycleError("compile-error", "noisy performance summary has a speedup")
        return
    if speedup is None or not math.isclose(speedup, baseline / compiled, rel_tol=1e-9):
        raise LifecycleError("compile-error", "performance summary speedup is inconsistent")
    passed = speedup >= minimum
    if passed != (status == "passed"):
        raise LifecycleError(
            "compile-error",
            f"performance status {status} contradicts its threshold",
        )


def _validate_performance_samples(report: dict[str, object], project_root: Path) -> None:
    """Verify measured arms produced one canonical result from their payloads.

    The final Atoll gate owns timing, but corpus evidence additionally proves
    that every measured subprocess imported the intended baseline or candidate
    payload and returned the same structured workload result. This closes the
    gap where a wrong-but-fast wheel could otherwise pass on exit code alone.

    Args:
        report: Schema-v6 report with final performance samples.
        project_root: Target root used for relative ``payload_root`` fields.

    Raises:
        LifecycleError: If sample evidence is incomplete, malformed, routed
            outside its payload, or semantically different between arms.
    """
    performance = _mapping_field(report, "performance")
    raw_samples = performance.get("samples")
    if not isinstance(raw_samples, list):
        raise LifecycleError("compile-error", "performance samples are not an array")
    counts = {"baseline": 0, "compiled": 0}
    canonical_digest: str | None = None
    for index, raw_sample in enumerate(cast(list[object], raw_samples)):
        mode, digest = _validated_performance_sample(raw_sample, project_root, index)
        if canonical_digest is None:
            canonical_digest = digest
        elif digest != canonical_digest:
            raise LifecycleError(
                "compatibility-regression",
                "baseline and compiled performance samples returned different canonical output",
            )
        counts[mode] += 1
    if any(count != _BENCHMARK_SAMPLES for count in counts.values()):
        raise LifecycleError(
            "compile-error",
            "performance report does not contain seven measured runs for each arm",
        )


def _validated_performance_sample(
    raw_sample: object,
    project_root: Path,
    index: int,
) -> tuple[PerformanceMode, str]:
    """Validate one measured command and return its arm plus canonical digest."""
    if not isinstance(raw_sample, dict):
        raise LifecycleError("compile-error", f"performance sample {index} is not an object")
    sample = cast(dict[str, object], raw_sample)
    raw_mode = sample.get("mode")
    if raw_mode == "baseline":
        mode: PerformanceMode = "baseline"
    elif raw_mode == "compiled":
        mode = "compiled"
    else:
        raise LifecycleError(
            "compile-error",
            f"performance sample {index} has invalid mode {raw_mode!r}",
        )
    if sample.get("success") is not True or sample.get("returncode") != 0:
        raise LifecycleError("compile-error", f"performance sample {index} was not successful")
    if _positive_finite_float(sample.get("duration_seconds")) is None:
        raise LifecycleError(
            "compile-error",
            f"performance sample {index} has invalid duration",
        )
    payload_root = _reported_payload_root(sample.get("payload_root"), project_root, index)
    payload = _performance_sample_payload(sample.get("stdout"), index)
    _validate_performance_imports(payload.get("imports"), payload_root, index)
    canonical = json.dumps(payload.get("canonical"), sort_keys=True, separators=(",", ":"))
    return mode, hashlib.sha256(canonical.encode()).hexdigest()


def _validate_performance_imports(value: object, payload_root: Path, index: int) -> None:
    """Prove every reported target import came from the measured payload root."""
    if not isinstance(value, list) or not value:
        raise LifecycleError(
            "compatibility-regression",
            f"performance sample {index} reported no imported payload paths",
        )
    for raw_import in cast(list[object], value):
        if not isinstance(raw_import, str) or not Path(raw_import).is_absolute():
            raise LifecycleError(
                "compatibility-regression",
                f"performance sample {index} reported an invalid import path",
            )
        imported = Path(raw_import).resolve()
        if not imported.is_relative_to(payload_root):
            raise LifecycleError(
                "compatibility-regression",
                f"performance sample {index} imported project code outside its payload",
            )


def _reported_payload_root(value: object, project_root: Path, index: int) -> Path:
    """Resolve one report payload root without requiring retained scratch files."""
    if not isinstance(value, str) or not value:
        raise LifecycleError(
            "compile-error",
            f"performance sample {index} has invalid payload_root",
        )
    path = Path(value)
    return (path if path.is_absolute() else project_root / path).resolve()


def _performance_sample_payload(value: object, index: int) -> dict[str, object]:
    """Parse one exact JSON object emitted by a reviewed workload adapter."""
    if not isinstance(value, str) or not value.strip():
        raise LifecycleError(
            "compatibility-regression",
            f"performance sample {index} emitted no structured output",
        )
    try:
        parsed: object = json.loads(value)
    except json.JSONDecodeError as error:
        raise LifecycleError(
            "compatibility-regression",
            f"performance sample {index} emitted invalid JSON",
        ) from error
    if not isinstance(parsed, dict):
        raise LifecycleError(
            "compatibility-regression",
            f"performance sample {index} output is not an object",
        )
    payload = cast(dict[str, object], parsed)
    if set(payload) != {"canonical", "imports"} or not isinstance(payload.get("canonical"), dict):
        raise LifecycleError(
            "compatibility-regression",
            f"performance sample {index} output lacks its canonical object",
        )
    return payload


def ratio_evidence_from_report(report: dict[str, object]) -> RatioEvidence:
    """Extract labeled performance ratios and successful raw samples.

    Schema v6 does not expose raw source-only search samples, so that tuple
    remains empty rather than fabricating observations from a median. Source
    and native-layer ratios are retained only when the final composition proves
    the corresponding layers were accepted.

    Args:
        report: Valid schema-v6 compile report.

    Returns:
        RatioEvidence: Raw final-gate samples and available composition ratios.
    """
    performance_value = report.get("performance")
    if not isinstance(performance_value, dict):
        return RatioEvidence()
    performance = cast(dict[str, object], performance_value)
    samples = performance.get("samples")
    baseline_samples = _successful_sample_durations(samples, "baseline")
    final_samples = _successful_sample_durations(samples, "compiled")
    final_speedup = _positive_finite_float(performance.get("speedup"))

    composition_value = report.get("final_composition")
    composition = (
        cast(dict[str, object], composition_value) if isinstance(composition_value, dict) else {}
    )
    source_ids = _string_list(composition.get("source_plan_ids"))
    native_ids = _string_list(composition.get("native_variant_ids"))
    source_speedup = _accepted_source_speedup(report, source_ids)
    native_speedup = (
        final_speedup / source_speedup
        if final_speedup is not None and source_speedup is not None and native_ids
        else None
    )
    return RatioEvidence(
        python_rewrite_vs_original=source_speedup,
        final_wheel_vs_original=final_speedup,
        native_vs_source_only=native_speedup,
        baseline_samples_seconds=baseline_samples,
        source_only_samples_seconds=(),
        final_wheel_samples_seconds=final_samples,
    )


def _accepted_source_speedup(
    report: dict[str, object],
    accepted_plan_ids: tuple[str, ...],
) -> float | None:
    """Return the last accepted source-plan ratio represented in the payload."""
    if not accepted_plan_ids:
        return None
    source_value = report.get("source_optimization")
    if not isinstance(source_value, dict):
        return None
    trials = cast(dict[str, object], source_value).get("trials")
    if not isinstance(trials, list):
        return None
    speedup: float | None = None
    for value in cast(list[object], trials):
        if not isinstance(value, dict):
            continue
        trial = cast(dict[str, object], value)
        if trial.get("status") != "accepted" or trial.get("plan_id") not in accepted_plan_ids:
            continue
        candidate = _positive_finite_float(trial.get("source_speedup"))
        if candidate is not None:
            speedup = candidate
    return speedup


def _successful_sample_durations(value: object, mode: str) -> tuple[float, ...]:
    """Retain successful measured durations for one final-gate arm."""
    if not isinstance(value, list):
        return ()
    durations: list[float] = []
    for item in cast(list[object], value):
        if not isinstance(item, dict):
            continue
        run = cast(dict[str, object], item)
        if run.get("mode") != mode or run.get("success") is not True:
            continue
        duration = _positive_finite_float(run.get("duration_seconds"))
        if duration is not None:
            durations.append(duration)
    return tuple(durations)


def _positive_finite_float(value: object) -> float | None:
    """Normalize a positive JSON number without accepting booleans or NaN."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number > 0.0 and math.isfinite(number) else None


def _string_list(value: object) -> tuple[str, ...]:
    """Return text-only list entries, otherwise an empty conservative value."""
    if not isinstance(value, list):
        return ()
    items = cast(list[object], value)
    if any(not isinstance(item, str) for item in items):
        return ()
    return tuple(cast(list[str], items))


def _is_clean_noop_report(report: dict[str, object]) -> bool:
    """Recognize only Atoll's structured no-candidate result as compatible."""
    if report.get("version") != _COMPILE_REPORT_SCHEMA_VERSION:
        return False
    build = _mapping_field(report, "build")
    return (
        report.get("success") is False
        and build.get("stderr") == "scan found no backend-supported typed regions"
        and build.get("command") == []
        and build.get("artifacts") == []
        and build.get("support_artifacts") == []
    )


def _is_native_timing(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    return cast(dict[object, object], value).get("name") in _NATIVE_PHASES


def _strict_routing_environment(report: dict[str, object]) -> dict[str, str]:
    composition = _mapping_field(report, "final_composition")
    environment: dict[str, str] = {}
    native = composition.get("native_variant_ids")
    source = composition.get("source_plan_ids")
    if isinstance(native, list) and native:
        environment["ATOLL_REQUIRE_COMPILED"] = "1"
    if isinstance(source, list) and source:
        environment["ATOLL_REQUIRE_OPTIMIZED"] = "1"
    return environment


def _write_source_manifest(manifest: TrackedSourceManifest, path: Path) -> None:
    path.write_text(
        f"{json.dumps(asdict(manifest), default=_json_default, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


def _extract_wheel(wheel: Path, destination: Path) -> None:
    destination.mkdir()
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            relative = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "\\" in member.filename
                or stat.S_ISLNK(mode)
            ):
                raise LifecycleError(
                    "security-violation",
                    f"wheel contains unsafe payload path {member.filename!r}",
                )
            target = destination.joinpath(*relative.parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _probe_site_root(
    context: _RunContext,
    label: str,
    python: Path,
    environment: dict[str, str],
) -> Path:
    result = _phase(
        context,
        _PhaseRequest(
            name=f"{label}-site-root",
            argv=(str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"),
            cwd=context.paths.workspace,
            environment=environment,
            timeout_seconds=60,
        ),
    )
    _require_success(result, "infrastructure-error", f"{label} site-packages probe")
    log = context.paths.evidence / context.state.phases[-1].log_path
    return Path(log.read_text(encoding="utf-8").strip()).resolve(strict=True)


def _replace_python(argv: tuple[str, ...], python: Path) -> tuple[str, ...]:
    if not argv:
        raise LifecycleError("infrastructure-error", "focused test command is empty")
    executable = str(python) if argv[0] in {"python", "python3"} else argv[0]
    return (executable, *argv[1:])


def _single_wheel(directory: Path) -> Path:
    wheels = tuple(sorted(directory.glob("*.whl")))
    if len(wheels) != 1:
        raise LifecycleError(
            "infrastructure-error",
            f"expected exactly one wheel under {directory}, found {len(wheels)}",
        )
    return wheels[0]


def tools_venv_command(uv: str, python_version: str, environment_root: Path) -> tuple[str, ...]:
    """Build an unseeded uv venv command for the isolated corpus toolchain.

    Args:
        uv: Absolute uv executable path.
        python_version: Manifest-selected Python minor version.
        environment_root: Case-local destination for the tools environment.

    Returns:
        tuple[str, ...]: Shell-free argv that does not fetch or seed pip.
    """
    return (uv, "venv", "--python", python_version, str(environment_root))


def dependency_bootstrap_commands(
    uv: str,
    python: Path,
    lock_path: Path,
    wheelhouse: Path,
) -> DependencyBootstrapCommands:
    """Build the exact offline-bootstrap and hash-verified dependency commands.

    Args:
        uv: Absolute uv executable path.
        python: Venv-local interpreter receiving the bundled pip bootstrap.
        lock_path: Reviewed exact and hash-verified requirements file.
        wheelhouse: Case-local directory receiving downloaded wheels.

    Returns:
        DependencyBootstrapCommands: Commands with one explicit network download
        and offline installation from the resulting wheelhouse.
    """
    return DependencyBootstrapCommands(
        ensure_pip=(str(python), "-m", "ensurepip", "--upgrade"),
        download=(
            str(python),
            "-m",
            "pip",
            "download",
            "--only-binary=:all:",
            "--require-hashes",
            "--requirement",
            str(lock_path),
            "--dest",
            str(wheelhouse),
        ),
        sync=(
            uv,
            "pip",
            "sync",
            "--python",
            str(python),
            "--require-hashes",
            "--offline",
            "--find-links",
            str(wheelhouse),
            "--link-mode",
            "copy",
            str(lock_path),
        ),
    )


def venv_python(environment_root: Path) -> Path:
    """Return the venv-local executable path without resolving its symlink.

    Args:
        environment_root: Root created by ``uv venv``.

    Returns:
        Path: Absolute executable path that still identifies the venv to uv.

    Raises:
        LifecycleError: If the environment has zero or multiple supported
            interpreter paths.
    """
    candidates = (environment_root / "bin" / "python", environment_root / "Scripts" / "python.exe")
    matches = tuple(path for path in candidates if path.is_file())
    if len(matches) != 1:
        raise LifecycleError(
            "infrastructure-error",
            f"cannot locate isolated Python under {environment_root}",
        )
    # uv locates the environment from the executable path.  Resolving the
    # standard venv symlink would point back at the base interpreter and make a
    # subsequent sync mutate the host environment instead of this venv.
    return matches[0].absolute()


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LifecycleError(
            "infrastructure-error", f"invalid JSON evidence {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise LifecycleError("infrastructure-error", f"JSON evidence is not an object: {path}")
    return cast(dict[str, object], payload)


def _mapping_field(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise LifecycleError("compile-error", f"compile report field {key} is not an object")
    return cast(dict[str, object], value)


def _string_field(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise LifecycleError("infrastructure-error", f"environment field {key} is invalid")
    return value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def _json_default(value: object) -> str:
    if isinstance(value, PurePosixPath):
        return value.as_posix()
    raise TypeError(f"unsupported corpus identity value: {type(value).__name__}")
