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

from scripts.benchmark_corpus.manifest import ManifestError
from scripts.benchmark_corpus.models import (
    CaseResult,
    CaseStatus,
    CompilePolicy,
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
CASE_RESULT_SCHEMA_VERSION = 1
_NATIVE_PHASES = frozenset({"mypycify", "cythonize", "build_ext"})
_COMPILE_REPORT_SCHEMA_VERSION = 6
_COMPILER_WRAPPER_NAME = "compiler-probe.py"
_BOOTSTRAP_ATTEMPTS = 2


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
        allow_unsandboxed: Explicit local opt-in when no sandbox is available.
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


@dataclass(frozen=True, slots=True)
class _RunContext:
    manifest: CorpusManifest
    case: CorpusCase
    options: LifecycleOptions
    paths: _Paths
    sandbox: Sandbox
    state: _State


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
    case_digest = _case_digest(case)
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
        result = _case_result(case, options, state, manifest_digest, case_digest)
        json_path, markdown_path = write_case_result(result, paths.evidence)
        if not options.keep_workspace:
            shutil.rmtree(paths.workspace, ignore_errors=True)
    return CaseRunSummary(result=result, json_path=json_path, markdown_path=markdown_path)


def _execute_case(context: _RunContext) -> None:
    manifest = context.manifest
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
    state.policy = append_compile_policy(
        paths.project / "pyproject.toml",
        CompilePolicy(backends=manifest.backends),
        paths.evidence,
        paths.checkout,
    )
    state.source_before = tracked_source_manifest(paths.checkout)
    _write_source_manifest(state.source_before, paths.evidence / "source-manifest-before.json")
    warm_report, compiled_wheel = _compile_cold_and_warm(
        context,
        tools_python,
        compiler_environment,
        compiler_log,
        baseline_wheel,
    )
    _verify_compiled_wheel(context, compiled_wheel, compiler_environment, lock_path, warm_report)
    state.status = classify_compile_report(warm_report, options.tier)
    state.comparison_key = _comparison_key(case, state.environment, state.policy)


def _compile_cold_and_warm(
    context: _RunContext,
    tools_python: Path,
    compiler_environment: dict[str, str],
    compiler_log: Path,
    baseline_wheel: Path,
) -> tuple[dict[str, object], Path]:
    """Run identical cold and warm compiles and prove native cache reuse."""
    manifest = context.manifest
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
    compile_timeout = (
        manifest.defaults.performance_timeout_seconds
        if options.tier == "performance"
        else manifest.defaults.compile_timeout_seconds
    )
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
    cold_noop = _is_clean_noop_report(cold_report)
    if not cold_noop:
        _require_success(cold, "compile-error", "cold Atoll compile")
    state.cold_compiler_invocations = _line_count(compiler_log)
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
    warm_noop = _is_clean_noop_report(warm_report)
    if warm_noop != cold_noop:
        raise LifecycleError("compile-error", "cold and warm no-op classification differed")
    if not warm_noop:
        _require_success(warm, "compile-error", "warm Atoll compile")
    state.warm_compiler_invocations = _line_count(compiler_log) - state.cold_compiler_invocations
    if state.warm_compiler_invocations:
        raise LifecycleError(
            "compatibility-regression",
            f"warm compile invoked native compiler {state.warm_compiler_invocations} time(s)",
        )
    _validate_warm_report(warm_report)
    state.source_after = tracked_source_manifest(paths.checkout)
    _write_source_manifest(state.source_after, paths.evidence / "source-manifest-after.json")
    compiled_wheel = (
        baseline_wheel if warm_noop else _single_wheel(paths.project / ".atoll" / "dist")
    )
    state.compiled_wheel_digest = _sha256_file(compiled_wheel)
    return warm_report, compiled_wheel


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
    options = context.options
    paths = context.paths
    state = context.state
    adapter_root = options.adapter_root or options.atoll_root / "benchmarks" / "corpus" / "adapters"
    root = adapter_root.resolve(strict=True)
    adapter = root.joinpath(*case.oracle_adapter.split(".")).with_suffix(".py").resolve(strict=True)
    if not adapter.is_relative_to(root) or not adapter.is_file() or adapter.is_symlink():
        raise LifecycleError(
            "security-violation",
            f"oracle adapter is not a reviewed regular file: {case.oracle_adapter}",
        )
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
        ratios=RatioEvidence(),
    )


def _validate_warm_report(report: dict[str, object]) -> None:
    version = report.get("version")
    if version != _COMPILE_REPORT_SCHEMA_VERSION:
        raise LifecycleError(
            "compile-error",
            f"warm compile report schema is {version!r}, expected {_COMPILE_REPORT_SCHEMA_VERSION}",
        )
    build = _mapping_field(report, "build")
    timings = build.get("phase_timings")
    if isinstance(timings, list) and any(
        _is_native_timing(item) for item in cast(list[object], timings)
    ):
        raise LifecycleError(
            "compatibility-regression",
            "warm compile report contains a native compiler phase",
        )


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


def _comparison_key(
    case: CorpusCase,
    environment: EnvironmentEvidence,
    policy: PolicyEvidence,
) -> str:
    workload = None if case.workload is None else asdict(case.workload)
    payload = {
        "architecture": environment.architecture,
        "compiler": environment.compiler,
        "cython": environment.cython,
        "dependency_lock": environment.dependency_lock_digest,
        "hardware_class": environment.hardware_class,
        "mypy": environment.mypy,
        "operating_system": environment.operating_system,
        "platform": environment.runner_image,
        "policy": policy.digest,
        "python": environment.python,
        "revision": case.revision,
        "workload": workload,
    }
    encoded = json.dumps(payload, default=_json_default, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _case_digest(case: CorpusCase) -> str:
    encoded = json.dumps(
        asdict(case),
        default=_json_default,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


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
