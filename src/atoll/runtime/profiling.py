"""Collect self-contained baseline profiling evidence for candidate selection.

The parent API in this module never imports target project modules. It validates
supported Python launch forms, runs a private bootstrap module in child
processes, reads scratch JSON evidence, and removes those scratch files before
returning. Candidate selection is deliberately recomputed from the supplied
profile evidence on every invocation; no cache or profitability state is kept.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, Unpack, cast

from atoll.models import SymbolId, SymbolRecord

type SymbolFact = SymbolRecord

ProfileStatus = Literal["profiled", "static-fallback", "invalid", "unconfigured"]
LaunchKind = Literal["script", "module", "unsupported", "unconfigured"]
ProfilePassKind = Literal["sampling", "types"]
CandidateDecisionReason = Literal[
    "selected",
    "unmapped",
    "below-threshold",
    "coverage-reached",
    "limit",
    "cache-replayed",
    "cache-replay-excluded",
]

_MIN_TOTAL_SAMPLES = 100
_MIN_CANDIDATE_SAMPLES = 20
_MIN_CANDIDATE_SHARE = 0.02
_TARGET_MAPPED_COVERAGE = 0.80
_MAX_SELECTED_CANDIDATES = 4
_BOOTSTRAP_PATH = Path(__file__).with_name("_profile_bootstrap.py")

_SCRIPT_COMMAND_LENGTH = 2
_MODULE_COMMAND_LENGTH = 3

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
type JsonObject = dict[str, JsonValue]
type _RunSubprocess = Callable[..., subprocess.CompletedProcess[str]]

_subprocess_run: _RunSubprocess = subprocess.run


class _BaselineProfileOptions(TypedDict):
    scratch_dir: Path
    enable_atoll: NotRequired[bool]
    observation_targets: NotRequired[tuple[SymbolId, ...]]
    spawn_targets: NotRequired[tuple[ProfileSpawnSiteTarget, ...]]
    call_edge_targets: NotRequired[tuple[ProfileCallEdgeTarget, ...]]


@dataclass(frozen=True, slots=True)
class _SubprocessInvocation:
    command: tuple[str, ...]
    cwd: Path
    env: dict[str, str]
    check: bool
    shell: bool
    capture_output: bool
    text: bool


@dataclass(frozen=True, slots=True)
class _BootstrapRequest:
    profile_stage: ProfilePassKind
    launch_plan: _LaunchPlan
    project_root: Path
    payload_root: Path
    module_paths: tuple[tuple[str, str], ...]
    scratch_dir: Path
    enable_atoll: bool
    targets: tuple[str, ...]
    spawn_targets: tuple[ProfileSpawnSiteTarget, ...]
    call_edge_targets: tuple[ProfileCallEdgeTarget, ...] = ()


@dataclass(frozen=True, slots=True)
class _CandidatePolicyContext:
    total_samples: int
    mapped_activity_samples: int
    selected_samples: int
    selected_count: int


@dataclass(frozen=True, slots=True)
class _ProfileMemberPayloadContext:
    """Merged sampling and lifecycle payloads used to construct one member record.

    Attributes:
        total_samples: Total statistical samples collected by the sampling pass.
        scheduler_overhead_counts: Nested non-project samples grouped by project caller.
        signature_payload: Bounded canonical type observations grouped by member.
        member_lifecycle_payload: Lifecycle event counts grouped by member.
    """

    total_samples: int
    scheduler_overhead_counts: JsonObject
    signature_payload: JsonObject
    member_lifecycle_payload: JsonObject


@dataclass(frozen=True, slots=True)
class SubprocessPassEvidence:
    """Captured evidence from one profiling child process.

    Attributes:
        pass_kind: Profiling pass represented by this child process.
        command: Exact argv tuple used for the profiling bootstrap child.
        returncode: Child process exit status.
        stdout: Captured benchmark standard output from the child.
        stderr: Captured benchmark standard error from the child.
        duration_seconds: Parent-observed elapsed duration for evidence only.
    """

    pass_kind: ProfilePassKind
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class LifecycleCounts:
    """Python lifecycle event counts observed through `sys.monitoring`.

    Attributes:
        start: Python frame start events.
        return_: Python frame return events.
        yield_: Python yield events.
        resume: Python resume events.
        unwind: Python unwind events.
        throw: Python throw events.
    """

    start: int
    return_: int
    yield_: int
    resume: int
    unwind: int
    throw: int


@dataclass(frozen=True, slots=True)
class ProfileSpawnSiteTarget:
    """Static scheduler call site to count during the targeted profiling pass.

    Attributes:
        id: Stable source-derived call-site identity.
        owner: Callable containing the scheduler call.
        lineno: One-based scheduler-call source line.
        col_offset: Zero-based scheduler-call source column.
        scheduler_method: Expected scheduler method name such as `create_task` or `start_soon`.
        end_lineno: One-based final scheduler-call source line.
        end_col_offset: Zero-based final scheduler-call source column.
    """

    id: str
    owner: SymbolId
    lineno: int
    col_offset: int
    scheduler_method: str
    end_lineno: int | None = None
    end_col_offset: int | None = None


@dataclass(frozen=True, slots=True)
class CanonicalCallableCount:
    """Canonical scheduler callable identity and observation count.

    Attributes:
        identity: Runtime callable type path without values or representations.
        count: Invocations attributed to this identity at one exact spawn site.
    """

    identity: str
    count: int


@dataclass(frozen=True, slots=True)
class ProfiledSpawnSite:
    """Runtime invocation evidence for one configured scheduler call site.

    Attributes:
        target: Static call-site contract used to collect the evidence.
        invocation_count: Scheduler calls observed at the exact source site.
        callable_identities: Canonical runtime scheduler callable identities and counts.
    """

    target: ProfileSpawnSiteTarget
    invocation_count: int
    callable_identities: tuple[CanonicalCallableCount, ...]


@dataclass(frozen=True, slots=True)
class ProfileCallEdgeTarget:
    """Static same-module direct call site to count during targeted profiling.

    The target carries only source identities and coordinates. Runtime profiling
    increments its count when the exact call instruction resolves to the
    configured canonical callable identity; argument values are never observed.

    Attributes:
        id: Stable source-derived call-site identity.
        owner: Project callable containing the direct call.
        callee: Expected same-module project callable invoked at the site.
        lineno: One-based call source line.
        col_offset: Zero-based call source column.
        end_lineno: One-based final call source line.
        end_col_offset: Zero-based final call source column.

    Raises:
        ValueError: The owner and callee belong to different modules.
    """

    id: str
    owner: SymbolId
    callee: SymbolId
    lineno: int
    col_offset: int
    end_lineno: int | None = None
    end_col_offset: int | None = None

    def __post_init__(self) -> None:
        """Reject cross-module edges before they reach the profiling child.

        Raises:
            ValueError: If owner and callee do not belong to the same project module.
        """
        if self.owner.module != self.callee.module:
            raise ValueError("profiled direct call edges must remain within one module")


@dataclass(frozen=True, slots=True)
class ProfiledCallEdge:
    """Exact runtime count for one canonical same-module direct call edge.

    Attributes:
        target: Static source call-site contract used for collection.
        invocation_count: Calls whose source position and runtime callable identity matched.
    """

    target: ProfileCallEdgeTarget
    invocation_count: int


@dataclass(frozen=True, slots=True)
class CanonicalTypeObservation:
    """Canonical parameter type count for a profiled callable.

    Attributes:
        parameter_name: Source parameter name observed in the frame locals.
        type_path: Canonical runtime type identity as `module.qualname`.
        count: Number of calls whose parameter had this canonical type.
    """

    parameter_name: str
    type_path: str
    count: int


@dataclass(frozen=True, slots=True)
class ObservedSignature:
    """Canonical argument-type signature observed for a profiled callable.

    Attributes:
        parameters: Parameter type observations in source argument order.
        count: Number of calls with this exact canonical signature.
    """

    parameters: tuple[CanonicalTypeObservation, ...]
    count: int


@dataclass(frozen=True, slots=True)
class ProfiledMember:
    """Profiled project member keyed by module and runtime qualified name.

    Attributes:
        module: Importable module name resolved from `module_paths`.
        qualname: Runtime code qualified name.
        samples: Statistical leaf-frame samples mapped to this member.
        coverage: Fraction of total workload samples represented by this member.
        scheduler_overhead_samples: Samples in non-project frames whose nearest active project
            caller was this member.
        scheduler_overhead_coverage: Fraction of workload samples attributed as nested scheduler
            or library overhead beneath this member.
        call_count: Targeted type-observation calls observed for this member.
        invocation_count: Total target invocations, including calls after type observation capped.
        lifecycle: Python lifecycle event counts observed for this member.
        signatures: Canonical argument-type signatures observed in the targeted pass.
        polymorphic_overflow: Whether more than eight unique signatures were observed.
        observation_capped: Whether type observation stopped at its per-member budget.
        completed_calls: Target invocations observed through a return or unwind event.
        max_active_calls: Maximum simultaneous active invocations observed for this member.
        pre_completion_suspensions: Yield events observed while a target invocation was active.
    """

    module: str
    qualname: str
    samples: int
    coverage: float
    call_count: int
    lifecycle: LifecycleCounts
    signatures: tuple[ObservedSignature, ...]
    polymorphic_overflow: bool
    scheduler_overhead_samples: int = 0
    scheduler_overhead_coverage: float = 0.0
    invocation_count: int = 0
    observation_capped: bool = False
    completed_calls: int = 0
    max_active_calls: int = 0
    pre_completion_suspensions: int = 0

    @property
    def immediate_result_ratio(self) -> float:
        """Return a conservative fraction of completed calls that never suspended.

        Monitoring reports suspension events rather than invocation identities. Treating every
        event as a distinct suspended call can only lower this ratio, preserving conservative
        source-optimization eligibility when one invocation suspends more than once.

        Returns:
            float: Value from zero to one, or zero when no call completed.
        """
        if self.completed_calls <= 0:
            return 0.0
        suspended_calls = min(self.completed_calls, self.pre_completion_suspensions)
        return (self.completed_calls - suspended_calls) / self.completed_calls

    @property
    def attributed_samples(self) -> int:
        """Return leaf work plus nested scheduler or library work owned by this member.

        Returns:
            int: Disjoint sampling events attributed to the callable or its active stack.
        """
        return self.samples + self.scheduler_overhead_samples

    @property
    def symbol(self) -> SymbolId:
        """Return the static symbol identity implied by the profiled member.

        Returns:
            SymbolId: Stable symbol identity matching the module and qualified name.
        """
        return SymbolId(module=self.module, qualname=self.qualname)


@dataclass(frozen=True, slots=True)
class MappedCandidateDecision:
    """Static mapping and hotness decision for one profiled member.

    Attributes:
        symbol: Static symbol when the profiled member maps to `symbols`.
        module: Runtime module name observed in the profile.
        qualname: Runtime qualified name observed in the profile.
        samples: Statistical leaf-frame samples for this member.
        coverage: Fraction of total workload samples represented by this member.
        selected: Whether the member passed the candidate policy.
        reason: Deterministic policy reason for selection or rejection.
        scheduler_overhead_samples: Nested scheduler or library samples owned by the member.
        attributed_samples: Leaf plus nested samples used by the hotness policy.
        attributed_coverage: Fraction of total workload samples used by the hotness policy.
    """

    symbol: SymbolId | None
    module: str
    qualname: str
    samples: int
    coverage: float
    scheduler_overhead_samples: int
    attributed_samples: int
    attributed_coverage: float
    selected: bool
    reason: CandidateDecisionReason


@dataclass(frozen=True, slots=True)
class ProfileResult:
    """Aggregate baseline profile and candidate-selection evidence.

    Attributes:
        status: Profile status describing whether dynamic evidence was collected.
        reason: Human-readable explanation for the current status.
        launch_kind: Supported launch shape used for child execution.
        total_samples: Statistical samples collected across the benchmark.
        mapped_project_samples: Samples mapped to configured project modules.
        mapped_coverage: Fraction of samples mapped to configured project modules.
        scheduler_overhead_samples: Nested non-project samples attributed to active project
            callers.
        scheduler_overhead_coverage: Fraction of total samples represented by that overhead.
        selected_hot_samples: Leaf and attributed scheduler samples covered by candidates.
        selected_hot_coverage: Fraction of mapped project activity covered by candidates.
        runs: Child-process evidence for each profiling pass.
        lifecycle: Python lifecycle event counts from the targeted observation pass.
        members: Profiled project members with sample and type evidence.
        candidates: Candidate mapping decisions derived from static symbols.
        selected_symbols: Static symbols accepted by the candidate policy.
        spawn_sites: Exact scheduler call-site invocation evidence.
        call_edges: Exact same-module direct call-edge invocation evidence.
    """

    status: ProfileStatus
    reason: str
    launch_kind: LaunchKind
    total_samples: int
    mapped_project_samples: int
    mapped_coverage: float
    selected_hot_samples: int
    selected_hot_coverage: float
    runs: tuple[SubprocessPassEvidence, ...]
    lifecycle: LifecycleCounts
    members: tuple[ProfiledMember, ...]
    candidates: tuple[MappedCandidateDecision, ...]
    selected_symbols: tuple[SymbolId, ...]
    scheduler_overhead_samples: int = 0
    scheduler_overhead_coverage: float = 0.0
    spawn_sites: tuple[ProfiledSpawnSite, ...] = ()
    call_edges: tuple[ProfiledCallEdge, ...] = ()


@dataclass(frozen=True, slots=True)
class _LaunchPlan:
    launch_kind: Literal["script", "module"]
    target: str
    args: tuple[str, ...]


def unconfigured_profile() -> ProfileResult:
    """Return deterministic no-benchmark profile evidence.

    Returns:
        ProfileResult: Empty profile evidence suitable for report and command defaults.
    """
    return ProfileResult(
        status="unconfigured",
        reason="no benchmark command configured",
        launch_kind="unconfigured",
        total_samples=0,
        mapped_project_samples=0,
        mapped_coverage=0.0,
        selected_hot_samples=0,
        selected_hot_coverage=0.0,
        runs=(),
        lifecycle=LifecycleCounts(start=0, return_=0, yield_=0, resume=0, unwind=0, throw=0),
        members=(),
        candidates=(),
        selected_symbols=(),
    )


def run_baseline_profile(
    command: tuple[str, ...],
    *,
    project_root: Path,
    payload_root: Path,
    module_paths: tuple[tuple[str, str], ...],
    **options: Unpack[_BaselineProfileOptions],
) -> ProfileResult:
    """Run the two-pass baseline profiler for a supported Python benchmark command.

    Args:
        command: Benchmark argv. Only Python script and `python -m module` forms are supported.
        project_root: Target project working directory for child processes.
        payload_root: Import root placed first on `PYTHONPATH` for child processes.
        module_paths: `(module, install-relative .py suffix)` entries used for mapping frames.
        options: Required `scratch_dir` directory for temporary JSON files removed before
            return, plus optional `observation_targets` symbols to observe even when sampling
            does not identify them as hot members and `spawn_targets` scheduler calls whose exact
            invocation counts and canonical callable identities must be retained. Set
            `enable_atoll` to profile the payload with inherited Atoll routing enabled instead of
            forcing `ATOLL_DISABLE=1`. Optional `call_edge_targets` count exact same-module direct
            calls without retaining values.

    Returns:
        ProfileResult: Baseline profile evidence without candidate decisions.
    """
    launch_plan = _launch_plan(command)
    if launch_plan is None:
        return replace(
            unconfigured_profile(),
            status="static-fallback",
            reason="unsupported benchmark launcher; static candidate fallback required",
            launch_kind="unsupported",
        )

    resolved_project_root = project_root.resolve()
    resolved_payload_root = payload_root.resolve()
    resolved_scratch_dir = options["scratch_dir"].resolve()
    enable_atoll = options.get("enable_atoll", False)
    resolved_scratch_dir.mkdir(parents=True, exist_ok=True)
    sampling_payload, sampling_run = _run_bootstrap_pass(
        _BootstrapRequest(
            profile_stage="sampling",
            launch_plan=launch_plan,
            project_root=resolved_project_root,
            payload_root=resolved_payload_root,
            module_paths=module_paths,
            scratch_dir=resolved_scratch_dir,
            enable_atoll=enable_atoll,
            targets=(),
            spawn_targets=(),
            call_edge_targets=(),
        )
    )
    runs: tuple[SubprocessPassEvidence, ...] = (sampling_run,)
    if sampling_run.returncode != 0:
        return _profile_from_payload(
            sampling_payload,
            status="invalid",
            reason=f"sampling profile exited with status {sampling_run.returncode}",
            launch_kind=launch_plan.launch_kind,
            runs=runs,
        )

    hot_targets = tuple(member_key for member_key, _ in _hot_member_keys(sampling_payload))
    explicit_targets = tuple(
        _member_key(target) for target in options.get("observation_targets", ())
    )
    call_edge_owner_targets = tuple(
        _member_key(target.owner) for target in options.get("call_edge_targets", ())
    )
    type_payload, type_run = _run_bootstrap_pass(
        _BootstrapRequest(
            profile_stage="types",
            launch_plan=launch_plan,
            project_root=resolved_project_root,
            payload_root=resolved_payload_root,
            module_paths=module_paths,
            scratch_dir=resolved_scratch_dir,
            enable_atoll=enable_atoll,
            targets=tuple(
                dict.fromkeys((*hot_targets, *explicit_targets, *call_edge_owner_targets))
            ),
            spawn_targets=options.get("spawn_targets", ()),
            call_edge_targets=options.get("call_edge_targets", ()),
        )
    )
    runs = (sampling_run, type_run)
    combined = _merge_payloads(sampling_payload, type_payload)
    if type_run.returncode != 0:
        return _profile_from_payload(
            combined,
            status="invalid",
            reason=f"type-observation profile exited with status {type_run.returncode}",
            launch_kind=launch_plan.launch_kind,
            runs=runs,
        )
    return _profile_from_payload(
        combined,
        status="profiled",
        reason="baseline profile collected",
        launch_kind=launch_plan.launch_kind,
        runs=runs,
    )


def select_profile_candidates(
    profile: ProfileResult,
    symbols: tuple[SymbolFact, ...],
) -> ProfileResult:
    """Map profiled members to static symbols and select hot baseline candidates.

    Args:
        profile: Baseline profile evidence to map and filter.
        symbols: Static symbol facts available for compilation planning.

    Returns:
        ProfileResult: The input profile with candidate decisions and selected symbols attached.
    """
    available = frozenset(symbol.id for symbol in symbols)
    if profile.total_samples < _MIN_TOTAL_SAMPLES:
        return replace(
            profile,
            status="static-fallback",
            reason=(
                "insufficient baseline profile samples: "
                f"observed {profile.total_samples}, required {_MIN_TOTAL_SAMPLES}"
            ),
            selected_hot_samples=0,
            selected_hot_coverage=0.0,
            candidates=tuple(
                MappedCandidateDecision(
                    symbol=member.symbol if member.symbol in available else None,
                    module=member.module,
                    qualname=member.qualname,
                    samples=member.samples,
                    coverage=member.coverage,
                    selected=False,
                    reason=("below-threshold" if member.symbol in available else "unmapped"),
                    scheduler_overhead_samples=member.scheduler_overhead_samples,
                    attributed_samples=member.attributed_samples,
                    attributed_coverage=_coverage(
                        member.attributed_samples,
                        profile.total_samples,
                    ),
                )
                for member in sorted(
                    profile.members,
                    key=lambda item: (-item.attributed_samples, item.module, item.qualname),
                )
            ),
            selected_symbols=(),
        )
    decisions: list[MappedCandidateDecision] = []
    selected_symbols: list[SymbolId] = []
    selected_samples = 0
    for member in sorted(
        profile.members,
        key=lambda item: (-item.attributed_samples, item.module, item.qualname),
    ):
        symbol = member.symbol if member.symbol in available else None
        reason = _candidate_reason(
            member=member,
            symbol=symbol,
            context=_CandidatePolicyContext(
                total_samples=profile.total_samples,
                mapped_activity_samples=(
                    profile.mapped_project_samples + profile.scheduler_overhead_samples
                ),
                selected_samples=selected_samples,
                selected_count=len(selected_symbols),
            ),
        )
        selected = reason == "selected"
        if selected:
            selected_symbols.append(member.symbol)
            selected_samples += member.attributed_samples
        decisions.append(
            MappedCandidateDecision(
                symbol=symbol,
                module=member.module,
                qualname=member.qualname,
                samples=member.samples,
                coverage=member.coverage,
                selected=selected,
                reason=reason,
                scheduler_overhead_samples=member.scheduler_overhead_samples,
                attributed_samples=member.attributed_samples,
                attributed_coverage=_coverage(
                    member.attributed_samples,
                    profile.total_samples,
                ),
            )
        )
    return replace(
        profile,
        selected_hot_samples=selected_samples,
        selected_hot_coverage=_coverage(
            selected_samples,
            profile.mapped_project_samples + profile.scheduler_overhead_samples,
        ),
        candidates=tuple(decisions),
        selected_symbols=tuple(selected_symbols),
    )


def _run_bootstrap_pass(request: _BootstrapRequest) -> tuple[JsonObject, SubprocessPassEvidence]:
    stem = f"atoll-profile-{os.getpid()}-{time.monotonic_ns()}-{request.profile_stage}"
    config_path = request.scratch_dir / f"{stem}.config.json"
    result_path = request.scratch_dir / f"{stem}.result.json"
    config = _bootstrap_config(
        request=request,
        result_path=result_path,
    )
    try:
        config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
        command = (sys.executable, str(_BOOTSTRAP_PATH), str(config_path))
        env = _profile_environment(
            payload_root=request.payload_root,
            enable_atoll=request.enable_atoll,
        )
        started = time.perf_counter()
        completed = _run_subprocess(
            _SubprocessInvocation(
                command=command,
                cwd=request.project_root,
                env=env,
                check=False,
                shell=False,
                capture_output=True,
                text=True,
            )
        )
        duration = time.perf_counter() - started
        payload = _read_json_object(result_path) if result_path.exists() else _empty_payload()
        return payload, SubprocessPassEvidence(
            pass_kind=request.profile_stage,
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_seconds=duration,
        )
    finally:
        config_path.unlink(missing_ok=True)
        result_path.unlink(missing_ok=True)


def _run_subprocess(invocation: _SubprocessInvocation) -> subprocess.CompletedProcess[str]:
    return _subprocess_run(
        invocation.command,
        cwd=invocation.cwd,
        env=invocation.env,
        check=invocation.check,
        shell=invocation.shell,
        capture_output=invocation.capture_output,
        text=invocation.text,
    )


def _bootstrap_config(request: _BootstrapRequest, *, result_path: Path) -> JsonObject:
    return {
        "profile_stage": request.profile_stage,
        "launch_kind": request.launch_plan.launch_kind,
        "target": request.launch_plan.target,
        "args": list(request.launch_plan.args),
        "project_root": str(request.project_root),
        "payload_root": str(request.payload_root),
        "module_paths": [[module, suffix] for module, suffix in request.module_paths],
        "result_path": str(result_path),
        "enable_atoll": request.enable_atoll,
        "targets": list(request.targets),
        "spawn_targets": [
            {
                "id": target.id,
                "owner": _member_key(target.owner),
                "lineno": target.lineno,
                "col_offset": target.col_offset,
                "scheduler_method": target.scheduler_method,
                "end_lineno": target.end_lineno,
                "end_col_offset": target.end_col_offset,
            }
            for target in request.spawn_targets
        ],
        "call_edge_targets": [
            {
                "id": target.id,
                "owner": _member_key(target.owner),
                "callee": _member_key(target.callee),
                "lineno": target.lineno,
                "col_offset": target.col_offset,
                "end_lineno": target.end_lineno,
                "end_col_offset": target.end_col_offset,
            }
            for target in request.call_edge_targets
        ],
    }


def _profile_environment(*, payload_root: Path, enable_atoll: bool) -> dict[str, str]:
    child_env = dict(os.environ)
    child_env["PYTHONDONTWRITEBYTECODE"] = "1"
    existing_pythonpath = tuple(
        path for path in child_env.get("PYTHONPATH", "").split(os.pathsep) if path
    )
    child_env["PYTHONPATH"] = os.pathsep.join((str(payload_root), *existing_pythonpath))
    if enable_atoll:
        child_env.pop("ATOLL_DISABLE", None)
    else:
        child_env["ATOLL_DISABLE"] = "1"
    child_env.pop("ATOLL_STRICT", None)
    child_env.pop("ATOLL_REQUIRE_COMPILED", None)
    child_env.pop("ATOLL_REQUIRE_OPTIMIZED", None)
    return child_env


def _launch_plan(command: tuple[str, ...]) -> _LaunchPlan | None:
    if len(command) < _SCRIPT_COMMAND_LENGTH or not _is_python_launcher(command[0]):
        return None
    if command[1] == "-m":
        if len(command) < _MODULE_COMMAND_LENGTH or not command[2]:
            return None
        return _LaunchPlan(launch_kind="module", target=command[2], args=command[3:])
    script = command[1]
    if script.endswith(".py"):
        return _LaunchPlan(launch_kind="script", target=script, args=command[2:])
    return None


def _is_python_launcher(value: str) -> bool:
    launcher = Path(value).name
    if launcher == Path(sys.executable).name or Path(value) == Path(sys.executable):
        return True
    if launcher in {"python", "python3"}:
        return True
    return launcher.startswith("python3.") and launcher[8:].isdigit()


def _profile_from_payload(
    payload: JsonObject,
    *,
    status: ProfileStatus,
    reason: str,
    launch_kind: Literal["script", "module"],
    runs: tuple[SubprocessPassEvidence, ...],
) -> ProfileResult:
    total_samples = _int_field(payload, "total_samples")
    sample_counts = _object_field(payload, "sample_counts")
    scheduler_overhead_counts = _object_field(payload, "scheduler_overhead_counts")
    signature_payload = _object_field(payload, "signatures")
    member_lifecycle_payload = _object_field(payload, "member_lifecycle")
    mapped_project_samples = sum(_int_value(value) for value in sample_counts.values())
    member_keys = frozenset(
        _string_key(key)
        for container in (
            sample_counts,
            scheduler_overhead_counts,
            signature_payload,
            member_lifecycle_payload,
        )
        for key in container
    )
    members = tuple(
        _profiled_member(
            key,
            _int_value(sample_counts.get(key, 0)),
            _ProfileMemberPayloadContext(
                total_samples=total_samples,
                scheduler_overhead_counts=scheduler_overhead_counts,
                signature_payload=signature_payload,
                member_lifecycle_payload=member_lifecycle_payload,
            ),
        )
        for key in sorted(
            member_keys,
            key=lambda item: (
                -(
                    _int_value(sample_counts.get(item, 0))
                    + _int_value(scheduler_overhead_counts.get(item, 0))
                ),
                item,
            ),
        )
    )
    return ProfileResult(
        status=status,
        reason=reason,
        launch_kind=launch_kind,
        total_samples=total_samples,
        mapped_project_samples=mapped_project_samples,
        mapped_coverage=_coverage(mapped_project_samples, total_samples),
        scheduler_overhead_samples=sum(
            _int_value(value) for value in scheduler_overhead_counts.values()
        ),
        scheduler_overhead_coverage=_coverage(
            sum(_int_value(value) for value in scheduler_overhead_counts.values()),
            total_samples,
        ),
        selected_hot_samples=0,
        selected_hot_coverage=0.0,
        runs=runs,
        lifecycle=_lifecycle_counts(_object_field(payload, "lifecycle")),
        members=members,
        candidates=(),
        selected_symbols=(),
        spawn_sites=tuple(
            _profiled_spawn_site(_object_value(item))
            for item in _list_value(payload.get("spawn_sites", []))
        ),
        call_edges=tuple(
            _profiled_call_edge(_object_value(item))
            for item in _list_value(payload.get("call_edges", []))
        ),
    )


def _profiled_member(
    key: str,
    samples: int,
    context: _ProfileMemberPayloadContext,
) -> ProfiledMember:
    module, qualname = _split_member_key(key)
    member_payload = _object_value(context.signature_payload.get(key, {}))
    return ProfiledMember(
        module=module,
        qualname=qualname,
        samples=samples,
        coverage=_coverage(samples, context.total_samples),
        scheduler_overhead_samples=_int_value(context.scheduler_overhead_counts.get(key, 0)),
        scheduler_overhead_coverage=_coverage(
            _int_value(context.scheduler_overhead_counts.get(key, 0)), context.total_samples
        ),
        call_count=_int_field(member_payload, "call_count"),
        invocation_count=_int_field(member_payload, "invocation_count"),
        lifecycle=_lifecycle_counts(_object_value(context.member_lifecycle_payload.get(key, {}))),
        signatures=_signatures(_list_value(member_payload.get("signatures", []))),
        polymorphic_overflow=_bool_value(member_payload.get("polymorphic_overflow", False)),
        observation_capped=_bool_value(member_payload.get("observation_capped", False)),
        completed_calls=_int_field(member_payload, "completed_calls"),
        max_active_calls=_int_field(member_payload, "max_active_calls"),
        pre_completion_suspensions=_int_field(member_payload, "pre_completion_suspensions"),
    )


def _profiled_spawn_site(payload: JsonObject) -> ProfiledSpawnSite:
    owner_module, owner_qualname = _split_member_key(_string_field(payload, "owner"))
    callable_identities = _object_field(payload, "callable_identities")
    return ProfiledSpawnSite(
        target=ProfileSpawnSiteTarget(
            id=_string_field(payload, "id"),
            owner=SymbolId(module=owner_module, qualname=owner_qualname),
            lineno=_int_field(payload, "lineno"),
            col_offset=_int_field(payload, "col_offset"),
            scheduler_method=_string_field(payload, "scheduler_method"),
            end_lineno=_optional_int_field(payload, "end_lineno"),
            end_col_offset=_optional_int_field(payload, "end_col_offset"),
        ),
        invocation_count=_int_field(payload, "invocation_count"),
        callable_identities=tuple(
            CanonicalCallableCount(identity=_string_key(identity), count=_int_value(count))
            for identity, count in sorted(callable_identities.items())
        ),
    )


def _profiled_call_edge(payload: JsonObject) -> ProfiledCallEdge:
    owner_module, owner_qualname = _split_member_key(_string_field(payload, "owner"))
    callee_module, callee_qualname = _split_member_key(_string_field(payload, "callee"))
    return ProfiledCallEdge(
        target=ProfileCallEdgeTarget(
            id=_string_field(payload, "id"),
            owner=SymbolId(module=owner_module, qualname=owner_qualname),
            callee=SymbolId(module=callee_module, qualname=callee_qualname),
            lineno=_int_field(payload, "lineno"),
            col_offset=_int_field(payload, "col_offset"),
            end_lineno=_optional_int_field(payload, "end_lineno"),
            end_col_offset=_optional_int_field(payload, "end_col_offset"),
        ),
        invocation_count=_int_field(payload, "invocation_count"),
    )


def _signatures(items: list[JsonValue]) -> tuple[ObservedSignature, ...]:
    signatures: list[ObservedSignature] = []
    for item in items:
        payload = _object_value(item)
        parameters = tuple(
            CanonicalTypeObservation(
                parameter_name=_string_field(_object_value(parameter), "parameter_name"),
                type_path=_string_field(_object_value(parameter), "type_path"),
                count=_int_field(_object_value(parameter), "count"),
            )
            for parameter in _list_value(payload.get("parameters", []))
        )
        signatures.append(
            ObservedSignature(
                parameters=parameters,
                count=_int_field(payload, "count"),
            )
        )
    return tuple(signatures)


def _lifecycle_counts(payload: JsonObject) -> LifecycleCounts:
    return LifecycleCounts(
        start=_int_field(payload, "start"),
        return_=_int_field(payload, "return"),
        yield_=_int_field(payload, "yield"),
        resume=_int_field(payload, "resume"),
        unwind=_int_field(payload, "unwind"),
        throw=_int_field(payload, "throw"),
    )


def _candidate_reason(
    *,
    member: ProfiledMember,
    symbol: SymbolId | None,
    context: _CandidatePolicyContext,
) -> CandidateDecisionReason:
    if symbol is None:
        return "unmapped"
    if (
        context.total_samples < _MIN_TOTAL_SAMPLES
        or member.attributed_samples < _MIN_CANDIDATE_SAMPLES
        or _coverage(member.attributed_samples, context.total_samples) < _MIN_CANDIDATE_SHARE
    ):
        return "below-threshold"
    if context.selected_count >= _MAX_SELECTED_CANDIDATES:
        return "limit"
    if (
        context.selected_count > 0
        and context.selected_samples >= context.mapped_activity_samples * _TARGET_MAPPED_COVERAGE
    ):
        return "coverage-reached"
    return "selected"


def _hot_member_keys(payload: JsonObject) -> tuple[tuple[str, int], ...]:
    sample_counts = _object_field(payload, "sample_counts")
    scheduler_counts = _object_field(payload, "scheduler_overhead_counts")
    members = frozenset((*sample_counts, *scheduler_counts))
    return tuple(
        sorted(
            (
                (
                    _string_key(key),
                    _int_value(sample_counts.get(key, 0))
                    + _int_value(scheduler_counts.get(key, 0)),
                )
                for key in members
            ),
            key=lambda item: (-item[1], item[0]),
        )[:_MAX_SELECTED_CANDIDATES]
    )


def _merge_payloads(sampling_payload: JsonObject, type_payload: JsonObject) -> JsonObject:
    merged = dict(sampling_payload)
    merged["scheduler_overhead_counts"] = sampling_payload.get("scheduler_overhead_counts", {})
    merged["signatures"] = type_payload.get("signatures", {})
    merged["lifecycle"] = type_payload.get("lifecycle", {})
    merged["member_lifecycle"] = type_payload.get("member_lifecycle", {})
    merged["spawn_sites"] = type_payload.get("spawn_sites", [])
    merged["call_edges"] = type_payload.get("call_edges", [])
    return merged


def _empty_payload() -> JsonObject:
    return {
        "total_samples": 0,
        "sample_counts": {},
        "scheduler_overhead_counts": {},
        "lifecycle": {"start": 0, "return": 0, "yield": 0, "resume": 0, "unwind": 0, "throw": 0},
        "member_lifecycle": {},
        "signatures": {},
        "spawn_sites": [],
        "call_edges": [],
    }


def _member_key(symbol: SymbolId) -> str:
    return f"{symbol.module}::{symbol.qualname}"


def _read_json_object(path: Path) -> JsonObject:
    return _object_value(cast(JsonValue, json.loads(path.read_text(encoding="utf-8"))))


def _split_member_key(key: str) -> tuple[str, str]:
    module, separator, qualname = key.partition("::")
    if not separator:
        return "", key
    return module, qualname


def _coverage(samples: int, total_samples: int) -> float:
    if total_samples <= 0:
        return 0.0
    return samples / total_samples


def _object_field(payload: JsonObject, key: str) -> JsonObject:
    return _object_value(payload.get(key, {}))


def _string_field(payload: JsonObject, key: str) -> str:
    value = payload.get(key, "")
    return value if isinstance(value, str) else ""


def _int_field(payload: JsonObject, key: str) -> int:
    return _int_value(payload.get(key, 0))


def _optional_int_field(payload: JsonObject, key: str) -> int | None:
    value = payload.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _int_value(value: JsonValue) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _bool_value(value: JsonValue) -> bool:
    return value if isinstance(value, bool) else False


def _object_value(value: JsonValue) -> JsonObject:
    if isinstance(value, dict):
        return value
    return {}


def _list_value(value: JsonValue) -> list[JsonValue]:
    if isinstance(value, list):
        return value
    return []


def _string_key(value: object) -> str:
    return value if isinstance(value, str) else ""
