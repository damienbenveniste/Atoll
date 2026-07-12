"""Immutable execution-plan models for scheduler-aware lowering.

Execution plans describe hot orchestration sites that can be lowered without
mutating the existing compiler-backend model. Their identities are content
addressed from source, call-site topology, dialect, and lowering version so
dynamic profile counts can rank plans without changing cache keys.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from atoll.models import SymbolId

ExecutionPlanAssessmentStatus = Literal["supported", "partial", "unsupported"]
ExecutionPlanDiagnosticSeverity = Literal["error", "warning", "note"]
ExecutionPlanNodeRole = Literal[
    "orchestrator",
    "producer",
    "consumer",
    "worker",
    "transport",
    "reducer",
    "wrapper",
    "report",
]
ExecutionPlanEdgeKind = Literal[
    "spawns",
    "passes_transport",
    "produces",
    "delivers",
    "reduces",
    "reports",
]
ExecutionPlanGuardKind = Literal["scheduler", "transport", "topology", "semantics"]
ExecutionPlanRejectionReason = Literal[
    "ambiguous-spawn",
    "public-transport",
    "multiple-consumer",
    "unknown-transport",
    "low-hotness",
    "coverage-reached",
    "selection-limit",
    "unstructured-task",
    "unknown-capacity",
    "dynamic-scheduler",
    "escaping-handle",
]
ExecutionPlanTrialStatus = Literal["accepted", "rejected", "failed-semantics", "unavailable"]
ExecutionPlanBenchmarkStatus = Literal[
    "not-run",
    "passed",
    "not-profitable",
    "invalid",
    "unavailable",
    "unbenchmarked",
]
ExecutionPlanCacheStatus = Literal["not-run", "hit", "miss", "invalid"]

_DIGEST_SIZE = 16


@dataclass(frozen=True, slots=True)
class ExecutionPlanIdentity:
    """Inputs that define a stable execution-plan identity.

    Attributes:
        source_module: Importable module containing the orchestration site.
        source_hash: Digest of source text relevant to the plan.
        callsite_fingerprint: Stable call-site coordinate and target digest.
        topology_fingerprint: Stable node and edge digest for the discovered graph.
        dialect: Scheduler dialect identifier used by the plan.
        lowering_version: Dialect lowering version that affects generated code.
        guarded_callable_identities: Canonical scheduler callable identities guarded at runtime.
    """

    source_module: str
    source_hash: str
    callsite_fingerprint: str
    topology_fingerprint: str
    dialect: str
    lowering_version: str
    guarded_callable_identities: tuple[str, ...] = ()


def stable_execution_plan_id(
    identity: ExecutionPlanIdentity,
) -> str:
    """Return a deterministic execution-plan identifier.

    Args:
        identity: Static source, topology, dialect, and lowering identity inputs.

    Returns:
        str: Short content-addressed plan identifier independent of profile counts.
    """
    digest = hashlib.blake2b(digest_size=_DIGEST_SIZE)
    for part in (
        identity.source_module,
        identity.source_hash,
        identity.callsite_fingerprint,
        identity.topology_fingerprint,
        identity.dialect,
        identity.lowering_version,
        *identity.guarded_callable_identities,
    ):
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return f"exec-plan-{digest.hexdigest()}"


@dataclass(frozen=True, slots=True)
class PlanNode:
    """One callable or report node in an execution plan.

    Attributes:
        id: Stable module-qualified node identity.
        symbol: Static symbol represented by the node, when source resolution succeeded.
        role: Plan role used by topology checks and reports.
        lineno: One-based source line for the node's relevant declaration or call site.
    """

    id: str
    symbol: SymbolId | None
    role: ExecutionPlanNodeRole
    lineno: int


@dataclass(frozen=True, slots=True)
class PlanEdge:
    """Directed relation between execution-plan nodes.

    Attributes:
        src: Source node identifier.
        dst: Destination node identifier.
        kind: Scheduler, transport, or report relation represented by the edge.
        transport: Private transport variable or stream endpoint proving producer-consumer shape.
        lineno: One-based source line where the relation was discovered.
    """

    src: str
    dst: str
    kind: ExecutionPlanEdgeKind
    transport: str | None
    lineno: int


@dataclass(frozen=True, slots=True)
class PlanGuard:
    """Runtime or lowering invariant that must hold for a staged plan.

    Attributes:
        kind: Guard category.
        expression: Stable textual predicate or invariant name.
        message: Human-readable explanation for reports and diagnostics.
    """

    kind: ExecutionPlanGuardKind
    expression: str
    message: str


@dataclass(frozen=True, slots=True)
class PlanRejection:
    """Report-only execution-plan rejection evidence.

    Attributes:
        id: Stable rejection identifier.
        source_module: Importable module containing the rejected site.
        owner: Static owner symbol for the rejected orchestration site.
        reason: Deterministic rejection category.
        message: Human-readable explanation of the rejected site.
        dialect: Scheduler dialect involved, when recognition reached that point.
        lineno: One-based source line for the rejected site.
        hotness: Dynamic hotness used only for ranking and explanation.
    """

    id: str
    source_module: str
    owner: SymbolId
    reason: ExecutionPlanRejectionReason
    message: str
    dialect: str | None
    lineno: int
    hotness: int


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Scheduler-aware lowering plan discovered from source and ranked by profile data.

    Attributes:
        id: Deterministic plan ID independent of dynamic profile counts.
        source_module: Importable module containing the orchestration site.
        owner: Static owner symbol for the orchestration site.
        dialect: Scheduler dialect identifier.
        lowering_version: Dialect lowering version that affects generated code.
        source_hash: Digest of the relevant source text, not the runtime profile.
        source_hashes: Complete per-module source digests covered by `source_hash`.
        source_members: Exact declarations covered by source_hash.
        callsite_fingerprint: Stable call-site coordinate and target digest.
        topology_fingerprint: Stable node and edge digest for the plan graph.
        nodes: Plan nodes in deterministic order.
        edges: Plan edges in deterministic order.
        guards: Required lowering and runtime invariants.
        completion_transport: Private result-transport identity.
        consumer: Callable that owns result delivery.
        reducer: Callable that performs or owns final reduction.
        transport_capacity: Statically known dialect-defined transport capacity.
        ordering_policy: Statically preserved result-delivery ordering policy.
        task_ownership: Static task-handle ownership proof used by backend assessment.
        observed_invocations: Profiled spawned-callable invocations used for ranking.
        lifecycle_starts: Profiled coroutine lifecycle starts attributed to the plan.
        lifecycle_share: Fraction of mapped async lifecycle starts attributed to the plan.
        guarded_callable_identities: Canonical scheduler callable identities required by the plan.
        rejections: Report-only rejection evidence associated with this plan.
        hotness: Dynamic profile hotness used for selection only.
    """

    id: str
    source_module: str
    owner: SymbolId
    dialect: str
    lowering_version: str
    source_hash: str
    callsite_fingerprint: str
    topology_fingerprint: str
    nodes: tuple[PlanNode, ...]
    edges: tuple[PlanEdge, ...]
    guards: tuple[PlanGuard, ...]
    completion_transport: str | None = None
    consumer: SymbolId | None = None
    reducer: SymbolId | None = None
    transport_capacity: int | None = None
    ordering_policy: str = "completion-order"
    task_ownership: str = "structured"
    observed_invocations: int = 0
    lifecycle_starts: int = 0
    lifecycle_share: float = 0.0
    guarded_callable_identities: tuple[str, ...] = ()
    source_members: tuple[SymbolId, ...] = ()
    source_hashes: tuple[tuple[str, str], ...] = ()
    rejections: tuple[PlanRejection, ...] = ()
    hotness: int = 0


@dataclass(frozen=True, slots=True)
class ChangedPayloadFile:
    """One staged payload file changed by an execution-plan backend.

    Attributes:
        install_path: POSIX path below the staged payload root.
        before_hash: Digest of the original payload file, or `None` for additions.
        after_hash: Digest of the staged payload file.
        role: Backend-defined role such as source, support, manifest, or report.
    """

    install_path: PurePosixPath
    before_hash: str | None
    after_hash: str
    role: str


@dataclass(frozen=True, slots=True)
class ExecutionPlanAssessment:
    """Backend capability decision for one execution plan.

    Attributes:
        plan_id: Stable execution-plan identifier.
        backend: Execution-plan backend identifier.
        status: Supported, partial, or unsupported decision.
        supported_nodes: Node IDs the backend can lower.
        unsupported_nodes: Node IDs the backend rejected.
        reasons: Deterministic explanation strings.
    """

    plan_id: str
    backend: str
    status: ExecutionPlanAssessmentStatus
    supported_nodes: tuple[str, ...]
    unsupported_nodes: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExecutionPlanAssessmentContext:
    """Read-only context passed to execution-plan backend assessment.

    Attributes:
        project_root: Root directory of the target project.
        source_root: Source root containing the plan module.
        profile_status: Status string from the profiling pass that ranked the plan.
    """

    project_root: Path
    source_root: Path
    profile_status: str


@dataclass(frozen=True, slots=True)
class ExecutionPlanStageContext:
    """Filesystem boundary for staging an accepted execution plan.

    Attributes:
        project_root: Root directory of the target project.
        payload_root: Directory where backend output should be staged.
        cache_root: Directory available for backend-local cache data.
    """

    project_root: Path
    payload_root: Path
    cache_root: Path


@dataclass(frozen=True, slots=True)
class StagedExecutionPlan:
    """Execution-plan backend output ready for package or trial use.

    Attributes:
        plan: Source execution plan that was staged.
        backend: Execution-plan backend identifier.
        payload_files: Files created or changed under the payload root.
        required_imports: Importable support modules required by the staged plan.
        guards: Guards that must be enforced when the payload is used.
    """

    plan: ExecutionPlan
    backend: str
    payload_files: tuple[ChangedPayloadFile, ...]
    required_imports: tuple[str, ...]
    guards: tuple[PlanGuard, ...]


@dataclass(frozen=True, slots=True)
class ExecutionPlanTrial:
    """Semantic or performance trial result for a staged execution plan.

    Attributes:
        plan_id: Stable execution-plan identifier.
        status: Trial outcome.
        command: Normalized command argument vector used for the trial.
        exit_code: Trial process exit status, when a process was run.
        duration_seconds: Parent-observed trial duration, when measured.
        diagnostics: Normalized backend or trial diagnostics.
        backend: Execution-plan backend that staged the candidate.
        reason: Plain-language acceptance or rejection reason.
        benchmark_command: Exact benchmark argv used for marginal comparison.
        benchmark_status: Marginal benchmark decision, or `not-run`.
        minimum_speedup: Required speedup over the current unplanned payload.
        minimum_overall_speedup: Required speedup over the interpreted baseline.
        baseline_median_seconds: Median duration of the interpreted baseline payload.
        unplanned_median_seconds: Median duration of the current accepted payload.
        planned_median_seconds: Median duration of the candidate planned payload.
        marginal_speedup: Unplanned-payload median divided by planned-payload median.
        overall_speedup: Baseline-payload median divided by planned-payload median.
        cache_status: Whether backend staging restored or generated the payload changes.
        payload_files: Staged payload changes validated before semantic testing.
    """

    plan_id: str
    status: ExecutionPlanTrialStatus
    command: tuple[str, ...]
    exit_code: int | None
    duration_seconds: float | None
    diagnostics: tuple[ExecutionPlanDiagnostic, ...] = ()
    backend: str | None = None
    reason: str | None = None
    benchmark_command: tuple[str, ...] = ()
    benchmark_status: ExecutionPlanBenchmarkStatus = "not-run"
    minimum_speedup: float | None = None
    minimum_overall_speedup: float | None = None
    baseline_median_seconds: float | None = None
    unplanned_median_seconds: float | None = None
    planned_median_seconds: float | None = None
    marginal_speedup: float | None = None
    overall_speedup: float | None = None
    cache_status: ExecutionPlanCacheStatus = "not-run"
    payload_files: tuple[ChangedPayloadFile, ...] = ()


@dataclass(frozen=True, slots=True)
class ExecutionPlanDiagnostic:
    """Normalized execution-plan backend or trial diagnostic.

    Attributes:
        code: Stable diagnostic code.
        severity: Diagnostic severity.
        message: Human-readable diagnostic message.
        details: Deterministic supporting detail lines.
    """

    code: str
    severity: ExecutionPlanDiagnosticSeverity
    message: str
    details: tuple[str, ...] = ()
