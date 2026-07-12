"""Immutable source-optimization contracts for 3x execution-plan lowering.

The models in this module describe source-level transformations that can be
derived from an execution plan and then assessed or trialed independently of
runtime profile counts. Stable plan identity is deliberately content-addressed
from static source and lowering inputs only so profiling can reprioritize a plan
without invalidating caches or patch records.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from atoll.models import SymbolId

SourceTransformationKind = Literal[
    "private-transport-batch-drain",
    "quiescent-callable-execution",
    "local-state-machine-fusion",
    "private-protocol-auto-forwarding",
]
SourceOptimizationAssessmentStatus = Literal[
    "unbenchmarked",
    "trial-ready",
    "partial",
    "unsupported",
]
SourceOptimizationTrialStatus = Literal[
    "not-run",
    "accepted",
    "rejected",
    "not-profitable",
    "failed-semantics",
    "unavailable",
]
SourceOptimizationApplicationStatus = Literal[
    "not-applied",
    "applied",
    "conflicted",
    "rolled-back",
    "stale-source",
    "failed",
    "unavailable",
]
SourceAccessKind = Literal[
    "read",
    "write",
    "call",
    "await",
    "iterate",
    "transport-send",
    "transport-receive",
    "transport-drain",
    "protocol-forward",
]
SourceOptimizationHazard = Literal[
    "public-transport",
    "shared-mutable-state",
    "escaping-callable",
    "dynamic-dispatch",
    "observable-ordering",
    "unknown-side-effect",
    "suspension",
    "task-introspection",
    "cancellation",
    "context-mutation",
    "unknown-dynamic-call",
]

_DIGEST_SIZE = 16


@dataclass(frozen=True, slots=True)
class SourceOptimizationIdentity:
    """Static content identity for one source-optimization plan.

    Runtime assessment counts, trial medians, and profile rankings are excluded
    from this object. Callers may pass source hashes or transformation versions
    in any order; `stable_source_optimization_plan_id` canonicalizes them before
    hashing.

    Attributes:
        execution_plan_id: Stable execution-plan identifier that produced the source plan.
        source_hashes: Per-source-file content hashes covered by the transformation plan.
        topology_fingerprint: Stable execution-plan topology digest.
        dialect: Scheduler dialect whose semantics the source transformation preserves.
        lowering_version: Source-lowering version that changes generated patch semantics.
        python_abi: Python ABI tag or interpreter compatibility boundary.
        transformation_versions: Static version for each transformation kind in the plan.
    """

    execution_plan_id: str
    source_hashes: tuple[tuple[PurePosixPath, str], ...]
    topology_fingerprint: str
    dialect: str
    lowering_version: str
    python_abi: str
    transformation_versions: tuple[tuple[SourceTransformationKind, str], ...]


def stable_source_optimization_plan_id(identity: SourceOptimizationIdentity) -> str:
    """Return a deterministic source-optimization plan identifier.

    Args:
        identity: Static execution-plan, source, topology, dialect, ABI, and
            transformation-version inputs that define patch compatibility.

    Returns:
        str: Short content-addressed source-optimization identifier that is
        independent of runtime assessment counts and benchmark measurements.
    """
    digest = hashlib.blake2b(digest_size=_DIGEST_SIZE)
    for part in (
        identity.execution_plan_id,
        identity.topology_fingerprint,
        identity.dialect,
        identity.lowering_version,
        identity.python_abi,
    ):
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    for source_path, source_hash in sorted(
        identity.source_hashes,
        key=lambda item: (item[0].as_posix(), item[1]),
    ):
        digest.update(source_path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(source_hash.encode("utf-8"))
        digest.update(b"\0")
    for kind, version in sorted(identity.transformation_versions):
        digest.update(kind.encode("utf-8"))
        digest.update(b"\0")
        digest.update(version.encode("utf-8"))
        digest.update(b"\0")
    return f"source-opt-{digest.hexdigest()}"


@dataclass(frozen=True, slots=True)
class SourceAccessSite:
    """One static source access site involved in a source optimization.

    Attributes:
        path: POSIX source path containing the access site.
        symbol: Symbol that owns the access site, when source resolution succeeded.
        kind: Access operation observed at the site.
        lineno: One-based source line for the access.
        expression: Stable expression, attribute, or transport name used in reports.
        hazards: Conservative static hazards that can block or qualify a transformation.
    """

    path: PurePosixPath
    symbol: SymbolId | None
    kind: SourceAccessKind
    lineno: int
    expression: str
    hazards: tuple[SourceOptimizationHazard, ...] = ()


@dataclass(frozen=True, slots=True)
class SourceCallableEvidence:
    """Runtime and static evidence for a callable touched by a source plan.

    Attributes:
        symbol: Callable represented by this evidence.
        static_role: Plan-facing role such as owner, worker, consumer, or reducer.
        observed_invocations: Runtime invocation count used for assessment only.
        completed_calls: Invocations observed through normal return or unwind.
        static_suspension_points: Suspension syntax found in the scanned declaration.
        observed_suspensions: Runtime pre-completion suspension events.
        immediate_result_ratio: Conservative fraction of completed invocations that did not
            suspend.
        median_seconds: Optional median runtime attributed to the callable.
        hot_share: Fraction of mapped runtime attributed to the callable.
        scheduler_overhead_samples: Nested scheduler or library samples attributed to the
            callable's active stack.
        task_introspection: Static task-identity or task-metadata reads.
        cancellation: Static cancellation APIs referenced by the callable.
        context_mutation: Static direct or indirect context-local mutation evidence.
        unknown_dynamic_calls: Calls whose runtime target could not be proven statically.
        hazards: Conservative static hazards associated with the callable.
    """

    symbol: SymbolId
    static_role: str
    observed_invocations: int = 0
    completed_calls: int = 0
    static_suspension_points: int = 0
    observed_suspensions: int = 0
    immediate_result_ratio: float = 0.0
    median_seconds: float | None = None
    hot_share: float = 0.0
    scheduler_overhead_samples: int = 0
    task_introspection: tuple[str, ...] = ()
    cancellation: tuple[str, ...] = ()
    context_mutation: tuple[str, ...] = ()
    unknown_dynamic_calls: tuple[str, ...] = ()
    hazards: tuple[SourceOptimizationHazard, ...] = ()


@dataclass(frozen=True, slots=True)
class TransformationStep:
    """One ordered source transformation inside a source-optimization plan.

    Attributes:
        kind: Transformation family applied by this step.
        version: Step-specific transformation version that participates in identity.
        source_symbol: Primary symbol read or rewritten by the step.
        target_symbol: Generated or rewritten symbol produced by the step.
        access_sites: Static accesses the step relies on or rewrites.
        semantic_boundary: Invariant name preserved by the step.
        description: Human-readable report text for the transformation.
    """

    kind: SourceTransformationKind
    version: str
    source_symbol: SymbolId
    target_symbol: SymbolId | None
    access_sites: tuple[SourceAccessSite, ...]
    semantic_boundary: str
    description: str

    @property
    def stable_id(self) -> str:
        """Return the report-facing identity for this static transformation step.

        Returns:
            str: Kind, version, and source symbol joined into a deterministic identifier.
        """
        return f"{self.kind}:{self.version}:{self.source_symbol.stable_id}"


@dataclass(frozen=True, slots=True)
class SourceEdit:
    """One source file edit generated by a source-optimization trial.

    Attributes:
        path: POSIX source path changed by the edit.
        before_hash: Digest of the source before the edit, or `None` for additions.
        after_hash: Digest of the source after the edit.
        summary: Stable summary of the generated change.
        touched_symbols: Symbols whose definitions or call sites were edited.
        transformation_id: Stable transformation step that produced the edit.
        start_line: One-based first changed source line, when known.
        end_line: One-based final changed source line, when known.
    """

    path: PurePosixPath
    before_hash: str | None
    after_hash: str
    summary: str
    touched_symbols: tuple[SymbolId, ...] = ()
    transformation_id: str | None = None
    start_line: int | None = None
    end_line: int | None = None


@dataclass(frozen=True, slots=True)
class SourceOptimizationPlan:
    """Source-level 3x optimization plan derived from an execution plan.

    Attributes:
        id: Deterministic plan identifier returned by `stable_source_optimization_plan_id`.
        identity: Static identity inputs that define patch compatibility.
        source: POSIX source path that owns the entrypoint orchestration site.
        owner: Orchestration owner symbol.
        worker: Worker callable transformed by the plan.
        consumer: Result consumer callable, when distinct from the owner.
        reducer: Reduction callable, when reduction is part of the optimized boundary.
        transport: Private transport expression or name used by the plan.
        access_sites: Static source accesses used to prove privacy and semantics.
        entrypoint: Callable used to enter the optimized source path.
        steps: Ordered transformations that make up the source patch.
        semantic_boundaries: Named invariants the plan must preserve.
    """

    id: str
    identity: SourceOptimizationIdentity
    source: PurePosixPath
    owner: SymbolId
    worker: SymbolId
    consumer: SymbolId | None
    reducer: SymbolId | None
    transport: str
    access_sites: tuple[SourceAccessSite, ...]
    entrypoint: SymbolId
    steps: tuple[TransformationStep, ...]
    semantic_boundaries: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceOptimizationAssessment:
    """Capability and profitability assessment for one source-optimization plan.

    Runtime counts live here rather than in `SourceOptimizationIdentity`, so
    repeated profiling can change rankings without changing stable plan IDs.

    Attributes:
        plan_id: Stable source-optimization plan identifier.
        status: Assessment outcome before trial execution.
        minimum_speedup: Required trial speedup for the source patch to be profitable.
        work_items: Static callable identities expected to benefit from the optimization.
        observed_work_items: Runtime work-item count represented by the plan.
        immediate_result_ratio: Conservative fraction of work that completed without suspension.
        attributed_hot_share: Fraction of total observed wall time represented by project and
            nested scheduler samples attributed to this plan.
        scheduler_overhead_samples: Nested scheduler or library samples attributed to the plan.
        scheduler_overhead_share: Fraction of total samples represented by that overhead.
        scheduler_overhead_evidence: Normalized evidence strings for scheduler overhead.
        callable_evidence: Per-callable static and runtime evidence used in the assessment.
        rejections: Deterministic rejection reasons or guarded caveats.
        headroom_speedup: Optional measured ceiling speedup for this exact plan and workload.
    """

    plan_id: str
    status: SourceOptimizationAssessmentStatus
    minimum_speedup: float
    work_items: tuple[SymbolId, ...]
    observed_work_items: int
    immediate_result_ratio: float
    attributed_hot_share: float
    scheduler_overhead_samples: int
    scheduler_overhead_share: float
    scheduler_overhead_evidence: tuple[str, ...]
    callable_evidence: tuple[SourceCallableEvidence, ...]
    rejections: tuple[str, ...] = ()
    headroom_speedup: float | None = None


@dataclass(frozen=True, slots=True)
class SourceOptimizationTrial:
    """Semantic and benchmark result for an attempted source optimization.

    Attributes:
        plan_id: Stable source-optimization plan identifier.
        status: Trial outcome after commands and benchmarks run.
        semantic_command: Command used to validate patched source behavior.
        benchmark_command: Command used to measure source and wheel performance.
        baseline_median_seconds: Median unoptimized source runtime.
        source_median_seconds: Median optimized source runtime.
        wheel_median_seconds: Median wheel or packaged optimized runtime.
        source_speedup: Baseline median divided by optimized source median.
        wheel_speedup: Baseline median divided by optimized wheel median.
        patch_path: Filesystem path to the generated patch file, when one exists.
        source_edits: Source edits represented by the patch.
        application_status: Whether the patch was applied to the working tree.
        diagnostics: Human-readable semantic, benchmark, or application diagnostics.
        candidate_id: Stable candidate-combination identity used by bounded search.
        transformation_ids: Ordered transformation steps enabled for this candidate.
        reason: Plain-language acceptance or rejection reason.
        semantic_exit_code: Semantic command exit status, when executed.
        semantic_duration_seconds: Parent-observed semantic command duration.
        current_median_seconds: Current accepted-candidate median before this trial.
    """

    plan_id: str
    status: SourceOptimizationTrialStatus
    semantic_command: tuple[str, ...]
    benchmark_command: tuple[str, ...]
    baseline_median_seconds: float | None
    source_median_seconds: float | None
    wheel_median_seconds: float | None
    source_speedup: float | None
    wheel_speedup: float | None
    patch_path: Path | None
    source_edits: tuple[SourceEdit, ...]
    application_status: SourceOptimizationApplicationStatus
    diagnostics: tuple[str, ...] = ()
    candidate_id: str = ""
    transformation_ids: tuple[str, ...] = ()
    reason: str = ""
    semantic_exit_code: int | None = None
    semantic_duration_seconds: float | None = None
    current_median_seconds: float | None = None
