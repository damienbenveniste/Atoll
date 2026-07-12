"""Form report-only source-optimization plans from scheduler execution plans.

This module owns static source facts and dynamic attribution needed before any
source rewrite is attempted. It does not import target modules, mutate source,
generate patches, or claim profitability. Execution-plan topology supplies the
private pipeline boundary, while profile counts decide only ranking and whether
a plan is credible enough for a later disposable trial.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from atoll.execution_plans.models import ExecutionPlan, PlanRejection
from atoll.models import CompileConfig, ModuleScan, SymbolId, SymbolRecord
from atoll.runtime.profiling import ProfiledMember, ProfileResult
from atoll.source_optimization.models import (
    SourceAccessKind,
    SourceAccessSite,
    SourceCallableEvidence,
    SourceOptimizationAssessment,
    SourceOptimizationAssessmentStatus,
    SourceOptimizationHazard,
    SourceOptimizationIdentity,
    SourceOptimizationPlan,
    SourceTransformationKind,
    TransformationStep,
    stable_source_optimization_plan_id,
)

SOURCE_OPTIMIZATION_LOWERING_VERSION = "source-optimization-analysis-v1"
MINIMUM_SOURCE_OPTIMIZATION_SPEEDUP = 3.0
MINIMUM_OBSERVED_WORK_ITEMS = 10_000
MINIMUM_ATTRIBUTED_HOT_SHARE = 0.70
MAX_SOURCE_OPTIMIZATION_PLANS = 2

_TRANSFORMATION_VERSIONS: tuple[tuple[SourceTransformationKind, str], ...] = (
    ("private-transport-batch-drain", "batch-drain-v1"),
    ("quiescent-callable-execution", "quiescent-callable-v1"),
    ("local-state-machine-fusion", "state-machine-v1"),
    ("private-protocol-auto-forwarding", "protocol-forward-v1"),
)
_SEMANTIC_BOUNDARIES = (
    "public return values and exception types remain unchanged",
    "each logical work item retains isolated context-local state",
    "private transport capacity and completion ordering remain unchanged",
    "public signatures, descriptors, class identities, and fallback APIs remain unchanged",
    "all runtime guards pass before the first transformed side effect",
    "optimized execution never retries interpreted work after entry",
)
_RECEIVE_METHODS = frozenset({"get", "receive", "__anext__"})
_DRAIN_METHODS = frozenset({"get_nowait", "receive_nowait"})
_SEND_METHODS = frozenset({"put", "put_nowait", "send", "send_nowait"})
_TASK_INTROSPECTION_METHODS = frozenset(
    {"all_tasks", "current_task", "get_coro", "get_name", "set_name", "get_stack"}
)
_CANCELLATION_METHODS = frozenset(
    {"cancel", "cancelled", "cancelling", "uncancel", "shield", "CancelScope"}
)
_DYNAMIC_CALLS = frozenset(
    {"__import__", "delattr", "eval", "exec", "getattr", "globals", "locals", "setattr", "vars"}
)


@dataclass(frozen=True, slots=True)
class SourceOptimizationPlanningResult:
    """Report-only source plans and their profile/static assessments.

    Attributes:
        plans: At most two static plans ranked by attributable wall-time share.
        assessments: One assessment for every returned plan in matching order.
    """

    plans: tuple[SourceOptimizationPlan, ...]
    assessments: tuple[SourceOptimizationAssessment, ...]


@dataclass(frozen=True, slots=True)
class SourceOptimizationPlanningOptions:
    """Dynamic policy and environment inputs for report-only source planning.

    Attributes:
        profile: Current-invocation baseline profile, when configured and successful.
        compile_config: Validated semantic and benchmark policy.
        project_root: Target project root used to normalize source paths.
        python_abi: Explicit ABI identity, or `None` to use the current interpreter tag.
    """

    profile: ProfileResult | None
    compile_config: CompileConfig
    project_root: Path
    python_abi: str | None = None


@dataclass(frozen=True, slots=True)
class _RankedSourcePlan:
    plan: SourceOptimizationPlan
    assessment: SourceOptimizationAssessment


@dataclass(frozen=True, slots=True)
class _AssessmentContext:
    source_plan: SourceOptimizationPlan
    execution_plan: ExecutionPlan
    scan: ModuleScan
    profile: ProfileResult | None
    profiled_members: dict[tuple[str, str], ProfiledMember]
    compile_config: CompileConfig


@dataclass(frozen=True, slots=True)
class _AssessmentEvidence:
    observed_work_items: int
    immediate_result_ratio: float
    attributed_hot_share: float
    work_evidence: tuple[SourceCallableEvidence, ...]


def build_source_optimization_plans(
    scans: tuple[ModuleScan, ...],
    execution_plans: tuple[ExecutionPlan | PlanRejection, ...],
    options: SourceOptimizationPlanningOptions,
) -> SourceOptimizationPlanningResult:
    """Build profile-ranked source plans without rewriting the target checkout.

    Args:
        scans: Static source facts for the selected compile scope.
        execution_plans: Scheduler plans and report-only rejections already discovered.
        options: Baseline profile, compile policy, project root, and ABI boundary.

    Returns:
        SourceOptimizationPlanningResult: Static plans and assessments in descending hotness
        order. Rejected execution-plan sites are not promoted into source plans.
    """
    scan_by_module = {scan.module.name: scan for scan in scans}
    profiled_members = (
        {(member.module, member.qualname): member for member in options.profile.members}
        if options.profile is not None
        else {}
    )
    resolved_abi = options.python_abi or sys.implementation.cache_tag or "python-unknown-abi"
    ranked: list[_RankedSourcePlan] = []
    for execution_plan in execution_plans:
        if isinstance(execution_plan, PlanRejection):
            continue
        scan = scan_by_module.get(execution_plan.source_module)
        if scan is None:
            continue
        source_plan = _source_plan(
            execution_plan,
            scan,
            project_root=options.project_root,
            python_abi=resolved_abi,
        )
        assessment = _assessment(
            _AssessmentContext(
                source_plan=source_plan,
                execution_plan=execution_plan,
                scan=scan,
                profile=options.profile,
                profiled_members=profiled_members,
                compile_config=options.compile_config,
            )
        )
        ranked.append(_RankedSourcePlan(plan=source_plan, assessment=assessment))
    selected = tuple(
        sorted(
            ranked,
            key=lambda item: (
                -item.assessment.attributed_hot_share,
                -item.assessment.observed_work_items,
                item.plan.id,
            ),
        )[:MAX_SOURCE_OPTIMIZATION_PLANS]
    )
    return SourceOptimizationPlanningResult(
        plans=tuple(item.plan for item in selected),
        assessments=tuple(item.assessment for item in selected),
    )


def _source_plan(
    execution_plan: ExecutionPlan,
    scan: ModuleScan,
    *,
    project_root: Path,
    python_abi: str,
) -> SourceOptimizationPlan:
    worker = _primary_worker(execution_plan)
    source_path = _relative_source_path(project_root, scan.module.path)
    access_sites = _access_sites(execution_plan, scan, source_path)
    entrypoint = _forwarding_entrypoint(scan, execution_plan) or execution_plan.owner
    steps = _transformation_steps(
        execution_plan,
        worker=worker,
        entrypoint=entrypoint,
        access_sites=access_sites,
    )
    transformation_versions: tuple[tuple[SourceTransformationKind, str], ...] = tuple(
        (step.kind, step.version) for step in steps
    )
    identity = SourceOptimizationIdentity(
        execution_plan_id=execution_plan.id,
        source_hashes=_source_hash_paths(execution_plan, scan, project_root),
        topology_fingerprint=execution_plan.topology_fingerprint,
        dialect=execution_plan.dialect,
        lowering_version=SOURCE_OPTIMIZATION_LOWERING_VERSION,
        python_abi=python_abi,
        transformation_versions=transformation_versions,
    )
    return SourceOptimizationPlan(
        id=stable_source_optimization_plan_id(identity),
        identity=identity,
        source=source_path,
        owner=execution_plan.owner,
        worker=worker,
        consumer=execution_plan.consumer,
        reducer=execution_plan.reducer,
        transport=execution_plan.completion_transport or "<unknown-private-transport>",
        access_sites=access_sites,
        entrypoint=entrypoint,
        steps=steps,
        semantic_boundaries=_SEMANTIC_BOUNDARIES,
    )


def _assessment(context: _AssessmentContext) -> SourceOptimizationAssessment:
    source_plan = context.source_plan
    execution_plan = context.execution_plan
    profile = context.profile
    evidence = _callable_evidence(
        execution_plan,
        context.scan,
        profile,
        context.profiled_members,
    )
    work_items = tuple(
        item.symbol for item in evidence if item.static_role in {"worker", "producer", "dependency"}
    ) or (source_plan.worker,)
    work_evidence = tuple(item for item in evidence if item.symbol in work_items)
    observed_work_items = execution_plan.observed_invocations
    immediate_result_ratio = min(
        (item.immediate_result_ratio for item in work_evidence if item.completed_calls > 0),
        default=0.0,
    )
    attributed_hot_share = min(1.0, sum(item.hot_share for item in evidence))
    scheduler_overhead_samples = sum(item.scheduler_overhead_samples for item in evidence)
    scheduler_overhead_share = (
        scheduler_overhead_samples / profile.total_samples
        if profile is not None and profile.total_samples > 0
        else 0.0
    )
    rejections = _assessment_rejections(
        source_plan,
        profile=profile,
        compile_config=context.compile_config,
        evidence=_AssessmentEvidence(
            observed_work_items=observed_work_items,
            immediate_result_ratio=immediate_result_ratio,
            attributed_hot_share=attributed_hot_share,
            work_evidence=work_evidence,
        ),
    )
    configured = (
        context.compile_config.test_command is not None
        and context.compile_config.benchmark_command is not None
    )
    status: SourceOptimizationAssessmentStatus
    if not configured or profile is None or profile.status != "profiled":
        status = "unbenchmarked"
    elif rejections:
        status = "unsupported"
    else:
        status = "trial-ready"
    return SourceOptimizationAssessment(
        plan_id=source_plan.id,
        status=status,
        minimum_speedup=max(
            MINIMUM_SOURCE_OPTIMIZATION_SPEEDUP,
            context.compile_config.minimum_speedup,
        ),
        work_items=work_items,
        observed_work_items=observed_work_items,
        immediate_result_ratio=immediate_result_ratio,
        attributed_hot_share=attributed_hot_share,
        scheduler_overhead_samples=scheduler_overhead_samples,
        scheduler_overhead_share=scheduler_overhead_share,
        scheduler_overhead_evidence=(
            f"{scheduler_overhead_samples} nested sample(s) attributed to active plan callables",
            (
                f"{attributed_hot_share:.1%} of attributable project and scheduler samples "
                "map to the execution-plan boundary"
            ),
        ),
        callable_evidence=evidence,
        rejections=rejections,
        headroom_speedup=None,
    )


def _assessment_rejections(
    source_plan: SourceOptimizationPlan,
    *,
    profile: ProfileResult | None,
    compile_config: CompileConfig,
    evidence: _AssessmentEvidence,
) -> tuple[str, ...]:
    rejections = [
        *_configuration_rejections(profile, compile_config),
        *_scale_rejections(evidence),
        *_source_safety_rejections(source_plan, evidence.work_evidence),
    ]
    return tuple(dict.fromkeys(rejections))


def _configuration_rejections(
    profile: ProfileResult | None,
    compile_config: CompileConfig,
) -> tuple[str, ...]:
    rejections: list[str] = []
    if compile_config.test_command is None or compile_config.benchmark_command is None:
        rejections.append("source optimization requires configured test and benchmark commands")
    if profile is None or profile.status != "profiled":
        rejections.append("source optimization requires a successful current-invocation profile")
    return tuple(rejections)


def _scale_rejections(evidence: _AssessmentEvidence) -> tuple[str, ...]:
    rejections: list[str] = []
    if evidence.observed_work_items < MINIMUM_OBSERVED_WORK_ITEMS:
        rejections.append(
            f"observed {evidence.observed_work_items} work items; "
            f"{MINIMUM_OBSERVED_WORK_ITEMS} required"
        )
    if evidence.attributed_hot_share < MINIMUM_ATTRIBUTED_HOT_SHARE:
        rejections.append(
            f"attributed hot share {evidence.attributed_hot_share:.1%}; "
            f"{MINIMUM_ATTRIBUTED_HOT_SHARE:.0%} required"
        )
    if evidence.immediate_result_ratio < 1.0:
        rejections.append(
            f"immediate-result ratio {evidence.immediate_result_ratio:.1%}; "
            "zero observed suspension required"
        )
    return tuple(rejections)


def _source_safety_rejections(
    source_plan: SourceOptimizationPlan,
    work_evidence: tuple[SourceCallableEvidence, ...],
) -> tuple[str, ...]:
    rejections: list[str] = []
    if source_plan.transport == "<unknown-private-transport>":
        rejections.append("execution plan has no statically owned private completion transport")
    if not any(site.kind == "transport-receive" for site in source_plan.access_sites):
        rejections.append("no private consumer receive site was found")
    for item in work_evidence:
        if item.static_suspension_points:
            rejections.append(
                f"{item.symbol.stable_id} has {item.static_suspension_points} non-transport "
                "suspension point(s)"
            )
        for label, values in (
            ("task introspection", item.task_introspection),
            ("cancellation", item.cancellation),
            ("context mutation", item.context_mutation),
            ("dynamic calls", item.unknown_dynamic_calls),
        ):
            if label == "context mutation" and item.symbol == source_plan.worker:
                continue
            if values:
                rejections.append(
                    f"{item.symbol.stable_id} references {label}: {', '.join(values)}"
                )
    return tuple(rejections)


def _callable_evidence(
    execution_plan: ExecutionPlan,
    scan: ModuleScan,
    profile: ProfileResult | None,
    profiled_members: dict[tuple[str, str], ProfiledMember],
) -> tuple[SourceCallableEvidence, ...]:
    records = {symbol.id: symbol for symbol in scan.symbols}
    attributable_samples = (
        profile.mapped_project_samples + profile.scheduler_overhead_samples
        if profile is not None
        else 0
    )
    return tuple(
        _callable_evidence_item(
            symbol_id,
            role=role,
            record=records.get(symbol_id),
            member=profiled_members.get((symbol_id.module, symbol_id.qualname)),
            total_samples=attributable_samples,
        )
        for symbol_id, role in sorted(
            _callable_roles(execution_plan, records).items(), key=lambda item: item[0].stable_id
        )
    )


def _callable_roles(
    execution_plan: ExecutionPlan,
    records: dict[SymbolId, SymbolRecord],
) -> dict[SymbolId, str]:
    roles: dict[SymbolId, str] = {execution_plan.owner: "owner"}
    for node in execution_plan.nodes:
        if node.symbol is not None:
            roles.setdefault(node.symbol, node.role)
    if execution_plan.consumer is not None:
        roles[execution_plan.consumer] = "consumer"
    if execution_plan.reducer is not None:
        roles[execution_plan.reducer] = "reducer"
    for source_member in execution_plan.source_members:
        roles.setdefault(source_member, "dependency")
    pending = list(roles)
    records_by_qualname = {symbol.id.qualname: symbol for symbol in records.values()}
    while pending:
        symbol_id = pending.pop()
        record = records_by_qualname.get(symbol_id.qualname)
        if record is None:
            continue
        for path in record.called_paths:
            dependency = records_by_qualname.get(path)
            if dependency is None or dependency.kind == "class" or dependency.id in roles:
                continue
            roles[dependency.id] = "dependency"
            pending.append(dependency.id)
    return roles


def _callable_evidence_item(
    symbol_id: SymbolId,
    *,
    role: str,
    record: SymbolRecord | None,
    member: ProfiledMember | None,
    total_samples: int,
) -> SourceCallableEvidence:
    task_introspection, cancellation, context_mutation, dynamic_calls = _hazard_paths(record)
    effective_suspensions = _effective_suspension_count(record)
    samples = member.samples + member.scheduler_overhead_samples if member is not None else 0
    hot_share = samples / total_samples if total_samples else 0.0
    return SourceCallableEvidence(
        symbol=symbol_id,
        static_role=role,
        observed_invocations=member.invocation_count if member is not None else 0,
        completed_calls=member.completed_calls if member is not None else 0,
        static_suspension_points=effective_suspensions,
        observed_suspensions=member.pre_completion_suspensions if member is not None else 0,
        immediate_result_ratio=member.immediate_result_ratio if member is not None else 0.0,
        hot_share=hot_share,
        scheduler_overhead_samples=member.scheduler_overhead_samples if member is not None else 0,
        task_introspection=task_introspection,
        cancellation=cancellation,
        context_mutation=context_mutation,
        unknown_dynamic_calls=dynamic_calls,
        hazards=_callable_hazards(
            effective_suspensions=effective_suspensions,
            task_introspection=task_introspection,
            cancellation=cancellation,
            context_mutation=context_mutation,
            dynamic_calls=dynamic_calls,
        ),
    )


def _callable_hazards(
    *,
    effective_suspensions: int,
    task_introspection: tuple[str, ...],
    cancellation: tuple[str, ...],
    context_mutation: tuple[str, ...],
    dynamic_calls: tuple[str, ...],
) -> tuple[SourceOptimizationHazard, ...]:
    hazards: list[SourceOptimizationHazard] = []
    if effective_suspensions:
        hazards.append("suspension")
    if task_introspection:
        hazards.append("task-introspection")
    if cancellation:
        hazards.append("cancellation")
    if context_mutation:
        hazards.append("context-mutation")
    if dynamic_calls:
        hazards.append("unknown-dynamic-call")
    return tuple(hazards)


def _hazard_paths(
    record: SymbolRecord | None,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if record is None:
        return (), (), (), ()
    paths = tuple(site.target for site in record.call_sites)
    task_introspection = _paths_with_tail(paths, _TASK_INTROSPECTION_METHODS)
    cancellation = _paths_with_tail(paths, _CANCELLATION_METHODS)
    context_mutation = tuple(
        sorted(
            path
            for path in paths
            if _tail(path) in {"set", "reset"}
            and any(
                name in {"ContextVar", "contextvars"} or "context" in name.lower()
                for name in record.referenced_names
            )
        )
    )
    dynamic_calls = _paths_with_tail(paths, _DYNAMIC_CALLS)
    return task_introspection, cancellation, context_mutation, dynamic_calls


def _effective_suspension_count(record: SymbolRecord | None) -> int:
    if record is None:
        return 0
    transport_lines = {
        site.lineno
        for site in record.call_sites
        if _tail(site.target) in _SEND_METHODS | _RECEIVE_METHODS | _DRAIN_METHODS
    }
    return sum(point.lineno not in transport_lines for point in record.suspension_points)


def _access_sites(
    execution_plan: ExecutionPlan,
    scan: ModuleScan,
    source_path: PurePosixPath,
) -> tuple[SourceAccessSite, ...]:
    plan_symbols = {
        execution_plan.owner,
        *(node.symbol for node in execution_plan.nodes if node.symbol is not None),
    }
    records = {symbol.id: symbol for symbol in scan.symbols}
    sites: list[SourceAccessSite] = []
    for symbol_id in sorted(plan_symbols, key=lambda item: item.stable_id):
        record = records.get(symbol_id)
        if record is None:
            continue
        for call in record.call_sites:
            tail = _tail(call.target)
            kind: SourceAccessKind | None = (
                "transport-receive"
                if tail in _RECEIVE_METHODS
                else "transport-drain"
                if tail in _DRAIN_METHODS
                else "transport-send"
                if tail in _SEND_METHODS
                else None
            )
            if kind is None:
                continue
            sites.append(
                SourceAccessSite(
                    path=source_path,
                    symbol=symbol_id,
                    kind=kind,
                    lineno=call.lineno,
                    expression=call.target,
                )
            )
    return tuple(
        sorted(sites, key=lambda item: (item.path.as_posix(), item.lineno, item.expression))
    )


def _transformation_steps(
    execution_plan: ExecutionPlan,
    *,
    worker: SymbolId,
    entrypoint: SymbolId,
    access_sites: tuple[SourceAccessSite, ...],
) -> tuple[TransformationStep, ...]:
    reads = tuple(site for site in access_sites if site.kind == "transport-receive")
    steps: list[TransformationStep] = []
    if reads:
        steps.append(
            TransformationStep(
                kind="private-transport-batch-drain",
                version="batch-drain-v1",
                source_symbol=execution_plan.consumer or execution_plan.owner,
                target_symbol=None,
                access_sites=reads,
                semantic_boundary="private transport ordering and capacity",
                description="Drain already available private records before awaiting transport.",
            )
        )
    steps.extend(
        (
            TransformationStep(
                kind="quiescent-callable-execution",
                version="quiescent-callable-v1",
                source_symbol=worker,
                target_symbol=None,
                access_sites=tuple(site for site in access_sites if site.kind == "transport-send"),
                semantic_boundary="copied context and fallback before entry",
                description="Drive a proven non-suspending work callable in a copied context.",
            ),
            TransformationStep(
                kind="local-state-machine-fusion",
                version="state-machine-v1",
                source_symbol=execution_plan.owner,
                target_symbol=None,
                access_sites=access_sites,
                semantic_boundary="completion ordering, exceptions, and no retry after entry",
                description="Fuse private producer, transport, consumer, and reducer transitions.",
            ),
        )
    )
    if entrypoint != execution_plan.owner:
        steps.append(
            TransformationStep(
                kind="private-protocol-auto-forwarding",
                version="protocol-forward-v1",
                source_symbol=entrypoint,
                target_symbol=None,
                access_sites=(),
                semantic_boundary="public incremental iteration stays on the original path",
                description="Add a private run-to-completion path for pure protocol forwarding.",
            )
        )
    return tuple(steps)


def _primary_worker(plan: ExecutionPlan) -> SymbolId:
    role_priority = {"worker": 0, "producer": 1, "wrapper": 2}
    candidates: list[tuple[int, str, SymbolId]] = [
        (role_priority[node.role], node.id, node.symbol)
        for node in plan.nodes
        if node.symbol is not None and node.role in role_priority
    ]
    if not candidates:
        return plan.owner
    return min(candidates)[2]


def _source_hash_paths(
    plan: ExecutionPlan,
    scan: ModuleScan,
    project_root: Path,
) -> tuple[tuple[PurePosixPath, str], ...]:
    paths_by_module = {scan.module.name: _relative_source_path(project_root, scan.module.path)}
    return tuple(
        (
            paths_by_module.get(module, PurePosixPath(*module.split(".")).with_suffix(".py")),
            source_hash,
        )
        for module, source_hash in sorted(plan.source_hashes)
    ) or ((paths_by_module[scan.module.name], plan.source_hash),)


def _forwarding_entrypoint(scan: ModuleScan, plan: ExecutionPlan) -> SymbolId | None:
    try:
        module = ast.parse(
            scan.module.path.read_text(encoding="utf-8"), filename=str(scan.module.path)
        )
    except (OSError, SyntaxError, UnicodeError):
        return None
    targets = {plan.owner.qualname}
    if plan.consumer is not None:
        targets.add(plan.consumer.qualname)
    for symbol_id, node in _function_nodes(scan.module.name, module):
        for statement in node.body:
            if not isinstance(statement, ast.AsyncFor) or len(statement.body) != 1:
                continue
            forwarded = statement.body[0]
            if not (
                isinstance(forwarded, ast.Expr)
                and isinstance(forwarded.value, ast.Yield)
                and _same_forwarded_name(statement.target, forwarded.value.value)
            ):
                continue
            target = _call_target(statement.iter)
            if target is not None and _resolved_target_qualname(symbol_id, target) in targets:
                return symbol_id
    return None


def _function_nodes(
    module_name: str,
    module: ast.Module,
) -> tuple[tuple[SymbolId, ast.FunctionDef | ast.AsyncFunctionDef], ...]:
    nodes: list[tuple[SymbolId, ast.FunctionDef | ast.AsyncFunctionDef]] = []
    for statement in module.body:
        if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
            nodes.append((SymbolId(module_name, statement.name), statement))
        elif isinstance(statement, ast.ClassDef):
            nodes.extend(
                (SymbolId(module_name, f"{statement.name}.{child.name}"), child)
                for child in statement.body
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
            )
    return tuple(nodes)


def _call_target(node: ast.expr) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    return _expression_path(node.func)


def _expression_path(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _expression_path(node.value)
        return f"{parent}.{node.attr}" if parent is not None else None
    return None


def _resolved_target_qualname(owner: SymbolId, target: str) -> str:
    if target.startswith(("self.", "cls.")) and "." in owner.qualname:
        class_name = owner.qualname.rsplit(".", maxsplit=1)[0]
        return f"{class_name}.{target.split('.', maxsplit=1)[1]}"
    return target


def _same_forwarded_name(target: ast.expr, value: ast.expr | None) -> bool:
    return isinstance(target, ast.Name) and isinstance(value, ast.Name) and target.id == value.id


def _paths_with_tail(paths: tuple[str, ...], tails: frozenset[str]) -> tuple[str, ...]:
    return tuple(sorted({path for path in paths if _tail(path) in tails}))


def _tail(path: str) -> str:
    return path.rsplit(".", maxsplit=1)[-1]


def _relative_source_path(root: Path, path: Path) -> PurePosixPath:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        relative = Path(path.name)
    return PurePosixPath(relative.as_posix())
