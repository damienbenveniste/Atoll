"""Discover scheduler execution plans from scanned source and profile evidence.

Discovery reparses `ModuleScan` source with `ast` and never imports target
modules. Static syntax determines plan identity and rejection reasons; dynamic
profile data only determines hotness, ordering, and coverage selection.
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from atoll.execution_plans.dialects import SchedulerDialect, SpawnCall, built_in_scheduler_dialects
from atoll.execution_plans.models import (
    ExecutionPlan,
    ExecutionPlanIdentity,
    ExecutionPlanRejectionReason,
    PlanEdge,
    PlanGuard,
    PlanNode,
    PlanRejection,
    stable_execution_plan_id,
)
from atoll.models import ModuleScan, SymbolId, SymbolRecord
from atoll.runtime.profiling import ProfileResult, ProfileSpawnSiteTarget

_MIN_INVOCATIONS = 1_000
_MIN_LIFECYCLE_START_SHARE = 0.05
_MAX_SELECTED_PLANS = 4
_TARGET_COVERAGE = 0.80
_MIN_QUALIFIED_METHOD_PARTS = 2
_MEMORY_STREAM_ENDPOINTS = 2
_INSTANCE_RECEIVER_PARTS = 2
_ISINSTANCE_ARGUMENTS = 2


@dataclass(frozen=True, slots=True)
class _CandidateSite:
    owner: SymbolRecord
    dialect: SchedulerDialect
    spawns: tuple[SpawnCall, ...]
    transport: _TransportEvidence | None
    producers: tuple[SymbolId, ...]
    consumer: SymbolId | None
    consumer_spawned: bool
    evidence_members: tuple[SymbolId, ...]
    rejection_reason: ExecutionPlanRejectionReason | None
    rejection_message: str | None


@dataclass(frozen=True, slots=True)
class _RankedSite:
    scan: ModuleScan
    site: _CandidateSite
    hotness: int
    observed_invocations: int
    lifecycle_starts: int
    start_share: float
    guarded_callable_identities: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ReflectionCandidate:
    """One callable whose repeated signature lookup can be guarded and cached.

    Attributes:
        symbol: Static callable containing the reflection site.
        lineno: One-based source line of the signature lookup.
    """

    symbol: SymbolRecord
    lineno: int


_TransportCapacityKind = Literal["bounded", "rendezvous", "unbounded", "unknown"]
_TransportScope = Literal["local", "instance"]


@dataclass(frozen=True, slots=True)
class _TransportCapacity:
    """Static transport capacity without conflating distinct zero semantics.

    Attributes:
        value: Literal capacity when known, including zero for rendezvous streams.
        kind: Bounded, rendezvous, unbounded, or statically unknown semantics.
    """

    value: int | None
    kind: _TransportCapacityKind


@dataclass(frozen=True, slots=True)
class _TransportEvidence:
    """Static identity shared by a queue or paired memory-stream endpoints.

    Attributes:
        identity: Stable local transport name or paired endpoint identity.
        endpoints: Stable names or receiver paths through which producers and the
            consumer access the transport.
        capacity: Statically classified capacity semantics.
        scope: Whether endpoints are local to one callable or stored on an instance.
        creator: Symbol containing the recognized transport factory assignment.
    """

    identity: str
    endpoints: tuple[str, ...]
    capacity: _TransportCapacity
    scope: _TransportScope
    creator: SymbolId


@dataclass(frozen=True, slots=True)
class _TransportUser:
    """Resolved spawned callable and its use of a private transport endpoint.

    Attributes:
        symbol: Static callable receiving the endpoint.
        spawn: Scheduler call that passes the endpoint to the callable.
        role: Producer, consumer, or unknown role inferred from endpoint API calls.
    """

    symbol: SymbolRecord
    spawn: SpawnCall
    role: str


@dataclass(frozen=True, slots=True)
class _TaskGroupScope:
    """Lexical task-group binding and the source span that owns spawned work.

    Attributes:
        dialect: Scheduler dialect recognized for the task-group context manager.
        lineno: First source line of the owning asynchronous context.
        end_lineno: Final source line before the task group joins all child work.
    """

    dialect: str
    lineno: int
    end_lineno: int


@dataclass(frozen=True, slots=True)
class _DiscoveryContext:
    """Parsed module state shared by one execution-plan discovery pass.

    Attributes:
        scan: Original scanner result for symbols and source identity.
        tree: Parsed source module used for conservative syntax proofs.
        integer_constants: Stable module integer literals usable as capacities.
        symbol_map: Module-local symbols indexed by qualified name.
        function_nodes: Top-level functions and direct class methods by qualified name.
        class_nodes: Top-level classes by name.
    """

    scan: ModuleScan
    tree: ast.Module
    integer_constants: Mapping[str, int]
    symbol_map: Mapping[str, SymbolRecord]
    function_nodes: Mapping[str, ast.FunctionDef | ast.AsyncFunctionDef]
    class_nodes: Mapping[str, ast.ClassDef]


@dataclass(frozen=True, slots=True)
class _OwnerContext:
    """Owner symbol, syntax node, and module context for one candidate site.

    Attributes:
        owner: Symbol record that owns the scheduler candidate being inspected.
        node: Function or method syntax node that contains the candidate site.
        discovery: Parsed module context shared by the surrounding discovery pass.
    """

    owner: SymbolRecord
    node: ast.FunctionDef | ast.AsyncFunctionDef
    discovery: _DiscoveryContext


@dataclass(frozen=True, slots=True)
class _SchedulerOwnership:
    """One conservative static proof for a scheduler receiver.

    Attributes:
        identity: Stable receiver ownership identity used to reject mixed owners.
        evidence_members: Symbols whose source establishes delegated ownership.
    """

    identity: str
    evidence_members: tuple[SymbolId, ...]


@dataclass(frozen=True, slots=True)
class _FieldBinding:
    """One explicit assignment to a stable instance field.

    Attributes:
        symbol: Method containing the assignment.
        node: Method AST used to resolve parameter and factory aliases.
        value: Assigned expression, or `None` for deletion or augmented mutation.
        annotation: Inline field annotation when the assignment has one.
    """

    symbol: SymbolRecord
    node: ast.FunctionDef | ast.AsyncFunctionDef
    value: ast.expr | None
    annotation: ast.expr | None


@dataclass(frozen=True, slots=True)
class _LifecycleUse:
    """Task-group lifecycle operations found in one class method.

    Attributes:
        managed_context: Whether the method owns an `async with` task-group lifetime.
        direct_enter: Whether the method directly calls an enter operation.
        direct_exit: Whether the method directly calls an exit operation.
        stack_enters: ExitStack-style enter calls indexed by receiver expression.
        stack_exits: ExitStack-style exit calls indexed by receiver expression.
    """

    managed_context: bool
    direct_enter: bool
    direct_exit: bool
    stack_enters: tuple[str, ...]
    stack_exits: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SchedulerField:
    """Stable instance field considered as a delegated scheduler receiver.

    Attributes:
        class_name: Class that owns the instance field.
        reference: Field reference in the class instance namespace.
        class_node: Parsed class node used to validate lifecycle evidence.
        annotation: Class-level annotation for the field, when one exists.
        class_evidence: Symbols whose source proves the class delegates scheduling.
    """

    class_name: str
    reference: str
    class_node: ast.ClassDef
    annotation: ast.AnnAssign | None
    class_evidence: tuple[SymbolId, ...]


def execution_plan_observation_targets(scans: Iterable[ModuleScan]) -> tuple[str, ...]:
    """Return static `module::qualname` observation targets for execution-plan ranking.

    Args:
        scans: AST-derived module scans to inspect without importing target code.

    Returns:
        tuple[str, ...]: Deterministically ordered symbols owning scheduler spawn sites.
    """
    targets: set[str] = set()
    scan_tuple = tuple(scans)
    for scan in scan_tuple:
        symbol_ids = {symbol.id.qualname: symbol.id for symbol in scan.symbols}
        for site in _candidate_sites(scan):
            targets.add(site.owner.id.stable_id)
            for spawn in site.spawns:
                name = _resolved_callee_name(site.owner, spawn.callee_name)
                if name is not None and name in symbol_ids:
                    targets.add(symbol_ids[name].stable_id)
    targets.update(
        candidate.symbol.id.stable_id for candidate in _reflection_candidates(scan_tuple)
    )
    return tuple(sorted(targets))


def execution_plan_profile_targets(
    scans: Iterable[ModuleScan],
) -> tuple[ProfileSpawnSiteTarget, ...]:
    """Return exact scheduler call sites for invocation profiling.

    Args:
        scans: AST-derived module scans to inspect without importing target code.

    Returns:
        tuple[ProfileSpawnSiteTarget, ...]: Deterministically ordered scheduler call sites.
    """
    targets = (
        ProfileSpawnSiteTarget(
            id=_spawn_target_id(scan.module.name, site.owner.id, spawn),
            owner=site.owner.id,
            lineno=spawn.lineno,
            col_offset=spawn.col_offset,
            scheduler_method=_scheduler_method(spawn),
            end_lineno=spawn.end_lineno,
            end_col_offset=spawn.end_col_offset,
        )
        for scan in scans
        for site in _candidate_sites(scan)
        for spawn in site.spawns
    )
    return tuple(sorted(targets, key=lambda target: target.id))


def build_execution_plans(
    scans: Iterable[ModuleScan],
    profile: ProfileResult | None,
) -> tuple[ExecutionPlan | PlanRejection, ...]:
    """Build selected execution plans and report rejected plan sites.

    Args:
        scans: AST-derived module scans to inspect without importing target code.
        profile: Runtime profile evidence used only for hotness ranking and thresholds,
            or `None` when plans should remain report-only static candidates.

    Returns:
        tuple[ExecutionPlan | PlanRejection, ...]: Selected plans followed by deterministic
            report-only rejections for unsafe or insufficiently hot sites.
    """
    scan_tuple = tuple(scans)
    lifecycle_starts = _mapped_lifecycle_starts(scan_tuple, profile)
    ranked = tuple(_ranked_sites(scan_tuple, profile, lifecycle_starts))
    selected: list[_RankedSite] = []
    rejected: list[PlanRejection] = []
    selected_hotness = 0
    eligible_hotness = sum(site.hotness for site in ranked if _is_hot(site))

    for ranked_site in ranked:
        if ranked_site.site.rejection_reason is not None:
            rejected.append(_rejection_for(ranked_site.scan, ranked_site.site, ranked_site.hotness))
            continue
        scheduler_rejection = _runtime_scheduler_rejection(ranked_site, profile)
        if scheduler_rejection is not None:
            rejected.append(scheduler_rejection)
            continue
        if not _is_hot(ranked_site):
            rejected.append(
                _rejection_for(
                    ranked_site.scan,
                    ranked_site.site,
                    ranked_site.hotness,
                    reason="low-hotness",
                    message="orchestration site did not meet execution-plan hotness thresholds",
                )
            )
            continue
        if len(selected) >= _MAX_SELECTED_PLANS:
            rejected.append(
                _rejection_for(
                    ranked_site.scan,
                    ranked_site.site,
                    ranked_site.hotness,
                    reason="selection-limit",
                    message="execution-plan selection limit was reached",
                )
            )
            continue
        if eligible_hotness > 0 and selected_hotness / eligible_hotness >= _TARGET_COVERAGE:
            rejected.append(
                _rejection_for(
                    ranked_site.scan,
                    ranked_site.site,
                    ranked_site.hotness,
                    reason="coverage-reached",
                    message="execution-plan hotness coverage target was already reached",
                )
            )
            continue
        selected.append(ranked_site)
        selected_hotness += ranked_site.hotness

    plans = tuple(_plan_for(ranked_site, scan_tuple, profile) for ranked_site in selected)
    return (*plans, *tuple(sorted(rejected, key=lambda rejection: rejection.id)))


def _candidate_sites(scan: ModuleScan) -> tuple[_CandidateSite, ...]:
    source = scan.module.path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(scan.module.path), type_comments=True)
    context = _DiscoveryContext(
        scan=scan,
        tree=tree,
        integer_constants=_module_integer_constants(tree),
        symbol_map={symbol.id.qualname: symbol for symbol in scan.symbols},
        function_nodes=_function_nodes(tree),
        class_nodes=_class_nodes(tree),
    )
    sites: list[_CandidateSite] = []
    for symbol in scan.symbols:
        node = context.function_nodes.get(symbol.id.qualname)
        if node is None:
            continue
        spawns = _spawn_calls(node)
        if not spawns:
            continue
        sites.append(_site_from_spawns(symbol, spawns, node, context))
    return tuple(sites)


def _site_from_spawns(
    owner: SymbolRecord,
    spawns: tuple[SpawnCall, ...],
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    context: _DiscoveryContext,
) -> _CandidateSite:
    dialect_names = {spawn.dialect for spawn in spawns}
    if len(dialect_names) != 1:
        return _rejected_site(owner, spawns, "ambiguous-spawn", "multiple scheduler dialects")
    callee_symbols = _resolved_callees(owner, spawns, context.symbol_map)
    if callee_symbols is None:
        return _rejected_site(owner, spawns, "ambiguous-spawn", "unresolved spawned callee")
    task_groups = _structured_task_groups(node)
    ownerships = tuple(
        _scheduler_ownership_for_spawn(owner, spawn, task_groups, context) for spawn in spawns
    )
    if any(ownership is None for ownership in ownerships):
        return _rejected_site(
            owner,
            spawns,
            "unstructured-task",
            "spawned work is not owned by a recognized task-group scope",
        )
    proven_ownerships = tuple(ownership for ownership in ownerships if ownership is not None)
    if len({ownership.identity for ownership in proven_ownerships}) != 1:
        return _rejected_site(
            owner,
            spawns,
            "ambiguous-spawn",
            "spawned work uses multiple scheduler owners",
        )
    if _task_handle_escapes(node):
        return _rejected_site(
            owner,
            spawns,
            "escaping-handle",
            "a spawned task handle escapes the orchestration callable",
        )
    owner_context = _OwnerContext(owner=owner, node=node, discovery=context)
    return _transport_site(
        owner_context,
        spawns,
        callee_symbols,
        _unique_symbol_ids(
            member for ownership in proven_ownerships for member in ownership.evidence_members
        ),
    )


def _transport_site(
    owner_context: _OwnerContext,
    spawns: tuple[SpawnCall, ...],
    callee_symbols: tuple[SymbolRecord, ...],
    scheduler_evidence: tuple[SymbolId, ...],
) -> _CandidateSite:
    owner = owner_context.owner
    node = owner_context.node
    context = owner_context.discovery
    dialect = _dialect_by_name(spawns[0].dialect)
    transport = _private_transport(owner, node, context)
    if transport is None:
        return _rejected_site(
            owner,
            spawns,
            "unknown-transport",
            "no private queue or memory-stream transport evidence was found",
        )
    users: list[tuple[SymbolRecord, SpawnCall]] = []
    for symbol, spawn in zip(callee_symbols, spawns, strict=True):
        callee_node = context.function_nodes.get(symbol.id.qualname)
        passes_endpoint = any(
            argument in transport.endpoints for argument in spawn.transport_arguments
        )
        reads_instance_endpoint = (
            transport.scope == "instance"
            and callee_node is not None
            and _expression_reads_references(callee_node, frozenset(transport.endpoints))
        )
        if passes_endpoint or reads_instance_endpoint:
            users.append((symbol, spawn))
    user_tuple = tuple(users)
    preflight_rejection = _transport_preflight_rejection(
        owner_context,
        transport,
        user_tuple,
        spawns,
    )
    if preflight_rejection is not None:
        return _rejected_site(owner, spawns, *preflight_rejection)
    classified_users = tuple(
        _TransportUser(
            symbol=symbol,
            spawn=spawn,
            role=_transport_role(
                symbol,
                context.function_nodes[symbol.id.qualname],
                spawn,
                transport.endpoints,
            ),
        )
        for symbol, spawn in user_tuple
    )
    spawned_consumers = tuple(user.symbol for user in classified_users if user.role == "consumer")
    direct_consumers = _direct_transport_consumers(owner, node, transport, context)
    consumers = _unique_symbols((*spawned_consumers, *direct_consumers))
    producers = tuple(user.symbol for user in classified_users if user.role == "producer")
    role_rejection = _transport_role_rejection(classified_users, consumers, producers)
    if role_rejection is not None:
        return _rejected_site(owner, spawns, *role_rejection)
    return _CandidateSite(
        owner=owner,
        dialect=dialect,
        spawns=spawns,
        transport=transport,
        producers=_unique_symbol_ids(symbol.id for symbol in producers),
        consumer=consumers[0].id,
        consumer_spawned=consumers[0].id in {symbol.id for symbol in spawned_consumers},
        evidence_members=_unique_symbol_ids((*scheduler_evidence, transport.creator)),
        rejection_reason=None,
        rejection_message=None,
    )


def _transport_preflight_rejection(
    owner_context: _OwnerContext,
    transport: _TransportEvidence,
    users: tuple[tuple[SymbolRecord, SpawnCall], ...],
    spawns: tuple[SpawnCall, ...],
) -> tuple[ExecutionPlanRejectionReason, str] | None:
    scope_nodes = _transport_scope_nodes(
        owner_context.owner,
        owner_context.node,
        transport,
        owner_context.discovery,
    )
    if not _transport_has_stable_ownership(scope_nodes, transport.endpoints):
        return (
            "unknown-transport",
            "result transport endpoints are rebound or have ambiguous ownership",
        )
    if _transport_escapes(scope_nodes, transport.endpoints, spawns):
        return "public-transport", "result transport escapes the orchestration callable"
    if transport.capacity.kind not in {"bounded", "rendezvous"}:
        return (
            "unknown-capacity",
            "result transport capacity must be statically bounded or rendezvous",
        )
    if not users:
        return (
            "unknown-transport",
            "private transport was not passed to any spawned callable",
        )
    return None


def _transport_role_rejection(
    users: tuple[_TransportUser, ...],
    consumers: tuple[SymbolRecord, ...],
    producers: tuple[SymbolRecord, ...],
) -> tuple[ExecutionPlanRejectionReason, str] | None:
    if len(consumers) != 1:
        reason: ExecutionPlanRejectionReason = (
            "multiple-consumer" if len(consumers) > 1 else "unknown-transport"
        )
        return reason, "transport must have exactly one statically evident consumer"
    if not producers:
        return (
            "unknown-transport",
            "transport must have at least one statically evident producer",
        )
    if any(user.role == "unknown" for user in users):
        return (
            "unknown-transport",
            (
                "every transport endpoint user must have a statically evident "
                "producer or consumer role"
            ),
        )
    return None


def _resolved_callees(
    owner: SymbolRecord,
    spawns: tuple[SpawnCall, ...],
    symbol_map: Mapping[str, SymbolRecord],
) -> tuple[SymbolRecord, ...] | None:
    resolved_names = tuple(_resolved_callee_name(owner, spawn.callee_name) for spawn in spawns)
    callee_symbols = tuple(
        symbol_map[name] for name in resolved_names if name is not None and name in symbol_map
    )
    if len(callee_symbols) != len(spawns):
        return None
    return callee_symbols


def _resolved_callee_name(owner: SymbolRecord, callee_name: str | None) -> str | None:
    if callee_name is None:
        return None
    if callee_name.startswith(("self.", "cls.")):
        owner_parts = owner.id.qualname.split(".")
        if len(owner_parts) < _MIN_QUALIFIED_METHOD_PARTS:
            return None
        return f"{'.'.join(owner_parts[:-1])}.{callee_name.partition('.')[2]}"
    return callee_name


def _rejected_site(
    owner: SymbolRecord,
    spawns: tuple[SpawnCall, ...],
    reason: ExecutionPlanRejectionReason,
    message: str,
) -> _CandidateSite:
    return _CandidateSite(
        owner=owner,
        dialect=_dialect_by_name(spawns[0].dialect),
        spawns=spawns,
        transport=None,
        producers=(),
        consumer=None,
        consumer_spawned=False,
        evidence_members=(),
        rejection_reason=reason,
        rejection_message=message,
    )


def _spawn_calls(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[SpawnCall, ...]:
    spawns: list[SpawnCall] = []
    dialects = built_in_scheduler_dialects()
    for child in _walk_function_scope(node):
        if not isinstance(child, ast.Call):
            continue
        for dialect in dialects:
            spawn = dialect.recognize_spawn(child)
            if spawn is not None:
                spawns.append(spawn)
                break
    return tuple(sorted(spawns, key=lambda spawn: (spawn.lineno, spawn.col_offset)))


def _private_transport(
    owner: SymbolRecord,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    context: _DiscoveryContext,
) -> _TransportEvidence | None:
    local = _transport_candidates(owner.id, node, context.integer_constants)
    if len(local) == 1:
        return local[0]
    if local:
        return None
    class_name = _owner_class_name(owner)
    if class_name is None:
        return None
    candidates = tuple(
        transport
        for qualname, method_node in _class_method_nodes(class_name, context)
        for symbol in (context.symbol_map.get(qualname),)
        if symbol is not None
        for transport in _transport_candidates(
            symbol.id,
            method_node,
            context.integer_constants,
        )
        if transport.scope == "instance"
    )
    if len(candidates) == 1:
        return candidates[0]
    return None


def _transport_candidates(
    creator: SymbolId,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    integer_constants: Mapping[str, int],
) -> tuple[_TransportEvidence, ...]:
    candidates: list[_TransportEvidence] = []
    for child in _walk_function_scope(node):
        if isinstance(child, ast.Assign):
            candidates.extend(_transports_from_assign(child, creator, integer_constants))
        elif isinstance(child, ast.AnnAssign):
            transport = _transport_from_ann_assign(child, creator, integer_constants)
            if transport is not None:
                candidates.append(transport)
    return tuple(candidates)


def _transports_from_assign(
    node: ast.Assign,
    creator: SymbolId,
    integer_constants: Mapping[str, int],
) -> tuple[_TransportEvidence, ...]:
    call_path = _call_path(node.value)
    if call_path is None:
        return ()
    transports: list[_TransportEvidence] = []
    if call_path in (("asyncio", "Queue"), ("Queue",)):
        for target in node.targets:
            reference = _stable_reference(target)
            scope = _transport_scope((reference,)) if reference is not None else None
            if reference is not None and scope is not None:
                transports.append(
                    _TransportEvidence(
                        identity=reference,
                        endpoints=(reference,),
                        capacity=_queue_capacity(node.value, integer_constants),
                        scope=scope,
                        creator=creator,
                    )
                )
    if call_path in (("anyio", "create_memory_object_stream"), ("create_memory_object_stream",)):
        for target in node.targets:
            endpoints = _tuple_references(target)
            scope = _transport_scope(endpoints)
            if len(endpoints) == _MEMORY_STREAM_ENDPOINTS and scope is not None:
                transports.append(
                    _TransportEvidence(
                        identity="|".join(endpoints),
                        endpoints=endpoints,
                        capacity=_memory_stream_capacity(node.value, integer_constants),
                        scope=scope,
                        creator=creator,
                    )
                )
    return tuple(transports)


def _transport_from_ann_assign(
    node: ast.AnnAssign,
    creator: SymbolId,
    integer_constants: Mapping[str, int],
) -> _TransportEvidence | None:
    call_path = _call_path(node.value)
    reference = _stable_reference(node.target)
    scope = _transport_scope((reference,)) if reference is not None else None
    if (
        call_path in (("asyncio", "Queue"), ("Queue",))
        and reference is not None
        and scope is not None
    ):
        return _TransportEvidence(
            identity=reference,
            endpoints=(reference,),
            capacity=_queue_capacity(node.value, integer_constants),
            scope=scope,
            creator=creator,
        )
    return None


def _tuple_references(node: ast.expr) -> tuple[str, ...]:
    if isinstance(node, ast.Tuple):
        references = tuple(_stable_reference(item) for item in node.elts)
        if all(reference is not None for reference in references):
            return tuple(reference for reference in references if reference is not None)
    return ()


def _transport_scope(endpoints: tuple[str, ...]) -> _TransportScope | None:
    if endpoints and all("." not in endpoint for endpoint in endpoints):
        return "local"
    if endpoints and all(
        endpoint.startswith("self.") and endpoint.count(".") == 1 for endpoint in endpoints
    ):
        return "instance"
    return None


def _transport_role(
    symbol: SymbolRecord,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    spawn: SpawnCall,
    endpoints: tuple[str, ...],
) -> str:
    parameters = tuple(
        parameter for parameter in symbol.parameters if parameter.name not in {"self", "cls"}
    )
    roles = _direct_transport_roles(
        node,
        tuple(endpoint for endpoint in endpoints if "." in endpoint),
    )
    for argument_index, argument_name in enumerate(spawn.transport_arguments):
        if argument_name not in endpoints or argument_index >= len(parameters):
            continue
        parameter_name = parameters[argument_index].name
        endpoint_index = endpoints.index(argument_name)
        roles.update(
            _reference_transport_roles(
                node,
                parameter_name,
                endpoint_index=endpoint_index,
                endpoint_count=len(endpoints),
            )
        )
    if len(roles) == 1:
        return next(iter(roles))
    return "unknown"


def _direct_transport_roles(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    endpoints: tuple[str, ...],
) -> set[str]:
    roles: set[str] = set()
    for endpoint_index, endpoint in enumerate(endpoints):
        roles.update(
            _reference_transport_roles(
                node,
                endpoint,
                endpoint_index=endpoint_index,
                endpoint_count=len(endpoints),
            )
        )
    return roles


def _reference_transport_roles(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    reference: str,
    *,
    endpoint_index: int,
    endpoint_count: int,
) -> set[str]:
    roles: set[str] = set()
    reference_path = tuple(reference.split("."))
    permits_producer = endpoint_count == 1 or endpoint_index == 0
    permits_consumer = endpoint_count == 1 or endpoint_index == 1
    for child in _walk_function_scope(node):
        if isinstance(child, ast.Call):
            path = _call_path(child)
            if path is not None and path[:-1] == reference_path:
                if permits_producer and path[-1] in {"put", "put_nowait", "send", "send_nowait"}:
                    roles.add("producer")
                if permits_consumer and path[-1] in {
                    "get",
                    "get_nowait",
                    "receive",
                    "receive_nowait",
                }:
                    roles.add("consumer")
            if (
                permits_consumer
                and path == ("anext",)
                and child.args
                and _stable_reference(child.args[0]) == reference
            ):
                roles.add("consumer")
        elif (
            isinstance(child, ast.AsyncFor)
            and permits_consumer
            and _stable_reference(child.iter) == reference
        ):
            roles.add("consumer")
    return roles


def _direct_transport_consumers(
    owner: SymbolRecord,
    owner_node: ast.FunctionDef | ast.AsyncFunctionDef,
    transport: _TransportEvidence,
    context: _DiscoveryContext,
) -> tuple[SymbolRecord, ...]:
    if transport.scope == "local":
        owner_is_consumer = _direct_transport_roles(owner_node, transport.endpoints) == {"consumer"}
        return (owner,) if owner_is_consumer else ()
    class_name = _owner_class_name(owner)
    if class_name is None:
        return ()
    consumers: list[SymbolRecord] = []
    for qualname, method_node in _class_method_nodes(class_name, context):
        symbol = context.symbol_map.get(qualname)
        if symbol is not None and _direct_transport_roles(method_node, transport.endpoints) == {
            "consumer"
        }:
            consumers.append(symbol)
    return tuple(consumers)


def _queue_capacity(
    node: ast.expr | None,
    integer_constants: Mapping[str, int],
) -> _TransportCapacity:
    value = _literal_capacity(node, default=0, integer_constants=integer_constants)
    if value is None:
        return _TransportCapacity(value=None, kind="unknown")
    if value <= 0:
        return _TransportCapacity(value=value, kind="unbounded")
    return _TransportCapacity(value=value, kind="bounded")


def _memory_stream_capacity(
    node: ast.expr | None,
    integer_constants: Mapping[str, int],
) -> _TransportCapacity:
    value = _literal_capacity(node, default=0, integer_constants=integer_constants)
    if value is None or value < 0:
        return _TransportCapacity(value=value, kind="unknown")
    if value == 0:
        return _TransportCapacity(value=0, kind="rendezvous")
    return _TransportCapacity(value=value, kind="bounded")


def _literal_capacity(
    node: ast.expr | None,
    *,
    default: int | None,
    integer_constants: Mapping[str, int],
) -> int | None:
    if not isinstance(node, ast.Call):
        return None
    capacity_node: ast.expr | None = node.args[0] if node.args else None
    if capacity_node is None:
        for keyword in node.keywords:
            if keyword.arg in {"maxsize", "max_buffer_size"}:
                capacity_node = keyword.value
                break
    if capacity_node is None:
        return default
    if isinstance(capacity_node, ast.Constant) and type(capacity_node.value) is int:
        return capacity_node.value
    if isinstance(capacity_node, ast.Name):
        return integer_constants.get(capacity_node.id)
    return None


def _module_integer_constants(tree: ast.Module) -> dict[str, int]:
    """Return direct module-level integer constants usable in capacity expressions.

    Only immutable literal bindings are retained. Imported values, arithmetic,
    rebinding, and function-local assignments remain unresolved so discovery
    cannot mistake runtime configuration for a statically fixed capacity.

    Args:
        tree: Parsed module containing scheduler workflows.

    Returns:
        dict[str, int]: Names bound exactly once to integer literals.
    """
    values: dict[str, int] = {}
    invalid: set[str] = set()
    for statement in tree.body:
        name: str | None = None
        value: ast.expr | None = None
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            name = statement.targets[0].id
            value = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            name = statement.target.id
            value = statement.value
        if name is None:
            continue
        if name in values or name in invalid:
            values.pop(name, None)
            invalid.add(name)
            continue
        if isinstance(value, ast.Constant) and type(value.value) is int:
            values[name] = value.value
        else:
            invalid.add(name)
    return values


def _transport_escapes(
    nodes: tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...],
    endpoints: tuple[str, ...],
    spawns: tuple[SpawnCall, ...],
) -> bool:
    endpoint_references = frozenset(endpoints)
    local_names = frozenset(endpoint for endpoint in endpoints if "." not in endpoint)
    for node in nodes:
        for child in _walk_function_scope(node):
            if isinstance(child, ast.Return | ast.Yield | ast.YieldFrom) and (
                _expression_exposes_references(child.value, endpoint_references)
            ):
                return True
            if isinstance(child, ast.Assign) and _expression_exposes_references(
                child.value,
                endpoint_references,
            ):
                return True
            if isinstance(child, ast.AnnAssign) and _expression_exposes_references(
                child.value,
                endpoint_references,
            ):
                return True
            if isinstance(child, ast.Call) and any(
                _expression_reads_references(argument, endpoint_references)
                for argument in (
                    *child.args,
                    *(keyword.value for keyword in child.keywords),
                )
            ):
                path = _call_path(child)
                is_local_consumer = path == ("anext",)
                is_lifecycle_registration = path is not None and path[-1] == "enter_async_context"
                if not (
                    is_local_consumer
                    or is_lifecycle_registration
                    or _call_is_spawn_handoff(child, spawns)
                ):
                    return True
            if isinstance(child, ast.Global | ast.Nonlocal) and local_names.intersection(
                child.names
            ):
                return True
    return False


def _transport_scope_nodes(
    owner: SymbolRecord,
    owner_node: ast.FunctionDef | ast.AsyncFunctionDef,
    transport: _TransportEvidence,
    context: _DiscoveryContext,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...]:
    if transport.scope == "local":
        return (owner_node,)
    class_name = _owner_class_name(owner)
    if class_name is None:
        return (owner_node,)
    return tuple(node for _, node in _class_method_nodes(class_name, context))


def _transport_has_stable_ownership(
    nodes: tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...],
    endpoints: tuple[str, ...],
) -> bool:
    writes = dict.fromkeys(endpoints, 0)
    for node in nodes:
        for child in _walk_function_scope(node):
            targets: tuple[ast.expr, ...] = ()
            if isinstance(child, ast.Assign):
                targets = tuple(child.targets)
            elif isinstance(child, ast.AnnAssign | ast.AugAssign | ast.NamedExpr):
                targets = (child.target,)
            elif isinstance(child, ast.Delete):
                targets = tuple(child.targets)
            for target in targets:
                for reference in _target_references(target):
                    if reference in writes:
                        writes[reference] += 1
    return bool(writes) and all(count == 1 for count in writes.values())


def _call_is_spawn_handoff(node: ast.Call, spawns: tuple[SpawnCall, ...]) -> bool:
    path = _call_path(node)
    for spawn in spawns:
        if (
            node.lineno == spawn.lineno
            and node.col_offset == spawn.col_offset
            and _scheduler_method(spawn) == (path[-1] if path is not None else None)
        ):
            return True
        if (
            spawn.dialect == "asyncio"
            and spawn.callee_name is not None
            and path is not None
            and ".".join(path) == spawn.callee_name
            and spawn.lineno <= node.lineno <= spawn.end_lineno
        ):
            return True
    return False


def _target_references(node: ast.expr) -> tuple[str, ...]:
    if isinstance(node, ast.Tuple | ast.List):
        return tuple(reference for item in node.elts for reference in _target_references(item))
    reference = _stable_reference(node)
    return (reference,) if reference is not None else ()


def _task_handle_escapes(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    handles: set[str] = set()
    dialects = built_in_scheduler_dialects()
    for child in _walk_function_scope(node):
        if not isinstance(child, ast.Assign | ast.AnnAssign):
            continue
        value = child.value
        if not isinstance(value, ast.Call):
            continue
        if not any(dialect.recognize_spawn(value) is not None for dialect in dialects):
            continue
        targets = child.targets if isinstance(child, ast.Assign) else (child.target,)
        handles.update(target.id for target in targets if isinstance(target, ast.Name))
    if not handles:
        return False
    return any(
        isinstance(child, ast.Return) and _expression_exposes_names(child.value, frozenset(handles))
        for child in _walk_function_scope(node)
    )


def _expression_reads_references(
    node: ast.AST | None,
    references: frozenset[str],
) -> bool:
    if node is None:
        return False
    return any(
        isinstance(child, ast.expr)
        and isinstance(getattr(child, "ctx", None), ast.Load)
        and _stable_reference(child) in references
        for child in ast.walk(node)
    )


def _expression_exposes_names(node: ast.expr | None, names: frozenset[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id in names
    if isinstance(node, ast.Starred):
        return _expression_exposes_names(node.value, names)
    if isinstance(node, ast.Tuple | ast.List | ast.Set):
        return any(_expression_exposes_names(item, names) for item in node.elts)
    if isinstance(node, ast.Dict):
        return any(
            _expression_exposes_names(item, names)
            for item in (*node.keys, *node.values)
            if item is not None
        )
    return False


def _expression_exposes_references(
    node: ast.expr | None,
    references: frozenset[str],
) -> bool:
    if node is None:
        return False
    reference = _stable_reference(node)
    if reference is not None:
        return reference in references
    if isinstance(node, ast.Starred):
        return _expression_exposes_references(node.value, references)
    if isinstance(node, ast.Tuple | ast.List | ast.Set):
        return any(_expression_exposes_references(item, references) for item in node.elts)
    if isinstance(node, ast.Dict):
        return any(
            _expression_exposes_references(item, references)
            for item in (*node.keys, *node.values)
            if item is not None
        )
    return False


def _structured_task_groups(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, tuple[_TaskGroupScope, ...]]:
    groups: dict[str, list[_TaskGroupScope]] = {}
    bindings = _stable_task_group_factory_bindings(node)
    for child in _walk_function_scope(node):
        if not isinstance(child, ast.AsyncWith):
            continue
        for item in child.items:
            context_reference = _stable_reference(item.context_expr)
            dialect = _task_group_factory_dialect(item.context_expr)
            if dialect is None and context_reference is not None:
                binding = bindings.get(context_reference)
                if binding is not None and binding[1] < child.lineno:
                    dialect = binding[0]
            if dialect is None:
                continue
            optional_reference = _stable_reference(item.optional_vars)
            receivers = tuple(
                dict.fromkeys(
                    reference
                    for reference in (context_reference, optional_reference)
                    if reference is not None
                )
            )
            for receiver in receivers:
                groups.setdefault(receiver, []).append(
                    _TaskGroupScope(
                        dialect=dialect,
                        lineno=child.lineno,
                        end_lineno=getattr(child, "end_lineno", child.lineno),
                    )
                )
    return {name: tuple(scopes) for name, scopes in groups.items()}


def _stable_task_group_factory_bindings(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, tuple[str, int]]:
    writes: dict[str, list[tuple[str | None, int]]] = {}
    for child in _walk_function_scope(node):
        targets, value, lineno = _assignment_parts(child)
        if not targets:
            continue
        dialect = _task_group_factory_dialect(value)
        for target in targets:
            for reference in _target_references(target):
                writes.setdefault(reference, []).append((dialect, lineno))
    bindings: dict[str, tuple[str, int]] = {}
    for reference, entries in writes.items():
        if len(entries) != 1:
            continue
        dialect, lineno = entries[0]
        if dialect is not None:
            bindings[reference] = (dialect, lineno)
    return bindings


def _assignment_parts(
    statement: ast.AST,
) -> tuple[tuple[ast.expr, ...], ast.expr | None, int]:
    if isinstance(statement, ast.Assign):
        return tuple(statement.targets), statement.value, statement.lineno
    if isinstance(statement, ast.AnnAssign | ast.AugAssign | ast.NamedExpr):
        return (statement.target,), statement.value, statement.lineno
    if isinstance(statement, ast.Delete):
        return tuple(statement.targets), None, statement.lineno
    return (), None, 0


def _task_group_factory_dialect(node: ast.expr | None) -> str | None:
    path = _call_path(node)
    if path == ("asyncio", "TaskGroup"):
        return "asyncio"
    if path in (("anyio", "create_task_group"), ("create_task_group",)):
        return "anyio-on-asyncio"
    return None


def _scheduler_ownership_for_spawn(
    owner: SymbolRecord,
    spawn: SpawnCall,
    task_groups: Mapping[str, tuple[_TaskGroupScope, ...]],
    context: _DiscoveryContext,
) -> _SchedulerOwnership | None:
    if _spawn_in_structured_scope(spawn, task_groups):
        if spawn.scheduler_owner is None:
            return None
        return _SchedulerOwnership(
            identity=f"{owner.id.stable_id}::{spawn.scheduler_owner}",
            evidence_members=(owner.id,),
        )
    return _scheduler_field_ownership(owner, spawn, context)


def _scheduler_field_ownership(
    owner: SymbolRecord,
    spawn: SpawnCall,
    context: _DiscoveryContext,
) -> _SchedulerOwnership | None:
    field = _scheduler_field(owner, spawn, context)
    if field is None:
        return None
    bindings = _field_bindings(field.class_name, field.reference, context)
    if not bindings:
        return _dataclass_scheduler_field_ownership(field)
    if len(bindings) != 1:
        return None
    binding = bindings[0]
    if _field_binding_factory_dialect(binding) == spawn.dialect:
        return _factory_scheduler_field_ownership(field, binding, context)
    if not _field_binding_is_delegated(binding):
        return None
    return _SchedulerOwnership(
        identity=f"{field.class_name}::{field.reference}",
        evidence_members=_unique_symbol_ids((binding.symbol.id, *field.class_evidence)),
    )


def _scheduler_field(
    owner: SymbolRecord,
    spawn: SpawnCall,
    context: _DiscoveryContext,
) -> _SchedulerField | None:
    if spawn.dialect != "anyio-on-asyncio" or spawn.scheduler_owner is None:
        return None
    receiver_parts = spawn.scheduler_owner.split(".")
    class_name = _owner_class_name(owner)
    if (
        len(receiver_parts) != _INSTANCE_RECEIVER_PARTS
        or receiver_parts[0] != "self"
        or class_name is None
    ):
        return None
    class_node = context.class_nodes.get(class_name)
    if class_node is None:
        return None
    class_symbol = context.symbol_map.get(class_name)
    return _SchedulerField(
        class_name=class_name,
        reference=spawn.scheduler_owner,
        class_node=class_node,
        annotation=_class_field_annotation(class_node, receiver_parts[1]),
        class_evidence=(class_symbol.id,) if class_symbol is not None else (),
    )


def _dataclass_scheduler_field_ownership(
    field: _SchedulerField,
) -> _SchedulerOwnership | None:
    annotation = field.annotation
    if (
        annotation is None
        or not _annotation_contains_task_group(annotation.annotation)
        or not _class_is_dataclass(field.class_node)
        or not _dataclass_field_is_init(annotation.value)
    ):
        return None
    return _SchedulerOwnership(
        identity=f"{field.class_name}::{field.reference}",
        evidence_members=field.class_evidence,
    )


def _field_binding_factory_dialect(binding: _FieldBinding) -> str | None:
    dialect = _task_group_factory_dialect(binding.value)
    if dialect is not None or not isinstance(binding.value, ast.Name):
        return dialect
    alias = _stable_task_group_factory_bindings(binding.node).get(binding.value.id)
    return alias[0] if alias is not None else None


def _factory_scheduler_field_ownership(
    field: _SchedulerField,
    binding: _FieldBinding,
    context: _DiscoveryContext,
) -> _SchedulerOwnership | None:
    lifecycle_members = _scheduler_field_lifecycle_members(
        field.class_name,
        field.reference,
        context,
    )
    if not lifecycle_members:
        return None
    return _SchedulerOwnership(
        identity=f"{field.class_name}::{field.reference}",
        evidence_members=_unique_symbol_ids((binding.symbol.id, *lifecycle_members)),
    )


def _field_binding_is_delegated(binding: _FieldBinding) -> bool:
    if not isinstance(binding.value, ast.Name):
        return False
    parameter_annotation = _parameter_annotation(binding.node, binding.value.id)
    return _annotation_contains_task_group(parameter_annotation)


def _class_field_annotation(class_node: ast.ClassDef, field_name: str) -> ast.AnnAssign | None:
    matches = tuple(
        statement
        for statement in class_node.body
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
        and statement.target.id == field_name
    )
    return matches[0] if len(matches) == 1 else None


def _field_bindings(
    class_name: str,
    field_reference: str,
    context: _DiscoveryContext,
) -> tuple[_FieldBinding, ...]:
    bindings: list[_FieldBinding] = []
    for qualname, method_node in _class_method_nodes(class_name, context):
        symbol = context.symbol_map.get(qualname)
        if symbol is None:
            continue
        for child in _walk_function_scope(method_node):
            binding = _field_binding_from_statement(
                symbol,
                method_node,
                child,
                field_reference,
            )
            if binding is not None:
                bindings.append(binding)
    return tuple(bindings)


def _field_binding_from_statement(
    symbol: SymbolRecord,
    method_node: ast.FunctionDef | ast.AsyncFunctionDef,
    statement: ast.AST,
    field_reference: str,
) -> _FieldBinding | None:
    targets: tuple[ast.expr, ...] = ()
    value: ast.expr | None = None
    annotation: ast.expr | None = None
    if isinstance(statement, ast.Assign):
        targets = tuple(statement.targets)
        value = statement.value
    elif isinstance(statement, ast.AnnAssign):
        targets = (statement.target,)
        value = statement.value
        annotation = statement.annotation
    elif isinstance(statement, ast.AugAssign | ast.NamedExpr):
        targets = (statement.target,)
    elif isinstance(statement, ast.Delete):
        targets = tuple(statement.targets)
    if not any(field_reference in _target_references(target) for target in targets):
        return None
    return _FieldBinding(
        symbol=symbol,
        node=method_node,
        value=value,
        annotation=annotation,
    )


def _annotation_contains_task_group(node: ast.expr | None) -> bool:
    if node is None:
        return False
    path = _attribute_path(node)
    if path is not None:
        return path[-1] == "TaskGroup"
    if isinstance(node, ast.Subscript):
        return _annotation_contains_task_group(node.value) or _annotation_contains_task_group(
            node.slice
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _annotation_contains_task_group(node.left) or _annotation_contains_task_group(
            node.right
        )
    if isinstance(node, ast.Tuple):
        return any(_annotation_contains_task_group(item) for item in node.elts)
    return False


def _class_is_dataclass(node: ast.ClassDef) -> bool:
    return any(
        (path := _attribute_path(decorator)) is not None and path[-1] == "dataclass"
        for decorator in node.decorator_list
    )


def _dataclass_field_is_init(node: ast.expr | None) -> bool:
    if node is None:
        return True
    if not isinstance(node, ast.Call):
        return False
    path = _call_path(node)
    if path is None or path[-1] != "field":
        return False
    return not any(
        keyword.arg == "init"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is False
        for keyword in node.keywords
    )


def _parameter_annotation(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    parameter_name: str,
) -> ast.expr | None:
    arguments = (
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
        *((node.args.vararg,) if node.args.vararg is not None else ()),
        *((node.args.kwarg,) if node.args.kwarg is not None else ()),
    )
    for argument in arguments:
        if argument.arg == parameter_name:
            annotation = argument.annotation
            return annotation if isinstance(annotation, ast.expr) else None
    return None


def _scheduler_field_lifecycle_members(
    class_name: str,
    field_reference: str,
    context: _DiscoveryContext,
) -> tuple[SymbolId, ...]:
    uses: list[tuple[SymbolId, _LifecycleUse]] = []
    for qualname, method_node in _class_method_nodes(class_name, context):
        symbol = context.symbol_map.get(qualname)
        if symbol is not None:
            uses.append((symbol.id, _lifecycle_use(method_node, field_reference)))
    managed_members = tuple(symbol for symbol, use in uses if use.managed_context)
    if managed_members:
        return _unique_symbol_ids(managed_members)
    direct_enter = tuple(symbol for symbol, use in uses if use.direct_enter)
    direct_exit = tuple(symbol for symbol, use in uses if use.direct_exit)
    if direct_enter and direct_exit:
        return _unique_symbol_ids((*direct_enter, *direct_exit))
    stack_references = {stack_reference for _, use in uses for stack_reference in use.stack_enters}
    for stack_reference in stack_references:
        enter_members = tuple(symbol for symbol, use in uses if stack_reference in use.stack_enters)
        exit_members = tuple(symbol for symbol, use in uses if stack_reference in use.stack_exits)
        if exit_members:
            return _unique_symbol_ids((*enter_members, *exit_members))
    return ()


def _lifecycle_use(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    field_reference: str,
) -> _LifecycleUse:
    managed_context = any(
        isinstance(child, ast.AsyncWith)
        and any(_stable_reference(item.context_expr) == field_reference for item in child.items)
        and any(isinstance(item, ast.Yield | ast.YieldFrom) for item in ast.walk(child))
        for child in _walk_function_scope(node)
    )
    calls = tuple(
        child.value
        for child in _walk_function_scope(node)
        if isinstance(child, ast.Await) and isinstance(child.value, ast.Call)
    )
    paths = tuple((call, _call_path(call)) for call in calls)
    field_path = tuple(field_reference.split("."))
    stack_enters = tuple(
        ".".join(path[:-1])
        for call, path in paths
        if path is not None
        and path[-1] == "enter_async_context"
        and call.args
        and _stable_reference(call.args[0]) == field_reference
    )
    stack_exits = tuple(
        ".".join(path[:-1]) for _, path in paths if path is not None and path[-1] == "__aexit__"
    )
    return _LifecycleUse(
        managed_context=managed_context,
        direct_enter=any(path == (*field_path, "__aenter__") for _, path in paths),
        direct_exit=any(path == (*field_path, "__aexit__") for _, path in paths),
        stack_enters=stack_enters,
        stack_exits=stack_exits,
    )


def _spawn_in_structured_scope(
    spawn: SpawnCall,
    task_groups: Mapping[str, tuple[_TaskGroupScope, ...]],
) -> bool:
    if spawn.scheduler_owner is None:
        return False
    return any(
        scope.dialect == spawn.dialect and scope.lineno <= spawn.lineno <= scope.end_lineno
        for scope in task_groups.get(spawn.scheduler_owner, ())
    )


def _walk_function_scope(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> Iterable[ast.AST]:
    pending: list[ast.AST] = list(reversed(node.body))
    while pending:
        child = pending.pop()
        yield child
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
            continue
        pending.extend(reversed(tuple(ast.iter_child_nodes(child))))


def _unique_symbol_ids(symbols: Iterable[SymbolId]) -> tuple[SymbolId, ...]:
    return tuple(dict.fromkeys(symbols))


def _unique_symbols(symbols: Iterable[SymbolRecord]) -> tuple[SymbolRecord, ...]:
    unique: dict[SymbolId, SymbolRecord] = {}
    for symbol in symbols:
        unique.setdefault(symbol.id, symbol)
    return tuple(unique.values())


def _ranked_sites(
    scans: tuple[ModuleScan, ...],
    profile: ProfileResult | None,
    lifecycle_starts: int,
) -> Iterable[_RankedSite]:
    if profile is None:
        return tuple(
            _RankedSite(
                scan=scan,
                site=site,
                hotness=0,
                observed_invocations=0,
                lifecycle_starts=0,
                start_share=0.0,
                guarded_callable_identities=(),
            )
            for scan in scans
            for site in _candidate_sites(scan)
        )
    spawn_invocations = {site.target.id: site.invocation_count for site in profile.spawn_sites}
    spawn_identities = {
        site.target.id: tuple(item.identity for item in site.callable_identities if item.count > 0)
        for site in profile.spawn_sites
    }
    starts_by_symbol = {member.symbol: member.lifecycle.start for member in profile.members}
    ranked: list[_RankedSite] = []
    for scan in scans:
        for site in _candidate_sites(scan):
            spawned_consumer = (
                (site.consumer,)
                if site.consumer_spawned
                and site.consumer is not None
                and site.consumer != site.owner.id
                else ()
            )
            spawned = (*site.producers, *spawned_consumer)
            per_spawn_invocations = tuple(
                spawn_invocations.get(
                    _spawn_target_id(scan.module.name, site.owner.id, spawn),
                    0,
                )
                for spawn in site.spawns
            )
            observed_invocations = max(per_spawn_invocations, default=0)
            spawn_activity = sum(per_spawn_invocations)
            guarded_identities = tuple(
                dict.fromkeys(
                    identity
                    for spawn in site.spawns
                    for identity in spawn_identities.get(
                        _spawn_target_id(scan.module.name, site.owner.id, spawn),
                        (),
                    )
                )
            )
            starts = sum(starts_by_symbol.get(symbol, 0) for symbol in spawned)
            hotness = max(observed_invocations, starts)
            share = spawn_activity / lifecycle_starts if lifecycle_starts > 0 else 0.0
            ranked.append(
                _RankedSite(
                    scan=scan,
                    site=site,
                    hotness=hotness,
                    observed_invocations=observed_invocations,
                    lifecycle_starts=starts,
                    start_share=share,
                    guarded_callable_identities=guarded_identities,
                )
            )
    return tuple(
        sorted(
            ranked,
            key=lambda site: (-site.hotness, site.scan.module.name, site.site.owner.id.qualname),
        )
    )


def _mapped_lifecycle_starts(scans: tuple[ModuleScan, ...], profile: ProfileResult | None) -> int:
    if profile is None:
        return 0
    spawn_activity = sum(site.invocation_count for site in profile.spawn_sites)
    if spawn_activity > 0:
        return spawn_activity
    symbols = {symbol.id for scan in scans for symbol in scan.symbols}
    return sum(member.lifecycle.start for member in profile.members if member.symbol in symbols)


def _is_hot(ranked_site: _RankedSite) -> bool:
    return (
        ranked_site.observed_invocations >= _MIN_INVOCATIONS
        and ranked_site.start_share >= _MIN_LIFECYCLE_START_SHARE
    )


def _runtime_scheduler_rejection(
    ranked_site: _RankedSite,
    profile: ProfileResult | None,
) -> PlanRejection | None:
    if profile is None:
        return None
    observations = {site.target.id: site for site in profile.spawn_sites}
    for spawn in ranked_site.site.spawns:
        target_id = _spawn_target_id(
            ranked_site.scan.module.name,
            ranked_site.site.owner.id,
            spawn,
        )
        observation = observations.get(target_id)
        if observation is None or observation.invocation_count == 0:
            continue
        identities = tuple(
            item.identity for item in observation.callable_identities if item.count > 0
        )
        expected_suffix = f".{_scheduler_method(spawn)}"
        expected_prefix = "anyio." if spawn.dialect == "anyio-on-asyncio" else "asyncio."
        if (
            len(identities) != 1
            or not identities[0].startswith(expected_prefix)
            or not identities[0].endswith(expected_suffix)
        ):
            return _rejection_for(
                ranked_site.scan,
                ranked_site.site,
                ranked_site.hotness,
                reason="dynamic-scheduler",
                message=(
                    "runtime scheduler callable identity was dynamic or did not match the dialect"
                ),
            )
    return None


def _reflection_candidates(
    scans: tuple[ModuleScan, ...],
) -> tuple[_ReflectionCandidate, ...]:
    """Find exact repeated ``len(inspect.signature(...).parameters)`` sites.

    The lowering keeps arbitrary callables interpreted. It only caches the
    parameter count later when the runtime callable is an unwrapped Python
    function whose code and reflection implementation remain unchanged.

    Args:
        scans: Project modules available to execution-plan discovery.

    Returns:
        tuple[_ReflectionCandidate, ...]: Stable callable and source-line evidence.
    """
    candidates: list[_ReflectionCandidate] = []
    for scan in scans:
        source = scan.module.path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(scan.module.path), type_comments=True)
        inspect_modules, signature_names = _inspect_signature_bindings(tree)
        if not inspect_modules and not signature_names:
            continue
        nodes = _function_nodes(tree)
        for symbol in scan.symbols:
            if symbol.kind not in {"function", "method"}:
                continue
            node = nodes.get(symbol.id.qualname)
            if node is None or _callable_binds_name(node, "len"):
                continue
            linenos = tuple(
                assignment.lineno
                for assignment in ast.walk(node)
                if isinstance(assignment, ast.Assign)
                and _signature_parameter_count_assignment(
                    assignment,
                    inspect_modules=inspect_modules,
                    signature_names=signature_names,
                )
            )
            if len(linenos) == 1:
                candidates.append(_ReflectionCandidate(symbol=symbol, lineno=linenos[0]))
    return tuple(
        sorted(candidates, key=lambda item: (item.symbol.id.module, item.symbol.id.qualname))
    )


def _inspect_signature_bindings(tree: ast.Module) -> tuple[frozenset[str], frozenset[str]]:
    modules: set[str] = set()
    functions: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                if alias.name == "inspect":
                    modules.add(alias.asname or alias.name)
        elif isinstance(statement, ast.ImportFrom) and statement.module == "inspect":
            for alias in statement.names:
                if alias.name == "signature":
                    functions.add(alias.asname or alias.name)
    return frozenset(modules), frozenset(functions)


def _callable_binds_name(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    name: str,
) -> bool:
    arguments = (
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
        *((node.args.vararg,) if node.args.vararg is not None else ()),
        *((node.args.kwarg,) if node.args.kwarg is not None else ()),
    )
    if any(argument.arg == name for argument in arguments):
        return True
    return any(
        isinstance(candidate, ast.Name)
        and candidate.id == name
        and isinstance(candidate.ctx, ast.Store)
        for candidate in _walk_without_nested_declarations(node)
    )


def _signature_parameter_count_assignment(
    assignment: ast.Assign,
    *,
    inspect_modules: frozenset[str],
    signature_names: frozenset[str],
) -> bool:
    if len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Name):
        return False
    value = assignment.value
    if (
        not isinstance(value, ast.Call)
        or not isinstance(value.func, ast.Name)
        or value.func.id != "len"
        or len(value.args) != 1
        or value.keywords
        or not isinstance(value.args[0], ast.Attribute)
        or value.args[0].attr != "parameters"
        or not isinstance(value.args[0].value, ast.Call)
    ):
        return False
    signature_call = value.args[0].value
    if len(signature_call.args) != 1 or signature_call.keywords:
        return False
    function = signature_call.func
    return (isinstance(function, ast.Name) and function.id in signature_names) or (
        isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Name)
        and function.value.id in inspect_modules
        and function.attr == "signature"
    )


def _linked_hot_reflection_reducers(
    ranked_site: _RankedSite,
    scans: tuple[ModuleScan, ...],
    profile: ProfileResult | None,
) -> tuple[_ReflectionCandidate, ...]:
    if profile is None:
        return ()
    candidates = _reflection_candidates(scans)
    if not candidates:
        return ()
    linked = _linked_reflection_symbols(ranked_site, scans, candidates)
    invocation_counts = {
        member.symbol: max(member.invocation_count, member.call_count, member.lifecycle.start)
        for member in profile.members
    }
    return tuple(
        candidate
        for candidate in candidates
        if candidate.symbol.id in linked
        and invocation_counts.get(candidate.symbol.id, 0) >= _MIN_INVOCATIONS
    )


def _linked_reflection_symbols(
    ranked_site: _RankedSite,
    scans: tuple[ModuleScan, ...],
    candidates: tuple[_ReflectionCandidate, ...],
) -> frozenset[SymbolId]:
    consumer = ranked_site.site.consumer or ranked_site.site.owner.id
    if consumer.module != ranked_site.scan.module.name:
        return frozenset()
    source = ranked_site.scan.module.path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ranked_site.scan.module.path), type_comments=True)
    node = _function_nodes(tree).get(consumer.qualname)
    if node is None:
        return frozenset()
    class_symbols = {
        symbol.id for scan in scans for symbol in scan.symbols if symbol.kind == "class"
    }
    imported_classes = _imported_class_bindings(
        tree,
        ranked_site.scan.module.name,
        class_symbols,
    )
    candidate_ids = {candidate.symbol.id for candidate in candidates}
    return frozenset(
        symbol
        for symbol in _narrowed_method_calls(node, imported_classes)
        if symbol in candidate_ids
    )


def _imported_class_bindings(
    tree: ast.Module,
    current_module: str,
    class_symbols: set[SymbolId],
) -> dict[str, SymbolId]:
    bindings: dict[str, SymbolId] = {}
    package_parts = current_module.split(".")[:-1]
    for statement in tree.body:
        if not isinstance(statement, ast.ImportFrom) or statement.module is None:
            continue
        if statement.level:
            keep = max(0, len(package_parts) - statement.level + 1)
            module = ".".join((*package_parts[:keep], statement.module))
        else:
            module = statement.module
        for alias in statement.names:
            symbol = SymbolId(module=module, qualname=alias.name)
            if symbol in class_symbols:
                bindings[alias.asname or alias.name] = symbol
    return bindings


def _narrowed_method_calls(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    imported_classes: dict[str, SymbolId],
) -> tuple[SymbolId, ...]:
    calls: list[SymbolId] = []

    def visit_block(statements: list[ast.stmt], narrowed: dict[str, SymbolId]) -> None:
        local = dict(narrowed)
        for statement in statements:
            for expression in _statement_expressions(statement):
                for candidate in ast.walk(expression):
                    if (
                        isinstance(candidate, ast.Call)
                        and isinstance(candidate.func, ast.Attribute)
                        and isinstance(candidate.func.value, ast.Name)
                        and candidate.func.value.id in local
                    ):
                        owner = local[candidate.func.value.id]
                        calls.append(
                            SymbolId(
                                module=owner.module,
                                qualname=f"{owner.qualname}.{candidate.func.attr}",
                            )
                        )
            narrowed_binding = _isinstance_narrowing(statement, imported_classes)
            if narrowed_binding is not None:
                local[narrowed_binding[0]] = narrowed_binding[1]
            for child_block in _statement_blocks(statement):
                visit_block(child_block, local)
            for assigned_name in _statement_assigned_names(statement):
                if narrowed_binding is None or assigned_name != narrowed_binding[0]:
                    local.pop(assigned_name, None)

    visit_block(node.body, {})
    return tuple(dict.fromkeys(calls))


def _isinstance_narrowing(
    statement: ast.stmt,
    imported_classes: dict[str, SymbolId],
) -> tuple[str, SymbolId] | None:
    if (
        not isinstance(statement, ast.Assert)
        or not isinstance(statement.test, ast.Call)
        or not isinstance(statement.test.func, ast.Name)
        or statement.test.func.id != "isinstance"
        or len(statement.test.args) != _ISINSTANCE_ARGUMENTS
        or statement.test.keywords
        or not isinstance(statement.test.args[0], ast.Name)
        or not isinstance(statement.test.args[1], ast.Name)
    ):
        return None
    class_symbol = imported_classes.get(statement.test.args[1].id)
    if class_symbol is None:
        return None
    return statement.test.args[0].id, class_symbol


def _statement_expressions(statement: ast.stmt) -> tuple[ast.expr, ...]:
    expression: ast.expr | None = None
    match statement:
        case ast.Assign(value=value) | ast.AugAssign(value=value) | ast.Expr(value=value):
            expression = value
        case ast.AnnAssign(value=value) | ast.Return(value=value) if value is not None:
            expression = value
        case ast.Assert(test=test) | ast.If(test=test) | ast.While(test=test):
            expression = test
        case ast.For(iter=iterable) | ast.AsyncFor(iter=iterable):
            expression = iterable
        case _:
            pass
    return (expression,) if expression is not None else ()


def _statement_blocks(statement: ast.stmt) -> tuple[list[ast.stmt], ...]:
    blocks: list[list[ast.stmt]] = []
    if isinstance(statement, ast.If | ast.For | ast.AsyncFor | ast.While):
        blocks.extend(block for block in (statement.body, statement.orelse) if block)
    elif isinstance(statement, ast.With | ast.AsyncWith):
        if statement.body:
            blocks.append(statement.body)
    elif isinstance(statement, ast.Try | ast.TryStar):
        blocks.extend(
            block for block in (statement.body, statement.orelse, statement.finalbody) if block
        )
        blocks.extend(handler.body for handler in statement.handlers if handler.body)
    elif isinstance(statement, ast.Match):
        blocks.extend(case.body for case in statement.cases if case.body)
    return tuple(blocks)


def _statement_assigned_names(statement: ast.stmt) -> frozenset[str]:
    expressions: list[ast.AST] = []
    if isinstance(statement, ast.Assign):
        expressions.extend(statement.targets)
    elif isinstance(statement, ast.AnnAssign | ast.AugAssign | ast.For | ast.AsyncFor):
        expressions.append(statement.target)
    return frozenset(
        candidate.id
        for expression in expressions
        for candidate in ast.walk(expression)
        if isinstance(candidate, ast.Name) and isinstance(candidate.ctx, ast.Store)
    )


def _walk_without_nested_declarations(node: ast.AST) -> Iterable[ast.AST]:
    pending = list(ast.iter_child_nodes(node))
    while pending:
        child = pending.pop()
        yield child
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
            continue
        pending.extend(ast.iter_child_nodes(child))


def _plan_for(
    ranked_site: _RankedSite,
    scans: tuple[ModuleScan, ...],
    profile: ProfileResult | None,
) -> ExecutionPlan:
    scan = ranked_site.scan
    site = ranked_site.site
    owner_id = site.owner.id
    consumer = site.consumer if site.consumer is not None else owner_id
    consumer_is_owner = consumer == owner_id
    reducers = _linked_hot_reflection_reducers(ranked_site, scans, profile)
    transport_name = site.transport.identity if site.transport is not None else ""
    transport_node_id = f"{owner_id.stable_id}::transport::{transport_name}"
    consumer_node_id = f"{owner_id.stable_id}::reducer" if consumer_is_owner else consumer.stable_id
    producer_nodes = tuple(
        PlanNode(
            id=producer.stable_id,
            symbol=producer,
            role="producer",
            lineno=_symbol_lineno(scan, producer),
        )
        for producer in site.producers
    )
    reducer_nodes = tuple(
        PlanNode(
            id=reducer.symbol.id.stable_id,
            symbol=reducer.symbol.id,
            role="reducer",
            lineno=reducer.lineno,
        )
        for reducer in reducers
    )
    nodes = (
        PlanNode(
            id=owner_id.stable_id,
            symbol=owner_id,
            role="orchestrator",
            lineno=site.owner.lineno,
        ),
        *producer_nodes,
        PlanNode(id=transport_node_id, symbol=None, role="transport", lineno=site.spawns[0].lineno),
        PlanNode(
            id=consumer_node_id,
            symbol=consumer,
            role="reducer" if consumer_is_owner else "consumer",
            lineno=_symbol_lineno(scan, consumer),
        ),
        *reducer_nodes,
    )
    spawn_edges = tuple(
        PlanEdge(
            src=owner_id.stable_id,
            dst=producer.stable_id,
            kind="spawns",
            transport=transport_name,
            lineno=_spawn_lineno_for(site, producer),
        )
        for producer in site.producers
    )
    producer_edges = tuple(
        PlanEdge(
            src=producer.stable_id,
            dst=transport_node_id,
            kind="produces",
            transport=transport_name,
            lineno=_spawn_lineno_for(site, producer),
        )
        for producer in site.producers
    )
    reducer_edges = tuple(
        PlanEdge(
            src=consumer_node_id,
            dst=reducer.symbol.id.stable_id,
            kind="reduces",
            transport=transport_name,
            lineno=reducer.lineno,
        )
        for reducer in reducers
    )
    edges = (
        *spawn_edges,
        *(
            ()
            if consumer_is_owner or not site.consumer_spawned
            else (
                PlanEdge(
                    src=owner_id.stable_id,
                    dst=consumer_node_id,
                    kind="spawns",
                    transport=transport_name,
                    lineno=site.spawns[-1].lineno,
                ),
            )
        ),
        *producer_edges,
        PlanEdge(
            src=transport_node_id,
            dst=consumer_node_id,
            kind="delivers",
            transport=transport_name,
            lineno=site.spawns[0].lineno,
        ),
        *reducer_edges,
    )
    source_members = _source_members(scan, site, reducers)
    source_hashes = _source_hashes(scans, source_members)
    source_hash = _digest_parts(f"{module}:{digest}" for module, digest in source_hashes)
    callsite_fingerprint = _callsite_fingerprint(site)
    topology_fingerprint = _topology_fingerprint(nodes, edges)
    plan_id = stable_execution_plan_id(
        ExecutionPlanIdentity(
            source_module=scan.module.name,
            source_hash=source_hash,
            callsite_fingerprint=callsite_fingerprint,
            topology_fingerprint=topology_fingerprint,
            dialect=site.dialect.name,
            lowering_version=site.dialect.lowering_version,
            guarded_callable_identities=ranked_site.guarded_callable_identities,
        )
    )
    return ExecutionPlan(
        id=plan_id,
        source_module=scan.module.name,
        owner=owner_id,
        dialect=site.dialect.name,
        lowering_version=site.dialect.lowering_version,
        source_hash=source_hash,
        callsite_fingerprint=callsite_fingerprint,
        topology_fingerprint=topology_fingerprint,
        nodes=nodes,
        edges=edges,
        guards=(
            PlanGuard(
                kind="scheduler",
                expression=site.dialect.name,
                message="scheduler dialect must match the discovered spawn semantics",
            ),
            PlanGuard(
                kind="transport",
                expression=transport_name,
                message="private transport must remain owned by the orchestration site",
            ),
            PlanGuard(
                kind="topology",
                expression="structured-task-group",
                message="all spawned work must remain joined before scope exit",
            ),
        ),
        completion_transport=transport_name,
        consumer=consumer,
        reducer=reducers[0].symbol.id if len(reducers) == 1 else consumer,
        transport_capacity=site.transport.capacity.value if site.transport is not None else None,
        ordering_policy="completion-order",
        task_ownership="structured",
        observed_invocations=ranked_site.observed_invocations,
        lifecycle_starts=ranked_site.lifecycle_starts,
        lifecycle_share=ranked_site.start_share,
        guarded_callable_identities=ranked_site.guarded_callable_identities,
        source_members=source_members,
        source_hashes=source_hashes,
        hotness=ranked_site.hotness,
    )


def _rejection_for(
    scan: ModuleScan,
    site: _CandidateSite,
    hotness: int,
    *,
    reason: ExecutionPlanRejectionReason | None = None,
    message: str | None = None,
) -> PlanRejection:
    resolved_reason = reason if reason is not None else site.rejection_reason
    if resolved_reason is None:
        resolved_reason = "unknown-transport"
    resolved_message = message if message is not None else site.rejection_message
    if resolved_message is None:
        resolved_message = "execution-plan site was rejected"
    digest = hashlib.blake2b(digest_size=8)
    for part in (
        scan.module.name,
        site.owner.id.qualname,
        resolved_reason,
        str(site.spawns[0].lineno),
        site.dialect.name,
    ):
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return PlanRejection(
        id=f"exec-plan-rejection-{digest.hexdigest()}",
        source_module=scan.module.name,
        owner=site.owner.id,
        reason=resolved_reason,
        message=resolved_message,
        dialect=site.dialect.name,
        lineno=site.spawns[0].lineno,
        hotness=hotness,
    )


def _source_hashes(
    scans: tuple[ModuleScan, ...],
    member_ids: tuple[SymbolId, ...],
) -> tuple[tuple[str, str], ...]:
    """Hash every covered declaration independently per source module.

    Args:
        scans: Complete project scan scope used by execution-plan discovery.
        member_ids: Exact declarations covered by the plan.

    Returns:
        tuple[tuple[str, str], ...]: Sorted module names and complete source digests.

    Raises:
        ValueError: If a covered declaration is absent from the supplied scans.
    """
    scans_by_module = {scan.module.name: scan for scan in scans}
    grouped: dict[str, list[SymbolId]] = {}
    for member in member_ids:
        grouped.setdefault(member.module, []).append(member)
    hashes: list[tuple[str, str]] = []
    for module_name, module_members in sorted(grouped.items()):
        scan = scans_by_module.get(module_name)
        if scan is None:
            raise ValueError(f"execution-plan source module is outside scan scope: {module_name}")
        records = {record.id for record in scan.symbols}
        for member in sorted(module_members, key=lambda item: item.qualname):
            if member not in records:
                raise ValueError(f"execution-plan source member is missing: {member.stable_id}")
        source = scan.module.path.read_text(encoding="utf-8")
        hashes.append((module_name, hashlib.sha256(source.encode("utf-8")).hexdigest()))
    return tuple(hashes)


def _source_members(
    scan: ModuleScan,
    site: _CandidateSite,
    reducers: tuple[_ReflectionCandidate, ...] = (),
) -> tuple[SymbolId, ...]:
    members = {site.owner.id, *site.producers, *site.evidence_members}
    if site.consumer is not None:
        members.add(site.consumer)
        members.update(
            edge.dst
            for edge in scan.dependency_edges
            if edge.src == site.consumer
            and edge.confidence == "high"
            and isinstance(edge.dst, SymbolId)
            and edge.dst.module == scan.module.name
            and edge.kind == "calls_method"
        )
    members.update(reducer.symbol.id for reducer in reducers)
    return tuple(sorted(members, key=lambda member: (member.module, member.qualname)))


def _callsite_fingerprint(site: _CandidateSite) -> str:
    return _digest_parts(
        f"{spawn.dialect}:{spawn.lineno}:{spawn.col_offset}:{spawn.callee_name or ''}"
        for spawn in site.spawns
    )


def _topology_fingerprint(nodes: tuple[PlanNode, ...], edges: tuple[PlanEdge, ...]) -> str:
    return _digest_parts(
        (
            *(f"node:{node.id}:{node.role}" for node in nodes),
            *(f"edge:{edge.src}:{edge.dst}:{edge.kind}:{edge.transport or ''}" for edge in edges),
        )
    )


def _digest_parts(parts: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _function_nodes(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    nodes: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            nodes[node.name] = node
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                    nodes[f"{node.name}.{child.name}"] = child
    return nodes


def _class_nodes(tree: ast.Module) -> dict[str, ast.ClassDef]:
    return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}


def _owner_class_name(owner: SymbolRecord) -> str | None:
    class_name, separator, _ = owner.id.qualname.rpartition(".")
    if not separator or "." in class_name:
        return None
    return class_name


def _class_method_nodes(
    class_name: str,
    context: _DiscoveryContext,
) -> tuple[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef], ...]:
    prefix = f"{class_name}."
    return tuple(
        (qualname, node)
        for qualname, node in context.function_nodes.items()
        if qualname.startswith(prefix) and "." not in qualname[len(prefix) :]
    )


def _dialect_by_name(name: str) -> SchedulerDialect:
    for dialect in built_in_scheduler_dialects():
        if dialect.name == name:
            return dialect
    return built_in_scheduler_dialects()[0]


def _call_path(node: ast.expr | None) -> tuple[str, ...] | None:
    if not isinstance(node, ast.Call):
        return None
    function = node.func
    while isinstance(function, ast.Subscript):
        function = function.value
    return _attribute_path(function)


def _attribute_path(node: ast.expr) -> tuple[str, ...] | None:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        parent = _attribute_path(node.value)
        if parent is None:
            return None
        return (*parent, node.attr)
    return None


def _stable_reference(node: ast.expr | None) -> str | None:
    if node is None:
        return None
    path = _attribute_path(node)
    if path is None:
        return None
    return ".".join(path)


def _symbol_lineno(scan: ModuleScan, symbol: SymbolId) -> int:
    for record in scan.symbols:
        if record.id == symbol:
            return record.lineno
    return 0


def _spawn_lineno_for(site: _CandidateSite, symbol: SymbolId) -> int:
    for spawn in site.spawns:
        if _resolved_callee_name(site.owner, spawn.callee_name) == symbol.qualname:
            return spawn.lineno
    return site.spawns[0].lineno


def _spawn_target_id(module: str, owner: SymbolId, spawn: SpawnCall) -> str:
    digest = _digest_parts(
        (
            module,
            owner.qualname,
            spawn.dialect,
            str(spawn.lineno),
            str(spawn.col_offset),
            str(spawn.end_lineno),
            str(spawn.end_col_offset),
            _scheduler_method(spawn),
        )
    )
    return f"spawn-site-{digest[:24]}"


def _scheduler_method(spawn: SpawnCall) -> str:
    return "start_soon" if spawn.dialect == "anyio-on-asyncio" else "create_task"
