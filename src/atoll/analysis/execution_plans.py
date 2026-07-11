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


@dataclass(frozen=True, slots=True)
class _CandidateSite:
    owner: SymbolRecord
    dialect: SchedulerDialect
    spawns: tuple[SpawnCall, ...]
    transport: _TransportEvidence | None
    producers: tuple[SymbolId, ...]
    consumer: SymbolId | None
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
class _TransportEvidence:
    """Static identity shared by a queue or paired memory-stream endpoints.

    Attributes:
        identity: Stable local transport name or paired endpoint identity.
        endpoints: Local names through which producers and the consumer access the transport.
        capacity: Statically known positive result capacity, or zero for unbounded transports.
    """

    identity: str
    endpoints: tuple[str, ...]
    capacity: int


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


def execution_plan_observation_targets(scans: Iterable[ModuleScan]) -> tuple[str, ...]:
    """Return static `module::qualname` observation targets for execution-plan ranking.

    Args:
        scans: AST-derived module scans to inspect without importing target code.

    Returns:
        tuple[str, ...]: Deterministically ordered symbols owning scheduler spawn sites.
    """
    targets: set[str] = set()
    for scan in scans:
        symbol_ids = {symbol.id.qualname: symbol.id for symbol in scan.symbols}
        for site in _candidate_sites(scan):
            targets.add(site.owner.id.stable_id)
            for spawn in site.spawns:
                name = _resolved_callee_name(site.owner, spawn.callee_name)
                if name is not None and name in symbol_ids:
                    targets.add(symbol_ids[name].stable_id)
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

    plans = tuple(_plan_for(ranked_site) for ranked_site in selected)
    return (*plans, *tuple(sorted(rejected, key=lambda rejection: rejection.id)))


def _candidate_sites(scan: ModuleScan) -> tuple[_CandidateSite, ...]:
    source = scan.module.path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(scan.module.path), type_comments=True)
    symbol_map = {symbol.id.qualname: symbol for symbol in scan.symbols}
    function_nodes = _function_nodes(tree)
    sites: list[_CandidateSite] = []
    for symbol in scan.symbols:
        node = function_nodes.get(symbol.id.qualname)
        if node is None:
            continue
        spawns = _spawn_calls(node)
        if not spawns:
            continue
        sites.append(_site_from_spawns(symbol, spawns, symbol_map, node))
    return tuple(sites)


def _site_from_spawns(
    owner: SymbolRecord,
    spawns: tuple[SpawnCall, ...],
    symbol_map: Mapping[str, SymbolRecord],
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> _CandidateSite:
    dialect_names = {spawn.dialect for spawn in spawns}
    if len(dialect_names) != 1:
        return _rejected_site(owner, spawns, "ambiguous-spawn", "multiple scheduler dialects")
    dialect = _dialect_by_name(spawns[0].dialect)
    callee_symbols = _resolved_callees(owner, spawns, symbol_map)
    if callee_symbols is None:
        return _rejected_site(owner, spawns, "ambiguous-spawn", "unresolved spawned callee")
    task_groups = _structured_task_groups(node)
    if any(not _spawn_in_structured_scope(spawn, task_groups) for spawn in spawns):
        return _rejected_site(
            owner,
            spawns,
            "unstructured-task",
            "spawned work is not owned by a recognized task-group scope",
        )
    if _task_handle_escapes(node):
        return _rejected_site(
            owner,
            spawns,
            "escaping-handle",
            "a spawned task handle escapes the orchestration callable",
        )
    return _transport_site(owner, spawns, dialect, callee_symbols, node)


def _transport_site(
    owner: SymbolRecord,
    spawns: tuple[SpawnCall, ...],
    dialect: SchedulerDialect,
    callee_symbols: tuple[SymbolRecord, ...],
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> _CandidateSite:
    transport = _private_transport(node)
    if transport is None:
        return _rejected_site(
            owner,
            spawns,
            "unknown-transport",
            "no private queue or memory-stream transport evidence was found",
        )
    users = tuple(
        (symbol, spawn)
        for symbol, spawn in zip(callee_symbols, spawns, strict=True)
        if any(endpoint in spawn.transport_arguments for endpoint in transport.endpoints)
    )
    preflight_rejection = _transport_preflight_rejection(node, transport, users)
    if preflight_rejection is not None:
        return _rejected_site(owner, spawns, *preflight_rejection)
    classified_users = tuple(
        _TransportUser(
            symbol=symbol,
            spawn=spawn,
            role=_transport_role(symbol, spawn, transport.endpoints),
        )
        for symbol, spawn in users
    )
    spawned_consumers = tuple(user.symbol for user in classified_users if user.role == "consumer")
    owner_consumes = _symbol_uses_local_consumer(owner, transport.endpoints)
    consumers = (
        *spawned_consumers,
        *((owner,) if owner_consumes else ()),
    )
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
        rejection_reason=None,
        rejection_message=None,
    )


def _transport_preflight_rejection(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    transport: _TransportEvidence,
    users: tuple[tuple[SymbolRecord, SpawnCall], ...],
) -> tuple[ExecutionPlanRejectionReason, str] | None:
    if _transport_escapes(node, transport.endpoints):
        return "public-transport", "result transport escapes the orchestration callable"
    if transport.capacity <= 0:
        return (
            "unknown-capacity",
            "result transport must have a statically known positive capacity",
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
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> _TransportEvidence | None:
    for child in _walk_function_scope(node):
        if isinstance(child, ast.Assign):
            transport = _transport_from_assign(child)
            if transport is not None:
                return transport
        if isinstance(child, ast.AnnAssign):
            transport = _transport_from_ann_assign(child)
            if transport is not None:
                return transport
    return None


def _transport_from_assign(node: ast.Assign) -> _TransportEvidence | None:
    call_path = _call_path(node.value)
    if call_path is None:
        return None
    if call_path in (("asyncio", "Queue"), ("Queue",)):
        for target in node.targets:
            if isinstance(target, ast.Name):
                capacity = _literal_capacity(node.value, default=0)
                if capacity is not None:
                    return _TransportEvidence(target.id, (target.id,), capacity)
    if call_path in (("anyio", "create_memory_object_stream"), ("create_memory_object_stream",)):
        for target in node.targets:
            endpoints = _tuple_names(target)
            capacity = _literal_capacity(node.value, default=None)
            if len(endpoints) == _MEMORY_STREAM_ENDPOINTS and capacity is not None:
                return _TransportEvidence("|".join(endpoints), endpoints, capacity)
    return None


def _transport_from_ann_assign(node: ast.AnnAssign) -> _TransportEvidence | None:
    call_path = _call_path(node.value)
    if call_path in (("asyncio", "Queue"), ("Queue",)) and isinstance(node.target, ast.Name):
        capacity = _literal_capacity(node.value, default=0)
        if capacity is not None:
            return _TransportEvidence(node.target.id, (node.target.id,), capacity)
    return None


def _tuple_names(node: ast.expr) -> tuple[str, ...]:
    if isinstance(node, ast.Tuple):
        return tuple(item.id for item in node.elts if isinstance(item, ast.Name))
    return ()


def _transport_role(
    symbol: SymbolRecord,
    spawn: SpawnCall,
    endpoints: tuple[str, ...],
) -> str:
    parameters = tuple(
        parameter for parameter in symbol.parameters if parameter.name not in {"self", "cls"}
    )
    roles: set[str] = set()
    for argument_index, argument_name in enumerate(spawn.transport_arguments):
        if argument_name not in endpoints or argument_index >= len(parameters):
            continue
        parameter_name = parameters[argument_index].name
        calls = {call.target for call in symbol.call_sites}
        consumer_api: set[str] = {
            f"{parameter_name}.get",
            f"{parameter_name}.get_nowait",
            f"{parameter_name}.receive",
            f"{parameter_name}.receive_nowait",
        }
        producer_api: set[str] = {
            f"{parameter_name}.put",
            f"{parameter_name}.put_nowait",
            f"{parameter_name}.send",
            f"{parameter_name}.send_nowait",
        }
        if len(endpoints) == _MEMORY_STREAM_ENDPOINTS and argument_name == endpoints[0]:
            consumer_api = set()
            producer_api = {
                f"{parameter_name}.send",
                f"{parameter_name}.send_nowait",
            }
        elif len(endpoints) == _MEMORY_STREAM_ENDPOINTS and argument_name == endpoints[1]:
            producer_api = set()
            consumer_api = {
                f"{parameter_name}.receive",
                f"{parameter_name}.receive_nowait",
            }
        if calls & consumer_api:
            roles.add("consumer")
        if calls & producer_api:
            roles.add("producer")
    if len(roles) == 1:
        return next(iter(roles))
    return "unknown"


def _symbol_uses_local_consumer(
    symbol: SymbolRecord,
    endpoints: tuple[str, ...],
) -> bool:
    consumer_targets = {
        target
        for endpoint in endpoints
        for target in (
            f"{endpoint}.get",
            f"{endpoint}.get_nowait",
            f"{endpoint}.receive",
            f"{endpoint}.receive_nowait",
        )
    }
    return any(call.target in consumer_targets for call in symbol.call_sites)


def _literal_capacity(node: ast.expr | None, *, default: int | None) -> int | None:
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
    if isinstance(capacity_node, ast.Constant) and isinstance(capacity_node.value, int):
        return capacity_node.value
    return None


def _transport_escapes(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    endpoints: tuple[str, ...],
) -> bool:
    endpoint_names = frozenset(endpoints)
    for child in _walk_function_scope(node):
        if isinstance(child, ast.Return) and _expression_exposes_names(child.value, endpoint_names):
            return True
        if (
            isinstance(child, ast.Assign)
            and _expression_reads_names(child.value, endpoint_names)
            and any(not isinstance(target, ast.Name | ast.Tuple) for target in child.targets)
        ):
            return True
        if (
            isinstance(child, ast.AnnAssign)
            and _expression_reads_names(child.value, endpoint_names)
            and not isinstance(child.target, ast.Name)
        ):
            return True
        if isinstance(child, ast.Global | ast.Nonlocal) and endpoint_names.intersection(
            child.names
        ):
            return True
    return False


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


def _expression_reads_names(node: ast.expr | None, names: frozenset[str]) -> bool:
    if node is None:
        return False
    return any(
        isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load) and child.id in names
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


def _structured_task_groups(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, tuple[_TaskGroupScope, ...]]:
    groups: dict[str, list[_TaskGroupScope]] = {}
    for child in _walk_function_scope(node):
        if not isinstance(child, ast.AsyncWith):
            continue
        for item in child.items:
            if not isinstance(item.optional_vars, ast.Name):
                continue
            path = _call_path(item.context_expr)
            if path == ("asyncio", "TaskGroup"):
                dialect = "asyncio"
            elif path == ("anyio", "create_task_group"):
                dialect = "anyio-on-asyncio"
            else:
                continue
            groups.setdefault(item.optional_vars.id, []).append(
                _TaskGroupScope(
                    dialect=dialect,
                    lineno=child.lineno,
                    end_lineno=getattr(child, "end_lineno", child.lineno),
                )
            )
    return {name: tuple(scopes) for name, scopes in groups.items()}


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
                if site.consumer is not None and site.consumer != site.owner.id
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


def _plan_for(ranked_site: _RankedSite) -> ExecutionPlan:
    scan = ranked_site.scan
    site = ranked_site.site
    owner_id = site.owner.id
    consumer = site.consumer if site.consumer is not None else owner_id
    consumer_is_owner = consumer == owner_id
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
    edges = (
        *spawn_edges,
        *(
            ()
            if consumer_is_owner
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
    )
    source_hash = _source_hash(scan, site)
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
        reducer=consumer,
        transport_capacity=site.transport.capacity if site.transport is not None else None,
        ordering_policy="completion-order",
        task_ownership="structured",
        observed_invocations=ranked_site.observed_invocations,
        lifecycle_starts=ranked_site.lifecycle_starts,
        lifecycle_share=ranked_site.start_share,
        guarded_callable_identities=ranked_site.guarded_callable_identities,
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


def _source_hash(scan: ModuleScan, site: _CandidateSite) -> str:
    lines = scan.module.path.read_text(encoding="utf-8").splitlines()
    member_ids = {site.owner.id, *site.producers}
    if site.consumer is not None:
        member_ids.add(site.consumer)
    digest = hashlib.sha256()
    for symbol in sorted(
        (record for record in scan.symbols if record.id in member_ids),
        key=lambda record: record.id.qualname,
    ):
        start = symbol.declaration_start_lineno or symbol.lineno
        source = "\n".join(lines[start - 1 : symbol.end_lineno])
        digest.update(symbol.id.stable_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


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


def _dialect_by_name(name: str) -> SchedulerDialect:
    for dialect in built_in_scheduler_dialects():
        if dialect.name == name:
            return dialect
    return built_in_scheduler_dialects()[0]


def _call_path(node: ast.expr | None) -> tuple[str, ...] | None:
    if not isinstance(node, ast.Call):
        return None
    return _attribute_path(node.func)


def _attribute_path(node: ast.expr) -> tuple[str, ...] | None:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        parent = _attribute_path(node.value)
        if parent is None:
            return None
        return (*parent, node.attr)
    return None


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
