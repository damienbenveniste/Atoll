"""Conservative report-only task fusion planning.

This module detects task-spawn sites that may be candidates for a future
lowering pass, but it never rewrites source or executes target project code.
Plans are intentionally conservative: dynamic profile evidence must prove a
single non-overlapping coroutine execution shape, and static source inspection
must not find cancellation, instrumentation, context-local, scheduling, or
dynamic-dispatch hazards in the spawned coroutine closure.
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from atoll.models import DependencyEdge, ModuleScan, SymbolId
from atoll.runtime.profiling import ProfiledMember, ProfileResult

_MIN_OBSERVED_CALLS = 20
_SPAWN_API_SUFFIXES = ("start_soon", "create_task", "ensure_future")
_STATIC_HAZARD_ORDER = (
    "static_cancellation",
    "static_instrumentation",
    "static_contextvars",
    "static_extra_concurrency",
    "static_dynamic_effects",
)


class _Hasher(Protocol):
    def update(self, data: bytes, /) -> object:
        """Add bytes to a hash object.

        Args:
            data: Bytes to mix into the digest.

        Returns:
            object: Hash implementations may return `None` or themselves.
        """
        ...


@dataclass(frozen=True, slots=True)
class TaskSpawnSite:
    """One AST-recognized task spawn inside a scanned module.

    Attributes:
        module: Importable module containing the spawn expression.
        caller: Module-local qualified name that owns the spawn expression.
        callee: Resolved same-module coroutine candidate, when resolution is unique.
        spawn_api: Source-level call target used to schedule the task.
        lineno: One-based first source line of the spawn call.
        end_lineno: One-based final source line of the spawn call.
        col_offset: Zero-based first source column of the spawn call.
        end_col_offset: Zero-based final source column of the spawn call.
        source_text: Exact source segment for the recognized spawn expression.
    """

    module: str
    caller: str
    callee: str | None
    spawn_api: str
    lineno: int
    end_lineno: int
    col_offset: int
    end_col_offset: int | None
    source_text: str

    @property
    def canonical_callee(self) -> str | None:
        """Return the report-facing callee identity when resolution succeeded.

        Returns:
            str | None: `module::qualname` callee identity, or `None` when the
                spawn target could not be resolved to exactly one same-module symbol.
        """
        if self.callee is None:
            return None
        return f"{self.module}::{self.callee}"


@dataclass(frozen=True, slots=True)
class FusionGateRejection:
    """Machine-readable reason a task-fusion plan is report-only rejected.

    Attributes:
        code: Stable rejection code for downstream reports.
        reason: Human-readable explanation suitable for surfacing directly.
    """

    code: str
    reason: str


@dataclass(frozen=True, slots=True)
class FusionPlan:
    """Conservative report-only task-fusion candidate and gate result.

    The `id` and `source_hash` are derived only from source identity and source
    content. Runtime counts decide eligibility but deliberately do not affect
    stable identifiers so reports can compare the same source site across
    profiling runs.

    Attributes:
        id: Stable content-derived plan identifier.
        source_hash: Hash of the spawn source and spawned coroutine closure.
        root: Hot selected symbol whose same-module dependency closure reached this site.
        caller: Symbol containing the recognized task spawn.
        callee: Resolved same-module coroutine candidate, when unique.
        spawn_api: Source-level scheduling API expression.
        lineno: One-based first source line of the spawn call.
        end_lineno: One-based final source line of the spawn call.
        col_offset: Zero-based first source column of the spawn call.
        end_col_offset: Zero-based final source column of the spawn call.
        eligible: Whether all static and dynamic gates passed.
        observed_calls: Targeted callee invocations represented by the dynamic evidence.
        completed_calls: Observed callee invocations reaching return or unwind.
        max_active_calls: Maximum overlapping active callee invocations.
        pre_completion_suspensions: Yield events observed while a callee invocation was active.
        observed_signatures: Number of retained canonical argument type signatures.
        observation_capped: Whether targeted type observation reached its bounded call budget.
        rejections: Ordered conservative rejection reasons.
        spawn_source: Exact unreported spawn source used to reject stale staged payloads.
    """

    id: str
    source_hash: str
    root: str
    caller: str
    callee: str | None
    spawn_api: str
    lineno: int
    end_lineno: int
    col_offset: int
    end_col_offset: int | None
    eligible: bool
    observed_calls: int
    completed_calls: int
    max_active_calls: int
    pre_completion_suspensions: int
    observed_signatures: int
    observation_capped: bool
    rejections: tuple[FusionGateRejection, ...]
    spawn_source: str = ""


@dataclass(frozen=True, slots=True)
class _ModuleAstIndex:
    scan: ModuleScan
    source: str
    symbols: dict[str, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef]
    symbol_sources: dict[str, str]
    local_calls: dict[str, tuple[str, ...]]
    spawn_sites: tuple[TaskSpawnSite, ...]
    hazards: dict[str, frozenset[str]]


def fusion_observation_targets(scans: tuple[ModuleScan, ...]) -> tuple[str, ...]:
    """Return resolved task-spawn callees that profile collection should observe.

    The discovery pass parses scanned source files without importing them and
    returns deterministic `module::qualname` identities for same-module callees
    scheduled through recognized task-spawn APIs.

    Args:
        scans: Module scans whose source files should be inspected.

    Returns:
        tuple[str, ...]: Canonical callee identities in deterministic order.
    """
    targets: set[str] = set()
    for scan in scans:
        index = _module_index(scan)
        for spawn_site in index.spawn_sites:
            canonical = spawn_site.canonical_callee
            if canonical is not None:
                targets.add(canonical)
    return tuple(sorted(targets))


def build_fusion_plans(
    scans: tuple[ModuleScan, ...], profile: ProfileResult
) -> tuple[FusionPlan, ...]:
    """Build conservative report-only task-fusion plans for selected hot roots.

    Plans are rooted only in `profile.selected_symbols`. For each selected root,
    the planner walks same-module static dependency edges and direct AST calls to
    find recognized task-spawn sites. Every resulting plan is gated by dynamic
    monomorphism/lifecycle evidence and static hazards over the spawned
    coroutine dependency closure.

    Args:
        scans: Module scans whose source files should be inspected.
        profile: Runtime evidence used for gate decisions and selected roots.

    Returns:
        tuple[FusionPlan, ...]: Deterministically ordered report-only plans.
    """
    index_by_module = {scan.module.name: _module_index(scan) for scan in scans}
    members = {(member.module, member.qualname): member for member in profile.members}
    plans: list[FusionPlan] = []
    seen: set[tuple[str, str, int, int, str | None]] = set()
    for root in profile.selected_symbols:
        index = index_by_module.get(root.module)
        if index is None:
            continue
        reachable = _reachable_symbols(index, root)
        for spawn_site in index.spawn_sites:
            if spawn_site.caller not in reachable:
                continue
            key = (
                root.stable_id,
                spawn_site.caller,
                spawn_site.lineno,
                spawn_site.col_offset,
                spawn_site.callee,
            )
            if key in seen:
                continue
            seen.add(key)
            member = (
                members.get((spawn_site.module, spawn_site.callee))
                if spawn_site.callee is not None
                else None
            )
            rejections = _gate_rejections(index, spawn_site, member)
            source_hash = _source_hash(index, spawn_site)
            plan_id = _plan_id(root, spawn_site, source_hash)
            plans.append(
                FusionPlan(
                    id=plan_id,
                    source_hash=source_hash,
                    root=root.stable_id,
                    caller=f"{spawn_site.module}::{spawn_site.caller}",
                    callee=spawn_site.canonical_callee,
                    spawn_api=spawn_site.spawn_api,
                    lineno=spawn_site.lineno,
                    end_lineno=spawn_site.end_lineno,
                    col_offset=spawn_site.col_offset,
                    end_col_offset=spawn_site.end_col_offset,
                    eligible=not rejections,
                    observed_calls=member.call_count if member is not None else 0,
                    completed_calls=member.completed_calls if member is not None else 0,
                    max_active_calls=member.max_active_calls if member is not None else 0,
                    pre_completion_suspensions=(
                        member.pre_completion_suspensions if member is not None else 0
                    ),
                    observed_signatures=len(member.signatures) if member is not None else 0,
                    observation_capped=(member.observation_capped if member is not None else False),
                    rejections=rejections,
                    spawn_source=spawn_site.source_text,
                )
            )
    return tuple(
        sorted(
            plans,
            key=lambda plan: (
                plan.root,
                plan.caller,
                plan.lineno,
                plan.col_offset,
                plan.callee or "",
            ),
        )
    )


def _module_index(scan: ModuleScan) -> _ModuleAstIndex:
    source = scan.module.path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(scan.module.path), type_comments=True)
    symbols: dict[str, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef] = {}
    symbol_sources: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            symbols[node.name] = node
            symbol_sources[node.name] = ast.get_source_segment(source, node) or ""
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                    qualname = f"{node.name}.{child.name}"
                    symbols[qualname] = child
                    symbol_sources[qualname] = ast.get_source_segment(source, child) or ""
    local_calls = {
        qualname: _direct_same_module_calls(qualname, node, symbols)
        for qualname, node in symbols.items()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    hazards = {
        qualname: _symbol_hazards(qualname, node, source, symbols)
        for qualname, node in symbols.items()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    spawn_sites = tuple(
        site
        for qualname, node in symbols.items()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        for site in _spawn_sites(scan.module.name, qualname, node, symbols, source)
    )
    return _ModuleAstIndex(
        scan=scan,
        source=source,
        symbols=symbols,
        symbol_sources=symbol_sources,
        local_calls=local_calls,
        spawn_sites=spawn_sites,
        hazards=hazards,
    )


def _spawn_sites(
    module_name: str,
    caller: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    symbols: Mapping[str, ast.AST],
    source: str,
) -> tuple[TaskSpawnSite, ...]:
    sites: list[TaskSpawnSite] = []
    for child in _runtime_nodes(node):
        if not isinstance(child, ast.Call):
            continue
        spawn_api = _call_path(child.func)
        if not _is_spawn_api(spawn_api):
            continue
        callee = _spawn_callee(child, caller, symbols)
        sites.append(
            TaskSpawnSite(
                module=module_name,
                caller=caller,
                callee=callee,
                spawn_api=spawn_api,
                lineno=child.lineno,
                end_lineno=child.end_lineno or child.lineno,
                col_offset=child.col_offset,
                end_col_offset=child.end_col_offset,
                source_text=ast.get_source_segment(source, child) or "",
            )
        )
    return tuple(sorted(sites, key=lambda site: (site.lineno, site.col_offset)))


def _spawn_callee(
    call: ast.Call,
    caller: str,
    symbols: Mapping[str, ast.AST],
) -> str | None:
    if not call.args:
        return None
    spawn_api = _call_path(call.func)
    candidate = call.args[0]
    if spawn_api.endswith("start_soon"):
        return _resolve_callable_expr(candidate, caller, symbols)
    if isinstance(candidate, ast.Call):
        return _resolve_callable_expr(candidate.func, caller, symbols)
    return _resolve_callable_expr(candidate, caller, symbols)


def _resolve_callable_expr(
    expr: ast.AST, caller: str, symbols: Mapping[str, ast.AST]
) -> str | None:
    path = _call_path(expr)
    if path == "":
        return None
    owner = caller.rsplit(".", maxsplit=1)[0] if "." in caller else None
    if path.startswith(("self.", "cls.")) and owner is not None:
        method = f"{owner}.{path.split('.', maxsplit=1)[1]}"
        return method if method in symbols else None
    if path in symbols:
        return path
    if "." not in path:
        matches = tuple(name for name in symbols if name.rsplit(".", maxsplit=1)[-1] == path)
        if len(matches) == 1:
            return matches[0]
    return None


def _direct_same_module_calls(
    caller: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    symbols: Mapping[str, ast.AST],
) -> tuple[str, ...]:
    calls: set[str] = set()
    for child in _runtime_nodes(node):
        if isinstance(child, ast.Call):
            resolved = _resolve_callable_expr(child.func, caller, symbols)
            if resolved is not None and resolved != caller:
                calls.add(resolved)
    return tuple(sorted(calls))


def _reachable_symbols(index: _ModuleAstIndex, root: SymbolId) -> frozenset[str]:
    graph: dict[str, set[str]] = {
        qualname: set(calls) for qualname, calls in index.local_calls.items()
    }
    for edge in index.scan.dependency_edges:
        if _is_same_module_call(edge, root.module) and isinstance(edge.dst, SymbolId):
            graph.setdefault(edge.src.qualname, set()).add(edge.dst.qualname)
    reachable: set[str] = set()
    pending = [root.qualname]
    while pending:
        current = pending.pop()
        if current in reachable:
            continue
        reachable.add(current)
        pending.extend(sorted(graph.get(current, ()), reverse=True))
    return frozenset(reachable)


def _is_same_module_call(edge: DependencyEdge, module_name: str) -> bool:
    return (
        edge.src.module == module_name
        and isinstance(edge.dst, SymbolId)
        and edge.dst.module == module_name
        and edge.kind in {"calls", "calls_method"}
    )


def _gate_rejections(
    index: _ModuleAstIndex,
    spawn_site: TaskSpawnSite,
    member: ProfiledMember | None,
) -> tuple[FusionGateRejection, ...]:
    rejections: list[FusionGateRejection] = []
    callee_node = index.symbols.get(spawn_site.callee or "")
    if spawn_site.callee is None or callee_node is None:
        rejections.append(
            FusionGateRejection(
                code="callee_unresolved",
                reason="spawn target did not resolve to exactly one same-module callable",
            )
        )
        return tuple(rejections)
    if not isinstance(callee_node, ast.AsyncFunctionDef):
        rejections.append(
            FusionGateRejection(
                code="callee_not_coroutine",
                reason="resolved spawn target is not an async function",
            )
        )
    if member is None:
        rejections.append(
            FusionGateRejection(
                code="missing_profile_evidence",
                reason="resolved coroutine has no dynamic profile evidence",
            )
        )
    else:
        rejections.extend(_profile_rejections(member))
    rejections.extend(_static_rejections(index, spawn_site))
    return tuple(rejections)


def _profile_rejections(member: ProfiledMember) -> tuple[FusionGateRejection, ...]:
    rejections: list[FusionGateRejection] = []
    if member.call_count < _MIN_OBSERVED_CALLS:
        rejections.append(
            FusionGateRejection(
                code="insufficient_observed_calls",
                reason="callee has fewer than 20 observed calls",
            )
        )
    signature_count = sum(signature.count for signature in member.signatures)
    if len(member.signatures) != 1 or signature_count != member.call_count:
        rejections.append(
            FusionGateRejection(
                code="non_monomorphic_signature",
                reason="callee does not have exactly one observed signature covering every call",
            )
        )
    if member.polymorphic_overflow or member.observation_capped:
        rejections.append(
            FusionGateRejection(
                code="polymorphic_evidence_capped",
                reason="callee profile evidence overflowed or was capped",
            )
        )
    completed_calls = _int_member_attr(member, "completed_calls", member.call_count)
    if completed_calls != member.call_count:
        rejections.append(
            FusionGateRejection(
                code="incomplete_calls",
                reason="not every observed callee invocation completed normally",
            )
        )
    max_active_calls = _int_member_attr(member, "max_active_calls", 1)
    if max_active_calls != 1:
        rejections.append(
            FusionGateRejection(
                code="overlapping_calls",
                reason="callee had overlapping active invocations",
            )
        )
    pre_completion_suspensions = _int_member_attr(member, "pre_completion_suspensions", 0)
    if pre_completion_suspensions != 0:
        rejections.append(
            FusionGateRejection(
                code="pre_completion_suspension",
                reason="callee suspended before completing an observed invocation",
            )
        )
    if (
        member.lifecycle.yield_ != 0
        or member.lifecycle.resume != 0
        or member.lifecycle.unwind != 0
        or member.lifecycle.throw != 0
    ):
        rejections.append(
            FusionGateRejection(
                code="lifecycle_suspension",
                reason="callee lifecycle evidence includes yield, resume, unwind, or throw events",
            )
        )
    return tuple(rejections)


def _int_member_attr(member: ProfiledMember, name: str, default: int) -> int:
    value = getattr(member, name, default)
    return value if isinstance(value, int) else default


def _static_rejections(
    index: _ModuleAstIndex, spawn_site: TaskSpawnSite
) -> tuple[FusionGateRejection, ...]:
    closure = _callee_closure(index, spawn_site.callee)
    hazards = _caller_hazards(index, spawn_site)
    for qualname in closure:
        hazards.update(index.hazards.get(qualname, frozenset()))
    return tuple(
        FusionGateRejection(code=code, reason=_static_reason(code))
        for code in _STATIC_HAZARD_ORDER
        if code in hazards
    )


def _caller_hazards(index: _ModuleAstIndex, spawn_site: TaskSpawnSite) -> set[str]:
    node = index.symbols.get(spawn_site.caller)
    if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return {"static_dynamic_effects"}
    hazards = set(
        _symbol_hazards(
            spawn_site.caller,
            node,
            index.source,
            index.symbols,
            ignored_spawn=spawn_site,
        )
    )
    runtime_nodes = _runtime_nodes(node)
    if any(_spawn_result_escapes(child, spawn_site) for child in runtime_nodes):
        hazards.add("static_dynamic_effects")
    if any(
        isinstance(child, ast.Await | ast.Yield | ast.YieldFrom | ast.AsyncFor | ast.AsyncWith)
        and not _inside_spawn(child, spawn_site)
        for child in runtime_nodes
    ):
        hazards.add("static_dynamic_effects")
    return hazards


def _callee_closure(index: _ModuleAstIndex, callee: str | None) -> frozenset[str]:
    if callee is None:
        return frozenset()
    reachable: set[str] = set()
    pending = [callee]
    while pending:
        current = pending.pop()
        if current in reachable:
            continue
        reachable.add(current)
        pending.extend(reversed(index.local_calls.get(current, ())))
    return frozenset(reachable)


def _symbol_hazards(
    qualname: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    source: str,
    symbols: Mapping[str, ast.AST],
    *,
    ignored_spawn: TaskSpawnSite | None = None,
) -> frozenset[str]:
    hazards: set[str] = set()
    for child in _runtime_nodes(node):
        if ignored_spawn is not None and _inside_spawn(child, ignored_spawn):
            continue
        hazards.update(_runtime_node_hazards(child, qualname, symbols))
    source_segment = ast.get_source_segment(source, node) or ""
    if "exec(" in source_segment:
        hazards.add("static_dynamic_effects")
    return frozenset(hazards)


def _runtime_node_hazards(
    node: ast.AST,
    qualname: str,
    symbols: Mapping[str, ast.AST],
) -> set[str]:
    hazards: set[str] = set()
    if isinstance(node, ast.Global | ast.Nonlocal):
        hazards.add("static_dynamic_effects")
    if isinstance(node, ast.Import | ast.ImportFrom):
        hazards.update(_import_hazards(node))
    if isinstance(node, ast.Name):
        hazards.update(_name_hazards(node.id))
    if isinstance(node, ast.Attribute):
        hazards.update(_path_hazards(_call_path(node)))
        if isinstance(node.ctx, ast.Store | ast.Del):
            hazards.add("static_dynamic_effects")
    if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store | ast.Del):
        hazards.add("static_dynamic_effects")
    if isinstance(node, ast.Call):
        target = _call_path(node.func)
        hazards.update(_path_hazards(target))
        if _is_spawn_api(target):
            hazards.add("static_extra_concurrency")
        elif _is_unresolved_dynamic_call(target, qualname, symbols):
            hazards.add("static_dynamic_effects")
    return hazards


def _spawn_result_escapes(node: ast.AST, spawn_site: TaskSpawnSite) -> bool:
    if not isinstance(node, ast.Assign | ast.AnnAssign | ast.NamedExpr | ast.Return | ast.Await):
        return False
    return any(
        isinstance(child, ast.Call) and _same_source_location(child, spawn_site)
        for child in ast.walk(node)
    )


def _inside_spawn(node: ast.AST, spawn_site: TaskSpawnSite) -> bool:
    lineno = getattr(node, "lineno", None)
    end_lineno = getattr(node, "end_lineno", None)
    col_offset = getattr(node, "col_offset", None)
    end_col_offset = getattr(node, "end_col_offset", None)
    if (
        not isinstance(lineno, int)
        or not isinstance(end_lineno, int)
        or not isinstance(col_offset, int)
    ):
        return False
    if lineno < spawn_site.lineno or end_lineno > spawn_site.end_lineno:
        return False
    if lineno == spawn_site.lineno and col_offset < spawn_site.col_offset:
        return False
    return not (
        end_lineno == spawn_site.end_lineno
        and spawn_site.end_col_offset is not None
        and isinstance(end_col_offset, int)
        and end_col_offset > spawn_site.end_col_offset
    )


def _same_source_location(node: ast.AST, spawn_site: TaskSpawnSite) -> bool:
    return (
        getattr(node, "lineno", None) == spawn_site.lineno
        and getattr(node, "end_lineno", None) == spawn_site.end_lineno
        and getattr(node, "col_offset", None) == spawn_site.col_offset
        and getattr(node, "end_col_offset", None) == spawn_site.end_col_offset
    )


def _import_hazards(node: ast.Import | ast.ImportFrom) -> set[str]:
    names: set[str] = set()
    if isinstance(node, ast.Import):
        names.update(alias.name for alias in node.names)
    else:
        if node.module is not None:
            names.add(node.module)
        names.update(alias.name for alias in node.names)
    hazards: set[str] = set()
    for name in names:
        hazards.update(_name_hazards(name))
        hazards.update(_path_hazards(name))
    return hazards


def _name_hazards(name: str) -> set[str]:
    lowered = name.lower()
    hazards: set[str] = set()
    if name == "CancelScope" or "cancel_scope" in lowered or lowered in {"cancel", "cancelled"}:
        hazards.add("static_cancellation")
    if "instrument" in lowered or "span" in lowered or lowered in {"trace", "tracer"}:
        hazards.add("static_instrumentation")
    if name in {"ContextVar", "Token"} or lowered in {"contextvars", "copy_context"}:
        hazards.add("static_contextvars")
    if lowered in {
        "getattr",
        "setattr",
        "delattr",
        "eval",
        "globals",
        "locals",
        "vars",
        "__import__",
        "send",
        "asend",
        "throw",
    }:
        hazards.add("static_dynamic_effects")
    return hazards


def _path_hazards(path: str) -> set[str]:
    tail = path.rsplit(".", maxsplit=1)[-1]
    hazards = _name_hazards(tail)
    if path.startswith("contextvars.") or ".contextvars." in path:
        hazards.add("static_contextvars")
    if path.endswith((".start_soon", ".create_task", ".ensure_future")):
        hazards.add("static_extra_concurrency")
    if path.endswith((".send", ".asend", ".throw")):
        hazards.add("static_dynamic_effects")
    return hazards


def _static_reason(code: str) -> str:
    reasons = {
        "static_cancellation": "callee closure references cancellation control",
        "static_instrumentation": "callee closure references instrumentation hooks",
        "static_contextvars": "callee closure references context-local state",
        "static_extra_concurrency": "callee closure schedules additional concurrent work",
        "static_dynamic_effects": "callee closure uses unresolved dynamic effects",
    }
    return reasons[code]


def _source_hash(index: _ModuleAstIndex, spawn_site: TaskSpawnSite) -> str:
    hasher = hashlib.sha256()
    _hash_part(hasher, spawn_site.source_text)
    for qualname in sorted(_callee_closure(index, spawn_site.callee)):
        _hash_part(hasher, qualname)
        _hash_part(hasher, index.symbol_sources.get(qualname, ""))
    return hasher.hexdigest()


def _plan_id(root: SymbolId, spawn_site: TaskSpawnSite, source_hash: str) -> str:
    hasher = hashlib.sha256()
    for value in (
        root.stable_id,
        spawn_site.module,
        spawn_site.caller,
        spawn_site.callee or "<unresolved>",
        spawn_site.spawn_api,
        str(spawn_site.lineno),
        str(spawn_site.col_offset),
        source_hash,
    ):
        _hash_part(hasher, value)
    return f"task-fusion:{hasher.hexdigest()[:16]}"


def _hash_part(hasher: _Hasher, value: str) -> None:
    hasher.update(value.encode("utf-8"))
    hasher.update(b"\0")


def _call_path(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        root = _call_path(node.value)
        return f"{root}.{node.attr}" if root else node.attr
    return ""


def _is_spawn_api(path: str) -> bool:
    return path.rsplit(".", maxsplit=1)[-1] in _SPAWN_API_SUFFIXES


def _runtime_nodes(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.AST, ...]:
    """Return callable-body nodes without attributing nested scopes to the caller.

    Args:
        node: Callable declaration whose executable body is being inspected.

    Returns:
        tuple[ast.AST, ...]: Deterministic depth-first body nodes excluding nested scopes.
    """
    discovered: list[ast.AST] = []
    pending: list[ast.AST] = list(reversed(node.body))
    while pending:
        current = pending.pop()
        discovered.append(current)
        if isinstance(current, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
            continue
        pending.extend(reversed(tuple(ast.iter_child_nodes(current))))
    return tuple(discovered)


def _is_unresolved_dynamic_call(
    target: str,
    caller: str,
    symbols: Mapping[str, ast.AST],
) -> bool:
    """Return whether a call target lacks a statically safe local or builtin identity.

    Args:
        target: Source-level call target path.
        caller: Qualified name containing the call.
        symbols: Same-module declarations available for resolution.

    Returns:
        bool: Whether task fusion must treat the call as a dynamic effect boundary.
    """
    if not target:
        return True
    return _resolve_callable_expr(_path_expression(target), caller, symbols) is None


def _path_expression(path: str) -> ast.expr:
    """Parse a dotted path back into an expression for local resolution.

    Args:
        path: Non-empty source-level dotted call target.

    Returns:
        ast.expr: Parsed expression node.
    """
    parsed = ast.parse(path, mode="eval")
    return parsed.body
