"""Callback-backed execution-plan backend for guarded asyncio fan-out lowering.

This backend owns a narrow payload-only lowering that replaces a proven
`TaskGroup.create_task` fan-out with one `loop.call_soon` callback per logical
item when strict runtime guards hold. It is intentionally independent from
native compiler regions: assessment proves a scheduler topology, staging edits
only the copied payload module, and runtime guard failure runs the original
real-task path before any callback is scheduled.
"""

from __future__ import annotations

import ast
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

from atoll.execution_plans.models import (
    ChangedPayloadFile,
    ExecutionPlan,
    ExecutionPlanAssessment,
    ExecutionPlanAssessmentContext,
    ExecutionPlanDiagnostic,
    ExecutionPlanStageContext,
    PlanGuard,
    StagedExecutionPlan,
)
from atoll.execution_plans.task_preserving import (
    source_attribute_path as _attribute_path,
)
from atoll.execution_plans.task_preserving import (
    source_callsite_lineno_is_planned as _callsite_lineno_is_planned,
)
from atoll.execution_plans.task_preserving import (
    source_create_task_scheduler as _create_task_scheduler,
)
from atoll.execution_plans.task_preserving import (
    source_function_node as _function_node,
)
from atoll.execution_plans.task_preserving import (
    source_module_path as _module_path,
)
from atoll.execution_plans.task_preserving import (
    source_name_exists as _name_exists,
)
from atoll.execution_plans.task_preserving import (
    source_name_is_bound as _name_is_bound,
)
from atoll.execution_plans.task_preserving import (
    source_sha256 as _sha256,
)
from atoll.execution_plans.task_preserving import (
    source_spawn_callee as _spawn_callee,
)
from atoll.execution_plans.task_preserving import (
    source_splice_expressions as _splice_expressions,
)
from atoll.execution_plans.task_preserving import (
    source_validate_callsite_fingerprint as _validate_callsite_fingerprint,
)
from atoll.execution_plans.task_preserving import (
    source_validate_hash as _validate_source_hash,
)

_BACKEND_NAME: Final = "callback-backed"
_LOWERING_VERSION: Final = "callback-backed-v1"
_SUPPORTED_DIALECT: Final = "asyncio"
_SUPPORT_VERSION: Final = "support-v1"
_PRODUCER_ARGUMENT_COUNT: Final = 2
_QUALNAME_CLASS_METHOD_PARTS: Final = 2


@dataclass(frozen=True, slots=True)
class CallbackBackedExecutionPlanBackend:
    """Payload-only backend that lowers proven producer fan-out to callbacks.

    The backend accepts only a useful strict subset: an asyncio `TaskGroup`
    fan-out over an exact builtin iterable, a private positive-capacity
    `asyncio.Queue`, a same-module module-level async producer that publishes
    exactly once with `put_nowait`, and an owner receive shaped as
    `await queue.get()`. Staged code falls back to the original real-task path
    unless all runtime scheduler, loop, queue, iterable, and callable identity
    guards pass before callback scheduling starts.

    Attributes:
        name: Stable backend identifier used in reports and fingerprints.
    """

    name: str = _BACKEND_NAME

    def assess(
        self,
        plan: ExecutionPlan,
        context: ExecutionPlanAssessmentContext,
    ) -> ExecutionPlanAssessment:
        """Classify whether the callback-backed backend can lower a plan.

        Args:
            plan: Scheduler-aware execution plan discovered from source.
            context: Read-only project and source-root context for assessment.

        Returns:
            ExecutionPlanAssessment: Deterministic support decision and
            explicit rejection reasons for unsupported shapes.
        """
        reasons = list(_static_rejection_reasons(plan))
        source_path = _module_path(context.source_root, plan.source_module)
        if source_path is None:
            reasons.append("source module is not present below the assessment source root")
        elif not reasons:
            try:
                source_text = source_path.read_text(encoding="utf-8")
                _validated_rewrite(source_text, plan)
            except (TypeError, ValueError) as error:
                reasons.append(str(error))
        unsupported = tuple(node.id for node in plan.nodes) if reasons else ()
        supported = () if reasons else tuple(node.id for node in plan.nodes)
        return ExecutionPlanAssessment(
            plan_id=plan.id,
            backend=self.name,
            status="unsupported" if reasons else "supported",
            supported_nodes=supported,
            unsupported_nodes=unsupported,
            reasons=tuple(reasons),
        )

    def stage(
        self,
        plan: ExecutionPlan,
        context: ExecutionPlanStageContext,
    ) -> StagedExecutionPlan:
        """Stage a guarded callback-backed source rewrite in the payload.

        Args:
            plan: Scheduler-aware execution plan to stage.
            context: Payload and cache roots for this staging attempt.

        Returns:
            StagedExecutionPlan: Changed payload source evidence and guards.

        Raises:
            TypeError: If the fan-out loop or producer uses a statically unsafe
                expression form.
            ValueError: If the payload module is absent, stale, or unsupported.
        """
        reasons = tuple(_static_rejection_reasons(plan))
        if reasons:
            raise ValueError(f"unsupported callback-backed execution plan: {'; '.join(reasons)}")
        payload_path = _module_path(context.payload_root, plan.source_module)
        if payload_path is None:
            raise ValueError(f"payload module is not present: {plan.source_module}")
        source_text = payload_path.read_text(encoding="utf-8")
        before_hash = _sha256(source_text)
        transformed = _validated_rewrite(source_text, plan)
        payload_path.write_text(transformed.source_text, encoding="utf-8")
        after_text = payload_path.read_text(encoding="utf-8")
        after_hash = _sha256(after_text)
        install_path = PurePosixPath(payload_path.relative_to(context.payload_root).as_posix())
        return StagedExecutionPlan(
            plan=plan,
            backend=self.name,
            payload_files=(
                ChangedPayloadFile(
                    install_path=install_path,
                    before_hash=before_hash,
                    after_hash=after_hash,
                    role="source",
                ),
            ),
            required_imports=(),
            guards=(
                *plan.guards,
                PlanGuard(
                    kind="semantics",
                    expression=transformed.guard_expression,
                    message=(
                        "callback scheduling is used only before any callback is scheduled "
                        "and when loop, task, queue, iterable, and callable identity guards hold"
                    ),
                ),
            ),
        )

    def fingerprint(
        self,
        plan: ExecutionPlan,
        context: ExecutionPlanStageContext,
    ) -> str:
        """Return a strict fingerprint for backend semantics and payload source.

        Args:
            plan: Scheduler-aware execution plan being fingerprinted.
            context: Payload root containing the source file that would be staged.

        Returns:
            str: Stable SHA-256 digest covering backend versions, plan identity,
            staged payload source, Python ABI/version, and generated support.

        Raises:
            ValueError: If the payload module is absent.
        """
        payload_path = _module_path(context.payload_root, plan.source_module)
        if payload_path is None:
            raise ValueError(f"payload module is not present: {plan.source_module}")
        support_source = _support_source(_support_names(plan), "<producer>")
        digest = hashlib.sha256()
        for part in (
            self.name,
            _LOWERING_VERSION,
            _SUPPORT_VERSION,
            plan.id,
            plan.source_module,
            plan.source_hash,
            plan.callsite_fingerprint,
            plan.topology_fingerprint,
            plan.dialect,
            plan.lowering_version,
            str(sys.version_info[:3]),
            sys.implementation.cache_tag or "",
            _sha256(payload_path.read_text(encoding="utf-8")),
            _sha256(support_source),
            *(node.id for node in plan.nodes),
            *(f"{edge.src}:{edge.dst}:{edge.kind}:{edge.transport or ''}" for edge in plan.edges),
            *plan.guarded_callable_identities,
        ):
            digest.update(part.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def normalize_diagnostic(
        self,
        error: BaseException,
        *,
        diagnostics: str,
        log_path: Path | None,
    ) -> ExecutionPlanDiagnostic:
        """Convert callback-backed failures into report diagnostics.

        Args:
            error: Exception raised while assessing, staging, or trialing a plan.
            diagnostics: Captured diagnostic text from the failed operation.
            log_path: Optional path to a complete backend log.

        Returns:
            ExecutionPlanDiagnostic: Normalized report diagnostic.
        """
        details = tuple(line.strip() for line in diagnostics.splitlines() if line.strip())
        if log_path is not None:
            details = (*details, f"log: {log_path}")
        return ExecutionPlanDiagnostic(
            code="CALLBACK_BACKED_EXECUTION_PLAN_ERROR",
            severity="error",
            message=str(error) or error.__class__.__name__,
            details=details,
        )


@dataclass(frozen=True, slots=True)
class _Rewrite:
    source_text: str
    guard_expression: str


@dataclass(frozen=True, slots=True)
class _SupportNames:
    original: str
    items: str
    spawn: str
    receive: str
    state: str
    last_mode: str


@dataclass(frozen=True, slots=True)
class _RewriteTarget:
    owner: ast.AsyncFunctionDef
    producer: ast.AsyncFunctionDef
    loop: ast.For
    spawn_statement: ast.Expr
    spawn_call: ast.Call
    receive_call: ast.Call
    scheduler_name: str
    queue_name: str
    iterable_source: str
    producer_name: str
    item_name: str


@dataclass(frozen=True, slots=True)
class _CandidateContext:
    source_text: str
    owner: ast.AsyncFunctionDef
    producer: ast.AsyncFunctionDef
    producer_name: str
    queue_name: str
    plan: ExecutionPlan


def _static_rejection_reasons(plan: ExecutionPlan) -> tuple[str, ...]:
    reasons: list[str] = []
    if plan.dialect != _SUPPORTED_DIALECT:
        reasons.append(f"unsupported scheduler dialect: {plan.dialect}")
    if plan.task_ownership != "structured":
        reasons.append(f"unsupported task ownership: {plan.task_ownership}")
    if plan.transport_capacity is None or plan.transport_capacity <= 0:
        reasons.append("transport capacity must be a statically known positive value")
    if plan.completion_transport is None:
        reasons.append("completion transport must be statically known")
    if len([edge for edge in plan.edges if edge.kind == "spawns"]) != 1:
        reasons.append("callback-backed lowering requires exactly one spawn edge")
    producer_nodes = [node for node in plan.nodes if node.role == "producer" and node.symbol]
    if len(producer_nodes) != 1:
        reasons.append("callback-backed lowering requires exactly one producer")
    elif producer_nodes[0].symbol is not None:
        symbol = producer_nodes[0].symbol
        if symbol.module != plan.source_module or "." in symbol.qualname:
            reasons.append("producer must be a same-module module-level callable")
    return tuple(reasons)


def _validated_rewrite(source_text: str, plan: ExecutionPlan) -> _Rewrite:
    tree = ast.parse(source_text, type_comments=True)
    _validate_source_hash(source_text, tree, plan)
    _validate_callsite_fingerprint(tree, plan)
    target = _rewrite_target(source_text, tree, plan)
    _validate_topology(tree, target, plan)
    names = _support_names(plan)
    for generated_name in (
        names.original,
        names.items,
        names.spawn,
        names.receive,
        names.state,
        names.last_mode,
        "_AtollCallbackFailure",
        "_atoll_callback_drive_once",
        "_atoll_callback_no_monitoring_hooks",
    ):
        if _name_exists(tree, generated_name):
            raise ValueError(f"callback-backed generated name already exists: {generated_name}")
    for builtin_name in (
        "ExceptionGroup",
        "BaseException",
        "RuntimeError",
        "StopIteration",
        "getattr",
        "id",
        "isinstance",
        "len",
        "list",
        "tuple",
        "range",
        "type",
    ):
        if _name_is_bound(tree, target.owner, builtin_name):
            raise ValueError(f"callback-backed guard requires unshadowed builtin {builtin_name}")
    rewritten = _splice_expressions(
        source_text,
        (
            (
                target.loop.iter,
                (
                    f"{names.items}({target.iterable_source}, {target.queue_name}, "
                    f"{target.scheduler_name}.create_task, {target.producer_name}, "
                    f"{names.original})"
                ),
            ),
            (
                target.spawn_call,
                (
                    f"{names.spawn}({target.queue_name}, {target.scheduler_name}.create_task, "
                    f"{target.producer_name}, {target.item_name})"
                ),
            ),
            (target.receive_call, f"{names.receive}({target.queue_name})"),
        ),
    )
    if not rewritten.endswith("\n"):
        rewritten = f"{rewritten}\n"
    support = _support_source(names, target.producer_name)
    return _Rewrite(
        source_text=f"{rewritten}\n{support}",
        guard_expression=(
            "sole current task, default task factory, debug disabled, no trace/profile/"
            "monitoring hooks, no queued loop work, exact asyncio.Queue, exact builtin "
            "iterable, capacity fits, and producer identity unchanged"
        ),
    )


def _rewrite_target(source_text: str, tree: ast.Module, plan: ExecutionPlan) -> _RewriteTarget:
    owner = _function_node(tree, plan.owner.qualname)
    if owner is None:
        raise ValueError(f"plan owner is missing from payload source: {plan.owner.qualname}")
    if not isinstance(owner, ast.AsyncFunctionDef):
        raise TypeError("callback-backed owner must be an async function")
    producer_name = _producer_name(plan)
    producer = _function_node(tree, producer_name)
    if not isinstance(producer, ast.AsyncFunctionDef):
        raise TypeError(f"producer is missing or not async: {producer_name}")
    if producer.decorator_list:
        raise ValueError("producer decorators are not supported")
    if "." in producer_name:
        raise ValueError("methods and dynamic producer callables are not supported")
    _validate_producer_body(producer)
    _validate_single_owner_spawn(owner, plan)
    queue_name = _queue_name(plan)
    candidates = _candidate_rewrite_targets(
        _CandidateContext(
            source_text=source_text,
            owner=owner,
            producer=producer,
            producer_name=producer_name,
            queue_name=queue_name,
            plan=plan,
        )
    )
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one callback-backed fan-out loop, found {len(candidates)}"
        )
    _validate_queue_assignment(
        tree,
        candidates[0].owner,
        queue_name,
        plan.transport_capacity,
    )
    return candidates[0]


def _candidate_rewrite_targets(context: _CandidateContext) -> list[_RewriteTarget]:
    candidates: list[_RewriteTarget] = []
    for loop in (
        node for node in ast.walk(context.owner) if isinstance(node, ast.For | ast.AsyncFor)
    ):
        if isinstance(loop, ast.AsyncFor):
            raise TypeError("async fan-out iteration is not supported")
        if not isinstance(loop.target, ast.Name):
            raise TypeError("fan-out target must be a single local name")
        if not isinstance(loop.iter, ast.Name):
            raise TypeError("fan-out iterable must be a side-effect-free local name")
        for statement in loop.body:
            target = _candidate_for_statement(context, loop, statement)
            if target is not None:
                candidates.append(target)
    return candidates


def _candidate_for_statement(
    context: _CandidateContext,
    loop: ast.For,
    statement: ast.stmt,
) -> _RewriteTarget | None:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return None
    call = statement.value
    scheduler_name = _create_task_scheduler(call)
    callee = _spawn_callee(call)
    if scheduler_name is None or callee is None:
        return None
    if not _callsite_lineno_is_planned(context.plan, call.lineno):
        return None
    if callee != context.producer_name:
        raise ValueError("spawn call must target the planned module-level producer")
    if not isinstance(loop.target, ast.Name):
        raise TypeError("fan-out target must be a single local name")
    _validate_spawn_call(call, context.queue_name, loop.target.id)
    receive = _receive_call(context.owner, context.queue_name)
    _validate_shared_task_group_scope(context.owner, loop, receive, scheduler_name)
    iterable_source = ast.get_source_segment(context.source_text, loop.iter)
    if iterable_source is None or "\n" in iterable_source:
        raise ValueError("fan-out iterable source is unavailable for rewrite")
    return _RewriteTarget(
        owner=context.owner,
        producer=context.producer,
        loop=loop,
        spawn_statement=statement,
        spawn_call=call,
        receive_call=receive,
        scheduler_name=scheduler_name,
        queue_name=context.queue_name,
        iterable_source=iterable_source,
        producer_name=context.producer_name,
        item_name=loop.target.id,
    )


def _validate_topology(tree: ast.Module, target: _RewriteTarget, plan: ExecutionPlan) -> None:
    """Reject staged payloads whose source no longer matches planned topology.

    Args:
        tree: Parsed payload module.
        target: Validated source rewrite target.
        plan: Scheduler-aware plan selected from checkout analysis.

    Raises:
        ValueError: If source symbols, call sites, or queue topology drifted.
    """
    del tree
    if target.queue_name != plan.completion_transport:
        raise ValueError("payload topology transport does not match the selected plan")
    produced = [edge for edge in plan.edges if edge.kind == "produces"]
    delivered = [edge for edge in plan.edges if edge.kind == "delivers"]
    if len(produced) != 1 or produced[0].transport != target.queue_name:
        raise ValueError("payload topology must contain one producer transport edge")
    if len(delivered) != 1 or delivered[0].transport != target.queue_name:
        raise ValueError("payload topology must contain one delivery edge")


def _validate_producer_body(producer: ast.AsyncFunctionDef) -> None:
    put_nowait_calls = 0
    for statement in producer.body:
        put_nowait_calls += _validate_producer_statement(statement, producer)
    if put_nowait_calls != 1:
        raise ValueError("producer must publish exactly once with queue.put_nowait")


def _validate_producer_statement(
    statement: ast.stmt,
    producer: ast.AsyncFunctionDef,
) -> int:
    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
        if not _is_queue_put_nowait(statement.value, producer):
            raise ValueError("producer opaque calls or dynamic dispatch are not supported")
        _validate_publish_argument(statement.value, producer)
        return 1
    if isinstance(statement, ast.Raise):
        raise TypeError("producer exceptions require task-preserving execution")
    if isinstance(statement, ast.Global | ast.Nonlocal):
        raise TypeError("producer global or nonlocal mutation is not supported")
    if any(isinstance(node, ast.Await | ast.Yield | ast.YieldFrom) for node in ast.walk(statement)):
        raise TypeError("producer transitive graph must not suspend")
    if _statement_has_attribute_or_subscript_store(statement):
        raise ValueError("producer attribute or subscript mutation is not supported")
    if _statement_has_mutation(statement):
        raise ValueError("producer local mutation is not supported")
    if _statement_has_forbidden_producer_node(statement):
        raise ValueError(
            "producer proof excludes suspensions, nested definitions, mutation, "
            "and dynamic operations"
        )
    raise ValueError("producer statements must be exact queue.put_nowait publishes")


def _statement_has_forbidden_producer_node(statement: ast.stmt) -> bool:
    forbidden_nodes = (
        ast.Await,
        ast.Yield,
        ast.YieldFrom,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Lambda,
        ast.NamedExpr,
        ast.Attribute,
        ast.Subscript,
        ast.BinOp,
        ast.BoolOp,
        ast.Compare,
        ast.UnaryOp,
        ast.IfExp,
        ast.Call,
    )
    return any(isinstance(node, forbidden_nodes) for node in ast.walk(statement))


def _statement_has_attribute_or_subscript_store(statement: ast.stmt) -> bool:
    store_targets: tuple[ast.expr, ...] = ()
    if isinstance(statement, ast.Assign):
        store_targets = tuple(statement.targets)
    elif isinstance(statement, ast.AnnAssign | ast.AugAssign):
        store_targets = (statement.target,)
    elif isinstance(statement, ast.Delete):
        store_targets = tuple(statement.targets)
    return any(
        isinstance(node, ast.Attribute | ast.Subscript)
        and isinstance(node.ctx, ast.Store | ast.Del)
        for target in store_targets
        for node in ast.walk(target)
    )


def _statement_has_mutation(statement: ast.stmt) -> bool:
    return isinstance(statement, ast.Assign | ast.AnnAssign | ast.AugAssign | ast.Delete)


def _is_queue_put_nowait(node: ast.Call, producer: ast.AsyncFunctionDef) -> bool:
    if not isinstance(node.func, ast.Attribute) or node.func.attr != "put_nowait":
        return False
    return (
        isinstance(node.func.value, ast.Name)
        and bool(producer.args.args)
        and node.func.value.id == producer.args.args[0].arg
    )


def _validate_publish_argument(node: ast.Call, producer: ast.AsyncFunctionDef) -> None:
    if len(node.args) != 1 or node.keywords:
        raise ValueError("producer queue.put_nowait must publish exactly one value")
    value = node.args[0]
    if any(
        isinstance(child, ast.BinOp | ast.BoolOp | ast.Compare | ast.UnaryOp | ast.IfExp)
        for child in ast.walk(value)
    ):
        raise ValueError("producer proof excludes dynamic operations")
    allowed_names = {argument.arg for argument in producer.args.args[1:]}
    if isinstance(value, ast.Name) and value.id in allowed_names:
        return
    if isinstance(value, ast.Constant):
        return
    raise ValueError("producer publish value must be a parameter or constant")


def _validate_spawn_call(call: ast.Call, queue_name: str, item_name: str) -> None:
    if not call.args or not isinstance(call.args[0], ast.Call):
        raise ValueError("create_task argument must be a direct producer coroutine call")
    producer_call = call.args[0]
    if len(producer_call.args) != _PRODUCER_ARGUMENT_COUNT or producer_call.keywords:
        raise ValueError("producer call must pass queue and item positionally")
    first, second = producer_call.args
    if not isinstance(first, ast.Name) or first.id != queue_name:
        raise ValueError("producer call must pass the planned private queue")
    if not isinstance(second, ast.Name) or second.id != item_name:
        raise ValueError("producer call must pass the current fan-out item")


def _receive_call(owner: ast.AsyncFunctionDef, queue_name: str) -> ast.Call:
    calls: list[ast.Call] = []
    for node in ast.walk(owner):
        if not isinstance(node, ast.Await) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "get"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == queue_name
            and not call.args
            and not call.keywords
        ):
            calls.append(call)
    if len(calls) != 1:
        raise ValueError(f"expected exactly one owner await {queue_name}.get(), found {len(calls)}")
    return calls[0]


def _validate_single_owner_spawn(owner: ast.AsyncFunctionDef, plan: ExecutionPlan) -> None:
    spawn_calls = tuple(
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call) and _create_task_scheduler(node) is not None
    )
    if len(spawn_calls) != 1 or not _callsite_lineno_is_planned(plan, spawn_calls[0].lineno):
        raise ValueError("callback-backed owner must contain exactly one planned spawn site")


def _validate_shared_task_group_scope(
    owner: ast.AsyncFunctionDef,
    loop: ast.For,
    receive: ast.Call,
    scheduler_name: str,
) -> None:
    matching_scopes = tuple(
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.AsyncWith)
        and isinstance(node.end_lineno, int)
        and node.lineno <= loop.lineno <= node.end_lineno
        and node.lineno <= receive.lineno <= node.end_lineno
        and any(
            isinstance(item.optional_vars, ast.Name)
            and item.optional_vars.id == scheduler_name
            and isinstance(item.context_expr, ast.Call)
            and _attribute_path(item.context_expr.func) == ("asyncio", "TaskGroup")
            for item in node.items
        )
    )
    if len(matching_scopes) != 1:
        raise ValueError("callback-backed receive must remain inside the spawning TaskGroup")


def _validate_queue_assignment(
    tree: ast.Module,
    owner: ast.AsyncFunctionDef,
    queue_name: str,
    capacity: int | None,
) -> None:
    expected = capacity if capacity is not None else -1
    matches = 0
    for node in ast.walk(owner):
        target: ast.expr
        value: ast.expr | None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            value = node.value
        else:
            continue
        if not isinstance(target, ast.Name) or target.id != queue_name:
            continue
        if value is None or not _is_positive_asyncio_queue_call(tree, value, expected):
            raise ValueError("private queue must be asyncio.Queue with known positive capacity")
        matches += 1
    if matches != 1:
        raise ValueError(f"expected exactly one private queue assignment for {queue_name}")


def _is_positive_asyncio_queue_call(
    tree: ast.Module,
    node: ast.expr,
    capacity: int,
) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if not (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "Queue"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "asyncio"
    ):
        return False
    capacity_node = node.args[0] if node.args else None
    if capacity_node is None:
        for keyword in node.keywords:
            if keyword.arg == "maxsize":
                capacity_node = keyword.value
                break
    resolved_capacity = _module_integer_constant(tree, capacity_node)
    return resolved_capacity is not None and resolved_capacity > 0 and resolved_capacity == capacity


def _module_integer_constant(tree: ast.Module, expression: ast.expr | None) -> int | None:
    if isinstance(expression, ast.Constant) and isinstance(expression.value, int):
        return expression.value
    if not isinstance(expression, ast.Name):
        return None
    matches: list[int] = []
    for statement in tree.body:
        value = _assigned_integer_constant(statement, expression.id)
        if value is not None:
            matches.append(value)
    return matches[0] if len(matches) == 1 else None


def _assigned_integer_constant(statement: ast.stmt, name: str) -> int | None:
    if isinstance(statement, ast.Assign):
        if (
            len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == name
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, int)
        ):
            return statement.value.value
        return None
    if (
        isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
        and statement.target.id == name
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, int)
    ):
        return statement.value.value
    return None


def _producer_name(plan: ExecutionPlan) -> str:
    symbols = tuple(
        node.symbol.qualname
        for node in plan.nodes
        if node.role == "producer" and node.symbol is not None
    )
    if len(symbols) != 1:
        raise ValueError("callback-backed lowering requires exactly one producer")
    return symbols[0]


def _queue_name(plan: ExecutionPlan) -> str:
    if plan.completion_transport is None:
        raise ValueError("completion transport must be statically known")
    return plan.completion_transport


def _support_names(plan: ExecutionPlan) -> _SupportNames:
    suffix = re.sub(r"[^A-Za-z0-9_]", "_", plan.id.rsplit("-", maxsplit=1)[-1])
    prefix = f"_atoll_callback_{suffix}"
    return _SupportNames(
        original=f"{prefix}_original",
        items=f"{prefix}_items",
        spawn=f"{prefix}_spawn",
        receive=f"{prefix}_receive",
        state=f"{prefix}_state",
        last_mode=f"{prefix}_last_mode",
    )


def _support_source(names: _SupportNames, producer_name: str) -> str:
    return f"""# Callback-backed execution-plan support. Appended by Atoll.
{names.original} = {producer_name}
{names.state} = {{}}
{names.last_mode} = None

class _AtollCallbackFailure:
    __slots__ = ("exception", "traceback")

    def __init__(self, exception, traceback):
        self.exception = exception
        self.traceback = traceback


def {names.items}(items, queue, create_task, producer, original_producer):
    global {names.last_mode}
    import asyncio
    import sys

    loop = asyncio.get_running_loop()
    if type(items) in (list, tuple, range) and len(items) == 0:
        {names.state}.pop(id(queue), None)
        {names.last_mode} = "fallback"
        return items
    optimized = (
        type(queue) is asyncio.Queue
        and type(items) in (list, tuple, range)
        and len(items) > 0
        and queue.maxsize > 0
        and len(items) <= queue.maxsize
        and queue.empty()
        and producer is original_producer
        and producer is {names.original}
        and getattr(create_task, "__self__", None).__class__ is asyncio.TaskGroup
        and getattr(create_task, "__func__", None) is asyncio.TaskGroup.create_task
        and asyncio.current_task(loop) is not None
        and len(asyncio.all_tasks(loop)) == 1
        and loop.get_task_factory() is None
        and not loop.get_debug()
        and sys.gettrace() is None
        and sys.getprofile() is None
        and _atoll_callback_no_monitoring_hooks(sys)
        and not getattr(loop, "_ready", ())
        and not getattr(loop, "_scheduled", ())
    )
    {names.state}[id(queue)] = {{
        "optimized": optimized,
        "remaining": len(items),
    }}
    {names.last_mode} = "optimized" if optimized else "fallback"
    return items


def {names.spawn}(queue, create_task, producer, item):
    import asyncio
    import contextvars

    state = {names.state}.get(id(queue))
    if not state or not state["optimized"]:
        return create_task(producer(queue, item))
    loop = asyncio.get_running_loop()
    context = contextvars.copy_context()
    loop.call_soon(context.run, _atoll_callback_drive_once, producer, queue, item)
    return None


async def {names.receive}(queue):
    key = id(queue)
    try:
        item = await queue.get()
        if isinstance(item, _AtollCallbackFailure):
            raise ExceptionGroup(
                "callback-backed execution plan failed",
                [item.exception.with_traceback(item.traceback)],
            )
        state = {names.state}.get(key)
        if state is not None:
            state["remaining"] -= 1
            if state["remaining"] <= 0:
                {names.state}.pop(key, None)
        return item
    except BaseException:
        {names.state}.pop(key, None)
        raise


def _atoll_callback_drive_once(producer, queue, item):
    coroutine = producer(queue, item)
    try:
        coroutine.send(None)
    except StopIteration:
        return
    except BaseException as error:
        queue.put_nowait(_AtollCallbackFailure(error, error.__traceback__))
        return
    try:
        coroutine.close()
    finally:
        error = RuntimeError("callback-backed producer unexpectedly suspended")
        queue.put_nowait(_AtollCallbackFailure(error, error.__traceback__))


def _atoll_callback_no_monitoring_hooks(sys_module):
    monitoring = getattr(sys_module, "monitoring", None)
    if monitoring is None:
        return True
    get_events = getattr(monitoring, "get_events", None)
    if get_events is None:
        return True
    for tool_id in range(6):
        try:
            if get_events(tool_id):
                return False
        except ValueError:
            continue
    return True
"""


CALLBACK_BACKED_BACKEND: Final = CallbackBackedExecutionPlanBackend()
"""Shared callback-backed execution-plan backend instance for integration."""
