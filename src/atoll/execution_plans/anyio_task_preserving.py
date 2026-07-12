"""Task-preserving lowering for structured AnyIO fan-out on asyncio.

This backend recognizes a narrow two-loop dispatch shape and retains the
source scheduler unchanged. It removes only cancellation of workers that have
already made a statically terminal private-stream handoff, and it can specialize
a linked hot reducer's guarded signature arity. One real task, every checkpoint,
rendezvous backpressure, ordering, and all nonterminal cancellation remain in
the original AnyIO implementation.
"""

from __future__ import annotations

import ast
import hashlib
import re
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
    source_function_node as _function_node,
)
from atoll.execution_plans.task_preserving import (
    source_module_path as _module_path,
)
from atoll.execution_plans.task_preserving import (
    source_name_exists as _name_exists,
)
from atoll.execution_plans.task_preserving import (
    source_sha256 as _sha256,
)
from atoll.execution_plans.task_preserving import (
    source_splice_expressions as _splice_expressions,
)
from atoll.execution_plans.task_preserving import (
    source_validate_hash as _validate_source_hash,
)
from atoll.models import SymbolId

_BACKEND_NAME: Final = "anyio-task-preserving"
_LOWERING_VERSION: Final = "anyio-task-preserving-v7"
_SUPPORTED_DIALECT: Final = "anyio-on-asyncio"
_CLASS_METHOD_PARTS: Final = 2
_FIELD_PATH_PARTS: Final = 2
_SCHEDULER_PATH_PARTS: Final = 3
_SPAWN_ARG_COUNT: Final = 2
_STREAM_ENDPOINT_COUNT: Final = 2
_POP_ARGUMENT_COUNT: Final = 2
_DISPATCH_BODY_STATEMENTS: Final = 2
_MIN_METHOD_ARGUMENTS: Final = 2
_BINARY_CALL_ARGUMENTS: Final = 2


@dataclass(frozen=True, slots=True)
class AnyioTaskPreservingExecutionPlanBackend:
    """Fuse proven AnyIO dispatch loops without changing child task semantics.

    Attributes:
        name: Stable backend identifier recorded in assessments, staged outputs, and
            cache fingerprints.
    """

    name: str = _BACKEND_NAME

    def assess(
        self,
        plan: ExecutionPlan,
        context: ExecutionPlanAssessmentContext,
    ) -> ExecutionPlanAssessment:
        """Determine whether the source still has the strict dispatch shape.

        Args:
            plan: Profile-selected scheduler execution plan.
            context: Source root and profile state used for read-only assessment.

        Returns:
            ExecutionPlanAssessment: Complete capability evidence for every plan node.
        """
        reasons = list(_static_rejection_reasons(plan))
        source_path = _module_path(context.source_root, plan.source_module)
        if source_path is None:
            reasons.append("source module is not present below the assessment source root")
        elif not reasons:
            try:
                _validate_assessment_sources(source_path, context.source_root, plan)
            except (TypeError, ValueError) as error:
                reasons.append(str(error))
        node_ids = tuple(node.id for node in plan.nodes)
        return ExecutionPlanAssessment(
            plan_id=plan.id,
            backend=self.name,
            status="unsupported" if reasons else "supported",
            supported_nodes=() if reasons else node_ids,
            unsupported_nodes=node_ids if reasons else (),
            reasons=tuple(reasons),
        )

    def stage(
        self,
        plan: ExecutionPlan,
        context: ExecutionPlanStageContext,
    ) -> StagedExecutionPlan:
        """Stage the guarded dispatch fast path in an unpacked wheel payload.

        Args:
            plan: Source-hashed execution plan accepted by this backend.
            context: Disposable payload and persistent cache boundaries.

        Returns:
            StagedExecutionPlan: Changed source evidence and runtime guards.

        Raises:
            ValueError: If the source, call site, or dispatch topology changed.
        """
        reasons = _static_rejection_reasons(plan)
        if reasons:
            raise ValueError(
                f"unsupported AnyIO task-preserving execution plan: {'; '.join(reasons)}"
            )
        payload_path = _module_path(context.payload_root, plan.source_module)
        if payload_path is None:
            raise ValueError(f"payload module is not present: {plan.source_module}")
        reducer_inputs: list[tuple[SymbolId, Path, str, _DirectReducer | None]] = []
        for reducer in _reflection_symbols(plan):
            reducer_path = _module_path(context.payload_root, reducer.module)
            if reducer_path is None:
                raise ValueError(f"linked reducer payload module is not present: {reducer.module}")
            reducer_source = reducer_path.read_text(encoding="utf-8")
            reducer_inputs.append(
                (
                    reducer,
                    reducer_path,
                    reducer_source,
                    _direct_reducer(reducer_source, reducer),
                )
            )
        source_text = payload_path.read_text(encoding="utf-8")
        transformed = _validated_rewrite(
            source_text,
            plan,
            tuple(item[3] for item in reducer_inputs if item[3] is not None),
        )
        rewrites: list[tuple[Path, str, str, str]] = [
            (payload_path, source_text, transformed.source_text, "source-overlay")
        ]
        for reducer, reducer_path, reducer_source, _direct in reducer_inputs:
            reducer_rewrite = _validated_reflection_rewrite(
                reducer_source,
                plan,
                reducer.module,
                reducer.qualname,
            )
            rewrites.append(
                (reducer_path, reducer_source, reducer_rewrite.source_text, "reducer-overlay")
            )
        for path, _before, after, _role in rewrites:
            path.write_text(after, encoding="utf-8")
        payload_files = tuple(
            ChangedPayloadFile(
                install_path=PurePosixPath(path.relative_to(context.payload_root).as_posix()),
                before_hash=_sha256(before),
                after_hash=_sha256(after),
                role=role,
            )
            for path, before, after, role in rewrites
        )
        return StagedExecutionPlan(
            plan=plan,
            backend=self.name,
            payload_files=payload_files,
            required_imports=(),
            guards=(
                *plan.guards,
                PlanGuard(
                    kind="semantics",
                    expression=transformed.guard_expression,
                    message=(
                        "terminal handoff is enabled only for the exact owner, private AnyIO "
                        "stream state, scope map, and instrumentation state"
                    ),
                ),
            ),
        )

    def fingerprint(
        self,
        plan: ExecutionPlan,
        context: ExecutionPlanStageContext,
    ) -> str:
        """Hash backend semantics, plan identity, and current payload source.

        Args:
            plan: Execution plan being fingerprinted.
            context: Payload root containing the module to rewrite.

        Returns:
            str: Stable cache fingerprint.

        Raises:
            ValueError: If the payload module is absent.
        """
        digest = hashlib.sha256()
        module_names = tuple(
            dict.fromkeys((plan.source_module, *(item[0] for item in plan.source_hashes)))
        )
        payload_hashes: list[str] = []
        for module_name in module_names:
            payload_path = _module_path(context.payload_root, module_name)
            if payload_path is None:
                raise ValueError(f"payload module is not present: {module_name}")
            payload_hashes.append(
                f"{module_name}:{_sha256(payload_path.read_text(encoding='utf-8'))}"
            )
        for part in (
            self.name,
            _LOWERING_VERSION,
            plan.id,
            plan.source_module,
            plan.source_hash,
            plan.callsite_fingerprint,
            plan.topology_fingerprint,
            plan.dialect,
            plan.lowering_version,
            *payload_hashes,
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
        """Normalize assessment, staging, or trial failures for report v4.

        Args:
            error: Exception raised by backend work.
            diagnostics: Captured diagnostic lines.
            log_path: Optional complete diagnostic log.

        Returns:
            ExecutionPlanDiagnostic: Stable backend-independent evidence.
        """
        details = tuple(line.strip() for line in diagnostics.splitlines() if line.strip())
        if log_path is not None:
            details = (*details, f"log: {log_path}")
        return ExecutionPlanDiagnostic(
            code="ANYIO_TASK_PRESERVING_EXECUTION_PLAN_ERROR",
            severity="error",
            message=str(error) or error.__class__.__name__,
            details=details,
        )


@dataclass(frozen=True, slots=True)
class _Rewrite:
    source_text: str
    guard_expression: str


@dataclass(frozen=True, slots=True)
class _DispatchTarget:
    owner: ast.FunctionDef
    registration_loop: ast.For
    spawn_loop: ast.For
    receiver_name: str
    request_name: str
    item_name: str
    scheduler_field: str
    worker_method: str
    registry_field: str
    key_field: str
    owner_class: str


@dataclass(frozen=True, slots=True)
class _TerminalHandoffTarget:
    """Private producer handoff and cleanup pair proven safe to coordinate.

    Attributes:
        producer: Async producer method that performs terminal private-stream sends.
        cleanup: Async cleanup method that cancels no-longer-needed receive scopes.
        terminal_sends: Producer send expressions proven to hand off terminal items.
        producer_receiver: Producer local name for the receive stream endpoint.
        item_key: Producer local name that identifies the terminal item key.
        cleanup_receiver: Cleanup local name for the receive stream endpoint.
        cleanup_key: Cleanup local name that identifies the terminal item key.
        cleanup_cancel: Cleanup branch that cancels the matching receive scope.
        cleanup_scope_name: Local cancellation-scope name used by the cleanup branch.
        scope_mapping_field: Owner field that maps item keys to cancellation scopes.
        sender_field: Owner field that stores the producer-side stream endpoint.
        receiver_field: Owner field that stores the consumer-side stream endpoint.
    """

    producer: ast.AsyncFunctionDef
    cleanup: ast.AsyncFunctionDef
    terminal_sends: tuple[ast.Expr, ...]
    producer_receiver: str
    item_key: str
    cleanup_receiver: str
    cleanup_key: str
    cleanup_cancel: ast.If
    cleanup_scope_name: str
    scope_mapping_field: str
    sender_field: str
    receiver_field: str


@dataclass(frozen=True, slots=True)
class _ReflectionTarget:
    """One linked reducer signature lookup eligible for guarded caching.

    Attributes:
        module: Import module containing the linked reducer.
        function: Reducer function or method that owns the signature lookup.
        assignment: Assignment node that stores the inspected callable signature.
        callable_expression: Expression passed to the signature lookup.
        signature_expression: Source expression for the signature object.
    """

    module: str
    function: ast.FunctionDef | ast.AsyncFunctionDef
    assignment: ast.Assign
    callable_expression: ast.expr
    signature_expression: str


@dataclass(frozen=True, slots=True)
class _DirectReducer:
    """Dispatch-only reducer method eligible for guarded consumer inlining.

    Attributes:
        module: Import module containing the reducer method.
        owner_class: Reducer class that owns the dispatch-only method.
        method_name: Method name invoked by the consumer hot path.
        callable_field: Instance field that stores the callable being reduced.
        signature_expression: Source expression for the cached signature.
        cast_expression: Optional source expression used to cast call arguments.
        argument_count: Number of positional arguments passed to the reducer.
        short_argument_indices: Argument indexes used by the short-call branch.
        long_argument_indices: Argument indexes used by the varargs-call branch.
        pure_context_classes: Context classes whose allocation can be delayed.
    """

    module: str
    owner_class: str
    method_name: str
    callable_field: str
    signature_expression: str
    cast_expression: str | None
    argument_count: int
    short_argument_indices: tuple[int, ...]
    long_argument_indices: tuple[int, ...]
    pure_context_classes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PureContext:
    """Side-effect-free context allocation that may be delayed until used.

    Attributes:
        class_name: Context class constructed by the delayed allocation.
        expression: Source expression that constructs the pure context.
        assignment: Assignment node that binds the context before reduction.
    """

    class_name: str
    expression: str
    assignment: ast.Assign


@dataclass(frozen=True, slots=True)
class _ConsumerReducerCall:
    """One consumer assignment that invokes a proven dispatch-only reducer.

    Attributes:
        reducer: Direct reducer metadata matched to the consumer call.
        assignment: Consumer assignment that receives the reducer result.
        call: Reducer call expression inside the consumer assignment.
        receiver_expression: Expression used as the reducer call receiver.
        class_expression: Source expression that resolves the receiver class.
        context: Pure context allocation tied to the call, when one is present.
    """

    reducer: _DirectReducer
    assignment: ast.Assign
    call: ast.Call
    receiver_expression: ast.expr
    class_expression: str
    context: _PureContext | None = None


@dataclass(frozen=True, slots=True)
class _SupportNames:
    owner_class: str
    last_mode: str
    monitoring: str
    original_all_events: str
    states: str
    enable: str
    terminal_state: str
    reducer_module: str
    reducer_class: str
    reducer_method: str
    reducer_descriptor: str
    reducer_signature: str
    reducer_cast: str
    reducer_function_type: str
    reducer_sys: str
    reducer_varargs: str
    reducer_varkeywords: str
    reducer_callable: str
    reducer_code: str
    reducer_arity: str
    context_class: str
    context_init: str
    context_new: str
    context_setattr: str


@dataclass(frozen=True, slots=True)
class _ReflectionNames:
    """Collision-checked support bindings for one reflection specialization.

    Attributes:
        helper: Helper function name used for guarded signature reuse.
        original_signature: Binding name for the original signature function.
        function_type: Binding name for the runtime function type.
        sys_module: Binding name for the imported `sys` module.
        varargs_flag: Binding name for the signature varargs sentinel.
        varkeywords_flag: Binding name for the signature varkeywords sentinel.
        original_all_events: Binding name for instrumentation event state.
        last_callable: Binding name for the last callable identity.
        last_code: Binding name for the last callable code object.
        last_arity: Binding name for the last callable arity.
        monitoring: Binding name for runtime monitoring state.
    """

    helper: str
    original_signature: str
    function_type: str
    sys_module: str
    varargs_flag: str
    varkeywords_flag: str
    original_all_events: str
    last_callable: str
    last_code: str
    last_arity: str
    monitoring: str


def _validate_assessment_sources(
    source_path: Path,
    source_root: Path,
    plan: ExecutionPlan,
) -> None:
    """Validate every source module covered by one candidate before assessment.

    Args:
        source_path: Primary source module path for the execution plan.
        source_root: Root used to resolve linked reducer modules during assessment.
        plan: Candidate execution plan whose source shape must still match.

    Raises:
        ValueError: If a linked reducer module is missing or the source no longer
            matches the guarded AnyIO task-preserving rewrite shape.
        TypeError: If parsed source expressions have an unexpected syntax shape.
    """
    direct_reducers: list[_DirectReducer] = []
    for reducer in _reflection_symbols(plan):
        reducer_path = _module_path(source_root, reducer.module)
        if reducer_path is None:
            raise ValueError(f"linked reducer module is not present: {reducer.module}")
        reducer_source = reducer_path.read_text(encoding="utf-8")
        direct = _direct_reducer(reducer_source, reducer)
        if direct is not None:
            direct_reducers.append(direct)
        _validated_reflection_rewrite(
            reducer_source,
            plan,
            reducer.module,
            reducer.qualname,
        )
    _validated_rewrite(
        source_path.read_text(encoding="utf-8"),
        plan,
        tuple(direct_reducers),
    )


def _static_rejection_reasons(plan: ExecutionPlan) -> tuple[str, ...]:
    reasons: list[str] = []
    if plan.dialect != _SUPPORTED_DIALECT:
        reasons.append(f"unsupported scheduler dialect: {plan.dialect}")
    if plan.task_ownership != "structured":
        reasons.append(f"unsupported task ownership: {plan.task_ownership}")
    if plan.transport_capacity is None or plan.transport_capacity < 0:
        reasons.append("transport capacity must be statically known")
    if plan.completion_transport is None:
        reasons.append("completion transport must be statically known")
    if sum(edge.kind == "spawns" for edge in plan.edges) != 1:
        reasons.append("plan must contain exactly one spawn edge")
    if len(plan.owner.qualname.split(".")) != _CLASS_METHOD_PARTS:
        reasons.append("plan owner must be a direct class method")
    return tuple(reasons)


def _validated_rewrite(
    source_text: str,
    plan: ExecutionPlan,
    direct_reducers: tuple[_DirectReducer, ...] = (),
) -> _Rewrite:
    tree = ast.parse(source_text, type_comments=True)
    _validate_source_hash(source_text, tree, plan)
    _validate_callsite_fingerprint(tree, plan)
    target = _dispatch_target(tree, plan)
    handoff = _terminal_handoff_target(tree, plan, target)
    consumer_reducer = _consumer_reducer_call(tree, plan, direct_reducers)
    names = _support_names(plan.id)
    for name in (
        names.owner_class,
        names.last_mode,
        names.monitoring,
        names.original_all_events,
        names.states,
        names.enable,
        names.terminal_state,
        names.reducer_module,
        names.reducer_class,
        names.reducer_method,
        names.reducer_descriptor,
        names.reducer_signature,
        names.reducer_cast,
        names.reducer_function_type,
        names.reducer_sys,
        names.reducer_varargs,
        names.reducer_varkeywords,
        names.reducer_callable,
        names.reducer_code,
        names.reducer_arity,
        names.context_class,
        names.context_init,
        names.context_new,
        names.context_setattr,
    ):
        if _name_exists(tree, name):
            raise ValueError(f"AnyIO task-preserving support name already exists: {name}")
    newline = _newline(source_text)
    edits: list[tuple[int, int, str]] = [_fused_dispatch_edit(source_text, target, names, newline)]
    if consumer_reducer is not None:
        edits.append(_direct_reducer_edit(source_text, consumer_reducer, names, newline))
    for send in handoff.terminal_sends:
        indent = _line_indent(source_text, send.lineno)
        edits.append(
            (
                send.lineno,
                send.lineno - 1,
                (
                    f"{indent}{names.terminal_state} = {names.states}.get("
                    f"id({handoff.producer_receiver})){newline}"
                    f"{indent}if {names.terminal_state} is not None and "
                    f"{names.terminal_state}[0]() is {handoff.producer_receiver}:{newline}"
                    f"{indent}    {handoff.producer_receiver}."
                    f"{handoff.scope_mapping_field}[{handoff.item_key}] = None{newline}"
                ),
            )
        )
    staged_source = _apply_line_edits(source_text, tuple(edits))
    staged_source = (
        staged_source.rstrip()
        + newline
        + newline
        + _support_source(names, target, handoff, plan, consumer_reducer)
    )
    return _Rewrite(
        source_text=staged_source,
        guard_expression=(
            "exact owner, private AnyIO stream state and capacity, private scope map, "
            "exact built-in request and registry types, unmodified AnyIO task group, "
            "default task factory, dispatch-only reducer identities, debug, trace, profile, "
            "and monitoring identities"
        ),
    )


def _fused_dispatch_edit(
    source_text: str,
    target: _DispatchTarget,
    names: _SupportNames,
    newline: str,
) -> tuple[int, int, str]:
    """Build a guarded one-pass dispatch with the original loops as fallback.

    The optimized branch is selected only for exact built-in requests and an
    unmodified AnyIO asyncio task group. The fallback embeds the original source
    text so custom factories, custom sequences, and dynamic scheduler objects
    retain their original registration-before-scheduling behavior.

    Args:
        source_text: Original module source containing both adjacent loops.
        target: Statically validated registration and spawn loop facts.
        names: Collision-checked generated support names.
        newline: Source newline convention.

    Returns:
        A line edit replacing both loops with guarded fused and fallback arms.
    """
    start = target.registration_loop.lineno
    end = target.spawn_loop.end_lineno or target.spawn_loop.lineno
    indent = _line_indent(source_text, start)
    body_indent = f"{indent}    "
    loop_indent = f"{body_indent}    "
    original = _source_lines(source_text, start, end)
    fallback = "".join(
        f"    {line}" if line.strip() else line for line in original.splitlines(keepends=True)
    )
    fused = (
        f"{indent}if {names.enable}({target.receiver_name}, {target.request_name}):{newline}"
        f"{body_indent}for {target.item_name} in {target.request_name}:{newline}"
        f"{loop_indent}{target.receiver_name}.{target.registry_field}"
        f"[{target.item_name}.{target.key_field}] = {target.item_name}{newline}"
        f"{loop_indent}{target.receiver_name}.{target.scheduler_field}.start_soon("
        f"{target.receiver_name}.{target.worker_method}, {target.item_name}){newline}"
        f"{indent}else:{newline}"
        f"{fallback}"
    )
    return start, end, fused


def _validate_callsite_fingerprint(tree: ast.Module, plan: ExecutionPlan) -> None:
    owner = _function_node(tree, plan.owner.qualname)
    if owner is None:
        raise ValueError(f"plan owner is missing from payload source: {plan.owner.qualname}")
    parts: list[str] = []
    for node in ast.walk(owner):
        if not isinstance(node, ast.Call):
            continue
        path = _attribute_path(node.func)
        if path is None or path[-1] != "start_soon" or not node.args:
            continue
        callee_path = _attribute_path(node.args[0])
        if callee_path is None:
            continue
        parts.append(
            f"{_SUPPORTED_DIALECT}:{node.lineno}:{node.col_offset}:{'.'.join(callee_path)}"
        )
    if _digest_parts(parts) != plan.callsite_fingerprint:
        raise ValueError("payload call-site fingerprint does not match the selected plan")


def _dispatch_target(tree: ast.Module, plan: ExecutionPlan) -> _DispatchTarget:
    owner = _function_node(tree, plan.owner.qualname)
    if not isinstance(owner, ast.FunctionDef):
        raise TypeError("AnyIO task-preserving owner must be synchronous")
    owner_parts = plan.owner.qualname.split(".")
    receiver_name = _receiver_name(owner)
    spawn_candidates: list[tuple[int, ast.For, ast.Call]] = []
    for index, statement in enumerate(owner.body):
        call = _single_start_soon_call(statement)
        if (
            isinstance(statement, ast.For)
            and call is not None
            and any(edge.kind == "spawns" and edge.lineno == call.lineno for edge in plan.edges)
        ):
            spawn_candidates.append((index, statement, call))
    if len(spawn_candidates) != 1:
        raise ValueError(
            f"expected exactly one direct start_soon loop, found {len(spawn_candidates)}"
        )
    spawn_index, spawn_loop, spawn_call = spawn_candidates[0]
    if spawn_index < 1 or spawn_index != len(owner.body) - 1:
        raise ValueError("start_soon loop must immediately follow registration and end the owner")
    registration_loop = owner.body[spawn_index - 1]
    if not isinstance(registration_loop, ast.For):
        raise TypeError("start_soon loop must follow a registration loop")
    request_name, item_name = _matching_loop_names(registration_loop, spawn_loop)
    scheduler_field, worker_method = _spawn_fields(spawn_call, receiver_name, item_name)
    registry_field, key_field = _registration_fields(
        registration_loop,
        receiver_name,
        item_name,
    )
    producer_methods = {
        node.symbol.qualname.split(".")[-1]
        for node in plan.nodes
        if node.role == "producer" and node.symbol is not None
    }
    if producer_methods != {worker_method}:
        raise ValueError("spawned worker no longer matches the planned producer")
    producer = _function_node(tree, f"{owner_parts[0]}.{worker_method}")
    if not isinstance(producer, ast.AsyncFunctionDef) or any(
        isinstance(node, ast.Yield | ast.YieldFrom) for node in ast.walk(producer)
    ):
        raise ValueError("spawned worker must remain an ordinary coroutine method")
    return _DispatchTarget(
        owner=owner,
        registration_loop=registration_loop,
        spawn_loop=spawn_loop,
        receiver_name=receiver_name,
        request_name=request_name,
        item_name=item_name,
        scheduler_field=scheduler_field,
        worker_method=worker_method,
        registry_field=registry_field,
        key_field=key_field,
        owner_class=owner_parts[0],
    )


def _terminal_handoff_target(
    tree: ast.Module,
    plan: ExecutionPlan,
    dispatch: _DispatchTarget,
) -> _TerminalHandoffTarget:
    """Prove terminal private-stream sends and their keyed cleanup method.

    Args:
        tree: Parsed orchestration module.
        plan: Source-hashed execution plan.
        dispatch: Validated owner and producer dispatch shape.

    Returns:
        _TerminalHandoffTarget: Exact producer sends and cleanup cancellation to rewrite.

    Raises:
        ValueError: If ownership, tail position, scope storage, or cleanup is ambiguous.
    """
    producer = _function_node(tree, f"{dispatch.owner_class}.{dispatch.worker_method}")
    if not isinstance(producer, ast.AsyncFunctionDef):
        raise TypeError("terminal handoff producer is not an ordinary coroutine method")
    producer_args = (*producer.args.posonlyargs, *producer.args.args)
    if len(producer_args) < _SPAWN_ARG_COUNT:
        raise ValueError("terminal handoff producer has no stable work-item parameter")
    producer_receiver = producer_args[0].arg
    item_name = producer_args[1].arg
    sender_field, receiver_field = _transport_fields(plan, producer_receiver)
    scope_mapping_field, item_key = _scope_registration(
        producer,
        producer_receiver,
        item_name,
    )
    terminal_sends = _terminal_send_expressions(
        producer,
        (producer_receiver, sender_field, "send"),
    )
    if not terminal_sends:
        raise ValueError("producer has no statically terminal private-stream send")
    cleanup = _cleanup_target(
        tree,
        dispatch.owner_class,
        scope_mapping_field,
        plan,
    )
    return _TerminalHandoffTarget(
        producer=producer,
        cleanup=cleanup[0],
        terminal_sends=terminal_sends,
        producer_receiver=producer_receiver,
        item_key=item_key,
        cleanup_receiver=cleanup[1],
        cleanup_key=cleanup[2],
        cleanup_cancel=cleanup[3],
        cleanup_scope_name=cleanup[4],
        scope_mapping_field=scope_mapping_field,
        sender_field=sender_field,
        receiver_field=receiver_field,
    )


def _transport_fields(plan: ExecutionPlan, receiver_name: str) -> tuple[str, str]:
    transport = plan.completion_transport
    if transport is None:
        raise ValueError("terminal handoff has no private completion transport")
    endpoints = transport.split("|")
    prefix = f"{receiver_name}."
    if len(endpoints) != _STREAM_ENDPOINT_COUNT or any(
        not endpoint.startswith(prefix) for endpoint in endpoints
    ):
        raise ValueError("terminal handoff requires two direct instance stream endpoints")
    fields = tuple(endpoint[len(prefix) :] for endpoint in endpoints)
    if any("." in field or not field.isidentifier() for field in fields):
        raise ValueError("terminal handoff stream endpoint is not a stable instance field")
    return fields[0], fields[1]


def _scope_registration(
    producer: ast.AsyncFunctionDef,
    receiver_name: str,
    item_name: str,
) -> tuple[str, str]:
    parents = {
        child: parent for parent in ast.walk(producer) for child in ast.iter_child_nodes(parent)
    }
    candidates: list[tuple[str, str]] = []
    for assignment in ast.walk(producer):
        if (
            not isinstance(assignment, ast.Assign)
            or len(assignment.targets) != 1
            or not isinstance(assignment.targets[0], ast.Subscript)
            or not isinstance(assignment.value, ast.Name)
        ):
            continue
        target = assignment.targets[0]
        mapping_path = _attribute_path(target.value)
        key_path = _attribute_path(target.slice)
        if (
            mapping_path is None
            or len(mapping_path) != _FIELD_PATH_PARTS
            or mapping_path[0] != receiver_name
            or key_path is None
            or len(key_path) < _FIELD_PATH_PARTS
            or key_path[0] != item_name
            or not _inside_named_with_scope(
                assignment,
                assignment.value.id,
                parents,
            )
        ):
            continue
        candidates.append((mapping_path[1], ast.unparse(target.slice)))
    if len(candidates) != 1:
        raise ValueError(
            f"expected one private cancellation-scope registration, found {len(candidates)}"
        )
    return candidates[0]


def _inside_named_with_scope(
    node: ast.AST,
    scope_name: str,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    parent = parents.get(node)
    while parent is not None:
        if isinstance(parent, ast.With | ast.AsyncWith) and any(
            isinstance(item.optional_vars, ast.Name) and item.optional_vars.id == scope_name
            for item in parent.items
        ):
            return True
        if isinstance(parent, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            return False
        parent = parents.get(parent)
    return False


def _terminal_send_expressions(
    producer: ast.AsyncFunctionDef,
    send_path: tuple[str, ...],
) -> tuple[ast.Expr, ...]:
    return _terminal_sends_in_block(producer.body, send_path, continuation_is_terminal=True)


def _terminal_sends_in_block(
    statements: list[ast.stmt],
    send_path: tuple[str, ...],
    *,
    continuation_is_terminal: bool,
) -> tuple[ast.Expr, ...]:
    terminal: list[ast.Expr] = []
    for index, statement in enumerate(statements):
        suffix_is_terminal = continuation_is_terminal and _terminal_suffix(statements[index + 1 :])
        terminal.extend(_terminal_sends_in_statement(statement, send_path, suffix_is_terminal))
    return tuple(terminal)


def _terminal_sends_in_statement(
    statement: ast.stmt,
    send_path: tuple[str, ...],
    continuation_is_terminal: bool,
) -> tuple[ast.Expr, ...]:
    if isinstance(statement, ast.Expr) and _is_send_await(statement, send_path):
        return (statement,) if continuation_is_terminal else ()
    child_blocks = _terminal_child_blocks(statement)
    return tuple(
        send
        for block in child_blocks
        for send in _terminal_sends_in_block(
            block,
            send_path,
            continuation_is_terminal=continuation_is_terminal,
        )
    )


def _terminal_child_blocks(statement: ast.stmt) -> tuple[list[ast.stmt], ...]:
    if isinstance(statement, ast.If):
        return statement.body, statement.orelse
    if isinstance(statement, ast.Try | ast.TryStar):
        if statement.finalbody:
            return ()
        return (
            statement.body,
            statement.orelse,
            *(handler.body for handler in statement.handlers),
        )
    if isinstance(statement, ast.With | ast.AsyncWith):
        return (statement.body,)
    if isinstance(statement, ast.Match):
        return tuple(case.body for case in statement.cases)
    return ()


def _terminal_suffix(statements: list[ast.stmt]) -> bool:
    return all(
        isinstance(statement, ast.Pass)
        or (isinstance(statement, ast.Return) and statement.value is None)
        for statement in statements
    )


def _is_send_await(statement: ast.Expr, send_path: tuple[str, ...]) -> bool:
    value = statement.value
    return (
        isinstance(value, ast.Await)
        and isinstance(value.value, ast.Call)
        and _attribute_path(value.value.func) == send_path
        and len(value.value.args) == 1
        and not value.value.keywords
    )


def _cleanup_target(
    tree: ast.Module,
    owner_class: str,
    scope_mapping_field: str,
    plan: ExecutionPlan,
) -> tuple[ast.AsyncFunctionDef, str, str, ast.If, str]:
    owner = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner_class),
        None,
    )
    if owner is None:
        raise ValueError("terminal handoff owner class is missing")
    planned_members = set(plan.source_members)
    candidates: list[tuple[ast.AsyncFunctionDef, str, str, ast.If, str]] = []
    for method in owner.body:
        if not isinstance(method, ast.AsyncFunctionDef):
            continue
        method_id = f"{owner_class}.{method.name}"
        if planned_members and not any(
            member.module == plan.source_module and member.qualname == method_id
            for member in planned_members
        ):
            continue
        arguments = (*method.args.posonlyargs, *method.args.args)
        if len(arguments) < _SPAWN_ARG_COUNT:
            continue
        receiver_name, key_name = arguments[0].arg, arguments[1].arg
        for assignment in ast.walk(method):
            matched = _scope_pop_assignment(
                assignment,
                receiver_name,
                key_name,
                scope_mapping_field,
            )
            if matched is None:
                continue
            cancel = _scope_cancel_if(method, matched)
            if cancel is not None:
                candidates.append((method, receiver_name, key_name, cancel, matched))
    if len(candidates) != 1:
        raise ValueError(f"expected one private scope cleanup, found {len(candidates)}")
    return candidates[0]


def _scope_pop_assignment(
    node: ast.AST,
    receiver_name: str,
    key_name: str,
    scope_mapping_field: str,
) -> str | None:
    if (
        not isinstance(node, ast.Assign)
        or len(node.targets) != 1
        or not isinstance(node.targets[0], ast.Name)
        or not isinstance(node.value, ast.Call)
        or node.value.keywords
        or len(node.value.args) != _POP_ARGUMENT_COUNT
        or not isinstance(node.value.args[0], ast.Name)
        or node.value.args[0].id != key_name
        or not isinstance(node.value.args[1], ast.Constant)
        or node.value.args[1].value is not None
    ):
        return None
    path = _attribute_path(node.value.func)
    expected = (receiver_name, scope_mapping_field, "pop")
    return node.targets[0].id if path == expected else None


def _scope_cancel_if(
    method: ast.AsyncFunctionDef,
    scope_name: str,
) -> ast.If | None:
    candidates = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.If)
        and _is_not_none_test(node.test, scope_name)
        and not node.orelse
        and len(node.body) == 1
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Call)
        and not node.body[0].value.args
        and not node.body[0].value.keywords
        and _attribute_path(node.body[0].value.func) == (scope_name, "cancel")
    ]
    return candidates[0] if len(candidates) == 1 else None


def _is_not_none_test(test: ast.expr, name: str) -> bool:
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == name
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.IsNot)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value is None
    )


def _reflection_symbols(plan: ExecutionPlan) -> tuple[SymbolId, ...]:
    """Return linked reducer symbols distinct from the plan's consumer node.

    Args:
        plan: Execution plan whose reducer nodes should be inspected.

    Returns:
        tuple[SymbolId, ...]: Ordered unique reducer symbols outside the consumer
            and owner nodes.
    """
    symbols = tuple(
        node.symbol
        for node in plan.nodes
        if node.role == "reducer"
        and node.symbol is not None
        and node.symbol not in (plan.consumer, plan.owner)
    )
    return tuple(dict.fromkeys(symbols))


def _direct_reducer(source_text: str, symbol: SymbolId) -> _DirectReducer | None:
    """Recognize a reducer whose complete behavior is arity-based dispatch.

    Args:
        source_text: Complete source for the reducer module.
        symbol: Class method selected as the plan's reducer.

    Returns:
        A direct-dispatch description, or `None` when any extra behavior is
        present and the normal reflection-only rewrite must be retained.
    """
    shape = _direct_reducer_shape(source_text, symbol)
    if shape is None:
        return None
    tree, parts, function, assignment, branch = shape
    reflection = _reflection_target(symbol.module, function, assignment)
    if reflection is None or not isinstance(assignment.targets[0], ast.Name):
        return None
    arity_name = assignment.targets[0].id
    short_count = _arity_branch_count(branch.test, arity_name)
    if short_count is None or len(branch.body) != 1 or len(branch.orelse) != 1:
        return None
    short = _dispatch_return(branch.body[0], function)
    long = _dispatch_return(branch.orelse[0], function)
    if (
        short is None
        or long is None
        or short[0] != long[0]
        or short[2] != long[2]
        or short_count != len(short[1])
    ):
        return None
    short_field, short_indices, short_cast = short
    _long_field, long_indices, _long_cast = long
    parameters = (*function.args.posonlyargs, *function.args.args)
    if len(parameters) < _MIN_METHOD_ARGUMENTS:
        return None
    return _DirectReducer(
        module=symbol.module,
        owner_class=parts[0],
        method_name=parts[1],
        callable_field=short_field,
        signature_expression=reflection.signature_expression,
        cast_expression=short_cast,
        argument_count=len(parameters) - 1,
        short_argument_indices=short_indices,
        long_argument_indices=long_indices,
        pure_context_classes=_pure_context_class_names(tree),
    )


def _direct_reducer_shape(
    source_text: str,
    symbol: SymbolId,
) -> tuple[ast.Module, tuple[str, str], ast.FunctionDef, ast.Assign, ast.If] | None:
    parts = symbol.qualname.split(".")
    if len(parts) != _CLASS_METHOD_PARTS:
        return None
    tree = ast.parse(source_text, type_comments=True)
    function = _function_node(tree, symbol.qualname)
    if (
        not isinstance(function, ast.FunctionDef)
        or function.args.vararg
        or function.args.kwarg
        or function.args.kwonlyargs
    ):
        return None
    body = list(function.body)
    if body and _is_docstring_statement(body[0]):
        body.pop(0)
    if (
        len(body) != _DISPATCH_BODY_STATEMENTS
        or not isinstance(body[0], ast.Assign)
        or not isinstance(body[1], ast.If)
    ):
        return None
    return tree, (parts[0], parts[1]), function, body[0], body[1]


def _pure_context_class_names(tree: ast.Module) -> tuple[str, ...]:
    return tuple(
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and _is_pure_context_class(node)
    )


def _is_pure_context_class(node: ast.ClassDef) -> bool:
    forbidden = {
        "__new__",
        "__setattr__",
        "__getattribute__",
        "__getattr__",
        "__del__",
        "__post_init__",
    }
    methods = {
        child.name: child
        for child in node.body
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    initializer = methods.get("__init__")
    return (
        not node.keywords
        and all(_is_dataclass_decorator(decorator) for decorator in node.decorator_list)
        and all(_is_generic_base(base) for base in node.bases)
        and not forbidden.intersection(methods)
        and isinstance(initializer, ast.FunctionDef)
        and _is_assignment_only_initializer(initializer)
    )


def _is_dataclass_decorator(decorator: ast.expr) -> bool:
    candidate = decorator.func if isinstance(decorator, ast.Call) else decorator
    path = _attribute_path(candidate)
    return path is not None and path[-1] == "dataclass"


def _is_generic_base(base: ast.expr) -> bool:
    candidate = base.value if isinstance(base, ast.Subscript) else base
    path = _attribute_path(candidate)
    return path is not None and path[-1] == "Generic"


def _is_assignment_only_initializer(function: ast.FunctionDef) -> bool:
    if function.decorator_list or function.args.vararg or function.args.kwarg:
        return False
    parameters = (
        *function.args.posonlyargs,
        *function.args.args,
        *function.args.kwonlyargs,
    )
    if len(parameters) < _MIN_METHOD_ARGUMENTS:
        return False
    receiver = parameters[0].arg
    expected_values = {parameter.arg for parameter in parameters[1:]}
    body = list(function.body)
    if body and _is_docstring_statement(body[0]):
        body.pop(0)
    assigned_values: set[str] = set()
    for statement in body:
        if (
            not isinstance(statement, ast.Assign)
            or len(statement.targets) != 1
            or not isinstance(statement.targets[0], ast.Attribute)
            or not isinstance(statement.targets[0].value, ast.Name)
            or statement.targets[0].value.id != receiver
            or not isinstance(statement.value, ast.Name)
            or statement.value.id not in expected_values
        ):
            return False
        assigned_values.add(statement.value.id)
    return assigned_values == expected_values and len(body) == len(expected_values)


def _is_docstring_statement(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def _arity_branch_count(test: ast.expr, arity_name: str) -> int | None:
    if (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == arity_name
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and type(test.comparators[0].value) is int
    ):
        return int(test.comparators[0].value)
    return None


def _dispatch_return(
    statement: ast.stmt,
    function: ast.FunctionDef,
) -> tuple[str, tuple[int, ...], str | None] | None:
    if not isinstance(statement, ast.Return) or not isinstance(statement.value, ast.Call):
        return None
    call = statement.value
    if call.keywords:
        return None
    callable_expression = call.func
    cast_expression: str | None = None
    if isinstance(callable_expression, ast.Call):
        cast_path = _attribute_path(callable_expression.func)
        if (
            cast_path is None
            or cast_path[-1] != "cast"
            or len(callable_expression.args) != _BINARY_CALL_ARGUMENTS
            or callable_expression.keywords
        ):
            return None
        cast_expression = ast.unparse(callable_expression.func)
        callable_expression = callable_expression.args[1]
    callable_path = _attribute_path(callable_expression)
    parameters = (*function.args.posonlyargs, *function.args.args)
    if (
        callable_path is None
        or len(callable_path) != _FIELD_PATH_PARTS
        or not parameters
        or callable_path[0] != parameters[0].arg
    ):
        return None
    parameter_indices = {parameter.arg: index for index, parameter in enumerate(parameters[1:])}
    indices: list[int] = []
    for argument in call.args:
        if not isinstance(argument, ast.Name) or argument.id not in parameter_indices:
            return None
        indices.append(parameter_indices[argument.id])
    return callable_path[1], tuple(indices), cast_expression


def _consumer_reducer_call(
    tree: ast.Module,
    plan: ExecutionPlan,
    reducers: tuple[_DirectReducer, ...],
) -> _ConsumerReducerCall | None:
    consumer = plan.consumer
    if consumer is None or consumer.module != plan.source_module:
        return None
    function = _function_node(tree, consumer.qualname)
    if function is None:
        return None
    candidates: list[_ConsumerReducerCall] = []
    for reducer in reducers:
        for node in ast.walk(function):
            if (
                not isinstance(node, ast.Assign)
                or len(node.targets) != 1
                or not isinstance(node.value, ast.Call)
                or node.value.keywords
                or len(node.value.args) != reducer.argument_count
                or not isinstance(node.value.func, ast.Attribute)
                or node.value.func.attr != reducer.method_name
                or not isinstance(node.value.func.value, ast.Name)
            ):
                continue
            receiver = node.value.func.value
            class_expression = _asserted_class_expression(
                function,
                receiver.id,
                node.lineno,
                reducer.owner_class,
            )
            if class_expression is None:
                continue
            candidates.append(
                _ConsumerReducerCall(
                    reducer=reducer,
                    assignment=node,
                    call=node.value,
                    receiver_expression=receiver,
                    class_expression=class_expression,
                    context=_pure_context_assignment(function, node, reducer),
                )
            )
    return candidates[0] if len(candidates) == 1 else None


def _asserted_class_expression(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    receiver_name: str,
    before_lineno: int,
    owner_class: str,
) -> str | None:
    candidates: list[tuple[int, str]] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Assert) or node.lineno >= before_lineno:
            continue
        test = node.test
        if (
            not isinstance(test, ast.Call)
            or _attribute_path(test.func) != ("isinstance",)
            or len(test.args) != _BINARY_CALL_ARGUMENTS
            or test.keywords
            or not isinstance(test.args[0], ast.Name)
            or test.args[0].id != receiver_name
        ):
            continue
        type_path = _attribute_path(test.args[1])
        if type_path is None or type_path[-1] != owner_class:
            continue
        candidates.append((node.lineno, ast.unparse(test.args[1])))
    return max(candidates)[1] if candidates else None


def _pure_context_assignment(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    assignment: ast.Assign,
    reducer: _DirectReducer,
) -> _PureContext | None:
    if not isinstance(assignment.value, ast.Call):
        return None
    context_indices = set(reducer.long_argument_indices).difference(reducer.short_argument_indices)
    if len(context_indices) != 1:
        return None
    context_index = context_indices.pop()
    if context_index >= len(assignment.value.args):
        return None
    argument = assignment.value.args[context_index]
    previous = _previous_statement(function.body, assignment)
    if (
        not isinstance(argument, ast.Name)
        or not isinstance(previous, ast.Assign)
        or len(previous.targets) != 1
        or not isinstance(previous.targets[0], ast.Name)
        or previous.targets[0].id != argument.id
        or not isinstance(previous.value, ast.Call)
        or previous.value.args
        or any(keyword.arg is None for keyword in previous.value.keywords)
    ):
        return None
    class_path = _attribute_path(previous.value.func)
    if class_path is None or class_path[-1] not in reducer.pure_context_classes:
        return None
    return _PureContext(
        class_name=class_path[-1],
        expression=ast.unparse(previous.value.func),
        assignment=previous,
    )


def _previous_statement(statements: list[ast.stmt], target: ast.stmt) -> ast.stmt | None:
    for index, statement in enumerate(statements):
        if statement is target:
            return statements[index - 1] if index else None
        for block in _statement_blocks(statement):
            previous = _previous_statement(block, target)
            if previous is not None:
                return previous
    return None


def _statement_blocks(statement: ast.stmt) -> tuple[list[ast.stmt], ...]:
    if isinstance(statement, ast.If | ast.For | ast.AsyncFor | ast.While):
        return statement.body, statement.orelse
    if isinstance(statement, ast.Try | ast.TryStar):
        return (
            statement.body,
            statement.orelse,
            statement.finalbody,
            *(handler.body for handler in statement.handlers),
        )
    if isinstance(statement, ast.With | ast.AsyncWith):
        return (statement.body,)
    if isinstance(statement, ast.Match):
        return tuple(case.body for case in statement.cases)
    return ()


def _validated_reflection_rewrite(
    source_text: str,
    plan: ExecutionPlan,
    module_name: str,
    qualname: str,
) -> _Rewrite:
    tree = ast.parse(source_text, type_comments=True)
    _validate_source_hash(source_text, tree, plan, module_name)
    if _module_binds_name(tree, "len"):
        raise ValueError("linked reducer module shadows the required len builtin")
    function = _function_node(tree, qualname)
    if function is None:
        raise ValueError(f"linked reducer is missing from payload source: {qualname}")
    targets = tuple(
        target
        for node in ast.walk(function)
        for target in (_reflection_target(module_name, function, node),)
        if target is not None
    )
    if len(targets) != 1:
        raise ValueError(f"expected one guarded signature lookup, found {len(targets)}")
    target = targets[0]
    names = _reflection_names(plan.id, module_name, qualname)
    for name in (
        names.helper,
        names.original_signature,
        names.function_type,
        names.sys_module,
        names.varargs_flag,
        names.varkeywords_flag,
        names.original_all_events,
        names.last_callable,
        names.last_code,
        names.last_arity,
        names.monitoring,
    ):
        if _name_exists(tree, name):
            raise ValueError(f"reflection support name already exists: {name}")
    replacement = f"{names.helper}({ast.unparse(target.callable_expression)})"
    newline = _newline(source_text)
    rewritten = _splice_expressions(source_text, ((target.assignment.value, replacement),))
    rewritten = (
        rewritten.rstrip()
        + newline
        + newline
        + _reflection_support_source(names, target.signature_expression)
    )
    return _Rewrite(
        source_text=rewritten,
        guard_expression=(
            "signature specialization requires the original inspect functions, an unwrapped Python "
            "function, stable code identity, and no active tracing or monitoring"
        ),
    )


def _reflection_target(
    module_name: str,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    node: ast.AST,
) -> _ReflectionTarget | None:
    if (
        not isinstance(node, ast.Assign)
        or len(node.targets) != 1
        or not isinstance(node.targets[0], ast.Name)
        or not isinstance(node.value, ast.Call)
        or not isinstance(node.value.func, ast.Name)
        or node.value.func.id != "len"
        or len(node.value.args) != 1
        or node.value.keywords
        or not isinstance(node.value.args[0], ast.Attribute)
        or node.value.args[0].attr != "parameters"
        or not isinstance(node.value.args[0].value, ast.Call)
    ):
        return None
    signature_call = node.value.args[0].value
    if len(signature_call.args) != 1 or signature_call.keywords:
        return None
    signature_path = _attribute_path(signature_call.func)
    if signature_path is None or signature_path[-1] != "signature":
        return None
    return _ReflectionTarget(
        module=module_name,
        function=function,
        assignment=node,
        callable_expression=signature_call.args[0],
        signature_expression=ast.unparse(signature_call.func),
    )


def _module_binds_name(tree: ast.Module, name: str) -> bool:
    for statement in tree.body:
        if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            if statement.name == name:
                return True
            continue
        if isinstance(statement, ast.Import | ast.ImportFrom):
            if any((alias.asname or alias.name.split(".")[0]) == name for alias in statement.names):
                return True
            continue
        if any(
            isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Store)
            for node in ast.walk(statement)
        ):
            return True
    return False


def _reflection_names(plan_id: str, module_name: str, qualname: str) -> _ReflectionNames:
    digest = hashlib.sha256(f"{plan_id}:{module_name}:{qualname}".encode()).hexdigest()[:16]
    prefix = f"_atoll_signature_arity_{digest}"
    return _ReflectionNames(
        helper=prefix,
        original_signature=f"{prefix}_signature",
        function_type=f"{prefix}_function_type",
        sys_module=f"{prefix}_sys",
        varargs_flag=f"{prefix}_varargs",
        varkeywords_flag=f"{prefix}_varkeywords",
        original_all_events=f"{prefix}_all_events",
        last_callable=f"{prefix}_last_callable",
        last_code=f"{prefix}_last_code",
        last_arity=f"{prefix}_last_arity",
        monitoring=f"{prefix}_no_monitoring",
    )


def _reflection_support_source(names: _ReflectionNames, signature_expression: str) -> str:
    return f"""# Guarded reflection specialization appended by Atoll.
{names.original_signature} = {signature_expression}
{names.function_type} = type(lambda: None)
{names.sys_module} = __import__("sys")
{names.varargs_flag} = __import__("inspect").CO_VARARGS
{names.varkeywords_flag} = __import__("inspect").CO_VARKEYWORDS
{names.original_all_events} = getattr(
    getattr({names.sys_module}, "monitoring", None), "_all_events", None
)
{names.last_callable} = None
{names.last_code} = None
{names.last_arity} = 0

def {names.helper}(callable_value):
    global {names.last_callable}, {names.last_code}, {names.last_arity}

    signature = {signature_expression}
    specializable = (
        signature is {names.original_signature}
        and type(callable_value) is {names.function_type}
        and not hasattr(callable_value, "__signature__")
        and not hasattr(callable_value, "__wrapped__")
        and not hasattr(callable_value, "__text_signature__")
        and {names.sys_module}.gettrace() is None
        and {names.sys_module}.getprofile() is None
        and {names.monitoring}({names.sys_module})
    )
    if specializable:
        code = callable_value.__code__
        if callable_value is {names.last_callable} and code is {names.last_code}:
            return {names.last_arity}
        arity = (
            code.co_argcount
            + code.co_kwonlyargcount
            + bool(code.co_flags & {names.varargs_flag})
            + bool(code.co_flags & {names.varkeywords_flag})
        )
        {names.last_callable} = callable_value
        {names.last_code} = code
        {names.last_arity} = arity
        return arity
    return len(signature(callable_value).parameters)

def {names.monitoring}(sys_module):
    monitoring = getattr(sys_module, "monitoring", None)
    if monitoring is None:
        return True
    all_events = getattr(monitoring, "_all_events", None)
    if all_events is {names.original_all_events} and all_events is not None:
        return not all_events()
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


def _receiver_name(owner: ast.FunctionDef) -> str:
    arguments = (*owner.args.posonlyargs, *owner.args.args)
    if not arguments:
        raise ValueError("owner method has no receiver parameter")
    return arguments[0].arg


def _single_start_soon_call(statement: ast.stmt) -> ast.Call | None:
    if (
        not isinstance(statement, ast.For)
        or len(statement.body) != 1
        or statement.orelse
        or not isinstance(statement.body[0], ast.Expr)
        or not isinstance(statement.body[0].value, ast.Call)
    ):
        return None
    call = statement.body[0].value
    path = _attribute_path(call.func)
    return call if path is not None and path[-1] == "start_soon" else None


def _matching_loop_names(registration: ast.For, spawn: ast.For) -> tuple[str, str]:
    if (
        not isinstance(registration.iter, ast.Name)
        or not isinstance(spawn.iter, ast.Name)
        or registration.iter.id != spawn.iter.id
        or not isinstance(registration.target, ast.Name)
        or not isinstance(spawn.target, ast.Name)
        or registration.target.id != spawn.target.id
        or registration.orelse
        or spawn.orelse
    ):
        raise ValueError("registration and spawn loops must share local request and item names")
    return registration.iter.id, registration.target.id


def _spawn_fields(call: ast.Call, receiver_name: str, item_name: str) -> tuple[str, str]:
    scheduler_path = _attribute_path(call.func)
    worker_path = _attribute_path(call.args[0]) if call.args else None
    if (
        scheduler_path is None
        or len(scheduler_path) != _SCHEDULER_PATH_PARTS
        or scheduler_path[0] != receiver_name
        or scheduler_path[-1] != "start_soon"
        or worker_path is None
        or len(worker_path) != _FIELD_PATH_PARTS
        or worker_path[0] != receiver_name
        or len(call.args) != _SPAWN_ARG_COUNT
        or not isinstance(call.args[1], ast.Name)
        or call.args[1].id != item_name
        or call.keywords
    ):
        raise ValueError("start_soon call must use one receiver method and one loop item")
    return scheduler_path[1], worker_path[1]


def _registration_fields(
    loop: ast.For,
    receiver_name: str,
    item_name: str,
) -> tuple[str, str]:
    if len(loop.body) != 1 or not isinstance(loop.body[0], ast.Assign):
        raise ValueError("registration loop must contain one assignment")
    assignment = loop.body[0]
    if len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Subscript):
        raise ValueError("registration loop must assign one mapping entry")
    target = assignment.targets[0]
    registry_path = _attribute_path(target.value)
    key_path = _attribute_path(target.slice)
    if (
        registry_path is None
        or len(registry_path) != _FIELD_PATH_PARTS
        or registry_path[0] != receiver_name
        or key_path is None
        or len(key_path) != _FIELD_PATH_PARTS
        or key_path[0] != item_name
        or not isinstance(assignment.value, ast.Name)
        or assignment.value.id != item_name
    ):
        raise ValueError("registration assignment must map the loop item's stable key to itself")
    return registry_path[1], key_path[1]


def _direct_reducer_edit(
    source_text: str,
    target: _ConsumerReducerCall,
    names: _SupportNames,
    newline: str,
) -> tuple[int, int, str]:
    """Inline a guarded dispatch-only reducer at its hot consumer assignment.

    Args:
        source_text: Original consumer-module source.
        target: Proven reducer call and positional argument mapping.
        names: Collision-checked generated globals and locals.
        newline: Source newline convention.

    Returns:
        Inclusive source-line edit containing optimized and original fallback arms.
    """
    reducer = target.reducer
    assignment = target.assignment
    context_assignment = target.context.assignment if target.context is not None else None
    start = context_assignment.lineno if context_assignment is not None else assignment.lineno
    end = assignment.end_lineno or assignment.lineno
    indent = _line_indent(source_text, start)
    level1 = f"{indent}    "
    level2 = f"{level1}    "
    level3 = f"{level2}    "
    receiver = ast.unparse(target.receiver_expression)
    assignment_target = ast.unparse(assignment.targets[0])
    fallback = ast.unparse(assignment)
    context_statement = ast.unparse(context_assignment) if context_assignment is not None else ""
    arguments = tuple(ast.unparse(argument) for argument in target.call.args)
    short_arguments = ", ".join(arguments[index] for index in reducer.short_argument_indices)
    long_arguments = ", ".join(arguments[index] for index in reducer.long_argument_indices)
    signature_expression = _module_expression(names.reducer_module, reducer.signature_expression)
    cast_guard = ""
    if reducer.cast_expression is not None:
        cast_expression = _module_expression(names.reducer_module, reducer.cast_expression)
        cast_guard = f"{newline}{level1}and {cast_expression} is {names.reducer_cast}"
    context_guard = ""
    if target.context is not None:
        context_guard = (
            f"{newline}{level1}and {names.reducer_module}.{target.context.class_name} "
            f"is {names.context_class}{newline}"
            f"{level1}and {names.context_class}.__dict__.get('__init__') "
            f"is {names.context_init}{newline}"
            f"{level1}and {names.context_class}.__new__ is {names.context_new}{newline}"
            f"{level1}and {names.context_class}.__setattr__ is {names.context_setattr}"
        )
    guard = (
        f"{names.last_mode} == 'optimized'{newline}"
        f"{level1}and type({receiver}) is {names.reducer_class}{newline}"
        f"{level1}and {names.reducer_class}.__dict__.get({reducer.method_name!r}) "
        f"is {names.reducer_method}{newline}"
        f"{level1}and {names.reducer_class}.__dict__.get({reducer.callable_field!r}) "
        f"is {names.reducer_descriptor}{newline}"
        f"{level1}and {names.reducer_sys}.modules.get({reducer.module!r}) "
        f"is {names.reducer_module}{newline}"
        f"{level1}and {signature_expression} is {names.reducer_signature}"
        f"{cast_guard}{context_guard}{newline}"
        f"{level1}and {names.reducer_sys}.gettrace() is None{newline}"
        f"{level1}and {names.reducer_sys}.getprofile() is None{newline}"
        f"{level1}and {names.monitoring}({names.reducer_sys})"
    )
    long_context = f"{level3}{context_statement}{newline}" if context_statement else ""
    nested_fallback_context = f"{level2}{context_statement}{newline}" if context_statement else ""
    outer_fallback_context = f"{level1}{context_statement}{newline}" if context_statement else ""
    replacement = (
        f"{indent}if ({guard}{newline}{indent}):{newline}"
        f"{level1}{names.reducer_callable} = {receiver}.{reducer.callable_field}{newline}"
        f"{level1}if ({newline}"
        f"{level2}type({names.reducer_callable}) is {names.reducer_function_type}{newline}"
        f"{level2}and not hasattr({names.reducer_callable}, '__signature__'){newline}"
        f"{level2}and not hasattr({names.reducer_callable}, '__wrapped__'){newline}"
        f"{level2}and not hasattr({names.reducer_callable}, '__text_signature__'){newline}"
        f"{level1}):{newline}"
        f"{level2}{names.reducer_code} = {names.reducer_callable}.__code__{newline}"
        f"{level2}{names.reducer_arity} = ({newline}"
        f"{level3}{names.reducer_code}.co_argcount{newline}"
        f"{level3}+ {names.reducer_code}.co_kwonlyargcount{newline}"
        f"{level3}+ bool({names.reducer_code}.co_flags & {names.reducer_varargs}){newline}"
        f"{level3}+ bool({names.reducer_code}.co_flags & {names.reducer_varkeywords}){newline}"
        f"{level2}){newline}"
        f"{level2}if {names.reducer_arity} == {len(reducer.short_argument_indices)}:{newline}"
        f"{level3}{assignment_target} = {names.reducer_callable}({short_arguments}){newline}"
        f"{level2}else:{newline}"
        f"{long_context}"
        f"{level3}{assignment_target} = {names.reducer_callable}({long_arguments}){newline}"
        f"{level1}else:{newline}"
        f"{nested_fallback_context}"
        f"{level2}{fallback}{newline}"
        f"{indent}else:{newline}"
        f"{outer_fallback_context}"
        f"{level1}{fallback}{newline}"
    )
    return (
        start,
        end,
        replacement,
    )


def _module_expression(module_binding: str, expression: str) -> str:
    path = _attribute_path(ast.parse(expression, mode="eval").body)
    if path is None:
        raise ValueError("direct reducer runtime expression is not a stable attribute path")
    return ".".join((module_binding, *path))


def _direct_reducer_support(
    names: _SupportNames,
    target: _ConsumerReducerCall | None,
) -> str:
    if target is None:
        return ""
    reducer = target.reducer
    signature = _module_expression(names.reducer_module, reducer.signature_expression)
    cast = (
        _module_expression(names.reducer_module, reducer.cast_expression)
        if reducer.cast_expression is not None
        else "None"
    )
    if target.context is None:
        context_support = (
            f"{names.context_class} = None\n"
            f"{names.context_init} = None\n"
            f"{names.context_new} = None\n"
            f"{names.context_setattr} = None"
        )
    else:
        context_support = (
            f"{names.context_class} = {target.context.expression}\n"
            f"{names.context_init} = {names.context_class}.__dict__.get('__init__')\n"
            f"{names.context_new} = {names.context_class}.__new__\n"
            f"{names.context_setattr} = {names.context_class}.__setattr__"
        )
    return f"""
{names.reducer_sys} = __import__("sys")
{names.reducer_module} = {names.reducer_sys}.modules[{reducer.module!r}]
{names.reducer_class} = {target.class_expression}
{names.reducer_method} = {names.reducer_class}.__dict__.get({reducer.method_name!r})
{names.reducer_descriptor} = {names.reducer_class}.__dict__.get({reducer.callable_field!r})
{names.reducer_signature} = {signature}
{names.reducer_cast} = {cast}
{names.reducer_function_type} = type(lambda: None)
{names.reducer_varargs} = __import__("inspect").CO_VARARGS
{names.reducer_varkeywords} = __import__("inspect").CO_VARKEYWORDS
{context_support}
""".strip()


def _support_names(plan_id: str) -> _SupportNames:
    suffix = re.sub(r"[^A-Za-z0-9_]", "_", plan_id.rsplit("-", maxsplit=1)[-1])
    prefix = f"_atoll_anyio_dispatch_{suffix}"
    return _SupportNames(
        owner_class=f"{prefix}_owner_class",
        last_mode=f"{prefix}_last_mode",
        monitoring=f"{prefix}_no_monitoring",
        original_all_events=f"{prefix}_all_events",
        states=f"{prefix}_states",
        enable=f"{prefix}_enable",
        terminal_state=f"{prefix}_terminal_state",
        reducer_module=f"{prefix}_reducer_module",
        reducer_class=f"{prefix}_reducer_class",
        reducer_method=f"{prefix}_reducer_method",
        reducer_descriptor=f"{prefix}_reducer_descriptor",
        reducer_signature=f"{prefix}_reducer_signature",
        reducer_cast=f"{prefix}_reducer_cast",
        reducer_function_type=f"{prefix}_reducer_function_type",
        reducer_sys=f"{prefix}_reducer_sys",
        reducer_varargs=f"{prefix}_reducer_varargs",
        reducer_varkeywords=f"{prefix}_reducer_varkeywords",
        reducer_callable=f"{prefix}_reducer_callable",
        reducer_code=f"{prefix}_reducer_code",
        reducer_arity=f"{prefix}_reducer_arity",
        context_class=f"{prefix}_context_class",
        context_init=f"{prefix}_context_init",
        context_new=f"{prefix}_context_new",
        context_setattr=f"{prefix}_context_setattr",
    )


def _support_source(
    names: _SupportNames,
    target: _DispatchTarget,
    handoff: _TerminalHandoffTarget,
    plan: ExecutionPlan,
    consumer_reducer: _ConsumerReducerCall | None,
) -> str:
    capacity = plan.transport_capacity
    if capacity is None:
        raise ValueError("AnyIO terminal handoff requires a known stream capacity")
    reducer_support = _direct_reducer_support(names, consumer_reducer)
    return f"""# AnyIO terminal-handoff support appended by Atoll.
{names.owner_class} = {target.owner_class}
{names.last_mode} = None
{names.states} = {{}}
{names.original_all_events} = getattr(
    getattr(__import__("sys"), "monitoring", None), "_all_events", None
)
{reducer_support}

def {names.enable}(owner, request):
    global {names.last_mode}
    import asyncio
    import sys
    import weakref

    loop = asyncio.get_running_loop()
    scope_mapping = owner.{handoff.scope_mapping_field}
    sender = owner.{handoff.sender_field}
    receiver = owner.{handoff.receiver_field}
    owner_id = id(owner)
    current = {names.states}.get(owner_id)
    if current is not None and current[0]() is owner:
        {names.last_mode} = "optimized"
        scheduler = owner.{target.scheduler_field}
        return (
            type(request) in (list, tuple)
            and type(owner.{target.registry_field}) is dict
            and current[3] is not None
            and type(scheduler) is current[3]
            and type(scheduler).start_soon is current[4]
            and loop.get_task_factory() is None
        )
    if current is not None:
        {names.states}.pop(owner_id, None)
    try:
        from anyio._backends._asyncio import TaskGroup as _AtollTaskGroup
        from anyio.streams.memory import MemoryObjectReceiveStream as _AtollReceiveStream
        from anyio.streams.memory import MemoryObjectSendStream as _AtollSendStream
    except ImportError:
        {names.last_mode} = "fallback"
        {names.states}.pop(owner_id, None)
        return False
    scheduler = owner.{target.scheduler_field}
    optimized = (
        type(owner) is {names.owner_class}
        and type(scope_mapping) is dict
        and type(sender) is _AtollSendStream
        and type(receiver) is _AtollReceiveStream
        and sender._state is receiver._state
        and sender._state.max_buffer_size == {capacity}
        and not loop.get_debug()
        and sys.gettrace() is None
        and sys.getprofile() is None
        and {names.monitoring}(sys)
    )
    if not optimized:
        {names.last_mode} = "fallback"
        {names.states}.pop(owner_id, None)
        return False
    try:
        owner_ref = weakref.ref(
            owner,
            lambda _reference, key=owner_id: {names.states}.pop(key, None),
        )
    except TypeError:
        {names.last_mode} = "fallback"
        {names.states}.pop(owner_id, None)
        return False
    {names.states}[owner_id] = (
        owner_ref,
        _AtollSendStream,
        _AtollReceiveStream,
        _AtollTaskGroup,
        _AtollTaskGroup.start_soon,
    )
    {names.last_mode} = "optimized"
    return (
        type(request) in (list, tuple)
        and type(owner.{target.registry_field}) is dict
        and type(scheduler) is _AtollTaskGroup
        and type(scheduler).start_soon is _AtollTaskGroup.start_soon
        and loop.get_task_factory() is None
)

def {names.monitoring}(sys_module):
    monitoring = getattr(sys_module, "monitoring", None)
    if monitoring is None:
        return True
    all_events = getattr(monitoring, "_all_events", None)
    if all_events is {names.original_all_events} and all_events is not None:
        return not all_events()
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


def _apply_line_edits(
    source_text: str,
    edits: tuple[tuple[int, int, str], ...],
) -> str:
    lines = source_text.splitlines(keepends=True)
    for start, end, replacement in sorted(edits, key=lambda item: item[0], reverse=True):
        if start < 1 or start > len(lines) + 1 or (end >= start and end > len(lines)):
            raise ValueError("AnyIO task-preserving edit coordinates are outside source")
        lines[start - 1 : end] = [replacement]
    return "".join(lines)


def _source_lines(source_text: str, start: int, end: int) -> str:
    """Return an inclusive one-based source-line segment.

    Args:
        source_text: Complete source text.
        start: First one-based line to include.
        end: Last one-based line to include.

    Returns:
        The original source segment with line endings preserved.

    Raises:
        ValueError: If the requested coordinates are outside the source.
    """
    lines = source_text.splitlines(keepends=True)
    if start < 1 or end < start or end > len(lines):
        raise ValueError("AnyIO task-preserving source coordinates are outside source")
    return "".join(lines[start - 1 : end])


def _line_indent(source_text: str, lineno: int) -> str:
    line = source_text.splitlines()[lineno - 1]
    return line[: len(line) - len(line.lstrip())]


def _newline(source_text: str) -> str:
    return "\r\n" if "\r\n" in source_text else "\n"


def _digest_parts(parts: list[str]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


ANYIO_TASK_PRESERVING_BACKEND: Final = AnyioTaskPreservingExecutionPlanBackend()
"""Shared AnyIO-on-asyncio task-preserving backend."""
