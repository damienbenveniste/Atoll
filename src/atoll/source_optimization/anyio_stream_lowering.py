"""Guarded LibCST lowering for class-owned AnyIO result streams.

The lowerer recognizes a structural fan-out/fan-in shape rather than project
identifiers: one method registers work through ``TaskGroup.start_soon``, one
producer delegates to a same-class run coroutine and sends result records to a
private memory stream, and one async generator consumes that stream. Eligible
batches are driven synchronously in copied contexts and delivered through a
private local deque. Uncertain batches execute the original scheduler path.

This module only creates a transformation request for a temporary project copy.
It does not mutate a checkout, run semantic commands, benchmark candidates, or
promote a patch.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import re
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast, override

import libcst as cst

from atoll.models import SymbolId
from atoll.source_optimization.models import SourceOptimizationPlan
from atoll.source_optimization.transforms import (
    CallableBodyReplacement,
    SourceTransformationRequest,
)

_PAIR_SIZE = 2
_REDUCER_ARGUMENT_COUNT = 4
_REDUCER_PROPERTY_READ_COUNT = 3
_CONSUMER_REDUCER_ARGUMENT_COUNT = 3


@dataclass(frozen=True, slots=True)
class AnyioStreamLowering:
    """One structurally proven AnyIO source transformation.

    Attributes:
        request: LibCST request that rewrites the spawn owner, consumer, and
            initializer while inserting module-level guarded helpers.
        helper_names: Generated guard, driver, and route-counter names exposed
            to strict optimized-routing tests.
    """

    request: SourceTransformationRequest
    helper_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _AnyioShape:
    class_name: str
    owner: ast.FunctionDef
    worker: ast.AsyncFunctionDef
    consumer: ast.AsyncFunctionDef
    initializer: ast.FunctionDef
    run_method: ast.AsyncFunctionDef
    request_name: str
    spawn_item_name: str
    task_group_expression: str
    task_group_method: str
    worker_method: str
    run_method_name: str
    run_task_name: str
    task_type_expression: str
    node_mapping_expression: str
    node_key_expression: str
    awaited_node_type: str
    awaited_callable_attribute: str
    excluded_node_types: tuple[str, ...]
    result_constructor: str
    result_empty_expression: str
    result_error_keyword: str
    async_result_type: str | None
    sender_expression: str
    receiver_expression: str
    transport_capacity: int
    reducer: _ReducerShape | None
    protocol: _ProtocolShape | None


@dataclass(frozen=True, slots=True)
class _ReducerShape:
    class_expression: str
    owner_expression: str
    method_name: str
    callable_property: str


@dataclass(frozen=True, slots=True)
class _ProtocolShape:
    entry_class: str
    entry_method: str
    runner_class: str
    runner_name: str
    next_method: str
    owner_attribute: str
    terminal_type: str


@dataclass(frozen=True, slots=True)
class _Names:
    suffix: str
    collections: str
    asyncio: str
    contextvars: str
    dis: str
    inspect: str
    os: str
    sys: str
    typing: str
    weakref: str
    anyio_backend: str
    anyio_send_stream: str
    anyio_receive_stream: str
    deque_attribute: str
    eligibility_cache_attribute: str
    route_hits: str
    guard: str
    no_monitoring: str
    eligible: str
    safe_callable: str
    safe_code: str
    complete: str
    synchronous_run: str
    drive: str
    expected_worker: str
    expected_worker_code: str
    expected_run: str
    expected_run_code: str
    expected_owner_class: str
    expected_task_group_class: str
    expected_start_soon: str
    expected_consumer: str
    expected_consumer_code: str
    expected_result_constructor: str
    fast_local: str
    reducer: str
    reducer_states: str
    expected_reducer_class: str
    expected_reducer_method: str
    expected_reducer_code: str
    expected_reducer_property: str
    protocol_context: str
    protocol_next: str
    protocol_forward: str
    expected_entry_class: str
    expected_entry_method: str
    expected_entry_code: str
    expected_runner_class: str
    expected_runner_next: str
    expected_runner_next_code: str
    expected_terminal_type: str

    @property
    def public(self) -> tuple[str, ...]:
        """Return helper names needed by strict route verification.

        Returns:
            tuple[str, ...]: Guard, driver, and route-counter names.
        """
        return (self.guard, self.drive, self.route_hits)


def lower_anyio_stream_plan(
    project_root: Path,
    plan: SourceOptimizationPlan,
) -> AnyioStreamLowering:
    """Lower a structurally proven AnyIO stream pipeline into a patch request.

    Args:
        project_root: Target project root containing ``plan.source``.
        plan: Trial-ready AnyIO-on-asyncio source plan.

    Returns:
        AnyioStreamLowering: Deterministic transformation request and route helpers.

    Raises:
        OSError: If the planned source cannot be read.
        SyntaxError: If the target source is not valid Python.
        ValueError: If source identity is stale or the scheduler, producer,
            consumer, run coroutine, result record, or class shape is unsafe.
    """
    source_path = _source_path(project_root, plan.source)
    source = source_path.read_text(encoding="utf-8")
    expected_hash = _expected_source_hash(plan)
    if _sha256(source) != expected_hash:
        raise ValueError(f"stale source for {plan.source.as_posix()}")
    tree = ast.parse(source, filename=str(source_path))
    shape = _analyze_shape(project_root, tree, plan)
    module = cst.parse_module(source)
    names = _names(plan.id)
    owner_body = _owner_body(module, plan.owner.qualname, shape, names)
    consumer_body = _consumer_body(module, plan.consumer, shape, names)
    initializer_body = _initializer_body(module, shape, names)
    request = SourceTransformationRequest(
        path=plan.source,
        expected_sha256=expected_hash,
        target=plan.owner,
        declaration_kind="method",
        replacement_body=owner_body,
        helper_statements=_helper_statements(shape, names),
        trailing_statements=_identity_captures(shape, names),
        additional_replacements=(
            CallableBodyReplacement(
                target=cast(SymbolId, plan.consumer),
                declaration_kind="async_method",
                replacement_body=consumer_body,
            ),
            CallableBodyReplacement(
                target=SymbolId(plan.owner.module, f"{shape.class_name}.__post_init__"),
                declaration_kind="method",
                replacement_body=initializer_body,
            ),
            *(
                (
                    CallableBodyReplacement(
                        target=SymbolId(
                            plan.owner.module,
                            f"{shape.protocol.entry_class}.{shape.protocol.entry_method}",
                        ),
                        declaration_kind="async_method",
                        replacement_body=_protocol_body(module, shape.protocol, names),
                    ),
                )
                if shape.protocol is not None
                else ()
            ),
        ),
        summary="add a guarded copied-context fast path for a private AnyIO result stream",
        transformation_id=f"anyio-stream-state-machine-v1:{plan.id}",
    )
    return AnyioStreamLowering(request=request, helper_names=names.public)


def _source_path(project_root: Path, relative: PurePosixPath) -> Path:
    root = project_root.resolve()
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe source path: {relative.as_posix()}")
    path = (root / Path(relative.as_posix())).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"source path escapes project root: {relative.as_posix()}") from error
    if not path.is_file():
        raise ValueError(f"source plan file does not exist: {relative.as_posix()}")
    return path


def _expected_source_hash(plan: SourceOptimizationPlan) -> str:
    matches = tuple(
        source_hash for path, source_hash in plan.identity.source_hashes if path == plan.source
    )
    if len(matches) != 1:
        raise ValueError("source plan must contain one exact hash for its owner file")
    return matches[0]


def _sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _analyze_shape(
    project_root: Path,
    tree: ast.Module,
    plan: SourceOptimizationPlan,
) -> _AnyioShape:
    if plan.consumer is None:
        raise ValueError("AnyIO stream lowering requires a distinct consumer")
    class_name = _shared_owner_class(plan.owner, plan.worker, plan.consumer)
    class_node = _class_node(tree, class_name)
    _validate_class(class_node)
    owner = cast(
        ast.FunctionDef,
        _method(class_node, plan.owner, ast.FunctionDef, role="owner"),
    )
    worker = cast(
        ast.AsyncFunctionDef,
        _method(class_node, plan.worker, ast.AsyncFunctionDef, role="worker"),
    )
    consumer = cast(
        ast.AsyncFunctionDef,
        _method(class_node, plan.consumer, ast.AsyncFunctionDef, role="consumer"),
    )
    initializer = cast(
        ast.FunctionDef,
        _named_method(class_node, "__post_init__", ast.FunctionDef, role="initializer"),
    )
    sender, receiver = _transport_pair(plan.transport)
    transport_capacity = _stream_capacity(initializer, sender, receiver)
    if plan.transport_capacity is not None and plan.transport_capacity != transport_capacity:
        raise ValueError("planned AnyIO stream capacity does not match source")
    (
        request_name,
        spawn_item_name,
        task_group_expression,
        task_group_method,
        worker_method,
    ) = _spawn_shape(owner, plan.worker)
    run_method_name, worker_task_name, result_name = _worker_delegate(
        worker,
        sender_expression=sender,
    )
    run_method = cast(
        ast.AsyncFunctionDef,
        _named_method(
            class_node,
            run_method_name,
            ast.AsyncFunctionDef,
            role="run coroutine",
        ),
    )
    if len(run_method.args.args) != _PAIR_SIZE:
        raise ValueError("run coroutine must accept only self and one work item")
    run_task_name = run_method.args.args[1].arg
    result_constructor, empty_expression, error_keyword, async_result_type = _result_shape(
        worker,
        sender_expression=sender,
        task_name=worker_task_name,
        result_name=result_name,
    )
    _validate_worker_wrapper(
        worker,
        run_method_name=run_method_name,
        sender_expression=sender,
        result_constructor=result_constructor,
        async_result_type=async_result_type,
    )
    (
        task_type_expression,
        node_mapping_expression,
        node_key_expression,
        awaited_node_type,
        callable_attribute,
        excluded_node_types,
    ) = _run_shape(
        class_node,
        run_method,
        run_task_name,
        async_result_type=async_result_type,
    )
    _validate_consumer(consumer, receiver)
    reducer = _reducer_shape(project_root, tree, plan, consumer)
    protocol = _protocol_shape(tree, class_name, consumer.name)
    return _AnyioShape(
        class_name=class_name,
        owner=owner,
        worker=worker,
        consumer=consumer,
        initializer=initializer,
        run_method=run_method,
        request_name=request_name,
        spawn_item_name=spawn_item_name,
        task_group_expression=task_group_expression,
        task_group_method=task_group_method,
        worker_method=worker_method,
        run_method_name=run_method_name,
        run_task_name=run_task_name,
        task_type_expression=task_type_expression,
        node_mapping_expression=node_mapping_expression,
        node_key_expression=node_key_expression,
        awaited_node_type=awaited_node_type,
        awaited_callable_attribute=callable_attribute,
        excluded_node_types=excluded_node_types,
        result_constructor=result_constructor,
        result_empty_expression=empty_expression,
        result_error_keyword=error_keyword,
        async_result_type=async_result_type,
        sender_expression=sender,
        receiver_expression=receiver,
        transport_capacity=transport_capacity,
        reducer=reducer,
        protocol=protocol,
    )


def _shared_owner_class(*symbols: SymbolId) -> str:
    owners = {symbol.qualname.rsplit(".", maxsplit=1)[0] for symbol in symbols}
    if len(owners) != 1 or any("." not in symbol.qualname for symbol in symbols):
        raise ValueError("AnyIO stream owner, worker, and consumer must be same-class methods")
    class_name = owners.pop()
    if "." in class_name:
        raise ValueError("nested owner classes are not supported")
    return class_name


def _class_node(tree: ast.Module, class_name: str) -> ast.ClassDef:
    matches = [
        statement
        for statement in tree.body
        if isinstance(statement, ast.ClassDef) and statement.name == class_name
    ]
    if len(matches) != 1:
        raise ValueError(f"source must contain one owner class {class_name}")
    return matches[0]


def _validate_class(node: ast.ClassDef) -> None:
    if any(keyword.arg == "metaclass" for keyword in node.keywords):
        raise ValueError("owner classes with metaclasses are not supported")
    if any(
        isinstance(statement, ast.Assign | ast.AnnAssign)
        and any(name == "__slots__" for name in _assigned_names(statement))
        for statement in node.body
    ):
        raise ValueError("slotted owner classes cannot receive a private result deque")
    for decorator in node.decorator_list:
        call = decorator if isinstance(decorator, ast.Call) else None
        path = _expression_path(call.func if call is not None else decorator)
        if path is None or path.rsplit(".", maxsplit=1)[-1] != "dataclass":
            continue
        if call is not None and any(
            keyword.arg == "slots" and _literal_true(keyword.value) for keyword in call.keywords
        ):
            raise ValueError("slotted dataclasses cannot receive a private result deque")


def _assigned_names(statement: ast.Assign | ast.AnnAssign) -> tuple[str, ...]:
    targets = statement.targets if isinstance(statement, ast.Assign) else (statement.target,)
    return tuple(target.id for target in targets if isinstance(target, ast.Name))


def _literal_true(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _method(
    class_node: ast.ClassDef,
    symbol: SymbolId,
    expected: type[ast.FunctionDef] | type[ast.AsyncFunctionDef],
    *,
    role: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    return _named_method(
        class_node,
        symbol.qualname.rsplit(".", maxsplit=1)[-1],
        expected,
        role=role,
    )


def _named_method(
    class_node: ast.ClassDef,
    name: str,
    expected: type[ast.FunctionDef] | type[ast.AsyncFunctionDef],
    *,
    role: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    matches = [
        statement
        for statement in class_node.body
        if isinstance(statement, expected) and statement.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"source must contain one {role} method {class_node.name}.{name}")
    method = matches[0]
    if method.decorator_list:
        raise ValueError(f"{role} method decorators are not supported")
    if not method.args.args or method.args.args[0].arg != "self":
        raise ValueError(f"{role} must be an instance method")
    return method


def _transport_pair(transport: str) -> tuple[str, str]:
    parts = tuple(part.strip() for part in transport.split("|") if part.strip())
    if len(parts) != _PAIR_SIZE or any(
        not re.fullmatch(r"self\.[A-Za-z_]\w*", part) for part in parts
    ):
        raise ValueError("AnyIO stream transport must be an exact self.sender|self.receiver pair")
    if parts[0] == parts[1]:
        raise ValueError("AnyIO stream sender and receiver must be distinct")
    return (parts[0], parts[1])


def _stream_capacity(
    initializer: ast.FunctionDef,
    sender_expression: str,
    receiver_expression: str,
) -> int:
    """Read the literal capacity of one private AnyIO stream factory call.

    Args:
        initializer: Owner initializer that creates the private stream endpoints.
        sender_expression: Exact sender attribute from the execution plan.
        receiver_expression: Exact receiver attribute from the execution plan.

    Returns:
        int: Non-negative literal capacity, with omitted capacity normalized to zero.

    Raises:
        ValueError: If endpoint assignment or capacity is dynamic or ambiguous.
        TypeError: If the capacity expression is not an integer literal.
    """
    sender_name = sender_expression.rsplit(".", maxsplit=1)[-1]
    receiver_name = receiver_expression.rsplit(".", maxsplit=1)[-1]
    matches = _stream_factory_calls(
        initializer,
        sender_name=sender_name,
        receiver_name=receiver_name,
    )
    if len(matches) != 1:
        raise ValueError("initializer must create the planned sender and receiver together")
    call = matches[0]
    if call.keywords or len(call.args) > 1:
        raise ValueError("private stream capacity must be one positional literal")
    if not call.args:
        return 0
    capacity = call.args[0]
    if isinstance(capacity, ast.Constant) and isinstance(capacity.value, int):
        capacity_value = capacity.value
    elif (
        isinstance(capacity, ast.UnaryOp)
        and isinstance(capacity.op, ast.USub)
        and isinstance(capacity.operand, ast.Constant)
        and isinstance(capacity.operand.value, int)
    ):
        capacity_value = -capacity.operand.value
    else:
        raise TypeError("private stream capacity must be a non-negative integer literal")
    if capacity_value < 0:
        raise ValueError("private stream capacity must be non-negative")
    return capacity_value


def _stream_factory_calls(
    initializer: ast.FunctionDef,
    *,
    sender_name: str,
    receiver_name: str,
) -> list[ast.Call]:
    matches: list[ast.Call] = []
    for node in ast.walk(initializer):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Tuple) or len(target.elts) != _PAIR_SIZE:
            continue
        attributes = tuple(_expression_path(element) for element in target.elts)
        if attributes != (f"self.{sender_name}", f"self.{receiver_name}"):
            continue
        if isinstance(node.value, ast.Call):
            matches.append(node.value)
    return matches


def _spawn_shape(
    owner: ast.FunctionDef,
    worker_symbol: SymbolId,
) -> tuple[str, str, str, str, str]:
    if len(owner.args.args) != _PAIR_SIZE:
        raise ValueError("spawn owner must accept only self and one request sequence")
    request_name = owner.args.args[1].arg
    loop, call = _find_spawn_loop(owner)
    if not isinstance(loop.target, ast.Name) or not isinstance(loop.iter, ast.Name):
        raise TypeError("spawn loop must use one named item and request sequence")
    if loop.iter.id != request_name or len(call.args) != _PAIR_SIZE or call.keywords:
        raise ValueError("spawn loop must schedule exactly one worker and one item")
    worker_path = _expression_path(call.args[0])
    expected_worker = f"self.{worker_symbol.qualname.rsplit('.', maxsplit=1)[-1]}"
    if worker_path != expected_worker or not isinstance(call.args[1], ast.Name):
        raise ValueError("spawn loop must schedule the planned same-class worker")
    if call.args[1].id != loop.target.id:
        raise ValueError("spawned work item must be the loop target")
    task_group_path = _expression_path(call.func)
    if task_group_path is None:
        raise ValueError("spawn scheduler expression is unavailable")
    task_group_expression, task_group_method = task_group_path.rsplit(".", maxsplit=1)
    return (
        request_name,
        loop.target.id,
        task_group_expression,
        task_group_method,
        worker_symbol.qualname.rsplit(".", maxsplit=1)[-1],
    )


def _find_spawn_loop(owner: ast.FunctionDef) -> tuple[ast.For, ast.Call]:
    """Find the sole simple ``start_soon`` loop in a spawn owner.

    Args:
        owner: Synchronous owner method selected by the execution plan.

    Returns:
        tuple[ast.For, ast.Call]: Spawn loop and its scheduler call.

    Raises:
        ValueError: If the owner has zero or multiple candidate spawn loops.
    """
    candidates: list[tuple[ast.For, ast.Call]] = []
    for node in ast.walk(owner):
        if not isinstance(node, ast.For) or len(node.body) != 1 or node.orelse:
            continue
        statement = node.body[0]
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            continue
        path = _expression_path(statement.value.func)
        if path is not None and path.endswith(".start_soon"):
            candidates.append((node, statement.value))
    if len(candidates) != 1:
        raise ValueError("spawn owner must contain one simple TaskGroup.start_soon loop")
    return candidates[0]


def _worker_delegate(
    worker: ast.AsyncFunctionDef,
    *,
    sender_expression: str,
) -> tuple[str, str, str]:
    if len(worker.args.args) != _PAIR_SIZE:
        raise ValueError("producer worker must accept only self and one work item")
    task_name = worker.args.args[1].arg
    delegates: list[tuple[str, str]] = []
    for node in ast.walk(worker):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        value = node.value
        if not (
            isinstance(target, ast.Name)
            and isinstance(value, ast.Await)
            and isinstance(value.value, ast.Call)
            and len(value.value.args) == 1
            and not value.value.keywords
            and isinstance(value.value.args[0], ast.Name)
            and value.value.args[0].id == task_name
        ):
            continue
        path = _expression_path(value.value.func)
        if path is not None and re.fullmatch(r"self\.[A-Za-z_]\w*", path):
            delegates.append((path.rsplit(".", maxsplit=1)[1], target.id))
    if len(delegates) != 1:
        raise ValueError("producer must delegate once to one same-class run coroutine")
    send_path = f"{sender_expression}.send"
    if not any(
        isinstance(node, ast.Await)
        and isinstance(node.value, ast.Call)
        and _expression_path(node.value.func) == send_path
        for node in ast.walk(worker)
    ):
        raise ValueError("producer does not send records through the planned private sender")
    run_method, result_name = delegates[0]
    return run_method, task_name, result_name


def _run_shape(
    class_node: ast.ClassDef,
    run_method: ast.AsyncFunctionDef,
    task_name: str,
    *,
    async_result_type: str | None,
) -> tuple[str, str, str, str, str, tuple[str, ...]]:
    if any(
        isinstance(node, ast.Global | ast.Nonlocal | ast.Yield | ast.YieldFrom)
        for node in ast.walk(run_method)
    ):
        raise ValueError("run coroutine contains global, closure, or generator state")
    awaits = [node for node in ast.walk(run_method) if isinstance(node, ast.Await)]
    if len(awaits) != 1 or not isinstance(awaits[0].value, ast.Call):
        raise ValueError("run coroutine must contain exactly one awaited callable branch")
    awaited_call = awaits[0].value
    if not isinstance(awaited_call.func, ast.Attribute) or not isinstance(
        awaited_call.func.value, ast.Name
    ):
        raise TypeError("run coroutine await must target one loaded node attribute")
    node_name = awaited_call.func.value.id
    node_assignment = next(
        (
            node
            for node in run_method.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == node_name
            and isinstance(node.value, ast.Subscript)
        ),
        None,
    )
    if node_assignment is None:
        raise ValueError("run coroutine must load its node from one indexed mapping")
    aliases = _simple_aliases_before(run_method.body, node_assignment)
    node_lookup = _substitute_aliases(node_assignment.value, aliases)
    if not isinstance(node_lookup, ast.Subscript):
        raise TypeError("run node lookup could not be normalized")
    if not _expression_mentions_name(node_lookup.slice, task_name):
        raise ValueError("run node key must derive from the work item")
    condition = _awaited_type_condition(run_method, awaits[0], node_name)
    task_annotation = (
        run_method.args.args[1].annotation if len(run_method.args.args) == _PAIR_SIZE else None
    )
    if task_annotation is None:
        raise ValueError("run work item requires an explicit nominal annotation")
    excluded_node_types = _async_protocol_node_types(
        class_node,
        run_method,
        node_name=node_name,
        async_result_type=async_result_type,
    )
    return (
        ast.unparse(task_annotation),
        ast.unparse(node_lookup.value),
        ast.unparse(node_lookup.slice),
        ast.unparse(condition),
        awaited_call.func.attr,
        excluded_node_types,
    )


def _simple_aliases_before(
    statements: list[ast.stmt],
    boundary: ast.Assign,
) -> dict[str, ast.expr]:
    aliases: dict[str, ast.expr] = {}
    for statement in statements:
        if statement is boundary:
            break
        if not (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            continue
        aliases[statement.targets[0].id] = _substitute_aliases(statement.value, aliases)
    return aliases


def _async_protocol_node_types(
    class_node: ast.ClassDef,
    run_method: ast.AsyncFunctionDef,
    *,
    node_name: str,
    async_result_type: str | None,
) -> tuple[str, ...]:
    """Find run-method node branches that can produce an async result protocol.

    Args:
        class_node: Owner class containing the run method and its direct helpers.
        run_method: One-await run coroutine selected by the source plan.
        node_name: Local node variable used by branch ``isinstance`` tests.
        async_result_type: Producer wrapper type handled by ``async for``.

    Returns:
        tuple[str, ...]: Nominal node-type expressions that must retain task fallback.
    """
    if async_result_type is None:
        return ()
    methods = {
        method.name: method
        for method in class_node.body
        if isinstance(method, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    excluded: set[str] = set()
    for branch in (node for node in ast.walk(run_method) if isinstance(node, ast.If)):
        branch_type = _isinstance_type(branch.test, node_name)
        if branch_type is None:
            continue
        helper_names = _returned_self_helpers(branch.body)
        for helper_name in helper_names:
            helper = methods.get(helper_name)
            if helper is None or not _annotation_mentions(helper.returns, async_result_type):
                continue
            narrowed = _narrow_async_helper_types(
                helper,
                methods=methods,
                async_result_type=async_result_type,
            )
            excluded.update(narrowed or (ast.unparse(branch_type),))
    return tuple(sorted(excluded))


def _narrow_async_helper_types(
    helper: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
    async_result_type: str,
) -> tuple[str, ...]:
    if len(helper.args.args) < _PAIR_SIZE:
        return ()
    node_parameter = helper.args.args[1].arg
    narrowed: set[str] = set()
    for branch in (node for node in ast.walk(helper) if isinstance(node, ast.If)):
        branch_type = _isinstance_type(branch.test, node_parameter)
        if branch_type is None:
            continue
        for helper_name in _returned_self_helpers(branch.body):
            nested = methods.get(helper_name)
            if nested is not None and _annotation_mentions(nested.returns, async_result_type):
                narrowed.add(ast.unparse(branch_type))
    return tuple(sorted(narrowed))


def _isinstance_type(test: ast.expr, variable_name: str) -> ast.expr | None:
    for call in ast.walk(test):
        if not (
            isinstance(call, ast.Call)
            and _expression_path(call.func) == "isinstance"
            and len(call.args) == _PAIR_SIZE
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id == variable_name
        ):
            continue
        return call.args[1]
    return None


def _returned_self_helpers(statements: list[ast.stmt]) -> tuple[str, ...]:
    helpers: set[str] = set()
    for statement in statements:
        for node in ast.walk(statement):
            if not (
                isinstance(node, ast.Return)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == "self"
            ):
                continue
            helpers.add(node.value.func.attr)
    return tuple(sorted(helpers))


def _annotation_mentions(annotation: ast.expr | None, type_name: str) -> bool:
    return annotation is not None and type_name in ast.unparse(annotation)


class _AliasSubstitution(ast.NodeTransformer):
    def __init__(self, aliases: dict[str, ast.expr]) -> None:
        self._aliases = aliases

    @override
    def visit_Name(self, node: ast.Name) -> ast.expr:
        replacement = self._aliases.get(node.id)
        return copy.deepcopy(replacement) if replacement is not None else node


def _substitute_aliases(node: ast.expr, aliases: dict[str, ast.expr]) -> ast.expr:
    substituted = _AliasSubstitution(aliases).visit(copy.deepcopy(node))
    return cast(ast.expr, ast.fix_missing_locations(substituted))


def _expression_mentions_name(node: ast.expr, name: str) -> bool:
    return any(isinstance(child, ast.Name) and child.id == name for child in ast.walk(node))


def _awaited_type_condition(
    run_method: ast.AsyncFunctionDef,
    awaited: ast.Await,
    node_name: str,
) -> ast.expr:
    candidates: list[ast.expr] = []
    for node in ast.walk(run_method):
        if not isinstance(node, ast.If) or not _contains(node, awaited):
            continue
        for call in ast.walk(node.test):
            if not (
                isinstance(call, ast.Call)
                and _expression_path(call.func) == "isinstance"
                and len(call.args) == _PAIR_SIZE
                and not call.keywords
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id == node_name
            ):
                continue
            candidates.append(call.args[1])
    if not candidates:
        raise ValueError("awaited node branch requires an explicit isinstance guard")
    return candidates[-1]


def _contains(parent: ast.AST, child: ast.AST) -> bool:
    return any(node is child for node in ast.walk(parent))


def _result_shape(
    worker: ast.AsyncFunctionDef,
    *,
    sender_expression: str,
    task_name: str,
    result_name: str,
) -> tuple[str, str, str, str | None]:
    records: list[ast.Call] = []
    send_path = f"{sender_expression}.send"
    for node in ast.walk(worker):
        if not (
            isinstance(node, ast.Await)
            and isinstance(node.value, ast.Call)
            and _expression_path(node.value.func) == send_path
            and len(node.value.args) == 1
            and not node.value.keywords
            and isinstance(node.value.args[0], ast.Call)
        ):
            continue
        record = node.value.args[0]
        if record.args and isinstance(record.args[0], ast.Name) and record.args[0].id == task_name:
            records.append(record)
    if not records:
        raise ValueError("producer must send constructed result records")
    constructors = {_expression_path(record.func) for record in records}
    if None in constructors or len(constructors) != 1:
        raise ValueError("producer result record constructor must be statically stable")
    success = next(
        (
            record
            for record in records
            if len(record.args) >= _PAIR_SIZE
            and isinstance(record.args[1], ast.Name)
            and record.args[1].id == result_name
            and not record.keywords
        ),
        None,
    )
    error = next((record for record in records if record.keywords), None)
    if success is None or error is None or len(error.args) < _PAIR_SIZE or len(error.keywords) != 1:
        raise ValueError("producer must expose ordinary success and exception result records")
    keyword = error.keywords[0]
    if keyword.arg is None:
        raise ValueError("producer exception record cannot use keyword expansion")
    async_result_type = _async_result_wrapper(worker, result_name)
    constructor = constructors.pop()
    return cast(str, constructor), ast.unparse(error.args[1]), keyword.arg, async_result_type


def _async_result_wrapper(worker: ast.AsyncFunctionDef, result_name: str) -> str | None:
    for node in ast.walk(worker):
        if not isinstance(node, ast.If):
            continue
        for call in ast.walk(node.test):
            if (
                isinstance(call, ast.Call)
                and _expression_path(call.func) == "isinstance"
                and len(call.args) == _PAIR_SIZE
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id == result_name
            ):
                return ast.unparse(call.args[1])
    return None


def _validate_worker_wrapper(
    worker: ast.AsyncFunctionDef,
    *,
    run_method_name: str,
    sender_expression: str,
    result_constructor: str,
    async_result_type: str | None,
) -> None:
    """Prove that bypassing a producer wrapper omits no arbitrary behavior.

    Args:
        worker: Producer wrapper selected by the execution plan.
        run_method_name: Same-class coroutine delegated to for actual work.
        sender_expression: Exact private result sender.
        result_constructor: Stable result-record constructor.
        async_result_type: Optional producer branch retained through fallback.

    Raises:
        ValueError: If the wrapper contains nested state, arbitrary calls, or
            awaits outside delegated work and private result publication.
    """
    if any(
        isinstance(
            node,
            ast.Global
            | ast.Nonlocal
            | ast.FunctionDef
            | ast.AsyncFunctionDef
            | ast.Lambda
            | ast.Yield
            | ast.YieldFrom,
        )
        and node is not worker
        for node in ast.walk(worker)
    ):
        raise ValueError("producer wrapper contains nested or nonlocal execution state")
    context_calls = {
        _expression_path(item.context_expr.func)
        for node in ast.walk(worker)
        if isinstance(node, ast.With)
        for item in node.items
        if isinstance(item.context_expr, ast.Call)
    }
    allowed_calls = {
        f"self.{run_method_name}",
        f"{sender_expression}.send",
        result_constructor,
        "isinstance",
        *context_calls,
    }
    calls = tuple(
        _expression_path(node.func) for node in ast.walk(worker) if isinstance(node, ast.Call)
    )
    if any(path is None or path not in allowed_calls for path in calls):
        raise ValueError(
            "producer wrapper contains behavior outside run, publication, and scope setup"
        )
    allowed_awaits = {f"self.{run_method_name}", f"{sender_expression}.send"}
    if any(
        not isinstance(node.value, ast.Call)
        or _expression_path(node.value.func) not in allowed_awaits
        for node in ast.walk(worker)
        if isinstance(node, ast.Await)
    ):
        raise ValueError("producer wrapper awaits behavior outside run and private publication")
    if async_result_type is not None and not any(
        isinstance(node, ast.AsyncFor) for node in ast.walk(worker)
    ):
        raise ValueError("producer async result type has no matching async-for fallback")


def _validate_consumer(consumer: ast.AsyncFunctionDef, receiver_expression: str) -> None:
    matches = [
        node
        for node in ast.walk(consumer)
        if isinstance(node, ast.AsyncFor) and _expression_path(node.iter) == receiver_expression
    ]
    if len(matches) != 1 or matches[0].orelse:
        raise ValueError("consumer must contain one private receiver async-for without else")
    if not isinstance(matches[0].target, ast.Name):
        raise TypeError("consumer result loop must bind one named record")


def _reducer_shape(
    project_root: Path,
    owner_tree: ast.Module,
    plan: SourceOptimizationPlan,
    consumer: ast.AsyncFunctionDef,
) -> _ReducerShape | None:
    """Recognize a reducer whose repeated signature reflection can be cached.

    Args:
        project_root: Target source root used to verify the reducer module hash.
        owner_tree: Module containing the transformed consumer and reducer call site.
        plan: Source plan containing the optional reducer symbol and source hashes.
        consumer: Consumer method whose exact reducer call will be rewritten.

    Returns:
        _ReducerShape | None: Proven reducer shape, or ``None`` when the plan has
        no reducer.

    Raises:
        ValueError: If a reported reducer exists but its source, property,
            reflection branch, or consumer call cannot be proven exactly.
    """
    if plan.reducer is None:
        return None
    if "." not in plan.reducer.qualname:
        raise ValueError("source reducer must be a direct class method")
    class_name, method_name = plan.reducer.qualname.rsplit(".", maxsplit=1)
    if "." in class_name:
        raise ValueError("nested reducer classes are not supported")
    reducer_path, expected_hash = _module_source_identity(plan, plan.reducer.module)
    source_path = _source_path(project_root, reducer_path)
    source = source_path.read_text(encoding="utf-8")
    if _sha256(source) != expected_hash:
        raise ValueError(f"stale reducer source for {reducer_path.as_posix()}")
    reducer_tree = ast.parse(source, filename=str(source_path))
    reducer_class = _class_node(reducer_tree, class_name)
    _validate_class(reducer_class)
    reducer_method = cast(
        ast.FunctionDef,
        _named_method(reducer_class, method_name, ast.FunctionDef, role="reducer"),
    )
    callable_property = _reflected_callable_property(reducer_method)
    _validate_callable_property(reducer_class, callable_property)
    owner_expression, class_expression = _consumer_reducer_call(
        owner_tree,
        consumer,
        method_name=method_name,
        class_name=class_name,
    )
    return _ReducerShape(
        class_expression=class_expression,
        owner_expression=owner_expression,
        method_name=method_name,
        callable_property=callable_property,
    )


def _module_source_identity(
    plan: SourceOptimizationPlan,
    module_name: str,
) -> tuple[PurePosixPath, str]:
    suffix = PurePosixPath(*module_name.split(".")).with_suffix(".py")
    matches = tuple(
        (path, source_hash)
        for path, source_hash in plan.identity.source_hashes
        if path == suffix or path.as_posix().endswith(f"/{suffix.as_posix()}")
    )
    if len(matches) != 1:
        raise ValueError(f"source plan has no exact hash for reducer module {module_name}")
    return matches[0]


def _reflected_callable_property(method: ast.FunctionDef) -> str:
    if len(method.args.args) != _REDUCER_ARGUMENT_COUNT:
        raise ValueError("reducer method must accept self, context, current, and input")
    if any(
        isinstance(node, ast.Global | ast.Nonlocal | ast.Await | ast.Yield | ast.YieldFrom)
        for node in ast.walk(method)
    ):
        raise ValueError("reducer method contains unsupported execution state")
    signature_calls = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and _expression_path(node.func) == "inspect.signature"
        and len(node.args) == 1
        and not node.keywords
        and isinstance(node.args[0], ast.Attribute)
        and isinstance(node.args[0].value, ast.Name)
        and node.args[0].value.id == "self"
    ]
    if len(signature_calls) != 1:
        raise ValueError("reducer must reflect one direct callable property")
    property_name = cast(ast.Attribute, signature_calls[0].args[0]).attr
    property_reads = sum(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr == property_name
        for node in ast.walk(method)
    )
    if property_reads < _REDUCER_PROPERTY_READ_COUNT:
        raise ValueError("reducer callable property is not used in both arity branches")
    comparisons = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Eq)
        and len(node.comparators) == 1
        and isinstance(node.comparators[0], ast.Constant)
        and node.comparators[0].value == _PAIR_SIZE
    ]
    if len(comparisons) != 1:
        raise ValueError("reducer must choose a two-argument or contextual callable")
    returns = [node for node in ast.walk(method) if isinstance(node, ast.Return)]
    if len(returns) != _PAIR_SIZE:
        raise ValueError("reducer must contain exactly two return branches")
    return property_name


def _validate_callable_property(class_node: ast.ClassDef, property_name: str) -> None:
    matches = [
        statement
        for statement in class_node.body
        if isinstance(statement, ast.FunctionDef)
        and statement.name == property_name
        and len(statement.decorator_list) == 1
        and _expression_path(statement.decorator_list[0]) == "property"
    ]
    if len(matches) != 1:
        raise ValueError("reflected reducer callable must use one read-only property")
    method = matches[0]
    if len(method.body) != 1 or not isinstance(method.body[0], ast.Return):
        raise ValueError("reducer callable property must return one stored value")
    value = method.body[0].value
    if not (
        isinstance(value, ast.Attribute)
        and isinstance(value.value, ast.Name)
        and value.value.id == "self"
    ):
        raise ValueError("reducer callable property must return a direct instance field")


def _consumer_reducer_call(
    owner_tree: ast.Module,
    consumer: ast.AsyncFunctionDef,
    *,
    method_name: str,
    class_name: str,
) -> tuple[str, str]:
    calls = [
        node
        for node in ast.walk(consumer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == method_name
        and len(node.args) == _CONSUMER_REDUCER_ARGUMENT_COUNT
        and not node.keywords
        and _expression_path(node.func.value) is not None
    ]
    if len(calls) != 1:
        raise ValueError("consumer must call the planned reducer method exactly once")
    reducer_call = cast(ast.Attribute, calls[0].func)
    owner_expression = cast(str, _expression_path(reducer_call.value))
    class_expressions: list[str] = []
    for node in ast.walk(consumer):
        if not isinstance(node, ast.Call) or _expression_path(node.func) != "isinstance":
            continue
        if len(node.args) != _PAIR_SIZE or _expression_path(node.args[0]) != owner_expression:
            continue
        expression = _expression_path(node.args[1])
        if expression is not None and expression.rsplit(".", maxsplit=1)[-1] == class_name:
            class_expressions.append(expression)
    class_expressions = sorted(set(class_expressions))
    if len(class_expressions) != 1:
        raise ValueError("consumer reducer owner requires one exact isinstance proof")
    if not any(
        isinstance(node, ast.ImportFrom)
        and any((alias.asname or alias.name) == class_expressions[0] for alias in node.names)
        for node in owner_tree.body
    ):
        raise ValueError("consumer module must import the reducer class explicitly")
    return owner_expression, class_expressions[0]


def _protocol_shape(
    tree: ast.Module,
    owner_class: str,
    consumer_name: str,
) -> _ProtocolShape | None:
    """Recognize a private run-to-completion loop that only echoes iterator events.

    Args:
        tree: Owner module containing the scheduler, iterator wrapper, and entrypoint.
        owner_class: Class whose consumer yields private pipeline events.
        consumer_name: Exact consumer method captured by runtime identity guards.

    Returns:
        _ProtocolShape | None: Exact echo protocol and iterator ownership, or
        ``None`` when no unique protocol is present.
    """
    del consumer_name
    candidates: list[tuple[str, ast.AsyncFunctionDef, str, str, str]] = []
    for class_node in (node for node in tree.body if isinstance(node, ast.ClassDef)):
        for method in (node for node in class_node.body if isinstance(node, ast.AsyncFunctionDef)):
            candidate = _echo_protocol_candidate(method)
            if candidate is not None:
                runner_name, next_method, terminal_type = candidate
                candidates.append(
                    (class_node.name, method, runner_name, next_method, terminal_type)
                )
    if len(candidates) != 1:
        return None
    entry_class, entry_method, runner_name, next_method, terminal_type = candidates[0]
    runner_matches: list[tuple[str, str]] = []
    for class_node in (node for node in tree.body if isinstance(node, ast.ClassDef)):
        if not any(
            isinstance(node, ast.AsyncFunctionDef) and node.name == next_method
            for node in class_node.body
        ):
            continue
        initializer = next(
            (
                node
                for node in class_node.body
                if isinstance(node, ast.FunctionDef) and node.name == "__init__"
            ),
            None,
        )
        if initializer is None:
            continue
        owner_attributes = _constructed_owner_attributes(initializer, owner_class)
        runner_matches.extend((class_node.name, attribute) for attribute in owner_attributes)
    if len(runner_matches) != 1:
        return None
    runner_class, owner_attribute = runner_matches[0]
    return _ProtocolShape(
        entry_class=entry_class,
        entry_method=entry_method.name,
        runner_class=runner_class,
        runner_name=runner_name,
        next_method=next_method,
        owner_attribute=owner_attribute,
        terminal_type=terminal_type,
    )


def _echo_protocol_candidate(
    method: ast.AsyncFunctionDef,
) -> tuple[str, str, str] | None:
    if method.decorator_list or any(isinstance(node, ast.Yield) for node in ast.walk(method)):
        return None
    bound_runners = {
        item.optional_vars.id
        for node in ast.walk(method)
        if isinstance(node, ast.AsyncWith)
        for item in node.items
        if isinstance(item.optional_vars, ast.Name)
    }
    calls: list[tuple[str, str, str]] = []
    for node in ast.walk(method):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Await)
            and isinstance(node.value.value, ast.Call)
            and isinstance(node.value.value.func, ast.Attribute)
            and isinstance(node.value.value.func.value, ast.Name)
        ):
            continue
        call = node.value.value
        runner_value = cast(ast.Attribute, call.func).value
        runner_name = cast(ast.Name, runner_value).id
        event_name = node.targets[0].id
        if runner_name not in bound_runners or call.keywords:
            continue
        if len(call.args) != 1 or not isinstance(call.args[0], ast.Name):
            continue
        if call.args[0].id != event_name:
            continue
        calls.append((runner_name, cast(ast.Attribute, call.func).attr, event_name))
    if len(calls) != 1:
        return None
    runner_name, next_method, event_name = calls[0]
    initializes_none = any(
        (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == event_name
            and isinstance(node.value, ast.Constant)
            and node.value.value is None
        )
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == event_name for target in node.targets
            )
            and isinstance(node.value, ast.Constant)
            and node.value.value is None
        )
        for node in ast.walk(method)
    )
    if not initializes_none:
        return None
    terminal_types = {
        expression
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and _expression_path(node.func) == "isinstance"
        and len(node.args) == _PAIR_SIZE
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == event_name
        and (expression := _expression_path(node.args[1])) is not None
    }
    catches_stop = any(
        isinstance(node, ast.ExceptHandler)
        and node.type is not None
        and _expression_path(node.type) == "StopAsyncIteration"
        and any(
            isinstance(child, ast.Return)
            and child.value is not None
            and _expression_mentions_name(child.value, event_name)
            for child in ast.walk(node)
        )
        for node in ast.walk(method)
    )
    if len(terminal_types) != 1 or not catches_stop:
        return None
    return runner_name, next_method, terminal_types.pop()


def _constructed_owner_attributes(
    initializer: ast.FunctionDef,
    owner_class: str,
) -> tuple[str, ...]:
    attributes: list[str] = []
    for node in ast.walk(initializer):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Attribute)
            and isinstance(node.targets[0].value, ast.Name)
            and node.targets[0].value.id == "self"
            and isinstance(node.value, ast.Call)
        ):
            continue
        constructor = _subscript_base_path(node.value.func)
        if constructor is not None and constructor.rsplit(".", maxsplit=1)[-1] == owner_class:
            attributes.append(node.targets[0].attr)
    return tuple(attributes)


def _subscript_base_path(node: ast.expr) -> str | None:
    expression = node.value if isinstance(node, ast.Subscript) else node
    return _expression_path(expression)


def _owner_body(
    module: cst.Module,
    qualname: str,
    shape: _AnyioShape,
    names: _Names,
) -> str:
    method = _cst_method(module, qualname)
    body = cast(cst.IndentedBlock, method.body)
    spawn_indexes = [
        index
        for index, statement in enumerate(body.body)
        if isinstance(statement, cst.For) and _cst_spawn_loop(statement, shape)
    ]
    if len(spawn_indexes) != 1:
        raise ValueError("LibCST could not resolve the planned spawn loop")
    index = spawn_indexes[0]
    spawn_loop = cast(cst.For, body.body[index])
    fast_call = cst.parse_statement(f"{names.drive}(self, {shape.spawn_item_name})\n")
    fast_loop = spawn_loop.with_changes(
        body=cst.IndentedBlock(body=(fast_call,)),
        leading_lines=(),
    )
    fallback_loop = spawn_loop.with_changes(leading_lines=())
    branch = cst.If(
        test=cst.Name(names.fast_local),
        body=cst.IndentedBlock(body=(fast_loop,)),
        orelse=cst.Else(body=cst.IndentedBlock(body=(fallback_loop,))),
        leading_lines=spawn_loop.leading_lines,
    )
    guard = cst.parse_statement(f"{names.fast_local} = {names.guard}(self, {shape.request_name})\n")
    insertion = _after_docstring(body.body)
    statements = list(body.body)
    statements[index] = branch
    statements.insert(insertion, guard)
    return _body_source(module, body.with_changes(body=tuple(statements)))


def _cst_spawn_loop(node: cst.For, shape: _AnyioShape) -> bool:
    if not isinstance(node.target, cst.Name) or node.target.value != shape.spawn_item_name:
        return False
    if not isinstance(node.iter, cst.Name) or node.iter.value != shape.request_name:
        return False
    calls = [child for child in _cst_nodes(node.body) if isinstance(child, cst.Call)]
    scheduler = f"{shape.task_group_expression}.{shape.task_group_method}"
    return any(_cst_path(call.func) == scheduler for call in calls)


def _consumer_body(
    module: cst.Module,
    symbol: SymbolId | None,
    shape: _AnyioShape,
    names: _Names,
) -> str:
    if symbol is None:
        raise ValueError("AnyIO stream consumer is unavailable")
    method = _cst_method(module, symbol.qualname)
    body = cast(cst.IndentedBlock, method.body)
    transformer = _ConsumerBodyTransformer(
        shape.receiver_expression,
        shape.reducer,
        shape.protocol,
        names,
    )
    updated = cast(cst.IndentedBlock, body.visit(transformer))
    if transformer.receiver_replacements != 1:
        raise ValueError("LibCST could not resolve exactly one receiver async-for")
    if shape.reducer is not None and transformer.reducer_replacements != 1:
        raise ValueError("LibCST could not resolve exactly one planned reducer call")
    if shape.protocol is not None and transformer.protocol_yield_replacements < 1:
        raise ValueError("LibCST could not resolve private protocol yield assignments")
    return _body_source(module, updated)


class _ConsumerBodyTransformer(cst.CSTTransformer):
    """Replace a private receiver loop and lazily guard reducer scans."""

    def __init__(
        self,
        receiver_expression: str,
        reducer: _ReducerShape | None,
        protocol: _ProtocolShape | None,
        names: _Names,
    ) -> None:
        self._receiver_expression = receiver_expression
        self._reducer = reducer
        self._protocol = protocol
        self._names = names
        self.receiver_replacements = 0
        self.reducer_replacements = 0
        self.protocol_yield_replacements = 0

    @override
    def leave_SimpleStatementLine(
        self,
        original_node: cst.SimpleStatementLine,
        updated_node: cst.SimpleStatementLine,
    ) -> cst.BaseStatement:
        """Keep public yields while auto-forwarding private echo-protocol values.

        Args:
            original_node: Original statement used to recognize an assignment yield.
            updated_node: Statement after nested expression transformations.

        Returns:
            cst.BaseStatement: Original statement or guarded fast/fallback branch.
        """
        if self._protocol is None or len(original_node.body) != 1:
            return updated_node
        original = original_node.body[0]
        updated = updated_node.body[0]
        if not (
            isinstance(original, cst.Assign)
            and isinstance(original.value, cst.Yield)
            and isinstance(original.value.value, cst.BaseExpression)
            and isinstance(updated, cst.Assign)
            and isinstance(updated.value, cst.Yield)
            and isinstance(updated.value.value, cst.BaseExpression)
        ):
            return updated_node
        value = updated.value.value
        fast_assignment = updated.with_changes(value=value)
        condition = cst.Call(
            func=cst.Name(self._names.protocol_forward),
            args=(cst.Arg(cst.Name("self")), cst.Arg(value)),
        )
        self.protocol_yield_replacements += 1
        return cst.If(
            test=condition,
            body=cst.IndentedBlock(
                body=(updated_node.with_changes(body=(fast_assignment,), leading_lines=()),)
            ),
            orelse=cst.Else(
                body=cst.IndentedBlock(body=(updated_node.with_changes(leading_lines=()),))
            ),
            leading_lines=updated_node.leading_lines,
        )

    @override
    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
        """Route one exact reducer method call through the guarded reflection cache.

        Args:
            original_node: Original call used for exact structural matching.
            updated_node: Call after child transformations.

        Returns:
            cst.Call: Original call or generated reducer helper invocation.
        """
        reducer = self._reducer
        if reducer is None:
            return updated_node
        expected = f"{reducer.owner_expression}.{reducer.method_name}"
        if _cst_path(original_node.func) != expected or not isinstance(
            updated_node.func, cst.Attribute
        ):
            return updated_node
        self.reducer_replacements += 1
        return updated_node.with_changes(
            func=cst.Name(self._names.reducer),
            args=(cst.Arg(updated_node.func.value), *updated_node.args),
        )

    @override
    def leave_For(self, original_node: cst.For, updated_node: cst.For) -> cst.BaseStatement:
        """Replace the exact async receiver iterator with deque-first delivery.

        Args:
            original_node: Original loop used for exact receiver matching.
            updated_node: Loop after child transformations.

        Returns:
            cst.BaseStatement: Original loop or equivalent deque-first while loop.
        """
        if (
            original_node.asynchronous is None
            or _cst_path(original_node.iter) != self._receiver_expression
        ):
            return updated_node
        if (
            not isinstance(original_node.target, cst.Name)
            or original_node.orelse is not None
            or not isinstance(updated_node.body, cst.IndentedBlock)
        ):
            return updated_node
        self.receiver_replacements += 1
        target = original_node.target.value
        receive = (
            "try:\n"
            f"    {target} = self.{self._names.deque_attribute}.popleft()\n"
            "except IndexError:\n"
            "    try:\n"
            f"        {target} = {self._receiver_expression}.receive_nowait()\n"
            f"    except {self.anyio_would_block}:\n"
            "        try:\n"
            f"            {target} = await {self._receiver_expression}.receive()\n"
            f"        except {self.anyio_end_of_stream}:\n"
            "            break\n"
        )
        try_statement = cast(cst.BaseStatement, cst.parse_statement(receive))
        return cst.While(
            test=cst.Name("True"),
            body=cst.IndentedBlock(body=(try_statement, *updated_node.body.body)),
            leading_lines=updated_node.leading_lines,
        )

    @override
    def leave_IndentedBlock(
        self,
        original_node: cst.IndentedBlock,
        updated_node: cst.IndentedBlock,
    ) -> cst.IndentedBlock:
        """Move task snapshots behind a structurally discovered reducer guard.

        Args:
            original_node: Original block, unused because children are already updated.
            updated_node: Block whose adjacent snapshot/list/scan statements may match.

        Returns:
            cst.IndentedBlock: Block with safe lazy reducer scans when present.
        """
        del original_node
        statements = list(updated_node.body)
        output: list[cst.BaseStatement] = []
        index = 0
        while index < len(statements):
            match = _lazy_scan_match(statements, index)
            if match is None:
                output.append(statements[index])
                index += 1
                continue
            snapshot, accumulator, scan, reducer_attribute = match
            guarded_snapshot = snapshot.with_changes(leading_lines=())
            guard = cst.If(
                test=cst.Attribute(cst.Name("self"), cst.Name(reducer_attribute)),
                body=cst.IndentedBlock(body=(guarded_snapshot, scan)),
                leading_lines=snapshot.leading_lines,
            )
            output.extend((accumulator, guard))
            index += 3
        return updated_node.with_changes(body=tuple(output))

    @property
    def anyio_would_block(self) -> str:
        """Return generated AnyIO WouldBlock reference.

        Returns:
            str: Module alias and exception attribute.
        """
        return f"{self._names.typing}_anyio.WouldBlock"

    @property
    def anyio_end_of_stream(self) -> str:
        """Return generated AnyIO EndOfStream reference.

        Returns:
            str: Module alias and exception attribute.
        """
        return f"{self._names.typing}_anyio.EndOfStream"


def _lazy_scan_match(
    statements: list[cst.BaseStatement],
    index: int,
) -> tuple[cst.SimpleStatementLine, cst.SimpleStatementLine, cst.For, str] | None:
    if index + 2 >= len(statements):
        return None
    snapshot = statements[index]
    accumulator = statements[index + 1]
    scan = statements[index + 2]
    snapshot_name = _snapshot_assignment(snapshot)
    if (
        snapshot_name is None
        or not _empty_list_assignment(accumulator)
        or not isinstance(scan, cst.For)
    ):
        return None
    if scan.asynchronous is not None or not _call_uses_name(scan.iter, snapshot_name):
        return None
    reducer_attributes = {
        path.split(".")[1]
        for node in _cst_nodes(scan.body)
        if isinstance(node, cst.Call)
        and (path := _cst_path(node.func)) is not None
        and re.fullmatch(r"self\.[A-Za-z_]\w*\.pop", path)
    }
    if len(reducer_attributes) != 1:
        return None
    return (
        cast(cst.SimpleStatementLine, snapshot),
        cast(cst.SimpleStatementLine, accumulator),
        scan,
        reducer_attributes.pop(),
    )


def _snapshot_assignment(statement: cst.BaseStatement) -> str | None:
    if not isinstance(statement, cst.SimpleStatementLine) or len(statement.body) != 1:
        return None
    assignment = statement.body[0]
    if not isinstance(assignment, cst.Assign) or len(assignment.targets) != 1:
        return None
    target = assignment.targets[0].target
    value = assignment.value
    if (
        not isinstance(target, cst.Name)
        or not isinstance(value, cst.Call)
        or _cst_path(value.func) != "list"
    ):
        return None
    if len(value.args) != 1 or not isinstance(value.args[0].value, cst.Call):
        return None
    values_call = value.args[0].value
    if not re.fullmatch(r"self\.[A-Za-z_]\w*\.values", _cst_path(values_call.func) or ""):
        return None
    return target.value


def _empty_list_assignment(statement: cst.BaseStatement) -> bool:
    if not isinstance(statement, cst.SimpleStatementLine) or len(statement.body) != 1:
        return False
    small = statement.body[0]
    value = small.value if isinstance(small, cst.Assign | cst.AnnAssign) else None
    return isinstance(value, cst.List) and not value.elements


def _call_uses_name(expression: cst.BaseExpression, name: str) -> bool:
    return any(isinstance(node, cst.Name) and node.value == name for node in _cst_nodes(expression))


def _protocol_body(
    module: cst.Module,
    protocol: _ProtocolShape,
    names: _Names,
) -> str:
    method = _cst_method(module, f"{protocol.entry_class}.{protocol.entry_method}")
    body = cast(cst.IndentedBlock, method.body)
    transformer = _ProtocolNextTransformer(protocol, names)
    updated = cast(cst.IndentedBlock, body.visit(transformer))
    if transformer.replacements != 1:
        raise ValueError("LibCST must replace one private protocol next call")
    return _body_source(module, updated)


class _ProtocolNextTransformer(cst.CSTTransformer):
    def __init__(self, protocol: _ProtocolShape, names: _Names) -> None:
        self._protocol = protocol
        self._names = names
        self.replacements = 0

    @override
    def leave_Await(self, original_node: cst.Await, updated_node: cst.Await) -> cst.Await:
        """Wrap the exact echo-loop next call with a plan-specific context token.

        Args:
            original_node: Original await expression used for exact call matching.
            updated_node: Await expression after child transformations.

        Returns:
            cst.Await: Original await or private protocol helper call.
        """
        if not isinstance(original_node.expression, cst.Call) or not isinstance(
            updated_node.expression, cst.Call
        ):
            return updated_node
        expected = f"{self._protocol.runner_name}.{self._protocol.next_method}"
        if _cst_path(original_node.expression.func) != expected:
            return updated_node
        self.replacements += 1
        return updated_node.with_changes(
            expression=cst.Call(
                func=cst.Name(self._names.protocol_next),
                args=(
                    cst.Arg(cst.Name("self")),
                    cst.Arg(cst.Name(self._protocol.runner_name)),
                    *updated_node.expression.args,
                ),
            )
        )


def _initializer_body(module: cst.Module, shape: _AnyioShape, names: _Names) -> str:
    method = _cst_method(module, f"{shape.class_name}.__post_init__")
    body = cast(cst.IndentedBlock, method.body)
    assignment = cst.parse_statement(
        f"self.{names.deque_attribute} = {names.collections}.deque()\n"
    )
    cache_assignment = cst.parse_statement(f"self.{names.eligibility_cache_attribute} = {{}}\n")
    return _body_source(
        module,
        body.with_changes(body=(*body.body, assignment, cache_assignment)),
    )


def _cst_method(module: cst.Module, qualname: str) -> cst.FunctionDef:
    collector = _MethodCollector(qualname)
    module.visit(collector)
    if len(collector.matches) != 1:
        raise ValueError(f"LibCST must resolve one method {qualname}")
    return collector.matches[0]


class _MethodCollector(cst.CSTVisitor):
    def __init__(self, target: str) -> None:
        self._target = target
        self._classes: list[str] = []
        self._function_depth = 0
        self.matches: list[cst.FunctionDef] = []

    @override
    def visit_ClassDef(self, node: cst.ClassDef) -> bool | None:
        """Enter class context used to form one-hop method qualnames.

        Args:
            node: Class definition being visited.

        Returns:
            bool | None: ``None`` so traversal continues.
        """
        self._classes.append(node.name.value)
        return None

    @override
    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        """Leave class context after visiting its declarations.

        Args:
            original_node: Class definition being left.
        """
        del original_node
        self._classes.pop()

    @override
    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        """Collect the exact top-level class method and skip nested ambiguity.

        Args:
            node: Function definition being visited.

        Returns:
            bool | None: ``None`` so nested syntax remains traversable.
        """
        if self._function_depth == 0 and len(self._classes) == 1:
            qualname = f"{self._classes[0]}.{node.name.value}"
            if qualname == self._target:
                self.matches.append(node)
        self._function_depth += 1
        return None

    @override
    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        """Leave function context.

        Args:
            original_node: Function definition being left.
        """
        del original_node
        self._function_depth -= 1


def _body_source(module: cst.Module, body: cst.IndentedBlock) -> str:
    return textwrap.dedent(module.code_for_node(body)).lstrip("\n")


def _after_docstring(statements: Sequence[cst.BaseStatement]) -> int:
    if not statements or not isinstance(statements[0], cst.SimpleStatementLine):
        return 0
    line = statements[0]
    if (
        len(line.body) == 1
        and isinstance(line.body[0], cst.Expr)
        and isinstance(line.body[0].value, cst.SimpleString)
    ):
        return 1
    return 0


def _helper_statements(shape: _AnyioShape, names: _Names) -> tuple[str, ...]:
    async_result_guard = (
        f"        if isinstance(completed_value, {shape.async_result_type}):\n"
        "            raise RuntimeError(\n"
        "                'optimized work produced an async result protocol'\n"
        "            )\n"
        if shape.async_result_type is not None
        else ""
    )
    reducer_support = _reducer_support(shape.reducer, names)
    protocol_support = _protocol_support(shape.protocol, names)
    synchronous_run = _synchronous_run_source(shape, names)
    excluded_node_guard = _excluded_node_guard(shape.excluded_node_types)
    support = f'''
import asyncio as {names.asyncio}
import collections as {names.collections}
import contextvars as {names.contextvars}
import dis as {names.dis}
import inspect as {names.inspect}
import os as {names.os}
import sys as {names.sys}
import typing as {names.typing}
import weakref as {names.weakref}
import anyio as {names.typing}_anyio
from anyio._backends import _asyncio as {names.anyio_backend}
from anyio.streams.memory import (
    MemoryObjectReceiveStream as {names.anyio_receive_stream},
    MemoryObjectSendStream as {names.anyio_send_stream},
)

{names.route_hits} = 0
{reducer_support}
{protocol_support}


def {names.safe_code}(function, seen):
    function = getattr(function, "__func__", function)
    code = getattr(function, "__code__", None)
    if code is None or code in seen:
        return code is not None
    seen.add(code)
    blocked_names = {{
        "ContextVar", "cancel", "cancelled", "cancelling", "checkpoint",
        "copy_context", "create_task", "current_task", "ensure_future",
        "get_event_loop", "get_running_loop", "reset", "set", "set_task_factory",
        "shield", "sleep", "start_soon", "uncancel",
    }}
    blocked_opcodes = {{
        "DELETE_DEREF", "DELETE_GLOBAL", "IMPORT_NAME", "SEND", "STORE_DEREF",
        "STORE_GLOBAL", "YIELD_FROM", "YIELD_VALUE",
    }}
    if blocked_names.intersection(code.co_names):
        return False
    if any(
        instruction.opname in blocked_opcodes
        for instruction in {names.dis}.get_instructions(code)
    ):
        return False
    globals_map = getattr(function, "__globals__", {{}})
    for referenced_name in code.co_names:
        referenced = globals_map.get(referenced_name)
        if {names.inspect}.isfunction(referenced) and not {names.safe_code}(referenced, seen):
            return False
    closure = getattr(function, "__closure__", None) or ()
    for cell in closure:
        try:
            referenced = cell.cell_contents
        except ValueError:
            return False
        if {names.inspect}.isfunction(referenced) and not {names.safe_code}(referenced, seen):
            return False
    return True


def {names.no_monitoring}(sys_module):
    monitoring = getattr(sys_module, "monitoring", None)
    if monitoring is None:
        return True
    get_events = getattr(monitoring, "get_events", None)
    if get_events is None:
        all_events = getattr(monitoring, "_all_events", None)
        return all_events is None or not all_events()
    for tool_id in range(6):
        try:
            if get_events(tool_id):
                return False
        except ValueError:
            continue
    return True


def {names.safe_callable}(call):
    if not {names.inspect}.iscoroutinefunction(call) or {names.inspect}.isasyncgenfunction(call):
        return False
    try:
        annotation = {names.inspect}.signature(call, follow_wrapped=False).return_annotation
    except (TypeError, ValueError):
        return False
    if annotation is {names.inspect}.Signature.empty:
        return False
    annotation_text = str(annotation)
    if any(
        marker in annotation_text
        for marker in ("AsyncGenerator", "AsyncIterable", "AsyncIterator")
    ):
        return False
    return {names.safe_code}(call, set())


def {names.complete}(call, *args, **kwargs):
    coroutine = call(*args, **kwargs)
    try:
        coroutine.send(None)
    except StopIteration as completed:
        return completed.value
    else:
        coroutine.close()
        raise RuntimeError("guarded immediate callable suspended unexpectedly")


{synchronous_run}


def {names.eligible}(self, {shape.run_task_name}):
    if type({shape.run_task_name}) is not {shape.task_type_expression}:
        return False
    mapping = {shape.node_mapping_expression}
    if type(mapping) is not dict:
        return False
    try:
        node = mapping[{shape.node_key_expression}]
    except (KeyError, TypeError):
        return False
{excluded_node_guard}
    if isinstance(node, {shape.awaited_node_type}):
        descriptor = type(node).__dict__.get("{shape.awaited_callable_attribute}")
        call = getattr(node, "{shape.awaited_callable_attribute}", None)
        function = getattr(call, "__func__", call)
        code = getattr(function, "__code__", None)
        cache = self.{names.eligibility_cache_attribute}
        cache_key = id(node)
        cached = cache.get(cache_key)
        if cached is not None:
            cached_node, cached_descriptor, cached_function, cached_code = cached
            if (
                cached_node is node
                and cached_descriptor is descriptor
                and cached_function is function
                and cached_code is code
            ):
                return True
            cache.pop(cache_key, None)
        if isinstance(descriptor, property):
            if (
                descriptor.fget is None
                or descriptor.fset is not None
                or descriptor.fdel is not None
                or not {names.safe_code}(descriptor.fget, set())
            ):
                return False
        elif descriptor is not None and hasattr(descriptor, "__get__"):
            return False
        safe = {names.safe_callable}(call)
        if safe:
            globals_map = getattr(function, "__globals__", {{}})
            closure = getattr(function, "__closure__", None) or ()
            has_python_dependency = any(
                {names.inspect}.isfunction(globals_map.get(name))
                for name in code.co_names
            )
            has_callable_closure = any(
                {names.inspect}.isfunction(cell.cell_contents)
                for cell in closure
            )
            if not has_python_dependency and not has_callable_closure:
                cache[cache_key] = (node, descriptor, function, code)
        return safe
    return True


def {names.guard}(self, request):
    owner_type = type(self)
    task_group = {shape.task_group_expression}
    start = getattr(task_group, "{shape.task_group_method}", None)
    sender = {shape.sender_expression}
    receiver = {shape.receiver_expression}
    stream_state = getattr(sender, "_state", None)
    try:
        loop = {names.asyncio}.get_running_loop()
    except RuntimeError:
        enabled = False
    else:
        enabled = (
            {names.os}.getenv("ATOLL_DISABLE") != "1"
            and type(request) in (list, tuple)
            and owner_type is {names.expected_owner_class}
            and getattr(owner_type, "{shape.worker_method}", None)
            is {names.expected_worker}
            and getattr({names.expected_worker}, "__code__", None)
            is {names.expected_worker_code}
            and getattr(owner_type, "{shape.run_method_name}", None)
            is {names.expected_run}
            and getattr({names.expected_run}, "__code__", None)
            is {names.expected_run_code}
            and getattr(owner_type, "{shape.consumer.name}", None)
            is {names.expected_consumer}
            and getattr({names.expected_consumer}, "__code__", None)
            is {names.expected_consumer_code}
            and {shape.result_constructor} is {names.expected_result_constructor}
            and type(getattr(self, "{names.deque_attribute}", None))
            is {names.collections}.deque
            and loop.get_task_factory() is None
            and not loop.get_debug()
            and {names.sys}.gettrace() is None
            and {names.sys}.getprofile() is None
            and type(task_group) is {names.expected_task_group_class}
            and type(task_group).start_soon is {names.expected_start_soon}
            and getattr(start, "__self__", None) is task_group
            and getattr(start, "__func__", None) is {names.expected_start_soon}
            and type(sender) is {names.anyio_send_stream}
            and type(receiver) is {names.anyio_receive_stream}
            and stream_state is getattr(receiver, "_state", None)
            and getattr(stream_state, "max_buffer_size", None)
            == {shape.transport_capacity}
            and not getattr(sender, "_closed", True)
            and not getattr(receiver, "_closed", True)
            and not stream_state.buffer
            and not stream_state.waiting_senders
            and not stream_state.waiting_receivers
            and {names.no_monitoring}({names.sys})
        )
    if enabled:
        for item in request:
            if not {names.eligible}(self, item):
                enabled = False
                break
    if (
        not enabled
        and {names.route_hits} == 0
        and {names.os}.getenv("ATOLL_REQUIRE_OPTIMIZED") == "1"
    ):
        raise RuntimeError("ATOLL_REQUIRE_OPTIMIZED=1 but AnyIO source guards failed")
    return enabled


def {names.drive}(self, {shape.run_task_name}):
    global {names.route_hits}
    child_context = {names.contextvars}.copy_context()
    try:
        completed_value = child_context.run(
            {names.synchronous_run}, self, {shape.run_task_name}
        )
{async_result_guard}        record = {shape.result_constructor}(
            {shape.run_task_name}, completed_value
        )
    except BaseException as exc:
        record = {shape.result_constructor}(
            {shape.run_task_name}, {shape.result_empty_expression}, {shape.result_error_keyword}=exc
        )
    self.{names.deque_attribute}.append(record)
    {names.route_hits} += 1
'''
    return (textwrap.dedent(support).strip(),)


def _excluded_node_guard(excluded_node_types: tuple[str, ...]) -> str:
    """Generate an eligibility rejection for async-protocol-producing branches.

    Args:
        excluded_node_types: Nominal type expressions proven to require producer fallback.

    Returns:
        str: Indented generated guard source, or an empty string.
    """
    if not excluded_node_types:
        return ""
    types = ", ".join(excluded_node_types)
    if len(excluded_node_types) == 1:
        types = f"{types},"
    return f"    if isinstance(node, ({types})):\n        return False"


class _AwaitCompletionLowering(ast.NodeTransformer):
    def __init__(self, helper_name: str) -> None:
        self._helper_name = helper_name
        self.replacements = 0

    @override
    def visit_Await(self, node: ast.Await) -> ast.expr:
        """Replace the sole proven await with a synchronous completion helper.

        Args:
            node: Await expression from the statically checked run coroutine.

        Returns:
            ast.expr: Helper call preserving the original callable arguments.

        Raises:
            TypeError: If the checked await no longer contains a direct call.
        """
        if not isinstance(node.value, ast.Call):
            raise TypeError("run coroutine await must contain one direct call")
        self.replacements += 1
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id=self._helper_name, ctx=ast.Load()),
                args=[node.value.func, *node.value.args],
                keywords=node.value.keywords,
            ),
            node,
        )


def _synchronous_run_source(shape: _AnyioShape, names: _Names) -> str:
    """Generate a private synchronous clone of a proven one-await run method.

    Args:
        shape: AnyIO pipeline shape containing the original run-method AST.
        names: Generated completion and clone helper names.

    Returns:
        str: Module-level helper definition used only after runtime identity guards.

    Raises:
        ValueError: If the run method no longer contains exactly one await expression.
    """
    transformer = _AwaitCompletionLowering(names.complete)
    statements = tuple(
        cast(ast.stmt, transformer.visit(copy.deepcopy(statement)))
        for statement in shape.run_method.body
    )
    if transformer.replacements != 1:
        raise ValueError("synchronous run lowering must replace exactly one await")
    ast.fix_missing_locations(ast.Module(body=list(statements), type_ignores=[]))
    body = "\n".join(ast.unparse(statement) for statement in statements)
    indented = textwrap.indent(body, "    ")
    return f"def {names.synchronous_run}(self, {shape.run_task_name}):\n{indented}"


def _identity_captures(shape: _AnyioShape, names: _Names) -> tuple[str, ...]:
    captures = [
        f"{names.expected_owner_class} = {shape.class_name}",
        f"{names.expected_task_group_class} = {names.anyio_backend}.TaskGroup",
        (f"{names.expected_start_soon} = {names.expected_task_group_class}.start_soon"),
        f"{names.expected_worker} = {shape.class_name}.{shape.worker_method}",
        f"{names.expected_worker_code} = {names.expected_worker}.__code__",
        f"{names.expected_run} = {shape.class_name}.{shape.run_method_name}",
        f"{names.expected_run_code} = {names.expected_run}.__code__",
        f"{names.expected_consumer} = {shape.class_name}.{shape.consumer.name}",
        f"{names.expected_consumer_code} = {names.expected_consumer}.__code__",
        f"{names.expected_result_constructor} = {shape.result_constructor}",
    ]
    if shape.reducer is not None:
        reducer = shape.reducer
        captures.extend(
            (
                f"{names.expected_reducer_class} = {reducer.class_expression}",
                (
                    f"{names.expected_reducer_method} = "
                    f"{names.expected_reducer_class}.__dict__[{reducer.method_name!r}]"
                ),
                (f"{names.expected_reducer_code} = {names.expected_reducer_method}.__code__"),
                (
                    f"{names.expected_reducer_property} = "
                    f"{names.expected_reducer_class}.__dict__[{reducer.callable_property!r}]"
                ),
            )
        )
    if shape.protocol is not None:
        protocol = shape.protocol
        captures.extend(
            (
                f"{names.expected_entry_class} = {protocol.entry_class}",
                (
                    f"{names.expected_entry_method} = "
                    f"{names.expected_entry_class}.__dict__[{protocol.entry_method!r}]"
                ),
                (f"{names.expected_entry_code} = {names.expected_entry_method}.__code__"),
                f"{names.expected_runner_class} = {protocol.runner_class}",
                (
                    f"{names.expected_runner_next} = "
                    f"{names.expected_runner_class}.__dict__[{protocol.next_method!r}]"
                ),
                (f"{names.expected_runner_next_code} = {names.expected_runner_next}.__code__"),
                f"{names.expected_terminal_type} = {protocol.terminal_type}",
            )
        )
    return tuple(captures)


def _reducer_support(reducer: _ReducerShape | None, names: _Names) -> str:
    """Generate a guarded per-instance cache for repeated reducer reflection.

    Args:
        reducer: Proven reducer call and property shape, when the plan has one.
        names: Collision-resistant generated helper names.

    Returns:
        str: Module-level support source, or an empty string without a reducer.
    """
    if reducer is None:
        return ""
    return f'''
{names.reducer_states} = {{}}


def {names.reducer}(owner, context, current, inputs):
    owner_type = type(owner)
    method = owner_type.__dict__.get("{reducer.method_name}")
    descriptor = owner_type.__dict__.get("{reducer.callable_property}")
    if (
        owner_type is not {names.expected_reducer_class}
        or method is not {names.expected_reducer_method}
        or getattr(method, "__code__", None) is not {names.expected_reducer_code}
        or descriptor is not {names.expected_reducer_property}
    ):
        return {names.expected_reducer_method}(owner, context, current, inputs)
    callable_object = getattr(owner, "{reducer.callable_property}")
    callable_code = getattr(callable_object, "__code__", None)
    state_key = id(owner)
    state = {names.reducer_states}.get(state_key)
    if (
        state is not None
        and state[0]() is owner
        and state[1] is callable_object
        and state[2] is callable_code
    ):
        parameter_count = state[3]
    else:
        if (
            not {names.inspect}.isfunction(callable_object)
            or hasattr(callable_object, "__signature__")
            or hasattr(callable_object, "__wrapped__")
        ):
            return {names.expected_reducer_method}(owner, context, current, inputs)
        parameter_count = len({names.inspect}.signature(callable_object).parameters)
        try:
            owner_reference = {names.weakref}.ref(
                owner,
                lambda _reference, key=state_key: {names.reducer_states}.pop(key, None),
            )
        except TypeError:
            return {names.expected_reducer_method}(owner, context, current, inputs)
        {names.reducer_states}[state_key] = (
            owner_reference,
            callable_object,
            callable_code,
            parameter_count,
        )
    if parameter_count == 2:
        return callable_object(current, inputs)
    return callable_object(context, current, inputs)
'''.strip()


def _protocol_support(protocol: _ProtocolShape | None, names: _Names) -> str:
    """Generate private run-to-completion auto-forwarding support.

    Args:
        protocol: Exact echo-loop and iterator ownership shape, when discovered.
        names: Collision-resistant generated helper names.

    Returns:
        str: Context-token helpers, or an empty string without a protocol.
    """
    if protocol is None:
        return ""
    return f'''
{names.protocol_context} = {names.contextvars}.ContextVar(
    "{names.protocol_context}", default=None
)


async def {names.protocol_next}(entry_owner, runner, *args):
    next_call = getattr(runner, "{protocol.next_method}", None)
    owner = getattr(runner, "{protocol.owner_attribute}", None)
    enabled = {names.os}.getenv("ATOLL_DISABLE") != "1"
    enabled = enabled and type(entry_owner) is {names.expected_entry_class}
    enabled = enabled and (
        type(entry_owner).__dict__.get("{protocol.entry_method}")
        is {names.expected_entry_method}
    )
    enabled = enabled and (
        getattr({names.expected_entry_method}, "__code__", None)
        is {names.expected_entry_code}
    )
    enabled = enabled and type(runner) is {names.expected_runner_class}
    enabled = enabled and (
        type(runner).__dict__.get("{protocol.next_method}")
        is {names.expected_runner_next}
    )
    enabled = enabled and (
        getattr({names.expected_runner_next}, "__code__", None)
        is {names.expected_runner_next_code}
    )
    enabled = enabled and getattr(next_call, "__self__", None) is runner
    enabled = enabled and getattr(next_call, "__func__", None) is {names.expected_runner_next}
    enabled = enabled and type(owner) is {names.expected_owner_class}
    enabled = enabled and (
        type(owner).__dict__.get("{protocol.owner_attribute}") is None
    )
    enabled = enabled and (
        type(owner).__dict__.get("{protocol.next_method}") is None
    )
    if not enabled:
        return await next_call(*args)
    token = {names.protocol_context}.set(owner)
    try:
        try:
            return await next_call(*args)
        except StopAsyncIteration:
            if len(args) == 1 and args[0] is None:
                terminal_values = [
                    value
                    for value in vars(runner).values()
                    if isinstance(value, {names.expected_terminal_type})
                ]
                if len(terminal_values) == 1:
                    return terminal_values[0]
            raise
    finally:
        {names.protocol_context}.reset(token)


def {names.protocol_forward}(owner, value):
    return (
        {names.protocol_context}.get() is owner
        and {protocol.terminal_type} is {names.expected_terminal_type}
        and not isinstance(value, {names.expected_terminal_type})
    )
'''.strip()


def _names(plan_id: str) -> _Names:
    suffix = re.sub(r"[^A-Za-z0-9_]", "_", plan_id)[-24:]
    prefix = f"_atoll_{suffix}"
    return _Names(
        suffix=suffix,
        collections=f"{prefix}_collections",
        asyncio=f"{prefix}_asyncio",
        contextvars=f"{prefix}_contextvars",
        dis=f"{prefix}_dis",
        inspect=f"{prefix}_inspect",
        os=f"{prefix}_os",
        sys=f"{prefix}_sys",
        typing=f"{prefix}_typing",
        weakref=f"{prefix}_weakref",
        anyio_backend=f"{prefix}_anyio_backend",
        anyio_send_stream=f"{prefix}_anyio_send_stream",
        anyio_receive_stream=f"{prefix}_anyio_receive_stream",
        deque_attribute=f"{prefix}_ready_results",
        eligibility_cache_attribute=f"{prefix}_eligibility_cache",
        route_hits=f"{prefix}_route_hits",
        guard=f"{prefix}_guard",
        no_monitoring=f"{prefix}_no_monitoring",
        eligible=f"{prefix}_eligible",
        safe_callable=f"{prefix}_safe_callable",
        safe_code=f"{prefix}_safe_code",
        complete=f"{prefix}_complete",
        synchronous_run=f"{prefix}_synchronous_run",
        drive=f"{prefix}_drive",
        expected_worker=f"{prefix}_expected_worker",
        expected_worker_code=f"{prefix}_expected_worker_code",
        expected_run=f"{prefix}_expected_run",
        expected_run_code=f"{prefix}_expected_run_code",
        expected_owner_class=f"{prefix}_expected_owner_class",
        expected_task_group_class=f"{prefix}_expected_task_group_class",
        expected_start_soon=f"{prefix}_expected_start_soon",
        expected_consumer=f"{prefix}_expected_consumer",
        expected_consumer_code=f"{prefix}_expected_consumer_code",
        expected_result_constructor=f"{prefix}_expected_result_constructor",
        fast_local=f"{prefix}_enabled",
        reducer=f"{prefix}_reduce",
        reducer_states=f"{prefix}_reducer_states",
        expected_reducer_class=f"{prefix}_expected_reducer_class",
        expected_reducer_method=f"{prefix}_expected_reducer_method",
        expected_reducer_code=f"{prefix}_expected_reducer_code",
        expected_reducer_property=f"{prefix}_expected_reducer_property",
        protocol_context=f"{prefix}_protocol_context",
        protocol_next=f"{prefix}_protocol_next",
        protocol_forward=f"{prefix}_protocol_forward",
        expected_entry_class=f"{prefix}_expected_entry_class",
        expected_entry_method=f"{prefix}_expected_entry_method",
        expected_entry_code=f"{prefix}_expected_entry_code",
        expected_runner_class=f"{prefix}_expected_runner_class",
        expected_runner_next=f"{prefix}_expected_runner_next",
        expected_runner_next_code=f"{prefix}_expected_runner_next_code",
        expected_terminal_type=f"{prefix}_expected_terminal_type",
    )


def _expression_path(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _expression_path(node.value)
        return f"{parent}.{node.attr}" if parent is not None else None
    return None


def _cst_path(node: cst.CSTNode) -> str | None:
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        parent = _cst_path(node.value)
        return f"{parent}.{node.attr.value}" if parent is not None else None
    return None


def _cst_nodes(node: cst.CSTNode) -> tuple[cst.CSTNode, ...]:
    collector = _NodeCollector()
    node.visit(collector)
    return tuple(collector.nodes)


class _NodeCollector(cst.CSTVisitor):
    def __init__(self) -> None:
        self.nodes: list[cst.CSTNode] = []

    @override
    def on_visit(self, node: cst.CSTNode) -> bool:
        """Collect every node in deterministic pre-order.

        Args:
            node: Current LibCST node.

        Returns:
            bool: ``True`` so child traversal continues.
        """
        self.nodes.append(node)
        return True
