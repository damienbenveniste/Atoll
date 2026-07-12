"""Guarded source lowering for private asyncio fan-out pipelines.

This module recognizes one deliberately narrow orchestration shape and lowers
it into a source patch request. AST analysis proves the scheduling topology;
LibCST clones the original callable for fallback and rewrites a private fast
copy without editing by text anchor. Unsupported shapes return deterministic
reasons and remain interpreted.
"""

from __future__ import annotations

import ast
import builtins
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast, override

import libcst as cst

from atoll.models import SymbolId
from atoll.source_optimization.models import (
    SourceCallableEvidence,
    SourceOptimizationAssessment,
    SourceOptimizationPlan,
)
from atoll.source_optimization.transforms import SourceTransformationRequest

SourceLoweringStatus = Literal["lowered", "unsupported"]
_PIPELINE_STATEMENT_COUNT = 2
_WORKER_ARGUMENT_COUNT = 2


@dataclass(frozen=True, slots=True)
class SourceLoweringResult:
    """Result of lowering one source-optimization plan.

    Attributes:
        plan_id: Stable source-optimization plan identifier.
        status: Whether a guarded source request was produced.
        request: LibCST transformation request for a supported plan.
        helper_names: Generated private helpers available to strict routing tests.
        rejections: Deterministic reasons that kept the plan interpreted.
    """

    plan_id: str
    status: SourceLoweringStatus
    request: SourceTransformationRequest | None
    helper_names: tuple[str, ...] = ()
    rejections: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _PipelineShape:
    owner: ast.AsyncFunctionDef
    worker: ast.AsyncFunctionDef
    scheduler_name: str
    group_name: str
    queue_name: str
    count_expression: str
    item_name: str
    iterable_name: str
    receive_target: str
    task_group: ast.AsyncWith


@dataclass(frozen=True, slots=True)
class _QueueValidation:
    owner: ast.AsyncFunctionDef
    task_group: ast.AsyncWith
    scheduler_name: str
    queue_name: str
    count: ast.expr
    iterable_name: str


@dataclass(frozen=True, slots=True)
class _GuardInputs:
    global_names: tuple[str, ...]
    callable_globals: frozenset[str]


@dataclass(frozen=True, slots=True)
class _HelperNames:
    prefix: str
    original: str
    fast: str
    guard: str
    require: str
    drive: str
    batch: str
    context_module: str
    os_module: str
    expected_worker: str
    expected_worker_code: str
    expected_queue: str
    expected_queue_empty: str
    expected_task_group: str
    expected_get_running_loop: str
    expected_copy_context: str

    @property
    def public_tuple(self) -> tuple[str, ...]:
        """Return generated helper names exposed to strict-routing tests.

        Returns:
            tuple[str, ...]: Original, fast, guard, driver, and batch helper names.
        """
        return (self.original, self.fast, self.guard, self.drive, self.batch)


def lower_batch_quiescent_plan(
    project_root: Path,
    plan: SourceOptimizationPlan,
    assessment: SourceOptimizationAssessment,
) -> SourceLoweringResult:
    """Lower one proven asyncio queue pipeline into a guarded source request.

    The fast path runs each statically quiescent worker coroutine to completion
    inside its own copied context, then drains the private queue synchronously.
    Runtime identity guards execute before the original owner performs any side
    effect. A guard failure calls the copied original implementation; a failure
    after fast-path entry is surfaced and never retried.

    Args:
        project_root: Target project root containing the plan source path.
        plan: Static source-optimization plan derived from an execution plan.
        assessment: Current-invocation profile and safety assessment for `plan`.

    Returns:
        SourceLoweringResult: A deterministic request or conservative rejection reasons.
    """
    preflight = _preflight_rejections(plan, assessment)
    if preflight:
        return _unsupported(plan, preflight)
    try:
        source_path = _source_path(project_root, plan)
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(source_path))
        shape = _analyze_pipeline(tree, plan)
        module = cst.parse_module(source)
        names = _helper_names(plan.id)
        request = _build_request(plan, assessment, module, shape, names)
    except (OSError, SyntaxError, TypeError, ValueError, cst.ParserSyntaxError) as error:
        return _unsupported(plan, (str(error),))
    return SourceLoweringResult(
        plan_id=plan.id,
        status="lowered",
        request=request,
        helper_names=names.public_tuple,
    )


def _preflight_rejections(
    plan: SourceOptimizationPlan,
    assessment: SourceOptimizationAssessment,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if assessment.plan_id != plan.id:
        reasons.append("source assessment belongs to a different plan")
    if assessment.status != "trial-ready":
        reasons.append(f"source assessment is {assessment.status}, not trial-ready")
    if plan.identity.dialect != "asyncio":
        reasons.append(f"source lowering does not support dialect {plan.identity.dialect}")
    required_steps = {
        "private-transport-batch-drain",
        "quiescent-callable-execution",
    }
    available_steps = {step.kind for step in plan.steps}
    if not required_steps.issubset(available_steps):
        reasons.append("source plan lacks batch-drain and quiescent-execution steps")
    if assessment.immediate_result_ratio != 1.0:
        reasons.append("quiescent lowering requires a 100% immediate-result ratio")
    reasons.extend(_evidence_rejections(plan, assessment.callable_evidence))
    return tuple(reasons)


def _evidence_rejections(
    plan: SourceOptimizationPlan,
    evidence: tuple[SourceCallableEvidence, ...],
) -> tuple[str, ...]:
    reasons: list[str] = []
    worker_evidence = next((item for item in evidence if item.symbol == plan.worker), None)
    if worker_evidence is None:
        reasons.append("source assessment has no worker evidence")
        return tuple(reasons)
    for item in evidence:
        if item.static_role not in {"worker", "producer", "wrapper", "dependency"}:
            continue
        if item.static_suspension_points or item.observed_suspensions:
            reasons.append(f"{item.symbol.stable_id} can suspend")
        if item.task_introspection:
            reasons.append(f"{item.symbol.stable_id} observes task state")
        if item.cancellation:
            reasons.append(f"{item.symbol.stable_id} uses cancellation APIs")
        if item.unknown_dynamic_calls:
            reasons.append(f"{item.symbol.stable_id} has unknown dynamic calls")
        if item.context_mutation and item.symbol != plan.worker:
            reasons.append(f"{item.symbol.stable_id} mutates context indirectly")
    return tuple(reasons)


def _source_path(project_root: Path, plan: SourceOptimizationPlan) -> Path:
    root = project_root.resolve()
    if plan.source.is_absolute() or ".." in plan.source.parts:
        raise ValueError(f"unsafe source path: {plan.source.as_posix()}")
    path = (root / Path(plan.source.as_posix())).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"source path escapes project root: {plan.source.as_posix()}") from error
    if not path.is_file():
        raise ValueError(f"source plan file does not exist: {plan.source.as_posix()}")
    return path


def _analyze_pipeline(tree: ast.Module, plan: SourceOptimizationPlan) -> _PipelineShape:
    owner = _top_level_async_function(tree, plan.owner, role="owner")
    worker = _top_level_async_function(tree, plan.worker, role="worker")
    task_groups = [node for node in owner.body if _is_task_group(node)]
    if len(task_groups) != 1:
        raise ValueError("source owner must contain exactly one top-level asyncio.TaskGroup")
    task_group = cast(ast.AsyncWith, task_groups[0])
    scheduler_name, group_name = _task_group_names(task_group)
    loop, receive = _task_group_statements(task_group)
    queue_name, item_name = _spawn_shape(loop, group_name, plan.worker)
    receive_target, count = _receive_shape(receive, queue_name)
    iterable_name = _name(loop.iter, label="task loop iterable")
    _validate_queue_capacity(
        _QueueValidation(
            owner=owner,
            task_group=task_group,
            scheduler_name=scheduler_name,
            queue_name=queue_name,
            count=count,
            iterable_name=iterable_name,
        )
    )
    _validate_owner_awaits(owner)
    _validate_worker(worker, queue_name=queue_name, item_name=item_name)
    return _PipelineShape(
        owner=owner,
        worker=worker,
        scheduler_name=scheduler_name,
        group_name=group_name,
        queue_name=queue_name,
        count_expression=ast.unparse(count),
        item_name=item_name,
        iterable_name=iterable_name,
        receive_target=receive_target,
        task_group=task_group,
    )


def _top_level_async_function(
    tree: ast.Module,
    symbol: SymbolId,
    *,
    role: str,
) -> ast.AsyncFunctionDef:
    if "." in symbol.qualname:
        raise ValueError(f"{role} method lowering is not supported by this milestone")
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == symbol.qualname
    ]
    if len(matches) != 1:
        raise ValueError(f"source {role} must resolve to one top-level async function")
    return matches[0]


def _is_task_group(node: ast.stmt) -> bool:
    if not isinstance(node, ast.AsyncWith) or len(node.items) != 1:
        return False
    expression = node.items[0].context_expr
    return (
        isinstance(expression, ast.Call)
        and not expression.args
        and not expression.keywords
        and _attribute_path(expression.func) == "asyncio.TaskGroup"
    )


def _task_group_names(node: ast.AsyncWith) -> tuple[str, str]:
    item = node.items[0]
    if not isinstance(item.optional_vars, ast.Name):
        raise TypeError("asyncio.TaskGroup must bind one local group name")
    path = _attribute_path(cast(ast.Call, item.context_expr).func)
    if path is None or "." not in path:
        raise ValueError("asyncio.TaskGroup scheduler path is unavailable")
    return path.split(".", maxsplit=1)[0], item.optional_vars.id


def _task_group_statements(node: ast.AsyncWith) -> tuple[ast.For, ast.Assign]:
    if len(node.body) != _PIPELINE_STATEMENT_COUNT:
        raise ValueError("private TaskGroup body must contain one spawn loop and one receive")
    loop, receive = node.body
    if not isinstance(loop, ast.For) or not isinstance(receive, ast.Assign):
        raise TypeError("private TaskGroup must spawn before receiving results")
    return loop, receive


def _spawn_shape(loop: ast.For, group_name: str, worker: SymbolId) -> tuple[str, str]:
    if loop.orelse or len(loop.body) != 1 or not isinstance(loop.target, ast.Name):
        raise ValueError("task spawn loop must have one simple target and no else branch")
    statement = loop.body[0]
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        raise TypeError("task spawn loop must contain one create_task call")
    create_task = statement.value
    if (
        _attribute_path(create_task.func) != f"{group_name}.create_task"
        or len(create_task.args) != 1
        or create_task.keywords
    ):
        raise ValueError("task spawn must use the bound TaskGroup without options")
    worker_call = create_task.args[0]
    if (
        not isinstance(worker_call, ast.Call)
        or _attribute_path(worker_call.func) != worker.qualname
        or len(worker_call.args) != _WORKER_ARGUMENT_COUNT
        or worker_call.keywords
    ):
        raise ValueError("spawned worker must be one exact two-argument module callable")
    queue_name = _name(worker_call.args[0], label="worker queue argument")
    item_name = _name(worker_call.args[1], label="worker item argument")
    if item_name != loop.target.id:
        raise ValueError("worker item argument must be the task loop target")
    return queue_name, item_name


def _receive_shape(statement: ast.Assign, queue_name: str) -> tuple[str, ast.expr]:
    if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
        raise ValueError("private receive must assign one local result list")
    value = statement.value
    if not isinstance(value, ast.ListComp) or len(value.generators) != 1:
        raise ValueError("private receive must be one queue list comprehension")
    generator = value.generators[0]
    if generator.is_async or generator.ifs:
        raise ValueError("private receive comprehension cannot be async or filtered")
    if not (
        isinstance(value.elt, ast.Await)
        and isinstance(value.elt.value, ast.Call)
        and _attribute_path(value.elt.value.func) == f"{queue_name}.get"
        and not value.elt.value.args
        and not value.elt.value.keywords
    ):
        raise ValueError("private receive must await the owned queue")
    if not (
        isinstance(generator.iter, ast.Call)
        and _attribute_path(generator.iter.func) == "range"
        and len(generator.iter.args) == 1
        and not generator.iter.keywords
    ):
        raise ValueError("private receive count must use range(count)")
    return statement.targets[0].id, generator.iter.args[0]


def _validate_queue_capacity(context: _QueueValidation) -> None:
    preceding = context.owner.body[: context.owner.body.index(context.task_group)]
    queue_values = [
        value
        for statement in preceding
        if (value := _assigned_value(statement, context.queue_name)) is not None
    ]
    if len(queue_values) != 1 or not isinstance(queue_values[0], ast.Call):
        raise ValueError("private queue must have one local constructor assignment")
    queue_call = queue_values[0]
    if _attribute_path(queue_call.func) != f"{context.scheduler_name}.Queue":
        raise ValueError("private transport must be the scheduler's exact Queue type")
    capacity = _queue_capacity(queue_call)
    if ast.dump(capacity, include_attributes=False) != ast.dump(
        context.count, include_attributes=False
    ):
        raise ValueError("queue capacity and receive count must be the same expression")
    if not _is_len_of(context.count, context.iterable_name):
        raise ValueError("work count must be len(task_iterable)")


def _assigned_value(statement: ast.stmt, name: str) -> ast.expr | None:
    if (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == name
    ):
        return statement.value
    if (
        isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
        and statement.target.id == name
    ):
        return statement.value
    return None


def _queue_capacity(call: ast.Call) -> ast.expr:
    keyword_values = [item.value for item in call.keywords if item.arg == "maxsize"]
    if len(keyword_values) == 1 and not call.args:
        return keyword_values[0]
    if len(call.args) == 1 and not call.keywords:
        return call.args[0]
    raise ValueError("private queue must have one explicit maxsize")


def _is_len_of(node: ast.expr, iterable_name: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "len"
        and len(node.args) == 1
        and not node.keywords
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == iterable_name
    )


def _validate_owner_awaits(owner: ast.AsyncFunctionDef) -> None:
    awaits = [node for node in ast.walk(owner) if isinstance(node, ast.Await)]
    if len(awaits) != 1:
        raise ValueError("source owner may await only its private queue receive")
    if any(isinstance(node, ast.Yield | ast.YieldFrom) for node in ast.walk(owner)):
        raise ValueError("source owner cannot be a generator")


def _validate_worker(
    worker: ast.AsyncFunctionDef,
    *,
    queue_name: str,
    item_name: str,
) -> None:
    positional = (*worker.args.posonlyargs, *worker.args.args)
    if (
        len(positional) != _WORKER_ARGUMENT_COUNT
        or worker.args.vararg
        or worker.args.kwarg
        or worker.args.kwonlyargs
    ):
        raise ValueError("quiescent worker must have exactly two positional parameters")
    queue_parameter, item_parameter = (parameter.arg for parameter in positional)
    if queue_parameter != queue_name or item_parameter != item_name:
        raise ValueError("worker parameters must match the queue and loop item names")
    if any(
        isinstance(node, ast.Await | ast.Yield | ast.YieldFrom | ast.Global | ast.Nonlocal)
        for node in ast.walk(worker)
    ):
        raise ValueError("quiescent worker contains suspension or nonlocal state")
    publications = [
        node
        for node in ast.walk(worker)
        if isinstance(node, ast.Call)
        and _attribute_path(node.func) == f"{queue_parameter}.put_nowait"
    ]
    if len(publications) != 1:
        raise ValueError("quiescent worker must publish exactly one private result")
    final_statement = worker.body[-1] if worker.body else None
    if not (isinstance(final_statement, ast.Expr) and final_statement.value is publications[0]):
        raise ValueError("quiescent worker publication must be its final top-level statement")
    if any(_writes_attribute_or_subscript(node) for node in ast.walk(worker)):
        raise ValueError("quiescent worker mutates attribute or container state")


def _writes_attribute_or_subscript(node: ast.AST) -> bool:
    targets: tuple[ast.expr, ...] = ()
    if isinstance(node, ast.Assign):
        targets = tuple(node.targets)
    elif isinstance(node, ast.AnnAssign | ast.AugAssign):
        targets = (node.target,)
    elif isinstance(node, ast.Delete):
        targets = tuple(node.targets)
    return any(isinstance(target, ast.Attribute | ast.Subscript) for target in targets)


def _build_request(
    plan: SourceOptimizationPlan,
    assessment: SourceOptimizationAssessment,
    module: cst.Module,
    shape: _PipelineShape,
    names: _HelperNames,
) -> SourceTransformationRequest:
    owner = _top_level_function(module, plan.owner)
    call_arguments = _call_arguments(owner.params)
    docstring = ast.get_docstring(shape.owner, clean=False)
    wrapper_body = _wrapper_body(names, call_arguments, docstring)
    helper_params = _required_parameters(owner.params)
    original = owner.with_changes(
        name=cst.Name(names.original),
        decorators=(),
        params=helper_params,
        returns=None,
        type_parameters=None,
    )
    fast = owner.with_changes(
        name=cst.Name(names.fast),
        asynchronous=None,
        decorators=(),
        params=helper_params,
        returns=None,
        type_parameters=None,
    )
    fast = _replace_task_group(fast, shape, names)
    guard_inputs = _GuardInputs(
        global_names=_guarded_globals(shape.worker),
        callable_globals=frozenset(
            item.symbol.qualname
            for item in assessment.callable_evidence
            if item.symbol.module == plan.worker.module and "." not in item.symbol.qualname
        ),
    )
    trailing = _trailing_helpers(
        module,
        (original, fast),
        shape,
        names,
        guard_inputs,
    )
    source_hash = _plan_source_hash(plan)
    return SourceTransformationRequest(
        path=plan.source,
        expected_sha256=source_hash,
        target=plan.owner,
        declaration_kind="async_function",
        replacement_body=wrapper_body,
        trailing_statements=trailing,
        summary="add guarded private batch drain and copied-context quiescent execution",
        transformation_id=f"{plan.id}:batch-quiescent-v1",
    )


def _wrapper_body(names: _HelperNames, arguments: str, docstring: str | None) -> str:
    lines: list[str] = []
    if docstring is not None:
        lines.append(f"{docstring!r}")
    lines.extend(
        (
            f"if {names.guard}():",
            f"    return {names.fast}({arguments})",
            f"if {names.require}():",
            "    raise RuntimeError('ATOLL_REQUIRE_OPTIMIZED=1 but source guards failed')",
            f"return await {names.original}({arguments})",
        )
    )
    return "\n".join(lines) + "\n"


def _required_parameters(parameters: cst.Parameters) -> cst.Parameters:
    return parameters.with_changes(
        posonly_params=tuple(_required_parameter(item) for item in parameters.posonly_params),
        params=tuple(_required_parameter(item) for item in parameters.params),
        star_arg=(
            _unannotated_variadic(parameters.star_arg)
            if isinstance(parameters.star_arg, cst.Param)
            else parameters.star_arg
        ),
        kwonly_params=tuple(_required_parameter(item) for item in parameters.kwonly_params),
        star_kwarg=(
            _unannotated_variadic(parameters.star_kwarg)
            if parameters.star_kwarg is not None
            else None
        ),
    )


def _required_parameter(parameter: cst.Param) -> cst.Param:
    return parameter.with_changes(
        annotation=None,
        default=None,
        equal=cst.MaybeSentinel.DEFAULT,
    )


def _unannotated_variadic(parameter: cst.Param) -> cst.Param:
    return parameter.with_changes(annotation=None)


def _call_arguments(parameters: cst.Parameters) -> str:
    arguments = [item.name.value for item in (*parameters.posonly_params, *parameters.params)]
    if isinstance(parameters.star_arg, cst.Param):
        arguments.append(f"*{parameters.star_arg.name.value}")
    arguments.extend(f"{item.name.value}={item.name.value}" for item in parameters.kwonly_params)
    if parameters.star_kwarg is not None:
        arguments.append(f"**{parameters.star_kwarg.name.value}")
    return ", ".join(arguments)


class _TaskGroupTransformer(cst.CSTTransformer):
    """Replace the one validated TaskGroup with generated synchronous statements."""

    def __init__(
        self,
        scheduler_name: str,
        replacement: tuple[cst.BaseStatement, ...],
    ) -> None:
        self._scheduler_name = scheduler_name
        self._replacement = replacement
        self.replacements = 0

    @override
    def leave_With(
        self,
        original_node: cst.With,
        updated_node: cst.With,
    ) -> cst.BaseStatement | cst.FlattenSentinel[cst.BaseStatement]:
        """Replace the exact scheduler TaskGroup after AST validation.

        Args:
            original_node: Original asynchronous context manager.
            updated_node: Child-transformed asynchronous context manager.

        Returns:
            The untouched context manager or flattened fast-path statements.
        """
        if not _is_cst_task_group(original_node, self._scheduler_name):
            return updated_node
        self.replacements += 1
        return cst.FlattenSentinel(self._replacement)


def _replace_task_group(
    fast: cst.FunctionDef,
    shape: _PipelineShape,
    names: _HelperNames,
) -> cst.FunctionDef:
    generated = _fast_task_group_source(shape, names)
    replacement = tuple(cst.parse_module(generated).body)
    transformer = _TaskGroupTransformer(shape.scheduler_name, replacement)
    transformed_module = cst.Module(body=(fast,)).visit(transformer)
    if transformer.replacements != 1:
        raise ValueError("LibCST could not replace exactly one validated TaskGroup")
    transformed = transformed_module.body[0]
    if not isinstance(transformed, cst.FunctionDef):
        raise TypeError("LibCST fast helper is not a function declaration")
    return transformed


def _fast_task_group_source(shape: _PipelineShape, names: _HelperNames) -> str:
    return (
        f"{names.prefix}_errors = []\n"
        f"for {shape.item_name} in {shape.iterable_name}:\n"
        f"    {names.prefix}_context = {names.context_module}.copy_context()\n"
        f"    {names.prefix}_coroutine = {names.prefix}_context.run(\n"
        f"        {shape.worker.name}, {shape.queue_name}, {shape.item_name}\n"
        "    )\n"
        "    try:\n"
        f"        {names.drive}({names.prefix}_coroutine, {names.prefix}_context)\n"
        "    except (KeyboardInterrupt, SystemExit):\n"
        "        raise\n"
        f"    except BaseException as {names.prefix}_error:\n"
        f"        {names.prefix}_errors.append({names.prefix}_error)\n"
        f"if {names.prefix}_errors:\n"
        "    raise BaseExceptionGroup(\n"
        "        'unhandled errors in a TaskGroup',\n"
        f"        {names.prefix}_errors,\n"
        "    )\n"
        f"{shape.receive_target} = {names.batch}({shape.queue_name}, "
        f"{shape.count_expression})\n"
    )


def _is_cst_task_group(node: cst.With, scheduler_name: str) -> bool:
    if node.asynchronous is None or len(node.items) != 1:
        return False
    expression = node.items[0].item
    return (
        isinstance(expression, cst.Call)
        and not expression.args
        and isinstance(expression.func, cst.Attribute)
        and isinstance(expression.func.value, cst.Name)
        and expression.func.value.value == scheduler_name
        and expression.func.attr.value == "TaskGroup"
    )


def _trailing_helpers(
    module: cst.Module,
    declarations: tuple[cst.FunctionDef, cst.FunctionDef],
    shape: _PipelineShape,
    names: _HelperNames,
    guard_inputs: _GuardInputs,
) -> tuple[str, ...]:
    original, fast = declarations
    captures, guards = _global_guards(
        names,
        guard_inputs.global_names,
        guard_inputs.callable_globals,
    )
    support = _support_source(shape, names, captures, guards)
    original_source = module.code_for_node(original)
    fast_source = module.code_for_node(fast)
    return (support, original_source, fast_source)


def _support_source(
    shape: _PipelineShape,
    names: _HelperNames,
    captures: tuple[str, ...],
    guards: tuple[str, ...],
) -> str:
    guard_expression = "\n        and ".join(
        (
            f"{shape.worker.name} is {names.expected_worker}",
            f"getattr({shape.worker.name}, '__code__', None) is {names.expected_worker_code}",
            f"{shape.scheduler_name}.Queue is {names.expected_queue}",
            f"{shape.scheduler_name}.QueueEmpty is {names.expected_queue_empty}",
            f"{shape.scheduler_name}.TaskGroup is {names.expected_task_group}",
            f"{shape.scheduler_name}.get_running_loop is {names.expected_get_running_loop}",
            f"{names.context_module}.copy_context is {names.expected_copy_context}",
            *guards,
        )
    )
    capture_source = "\n".join(captures)
    return f"""
import contextvars as {names.context_module}
import os as {names.os_module}

{names.expected_worker} = {shape.worker.name}
{names.expected_worker_code} = getattr({shape.worker.name}, "__code__", None)
{names.expected_queue} = {shape.scheduler_name}.Queue
{names.expected_queue_empty} = {shape.scheduler_name}.QueueEmpty
{names.expected_task_group} = {shape.scheduler_name}.TaskGroup
{names.expected_get_running_loop} = {shape.scheduler_name}.get_running_loop
{names.expected_copy_context} = {names.context_module}.copy_context
{capture_source}

def {names.require}():
    return {names.os_module}.getenv("ATOLL_REQUIRE_OPTIMIZED") == "1"


def {names.guard}():
    if {names.os_module}.getenv("ATOLL_DISABLE") == "1":
        return False
    loop = {shape.scheduler_name}.get_running_loop()
    if loop.get_task_factory() is not None:
        return False
    return (
        {guard_expression}
    )


def {names.drive}(coroutine, context):
    try:
        context.run(coroutine.send, None)
    except StopIteration as completed:
        return completed.value
    except BaseException:
        context.run(coroutine.close)
        raise
    context.run(coroutine.close)
    raise RuntimeError("Atoll quiescent worker suspended after optimized entry")


def {names.batch}(queue, count):
    records = []
    for _index in range(count):
        try:
            records.append(queue.get_nowait())
        except {shape.scheduler_name}.QueueEmpty as error:
            raise RuntimeError("Atoll private queue was not ready after optimized entry") from error
    return records
"""


def _global_guards(
    names: _HelperNames,
    global_names: tuple[str, ...],
    callable_globals: frozenset[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    captures: list[str] = []
    guards: list[str] = []
    for index, name in enumerate(global_names):
        expected = f"{names.prefix}_expected_global_{index}"
        captures.append(f"{expected} = {name}")
        guards.append(f"{name} is {expected}")
        if name in callable_globals:
            expected_code = f"{expected}_code"
            captures.append(f"{expected_code} = getattr({name}, '__code__', None)")
            guards.append(f"getattr({name}, '__code__', None) is {expected_code}")
    return tuple(captures), tuple(guards)


def _guarded_globals(worker: ast.AsyncFunctionDef) -> tuple[str, ...]:
    parameter_names = {
        item.arg
        for item in (
            *worker.args.posonlyargs,
            *worker.args.args,
            *worker.args.kwonlyargs,
        )
    }
    local_names = parameter_names | {
        node.id
        for node in ast.walk(worker)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    builtin_names = frozenset(dir(builtins))
    return tuple(
        sorted(
            {
                node.id
                for node in ast.walk(worker)
                if isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id not in local_names
                and node.id not in builtin_names
            }
        )
    )


def _top_level_function(module: cst.Module, symbol: SymbolId) -> cst.FunctionDef:
    matches = [
        node
        for node in module.body
        if isinstance(node, cst.FunctionDef) and node.name.value == symbol.qualname
    ]
    if len(matches) != 1:
        raise ValueError("LibCST owner declaration is missing or duplicated")
    if matches[0].asynchronous is None:
        raise ValueError("LibCST owner declaration is not async")
    return matches[0]


def _plan_source_hash(plan: SourceOptimizationPlan) -> str:
    matches = [digest for path, digest in plan.identity.source_hashes if path == plan.source]
    if len(matches) != 1:
        raise ValueError("source plan does not contain one hash for its owner file")
    return matches[0]


def _helper_names(plan_id: str) -> _HelperNames:
    suffix = re.sub(r"[^a-zA-Z0-9_]", "_", plan_id)[-16:]
    prefix = f"_atoll_source_{suffix}"
    return _HelperNames(
        prefix=prefix,
        original=f"{prefix}_original",
        fast=f"{prefix}_fast",
        guard=f"{prefix}_guard",
        require=f"{prefix}_require",
        drive=f"{prefix}_drive",
        batch=f"{prefix}_batch",
        context_module=f"{prefix}_contextvars",
        os_module=f"{prefix}_os",
        expected_worker=f"{prefix}_expected_worker",
        expected_worker_code=f"{prefix}_expected_worker_code",
        expected_queue=f"{prefix}_expected_queue",
        expected_queue_empty=f"{prefix}_expected_queue_empty",
        expected_task_group=f"{prefix}_expected_task_group",
        expected_get_running_loop=f"{prefix}_expected_get_running_loop",
        expected_copy_context=f"{prefix}_expected_copy_context",
    )


def _attribute_path(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _attribute_path(node.value)
        return f"{parent}.{node.attr}" if parent is not None else None
    return None


def _name(node: ast.expr, *, label: str) -> str:
    if not isinstance(node, ast.Name):
        raise TypeError(f"{label} must be one local name")
    return node.id


def _unsupported(
    plan: SourceOptimizationPlan,
    reasons: tuple[str, ...],
) -> SourceLoweringResult:
    return SourceLoweringResult(
        plan_id=plan.id,
        status="unsupported",
        request=None,
        rejections=reasons,
    )
