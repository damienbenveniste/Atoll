"""Task-preserving execution-plan backend for payload-only scheduler rewrites.

The backend owns a deliberately small lowering surface: it binds a stable
`TaskGroup.create_task` method once before a fan-out loop and leaves each
logical work item as one real scheduler task. It does not fuse tasks, inline
workers, touch source checkouts, or attempt to recover from changed topology.
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

_BACKEND_NAME: Final = "task-preserving"
_LOWERING_VERSION: Final = "task-preserving-v1"
_SUPPORTED_DIALECT: Final = "asyncio"
_QUALNAME_CLASS_METHOD_PARTS: Final = 2


@dataclass(frozen=True, slots=True)
class TaskPreservingExecutionPlanBackend:
    """Payload-only backend that preserves scheduler task identity.

    The backend is intentionally conservative. It accepts only structured
    asyncio `TaskGroup.create_task` fan-out loops whose call site still matches
    the plan selected by discovery. Staging edits only the unpacked payload file
    under `ExecutionPlanStageContext.payload_root`.

    Attributes:
        name: Stable backend identifier used in reports and fingerprints.
    """

    name: str = _BACKEND_NAME

    def assess(
        self,
        plan: ExecutionPlan,
        context: ExecutionPlanAssessmentContext,
    ) -> ExecutionPlanAssessment:
        """Classify whether the backend can preserve the plan's task shape.

        Args:
            plan: Scheduler-aware execution plan discovered from source.
            context: Read-only project and source-root context for assessment.

        Returns:
            ExecutionPlanAssessment: Deterministic capability decision with
            unsupported node IDs and human-readable reasons.
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
        """Stage a task-preserving source rewrite in the payload directory.

        Args:
            plan: Scheduler-aware execution plan to stage.
            context: Payload and cache roots for this staging attempt.

        Returns:
            StagedExecutionPlan: Changed payload file evidence and runtime guards.

        Raises:
            TypeError: If the fan-out loop uses an unsupported iteration shape.
            ValueError: If the payload module is absent, the source hash is stale,
                the call-site fingerprint changed, or the shape is unsupported.
        """
        reasons = tuple(_static_rejection_reasons(plan))
        if reasons:
            raise ValueError(f"unsupported task-preserving execution plan: {'; '.join(reasons)}")
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
                        "optimized scheduling is used only when the create_task binding still "
                        "belongs to the original task group"
                    ),
                ),
            ),
        )

    def fingerprint(
        self,
        plan: ExecutionPlan,
        context: ExecutionPlanStageContext,
    ) -> str:
        """Return a strict fingerprint for the backend, plan, and payload source.

        Args:
            plan: Scheduler-aware execution plan being fingerprinted.
            context: Payload root containing the source file that would be staged.

        Returns:
            str: Stable SHA-256 digest covering backend semantics, plan identity,
            and current payload contents.

        Raises:
            ValueError: If the payload module is absent.
        """
        payload_path = _module_path(context.payload_root, plan.source_module)
        if payload_path is None:
            raise ValueError(f"payload module is not present: {plan.source_module}")
        digest = hashlib.sha256()
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
            _sha256(payload_path.read_text(encoding="utf-8")),
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
        """Convert backend failures into deterministic diagnostic fields.

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
            code="TASK_PRESERVING_EXECUTION_PLAN_ERROR",
            severity="error",
            message=str(error) or error.__class__.__name__,
            details=details,
        )


@dataclass(frozen=True, slots=True)
class _Rewrite:
    source_text: str
    guard_expression: str


@dataclass(frozen=True, slots=True)
class _LoopRewriteTarget:
    function: ast.AsyncFunctionDef | ast.FunctionDef
    loop: ast.For | ast.AsyncFor
    statement: ast.Expr
    call: ast.Call
    scheduler_name: str


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
    if not any(edge.kind == "spawns" for edge in plan.edges):
        reasons.append("plan has no spawn edges")
    return tuple(reasons)


def _validated_rewrite(source_text: str, plan: ExecutionPlan) -> _Rewrite:
    tree = ast.parse(source_text, type_comments=True)
    _validate_source_hash(source_text, tree, plan)
    _validate_callsite_fingerprint(tree, plan)
    target = _rewrite_target(tree, plan)
    binding_name = _binding_name(plan.id)
    guard_name = f"{binding_name}_valid"
    for generated_name in (binding_name, guard_name):
        if _name_exists(tree, generated_name):
            raise ValueError(f"task-preserving binding name already exists: {generated_name}")
    if isinstance(target.loop, ast.AsyncFor):
        raise TypeError("async fan-out iteration is not safe for descriptor hoisting")
    if not isinstance(target.loop.iter, ast.Name):
        raise TypeError("fan-out iterable must be a side-effect-free local name")
    _reject_scheduler_reassignment(target.loop, target.scheduler_name)
    iterable_source = ast.get_source_segment(source_text, target.loop.iter)
    if iterable_source is None or "\n" in iterable_source:
        raise ValueError("fan-out iterable source is unavailable for a line-preserving rewrite")
    for builtin_name in ("type", "list", "tuple", "range"):
        if _name_is_bound(tree, target.function, builtin_name):
            raise ValueError(
                f"task-preserving iterable guard requires unshadowed builtin {builtin_name}"
            )
    safe_iterable = f"type({iterable_source}) in (list, tuple, range)"
    bound_method = f"{target.scheduler_name}.create_task"
    bound_guard = f"getattr({binding_name}, '__self__', None) is {target.scheduler_name}"
    guard_expression = f"{safe_iterable} and {bound_guard}"
    rewritten_iterable = (
        f"((({binding_name} := {bound_method}), "
        f"({guard_name} := {bound_guard}), {iterable_source})[2] "
        f"if {safe_iterable} else (({guard_name} := False), {iterable_source})[1])"
    )
    rewritten_callable = (
        f"({binding_name} if {guard_name} else {target.scheduler_name}.create_task)"
    )
    return _Rewrite(
        source_text=_splice_expressions(
            source_text,
            (
                (target.call.func, rewritten_callable),
                (target.loop.iter, rewritten_iterable),
            ),
        ),
        guard_expression=guard_expression,
    )


def _validate_source_hash(
    source_text: str,
    tree: ast.Module,
    plan: ExecutionPlan,
) -> None:
    digest = hashlib.sha256()
    lines = source_text.splitlines()
    for qualname in sorted(_planned_symbol_qualnames(plan)):
        node = _function_node(tree, qualname)
        if node is None:
            raise ValueError(f"planned symbol is missing from payload source: {qualname}")
        start = _declaration_start(node)
        digest.update(f"{plan.source_module}::{qualname}".encode())
        digest.update(b"\0")
        digest.update("\n".join(lines[start - 1 : node.end_lineno]).encode())
        digest.update(b"\0")
    if digest.hexdigest() != plan.source_hash:
        raise ValueError("payload source hash does not match the selected execution plan")


def _validate_callsite_fingerprint(tree: ast.Module, plan: ExecutionPlan) -> None:
    owner = _function_node(tree, plan.owner.qualname)
    if owner is None:
        raise ValueError(f"plan owner is missing from payload source: {plan.owner.qualname}")
    parts: list[str] = []
    for node in ast.walk(owner):
        if not isinstance(node, ast.Call):
            continue
        callee = _spawn_callee(node)
        if callee is None:
            continue
        scheduler = _create_task_scheduler(node)
        if scheduler is None:
            continue
        parts.append(f"asyncio:{node.lineno}:{node.col_offset}:{callee}")
    if _digest_parts(parts) != plan.callsite_fingerprint:
        raise ValueError("payload call-site fingerprint does not match the selected plan")


def _rewrite_target(tree: ast.Module, plan: ExecutionPlan) -> _LoopRewriteTarget:
    owner = _function_node(tree, plan.owner.qualname)
    if owner is None:
        raise ValueError(f"plan owner is missing from payload source: {plan.owner.qualname}")
    targets: list[_LoopRewriteTarget] = []
    for loop in (node for node in ast.walk(owner) if isinstance(node, ast.For | ast.AsyncFor)):
        if _loop_contains_planned_create_task(plan, loop):
            _reject_dynamic_loop_shape(loop)
        for statement in loop.body:
            if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
                continue
            scheduler_name = _create_task_scheduler(statement.value)
            callee = _spawn_callee(statement.value)
            if scheduler_name is None or callee is None:
                continue
            if not _callsite_lineno_is_planned(plan, statement.value.lineno):
                continue
            _reject_dynamic_loop_shape(loop)
            targets.append(
                _LoopRewriteTarget(
                    function=owner,
                    loop=loop,
                    statement=statement,
                    call=statement.value,
                    scheduler_name=scheduler_name,
                )
            )
    if len(targets) != 1:
        raise ValueError(f"expected exactly one loop create_task call, found {len(targets)}")
    return targets[0]


def _loop_contains_planned_create_task(plan: ExecutionPlan, loop: ast.For | ast.AsyncFor) -> bool:
    return any(
        isinstance(node, ast.Call)
        and _create_task_scheduler(node) is not None
        and _callsite_lineno_is_planned(plan, node.lineno)
        for node in ast.walk(loop)
    )


def _reject_dynamic_loop_shape(loop: ast.For | ast.AsyncFor) -> None:
    for node in ast.walk(loop):
        if isinstance(node, ast.Assign | ast.AnnAssign | ast.AugAssign | ast.NamedExpr):
            value = (
                node.value if isinstance(node, ast.Assign | ast.AnnAssign | ast.NamedExpr) else None
            )
            if isinstance(value, ast.Call) and _create_task_scheduler(value) is not None:
                raise ValueError("create_task result must not be assigned or captured")
        if (
            isinstance(node, ast.Call)
            and _create_task_scheduler(node) is not None
            and _spawn_callee(node) is None
        ):
            raise ValueError("create_task argument must be a direct coroutine call")


def _reject_scheduler_reassignment(loop: ast.For, scheduler_name: str) -> None:
    """Reject loop bodies that replace or delete the scheduler binding.

    Args:
        loop: Candidate fan-out loop.
        scheduler_name: Local task-group receiver whose descriptor would be hoisted.

    Raises:
        ValueError: If the loop can rebind or delete the scheduler local.
    """
    for node in ast.walk(loop):
        targets: tuple[ast.expr, ...] = ()
        if isinstance(node, ast.Assign):
            targets = tuple(node.targets)
        elif isinstance(node, ast.AnnAssign | ast.AugAssign | ast.NamedExpr):
            targets = (node.target,)
        elif isinstance(node, ast.Delete):
            targets = tuple(node.targets)
        if any(_target_binds_name(target, scheduler_name) for target in targets):
            raise ValueError("scheduler binding must remain stable throughout the fan-out loop")


def _target_binds_name(target: ast.expr, name: str) -> bool:
    """Return whether a direct or destructuring target binds one name.

    Args:
        target: Assignment or deletion target.
        name: Local binding that must remain stable.

    Returns:
        bool: Whether the target contains the requested name.
    """
    return any(isinstance(node, ast.Name) and node.id == name for node in ast.walk(target))


def _callsite_lineno_is_planned(plan: ExecutionPlan, lineno: int) -> bool:
    return any(edge.kind == "spawns" and edge.lineno == lineno for edge in plan.edges)


def _splice_expressions(
    source_text: str,
    replacements: tuple[tuple[ast.expr, str], ...],
) -> str:
    """Replace same-line expressions without changing unrelated source bytes.

    Args:
        source_text: Original payload module source.
        replacements: AST expressions and their generated single-line replacements.

    Returns:
        str: Source with only the requested expression ranges changed.

    Raises:
        ValueError: If a replacement spans lines or has incomplete coordinates.
    """
    source_bytes = source_text.encode("utf-8")
    spans: list[tuple[int, int, bytes]] = []
    for node, replacement in replacements:
        end_lineno = getattr(node, "end_lineno", None)
        end_col_offset = getattr(node, "end_col_offset", None)
        if (
            not isinstance(end_lineno, int)
            or end_lineno != node.lineno
            or not isinstance(end_col_offset, int)
            or "\n" in replacement
        ):
            raise ValueError("task-preserving rewrite requires same-line expression coordinates")
        spans.append(
            (
                _byte_offset(source_bytes, node.lineno, node.col_offset),
                _byte_offset(source_bytes, end_lineno, end_col_offset),
                replacement.encode("utf-8"),
            )
        )
    for start, end, replacement_bytes in sorted(spans, reverse=True):
        source_bytes = source_bytes[:start] + replacement_bytes + source_bytes[end:]
    return source_bytes.decode("utf-8")


def _byte_offset(source: bytes, lineno: int, col_offset: int) -> int:
    """Convert AST UTF-8 line and byte-column coordinates into a byte offset.

    Args:
        source: Full UTF-8 encoded module source.
        lineno: One-based AST line number.
        col_offset: Zero-based AST UTF-8 byte column.

    Returns:
        int: Absolute byte offset into `source`.

    Raises:
        ValueError: If the line or column lies outside the source.
    """
    lines = source.splitlines(keepends=True)
    if lineno < 1 or lineno > len(lines) or col_offset > len(lines[lineno - 1]):
        raise ValueError("task-preserving source coordinates are outside the payload")
    return sum(len(line) for line in lines[: lineno - 1]) + col_offset


def _module_path(root: Path, module: str) -> Path | None:
    parts = module.split(".")
    module_path = root.joinpath(*parts).with_suffix(".py")
    package_path = root.joinpath(*parts, "__init__.py")
    if module_path.is_file():
        return module_path
    if package_path.is_file():
        return package_path
    return None


def _planned_symbol_qualnames(plan: ExecutionPlan) -> set[str]:
    return {
        node.symbol.qualname
        for node in plan.nodes
        if node.symbol is not None and node.symbol.module == plan.source_module
    }


def _function_node(
    tree: ast.Module,
    qualname: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    parts = qualname.split(".")
    if len(parts) == 1:
        for node in tree.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == qualname:
                return node
        return None
    if len(parts) == _QUALNAME_CLASS_METHOD_PARTS:
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == parts[0]:
                for child in node.body:
                    if (
                        isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
                        and child.name == parts[1]
                    ):
                        return child
    return None


def _declaration_start(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    decorated_lines = [decorator.lineno for decorator in node.decorator_list]
    return min((*decorated_lines, node.lineno))


def _create_task_scheduler(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Attribute) and node.func.attr == "create_task":
        if isinstance(node.func.value, ast.Name):
            return node.func.value.id
        return None
    return None


def _spawn_callee(node: ast.Call) -> str | None:
    if not node.args or not isinstance(node.args[0], ast.Call):
        return None
    path = _attribute_path(node.args[0].func)
    return ".".join(path) if path is not None else None


def _attribute_path(node: ast.expr) -> tuple[str, ...] | None:
    """Return the lexical dotted path for one name or attribute expression.

    Args:
        node: Expression to resolve without importing target code.

    Returns:
        tuple[str, ...] | None: Lexical path components, or `None` for dynamic expressions.
    """
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        parent = _attribute_path(node.value)
        return (*parent, node.attr) if parent is not None else None
    return None


def _name_exists(tree: ast.Module, name: str) -> bool:
    return any(
        (isinstance(node, ast.Name) and node.id == name)
        or (
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
            and node.name == name
        )
        for node in ast.walk(tree)
    )


def _name_is_bound(
    tree: ast.Module,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    name: str,
) -> bool:
    """Return whether generated builtin references would resolve to a shadowing binding.

    Args:
        tree: Parsed payload module.
        function: Orchestration function receiving generated expressions.
        name: Builtin name required by the runtime guard.

    Returns:
        bool: Whether module or function scope binds the requested name.
    """
    module_scope = tuple(node for node in tree.body if node is not function)
    return (
        _scope_binds_name(module_scope, name)
        or _scope_binds_name(function.body, name)
        or any(
            argument.arg == name
            for argument in (
                *function.args.posonlyargs,
                *function.args.args,
                *function.args.kwonlyargs,
                *((function.args.vararg,) if function.args.vararg is not None else ()),
                *((function.args.kwarg,) if function.args.kwarg is not None else ()),
            )
        )
    )


def _scope_binds_name(nodes: tuple[ast.stmt, ...] | list[ast.stmt], name: str) -> bool:
    """Return whether statements contain a lexical binding for one name.

    Args:
        nodes: Statements in one module or function scope.
        name: Binding name to detect.

    Returns:
        bool: Whether an assignment, import, or declaration binds `name`.
    """
    visitor = _BindingVisitor(name)
    for node in nodes:
        visitor.visit(node)
        if visitor.found:
            return True
    return False


class _BindingVisitor(ast.NodeVisitor):
    """Find one lexical binding without descending into nested scopes."""

    def __init__(self, name: str) -> None:
        self._name = name
        self.found = False

    def visit_Name(self, node: ast.Name) -> None:
        """Record assignment targets in the current lexical scope.

        Args:
            node: Name expression visited in the current scope.
        """
        if isinstance(node.ctx, ast.Store) and node.id == self._name:
            self.found = True

    def visit_Import(self, node: ast.Import) -> None:
        """Record an import binding without importing target code.

        Args:
            node: Import statement visited in the current scope.
        """
        if any((alias.asname or alias.name.split(".")[0]) == self._name for alias in node.names):
            self.found = True

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Record a from-import binding without importing target code.

        Args:
            node: From-import statement visited in the current scope.
        """
        if any((alias.asname or alias.name) == self._name for alias in node.names):
            self.found = True

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Record the nested definition name but do not enter its local scope.

        Args:
            node: Nested synchronous function definition.
        """
        if node.name == self._name:
            self.found = True

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Record the nested definition name but do not enter its local scope.

        Args:
            node: Nested coroutine or async-generator definition.
        """
        if node.name == self._name:
            self.found = True

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Record the class binding but do not enter its namespace.

        Args:
            node: Nested class definition.
        """
        if node.name == self._name:
            self.found = True

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Do not treat lambda-local parameters as outer-scope bindings.

        Args:
            node: Lambda expression whose local scope is deliberately skipped.
        """
        del node


def _binding_name(plan_id: str) -> str:
    suffix = re.sub(r"[^A-Za-z0-9_]", "_", plan_id.rsplit("-", maxsplit=1)[-1])
    return f"_atoll_create_task_{suffix}"


def _digest_parts(parts: list[str]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


TASK_PRESERVING_BACKEND: Final = TaskPreservingExecutionPlanBackend()
"""Shared task-preserving execution-plan backend instance for command integration."""
