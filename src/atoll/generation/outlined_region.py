"""Generate outlined synchronous helpers for suspension-owning callables.

Milestone 4 keeps coroutine and generator suspension protocols in Python while
moving planner-approved synchronous statement islands into private helper
functions. This module does not mutate source modules directly. It writes the
helper source requested for a staged native build and returns an
``OutlinedShellConfig`` that the managed region shim can execute in the staged
wheel payload.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from atoll.analysis.suspension_planner import (
    StatementEvidence,
    SuspensionBlock,
    SuspensionPlan,
    plan_suspension_blocks,
)
from atoll.generation.region_shim import OutlinedShellConfig
from atoll.models import BindingTarget, RegionMember, SymbolId, TypedRegion

OUTLINED_REGION_GENERATOR_VERSION: Final = "outlined-region-v1"
_CONTINUE_TAG: Final = "continue"
_FACTORY_NAME: Final = "_atoll_make_outlined_shell"
_NATIVE_ARGUMENT: Final = "_atoll_native"
_RESOLVER_ARGUMENT: Final = "_atoll_resolve_global"
_BUILTINS_LOCAL: Final = "_atoll_runtime_builtins"
_GLOBALS_LOCAL: Final = "_atoll_runtime_globals"


@dataclass(frozen=True, slots=True)
class OutlinedRegionGeneration:
    """Generated helper source and Python shell contract for one outlined member.

    Attributes:
        region: Typed region that owns the selected suspension callable.
        member: Region member whose synchronous blocks were outlined.
        plan: Conservative suspension plan used to select block helpers.
        source_path: Path where the staged helper source was written.
        source_text: Deterministic synchronous helper source intended for native compilation.
        source_hash: SHA-256 digest of ``source_text`` and generator version.
        binding: Public binding restored by the region shim.
        shell: Staged-only Python shell factory configuration.
        helper_names: Private native helper names required by the shell.
    """

    region: TypedRegion
    member: RegionMember
    plan: SuspensionPlan
    source_path: Path
    source_text: str
    source_hash: str
    binding: BindingTarget
    shell: OutlinedShellConfig
    helper_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PlannedHelper:
    """Internal helper metadata shared by native source and shell rewriting.

    Attributes:
        block: Planner-approved synchronous block represented by the helper.
        name: Deterministic private native helper name.
        arguments: Explicit live-ins passed from the Python shell.
        global_dependencies: Late-bound global names resolved at each native read.
        live_outs: Local values returned to the Python shell.
        statements: Parsed source statements copied into the helper.
    """

    block: SuspensionBlock
    name: str
    arguments: tuple[str, ...]
    global_dependencies: tuple[str, ...]
    live_outs: tuple[str, ...]
    statements: tuple[ast.stmt, ...]


def generate_outlined_region(
    region: TypedRegion,
    member_id: SymbolId,
    binding: BindingTarget,
    *,
    output_path: Path,
) -> OutlinedRegionGeneration:
    """Generate synchronous helpers and a staged Python suspension shell.

    Args:
        region: Typed region containing the selected suspension-owning callable.
        member_id: Stable identity of the callable to outline.
        binding: Public runtime binding that the generated shell will replace.
        output_path: File path for the generated synchronous helper module source.

    Returns:
        OutlinedRegionGeneration: Immutable helper source, shell config, plan, and hashes.

    Raises:
        ValueError: If the region, binding, source syntax, or planner evidence cannot be
            outlined without changing suspension semantics.
    """
    member = _member(region, member_id)
    _validate_member_shape(region, member, binding)
    _reject_shell_blockers(member)

    plan = plan_suspension_blocks(member)
    if plan.rejections:
        messages = ", ".join(rejection.code for rejection in plan.rejections)
        raise ValueError(f"outlined region rejected by suspension planner: {messages}")
    eligible_blocks = tuple(block for block in plan.blocks if block.id in plan.eligible_block_ids)
    if not eligible_blocks:
        raise ValueError("outlined region requires at least one eligible suspension block")

    callable_node = _single_callable(member.source_text)
    helpers = tuple(
        _PlannedHelper(
            block=block,
            name=_helper_name(member, index, block),
            arguments=tuple(
                name for name in block.live_ins if name not in block.late_bound_globals
            ),
            global_dependencies=block.late_bound_globals,
            live_outs=block.live_outs,
            statements=_statements_for_block(callable_node, block),
        )
        for index, block in enumerate(eligible_blocks)
    )
    _reject_name_collisions(callable_node, helpers)

    helper_source = _helper_source(helpers)
    shell_source = _shell_source(callable_node, helpers, binding)
    helper_names = tuple(helper.name for helper in helpers)
    shell = OutlinedShellConfig(
        factory_name=_FACTORY_NAME,
        factory_source=shell_source,
        helper_names=helper_names,
    )
    source_hash = _source_hash(helper_source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(helper_source, encoding="utf-8")

    return OutlinedRegionGeneration(
        region=region,
        member=member,
        plan=plan,
        source_path=output_path,
        source_text=helper_source,
        source_hash=source_hash,
        binding=binding,
        shell=shell,
        helper_names=helper_names,
    )


def _member(region: TypedRegion, member_id: SymbolId) -> RegionMember:
    matches = tuple(member for member in region.members if member.id == member_id)
    if len(matches) != 1:
        raise ValueError(f"outlined region requires exactly one selected member: {member_id}")
    return matches[0]


def _validate_member_shape(
    region: TypedRegion,
    member: RegionMember,
    binding: BindingTarget,
) -> None:
    if member.execution_kind not in {"coroutine", "generator", "async_generator"}:
        raise ValueError("outlined region requires a coroutine or generator member")
    if binding.source != member.id:
        raise ValueError("outlined region binding does not target the selected member")
    if binding not in region.bindings:
        raise ValueError("outlined region binding is not promised by the region")
    if binding.execution_kind != member.execution_kind:
        raise ValueError("outlined region binding execution kind does not match the member")
    if binding.kind != member.binding_kind or binding.owner_class != member.owner_class:
        raise ValueError("outlined region binding kind or owner does not match the member")


def _reject_shell_blockers(member: RegionMember) -> None:
    callable_node = _single_callable(member.source_text)
    decorator_names = tuple(
        _decorator_name(decorator) for decorator in callable_node.decorator_list
    )
    unknown_decorators = tuple(
        ast.unparse(decorator)
        for decorator, name in zip(
            callable_node.decorator_list,
            decorator_names,
            strict=True,
        )
        if name not in {"staticmethod", "classmethod"}
    )
    if unknown_decorators:
        raise ValueError(
            f"outlined region rejects unknown decorator(s): {', '.join(unknown_decorators)}"
        )
    expected_decorators = {
        "staticmethod": ("staticmethod",),
        "classmethod": ("classmethod",),
    }.get(member.binding_kind, ())
    if decorator_names != expected_decorators:
        raise ValueError("outlined region descriptor decorators do not match the binding kind")
    for node in ast.walk(callable_node):
        if isinstance(node, ast.Name) and node.id == "__class__":
            raise ValueError("outlined region rejects __class__ closure dependencies")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "super"
        ):
            raise ValueError("outlined region rejects super() dependencies")


def _decorator_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    return ast.unparse(node)


def _single_callable(source_text: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    module = ast.parse(textwrap.dedent(source_text).strip("\n"))
    declarations = tuple(
        node for node in module.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    )
    if len(declarations) != 1:
        raise ValueError("outlined region source must contain exactly one callable declaration")
    return declarations[0]


def _reject_name_collisions(
    callable_node: ast.FunctionDef | ast.AsyncFunctionDef,
    helpers: tuple[_PlannedHelper, ...],
) -> None:
    generated_names = {
        _FACTORY_NAME,
        _NATIVE_ARGUMENT,
        _RESOLVER_ARGUMENT,
        _BUILTINS_LOCAL,
        _GLOBALS_LOCAL,
        *(helper.name for helper in helpers),
        *(f"_atoll_outlined_result_{helper.name}" for helper in helpers),
        *(f"_atoll_outlined_control_{helper.name}" for helper in helpers),
    }
    source_names = {
        node.id
        for node in ast.walk(callable_node)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Load)
    }
    parameter_names = {argument.arg for argument in _all_arguments(callable_node.args)}
    if generated_names & (source_names | parameter_names):
        raise ValueError("outlined region generated names collide with source names")


def _helper_name(member: RegionMember, index: int, block: SuspensionBlock) -> str:
    digest = hashlib.sha256(
        "\0".join(
            (
                OUTLINED_REGION_GENERATOR_VERSION,
                member.id.stable_id,
                block.id,
                str(index),
            )
        ).encode()
    ).hexdigest()[:10]
    qualname = member.id.qualname.replace(".", "__")
    return f"_{qualname}__outlined_{index}_{digest}"


def _helper_source(helpers: tuple[_PlannedHelper, ...]) -> str:
    module = ast.Module(
        body=[_helper_function(helper) for helper in helpers],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    return f"{ast.unparse(module)}\n"


def _helper_function(helper: _PlannedHelper) -> ast.FunctionDef:
    returns: list[ast.expr] = [ast.Constant(value=_CONTINUE_TAG)]
    returns.extend(ast.Name(id=name, ctx=ast.Load()) for name in helper.live_outs)
    body: list[ast.stmt] = [
        *(
            _lower_helper_statement(statement, helper.global_dependencies)
            for statement in helper.statements
        ),
        ast.Return(value=ast.Tuple(elts=returns, ctx=ast.Load())),
    ]
    return ast.FunctionDef(
        name=helper.name,
        args=_arguments((_RESOLVER_ARGUMENT, *helper.arguments)),
        body=body,
        decorator_list=[],
        returns=None,
        type_comment=None,
        type_params=[],
    )


class _LateBoundGlobalRewriter(ast.NodeTransformer):
    """Resolve every global load against the source module at read time."""

    def __init__(self, global_dependencies: tuple[str, ...]) -> None:
        self._global_dependencies = frozenset(global_dependencies)

    def visit_Name(self, node: ast.Name) -> ast.expr:
        """Replace one global load with a call to the explicit resolver.

        Args:
            node: Name expression visited in a copied helper statement.

        Returns:
            ast.expr: Original local name or a late-bound resolver call.
        """
        if not isinstance(node.ctx, ast.Load) or node.id not in self._global_dependencies:
            return node
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id=_RESOLVER_ARGUMENT, ctx=ast.Load()),
                args=[ast.Constant(value=node.id)],
                keywords=[],
            ),
            node,
        )


def _lower_helper_statement(
    statement: ast.stmt,
    global_dependencies: tuple[str, ...],
) -> ast.stmt:
    lowered = _LateBoundGlobalRewriter(global_dependencies).visit(copy.deepcopy(statement))
    if not isinstance(lowered, ast.stmt):
        raise TypeError("outlined global rewriting removed a helper statement")
    return lowered


def _shell_source(
    callable_node: ast.FunctionDef | ast.AsyncFunctionDef,
    helpers: tuple[_PlannedHelper, ...],
    binding: BindingTarget,
) -> str:
    shell_node = copy.deepcopy(callable_node)
    _strip_callable_metadata(shell_node)
    rewrite_count = _BlockRewriter(helpers).rewrite(shell_node)
    if rewrite_count != len(helpers):
        raise ValueError("outlined region did not rewrite every eligible block")

    if binding.owner_class is None:
        shell_body: list[ast.stmt] = [shell_node, _return_shell_function(shell_node.name)]
    else:
        owner_leaf = binding.owner_class.rsplit(".", maxsplit=1)[-1]
        owner_class = ast.ClassDef(
            name=owner_leaf,
            bases=[],
            keywords=[],
            body=[shell_node],
            decorator_list=[],
            type_params=[],
        )
        shell_body = [owner_class, _return_owner_function(owner_leaf, shell_node.name)]

    factory_body = [*_resolver_factory_statements(), *shell_body]

    factory = ast.FunctionDef(
        name=_FACTORY_NAME,
        args=_arguments((_NATIVE_ARGUMENT,)),
        body=factory_body,
        decorator_list=[],
        returns=None,
        type_comment=None,
        type_params=[],
    )
    module = ast.Module(body=[factory], type_ignores=[])
    ast.fix_missing_locations(module)
    return f"{ast.unparse(module)}\n"


def _resolver_factory_statements() -> tuple[ast.stmt, ...]:
    module = ast.parse(
        f"""import builtins as {_BUILTINS_LOCAL}

def {_RESOLVER_ARGUMENT}(name):
    {_GLOBALS_LOCAL} = {_BUILTINS_LOCAL}.globals()
    if name in {_GLOBALS_LOCAL}:
        return {_GLOBALS_LOCAL}[name]
    try:
        return {_BUILTINS_LOCAL}.getattr({_BUILTINS_LOCAL}, name)
    except AttributeError:
        raise {_BUILTINS_LOCAL}.NameError(
            f"name '{{name}}' is not defined"
        ) from None
"""
    )
    return tuple(module.body)


def _statements_for_block(
    callable_node: ast.FunctionDef | ast.AsyncFunctionDef,
    block: SuspensionBlock,
) -> tuple[ast.stmt, ...]:
    for parent in ast.walk(callable_node):
        for _field_name, value in ast.iter_fields(parent):
            statement_body = _statement_body(value)
            if statement_body is not None:
                for index in range(len(statement_body)):
                    candidate = statement_body[index : index + len(block.statements)]
                    if len(candidate) != len(block.statements):
                        continue
                    if all(
                        _statement_matches(statement, item)
                        for statement, item in zip(candidate, block.statements, strict=True)
                    ):
                        return tuple(candidate)
    raise ValueError("outlined region could not recover planned block statements")


def _strip_callable_metadata(node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
    node.decorator_list = []
    node.returns = None
    if hasattr(node, "type_params"):
        node.type_params = []
    for argument in _all_arguments(node.args):
        argument.annotation = None
    node.args.defaults = [ast.Constant(value=None) for _ in node.args.defaults]
    node.args.kw_defaults = [
        ast.Constant(value=None) if default is not None else None
        for default in node.args.kw_defaults
    ]


class _BlockRewriter:
    """Replace source-ordered statement sequences with helper calls."""

    def __init__(self, helpers: tuple[_PlannedHelper, ...]) -> None:
        self._helpers = helpers
        self._rewritten = 0

    def rewrite(self, node: ast.AST) -> int:
        """Rewrite all nested statement bodies in ``node``.

        Args:
            node: Callable AST whose statement bodies should be transformed.

        Returns:
            int: Number of helper blocks replaced.
        """
        for parent in ast.walk(node):
            for field_name, value in ast.iter_fields(parent):
                statement_body = _statement_body(value)
                if statement_body is not None:
                    replacement = self._rewrite_body(statement_body)
                    setattr(parent, field_name, replacement)
        return self._rewritten

    def _rewrite_body(self, body: list[ast.stmt]) -> list[ast.stmt]:
        rewritten: list[ast.stmt] = []
        index = 0
        while index < len(body):
            helper = self._matching_helper(body, index)
            if helper is None:
                rewritten.append(body[index])
                index += 1
            else:
                rewritten.extend(_helper_call_statements(helper))
                index += len(helper.block.statements)
                self._rewritten += 1
        return rewritten

    def _matching_helper(self, body: list[ast.stmt], index: int) -> _PlannedHelper | None:
        for helper in self._helpers:
            evidence = helper.block.statements
            candidate = body[index : index + len(evidence)]
            if len(candidate) != len(evidence):
                continue
            if all(
                _statement_matches(statement, item)
                for statement, item in zip(candidate, evidence, strict=True)
            ):
                return helper
        return None


def _statement_matches(statement: ast.stmt, evidence: StatementEvidence) -> bool:
    return (
        statement.lineno,
        statement.col_offset,
        statement.end_lineno or statement.lineno,
        statement.end_col_offset or statement.col_offset,
    ) == (
        evidence.start_lineno,
        evidence.start_col_offset,
        evidence.end_lineno,
        evidence.end_col_offset,
    )


def _statement_body(value: object) -> list[ast.stmt] | None:
    if not isinstance(value, list):
        return None
    items = cast(list[object], value)
    if not all(isinstance(item, ast.stmt) for item in items):
        return None
    return cast(list[ast.stmt], items)


def _helper_call_statements(helper: _PlannedHelper) -> list[ast.stmt]:
    result_name = f"_atoll_outlined_result_{helper.name}"
    control_name = f"_atoll_outlined_control_{helper.name}"
    call = ast.Call(
        func=ast.Attribute(
            value=ast.Name(id=_NATIVE_ARGUMENT, ctx=ast.Load()),
            attr=helper.name,
            ctx=ast.Load(),
        ),
        args=[
            ast.Name(id=_RESOLVER_ARGUMENT, ctx=ast.Load()),
            *(ast.Name(id=name, ctx=ast.Load()) for name in helper.arguments),
        ],
        keywords=[],
    )
    result_assign = ast.Assign(
        targets=[ast.Name(id=result_name, ctx=ast.Store())],
        value=call,
    )
    control_assign = ast.Assign(
        targets=[ast.Name(id=control_name, ctx=ast.Store())],
        value=ast.Subscript(
            value=ast.Name(id=result_name, ctx=ast.Load()),
            slice=ast.Constant(value=0),
            ctx=ast.Load(),
        ),
    )
    live_out_assigns = [
        ast.Assign(
            targets=[ast.Name(id=name, ctx=ast.Store())],
            value=ast.Subscript(
                value=ast.Name(id=result_name, ctx=ast.Load()),
                slice=ast.Constant(value=index),
                ctx=ast.Load(),
            ),
        )
        for index, name in enumerate(helper.live_outs, start=1)
    ]
    control_check = ast.If(
        test=ast.Compare(
            left=ast.Name(id=control_name, ctx=ast.Load()),
            ops=[ast.NotEq()],
            comparators=[ast.Constant(value=_CONTINUE_TAG)],
        ),
        body=[
            ast.Raise(
                exc=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id=_BUILTINS_LOCAL, ctx=ast.Load()),
                        attr="RuntimeError",
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.Constant(value="outlined helper returned an unsupported control tag")
                    ],
                    keywords=[],
                ),
                cause=None,
            )
        ],
        orelse=[],
    )
    return [result_assign, control_assign, *live_out_assigns, control_check]


def _return_shell_function(name: str) -> ast.Return:
    return ast.Return(value=ast.Name(id=name, ctx=ast.Load()))


def _return_owner_function(owner: str, name: str) -> ast.Return:
    return ast.Return(
        value=ast.Subscript(
            value=ast.Attribute(
                value=ast.Name(id=owner, ctx=ast.Load()),
                attr="__dict__",
                ctx=ast.Load(),
            ),
            slice=ast.Constant(value=name),
            ctx=ast.Load(),
        )
    )


def _arguments(names: tuple[str, ...]) -> ast.arguments:
    return ast.arguments(
        posonlyargs=[],
        args=[ast.arg(arg=name, annotation=None) for name in names],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[],
    )


def _all_arguments(arguments: ast.arguments) -> tuple[ast.arg, ...]:
    return (
        *arguments.posonlyargs,
        *arguments.args,
        *((arguments.vararg,) if arguments.vararg is not None else ()),
        *arguments.kwonlyargs,
        *((arguments.kwarg,) if arguments.kwarg is not None else ()),
    )


def _source_hash(source_text: str) -> str:
    digest = hashlib.sha256()
    digest.update(OUTLINED_REGION_GENERATOR_VERSION.encode())
    digest.update(b"\0")
    digest.update(source_text.encode())
    return digest.hexdigest()
