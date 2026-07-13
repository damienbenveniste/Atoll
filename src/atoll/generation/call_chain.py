"""Generate proof-authorized Cython units for direct scalar call chains.

The generated module keeps one Python-visible root used by the staged runtime
dispatcher and lowers every proven helper to a private ``cdef inline``
function. Call arguments are bound statically, including source defaults and
keywords, so native calls remain positional and unboxed. Generated `.pyx`
files stay under the disposable source-clean build root.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from atoll.generation.scalar_kernel import (
    SCALAR_KERNEL_GENERATOR_VERSION,
    ScalarPowerExpansion,
    render_scalar_statement,
    scalar_binding,
    scalar_c_type,
    scalar_callable_node,
    scalar_local_integer_names,
    scalar_wrapper_parameter_declarations,
)
from atoll.generation.typed_region import TypedRegionGeneration
from atoll.models import ModuleScan, RegionMember, SymbolId, TypedRegion
from atoll.native_optimization.call_chains import (
    CallChainFieldBinding,
    CallChainPlan,
    bind_member_call_arguments,
    call_chain_scalar_parameters,
    resolve_callee,
)
from atoll.native_optimization.scalar_analysis import ScalarWidthProof

CALL_CHAIN_GENERATOR_VERSION = "atoll-call-chain-v1"


@dataclass(frozen=True, slots=True)
class CallChainGeneration:
    """Generated Cython source and private call-chain symbol metadata.

    Attributes:
        generation: Standard typed-region generation contract used by packaging.
        plan: Static call-chain topology and scalar proof authorizing lowering.
        width_proof: Exact native width and pre-entry integer guards.
        compiled_name: Python-visible private root loaded by the staged dispatcher.
        helper_names: Cython-only helper names in leaves-first order.
        c_type: Fixed-width C integer type used throughout the chain.
    """

    generation: TypedRegionGeneration
    plan: CallChainPlan
    width_proof: ScalarWidthProof
    compiled_name: str
    helper_names: tuple[str, ...]
    c_type: str


@dataclass(frozen=True, slots=True)
class CallChainGenerationRequest:
    """Staged call-chain evidence and destination for one width variant.

    Attributes:
        scan: Module scan containing exact root, helper, and constant source.
        region: Typed region containing the complete direct call chain.
        plan: Revalidated static topology and flattened scalar proof.
        width_proof: Selected 32-bit or 64-bit proof.
        logical_module: Private extension module name used by the wheel payload.
        output_path: Disposable proof-generated `.pyx` path.
    """

    scan: ModuleScan
    region: TypedRegion
    plan: CallChainPlan
    width_proof: ScalarWidthProof
    logical_module: str
    output_path: Path


@dataclass(frozen=True, slots=True)
class _CallChainRenderContext:
    request: CallChainGenerationRequest
    root: RegionMember
    helpers: tuple[RegionMember, ...]
    nodes: dict[SymbolId, ast.FunctionDef]
    compiled_name: str
    helper_names: dict[SymbolId, str]
    field_bindings: tuple[CallChainFieldBinding, ...]
    c_type: str


class _NativeCallRewriter(ast.NodeTransformer):
    def __init__(
        self,
        members: dict[SymbolId, RegionMember],
        nodes: dict[SymbolId, ast.FunctionDef],
        helper_names: dict[SymbolId, str],
        field_bindings: tuple[CallChainFieldBinding, ...],
        caller: RegionMember,
    ) -> None:
        self._members = members
        self._nodes = nodes
        self._helper_names = helper_names
        self._field_bindings = field_bindings
        self._caller = caller

    def visit_Call(self, node: ast.Call) -> ast.expr:
        """Rewrite one direct helper call to its positional private Cython target.

        Args:
            node: Call expression from the proven root or helper body.

        Returns:
            ast.expr: Rewritten private call or recursively visited builtins.range call.

        Raises:
            ValueError: If generated source no longer matches the proven direct-call topology.
        """
        visited = self.generic_visit(node)
        if not isinstance(visited, ast.Call):
            raise TypeError("call-chain rewrite produced a non-call expression")
        if isinstance(visited.func, ast.Name) and visited.func.id == "range":
            return visited
        callee = resolve_callee(self._caller, visited.func, self._members)
        if callee is None or callee not in self._helper_names:
            raise ValueError(f"unresolved generated call-chain target: {ast.unparse(visited.func)}")
        member = self._members[callee]
        bound = bind_member_call_arguments(member, self._nodes[callee], visited)
        parameters = call_chain_scalar_parameters(member)
        arguments = [
            *(ast.Name(id=item.synthetic_name, ctx=ast.Load()) for item in self._field_bindings),
            *(copy.deepcopy(bound[parameter.name]) for parameter in parameters),
        ]
        rewritten = ast.Call(
            func=ast.Name(id=self._helper_names[callee], ctx=ast.Load()),
            args=arguments,
            keywords=[],
        )
        return ast.copy_location(rewritten, node)


class _GeneratedFieldReadSubstitution(ast.NodeTransformer):
    def __init__(self, bindings: tuple[CallChainFieldBinding, ...]) -> None:
        self._names = {binding.field_name: binding.synthetic_name for binding in bindings}

    def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
        """Replace one analyzed receiver-field read with its private native input.

        Args:
            node: Attribute expression reached during generated-source rewriting.

        Returns:
            ast.expr: Synthetic field input or recursively visited expression.

        Raises:
            ValueError: If the receiver field was not authorized by the static plan.
            TypeError: If AST rewriting produces a non-expression value.
        """
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            if not isinstance(node.ctx, ast.Load) or node.attr not in self._names:
                raise ValueError(f"unproven generated receiver attribute: self.{node.attr}")
            return ast.copy_location(ast.Name(id=self._names[node.attr], ctx=ast.Load()), node)
        visited = self.generic_visit(node)
        if not isinstance(visited, ast.expr):
            raise TypeError("generated field substitution produced a non-expression")
        return visited


def generate_call_chain_kernel(request: CallChainGenerationRequest) -> CallChainGeneration:
    """Write one fixed-width Cython unit with private inline helpers.

    Args:
        request: Staged scan, region, plan, proof, module identity, and output path.

    Returns:
        CallChainGeneration: Generated source and binding metadata.

    Raises:
        ValueError: If output provenance, region topology, or retained source hashes drift.
    """
    if request.output_path.suffix != ".pyx":
        raise ValueError("call-chain kernels require a .pyx output path")
    if request.region.id != request.plan.region_id:
        raise ValueError("staged call-chain region differs from analyzed region")
    if request.width_proof not in request.plan.scalar_plan.width_proofs:
        raise ValueError("staged call-chain width proof was not authorized by the analyzed plan")
    members = {member.id: member for member in request.region.members}
    selected_ids = (request.plan.root, *request.plan.helpers)
    selected = tuple(members.get(member_id) for member_id in selected_ids)
    if any(member is None for member in selected):
        raise ValueError("staged call-chain member is absent from analyzed region")
    retained = tuple(member for member in selected if member is not None)
    expected_hashes = dict(request.plan.source_hashes)
    if any(_source_hash(member) != expected_hashes.get(member.id) for member in retained):
        raise ValueError("staged call-chain source differs from analyzed source")
    root = retained[0]
    helpers = retained[1:]
    nodes = {member.id: scalar_callable_node(member) for member in retained}
    c_type = scalar_c_type(request.width_proof)
    width_label = (
        f"{'i' if request.width_proof.native.signed else 'u'}{request.width_proof.native.width}"
    )
    compiled_name = f"_atoll_chain_{_safe_name(root.id.qualname)}_{width_label}"
    helper_names = {
        helper.id: f"_atoll_inline_{_safe_name(helper.id.qualname)}_{width_label}"
        for helper in helpers
    }
    source_text = _render_source(
        _CallChainRenderContext(
            request=request,
            root=root,
            helpers=helpers,
            nodes=nodes,
            compiled_name=compiled_name,
            helper_names=helper_names,
            field_bindings=request.plan.field_bindings,
            c_type=c_type,
        )
    )
    request.output_path.parent.mkdir(parents=True, exist_ok=True)
    request.output_path.write_text(source_text, encoding="utf-8")
    generation = TypedRegionGeneration(
        region=request.region,
        logical_module=request.logical_module,
        source_path=request.output_path,
        source_text=source_text,
        source_hash=hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        selected_members=selected_ids,
        bindings=(scalar_binding(request.region, root, compiled_name),),
        backend="cython",
    )
    return CallChainGeneration(
        generation=generation,
        plan=request.plan,
        width_proof=request.width_proof,
        compiled_name=compiled_name,
        helper_names=tuple(helper_names[helper.id] for helper in helpers),
        c_type=c_type,
    )


def _render_source(context: _CallChainRenderContext) -> str:
    request = context.request
    root = context.root
    members = {member.id: member for member in (root, *context.helpers)}
    rendered_helpers = tuple(_render_helper(context, helper, members) for helper in context.helpers)
    root_node = _rewritten_node(
        context,
        root,
        context.nodes[root.id],
        members=members,
    )
    root_body = _callable_body(root_node)
    local_names = scalar_local_integer_names(
        root_node,
        frozenset(
            (
                *(item.synthetic_name for item in context.field_bindings),
                *(item.name for item in call_chain_scalar_parameters(root)),
            )
        ),
    )
    native_name = f"{context.compiled_name}_native"
    scalar_parameters = call_chain_scalar_parameters(root)
    native_parameter_names = (
        *(item.synthetic_name for item in context.field_bindings),
        *(item.name for item in scalar_parameters),
    )
    native_parameters = ", ".join(f"{context.c_type} {name}" for name in native_parameter_names)
    wrapper_parameters = scalar_wrapper_parameter_declarations(root)
    native_arguments = ", ".join(
        (
            *(f"{item.owner_subject}.{item.field_name}" for item in context.field_bindings),
            *(item.name for item in scalar_parameters),
        )
    )
    sections = [
        "# cython: language_level=3, boundscheck=True, wraparound=True, cdivision=False",
        "from libc.stdint cimport int32_t, uint32_t, int64_t, uint64_t",
        "",
        f"# atoll scalar proof {SCALAR_KERNEL_GENERATOR_VERSION}",
        f"# atoll call chain {CALL_CHAIN_GENERATOR_VERSION}",
        f"# plan={request.plan.id}; width={request.width_proof.native.width}",
        "",
        *rendered_helpers,
        f"cdef {context.c_type} {native_name}({native_parameters}):",
    ]
    if local_names:
        sections.append(f"    cdef {context.c_type} {', '.join(local_names)}")
    sections.extend(render_scalar_statement(statement) for statement in root_body)
    sections.extend(
        (
            "",
            f"def {context.compiled_name}({wrapper_parameters}):",
            f"    return {native_name}({native_arguments})",
            "",
            f"__atoll_execution_kinds__ = {{{context.compiled_name!r}: 'sync'}}",
            "",
        )
    )
    return "\n".join(sections)


def _render_helper(
    context: _CallChainRenderContext,
    member: RegionMember,
    members: dict[SymbolId, RegionMember],
) -> str:
    rewritten = _rewritten_node(
        context,
        member,
        context.nodes[member.id],
        members=members,
    )
    body = _callable_body(rewritten)
    parameter_names = (
        *(item.synthetic_name for item in context.field_bindings),
        *(item.name for item in call_chain_scalar_parameters(member)),
    )
    parameters = ", ".join(f"{context.c_type} {name}" for name in parameter_names)
    lines = [f"cdef inline {context.c_type} {context.helper_names[member.id]}({parameters}):"]
    lines.extend(render_scalar_statement(statement) for statement in body)
    lines.append("")
    return "\n".join(lines)


def _rewritten_node(
    context: _CallChainRenderContext,
    member: RegionMember,
    node: ast.FunctionDef,
    *,
    members: dict[SymbolId, RegionMember],
) -> ast.FunctionDef:
    rewritten = _NativeCallRewriter(
        members,
        context.nodes,
        context.helper_names,
        context.field_bindings,
        member,
    ).visit(copy.deepcopy(node))
    rewritten = _rewrite_direct_fields(rewritten, member, context.field_bindings)
    rewritten = ScalarPowerExpansion().visit(ast.fix_missing_locations(rewritten))
    if not isinstance(rewritten, ast.FunctionDef):
        raise TypeError("call-chain source rewrite produced a non-function")
    rewritten.decorator_list = []
    return rewritten


def _rewrite_direct_fields(
    node: ast.AST,
    member: RegionMember,
    bindings: tuple[CallChainFieldBinding, ...],
) -> ast.AST:
    if member.binding_kind != "instance_method":
        return node
    rewritten = _GeneratedFieldReadSubstitution(bindings).visit(node)
    if not isinstance(rewritten, ast.AST):
        raise TypeError("generated field substitution produced a non-AST value")
    if any(
        isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load) and item.id == "self"
        for item in ast.walk(rewritten)
    ):
        raise ValueError(f"generated instance chain retains opaque self use: {member.id.stable_id}")
    return rewritten


def _callable_body(node: ast.FunctionDef) -> list[ast.stmt]:
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body.pop(0)
    return body


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def _source_hash(member: RegionMember) -> str:
    return hashlib.sha256(member.source_text.encode("utf-8")).hexdigest()
