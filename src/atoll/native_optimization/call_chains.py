"""Plan fixed-width native fusion for direct same-module scalar call chains.

The frontend uses retained typed-region call facts to find acyclic roots and
pure scalar helpers. Helper expressions are inlined only into a disposable AST
used by the interval prover; lowering keeps helpers as private native functions
so values remain unboxed across calls. Dynamic dispatch, recursion, opaque
calls, side effects, and cross-region dependencies remain Python fallbacks.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import textwrap
from dataclasses import dataclass, replace
from typing import Literal

from atoll.models import (
    CallSiteFact,
    ModuleScan,
    ParameterRecord,
    RegionMember,
    SymbolId,
    SymbolRecord,
    TypedRegion,
)
from atoll.native_optimization.models import (
    CallableCodeIdentityGuardPayload,
    DirectFieldGuardPayload,
    ExactTypeGuardPayload,
    GuardExpression,
    IntegerDomainGuardPayload,
)
from atoll.native_optimization.scalar_analysis import (
    ScalarKernelPlan,
    ScalarRejection,
    ScalarWidthProof,
    analyze_scalar_scan,
)

CallChainRejectionCode = Literal[
    "unsupported-root",
    "unsupported-helper",
    "split-region",
    "recursive-chain",
    "ambiguous-call",
    "opaque-call",
    "unproven-root",
    "unparseable-source",
]

CALL_CHAIN_ANALYSIS_VERSION = "call-chain-analysis-v1"


@dataclass(frozen=True, slots=True)
class CallChainEdge:
    """One resolved direct call retained by a native call-chain plan.

    Attributes:
        caller: Same-module callable containing the call site.
        callee: Same-module helper called directly by `caller`.
        lineno: One-based source line of the call relative to the declaration.
        col_offset: Zero-based source column of the call.
        end_lineno: One-based final source line of the call.
        end_col_offset: Zero-based final source column of the call.
    """

    caller: SymbolId
    callee: SymbolId
    lineno: int
    col_offset: int
    end_lineno: int
    end_col_offset: int | None


@dataclass(frozen=True, slots=True)
class CallChainFieldBinding:
    """One direct instance field converted into a private scalar input.

    Attributes:
        synthetic_name: Private proof and Cython parameter name.
        owner_subject: Source receiver parameter checked by the dispatcher.
        owner_module: Importable module containing the exact owner class.
        owner_qualname: Qualified source owner class name.
        field_name: Direct integer field read before native entry.
    """

    synthetic_name: str
    owner_subject: str
    owner_module: str
    owner_qualname: str
    field_name: str


@dataclass(frozen=True, slots=True)
class CallChainPlan:
    """Content-addressed fixed-width proof for one fused call-chain root.

    Attributes:
        id: Stable identity derived from source, topology, and proof versions.
        region_id: Typed region containing the complete call chain.
        root: Public source binding replaced by the guarded dispatcher.
        helpers: Private helper members in leaves-first lowering order.
        edges: Resolved direct call edges in deterministic source order.
        scalar_plan: Scalar proof over the semantically equivalent inlined root.
        source_hashes: Exact source hashes for root followed by helpers.
        callable_guards: Same-module callable and code identities checked before entry.
        receiver_guards: Exact receiver-class guards checked before instance native entry.
        field_bindings: Direct integer fields converted to private scalar inputs.
    """

    id: str
    region_id: str
    root: SymbolId
    helpers: tuple[SymbolId, ...]
    edges: tuple[CallChainEdge, ...]
    scalar_plan: ScalarKernelPlan
    source_hashes: tuple[tuple[SymbolId, str], ...]
    callable_guards: tuple[GuardExpression, ...]
    receiver_guards: tuple[GuardExpression, ...] = ()
    field_bindings: tuple[CallChainFieldBinding, ...] = ()


@dataclass(frozen=True, slots=True)
class CallChainRejection:
    """Deterministic reason a direct call-chain root remains interpreted.

    Attributes:
        root: Candidate public root rejected by the frontend.
        code: Stable rejection category.
        message: Report-facing explanation of the failed proof.
        lineno: One-based source line associated with the rejection, when known.
    """

    root: SymbolId
    code: CallChainRejectionCode
    message: str
    lineno: int | None = None


@dataclass(frozen=True, slots=True)
class CallChainAnalysisResult:
    """Native call-chain plans and conservative root-level rejections.

    Attributes:
        plans: Proven acyclic call chains in source order.
        rejections: Candidate roots retained as Python fallback.
    """

    plans: tuple[CallChainPlan, ...]
    rejections: tuple[CallChainRejection, ...]


class _ChainError(Exception):
    def __init__(
        self,
        code: CallChainRejectionCode,
        message: str,
        lineno: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code: CallChainRejectionCode = code
        self.message: str = message
        self.lineno: int | None = lineno


@dataclass(frozen=True, slots=True)
class _ChainContext:
    scan: ModuleScan
    region: TypedRegion
    members: dict[SymbolId, RegionMember]
    nodes: dict[SymbolId, ast.FunctionDef]
    class_members: dict[str, SymbolRecord]
    symbols: dict[SymbolId, SymbolRecord]
    custom_metaclass_owners: frozenset[str]


class _NameSubstitution(ast.NodeTransformer):
    def __init__(self, replacements: dict[str, ast.expr]) -> None:
        self._replacements = replacements

    def visit_Name(self, node: ast.Name) -> ast.expr:
        """Replace one helper parameter read with its bound call argument.

        Args:
            node: Name expression from a pure helper return expression.

        Returns:
            ast.expr: Copied bound argument or the unchanged source name.
        """
        if isinstance(node.ctx, ast.Load) and node.id in self._replacements:
            return ast.copy_location(copy.deepcopy(self._replacements[node.id]), node)
        return node


class _DirectFieldSubstitution(ast.NodeTransformer):
    def __init__(self, bindings: tuple[CallChainFieldBinding, ...]) -> None:
        self._names = {binding.field_name: binding.synthetic_name for binding in bindings}

    def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
        """Replace one proven direct receiver-field read with a scalar input.

        Args:
            node: Source attribute expression reached during proof flattening.

        Returns:
            ast.expr: Synthetic scalar input or recursively visited expression.

        Raises:
            _ChainError: If receiver access is not one proven direct field read.
            TypeError: If AST rewriting produces a non-expression value.
        """
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            if not isinstance(node.ctx, ast.Load) or node.attr not in self._names:
                raise _ChainError(
                    "unsupported-root",
                    f"instance chain uses unsupported receiver attribute self.{node.attr}",
                    node.lineno,
                )
            return ast.copy_location(ast.Name(id=self._names[node.attr], ctx=ast.Load()), node)
        visited = self.generic_visit(node)
        if not isinstance(visited, ast.expr):
            raise TypeError("direct-field substitution produced a non-expression")
        return visited


class _CallInliner(ast.NodeTransformer):
    def __init__(self, context: _ChainContext, root: RegionMember) -> None:
        self._context = context
        self._root = root
        self._stack: list[SymbolId] = [root.id]
        self._site_indices: dict[tuple[SymbolId, str], int] = {}
        self.edges: list[CallChainEdge] = []
        self.helpers: list[SymbolId] = []

    def visit_Call(self, node: ast.Call) -> ast.expr:
        """Inline a direct pure helper call into the disposable proof AST.

        Args:
            node: Call expression reachable from the root body.

        Returns:
            ast.expr: Recursively inlined helper return expression.

        Raises:
            _ChainError: If the call is opaque, recursive, or cannot be bound exactly.
        """
        if isinstance(node.func, ast.Name) and node.func.id == "range":
            visited = self.generic_visit(node)
            if not isinstance(visited, ast.expr):
                raise TypeError("range call transformation produced a non-expression")
            return visited
        caller = self._context.members[self._stack[-1]]
        callee_id = resolve_callee(caller, node.func, self._context.members)
        if callee_id is None:
            raise _ChainError(
                "opaque-call",
                f"call target {ast.unparse(node.func)} is not one direct same-module helper",
                node.lineno,
            )
        if callee_id in self._stack:
            raise _ChainError(
                "recursive-chain",
                f"recursive call chain reaches {callee_id.stable_id}",
                node.lineno,
            )
        callee, expression = self._validated_callee(callee_id, node)
        replacements = bind_member_call_arguments(callee, self._context.nodes[callee_id], node)
        substituted = _NameSubstitution(replacements).visit(copy.deepcopy(expression))
        if not isinstance(substituted, ast.expr):
            raise TypeError("call-chain substitution produced a non-expression")
        site = self._consume_call_site(caller, node, callee_id)
        self.edges.append(
            CallChainEdge(
                caller=caller.id,
                callee=callee_id,
                lineno=site.lineno,
                col_offset=site.col_offset,
                end_lineno=site.end_lineno,
                end_col_offset=site.end_col_offset,
            )
        )
        self._stack.append(callee_id)
        try:
            inlined = self.visit(substituted)
        finally:
            self._stack.pop()
        if not isinstance(inlined, ast.expr):
            raise TypeError("call-chain inlining produced a non-expression")
        if callee_id not in self.helpers:
            self.helpers.append(callee_id)
        return inlined

    def _validated_callee(
        self,
        callee_id: SymbolId,
        node: ast.Call,
    ) -> tuple[RegionMember, ast.expr]:
        callee = self._context.members[callee_id]
        _validate_helper(callee)
        if callee.binding_kind == "instance_method":
            _validate_instance_owner(self._context, callee)
            if callee.owner_class != self._root.owner_class:
                raise _ChainError(
                    "unsupported-helper",
                    "instance call-chain helpers must share the root's exact owner class",
                    node.lineno,
                )
        if callee.binding_kind == "staticmethod":
            _validate_static_owner(self._context, callee)
        expression = _helper_return_expression(self._context.nodes[callee_id])
        _validate_helper_parameter_reads(callee, expression)
        return callee, expression

    def _consume_call_site(
        self,
        caller: RegionMember,
        node: ast.Call,
        callee: SymbolId,
    ) -> CallSiteFact:
        target = ast.unparse(node.func)
        candidates = tuple(
            site
            for site in caller.call_sites
            if site.target == target
            and _resolve_call_site(caller, site, self._context.members) == callee
        )
        key = (caller.id, target)
        index = self._site_indices.get(key, 0)
        if index >= len(candidates):
            raise _ChainError(
                "ambiguous-call",
                f"call site for {caller.id.stable_id} -> {callee.stable_id} is not retained",
                node.lineno,
            )
        self._site_indices[key] = index + 1
        return candidates[index]


def analyze_call_chain_scan(scan: ModuleScan) -> CallChainAnalysisResult:
    """Find acyclic scalar call chains and prove their inlined arithmetic.

    Args:
        scan: Enriched module scan containing typed regions and ordered call facts.

    Returns:
        CallChainAnalysisResult: Proven plans and deterministic candidate rejections.
    """
    plans: list[CallChainPlan] = []
    rejections: list[CallChainRejection] = []
    class_members = {
        symbol.id.qualname: symbol for symbol in scan.symbols if symbol.kind == "class"
    }
    symbols = {symbol.id: symbol for symbol in scan.symbols}
    for region in scan.typed_regions:
        members = {
            member.id: member
            for member in region.members
            if member.kind in {"function", "method"} and member.execution_kind == "sync"
        }
        nodes: dict[SymbolId, ast.FunctionDef] = {}
        for member_id, member in members.items():
            try:
                nodes[member_id] = _callable_node(member)
            except _ChainError as error:
                rejections.append(
                    CallChainRejection(
                        root=member.id,
                        code=error.code,
                        message=error.message,
                        lineno=error.lineno,
                    )
                )
        members = {member_id: member for member_id, member in members.items() if member_id in nodes}
        if not members:
            continue
        context = _ChainContext(
            scan=scan,
            region=region,
            members=members,
            nodes=nodes,
            class_members=class_members,
            symbols=symbols,
            custom_metaclass_owners=_custom_metaclass_owners(scan),
        )
        for member in members.values():
            if not _has_direct_member_call(member, members):
                continue
            try:
                plan = _analyze_root(context, member)
            except _ChainError as error:
                rejections.append(
                    CallChainRejection(
                        root=member.id,
                        code=error.code,
                        message=error.message,
                        lineno=error.lineno,
                    )
                )
            else:
                plans.append(plan)
    return CallChainAnalysisResult(plans=tuple(plans), rejections=tuple(rejections))


def call_chain_runtime_guards(
    plan: CallChainPlan,
    proof: ScalarWidthProof,
) -> tuple[GuardExpression, ...]:
    """Convert synthetic field proof inputs into structured receiver guards.

    Args:
        plan: Revalidated direct call-chain plan.
        proof: Fixed-width proof selected for runtime dispatch.

    Returns:
        tuple[GuardExpression, ...]: Pre-entry argument, owner, field, and callable guards.
    """
    fields = {binding.synthetic_name: binding for binding in plan.field_bindings}
    retained = tuple(
        guard
        for guard in proof.guards
        if not (
            isinstance(guard.payload, ExactTypeGuardPayload | IntegerDomainGuardPayload)
            and guard.payload.subject in fields
        )
    )
    domains = {parameter.name: parameter.interval for parameter in proof.parameters}
    direct_fields = tuple(
        GuardExpression(
            kind="direct-field",
            payload=DirectFieldGuardPayload(
                owner_subject=binding.owner_subject,
                owner_type_module=binding.owner_module,
                owner_type_qualname=binding.owner_qualname,
                field_name=binding.field_name,
                field_type="int",
                minimum=domains[binding.synthetic_name].minimum,
                maximum=domains[binding.synthetic_name].maximum,
            ),
            message=(
                f"{binding.owner_subject}.{binding.field_name} must be an exact int in the "
                "proven native domain"
            ),
        )
        for binding in plan.field_bindings
    )
    return (*retained, *plan.receiver_guards, *direct_fields, *plan.callable_guards)


def _analyze_root(context: _ChainContext, root: RegionMember) -> CallChainPlan:
    _validate_root(context, root)
    root_node = copy.deepcopy(context.nodes[root.id])
    inliner = _CallInliner(context, root)
    inlined = inliner.visit(root_node)
    if not isinstance(inlined, ast.FunctionDef):
        raise TypeError("call-chain inlining produced a non-function root")
    if not inliner.helpers:
        raise _ChainError("ambiguous-call", "root has no resolvable direct helper call")
    helpers = tuple(inliner.helpers)
    _reject_module_constant_dependencies(context, (root.id, *helpers))
    field_bindings = _instance_field_bindings(context, root, inlined)
    if field_bindings:
        inlined = _DirectFieldSubstitution(field_bindings).visit(inlined)
        if not isinstance(inlined, ast.FunctionDef):
            raise TypeError("direct-field substitution produced a non-function root")
    if root.binding_kind == "instance_method":
        _replace_receiver_with_fields(inlined, field_bindings)
    inlined.decorator_list = []
    synthetic_parameters = _synthetic_parameters(root, field_bindings)
    synthetic_member = replace(
        root,
        source_text=ast.unparse(ast.fix_missing_locations(inlined)),
        call_sites=(),
        kind="function",
        owner_class=None,
        binding_kind="module",
        parameters=synthetic_parameters,
    )
    synthetic_region = replace(context.region, members=(synthetic_member,))
    synthetic_scan = replace(context.scan, typed_regions=(synthetic_region,))
    scalar = analyze_scalar_scan(synthetic_scan)
    scalar_plan = next((item for item in scalar.plans if item.member == root.id), None)
    rejection = next((item for item in scalar.rejections if item.member == root.id), None)
    if scalar_plan is None:
        message = (
            rejection.message
            if isinstance(rejection, ScalarRejection)
            else "inlined root produced no scalar proof evidence"
        )
        raise _ChainError("unproven-root", message, getattr(rejection, "lineno", None))
    source_hashes = tuple(
        (member_id, _source_hash(context.members[member_id])) for member_id in (root.id, *helpers)
    )
    edges = tuple(inliner.edges)
    identity = "\0".join(
        (
            CALL_CHAIN_ANALYSIS_VERSION,
            context.region.id,
            *(f"{member.stable_id}:{digest}" for member, digest in source_hashes),
            *(
                f"{edge.caller.stable_id}>{edge.callee.stable_id}:"
                f"{edge.lineno}:{edge.col_offset}:{edge.end_lineno}:{edge.end_col_offset}"
                for edge in edges
            ),
            scalar_plan.id,
        )
    )
    plan_id = hashlib.blake2b(identity.encode("utf-8"), digest_size=16).hexdigest()
    return CallChainPlan(
        id=f"call-chain-{plan_id}",
        region_id=context.region.id,
        root=root.id,
        helpers=helpers,
        edges=edges,
        scalar_plan=scalar_plan,
        source_hashes=source_hashes,
        callable_guards=tuple(
            _callable_guard(context, context.members[helper]) for helper in helpers
        ),
        receiver_guards=_receiver_guards(root),
        field_bindings=field_bindings,
    )


def _validate_root(context: _ChainContext, member: RegionMember) -> None:
    if member.binding_kind not in {"module", "staticmethod", "instance_method"}:
        raise _ChainError(
            "unsupported-root",
            "direct call-chain roots require a module, staticmethod, or instance-method binding",
        )
    if member.binding_kind == "instance_method":
        _validate_instance_owner(context, member)
    if member.binding_kind == "staticmethod":
        _validate_static_owner(context, member)


def _validate_helper(member: RegionMember) -> None:
    if member.binding_kind not in {"module", "staticmethod", "instance_method"}:
        raise _ChainError(
            "unsupported-helper",
            f"helper {member.id.stable_id} has unsupported binding {member.binding_kind}",
        )
    parameters = call_chain_scalar_parameters(member)
    if member.return_annotation != "int" or any(
        parameter.annotation != "int" for parameter in parameters
    ):
        raise _ChainError(
            "unsupported-helper",
            f"helper {member.id.stable_id} requires exact int parameters and return type",
        )


def _validate_helper_parameter_reads(member: RegionMember, expression: ast.expr) -> None:
    loaded = {
        node.id
        for node in ast.walk(expression)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    unused = tuple(
        parameter.name
        for parameter in call_chain_scalar_parameters(member)
        if parameter.name not in loaded
    )
    if unused:
        raise _ChainError(
            "unsupported-helper",
            f"helper {member.id.stable_id} ignores scalar parameter(s): {', '.join(unused)}",
        )


def _callable_node(member: RegionMember) -> ast.FunctionDef:
    wrapped_source = "if True:\n" + textwrap.indent(member.source_text, "    ")
    try:
        parsed = ast.parse(wrapped_source)
    except SyntaxError as error:
        source_line = max(1, (error.lineno or 2) - 1)
        raise _ChainError(
            "unparseable-source",
            f"{member.id.stable_id} source cannot be parsed in its lexical context",
            source_line,
        ) from error
    container = parsed.body[0]
    if not isinstance(container, ast.If):
        raise TypeError("call-chain source wrapper produced an invalid syntax tree")
    declarations = tuple(node for node in container.body if isinstance(node, ast.FunctionDef))
    if len(declarations) != 1:
        raise _ChainError(
            "unsupported-helper",
            f"{member.id.stable_id} does not contain one synchronous declaration",
        )
    return declarations[0]


def _helper_return_expression(node: ast.FunctionDef) -> ast.expr:
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body.pop(0)
    if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
        raise _ChainError(
            "unsupported-helper",
            f"helper {node.name} must contain one pure return expression",
            node.lineno,
        )
    return body[0].value


def bind_call_arguments(node: ast.FunctionDef, call: ast.Call) -> dict[str, ast.expr]:
    """Bind one proven helper call without executing source default expressions.

    Args:
        node: Parsed helper declaration defining the source signature.
        call: Direct call expression being bound statically.

    Returns:
        dict[str, ast.expr]: Exact parameter-to-argument expression mapping.

    Raises:
        _ChainError: If variadics, dynamic keywords, or unsafe defaults prevent exact binding.
    """
    if node.args.vararg is not None or node.args.kwarg is not None:
        raise _ChainError("ambiguous-call", f"helper {node.name} uses variadic parameters")
    positional = (*node.args.posonlyargs, *node.args.args)
    if len(call.args) > len(positional) or any(isinstance(arg, ast.Starred) for arg in call.args):
        raise _ChainError("ambiguous-call", f"call to {node.name} cannot be bound statically")
    bound = {
        parameter.arg: argument for parameter, argument in zip(positional, call.args, strict=False)
    }
    _bind_keywords(node, call, bound)
    _bind_defaults(node, bound, positional)
    known = {parameter.arg for parameter in (*positional, *node.args.kwonlyargs)}
    if not set(bound) <= known:
        unknown = ", ".join(sorted(set(bound) - known))
        raise _ChainError("ambiguous-call", f"call to {node.name} has unknown keyword {unknown}")
    return bound


def bind_member_call_arguments(
    member: RegionMember,
    node: ast.FunctionDef,
    call: ast.Call,
) -> dict[str, ast.expr]:
    """Bind a module, static, or exact-receiver instance helper call.

    Args:
        member: Retained helper member and descriptor kind.
        node: Parsed helper declaration defining the source signature.
        call: Direct call expression being bound statically.

    Returns:
        dict[str, ast.expr]: Exact parameter-to-argument expression mapping.

    Raises:
        _ChainError: If an instance helper is not dispatched directly through `self`.
    """
    if member.binding_kind != "instance_method":
        return bind_call_arguments(node, call)
    if not (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "self"
    ):
        raise _ChainError(
            "ambiguous-call",
            f"instance helper {member.id.stable_id} requires direct self dispatch",
            call.lineno,
        )
    adjusted = copy.deepcopy(call)
    adjusted.args.insert(0, ast.Name(id="self", ctx=ast.Load()))
    return bind_call_arguments(node, adjusted)


def _bind_keywords(
    node: ast.FunctionDef,
    call: ast.Call,
    bound: dict[str, ast.expr],
) -> None:
    positional_only = {parameter.arg for parameter in node.args.posonlyargs}
    for keyword in call.keywords:
        if keyword.arg is None or keyword.arg in bound:
            raise _ChainError("ambiguous-call", f"call to {node.name} has dynamic keywords")
        if keyword.arg in positional_only:
            raise _ChainError(
                "ambiguous-call",
                f"call to {node.name} passes positional-only parameter {keyword.arg} by keyword",
            )
        bound[keyword.arg] = keyword.value


def _bind_defaults(
    node: ast.FunctionDef,
    bound: dict[str, ast.expr],
    positional: tuple[ast.arg, ...],
) -> None:
    positional_defaults = {
        parameter.arg: default
        for parameter, default in zip(
            positional[len(positional) - len(node.args.defaults) :],
            node.args.defaults,
            strict=True,
        )
    }
    keyword_defaults = {
        parameter.arg: default
        for parameter, default in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True)
        if default is not None
    }
    for parameter in (*positional, *node.args.kwonlyargs):
        if parameter.arg in bound:
            continue
        default = positional_defaults.get(parameter.arg) or keyword_defaults.get(parameter.arg)
        if default is None:
            raise _ChainError(
                "ambiguous-call",
                f"call to {node.name} omits required argument {parameter.arg}",
            )
        if not _is_exact_integer_default(default):
            raise _ChainError(
                "ambiguous-call",
                f"call to {node.name} uses a non-literal default for {parameter.arg}",
            )
        bound[parameter.arg] = default


def _is_exact_integer_default(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and type(node.value) is int


def call_chain_scalar_parameters(member: RegionMember) -> tuple[ParameterRecord, ...]:
    """Return exact integer parameters excluding an instance receiver.

    Args:
        member: Module, static, or instance callable in a direct chain.

    Returns:
        tuple[ParameterRecord, ...]: Caller-visible scalar parameters in source order.

    Raises:
        _ChainError: If an instance member lacks a conventional `self` receiver.
    """
    if member.binding_kind != "instance_method":
        return member.parameters
    if not member.parameters or member.parameters[0].name != "self":
        raise _ChainError(
            "unsupported-helper",
            f"instance member {member.id.stable_id} has no conventional self receiver",
        )
    return member.parameters[1:]


def _validate_instance_owner(context: _ChainContext, member: RegionMember) -> None:
    owner = member.owner_class
    if owner is None or owner not in context.class_members:
        raise _ChainError(
            "unsupported-root",
            f"instance member {member.id.stable_id} has no retained owner class",
        )
    unsafe_names = {
        f"{owner}.{name}" for name in ("__getattribute__", "__getattr__", "__setattr__")
    }
    all_member_ids = {candidate.id.qualname for candidate in context.scan.symbols}
    if unsafe_names.intersection(all_member_ids):
        raise _ChainError(
            "unsupported-root",
            f"owner {owner} defines custom attribute hooks",
        )


def _validate_static_owner(context: _ChainContext, member: RegionMember) -> None:
    owner = member.owner_class
    if owner is None or owner not in context.class_members:
        raise _ChainError(
            "unsupported-root",
            f"static member {member.id.stable_id} has no retained owner class",
        )
    owner_record = context.class_members[owner]
    inherited = tuple(base for base in owner_record.base_names if base != "object")
    if inherited or owner in context.custom_metaclass_owners:
        raise _ChainError(
            "unsupported-root",
            f"static owner {owner} may customize metaclass dispatch",
        )


def _custom_metaclass_owners(scan: ModuleScan) -> frozenset[str]:
    tree = ast.parse(scan.module.path.read_text(encoding="utf-8"))
    return frozenset(
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and any(keyword.arg == "metaclass" for keyword in node.keywords)
    )


def _reject_module_constant_dependencies(
    context: _ChainContext,
    members: tuple[SymbolId, ...],
) -> None:
    constants = {constant.name for constant in context.scan.constants}
    for member_id in members:
        symbol = context.symbols.get(member_id)
        used = sorted(constants.intersection(symbol.uses_globals if symbol is not None else ()))
        if used:
            raise _ChainError(
                "opaque-call",
                f"call chain reads mutable module constant(s): {', '.join(used)}",
            )


def _instance_field_bindings(
    context: _ChainContext,
    root: RegionMember,
    inlined: ast.FunctionDef,
) -> tuple[CallChainFieldBinding, ...]:
    if root.binding_kind != "instance_method":
        return ()
    owner = root.owner_class
    if owner is None:
        raise _ChainError("unsupported-root", "instance root has no owner class")
    owner_member = context.class_members[owner]
    fields = {
        field.name: field
        for field in owner_member.fields
        if not field.class_variable and field.annotation == "int"
    }
    names: list[str] = []
    for node in ast.walk(inlined):
        if not (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            continue
        if not isinstance(node.ctx, ast.Load) or node.attr not in fields:
            raise _ChainError(
                "unsupported-root",
                f"instance chain field self.{node.attr} is not one retained int field",
                node.lineno,
            )
        if node.attr not in names:
            names.append(node.attr)
    receiver = root.parameters[0].name if root.parameters else "self"
    return tuple(
        CallChainFieldBinding(
            synthetic_name=f"_atoll_field_{name}",
            owner_subject=receiver,
            owner_module=root.id.module,
            owner_qualname=owner,
            field_name=name,
        )
        for name in names
    )


def _replace_receiver_with_fields(
    node: ast.FunctionDef,
    bindings: tuple[CallChainFieldBinding, ...],
) -> None:
    receiver_removed = False
    if node.args.posonlyargs and node.args.posonlyargs[0].arg == "self":
        node.args.posonlyargs.pop(0)
        receiver_removed = True
    elif node.args.args and node.args.args[0].arg == "self":
        node.args.args.pop(0)
        receiver_removed = True
    if not receiver_removed:
        raise _ChainError("unsupported-root", "instance root has no conventional self receiver")
    field_arguments = [
        ast.arg(arg=binding.synthetic_name, annotation=ast.Name(id="int", ctx=ast.Load()))
        for binding in bindings
    ]
    node.args.args[0:0] = field_arguments


def _synthetic_parameters(
    root: RegionMember,
    bindings: tuple[CallChainFieldBinding, ...],
) -> tuple[ParameterRecord, ...]:
    field_parameters = tuple(
        ParameterRecord(
            name=binding.synthetic_name,
            kind="positional",
            annotation="int",
            default_source=None,
        )
        for binding in bindings
    )
    return (*field_parameters, *call_chain_scalar_parameters(root))


def _receiver_guards(root: RegionMember) -> tuple[GuardExpression, ...]:
    if root.binding_kind != "instance_method" or root.owner_class is None:
        return ()
    receiver = root.parameters[0].name if root.parameters else "self"
    return (
        GuardExpression(
            kind="exact-type",
            payload=ExactTypeGuardPayload(
                subject=receiver,
                type_module=root.id.module,
                type_qualname=root.owner_class,
            ),
            message=f"{receiver} must be exactly {root.id.module}.{root.owner_class}",
        ),
    )


def _has_direct_member_call(
    member: RegionMember,
    members: dict[SymbolId, RegionMember],
) -> bool:
    return any(_resolve_call_site(member, site, members) is not None for site in member.call_sites)


def _resolve_call_site(
    caller: RegionMember,
    site: CallSiteFact,
    members: dict[SymbolId, RegionMember],
) -> SymbolId | None:
    return _resolved_member_id(caller, site.target, members)


def resolve_callee(
    caller: RegionMember,
    expression: ast.expr,
    members: dict[SymbolId, RegionMember],
) -> SymbolId | None:
    """Resolve one direct call expression to a same-region member identity.

    Args:
        caller: Retained member containing the call.
        expression: Source call-target expression.
        members: Same-region members eligible for direct resolution.

    Returns:
        SymbolId | None: Resolved member identity, or `None` for a runtime boundary.
    """
    return _resolved_member_id(caller, ast.unparse(expression), members)


def _resolved_member_id(
    caller: RegionMember,
    target: str,
    members: dict[SymbolId, RegionMember],
) -> SymbolId | None:
    qualname = target
    if target.startswith(("self.", "cls.")) and caller.owner_class is not None:
        qualname = f"{caller.owner_class}.{target.partition('.')[2]}"
    candidate = SymbolId(module=caller.id.module, qualname=qualname)
    return candidate if candidate in members else None


def _callable_guard(context: _ChainContext, member: RegionMember) -> GuardExpression:
    digest = _source_hash(member)
    return GuardExpression(
        kind="callable-code-identity",
        payload=CallableCodeIdentityGuardPayload(
            subject=member.id.qualname,
            callable_module=member.id.module,
            callable_qualname=member.id.qualname,
            code_fingerprint=digest,
            receiver_subject=(
                member.parameters[0].name
                if member.binding_kind == "instance_method" and member.parameters
                else None
            ),
            code_firstlineno=context.symbols[member.id].declaration_start_lineno,
        ),
        message=f"{member.id.stable_id} must retain its analyzed callable and code identity",
    )


def _source_hash(member: RegionMember) -> str:
    return hashlib.sha256(member.source_text.encode("utf-8")).hexdigest()
