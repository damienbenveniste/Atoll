"""Generate fixed-width Cython callables from proven scalar kernel plans.

The generator consumes only scanner and scalar-proof evidence. It emits one
temporary ``.pyx`` compilation unit per width variant, preserves the original
call body, and changes integer representation only after the frontend proved
that every operand, intermediate, and return value fits the selected width.
Generated sources remain disposable build inputs and are never installed.
"""

from __future__ import annotations

import ast
import hashlib
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

from atoll.generation.typed_region import TypedRegionGeneration
from atoll.models import BindingTarget, ModuleScan, RegionMember, TypedRegion
from atoll.native_optimization.scalar_analysis import ScalarKernelPlan, ScalarWidthProof

SCALAR_KERNEL_GENERATOR_VERSION = "atoll-scalar-kernel-v1"


@dataclass(frozen=True, slots=True)
class ScalarKernelGeneration:
    """Generated Cython source plus private fixed-width proof metadata.

    Attributes:
        generation: Standard typed-region generation contract used by packaging.
        plan: Backend-neutral scalar proof plan that authorized lowering.
        width_proof: Exact native width and pre-entry guards for this unit.
        compiled_name: Private extension attribute bound by the staged dispatcher.
        c_type: Cython fixed-width integer type used for parameters, locals, and return.
    """

    generation: TypedRegionGeneration
    plan: ScalarKernelPlan
    width_proof: ScalarWidthProof
    compiled_name: str
    c_type: str


@dataclass(frozen=True, slots=True)
class ScalarKernelGenerationRequest:
    """Staged scalar evidence and destination for one width variant.

    Attributes:
        scan: Module scan containing the retained declaration and constants.
        region: Typed region that owns the scalar member.
        plan: Revalidated proof plan authorizing fixed-width lowering.
        width_proof: Exact width, argument domains, and operation proof.
        logical_module: Private extension module name used in the wheel payload.
        output_path: Disposable `.pyx` path under the source-clean build root.
    """

    scan: ModuleScan
    region: TypedRegion
    plan: ScalarKernelPlan
    width_proof: ScalarWidthProof
    logical_module: str
    output_path: Path


@dataclass(frozen=True, slots=True)
class _ScalarRenderContext:
    scan: ModuleScan
    member: RegionMember
    node: ast.FunctionDef
    proof: ScalarWidthProof
    compiled_name: str
    c_type: str


class ScalarPowerExpansion(ast.NodeTransformer):
    """Expand small constant powers into exact integer multiplication trees."""

    def visit_BinOp(self, node: ast.BinOp) -> ast.expr:
        """Replace ``value ** n`` with repeated multiplication.

        Args:
            node: Binary operation retained by the scalar frontend.

        Returns:
            ast.expr: Exact multiplication expression or the recursively visited node.

        Raises:
            TypeError: If recursive AST transformation violates the expression contract.
        """
        visited = self.generic_visit(node)
        if not isinstance(visited, ast.BinOp):
            raise TypeError("power expansion produced a non-expression")
        if not (
            isinstance(visited.op, ast.Pow)
            and isinstance(visited.right, ast.Constant)
            and type(visited.right.value) is int
        ):
            return visited
        exponent = visited.right.value
        if exponent == 0:
            return ast.copy_location(ast.Constant(value=1), visited)
        expanded = visited.left
        for _ in range(exponent - 1):
            expanded = ast.BinOp(
                left=expanded,
                op=ast.Mult(),
                right=visited.left,
            )
        return ast.copy_location(expanded, visited)


def generate_scalar_kernel(request: ScalarKernelGenerationRequest) -> ScalarKernelGeneration:
    """Write one fixed-width Cython scalar compilation unit.

    Args:
        request: Staged scan, region, proof, module identity, and disposable output path.

    Returns:
        ScalarKernelGeneration: Generated source, binding, and proof evidence.

    Raises:
        ValueError: If staged evidence does not match the plan, uses unsigned
            arithmetic without an explicit modular proof, or cannot be rendered
            as one supported callable.
    """
    if request.output_path.suffix != ".pyx":
        raise ValueError("scalar kernels require a .pyx output path")
    if not request.width_proof.native.signed and request.width_proof.explicit_modular_width is None:
        raise ValueError("unsigned scalar lowering requires an explicit modular proof")
    member = next(
        (item for item in request.region.members if item.id == request.plan.member),
        None,
    )
    if member is None:
        raise ValueError(
            f"scalar plan member is absent from staged region: {request.plan.member.stable_id}"
        )
    source_hash = hashlib.sha256(member.source_text.encode("utf-8")).hexdigest()
    if source_hash != request.plan.source_hash:
        raise ValueError("staged scalar member source differs from analyzed source")
    node = scalar_callable_node(member)
    compiled_name = _compiled_name(member, request.width_proof)
    c_type = scalar_c_type(request.width_proof)
    source_text = _render_source(
        _ScalarRenderContext(
            scan=request.scan,
            member=member,
            node=node,
            proof=request.width_proof,
            compiled_name=compiled_name,
            c_type=c_type,
        )
    )
    request.output_path.parent.mkdir(parents=True, exist_ok=True)
    request.output_path.write_text(source_text, encoding="utf-8")
    binding = scalar_binding(request.region, member, compiled_name)
    generation = TypedRegionGeneration(
        region=request.region,
        logical_module=request.logical_module,
        source_path=request.output_path,
        source_text=source_text,
        source_hash=hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        selected_members=(member.id,),
        bindings=(binding,),
        backend="cython",
    )
    return ScalarKernelGeneration(
        generation=generation,
        plan=request.plan,
        width_proof=request.width_proof,
        compiled_name=compiled_name,
        c_type=c_type,
    )


def scalar_callable_node(member: RegionMember) -> ast.FunctionDef:
    """Parse one retained synchronous scalar member declaration.

    Args:
        member: Retained region member containing exact source text.

    Returns:
        ast.FunctionDef: Parsed synchronous declaration.

    Raises:
        ValueError: If the retained source is not exactly one synchronous function.
    """
    parsed = ast.parse(textwrap.dedent(member.source_text))
    declarations = tuple(node for node in parsed.body if isinstance(node, ast.FunctionDef))
    if len(declarations) != 1:
        raise ValueError("scalar kernel source must contain one synchronous function")
    return declarations[0]


def _compiled_name(member: RegionMember, proof: ScalarWidthProof) -> str:
    stem = re.sub(r"[^A-Za-z0-9_]", "_", member.id.qualname)
    signed = "i" if proof.native.signed else "u"
    return f"_atoll_scalar_{stem}_{signed}{proof.native.width}"


def scalar_c_type(proof: ScalarWidthProof) -> str:
    """Render the portable stdint Cython type authorized by a width proof.

    Args:
        proof: Fixed-width scalar proof selected for lowering.

    Returns:
        str: Signed or unsigned stdint Cython type name.
    """
    signed = "int" if proof.native.signed else "uint"
    return f"{signed}{proof.native.width}_t"


def _render_source(context: _ScalarRenderContext) -> str:
    native_parameters = ", ".join(
        f"{context.c_type} {parameter.name}" for parameter in context.member.parameters
    )
    wrapper_parameters = scalar_wrapper_parameter_declarations(context.member)
    native_arguments = ", ".join(parameter.name for parameter in context.member.parameters)
    native_name = f"{context.compiled_name}_native"
    transformed = ScalarPowerExpansion().visit(ast.fix_missing_locations(context.node))
    if not isinstance(transformed, ast.FunctionDef):
        raise TypeError("scalar power expansion produced a non-function")
    body = list(transformed.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body.pop(0)
    local_names = scalar_local_integer_names(
        transformed,
        frozenset(parameter.name for parameter in context.member.parameters),
    )
    constants = _referenced_integer_constants(context.scan, transformed, context.proof)
    sections = [
        "# cython: language_level=3, boundscheck=True, wraparound=True, cdivision=False",
        "from libc.stdint cimport int32_t, uint32_t, int64_t, uint64_t",
        "",
        f"# atoll scalar proof {SCALAR_KERNEL_GENERATOR_VERSION}",
        f"# width={context.proof.native.width}; signed={context.proof.native.signed}",
        *(
            f"# domain {domain.name}={domain.interval.minimum}:{domain.interval.maximum}"
            for domain in context.proof.parameters
        ),
        *(f"cdef {context.c_type} {name} = {value}" for name, value in constants),
        "",
        f"cdef {context.c_type} {native_name}({native_parameters}):",
    ]
    if local_names:
        sections.append(f"    cdef {context.c_type} {', '.join(local_names)}")
    sections.extend(render_scalar_statement(statement) for statement in body)
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


def scalar_wrapper_parameter_declarations(member: RegionMember) -> str:
    """Render a signature-shaped Python wrapper declaration for a source member.

    Args:
        member: Source member whose Python calling convention must be preserved.

    Returns:
        str: Cython-compatible wrapper parameter declaration.

    Raises:
        ValueError: If the source uses a parameter shape unsupported by scalar lowering.
    """
    positional_only: list[str] = []
    positional: list[str] = []
    keyword_only: list[str] = []
    for parameter in member.parameters:
        declaration = parameter.name
        if parameter.kind == "positional_only":
            positional_only.append(declaration)
        elif parameter.kind == "positional":
            positional.append(declaration)
        elif parameter.kind == "keyword_only":
            keyword_only.append(declaration)
        else:
            raise ValueError(f"unsupported scalar parameter kind: {parameter.kind}")
    declarations = list(positional_only)
    if positional_only:
        declarations.append("/")
    declarations.extend(positional)
    if keyword_only:
        declarations.append("*")
        declarations.extend(keyword_only)
    return ", ".join(declarations)


def scalar_local_integer_names(
    node: ast.FunctionDef, parameters: frozenset[str]
) -> tuple[str, ...]:
    """Return deterministic local names that the proof permits as native integers.

    Args:
        node: Parsed proof-authorized function declaration.
        parameters: Names already represented by native parameters.

    Returns:
        tuple[str, ...]: Local integer names in deterministic source order.
    """
    names: set[str] = set()
    for descendant in ast.walk(node):
        if isinstance(descendant, ast.Assign):
            names.update(target.id for target in descendant.targets if isinstance(target, ast.Name))
        elif isinstance(descendant, ast.AnnAssign | ast.For) and isinstance(
            descendant.target, ast.Name
        ):
            names.add(descendant.target.id)
    return tuple(sorted(names - parameters))


def _referenced_integer_constants(
    scan: ModuleScan,
    node: ast.FunctionDef,
    proof: ScalarWidthProof,
) -> tuple[tuple[str, int], ...]:
    referenced = {
        name.id
        for statement in node.body
        for name in ast.walk(statement)
        if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Load)
    }
    native = proof.native
    constants: list[tuple[str, int]] = []
    for constant in scan.constants:
        if constant.name not in referenced or constant.kind != "literal_constant":
            continue
        statement = ast.parse(constant.source_text).body[0]
        if not isinstance(statement, ast.Assign):
            raise TypeError(f"scalar constant is not a simple assignment: {constant.name}")
        value = ast.literal_eval(statement.value)
        if type(value) is not int or not native.minimum <= value <= native.maximum:
            raise ValueError(f"scalar constant does not fit native width: {constant.name}")
        constants.append((constant.name, value))
    unresolved = referenced - {
        *[parameter.name for parameter in proof.parameters],
        *[name for name, _ in constants],
        *list(scalar_local_integer_names(node, frozenset())),
        "range",
    }
    if unresolved:
        raise ValueError(
            f"scalar kernel has unresolved runtime names: {', '.join(sorted(unresolved))}"
        )
    return tuple(constants)


def render_scalar_statement(statement: ast.stmt) -> str:
    """Render one proof-authorized scalar statement as Cython source.

    Args:
        statement: Statement accepted by the scalar proof frontend.

    Returns:
        str: Indented Cython statement source.

    Raises:
        ValueError: If an annotation-only local has no initializer.
    """
    if isinstance(statement, ast.AnnAssign):
        if statement.value is None:
            raise ValueError("scalar local annotation requires an initializer")
        statement = ast.Assign(targets=[statement.target], value=statement.value)
    return textwrap.indent(ast.unparse(statement), "    ")


def scalar_binding(region: TypedRegion, member: RegionMember, compiled_name: str) -> BindingTarget:
    """Retarget a source binding to one generated scalar entry point.

    Args:
        region: Typed region containing the source binding contract.
        member: Source callable represented by the generated entry point.
        compiled_name: Python-visible name exported by the native module.

    Returns:
        BindingTarget: Runtime binding preserving source descriptor metadata.

    Raises:
        ValueError: If a non-module callable has no reconstructable runtime binding.
    """
    source = next((binding for binding in region.bindings if binding.source == member.id), None)
    if source is None:
        if member.binding_kind not in {"module", "staticmethod", "instance_method"}:
            raise ValueError(f"scalar member has no runtime binding: {member.id.stable_id}")
        return BindingTarget(
            source=member.id,
            compiled_name=compiled_name,
            kind=member.binding_kind,
            owner_class=member.owner_class,
            execution_kind=member.execution_kind,
        )
    return BindingTarget(
        source=source.source,
        compiled_name=compiled_name,
        kind=source.kind,
        owner_class=source.owner_class,
        execution_kind=source.execution_kind,
        required=source.required,
        target_owner_class=source.target_owner_class,
        guards=source.guards,
    )
