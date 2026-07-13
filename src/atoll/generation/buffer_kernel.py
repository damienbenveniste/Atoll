"""Generate zero-copy Cython kernels from conservative buffer proofs.

The generator consumes one frontend-authorized unsigned-byte reduction and
emits a disposable Cython unit. Runtime exact-type, layout, and length guards
remain owned by the staged dispatcher; this module assumes they pass before
native entry and never catches overflow to retry interpreted code.
"""

from __future__ import annotations

import ast
import hashlib
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

from atoll.generation.scalar_kernel import scalar_binding, scalar_wrapper_parameter_declarations
from atoll.generation.typed_region import TypedRegionGeneration
from atoll.models import ModuleScan, RegionMember, TypedRegion
from atoll.native_optimization.buffer_analysis import BufferKernelPlan

BUFFER_KERNEL_GENERATOR_VERSION = "atoll-buffer-kernel-v1"


@dataclass(frozen=True, slots=True)
class BufferKernelGeneration:
    """Generated Cython source and retained zero-copy proof evidence.

    Attributes:
        generation: Standard typed-region generation contract used by packaging.
        plan: Revalidated buffer plan that authorized native lowering.
        compiled_name: Private extension attribute installed by the staged dispatcher.
    """

    generation: TypedRegionGeneration
    plan: BufferKernelPlan
    compiled_name: str


@dataclass(frozen=True, slots=True)
class BufferKernelGenerationRequest:
    """Staged source evidence and output identity for one buffer kernel.

    Attributes:
        scan: Module scan containing the retained declaration.
        region: Typed region owning the selected callable.
        plan: Revalidated zero-copy proof plan.
        logical_module: Private extension module name used by the wheel payload.
        output_path: Disposable `.pyx` path under the source-clean build root.
    """

    scan: ModuleScan
    region: TypedRegion
    plan: BufferKernelPlan
    logical_module: str
    output_path: Path


class _BufferNameLowering(ast.NodeTransformer):
    """Retarget source buffer reads to a typed contiguous memoryview local."""

    def __init__(self, source_name: str, view_name: str) -> None:
        self._source_name = source_name
        self._view_name = view_name

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Replace only reads of the proven buffer parameter.

        Args:
            node: Source name expression.

        Returns:
            ast.Name: Typed-view reference for buffer reads, otherwise the source name.
        """
        if node.id == self._source_name and isinstance(node.ctx, ast.Load):
            return ast.copy_location(ast.Name(id=self._view_name, ctx=node.ctx), node)
        return node


def generate_buffer_kernel(request: BufferKernelGenerationRequest) -> BufferKernelGeneration:
    """Write one proof-authorized unsigned-byte Cython reduction.

    Args:
        request: Staged scan, region, plan, module identity, and output path.

    Returns:
        BufferKernelGeneration: Generated source, binding, and proof evidence.

    Raises:
        ValueError: If staged source or proof evidence differs, the plan contains an
            unsupported layout, or the retained declaration is not one supported reduction.
    """
    if request.output_path.suffix != ".pyx":
        raise ValueError("buffer kernels require a .pyx output path")
    member = next(
        (item for item in request.region.members if item.id == request.plan.member),
        None,
    )
    if member is None:
        raise ValueError(
            f"buffer plan member is absent from staged region: {request.plan.member.stable_id}"
        )
    if hashlib.sha256(member.source_text.encode("utf-8")).hexdigest() != request.plan.source_hash:
        raise ValueError("staged buffer member source differs from analyzed source")
    if len(request.plan.buffers) != 1:
        raise ValueError("buffer lowering currently requires exactly one buffer parameter")
    buffer = request.plan.buffers[0]
    if (
        buffer.layout.format != "B"
        or buffer.layout.itemsize != 1
        or buffer.layout.ndim != 1
        or not buffer.layout.c_contiguous
    ):
        raise ValueError("buffer lowering currently requires a contiguous unsigned-byte layout")
    node = _buffer_callable_node(member)
    _validate_reduction_shape(node, request.plan)
    compiled_name = _compiled_name(member)
    source_text = _render_source(member, node, request.plan, compiled_name)
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
    return BufferKernelGeneration(
        generation=generation,
        plan=request.plan,
        compiled_name=compiled_name,
    )


def _buffer_callable_node(member: RegionMember) -> ast.FunctionDef:
    parsed = ast.parse(textwrap.dedent(member.source_text))
    declarations = tuple(node for node in parsed.body if isinstance(node, ast.FunctionDef))
    if len(declarations) != 1 or len(parsed.body) != 1:
        raise ValueError("buffer kernel source must contain one synchronous function")
    return declarations[0]


def _validate_reduction_shape(node: ast.FunctionDef, plan: BufferKernelPlan) -> None:
    if plan.reduction not in {"add", "xor", "count"}:
        raise ValueError(f"unsupported buffer reduction: {plan.reduction}")
    body = _without_docstring(node.body)
    accumulator = plan.returns[0].accumulator if plan.returns else None
    if accumulator is None:
        raise ValueError("buffer reduction has no direct accumulator return")
    initializers = tuple(
        statement for statement in body if _assignment_name(statement) == accumulator
    )
    if len(initializers) != 1:
        raise ValueError("buffer reduction requires one accumulator initializer")
    first = initializers[0]
    value = _assignment_value(first)
    if not isinstance(value, ast.Constant) or value.value != 0:
        raise ValueError("buffer reduction accumulator must initialize to exact zero")
    returns = tuple(statement for statement in body if isinstance(statement, ast.Return))
    if (
        len(returns) != 1
        or not isinstance(returns[0].value, ast.Name)
        or returns[0].value.id != accumulator
    ):
        raise ValueError("buffer reduction must return its accumulator directly")


def _render_source(
    member: RegionMember,
    node: ast.FunctionDef,
    plan: BufferKernelPlan,
    compiled_name: str,
) -> str:
    buffer = plan.buffers[0]
    used_names = {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}
    used_names.update(argument.arg for argument in node.args.posonlyargs)
    used_names.update(argument.arg for argument in node.args.args)
    used_names.update(argument.arg for argument in node.args.kwonlyargs)
    view_name = _fresh_private_name(f"_atoll_view_{buffer.name}", used_names)
    used_names.add(view_name)
    element_type = _fresh_private_name("_atoll_uint8_t", used_names)
    used_names.add(element_type)
    accumulator_type = _fresh_private_name("_atoll_uint64_t", used_names)
    used_names.add(accumulator_type)
    index_type = _fresh_private_name("_atoll_size_t", used_names)
    transformed = _BufferNameLowering(buffer.name, view_name).visit(node)
    if not isinstance(transformed, ast.FunctionDef):
        raise TypeError("buffer name lowering produced a non-function")
    body = _without_docstring(transformed.body)
    accumulator = plan.returns[0].accumulator
    if accumulator is None:
        raise ValueError("buffer reduction has no direct accumulator return")
    element_names, index_names = _loop_names(transformed, view_name)
    sections = [
        "# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=False",
        f"from libc.stdint cimport uint8_t as {element_type}",
        f"from libc.stdint cimport uint64_t as {accumulator_type}",
        f"from libc.stddef cimport size_t as {index_type}",
        "",
        f"# atoll buffer proof {BUFFER_KERNEL_GENERATOR_VERSION}",
        f"# reduction={plan.reduction}; format={buffer.layout.format}; itemsize=1",
        "",
        f"def {compiled_name}({scalar_wrapper_parameter_declarations(member)}):",
        f"    cdef const {element_type}[::1] {view_name} = {buffer.name}",
        f"    cdef {accumulator_type} {accumulator}",
    ]
    if element_names:
        sections.append(f"    cdef {element_type} {', '.join(element_names)}")
    if index_names:
        sections.append(f"    cdef {index_type} {', '.join(index_names)}")
    sections.extend(_render_statement(statement) for statement in body)
    sections.extend(
        (
            "",
            f"__atoll_execution_kinds__ = {{{compiled_name!r}: 'sync'}}",
            "",
        )
    )
    return "\n".join(sections)


def _loop_names(node: ast.FunctionDef, view_name: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    elements: set[str] = set()
    indexes: set[str] = set()
    for descendant in ast.walk(node):
        if not isinstance(descendant, ast.For) or not isinstance(descendant.target, ast.Name):
            continue
        if isinstance(descendant.iter, ast.Name) and descendant.iter.id == view_name:
            elements.add(descendant.target.id)
        else:
            indexes.add(descendant.target.id)
    return tuple(sorted(elements)), tuple(sorted(indexes))


def _without_docstring(statements: list[ast.stmt]) -> list[ast.stmt]:
    body = list(statements)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body.pop(0)
    return body


def _render_statement(statement: ast.stmt) -> str:
    if isinstance(statement, ast.AnnAssign):
        if statement.value is None:
            raise ValueError("buffer local annotation requires an initializer")
        statement = ast.copy_location(
            ast.Assign(targets=[statement.target], value=statement.value),
            statement,
        )
    return textwrap.indent(ast.unparse(statement), "    ")


def _assignment_name(statement: ast.stmt) -> str | None:
    if (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    ):
        return statement.targets[0].id
    if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
        return statement.target.id
    return None


def _assignment_value(statement: ast.stmt) -> ast.expr | None:
    if isinstance(statement, ast.Assign | ast.AnnAssign):
        return statement.value
    return None


def _fresh_private_name(base: str, used_names: set[str]) -> str:
    candidate = base
    suffix = 1
    while candidate in used_names:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _compiled_name(member: RegionMember) -> str:
    stem = re.sub(r"[^A-Za-z0-9_]", "_", member.id.qualname)
    return f"_atoll_buffer_{stem}_B"
