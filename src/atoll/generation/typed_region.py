"""Lower typed class methods into backend-ready source-clean compilation units.

The lowerer operates only on scanner evidence and never imports the target
project. It preserves explicit annotations and executable bodies, supplies a
narrow structural owner type for implicit ``self`` and ``cls`` parameters, and
writes generated source only inside a caller-owned temporary build tree.
"""

from __future__ import annotations

import ast
import hashlib
import textwrap
from dataclasses import dataclass
from pathlib import Path

from atoll.models import (
    BindingTarget,
    ImportRecord,
    ModuleScan,
    RegionMember,
    SymbolId,
    TypedRegion,
)

TYPED_METHOD_GENERATOR_VERSION = "atoll-typed-method-v1"
_SUPPORTED_BINDINGS = frozenset({"instance_method", "staticmethod", "classmethod"})
_SUPPORTED_EXECUTION_KINDS = frozenset({"sync", "generator", "coroutine"})


@dataclass(frozen=True, slots=True)
class TypedRegionGeneration:
    """One deterministic generated source unit for selected region methods.

    The file is temporary build input. ``bindings`` is the public runtime
    promise consumed by the staged-wheel shim; generated source paths must not
    be copied into the final install payload.
    """

    region: TypedRegion
    logical_module: str
    source_path: Path
    source_text: str
    source_hash: str
    selected_members: tuple[SymbolId, ...]
    bindings: tuple[BindingTarget, ...]


def generate_typed_method_region(
    scan: ModuleScan,
    region: TypedRegion,
    selected_members: tuple[SymbolId, ...],
    *,
    logical_module: str,
    output_path: Path,
) -> TypedRegionGeneration:
    """Write one preserved typed-method module for a native backend.

    Only method-level binding is supported here. Async generators deliberately
    remain for the Cython milestone, and unsafe decorators or unresolved member
    identifiers fail before any generated file is written.
    """
    members = _selected_region_members(region, selected_members)
    bindings = tuple(_binding_target(member) for member in members)
    source_text = _generated_source(scan, members, bindings)
    source_hash = hashlib.sha256(source_text.encode()).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(source_text, encoding="utf-8")
    return TypedRegionGeneration(
        region=region,
        logical_module=logical_module,
        source_path=output_path,
        source_text=source_text,
        source_hash=source_hash,
        selected_members=selected_members,
        bindings=bindings,
    )


def _selected_region_members(
    region: TypedRegion,
    selected_members: tuple[SymbolId, ...],
) -> tuple[RegionMember, ...]:
    if not selected_members:
        raise ValueError("typed method generation requires at least one selected member")
    if len(set(selected_members)) != len(selected_members):
        raise ValueError("typed method generation received duplicate selected members")
    by_id = {member.id: member for member in region.members}
    unknown = tuple(symbol for symbol in selected_members if symbol not in by_id)
    if unknown:
        names = ", ".join(symbol.stable_id for symbol in unknown)
        raise ValueError(f"selected members are outside typed region {region.id}: {names}")
    members = tuple(by_id[symbol] for symbol in selected_members)
    for member in members:
        if member.kind != "method" or member.binding_kind not in _SUPPORTED_BINDINGS:
            raise ValueError(f"unsupported typed-region binding: {member.id.stable_id}")
        if member.execution_kind not in _SUPPORTED_EXECUTION_KINDS:
            raise ValueError(
                f"unsupported typed-region execution kind {member.execution_kind}: "
                f"{member.id.stable_id}"
            )
        decorators = _method_node(member).decorator_list
        unknown_decorators = tuple(
            ast.unparse(decorator)
            for decorator in decorators
            if _decorator_name(decorator) not in {"staticmethod", "classmethod"}
        )
        if unknown_decorators:
            raise ValueError(
                f"unsupported method decorator(s) for {member.id.stable_id}: "
                f"{', '.join(unknown_decorators)}"
            )
    return members


def _generated_source(
    scan: ModuleScan,
    members: tuple[RegionMember, ...],
    bindings: tuple[BindingTarget, ...],
) -> str:
    owners = tuple(sorted({member.owner_class for member in members if member.owner_class}))
    imports = tuple(
        _preserved_import(scan, record)
        for record in scan.imports
        if not record.source_text.startswith("from __future__ import ")
    )
    constants = tuple(
        record.source_text for record in scan.constants if record.kind == "literal_constant"
    )
    sections: list[str] = [
        "from __future__ import annotations",
        "",
        "from typing import Protocol",
    ]
    if imports:
        sections.extend(("", *dict.fromkeys(imports)))
    if constants:
        sections.extend(("", *constants))
    for owner in owners:
        sections.extend(("", _owner_facade(scan, owner)))
    binding_by_source = {binding.source: binding for binding in bindings}
    for member in members:
        sections.extend(("", _lowered_method(member, binding_by_source[member.id])))
    return "\n".join(sections).rstrip() + "\n"


def _preserved_import(scan: ModuleScan, record: ImportRecord) -> str:
    """Return a normal import valid from the generated top-level module.

    Relative imports retain their import targets and aliases but are resolved to
    an absolute package name because temporary compilation units are not members
    of the target package.
    """
    if record.level == 0:
        return record.source_text
    package_parts = scan.module.name.split(".")
    if scan.module.path.name != "__init__.py":
        package_parts = package_parts[:-1]
    ascend = record.level - 1
    if ascend > len(package_parts):
        raise ValueError(
            f"relative import escapes package in {scan.module.name}: {record.source_text}"
        )
    prefix = package_parts[: len(package_parts) - ascend]
    module_parts = record.module.split(".") if record.module else []
    absolute_module = ".".join((*prefix, *module_parts))
    statement = ast.parse(record.source_text).body[0]
    if not isinstance(statement, ast.ImportFrom) or not absolute_module:
        raise ValueError(f"cannot preserve relative import: {record.source_text}")
    statement.level = 0
    statement.module = absolute_module
    return ast.unparse(statement)


def _owner_facade(scan: ModuleScan, owner: str) -> str:
    class_symbol = next(
        (
            symbol
            for symbol in scan.symbols
            if symbol.kind == "class" and symbol.id.qualname == owner
        ),
        None,
    )
    body: list[ast.stmt] = []
    if class_symbol is not None:
        body.extend(
            ast.AnnAssign(
                target=ast.Name(id=field.name, ctx=ast.Store()),
                annotation=ast.parse(field.annotation, mode="eval").body,
                value=None,
                simple=1,
            )
            for field in class_symbol.fields
        )
    for symbol in scan.symbols:
        if symbol.kind != "method" or symbol.owner_class != owner:
            continue
        if symbol.return_annotation is None:
            continue
        parameters = symbol.parameters
        visible_parameters = parameters[1:] if symbol.binding_kind != "staticmethod" else parameters
        if any(parameter.annotation is None for parameter in visible_parameters):
            continue
        member = next(
            (
                candidate
                for region in scan.typed_regions
                for candidate in region.members
                if candidate.id == symbol.id
            ),
            None,
        )
        if member is None:
            continue
        node = _method_node(member)
        node.body = [ast.Expr(value=ast.Constant(value=Ellipsis))]
        node.decorator_list = [
            decorator
            for decorator in node.decorator_list
            if _decorator_name(decorator) in {"staticmethod", "classmethod"}
        ]
        body.append(node)
    if not body:
        body.append(ast.Pass())
    facade = ast.ClassDef(
        name=_facade_name(owner),
        bases=[ast.Name(id="Protocol", ctx=ast.Load())],
        keywords=[],
        body=body,
        decorator_list=[],
        type_params=[],
    )
    return ast.unparse(ast.fix_missing_locations(facade))


def _lowered_method(member: RegionMember, binding: BindingTarget) -> str:
    node = _method_node(member)
    node.name = binding.compiled_name
    node.decorator_list = []
    owner = _required_owner(member)
    _rewrite_signature_owner_types(node, owner)
    first = _first_positional_parameter(node)
    if member.binding_kind == "instance_method":
        if first is None:
            raise ValueError(f"instance method has no self parameter: {member.id.stable_id}")
        if first.annotation is None:
            first.annotation = ast.Name(id=_facade_name(owner), ctx=ast.Load())
    elif member.binding_kind == "classmethod":
        if first is None:
            raise ValueError(f"classmethod has no cls parameter: {member.id.stable_id}")
        if first.annotation is None:
            first.annotation = ast.Subscript(
                value=ast.Name(id="type", ctx=ast.Load()),
                slice=ast.Name(id=_facade_name(owner), ctx=ast.Load()),
                ctx=ast.Load(),
            )
    return ast.unparse(ast.fix_missing_locations(node))


def _method_node(member: RegionMember) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(textwrap.dedent(member.source_text))
    node = next(
        (
            statement
            for statement in tree.body
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        ),
        None,
    )
    if node is None:
        raise ValueError(f"member source is not a method declaration: {member.id.stable_id}")
    return node


def _first_positional_parameter(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ast.arg | None:
    parameters = (*node.args.posonlyargs, *node.args.args)
    return parameters[0] if parameters else None


def _binding_target(member: RegionMember) -> BindingTarget:
    return BindingTarget(
        source=member.id,
        compiled_name=member.id.qualname.replace(".", "__"),
        kind=member.binding_kind,
        owner_class=member.owner_class,
        execution_kind=member.execution_kind,
    )


def _required_owner(member: RegionMember) -> str:
    if member.owner_class is None:
        raise ValueError(f"method has no owner class: {member.id.stable_id}")
    return member.owner_class


def _rewrite_signature_owner_types(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    owner: str,
) -> None:
    """Specialize method-scoped owner types for a top-level generated callable."""
    rewriter = _OwnerAnnotationRewriter(owner)
    parameters = (
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
        *((node.args.vararg,) if node.args.vararg is not None else ()),
        *((node.args.kwarg,) if node.args.kwarg is not None else ()),
    )
    for parameter in parameters:
        if parameter.annotation is not None:
            parameter.annotation = rewriter.rewrite(parameter.annotation)
    if node.returns is not None:
        node.returns = rewriter.rewrite(node.returns)


class _OwnerAnnotationRewriter(ast.NodeTransformer):
    """Replace owner and Self references with a structural owner facade."""

    def __init__(self, owner: str) -> None:
        self.owner = owner

    def rewrite(self, annotation: ast.expr) -> ast.expr:
        """Return one type expression with owner references specialized."""
        expression = _unquoted_annotation(annotation)
        rewritten = self.visit(expression)
        if not isinstance(rewritten, ast.expr):
            raise TypeError("owner annotation lowering produced a non-expression")
        return rewritten

    def visit_Name(self, node: ast.Name) -> ast.expr:
        """Specialize direct owner and Self names without changing other types."""
        if node.id in {self.owner, "Self"}:
            return ast.copy_location(
                ast.Name(id=_facade_name(self.owner), ctx=ast.Load()),
                node,
            )
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
        """Specialize qualified typing.Self references."""
        if node.attr == "Self":
            return ast.copy_location(
                ast.Name(id=_facade_name(self.owner), ctx=ast.Load()),
                node,
            )
        rewritten = self.generic_visit(node)
        if not isinstance(rewritten, ast.expr):
            raise TypeError("qualified annotation lowering produced a non-expression")
        return rewritten


def _unquoted_annotation(annotation: ast.expr) -> ast.expr:
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            return ast.parse(annotation.value, mode="eval").body
        except SyntaxError:
            return annotation
    return annotation


def _facade_name(owner: str) -> str:
    return f"_Atoll{owner.replace('.', '_')}"


def _decorator_name(decorator: ast.expr) -> str:
    if isinstance(decorator, ast.Name):
        return decorator.id
    if isinstance(decorator, ast.Attribute):
        return decorator.attr
    return ast.unparse(decorator)
