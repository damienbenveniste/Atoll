"""Lower typed callables and safe atomic classes into backend-ready units.

The lowerer operates only on scanner evidence and never imports the target
project. It preserves explicit annotations and executable bodies, supplies a
narrow structural owner type for implicit ``self`` and ``cls`` parameters,
applies only analysis-proven generic substitutions, and writes generated source
only inside a caller-owned temporary build tree.
"""

from __future__ import annotations

import ast
import hashlib
import textwrap
from dataclasses import dataclass
from pathlib import Path

from atoll.models import (
    Backend,
    BindingTarget,
    ImportRecord,
    ModuleScan,
    RegionMember,
    RegionSpecialization,
    SymbolId,
    TypedRegion,
)

TYPED_METHOD_GENERATOR_VERSION = "atoll-typed-region-v6"
_SUPPORTED_BINDINGS = frozenset({"instance_method", "staticmethod", "classmethod"})
_SUPPORTED_EXECUTION_KINDS = frozenset({"sync", "generator", "coroutine"})


@dataclass(frozen=True, slots=True)
class TypedRegionGeneration:
    """One deterministic generated source unit for selected region methods.

    The file is temporary build input. ``bindings`` is the public runtime
    promise consumed by the staged-wheel shim; generated source paths must not
    be copied into the final install payload.

    Attributes:
        region: Backend-neutral typed region represented by this record.
        logical_module: Importable module name represented by the compilation unit.
        source_path: Filesystem path of the source module or prepared source.
        source_text: Exact source text retained for analysis or generation.
        source_hash: Deterministic digest of generated or retained source.
        selected_members: Region members emitted by generated source.
        bindings: Source bindings promised by the compiled region or variant.
        backend: Native compiler backend selected for this record.
        specialization: Optional concrete specialization applied during generation.
    """

    region: TypedRegion
    logical_module: str
    source_path: Path
    source_text: str
    source_hash: str
    selected_members: tuple[SymbolId, ...]
    bindings: tuple[BindingTarget, ...]
    backend: Backend
    specialization: RegionSpecialization | None = None


@dataclass(frozen=True, slots=True)
class TypedRegionGenerationOptions:
    """Backend and optional analysis-proven specialization for one generated unit.

    Attributes:
        backend: Native compiler backend selected for this record.
        specialization: Optional concrete specialization applied during generation.
    """

    backend: Backend = "mypyc"
    specialization: RegionSpecialization | None = None


_DEFAULT_GENERATION_OPTIONS = TypedRegionGenerationOptions()


def generate_typed_method_region(
    scan: ModuleScan,
    region: TypedRegion,
    selected_members: tuple[SymbolId, ...],
    *,
    output_path: Path,
    options: TypedRegionGenerationOptions = _DEFAULT_GENERATION_OPTIONS,
) -> TypedRegionGeneration:
    """Write one preserved class, callable set, or concrete specialization.

    Atomic class selections retain the complete class declaration and are
    decorated only with backend directives needed to preserve Python subclass
    and pickle behavior. Ordinary callable selections retain their source
    annotations exactly. A generic
    selection must provide analysis-produced specialization evidence; this
    function never infers substitutions or erases unresolved types. Unsafe
    decorators and unresolved member identifiers fail before a file is written.

    Args:
        scan: Module scan containing source facts for typed-region generation.
        region: Backend-neutral typed region being assessed or generated.
        selected_members: Region members selected for this generated variant.
        output_path: Path where generated source should be written.
        options: Validated command options supplied by the CLI layer.

    Returns:
        TypedRegionGeneration: Generated source, bindings, hash, and specialization metadata.
    """
    members = _selected_region_members(
        region,
        selected_members,
        options.backend,
        options.specialization,
    )
    bindings = tuple(_binding_target(member, options.specialization) for member in members)
    source_text = _generated_source(
        scan,
        region,
        members,
        bindings,
        options,
    )
    source_hash = hashlib.sha256(source_text.encode()).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(source_text, encoding="utf-8")
    return TypedRegionGeneration(
        region=region,
        logical_module=output_path.stem,
        source_path=output_path,
        source_text=source_text,
        source_hash=source_hash,
        selected_members=selected_members,
        bindings=bindings,
        backend=options.backend,
        specialization=options.specialization,
    )


def _selected_region_members(
    region: TypedRegion,
    selected_members: tuple[SymbolId, ...],
    backend: Backend,
    specialization: RegionSpecialization | None,
) -> tuple[RegionMember, ...]:
    if not selected_members:
        raise ValueError("typed region generation requires at least one selected member")
    if len(set(selected_members)) != len(selected_members):
        raise ValueError("typed region generation received duplicate selected members")
    by_id = {member.id: member for member in region.members}
    unknown = tuple(symbol for symbol in selected_members if symbol not in by_id)
    if unknown:
        names = ", ".join(symbol.stable_id for symbol in unknown)
        raise ValueError(f"selected members are outside typed region {region.id}: {names}")
    members = tuple(by_id[symbol] for symbol in selected_members)
    if _is_atomic_class_selection(region, members, specialization):
        _validate_atomic_class_selection(region, members[0])
        return members
    if specialization is not None and (
        len(members) != 1 or members[0].id != specialization.source_member
    ):
        raise ValueError(
            f"specialization {specialization.id} must select only "
            f"{specialization.source_member.stable_id}"
        )
    for member in members:
        _validate_callable_selection(member, backend, specialization)
    return members


def _is_atomic_class_selection(
    region: TypedRegion,
    members: tuple[RegionMember, ...],
    specialization: RegionSpecialization | None,
) -> bool:
    return (
        region.atomic_class
        and specialization is None
        and len(members) == 1
        and members[0].kind == "class"
    )


def _validate_atomic_class_selection(region: TypedRegion, class_member: RegionMember) -> None:
    decisions = {decision.target: decision for decision in region.decisions}
    if any(
        member.kind not in {"class", "method"}
        or (member.kind == "method" and member.execution_kind not in _SUPPORTED_EXECUTION_KINDS)
        or decisions[member.id.stable_id].action != "preserve"
        for member in region.members
    ):
        raise ValueError(
            f"unsupported typed-region binding: atomic class {class_member.id.stable_id}"
        )


def _validate_callable_selection(
    member: RegionMember,
    backend: Backend,
    _specialization: RegionSpecialization | None,
) -> None:
    method_binding = member.kind == "method" and member.binding_kind in _SUPPORTED_BINDINGS
    module_function = member.kind == "function" and member.binding_kind == "module"
    if not method_binding and not module_function:
        raise ValueError(f"unsupported typed-region binding: {member.id.stable_id}")
    execution_supported = member.execution_kind in _SUPPORTED_EXECUTION_KINDS or (
        backend == "cython" and member.execution_kind == "async_generator"
    )
    if not execution_supported:
        raise ValueError(
            f"unsupported typed-region execution kind {member.execution_kind}: "
            f"{member.id.stable_id}"
        )
    unknown_decorators = tuple(
        ast.unparse(decorator)
        for decorator in _callable_node(member).decorator_list
        if _decorator_name(decorator) not in {"staticmethod", "classmethod"}
    )
    if unknown_decorators:
        raise ValueError(
            f"unsupported method decorator(s) for {member.id.stable_id}: "
            f"{', '.join(unknown_decorators)}"
        )


def _generated_source(
    scan: ModuleScan,
    region: TypedRegion,
    members: tuple[RegionMember, ...],
    bindings: tuple[BindingTarget, ...],
    options: TypedRegionGenerationOptions,
) -> str:
    backend = options.backend
    owner_names: set[str] = set()
    for binding in bindings:
        owner = binding.target_owner_class or binding.owner_class
        if owner is not None:
            owner_names.add(owner)
    owners = tuple(sorted(owner_names))
    atomic_class = len(members) == 1 and members[0].kind == "class"
    boundary_roots = _runtime_boundary_roots(region, members)
    sections = _generated_sections(
        scan,
        owners,
        options,
        atomic_class=atomic_class,
        runtime_boundary=bool(boundary_roots),
    )
    if atomic_class:
        sections.extend(("", _lowered_atomic_class(scan, members[0], backend)))
    else:
        binding_by_source = {binding.source: binding for binding in bindings}
        for member in members:
            sections.extend(
                (
                    "",
                    _lowered_callable(
                        scan,
                        member,
                        binding_by_source[member.id],
                        options,
                        boundary_roots,
                    ),
                )
            )
    execution_kinds = {
        binding.compiled_name: binding.execution_kind
        for binding in sorted(bindings, key=lambda item: item.compiled_name)
    }
    sections.extend(("", f"__atoll_execution_kinds__ = {execution_kinds!r}"))
    return "\n".join(sections).rstrip() + "\n"


def _generated_sections(
    scan: ModuleScan,
    owners: tuple[str, ...],
    options: TypedRegionGenerationOptions,
    *,
    atomic_class: bool,
    runtime_boundary: bool,
) -> list[str]:
    backend = options.backend
    specialization = options.specialization
    imports = tuple(
        _preserved_import(scan, record)
        for record in scan.imports
        if not record.source_text.startswith("from __future__ import ")
    )
    constants = tuple(
        record.source_text for record in scan.constants if record.kind == "literal_constant"
    )
    sections: list[str] = ["from __future__ import annotations"]
    if atomic_class and backend == "mypyc":
        sections.extend(
            (
                "",
                "from typing import TYPE_CHECKING",
                "",
                "if TYPE_CHECKING:",
                "    from mypy_extensions import mypyc_attr",
            )
        )
    if backend == "mypyc" and owners:
        sections.extend(("", "from typing import Protocol"))
    if imports:
        sections.extend(("", *dict.fromkeys(imports)))
    if constants:
        sections.extend(("", *constants))
    if runtime_boundary:
        sections.extend(("", f"import {scan.module.name} as _atoll_source"))
    if backend == "mypyc":
        for owner in owners:
            source_owner = (
                specialization.source_owner_class
                if specialization is not None and specialization.target_owner_class == owner
                else None
            )
            sections.extend(
                (
                    "",
                    _owner_facade(
                        scan,
                        owner,
                        source_owner=source_owner,
                        substitutions=(
                            specialization.substitutions if specialization is not None else ()
                        ),
                    ),
                )
            )
    return sections


def _lowered_atomic_class(
    scan: ModuleScan,
    member: RegionMember,
    backend: Backend,
) -> str:
    """Preserve one class declaration and add semantics-preserving backend flags.

    Args:
        scan: Module scan retaining the source package context.
        member: Typed-region member being assessed or generated.
        backend: Compiler backend selected for this operation.

    Returns:
        str: Generated atomic class source and promised bindings.
    """
    node = _class_node(member)
    _rewrite_relative_imports(scan, node)
    if backend == "mypyc":
        node.decorator_list.insert(
            0,
            ast.Call(
                func=ast.Name(id="mypyc_attr", ctx=ast.Load()),
                args=[],
                keywords=[
                    ast.keyword(arg="allow_interpreted_subclasses", value=ast.Constant(True)),
                    ast.keyword(arg="serializable", value=ast.Constant(True)),
                ],
            ),
        )
    return ast.unparse(ast.fix_missing_locations(node))


def _preserved_import(scan: ModuleScan, record: ImportRecord) -> str:
    """Return a normal import valid from the generated top-level module.

    Relative imports retain their import targets and aliases but are resolved to
    an absolute package name because temporary compilation units are not members
    of the target package.

    Args:
        scan: Module scan containing retained source facts.
        record: Cached or report record being converted.

    Returns:
        str: Import source required to preserve generated annotations.

    Raises:
        TypeError: If an import record does not contain an ``ImportFrom`` node.
        ValueError: If a relative import ascends beyond the source package.
    """
    if record.level == 0:
        return record.source_text
    statement = ast.parse(record.source_text).body[0]
    if not isinstance(statement, ast.ImportFrom):
        raise TypeError(f"cannot preserve relative import: {record.source_text}")
    _resolve_relative_import(scan, statement, source_text=record.source_text)
    return ast.unparse(statement)


def _rewrite_relative_imports(scan: ModuleScan, node: ast.AST) -> None:
    """Resolve function-local relative imports against the source package.

    Generated extensions use private top-level module names, so leaving a
    relative import inside a copied callable would resolve against the helper
    module instead of the original package. Rewriting only ``ImportFrom``
    nodes preserves conditional and function-local import timing.

    Args:
        scan: Module scan retaining the original package name and path.
        node: Generated declaration whose executable imports are rewritten.
    """
    _RelativeImportRewriter(scan).visit(node)


class _RelativeImportRewriter(ast.NodeTransformer):
    """Rewrite relative imports without moving them across execution scopes."""

    def __init__(self, scan: ModuleScan) -> None:
        self.scan = scan

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
        """Return an equivalent absolute import for generated code.

        Args:
            node: Relative or absolute import encountered in a declaration.

        Returns:
            ast.ImportFrom: The original absolute import or a resolved clone.
        """
        if node.level == 0:
            return node
        resolved = ast.ImportFrom(
            module=node.module,
            names=node.names,
            level=node.level,
        )
        _resolve_relative_import(self.scan, resolved, source_text=ast.unparse(node))
        return ast.copy_location(resolved, node)


def _resolve_relative_import(
    scan: ModuleScan,
    statement: ast.ImportFrom,
    *,
    source_text: str,
) -> None:
    """Resolve one relative import against its original module package.

    Args:
        scan: Module scan retaining the original package name and path.
        statement: Relative import node to mutate into an absolute import.
        source_text: Source spelling used in conservative failure diagnostics.

    Raises:
        ValueError: The source import ascends beyond its importable package.
    """
    if statement.level == 0:
        return
    package_parts = scan.module.name.split(".")
    if scan.module.path.name != "__init__.py":
        package_parts = package_parts[:-1]
    ascend = statement.level - 1
    if not package_parts or ascend >= len(package_parts):
        raise ValueError(f"relative import escapes package in {scan.module.name}: {source_text}")
    prefix = package_parts[: len(package_parts) - ascend]
    module_parts = statement.module.split(".") if statement.module else []
    absolute_module = ".".join((*prefix, *module_parts))
    if not absolute_module:
        raise ValueError(f"cannot preserve relative import: {source_text}")
    statement.level = 0
    statement.module = absolute_module


def _owner_facade(
    scan: ModuleScan,
    owner: str,
    *,
    source_owner: str | None = None,
    substitutions: tuple[tuple[str, str], ...] = (),
) -> str:
    """Render a structural receiver facade, including safe inherited evidence.

    Args:
        scan: Module scan containing retained source facts.
        owner: Owner class or symbol for the current operation.
        source_owner: Source owner class declared for the method.
        substitutions: Concrete generic type substitutions.

    Returns:
        str: Generated source facade for the selected owner class.
    """
    owner_order = tuple(dict.fromkeys(item for item in (source_owner, owner) if item is not None))
    class_symbols = tuple(
        symbol
        for owner_name in owner_order
        for symbol in scan.symbols
        if symbol.kind == "class" and symbol.id.qualname == owner_name
    )
    body: list[ast.stmt] = []
    fields = {field.name: field for class_symbol in class_symbols for field in class_symbol.fields}
    for field in fields.values():
        annotation = ast.parse(field.annotation, mode="eval").body
        if substitutions:
            annotation = _substituted_annotation(annotation, _parsed_substitutions(substitutions))
        body.append(
            ast.AnnAssign(
                target=ast.Name(id=field.name, ctx=ast.Store()),
                annotation=annotation,
                value=None,
                simple=1,
            )
        )
    methods = {
        symbol.id.qualname.rsplit(".", maxsplit=1)[-1]: symbol
        for owner_name in owner_order
        for symbol in scan.symbols
        if symbol.kind == "method" and symbol.owner_class == owner_name
    }
    substitution_names = {name for name, _ in substitutions}
    for symbol in methods.values():
        if set(symbol.scope_type_parameters) - substitution_names:
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
        node = _callable_node(member)
        if substitutions:
            _specialize_annotations(node, substitutions)
            node.type_params = []
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


def _lowered_callable(
    scan: ModuleScan,
    member: RegionMember,
    binding: BindingTarget,
    options: TypedRegionGenerationOptions,
    boundary_roots: frozenset[str],
) -> str:
    backend = options.backend
    specialization = options.specialization
    node = _callable_node(member)
    _rewrite_relative_imports(scan, node)
    node.name = binding.compiled_name
    node.decorator_list = []
    if backend == "cython" and specialization is None:
        _lower_cython_type_parameters(node, member)
    _rewrite_runtime_boundaries(node, boundary_roots)
    if specialization is not None:
        _specialize_annotations(node, specialization.substitutions)
        node.type_params = []
    if backend == "cython":
        return ast.unparse(ast.fix_missing_locations(node))
    if binding.kind != "module":
        _lower_mypyc_method_receiver(node, member, binding)
    return ast.unparse(ast.fix_missing_locations(node))


def _lower_mypyc_method_receiver(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    member: RegionMember,
    binding: BindingTarget,
) -> None:
    owner = binding.target_owner_class or _required_owner(member)
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


def _lower_cython_type_parameters(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    member: RegionMember,
) -> None:
    """Remove unsupported PEP 695 syntax from a Cython-private callable.

    Source annotation expressions remain unchanged and the staged runtime shim
    restores metadata from the original Python binding. A type parameter used
    by executable code cannot be removed safely and leaves the member
    interpreted instead.

    Args:
        node: Private generated callable submitted to Cython.
        member: Source member retaining declared type-parameter evidence.

    Raises:
        ValueError: A declared type parameter is used by runtime code.
    """
    names = frozenset(member.type_parameters)
    if not names or not node.type_params:
        return
    visitor = _RuntimeTypeParameterUseVisitor(names)
    for expression in (*node.args.defaults, *node.args.kw_defaults):
        if expression is not None:
            visitor.visit(expression)
    for statement in node.body:
        visitor.visit(statement)
    if visitor.used:
        raise ValueError(
            "Cython boxed lowering cannot remove runtime type parameter(s): "
            + ", ".join(sorted(visitor.used))
        )
    node.type_params = []


class _RuntimeTypeParameterUseVisitor(ast.NodeVisitor):
    """Find executable TypeVar name loads while skipping annotation syntax."""

    def __init__(self, names: frozenset[str]) -> None:
        self.names = names
        self.used: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        """Record one runtime type-parameter load.

        Args:
            node: Syntax node being visited without executing target code.
        """
        if isinstance(node.ctx, ast.Load) and node.id in self.names:
            self.used.add(node.id)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Visit assignment runtime expressions without traversing its annotation.

        Args:
            node: Syntax node being visited without executing target code.
        """
        self.visit(node.target)
        if node.value is not None:
            self.visit(node.value)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit nested function runtime expressions without annotations.

        Args:
            node: Syntax node being visited without executing target code.
        """
        self._visit_nested_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit nested async runtime expressions without annotations.

        Args:
            node: Syntax node being visited without executing target code.
        """
        self._visit_nested_function(node)

    def _visit_nested_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for expression in (*node.args.defaults, *node.args.kw_defaults):
            if expression is not None:
                self.visit(expression)
        for statement in node.body:
            self.visit(statement)


def _runtime_boundary_roots(
    region: TypedRegion,
    members: tuple[RegionMember, ...],
) -> frozenset[str]:
    selected = frozenset(member.id for member in members)
    member_by_id = {member.id: member for member in members}
    roots: set[str] = set()
    for dependency in region.dependencies:
        if dependency.src not in selected or dependency.role != "runtime":
            continue
        if dependency.requires_same_unit:
            continue
        if isinstance(dependency.dst, str):
            if dependency.kind == "uses_global" and dependency.dst.isidentifier():
                roots.add(dependency.dst)
            continue
        if dependency.dst.module != region.source_module.name or dependency.dst in selected:
            continue
        source_member = member_by_id[dependency.src]
        if not _uses_receiver_dispatch(source_member, dependency.dst, dependency.lineno):
            roots.add(dependency.dst.qualname.split(".", maxsplit=1)[0])
    return frozenset(roots)


def _uses_receiver_dispatch(
    member: RegionMember,
    destination: SymbolId,
    lineno: int | None,
) -> bool:
    if "." not in destination.qualname:
        return False
    method_name = destination.qualname.rsplit(".", maxsplit=1)[-1]
    return any(
        call.target in {f"self.{method_name}", f"cls.{method_name}"}
        and (lineno is None or call.lineno == lineno)
        for call in member.call_sites
    )


def _rewrite_runtime_boundaries(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    roots: frozenset[str],
) -> None:
    if not roots:
        return
    rewriter = _RuntimeBoundaryRewriter(roots)
    rewritten_body: list[ast.stmt] = []
    for statement in node.body:
        rewritten = rewriter.visit(statement)
        if not isinstance(rewritten, ast.stmt):
            raise TypeError("runtime-boundary rewrite produced a non-statement")
        rewritten_body.append(rewritten)
    node.body = rewritten_body
    rewritten_defaults: list[ast.expr] = []
    for default in node.args.defaults:
        rewritten = rewriter.visit(default)
        if not isinstance(rewritten, ast.expr):
            raise TypeError("runtime-boundary default rewrite produced a non-expression")
        rewritten_defaults.append(rewritten)
    node.args.defaults = rewritten_defaults
    node.args.kw_defaults = [
        _rewrite_optional_runtime_expression(rewriter, default) for default in node.args.kw_defaults
    ]


def _rewrite_optional_runtime_expression(
    rewriter: _RuntimeBoundaryRewriter,
    expression: ast.expr | None,
) -> ast.expr | None:
    if expression is None:
        return None
    rewritten = rewriter.visit(expression)
    if not isinstance(rewritten, ast.expr):
        raise TypeError("runtime-boundary expression rewrite produced a non-expression")
    return rewritten


class _RuntimeBoundaryRewriter(ast.NodeTransformer):
    """Route omitted same-module globals through the live source module."""

    def __init__(self, roots: frozenset[str]) -> None:
        self.roots = roots

    def visit_Name(self, node: ast.Name) -> ast.expr:
        """Rewrite one loaded boundary root without touching stores.

        Args:
            node: Syntax node being visited without executing target code.

        Returns:
            ast.expr: Live source-module lookup or the unchanged name.
        """
        if isinstance(node.ctx, ast.Load) and node.id in self.roots:
            return ast.copy_location(
                ast.Attribute(
                    value=ast.Name(id="_atoll_source", ctx=ast.Load()),
                    attr=node.id,
                    ctx=node.ctx,
                ),
                node,
            )
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AnnAssign:
        """Rewrite runtime parts of an annotated assignment, not its annotation.

        Args:
            node: Syntax node being visited without executing target code.

        Returns:
            ast.AnnAssign: Assignment with source annotation preserved exactly.

        Raises:
            TypeError: If rewriting produces a non-expression assignment target or value.
        """
        target = self.visit(node.target)
        if not isinstance(target, ast.Name | ast.Attribute | ast.Subscript):
            raise TypeError("runtime-boundary assignment target is not an expression")
        value = self.visit(node.value) if node.value is not None else None
        if value is not None and not isinstance(value, ast.expr):
            raise TypeError("runtime-boundary assignment value is not an expression")
        return ast.copy_location(
            ast.AnnAssign(
                target=target,
                annotation=node.annotation,
                value=value,
                simple=node.simple,
            ),
            node,
        )


def _callable_node(member: RegionMember) -> ast.FunctionDef | ast.AsyncFunctionDef:
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


def _class_node(member: RegionMember) -> ast.ClassDef:
    tree = ast.parse(textwrap.dedent(member.source_text))
    node = next((statement for statement in tree.body if isinstance(statement, ast.ClassDef)), None)
    if node is None:
        raise ValueError(f"member source is not a class declaration: {member.id.stable_id}")
    return node


def _first_positional_parameter(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ast.arg | None:
    parameters = (*node.args.posonlyargs, *node.args.args)
    return parameters[0] if parameters else None


def _binding_target(
    member: RegionMember,
    specialization: RegionSpecialization | None,
) -> BindingTarget:
    target_owner = specialization.target_owner_class if specialization is not None else None
    compiled_name = member.id.qualname.replace(".", "__")
    if specialization is not None:
        target = target_owner or member.id.qualname
        label = target.replace(".", "__")
        member_name = member.id.qualname.rsplit(".", maxsplit=1)[-1]
        compiled_name = f"{label}__{member_name}__{specialization.id[-12:]}"
    return BindingTarget(
        source=member.id,
        compiled_name=compiled_name,
        kind=member.binding_kind,
        owner_class=member.owner_class,
        execution_kind=member.execution_kind,
        target_owner_class=target_owner,
        guards=specialization.guards if specialization is not None else (),
    )


def _specialize_annotations(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    substitutions: tuple[tuple[str, str], ...],
) -> None:
    """Apply complete TypeVar substitutions only inside annotation positions.

    Args:
        node: Syntax node being visited without executing target code.
        substitutions: Concrete generic type substitutions.

    Raises:
        ValueError: If specialization is requested without any concrete substitutions.
    """
    if not substitutions:
        raise ValueError("generic specialization requires at least one type substitution")
    parsed = _parsed_substitutions(substitutions)
    parameters = (
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
        *((node.args.vararg,) if node.args.vararg is not None else ()),
        *((node.args.kwarg,) if node.args.kwarg is not None else ()),
    )
    for parameter in parameters:
        if parameter.annotation is not None:
            parameter.annotation = _substituted_annotation(parameter.annotation, parsed)
    if node.returns is not None:
        node.returns = _substituted_annotation(node.returns, parsed)
    for descendant in ast.walk(node):
        if isinstance(descendant, ast.AnnAssign):
            descendant.annotation = _substituted_annotation(descendant.annotation, parsed)
    unresolved = {
        name
        for annotation in (
            *(parameter.annotation for parameter in parameters if parameter.annotation is not None),
            *((node.returns,) if node.returns is not None else ()),
            *(
                descendant.annotation
                for descendant in ast.walk(node)
                if isinstance(descendant, ast.AnnAssign)
            ),
        )
        for name in parsed
        if any(isinstance(item, ast.Name) and item.id == name for item in ast.walk(annotation))
    }
    if unresolved:
        raise ValueError(
            "generic specialization left unresolved type parameter(s): "
            + ", ".join(sorted(unresolved))
        )


def _parsed_substitutions(
    substitutions: tuple[tuple[str, str], ...],
) -> dict[str, ast.expr]:
    return {name: ast.parse(annotation, mode="eval").body for name, annotation in substitutions}


def _substituted_annotation(
    annotation: ast.expr,
    substitutions: dict[str, ast.expr],
) -> ast.expr:
    expression = _unquoted_annotation(annotation)
    rewritten = _TypeParameterAnnotationRewriter(substitutions).visit(expression)
    if not isinstance(rewritten, ast.expr):
        raise TypeError("type-parameter specialization produced a non-expression")
    return rewritten


class _TypeParameterAnnotationRewriter(ast.NodeTransformer):
    """Replace TypeVar names in one annotation without touching runtime code."""

    def __init__(self, substitutions: dict[str, ast.expr]) -> None:
        self.substitutions = substitutions

    def visit_Name(self, node: ast.Name) -> ast.expr:
        """Return a fresh parsed replacement for a specialized TypeVar name.

        Args:
            node: Syntax node being visited without executing target code.

        Returns:
            ast.expr: Rewritten name expression or visitor-specific result.
        """
        replacement = self.substitutions.get(node.id)
        if replacement is None:
            return node
        return ast.copy_location(ast.parse(ast.unparse(replacement), mode="eval").body, node)


def _required_owner(member: RegionMember) -> str:
    if member.owner_class is None:
        raise ValueError(f"method has no owner class: {member.id.stable_id}")
    return member.owner_class


def _rewrite_signature_owner_types(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    owner: str,
) -> None:
    """Specialize method-scoped owner types for a top-level generated callable.

    Args:
        node: Syntax node being visited without executing target code.
        owner: Owner class or symbol for the current operation.
    """
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
        """Return one type expression with owner references specialized.

        Args:
            annotation: Source annotation expression being inspected or rewritten.

        Returns:
            ast.expr: Rewritten annotation or expression text.

        Raises:
            TypeError: If annotation rewriting produces a non-expression syntax node.
        """
        expression = _unquoted_annotation(annotation)
        rewritten = self.visit(expression)
        if not isinstance(rewritten, ast.expr):
            raise TypeError("owner annotation lowering produced a non-expression")
        return rewritten

    def visit_Name(self, node: ast.Name) -> ast.expr:
        """Specialize direct owner and Self names without changing other types.

        Args:
            node: Syntax node being visited without executing target code.

        Returns:
            ast.expr: Rewritten name expression or visitor-specific result.
        """
        if node.id in {self.owner, "Self"}:
            return ast.copy_location(
                ast.Name(id=_facade_name(self.owner), ctx=ast.Load()),
                node,
            )
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
        """Specialize qualified typing.Self references.

        Args:
            node: Syntax node being visited without executing target code.

        Returns:
            ast.expr: Rewritten attribute expression or visitor-specific result.

        Raises:
            TypeError: If recursive attribute rewriting produces a non-expression syntax node.
        """
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
    return ast.unparse(decorator)
