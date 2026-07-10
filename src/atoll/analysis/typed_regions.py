"""Build backend-neutral typed regions from enriched module scans.

This module is the semantic boundary between source analysis and compiler
lowering. It groups connected declarations while retaining exact source,
annotation, binding, and dependency evidence. It deliberately does not decide
which backend should compile a region or rewrite unsupported types.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from textwrap import dedent

from atoll.models import (
    BindingTarget,
    DependencyEdge,
    LoweringDecision,
    ModuleScan,
    ParameterRecord,
    RegionDependency,
    RegionMember,
    RegionSpecialization,
    RuntimeTypeGuard,
    SpecializationOrigin,
    SymbolId,
    SymbolRecord,
    TypeBinding,
    TypedRegion,
    TypeParameterRecord,
)

TYPED_REGION_SCHEMA_VERSION = "atoll-typed-region-v3"
_UNSAFE_SOFT_BLOCKERS = frozenset({"UNTYPED_DECORATOR"})
_DEFAULT_ANY_PATHS = frozenset({"Any", "typing.Any", "typing_extensions.Any"})
_GETATTR_REQUIRED_ARGS = 2


@dataclass(frozen=True, slots=True)
class _RegionBuildContext:
    """Immutable module evidence shared while materializing components.

    Attributes:
        module: Module identity or syntax module associated with the state.
        eligible: Members accepted for this region or backend.
        edges: Dependency edges available to region construction.
        source_lines: Original source lines used for precise extraction.
        tree: Parsed syntax tree represented by the state.
        downgraded_classes: Classes excluded from stronger atomic lowering.
        forbidden_type_paths: Runtime types that cannot satisfy specialization guards.
    """

    module: ModuleScan
    eligible: dict[SymbolId, SymbolRecord]
    edges: tuple[DependencyEdge, ...]
    source_lines: list[str]
    tree: ast.Module
    downgraded_classes: dict[str, LoweringDecision]
    forbidden_type_paths: frozenset[str]


@dataclass(frozen=True, slots=True)
class _SpecializationEvidence:
    """Normalized content used to create and hash specialization records.

    Attributes:
        symbol: Stable source symbol identity.
        origin: Evidence source for a specialization or generated name.
        target_owner_class: Concrete runtime class receiving a specialized binding.
        substitutions: Concrete generic type substitutions.
        guards: Runtime checks required before selecting the specialization.
        bindings: Runtime source bindings promised by the compiled variant.
    """

    symbol: SymbolRecord
    origin: SpecializationOrigin
    target_owner_class: str | None
    substitutions: tuple[tuple[str, str], ...]
    guards: tuple[RuntimeTypeGuard, ...]
    bindings: tuple[TypeBinding, ...]


@dataclass(frozen=True, slots=True)
class _MemberSpecializationInputs:
    """All proof inputs required to specialize one source member.

    Attributes:
        symbol: Stable source symbol identity.
        scope_type_parameter_records: Structured type parameters inherited from enclosing scopes.
        substitutions: Concrete generic type substitutions.
        origin: Evidence source for a specialization or generated name.
        target_owner_class: Concrete runtime class receiving a specialized binding.
        forbidden_type_paths: Runtime types that cannot satisfy specialization guards.
    """

    symbol: SymbolRecord
    scope_type_parameter_records: tuple[TypeParameterRecord, ...]
    substitutions: tuple[tuple[str, str], ...]
    origin: SpecializationOrigin
    target_owner_class: str | None
    forbidden_type_paths: frozenset[str]


@dataclass(frozen=True, slots=True)
class _SubclassSpecializationInputs:
    """Shared lookup tables for direct concrete subclass analysis.

    Attributes:
        context: Prepared state shared by the operation.
        scopes: Nested type-parameter scopes visible at this location.
        by_class: Declarations grouped by owning class name.
        methods_by_owner: Method declarations grouped by owner class.
        symbol_ids: Stable IDs of symbols included in the state.
    """

    context: _RegionBuildContext
    scopes: dict[SymbolId, tuple[TypeParameterRecord, ...]]
    by_class: dict[str, SymbolRecord]
    methods_by_owner: dict[str, tuple[SymbolRecord, ...]]
    symbol_ids: set[SymbolId]


@dataclass(frozen=True, slots=True)
class _ConcreteBaseTarget:
    """One resolved generic ancestor plus method names shadowed below it.

    Attributes:
        base: Base-class expression being specialized.
        substitutions: Concrete generic type substitutions.
        shadowed_method_names: Base methods replaced by a subclass declaration.
    """

    base: SymbolRecord
    substitutions: tuple[tuple[str, str], ...]
    shadowed_method_names: frozenset[str]


def build_typed_regions(
    module: ModuleScan,
    edges: tuple[DependencyEdge, ...],
) -> tuple[TypedRegion, ...]:
    """Return deterministic connected regions without lowering their types.

    Hard-blocked declarations and declarations changed by unknown decorators
    are excluded. A class is included atomically only when every directly
    declared method is eligible; otherwise its eligible methods remain
    independent callable members on the original source class.

    Args:
        module: Module scan or module identity being analyzed.
        edges: Dependency edges available when constructing typed regions.

    Returns:
        tuple[TypedRegion, ...]: Deterministically ordered backend-neutral typed regions.
    """
    source = module.module.path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(module.module.path), type_comments=True)
    eligible = {symbol.id: symbol for symbol in module.symbols if _eligible(symbol)}
    if not eligible:
        return ()
    adjacency = _adjacency(eligible, edges)
    downgraded_classes = _connect_atomic_classes(
        module.symbols,
        eligible,
        adjacency,
        tree,
    )
    components = _components(eligible, adjacency)
    source_lines = source.splitlines()
    context = _RegionBuildContext(
        module=module,
        eligible=eligible,
        edges=edges,
        source_lines=source_lines,
        tree=tree,
        downgraded_classes=downgraded_classes,
        forbidden_type_paths=_semantic_any_paths(module),
    )
    return tuple(_typed_region(context, component) for component in components)


def build_directed_region_slice(region: TypedRegion, root: SymbolId) -> TypedRegion:
    """Build one deterministic hot-binding slice from a connected source region.

    Only dependencies explicitly marked `requires_same_unit` expand the member
    set. Other local calls remain dependency evidence and are lowered as normal
    runtime dispatch boundaries. Profile counts and runtime type observations
    are deliberately absent from the resulting ID and source hash.

    Args:
        region: Connected backend-neutral source region containing the root.
        root: Public callable binding that owns this directed slice.

    Returns:
        TypedRegion: Non-atomic source slice rooted at exactly one public binding.

    Raises:
        ValueError: The root is absent or a required same-unit target is absent.
    """
    member_by_id = {member.id: member for member in region.members}
    if root not in member_by_id:
        raise ValueError(f"directed slice root is outside region {region.id}: {root.stable_id}")
    selected = {root}
    changed = True
    while changed:
        changed = False
        for dependency in region.dependencies:
            if (
                dependency.src not in selected
                or not dependency.requires_same_unit
                or not isinstance(dependency.dst, SymbolId)
            ):
                continue
            if dependency.dst not in member_by_id:
                raise ValueError(
                    "required same-unit dependency is outside region "
                    f"{region.id}: {dependency.dst.stable_id}"
                )
            if dependency.dst not in selected:
                selected.add(dependency.dst)
                changed = True
    members = tuple(member for member in region.members if member.id in selected)
    dependencies = tuple(
        dependency for dependency in region.dependencies if dependency.src in selected
    )
    member_prefixes = tuple(f"{member.id.qualname}." for member in members)
    type_bindings = tuple(
        binding for binding in region.type_bindings if binding.name.startswith(member_prefixes)
    )
    member_decision_targets = {member.id.stable_id for member in members}
    member_decision_targets.update(
        f"{region.source_module.name}::{member.owner_class}"
        for member in members
        if member.owner_class is not None
    )
    decisions = tuple(
        decision for decision in region.decisions if decision.target in member_decision_targets
    )
    specializations = tuple(
        specialization
        for specialization in region.specializations
        if specialization.source_member in selected
    )
    source_hash = _region_hash(
        region.source_module.name,
        members,
        dependencies,
        type_bindings,
        specializations,
    )
    label = root.qualname.replace(".", "_")
    bindings = tuple(binding for binding in region.bindings if binding.source == root)
    root_member = member_by_id[root]
    if not bindings and root_member.kind == "method":
        method_name = root.qualname.rsplit(".", maxsplit=1)[-1]
        if method_name.startswith("__"):
            raise ValueError(f"directed slice root is not independently bindable: {root.stable_id}")
        bindings = (
            BindingTarget(
                source=root,
                compiled_name=root.qualname.replace(".", "__"),
                kind=root_member.binding_kind,
                owner_class=root_member.owner_class,
                execution_kind=root_member.execution_kind,
            ),
        )
    if len(bindings) != 1:
        raise ValueError(f"directed slice root must have exactly one binding: {root.stable_id}")
    return TypedRegion(
        id=f"{region.source_module.name}::{label}_slice:{source_hash[:12]}",
        source_module=region.source_module,
        members=members,
        dependencies=dependencies,
        type_bindings=type_bindings,
        bindings=bindings,
        decisions=decisions,
        source_hash=source_hash,
        atomic_class=False,
        specializations=specializations,
    )


def _eligible(symbol: SymbolRecord) -> bool:
    if any(blocker.severity == "hard" for blocker in symbol.blockers):
        return False
    if any(blocker.code in _UNSAFE_SOFT_BLOCKERS for blocker in symbol.blockers):
        return False
    if symbol.kind == "class":
        return True
    return True


def _signature_is_complete(symbol: SymbolRecord) -> bool:
    parameters = symbol.parameters
    if symbol.binding_kind in {"instance_method", "classmethod"} and parameters:
        parameters = parameters[1:]
    return symbol.return_annotation is not None and all(
        parameter.annotation is not None for parameter in parameters
    )


def _adjacency(
    eligible: dict[SymbolId, SymbolRecord],
    edges: tuple[DependencyEdge, ...],
) -> dict[SymbolId, set[SymbolId]]:
    adjacency = {symbol_id: set[SymbolId]() for symbol_id in eligible}
    for edge in edges:
        if (
            edge.confidence != "high"
            or edge.src not in eligible
            or not isinstance(edge.dst, SymbolId)
            or edge.dst not in eligible
        ):
            continue
        adjacency[edge.src].add(edge.dst)
        adjacency[edge.dst].add(edge.src)
    return adjacency


def _connect_atomic_classes(
    symbols: tuple[SymbolRecord, ...],
    eligible: dict[SymbolId, SymbolRecord],
    adjacency: dict[SymbolId, set[SymbolId]],
    tree: ast.Module,
) -> dict[str, LoweringDecision]:
    methods_by_owner: dict[str, list[SymbolRecord]] = defaultdict(list)
    classes = {symbol.id.qualname: symbol for symbol in symbols if symbol.kind == "class"}
    downgraded: dict[str, LoweringDecision] = {}
    for symbol in symbols:
        if symbol.kind == "method" and symbol.owner_class is not None:
            methods_by_owner[symbol.owner_class].append(symbol)
    for owner, methods in methods_by_owner.items():
        class_symbol = classes.get(owner)
        if class_symbol is None:
            continue
        unsafe_reason = _atomic_class_unsafe_reason(class_symbol, tree)
        if unsafe_reason is not None:
            eligible.pop(class_symbol.id, None)
            adjacency.pop(class_symbol.id, None)
            _connect_interpreted_owner_methods(methods, eligible, adjacency)
            downgraded[owner] = LoweringDecision(
                target=class_symbol.id.stable_id,
                action="fallback",
                reason=unsafe_reason,
            )
            continue
        if class_symbol.id not in eligible:
            _connect_interpreted_owner_methods(methods, eligible, adjacency)
            downgraded[owner] = LoweringDecision(
                target=class_symbol.id.stable_id,
                action="fallback",
                reason="class remains interpreted because its dynamic behavior is blocked",
            )
            continue
        if any(
            method.id not in eligible or _method_requires_boxed_lowering(method)
            for method in methods
        ):
            eligible.pop(class_symbol.id, None)
            adjacency.pop(class_symbol.id, None)
            _connect_interpreted_owner_methods(methods, eligible, adjacency)
            downgraded[owner] = LoweringDecision(
                target=class_symbol.id.stable_id,
                action="fallback",
                reason="class remains interpreted because a required method is blocked",
            )
            continue
        for method in methods:
            adjacency[class_symbol.id].add(method.id)
            adjacency[method.id].add(class_symbol.id)
    return downgraded


def _method_requires_boxed_lowering(method: SymbolRecord) -> bool:
    return not _signature_is_complete(method) or method.has_any_annotation


def _connect_interpreted_owner_methods(
    methods: list[SymbolRecord],
    eligible: dict[SymbolId, SymbolRecord],
    adjacency: dict[SymbolId, set[SymbolId]],
) -> None:
    """Keep one owner-level method region after whole-class replacement is rejected.

    Args:
        methods: Method declarations considered for class-region safety.
        eligible: Members accepted by backend and specialization checks.
        adjacency: Dependency adjacency keyed by stable symbol identity.
    """
    method_ids = tuple(method.id for method in methods if method.id in eligible)
    if not method_ids:
        return
    anchor = method_ids[0]
    for method_id in method_ids[1:]:
        adjacency[anchor].add(method_id)
        adjacency[method_id].add(anchor)


def _atomic_class_unsafe_reason(class_symbol: SymbolRecord, tree: ast.Module) -> str | None:
    """Explain why replacing a source class after module execution is unsafe.

    Class replacement is allowed only when no object created while the module is
    loading can retain the original class identity. Class decorators may register
    or replace the class, duplicate module bindings obscure the public identity,
    and any later module-level reference may create an instance, subclass, default,
    annotation, or registry entry before the staged shim runs.

    Args:
        class_symbol: Scanned class declaration that owns the candidate methods.
        tree: Parsed syntax tree being traversed or rewritten.

    Returns:
        str | None: Concrete rejection reason, or `None` when atomic lowering is safe.
    """
    declaration_issue = _atomic_class_declaration_issue(class_symbol, tree)
    if declaration_issue is not None:
        return f"class remains interpreted because {declaration_issue}"
    identity_issue = _atomic_class_identity_issue(class_symbol, tree)
    return f"class remains interpreted because {identity_issue}" if identity_issue else None


def _atomic_class_declaration_issue(
    class_symbol: SymbolRecord,
    tree: ast.Module,
) -> str | None:
    if class_symbol.decorators and all(
        _is_dataclass_decorator(decorator) for decorator in class_symbol.decorators
    ):
        return "it is a dataclass whose declared methods may be rebound"
    if class_symbol.decorators:
        return "decorators may register or replace it"
    class_node = next(
        (
            statement
            for statement in tree.body
            if isinstance(statement, ast.ClassDef)
            and statement.name == class_symbol.id.qualname
            and statement.lineno == class_symbol.lineno
        ),
        None,
    )
    if class_node is None:
        return "its declaration cannot be resolved"
    return _atomic_class_definition_issue(class_node)


def _is_dataclass_decorator(decorator: str) -> bool:
    try:
        expression = ast.parse(decorator, mode="eval").body
    except SyntaxError:
        return False
    target = expression.func if isinstance(expression, ast.Call) else expression
    if isinstance(target, ast.Name):
        return target.id == "dataclass"
    return isinstance(target, ast.Attribute) and target.attr == "dataclass"


def _atomic_class_identity_issue(
    class_symbol: SymbolRecord,
    tree: ast.Module,
) -> str | None:
    class_name = class_symbol.id.qualname
    if class_name in _ambiguous_module_bindings(tree):
        return "its module binding is reassigned"
    if _has_later_module_reference(tree, class_symbol):
        return "module-time code retains its original identity"
    if _has_later_import(tree, class_symbol):
        return "a later import can expose its source identity"
    if _has_later_call(tree, class_symbol):
        return "a later call can expose its source identity"
    return None


def _atomic_class_definition_issue(node: ast.ClassDef) -> str | None:
    """Return a reason when evaluating a copied class could change behavior.

    Args:
        node: Syntax node being visited without executing target code.

    Returns:
        str | None: Unsupported class-definition reason, or `None` when safe.
    """
    if node.bases or node.keywords:
        return "source inheritance is outside the closed atomic-class subset"
    for statement in node.body:
        issue = _atomic_class_statement_issue(statement)
        if issue is not None:
            return issue
    return None


def _atomic_class_statement_issue(statement: ast.stmt) -> str | None:
    if isinstance(statement, ast.Pass | ast.AsyncFunctionDef):
        return None
    if (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    ):
        return None
    if (
        isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
        and statement.value is None
    ):
        return None
    if isinstance(statement, ast.FunctionDef):
        return _atomic_method_definition_issue(statement)
    return "its class body contains executable statements"


def _atomic_method_definition_issue(statement: ast.FunctionDef) -> str | None:
    if (
        statement.name.startswith("__")
        and statement.name.endswith("__")
        and statement.name != "__init__"
    ):
        return f"special method {statement.name} requires interpreted class semantics"
    if any(
        _atomic_decorator_name(decorator) not in {"staticmethod", "classmethod"}
        for decorator in statement.decorator_list
    ):
        return f"method {statement.name} has a runtime decorator"
    defaults = (*statement.args.defaults, *statement.args.kw_defaults)
    if any(default is not None and not _atomic_default_is_literal(default) for default in defaults):
        return f"method {statement.name} has a nonliteral default"
    return None


def _atomic_decorator_name(decorator: ast.expr) -> str | None:
    if isinstance(decorator, ast.Name):
        return decorator.id
    return None


def _atomic_default_is_literal(expression: ast.expr) -> bool:
    if isinstance(expression, ast.Constant):
        return True
    return (
        isinstance(expression, ast.UnaryOp)
        and isinstance(expression.op, ast.UAdd | ast.USub)
        and isinstance(expression.operand, ast.Constant)
        and isinstance(expression.operand.value, int | float | complex)
    )


def _has_later_import(tree: ast.Module, class_symbol: SymbolRecord) -> bool:
    class_statement_seen = False
    for statement in tree.body:
        if (
            isinstance(statement, ast.ClassDef)
            and statement.name == class_symbol.id.qualname
            and statement.lineno == class_symbol.lineno
        ):
            class_statement_seen = True
            continue
        if class_statement_seen:
            visitor = _ModuleTimeImportVisitor()
            visitor.visit(statement)
            if visitor.found:
                return True
    return False


class _ModuleTimeImportVisitor(ast.NodeVisitor):
    """Find imports executed after a class while skipping deferred function bodies."""

    def __init__(self) -> None:
        self.found = False

    def visit_Import(self, node: ast.Import) -> None:
        _ = node
        self.found = True

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        _ = node
        self.found = True

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        _ = node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        _ = node

    def visit_Lambda(self, node: ast.Lambda) -> None:
        _ = node


def _has_later_call(tree: ast.Module, class_symbol: SymbolRecord) -> bool:
    class_statement_seen = False
    for statement in tree.body:
        if (
            isinstance(statement, ast.ClassDef)
            and statement.name == class_symbol.id.qualname
            and statement.lineno == class_symbol.lineno
        ):
            class_statement_seen = True
            continue
        if class_statement_seen:
            visitor = _ModuleTimeCallVisitor()
            visitor.visit(statement)
            if visitor.found:
                return True
    return False


class _ModuleTimeCallVisitor(ast.NodeVisitor):
    """Find arbitrary calls that can re-enter a partially initialized module."""

    def __init__(self) -> None:
        self.found = False

    def visit_Call(self, node: ast.Call) -> None:
        _ = node
        self.found = True

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function_header(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function_header(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._visit_arguments(node.args)

    def _visit_function_header(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        self._visit_arguments(node.args)
        if node.returns is not None:
            self.visit(node.returns)

    def _visit_arguments(self, arguments: ast.arguments) -> None:
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
            *((arguments.vararg,) if arguments.vararg is not None else ()),
            *((arguments.kwarg,) if arguments.kwarg is not None else ()),
        ):
            if argument.annotation is not None:
                self.visit(argument.annotation)
        for default in (*arguments.defaults, *arguments.kw_defaults):
            if default is not None:
                self.visit(default)


def _has_later_module_reference(tree: ast.Module, class_symbol: SymbolRecord) -> bool:
    class_statement_seen = False
    for statement in tree.body:
        if (
            isinstance(statement, ast.ClassDef)
            and statement.name == class_symbol.id.qualname
            and statement.lineno == class_symbol.lineno
        ):
            class_statement_seen = True
            continue
        if not class_statement_seen:
            continue
        visitor = _ModuleTimeClassReferenceVisitor(class_symbol.id.qualname)
        visitor.visit(statement)
        if visitor.found:
            return True
    return False


class _ModuleTimeClassReferenceVisitor(ast.NodeVisitor):
    """Find class-name use in expressions evaluated while a module is loading."""

    def __init__(self, class_name: str) -> None:
        self.class_name = class_name
        self.found = False

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == self.class_name:
            self.found = True

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function_header(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function_header(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._visit_arguments(node.args)

    def _visit_function_header(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        self._visit_arguments(node.args)
        if node.returns is not None:
            self.visit(node.returns)
        for type_parameter in getattr(node, "type_params", ()):
            self.visit(type_parameter)

    def _visit_arguments(self, arguments: ast.arguments) -> None:
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
            *((arguments.vararg,) if arguments.vararg is not None else ()),
            *((arguments.kwarg,) if arguments.kwarg is not None else ()),
        ):
            if argument.annotation is not None:
                self.visit(argument.annotation)
        for default in (*arguments.defaults, *arguments.kw_defaults):
            if default is not None:
                self.visit(default)


def _components(
    eligible: dict[SymbolId, SymbolRecord],
    adjacency: dict[SymbolId, set[SymbolId]],
) -> tuple[tuple[SymbolId, ...], ...]:
    unseen = set(eligible)
    components: list[tuple[SymbolId, ...]] = []
    while unseen:
        seed = min(unseen, key=lambda symbol: symbol.stable_id)
        pending = [seed]
        component: set[SymbolId] = set()
        while pending:
            current = pending.pop()
            if current in component or current not in eligible:
                continue
            component.add(current)
            pending.extend(adjacency.get(current, ()))
        unseen.difference_update(component)
        components.append(tuple(sorted(component, key=lambda symbol: symbol.stable_id)))
    return tuple(components)


def _typed_region(
    context: _RegionBuildContext,
    component: tuple[SymbolId, ...],
) -> TypedRegion:
    symbols = tuple(context.eligible[symbol_id] for symbol_id in component)
    classes = {
        symbol.id.qualname: symbol for symbol in context.module.symbols if symbol.kind == "class"
    }
    scopes = {symbol.id: _scope_type_parameter_records(symbol, classes) for symbol in symbols}
    members = tuple(
        _region_member(symbol, context.source_lines, scopes[symbol.id]) for symbol in symbols
    )
    dependencies = _region_dependencies(component, context.edges)
    type_bindings = tuple(
        binding
        for symbol in symbols
        for binding in _symbol_type_bindings(symbol, scopes[symbol.id])
    )
    specializations = _region_specializations(context, symbols, scopes)
    atomic_class = any(symbol.kind == "class" for symbol in symbols)
    bindings = _binding_targets(symbols, atomic_class=atomic_class)
    owner_names = {symbol.owner_class for symbol in symbols if symbol.owner_class is not None}
    decisions = (
        *(_lowering_decision(symbol, scopes[symbol.id]) for symbol in symbols),
        *(
            context.downgraded_classes[owner]
            for owner in sorted(owner_names & context.downgraded_classes.keys())
        ),
    )
    source_hash = _region_hash(
        context.module.module.name,
        members,
        dependencies,
        type_bindings,
        specializations,
    )
    label = component[0].qualname.replace(".", "_")
    return TypedRegion(
        id=f"{context.module.module.name}::{label}:{source_hash[:12]}",
        source_module=context.module.module,
        members=members,
        dependencies=dependencies,
        type_bindings=type_bindings,
        bindings=bindings,
        decisions=decisions,
        source_hash=source_hash,
        atomic_class=atomic_class,
        specializations=specializations,
    )


def _region_member(
    symbol: SymbolRecord,
    source_lines: list[str],
    scope_type_parameter_records: tuple[TypeParameterRecord, ...],
) -> RegionMember:
    start_lineno = symbol.declaration_start_lineno or symbol.lineno
    source_text = "\n".join(source_lines[start_lineno - 1 : symbol.end_lineno])
    return RegionMember(
        id=symbol.id,
        kind=symbol.kind,
        owner_class=symbol.owner_class,
        binding_kind=symbol.binding_kind,
        execution_kind=symbol.execution_kind,
        source_text=source_text,
        type_parameters=symbol.type_parameters,
        type_parameter_records=symbol.type_parameter_records,
        scope_type_parameters=tuple(record.name for record in scope_type_parameter_records),
        scope_type_parameter_records=scope_type_parameter_records,
        parameters=symbol.parameters,
        return_annotation=symbol.return_annotation,
        call_sites=symbol.call_sites,
        suspension_points=symbol.suspension_points,
        runtime_imports=symbol.runtime_imports,
        fields=symbol.fields,
    )


def _region_dependencies(
    component: tuple[SymbolId, ...],
    edges: tuple[DependencyEdge, ...],
) -> tuple[RegionDependency, ...]:
    member_ids = set(component)
    return tuple(
        RegionDependency(
            src=edge.src,
            dst=edge.dst,
            kind=edge.kind,
            confidence=edge.confidence,
            role="typing" if edge.kind == "annotation" else "runtime",
            type_only=edge.kind == "annotation",
            lineno=edge.lineno,
            invocation_mode=edge.invocation_mode,
            requires_same_unit=edge.requires_same_unit,
        )
        for edge in edges
        if edge.src in member_ids
    )


def _scope_type_parameter_records(
    symbol: SymbolRecord,
    classes: dict[str, SymbolRecord],
) -> tuple[TypeParameterRecord, ...]:
    owner_parameters: tuple[TypeParameterRecord, ...] = ()
    if symbol.owner_class is not None and symbol.owner_class in classes:
        owner_parameters = classes[symbol.owner_class].scope_type_parameter_records
    records = (*owner_parameters, *symbol.scope_type_parameter_records)
    unique: dict[str, TypeParameterRecord] = {}
    for record in records:
        unique.setdefault(record.name, record)
    return tuple(unique.values())


def _symbol_type_bindings(
    symbol: SymbolRecord,
    scope_type_parameter_records: tuple[TypeParameterRecord, ...],
) -> tuple[TypeBinding, ...]:
    scope_type_parameters = tuple(record.name for record in scope_type_parameter_records)
    bindings = [
        _parameter_type_binding(
            symbol,
            parameter,
            scope_type_parameters,
        )
        for parameter in symbol.parameters
        if parameter.annotation is not None
    ]
    if symbol.return_annotation is not None:
        bindings.append(
            TypeBinding(
                name=f"{symbol.id.qualname}.return",
                annotation=symbol.return_annotation,
                source="return",
                concrete=_is_concrete(
                    symbol.return_annotation,
                    scope_type_parameters,
                    symbol.any_annotation_sources,
                ),
            )
        )
    bindings.extend(
        TypeBinding(
            name=f"{symbol.id.qualname}.{type_parameter}",
            annotation=record.declaration,
            source="type_parameter",
            concrete=False,
        )
        for record in scope_type_parameter_records
        for type_parameter in (record.name,)
    )
    bindings.extend(
        TypeBinding(
            name=f"{symbol.id.qualname}.base",
            annotation=base,
            source="base",
            concrete=_is_concrete(base, scope_type_parameters, symbol.any_annotation_sources),
        )
        for base in symbol.base_names
    )
    bindings.extend(
        TypeBinding(
            name=f"{symbol.id.qualname}.{field.name}",
            annotation=field.annotation,
            source="field",
            concrete=_is_concrete(
                field.annotation,
                scope_type_parameters,
                symbol.any_annotation_sources,
            ),
        )
        for field in symbol.fields
    )
    return tuple(bindings)


def _parameter_type_binding(
    symbol: SymbolRecord,
    parameter: ParameterRecord,
    scope_type_parameters: tuple[str, ...],
) -> TypeBinding:
    annotation = parameter.annotation
    if annotation is None:
        raise ValueError("parameter type binding requires an annotation")
    return TypeBinding(
        name=f"{symbol.id.qualname}.{parameter.name}",
        annotation=annotation,
        source="parameter",
        concrete=_is_concrete(
            annotation,
            scope_type_parameters,
            symbol.any_annotation_sources,
        ),
    )


def _is_concrete(
    annotation: str,
    type_parameters: tuple[str, ...],
    any_annotation_sources: tuple[str, ...],
) -> bool:
    if annotation in any_annotation_sources:
        return False
    try:
        expression = ast.parse(annotation, mode="eval").body
    except SyntaxError:
        return False
    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        return _is_concrete(expression.value, type_parameters, any_annotation_sources)
    return not any(
        isinstance(node, ast.Name) and node.id in type_parameters for node in ast.walk(expression)
    )


def _binding_targets(
    symbols: tuple[SymbolRecord, ...],
    *,
    atomic_class: bool,
) -> tuple[BindingTarget, ...]:
    targets: list[BindingTarget] = []
    for symbol in symbols:
        if atomic_class and symbol.kind == "method":
            continue
        if symbol.kind == "method" and symbol.id.qualname.rsplit(".", maxsplit=1)[-1].startswith(
            "__"
        ):
            continue
        targets.append(
            BindingTarget(
                source=symbol.id,
                compiled_name=symbol.id.qualname.replace(".", "__"),
                kind=symbol.binding_kind,
                owner_class=symbol.owner_class,
                execution_kind=symbol.execution_kind,
            )
        )
    return tuple(targets)


def _lowering_decision(
    symbol: SymbolRecord,
    scope_type_parameter_records: tuple[TypeParameterRecord, ...],
) -> LoweringDecision:
    if symbol.has_any_annotation:
        return LoweringDecision(
            target=symbol.id.stable_id,
            action="box",
            reason="source annotation explicitly contains Any",
        )
    if symbol.kind != "class" and not _signature_is_complete(symbol):
        return LoweringDecision(
            target=symbol.id.stable_id,
            action="box",
            reason="source callable has incomplete annotations",
        )
    if scope_type_parameter_records:
        return LoweringDecision(
            target=symbol.id.stable_id,
            action="fallback",
            reason="generic declaration requires a concrete specialization",
        )
    return LoweringDecision(
        target=symbol.id.stable_id,
        action="preserve",
        reason="source typing and declaration remain unchanged",
    )


def _region_specializations(
    context: _RegionBuildContext,
    symbols: tuple[SymbolRecord, ...],
    scopes: dict[SymbolId, tuple[TypeParameterRecord, ...]],
) -> tuple[RegionSpecialization, ...]:
    """Find concrete same-module specializations without mutating source facts.

    Args:
        context: Prepared state shared by this operation.
        symbols: Source symbols processed in deterministic order.
        scopes: Nested type-parameter scopes visible at this location.

    Returns:
        tuple[RegionSpecialization, ...]: Deterministically ordered guarded specializations for
            the region.
    """
    symbol_ids = {symbol.id for symbol in symbols}
    by_class = {
        symbol.id.qualname: symbol for symbol in context.module.symbols if symbol.kind == "class"
    }
    methods_by_owner: dict[str, tuple[SymbolRecord, ...]] = {}
    for owner in {symbol.owner_class for symbol in context.module.symbols if symbol.owner_class}:
        methods_by_owner[owner] = tuple(
            symbol
            for symbol in context.module.symbols
            if symbol.kind == "method" and symbol.owner_class == owner
        )
    specs = [
        spec
        for spec in _subclass_specializations(
            _SubclassSpecializationInputs(
                context=context,
                scopes=scopes,
                by_class=by_class,
                methods_by_owner=methods_by_owner,
                symbol_ids=symbol_ids,
            )
        )
        if spec is not None
    ]
    specs.extend(
        spec for spec in _closed_call_specializations(context, symbols, scopes) if spec is not None
    )
    return tuple(sorted(specs, key=lambda spec: spec.id))


def _subclass_specializations(
    inputs: _SubclassSpecializationInputs,
) -> tuple[RegionSpecialization | None, ...]:
    specializations: list[RegionSpecialization | None] = []
    for subclass in inputs.by_class.values():
        if (
            not _class_supports_member_binding(subclass, inputs.context)
            or subclass.has_any_annotation
            or subclass.type_parameters
            or subclass.scope_type_parameter_records
        ):
            continue
        claimed_method_names = {
            method.id.qualname.rsplit(".", maxsplit=1)[-1]
            for method in inputs.methods_by_owner.get(subclass.id.qualname, ())
        }
        for base_name in subclass.base_names:
            base_method_names = _base_method_names(
                base_name,
                inputs.by_class,
                inputs.methods_by_owner,
            )
            base_targets = _concrete_base_targets(
                base_name,
                inputs.by_class,
                inputs.methods_by_owner,
                inputs.context.forbidden_type_paths,
            )
            for base_target in base_targets:
                base = base_target.base
                substitutions = base_target.substitutions
                if not _generic_base_is_safe(base, inputs.context) or any(
                    method.id not in inputs.context.eligible
                    for method in inputs.methods_by_owner.get(base.id.qualname, ())
                ):
                    continue
                for method in inputs.methods_by_owner.get(base.id.qualname, ()):
                    method_name = method.id.qualname.rsplit(".", maxsplit=1)[-1]
                    if (
                        method.id not in inputs.symbol_ids
                        or method.id not in inputs.context.eligible
                    ):
                        continue
                    if (
                        method_name.startswith("__")
                        or method_name in claimed_method_names
                        or method_name in base_target.shadowed_method_names
                    ):
                        continue
                    if _method_uses_generic_field(
                        method,
                        base,
                        substitutions,
                        inputs.context.source_lines,
                    ):
                        continue
                    specializations.append(
                        _member_specialization(
                            _MemberSpecializationInputs(
                                symbol=method,
                                scope_type_parameter_records=inputs.scopes[method.id],
                                substitutions=substitutions,
                                origin="concrete_subclass",
                                target_owner_class=subclass.id.qualname,
                                forbidden_type_paths=inputs.context.forbidden_type_paths,
                            )
                        )
                    )
            claimed_method_names.update(base_method_names)
    return tuple(specializations)


def _base_method_names(
    base_name: str,
    by_class: dict[str, SymbolRecord],
    methods_by_owner: dict[str, tuple[SymbolRecord, ...]],
    seen: frozenset[str] = frozenset(),
) -> frozenset[str]:
    expression = _annotation_expression(base_name)
    if expression is None:
        return frozenset()
    target = expression.value if isinstance(expression, ast.Subscript) else expression
    path = _annotation_path(target)
    base = by_class.get(path[0]) if path is not None and len(path) == 1 else None
    if base is None or base.id.qualname in seen:
        return frozenset()
    names = {
        method.id.qualname.rsplit(".", maxsplit=1)[-1]
        for method in methods_by_owner.get(base.id.qualname, ())
    }
    for parent in base.base_names:
        names.update(
            _base_method_names(
                parent,
                by_class,
                methods_by_owner,
                seen | {base.id.qualname},
            )
        )
    return frozenset(names)


def _concrete_base_targets(
    base_name: str,
    by_class: dict[str, SymbolRecord],
    methods_by_owner: dict[str, tuple[SymbolRecord, ...]],
    forbidden_type_paths: frozenset[str],
    seen: frozenset[str] = frozenset(),
) -> tuple[_ConcreteBaseTarget, ...]:
    base, substitutions = _concrete_base_substitution(
        base_name,
        by_class,
        forbidden_type_paths,
    )
    if base is None or substitutions is None or base.id.qualname in seen:
        return ()
    targets = [
        _ConcreteBaseTarget(
            base=base,
            substitutions=substitutions,
            shadowed_method_names=frozenset(),
        )
    ]
    direct_method_names = frozenset(
        method.id.qualname.rsplit(".", maxsplit=1)[-1]
        for method in methods_by_owner.get(base.id.qualname, ())
    )
    for parent_name in base.base_names:
        specialized_parent = _substitute_annotation(parent_name, substitutions)
        targets.extend(
            _ConcreteBaseTarget(
                base=parent.base,
                substitutions=parent.substitutions,
                shadowed_method_names=parent.shadowed_method_names | direct_method_names,
            )
            for parent in _concrete_base_targets(
                specialized_parent,
                by_class,
                methods_by_owner,
                forbidden_type_paths,
                seen | {base.id.qualname},
            )
        )
    return tuple(targets)


def _generic_base_is_safe(base: SymbolRecord, context: _RegionBuildContext) -> bool:
    """Allow a legacy generic base blocked only by its own TypeVar assignment.

    Args:
        base: Base-class expression being resolved.
        context: Prepared state shared by this operation.

    Returns:
        bool: Whether the generic base can be retained without unsafe erasure.
    """
    if _class_supports_member_binding(base, context):
        return True
    hard_blockers = tuple(blocker for blocker in base.blockers if blocker.severity == "hard")
    type_parameters = set(base.scope_type_parameters)
    return (
        not any(blocker.code in _UNSAFE_SOFT_BLOCKERS for blocker in base.blockers)
        and bool(type_parameters)
        and bool(hard_blockers)
        and all(
            blocker.code == "DYNAMIC_GLOBAL_DEP"
            and any(f"global '{name}'" in blocker.message for name in type_parameters)
            for blocker in hard_blockers
        )
    )


def _concrete_base_substitution(
    base_name: str,
    by_class: dict[str, SymbolRecord],
    forbidden_type_paths: frozenset[str],
) -> tuple[SymbolRecord | None, tuple[tuple[str, str], ...] | None]:
    base: SymbolRecord | None = None
    substitutions: tuple[tuple[str, str], ...] | None = None
    try:
        expression = ast.parse(base_name, mode="eval").body
    except SyntaxError:
        expression = None
    if isinstance(expression, ast.Subscript):
        base_path = _annotation_path(expression.value)
        candidate = (
            by_class.get(base_path[0]) if base_path is not None and len(base_path) == 1 else None
        )
        if candidate is not None:
            substitutions = _base_substitutions(
                candidate,
                expression.slice,
                forbidden_type_paths,
            )
            base = candidate if substitutions is not None else None
    return base, substitutions


def _base_substitutions(
    base: SymbolRecord,
    slice_node: ast.expr,
    forbidden_type_paths: frozenset[str],
) -> tuple[tuple[str, str], ...] | None:
    type_parameters = base.type_parameter_records or base.scope_type_parameter_records
    if not type_parameters or any(record.kind != "type_var" for record in type_parameters):
        return None
    arguments = _subscript_arguments(slice_node)
    if len(arguments) != len(type_parameters):
        return None
    substitutions: list[tuple[str, str]] = []
    for record, argument in zip(type_parameters, arguments, strict=True):
        annotation = ast.unparse(argument)
        if not _specialized_annotation_is_guardable(annotation, forbidden_type_paths):
            return None
        substitutions.append((record.name, annotation))
    return tuple(substitutions)


def _closed_call_specializations(
    context: _RegionBuildContext,
    symbols: tuple[SymbolRecord, ...],
    scopes: dict[SymbolId, tuple[TypeParameterRecord, ...]],
) -> tuple[RegionSpecialization | None, ...]:
    call_sites = _direct_call_sites(context.tree, context.forbidden_type_paths)
    ambiguous_module_bindings = _ambiguous_module_bindings(context.tree)
    specializations: list[RegionSpecialization | None] = []
    for symbol in symbols:
        if symbol.kind not in {"function", "method"}:
            continue
        if symbol.kind == "method" and not _owner_class_is_eligible(symbol, context):
            continue
        module_binding = symbol.id.qualname if symbol.kind == "function" else symbol.owner_class
        if module_binding in ambiguous_module_bindings:
            continue
        type_parameters = scopes[symbol.id]
        if not type_parameters or any(record.kind != "type_var" for record in type_parameters):
            continue
        calls = call_sites.get(symbol.id.qualname, ())
        if not calls:
            continue
        inferred = tuple(_infer_call_substitutions(symbol, type_parameters, call) for call in calls)
        if any(substitutions is None for substitutions in inferred):
            continue
        concrete = tuple(substitutions for substitutions in inferred if substitutions is not None)
        if not concrete or len(set(concrete)) != 1:
            continue
        specializations.append(
            _member_specialization(
                _MemberSpecializationInputs(
                    symbol=symbol,
                    scope_type_parameter_records=type_parameters,
                    substitutions=concrete[0],
                    origin="closed_call",
                    target_owner_class=symbol.owner_class,
                    forbidden_type_paths=context.forbidden_type_paths,
                )
            )
        )
    return tuple(specializations)


def _owner_class_is_eligible(symbol: SymbolRecord, context: _RegionBuildContext) -> bool:
    if symbol.owner_class is None:
        return False
    owner_id = SymbolId(module=symbol.id.module, qualname=symbol.owner_class)
    owner = next(
        (candidate for candidate in context.module.symbols if candidate.id == owner_id),
        None,
    )
    return owner is not None and _class_supports_member_binding(owner, context)


def _class_supports_member_binding(
    class_symbol: SymbolRecord,
    context: _RegionBuildContext,
) -> bool:
    """Keep method evidence when only whole-class replacement was downgraded.

    Args:
        class_symbol: Scanned class declaration that owns the candidate methods.
        context: Prepared state shared by this operation.

    Returns:
        bool: Whether runtime binding can preserve the selected class member.
    """
    if class_symbol.id in context.eligible:
        return True
    decision = context.downgraded_classes.get(class_symbol.id.qualname)
    if decision is None or class_symbol.blockers:
        return False
    return any(
        reason in decision.reason
        for reason in (
            "source inheritance is outside",
            "module-time code retains",
            "later import can expose",
            "later call can expose",
        )
    )


@dataclass(frozen=True, slots=True)
class _CallSite:
    """A direct same-module function call plus concrete enclosing annotations.

    Attributes:
        node: Syntax node represented by the state.
        enclosing_annotations: Annotations inherited from enclosing declarations.
        receiver_kind: Binding shape of the method receiver.
    """

    node: ast.Call
    enclosing_annotations: dict[str, str]
    receiver_kind: str


class _DirectCallCollector(ast.NodeVisitor):
    """Collect direct calls without importing or evaluating the target module."""

    def __init__(self, forbidden_type_paths: frozenset[str]) -> None:
        self.calls: dict[str, list[_CallSite]] = defaultdict(list)
        self._annotation_stack: list[dict[str, str]] = []
        self._forbidden_type_paths = forbidden_type_paths
        self._class_stack: list[str] = []
        self._shadowed_stack: list[set[str]] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._class_stack.append(node.name)
        try:
            for statement in node.body:
                self.visit(statement)
        finally:
            self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Call(self, node: ast.Call) -> None:
        call_name: str | None = None
        receiver_kind = "none"
        if isinstance(node.func, ast.Name):
            shadowed = self._shadowed_stack[-1] if self._shadowed_stack else set[str]()
            if node.func.id not in shadowed:
                call_name = node.func.id
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            receiver = node.func.value.id
            shadowed = self._shadowed_stack[-1] if self._shadowed_stack else set[str]()
            if receiver in {"self", "cls"} and self._class_stack:
                call_name = f"{self._class_stack[-1]}.{node.func.attr}"
                receiver_kind = receiver
            elif receiver[:1].isupper() and receiver not in shadowed:
                call_name = f"{receiver}.{node.func.attr}"
                receiver_kind = "class"
        if call_name is not None:
            annotations = self._annotation_stack[-1] if self._annotation_stack else {}
            self.calls[call_name].append(
                _CallSite(
                    node=node,
                    enclosing_annotations=annotations,
                    receiver_kind=receiver_kind,
                )
            )
        self.generic_visit(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        type_parameter_names = _node_type_parameter_names(node)
        annotations = {
            argument.arg: ast.unparse(argument.annotation)
            for argument in _function_ast_arguments(node)
            if argument.annotation is not None
            and _specialized_annotation_is_guardable(
                ast.unparse(argument.annotation),
                self._forbidden_type_paths,
            )
            and not _annotation_uses_any(ast.unparse(argument.annotation), type_parameter_names)
        }
        self._annotation_stack.append(annotations)
        self._shadowed_stack.append(_local_bound_names(node))
        try:
            for statement in node.body:
                self.visit(statement)
        finally:
            self._shadowed_stack.pop()
            self._annotation_stack.pop()


class _LocalBindingCollector(ast.NodeVisitor):
    """Collect names that make a bare call local rather than module-global."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.names.add(node.id)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)

    def visit_Import(self, node: ast.Import) -> None:
        self.names.update(
            alias.asname or alias.name.split(".", maxsplit=1)[0] for alias in node.names
        )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.names.update(alias.asname or alias.name for alias in node.names)


def _local_bound_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    collector = _LocalBindingCollector()
    collector.names.update(argument.arg for argument in _function_ast_arguments(node))
    if node.args.vararg is not None:
        collector.names.add(node.args.vararg.arg)
    if node.args.kwarg is not None:
        collector.names.add(node.args.kwarg.arg)
    for statement in node.body:
        collector.visit(statement)
    return collector.names


def _direct_call_sites(
    tree: ast.Module,
    forbidden_type_paths: frozenset[str],
) -> dict[str, tuple[_CallSite, ...]]:
    collector = _DirectCallCollector(forbidden_type_paths)
    for statement in tree.body:
        collector.visit(statement)
    return {name: tuple(calls) for name, calls in collector.calls.items()}


def _ambiguous_module_bindings(tree: ast.Module) -> frozenset[str]:
    counts: defaultdict[str, int] = defaultdict(int)
    for statement in tree.body:
        if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            counts[statement.name] += 1
        elif isinstance(statement, ast.Assign):
            for target in statement.targets:
                for name in _assigned_names(target):
                    counts[name] += 1
        elif isinstance(statement, ast.AnnAssign):
            for name in _assigned_names(statement.target):
                counts[name] += 1
        elif isinstance(statement, ast.Import | ast.ImportFrom):
            for alias in statement.names:
                name = alias.asname or alias.name.split(".", maxsplit=1)[0]
                counts[name] += 1
    return frozenset(name for name, count in counts.items() if count > 1)


def _assigned_names(target: ast.expr) -> tuple[str, ...]:
    return tuple(
        node.id
        for node in ast.walk(target)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    )


def _function_ast_arguments(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[ast.arg, ...]:
    return (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)


def _node_type_parameter_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
    return tuple(
        type_parameter.name
        for type_parameter in getattr(node, "type_params", ())
        if isinstance(type_parameter, ast.TypeVar | ast.ParamSpec | ast.TypeVarTuple)
    )


def _infer_call_substitutions(
    symbol: SymbolRecord,
    type_parameters: tuple[TypeParameterRecord, ...],
    call: _CallSite,
) -> tuple[tuple[str, str], ...] | None:
    parameters = symbol.parameters
    if any(parameter.kind in {"vararg", "kwarg"} for parameter in parameters):
        return None
    arguments = _call_argument_map(symbol, call)
    if arguments is None:
        return None
    type_parameter_names = tuple(record.name for record in type_parameters)
    inferred: dict[str, str] = {}
    for parameter in parameters:
        parameter_inferred = _infer_parameter_substitutions(
            parameter,
            arguments,
            call.enclosing_annotations,
            type_parameter_names,
        )
        if parameter_inferred == ():
            continue
        if parameter_inferred is None:
            return None
        for name, annotation in parameter_inferred:
            if name in inferred and inferred[name] != annotation:
                return None
            inferred[name] = annotation
    if any(name not in inferred for name in type_parameter_names):
        return None
    return tuple((name, inferred[name]) for name in type_parameter_names)


def _call_argument_map(
    symbol: SymbolRecord,
    call_site: _CallSite,
) -> dict[str, ast.expr] | None:
    parameters = symbol.parameters
    call = call_site.node
    if call.keywords and any(keyword.arg is None for keyword in call.keywords):
        return None
    positional_parameters = tuple(
        parameter for parameter in parameters if parameter.kind in {"positional_only", "positional"}
    )
    receiver_is_bound = (
        symbol.binding_kind == "instance_method" and call_site.receiver_kind == "self"
    ) or (symbol.binding_kind == "classmethod" and call_site.receiver_kind in {"cls", "class"})
    if receiver_is_bound:
        positional_parameters = positional_parameters[1:]
    if len(call.args) > len(positional_parameters):
        return None
    arguments: dict[str, ast.expr] = {
        parameter.name: argument
        for parameter, argument in zip(positional_parameters, call.args, strict=False)
    }
    parameter_by_name = {parameter.name: parameter for parameter in parameters}
    for keyword in call.keywords:
        if (
            keyword.arg is None
            or keyword.arg not in parameter_by_name
            or keyword.arg in arguments
            or parameter_by_name[keyword.arg].kind == "positional_only"
        ):
            return None
        arguments[keyword.arg] = keyword.value
    return arguments


def _infer_parameter_substitutions(
    parameter: ParameterRecord,
    arguments: dict[str, ast.expr],
    enclosing_annotations: dict[str, str],
    type_parameter_names: tuple[str, ...],
) -> tuple[tuple[str, str], ...] | None:
    if parameter.annotation is None or not _annotation_uses_any(
        parameter.annotation, type_parameter_names
    ):
        return ()
    argument = arguments.get(parameter.name)
    if argument is None:
        return None
    argument_annotation = _argument_annotation(argument, enclosing_annotations)
    if argument_annotation is None:
        return None
    return _infer_from_parameter_annotation(
        parameter.annotation,
        argument_annotation,
        type_parameter_names,
    )


def _argument_annotation(
    argument: ast.expr,
    enclosing_annotations: dict[str, str],
) -> str | None:
    if isinstance(argument, ast.Constant):
        if argument.value is None:
            return "None"
        return type(argument.value).__name__
    if (
        isinstance(argument, ast.UnaryOp)
        and isinstance(argument.op, ast.UAdd | ast.USub)
        and isinstance(argument.operand, ast.Constant)
        and isinstance(argument.operand.value, int | float)
    ):
        return type(argument.operand.value).__name__
    if isinstance(argument, ast.Name):
        return enclosing_annotations.get(argument.id)
    if isinstance(argument, ast.Call) and not argument.args and not argument.keywords:
        path = _annotation_path(argument.func)
        return ".".join(path) if path is not None else None
    return None


def _infer_from_parameter_annotation(
    parameter_annotation: str,
    argument_annotation: str,
    type_parameter_names: tuple[str, ...],
) -> tuple[tuple[str, str], ...] | None:
    try:
        expression = ast.parse(parameter_annotation, mode="eval").body
    except SyntaxError:
        return None
    if isinstance(expression, ast.Name) and expression.id in type_parameter_names:
        return ((expression.id, argument_annotation),)
    union_items = _union_items(expression)
    if union_items is None:
        return None
    inferred = [
        item.id
        for item in union_items
        if isinstance(item, ast.Name) and item.id in type_parameter_names
    ]
    if len(inferred) != 1 or argument_annotation == "None":
        return None
    return ((inferred[0], argument_annotation),)


def _member_specialization(inputs: _MemberSpecializationInputs) -> RegionSpecialization | None:
    symbol = inputs.symbol
    scope_type_parameter_records = inputs.scope_type_parameter_records
    substitutions = inputs.substitutions
    forbidden_type_paths = inputs.forbidden_type_paths
    type_parameter_names = tuple(record.name for record in scope_type_parameter_records)
    if (
        any(record.kind != "type_var" for record in scope_type_parameter_records)
        or {name for name, _ in substitutions} != set(type_parameter_names)
        or bool(set(type_parameter_names) & set(symbol.uses_globals))
        or any(
            not _specialized_annotation_is_guardable(annotation, forbidden_type_paths)
            for _, annotation in substitutions
        )
    ):
        return None
    guards = _runtime_guards(
        symbol,
        type_parameter_names,
        substitutions,
        forbidden_type_paths,
    )
    if guards is None:
        return None
    bindings = _specialized_type_bindings(
        symbol,
        scope_type_parameter_records,
        substitutions,
        forbidden_type_paths,
    )
    if bindings is None:
        return None
    evidence = _SpecializationEvidence(
        symbol=symbol,
        origin=inputs.origin,
        target_owner_class=inputs.target_owner_class,
        substitutions=substitutions,
        guards=guards,
        bindings=bindings,
    )
    spec_id = _specialization_id(evidence)
    return RegionSpecialization(
        id=spec_id,
        source_member=symbol.id,
        source_owner_class=symbol.owner_class,
        target_owner_class=inputs.target_owner_class,
        origin=inputs.origin,
        substitutions=substitutions,
        guards=guards,
        type_bindings=bindings,
    )


def _runtime_guards(
    symbol: SymbolRecord,
    type_parameter_names: tuple[str, ...],
    substitutions: tuple[tuple[str, str], ...],
    forbidden_type_paths: frozenset[str],
) -> tuple[RuntimeTypeGuard, ...] | None:
    if any(parameter.kind in {"vararg", "kwarg"} for parameter in symbol.parameters):
        return None
    guards: list[RuntimeTypeGuard] = []
    for index, parameter in enumerate(symbol.parameters):
        if parameter.annotation is None:
            continue
        if not _annotation_uses_any(parameter.annotation, type_parameter_names):
            continue
        if parameter.default_source is not None:
            return None
        annotation = _substitute_annotation(parameter.annotation, substitutions)
        guard_shape = _guard_shape(annotation, forbidden_type_paths)
        if guard_shape is None:
            return None
        nominal_paths, allow_none = guard_shape
        guards.append(
            RuntimeTypeGuard(
                parameter_name=parameter.name,
                positional_index=None if parameter.kind == "keyword_only" else index,
                annotation=annotation,
                nominal_type_paths=nominal_paths,
                allow_none=allow_none,
            )
        )
    return tuple(guards)


def _specialized_type_bindings(
    symbol: SymbolRecord,
    scope_type_parameter_records: tuple[TypeParameterRecord, ...],
    substitutions: tuple[tuple[str, str], ...],
    forbidden_type_paths: frozenset[str],
) -> tuple[TypeBinding, ...] | None:
    bindings = [
        binding
        for binding in _symbol_type_bindings(symbol, scope_type_parameter_records)
        if binding.source != "type_parameter"
    ]
    specialized: list[TypeBinding] = []
    type_parameter_names = tuple(record.name for record in scope_type_parameter_records)
    for binding in bindings:
        annotation = _substitute_annotation(binding.annotation, substitutions)
        if _annotation_uses_any(annotation, type_parameter_names) or _annotation_has_forbidden_path(
            annotation,
            forbidden_type_paths,
        ):
            return None
        if not _is_concrete(annotation, (), ()):
            return None
        specialized.append(
            TypeBinding(
                name=binding.name,
                annotation=annotation,
                source=binding.source,
                concrete=True,
                substitutions=substitutions,
            )
        )
    return tuple(specialized)


def _method_uses_generic_field(
    method: SymbolRecord,
    owner: SymbolRecord,
    substitutions: tuple[tuple[str, str], ...],
    source_lines: list[str],
) -> bool:
    generic_fields = {
        field.name
        for field in owner.fields
        if _annotation_uses_any(field.annotation, tuple(name for name, _ in substitutions))
    }
    if not generic_fields:
        return False
    start_lineno = method.declaration_start_lineno or method.lineno
    source_text = "\n".join(source_lines[start_lineno - 1 : method.end_lineno])
    try:
        method_tree = ast.parse(dedent(source_text))
    except SyntaxError:
        method_tree = None
    if method_tree is None:
        return bool(generic_fields & set(method.referenced_names))
    return any(
        (isinstance(node, ast.Attribute) and node.attr in generic_fields)
        or _literal_getattr_field(node) in generic_fields
        for node in ast.walk(method_tree)
    )


def _literal_getattr_field(node: ast.AST) -> str | None:
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= _GETATTR_REQUIRED_ARGS
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id in {"self", "cls"}
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    ):
        return node.args[1].value
    return None


def _substitute_annotation(
    annotation: str,
    substitutions: tuple[tuple[str, str], ...],
) -> str:
    expression = _annotation_expression(annotation)
    if expression is None:
        return annotation
    transformer = _AnnotationSubstituter(dict(substitutions))
    substituted = transformer.visit(expression)
    ast.fix_missing_locations(substituted)
    return ast.unparse(substituted)


class _AnnotationSubstituter(ast.NodeTransformer):
    """Replace type variables in an annotation AST with concrete expressions."""

    def __init__(self, substitutions: dict[str, str]) -> None:
        self._substitutions = substitutions

    def visit_Name(self, node: ast.Name) -> ast.AST:
        replacement = self._substitutions.get(node.id)
        if replacement is None:
            return node
        return ast.parse(replacement, mode="eval").body


def _annotation_uses_any(annotation: str, names: tuple[str, ...]) -> bool:
    expression = _annotation_expression(annotation)
    if expression is None:
        return False
    return any(isinstance(node, ast.Name) and node.id in names for node in ast.walk(expression))


def _specialized_annotation_is_guardable(
    annotation: str,
    forbidden_type_paths: frozenset[str],
) -> bool:
    return _guard_shape(annotation, forbidden_type_paths) is not None


def _guard_shape(
    annotation: str,
    forbidden_type_paths: frozenset[str],
) -> tuple[tuple[str, ...], bool] | None:
    expression = _annotation_expression(annotation)
    if expression is None:
        return None
    items = _union_items(expression) or (expression,)
    nominal_paths: list[str] = []
    allow_none = False
    for item in items:
        if isinstance(item, ast.Constant) and item.value is None:
            allow_none = True
            continue
        path = _annotation_path(item)
        if path is None:
            return None
        path_text = ".".join(path)
        if _type_path_is_forbidden(path_text, forbidden_type_paths):
            return None
        nominal_paths.append(path_text)
    return tuple(sorted(set(nominal_paths))), allow_none


def _union_items(expression: ast.expr) -> tuple[ast.expr, ...] | None:
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.BitOr):
        left = _union_items(expression.left) or (expression.left,)
        right = _union_items(expression.right) or (expression.right,)
        return (*left, *right)
    if isinstance(expression, ast.Subscript):
        path = _annotation_path(expression.value)
        if path is not None and path[-1] == "Union":
            return _subscript_arguments(expression.slice)
        if path is not None and path[-1] == "Optional":
            return (*_subscript_arguments(expression.slice), ast.Constant(value=None))
    return None


def _annotation_expression(annotation: str) -> ast.expr | None:
    """Parse one annotation and unwrap forward-reference string literals.

    Args:
        annotation: Source annotation expression being inspected or rewritten.

    Returns:
        ast.expr | None: Parsed annotation expression, or `None` when syntax is invalid.
    """
    try:
        expression = ast.parse(annotation, mode="eval").body
    except SyntaxError:
        return None
    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        try:
            return ast.parse(expression.value, mode="eval").body
        except SyntaxError:
            return None
    return expression


def _annotation_has_forbidden_path(
    annotation: str,
    forbidden_type_paths: frozenset[str],
) -> bool:
    expression = _annotation_expression(annotation)
    if expression is None:
        return True
    return any(
        _type_path_is_forbidden(".".join(path), forbidden_type_paths)
        for node in ast.walk(expression)
        for path in (_annotation_path(node),)
        if path is not None
    )


def _type_path_is_forbidden(path: str, forbidden_type_paths: frozenset[str]) -> bool:
    return path in forbidden_type_paths or path.rsplit(".", maxsplit=1)[-1] == "Any"


def _semantic_any_paths(module: ModuleScan) -> frozenset[str]:
    """Return direct and aliased typing.Any paths visible in one source module.

    Args:
        module: Scanned module or syntax module being processed.

    Returns:
        frozenset[str]: Stable paths through which `Any` enters the annotation.
    """
    paths = set(_DEFAULT_ANY_PATHS)
    for record in module.imports:
        try:
            statement = ast.parse(record.source_text).body[0]
        except (SyntaxError, IndexError):
            continue
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                if alias.name in {"typing", "typing_extensions"}:
                    paths.add(f"{alias.asname or alias.name}.Any")
        elif isinstance(statement, ast.ImportFrom) and statement.module in {
            "typing",
            "typing_extensions",
        }:
            paths.update(
                alias.asname or alias.name for alias in statement.names if alias.name == "Any"
            )
    return frozenset(paths)


def _annotation_path(expression: ast.AST) -> tuple[str, ...] | None:
    if isinstance(expression, ast.Name):
        return (expression.id,)
    if isinstance(expression, ast.Attribute):
        parent = _annotation_path(expression.value)
        return None if parent is None else (*parent, expression.attr)
    return None


def _subscript_arguments(slice_node: ast.expr) -> tuple[ast.expr, ...]:
    if isinstance(slice_node, ast.Tuple):
        return tuple(slice_node.elts)
    return (slice_node,)


def _specialization_id(evidence: _SpecializationEvidence) -> str:
    payload = _specialization_payload(evidence)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"{evidence.symbol.id.stable_id}:specialized:{hashlib.sha256(encoded).hexdigest()[:12]}"


def _specialization_payload(evidence: _SpecializationEvidence) -> dict[str, object]:
    return {
        "source_member": evidence.symbol.id.stable_id,
        "source_owner_class": evidence.symbol.owner_class,
        "target_owner_class": evidence.target_owner_class,
        "origin": evidence.origin,
        "substitutions": list(evidence.substitutions),
        "guards": [
            {
                "parameter_name": guard.parameter_name,
                "positional_index": guard.positional_index,
                "annotation": guard.annotation,
                "nominal_type_paths": list(guard.nominal_type_paths),
                "allow_none": guard.allow_none,
            }
            for guard in evidence.guards
        ],
        "type_bindings": [
            {
                "name": binding.name,
                "annotation": binding.annotation,
                "source": binding.source,
                "concrete": binding.concrete,
                "substitutions": list(binding.substitutions),
            }
            for binding in evidence.bindings
        ],
    }


def _region_hash(
    module_name: str,
    members: tuple[RegionMember, ...],
    dependencies: tuple[RegionDependency, ...],
    type_bindings: tuple[TypeBinding, ...],
    specializations: tuple[RegionSpecialization, ...],
) -> str:
    payload = {
        "version": TYPED_REGION_SCHEMA_VERSION,
        "module": module_name,
        "members": [
            {
                "id": member.id.stable_id,
                "source": member.source_text,
                "binding": member.binding_kind,
                "execution": member.execution_kind,
                "type_parameters": list(member.type_parameters),
                "scope_type_parameters": list(member.scope_type_parameters),
                "type_parameter_records": [
                    {
                        "name": record.name,
                        "kind": record.kind,
                        "declaration": record.declaration,
                    }
                    for record in member.type_parameter_records
                ],
                "scope_type_parameter_records": [
                    {
                        "name": record.name,
                        "kind": record.kind,
                        "declaration": record.declaration,
                    }
                    for record in member.scope_type_parameter_records
                ],
                "call_sites": [
                    {
                        "target": call.target,
                        "root_name": call.root_name,
                        "invocation_mode": call.invocation_mode,
                        "lineno": call.lineno,
                        "end_lineno": call.end_lineno,
                        "col_offset": call.col_offset,
                        "end_col_offset": call.end_col_offset,
                        "requires_same_unit": call.requires_same_unit,
                    }
                    for call in member.call_sites
                ],
                "suspension_points": [
                    {
                        "kind": point.kind,
                        "lineno": point.lineno,
                        "end_lineno": point.end_lineno,
                        "col_offset": point.col_offset,
                        "end_col_offset": point.end_col_offset,
                    }
                    for point in member.suspension_points
                ],
                "runtime_imports": [record.source_text for record in member.runtime_imports],
            }
            for member in members
        ],
        "dependencies": [
            {
                "src": dependency.src.stable_id,
                "dst": (
                    dependency.dst.stable_id
                    if isinstance(dependency.dst, SymbolId)
                    else dependency.dst
                ),
                "kind": dependency.kind,
                "confidence": dependency.confidence,
                "role": dependency.role,
                "type_only": dependency.type_only,
                "lineno": dependency.lineno,
                "invocation_mode": dependency.invocation_mode,
                "requires_same_unit": dependency.requires_same_unit,
            }
            for dependency in dependencies
        ],
        "types": [
            {
                "name": binding.name,
                "annotation": binding.annotation,
                "source": binding.source,
                "concrete": binding.concrete,
            }
            for binding in type_bindings
        ],
        "specializations": [
            {
                "id": specialization.id,
                "source_member": specialization.source_member.stable_id,
                "source_owner_class": specialization.source_owner_class,
                "target_owner_class": specialization.target_owner_class,
                "origin": specialization.origin,
                "substitutions": list(specialization.substitutions),
                "guards": [
                    {
                        "parameter_name": guard.parameter_name,
                        "positional_index": guard.positional_index,
                        "annotation": guard.annotation,
                        "nominal_type_paths": list(guard.nominal_type_paths),
                        "allow_none": guard.allow_none,
                    }
                    for guard in specialization.guards
                ],
                "type_bindings": [
                    {
                        "name": binding.name,
                        "annotation": binding.annotation,
                        "source": binding.source,
                        "concrete": binding.concrete,
                        "substitutions": list(binding.substitutions),
                    }
                    for binding in specialization.type_bindings
                ],
            }
            for specialization in specializations
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
