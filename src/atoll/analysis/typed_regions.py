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

from atoll.models import (
    BindingTarget,
    DependencyEdge,
    LoweringDecision,
    ModuleScan,
    ParameterRecord,
    RegionDependency,
    RegionMember,
    SymbolId,
    SymbolRecord,
    TypeBinding,
    TypedRegion,
    TypeParameterRecord,
)

TYPED_REGION_SCHEMA_VERSION = "atoll-typed-region-v1"
_UNSAFE_SOFT_BLOCKERS = frozenset({"UNTYPED_DECORATOR"})


@dataclass(frozen=True, slots=True)
class _RegionBuildContext:
    """Immutable module evidence shared while materializing components."""

    module: ModuleScan
    eligible: dict[SymbolId, SymbolRecord]
    edges: tuple[DependencyEdge, ...]
    source_lines: list[str]
    downgraded_classes: dict[str, LoweringDecision]


def build_typed_regions(
    module: ModuleScan,
    edges: tuple[DependencyEdge, ...],
) -> tuple[TypedRegion, ...]:
    """Return deterministic connected regions without lowering their types.

    Hard-blocked declarations and declarations changed by unknown decorators
    are excluded. A class is included atomically only when every directly
    declared method is eligible; otherwise its eligible methods remain
    independent callable members on the original source class.
    """
    eligible = {symbol.id: symbol for symbol in module.symbols if _eligible(symbol)}
    if not eligible:
        return ()
    adjacency = _adjacency(eligible, edges)
    downgraded_classes = _connect_atomic_classes(module.symbols, eligible, adjacency)
    components = _components(eligible, adjacency)
    source_lines = module.module.path.read_text(encoding="utf-8").splitlines()
    context = _RegionBuildContext(
        module=module,
        eligible=eligible,
        edges=edges,
        source_lines=source_lines,
        downgraded_classes=downgraded_classes,
    )
    return tuple(_typed_region(context, component) for component in components)


def _eligible(symbol: SymbolRecord) -> bool:
    if any(blocker.severity == "hard" for blocker in symbol.blockers):
        return False
    if any(blocker.code in _UNSAFE_SOFT_BLOCKERS for blocker in symbol.blockers):
        return False
    if symbol.kind == "class":
        return True
    return _signature_is_complete(symbol)


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
        if class_symbol.id not in eligible:
            downgraded[owner] = LoweringDecision(
                target=class_symbol.id.stable_id,
                action="fallback",
                reason="class remains interpreted because its dynamic behavior is blocked",
            )
            continue
        if any(method.id not in eligible for method in methods):
            eligible.pop(class_symbol.id, None)
            adjacency.pop(class_symbol.id, None)
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


def _region_hash(
    module_name: str,
    members: tuple[RegionMember, ...],
    dependencies: tuple[RegionDependency, ...],
    type_bindings: tuple[TypeBinding, ...],
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
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
