"""AST scanner for Atoll's first-pass module analysis.

The scanner reads source text, never imports project modules, and records only
facts that can be derived from Python syntax. Later phases decide whether those
facts are safe enough for sidecar extraction, so this module favors conservative
boundaries over speculative call resolution.
"""

from __future__ import annotations

import ast
import builtins
from collections.abc import Iterable
from dataclasses import dataclass

from atoll.analysis.blockers import (
    detect_class_blockers,
    detect_function_blockers,
    module_level_blockers,
)
from atoll.models import (
    BindingKind,
    ConstantKind,
    ConstantRecord,
    ExecutionKind,
    FieldRecord,
    ImportRecord,
    ModuleId,
    ModuleScan,
    ParameterKind,
    ParameterRecord,
    SymbolId,
    SymbolKind,
    SymbolRecord,
    TypeParameterKind,
    TypeParameterRecord,
    Visibility,
)

_BUILTIN_NAMES = frozenset(dir(builtins))
_TYPING_MODULES = frozenset({"typing", "typing_extensions"})
_TYPE_PARAMETER_FACTORIES = frozenset({"TypeVar", "ParamSpec", "TypeVarTuple"})
_QUALIFIED_TYPING_PATH_LENGTH = 2


@dataclass(frozen=True, slots=True)
class _TypingAliases:
    """Imported typing aliases and legacy module-level type parameters.

    Attributes:
        any_names: Names that resolve to `typing.Any` in the module.
        typing_module_names: Aliases bound to the `typing` module.
        module_type_parameters: Type parameters declared by the module scope.
    """

    any_names: frozenset[str]
    typing_module_names: frozenset[str]
    module_type_parameters: tuple[TypeParameterRecord, ...]


@dataclass(frozen=True, slots=True)
class _FunctionRecordContext:
    """Source ownership and import context needed to build a function symbol.

    Attributes:
        module_name: Importable source module name.
        kind: Classified declaration, binding, or receiver kind.
        qualname: Module-local qualified declaration name.
        owner_class: Source class owning the selected member.
        typing_aliases: Typing names and aliases visible in the module.
    """

    module_name: str
    kind: SymbolKind
    qualname: str
    owner_class: str | None
    typing_aliases: _TypingAliases


def scan_module(module: ModuleId) -> ModuleScan:
    """Parse one Python file and return its first-pass scan facts.

    The scan includes top-level imports, literal/dynamic constant
    classifications, functions/classes/simple methods, module blockers, and
    executable statement locations. It deliberately omits mypy diagnostics and
    candidate scoring because those depend on later enrichment phases.

    Args:
        module: Module scan or module identity being analyzed.

    Returns:
        ModuleScan: Immutable AST-derived scan facts for the module.
    """
    source = module.path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(module.path), type_comments=True)
    lines = source.splitlines()
    imports = tuple(_import_record(node, lines) for node in tree.body if _is_import_node(node))
    typing_aliases = _typing_aliases(tree.body)
    constants = tuple(record for node in tree.body for record in _constant_records(node, lines))
    symbols = tuple(
        record
        for node in tree.body
        for record in _symbol_records(module.name, node, typing_aliases)
    )
    module_blockers = module_level_blockers(tree.body, module.name)
    statement_lines = tuple(
        node.lineno for node in tree.body if _is_top_level_executable_statement(node)
    )
    return ModuleScan(
        module=module,
        imports=imports,
        constants=constants,
        symbols=symbols,
        blockers=module_blockers,
        top_level_statement_lines=statement_lines,
    )


def _symbol_records(
    module_name: str,
    node: ast.stmt,
    typing_aliases: _TypingAliases,
) -> tuple[SymbolRecord, ...]:
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return (
            _function_record(
                node,
                _FunctionRecordContext(
                    module_name=module_name,
                    kind="function",
                    qualname=node.name,
                    owner_class=None,
                    typing_aliases=typing_aliases,
                ),
            ),
        )
    if isinstance(node, ast.ClassDef):
        class_record = _class_record(module_name, node, typing_aliases)
        methods = tuple(
            _function_record(
                child,
                _FunctionRecordContext(
                    module_name=module_name,
                    kind="method",
                    qualname=f"{node.name}.{child.name}",
                    owner_class=node.name,
                    typing_aliases=typing_aliases,
                ),
            )
            for child in node.body
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
        )
        return (class_record, *methods)
    return ()


def _function_record(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    context: _FunctionRecordContext,
) -> SymbolRecord:
    symbol = SymbolId(module=context.module_name, qualname=context.qualname)
    collector = _FunctionNameCollector()
    collector.collect(node)
    annotations = _function_annotations(node)
    annotation_names = _annotation_names(annotations)
    type_parameter_records = _type_parameter_records(node)
    scope_type_parameter_records = _scope_type_parameter_records(
        type_parameter_records,
        annotation_names,
        context.typing_aliases.module_type_parameters,
    )
    arg_count, annotated_arg_count = _argument_counts(node)
    blockers = detect_function_blockers(node, symbol)
    referenced_names = tuple(sorted(collector.referenced_names))
    runtime_referenced_names = tuple(sorted(collector.runtime_referenced_names))
    local_names = tuple(sorted(collector.local_names))
    return SymbolRecord(
        id=symbol,
        kind=context.kind,
        visibility=_visibility(symbol.qualname),
        lineno=node.lineno,
        end_lineno=_end_lineno(node),
        col_offset=node.col_offset,
        end_col_offset=node.end_col_offset,
        decorators=_decorators(node.decorator_list),
        owner_class=context.owner_class,
        binding_kind=_binding_kind(context.owner_class, node.decorator_list),
        execution_kind=_execution_kind(node),
        type_parameters=tuple(record.name for record in type_parameter_records),
        scope_type_parameters=tuple(record.name for record in scope_type_parameter_records),
        type_parameter_records=type_parameter_records,
        scope_type_parameter_records=scope_type_parameter_records,
        parameters=_parameter_records(node),
        return_annotation=_annotation_source(node.returns),
        annotation_names=annotation_names,
        called_paths=tuple(sorted(collector.called_paths)),
        base_names=(),
        declaration_start_lineno=_declaration_start_lineno(node),
        any_annotation_sources=_any_annotation_sources(
            annotations,
            context.typing_aliases,
        ),
        arg_count=arg_count,
        annotated_arg_count=annotated_arg_count,
        has_return_annotation=node.returns is not None,
        has_any_annotation=_has_any_annotation(annotations, context.typing_aliases),
        called_names=tuple(sorted(collector.called_names)),
        uses_globals=_global_names(runtime_referenced_names, local_names),
        local_names=local_names,
        referenced_names=referenced_names,
        blockers=blockers,
    )


def _class_record(
    module_name: str,
    node: ast.ClassDef,
    typing_aliases: _TypingAliases,
) -> SymbolRecord:
    symbol = SymbolId(module=module_name, qualname=node.name)
    collector = _ClassNameCollector()
    collector.visit(node)
    blockers = detect_class_blockers(node, symbol)
    referenced_names = tuple(sorted(collector.referenced_names))
    local_names = tuple(sorted(collector.local_names))
    fields = _class_field_records(node)
    class_annotations = (
        *node.bases,
        *(
            child.annotation
            for child in node.body
            if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name)
        ),
    )
    annotation_names = _annotation_names(class_annotations)
    type_parameter_records = _type_parameter_records(node)
    scope_type_parameter_records = _scope_type_parameter_records(
        type_parameter_records,
        annotation_names,
        typing_aliases.module_type_parameters,
    )
    return SymbolRecord(
        id=symbol,
        kind="class",
        visibility=_visibility(symbol.qualname),
        lineno=node.lineno,
        end_lineno=_end_lineno(node),
        col_offset=node.col_offset,
        end_col_offset=node.end_col_offset,
        decorators=_decorators(node.decorator_list),
        owner_class=None,
        binding_kind="class",
        execution_kind="class",
        type_parameters=tuple(record.name for record in type_parameter_records),
        scope_type_parameters=tuple(record.name for record in scope_type_parameter_records),
        type_parameter_records=type_parameter_records,
        scope_type_parameter_records=scope_type_parameter_records,
        parameters=(),
        return_annotation=None,
        annotation_names=annotation_names,
        called_paths=(),
        base_names=_base_names(node),
        fields=fields,
        declaration_start_lineno=_declaration_start_lineno(node),
        any_annotation_sources=_any_annotation_sources(class_annotations, typing_aliases),
        arg_count=0,
        annotated_arg_count=0,
        has_return_annotation=False,
        has_any_annotation=_has_any_annotation(class_annotations, typing_aliases),
        called_names=(),
        uses_globals=_global_names(referenced_names, local_names),
        local_names=local_names,
        referenced_names=referenced_names,
        blockers=blockers,
    )


def _global_names(
    referenced_names: tuple[str, ...],
    local_names: tuple[str, ...],
) -> tuple[str, ...]:
    global_names = sorted(set(referenced_names) - set(local_names) - _BUILTIN_NAMES)
    return tuple(global_names)


def _declaration_start_lineno(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> int:
    """Return the first decorator line or the declaration line when undecorated.

    Args:
        node: Syntax node being visited without executing target code.

    Returns:
        int: First line occupied by decorators or declaration syntax.
    """
    return min((decorator.lineno for decorator in node.decorator_list), default=node.lineno)


def _typing_aliases(nodes: list[ast.stmt]) -> _TypingAliases:
    any_names: set[str] = set()
    typing_module_names: set[str] = set()
    type_parameter_factory_names: dict[str, str] = {}
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _TYPING_MODULES:
                    typing_module_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module in _TYPING_MODULES:
            for alias in node.names:
                if alias.name == "Any":
                    any_names.add(alias.asname or alias.name)
                if alias.name in _TYPE_PARAMETER_FACTORIES:
                    type_parameter_factory_names[alias.asname or alias.name] = alias.name
    module_type_parameters = _module_type_parameter_records(
        nodes,
        typing_module_names=typing_module_names,
        type_parameter_factory_names=type_parameter_factory_names,
    )
    return _TypingAliases(
        any_names=frozenset(any_names),
        typing_module_names=frozenset(typing_module_names),
        module_type_parameters=module_type_parameters,
    )


def _module_type_parameter_records(
    nodes: list[ast.stmt],
    *,
    typing_module_names: set[str],
    type_parameter_factory_names: dict[str, str],
) -> tuple[TypeParameterRecord, ...]:
    records: list[TypeParameterRecord] = []
    for node in nodes:
        value: ast.expr | None
        if isinstance(node, ast.Assign):
            targets = tuple(target.id for target in node.targets if isinstance(target, ast.Name))
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = (node.target.id,)
            value = node.value
        else:
            continue
        if value is None:
            continue
        factory_name = _type_parameter_factory_name(
            value,
            typing_module_names=typing_module_names,
            type_parameter_factory_names=type_parameter_factory_names,
        )
        if factory_name is None:
            continue
        records.extend(
            TypeParameterRecord(
                name=target,
                kind=_type_parameter_kind(factory_name),
                declaration=f"{target} = {ast.unparse(value)}",
            )
            for target in targets
        )
    unique: dict[str, TypeParameterRecord] = {}
    for record in records:
        unique.setdefault(record.name, record)
    return tuple(unique.values())


def _type_parameter_factory_name(
    node: ast.expr,
    *,
    typing_module_names: set[str],
    type_parameter_factory_names: dict[str, str],
) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    if isinstance(node.func, ast.Name):
        return type_parameter_factory_names.get(node.func.id)
    path = _attribute_path(node.func)
    if (
        path is not None
        and len(path) == _QUALIFIED_TYPING_PATH_LENGTH
        and path[0] in typing_module_names
        and path[1] in _TYPE_PARAMETER_FACTORIES
    ):
        return path[1]
    return None


def _import_record(node: ast.stmt, lines: list[str]) -> ImportRecord:
    if isinstance(node, ast.Import):
        imported_names = tuple(
            alias.asname or alias.name.split(".", maxsplit=1)[0] for alias in node.names
        )
        return ImportRecord(
            source_text=_source_text(node, lines),
            imported_names=imported_names,
            module=None,
            level=0,
            lineno=node.lineno,
            end_lineno=_end_lineno(node),
        )
    if isinstance(node, ast.ImportFrom):
        imported_names = tuple(
            alias.asname or alias.name for alias in node.names if alias.name != "*"
        )
        return ImportRecord(
            source_text=_source_text(node, lines),
            imported_names=imported_names,
            module=node.module,
            level=node.level,
            lineno=node.lineno,
            end_lineno=_end_lineno(node),
        )
    raise TypeError(f"unsupported import node: {type(node).__name__}")


def _constant_records(node: ast.stmt, lines: list[str]) -> tuple[ConstantRecord, ...]:
    if isinstance(node, ast.Assign):
        return tuple(
            ConstantRecord(
                name=target.id,
                kind=_constant_kind(node.value),
                source_text=_source_text(node, lines),
                lineno=node.lineno,
                end_lineno=_end_lineno(node),
            )
            for target in node.targets
            if isinstance(target, ast.Name)
        )
    if (
        isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.value is not None
    ):
        return (
            ConstantRecord(
                name=node.target.id,
                kind=_constant_kind(node.value),
                source_text=_source_text(node, lines),
                lineno=node.lineno,
                end_lineno=_end_lineno(node),
            ),
        )
    return ()


def _constant_kind(node: ast.expr) -> ConstantKind:
    if _is_safe_literal(node):
        return "literal_constant"
    if isinstance(node, ast.Call):
        return "runtime_dynamic"
    return "unknown"


def _is_safe_literal(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str | int | float | bool | type(None))
    if isinstance(node, ast.Tuple):
        return all(_is_safe_literal(element) for element in node.elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
        return _is_safe_literal(node.operand)
    return False


def _argument_counts(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[int, int]:
    args = _function_arguments(node)
    return len(args), sum(argument.annotation is not None for argument in args)


def _parameter_records(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[ParameterRecord, ...]:
    positional_defaults = _positional_defaults(node)
    records = [
        *[
            _parameter_record(argument, "positional_only", positional_defaults.get(argument.arg))
            for argument in node.args.posonlyargs
        ],
        *[
            _parameter_record(argument, "positional", positional_defaults.get(argument.arg))
            for argument in node.args.args
        ],
    ]
    if node.args.vararg is not None:
        records.append(_parameter_record(node.args.vararg, "vararg", None))
    for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True):
        records.append(_parameter_record(argument, "keyword_only", _annotation_source(default)))
    if node.args.kwarg is not None:
        records.append(_parameter_record(node.args.kwarg, "kwarg", None))
    return tuple(records)


def _parameter_record(
    argument: ast.arg,
    kind: ParameterKind,
    default_source: str | None,
) -> ParameterRecord:
    return ParameterRecord(
        name=argument.arg,
        kind=kind,
        annotation=_annotation_source(argument.annotation),
        default_source=default_source,
    )


def _class_field_records(node: ast.ClassDef) -> tuple[FieldRecord, ...]:
    return tuple(
        FieldRecord(
            name=child.target.id,
            annotation=ast.unparse(child.annotation),
            default_source=_annotation_source(child.value),
            class_variable=_is_class_variable(child.annotation),
        )
        for child in node.body
        if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name)
    )


def _is_class_variable(annotation: ast.expr) -> bool:
    target = annotation.value if isinstance(annotation, ast.Subscript) else annotation
    path = _attribute_path(target)
    return path is not None and path[-1] == "ClassVar"


def _positional_defaults(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, str | None]:
    positional = (*node.args.posonlyargs, *node.args.args)
    default_offset = len(positional) - len(node.args.defaults)
    return {
        argument.arg: _annotation_source(default)
        for argument, default in zip(positional[default_offset:], node.args.defaults, strict=True)
    }


def _function_annotations(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.expr | None, ...]:
    return (
        *(argument.annotation for argument in _function_arguments(node)),
        node.returns,
    )


def _decorators(decorators: list[ast.expr]) -> tuple[str, ...]:
    return tuple(ast.unparse(decorator) for decorator in decorators)


def _binding_kind(
    owner_class: str | None,
    decorators: list[ast.expr],
) -> BindingKind:
    if owner_class is None:
        return "module"
    decorator_names = {_decorator_name(decorator) for decorator in decorators}
    if "staticmethod" in decorator_names:
        return "staticmethod"
    if "classmethod" in decorator_names:
        return "classmethod"
    return "instance_method"


def _decorator_name(decorator: ast.expr) -> str | None:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    path = _attribute_path(target)
    return path[-1] if path is not None else None


def _execution_kind(node: ast.FunctionDef | ast.AsyncFunctionDef) -> ExecutionKind:
    contains_yield = _contains_yield(node)
    if isinstance(node, ast.AsyncFunctionDef):
        return "async_generator" if contains_yield else "coroutine"
    return "generator" if contains_yield else "sync"


def _contains_yield(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    visitor = _YieldShapeVisitor()
    for child in node.body:
        visitor.visit(child)
    return visitor.contains_yield


def _type_parameter_records(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> tuple[TypeParameterRecord, ...]:
    type_params = getattr(node, "type_params", ())
    return tuple(
        TypeParameterRecord(
            name=_type_parameter_name(type_param),
            kind=_ast_type_parameter_kind(type_param),
            declaration=ast.unparse(type_param),
        )
        for type_param in type_params
    )


def _scope_type_parameter_records(
    declared: tuple[TypeParameterRecord, ...],
    annotation_names: tuple[str, ...],
    module_type_parameters: tuple[TypeParameterRecord, ...],
) -> tuple[TypeParameterRecord, ...]:
    referenced_names = set(annotation_names)
    records = [
        *declared,
        *(record for record in module_type_parameters if record.name in referenced_names),
    ]
    unique: dict[str, TypeParameterRecord] = {}
    for record in records:
        unique.setdefault(record.name, record)
    return tuple(unique.values())


def _type_parameter_name(node: ast.AST) -> str:
    if isinstance(node, ast.TypeVar | ast.ParamSpec | ast.TypeVarTuple):
        return node.name
    raise TypeError(f"unsupported type parameter node: {type(node).__name__}")


def _ast_type_parameter_kind(node: ast.AST) -> TypeParameterKind:
    if isinstance(node, ast.TypeVar):
        return "type_var"
    if isinstance(node, ast.ParamSpec):
        return "param_spec"
    if isinstance(node, ast.TypeVarTuple):
        return "type_var_tuple"
    raise TypeError(f"unsupported type parameter node: {type(node).__name__}")


def _type_parameter_kind(factory_name: str) -> TypeParameterKind:
    if factory_name == "TypeVar":
        return "type_var"
    if factory_name == "ParamSpec":
        return "param_spec"
    if factory_name == "TypeVarTuple":
        return "type_var_tuple"
    raise ValueError(f"unsupported type parameter factory: {factory_name}")


def _annotation_source(node: ast.expr | None) -> str | None:
    return ast.unparse(node) if node is not None else None


def _annotation_names(annotations: tuple[ast.expr | None, ...]) -> tuple[str, ...]:
    names: list[str] = []
    for annotation in annotations:
        if annotation is None:
            continue
        for child in _expanded_annotation_nodes(annotation):
            name = _annotation_node_name(child)
            if name is not None:
                names.append(name)
    return _unique(names)


def _has_any_annotation(
    annotations: tuple[ast.expr | None, ...],
    typing_aliases: _TypingAliases,
) -> bool:
    return any(
        _annotation_contains_any(annotation, typing_aliases)
        for annotation in annotations
        if annotation is not None
    )


def _any_annotation_sources(
    annotations: tuple[ast.expr | None, ...],
    typing_aliases: _TypingAliases,
) -> tuple[str, ...]:
    return tuple(
        ast.unparse(annotation)
        for annotation in annotations
        if annotation is not None and _annotation_contains_any(annotation, typing_aliases)
    )


def _annotation_contains_any(annotation: ast.expr, typing_aliases: _TypingAliases) -> bool:
    return any(
        _is_any_annotation_node(node, typing_aliases)
        for node in _expanded_annotation_nodes(annotation)
    )


def _is_any_annotation_node(node: ast.AST, typing_aliases: _TypingAliases) -> bool:
    qualified_any_path_length = 2
    if isinstance(node, ast.Name):
        return node.id in typing_aliases.any_names
    path = _attribute_path(node)
    return (
        path is not None
        and len(path) == qualified_any_path_length
        and path[0] in typing_aliases.typing_module_names
        and path[1] == "Any"
    )


def _expanded_annotation_nodes(annotation: ast.expr) -> tuple[ast.AST, ...]:
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            parsed = ast.parse(annotation.value, mode="eval")
        except SyntaxError:
            return (annotation,)
        return tuple(ast.walk(parsed.body))
    return tuple(ast.walk(annotation))


def _annotation_node_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        path = _attribute_path(node)
        if path is not None:
            return ".".join(path)
    if isinstance(node, ast.Constant) and node.value is None:
        return "None"
    return None


def _base_names(node: ast.ClassDef) -> tuple[str, ...]:
    return tuple(ast.unparse(base) for base in node.bases)


def _attribute_path(node: ast.AST) -> tuple[str, ...] | None:
    if isinstance(node, ast.Name):
        return (node.id,)
    if not isinstance(node, ast.Attribute):
        return None
    parts: list[str] = [node.attr]
    current = node.value
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return tuple(reversed(parts))


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _source_text(node: ast.stmt, lines: list[str]) -> str:
    return "\n".join(lines[node.lineno - 1 : _end_lineno(node)])


def _end_lineno(node: ast.stmt | ast.expr) -> int:
    return node.end_lineno if node.end_lineno is not None else node.lineno


def _visibility(qualname: str) -> Visibility:
    name = qualname.rsplit(".", maxsplit=1)[-1]
    return "private" if name.startswith("_") else "public"


def _is_import_node(node: ast.stmt) -> bool:
    return isinstance(node, ast.Import | ast.ImportFrom)


def _is_top_level_executable_statement(node: ast.stmt) -> bool:
    return not isinstance(
        node,
        ast.Import
        | ast.ImportFrom
        | ast.FunctionDef
        | ast.AsyncFunctionDef
        | ast.ClassDef
        | ast.Assign
        | ast.AnnAssign,
    )


class _FunctionNameCollector(ast.NodeVisitor):
    """Collect local, referenced, called, and runtime-only names for a function.

    Annotation references are tracked separately from runtime references so type
    hints do not create false global dependencies for sidecar extraction.
    Nested functions and classes are treated as local bindings rather than
    traversed bodies because Atoll V1 does not extract nested symbols.
    """

    def __init__(self) -> None:
        """Initialize independent name sets for one function definition."""
        self.called_names: set[str] = set()
        self.called_paths: set[str] = set()
        self.local_names: set[str] = set()
        self.referenced_names: set[str] = set()
        self.runtime_referenced_names: set[str] = set()
        self._in_annotation = False

    def collect(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Collect names from a function without visiting its decorator wrapper.

        Args:
            node: Syntax node being visited without executing target code.
        """
        self._collect_arguments(node)
        self._collect_defaults(node)
        self._collect_decorators(node)
        self._collect_return_annotation(node)
        for child in node.body:
            self.visit(child)

    def _collect_arguments(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for argument in _function_args(node):
            self.local_names.add(argument.arg)
            if argument.annotation is not None:
                self._visit_annotation(argument.annotation)
        if node.args.vararg is not None:
            self.local_names.add(node.args.vararg.arg)
            if node.args.vararg.annotation is not None:
                self._visit_annotation(node.args.vararg.annotation)
        if node.args.kwarg is not None:
            self.local_names.add(node.args.kwarg.arg)
            if node.args.kwarg.annotation is not None:
                self._visit_annotation(node.args.kwarg.annotation)

    def _collect_defaults(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)

    def _collect_decorators(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)

    def _collect_return_annotation(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if node.returns is not None:
            self._visit_annotation(node.returns)

    def visit_arg(self, node: ast.arg) -> None:
        """Record argument names as local bindings while preserving annotations.

        Args:
            node: Syntax node being visited without executing target code.
        """
        self.local_names.add(node.arg)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Classify loaded names as references and stored names as locals.

        Args:
            node: Syntax node being visited without executing target code.
        """
        if isinstance(node.ctx, ast.Load):
            self.referenced_names.add(node.id)
            if not self._in_annotation:
                self.runtime_referenced_names.add(node.id)
        elif isinstance(node.ctx, ast.Store | ast.Del):
            self.local_names.add(node.id)

    def visit_Call(self, node: ast.Call) -> None:
        """Record directly named calls for conservative same-module edges.

        Args:
            node: Syntax node being visited without executing target code.
        """
        if isinstance(node.func, ast.Name):
            self.called_names.add(node.func.id)
        path = _attribute_path(node.func)
        if path is not None:
            self.called_paths.add(".".join(path))
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Treat nested functions as local bindings and do not traverse them.

        Args:
            node: Syntax node being visited without executing target code.
        """
        self.local_names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Treat nested async functions as local bindings and do not traverse them.

        Args:
            node: Syntax node being visited without executing target code.
        """
        self.local_names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Treat nested classes as local bindings and do not traverse them.

        Args:
            node: Syntax node being visited without executing target code.
        """
        self.local_names.add(node.name)

    def _visit_annotation(self, node: ast.expr) -> None:
        was_in_annotation = self._in_annotation
        self._in_annotation = True
        try:
            self.visit(node)
        finally:
            self._in_annotation = was_in_annotation


def _function_arguments(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[ast.arg, ...]:
    return (
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
        *((node.args.vararg,) if node.args.vararg is not None else ()),
        *((node.args.kwarg,) if node.args.kwarg is not None else ()),
    )


def _function_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[ast.arg, ...]:
    return (
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
    )


class _YieldShapeVisitor(ast.NodeVisitor):
    """Detect yields in one symbol body without traversing nested symbols."""

    def __init__(self) -> None:
        """Initialize the yield flag for a single function scan."""
        self.contains_yield = False

    def visit_Yield(self, node: ast.Yield) -> None:
        """Mark a sync or async generator yield expression.

        Args:
            node: Syntax node being visited without executing target code.
        """
        del node
        self.contains_yield = True

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        """Mark a generator delegation expression.

        Args:
            node: Syntax node being visited without executing target code.
        """
        del node
        self.contains_yield = True

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Skip nested functions when detecting the outer execution shape.

        Args:
            node: Syntax node being visited without executing target code.
        """

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Skip nested async functions when detecting the outer execution shape.

        Args:
            node: Syntax node being visited without executing target code.
        """

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Skip nested class bodies when detecting the outer execution shape.

        Args:
            node: Syntax node being visited without executing target code.
        """


class _ClassNameCollector(ast.NodeVisitor):
    """Collect names referenced by class decorators, bases, and keyword bases.

    The collector does not visit the class body because method bodies are scanned
    as separate symbols. This keeps class-level dependency facts limited to the
    inheritance and decorator boundary.
    """

    def __init__(self) -> None:
        """Initialize the local and referenced name sets for one class."""
        self.local_names: set[str] = set()
        self.referenced_names: set[str] = set()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit only the class header and decorator expressions.

        Args:
            node: Syntax node being visited without executing target code.
        """
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Name(self, node: ast.Name) -> None:
        """Record names loaded by class headers as references.

        Args:
            node: Syntax node being visited without executing target code.
        """
        if isinstance(node.ctx, ast.Load):
            self.referenced_names.add(node.id)
