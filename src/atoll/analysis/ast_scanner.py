"""AST scanner for Atoll's first-pass module analysis."""

from __future__ import annotations

import ast
import builtins

from atoll.analysis.blockers import (
    detect_class_blockers,
    detect_function_blockers,
    module_level_blockers,
)
from atoll.models import (
    ConstantKind,
    ConstantRecord,
    ImportRecord,
    ModuleId,
    ModuleScan,
    SymbolId,
    SymbolKind,
    SymbolRecord,
    Visibility,
)

_BUILTIN_NAMES = frozenset(dir(builtins))


def scan_module(module: ModuleId) -> ModuleScan:
    """Parse and scan one Python module."""
    source = module.path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(module.path), type_comments=True)
    lines = source.splitlines()
    imports = tuple(_import_record(node, lines) for node in tree.body if _is_import_node(node))
    constants = tuple(record for node in tree.body for record in _constant_records(node, lines))
    symbols = tuple(record for node in tree.body for record in _symbol_records(module.name, node))
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


def _symbol_records(module_name: str, node: ast.stmt) -> tuple[SymbolRecord, ...]:
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return (_function_record(module_name, node, "function", node.name),)
    if isinstance(node, ast.ClassDef):
        class_record = _class_record(module_name, node)
        methods = tuple(
            _function_record(module_name, child, "method", f"{node.name}.{child.name}")
            for child in node.body
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
        )
        return (class_record, *methods)
    return ()


def _function_record(
    module_name: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    kind: SymbolKind,
    qualname: str,
) -> SymbolRecord:
    symbol = SymbolId(module=module_name, qualname=qualname)
    collector = _FunctionNameCollector()
    collector.collect(node)
    arg_count, annotated_arg_count = _argument_counts(node)
    blockers = detect_function_blockers(node, symbol)
    referenced_names = tuple(sorted(collector.referenced_names))
    local_names = tuple(sorted(collector.local_names))
    return SymbolRecord(
        id=symbol,
        kind=kind,
        visibility=_visibility(symbol.qualname),
        lineno=node.lineno,
        end_lineno=_end_lineno(node),
        col_offset=node.col_offset,
        end_col_offset=node.end_col_offset,
        decorators=_decorators(node.decorator_list),
        arg_count=arg_count,
        annotated_arg_count=annotated_arg_count,
        has_return_annotation=node.returns is not None,
        has_any_annotation=annotated_arg_count > 0 or node.returns is not None,
        called_names=tuple(sorted(collector.called_names)),
        uses_globals=_global_names(referenced_names, local_names),
        local_names=local_names,
        referenced_names=referenced_names,
        blockers=blockers,
    )


def _class_record(module_name: str, node: ast.ClassDef) -> SymbolRecord:
    symbol = SymbolId(module=module_name, qualname=node.name)
    collector = _ClassNameCollector()
    collector.visit(node)
    blockers = detect_class_blockers(node, symbol)
    referenced_names = tuple(sorted(collector.referenced_names))
    local_names = tuple(sorted(collector.local_names))
    return SymbolRecord(
        id=symbol,
        kind="class",
        visibility=_visibility(symbol.qualname),
        lineno=node.lineno,
        end_lineno=_end_lineno(node),
        col_offset=node.col_offset,
        end_col_offset=node.end_col_offset,
        decorators=_decorators(node.decorator_list),
        arg_count=0,
        annotated_arg_count=0,
        has_return_annotation=False,
        has_any_annotation=False,
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
    args = (
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
        *((node.args.vararg,) if node.args.vararg is not None else ()),
        *((node.args.kwarg,) if node.args.kwarg is not None else ()),
    )
    return len(args), sum(argument.annotation is not None for argument in args)


def _decorators(decorators: list[ast.expr]) -> tuple[str, ...]:
    return tuple(ast.unparse(decorator) for decorator in decorators)


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
    def __init__(self) -> None:
        self.called_names: set[str] = set()
        self.local_names: set[str] = set()
        self.referenced_names: set[str] = set()

    def collect(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
            self.visit(argument)
        if node.args.vararg is not None:
            self.visit(node.args.vararg)
        if node.args.kwarg is not None:
            self.visit(node.args.kwarg)
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        for decorator in node.decorator_list:
            self.visit(decorator)
        if node.returns is not None:
            self.visit(node.returns)
        for child in node.body:
            self.visit(child)

    def visit_arg(self, node: ast.arg) -> None:
        self.local_names.add(node.arg)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.referenced_names.add(node.id)
        elif isinstance(node.ctx, ast.Store | ast.Del):
            self.local_names.add(node.id)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            self.called_names.add(node.func.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.local_names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.local_names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.local_names.add(node.name)


class _ClassNameCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.local_names: set[str] = set()
        self.referenced_names: set[str] = set()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.referenced_names.add(node.id)
