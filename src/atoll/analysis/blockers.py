"""Detection of dynamic Python patterns that block safe sidecar extraction."""

from __future__ import annotations

import ast
from collections.abc import Iterable

from atoll.models import Blocker, BlockerSeverity, SymbolId

_HARDCODED_CALL_BLOCKERS = {
    "eval": ("DYN_EVAL", "eval() prevents safe extraction"),
    "exec": ("DYN_EXEC", "exec() prevents safe extraction"),
    "globals": ("DYN_GLOBALS", "globals() prevents safe extraction"),
    "locals": ("DYN_LOCALS", "locals() prevents safe extraction"),
    "vars": ("DYN_LOCALS", "vars() prevents safe extraction"),
    "__import__": ("DYN_IMPORT_CALL", "__import__() prevents safe extraction"),
}
_GETATTR_MIN_ARGS = 2


def detect_function_blockers(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    symbol: SymbolId,
) -> tuple[Blocker, ...]:
    """Return blockers found inside a function or method definition."""
    visitor = _BlockerVisitor(symbol)
    for child in node.body:
        visitor.visit(child)
    blockers = [*visitor.blockers]
    blockers.extend(_annotation_blockers(node, symbol))
    blockers.extend(_decorator_blockers(node.decorator_list, symbol))
    if isinstance(node, ast.AsyncFunctionDef):
        blockers.append(
            Blocker(
                severity="soft",
                code="ASYNC_FUNCTION",
                message="async functions are experimental for Atoll V1",
                lineno=node.lineno,
                symbol=symbol,
            )
        )
    return tuple(blockers)


def detect_class_blockers(node: ast.ClassDef, symbol: SymbolId) -> tuple[Blocker, ...]:
    """Return conservative blockers for a top-level class definition."""
    return (
        *_decorator_blockers(node.decorator_list, symbol),
        *_metaclass_blockers(node, symbol),
        *_dynamic_class_method_blockers(node, symbol),
    )


def module_level_blockers(nodes: Iterable[ast.stmt], module: str) -> tuple[Blocker, ...]:
    """Detect obvious top-level monkey-patching statements."""
    return tuple(
        Blocker(
            severity="hard",
            code="DYN_MODULE_MONKEYPATCH",
            message="top-level attribute assignment can monkey-patch runtime state",
            lineno=node.lineno,
            symbol=SymbolId(module=module, qualname="<module>"),
        )
        for node in nodes
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Attribute) for target in node.targets)
    )


def _metaclass_blockers(node: ast.ClassDef, symbol: SymbolId) -> tuple[Blocker, ...]:
    return tuple(
        Blocker(
            severity="hard",
            code="DYN_CLASS_MONKEYPATCH",
            message="metaclasses are outside Atoll V1 class support",
            lineno=node.lineno,
            symbol=symbol,
        )
        for keyword in node.keywords
        if keyword.arg == "metaclass"
    )


def _dynamic_class_method_blockers(node: ast.ClassDef, symbol: SymbolId) -> tuple[Blocker, ...]:
    dynamic_methods = {"__getattr__", "__getattribute__", "__setattr__"}
    return tuple(
        Blocker(
            severity="hard",
            code="DYN_CLASS_MONKEYPATCH",
            message=f"{child.name} makes class extraction dynamic",
            lineno=child.lineno,
            symbol=symbol,
        )
        for child in node.body
        if isinstance(child, ast.FunctionDef) and child.name in dynamic_methods
    )


class _BlockerVisitor(ast.NodeVisitor):
    def __init__(self, symbol: SymbolId) -> None:
        self.symbol = symbol
        self.blockers: list[Blocker] = []

    def visit_Call(self, node: ast.Call) -> None:
        self._record_call_blocker(node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record_nested_symbol(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record_nested_symbol(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._record_nested_symbol(node)

    def _record_call_blocker(self, node: ast.Call) -> None:
        function_name = _call_name(node.func)
        if function_name in _HARDCODED_CALL_BLOCKERS:
            code, message = _HARDCODED_CALL_BLOCKERS[function_name]
            self._append("hard", code, message, node.lineno)
        elif function_name == "getattr":
            self._record_getattr(node)
        elif function_name == "setattr":
            self._append("hard", "DYN_SETATTR", "setattr() prevents safe extraction", node.lineno)
        elif function_name == "delattr":
            self._append("hard", "DYN_DELATTR", "delattr() prevents safe extraction", node.lineno)
        elif _is_importlib_import_module(node.func):
            self._append(
                "hard",
                "DYN_IMPORTLIB",
                "importlib.import_module() prevents safe extraction",
                node.lineno,
            )

    def _record_getattr(self, node: ast.Call) -> None:
        if len(node.args) >= _GETATTR_MIN_ARGS and _is_literal_string(node.args[1]):
            self._append(
                "soft",
                "DYN_GETATTR_LITERAL",
                "literal getattr() is inspectable but still dynamic",
                node.lineno,
            )
            return
        self._append(
            "hard",
            "DYN_GETATTR_DYNAMIC",
            "dynamic getattr() prevents safe extraction",
            node.lineno,
        )

    def _record_nested_symbol(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    ) -> None:
        self._append(
            "hard",
            "NESTED_SYMBOL",
            f"nested symbol {node.name!r} is outside Atoll V1 extraction",
            node.lineno,
        )

    def _append(self, severity: BlockerSeverity, code: str, message: str, lineno: int) -> None:
        blocker = Blocker(
            severity=severity,
            code=code,
            message=message,
            lineno=lineno,
            symbol=self.symbol,
        )
        self.blockers.append(blocker)


def _annotation_blockers(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    symbol: SymbolId,
) -> tuple[Blocker, ...]:
    missing_arg_annotation = any(argument.annotation is None for argument in _annotation_args(node))
    if not missing_arg_annotation and node.returns is not None:
        return ()
    return (
        Blocker(
            severity="soft",
            code="UNTYPED_DEF",
            message="function signature is not fully annotated",
            lineno=node.lineno,
            symbol=symbol,
        ),
    )


def _decorator_blockers(decorators: Iterable[ast.expr], symbol: SymbolId) -> tuple[Blocker, ...]:
    blockers: list[Blocker] = []
    for decorator in decorators:
        decorator_name = _call_name(decorator)
        if decorator_name not in {"dataclass", "staticmethod", "classmethod"}:
            blockers.append(
                Blocker(
                    severity="soft",
                    code="UNTYPED_DECORATOR",
                    message=f"decorator {decorator_name!r} may change runtime behavior",
                    lineno=decorator.lineno,
                    symbol=symbol,
                )
            )
    return tuple(blockers)


def _annotation_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[ast.arg, ...]:
    args = (
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
        *((node.args.vararg,) if node.args.vararg is not None else ()),
        *((node.args.kwarg,) if node.args.kwarg is not None else ()),
    )
    if "." in node.name:
        return args
    if args and args[0].arg in {"self", "cls"}:
        return args[1:]
    return args


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ast.unparse(node)


def _is_importlib_import_module(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "import_module"
        and isinstance(node.value, ast.Name)
        and node.value.id == "importlib"
    )


def _is_literal_string(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)
