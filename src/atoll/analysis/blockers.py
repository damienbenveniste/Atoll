"""Detect dynamic Python patterns that block safe sidecar extraction.

Atoll relies on mypyc-generated sidecars, so runtime reflection, frame access,
dynamic imports, and module/class monkey-patching are treated conservatively.
The blockers produced here explain why a symbol is excluded before build-time
or runtime verification has a chance to fail less clearly.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass

from atoll.models import Blocker, BlockerSeverity, SymbolId

_HARDCODED_CALL_BLOCKERS = {
    "eval": ("DYN_EVAL", "eval() prevents safe extraction"),
    "exec": ("DYN_EXEC", "exec() prevents safe extraction"),
    "globals": ("DYN_GLOBALS", "globals() prevents safe extraction"),
    "locals": ("DYN_LOCALS", "locals() prevents safe extraction"),
    "vars": ("DYN_LOCALS", "vars() prevents safe extraction"),
    "__import__": ("DYN_IMPORT_CALL", "__import__() prevents safe extraction"),
}
_FRAME_CALL_BLOCKERS: dict[tuple[str, ...], str] = {
    ("inspect", "currentframe"): "inspect.currentframe() depends on Python frame semantics",
    ("inspect", "stack"): "inspect.stack() depends on Python frame semantics",
    ("inspect", "getouterframes"): "inspect.getouterframes() depends on Python frame semantics",
    ("inspect", "getinnerframes"): "inspect.getinnerframes() depends on Python frame semantics",
    ("sys", "_getframe"): "sys._getframe() depends on Python frame semantics",
}
_BARE_FRAME_CALL_BLOCKERS = {
    "currentframe": "currentframe() depends on Python frame semantics",
    "_getframe": "_getframe() depends on Python frame semantics",
    "getouterframes": "getouterframes() depends on Python frame semantics",
    "getinnerframes": "getinnerframes() depends on Python frame semantics",
}
_FRAME_ATTRIBUTE_BLOCKERS = frozenset(
    {
        "f_back",
        "f_builtins",
        "f_code",
        "f_globals",
        "f_lasti",
        "f_lineno",
        "f_locals",
        "f_trace",
        "f_trace_lines",
        "f_trace_opcodes",
    }
)
_GETATTR_MIN_ARGS = 2
_TYPING_MODULES = frozenset({"typing", "typing_extensions"})
_MYPYC_UNSUPPORTED_TYPEVAR_KEYWORDS = frozenset({"default", "infer_variance"})
_MODULE_RUNTIME_HOOKS = frozenset({"__dir__", "__getattr__"})


def detect_function_blockers(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    symbol: SymbolId,
) -> tuple[Blocker, ...]:
    """Return blockers found inside a function or method definition.

    The visitor checks executable body statements for dynamic behavior, then
    separately adds annotation, decorator, and async-function blockers. Nested
    definitions are reported as unsupported instead of traversed as candidates.

    Args:
        node: Syntax node inspected without executing target-project code.
        symbol: Stable symbol identity associated with detected blockers.

    Returns:
        tuple[Blocker, ...]: Conservative blockers attached to the function symbol.
    """
    visitor = _BlockerVisitor(symbol)
    for child in node.body:
        visitor.visit(child)
    blockers = [*visitor.blockers]
    blockers.extend(_annotation_blockers(node, symbol))
    blockers.extend(_decorator_blockers(node.decorator_list, symbol))
    if "." not in symbol.qualname and node.name in _MODULE_RUNTIME_HOOKS:
        blockers.append(
            Blocker(
                severity="hard",
                code="DYN_MODULE_HOOK",
                message=f"module runtime hook {node.name} must remain interpreted",
                lineno=node.lineno,
                symbol=symbol,
            )
        )
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
    """Return conservative blockers for a top-level class definition.

    Atoll V1 records classes for dependency analysis, but extraction focuses on
    functions. Dynamic class construction such as metaclasses or custom
    attribute hooks is therefore reported early as a hard boundary.

    Args:
        node: Syntax node inspected without executing target-project code.
        symbol: Stable symbol identity associated with detected blockers.

    Returns:
        tuple[Blocker, ...]: Conservative blockers attached to the class symbol.
    """
    return (
        *_decorator_blockers(node.decorator_list, symbol),
        *_metaclass_blockers(node, symbol),
        *_dynamic_class_method_blockers(node, symbol),
    )


def module_level_blockers(nodes: Iterable[ast.stmt], module: str) -> tuple[Blocker, ...]:
    """Detect module-level statements that make native extraction unsafe.

    The detector flags top-level attribute assignment as monkey-patching and
    records mypyc-incompatible `TypeVar` keyword usage before compilation. It
    intentionally avoids executing imports or evaluating constants.

    Args:
        nodes: Top-level syntax statements inspected for module blockers.
        module: Module scan or module identity being analyzed.

    Returns:
        tuple[Blocker, ...]: Module blockers that cannot be assigned to one symbol.
    """
    node_tuple = tuple(nodes)
    monkey_patch_blockers = tuple(
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
    return (*monkey_patch_blockers, *_module_typevar_blockers(node_tuple, module))


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
    """Collect body-level blockers for one function-like symbol."""

    def __init__(self, symbol: SymbolId) -> None:
        """Bind all emitted blockers to the source symbol being inspected.

        Args:
            symbol: Stable source symbol identity.
        """
        self.symbol = symbol
        self.blockers: list[Blocker] = []

    def visit_Call(self, node: ast.Call) -> None:
        """Record blockers caused by dynamic calls before visiting arguments.

        Args:
            node: Syntax node being visited without executing target code.
        """
        self._record_call_blocker(node)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Flag direct access to frame attributes under compiled execution.

        Args:
            node: Syntax node being visited without executing target code.
        """
        if node.attr in _FRAME_ATTRIBUTE_BLOCKERS:
            self._append(
                "hard",
                "FRAME_ATTRIBUTE_INTROSPECTION",
                f"frame attribute {node.attr!r} changes under compiled execution",
                node.lineno,
            )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Report nested functions without descending into their bodies.

        Args:
            node: Syntax node being visited without executing target code.
        """
        self._record_nested_symbol(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Report nested async functions as unsupported extraction boundaries.

        Args:
            node: Syntax node being visited without executing target code.
        """
        self._record_nested_symbol(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Report nested classes without scanning their class bodies.

        Args:
            node: Syntax node being visited without executing target code.
        """
        self._record_nested_symbol(node)

    def _record_call_blocker(self, node: ast.Call) -> None:
        function_name = _call_name(node.func)
        if function_name in _HARDCODED_CALL_BLOCKERS:
            code, message = _HARDCODED_CALL_BLOCKERS[function_name]
            self._append("hard", code, message, node.lineno)
        elif (call_path := _call_path(node.func)) in _FRAME_CALL_BLOCKERS:
            self._append(
                "hard",
                "FRAME_INTROSPECTION",
                _FRAME_CALL_BLOCKERS[call_path],
                node.lineno,
            )
        elif isinstance(node.func, ast.Name) and function_name in _BARE_FRAME_CALL_BLOCKERS:
            self._append(
                "hard",
                "FRAME_INTROSPECTION",
                _BARE_FRAME_CALL_BLOCKERS[function_name],
                node.lineno,
            )
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


def _module_typevar_blockers(nodes: Iterable[ast.stmt], module: str) -> tuple[Blocker, ...]:
    aliases = _typing_aliases(nodes)
    if not aliases.typevar_names and not aliases.typing_module_names:
        return ()
    visitor = _ModuleTypeVarVisitor(module, aliases)
    for node in nodes:
        visitor.visit(node)
    return tuple(visitor.blockers)


@dataclass(frozen=True, slots=True)
class _TypingAliases:
    typevar_names: frozenset[str]
    typing_module_names: frozenset[str]


def _typing_aliases(nodes: Iterable[ast.stmt]) -> _TypingAliases:
    typevar_names: set[str] = set()
    typing_module_names: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _TYPING_MODULES:
                    typing_module_names.add(alias.asname or alias.name)
        elif (
            isinstance(node, ast.ImportFrom) and node.module in _TYPING_MODULES and node.level == 0
        ):
            for alias in node.names:
                if alias.name == "TypeVar":
                    typevar_names.add(alias.asname or alias.name)
    return _TypingAliases(
        typevar_names=frozenset(typevar_names),
        typing_module_names=frozenset(typing_module_names),
    )


class _ModuleTypeVarVisitor(ast.NodeVisitor):
    """Find module-level TypeVar calls that mypyc rejects.

    Function bodies are skipped because this visitor is concerned with module
    setup expressions that can break compilation before sidecar code is reached.
    """

    def __init__(self, module: str, aliases: _TypingAliases) -> None:
        """Initialize a module-scoped visitor with known typing aliases.

        Args:
            module: Scanned module or syntax module being processed.
            aliases: Typing aliases visible in the current module.
        """
        self.module = module
        self.aliases = aliases
        self.blockers: list[Blocker] = []

    def visit_Call(self, node: ast.Call) -> None:
        """Record unsupported TypeVar keyword arguments at module scope.

        Args:
            node: Syntax node being visited without executing target code.
        """
        unsupported = _unsupported_typevar_keywords(node, self.aliases)
        if unsupported:
            keywords = ", ".join(unsupported)
            self.blockers.append(
                Blocker(
                    severity="hard",
                    code="MYPYC_UNSUPPORTED_TYPEVAR",
                    message=f"TypeVar keyword(s) {keywords} are rejected by mypyc",
                    lineno=node.lineno,
                    symbol=SymbolId(module=self.module, qualname="<module>"),
                )
            )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Skip function bodies while scanning module-level TypeVar declarations.

        Args:
            node: Syntax node being visited without executing target code.
        """
        _ = node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Skip async function bodies while scanning module-level declarations.

        Args:
            node: Syntax node being visited without executing target code.
        """
        _ = node


def _unsupported_typevar_keywords(node: ast.Call, aliases: _TypingAliases) -> tuple[str, ...]:
    if not _is_typevar_call(node.func, aliases):
        return ()
    unsupported = [
        keyword.arg
        for keyword in node.keywords
        if keyword.arg is not None and keyword.arg in _MYPYC_UNSUPPORTED_TYPEVAR_KEYWORDS
    ]
    return tuple(sorted(unsupported))


def _is_typevar_call(node: ast.AST, aliases: _TypingAliases) -> bool:
    if isinstance(node, ast.Name):
        return node.id in aliases.typevar_names
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "TypeVar"
        and isinstance(node.value, ast.Name)
        and node.value.id in aliases.typing_module_names
    )


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


def _call_path(node: ast.AST) -> tuple[str, ...]:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return tuple(reversed(parts))
    return ()


def _is_importlib_import_module(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "import_module"
        and isinstance(node.value, ast.Name)
        and node.value.id == "importlib"
    )


def _is_literal_string(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)
