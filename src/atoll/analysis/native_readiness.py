"""Score generated sidecars for direct native-compilation suitability.

The analyzer inspects generated sidecar source without importing or executing
it. It deliberately focuses on facts that are cheap and deterministic from the
AST: fully native annotations, absence of dynamic top-level ``getattr`` aliases,
and enough repeated primitive work to justify native compilation. It does not
prove semantic equivalence or replace runtime verification.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Final

_NATIVE_ANNOTATION_NAMES: Final = frozenset(
    {
        "bool",
        "bytes",
        "dict",
        "float",
        "frozenset",
        "int",
        "list",
        "None",
        "NoneType",
        "set",
        "str",
        "tuple",
    }
)
_ANY_NAMES: Final = frozenset({"Any", "typing.Any"})
_MIN_NATIVE_OPERATION_SIGNAL: Final = 4


@dataclass(frozen=True, slots=True)
class NativeReadiness:
    """Summarize whether a generated sidecar is worth native compilation.

    Instances are immutable value objects safe to compare in tests and pass
    across command/report boundaries. Tuple fields contain stable, sorted names
    where ordering is not already determined by source order. ``score`` is a
    bounded 0-100 heuristic: hard blockers reduce it, while eligibility requires
    no blockers and a repeated/native work signal.

    Attributes:
        source_module: Importable source module name.
        symbol: Exported source symbol represented by the generated sidecar.
        eligible: Whether generated code passes the native-readiness gate.
        score: Scan-only extraction-safety or native-readiness score.
        function_count: Number of generated functions inspected.
        any_typed_functions: Generated functions whose annotations contain `Any`.
        boxed_typed_functions: Generated functions whose useful values remain boxed Python objects.
        dynamic_dependencies: Generated dependencies requiring dynamic runtime lookup.
        loop_count: Number of loops found in generated functions.
        native_operation_count: Count of operations likely to benefit from native lowering.
        reasons: Deterministically ordered evidence supporting the decision.
    """

    source_module: str
    symbol: str
    eligible: bool
    score: int
    function_count: int
    any_typed_functions: tuple[str, ...]
    boxed_typed_functions: tuple[str, ...]
    dynamic_dependencies: tuple[str, ...]
    loop_count: int
    native_operation_count: int
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ReasonFacts:
    """Internal rejection facts used to build stable, concise reason strings.

    Attributes:
        exported_symbol: Public source symbol represented by generated code.
        function_names: Generated function names in source order.
        missing_annotations: Generated functions with incomplete annotations.
        any_typed: Generated functions whose annotations contain `Any`.
        boxed_typed: Generated functions whose useful values remain boxed.
        dynamic_dependencies: Generated dependencies requiring runtime lookup.
        loop_count: Number of loops found in generated source.
        native_operation_count: Operations likely to benefit from native lowering.
    """

    exported_symbol: str
    function_names: tuple[str, ...]
    missing_annotations: tuple[str, ...]
    any_typed: tuple[str, ...]
    boxed_typed: tuple[str, ...]
    dynamic_dependencies: tuple[str, ...]
    loop_count: int
    native_operation_count: int


def analyze_native_readiness(
    source_module: str,
    exported_symbol: str,
    generated_source: str,
) -> NativeReadiness:
    """Evaluate generated sidecar source for native-compilation readiness.

    The generated source is parsed as Python and every top-level generated
    function is evaluated as part of the sidecar surface. Missing annotations,
    ``Any`` annotations, non-builtin boxed annotations, dynamic ``getattr``
    aliases used at runtime, and trivial bodies without repeated/native work are
    reported as concise rejection reasons.

    Args:
        source_module: Importable source module name.
        exported_symbol: Public source symbol represented by the generated code.
        generated_source: Generated Python source evaluated for native readiness.

    Returns:
        NativeReadiness: Scored native-readiness evidence for the generated symbol.
    """
    module = ast.parse(generated_source)
    functions = tuple(
        node for node in module.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    )
    function_names = tuple(function.name for function in functions)
    missing_annotations = tuple(
        function.name for function in functions if _has_missing_annotation(function)
    )
    any_typed = tuple(function.name for function in functions if _has_any_annotation(function))
    boxed_typed = tuple(function.name for function in functions if _has_boxed_annotation(function))
    dynamic_dependencies = _dynamic_dependencies(module, functions)
    work_counter = _WorkCounter()
    for function in functions:
        work_counter.visit_function_body(function)

    reasons = _reasons(
        _ReasonFacts(
            exported_symbol=exported_symbol,
            function_names=function_names,
            missing_annotations=missing_annotations,
            any_typed=any_typed,
            boxed_typed=boxed_typed,
            dynamic_dependencies=dynamic_dependencies,
            loop_count=work_counter.loop_count,
            native_operation_count=work_counter.native_operation_count,
        )
    )
    eligible = not reasons
    return NativeReadiness(
        source_module=source_module,
        symbol=exported_symbol,
        eligible=eligible,
        score=_score(reasons),
        function_count=len(functions),
        any_typed_functions=any_typed,
        boxed_typed_functions=boxed_typed,
        dynamic_dependencies=dynamic_dependencies,
        loop_count=work_counter.loop_count,
        native_operation_count=work_counter.native_operation_count,
        reasons=reasons,
    )


def _reasons(facts: _ReasonFacts) -> tuple[str, ...]:
    reasons: list[str] = []
    if facts.exported_symbol not in facts.function_names:
        reasons.append(f"exported symbol not generated: {facts.exported_symbol}")
    if facts.missing_annotations:
        reasons.append(f"missing annotations: {', '.join(facts.missing_annotations)}")
    if facts.any_typed:
        reasons.append(f"Any annotations: {', '.join(facts.any_typed)}")
    if facts.boxed_typed:
        reasons.append(f"boxed annotations: {', '.join(facts.boxed_typed)}")
    if facts.dynamic_dependencies:
        reasons.append(f"dynamic getattr dependencies: {', '.join(facts.dynamic_dependencies)}")
    if facts.loop_count == 0 and facts.native_operation_count < _MIN_NATIVE_OPERATION_SIGNAL:
        reasons.append("no repeated/native work signal")
    return tuple(reasons)


def _score(reasons: tuple[str, ...]) -> int:
    score = 100
    for reason in reasons:
        if reason.startswith("no repeated/native work"):
            score -= 30
        elif reason.startswith("exported symbol"):
            score -= 50
        else:
            score -= 35
    return max(0, score)


def _has_missing_annotation(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    if node.returns is None:
        return True
    return any(argument.annotation is None for argument in _function_arguments(node))


def _has_any_annotation(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        _annotation_contains_any(annotation)
        for annotation in _function_annotations(node)
        if annotation is not None
    )


def _has_boxed_annotation(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        _annotation_contains_boxed_name(annotation)
        for annotation in _function_annotations(node)
        if annotation is not None
    )


def _function_arguments(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[ast.arg, ...]:
    arguments = node.args
    return (
        *arguments.posonlyargs,
        *arguments.args,
        *arguments.kwonlyargs,
        *((arguments.vararg,) if arguments.vararg is not None else ()),
        *((arguments.kwarg,) if arguments.kwarg is not None else ()),
    )


def _function_annotations(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.expr | None, ...]:
    return (
        *(argument.annotation for argument in _function_arguments(node)),
        node.returns,
    )


def _annotation_contains_any(annotation: ast.expr) -> bool:
    return any(
        _annotation_name(node) in _ANY_NAMES for node in _expanded_annotation_nodes(annotation)
    )


def _annotation_contains_boxed_name(annotation: ast.expr) -> bool:
    names = tuple(
        name
        for node in _expanded_annotation_nodes(annotation)
        if (name := _annotation_name(node)) is not None
    )
    return any(
        name not in _ANY_NAMES
        and name not in _NATIVE_ANNOTATION_NAMES
        and not any(other.startswith(f"{name}.") for other in names)
        for name in names
    )


def _expanded_annotation_nodes(annotation: ast.expr) -> tuple[ast.AST, ...]:
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            parsed = ast.parse(annotation.value, mode="eval")
        except SyntaxError:
            return (annotation,)
        return tuple(ast.walk(parsed.body))
    return tuple(ast.walk(annotation))


def _annotation_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        path = _attribute_path(node)
        if path is not None:
            return ".".join(path)
    if isinstance(node, ast.Constant) and node.value is None:
        return "None"
    return None


def _attribute_path(node: ast.Attribute) -> tuple[str, ...] | None:
    parts: list[str] = [node.attr]
    current = node.value
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return tuple(reversed(parts))


def _dynamic_dependencies(
    module: ast.Module,
    functions: tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...],
) -> tuple[str, ...]:
    dynamic_bindings = _top_level_getattr_bindings(module)
    if not dynamic_bindings:
        return ()
    referenced = set[str]()
    for function in functions:
        visitor = _RuntimeNameCollector()
        visitor.visit_function_body(function)
        referenced.update(visitor.names)
    return tuple(name for name in sorted(dynamic_bindings) if name in referenced)


def _top_level_getattr_bindings(module: ast.Module) -> set[str]:
    bindings = set[str]()
    for node in module.body:
        if isinstance(node, ast.Assign) and _is_getattr_call(node.value):
            bindings.update(_target_names(node.targets))
        elif (
            isinstance(node, ast.AnnAssign)
            and node.value is not None
            and _is_getattr_call(node.value)
        ):
            bindings.update(_target_names((node.target,)))
    return bindings


def _is_getattr_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr"
    )


def _target_names(targets: tuple[ast.expr, ...] | list[ast.expr]) -> set[str]:
    names = set[str]()
    for target in targets:
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Tuple | ast.List):
            names.update(_target_names(list(target.elts)))
    return names


class _RuntimeNameCollector(ast.NodeVisitor):
    """Collect runtime name loads from a function body without nested scopes."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_function_body(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Visit executable statements for one generated function.

        Args:
            node: Syntax node being visited without executing target code.
        """
        for statement in node.body:
            self.visit(statement)

    def visit_Name(self, node: ast.Name) -> None:
        """Record loaded names that may bind to top-level dynamic aliases.

        Args:
            node: Syntax node being visited without executing target code.
        """
        if isinstance(node.ctx, ast.Load):
            self.names.add(node.id)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Skip nested function scopes when collecting parent dependencies.

        Args:
            node: Syntax node being visited without executing target code.
        """

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Skip nested async function scopes when collecting parent dependencies.

        Args:
            node: Syntax node being visited without executing target code.
        """

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Skip nested class scopes when collecting parent dependencies.

        Args:
            node: Syntax node being visited without executing target code.
        """


class _WorkCounter(ast.NodeVisitor):
    """Count loop/comprehension and primitive operation signal in function bodies."""

    def __init__(self) -> None:
        self.loop_count = 0
        self.native_operation_count = 0

    def visit_function_body(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Visit executable statements for one generated function.

        Args:
            node: Syntax node being visited without executing target code.
        """
        for statement in node.body:
            self.visit(statement)

    def visit_For(self, node: ast.For) -> None:
        """Count explicit loops as repeated work.

        Args:
            node: Syntax node being visited without executing target code.
        """
        self.loop_count += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        """Count async loops as repeated work.

        Args:
            node: Syntax node being visited without executing target code.
        """
        self.loop_count += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        """Count while loops as repeated work.

        Args:
            node: Syntax node being visited without executing target code.
        """
        self.loop_count += 1
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        """Count comprehensions as repeated work.

        Args:
            node: Syntax node being visited without executing target code.
        """
        self._visit_comprehension(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        """Count comprehensions as repeated work.

        Args:
            node: Syntax node being visited without executing target code.
        """
        self._visit_comprehension(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        """Count comprehensions as repeated work.

        Args:
            node: Syntax node being visited without executing target code.
        """
        self._visit_comprehension(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        """Count generator expressions as repeated work.

        Args:
            node: Syntax node being visited without executing target code.
        """
        self._visit_comprehension(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        """Count arithmetic operations as native work signal.

        Args:
            node: Syntax node being visited without executing target code.
        """
        self.native_operation_count += 1
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        """Count each comparison operator as native work signal.

        Args:
            node: Syntax node being visited without executing target code.
        """
        self.native_operation_count += len(node.ops)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Count indexed access as native work signal.

        Args:
            node: Syntax node being visited without executing target code.
        """
        self.native_operation_count += 1
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Skip nested function scopes when counting parent work.

        Args:
            node: Syntax node being visited without executing target code.
        """

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Skip nested async function scopes when counting parent work.

        Args:
            node: Syntax node being visited without executing target code.
        """

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Skip nested class scopes when counting parent work.

        Args:
            node: Syntax node being visited without executing target code.
        """

    def _visit_comprehension(
        self,
        node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
    ) -> None:
        self.loop_count += len(node.generators)
        self.generic_visit(node)
