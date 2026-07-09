"""Generate pure-Python Atoll sidecar modules.

Sidecars copy selected top-level functions plus only the imports, literal
constants, and type-only references those functions need. The generated Python
is later compiled by mypyc, so this module also normalizes annotations and
returns to avoid known mypyc rejection paths.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path

from atoll.models import (
    ConstantRecord,
    DependencyEdge,
    EnabledIslandConfig,
    ImportRecord,
    ModuleScan,
    SidecarGeneration,
    SymbolId,
    SymbolRecord,
)

SIDECAR_GENERATOR_VERSION = "atoll-sidecar-v3"
_TYPING_IMPORT_MODULES = frozenset(
    {
        "collections",
        "collections.abc",
        "typing",
        "typing_extensions",
    }
)


@dataclass(frozen=True, slots=True)
class _TypingAliases:
    typevar_names: frozenset[str]
    typing_module_names: frozenset[str]


@dataclass(frozen=True, slots=True)
class SidecarPlan:
    """Concrete source fragments copied into a generated sidecar.

    Plans are deterministic products of an enriched module scan and the requested
    export list. They keep source fragments separate from rendering metadata so
    stale-file checks can hash the same evidence that is written to disk.
    """

    imports: tuple[ImportRecord, ...]
    extra_typing_names: tuple[str, ...]
    type_checking_names: tuple[str, ...]
    constants: tuple[ConstantRecord, ...]
    symbols: tuple[SymbolRecord, ...]
    included_symbol_names: tuple[str, ...]
    copied_sources: tuple[str, ...]


def generate_sidecar(module: ModuleScan, config: EnabledIslandConfig) -> SidecarGeneration:
    """Generate deterministic sidecar source for one enabled island.

    The source is not written here; callers decide whether they are checking,
    previewing, or applying changes. Unknown exported symbols raise `ValueError`
    during plan construction.
    """
    plan = build_sidecar_plan(module, config.symbols)
    source_hash = _source_hash(plan, config)
    source = _render_sidecar(config, plan, source_hash)
    return SidecarGeneration(
        config=config,
        included_symbols=plan.included_symbol_names,
        source_hash=source_hash,
        source_text=source,
    )


def write_sidecar(generation: SidecarGeneration) -> None:
    """Write generated sidecar source, creating the sidecar directory if needed."""
    generation.config.sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    generation.config.sidecar_path.write_text(generation.source_text, encoding="utf-8")


def build_sidecar_plan(
    module: ModuleScan,
    exported_symbols: tuple[str, ...],
) -> SidecarPlan:
    """Select imports, constants, copied symbols, and type-only names.

    The selected symbol set expands through direct same-module function calls so
    exported functions do not lose local helpers. Dynamic constants and erased
    annotation-only names are deliberately handled outside the runtime payload.
    """
    top_level_functions = {
        symbol.id.qualname: symbol for symbol in module.symbols if symbol.kind == "function"
    }
    _validate_exported_symbols(exported_symbols, top_level_functions)
    included_names = _expand_included_symbols(exported_symbols, module.dependency_edges)
    included_symbols = tuple(
        symbol
        for symbol in sorted(top_level_functions.values(), key=lambda item: item.lineno)
        if symbol.id.qualname in included_names
    )
    referenced_names = {name for symbol in included_symbols for name in symbol.referenced_names}
    raw_imports = tuple(
        record
        for record in module.imports
        if record.source_text == "from __future__ import annotations"
        or any(name in referenced_names for name in record.imported_names)
    )
    constants = tuple(
        constant
        for constant in module.constants
        if constant.name in referenced_names and constant.kind == "literal_constant"
    )
    source_lines = module.module.path.read_text(encoding="utf-8").splitlines()
    copied_sources, added_typing_names = _copied_sources(
        source_lines=source_lines,
        symbols=included_symbols,
        erased_type_arg_names=_erased_type_arg_names(raw_imports),
        erased_annotation_names=_erased_annotation_names(module.imports, module.constants)
        | _erased_type_arg_names(raw_imports),
    )
    runtime_names = _runtime_referenced_names(copied_sources, constants)
    imports = _runtime_imports(raw_imports, runtime_names)
    imported_names = {name for record in imports for name in record.imported_names}
    extra_typing_names = tuple(sorted(set(added_typing_names) - imported_names))
    type_checking_names = _type_checking_names(
        referenced_names=referenced_names,
        imported_names=imported_names,
        module_symbols=module.symbols,
        included_names=included_names,
    )
    return SidecarPlan(
        imports=imports,
        extra_typing_names=extra_typing_names,
        type_checking_names=type_checking_names,
        constants=constants,
        symbols=included_symbols,
        included_symbol_names=tuple(symbol.id.qualname for symbol in included_symbols),
        copied_sources=copied_sources,
    )


def default_sidecar_module(source_module: str) -> str:
    """Return Atoll's default unique sidecar module name for a source module."""
    normalized = "".join(character if character.isalnum() else "_" for character in source_module)
    sidecar_name = f"_atoll_{normalized}"
    package, _, _ = source_module.rpartition(".")
    return f"{package}.{sidecar_name}" if package else sidecar_name


def expected_sidecar_path(root: Path, sidecar_module: str) -> Path:
    """Return the generated sidecar source path under Atoll's private directory."""
    sidecar_name = sidecar_module.rsplit(".", maxsplit=1)[-1]
    return root.resolve() / ".atoll" / "sidecars" / f"{sidecar_name}.py"


def _validate_exported_symbols(
    exported_symbols: tuple[str, ...],
    top_level_functions: dict[str, SymbolRecord],
) -> None:
    unknown = tuple(symbol for symbol in exported_symbols if symbol not in top_level_functions)
    if unknown:
        raise ValueError(f"unknown top-level function symbols: {', '.join(unknown)}")


def _expand_included_symbols(
    exported_symbols: tuple[str, ...],
    edges: tuple[DependencyEdge, ...],
) -> set[str]:
    included = set(exported_symbols)
    changed = True
    while changed:
        changed = False
        for edge in edges:
            if edge.kind != "calls" or not isinstance(edge.dst, SymbolId):
                continue
            if edge.src.qualname in included and edge.dst.qualname not in included:
                included.add(edge.dst.qualname)
                changed = True
    return included


def _render_sidecar(
    config: EnabledIslandConfig,
    plan: SidecarPlan,
    source_hash: str,
) -> str:
    lines = [
        "# This file is generated by Atoll. Do not edit manually.",
        f"# Source module: {config.source_module}",
        f"# Exported symbols: {', '.join(config.symbols)}",
        f"# Source hash: {source_hash}",
        "",
    ]
    if not any(
        record.source_text == "from __future__ import annotations" for record in plan.imports
    ):
        lines.append("from __future__ import annotations")
    lines.extend(record.source_text for record in plan.imports)
    if plan.extra_typing_names:
        lines.append(f"from typing import {', '.join(plan.extra_typing_names)}")
    if plan.type_checking_names:
        if not any("TYPE_CHECKING" in record.imported_names for record in plan.imports):
            lines.append("from typing import TYPE_CHECKING")
        names = ", ".join(plan.type_checking_names)
        lines.extend(["", "if TYPE_CHECKING:", f"    from {config.source_module} import {names}"])
    if plan.imports or lines[-1] != "":
        lines.append("")
    lines.extend(constant.source_text for constant in plan.constants)
    if plan.constants:
        lines.append("")
    for copied_source in plan.copied_sources:
        lines.extend(copied_source.splitlines())
        lines.append("")
    lines.extend(
        [
            "__atoll_metadata__ = {",
            f'    "source_module": "{config.source_module}",',
            f'    "sidecar_module": "{config.sidecar_module}",',
            f'    "exported_symbols": {tuple(config.symbols)!r},',
            f'    "included_symbols": {plan.included_symbol_names!r},',
            f'    "source_hash": "{source_hash}",',
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def _source_hash(plan: SidecarPlan, config: EnabledIslandConfig) -> str:
    digest = hashlib.sha256()
    digest.update(SIDECAR_GENERATOR_VERSION.encode())
    digest.update(config.source_module.encode())
    digest.update(config.sidecar_module.encode())
    for record in plan.imports:
        digest.update(record.source_text.encode())
    for name in plan.extra_typing_names:
        digest.update(name.encode())
    for name in plan.type_checking_names:
        digest.update(name.encode())
    for constant in plan.constants:
        digest.update(constant.source_text.encode())
    for symbol in plan.symbols:
        digest.update(symbol.id.qualname.encode())
    for copied_source in plan.copied_sources:
        digest.update(copied_source.encode())
    return digest.hexdigest()


def _copied_sources(
    *,
    source_lines: list[str],
    symbols: tuple[SymbolRecord, ...],
    erased_type_arg_names: set[str],
    erased_annotation_names: set[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    copied_sources: list[str] = []
    extra_typing_names: set[str] = set()
    for symbol in symbols:
        source = "\n".join(source_lines[symbol.lineno - 1 : symbol.end_lineno])
        transformed_source, added_typing_names = _render_mypyc_friendly_source(
            source,
            erased_type_arg_names=erased_type_arg_names,
            erased_annotation_names=erased_annotation_names,
        )
        copied_sources.append(transformed_source)
        extra_typing_names.update(added_typing_names)
    return tuple(copied_sources), tuple(sorted(extra_typing_names))


def _runtime_referenced_names(
    copied_sources: tuple[str, ...],
    constants: tuple[ConstantRecord, ...],
) -> set[str]:
    names: set[str] = set()
    for source in (*copied_sources, *(constant.source_text for constant in constants)):
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        names.update(node.id for node in ast.walk(tree) if isinstance(node, ast.Name))
    return names


def _runtime_imports(
    imports: tuple[ImportRecord, ...],
    runtime_names: set[str],
) -> tuple[ImportRecord, ...]:
    return tuple(
        record
        for record in imports
        if record.source_text == "from __future__ import annotations"
        or any(name in runtime_names for name in record.imported_names)
    )


def _render_mypyc_friendly_source(
    source: str,
    *,
    erased_type_arg_names: set[str],
    erased_annotation_names: set[str],
) -> tuple[str, set[str]]:
    tree = ast.parse(source)
    transformer = _MypycSidecarTransformer(
        erased_type_arg_names=erased_type_arg_names,
        erased_annotation_names=erased_annotation_names,
    )
    transformed = transformer.visit(tree)
    ast.fix_missing_locations(transformed)
    return ast.unparse(transformed), transformer.extra_typing_names


def _erased_type_arg_names(imports: tuple[ImportRecord, ...]) -> set[str]:
    names: set[str] = set()
    for record in imports:
        if record.module is None:
            continue
        if record.module in _TYPING_IMPORT_MODULES:
            continue
        names.update(record.imported_names)
    return names


def _erased_annotation_names(
    imports: tuple[ImportRecord, ...],
    constants: tuple[ConstantRecord, ...],
) -> set[str]:
    return {
        constant.name for constant in constants if constant.kind != "literal_constant"
    } | _module_typevar_names(imports, constants)


def _module_typevar_names(
    imports: tuple[ImportRecord, ...],
    constants: tuple[ConstantRecord, ...],
) -> set[str]:
    aliases = _typing_aliases(imports)
    if not aliases.typevar_names and not aliases.typing_module_names:
        return set()
    return {
        constant.name
        for constant in constants
        if _is_typevar_assignment(constant.source_text, aliases)
    }


def _typing_aliases(imports: tuple[ImportRecord, ...]) -> _TypingAliases:
    typevar_names: set[str] = set()
    typing_module_names: set[str] = set()
    for record in imports:
        try:
            tree = ast.parse(record.source_text)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in _TYPING_IMPORT_MODULES:
                        typing_module_names.add(alias.asname or alias.name)
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module in _TYPING_IMPORT_MODULES
                and node.level == 0
            ):
                for alias in node.names:
                    if alias.name == "TypeVar":
                        typevar_names.add(alias.asname or alias.name)
    return _TypingAliases(
        typevar_names=frozenset(typevar_names),
        typing_module_names=frozenset(typing_module_names),
    )


def _is_typevar_assignment(source_text: str, aliases: _TypingAliases) -> bool:
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return False
    return any(
        _is_typevar_call(_assignment_value(node), aliases)
        for node in tree.body
        if isinstance(node, ast.Assign | ast.AnnAssign)
    )


def _assignment_value(node: ast.Assign | ast.AnnAssign) -> ast.expr | None:
    if isinstance(node, ast.Assign):
        return node.value
    return node.value


def _is_typevar_call(node: ast.expr | None, aliases: _TypingAliases) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name):
        return node.func.id in aliases.typevar_names
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "TypeVar"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in aliases.typing_module_names
    )


def _type_checking_names(
    *,
    referenced_names: set[str],
    imported_names: set[str],
    module_symbols: tuple[SymbolRecord, ...],
    included_names: set[str],
) -> tuple[str, ...]:
    local_classes = {
        symbol.id.qualname
        for symbol in module_symbols
        if symbol.kind == "class" and "." not in symbol.id.qualname
    }
    return tuple(
        sorted(
            name
            for name in referenced_names
            if name in local_classes and name not in imported_names and name not in included_names
        )
    )


class _MypycSidecarTransformer(ast.NodeTransformer):
    """Rewrite copied function source into a mypyc-friendly sidecar form.

    The transformer preserves runtime statements while erasing type constructs
    that are valid in the source project but rejected or unavailable in the
    generated sidecar context.
    """

    def __init__(
        self,
        *,
        erased_type_arg_names: set[str],
        erased_annotation_names: set[str],
    ) -> None:
        """Store erased names and collect extra typing imports needed later."""
        self._erased_type_arg_names = erased_type_arg_names
        self._erased_annotation_names = erased_annotation_names
        self.extra_typing_names: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Normalize a synchronous function after visiting its body."""
        self.generic_visit(node)
        self._normalize_function(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        """Normalize an async function after visiting its body."""
        self.generic_visit(node)
        self._normalize_function(node)
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
        """Erase subscripts for imported runtime types that are not in sidecars."""
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id in self._erased_type_arg_names:
            return node.value
        return node

    def _normalize_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        self._sanitize_function_annotations(node)
        if node.returns is None:
            node.returns = ast.Name(id="Any", ctx=ast.Load())
            self.extra_typing_names.add("Any")
        elif _annotation_allows_none(node.returns) and not _last_statement_exits(node.body):
            node.body.append(ast.Return(value=ast.Constant(value=None)))

    def _sanitize_function_annotations(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        for argument in _function_args(node):
            if argument.annotation is not None:
                argument.annotation = self._sanitize_annotation(argument.annotation)
        if node.args.vararg is not None and node.args.vararg.annotation is not None:
            node.args.vararg.annotation = self._sanitize_annotation(node.args.vararg.annotation)
        if node.args.kwarg is not None and node.args.kwarg.annotation is not None:
            node.args.kwarg.annotation = self._sanitize_annotation(node.args.kwarg.annotation)
        if node.returns is not None:
            node.returns = self._sanitize_annotation(node.returns)

    def _sanitize_annotation(self, annotation: ast.expr) -> ast.expr:
        sanitizer = _TypeVarAnnotationSanitizer(self._erased_annotation_names)
        sanitized = sanitizer.visit(annotation)
        if sanitizer.replaced:
            self.extra_typing_names.add("Any")
        return sanitized if isinstance(sanitized, ast.expr) else annotation


class _TypeVarAnnotationSanitizer(ast.NodeTransformer):
    """Replace erased annotation-only names with `Any` in copied signatures."""

    def __init__(self, erased_annotation_names: set[str]) -> None:
        """Track names that should not survive in generated annotations."""
        self._erased_annotation_names = erased_annotation_names
        self.replaced = False

    def visit_Name(self, node: ast.Name) -> ast.expr:
        """Return `Any` for erased annotation names while preserving location."""
        if node.id not in self._erased_annotation_names:
            return node
        self.replaced = True
        return ast.copy_location(ast.Name(id="Any", ctx=ast.Load()), node)


def _annotation_allows_none(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return node.value is None
    if isinstance(node, ast.Name):
        return node.id == "None"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _annotation_allows_none(node.left) or _annotation_allows_none(node.right)
    if isinstance(node, ast.Subscript) and _is_optional(node.value):
        return True
    if isinstance(node, ast.Subscript) and _is_union(node.value):
        return _slice_allows_none(node.slice)
    return False


def _function_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[ast.arg, ...]:
    return (
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
    )


def _is_optional(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "Optional"
    if isinstance(node, ast.Attribute):
        return node.attr == "Optional"
    return False


def _is_union(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "Union"
    if isinstance(node, ast.Attribute):
        return node.attr == "Union"
    return False


def _slice_allows_none(node: ast.expr) -> bool:
    if isinstance(node, ast.Tuple):
        return any(_annotation_allows_none(element) for element in node.elts)
    return _annotation_allows_none(node)


def _last_statement_exits(body: list[ast.stmt]) -> bool:
    return bool(body) and isinstance(body[-1], ast.Return | ast.Raise)
