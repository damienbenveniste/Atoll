"""Enforce structured Google-style documentation on Atoll's public Python API."""

from __future__ import annotations

import ast
import re
from pathlib import Path

_SOURCE_ROOT = Path(__file__).parents[2] / "src" / "atoll"
_SECTION_HEADER = re.compile(r"^[A-Z][A-Za-z ]+:$")
_NONE_ANNOTATIONS = frozenset({"None", "Never", "NoReturn"})


def _section_entries(docstring: str, section: str) -> set[str]:
    """Return names documented in one Google-style section.

    Args:
        docstring: Parsed docstring text without quote delimiters.
        section: Section heading including its trailing colon.

    Returns:
        Entry names found before the first colon on indented section lines.
    """
    lines = docstring.splitlines()
    try:
        start = next(index for index, line in enumerate(lines) if line.strip() == section) + 1
    except StopIteration:
        return set()
    entries: set[str] = set()
    for line in lines[start:]:
        stripped = line.strip()
        if _SECTION_HEADER.fullmatch(stripped):
            break
        if stripped and ":" in stripped:
            entries.add(stripped.split(":", maxsplit=1)[0].lstrip("*"))
    return entries


def _function_parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
    """Collect caller-supplied parameter names from a function definition.

    Args:
        node: Function definition being audited.

    Returns:
        Parameter names excluding the conventional `self` and `cls` receivers.
    """
    parameters = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    names = [parameter.arg for parameter in parameters if parameter.arg not in {"self", "cls"}]
    if node.args.vararg is not None:
        names.append(node.args.vararg.arg)
    if node.args.kwarg is not None:
        names.append(node.args.kwarg.arg)
    return tuple(names)


def _contains_yield(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Detect yields owned by a function without entering nested definitions.

    Args:
        node: Function definition whose body should be inspected.

    Returns:
        Whether the function itself contains `yield` or `yield from`.
    """
    pending: list[ast.AST] = list(node.body)
    while pending:
        current = pending.pop()
        if isinstance(current, (ast.Yield, ast.YieldFrom)):
            return True
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        pending.extend(ast.iter_child_nodes(current))
    return False


def _contains_raise(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Detect explicit raises owned by a function without entering nested definitions.

    Args:
        node: Function definition whose body should be inspected.

    Returns:
        Whether the function itself contains an explicit `raise` statement.
    """
    pending: list[ast.AST] = list(node.body)
    while pending:
        current = pending.pop()
        if isinstance(current, ast.Raise):
            return True
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        pending.extend(ast.iter_child_nodes(current))
    return False


def _class_fields(node: ast.ClassDef) -> tuple[str, ...]:
    """Collect directly declared structured fields from a public class.

    Args:
        node: Class definition being audited.

    Returns:
        Names assigned with direct annotations in the class body.
    """
    return tuple(
        child.target.id
        for child in node.body
        if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name)
    )


def _audit_function(
    path: Path,
    qualified_name: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[str]:
    """Return Google-style documentation failures for one function.

    Args:
        path: Source file containing the definition.
        qualified_name: Module-local function or method name used in failures.
        node: Function definition being audited.

    Returns:
        Human-readable failures, empty when the docstring is complete.
    """
    location = f"{path.relative_to(_SOURCE_ROOT.parent.parent)}::{qualified_name}"
    docstring = ast.get_docstring(node)
    if docstring is None:
        return [f"{location}: missing docstring"]
    failures: list[str] = []
    parameters = _function_parameters(node)
    documented_parameters = _section_entries(docstring, "Args:")
    missing_parameters = sorted(set(parameters) - documented_parameters)
    if missing_parameters:
        failures.append(f"{location}: Args missing {', '.join(missing_parameters)}")
    return_annotation = ast.unparse(node.returns) if node.returns is not None else ""
    if _contains_yield(node):
        if "Yields:" not in docstring:
            failures.append(f"{location}: missing Yields section")
    elif (
        return_annotation
        and return_annotation not in _NONE_ANNOTATIONS
        and "Returns:" not in docstring
    ):
        failures.append(f"{location}: missing Returns section")
    if _contains_raise(node) and "Raises:" not in docstring:
        failures.append(f"{location}: missing Raises section")
    return failures


def test_public_docstrings_use_google_sections() -> None:
    """Require structured sections for every public top-level API and class field."""
    failures: list[str] = []
    for path in sorted(_SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("_"):
                    failures.extend(_audit_function(path, node.name, node))
                continue
            if not isinstance(node, ast.ClassDef) or node.name.startswith("_"):
                continue
            docstring = ast.get_docstring(node)
            location = f"{path.relative_to(_SOURCE_ROOT.parent.parent)}::{node.name}"
            if docstring is None:
                failures.append(f"{location}: missing docstring")
            else:
                missing_fields = sorted(
                    set(_class_fields(node)) - _section_entries(docstring, "Attributes:")
                )
                if missing_fields:
                    failures.append(f"{location}: Attributes missing {', '.join(missing_fields)}")
            for child in node.body:
                if isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) and not child.name.startswith("_"):
                    failures.extend(_audit_function(path, f"{node.name}.{child.name}", child))
    assert not failures, "\n" + "\n".join(failures)


def test_documented_internal_definitions_use_google_sections() -> None:
    """Require every existing internal docstring to follow the same section contract."""
    failures: list[str] = []
    for path in sorted(_SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            docstring = ast.get_docstring(node)
            if docstring is None:
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                failures.extend(_audit_function(path, f"{node.name}@{node.lineno}", node))
                continue
            missing_fields = sorted(
                set(_class_fields(node)) - _section_entries(docstring, "Attributes:")
            )
            if missing_fields:
                location = (
                    f"{path.relative_to(_SOURCE_ROOT.parent.parent)}::{node.name}@{node.lineno}"
                )
                failures.append(f"{location}: Attributes missing {', '.join(missing_fields)}")
    assert not failures, "\n" + "\n".join(failures)
