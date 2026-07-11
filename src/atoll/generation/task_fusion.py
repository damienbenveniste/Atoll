"""Stage one conservative eager-task fusion experiment in a copied payload.

The generator rewrites exactly one planner-approved task-spawn expression. It
does not own profile selection, profitability decisions, filesystem copies, or
wheel promotion. Callers must apply the returned source only to disposable
payloads because eager task start can change scheduling order even when the
profile and static safety gates accepted the site.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from atoll.analysis.task_fusion import FusionPlan

_CODING_COOKIE = re.compile(rb"coding[=:]\s*([-\w.]+)")


@dataclass(frozen=True, slots=True)
class TaskFusionGeneration:
    """One staged eager-task source transformation.

    Attributes:
        plan_id: Stable planner identity represented by the transformation.
        helper_name: Collision-checked private helper inserted into the module.
        new_text: Complete transformed module source.
    """

    plan_id: str
    helper_name: str
    new_text: str


def generate_eager_task_fusion(source_text: str, plan: FusionPlan) -> TaskFusionGeneration:
    """Wrap one approved spawn call with a temporary eager task factory.

    The helper falls back to the original spawn call when no running asyncio
    loop or eager task factory exists. The prior loop task factory is restored
    synchronously before control returns from the spawn expression.

    Args:
        source_text: Installed module source from a disposable payload copy.
        plan: Eligible task-fusion plan whose source coordinates must still match.

    Returns:
        TaskFusionGeneration: Complete transformed source and helper identity.

    Raises:
        ValueError: If the plan is rejected, coordinates are stale or ambiguous,
            the spawn expression changed, or the generated helper would collide.
    """
    if not plan.eligible:
        raise ValueError(f"task-fusion plan is not eligible: {plan.id}")
    tree = ast.parse(source_text, type_comments=True)
    matches = tuple(
        node for node in ast.walk(tree) if isinstance(node, ast.Call) and _matches_plan(node, plan)
    )
    if len(matches) != 1:
        raise ValueError(
            f"task-fusion spawn coordinates resolved to {len(matches)} calls for {plan.id}"
        )
    spawn = matches[0]
    if _call_path(spawn.func) != plan.spawn_api:
        raise ValueError(f"task-fusion spawn API changed for {plan.id}: {_call_path(spawn.func)!r}")
    spawn_source = ast.get_source_segment(source_text, spawn) or ""
    if not plan.spawn_source or spawn_source != plan.spawn_source:
        raise ValueError(f"task-fusion spawn source changed for {plan.id}")
    helper_name = _helper_name(plan.id)
    if _name_exists(tree, helper_name):
        raise ValueError(f"task-fusion helper name already exists: {helper_name}")

    replacement = ast.unparse(
        ast.Call(
            func=ast.Name(id=helper_name, ctx=ast.Load()),
            args=[spawn.func, *spawn.args],
            keywords=spawn.keywords,
        )
    )
    source_bytes = source_text.encode("utf-8")
    call_start = _source_offset(source_bytes, spawn.lineno, spawn.col_offset)
    call_end = _source_offset(
        source_bytes,
        spawn.end_lineno or spawn.lineno,
        spawn.end_col_offset or spawn.col_offset,
    )
    insertion = _helper_insertion_offset(source_bytes, tree)
    transformed = _apply_edits(
        source_bytes,
        (
            (call_start, call_end, replacement.encode("utf-8")),
            (insertion, insertion, _helper_source(helper_name).encode("utf-8")),
        ),
    )
    return TaskFusionGeneration(
        plan_id=plan.id,
        helper_name=helper_name,
        new_text=transformed.decode("utf-8"),
    )


def _matches_plan(node: ast.Call, plan: FusionPlan) -> bool:
    return (
        node.lineno == plan.lineno
        and (node.end_lineno or node.lineno) == plan.end_lineno
        and node.col_offset == plan.col_offset
        and node.end_col_offset == plan.end_col_offset
    )


def _helper_name(plan_id: str) -> str:
    suffix = re.sub(r"[^A-Za-z0-9_]", "_", plan_id.rsplit(":", maxsplit=1)[-1])
    return f"_atoll_eager_spawn_{suffix}"


def _name_exists(tree: ast.Module, name: str) -> bool:
    return any(
        (isinstance(node, ast.Name) and node.id == name)
        or (
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
            and node.name == name
        )
        for node in ast.walk(tree)
    )


def _call_path(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        root = _call_path(node.value)
        return f"{root}.{node.attr}" if root else node.attr
    return ""


def _helper_insertion_offset(source: bytes, tree: ast.Module) -> int:
    insertion_line = _header_line_count(source)
    body = tree.body
    prefix_index = 0
    if body and _is_module_docstring(body[0]):
        insertion_line = max(insertion_line, body[0].end_lineno or body[0].lineno)
        prefix_index = 1
    for node in body[prefix_index:]:
        if not isinstance(node, ast.ImportFrom) or node.module != "__future__":
            break
        insertion_line = max(insertion_line, node.end_lineno or node.lineno)
    return _line_start_offset(source, insertion_line + 1)


def _header_line_count(source: bytes) -> int:
    lines = source.splitlines(keepends=True)
    count = 1 if lines and lines[0].startswith(b"#!") else 0
    for index, line in enumerate(lines[:2], start=1):
        if _CODING_COOKIE.search(line):
            count = max(count, index)
    return count


def _is_module_docstring(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _source_offset(source: bytes, lineno: int, col_offset: int) -> int:
    return _line_start_offset(source, lineno) + col_offset


def _line_start_offset(source: bytes, lineno: int) -> int:
    if lineno <= 1:
        return 0
    lines = source.splitlines(keepends=True)
    if lineno > len(lines) + 1:
        raise ValueError(f"source line is outside module: {lineno}")
    return sum(len(line) for line in lines[: lineno - 1])


def _apply_edits(
    source: bytes,
    edits: tuple[tuple[int, int, bytes], ...],
) -> bytes:
    updated = source
    for start, end, replacement in sorted(edits, key=lambda edit: edit[0], reverse=True):
        if start < 0 or end < start or end > len(updated):
            raise ValueError("task-fusion source edit is outside module bounds")
        updated = updated[:start] + replacement + updated[end:]
    return updated


def _helper_source(helper_name: str) -> str:
    return (
        "\n\n"
        f"def {helper_name}(_atoll_spawn, /, *args, **kwargs):\n"
        "    try:\n"
        "        import asyncio as _atoll_asyncio\n"
        "        _atoll_loop = _atoll_asyncio.get_running_loop()\n"
        "        _atoll_eager_factory = _atoll_asyncio.eager_task_factory\n"
        "    except (AttributeError, RuntimeError):\n"
        "        return _atoll_spawn(*args, **kwargs)\n"
        "    _atoll_previous_factory = _atoll_loop.get_task_factory()\n"
        "    _atoll_loop.set_task_factory(_atoll_eager_factory)\n"
        "    try:\n"
        "        return _atoll_spawn(*args, **kwargs)\n"
        "    finally:\n"
        "        _atoll_loop.set_task_factory(_atoll_previous_factory)\n"
        "\n\n"
    )
