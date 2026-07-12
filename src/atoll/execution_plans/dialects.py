"""Built-in scheduler dialect recognition for execution-plan discovery."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

SchedulerDialectName = Literal["asyncio", "anyio-on-asyncio"]
_ATTRIBUTE_CALL_LENGTH = 2


@dataclass(frozen=True, slots=True)
class SpawnCall:
    """One scheduler spawn call recognized in source syntax.

    Attributes:
        dialect: Scheduler dialect that recognized the spawn.
        callee_name: Lexical callee name when the target is resolvable locally.
        lineno: One-based source line of the spawn call.
        col_offset: Zero-based source column of the spawn call.
        end_lineno: One-based final source line of the spawn call.
        end_col_offset: Zero-based final source column of the spawn call.
        scheduler_owner: Stable task-group or scheduler receiver path.
        transport_arguments: Stable positional argument paths passed to the spawned
            callee. Unsupported expressions remain `None` so argument indexes do not
            shift during parameter matching.
    """

    dialect: SchedulerDialectName
    callee_name: str | None
    lineno: int
    col_offset: int
    end_lineno: int
    end_col_offset: int
    scheduler_owner: str | None
    transport_arguments: tuple[str | None, ...]


@runtime_checkable
class SchedulerDialect(Protocol):
    """AST-only scheduler dialect recognizer used during execution-plan discovery."""

    @property
    def name(self) -> SchedulerDialectName:
        """Return the stable scheduler dialect identifier.

        Returns:
            SchedulerDialectName: Dialect identifier used in execution-plan IDs.
        """
        ...

    @property
    def lowering_version(self) -> str:
        """Return the lowering version that affects generated scheduler code.

        Returns:
            str: Stable lowering version included in plan IDs and cache keys.
        """
        ...

    def recognize_spawn(self, node: ast.Call) -> SpawnCall | None:
        """Recognize a scheduler spawn call without evaluating target code.

        Args:
            node: AST call expression being inspected.

        Returns:
            SpawnCall | None: Structured spawn evidence when this dialect recognizes the call.
        """
        ...


@dataclass(frozen=True, slots=True)
class AsyncioDialect:
    """Recognizer for asyncio task scheduling forms.

    Attributes:
        name: Stable scheduler dialect identifier.
        lowering_version: Lowering version included in execution-plan cache keys.
    """

    name: SchedulerDialectName = "asyncio"
    lowering_version: str = "asyncio-v1"

    def recognize_spawn(self, node: ast.Call) -> SpawnCall | None:
        """Recognize `asyncio.create_task` and `TaskGroup.create_task` calls.

        Args:
            node: AST call expression being inspected.

        Returns:
            SpawnCall | None: Spawn evidence when the call has a coroutine call argument.
        """
        path = _attribute_path(node.func)
        if path is None:
            return None
        is_create_task = path in (("asyncio", "create_task"), ("create_task",))
        is_task_group_create = len(path) == _ATTRIBUTE_CALL_LENGTH and path[1] == "create_task"
        if not is_create_task and not is_task_group_create:
            return None
        if not node.args or not isinstance(node.args[0], ast.Call):
            return None
        callee_path = _attribute_path(node.args[0].func)
        callee_name = ".".join(callee_path) if callee_path is not None else None
        return SpawnCall(
            dialect=self.name,
            callee_name=callee_name,
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=getattr(node, "end_lineno", node.lineno),
            end_col_offset=getattr(node, "end_col_offset", node.col_offset),
            scheduler_owner=path[0] if len(path) == _ATTRIBUTE_CALL_LENGTH else None,
            transport_arguments=_argument_paths(node.args[0].args),
        )


@dataclass(frozen=True, slots=True)
class AnyioOnAsyncioDialect:
    """Recognizer for AnyIO task groups intended for asyncio-compatible lowering.

    Attributes:
        name: Stable scheduler dialect identifier.
        lowering_version: Lowering version included in execution-plan cache keys.
    """

    name: SchedulerDialectName = "anyio-on-asyncio"
    lowering_version: str = "anyio-on-asyncio-v1"

    def recognize_spawn(self, node: ast.Call) -> SpawnCall | None:
        """Recognize `TaskGroup.start_soon` spawn calls.

        Args:
            node: AST call expression being inspected.

        Returns:
            SpawnCall | None: Spawn evidence when the first argument names a callable.
        """
        path = _attribute_path(node.func)
        if path is None or len(path) < _ATTRIBUTE_CALL_LENGTH or path[-1] != "start_soon":
            return None
        if not node.args:
            return None
        callee_path = _attribute_path(node.args[0])
        callee_name = ".".join(callee_path) if callee_path is not None else None
        return SpawnCall(
            dialect=self.name,
            callee_name=callee_name,
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=getattr(node, "end_lineno", node.lineno),
            end_col_offset=getattr(node, "end_col_offset", node.col_offset),
            scheduler_owner=".".join(path[:-1]),
            transport_arguments=_argument_paths(node.args[1:]),
        )


def built_in_scheduler_dialects() -> tuple[SchedulerDialect, ...]:
    """Return the deterministic set of built-in scheduler dialects.

    Returns:
        tuple[SchedulerDialect, ...]: Asyncio and AnyIO-on-asyncio dialect recognizers.
    """
    return (AsyncioDialect(), AnyioOnAsyncioDialect())


def _attribute_path(node: ast.expr) -> tuple[str, ...] | None:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        parent = _attribute_path(node.value)
        if parent is None:
            return None
        return (*parent, node.attr)
    return None


def _argument_paths(nodes: list[ast.expr]) -> tuple[str | None, ...]:
    return tuple(
        ".".join(path) if (path := _attribute_path(node)) is not None else None for node in nodes
    )
