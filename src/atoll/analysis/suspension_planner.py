"""Conservative suspension-aware block planner for typed-region members.

The planner inspects one retained :class:`atoll.models.RegionMember` without
executing it. It finds synchronous statement islands inside a callable body and
keeps suspension, exception, context-manager, and async-control boundaries in
the surrounding shell. The result is intentionally evidence-heavy so later
Milestone 4 work can explain why a candidate block was accepted or rejected.
"""

from __future__ import annotations

import ast
import hashlib
import symtable
import textwrap
from dataclasses import dataclass
from typing import Final, cast

from atoll.models import RegionMember

_MIN_CREDIBLE_OPERATIONS: Final = 4
_LOAD_EVENT: Final = "load"
_STORE_EVENT: Final = "store"
_DELETE_EVENT: Final = "delete"


@dataclass(frozen=True, slots=True)
class StatementEvidence:
    """Source statement retained as block evidence.

    Attributes:
        source: Exact dedented source text for the statement.
        start_lineno: One-based line where the statement begins in the retained member source.
        start_col_offset: Zero-based UTF-8 byte offset where the statement begins.
        end_lineno: One-based line where the statement ends in the retained member source.
        end_col_offset: Zero-based UTF-8 byte offset immediately after the statement.
    """

    source: str
    start_lineno: int
    start_col_offset: int
    end_lineno: int
    end_col_offset: int


@dataclass(frozen=True, slots=True)
class RejectionEvidence:
    """Conservative reason that prevents member or block extraction.

    Attributes:
        code: Stable machine-readable reason category.
        message: Human-readable explanation of the conservative boundary.
        lineno: One-based source line for the evidence, when syntax-local.
    """

    code: str
    message: str
    lineno: int | None = None


@dataclass(frozen=True, slots=True)
class SuspensionBlock:
    """A deterministic synchronous block candidate inside a member body.

    Blocks are immutable value objects. Tuple fields are sorted where they are
    set-like and source ordered where they describe statements. ``eligible`` is
    true only when the block is suspension-free, single-entry/single-exit, has
    conservative liveness, and carries enough work signal to justify planning.

    Attributes:
        id: Stable content-derived block identifier.
        statements: Source/range evidence for the statements in the block.
        source_text: Exact dedented block source text.
        start_lineno: One-based first line of the block.
        start_col_offset: Zero-based UTF-8 byte offset of the first statement.
        end_lineno: One-based final line of the block.
        end_col_offset: Zero-based UTF-8 byte offset after the final statement.
        live_ins: Names read before local assignment inside the block.
        live_outs: Names assigned by the block and read later before redefinition.
        late_bound_globals: Global names loaded by the block at runtime.
        receiver_dependencies: Receiver attribute loads such as ``self.value``.
        assigned_names: Names stored by the block.
        loaded_names: Names loaded by the block.
        loop_count: Number of synchronous loops inside the block.
        operation_count: Lightweight native-work signal for the block.
        eligible: Whether the block passes all local planning gates.
        rejections: Block-local rejection evidence.
    """

    id: str
    statements: tuple[StatementEvidence, ...]
    source_text: str
    start_lineno: int
    start_col_offset: int
    end_lineno: int
    end_col_offset: int
    live_ins: tuple[str, ...]
    live_outs: tuple[str, ...]
    late_bound_globals: tuple[str, ...]
    receiver_dependencies: tuple[str, ...]
    assigned_names: tuple[str, ...]
    loaded_names: tuple[str, ...]
    loop_count: int
    operation_count: int
    eligible: bool
    rejections: tuple[RejectionEvidence, ...] = ()


@dataclass(frozen=True, slots=True)
class SuspensionPlan:
    """Suspension-aware synchronous block plan for one retained region member.

    Attributes:
        member_qualname: Region-member qualified name used for deterministic IDs.
        blocks: Source-ordered synchronous block candidates.
        rejections: Member-level rejection evidence.
        eligible_block_ids: IDs of blocks that pass conservative extraction gates.
    """

    member_qualname: str
    blocks: tuple[SuspensionBlock, ...]
    rejections: tuple[RejectionEvidence, ...]
    eligible_block_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _NameFacts:
    """Internal immutable name-use facts for one statement sequence or block.

    Attributes:
        loaded: Names read by the block.
        assigned: Names assigned by the block.
        live_ins: Names whose values must enter the block from its shell.
        late_bound_globals: Runtime global names passed explicitly to a helper.
        receiver_dependencies: Receiver attributes loaded by the block.
    """

    loaded: frozenset[str]
    assigned: frozenset[str]
    live_ins: frozenset[str]
    late_bound_globals: frozenset[str]
    receiver_dependencies: frozenset[str]


@dataclass(slots=True)
class _MutableNameFacts:
    """Mutable accumulator used only while walking one candidate block.

    Attributes:
        loaded: Names read so far.
        assigned: Names assigned so far.
        live_ins: Names read before a block-local assignment.
        late_bound_globals: Runtime global names observed so far.
        receiver_dependencies: Receiver attributes observed so far.
    """

    loaded: set[str]
    assigned: set[str]
    live_ins: set[str]
    late_bound_globals: set[str]
    receiver_dependencies: set[str]


@dataclass(frozen=True, slots=True)
class _NameEvent:
    """Source-ordered load/store event used for conservative live-out checks.

    Attributes:
        lineno: One-based source line containing the name event.
        col_offset: Zero-based UTF-8 byte offset containing the name event.
        name: Local or global identifier read or written at the event.
        kind: Stable load or store event category.
    """

    lineno: int
    col_offset: int
    name: str
    kind: str


@dataclass(frozen=True, slots=True)
class _FlowResult:
    """Definite assignments and external reads after one statement sequence.

    Attributes:
        definite: Names assigned on every path that reaches the sequence exit.
        live_ins: Names read on at least one path before a definite assignment.
    """

    definite: frozenset[str]
    live_ins: frozenset[str]


@dataclass(frozen=True, slots=True)
class _BlockIdFacts:
    """Stable facts included in a content-derived block identifier.

    Attributes:
        member_qualname: Source callable qualified name.
        source: Exact candidate block source.
        start_lineno: One-based first source line.
        start_col_offset: Zero-based first-statement byte offset.
        end_lineno: One-based final source line.
        end_col_offset: Zero-based final-statement byte offset.
        live_ins: Deterministically ordered block inputs.
        live_outs: Deterministically ordered block outputs.
    """

    member_qualname: str
    source: str
    start_lineno: int
    start_col_offset: int
    end_lineno: int
    end_col_offset: int
    live_ins: tuple[str, ...]
    live_outs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _BlockPlanningContext:
    """Callable-wide facts reused while materializing candidate blocks.

    Attributes:
        member_qualname: Source callable identity used in block IDs.
        source_text: Dedented retained callable source.
        global_names: Symtable-classified global dependencies.
        events: Ordered name events used for live-out checks.
        definite_entries: Names definitely assigned before each statement.
    """

    member_qualname: str
    source_text: str
    global_names: frozenset[str]
    events: tuple[_NameEvent, ...]
    definite_entries: dict[tuple[int, int], frozenset[str]]


@dataclass(frozen=True, slots=True)
class _BlockGateFacts:
    """Liveness and work evidence consumed by conservative block gates.

    Attributes:
        names: Immutable loaded, assigned, and dependency facts.
        live_outs: Values that must return to the Python shell.
        loop_count: Synchronous loops retained in the block.
        operation_count: Conservative native-work signal.
        guaranteed_names: Locals definitely available before helper invocation.
    """

    names: _NameFacts
    live_outs: frozenset[str]
    loop_count: int
    operation_count: int
    guaranteed_names: frozenset[str]


def plan_suspension_blocks(member: RegionMember) -> SuspensionPlan:
    """Build a deterministic conservative suspension-aware CFG/liveness plan.

    Args:
        member: Typed-region member whose retained source is analyzed.

    Returns:
        SuspensionPlan: Immutable block and rejection evidence derived only from
        the member source text.
    """
    source_text = textwrap.dedent(member.source_text).strip("\n")
    qualname = member.id.qualname
    module = ast.parse(source_text)
    callable_node = _single_callable(module)
    if callable_node is None:
        return SuspensionPlan(
            member_qualname=qualname,
            blocks=(),
            rejections=(
                RejectionEvidence(
                    code="unsupported_member",
                    message="suspension planning requires one function or async function member",
                ),
            ),
            eligible_block_ids=(),
        )

    member_rejections = (
        *_scope_rejections(source_text, callable_node.name),
        *_syntax_rejections(callable_node),
    )
    global_names = _global_names(source_text, callable_node.name)
    events = _name_events(callable_node)
    definite_entries = _definite_entry_names(callable_node)
    context = _BlockPlanningContext(
        member_qualname=qualname,
        source_text=source_text,
        global_names=global_names,
        events=events,
        definite_entries=definite_entries,
    )
    blocks = tuple(
        _build_block(context, statements) for statements in _statement_blocks(callable_node.body)
    )
    return SuspensionPlan(
        member_qualname=qualname,
        blocks=blocks,
        rejections=member_rejections,
        eligible_block_ids=tuple(
            block.id for block in blocks if block.eligible and not member_rejections
        ),
    )


def _single_callable(module: ast.Module) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    declarations = tuple(
        node for node in module.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    )
    if len(declarations) != 1:
        return None
    return declarations[0]


def _scope_rejections(source_text: str, function_name: str) -> tuple[RejectionEvidence, ...]:
    table = symtable.symtable(source_text, "<atoll-suspension-planner>", "exec")
    function_table = _function_table(table, function_name)
    if function_table is None:
        return (
            RejectionEvidence(
                code="missing_symbol_table",
                message=f"symtable did not expose callable scope: {function_name}",
            ),
        )
    rejections: list[RejectionEvidence] = []
    for symbol in function_table.get_symbols():
        if symbol.is_free():
            rejections.append(
                RejectionEvidence(
                    code="free_variable",
                    message=f"member reads free variable: {symbol.get_name()}",
                )
            )
        if symbol.is_nonlocal():
            rejections.append(
                RejectionEvidence(
                    code="nonlocal_variable",
                    message=f"member declares nonlocal variable: {symbol.get_name()}",
                )
            )
    cell_names = _cell_names(function_table)
    rejections.extend(
        RejectionEvidence(code="cell_variable", message=f"member creates cell variable: {name}")
        for name in cell_names
    )
    return tuple(rejections)


def _function_table(
    table: symtable.SymbolTable,
    function_name: str,
) -> symtable.SymbolTable | None:
    for child in table.get_children():
        if child.get_name() == function_name and child.get_type() == "function":
            return child
    return None


def _cell_names(function_table: symtable.SymbolTable) -> tuple[str, ...]:
    local_names = {
        symbol.get_name() for symbol in function_table.get_symbols() if symbol.is_local()
    }
    child_free_names: set[str] = set()
    for child in function_table.get_children():
        child_free_names.update(
            symbol.get_name() for symbol in child.get_symbols() if symbol.is_free()
        )
    return tuple(sorted(local_names & child_free_names))


def _syntax_rejections(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[RejectionEvidence, ...]:
    rejections: list[RejectionEvidence] = []
    for descendant in ast.walk(node):
        if descendant is node:
            continue
        if isinstance(
            descendant,
            ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda,
        ):
            rejections.append(
                RejectionEvidence(
                    code="nested_scope",
                    message="nested definitions, classes, and lambdas stay in the shell",
                    lineno=descendant.lineno,
                )
            )
        elif isinstance(descendant, ast.YieldFrom):
            rejections.append(
                RejectionEvidence(
                    code="yield_from",
                    message="yield from is not split into synchronous blocks",
                    lineno=descendant.lineno,
                )
            )
        elif isinstance(descendant, ast.Global):
            rejections.append(
                RejectionEvidence(
                    code="global_declaration",
                    message=(
                        "global declarations stay in the Python shell: "
                        f"{', '.join(descendant.names)}"
                    ),
                    lineno=descendant.lineno,
                )
            )
        elif isinstance(descendant, ast.Raise) and descendant.exc is None:
            rejections.append(
                RejectionEvidence(
                    code="bare_raise",
                    message="bare raise depends on active exception state",
                    lineno=descendant.lineno,
                )
            )
    return tuple(rejections)


def _global_names(source_text: str, function_name: str) -> frozenset[str]:
    function_table = _function_table(
        symtable.symtable(source_text, "<atoll-suspension-planner>", "exec"),
        function_name,
    )
    if function_table is None:
        return frozenset()
    return frozenset(
        symbol.get_name()
        for symbol in function_table.get_symbols()
        if symbol.is_global() and not symbol.is_declared_global()
    )


def _statement_blocks(statements: list[ast.stmt]) -> tuple[tuple[ast.stmt, ...], ...]:
    blocks: list[tuple[ast.stmt, ...]] = []
    current: list[ast.stmt] = []
    for statement in statements:
        if _is_shell_boundary(statement) or _contains_suspension(statement):
            if current:
                blocks.append(tuple(current))
                current = []
            blocks.extend(_nested_statement_blocks(statement))
        else:
            current.append(statement)
    if current:
        blocks.append(tuple(current))
    return tuple(blocks)


def _nested_statement_blocks(statement: ast.stmt) -> tuple[tuple[ast.stmt, ...], ...]:
    if isinstance(statement, ast.If):
        return (*_statement_blocks(statement.body), *_statement_blocks(statement.orelse))
    if isinstance(statement, ast.For | ast.While):
        return (*_statement_blocks(statement.body), *_statement_blocks(statement.orelse))
    return ()


def _is_shell_boundary(statement: ast.stmt) -> bool:
    return isinstance(
        statement,
        ast.Try
        | ast.TryStar
        | ast.AsyncFor
        | ast.AsyncWith
        | ast.With
        | ast.Match
        | ast.Raise
        | ast.Return
        | ast.Break
        | ast.Continue,
    )


def _contains_suspension(node: ast.AST) -> bool:
    return any(
        isinstance(descendant, ast.Await | ast.Yield | ast.YieldFrom | ast.AsyncFor | ast.AsyncWith)
        for descendant in ast.walk(node)
    )


def _build_block(
    context: _BlockPlanningContext,
    statements: tuple[ast.stmt, ...],
) -> SuspensionBlock:
    facts = _name_facts(statements, context.global_names)
    first_statement = min(statements, key=_statement_start)
    final_statement = max(statements, key=_statement_end)
    start_lineno = first_statement.lineno
    start_col_offset = first_statement.col_offset
    end_lineno = final_statement.end_lineno or final_statement.lineno
    end_col_offset = final_statement.end_col_offset or final_statement.col_offset
    statement_evidence = tuple(
        _statement_evidence(statement, context.source_text) for statement in statements
    )
    source = "\n".join(evidence.source for evidence in statement_evidence)
    loop_count = sum(
        isinstance(node, ast.For | ast.While)
        for statement in statements
        for node in ast.walk(statement)
    )
    operation_count = sum(_operation_count(statement) for statement in statements)
    live_outs = _live_outs(
        facts.assigned,
        end_lineno,
        end_col_offset,
        context.events,
    )
    rejections = _block_rejections(
        statements,
        _BlockGateFacts(
            names=facts,
            live_outs=live_outs,
            loop_count=loop_count,
            operation_count=operation_count,
            guaranteed_names=context.definite_entries.get(
                _statement_start(first_statement),
                frozenset(),
            ),
        ),
    )
    block_id = _block_id(
        _BlockIdFacts(
            member_qualname=context.member_qualname,
            source=source,
            start_lineno=start_lineno,
            start_col_offset=start_col_offset,
            end_lineno=end_lineno,
            end_col_offset=end_col_offset,
            live_ins=tuple(sorted(facts.live_ins)),
            live_outs=tuple(sorted(live_outs)),
        )
    )
    return SuspensionBlock(
        id=block_id,
        statements=statement_evidence,
        source_text=source,
        start_lineno=start_lineno,
        start_col_offset=start_col_offset,
        end_lineno=end_lineno,
        end_col_offset=end_col_offset,
        live_ins=tuple(sorted(facts.live_ins)),
        live_outs=tuple(sorted(live_outs)),
        late_bound_globals=tuple(sorted(facts.late_bound_globals)),
        receiver_dependencies=tuple(sorted(facts.receiver_dependencies)),
        assigned_names=tuple(sorted(facts.assigned)),
        loaded_names=tuple(sorted(facts.loaded)),
        loop_count=loop_count,
        operation_count=operation_count,
        eligible=not rejections,
        rejections=rejections,
    )


def _statement_evidence(statement: ast.stmt, source_text: str) -> StatementEvidence:
    source = ast.get_source_segment(source_text, statement)
    return StatementEvidence(
        source=source if source is not None else ast.unparse(statement),
        start_lineno=statement.lineno,
        start_col_offset=statement.col_offset,
        end_lineno=statement.end_lineno or statement.lineno,
        end_col_offset=statement.end_col_offset or statement.col_offset,
    )


def _statement_start(statement: ast.stmt) -> tuple[int, int]:
    return statement.lineno, statement.col_offset


def _statement_end(statement: ast.stmt) -> tuple[int, int]:
    return (
        statement.end_lineno or statement.lineno,
        statement.end_col_offset or statement.col_offset,
    )


def _name_facts(statements: tuple[ast.stmt, ...], global_names: frozenset[str]) -> _NameFacts:
    facts = _MutableNameFacts(set(), set(), set(), set(), set())
    for statement in statements:
        for descendant in _walk_without_nested_scopes(statement):
            _record_name_fact(facts, descendant, global_names)
    facts.live_ins.update(_flow_sequence(statements, frozenset(), entries=None).live_ins)
    return _NameFacts(
        loaded=frozenset(facts.loaded),
        assigned=frozenset(facts.assigned),
        live_ins=frozenset(facts.live_ins),
        late_bound_globals=frozenset(facts.late_bound_globals),
        receiver_dependencies=frozenset(facts.receiver_dependencies),
    )


def _record_name_fact(
    facts: _MutableNameFacts,
    node: ast.AST,
    global_names: frozenset[str],
) -> None:
    if isinstance(node, ast.AugAssign):
        for name in _target_names(node.target):
            _record_loaded_name(facts, name, global_names)
    elif isinstance(node, ast.Import | ast.ImportFrom):
        facts.assigned.update(_import_bound_names(node))
    elif isinstance(node, ast.Name):
        if isinstance(node.ctx, ast.Load):
            _record_loaded_name(facts, node.id, global_names)
        elif isinstance(node.ctx, ast.Store):
            facts.assigned.add(node.id)
    elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
        dependency = _receiver_dependency(node)
        if dependency is not None:
            facts.receiver_dependencies.add(dependency)


def _record_loaded_name(
    facts: _MutableNameFacts,
    name: str,
    global_names: frozenset[str],
) -> None:
    facts.loaded.add(name)
    if name in global_names:
        facts.late_bound_globals.add(name)


def _receiver_dependency(node: ast.Attribute) -> str | None:
    value = node.value
    if isinstance(value, ast.Name) and value.id in {"self", "cls"}:
        return f"{value.id}.{node.attr}"
    return None


def _definite_entry_names(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[tuple[int, int], frozenset[str]]:
    entries: dict[tuple[int, int], frozenset[str]] = {}
    _flow_sequence(
        tuple(node.body),
        frozenset(_argument_names(node.args)),
        entries=entries,
    )
    return entries


def _argument_names(arguments: ast.arguments) -> tuple[str, ...]:
    return (
        *(argument.arg for argument in arguments.posonlyargs),
        *(argument.arg for argument in arguments.args),
        *((arguments.vararg.arg,) if arguments.vararg is not None else ()),
        *(argument.arg for argument in arguments.kwonlyargs),
        *((arguments.kwarg.arg,) if arguments.kwarg is not None else ()),
    )


def _flow_sequence(
    statements: tuple[ast.stmt, ...],
    definite_in: frozenset[str],
    *,
    entries: dict[tuple[int, int], frozenset[str]] | None,
) -> _FlowResult:
    definite = definite_in
    live_ins: set[str] = set()
    for statement in statements:
        if entries is not None:
            entries[_statement_start(statement)] = definite
        result = _flow_statement(statement, definite, entries=entries)
        definite = result.definite
        live_ins.update(result.live_ins)
    return _FlowResult(definite=definite, live_ins=frozenset(live_ins))


def _flow_statement(
    statement: ast.stmt,
    definite_in: frozenset[str],
    *,
    entries: dict[tuple[int, int], frozenset[str]] | None,
) -> _FlowResult:
    if isinstance(statement, ast.If):
        return _if_flow(statement, definite_in, entries=entries)
    if isinstance(statement, ast.For | ast.AsyncFor):
        return _loop_flow(statement, definite_in, entries=entries)
    if isinstance(statement, ast.While):
        return _while_flow(statement, definite_in, entries=entries)
    if isinstance(statement, ast.Try | ast.TryStar | ast.With | ast.AsyncWith | ast.Match):
        _record_boundary_entries(statement, definite_in, entries=entries)
        return _FlowResult(
            definite=definite_in,
            live_ins=frozenset(_loaded_names(statement) - set(definite_in)),
        )
    loaded = _statement_loaded_names(statement)
    definite = set(definite_in)
    definite.update(_statement_stored_names(statement))
    definite.difference_update(_deleted_names(statement))
    return _FlowResult(
        definite=frozenset(definite),
        live_ins=frozenset(loaded - set(definite_in)),
    )


def _if_flow(
    statement: ast.If,
    definite_in: frozenset[str],
    *,
    entries: dict[tuple[int, int], frozenset[str]] | None,
) -> _FlowResult:
    body = _flow_sequence(tuple(statement.body), definite_in, entries=entries)
    orelse = (
        _flow_sequence(tuple(statement.orelse), definite_in, entries=entries)
        if statement.orelse
        else _FlowResult(definite=definite_in, live_ins=frozenset())
    )
    return _FlowResult(
        definite=body.definite & orelse.definite,
        live_ins=frozenset(
            (_loaded_names(statement.test) - set(definite_in))
            | set(body.live_ins)
            | set(orelse.live_ins)
        ),
    )


def _loop_flow(
    statement: ast.For | ast.AsyncFor,
    definite_in: frozenset[str],
    *,
    entries: dict[tuple[int, int], frozenset[str]] | None,
) -> _FlowResult:
    body_entry = definite_in | frozenset(_stored_names(statement.target))
    body = _flow_sequence(tuple(statement.body), body_entry, entries=entries)
    orelse = _flow_sequence(tuple(statement.orelse), definite_in, entries=entries)
    target_loads = _loaded_names(statement.target)
    return _FlowResult(
        definite=definite_in,
        live_ins=frozenset(
            (_loaded_names(statement.iter) | target_loads) - set(definite_in)
            | set(body.live_ins)
            | set(orelse.live_ins)
        ),
    )


def _while_flow(
    statement: ast.While,
    definite_in: frozenset[str],
    *,
    entries: dict[tuple[int, int], frozenset[str]] | None,
) -> _FlowResult:
    body = _flow_sequence(tuple(statement.body), definite_in, entries=entries)
    orelse = _flow_sequence(tuple(statement.orelse), definite_in, entries=entries)
    return _FlowResult(
        definite=definite_in,
        live_ins=frozenset(
            (_loaded_names(statement.test) - set(definite_in))
            | set(body.live_ins)
            | set(orelse.live_ins)
        ),
    )


def _record_boundary_entries(
    statement: ast.Try | ast.TryStar | ast.With | ast.AsyncWith | ast.Match,
    definite_in: frozenset[str],
    *,
    entries: dict[tuple[int, int], frozenset[str]] | None,
) -> None:
    if entries is None:
        return
    for field_name in ("body", "orelse", "finalbody"):
        nested = _statement_tuple(cast(object, getattr(statement, field_name, None)))
        if nested is not None:
            _flow_sequence(nested, definite_in, entries=entries)


def _statement_tuple(value: object) -> tuple[ast.stmt, ...] | None:
    if not isinstance(value, list):
        return None
    items = cast(list[object], value)
    if not all(isinstance(item, ast.stmt) for item in items):
        return None
    return tuple(cast(list[ast.stmt], items))


def _statement_loaded_names(statement: ast.stmt) -> set[str]:
    if isinstance(statement, ast.Assign):
        return _loaded_names(statement.value) | set().union(
            *(_loaded_names(target) for target in statement.targets)
        )
    if isinstance(statement, ast.AnnAssign):
        loaded = _loaded_names(statement.annotation) | _loaded_names(statement.target)
        if statement.value is not None:
            loaded.update(_loaded_names(statement.value))
        return loaded
    if isinstance(statement, ast.AugAssign):
        return (
            _loaded_names(statement.target)
            | _loaded_names(statement.value)
            | set(_target_names(statement.target))
        )
    return _loaded_names(statement)


def _statement_stored_names(statement: ast.stmt) -> set[str]:
    if isinstance(statement, ast.AnnAssign) and statement.value is None:
        return set()
    if isinstance(statement, ast.Import | ast.ImportFrom):
        return set(_import_bound_names(statement))
    if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return {statement.name}
    return _stored_names(statement)


def _loaded_names(node: ast.AST) -> set[str]:
    return {
        descendant.id
        for descendant in _walk_without_nested_scopes(node)
        if isinstance(descendant, ast.Name) and isinstance(descendant.ctx, ast.Load)
    }


def _stored_names(node: ast.AST) -> set[str]:
    return {
        descendant.id
        for descendant in _walk_without_nested_scopes(node)
        if isinstance(descendant, ast.Name) and isinstance(descendant.ctx, ast.Store)
    }


def _deleted_names(node: ast.AST) -> set[str]:
    return {
        descendant.id
        for descendant in _walk_without_nested_scopes(node)
        if isinstance(descendant, ast.Name) and isinstance(descendant.ctx, ast.Del)
    }


def _name_events(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[_NameEvent, ...]:
    events: list[_NameEvent] = []
    for descendant in _walk_without_nested_scopes(node):
        if isinstance(descendant, ast.AugAssign):
            events.extend(
                _NameEvent(
                    lineno=descendant.lineno,
                    col_offset=descendant.target.col_offset,
                    name=name,
                    kind=_LOAD_EVENT,
                )
                for name in _target_names(descendant.target)
            )
        elif isinstance(descendant, ast.Import | ast.ImportFrom):
            events.extend(
                _NameEvent(
                    lineno=descendant.lineno,
                    col_offset=descendant.col_offset,
                    name=name,
                    kind=_STORE_EVENT,
                )
                for name in _import_bound_names(descendant)
            )
        elif isinstance(descendant, ast.Name):
            if isinstance(descendant.ctx, ast.Load):
                events.append(
                    _NameEvent(
                        lineno=descendant.lineno,
                        col_offset=descendant.col_offset,
                        name=descendant.id,
                        kind=_LOAD_EVENT,
                    )
                )
            elif isinstance(descendant.ctx, ast.Store):
                events.append(
                    _NameEvent(
                        lineno=descendant.lineno,
                        col_offset=descendant.col_offset,
                        name=descendant.id,
                        kind=_STORE_EVENT,
                    )
                )
            elif isinstance(descendant.ctx, ast.Del):
                events.append(
                    _NameEvent(
                        lineno=descendant.lineno,
                        col_offset=descendant.col_offset,
                        name=descendant.id,
                        kind=_DELETE_EVENT,
                    )
                )
    return tuple(
        sorted(
            events,
            key=lambda event: (
                event.lineno,
                event.col_offset,
                0 if event.kind == _LOAD_EVENT else 1,
                event.name,
            ),
        )
    )


def _live_outs(
    assigned_names: frozenset[str],
    end_lineno: int,
    end_col_offset: int,
    events: tuple[_NameEvent, ...],
) -> frozenset[str]:
    end_position = (end_lineno, end_col_offset)
    return frozenset(
        name
        for name in assigned_names
        if any(
            event.name == name
            and event.kind in {_LOAD_EVENT, _DELETE_EVENT}
            and (event.lineno, event.col_offset) > end_position
            for event in events
        )
    )


def _block_rejections(
    statements: tuple[ast.stmt, ...],
    facts: _BlockGateFacts,
) -> tuple[RejectionEvidence, ...]:
    first_line = min(statement.lineno for statement in statements)
    rejections = list(_block_syntax_rejections(statements, first_line))
    unavailable_live_ins = (
        set(facts.names.live_ins)
        - set(facts.names.late_bound_globals)
        - set(facts.guaranteed_names)
    )
    if unavailable_live_ins:
        rejections.append(
            RejectionEvidence(
                code="unsafe_live_in",
                message=(
                    "block reads locals not definitely assigned at entry: "
                    f"{', '.join(sorted(unavailable_live_ins))}"
                ),
                lineno=first_line,
            )
        )
    if facts.live_outs and any(_contains_control(statement) for statement in statements):
        rejections.append(
            RejectionEvidence(
                code="unsafe_liveness",
                message="control-flow block assigns values that remain live after the block",
                lineno=first_line,
            )
        )
    if facts.names.assigned & facts.names.late_bound_globals:
        rejections.append(
            RejectionEvidence(
                code="global_write",
                message="block writes a name also resolved as a late-bound global",
                lineno=first_line,
            )
        )
    if facts.loop_count == 0 and facts.operation_count < _MIN_CREDIBLE_OPERATIONS:
        rejections.append(
            RejectionEvidence(
                code="not_credible",
                message="block lacks loop or repeated-operation work signal",
                lineno=first_line,
            )
        )
    return tuple(rejections)


def _block_syntax_rejections(
    statements: tuple[ast.stmt, ...],
    first_line: int,
) -> tuple[RejectionEvidence, ...]:
    """Reject syntax whose scope, exit, or state cannot move into a helper.

    Args:
        statements: Source-ordered candidate block statements.
        first_line: One-based source line attached to block-level evidence.

    Returns:
        tuple[RejectionEvidence, ...]: Deterministic syntax rejection evidence.
    """
    rejections: list[RejectionEvidence] = []
    if any(_contains_suspension(statement) for statement in statements):
        rejections.append(
            RejectionEvidence(
                code="suspension",
                message="block contains await, yield, async for, or async with",
                lineno=first_line,
            )
        )
    if any(_contains_unsafe_exit(statement) for statement in statements):
        rejections.append(
            RejectionEvidence(
                code="unsafe_exit",
                message="return, break, continue, and raise stay in the shell",
                lineno=first_line,
            )
        )
    if any(
        isinstance(
            descendant,
            ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
        )
        for statement in statements
        for descendant in _walk_without_nested_scopes(statement)
    ):
        rejections.append(
            RejectionEvidence(
                code="comprehension_scope",
                message="comprehension scopes stay in the Python shell",
                lineno=first_line,
            )
        )
    if any(
        isinstance(descendant, ast.Delete)
        for statement in statements
        for descendant in _walk_without_nested_scopes(statement)
    ):
        rejections.append(
            RejectionEvidence(
                code="local_delete",
                message="local and target deletion stays in the Python shell",
                lineno=first_line,
            )
        )
    if any(
        isinstance(descendant, ast.AnnAssign) and descendant.value is None
        for statement in statements
        for descendant in _walk_without_nested_scopes(statement)
    ):
        rejections.append(
            RejectionEvidence(
                code="annotation_only_local",
                message="annotation-only locals do not create transferable runtime values",
                lineno=first_line,
            )
        )
    if any(
        isinstance(descendant, ast.TypeAlias)
        for statement in statements
        for descendant in _walk_without_nested_scopes(statement)
    ):
        rejections.append(
            RejectionEvidence(
                code="local_type_alias",
                message="local type alias creation stays in the Python shell",
                lineno=first_line,
            )
        )
    if any(
        isinstance(descendant, ast.NamedExpr)
        for statement in statements
        for descendant in _walk_without_nested_scopes(statement)
    ):
        rejections.append(
            RejectionEvidence(
                code="named_expression",
                message="assignment expressions stay in the Python shell",
                lineno=first_line,
            )
        )
    if any(
        isinstance(descendant, ast.ImportFrom)
        and any(alias.name == "*" for alias in descendant.names)
        for statement in statements
        for descendant in _walk_without_nested_scopes(statement)
    ):
        rejections.append(
            RejectionEvidence(
                code="star_import",
                message="star imports cannot expose a deterministic live-out set",
                lineno=first_line,
            )
        )
    return tuple(rejections)


def _contains_unsafe_exit(statement: ast.stmt) -> bool:
    return any(
        isinstance(descendant, ast.Return | ast.Break | ast.Continue | ast.Raise)
        for descendant in _walk_without_nested_scopes(statement)
    )


def _contains_control(statement: ast.stmt) -> bool:
    return any(
        isinstance(descendant, ast.If | ast.For | ast.While | ast.Match)
        for descendant in _walk_without_nested_scopes(statement)
    )


def _operation_count(node: ast.AST) -> int:
    operation_types = (
        ast.Assign,
        ast.AnnAssign,
        ast.AugAssign,
        ast.BinOp,
        ast.BoolOp,
        ast.Call,
        ast.Compare,
        ast.Subscript,
    )
    return sum(
        isinstance(descendant, operation_types) for descendant in _walk_without_nested_scopes(node)
    )


def _target_names(node: ast.AST) -> tuple[str, ...]:
    return tuple(
        descendant.id
        for descendant in _walk_without_nested_scopes(node)
        if isinstance(descendant, ast.Name)
    )


def _import_bound_names(node: ast.Import | ast.ImportFrom) -> tuple[str, ...]:
    if isinstance(node, ast.ImportFrom):
        return tuple(alias.asname or alias.name for alias in node.names if alias.name != "*")
    return tuple(alias.asname or alias.name.split(".", maxsplit=1)[0] for alias in node.names)


def _walk_without_nested_scopes(node: ast.AST) -> tuple[ast.AST, ...]:
    walked: list[ast.AST] = []
    stack = [node]
    while stack:
        current = stack.pop()
        if current is not node and isinstance(
            current,
            ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda,
        ):
            continue
        walked.append(current)
        stack.extend(reversed(list(ast.iter_child_nodes(current))))
    return tuple(walked)


def _block_id(facts: _BlockIdFacts) -> str:
    digest = hashlib.sha256()
    digest.update(facts.member_qualname.encode())
    digest.update(b"\0")
    digest.update(facts.source.encode())
    digest.update(b"\0")
    digest.update(f"{facts.start_lineno}:{facts.end_lineno}".encode())
    digest.update(b":")
    digest.update(f"{facts.start_col_offset}:{facts.end_col_offset}".encode())
    digest.update(b"\0")
    digest.update(",".join(facts.live_ins).encode())
    digest.update(b"\0")
    digest.update(",".join(facts.live_outs).encode())
    return f"susp-{digest.hexdigest()[:16]}"
