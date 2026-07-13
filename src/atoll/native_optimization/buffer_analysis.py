"""Identify read-only zero-copy buffer reductions for later native lowering.

This module performs conservative frontend analysis only. It proves that a
pure synchronous function or static method reads exact buffer arguments without
materializing Python containers, mutating memory, suspending execution, or
calling opaque helpers. It does not generate native code or integrate plans
into package-time compilation.
"""

from __future__ import annotations

import ast
import hashlib
import symtable
import textwrap
from dataclasses import dataclass, replace
from typing import Literal

from atoll.models import ModuleScan, RegionMember, SymbolId, TypedRegion
from atoll.native_optimization.models import (
    BufferLayoutGuardPayload,
    ExactTypeGuardPayload,
    GuardExpression,
)

BufferRejectionCode = Literal[
    "unsupported-binding",
    "unsupported-execution",
    "unsupported-signature",
    "unsupported-annotation",
    "unsupported-decorator",
    "unsupported-scope",
    "unsupported-statement",
    "unsupported-expression",
    "opaque-call",
    "external-mutation",
    "unsupported-indexing",
    "unsupported-layout",
    "no-return",
]
BufferAccessKind = Literal["iteration", "indexed", "len"]
AccumulatorUpdateKind = Literal["initialize", "update"]
BufferReductionKind = Literal["add", "xor", "count"]

_BUFFER_ANALYSIS_VERSION = "buffer-analysis-v1"
_UINT64_MAX = 2**64 - 1
_BYTE_MAX = 255
_ALLOWED_PARAMETER_KINDS = frozenset({"positional_only", "positional", "keyword_only"})
_ALLOWED_GLOBALS = frozenset({"int", "len", "range"})
_SUPPORTED_ANNOTATIONS = frozenset(
    {"bytes", "bytearray", "memoryview", "array.array", "array.array[int]"}
)
_EXACT_TYPES: dict[str, tuple[str, str]] = {
    "bytes": ("builtins", "bytes"),
    "bytearray": ("builtins", "bytearray"),
    "memoryview": ("builtins", "memoryview"),
    "array.array": ("array", "array"),
    "array.array[int]": ("array", "array"),
}
_READONLY_BY_ANNOTATION: dict[str, bool | None] = {
    "bytes": True,
    "bytearray": False,
    "memoryview": None,
    "array.array": False,
    "array.array[int]": False,
}


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """Absolute source span for one buffer-analysis evidence record.

    Attributes:
        lineno: One-based first source line.
        end_lineno: One-based final source line.
        col_offset: Zero-based starting UTF-8 byte offset.
        end_col_offset: Zero-based ending UTF-8 byte offset, when available.
    """

    lineno: int
    end_lineno: int
    col_offset: int
    end_col_offset: int | None


@dataclass(frozen=True, slots=True)
class BufferRejection:
    """One deterministic reason a callable cannot use zero-copy buffer lowering.

    Attributes:
        member: Source member rejected by buffer analysis.
        code: Stable machine-readable rejection category.
        message: Concrete report-facing explanation.
        lineno: One-based source line relative to the retained declaration, or
            absolute after module-scan rebasing.
    """

    member: SymbolId
    code: BufferRejectionCode
    message: str
    lineno: int | None = None

    def __post_init__(self) -> None:
        """Reject blank diagnostic messages.

        Raises:
            ValueError: If ``message`` is empty or whitespace only.
        """
        if not self.message.strip():
            raise ValueError("buffer rejection message must be non-empty")


@dataclass(frozen=True, slots=True)
class BufferParameterEvidence:
    """Exact buffer parameter and runtime layout required by one plan.

    Attributes:
        name: Source parameter name.
        annotation: Exact source annotation admitted by analysis.
        type_module: Importable module that owns the expected exact runtime type.
        type_qualname: Qualified runtime type name.
        layout: Required one-dimensional contiguous buffer layout. For ``memoryview``
            and ``array.array``, this initial frontend accepts only the guarded
            unsigned-byte format and rejects all other runtime formats by requiring
            ``format="B"`` and ``itemsize=1``.
        max_length: Maximum admitted buffer length for additive and count reductions, or
            ``None`` when the reduction cannot overflow an unsigned 64-bit accumulator.
    """

    name: str
    annotation: str
    type_module: str
    type_qualname: str
    layout: BufferLayoutGuardPayload
    max_length: int | None = None


@dataclass(frozen=True, slots=True)
class BufferAccessEvidence:
    """Source evidence for one length, iteration, or indexed buffer read.

    Attributes:
        span: Absolute source span for the expression or loop.
        expression: Stable AST rendering of the source construct.
        buffer: Buffer parameter read by the construct.
        kind: Whether the proof covers direct iteration, guarded indexing, or length.
        index_name: Proven range induction variable for indexed access, when used.
    """

    span: SourceSpan
    expression: str
    buffer: str
    kind: BufferAccessKind
    index_name: str | None = None


@dataclass(frozen=True, slots=True)
class AccumulatorEvidence:
    """Source evidence for one scalar accumulator initializer or update.

    Attributes:
        span: Absolute source span for the assignment.
        name: Local scalar accumulator name.
        expression: Stable AST rendering of the initializer or update expression.
        kind: Whether this record initializes or updates the accumulator.
    """

    span: SourceSpan
    name: str
    expression: str
    kind: AccumulatorUpdateKind


@dataclass(frozen=True, slots=True)
class ReturnEvidence:
    """Source evidence for the final scalar return value.

    Attributes:
        span: Absolute source span for the return statement.
        expression: Stable AST rendering of the returned scalar expression.
        accumulator: Returned accumulator name when the return is a direct local.
    """

    span: SourceSpan
    expression: str
    accumulator: str | None


@dataclass(frozen=True, slots=True)
class BufferKernelPlan:
    """Content-addressed zero-copy buffer proof plan for one callable.

    Attributes:
        id: Stable identity derived only from source and proof evidence.
        region_id: Typed region that owns the source declaration.
        member: Function or static method selected for later lowering.
        source_hash: Hash of the exact retained declaration.
        declaration_start_lineno: Absolute first decorator or declaration line.
        end_lineno: Absolute final declaration line.
        buffers: Exact buffer parameter and layout evidence in signature order.
        accesses: Proven read-only buffer access sites in source order.
        accumulators: Scalar accumulator initializer and update evidence.
        returns: Return statements proven to expose scalar results.
        guards: Exact-type and buffer-layout checks required before entry.
        reduction: Proven reduction operation used by accumulator updates.
    """

    id: str
    region_id: str
    member: SymbolId
    source_hash: str
    declaration_start_lineno: int
    end_lineno: int
    buffers: tuple[BufferParameterEvidence, ...]
    accesses: tuple[BufferAccessEvidence, ...]
    accumulators: tuple[AccumulatorEvidence, ...]
    returns: tuple[ReturnEvidence, ...]
    guards: tuple[GuardExpression, ...]
    reduction: BufferReductionKind


@dataclass(frozen=True, slots=True)
class BufferAnalysisResult:
    """Buffer plans and explicit fallbacks discovered in one module scan.

    Attributes:
        plans: Eligible callable plans in typed-region source order.
        rejections: Ineligible callable diagnostics in source order.
    """

    plans: tuple[BufferKernelPlan, ...]
    rejections: tuple[BufferRejection, ...]


@dataclass(frozen=True, slots=True)
class BufferMemberAnalysisOptions:
    """Module context required to analyze one retained member safely.

    Attributes:
        builtin_len_available: Whether module scope leaves ``len`` bound to the builtin.
        builtin_range_available: Whether module scope leaves ``range`` bound to the builtin.
        declaration_start_lineno: Absolute first decorator or declaration line.
        end_lineno: Absolute final declaration line, when known.
    """

    builtin_len_available: bool = True
    builtin_range_available: bool = True
    declaration_start_lineno: int = 1
    end_lineno: int | None = None


@dataclass(frozen=True, slots=True)
class _AnalysisState:
    accumulator: str | None
    accumulator_initialized: bool
    reduction: BufferReductionKind | None
    buffer_names: frozenset[str]
    element_names: frozenset[str]
    index_names: frozenset[str]
    length_names: dict[str, str]
    accesses: tuple[BufferAccessEvidence, ...]
    accumulators: tuple[AccumulatorEvidence, ...]
    returns: tuple[ReturnEvidence, ...]
    terminated: bool = False


class _AnalysisError(Exception):
    def __init__(
        self,
        code: BufferRejectionCode,
        message: str,
        lineno: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code: BufferRejectionCode = code
        self.message = message
        self.lineno = lineno


def analyze_buffer_scan(scan: ModuleScan) -> BufferAnalysisResult:
    """Analyze every synchronous buffer callable retained by a module scan.

    Args:
        scan: Enriched module scan containing typed regions and scanner symbols.

    Returns:
        BufferAnalysisResult: Eligible zero-copy proof plans and conservative rejections.
    """
    plans: list[BufferKernelPlan] = []
    rejections: list[BufferRejection] = []
    symbols = {symbol.id: symbol for symbol in scan.symbols}
    module_blocker = next(
        (blocker for blocker in scan.blockers if blocker.severity == "hard"), None
    )
    builtin_len_available = _builtin_available(scan, "len")
    builtin_range_available = _builtin_available(scan, "range")
    for region in scan.typed_regions:
        for member in region.members:
            if member.kind not in {"function", "method"}:
                continue
            symbol = symbols.get(member.id)
            if symbol is None:
                rejections.append(
                    BufferRejection(
                        member=member.id,
                        code="unsupported-scope",
                        message="typed-region member has no matching scanner symbol",
                    )
                )
                continue
            blocker = next(
                (item for item in symbol.blockers if item.severity == "hard"),
                module_blocker,
            )
            if blocker is not None:
                rejections.append(
                    BufferRejection(
                        member=member.id,
                        code="unsupported-scope",
                        message=f"scanner blocker {blocker.code}: {blocker.message}",
                        lineno=blocker.lineno,
                    )
                )
                continue
            declaration_start = symbol.declaration_start_lineno or symbol.lineno
            outcome = analyze_buffer_member(
                region,
                member,
                options=BufferMemberAnalysisOptions(
                    builtin_len_available=builtin_len_available,
                    builtin_range_available=builtin_range_available,
                    declaration_start_lineno=declaration_start,
                    end_lineno=symbol.end_lineno,
                ),
            )
            if isinstance(outcome, BufferRejection) and outcome.lineno is not None:
                outcome = replace(outcome, lineno=declaration_start + outcome.lineno - 1)
            if isinstance(outcome, BufferKernelPlan):
                plans.append(outcome)
            else:
                rejections.append(outcome)
    return BufferAnalysisResult(plans=tuple(plans), rejections=tuple(rejections))


def analyze_buffer_member(
    region: TypedRegion,
    member: RegionMember,
    *,
    options: BufferMemberAnalysisOptions | None = None,
) -> BufferKernelPlan | BufferRejection:
    """Prove that one callable is a read-only zero-copy buffer reduction.

    Args:
        region: Typed region owning ``member``.
        member: Exact source declaration to assess.
        options: Builtin identity and absolute report coordinates.

    Returns:
        BufferKernelPlan | BufferRejection: A plan with required guards and source evidence,
        or the first deterministic reason the callable remains interpreted.
    """
    resolved = options or BufferMemberAnalysisOptions()
    try:
        node, source = _validate_member(member)
        _validate_local_annotations(node)
        _validate_scope(
            source,
            node,
            resolved.builtin_len_available,
            resolved.builtin_range_available,
        )
        buffers = tuple(
            _buffer_evidence(parameter.name, parameter.annotation or "")
            for parameter in member.parameters
        )
        body = _callable_body(node)
        initial = _AnalysisState(
            accumulator=None,
            accumulator_initialized=False,
            reduction=None,
            buffer_names=frozenset(buffer.name for buffer in buffers),
            element_names=frozenset(),
            index_names=frozenset(),
            length_names={},
            accesses=(),
            accumulators=(),
            returns=(),
        )
        state = _analyze_block(body, initial)
        _validate_single_traversal(
            body,
            frozenset(buffer.name for buffer in buffers),
        )
        _validate_completed_state(state, node)
    except _AnalysisError as error:
        return BufferRejection(
            member=member.id,
            code=error.code,
            message=error.message,
            lineno=error.lineno,
        )
    line_offset = resolved.declaration_start_lineno - 1
    reduction = _require_reduction(state, node)
    buffers = tuple(
        _with_max_length(_rebase_buffer_evidence(buffer), reduction) for buffer in buffers
    )
    accesses = tuple(_rebase_access(access, line_offset) for access in state.accesses)
    accumulators = tuple(_rebase_accumulator(item, line_offset) for item in state.accumulators)
    returns = tuple(_rebase_return(item, line_offset) for item in state.returns)
    guards = tuple(guard for buffer in buffers for guard in _buffer_guards(buffer))
    source_hash = hashlib.sha256(member.source_text.encode("utf-8")).hexdigest()
    identity = "\0".join(
        (
            _BUFFER_ANALYSIS_VERSION,
            region.id,
            member.id.stable_id,
            source_hash,
            *(
                f"{buffer.name}:{buffer.annotation}:{buffer.layout.format}:"
                f"{buffer.layout.itemsize}:{buffer.layout.ndim}:{buffer.max_length}"
                for buffer in buffers
            ),
            reduction,
            *(f"{access.kind}:{access.buffer}:{access.expression}" for access in accesses),
            *(f"{item.kind}:{item.name}:{item.expression}" for item in accumulators),
            *(item.expression for item in returns),
        )
    )
    plan_id = hashlib.blake2b(identity.encode("utf-8"), digest_size=16).hexdigest()
    return BufferKernelPlan(
        id=f"buffer-{plan_id}",
        region_id=region.id,
        member=member.id,
        source_hash=source_hash,
        declaration_start_lineno=resolved.declaration_start_lineno,
        end_lineno=(
            resolved.end_lineno
            or resolved.declaration_start_lineno + (node.end_lineno or node.lineno) - 1
        ),
        buffers=buffers,
        accesses=accesses,
        accumulators=accumulators,
        returns=returns,
        guards=guards,
        reduction=reduction,
    )


def _validate_member(member: RegionMember) -> tuple[ast.FunctionDef, str]:
    _validate_member_contract(member)
    source = textwrap.dedent(member.source_text)
    parsed = ast.parse(source)
    declarations = tuple(node for node in parsed.body if isinstance(node, ast.FunctionDef))
    if len(declarations) != 1 or len(parsed.body) != 1:
        raise _AnalysisError(
            "unsupported-statement",
            "buffer analysis requires one retained function declaration",
        )
    node = declarations[0]
    decorator_names = tuple(_simple_name(decorator) for decorator in node.decorator_list)
    expected = () if member.binding_kind == "module" else ("staticmethod",)
    if decorator_names != expected:
        raise _AnalysisError(
            "unsupported-decorator",
            "only an exact staticmethod decorator is supported on buffer methods",
            node.lineno,
        )
    return node, source


def _validate_local_annotations(node: ast.FunctionDef) -> None:
    """Reject non-integer local annotations before version-specific scope analysis.

    Args:
        node: Parsed synchronous callable whose local declarations are being proved.

    Raises:
        _AnalysisError: If a local annotation is not exactly ``int``.
    """
    for descendant in ast.walk(node):
        if isinstance(descendant, ast.AnnAssign) and ast.unparse(descendant.annotation) != "int":
            raise _AnalysisError(
                "unsupported-annotation",
                "buffer reduction locals must use exact int annotations",
                descendant.lineno,
            )


def _validate_member_contract(member: RegionMember) -> None:
    _validate_member_shape(member)
    _validate_member_parameters(member)


def _validate_member_shape(member: RegionMember) -> None:
    if member.execution_kind != "sync":
        raise _AnalysisError(
            "unsupported-execution",
            "zero-copy buffer variants require a synchronous callable",
        )
    if member.binding_kind not in {"module", "staticmethod"}:
        raise _AnalysisError(
            "unsupported-binding",
            "zero-copy buffer variants initially support module functions and static methods",
        )
    if member.type_parameters or member.scope_type_parameters:
        raise _AnalysisError(
            "unsupported-signature",
            "generic buffer declarations require a separate concrete specialization",
        )
    if member.return_annotation != "int":
        raise _AnalysisError(
            "unsupported-annotation",
            "zero-copy buffer reductions require an exact int return annotation",
        )


def _validate_member_parameters(member: RegionMember) -> None:
    if not member.parameters:
        raise _AnalysisError(
            "unsupported-signature",
            "zero-copy buffer reductions require at least one buffer parameter",
        )
    if len(member.parameters) != 1:
        raise _AnalysisError(
            "unsupported-signature",
            "zero-copy buffer reductions currently support exactly one buffer parameter",
        )
    for parameter in member.parameters:
        if parameter.kind not in _ALLOWED_PARAMETER_KINDS:
            raise _AnalysisError(
                "unsupported-signature",
                f"variadic parameter {parameter.name} is not supported",
            )
        if parameter.annotation not in _SUPPORTED_ANNOTATIONS:
            raise _AnalysisError(
                "unsupported-annotation",
                f"parameter {parameter.name} must use an exact supported buffer annotation",
            )
        if parameter.default_source is not None:
            raise _AnalysisError(
                "unsupported-signature",
                f"buffer parameter {parameter.name} cannot have a default",
            )


def _validate_scope(
    source: str,
    node: ast.FunctionDef,
    builtin_len_available: bool,
    builtin_range_available: bool,
) -> None:
    table = symtable.symtable(source, "<atoll-buffer-analysis>", "exec")
    function_table = next(
        (
            child
            for child in table.get_children()
            if child.get_type() == "function" and child.get_name() == node.name
        ),
        None,
    )
    if function_table is None:
        raise _AnalysisError("unsupported-scope", "symtable did not expose callable scope")
    runtime_children = tuple(
        child for child in function_table.get_children() if str(child.get_type()) != "annotation"
    )
    if runtime_children:
        raise _AnalysisError(
            "unsupported-scope",
            "nested functions, classes, comprehensions, and lambdas are not supported",
        )
    _validate_builtin_binding(function_table, node, "len", builtin_len_available)
    _validate_builtin_binding(function_table, node, "range", builtin_range_available)
    for symbol in function_table.get_symbols():
        name = symbol.get_name()
        if symbol.is_free() or symbol.is_nonlocal() or symbol.is_declared_global():
            raise _AnalysisError(
                "unsupported-scope",
                f"buffer callable cannot capture or mutate external name {name}",
            )
        if symbol.is_global() and name not in _ALLOWED_GLOBALS:
            raise _AnalysisError("opaque-call", f"global dependency {name} is opaque")


def _validate_builtin_binding(
    function_table: symtable.SymbolTable,
    node: ast.FunctionDef,
    name: Literal["len", "range"],
    available: bool,
) -> None:
    uses_name = any(
        isinstance(descendant, ast.Call)
        and isinstance(descendant.func, ast.Name)
        and descendant.func.id == name
        for descendant in ast.walk(node)
    )
    symbol = next(
        (item for item in function_table.get_symbols() if item.get_name() == name),
        None,
    )
    if uses_name and (
        not available
        or symbol is None
        or symbol.is_local()
        or symbol.is_parameter()
        or symbol.is_imported()
    ):
        raise _AnalysisError(
            "opaque-call",
            f"{name} must resolve to the unshadowed builtins.{name} callable",
            node.lineno,
        )


def _callable_body(node: ast.FunctionDef) -> list[ast.stmt]:
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body.pop(0)
    return body


def _analyze_block(statements: list[ast.stmt], initial: _AnalysisState) -> _AnalysisState:
    state = initial
    for statement in statements:
        if state.terminated:
            break
        state = _analyze_statement(statement, state)
    return state


def _validate_single_traversal(
    statements: list[ast.stmt],
    buffer_names: frozenset[str],
) -> None:
    """Require one loop and one reduction update per logical buffer element.

    The length proof for additive and count reductions assumes each exported
    element contributes at most once. Repeated or nested traversals would need
    multiplicity-aware interval evidence, so this first lowering rejects them.

    Args:
        statements: Callable body after removal of its optional docstring.
        buffer_names: Exact source parameters whose original storage the native view reads.

    Raises:
        _AnalysisError: If traversal or update multiplicity exceeds the v1 proof.
    """
    loops = tuple(statement for statement in statements if isinstance(statement, ast.For))
    if len(loops) != 1:
        raise _AnalysisError(
            "unsupported-statement",
            "zero-copy buffer reductions require exactly one top-level traversal",
            next((getattr(item, "lineno", None) for item in statements), None),
        )
    loop = loops[0]
    loop_index = statements.index(loop)
    returns = tuple(
        index for index, statement in enumerate(statements) if isinstance(statement, ast.Return)
    )
    if len(returns) != 1 or returns[0] <= loop_index:
        raise _AnalysisError(
            "unsupported-statement",
            "zero-copy buffer reductions require one top-level return after traversal",
            loop.lineno,
        )
    _validate_protected_names(statements, loop, buffer_names)
    if any(
        isinstance(descendant, ast.For)
        for statement in loop.body
        for descendant in ast.walk(statement)
    ):
        raise _AnalysisError(
            "unsupported-statement",
            "nested buffer traversals require multiplicity-aware bounds",
            loop.lineno,
        )
    if any(
        isinstance(descendant, ast.Assign | ast.AnnAssign)
        for statement in loop.body
        for descendant in ast.walk(statement)
    ):
        raise _AnalysisError(
            "unsupported-statement",
            "buffer traversal cannot initialize or rebind scalar locals",
            loop.lineno,
        )
    updates = tuple(
        descendant for descendant in ast.walk(loop) if isinstance(descendant, ast.AugAssign)
    )
    if len(updates) != 1:
        raise _AnalysisError(
            "unsupported-statement",
            "each buffer traversal must update the accumulator exactly once per element",
            loop.lineno,
        )


def _validate_protected_names(
    statements: list[ast.stmt],
    loop: ast.For,
    buffer_names: frozenset[str],
) -> None:
    """Reject rebinding that would diverge from captured native views.

    Args:
        statements: Top-level callable statements surrounding the traversal.
        loop: Single top-level traversal authorized by the multiplicity proof.
        buffer_names: Protected source parameters captured as native views.

    Raises:
        _AnalysisError: If initialization, assignment, or the loop target rebinds a
            protected buffer or accumulator name.
    """
    loop_index = statements.index(loop)
    initializers = tuple(
        statement
        for statement in statements[:loop_index]
        if (value := _assignment_value(statement)) is not None and _is_zero_literal(value)
    )
    if len(initializers) != 1:
        raise _AnalysisError(
            "unsupported-statement",
            "zero-copy buffer reductions require one top-level zero accumulator before traversal",
            loop.lineno,
        )
    accumulator = _assignment_name(initializers[0])
    if accumulator is None:
        raise _AnalysisError(
            "unsupported-statement",
            "buffer accumulator initialization must target one local name",
            loop.lineno,
        )
    for statement in statements:
        name = _assignment_name(statement)
        if name in buffer_names:
            raise _AnalysisError(
                "external-mutation",
                f"buffer parameter {name} cannot be rebound before native traversal",
                getattr(statement, "lineno", loop.lineno),
            )
        if name == accumulator and statement is not initializers[0]:
            raise _AnalysisError(
                "unsupported-statement",
                "buffer accumulator cannot be rebound after zero initialization",
                getattr(statement, "lineno", loop.lineno),
            )
    if not isinstance(loop.target, ast.Name) or loop.target.id in buffer_names | {accumulator}:
        raise _AnalysisError(
            "external-mutation",
            "buffer loop target cannot shadow a protected buffer or accumulator name",
            loop.lineno,
        )


def _assignment_name(statement: ast.stmt) -> str | None:
    if (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    ):
        return statement.targets[0].id
    if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
        return statement.target.id
    return None


def _assignment_value(statement: ast.stmt) -> ast.expr | None:
    if isinstance(statement, ast.Assign | ast.AnnAssign):
        return statement.value
    return None


def _analyze_statement(statement: ast.stmt, state: _AnalysisState) -> _AnalysisState:
    if isinstance(statement, ast.Assign):
        result = _analyze_assign(statement, state)
    elif isinstance(statement, ast.AnnAssign):
        result = _analyze_annotated_assign(statement, state)
    elif isinstance(statement, ast.AugAssign):
        result = _analyze_reduction_update(statement, state)
    elif isinstance(statement, ast.For):
        result = _analyze_for(statement, state)
    elif isinstance(statement, ast.If):
        result = _analyze_count_if(statement, state)
    elif isinstance(statement, ast.Return):
        result = _analyze_return(statement, state)
    elif isinstance(statement, ast.Pass):
        result = state
    else:
        raise _AnalysisError(
            "unsupported-statement",
            f"{type(statement).__name__} is not supported in a zero-copy buffer reduction",
            getattr(statement, "lineno", None),
        )
    return result


def _analyze_assign(statement: ast.Assign, state: _AnalysisState) -> _AnalysisState:
    if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
        raise _AnalysisError(
            "external-mutation",
            "assignments must target one local scalar name",
            statement.lineno,
        )
    name = statement.targets[0].id
    if _is_zero_literal(statement.value):
        return _initialize_accumulator(state, name, statement)
    buffer = _len_buffer(statement.value, state)
    if buffer is not None:
        access = _length_access(statement.value, buffer)
        return _replace_length(state, name, buffer, (access,))
    raise _AnalysisError(
        "unsupported-expression",
        "assignments must initialize an accumulator from exact zero or cache len(buffer)",
        statement.lineno,
    )


def _analyze_annotated_assign(
    statement: ast.AnnAssign,
    state: _AnalysisState,
) -> _AnalysisState:
    if not isinstance(statement.target, ast.Name) or statement.value is None:
        raise _AnalysisError(
            "external-mutation",
            "annotated assignments must initialize one local scalar name",
            statement.lineno,
        )
    if ast.unparse(statement.annotation) != "int":
        raise _AnalysisError(
            "unsupported-annotation",
            "buffer reduction locals must use exact int annotations",
            statement.lineno,
        )
    if not _is_zero_literal(statement.value):
        raise _AnalysisError(
            "unsupported-expression",
            "accumulators must initialize from the exact integer literal 0",
            statement.lineno,
        )
    return _initialize_accumulator(state, statement.target.id, statement)


def _initialize_accumulator(
    state: _AnalysisState,
    name: str,
    statement: ast.stmt,
) -> _AnalysisState:
    if state.accumulator is not None:
        raise _AnalysisError(
            "unsupported-statement",
            "zero-copy buffer reductions support exactly one accumulator",
            statement.lineno,
        )
    return replace(
        state,
        accumulator=name,
        accumulator_initialized=True,
        accumulators=(
            *state.accumulators,
            AccumulatorEvidence(
                span=_span(statement),
                name=name,
                expression=ast.unparse(statement),
                kind="initialize",
            ),
        ),
    )


def _analyze_reduction_update(
    statement: ast.AugAssign,
    state: _AnalysisState,
) -> _AnalysisState:
    if not isinstance(statement.target, ast.Name):
        raise _AnalysisError(
            "external-mutation",
            "augmented assignments must target one local scalar accumulator",
            statement.lineno,
        )
    _require_accumulator(statement.target.id, state, statement)
    if isinstance(statement.op, ast.Add):
        if _is_int_constant(statement.value, 1):
            raise _AnalysisError(
                "unsupported-expression",
                "count reductions require += 1 under a direct element comparison",
                statement.lineno,
            )
        access = _element_access(statement.value, state)
        return _replace_reduction(state, "add", (access,), statement)
    if isinstance(statement.op, ast.BitXor):
        access = _element_access(statement.value, state)
        return _replace_reduction(state, "xor", (access,), statement)
    raise _AnalysisError(
        "unsupported-expression",
        "buffer updates must use += element, ^= element, or += 1 under a comparison",
        statement.lineno,
    )


def _analyze_count_if(statement: ast.If, state: _AnalysisState) -> _AnalysisState:
    if statement.orelse or len(statement.body) != 1:
        raise _AnalysisError(
            "unsupported-statement",
            "conditional count reductions require one body statement and no else",
            statement.lineno,
        )
    access = _direct_element_comparison(statement.test, state)
    body_statement = statement.body[0]
    if (
        not isinstance(body_statement, ast.AugAssign)
        or not isinstance(body_statement.op, ast.Add)
        or not isinstance(body_statement.target, ast.Name)
        or not _is_int_constant(body_statement.value, 1)
    ):
        raise _AnalysisError(
            "unsupported-statement",
            "conditional count reductions must update the accumulator with += 1",
            getattr(body_statement, "lineno", statement.lineno),
        )
    _require_accumulator(body_statement.target.id, state, body_statement)
    return _replace_reduction(state, "count", (access,), body_statement)


def _analyze_return(statement: ast.Return, state: _AnalysisState) -> _AnalysisState:
    if statement.value is None:
        raise _AnalysisError(
            "unsupported-annotation",
            "zero-copy buffer reductions cannot return None",
            statement.lineno,
        )
    if not isinstance(statement.value, ast.Name) or statement.value.id != state.accumulator:
        raise _AnalysisError(
            "unsupported-expression",
            "zero-copy buffer reductions must return the direct accumulator",
            statement.lineno,
        )
    return replace(
        state,
        returns=(
            *state.returns,
            ReturnEvidence(
                span=_span(statement),
                expression=ast.unparse(statement.value),
                accumulator=statement.value.id,
            ),
        ),
        terminated=True,
    )


def _analyze_for(statement: ast.For, state: _AnalysisState) -> _AnalysisState:
    if statement.orelse:
        raise _AnalysisError(
            "unsupported-statement",
            "buffer loops cannot use else clauses",
            statement.lineno,
        )
    if not isinstance(statement.target, ast.Name):
        raise _AnalysisError(
            "external-mutation",
            "buffer loops must bind one local induction name",
            statement.lineno,
        )
    target = statement.target.id
    direct_buffer = statement.iter.id if isinstance(statement.iter, ast.Name) else None
    if direct_buffer in state.buffer_names:
        loop_state = replace(
            state,
            element_names=state.element_names | frozenset((target,)),
            accesses=(
                *state.accesses,
                BufferAccessEvidence(
                    span=_span(statement),
                    expression=ast.unparse(statement.iter),
                    buffer=direct_buffer,
                    kind="iteration",
                ),
            ),
        )
        analyzed = _analyze_block(statement.body, loop_state)
        return replace(analyzed, element_names=state.element_names)
    indexed_buffer = _range_len_buffer(statement.iter, state)
    if indexed_buffer is None:
        raise _AnalysisError(
            "unsupported-indexing",
            "for loops must directly iterate a buffer or use range(len(buffer))",
            getattr(statement.iter, "lineno", statement.lineno),
        )
    loop_state = replace(state, index_names=state.index_names | frozenset((target,)))
    analyzed = _analyze_block(statement.body, loop_state)
    return replace(analyzed, index_names=state.index_names)


def _element_access(expression: ast.expr, state: _AnalysisState) -> BufferAccessEvidence:
    if isinstance(expression, ast.Name) and expression.id in state.element_names:
        buffer = _single_buffer_name(state, expression)
        return BufferAccessEvidence(
            span=_span(expression),
            expression=ast.unparse(expression),
            buffer=buffer,
            kind="iteration",
        )
    if not isinstance(expression, ast.Subscript):
        raise _AnalysisError(
            "unsupported-expression",
            "buffer updates must read the direct loop element or buffer[index]",
            getattr(expression, "lineno", None),
        )
    if not isinstance(expression.value, ast.Name) or expression.value.id not in state.buffer_names:
        raise _AnalysisError(
            "unsupported-indexing",
            "indexed reads must target one supported buffer parameter",
            expression.lineno,
        )
    buffer = expression.value.id
    if isinstance(expression.slice, ast.Name) and expression.slice.id in state.index_names:
        return BufferAccessEvidence(
            span=_span(expression),
            expression=ast.unparse(expression),
            buffer=buffer,
            kind="indexed",
            index_name=expression.slice.id,
        )
    raise _AnalysisError(
        "unsupported-indexing",
        "buffer indexes must be the induction variable from range(len(buffer))",
        expression.lineno,
    )


def _direct_element_comparison(
    expression: ast.expr,
    state: _AnalysisState,
) -> BufferAccessEvidence:
    if not (
        isinstance(expression, ast.Compare)
        and len(expression.ops) == len(expression.comparators) == 1
        and isinstance(expression.ops[0], ast.Eq | ast.NotEq | ast.Lt | ast.LtE | ast.Gt | ast.GtE)
    ):
        raise _AnalysisError(
            "unsupported-expression",
            "conditional count reductions require one direct element comparison",
            getattr(expression, "lineno", None),
        )
    left_access = _optional_element_access(expression.left, state)
    right_access = _optional_element_access(expression.comparators[0], state)
    if (left_access is None) == (right_access is None):
        raise _AnalysisError(
            "unsupported-expression",
            "conditional count comparisons must compare one element with one int literal",
            expression.lineno,
        )
    literal = expression.comparators[0] if left_access is not None else expression.left
    if not _is_int_literal(literal):
        raise _AnalysisError(
            "unsupported-expression",
            "conditional count comparisons must compare against an int literal",
            expression.lineno,
        )
    if left_access is not None:
        return left_access
    if right_access is not None:
        return right_access
    raise AssertionError("element comparison access presence checked above")


def _optional_element_access(
    expression: ast.expr,
    state: _AnalysisState,
) -> BufferAccessEvidence | None:
    try:
        return _element_access(expression, state)
    except _AnalysisError as error:
        if error.code in {"unsupported-expression", "unsupported-indexing"}:
            return None
        raise


def _range_len_buffer(expression: ast.expr, state: _AnalysisState) -> str | None:
    if not isinstance(expression, ast.Call) or _simple_name(expression.func) != "range":
        return None
    if expression.keywords or len(expression.args) != 1:
        raise _AnalysisError(
            "unsupported-indexing",
            "indexed buffer loops require range(len(buffer))",
            expression.lineno,
        )
    argument = expression.args[0]
    if isinstance(argument, ast.Name):
        return state.length_names.get(argument.id)
    return _len_buffer(argument, state)


def _len_buffer(expression: ast.expr, state: _AnalysisState) -> str | None:
    if (
        isinstance(expression, ast.Call)
        and _simple_name(expression.func) == "len"
        and not expression.keywords
        and len(expression.args) == 1
        and isinstance(expression.args[0], ast.Name)
        and expression.args[0].id in state.buffer_names
    ):
        return expression.args[0].id
    return None


def _length_access(expression: ast.expr, buffer: str) -> BufferAccessEvidence:
    return BufferAccessEvidence(
        span=_span(expression),
        expression=ast.unparse(expression),
        buffer=buffer,
        kind="len",
    )


def _replace_length(
    state: _AnalysisState,
    name: str,
    buffer: str,
    accesses: tuple[BufferAccessEvidence, ...],
) -> _AnalysisState:
    lengths = dict(state.length_names)
    lengths[name] = buffer
    return replace(
        state,
        length_names=lengths,
        accesses=(*state.accesses, *accesses),
    )


def _replace_reduction(
    state: _AnalysisState,
    reduction: BufferReductionKind,
    accesses: tuple[BufferAccessEvidence, ...],
    statement: ast.stmt,
) -> _AnalysisState:
    if state.reduction is not None and state.reduction != reduction:
        raise _AnalysisError(
            "unsupported-expression",
            "zero-copy buffer reductions cannot mix accumulator operations",
            statement.lineno,
        )
    if state.accumulator is None:
        raise _AnalysisError(
            "unsupported-expression",
            "buffer updates require an initialized accumulator",
            statement.lineno,
        )
    return replace(
        state,
        reduction=reduction,
        accesses=(*state.accesses, *accesses),
        accumulators=(
            *state.accumulators,
            AccumulatorEvidence(
                span=_span(statement),
                name=state.accumulator,
                expression=ast.unparse(statement),
                kind="update",
            ),
        ),
    )


def _require_accumulator(name: str, state: _AnalysisState, node: ast.AST) -> None:
    if not state.accumulator_initialized or name != state.accumulator:
        raise _AnalysisError(
            "unsupported-expression",
            f"name {name} is not the initialized scalar accumulator",
            getattr(node, "lineno", None),
        )


def _validate_completed_state(state: _AnalysisState, node: ast.FunctionDef) -> None:
    if not state.returns:
        raise _AnalysisError("no-return", "buffer reductions require a scalar return", node.lineno)
    if not state.accesses:
        raise _AnalysisError(
            "unsupported-expression",
            "buffer reductions require at least one proven buffer read",
            node.lineno,
        )
    if state.reduction is None:
        raise _AnalysisError(
            "unsupported-expression",
            "buffer reductions require one supported accumulator update",
            node.lineno,
        )


def _require_reduction(state: _AnalysisState, node: ast.FunctionDef) -> BufferReductionKind:
    if state.reduction is None:
        raise _AnalysisError(
            "unsupported-expression",
            "buffer reductions require one supported accumulator update",
            node.lineno,
        )
    return state.reduction


def _single_buffer_name(state: _AnalysisState, node: ast.AST) -> str:
    if len(state.buffer_names) != 1:
        raise _AnalysisError(
            "unsupported-signature",
            "zero-copy buffer reductions currently support exactly one buffer parameter",
            getattr(node, "lineno", None),
        )
    return next(iter(state.buffer_names))


def _is_int_literal(expression: ast.expr) -> bool:
    return isinstance(expression, ast.Constant) and type(expression.value) is int


def _is_zero_literal(expression: ast.expr) -> bool:
    return _is_int_constant(expression, 0)


def _is_int_constant(expression: ast.expr, expected: int) -> bool:
    return (
        isinstance(expression, ast.Constant)
        and type(expression.value) is int
        and expression.value == expected
    )


def _buffer_evidence(name: str, annotation: str) -> BufferParameterEvidence:
    type_module, type_qualname = _EXACT_TYPES[annotation]
    return BufferParameterEvidence(
        name=name,
        annotation=annotation,
        type_module=type_module,
        type_qualname=type_qualname,
        layout=BufferLayoutGuardPayload(
            subject=name,
            format="B",
            itemsize=1,
            ndim=1,
            c_contiguous=True,
            f_contiguous=True,
            readonly=_READONLY_BY_ANNOTATION[annotation],
        ),
    )


def _with_max_length(
    buffer: BufferParameterEvidence,
    reduction: BufferReductionKind,
) -> BufferParameterEvidence:
    max_length = _max_length_for_reduction(reduction)
    layout = replace(
        buffer.layout,
        minimum_length=None if max_length is None else 0,
        maximum_length=max_length,
    )
    return replace(buffer, layout=layout, max_length=max_length)


def _max_length_for_reduction(reduction: BufferReductionKind) -> int | None:
    if reduction == "add":
        return _UINT64_MAX // _BYTE_MAX
    if reduction == "count":
        return _UINT64_MAX
    return None


def _buffer_guards(buffer: BufferParameterEvidence) -> tuple[GuardExpression, GuardExpression]:
    max_length = "" if buffer.max_length is None else f" and length <= {buffer.max_length}"
    return (
        GuardExpression(
            kind="exact-type",
            payload=ExactTypeGuardPayload(
                subject=buffer.name,
                type_module=buffer.type_module,
                type_qualname=buffer.type_qualname,
            ),
            message=f"type({buffer.name}) is {buffer.type_module}.{buffer.type_qualname}",
        ),
        GuardExpression(
            kind="buffer-layout",
            payload=buffer.layout,
            message=(
                f"{buffer.name} is a 1D C/F-contiguous {buffer.layout.format} buffer "
                f"with itemsize {buffer.layout.itemsize}{max_length}"
            ),
        ),
    )


def _span(node: ast.AST) -> SourceSpan:
    return SourceSpan(
        lineno=getattr(node, "lineno", 0),
        end_lineno=getattr(node, "end_lineno", getattr(node, "lineno", 0)),
        col_offset=getattr(node, "col_offset", 0),
        end_col_offset=getattr(node, "end_col_offset", None),
    )


def _rebase_buffer_evidence(buffer: BufferParameterEvidence) -> BufferParameterEvidence:
    return buffer


def _rebase_access(access: BufferAccessEvidence, line_offset: int) -> BufferAccessEvidence:
    return replace(access, span=_rebase_span(access.span, line_offset))


def _rebase_accumulator(item: AccumulatorEvidence, line_offset: int) -> AccumulatorEvidence:
    return replace(item, span=_rebase_span(item.span, line_offset))


def _rebase_return(item: ReturnEvidence, line_offset: int) -> ReturnEvidence:
    return replace(item, span=_rebase_span(item.span, line_offset))


def _rebase_span(span: SourceSpan, line_offset: int) -> SourceSpan:
    return replace(span, lineno=span.lineno + line_offset, end_lineno=span.end_lineno + line_offset)


def _simple_name(expression: ast.expr) -> str | None:
    if isinstance(expression, ast.Name):
        return expression.id
    if isinstance(expression, ast.Attribute):
        parent = _simple_name(expression.value)
        return f"{parent}.{expression.attr}" if parent is not None else None
    return None


def _builtin_available(scan: ModuleScan, name: Literal["len", "range"]) -> bool:
    for symbol in scan.symbols:
        if symbol.id.qualname == name:
            return False
    for constant in scan.constants:
        if constant.name == name:
            return False
    return all(name not in imported.imported_names for imported in scan.imports)
