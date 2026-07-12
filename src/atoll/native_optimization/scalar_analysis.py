"""Prove bounded integer callables before fixed-width native lowering.

This module owns conservative source analysis for scalar native variants. It
consumes backend-neutral typed regions, validates callable scope with both AST
and ``symtable`` evidence, and derives closed argument domains whose every
intermediate fits a requested native width. It does not generate Cython, import
target modules, or execute project code.
"""

from __future__ import annotations

import ast
import hashlib
import symtable
import textwrap
from dataclasses import dataclass, replace
from typing import Literal

from atoll.models import ModuleScan, RegionMember, SymbolId, TypedRegion
from atoll.native_optimization.intervals import (
    ClosedIntInterval,
    NativeInteger,
    OperationProof,
    OperationReason,
    accumulate_additive,
    add,
    bitwise,
    compare,
    floor_divide,
    join,
    modulo,
    multiply,
    power,
    shift,
    subtract,
)
from atoll.native_optimization.models import (
    ExactTypeGuardPayload,
    GuardExpression,
    IntegerDomainGuardPayload,
)

ScalarRejectionCode = Literal[
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
    "unbounded-loop",
    "unproven-arithmetic",
    "no-return",
]

_SCALAR_ANALYSIS_VERSION = "scalar-analysis-v1"
_WIDTHS: tuple[Literal[32, 64], ...] = (32, 64)
_ALLOWED_PARAMETER_KINDS = frozenset({"positional_only", "positional", "keyword_only"})
_ALLOWED_GLOBALS = frozenset({"int", "range"})
_MAX_RANGE_ARGUMENTS = 3


@dataclass(frozen=True, slots=True)
class ScalarRejection:
    """One deterministic reason a callable cannot use fixed-width lowering.

    Attributes:
        member: Source member rejected by scalar analysis.
        code: Stable machine-readable rejection category.
        message: Concrete report-facing explanation.
        lineno: One-based source line relative to the retained declaration.
    """

    member: SymbolId
    code: ScalarRejectionCode
    message: str
    lineno: int | None = None

    def __post_init__(self) -> None:
        """Reject blank diagnostic messages.

        Raises:
            ValueError: If ``message`` is empty or whitespace only.
        """
        if not self.message.strip():
            raise ValueError("scalar rejection message must be non-empty")


@dataclass(frozen=True, slots=True)
class ScalarOperationRecord:
    """Source location paired with one conservative interval proof.

    Attributes:
        lineno: One-based line relative to the retained declaration.
        end_lineno: One-based final source line covered by the expression.
        col_offset: Zero-based starting UTF-8 byte offset.
        end_col_offset: Zero-based ending UTF-8 byte offset, when available.
        expression: Stable AST rendering of the proved expression.
        proof: Structured interval operation and result evidence.
    """

    lineno: int
    end_lineno: int
    col_offset: int
    end_col_offset: int | None
    expression: str
    proof: OperationProof


@dataclass(frozen=True, slots=True)
class ScalarParameterDomain:
    """Closed argument domain admitted by one fixed-width variant.

    Attributes:
        name: Original source parameter name.
        interval: Inclusive Python integer domain proven safe for the variant.
    """

    name: str
    interval: ClosedIntInterval


@dataclass(frozen=True, slots=True)
class ScalarWidthProof:
    """Complete pre-entry proof for one native signed integer width.

    ``guards`` are constant-time checks over Python argument objects. Every
    operation record and the return interval fits ``native`` whenever those
    guards pass, so lowering must never catch overflow and retry Python after
    native execution begins.

    Attributes:
        native: Signed 32-bit or 64-bit target representation.
        parameters: Proven closed domains in source signature order.
        return_interval: Conservative interval for every reachable return.
        operations: Evaluation-order arithmetic and loop proof records.
        guards: Exact-type and integer-domain checks required before entry.
        explicit_modular_width: Unsigned modular width proven from an explicit
            source mask, or ``None`` for ordinary signed arithmetic.
    """

    native: NativeInteger
    parameters: tuple[ScalarParameterDomain, ...]
    return_interval: ClosedIntInterval
    operations: tuple[ScalarOperationRecord, ...]
    guards: tuple[GuardExpression, ...]
    explicit_modular_width: int | None = None


@dataclass(frozen=True, slots=True)
class ScalarKernelPlan:
    """Content-addressed scalar proof plan for one typed-region member.

    Attributes:
        id: Stable identity derived only from source and proof evidence.
        region_id: Typed region that owns the source declaration.
        member: Function or static method selected for later lowering.
        source_hash: Hash of the exact retained declaration.
        declaration_start_lineno: Absolute first decorator or declaration line.
        end_lineno: Absolute final declaration line.
        width_proofs: Safe variants in 32-bit then 64-bit dispatch order.
    """

    id: str
    region_id: str
    member: SymbolId
    source_hash: str
    declaration_start_lineno: int
    end_lineno: int
    width_proofs: tuple[ScalarWidthProof, ...]


@dataclass(frozen=True, slots=True)
class ScalarAnalysisResult:
    """Scalar plans and explicit fallbacks discovered in one module scan.

    Attributes:
        plans: Eligible callable plans in typed-region source order.
        rejections: Ineligible callable diagnostics in source order.
    """

    plans: tuple[ScalarKernelPlan, ...]
    rejections: tuple[ScalarRejection, ...]


@dataclass(frozen=True, slots=True)
class ScalarMemberAnalysisOptions:
    """Module context required to analyze one retained member safely.

    Attributes:
        constants: Immutable literal integer names and values available to the body.
        builtin_range_available: Whether module scope leaves ``range`` bound to the builtin.
        declaration_start_lineno: Absolute first decorator or declaration line.
        end_lineno: Absolute final declaration line, when known.
    """

    constants: tuple[tuple[str, int], ...] = ()
    builtin_range_available: bool = True
    declaration_start_lineno: int = 1
    end_lineno: int | None = None


@dataclass(frozen=True, slots=True)
class _ProofState:
    environment: dict[str, ClosedIntInterval]
    operations: tuple[ScalarOperationRecord, ...]
    returns: tuple[ClosedIntInterval, ...]
    terminated: bool = False


@dataclass(frozen=True, slots=True)
class _WidthContext:
    member: RegionMember
    node: ast.FunctionDef
    native: NativeInteger
    constants: dict[str, ClosedIntInterval]
    minimums: dict[str, int]


class _AnalysisError(Exception):
    def __init__(
        self,
        code: ScalarRejectionCode,
        message: str,
        lineno: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code: ScalarRejectionCode = code
        self.message = message
        self.lineno = lineno


def analyze_scalar_scan(scan: ModuleScan) -> ScalarAnalysisResult:
    """Analyze every synchronous callable retained by a module scan.

    Literal top-level integer constants are admitted as immutable proof inputs.
    Dynamic constants and imports remain opaque and cause member-level fallback
    when referenced.

    Args:
        scan: Enriched module scan containing typed regions and literal constants.

    Returns:
        ScalarAnalysisResult: Eligible proof plans and conservative rejections.
    """
    constants = _literal_integer_constants(scan)
    builtin_range_available = _builtin_range_available(scan)
    plans: list[ScalarKernelPlan] = []
    rejections: list[ScalarRejection] = []
    symbols = {symbol.id: symbol for symbol in scan.symbols}
    module_blocker = next(
        (blocker for blocker in scan.blockers if blocker.severity == "hard"), None
    )
    for region in scan.typed_regions:
        for member in region.members:
            if member.kind not in {"function", "method"}:
                continue
            symbol = symbols.get(member.id)
            if symbol is None:
                rejections.append(
                    ScalarRejection(
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
                    ScalarRejection(
                        member=member.id,
                        code="unsupported-scope",
                        message=f"scanner blocker {blocker.code}: {blocker.message}",
                        lineno=blocker.lineno,
                    )
                )
                continue
            declaration_start = symbol.declaration_start_lineno or symbol.lineno
            outcome = analyze_scalar_member(
                region,
                member,
                options=ScalarMemberAnalysisOptions(
                    constants=tuple(sorted(constants.items())),
                    builtin_range_available=builtin_range_available,
                    declaration_start_lineno=declaration_start,
                    end_lineno=symbol.end_lineno,
                ),
            )
            if isinstance(outcome, ScalarRejection) and outcome.lineno is not None:
                outcome = replace(
                    outcome,
                    lineno=declaration_start + outcome.lineno - 1,
                )
            if isinstance(outcome, ScalarKernelPlan):
                plans.append(outcome)
            else:
                rejections.append(outcome)
    return ScalarAnalysisResult(plans=tuple(plans), rejections=tuple(rejections))


def analyze_scalar_member(
    region: TypedRegion,
    member: RegionMember,
    *,
    options: ScalarMemberAnalysisOptions | None = None,
) -> ScalarKernelPlan | ScalarRejection:
    """Prove safe 32-bit and 64-bit domains for one callable.

    Args:
        region: Typed region owning ``member``.
        member: Exact source declaration to assess.
        options: Immutable constants, builtin identity, and absolute report coordinates.

    Returns:
        ScalarKernelPlan | ScalarRejection: A plan with at least one width proof,
        or the first deterministic reason the callable remains interpreted.
    """
    resolved = options or ScalarMemberAnalysisOptions()
    try:
        node, source = _validate_member(member)
        constant_intervals = {
            name: ClosedIntInterval.point(value) for name, value in resolved.constants
        }
        _validate_scope(source, node, constant_intervals, resolved.builtin_range_available)
        width_proofs = tuple(
            proof
            for width in _WIDTHS
            if (proof := _prove_width(member, node, width, constant_intervals)) is not None
        )
    except _AnalysisError as error:
        return ScalarRejection(
            member=member.id,
            code=error.code,
            message=error.message,
            lineno=error.lineno,
        )
    if not width_proofs:
        return ScalarRejection(
            member=member.id,
            code="unproven-arithmetic",
            message=(
                "no non-empty exact-int argument domain keeps every intermediate in 32 or 64 bits"
            ),
            lineno=node.lineno,
        )
    line_offset = resolved.declaration_start_lineno - 1
    width_proofs = tuple(_rebase_width_proof(proof, line_offset) for proof in width_proofs)
    source_hash = hashlib.sha256(member.source_text.encode("utf-8")).hexdigest()
    identity = "\0".join(
        (
            _SCALAR_ANALYSIS_VERSION,
            region.id,
            member.id.stable_id,
            source_hash,
            *(
                f"{proof.native.width}:"
                + ",".join(
                    f"{domain.name}={domain.interval.minimum}:{domain.interval.maximum}"
                    for domain in proof.parameters
                )
                for proof in width_proofs
            ),
        )
    )
    plan_id = hashlib.blake2b(identity.encode("utf-8"), digest_size=16).hexdigest()
    return ScalarKernelPlan(
        id=f"scalar-{plan_id}",
        region_id=region.id,
        member=member.id,
        source_hash=source_hash,
        declaration_start_lineno=resolved.declaration_start_lineno,
        end_lineno=(
            resolved.end_lineno
            or resolved.declaration_start_lineno + (node.end_lineno or node.lineno) - 1
        ),
        width_proofs=width_proofs,
    )


def _validate_member(member: RegionMember) -> tuple[ast.FunctionDef, str]:
    _validate_member_contract(member)
    source = textwrap.dedent(member.source_text)
    parsed = ast.parse(source)
    declarations = tuple(node for node in parsed.body if isinstance(node, ast.FunctionDef))
    if len(declarations) != 1 or len(parsed.body) != 1:
        raise _AnalysisError(
            "unsupported-statement",
            "scalar analysis requires one retained function declaration",
        )
    node = declarations[0]
    decorator_names = tuple(_simple_name(decorator) for decorator in node.decorator_list)
    expected = () if member.binding_kind == "module" else ("staticmethod",)
    if decorator_names != expected:
        raise _AnalysisError(
            "unsupported-decorator",
            "only an exact staticmethod decorator is supported on scalar methods",
            node.lineno,
        )
    return node, source


def _validate_member_contract(member: RegionMember) -> None:
    if member.execution_kind != "sync":
        raise _AnalysisError(
            "unsupported-execution",
            "fixed-width scalar variants require a synchronous callable",
        )
    if member.binding_kind not in {"module", "staticmethod"}:
        raise _AnalysisError(
            "unsupported-binding",
            "fixed-width scalar variants initially support module functions and static methods",
        )
    if member.type_parameters or member.scope_type_parameters:
        raise _AnalysisError(
            "unsupported-signature",
            "generic scalar declarations require a separate concrete specialization",
        )
    if member.return_annotation != "int":
        raise _AnalysisError(
            "unsupported-annotation",
            "fixed-width scalar variants require an exact int return annotation",
        )
    if not member.parameters:
        raise _AnalysisError(
            "unsupported-signature",
            "fixed-width scalar variants require at least one exact int parameter",
        )
    for parameter in member.parameters:
        if parameter.kind not in _ALLOWED_PARAMETER_KINDS:
            raise _AnalysisError(
                "unsupported-signature",
                f"variadic parameter {parameter.name} is not supported",
            )
        if parameter.annotation != "int":
            raise _AnalysisError(
                "unsupported-annotation",
                f"parameter {parameter.name} must use the exact int annotation",
            )
        if parameter.default_source is not None:
            _validate_default(parameter.name, parameter.default_source)


def _validate_default(name: str, source: str) -> None:
    try:
        value = ast.literal_eval(source)
    except (SyntaxError, ValueError) as error:
        raise _AnalysisError(
            "unsupported-signature",
            f"default for {name} must be a literal int",
        ) from error
    if type(value) is not int:
        raise _AnalysisError(
            "unsupported-signature",
            f"default for {name} must be an exact int",
        )


def _validate_scope(
    source: str,
    node: ast.FunctionDef,
    constants: dict[str, ClosedIntInterval],
    builtin_range_available: bool,
) -> None:
    table = symtable.symtable(source, "<atoll-scalar-analysis>", "exec")
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
    if function_table.get_children():
        raise _AnalysisError(
            "unsupported-scope",
            "nested functions, classes, comprehensions, and lambdas are not supported",
        )
    _validate_range_binding(function_table, node, builtin_range_available)
    _validate_scope_symbols(function_table, constants)


def _validate_range_binding(
    function_table: symtable.SymbolTable,
    node: ast.FunctionDef,
    builtin_range_available: bool,
) -> None:
    uses_range = any(
        isinstance(descendant, ast.Call)
        and isinstance(descendant.func, ast.Name)
        and descendant.func.id == "range"
        for descendant in ast.walk(node)
    )
    range_symbol = next(
        (symbol for symbol in function_table.get_symbols() if symbol.get_name() == "range"),
        None,
    )
    if uses_range and (
        not builtin_range_available
        or range_symbol is None
        or range_symbol.is_local()
        or range_symbol.is_parameter()
        or range_symbol.is_imported()
    ):
        raise _AnalysisError(
            "opaque-call",
            "range must resolve to the unshadowed builtins.range callable",
            node.lineno,
        )


def _validate_scope_symbols(
    function_table: symtable.SymbolTable,
    constants: dict[str, ClosedIntInterval],
) -> None:
    for symbol in function_table.get_symbols():
        name = symbol.get_name()
        if symbol.is_free() or symbol.is_nonlocal() or symbol.is_declared_global():
            raise _AnalysisError(
                "unsupported-scope",
                f"scalar callable cannot capture or mutate external name {name}",
            )
        if symbol.is_global() and name not in _ALLOWED_GLOBALS and name not in constants:
            raise _AnalysisError(
                "opaque-call",
                f"global dependency {name} is not an immutable literal integer",
            )


def _prove_width(
    member: RegionMember,
    node: ast.FunctionDef,
    width: Literal[32, 64],
    constants: dict[str, ClosedIntInterval],
) -> ScalarWidthProof | None:
    native = NativeInteger(width=width)
    minimums = _parameter_minimums(node, member)
    context = _WidthContext(member, node, native, constants, minimums)
    lower = max(minimums.values(), default=0)
    if _attempt_width(context, lower) is None:
        return None
    lo = lower
    hi = native.maximum
    while lo < hi:
        middle = lo + ((hi - lo + 1) // 2)
        if _attempt_width(context, middle) is None:
            hi = middle - 1
        else:
            lo = middle
    state = _attempt_width(context, lo)
    if state is None or not state.returns:
        return None
    domains = tuple(
        ScalarParameterDomain(
            name=parameter.name,
            interval=ClosedIntInterval.closed(minimums[parameter.name], lo),
        )
        for parameter in member.parameters
    )
    guards = tuple(guard for domain in domains for guard in _parameter_guards(domain, width))
    return ScalarWidthProof(
        native=native,
        parameters=domains,
        return_interval=_join_intervals(state.returns),
        operations=state.operations,
        guards=guards,
    )


def _attempt_width(
    context: _WidthContext,
    maximum: int,
) -> _ProofState | None:
    environment = dict(context.constants)
    for parameter in context.member.parameters:
        minimum = context.minimums[parameter.name]
        environment[parameter.name] = ClosedIntInterval.closed(minimum, maximum)
    try:
        state = _analyze_block(
            context.node.body,
            _ProofState(environment, (), ()),
            context.native,
        )
    except _AnalysisError:
        return None
    if not state.returns:
        return None
    return_interval = _join_intervals(state.returns)
    if not return_interval.fits_native(context.native):
        return None
    if any(not record.proof.fits_native(context.native) for record in state.operations):
        return None
    return state


def _parameter_minimums(
    node: ast.FunctionDef,
    member: RegionMember,
) -> dict[str, int]:
    divisors = {
        expression.right.id
        for expression in ast.walk(node)
        if isinstance(expression, ast.BinOp)
        and isinstance(expression.op, ast.FloorDiv | ast.Mod)
        and isinstance(expression.right, ast.Name)
    }
    return {
        parameter.name: 1 if parameter.name in divisors else 0 for parameter in member.parameters
    }


def _parameter_guards(
    domain: ScalarParameterDomain,
    width: Literal[32, 64],
) -> tuple[GuardExpression, GuardExpression]:
    return (
        GuardExpression(
            kind="exact-type",
            payload=ExactTypeGuardPayload(
                subject=domain.name,
                type_module="builtins",
                type_qualname="int",
            ),
            message=f"type({domain.name}) is int",
        ),
        GuardExpression(
            kind="integer-domain",
            payload=IntegerDomainGuardPayload(
                subject=domain.name,
                minimum=domain.interval.minimum,
                maximum=domain.interval.maximum,
                bit_width=width,
            ),
            message=(
                f"{domain.interval.lower.decimal} <= {domain.name} <= "
                f"{domain.interval.upper.decimal}"
            ),
        ),
    )


def _analyze_block(
    statements: list[ast.stmt],
    initial: _ProofState,
    native: NativeInteger,
) -> _ProofState:
    state = initial
    for statement in statements:
        if state.terminated:
            break
        state = _analyze_statement(statement, state, native)
    return state


def _analyze_statement(
    statement: ast.stmt,
    state: _ProofState,
    native: NativeInteger,
) -> _ProofState:
    simple = _analyze_simple_statement(statement, state)
    if simple is not None:
        return simple
    return _analyze_control_statement(statement, state, native)


def _analyze_simple_statement(
    statement: ast.stmt,
    state: _ProofState,
) -> _ProofState | None:
    if isinstance(statement, ast.Assign):
        if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
            raise _AnalysisError(
                "external-mutation",
                "assignments must target one local name",
                statement.lineno,
            )
        value, records = _evaluate(statement.value, state.environment)
        return _replace_local(state, statement.targets[0].id, value, records)
    if isinstance(statement, ast.AnnAssign):
        if not isinstance(statement.target, ast.Name) or statement.value is None:
            raise _AnalysisError(
                "external-mutation",
                "annotated assignments must initialize one local name",
                statement.lineno,
            )
        if ast.unparse(statement.annotation) != "int":
            raise _AnalysisError(
                "unsupported-annotation",
                "scalar local annotations must be exact int",
                statement.lineno,
            )
        value, records = _evaluate(statement.value, state.environment)
        return _replace_local(state, statement.target.id, value, records)
    if isinstance(statement, ast.AugAssign):
        if not isinstance(statement.target, ast.Name):
            raise _AnalysisError(
                "external-mutation",
                "augmented assignments must target one local name",
                statement.lineno,
            )
        left = _lookup(statement.target, state.environment)
        right, records = _evaluate(statement.value, state.environment)
        proof = _binary_proof(statement.op, left, right, statement)
        result = _proved_result(proof, statement)
        return _replace_local(
            state,
            statement.target.id,
            result,
            (*records, _record(statement, proof)),
        )
    return None


def _analyze_control_statement(
    statement: ast.stmt,
    state: _ProofState,
    native: NativeInteger,
) -> _ProofState:
    if isinstance(statement, ast.If):
        _, test_records = _evaluate_condition(statement.test, state.environment)
        branch_start = _ProofState(dict(state.environment), (), state.returns)
        body = _analyze_block(statement.body, branch_start, native)
        other = (
            _analyze_block(statement.orelse, branch_start, native)
            if statement.orelse
            else branch_start
        )
        continuing = tuple(branch for branch in (body, other) if not branch.terminated)
        environment = (
            _join_environments(tuple(branch.environment for branch in continuing))
            if continuing
            else dict(state.environment)
        )
        returns = (*body.returns, *(value for value in other.returns if value not in body.returns))
        return _ProofState(
            environment=environment,
            operations=(
                *state.operations,
                *test_records,
                *body.operations,
                *other.operations,
            ),
            returns=returns,
            terminated=not continuing,
        )
    if isinstance(statement, ast.For):
        return _analyze_range_loop(statement, state, native)
    if isinstance(statement, ast.Return):
        if statement.value is None:
            raise _AnalysisError(
                "unsupported-annotation",
                "an exact int callable cannot return None",
                statement.lineno,
            )
        value, records = _evaluate(statement.value, state.environment)
        return _ProofState(
            environment=dict(state.environment),
            operations=(*state.operations, *records),
            returns=(*state.returns, value),
            terminated=True,
        )
    if isinstance(statement, ast.Pass):
        return state
    raise _AnalysisError(
        "unsupported-statement",
        f"{type(statement).__name__} is not supported in a fixed-width scalar kernel",
        getattr(statement, "lineno", None),
    )


def _analyze_range_loop(
    statement: ast.For,
    state: _ProofState,
    native: NativeInteger,
) -> _ProofState:
    del native
    if not isinstance(statement.target, ast.Name):
        raise _AnalysisError(
            "external-mutation",
            "range induction must target one local name",
            statement.lineno,
        )
    if statement.orelse:
        raise _AnalysisError(
            "unsupported-statement",
            "range loop else clauses are not supported",
            statement.lineno,
        )
    induction, iterations, range_records = _range_summary(statement.iter, state.environment)
    loop_environment = dict(state.environment)
    loop_environment[statement.target.id] = induction
    updated = dict(state.environment)
    records: list[ScalarOperationRecord] = [*state.operations, *range_records]
    for body_statement in statement.body:
        target, expression = _additive_update(body_statement)
        if target is None or expression is None or target not in updated:
            raise _AnalysisError(
                "unproven-arithmetic",
                "bounded loops require additive local accumulator updates",
                getattr(body_statement, "lineno", statement.lineno),
            )
        if target in _loaded_names(expression):
            raise _AnalysisError(
                "unproven-arithmetic",
                f"loop recurrence for {target} is not an additive reduction",
                body_statement.lineno,
            )
        delta, delta_records = _evaluate(expression, loop_environment)
        proof = accumulate_additive(updated[target], delta, iterations)
        result = _proved_result(proof, body_statement)
        updated[target] = _join_intervals((updated[target], result))
        loop_environment[target] = updated[target]
        records.extend(delta_records)
        records.append(_record(body_statement, proof))
    updated[statement.target.id] = induction
    return _ProofState(
        environment=updated,
        operations=tuple(records),
        returns=state.returns,
        terminated=False,
    )


def _range_summary(
    expression: ast.expr,
    environment: dict[str, ClosedIntInterval],
) -> tuple[ClosedIntInterval, int, tuple[ScalarOperationRecord, ...]]:
    if not isinstance(expression, ast.Call) or _simple_name(expression.func) != "range":
        raise _AnalysisError(
            "unbounded-loop",
            "for loops must iterate directly over builtins.range",
            getattr(expression, "lineno", None),
        )
    if expression.keywords or not 1 <= len(expression.args) <= _MAX_RANGE_ARGUMENTS:
        raise _AnalysisError(
            "unbounded-loop",
            "range requires one to three positional integer arguments",
            expression.lineno,
        )
    evaluated = tuple(_evaluate(argument, environment) for argument in expression.args)
    intervals = tuple(value for value, _ in evaluated)
    records = tuple(record for _, argument_records in evaluated for record in argument_records)
    if len(intervals) == 1:
        start = 0
        stop = intervals[0]
        step = 1
    else:
        if not intervals[0].is_singleton:
            raise _AnalysisError(
                "unbounded-loop",
                "range start must be a compile-time integer",
                expression.lineno,
            )
        start = intervals[0].minimum
        stop = intervals[1]
        step_interval = (
            intervals[2] if len(intervals) == _MAX_RANGE_ARGUMENTS else ClosedIntInterval.point(1)
        )
        if not step_interval.is_singleton or step_interval.minimum <= 0:
            raise _AnalysisError(
                "unbounded-loop",
                "range step must be a positive compile-time integer",
                expression.lineno,
            )
        step = step_interval.minimum
    if stop.minimum < 0:
        raise _AnalysisError(
            "unbounded-loop",
            "range stop must be nonnegative throughout the guarded domain",
            expression.lineno,
        )
    maximum_range = range(start, stop.maximum, step)
    iterations = len(maximum_range)
    if iterations == 0:
        induction = ClosedIntInterval.point(start)
    else:
        last = start + ((iterations - 1) * step)
        induction = ClosedIntInterval.closed(min(start, last), max(start, last))
    proof = OperationProof(
        operation="range-induction",
        operands=intervals,
        status="proved",
        result=induction,
        reasons=(
            OperationReason(
                code="unproven-operation",
                message=f"guarded range executes at most {iterations} iterations",
            ),
        ),
    )
    return induction, iterations, (*records, _record(expression, proof))


def _additive_update(statement: ast.stmt) -> tuple[str | None, ast.expr | None]:
    if (
        isinstance(statement, ast.AugAssign)
        and isinstance(statement.target, ast.Name)
        and isinstance(statement.op, ast.Add)
    ):
        return statement.target.id, statement.value
    if (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and isinstance(statement.value, ast.BinOp)
        and isinstance(statement.value.op, ast.Add)
    ):
        target = statement.targets[0].id
        if isinstance(statement.value.left, ast.Name) and statement.value.left.id == target:
            return target, statement.value.right
        if isinstance(statement.value.right, ast.Name) and statement.value.right.id == target:
            return target, statement.value.left
    return None, None


def _evaluate(
    expression: ast.expr,
    environment: dict[str, ClosedIntInterval],
) -> tuple[ClosedIntInterval, tuple[ScalarOperationRecord, ...]]:
    if isinstance(expression, ast.Constant) and type(expression.value) is int:
        return ClosedIntInterval.point(expression.value), ()
    if isinstance(expression, ast.Name):
        return _lookup(expression, environment), ()
    if isinstance(expression, ast.UnaryOp) and isinstance(expression.op, ast.UAdd):
        return _evaluate(expression.operand, environment)
    if isinstance(expression, ast.UnaryOp) and isinstance(expression.op, ast.USub):
        operand, records = _evaluate(expression.operand, environment)
        proof = subtract(ClosedIntInterval.point(0), operand)
        return _proved_result(proof, expression), (*records, _record(expression, proof))
    if isinstance(expression, ast.BinOp):
        left, left_records = _evaluate(expression.left, environment)
        right, right_records = _evaluate(expression.right, environment)
        proof = _binary_proof(expression.op, left, right, expression)
        return (
            _proved_result(proof, expression),
            (*left_records, *right_records, _record(expression, proof)),
        )
    if isinstance(expression, ast.IfExp):
        _, test_records = _evaluate_condition(expression.test, environment)
        body, body_records = _evaluate(expression.body, environment)
        other, other_records = _evaluate(expression.orelse, environment)
        proof = join(body, other)
        return (
            _proved_result(proof, expression),
            (*test_records, *body_records, *other_records, _record(expression, proof)),
        )
    raise _AnalysisError(
        "unsupported-expression",
        f"{type(expression).__name__} is not supported in scalar arithmetic",
        getattr(expression, "lineno", None),
    )


def _evaluate_condition(
    expression: ast.expr,
    environment: dict[str, ClosedIntInterval],
) -> tuple[ClosedIntInterval, tuple[ScalarOperationRecord, ...]]:
    if (
        isinstance(expression, ast.Compare)
        and len(expression.ops) == len(expression.comparators) == 1
    ):
        left, left_records = _evaluate(expression.left, environment)
        right, right_records = _evaluate(expression.comparators[0], environment)
        operator = _comparison_operator(expression.ops[0], expression)
        proof = compare(left, operator, right)
        return (
            _proved_result(proof, expression),
            (*left_records, *right_records, _record(expression, proof)),
        )
    raise _AnalysisError(
        "unsupported-expression",
        "conditions must be one direct integer comparison",
        getattr(expression, "lineno", None),
    )


def _binary_proof(
    operator: ast.operator,
    left: ClosedIntInterval,
    right: ClosedIntInterval,
    expression: ast.AST,
) -> OperationProof:
    basic = _basic_binary_proof(operator, left, right)
    if basic is not None:
        return basic
    if isinstance(operator, ast.LShift | ast.RShift):
        return shift(left, "<<" if isinstance(operator, ast.LShift) else ">>", right)
    if isinstance(operator, ast.BitAnd | ast.BitOr | ast.BitXor):
        return _bitwise_proof(operator, left, right)
    raise _AnalysisError(
        "unsupported-expression",
        f"{type(operator).__name__} is not supported in scalar arithmetic",
        getattr(expression, "lineno", None),
    )


def _basic_binary_proof(
    operator: ast.operator,
    left: ClosedIntInterval,
    right: ClosedIntInterval,
) -> OperationProof | None:
    if isinstance(operator, ast.Add):
        return add(left, right)
    if isinstance(operator, ast.Sub):
        return subtract(left, right)
    if isinstance(operator, ast.Mult):
        return multiply(left, right)
    if isinstance(operator, ast.FloorDiv | ast.Mod | ast.Pow):
        if isinstance(operator, ast.FloorDiv):
            proof = floor_divide(left, right)
        elif isinstance(operator, ast.Mod):
            proof = modulo(left, right)
        else:
            proof = power(left, right)
        return proof
    return None


def _bitwise_proof(
    operator: ast.BitAnd | ast.BitOr | ast.BitXor,
    left: ClosedIntInterval,
    right: ClosedIntInterval,
) -> OperationProof:
    symbol: Literal["&", "|", "^"]
    if isinstance(operator, ast.BitAnd):
        symbol = "&"
    elif isinstance(operator, ast.BitOr):
        symbol = "|"
    else:
        symbol = "^"
    if symbol != "&" or not right.is_singleton or right.minimum < 0:
        return bitwise(left, symbol, right)
    proof = bitwise(left, symbol, right)
    if proof.proved:
        return proof
    return OperationProof(
        operation="bitwise-&",
        operands=(left, right),
        status="proved",
        result=ClosedIntInterval.closed(0, right.minimum),
        reasons=(
            OperationReason(
                code="unproven-operation",
                message="nonnegative constant mask bounds every result bit",
            ),
        ),
    )


def _proved_result(proof: OperationProof, expression: ast.AST) -> ClosedIntInterval:
    if proof.result is None:
        reason = proof.reasons[0]
        raise _AnalysisError(
            "unproven-arithmetic",
            reason.message,
            getattr(expression, "lineno", None),
        )
    return proof.result


def _record(expression: ast.AST, proof: OperationProof) -> ScalarOperationRecord:
    return ScalarOperationRecord(
        lineno=getattr(expression, "lineno", 0),
        end_lineno=getattr(expression, "end_lineno", getattr(expression, "lineno", 0)),
        col_offset=getattr(expression, "col_offset", 0),
        end_col_offset=getattr(expression, "end_col_offset", None),
        expression=ast.unparse(expression),
        proof=proof,
    )


def _rebase_width_proof(proof: ScalarWidthProof, line_offset: int) -> ScalarWidthProof:
    return replace(
        proof,
        operations=tuple(
            replace(
                operation,
                lineno=operation.lineno + line_offset,
                end_lineno=operation.end_lineno + line_offset,
            )
            for operation in proof.operations
        ),
    )


def _replace_local(
    state: _ProofState,
    name: str,
    value: ClosedIntInterval,
    records: tuple[ScalarOperationRecord, ...],
) -> _ProofState:
    environment = dict(state.environment)
    environment[name] = value
    return _ProofState(
        environment=environment,
        operations=(*state.operations, *records),
        returns=state.returns,
        terminated=state.terminated,
    )


def _lookup(expression: ast.Name, environment: dict[str, ClosedIntInterval]) -> ClosedIntInterval:
    try:
        return environment[expression.id]
    except KeyError as error:
        raise _AnalysisError(
            "unsupported-expression",
            f"name {expression.id} has no proven integer interval",
            expression.lineno,
        ) from error


def _join_intervals(intervals: tuple[ClosedIntInterval, ...]) -> ClosedIntInterval:
    proof = join(*intervals)
    if proof.result is None:
        raise ValueError("scalar interval join requires at least one interval")
    return proof.result


def _join_environments(
    environments: tuple[dict[str, ClosedIntInterval], ...],
) -> dict[str, ClosedIntInterval]:
    common_names: set[str] = set(environments[0])
    for environment in environments[1:]:
        common_names.intersection_update(environment)
    return {
        name: _join_intervals(tuple(environment[name] for environment in environments))
        for name in sorted(common_names)
    }


def _comparison_operator(
    operator: ast.cmpop, expression: ast.AST
) -> Literal["<", "<=", ">", ">=", "==", "!="]:
    if isinstance(operator, ast.Lt):
        return "<"
    if isinstance(operator, ast.LtE):
        return "<="
    if isinstance(operator, ast.Gt):
        return ">"
    if isinstance(operator, ast.GtE):
        return ">="
    if isinstance(operator, ast.Eq):
        return "=="
    if isinstance(operator, ast.NotEq):
        return "!="
    raise _AnalysisError(
        "unsupported-expression",
        f"{type(operator).__name__} comparison is not supported",
        getattr(expression, "lineno", None),
    )


def _simple_name(expression: ast.expr) -> str | None:
    return expression.id if isinstance(expression, ast.Name) else None


def _loaded_names(expression: ast.AST) -> frozenset[str]:
    return frozenset(
        node.id
        for node in ast.walk(expression)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    )


def _literal_integer_constants(scan: ModuleScan) -> dict[str, int]:
    values: dict[str, int] = {}
    for constant in scan.constants:
        if constant.kind != "literal_constant":
            continue
        try:
            statement = ast.parse(constant.source_text).body[0]
        except (SyntaxError, IndexError):
            continue
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            value = ast.literal_eval(statement.value)
        except (ValueError, TypeError):
            continue
        if type(value) is int:
            values[target.id] = value
    return values


def _builtin_range_available(scan: ModuleScan) -> bool:
    if any(constant.name == "range" for constant in scan.constants):
        return False
    if any("range" in imported.imported_names for imported in scan.imports):
        return False
    return not any(
        symbol.kind in {"function", "class"} and symbol.id.qualname == "range"
        for symbol in scan.symbols
    )
