"""Plan and revalidate source-fused run-guard native variants.

The source optimizer emits a Python helper whose fallback always executes the
complete source guard.  This module describes a narrower Cython implementation
that may reuse a successful top-level guard only while a private
run-to-completion protocol remains synchronous.  Source lowering owns cache
invalidation at suspension boundaries; this module owns only immutable plan
identity, staged-source validation, and the backend-neutral region slice.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, replace
from pathlib import PurePosixPath

from atoll.models import (
    LoweringDecision,
    ModuleScan,
    RegionDependency,
    RegionMember,
    SymbolId,
    TypedRegion,
)

RUN_GUARD_LOWERING_VERSION = "run-guard-native-v3"
EXPECTED_ELIGIBILITY_NAME = "__atoll_run_guard_expected_eligibility"
EXPECTED_ELIGIBILITY_CODE_NAME = "__atoll_run_guard_expected_eligibility_code"
EXPECTED_COMPLETION_OWNER_NAME = "__atoll_completion_expected_owner"
EXPECTED_COMPLETION_PREDICATE_NAME = "__atoll_completion_expected_predicate"
EXPECTED_COMPLETION_PREDICATE_CODE_NAME = "__atoll_completion_expected_predicate_code"
_DIGEST_SIZE = 16
_PAIR_SIZE = 2
_QUERY_ARGUMENT_COUNT = 3


@dataclass(frozen=True, slots=True)
class CompletionIndexNativePlan:
    """Native query helpers for one statically indexed completion scan.

    Source lowering owns mutation hooks and maintains the private index in both
    optimized and fallback modes. The native variant replaces only the active
    snapshot and completion predicate, so missing artifacts and ``ATOLL_DISABLE``
    retain the original full-scan behavior.

    Attributes:
        snapshot: Source fallback that materializes the original active-value snapshot.
        query: Source fallback that invokes the original completion predicate.
        index_attribute: Private owner attribute containing run-to-node counts.
        count_attribute: Private owner attribute counting indexed active tasks.
        active_attribute: Direct owner mapping whose values form the fallback snapshot.
        fallback_predicate_method: Original owner predicate used when native routing is disabled.
        graph_attribute: Direct owner field containing the topology object.
        parent_lookup_method: Direct topology method resolving a parent group.
        intermediate_nodes_attribute: Direct parent field containing relevant node IDs.
    """

    snapshot: SymbolId
    query: SymbolId
    index_attribute: str
    count_attribute: str
    active_attribute: str
    fallback_predicate_method: str
    graph_attribute: str
    parent_lookup_method: str
    intermediate_nodes_attribute: str

    def __post_init__(self) -> None:
        """Reject cross-module or expression-shaped completion metadata.

        Raises:
            ValueError: If helper modules differ or rendered attribute names are unsafe.
        """
        if self.snapshot.module != self.query.module:
            raise ValueError("completion-index helpers must belong to one module")
        for symbol in (self.snapshot, self.query):
            if not symbol.qualname.isidentifier():
                raise ValueError("completion-index helpers must be module-level identifiers")
        for name in (
            self.index_attribute,
            self.count_attribute,
            self.active_attribute,
            self.fallback_predicate_method,
            self.graph_attribute,
            self.parent_lookup_method,
            self.intermediate_nodes_attribute,
        ):
            if not name.isidentifier():
                raise ValueError("completion-index attributes must be direct identifiers")

    @property
    def stable_id(self) -> str:
        """Return a deterministic identity for native completion semantics.

        Returns:
            str: Content-derived completion-index identity.
        """
        digest = hashlib.blake2b(digest_size=_DIGEST_SIZE)
        for value in (
            self.snapshot.stable_id,
            self.query.stable_id,
            self.index_attribute,
            self.count_attribute,
            self.active_attribute,
            self.fallback_predicate_method,
            self.graph_attribute,
            self.parent_lookup_method,
            self.intermediate_nodes_attribute,
        ):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
        return f"completion-index-{digest.hexdigest()}"


@dataclass(frozen=True, slots=True)
class RunGuardNativePlan:
    """One source-fused run-guard helper eligible for Cython lowering.

    The plan contains names, never arbitrary source expressions.  Every symbol
    must be a module-level binding in one transformed source module, while the
    state attributes are private fields installed on the exact source owner.

    Attributes:
        source_plan_id: Source-optimization plan that introduced the helper.
        source: Project-relative transformed source path.
        owner: Source owner method whose request path calls the helper.
        helper: Python fallback helper replaced transactionally by the variant.
        source_guard: Complete original source guard retained as fallback.
        eligibility_helper: Per-item callable/code eligibility check.
        protocol_context: Context variable proving private run-to-completion entry.
        disable_module: Module alias used to read ``ATOLL_DISABLE``.
        clear_helper: Source helper that invalidates cached guard state.
        protocol_await_helper: Suspension-aware await helper that invokes
            ``clear_helper`` before yielding control.
        fallback_attribute: Instance field retaining the complete source guard.
        state_attribute: Instance field recording one successful synchronous run guard.
        run_identity_attribute: Instance field identifying the private protocol runner
            whose successful guard may be reused.
        completion_index: Optional source-maintained index whose hot snapshot and
            query helpers are replaced in the same transactional native variant.
        lowering_version: Generator and runtime semantics version used in fingerprints.
    """

    source_plan_id: str
    source: PurePosixPath
    owner: SymbolId
    helper: SymbolId
    source_guard: SymbolId
    eligibility_helper: SymbolId
    protocol_context: SymbolId
    disable_module: SymbolId
    clear_helper: SymbolId
    protocol_await_helper: SymbolId
    fallback_attribute: str
    state_attribute: str
    run_identity_attribute: str
    completion_index: CompletionIndexNativePlan | None = None
    lowering_version: str = RUN_GUARD_LOWERING_VERSION

    def __post_init__(self) -> None:
        """Reject plans that could render arbitrary names or cross modules.

        Raises:
            ValueError: If identities, source paths, module ownership, or private
                attribute names are malformed.
        """
        if not self.source_plan_id.strip() or not self.lowering_version.strip():
            raise ValueError("run-guard plan identities must be non-empty")
        if self.source.is_absolute() or ".." in self.source.parts:
            raise ValueError("run-guard source path must remain project-relative")
        symbols = (
            self.owner,
            self.helper,
            self.source_guard,
            self.eligibility_helper,
            self.protocol_context,
            self.disable_module,
            self.clear_helper,
            self.protocol_await_helper,
        )
        if len({symbol.module for symbol in symbols}) != 1:
            raise ValueError("run-guard plan symbols must belong to one module")
        if (
            self.completion_index is not None
            and self.completion_index.snapshot.module != self.helper.module
        ):
            raise ValueError("completion-index plan belongs to another source module")
        for symbol in symbols[1:]:
            if not symbol.qualname.isidentifier():
                raise ValueError("run-guard helper symbols must be module-level identifiers")
        if "." not in self.owner.qualname:
            raise ValueError("run-guard owner must be a class method")
        for attribute in (
            self.fallback_attribute,
            self.state_attribute,
            self.run_identity_attribute,
        ):
            if not attribute.isidentifier():
                raise ValueError("run-guard state names must be direct attributes")

    @property
    def stable_id(self) -> str:
        """Return a content-derived identity independent of runtime counts.

        Returns:
            str: Stable plan identity included in variant and cache fingerprints.
        """
        digest = hashlib.blake2b(digest_size=_DIGEST_SIZE)
        for value in (
            self.source_plan_id,
            self.source.as_posix(),
            self.owner.stable_id,
            self.helper.stable_id,
            self.source_guard.stable_id,
            self.eligibility_helper.stable_id,
            self.protocol_context.stable_id,
            self.disable_module.stable_id,
            self.clear_helper.stable_id,
            self.protocol_await_helper.stable_id,
            self.fallback_attribute,
            self.state_attribute,
            self.run_identity_attribute,
            self.completion_index.stable_id if self.completion_index is not None else "none",
            self.lowering_version,
        ):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
        return f"run-guard-{digest.hexdigest()}"


def run_guard_function_source(plan: RunGuardNativePlan) -> str:
    """Render the guarded Cython-compatible helper body from structured names.

    The complete Python guard remains the fallback.  Cached calls still execute
    the source eligibility helper for every work item, so callable, descriptor,
    and code-object replacement is observed before native execution continues.

    Args:
        plan: Revalidated source-fused run-guard plan.

    Returns:
        str: One module-level function definition with the source signature.
    """
    return f"""def {plan.helper.qualname}(self, request):
    fallback = self.{plan.fallback_attribute}
    if _atoll_source.{plan.disable_module.qualname}.getenv("ATOLL_DISABLE") == "1":
        return fallback(self, request)
    if _atoll_source.{plan.protocol_context.qualname}.get() is not self:
        return fallback(self, request)
    state_attribute = {plan.state_attribute!r}
    if not getattr(self, state_attribute, False):
        enabled = fallback(self, request)
        if enabled:
            setattr(self, state_attribute, True)
        return enabled
    if type(request) not in (list, tuple):
        return fallback(self, request)
    for item in request:
        if not _atoll_source.{plan.eligibility_helper.qualname}(self, item):
            return fallback(self, request)
    return True"""


def compiled_run_guard_function_source(plan: RunGuardNativePlan) -> str:
    """Render the native helper that calls its colocated eligibility function.

    The generated extension captures the source eligibility callable and code
    object when it imports the transformed module. Every cached invocation
    verifies those identities before entering the copied Cython implementation;
    replacement therefore returns to the complete source guard before any
    transformed side effect.

    Args:
        plan: Revalidated source-fused run-guard plan.

    Returns:
        str: Native helper source referencing the private compiled eligibility body.
    """
    return f"""def {plan.helper.qualname}(self, request):
    fallback = self.{plan.fallback_attribute}
    if _atoll_source.{plan.disable_module.qualname}.getenv("ATOLL_DISABLE") == "1":
        return fallback(self, request)
    if _atoll_source.{plan.protocol_context.qualname}.get() is not self:
        return fallback(self, request)
    source_eligibility = _atoll_source.{plan.eligibility_helper.qualname}
    if source_eligibility is not {EXPECTED_ELIGIBILITY_NAME}:
        return fallback(self, request)
    if getattr(source_eligibility, "__code__", None) is not {EXPECTED_ELIGIBILITY_CODE_NAME}:
        return fallback(self, request)
    state_attribute = {plan.state_attribute!r}
    if not getattr(self, state_attribute, False):
        enabled = fallback(self, request)
        if enabled:
            setattr(self, state_attribute, True)
        return enabled
    if type(request) not in (list, tuple):
        return fallback(self, request)
    for item in request:
        if not {plan.eligibility_helper.qualname}(self, item):
            return fallback(self, request)
    return True"""


def compiled_completion_snapshot_source(plan: CompletionIndexNativePlan) -> str:
    """Render the O(1) snapshot replacement for an indexed completion scan.

    Args:
        plan: Structured completion-index helper and owner metadata.

    Returns:
        str: Native helper source with the exact one-owner fallback signature.
    """
    return f"""def {plan.snapshot.qualname}(owner):
    if type(owner) is not {EXPECTED_COMPLETION_OWNER_NAME}:
        return _atoll_source.{plan.snapshot.qualname}(owner)
    active = getattr(owner, {plan.active_attribute!r})
    if getattr(owner, {plan.count_attribute!r}) != len(active):
        return _atoll_source.{plan.snapshot.qualname}(owner)
    return ()"""


def compiled_completion_query_source(plan: CompletionIndexNativePlan) -> str:
    """Render the indexed completion predicate with boxed Python semantics.

    The source-maintained index maps each run identity to node occurrence
    counts. The native helper keeps Python objects and the GIL, but removes the
    repeated active-task snapshot and nested full scan.

    Args:
        plan: Structured completion-index helper and topology metadata.

    Returns:
        str: Native helper source preserving the source fallback signature.
    """
    return f"""def {plan.query.qualname}(owner, active_tasks, join_id, fork_run_id):
    if type(owner) is not {EXPECTED_COMPLETION_OWNER_NAME}:
        return _atoll_source.{plan.query.qualname}(owner, active_tasks, join_id, fork_run_id)
    predicate = type(owner).__dict__.get({plan.fallback_predicate_method!r})
    if predicate is not {EXPECTED_COMPLETION_PREDICATE_NAME}:
        return _atoll_source.{plan.query.qualname}(owner, active_tasks, join_id, fork_run_id)
    if getattr(predicate, "__code__", None) is not {EXPECTED_COMPLETION_PREDICATE_CODE_NAME}:
        return _atoll_source.{plan.query.qualname}(owner, active_tasks, join_id, fork_run_id)
    active = getattr(owner, {plan.active_attribute!r})
    if getattr(owner, {plan.count_attribute!r}) != len(active):
        return _atoll_source.{plan.query.qualname}(owner, active_tasks, join_id, fork_run_id)
    del active_tasks
    by_node = getattr(owner, {plan.index_attribute!r}).get(fork_run_id)
    if not by_node:
        return True
    parent = getattr(owner, {plan.graph_attribute!r})
    parent = getattr(parent, {plan.parent_lookup_method!r})(join_id)
    if by_node.get(join_id, 0):
        return False
    intermediate_nodes = getattr(parent, {plan.intermediate_nodes_attribute!r})
    return not any(by_node.get(node_id, 0) for node_id in intermediate_nodes)"""


def build_run_guard_region(scan: ModuleScan, plan: RunGuardNativePlan) -> TypedRegion:
    """Revalidate transformed source and return its transactional native slice.

    Args:
        scan: Fresh scan of the staged transformed source module.
        plan: Static plan emitted by the accepted source transformation.

    Returns:
        TypedRegion: Synthetic region whose public helper bindings and private
        eligibility member are covered by transformed source, plan identity,
        and generated Cython helper source.

    Raises:
        ValueError: If the staged helper, fallback route, owner call site,
            invalidation helpers, or source module no longer matches the plan.
    """
    if scan.module.name != plan.helper.module:
        raise ValueError("run-guard plan belongs to a different staged module")
    source = scan.module.path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(scan.module.path))
    _validate_transformed_source(tree, plan)
    selected_symbols = [plan.eligibility_helper, plan.helper]
    native_sources = {plan.helper: compiled_run_guard_function_source(plan)}
    if plan.completion_index is not None:
        selected_symbols.extend((plan.completion_index.snapshot, plan.completion_index.query))
        native_sources.update(
            {
                plan.completion_index.snapshot: compiled_completion_snapshot_source(
                    plan.completion_index
                ),
                plan.completion_index.query: compiled_completion_query_source(
                    plan.completion_index
                ),
            }
        )
    selected = tuple(_region_member(scan, symbol) for symbol in selected_symbols)
    regions = tuple(region for region, _member in selected)
    source_region = regions[1]
    members = tuple(
        replace(member, source_text=native_sources.get(member.id, member.source_text))
        for _region, member in selected
    )
    selected_set = frozenset(selected_symbols)
    bindings = tuple(
        dict.fromkeys(
            binding
            for region in regions
            for binding in region.bindings
            if binding.source in selected_set
        )
    )
    if frozenset(binding.source for binding in bindings) != selected_set:
        raise ValueError("source-fused region requires every selected helper binding")
    decisions = tuple(
        dict.fromkeys(
            decision
            for region in regions
            for decision in region.decisions
            if decision.target in {symbol.stable_id for symbol in selected_symbols}
        )
    ) or (
        LoweringDecision(
            target=plan.helper.stable_id,
            action="box",
            reason="source-fused guard preserves Python objects and exact fallback semantics",
        ),
    )
    dependencies = tuple(
        RegionDependency(
            src=plan.helper,
            dst=destination,
            kind="uses_global",
            confidence="high",
            role="runtime",
            type_only=False,
        )
        for destination in (
            plan.source_guard,
            plan.eligibility_helper,
            plan.protocol_context,
            plan.disable_module,
        )
    ) + tuple(
        dependency
        for region in regions
        for dependency in region.dependencies
        if dependency.src in selected_set
    )
    digest = hashlib.sha256()
    for value in (source, plan.stable_id, *native_sources.values()):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return replace(
        source_region,
        id=f"{scan.module.name}::source-fused:{plan.stable_id}",
        members=members,
        dependencies=tuple(dict.fromkeys(dependencies)),
        type_bindings=(),
        bindings=bindings,
        decisions=decisions,
        source_hash=digest.hexdigest(),
        atomic_class=False,
        specializations=(),
    )


def _region_member(
    scan: ModuleScan,
    symbol: SymbolId,
) -> tuple[TypedRegion, RegionMember]:
    """Return one staged source member with its owning typed region.

    Args:
        scan: Fresh transformed-module scan.
        symbol: Planned module-level helper identity.

    Returns:
        tuple[TypedRegion, RegionMember]: Owning region and retained member.

    Raises:
        ValueError: If the helper is absent or ambiguously repeated.
    """
    matches = tuple(
        (region, member)
        for region in scan.typed_regions
        for member in region.members
        if member.id == symbol
    )
    if len(matches) != 1:
        raise ValueError(f"staged source-fused helper is absent or ambiguous: {symbol.stable_id}")
    return matches[0]


def _validate_transformed_source(tree: ast.Module, plan: RunGuardNativePlan) -> None:
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    required_functions = [
        plan.helper.qualname,
        plan.eligibility_helper.qualname,
        plan.clear_helper.qualname,
        plan.protocol_await_helper.qualname,
    ]
    if plan.completion_index is not None:
        required_functions.extend(
            (
                plan.completion_index.snapshot.qualname,
                plan.completion_index.query.qualname,
            )
        )
    missing = tuple(name for name in required_functions if name not in functions)
    if missing:
        raise ValueError("staged run-guard support is incomplete: " + ", ".join(missing))
    _validate_python_fallback(functions[plan.helper.qualname], plan)
    if plan.completion_index is not None:
        _validate_completion_fallbacks(functions, plan.completion_index)
    bound_names = _module_bound_names(tree)
    for symbol in (plan.source_guard, plan.protocol_context, plan.disable_module):
        if symbol.qualname not in bound_names:
            raise ValueError(f"staged run-guard dependency is absent: {symbol.stable_id}")
    _validate_owner(tree, plan)


def _validate_owner(tree: ast.Module, plan: RunGuardNativePlan) -> None:
    """Verify the transformed owner still owns every planned source route.

    Args:
        tree: Fresh transformed source syntax tree.
        plan: Static run-guard and optional completion-index contract.

    Raises:
        ValueError: If the owner, call site, predicate, or initialized state changed.
    """
    owner_class, owner_method = plan.owner.qualname.split(".", maxsplit=1)
    class_node = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner_class),
        None,
    )
    if class_node is None:
        raise ValueError("staged run-guard owner class is absent")
    owner = next(
        (
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name == owner_method
        ),
        None,
    )
    if owner is None or not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == plan.helper.qualname
        for node in ast.walk(owner)
    ):
        raise ValueError("staged owner no longer calls the planned run-guard helper")
    if plan.completion_index is not None and not any(
        isinstance(method, ast.FunctionDef)
        and method.name == plan.completion_index.fallback_predicate_method
        for method in class_node.body
    ):
        raise ValueError("staged owner lost the original completion predicate")
    initializer = next(
        (
            method
            for method in class_node.body
            if isinstance(method, ast.FunctionDef) and method.name == "__post_init__"
        ),
        None,
    )
    initialized_attributes: frozenset[str] = (
        frozenset(
            attribute
            for node in ast.walk(initializer)
            if (attribute := _assigned_attribute(node)) is not None
        )
        if initializer is not None
        else frozenset()
    )
    required_attributes = {
        plan.fallback_attribute,
        plan.state_attribute,
        plan.run_identity_attribute,
    }
    if plan.completion_index is not None:
        required_attributes.add(plan.completion_index.index_attribute)
        required_attributes.add(plan.completion_index.count_attribute)
    if not required_attributes.issubset(initialized_attributes):
        raise ValueError("staged owner does not initialize complete run-guard state")
    if plan.completion_index is not None and not _initializes_empty_mapping(
        initializer,
        plan.completion_index.index_attribute,
    ):
        raise ValueError("staged owner does not initialize an empty completion index")
    if plan.completion_index is not None and not _initializes_zero(
        initializer,
        plan.completion_index.count_attribute,
    ):
        raise ValueError("staged owner does not initialize a zero completion count")


def _validate_completion_fallbacks(
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
    plan: CompletionIndexNativePlan,
) -> None:
    """Verify source helpers retain the original scan when native routing is absent.

    Args:
        functions: Module-level staged functions indexed by name.
        plan: Structured names for source fallbacks and native replacements.

    Raises:
        ValueError: If either helper changes its signature or no longer delegates
            to the proven active mapping and original completion predicate.
    """
    snapshot = functions[plan.snapshot.qualname]
    if not _has_exact_parameters(snapshot, ("owner",)):
        raise ValueError("completion snapshot fallback signature changed")
    snapshot_return = _single_return_call(snapshot)
    snapshot_values = snapshot_return.args[0] if len(snapshot_return.args) == 1 else None
    if (
        not isinstance(snapshot_return.func, ast.Name)
        or snapshot_return.func.id != "list"
        or snapshot_return.keywords
        or not isinstance(snapshot_values, ast.Call)
        or snapshot_values.args
        or snapshot_values.keywords
        or not isinstance(snapshot_values.func, ast.Attribute)
        or snapshot_values.func.attr != "values"
        or not isinstance(snapshot_values.func.value, ast.Attribute)
        or snapshot_values.func.value.attr != plan.active_attribute
        or not isinstance(snapshot_values.func.value.value, ast.Name)
        or snapshot_values.func.value.value.id != "owner"
    ):
        raise ValueError("completion snapshot fallback no longer scans the active mapping")

    query = functions[plan.query.qualname]
    parameters = ("owner", "active_tasks", "join_id", "fork_run_id")
    if not _has_exact_parameters(query, parameters):
        raise ValueError("completion query fallback signature changed")
    query_return = _single_return_call(query)
    if (
        not isinstance(query_return.func, ast.Attribute)
        or query_return.func.attr != plan.fallback_predicate_method
        or not isinstance(query_return.func.value, ast.Name)
        or query_return.func.value.id != "owner"
        or query_return.keywords
        or len(query_return.args) != _QUERY_ARGUMENT_COUNT
        or not all(isinstance(argument, ast.Name) for argument in query_return.args)
        or tuple(argument.id for argument in query_return.args if isinstance(argument, ast.Name))
        != parameters[1:]
    ):
        raise ValueError("completion query fallback no longer calls the original predicate")


def _has_exact_parameters(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    names: tuple[str, ...],
) -> bool:
    """Return whether a synchronous helper has one exact positional signature.

    Args:
        node: Staged helper declaration being validated.
        names: Required positional parameter names in source order.

    Returns:
        bool: Whether the declaration has only those positional parameters.
    """
    positional = (*node.args.posonlyargs, *node.args.args)
    return (
        isinstance(node, ast.FunctionDef)
        and tuple(parameter.arg for parameter in positional) == names
        and node.args.vararg is None
        and node.args.kwarg is None
        and not node.args.kwonlyargs
        and not node.args.defaults
        and not node.args.kw_defaults
    )


def _single_return_call(node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.Call:
    """Return one direct helper call or reject a changed source fallback.

    Args:
        node: Staged source fallback declaration.

    Returns:
        ast.Call: Sole call returned by the synchronous helper.

    Raises:
        ValueError: If the helper contains another statement or callable shape.
    """
    if (
        not isinstance(node, ast.FunctionDef)
        or len(node.body) != 1
        or not isinstance(node.body[0], ast.Return)
        or not isinstance(node.body[0].value, ast.Call)
    ):
        raise ValueError("completion fallback must remain one synchronous return")
    return node.body[0].value


def _validate_python_fallback(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    plan: RunGuardNativePlan,
) -> None:
    if not isinstance(node, ast.FunctionDef) or len(node.body) != 1:
        raise ValueError("run-guard Python fallback must be one synchronous return")
    positional = (*node.args.posonlyargs, *node.args.args)
    if (
        [parameter.arg for parameter in positional] != ["self", "request"]
        or node.args.vararg is not None
        or node.args.kwarg is not None
        or node.args.kwonlyargs
    ):
        raise ValueError("run-guard Python fallback signature changed")
    statement = node.body[0]
    if not isinstance(statement, ast.Return) or not isinstance(statement.value, ast.Call):
        raise TypeError("run-guard Python fallback no longer delegates directly")
    call = statement.value
    if (
        not isinstance(call.func, ast.Attribute)
        or not isinstance(call.func.value, ast.Name)
        or call.func.value.id != "self"
        or call.func.attr != plan.fallback_attribute
        or len(call.args) != _PAIR_SIZE
        or not all(isinstance(argument, ast.Name) for argument in call.args)
        or [argument.id for argument in call.args if isinstance(argument, ast.Name)]
        != ["self", "request"]
        or call.keywords
    ):
        raise ValueError("run-guard Python fallback no longer calls the retained source guard")


def _module_bound_names(tree: ast.Module) -> frozenset[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(
                alias.asname or alias.name.split(".", maxsplit=1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names if alias.name != "*")
        elif isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
    return frozenset(names)


def _assigned_attribute(node: ast.AST) -> str | None:
    target: ast.expr | None = None
    if isinstance(node, ast.AnnAssign):
        target = node.target
    elif isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
    return target.attr if isinstance(target, ast.Attribute) else None


def _initializes_empty_mapping(node: ast.FunctionDef | None, attribute: str) -> bool:
    """Return whether an initializer assigns one literal empty private mapping.

    Args:
        node: Owner initializer, when one exists.
        attribute: Private mapping attribute required by the plan.

    Returns:
        bool: Whether exactly one literal empty-dict assignment initializes the field.
    """
    if node is None:
        return False
    values: list[ast.expr | None] = []
    for candidate in ast.walk(node):
        if _assigned_attribute(candidate) != attribute:
            continue
        if isinstance(candidate, ast.AnnAssign | ast.Assign):
            values.append(candidate.value)
    return len(values) == 1 and isinstance(values[0], ast.Dict) and not values[0].keys


def _initializes_zero(node: ast.FunctionDef | None, attribute: str) -> bool:
    """Return whether an initializer assigns one literal zero private count.

    Args:
        node: Owner initializer, when one exists.
        attribute: Private count attribute required by the plan.

    Returns:
        bool: Whether exactly one literal-zero assignment initializes the field.
    """
    if node is None:
        return False
    values: list[ast.expr | None] = []
    for candidate in ast.walk(node):
        if _assigned_attribute(candidate) != attribute:
            continue
        if isinstance(candidate, ast.AnnAssign | ast.Assign):
            values.append(candidate.value)
    return len(values) == 1 and isinstance(values[0], ast.Constant) and values[0].value == 0
