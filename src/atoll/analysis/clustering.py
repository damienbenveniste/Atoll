"""Conservative island clustering and scoring for Atoll scans."""

from __future__ import annotations

from dataclasses import replace

from atoll.analysis.call_graph import build_dependency_edges
from atoll.models import (
    Blocker,
    ConstantRecord,
    DependencyEdge,
    IslandCandidate,
    IslandRisk,
    ModuleScan,
    PoisonRadius,
    SymbolId,
    SymbolRecord,
)

_MIN_SYMBOL_SCORE = 60
_MIN_CLUSTER_SCORE = 70
_MIN_CLUSTER_LINES = 3
_HIGH_CONFIDENCE_SCORE = 90
_MODULE_COMPILE_BLOCKERS = frozenset({"MYPYC_UNSUPPORTED_TYPEVAR"})


def enrich_island_analysis(module: ModuleScan) -> ModuleScan:
    """Attach dependency edges, island candidates, and poison-radius records."""
    symbols = _attach_dynamic_global_blockers(module)
    module_with_blockers = replace(module, symbols=symbols)
    edges = build_dependency_edges(module_with_blockers)
    symbols = _attach_blocked_local_call_blockers(module_with_blockers, edges)
    module_with_blockers = replace(module_with_blockers, symbols=symbols)
    candidates = _cluster_candidates(module_with_blockers, edges)
    poison_radii = _poison_radii(module_with_blockers, edges, candidates)
    return replace(
        module_with_blockers,
        dependency_edges=edges,
        island_candidates=candidates,
        poison_radii=poison_radii,
    )


def _attach_dynamic_global_blockers(module: ModuleScan) -> tuple[SymbolRecord, ...]:
    constants = {constant.name: constant for constant in module.constants}
    imported_names = {
        imported_name for record in module.imports for imported_name in record.imported_names
    }
    local_symbols = {symbol.id.qualname for symbol in module.symbols}
    return tuple(
        _attach_symbol_dynamic_global_blockers(symbol, constants, imported_names, local_symbols)
        for symbol in module.symbols
    )


def _attach_symbol_dynamic_global_blockers(
    symbol: SymbolRecord,
    constants: dict[str, ConstantRecord],
    imported_names: set[str],
    local_symbols: set[str],
) -> SymbolRecord:
    blockers = tuple(
        Blocker(
            severity="hard",
            code="DYNAMIC_GLOBAL_DEP",
            message=f"global {name!r} is not a safe literal constant, import, or local symbol",
            lineno=symbol.lineno,
            symbol=symbol.id,
        )
        for name in symbol.uses_globals
        if _is_dynamic_global(name, constants, imported_names, local_symbols)
    )
    if not blockers:
        return symbol
    return replace(symbol, blockers=(*symbol.blockers, *blockers))


def _is_dynamic_global(
    name: str,
    constants: dict[str, ConstantRecord],
    imported_names: set[str],
    local_symbols: set[str],
) -> bool:
    if name in imported_names or name in local_symbols:
        return False
    if name in constants:
        return constants[name].kind != "literal_constant"
    return True


def _attach_blocked_local_call_blockers(
    module: ModuleScan,
    edges: tuple[DependencyEdge, ...],
) -> tuple[SymbolRecord, ...]:
    by_id = {symbol.id: symbol for symbol in module.symbols}
    blockers_by_id = {symbol.id: list(symbol.blockers) for symbol in module.symbols}
    blocked = {
        symbol.id
        for symbol in module.symbols
        if any(blocker.severity == "hard" for blocker in symbol.blockers)
    }
    changed = True
    while changed:
        changed = False
        for edge in edges:
            if edge.kind != "calls" or not isinstance(edge.dst, SymbolId):
                continue
            if edge.src in blocked or edge.dst not in blocked:
                continue
            source = by_id[edge.src]
            blockers_by_id[edge.src].append(
                Blocker(
                    severity="hard",
                    code="LOCAL_BLOCKED_DEP",
                    message=f"calls blocked local symbol {edge.dst.qualname!r}",
                    lineno=edge.lineno or source.lineno,
                    symbol=source.id,
                )
            )
            blocked.add(edge.src)
            changed = True
    return tuple(
        replace(symbol, blockers=tuple(blockers_by_id[symbol.id])) for symbol in module.symbols
    )


def _cluster_candidates(
    module: ModuleScan,
    edges: tuple[DependencyEdge, ...],
) -> tuple[IslandCandidate, ...]:
    if _module_blocks_compilation(module):
        return ()
    symbols = {symbol.id: symbol for symbol in module.symbols}
    clean_function_ids = {
        symbol.id
        for symbol in module.symbols
        if symbol.kind == "function" and _symbol_score(symbol) >= _MIN_SYMBOL_SCORE
    }
    candidates: list[IslandCandidate] = []
    seen_clusters: set[tuple[str, ...]] = set()
    for seed in sorted(clean_function_ids, key=lambda symbol: symbol.stable_id):
        cluster = _expand_cluster(seed, clean_function_ids, edges)
        cluster_key = tuple(sorted(symbol.stable_id for symbol in cluster))
        if cluster_key in seen_clusters:
            continue
        seen_clusters.add(cluster_key)
        candidate = _candidate_from_cluster(module, symbols, cluster, edges)
        if candidate.score >= _MIN_CLUSTER_SCORE:
            candidates.append(candidate)
    return _maximal_candidates(tuple(candidates))


def _expand_cluster(
    seed: SymbolId,
    clean_function_ids: set[SymbolId],
    edges: tuple[DependencyEdge, ...],
) -> tuple[SymbolId, ...]:
    cluster = {seed}
    changed = True
    while changed:
        changed = False
        for edge in edges:
            if (
                edge.kind != "calls"
                or edge.confidence != "high"
                or not isinstance(edge.dst, SymbolId)
            ):
                continue
            if edge.src in cluster and edge.dst in clean_function_ids and edge.dst not in cluster:
                cluster.add(edge.dst)
                changed = True
    return tuple(sorted(cluster, key=lambda symbol: symbol.stable_id))


def _candidate_from_cluster(
    module: ModuleScan,
    symbols: dict[SymbolId, SymbolRecord],
    cluster: tuple[SymbolId, ...],
    edges: tuple[DependencyEdge, ...],
) -> IslandCandidate:
    cluster_symbols = tuple(symbols[symbol_id] for symbol_id in cluster)
    score = _cluster_score(cluster_symbols)
    risk = _cluster_risk(cluster_symbols, edges)
    constants = tuple(
        sorted(
            {
                str(edge.dst)
                for edge in edges
                if edge.src in cluster and edge.kind == "uses_global" and isinstance(edge.dst, str)
            }
        )
    )
    reasons = _candidate_reasons(cluster_symbols, score)
    return IslandCandidate(
        source_module=module.module,
        symbols=cluster,
        required_imports=_required_imports(cluster, edges),
        required_constants=constants,
        required_local_symbols=cluster,
        rejected_symbols=_rejected_symbols(module),
        score=score,
        risk=risk,
        reasons=reasons,
    )


def _maximal_candidates(candidates: tuple[IslandCandidate, ...]) -> tuple[IslandCandidate, ...]:
    return tuple(
        candidate
        for candidate in candidates
        if not any(
            set(candidate.symbols) < set(other.symbols)
            for other in candidates
            if candidate is not other
        )
    )


def _symbol_score(symbol: SymbolRecord) -> int:
    score = 0
    if not _has_hard_blocker(symbol):
        score += 25
    if not any(blocker.code == "MYPY_ERROR" for blocker in symbol.blockers):
        score += 20
    if symbol.has_return_annotation and symbol.arg_count == symbol.annotated_arg_count:
        score += 15
    if not any(blocker.code == "DYNAMIC_GLOBAL_DEP" for blocker in symbol.blockers):
        score += 10
    if symbol.called_names:
        score += 10
    if _symbol_line_count(symbol) < _MIN_CLUSTER_LINES:
        score -= 10
    return score


def _cluster_score(symbols: tuple[SymbolRecord, ...]) -> int:
    if not symbols:
        return 0
    score = sum(_symbol_score(symbol) for symbol in symbols) // len(symbols)
    line_count = sum(_symbol_line_count(symbol) for symbol in symbols)
    if line_count >= _MIN_CLUSTER_LINES:
        score += 10
    return min(score, 100)


def _cluster_risk(
    symbols: tuple[SymbolRecord, ...],
    edges: tuple[DependencyEdge, ...],
) -> IslandRisk:
    symbol_ids = {symbol.id for symbol in symbols}
    if any(edge.confidence == "low" and edge.src in symbol_ids for edge in edges):
        return "medium"
    return "low"


def _candidate_reasons(symbols: tuple[SymbolRecord, ...], score: int) -> tuple[str, ...]:
    reasons = ["no hard blockers", "typed signatures"]
    if len(symbols) > 1:
        reasons.append("connected same-module call cluster")
    if score >= _HIGH_CONFIDENCE_SCORE:
        reasons.append("high confidence candidate")
    return tuple(reasons)


def _module_blocks_compilation(module: ModuleScan) -> bool:
    return any(
        blocker.severity == "hard" and blocker.code in _MODULE_COMPILE_BLOCKERS
        for blocker in module.blockers
    )


def _required_imports(
    cluster: tuple[SymbolId, ...],
    edges: tuple[DependencyEdge, ...],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(edge.dst)
                for edge in edges
                if edge.src in cluster and edge.kind == "imports" and isinstance(edge.dst, str)
            }
        )
    )


def _rejected_symbols(module: ModuleScan) -> tuple[SymbolId, ...]:
    return tuple(
        symbol.id
        for symbol in module.symbols
        if symbol.kind == "function"
        and (_has_hard_blocker(symbol) or _symbol_score(symbol) < _MIN_SYMBOL_SCORE)
    )


def _poison_radii(
    module: ModuleScan,
    edges: tuple[DependencyEdge, ...],
    candidates: tuple[IslandCandidate, ...],
) -> tuple[PoisonRadius, ...]:
    candidate_symbols = {symbol for candidate in candidates for symbol in candidate.symbols}
    poisons = _rejected_symbols(module)
    radii: list[PoisonRadius] = []
    for poison in poisons:
        impacted = tuple(
            sorted(
                {
                    edge.src
                    for edge in edges
                    if edge.dst == poison and edge.src in candidate_symbols
                },
                key=lambda symbol: symbol.stable_id,
            )
        )
        radii.append(
            PoisonRadius(
                poison=poison,
                impacted=impacted,
                reason="hard blocker" if impacted else "isolated residue",
            )
        )
    return tuple(radii)


def _has_hard_blocker(symbol: SymbolRecord) -> bool:
    return any(blocker.severity == "hard" for blocker in symbol.blockers)


def _symbol_line_count(symbol: SymbolRecord) -> int:
    return symbol.end_lineno - symbol.lineno + 1
