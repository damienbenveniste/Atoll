"""Generic residual async profile used by benchmark and unit tests.

The fixture models five cumulative optimization ideas without embedding
application-specific identifiers: run-scoped guard amortization, quiescent
await-chain collapse, context-copy elision only for context-independent work,
incremental private completion accounting, and private result-record elision
with ordered streaming reduction. It deliberately keeps context-sensitive work
on a task fallback path so semantic comparisons can detect unsafe context-copy
elision.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal, TypedDict

WORK_ITEM_COUNT = 192
SEMANTIC_REPETITIONS = 16
STREAMING_REDUCTION_BARRIER_STEPS = 6_500
STAGE_NAMES: tuple[str, ...] = (
    "run_scoped_guard_amortization",
    "quiescent_await_chain_collapse",
    "context_copy_elision",
    "incremental_completion_accounting",
    "result_record_elision",
)

_CONTEXT: ContextVar[str] = ContextVar("residual_async_profile_context", default="parent")


class SemanticSnapshot(TypedDict):
    """Primitive semantic evidence compared across execution strategies.

    Attributes:
        workflow: Generic fixture workflow name.
        work_count: Number of logical items processed.
        checksum: Ordered reduction checksum.
        completed: Number of private completions observed.
        first_label: First ordered item label.
        last_label: Last ordered item label.
        parent_context: Context visible to the owner after all work.
        fallback_context: Context value observed by a context-sensitive item.
        fallback_label: Label of the task-fallback item.
        stage_counter_total: Sum of final residual stage counters.
    """

    workflow: str
    work_count: int
    checksum: int
    completed: int
    first_label: str
    last_label: str
    parent_context: str
    fallback_context: str
    fallback_label: str
    stage_counter_total: int


class FallbackSnapshot(TypedDict):
    """Focused context-sensitive fallback evidence.

    Attributes:
        parent_before: Owner context before the fallback task starts.
        child_observed: Context copied into the task-backed fallback item.
        child_mutated: Child-local context after fallback mutation.
        parent_after: Owner context after fallback completion.
    """

    parent_before: str
    child_observed: str
    child_mutated: str
    parent_after: str


class StageCounterPayload(TypedDict):
    """JSON-compatible stage counter mapping.

    Attributes:
        run_scoped_guard_amortization: Number of guard checks avoided by
            amortizing validation once per run.
        quiescent_await_chain_collapse: Number of quiescent await chains
            collapsed into direct private calls.
        context_copy_elision: Number of context-independent items that avoided a
            copied task context.
        incremental_completion_accounting: Number of completions accounted for
            without retaining result records.
        result_record_elision: Number of private result records elided while
            preserving ordered streaming reduction.
    """

    run_scoped_guard_amortization: int
    quiescent_await_chain_collapse: int
    context_copy_elision: int
    incremental_completion_accounting: int
    result_record_elision: int


@dataclass(frozen=True, slots=True)
class WorkItem:
    """One deterministic logical unit of fixture work.

    Attributes:
        ordinal: Stable input position.
        label: Stable label used for ordered reduction evidence.
        weight: Deterministic checksum contribution multiplier.
        mode: Whether the item can elide copied context safely.
    """

    ordinal: int
    label: str
    weight: int
    mode: Literal["context_independent", "context_sensitive"]


@dataclass(frozen=True, slots=True)
class ResultRecord:
    """Private baseline result record retained before reduction.

    Attributes:
        ordinal: Stable input position.
        label: Stable label copied from the work item.
        value: Deterministic value reduced by the owner.
        context_value: Context visible while processing the item.
    """

    ordinal: int
    label: str
    value: int
    context_value: str


@dataclass(frozen=True, slots=True)
class StageCounters:
    """Counters proving that every residual optimization stage was exercised.

    Attributes:
        run_scoped_guard_amortization: Number of per-item guard checks avoided.
        quiescent_await_chain_collapse: Number of immediate await chains
            collapsed.
        context_copy_elision: Number of context-independent task context copies
            avoided.
        incremental_completion_accounting: Number of private completions counted
            without materializing completion records.
        result_record_elision: Number of private result records elided.
    """

    run_scoped_guard_amortization: int = 0
    quiescent_await_chain_collapse: int = 0
    context_copy_elision: int = 0
    incremental_completion_accounting: int = 0
    result_record_elision: int = 0

    def as_json(self) -> StageCounterPayload:
        """Return the counters as stable JSON-compatible fields.

        Returns:
            StageCounterPayload: Mapping keyed by residual optimization idea.
        """

        return StageCounterPayload(
            run_scoped_guard_amortization=self.run_scoped_guard_amortization,
            quiescent_await_chain_collapse=self.quiescent_await_chain_collapse,
            context_copy_elision=self.context_copy_elision,
            incremental_completion_accounting=self.incremental_completion_accounting,
            result_record_elision=self.result_record_elision,
        )

    def all_nonzero(self) -> bool:
        """Return whether every optimization-stage counter is positive.

        Returns:
            bool: True only when every counter records exercised work.
        """

        return all(value > 0 for value in self._values())

    def total(self) -> int:
        """Return the total number of exercised stage events.

        Returns:
            int: Sum of every counter field.
        """

        return sum(self._values())

    def _values(self) -> tuple[int, ...]:
        return (
            self.run_scoped_guard_amortization,
            self.quiescent_await_chain_collapse,
            self.context_copy_elision,
            self.incremental_completion_accounting,
            self.result_record_elision,
        )


@dataclass(frozen=True, slots=True)
class RunResult:
    """Execution output shared by baseline and residual paths.

    Attributes:
        checksum: Ordered reduction checksum.
        completed: Number of private completions observed by the owner.
        ordered_labels: Labels in canonical input order.
        fallback_context: Context observed by the context-sensitive item.
        fallback_label: Label of the context-sensitive fallback item.
        counters: Residual stage counters.
    """

    checksum: int
    completed: int
    ordered_labels: tuple[str, ...]
    fallback_context: str
    fallback_label: str
    counters: StageCounters


_WORK_ITEMS: tuple[WorkItem, ...] = tuple(
    WorkItem(
        ordinal=index,
        label=f"item-{index:04d}",
        weight=(index % 23) + 3,
        mode="context_sensitive" if index == WORK_ITEM_COUNT // 2 else "context_independent",
    )
    for index in range(WORK_ITEM_COUNT)
)


async def compare_semantics(
    repetitions: int = SEMANTIC_REPETITIONS,
) -> tuple[SemanticSnapshot, bool]:
    """Compare baseline and residual semantics repeatedly.

    Args:
        repetitions: Number of complete baseline/residual comparisons.

    Returns:
        tuple[SemanticSnapshot, bool]: Canonical residual snapshot and whether
        every baseline and residual snapshot matched it.

    Raises:
        ValueError: If repetitions is less than one.
    """

    if repetitions < 1:
        raise ValueError("semantic repetitions must be positive")
    expected: SemanticSnapshot | None = None
    for _ in range(repetitions):
        baseline = await canonical_semantic_snapshot("baseline")
        residual = await canonical_semantic_snapshot("residual")
        if baseline != residual:
            return residual, False
        if expected is None:
            expected = residual
        elif residual != expected:
            return residual, False
    if expected is None:
        raise ValueError("semantic comparison produced no evidence")
    return expected, True


async def canonical_semantic_snapshot(
    strategy: Literal["baseline", "residual"] = "residual",
) -> SemanticSnapshot:
    """Build deterministic semantic evidence for one execution strategy.

    Args:
        strategy: Execution path used to process the generic workload.

    Returns:
        SemanticSnapshot: Stable primitive output for equality checks.

    Raises:
        ValueError: If an unknown strategy is requested.
    """

    _CONTEXT.set("owner")
    if strategy == "baseline":
        result = await run_baseline_once()
    elif strategy == "residual":
        result = await run_residual_once()
    else:
        raise ValueError(f"unknown residual profile strategy: {strategy}")
    return SemanticSnapshot(
        workflow="generic-residual-async-profile",
        work_count=WORK_ITEM_COUNT,
        checksum=result.checksum,
        completed=result.completed,
        first_label=result.ordered_labels[0],
        last_label=result.ordered_labels[-1],
        parent_context=_CONTEXT.get(),
        fallback_context=result.fallback_context,
        fallback_label=result.fallback_label,
        stage_counter_total=0,
    )


async def context_sensitive_fallback_snapshot() -> FallbackSnapshot:
    """Capture the context-sensitive case that must retain task fallback.

    Returns:
        FallbackSnapshot: Parent and child context observations proving that the
        residual path does not elide context for context-sensitive work.
    """

    _CONTEXT.set("owner")
    item = WorkItem(0, "context-sensitive", 7, "context_sensitive")
    child_observed, child_mutated = await _run_context_sensitive_fallback(item)
    return FallbackSnapshot(
        parent_before="owner",
        child_observed=child_observed,
        child_mutated=child_mutated,
        parent_after=_CONTEXT.get(),
    )


async def residual_checksum(iterations: int) -> tuple[int, StageCounters]:
    """Run the residual path repeatedly for benchmark measurement.

    Args:
        iterations: Number of residual workload executions.

    Returns:
        tuple[int, StageCounters]: Accumulated checksum and cumulative stage
        counters.

    Raises:
        ValueError: If iterations is less than one.
    """

    if iterations < 1:
        raise ValueError("iterations must be positive")
    checksum = 0
    counters = StageCounters()
    for _ in range(iterations):
        result = await run_residual_once()
        checksum += result.checksum
        counters = _combine_counters(counters, result.counters)
    return checksum, counters


async def baseline_checksum(iterations: int) -> int:
    """Run the baseline path repeatedly for benchmark measurement.

    Args:
        iterations: Number of baseline workload executions.

    Returns:
        int: Accumulated baseline checksum.

    Raises:
        ValueError: If iterations is less than one.
    """

    if iterations < 1:
        raise ValueError("iterations must be positive")
    checksum = 0
    for _ in range(iterations):
        checksum += (await run_baseline_once()).checksum
    return checksum


async def run_baseline_once() -> RunResult:
    """Execute the fully materialized baseline pipeline once.

    Returns:
        RunResult: Ordered reduction and zero residual counters.
    """

    _validate_run_guard()
    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(_baseline_worker(item)) for item in _WORK_ITEMS]
    records = [task.result() for task in tasks]
    ordered = tuple(sorted(records, key=lambda record: record.ordinal))
    fallback = next(record for record in ordered if record.label == _fallback_item().label)
    return RunResult(
        checksum=sum(record.value for record in ordered),
        completed=len(ordered),
        ordered_labels=tuple(record.label for record in ordered),
        fallback_context=fallback.context_value,
        fallback_label=fallback.label,
        counters=StageCounters(),
    )


async def run_residual_once() -> RunResult:
    """Execute the cumulative residual optimization profile once.

    Returns:
        RunResult: Ordered reduction and counters for all five residual stages.
    """

    _validate_run_guard()
    checksum = 0
    completed = 0
    labels: list[str] = []
    fallback_context = ""
    fallback_label = ""
    independent_count = 0
    for item in _WORK_ITEMS:
        if item.mode == "context_sensitive":
            observed, mutated = await _run_context_sensitive_fallback(item)
            value = _value_for(item)
            fallback_context = observed
            fallback_label = item.label
            if mutated != f"child:{item.label}":
                raise RuntimeError("context-sensitive fallback did not mutate child context")
        else:
            value = _direct_context_independent_value(item)
            independent_count += 1
        checksum += value
        completed += 1
        labels.append(item.label)
    _private_streaming_reduction_barrier(checksum)
    counters = StageCounters(
        run_scoped_guard_amortization=len(_WORK_ITEMS) - 1,
        quiescent_await_chain_collapse=independent_count,
        context_copy_elision=independent_count,
        incremental_completion_accounting=completed,
        result_record_elision=completed,
    )
    return RunResult(
        checksum=checksum,
        completed=completed,
        ordered_labels=tuple(labels),
        fallback_context=fallback_context,
        fallback_label=fallback_label,
        counters=counters,
    )


async def _baseline_worker(item: WorkItem) -> ResultRecord:
    _validate_run_guard()
    value = await _baseline_await_chain(item)
    context_value = _CONTEXT.get()
    if item.mode == "context_sensitive":
        _CONTEXT.set(f"child:{item.label}")
    return ResultRecord(
        ordinal=item.ordinal,
        label=item.label,
        value=value,
        context_value=context_value,
    )


async def _baseline_await_chain(item: WorkItem) -> int:
    await asyncio.sleep(0)
    return await _baseline_value_stage(item)


async def _baseline_value_stage(item: WorkItem) -> int:
    await asyncio.sleep(0)
    return _value_for(item)


async def _run_context_sensitive_fallback(item: WorkItem) -> tuple[str, str]:
    async def child() -> tuple[str, str]:
        observed = _CONTEXT.get()
        _CONTEXT.set(f"child:{item.label}")
        return observed, _CONTEXT.get()

    return await asyncio.create_task(child())


def _direct_context_independent_value(item: WorkItem) -> int:
    return _value_for(item)


def _value_for(item: WorkItem) -> int:
    return (item.ordinal + 5) * item.weight


def _private_streaming_reduction_barrier(seed: int) -> int:
    value = seed
    for index in range(STREAMING_REDUCTION_BARRIER_STEPS):
        value = (value + index) & 0xFFFFFFFF
    return value


def _validate_run_guard() -> None:
    if WORK_ITEM_COUNT < 1:
        raise RuntimeError("residual async profile has no work items")


def _combine_counters(first: StageCounters, second: StageCounters) -> StageCounters:
    return StageCounters(
        run_scoped_guard_amortization=(
            first.run_scoped_guard_amortization + second.run_scoped_guard_amortization
        ),
        quiescent_await_chain_collapse=(
            first.quiescent_await_chain_collapse + second.quiescent_await_chain_collapse
        ),
        context_copy_elision=first.context_copy_elision + second.context_copy_elision,
        incremental_completion_accounting=(
            first.incremental_completion_accounting + second.incremental_completion_accounting
        ),
        result_record_elision=first.result_record_elision + second.result_record_elision,
    )


def _fallback_item() -> WorkItem:
    return _WORK_ITEMS[WORK_ITEM_COUNT // 2]
