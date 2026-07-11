"""Measure semantics and headroom for callback-backed async execution plans.

This benchmark is deliberately independent from Atoll's product pipeline. It
models one shared capacity-one completion channel and compares ordinary tasks,
task-preserving dispatch, and callback-backed dispatch. The callback arm only
drives source-known non-suspending coroutines; suspending or task-observing work
retains a real task. The harness owns semantic evidence and feasibility timing,
not production eligibility or wheel generation.
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import json
import sys
import time
from collections import deque
from collections.abc import Callable, Coroutine, Iterable, Sequence
from dataclasses import dataclass
from statistics import median
from typing import Literal, TypedDict, Unpack, cast

ArmName = Literal["baseline", "task_preserving", "callback_backed"]
WorkMode = Literal["immediate", "suspending", "failure", "task_observing", "cold_decoy"]
ExecutionPath = Literal["task", "task_preserving", "callback", "task_fallback"]
ResultStatus = Literal["ok", "error"]
JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

ARMS: tuple[ArmName, ...] = ("baseline", "task_preserving", "callback_backed")
DEFAULT_MINIMUM_SPEEDUP = 1.50
DEFAULT_SEMANTIC_REPETITIONS = 32
DEFAULT_BENCHMARK_WIDTH = 5_000
DEFAULT_WARMUPS = 1
DEFAULT_SAMPLES = 7
MINIMUM_STABLE_SECONDS = 0.25
MINIMUM_WORKLOAD_SIZE = 6
MAX_CALIBRATION_ROUNDS = 1_024

_RUN_CONTEXT: contextvars.ContextVar[str] = contextvars.ContextVar(
    "atoll_execution_plan_context",
    default="parent",
)


class FeasibilityError(RuntimeError):
    """Raised when feasibility evidence violates the benchmark contract."""


@dataclass(frozen=True, slots=True)
class WorkItem:
    """One deterministic unit of semantic-fixture work.

    Attributes:
        name: Stable identity used in traces and results.
        order: Registration order and deterministic reduction key.
        value: Integer payload returned by successful work.
        mode: Execution shape used by strict callback eligibility.
    """

    name: str
    order: int
    value: int
    mode: WorkMode


@dataclass(frozen=True, slots=True)
class ErrorEvidence:
    """Stable exception evidence retained without addresses or traceback text.

    Attributes:
        type_name: Concrete exception class name.
        message: Exception message.
        cause_type: Explicit cause class name.
        notes: Stable PEP 678 exception notes.
        work_frame_present: Whether the traceback retained the work function.
    """

    type_name: str
    message: str
    cause_type: str | None
    notes: tuple[str, ...]
    work_frame_present: bool


@dataclass(frozen=True, slots=True)
class ResultRecord:
    """One observable logical-work result delivered through the shared channel.

    Attributes:
        name: Stable work identity.
        order: Registration order.
        status: Successful or controlled-error result status.
        value: Successful integer value.
        error: Stable exception evidence for failures.
        starting_context: Context captured independently for the logical item.
        mutated_context: Child-only context value after indirect mutation.
        task_identity_observed: Whether explicitly task-observing work saw a task.
    """

    name: str
    order: int
    status: ResultStatus
    value: int | None
    error: ErrorEvidence | None
    starting_context: str
    mutated_context: str
    task_identity_observed: bool | None


@dataclass(frozen=True, slots=True)
class ArmExecution:
    """Deterministic semantic evidence from one scheduler arm.

    Attributes:
        arm: Scheduler strategy used for the run.
        records: Results in completion-channel receive order.
        paths: Per-item execution route, excluded from semantic equivalence.
        trace: Stable registration, work, publish, and receive events.
        parent_context_after: Parent context after all child cleanup.
        active_after_cleanup: Remaining tracked task count.
        channel_max_depth: Maximum occupied shared-channel slots.
        blocked_publications: Number of producers blocked by capacity one.
        cold_decoys: Work items intentionally never scheduled.
    """

    arm: ArmName
    records: tuple[ResultRecord, ...]
    paths: tuple[tuple[str, ExecutionPath], ...]
    trace: tuple[str, ...]
    parent_context_after: str
    active_after_cleanup: int
    channel_max_depth: int
    blocked_publications: int
    cold_decoys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    """Wall-clock samples and median for one feasibility arm.

    Attributes:
        arm: Scheduler strategy represented by the samples.
        sample_seconds: Measured elapsed durations after warmup.
        median_seconds: Median measured duration.
        speedup_over_baseline: Baseline median divided by this median.
    """

    arm: ArmName
    sample_seconds: tuple[float, ...]
    median_seconds: float
    speedup_over_baseline: float


@dataclass(frozen=True, slots=True)
class FeasibilityReport:
    """Complete semantic and performance feasibility verdict.

    Attributes:
        semantic_repetitions: Number of repeated semantic comparisons.
        semantic_snapshot: Canonical baseline-equivalent semantic evidence.
        semantics_match: Whether every repetition and arm matched baseline.
        benchmark_width: Immediate work items scheduled per benchmark round.
        benchmark_rounds: Calibrated rounds per measured arm invocation.
        benchmark_summaries: Rotating wall-clock evidence for every arm.
        callback_speedup: Callback-backed median speedup over baseline.
        minimum_callback_speedup: Required feasibility threshold.
        stable_timings: Whether every median reached the noise floor.
        gate_passed: Combined semantic, stability, and speed verdict.
    """

    semantic_repetitions: int
    semantic_snapshot: dict[str, JsonValue]
    semantics_match: bool
    benchmark_width: int
    benchmark_rounds: int
    benchmark_summaries: tuple[BenchmarkSummary, ...]
    callback_speedup: float
    minimum_callback_speedup: float
    stable_timings: bool
    gate_passed: bool


@dataclass(frozen=True, slots=True)
class FeasibilityOptions:
    """Validated semantic and performance policy for one feasibility run.

    Attributes:
        semantic_repetitions: Complete cross-arm semantic comparisons.
        benchmark_width: Immediate logical items per benchmark round.
        warmups: Unmeasured rotating benchmark groups.
        samples: Measured rotating benchmark groups.
        minimum_callback_speedup: Required callback-to-baseline ratio.
        minimum_stable_seconds: Required median duration for every arm.
    """

    semantic_repetitions: int = DEFAULT_SEMANTIC_REPETITIONS
    benchmark_width: int = DEFAULT_BENCHMARK_WIDTH
    warmups: int = DEFAULT_WARMUPS
    samples: int = DEFAULT_SAMPLES
    minimum_callback_speedup: float = DEFAULT_MINIMUM_SPEEDUP
    minimum_stable_seconds: float = MINIMUM_STABLE_SECONDS


DEFAULT_FEASIBILITY_OPTIONS = FeasibilityOptions()


class _BenchmarkJson(TypedDict):
    arms: list[dict[str, JsonValue]]
    callback_speedup: float
    rounds: int
    stable_timings: bool
    width: int


class _ReportJson(TypedDict):
    benchmark: _BenchmarkJson
    gate_passed: bool
    minimum_callback_speedup: float
    semantic_repetitions: int
    semantic_snapshot: dict[str, JsonValue]
    semantics_match: bool


class _TaskFactoryOptions(TypedDict, total=False):
    name: str | None
    context: contextvars.Context | None
    eager_start: bool


@dataclass(slots=True)
class _PendingSend:
    record: ResultRecord
    completion: asyncio.Future[None]


class _CapacityOneChannel:
    """Shared asynchronous capacity-one channel with cancellable senders."""

    def __init__(self, trace: list[str]) -> None:
        self._trace = trace
        self._slot: ResultRecord | None = None
        self._pending: deque[_PendingSend] = deque()
        self._receiver: asyncio.Future[ResultRecord] | None = None
        self.max_depth = 0
        self.blocked_publications = 0
        self.closed = False

    def begin_send(self, record: ResultRecord) -> asyncio.Future[None]:
        """Begin publishing a record and return its completion future.

        Args:
            record: Result to deliver in producer-completion order.

        Returns:
            asyncio.Future[None]: Completed immediately when delivery is
                possible, otherwise completed when a receiver frees capacity.

        Raises:
            FeasibilityError: If publication starts after channel closure.
        """
        if self.closed:
            raise FeasibilityError("cannot publish to a closed result channel")
        loop = asyncio.get_running_loop()
        completion: asyncio.Future[None] = loop.create_future()
        if self._receiver is not None:
            receiver = self._receiver
            self._receiver = None
            receiver.set_result(record)
            completion.set_result(None)
            self._trace.append(f"published:{record.name}")
            return completion
        if self._slot is None:
            self._slot = record
            self.max_depth = 1
            completion.set_result(None)
            self._trace.append(f"published:{record.name}")
            return completion
        self._pending.append(_PendingSend(record=record, completion=completion))
        self.blocked_publications += 1
        self._trace.append(f"publish-blocked:{record.name}")
        return completion

    async def send(self, record: ResultRecord) -> None:
        """Publish a record with capacity-one backpressure.

        Args:
            record: Result to deliver.

        Raises:
            asyncio.CancelledError: If the producer is cancelled while blocked.
        """
        completion = self.begin_send(record)
        try:
            await completion
        except asyncio.CancelledError:
            self.cancel_send(completion)
            raise

    async def receive(self) -> ResultRecord:
        """Receive the next record and release one blocked producer.

        Returns:
            ResultRecord: Next record in publication order.

        Raises:
            FeasibilityError: If a second receiver is registered or the closed
                channel has no remaining data.
        """
        if self._slot is not None:
            record = self._slot
            self._slot = None
            self._promote_pending()
            self._trace.append(f"received:{record.name}")
            return record
        if self.closed:
            raise FeasibilityError("cannot receive from an empty closed channel")
        if self._receiver is not None:
            raise FeasibilityError("capacity-one channel supports one consumer")
        loop = asyncio.get_running_loop()
        self._receiver = loop.create_future()
        record = await self._receiver
        self._trace.append(f"received:{record.name}")
        return record

    def cancel_send(self, completion: asyncio.Future[None]) -> bool:
        """Remove one blocked publication before it becomes visible.

        Args:
            completion: Future returned by :meth:`begin_send`.

        Returns:
            bool: Whether a blocked publication was removed.
        """
        for pending in tuple(self._pending):
            if pending.completion is completion:
                self._pending.remove(pending)
                if not completion.done():
                    completion.cancel()
                self._trace.append(f"publish-cancelled:{pending.record.name}")
                return True
        return False

    def close(self) -> None:
        """Close an empty channel after all logical producers finish.

        Raises:
            FeasibilityError: If records, blocked senders, or a receiver remain.
        """
        if self._slot is not None or self._pending or self._receiver is not None:
            raise FeasibilityError("result channel closed with pending state")
        self.closed = True
        self._trace.append("channel-closed")

    def _promote_pending(self) -> None:
        if not self._pending:
            return
        pending = self._pending.popleft()
        self._slot = pending.record
        self.max_depth = 1
        if not pending.completion.done():
            pending.completion.set_result(None)
        self._trace.append(f"published:{pending.record.name}")


@dataclass(slots=True)
class _PendingBenchmarkSend:
    value: int
    completion: asyncio.Future[None]


class _BenchmarkChannel:
    """Minimal capacity-one integer channel for scheduler-only timing."""

    def __init__(self) -> None:
        self._slot = 0
        self._occupied = False
        self._pending: deque[_PendingBenchmarkSend] = deque()
        self._receiver: asyncio.Future[int] | None = None

    def begin_send(self, value: int) -> asyncio.Future[None]:
        """Begin one benchmark publication without diagnostic allocations.

        Args:
            value: Integer checksum contribution.

        Returns:
            asyncio.Future[None]: Publication completion future.
        """
        loop = asyncio.get_running_loop()
        completion: asyncio.Future[None] = loop.create_future()
        if self._receiver is not None:
            receiver = self._receiver
            self._receiver = None
            receiver.set_result(value)
            completion.set_result(None)
        elif not self._occupied:
            self._slot = value
            self._occupied = True
            completion.set_result(None)
        else:
            self._pending.append(_PendingBenchmarkSend(value, completion))
        return completion

    async def send(self, value: int) -> None:
        """Publish one integer with capacity-one backpressure.

        Args:
            value: Integer checksum contribution.
        """
        await self.begin_send(value)

    def send_callback(self, value: int, completion: asyncio.Future[None]) -> None:
        """Publish from a callback without allocating a second future.

        Args:
            value: Integer checksum contribution.
            completion: Logical callback completion owned by the caller.
        """
        if self._receiver is not None:
            receiver = self._receiver
            self._receiver = None
            receiver.set_result(value)
            completion.set_result(None)
        elif not self._occupied:
            self._slot = value
            self._occupied = True
            completion.set_result(None)
        else:
            self._pending.append(_PendingBenchmarkSend(value, completion))

    async def receive(self) -> int:
        """Receive one integer and release a blocked producer.

        Returns:
            int: Next checksum contribution.
        """
        if self._occupied:
            value = self._slot
            if self._pending:
                pending = self._pending.popleft()
                self._slot = pending.value
                pending.completion.set_result(None)
            else:
                self._occupied = False
            return value
        loop = asyncio.get_running_loop()
        self._receiver = loop.create_future()
        return await self._receiver

    def verify_empty(self) -> None:
        """Reject leaked benchmark channel state after a measured round.

        Raises:
            FeasibilityError: If a value, producer, or receiver remains.
        """
        if self._occupied or self._pending or self._receiver is not None:
            raise FeasibilityError("benchmark channel retained pending state")


class _ControlledWorkError(RuntimeError):
    """Controlled failure used to compare exception evidence."""


def main(argv: tuple[str, ...] | None = None) -> int:
    """Run the complete feasibility gate and emit canonical JSON evidence.

    Args:
        argv: Optional CLI arguments instead of ``sys.argv``.

    Returns:
        int: Zero only when semantics, timing stability, and speed pass.
    """
    args = _parse_args(tuple(sys.argv[1:] if argv is None else argv))
    try:
        report = run_feasibility(
            FeasibilityOptions(
                semantic_repetitions=args.semantic_repetitions,
                benchmark_width=args.benchmark_width,
                warmups=args.warmups,
                samples=args.samples,
                minimum_callback_speedup=args.minimum_speedup,
                minimum_stable_seconds=args.minimum_stable_seconds,
            )
        )
    except (FeasibilityError, ValueError) as error:
        print(f"async execution-plan feasibility failed: {error}", file=sys.stderr)
        return 1
    print(canonical_json(report_as_json(report)))
    if report.gate_passed:
        return 0
    print(
        "async execution-plan feasibility gate failed: "
        f"callback={report.callback_speedup:.3f}x, "
        f"required={report.minimum_callback_speedup:.3f}x, "
        f"semantics={report.semantics_match}, stable={report.stable_timings}",
        file=sys.stderr,
    )
    return 1


def run_feasibility(
    options: FeasibilityOptions = DEFAULT_FEASIBILITY_OPTIONS,
) -> FeasibilityReport:
    """Run repeated semantics and a calibrated wall-clock benchmark.

    Args:
        options: Semantic repetition, benchmark sizing, and promotion policy.

    Returns:
        FeasibilityReport: Combined evidence and promotion verdict.

    Raises:
        ValueError: If counts or thresholds are invalid.
        FeasibilityError: If deterministic semantics diverge.
    """
    _validate_options(options)
    semantic_snapshot, semantics_match = compare_semantics(options.semantic_repetitions)
    rounds = calibrate_benchmark_rounds(
        width=options.benchmark_width,
        minimum_seconds=options.minimum_stable_seconds,
    )
    summaries = benchmark(
        width=options.benchmark_width,
        rounds=rounds,
        warmups=options.warmups,
        samples=options.samples,
    )
    callback_speedup = _summary_for(summaries, "callback_backed").speedup_over_baseline
    stable = all(summary.median_seconds >= options.minimum_stable_seconds for summary in summaries)
    return FeasibilityReport(
        semantic_repetitions=options.semantic_repetitions,
        semantic_snapshot=semantic_snapshot,
        semantics_match=semantics_match,
        benchmark_width=options.benchmark_width,
        benchmark_rounds=rounds,
        benchmark_summaries=summaries,
        callback_speedup=callback_speedup,
        minimum_callback_speedup=options.minimum_callback_speedup,
        stable_timings=stable,
        gate_passed=semantics_match
        and stable
        and callback_speedup >= options.minimum_callback_speedup,
    )


def build_workload(workload_size: int = MINIMUM_WORKLOAD_SIZE) -> tuple[WorkItem, ...]:
    """Create mixed semantic work plus two unscheduled cold decoys.

    Args:
        workload_size: Number of runnable work items.

    Returns:
        tuple[WorkItem, ...]: Stable mixed-shape workload.

    Raises:
        ValueError: If the requested workload cannot cover every required shape.
    """
    if workload_size < MINIMUM_WORKLOAD_SIZE:
        raise ValueError(f"workload_size must be at least {MINIMUM_WORKLOAD_SIZE}")
    modes: tuple[WorkMode, ...] = (
        "immediate",
        "suspending",
        "immediate",
        "failure",
        "task_observing",
        "immediate",
    )
    runnable = tuple(
        WorkItem(
            name=f"work-{index:03d}",
            order=index,
            value=(index + 1) * 3,
            mode=modes[index % len(modes)],
        )
        for index in range(workload_size)
    )
    return (
        *runnable,
        WorkItem("cold-decoy-a", workload_size, -1, "cold_decoy"),
        WorkItem("cold-decoy-b", workload_size + 1, -2, "cold_decoy"),
    )


def compare_semantics(repetitions: int) -> tuple[dict[str, JsonValue], bool]:
    """Compare every arm against baseline repeatedly.

    Args:
        repetitions: Number of complete comparison groups.

    Returns:
        tuple[dict[str, JsonValue], bool]: Canonical baseline snapshot and
            whether every arm and repetition matched it.

    Raises:
        ValueError: If repetitions is not positive.
    """
    if repetitions < 1:
        raise ValueError("semantic repetitions must be positive")
    expected: dict[str, JsonValue] | None = None
    matched = True
    workload = build_workload()
    for _ in range(repetitions):
        for arm in ARMS:
            execution = asyncio.run(execute_arm(arm, workload))
            snapshot = semantic_snapshot(execution)
            suite = asyncio.run(run_semantic_probes(arm))
            combined: dict[str, JsonValue] = {
                "execution": snapshot,
                "probes": suite,
            }
            if expected is None:
                expected = combined
            elif combined != expected:
                matched = False
    if expected is None:
        raise FeasibilityError("semantic comparison produced no evidence")
    return expected, matched


async def execute_arm(
    arm: ArmName,
    workload: tuple[WorkItem, ...],
    *,
    force_task_fallback: bool = False,
) -> ArmExecution:
    """Execute one mixed workload through a selected scheduler arm.

    Args:
        arm: Scheduler strategy to exercise.
        workload: Runnable items plus optional cold decoys.
        force_task_fallback: Disable callback execution as a runtime guard would.

    Returns:
        ArmExecution: Stable records, trace, routes, and cleanup evidence.
    """
    trace: list[str] = []
    channel = _CapacityOneChannel(trace)
    runnable = tuple(item for item in workload if item.mode != "cold_decoy")
    cold_decoys = tuple(item.name for item in workload if item.mode == "cold_decoy")
    paths: list[tuple[str, ExecutionPath]] = []
    tasks: list[asyncio.Task[None]] = []
    callback_completions: list[asyncio.Future[None]] = []
    token = _RUN_CONTEXT.set("parent")
    try:
        callback_allowed = (
            arm == "callback_backed"
            and not force_task_fallback
            and asyncio.get_running_loop().get_task_factory() is None
        )
        for item in runnable:
            trace.append(f"registered:{item.name}")
            item_context = _item_context(item)
            if callback_allowed and _callback_eligible(item):
                completion = _schedule_callback_item(item, item_context, channel, trace)
                callback_completions.append(completion)
                paths.append((item.name, "callback"))
                continue
            task = _create_task(
                _task_producer(item, channel, trace),
                name=item.name,
                context=item_context,
            )
            tasks.append(task)
            if arm == "baseline":
                paths.append((item.name, "task"))
            elif arm == "task_preserving":
                paths.append((item.name, "task_preserving"))
            else:
                paths.append((item.name, "task_fallback"))
        records = tuple([await channel.receive() for _ in runnable])
        if tasks:
            await asyncio.gather(*tasks)
        if callback_completions:
            await asyncio.gather(*callback_completions)
        channel.close()
        return ArmExecution(
            arm=arm,
            records=records,
            paths=tuple(paths),
            trace=tuple(trace),
            parent_context_after=_RUN_CONTEXT.get(),
            active_after_cleanup=sum(not task.done() for task in tasks),
            channel_max_depth=channel.max_depth,
            blocked_publications=channel.blocked_publications,
            cold_decoys=cold_decoys,
        )
    finally:
        _RUN_CONTEXT.reset(token)


def semantic_snapshot(execution: ArmExecution) -> dict[str, JsonValue]:
    """Normalize arm-independent observable semantics.

    Args:
        execution: Completed arm evidence.

    Returns:
        dict[str, JsonValue]: Canonical JSON-compatible semantics excluding
            implementation paths and arm identity.
    """
    ordered = tuple(sorted(execution.records, key=lambda record: record.order))
    completion_order = [record.name for record in execution.records]
    values = [record.value for record in ordered if record.value is not None]
    return {
        "active_after_cleanup": execution.active_after_cleanup,
        "blocked_publications": execution.blocked_publications,
        "capacity_one": execution.channel_max_depth == 1,
        "cold_decoys": list(execution.cold_decoys),
        "completion_order": [cast(JsonValue, name) for name in completion_order],
        "context_isolated": execution.parent_context_after == "parent"
        and all(
            record.starting_context == f"scheduled:{record.name}"
            and record.mutated_context == f"child:{record.name}"
            for record in ordered
        ),
        "deterministic_reduction": sum(values),
        "records": [_record_json(record) for record in ordered],
        "trace": list(execution.trace),
    }


async def run_semantic_probes(arm: ArmName) -> dict[str, JsonValue]:
    """Exercise cancellation, blocked publication, and task-factory fallback.

    Args:
        arm: Arm under test.

    Returns:
        dict[str, JsonValue]: Stable probe evidence comparable across arms.
    """
    cancellation = await _cancellation_probe()
    blocked = await _blocked_publication_probe()
    factory = await _task_factory_probe(arm)
    return {
        "blocked_publication": blocked,
        "cancellation": cancellation,
        "custom_factory": factory,
    }


def benchmark(
    *,
    width: int,
    rounds: int,
    warmups: int,
    samples: int,
) -> tuple[BenchmarkSummary, ...]:
    """Measure arms in rotating wall-clock order.

    Args:
        width: Immediate logical items per round.
        rounds: Repeated rounds per measured invocation.
        warmups: Unmeasured rotating groups.
        samples: Measured rotating groups.

    Returns:
        tuple[BenchmarkSummary, ...]: Per-arm samples, medians, and speedups.

    Raises:
        ValueError: If counts are invalid.
    """
    if width < 1 or rounds < 1 or warmups < 0 or samples < 1:
        raise ValueError("benchmark width, rounds, and samples must be positive")
    durations: dict[ArmName, list[float]] = {arm: [] for arm in ARMS}
    for group_index in range(warmups + samples):
        for arm in rotated_arms(group_index):
            duration = _measure_benchmark_arm(arm, width=width, rounds=rounds)
            if group_index >= warmups:
                durations[arm].append(duration)
    baseline_median = median(durations["baseline"])
    return tuple(
        BenchmarkSummary(
            arm=arm,
            sample_seconds=tuple(durations[arm]),
            median_seconds=median(durations[arm]),
            speedup_over_baseline=baseline_median / median(durations[arm]),
        )
        for arm in ARMS
    )


def calibrate_benchmark_rounds(*, width: int, minimum_seconds: float) -> int:
    """Double benchmark work until every arm reaches the timing floor.

    Args:
        width: Immediate items per round.
        minimum_seconds: Required elapsed duration for every arm.

    Returns:
        int: Calibrated rounds per measured arm invocation.

    Raises:
        FeasibilityError: If the timing floor cannot be reached safely.
    """
    if minimum_seconds <= 0:
        return 1
    rounds = 1
    while rounds <= MAX_CALIBRATION_ROUNDS:
        durations = tuple(_measure_benchmark_arm(arm, width=width, rounds=rounds) for arm in ARMS)
        if min(durations) >= minimum_seconds:
            return rounds
        rounds *= 2
    raise FeasibilityError(
        f"benchmark could not reach {minimum_seconds:.3f}s with {MAX_CALIBRATION_ROUNDS} rounds"
    )


def summarize_samples(
    samples: dict[ArmName, Sequence[float]],
) -> tuple[BenchmarkSummary, ...]:
    """Summarize externally supplied samples for deterministic unit tests.

    Args:
        samples: Non-empty elapsed samples for every arm.

    Returns:
        tuple[BenchmarkSummary, ...]: Summaries in stable arm order.

    Raises:
        ValueError: If an arm has no positive samples.
    """
    for arm in ARMS:
        if not samples.get(arm) or any(value <= 0 for value in samples[arm]):
            raise ValueError(f"positive samples are required for {arm}")
    baseline_median = median(samples["baseline"])
    return tuple(
        BenchmarkSummary(
            arm=arm,
            sample_seconds=tuple(samples[arm]),
            median_seconds=median(samples[arm]),
            speedup_over_baseline=baseline_median / median(samples[arm]),
        )
        for arm in ARMS
    )


def rotated_arms(group_index: int) -> tuple[ArmName, ...]:
    """Return a deterministic rotation for one benchmark group.

    Args:
        group_index: Zero-based warmup or sample group.

    Returns:
        tuple[ArmName, ...]: Rotated arm order.
    """
    offset = group_index % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def canonical_json(payload: object) -> str:
    """Serialize stable evidence without addresses or formatting variance.

    Args:
        payload: JSON-compatible evidence.

    Returns:
        str: Compact key-sorted JSON.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def report_as_json(report: FeasibilityReport) -> _ReportJson:
    """Convert a feasibility report into its stable JSON contract.

    Args:
        report: Completed feasibility evidence.

    Returns:
        _ReportJson: JSON-compatible report.
    """
    return {
        "benchmark": {
            "arms": [_summary_json(summary) for summary in report.benchmark_summaries],
            "callback_speedup": report.callback_speedup,
            "rounds": report.benchmark_rounds,
            "stable_timings": report.stable_timings,
            "width": report.benchmark_width,
        },
        "gate_passed": report.gate_passed,
        "minimum_callback_speedup": report.minimum_callback_speedup,
        "semantic_repetitions": report.semantic_repetitions,
        "semantic_snapshot": report.semantic_snapshot,
        "semantics_match": report.semantics_match,
    }


async def _task_producer(
    item: WorkItem,
    channel: _CapacityOneChannel,
    trace: list[str],
) -> None:
    trace.append(f"started:{item.name}")
    record = await _capture_result(item)
    trace.append(f"completed:{item.name}")
    await channel.send(record)


def _schedule_callback_item(
    item: WorkItem,
    item_context: contextvars.Context,
    channel: _CapacityOneChannel,
    trace: list[str],
) -> asyncio.Future[None]:
    loop = asyncio.get_running_loop()
    logical_completion: asyncio.Future[None] = loop.create_future()
    coroutine = item_context.run(_capture_result, item)
    loop.call_soon(
        _drive_callback_item,
        item,
        coroutine,
        channel,
        trace,
        logical_completion,
        context=item_context,
    )
    return logical_completion


def _drive_callback_item(
    item: WorkItem,
    coroutine: Coroutine[object, object, ResultRecord],
    channel: _CapacityOneChannel,
    trace: list[str],
    logical_completion: asyncio.Future[None],
) -> None:
    trace.append(f"started:{item.name}")
    try:
        coroutine.send(None)
    except StopIteration as completed:
        record = completed.value
    except BaseException as error:
        logical_completion.set_exception(error)
        return
    else:
        coroutine.close()
        logical_completion.set_exception(
            FeasibilityError(f"callback-eligible work suspended unexpectedly: {item.name}")
        )
        return
    trace.append(f"completed:{item.name}")
    publication = channel.begin_send(record)
    if publication.done():
        logical_completion.set_result(None)
    else:
        publication.add_done_callback(
            lambda _future: _complete_logical_callback(logical_completion)
        )


def _complete_logical_callback(completion: asyncio.Future[None]) -> None:
    if not completion.done():
        completion.set_result(None)


def _create_task(
    coroutine: Coroutine[object, object, None],
    *,
    name: str,
    context: contextvars.Context,
) -> asyncio.Task[None]:
    """Create a task while honoring custom factories and captured context.

    Args:
        coroutine: Producer coroutine to schedule.
        name: Stable task name.
        context: Context snapshot belonging to the logical work item.

    Returns:
        asyncio.Task[None]: Real task created through the active loop policy.
    """
    if asyncio.get_running_loop().get_task_factory() is None:
        return asyncio.create_task(coroutine, name=name, context=context)
    return context.run(asyncio.create_task, coroutine, name=name)


async def _capture_result(item: WorkItem) -> ResultRecord:
    starting_context = _RUN_CONTEXT.get()
    token = _RUN_CONTEXT.set(f"child:{item.name}")
    try:
        try:
            value, task_observed = await _run_work(item)
        except _ControlledWorkError as error:
            return ResultRecord(
                name=item.name,
                order=item.order,
                status="error",
                value=None,
                error=_error_evidence(error),
                starting_context=starting_context,
                mutated_context=_RUN_CONTEXT.get(),
                task_identity_observed=None,
            )
        return ResultRecord(
            name=item.name,
            order=item.order,
            status="ok",
            value=value,
            error=None,
            starting_context=starting_context,
            mutated_context=_RUN_CONTEXT.get(),
            task_identity_observed=task_observed,
        )
    finally:
        _RUN_CONTEXT.reset(token)


async def _run_work(item: WorkItem) -> tuple[int, bool | None]:
    _mutate_context_indirectly(item.name)
    if item.mode == "suspending":
        await asyncio.sleep(0)
    if item.mode == "failure":
        _raise_controlled_error(item.name)
    if item.mode == "task_observing":
        return item.value * 2, asyncio.current_task() is not None
    if item.mode == "cold_decoy":
        raise FeasibilityError(f"cold decoy executed: {item.name}")
    return item.value * 2, None


def _mutate_context_indirectly(name: str) -> None:
    _RUN_CONTEXT.set(f"child:{name}")


def _raise_controlled_error(name: str) -> None:
    try:
        _raise_controlled_cause(name)
    except LookupError as cause:
        error = _ControlledWorkError(f"failure:{name}")
        error.add_note(f"note:{name}")
        raise error from cause


def _raise_controlled_cause(name: str) -> None:
    raise LookupError(f"cause:{name}")


def _error_evidence(error: _ControlledWorkError) -> ErrorEvidence:
    traceback_names: list[str] = []
    traceback = error.__traceback__
    while traceback is not None:
        traceback_names.append(traceback.tb_frame.f_code.co_name)
        traceback = traceback.tb_next
    return ErrorEvidence(
        type_name=type(error).__name__,
        message=str(error),
        cause_type=type(error.__cause__).__name__ if error.__cause__ is not None else None,
        notes=tuple(error.__notes__ if hasattr(error, "__notes__") else ()),
        work_frame_present="_run_work" in traceback_names,
    )


def _item_context(item: WorkItem) -> contextvars.Context:
    context = contextvars.copy_context()
    context.run(_RUN_CONTEXT.set, f"scheduled:{item.name}")
    return context


def _callback_eligible(item: WorkItem) -> bool:
    return item.mode in {"immediate", "failure"}


async def _cancellation_probe() -> dict[str, JsonValue]:
    entered = asyncio.Event()
    cleanup_count = 0

    async def cancellable() -> None:
        nonlocal cleanup_count
        try:
            entered.set()
            await asyncio.Event().wait()
        finally:
            cleanup_count += 1

    task = asyncio.create_task(cancellable())
    await entered.wait()
    task.cancel()
    cancelled = False
    try:
        await task
    except asyncio.CancelledError:
        cancelled = True
    return {
        "cancelled": cancelled,
        "cleanup_count": cleanup_count,
        "task_done": task.done(),
    }


async def _blocked_publication_probe() -> dict[str, JsonValue]:
    trace: list[str] = []
    channel = _CapacityOneChannel(trace)
    first = _probe_record("first", 0)
    second = _probe_record("second", 1)
    first_completion = channel.begin_send(first)
    second_completion = channel.begin_send(second)
    second_was_blocked = not second_completion.done()
    first_received = (await channel.receive()).name
    second_released = second_completion.done()
    second_received = (await channel.receive()).name
    channel.close()

    cancel_trace: list[str] = []
    cancel_channel = _CapacityOneChannel(cancel_trace)
    cancel_channel.begin_send(first)
    cancelled_completion = cancel_channel.begin_send(second)
    removed = cancel_channel.cancel_send(cancelled_completion)
    retained = (await cancel_channel.receive()).name
    cancel_channel.close()
    return {
        "cancelled_blocked_send": removed and cancelled_completion.cancelled(),
        "first_completion_immediate": first_completion.done(),
        "first_received": first_received,
        "retained_after_cancel": retained,
        "second_received": second_received,
        "second_released_after_drain": second_released,
        "second_was_blocked": second_was_blocked,
    }


async def _task_factory_probe(arm: ArmName) -> dict[str, JsonValue]:
    loop = asyncio.get_running_loop()
    factory_calls = 0

    def factory(
        task_loop: asyncio.AbstractEventLoop,
        coroutine: Coroutine[object, object, object],
        /,
        **options: Unpack[_TaskFactoryOptions],
    ) -> asyncio.Task[object]:
        nonlocal factory_calls
        factory_calls += 1
        return asyncio.Task(
            coroutine,
            loop=task_loop,
            name=options.get("name"),
            context=options.get("context"),
            eager_start=options.get("eager_start", False),
        )

    previous = loop.get_task_factory()
    set_factory = cast(Callable[[object], None], loop.set_task_factory)
    set_factory(factory)
    try:
        workload = (WorkItem("factory-work", 0, 1, "immediate"),)
        execution = await execute_arm(arm, workload, force_task_fallback=True)
    finally:
        set_factory(previous)
    return {
        "factory_calls": factory_calls,
        "real_task_path": dict(execution.paths)["factory-work"] != "callback",
        "result": execution.records[0].value,
    }


def _probe_record(name: str, order: int) -> ResultRecord:
    return ResultRecord(
        name=name,
        order=order,
        status="ok",
        value=order,
        error=None,
        starting_context=f"scheduled:{name}",
        mutated_context=f"child:{name}",
        task_identity_observed=None,
    )


def _measure_benchmark_arm(arm: ArmName, *, width: int, rounds: int) -> float:
    started = time.perf_counter()
    checksum = asyncio.run(_benchmark_arm(arm, width=width, rounds=rounds))
    duration = time.perf_counter() - started
    expected_round = width * (width + 1) // 2
    if checksum != expected_round * rounds:
        raise FeasibilityError(
            f"benchmark checksum mismatch for {arm}: {checksum} != {expected_round * rounds}"
        )
    return duration


async def _benchmark_arm(arm: ArmName, *, width: int, rounds: int) -> int:
    checksum = 0
    for _ in range(rounds):
        checksum += await _benchmark_round(arm, width)
    return checksum


async def _benchmark_round(arm: ArmName, width: int) -> int:
    channel = _BenchmarkChannel()
    tasks, completions = _schedule_benchmark_work(arm, width, channel)
    total = 0
    for _ in range(width):
        total += await channel.receive()
    if tasks:
        await asyncio.gather(*tasks)
    if completions:
        await asyncio.gather(*completions)
    channel.verify_empty()
    return total


def _schedule_benchmark_work(
    arm: ArmName,
    width: int,
    channel: _BenchmarkChannel,
) -> tuple[list[asyncio.Task[None]], list[asyncio.Future[None]]]:
    tasks: list[asyncio.Task[None]] = []
    completions: list[asyncio.Future[None]] = []
    loop = asyncio.get_running_loop()
    for value in range(width):
        if arm == "callback_backed":
            context = contextvars.copy_context()
            completion: asyncio.Future[None] = loop.create_future()
            completions.append(completion)
            coroutine = context.run(_benchmark_immediate, value)
            loop.call_soon(
                _benchmark_callback,
                coroutine,
                channel,
                completion,
                context=context,
            )
        else:
            tasks.append(asyncio.create_task(_benchmark_producer(value, channel)))
    return tasks, completions


async def _benchmark_immediate(value: int) -> int:
    return value + 1


async def _benchmark_producer(value: int, channel: _BenchmarkChannel) -> None:
    await channel.send(await _benchmark_immediate(value))


def _benchmark_callback(
    coroutine: Coroutine[object, object, int],
    channel: _BenchmarkChannel,
    completion: asyncio.Future[None],
) -> None:
    try:
        coroutine.send(None)
    except StopIteration as completed:
        channel.send_callback(completed.value, completion)
    else:
        coroutine.close()
        completion.set_exception(FeasibilityError("benchmark leaf suspended"))


def _record_json(record: ResultRecord) -> dict[str, JsonValue]:
    return {
        "error": _error_json(record.error),
        "mutated_context": record.mutated_context,
        "name": record.name,
        "order": record.order,
        "starting_context": record.starting_context,
        "status": record.status,
        "task_identity_observed": record.task_identity_observed,
        "value": record.value,
    }


def _error_json(error: ErrorEvidence | None) -> dict[str, JsonValue] | None:
    if error is None:
        return None
    return {
        "cause_type": error.cause_type,
        "message": error.message,
        "notes": list(error.notes),
        "type_name": error.type_name,
        "work_frame_present": error.work_frame_present,
    }


def _summary_json(summary: BenchmarkSummary) -> dict[str, JsonValue]:
    return {
        "arm": summary.arm,
        "median_seconds": summary.median_seconds,
        "sample_seconds": list(summary.sample_seconds),
        "speedup_over_baseline": summary.speedup_over_baseline,
    }


def _summary_for(
    summaries: Iterable[BenchmarkSummary],
    arm: ArmName,
) -> BenchmarkSummary:
    for summary in summaries:
        if summary.arm == arm:
            return summary
    raise FeasibilityError(f"missing benchmark summary for {arm}")


def _validate_options(options: FeasibilityOptions) -> None:
    if options.semantic_repetitions < 1:
        raise ValueError("semantic repetitions must be positive")
    if options.benchmark_width < 1:
        raise ValueError("benchmark width must be positive")
    if options.warmups < 0 or options.samples < 1:
        raise ValueError("warmups must be non-negative and samples positive")
    if options.minimum_callback_speedup <= 1:
        raise ValueError("minimum callback speedup must be greater than one")
    if options.minimum_stable_seconds < 0:
        raise ValueError("minimum stable seconds must be non-negative")


def _parse_args(argv: tuple[str, ...]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the generic async execution-plan feasibility gate.",
    )
    parser.add_argument(
        "--semantic-repetitions",
        type=int,
        default=DEFAULT_SEMANTIC_REPETITIONS,
    )
    parser.add_argument("--benchmark-width", type=int, default=DEFAULT_BENCHMARK_WIDTH)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--minimum-speedup", type=float, default=DEFAULT_MINIMUM_SPEEDUP)
    parser.add_argument(
        "--minimum-stable-seconds",
        type=float,
        default=MINIMUM_STABLE_SECONDS,
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
