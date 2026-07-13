"""Run the generic residual async profile hard benchmark."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
import time
from collections.abc import Coroutine, Mapping
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Literal, Protocol, TypedDict, cast

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "residual_async_profile"
DEFAULT_WARMUPS = 1
DEFAULT_SAMPLES = 7
DEFAULT_MINIMUM_SECONDS = 0.25
DEFAULT_MINIMUM_SPEEDUP = 3.0
DEFAULT_SEMANTIC_REPETITIONS = 8
DEFAULT_ITERATIONS = 2
MAX_CALIBRATION_ITERATIONS = 4096
CALIBRATION_HEADROOM_FACTOR = 1.25

ArmName = Literal["baseline", "residual"]
JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class BenchmarkError(RuntimeError):
    """Raised when benchmark policy or evidence is invalid."""


class StageCountersProtocol(Protocol):
    """Residual stage counter surface used by the benchmark.

    The fixture owns the concrete counter implementation; the script only needs
    the stable JSON conversion used for evidence aggregation.
    """

    def as_json(self) -> Mapping[str, int]:
        """Return JSON-compatible residual stage counters.

        Returns:
            Mapping[str, int]: Counter values keyed by stage name.
        """
        ...


class FixtureModule(Protocol):
    """Fixture module surface used by this standalone benchmark.

    Attributes:
        STAGE_NAMES: Stable residual optimization stage identifiers.
    """

    STAGE_NAMES: tuple[str, ...]

    def compare_semantics(
        self,
        repetitions: int,
    ) -> Coroutine[object, object, tuple[dict[str, JsonValue], bool]]:
        """Compare baseline and residual semantics.

        Args:
            repetitions: Number of repeated semantic comparisons.

        Returns:
            Coroutine[object, object, tuple[dict[str, JsonValue], bool]]:
            Awaitable canonical snapshot and match verdict.
        """
        ...

    def baseline_checksum(self, iterations: int) -> Coroutine[object, object, int]:
        """Run baseline checksum work.

        Args:
            iterations: Number of benchmark iterations.

        Returns:
            Coroutine[object, object, int]: Awaitable checksum.
        """
        ...

    def residual_checksum(
        self,
        iterations: int,
    ) -> Coroutine[object, object, tuple[int, StageCountersProtocol]]:
        """Run residual checksum work.

        Args:
            iterations: Number of benchmark iterations.

        Returns:
            Coroutine[object, object, tuple[int, StageCountersProtocol]]:
            Awaitable checksum and stage counters.
        """
        ...


class SampleSummaryPayload(TypedDict):
    """JSON payload for one benchmark arm.

    Attributes:
        arm: Name of the measured benchmark arm.
        samples: Seven rotating measured sample durations.
        median_seconds: Median measured duration for the arm.
        speedup_over_baseline: Baseline median divided by this arm median.
    """

    arm: str
    samples: list[float]
    median_seconds: float
    speedup_over_baseline: float


class PolicyPayload(TypedDict):
    """JSON payload for benchmark policy.

    Attributes:
        warmups: Number of unmeasured rotating warmup groups.
        samples: Number of measured rotating sample groups.
        minimum_seconds: Required median floor for every arm.
        minimum_speedup: Required residual speedup over baseline.
        semantic_repetitions: Number of deterministic semantic comparisons.
    """

    warmups: int
    samples: int
    minimum_seconds: float
    minimum_speedup: float
    semantic_repetitions: int


class ReportPayload(TypedDict):
    """Stable JSON report emitted by the hard benchmark.

    Attributes:
        gate_passed: Whether semantics, counters, medians, and speed passed.
        semantics_match: Whether residual semantics matched baseline.
        iterations: Calibrated iterations per measured arm invocation.
        policy: Benchmark policy used for the run.
        summaries: Per-arm timing summaries.
        final_speedup: Residual median speedup over baseline.
        stage_counters: Residual stage counters accumulated during measurement.
        stage_counters_nonzero: Whether every residual stage counter was used.
        semantic_snapshot: Canonical primitive semantic evidence.
    """

    gate_passed: bool
    semantics_match: bool
    iterations: int
    policy: PolicyPayload
    summaries: list[SampleSummaryPayload]
    final_speedup: float
    stage_counters: dict[str, int]
    stage_counters_nonzero: bool
    semantic_snapshot: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class BenchmarkOptions:
    """Validated hard-benchmark policy.

    Attributes:
        warmups: Number of unmeasured rotating warmup groups. The hard profile
            requires exactly one warmup.
        samples: Number of measured rotating sample groups. The hard profile
            requires exactly seven samples.
        minimum_seconds: Required median floor for every measured arm.
        minimum_speedup: Required final residual speedup over baseline.
        semantic_repetitions: Number of repeated semantic comparisons.
        initial_iterations: Initial iteration count before calibration.
    """

    warmups: int = DEFAULT_WARMUPS
    samples: int = DEFAULT_SAMPLES
    minimum_seconds: float = DEFAULT_MINIMUM_SECONDS
    minimum_speedup: float = DEFAULT_MINIMUM_SPEEDUP
    semantic_repetitions: int = DEFAULT_SEMANTIC_REPETITIONS
    initial_iterations: int = DEFAULT_ITERATIONS


@dataclass(frozen=True, slots=True)
class SampleSummary:
    """Measured durations and median for one benchmark arm.

    Attributes:
        arm: Name of the measured arm.
        samples: Seven rotating measured durations.
        median_seconds: Median measured duration.
        speedup_over_baseline: Baseline median divided by this arm median.
    """

    arm: ArmName
    samples: tuple[float, ...]
    median_seconds: float
    speedup_over_baseline: float


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Complete hard-benchmark evidence.

    Attributes:
        gate_passed: Combined semantic, counter, timing, and speed verdict.
        semantics_match: Deterministic semantic comparison result.
        iterations: Calibrated iterations per arm invocation.
        policy: Validated benchmark policy.
        summaries: Timing summaries for baseline and residual arms.
        final_speedup: Residual median speedup over baseline.
        stage_counters: Residual optimization-stage counters.
        semantic_snapshot: Canonical primitive semantic evidence.
    """

    gate_passed: bool
    semantics_match: bool
    iterations: int
    policy: BenchmarkOptions
    summaries: tuple[SampleSummary, ...]
    final_speedup: float
    stage_counters: dict[str, int]
    semantic_snapshot: dict[str, JsonValue]


def main(argv: tuple[str, ...] | None = None) -> int:
    """Run the hard benchmark and print canonical JSON evidence.

    Args:
        argv: Optional arguments replacing ``sys.argv`` for tests.

    Returns:
        int: Zero only when the hard gate passes.
    """
    args = _parse_args(tuple(sys.argv[1:] if argv is None else argv))
    try:
        report = run_benchmark(
            BenchmarkOptions(
                warmups=args.warmups,
                samples=args.samples,
                minimum_seconds=args.minimum_seconds,
                minimum_speedup=args.minimum_speedup,
                semantic_repetitions=args.semantic_repetitions,
                initial_iterations=args.initial_iterations,
            )
        )
    except (BenchmarkError, ValueError) as error:
        print(f"residual async profile benchmark failed: {error}", file=sys.stderr)
        return 1
    print(_canonical_json(report_as_json(report)))
    if report.gate_passed:
        return 0
    print(
        "residual async profile hard gate failed: "
        f"speedup={report.final_speedup:.3f}x/{report.policy.minimum_speedup:.3f}x, "
        f"semantics={report.semantics_match}, "
        f"stage_counters={all(value > 0 for value in report.stage_counters.values())}",
        file=sys.stderr,
    )
    return 1


def run_benchmark(options: BenchmarkOptions | None = None) -> BenchmarkReport:
    """Run semantic comparison, calibration, and rotating samples.

    Args:
        options: Hard-benchmark policy.

    Returns:
        BenchmarkReport: Complete evidence and promotion verdict.

    Raises:
        BenchmarkError: If policy or measured evidence violates the hard gate.
    """
    options = BenchmarkOptions() if options is None else options
    _validate_options(options)
    fixture = _fixture_module()
    semantic_snapshot, semantics_match = asyncio.run(
        fixture.compare_semantics(options.semantic_repetitions)
    )
    iterations = _calibrate_iterations(options)
    summaries, stage_counters = _measure_samples(iterations, options)
    baseline = _summary_for(summaries, "baseline")
    residual = _summary_for(summaries, "residual")
    final_speedup = baseline.median_seconds / residual.median_seconds
    gate_passed = (
        semantics_match
        and all(summary.median_seconds >= options.minimum_seconds for summary in summaries)
        and all(value > 0 for value in stage_counters.values())
        and final_speedup >= options.minimum_speedup
    )
    return BenchmarkReport(
        gate_passed=gate_passed,
        semantics_match=semantics_match,
        iterations=iterations,
        policy=options,
        summaries=summaries,
        final_speedup=final_speedup,
        stage_counters=stage_counters,
        semantic_snapshot=semantic_snapshot,
    )


def report_as_json(report: BenchmarkReport) -> ReportPayload:
    """Convert benchmark evidence to stable JSON fields.

    Args:
        report: Benchmark report to serialize.

    Returns:
        ReportPayload: JSON-compatible report.
    """
    return ReportPayload(
        gate_passed=report.gate_passed,
        semantics_match=report.semantics_match,
        iterations=report.iterations,
        policy=PolicyPayload(
            warmups=report.policy.warmups,
            samples=report.policy.samples,
            minimum_seconds=report.policy.minimum_seconds,
            minimum_speedup=report.policy.minimum_speedup,
            semantic_repetitions=report.policy.semantic_repetitions,
        ),
        summaries=[
            SampleSummaryPayload(
                arm=summary.arm,
                samples=list(summary.samples),
                median_seconds=summary.median_seconds,
                speedup_over_baseline=summary.speedup_over_baseline,
            )
            for summary in report.summaries
        ],
        final_speedup=report.final_speedup,
        stage_counters=report.stage_counters,
        stage_counters_nonzero=all(value > 0 for value in report.stage_counters.values()),
        semantic_snapshot=report.semantic_snapshot,
    )


def _measure_samples(
    iterations: int,
    options: BenchmarkOptions,
) -> tuple[tuple[SampleSummary, ...], dict[str, int]]:
    fixture = _fixture_module()
    arms: tuple[ArmName, ...] = ("baseline", "residual")
    measured: dict[ArmName, list[float]] = {arm: [] for arm in arms}
    stage_counters: dict[str, int] = dict.fromkeys(fixture.STAGE_NAMES, 0)
    for sample_index in range(options.warmups + options.samples):
        ordered_arms = arms if sample_index % 2 == 0 else tuple(reversed(arms))
        for arm in ordered_arms:
            elapsed, counters = _time_arm(arm, iterations)
            if sample_index >= options.warmups:
                measured[arm].append(elapsed)
                if counters is not None:
                    for name, value in counters.items():
                        stage_counters[name] += value
    baseline_median = median(measured["baseline"])
    summaries = tuple(
        SampleSummary(
            arm=arm,
            samples=tuple(measured[arm]),
            median_seconds=median(measured[arm]),
            speedup_over_baseline=baseline_median / median(measured[arm]),
        )
        for arm in arms
    )
    return summaries, stage_counters


def _calibrate_iterations(options: BenchmarkOptions) -> int:
    iterations = options.initial_iterations
    target_seconds = options.minimum_seconds * CALIBRATION_HEADROOM_FACTOR
    while iterations <= MAX_CALIBRATION_ITERATIONS:
        baseline_seconds, _ = _time_arm("baseline", iterations)
        residual_seconds, _ = _time_arm("residual", iterations)
        if baseline_seconds >= target_seconds and residual_seconds >= target_seconds:
            return iterations
        iterations *= 2
    raise BenchmarkError("could not calibrate both benchmark arms above the median floor")


def _time_arm(
    arm: ArmName,
    iterations: int,
) -> tuple[float, dict[str, int] | None]:
    fixture = _fixture_module()
    started = time.perf_counter()
    if arm == "baseline":
        checksum = asyncio.run(fixture.baseline_checksum(iterations))
        counters = None
    else:
        checksum, stage_counters = asyncio.run(fixture.residual_checksum(iterations))
        counters = dict(stage_counters.as_json())
    elapsed = time.perf_counter() - started
    if checksum <= 0:
        raise BenchmarkError(f"{arm} produced a non-positive checksum")
    return elapsed, counters


def _validate_options(options: BenchmarkOptions) -> None:
    if options.warmups != DEFAULT_WARMUPS:
        raise BenchmarkError("residual async profile benchmark requires exactly one warmup")
    if options.samples != DEFAULT_SAMPLES:
        raise BenchmarkError("residual async profile benchmark requires exactly seven samples")
    if options.minimum_seconds < DEFAULT_MINIMUM_SECONDS:
        raise BenchmarkError("minimum median seconds must be at least 0.25")
    if options.minimum_speedup < DEFAULT_MINIMUM_SPEEDUP:
        raise BenchmarkError("minimum speedup must be at least 3.0x")
    if options.semantic_repetitions < 1:
        raise BenchmarkError("semantic repetitions must be positive")
    if options.initial_iterations < 1:
        raise BenchmarkError("initial iterations must be positive")


def _summary_for(summaries: tuple[SampleSummary, ...], arm: ArmName) -> SampleSummary:
    return next(summary for summary in summaries if summary.arm == arm)


def _fixture_module() -> FixtureModule:
    source_root = str((FIXTURE_ROOT / "src").resolve())
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    return cast(FixtureModule, importlib.import_module("residual_async_profile.profile"))


def _parse_args(argv: tuple[str, ...]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--minimum-seconds", type=float, default=DEFAULT_MINIMUM_SECONDS)
    parser.add_argument("--minimum-speedup", type=float, default=DEFAULT_MINIMUM_SPEEDUP)
    parser.add_argument("--semantic-repetitions", type=int, default=DEFAULT_SEMANTIC_REPETITIONS)
    parser.add_argument("--initial-iterations", type=int, default=DEFAULT_ITERATIONS)
    return parser.parse_args(argv)


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


if __name__ == "__main__":
    raise SystemExit(main())
