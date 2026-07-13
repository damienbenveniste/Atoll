"""Measure Atoll's cold Cython batching and warm cache behavior."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.clustering import enrich_island_analysis
from atoll.backends.cython import CYTHON_BACKEND
from atoll.models import BackendCompileContext, BackendLoweringRequest, CompilationUnit, ModuleId
from atoll.region_cache import compile_many_with_region_cache, compile_with_region_cache

DEFAULT_UNITS = 8
DEFAULT_SAMPLES = 3
DEFAULT_MINIMUM_REDUCTION = 0.20
MINIMUM_BATCH_UNITS = 2
_NATIVE_PHASES = frozenset({"cython_batch", "cythonize", "build_ext"})


@dataclass(frozen=True, slots=True)
class BatchBenchmarkEvidence:
    """Measured cold-build and warm-cache acceptance evidence."""

    artifact_parity: bool
    batch_median_seconds: float
    batch_samples_seconds: tuple[float, ...]
    cold_reduction: float
    minimum_reduction: float
    passed: bool
    sequential_median_seconds: float
    sequential_samples_seconds: tuple[float, ...]
    unit_count: int
    warm_cache_hits: int
    warm_native_phase_count: int


@dataclass(frozen=True, slots=True)
class BatchEvidenceInputs:
    """Raw paired samples and cache evidence evaluated by the hard policy."""

    sequential_samples: tuple[float, ...]
    batch_samples: tuple[float, ...]
    artifact_parity: bool
    warm_cache_hits: int
    warm_native_phase_count: int
    unit_count: int
    minimum_reduction: float


def main(argv: tuple[str, ...] | None = None) -> int:
    """Run the representative benchmark and emit canonical JSON evidence.

    Args:
        argv: Optional command-line arguments replacing ``sys.argv``.

    Returns:
        int: Zero only when batching cuts cold time by the required fraction,
            preserves artifacts, and restores every warm unit without a compiler.
    """
    args = _parse_args(tuple(sys.argv[1:] if argv is None else argv))
    if args.workspace is None:
        with tempfile.TemporaryDirectory(prefix="atoll-cython-batch-") as temporary:
            evidence = run_benchmark(
                Path(temporary),
                unit_count=args.units,
                samples=args.samples,
                minimum_reduction=args.minimum_reduction,
            )
    else:
        evidence = run_benchmark(
            args.workspace,
            unit_count=args.units,
            samples=args.samples,
            minimum_reduction=args.minimum_reduction,
        )
    payload = json.dumps(asdict(evidence), sort_keys=True, separators=(",", ":"))
    print(payload)
    if args.evidence is not None:
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_text(f"{payload}\n", encoding="utf-8")
    return 0 if evidence.passed else 1


def run_benchmark(
    workspace: Path,
    *,
    unit_count: int = DEFAULT_UNITS,
    samples: int = DEFAULT_SAMPLES,
    minimum_reduction: float = DEFAULT_MINIMUM_REDUCTION,
) -> BatchBenchmarkEvidence:
    """Build identical units sequentially and in one Cython batch.

    Args:
        workspace: Empty directory receiving disposable sources, builds, and caches.
        unit_count: Number of independent extensions in each cold measurement.
        samples: Number of alternating cold sequential/batch pairs.
        minimum_reduction: Required fractional reduction in median cold time.

    Returns:
        BatchBenchmarkEvidence: Artifact, timing, and warm-cache evidence.

    Raises:
        ValueError: If policy is invalid or the workspace is not empty.
    """
    _validate_policy(unit_count, samples, minimum_reduction)
    if workspace.exists() and any(workspace.iterdir()):
        raise ValueError(f"benchmark workspace is not empty: {workspace}")
    workspace.mkdir(parents=True, exist_ok=True)
    units = _generate_units(workspace, unit_count)
    sequential_samples: list[float] = []
    batch_samples: list[float] = []
    artifact_parity = True
    final_batch_cache = workspace / "batch-cache-final"
    for sample in range(samples):
        ordered = ("sequential", "batch") if sample % 2 == 0 else ("batch", "sequential")
        sample_results: dict[str, tuple[float, frozenset[str]]] = {}
        for arm in ordered:
            elapsed, artifacts = _run_cold_arm(
                workspace,
                units,
                arm=arm,
                sample=sample,
                cache_root=(
                    final_batch_cache
                    if arm == "batch" and sample == samples - 1
                    else workspace / f"{arm}-cache-{sample}"
                ),
            )
            sample_results[arm] = (elapsed, artifacts)
        sequential_elapsed, sequential_artifacts = sample_results["sequential"]
        batch_elapsed, batch_artifacts = sample_results["batch"]
        sequential_samples.append(sequential_elapsed)
        batch_samples.append(batch_elapsed)
        artifact_parity &= sequential_artifacts == batch_artifacts
    warm_context = _compile_context(workspace, workspace / "warm-restore" / "build")
    warm_results = compile_many_with_region_cache(
        CYTHON_BACKEND,
        units,
        warm_context,
        cache_root=final_batch_cache,
    )
    warm_cache_hits = sum(result.attempt.cache_status == "hit" for result in warm_results)
    warm_native_phase_count = sum(
        timing.name in _NATIVE_PHASES
        for result in warm_results
        for timing in result.attempt.phase_timings
    )
    return evaluate_evidence(
        BatchEvidenceInputs(
            sequential_samples=tuple(sequential_samples),
            batch_samples=tuple(batch_samples),
            artifact_parity=artifact_parity,
            warm_cache_hits=warm_cache_hits,
            warm_native_phase_count=warm_native_phase_count,
            unit_count=unit_count,
            minimum_reduction=minimum_reduction,
        )
    )


def evaluate_evidence(inputs: BatchEvidenceInputs) -> BatchBenchmarkEvidence:
    """Apply the cold reduction, artifact parity, and warm cache policy.

    Args:
        inputs: Paired cold timings, artifact parity, and warm-cache evidence.

    Returns:
        BatchBenchmarkEvidence: Normalized benchmark verdict and metrics.
    """
    sequential_samples = inputs.sequential_samples
    batch_samples = inputs.batch_samples
    if not sequential_samples or len(sequential_samples) != len(batch_samples):
        raise ValueError("batch benchmark requires paired non-empty samples")
    sequential_median = median(sequential_samples)
    batch_median = median(batch_samples)
    if sequential_median <= 0.0 or batch_median <= 0.0:
        raise ValueError("batch benchmark durations must be positive")
    reduction = 1.0 - (batch_median / sequential_median)
    passed = (
        reduction >= inputs.minimum_reduction
        and inputs.artifact_parity
        and inputs.warm_cache_hits == inputs.unit_count
        and inputs.warm_native_phase_count == 0
    )
    return BatchBenchmarkEvidence(
        artifact_parity=inputs.artifact_parity,
        batch_median_seconds=batch_median,
        batch_samples_seconds=batch_samples,
        cold_reduction=reduction,
        minimum_reduction=inputs.minimum_reduction,
        passed=passed,
        sequential_median_seconds=sequential_median,
        sequential_samples_seconds=sequential_samples,
        unit_count=inputs.unit_count,
        warm_cache_hits=inputs.warm_cache_hits,
        warm_native_phase_count=inputs.warm_native_phase_count,
    )


def _run_cold_arm(
    workspace: Path,
    units: tuple[CompilationUnit, ...],
    *,
    arm: str,
    sample: int,
    cache_root: Path,
) -> tuple[float, frozenset[str]]:
    root = workspace / f"{arm}-{sample}"
    shutil.rmtree(root, ignore_errors=True)
    shutil.rmtree(cache_root, ignore_errors=True)
    started = time.perf_counter()
    if arm == "batch":
        results = compile_many_with_region_cache(
            CYTHON_BACKEND,
            units,
            _compile_context(workspace, root / "build"),
            cache_root=cache_root,
        )
    else:
        results = tuple(
            compile_with_region_cache(
                CYTHON_BACKEND,
                unit,
                _compile_context(workspace, root / f"unit-{index}" / "build"),
                cache_root=cache_root,
            )
            for index, unit in enumerate(units)
        )
    elapsed = time.perf_counter() - started
    failed = tuple(result for result in results if not result.attempt.success)
    if failed:
        raise RuntimeError(f"{arm} Cython build failed: {failed[0].attempt.stderr}")
    artifacts = frozenset(
        record.logical_module
        for result in results
        for record in result.artifacts
        if record.role == "primary"
    )
    return elapsed, artifacts


def _generate_units(workspace: Path, unit_count: int) -> tuple[CompilationUnit, ...]:
    source_root = workspace / "sources"
    source_root.mkdir()
    units: list[CompilationUnit] = []
    for index in range(unit_count):
        module_name = f"batch_kernel_{index}"
        source_path = source_root / f"{module_name}.py"
        source_path.write_text(
            "def kernel(value: int) -> int:\n"
            "    total = 0\n"
            "    for offset in range(128):\n"
            f"        total += (value + offset) * {index + 3}\n"
            "    return total\n",
            encoding="utf-8",
        )
        scan = enrich_island_analysis(scan_module(ModuleId(name=module_name, path=source_path)))
        region = next(
            item
            for item in scan.typed_regions
            if any(member.id.qualname == "kernel" for member in item.members)
        )
        units.append(
            CYTHON_BACKEND.lower(
                BackendLoweringRequest(
                    region=region,
                    source_path=source_path,
                    logical_module=f"_atoll_batch_kernel_{index}",
                    install_relative_dir=f"compiled/{index}",
                    members=tuple(member.id for member in region.members),
                    variant_id=f"{region.id}@cython-batch-benchmark-{index}",
                )
            )
        )
    return tuple(units)


def _compile_context(workspace: Path, build_dir: Path) -> BackendCompileContext:
    return BackendCompileContext(
        project_root=workspace,
        build_dir=build_dir,
        source_roots=(workspace / "sources",),
    )


def _validate_policy(unit_count: int, samples: int, minimum_reduction: float) -> None:
    if unit_count < MINIMUM_BATCH_UNITS:
        raise ValueError("batch benchmark requires at least two units")
    if samples < 1:
        raise ValueError("batch benchmark requires at least one sample")
    if not 0.0 < minimum_reduction < 1.0:
        raise ValueError("minimum reduction must be between zero and one")


def _parse_args(argv: tuple[str, ...]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--units", type=int, default=DEFAULT_UNITS)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--minimum-reduction", type=float, default=DEFAULT_MINIMUM_REDUCTION)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
