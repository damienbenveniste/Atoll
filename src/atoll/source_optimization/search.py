"""Search, verify, and promote profitable source-optimization patches.

This module owns disposable source-tree trials after static plans have passed
their profile and safety assessments. It never edits the checkout directly.
Candidates are formed from cumulative lowering variants, evaluated in bounded
beam order, rebuilt through the target project's normal PEP 517 backend, and
promoted only when both source and wheel measurements meet the hard 3x floor.
Git application is delegated to the transactional application module.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import platform
import shutil
import sys
import sysconfig
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from importlib import metadata as importlib_metadata
from pathlib import Path
from statistics import median
from typing import Literal

from atoll.models import CompileAttempt, CompileConfig, CompilePhaseTiming
from atoll.native_optimization.run_guard import RunGuardNativePlan
from atoll.optimization_policy import (
    DEFAULT_MINIMUM_MARGINAL_SPEEDUP,
    HARD_BENCHMARK_MINIMUM_SPEEDUP,
    assess_paired_speedup,
    assess_speedup,
)
from atoll.runtime.performance import (
    BenchmarkGateResult,
    BenchmarkStatus,
    CommandRunEvidence,
    RuntimeMode,
    run_performance_command,
)
from atoll.runtime.profiling import ProfileResult
from atoll.source_optimization.application import (
    apply_source_patch_transactionally,
    validate_source_application_root,
)
from atoll.source_optimization.cache import restore_or_build_source_patch
from atoll.source_optimization.lowering import (
    SourceLoweringResult,
    lower_batch_quiescent_plan,
    lower_residual_state_machine_plan,
    lower_state_machine_plan,
)
from atoll.source_optimization.models import (
    SourceOptimizationApplicationStatus,
    SourceOptimizationAssessment,
    SourceOptimizationPlan,
    SourceOptimizationTrial,
    SourceOptimizationTrialStatus,
    SourceTransformationKind,
)
from atoll.source_optimization.transforms import (
    GeneratedSourcePatch,
    SourceTransformationRequest,
    materialize_transformed_files,
)
from atoll.source_optimization.winner_cache import (
    SourceWinnerIdentity,
    load_source_winner,
    store_source_winner,
)
from atoll.source_snapshot import copy_source_snapshot
from atoll.wheel_overlay import (
    WheelBuildEvidence,
    WheelOverlayError,
    build_baseline_wheel,
    unpack_wheel_payload,
)

SourceOptimizationProgress = Callable[[str], None]
SourceSearchArm = Literal["baseline", "current", "candidate"]

HARD_MINIMUM_SOURCE_SPEEDUP = HARD_BENCHMARK_MINIMUM_SPEEDUP
SOURCE_SEARCH_BEAM_WIDTH = 2
SOURCE_SEARCH_MAX_DEPTH = 4
SOURCE_SEARCH_MAX_TRIALS = 8
SOURCE_SEARCH_WARMUPS = 1
SOURCE_SEARCH_SAMPLES = 3
SOURCE_SEARCH_MINIMUM_MARGINAL_SPEEDUP = DEFAULT_MINIMUM_MARGINAL_SPEEDUP
SOURCE_SEARCH_VERSION = "source-search-v2"
SOURCE_SEARCH_LOWERING_VERSION = "source-search-lowering-v1"
_RESIDUAL_TRANSFORMATIONS: tuple[SourceTransformationKind, ...] = (
    "run-scoped-guard-amortization",
    "transparent-quiescent-await-chain-collapse",
    "context-copy-elision",
    "incremental-private-completion-accounting",
    "private-result-record-elision",
)
_IGNORED_PROJECT_NAMES = frozenset(
    {
        ".atoll",
        ".git",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "site",
        "venv",
    }
)

type SourceCandidateProfiler = Callable[[Path, Path, Path, str], ProfileResult]


@dataclass(frozen=True, slots=True)
class SourceOptimizationSearchOptions:
    """Prepared target-project inputs for one source candidate search.

    Attributes:
        project_root: Immutable target checkout used to generate source patches.
        source_roots: Import roots discovered for the target project.
        module_paths: Project-relative importable module files removed from quality copies.
        output_dir: Directory receiving an accepted normally built wheel.
        scratch_root: Disposable root for transformed projects and wheel payloads.
        cache_root: Persistent Atoll-owned transformation cache root.
        baseline_payload_root: Unpacked ordinary wheel used for interpreted measurements.
        quality_project_root: Source-stripped project copy containing tests and benchmarks.
        compile_config: Validated semantic and benchmark policy.
        baseline_build: Existing baseline PEP 517 build evidence.
        apply_source: Whether an accepted patch should be applied transactionally.
        candidate_profiler: Optional current-invocation profiler for transformed candidates.
        progress: Optional user-facing progress callback.
    """

    project_root: Path
    source_roots: tuple[Path, ...]
    module_paths: tuple[Path, ...]
    output_dir: Path
    scratch_root: Path
    cache_root: Path
    baseline_payload_root: Path
    quality_project_root: Path
    compile_config: CompileConfig
    baseline_build: CompileAttempt
    apply_source: bool = False
    candidate_profiler: SourceCandidateProfiler | None = None
    progress: SourceOptimizationProgress | None = None


@dataclass(frozen=True, slots=True)
class SourceOptimizationSearchResult:
    """Bounded search outcome returned to source-clean package orchestration.

    Attributes:
        attempted: Whether configured plans reached disposable candidate trials.
        accepted: Whether source and wheel gates promoted a patch and wheel.
        wheel_path: Promoted normal PEP 517 wheel, when accepted.
        patch_path: Reviewable accepted patch under `.atoll/patches`, when accepted.
        materialization_patch: Immutable accepted patch payload for recreating
            transformed sources after scratch cleanup, when accepted.
        native_plans: Source-fused helpers introduced by the accepted patch and
            available to the later composable native stage.
        trials: Ordered candidate, final-gate, and application evidence.
        test_results: Semantic command evidence for the promoted source and wheel.
        performance: Authoritative final wheel performance result.
        build: Baseline and transformed-wheel build timing evidence.
        error: Fatal requested-application failure; ordinary unprofitability is not fatal.
    """

    attempted: bool
    accepted: bool
    wheel_path: Path | None
    patch_path: Path | None
    trials: tuple[SourceOptimizationTrial, ...]
    test_results: tuple[CommandRunEvidence, ...]
    performance: BenchmarkGateResult | None
    build: CompileAttempt
    error: str | None = None
    materialization_patch: GeneratedSourcePatch | None = None
    native_plans: tuple[RunGuardNativePlan, ...] = ()


@dataclass(frozen=True, slots=True)
class _PlanVariant:
    plan: SourceOptimizationPlan
    request: SourceTransformationRequest
    transformation_ids: tuple[str, ...]
    depth: int
    hot_share: float
    native_plans: tuple[RunGuardNativePlan, ...] = ()


@dataclass(frozen=True, slots=True)
class _SourceCandidate:
    id: str
    plan_ids: tuple[str, ...]
    requests: tuple[SourceTransformationRequest, ...]
    transformation_ids: tuple[str, ...]
    depth: int
    hot_share: float
    native_plans: tuple[RunGuardNativePlan, ...]


@dataclass(frozen=True, slots=True)
class _CandidateWorkspace:
    project_root: Path
    quality_root: Path
    source_payload_root: Path


@dataclass(frozen=True, slots=True)
class _SearchBenchmarkResult:
    success: bool
    reason: str
    baseline_median_seconds: float | None
    current_median_seconds: float | None
    candidate_median_seconds: float | None
    paired_marginal_speedup: float | None
    paired_marginal_passed: bool
    runs: tuple[CommandRunEvidence, ...]


@dataclass(frozen=True, slots=True)
class _CandidateEvaluation:
    candidate: _SourceCandidate
    patch: GeneratedSourcePatch
    workspace: _CandidateWorkspace
    semantic: CommandRunEvidence
    benchmark: _SearchBenchmarkResult
    source_speedup: float
    residual_profile: ProfileResult | None
    diagnostics: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _CandidateFormation:
    candidates: tuple[_SourceCandidate, ...]
    trials: tuple[SourceOptimizationTrial, ...]


@dataclass(frozen=True, slots=True)
class _SearchBenchmarkContext:
    command: tuple[str, ...]
    project_root: Path
    baseline_payload_root: Path
    current: _CandidateEvaluation | None
    candidate_payload_root: Path
    progress: SourceOptimizationProgress | None


@dataclass(frozen=True, slots=True)
class _PairGateContext:
    command: tuple[str, ...]
    project_root: Path
    baseline_payload_root: Path
    optimized_payload_root: Path
    minimum_speedup: float
    warmups: int
    samples: int
    progress: SourceOptimizationProgress | None
    label: str


@dataclass(frozen=True, slots=True)
class _WinnerTrialUpdate:
    status: SourceOptimizationTrialStatus
    reason: str
    source_gate: BenchmarkGateResult
    wheel_gate: BenchmarkGateResult | None = None
    semantic: CommandRunEvidence | None = None
    patch_path: Path | None = None
    application_status: SourceOptimizationApplicationStatus = "not-applied"
    diagnostics: tuple[str, ...] = ()


def run_source_optimization_search(
    plans: tuple[SourceOptimizationPlan, ...],
    assessments: tuple[SourceOptimizationAssessment, ...],
    options: SourceOptimizationSearchOptions,
) -> SourceOptimizationSearchResult:
    """Run bounded source trials and promote only a source-and-wheel 3x result.

    Args:
        plans: Profile-ranked source plans, limited to two by static planning.
        assessments: Current-invocation profile and safety evidence for the plans.
        options: Baseline payload, project paths, quality policy, and output boundaries.

    Returns:
        SourceOptimizationSearchResult: Trial evidence and optional promoted patch/wheel.
    """
    config = options.compile_config
    commands_configured = config.test_command is not None and config.benchmark_command is not None
    if not commands_configured:
        if options.apply_source:
            return replace(
                _not_attempted(options),
                error="--apply-source requires configured test_command and benchmark_command",
            )
        return _not_attempted(options)
    if options.apply_source:
        application_error = _source_application_preflight(options.project_root)
        if application_error is not None:
            return replace(_not_attempted(options), error=application_error)

    formation = _form_candidates(plans, assessments, options.project_root)
    if not formation.candidates:
        return SourceOptimizationSearchResult(
            attempted=bool(formation.trials),
            accepted=False,
            wheel_path=None,
            patch_path=None,
            trials=formation.trials,
            test_results=(),
            performance=None,
            build=options.baseline_build,
        )

    winner_identity = _winner_identity(plans, formation.candidates, options)
    return _run_formed_search(formation, options, winner_identity)


def _run_formed_search(
    formation: _CandidateFormation,
    options: SourceOptimizationSearchOptions,
    winner_identity: SourceWinnerIdentity,
) -> SourceOptimizationSearchResult:
    """Evaluate formed candidates while owning disposable scratch cleanup.

    Args:
        formation: Deterministic candidates and lowering rejection evidence.
        options: Prepared source-search paths and command policy.
        winner_identity: Exact static compatibility identity for accepted replay.

    Returns:
        SourceOptimizationSearchResult: Bounded beam and final-gate outcome.
    """
    _reset_dir(options.scratch_root)
    try:
        lookup = load_source_winner(options.cache_root, winner_identity)
        _progress(options.progress, lookup.diagnostic)
        replay = next(
            (
                candidate
                for candidate in formation.candidates
                if candidate.id == lookup.candidate_id
            ),
            None,
        )
        if replay is not None:
            replay_result, replay_timings = _replay_cached_winner(
                replay,
                formation=formation,
                options=options,
            )
            if replay_result.accepted:
                _store_accepted_winner(replay_result, options, winner_identity)
                return replay_result
            _progress(
                options.progress,
                f"accepted winner replay failed for {replay.id}; restarting full source search",
            )
            _reset_dir(options.scratch_root)
            result = _run_full_search(
                formation,
                options,
                excluded_candidate_ids=frozenset((replay.id,)),
                initial_trials=replay_result.trials,
                initial_timings=replay_timings,
            )
        else:
            result = _run_full_search(formation, options)
        if result.accepted:
            _store_accepted_winner(result, options, winner_identity)
        return result
    finally:
        if options.scratch_root.exists():
            shutil.rmtree(options.scratch_root)


def _replay_cached_winner(
    candidate: _SourceCandidate,
    *,
    formation: _CandidateFormation,
    options: SourceOptimizationSearchOptions,
) -> tuple[SourceOptimizationSearchResult, tuple[CompilePhaseTiming, ...]]:
    """Re-evaluate only a cached candidate and reproduce every acceptance gate.

    Args:
        candidate: Previously accepted candidate still present in the exact universe.
        formation: Current deterministic formation and lowering evidence.
        options: Current invocation paths, commands, profiler, and gate policy.

    Returns:
        tuple[SourceOptimizationSearchResult, tuple[CompilePhaseTiming, ...]]:
            Fresh replay result and its current-invocation phase timings.
    """
    evaluation, trial, timings = _evaluate_candidate(candidate, options=options, current=None)
    trials = (*formation.trials, trial)
    if evaluation is None:
        return _search_rejected(options, list(trials), list(timings)), timings
    return (
        _finalize_candidate(
            evaluation,
            options=options,
            trials=trials,
            timings=timings,
        ),
        timings,
    )


def _run_full_search(
    formation: _CandidateFormation,
    options: SourceOptimizationSearchOptions,
    *,
    excluded_candidate_ids: frozenset[str] = frozenset(),
    initial_trials: tuple[SourceOptimizationTrial, ...] | None = None,
    initial_timings: tuple[CompilePhaseTiming, ...] = (),
) -> SourceOptimizationSearchResult:
    """Run the ordinary bounded beam search over the complete formed universe.

    Args:
        formation: Complete deterministic candidates and lowering rejections.
        options: Prepared source-search paths and command policy.
        excluded_candidate_ids: Candidates that failed a semantic gate earlier
            in this invocation and must not be retried.
        initial_trials: Earlier current-invocation evidence to retain.
        initial_timings: Earlier current-invocation phase timings to retain.

    Returns:
        SourceOptimizationSearchResult: Fresh bounded-search and final-gate result.
    """
    timings = list(initial_timings)
    evaluations: list[_CandidateEvaluation] = []
    trials = list(formation.trials if initial_trials is None else initial_trials)
    evaluated_count = 0
    for depth, candidates in itertools.groupby(
        formation.candidates,
        key=lambda candidate: candidate.depth,
    ):
        current = _fastest_evaluation(evaluations)
        depth_evaluations: list[_CandidateEvaluation] = []
        for candidate in candidates:
            if candidate.id in excluded_candidate_ids:
                continue
            if evaluated_count >= SOURCE_SEARCH_MAX_TRIALS:
                break
            evaluation, trial, candidate_timings = _evaluate_candidate(
                candidate,
                options=options,
                current=current,
            )
            evaluated_count += 1
            timings.extend(candidate_timings)
            trials.append(trial)
            if evaluation is not None:
                depth_evaluations.append(evaluation)
        evaluations = sorted(
            (*evaluations, *depth_evaluations),
            key=lambda item: (
                item.benchmark.candidate_median_seconds or float("inf"),
                item.candidate.id,
            ),
        )[:SOURCE_SEARCH_BEAM_WIDTH]
        _progress(
            options.progress,
            f"source search depth {depth} retained {len(evaluations)} beam candidate(s)",
        )
        if evaluated_count >= SOURCE_SEARCH_MAX_TRIALS:
            break

    winner = _fastest_evaluation(evaluations)
    if winner is None:
        return _search_rejected(options, trials, timings)
    return _finalize_candidate(
        winner,
        options=options,
        trials=tuple(trials),
        timings=tuple(timings),
    )


def _store_accepted_winner(
    result: SourceOptimizationSearchResult,
    options: SourceOptimizationSearchOptions,
    identity: SourceWinnerIdentity,
) -> None:
    """Persist an accepted candidate without making cache I/O promotion-critical.

    Args:
        result: Fresh search result whose accepted trial identifies the winner.
        options: Search paths and progress sink for the active invocation.
        identity: Strict static compatibility boundary for later replay.
    """
    candidate_id = next(
        (trial.candidate_id for trial in result.trials if trial.status == "accepted"),
        None,
    )
    if candidate_id is None:
        return
    try:
        store_source_winner(options.cache_root, identity, candidate_id)
    except (OSError, ValueError) as error:
        _progress(options.progress, f"could not store accepted winner cache: {error}")


def _winner_identity(
    plans: tuple[SourceOptimizationPlan, ...],
    candidates: tuple[_SourceCandidate, ...],
    options: SourceOptimizationSearchOptions,
) -> SourceWinnerIdentity:
    """Build the strict static identity that permits accepted-winner replay.

    Args:
        plans: Source plans whose hashes define candidate inputs.
        candidates: Complete formed candidate universe, independent of ranking.
        options: Commands, performance policy, quality tree, baseline payload,
            and environment for the current invocation.

    Returns:
        SourceWinnerIdentity: Content-derived replay compatibility boundary.
    """
    config = options.compile_config
    candidate_plan_ids = {plan_id for candidate in candidates for plan_id in candidate.plan_ids}
    plan_sources = tuple(
        sorted(
            (
                plan.id,
                tuple(
                    sorted(
                        (path.as_posix(), source_hash)
                        for path, source_hash in plan.identity.source_hashes
                    )
                ),
            )
            for plan in plans
            if plan.id in candidate_plan_ids
        )
    )
    configuration = (
        ("backends", "\0".join(config.backends)),
        ("benchmark_warmups", str(config.benchmark_warmups)),
        ("benchmark_samples", str(config.benchmark_samples)),
        ("minimum_speedup", config.minimum_speedup.hex()),
        ("beam_width", str(SOURCE_SEARCH_BEAM_WIDTH)),
        ("max_depth", str(SOURCE_SEARCH_MAX_DEPTH)),
        ("max_trials", str(SOURCE_SEARCH_MAX_TRIALS)),
        ("search_warmups", str(SOURCE_SEARCH_WARMUPS)),
        ("search_samples", str(SOURCE_SEARCH_SAMPLES)),
        ("marginal_speedup", SOURCE_SEARCH_MINIMUM_MARGINAL_SPEEDUP.hex()),
        ("hard_speedup", HARD_MINIMUM_SOURCE_SPEEDUP.hex()),
    )
    return SourceWinnerIdentity(
        plan_sources=plan_sources,
        # Candidate hot-share ordering is profile evidence and may jitter.
        # Membership, not runtime ranking, defines replay compatibility.
        candidate_ids=tuple(sorted(candidate.id for candidate in candidates)),
        test_command=config.test_command or (),
        benchmark_command=config.benchmark_command or (),
        quality_project_digest=_content_tree_digest(
            options.quality_project_root,
            ignored_names=_IGNORED_PROJECT_NAMES,
        ),
        baseline_payload_digest=_content_tree_digest(
            options.baseline_payload_root,
            ignored_names=frozenset(("__pycache__",)),
        ),
        environment_digest=_runtime_environment_digest(),
        configuration=configuration,
        python_abi=sys.implementation.cache_tag or sys.implementation.name,
        platform=sysconfig.get_platform(),
        versions=(
            ("search", SOURCE_SEARCH_VERSION),
            ("lowering", SOURCE_SEARCH_LOWERING_VERSION),
        ),
    )


def _content_tree_digest(root: Path, *, ignored_names: frozenset[str]) -> str:
    """Hash stable paths, entry kinds, symlink text, and regular-file content.

    Args:
        root: Existing tree whose content participates in winner compatibility.
        ignored_names: Path components that are disposable runtime output.

    Returns:
        str: Deterministic SHA-256 digest independent of mtimes and inode metadata.

    Raises:
        ValueError: If ``root`` is not an existing directory.
        OSError: If a selected entry cannot be inspected or read.
    """
    if not root.is_dir():
        raise ValueError(f"winner identity root is not a directory: {root}")
    digest = hashlib.sha256(b"atoll-source-winner-tree-v1\0")
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root)
        if any(part in ignored_names for part in relative.parts):
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        if path.is_symlink():
            digest.update(b"symlink\0")
            digest.update(path.readlink().as_posix().encode("utf-8"))
        elif path.is_file():
            digest.update(b"file\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        elif path.is_dir():
            digest.update(b"directory\0")
        else:
            digest.update(b"other\0")
        digest.update(b"\0")
    return digest.hexdigest()


def _runtime_environment_digest() -> str:
    """Hash dependency versions and process inputs that can affect winner choice.

    The manifest stores only this digest. Environment values and distribution
    metadata are never persisted in source-optimization cache files.

    Returns:
        str: Deterministic SHA-256 digest for the active runtime/build environment.
    """
    distributions = {
        (name.casefold(), distribution.version)
        for distribution in importlib_metadata.distributions()
        if isinstance((name := distribution.metadata.get("Name")), str) and name
    }
    payload = {
        "distributions": sorted(distributions),
        "environment": sorted(os.environ.items()),
        "python": {
            "executable": sys.executable,
            "implementation": sys.implementation.name,
            "version": platform.python_version(),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _not_attempted(options: SourceOptimizationSearchOptions) -> SourceOptimizationSearchResult:
    return SourceOptimizationSearchResult(
        attempted=False,
        accepted=False,
        wheel_path=None,
        patch_path=None,
        trials=(),
        test_results=(),
        performance=None,
        build=options.baseline_build,
    )


def _form_candidates(
    plans: tuple[SourceOptimizationPlan, ...],
    assessments: tuple[SourceOptimizationAssessment, ...],
    project_root: Path,
) -> _CandidateFormation:
    assessment_by_plan = {assessment.plan_id: assessment for assessment in assessments}
    variants_by_plan: list[tuple[_PlanVariant, ...]] = []
    unavailable: list[SourceOptimizationTrial] = []
    for plan in plans[:2]:
        assessment = assessment_by_plan.get(plan.id)
        if assessment is None or assessment.status != "trial-ready":
            continue
        variants, rejections = _plan_variants(project_root, plan, assessment)
        if variants:
            variants_by_plan.append(variants)
        unavailable.extend(rejections)
    combinations = itertools.product(*((None, *variants) for variants in variants_by_plan))
    candidates: list[_SourceCandidate] = []
    for combination in combinations:
        selected = tuple(variant for variant in combination if variant is not None)
        if not selected or len({variant.request.path for variant in selected}) != len(selected):
            continue
        candidates.append(_source_candidate(selected))
    ordered = tuple(
        sorted(
            {candidate.id: candidate for candidate in candidates}.values(),
            key=lambda candidate: (
                candidate.depth,
                -candidate.hot_share,
                candidate.id,
            ),
        )
    )
    return _CandidateFormation(candidates=ordered, trials=tuple(unavailable))


def _plan_variants(
    project_root: Path,
    plan: SourceOptimizationPlan,
    assessment: SourceOptimizationAssessment,
) -> tuple[tuple[_PlanVariant, ...], tuple[SourceOptimizationTrial, ...]]:
    lowered = (
        lower_batch_quiescent_plan(project_root, plan, assessment),
        lower_state_machine_plan(project_root, plan, assessment),
    )
    variants: list[_PlanVariant] = []
    rejections: list[SourceOptimizationTrial] = []
    for result in lowered:
        if result.request is None:
            rejections.append(_lowering_rejection(plan, result))
            continue
        transformation_ids = _variant_transformation_ids(plan, result)
        depth = 1 if result.mode == "batch-quiescent" else 2
        variants.append(
            _PlanVariant(
                plan=plan,
                request=result.request,
                transformation_ids=transformation_ids,
                depth=depth,
                hot_share=assessment.attributed_hot_share,
                native_plans=result.native_plans,
            )
        )
    available_residual: tuple[SourceTransformationKind, ...] = tuple(
        kind for kind in _RESIDUAL_TRANSFORMATIONS if any(step.kind == kind for step in plan.steps)
    )
    enabled: tuple[SourceTransformationKind, ...] = ()
    for kind in available_residual:
        trial_steps = (*enabled, kind)
        result = lower_residual_state_machine_plan(
            project_root,
            plan,
            assessment,
            trial_steps,
        )
        if result.request is None:
            rejections.append(_lowering_rejection(plan, result))
            continue
        enabled = trial_steps
        variants.append(
            _PlanVariant(
                plan=plan,
                request=result.request,
                transformation_ids=_variant_transformation_ids(
                    plan,
                    result,
                    residual_steps=trial_steps,
                ),
                depth=min(2 + len(trial_steps), SOURCE_SEARCH_MAX_DEPTH),
                hot_share=assessment.attributed_hot_share,
                native_plans=result.native_plans,
            )
        )
    return tuple(variants), tuple(rejections)


def _variant_transformation_ids(
    plan: SourceOptimizationPlan,
    result: SourceLoweringResult,
    *,
    residual_steps: tuple[SourceTransformationKind, ...] = (),
) -> tuple[str, ...]:
    if result.mode == "batch-quiescent":
        enabled = {
            "private-transport-batch-drain",
            "quiescent-callable-execution",
        }
        return tuple(step.stable_id for step in plan.steps if step.kind in enabled)
    base = {
        "private-transport-batch-drain",
        "quiescent-callable-execution",
        "local-state-machine-fusion",
        "private-protocol-auto-forwarding",
    }
    selected = base | set(residual_steps)
    return tuple(step.stable_id for step in plan.steps if step.kind in selected)


def _lowering_rejection(
    plan: SourceOptimizationPlan,
    result: SourceLoweringResult,
) -> SourceOptimizationTrial:
    reason = "; ".join(result.rejections) or "lowering did not produce a request"
    return SourceOptimizationTrial(
        plan_id=plan.id,
        status="unavailable",
        semantic_command=(),
        benchmark_command=(),
        baseline_median_seconds=None,
        source_median_seconds=None,
        wheel_median_seconds=None,
        source_speedup=None,
        wheel_speedup=None,
        patch_path=None,
        source_edits=(),
        application_status="not-applied",
        diagnostics=result.rejections,
        candidate_id=f"{plan.id}:{result.mode}",
        transformation_ids=(),
        reason=reason,
    )


def _source_candidate(variants: tuple[_PlanVariant, ...]) -> _SourceCandidate:
    plan_ids = tuple(sorted(variant.plan.id for variant in variants))
    transformation_ids = tuple(
        sorted(identifier for variant in variants for identifier in variant.transformation_ids)
    )
    native_plans = tuple(
        sorted(
            {
                native_plan.stable_id: native_plan
                for variant in variants
                for native_plan in variant.native_plans
            }.values(),
            key=lambda native_plan: native_plan.stable_id,
        )
    )
    digest = hashlib.blake2b(digest_size=16)
    for value in (*plan_ids, *transformation_ids):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    for native_plan in native_plans:
        digest.update(native_plan.stable_id.encode("utf-8"))
        digest.update(b"\0")
    for request in sorted((variant.request for variant in variants), key=lambda item: item.path):
        digest.update(_request_fingerprint(request).encode("ascii"))
        digest.update(b"\0")
    return _SourceCandidate(
        id=f"source-candidate-{digest.hexdigest()}",
        plan_ids=plan_ids,
        requests=tuple(
            sorted((variant.request for variant in variants), key=lambda item: item.path)
        ),
        transformation_ids=transformation_ids,
        depth=max(variant.depth for variant in variants),
        hot_share=min(sum(variant.hot_share for variant in variants), 1.0),
        native_plans=native_plans,
    )


def _request_fingerprint(request: SourceTransformationRequest) -> str:
    digest = hashlib.sha256()
    values = (
        request.path.as_posix(),
        request.expected_sha256,
        request.target.stable_id,
        request.declaration_kind,
        request.replacement_body,
        *request.helper_statements,
        *request.trailing_statements,
        request.summary,
        request.transformation_id or "",
    )
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    for replacement in request.additional_replacements:
        for value in (
            replacement.target.stable_id,
            replacement.declaration_kind,
            replacement.replacement_body,
        ):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def _evaluate_candidate(
    candidate: _SourceCandidate,
    *,
    options: SourceOptimizationSearchOptions,
    current: _CandidateEvaluation | None,
) -> tuple[
    _CandidateEvaluation | None,
    SourceOptimizationTrial,
    tuple[CompilePhaseTiming, ...],
]:
    started = time.perf_counter()
    try:
        cached = restore_or_build_source_patch(
            options.cache_root,
            candidate.id,
            candidate.plan_ids,
            candidate.transformation_ids,
            options.project_root,
            candidate.requests,
        )
        patch = cached.patch
        workspace = _candidate_workspace(candidate, patch=patch, options=options)
    except (OSError, TypeError, ValueError) as error:
        duration = time.perf_counter() - started
        trial = _failed_candidate_trial(candidate, options, str(error))
        return None, trial, (_timing("source_candidate_stage", duration, candidate.id),)
    staging_duration = time.perf_counter() - started
    semantic = run_performance_command(
        options.compile_config.test_command or (),
        project_root=workspace.quality_root,
        payload_root=workspace.source_payload_root,
        mode="compiled",
        require_optimized=True,
    )
    timings = [
        _timing("source_candidate_stage", staging_duration, candidate.id),
        _timing(
            "source_candidate_semantic",
            semantic.duration_seconds,
            f"{candidate.id}; exit {semantic.returncode}",
        ),
    ]
    if not semantic.succeeded:
        return (
            None,
            _semantic_failure_trial(candidate, patch, options, semantic),
            tuple(timings),
        )

    benchmark = _run_search_benchmark(
        _SearchBenchmarkContext(
            command=options.compile_config.benchmark_command or (),
            project_root=workspace.quality_root,
            baseline_payload_root=options.baseline_payload_root,
            current=current,
            candidate_payload_root=workspace.source_payload_root,
            progress=options.progress,
        )
    )
    timings.extend(
        _timing(
            "source_candidate_benchmark",
            run.duration_seconds,
            f"{candidate.id}; {run.mode}",
        )
        for run in benchmark.runs
    )
    if not benchmark.success or benchmark.candidate_median_seconds is None:
        return (
            None,
            _invalid_benchmark_trial(candidate, patch, options, semantic, benchmark),
            tuple(timings),
        )
    baseline_median = benchmark.baseline_median_seconds
    current_median = benchmark.current_median_seconds
    if baseline_median is None or current_median is None:
        return (
            None,
            _invalid_benchmark_trial(candidate, patch, options, semantic, benchmark),
            tuple(timings),
        )
    source_assessment = assess_speedup(
        baseline_median,
        benchmark.candidate_median_seconds,
        minimum_speedup=SOURCE_SEARCH_MINIMUM_MARGINAL_SPEEDUP,
    )
    source_speedup = source_assessment.speedup or 0.0
    improves_current = benchmark.paired_marginal_passed
    residual_profile: ProfileResult | None = None
    profile_diagnostics: tuple[str, ...] = ()
    profile_failed = False
    if improves_current and options.candidate_profiler is not None:
        try:
            residual_profile = options.candidate_profiler(
                workspace.project_root,
                workspace.quality_root,
                workspace.source_payload_root,
                candidate.id,
            )
        except (OSError, TypeError, ValueError) as error:
            improves_current = False
            profile_failed = True
            profile_diagnostics = (f"residual profile failed: {error}",)
        else:
            profile_diagnostics = (
                f"residual profile {residual_profile.status}: {residual_profile.reason}",
            )
            if residual_profile.status != "profiled":
                improves_current = False
    evaluation = _CandidateEvaluation(
        candidate=candidate,
        patch=patch,
        workspace=workspace,
        semantic=semantic,
        benchmark=benchmark,
        source_speedup=source_speedup,
        residual_profile=residual_profile,
        diagnostics=(*cached.diagnostics, benchmark.reason, *profile_diagnostics),
    )
    trial = SourceOptimizationTrial(
        plan_id=_trial_plan_id(candidate),
        status="not-run" if improves_current else "not-profitable",
        semantic_command=options.compile_config.test_command or (),
        benchmark_command=options.compile_config.benchmark_command or (),
        baseline_median_seconds=baseline_median,
        source_median_seconds=benchmark.candidate_median_seconds,
        wheel_median_seconds=None,
        source_speedup=source_speedup,
        wheel_speedup=None,
        patch_path=None,
        source_edits=patch.source_edits,
        application_status="not-applied",
        diagnostics=(*cached.diagnostics, benchmark.reason, *profile_diagnostics),
        candidate_id=candidate.id,
        transformation_ids=candidate.transformation_ids,
        reason=(
            "candidate advanced into the bounded beam"
            if improves_current
            else (
                (
                    f"candidate residual profile was {residual_profile.status}; "
                    "profiled evidence is required"
                )
                if residual_profile is not None and residual_profile.status != "profiled"
                else "candidate residual profiling failed"
                if profile_failed
                else "candidate did not meet the 1.05x marginal speedup floor"
            )
        ),
        semantic_exit_code=semantic.returncode,
        semantic_duration_seconds=semantic.duration_seconds,
        current_median_seconds=current_median,
        residual_profile=residual_profile,
    )
    return (evaluation if improves_current else None), trial, tuple(timings)


def _candidate_workspace(
    candidate: _SourceCandidate,
    *,
    patch: GeneratedSourcePatch,
    options: SourceOptimizationSearchOptions,
) -> _CandidateWorkspace:
    root = options.scratch_root / candidate.id
    project_copy = root / "project"
    quality_copy = root / "quality"
    _copy_project(options.project_root, project_copy, excluded_output=options.output_dir)
    materialize_transformed_files(options.project_root, project_copy, patch)
    copy_source_snapshot(project_copy, quality_copy, ignore=_quality_copy_ignore)
    for module_path in options.module_paths:
        _safe_relative_path(quality_copy, module_path).unlink(missing_ok=True)
    source_payload = _source_payload_root(
        original_root=options.project_root,
        copied_root=project_copy,
        source_roots=options.source_roots,
        merge_root=root / "source-payload",
    )
    return _CandidateWorkspace(
        project_root=project_copy,
        quality_root=quality_copy,
        source_payload_root=source_payload,
    )


def _run_search_benchmark(context: _SearchBenchmarkContext) -> _SearchBenchmarkResult:
    current = context.current
    current_payload = (
        current.workspace.source_payload_root
        if current is not None
        else context.baseline_payload_root
    )
    payloads = {
        "baseline": context.baseline_payload_root,
        "current": current_payload,
        "candidate": context.candidate_payload_root,
    }
    optimized = {
        "baseline": False,
        "current": current is not None,
        "candidate": True,
    }
    permutations = tuple(itertools.permutations(("baseline", "current", "candidate")))
    runs: list[tuple[SourceSearchArm, CommandRunEvidence]] = []
    for phase, count in (("warmup", SOURCE_SEARCH_WARMUPS), ("sample", SOURCE_SEARCH_SAMPLES)):
        for index in range(count):
            order = permutations[index % len(permutations)]
            for arm_value in order:
                arm = _source_search_arm(arm_value)
                run = run_performance_command(
                    context.command,
                    project_root=context.project_root,
                    payload_root=payloads[arm],
                    mode="compiled" if optimized[arm] else "baseline",
                    require_optimized=optimized[arm],
                )
                runs.append((arm, run))
                _progress(
                    context.progress,
                    f"source search {phase} {index + 1} {arm} completed in "
                    f"{run.duration_seconds:.2f}s",
                )
                if not run.succeeded:
                    return _SearchBenchmarkResult(
                        success=False,
                        reason=f"{phase} {arm} benchmark exited {run.returncode}",
                        baseline_median_seconds=None,
                        current_median_seconds=None,
                        candidate_median_seconds=None,
                        paired_marginal_speedup=None,
                        paired_marginal_passed=False,
                        runs=tuple(item for _arm, item in runs),
                    )
    sample_runs = runs[SOURCE_SEARCH_WARMUPS * 3 :]
    medians = {
        arm: median(
            run.duration_seconds for observed_arm, run in sample_runs if observed_arm == arm
        )
        for arm in ("baseline", "current", "candidate")
    }
    sample_groups = tuple(sample_runs[index : index + 3] for index in range(0, len(sample_runs), 3))
    paired = assess_paired_speedup(
        tuple(
            next(run.duration_seconds for arm, run in group if arm == "current")
            for group in sample_groups
        ),
        tuple(
            next(run.duration_seconds for arm, run in group if arm == "candidate")
            for group in sample_groups
        ),
        minimum_speedup=SOURCE_SEARCH_MINIMUM_MARGINAL_SPEEDUP,
    )
    paired_text = (
        f"{paired.median_pair_speedup:.3f}x"
        if paired.median_pair_speedup is not None
        else "unstable"
    )
    return _SearchBenchmarkResult(
        success=True,
        reason=(
            "candidate search medians: "
            f"baseline={medians['baseline']:.3f}s, "
            f"current={medians['current']:.3f}s, "
            f"candidate={medians['candidate']:.3f}s, "
            f"paired marginal={paired_text}"
        ),
        baseline_median_seconds=medians["baseline"],
        current_median_seconds=medians["current"],
        candidate_median_seconds=medians["candidate"],
        paired_marginal_speedup=paired.median_pair_speedup,
        paired_marginal_passed=paired.passed,
        runs=tuple(item for _arm, item in runs),
    )


def _source_search_arm(value: str) -> SourceSearchArm:
    if value == "baseline":
        return "baseline"
    if value == "current":
        return "current"
    if value == "candidate":
        return "candidate"
    raise ValueError(f"invalid source search arm: {value}")


def _fastest_evaluation(
    evaluations: list[_CandidateEvaluation],
) -> _CandidateEvaluation | None:
    if not evaluations:
        return None
    return min(
        evaluations,
        key=lambda item: (
            item.benchmark.candidate_median_seconds or float("inf"),
            item.candidate.id,
        ),
    )


def _finalize_candidate(
    winner: _CandidateEvaluation,
    *,
    options: SourceOptimizationSearchOptions,
    trials: tuple[SourceOptimizationTrial, ...],
    timings: tuple[CompilePhaseTiming, ...],
) -> SourceOptimizationSearchResult:
    config = options.compile_config
    minimum_speedup = max(HARD_MINIMUM_SOURCE_SPEEDUP, config.minimum_speedup)
    source_gate = _run_pair_gate(
        _PairGateContext(
            command=config.benchmark_command or (),
            project_root=winner.workspace.quality_root,
            baseline_payload_root=options.baseline_payload_root,
            optimized_payload_root=winner.workspace.source_payload_root,
            minimum_speedup=minimum_speedup,
            warmups=1,
            samples=config.benchmark_samples,
            progress=options.progress,
            label="source",
        )
    )
    updated_timings = (
        *timings,
        *(
            _timing("source_final_benchmark", run.duration_seconds, run.mode)
            for run in (*source_gate.warmups, *source_gate.samples)
        ),
    )
    if not source_gate.succeeded:
        rejected = _replace_winner_trial(
            trials,
            winner,
            _WinnerTrialUpdate(
                status="not-profitable",
                reason=f"final transformed-source gate rejected candidate: {source_gate.reason}",
                source_gate=source_gate,
            ),
        )
        return _search_rejected(options, list(rejected), list(updated_timings), source_gate)

    wheel_build, wheel_path, wheel_payload = _build_candidate_wheel(winner)
    wheel_timing = _timing(
        "source_pep517_wheel",
        wheel_build.duration_seconds,
        f"exit {wheel_build.returncode}",
    )
    updated_timings = (*updated_timings, wheel_timing)
    if wheel_path is None or wheel_payload is None:
        rejected = _replace_winner_trial(
            trials,
            winner,
            _WinnerTrialUpdate(
                status="unavailable",
                reason=_wheel_build_reason(wheel_build),
                source_gate=source_gate,
            ),
        )
        return _search_rejected(options, list(rejected), list(updated_timings), source_gate)

    wheel_semantic = run_performance_command(
        config.test_command or (),
        project_root=winner.workspace.quality_root,
        payload_root=wheel_payload,
        mode="compiled",
        require_optimized=True,
    )
    updated_timings = (
        *updated_timings,
        _timing(
            "source_wheel_semantic",
            wheel_semantic.duration_seconds,
            f"exit {wheel_semantic.returncode}",
        ),
    )
    if not wheel_semantic.succeeded:
        rejected = _replace_winner_trial(
            trials,
            winner,
            _WinnerTrialUpdate(
                status="failed-semantics",
                reason="normally built transformed wheel failed semantic tests",
                source_gate=source_gate,
                semantic=wheel_semantic,
            ),
        )
        return _search_rejected(options, list(rejected), list(updated_timings), source_gate)

    wheel_gate = _run_pair_gate(
        _PairGateContext(
            command=config.benchmark_command or (),
            project_root=winner.workspace.quality_root,
            baseline_payload_root=options.baseline_payload_root,
            optimized_payload_root=wheel_payload,
            minimum_speedup=minimum_speedup,
            warmups=1,
            samples=config.benchmark_samples,
            progress=options.progress,
            label="wheel",
        )
    )
    updated_timings = (
        *updated_timings,
        *(
            _timing("source_wheel_benchmark", run.duration_seconds, run.mode)
            for run in (*wheel_gate.warmups, *wheel_gate.samples)
        ),
    )
    if not wheel_gate.succeeded:
        rejected = _replace_winner_trial(
            trials,
            winner,
            _WinnerTrialUpdate(
                status="not-profitable",
                reason=f"normally built wheel gate rejected candidate: {wheel_gate.reason}",
                source_gate=source_gate,
                wheel_gate=wheel_gate,
                semantic=wheel_semantic,
            ),
        )
        return _search_rejected(options, list(rejected), list(updated_timings), wheel_gate)

    patch_path = _persist_patch(winner, options.project_root)
    application_status: SourceOptimizationApplicationStatus = "not-applied"
    application_diagnostics: tuple[str, ...] = ()
    application_error: str | None = None
    if options.apply_source:
        application_status, application_diagnostics = _apply_accepted_patch(
            winner,
            patch_path=patch_path,
            options=options,
            minimum_speedup=minimum_speedup,
        )
        if application_status != "applied":
            application_error = (
                application_diagnostics[-1]
                if application_diagnostics
                else ("accepted source patch could not be applied transactionally")
            )
    promoted_wheel = (
        _promote_wheel(wheel_path, options.output_dir) if application_error is None else None
    )

    accepted_trials = _replace_winner_trial(
        trials,
        winner,
        _WinnerTrialUpdate(
            status="accepted",
            reason=f"source and normal wheel met the {minimum_speedup:.3f}x promotion floor",
            source_gate=source_gate,
            wheel_gate=wheel_gate,
            semantic=wheel_semantic,
            patch_path=patch_path,
            application_status=application_status,
            diagnostics=application_diagnostics,
        ),
    )
    build = _search_build_attempt(
        options,
        success=application_error is None,
        timings=updated_timings,
        wheel_build=wheel_build,
        error=application_error,
    )
    return SourceOptimizationSearchResult(
        attempted=True,
        accepted=application_error is None,
        wheel_path=promoted_wheel,
        patch_path=patch_path,
        trials=accepted_trials,
        test_results=(winner.semantic, wheel_semantic),
        performance=wheel_gate,
        build=build,
        error=application_error,
        materialization_patch=winner.patch if application_error is None else None,
        native_plans=winner.candidate.native_plans if application_error is None else (),
    )


def _run_pair_gate(context: _PairGateContext) -> BenchmarkGateResult:
    warmup_runs, failure = _run_pairs(
        context,
        count=context.warmups,
        phase=f"{context.label} warmup",
    )
    if failure is not None:
        return _invalid_gate(
            context.minimum_speedup,
            f"{context.label} warmup exited {failure.returncode}",
            warmup_runs,
        )
    sample_runs, failure = _run_pairs(
        context,
        count=context.samples,
        phase=f"{context.label} sample",
    )
    if failure is not None:
        return _invalid_gate(
            context.minimum_speedup,
            f"{context.label} sample exited {failure.returncode}",
            warmup_runs,
            sample_runs,
        )
    baseline_median = median(run.duration_seconds for run in sample_runs if run.mode == "baseline")
    optimized_median = median(run.duration_seconds for run in sample_runs if run.mode == "compiled")
    assessment = assess_speedup(
        baseline_median,
        optimized_median,
        minimum_speedup=context.minimum_speedup,
    )
    speedup = assessment.speedup
    if not assessment.stable:
        status: BenchmarkStatus = "invalid"
        reason = (
            f"{context.label} medians are too noisy: baseline={baseline_median:.3f}s "
            f"optimized={optimized_median:.3f}s"
        )
    elif assessment.passed:
        if speedup is None:
            raise AssertionError("stable source assessment must include speedup")
        status = "passed"
        reason = f"{context.label} speedup {speedup:.3f}x met {context.minimum_speedup:.3f}x"
    else:
        if speedup is None:
            raise AssertionError("stable source assessment must include speedup")
        status = "not-profitable"
        reason = f"{context.label} speedup {speedup:.3f}x missed {context.minimum_speedup:.3f}x"
    return BenchmarkGateResult(
        status=status,
        reason=reason,
        minimum_speedup=context.minimum_speedup,
        baseline_median_seconds=baseline_median,
        compiled_median_seconds=optimized_median,
        speedup=speedup,
        warmups=warmup_runs,
        samples=sample_runs,
    )


def _run_pairs(
    context: _PairGateContext,
    *,
    count: int,
    phase: str,
) -> tuple[tuple[CommandRunEvidence, ...], CommandRunEvidence | None]:
    runs: list[CommandRunEvidence] = []
    for index in range(count):
        modes: tuple[RuntimeMode, RuntimeMode] = (
            ("baseline", "compiled") if index % 2 == 0 else ("compiled", "baseline")
        )
        for mode in modes:
            run = run_performance_command(
                context.command,
                project_root=context.project_root,
                payload_root=(
                    context.baseline_payload_root
                    if mode == "baseline"
                    else context.optimized_payload_root
                ),
                mode=mode,
                require_optimized=mode == "compiled",
            )
            runs.append(run)
            _progress(
                context.progress,
                f"source optimization {phase} {index + 1} {mode} completed in "
                f"{run.duration_seconds:.2f}s",
            )
            if not run.succeeded:
                return tuple(runs), run
    return tuple(runs), None


def _invalid_gate(
    minimum_speedup: float,
    reason: str,
    warmups: tuple[CommandRunEvidence, ...],
    samples: tuple[CommandRunEvidence, ...] = (),
) -> BenchmarkGateResult:
    return BenchmarkGateResult(
        status="invalid",
        reason=reason,
        minimum_speedup=minimum_speedup,
        baseline_median_seconds=None,
        compiled_median_seconds=None,
        speedup=None,
        warmups=warmups,
        samples=samples,
    )


def _build_candidate_wheel(
    winner: _CandidateEvaluation,
) -> tuple[WheelBuildEvidence, Path | None, Path | None]:
    wheel_output = winner.workspace.project_root.parent / "wheel-dist"
    wheel_payload = winner.workspace.project_root.parent / "wheel-payload"
    evidence = build_baseline_wheel(winner.workspace.project_root, wheel_output)
    if not evidence.succeeded or len(evidence.wheel_paths) != 1:
        return evidence, None, None
    wheel_path = evidence.wheel_paths[0]
    try:
        unpack_wheel_payload(wheel_path, wheel_payload)
    except (OSError, WheelOverlayError, zipfile.BadZipFile):
        return evidence, None, None
    return evidence, wheel_path, wheel_payload


def _wheel_build_reason(evidence: WheelBuildEvidence) -> str:
    if not evidence.succeeded:
        return evidence.stderr or f"transformed PEP 517 build exited {evidence.returncode}"
    return f"transformed PEP 517 build produced {len(evidence.wheel_paths)} wheels"


def _persist_patch(winner: _CandidateEvaluation, project_root: Path) -> Path:
    patch_dir = project_root / ".atoll" / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    patch_path = patch_dir / f"{winner.candidate.id}.patch"
    temporary = patch_path.with_suffix(".patch.tmp")
    temporary.write_text(winner.patch.patch_text, encoding="utf-8")
    temporary.replace(patch_path)
    return patch_path


def _promote_wheel(wheel_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / wheel_path.name
    wheel_path.replace(destination)
    return destination


def _apply_accepted_patch(
    winner: _CandidateEvaluation,
    *,
    patch_path: Path,
    options: SourceOptimizationSearchOptions,
    minimum_speedup: float,
) -> tuple[SourceOptimizationApplicationStatus, tuple[str, ...]]:

    def validate() -> tuple[bool, tuple[str, ...]]:
        quality_root = options.scratch_root / "applied-quality"
        _copy_project(
            options.project_root,
            quality_root,
            excluded_output=options.output_dir,
        )
        for module_path in options.module_paths:
            _safe_relative_path(quality_root, module_path).unlink(missing_ok=True)
        source_payload = _source_payload_root(
            original_root=options.project_root,
            copied_root=options.project_root,
            source_roots=options.source_roots,
            merge_root=options.scratch_root / "applied-source-payload",
        )
        semantic = run_performance_command(
            options.compile_config.test_command or (),
            project_root=quality_root,
            payload_root=source_payload,
            mode="compiled",
            require_optimized=True,
        )
        if not semantic.succeeded:
            return False, (f"applied semantic command exited {semantic.returncode}",)
        gate = _run_pair_gate(
            _PairGateContext(
                command=options.compile_config.benchmark_command or (),
                project_root=quality_root,
                baseline_payload_root=options.baseline_payload_root,
                optimized_payload_root=source_payload,
                minimum_speedup=minimum_speedup,
                warmups=1,
                samples=options.compile_config.benchmark_samples,
                progress=options.progress,
                label="applied source",
            )
        )
        return gate.succeeded, (gate.reason,)

    result = apply_source_patch_transactionally(
        options.project_root,
        patch_path,
        winner.patch,
        validate,
    )
    return result.status, result.diagnostics


def _source_application_preflight(project_root: Path) -> str | None:
    return validate_source_application_root(project_root)


def _replace_winner_trial(
    trials: tuple[SourceOptimizationTrial, ...],
    winner: _CandidateEvaluation,
    update: _WinnerTrialUpdate,
) -> tuple[SourceOptimizationTrial, ...]:
    source_gate = update.source_gate
    wheel_gate = update.wheel_gate
    semantic = update.semantic
    replacement_trial = SourceOptimizationTrial(
        plan_id=_trial_plan_id(winner.candidate),
        status=update.status,
        semantic_command=winner.semantic.command,
        benchmark_command=winner.benchmark.runs[0].command if winner.benchmark.runs else (),
        baseline_median_seconds=source_gate.baseline_median_seconds,
        source_median_seconds=source_gate.compiled_median_seconds,
        wheel_median_seconds=(
            wheel_gate.compiled_median_seconds if wheel_gate is not None else None
        ),
        source_speedup=source_gate.speedup,
        wheel_speedup=wheel_gate.speedup if wheel_gate is not None else None,
        patch_path=update.patch_path,
        source_edits=winner.patch.source_edits,
        application_status=update.application_status,
        diagnostics=(*winner.diagnostics, *update.diagnostics),
        candidate_id=winner.candidate.id,
        transformation_ids=winner.candidate.transformation_ids,
        reason=update.reason,
        semantic_exit_code=(semantic or winner.semantic).returncode,
        semantic_duration_seconds=(semantic or winner.semantic).duration_seconds,
        current_median_seconds=winner.benchmark.current_median_seconds,
        residual_profile=winner.residual_profile,
    )
    return tuple(
        replacement_trial if trial.candidate_id == winner.candidate.id else trial
        for trial in trials
    )


def _failed_candidate_trial(
    candidate: _SourceCandidate,
    options: SourceOptimizationSearchOptions,
    reason: str,
) -> SourceOptimizationTrial:
    return SourceOptimizationTrial(
        plan_id=_trial_plan_id(candidate),
        status="unavailable",
        semantic_command=options.compile_config.test_command or (),
        benchmark_command=options.compile_config.benchmark_command or (),
        baseline_median_seconds=None,
        source_median_seconds=None,
        wheel_median_seconds=None,
        source_speedup=None,
        wheel_speedup=None,
        patch_path=None,
        source_edits=(),
        application_status="not-applied",
        diagnostics=(reason,),
        candidate_id=candidate.id,
        transformation_ids=candidate.transformation_ids,
        reason=f"candidate staging failed: {reason}",
    )


def _semantic_failure_trial(
    candidate: _SourceCandidate,
    patch: GeneratedSourcePatch,
    options: SourceOptimizationSearchOptions,
    semantic: CommandRunEvidence,
) -> SourceOptimizationTrial:
    return SourceOptimizationTrial(
        plan_id=_trial_plan_id(candidate),
        status="failed-semantics",
        semantic_command=options.compile_config.test_command or (),
        benchmark_command=options.compile_config.benchmark_command or (),
        baseline_median_seconds=None,
        source_median_seconds=None,
        wheel_median_seconds=None,
        source_speedup=None,
        wheel_speedup=None,
        patch_path=None,
        source_edits=patch.source_edits,
        application_status="not-applied",
        diagnostics=(semantic.stderr or semantic.stdout,),
        candidate_id=candidate.id,
        transformation_ids=candidate.transformation_ids,
        reason=f"candidate semantic command exited {semantic.returncode}",
        semantic_exit_code=semantic.returncode,
        semantic_duration_seconds=semantic.duration_seconds,
    )


def _invalid_benchmark_trial(
    candidate: _SourceCandidate,
    patch: GeneratedSourcePatch,
    options: SourceOptimizationSearchOptions,
    semantic: CommandRunEvidence,
    benchmark: _SearchBenchmarkResult,
) -> SourceOptimizationTrial:
    return SourceOptimizationTrial(
        plan_id=_trial_plan_id(candidate),
        status="unavailable",
        semantic_command=options.compile_config.test_command or (),
        benchmark_command=options.compile_config.benchmark_command or (),
        baseline_median_seconds=benchmark.baseline_median_seconds,
        source_median_seconds=benchmark.candidate_median_seconds,
        wheel_median_seconds=None,
        source_speedup=None,
        wheel_speedup=None,
        patch_path=None,
        source_edits=patch.source_edits,
        application_status="not-applied",
        diagnostics=(benchmark.reason,),
        candidate_id=candidate.id,
        transformation_ids=candidate.transformation_ids,
        reason=benchmark.reason,
        semantic_exit_code=semantic.returncode,
        semantic_duration_seconds=semantic.duration_seconds,
        current_median_seconds=benchmark.current_median_seconds,
    )


def _search_rejected(
    options: SourceOptimizationSearchOptions,
    trials: list[SourceOptimizationTrial],
    timings: list[CompilePhaseTiming],
    performance: BenchmarkGateResult | None = None,
) -> SourceOptimizationSearchResult:
    return SourceOptimizationSearchResult(
        attempted=True,
        accepted=False,
        wheel_path=None,
        patch_path=None,
        trials=tuple(trials),
        test_results=(),
        performance=performance,
        build=_search_build_attempt(
            options,
            success=False,
            timings=tuple(timings),
            wheel_build=None,
            error=None,
        ),
    )


def _search_build_attempt(
    options: SourceOptimizationSearchOptions,
    *,
    success: bool,
    timings: tuple[CompilePhaseTiming, ...],
    wheel_build: WheelBuildEvidence | None,
    error: str | None,
) -> CompileAttempt:
    return CompileAttempt(
        success=success,
        command=("atoll", "source-optimization"),
        stdout=wheel_build.stdout if wheel_build is not None else "",
        stderr=error or (wheel_build.stderr if wheel_build is not None else ""),
        artifact_paths=(),
        duration_seconds=(
            options.baseline_build.duration_seconds + sum(item.duration_seconds for item in timings)
        ),
        phase_timings=(*options.baseline_build.phase_timings, *timings),
    )


def _trial_plan_id(candidate: _SourceCandidate) -> str:
    return candidate.plan_ids[0] if len(candidate.plan_ids) == 1 else candidate.id


def _timing(name: str, duration: float, detail: str) -> CompilePhaseTiming:
    return CompilePhaseTiming(name=name, duration_seconds=duration, detail=detail)


def _copy_project(source: Path, destination: Path, *, excluded_output: Path) -> None:
    source_root = source.resolve()
    excluded_root = excluded_output.resolve()

    def ignore(directory: str, names: list[str]) -> set[str]:
        directory_path = Path(directory).resolve()
        ignored = {
            name
            for name in names
            if name in _IGNORED_PROJECT_NAMES or name.endswith((".so", ".pyd"))
        }
        for name in names:
            if (directory_path / name).resolve() == excluded_root:
                ignored.add(name)
        return ignored

    copy_source_snapshot(source_root, destination, ignore=ignore)
    _write_git_pointer(source_root, destination)


def _write_git_pointer(source: Path, destination: Path) -> None:
    git_path = source / ".git"
    if git_path.is_dir():
        git_dir = git_path.resolve()
    elif git_path.is_file():
        first_line = git_path.read_text(encoding="utf-8").splitlines()[0]
        if not first_line.startswith("gitdir:"):
            return
        raw_path = Path(first_line.removeprefix("gitdir:").strip())
        git_dir = (source / raw_path).resolve() if not raw_path.is_absolute() else raw_path
    else:
        return
    (destination / ".git").write_text(f"gitdir: {git_dir}\n", encoding="utf-8")


def _quality_copy_ignore(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in {".git", "__pycache__"}}


def _source_payload_root(
    *,
    original_root: Path,
    copied_root: Path,
    source_roots: tuple[Path, ...],
    merge_root: Path,
) -> Path:
    relative_roots = tuple(_relative_source_root(original_root, root) for root in source_roots)
    copied_source_roots = tuple(_safe_relative_path(copied_root, path) for path in relative_roots)
    if len(copied_source_roots) == 1:
        return copied_source_roots[0]
    _reset_dir(merge_root)
    for source_root in copied_source_roots:
        for item in source_root.iterdir():
            destination = merge_root / item.name
            if destination.exists():
                raise ValueError(f"source roots overlap at {item.name}")
            if item.is_symlink():
                shutil.copy2(item, destination, follow_symlinks=False)
            elif item.is_dir():
                copy_source_snapshot(item, destination)
            else:
                shutil.copy2(item, destination)
    return merge_root


def _relative_source_root(project_root: Path, source_root: Path) -> Path:
    try:
        return source_root.resolve().relative_to(project_root.resolve())
    except ValueError as error:
        raise ValueError(f"source root escapes project root: {source_root}") from error


def _safe_relative_path(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe project-relative path: {relative}")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"path escapes temporary project: {relative}") from error
    return candidate


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _progress(progress: SourceOptimizationProgress | None, message: str) -> None:
    if progress is not None:
        progress(message)
