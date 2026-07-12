"""Tests for bounded source candidate search and hard 3x promotion gates."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import cast

import pytest

import atoll.source_optimization.search as source_search
from atoll.models import CompileAttempt, CompileConfig, SymbolId
from atoll.runtime.performance import CommandRunEvidence
from atoll.source_optimization.application import SourcePatchApplicationResult
from atoll.source_optimization.lowering import SourceLoweringResult
from atoll.source_optimization.models import (
    SourceCallableEvidence,
    SourceOptimizationAssessment,
    SourceOptimizationIdentity,
    SourceOptimizationPlan,
    TransformationStep,
    stable_source_optimization_plan_id,
)
from atoll.source_optimization.search import (
    SOURCE_SEARCH_MAX_DEPTH,
    SOURCE_SEARCH_MAX_TRIALS,
    SourceOptimizationSearchOptions,
    run_source_optimization_search,
)
from atoll.source_optimization.transforms import (
    CallableBodyReplacement,
    GeneratedSourcePatch,
    SourceTransformationRequest,
)
from atoll.wheel_overlay import WheelBuildEvidence

FIXTURE_ROOT = Path("tests/fixtures/source_optimization_project")
SOURCE_PATH = PurePosixPath("src/source_optimization_fixture/workflow.py")
OWNER = SymbolId("source_optimization_fixture.workflow", "_run_hot_private_pipeline")
WORKER = SymbolId("source_optimization_fixture.workflow", "_immediate_worker")
LOWERING_VARIANT_COUNT = 2
SEARCH_BENCHMARK_RUNS_FOR_ONE_CANDIDATE = 12
WHEEL_SEMANTIC_FAILURE_CODE = 9


def test_candidate_search_is_stable_and_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public search reports deterministic IDs within the hard exploration bounds."""
    project_root = tmp_path / "project"
    _copy_fixture(project_root)
    plan, assessment = _plan_and_assessment(project_root)
    options = _search_options(project_root, tmp_path)

    def fake_run(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        mode: str,
        **_options: object,
    ) -> CommandRunEvidence:
        duration = 0.05 if command == ("test",) else (0.8 if mode == "baseline" else 0.4)
        return _command_evidence(command, (project_root, payload_root), mode, duration)

    monkeypatch.setattr("atoll.source_optimization.search.run_performance_command", fake_run)

    first = run_source_optimization_search((plan,), (assessment,), options)
    second = run_source_optimization_search(
        (
            replace(
                plan,
                steps=tuple(replace(step, description="report text") for step in plan.steps),
            ),
        ),
        (assessment,),
        options,
    )
    first_trials = tuple(
        trial for trial in first.trials if trial.candidate_id.startswith("source-")
    )
    second_trials = tuple(
        trial for trial in second.trials if trial.candidate_id.startswith("source-")
    )

    assert [trial.candidate_id for trial in first_trials] == [
        trial.candidate_id for trial in second_trials
    ]
    assert 1 <= len(first_trials) <= SOURCE_SEARCH_MAX_TRIALS
    assert all(len(trial.transformation_ids) <= SOURCE_SEARCH_MAX_DEPTH for trial in first_trials)


@pytest.mark.parametrize(
    ("optimized_seconds", "accepted"),
    [(0.1, True), (0.4, False)],
)
def test_search_promotes_only_source_and_wheel_results_above_three_x(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    optimized_seconds: float,
    accepted: bool,
) -> None:
    """Patch and wheel persistence are both controlled by the hard final gates."""
    project_root = tmp_path / "project"
    _copy_fixture(project_root)
    plan, assessment = _plan_and_assessment(project_root)
    options = _search_options(project_root, tmp_path)
    source_before = (project_root / SOURCE_PATH).read_bytes()
    command_calls = 0

    def fake_run(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        mode: str,
        **_options: object,
    ) -> CommandRunEvidence:
        nonlocal command_calls
        command_calls += 1
        duration = (
            0.05 if command == ("test",) else (0.8 if mode == "baseline" else optimized_seconds)
        )
        return _command_evidence(command, (project_root, payload_root), mode, duration)

    def fake_wheel(
        _winner: object,
    ) -> tuple[WheelBuildEvidence, Path, Path]:
        wheel_path = tmp_path / "candidate-py3-none-any.whl"
        wheel_path.write_bytes(b"wheel")
        payload = tmp_path / "wheel-payload"
        payload.mkdir(exist_ok=True)
        evidence = WheelBuildEvidence(
            command=("build",),
            project_root=project_root,
            outdir=tmp_path,
            returncode=0,
            stdout="",
            stderr="",
            duration_seconds=0.2,
            wheel_paths=(wheel_path,),
        )
        return evidence, wheel_path, payload

    monkeypatch.setattr("atoll.source_optimization.search.run_performance_command", fake_run)
    monkeypatch.setattr("atoll.source_optimization.search._build_candidate_wheel", fake_wheel)

    result = run_source_optimization_search((plan,), (assessment,), options)

    assert result.accepted is accepted
    assert command_calls > 0
    assert (project_root / SOURCE_PATH).read_bytes() == source_before
    assert not options.scratch_root.exists()
    if accepted:
        assert result.patch_path is not None
        assert result.patch_path.is_file()
        assert result.wheel_path is not None
        assert result.wheel_path.is_file()
        assert result.materialization_patch is not None
        assert result.materialization_patch.patch_text == result.patch_path.read_text(
            encoding="utf-8"
        )
        assert tuple(file.path for file in result.materialization_patch.files) == (SOURCE_PATH,)
        assert result.materialization_patch.files[0].after_source != source_before.decode()
        assert result.performance is not None
        assert result.performance.speedup == pytest.approx(8.0)
    else:
        assert result.patch_path is None
        assert result.wheel_path is None
        assert result.materialization_patch is None
        assert not (project_root / ".atoll" / "patches").exists()


def test_apply_source_revalidates_in_isolated_quality_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The applied arm cannot leak patched checkout imports into its baseline arm."""
    project_root = tmp_path / "project"
    _copy_fixture(project_root)
    plan, assessment = _plan_and_assessment(project_root)
    options = replace(_search_options(project_root, tmp_path), apply_source=True)
    source_before = (project_root / SOURCE_PATH).read_bytes()
    application_runs: list[tuple[Path, Path, str]] = []

    def fake_run(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        mode: str,
        **_options: object,
    ) -> CommandRunEvidence:
        if "applied-quality" in project_root.parts:
            application_runs.append((project_root, payload_root, mode))
        duration = 0.05 if command == ("test",) else (0.8 if mode == "baseline" else 0.1)
        return _command_evidence(command, (project_root, payload_root), mode, duration)

    def fake_wheel(_winner: object) -> tuple[WheelBuildEvidence, Path, Path]:
        wheel_path = tmp_path / "candidate-py3-none-any.whl"
        wheel_path.write_bytes(b"wheel")
        payload = tmp_path / "wheel-payload"
        payload.mkdir(exist_ok=True)
        return (
            WheelBuildEvidence(
                command=("build",),
                project_root=project_root,
                outdir=tmp_path,
                returncode=0,
                stdout="",
                stderr="",
                duration_seconds=0.2,
                wheel_paths=(wheel_path,),
            ),
            wheel_path,
            payload,
        )

    def fake_apply(
        root: Path,
        _patch_path: Path,
        patch: GeneratedSourcePatch,
        validate_callback: Callable[[], tuple[bool, tuple[str, ...]]],
    ) -> SourcePatchApplicationResult:
        for transformed in patch.files:
            (root / transformed.path).write_text(transformed.after_source, encoding="utf-8")
        succeeded, diagnostics = validate_callback()
        return SourcePatchApplicationResult(
            status="applied" if succeeded else "rolled-back",
            diagnostics=diagnostics,
        )

    def accept_application_root(_root: Path) -> None:
        return None

    monkeypatch.setattr("atoll.source_optimization.search.run_performance_command", fake_run)
    monkeypatch.setattr("atoll.source_optimization.search._build_candidate_wheel", fake_wheel)
    monkeypatch.setattr(
        "atoll.source_optimization.search.validate_source_application_root",
        accept_application_root,
    )
    monkeypatch.setattr(
        "atoll.source_optimization.search.apply_source_patch_transactionally",
        fake_apply,
    )

    result = run_source_optimization_search((plan,), (assessment,), options)

    assert result.accepted is True
    assert (project_root / SOURCE_PATH).read_bytes() != source_before
    assert application_runs
    assert all(run_root != project_root for run_root, _payload, _mode in application_runs)
    baseline_payloads = {payload for _root, payload, mode in application_runs if mode == "baseline"}
    compiled_payloads = {payload for _root, payload, mode in application_runs if mode == "compiled"}
    assert baseline_payloads == {options.baseline_payload_root}
    assert compiled_payloads == {project_root / "src"}
    accepted_trial = next(trial for trial in result.trials if trial.status == "accepted")
    assert accepted_trial.application_status == "applied"


def test_failed_source_application_never_promotes_custom_output_wheel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rolled-back apply transaction leaves no wheel in a custom output path."""
    project_root = tmp_path / "project"
    _copy_fixture(project_root)
    plan, assessment = _plan_and_assessment(project_root)
    output_dir = tmp_path / "custom-output"
    options = replace(
        _search_options(project_root, tmp_path),
        apply_source=True,
        output_dir=output_dir,
    )
    source_before = (project_root / SOURCE_PATH).read_bytes()

    def fake_run(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        mode: str,
        **_options: object,
    ) -> CommandRunEvidence:
        duration = 0.05 if command == ("test",) else (0.8 if mode == "baseline" else 0.1)
        return _command_evidence(command, (project_root, payload_root), mode, duration)

    def fake_wheel(_winner: object) -> tuple[WheelBuildEvidence, Path, Path]:
        wheel_path = options.scratch_root / "candidate-py3-none-any.whl"
        wheel_path.write_bytes(b"wheel")
        payload = options.scratch_root / "wheel-payload"
        payload.mkdir(exist_ok=True)
        return (
            WheelBuildEvidence(
                command=("build",),
                project_root=project_root,
                outdir=options.scratch_root,
                returncode=0,
                stdout="",
                stderr="",
                duration_seconds=0.2,
                wheel_paths=(wheel_path,),
            ),
            wheel_path,
            payload,
        )

    def reject_application(
        _root: Path,
        _patch_path: Path,
        _patch: GeneratedSourcePatch,
        _validate_callback: Callable[[], tuple[bool, tuple[str, ...]]],
    ) -> SourcePatchApplicationResult:
        return SourcePatchApplicationResult("rolled-back", ("forced rollback",))

    def accept_application_root(_root: Path) -> None:
        return None

    monkeypatch.setattr("atoll.source_optimization.search.run_performance_command", fake_run)
    monkeypatch.setattr("atoll.source_optimization.search._build_candidate_wheel", fake_wheel)
    monkeypatch.setattr(
        "atoll.source_optimization.search.validate_source_application_root",
        accept_application_root,
    )
    monkeypatch.setattr(
        "atoll.source_optimization.search.apply_source_patch_transactionally",
        reject_application,
    )

    result = run_source_optimization_search((plan,), (assessment,), options)

    assert result.accepted is False
    assert result.wheel_path is None
    assert result.patch_path is not None
    assert result.patch_path.is_file()
    assert result.materialization_patch is None
    assert result.error == "forced rollback"
    assert not output_dir.exists()
    assert (project_root / SOURCE_PATH).read_bytes() == source_before


def test_unconfigured_search_is_not_attempted_and_apply_reports_preflight(
    tmp_path: Path,
) -> None:
    """Missing commands prevent trials, and apply mode makes that fatal."""
    project_root = tmp_path / "project"
    _copy_fixture(project_root)
    plan, assessment = _plan_and_assessment(project_root)
    options = replace(
        _search_options(project_root, tmp_path),
        compile_config=CompileConfig(test_command=None, benchmark_command=None),
    )

    skipped = run_source_optimization_search((plan,), (assessment,), options)
    apply_skipped = run_source_optimization_search(
        (plan,),
        (assessment,),
        replace(options, apply_source=True),
    )

    assert skipped.attempted is False
    assert skipped.accepted is False
    assert skipped.materialization_patch is None
    assert skipped.error is None
    assert apply_skipped.attempted is False
    assert apply_skipped.materialization_patch is None
    assert (
        apply_skipped.error
        == "--apply-source requires configured test_command and benchmark_command"
    )


def test_apply_source_rejects_invalid_application_root_before_trials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply mode validates the checkout before forming source candidates."""
    project_root = tmp_path / "project"
    _copy_fixture(project_root)
    plan, assessment = _plan_and_assessment(project_root)
    options = replace(_search_options(project_root, tmp_path), apply_source=True)

    def reject_application_root(_root: Path) -> str:
        return "checkout is not a git worktree"

    monkeypatch.setattr(
        "atoll.source_optimization.search.validate_source_application_root",
        reject_application_root,
    )

    result = run_source_optimization_search((plan,), (assessment,), options)

    assert result.attempted is False
    assert result.accepted is False
    assert result.trials == ()
    assert result.error == "checkout is not a git worktree"


def test_lowering_rejections_are_returned_when_no_candidates_form(
    tmp_path: Path,
) -> None:
    """Unsupported lowering still returns deterministic unavailable trial evidence."""
    project_root = tmp_path / "project"
    _copy_fixture(project_root)
    plan, assessment = _plan_and_assessment(project_root)
    unsupported = replace(assessment, immediate_result_ratio=0.5)

    result = run_source_optimization_search(
        (plan,),
        (unsupported,),
        _search_options(project_root, tmp_path),
    )

    assert result.attempted is True
    assert result.accepted is False
    assert {trial.status for trial in result.trials} == {"unavailable"}
    assert len(result.trials) == LOWERING_VARIANT_COUNT
    assert all(
        "quiescent lowering requires a 100% immediate-result ratio" in trial.reason
        for trial in result.trials
    )


def test_missing_or_non_ready_assessments_are_skipped(
    tmp_path: Path,
) -> None:
    """Plans without trial-ready assessment evidence do not enter candidate formation."""
    project_root = tmp_path / "project"
    _copy_fixture(project_root)
    plan, assessment = _plan_and_assessment(project_root)
    options = _search_options(project_root, tmp_path)

    missing = run_source_optimization_search(
        (plan,),
        (),
        options,
    )
    non_ready = run_source_optimization_search(
        (plan,),
        (replace(assessment, status="unsupported"),),
        options,
    )

    assert missing.attempted is False
    assert missing.trials == ()
    assert non_ready.attempted is False
    assert non_ready.trials == ()


def test_candidate_stage_failure_records_unavailable_trial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch restoration or workspace staging errors reject only that candidate."""
    project_root = tmp_path / "project"
    _copy_fixture(project_root)
    plan, assessment = _plan_and_assessment(project_root)
    options = _search_options(project_root, tmp_path)

    def fail_restore(*_args: object, **_kwargs: object) -> object:
        raise ValueError("cached patch is inconsistent")

    monkeypatch.setattr(
        "atoll.source_optimization.search.restore_or_build_source_patch",
        fail_restore,
    )

    result = run_source_optimization_search((plan,), (assessment,), options)

    assert result.attempted is True
    assert result.accepted is False
    assert {trial.status for trial in result.trials} == {"unavailable"}
    assert any(
        "candidate staging failed: cached patch is inconsistent" in trial.reason
        for trial in result.trials
    )
    assert not options.scratch_root.exists()


def test_candidate_semantic_failure_records_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing source semantic command prevents benchmark execution."""
    project_root = tmp_path / "project"
    _copy_fixture(project_root)
    plan, assessment = _plan_and_assessment(project_root)
    options = _search_options(project_root, tmp_path)
    benchmark_calls = 0

    def fake_run(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        mode: str,
        **_options: object,
    ) -> CommandRunEvidence:
        nonlocal benchmark_calls
        if command == ("benchmark",):
            benchmark_calls += 1
        return _command_evidence(
            command,
            (project_root, payload_root),
            mode,
            0.05,
            result=(
                1 if command == ("test",) else 0,
                "",
                "semantic broke" if command == ("test",) else "",
            ),
        )

    monkeypatch.setattr("atoll.source_optimization.search.run_performance_command", fake_run)

    result = run_source_optimization_search((plan,), (assessment,), options)

    assert result.accepted is False
    assert benchmark_calls == 0
    assert {trial.status for trial in result.trials} == {"failed-semantics"}
    assert all(trial.semantic_exit_code == 1 for trial in result.trials)
    assert all(trial.diagnostics == ("semantic broke",) for trial in result.trials)


def test_candidate_benchmark_failure_records_unavailable_trial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed benchmark sample is reported as unavailable candidate evidence."""
    project_root = tmp_path / "project"
    _copy_fixture(project_root)
    plan, assessment = _plan_and_assessment(project_root)
    options = _search_options(project_root, tmp_path)

    def fake_run(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        mode: str,
        **_options: object,
    ) -> CommandRunEvidence:
        failed_benchmark = command == ("benchmark",) and mode != "baseline"
        return _command_evidence(
            command,
            (project_root, payload_root),
            mode,
            0.05,
            result=(7 if failed_benchmark else 0, "", ""),
        )

    monkeypatch.setattr("atoll.source_optimization.search.run_performance_command", fake_run)

    result = run_source_optimization_search((plan,), (assessment,), options)

    assert result.accepted is False
    assert {trial.status for trial in result.trials} == {"unavailable"}
    assert all("benchmark exited 7" in trial.reason for trial in result.trials)


def test_candidate_that_does_not_improve_current_beam_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The beam rejects later candidates that are slower than the retained current arm."""
    project_root = tmp_path / "project"
    _copy_fixture(project_root)
    first_plan, first_assessment = _plan_and_assessment(project_root)
    second_plan = replace(
        first_plan,
        id="exec-plan-source-search-second",
        identity=replace(
            first_plan.identity,
            execution_plan_id="exec-plan-source-search-second",
        ),
    )
    second_assessment = replace(first_assessment, plan_id=second_plan.id)
    options = _search_options(project_root, tmp_path)
    candidate_payload_durations: dict[Path, float] = {}

    def lowering(
        _project_root: Path,
        plan: SourceOptimizationPlan,
        _assessment: SourceOptimizationAssessment,
    ) -> SourceLoweringResult:
        source = (project_root / SOURCE_PATH).read_text(encoding="utf-8")
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        replacement = (
            "total = 0\n"
            "for item in items:\n"
            "    total += await _immediate_worker(item, context)\n"
            "return total\n"
        )
        return SourceLoweringResult(
            plan_id=plan.id,
            status="lowered",
            request=SourceTransformationRequest(
                path=SOURCE_PATH,
                expected_sha256=digest,
                target=OWNER,
                declaration_kind="async_function",
                replacement_body=replacement,
                additional_replacements=(
                    CallableBodyReplacement(
                        target=WORKER,
                        declaration_kind="async_function",
                        replacement_body="return item\n",
                    ),
                ),
                summary=f"rewrite {plan.id}",
                transformation_id=plan.steps[0].stable_id,
            ),
            mode="batch-quiescent",
        )

    def batch_lowering(
        project_root: Path,
        plan: SourceOptimizationPlan,
        assessment: SourceOptimizationAssessment,
    ) -> SourceLoweringResult:
        if plan.id == second_plan.id:
            return SourceLoweringResult(
                plan_id=plan.id,
                status="unsupported",
                request=None,
                rejections=("batch skipped",),
                mode="batch-quiescent",
            )
        return lowering(project_root, plan, assessment)

    def state_machine_lowering(
        project_root: Path,
        plan: SourceOptimizationPlan,
        assessment: SourceOptimizationAssessment,
    ) -> SourceLoweringResult:
        if plan.id == first_plan.id:
            return SourceLoweringResult(
                plan_id=plan.id,
                status="unsupported",
                request=None,
                rejections=("state machine skipped",),
                mode="state-machine",
            )
        lowered = lowering(project_root, plan, assessment)
        return replace(lowered, mode="state-machine")

    def no_state_machine(
        _project_root: Path,
        plan: SourceOptimizationPlan,
        _assessment: SourceOptimizationAssessment,
    ) -> SourceLoweringResult:
        return SourceLoweringResult(
            plan_id=plan.id,
            status="unsupported",
            request=None,
            rejections=("state machine skipped",),
            mode="state-machine",
        )

    del no_state_machine

    def fake_run(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        mode: str,
        **_options: object,
    ) -> CommandRunEvidence:
        if command == ("test",):
            duration = 0.05
        elif mode == "baseline":
            duration = 0.8
        else:
            duration = candidate_payload_durations.setdefault(
                payload_root,
                0.1 if not candidate_payload_durations else 0.2,
            )
        return _command_evidence(command, (project_root, payload_root), mode, duration)

    monkeypatch.setattr(
        "atoll.source_optimization.search.lower_batch_quiescent_plan",
        batch_lowering,
    )
    monkeypatch.setattr(
        "atoll.source_optimization.search.lower_state_machine_plan",
        state_machine_lowering,
    )
    monkeypatch.setattr("atoll.source_optimization.search.run_performance_command", fake_run)

    result = run_source_optimization_search(
        (first_plan, second_plan),
        (first_assessment, second_assessment),
        options,
    )

    rejected_trials = [trial for trial in result.trials if trial.status == "rejected"]
    assert rejected_trials
    assert rejected_trials[0].reason == "candidate did not improve the current beam payload"
    assert rejected_trials[0].current_median_seconds == pytest.approx(0.1)


@pytest.mark.parametrize("fail_on_sample", [False, True])
def test_final_source_gate_command_failures_are_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_on_sample: bool,
) -> None:
    """Source final-gate warmup and sample command failures reject the winner."""
    project_root = tmp_path / "project"
    _copy_fixture(project_root)
    plan, assessment = _plan_and_assessment(project_root)
    options = _search_options(project_root, tmp_path)
    benchmark_calls_after_search = 0
    search_benchmark_calls = 12

    def no_state_machine(
        _project_root: Path,
        plan: SourceOptimizationPlan,
        _assessment: SourceOptimizationAssessment,
    ) -> SourceLoweringResult:
        return SourceLoweringResult(
            plan_id=plan.id,
            status="unsupported",
            request=None,
            rejections=("state machine skipped",),
            mode="state-machine",
        )

    def fake_run(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        mode: str,
        **_options: object,
    ) -> CommandRunEvidence:
        nonlocal benchmark_calls_after_search
        returncode = 0
        if command == ("benchmark",):
            benchmark_calls_after_search += 1
            final_gate_call = benchmark_calls_after_search > search_benchmark_calls
            failing_call = 3 if fail_on_sample else 1
            if (
                final_gate_call
                and benchmark_calls_after_search == search_benchmark_calls + failing_call
            ):
                returncode = 5
        duration = 0.05 if command == ("test",) else (0.8 if mode == "baseline" else 0.1)
        return _command_evidence(
            command,
            (project_root, payload_root),
            mode,
            duration,
            result=(returncode, "", ""),
        )

    monkeypatch.setattr(
        "atoll.source_optimization.search.lower_state_machine_plan",
        no_state_machine,
    )
    monkeypatch.setattr("atoll.source_optimization.search.run_performance_command", fake_run)

    result = run_source_optimization_search((plan,), (assessment,), options)

    assert result.accepted is False
    assert result.performance is not None
    assert result.performance.status == "invalid"
    expected_reason = "source sample exited 5" if fail_on_sample else "source warmup exited 5"
    assert result.performance.reason == expected_reason


def test_final_source_gate_rejects_noisy_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final source gate can reject an otherwise fast candidate as invalid."""
    project_root = tmp_path / "project"
    _copy_fixture(project_root)
    plan, assessment = _plan_and_assessment(project_root)
    options = _search_options(project_root, tmp_path)

    def fake_run(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        mode: str,
        **_options: object,
    ) -> CommandRunEvidence:
        duration = 0.05 if command == ("test",) else (0.2 if mode == "baseline" else 0.01)
        return _command_evidence(command, (project_root, payload_root), mode, duration)

    monkeypatch.setattr("atoll.source_optimization.search.run_performance_command", fake_run)

    result = run_source_optimization_search((plan,), (assessment,), options)

    assert result.accepted is False
    assert result.performance is not None
    assert result.performance.status == "invalid"
    assert "baseline median is too noisy" in result.performance.reason
    assert any(trial.status == "not-profitable" for trial in result.trials)


def test_candidate_wheel_build_failure_rejects_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed normal PEP 517 build turns the winner into unavailable evidence."""
    project_root = tmp_path / "project"
    _copy_fixture(project_root)
    plan, assessment = _plan_and_assessment(project_root)
    options = _search_options(project_root, tmp_path)

    def fake_run(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        mode: str,
        **_options: object,
    ) -> CommandRunEvidence:
        duration = 0.05 if command == ("test",) else (0.8 if mode == "baseline" else 0.1)
        return _command_evidence(command, (project_root, payload_root), mode, duration)

    def fail_build(_project_root: Path, _output_dir: Path) -> WheelBuildEvidence:
        return WheelBuildEvidence(
            command=("build",),
            project_root=project_root,
            outdir=tmp_path,
            returncode=2,
            stdout="",
            stderr="build exploded",
            duration_seconds=0.3,
            wheel_paths=(),
        )

    monkeypatch.setattr("atoll.source_optimization.search.run_performance_command", fake_run)
    monkeypatch.setattr("atoll.source_optimization.search.build_baseline_wheel", fail_build)

    result = run_source_optimization_search((plan,), (assessment,), options)

    assert result.accepted is False
    assert result.wheel_path is None
    assert any(
        trial.status == "unavailable" and trial.reason == "build exploded"
        for trial in result.trials
    )


def test_candidate_wheel_unpack_failure_rejects_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wheel that builds but cannot be unpacked is not promoted."""
    project_root = tmp_path / "project"
    _copy_fixture(project_root)
    plan, assessment = _plan_and_assessment(project_root)
    options = _search_options(project_root, tmp_path)

    def fake_run(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        mode: str,
        **_options: object,
    ) -> CommandRunEvidence:
        duration = 0.05 if command == ("test",) else (0.8 if mode == "baseline" else 0.1)
        return _command_evidence(command, (project_root, payload_root), mode, duration)

    def fake_build(_project_root: Path, output_dir: Path) -> WheelBuildEvidence:
        output_dir.mkdir(parents=True)
        wheel_path = output_dir / "candidate-py3-none-any.whl"
        wheel_path.write_bytes(b"not a wheel")
        return WheelBuildEvidence(
            command=("build",),
            project_root=project_root,
            outdir=output_dir,
            returncode=0,
            stdout="",
            stderr="",
            duration_seconds=0.3,
            wheel_paths=(wheel_path,),
        )

    def fail_unpack(_wheel_path: Path, _payload_root: Path) -> None:
        raise OSError("cannot unpack")

    monkeypatch.setattr("atoll.source_optimization.search.run_performance_command", fake_run)
    monkeypatch.setattr("atoll.source_optimization.search.build_baseline_wheel", fake_build)
    monkeypatch.setattr("atoll.source_optimization.search.unpack_wheel_payload", fail_unpack)

    result = run_source_optimization_search((plan,), (assessment,), options)

    assert result.accepted is False
    assert result.wheel_path is None
    assert any(
        trial.status == "unavailable"
        and trial.reason == "transformed PEP 517 build produced 1 wheels"
        for trial in result.trials
    )


def test_wheel_semantic_and_final_gate_failures_reject_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wheel semantic failures and wheel speedup misses are both final rejections."""
    project_root = tmp_path / "project"
    _copy_fixture(project_root)
    plan, assessment = _plan_and_assessment(project_root)
    options = _search_options(project_root, tmp_path)

    def fake_wheel(_winner: object) -> tuple[WheelBuildEvidence, Path, Path]:
        wheel_path = tmp_path / "candidate-py3-none-any.whl"
        wheel_path.write_bytes(b"wheel")
        payload = tmp_path / "wheel-payload"
        payload.mkdir(exist_ok=True)
        evidence = WheelBuildEvidence(
            command=("build",),
            project_root=project_root,
            outdir=tmp_path,
            returncode=0,
            stdout="",
            stderr="",
            duration_seconds=0.2,
            wheel_paths=(wheel_path,),
        )
        return evidence, wheel_path, payload

    def semantic_failure_run(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        mode: str,
        **_options: object,
    ) -> CommandRunEvidence:
        wheel_semantic = command == ("test",) and payload_root == tmp_path / "wheel-payload"
        duration = 0.05 if command == ("test",) else (0.8 if mode == "baseline" else 0.1)
        return _command_evidence(
            command,
            (project_root, payload_root),
            mode,
            duration,
            result=(WHEEL_SEMANTIC_FAILURE_CODE if wheel_semantic else 0, "", ""),
        )

    monkeypatch.setattr("atoll.source_optimization.search._build_candidate_wheel", fake_wheel)
    monkeypatch.setattr(
        "atoll.source_optimization.search.run_performance_command", semantic_failure_run
    )
    semantic_result = run_source_optimization_search((plan,), (assessment,), options)

    assert semantic_result.accepted is False
    assert any(
        trial.status == "failed-semantics"
        and trial.reason == "normally built transformed wheel failed semantic tests"
        and trial.semantic_exit_code == WHEEL_SEMANTIC_FAILURE_CODE
        for trial in semantic_result.trials
    )

    def wheel_gate_failure_run(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        mode: str,
        **_options: object,
    ) -> CommandRunEvidence:
        del project_root
        wheel_payload = tmp_path / "wheel-payload"
        if command == ("test",):
            duration = 0.05
        elif payload_root == wheel_payload and mode == "compiled":
            duration = 0.4
        elif mode == "baseline":
            duration = 0.8
        else:
            duration = 0.1
        return _command_evidence(command, (options.project_root, payload_root), mode, duration)

    monkeypatch.setattr(
        "atoll.source_optimization.search.run_performance_command", wheel_gate_failure_run
    )
    wheel_gate_result = run_source_optimization_search((plan,), (assessment,), options)

    assert wheel_gate_result.accepted is False
    assert wheel_gate_result.performance is not None
    assert wheel_gate_result.performance.status == "not-profitable"
    assert "normally built wheel gate rejected candidate" in next(
        trial.reason for trial in wheel_gate_result.trials if trial.status == "not-profitable"
    )


def test_application_exception_cleans_scratch_and_preserves_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected transactional apply exceptions do not leave scratch state behind."""
    project_root = tmp_path / "project"
    _copy_fixture(project_root)
    plan, assessment = _plan_and_assessment(project_root)
    options = replace(_search_options(project_root, tmp_path), apply_source=True)
    source_before = (project_root / SOURCE_PATH).read_bytes()

    def fake_run(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        mode: str,
        **_options: object,
    ) -> CommandRunEvidence:
        duration = 0.05 if command == ("test",) else (0.8 if mode == "baseline" else 0.1)
        return _command_evidence(command, (project_root, payload_root), mode, duration)

    def fake_wheel(_winner: object) -> tuple[WheelBuildEvidence, Path, Path]:
        wheel_path = tmp_path / "candidate-py3-none-any.whl"
        wheel_path.write_bytes(b"wheel")
        payload = tmp_path / "wheel-payload"
        payload.mkdir(exist_ok=True)
        return (
            WheelBuildEvidence(
                command=("build",),
                project_root=project_root,
                outdir=tmp_path,
                returncode=0,
                stdout="",
                stderr="",
                duration_seconds=0.2,
                wheel_paths=(wheel_path,),
            ),
            wheel_path,
            payload,
        )

    def raise_application(
        _root: Path,
        _patch_path: Path,
        _patch: GeneratedSourcePatch,
        _validate_callback: Callable[[], tuple[bool, tuple[str, ...]]],
    ) -> SourcePatchApplicationResult:
        raise RuntimeError("transaction died")

    def accept_application_root(_root: Path) -> None:
        return None

    monkeypatch.setattr("atoll.source_optimization.search.run_performance_command", fake_run)
    monkeypatch.setattr("atoll.source_optimization.search._build_candidate_wheel", fake_wheel)
    monkeypatch.setattr(
        "atoll.source_optimization.search.validate_source_application_root",
        accept_application_root,
    )
    monkeypatch.setattr(
        "atoll.source_optimization.search.apply_source_patch_transactionally",
        raise_application,
    )

    with pytest.raises(RuntimeError, match="transaction died"):
        run_source_optimization_search((plan,), (assessment,), options)

    assert not options.scratch_root.exists()
    assert (project_root / SOURCE_PATH).read_bytes() == source_before


def test_applied_validation_semantic_failure_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Application validation reports semantic failures through rollback diagnostics."""
    project_root = tmp_path / "project"
    _copy_fixture(project_root)
    plan, assessment = _plan_and_assessment(project_root)
    options = replace(_search_options(project_root, tmp_path), apply_source=True)
    source_before = (project_root / SOURCE_PATH).read_bytes()

    def fake_run(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        mode: str,
        **_options: object,
    ) -> CommandRunEvidence:
        applied_semantic = command == ("test",) and "applied-quality" in project_root.parts
        duration = 0.05 if command == ("test",) else (0.8 if mode == "baseline" else 0.1)
        return _command_evidence(
            command,
            (project_root, payload_root),
            mode,
            duration,
            result=(11 if applied_semantic else 0, "", ""),
        )

    def fake_wheel(_winner: object) -> tuple[WheelBuildEvidence, Path, Path]:
        wheel_path = tmp_path / "candidate-py3-none-any.whl"
        wheel_path.write_bytes(b"wheel")
        payload = tmp_path / "wheel-payload"
        payload.mkdir(exist_ok=True)
        return (
            WheelBuildEvidence(
                command=("build",),
                project_root=project_root,
                outdir=tmp_path,
                returncode=0,
                stdout="",
                stderr="",
                duration_seconds=0.2,
                wheel_paths=(wheel_path,),
            ),
            wheel_path,
            payload,
        )

    def validating_application(
        root: Path,
        _patch_path: Path,
        patch: GeneratedSourcePatch,
        validate_callback: Callable[[], tuple[bool, tuple[str, ...]]],
    ) -> SourcePatchApplicationResult:
        for transformed in patch.files:
            (root / transformed.path).write_text(transformed.after_source, encoding="utf-8")
        succeeded, diagnostics = validate_callback()
        if not succeeded:
            for transformed in patch.files:
                (root / transformed.path).write_text(
                    transformed.before_source,
                    encoding="utf-8",
                )
        return SourcePatchApplicationResult(
            "applied" if succeeded else "rolled-back",
            diagnostics,
        )

    def accept_application_root(_root: Path) -> None:
        return None

    monkeypatch.setattr("atoll.source_optimization.search.run_performance_command", fake_run)
    monkeypatch.setattr("atoll.source_optimization.search._build_candidate_wheel", fake_wheel)
    monkeypatch.setattr(
        "atoll.source_optimization.search.validate_source_application_root",
        accept_application_root,
    )
    monkeypatch.setattr(
        "atoll.source_optimization.search.apply_source_patch_transactionally",
        validating_application,
    )

    result = run_source_optimization_search((plan,), (assessment,), options)

    assert result.accepted is False
    assert result.error == "applied semantic command exited 11"
    assert (project_root / SOURCE_PATH).read_bytes() == source_before
    assert any(trial.application_status == "rolled-back" for trial in result.trials)


def test_workspace_copy_ignores_outputs_and_rejects_unsafe_module_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidate staging strips build artifacts and fails unsafe helper paths."""
    project_root = tmp_path / "project"
    _copy_fixture(project_root)
    output_dir = project_root / ".atoll" / "dist"
    output_dir.mkdir(parents=True)
    (output_dir / "old.whl").write_bytes(b"wheel")
    (project_root / ".git").mkdir()
    (project_root / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (project_root / "extension.so").write_bytes(b"compiled")
    (project_root / ".venv").mkdir()
    (project_root / ".venv" / "marker").write_text("ignored", encoding="utf-8")
    extra_source = project_root / "extra_src"
    extra_source.mkdir()
    (extra_source / "standalone.py").write_text("VALUE = 1\n", encoding="utf-8")
    (extra_source / "extra_pkg").mkdir()
    (extra_source / "extra_pkg" / "__init__.py").write_text("", encoding="utf-8")
    plan, assessment = _plan_and_assessment(project_root)
    options = replace(
        _search_options(project_root, tmp_path),
        source_roots=(project_root / "src", extra_source),
    )
    observed_quality_roots: list[Path] = []
    observed_payload_roots: list[Path] = []
    workspace_checks: list[bool] = []

    def fake_run(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        mode: str,
        **_options: object,
    ) -> CommandRunEvidence:
        observed_quality_roots.append(project_root)
        observed_payload_roots.append(payload_root)
        if not workspace_checks:
            copied_project = project_root.parent / "project"
            workspace_checks.append(
                not (copied_project / ".atoll" / "dist" / "old.whl").exists()
                and not (copied_project / "extension.so").exists()
                and not (copied_project / ".venv").exists()
                and (payload_root / "source_optimization_fixture").is_dir()
                and (payload_root / "standalone.py").is_file()
                and (payload_root / "extra_pkg").is_dir()
            )
        duration = 0.05 if command == ("test",) else (0.8 if mode == "baseline" else 0.1)
        return _command_evidence(command, (project_root, payload_root), mode, duration)

    monkeypatch.setattr("atoll.source_optimization.search.run_performance_command", fake_run)

    result = run_source_optimization_search((plan,), (assessment,), options)

    assert observed_quality_roots
    assert observed_payload_roots
    assert workspace_checks == [True]
    assert result.attempted is True

    unsafe_options = replace(options, module_paths=(Path("../escape.py"),))
    unsafe_result = run_source_optimization_search((plan,), (assessment,), unsafe_options)

    assert unsafe_result.accepted is False
    assert any(
        trial.status == "unavailable" and "unsafe project-relative path" in trial.reason
        for trial in unsafe_result.trials
    )


def test_helper_path_safety_copy_and_progress_edges(tmp_path: Path) -> None:
    """Low-level helpers reject escaping paths and preserve safe copy boundaries."""
    copy_project = cast(
        Callable[..., None],
        source_search.__dict__["_copy_project"],
    )
    source_payload_root = cast(
        Callable[..., Path],
        source_search.__dict__["_source_payload_root"],
    )
    relative_source_root = cast(
        Callable[[Path, Path], Path],
        source_search.__dict__["_relative_source_root"],
    )
    safe_relative_path = cast(
        Callable[[Path, Path], Path],
        source_search.__dict__["_safe_relative_path"],
    )
    reset_dir = cast(Callable[[Path], None], source_search.__dict__["_reset_dir"])
    progress = cast(
        Callable[[Callable[[str], None] | None, str], None],
        source_search.__dict__["_progress"],
    )
    source_search_arm = cast(
        Callable[[str], str],
        source_search.__dict__["_source_search_arm"],
    )

    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / ".git").write_text("gitdir: ../actual-git\n", encoding="utf-8")
    (tmp_path / "actual-git").mkdir()
    excluded_output = source_root / "dist"
    excluded_output.mkdir()
    (excluded_output / "artifact.whl").write_bytes(b"wheel")
    (source_root / "pkg").mkdir()
    (source_root / "pkg" / "__init__.py").write_text("", encoding="utf-8")

    destination = tmp_path / "copy"
    copy_project(source_root, destination, excluded_output=excluded_output)

    assert (destination / ".git").read_text(encoding="utf-8").startswith("gitdir: ")
    assert not (destination / "dist" / "artifact.whl").exists()
    assert relative_source_root(source_root, source_root / "pkg") == Path("pkg")

    with pytest.raises(ValueError, match="source root escapes project root"):
        relative_source_root(source_root, tmp_path)
    with pytest.raises(ValueError, match="unsafe project-relative path"):
        safe_relative_path(source_root, Path(chr(47)) / "absolute.py")
    with pytest.raises(ValueError, match="unsafe project-relative path"):
        safe_relative_path(source_root, Path("../escape.py"))

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "target.py").write_text("", encoding="utf-8")
    (source_root / "link.py").symlink_to(outside / "target.py")
    with pytest.raises(ValueError, match="path escapes temporary project"):
        safe_relative_path(source_root, Path("link.py"))

    left = source_root / "left"
    right = source_root / "right"
    left.mkdir()
    right.mkdir()
    (left / "shared.py").write_text("LEFT = True\n", encoding="utf-8")
    (right / "shared.py").write_text("RIGHT = True\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"source roots overlap at shared\.py"):
        source_payload_root(
            original_root=source_root,
            copied_root=source_root,
            source_roots=(left, right),
            merge_root=tmp_path / "merge",
        )

    existing = tmp_path / "reset"
    existing.mkdir()
    (existing / "old").write_text("", encoding="utf-8")
    reset_dir(existing)
    assert existing.is_dir()
    assert not (existing / "old").exists()

    messages: list[str] = []
    progress(messages.append, "step done")
    progress(None, "ignored")
    assert messages == ["step done"]
    with pytest.raises(ValueError, match="invalid source search arm"):
        source_search_arm("other")


def _plan_and_assessment(
    project_root: Path = FIXTURE_ROOT,
) -> tuple[SourceOptimizationPlan, SourceOptimizationAssessment]:
    source = (project_root / SOURCE_PATH).read_text(encoding="utf-8")
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    identity = SourceOptimizationIdentity(
        execution_plan_id="exec-plan-source-search",
        source_hashes=((SOURCE_PATH, source_hash),),
        topology_fingerprint="taskgroup-private-queue-v1",
        dialect="asyncio",
        lowering_version="source-search-v1",
        python_abi="cp312",
        transformation_versions=(
            ("private-transport-batch-drain", "batch-drain-v1"),
            ("quiescent-callable-execution", "quiescent-callable-v1"),
            ("local-state-machine-fusion", "state-machine-v1"),
        ),
    )
    plan_id = stable_source_optimization_plan_id(identity)
    steps = tuple(
        TransformationStep(
            kind=kind,
            version=version,
            source_symbol=WORKER if "quiescent" in kind else OWNER,
            target_symbol=None,
            access_sites=(),
            semantic_boundary=kind,
            description=f"Apply {kind}.",
        )
        for kind, version in identity.transformation_versions
    )
    plan = SourceOptimizationPlan(
        id=plan_id,
        identity=identity,
        source=SOURCE_PATH,
        owner=OWNER,
        worker=WORKER,
        consumer=OWNER,
        reducer=OWNER,
        transport="queue",
        access_sites=(),
        entrypoint=OWNER,
        steps=steps,
        semantic_boundaries=("fallback before entry", "no retry"),
    )
    assessment = SourceOptimizationAssessment(
        plan_id=plan_id,
        status="trial-ready",
        minimum_speedup=3.0,
        work_items=(WORKER,),
        observed_work_items=20_000,
        immediate_result_ratio=1.0,
        attributed_hot_share=0.9,
        scheduler_overhead_samples=10_000,
        scheduler_overhead_share=0.5,
        scheduler_overhead_evidence=("scheduler overhead",),
        callable_evidence=(
            SourceCallableEvidence(
                symbol=WORKER,
                static_role="worker",
                observed_invocations=20_000,
                completed_calls=20_000,
                immediate_result_ratio=1.0,
                hot_share=0.9,
                context_mutation=("_WORKER_CONTEXT.set",),
            ),
        ),
    )
    return plan, assessment


def _search_options(project_root: Path, tmp_path: Path) -> SourceOptimizationSearchOptions:
    baseline = tmp_path / "baseline"
    quality = tmp_path / "quality"
    baseline.mkdir()
    quality.mkdir()
    return SourceOptimizationSearchOptions(
        project_root=project_root,
        source_roots=(project_root / "src",),
        module_paths=(
            Path("src/source_optimization_fixture/__init__.py"),
            Path(SOURCE_PATH.as_posix()),
        ),
        output_dir=project_root / ".atoll" / "dist",
        scratch_root=project_root / ".atoll" / "dist" / "build" / "source-search",
        cache_root=project_root / ".atoll" / "cache" / "source-optimization",
        baseline_payload_root=baseline,
        quality_project_root=quality,
        compile_config=CompileConfig(
            test_command=("test",),
            benchmark_command=("benchmark",),
            benchmark_samples=7,
            minimum_speedup=1.1,
        ),
        baseline_build=CompileAttempt(
            success=True,
            command=("baseline",),
            stdout="",
            stderr="",
            artifact_paths=(),
            duration_seconds=0.1,
        ),
    )


def _copy_fixture(destination: Path) -> None:
    shutil.copytree(FIXTURE_ROOT, destination)


def _command_evidence(
    command: tuple[str, ...],
    roots: tuple[Path, Path],
    mode: str,
    duration: float,
    *,
    result: tuple[int, str, str] = (0, "", ""),
) -> CommandRunEvidence:
    returncode, stdout, stderr = result
    project_root, payload_root = roots
    return CommandRunEvidence(
        command=command,
        project_root=project_root,
        payload_root=payload_root,
        mode="baseline" if mode == "baseline" else "compiled",
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=duration,
    )
