"""Disposable package-trial tests for scheduler execution plans."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Protocol, cast

import pytest

from atoll.commands import package as package_command
from atoll.execution_plans.models import (
    ChangedPayloadFile,
    ExecutionPlan,
    ExecutionPlanAssessment,
    ExecutionPlanAssessmentContext,
    ExecutionPlanAssessmentStatus,
    ExecutionPlanDiagnostic,
    ExecutionPlanStageContext,
    ExecutionPlanTrial,
    PlanGuard,
    PlanNode,
    StagedExecutionPlan,
)
from atoll.models import CompileAttempt, CompilePhaseTiming, SymbolId
from atoll.project import discover_project
from atoll.runtime.performance import (
    BenchmarkGateConfig,
    BenchmarkGateResult,
    BenchmarkProgress,
    CommandRunEvidence,
    RuntimeMode,
)

_PLAN_BENCHMARK_SAMPLES = 7


class _FakeTaskPreservingBackend:
    """Minimal backend that appends one validated payload marker."""

    name = "task-preserving-test"

    def __init__(
        self,
        *,
        report_extra_change: bool = False,
        assessment_status: ExecutionPlanAssessmentStatus = "supported",
    ) -> None:
        self._report_extra_change = report_extra_change
        self._assessment_status: ExecutionPlanAssessmentStatus = assessment_status

    def assess(
        self,
        plan: ExecutionPlan,
        context: ExecutionPlanAssessmentContext,
    ) -> ExecutionPlanAssessment:
        del context
        supported = tuple(node.id for node in plan.nodes)
        return ExecutionPlanAssessment(
            plan_id=plan.id,
            backend=self.name,
            status=self._assessment_status,
            supported_nodes=supported if self._assessment_status == "supported" else (),
            unsupported_nodes=supported if self._assessment_status == "unsupported" else (),
            reasons=("fixture supports the complete plan",),
        )

    def stage(
        self,
        plan: ExecutionPlan,
        context: ExecutionPlanStageContext,
    ) -> StagedExecutionPlan:
        target = context.payload_root / "app" / "scheduler.py"
        before_hash = _digest(target)
        target.write_text(target.read_text(encoding="utf-8") + "# planned\n", encoding="utf-8")
        if self._report_extra_change:
            (context.payload_root / "app" / "unreported.py").write_text(
                "CHANGED = True\n",
                encoding="utf-8",
            )
        return StagedExecutionPlan(
            plan=plan,
            backend=self.name,
            payload_files=(
                ChangedPayloadFile(
                    install_path=PurePosixPath("app/scheduler.py"),
                    before_hash=before_hash,
                    after_hash=_digest(target),
                    role="source-overlay",
                ),
            ),
            required_imports=(),
            guards=plan.guards,
        )

    def fingerprint(
        self,
        plan: ExecutionPlan,
        context: ExecutionPlanStageContext,
    ) -> str:
        del context
        return hashlib.sha256(f"{self.name}:{plan.id}".encode()).hexdigest()

    def normalize_diagnostic(
        self,
        error: BaseException,
        *,
        diagnostics: str,
        log_path: Path | None,
    ) -> ExecutionPlanDiagnostic:
        del diagnostics, log_path
        return ExecutionPlanDiagnostic(
            code="fixture-error",
            severity="error",
            message=str(error),
        )


class _FakeCallbackBackend(_FakeTaskPreservingBackend):
    """Callback-shaped fake used to assert backend preference and fallback."""

    name = "callback-backed-test"


class _PlanTrialOutcome(Protocol):
    applied_plan_ids: tuple[str, ...]
    trials: tuple[ExecutionPlanTrial, ...]
    timings: tuple[CompilePhaseTiming, ...]


def _package_attr(name: str) -> object:
    return getattr(package_command, name)


_apply_execution_plan_trials = cast(
    Callable[[object], _PlanTrialOutcome],
    _package_attr("_apply_execution_plan_trials"),
)
_ExecutionPlanApplicationContext = cast(
    Callable[..., object],
    _package_attr("_ExecutionPlanApplicationContext"),
)
_BaselineWheelPayload = cast(
    Callable[..., object],
    _package_attr("_BaselineWheelPayload"),
)


def test_passing_execution_plan_replaces_payload_after_disposable_trial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A semantic and marginal-performance pass promotes only the staged copy."""
    context, source_path, install_path = _trial_context(tmp_path)
    original_source = source_path.read_text(encoding="utf-8")
    original_payload = install_path.read_text(encoding="utf-8")
    events: list[str] = []

    def semantics(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        mode: RuntimeMode,
        region_allowlist: frozenset[str] | None = None,
    ) -> CommandRunEvidence:
        events.append("semantic")
        return _passing_semantics(
            command,
            project_root=project_root,
            payload_root=payload_root,
            mode=mode,
            region_allowlist=region_allowlist,
        )

    def benchmark(
        config: BenchmarkGateConfig,
        **kwargs: object,
    ) -> BenchmarkGateResult:
        assert events == ["semantic"]
        assert config.warmups == 1
        assert config.samples == _PLAN_BENCHMARK_SAMPLES
        assert config.minimum_speedup == pytest.approx(1.05)
        progress = cast(Callable[[BenchmarkProgress], None], kwargs["progress"])
        progress(
            BenchmarkProgress(
                phase="sample",
                pair_index=1,
                sample_index=1,
                mode="compiled",
                duration_seconds=0.8,
            )
        )
        events.append("benchmark")
        return _passing_benchmark()

    monkeypatch.setattr(
        package_command,
        "_EXECUTION_PLAN_BACKENDS",
        (_FakeTaskPreservingBackend(),),
    )
    monkeypatch.setattr(package_command, "run_performance_command", semantics)
    monkeypatch.setattr(package_command, "run_benchmark_gate", benchmark)

    outcome = _apply_execution_plan_trials(context)

    assert outcome.applied_plan_ids == ("exec-plan-fixture",)
    assert outcome.trials[0].status == "accepted"
    assert outcome.trials[0].marginal_speedup == pytest.approx(1.25)
    assert install_path.read_text(encoding="utf-8") == original_payload + "# planned\n"
    assert source_path.read_text(encoding="utf-8") == original_source
    assert events == ["semantic", "benchmark"]
    assert {timing.name for timing in outcome.timings} == {
        "execution_plan_staging",
        "execution_plan_semantic_test",
        "execution_plan_benchmark",
    }


def test_rejected_execution_plan_leaves_current_payload_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A candidate below the 1.05x marginal gate cannot alter the wheel payload."""
    context, source_path, install_path = _trial_context(tmp_path)
    original_source = source_path.read_text(encoding="utf-8")
    original_payload = install_path.read_text(encoding="utf-8")
    monkeypatch.setattr(
        package_command,
        "_EXECUTION_PLAN_BACKENDS",
        (_FakeTaskPreservingBackend(),),
    )
    monkeypatch.setattr(package_command, "run_performance_command", _passing_semantics)
    monkeypatch.setattr(package_command, "run_benchmark_gate", _rejected_benchmark)

    outcome = _apply_execution_plan_trials(context)

    assert outcome.applied_plan_ids == ()
    assert outcome.trials[0].status == "rejected"
    assert install_path.read_text(encoding="utf-8") == original_payload
    assert source_path.read_text(encoding="utf-8") == original_source


def test_semantic_failure_stops_before_execution_plan_benchmark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed project semantic command removes the candidate without benchmarking."""
    context, _source_path, install_path = _trial_context(tmp_path)
    original_payload = install_path.read_text(encoding="utf-8")
    benchmark_calls = 0

    def benchmark(*args: object, **kwargs: object) -> BenchmarkGateResult:
        del args, kwargs
        nonlocal benchmark_calls
        benchmark_calls += 1
        raise AssertionError("benchmark must not run after semantic failure")

    monkeypatch.setattr(
        package_command,
        "_EXECUTION_PLAN_BACKENDS",
        (_FakeTaskPreservingBackend(),),
    )
    monkeypatch.setattr(package_command, "run_performance_command", _failing_semantics)
    monkeypatch.setattr(package_command, "run_benchmark_gate", benchmark)

    outcome = _apply_execution_plan_trials(context)

    assert benchmark_calls == 0
    assert outcome.applied_plan_ids == ()
    assert outcome.trials[0].status == "failed-semantics"
    assert outcome.trials[0].benchmark_status == "not-run"
    assert install_path.read_text(encoding="utf-8") == original_payload


def test_rejected_backend_records_unavailable_plan_without_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A complete backend rejection remains report evidence and leaves payload untouched."""
    context, _source_path, install_path = _trial_context(tmp_path)
    original_payload = install_path.read_text(encoding="utf-8")
    monkeypatch.setattr(
        package_command,
        "_EXECUTION_PLAN_BACKENDS",
        (_FakeTaskPreservingBackend(assessment_status="unsupported"),),
    )

    outcome = _apply_execution_plan_trials(context)

    assert outcome.applied_plan_ids == ()
    assert outcome.trials[0].status == "unavailable"
    assert outcome.trials[0].backend is None
    assert install_path.read_text(encoding="utf-8") == original_payload


def test_callback_backend_is_preferred_when_both_backends_support_the_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The strict callback lowering receives the first disposable trial."""
    context, _source_path, _install_path = _trial_context(tmp_path)
    monkeypatch.setattr(
        package_command,
        "_EXECUTION_PLAN_BACKENDS",
        (_FakeCallbackBackend(), _FakeTaskPreservingBackend()),
    )
    monkeypatch.setattr(package_command, "run_performance_command", _passing_semantics)
    monkeypatch.setattr(package_command, "run_benchmark_gate", _passing_benchmark)

    outcome = _apply_execution_plan_trials(context)

    assert outcome.trials[0].status == "accepted"
    assert outcome.trials[0].backend == "callback-backed-test"


def test_task_preserving_backend_follows_a_callback_capability_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A strict callback rejection falls through to real-task lowering."""
    context, _source_path, _install_path = _trial_context(tmp_path)
    monkeypatch.setattr(
        package_command,
        "_EXECUTION_PLAN_BACKENDS",
        (
            _FakeCallbackBackend(assessment_status="unsupported"),
            _FakeTaskPreservingBackend(),
        ),
    )
    monkeypatch.setattr(package_command, "run_performance_command", _passing_semantics)
    monkeypatch.setattr(package_command, "run_benchmark_gate", _passing_benchmark)

    outcome = _apply_execution_plan_trials(context)

    assert outcome.trials[0].status == "accepted"
    assert outcome.trials[0].backend == "task-preserving-test"
    assert [diagnostic.code for diagnostic in outcome.trials[0].diagnostics[:2]] == [
        "backend-rejected",
        "backend-supported",
    ]


def test_task_preserving_backend_follows_a_callback_semantic_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed speculative callback payload cannot suppress the safe backend."""
    context, _source_path, install_path = _trial_context(tmp_path)
    semantic_calls = 0

    def semantics(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        mode: RuntimeMode,
        region_allowlist: frozenset[str] | None = None,
    ) -> CommandRunEvidence:
        nonlocal semantic_calls
        semantic_calls += 1
        runner = _failing_semantics if semantic_calls == 1 else _passing_semantics
        return runner(
            command,
            project_root=project_root,
            payload_root=payload_root,
            mode=mode,
            region_allowlist=region_allowlist,
        )

    monkeypatch.setattr(
        package_command,
        "_EXECUTION_PLAN_BACKENDS",
        (_FakeCallbackBackend(), _FakeTaskPreservingBackend()),
    )
    monkeypatch.setattr(package_command, "run_performance_command", semantics)
    monkeypatch.setattr(package_command, "run_benchmark_gate", _passing_benchmark)

    outcome = _apply_execution_plan_trials(context)

    assert [trial.status for trial in outcome.trials] == ["failed-semantics", "accepted"]
    assert [trial.backend for trial in outcome.trials] == [
        "callback-backed-test",
        "task-preserving-test",
    ]
    assert outcome.applied_plan_ids == ("exec-plan-fixture",)
    assert install_path.read_text(encoding="utf-8").endswith("# planned\n")


def test_unreported_payload_change_rejects_plan_before_semantic_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changed-file validation stops a backend that omits one staged file."""
    context, _source_path, install_path = _trial_context(tmp_path)
    original_payload = install_path.read_text(encoding="utf-8")
    semantic_calls = 0

    def semantics(*args: object, **kwargs: object) -> CommandRunEvidence:
        del args, kwargs
        nonlocal semantic_calls
        semantic_calls += 1
        raise AssertionError("semantic command must not run")

    monkeypatch.setattr(
        package_command,
        "_EXECUTION_PLAN_BACKENDS",
        (_FakeTaskPreservingBackend(report_extra_change=True),),
    )
    monkeypatch.setattr(package_command, "run_performance_command", semantics)

    outcome = _apply_execution_plan_trials(context)

    assert semantic_calls == 0
    assert outcome.applied_plan_ids == ()
    assert outcome.trials[0].status == "unavailable"
    assert "changed-file report is incomplete" in (outcome.trials[0].reason or "")
    assert install_path.read_text(encoding="utf-8") == original_payload


def test_post_swap_backup_cleanup_cannot_invalidate_an_applied_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Best-effort backup cleanup cannot desynchronize payload and trial evidence."""
    context, _source_path, install_path = _trial_context(tmp_path)
    original_payload = install_path.read_text(encoding="utf-8")
    real_rmtree = shutil.rmtree
    backup_cleanup_used_ignore_errors = False

    def rmtree(path: str | Path, ignore_errors: bool = False) -> None:
        nonlocal backup_cleanup_used_ignore_errors
        if Path(path).name == ".accepted-backup":
            backup_cleanup_used_ignore_errors = ignore_errors
            return
        real_rmtree(path, ignore_errors=ignore_errors)

    monkeypatch.setattr(
        package_command,
        "_EXECUTION_PLAN_BACKENDS",
        (_FakeTaskPreservingBackend(),),
    )
    monkeypatch.setattr(package_command, "run_performance_command", _passing_semantics)
    monkeypatch.setattr(package_command, "run_benchmark_gate", _passing_benchmark)
    monkeypatch.setattr(shutil, "rmtree", rmtree)

    outcome = _apply_execution_plan_trials(context)

    assert backup_cleanup_used_ignore_errors is True
    assert outcome.applied_plan_ids == ("exec-plan-fixture",)
    assert outcome.trials[0].status == "accepted"
    assert install_path.read_text(encoding="utf-8") == original_payload + "# planned\n"


def _trial_context(
    tmp_path: Path,
) -> tuple[object, Path, Path]:
    project_root = tmp_path / "project"
    source_path = project_root / "src" / "app" / "scheduler.py"
    source_path.parent.mkdir(parents=True)
    source_text = "async def run() -> None:\n    return None\n"
    source_path.write_text(source_text, encoding="utf-8")
    (project_root / "pyproject.toml").write_text(
        """[project]
name = "execution-plan-fixture"
version = "0.1.0"

[tool.atoll.compile]
test_command = ["python", "verify.py"]
benchmark_command = ["python", "bench.py"]
""",
        encoding="utf-8",
    )
    project = discover_project(project_root)
    install_path = tmp_path / "dist" / "install" / "app" / "scheduler.py"
    install_path.parent.mkdir(parents=True)
    install_path.write_text(source_text, encoding="utf-8")
    quality_root = tmp_path / "quality"
    quality_root.mkdir()
    owner = SymbolId("app.scheduler", "run")
    plan = ExecutionPlan(
        id="exec-plan-fixture",
        source_module="app.scheduler",
        owner=owner,
        dialect="asyncio",
        lowering_version="asyncio-v1",
        source_hash=hashlib.sha256(source_text.encode()).hexdigest(),
        callsite_fingerprint="b" * 64,
        topology_fingerprint="c" * 64,
        nodes=(PlanNode(owner.stable_id, owner, "orchestrator", 1),),
        edges=(),
        guards=(PlanGuard("scheduler", "asyncio", "stdlib asyncio remains active"),),
    )
    baseline = _BaselineWheelPayload(
        wheel_path=tmp_path / "baseline.whl",
        build=CompileAttempt(
            success=True,
            command=("python", "-m", "build"),
            stdout="",
            stderr="",
            artifact_paths=(),
            duration_seconds=0.1,
        ),
        baseline_install_root=tmp_path / "baseline-install",
        quality_project_root=quality_root,
    )
    context = _ExecutionPlanApplicationContext(
        options=package_command.PackageOptions(root=project_root),
        project=project,
        baseline=baseline,
        build_root=tmp_path / "dist" / "build",
        install_root=install_path.parents[1],
        plans=(plan,),
        accepted_region_ids=frozenset({"native-region"}),
    )
    return context, source_path, install_path


def _passing_semantics(
    command: tuple[str, ...],
    *,
    project_root: Path,
    payload_root: Path,
    mode: RuntimeMode,
    region_allowlist: frozenset[str] | None = None,
) -> CommandRunEvidence:
    assert mode == "compiled"
    assert region_allowlist == frozenset({"native-region"})
    return CommandRunEvidence(
        command=command,
        project_root=project_root,
        payload_root=payload_root,
        mode="compiled",
        returncode=0,
        stdout="",
        stderr="",
        duration_seconds=0.2,
    )


def _failing_semantics(
    command: tuple[str, ...],
    *,
    project_root: Path,
    payload_root: Path,
    mode: RuntimeMode,
    region_allowlist: frozenset[str] | None = None,
) -> CommandRunEvidence:
    del region_allowlist
    return CommandRunEvidence(
        command=command,
        project_root=project_root,
        payload_root=payload_root,
        mode=mode,
        returncode=9,
        stdout="",
        stderr="semantic fixture failed",
        duration_seconds=0.2,
    )


def _passing_benchmark(*args: object, **kwargs: object) -> BenchmarkGateResult:
    del args, kwargs
    return BenchmarkGateResult(
        status="passed",
        reason="planned payload met the marginal threshold",
        minimum_speedup=1.05,
        baseline_median_seconds=1.0,
        compiled_median_seconds=0.8,
        speedup=1.25,
        warmups=(),
        samples=(
            _benchmark_sample("baseline", 1.0),
            _benchmark_sample("compiled", 0.8),
        ),
    )


def _rejected_benchmark(*args: object, **kwargs: object) -> BenchmarkGateResult:
    del args, kwargs
    return BenchmarkGateResult(
        status="not-profitable",
        reason="planned payload missed the marginal threshold",
        minimum_speedup=1.05,
        baseline_median_seconds=1.0,
        compiled_median_seconds=1.0,
        speedup=1.0,
        warmups=(),
        samples=(),
    )


def _benchmark_sample(mode: RuntimeMode, duration_seconds: float) -> CommandRunEvidence:
    return CommandRunEvidence(
        command=("python", "bench.py"),
        project_root=Path.cwd(),
        payload_root=Path.cwd(),
        mode=mode,
        returncode=0,
        stdout="",
        stderr="",
        duration_seconds=duration_seconds,
    )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
