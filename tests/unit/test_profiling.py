"""Unit tests for baseline profiling evidence and candidate selection."""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import os
import subprocess
import sys
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, replace
from dis import get_instructions
from pathlib import Path
from types import FrameType, FunctionType
from typing import Protocol, cast

import pytest

from atoll.models import SymbolId, SymbolRecord
from atoll.runtime import profiling
from atoll.runtime.profiling import (
    CanonicalTypeObservation,
    LifecycleCounts,
    MappedCandidateDecision,
    ObservedSignature,
    ProfileCallEdgeTarget,
    ProfiledMember,
    ProfileResult,
    ProfileSpawnSiteTarget,
    run_baseline_profile,
    select_profile_candidates,
    unconfigured_profile,
)

MAX_SIGNATURES_PER_MEMBER = 8
MAX_TYPE_OBSERVATIONS_PER_MEMBER = 256
ASYNC_WORKER_CALLS = 5
ASYNC_CAPPED_WORKER_CALLS = 300
FREQUENT_CALLS = 800_000
SPAWN_SITE_CALLS = 1_200
DIRECT_CALL_EDGE_CALLS = 1_200
DIRECT_SPAWN_CALLBACKS = 2
SCHEDULER_OVERHEAD_SAMPLES = 60
SCHEDULER_OVERHEAD_COVERAGE = 0.6
ATTRIBUTED_OWNER_LEAF_SAMPLES = 10
ATTRIBUTED_OWNER_SCHEDULER_SAMPLES = 70
ATTRIBUTED_OWNER_SAMPLES = 80
ATTRIBUTED_OWNER_COVERAGE = 0.4
SELECTED_ACTIVITY_SAMPLES = 170
BOOTSTRAP_EXIT_CODE = 7
BOOTSTRAP_USAGE_ERROR_CODE = 2
BOOTSTRAP_STRING_EXIT_CODE = 1


class SubprocessInvocationView(Protocol):
    command: tuple[str, ...]
    env: dict[str, str]


class ProfileBootstrapModule(Protocol):
    @staticmethod
    def main(argv: tuple[str, ...] | None = None) -> int: ...


class CallbackCaseModule(Protocol):
    OBSERVER: Callable[[object, int], object]
    hot: Callable[..., object]
    nested_code: Callable[[], object]


@dataclass(frozen=True, slots=True)
class _TypeProfilerHarness:
    wrapped: object

    def start(self) -> None:
        _callable_attribute(self.wrapped, "start")()

    def stop(self) -> None:
        _callable_attribute(self.wrapped, "stop")()

    def payload(self) -> dict[str, object]:
        return cast(dict[str, object], _callable_attribute(self.wrapped, "payload")())

    @property
    def observer(self) -> Callable[[object, int], object]:
        return cast(
            Callable[[object, int], object],
            _callable_attribute(self.wrapped, "_on_start"),
        )

    def lifecycle_callback(self, name: str) -> Callable[..., object]:
        factory = cast(
            Callable[[str], Callable[..., object]],
            _callable_attribute(self.wrapped, "_lifecycle_callback"),
        )
        return factory(name)

    def call_callback(
        self,
        code: object,
        offset: int,
        callable_object: object,
        arg0: object,
    ) -> object:
        callback = _callable_attribute(self.wrapped, "_on_call")
        return callback(code, offset, callable_object, arg0)


@dataclass(frozen=True, slots=True)
class _SamplingProfilerHarness:
    wrapped: object

    def sample(self, frame: FrameType | None) -> None:
        _callable_attribute(self.wrapped, "_sample")(0, frame)

    def payload(self) -> dict[str, object]:
        return cast(dict[str, object], _callable_attribute(self.wrapped, "payload")())


@dataclass(frozen=True, slots=True)
class _BootstrapConfigInput:
    profile_stage: str
    launch_kind: str
    target: str
    args: tuple[str, ...] = ()
    module_paths: tuple[tuple[str, str], ...] = ()
    enable_atoll: bool = False
    targets: tuple[str, ...] = ()
    spawn_targets: tuple[ProfileSpawnSiteTarget, ...] = ()
    call_edge_targets: tuple[ProfileCallEdgeTarget, ...] = ()


def test_unconfigured_profile_returns_deterministic_no_benchmark_evidence() -> None:
    first = unconfigured_profile()
    second = unconfigured_profile()

    assert first == second
    assert first.status == "unconfigured"
    assert first.reason == "no benchmark command configured"
    assert first.launch_kind == "unconfigured"
    assert first.total_samples == 0
    assert first.selected_symbols == ()
    assert first.call_edges == ()


def test_profiled_member_reports_conservative_immediate_result_ratio() -> None:
    member = ProfiledMember(
        module="workload",
        qualname="worker",
        samples=0,
        coverage=0.0,
        call_count=10,
        lifecycle=LifecycleCounts(start=10, return_=10, yield_=3, resume=3, unwind=0, throw=0),
        signatures=(),
        polymorphic_overflow=False,
        invocation_count=10,
        completed_calls=10,
        pre_completion_suspensions=3,
    )

    assert member.immediate_result_ratio == pytest.approx(0.7)
    assert replace(member, pre_completion_suspensions=30).immediate_result_ratio == 0.0
    assert replace(member, completed_calls=0).immediate_result_ratio == 0.0


def test_profile_preserves_nested_scheduler_overhead_attribution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(invocation: SubprocessInvocationView) -> subprocess.CompletedProcess[str]:
        config_path = Path(invocation.command[-1])
        config = cast(dict[str, object], json.loads(config_path.read_text(encoding="utf-8")))
        result_path = config["result_path"]
        profile_stage = config["profile_stage"]
        assert isinstance(result_path, str)
        assert isinstance(profile_stage, str)
        payload: dict[str, object]
        if profile_stage == "sampling":
            payload = {
                "total_samples": 100,
                "sample_counts": {"workload::worker": 40},
                "scheduler_overhead_counts": {"workload::worker": SCHEDULER_OVERHEAD_SAMPLES},
                "lifecycle": {},
                "member_lifecycle": {},
                "signatures": {},
                "spawn_sites": [],
            }
        else:
            payload = {
                "total_samples": 0,
                "sample_counts": {},
                "scheduler_overhead_counts": {},
                "lifecycle": {},
                "member_lifecycle": {
                    "workload::worker": {
                        "start": 100,
                        "return": 100,
                        "yield": 0,
                        "resume": 0,
                        "unwind": 0,
                        "throw": 0,
                    }
                },
                "signatures": {
                    "workload::worker": {
                        "call_count": 100,
                        "invocation_count": 100,
                        "completed_calls": 100,
                        "max_active_calls": 1,
                        "pre_completion_suspensions": 0,
                        "polymorphic_overflow": False,
                        "observation_capped": False,
                        "signatures": [],
                    }
                },
                "spawn_sites": [],
            }
        Path(result_path).write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(invocation.command, 0, "", "")

    monkeypatch.setattr(profiling, "_run_subprocess", fake_run)

    result = run_baseline_profile(
        (sys.executable, "bench.py"),
        project_root=tmp_path,
        payload_root=tmp_path,
        module_paths=(("workload", "workload.py"),),
        scratch_dir=tmp_path / "scratch",
    )

    member = _member(result, "workload", "worker")
    assert result.scheduler_overhead_samples == SCHEDULER_OVERHEAD_SAMPLES
    assert result.scheduler_overhead_coverage == pytest.approx(SCHEDULER_OVERHEAD_COVERAGE)
    assert member.scheduler_overhead_samples == SCHEDULER_OVERHEAD_SAMPLES
    assert member.scheduler_overhead_coverage == pytest.approx(SCHEDULER_OVERHEAD_COVERAGE)
    assert member.immediate_result_ratio == 1.0
    assert result.call_edges == ()


def test_sampling_profiler_attributes_unmapped_leaf_to_project_ancestor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "project_stack.py"
    source_path.write_text("def project_call(callback):\n    callback()\n", encoding="utf-8")
    config_path, _result_path = _write_bootstrap_config(
        tmp_path,
        _BootstrapConfigInput(
            profile_stage="sampling",
            launch_kind="script",
            target="unused.py",
            module_paths=(("project_stack", "project_stack.py"),),
        ),
    )
    _prepare_in_process_bootstrap(monkeypatch, tmp_path, "project_stack", add_root=True)
    profiler = _sampling_profiler(config_path)
    project_stack = importlib.import_module("project_stack")
    project_call = cast(
        Callable[[Callable[[], None]], None],
        _callable_attribute(project_stack, "project_call"),
    )

    def external_leaf() -> None:
        frame = inspect.currentframe()
        assert frame is not None
        profiler.sample(frame)

    project_call(external_leaf)

    payload = profiler.payload()
    overhead = cast(dict[str, int], payload["scheduler_overhead_counts"])
    assert payload["sample_counts"] == {}
    assert overhead == {"project_stack::project_call": 1}


def test_sampling_profiler_ignores_fully_unmapped_stack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An external stack terminates without inventing project attribution."""
    config_path, _result_path = _write_bootstrap_config(
        tmp_path,
        _BootstrapConfigInput(
            profile_stage="sampling",
            launch_kind="script",
            target="unused.py",
        ),
    )
    _prepare_in_process_bootstrap(monkeypatch, tmp_path)
    profiler = _sampling_profiler(config_path)
    frame = inspect.currentframe()
    assert frame is not None

    profiler.sample(frame)

    payload = profiler.payload()
    assert payload["total_samples"] == 1
    assert payload["sample_counts"] == {}
    assert payload["scheduler_overhead_counts"] == {}


def test_unsupported_launcher_returns_static_fallback_without_scratch_files(tmp_path: Path) -> None:
    scratch_dir = tmp_path / "scratch"

    result = run_baseline_profile(
        ("pytest", "tests"),
        project_root=tmp_path,
        payload_root=tmp_path,
        module_paths=(("workload", "workload.py"),),
        scratch_dir=scratch_dir,
    )

    assert result.status == "static-fallback"
    assert result.launch_kind == "unsupported"
    assert result.runs == ()
    assert not list(scratch_dir.glob("*.json")) if scratch_dir.exists() else True


def test_script_launch_collects_lifecycle_types_and_cleans_scratch(tmp_path: Path) -> None:
    _write_workload_project(tmp_path)

    result = run_baseline_profile(
        (sys.executable, "bench_script.py", "9"),
        project_root=tmp_path,
        payload_root=tmp_path,
        module_paths=(("workload", "workload.py"),),
        scratch_dir=tmp_path / "scratch",
    )

    assert result.status == "profiled"
    assert result.launch_kind == "script"
    assert [run.pass_kind for run in result.runs] == ["sampling", "types"]
    assert result.runs[0].stdout == "script-done\n"
    assert result.runs[1].stdout == "script-done\n"
    assert result.lifecycle.start > 0
    assert result.lifecycle.return_ > 0
    assert any(
        member.module == "workload" and member.qualname == "hot" for member in result.members
    )
    hot = _member(result, "workload", "hot")
    assert hot.lifecycle.start > 0
    assert hot.lifecycle.return_ > 0
    assert hot.call_count > 0
    assert hot.signatures
    assert not list((tmp_path / "scratch").glob("*.json"))


def test_module_launch_collects_project_samples(tmp_path: Path) -> None:
    _write_workload_project(tmp_path)

    result = run_baseline_profile(
        (sys.executable, "-m", "benchpkg.runner", "7"),
        project_root=tmp_path,
        payload_root=tmp_path,
        module_paths=(("workload", "workload.py"),),
        scratch_dir=tmp_path / "scratch",
    )

    assert result.status == "profiled"
    assert result.launch_kind == "module"
    assert result.runs[0].stdout == "module-done\n"
    assert result.mapped_project_samples >= 0


def test_observation_targets_track_async_overlap_and_zero_sample_members(
    tmp_path: Path,
) -> None:
    _write_async_observation_project(tmp_path)

    result = run_baseline_profile(
        (sys.executable, "bench_async.py"),
        project_root=tmp_path,
        payload_root=tmp_path,
        module_paths=(("async_case", "async_case.py"),),
        scratch_dir=tmp_path / "scratch",
        observation_targets=(
            SymbolId("async_case", "worker"),
            SymbolId("async_case", "zero_sample"),
            SymbolId("async_case", "dormant"),
        ),
    )

    worker = _member(result, "async_case", "worker")
    zero_sample = _member(result, "async_case", "zero_sample")
    dormant = _member(result, "async_case", "dormant")
    assert worker.call_count == ASYNC_WORKER_CALLS
    assert worker.completed_calls == ASYNC_WORKER_CALLS
    assert worker.max_active_calls > 1
    assert worker.pre_completion_suspensions >= ASYNC_WORKER_CALLS
    assert zero_sample.call_count == 1
    assert zero_sample.completed_calls == 1
    assert dormant.samples == 0
    assert dormant.call_count == 0
    assert dormant.completed_calls == 0


def test_capped_async_type_observation_keeps_counting_lifecycle(tmp_path: Path) -> None:
    _write_capped_async_observation_project(tmp_path)

    result = run_baseline_profile(
        (sys.executable, "bench_async_capped.py"),
        project_root=tmp_path,
        payload_root=tmp_path,
        module_paths=(("async_capped", "async_capped.py"),),
        scratch_dir=tmp_path / "scratch",
        observation_targets=(SymbolId("async_capped", "worker"),),
    )

    worker = _member(result, "async_capped", "worker")
    assert worker.observation_capped is True
    assert worker.call_count == MAX_TYPE_OBSERVATIONS_PER_MEMBER
    assert worker.invocation_count == ASYNC_CAPPED_WORKER_CALLS
    assert worker.completed_calls == ASYNC_CAPPED_WORKER_CALLS
    assert worker.lifecycle.start == ASYNC_CAPPED_WORKER_CALLS
    assert worker.max_active_calls > 1


def test_targeted_profile_counts_exact_spawn_site_beyond_type_budget(tmp_path: Path) -> None:
    """Scheduler call counts remain exact after argument-type sampling caps."""
    _write_spawn_site_project(tmp_path)
    target = ProfileSpawnSiteTarget(
        id="spawn-site-test",
        owner=SymbolId("scheduler_case", "run"),
        lineno=10,
        col_offset=12,
        scheduler_method="create_task",
    )

    result = run_baseline_profile(
        (sys.executable, "bench_scheduler.py"),
        project_root=tmp_path,
        payload_root=tmp_path,
        module_paths=(("scheduler_case", "scheduler_case.py"),),
        scratch_dir=tmp_path / "scratch",
        observation_targets=(
            SymbolId("scheduler_case", "run"),
            SymbolId("scheduler_case", "worker"),
        ),
        spawn_targets=(target,),
    )

    worker = _member(result, "scheduler_case", "worker")
    spawn = result.spawn_sites[0]
    assert spawn.target == target
    assert spawn.invocation_count == SPAWN_SITE_CALLS
    assert sum(item.count for item in spawn.callable_identities) == SPAWN_SITE_CALLS
    assert all("object at" not in item.identity for item in spawn.callable_identities)
    assert worker.call_count == MAX_TYPE_OBSERVATIONS_PER_MEMBER
    assert worker.invocation_count == SPAWN_SITE_CALLS
    assert worker.lifecycle.start == SPAWN_SITE_CALLS


def test_targeted_profile_counts_canonical_same_module_direct_call_edge(tmp_path: Path) -> None:
    """Direct edge evidence stays exact after bounded type observation stops."""
    source = _write_call_edge_project(tmp_path)
    call = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "leaf"
    )
    target = ProfileCallEdgeTarget(
        id="call-edge-test",
        owner=SymbolId("call_edge_case", "root"),
        callee=SymbolId("call_edge_case", "leaf"),
        lineno=call.lineno,
        col_offset=call.col_offset,
        end_lineno=call.end_lineno,
        end_col_offset=call.end_col_offset,
    )

    result = run_baseline_profile(
        (sys.executable, "bench_call_edge.py"),
        project_root=tmp_path,
        payload_root=tmp_path,
        module_paths=(("call_edge_case", "call_edge_case.py"),),
        scratch_dir=tmp_path / "scratch",
        call_edge_targets=(target,),
    )

    assert result.call_edges[0].target == target
    assert result.call_edges[0].invocation_count == DIRECT_CALL_EDGE_CALLS
    assert _member(result, "call_edge_case", "root").invocation_count == 1


def test_profile_call_edge_target_rejects_cross_module_edge() -> None:
    with pytest.raises(ValueError, match="within one module"):
        ProfileCallEdgeTarget(
            id="cross-module",
            owner=SymbolId("owner_module", "root"),
            callee=SymbolId("callee_module", "leaf"),
            lineno=1,
            col_offset=0,
        )


def test_type_observation_stores_canonical_types_without_values_or_repr(tmp_path: Path) -> None:
    _write_privacy_project(tmp_path)

    result = run_baseline_profile(
        (sys.executable, "bench_privacy.py"),
        project_root=tmp_path,
        payload_root=tmp_path,
        module_paths=(("privacy", "privacy.py"),),
        scratch_dir=tmp_path / "scratch",
    )

    observed = "\n".join(
        type_observation.type_path
        for signature in _member(result, "privacy", "accept").signatures
        for type_observation in signature.parameters
    )
    assert "SecretPayload" in observed
    assert "do-not-leak" not in observed
    assert "repr" not in observed


def test_polymorphic_overflow_is_reported_without_extra_signatures(tmp_path: Path) -> None:
    _write_polymorphic_project(tmp_path)

    result = run_baseline_profile(
        (sys.executable, "bench_poly.py"),
        project_root=tmp_path,
        payload_root=tmp_path,
        module_paths=(("poly", "poly.py"),),
        scratch_dir=tmp_path / "scratch",
    )

    member = _member(result, "poly", "accept")
    assert member.polymorphic_overflow
    assert len(member.signatures) == MAX_SIGNATURES_PER_MEMBER


def test_type_observation_is_bounded_for_frequently_called_hot_member(tmp_path: Path) -> None:
    _write_frequent_call_project(tmp_path)

    result = run_baseline_profile(
        (sys.executable, "bench_frequent.py"),
        project_root=tmp_path,
        payload_root=tmp_path,
        module_paths=(("frequent", "frequent.py"),),
        scratch_dir=tmp_path / "scratch",
    )

    member = _member(result, "frequent", "hot")
    assert member.call_count == MAX_TYPE_OBSERVATIONS_PER_MEMBER
    assert member.observation_capped
    assert sum(signature.count for signature in member.signatures) == member.call_count


def test_select_profile_candidates_applies_threshold_coverage_and_limit_policy() -> None:
    profile = _profile_with_members(
        total_samples=200,
        members=(
            ("pkg.hot", "alpha", 80),
            ("pkg.hot", "beta", 50),
            ("pkg.hot", "gamma", 30),
            ("pkg.hot", "delta", 20),
            ("pkg.hot", "epsilon", 10),
            ("pkg.hot", "zeta", 10),
        ),
    )

    selected = select_profile_candidates(
        profile,
        tuple(_symbol(module, qualname) for module, qualname, _samples in _member_specs(profile)),
    )

    assert selected.selected_symbols == (
        SymbolId("pkg.hot", "alpha"),
        SymbolId("pkg.hot", "beta"),
        SymbolId("pkg.hot", "gamma"),
    )
    assert selected.selected_hot_coverage == 160 / 200
    assert [(candidate.qualname, candidate.reason) for candidate in selected.candidates] == [
        ("alpha", "selected"),
        ("beta", "selected"),
        ("gamma", "selected"),
        ("delta", "coverage-reached"),
        ("epsilon", "below-threshold"),
        ("zeta", "below-threshold"),
    ]


def test_candidate_selection_includes_attributed_scheduler_activity() -> None:
    """Nested scheduler samples can identify a hot owner with a cold leaf frame."""
    profile = _profile_with_members(
        total_samples=200,
        members=(
            ("pkg.hot", "owner", 10),
            ("pkg.hot", "leaf", 50),
            ("pkg.hot", "helper", 40),
        ),
    )
    members = tuple(
        replace(
            member,
            scheduler_overhead_samples=(
                ATTRIBUTED_OWNER_SCHEDULER_SAMPLES if member.qualname == "owner" else 0
            ),
            scheduler_overhead_coverage=(0.35 if member.qualname == "owner" else 0.0),
        )
        for member in profile.members
    )
    profile = replace(
        profile,
        mapped_project_samples=100,
        mapped_coverage=0.5,
        scheduler_overhead_samples=ATTRIBUTED_OWNER_SCHEDULER_SAMPLES,
        scheduler_overhead_coverage=0.35,
        members=members,
    )

    selected = select_profile_candidates(
        profile,
        tuple(_symbol(module, qualname) for module, qualname, _samples in _member_specs(profile)),
    )

    assert [symbol.qualname for symbol in selected.selected_symbols] == [
        "owner",
        "leaf",
        "helper",
    ]
    assert selected.selected_hot_samples == SELECTED_ACTIVITY_SAMPLES
    assert selected.selected_hot_coverage == 1.0
    owner = _candidate(selected, "owner")
    assert owner.samples == ATTRIBUTED_OWNER_LEAF_SAMPLES
    assert owner.scheduler_overhead_samples == ATTRIBUTED_OWNER_SCHEDULER_SAMPLES
    assert owner.attributed_samples == ATTRIBUTED_OWNER_SAMPLES
    assert owner.attributed_coverage == ATTRIBUTED_OWNER_COVERAGE


def test_targeted_observation_ranks_leaf_and_scheduler_samples_together() -> None:
    """The type pass observes active owners instead of only Python leaf frames."""
    hot_member_keys = cast(
        Callable[[dict[str, object]], tuple[tuple[str, int], ...]],
        _callable_attribute(profiling, "_hot_member_keys"),
    )

    ranked = hot_member_keys(
        {
            "sample_counts": {"pkg::leaf": 60, "pkg::owner": 10},
            "scheduler_overhead_counts": {"pkg::owner": 70},
        }
    )

    assert ranked[:2] == (("pkg::owner", 80), ("pkg::leaf", 60))


def test_select_profile_candidates_reports_unmapped_without_selecting_low_total_samples() -> None:
    profile = _profile_with_members(
        total_samples=90,
        members=(("pkg.hot", "alpha", 80), ("pkg.hot", "missing", 10)),
    )

    selected = select_profile_candidates(profile, (_symbol("pkg.hot", "alpha"),))

    assert selected.status == "static-fallback"
    assert selected.reason == "insufficient baseline profile samples: observed 90, required 100"
    assert [(candidate.qualname, candidate.reason) for candidate in selected.candidates] == [
        ("alpha", "below-threshold"),
        ("missing", "unmapped"),
    ]
    assert selected.selected_symbols == ()


def test_select_profile_candidates_reports_limit_after_four_selected() -> None:
    profile = _profile_with_members(
        total_samples=300,
        members=(
            ("pkg.hot", "alpha", 40),
            ("pkg.hot", "beta", 40),
            ("pkg.hot", "gamma", 40),
            ("pkg.hot", "delta", 40),
            ("pkg.hot", "epsilon", 40),
            ("pkg.hot", "zeta", 100),
        ),
    )

    selected = select_profile_candidates(
        profile,
        tuple(_symbol(module, qualname) for module, qualname, _samples in _member_specs(profile)),
    )

    assert [symbol.qualname for symbol in selected.selected_symbols] == [
        "zeta",
        "alpha",
        "beta",
        "delta",
    ]
    assert _candidate(selected, "epsilon").reason == "limit"


def test_scratch_files_are_removed_when_bootstrap_subprocess_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_workload_project(tmp_path)

    def failing_run(_invocation: SubprocessInvocationView) -> subprocess.CompletedProcess[str]:
        raise RuntimeError("subprocess unavailable")

    monkeypatch.setattr(profiling, "_run_subprocess", failing_run)

    with suppress(RuntimeError):
        run_baseline_profile(
            (sys.executable, "bench_script.py", "9"),
            project_root=tmp_path,
            payload_root=tmp_path,
            module_paths=(("workload", "workload.py"),),
            scratch_dir=tmp_path / "scratch",
        )

    assert not list((tmp_path / "scratch").glob("*.json"))


def test_bootstrap_launch_uses_absolute_file_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_workload_project(tmp_path)
    captured: list[tuple[str, ...]] = []

    def fake_run(invocation: SubprocessInvocationView) -> subprocess.CompletedProcess[str]:
        captured.append(invocation.command)
        assert invocation.env["PYTHONDONTWRITEBYTECODE"] == "1"
        config_path = Path(invocation.command[-1])
        payload = cast(dict[str, object], json.loads(config_path.read_text(encoding="utf-8")))
        result_path = payload["result_path"]
        assert isinstance(result_path, str)
        Path(str(result_path)).write_text(
            json.dumps(
                {
                    "total_samples": 0,
                    "sample_counts": {},
                    "lifecycle": {
                        "start": 0,
                        "return": 0,
                        "yield": 0,
                        "resume": 0,
                        "unwind": 0,
                        "throw": 0,
                    },
                    "member_lifecycle": {},
                    "signatures": {},
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(invocation.command, 0, "", "")

    monkeypatch.setattr(profiling, "_run_subprocess", fake_run)

    run_baseline_profile(
        (sys.executable, "bench_script.py", "9"),
        project_root=tmp_path,
        payload_root=tmp_path,
        module_paths=(("workload", "workload.py"),),
        scratch_dir=tmp_path / "scratch",
    )

    assert captured
    assert captured[0][1].endswith("src/atoll/runtime/_profile_bootstrap.py")
    assert Path(captured[0][1]).is_absolute()


def test_baseline_profile_disables_atoll_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_workload_project(tmp_path)
    monkeypatch.setenv("ATOLL_DISABLE", "0")
    monkeypatch.setenv("ATOLL_STRICT", "1")
    monkeypatch.setenv("ATOLL_REQUIRE_COMPILED", "1")
    monkeypatch.setenv("ATOLL_REQUIRE_OPTIMIZED", "1")
    requests: list[dict[str, object]] = []
    environments: list[dict[str, str]] = []

    def fake_run(invocation: SubprocessInvocationView) -> subprocess.CompletedProcess[str]:
        environments.append(dict(invocation.env))
        config = cast(
            dict[str, object],
            json.loads(Path(invocation.command[-1]).read_text(encoding="utf-8")),
        )
        requests.append(config)
        _write_empty_profile_payload(Path(cast(str, config["result_path"])))
        return subprocess.CompletedProcess(invocation.command, 0, "", "")

    monkeypatch.setattr(profiling, "_run_subprocess", fake_run)

    run_baseline_profile(
        (sys.executable, "bench_script.py", "9"),
        project_root=tmp_path,
        payload_root=tmp_path,
        module_paths=(("workload", "workload.py"),),
        scratch_dir=tmp_path / "scratch",
    )

    assert [request["enable_atoll"] for request in requests] == [False, False]
    assert [environment["ATOLL_DISABLE"] for environment in environments] == ["1", "1"]
    assert all("ATOLL_STRICT" not in environment for environment in environments)
    assert all("ATOLL_REQUIRE_COMPILED" not in environment for environment in environments)
    assert all("ATOLL_REQUIRE_OPTIMIZED" not in environment for environment in environments)


def test_baseline_profile_can_enable_atoll_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_workload_project(tmp_path)
    monkeypatch.setenv("ATOLL_DISABLE", "1")
    monkeypatch.setenv("ATOLL_STRICT", "1")
    monkeypatch.setenv("ATOLL_REQUIRE_COMPILED", "1")
    monkeypatch.setenv("ATOLL_REQUIRE_OPTIMIZED", "1")
    requests: list[dict[str, object]] = []
    environments: list[dict[str, str]] = []

    def fake_run(invocation: SubprocessInvocationView) -> subprocess.CompletedProcess[str]:
        environments.append(dict(invocation.env))
        config = cast(
            dict[str, object],
            json.loads(Path(invocation.command[-1]).read_text(encoding="utf-8")),
        )
        requests.append(config)
        _write_empty_profile_payload(Path(cast(str, config["result_path"])))
        return subprocess.CompletedProcess(invocation.command, 0, "", "")

    monkeypatch.setattr(profiling, "_run_subprocess", fake_run)

    run_baseline_profile(
        (sys.executable, "bench_script.py", "9"),
        project_root=tmp_path,
        payload_root=tmp_path,
        module_paths=(("workload", "workload.py"),),
        scratch_dir=tmp_path / "scratch",
        enable_atoll=True,
    )

    assert [request["enable_atoll"] for request in requests] == [True, True]
    assert all("ATOLL_DISABLE" not in environment for environment in environments)
    assert all("ATOLL_STRICT" not in environment for environment in environments)
    assert all("ATOLL_REQUIRE_COMPILED" not in environment for environment in environments)
    assert all("ATOLL_REQUIRE_OPTIMIZED" not in environment for environment in environments)


def test_signatures_observe_parameters_without_ordinary_locals(tmp_path: Path) -> None:
    _write_signature_project(tmp_path)

    result = run_baseline_profile(
        (sys.executable, "bench_signature.py"),
        project_root=tmp_path,
        payload_root=tmp_path,
        module_paths=(("signature_case", "signature_case.py"),),
        scratch_dir=tmp_path / "scratch",
    )

    member = _member(result, "signature_case", "target")
    assert member.signatures
    assert [item.parameter_name for item in member.signatures[0].parameters] == [
        "first",
        "second",
        "flag",
        "items",
        "options",
    ]


def test_profile_bootstrap_sampling_entrypoint_runs_in_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_workload_project(tmp_path)
    config_path, result_path = _write_bootstrap_config(
        tmp_path,
        _BootstrapConfigInput(
            profile_stage="sampling",
            launch_kind="script",
            target="bench_script.py",
            args=("9",),
            module_paths=(("workload", "workload.py"),),
        ),
    )
    _prepare_in_process_bootstrap(monkeypatch, tmp_path, "workload")

    exit_code = _profile_bootstrap().main((str(config_path),))

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["total_samples"] > 0
    assert payload["sample_counts"]["workload::hot"] > 0


def test_profile_bootstrap_script_can_import_sibling_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profiled scripts receive the same import root as direct Python launch."""
    script_root = tmp_path / "benchmark"
    script_root.mkdir()
    (script_root / "sibling.py").write_text("VALUE = 3\n", encoding="utf-8")
    (script_root / "runner.py").write_text(
        "from sibling import VALUE\n"
        "if VALUE != 3:\n"
        "    raise RuntimeError('wrong sibling module')\n",
        encoding="utf-8",
    )
    config_path, result_path = _write_bootstrap_config(
        tmp_path,
        _BootstrapConfigInput(
            profile_stage="sampling",
            launch_kind="script",
            target="benchmark/runner.py",
        ),
    )
    _prepare_in_process_bootstrap(monkeypatch, tmp_path, "sibling")

    exit_code = _profile_bootstrap().main((str(config_path),))

    assert exit_code == 0
    assert result_path.is_file()
    assert str(script_root.resolve()) not in sys.path


def test_profile_bootstrap_enables_optimized_payload_in_child_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bootstrap honors optimized profiling after receiving the parent request."""
    _write_workload_project(tmp_path)
    config_path, result_path = _write_bootstrap_config(
        tmp_path,
        _BootstrapConfigInput(
            profile_stage="sampling",
            launch_kind="script",
            target="bench_script.py",
            args=("9",),
            module_paths=(("workload", "workload.py"),),
            enable_atoll=True,
        ),
    )
    monkeypatch.setenv("ATOLL_DISABLE", "1")
    monkeypatch.setenv("ATOLL_STRICT", "1")
    monkeypatch.setenv("ATOLL_REQUIRE_COMPILED", "1")
    monkeypatch.setenv("ATOLL_REQUIRE_OPTIMIZED", "1")
    _prepare_in_process_bootstrap(monkeypatch, tmp_path, "workload")

    exit_code = _profile_bootstrap().main((str(config_path),))

    assert exit_code == 0
    assert result_path.exists()
    assert "ATOLL_DISABLE" not in os.environ
    assert "ATOLL_STRICT" not in os.environ
    assert "ATOLL_REQUIRE_COMPILED" not in os.environ
    assert "ATOLL_REQUIRE_OPTIMIZED" not in os.environ


def test_profile_bootstrap_type_entrypoint_bounds_hot_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_frequent_call_project(tmp_path)
    config_path, result_path = _write_bootstrap_config(
        tmp_path,
        _BootstrapConfigInput(
            profile_stage="types",
            launch_kind="script",
            target="bench_frequent.py",
            module_paths=(("frequent", "frequent.py"),),
            targets=("frequent::hot",),
        ),
    )
    _prepare_in_process_bootstrap(monkeypatch, tmp_path, "frequent")

    exit_code = _profile_bootstrap().main((str(config_path),))

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    hot = payload["signatures"]["frequent::hot"]
    assert exit_code == 0
    assert hot["call_count"] == MAX_TYPE_OBSERVATIONS_PER_MEMBER
    assert hot["invocation_count"] == FREQUENT_CALLS
    assert hot["completed_calls"] == FREQUENT_CALLS
    assert hot["max_active_calls"] == 1
    assert hot["pre_completion_suspensions"] == 0
    assert hot["observation_capped"] is True
    assert payload["member_lifecycle"]["frequent::hot"]["start"] == FREQUENT_CALLS
    assert payload["member_lifecycle"]["frequent::hot"]["return"] == FREQUENT_CALLS


def test_profile_bootstrap_module_entrypoint_returns_system_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "bootstrap_exit_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "runner.py").write_text("raise SystemExit(7)\n", encoding="utf-8")
    config_path, result_path = _write_bootstrap_config(
        tmp_path,
        _BootstrapConfigInput(
            profile_stage="sampling",
            launch_kind="module",
            target="bootstrap_exit_pkg.runner",
        ),
    )
    _prepare_in_process_bootstrap(monkeypatch, tmp_path, "bootstrap_exit_pkg")

    exit_code = _profile_bootstrap().main((str(config_path),))

    assert exit_code == BOOTSTRAP_EXIT_CODE
    assert result_path.exists()


@pytest.mark.parametrize(
    ("statement", "expected_exit_code", "add_root"),
    [
        ("raise SystemExit", 0, True),
        ("raise SystemExit('message')", BOOTSTRAP_STRING_EXIT_CODE, False),
    ],
)
def test_profile_bootstrap_normalizes_non_integer_system_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    statement: str,
    expected_exit_code: int,
    add_root: bool,
) -> None:
    script_path = tmp_path / f"bench_exit_{expected_exit_code}.py"
    script_path.write_text(f"{statement}\n", encoding="utf-8")
    config_path, result_path = _write_bootstrap_config(
        tmp_path,
        _BootstrapConfigInput(
            profile_stage="sampling",
            launch_kind="script",
            target=str(script_path),
        ),
    )
    _prepare_in_process_bootstrap(monkeypatch, tmp_path, add_root=add_root)

    exit_code = _profile_bootstrap().main((str(config_path),))

    assert exit_code == expected_exit_code
    assert result_path.exists()


def test_profile_bootstrap_persists_partial_evidence_before_target_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "bench_error.py").write_text(
        "raise RuntimeError('expected profile target failure')\n",
        encoding="utf-8",
    )
    config_path, result_path = _write_bootstrap_config(
        tmp_path,
        _BootstrapConfigInput(
            profile_stage="sampling",
            launch_kind="script",
            target="bench_error.py",
        ),
    )
    _prepare_in_process_bootstrap(monkeypatch, tmp_path)

    with pytest.raises(RuntimeError, match="expected profile target failure"):
        _profile_bootstrap().main((str(config_path),))

    assert result_path.exists()
    assert json.loads(result_path.read_text(encoding="utf-8"))["total_samples"] >= 0


def test_profile_bootstrap_rejects_missing_config_argument() -> None:
    assert _profile_bootstrap().main(()) == BOOTSTRAP_USAGE_ERROR_CODE


def test_profile_bootstrap_config_rejects_non_structured_sequences(tmp_path: Path) -> None:
    module = importlib.import_module("atoll.runtime._profile_bootstrap")
    read_config = _callable_attribute(module, "_read_config")
    list_payload = tmp_path / "list.json"
    list_payload.write_text("[]", encoding="utf-8")
    malformed_sequences = tmp_path / "malformed-sequences.json"
    malformed_sequences.write_text(
        json.dumps(
            {
                "profile_stage": "sampling",
                "launch_kind": "script",
                "target": "bench.py",
                "args": "not-a-list",
                "project_root": str(tmp_path),
                "payload_root": str(tmp_path),
                "module_paths": {},
                "result_path": str(tmp_path / "result.json"),
                "targets": 3,
            }
        ),
        encoding="utf-8",
    )

    assert read_config(list_payload) is not None
    parsed = read_config(malformed_sequences)
    assert parsed is not None
    assert _attribute(parsed, "call_edge_targets") == ()


def test_type_profiler_callbacks_are_bounded_and_ignore_non_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "callback_case.py").write_text(
        "OBSERVER = None\n"
        "def hot(value: object, *items: object, flag: bool = False, **options: object) -> object:\n"
        "    return OBSERVER(hot.__code__, 0)\n"
        "def nested_code() -> object:\n"
        "    def inner() -> None:\n"
        "        return None\n"
        "    return inner.__code__\n",
        encoding="utf-8",
    )
    config_path, _result_path = _write_bootstrap_config(
        tmp_path,
        _BootstrapConfigInput(
            profile_stage="types",
            launch_kind="script",
            target="unused.py",
            module_paths=(("callback_case", "callback_case.py"),),
            targets=("callback_case::hot",),
        ),
    )
    _prepare_in_process_bootstrap(
        monkeypatch,
        tmp_path,
        "callback_case",
        add_root=True,
    )
    callback_case = cast(CallbackCaseModule, importlib.import_module("callback_case"))
    profiler = _type_profiler(config_path)
    callback_case.OBSERVER = profiler.observer

    sys.monitoring.use_tool_id(5, "direct-callback-test")
    try:
        profiler.observer(callback_case.hot.__code__, 0)
        assert profiler.observer(callback_case.nested_code(), 0) is sys.monitoring.DISABLE
        distinct_values: tuple[object, ...] = (
            0,
            "text",
            1.5,
            (),
            list[int](),
            dict[str, int](),
            set[int](),
            frozenset[int](),
            object(),
        )
        for value in distinct_values:
            callback_case.hot(value, 1, flag=True, extra=None)
        for value in range(MAX_TYPE_OBSERVATIONS_PER_MEMBER - len(distinct_values)):
            callback_case.hot(value)
        profiler.lifecycle_callback("throw")(callback_case.hot.__code__, 0)
        profiler.lifecycle_callback("throw")((lambda: None).__code__, 0)
    finally:
        sys.monitoring.set_local_events(
            5,
            callback_case.hot.__code__,
            sys.monitoring.events.NO_EVENTS,
        )
        sys.monitoring.free_tool_id(5)
    profiler.stop()

    payload = profiler.payload()
    signatures = cast(dict[str, dict[str, object]], payload["signatures"])
    hot_payload = signatures["callback_case::hot"]
    assert hot_payload["call_count"] == MAX_TYPE_OBSERVATIONS_PER_MEMBER
    assert hot_payload["completed_calls"] == 0
    assert cast(int, hot_payload["max_active_calls"]) >= MAX_TYPE_OBSERVATIONS_PER_MEMBER
    assert hot_payload["observation_capped"] is True
    assert hot_payload["polymorphic_overflow"] is True


def test_type_profiler_matches_full_spawn_spans_and_helper_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct callback tests cover exact spans, cache reuse, and conservative fallbacks."""
    source = (
        "class TaskGroup:\n"
        "    def create_task(self, value):\n"
        "        return value\n\n"
        "def schedule(group):\n"
        "    return group.create_task(1)\n"
    )
    source_path = tmp_path / "spawn_callback.py"
    source_path.write_text(source, encoding="utf-8")
    tree = ast.parse(source)
    scheduler_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "create_task"
    )
    target = ProfileSpawnSiteTarget(
        id="spawn-callback",
        owner=SymbolId("spawn_callback", "schedule"),
        lineno=scheduler_call.lineno,
        col_offset=scheduler_call.col_offset,
        scheduler_method="create_task",
        end_lineno=scheduler_call.end_lineno,
        end_col_offset=scheduler_call.end_col_offset,
    )
    config_path, _result_path = _write_bootstrap_config(
        tmp_path,
        _BootstrapConfigInput(
            profile_stage="types",
            launch_kind="script",
            target="unused.py",
            module_paths=(("spawn_callback", "spawn_callback.py"),),
            targets=("spawn_callback::schedule",),
            spawn_targets=(target,),
        ),
    )
    _prepare_in_process_bootstrap(monkeypatch, tmp_path, "spawn_callback", add_root=True)
    callback_module = importlib.import_module("spawn_callback")
    schedule = cast(FunctionType, _callable_attribute(callback_module, "schedule"))
    task_group_type = cast(type[object], _attribute(callback_module, "TaskGroup"))
    create_task = _callable_attribute(task_group_type(), "create_task")
    call_instruction = next(
        instruction
        for instruction in get_instructions(schedule)
        if instruction.opname == "CALL"
        and instruction.positions is not None
        and instruction.positions.col_offset == scheduler_call.col_offset
        and instruction.positions.end_col_offset == scheduler_call.end_col_offset
    )
    profiler = _type_profiler(config_path)

    profiler.call_callback(schedule.__code__, call_instruction.offset, create_task, None)
    profiler.call_callback(schedule.__code__, call_instruction.offset, create_task, None)
    profiler.call_callback(schedule.__code__, call_instruction.offset, lambda: None, None)
    profiler.call_callback((lambda: None).__code__, 0, create_task, None)

    payload = profiler.payload()
    spawn_sites = cast(list[dict[str, object]], payload["spawn_sites"])
    assert spawn_sites[0]["invocation_count"] == DIRECT_SPAWN_CALLBACKS
    assert spawn_sites[0]["callable_identities"] == {
        "spawn_callback.TaskGroup.create_task": DIRECT_SPAWN_CALLBACKS
    }
    _assert_profile_bootstrap_position_helpers(profiler, target)


def test_type_profiler_matches_call_edge_position_and_callable_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only exact source positions resolving to the configured callee increment an edge."""
    source = (
        "def leaf(value):\n"
        "    return value + 1\n\n"
        "def other(value):\n"
        "    return value - 1\n\n"
        "def root(value):\n"
        "    return leaf(value) + leaf(value + 1)\n"
    )
    (tmp_path / "call_edge_callback.py").write_text(source, encoding="utf-8")
    calls = sorted(
        (
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "leaf"
        ),
        key=lambda node: node.col_offset,
    )
    targets = tuple(
        ProfileCallEdgeTarget(
            id=f"edge-{index}",
            owner=SymbolId("call_edge_callback", "root"),
            callee=SymbolId("call_edge_callback", "leaf"),
            lineno=call.lineno,
            col_offset=call.col_offset,
            end_lineno=call.end_lineno,
            end_col_offset=call.end_col_offset,
        )
        for index, call in enumerate(calls)
    )
    config_path, _result_path = _write_bootstrap_config(
        tmp_path,
        _BootstrapConfigInput(
            profile_stage="types",
            launch_kind="script",
            target="unused.py",
            module_paths=(("call_edge_callback", "call_edge_callback.py"),),
            targets=("call_edge_callback::root",),
            call_edge_targets=targets,
        ),
    )
    _prepare_in_process_bootstrap(monkeypatch, tmp_path, "call_edge_callback", add_root=True)
    callback_module = importlib.import_module("call_edge_callback")
    root = cast(FunctionType, _callable_attribute(callback_module, "root"))
    leaf = _callable_attribute(callback_module, "leaf")
    other = _callable_attribute(callback_module, "other")
    instructions = {
        (
            instruction.positions.col_offset,
            instruction.positions.end_col_offset,
        ): instruction
        for instruction in get_instructions(root)
        if instruction.opname == "CALL" and instruction.positions is not None
    }
    profiler = _type_profiler(config_path)

    first = instructions[(calls[0].col_offset, calls[0].end_col_offset)]
    second = instructions[(calls[1].col_offset, calls[1].end_col_offset)]
    profiler.call_callback(root.__code__, first.offset, leaf, object())
    profiler.call_callback(root.__code__, first.offset, other, object())
    profiler.call_callback(root.__code__, second.offset, leaf, object())
    profiler.call_callback((lambda: None).__code__, first.offset, leaf, object())

    payload = profiler.payload()
    call_edges = cast(list[dict[str, object]], payload["call_edges"])
    assert [edge["invocation_count"] for edge in call_edges] == [1, 1]
    assert all("callable_identities" not in edge for edge in call_edges)


def test_type_profiler_reports_monitoring_tool_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, _result_path = _write_bootstrap_config(
        tmp_path,
        _BootstrapConfigInput(
            profile_stage="types",
            launch_kind="script",
            target="unused.py",
        ),
    )
    _prepare_in_process_bootstrap(monkeypatch, tmp_path)
    profiler = _type_profiler(config_path)
    sys.monitoring.use_tool_id(5, "occupied-by-test")
    try:
        with pytest.raises(RuntimeError, match="unable to reserve"):
            profiler.start()
    finally:
        sys.monitoring.free_tool_id(5)
    profiler.stop()


def _write_workload_project(root: Path) -> None:
    (root / "workload.py").write_text(
        "\n".join(
            [
                "def hot(value: int) -> int:",
                "    total = 0",
                "    for item in range(600000):",
                "        total += (item % 7) * value",
                "    return total",
                "",
                "def cold(value: int) -> int:",
                "    return value + 1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (root / "bench_script.py").write_text(
        "\n".join(
            [
                "import sys",
                "from workload import cold, hot",
                "hot(int(sys.argv[1])); cold(1); print('script-done')",
                "",
            ]
        ),
        encoding="utf-8",
    )
    package = root / "benchpkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "runner.py").write_text(
        "import sys\nfrom workload import hot\nhot(int(sys.argv[1])); print('module-done')\n",
        encoding="utf-8",
    )


def _write_privacy_project(root: Path) -> None:
    (root / "privacy.py").write_text(
        "\n".join(
            [
                "class SecretPayload:",
                "    def __repr__(self) -> str:",
                "        return 'do-not-leak-repr'",
                "",
                "def accept(payload: SecretPayload) -> int:",
                "    total = 0",
                "    for item in range(500000):",
                "        total += item % 5",
                "    return total",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (root / "bench_privacy.py").write_text(
        "from privacy import SecretPayload, accept\naccept(SecretPayload())\n",
        encoding="utf-8",
    )


def _write_polymorphic_project(root: Path) -> None:
    class_names = tuple(f"Payload{index}" for index in range(9))
    (root / "poly.py").write_text(
        "\n".join(
            [
                *(f"class {name}: pass" for name in class_names),
                "",
                "def accept(payload: object) -> int:",
                "    total = 0",
                "    for item in range(80000):",
                "        total += item % 3",
                "    return total",
                "",
            ]
        ),
        encoding="utf-8",
    )
    calls = "\n".join(f"accept({name}())" for name in class_names)
    (root / "bench_poly.py").write_text(
        f"from poly import accept, {', '.join(class_names)}\n{calls}\n",
        encoding="utf-8",
    )


def _write_frequent_call_project(root: Path) -> None:
    (root / "frequent.py").write_text(
        "def hot(value: int) -> int:\n    return (value * 3) % 11\n",
        encoding="utf-8",
    )
    (root / "bench_frequent.py").write_text(
        f"from frequent import hot\nfor item in range({FREQUENT_CALLS}): hot(item)\n",
        encoding="utf-8",
    )


def _write_signature_project(root: Path) -> None:
    (root / "signature_case.py").write_text(
        "\n".join(
            [
                "def target(",
                "    first: int, /, second: str, *items: float, flag: bool, **options: object",
                ") -> int:",
                "    ordinary_local = object()",
                "    total = first + len(second) + len(items) + int(flag) + len(options)",
                "    for item in range(500000):",
                "        total += item % 3",
                "    return total + (ordinary_local is None)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (root / "bench_signature.py").write_text(
        "\n".join(
            [
                "from signature_case import target",
                "target(1, 'two', 3.0, flag=True, extra=object())",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_async_observation_project(root: Path) -> None:
    (root / "async_case.py").write_text(
        "\n".join(
            [
                "import asyncio",
                "",
                "async def worker(value: int) -> int:",
                "    await asyncio.sleep(0)",
                "    await asyncio.sleep(0)",
                "    return value",
                "",
                "def zero_sample(value: int) -> int:",
                "    return value + 1",
                "",
                "def dormant(value: int) -> int:",
                "    return value - 1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (root / "bench_async.py").write_text(
        "\n".join(
            [
                "import asyncio",
                "from async_case import worker, zero_sample",
                "",
                "async def main() -> None:",
                "    tasks = [asyncio.create_task(worker(item)) for item in range(5)]",
                "    await asyncio.gather(*tasks)",
                "    zero_sample(10)",
                "    print('async-done')",
                "",
                "asyncio.run(main())",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_capped_async_observation_project(root: Path) -> None:
    (root / "async_capped.py").write_text(
        "import asyncio\n\n"
        "async def worker(value: int) -> int:\n"
        "    await asyncio.sleep(0)\n"
        "    return value\n",
        encoding="utf-8",
    )
    (root / "bench_async_capped.py").write_text(
        "import asyncio\n"
        "from async_capped import worker\n\n"
        "async def main() -> None:\n"
        "    tasks = [asyncio.create_task(worker(item)) "
        f"for item in range({ASYNC_CAPPED_WORKER_CALLS})]\n"
        "    await asyncio.gather(*tasks)\n\n"
        "asyncio.run(main())\n",
        encoding="utf-8",
    )


def _write_spawn_site_project(root: Path) -> None:
    (root / "scheduler_case.py").write_text(
        "import asyncio\n\n"
        "async def worker(value: int) -> int:\n"
        "    await asyncio.sleep(0)\n"
        "    return value\n\n"
        "async def run() -> None:\n"
        "    async with asyncio.TaskGroup() as group:\n"
        f"        for item in range({SPAWN_SITE_CALLS}):\n"
        "            group.create_task(worker(item))\n",
        encoding="utf-8",
    )
    (root / "bench_scheduler.py").write_text(
        "import asyncio\nfrom scheduler_case import run\nasyncio.run(run())\n",
        encoding="utf-8",
    )


def _write_call_edge_project(root: Path) -> str:
    source = (
        "def leaf(value: int) -> int:\n"
        "    return value + 1\n\n"
        "def root() -> int:\n"
        "    total = 0\n"
        f"    for item in range({DIRECT_CALL_EDGE_CALLS}):\n"
        "        total += leaf(item)\n"
        "    return total\n"
    )
    (root / "call_edge_case.py").write_text(source, encoding="utf-8")
    (root / "bench_call_edge.py").write_text(
        "from call_edge_case import root\nroot()\n",
        encoding="utf-8",
    )
    return source


def _profile_bootstrap() -> ProfileBootstrapModule:
    return cast(
        ProfileBootstrapModule,
        importlib.import_module("atoll.runtime._profile_bootstrap"),
    )


def _type_profiler(config_path: Path) -> _TypeProfilerHarness:
    module = importlib.import_module("atoll.runtime._profile_bootstrap")
    read_config = cast(
        Callable[[Path], object],
        _callable_attribute(module, "_read_config"),
    )
    factory = cast(
        Callable[[object], object],
        _callable_attribute(module, "_TypeProfiler"),
    )
    return _TypeProfilerHarness(factory(read_config(config_path)))


def _sampling_profiler(config_path: Path) -> _SamplingProfilerHarness:
    module = importlib.import_module("atoll.runtime._profile_bootstrap")
    read_config = cast(
        Callable[[Path], object],
        _callable_attribute(module, "_read_config"),
    )
    factory = cast(
        Callable[[object], object],
        _callable_attribute(module, "_SamplingProfiler"),
    )
    return _SamplingProfilerHarness(factory(read_config(config_path)))


def _callable_attribute(value: object, name: str) -> Callable[..., object]:
    return cast(Callable[..., object], _attribute(value, name))


def _attribute(value: object, name: str) -> object:
    return getattr(value, name)


def _assert_profile_bootstrap_position_helpers(
    profiler: _TypeProfilerHarness,
    target: ProfileSpawnSiteTarget,
) -> None:
    module = importlib.import_module("atoll.runtime._profile_bootstrap")
    position_matches = _callable_attribute(module, "_position_within_spawn_target")
    source_position = _callable_attribute(module, "_instruction_source_position")
    canonical_callable = _callable_attribute(module, "_canonical_callable_identity")
    target_factory = _callable_attribute(module, "_SpawnTarget")
    internal_targets = cast(tuple[object, ...], _attribute(profiler.wrapped, "_spawn_targets"))
    internal_target = internal_targets[0]
    fallback_target = target_factory(
        id=target.id,
        owner=target.owner.stable_id,
        lineno=target.lineno,
        col_offset=target.col_offset,
        scheduler_method=target.scheduler_method,
        end_lineno=None,
        end_col_offset=None,
    )

    assert source_position(None) == (None, None, None, None)
    assert position_matches((None, None, None, None), internal_target) is False
    assert (
        position_matches((target.lineno, target.col_offset, None, None), internal_target) is False
    )
    assert (
        position_matches(
            (target.lineno, target.col_offset, target.end_lineno, target.end_col_offset),
            internal_target,
        )
        is True
    )
    assert (
        position_matches(
            (target.lineno, target.col_offset - 1, target.end_lineno, target.end_col_offset),
            internal_target,
        )
        is False
    )
    assert position_matches((target.lineno, target.col_offset, None, None), fallback_target) is True
    assert cast(str, canonical_callable(object())).startswith("builtins.object")


def _write_bootstrap_config(
    root: Path,
    config: _BootstrapConfigInput,
) -> tuple[Path, Path]:
    config_path = root / f"{config.profile_stage}.config.json"
    result_path = root / f"{config.profile_stage}.result.json"
    config_path.write_text(
        json.dumps(
            {
                "profile_stage": config.profile_stage,
                "launch_kind": config.launch_kind,
                "target": config.target,
                "args": list(config.args),
                "project_root": str(root),
                "payload_root": str(root),
                "module_paths": [list(item) for item in config.module_paths],
                "result_path": str(result_path),
                "enable_atoll": config.enable_atoll,
                "targets": list(config.targets),
                "spawn_targets": [
                    {
                        "id": target.id,
                        "owner": target.owner.stable_id,
                        "lineno": target.lineno,
                        "col_offset": target.col_offset,
                        "scheduler_method": target.scheduler_method,
                        "end_lineno": target.end_lineno,
                        "end_col_offset": target.end_col_offset,
                    }
                    for target in config.spawn_targets
                ],
                "call_edge_targets": [
                    {
                        "id": target.id,
                        "owner": target.owner.stable_id,
                        "callee": target.callee.stable_id,
                        "lineno": target.lineno,
                        "col_offset": target.col_offset,
                        "end_lineno": target.end_lineno,
                        "end_col_offset": target.end_col_offset,
                    }
                    for target in config.call_edge_targets
                ],
            }
        ),
        encoding="utf-8",
    )
    return config_path, result_path


def _write_empty_profile_payload(result_path: Path) -> None:
    result_path.write_text(
        json.dumps(
            {
                "total_samples": 0,
                "sample_counts": {},
                "scheduler_overhead_counts": {},
                "lifecycle": {
                    "start": 0,
                    "return": 0,
                    "yield": 0,
                    "resume": 0,
                    "unwind": 0,
                    "throw": 0,
                },
                "member_lifecycle": {},
                "signatures": {},
                "spawn_sites": [],
                "call_edges": [],
            }
        ),
        encoding="utf-8",
    )


def _prepare_in_process_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *module_names: str,
    add_root: bool = False,
) -> None:
    monkeypatch.chdir(root)
    monkeypatch.setattr(sys, "path", list(sys.path))
    if add_root:
        sys.path.insert(0, str(root))
    monkeypatch.setenv("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    monkeypatch.setenv("ATOLL_DISABLE", os.environ.get("ATOLL_DISABLE", ""))
    monkeypatch.setenv(
        "ATOLL_REQUIRE_COMPILED",
        os.environ.get("ATOLL_REQUIRE_COMPILED", ""),
    )
    for module_name in module_names:
        monkeypatch.delitem(sys.modules, module_name, raising=False)


def _profile_with_members(
    *,
    total_samples: int,
    members: tuple[tuple[str, str, int], ...],
) -> ProfileResult:
    base = unconfigured_profile()
    profiled_members = tuple(
        ProfiledMember(
            module=module,
            qualname=qualname,
            samples=samples,
            coverage=samples / total_samples,
            call_count=samples,
            lifecycle=LifecycleCounts(
                start=samples,
                return_=samples,
                yield_=0,
                resume=0,
                unwind=0,
                throw=0,
            ),
            signatures=(
                ObservedSignature(
                    parameters=(
                        CanonicalTypeObservation(
                            parameter_name="value",
                            type_path="builtins.int",
                            count=samples,
                        ),
                    ),
                    count=samples,
                ),
            ),
            polymorphic_overflow=False,
        )
        for module, qualname, samples in members
    )
    return replace(
        base,
        status="profiled",
        reason="synthetic profile",
        launch_kind="script",
        total_samples=total_samples,
        mapped_project_samples=sum(samples for _module, _qualname, samples in members),
        mapped_coverage=1.0,
        lifecycle=LifecycleCounts(start=1, return_=1, yield_=0, resume=0, unwind=0, throw=0),
        members=profiled_members,
    )


def _symbol(module: str, qualname: str) -> SymbolRecord:
    return SymbolRecord(
        id=SymbolId(module, qualname),
        kind="function",
        visibility="public",
        lineno=1,
        end_lineno=2,
        col_offset=0,
        end_col_offset=0,
        decorators=(),
        arg_count=1,
        annotated_arg_count=1,
        has_return_annotation=True,
        has_any_annotation=False,
        called_names=(),
        uses_globals=(),
        local_names=(),
        referenced_names=(),
        blockers=(),
    )


def _member(result: ProfileResult, module: str, qualname: str) -> ProfiledMember:
    return next(
        member
        for member in result.members
        if member.module == module and member.qualname == qualname
    )


def _candidate(result: ProfileResult, qualname: str) -> MappedCandidateDecision:
    return next(candidate for candidate in result.candidates if candidate.qualname == qualname)


def _member_specs(result: ProfileResult) -> tuple[tuple[str, str, int], ...]:
    return tuple((member.module, member.qualname, member.samples) for member in result.members)
