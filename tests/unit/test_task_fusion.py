"""Tests for conservative task-fusion report planning."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from textwrap import dedent

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.task_fusion import build_fusion_plans, fusion_observation_targets
from atoll.models import ModuleId, ModuleScan, SymbolId
from atoll.runtime.profiling import (
    CanonicalTypeObservation,
    LifecycleCounts,
    ObservedSignature,
    ProfiledMember,
    ProfileResult,
)

_ROOT_SPAWN_LINE = 6
_OBSERVED_CALLS = 25


def _scan(tmp_path: Path, source: str) -> ModuleScan:
    path = tmp_path / "sample.py"
    path.write_text(dedent(source), encoding="utf-8")
    return scan_module(ModuleId(name="sample", path=path))


def _profile(
    *,
    selected: tuple[str, ...] = ("root",),
    members: tuple[ProfiledMember, ...],
) -> ProfileResult:
    return ProfileResult(
        status="profiled",
        reason="ok",
        launch_kind="script",
        total_samples=100,
        mapped_project_samples=100,
        mapped_coverage=1.0,
        selected_hot_samples=100,
        selected_hot_coverage=1.0,
        runs=(),
        lifecycle=LifecycleCounts(start=0, return_=0, yield_=0, resume=0, unwind=0, throw=0),
        members=members,
        candidates=(),
        selected_symbols=tuple(SymbolId(module="sample", qualname=name) for name in selected),
    )


def _signature(count: int) -> ObservedSignature:
    return ObservedSignature(
        parameters=(
            CanonicalTypeObservation(
                parameter_name="value",
                type_path="builtins.int",
                count=count,
            ),
        ),
        count=count,
    )


def _member(
    qualname: str = "worker",
    *,
    call_count: int = _OBSERVED_CALLS,
    signatures: tuple[ObservedSignature, ...] | None = None,
    lifecycle: LifecycleCounts | None = None,
) -> ProfiledMember:
    lifecycle = lifecycle or LifecycleCounts(
        start=call_count,
        return_=call_count,
        yield_=0,
        resume=0,
        unwind=0,
        throw=0,
    )
    return ProfiledMember(
        module="sample",
        qualname=qualname,
        samples=25,
        coverage=0.5,
        call_count=call_count,
        lifecycle=lifecycle,
        signatures=signatures if signatures is not None else (_signature(call_count),),
        polymorphic_overflow=False,
        observation_capped=False,
        completed_calls=_OBSERVED_CALLS,
        max_active_calls=1,
        pre_completion_suspensions=0,
    )


def test_builds_eligible_plan_for_single_completed_coroutine_spawn(tmp_path: Path) -> None:
    scan = _scan(
        tmp_path,
        """
        async def worker(value):
            return value + 1

        async def root(nursery, value):
            nursery.start_soon(worker, value)
        """,
    )

    plans = build_fusion_plans((scan,), _profile(members=(_member(),)))

    assert len(plans) == 1
    plan = plans[0]
    assert plan.eligible is True
    assert plan.rejections == ()
    assert plan.root == "sample::root"
    assert plan.caller == "sample::root"
    assert plan.callee == "sample::worker"
    assert plan.spawn_api == "nursery.start_soon"
    assert plan.lineno == _ROOT_SPAWN_LINE
    assert plan.observed_calls == _OBSERVED_CALLS
    assert plan.completed_calls == _OBSERVED_CALLS
    assert plan.max_active_calls == 1
    assert plan.pre_completion_suspensions == 0
    assert plan.observed_signatures == 1
    assert plan.observation_capped is False


def test_rejects_overlap_suspension_and_non_monomorphic_evidence(tmp_path: Path) -> None:
    scan = _scan(
        tmp_path,
        """
        async def worker(value):
            return value + 1

        async def root(nursery, value):
            nursery.start_soon(worker, value)
        """,
    )
    member = replace(
        _member(),
        signatures=(_signature(10), _signature(15)),
        lifecycle=LifecycleCounts(start=25, return_=25, yield_=1, resume=1, unwind=0, throw=0),
        max_active_calls=2,
        pre_completion_suspensions=1,
    )

    plans = build_fusion_plans((scan,), _profile(members=(member,)))

    assert [rejection.code for rejection in plans[0].rejections] == [
        "non_monomorphic_signature",
        "overlapping_calls",
        "pre_completion_suspension",
        "lifecycle_suspension",
    ]


def test_rejects_incomplete_capped_low_count_profile_evidence(tmp_path: Path) -> None:
    scan = _scan(
        tmp_path,
        """
        async def worker(value):
            return value + 1

        async def root(nursery, value):
            nursery.start_soon(worker, value)
        """,
    )
    member = replace(
        _member(call_count=10),
        signatures=(_signature(9),),
        polymorphic_overflow=True,
        observation_capped=True,
        completed_calls=9,
        max_active_calls=2,
    )

    plans = build_fusion_plans((scan,), _profile(members=(member,)))

    assert [rejection.code for rejection in plans[0].rejections] == [
        "insufficient_observed_calls",
        "non_monomorphic_signature",
        "polymorphic_evidence_capped",
        "incomplete_calls",
        "overlapping_calls",
    ]


def test_distinguishes_unresolved_sync_and_unprofiled_spawn_targets(tmp_path: Path) -> None:
    scan = _scan(
        tmp_path,
        """
        def sync_worker(value):
            return value

        async def unprofiled(value):
            return value

        async def root_missing(nursery, value):
            nursery.start_soon(missing, value)

        async def root_sync(nursery, value):
            nursery.start_soon(sync_worker, value)

        async def root_unprofiled(nursery, value):
            nursery.start_soon(unprofiled, value)
        """,
    )

    plans = build_fusion_plans(
        (scan,),
        _profile(
            selected=("root_missing", "root_sync", "root_unprofiled"),
            members=(_member("sync_worker"),),
        ),
    )

    assert fusion_observation_targets((scan,)) == (
        "sample::sync_worker",
        "sample::unprofiled",
    )
    by_callee = {plan.callee: plan for plan in plans}
    assert [item.code for item in by_callee[None].rejections] == ["callee_unresolved"]
    assert [item.code for item in by_callee["sample::sync_worker"].rejections] == [
        "callee_not_coroutine"
    ]
    assert [item.code for item in by_callee["sample::unprofiled"].rejections] == [
        "missing_profile_evidence"
    ]


def test_selected_root_from_another_module_produces_no_plan(tmp_path: Path) -> None:
    scan = _scan(
        tmp_path,
        """
        async def worker(value):
            return value

        async def root(nursery, value):
            nursery.start_soon(worker, value)
        """,
    )
    profile = replace(
        _profile(members=(_member(),)),
        selected_symbols=(SymbolId("other", "root"),),
    )

    assert build_fusion_plans((scan,), profile) == ()


def test_rejects_cancel_scope_start_soon_send_and_instrumentation(tmp_path: Path) -> None:
    scan = _scan(
        tmp_path,
        """
        async def worker(value, nursery, stream, instrument):
            with CancelScope():
                instrument.record(value)
                nursery.start_soon(helper, value)
                stream.send(value)

        async def helper(value):
            return value

        async def root(nursery, value):
            nursery.start_soon(worker, value)
        """,
    )

    plans = build_fusion_plans((scan,), _profile(members=(_member(),)))

    assert [rejection.code for rejection in plans[0].rejections] == [
        "static_cancellation",
        "static_instrumentation",
        "static_extra_concurrency",
        "static_dynamic_effects",
    ]


def test_rejects_caller_hazards_and_escaping_task_handle(tmp_path: Path) -> None:
    scan = _scan(
        tmp_path,
        """
        async def worker(value):
            return value + 1

        async def root(value, cancel_scope, tracer):
            cancel_scope.cancel()
            tracer.start_span("worker")
            contextvars.copy_context()
            task = asyncio.create_task(worker(value))
            return task
        """,
    )

    plans = build_fusion_plans((scan,), _profile(members=(_member(),)))

    assert [rejection.code for rejection in plans[0].rejections] == [
        "static_cancellation",
        "static_instrumentation",
        "static_contextvars",
        "static_dynamic_effects",
    ]


def test_rejects_caller_suspension_after_spawn(tmp_path: Path) -> None:
    scan = _scan(
        tmp_path,
        """
        async def worker(value):
            return value + 1

        async def checkpoint():
            return None

        async def root(nursery, value):
            nursery.start_soon(worker, value)
            await checkpoint()
        """,
    )

    plans = build_fusion_plans((scan,), _profile(members=(_member(),)))

    assert [rejection.code for rejection in plans[0].rejections] == ["static_dynamic_effects"]


def test_rejects_callee_global_and_attribute_mutation(tmp_path: Path) -> None:
    scan = _scan(
        tmp_path,
        """
        total = 0

        async def worker(state, values, value):
            global total
            total += value
            state.value = value
            values[0] = value
            return value

        async def root(nursery, state, values, value):
            nursery.start_soon(worker, state, values, value)
        """,
    )

    plans = build_fusion_plans((scan,), _profile(members=(_member(),)))

    assert [rejection.code for rejection in plans[0].rejections] == ["static_dynamic_effects"]


def test_rejects_builtin_calls_that_can_reorder_observable_effects(tmp_path: Path) -> None:
    scan = _scan(
        tmp_path,
        """
        async def worker(value):
            print(value)
            return value

        async def root(nursery, value):
            nursery.start_soon(worker, value)
        """,
    )

    plans = build_fusion_plans((scan,), _profile(members=(_member(),)))

    assert [rejection.code for rejection in plans[0].rejections] == ["static_dynamic_effects"]


def test_rejects_runtime_import_exec_and_indirect_call_hazards(tmp_path: Path) -> None:
    scan = _scan(
        tmp_path,
        """
        async def worker(value):
            import contextvars
            from anyio import CancelScope
            exec("value = 2")
            (lambda: value)()
            return contextvars.copy_context(), CancelScope

        async def root(nursery, value):
            nursery.start_soon(worker, value)
        """,
    )

    plans = build_fusion_plans((scan,), _profile(members=(_member(),)))

    assert [rejection.code for rejection in plans[0].rejections] == [
        "static_cancellation",
        "static_contextvars",
        "static_dynamic_effects",
    ]


def test_rejects_spawn_without_a_target_argument(tmp_path: Path) -> None:
    scan = _scan(
        tmp_path,
        """
        async def root():
            asyncio.create_task()
        """,
    )

    plans = build_fusion_plans((scan,), _profile(members=()))

    assert [rejection.code for rejection in plans[0].rejections] == ["callee_unresolved"]


def test_does_not_match_spawn_api_suffix_inside_another_name(tmp_path: Path) -> None:
    scan = _scan(
        tmp_path,
        """
        async def worker(value):
            return value + 1

        async def root(nursery, value):
            nursery.restart_soon(worker, value)
        """,
    )

    assert fusion_observation_targets((scan,)) == ()
    assert build_fusion_plans((scan,), _profile(members=(_member(),))) == ()


def test_stable_ids_do_not_change_when_profile_counts_change(tmp_path: Path) -> None:
    scan = _scan(
        tmp_path,
        """
        async def worker(value):
            return value + 1

        async def root(nursery, value):
            nursery.start_soon(worker, value)
        """,
    )

    first = build_fusion_plans((scan,), _profile(members=(_member(call_count=25),)))[0]
    second = build_fusion_plans((scan,), _profile(members=(_member(call_count=50),)))[0]

    assert first.id == second.id
    assert first.source_hash == second.source_hash


def test_observation_target_discovery_resolves_spawn_styles(tmp_path: Path) -> None:
    scan = _scan(
        tmp_path,
        """
        async def direct(value):
            return value

        async def method(value):
            return value

        class Runner:
            async def method(self, value):
                return value

            async def root(self, nursery, value):
                nursery.start_soon(self.method, value)

        async def root(nursery, value):
            nursery.start_soon(direct, value)
            asyncio.create_task(method(value))
            ensure_future(direct(value))
        """,
    )

    assert fusion_observation_targets((scan,)) == (
        "sample::Runner.method",
        "sample::direct",
        "sample::method",
    )


def test_nested_spawn_is_not_attributed_to_enclosing_hot_root(tmp_path: Path) -> None:
    scan = _scan(
        tmp_path,
        """
        async def worker(value):
            return value

        async def root(nursery, value):
            async def nested():
                nursery.start_soon(worker, value)
            return nested
        """,
    )

    plans = build_fusion_plans((scan,), _profile(members=(_member(),)))

    assert plans == ()
