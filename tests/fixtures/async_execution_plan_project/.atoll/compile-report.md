# Atoll Compilation Report

## Summary

- Operation: compile
- Mode: source-clean
- Status: failed
- Module filter: all discovered modules
- Wheel: none
- Legacy islands: 0
- Typed regions: 9
- Compiled regions: 0
- Symbols: 0
- Artifacts: 0
- Support artifacts: 0
- Skipped modules: 0
- Preflight blockers: 0
- Legacy island verifications: 0
- Legacy verification failures: 0
- Subprocess verifications: 0
- Subprocess verification failures: 0
- Semantic tests: not run
- Performance: invalid
- Profitable hot-path coverage: 0.0%
- Execution plans: 3 candidates, 1 selected, 0 applied
- Execution-plan trials: 0
- Task-fusion plans: 0 total, 0 eligible
- Task-fusion trials: 0
- Build duration: 2.742s

## Verification Scope

Source-clean compile overlays native artifacts and staged shims onto the target project's normal PEP 517 wheel. Fresh child interpreters verify both the unpacked payload and final wheel. Semantic equivalence and speedup are claimed only when the configured test and benchmark gates pass.

## Profile-Guided Selection

- Status: profiled
- Reason: baseline profile collected
- Sample sufficiency: 134 total sample(s), 3 mapped to project code
- Mapped coverage: 2.2%
- Selected hot coverage: 0.0%
- Unmeasured profiling passes: sampling 0.301s, types 0.333s
- Selected candidates: none
- Rejected candidates: `execution_plan_fixture.workflow::_capture_immediate_failure` (below-threshold), `execution_plan_fixture.workflow::_context_isolation_probe` (below-threshold), `execution_plan_fixture.workflow::run_supported_workflow` (below-threshold), `execution_plan_fixture.workflow::cold_cancellation_workflow` (below-threshold), `execution_plan_fixture.workflow::cold_custom_task_factory_workflow` (below-threshold), `execution_plan_fixture.workflow::publish_immediate` (below-threshold)
- Bounded type observation reached: `execution_plan_fixture.workflow::_capture_immediate_failure`, `execution_plan_fixture.workflow::_context_isolation_probe`, `execution_plan_fixture.workflow::run_supported_workflow`, `execution_plan_fixture.workflow::publish_immediate`

## Async Execution Plans

- `exec-plan-6e3dd98b7fe7b256ebdbc919a974afd3` [selected]: `execution_plan_fixture.workflow::run_supported_workflow`; dialect `asyncio`; observed invocations 3552; lifecycle starts 3552 (100.0% mapped async activity); selected for backend assessment
- `exec-plan-rejection-0fb5a2abce494ef8` [rejected]: `execution_plan_fixture.workflow::cold_custom_task_factory_workflow`; dialect `asyncio`; observed invocations 0; lifecycle starts 0 (0.0% mapped async activity); ambiguous-spawn: unresolved spawned callee
- `exec-plan-rejection-157e3eda3c577e7b` [rejected]: `execution_plan_fixture.workflow::cold_cancellation_workflow`; dialect `asyncio`; observed invocations 0; lifecycle starts 0 (0.0% mapped async activity); ambiguous-spawn: unresolved spawned callee
- Applied plans: none
- Runtime status: report-only unless an applied plan and passing trial are listed.

## Suspension Handling

- `execution_plan_fixture.workflow::_capture_immediate_failure`: interpreted (await at line 142); 0/0 synchronous blocks eligible; no accepted compiled binding replaced this callable
- `execution_plan_fixture.workflow::run_supported_workflow`: interpreted (async_with at line 128, await at line 131, await at line 133); 1/1 synchronous blocks eligible; no accepted compiled binding replaced this callable
- `execution_plan_fixture.workflow::cold_custom_task_factory_workflow`: interpreted (await at line 234); 1/1 synchronous blocks eligible; no accepted compiled binding replaced this callable
- `execution_plan_fixture.workflow::_side_effecting_values`: interpreted (yield at line 274); 0/0 synchronous blocks eligible; no accepted compiled binding replaced this callable
- `execution_plan_fixture.workflow::cold_cancellation_workflow`: interpreted (await at line 245); 1/1 synchronous blocks eligible; no accepted compiled binding replaced this callable
- `execution_plan_fixture.workflow::cold_suspension_workflow`: interpreted (await at line 216); 0/1 synchronous blocks eligible; no accepted compiled binding replaced this callable

## Planned Regions

- `execution_plan_fixture.workflow::ControlledImmediateError:0fa7cb07119a`: execution_plan_fixture.workflow::ControlledImmediateError, execution_plan_fixture.workflow::TracebackEvidence, execution_plan_fixture.workflow::WorkItem, execution_plan_fixture.workflow::_capture_immediate_failure, execution_plan_fixture.workflow::_successful_items, execution_plan_fixture.workflow::fail_immediate, execution_plan_fixture.workflow::publish_immediate, execution_plan_fixture.workflow::run_supported_workflow
- `execution_plan_fixture.workflow::SemanticSnapshot:277033b74ee9`: execution_plan_fixture.workflow::SemanticSnapshot
- `execution_plan_fixture.workflow::_cold_decoy_names:8a7278f5206f`: execution_plan_fixture.workflow::_cold_decoy_names
- `execution_plan_fixture.workflow::_custom_task_factory:55bb935d65e1`: execution_plan_fixture.workflow::_custom_task_factory, execution_plan_fixture.workflow::cold_custom_task_factory_workflow
- `execution_plan_fixture.workflow::_side_effecting_values:057c357664c0`: execution_plan_fixture.workflow::_side_effecting_values, execution_plan_fixture.workflow::cold_side_effecting_iterable_workflow
- `execution_plan_fixture.workflow::cold_cancellation_workflow:ba3047ada71d`: execution_plan_fixture.workflow::cold_cancellation_workflow
- `execution_plan_fixture.workflow::cold_debug_mode_workflow:9440c2f98275`: execution_plan_fixture.workflow::cold_debug_mode_workflow
- `execution_plan_fixture.workflow::cold_suspension_workflow:aa52e4b23edf`: execution_plan_fixture.workflow::cold_suspension_workflow
- `execution_plan_fixture.workflow::cold_task_introspection_workflow:91d8073c25ec`: execution_plan_fixture.workflow::cold_task_introspection_workflow

## Build

- Success: no
- Command: `/Users/damienbenveniste/Projects/agentic-management/Atoll/.venv/bin/python -I -m build --wheel --outdir /Users/damienbenveniste/Projects/agentic-management/Atoll/tests/fixtures/async_execution_plan_project/.atoll/dist/build/pep517-dist`
- Cache: disabled
- Error: `baseline profile found no credible hot project members in the compile scope`

### Phase Timings

- pep517_project_copy: 0.003s (source-clean project copy)
- pep517_wheel: 2.103s (exit 0)
- wheel_unpack: 0.002s (async_execution_plan_project-0.1.0-py3-none-any.whl)
- baseline_payload_copy: 0.001s (benchmark baseline)
- profile_sampling: 0.301s (exit 0)
- profile_types: 0.333s (exit 0)

## Test Gate

- baseline: `python -m pytest tests -q`
  exit 0, 0.117s, passed

## Package Verification

- Not run

## Performance

- Status: invalid
- Reason: baseline profile found no credible hot project members in the compile scope
- Minimum speedup: 1.100x

## Cleanup

- Removed: none
- Kept `.atoll/dist/build`
- Kept `.atoll/dist/install`

## Islands

- None
