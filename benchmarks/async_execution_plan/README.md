# Async Execution Plan Feasibility

This directory holds Milestone 1 benchmark notes for the deterministic async
scheduler feasibility harness in `scripts/async_execution_plan_feasibility.py`.

The harness repeats canonical semantic comparisons 32 times without timestamps
or object addresses. It compares baseline, task-preserving, and callback-backed
scheduler arms over a generated workload that covers immediate work, suspending
and task-observing fallback, one shared capacity-one result channel,
deterministic reduction, context isolation, exception causes and notes,
cancellation, ordering, cleanup, custom task factories, and cold decoys.

The performance gate uses real wall-clock durations. It calibrates an immediate
integer fan-out workload until every arm takes at least 0.25 seconds, then runs
one rotating warmup and seven rotating samples. Diagnostic trace construction
is deliberately excluded from this timing workload so the result measures
scheduler allocation and dispatch rather than report formatting.

The CLI exits nonzero when semantic snapshots diverge or when the callback-backed
arm fails the required speedup gate:

```bash
uv run python scripts/async_execution_plan_feasibility.py
```

The callback-backed arm is feasible only when its semantic snapshots match and
its median is at least `1.50x` faster than baseline. Passing this experiment is
permission to build the guarded product path; it is not evidence that arbitrary
async code can skip tasks.
