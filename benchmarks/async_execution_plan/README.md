# Async Execution Plan Feasibility

This directory holds Milestone 1 benchmark notes for the deterministic async
scheduler feasibility harness in `scripts/async_execution_plan_feasibility.py`.

The harness repeats canonical semantic comparisons 32 times without timestamps
or object addresses. It preserves baseline, task-preserving, and
callback-backed evidence, and adds a guarded fused-state-machine semantic arm.
Those semantic arms run over a generated workload that covers immediate work,
suspending and task-observing fallback, one shared capacity-one result channel,
deterministic reduction, context isolation, exception causes and notes,
cancellation, ordering, cleanup, custom task factories, and cold decoys.

The unsafe fused-state-machine arm is benchmark-only. It deliberately skips the
semantic protections needed for arbitrary coroutine work, so its wall-clock
ratio is useful only as a manual upper-bound signal inside this harness. The
guarded fused arm is the promotion candidate: it drives only non-suspending
coroutine work with coroutine `send` and `close` executed under a copied
`Context`, and retains task fallback for semantic probes and task-sensitive
work. A dedicated guarded-direct semantic probe verifies that the copied
context is observed during direct driving and that parent context is not
leaked.

The performance gate uses real wall-clock durations. It calibrates an immediate
integer fan-out workload until every benchmark arm takes at least 0.25 seconds,
then runs one rotating warmup and seven rotating samples. Diagnostic trace
construction is deliberately excluded from this timing workload so the result
measures scheduler allocation and dispatch rather than report formatting. The
fused benchmark arms still execute one coroutine per item, publish every result
into a bounded FIFO queue, and drain only after the queue is full. Those
benchmark-only batch-drain ratios stay local to the manual harness and are not
product throughput claims.

The CLI exits nonzero when semantic snapshots diverge, timing samples are below
the stability floor, or either fused arm misses its speedup gate:

```bash
uv run python scripts/async_execution_plan_feasibility.py
```

Milestone 1 requires semantic equivalence plus separate wall-clock ratios:
unsafe fused must be at least `3.30x` faster than baseline, and guarded fused
must be at least `3.00x` faster than baseline. The callback-backed ratio remains
reported as historical evidence, but it is not the fused feasibility gate.
Passing this experiment is permission to build the guarded product path; it is
not evidence that arbitrary async code can skip tasks.
