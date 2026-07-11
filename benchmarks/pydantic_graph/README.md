# Pydantic Graph hard benchmark

This manual acceptance benchmark pins Pydantic AI at
`e6ff64409f74124de581068be644a3dbf8999e7d` and exercises Pydantic Graph's real
fan-out, task scheduling, async iteration, and reduction path. It is a hard external target, not a
source of Pydantic-specific compiler rules.

Run it from the Atoll checkout with a new disposable workspace:

```bash
uv run --python 3.12 python scripts/run_pydantic_graph_benchmark.py \
  --workspace /tmp/atoll-pydantic-ai \
  --evidence-root /tmp/atoll-pydantic-graph-evidence
```

The runner clones the pinned revision, materializes the workload outside the target checkout,
adds benchmark policy only to the disposable `pydantic_graph/pyproject.toml`, and invokes the same
source-clean compile twice. It never edits `pydantic_graph/*.py`.

Evidence includes cold and warm JSON/Markdown reports, compiler logs, compiler-probe events, and
source hash manifests. The run fails unless:

- both reports are schema version 3 and profile-guided;
- every accepted candidate measures at least `1.01x` marginal speedup;
- the final payload measures at least `1.10x` end-to-end speedup;
- cold mypyc time is at most 50% of the recorded `192.701915s` reference;
- the cold run records native compiler phases and independent compiler-probe events;
- the warm compile restores every region without a native compiler invocation;
- typed-region, task-fusion-plan, and checkout source hashes remain unchanged;
- the report contains task-fusion safety evidence and, when the safe payload misses `1.10x`, every
  eligible plan has a passing plan-bound three-arm trial;
- the warm compile promotes a verified wheel.

The fixed cold-time reference came from the Apple Silicon environment described in
`baseline.json`. Its absolute comparison is useful regression evidence but is not a substitute for
same-machine timing. The end-to-end speedup and candidate decisions are paired within each run.
Zero cold mypyc time is valid when backend selection routes every hot region directly to Cython;
the separate cold native-phase and compiler-probe requirements prevent that from becoming a
no-compilation false positive.

Task-fusion plans are evidence, not an enabled optimization. Pydantic Graph's scheduler intentionally
stresses overlapping tasks, suspension, cancellation, stream dispatch, and instrumentation. A plan
that detects any of those conditions is rejected before a fused variant runs. The public
`experimental_task_fusion` setting remains absent unless a safe plan later passes semantic checks,
at least `1.05x` over the unfused payload, and at least `1.10x` over baseline.

The GitHub workflow **Pydantic Graph Hard Benchmark** exposes the same gate through
`workflow_dispatch` and uploads evidence even when an acceptance condition fails.

## Optimization ceiling experiment

Before promoting scheduler-level optimization into `atoll compile`, run the
disposable ceiling experiment against an existing pinned checkout:

```bash
uv run --python 3.12 python -m scripts.run_pydantic_graph_ceiling_experiment \
  --checkout /tmp/atoll-pydantic-ai \
  --evidence-root /tmp/atoll-pydantic-graph-ceiling
```

The experiment never edits the checkout. It separates unchanged source, lazy
guarded reducer-signature hoisting, unsafe result buffering, and an unsafe
combined immediate-completion ceiling. The last two arms deliberately change
observable async behavior and are not candidates for wheel promotion. A
`1.15x` ceiling only recommends investigating a separately proven guarded
design; it does not establish semantic equivalence or an Atoll speedup.
