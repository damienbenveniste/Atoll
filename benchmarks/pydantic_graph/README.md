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

Each measured sample runs eight executions of the 5,000-item fan-out. This keeps interpreter
startup and the separate graph-construction checksum from dominating the async scheduling ratio.

Evidence includes cold and warm JSON/Markdown reports, compiler logs, compiler-probe events, and
source hash manifests. The run fails unless:

- both reports are schema version 4 and profile-guided, proving the profile pass ran on both cold
  and warm compiles;
- at least one execution plan is discovered, applied, and accepted by an execution-plan trial;
- each applied execution-plan trial records a cold staging-cache miss, a warm staging-cache hit, at
  least `1.05x` marginal speedup, and at least `1.10x` overall speedup;
- every accepted native candidate, when native typed-region evidence exists, measures at least
  `1.05x` marginal speedup;
- the final payload measures at least `1.10x` end-to-end speedup;
- execution-plan IDs, aggregate source hashes, and nonempty per-module source hashes remain stable
  between cold and warm reports;
- checkout source hashes remain unchanged;
- typed-region source hashes remain stable when typed-region evidence is present;
- the warm compile performs no native compiler phases or compiler-probe invocations;
- when native typed-region evidence exists, cold mypyc time is at most 50% of the recorded
  `192.701915s` reference, the cold run records native compiler phases and independent
  compiler-probe events, and the warm compile restores every native region from cache;
- the warm compile promotes a verified wheel.

The fixed cold-time reference came from the Apple Silicon environment described in
`baseline.json`. Its absolute comparison is useful regression evidence but is not a substitute for
same-machine timing. The end-to-end speedup and candidate decisions are paired within each run.
Zero cold native phases and compiler-probe calls are valid only for plan-only success, where the
accepted payload comes from execution-plan staging and the report contains no typed/native region
evidence. If native regions are present, the cold compiler and warm region-cache checks remain hard
requirements.

Task-fusion fields, when present, are compatibility and research evidence only. They are not a
promotion requirement for this benchmark. Pydantic Graph's scheduler now promotes through the
generic execution-plan path, so acceptance depends on applied execution-plan trials rather than
fusion-specific safety plans.

The GitHub workflow **Pydantic Graph Hard Benchmark** exposes the same gate through
`workflow_dispatch` and uploads schema-v4 cold/warm reports, logs, source manifests, compiler-probe
events, and the acceptance summary even when an acceptance condition fails.

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
