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

Each measured sample runs eight executions of the 5,000-item fan-out and sets
`--build-repetitions 0`. The semantic command still verifies graph construction, but the hard
performance gate isolates the async execution pipeline that the source plan transforms.

Evidence includes cold and warm schema-v6 JSON/Markdown reports, compile logs, compiler-probe
events, source manifests, the accepted patch, and an acceptance summary. The run fails unless:

- both compiles rerun profile collection and report a selected source plan with valid per-file
  source hashes;
- the accepted plan represents at least 10,000 work items and 70% of the attributable hot path;
- one accepted trial contains private transport draining, copied-context quiescent execution,
  local state-machine fusion, and private protocol forwarding;
- both transformed source and its normal PEP 517 wheel measure at least `3.0x` over baseline;
- the final composed payload retains a native variant and improves the accepted source-only wheel
  by at least `1.05x`;
- cold and warm runs each contain seven fresh baseline/optimized timing pairs;
- source-plan identity, source hashes, candidate ID, transformation IDs, and patch path remain
  identical between runs;
- patch generation reports a cold cache miss and a warm cache hit while profitability is still
  remeasured on both invocations;
- the patch remains under `.atoll/patches/`, is not applied by default, and exists beside the
  promoted wheel;
- checkout Python source hashes remain unchanged;
- the warm run invokes no native compiler and contains no native compiler phase.

The July 11, 2026 Apple Silicon acceptance run measured `4.982x` for the cold final wheel and
`5.008x` for the warm final wheel; the warm transformed-source ratio was `5.001x`. These ratios are
paired same-machine evidence for the pinned workload, not a universal Python or Pydantic claim.
`baseline.json` retains the older native-compiler reference only as historical regression data.
Task-fusion and execution-plan fields remain compatibility and research evidence; schema-v6 policy,
stage-median, final-composition, and source-trial evidence control this benchmark's promotion.

The GitHub workflow **Pydantic Graph Hard Benchmark** exposes the same gate through
`workflow_dispatch` and uploads schema-v6 cold/warm reports, logs, source manifests, compiler-probe
events, and the acceptance summary even when an acceptance condition fails.

## Optimization ceiling experiment

Before promoting scheduler-level optimization into `atoll compile`, run the
disposable ceiling experiment against an existing pinned checkout:

```bash
uv run --python 3.12 python -m scripts.run_pydantic_graph_ceiling_experiment \
  --checkout /tmp/atoll-pydantic-ai \
  --evidence-root /tmp/atoll-pydantic-graph-ceiling
```

The experiment never edits the checkout. It separates unchanged source,
guarded reducer-signature hoisting, result buffering, immediate execution,
nonblocking batch draining, an absolute unsafe ceiling, and a context-isolated
guarded scheduler fusion.

The guarded arm keeps the existing graph scheduler and combines copied-context
immediate execution, batch draining, and lazy reducer scanning. It falls back
for suspension and task-sensitive shapes before transformed execution begins.
The experiment proceeds to product work only when the unsafe ceiling reaches
`3.30x`, the guarded arm reaches `3.00x`, every correctness probe passes, the
guarded route is exercised, and checkout source hashes remain unchanged. The
unsafe and intermediate arms remain research evidence and are never promotion
candidates.
