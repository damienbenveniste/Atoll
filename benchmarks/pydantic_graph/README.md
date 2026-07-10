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
- typed-region and checkout source hashes remain unchanged; and
- the warm compile promotes a verified wheel.

The fixed cold-time reference came from the Apple Silicon environment described in
`baseline.json`. Its absolute comparison is useful regression evidence but is not a substitute for
same-machine timing. The end-to-end speedup and candidate decisions are paired within each run.
Zero cold mypyc time is valid when backend selection routes every hot region directly to Cython;
the separate cold native-phase and compiler-probe requirements prevent that from becoming a
no-compilation false positive.

The GitHub workflow **Pydantic Graph Hard Benchmark** exposes the same gate through
`workflow_dispatch` and uploads evidence even when an acceptance condition fails.
