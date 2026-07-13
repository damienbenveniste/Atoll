# Multi-Repository Corpus

This directory contains immutable external-project metadata, reviewed dependency
constraints, deterministic workload adapters, and manually promoted evidence for
Atoll's real-repository corpus.

Validate the current manifest without cloning or executing external code:

```bash
uv run python -m scripts.benchmark_corpus validate
```

Verify the reviewed dependency-lock identities, optionally for one case:

```bash
uv run python -m scripts.benchmark_corpus lock --case CASE_ID
```

Run one pinned case into disposable workspace and persistent evidence roots:

```bash
uv run python -m scripts.benchmark_corpus run CASE_ID \
  --tier compatibility \
  --platform ubuntu-24.04 \
  --workspace-root .atoll/corpus-work \
  --evidence-root .atoll/corpus-results/CASE_ID
```

Local execution requires `sandbox-exec` on macOS or Bubblewrap on Linux. The
explicit `--allow-unsandboxed` escape hatch is intended only for a disposable
machine whose external code risk is understood.

Repository cases are added only after their dependency lock and canonical oracle
are present. External code is executed by the isolated lifecycle runner, never by
manifest validation, lock inspection, or matrix generation. Successful runs
delete cloned sources and wheels while retaining bounded logs, source manifests,
policy evidence, toolchain identity, compile reports, and wheel digests.
