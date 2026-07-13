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

Regenerate selected locks from their reviewed inputs when dependencies are
intentionally changed:

```bash
uv run python -m scripts.benchmark_corpus lock --write --case CASE_ID
```

Lock generation targets Python 3.12 across supported platforms, includes
distribution hashes, and excludes artifacts published after July 13, 2026.
The generated header records the exact regeneration command. Validation rejects
unhashed or non-exact requirements, direct URLs, editable installs, local paths,
and package-index directives before a case can execute.

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
manifest validation, lock inspection, matrix generation, or aggregation. Successful runs
delete cloned sources and wheels while retaining bounded logs, source manifests,
policy evidence, toolchain identity, compile reports, and wheel digests.

Twelve cases also carry the `performance` tier. Each uses one reviewed workload,
the fixed seed `1729`, one warmup, and seven measured baseline/compiled pairs.
Manifest validation and the lifecycle verify a bundle digest covering the
workload, case adapter, shared runner, golden output, and notice before use. The
adapter rejects any default result that differs from `workloads/golden.json`, and
the lifecycle independently verifies that every timed run imports from its
declared payload and produces the same canonical output. Run one measured case
with the same command shape:

```bash
uv run python -m scripts.benchmark_corpus run pydantic \
  --tier performance \
  --platform ubuntu-24.04 \
  --workspace-root .atoll/corpus-work \
  --evidence-root .atoll/corpus-results/pydantic
```

Aggregate a complete tier and platform slice only after every expected case has
produced `case-result.json`:

```bash
uv run python -m scripts.benchmark_corpus aggregate \
  --tier performance \
  --platform ubuntu-24.04 \
  --results-root .atoll/corpus-results \
  --output-root .atoll/corpus-aggregate
```

Aggregation rejects missing, duplicate, malformed, stale, or cross-platform
evidence. It reports an accepted-case geometric mean and an effective corpus
speedup where supported no-ops, unsupported cases, and unprofitable candidates
contribute `1.0x`. Infrastructure and semantic failures remain separate and make
the aggregate command exit nonzero. Case reports label ratios as "Python rewrite
versus original," "final wheel versus original," and "native layer versus
source-only wheel."

The version-1 compatibility matrix contains all 25 planned repositories on both
Ubuntu 24.04 and macOS 14. Pins use the inspected default-branch revision when it
passes baseline qualification. SQLGlot uses the newest qualifying Python 3.12
release revision because newer revisions contain a forbidden submodule.
html5lib remains in the matrix even though its newest Python 3.12 release still
contains a test-data submodule; its explicit `security-violation` result is part
of corpus coverage rather than an omitted hard case.
