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
explicit `--allow-unsandboxed` escape hatch selects the host itself as the
isolation boundary and is intended only for a disposable machine whose external
code risk is understood. The trusted-branch GitHub workflows set this flag
because each corpus case owns an ephemeral VM; local commands omit it by
default.

Repository cases are added only after their dependency lock and canonical oracle
are present. Git is the default source provider. A case whose qualifying release
cannot be represented without a Git submodule may instead lock an HTTPS sdist by
exact byte count, archive SHA-256, and normalized regular-file tree digest. Archive
extraction rejects links, special files, traversal, duplicate paths, multiple roots,
VCS administration paths, and `.gitmodules` before external code executes.
External code is executed by the isolated lifecycle runner, never by manifest
validation, lock inspection, matrix generation, or aggregation. Successful runs
delete materialized sources and wheels while retaining bounded logs, source
manifests, archive identity when applicable, policy evidence, toolchain identity,
compile reports, and wheel digests.

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

After reviewing the complete raw evidence, retain a compact immutable snapshot
and refresh only the generated history block in the benchmark documentation:

```bash
uv run python -m scripts.benchmark_corpus promote \
  --tier performance \
  --platform ubuntu-24.04 \
  --results-root .atoll/corpus-results \
  --label 2026-07-13-initial \
  --reviewed-by REVIEWER
```

Promotion is intentionally manual and deterministic. Reusing a label with
different evidence fails, raw samples and wheels remain workflow artifacts,
and CI never commits snapshots. Performance promotion requires every result's
adjacent `experiment.json` to carry the same GitHub run ID, run attempt,
workflow ref, head SHA, and reviewed label. Missing, mixed, or independently
renamed workflow evidence is rejected. Historical comparison is valid only for
cases whose retained comparison keys match.

`calibration.toml` separately pins the existing scalar, call-chain, and buffer
fixtures plus pyperformance's Richards, spectral-norm, and Hexiom workloads.
Normal validation rehashes the complete repository-owned execution bundle, not
only the three named wrappers: the hard-suite runner and every non-generated
file copied from the native fixture are covered. The authoritative native hard
workflow runs this validation before executing its command template with
explicit workspace and evidence destinations. The three external pyperformance
pins are reported as not yet checkout-verified and deliberately have no ambient
`python -m pyperformance` runner. Authenticate an existing detached
pyperformance checkout before using those entries:

```bash
uv run python -m scripts.benchmark_corpus validate \
  --calibration-checkout /path/to/pinned/pyperformance
```

That check requires the exact catalogued `HEAD`, a clean tracked tree, and each
reviewed source digest. Every entry sets
`included_in_repository_aggregate = false`, so calibration headroom, including
narrow 22x-style kernels, is never included in either real-repository geometric
mean. Semantic-corruption, upstream-failure, and compatible-no-op lifecycle
fixtures serve as negative controls and likewise do not enter repository
performance aggregates.

The version-1 compatibility matrix contains all 25 planned repositories on both
Ubuntu 24.04 and macOS 14. Pins use the inspected default-branch revision when it
passes baseline qualification. SQLGlot uses the newest qualifying Python 3.12
release revision because newer revisions contain a forbidden submodule.
html5lib retains the full release commit as provenance but materializes the
official 1.1 sdist, whose content-addressed lock includes the test data without
relaxing the Git-source submodule prohibition. Its source-only test dependency
is hash-pinned with the rest of the lock, downloaded during the one networked
bootstrap phase, and built from the case-local wheelhouse after offline mode is
enforced.
