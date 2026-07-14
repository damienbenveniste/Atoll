# Benchmarks

Atoll uses four evidence groups with different meanings. They are deliberately
not pooled into one headline number.

| Group | Question | Included in real-repository aggregate |
| --- | --- | --- |
| Compatibility | Can Atoll process the complete pinned project without changing its checkout or breaking its oracle? | Status coverage only |
| Performance | Does the final Atoll wheel improve a deterministic real workload? | Yes, per platform |
| Calibration | Can a concentrated compiler kernel expose optimization headroom? | No |
| Negative control | Does the runner correctly detect no-ops, upstream failures, and semantic corruption? | No |

## Repository Corpus

The schema-v1 manifest pins 25 repositories at full commit SHAs. Git is the
default source provider; an optional content-addressed sdist retains the commit
as provenance when a qualifying release cannot be represented without a Git
submodule. Every case compiles the complete declared project root with both
mypyc and Cython enabled; the runner does not choose favorable modules. Twelve
cases add reviewed, seeded performance workloads covering validation, parsing,
rendering, graph algorithms, symbolic work, typing, ORM construction, and async
fan-out.

A clean no-op is `supported-no-op`: it is compatible but not accelerated.
Unsupported and unprofitable cases stay in the result set. Setup, timeout,
compiler, security, semantic-regression, and unstable outcomes remain explicit
rather than being dropped from the denominator.

The aggregate exposes two geometric means:

- **Accepted-only geometric mean** includes only cases classified as
  `accelerated`.
- **Effective corpus speedup** gives valid no-op, unsupported, and
  not-profitable cases `1.0x`; invalid infrastructure or semantic results are
  counted separately rather than converted into performance observations.

Ubuntu and macOS results are never combined.

## Ratio Labels

Reports and snapshots use three names with distinct denominators:

- **Python rewrite versus original** is the original Python median divided by
  the accepted rewritten-source median.
- **Final wheel versus original** is the original Python median divided by the
  final Atoll wheel median. This is the end-to-end product result.
- **Native layer versus source-only wheel** is the accepted source-only wheel
  median divided by the final composed wheel median. This measures only the
  native layer added after source optimization.

These ratios answer different questions. They must not be added or described
as interchangeable speedups.

## Isolation And Evidence

Each case runs from a fresh immutable source materialization in an isolated
environment. Git cases validate the pinned detached `HEAD` and reject
submodules, unresolved Git LFS pointers, escaping symlinks, and pre-existing
Atoll policy. Archive cases authenticate the byte size and SHA-256 before tar
parsing, safely extract one regular-file root, and verify its normalized tree
digest. The runner then appends the reviewed benchmark policy only to the
disposable `pyproject.toml`. Dependency bootstrap is network-enabled once;
project builds, focused tests, oracles, and benchmarks then use the offline
wheelhouse and a sanitized credential-free environment. Exact lock hashes cover
both wheels and source-only test tools; any source distribution is downloaded
during bootstrap and built only after package tooling switches to offline mode.
Archive baseline wheels use a disposable source copy, preserving the
content-addressed extraction for identity checks and the later Atoll compile.

Evidence retains bounded logs, source manifests, toolchain and runner identity,
policy patches, compiler probes, compile reports, canonical oracle digests, and
wheel digests. Materialized source and wheel payloads are deleted rather than
uploaded. A warm run uses only the first run's case-local Atoll cache and must
invoke no native compiler.

Compatibility compiles default to 45 minutes. Large cases identified by an
observed cold compile may declare the existing 90-minute per-case override;
the runner still records and classifies any timeout rather than omitting it.

## Running The Corpus

Validate metadata without cloning or executing external code:

```bash
uv run python -m scripts.benchmark_corpus validate
```

Run one case on a machine with the supported platform sandbox:

```bash
uv run python -m scripts.benchmark_corpus run pydantic \
  --tier performance \
  --platform ubuntu-24.04 \
  --workspace-root .atoll/corpus-work \
  --evidence-root .atoll/corpus-results/pydantic
```

The weekly **Multi-Repository Compatibility Corpus** workflow runs all 25
Ubuntu cases with at most four VMs in parallel. The manual
**Multi-Repository Performance Corpus** workflow selects one or all 12 cases,
one platform, and an experiment label. Both workflows run only from the trusted
default branch, retain evidence for 30 days, and write aggregates to the GitHub
workflow summary. Neither runs on pull requests or commits history.
Each scheduled case explicitly uses its ephemeral VM as the external-code
boundary. Local runs remain sandboxed by default and require an explicit
`--allow-unsandboxed` acknowledgement to use the same mode.

## Reviewed History

After examining a complete tier/platform evidence slice, a maintainer promotes
its compact snapshot explicitly:

```bash
uv run python -m scripts.benchmark_corpus promote \
  --tier performance \
  --platform ubuntu-24.04 \
  --results-root .atoll/corpus-results \
  --label 2026-07-13-initial \
  --reviewed-by REVIEWER
```

Promotion retains status, comparison keys, unambiguous ratios, and aggregate
statistics under `benchmarks/corpus/history/`. It omits raw samples, logs, and
wheels. Reusing a label for different evidence fails. For performance evidence,
every case must retain one identical `experiment.json` identity containing the
GitHub run ID, run attempt, workflow ref, head SHA, and label. Promotion requires
that label to equal `--label` and rejects cases mixed across workflow runs.
Historical performance is comparable only when the upstream revision, workload,
policy, dependency, Python, compiler, platform, and hardware fingerprints match;
the Atoll revision is recorded but intentionally excluded from that comparison
key.

<!-- corpus-history:start -->
No reviewed corpus snapshots have been promoted.
<!-- corpus-history:end -->

## Existing Experiment Evidence

The July 12, 2026 Apple Silicon Pydantic Graph ratios have been withdrawn.
Payload verification imported the compiled tree before timing and left
`__pycache__` files that were then repacked into the candidate wheel. The
baseline tree did not receive equivalent bytecode caches, so the reported
final-wheel and native-layer ratios did not isolate Atoll's optimizations. A
replacement result must come from the corrected bytecode-neutral pipeline and
pass the existing hard benchmark before it is promoted here.

The older Pydantic Graph compiler baseline records 192.702 seconds in cold
`mypycify` work at Atoll revision `77d95c0`. It is compile-time regression
evidence, not a runtime speedup.

Callback-backed scheduling, immediate execution, task fusion, and unsafe fused
arms remain ceiling or research experiments. They are not promoted results
when task identity, context isolation, cancellation, or other required
semantics are rejected. No ratio from those arms contributes to reviewed
history.

## Calibration And Controls

`benchmarks/corpus/calibration.toml` pins the scalar, call-chain, and buffer
native fixtures plus pyperformance's Richards, spectral-norm, and Hexiom
workloads. Normal validation authenticates the local hard-suite runner and every
non-generated file that its fixture copy can execute, so changing a helper,
kernel, fixture test, or project policy invalidates the reviewed bundle digest.
The native hard workflow validates that bundle before running it. The three
external pins require an exact detached-checkout check: passing
`--calibration-checkout PATH` to `validate` verifies the checkout `HEAD`, clean
state, and source digests. External entries have no ambient runner. Every entry
explicitly sets `included_in_repository_aggregate = false`; narrow 22x-style
kernel results can therefore show compiler potential without inflating
real-project coverage.

The lifecycle regression fixtures intentionally exercise upstream baseline
failure, compatible no-op output, compiled-wheel corruption, source mutation,
and warm-cache behavior. These negative controls validate attribution and
semantic gates, but they are never treated as repository acceleration cases.
