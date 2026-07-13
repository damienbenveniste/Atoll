# Native optimizer hard benchmarks

These manual gates exercise Atoll's generic fixed-width scalar, direct call-chain, and standard
buffer optimizers. The fixture contains no application-specific production rule and the runner
compiles a disposable copy without changing the repository fixture.

Run the family gate from the Atoll checkout with new output paths:

```bash
uv run --python 3.12 python scripts/run_native_optimizer_benchmark.py \
  --workspace /tmp/atoll-native-optimizer \
  --evidence-root /tmp/atoll-native-optimizer-evidence
```

The runner compiles cold and warm, installs the promoted wheel into an isolated payload, runs the
semantic suite against baseline and compiled imports, and then measures one warmup plus seven
rotating pairs for each family. Both medians must be at least 0.25 seconds. Scalar, call-chain, and
buffer workloads must each reach `3.0x`; a result from one family cannot satisfy another family's
gate. Scalar evidence combines a polynomial loop, a staticmethod reduction, and branch-heavy
arithmetic rather than relying on one sum-of-squares kernel.

The acceptance summary also requires unchanged source hashes, schema-v6 composition evidence,
native artifacts, at least one cold compiler invocation, and zero compiler invocations or native
compiler phases on the warm run. Full compile logs, reports, test output, compiler-probe events, the
unpacked wheel payload, and `summary.json` remain under the evidence root.

Cold Cython batching has a separate representative gate:

```bash
uv run --python 3.12 python scripts/run_cython_batch_benchmark.py \
  --units 8 \
  --samples 3 \
  --evidence /tmp/atoll-cython-batch.json
```

It compares the same independently fingerprinted extensions sequentially and in one bounded
parallel build. The gate requires artifact parity, at least a 20% median cold-time reduction, a warm
cache hit for every unit, and zero warm native compiler phases. These workflows are manual because
wall-clock ratios depend on the host; ordinary CI retains deterministic semantic and cache tests.
