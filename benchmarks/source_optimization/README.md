# Source optimizer hard benchmark

The generic hard benchmark exercises Atoll's guarded async source-optimization model without any
application-specific compiler rule. It covers immediate and suspending workers, task observation,
bounded delivery, deterministic reduction, copied `Context` isolation, exceptions, cancellation,
cleanup, custom task factories, and cold decoys.

Run the promoted residual-stage gate from the Atoll checkout:

```bash
uv run --python 3.12 python scripts/run_residual_async_profile_benchmark.py
```

The harness repeats canonical semantic comparisons, calibrates each arm independently above 0.25
seconds, then runs one rotating warmup and seven samples. Speedup compares median seconds per
logical workload execution, so the faster arm needs no synthetic delay. It fails unless every
residual stage is exercised and the guarded residual pipeline is at least `3.00x` faster than
baseline. The stage counters cover run guard amortization, quiescent await-chain collapse,
context-copy elision, incremental completion accounting, and private result-record projection.

The earlier `scripts/async_execution_plan_feasibility.py` command remains a ceiling and semantic
research harness. Its callback-backed, task-preserving, guarded, and unsafe arms do not replace the
residual-stage or project-specific source-patch promotion gates. A real project must still pass its
configured semantic command and the separate transformed-source and normal-wheel `3.0x` gates.

The manual GitHub workflow **Generic Source Optimizer Hard Benchmark** runs the same command and
uploads its JSON evidence. Ordinary CI exercises deterministic semantics across supported Python
versions without enforcing host-dependent timing ratios.
