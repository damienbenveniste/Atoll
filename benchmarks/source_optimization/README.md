# Source optimizer hard benchmark

The generic hard benchmark exercises Atoll's guarded async source-optimization model without any
application-specific compiler rule. It covers immediate and suspending workers, task observation,
bounded delivery, deterministic reduction, copied `Context` isolation, exceptions, cancellation,
cleanup, custom task factories, and cold decoys.

Run the gate from the Atoll checkout:

```bash
uv run --python 3.12 python scripts/async_execution_plan_feasibility.py
```

The harness performs 32 canonical semantic comparisons, calibrates every timing arm above 0.25
seconds, then runs one rotating warmup and seven samples. It fails unless the unsafe ceiling is at
least `3.30x` and the copied-context guarded state machine is at least `3.00x`. The callback-backed
and task-preserving arms remain comparison evidence; they do not satisfy source-patch promotion.

A July 11, 2026 Apple Silicon run measured `8.858x` for the guarded arm and `10.738x` for the unsafe
ceiling. These are same-machine feasibility results, not universal application speedup claims. A
real project must still pass its configured semantic command and the separate transformed-source
and normal-wheel `3.0x` gates.

The manual GitHub workflow **Generic Source Optimizer Hard Benchmark** runs the same command and
uploads its JSON evidence. Ordinary CI exercises deterministic semantics across supported Python
versions without enforcing host-dependent timing ratios.
