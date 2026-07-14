# Atoll

Atoll finds typed, CPU-bound regions in Python projects that can be extracted into native
extensions without rewriting the original source tree. It explains why code is accepted or
rejected, compiles eligible functions and methods with mypyc or Cython, composes guarded native and
source optimizations, and preserves interpreted Python as the fallback.

## Quickstart

```bash
uv run atoll scan .
uv run atoll compile your_package.hot_path
uv run atoll compile --root /path/to/project
```

Run these commands from the Python project you want to analyze, with Atoll installed in that
project's environment. Scanning writes human-readable and JSON reports under `.atoll/`; compiling
produces a wheel under `.atoll/dist/` while leaving the source checkout untouched. `--root ROOT`
selects the target checkout; it does not change the compile contract or make source edits.

## Scan Python modules

Atoll implements project analysis plus source-clean native compilation:
project discovery, AST scanning, import/constant/symbol extraction, dynamic blocker detection,
mypy diagnostic mapping, same-module dependency edges, backend-neutral typed regions,
conservative island candidates,
poison-residue reporting, JSON/Markdown reports, backend assessment, native artifact caching,
PEP 517 wheel overlay, and runtime verification.

```bash
uv run atoll scan . --source-root src
```

Reports are written to `.atoll/report.json` and `.atoll/report.md`.

Report schema v3 records exact callable annotations, type parameters, class ownership,
descriptor and execution kinds, class fields, and connected typed regions. These regions preserve
source typing for backend assessment, fixed-width and buffer proofs, directed call-chain slices,
and method-level source-clean compilation.

Atoll's compiler boundary is backend-neutral. `CompilerBackend` separates per-member capability
assessment, registration of prepared compilation units, native compilation, strict fingerprinting,
and diagnostic normalization. The mypyc and Cython adapters report member-level capabilities,
record native artifacts with region ownership and ABI/platform metadata, and normalize backend
diagnostics. The existing `build_sidecars` facade remains available only for explicit in-place
enable/build workflows.

Use `--no-mypy` to skip mypy diagnostics during a scan.

Candidate scores are 0-100 scan-only heuristics for extraction safety, not predicted speed.
Candidate risk is also extraction risk: `low` means Atoll only saw high-confidence internal
dependencies. Frame-introspection code such as `inspect.currentframe()`, `sys._getframe()`, and
direct frame attributes such as `f_locals` are hard blockers because mypyc changes Python frame
semantics.

## Compile candidates

Use `compile` for the normal workflow. It scans the target package, forms typed regions, asks the
configured backends to assess each region automatically, compiles and caches supported variants,
overlays staged routing code and region-owned native artifacts onto the project's normal PEP 517
wheel, verifies the result, and removes temporary artifacts by default. The hidden `package`
command uses the same source-clean typed-region pipeline. Neither command rewrites the original
source files. Project discovery prefers the conventional `src/` import root, then honors explicit
Setuptools, Hatch, Poetry, PDM, or Maturin source-root metadata for layouts such as
`lib/package_name`; undeclared layouts fall back to the project root.

```bash
uv run atoll compile app.ranking
uv run atoll compile
uv run atoll compile --root /path/to/project
```

The default persistent outputs are the wheel in `.atoll/dist/*.whl` plus
`.atoll/compile-report.json` and `.atoll/compile-report.md`. Use `--output` to place generated
wheel artifacts somewhere else. Pass `--keep-install-tree` only when you need to inspect the
temporary install tree for debugging; the report marks that tree as retained.
Atoll first builds the target project's normal PEP 517 wheel from a temporary source-clean copy,
using an isolated build environment that installs the project's declared build requirements. It
then overlays only staged routing code and region-owned native artifacts. Package data, entry
points, and distribution metadata therefore come from the project's build backend. Atoll rewrites
the platform tag and `RECORD`, verifies both the staged payload and final wheel in fresh
interpreters, and removes the copied project and install payload after a successful default run.
For typed regions, the report's `compiled_regions` section names each backend variant, every class
or descriptor-aware binding, its runtime guards and concrete target owner, and the artifact paths
associated with that variant. Each variant also reports `lowering_mode`; an `outlined-block`
variant lists the private synchronous `native_helpers` called by its staged Python suspension
shell. Typed-region entries retain the original generic declaration plus the specialization origin,
substitutions, and concrete type bindings.
Compile report schema v6 includes profile coverage, backend decisions, integer and buffer proofs,
dispatch order, suspension plans, candidate trials, execution-plan and source-optimization
evidence, cache decisions, stage medians, the active optimization policy, and the final accepted
composition. Source optimization remains report-only without configured semantic and
benchmark commands. With both commands, Atoll may emit a patch only after the transformed source
and its normally built PEP 517 wheel pass the hard source-optimization gate. Execution plans are
reported separately from mypyc and Cython typed regions. It lists discovered and rejected plans,
`applied_execution_plans`, three-arm execution-plan trials, staging cache status, payload-file
evidence, marginal speedup over the unplanned payload, and overall speedup over the interpreted
baseline. Until a report lists a plan in `applied_execution_plans` with passing semantic and
benchmark evidence, the plan is discovery evidence only and does not change runtime behavior. The
schema retains all v5 fields plus the legacy v2 compatibility fields `islands` and
`native_readiness`; for source-clean typed-region compile they
are legacy views and normally remain empty with zero counts. Region members also expose ordered
call sites, runtime imports, and suspension points; dependency records state the invocation mode
and whether a dependency must share a native compilation unit. Suspension plans include each
block's source coordinates, live-ins, live-outs, runtime dependencies, work signal, eligibility,
and conservative rejection evidence.

For a precise coroutine, generator, or async-generator binding, Atoll can fall back from a
deterministically rejected whole-callable Cython variant to planner-approved synchronous blocks.
The staged Python shell retains `await`, `yield`, cancellation, exception handlers, and cleanup;
the Cython extension receives explicit live-ins and returns live-outs. Blocks involving unsafe
control flow, cells, nonlocals, nested definitions, comprehensions, deletion, global declarations,
or exception/context boundaries remain interpreted. A native helper failure is surfaced directly;
Atoll never retries the interpreted block after native execution starts. Global and builtin names
are resolved against the source module at each native read, so a call that rebinds a global remains
observable later in the same block. Only bare `@staticmethod` and `@classmethod` descriptors are
accepted; qualified and custom decorators remain interpreted.

Source-clean compile no longer generates Python sidecars or performs generated-AST
native-readiness scoring before backend compilation. Ordinary functions, methods, classes, sync
generators, coroutines, and async generators are eligible when a backend reports that it can lower
the member while preserving the source contract. A class is replaced atomically only when every
method is supported and module loading cannot retain the original class through a decorator,
reassignment, instance, subclass, annotation, default, registry use, source-defined base,
asynchronous method, or class-body side effect. Otherwise Atoll preserves the source class and
compiles eligible methods independently. Cython owns atomic classes because its Python-compatible
class objects can preserve method reflection; mypyc remains preferred for callable members, while
Cython also handles member execution shapes mypyc rejects and deterministic mypyc type failures.
Cython annotation typing and C-type inference are disabled so Python integer and container
semantics are not silently narrowed. Relative imports inside copied callables remain in their
original execution scope and resolve against the source package rather than the private extension
module. Reads, writes, and deletions of omitted same-module state route through the original source
module so native bindings and Python fallbacks share one cache or registry. When profiling
identifies a hot callable, Cython may compile
explicit `Any`, incomplete annotations, or unresolved TypeVars with boxed Python semantics. PEP
695 function type-parameter syntax is removed only from the private generated Cython input; the
public wrapper retains the source function's annotations and type-parameter metadata. Without a
configured benchmark, those boxed candidates remain interpreted.

Generic definitions remain the authoritative Python fallback. Atoll may compile a separate
specialization when every TypeVar closes from a same-module concrete subclass or an unambiguous
same-module call with statically concrete inputs. An inherited specialization is installed only on
the concrete subclass, never on its generic base. The wrapper routes to native code only after
constant-time checks for scalar or nominal classes, `None`, or unions of those; incompatible calls
use the captured Python function. Parameterized containers, generic defaults or variadics,
conflicting call sites, unresolved TypeVars, semantic `Any`, subclass overrides, and dynamic owner
classes are not specialized. Profile-hot boxed callables can still be compiled without pretending
that their types became concrete. When boxed executable code reads a canonical module-level
`TypeVar`, `ParamSpec`, or `TypeVarTuple`, the generated unit reads the original runtime object from
the live source module instead of copying or erasing its declaration.

Pure synchronous functions and static methods with exact `int` inputs can receive guarded Cython
`int32_t` and `int64_t` variants when interval analysis proves every intermediate and return value.
Dispatch checks exact `int` types and decimal-string proof bounds before native entry, then tries
32-bit, 64-bit, a generic compiled target, and Python in deterministic order. `bool`, integer
subclasses, negative or oversized values, opaque calls, external mutation, and unproved arithmetic
fall back before native execution; overflow is never retried.

For an acyclic same-module call chain, Atoll can lower the root plus proven helpers into one Cython
unit with private `cdef inline` calls. Callable and code identities, exact receiver class, and any
direct scalar fields are checked before entry. Atoll-managed helper dispatchers are compared through
their original Python fallback identity so independently composed variants do not disable a valid
chain. Recursion, indirect dispatch, changed helpers, subclasses, and custom attribute behavior
retain normal Python dispatch.

For exact `bytes`, `bytearray`, `memoryview`, and `array.array` parameters, Atoll can lower a
conservatively proved read-only sum, XOR, or conditional-count loop through a Cython typed
memoryview without copying. Dispatch first checks the exact runtime type, one-dimensional
contiguous format and item size, mutability when required, and a constant-time length bound for
overflow-sensitive reductions. The initial `memoryview` and `array.array` specialization accepts
unsigned-byte (`B`) layouts; other formats, strided views, mutation, complex indexing, and mixed
reductions use the original Python function.

Profile-selected methods and functions are lowered as directed slices rooted at one public
binding. Ordinary same-module calls, awaited calls, class construction, and `self` or `cls` method
dispatch remain late-bound Python runtime boundaries unless syntax proves the callee must share the
native unit. This keeps blocked or dynamic callees interpreted without poisoning a hot caller.
Explicitly declared methods on recognized dataclasses may be rebound while preserving the original
class object and descriptor kind.

During source-clean compile, Atoll prints timed progress lines to stderr for discovery, scanning,
staging, cache lookup or restore, backend compilation, wheel writing, verification, and cleanup.
Compile reports include cache status plus subphase timings such as `mypycify`, `cythonize`, and
`build_ext`. Compatible cold Cython misses share one build and use bounded parallel translation and
object compilation; deterministic failures are bisected to individual variants. The representative
cold-build gate requires at least a 20% reduction with artifact parity. A fully warm batch restores
each variant independently and invokes no native compiler.
Duplicate macOS linker `-rpath` warnings are filtered from terminal output; other native compiler
diagnostics are still captured in Atoll's build diagnostics. Atoll keeps strict reusable compile
and mypy cache state under `.atoll/cache/`. Typed artifacts are cached independently by backend and
region under `.atoll/cache/compile/regions/`; deterministic non-transient backend rejections are
stored separately under `.atoll/cache/compile/decisions/`. An unchanged variant restores the
rejection decision or native files without invoking mypyc or Cython again. `atoll clean --cache`
removes all reusable compiler state.
For benchmark-guided builds, Atoll still collects a fresh profile on every invocation. It stores the
first source- and environment-bound native candidate plan under `.atoll/cache/profile-plans/` so
sampling jitter cannot force new compiler work on an unchanged warm build. A replayed plan still
runs the configured semantic, marginal-profitability, and final benchmark gates.
Module-level typing diagnostics, such as unsupported `TypeVar` keyword arguments, remain visible in
scan and compile reports. A callable from such a module is compiled only when the typed-region
analysis and backend capability assessment can still preserve its source behavior.

Atoll v1 source-clean compile targets backend-supported typed regions, including ordinary
functions, eligible classes and methods, async shapes, and narrowly guarded concrete generic
specializations. It still leaves dynamic, unresolved generic, runtime-incomplete, or
identity-sensitive regions in Python and does not treat object-rich orchestration as one native
unit. Large gains are therefore expected only when meaningful application time is spent inside
accepted CPU-bound code; successful compilation is not a speedup claim.

### Profile-guided source optimization

When both `test_command` and `benchmark_command` are configured, `atoll compile` also evaluates
source-to-source optimization for hot async fan-out and fan-in pipelines. It requires at least
10,000 observed work items, zero observed suspension for the fused callable shape, and 70% mapped
hot-path coverage before a source plan can be trialed. Atoll ranks at most two plans and searches at
most eight ordered compositions with beam width two and depth four. An unsafe residual step is
reported and skipped without blocking a later independently proven step.

The source lowerer uses LibCST in a temporary project copy. Its cumulative guarded path can batch
drain a private transport, execute proven quiescent coroutine work in one copied `Context` per
logical item, fuse private producer/transport/consumer state transitions, and auto-forward a
private run-to-completion protocol. Runtime guards validate source, callable, scheduler, stream,
descriptor, and code identities before the first transformed side effect. Suspension, task or
cancellation introspection, context mutation, dynamic scheduling, changed descriptors, tracing,
profiling, and monitoring keep the original path. Optimized work is never retried through Python
after entry. `ATOLL_DISABLE=1` forces the original path; `ATOLL_REQUIRE_OPTIMIZED=1` makes a failed
guard visible to strict tests.

After a transformed candidate passes semantics and the `1.05x` marginal search gate, Atoll profiles
that staged payload again with optimized routing enabled before it can seed another search depth.
The marginal gate uses the median of corresponding current/candidate ratios from the rotating
three-arm samples, so one order-biased current measurement cannot select a different source patch.
Only a completed dynamic profile may advance the beam; unsupported launchers, insufficient samples,
and failed passes remain rejection evidence. For structurally owned AnyIO-on-asyncio streams, later
ordered trials can amortize run guards, collapse quiescent await chains, elide context copies only
with context-independent evidence, count private completions incrementally, and replace private
result records with a fixed projection. The compile report records the fresh residual profile on each
trial.

Atoll can also recognize a private exact-dictionary completion scan whose predicate checks stack-run
and node membership. The source fallback maintains a private count and index at every proven mapping
write and removal, but still materializes the original value snapshot and calls the original
predicate. A transactional Cython variant may replace the cached run guard, snapshot, and indexed
query together. Exact owner and predicate-code identities plus the active-count invariant are checked
before native routing; missing artifacts, changed identities, stale counts, or `ATOLL_DISABLE=1`
retain the source scan.

The promotion floor is `max(3.0, minimum_speedup)`. Both the transformed source tree and a normal
PEP 517 wheel built from that tree must meet the floor over seven alternating benchmark pairs. An
accepted default compile leaves the checkout unchanged and writes the reproducible patch to
`.atoll/patches/<candidate-id>.patch`; rejected candidates may remain only as cache evidence under
`.atoll/cache/source-optimization/`. Below the floor, Atoll emits no source patch.

An accepted source candidate is also an optimization baseline, not a terminal branch. Atoll
recreates that patch in disposable build storage, rescans the transformed project, and may overlay
profitable native regions or execution plans onto its wheel. A later semantic or performance
rejection retains the already accepted source-only wheel. The temporary transformed project is
removed with normal build scratch and never changes the checkout.

Use `--apply-source` only after reviewing the accepted report and patch:

```bash
uv run atoll compile --root . --apply-source
```

Application requires a Git checkout, rejects `--in-place`, stale hashes, and files changed since
profiling, then runs `git apply --check`, applies the exact accepted patch, and reruns tests plus the
full benchmark. A failed post-apply gate reverses the patch. Default `atoll compile` never changes
checkout sources.

### Semantic and performance gates

Source-clean compile is valid without a benchmark, but its report then records performance as
`unbenchmarked`; Atoll does not claim an acceleration. Projects that want wheel promotion to depend
on behavior and measured profitability can configure argv commands in `pyproject.toml`:

```toml
[tool.atoll.compile]
backends = ["mypyc", "cython"]
test_command = ["pytest", "-q"]
benchmark_command = ["python", "benchmarks/atoll_workload.py"]
benchmark_warmups = 1
benchmark_samples = 7
minimum_speedup = 1.10
```

`minimum_speedup` must be greater than `1.0`. Specialized native and execution-plan candidates
must improve the current accepted arm by at least `1.05x`; profile-guided generic regions retain
their `1.01x` exploratory gate. Both compared medians must be at least 0.25 seconds. Source patches
and representative family promotion use the separate `3.0x` hard floor.

Commands run directly with `shell=False`. For `python script.py` and `python -m module` benchmark
commands, Atoll first builds and tests the baseline wheel, then runs unmeasured profile passes before
selecting regions. A 2 ms statistical sampler attributes both project leaf frames and nested
scheduler or library frames to the active project caller. A bounded Python 3.12 monitoring pass
records lifecycle counts and canonical `module.qualname` argument type identities for the hottest
combined activity. Reports never persist argument values or representations, retain at most eight
signatures per member, and mark polymorphic or observation-capped evidence explicitly. Recognized
task-spawn callees are also observed directly. Their evidence includes completed calls, maximum
overlapping active calls, and suspensions before completion.
Atoll requires 100 workload samples, then considers members with at least 20 attributed samples and
2% of the workload. It selects at most four in descending order until they cover 80% of mapped
project activity. Unsupported launchers or insufficient samples use static selection, but still run
the final benchmark gate.

For a supported profile, Atoll compiles the selected candidates once and evaluates them in hotness
order. Each candidate combination runs the semantic command once, then one warmup and three
alternating benchmark pairs compare it with the previously accepted set. A candidate is retained
only when its marginal median speedup is at least `1.01x`. Rejected shims and native artifacts are
removed before final packaging. The report records candidate coverage, lowering mode, fallback
reason, marginal speedup, and the hot-path coverage retained by the accepted set.
If every profiled candidate is rejected, Atoll still records the final full-gate evidence but does
not publish a wheel with zero native regions.

Profile-selected async execution plans are trialed only when both `test_command` and
`benchmark_command` are configured. Discovery is automatic for the built-in `asyncio` and
AnyIO-on-asyncio dialects, and execution plans are evaluated independently of mypyc and Cython typed
regions. Plan staging happens in a disposable copy of the accepted payload, keeps the original
implementation as a guarded fallback, and validates every reported payload change before project
code runs. Atoll runs the semantic command once, then uses one warmup and seven alternating
benchmark trios across interpreted baseline, unplanned compiled payload, and planned payload. A
plan replaces the unplanned payload only when its marginal median speedup is at least `1.05x`.
The trial records a provisional overall ratio, but the configured full benchmark is the sole
`minimum_speedup` gate and removes the wheel when the final payload misses it. Without both commands,
plans remain report-only and the staged wheel keeps the native-region behavior it already had. A
plan-only overlay that contains no native artifacts preserves the baseline wheel's pure tag, such
as `py3-none-any`.

For field-backed AnyIO rendezvous workflows, the task-preserving backend guards the exact task-group
and source coroutine identities, hoists the stable worker name, and calls AnyIO's original
`create_task()` path. This retains the task factory, task objects, handles, context, cancellation,
stream sends, and stream receives; the original `start_soon()` branch remains the fallback. It can
skip cancellation only after a source-hashed worker has made a tail-position terminal handoff on the
plan's private stream. Custom factories, changed workers or task groups, nonterminal sends, sibling
cancellation, changed stream topology, debugging, tracing, and monitoring retain the original path.
A linked hot reducer may replace
`len(inspect.signature(function).parameters)` with an exact code-object parameter count only for an
unwrapped Python function under the original `inspect` implementation. Every other callable uses
the original reflection expression. Cross-module plan members and their complete source hashes are
recorded in schema v6.

Execution-plan staging cache entries live under `.atoll/cache/execution-plans/`. A cache hit restores
the planned payload files but does not skip semantic or profitability gates: profile collection,
the semantic command, three-arm execution-plan trials, and the final benchmark still run before a
plan can be applied. Atoll does not report speedup unless the configured gates pass; failed,
unavailable, or unprofitable plan trials remain report evidence only.

Schema v3 also records deterministic, report-only task-fusion plans for recognized `start_soon`,
`create_task`, and `ensure_future` sites reachable from selected hot roots. A plan is eligible for
research only when one same-module coroutine has at least 20 complete monomorphic observations,
no overlapping invocation, no pre-completion suspension, and no static cancellation,
instrumentation, context-variable, additional concurrency, or unresolved dynamic effect. Rejected
plans list every failed gate. When the normal compiled payload misses its full performance gate,
each eligible plan is staged in a disposable payload and tested through separate baseline,
unfused, and fused semantic and timing arms, including at least `1.05x` over unfused and `1.10x`
overall.
Task fusion remains unavailable in normal compile configuration and is never enabled by default;
`experimental_task_fusion` will not become public unless the pinned hard benchmark satisfies those
gates.

The manual [generic source-optimizer benchmark](benchmarks/source_optimization/README.md) enforces
the copied-context semantic matrix and `3.0x` guarded feasibility floor independently of any target
project. Each arm calibrates independently above the timing floor, and speedup uses median time per
logical workload execution.

The manual [native optimizer benchmark](benchmarks/native_optimization/README.md) builds the generic
fixture cold and warm, verifies a zero-compiler warm cache hit, and applies one warmup plus seven
rotating pairs to mixed scalar, direct call-chain, and standard-buffer workloads. Every family must
independently reach `3.0x`; one fast polynomial cannot stand in for the other families.

Atoll's manual [Pydantic Graph hard benchmark](benchmarks/pydantic_graph/README.md) pins a difficult
async orchestration workload, compiles it twice, and retains cold/warm reports, source hashes, and
patch-cache evidence. It requires at least `3.0x` for both transformed source and the normal wheel,
stable source-plan and patch identities, an unchanged checkout, and a warm patch-cache hit. It is
intentionally separate from normal CI so ordinary correctness checks do not depend on host timing.

The repository-local [multi-repository corpus](benchmarks/corpus/README.md) separately measures
whole-project compatibility across 25 pinned projects and end-to-end performance across 12 reviewed
workloads. It retains unsupported and compatible no-op outcomes, reports accepted-only and
effective-corpus geometric means per platform, and promotes history only through an explicit human
review. Compiler calibration and semantic negative controls are reported independently and never
inflate the real-repository aggregate.

Profiling and candidate-trial durations are excluded from the final performance medians. Atoll then
tests the accepted payload, runs the configured alternating baseline/compiled benchmark pairs, and
compares median durations. A median below 0.25 seconds is rejected as too noisy. Failed tests,
invalid measurements, or speedup below `minimum_speedup` remove the candidate wheel and are recorded
in `.atoll/compile-report.*`; only this full gate promotes the wheel into `.atoll/dist`. Commands run
from a temporary copy that retains project tests and benchmark files but removes importable checkout
modules, so a flat-layout checkout cannot shadow the baseline or compiled payload. On verification
or gate failure, Atoll removes the disposable build tree, install payload, and rejected wheel. The
JSON and Markdown reports retain the command, verification, and performance evidence; no candidate
remains in `.atoll/dist/*.whl`. Runtime safety selection can record failed probes for native variants
that Atoll rejected; those probes remain diagnostic evidence, while the report's overall status reflects
the final reduced payload and promoted wheel.

Compiled functions and methods retain their source name, qualified name, documentation,
annotations, signature, and sync, coroutine, generator, or async-generator shape. Async-generator
wrappers forward `asend`, `athrow`, and `aclose`; method routing preserves normal, static, and class
descriptors on the original source class. Atomic classes preserve their public module, qualified
name, documentation, annotations, constructor signature, bases, subclass behavior, and pickle
identity. `ATOLL_DISABLE=1` retains interpreted routing, and `ATOLL_REQUIRE_COMPILED=1` checks only
bindings promised by the staged wheel. `ATOLL_REQUIRE_OPTIMIZED=1` checks the generated source fast
path when a source patch was accepted.

```bash
uv pip install --force-reinstall .atoll/dist/*.whl
```

Use `--in-place` only when you intentionally want Atoll to modify the checkout with managed shim
blocks marked `BEGIN ATOLL MANAGED`. In-place compile stores enabled islands in `.atoll.toml`,
compiled extensions in `.atoll/artifacts`, and compilation reports in
`.atoll/compilation-report.json` and `.atoll/compilation-report.md`.
Routing verification proves that managed shims import compiled extensions and rebound configured
symbols; it does not prove semantic equivalence. Add `--test` to an in-place compile to run the
target project's pytest suite with `ATOLL_REQUIRE_COMPILED=1`.

```bash
uv run atoll compile app.ranking --in-place
uv run atoll compile app.ranking --in-place --test "pytest tests"
```

Generated Python sidecars in `.atoll/sidecars`, native compiler scratch files, and mypy's
internal mypyc cache in `.atoll/build` are disposable build inputs for explicit in-place
enable/build workflows; successful in-place compile runs remove them and record the cleanup in the
compilation report. If `--test` fails, Atoll leaves the generated build inputs in place for
debugging and marks the report failed. The older hidden `atoll package` command remains available
as a compatibility alias for source-clean typed-region artifacts.

Source-clean build failures print a concise summary, write `.atoll/compile-report.*`, and remove
temporary build and install roots. Run compile commands inside the target project's Python
environment, since native backends use the active interpreter and installed dependencies.

Lower-level commands remain available when you need to inspect or repair one step:

```bash
uv run atoll enable app.ranking --all-candidates
uv run atoll generate --check
uv run atoll build --clean-first
uv run atoll verify --require-compiled
```

Managed shims use `ATOLL_REQUIRE_COMPILED=1`, `ATOLL_DISABLE=1`, and `ATOLL_STRICT=1`.
Use `uv run atoll disable app.ranking` to remove the shim and mark the island disabled.

## Explain, trial, and clean

```bash
uv run atoll explain app.ranking
uv run atoll explain app.ranking::score_user
uv run atoll trial --top 3 --test "pytest tests"
uv run atoll trial --candidate app.ranking::score_user,rank_candidates --test "pytest" --require-compiled
uv run atoll trial --top 3 --test "pytest tests" --benchmark "pytest benchmarks"
uv run atoll clean --all
```

`explain` includes fixed-width proof bounds, direct call-chain helpers, zero-copy buffer plans, and
the specific fallback reason for each rejected specialization. These are static capability facts,
not performance predictions.

Trial mode preserves the scan-candidate selection UX from `--candidate` and `--top`, then compiles
the selected callable closure through the same source-clean package pipeline used by `atoll
compile`. It uses temporary wheel, install, and cache artifacts, keeps only the requested public
function bindings visible, and leaves helper callables private inside the compiled closure.
Configured compile quality gates are ignored for trial; only the optional `--test` and
`--benchmark` commands are run, and both are one-shot pytest-style exit checks. A failing `--test`
stops before the benchmark. Temporary artifacts are removed by default; `--keep-temp` retains the
wheel, install payload, and isolated cache for inspection. Compiled routing is required by default;
use `--allow-interpreted` to permit interpreted fallback during trial commands.
`--allow-python-sidecar` remains as a compatibility alias.

## Project Layout

- `src/atoll/` contains package source.
- `tests/` contains unit and integration tests.
- `docs/` contains MkDocs documentation.
- `examples/` contains runnable examples.
- `AGENTS.md` contains shared coding-agent instructions.

## Development

Atoll uses `scaffold-guard` for repository policy and validation. From an Atoll checkout:

```bash
uv sync --all-groups
scaffold-guard inspect-diff
scaffold-guard validate
```
