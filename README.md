# Atoll

Atoll finds typed, CPU-bound regions in Python projects that can be extracted into native
extensions without rewriting the original source tree. It explains why code is accepted or
rejected, compiles eligible functions and methods with mypyc, and preserves interpreted Python as
the fallback.

## Quickstart

```bash
uv run atoll scan .
uv run atoll compile your_package.hot_path
```

Run these commands from the Python project you want to analyze, with Atoll installed in that
project's environment. Scanning writes human-readable and JSON reports under `.atoll/`; compiling
produces a platform wheel under `.atoll/dist/` while leaving the source checkout untouched.

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

Report schema v2 records exact callable annotations, type parameters, class ownership,
descriptor and execution kinds, class fields, and connected typed regions. These regions preserve
source typing for backend assessment and method-level source-clean compilation.

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
source files.

```bash
uv run atoll compile app.ranking
uv run atoll compile
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
associated with that variant. Typed-region entries retain the original generic declaration plus the
specialization origin, substitutions, and concrete type bindings.
Report schema v2 still includes the compatibility fields `islands` and `native_readiness`; for
source-clean typed-region compile they are legacy views and normally remain empty with zero counts.

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
semantics are not silently narrowed. Special methods other than a closed class constructor,
unresolved generics, runtime-incomplete regions, and explicit `Any` remain interpreted.

Generic definitions remain the authoritative Python fallback. Atoll may compile a separate
specialization when every TypeVar closes from a same-module concrete subclass or an unambiguous
same-module call with statically concrete inputs. An inherited specialization is installed only on
the concrete subclass, never on its generic base. The wrapper routes to native code only after
constant-time checks for scalar or nominal classes, `None`, or unions of those; incompatible calls
use the captured Python function. Parameterized containers, generic defaults or variadics,
conflicting call sites, unresolved TypeVars, semantic `Any`, subclass overrides, and dynamic owner
classes remain interpreted.

During source-clean compile, Atoll prints timed progress lines to stderr for discovery, scanning,
staging, cache lookup or restore, backend compilation, wheel writing, verification, and cleanup.
Compile reports include cache status plus subphase timings such as `mypycify` and `build_ext`.
Duplicate macOS linker `-rpath` warnings are filtered from terminal output; other native compiler
diagnostics are still captured in Atoll's build diagnostics. Atoll keeps strict reusable compile
and mypy cache state under `.atoll/cache/`. Typed artifacts are cached independently by backend and
region under `.atoll/cache/compile/regions/`; unchanged variants restore their native files without
invoking mypyc or Cython again. `atoll clean --cache` removes all reusable compiler state.
Module-level typing diagnostics, such as unsupported `TypeVar` keyword arguments, remain visible in
scan and compile reports. A callable from such a module is compiled only when the typed-region
analysis and backend capability assessment can still preserve its source behavior.

Atoll v1 source-clean compile targets backend-supported typed regions, including ordinary
functions, eligible classes and methods, async shapes, and narrowly guarded concrete generic
specializations. It still leaves dynamic, unresolved generic, runtime-incomplete, or
identity-sensitive regions in Python and does not treat object-rich orchestration as one native
unit. Large gains are therefore expected only when meaningful application time is spent inside
accepted CPU-bound code; successful compilation is not a speedup claim.

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

Commands run directly with `shell=False`. When a benchmark is configured, Atoll tests both the
baseline wheel payload and compiled payload, runs alternating baseline/compiled benchmark pairs,
and compares median durations. A median below 0.25 seconds is rejected as too noisy. Failed tests,
invalid measurements, or speedup below `minimum_speedup` remove the candidate wheel and are recorded
in `.atoll/compile-report.*`; only a passing gate promotes the wheel. Commands run from a temporary
copy that retains project tests and benchmark files but removes importable checkout modules, so a
flat-layout checkout cannot shadow the baseline or compiled payload. On verification or gate
failure, Atoll keeps `.atoll/dist/install` and moves the rejected wheel under
`.atoll/dist/build/diagnostics/` for inspection; no candidate remains in `.atoll/dist/*.whl`.

Compiled functions and methods retain their source name, qualified name, documentation,
annotations, signature, and sync, coroutine, generator, or async-generator shape. Async-generator
wrappers forward `asend`, `athrow`, and `aclose`; method routing preserves normal, static, and class
descriptors on the original source class. Atomic classes preserve their public module, qualified
name, documentation, annotations, constructor signature, bases, subclass behavior, and pickle
identity. `ATOLL_DISABLE=1` retains interpreted routing, and `ATOLL_REQUIRE_COMPILED=1` checks only
bindings promised by the staged wheel.

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

Source-clean build failures print a concise summary, write `.atoll/compile-report.*`, and list any
retained diagnostic scratch path in the report. Run compile commands inside the target project's
Python environment, since native backends use the active interpreter and installed dependencies.

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
