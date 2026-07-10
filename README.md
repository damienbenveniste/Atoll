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
poison-residue reporting, JSON/Markdown reports, generated sidecars, managed shims,
mypyc builds, and runtime verification.

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
diagnostics. The existing `build_sidecars` facade remains available for legacy compile, build, and
trial workflows.

Use `--no-mypy` to skip mypy diagnostics during a scan.

Candidate scores are 0-100 scan-only heuristics for extraction safety, not predicted speed.
Candidate risk is also extraction risk: `low` means Atoll only saw high-confidence internal
dependencies. Source-clean compile applies a separate native-readiness gate to the generated
code before invoking mypyc. Frame-introspection code such as `inspect.currentframe()`,
`sys._getframe()`, and direct frame attributes such as `f_locals` are hard blockers because
mypyc changes Python frame semantics.

## Compile candidates

Use `compile` for the normal workflow. It copies the target package into a temporary Atoll build
area, inserts shims only in that copy, compiles generated function islands or typed method regions,
writes a platform wheel, and removes the temporary install tree. The original source files are left
untouched.

```bash
uv run atoll compile app.ranking
uv run atoll compile
```

The default persistent outputs are the wheel in `.atoll/dist/*.whl` plus
`.atoll/compile-report.json` and `.atoll/compile-report.md`. Use `--output` to place generated
wheel artifacts somewhere else. Pass `--keep-install-tree` only when you need to inspect the
temporary install tree for debugging; the report marks that tree as retained.
For typed methods, the report's `compiled_regions` section names each backend variant, every
descriptor-aware binding, and the artifact paths associated with that variant.

Before mypyc runs, Atoll regenerates each scan candidate independently and retains only leaf
kernels whose complete generated function set has concrete builtin annotations and repeated
primitive work. Generated functions containing `Any`, erased or boxed project types, runtime
`getattr` dependencies, or only trivial delegation are rejected. The compile report lists every
accepted and rejected scan candidate with its native-readiness evidence. If no performance-worthy
kernel remains, compile fails before mypyc and removes any stale wheel for the same package and
platform tag.

When a selected module has no accepted top-level function island, source-clean compile can lower
concretely typed instance methods, static methods, class methods, generators, coroutines, and async
generators from a safe owner class. Backend selection is automatic per member: mypyc is preferred,
while Cython handles execution shapes mypyc rejects and deterministic mypyc type failures. Cython
annotation typing and C-type inference are disabled so Python integer and container semantics are
not silently narrowed. Dynamic owner classes, dunder methods, unresolved generics, and explicit
`Any` remain interpreted.

During source-clean compile, Atoll prints timed progress lines to stderr for discovery, scanning,
staging, cache lookup or restore, mypyc batch or retry builds, wheel writing, and cleanup. Compile
reports include cache status plus subphase timings such as `mypycify` and `build_ext`. Duplicate
macOS linker `-rpath` warnings are filtered from terminal output; other native compiler diagnostics
are still captured in Atoll's build diagnostics. Atoll keeps strict reusable compile and mypy cache
state under `.atoll/cache/`; unchanged legacy function-island builds restore successful artifacts
and cached skips without invoking mypyc again. `atoll clean --cache` removes that state.
When compiling a whole project, Atoll retries modules individually if the batch mypyc build fails
and skips islands that cannot be compiled; if none compile, the command fails with a representative
mypyc diagnostic. Module-level typing diagnostics, such as unsupported `TypeVar` keyword
arguments, remain visible in scan and compile reports. A function from such a module is compiled
only when its generated kernel still preserves concrete native types.

Atoll v1 source-clean compile targets top-level typed leaf kernels and safe typed methods. It does
not yet replace whole classes or treat object-rich orchestration as one native unit. Large gains are
therefore expected only when meaningful application time is spent inside accepted CPU-bound code;
successful compilation is not a speedup claim.

Compiled exports retain the source function or method's name, qualified name, documentation,
annotations, signature, and sync, coroutine, generator, or async-generator shape. Async-generator
wrappers forward `asend`, `athrow`, and `aclose`; method routing preserves normal, static, and class
descriptors on the original source class. `ATOLL_DISABLE=1` retains interpreted routing, and
`ATOLL_REQUIRE_COMPILED=1` checks only bindings promised by the staged wheel.

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
internal mypyc cache in `.atoll/build` are disposable build inputs; successful in-place compile
runs remove them and record the cleanup in the compilation report. If `--test` fails, Atoll leaves
the generated build inputs in place for debugging and marks the report failed. The older
`atoll package` command remains available as a compatibility alias for source-clean artifacts.

Source-clean build failures print a concise summary, write `.atoll/compile-report.*`, and list any
retained diagnostic scratch path in the report. Run compile commands inside the target project's
Python environment, since mypyc imports and type-checks the generated sidecars with the active
interpreter and installed dependencies.

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

Trial mode copies source roots into a temporary overlay, inserts Atoll shims there, builds compiled
sidecars, verifies compiled routing, and then runs supported pytest commands with the overlay first
on `PYTHONPATH`. Compiled routing is required by default; use `--allow-python-sidecar` only when
debugging pure-Python sidecar behavior.

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
