# PyIslands V1 Implementation Plan

**Date:** 2026-07-06
**Working name:** `pyislands`
**Goal:** Find clean function-level islands inside messy Python modules, extract them into generated sidecar modules, compile those sidecars with `mypyc`, and make the real application use the compiled symbols through explicit opt-in shims.

---

## 0. One-sentence product definition

`pyislands` is a Python tool that discovers compileable clusters of functions/classes inside normal Python code, generates sidecar modules for those clusters, compiles the sidecars with `mypyc`, patches the original modules with explicit managed shims, and verifies that production imports use the compiled implementations.

---

## 1. Non-negotiable V1 principle

V1 must not stop at analysis or test-only experiments.

The user must be able to run:

```bash
pyislands scan .
pyislands trial --top 5 --test "pytest"
pyislands enable app.ranking --symbols score_user,rank_candidates
pyislands build
PYISLANDS_REQUIRE_COMPILED=1 python -m app
```

At that point, the application should import and call compiled implementations for enabled symbols.

The `trial` command is only a safety gate. The real product is `enable + build + verify`.

---

## 2. Definitions

### 2.1 Source module

A normal Python module owned by the user.

Example:

```text
src/app/ranking.py
```

Python import name:

```text
app.ranking
```

### 2.2 Island

A connected cluster of symbols from a source module that appears safe and useful to compile together.

Examples:

```text
app.ranking::score_user
app.ranking::rank_candidates
app.ranking::normalize_features
```

V1 islands should normally be formed from top-level functions and simple top-level classes. Avoid nested functions/classes in V1.

### 2.3 Poison symbol

A symbol that makes compilation risky or impossible.

Examples:

```python
def debug_dump(obj):
    return getattr(obj, input("field: "))


def load_plugin(name: str):
    return importlib.import_module(name)
```

The tool should not merely say “module failed.” It should identify the poison symbol and its radius.

### 2.4 Sidecar module

A generated Python module containing only the island and its required dependencies.

Example:

```text
src/app/_ranking_pyisland.py
```

Compiled artifact after `mypyc`:

```text
src/app/_ranking_pyisland.cpython-312-darwin.so
```

or on Windows:

```text
src/app/_ranking_pyisland.cp312-win_amd64.pyd
```

### 2.5 Managed shim

A generated block inserted into the original source module that redirects selected symbols to the sidecar implementation.

Example:

```python
# BEGIN PYISLANDS MANAGED: app.ranking
...
# END PYISLANDS MANAGED: app.ranking
```

The shim must be explicit, inspectable, reversible, and safe under strict mode.

### 2.6 Enabled island

An island listed in project configuration and wired into real imports via a managed shim.

### 2.7 Trial island

An island tested in a temporary overlay/build workspace. It should be used by tests/benchmarks during `trial`, but it is not enabled in the real source tree.

---

## 3. V1 goals

V1 should implement the following capabilities:

1. Discover Python modules in a project.
2. Parse modules with `ast`.
3. Build symbol records for top-level functions/classes and simple methods.
4. Detect hard and soft dynamic blockers.
5. Run `mypy` and map diagnostics back to symbols.
6. Build a conservative same-module symbol dependency graph.
7. Find function/class clusters that are likely compileable.
8. Generate sidecar modules for selected clusters.
9. Compile generated sidecars with `mypyc`.
10. Insert/remove managed shims into original modules.
11. Build enabled sidecars in-place so real application imports can use them.
12. Verify that enabled symbols are actually routed to sidecar modules, and optionally that the sidecar module loaded is a compiled extension.
13. Generate JSON and Markdown reports.
14. Cache scan results using file hashes.

---

## 4. V1 non-goals

Do not implement these in V1:

1. A new compiler backend.
2. A Rust implementation.
3. Fully automatic semantic-preserving refactors beyond managed shim insertion/removal.
4. Arbitrary function-level native compilation without sidecar modules.
5. Whole-program compilation.
6. Standalone binaries.
7. Import-hook-based production routing.
8. Formal equivalence proofs.
9. AI-generated patches.
10. Deep framework support for Django/FastAPI/SQLAlchemy/Pydantic-heavy modules.
11. Cross-module island extraction, except via normal imports.
12. Exhaustive Python semantic support.

---

## 5. High-level architecture

```text
normal Python repo
    ↓
project discovery
    ↓
AST scanner
    ↓
symbol table + dependency graph
    ↓
blocker detector + mypy diagnostics
    ↓
island clustering/scoring
    ↓
sidecar generation
    ↓
mypyc build backend
    ↓
managed shim insertion
    ↓
verify compiled routing
    ↓
report + CI integration
```

---

## 6. Repository layout for `pyislands`

Recommended implementation layout:

```text
pyislands/
  __init__.py
  __main__.py
  cli.py

  config.py
  project.py
  paths.py
  logging.py

  models.py
  cache.py
  report.py

  analysis/
    __init__.py
    ast_scanner.py
    symbol_table.py
    blockers.py
    type_readiness.py
    call_graph.py
    clustering.py
    scoring.py

  backends/
    __init__.py
    mypy.py
    mypyc.py

  generation/
    __init__.py
    sidecar.py
    shim.py
    overlay.py

  runtime/
    __init__.py
    verify.py

  commands/
    __init__.py
    scan.py
    explain.py
    trial.py
    enable.py
    disable.py
    generate.py
    build.py
    verify.py
    clean.py

tests/
  fixtures/
    simple_project/
    dynamic_blockers_project/
    strict_mode_project/
  test_scan.py
  test_blockers.py
  test_clustering.py
  test_sidecar_generation.py
  test_shim.py
  test_verify.py
```

Use Python 3.11+ so `tomllib` is available in the standard library.

---

## 7. External tools and dependencies

### 7.1 Required runtime/development dependencies

V1 can be implemented with minimal dependencies:

```text
mypy
setuptools
wheel
```

Optional but convenient:

```text
typer or click      # CLI ergonomics
rich                # readable console output
```

Prefer `argparse` if dependency minimization is more important than CLI polish.

### 7.2 Why `mypyc`

V1 should use `mypyc` as the compiler backend because it already compiles Python modules into C extension modules using standard type hints. The innovation in `pyislands` is not native code generation; it is discovery, extraction, routing, verification, and adoption.

### 7.3 Build toolchain assumption

Users need a working native build environment for `mypyc` extension builds.

Examples:

```text
macOS: Xcode Command Line Tools
Linux: gcc/clang + Python development headers
Windows: Visual Studio Build Tools
```

V1 should detect obvious build failures and report them as environment problems rather than island problems.

---

## 8. CLI specification

### 8.1 `scan`

Analyze project and report candidate islands.

```bash
pyislands scan .
```

Options:

```bash
pyislands scan . --source-root src
pyislands scan . --json .pyislands/report.json
pyislands scan . --markdown .pyislands/report.md
pyislands scan . --no-mypy
pyislands scan . --max-files 1000
```

Outputs:

```text
.pyislands/report.json
.pyislands/report.md
.pyislands/cache/index.json
```

Scan does not modify user source code.

### 8.2 `explain`

Explain why a module or symbol is/is not a candidate.

```bash
pyislands explain app.ranking
pyislands explain app.ranking::score_user
```

Output should include:

```text
- symbol score
- type readiness
- blockers
- dependency edges
- poison radius
- suggested island cluster
```

### 8.3 `trial`

Generate sidecars in a temporary overlay, compile them, and run tests/benchmarks against the compiled versions.

```bash
pyislands trial --top 5 --test "pytest"
```

Options:

```bash
pyislands trial --candidate app.ranking::score_user,rank_candidates --test "pytest"
pyislands trial --top 10 --benchmark "pytest benchmarks/"
pyislands trial --keep-temp
pyislands trial --require-compiled
```

`trial` should actually route imports to sidecars in the temporary workspace. It is not decorative.

### 8.4 `enable`

Enable an island in the real source tree.

```bash
pyislands enable app.ranking --symbols score_user,rank_candidates
```

Effects:

1. Add/update `[tool.pyislands]` config in `pyproject.toml`, or create `.pyislands.toml` if no project config exists.
2. Generate sidecar source file.
3. Insert managed shim into original module.
4. Do not compile unless `--build` is passed.

Options:

```bash
pyislands enable app.ranking --symbols score_user,rank_candidates --build
pyislands enable app.ranking --symbols score_user,rank_candidates --dry-run
pyislands enable app.ranking --symbols score_user,rank_candidates --sidecar app._ranking_pyisland
```

### 8.5 `disable`

Disable an enabled island.

```bash
pyislands disable app.ranking
```

Effects:

1. Remove managed shim block from `app/ranking.py`.
2. Remove or mark disabled config entry.
3. Leave generated sidecar by default unless `--delete-sidecar` is passed.

Options:

```bash
pyislands disable app.ranking --delete-sidecar
pyislands disable app.ranking --dry-run
```

### 8.6 `generate`

Regenerate sidecar files for enabled islands.

```bash
pyislands generate
```

Options:

```bash
pyislands generate --check
pyislands generate --module app.ranking
```

`--check` should fail if generated sidecars are stale.

### 8.7 `build`

Compile enabled sidecars with `mypyc`.

```bash
pyislands build
```

Options:

```bash
pyislands build --module app.ranking
pyislands build --clean-first
pyislands build --inplace
```

Default V1 behavior: build in-place so the original application import path can find compiled extension modules.

### 8.8 `verify`

Verify that enabled symbols are active and optionally compiled.

```bash
pyislands verify
```

Options:

```bash
pyislands verify --require-compiled
pyislands verify --module app.ranking
pyislands verify --import-command "python -m app.healthcheck"
```

The command should check:

```text
- source module imports successfully
- managed shim status says active
- each enabled symbol is rebound
- sidecar module origin exists
- if --require-compiled: sidecar origin ends with a known extension suffix
```

### 8.9 `clean`

Remove build artifacts and caches.

```bash
pyislands clean
```

Options:

```bash
pyislands clean --artifacts
pyislands clean --cache
pyislands clean --all
```

---

## 9. Configuration format

Prefer `pyproject.toml` if available.

Example:

```toml
[tool.pyislands]
backend = "mypyc"
source_roots = ["src"]
build_dir = ".pyislands/build"
cache_dir = ".pyislands/cache"
report_dir = ".pyislands"
require_compiled_env = "PYISLANDS_REQUIRE_COMPILED"
disable_env = "PYISLANDS_DISABLE"
strict_env = "PYISLANDS_STRICT"

[[tool.pyislands.island]]
source_module = "app.ranking"
source_path = "src/app/ranking.py"
sidecar_module = "app._ranking_pyisland"
sidecar_path = "src/app/_ranking_pyisland.py"
symbols = ["score_user", "rank_candidates", "normalize_features"]
mode = "shim"
enabled = true
```

If editing `pyproject.toml` is too risky in V1, create `.pyislands.toml` instead and document that as the first implementation path. Supporting both is fine, but do not spend excessive time on TOML-preserving edits in V1.

Recommended V1 approach:

1. Read `pyproject.toml` if it exists.
2. Read `.pyislands.toml` if it exists.
3. Write `.pyislands.toml` by default.
4. Add `--write-pyproject` later.

---

## 10. Core data model

Create these dataclasses in `models.py`.

### 10.1 `ProjectConfig`

```python
@dataclass(frozen=True)
class ProjectConfig:
    root: Path
    source_roots: tuple[Path, ...]
    backend: Literal["mypyc"]
    build_dir: Path
    cache_dir: Path
    report_dir: Path
    islands: tuple[EnabledIslandConfig, ...]
```

### 10.2 `ModuleId`

```python
@dataclass(frozen=True)
class ModuleId:
    name: str          # app.ranking
    path: Path         # src/app/ranking.py
```

### 10.3 `SymbolId`

```python
@dataclass(frozen=True)
class SymbolId:
    module: str        # app.ranking
    qualname: str      # score_user or Ranker.score

    @property
    def stable_id(self) -> str:
        return f"{self.module}::{self.qualname}"
```

### 10.4 `SymbolRecord`

```python
@dataclass
class SymbolRecord:
    id: SymbolId
    kind: Literal["function", "class", "method"]
    visibility: Literal["public", "private"]
    lineno: int
    end_lineno: int
    col_offset: int
    end_col_offset: int | None
    decorators: tuple[str, ...]
    arg_count: int
    annotated_arg_count: int
    has_return_annotation: bool
    has_any_annotation: bool
    uses_globals: tuple[str, ...]
    local_names: tuple[str, ...]
    referenced_names: tuple[str, ...]
    blockers: tuple["Blocker", ...]
```

### 10.5 `Blocker`

```python
@dataclass(frozen=True)
class Blocker:
    severity: Literal["hard", "soft", "info"]
    code: str
    message: str
    lineno: int | None = None
    symbol: SymbolId | None = None
```

Suggested blocker codes:

```text
DYN_EVAL
DYN_EXEC
DYN_GLOBALS
DYN_LOCALS
DYN_GETATTR_DYNAMIC
DYN_SETATTR
DYN_DELATTR
DYN_IMPORTLIB
DYN_IMPORT_CALL
DYN_CLASS_MONKEYPATCH
DYN_MODULE_MONKEYPATCH
INTROSPECTION_INSPECT
UNTYPED_DEF
UNTYPED_DECORATOR
ANY_PUBLIC_BOUNDARY
MYPY_ERROR
DYNAMIC_GLOBAL_DEP
TOP_LEVEL_SIDE_EFFECT
NESTED_SYMBOL
UNSUPPORTED_DECORATOR
```

### 10.6 `DependencyEdge`

```python
@dataclass(frozen=True)
class DependencyEdge:
    src: SymbolId
    dst: SymbolId | str
    kind: Literal["calls", "uses_global", "inherits", "decorated_by", "imports", "unknown"]
    confidence: Literal["high", "medium", "low"]
    lineno: int | None = None
```

### 10.7 `IslandCandidate`

```python
@dataclass
class IslandCandidate:
    source_module: ModuleId
    symbols: tuple[SymbolId, ...]
    required_imports: tuple[str, ...]
    required_constants: tuple[str, ...]
    required_local_symbols: tuple[SymbolId, ...]
    rejected_symbols: tuple[SymbolId, ...]
    score: int
    risk: Literal["low", "medium", "high"]
    reasons: tuple[str, ...]
```

### 10.8 `CompileAttempt`

```python
@dataclass
class CompileAttempt:
    candidate_id: str
    sidecar_path: Path
    sidecar_module: str
    success: bool
    command: tuple[str, ...]
    stdout: str
    stderr: str
    artifact_paths: tuple[Path, ...]
    duration_seconds: float
```

### 10.9 `VerifyResult`

```python
@dataclass
class VerifyResult:
    source_module: str
    sidecar_module: str
    active: bool
    compiled: bool
    origin: str | None
    symbols: dict[str, bool]
    error: str | None = None
```

---

## 11. Project discovery

Implement in `project.py`.

### 11.1 Source root detection

Order of precedence:

1. CLI `--source-root` values.
2. `[tool.pyislands].source_roots`.
3. Common defaults:
   - `src/` if present.
   - project root if packages are directly under root.

V1 can be conservative and require `--source-root` when ambiguous.

### 11.2 Module name resolution

Given:

```text
root = /repo
source_root = /repo/src
path = /repo/src/app/ranking.py
```

Module name:

```text
app.ranking
```

Rules:

1. Ignore files under `.venv`, `venv`, `.tox`, `.nox`, `build`, `dist`, `.pyislands`, `.mypy_cache`, `site-packages`.
2. Ignore test files by default for island candidates, but include them for dependency/test discovery.
3. Include modules only under configured source roots.

### 11.3 Package validation

V1 should support both traditional packages with `__init__.py` and namespace packages.

If namespace-package resolution is ambiguous, report a warning but continue.

---

## 12. AST scanning

Implement in `analysis/ast_scanner.py`.

Use `ast.parse(source, filename=str(path), type_comments=True)`.

For each module, collect:

```text
- top-level imports
- top-level functions
- top-level classes
- simple methods inside top-level classes
- top-level assignments
- top-level executable statements
- dynamic blocker calls
```

### 12.1 Top-level imports

Record these forms:

```python
import math
import numpy as np
from app.types import User, Event
from __future__ import annotations
```

Store:

```python
@dataclass
class ImportRecord:
    source_text: str
    imported_names: tuple[str, ...]
    module: str | None
    level: int
    lineno: int
    end_lineno: int
```

### 12.2 Top-level assignments

Classify assignments as:

```text
literal_constant      SAFE to copy if referenced
simple_type_alias     SAFE-ish to copy if parseable
runtime_dynamic       BLOCKS if referenced by island
unknown               BLOCKS if referenced by island
```

Safe literal examples:

```python
DEFAULT_LIMIT = 100
FEATURE_NAME = "ranking"
WEIGHTS = (0.2, 0.8)
```

Dynamic examples:

```python
RATE = load_rate_from_config()
CLIENT = boto3.client("s3")
NOW = datetime.now()
```

If an island symbol references a dynamic global, reject the island or require user intervention.

### 12.3 Top-level executable statements

Anything that is not import, function/class definition, simple assignment, or `if TYPE_CHECKING` should be recorded.

Examples:

```python
register_handler(score_user)
print("loading ranking")
configure_global_state()
```

If a selected symbol is used by a top-level executable statement before the shim block would run, mark this as `TOP_LEVEL_SIDE_EFFECT` or `SHIM_LATE_BINDING_RISK`.

### 12.4 Function/method symbol extraction

For each `FunctionDef` or `AsyncFunctionDef`:

1. Record name, qualname, line range.
2. Record decorators as source-ish strings.
3. Count arg annotations.
4. Record return annotation status.
5. Record referenced names.
6. Record `Call` expressions.
7. Detect dynamic blockers.
8. Reject nested functions in V1 unless they are purely local implementation details copied with the parent function.

V1 should avoid async functions unless they are very simple and tests prove they work. Mark async as soft blocker initially.

### 12.5 Class extraction

V1 should be conservative with classes.

Allowed simple classes:

```text
- dataclasses without dynamic decorators beyond @dataclass
- simple classes with typed methods and no metaclass
- classes with __slots__ may be okay but mark medium risk
```

Block or mark high-risk:

```text
- metaclasses
- __getattr__ / __getattribute__ / __setattr__
- dynamic class attribute mutation
- descriptors beyond @property in V1
- class decorators other than allowlisted ones
- runtime registration patterns
```

Implementation simplification: V1 may analyze classes but only auto-enable top-level function islands by default. Add `--include-classes` as experimental.

---

## 13. Blocker detection

Implement in `analysis/blockers.py`.

### 13.1 Hard blockers

Hard blockers should normally exclude a symbol from an island.

Detect calls or constructs involving:

```python
eval(...)
exec(...)
globals(...)
locals(...)
vars(...)
__import__(...)
importlib.import_module(...)
setattr(...)
delattr(...)
```

Detect dynamic `getattr`:

```python
getattr(obj, name)
getattr(obj, payload["field"])
getattr(obj, input())
```

A static literal `getattr(obj, "field")` can be soft, not hard.

Detect monkey-patching:

```python
SomeClass.method = replacement
some_module.function = replacement
```

Detect class/module dictionary mutation:

```python
SomeClass.__dict__[name] = value
module.__dict__[name] = value
```

### 13.2 Soft blockers

Soft blockers reduce score but may still allow trial.

Examples:

```text
- untyped function
- untyped decorator
- Any in public signature
- bare dict/list/set in public signature
- *args/**kwargs without annotations
- dynamic attribute reads
- inspect usage
- framework decorators
- async function
- property-heavy class
```

### 13.3 Informational warnings

Examples:

```text
- private function selected as part of public island dependency
- sidecar imports heavy third-party dependency
- benchmark not found
- tests not configured
```

---

## 14. Mypy integration

Implement in `backends/mypy.py`.

### 14.1 Basic command

Run mypy as a subprocess using the project’s existing config when available.

Suggested command:

```bash
python -m mypy <source_roots> \
  --show-column-numbers \
  --show-error-codes \
  --no-error-summary \
  --no-color-output
```

Do not force `--strict` by default. Respect the project config.

### 14.2 Mapping diagnostics to symbols

Parse diagnostics into:

```python
@dataclass(frozen=True)
class MypyDiagnostic:
    path: Path
    line: int
    column: int | None
    severity: Literal["error", "note"]
    code: str | None
    message: str
```

Map a diagnostic to a symbol if:

```text
symbol.lineno <= diagnostic.line <= symbol.end_lineno
```

If no symbol matches, attach it to the module.

### 14.3 Mypy daemon later

Do not require `dmypy` in first implementation. Add optional support after basic `mypy` subprocess mode works.

---

## 15. Dependency graph and call graph

Implement in `analysis/call_graph.py`.

### 15.1 Same-module high-confidence edges

For V1, build high-confidence edges only when the target is a known symbol in the same module.

Example:

```python
def score_user(...):
    return normalize_features(...)
```

Edge:

```text
app.ranking::score_user --calls/high--> app.ranking::normalize_features
```

### 15.2 Imported boundary edges

If a symbol calls an imported name, record a boundary edge.

Example:

```python
from app.math_utils import sigmoid


def score(...):
    return sigmoid(x)
```

Edge:

```text
app.ranking::score --imports/medium--> app.math_utils.sigmoid
```

V1 should not attempt to extract cross-module islands automatically.

### 15.3 Low-confidence attribute calls

Example:

```python
user.compute_score()
client.send(payload)
```

Record as low-confidence boundary edges unless the receiver can be resolved trivially.

### 15.4 Uses-global edges

If a symbol references a top-level name that is not local, imported, builtin, or selected as a symbol, record a global dependency.

If the global is a safe literal constant, sidecar generation can copy it.

If the global is dynamic, reject the candidate.

---

## 16. Island clustering

Implement in `analysis/clustering.py`.

### 16.1 Candidate seed selection

A symbol can seed an island if:

```text
- top-level function, or allowed simple class
- no hard blockers
- no mypy errors directly inside symbol
- type-readiness score above threshold, or user explicitly targeted it
- line count above tiny threshold, or part of a larger call cluster
```

Initial thresholds:

```text
min_symbol_score = 60
min_cluster_score = 70
min_cluster_lines = 15
```

Keep these configurable.

### 16.2 Cluster expansion

For each seed:

1. Add same-module high-confidence callees if they are clean.
2. Add helper functions even if private.
3. Add simple classes/constants required by the selected functions.
4. Stop expansion at hard blockers.
5. Stop expansion at dynamic globals.
6. Treat imported symbols as boundaries, not cluster members.

### 16.3 Poison radius

For every rejected symbol, compute what it blocks:

```text
poison symbol -> direct callers -> impacted candidates
```

Report example:

```text
load_plugin() blocks rank_all_plugins(), but does not block score_user().
debug_dump() is isolated and should be left interpreted.
```

### 16.4 Scoring

Suggested score components:

```text
+25 no hard blockers
+20 no mypy errors inside cluster
+15 mostly typed public signatures
+10 no dynamic globals
+10 cluster has loops/computation
+10 cluster has direct tests or test coverage signal
+10 low boundary risk

-30 dynamic global dependency
-25 untyped decorator
-20 Any in public boundary
-20 high-risk framework decorator
-15 low-confidence internal calls
-10 very small cluster
```

### 16.5 Categories

Report candidates as:

```text
✅ Proven compiled island        # after compile/test/verify success
🟢 Strong candidate              # scan-only, likely good
🟡 Almost compileable            # local blockers remain
🔴 Bad candidate                 # too dynamic/high-risk
⚪ Compileable but likely useless # too small/no speed expectation
```

---

## 17. Sidecar generation

Implement in `generation/sidecar.py`.

### 17.1 Sidecar naming

For source module:

```text
app.ranking
```

Default sidecar:

```text
app._ranking_pyisland
```

Path:

```text
src/app/_ranking_pyisland.py
```

Avoid one sidecar per tiny function. One sidecar per source module is fine in V1.

### 17.2 Sidecar file structure

Generated sidecar template:

```python
# This file is generated by pyislands. Do not edit manually.
# Source module: app.ranking
# Island symbols: score_user, rank_candidates
# Source hash: <hash>
# Generated at: <timestamp>

from __future__ import annotations

# copied imports
from app.types import User, Event, Score
import math

# copied safe constants
DEFAULT_WEIGHT = 1.5

# copied required helper symbols
def normalize_features(...):
    ...

# exported island symbols
def score_user(...):
    ...


def rank_candidates(...):
    ...

__pyislands_metadata__ = {
    "source_module": "app.ranking",
    "sidecar_module": "app._ranking_pyisland",
    "symbols": ("score_user", "rank_candidates"),
    "source_hash": "<hash>",
}
```

### 17.3 What to copy

Copy:

```text
- `from __future__ import annotations`
- imports needed by referenced names
- selected functions/classes
- selected helper functions/classes
- safe literal constants referenced by selected symbols
- TYPE_CHECKING imports if needed for annotations
```

Do not copy:

```text
- arbitrary top-level executable statements
- dynamic global initializations
- framework registration calls
- unrelated functions/classes
- test-only monkey patches
```

### 17.4 Import copying strategy

V1 simple strategy:

1. Collect all `Name` nodes referenced by selected symbols.
2. Include imports that bind any of those names.
3. Always include `from __future__ import annotations` if present, or add it.
4. Include `typing` imports used in annotations.
5. Do not include unused imports.

If resolution fails, include the import and mark a warning rather than silently producing broken sidecar.

### 17.5 Constants strategy

Copy only safe literal constants.

Safe:

```python
LIMIT = 100
NAME = "ranking"
WEIGHTS = (1.0, 2.0)
OPTIONS = frozenset({"a", "b"})  # optional V1 support
```

Unsafe:

```python
RATE = load_rate()
CLIENT = Client()
NOW = datetime.now()
CACHE = {}
```

If a selected symbol needs an unsafe global, reject candidate with:

```text
DYNAMIC_GLOBAL_DEP
```

### 17.6 Source extraction

Use `lineno` and `end_lineno` from AST to slice original source lines.

Preserve function/class source exactly where possible.

Do not use `ast.unparse` for the first implementation unless necessary, because it rewrites formatting and can create confusing diffs.

### 17.7 Generated source hash

Compute a hash from:

```text
- selected symbol source slices
- copied imports
- copied constants
- pyislands version
- sidecar generation config
```

Store in:

```python
__pyislands_metadata__["source_hash"]
```

`pyislands generate --check` should fail if the expected hash differs from the sidecar metadata.

---

## 18. Managed shim generation

Implement in `generation/shim.py`.

### 18.1 Managed block placement

Default: append block near end of source module.

Reason: the source module defines interpreted functions/classes first, then the shim rebinds selected names to sidecar implementations.

Important limitation: if selected symbols are used in top-level executable statements before the shim, those uses will still see the interpreted version. Detect and warn/block with `SHIM_LATE_BINDING_RISK`.

### 18.2 Shim template

Use importlib so we can inspect module origin and require compiled extensions in production.

```python
# BEGIN PYISLANDS MANAGED: app.ranking
# This block is managed by pyislands. Do not edit manually.
try:
    import importlib as _pyislands_importlib
    import importlib.machinery as _pyislands_machinery
    import os as _pyislands_os

    __pyislands_status__ = {
        "source_module": "app.ranking",
        "sidecar_module": "app._ranking_pyisland",
        "active": False,
        "compiled": False,
        "symbols": ("score_user", "rank_candidates"),
        "origin": None,
        "error": None,
    }

    if _pyislands_os.getenv("PYISLANDS_DISABLE") != "1":
        try:
            _pyislands_mod = _pyislands_importlib.import_module("app._ranking_pyisland")
            _pyislands_origin = getattr(_pyislands_mod, "__file__", "") or ""
            _pyislands_compiled = any(
                _pyislands_origin.endswith(_suffix)
                for _suffix in _pyislands_machinery.EXTENSION_SUFFIXES
            )

            if _pyislands_os.getenv("PYISLANDS_REQUIRE_COMPILED") == "1" and not _pyislands_compiled:
                raise ImportError(
                    "pyislands sidecar app._ranking_pyisland imported, "
                    "but it is not a compiled extension"
                )

            score_user = _pyislands_mod.score_user
            rank_candidates = _pyislands_mod.rank_candidates

            __pyislands_status__.update({
                "active": True,
                "compiled": _pyislands_compiled,
                "origin": _pyislands_origin,
            })
        except ImportError as _pyislands_error:
            __pyislands_status__["error"] = repr(_pyislands_error)
            if (
                _pyislands_os.getenv("PYISLANDS_STRICT") == "1"
                or _pyislands_os.getenv("PYISLANDS_REQUIRE_COMPILED") == "1"
            ):
                raise
finally:
    for _pyislands_name in (
        "_pyislands_importlib",
        "_pyislands_machinery",
        "_pyislands_os",
    ):
        globals().pop(_pyislands_name, None)
# END PYISLANDS MANAGED: app.ranking
```

Notes:

1. In normal development, if sidecar is missing, the module falls back to interpreted definitions.
2. In production, set `PYISLANDS_REQUIRE_COMPILED=1` to fail if the sidecar is not compiled.
3. `PYISLANDS_DISABLE=1` forces interpreted mode.
4. Catch only `ImportError`. Other sidecar import exceptions should usually surface.

### 18.3 Shim update/removal

Find blocks using exact markers:

```text
# BEGIN PYISLANDS MANAGED: <module>
# END PYISLANDS MANAGED: <module>
```

Rules:

1. If no block exists, append one.
2. If one block exists, replace it.
3. If multiple blocks exist, fail and ask user to clean manually.
4. `disable` removes the block.

V1 can do marker-based text replacement. Use LibCST later only if more precise placement is required.

---

## 19. Mypyc backend

Implement in `backends/mypyc.py`.

### 19.1 Generated build script

Generate a temporary build script under `.pyislands/build/build_mypyc.py`.

Example:

```python
from __future__ import annotations

from setuptools import setup
from mypyc.build import mypycify

setup(
    name="pyislands_generated",
    ext_modules=mypycify([
        "src/app/_ranking_pyisland.py",
        "src/app/_pricing_pyisland.py",
    ]),
)
```

Run:

```bash
python .pyislands/build/build_mypyc.py build_ext --inplace
```

### 19.2 Build command implementation

```python
def build_sidecars(paths: list[Path], *, project_root: Path) -> CompileAttempt:
    script = write_build_script(paths)
    cmd = [sys.executable, str(script), "build_ext", "--inplace"]
    result = subprocess.run(
        cmd,
        cwd=project_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    ...
```

### 19.3 Artifact detection

After build, search next to each sidecar path for files matching extension suffixes:

```python
import importlib.machinery

suffixes = importlib.machinery.EXTENSION_SUFFIXES
```

For sidecar:

```text
src/app/_ranking_pyisland.py
```

Look for:

```text
src/app/_ranking_pyisland*.so
src/app/_ranking_pyisland*.pyd
```

Do not assume exact extension suffix.

### 19.4 Failure categories

Classify failures as:

```text
MYPYC_TYPE_ERROR
MYPYC_UNSUPPORTED_FEATURE
NATIVE_BUILD_ENV_ERROR
IMPORT_PATH_ERROR
UNKNOWN_BUILD_ERROR
```

Use stderr/stdout heuristics initially, but keep raw logs in report.

---

## 20. Trial mode design

Implement in `commands/trial.py` and `generation/overlay.py`.

### 20.1 Purpose

Trial should prove that compiled islands can run under the project’s tests without modifying the real source tree.

### 20.2 Approach

Create a temporary overlay workspace:

```text
/tmp/pyislands-trial-abc123/
  src/app/ranking.py              # shimmed temporary copy
  src/app/_ranking_pyisland.py    # generated sidecar
  src/app/_ranking_pyisland*.so   # compiled artifact
```

Then run tests with overlay first on `PYTHONPATH`:

```bash
PYTHONPATH=/tmp/pyislands-trial-abc123/src:/repo/src pytest
```

### 20.3 Overlay copying strategy

V1 can copy only modified modules and generated sidecars into overlay, but Python namespace/path resolution can get tricky.

Simpler V1 strategy:

1. Copy the source root tree into a temp dir, excluding `.venv`, `.mypy_cache`, `.pyislands`, `__pycache__`, `build`, `dist`.
2. Apply shims and sidecars in temp dir.
3. Build sidecars in-place inside temp dir.
4. Run tests with temp source root first.

This is slower but simpler and safer for V1.

### 20.4 Trial success criteria

A trial candidate is successful only if:

```text
- sidecar generated
- mypyc build succeeded
- verify passed in temp workspace
- test command passed
- benchmark command passed if provided
```

If tests pass but `verify --require-compiled` fails, the trial is a failure.

---

## 21. Verify implementation

Implement in `runtime/verify.py` and `commands/verify.py`.

### 21.1 Verification script

Generate/execute Python code like:

```python
import importlib
import importlib.machinery

module = importlib.import_module("app.ranking")
status = getattr(module, "__pyislands_status__", None)
assert status is not None
assert status["active"] is True
assert "score_user" in status["symbols"]

origin = status.get("origin")
compiled = any(origin.endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES)
```

### 21.2 Symbol-level checks

For each enabled symbol:

```python
obj = getattr(module, symbol_name)
assert getattr(obj, "__module__", None) == sidecar_module
```

This is a useful but not perfect check. For compiled functions/classes, `__module__` should generally identify the sidecar module.

Also inspect sidecar module origin:

```python
sidecar = importlib.import_module(sidecar_module)
origin = sidecar.__file__
```

### 21.3 `--require-compiled`

When `--require-compiled` is passed, fail unless sidecar origin ends with a suffix from:

```python
importlib.machinery.EXTENSION_SUFFIXES
```

This prevents accidentally using the generated `.py` sidecar instead of the compiled `.so`/`.pyd`.

---

## 22. Real production usage

### 22.1 Development mode

Developers can run without compiled artifacts:

```bash
python -m app
```

If sidecar is missing, the shim falls back to interpreted definitions.

### 22.2 Production compiled-required mode

Production should run:

```bash
pyislands generate --check
pyislands build
PYISLANDS_REQUIRE_COMPILED=1 python -m app
```

If the compiled sidecar is missing or only the `.py` sidecar is importable, import fails loudly.

### 22.3 CI recommendation

Add a job:

```bash
pyislands generate --check
pyislands build
pyislands verify --require-compiled
pytest
```

Optional benchmark gate:

```bash
pyislands trial --enabled --benchmark "pytest benchmarks/"
```

---

## 23. Report format

Generate both JSON and Markdown.

### 23.1 JSON report

Example:

```json
{
  "version": 1,
  "project_root": "/repo",
  "modules": [
    {
      "module": "app.ranking",
      "path": "src/app/ranking.py",
      "symbols": [
        {
          "qualname": "score_user",
          "kind": "function",
          "score": 91,
          "blockers": [],
          "mypy_errors": []
        },
        {
          "qualname": "debug_dump",
          "kind": "function",
          "score": 10,
          "blockers": [
            {
              "code": "DYN_GETATTR_DYNAMIC",
              "severity": "hard",
              "message": "dynamic getattr blocks typed extraction"
            }
          ]
        }
      ],
      "island_candidates": [
        {
          "symbols": ["score_user", "rank_candidates", "normalize_features"],
          "score": 88,
          "risk": "low",
          "reasons": ["no hard blockers", "typed signatures", "isolated from debug_dump"]
        }
      ]
    }
  ]
}
```

### 23.2 Markdown report

Example:

```markdown
# PyIslands Report

## Summary

- Modules scanned: 42
- Candidate islands: 7
- Strong candidates: 3
- Almost compileable: 4
- Bad candidates: 12

## Strong candidates

### app.ranking

Clean island:

- `score_user`
- `rank_candidates`
- `normalize_features`

Poison symbols left interpreted:

- `debug_dump`: dynamic getattr
- `load_plugin`: dynamic importlib

Recommended command:

```bash
pyislands enable app.ranking --symbols score_user,rank_candidates,normalize_features --build
```
```

---

## 24. Caching and incremental scanning

Implement in `cache.py`.

### 24.1 File hash cache

For each source file:

```json
{
  "path": "src/app/ranking.py",
  "sha256": "...",
  "python_version": "3.12.4",
  "scanner_version": "0.1.0",
  "symbols": [...],
  "imports": [...],
  "assignments": [...],
  "blockers": [...]
}
```

If hash and scanner version match, reuse cached scan result.

### 24.2 Mypy cache

Do not implement deep mypy daemon caching in V1. Let mypy use its own `.mypy_cache`.

### 24.3 Invalidation

Invalidate if:

```text
- file hash changes
- pyislands scanner version changes
- Python minor version changes
- config changes
```

---

## 25. Implementation milestones

### Milestone 1: CLI skeleton and project discovery

Deliverables:

```text
- `pyislands scan .` command exists
- discovers source roots
- resolves module names
- ignores venv/build/cache directories
- writes empty report shell
```

Acceptance:

```bash
pyislands scan tests/fixtures/simple_project
```

Produces:

```text
.pyislands/report.json
.pyislands/report.md
```

### Milestone 2: AST scanner

Deliverables:

```text
- parse Python files
- extract top-level functions/classes
- extract imports
- extract simple constants
- extract referenced names
- extract line ranges
```

Acceptance:

A fixture module with three functions yields three `SymbolRecord`s with correct line ranges.

### Milestone 3: Blocker detector

Deliverables:

```text
- detect eval/exec/globals/locals
- detect importlib.import_module
- detect dynamic getattr/setattr/delattr
- detect untyped defs
- detect untyped decorators
- attach blockers to symbols
```

Acceptance:

A fixture with `debug_dump()` using dynamic `getattr` marks only that function as poisoned, not the whole module.

### Milestone 4: Mypy integration

Deliverables:

```text
- run mypy subprocess
- parse diagnostics
- map diagnostics to symbols by line range
- include diagnostics in report
```

Acceptance:

A fixture with one mypy error inside `bad_func()` attaches the error to `bad_func()`.

### Milestone 5: Dependency graph

Deliverables:

```text
- same-module function call edges
- uses-global edges
- imported boundary edges
- low-confidence attribute call records
```

Acceptance:

`rank_candidates -> score_user -> normalize_features` appears as a connected component.

### Milestone 6: Island clustering/scoring

Deliverables:

```text
- seed clean symbols
- expand to clean same-module helpers
- exclude hard blockers
- compute cluster score/risk
- compute poison radius
```

Acceptance:

Fixture:

```python
def rank_candidates(...): ...
def score_user(...): ...
def normalize_features(...): ...
def debug_dump(obj): return getattr(obj, input())
```

Produces one island with the first three functions and marks `debug_dump` as residue.

### Milestone 7: Sidecar generation

Deliverables:

```text
- generate `_module_pyisland.py`
- copy required imports
- copy safe constants
- copy selected functions/helpers
- write metadata
- hash generated content
```

Acceptance:

Generated sidecar imports successfully as pure Python.

### Milestone 8: Mypyc build backend

Deliverables:

```text
- generate temporary setup script
- run build_ext --inplace
- detect generated .so/.pyd
- classify build success/failure
```

Acceptance:

Generated sidecar compiles with mypyc in fixture project.

### Milestone 9: Managed shim insertion/removal

Deliverables:

```text
- append managed shim
- replace existing managed shim
- remove managed shim
- dry-run diff
```

Acceptance:

After enabling, importing original module routes selected symbols to sidecar module.

### Milestone 10: Verify command

Deliverables:

```text
- import source module
- read `__pyislands_status__`
- verify active symbols
- verify compiled origin when required
```

Acceptance:

`pyislands verify --require-compiled` passes only after `pyislands build`.

### Milestone 11: Trial mode

Deliverables:

```text
- temporary overlay workspace
- sidecar generation in overlay
- shim in overlay
- mypyc build in overlay
- run test command with overlay PYTHONPATH
- keep temp option
```

Acceptance:

`pyislands trial --candidate app.ranking::score_user,rank_candidates --test pytest` runs tests against compiled sidecar and verifies compiled routing.

### Milestone 12: Reports and UX polish

Deliverables:

```text
- good console summary
- Markdown report
- JSON report
- explain command
- actionable recommended commands
```

Acceptance:

A user can understand:

```text
- what can compile
- what cannot compile
- what to enable
- what command to run next
```

---

## 26. Fixture projects for testing

Create several tiny fixture projects.

### 26.1 `simple_project`

```text
simple_project/
  pyproject.toml
  src/app/__init__.py
  src/app/types.py
  src/app/ranking.py
  tests/test_ranking.py
```

`ranking.py`:

```python
from __future__ import annotations

from app.types import Event, Score, User

DEFAULT_WEIGHT = 1.5


def normalize_features(xs: list[float]) -> list[float]:
    total = sum(xs)
    if total == 0:
        return xs
    return [x / total for x in xs]


def score_user(user: User, events: list[Event]) -> Score:
    features = normalize_features([float(len(events)), user.activity])
    return Score(value=sum(features) * DEFAULT_WEIGHT)


def rank_candidates(users: list[User], events: list[Event]) -> list[Score]:
    return [score_user(user, events) for user in users]


def debug_dump(obj):
    return getattr(obj, input("field: "))
```

Expected:

```text
Island: normalize_features, score_user, rank_candidates
Residue: debug_dump
```

### 26.2 `dynamic_global_project`

```python
RATE = load_rate_from_config()


def score(x: float) -> float:
    return x * RATE
```

Expected:

```text
Rejected: DYNAMIC_GLOBAL_DEP
```

### 26.3 `mypy_error_project`

```python
def good(x: int) -> int:
    return x + 1


def bad(x: int) -> str:
    return x + 1
```

Expected:

```text
`bad` gets MYPY_ERROR.
`good` can still be candidate.
```

### 26.4 `strict_mode_project`

Use enabled shim but do not build compiled sidecar.

Expected:

```bash
PYISLANDS_REQUIRE_COMPILED=1 python -c "import app.ranking"
```

fails.

After `pyislands build`, it passes.

---

## 27. Edge cases and explicit V1 behavior

### 27.1 Tiny functions

Do not recommend compiling tiny isolated functions unless user explicitly targets them.

Reason: Python ↔ compiled boundary overhead may erase speedup.

### 27.2 Import-time side effects

If selected symbols are used in top-level executable statements before the shim runs, warn or reject.

### 27.3 Monkey-patching and tests

If tests monkey-patch selected functions, compiled rebinding may alter behavior.

Detect obvious monkey-patching patterns in tests later. In V1, rely on test failure and report warning.

### 27.4 Function identity

Rebinding changes function identity:

```python
app.ranking.score_user.__module__ == "app._ranking_pyisland"
```

This is expected. Document it.

### 27.5 Tracebacks

Tracebacks may point into generated sidecar files or compiled extension frames. Document this.

### 27.6 Decorators

Allowed decorators in V1:

```text
- none
- @staticmethod for methods, experimental
- @classmethod for methods, experimental
- @dataclass for simple classes, experimental
```

Everything else is soft or hard depending on whether the decorator can be resolved and copied.

### 27.7 Relative imports

When copying imports into sidecar, preserve relative imports if the sidecar is in the same package.

Example:

```python
from .types import User
```

should remain valid in `app._ranking_pyisland`.

### 27.8 `if TYPE_CHECKING`

Copy `TYPE_CHECKING` imports if annotations require them. Prefer `from __future__ import annotations` in generated sidecars to reduce runtime annotation import problems.

### 27.9 Generated `.py` sidecar accidentally used

This is why `PYISLANDS_REQUIRE_COMPILED=1` and `verify --require-compiled` must exist.

Without strict compiled requirement, Python may import the generated `.py` sidecar when the extension artifact is absent.

### 27.10 Multiple islands per module

V1 should prefer one sidecar per source module. Multiple islands can be listed in metadata but generated into the same sidecar.

### 27.11 Cross-module helpers

Do not copy helper functions from other modules in V1. Import them normally. This creates a boundary edge.

### 27.12 Classes and `isinstance`

If a class is rebound to a compiled sidecar class, `isinstance` behavior may differ for objects created before rebinding or by the interpreted class. For V1, be conservative with class extraction.

Default auto-enable should focus on functions.

---

## 28. Security and safety model

`pyislands` modifies source files only when the user explicitly runs `enable` or `disable` without `--dry-run`.

Rules:

1. Never modify files during `scan`.
2. Never modify real source files during `trial`.
3. Before `enable`, show a diff unless `--yes` is passed.
4. Managed blocks must be marker-delimited.
5. `disable` must be able to reverse `enable`.
6. Production strict mode must fail loudly if compiled artifacts are missing.
7. Reports must separate predicted compileability from proven compiled success.

---

## 29. CI integration recipes

### 29.1 Basic CI

```yaml
- name: Install dependencies
  run: pip install -e . mypy setuptools wheel

- name: Generate PyIslands sidecars
  run: pyislands generate --check

- name: Build PyIslands compiled extensions
  run: pyislands build

- name: Verify PyIslands compiled routing
  run: pyislands verify --require-compiled

- name: Run tests
  env:
    PYISLANDS_REQUIRE_COMPILED: "1"
  run: pytest
```

### 29.2 Docker build

```dockerfile
RUN pip install -e . mypy setuptools wheel
RUN pyislands generate --check
RUN pyislands build
ENV PYISLANDS_REQUIRE_COMPILED=1
CMD ["python", "-m", "app"]
```

---

## 30. Documentation to write with V1

Create docs:

```text
docs/
  quickstart.md
  concepts.md
  commands.md
  production.md
  limitations.md
  troubleshooting.md
```

### 30.1 Quickstart should show

```bash
pyislands scan .
pyislands trial --top 3 --test "pytest"
pyislands enable app.ranking --symbols score_user,rank_candidates --build
pyislands verify --require-compiled
PYISLANDS_REQUIRE_COMPILED=1 pytest
```

### 30.2 Limitations should explicitly state

```text
- V1 uses mypyc as backend.
- V1 does not compile arbitrary dynamic Python.
- V1 focuses on sidecar modules and explicit shims.
- V1 may not help tiny functions.
- V1 is conservative with classes.
- V1 does not guarantee semantic equivalence; tests/verification are required.
```

---

## 31. Definition of done for V1

V1 is done when all of these are true:

1. `scan` finds function-level islands in a normal Python fixture with dynamic residue.
2. `trial` compiles at least one sidecar with `mypyc` and runs tests against the compiled sidecar.
3. `enable` inserts a managed shim into the real source module.
4. `build` compiles enabled sidecars in-place.
5. `verify --require-compiled` proves real imports route to compiled extension modules.
6. `PYISLANDS_REQUIRE_COMPILED=1 pytest` passes in the fixture project.
7. `disable` removes the shim and returns the module to interpreted mode.
8. Reports identify poison symbols and clean islands separately.
9. Source modifications are explicit, marker-delimited, and reversible.
10. The implementation is in Python and the analysis engine is cleanly separated so it can be replaced by Rust later if necessary.

---

## 32. Suggested first coding-agent prompt

Use this prompt to start implementation:

```text
Implement PyIslands V1 according to `pyislands_v1_implementation_plan.md`.

Start with milestones 1-3 only:
1. CLI skeleton with `scan`.
2. Project discovery and module-name resolution.
3. AST scanner that extracts imports, top-level functions/classes, simple constants, and blocker records.

Do not implement mypyc yet.
Do not implement Rust.
Do not modify source files during scan.
Create fixture projects and tests for the scanner/blocker behavior.
Keep data models in `pyislands/models.py`.
Use Python 3.11+.
```

Then follow-up prompt:

```text
Continue PyIslands V1 with milestones 4-6:
- mypy subprocess integration
- diagnostic-to-symbol mapping
- same-module call graph
- island clustering/scoring
- poison-radius reporting

Do not implement source modification yet.
Add tests for a module where one dynamic function is residue and three clean functions form an island.
```

Then:

```text
Continue PyIslands V1 with milestones 7-10:
- sidecar generation
- mypyc build backend
- managed shim insertion/removal
- verify command

Ensure `PYISLANDS_REQUIRE_COMPILED=1` fails when only the generated `.py` sidecar exists and passes after mypyc build creates an extension artifact.
```

Then:

```text
Finish PyIslands V1 with trial mode, reports, CI docs, disable/clean commands, and end-to-end fixture tests.
The final demo should show a normal module with dynamic residue, enable a clean island, build it, and verify that real imports use compiled code.
```

---

## 33. Official references to consult during implementation

- `mypyc` introduction: https://mypyc.readthedocs.io/en/stable/introduction.html
- `mypyc` getting started/build flow: https://mypyc.readthedocs.io/en/stable/getting_started.html
- Python `ast` documentation: https://docs.python.org/3/library/ast.html
- Python packaging / `pyproject.toml`: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
- mypy daemon docs, for later incremental mode: https://mypy.readthedocs.io/en/stable/mypy_daemon.html
