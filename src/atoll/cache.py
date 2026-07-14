"""File-hash cache for Atoll's first-pass AST scans.

The cache stores only deterministic scanner facts derived from source files. It
is invalidated by file hash, Python version, module name, and scanner version so
later enrichment can safely recompute mypy diagnostics and candidate analysis on
top of cached AST records.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Literal, TypedDict, cast

from atoll.analysis.ast_scanner import scan_module
from atoll.models import (
    BindingKind,
    Blocker,
    BlockerSeverity,
    CallSiteFact,
    ConstantKind,
    ConstantRecord,
    ExecutionKind,
    FieldRecord,
    ImportRecord,
    InvocationMode,
    ModuleId,
    ModuleScan,
    ParameterKind,
    ParameterRecord,
    ProjectConfig,
    SuspensionKind,
    SuspensionPoint,
    SymbolId,
    SymbolKind,
    SymbolRecord,
    TypeParameterKind,
    TypeParameterRecord,
    Visibility,
)

SCANNER_VERSION = "7"


class CacheStats(TypedDict):
    """Hit and miss counts returned with a cached scan run.

    Attributes:
        hits: Number of module scans restored from cache.
        misses: Number of module scans recomputed from source.
    """

    hits: int
    misses: int


class BlockerCacheEntry(TypedDict):
    """JSON-safe representation of a `Blocker`.

    The symbol identity is split into nullable module and qualname fields because
    module-level blockers do not belong to a concrete symbol.

    Attributes:
        severity: Diagnostic severity used for filtering and reporting.
        code: Stable machine-readable diagnostic or blocker code.
        message: Human-readable diagnostic or blocker explanation.
        lineno: One-based first source line covered by the record.
        symbol_module: Module portion of an optional cached symbol identity.
        symbol_qualname: Qualified-name portion of an optional cached symbol identity.
    """

    severity: BlockerSeverity
    code: str
    message: str
    lineno: int | None
    symbol_module: str | None
    symbol_qualname: str | None


class ImportCacheEntry(TypedDict):
    """Cached top-level import record used for dependency and sidecar analysis.

    Attributes:
        source_text: Exact source text retained for analysis or generation.
        imported_names: Names introduced into the module namespace by the import.
        module: Imported module path, or `None` for imports without one.
        level: Relative import level; zero denotes an absolute import.
        lineno: One-based first source line covered by the record.
        end_lineno: One-based final source line covered by the record.
    """

    source_text: str
    imported_names: list[str]
    module: str | None
    level: int
    lineno: int
    end_lineno: int


class ConstantCacheEntry(TypedDict):
    """Cached top-level assignment record and its literal-safety classification.

    Attributes:
        name: Top-level assignment name.
        kind: Literal, runtime-dynamic, or unknown safety classification.
        source_text: Exact source text retained for analysis or generation.
        lineno: One-based first source line covered by the record.
        end_lineno: One-based final source line covered by the record.
    """

    name: str
    kind: ConstantKind
    source_text: str
    lineno: int
    end_lineno: int


class ParameterCacheEntry(TypedDict):
    """Cached exact source parameter facts for typed-region planning.

    Attributes:
        name: Source parameter name without `*` or `**` prefixes.
        kind: Positional, variadic, keyword-only, or keyword variadic kind.
        annotation: Exact source annotation text.
        default_source: Exact default-value source text, or `None` when required.
    """

    name: str
    kind: ParameterKind
    annotation: str | None
    default_source: str | None


class FieldCacheEntry(TypedDict):
    """Cached typed class field facts for class-region planning.

    Attributes:
        name: Class field name.
        annotation: Exact source annotation text.
        default_source: Exact default-value source text, or `None` when required.
        class_variable: Whether the field is declared as a class variable.
    """

    name: str
    annotation: str
    default_source: str | None
    class_variable: bool


class TypeParameterCacheEntry(TypedDict):
    """Cached exact type-parameter declaration and structured identity.

    Attributes:
        name: Type parameter name visible in source.
        kind: `TypeVar`, `ParamSpec`, or `TypeVarTuple` classification.
        declaration: Exact source declaration for the type parameter.
    """

    name: str
    kind: TypeParameterKind
    declaration: str


class CallSiteCacheEntry(TypedDict):
    """Cached ordered call-site evidence used by directed region planning.

    Attributes:
        target: Source-level call target expression.
        root_name: First lexical name in the target expression.
        invocation_mode: Ordinary, awaited, or async-iteration call mode.
        lineno: One-based first source line covered by the call.
        end_lineno: One-based final source line covered by the call.
        col_offset: Zero-based first source column covered by the call.
        end_col_offset: Zero-based final source column, when available.
        requires_same_unit: Whether the call must share a native compilation unit.
    """

    target: str
    root_name: str
    invocation_mode: InvocationMode
    lineno: int
    end_lineno: int
    col_offset: int
    end_col_offset: int | None
    requires_same_unit: bool


class SuspensionPointCacheEntry(TypedDict):
    """Cached source location for one coroutine or generator suspension.

    Attributes:
        kind: Await, yield, async-loop, or async-context suspension kind.
        lineno: One-based first source line covered by the suspension.
        end_lineno: One-based final source line covered by the suspension.
        col_offset: Zero-based first source column covered by the suspension.
        end_col_offset: Zero-based final source column, when available.
    """

    kind: SuspensionKind
    lineno: int
    end_lineno: int
    col_offset: int
    end_col_offset: int | None


class SymbolCacheEntry(TypedDict):
    """Cached AST facts for a function, class, or simple method.

    The payload intentionally excludes mypy diagnostics and candidate data
    because those are enrichment outputs that can change without the source file
    itself changing.

    Attributes:
        module: Importable module containing the cached declaration.
        qualname: Module-local qualified symbol name.
        kind: Function, class, or method declaration kind.
        visibility: Public or private source visibility.
        lineno: One-based first source line covered by the record.
        end_lineno: One-based final source line covered by the record.
        col_offset: Zero-based source column where the declaration starts.
        end_col_offset: Zero-based source column where the declaration ends, when available.
        decorators: Source text for decorators applied to the symbol.
        arg_count: Total caller-visible parameter count.
        annotated_arg_count: Number of parameters with explicit annotations.
        has_return_annotation: Whether the callable declares a return annotation.
        has_any_annotation: Whether any visible annotation contains `Any`.
        called_names: Simple names observed in call position.
        uses_globals: Module globals read by the symbol body.
        local_names: Names bound locally within the symbol body.
        referenced_names: All names read by the symbol body or annotations.
        owner_class: Source owner class for a method binding, when applicable.
        binding_kind: Runtime descriptor or module binding classification.
        execution_kind: Synchronous, generator, coroutine, async-generator, or class shape.
        type_parameters: Type parameter names declared directly by the symbol.
        parameters: Exact source parameter declarations in call order.
        return_annotation: Exact source return annotation, when present.
        annotation_names: Names referenced by source annotations.
        called_paths: Dotted call targets recovered from source syntax.
        call_sites: Ordered source call facts retained for directed slicing.
        suspension_points: Ordered coroutine and generator suspension boundaries.
        runtime_imports: Function-local imports executed by the declaration.
        base_names: Base-class expressions referenced by the declaration.
        fields: Typed class fields retained for region planning.
        declaration_start_lineno: First line of decorators or declaration syntax.
        scope_type_parameters: Type parameter names inherited from enclosing scopes.
        type_parameter_records: Structured type parameters declared directly by the symbol.
        scope_type_parameter_records: Structured type parameters inherited from enclosing scopes.
        any_annotation_sources: Locations where `Any` enters the symbol's type surface.
        blockers: Conservative blockers attached to this module or symbol.
    """

    module: str
    qualname: str
    kind: SymbolKind
    visibility: Visibility
    lineno: int
    end_lineno: int
    col_offset: int
    end_col_offset: int | None
    decorators: list[str]
    arg_count: int
    annotated_arg_count: int
    has_return_annotation: bool
    has_any_annotation: bool
    called_names: list[str]
    uses_globals: list[str]
    local_names: list[str]
    referenced_names: list[str]
    owner_class: str | None
    binding_kind: BindingKind
    execution_kind: ExecutionKind
    type_parameters: list[str]
    parameters: list[ParameterCacheEntry]
    return_annotation: str | None
    annotation_names: list[str]
    called_paths: list[str]
    call_sites: list[CallSiteCacheEntry]
    suspension_points: list[SuspensionPointCacheEntry]
    runtime_imports: list[ImportCacheEntry]
    base_names: list[str]
    fields: list[FieldCacheEntry]
    declaration_start_lineno: int | None
    scope_type_parameters: list[str]
    type_parameter_records: list[TypeParameterCacheEntry]
    scope_type_parameter_records: list[TypeParameterCacheEntry]
    any_annotation_sources: list[str]
    blockers: list[BlockerCacheEntry]


class ModuleScanCacheEntry(TypedDict):
    """Cached first-pass scan for one module before enrichment.

    Attributes:
        module_name: Importable module name used to restrict the command.
        path: Absolute source path serialized for cache reconstruction.
        imports: Top-level imports retained for analysis or generation.
        constants: Top-level constants retained for analysis or sidecar generation.
        symbols: Cached declaration facts in source order.
        blockers: Conservative blockers attached to this module or symbol.
        top_level_statement_lines: Executable module-level statements that may affect extraction.
    """

    module_name: str
    path: str
    imports: list[ImportCacheEntry]
    constants: list[ConstantCacheEntry]
    symbols: list[SymbolCacheEntry]
    blockers: list[BlockerCacheEntry]
    top_level_statement_lines: list[int]


class FileCacheEntry(TypedDict):
    """One indexed source file and the scanner inputs that validate its cache.

    Attributes:
        path: Normalized source path used as the cache-index key.
        module_name: Importable module name used to restrict the command.
        sha256: Source content digest used for cache invalidation.
        python_version: Python version included in cache invalidation.
        scanner_version: Scanner schema version included in cache invalidation.
        scan: Cached module scan payload.
    """

    path: str
    module_name: str
    sha256: str
    python_version: str
    scanner_version: str
    scan: ModuleScanCacheEntry


class CacheIndex(TypedDict):
    """Root cache file mapping relative source paths to scan entries.

    Attributes:
        version: Schema or cache format version.
        files: Cache entries keyed by normalized source path.
    """

    version: Literal[1]
    files: dict[str, FileCacheEntry]


def scan_modules_with_cache(
    config: ProjectConfig,
    modules: tuple[ModuleId, ...],
) -> tuple[tuple[ModuleScan, ...], CacheStats]:
    """Return first-pass scans, reusing cached AST facts when inputs match.

    The function updates the cache index after scanning misses. It does not cache
    mypy diagnostics, dependency edges, candidate scores, or reports; callers
    must run those enrichment phases on the returned scans every time.

    Args:
        config: Resolved configuration governing the requested operation.
        modules: Discovered or scanned modules processed in deterministic order.

    Returns:
        tuple[tuple[ModuleScan, ...], CacheStats]: Module scans in input order together with cache
            hit and miss counts.
    """
    index = _read_index(config.cache_dir / "index.json")
    files = dict(index["files"])
    scans: list[ModuleScan] = []
    hits = 0
    misses = 0
    for module in modules:
        cache_key = _cache_key(config.root, module.path)
        digest = _file_hash(module.path)
        entry = files.get(cache_key)
        cached = _cached_scan(entry, module, digest, config.root)
        if cached is None:
            scan = scan_module(module)
            files[cache_key] = _file_entry(config.root, scan, digest)
            misses += 1
        else:
            scan = cached
            hits += 1
        scans.append(scan)
    _write_index(config.cache_dir / "index.json", {"version": 1, "files": files})
    return tuple(scans), {"hits": hits, "misses": misses}


def clear_scan_cache(root: Path) -> None:
    """Remove the file-hash scan cache for `root` if it exists.

    This is a best-effort cleanup helper used by commands that want a fresh AST
    scan. Missing cache files are treated as already clean.

    Args:
        root: Root directory of the target Python project.
    """
    index = root.resolve() / ".atoll" / "cache" / "index.json"
    if index.exists():
        index.unlink()


def _cached_scan(
    entry: FileCacheEntry | None,
    module: ModuleId,
    digest: str,
    root: Path,
) -> ModuleScan | None:
    if entry is None:
        return None
    if (
        entry["sha256"] != digest
        or entry["module_name"] != module.name
        or entry["python_version"] != _python_version()
        or entry["scanner_version"] != SCANNER_VERSION
    ):
        return None
    return _module_scan_from_cache(entry["scan"], root)


def _file_entry(root: Path, scan: ModuleScan, digest: str) -> FileCacheEntry:
    return {
        "path": _cache_key(root, scan.module.path),
        "module_name": scan.module.name,
        "sha256": digest,
        "python_version": _python_version(),
        "scanner_version": SCANNER_VERSION,
        "scan": _module_scan_to_cache(root, scan),
    }


def _module_scan_to_cache(root: Path, scan: ModuleScan) -> ModuleScanCacheEntry:
    return {
        "module_name": scan.module.name,
        "path": _cache_key(root, scan.module.path),
        "imports": [
            {
                "source_text": record.source_text,
                "imported_names": list(record.imported_names),
                "module": record.module,
                "level": record.level,
                "lineno": record.lineno,
                "end_lineno": record.end_lineno,
            }
            for record in scan.imports
        ],
        "constants": [
            {
                "name": record.name,
                "kind": record.kind,
                "source_text": record.source_text,
                "lineno": record.lineno,
                "end_lineno": record.end_lineno,
            }
            for record in scan.constants
        ],
        "symbols": [_symbol_to_cache(symbol) for symbol in scan.symbols],
        "blockers": [_blocker_to_cache(blocker) for blocker in scan.blockers],
        "top_level_statement_lines": list(scan.top_level_statement_lines),
    }


def _module_scan_from_cache(entry: ModuleScanCacheEntry, root: Path) -> ModuleScan:
    return ModuleScan(
        module=ModuleId(name=entry["module_name"], path=_cached_path(root, entry["path"])),
        imports=tuple(_import_from_cache(record) for record in entry["imports"]),
        constants=tuple(_constant_from_cache(record) for record in entry["constants"]),
        symbols=tuple(_symbol_from_cache(record) for record in entry["symbols"]),
        blockers=tuple(_blocker_from_cache(record) for record in entry["blockers"]),
        top_level_statement_lines=tuple(entry["top_level_statement_lines"]),
    )


def _symbol_to_cache(symbol: SymbolRecord) -> SymbolCacheEntry:
    return {
        "module": symbol.id.module,
        "qualname": symbol.id.qualname,
        "kind": symbol.kind,
        "visibility": symbol.visibility,
        "lineno": symbol.lineno,
        "end_lineno": symbol.end_lineno,
        "col_offset": symbol.col_offset,
        "end_col_offset": symbol.end_col_offset,
        "decorators": list(symbol.decorators),
        "arg_count": symbol.arg_count,
        "annotated_arg_count": symbol.annotated_arg_count,
        "has_return_annotation": symbol.has_return_annotation,
        "has_any_annotation": symbol.has_any_annotation,
        "called_names": list(symbol.called_names),
        "uses_globals": list(symbol.uses_globals),
        "local_names": list(symbol.local_names),
        "referenced_names": list(symbol.referenced_names),
        "owner_class": symbol.owner_class,
        "binding_kind": symbol.binding_kind,
        "execution_kind": symbol.execution_kind,
        "type_parameters": list(symbol.type_parameters),
        "parameters": [
            {
                "name": parameter.name,
                "kind": parameter.kind,
                "annotation": parameter.annotation,
                "default_source": parameter.default_source,
            }
            for parameter in symbol.parameters
        ],
        "return_annotation": symbol.return_annotation,
        "annotation_names": list(symbol.annotation_names),
        "called_paths": list(symbol.called_paths),
        "call_sites": [
            {
                "target": call.target,
                "root_name": call.root_name,
                "invocation_mode": call.invocation_mode,
                "lineno": call.lineno,
                "end_lineno": call.end_lineno,
                "col_offset": call.col_offset,
                "end_col_offset": call.end_col_offset,
                "requires_same_unit": call.requires_same_unit,
            }
            for call in symbol.call_sites
        ],
        "suspension_points": [
            {
                "kind": point.kind,
                "lineno": point.lineno,
                "end_lineno": point.end_lineno,
                "col_offset": point.col_offset,
                "end_col_offset": point.end_col_offset,
            }
            for point in symbol.suspension_points
        ],
        "runtime_imports": [
            {
                "source_text": record.source_text,
                "imported_names": list(record.imported_names),
                "module": record.module,
                "level": record.level,
                "lineno": record.lineno,
                "end_lineno": record.end_lineno,
            }
            for record in symbol.runtime_imports
        ],
        "base_names": list(symbol.base_names),
        "fields": [
            {
                "name": field.name,
                "annotation": field.annotation,
                "default_source": field.default_source,
                "class_variable": field.class_variable,
            }
            for field in symbol.fields
        ],
        "declaration_start_lineno": symbol.declaration_start_lineno,
        "scope_type_parameters": list(symbol.scope_type_parameters),
        "type_parameter_records": [
            _type_parameter_to_cache(record) for record in symbol.type_parameter_records
        ],
        "scope_type_parameter_records": [
            _type_parameter_to_cache(record) for record in symbol.scope_type_parameter_records
        ],
        "any_annotation_sources": list(symbol.any_annotation_sources),
        "blockers": [_blocker_to_cache(blocker) for blocker in symbol.blockers],
    }


def _symbol_from_cache(entry: SymbolCacheEntry) -> SymbolRecord:
    return SymbolRecord(
        id=SymbolId(module=entry["module"], qualname=entry["qualname"]),
        kind=entry["kind"],
        visibility=entry["visibility"],
        lineno=entry["lineno"],
        end_lineno=entry["end_lineno"],
        col_offset=entry["col_offset"],
        end_col_offset=entry["end_col_offset"],
        decorators=tuple(entry["decorators"]),
        arg_count=entry["arg_count"],
        annotated_arg_count=entry["annotated_arg_count"],
        has_return_annotation=entry["has_return_annotation"],
        has_any_annotation=entry["has_any_annotation"],
        called_names=tuple(entry["called_names"]),
        uses_globals=tuple(entry["uses_globals"]),
        local_names=tuple(entry["local_names"]),
        referenced_names=tuple(entry["referenced_names"]),
        blockers=tuple(_blocker_from_cache(blocker) for blocker in entry["blockers"]),
        owner_class=entry["owner_class"],
        binding_kind=entry["binding_kind"],
        execution_kind=entry["execution_kind"],
        type_parameters=tuple(entry["type_parameters"]),
        parameters=tuple(
            ParameterRecord(
                name=parameter["name"],
                kind=parameter["kind"],
                annotation=parameter["annotation"],
                default_source=parameter["default_source"],
            )
            for parameter in entry["parameters"]
        ),
        return_annotation=entry["return_annotation"],
        annotation_names=tuple(entry["annotation_names"]),
        called_paths=tuple(entry["called_paths"]),
        call_sites=tuple(
            CallSiteFact(
                target=call["target"],
                root_name=call["root_name"],
                invocation_mode=call["invocation_mode"],
                lineno=call["lineno"],
                end_lineno=call["end_lineno"],
                col_offset=call["col_offset"],
                end_col_offset=call["end_col_offset"],
                requires_same_unit=call["requires_same_unit"],
            )
            for call in entry["call_sites"]
        ),
        suspension_points=tuple(
            SuspensionPoint(
                kind=point["kind"],
                lineno=point["lineno"],
                end_lineno=point["end_lineno"],
                col_offset=point["col_offset"],
                end_col_offset=point["end_col_offset"],
            )
            for point in entry["suspension_points"]
        ),
        runtime_imports=tuple(_import_from_cache(record) for record in entry["runtime_imports"]),
        base_names=tuple(entry["base_names"]),
        fields=tuple(
            FieldRecord(
                name=field["name"],
                annotation=field["annotation"],
                default_source=field["default_source"],
                class_variable=field["class_variable"],
            )
            for field in entry["fields"]
        ),
        declaration_start_lineno=entry["declaration_start_lineno"],
        scope_type_parameters=tuple(entry["scope_type_parameters"]),
        type_parameter_records=tuple(
            _type_parameter_from_cache(record) for record in entry["type_parameter_records"]
        ),
        scope_type_parameter_records=tuple(
            _type_parameter_from_cache(record) for record in entry["scope_type_parameter_records"]
        ),
        any_annotation_sources=tuple(entry["any_annotation_sources"]),
    )


def _type_parameter_to_cache(record: TypeParameterRecord) -> TypeParameterCacheEntry:
    return {
        "name": record.name,
        "kind": record.kind,
        "declaration": record.declaration,
    }


def _type_parameter_from_cache(entry: TypeParameterCacheEntry) -> TypeParameterRecord:
    return TypeParameterRecord(
        name=entry["name"],
        kind=entry["kind"],
        declaration=entry["declaration"],
    )


def _blocker_to_cache(blocker: Blocker) -> BlockerCacheEntry:
    return {
        "severity": blocker.severity,
        "code": blocker.code,
        "message": blocker.message,
        "lineno": blocker.lineno,
        "symbol_module": blocker.symbol.module if blocker.symbol is not None else None,
        "symbol_qualname": blocker.symbol.qualname if blocker.symbol is not None else None,
    }


def _blocker_from_cache(entry: BlockerCacheEntry) -> Blocker:
    symbol = _optional_symbol(entry["symbol_module"], entry["symbol_qualname"])
    return Blocker(
        severity=entry["severity"],
        code=entry["code"],
        message=entry["message"],
        lineno=entry["lineno"],
        symbol=symbol,
    )


def _import_from_cache(entry: ImportCacheEntry) -> ImportRecord:
    return ImportRecord(
        source_text=entry["source_text"],
        imported_names=tuple(entry["imported_names"]),
        module=entry["module"],
        level=entry["level"],
        lineno=entry["lineno"],
        end_lineno=entry["end_lineno"],
    )


def _constant_from_cache(entry: ConstantCacheEntry) -> ConstantRecord:
    return ConstantRecord(
        name=entry["name"],
        kind=entry["kind"],
        source_text=entry["source_text"],
        lineno=entry["lineno"],
        end_lineno=entry["end_lineno"],
    )


def _optional_symbol(module: str | None, qualname: str | None) -> SymbolId | None:
    if module is None or qualname is None:
        return None
    return SymbolId(module=module, qualname=qualname)


def _read_index(path: Path) -> CacheIndex:
    if not path.exists():
        return {"version": 1, "files": {}}
    try:
        raw = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
        if raw.get("version") != 1:
            return {"version": 1, "files": {}}
        raw_files = raw.get("files")
        if not isinstance(raw_files, dict):
            return {"version": 1, "files": {}}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {"version": 1, "files": {}}
    else:
        files = cast(dict[str, FileCacheEntry], raw_files)
        return {"version": 1, "files": files}


def _write_index(path: Path, index: CacheIndex) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(index, indent=2, sort_keys=True)}\n", encoding="utf-8")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _cache_key(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _cached_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def _python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"
