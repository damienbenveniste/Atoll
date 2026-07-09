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
    Blocker,
    BlockerSeverity,
    ConstantKind,
    ConstantRecord,
    ImportRecord,
    ModuleId,
    ModuleScan,
    ProjectConfig,
    SymbolId,
    SymbolKind,
    SymbolRecord,
    Visibility,
)

SCANNER_VERSION = "1"


class CacheStats(TypedDict):
    """Hit and miss counts returned with a cached scan run."""

    hits: int
    misses: int


class BlockerCacheEntry(TypedDict):
    """JSON-safe representation of a `Blocker`.

    The symbol identity is split into nullable module and qualname fields because
    module-level blockers do not belong to a concrete symbol.
    """

    severity: BlockerSeverity
    code: str
    message: str
    lineno: int | None
    symbol_module: str | None
    symbol_qualname: str | None


class ImportCacheEntry(TypedDict):
    """Cached top-level import record used for dependency and sidecar analysis."""

    source_text: str
    imported_names: list[str]
    module: str | None
    level: int
    lineno: int
    end_lineno: int


class ConstantCacheEntry(TypedDict):
    """Cached top-level assignment record and its literal-safety classification."""

    name: str
    kind: ConstantKind
    source_text: str
    lineno: int
    end_lineno: int


class SymbolCacheEntry(TypedDict):
    """Cached AST facts for a function, class, or simple method.

    The payload intentionally excludes mypy diagnostics and candidate data
    because those are enrichment outputs that can change without the source file
    itself changing.
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
    blockers: list[BlockerCacheEntry]


class ModuleScanCacheEntry(TypedDict):
    """Cached first-pass scan for one module before enrichment."""

    module_name: str
    path: str
    imports: list[ImportCacheEntry]
    constants: list[ConstantCacheEntry]
    symbols: list[SymbolCacheEntry]
    blockers: list[BlockerCacheEntry]
    top_level_statement_lines: list[int]


class FileCacheEntry(TypedDict):
    """One indexed source file and the scanner inputs that validate its cache."""

    path: str
    module_name: str
    sha256: str
    python_version: str
    scanner_version: str
    scan: ModuleScanCacheEntry


class CacheIndex(TypedDict):
    """Root cache file mapping relative source paths to scan entries."""

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
