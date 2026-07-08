"""Verify that Atoll-managed source modules route to sidecars."""

from __future__ import annotations

import importlib
import importlib.machinery
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

from atoll.models import EnabledIslandConfig, ProjectConfig, VerifyResult


def verify_islands(
    config: ProjectConfig,
    *,
    module_name: str | None = None,
    require_compiled: bool = False,
) -> tuple[VerifyResult, ...]:
    """Verify configured enabled islands by importing the source modules."""
    enabled = tuple(
        island
        for island in config.islands
        if island.enabled and (module_name is None or island.source_module == module_name)
    )
    with _python_path(config.source_roots):
        return tuple(
            _verify_island(island, require_compiled=require_compiled) for island in enabled
        )


def _verify_island(island: EnabledIslandConfig, *, require_compiled: bool) -> VerifyResult:
    try:
        _clear_modules(island.source_module, island.sidecar_module)
        module = importlib.import_module(island.source_module)
        raw_status = getattr(module, "__atoll_status__", None)
        if not isinstance(raw_status, dict):
            return _failed(island, "source module has no __atoll_status__")
        status = cast(dict[object, object], raw_status)
        active = bool(status.get("active"))
        compiled = bool(status.get("compiled"))
        origin = _optional_string(status.get("origin"))
        symbols = tuple(
            (
                symbol,
                getattr(getattr(module, symbol, None), "__module__", None) == island.sidecar_module,
            )
            for symbol in island.symbols
        )
        error = _status_error(status, active, compiled, symbols, require_compiled)
        return VerifyResult(
            source_module=island.source_module,
            sidecar_module=island.sidecar_module,
            active=active,
            compiled=compiled,
            origin=origin,
            symbols=symbols,
            error=error,
        )
    except Exception as error:
        return _failed(island, repr(error))


def _status_error(
    status: dict[object, object],
    active: bool,
    compiled: bool,
    symbols: tuple[tuple[str, bool], ...],
    require_compiled: bool,
) -> str | None:
    if not active:
        return _optional_string(status.get("error")) or "Atoll shim is not active"
    if any(not ok for _, ok in symbols):
        return "one or more symbols are not rebound to the sidecar module"
    if require_compiled and not compiled:
        return "sidecar is active but is not a compiled extension"
    origin = _optional_string(status.get("origin"))
    if require_compiled and origin is not None and not _is_extension_origin(origin):
        return "sidecar origin does not use a compiled extension suffix"
    return None


def _failed(island: EnabledIslandConfig, error: str) -> VerifyResult:
    return VerifyResult(
        source_module=island.source_module,
        sidecar_module=island.sidecar_module,
        active=False,
        compiled=False,
        origin=None,
        symbols=tuple((symbol, False) for symbol in island.symbols),
        error=error,
    )


def _clear_modules(source_module: str, sidecar_module: str) -> None:
    module_names = {
        source_module,
        sidecar_module,
        *_module_ancestors(source_module),
        *_module_ancestors(sidecar_module),
    }
    for module_name in sorted(module_names, reverse=True):
        sys.modules.pop(module_name, None)
    importlib.invalidate_caches()


def _module_ancestors(module_name: str) -> tuple[str, ...]:
    parts = module_name.split(".")
    return tuple(".".join(parts[:index]) for index in range(1, len(parts)))


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _is_extension_origin(origin: str) -> bool:
    return any(origin.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES)


@contextmanager
def _python_path(source_roots: tuple[Path, ...]) -> Generator[None]:
    original = list(sys.path)
    sys.path[:0] = [str(path) for path in source_roots]
    try:
        yield
    finally:
        sys.path[:] = original
