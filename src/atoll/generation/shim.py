"""Managed shim insertion and removal for Atoll-enabled modules."""

from __future__ import annotations

import difflib
from dataclasses import dataclass

from atoll.models import EnabledIslandConfig


@dataclass(frozen=True, slots=True)
class ShimEdit:
    """Result of rendering a shim edit for a source module."""

    old_text: str
    new_text: str
    diff: str


def insert_or_replace_shim(source_text: str, config: EnabledIslandConfig) -> ShimEdit:
    """Append or replace the managed Atoll shim for `config`."""
    new_text = _replace_block(source_text, config, render_shim(config))
    return _edit(source_text, new_text, config.source_path.name)


def remove_shim(source_text: str, config: EnabledIslandConfig) -> ShimEdit:
    """Remove the managed Atoll shim for `config` if present."""
    new_text = _replace_block(source_text, config, "")
    return _edit(source_text, new_text, config.source_path.name)


def render_shim(config: EnabledIslandConfig) -> str:
    """Render a marker-delimited Atoll managed shim."""
    status_symbols = tuple(config.symbols)
    assignments = "\n".join(
        f"            {symbol} = _atoll_mod.{symbol}" for symbol in config.symbols
    )
    return "\n".join(
        [
            _begin_marker(config.source_module),
            "# This block is managed by Atoll. Do not edit manually.",
            "try:",
            "    import importlib as _atoll_importlib",
            "    import importlib.machinery as _atoll_machinery",
            "    import os as _atoll_os",
            "",
            "    __atoll_status__ = {",
            f'        "source_module": "{config.source_module}",',
            f'        "sidecar_module": "{config.sidecar_module}",',
            '        "active": False,',
            '        "compiled": False,',
            f'        "symbols": {status_symbols!r},',
            '        "origin": None,',
            '        "error": None,',
            "    }",
            "",
            '    if _atoll_os.getenv("ATOLL_DISABLE") != "1":',
            "        try:",
            f'            _atoll_mod = _atoll_importlib.import_module("{config.sidecar_module}")',
            '            _atoll_origin = getattr(_atoll_mod, "__file__", "") or ""',
            "            _atoll_compiled = any(",
            "                _atoll_origin.endswith(_suffix)",
            "                for _suffix in _atoll_machinery.EXTENSION_SUFFIXES",
            "            )",
            "",
            "            if (",
            '                _atoll_os.getenv("ATOLL_REQUIRE_COMPILED") == "1"',
            "                and not _atoll_compiled",
            "            ):",
            "                raise ImportError(",
            f'                    "Atoll sidecar {config.sidecar_module} imported, "',
            '                    "but it is not a compiled extension"',
            "                )",
            "",
            assignments,
            "",
            "            __atoll_status__.update({",
            '                "active": True,',
            '                "compiled": _atoll_compiled,',
            '                "origin": _atoll_origin,',
            "            })",
            "        except ImportError as _atoll_error:",
            '            __atoll_status__["error"] = repr(_atoll_error)',
            "            if (",
            '                _atoll_os.getenv("ATOLL_STRICT") == "1"',
            '                or _atoll_os.getenv("ATOLL_REQUIRE_COMPILED") == "1"',
            "            ):",
            "                raise",
            "finally:",
            "    for _atoll_name in (",
            '        "_atoll_importlib",',
            '        "_atoll_machinery",',
            '        "_atoll_os",',
            "    ):",
            "        globals().pop(_atoll_name, None)",
            _end_marker(config.source_module),
            "",
        ]
    )


def _replace_block(source_text: str, config: EnabledIslandConfig, block: str) -> str:
    begin = _begin_marker(config.source_module)
    end = _end_marker(config.source_module)
    begin_count = source_text.count(begin)
    end_count = source_text.count(end)
    if begin_count != end_count:
        raise ValueError(f"unbalanced Atoll managed block markers for {config.source_module}")
    if begin_count > 1:
        raise ValueError(f"multiple Atoll managed blocks for {config.source_module}")
    if begin_count == 0:
        if not block:
            return source_text
        return f"{source_text.rstrip()}\n\n{block}"
    start = source_text.index(begin)
    stop = source_text.index(end, start) + len(end)
    replacement = block.rstrip()
    prefix = source_text[:start].rstrip()
    suffix = source_text[stop:].lstrip("\n")
    if replacement and suffix:
        return f"{prefix}\n\n{replacement}\n\n{suffix}"
    if replacement:
        return f"{prefix}\n\n{replacement}\n"
    if suffix:
        return f"{prefix}\n\n{suffix}"
    return f"{prefix}\n"


def _edit(old_text: str, new_text: str, filename: str) -> ShimEdit:
    return ShimEdit(
        old_text=old_text,
        new_text=new_text,
        diff="".join(
            difflib.unified_diff(
                old_text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=f"{filename}:before",
                tofile=f"{filename}:after",
            )
        ),
    )


def _begin_marker(module: str) -> str:
    return f"# BEGIN ATOLL MANAGED: {module}"


def _end_marker(module: str) -> str:
    return f"# END ATOLL MANAGED: {module}"
