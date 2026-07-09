"""Managed shim insertion and removal for Atoll-enabled modules.

The shim is the runtime switch between original Python definitions and generated
sidecars. It is marker-delimited so commands can replace it safely, and it
records status in `__atoll_status__` for verification without requiring users to
inspect generated code.
"""

from __future__ import annotations

import difflib
import os
from dataclasses import dataclass
from pathlib import Path

from atoll.models import EnabledIslandConfig


@dataclass(frozen=True, slots=True)
class ShimEdit:
    """Preview and applied text for a managed shim edit.

    Commands use this immutable object for dry-run output and file writes. The
    unified diff is derived from the old and new source text and is safe to show
    to users before any mutation occurs.
    """

    old_text: str
    new_text: str
    diff: str


def insert_or_replace_shim(source_text: str, config: EnabledIslandConfig) -> ShimEdit:
    """Append or replace the managed Atoll shim for `config`.

    Existing balanced markers are replaced in place; missing markers append a new
    managed block. Unbalanced or duplicate markers raise `ValueError` so commands
    do not overwrite ambiguous user code.
    """
    new_text = _replace_block(source_text, config, render_shim(config))
    return _edit(source_text, new_text, config.source_path.name)


def remove_shim(source_text: str, config: EnabledIslandConfig) -> ShimEdit:
    """Remove the managed Atoll shim for `config` if present.

    Missing markers are treated as a no-op. Marker imbalance still raises because
    it means Atoll cannot determine a safe deletion range.
    """
    new_text = _replace_block(source_text, config, "")
    return _edit(source_text, new_text, config.source_path.name)


def render_shim(config: EnabledIslandConfig) -> str:
    """Render the marker-delimited runtime shim for one enabled island.

    The shim prefers compiled artifacts when present, falls back to the generated
    Python sidecar otherwise, and honors `ATOLL_DISABLE`, `ATOLL_STRICT`, and
    `ATOLL_REQUIRE_COMPILED`. It keeps transient helper names out of the module
    namespace after import.
    """
    status_symbols = tuple(config.symbols)
    assignments = "\n".join(
        f"            {symbol} = _atoll_mod.{symbol}" for symbol in config.symbols
    )
    sidecar_relative = _relative_path_text(config.source_path.parent, config.sidecar_path)
    artifact_relative = _relative_path_text(config.source_path.parent, _artifact_dir(config))
    return "\n".join(
        [
            _begin_marker(config.source_module),
            "# This block is managed by Atoll. Do not edit manually.",
            "try:",
            "    import importlib.machinery as _atoll_machinery",
            "    import importlib.util as _atoll_util",
            "    import os as _atoll_os",
            "    import pathlib as _atoll_pathlib",
            "    import sys as _atoll_sys",
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
            "    _atoll_added_artifact_path = False",
            "    _atoll_artifact_dir_text = None",
            "",
            '    if _atoll_os.getenv("ATOLL_DISABLE") != "1":',
            "        try:",
            "            _atoll_source_dir = _atoll_pathlib.Path(__file__).resolve().parent",
            "            _atoll_sidecar_path = (",
            f'                _atoll_source_dir / "{sidecar_relative}"',
            "            ).resolve()",
            "            _atoll_artifact_dir = (",
            f'                _atoll_source_dir / "{artifact_relative}"',
            "            ).resolve()",
            "            _atoll_compiled_paths = tuple(",
            "                sorted(",
            "                    _atoll_candidate",
            "                    for _atoll_suffix in _atoll_machinery.EXTENSION_SUFFIXES",
            "                    for _atoll_candidate in _atoll_artifact_dir.rglob(",
            '                        f"{_atoll_sidecar_path.stem}*{_atoll_suffix}"',
            "                    )",
            "                )",
            "            )",
            "            _atoll_compiled = bool(_atoll_compiled_paths)",
            "            if (",
            '                _atoll_os.getenv("ATOLL_REQUIRE_COMPILED") == "1"',
            "                and not _atoll_compiled",
            "            ):",
            "                raise ImportError(",
            "                    "
            f'"Atoll sidecar {config.sidecar_module} has no compiled extension"',
            "                )",
            "            _atoll_origin_path = (",
            "                _atoll_compiled_paths[0] if _atoll_compiled else _atoll_sidecar_path",
            "            )",
            "            _atoll_origin = str(_atoll_origin_path)",
            "            if _atoll_compiled:",
            "                _atoll_artifact_dir_text = str(_atoll_artifact_dir)",
            "                if _atoll_artifact_dir_text not in _atoll_sys.path:",
            "                    _atoll_sys.path.insert(0, _atoll_artifact_dir_text)",
            "                    _atoll_added_artifact_path = True",
            "            _atoll_spec = _atoll_util.spec_from_file_location(",
            f'                "{config.sidecar_module}", _atoll_origin_path',
            "            )",
            "            if _atoll_spec is None or _atoll_spec.loader is None:",
            "                raise ImportError(",
            f'                    "Atoll sidecar {config.sidecar_module} cannot be loaded"',
            "                )",
            "            _atoll_mod = _atoll_util.module_from_spec(_atoll_spec)",
            f'            _atoll_sys.modules["{config.sidecar_module}"] = _atoll_mod',
            "            _atoll_spec.loader.exec_module(_atoll_mod)",
            "            if _atoll_added_artifact_path and _atoll_artifact_dir_text is not None:",
            "                _atoll_sys.path.remove(_atoll_artifact_dir_text)",
            "                _atoll_added_artifact_path = False",
            "",
            assignments,
            "",
            "            __atoll_status__.update({",
            '                "active": True,',
            '                "compiled": _atoll_compiled,',
            '                "origin": _atoll_origin,',
            "            })",
            "        except Exception as _atoll_error:",
            "            if _atoll_added_artifact_path and _atoll_artifact_dir_text is not None:",
            "                try:",
            "                    _atoll_sys.path.remove(_atoll_artifact_dir_text)",
            "                except ValueError:",
            "                    pass",
            '            __atoll_status__["error"] = repr(_atoll_error)',
            "            if (",
            '                _atoll_os.getenv("ATOLL_STRICT") == "1"',
            '                or _atoll_os.getenv("ATOLL_REQUIRE_COMPILED") == "1"',
            "            ):",
            "                raise",
            "finally:",
            "    for _atoll_name in (",
            '        "_atoll_added_artifact_path",',
            '        "_atoll_artifact_dir",',
            '        "_atoll_artifact_dir_text",',
            '        "_atoll_candidate",',
            '        "_atoll_compiled",',
            '        "_atoll_compiled_paths",',
            '        "_atoll_error",',
            '        "_atoll_machinery",',
            '        "_atoll_mod",',
            '        "_atoll_os",',
            '        "_atoll_origin",',
            '        "_atoll_origin_path",',
            '        "_atoll_pathlib",',
            '        "_atoll_sidecar_path",',
            '        "_atoll_source_dir",',
            '        "_atoll_spec",',
            '        "_atoll_suffix",',
            '        "_atoll_sys",',
            '        "_atoll_util",',
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


def _artifact_dir(config: EnabledIslandConfig) -> Path:
    if config.sidecar_path.parent.name == "sidecars":
        return config.sidecar_path.parent.parent / "artifacts"
    return config.sidecar_path.parent


def _relative_path_text(start: Path, path: Path) -> str:
    return os.path.relpath(os.fspath(path.resolve()), start=os.fspath(start.resolve()))
