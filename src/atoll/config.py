"""Read and write Atoll project configuration.

Atoll accepts island definitions from `.atoll.toml` and `pyproject.toml`, then
writes its managed minimal configuration back to `.atoll.toml`. Source-clean
compile policy is read only from `pyproject.toml` so legacy in-place commands
cannot rewrite user-owned test or benchmark settings.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import cast

from atoll.models import Backend, CompileConfig, EnabledIslandConfig
from atoll.optimization_policy import DEFAULT_MINIMUM_FINAL_SPEEDUP

CONFIG_PATH = ".atoll.toml"


class CompileConfigError(ValueError):
    """Raised when source-clean compile policy cannot be parsed or validated."""


def load_compile_config(root: Path) -> CompileConfig:
    """Load and validate optional `[tool.atoll.compile]` source-clean policy.

    Args:
        root: Root directory of the target Python project.

    Returns:
        CompileConfig: Normalized compile, test, and benchmark configuration.

    Raises:
        CompileConfigError: If TOML syntax, field types, backend names, or gate policy are invalid.
    """
    try:
        return _load_compile_config(root)
    except (tomllib.TOMLDecodeError, TypeError, ValueError) as error:
        raise CompileConfigError(f"invalid [tool.atoll.compile] configuration: {error}") from error


def _load_compile_config(root: Path) -> CompileConfig:
    """Parse compile policy before the public loader normalizes configuration errors.

    Args:
        root: Root directory of the target Python project.

    Returns:
        CompileConfig: Parsed compile policy before public error normalization.
    """
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        return CompileConfig()
    data = cast(dict[str, object], tomllib.loads(pyproject.read_text(encoding="utf-8")))
    compile_data = _mapping(_mapping(_mapping(data.get("tool")).get("atoll")).get("compile"))
    if not compile_data:
        return CompileConfig()
    backends = _backend_sequence(compile_data.get("backends"))
    test_command = _optional_command(compile_data, "test_command")
    benchmark_command = _optional_command(compile_data, "benchmark_command")
    return CompileConfig(
        backends=backends if backends is not None else CompileConfig().backends,
        test_command=test_command,
        benchmark_command=benchmark_command,
        benchmark_warmups=_integer(
            compile_data.get("benchmark_warmups"),
            field="benchmark_warmups",
            default=1,
        ),
        benchmark_samples=_integer(
            compile_data.get("benchmark_samples"),
            field="benchmark_samples",
            default=7,
        ),
        minimum_speedup=_number(
            compile_data.get("minimum_speedup"),
            field="minimum_speedup",
            default=DEFAULT_MINIMUM_FINAL_SPEEDUP,
        ),
    )


def load_enabled_islands(root: Path) -> tuple[EnabledIslandConfig, ...]:
    """Load enabled islands from `.atoll.toml` and `pyproject.toml` if present.

    Args:
        root: Root directory of the target Python project.

    Returns:
        tuple[EnabledIslandConfig, ...]: Configured islands in persisted order, or an empty tuple
            when absent.
    """
    islands: list[EnabledIslandConfig] = []
    atoll_config = root / CONFIG_PATH
    if atoll_config.exists():
        islands.extend(_read_islands_from_file(atoll_config, root))
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        islands.extend(_read_islands_from_file(pyproject, root))
    return tuple(_dedupe_islands(islands))


def write_atoll_config(root: Path, islands: tuple[EnabledIslandConfig, ...]) -> Path:
    """Write Atoll's minimal `.atoll.toml` configuration.

    The file is fully rewritten from the supplied island tuple, so callers must
    preserve entries they do not intend to remove. Paths are emitted relative to
    the project root when possible.

    Args:
        root: Root directory of the target Python project.
        islands: Enabled island configurations to serialize.

    Returns:
        Path: Path to the atomically rewritten Atoll configuration file.
    """
    path = root / CONFIG_PATH
    lines = [
        "[tool.atoll]",
        'backend = "mypyc"',
        'source_roots = ["src"]',
        'cache_dir = ".atoll/cache"',
        'report_dir = ".atoll"',
        "",
    ]
    for island in islands:
        lines.extend(
            [
                "[[tool.atoll.island]]",
                f'source_module = "{_escape_toml_string(island.source_module)}"',
                f'source_path = "{_escape_toml_string(_relative_text(root, island.source_path))}"',
                f'sidecar_module = "{_escape_toml_string(island.sidecar_module)}"',
                (
                    'sidecar_path = "'
                    f'{_escape_toml_string(_relative_text(root, island.sidecar_path))}"'
                ),
                f"symbols = [{_toml_string_list(island.symbols)}]",
                f"enabled = {_toml_bool(island.enabled)}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def upsert_enabled_island(
    root: Path,
    island: EnabledIslandConfig,
) -> tuple[EnabledIslandConfig, ...]:
    """Insert or replace one enabled island in project configuration.

    Args:
        root: Root directory of the target Python project.
        island: Enabled island configuration to insert or update.

    Returns:
        tuple[EnabledIslandConfig, ...]: Complete island configuration after inserting or replacing
            the module.
    """
    islands = [
        existing
        for existing in load_enabled_islands(root)
        if existing.source_module != island.source_module
    ]
    islands.append(island)
    resolved = tuple(sorted(islands, key=lambda item: item.source_module))
    write_atoll_config(root, resolved)
    return resolved


def disable_island(root: Path, source_module: str) -> tuple[EnabledIslandConfig, ...]:
    """Mark one configured island disabled and rewrite configuration.

    Non-matching islands are preserved exactly. If the module is not configured,
    the resulting config is unchanged apart from normal rewrite formatting.

    Args:
        root: Root directory of the target Python project.
        source_module: Importable source module name.

    Returns:
        tuple[EnabledIslandConfig, ...]: Complete island configuration with the selected module
            disabled.
    """
    islands = tuple(
        EnabledIslandConfig(
            source_module=island.source_module,
            source_path=island.source_path,
            sidecar_module=island.sidecar_module,
            sidecar_path=island.sidecar_path,
            symbols=island.symbols,
            enabled=False if island.source_module == source_module else island.enabled,
        )
        for island in load_enabled_islands(root)
    )
    write_atoll_config(root, islands)
    return islands


def _read_islands_from_file(path: Path, root: Path) -> tuple[EnabledIslandConfig, ...]:
    data = cast(dict[str, object], tomllib.loads(path.read_text(encoding="utf-8")))
    tool = _mapping(data.get("tool"))
    atoll = _mapping(tool.get("atoll"))
    island_entries = _sequence(atoll.get("island"))
    return tuple(
        island
        for entry in island_entries
        if (island := _parse_island(_mapping(entry), root)) is not None
    )


def _parse_island(data: dict[str, object], root: Path) -> EnabledIslandConfig | None:
    source_module = _string(data.get("source_module"))
    source_path = _string(data.get("source_path"))
    sidecar_module = _string(data.get("sidecar_module"))
    sidecar_path = _string(data.get("sidecar_path"))
    symbols = tuple(_string(item) for item in _sequence(data.get("symbols")))
    enabled = _bool(data.get("enabled"), default=True)
    if (
        source_module is None
        or source_path is None
        or sidecar_module is None
        or sidecar_path is None
        or not symbols
        or any(symbol is None for symbol in symbols)
    ):
        return None
    return EnabledIslandConfig(
        source_module=source_module,
        source_path=(root / source_path).resolve(),
        sidecar_module=sidecar_module,
        sidecar_path=(root / sidecar_path).resolve(),
        symbols=tuple(symbol for symbol in symbols if symbol is not None),
        enabled=enabled,
    )


def _dedupe_islands(islands: list[EnabledIslandConfig]) -> tuple[EnabledIslandConfig, ...]:
    by_module = {island.source_module: island for island in islands}
    return tuple(by_module[module] for module in sorted(by_module))


def _mapping(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        raw = cast(dict[object, object], value)
        return {str(key): item for key, item in raw.items()}
    return {}


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, list):
        return tuple(cast(list[object], value))
    return ()


def _string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _backend_sequence(value: object) -> tuple[Backend, ...] | None:
    if value is None:
        return None
    raw = _required_string_sequence(value, field="backends")
    invalid = tuple(backend for backend in raw if backend not in {"mypyc", "cython"})
    if invalid:
        raise ValueError(
            "tool.atoll.compile.backends supports only mypyc and cython: " + ", ".join(invalid)
        )
    return cast(tuple[Backend, ...], raw)


def _optional_command(data: dict[str, object], field: str) -> tuple[str, ...] | None:
    value = data.get(field)
    if value is None:
        return None
    return _required_string_sequence(value, field=field)


def _required_string_sequence(value: object, *, field: str) -> tuple[str, ...]:
    values = _sequence(value)
    if not values or any(not isinstance(item, str) or not item.strip() for item in values):
        raise ValueError(f"tool.atoll.compile.{field} must be a non-empty string list")
    return tuple(cast(str, item) for item in values)


def _integer(value: object, *, field: str, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"tool.atoll.compile.{field} must be an integer")
    return value


def _number(value: object, *, field: str, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"tool.atoll.compile.{field} must be a number")
    return float(value)


def _bool(value: object, *, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _relative_text(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _toml_string_list(values: tuple[str, ...]) -> str:
    return ", ".join(f'"{_escape_toml_string(value)}"' for value in values)


def _toml_bool(value: bool) -> str:
    return "true" if value else "false"


def _escape_toml_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
