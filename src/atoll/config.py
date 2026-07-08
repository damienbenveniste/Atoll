"""Read and write Atoll project configuration."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import cast

from atoll.models import EnabledIslandConfig

CONFIG_PATH = ".atoll.toml"


def load_enabled_islands(root: Path) -> tuple[EnabledIslandConfig, ...]:
    """Load enabled islands from `.atoll.toml` and `pyproject.toml` if present."""
    islands: list[EnabledIslandConfig] = []
    atoll_config = root / CONFIG_PATH
    if atoll_config.exists():
        islands.extend(_read_islands_from_file(atoll_config, root))
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        islands.extend(_read_islands_from_file(pyproject, root))
    return tuple(_dedupe_islands(islands))


def write_atoll_config(root: Path, islands: tuple[EnabledIslandConfig, ...]) -> Path:
    """Write Atoll's minimal `.atoll.toml` configuration."""
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
    """Insert or replace one enabled island in project configuration."""
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
    """Mark one configured island disabled."""
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
