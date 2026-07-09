"""Tests for Atoll managed shim edits."""

import asyncio
import importlib
import importlib.machinery
import inspect
import sys
from pathlib import Path

import pytest

from atoll.generation.shim import insert_or_replace_shim, remove_shim, render_shim
from atoll.models import EnabledIslandConfig


def _island(
    tmp_path: Path,
    *,
    symbols: tuple[str, ...] = ("score_user", "rank_candidates"),
) -> EnabledIslandConfig:
    return EnabledIslandConfig(
        source_module="app.ranking",
        source_path=tmp_path / "src" / "app" / "ranking.py",
        sidecar_module="app._atoll_app_ranking",
        sidecar_path=tmp_path / ".atoll" / "sidecars" / "_atoll_app_ranking.py",
        symbols=symbols,
    )


def test_insert_replace_and_remove_shim(tmp_path: Path) -> None:
    """Managed shims are marker-delimited, replaceable, and removable."""
    island = _island(tmp_path)
    source = "def score_user() -> int:\n    return 1\n"

    inserted = insert_or_replace_shim(source, island)
    replaced = insert_or_replace_shim(inserted.new_text, island)
    removed = remove_shim(replaced.new_text, island)

    assert "# BEGIN ATOLL MANAGED: app.ranking" in inserted.new_text
    assert inserted.new_text.count("# BEGIN ATOLL MANAGED") == 1
    assert replaced.new_text.count("# BEGIN ATOLL MANAGED") == 1
    assert "# BEGIN ATOLL MANAGED" not in removed.new_text
    assert "def score_user" in removed.new_text


def test_remove_shim_without_block_is_noop(tmp_path: Path) -> None:
    """Removing a missing shim leaves source unchanged."""
    source = "VALUE = 1\n"

    edit = remove_shim(source, _island(tmp_path))

    assert edit.new_text == source
    assert edit.diff == ""


def test_shim_rejects_unbalanced_and_duplicate_markers(tmp_path: Path) -> None:
    """Invalid managed marker layout fails loudly."""
    island = _island(tmp_path)

    with pytest.raises(ValueError, match="unbalanced"):
        insert_or_replace_shim("# BEGIN ATOLL MANAGED: app.ranking\n", island)

    duplicate = (
        "# BEGIN ATOLL MANAGED: app.ranking\n"
        "# END ATOLL MANAGED: app.ranking\n"
        "# BEGIN ATOLL MANAGED: app.ranking\n"
        "# END ATOLL MANAGED: app.ranking\n"
    )
    with pytest.raises(ValueError, match="multiple"):
        insert_or_replace_shim(duplicate, island)


def test_compiled_bindings_preserve_callable_metadata_and_kind(tmp_path: Path) -> None:
    """Generated compiled bindings wrap native targets with source metadata."""
    shim = render_shim(_island(tmp_path))

    assert "_atoll_functools.wraps(_atoll_source)" in shim
    assert "_atoll_wrapped.__signature__ = _atoll_inspect.signature(_atoll_source)" in shim
    assert "_atoll_wrapped.__atoll_compiled_target__ = _atoll_target" in shim
    assert "_atoll_inspect.isasyncgenfunction(_atoll_source)" in shim
    assert "_atoll_inspect.iscoroutinefunction(_atoll_source)" in shim
    assert "_atoll_inspect.isgeneratorfunction(_atoll_source)" in shim
    assert "score_user = _atoll_bind_compiled(score_user, _atoll_mod.score_user)" in shim


def test_compiled_bindings_preserve_runtime_callable_kinds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managed wrappers retain callable kinds while dispatching to native targets."""
    symbols = ("sync_call", "coroutine_call", "generator_call", "async_generator_call")
    island = _island(tmp_path, symbols=symbols)
    island.source_path.parent.mkdir(parents=True)
    (island.source_path.parent / "__init__.py").write_text("", encoding="utf-8")
    source = """
def sync_call(value: int) -> int:
    return value + 1

async def coroutine_call(value: int) -> int:
    return value + 1

def generator_call(value: int):
    yield value + 1

async def async_generator_call(value: int):
    yield value + 1
""".lstrip()
    compiled_source = """
def sync_call(value):
    return value * 2

async def coroutine_call(value):
    return value * 3

def generator_call(value):
    yield value * 4

async def async_generator_call(value):
    yield value * 5
""".lstrip()
    artifact_dir = tmp_path / ".atoll" / "artifacts"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / island.sidecar_path.name).write_text(compiled_source, encoding="utf-8")
    island.source_path.write_text(f"{source}\n{render_shim(island)}", encoding="utf-8")

    monkeypatch.setattr(importlib.machinery, "EXTENSION_SUFFIXES", [".py"])
    source_root = str(tmp_path / "src")
    sys.path.insert(0, source_root)
    for module_name in (island.source_module, island.sidecar_module, "app"):
        sys.modules.pop(module_name, None)

    try:
        module = importlib.import_module(island.source_module)

        assert inspect.isfunction(module.sync_call)
        assert inspect.iscoroutinefunction(module.coroutine_call)
        assert inspect.isgeneratorfunction(module.generator_call)
        assert inspect.isasyncgenfunction(module.async_generator_call)
        expected_sync_result = 6
        assert module.sync_call(3) == expected_sync_result
        assert list(module.generator_call(3)) == [12]

        async def collect_async_results() -> tuple[int, list[int]]:
            coroutine_result = await module.coroutine_call(3)
            generator_result = [item async for item in module.async_generator_call(3)]
            return coroutine_result, generator_result

        assert asyncio.run(collect_async_results()) == (9, [15])
        for symbol in symbols:
            exported = getattr(module, symbol)
            assert inspect.signature(exported) == inspect.signature(inspect.unwrap(exported))
            assert exported.__atoll_compiled_target__.__module__ == island.sidecar_module
        assert not hasattr(module, "_atoll_bind_compiled")
        assert not hasattr(module, "_atoll_functools")
        assert not hasattr(module, "_atoll_inspect")
    finally:
        sys.path.remove(source_root)
        for module_name in (island.source_module, island.sidecar_module, "app"):
            sys.modules.pop(module_name, None)
