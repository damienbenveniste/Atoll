"""Tests for strict profile candidate-plan selection caching."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from atoll.models import SymbolId
from atoll.profile_plan_cache import ProfilePlanIdentity, select_profile_plan

_BASE_IDENTITY = ProfilePlanIdentity(
    scope_identity="scope-a",
    candidate_identity=("candidate-a", "candidate-b"),
    module_source_hashes=(("app.helper", "hash-helper"), ("app.worker", "hash-worker")),
    benchmark_argv=("python", "-m", "bench"),
    backend_order=("mypyc", "cython"),
    module_scope="app",
    python_cache_tag="cpython-312",
    python_platform="linux-x86_64",
    cache_format_version="1",
    lowering_version="1",
)

_INVALID_IDENTITY_FACTORIES: dict[str, Callable[[], ProfilePlanIdentity]] = {
    "scope": lambda: replace(_BASE_IDENTITY, scope_identity=""),
    "module-scope": lambda: replace(_BASE_IDENTITY, module_scope=""),
    "candidates-empty": lambda: replace(_BASE_IDENTITY, candidate_identity=()),
    "candidates-duplicate": lambda: replace(
        _BASE_IDENTITY,
        candidate_identity=("candidate-a", "candidate-a"),
    ),
    "benchmark": lambda: replace(_BASE_IDENTITY, benchmark_argv=()),
    "backends": lambda: replace(_BASE_IDENTITY, backend_order=()),
    "sources-empty": lambda: replace(_BASE_IDENTITY, module_source_hashes=()),
    "source-value": lambda: replace(
        _BASE_IDENTITY,
        module_source_hashes=(("app.worker", ""),),
    ),
}


def test_first_selection_is_canonical_miss_then_strict_hit(tmp_path: Path) -> None:
    identity = _identity()
    current = (_symbol("run"), _symbol("work"))
    available = frozenset(current)

    miss = select_profile_plan(tmp_path, identity, current, available)
    hit = select_profile_plan(tmp_path, identity, tuple(reversed(current)), available)

    assert miss.status == "miss"
    assert miss.selection == current
    assert miss.diagnostic == "absent"
    assert hit.status == "hit"
    assert hit.selection == current
    assert hit.diagnostic == "strict-hit"
    assert hit.cache_path == miss.cache_path
    assert hit.identity_digest == miss.identity_digest
    assert miss.cache_path.parent == tmp_path / "profile-plans"
    assert miss.cache_path.name == f"{miss.identity_digest}.json"
    text = miss.cache_path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert text.strip() == json.dumps(json.loads(text), sort_keys=True, separators=(",", ":"))


@pytest.mark.parametrize(
    "changed",
    [
        replace(_BASE_IDENTITY, scope_identity="scope-b"),
        replace(_BASE_IDENTITY, candidate_identity=("candidate-b",)),
        replace(_BASE_IDENTITY, module_source_hashes=(("app.worker", "hash-b"),)),
        replace(_BASE_IDENTITY, benchmark_argv=("python", "-m", "bench.other")),
        replace(_BASE_IDENTITY, backend_order=("cython", "mypyc")),
        replace(_BASE_IDENTITY, module_scope=None),
        replace(_BASE_IDENTITY, python_cache_tag="cpython-313"),
        replace(_BASE_IDENTITY, python_platform="other-platform"),
        replace(_BASE_IDENTITY, cache_format_version="2"),
        replace(_BASE_IDENTITY, lowering_version="2"),
    ],
)
def test_every_identity_component_invalidates_the_entry_path(
    tmp_path: Path, changed: ProfilePlanIdentity
) -> None:
    identity = _identity()
    current = (_symbol("run"),)
    baseline = select_profile_plan(tmp_path, identity, current, frozenset(current))
    result = select_profile_plan(tmp_path, changed, current, frozenset(current))

    assert result.status == "miss"
    assert result.diagnostic == "absent"
    assert result.cache_path != baseline.cache_path


@pytest.mark.parametrize(
    ("mutation", "diagnostic"),
    [
        ("malformed-json", "replaced:malformed-json"),
        ("non-object", "replaced:malformed-entry"),
        ("extra-field", "replaced:malformed-entry"),
        ("identity-not-object", "replaced:malformed-entry"),
        ("identity-extra-field", "replaced:malformed-entry"),
        ("selection-not-list", "replaced:malformed-entry"),
        ("digest-not-string", "replaced:malformed-entry"),
        ("digest", "replaced:digest-mismatch"),
        ("format-version", "replaced:version-mismatch"),
        ("lowering-version", "replaced:version-mismatch"),
        ("identity", "replaced:identity-mismatch"),
        ("empty", "replaced:empty-selection"),
        ("duplicate", "replaced:duplicate-selection"),
        ("malformed-symbol", "replaced:malformed-entry"),
        ("non-object-symbol", "replaced:malformed-entry"),
        ("empty-symbol", "replaced:malformed-entry"),
    ],
)
def test_invalid_entries_are_replaced(tmp_path: Path, mutation: str, diagnostic: str) -> None:
    identity = _identity()
    old = (_symbol("run"),)
    current = (_symbol("work"),)
    first = select_profile_plan(tmp_path, identity, old, frozenset((*old, *current)))
    _mutate_entry(first.cache_path, mutation)

    result = select_profile_plan(tmp_path, identity, current, frozenset((*old, *current)))
    replay = select_profile_plan(tmp_path, identity, old, frozenset((*old, *current)))

    assert result.status == "miss"
    assert result.selection == current
    assert result.diagnostic == diagnostic
    assert replay.status == "hit"
    assert replay.selection == current


def test_unavailable_stored_symbol_is_replaced(tmp_path: Path) -> None:
    identity = _identity()
    unavailable = _symbol("old")
    current = (_symbol("run"),)
    first = select_profile_plan(
        tmp_path, identity, (unavailable,), frozenset((unavailable, *current))
    )

    result = select_profile_plan(tmp_path, identity, current, frozenset(current))

    assert result.status == "miss"
    assert result.selection == current
    assert result.diagnostic == "replaced:unavailable-symbol"
    assert first.cache_path == result.cache_path


def test_replacement_uses_same_directory_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replacement publishes a complete entry from a temporary sibling path."""
    identity = _identity()
    current = (_symbol("run"),)
    first = select_profile_plan(tmp_path, identity, current, frozenset(current))
    first.cache_path.write_text("broken", encoding="utf-8")
    original_replace = Path.replace
    observed: list[tuple[Path, Path, bool]] = []

    def observe_replace(source: Path, target: Path) -> Path:
        observed.append((source, target, target.exists()))
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", observe_replace)
    result = select_profile_plan(tmp_path, identity, current, frozenset(current))

    assert result.diagnostic == "replaced:malformed-json"
    assert len(observed) == 1
    source, target, target_existed = observed[0]
    assert source.parent == target.parent
    assert source.suffix == ".tmp"
    assert target == first.cache_path
    assert target_existed is True
    assert not source.exists()


def test_read_failure_has_deterministic_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = _identity()
    current = (_symbol("run"),)
    first = select_profile_plan(tmp_path, identity, current, frozenset(current))
    original_read_text = Path.read_text
    calls = 0

    def fail_first_read(path: Path, encoding: str | None = None, errors: str | None = None) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("platform-dependent detail")
        return original_read_text(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", fail_first_read)
    result = select_profile_plan(tmp_path, identity, current, frozenset(current))

    assert result.status == "miss"
    assert result.diagnostic == "replaced:io-error"
    assert result.cache_path == first.cache_path


def test_empty_current_selection_never_reads_or_writes(tmp_path: Path) -> None:
    identity = _identity()
    symbol = _symbol("run")
    stored = select_profile_plan(tmp_path, identity, (symbol,), frozenset((symbol,)))
    before = stored.cache_path.read_bytes()

    disabled = select_profile_plan(tmp_path, identity, (), frozenset())

    assert disabled.status == "disabled"
    assert disabled.selection == ()
    assert disabled.diagnostic == "empty-current-selection"
    assert disabled.cache_path == stored.cache_path
    assert stored.cache_path.read_bytes() == before


def test_empty_current_selection_does_not_create_cache_root(tmp_path: Path) -> None:
    cache_root = tmp_path / "absent"

    result = select_profile_plan(cache_root, _identity(), (), frozenset())

    assert result.status == "disabled"
    assert not cache_root.exists()


def test_invalid_current_selection_is_rejected_before_persistence(tmp_path: Path) -> None:
    symbol = _symbol("run")

    with pytest.raises(ValueError, match="duplicates"):
        select_profile_plan(tmp_path, _identity(), (symbol, symbol), frozenset((symbol,)))
    with pytest.raises(ValueError, match="unavailable"):
        select_profile_plan(tmp_path, _identity(), (symbol,), frozenset())

    assert not tmp_path.joinpath("profile-plans").exists()


def test_identity_rejects_unsorted_source_hashes() -> None:
    with pytest.raises(ValueError, match="sorted and unique"):
        replace(
            _identity(),
            module_source_hashes=(("app.z", "z"), ("app.a", "a")),
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("scope", "scope_identity is empty"),
        ("module-scope", "module_scope is empty"),
        ("candidates-empty", "candidate_identity is empty"),
        ("candidates-duplicate", "candidate_identity contains duplicates"),
        ("benchmark", "benchmark_argv is empty"),
        ("backends", "backend_order is empty"),
        ("sources-empty", "module_source_hashes is empty"),
        ("source-value", "module source hash is empty"),
    ],
)
def test_identity_rejects_empty_or_duplicate_components(
    case: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _invalid_identity(case)


def test_cache_refuses_to_replace_symlink_entry(tmp_path: Path) -> None:
    symbol = _symbol("run")
    first = select_profile_plan(tmp_path, _identity(), (symbol,), frozenset((symbol,)))
    target = tmp_path / "external.json"
    target.write_text("broken", encoding="utf-8")
    first.cache_path.unlink()
    first.cache_path.symlink_to(target)

    with pytest.raises(ValueError, match="refusing to replace symlink"):
        select_profile_plan(tmp_path, _identity(), (symbol,), frozenset((symbol,)))

    assert target.read_text(encoding="utf-8") == "broken"


def test_failed_atomic_replace_removes_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_replace(_source: Path, _target: Path) -> Path:
        raise OSError("fixture replace failure")

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError, match="fixture replace failure"):
        select_profile_plan(
            tmp_path,
            _identity(),
            (_symbol("run"),),
            frozenset((_symbol("run"),)),
        )

    assert not tuple(tmp_path.joinpath("profile-plans").glob("*.tmp"))


def test_identity_allows_repeated_benchmark_arguments() -> None:
    identity = replace(_identity(), benchmark_argv=("python", "bench.py", "--flag", "--flag"))

    assert identity.benchmark_argv[-2:] == ("--flag", "--flag")


def _identity() -> ProfilePlanIdentity:
    return _BASE_IDENTITY


def _invalid_identity(case: str) -> ProfilePlanIdentity:
    try:
        factory = _INVALID_IDENTITY_FACTORIES[case]
    except KeyError as error:
        raise AssertionError(f"unknown invalid identity case: {case}") from error
    return factory()


def _symbol(qualname: str) -> SymbolId:
    return SymbolId(module="app.worker", qualname=qualname)


def _mutate_entry(path: Path, mutation: str) -> None:
    if mutation == "malformed-json":
        path.write_text("{broken", encoding="utf-8")
        return
    if mutation == "non-object":
        path.write_text("[]", encoding="utf-8")
        return
    raw = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    _mutate_valid_entry(raw, mutation)
    path.write_text(json.dumps(raw), encoding="utf-8")


def _mutate_valid_entry(raw: dict[str, object], mutation: str) -> None:
    if _mutate_entry_shape(raw, mutation):
        return
    identity = cast(dict[str, object], raw["identity"])
    selection = cast(list[object], raw["selection"])
    _mutate_entry_content(identity, selection, mutation)
    raw["digest"] = _entry_digest(raw)


def _mutate_entry_shape(raw: dict[str, object], mutation: str) -> bool:
    if mutation == "extra-field":
        raw["extra"] = True
        return True
    if mutation == "digest":
        raw["digest"] = "0" * 64
        return True
    if mutation == "identity-not-object":
        raw["identity"] = []
        raw["digest"] = _entry_digest(raw)
        return True
    if mutation == "selection-not-list":
        raw["selection"] = {}
        return True
    if mutation == "digest-not-string":
        raw["digest"] = 1
        return True
    return False


def _mutate_entry_content(
    identity: dict[str, object],
    selection: list[object],
    mutation: str,
) -> None:
    if mutation == "identity-extra-field":
        identity["extra"] = True
    elif mutation == "format-version":
        identity["cache_format_version"] = "old"
    elif mutation == "lowering-version":
        identity["lowering_version"] = "old"
    elif mutation == "identity":
        identity["scope_identity"] = "other"
    elif mutation == "empty":
        selection.clear()
    elif mutation == "duplicate":
        selection.append(selection[0])
    elif mutation == "malformed-symbol":
        selection[0] = {"module": "app.worker"}
    elif mutation == "non-object-symbol":
        selection[0] = "app.worker::run"
    elif mutation == "empty-symbol":
        selection[0] = {"module": "", "qualname": "run"}


def _entry_digest(entry: dict[str, object]) -> str:
    unsigned = {"identity": entry["identity"], "selection": entry["selection"]}
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
