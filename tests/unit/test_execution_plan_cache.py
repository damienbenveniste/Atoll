"""Tests for the execution-plan helper cache."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import cast

import pytest

from atoll.execution_plans import cache as execution_plan_cache
from atoll.execution_plans.cache import (
    restore_execution_plan_cache,
    store_execution_plan_cache,
)
from atoll.execution_plans.models import (
    ChangedPayloadFile,
    ExecutionPlan,
    ExecutionPlanStageContext,
    PlanGuard,
    PlanNode,
    StagedExecutionPlan,
)
from atoll.models import SymbolId

_MetadataMutation = Callable[[dict[str, object]], None]
_SECOND_WRITE = 2
_RESTORE_AND_ROLLBACK_WRITES = 3
_write_file_atomic = cast(
    Callable[[Path, bytes], None],
    vars(execution_plan_cache)["_write_file_atomic"],
)


def _set_backend_mismatch(metadata: dict[str, object]) -> None:
    metadata["backend"] = "other"


def _set_required_imports_scalar(metadata: dict[str, object]) -> None:
    metadata["required_imports"] = "bad"


def _set_required_imports_item(metadata: dict[str, object]) -> None:
    metadata["required_imports"] = [1]


def _set_guard_scalar(metadata: dict[str, object]) -> None:
    metadata["guards"] = ["bad"]


def _set_guard_incomplete(metadata: dict[str, object]) -> None:
    metadata["guards"] = [{"kind": "semantics"}]


def _set_guard_invalid_value(metadata: dict[str, object]) -> None:
    metadata["guards"] = [{"kind": "bad", "expression": "guard", "message": "message"}]


def _set_payload_file_scalar(metadata: dict[str, object]) -> None:
    metadata["payload_files"] = ["bad"]


def _set_before_hash_invalid(metadata: dict[str, object]) -> None:
    _first_payload_file(metadata)["before_hash"] = "bad"


def _set_after_hash_invalid(metadata: dict[str, object]) -> None:
    _first_payload_file(metadata)["after_hash"] = "bad"


def _set_role_empty(metadata: dict[str, object]) -> None:
    _first_payload_file(metadata)["role"] = ""


def _first_payload_file(metadata: dict[str, object]) -> dict[str, object]:
    payload_files = metadata["payload_files"]
    assert isinstance(payload_files, list)
    first = cast(list[object], payload_files)[0]
    assert isinstance(first, dict)
    return cast(dict[str, object], first)


_METADATA_MUTATIONS: tuple[tuple[_MetadataMutation, str], ...] = (
    (_set_backend_mismatch, "backend mismatch"),
    (_set_required_imports_scalar, "required_imports"),
    (_set_required_imports_item, "required import"),
    (_set_guard_scalar, "guard"),
    (_set_guard_incomplete, "guard"),
    (_set_guard_invalid_value, "guard"),
    (_set_payload_file_scalar, "payload file"),
    (_set_before_hash_invalid, "before digest"),
    (_set_after_hash_invalid, "after digest"),
    (_set_role_empty, "role is empty"),
)


def test_execution_plan_cache_restores_hit_without_checkout_writes(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    payload_root = tmp_path / "payload"
    cache_root = tmp_path / "caller-cache"
    checkout_file = project_root / "src" / "app" / "worker.py"
    checkout_file.parent.mkdir(parents=True)
    checkout_file.write_text("checkout\n", encoding="utf-8")
    staged = _staged_plan(payload_root, "optimized\n")
    context = _context(project_root, payload_root)

    state = store_execution_plan_cache(cache_root, context, staged, fingerprint="fingerprint-a")
    (payload_root / "app" / "worker.py").unlink()
    result = restore_execution_plan_cache(
        cache_root,
        context,
        staged.plan,
        backend=staged.backend,
        fingerprint="fingerprint-a",
    )

    assert result.status == "hit"
    assert result.state == state
    assert result.staged == staged
    assert (payload_root / "app" / "worker.py").read_text(encoding="utf-8") == "optimized\n"
    assert checkout_file.read_text(encoding="utf-8") == "checkout\n"


def test_execution_plan_cache_reports_miss_for_absent_entry(tmp_path: Path) -> None:
    plan = _plan()
    result = restore_execution_plan_cache(
        tmp_path / "cache",
        _context(tmp_path / "project", tmp_path / "payload"),
        plan,
        backend="callback",
        fingerprint="missing",
    )

    assert result.status == "miss"
    assert result.staged is None
    assert result.reason == "absent"


def test_execution_plan_cache_refuses_to_overwrite_an_unexpected_payload_file(
    tmp_path: Path,
) -> None:
    """Generated cache output cannot replace a file absent from the staged baseline."""
    payload_root = tmp_path / "payload"
    cache_root = tmp_path / "cache"
    staged = _staged_plan(payload_root, "optimized\n")
    context = _context(tmp_path / "project", payload_root)
    store_execution_plan_cache(cache_root, context, staged, fingerprint="fingerprint-a")
    target = payload_root / "app" / "worker.py"
    target.write_text("unexpected\n", encoding="utf-8")

    result = restore_execution_plan_cache(
        cache_root,
        context,
        staged.plan,
        backend=staged.backend,
        fingerprint="fingerprint-a",
    )

    assert result.status == "invalid"
    assert "already exists" in (result.reason or "")
    assert target.read_text(encoding="utf-8") == "unexpected\n"


def test_execution_plan_cache_rejects_corrupt_metadata(tmp_path: Path) -> None:
    payload_root = tmp_path / "payload"
    cache_root = tmp_path / "cache"
    staged = _staged_plan(payload_root, "optimized\n")
    context = _context(tmp_path / "project", payload_root)
    state = store_execution_plan_cache(cache_root, context, staged, fingerprint="fingerprint-a")
    state.metadata_path.write_text("{not-json", encoding="utf-8")

    result = restore_execution_plan_cache(
        cache_root,
        context,
        staged.plan,
        backend=staged.backend,
        fingerprint="fingerprint-a",
    )

    assert result.status == "invalid"
    assert result.staged is None


@pytest.mark.parametrize(("mutation", "reason"), _METADATA_MUTATIONS)
def test_execution_plan_cache_rejects_invalid_metadata_shapes(
    tmp_path: Path,
    mutation: _MetadataMutation,
    reason: str,
) -> None:
    payload_root = tmp_path / "payload"
    cache_root = tmp_path / "cache"
    staged = _staged_plan(payload_root, "optimized\n")
    context = _context(tmp_path / "project", payload_root)
    state = store_execution_plan_cache(cache_root, context, staged, fingerprint="fingerprint-a")
    metadata = cast(dict[str, object], json.loads(state.metadata_path.read_text(encoding="utf-8")))
    mutation(metadata)
    state.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    result = restore_execution_plan_cache(
        cache_root,
        context,
        staged.plan,
        backend=staged.backend,
        fingerprint="fingerprint-a",
    )

    assert result.status == "invalid"
    assert reason in (result.reason or "")


def test_execution_plan_cache_rejects_non_object_metadata(tmp_path: Path) -> None:
    payload_root = tmp_path / "payload"
    cache_root = tmp_path / "cache"
    staged = _staged_plan(payload_root, "optimized\n")
    context = _context(tmp_path / "project", payload_root)
    state = store_execution_plan_cache(cache_root, context, staged, fingerprint="fingerprint-a")
    state.metadata_path.write_text("[]", encoding="utf-8")

    result = restore_execution_plan_cache(
        cache_root,
        context,
        staged.plan,
        backend=staged.backend,
        fingerprint="fingerprint-a",
    )

    assert result.status == "invalid"
    assert "not an object" in (result.reason or "")


def test_execution_plan_cache_invalidates_source_backend_and_fingerprint(
    tmp_path: Path,
) -> None:
    payload_root = tmp_path / "payload"
    cache_root = tmp_path / "cache"
    staged = _staged_plan(payload_root, "optimized\n")
    context = _context(tmp_path / "project", payload_root)
    store_execution_plan_cache(cache_root, context, staged, fingerprint="fingerprint-a")

    changed_source = replace(staged.plan, source_hash="changed")
    changed_backend = restore_execution_plan_cache(
        cache_root,
        context,
        staged.plan,
        backend="other-backend",
        fingerprint="fingerprint-a",
    )
    changed_fingerprint = restore_execution_plan_cache(
        cache_root,
        context,
        staged.plan,
        backend=staged.backend,
        fingerprint="fingerprint-b",
    )
    changed_plan = restore_execution_plan_cache(
        cache_root,
        context,
        changed_source,
        backend=staged.backend,
        fingerprint="fingerprint-a",
    )

    assert changed_backend.status == "miss"
    assert changed_fingerprint.status == "miss"
    assert changed_plan.status == "miss"


def test_execution_plan_cache_rejects_path_traversal_metadata(tmp_path: Path) -> None:
    payload_root = tmp_path / "payload"
    cache_root = tmp_path / "cache"
    staged = _staged_plan(payload_root, "optimized\n")
    context = _context(tmp_path / "project", payload_root)
    state = store_execution_plan_cache(cache_root, context, staged, fingerprint="fingerprint-a")
    metadata = json.loads(state.metadata_path.read_text(encoding="utf-8"))
    metadata["payload_files"][0]["install_path"] = "../escape.py"
    state.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    result = restore_execution_plan_cache(
        cache_root,
        context,
        staged.plan,
        backend=staged.backend,
        fingerprint="fingerprint-a",
    )

    assert result.status == "invalid"
    assert "unsafe cache install path" in (result.reason or "")


def test_execution_plan_cache_rejects_payload_root_symlink_escape(tmp_path: Path) -> None:
    payload_root = tmp_path / "payload"
    cache_root = tmp_path / "cache"
    staged = _staged_plan(payload_root, "optimized\n")
    context = _context(tmp_path / "project", payload_root)
    store_execution_plan_cache(cache_root, context, staged, fingerprint="fingerprint-a")
    payload_file = payload_root / "app" / "worker.py"
    payload_file.unlink()
    payload_file.parent.rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    payload_file.parent.symlink_to(outside)

    result = restore_execution_plan_cache(
        cache_root,
        context,
        staged.plan,
        backend=staged.backend,
        fingerprint="fingerprint-a",
    )

    assert result.status == "invalid"
    assert "escapes root" in (result.reason or "")


def test_execution_plan_cache_rejects_symlink_payloads(tmp_path: Path) -> None:
    payload_root = tmp_path / "payload"
    cache_root = tmp_path / "cache"
    staged = _staged_plan(payload_root, "optimized\n")
    context = _context(tmp_path / "project", payload_root)
    state = store_execution_plan_cache(cache_root, context, staged, fingerprint="fingerprint-a")
    cached_file = state.entry_root / "payload" / "app" / "worker.py"
    cached_file.unlink()
    cached_file.symlink_to(tmp_path / "outside.py")

    result = restore_execution_plan_cache(
        cache_root,
        context,
        staged.plan,
        backend=staged.backend,
        fingerprint="fingerprint-a",
    )

    assert result.status == "invalid"
    assert "symlink" in (result.reason or "")


@pytest.mark.parametrize(
    ("destination_state", "reason"),
    [
        ("symlink", "destination is a symlink"),
        ("missing", "destination is missing"),
        ("digest-mismatch", "before digest mismatch"),
    ],
)
def test_execution_plan_cache_rejects_invalid_existing_destinations(
    tmp_path: Path,
    destination_state: str,
    reason: str,
) -> None:
    """A cache hit cannot overwrite a baseline file whose state no longer matches."""
    payload_root = tmp_path / "payload"
    cache_root = tmp_path / "cache"
    staged = _staged_existing_files(payload_root)
    context = _context(tmp_path / "project", payload_root)
    store_execution_plan_cache(cache_root, context, staged, fingerprint="fingerprint-a")
    first = payload_root / staged.payload_files[0].install_path
    for changed_file in staged.payload_files:
        (payload_root / changed_file.install_path).write_text("baseline\n", encoding="utf-8")
    if destination_state == "symlink":
        first.unlink()
        outside = tmp_path / "outside.py"
        outside.write_text("baseline\n", encoding="utf-8")
        first.symlink_to(outside)
    elif destination_state == "missing":
        first.unlink()
    else:
        first.write_text("changed\n", encoding="utf-8")

    result = restore_execution_plan_cache(
        cache_root,
        context,
        staged.plan,
        backend=staged.backend,
        fingerprint="fingerprint-a",
    )

    assert result.status == "invalid"
    assert reason in (result.reason or "")


def test_execution_plan_cache_rejects_unsafe_store_payloads(tmp_path: Path) -> None:
    payload_root = tmp_path / "payload"
    context = _context(tmp_path / "project", payload_root)
    staged = _staged_plan(payload_root, "optimized\n")
    payload_file = payload_root / "app" / "worker.py"
    outside = tmp_path / "outside.py"
    outside.write_text("outside\n", encoding="utf-8")
    payload_file.unlink()
    payload_file.symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        store_execution_plan_cache(tmp_path / "cache", context, staged, fingerprint="fingerprint-a")


def test_execution_plan_cache_rejects_store_digest_mismatch(tmp_path: Path) -> None:
    payload_root = tmp_path / "payload"
    context = _context(tmp_path / "project", payload_root)
    staged = _staged_plan(payload_root, "optimized\n")
    mismatched = replace(
        staged,
        payload_files=(
            replace(
                staged.payload_files[0],
                after_hash="0" * 64,
            ),
        ),
    )

    with pytest.raises(ValueError, match="digest mismatch"):
        store_execution_plan_cache(
            tmp_path / "cache", context, mismatched, fingerprint="fingerprint-a"
        )


def test_execution_plan_cache_rejects_incomplete_and_digest_mismatched_entries(
    tmp_path: Path,
) -> None:
    payload_root = tmp_path / "payload"
    cache_root = tmp_path / "cache"
    staged = _staged_plan(payload_root, "optimized\n")
    context = _context(tmp_path / "project", payload_root)
    state = store_execution_plan_cache(cache_root, context, staged, fingerprint="fingerprint-a")
    cached_file = state.entry_root / "payload" / "app" / "worker.py"
    cached_file.unlink()
    incomplete = restore_execution_plan_cache(
        cache_root,
        context,
        staged.plan,
        backend=staged.backend,
        fingerprint="fingerprint-a",
    )
    cached_file.parent.mkdir(parents=True, exist_ok=True)
    cached_file.write_text("corrupt\n", encoding="utf-8")
    corrupt = restore_execution_plan_cache(
        cache_root,
        context,
        staged.plan,
        backend=staged.backend,
        fingerprint="fingerprint-a",
    )

    assert incomplete.status == "invalid"
    assert "missing" in (incomplete.reason or "")
    assert corrupt.status == "invalid"
    assert "digest mismatch" in (corrupt.reason or "")


def test_execution_plan_cache_does_not_cache_transient_failures(tmp_path: Path) -> None:
    payload_root = tmp_path / "payload"
    context = _context(tmp_path / "project", payload_root)
    staged = _staged_plan(payload_root, "optimized\n")
    (payload_root / "app" / "worker.py").unlink()

    with pytest.raises(ValueError, match="missing"):
        store_execution_plan_cache(tmp_path / "cache", context, staged, fingerprint="fingerprint-a")

    result = restore_execution_plan_cache(
        tmp_path / "cache",
        context,
        staged.plan,
        backend=staged.backend,
        fingerprint="fingerprint-a",
    )
    assert result.status == "miss"


def test_execution_plan_cache_rolls_back_partial_payload_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A write failure restores every payload file changed earlier in the transaction."""
    cache_root = tmp_path / "cache"
    payload_root = tmp_path / "payload"
    context = _context(tmp_path / "project", payload_root)
    staged = _staged_existing_files(payload_root)
    store_execution_plan_cache(cache_root, context, staged, fingerprint="fingerprint-a")
    for changed_file in staged.payload_files:
        (payload_root / changed_file.install_path).write_text("baseline\n", encoding="utf-8")
    original_write = _write_file_atomic
    calls = 0

    def fail_second_write(path: Path, data: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == _SECOND_WRITE:
            raise OSError("injected restore failure")
        original_write(path, data)

    monkeypatch.setattr(execution_plan_cache, "_write_file_atomic", fail_second_write)
    result = restore_execution_plan_cache(
        cache_root,
        context,
        staged.plan,
        backend=staged.backend,
        fingerprint="fingerprint-a",
    )

    assert result.status == "invalid"
    assert "injected restore failure" in (result.reason or "")
    assert calls == _RESTORE_AND_ROLLBACK_WRITES
    assert {
        (payload_root / changed_file.install_path).read_text(encoding="utf-8")
        for changed_file in staged.payload_files
    } == {"baseline\n"}


def test_execution_plan_cache_reports_incomplete_payload_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rollback write failure is reported instead of hiding a mixed payload state."""
    cache_root = tmp_path / "cache"
    payload_root = tmp_path / "payload"
    context = _context(tmp_path / "project", payload_root)
    staged = _staged_existing_files(payload_root)
    store_execution_plan_cache(cache_root, context, staged, fingerprint="fingerprint-a")
    for changed_file in staged.payload_files:
        (payload_root / changed_file.install_path).write_text("baseline\n", encoding="utf-8")
    original_write = _write_file_atomic
    calls = 0

    def fail_restore_and_rollback(path: Path, data: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls >= _SECOND_WRITE:
            raise OSError("injected write failure")
        original_write(path, data)

    monkeypatch.setattr(execution_plan_cache, "_write_file_atomic", fail_restore_and_rollback)
    result = restore_execution_plan_cache(
        cache_root,
        context,
        staged.plan,
        backend=staged.backend,
        fingerprint="fingerprint-a",
    )

    assert result.status == "invalid"
    assert "rollback was incomplete" in (result.reason or "")
    assert calls == _RESTORE_AND_ROLLBACK_WRITES


def _context(project_root: Path, payload_root: Path) -> ExecutionPlanStageContext:
    return ExecutionPlanStageContext(
        project_root=project_root,
        payload_root=payload_root,
        cache_root=project_root / ".cache",
    )


def _staged_plan(payload_root: Path, text: str) -> StagedExecutionPlan:
    payload_file = payload_root / "app" / "worker.py"
    payload_file.parent.mkdir(parents=True)
    payload_file.write_text(text, encoding="utf-8")
    return StagedExecutionPlan(
        plan=_plan(),
        backend="callback",
        payload_files=(
            ChangedPayloadFile(
                install_path=PurePosixPath("app/worker.py"),
                before_hash=None,
                after_hash=_sha256(text),
                role="source",
            ),
        ),
        required_imports=("app.worker",),
        guards=(
            PlanGuard(
                kind="semantics",
                expression="guard",
                message="guard message",
            ),
        ),
    )


def _staged_existing_files(payload_root: Path) -> StagedExecutionPlan:
    payload_files: list[ChangedPayloadFile] = []
    for name in ("first.py", "second.py"):
        path = payload_root / "app" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("optimized\n", encoding="utf-8")
        payload_files.append(
            ChangedPayloadFile(
                install_path=PurePosixPath("app") / name,
                before_hash=_sha256("baseline\n"),
                after_hash=_sha256("optimized\n"),
                role="source",
            )
        )
    return StagedExecutionPlan(
        plan=_plan(),
        backend="callback",
        payload_files=tuple(payload_files),
        required_imports=("app.first", "app.second"),
        guards=(),
    )


def _plan() -> ExecutionPlan:
    return ExecutionPlan(
        id="exec-plan-cache-test",
        source_module="app.worker",
        owner=SymbolId(module="app.worker", qualname="run"),
        dialect="asyncio-task-group",
        lowering_version="1",
        source_hash="source-a",
        callsite_fingerprint="callsite-a",
        topology_fingerprint="topology-a",
        nodes=(
            PlanNode(
                id="app.worker::run",
                symbol=SymbolId(module="app.worker", qualname="run"),
                role="orchestrator",
                lineno=1,
            ),
        ),
        edges=(),
        guards=(),
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
