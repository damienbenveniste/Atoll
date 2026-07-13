"""Smoke tests for the residual async profile fixture."""

from __future__ import annotations

import asyncio
import importlib
import sys
from collections.abc import Coroutine
from pathlib import Path
from typing import Protocol, cast

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"


class FixtureModule(Protocol):
    """Loaded residual profile fixture surface used by smoke tests."""

    def compare_semantics(
        self,
        repetitions: int,
    ) -> Coroutine[object, object, tuple[dict[str, object], bool]]:
        """Return an awaitable semantic comparison result."""
        ...

    def context_sensitive_fallback_snapshot(
        self,
    ) -> Coroutine[object, object, dict[str, object]]:
        """Return an awaitable fallback snapshot."""
        ...


def test_residual_profile_semantics_match_baseline() -> None:
    module = _fixture_module()
    snapshot, matched = asyncio.run(module.compare_semantics(2))

    assert matched
    assert snapshot["workflow"] == "generic-residual-async-profile"


def test_context_sensitive_work_retains_task_fallback() -> None:
    module = _fixture_module()
    snapshot = asyncio.run(module.context_sensitive_fallback_snapshot())

    assert snapshot == {
        "parent_before": "owner",
        "child_observed": "owner",
        "child_mutated": "child:context-sensitive",
        "parent_after": "owner",
    }


def _fixture_module() -> FixtureModule:
    source_root = str(SOURCE_ROOT.resolve())
    sys.path.insert(0, source_root)
    try:
        return cast(FixtureModule, importlib.import_module("residual_async_profile"))
    finally:
        sys.path.remove(source_root)
