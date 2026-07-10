"""Tests for trial subprocess routing environment setup."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from atoll.runtime import test_runner

_configure_test_environment = cast(
    Callable[..., None],
    vars(test_runner)["_configure_test_environment"],
)


@pytest.mark.parametrize("require_compiled", [True, False])
def test_configure_test_environment_resets_inherited_routing_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    require_compiled: bool,
) -> None:
    """Trial routing is deterministic despite inherited Atoll control variables."""
    original_path = list(sys.path)
    monkeypatch.setenv("ATOLL_DISABLE", "1")
    monkeypatch.setenv("ATOLL_REQUIRE_COMPILED", "1")
    monkeypatch.setenv("PYTHONPATH", "/existing")
    try:
        _configure_test_environment(
            source_roots=(tmp_path,),
            require_compiled=require_compiled,
        )

        assert "ATOLL_DISABLE" not in os.environ
        assert os.environ.get("ATOLL_REQUIRE_COMPILED") == ("1" if require_compiled else None)
        assert os.environ["PYTHONPATH"].split(os.pathsep) == [str(tmp_path), "/existing"]
        assert sys.path[0] == str(tmp_path)
    finally:
        sys.path[:] = original_path
