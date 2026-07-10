"""Run target-project pytest gates with Atoll routing controls.

Trial mode uses this module to execute supported pytest commands in a spawned
child process. The child receives an adjusted `PYTHONPATH` and optional
`ATOLL_REQUIRE_COMPILED` flag so tests exercise the staged install payload.
"""

from __future__ import annotations

import importlib
import os
import shlex
import sys
from multiprocessing import get_context
from pathlib import Path

from atoll.models import PytestRunResult


def run_pytest_command(
    command: str,
    *,
    root: Path,
    source_roots: tuple[Path, ...],
    require_compiled: bool,
) -> PytestRunResult:
    """Run a supported pytest command in a spawned target-project process.

    The parent process receives only the normalized command and exit code; pytest
    output remains attached to the child process streams.
    """
    command_parts = parse_pytest_command(command)
    context = get_context("spawn")
    process = context.Process(
        target=_run_pytest_child,
        args=(
            tuple(_pytest_args(command_parts)),
            str(root),
            tuple(str(path) for path in source_roots),
            require_compiled,
        ),
    )
    process.start()
    process.join()
    exit_code = process.exitcode if process.exitcode is not None else 1
    return PytestRunResult(
        command=command_parts,
        exit_code=exit_code,
        success=exit_code == 0,
    )


def parse_pytest_command(command: str) -> tuple[str, ...]:
    """Parse and validate a supported pytest command string.

    Only `pytest ...` and `python -m pytest ...` forms are accepted so trial mode
    does not execute arbitrary shell commands.
    """
    command_parts = tuple(shlex.split(command))
    _pytest_args(command_parts)
    return command_parts


def _pytest_args(command_parts: tuple[str, ...]) -> list[str]:
    if command_parts[:1] == ("pytest",):
        return list(command_parts[1:])
    if command_parts[:3] == ("python", "-m", "pytest"):
        return list(command_parts[3:])
    raise ValueError("test gate currently supports pytest commands only")


def _run_pytest_child(
    args: tuple[str, ...],
    root: str,
    source_roots: tuple[str, ...],
    require_compiled: bool,
) -> None:
    _configure_test_environment(source_roots=source_roots, require_compiled=require_compiled)
    os.chdir(root)
    pytest = importlib.import_module("pytest")
    raise SystemExit(int(pytest.main(list(args))))


def _configure_test_environment(
    *,
    source_roots: tuple[Path | str, ...],
    require_compiled: bool,
) -> None:
    original_pythonpath = os.environ.get("PYTHONPATH")
    existing_pythonpath = tuple(
        path for path in (original_pythonpath or "").split(os.pathsep) if path
    )
    source_root_texts = tuple(str(path) for path in source_roots)
    sys.path[:0] = list(source_root_texts)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [
            *source_root_texts,
            *existing_pythonpath,
        ]
    )
    os.environ.pop("ATOLL_DISABLE", None)
    if require_compiled:
        os.environ["ATOLL_REQUIRE_COMPILED"] = "1"
    else:
        os.environ.pop("ATOLL_REQUIRE_COMPILED", None)
