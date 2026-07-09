"""Subprocess integration for mypy diagnostics.

Atoll uses mypy as a readiness signal for mypyc compilation. This module runs
mypy through its public API, parses line-oriented diagnostics, and resolves paths
before the analysis layer maps errors back to scanned symbols.
"""

from __future__ import annotations

import re
from contextlib import chdir
from dataclasses import dataclass
from pathlib import Path

from mypy import api as mypy_api

from atoll.models import DiagnosticSeverity, MypyDiagnostic, ProjectConfig

_MYPY_LINE_RE = re.compile(
    r"^(?P<path>.*?):(?P<line>\d+)(?::(?P<column>\d+))?: "
    r"(?P<severity>error|note): (?P<message>.*?)(?:  \[(?P<code>[^\]]+)\])?$"
)


@dataclass(frozen=True, slots=True)
class MypyRun:
    """Raw and parsed result from a mypy invocation.

    The return code and captured streams are preserved for diagnostics, while
    `diagnostics` contains only lines that match mypy's expected diagnostic
    format.
    """

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    diagnostics: tuple[MypyDiagnostic, ...]


def run_mypy(config: ProjectConfig) -> MypyRun:
    """Run mypy over configured source roots and parse diagnostics.

    If the project root contains `pyproject.toml`, that file is passed as the
    mypy config. The working directory is temporarily changed to the project root
    so relative diagnostic paths can be resolved consistently.
    """
    args = (
        *(str(path) for path in config.source_roots),
        "--show-column-numbers",
        "--show-error-codes",
        "--no-error-summary",
        "--no-color-output",
    )
    command_args = [*args]
    config_file = config.root / "pyproject.toml"
    if config_file.exists():
        command_args.extend(("--config-file", str(config_file)))
    with chdir(config.root):
        stdout, stderr, returncode = mypy_api.run(command_args)
    diagnostics = parse_mypy_output(stdout, cwd=config.root)
    return MypyRun(
        command=("mypy", *command_args),
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        diagnostics=diagnostics,
    )


def parse_mypy_output(output: str, *, cwd: Path) -> tuple[MypyDiagnostic, ...]:
    """Parse mypy's line-oriented diagnostic output.

    Non-diagnostic lines are ignored so callers can pass full stdout safely.
    Returned paths are absolute and columns are optional because mypy may omit
    column numbers for some messages.
    """
    diagnostics: list[MypyDiagnostic] = []
    for line in output.splitlines():
        match = _MYPY_LINE_RE.match(line)
        if match is None:
            continue
        diagnostics.append(
            MypyDiagnostic(
                path=_resolve_diagnostic_path(cwd, match.group("path")),
                line=int(match.group("line")),
                column=_optional_int(match.group("column")),
                severity=_diagnostic_severity(match.group("severity")),
                code=match.group("code"),
                message=match.group("message"),
            )
        )
    return tuple(diagnostics)


def _resolve_diagnostic_path(cwd: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (cwd / path).resolve()


def _optional_int(value: str | None) -> int | None:
    return int(value) if value is not None else None


def _diagnostic_severity(value: str) -> DiagnosticSeverity:
    return "note" if value == "note" else "error"
