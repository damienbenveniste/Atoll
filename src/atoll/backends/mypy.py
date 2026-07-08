"""Subprocess integration for mypy diagnostics."""

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
    """Raw and parsed result from a mypy subprocess run."""

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    diagnostics: tuple[MypyDiagnostic, ...]


def run_mypy(config: ProjectConfig) -> MypyRun:
    """Run mypy over configured source roots and parse diagnostics."""
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
    """Parse mypy's line-oriented diagnostic output."""
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
