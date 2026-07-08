"""Implementation of the `atoll verify` command."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from atoll.models import VerifyResult
from atoll.project import discover_project
from atoll.runtime.verify import verify_islands


@dataclass(frozen=True, slots=True)
class VerifyOptions:
    """User-facing options for runtime verification."""

    root: Path
    module_name: str | None = None
    require_compiled: bool = False


def execute_verify(options: VerifyOptions) -> tuple[VerifyResult, ...]:
    """Verify enabled Atoll shims."""
    project = discover_project(options.root)
    return verify_islands(
        project.config,
        module_name=options.module_name,
        require_compiled=options.require_compiled,
    )
