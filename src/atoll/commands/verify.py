"""Implementation of the `atoll verify` command.

Verify imports enabled source modules and checks the managed shim status without
building anything. It can optionally require compiled extension routing instead
of accepting pure-Python sidecars.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from atoll.models import VerifyResult
from atoll.project import discover_project
from atoll.runtime.verify import verify_islands


@dataclass(frozen=True, slots=True)
class VerifyOptions:
    """User-facing options for runtime verification scope and strictness."""

    root: Path
    module_name: str | None = None
    require_compiled: bool = False


def execute_verify(options: VerifyOptions) -> tuple[VerifyResult, ...]:
    """Verify enabled Atoll shims for all or one configured source module."""
    project = discover_project(options.root)
    return verify_islands(
        project.config,
        module_name=options.module_name,
        require_compiled=options.require_compiled,
    )
