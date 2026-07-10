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
    """User-facing options for runtime verification scope and strictness.

    Attributes:
        root: Root directory of the target Python project.
        module_name: Importable module name used to restrict the command.
        require_compiled: Whether interpreted fallback fails verification.
    """

    root: Path
    module_name: str | None = None
    require_compiled: bool = False


def execute_verify(options: VerifyOptions) -> tuple[VerifyResult, ...]:
    """Verify enabled Atoll shims for all or one configured source module.

    Args:
        options: Validated command options supplied by the CLI layer.

    Returns:
        tuple[VerifyResult, ...]: Runtime routing results for matching enabled islands.
    """
    project = discover_project(options.root)
    return verify_islands(
        project.config,
        module_name=options.module_name,
        require_compiled=options.require_compiled,
    )
