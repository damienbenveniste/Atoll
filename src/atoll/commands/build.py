"""Implementation of the `atoll build` command."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from atoll.backends.mypyc import build_sidecars
from atoll.models import CompileAttempt
from atoll.project import discover_project


@dataclass(frozen=True, slots=True)
class BuildOptions:
    """User-facing options for building enabled sidecars."""

    root: Path
    module_name: str | None = None
    clean_first: bool = False


def execute_build(options: BuildOptions) -> CompileAttempt:
    """Compile enabled Atoll sidecars in place."""
    project = discover_project(options.root)
    build_dir = project.config.root / ".atoll" / "build"
    if options.clean_first and build_dir.exists():
        shutil.rmtree(build_dir)
    sidecars = tuple(
        island.sidecar_path
        for island in project.config.islands
        if island.enabled
        and (options.module_name is None or island.source_module == options.module_name)
    )
    return build_sidecars(
        sidecars,
        project_root=project.config.root,
        build_dir=build_dir,
        source_roots=project.config.source_roots,
    )
