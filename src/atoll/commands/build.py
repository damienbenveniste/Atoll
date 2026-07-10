"""Implementation of the `atoll build` command.

The build command regenerates enabled sidecars, optionally clears Atoll build
state, and compiles sidecar sources into the project-local artifact directory.
It does not modify user source except through the generation step's managed
sidecar outputs.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from atoll.backends.mypyc import build_sidecars
from atoll.commands.generate import GenerateOptions, execute_generate
from atoll.models import CompileAttempt
from atoll.project import discover_project


@dataclass(frozen=True, slots=True)
class BuildOptions:
    """User-facing options for building enabled sidecars in place.

    Attributes:
        root: Root directory of the target Python project.
        module_name: Importable module name used to restrict the command.
        clean_first: Whether stale native outputs are removed before building.
    """

    root: Path
    module_name: str | None = None
    clean_first: bool = False


def execute_build(options: BuildOptions) -> CompileAttempt:
    """Compile enabled Atoll sidecars after refreshing generated source.

    `module_name` narrows the build to one configured source module. When
    `clean_first` is set, only Atoll's build directory is removed before
    compilation; source files and configuration are left intact.

    Args:
        options: Validated command options supplied by the CLI layer.

    Returns:
        CompileAttempt: Native compilation attempt for all selected generated sidecars.
    """
    project = discover_project(options.root)
    build_dir = project.config.root / ".atoll" / "build"
    if options.clean_first and build_dir.exists():
        shutil.rmtree(build_dir)
    execute_generate(
        GenerateOptions(
            root=project.config.root,
            module_name=options.module_name,
        )
    )
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
