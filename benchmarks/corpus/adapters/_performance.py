"""Shared command boundary for reviewed corpus performance workloads.

Individual adapters select one workload by a fixed identifier.  This module
owns argument parsing, deterministic JSON encoding, and imported-payload path
reporting; package-specific benchmark behavior stays in ``../workloads`` so
its exact bytes can be recorded in the corpus manifest.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

_DEFAULT_REPETITIONS = 1
_DEFAULT_SEED = 1729
_EXPECTED_CASES = frozenset(
    {
        "anyio",
        "html5lib",
        "mako",
        "mypy",
        "networkx",
        "pydantic",
        "pydantic-graph",
        "rich",
        "sqlalchemy",
        "sqlglot",
        "sympy",
        "tomli",
    }
)


class _Workload(Protocol):
    """Interface implemented by one digest-pinned workload module."""

    def run(
        self, *, repetitions: int, seed: int
    ) -> tuple[dict[str, object], tuple[ModuleType, ...]]:
        """Execute the workload and return its canonical result and imports."""


def _load_workload(case_id: str) -> _Workload:
    """Load one reviewed workload without making the workloads directory a package."""
    filename = case_id.replace("-", "_") + ".py"
    adapter_root = Path(__file__).resolve().parent
    path = adapter_root.parent / "workloads" / filename
    spec = importlib.util.spec_from_file_location(f"corpus_workload_{case_id}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reviewed workload {case_id!r}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    original_path = sys.path[:]
    try:
        sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != adapter_root]
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_path
    return cast(_Workload, module)


def _module_paths(modules: tuple[ModuleType, ...]) -> list[str]:
    """Return stable absolute files proving which installed payload ran."""
    paths: set[str] = set()
    for module in modules:
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str) or not raw_path:
            raise RuntimeError(f"imported module {module.__name__!r} has no package file")
        paths.add(str(Path(raw_path).resolve(strict=True)))
    if not paths:
        raise RuntimeError("performance workload reported no imported project modules")
    return sorted(paths)


def _golden_result(case_id: str) -> dict[str, object]:
    """Load the reviewed result for one default seeded workload.

    Args:
        case_id: Fixed performance case selected by the executable adapter.

    Returns:
        dict[str, object]: Exact result observed from the pinned default run.

    Raises:
        RuntimeError: If the golden file is malformed, incomplete, or contains
            an unreviewed case.
    """
    path = Path(__file__).resolve().parent.parent / "workloads" / "golden.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read performance golden file: {error}") from error
    if not isinstance(payload, dict):
        raise TypeError("performance golden payload must be a JSON object")
    if payload.get("repetitions") != _DEFAULT_REPETITIONS or payload.get("seed") != _DEFAULT_SEED:
        raise RuntimeError("performance golden defaults do not match adapter defaults")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, dict) or set(raw_cases) != _EXPECTED_CASES:
        raise RuntimeError("performance golden cases do not exactly match reviewed adapters")
    result = raw_cases.get(case_id)
    if not isinstance(result, dict):
        raise TypeError(f"performance golden result is missing for {case_id!r}")
    return cast(dict[str, object], result)


def _verify_default_result(
    case_id: str,
    *,
    repetitions: int,
    seed: int,
    result: dict[str, object],
) -> None:
    """Reject semantic drift in the default benchmark command."""
    if repetitions != _DEFAULT_REPETITIONS or seed != _DEFAULT_SEED:
        return
    expected = _golden_result(case_id)
    expected_json = json.dumps(expected, sort_keys=True, separators=(",", ":"))
    observed_json = json.dumps(result, sort_keys=True, separators=(",", ":"))
    if observed_json != expected_json:
        raise RuntimeError(
            f"default canonical mismatch for {case_id}: "
            f"expected {expected_json}; observed {observed_json}"
        )


def main(case_id: str, argv: tuple[str, ...] | None = None) -> int:
    """Run one fixed workload and print exactly one canonical JSON object.

    Args:
        case_id: Manifest case selected by the executable adapter.
        argv: Optional arguments used by tests instead of process arguments.

    Returns:
        int: Zero after the canonical result is written.
    """
    parser = argparse.ArgumentParser(description=f"Run the {case_id} corpus workload")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=_DEFAULT_REPETITIONS)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    arguments = parser.parse_args(argv)
    if arguments.repetitions <= 0:
        parser.error("--repetitions must be positive")
    if not arguments.project_root.is_dir():
        parser.error("--project-root must be an existing directory")

    workload = _load_workload(case_id)
    canonical, modules = workload.run(
        repetitions=arguments.repetitions,
        seed=arguments.seed,
    )
    _verify_default_result(
        case_id,
        repetitions=arguments.repetitions,
        seed=arguments.seed,
        result=canonical,
    )
    payload = {
        "canonical": {
            "case": case_id,
            "repetitions": arguments.repetitions,
            "seed": arguments.seed,
            "result": canonical,
        },
        "imports": _module_paths(modules),
    }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0
