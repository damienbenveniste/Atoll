"""Run the disposable Pydantic Graph optimization-ceiling experiment."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from scripts.pydantic_graph_ceiling import (
    CeilingExperimentError,
    CeilingExperimentOptions,
    run_ceiling_experiment,
)


@dataclass(frozen=True, slots=True)
class _CliOptions:
    checkout: Path
    evidence_root: Path
    warmups: int
    samples: int


def main(argv: tuple[str, ...] | None = None) -> int:
    """Prepare an isolated interpreter, run the experiment, and print its verdict."""
    args = _parse_args(tuple(sys.argv[1:] if argv is None else argv))
    checkout = args.checkout.resolve()
    evidence_root = args.evidence_root.resolve()
    atoll_root = Path(__file__).resolve().parents[1]
    workload = atoll_root / "benchmarks" / "pydantic_graph" / "workload.py.in"
    semantic_probe = atoll_root / "benchmarks" / "pydantic_graph" / "ceiling_semantics.py.in"
    try:
        python = _prepare_environment(checkout, evidence_root.parent / f"{evidence_root.name}-venv")
        result = run_ceiling_experiment(
            CeilingExperimentOptions(
                checkout=checkout,
                evidence_root=evidence_root,
                workload=workload,
                semantic_probe=semantic_probe,
                python=python,
                warmups=args.warmups,
                samples=args.samples,
            )
        )
    except (CeilingExperimentError, OSError, subprocess.SubprocessError, ValueError) as error:
        print(f"Pydantic Graph ceiling experiment failed: {error}", file=sys.stderr)
        return 1
    print(
        "Pydantic Graph ceiling experiment completed: "
        f"{result.observed_headroom:.3f}x unsafe scheduler ceiling; "
        "recommendation="
        f"{'investigate-guarded-design' if result.promising_research_direction else 'stop'}."
    )
    print(f"Reports: {result.report_json}, {result.report_markdown}")
    return 0


def _parse_args(argv: tuple[str, ...]) -> _CliOptions:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--samples", type=int, default=7)
    namespace = parser.parse_args(argv)
    return _CliOptions(
        checkout=namespace.checkout,
        evidence_root=namespace.evidence_root,
        warmups=namespace.warmups,
        samples=namespace.samples,
    )


def _prepare_environment(checkout: Path, environment_root: Path) -> Path:
    """Create one target environment outside checkout and payload directories."""
    if environment_root.exists():
        raise CeilingExperimentError(f"experiment environment already exists: {environment_root}")
    uv = shutil.which("uv")
    if uv is None:
        raise CeilingExperimentError("uv is required for the ceiling experiment")
    subprocess.run(
        (uv, "venv", "--python", "3.12", str(environment_root)),
        check=True,
    )
    python = environment_root / "bin" / "python"
    subprocess.run(
        (
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "--editable",
            str((checkout / "pydantic_graph").resolve()),
        ),
        check=True,
    )
    return python


if __name__ == "__main__":
    raise SystemExit(main())
