"""Command-line entry point for repository-local corpus tooling.

Only manifest validation and matrix projection are owned by this initial
module.  Lifecycle, aggregation, and history handlers are imported lazily as
their milestones add them, keeping command startup network-free.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.benchmark_corpus.manifest import ManifestError, load_manifest, manifest_matrix
from scripts.benchmark_corpus.models import CorpusPlatform, CorpusTier

DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "benchmarks" / "corpus" / "manifest.toml"


def main(argv: tuple[str, ...] | None = None) -> int:
    """Validate corpus input and print machine-readable command results.

    Args:
        argv: Optional arguments replacing ``sys.argv``.

    Returns:
        int: Zero on success and two for invalid corpus input.
    """
    parser = _parser()
    args = parser.parse_args(tuple(sys.argv[1:] if argv is None else argv))
    try:
        manifest = load_manifest(args.manifest)
        if args.command == "validate":
            print(
                json.dumps(
                    {
                        "backends": list(manifest.backends),
                        "cases": len(manifest.cases),
                        "python": manifest.python_version,
                        "schema_version": manifest.schema_version,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0
        if args.command == "matrix":
            rows = manifest_matrix(
                manifest,
                tier=args.tier,
                platform=args.platform,
                case_ids=tuple(args.case),
            )
            print(
                json.dumps(
                    {"include": [row.as_json() for row in rows]},
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0
        parser.error(f"unsupported command: {args.command}")
    except ManifestError as error:
        print(f"benchmark corpus error: {error}", file=sys.stderr)
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.benchmark_corpus")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate", help="validate the corpus manifest")
    matrix = commands.add_parser("matrix", help="emit a deterministic workflow matrix")
    matrix.add_argument("--tier", choices=_tier_choices())
    matrix.add_argument("--platform", choices=_platform_choices())
    matrix.add_argument("--case", action="append", default=[])
    return parser


def _tier_choices() -> tuple[CorpusTier, ...]:
    return ("compatibility", "performance", "calibration", "negative-control")


def _platform_choices() -> tuple[CorpusPlatform, ...]:
    return ("ubuntu-24.04", "macos-14")
