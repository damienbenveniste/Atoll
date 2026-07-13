"""Command-line entry point for repository-local corpus tooling.

Validation, lock inspection, and matrix projection remain network-free.
Aggregation and promotion are added by their later delivery milestones.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from scripts.benchmark_corpus.lifecycle import LifecycleOptions, run_case
from scripts.benchmark_corpus.manifest import ManifestError, load_manifest, manifest_matrix
from scripts.benchmark_corpus.models import CorpusManifest, CorpusPlatform, CorpusTier

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
        if args.command == "lock":
            print(
                json.dumps(
                    _lock_identities(manifest, args.atoll_root, tuple(args.case)),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0
        if args.command == "run":
            summary = run_case(
                manifest,
                args.case,
                LifecycleOptions(
                    atoll_root=args.atoll_root,
                    workspace_root=args.workspace_root,
                    evidence_root=args.evidence_root,
                    tier=args.tier,
                    platform=args.platform,
                    allow_unsandboxed=args.allow_unsandboxed,
                    keep_workspace=args.keep_workspace,
                ),
            )
            print(
                json.dumps(
                    {
                        "case": summary.result.case_id,
                        "json": str(summary.json_path),
                        "markdown": str(summary.markdown_path),
                        "status": summary.result.status,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0 if summary.result.status not in _FAILED_STATUSES else 1
        parser.error(f"unsupported command: {args.command}")
    except (ManifestError, OSError, RuntimeError) as error:
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
    lock = commands.add_parser("lock", help="verify reviewed dependency lock identities")
    lock.add_argument("--case", action="append", default=[])
    lock.add_argument("--atoll-root", type=Path, default=DEFAULT_MANIFEST.parents[2])
    run = commands.add_parser("run", help="execute one isolated pinned corpus case")
    run.add_argument("case")
    run.add_argument("--tier", choices=_tier_choices(), required=True)
    run.add_argument("--platform", choices=_platform_choices(), required=True)
    run.add_argument("--atoll-root", type=Path, default=DEFAULT_MANIFEST.parents[2])
    run.add_argument("--workspace-root", type=Path, required=True)
    run.add_argument("--evidence-root", type=Path, required=True)
    run.add_argument("--allow-unsandboxed", action="store_true")
    run.add_argument("--keep-workspace", action="store_true")
    return parser


_FAILED_STATUSES = frozenset(
    {
        "upstream-broken",
        "compile-error",
        "compatibility-regression",
        "unstable",
        "timeout",
        "infrastructure-error",
        "security-violation",
    }
)


def _lock_identities(
    manifest: CorpusManifest,
    atoll_root: Path,
    case_ids: tuple[str, ...],
) -> dict[str, object]:
    selected = set(case_ids)
    known = {case.id for case in manifest.cases}
    unknown = selected - known
    if unknown:
        raise ManifestError(f"unknown corpus case(s): {', '.join(sorted(unknown))}")
    root = atoll_root.resolve(strict=True)
    locks: list[dict[str, str]] = []
    for case in manifest.cases:
        if selected and case.id not in selected:
            continue
        candidate = root.joinpath(*case.dependency_lock.parts)
        if candidate.is_symlink():
            raise ManifestError(f"unsafe dependency lock for {case.id}: {case.dependency_lock}")
        path = candidate.resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ManifestError(f"unsafe dependency lock for {case.id}: {case.dependency_lock}")
        locks.append(
            {
                "case": case.id,
                "path": case.dependency_lock.as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return {"locks": locks, "schema_version": manifest.schema_version}


def _tier_choices() -> tuple[CorpusTier, ...]:
    return ("compatibility", "performance", "calibration", "negative-control")


def _platform_choices() -> tuple[CorpusPlatform, ...]:
    return ("ubuntu-24.04", "macos-14")
