"""Command-line entry point for repository-local corpus tooling.

Validation, lock inspection, and matrix projection remain network-free.
Aggregation and promotion are added by their later delivery milestones.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from packaging.requirements import InvalidRequirement, Requirement

from scripts.benchmark_corpus.lifecycle import LifecycleOptions, run_case
from scripts.benchmark_corpus.manifest import ManifestError, load_manifest, manifest_matrix
from scripts.benchmark_corpus.models import CorpusManifest, CorpusPlatform, CorpusTier

DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "benchmarks" / "corpus" / "manifest.toml"
LOCK_EXCLUDE_NEWER = "2026-07-13T23:59:59Z"
_SHA256_HASH = re.compile(r"(?:^|\s)--hash=sha256:([0-9a-f]{64})(?=\s|$)")


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
            if args.write:
                _write_locks(manifest, args.atoll_root, tuple(args.case))
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
    lock.add_argument(
        "--write",
        action="store_true",
        help="regenerate selected locks from their reviewed .in files",
    )
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
        _validate_hashed_lock(path, case.id)
        locks.append(
            {
                "case": case.id,
                "path": case.dependency_lock.as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return {"locks": locks, "schema_version": manifest.schema_version}


def _write_locks(
    manifest: CorpusManifest,
    atoll_root: Path,
    case_ids: tuple[str, ...],
) -> None:
    """Regenerate selected hash-locked constraints with one stable uv policy.

    Args:
        manifest: Validated corpus manifest owning the selected cases.
        atoll_root: Repository root containing reviewed ``.in`` files.
        case_ids: Optional exact case allowlist.

    Raises:
        ManifestError: If a case, source path, or output path is unsafe.
        RuntimeError: If uv is unavailable or dependency resolution fails.
    """
    selected = set(case_ids)
    known = {case.id for case in manifest.cases}
    unknown = selected - known
    if unknown:
        raise ManifestError(f"unknown corpus case(s): {', '.join(sorted(unknown))}")
    root = atoll_root.resolve(strict=True)
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv executable is unavailable")
    for case in manifest.cases:
        if selected and case.id not in selected:
            continue
        output = root.joinpath(*case.dependency_lock.parts)
        source = output.with_suffix(".in")
        if source.is_symlink() or output.is_symlink():
            raise ManifestError(f"unsafe dependency lock input for {case.id}")
        resolved_source = source.resolve(strict=True)
        resolved_output = output.resolve(strict=False)
        if not resolved_source.is_relative_to(root) or not resolved_output.is_relative_to(root):
            raise ManifestError(f"unsafe dependency lock input for {case.id}")
        with TemporaryDirectory(
            prefix=f".{case.id}-lock-", dir=resolved_output.parent
        ) as temporary:
            candidate = Path(temporary) / resolved_output.name
            result = subprocess.run(
                (
                    uv,
                    "pip",
                    "compile",
                    str(resolved_source),
                    "--universal",
                    "--python-version",
                    manifest.python_version,
                    "--generate-hashes",
                    "--exclude-newer",
                    LOCK_EXCLUDE_NEWER,
                    "--custom-compile-command",
                    f"uv run python -m scripts.benchmark_corpus lock --write --case {case.id}",
                    "--output-file",
                    str(candidate),
                    "--quiet",
                ),
                cwd=root,
                shell=False,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(f"uv failed to resolve dependency lock for {case.id}")
            _validate_hashed_lock(candidate, case.id)
            candidate.replace(resolved_output)


def _validate_hashed_lock(path: Path, case_id: str) -> None:
    """Reject dependency locks that are not exact, hash-verified requirements.

    Args:
        path: Reviewed or newly generated requirements file.
        case_id: Corpus case used in actionable validation errors.

    Raises:
        ManifestError: If a requirement is unpinned, unhashed, direct, editable,
            malformed, or introduces a package-index/path directive.
    """
    statements = _logical_lock_statements(path, case_id)
    if not statements:
        raise ManifestError(f"dependency lock for {case_id} contains no requirements")
    for statement in statements:
        hashes = _SHA256_HASH.findall(statement)
        requirement_text = _SHA256_HASH.sub("", statement).strip()
        if not hashes or "--" in requirement_text or requirement_text.startswith("-"):
            raise ManifestError(
                f"dependency lock for {case_id} must contain only exact hashed requirements"
            )
        try:
            requirement = Requirement(requirement_text)
        except InvalidRequirement as error:
            raise ManifestError(
                f"dependency lock for {case_id} contains an invalid requirement"
            ) from error
        specifiers = tuple(requirement.specifier)
        exact = (
            requirement.url is None
            and len(specifiers) == 1
            and specifiers[0].operator == "=="
            and not specifiers[0].version.endswith(".*")
        )
        if not exact:
            raise ManifestError(
                f"dependency lock for {case_id} must contain only exact hashed requirements"
            )


def _logical_lock_statements(path: Path, case_id: str) -> tuple[str, ...]:
    """Join pip continuation lines without interpreting comments or directives."""
    statements: list[str] = []
    fragments: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ManifestError(f"cannot read dependency lock for {case_id}") from error
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        continued = stripped.endswith("\\")
        fragments.append(stripped[:-1].rstrip() if continued else stripped)
        if not continued:
            statements.append(" ".join(fragments))
            fragments.clear()
    if fragments:
        raise ManifestError(f"dependency lock for {case_id} has an unterminated continuation")
    return tuple(statements)


def _tier_choices() -> tuple[CorpusTier, ...]:
    return ("compatibility", "performance", "calibration", "negative-control")


def _platform_choices() -> tuple[CorpusPlatform, ...]:
    return ("ubuntu-24.04", "macos-14")
