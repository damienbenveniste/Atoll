"""Command-line entry point for repository-local corpus tooling.

Validation, lock inspection, matrix projection, strict aggregation, and manual
history promotion remain network-free.
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

from scripts.benchmark_corpus.aggregation import (
    AggregationError,
    aggregate_case_results,
    render_aggregate_json,
    render_aggregate_markdown,
)
from scripts.benchmark_corpus.calibration import (
    CalibrationError,
    load_calibration_catalog,
    verify_external_calibration,
)
from scripts.benchmark_corpus.history import (
    HistoryError,
    PromotionOptions,
    promote_results,
)
from scripts.benchmark_corpus.lifecycle import (
    LifecycleOptions,
    run_case,
    validate_performance_assets,
)
from scripts.benchmark_corpus.manifest import ManifestError, load_manifest, manifest_matrix
from scripts.benchmark_corpus.models import CorpusManifest, CorpusPlatform, CorpusTier

DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "benchmarks" / "corpus" / "manifest.toml"
DEFAULT_CALIBRATION = DEFAULT_MANIFEST.parent / "calibration.toml"
DEFAULT_HISTORY = DEFAULT_MANIFEST.parent / "history"
DEFAULT_BENCHMARK_DOCS = DEFAULT_MANIFEST.parents[2] / "docs" / "benchmarks.md"
LOCK_EXCLUDE_NEWER = "2026-07-13T23:59:59Z"
_SHA256_HASH = re.compile(r"(?:^|\s)--hash=sha256:([0-9a-f]{64})(?=\s|$)")


def main(argv: tuple[str, ...] | None = None) -> int:
    """Validate corpus input and print machine-readable command results.

    Args:
        argv: Optional arguments replacing ``sys.argv``.

    Returns:
        int: Zero for valid outcomes, one for retained case/aggregate failures,
            and two for invalid corpus input or command infrastructure.
    """
    parser = _parser()
    args = parser.parse_args(tuple(sys.argv[1:] if argv is None else argv))
    try:
        manifest = load_manifest(args.manifest)
        if args.command == "validate":
            return _run_validate(manifest, args)
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
        if args.command in {"aggregate", "promote"}:
            return _run_result_command(manifest, args)
        parser.error(f"unsupported command: {args.command}")
    except (
        AggregationError,
        CalibrationError,
        HistoryError,
        ManifestError,
        OSError,
        RuntimeError,
    ) as error:
        print(f"benchmark corpus error: {error}", file=sys.stderr)
        return 2


def _run_validate(manifest: CorpusManifest, args: argparse.Namespace) -> int:
    """Validate local corpus assets and optionally authenticate external calibrations."""
    atoll_root = args.atoll_root.resolve(strict=True)
    calibration = load_calibration_catalog(args.calibration, atoll_root)
    external = tuple(
        benchmark for benchmark in calibration.benchmarks if not benchmark.repository_verified
    )
    verified_external = 0
    if args.calibration_checkout is not None:
        for benchmark in external:
            verify_external_calibration(benchmark, args.calibration_checkout)
            verified_external += 1
    adapter_root = atoll_root / "benchmarks" / "corpus" / "adapters"
    for case in manifest.cases:
        if "performance" in case.tiers:
            validate_performance_assets(atoll_root, case, adapter_root)
    print(
        json.dumps(
            {
                "backends": list(manifest.backends),
                "calibration_catalogued": len(calibration.benchmarks),
                "calibration_external_pins": len(external),
                "calibration_external_verified": verified_external,
                "calibration_local_bundles_verified": sum(
                    benchmark.repository_verified for benchmark in calibration.benchmarks
                ),
                "calibration_local_runnable": sum(
                    benchmark.runnable for benchmark in calibration.benchmarks
                ),
                "cases": len(manifest.cases),
                "python": manifest.python_version,
                "schema_version": manifest.schema_version,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def _run_result_command(manifest: CorpusManifest, args: argparse.Namespace) -> int:
    """Aggregate or manually promote one complete result matrix slice."""
    result_paths = _case_result_paths(args.results_root)
    if args.command == "aggregate":
        aggregate = aggregate_case_results(
            manifest,
            result_paths,
            tier=args.tier,
            platform=args.platform,
        )
        output_root = args.output_root or args.results_root
        json_path, markdown_path = _write_aggregate(
            aggregate_json=render_aggregate_json(aggregate),
            aggregate_markdown=render_aggregate_markdown(aggregate),
            output_root=output_root,
            tier=args.tier,
            platform=args.platform,
        )
        print(
            json.dumps(
                {
                    "json": str(json_path),
                    "markdown": str(markdown_path),
                    "platform": aggregate.platform,
                    "tier": aggregate.tier,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        invalid = aggregate.infrastructure_invalid_case_ids or aggregate.semantic_invalid_case_ids
        return 1 if invalid else 0
    snapshot_path, docs_path = promote_results(
        manifest,
        result_paths,
        PromotionOptions(
            tier=args.tier,
            platform=args.platform,
            label=args.label,
            reviewed_by=args.reviewed_by,
            history_root=args.history_root,
            docs_path=args.docs_path,
        ),
    )
    print(
        json.dumps(
            {"docs": str(docs_path), "snapshot": str(snapshot_path)},
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.benchmark_corpus")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate corpus and calibration manifests")
    validate.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    validate.add_argument(
        "--calibration-checkout",
        type=Path,
        help="authenticate external calibration pins against an existing detached checkout",
    )
    validate.add_argument("--atoll-root", type=Path, default=DEFAULT_MANIFEST.parents[2])
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
    aggregate = commands.add_parser(
        "aggregate",
        help="strictly aggregate one complete tier and platform result slice",
    )
    aggregate.add_argument("--tier", choices=_tier_choices(), required=True)
    aggregate.add_argument("--platform", choices=_platform_choices(), required=True)
    aggregate.add_argument("--results-root", type=Path, required=True)
    aggregate.add_argument("--output-root", type=Path)
    promote = commands.add_parser(
        "promote",
        help="retain one manually reviewed compact snapshot and refresh benchmark docs",
    )
    promote.add_argument("--tier", choices=_tier_choices(), required=True)
    promote.add_argument("--platform", choices=_platform_choices(), required=True)
    promote.add_argument("--results-root", type=Path, required=True)
    promote.add_argument("--label", required=True)
    promote.add_argument("--reviewed-by", required=True)
    promote.add_argument("--history-root", type=Path, default=DEFAULT_HISTORY)
    promote.add_argument("--docs-path", type=Path, default=DEFAULT_BENCHMARK_DOCS)
    return parser


def _case_result_paths(results_root: Path) -> tuple[Path, ...]:
    """Discover regular, non-symlink case envelopes beneath one evidence root.

    Args:
        results_root: Workflow artifact tree containing case-specific evidence.

    Returns:
        tuple[Path, ...]: Deterministically sorted ``case-result.json`` paths.

    Raises:
        AggregationError: If the root or a discovered result is unsafe.
    """
    if results_root.is_symlink():
        raise AggregationError(f"results root is a symlink: {results_root}")
    try:
        root = results_root.resolve(strict=True)
    except OSError as error:
        raise AggregationError(f"results root is unavailable: {results_root}") from error
    if not root.is_dir():
        raise AggregationError(f"results root is not a directory: {root}")
    paths: list[Path] = []
    for candidate in root.rglob("case-result.json"):
        if candidate.is_symlink():
            raise AggregationError(f"case result is a symlink: {candidate}")
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise AggregationError(f"case result escapes the results root: {candidate}")
        paths.append(resolved)
    return tuple(sorted(paths))


def _write_aggregate(
    *,
    aggregate_json: str,
    aggregate_markdown: str,
    output_root: Path,
    tier: CorpusTier,
    platform: CorpusPlatform,
) -> tuple[Path, Path]:
    """Write canonical aggregate reports beneath an explicit output root."""
    output = output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = f"aggregate-{tier}-{platform}"
    json_path = output / f"{stem}.json"
    markdown_path = output / f"{stem}.md"
    json_path.write_text(aggregate_json, encoding="utf-8")
    markdown_path.write_text(aggregate_markdown, encoding="utf-8")
    return json_path, markdown_path


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
