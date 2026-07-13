"""Strict metadata for compiler calibration workloads.

Calibration measures compiler headroom on intentionally concentrated kernels.
It is not a repository-compatibility sample and must never contribute to the
real-project aggregate.  This module validates that boundary and the immutable
source identities without downloading or executing third-party code.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol, cast
from urllib.parse import urlparse

from scripts.benchmark_corpus.models import CorpusPlatform

CALIBRATION_SCHEMA_VERSION = 1
CalibrationSource = Literal["atoll-fixture", "pyperformance"]
CalibrationExecution = Literal["atoll-native-hard-suite", "external-checkout-required"]
_SOURCES: tuple[CalibrationSource, ...] = ("atoll-fixture", "pyperformance")
_EXECUTIONS: tuple[CalibrationExecution, ...] = (
    "atoll-native-hard-suite",
    "external-checkout-required",
)
_PLATFORMS: tuple[CorpusPlatform, ...] = ("ubuntu-24.04", "macos-14")
_FIELDS = frozenset({"schema_version", "benchmark"})
_BENCHMARK_FIELDS = frozenset(
    {
        "id",
        "included_in_repository_aggregate",
        "execution",
        "execution_paths",
        "execution_sha256",
        "name",
        "notice",
        "platforms",
        "repository",
        "revision",
        "runner",
        "source",
        "source_path",
        "source_sha256",
    }
)
_SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_SHA = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_LOCAL_RUNNER = (
    "uv",
    "run",
    "--python",
    "3.12",
    "python",
    "scripts/run_native_optimizer_benchmark.py",
    "--workspace",
    "{workspace}",
    "--evidence-root",
    "{evidence_root}",
)


class CalibrationError(ValueError):
    """Raised when calibration metadata could contaminate corpus evidence."""


class _DigestUpdate(Protocol):
    """Minimal hashlib update surface used by canonical bundle encoding."""

    def update(self, value: bytes, /) -> None:
        """Add bytes to the digest state."""


@dataclass(frozen=True, slots=True)
class _ExecutionContract:
    """Parsed lowering and provenance fields validated as one unit."""

    source: CalibrationSource
    source_path: PurePosixPath
    execution: CalibrationExecution
    execution_paths: tuple[PurePosixPath, ...]
    execution_sha256: str | None
    runner: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class CalibrationBenchmark:
    """One pinned calibration identity excluded from repository aggregates.

    ``source_sha256`` authenticates the exact benchmark body.  Atoll-owned
    fixture sources are rehashed locally; remote pyperformance sources retain
    their reviewed digest for bootstrap verification by a manual runner.
    """

    id: str
    name: str
    source: CalibrationSource
    repository: str
    revision: str
    source_path: PurePosixPath
    source_sha256: str
    execution: CalibrationExecution
    execution_paths: tuple[PurePosixPath, ...]
    execution_sha256: str | None
    runner: tuple[str, ...] | None
    platforms: tuple[CorpusPlatform, ...]
    notice: PurePosixPath
    included_in_repository_aggregate: Literal[False]

    @property
    def repository_verified(self) -> bool:
        """Whether normal catalog loading authenticates the source bytes."""
        return self.source == "atoll-fixture"

    @property
    def runnable(self) -> bool:
        """Whether a materializable repository-local runner is declared."""
        return self.runner is not None


@dataclass(frozen=True, slots=True)
class CalibrationCatalog:
    """Validated, sorted calibration group safe to report independently."""

    path: Path
    schema_version: int
    benchmarks: tuple[CalibrationBenchmark, ...]


def load_calibration_catalog(path: Path, repository_root: Path) -> CalibrationCatalog:
    """Load calibration metadata and verify all repository-owned evidence.

    Args:
        path: Schema-v1 calibration TOML.
        repository_root: Atoll checkout containing local fixtures and notices.

    Returns:
        CalibrationCatalog: Immutable calibration identities sorted by ID.

    Raises:
        CalibrationError: If metadata, paths, digests, or aggregate exclusions
            are incomplete or unsafe.
    """
    try:
        raw: object = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise CalibrationError(f"cannot read calibration catalog {path}: {error}") from error
    payload = _mapping(raw, "calibration")
    _reject_unknown(payload, _FIELDS, "calibration")
    version = _integer(payload, "schema_version", "calibration")
    if version != CALIBRATION_SCHEMA_VERSION:
        raise CalibrationError(
            f"calibration.schema_version must be {CALIBRATION_SCHEMA_VERSION}, got {version}"
        )
    values = payload.get("benchmark")
    if not isinstance(values, list) or not values:
        raise CalibrationError("calibration.benchmark must be a non-empty array of tables")
    benchmarks = tuple(
        _parse_benchmark(_mapping(item, f"calibration.benchmark[{index}]"), index)
        for index, item in enumerate(cast(list[object], values))
    )
    ids = tuple(item.id for item in benchmarks)
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise CalibrationError(f"duplicate calibration benchmark(s): {', '.join(duplicates)}")
    root = repository_root.resolve(strict=True)
    for benchmark in benchmarks:
        _verify_notice(root, benchmark)
        if benchmark.source == "atoll-fixture":
            _verify_local_source(root, benchmark)
            _verify_execution_bundle(root, benchmark)
    return CalibrationCatalog(
        path=path.resolve(strict=True),
        schema_version=version,
        benchmarks=tuple(sorted(benchmarks, key=lambda item: item.id)),
    )


def _parse_benchmark(payload: dict[str, object], index: int) -> CalibrationBenchmark:
    label = f"calibration.benchmark[{index}]"
    _reject_unknown(payload, _BENCHMARK_FIELDS, label)
    benchmark_id = _string(payload, "id", label)
    if _SLUG.fullmatch(benchmark_id) is None:
        raise CalibrationError(f"{label}.id must be a lowercase slug")
    source = _string(payload, "source", label)
    if source not in _SOURCES:
        raise CalibrationError(f"{label}.source has unsupported value {source!r}")
    included = payload.get("included_in_repository_aggregate")
    if included is not False:
        raise CalibrationError(f"{label}.included_in_repository_aggregate must be explicitly false")
    source_path = _safe_path(payload, "source_path", label)
    source_sha256 = _matched(
        payload,
        "source_sha256",
        label,
        _SHA256,
        "64-character lowercase SHA-256",
    )
    platforms = _literal_sequence(payload, "platforms", label, _PLATFORMS)
    execution_value = _string(payload, "execution", label)
    if execution_value not in _EXECUTIONS:
        raise CalibrationError(f"{label}.execution has unsupported value {execution_value!r}")
    runner = _optional_command(payload, "runner", label)
    execution_paths = _optional_paths(payload, "execution_paths", label)
    execution_sha256 = _optional_matched(
        payload,
        "execution_sha256",
        label,
        _SHA256,
        "64-character lowercase SHA-256",
    )
    execution = execution_value
    _validate_execution(
        _ExecutionContract(
            source=source,
            source_path=source_path,
            execution=execution,
            execution_paths=execution_paths,
            execution_sha256=execution_sha256,
            runner=runner,
        ),
        label,
    )
    return CalibrationBenchmark(
        id=benchmark_id,
        name=_string(payload, "name", label),
        source=source,
        repository=_repository(_string(payload, "repository", label), label),
        revision=_matched(payload, "revision", label, _SHA, "full lowercase Git SHA"),
        source_path=source_path,
        source_sha256=source_sha256,
        execution=execution,
        execution_paths=execution_paths,
        execution_sha256=execution_sha256,
        runner=runner,
        platforms=cast(tuple[CorpusPlatform, ...], platforms),
        notice=_safe_path(payload, "notice", label),
        included_in_repository_aggregate=False,
    )


def calibration_runner_argv(
    benchmark: CalibrationBenchmark,
    *,
    workspace: Path,
    evidence_root: Path,
) -> tuple[str, ...]:
    """Materialize one reviewed local calibration command template.

    External pyperformance pins deliberately have no ambient runner.  Their
    checkout must first be authenticated with :func:`verify_external_calibration`.

    Args:
        benchmark: Catalog entry to materialize.
        workspace: Disposable hard-benchmark workspace.
        evidence_root: Persistent bounded evidence destination.

    Returns:
        tuple[str, ...]: Shell-free argv with both required paths supplied.

    Raises:
        CalibrationError: If the entry requires an externally verified runner.
    """
    if benchmark.runner is None:
        raise CalibrationError(f"calibration {benchmark.id} has no repository-local runner")
    replacements = {
        "{workspace}": str(workspace),
        "{evidence_root}": str(evidence_root),
    }
    return tuple(replacements.get(argument, argument) for argument in benchmark.runner)


def calibration_execution_digest(
    repository_root: Path,
    execution_paths: tuple[PurePosixPath, ...],
) -> str:
    """Hash every file that the repository-local hard suite can execute.

    Directory entries use the same ignored generated-file names as the hard
    runner's fixture copy.  Any additional non-ignored file changes the digest,
    including an untracked file that would otherwise enter the copied fixture.

    Args:
        repository_root: Atoll checkout containing the execution bundle.
        execution_paths: Reviewed files or directory roots relative to the checkout.

    Returns:
        str: Canonical SHA-256 over relative paths and complete file bytes.

    Raises:
        CalibrationError: If a path escapes, is missing, is a symlink, or contains
            a non-regular filesystem entry.
    """
    root = repository_root.resolve(strict=True)
    files: dict[PurePosixPath, Path] = {}
    for relative in execution_paths:
        _collect_execution_path(root, relative, files)
    if not files:
        raise CalibrationError("calibration execution bundle contains no files")
    digest = hashlib.sha256()
    for relative, path in sorted(files.items(), key=lambda item: item[0].as_posix()):
        _update_digest(digest, relative.as_posix().encode("utf-8"))
        _update_digest(digest, path.read_bytes())
    return digest.hexdigest()


def _collect_execution_path(
    root: Path,
    relative: PurePosixPath,
    files: dict[PurePosixPath, Path],
) -> None:
    candidate = root.joinpath(*relative.parts)
    if candidate.is_symlink():
        raise CalibrationError(f"calibration execution path is a symlink: {relative}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise CalibrationError(f"calibration execution path is unavailable: {relative}") from error
    if not resolved.is_relative_to(root):
        raise CalibrationError(f"calibration execution path escapes repository: {relative}")
    if resolved.is_file():
        files[relative] = resolved
        return
    if not resolved.is_dir():
        raise CalibrationError(f"calibration execution path is not regular: {relative}")
    _collect_execution_directory(root, resolved, files)


def _collect_execution_directory(
    root: Path,
    directory: Path,
    files: dict[PurePosixPath, Path],
) -> None:
    for child in sorted(directory.rglob("*")):
        child_relative = PurePosixPath(child.relative_to(root).as_posix())
        if _ignored_execution_path(child_relative):
            continue
        if child.is_symlink():
            raise CalibrationError(
                f"calibration execution bundle contains symlink: {child_relative}"
            )
        if child.is_file():
            files[child_relative] = child
        elif not child.is_dir():
            raise CalibrationError(
                f"calibration execution bundle contains non-regular path: {child_relative}"
            )


def verify_external_calibration(
    benchmark: CalibrationBenchmark,
    checkout_root: Path,
) -> Path:
    """Authenticate an external calibration checkout without executing it.

    The detached checkout must have the exact pinned ``HEAD``, no tracked
    modifications, and the reviewed benchmark source digest.  This explicit
    step prevents an ambient pyperformance installation from satisfying the
    catalog contract.

    Args:
        benchmark: Externally pinned calibration entry.
        checkout_root: Existing Git checkout at the reviewed revision.

    Returns:
        Path: Authenticated benchmark source path.

    Raises:
        CalibrationError: If the entry is local or checkout provenance differs.
    """
    if benchmark.execution != "external-checkout-required":
        raise CalibrationError(f"calibration {benchmark.id} does not require external checkout")
    if checkout_root.is_symlink():
        raise CalibrationError(f"external calibration checkout is a symlink: {checkout_root}")
    try:
        root = checkout_root.resolve(strict=True)
    except OSError as error:
        raise CalibrationError(
            f"external calibration checkout is unavailable: {checkout_root}"
        ) from error
    if not root.is_dir():
        raise CalibrationError(f"external calibration checkout is not a directory: {root}")
    head = _git_output(root, ("rev-parse", "--verify", "HEAD^{commit}"))
    if head != benchmark.revision:
        raise CalibrationError(f"external calibration revision mismatch for {benchmark.id}: {head}")
    if _git_output(root, ("branch", "--show-current")):
        raise CalibrationError(f"external calibration checkout is not detached for {benchmark.id}")
    if _git_output(root, ("status", "--porcelain", "--untracked-files=normal")):
        raise CalibrationError(f"external calibration checkout is modified for {benchmark.id}")
    source = _regular_repository_file(root, benchmark.source_path, benchmark.id, "source")
    observed = hashlib.sha256(source.read_bytes()).hexdigest()
    if observed != benchmark.source_sha256:
        raise CalibrationError(f"calibration source digest mismatch for {benchmark.id}")
    return source


def _validate_execution(contract: _ExecutionContract, label: str) -> None:
    if contract.source == "atoll-fixture":
        _validate_local_execution(contract, label)
        return
    if (
        contract.execution != "external-checkout-required"
        or contract.execution_paths
        or contract.execution_sha256 is not None
        or contract.runner is not None
    ):
        raise CalibrationError(
            f"{label} pyperformance entries require external checkout verification only"
        )


def _validate_local_execution(contract: _ExecutionContract, label: str) -> None:
    runner = contract.runner
    if (
        contract.execution != "atoll-native-hard-suite"
        or runner is None
        or contract.execution_sha256 is None
    ):
        raise CalibrationError(
            f"{label} Atoll fixtures require a digested atoll-native-hard-suite runner"
        )
    required_execution_paths = {
        PurePosixPath("scripts/run_native_optimizer_benchmark.py"),
        PurePosixPath("tests/fixtures/native_optimization_project"),
    }
    if set(contract.execution_paths) != required_execution_paths:
        raise CalibrationError(f"{label}.execution_paths must cover the complete native suite")
    fixture_root = PurePosixPath("tests/fixtures/native_optimization_project")
    if not contract.source_path.is_relative_to(fixture_root):
        raise CalibrationError(f"{label}.source_path must identify an executed fixture file")
    if runner != _LOCAL_RUNNER:
        raise CalibrationError(
            f"{label}.runner must exactly match the native hard workflow command"
        )


def _verify_execution_bundle(root: Path, benchmark: CalibrationBenchmark) -> None:
    expected = benchmark.execution_sha256
    if expected is None:
        raise CalibrationError(f"calibration execution digest is missing for {benchmark.id}")
    observed = calibration_execution_digest(root, benchmark.execution_paths)
    if observed != expected:
        raise CalibrationError(f"calibration execution bundle digest mismatch for {benchmark.id}")


def _ignored_execution_path(path: PurePosixPath) -> bool:
    return any(
        part in {".atoll", ".pytest_cache", "__pycache__"} or part.endswith(".egg-info")
        for part in path.parts
    )


def _update_digest(digest: _DigestUpdate, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, byteorder="big"))
    digest.update(value)


def _git_output(root: Path, arguments: tuple[str, ...]) -> str:
    git = shutil.which("git")
    if git is None:
        raise CalibrationError("git executable is unavailable for calibration verification")
    try:
        completed = subprocess.run(
            (
                git,
                "-c",
                "core.fsmonitor=false",
                "-c",
                f"core.hooksPath={os.devnull}",
                "-C",
                str(root),
                *arguments,
            ),
            check=False,
            capture_output=True,
            env={
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_OPTIONAL_LOCKS": "0",
                "GIT_TERMINAL_PROMPT": "0",
                "LC_ALL": "C",
                "PATH": os.environ.get("PATH", ""),
            },
            shell=False,
            text=True,
            timeout=30.0,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise CalibrationError(f"cannot inspect external calibration checkout: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "git inspection failed"
        raise CalibrationError(f"cannot inspect external calibration checkout: {detail}")
    return completed.stdout.strip()


def _verify_local_source(root: Path, benchmark: CalibrationBenchmark) -> None:
    source = _regular_repository_file(root, benchmark.source_path, benchmark.id, "source")
    observed = hashlib.sha256(source.read_bytes()).hexdigest()
    if observed != benchmark.source_sha256:
        raise CalibrationError(f"calibration source digest mismatch for {benchmark.id}")


def _verify_notice(root: Path, benchmark: CalibrationBenchmark) -> None:
    _regular_repository_file(root, benchmark.notice, benchmark.id, "notice")


def _regular_repository_file(
    root: Path,
    relative: PurePosixPath,
    benchmark_id: str,
    role: str,
) -> Path:
    candidate = root.joinpath(*relative.parts)
    if candidate.is_symlink():
        raise CalibrationError(f"calibration {role} is a symlink for {benchmark_id}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise CalibrationError(
            f"calibration {role} is unavailable for {benchmark_id}: {relative}"
        ) from error
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise CalibrationError(f"calibration {role} escapes the repository for {benchmark_id}")
    return resolved


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CalibrationError(f"{label} must be a table")
    raw = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in raw):
        raise CalibrationError(f"{label} must have only string keys")
    return cast(dict[str, object], value)


def _reject_unknown(payload: dict[str, object], allowed: frozenset[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise CalibrationError(f"{label} contains unknown field(s): {', '.join(unknown)}")


def _string(payload: dict[str, object], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise CalibrationError(f"{label}.{key} must be a non-empty string")
    return value


def _integer(payload: dict[str, object], key: str, label: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise CalibrationError(f"{label}.{key} must be an integer")
    return value


def _matched(
    payload: dict[str, object],
    key: str,
    label: str,
    pattern: re.Pattern[str],
    description: str,
) -> str:
    value = _string(payload, key, label)
    if pattern.fullmatch(value) is None:
        raise CalibrationError(f"{label}.{key} must be a {description}")
    return value


def _optional_matched(
    payload: dict[str, object],
    key: str,
    label: str,
    pattern: re.Pattern[str],
    description: str,
) -> str | None:
    if key not in payload:
        return None
    return _matched(payload, key, label, pattern, description)


def _safe_path(payload: dict[str, object], key: str, label: str) -> PurePosixPath:
    return _safe_path_value(_string(payload, key, label), f"{label}.{key}")


def _safe_path_value(raw: str, label: str) -> PurePosixPath:
    value = PurePosixPath(raw)
    if value.is_absolute() or not value.parts or ".." in value.parts:
        raise CalibrationError(f"{label} must be a safe relative path")
    return value


def _optional_paths(payload: dict[str, object], key: str, label: str) -> tuple[PurePosixPath, ...]:
    if key not in payload:
        return ()
    value = payload[key]
    if not isinstance(value, list) or not value:
        raise CalibrationError(f"{label}.{key} must be a non-empty path array")
    items = tuple(cast(list[object], value))
    if any(not isinstance(item, str) or not item for item in items):
        raise CalibrationError(f"{label}.{key} must contain non-empty strings")
    paths = tuple(
        _safe_path_value(cast(str, item), f"{label}.{key}[{index}]")
        for index, item in enumerate(items)
    )
    if len(paths) != len(set(paths)):
        raise CalibrationError(f"{label}.{key} must not contain duplicate paths")
    return paths


def _command(payload: dict[str, object], key: str, label: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise CalibrationError(f"{label}.{key} must be a non-empty argv array")
    items = tuple(cast(list[object], value))
    if any(not isinstance(item, str) or not item or "\0" in item for item in items):
        raise CalibrationError(f"{label}.{key} must contain safe non-empty strings")
    return cast(tuple[str, ...], items)


def _optional_command(payload: dict[str, object], key: str, label: str) -> tuple[str, ...] | None:
    if key not in payload:
        return None
    return _command(payload, key, label)


def _literal_sequence(
    payload: dict[str, object], key: str, label: str, allowed: tuple[str, ...]
) -> tuple[str, ...]:
    values = _command(payload, key, label)
    if any(value not in allowed for value in values) or len(values) != len(set(values)):
        raise CalibrationError(
            f"{label}.{key} must contain unique values from {', '.join(allowed)}"
        )
    return values


def _repository(value: str, label: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.query or parsed.fragment:
        raise CalibrationError(f"{label}.repository must be a canonical HTTPS Git URL")
    if not parsed.path.endswith(".git"):
        raise CalibrationError(f"{label}.repository must end in .git")
    return value
