"""Content-addressed cache for a target project's normal PEP 517 wheel.

The source-clean compiler overlays Atoll artifacts onto the project's own
wheel. This module caches that immutable base wheel independently from native
regions so an unchanged warm compile does not rebuild target-owned C
extensions. It does not build, unpack, or modify wheels; callers retain those
validation and packaging responsibilities.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shlex
import shutil
import sys
import sysconfig
import tempfile
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict, cast
from urllib.parse import unquote, urlparse

from packaging import tags
from packaging.requirements import InvalidRequirement, Requirement

BASELINE_WHEEL_CACHE_VERSION = 1
BASELINE_WHEEL_CACHE_CONTEXT_ENV = "ATOLL_BASELINE_CACHE_CONTEXT"
_MANIFEST_NAME = "manifest.json"
_CACHE_KEY_PATTERN = re.compile(r"[0-9a-f]{64}")
_TRUTHY_ENVIRONMENT_VALUES = frozenset({"1", "on", "true", "yes"})
_TOOL_ENVIRONMENT_NAMES = (
    "AR",
    "CARGO",
    "CC",
    "CMAKE",
    "CXX",
    "LD",
    "MESON",
    "NINJA",
    "RANLIB",
    "RUSTC",
    "STRIP",
)
_DEFAULT_BUILD_TOOLS = (
    "ar",
    "c++",
    "cargo",
    "cc",
    "clang",
    "clang++",
    "cmake",
    "g++",
    "gcc",
    "ld",
    "link",
    "meson",
    "ninja",
    "rustc",
)
_SYSCONFIG_TOOL_NAMES = ("AR", "CC", "CXX", "LDSHARED")
_CONSTRAINT_ENVIRONMENT_NAMES = ("PIP_BUILD_CONSTRAINT", "PIP_CONSTRAINT")
_LOCAL_PACKAGE_SUFFIXES = (".tar.bz2", ".tar.gz", ".tar.xz", ".whl", ".zip")


class _BaselineWheelCacheManifest(TypedDict):
    """Persisted identity and integrity evidence for one cached wheel.

    Attributes:
        version: Cache format version understood by this Atoll release.
        key: Content-derived project and toolchain fingerprint.
        wheel_name: Installable wheel filename stored beside the manifest.
        wheel_sha256: Digest checked before every restore.
    """

    version: int
    key: str
    wheel_name: str
    wheel_sha256: str


@dataclass(frozen=True, slots=True)
class BaselineWheelCacheProbe:
    """Result of looking up and optionally restoring one baseline wheel.

    Attributes:
        key: Content-derived cache identity used for a later store, when reusable.
        status: Whether lookup hit, missed, or bypassed unsafe reuse.
        wheel_path: Restored output wheel, or ``None`` on a miss.
        reason: Human-readable explanation for a miss or bypass.
        lookup_duration_seconds: Fingerprint and entry-validation duration.
        restore_duration_seconds: Verified wheel-copy duration on a hit.
    """

    key: str | None
    status: Literal["hit", "miss", "bypass"]
    wheel_path: Path | None
    reason: str
    lookup_duration_seconds: float
    restore_duration_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class BaselineWheelCacheStore:
    """Best-effort cache-store evidence retained for phase reporting.

    Attributes:
        stored: Whether a verified entry exists after the operation.
        duration_seconds: Time spent hashing and atomically storing the wheel.
    """

    stored: bool
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class _ReproducibleBuildContext:
    """Externally stable dependency-resolution context for one wheel build.

    Attributes:
        kind: Source of the reproducibility promise.
        digest: Identity of the caller context or complete offline inputs.
    """

    kind: Literal["caller", "offline-wheelhouse"]
    digest: str


@dataclass(frozen=True, slots=True)
class _VerifiedWheel:
    """Cache path paired with the manifest digest checked after restoration.

    Attributes:
        path: Cached wheel path validated before copying.
        digest: Manifest digest required of the restored destination.
    """

    path: Path
    digest: str


def restore_baseline_wheel(
    *,
    project_root: Path,
    cache_root: Path,
    output_dir: Path,
) -> BaselineWheelCacheProbe:
    """Restore a verified baseline wheel for unchanged build inputs.

    The fingerprint is computed from the disposable PEP 517 project copy, so
    generated Atoll state and prior wheels cannot influence the key. Corrupt,
    incomplete, or unknown cache entries are ordinary misses.

    Args:
        project_root: Disposable complete project copy submitted to PEP 517.
        cache_root: Persistent Atoll-owned baseline cache directory.
        output_dir: Disposable directory receiving a restored wheel.

    Returns:
        BaselineWheelCacheProbe: Verified hit or deterministic miss evidence.
    """
    started = time.perf_counter()
    build_context = _reproducible_build_context(project_root)
    if build_context is None:
        return BaselineWheelCacheProbe(
            key=None,
            status="bypass",
            wheel_path=None,
            reason="reproducible PEP 517 dependency context unavailable",
            lookup_duration_seconds=time.perf_counter() - started,
        )
    key = _baseline_wheel_cache_key(project_root, build_context)
    entry_root = cache_root / key
    cached_wheel = _verified_entry_wheel(entry_root, key)
    lookup_duration = time.perf_counter() - started
    if cached_wheel is None:
        return BaselineWheelCacheProbe(
            key=key,
            status="miss",
            wheel_path=None,
            reason="no verified entry",
            lookup_duration_seconds=lookup_duration,
        )
    restore_started = time.perf_counter()
    _reset_dir(output_dir)
    restored = output_dir / cached_wheel.path.name
    try:
        shutil.copy2(cached_wheel.path, restored)
        restored_digest = _file_digest(restored)
    except OSError:
        return BaselineWheelCacheProbe(
            key=key,
            status="miss",
            wheel_path=None,
            reason="verified entry could not be restored",
            lookup_duration_seconds=lookup_duration,
            restore_duration_seconds=time.perf_counter() - restore_started,
        )
    if restored_digest != cached_wheel.digest:
        restored.unlink(missing_ok=True)
        return BaselineWheelCacheProbe(
            key=key,
            status="miss",
            wheel_path=None,
            reason="restored wheel failed digest verification",
            lookup_duration_seconds=lookup_duration,
            restore_duration_seconds=time.perf_counter() - restore_started,
        )
    return BaselineWheelCacheProbe(
        key=key,
        status="hit",
        wheel_path=restored.resolve(),
        reason="verified entry restored",
        lookup_duration_seconds=lookup_duration,
        restore_duration_seconds=time.perf_counter() - restore_started,
    )


def store_baseline_wheel(
    *,
    key: str,
    wheel_path: Path,
    cache_root: Path,
) -> BaselineWheelCacheStore:
    """Atomically retain one wheel after the caller has unpacked it safely.

    Cache storage is an optimization boundary. Filesystem failures never
    replace a successful project build, and a concurrent writer wins when it
    has already produced a valid entry for the same key.

    Args:
        key: Fingerprint returned by :func:`restore_baseline_wheel`.
        wheel_path: Built wheel already validated by the caller.
        cache_root: Persistent Atoll-owned baseline cache directory.

    Returns:
        BaselineWheelCacheStore: Whether a valid entry exists and elapsed time.
    """
    started = time.perf_counter()
    if _CACHE_KEY_PATTERN.fullmatch(key) is None:
        return BaselineWheelCacheStore(
            stored=False,
            duration_seconds=time.perf_counter() - started,
        )
    target = cache_root / key
    if _verified_entry_wheel(target, key) is not None:
        return BaselineWheelCacheStore(
            stored=True,
            duration_seconds=time.perf_counter() - started,
        )
    temporary: Path | None = None
    stored = False
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=cache_root))
        cached_wheel = temporary / wheel_path.name
        shutil.copy2(wheel_path, cached_wheel)
        manifest: _BaselineWheelCacheManifest = {
            "version": BASELINE_WHEEL_CACHE_VERSION,
            "key": key,
            "wheel_name": wheel_path.name,
            "wheel_sha256": _file_digest(cached_wheel),
        }
        (temporary / _MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if target.exists():
            shutil.rmtree(target)
        temporary.replace(target)
        temporary = None
        stored = _verified_entry_wheel(target, key) is not None
    except OSError:
        stored = _verified_entry_wheel(target, key) is not None
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
    return BaselineWheelCacheStore(
        stored=stored,
        duration_seconds=time.perf_counter() - started,
    )


def baseline_wheel_cache_key(project_root: Path) -> str:
    """Fingerprint copied build inputs and the active build toolchain.

    Args:
        project_root: Disposable project copy containing exact PEP 517 inputs.

    Returns:
        str: Lowercase SHA-256 cache key.
    """
    build_context = _reproducible_build_context(project_root)
    return _baseline_wheel_cache_key(project_root, build_context)


def _baseline_wheel_cache_key(
    project_root: Path,
    build_context: _ReproducibleBuildContext | None,
) -> str:
    payload = {
        "cache_version": BASELINE_WHEEL_CACHE_VERSION,
        "project_tree": _project_tree_digest(project_root),
        "git_repository": _git_repository_digest(project_root),
        "build_context": (
            {"kind": build_context.kind, "digest": build_context.digest}
            if build_context is not None
            else None
        ),
        "python": {
            "implementation": sys.implementation.name,
            "cache_tag": sys.implementation.cache_tag,
            "version": platform.python_version(),
            "abiflags": getattr(sys, "abiflags", ""),
            "wheel_tag": str(next(tags.sys_tags())),
        },
        "platform": {
            "machine": platform.machine(),
            "system": platform.system(),
            "release": platform.release(),
        },
        "build_frontend": _distribution_version("build"),
        "environment": _build_environment_fingerprint(),
        "toolchain": _toolchain_fingerprint(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _project_tree_digest(project_root: Path) -> str:
    digest = hashlib.sha256()
    for path in (project_root, *sorted(project_root.rglob("*"))):
        relative = path.relative_to(project_root)
        if relative.parts and relative.parts[0] == ".git":
            continue
        if not path.is_file() and not path.is_dir():
            continue
        metadata = path.stat()
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(b"file" if path.is_file() else b"directory")
        digest.update(b"\0")
        digest.update(str(metadata.st_mode).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(metadata.st_mtime_ns).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(metadata.st_size).encode("ascii"))
        digest.update(b"\0")
        if path.is_file():
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _git_repository_digest(project_root: Path) -> str | None:
    marker = project_root / ".git"
    git_dir = _git_directory(marker, project_root)
    if git_dir is None:
        return None
    common_dir = git_dir
    try:
        common_marker = git_dir / "commondir"
        if common_marker.is_file():
            common_value = Path(common_marker.read_text(encoding="utf-8").strip())
            common_dir = (
                common_value.resolve()
                if common_value.is_absolute()
                else (git_dir / common_value).resolve()
            )
        candidates = [git_dir / "HEAD", git_dir / "index", common_dir / "packed-refs"]
        candidates.extend(sorted((common_dir / "refs").rglob("*")))
        digest = hashlib.sha256()
        hashed = False
        for path in candidates:
            if not path.is_file():
                continue
            relative_root = git_dir if path.is_relative_to(git_dir) else common_dir
            digest.update(path.relative_to(relative_root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
            hashed = True
    except OSError:
        return None
    return digest.hexdigest() if hashed else None


def _git_directory(marker: Path, project_root: Path) -> Path | None:
    if marker.is_dir():
        return marker.resolve()
    if not marker.is_file():
        return None
    try:
        first_line = marker.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError):
        return None
    prefix = "gitdir:"
    if not first_line.startswith(prefix):
        return None
    value = Path(first_line.removeprefix(prefix).strip())
    return value.resolve() if value.is_absolute() else (project_root / value).resolve()


def _build_environment_fingerprint() -> str:
    encoded = json.dumps(sorted(os.environ.items()), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reproducible_build_context(project_root: Path) -> _ReproducibleBuildContext | None:
    explicit = os.environ.get(BASELINE_WHEEL_CACHE_CONTEXT_ENV)
    if explicit:
        return _ReproducibleBuildContext(
            kind="caller",
            digest=hashlib.sha256(explicit.encode("utf-8")).hexdigest(),
        )
    return _offline_build_context(project_root)


def _offline_build_context(project_root: Path) -> _ReproducibleBuildContext | None:
    if os.environ.get("PIP_NO_INDEX", "").lower() not in _TRUTHY_ENVIRONMENT_VALUES:
        return None
    find_links = os.environ.get("PIP_FIND_LINKS")
    if (
        not find_links
        or any(os.environ.get(name) for name in _CONSTRAINT_ENVIRONMENT_NAMES)
        or _has_direct_build_requirement(project_root)
    ):
        return None
    inputs: list[tuple[str, str]] = []
    for token in shlex.split(find_links, posix=os.name != "nt"):
        path = _local_build_input(token, project_root)
        if path is None or (path.is_file() and not path.name.endswith(_LOCAL_PACKAGE_SUFFIXES)):
            return None
        inputs.append((f"find-links:{token}", _path_digest(path)))
    if not inputs:
        return None
    encoded = json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _ReproducibleBuildContext(
        kind="offline-wheelhouse",
        digest=hashlib.sha256(encoded).hexdigest(),
    )


def _local_build_input(value: str, project_root: Path) -> Path | None:
    parsed = urlparse(value)
    if parsed.scheme and parsed.scheme != "file":
        return None
    raw_path = unquote(parsed.path) if parsed.scheme == "file" else value
    candidate = Path(raw_path)
    path = candidate if candidate.is_absolute() else project_root / candidate
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return None
    return resolved if resolved.is_file() or resolved.is_dir() else None


def _path_digest(root: Path) -> str:
    digest = hashlib.sha256()
    paths = (root,) if root.is_file() else (root, *sorted(root.rglob("*")))
    for path in paths:
        if not path.is_file() and not path.is_dir():
            continue
        relative = path.name if root.is_file() else path.relative_to(root).as_posix()
        metadata = path.stat()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(metadata.st_mode).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(metadata.st_mtime_ns).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(metadata.st_size).encode("ascii"))
        digest.update(b"\0")
        if path.is_file():
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _has_direct_build_requirement(project_root: Path) -> bool:
    pyproject = project_root / "pyproject.toml"
    try:
        payload = cast(dict[str, object], tomllib.loads(pyproject.read_text(encoding="utf-8")))
    except (OSError, tomllib.TOMLDecodeError):
        return False
    build_system = payload.get("build-system")
    raw = (
        cast(dict[object, object], build_system).get("requires")
        if isinstance(build_system, dict)
        else None
    )
    direct = False
    if isinstance(raw, list):
        for item in cast(list[object], raw):
            if not isinstance(item, str):
                direct = True
                break
            try:
                url = Requirement(item).url
            except InvalidRequirement:
                direct = True
                break
            if url is not None:
                direct = True
                break
    return direct


def _toolchain_fingerprint() -> tuple[tuple[str, str], ...]:
    command_names: list[str] = list(_DEFAULT_BUILD_TOOLS)
    for name in _TOOL_ENVIRONMENT_NAMES:
        command_names.extend(_command_executables(os.environ.get(name)))
    for name in _SYSCONFIG_TOOL_NAMES:
        value = sysconfig.get_config_var(name)
        command_names.extend(_command_executables(value if isinstance(value, str) else None))
    command_names.append(sys.executable)
    fingerprints: dict[str, str] = {}
    for command in command_names:
        executable = shutil.which(command)
        if executable is None:
            candidate = Path(command)
            executable = str(candidate) if candidate.is_file() else None
        if executable is None:
            continue
        try:
            path = Path(executable).resolve(strict=True)
            fingerprints[str(path)] = _file_digest(path)
        except OSError:
            continue
    return tuple(sorted(fingerprints.items()))


def _command_executables(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    try:
        parts = shlex.split(value, posix=os.name != "nt")
    except ValueError:
        return ()
    return (parts[0],) if parts else ()


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _verified_entry_wheel(entry_root: Path, key: str) -> _VerifiedWheel | None:
    manifest_path = entry_root / _MANIFEST_NAME
    try:
        raw = cast(object, json.loads(manifest_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return None
    manifest = _manifest(raw)
    if manifest is None or manifest["key"] != key:
        return None
    wheel_name = manifest["wheel_name"]
    if Path(wheel_name).name != wheel_name or not wheel_name.endswith(".whl"):
        return None
    wheel_path = entry_root / wheel_name
    try:
        digest = _file_digest(wheel_path)
    except OSError:
        return None
    if digest != manifest["wheel_sha256"]:
        return None
    return _VerifiedWheel(path=wheel_path, digest=digest)


def _manifest(raw: object) -> _BaselineWheelCacheManifest | None:
    if not isinstance(raw, dict):
        return None
    payload = cast(dict[object, object], raw)
    if payload.get("version") != BASELINE_WHEEL_CACHE_VERSION:
        return None
    if not all(
        isinstance(payload.get(name), str) for name in ("key", "wheel_name", "wheel_sha256")
    ):
        return None
    return {
        "version": BASELINE_WHEEL_CACHE_VERSION,
        "key": cast(str, payload["key"]),
        "wheel_name": cast(str, payload["wheel_name"]),
        "wheel_sha256": cast(str, payload["wheel_sha256"]),
    }


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
