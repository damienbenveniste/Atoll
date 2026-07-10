"""Wheel build and overlay helpers for native Atoll package payloads.

This module owns the archive-level mechanics needed after Atoll has prepared an
overlaid wheel payload. It deliberately does not know how projects select files,
compile extensions, or generate native artifacts. Callers provide a target
project root, a baseline wheel, a resettable payload directory, and a final
platform tag; the helpers return immutable evidence or write deterministic wheel
archives without project-specific branching.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from email.parser import Parser
from email.policy import compat32
from pathlib import Path, PurePosixPath

_DIST_INFO_SUFFIX = ".dist-info"
_RECORD_NAME = "RECORD"
_WHEEL_NAME = "WHEEL"
_WHEEL_FILENAME_SUFFIX = ".whl"
_ZIP_SYMLINK_MODE = 0o120000
_WHEEL_TAG_COMPONENT_COUNT = 3
_MINIMUM_TAGGED_WHEEL_COMPONENT_COUNT = 5
_RECORD_COLUMN_COUNT = 3


@dataclass(frozen=True, slots=True)
class WheelBuildEvidence:
    """Observed result from invoking the target project's normal wheel build.

    `command` is the exact argv passed to `subprocess.run` with `shell=False`.
    Backend failures are represented by `returncode`, `stdout`, and `stderr`
    rather than raised as exceptions, so callers can include the evidence in a
    larger package report. `wheel_paths` lists wheels produced in `outdir` after
    the command completes; it can be empty when the backend failed.
    """

    command: tuple[str, ...]
    project_root: Path
    outdir: Path
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float
    wheel_paths: tuple[Path, ...]

    @property
    def succeeded(self) -> bool:
        """Return whether the backend process exited successfully."""
        return self.returncode == 0


class WheelOverlayError(ValueError):
    """Raised when a wheel archive or payload cannot be safely overlaid.

    These failures indicate invalid inputs, unsafe archive members, or metadata
    shapes Atoll cannot rewrite without risking a malformed wheel. They are
    programming or input validation errors, unlike normal PEP 517 backend
    process failures captured by `WheelBuildEvidence`.
    """


def build_baseline_wheel(project_root: Path, outdir: Path) -> WheelBuildEvidence:
    """Invoke the target project's normal PEP 517 wheel build.

    The build runs in `project_root` with the current interpreter as
    `python -m build --wheel --no-isolation --outdir <outdir>`.
    `shell=False` is used through argv form. Non-zero backend exits are returned
    as structured evidence and are not raised.
    """

    resolved_project_root = project_root.resolve()
    resolved_outdir = outdir.resolve()
    resolved_outdir.mkdir(parents=True, exist_ok=True)
    command = (
        sys.executable,
        "-I",
        "-m",
        "build",
        "--wheel",
        "--no-isolation",
        "--outdir",
        str(resolved_outdir),
    )
    started = time.perf_counter()
    completed = _run_build_command(command, cwd=resolved_project_root)
    duration = time.perf_counter() - started
    wheel_paths = tuple(
        sorted(path.resolve() for path in resolved_outdir.glob(f"*{_WHEEL_FILENAME_SUFFIX}"))
    )
    return WheelBuildEvidence(
        command=command,
        project_root=resolved_project_root,
        outdir=resolved_outdir,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        duration_seconds=duration,
        wheel_paths=wheel_paths,
    )


def unpack_wheel_payload(wheel_path: Path, payload_dir: Path) -> Path:
    """Reset `payload_dir` and unpack exactly one safe wheel archive into it.

    Every archive member is validated before extraction. Absolute paths, parent
    traversal, backslash-separated names, duplicate members, and symlinks are
    rejected so a malicious wheel cannot write outside the reset payload
    directory or create ambiguous payload state.
    """

    resolved_payload_dir = payload_dir.resolve()
    _reset_dir(resolved_payload_dir)
    with zipfile.ZipFile(wheel_path) as wheel:
        _validate_zip_members(wheel.infolist())
        _validate_record(wheel)
        wheel.extractall(resolved_payload_dir)
    return _single_dist_info_dir(resolved_payload_dir)


def unpack_single_wheel_payload(wheel_dir: Path, payload_dir: Path) -> Path:
    """Reset `payload_dir` and unpack the only wheel present in `wheel_dir`.

    PEP 517 build output is ambiguous if zero or multiple wheels are present.
    This helper enforces the single-artifact contract before delegating to the
    same safe archive extraction path used for explicit wheel files.
    """

    wheel_paths = tuple(sorted(wheel_dir.glob(f"*{_WHEEL_FILENAME_SUFFIX}")))
    if len(wheel_paths) != 1:
        raise WheelOverlayError(f"expected exactly one wheel, found {len(wheel_paths)}")
    return unpack_wheel_payload(wheel_paths[0], payload_dir)


def repack_overlaid_wheel(
    *,
    baseline_wheel_path: Path,
    payload_dir: Path,
    output_dir: Path,
    platform_tag: str,
) -> Path:
    """Repack an overlaid payload as a platform wheel with a fresh RECORD.

    The payload directory is expected to contain files unpacked from one
    baseline wheel plus caller-supplied overlay files. Baseline metadata and
    package data are preserved unless the caller changed them in `payload_dir`.
    This helper changes the wheel metadata to `Root-Is-Purelib: false`, replaces
    all `Tag` headers with `platform_tag`, names the output by replacing the
    baseline filename's final three tag components, and recomputes every RECORD
    hash and size from bytes written into the archive.
    """

    payload_root = payload_dir.resolve()
    dist_info_dir = validate_dist_info_dir(payload_root)
    for signature_name in ("RECORD.jws", "RECORD.p7s"):
        (dist_info_dir / signature_name).unlink(missing_ok=True)
    wheel_metadata_path = dist_info_dir / _WHEEL_NAME
    if not wheel_metadata_path.is_file():
        raise WheelOverlayError(f"missing WHEEL metadata: {wheel_metadata_path}")

    archive_files = tuple(_iter_payload_files(payload_root))
    relative_wheel_path = _archive_path(payload_root, wheel_metadata_path)
    rewritten_wheel_metadata = _rewrite_wheel_metadata(
        wheel_metadata_path.read_text(encoding="utf-8"),
        platform_tag,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / _retagged_wheel_name(baseline_wheel_path.name, platform_tag)

    record_path = f"{dist_info_dir.name}/{_RECORD_NAME}"
    record_entries: list[tuple[str, bytes]] = []
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        for archive_path, source_path in archive_files:
            if archive_path == record_path:
                continue
            data = (
                rewritten_wheel_metadata.encode("utf-8")
                if archive_path == relative_wheel_path
                else source_path.read_bytes()
            )
            wheel.writestr(archive_path, data)
            record_entries.append((archive_path, data))
        wheel.writestr(record_path, _record_bytes((*record_entries, (record_path, b""))))
    validate_wheel_archive(output_path, expected_tag=platform_tag)
    return output_path


def validate_wheel_archive(wheel_path: Path, *, expected_tag: str | None = None) -> None:
    """Read and validate every final wheel member, RECORD entry, and optional tag."""
    with zipfile.ZipFile(wheel_path) as wheel:
        _validate_zip_members(wheel.infolist())
        _validate_record(wheel)
        if expected_tag is None:
            return
        wheel_paths = tuple(
            name for name in wheel.namelist() if name.endswith(f".dist-info/{_WHEEL_NAME}")
        )
        if len(wheel_paths) != 1:
            raise WheelOverlayError(f"expected exactly one WHEEL file, found {len(wheel_paths)}")
        message = Parser(policy=compat32).parsestr(wheel.read(wheel_paths[0]).decode("utf-8"))
        tags = tuple(message.get_all("Tag", failobj=[]))
        if tags != (expected_tag,):
            raise WheelOverlayError(
                f"wheel metadata tags {tags!r} do not match expected tag {expected_tag!r}"
            )
        if not wheel_path.name.endswith(f"-{expected_tag}{_WHEEL_FILENAME_SUFFIX}"):
            raise WheelOverlayError("wheel filename tag does not match WHEEL metadata")


def _reset_dir(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise WheelOverlayError(f"payload path is not a directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def validate_dist_info_dir(payload_root: Path) -> Path:
    """Return the one top-level `.dist-info` directory for a wheel payload.

    Wheel metadata belongs to a single distribution. Missing or multiple
    dist-info directories make WHEEL and RECORD updates ambiguous, so callers
    should treat this exception as invalid payload input.
    """

    return _single_dist_info_dir(payload_root.resolve())


def _run_build_command(command: tuple[str, ...], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        shell=False,
        capture_output=True,
        text=True,
    )


def _validate_zip_members(infos: list[zipfile.ZipInfo]) -> None:
    names: set[str] = set()
    for info in infos:
        name = info.filename
        if not name:
            raise WheelOverlayError("wheel contains an empty archive member name")
        if name in names:
            raise WheelOverlayError(f"wheel contains duplicate archive member: {name}")
        names.add(name)
        if _is_unsafe_archive_name(name):
            raise WheelOverlayError(f"wheel contains unsafe archive member: {name}")
        file_type = (info.external_attr >> 16) & 0o170000
        if file_type == _ZIP_SYMLINK_MODE:
            raise WheelOverlayError(f"wheel contains unsupported symlink member: {name}")


def _validate_record(wheel: zipfile.ZipFile) -> None:
    file_names = tuple(info.filename for info in wheel.infolist() if not info.is_dir())
    record_paths = tuple(name for name in file_names if name.endswith(f".dist-info/{_RECORD_NAME}"))
    if len(record_paths) != 1:
        raise WheelOverlayError(f"expected exactly one RECORD file, found {len(record_paths)}")
    record_path = record_paths[0]
    try:
        rows = tuple(csv.reader(io.StringIO(wheel.read(record_path).decode("utf-8"))))
    except (UnicodeDecodeError, csv.Error) as error:
        raise WheelOverlayError(f"invalid wheel RECORD: {error}") from error
    seen: set[str] = set()
    for row in rows:
        seen.add(
            _validate_record_row(
                wheel=wheel,
                row=row,
                record_path=record_path,
                file_names=file_names,
                seen=seen,
            )
        )
    signature_suffixes = (".dist-info/RECORD.jws", ".dist-info/RECORD.p7s")
    expected = {name for name in file_names if not name.endswith(signature_suffixes)}
    if seen != expected:
        missing = ", ".join(sorted(expected - seen))
        raise WheelOverlayError(f"wheel RECORD omits archive member(s): {missing}")


def _validate_record_row(
    *,
    wheel: zipfile.ZipFile,
    row: list[str],
    record_path: str,
    file_names: tuple[str, ...],
    seen: set[str],
) -> str:
    if len(row) != _RECORD_COLUMN_COUNT:
        raise WheelOverlayError("wheel RECORD row must have path, hash, and size")
    path, encoded_hash, size_text = row
    if path in seen or path not in file_names:
        raise WheelOverlayError(f"wheel RECORD has duplicate or missing member: {path}")
    if path == record_path:
        if encoded_hash or size_text:
            raise WheelOverlayError("wheel RECORD must leave its own hash and size empty")
        return path
    data = wheel.read(path)
    if not encoded_hash or not size_text:
        raise WheelOverlayError(f"wheel RECORD evidence is incomplete for {path}")
    _validate_record_digest(path, data, encoded_hash, size_text)
    return path


def _validate_record_digest(
    path: str,
    data: bytes,
    encoded_hash: str,
    size_text: str,
) -> None:
    algorithm, separator, encoded_digest = encoded_hash.partition("=")
    if not separator or not algorithm or not encoded_digest:
        raise WheelOverlayError(f"wheel RECORD hash is invalid for {path}")
    try:
        digest = hashlib.new(algorithm, data).digest()
    except ValueError as error:
        raise WheelOverlayError(f"wheel RECORD hash algorithm is invalid: {algorithm}") from error
    actual_digest = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    if actual_digest != encoded_digest or size_text != str(len(data)):
        raise WheelOverlayError(f"wheel RECORD integrity check failed for {path}")


def _is_unsafe_archive_name(name: str) -> bool:
    if not name or name in {".", ".."} or "\\" in name:
        return True
    raw_parts = name.removesuffix("/").split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        return True
    path = PurePosixPath(name)
    return path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts)


def _single_dist_info_dir(payload_root: Path) -> Path:
    dist_info_dirs = tuple(
        path
        for path in payload_root.iterdir()
        if path.is_dir() and path.name.endswith(_DIST_INFO_SUFFIX)
    )
    if len(dist_info_dirs) != 1:
        raise WheelOverlayError(
            f"expected exactly one dist-info directory, found {len(dist_info_dirs)}"
        )
    return dist_info_dirs[0]


def _iter_payload_files(payload_root: Path) -> tuple[tuple[str, Path], ...]:
    files: list[tuple[str, Path]] = []
    for path in payload_root.rglob("*"):
        if path.is_file():
            archive_path = _archive_path(payload_root, path)
            if _is_unsafe_archive_name(archive_path):
                raise WheelOverlayError(f"payload contains unsafe path: {archive_path}")
            files.append((archive_path, path))
    return tuple(sorted(files, key=lambda item: item[0]))


def _archive_path(payload_root: Path, path: Path) -> str:
    return path.relative_to(payload_root).as_posix()


def _rewrite_wheel_metadata(content: str, platform_tag: str) -> str:
    _validate_platform_tag(platform_tag)
    message = Parser(policy=compat32).parsestr(content)
    if "Root-Is-Purelib" in message:
        message.replace_header("Root-Is-Purelib", "false")
    else:
        message["Root-Is-Purelib"] = "false"
    del message["Tag"]
    message["Tag"] = platform_tag
    text = message.as_string(policy=compat32)
    if not text.endswith("\n"):
        text = f"{text}\n"
    return text


def _validate_platform_tag(platform_tag: str) -> None:
    parts = platform_tag.split("-")
    if len(parts) != _WHEEL_TAG_COMPONENT_COUNT or any(
        not _is_safe_tag_part(part) for part in parts
    ):
        raise WheelOverlayError(f"platform tag must have three safe components: {platform_tag}")


def _is_safe_tag_part(part: str) -> bool:
    return bool(part) and all(character.isalnum() or character in "._" for character in part)


def _retagged_wheel_name(baseline_name: str, platform_tag: str) -> str:
    if not baseline_name.endswith(_WHEEL_FILENAME_SUFFIX):
        raise WheelOverlayError(f"baseline wheel name must end with .whl: {baseline_name}")
    _validate_platform_tag(platform_tag)
    stem = baseline_name.removesuffix(_WHEEL_FILENAME_SUFFIX)
    parts = stem.split("-")
    if len(parts) < _MINIMUM_TAGGED_WHEEL_COMPONENT_COUNT:
        raise WheelOverlayError(
            f"baseline wheel name has no replaceable wheel tags: {baseline_name}"
        )
    return (
        "-".join((*parts[:-_WHEEL_TAG_COMPONENT_COUNT], *platform_tag.split("-")))
        + _WHEEL_FILENAME_SUFFIX
    )


def _record_bytes(entries: tuple[tuple[str, bytes], ...]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    for archive_path, data in entries:
        if archive_path.endswith(f"/{_RECORD_NAME}"):
            writer.writerow((archive_path, "", ""))
            continue
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
        )
        writer.writerow((archive_path, f"sha256={digest}", str(len(data))))
    return output.getvalue().encode("utf-8")
