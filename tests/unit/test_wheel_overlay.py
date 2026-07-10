"""Unit tests for archive-level wheel overlay helpers."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import cast

import pytest

from atoll.wheel_overlay import (
    WheelOverlayError,
    build_baseline_wheel,
    repack_overlaid_wheel,
    unpack_single_wheel_payload,
    unpack_wheel_payload,
    validate_wheel_archive,
)

BASELINE_NAME = "demo_pkg-1.2.3-py3-none-any.whl"
PLATFORM_TAG = "cp312-cp312-macosx_14_0_arm64"
BACKEND_FAILURE_RETURNCODE = 17


def test_build_baseline_wheel_captures_command_output_and_duration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(command: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[str]:
        cwd = cast(Path, kwargs["cwd"])
        captured.update(
            {
                "command": command,
                "cwd": cwd,
                "check": kwargs["check"],
                "shell": kwargs["shell"],
                "capture_output": kwargs["capture_output"],
                "text": kwargs["text"],
            }
        )
        (tmp_path / "dist" / BASELINE_NAME).write_bytes(b"wheel")
        return subprocess.CompletedProcess(
            command,
            BACKEND_FAILURE_RETURNCODE,
            "stdout text",
            "stderr text",
        )

    monkeypatch.setattr("atoll.wheel_overlay.subprocess.run", fake_run)

    evidence = build_baseline_wheel(tmp_path, tmp_path / "dist")

    assert evidence.command == (
        sys.executable,
        "-I",
        "-m",
        "build",
        "--wheel",
        "--no-isolation",
        "--outdir",
        str((tmp_path / "dist").resolve()),
    )
    assert captured == {
        "command": evidence.command,
        "cwd": tmp_path.resolve(),
        "check": False,
        "shell": False,
        "capture_output": True,
        "text": True,
    }
    assert not evidence.succeeded
    assert evidence.returncode == BACKEND_FAILURE_RETURNCODE
    assert evidence.stdout == "stdout text"
    assert evidence.stderr == "stderr text"
    assert evidence.duration_seconds >= 0
    assert evidence.wheel_paths == ((tmp_path / "dist" / BASELINE_NAME).resolve(),)


def test_repack_preserves_payload_metadata_and_recomputes_record(tmp_path: Path) -> None:
    baseline = tmp_path / BASELINE_NAME
    _write_baseline_wheel(
        baseline,
        extra_members={
            "demo_pkg-1.2.3.dist-info/RECORD.jws": b"stale signature",
            "demo_pkg-1.2.3.dist-info/RECORD.p7s": b"stale signature",
        },
    )
    payload_dir = tmp_path / "payload"
    dist_info_dir = unpack_wheel_payload(baseline, payload_dir)
    (payload_dir / "demo_pkg" / "native_ext.cpython-312-darwin.so").write_bytes(b"native")

    output = repack_overlaid_wheel(
        baseline_wheel_path=baseline,
        payload_dir=payload_dir,
        output_dir=tmp_path / "dist",
        platform_tag=PLATFORM_TAG,
    )

    assert output.name == f"demo_pkg-1.2.3-{PLATFORM_TAG}.whl"
    assert dist_info_dir.name == "demo_pkg-1.2.3.dist-info"
    with zipfile.ZipFile(output) as wheel:
        names = set(wheel.namelist())
        assert "demo_pkg/__init__.py" in names
        assert "demo_pkg/data/schema.json" in names
        assert "demo_pkg-1.2.3.dist-info/METADATA" in names
        assert "demo_pkg-1.2.3.dist-info/entry_points.txt" in names
        assert "demo_pkg/native_ext.cpython-312-darwin.so" in names
        assert "demo_pkg-1.2.3.dist-info/RECORD.jws" not in names
        assert "demo_pkg-1.2.3.dist-info/RECORD.p7s" not in names
        wheel_metadata = wheel.read("demo_pkg-1.2.3.dist-info/WHEEL").decode("utf-8")
        assert "Root-Is-Purelib: false\n" in wheel_metadata
        assert f"Tag: {PLATFORM_TAG}\n" in wheel_metadata
        assert "Tag: py3-none-any\n" not in wheel_metadata
        assert "Wheel-Version: 1.0\n" in wheel_metadata
        assert _record_rows(wheel) == _expected_record_rows(wheel)


def test_unpack_resets_destination_and_rejects_multiple_dist_info(tmp_path: Path) -> None:
    wheel_path = tmp_path / BASELINE_NAME
    _write_baseline_wheel(
        wheel_path,
        extra_members={"other-1.0.dist-info/METADATA": b"Name: other\n"},
    )
    payload_dir = tmp_path / "payload"
    payload_dir.mkdir()
    stale = payload_dir / "stale.txt"
    stale.write_text("remove me", encoding="utf-8")

    with pytest.raises(WheelOverlayError, match="expected exactly one dist-info"):
        unpack_wheel_payload(wheel_path, payload_dir)

    assert not stale.exists()


def test_unpack_single_wheel_payload_rejects_ambiguous_output(tmp_path: Path) -> None:
    wheel_dir = tmp_path / "dist"
    wheel_dir.mkdir()

    with pytest.raises(WheelOverlayError, match="expected exactly one wheel, found 0"):
        unpack_single_wheel_payload(wheel_dir, tmp_path / "payload")

    _write_baseline_wheel(wheel_dir / BASELINE_NAME)
    _write_baseline_wheel(wheel_dir / "other-1.0.0-py3-none-any.whl")

    with pytest.raises(WheelOverlayError, match="expected exactly one wheel, found 2"):
        unpack_single_wheel_payload(wheel_dir, tmp_path / "payload")


def test_unpack_rejects_tampered_record_digest(tmp_path: Path) -> None:
    """Baseline wheel bytes must match the hashes declared by its RECORD."""
    wheel_path = tmp_path / BASELINE_NAME
    _write_baseline_wheel(
        wheel_path,
        record_data_overrides={"demo_pkg/__init__.py": b"stale content\n"},
    )

    with pytest.raises(WheelOverlayError, match="RECORD integrity check failed"):
        unpack_wheel_payload(wheel_path, tmp_path / "payload")


@pytest.mark.parametrize(
    ("variant", "message"),
    [
        ("invalid_utf8", "invalid wheel RECORD"),
        ("wrong_columns", "row must have path, hash, and size"),
        ("missing_member", "duplicate or missing member"),
        ("own_evidence", "leave its own hash and size empty"),
        ("incomplete", "evidence is incomplete"),
        ("invalid_hash", "hash is invalid"),
        ("invalid_algorithm", "hash algorithm is invalid"),
        ("duplicate", "duplicate or missing member"),
        ("omitted", "omits archive member"),
    ],
)
def test_unpack_rejects_malformed_record_evidence(
    tmp_path: Path,
    variant: str,
    message: str,
) -> None:
    """Every wheel member requires unique, complete, parseable RECORD evidence."""
    wheel_path = tmp_path / BASELINE_NAME
    _write_record_variant(wheel_path, variant)

    with pytest.raises(WheelOverlayError, match=message):
        unpack_wheel_payload(wheel_path, tmp_path / "payload")


def test_unpack_rejects_duplicate_and_symlink_archive_members(tmp_path: Path) -> None:
    """Archive structure is rejected before extraction or RECORD processing."""
    duplicate = tmp_path / "duplicate.whl"
    with pytest.warns(UserWarning, match="Duplicate name"):
        _write_duplicate_wheel(duplicate)
    with pytest.raises(WheelOverlayError, match="duplicate archive member"):
        unpack_wheel_payload(duplicate, tmp_path / "duplicate-payload")

    symlink = tmp_path / "symlink.whl"
    link_info = zipfile.ZipInfo("pkg/link.py")
    link_info.create_system = 3
    link_info.external_attr = 0o120777 << 16
    with zipfile.ZipFile(symlink, "w") as wheel:
        wheel.writestr(link_info, b"module.py")
    with pytest.raises(WheelOverlayError, match="unsupported symlink"):
        unpack_wheel_payload(symlink, tmp_path / "symlink-payload")


def test_unpack_rejects_file_destination(tmp_path: Path) -> None:
    """A payload destination must be a resettable directory."""
    wheel_path = tmp_path / BASELINE_NAME
    _write_baseline_wheel(wheel_path)
    payload = tmp_path / "payload"
    payload.write_text("not a directory", encoding="utf-8")

    with pytest.raises(WheelOverlayError, match="payload path is not a directory"):
        unpack_wheel_payload(wheel_path, payload)


def test_final_wheel_validation_checks_metadata_and_filename_tags(tmp_path: Path) -> None:
    """Final validation accepts generic checks and rejects routing-tag mismatches."""
    baseline = tmp_path / BASELINE_NAME
    _write_baseline_wheel(baseline)
    validate_wheel_archive(baseline)

    with pytest.raises(WheelOverlayError, match="metadata tags"):
        validate_wheel_archive(baseline, expected_tag=PLATFORM_TAG)

    tagged_name_mismatch = tmp_path / "demo_pkg-1.2.3-py3-none-any.whl"
    _write_baseline_wheel(tagged_name_mismatch, wheel_tag=PLATFORM_TAG)
    with pytest.raises(WheelOverlayError, match="filename tag"):
        validate_wheel_archive(tagged_name_mismatch, expected_tag=PLATFORM_TAG)

    multiple_wheel_files = tmp_path / "multiple-wheel-metadata.whl"
    _write_baseline_wheel(
        multiple_wheel_files,
        extra_members={"other-1.0.dist-info/WHEEL": b"Tag: py3-none-any\n"},
    )
    with pytest.raises(WheelOverlayError, match="expected exactly one WHEEL file"):
        validate_wheel_archive(multiple_wheel_files, expected_tag=PLATFORM_TAG)


def test_repack_rejects_missing_metadata_invalid_tags_and_bad_names(tmp_path: Path) -> None:
    """Overlay promotion rejects ambiguous metadata and unsafe output names."""
    baseline = tmp_path / BASELINE_NAME
    _write_baseline_wheel(baseline)
    payload = tmp_path / "payload"
    dist_info = unpack_wheel_payload(baseline, payload)
    (dist_info / "WHEEL").unlink()
    with pytest.raises(WheelOverlayError, match="missing WHEEL metadata"):
        repack_overlaid_wheel(
            baseline_wheel_path=baseline,
            payload_dir=payload,
            output_dir=tmp_path / "out-missing",
            platform_tag=PLATFORM_TAG,
        )

    unpack_wheel_payload(baseline, payload)
    with pytest.raises(WheelOverlayError, match="three safe components"):
        repack_overlaid_wheel(
            baseline_wheel_path=baseline,
            payload_dir=payload,
            output_dir=tmp_path / "out-tag",
            platform_tag="unsafe-tag",
        )
    with pytest.raises(WheelOverlayError, match=r"must end with \.whl"):
        repack_overlaid_wheel(
            baseline_wheel_path=tmp_path / "not-a-wheel.zip",
            payload_dir=payload,
            output_dir=tmp_path / "out-suffix",
            platform_tag=PLATFORM_TAG,
        )
    with pytest.raises(WheelOverlayError, match="no replaceable wheel tags"):
        repack_overlaid_wheel(
            baseline_wheel_path=tmp_path / "short.whl",
            payload_dir=payload,
            output_dir=tmp_path / "out-short",
            platform_tag=PLATFORM_TAG,
        )


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("../escape.py", "unsafe archive member"),
        ("/absolute.py", "unsafe archive member"),
        ("pkg\\module.py", "unsafe archive member"),
    ],
)
def test_unpack_rejects_unsafe_archive_entries(
    tmp_path: Path,
    name: str,
    message: str,
) -> None:
    wheel_path = tmp_path / BASELINE_NAME
    _write_baseline_wheel(wheel_path, extra_members={name: b"bad"})

    with pytest.raises(WheelOverlayError, match=message):
        unpack_wheel_payload(wheel_path, tmp_path / "payload")


def _write_baseline_wheel(
    path: Path,
    *,
    extra_members: dict[str, bytes] | None = None,
    record_data_overrides: dict[str, bytes] | None = None,
    wheel_tag: str = "py3-none-any",
) -> None:
    members = {
        "demo_pkg/__init__.py": b"VALUE = 1\n",
        "demo_pkg/data/schema.json": b'{"kind": "demo"}\n',
        "demo_pkg-1.2.3.dist-info/METADATA": b"Metadata-Version: 2.4\nName: demo-pkg\n",
        "demo_pkg-1.2.3.dist-info/WHEEL": (
            f"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: {wheel_tag}\n"
        ).encode(),
        "demo_pkg-1.2.3.dist-info/entry_points.txt": (b"[console_scripts]\ndemo=demo_pkg:main\n"),
    }
    members.update(extra_members or {})
    record_members = {**members, **(record_data_overrides or {})}
    record_rows = _rows_for_members(record_members)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        for archive_path, data in members.items():
            wheel.writestr(archive_path, data)
        wheel.writestr("demo_pkg-1.2.3.dist-info/RECORD", record_rows)


def _write_record_variant(path: Path, variant: str) -> None:
    members = {
        "demo_pkg/__init__.py": b"VALUE = 1\n",
        "demo_pkg-1.2.3.dist-info/METADATA": b"Name: demo-pkg\n",
        "demo_pkg-1.2.3.dist-info/WHEEL": b"Tag: py3-none-any\n",
    }
    record_path = "demo_pkg-1.2.3.dist-info/RECORD"
    valid_rows = list(csv.reader(io.StringIO(_rows_for_members(members).decode("utf-8"))))
    record = {
        "invalid_utf8": b"\xff",
        "wrong_columns": b"only,two\n",
        "missing_member": b"missing.py,sha256=abc,1\n",
        "own_evidence": f"{record_path},sha256=abc,1\n".encode(),
        "incomplete": b"demo_pkg/__init__.py,,\n",
        "invalid_hash": b"demo_pkg/__init__.py,broken,10\n",
        "invalid_algorithm": b"demo_pkg/__init__.py,unknown=abc,10\n",
    }.get(variant)
    if variant == "duplicate":
        valid_rows.insert(1, valid_rows[0])
        record = _csv_rows(valid_rows)
    elif variant == "omitted":
        record = _csv_rows(valid_rows[1:])
    elif record is None:
        raise AssertionError(f"unknown RECORD variant: {variant}")
    with zipfile.ZipFile(path, "w") as wheel:
        for archive_path, data in members.items():
            wheel.writestr(archive_path, data)
        wheel.writestr(record_path, record)


def _write_duplicate_wheel(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr("pkg/module.py", b"first")
        wheel.writestr("pkg/module.py", b"second")


def _csv_rows(rows: list[list[str]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _rows_for_members(members: dict[str, bytes]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    for archive_path, data in members.items():
        if archive_path.endswith((".dist-info/RECORD.jws", ".dist-info/RECORD.p7s")):
            continue
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
        )
        writer.writerow((archive_path, f"sha256={digest}", str(len(data))))
    writer.writerow(("demo_pkg-1.2.3.dist-info/RECORD", "", ""))
    return output.getvalue().encode("utf-8")


def _record_rows(wheel: zipfile.ZipFile) -> dict[str, tuple[str, str]]:
    record = wheel.read("demo_pkg-1.2.3.dist-info/RECORD").decode("utf-8")
    return {row[0]: (row[1], row[2]) for row in csv.reader(io.StringIO(record))}


def _expected_record_rows(wheel: zipfile.ZipFile) -> dict[str, tuple[str, str]]:
    rows: dict[str, tuple[str, str]] = {}
    for archive_path in wheel.namelist():
        if archive_path.endswith("/RECORD"):
            rows[archive_path] = ("", "")
            continue
        data = wheel.read(archive_path)
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
        )
        rows[archive_path] = (f"sha256={digest}", str(len(data)))
    return rows
