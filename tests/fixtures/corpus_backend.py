"""Dependency-free PEP 517 backend for the corpus lifecycle fixture."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import zipfile
from pathlib import Path

_DIST_INFO = "simple_project-0.1.0.dist-info"
_WHEEL_NAME = "simple_project-0.1.0-py3-none-any.whl"


def get_requires_for_build_wheel(
    config_settings: dict[str, object] | None = None,
) -> list[str]:
    """Return no requirements so the fixture can build entirely offline."""
    del config_settings
    return []


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, object] | None = None,
    metadata_directory: str | None = None,
) -> str:
    """Build the fixed fixture package using only the standard library."""
    del config_settings, metadata_directory
    root = Path.cwd()
    members = {
        path.relative_to(root / "src").as_posix(): path.read_bytes()
        for path in sorted((root / "src" / "app").rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }
    members.update(
        {
            f"{_DIST_INFO}/METADATA": (
                b"Metadata-Version: 2.1\n"
                b"Name: simple-project\n"
                b"Version: 0.1.0\n"
                b"Summary: Fixture metadata preserved by Atoll wheel overlays.\n"
                b"Requires-Python: >=3.12\n"
            ),
            f"{_DIST_INFO}/WHEEL": (
                b"Wheel-Version: 1.0\n"
                b"Generator: corpus-backend\n"
                b"Root-Is-Purelib: true\n"
                b"Tag: py3-none-any\n"
            ),
            f"{_DIST_INFO}/entry_points.txt": (
                b"[console_scripts]\nsimple-project-rank = app.ranking:normalize_features\n"
            ),
        }
    )
    record_path = f"{_DIST_INFO}/RECORD"
    members[record_path] = _record(members, record_path)
    destination = Path(wheel_directory) / _WHEEL_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in sorted(members.items()):
            archive.writestr(name, payload)
    return _WHEEL_NAME


def _record(members: dict[str, bytes], record_path: str) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    for name, payload in sorted(members.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
        writer.writerow((name, f"sha256={digest.decode()}", str(len(payload))))
    writer.writerow((record_path, "", ""))
    return stream.getvalue().encode()
