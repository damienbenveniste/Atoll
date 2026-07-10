"""Tests for isolated payload and wheel routing verification."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

from atoll.runtime.package_verify import (
    PackageVerificationPlan,
    VerificationArtifact,
    verify_package_subprocess,
)


def test_verify_payload_imports_promised_region_and_checks_artifact(tmp_path: Path) -> None:
    """A fresh interpreter accepts matching status and native bytes."""
    payload = tmp_path / "payload"
    package = payload / "pkg"
    artifact = payload / ".atoll" / "artifacts" / "region" / "native.so"
    package.mkdir(parents=True)
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"native")
    (package / "__init__.py").write_text(
        "__atoll_region_status__ = {'region-1': {'compiled': True}}\n",
        encoding="utf-8",
    )
    plan = _plan(artifact)

    result = verify_package_subprocess(
        stage="payload",
        target=payload,
        plan=plan,
        project_root=tmp_path,
    )

    assert result.success is True
    assert result.exit_code == 0
    assert result.stderr == ""
    assert result.command[:3] == (result.command[0], "-I", "-c")


def test_verify_wheel_extracts_before_importing(tmp_path: Path) -> None:
    """Wheel verification imports from a child-owned extraction directory."""
    source = tmp_path / "source"
    package = source / "pkg"
    artifact = source / ".atoll" / "artifacts" / "region" / "native.so"
    package.mkdir(parents=True)
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"native")
    (package / "__init__.py").write_text(
        "__atoll_region_status__ = {'region-1': {'compiled': True}}\n",
        encoding="utf-8",
    )
    wheel = tmp_path / "pkg-1.0-cp312-cp312-test.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source).as_posix())

    result = verify_package_subprocess(
        stage="wheel",
        target=wheel,
        plan=_plan(artifact),
        project_root=tmp_path,
    )

    assert result.success is True
    assert result.exit_code == 0


def test_verify_payload_reports_missing_compiled_status(tmp_path: Path) -> None:
    """A child import without its promised region is a hard verification failure."""
    payload = tmp_path / "payload"
    package = payload / "pkg"
    artifact = payload / ".atoll" / "artifacts" / "region" / "native.so"
    package.mkdir(parents=True)
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"native")
    (package / "__init__.py").write_text(
        "__atoll_region_status__ = {}\n",
        encoding="utf-8",
    )

    result = verify_package_subprocess(
        stage="payload",
        target=payload,
        plan=_plan(artifact),
        project_root=tmp_path,
    )

    assert result.success is False
    assert result.exit_code != 0
    assert "Atoll region did not compile: region-1" in result.stderr


def _plan(artifact: Path) -> PackageVerificationPlan:
    return PackageVerificationPlan(
        modules=("pkg",),
        regions=(("pkg", ("region-1",)),),
        artifacts=(
            VerificationArtifact(
                path=".atoll/artifacts/region/native.so",
                digest=hashlib.sha256(artifact.read_bytes()).hexdigest(),
            ),
        ),
    )
