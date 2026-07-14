"""Tests for isolated payload and wheel routing verification."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from atoll.runtime.package_verify import (
    PackageVerificationPlan,
    VerificationArtifact,
    VerificationBinding,
    verify_package_subprocess,
)

_VERIFY_COMMAND_ARGUMENT_COUNT = 6
_MAX_VERIFY_COMMAND_ARGUMENT_LENGTH = 10_000


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
    assert len(result.command) == _VERIFY_COMMAND_ARGUMENT_COUNT
    assert "region-1" not in result.command
    assert not (package / "__pycache__").exists()


def test_verify_payload_transports_large_plan_over_stdin(tmp_path: Path) -> None:
    """Large verification plans do not exceed operating-system argv limits."""
    payload = tmp_path / "payload"
    package = payload / "pkg"
    artifact = payload / ".atoll" / "artifacts" / "region" / "native.so"
    package.mkdir(parents=True)
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"native")
    region_ids = tuple(f"region-{index:05d}" for index in range(20_000))
    (package / "__init__.py").write_text(
        "__atoll_region_status__ = {\n"
        "    f'region-{index:05d}': {'compiled': True}\n"
        "    for index in range(20_000)\n"
        "}\n",
        encoding="utf-8",
    )
    plan = PackageVerificationPlan(
        modules=("pkg",),
        regions=(("pkg", region_ids),),
        artifacts=(_artifact(artifact),),
    )

    result = verify_package_subprocess(
        stage="payload",
        target=payload,
        plan=plan,
        project_root=tmp_path,
        variant_allowlist=frozenset(region_ids),
    )

    assert result.success is True
    assert len(result.command) == _VERIFY_COMMAND_ARGUMENT_COUNT
    assert max(map(len, result.command)) < _MAX_VERIFY_COMMAND_ARGUMENT_LENGTH


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


def test_verify_payload_checks_descriptor_and_execution_kind(tmp_path: Path) -> None:
    """Fresh verification rejects a promised descriptor with the wrong callable shape."""
    payload = tmp_path / "payload"
    package = payload / "pkg"
    artifact = payload / ".atoll" / "artifacts" / "region" / "native.so"
    package.mkdir(parents=True)
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"native")
    (package / "__init__.py").write_text(
        """def fallback(value: int = 1) -> int:
    return value

def wrapped(value: int = 1) -> int:
    return value

wrapped.__atoll_compiled_target__ = wrapped
wrapped.__atoll_python_fallback__ = fallback
wrapped.__defaults__ = fallback.__defaults__
wrapped.__kwdefaults__ = fallback.__kwdefaults__

class Worker:
    run = staticmethod(wrapped)

__atoll_region_status__ = {'region-1': {'compiled': True}}
""",
        encoding="utf-8",
    )
    base = _plan(artifact)
    plan = PackageVerificationPlan(
        modules=base.modules,
        regions=base.regions,
        artifacts=base.artifacts,
        bindings=(
            VerificationBinding(
                module="pkg",
                qualname="Worker.run",
                kind="staticmethod",
                execution_kind="coroutine",
            ),
        ),
    )

    result = verify_package_subprocess(
        stage="payload",
        target=payload,
        plan=plan,
        project_root=tmp_path,
    )

    assert result.success is False
    assert "expected coroutine, got sync" in result.stderr


def test_verify_payload_accepts_preserved_async_descriptor(tmp_path: Path) -> None:
    """Fresh verification accepts a compiled wrapper with matching descriptor and signature."""
    payload = tmp_path / "payload"
    package = payload / "pkg"
    artifact = payload / ".atoll" / "artifacts" / "region" / "native.so"
    package.mkdir(parents=True)
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"native")
    (package / "__init__.py").write_text(
        """async def fallback(value: int = 1) -> int:
    return value

async def wrapped(value: int = 1) -> int:
    return value

wrapped.__atoll_compiled_target__ = wrapped
wrapped.__atoll_python_fallback__ = fallback
wrapped.__defaults__ = fallback.__defaults__
wrapped.__kwdefaults__ = fallback.__kwdefaults__

class Worker:
    run = classmethod(wrapped)

__atoll_region_status__ = {'region-1': {'compiled': True}}
""",
        encoding="utf-8",
    )
    base = _plan(artifact)
    plan = PackageVerificationPlan(
        modules=base.modules,
        regions=base.regions,
        artifacts=base.artifacts,
        bindings=(
            VerificationBinding(
                module="pkg",
                qualname="Worker.run",
                kind="classmethod",
                execution_kind="coroutine",
            ),
        ),
    )

    result = verify_package_subprocess(
        stage="payload",
        target=payload,
        plan=plan,
        project_root=tmp_path,
    )

    assert result.success is True


def test_verify_payload_activates_only_allowlisted_native_variant(tmp_path: Path) -> None:
    """Verification can exclude a staged variant that would fail during import."""
    payload, artifact = _variant_payload(tmp_path)
    plan = PackageVerificationPlan(
        modules=("pkg",),
        regions=(("pkg", ("selected-a", "selected-b")),),
        artifacts=(_artifact(artifact),),
    )

    result = verify_package_subprocess(
        stage="payload",
        target=payload,
        plan=plan,
        project_root=tmp_path,
        variant_allowlist=frozenset(("selected-b", "selected-a")),
    )

    assert result.success is True
    assert result.stderr == ""


def test_verify_payload_default_activates_all_native_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default verification ignores an inherited allowlist and activates every variant."""
    payload, artifact = _variant_payload(tmp_path)
    monkeypatch.setenv("ATOLL_VARIANT_ALLOWLIST", "selected-a")
    plan = PackageVerificationPlan(
        modules=("pkg",),
        regions=(("pkg", ("selected-a", "selected-b", "unselected")),),
        artifacts=(_artifact(artifact),),
    )

    result = verify_package_subprocess(
        stage="payload",
        target=payload,
        plan=plan,
        project_root=tmp_path,
    )

    assert result.success is False
    assert "unselected variant activated" in result.stderr


def _variant_payload(tmp_path: Path) -> tuple[Path, Path]:
    """Create a staged package whose second native variant fails on activation."""
    payload = tmp_path / "payload"
    package = payload / "pkg"
    artifact = payload / ".atoll" / "artifacts" / "region" / "native.so"
    package.mkdir(parents=True)
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"native")
    (package / "__init__.py").write_text(
        """import os

allowlist_text = os.getenv("ATOLL_VARIANT_ALLOWLIST")
allowlist = None if allowlist_text is None else frozenset(allowlist_text.splitlines())
if allowlist is not None and allowlist_text != "selected-a\\nselected-b":
    raise RuntimeError(f"variant allowlist is not deterministic: {allowlist_text!r}")
variants = ("selected-a", "selected-b", "unselected")
__atoll_region_status__ = {}
for variant in variants:
    if allowlist is not None and variant not in allowlist:
        continue
    if variant == "unselected":
        raise RuntimeError("unselected variant activated")
    __atoll_region_status__[variant] = {"compiled": True}
""",
        encoding="utf-8",
    )
    return payload, artifact


def _artifact(artifact: Path) -> VerificationArtifact:
    """Return verification metadata for the shared staged native artifact."""
    return VerificationArtifact(
        path=".atoll/artifacts/region/native.so",
        digest=hashlib.sha256(artifact.read_bytes()).hexdigest(),
    )


def _plan(artifact: Path) -> PackageVerificationPlan:
    return PackageVerificationPlan(
        modules=("pkg",),
        regions=(("pkg", ("region-1",)),),
        artifacts=(_artifact(artifact),),
    )
