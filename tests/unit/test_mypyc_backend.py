"""Tests for the programmatic mypyc build backend."""

from __future__ import annotations

import importlib.machinery
from pathlib import Path

import pytest

from atoll.backends import mypyc as mypyc_backend


class FakeBuildExt:
    """Small build_ext stand-in that writes a fake extension artifact."""

    artifact_path: Path | None = None

    def __init__(self, distribution: object) -> None:
        self.distribution = distribution
        self.inplace = False
        self.build_lib = ""
        self.build_temp = ""

    def ensure_finalized(self) -> None:
        """Match the build_ext API used by Atoll."""

    def run(self) -> None:
        """Pretend to compile by writing an extension-suffixed file."""
        if self.artifact_path is not None:
            self.artifact_path.write_text("", encoding="utf-8")


def test_build_sidecars_succeeds_for_empty_input(tmp_path: Path) -> None:
    """Building with no enabled sidecars is a no-op success."""
    result = mypyc_backend.build_sidecars(
        (),
        project_root=tmp_path,
        build_dir=tmp_path / ".atoll" / "build",
        source_roots=(tmp_path,),
    )

    assert result.success is True
    assert result.stdout == "no enabled Atoll sidecars to build"


def test_build_sidecars_detects_generated_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful backend run returns extension artifacts next to sidecars."""
    sidecar = tmp_path / "_ranking_atoll.py"
    sidecar.write_text("def score_user() -> int:\n    return 1\n", encoding="utf-8")
    FakeBuildExt.artifact_path = sidecar.with_name(
        f"{sidecar.stem}{importlib.machinery.EXTENSION_SUFFIXES[0]}"
    )

    def fake_mypycify(paths: list[str]) -> list[object]:
        assert paths == [sidecar.name]
        return []

    monkeypatch.setattr(mypyc_backend, "mypycify", fake_mypycify)
    monkeypatch.setattr(mypyc_backend, "build_ext", FakeBuildExt)

    result = mypyc_backend.build_sidecars(
        (sidecar,),
        project_root=tmp_path,
        build_dir=tmp_path / ".atoll" / "build",
        source_roots=(tmp_path,),
    )

    assert result.success is True
    assert FakeBuildExt.artifact_path is not None
    assert result.artifact_paths == (FakeBuildExt.artifact_path,)


def test_build_sidecars_classifies_build_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build failures are returned as classified compile attempts."""
    sidecar = tmp_path / "_ranking_atoll.py"
    sidecar.write_text("def score_user() -> int:\n    return 1\n", encoding="utf-8")

    def failing_mypycify(paths: list[str]) -> list[object]:
        raise RuntimeError(f"mypy type issue in {paths[0]}")

    monkeypatch.setattr(mypyc_backend, "mypycify", failing_mypycify)

    result = mypyc_backend.build_sidecars(
        (sidecar,),
        project_root=tmp_path,
        build_dir=tmp_path / ".atoll" / "build",
        source_roots=(tmp_path,),
    )

    assert result.success is False
    assert result.stderr.startswith("MYPYC_TYPE_ERROR")


def test_build_sidecars_classifies_native_build_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native build-looking failures are classified as environment failures."""
    sidecar = tmp_path / "_ranking_atoll.py"
    sidecar.write_text("def score_user() -> int:\n    return 1\n", encoding="utf-8")

    def failing_mypycify(paths: list[str]) -> list[object]:
        raise RuntimeError(f"compiler missing for {paths[0]}")

    monkeypatch.setattr(mypyc_backend, "mypycify", failing_mypycify)

    result = mypyc_backend.build_sidecars(
        (sidecar,),
        project_root=tmp_path,
        build_dir=tmp_path / ".atoll" / "build",
        source_roots=(tmp_path,),
    )

    assert result.success is False
    assert result.stderr.startswith("NATIVE_BUILD_ENV_ERROR")


def test_build_sidecars_handles_external_source_without_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sidecars outside the project root still use an absolute source path."""
    sidecar = tmp_path.parent / "_external_atoll.py"
    sidecar.write_text("def score_user() -> int:\n    return 1\n", encoding="utf-8")

    def fake_mypycify(paths: list[str]) -> list[object]:
        assert paths == [str(sidecar)]
        return []

    monkeypatch.setattr(mypyc_backend, "mypycify", fake_mypycify)
    monkeypatch.setattr(mypyc_backend, "build_ext", FakeBuildExt)

    result = mypyc_backend.build_sidecars(
        (sidecar,),
        project_root=tmp_path,
        build_dir=tmp_path / ".atoll" / "build",
        source_roots=(),
    )

    assert result.success is False
    assert result.stderr == "mypyc build completed but no extension artifacts were found"
    sidecar.unlink()
