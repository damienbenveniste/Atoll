"""Tests for the programmatic mypyc build backend."""

from __future__ import annotations

import hashlib
import importlib.machinery
import os
from pathlib import Path

import pytest

from atoll.backends import mypyc as mypyc_backend
from atoll.models import ArtifactRecord, BackendCompileContext, CompilationUnit


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


class FakeBuildExtWithSupport(FakeBuildExt):
    """Build stand-in that writes a primary and support extension artifact."""

    support_path: Path | None = None

    def run(self) -> None:
        """Pretend mypyc emitted a sidecar extension and a helper extension."""
        super().run()
        if self.support_path is not None:
            self.support_path.write_text("", encoding="utf-8")


class FakeBuildExtWithNativeWarnings(FakeBuildExt):
    """Build stand-in that writes native compiler warnings to fd stderr."""

    def run(self) -> None:
        """Emit one ignored linker warning and one useful warning."""
        os.write(2, b"ld: warning: duplicate -rpath '/opt/anaconda3/lib' ignored\n")
        os.write(2, b"compiler: warning: useful diagnostic\n")
        super().run()


class FakeBuildExtDuplicateStems(FakeBuildExt):
    """Build stand-in that writes same-named extensions under two packages."""

    artifact_paths: tuple[Path, ...] = ()

    def run(self) -> None:
        """Write every configured nested extension artifact."""
        for artifact in self.artifact_paths:
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(artifact.parent.name.encode())


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
    """A successful backend run returns extension artifacts from `.atoll/artifacts`."""
    sidecar = tmp_path / ".atoll" / "sidecars" / "_atoll_app_ranking.py"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text("def score_user() -> int:\n    return 1\n", encoding="utf-8")
    FakeBuildExt.artifact_path = (tmp_path / ".atoll" / "artifacts").joinpath(
        f"{sidecar.stem}{importlib.machinery.EXTENSION_SUFFIXES[0]}"
    )

    def fake_mypycify(paths: list[str], *, target_dir: str | None = None) -> list[object]:
        assert paths == [".atoll/sidecars/_atoll_app_ranking.py"]
        assert target_dir == str(tmp_path / ".atoll" / "build" / "generated")
        return []

    def fail_artifact_records(
        units: tuple[CompilationUnit, ...],
        artifact_paths: tuple[Path, ...],
        artifact_dir: Path,
    ) -> tuple[ArtifactRecord, ...]:
        _ = (units, artifact_paths, artifact_dir)
        pytest.fail("legacy facade must not materialize structured records")

    monkeypatch.setattr(mypyc_backend, "mypycify", fake_mypycify)
    monkeypatch.setattr(mypyc_backend, "build_ext", FakeBuildExt)
    monkeypatch.setattr(
        mypyc_backend,
        "_artifact_records",
        fail_artifact_records,
    )

    result = mypyc_backend.build_sidecars(
        (sidecar,),
        project_root=tmp_path,
        build_dir=tmp_path / ".atoll" / "build",
        source_roots=(tmp_path,),
    )

    assert result.success is True
    assert FakeBuildExt.artifact_path is not None
    assert result.artifact_paths == (FakeBuildExt.artifact_path,)
    assert tuple(timing.name for timing in result.phase_timings) == (
        "mypycify",
        "build_ext",
        "artifact_discovery",
    )


def test_build_sidecars_reports_support_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful backend run includes helper extensions written by mypyc."""
    sidecar = tmp_path / ".atoll" / "sidecars" / "_atoll_app_ranking.py"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text("def score_user() -> int:\n    return 1\n", encoding="utf-8")
    artifact_dir = tmp_path / ".atoll" / "artifacts"
    FakeBuildExtWithSupport.artifact_path = artifact_dir.joinpath(
        f"{sidecar.stem}{importlib.machinery.EXTENSION_SUFFIXES[0]}"
    )
    FakeBuildExtWithSupport.support_path = artifact_dir.joinpath(
        f"{sidecar.stem}__mypyc{importlib.machinery.EXTENSION_SUFFIXES[0]}"
    )

    def fake_mypycify(paths: list[str], *, target_dir: str | None = None) -> list[object]:
        assert paths == [".atoll/sidecars/_atoll_app_ranking.py"]
        assert target_dir == str(tmp_path / ".atoll" / "build" / "generated")
        return []

    monkeypatch.setattr(mypyc_backend, "mypycify", fake_mypycify)
    monkeypatch.setattr(mypyc_backend, "build_ext", FakeBuildExtWithSupport)

    result = mypyc_backend.build_sidecars(
        (sidecar,),
        project_root=tmp_path,
        build_dir=tmp_path / ".atoll" / "build",
        source_roots=(tmp_path,),
    )

    assert result.success is True
    assert result.artifact_paths == (
        FakeBuildExtWithSupport.artifact_path,
        FakeBuildExtWithSupport.support_path,
    )


def test_mypyc_adapter_returns_structured_primary_and_support_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adapter compilation records ownership, install paths, digests, and ABI tags."""
    sidecar = tmp_path / ".atoll" / "sidecars" / "_atoll_app_ranking.py"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text("def score_user() -> int:\n    return 1\n", encoding="utf-8")
    artifact_dir = tmp_path / ".atoll" / "artifacts"
    FakeBuildExtWithSupport.artifact_path = artifact_dir.joinpath(
        f"{sidecar.stem}{importlib.machinery.EXTENSION_SUFFIXES[0]}"
    )
    FakeBuildExtWithSupport.support_path = artifact_dir.joinpath(
        f"{sidecar.stem}__mypyc{importlib.machinery.EXTENSION_SUFFIXES[0]}"
    )

    def fake_mypycify(paths: list[str], *, target_dir: str | None = None) -> list[object]:
        assert paths == [".atoll/sidecars/_atoll_app_ranking.py"]
        assert target_dir == str(tmp_path / ".atoll" / "build" / "generated")
        return []

    monkeypatch.setattr(mypyc_backend, "mypycify", fake_mypycify)
    monkeypatch.setattr(mypyc_backend, "build_ext", FakeBuildExtWithSupport)
    unit = CompilationUnit(
        region_id="app.ranking::score_user:fixture",
        backend="mypyc",
        logical_module="app._atoll_app_ranking",
        source_paths=(sidecar,),
        source_hash="fixture",
        members=(),
        install_relative_dir=".atoll/artifacts",
    )

    result = mypyc_backend.MypycBackend().compile(
        (unit,),
        BackendCompileContext(
            project_root=tmp_path,
            build_dir=tmp_path / ".atoll" / "build",
            source_roots=(tmp_path,),
        ),
    )

    assert result.attempt.success is True
    assert [record.role for record in result.artifacts] == ["primary", "support"]
    primary, support = result.artifacts
    assert primary.region_id == unit.region_id
    assert primary.logical_module == unit.logical_module
    assert primary.install_relative_path == (
        f".atoll/artifacts/{FakeBuildExtWithSupport.artifact_path.name}"
    )
    assert support.region_id == "__shared__"
    assert support.logical_module == f"{sidecar.stem}__mypyc"
    assert support.install_relative_path == (
        f".atoll/artifacts/{FakeBuildExtWithSupport.support_path.name}"
    )
    empty_digest = hashlib.sha256(b"").hexdigest()
    assert all(record.digest == empty_digest for record in result.artifacts)
    assert all(record.abi for record in result.artifacts)
    assert all(record.platform_tag for record in result.artifacts)


def test_artifact_ownership_uses_logical_module_for_duplicate_stems(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Artifacts from same-named modules retain the correct region and install path."""
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    artifact_dir = tmp_path / ".atoll" / "artifacts"
    first_artifact = artifact_dir / "first" / f"shared{suffix}"
    second_artifact = artifact_dir / "second" / f"shared{suffix}"
    first_source = tmp_path / "first" / "shared.py"
    second_source = tmp_path / "second" / "shared.py"
    first_source.parent.mkdir(parents=True)
    second_source.parent.mkdir(parents=True)
    first_source.write_text("def first() -> int:\n    return 1\n", encoding="utf-8")
    second_source.write_text("def second() -> int:\n    return 2\n", encoding="utf-8")
    FakeBuildExtDuplicateStems.artifact_paths = (first_artifact, second_artifact)

    def fake_mypycify(paths: list[str], *, target_dir: str | None = None) -> list[object]:
        assert paths == ["first/shared.py", "second/shared.py"]
        assert target_dir == str(tmp_path / ".atoll" / "build" / "generated")
        return []

    monkeypatch.setattr(mypyc_backend, "mypycify", fake_mypycify)
    monkeypatch.setattr(mypyc_backend, "build_ext", FakeBuildExtDuplicateStems)
    units = (
        CompilationUnit(
            region_id="first-region",
            backend="mypyc",
            logical_module="first.shared",
            source_paths=(first_source,),
            source_hash="first",
            members=(),
        ),
        CompilationUnit(
            region_id="second-region",
            backend="mypyc",
            logical_module="second.shared",
            source_paths=(second_source,),
            source_hash="second",
            members=(),
        ),
    )

    result = mypyc_backend.MypycBackend().compile(
        units,
        BackendCompileContext(
            project_root=tmp_path,
            build_dir=tmp_path / ".atoll" / "build",
            source_roots=(tmp_path,),
        ),
    )
    assert [(record.region_id, record.install_relative_path) for record in result.artifacts] == [
        ("first-region", f"first/shared{suffix}"),
        ("second-region", f"second/shared{suffix}"),
    ]


def test_build_sidecars_captures_and_filters_native_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Native compiler stderr is captured, with noisy duplicate rpath warnings filtered."""
    sidecar = tmp_path / ".atoll" / "sidecars" / "_atoll_app_ranking.py"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text("def score_user() -> int:\n    return 1\n", encoding="utf-8")
    FakeBuildExtWithNativeWarnings.artifact_path = (tmp_path / ".atoll" / "artifacts").joinpath(
        f"{sidecar.stem}{importlib.machinery.EXTENSION_SUFFIXES[0]}"
    )

    def fake_mypycify(paths: list[str], *, target_dir: str | None = None) -> list[object]:
        assert paths == [".atoll/sidecars/_atoll_app_ranking.py"]
        assert target_dir == str(tmp_path / ".atoll" / "build" / "generated")
        return []

    monkeypatch.setattr(mypyc_backend, "mypycify", fake_mypycify)
    monkeypatch.setattr(mypyc_backend, "build_ext", FakeBuildExtWithNativeWarnings)

    result = mypyc_backend.build_sidecars(
        (sidecar,),
        project_root=tmp_path,
        build_dir=tmp_path / ".atoll" / "build",
        source_roots=(tmp_path,),
    )

    captured = capsys.readouterr()
    assert result.success is True
    assert captured.err == ""
    assert "duplicate -rpath" not in result.stdout
    assert "useful diagnostic" in result.stdout


def test_build_sidecars_adds_source_roots_to_mypy_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sidecars outside source roots still let mypyc resolve target package imports."""
    source_root = tmp_path / "src"
    source_root.mkdir()
    sidecar = tmp_path / ".atoll" / "sidecars" / "_atoll_app_ranking.py"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text("def score_user() -> int:\n    return 1\n", encoding="utf-8")
    FakeBuildExt.artifact_path = (tmp_path / ".atoll" / "artifacts").joinpath(
        f"{sidecar.stem}{importlib.machinery.EXTENSION_SUFFIXES[0]}"
    )
    seen_mypy_path: list[str | None] = []
    seen_cache_dir: list[str | None] = []
    monkeypatch.setenv("MYPYPATH", "existing")
    monkeypatch.setenv("MYPY_CACHE_DIR", "existing_cache")

    def fake_mypycify(paths: list[str], *, target_dir: str | None = None) -> list[object]:
        assert paths == [".atoll/sidecars/_atoll_app_ranking.py"]
        assert target_dir == str(tmp_path / ".atoll" / "build" / "generated")
        seen_mypy_path.append(os.environ.get("MYPYPATH"))
        seen_cache_dir.append(os.environ.get("MYPY_CACHE_DIR"))
        return []

    monkeypatch.setattr(mypyc_backend, "mypycify", fake_mypycify)
    monkeypatch.setattr(mypyc_backend, "build_ext", FakeBuildExt)

    result = mypyc_backend.build_sidecars(
        (sidecar,),
        project_root=tmp_path,
        build_dir=tmp_path / ".atoll" / "build",
        source_roots=(source_root,),
    )

    assert result.success is True
    assert seen_mypy_path == [f"{source_root.resolve()}{os.pathsep}existing"]
    assert seen_cache_dir == [str(tmp_path / ".atoll" / "build" / "mypy_cache")]
    assert os.environ["MYPYPATH"] == "existing"
    assert os.environ["MYPY_CACHE_DIR"] == "existing_cache"


def test_build_sidecars_classifies_build_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build failures are returned as classified compile attempts."""
    sidecar = tmp_path / "_ranking_atoll.py"
    sidecar.write_text("def score_user() -> int:\n    return 1\n", encoding="utf-8")

    def failing_mypycify(paths: list[str], *, target_dir: str | None = None) -> list[object]:
        assert target_dir == str(tmp_path / ".atoll" / "build" / "generated")
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


def test_build_sidecars_captures_mypyc_system_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """mypyc diagnostic output is summarized instead of leaking to the terminal."""
    sidecar = tmp_path / "_ranking_atoll.py"
    sidecar.write_text("def score_user() -> int:\n    return 1\n", encoding="utf-8")

    def failing_mypycify(paths: list[str], *, target_dir: str | None = None) -> list[object]:
        assert target_dir == str(tmp_path / ".atoll" / "build" / "generated")
        print(f"{paths[0]}:1: error: fixture type failure  [misc]")
        print("dep.py:1: error: Cannot find implementation or library stub  [import-not-found]")
        raise SystemExit(1)

    monkeypatch.setattr(mypyc_backend, "mypycify", failing_mypycify)

    result = mypyc_backend.build_sidecars(
        (sidecar,),
        project_root=tmp_path,
        build_dir=tmp_path / ".atoll" / "build",
        source_roots=(tmp_path,),
    )

    captured = capsys.readouterr()
    log_path = tmp_path / ".atoll" / "build" / "mypyc.log"
    assert captured.out == ""
    assert result.success is False
    assert result.stderr.startswith("MYPYC_TYPE_ERROR")
    assert result.stderr.splitlines()[:2] == [
        "MYPYC_TYPE_ERROR: SystemExit(1)",
        "Captured 2 mypyc error line(s).",
    ]
    assert "Captured 2 mypyc error line(s)." in result.stderr
    assert str(log_path) in result.stderr
    assert "target project's environment" in result.stderr
    assert "fixture type failure" in log_path.read_text(encoding="utf-8")


def test_build_sidecars_classifies_native_build_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native build-looking failures are classified as environment failures."""
    sidecar = tmp_path / "_ranking_atoll.py"
    sidecar.write_text("def score_user() -> int:\n    return 1\n", encoding="utf-8")

    def failing_mypycify(paths: list[str], *, target_dir: str | None = None) -> list[object]:
        assert target_dir == str(tmp_path / ".atoll" / "build" / "generated")
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

    def fake_mypycify(paths: list[str], *, target_dir: str | None = None) -> list[object]:
        assert paths == [str(sidecar)]
        assert target_dir == str(tmp_path / ".atoll" / "build" / "generated")
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
