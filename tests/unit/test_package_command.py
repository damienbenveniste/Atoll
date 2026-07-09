"""Unit tests for source-clean package artifact helpers."""

from __future__ import annotations

import importlib.machinery
import shutil
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

import pytest

from atoll.commands import package as package_command
from atoll.models import CompileAttempt, EnabledIslandConfig, ModuleId
from atoll.project import DiscoveredProject, discover_project

FIXTURE_ROOT = Path("tests/fixtures/simple_project")


class _Metadata(Protocol):
    name: str
    version: str
    requires_python: str | None
    dependencies: tuple[str, ...]


class _ProjectMetadataFactory(Protocol):
    def __call__(
        self,
        *,
        name: str,
        version: str,
        requires_python: str | None,
        dependencies: tuple[str, ...],
    ) -> _Metadata: ...


class _WriteWheel(Protocol):
    def __call__(
        self,
        *,
        install_root: Path,
        output_dir: Path,
        metadata: _Metadata,
    ) -> Path: ...


def _package_attr(name: str) -> object:
    return vars(package_command)[name]


_ProjectMetadata = cast(
    _ProjectMetadataFactory,
    _package_attr("_ProjectMetadata"),
)
_copy_install_payload = cast(
    Callable[[tuple[Path, ...], Path], None],
    _package_attr("_copy_install_payload"),
)
_copy_atoll_artifacts = cast(
    Callable[[tuple[Path, ...], Path], None],
    _package_attr("_copy_atoll_artifacts"),
)
_copy_if_different = cast(
    Callable[[Path, Path], None],
    _package_attr("_copy_if_different"),
)
_copy_source_roots = cast(
    Callable[[DiscoveredProject, Path], tuple[Path, ...]],
    _package_attr("_copy_source_roots"),
)
_artifact_dir = cast(
    Callable[[EnabledIslandConfig], Path],
    _package_attr("_artifact_dir"),
)
_find_module = cast(
    Callable[[tuple[ModuleId, ...], str], ModuleId],
    _package_attr("_find_module"),
)
_mapping = cast(
    Callable[[object], dict[str, object]],
    _package_attr("_mapping"),
)
_project_metadata = cast(
    Callable[[Path], _Metadata],
    _package_attr("_project_metadata"),
)
_relative_source_root = cast(
    Callable[[Path, Path], Path],
    _package_attr("_relative_source_root"),
)
_reset_dir = cast(Callable[[Path], None], _package_attr("_reset_dir"))
_sequence = cast(
    Callable[[object], tuple[object, ...]],
    _package_attr("_sequence"),
)
_staged_module = cast(
    Callable[[ModuleId, DiscoveredProject, tuple[Path, ...]], ModuleId],
    _package_attr("_staged_module"),
)
_string = cast(Callable[[object], str | None], _package_attr("_string"))
_wheel_payload = cast(
    Callable[[Path], tuple[tuple[str, Path], ...]],
    _package_attr("_wheel_payload"),
)
_write_wheel = cast(_WriteWheel, _package_attr("_write_wheel"))


def test_package_reports_build_failure_without_source_edits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Package mode reports mypyc failures and still leaves checkout source untouched."""
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    source_path = project_root / "src" / "app" / "ranking.py"
    original_source = source_path.read_text(encoding="utf-8")

    def failing_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        assert args
        assert kwargs
        return CompileAttempt(
            success=False,
            command=("mypyc",),
            stdout="",
            stderr="MYPYC_TYPE_ERROR: fixture",
            artifact_paths=(),
            duration_seconds=0.0,
        )

    monkeypatch.setattr(package_command, "build_sidecars", failing_build_sidecars)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
        )
    )

    assert result.success is False
    assert result.error == "MYPYC_TYPE_ERROR: fixture"
    assert result.wheel_path is None
    assert source_path.read_text(encoding="utf-8") == original_source
    assert (output_dir / "build").exists()
    assert not (output_dir / "install").exists()
    assert result.cleanup_removed == (output_dir / "install",)
    assert result.cleanup_kept == (output_dir / "build",)


def test_package_whole_project_retries_and_skips_failed_islands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whole-project package mode keeps buildable islands when one island fails."""
    project_root = tmp_path / "project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    package_dir = project_root / "src" / "app"
    ranking_source = package_dir / "ranking.py"
    (package_dir / "good.py").write_text(ranking_source.read_text(encoding="utf-8"))
    (package_dir / "bad.py").write_text(ranking_source.read_text(encoding="utf-8"))
    ranking_source.unlink()
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]

    def mixed_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        assert kwargs
        assert args
        paths = cast(tuple[Path, ...], args[0])
        assert paths
        if len(paths) > 1:
            return CompileAttempt(
                success=False,
                command=("mypyc", "batch"),
                stdout="",
                stderr="MYPYC_TYPE_ERROR: batch failed",
                artifact_paths=(),
                duration_seconds=0.1,
            )
        path = next(iter(paths))
        if path.stem.endswith("_good"):
            artifact = tmp_path / f"{path.stem}{suffix}"
            artifact.write_text("binary", encoding="utf-8")
            return CompileAttempt(
                success=True,
                command=("mypyc", str(path)),
                stdout="",
                stderr="",
                artifact_paths=(artifact,),
                duration_seconds=0.1,
            )
        return CompileAttempt(
            success=False,
            command=("mypyc", str(path)),
            stdout="",
            stderr="MYPYC_TYPE_ERROR: bad failed",
            artifact_paths=(),
            duration_seconds=0.1,
        )

    monkeypatch.setattr(package_command, "build_sidecars", mixed_build_sidecars)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            output_dir=output_dir,
            keep_install_tree=True,
        )
    )

    good_text = (output_dir / "install" / "app" / "good.py").read_text(encoding="utf-8")
    bad_text = (output_dir / "install" / "app" / "bad.py").read_text(encoding="utf-8")
    assert result.success is True
    assert result.install_tree_kept is True
    assert result.cleanup_removed == (output_dir / "build",)
    assert result.cleanup_kept == (output_dir / "install",)
    assert tuple(island.source_module for island in result.islands) == ("app.good",)
    assert tuple(failure.island.source_module for failure in result.skipped) == ("app.bad",)
    assert "Initial batch build failed; retried islands individually" in result.build.stdout
    assert "# BEGIN ATOLL MANAGED: app.good" in good_text
    assert "# BEGIN ATOLL MANAGED" not in bad_text
    assert (output_dir / "install" / ".atoll" / "artifacts" / f"_atoll_app_good{suffix}").exists()
    assert result.wheel_path is not None


def test_package_whole_project_reports_zero_successful_retries_concisely(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whole-project package mode reports one representative error when all retries fail."""
    project_root = tmp_path / "project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    package_dir = project_root / "src" / "app"
    ranking_source = package_dir / "ranking.py"
    (package_dir / "first.py").write_text(ranking_source.read_text(encoding="utf-8"))
    (package_dir / "second.py").write_text(ranking_source.read_text(encoding="utf-8"))
    ranking_source.unlink()

    def failing_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        assert kwargs
        assert args
        paths = cast(tuple[Path, ...], args[0])
        assert paths
        if len(paths) > 1:
            return CompileAttempt(
                success=False,
                command=("mypyc", "batch"),
                stdout="",
                stderr="MYPYC_TYPE_ERROR: batch failed",
                artifact_paths=(),
                duration_seconds=0.1,
            )
        path = next(iter(paths))
        return CompileAttempt(
            success=False,
            command=("mypyc", str(path)),
            stdout="",
            stderr=f"MYPYC_TYPE_ERROR: {path.stem} failed",
            artifact_paths=(),
            duration_seconds=0.1,
        )

    monkeypatch.setattr(package_command, "build_sidecars", failing_build_sidecars)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            output_dir=output_dir,
            keep_install_tree=True,
        )
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.startswith("No selected islands compiled")
    assert result.error.count("MYPYC_TYPE_ERROR") == 1
    assert result.cleanup_removed == (output_dir / "install",)
    assert result.cleanup_kept == (output_dir / "build",)
    assert tuple(failure.island.source_module for failure in result.skipped) == (
        "app.first",
        "app.second",
    )


def test_package_attempts_typevar_blocked_selected_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Module-level TypeVar blockers do not prevent trying clean candidate sidecars."""
    project_root = tmp_path / "project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    bad_module = project_root / "src" / "app" / "typing_features.py"
    bad_module.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "from typing_extensions import TypeVar",
                "",
                "T = TypeVar('T', infer_variance=True)",
                "",
                "def helper(value: int) -> int:",
                "    return value + 1",
                "",
                "def candidate(value: int) -> int:",
                "    adjusted = helper(value)",
                "    return adjusted",
                "",
            ]
        ),
        encoding="utf-8",
    )

    build_calls: list[tuple[Path, ...]] = []

    def failing_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        assert kwargs
        paths = cast(tuple[Path, ...], args[0])
        build_calls.append(paths)
        return CompileAttempt(
            success=False,
            command=("mypyc", *(str(path) for path in paths)),
            stdout="",
            stderr="MYPYC_TYPE_ERROR: generated sidecar failed",
            artifact_paths=(),
            duration_seconds=0.1,
        )

    monkeypatch.setattr(package_command, "build_sidecars", failing_build_sidecars)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.typing_features",
            output_dir=output_dir,
        )
    )

    assert result.success is False
    assert build_calls
    assert result.error == "MYPYC_TYPE_ERROR: generated sidecar failed"
    assert result.preflight_skipped == ()
    assert (output_dir / "build").exists()
    assert result.cleanup_removed == (output_dir / "install",)
    assert result.cleanup_kept == (output_dir / "build",)


def test_package_whole_project_attempts_typevar_blocked_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whole-project package mode tries clean candidates in modules with TypeVar blockers."""
    project_root = tmp_path / "project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    package_dir = project_root / "src" / "app"
    clean_source = package_dir / "ranking.py"
    original_clean_source = clean_source.read_text(encoding="utf-8")
    (package_dir / "blocked.py").write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "from typing_extensions import TypeVar",
                "",
                "T = TypeVar('T', default=str)",
                "",
                "def helper(value: int) -> int:",
                "    return value + 1",
                "",
                "def candidate(value: int) -> int:",
                "    adjusted = helper(value)",
                "    return adjusted",
                "",
            ]
        ),
        encoding="utf-8",
    )
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]

    def successful_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        assert kwargs
        paths = cast(tuple[Path, ...], args[0])
        artifacts: list[Path] = []
        for path in paths:
            artifact = tmp_path / f"{path.stem}{suffix}"
            artifact.write_text("binary", encoding="utf-8")
            artifacts.append(artifact)
        return CompileAttempt(
            success=True,
            command=("mypyc",),
            stdout="",
            stderr="",
            artifact_paths=tuple(artifacts),
            duration_seconds=0.1,
        )

    monkeypatch.setattr(package_command, "build_sidecars", successful_build_sidecars)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            output_dir=output_dir,
            keep_install_tree=True,
        )
    )

    assert result.success is True
    assert result.install_tree_kept is True
    assert result.cleanup_removed == (output_dir / "build",)
    assert result.cleanup_kept == (output_dir / "install",)
    assert {island.source_module for island in result.islands} == {
        "app.blocked",
        "app.ranking",
    }
    assert result.preflight_skipped == ()
    assert "# BEGIN ATOLL MANAGED: app.ranking" in (
        output_dir / "install" / "app" / "ranking.py"
    ).read_text(encoding="utf-8")
    assert "# BEGIN ATOLL MANAGED: app.blocked" in (
        output_dir / "install" / "app" / "blocked.py"
    ).read_text(encoding="utf-8")
    assert clean_source.read_text(encoding="utf-8") == original_clean_source


def test_package_helpers_handle_flat_source_roots(tmp_path: Path) -> None:
    """Flat source roots copy their contents into the build root."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    project = discover_project(tmp_path)
    build_root = tmp_path / "build"
    build_root.mkdir()

    staged_roots = _copy_source_roots(project, build_root)

    assert staged_roots == (build_root,)
    assert (build_root / "pkg" / "mod.py").exists()


def test_copy_install_payload_filters_package_files(tmp_path: Path) -> None:
    """Install payloads include importable files and native artifacts, not tests or docs."""
    source_root = tmp_path / "source"
    install_root = tmp_path / "install"
    package_dir = source_root / "pkg"
    tests_dir = source_root / "tests"
    package_dir.mkdir(parents=True)
    tests_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package_dir / "py.typed").write_text("", encoding="utf-8")
    (package_dir / "data.txt").write_text("skip", encoding="utf-8")
    (source_root / "top.py").write_text("VALUE = 2\n", encoding="utf-8")
    (source_root / "README.md").write_text("skip", encoding="utf-8")
    (tests_dir / "test_mod.py").write_text("skip", encoding="utf-8")
    native = package_dir / f"native{importlib.machinery.EXTENSION_SUFFIXES[0]}"
    native.write_text("", encoding="utf-8")

    _copy_install_payload((source_root,), install_root)

    assert (install_root / "pkg" / "__init__.py").exists()
    assert (install_root / "pkg" / "mod.py").exists()
    assert (install_root / "pkg" / "py.typed").exists()
    assert (install_root / "pkg" / native.name).exists()
    assert (install_root / "top.py").exists()
    assert not (install_root / "pkg" / "data.txt").exists()
    assert not (install_root / "README.md").exists()
    assert not (install_root / "tests" / "test_mod.py").exists()


def test_atoll_artifact_helpers_copy_artifacts_and_skip_same_file(tmp_path: Path) -> None:
    """Source-clean artifact helpers place compiled extensions under install `.atoll`."""
    source_root = tmp_path / "source"
    artifact_dir = source_root / ".atoll" / "artifacts"
    install_root = tmp_path / "install"
    artifact_dir.mkdir(parents=True)
    native = artifact_dir / f"_sidecar{importlib.machinery.EXTENSION_SUFFIXES[0]}"
    native.write_text("binary", encoding="utf-8")

    _copy_atoll_artifacts((tmp_path / "missing", source_root), install_root)
    copied = install_root / ".atoll" / "artifacts" / native.name
    _copy_if_different(copied, copied)

    package_sidecar = EnabledIslandConfig(
        source_module="pkg.mod",
        source_path=source_root / "pkg" / "mod.py",
        sidecar_module="pkg._sidecar",
        sidecar_path=source_root / "pkg" / "_sidecar.py",
        symbols=("func",),
    )
    external_sidecar = EnabledIslandConfig(
        source_module="pkg.mod",
        source_path=source_root / "pkg" / "mod.py",
        sidecar_module="pkg._sidecar",
        sidecar_path=source_root / ".atoll" / "sidecars" / "_sidecar.py",
        symbols=("func",),
    )

    assert copied.read_text(encoding="utf-8") == "binary"
    assert _artifact_dir(package_sidecar) == source_root / "pkg"
    assert _artifact_dir(external_sidecar) == source_root / ".atoll" / "artifacts"


def test_wheel_writer_replaces_existing_wheel_and_records_metadata(tmp_path: Path) -> None:
    """Wheel helper writes payload, metadata, WHEEL, and RECORD entries."""
    install_root = tmp_path / "install"
    package_dir = install_root / "pkg"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    output_dir = tmp_path / "dist"
    metadata = _ProjectMetadata(
        name="Demo-Project",
        version="1.0-rc1",
        requires_python=">=3.12",
        dependencies=("requests>=2",),
    )

    first = _write_wheel(install_root=install_root, output_dir=output_dir, metadata=metadata)
    second = _write_wheel(install_root=install_root, output_dir=output_dir, metadata=metadata)

    assert first == second
    with zipfile.ZipFile(second) as wheel:
        names = set(wheel.namelist())
        metadata_text = wheel.read("demo_project-1.0_rc1.dist-info/METADATA").decode()
    assert _wheel_payload(install_root) == (
        ("pkg/__init__.py", package_dir / "__init__.py"),
        ("pkg/mod.py", package_dir / "mod.py"),
    )
    assert "pkg/mod.py" in names
    assert "demo_project-1.0_rc1.dist-info/WHEEL" in names
    assert "demo_project-1.0_rc1.dist-info/RECORD" in names
    assert "Requires-Python: >=3.12" in metadata_text
    assert "Requires-Dist: requests>=2" in metadata_text


def test_project_metadata_falls_back_for_missing_or_dynamic_version(tmp_path: Path) -> None:
    """Project metadata falls back to stable Atoll values when version is dynamic."""
    fallback = _project_metadata(tmp_path)
    assert fallback.name == tmp_path.name
    assert fallback.version == "0+atoll"

    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "dynamic-project"',
                'dynamic = ["version"]',
                'requires-python = ">=3.12"',
                'dependencies = ["pydantic>=2"]',
            ]
        ),
        encoding="utf-8",
    )

    metadata = _project_metadata(project_root)
    assert metadata.name == "dynamic-project"
    assert metadata.version == "0+atoll"
    assert metadata.requires_python == ">=3.12"
    assert metadata.dependencies == ("pydantic>=2",)


def test_package_small_helpers_cover_fallbacks(tmp_path: Path) -> None:
    """Small helper fallbacks stay deterministic."""
    path = tmp_path / "existing"
    path.mkdir()
    (path / "old.txt").write_text("old", encoding="utf-8")
    _reset_dir(path)
    assert path.exists()
    assert not (path / "old.txt").exists()

    assert _relative_source_root(tmp_path, tmp_path / "src") == Path("src")
    outside_root = tmp_path.parent / "not-under-root"
    assert _relative_source_root(tmp_path, outside_root) != outside_root
    assert _mapping(None) == {}
    assert _sequence(None) == ()
    assert _string(1) is None


def test_package_helpers_report_missing_modules(tmp_path: Path) -> None:
    """Module lookup helpers fail clearly for impossible paths."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    project = DiscoveredProject(
        config=discover_project(project_root).config,
        modules=(),
    )

    with pytest.raises(ValueError, match="module not found"):
        _find_module((), "missing")
    with pytest.raises(ValueError, match="outside configured source roots"):
        _staged_module(
            ModuleId(name="missing", path=tmp_path / "outside.py"),
            project,
            (tmp_path / "stage",),
        )
