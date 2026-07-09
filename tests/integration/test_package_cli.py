"""Integration tests for source-clean compile artifact builds."""

from __future__ import annotations

import importlib.machinery
import inspect
import json
import shutil
import zipfile
from pathlib import Path

import pytest

from atoll.cli import main
from atoll.commands.package import PackageCommandResult, PackagePreflightFailure
from atoll.models import (
    Blocker,
    CompileAttempt,
    EnabledIslandConfig,
    ModuleId,
    ModuleScan,
    ProjectConfig,
)
from atoll.runtime.verify import verify_islands

FIXTURE_ROOT = Path("tests/fixtures/simple_project")
EXIT_USAGE = 2
RANKING_SYMBOL_COUNT = 1
REJECTED_SYMBOL_COUNT = 2


def test_compile_builds_wheel_without_source_edits_or_kept_install_tree(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`atoll compile` keeps the checkout and output directory clean by default."""
    project_root = tmp_path / "simple_project"
    output_dir = project_root / ".atoll" / "dist"
    shutil.copytree(FIXTURE_ROOT, project_root)
    source_path = project_root / "src" / "app" / "ranking.py"
    original_source = source_path.read_text(encoding="utf-8")

    exit_code = main(["compile", "app.ranking", "--root", str(project_root)])

    captured = capsys.readouterr()
    wheel_path = next(output_dir.glob("*.whl"))
    report_json_path = project_root / ".atoll" / "compile-report.json"
    report_markdown_path = project_root / ".atoll" / "compile-report.md"
    report = json.loads(report_json_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert source_path.read_text(encoding="utf-8") == original_source
    assert "# BEGIN ATOLL MANAGED" not in original_source
    assert not (output_dir / "build").exists()
    assert not (output_dir / "install").exists()
    assert report_markdown_path.exists()
    assert not (project_root / ".atoll" / "compilation-report.json").exists()
    assert report["mode"] == "source-clean"
    assert report["wheel_path"] == f".atoll/dist/{wheel_path.name}"
    assert report["summary"]["islands"] == 1
    assert report["summary"]["symbols"] == RANKING_SYMBOL_COUNT
    assert report["summary"]["native_ready_symbols"] == 1
    assert report["summary"]["native_rejected_symbols"] == REJECTED_SYMBOL_COUNT
    readiness_by_symbol = {
        readiness["symbol"]: readiness for readiness in report["native_readiness"]
    }
    assert readiness_by_symbol["normalize_features"]["eligible"] is True
    assert readiness_by_symbol["score_user"]["eligible"] is False
    assert readiness_by_symbol["rank_candidates"]["eligible"] is False
    assert report["build"]["cache_status"] == "miss"
    assert any(timing["name"] == "mypycify" for timing in report["build"]["phase_timings"])
    assert any(timing["name"] == "build_ext" for timing in report["build"]["phase_timings"])
    assert report["cleanup"]["removed"] == [".atoll/dist/build", ".atoll/dist/install"]
    assert report["cleanup"]["kept"] == []
    assert "Source-clean compile builds a wheel" in report_markdown_path.read_text(encoding="utf-8")
    assert "Install tree:" not in captured.out
    assert "Compile reports:" in captured.out
    assert "Atoll compile [" in captured.err
    assert "discovering project" in captured.err
    assert "running mypyc batch" in captured.err
    assert "cleaned temporary outputs" in captured.err
    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())
    assert "app/ranking.py" in names
    assert any(
        name.startswith(".atoll/artifacts/_atoll_app_ranking") and name.endswith(".so")
        for name in names
    )
    artifact_names = sorted(
        name for name in names if name.startswith(".atoll/artifacts/") and name.endswith(".so")
    )
    assert report["summary"]["artifacts"] == len(artifact_names)
    assert sorted(report["build"]["artifacts"]) == artifact_names


def test_compile_uses_cache_on_unchanged_second_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A repeated source-clean compile restores unchanged artifacts from cache."""
    project_root = tmp_path / "simple_project"
    output_dir = project_root / ".atoll" / "dist"
    shutil.copytree(FIXTURE_ROOT, project_root)

    first_exit = main(["compile", "app.ranking", "--root", str(project_root)])
    first_report = json.loads((project_root / ".atoll" / "compile-report.json").read_text())
    first_capture = capsys.readouterr()
    second_exit = main(["compile", "app.ranking", "--root", str(project_root)])
    second_capture = capsys.readouterr()

    wheel_path = next(output_dir.glob("*.whl"))
    second_report = json.loads((project_root / ".atoll" / "compile-report.json").read_text())
    assert first_exit == 0
    assert second_exit == 0
    assert first_report["build"]["cache_status"] == "miss"
    assert second_report["build"]["cache_status"] == "hit"
    assert "compile cache miss" in first_capture.err
    assert "compile cache hit" in second_capture.err
    assert [timing["name"] for timing in second_report["build"]["phase_timings"]] == [
        "cache_lookup",
        "cache_restore",
    ]
    assert not (output_dir / "build").exists()
    assert not (output_dir / "install").exists()
    with zipfile.ZipFile(wheel_path) as wheel:
        assert any(
            name.startswith(".atoll/artifacts/_atoll_app_ranking") for name in wheel.namelist()
        )


def test_compile_can_keep_install_tree_for_debugging(
    tmp_path: Path,
) -> None:
    """`--keep-install-tree` preserves the generated tree for direct routing checks."""
    project_root = tmp_path / "simple_project"
    output_dir = project_root / ".atoll" / "dist"
    shutil.copytree(FIXTURE_ROOT, project_root)
    source_path = project_root / "src" / "app" / "ranking.py"
    original_source = source_path.read_text(encoding="utf-8")

    exit_code = main(
        [
            "compile",
            "app.ranking",
            "--root",
            str(project_root),
            "--keep-install-tree",
        ]
    )

    install_root = output_dir / "install"
    install_source = install_root / "app" / "ranking.py"
    wheel_path = next(output_dir.glob("*.whl"))
    report = json.loads((project_root / ".atoll" / "compile-report.json").read_text())
    assert exit_code == 0
    assert source_path.read_text(encoding="utf-8") == original_source
    assert "# BEGIN ATOLL MANAGED" not in original_source
    assert "# BEGIN ATOLL MANAGED: app.ranking" in install_source.read_text(encoding="utf-8")
    assert not (output_dir / "build").exists()
    assert not (install_root / "app" / "_atoll_app_ranking.py").exists()
    assert _extension_artifacts(install_root / ".atoll" / "artifacts", "_atoll_app_ranking")
    assert report["cleanup"]["removed"] == [".atoll/dist/build"]
    assert report["cleanup"]["kept"] == [".atoll/dist/install"]

    result = verify_islands(
        ProjectConfig(
            root=install_root,
            source_roots=(install_root,),
            backend="mypyc",
            cache_dir=output_dir / "cache",
            report_dir=output_dir,
            islands=(
                EnabledIslandConfig(
                    source_module="app.ranking",
                    source_path=install_source,
                    sidecar_module="app._atoll_app_ranking",
                    sidecar_path=install_root / ".atoll" / "sidecars" / "_atoll_app_ranking.py",
                    symbols=("normalize_features",),
                ),
            ),
        ),
        require_compiled=True,
    )[0]

    assert result.error is None
    assert result.compiled is True
    ranking = importlib.import_module("app.ranking")
    normalize_features = ranking.normalize_features
    native_target = normalize_features.__atoll_compiled_target__
    assert inspect.isfunction(normalize_features)
    wrapped_normalize_features = inspect.unwrap(normalize_features)
    assert inspect.signature(normalize_features) == inspect.signature(wrapped_normalize_features)
    assert normalize_features.__name__ == "normalize_features"
    assert normalize_features.__qualname__ == "normalize_features"
    assert normalize_features.__module__ == "app.ranking"
    assert normalize_features.__doc__ == "Normalize feature values by their total."
    assert normalize_features.__annotations__ == wrapped_normalize_features.__annotations__
    assert native_target.__module__ == "app._atoll_app_ranking"
    assert not hasattr(ranking, "_atoll_bind_compiled")
    assert not hasattr(ranking, "_atoll_functools")
    assert not hasattr(ranking, "_atoll_inspect")
    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())
    assert "app/ranking.py" in names
    assert "app/types.py" in names
    assert any(
        name.startswith(".atoll/artifacts/_atoll_app_ranking") and name.endswith(".so")
        for name in names
    )
    assert any(name.endswith(".dist-info/METADATA") for name in names)
    assert any(name.endswith(".dist-info/WHEEL") for name in names)
    assert any(name.endswith(".dist-info/RECORD") for name in names)


def test_package_command_reports_no_candidates(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Package mode reports when no modules contain candidate islands."""
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "__init__.py").write_text("", encoding="utf-8")

    exit_code = main(["package", "--root", str(tmp_path), "--output", str(tmp_path / "out")])

    captured = capsys.readouterr()
    report = json.loads((tmp_path / ".atoll" / "compile-report.json").read_text())
    assert exit_code == 1
    assert "scan found no candidate islands" in captured.out
    assert report["success"] is False
    assert report["build"]["stderr"] == "scan found no candidate islands"


def test_compile_reports_no_candidates_from_source_clean_default(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Source-clean compile reports when no modules contain candidate islands."""
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "__init__.py").write_text("", encoding="utf-8")

    exit_code = main(["compile", "--root", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "scan found no candidate islands" in captured.out


def test_compile_reports_preflight_skipped_modules(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Source-clean compile reports modules skipped before mypyc runs."""
    module_path = tmp_path / "src" / "pkg" / "blocked.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("", encoding="utf-8")
    blocker = Blocker(
        severity="hard",
        code="MYPYC_UNSUPPORTED_TYPEVAR",
        message="TypeVar keyword(s) default are rejected by mypyc",
        lineno=4,
        symbol=None,
    )
    scan = ModuleScan(
        module=ModuleId(name="pkg.blocked", path=module_path),
        imports=(),
        constants=(),
        symbols=(),
        blockers=(blocker,),
        top_level_statement_lines=(),
    )

    def fake_execute_package(*_args: object, **_kwargs: object) -> PackageCommandResult:
        return PackageCommandResult(
            success=True,
            project_root=tmp_path,
            output_dir=tmp_path / "out",
            install_root=tmp_path / "out" / "install",
            wheel_path=tmp_path / "out" / "pkg-0-atoll.whl",
            islands=(),
            build=CompileAttempt(
                success=True,
                command=("mypyc",),
                stdout="",
                stderr="",
                artifact_paths=(),
                duration_seconds=0.0,
            ),
            preflight_skipped=(PackagePreflightFailure(scan=scan, blockers=(blocker,)),),
        )

    monkeypatch.setattr("atoll.cli.execute_package", fake_execute_package)

    exit_code = main(["compile", "--root", str(tmp_path)])

    captured = capsys.readouterr()
    report = json.loads((tmp_path / ".atoll" / "compile-report.json").read_text())
    assert exit_code == 0
    assert "Skipped 1 module(s) with known mypyc typing blockers." in captured.out
    assert "- pkg.blocked: line 4: TypeVar keyword(s) default are rejected by mypyc" in captured.out
    assert report["summary"]["preflight_blockers"] == 1
    assert report["preflight_blockers"][0]["module"] == "pkg.blocked"


def test_compile_rejects_typevar_erased_functions_before_mypyc(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """TypeVar-erased generated functions are not reported as native compilation."""
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        "\n".join(["[project]", 'name = "pkg"', 'version = "0.1.0"', ""]),
        encoding="utf-8",
    )
    output_dir = tmp_path / ".atoll" / "dist"
    output_dir.mkdir(parents=True)
    (output_dir / "pkg-0.1.0-cp311-cp311-macosx_11_0_arm64.whl").write_bytes(b"stale")
    (tmp_path / "src" / "pkg" / "mod.py").write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "from typing_extensions import TypeVar",
                "",
                "T = TypeVar('T', default=str)",
                "",
                "def helper(value: T) -> T:",
                "    return value",
                "",
                "def candidate(value: T) -> T:",
                "    adjusted = helper(value)",
                "    return adjusted",
                "",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["compile", "--root", str(tmp_path)])

    captured = capsys.readouterr()
    report = json.loads((tmp_path / ".atoll" / "compile-report.json").read_text())
    assert exit_code == 1
    assert "Rejected 2 scan candidate symbol(s)" in captured.out
    assert "mypyc was not invoked" in captured.out
    assert report["success"] is False
    assert report["summary"]["preflight_blockers"] == 0
    assert report["summary"]["symbols"] == 0
    assert report["summary"]["typed_regions"] == 1
    assert report["typed_regions"][0]["decisions"][0]["action"] == "fallback"
    assert report["summary"]["native_rejected_symbols"] == REJECTED_SYMBOL_COUNT
    assert report["build"]["command"] == []
    assert not tuple(output_dir.glob("*.whl"))


def test_compile_in_place_rejects_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`compile --output` only applies to the source-clean artifact mode."""
    exit_code = main(
        ["compile", "--root", str(tmp_path), "--in-place", "--output", str(tmp_path / "out")]
    )

    captured = capsys.readouterr()
    assert exit_code == EXIT_USAGE
    assert "--output cannot be used with --in-place" in captured.out


def test_compile_in_place_rejects_keep_install_tree(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--keep-install-tree` only applies to source-clean artifact mode."""
    exit_code = main(["compile", "--root", str(tmp_path), "--in-place", "--keep-install-tree"])

    captured = capsys.readouterr()
    assert exit_code == EXIT_USAGE
    assert "--keep-install-tree cannot be used with --in-place" in captured.out


def test_compile_source_clean_default_rejects_test_gate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The semantic test gate remains scoped to in-place compile mode."""
    exit_code = main(["compile", "--root", str(tmp_path), "--test", "pytest"])

    captured = capsys.readouterr()
    assert exit_code == EXIT_USAGE
    assert "--test requires --in-place" in captured.out


def _extension_artifacts(directory: Path, stem: str) -> tuple[Path, ...]:
    return tuple(
        path
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
        for path in directory.glob(f"{stem}*{suffix}")
    )
