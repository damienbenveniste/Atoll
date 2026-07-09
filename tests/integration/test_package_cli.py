"""Integration tests for source-clean compile artifact builds."""

from __future__ import annotations

import importlib.machinery
import shutil
import zipfile
from pathlib import Path

import pytest

from atoll.cli import main
from atoll.models import EnabledIslandConfig, ProjectConfig
from atoll.runtime.verify import verify_islands

FIXTURE_ROOT = Path("tests/fixtures/simple_project")
EXIT_USAGE = 2


def test_compile_builds_install_tree_and_wheel_without_source_edits(
    tmp_path: Path,
) -> None:
    """`atoll compile` compiles without patching the checkout by default."""
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
        ]
    )

    install_root = output_dir / "install"
    install_source = install_root / "app" / "ranking.py"
    wheel_path = next(output_dir.glob("*.whl"))
    assert exit_code == 0
    assert source_path.read_text(encoding="utf-8") == original_source
    assert "# BEGIN ATOLL MANAGED" not in original_source
    assert "# BEGIN ATOLL MANAGED: app.ranking" in install_source.read_text(encoding="utf-8")
    assert not (output_dir / "build").exists()
    assert _extension_artifacts(install_root / "app", "_atoll_app_ranking")

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
                    sidecar_path=install_root / "app" / "_atoll_app_ranking.py",
                    symbols=("normalize_features", "score_user", "rank_candidates"),
                ),
            ),
        ),
        require_compiled=True,
    )[0]

    assert result.error is None
    assert result.compiled is True
    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())
    assert "app/ranking.py" in names
    assert "app/types.py" in names
    assert any(name.startswith("app/_atoll_app_ranking") and name.endswith(".so") for name in names)
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
    assert exit_code == 1
    assert "scan found no candidate islands" in captured.out


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


def test_compile_reports_mypyc_preflight_blockers(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Source-clean compile surfaces known mypyc blockers without retry noise."""
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "pkg" / "mod.py").write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "from typing_extensions import TypeVar",
                "",
                "T = TypeVar('T', default=str)",
                "",
                "def candidate(value: int) -> int:",
                "    return value + 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["compile", "--root", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "mypy/mypyc rejects typing constructs" in captured.out
    assert "mod.py:4" in captured.out
    assert "default" in captured.out


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
