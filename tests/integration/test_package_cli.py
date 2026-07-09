"""Integration tests for source-clean package artifact builds."""

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


def test_package_command_builds_install_tree_and_wheel_without_source_edits(
    tmp_path: Path,
) -> None:
    """`atoll package` creates installable artifacts without patching the checkout."""
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    source_path = project_root / "src" / "app" / "ranking.py"
    original_source = source_path.read_text(encoding="utf-8")

    exit_code = main(
        [
            "package",
            "app.ranking",
            "--root",
            str(project_root),
            "--output",
            str(output_dir),
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


def _extension_artifacts(directory: Path, stem: str) -> tuple[Path, ...]:
    return tuple(
        path
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
        for path in directory.glob(f"{stem}*{suffix}")
    )
