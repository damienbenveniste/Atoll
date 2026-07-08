"""Tests for Atoll's file-hash scan cache."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from atoll.cache import clear_scan_cache, scan_modules_with_cache
from atoll.project import discover_project

FIXTURE_ROOT = Path("tests/fixtures/simple_project")


def test_scan_modules_with_cache_hits_on_second_run(tmp_path: Path) -> None:
    """A repeated scan reuses cached AST facts when hashes match."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    project = discover_project(project_root)

    first_scans, first_stats = scan_modules_with_cache(project.config, project.modules)
    second_scans, second_stats = scan_modules_with_cache(project.config, project.modules)

    assert first_stats == {"hits": 0, "misses": 3}
    assert second_stats == {"hits": 3, "misses": 0}
    assert [scan.module.name for scan in first_scans] == [scan.module.name for scan in second_scans]
    assert (project_root / ".atoll" / "cache" / "index.json").exists()


def test_scan_cache_invalidates_changed_file(tmp_path: Path) -> None:
    """Changing one source file invalidates only that cached scan entry."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    scan_modules_with_cache(project.config, project.modules)
    ranking = project_root / "src" / "app" / "ranking.py"
    ranking.write_text(
        f"{ranking.read_text(encoding='utf-8')}\n\n{_ADDED_FUNCTION}", encoding="utf-8"
    )
    changed_project = discover_project(project_root)

    scans, stats = scan_modules_with_cache(changed_project.config, changed_project.modules)
    ranking_scan = next(scan for scan in scans if scan.module.name == "app.ranking")

    assert stats == {"hits": 2, "misses": 1}
    assert any(symbol.id.qualname == "added" for symbol in ranking_scan.symbols)


def test_scan_cache_recovers_from_corrupt_index(tmp_path: Path) -> None:
    """A corrupt cache index is ignored and rewritten."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    index_path = project_root / ".atoll" / "cache" / "index.json"
    index_path.parent.mkdir(parents=True)
    index_path.write_text("{not-json", encoding="utf-8")

    _, stats = scan_modules_with_cache(project.config, project.modules)
    data = json.loads(index_path.read_text(encoding="utf-8"))

    assert stats == {"hits": 0, "misses": 3}
    assert data["version"] == 1


def test_scan_cache_ignores_invalid_metadata(tmp_path: Path) -> None:
    """Wrong cache index versions and shapes are treated as cold caches."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    index_path = project_root / ".atoll" / "cache" / "index.json"
    index_path.parent.mkdir(parents=True)
    index_path.write_text('{"version": 0, "files": {}}', encoding="utf-8")

    _, version_stats = scan_modules_with_cache(project.config, project.modules)
    index_path.write_text('{"version": 1, "files": []}', encoding="utf-8")
    _, shape_stats = scan_modules_with_cache(project.config, project.modules)

    assert version_stats == {"hits": 0, "misses": 3}
    assert shape_stats == {"hits": 0, "misses": 3}


def test_scan_cache_can_be_cleared(tmp_path: Path) -> None:
    """The scan cache clear helper removes the cache index."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    scan_modules_with_cache(project.config, project.modules)
    index_path = project_root / ".atoll" / "cache" / "index.json"

    clear_scan_cache(project_root)

    assert not index_path.exists()


def test_scan_cache_handles_absolute_source_roots_outside_project(tmp_path: Path) -> None:
    """Absolute source roots outside the project root are cached and restored correctly."""
    project_root = tmp_path / "project"
    source_root = tmp_path / "external_src"
    package = source_root / "pkg"
    package.mkdir(parents=True)
    (package / "module.py").write_text(
        "def score(value: int) -> int:\n    return value\n", encoding="utf-8"
    )
    project = discover_project(project_root, source_roots=(source_root,))

    first_scans, first_stats = scan_modules_with_cache(project.config, project.modules)
    second_scans, second_stats = scan_modules_with_cache(project.config, project.modules)

    assert first_stats == {"hits": 0, "misses": 1}
    assert second_stats == {"hits": 1, "misses": 0}
    assert second_scans[0].module.path == first_scans[0].module.path.resolve()


_ADDED_FUNCTION = "def added(value: int) -> int:\n    return value\n"
