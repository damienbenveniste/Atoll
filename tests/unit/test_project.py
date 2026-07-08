"""Tests for Atoll project discovery."""

from pathlib import Path

from atoll.project import discover_project, module_name_for_path

FIXTURE_ROOT = Path("tests/fixtures/simple_project")


def test_module_name_for_regular_module() -> None:
    """Module names are resolved relative to the selected source root."""
    source_root = (FIXTURE_ROOT / "src").resolve()
    module_path = source_root / "app" / "ranking.py"

    assert module_name_for_path(module_path, source_root) == "app.ranking"


def test_module_name_for_package_init() -> None:
    """Package `__init__` files resolve to the package import name."""
    source_root = (FIXTURE_ROOT / "src").resolve()
    module_path = source_root / "app" / "__init__.py"

    assert module_name_for_path(module_path, source_root) == "app"


def test_discover_project_defaults_to_src_root() -> None:
    """The scanner prefers `src/` when it is present."""
    discovered = discover_project(FIXTURE_ROOT)

    assert [module.name for module in discovered.modules] == [
        "app",
        "app.ranking",
        "app.types",
    ]
    assert discovered.config.report_dir == FIXTURE_ROOT.resolve() / ".atoll"


def test_discover_project_honors_max_files() -> None:
    """The max-files guard caps scan size deterministically."""
    discovered = discover_project(FIXTURE_ROOT, max_files=1)

    assert len(discovered.modules) == 1


def test_discover_project_accepts_absolute_source_root(tmp_path: Path) -> None:
    """Explicit absolute source roots bypass default root detection."""
    source_root = tmp_path / "custom_src"
    package_dir = source_root / "pkg"
    package_dir.mkdir(parents=True)
    (package_dir / "module.py").write_text("VALUE = 1\n", encoding="utf-8")

    discovered = discover_project(tmp_path, source_roots=(source_root,))

    assert [module.name for module in discovered.modules] == ["pkg.module"]


def test_discover_project_ignores_tests_and_cache_dirs(tmp_path: Path) -> None:
    """Discovery excludes tests, cache directories, and test-prefixed modules."""
    package_dir = tmp_path / "pkg"
    tests_dir = tmp_path / "tests"
    cache_dir = tmp_path / ".venv"
    package_dir.mkdir()
    tests_dir.mkdir()
    cache_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "test_skip.py").write_text("", encoding="utf-8")
    (tests_dir / "test_sample.py").write_text("", encoding="utf-8")
    (cache_dir / "ignored.py").write_text("", encoding="utf-8")

    discovered = discover_project(tmp_path)

    assert [module.name for module in discovered.modules] == ["pkg"]
