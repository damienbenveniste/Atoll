"""Tests for Atoll project discovery."""

import tomllib
from pathlib import Path

import pytest

from atoll.models import CompileConfig
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
    assert discovered.config.compile == CompileConfig()


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


def test_discover_project_honors_nonstandard_setuptools_source_root(tmp_path: Path) -> None:
    """Setuptools ``where`` metadata maps a nonstandard layout to import names."""
    package_dir = tmp_path / "lib" / "pkg"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "decoy.py").write_text("VALUE = 2\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[tool.setuptools.packages.find]\nwhere = ["lib"]\n',
        encoding="utf-8",
    )

    discovered = discover_project(tmp_path)

    assert discovered.config.source_roots == ((tmp_path / "lib").resolve(),)
    assert [module.name for module in discovered.modules] == ["pkg", "pkg.module"]


@pytest.mark.parametrize(
    "configuration",
    [
        '[tool.setuptools.package-dir]\n"" = "python"\n',
        '[tool.setuptools]\npackage_dir = {"" = "python"}\n',
        '[tool.hatch.build]\nsources = ["python"]\n',
        '[tool.hatch.build.targets.wheel]\nsources = ["python"]\n',
        '[tool.hatch.build.targets.wheel]\npackages = ["python/pkg"]\n',
        '[tool.hatch.build.sources]\n"python/pkg" = "pkg"\n',
        '[tool.hatch.build.targets.wheel.sources]\n"python/pkg" = "pkg"\n',
        '[[tool.poetry.packages]]\ninclude = "pkg"\nfrom = "python"\n',
        '[tool.pdm.build]\npackage-dir = "python"\n',
        '[tool.maturin]\npython-source = "python"\n',
    ],
)
def test_discover_project_honors_supported_backend_source_roots(
    tmp_path: Path,
    configuration: str,
) -> None:
    """Common backend metadata identifies its declared import root.

    Args:
        tmp_path: Isolated project root supplied by pytest.
        configuration: Backend-specific TOML source-root declaration.
    """
    package_dir = tmp_path / "python" / "pkg"
    package_dir.mkdir(parents=True)
    (package_dir / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(configuration, encoding="utf-8")

    discovered = discover_project(tmp_path)

    assert discovered.config.source_roots == ((tmp_path / "python").resolve(),)
    assert [module.name for module in discovered.modules] == ["pkg.module"]


def test_discover_project_keeps_src_precedence_over_backend_metadata(tmp_path: Path) -> None:
    """The established ``src`` convention remains authoritative when present."""
    package_dir = tmp_path / "src" / "pkg"
    package_dir.mkdir(parents=True)
    (package_dir / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "lib").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        '[tool.setuptools.packages.find]\nwhere = ["lib"]\n',
        encoding="utf-8",
    )

    discovered = discover_project(tmp_path)

    assert discovered.config.source_roots == ((tmp_path / "src").resolve(),)
    assert [module.name for module in discovered.modules] == ["pkg.module"]


def test_discover_project_ignores_packaging_source_root_escape(tmp_path: Path) -> None:
    """Build metadata cannot make automatic discovery scan outside the project."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[tool.setuptools.packages.find]\nwhere = [".."]\n',
        encoding="utf-8",
    )

    discovered = discover_project(tmp_path)

    assert discovered.config.source_roots == (tmp_path.resolve(),)
    assert [module.name for module in discovered.modules] == ["pkg.module"]


def test_discover_project_surfaces_malformed_packaging_metadata(tmp_path: Path) -> None:
    """Source-root probing does not hide the compile config loader's TOML error."""
    (tmp_path / "pyproject.toml").write_text("[tool.setuptools\n", encoding="utf-8")

    with pytest.raises(tomllib.TOMLDecodeError):
        discover_project(tmp_path)


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
