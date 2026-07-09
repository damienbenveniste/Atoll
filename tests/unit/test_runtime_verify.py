"""Tests for runtime Atoll verification."""

from __future__ import annotations

from pathlib import Path

from atoll.models import EnabledIslandConfig, ProjectConfig
from atoll.runtime.verify import verify_islands


def _config(tmp_path: Path, module_source: str, symbols: tuple[str, ...]) -> ProjectConfig:
    package = tmp_path / "src" / "app"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    source_path = package / "ranking.py"
    source_path.write_text(module_source, encoding="utf-8")
    return ProjectConfig(
        root=tmp_path,
        source_roots=(tmp_path / "src",),
        backend="mypyc",
        cache_dir=tmp_path / ".atoll" / "cache",
        report_dir=tmp_path / ".atoll",
        islands=(
            EnabledIslandConfig(
                source_module="app.ranking",
                source_path=source_path,
                sidecar_module="app._atoll_app_ranking",
                sidecar_path=tmp_path / ".atoll" / "sidecars" / "_atoll_app_ranking.py",
                symbols=symbols,
            ),
        ),
    )


def test_verify_reports_missing_status(tmp_path: Path) -> None:
    """A module without a managed shim fails verification."""
    config = _config(tmp_path, "def score_user() -> int:\n    return 1\n", ("score_user",))

    result = verify_islands(config)[0]

    assert result.error == "source module has no __atoll_status__"


def test_verify_reports_inactive_status(tmp_path: Path) -> None:
    """An inactive shim reports the status error."""
    config = _config(
        tmp_path,
        "__atoll_status__ = {'active': False, 'error': 'ImportError()'}\n",
        ("score_user",),
    )

    result = verify_islands(config)[0]

    assert result.error == "ImportError()"


def test_verify_reports_symbol_not_rebound(tmp_path: Path) -> None:
    """Active status is not enough when symbols still point at the source module."""
    config = _config(
        tmp_path,
        "def score_user() -> int:\n"
        "    return 1\n"
        "__atoll_status__ = {'active': True, 'compiled': False, 'origin': 'x.py'}\n",
        ("score_user",),
    )

    result = verify_islands(config)[0]

    assert result.error == "one or more symbols are not rebound to the sidecar module"


def test_verify_reports_bad_compiled_origin(tmp_path: Path) -> None:
    """Require-compiled mode checks extension suffixes when status claims compiled."""
    config = _config(
        tmp_path,
        "def score_user() -> int:\n"
        "    return 1\n"
        "score_user.__module__ = 'app._atoll_app_ranking'\n"
        "__atoll_status__ = {'active': True, 'compiled': True, 'origin': 'x.py'}\n",
        ("score_user",),
    )

    result = verify_islands(config, require_compiled=True)[0]

    assert result.error == "sidecar origin does not use a compiled extension suffix"


def test_verify_reports_import_exception(tmp_path: Path) -> None:
    """Import failures become verification errors rather than uncaught exceptions."""
    config = _config(tmp_path, "raise RuntimeError('boom')\n", ("score_user",))

    result = verify_islands(config)[0]

    assert "RuntimeError" in str(result.error)
