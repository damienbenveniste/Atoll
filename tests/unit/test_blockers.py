"""Tests for dynamic blocker detection."""

from pathlib import Path

from atoll.analysis.ast_scanner import scan_module
from atoll.models import ModuleId

FIXTURE_ROOT = Path("tests/fixtures/simple_project")


def test_dynamic_getattr_marks_only_residue_function() -> None:
    """A dynamic debug helper does not poison nearby clean functions."""
    scan = scan_module(
        ModuleId(
            name="app.ranking",
            path=(FIXTURE_ROOT / "src" / "app" / "ranking.py").resolve(),
        )
    )
    blockers_by_symbol = {
        symbol.id.qualname: {blocker.code for blocker in symbol.blockers} for symbol in scan.symbols
    }

    assert blockers_by_symbol["debug_dump"] == {"DYN_GETATTR_DYNAMIC"}
    assert blockers_by_symbol["normalize_features"] == set()
    assert blockers_by_symbol["score_user"] == set()
    assert blockers_by_symbol["rank_candidates"] == set()


def test_untyped_function_gets_soft_blocker(tmp_path: Path) -> None:
    """Untyped signatures are tracked as local soft blockers."""
    module_path = tmp_path / "sample.py"
    module_path.write_text("def loose(value):\n    return value\n", encoding="utf-8")

    scan = scan_module(ModuleId(name="sample", path=module_path))

    assert {blocker.code for blocker in scan.symbols[0].blockers} == {"UNTYPED_DEF"}


def test_dynamic_call_blockers_are_attached_to_symbol(tmp_path: Path) -> None:
    """Hard dynamic calls and soft literal getattr calls are symbol-local."""
    module_path = tmp_path / "dynamic_calls.py"
    module_path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import importlib",
                "",
                "def dynamic(name: str, obj: object) -> object:",
                "    getattr(obj, 'field')",
                "    setattr(obj, name, 1)",
                "    delattr(obj, name)",
                "    importlib.import_module(name)",
                "    __import__(name)",
                "    return obj",
                "",
                "async def async_worker(value: int) -> int:",
                "    return value",
                "",
            ]
        ),
        encoding="utf-8",
    )

    scan = scan_module(ModuleId(name="dynamic_calls", path=module_path))
    blockers_by_symbol = {
        symbol.id.qualname: {blocker.code for blocker in symbol.blockers} for symbol in scan.symbols
    }

    assert blockers_by_symbol["dynamic"] == {
        "DYN_GETATTR_LITERAL",
        "DYN_SETATTR",
        "DYN_DELATTR",
        "DYN_IMPORTLIB",
        "DYN_IMPORT_CALL",
    }
    assert blockers_by_symbol["async_worker"] == {"ASYNC_FUNCTION"}


def test_module_level_monkey_patch_is_recorded(tmp_path: Path) -> None:
    """Top-level attribute assignment is tracked as module-level risk."""
    module_path = tmp_path / "monkey.py"
    module_path.write_text("service.handler = replacement\n", encoding="utf-8")

    scan = scan_module(ModuleId(name="monkey", path=module_path))

    assert [blocker.code for blocker in scan.blockers] == ["DYN_MODULE_MONKEYPATCH"]
