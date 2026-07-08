"""Tests for AST scanning of fixture modules."""

from pathlib import Path

import pytest

from atoll.analysis.ast_scanner import scan_module
from atoll.models import ModuleId, SymbolRecord

FIXTURE_ROOT = Path("tests/fixtures/simple_project")
EXPECTED_OUTER_ARG_COUNT = 4


def _scan_ranking_module() -> dict[str, SymbolRecord]:
    module = ModuleId(
        name="app.ranking",
        path=(FIXTURE_ROOT / "src" / "app" / "ranking.py").resolve(),
    )
    scan = scan_module(module)
    return {symbol.id.qualname: symbol for symbol in scan.symbols}


def test_scan_extracts_top_level_functions() -> None:
    """The ranking fixture exposes four top-level function symbols."""
    symbols = _scan_ranking_module()

    assert tuple(symbols) == (
        "normalize_features",
        "score_user",
        "rank_candidates",
        "debug_dump",
    )


def test_scan_records_imports_and_constants() -> None:
    """Imports and simple constants are available for later sidecar generation."""
    module = ModuleId(
        name="app.ranking",
        path=(FIXTURE_ROOT / "src" / "app" / "ranking.py").resolve(),
    )
    scan = scan_module(module)

    assert [record.imported_names for record in scan.imports] == [
        ("annotations",),
        ("Event", "Score", "User"),
    ]
    assert [(record.name, record.kind) for record in scan.constants] == [
        ("DEFAULT_WEIGHT", "literal_constant")
    ]


def test_scan_records_symbol_references_and_globals() -> None:
    """Referenced names distinguish helper calls and global constants."""
    symbols = _scan_ranking_module()
    score_user = symbols["score_user"]

    assert "normalize_features" in score_user.referenced_names
    assert "DEFAULT_WEIGHT" in score_user.uses_globals
    assert "features" in score_user.local_names


def test_scan_records_import_aliases_constants_classes_and_methods(tmp_path: Path) -> None:
    """Scanner records V1 facts across imports, constants, classes, and methods."""
    module_path = tmp_path / "sample.py"
    module_path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import math as m",
                "import xml.etree.ElementTree",
                "from .local import value as local_value",
                "from .everything import *",
                "",
                "NEGATIVE = -1",
                "PAIR = (1, 'a')",
                "DYNAMIC = make_value()",
                "UNKNOWN = BASE + 1",
                "TYPED: int = 3",
                "",
                "if True:",
                "    pass",
                "",
                "def outer(x: int, *items: str, flag: bool = True, **kwargs: str) -> int:",
                "    def nested() -> int:",
                "        return x",
                "    class Inner:",
                "        pass",
                "    return eval('1')",
                "",
                "class Dynamic(metaclass=Meta):",
                "    def __getattr__(self, name: str) -> object:",
                "        return name",
                "",
                "class Worker(Base):",
                "    @classmethod",
                "    def make(cls, value: int) -> Worker:",
                "        return cls()",
                "",
                "    @staticmethod",
                "    def helper(value: int) -> int:",
                "        return value",
                "",
            ]
        ),
        encoding="utf-8",
    )

    scan = scan_module(ModuleId(name="sample", path=module_path))
    symbols = {symbol.id.qualname: symbol for symbol in scan.symbols}

    assert [record.imported_names for record in scan.imports] == [
        ("annotations",),
        ("m",),
        ("xml",),
        ("local_value",),
        (),
    ]
    assert [(record.name, record.kind) for record in scan.constants] == [
        ("NEGATIVE", "literal_constant"),
        ("PAIR", "literal_constant"),
        ("DYNAMIC", "runtime_dynamic"),
        ("UNKNOWN", "unknown"),
        ("TYPED", "literal_constant"),
    ]
    assert scan.top_level_statement_lines == (13,)
    assert symbols["outer"].arg_count == EXPECTED_OUTER_ARG_COUNT
    assert symbols["outer"].annotated_arg_count == EXPECTED_OUTER_ARG_COUNT
    assert {blocker.code for blocker in symbols["outer"].blockers} == {
        "DYN_EVAL",
        "NESTED_SYMBOL",
    }
    assert {blocker.code for blocker in symbols["Dynamic"].blockers} == {"DYN_CLASS_MONKEYPATCH"}
    assert symbols["Worker"].uses_globals == ("Base",)
    assert symbols["Worker.make"].decorators == ("classmethod",)
    assert symbols["Worker.helper"].decorators == ("staticmethod",)


def test_scan_raises_for_syntax_errors(tmp_path: Path) -> None:
    """Syntax errors are surfaced to the caller during scanning."""
    module_path = tmp_path / "broken.py"
    module_path.write_text("def broken(:\n", encoding="utf-8")

    with pytest.raises(SyntaxError):
        scan_module(ModuleId(name="broken", path=module_path))
