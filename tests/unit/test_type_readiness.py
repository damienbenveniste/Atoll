"""Tests for mapping type-checker diagnostics to symbols."""

from pathlib import Path

from atoll.analysis.ast_scanner import scan_module
from atoll.analysis.type_readiness import attach_mypy_diagnostics
from atoll.models import ModuleId, MypyDiagnostic


def test_attach_mypy_diagnostics_maps_errors_to_symbol(tmp_path: Path) -> None:
    """A diagnostic inside a symbol becomes a symbol-local MYPY_ERROR blocker."""
    module_path = tmp_path / "sample.py"
    module_path.write_text(
        "def good(x: int) -> int:\n    return x + 1\n\ndef bad(x: int) -> str:\n    return x + 1\n",
        encoding="utf-8",
    )
    module = scan_module(ModuleId(name="sample", path=module_path))
    diagnostic = MypyDiagnostic(
        path=module_path,
        line=5,
        column=12,
        severity="error",
        code="return-value",
        message="Incompatible return value type",
    )

    enriched = attach_mypy_diagnostics((module,), (diagnostic,))[0]
    blockers_by_symbol = {
        symbol.id.qualname: {blocker.code for blocker in symbol.blockers}
        for symbol in enriched.symbols
    }

    assert blockers_by_symbol["good"] == set()
    assert blockers_by_symbol["bad"] == {"MYPY_ERROR"}
    assert enriched.symbols[1].mypy_diagnostics[0].symbol is not None


def test_attach_mypy_diagnostics_keeps_module_level_diagnostics(tmp_path: Path) -> None:
    """Diagnostics outside symbol ranges remain module-level diagnostics."""
    module_path = tmp_path / "sample.py"
    module_path.write_text("VALUE = missing\n", encoding="utf-8")
    module = scan_module(ModuleId(name="sample", path=module_path))
    diagnostic = MypyDiagnostic(
        path=module_path,
        line=1,
        column=9,
        severity="error",
        code="name-defined",
        message="Name 'missing' is not defined",
    )

    enriched = attach_mypy_diagnostics((module,), (diagnostic,))[0]

    assert enriched.mypy_diagnostics == (diagnostic,)
