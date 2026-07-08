"""Tests for mypy diagnostic parsing."""

from pathlib import Path

from atoll.backends.mypy import parse_mypy_output


def test_parse_mypy_output_extracts_diagnostics(tmp_path: Path) -> None:
    """Mypy output is parsed into stable diagnostic records."""
    output = "\n".join(
        [
            "src/app/ranking.py:10:5: error: Incompatible return value type  [return-value]",
            "src/app/ranking.py:10:5: note: expected str",
            "Found 1 error in 1 file (checked 1 source file)",
        ]
    )

    diagnostics = parse_mypy_output(output, cwd=tmp_path)

    assert [(item.line, item.column, item.severity, item.code) for item in diagnostics] == [
        (10, 5, "error", "return-value"),
        (10, 5, "note", None),
    ]
    assert diagnostics[0].path == tmp_path / "src" / "app" / "ranking.py"
