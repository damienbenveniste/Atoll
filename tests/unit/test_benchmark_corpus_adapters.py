"""Contract tests for the shared installed-wheel corpus oracle adapter."""

from __future__ import annotations

import importlib.util
import json
import tomllib
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

ADAPTER_PATH = Path("benchmarks/corpus/adapters/compatibility.py")
MANIFEST_PATH = Path("benchmarks/corpus/manifest.toml")
EXPECTED_CASE_COUNT = 25
ARGPARSE_ERROR_EXIT_CODE = 2


def _load_adapter() -> ModuleType:
    spec = importlib.util.spec_from_file_location("corpus_compatibility", ADAPTER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest_case_ids() -> set[str]:
    payload = tomllib.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    cases = cast(list[dict[str, object]], payload["case"])
    return {cast(str, case["id"]) for case in cases}


def test_probe_registry_exactly_covers_manifest_cases() -> None:
    adapter = _load_adapter()
    probes = cast(dict[str, object], adapter.PROBES)

    assert set(probes) == _manifest_case_ids()
    assert len(probes) == EXPECTED_CASE_COUNT


def test_main_rejects_unknown_case(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    adapter = _load_adapter()

    with pytest.raises(SystemExit) as raised:
        adapter.main(("--project-root", str(tmp_path), "--case", "unknown"))

    assert raised.value.code == ARGPARSE_ERROR_EXIT_CODE
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "invalid choice: 'unknown'" in captured.err


def test_main_prints_one_stable_json_object(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _load_adapter()
    package_file = tmp_path / "package.py"
    package_file.write_text("# installed package fixture\n", encoding="utf-8")
    package = ModuleType("package")
    package.__file__ = str(package_file)

    def probe() -> tuple[dict[str, object], tuple[ModuleType, ...]]:
        return {"z": [2, 1], "a": True}, (package,)

    monkeypatch.setitem(adapter.PROBES, "anyio", probe)

    assert adapter.main(("--project-root", str(tmp_path), "--case", "anyio")) == 0

    captured = capsys.readouterr()
    expected = {
        "canonical": {"a": True, "z": [2, 1]},
        "imports": [str(package_file.resolve())],
    }
    assert captured.err == ""
    assert captured.out == json.dumps(expected, sort_keys=True, separators=(",", ":")) + "\n"
    assert json.loads(captured.out) == expected


def test_module_path_returns_non_empty_absolute_existing_path(tmp_path: Path) -> None:
    adapter = _load_adapter()
    package_file = tmp_path / "package.py"
    package_file.write_text("# fixture\n", encoding="utf-8")
    package = ModuleType("package")
    package.__file__ = str(package_file)

    extracted = adapter._module_path(package)

    assert extracted
    assert Path(extracted).is_absolute()
    assert Path(extracted).is_file()
    assert adapter._module_paths((package,)) == [extracted]


@pytest.mark.parametrize("package_file", [None, ""])
def test_module_path_rejects_missing_package_file(package_file: str | None) -> None:
    adapter = _load_adapter()
    package = ModuleType("namespace_package")
    package.__file__ = package_file

    with pytest.raises(RuntimeError, match="has no package file"):
        adapter._module_path(package)
