"""Contracts for deterministic corpus performance adapters and provenance."""

from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest
from scripts.benchmark_corpus.lifecycle import performance_asset_digest
from scripts.benchmark_corpus.manifest import load_manifest

ADAPTER_ROOT = Path("benchmarks/corpus/adapters")
WORKLOAD_ROOT = Path("benchmarks/corpus/workloads")
NOTICE_ROOT = Path("benchmarks/corpus/notices")
MANIFEST_PATH = Path("benchmarks/corpus/manifest.toml")
GOLDEN_PATH = WORKLOAD_ROOT / "golden.json"
DEFAULT_REPETITIONS = 1
DEFAULT_SEED = 1729
PERFORMANCE_PROVENANCE_REVISION = "e9a5c20ab85369d7cc69772975a34fafc251b239"
PERFORMANCE_CASES = {
    "anyio",
    "html5lib",
    "mako",
    "mypy",
    "networkx",
    "pydantic",
    "pydantic-graph",
    "rich",
    "sqlalchemy",
    "sqlglot",
    "sympy",
    "tomli",
}


def _load_runner() -> ModuleType:
    """Load the shared adapter command boundary without importing workloads."""
    path = ADAPTER_ROOT / "_performance.py"
    spec = importlib.util.spec_from_file_location("corpus_performance", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _manifest_cases() -> dict[str, dict[str, object]]:
    """Return raw case tables keyed by stable manifest identifier."""
    payload = tomllib.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    cases = cast(list[dict[str, object]], payload["case"])
    return {cast(str, case["id"]): case for case in cases}


def _golden_cases() -> dict[str, dict[str, object]]:
    """Return reviewed default results keyed by performance case."""
    payload = cast(dict[str, object], json.loads(GOLDEN_PATH.read_text(encoding="utf-8")))
    assert payload["repetitions"] == DEFAULT_REPETITIONS
    assert payload["seed"] == DEFAULT_SEED
    return cast(dict[str, dict[str, object]], payload["cases"])


def test_performance_adapter_case_set_is_exact() -> None:
    """Only the twelve reviewed milestone workloads receive dedicated adapters."""
    adapters = {
        path.stem.replace("_", "-")
        for path in ADAPTER_ROOT.glob("*.py")
        if path.name not in {"_performance.py", "compatibility.py"}
    }
    workloads = {path.stem.replace("_", "-") for path in WORKLOAD_ROOT.glob("*.py")}
    notices = {path.stem for path in NOTICE_ROOT.glob("*.txt")}

    assert adapters == PERFORMANCE_CASES
    assert workloads == PERFORMANCE_CASES
    assert notices == PERFORMANCE_CASES | {"pyperformance"}
    assert set(_golden_cases()) == PERFORMANCE_CASES


def test_default_run_accepts_exact_golden_result(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default benchmark emits its reviewed result when every field matches."""
    runner = _load_runner()
    golden = _golden_cases()["pydantic"]
    package_file = tmp_path / "installed_package.py"
    package_file.write_text("# installed payload fixture\n", encoding="utf-8")
    package = ModuleType("installed_package")
    package.__file__ = str(package_file)

    class _GoldenWorkload:
        def run(
            self,
            *,
            repetitions: int,
            seed: int,
        ) -> tuple[dict[str, object], tuple[ModuleType, ...]]:
            assert repetitions == DEFAULT_REPETITIONS
            assert seed == DEFAULT_SEED
            return golden, (package,)

    def load_workload(_case_id: str) -> _GoldenWorkload:
        return _GoldenWorkload()

    monkeypatch.setattr(runner, "_load_workload", load_workload)

    assert runner.main("pydantic", ("--project-root", str(tmp_path))) == 0

    expected = {
        "canonical": {
            "case": "pydantic",
            "repetitions": DEFAULT_REPETITIONS,
            "result": golden,
            "seed": DEFAULT_SEED,
        },
        "imports": [str(package_file.resolve())],
    }
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == json.dumps(expected, sort_keys=True, separators=(",", ":")) + "\n"


def test_default_run_rejects_mismatching_golden_result(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even JSON type drift that compares equal in Python fails before output."""
    runner = _load_runner()
    wrong = dict(_golden_cases()["pydantic"])
    wrong["validated"] = 20_000.0
    assert wrong == _golden_cases()["pydantic"]

    class _WrongWorkload:
        def run(
            self,
            *,
            repetitions: int,
            seed: int,
        ) -> tuple[dict[str, object], tuple[ModuleType, ...]]:
            assert repetitions == DEFAULT_REPETITIONS
            assert seed == DEFAULT_SEED
            return wrong, ()

    def load_workload(_case_id: str) -> _WrongWorkload:
        return _WrongWorkload()

    monkeypatch.setattr(runner, "_load_workload", load_workload)

    with pytest.raises(RuntimeError, match="default canonical mismatch for pydantic"):
        runner.main("pydantic", ("--project-root", str(tmp_path)))

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_runner_prints_one_exact_seeded_canonical_object(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seed and repetitions are explicit and serialized with stable key ordering."""
    runner = _load_runner()
    package_file = tmp_path / "installed_package.py"
    package_file.write_text("# installed payload fixture\n", encoding="utf-8")
    package = ModuleType("installed_package")
    package.__file__ = str(package_file)

    class _SeededWorkload:
        def run(
            self,
            *,
            repetitions: int,
            seed: int,
        ) -> tuple[dict[str, object], tuple[ModuleType, ...]]:
            return {
                "checksum": seed * repetitions,
                "samples": [seed + index for index in range(repetitions)],
            }, (package,)

    def load_workload(_case_id: str) -> _SeededWorkload:
        return _SeededWorkload()

    monkeypatch.setattr(runner, "_load_workload", load_workload)

    assert (
        runner.main(
            "pydantic",
            (
                "--project-root",
                str(tmp_path),
                "--repetitions",
                "3",
                "--seed",
                "41",
            ),
        )
        == 0
    )

    expected = {
        "canonical": {
            "case": "pydantic",
            "repetitions": 3,
            "result": {"checksum": 123, "samples": [41, 42, 43]},
            "seed": 41,
        },
        "imports": [str(package_file.resolve())],
    }
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == json.dumps(expected, sort_keys=True, separators=(",", ":")) + "\n"
    assert captured.out.count("\n") == 1


def test_manifest_performance_provenance_matches_reviewed_bytes() -> None:
    """Every performance tier pins its complete workload and harness bundle."""
    cases = _manifest_cases()
    manifest = load_manifest(MANIFEST_PATH)
    typed_cases = {case.id: case for case in manifest.cases}
    selected = {
        case_id
        for case_id, case in cases.items()
        if "performance" in cast(list[str], case["tiers"])
    }
    assert selected == PERFORMANCE_CASES

    for case_id in sorted(PERFORMANCE_CASES):
        case = cases[case_id]
        workload = cast(dict[str, object], case["workload"])
        workload_path = Path(cast(str, workload["path"]))
        notice_path = Path(cast(str, workload["notice"]))

        assert workload["source"] == "atoll"
        assert workload["revision"] == PERFORMANCE_PROVENANCE_REVISION
        assert workload_path == WORKLOAD_ROOT / f"{case_id.replace('-', '_')}.py"
        assert notice_path == NOTICE_ROOT / f"{case_id}.txt"
        assert notice_path.is_file()
        assert (
            performance_asset_digest(Path.cwd(), typed_cases[case_id], ADAPTER_ROOT)
            == workload["sha256"]
        )
        assert case["oracle_adapter"] == "compatibility"
        assert case["oracle_arguments"] == ["--case", case_id]
