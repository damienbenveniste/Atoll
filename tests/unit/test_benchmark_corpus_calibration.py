"""Tests for the compiler-calibration catalog and aggregate boundary."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path, PurePosixPath

import pytest
from scripts.benchmark_corpus.calibration import (
    CalibrationError,
    calibration_execution_digest,
    calibration_runner_argv,
    load_calibration_catalog,
    verify_external_calibration,
)

_SHA = "a" * 40
_SHA256 = "b" * 64
_LOCAL_CALIBRATION_COUNT = 3
_EXPECTED_CALIBRATIONS = {
    "hexiom",
    "native-buffer",
    "native-call-chain",
    "native-scalar",
    "richards",
    "spectral-norm",
}


def test_repository_catalog_keeps_calibrations_out_of_repository_aggregate() -> None:
    catalog = load_calibration_catalog(
        Path("benchmarks/corpus/calibration.toml"),
        Path(),
    )

    assert catalog.schema_version == 1
    assert {benchmark.id for benchmark in catalog.benchmarks} == _EXPECTED_CALIBRATIONS
    assert all(not benchmark.included_in_repository_aggregate for benchmark in catalog.benchmarks)
    assert {benchmark.source for benchmark in catalog.benchmarks} == {
        "atoll-fixture",
        "pyperformance",
    }
    assert (
        sum(benchmark.repository_verified for benchmark in catalog.benchmarks)
        == _LOCAL_CALIBRATION_COUNT
    )
    assert sum(benchmark.runnable for benchmark in catalog.benchmarks) == _LOCAL_CALIBRATION_COUNT
    assert all(
        benchmark.runner is None
        for benchmark in catalog.benchmarks
        if benchmark.source == "pyperformance"
    )


def test_local_calibration_runner_materializes_required_destinations(tmp_path: Path) -> None:
    catalog = load_calibration_catalog(
        Path("benchmarks/corpus/calibration.toml"),
        Path(),
    )
    benchmark = next(item for item in catalog.benchmarks if item.id == "native-scalar")

    argv = calibration_runner_argv(
        benchmark,
        workspace=tmp_path / "workspace",
        evidence_root=tmp_path / "evidence",
    )

    assert argv[argv.index("--workspace") + 1] == str(tmp_path / "workspace")
    assert argv[argv.index("--evidence-root") + 1] == str(tmp_path / "evidence")


def test_calibration_catalog_rejects_aggregate_inclusion(tmp_path: Path) -> None:
    path = _catalog(tmp_path, included="true")

    with pytest.raises(CalibrationError, match="explicitly false"):
        load_calibration_catalog(path, tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("schema_version = 2", "schema_version"),
        (f'revision = "{_SHA[:-1]}"', "full lowercase Git SHA"),
        (f'source_sha256 = "{_SHA256[:-1]}"', "SHA-256"),
        ('source_path = "../escape.py"', "safe relative path"),
        ('repository = "http://example.invalid/source.git"', "HTTPS"),
        ('unexpected = "field"', "unknown field"),
    ],
)
def test_calibration_catalog_rejects_malformed_metadata(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    path = _catalog(tmp_path)
    content = path.read_text(encoding="utf-8")
    if mutation.startswith("schema_version"):
        content = content.replace("schema_version = 1", mutation)
    elif mutation.startswith("unexpected"):
        content = f"{content}\n{mutation}\n"
    else:
        key = mutation.split(" =", maxsplit=1)[0]
        content = "\n".join(
            mutation if line.startswith(f"{key} =") else line for line in content.splitlines()
        )
    path.write_text(content, encoding="utf-8")

    with pytest.raises(CalibrationError, match=message):
        load_calibration_catalog(path, tmp_path)


def test_atoll_fixture_digest_is_verified_against_local_source(tmp_path: Path) -> None:
    _source, execution_digest = _local_bundle(tmp_path)
    notice = tmp_path / "NOTICE"
    notice.write_text("notice\n", encoding="utf-8")
    path = tmp_path / "calibration.toml"
    path.write_text(
        _document(
            source="atoll-fixture",
            source_sha256=_SHA256,
            execution_sha256=execution_digest,
            notice="NOTICE",
        ),
        encoding="utf-8",
    )

    with pytest.raises(CalibrationError, match="source digest mismatch"):
        load_calibration_catalog(path, tmp_path)


def test_local_execution_bundle_rejects_unreviewed_executable_changes(tmp_path: Path) -> None:
    source, execution_digest = _local_bundle(tmp_path)
    notice = tmp_path / "NOTICE"
    notice.write_text("notice\n", encoding="utf-8")
    path = tmp_path / "calibration.toml"
    path.write_text(
        _document(
            source="atoll-fixture",
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            execution_sha256=execution_digest,
            notice="NOTICE",
        ),
        encoding="utf-8",
    )
    load_calibration_catalog(path, tmp_path)

    kernels = tmp_path / "tests/fixtures/native_optimization_project/src/package/kernels.py"
    kernels.write_text("def changed():\n    return 2\n", encoding="utf-8")
    with pytest.raises(CalibrationError, match="execution bundle digest mismatch"):
        load_calibration_catalog(path, tmp_path)


def test_external_calibration_requires_exact_clean_checkout(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    source = checkout / "fixture.py"
    source.write_text("value = 1\n", encoding="utf-8")
    _git(checkout, "init")
    _git(checkout, "config", "user.email", "corpus@example.invalid")
    _git(checkout, "config", "user.name", "Corpus Test")
    _git(checkout, "add", "fixture.py")
    _git(checkout, "commit", "-m", "fixture")
    revision = _git(checkout, "rev-parse", "HEAD")
    _git(checkout, "checkout", "--detach", revision)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    notice = tmp_path / "NOTICE"
    notice.write_text("notice\n", encoding="utf-8")
    catalog_path = tmp_path / "calibration.toml"
    catalog_path.write_text(
        _document(
            notice="NOTICE",
            revision=revision,
            source_sha256=digest,
        ),
        encoding="utf-8",
    )
    benchmark = load_calibration_catalog(catalog_path, tmp_path).benchmarks[0]

    assert verify_external_calibration(benchmark, checkout) == source

    source.write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(CalibrationError, match="checkout is modified"):
        verify_external_calibration(benchmark, checkout)


def test_calibration_execution_contract_rejects_ambient_or_incomplete_runners(
    tmp_path: Path,
) -> None:
    notice = tmp_path / "NOTICE"
    notice.write_text("notice\n", encoding="utf-8")
    external = tmp_path / "external.toml"
    external.write_text(
        _document(notice="NOTICE").replace(
            'platforms = ["ubuntu-24.04"]',
            'runner = ["python", "-m", "pyperformance"]\nplatforms = ["ubuntu-24.04"]',
        ),
        encoding="utf-8",
    )
    with pytest.raises(CalibrationError, match="verification only"):
        load_calibration_catalog(external, tmp_path)

    source, execution_digest = _local_bundle(tmp_path)
    local = tmp_path / "local.toml"
    local.write_text(
        _document(
            source="atoll-fixture",
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            execution_sha256=execution_digest,
            notice="NOTICE",
        ).replace(
            'runner = ["uv", "run", "--python", "3.12", "python", '
            '"scripts/run_native_optimizer_benchmark.py", "--workspace", '
            '"{workspace}", "--evidence-root", "{evidence_root}"]',
            'runner = ["python", "scripts/run_native_optimizer_benchmark.py"]',
        ),
        encoding="utf-8",
    )
    with pytest.raises(CalibrationError, match="native hard workflow command"):
        load_calibration_catalog(local, tmp_path)


def _catalog(tmp_path: Path, *, included: str = "false") -> Path:
    notice = tmp_path / "NOTICE"
    notice.write_text("notice\n", encoding="utf-8")
    path = tmp_path / "calibration.toml"
    content = _document(notice="NOTICE").replace(
        "included_in_repository_aggregate = false",
        f"included_in_repository_aggregate = {included}",
    )
    path.write_text(content, encoding="utf-8")
    return path


def _document(
    *,
    source: str = "pyperformance",
    source_sha256: str = _SHA256,
    notice: str,
    revision: str = _SHA,
    execution_sha256: str | None = None,
) -> str:
    execution = (
        "atoll-native-hard-suite" if source == "atoll-fixture" else "external-checkout-required"
    )
    runner_line: str | None = None
    execution_lines: tuple[str, ...] = ()
    source_path = "fixture.py"
    if source == "atoll-fixture":
        runner_line = (
            'runner = ["uv", "run", "--python", "3.12", "python", '
            '"scripts/run_native_optimizer_benchmark.py", '
            '"--workspace", "{workspace}", "--evidence-root", "{evidence_root}"]'
        )
        source_path = "tests/fixtures/native_optimization_project/benchmarks/run_scalar_hard.py"
        execution_lines = (
            'execution_paths = ["scripts/run_native_optimizer_benchmark.py", '
            '"tests/fixtures/native_optimization_project"]',
            f'execution_sha256 = "{execution_sha256 or _SHA256}"',
        )
    optional_runner = (runner_line,) if runner_line is not None else ()
    return "\n".join(
        (
            "schema_version = 1",
            "",
            "[[benchmark]]",
            'id = "alpha"',
            'name = "Alpha"',
            f'source = "{source}"',
            f'execution = "{execution}"',
            *execution_lines,
            'repository = "https://example.invalid/source.git"',
            f'revision = "{revision}"',
            f'source_path = "{source_path}"',
            f'source_sha256 = "{source_sha256}"',
            *optional_runner,
            'platforms = ["ubuntu-24.04"]',
            f'notice = "{notice}"',
            "included_in_repository_aggregate = false",
            "",
        )
    )


def _local_bundle(tmp_path: Path) -> tuple[Path, str]:
    runner = tmp_path / "scripts/run_native_optimizer_benchmark.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("def main():\n    return 0\n", encoding="utf-8")
    fixture = tmp_path / "tests/fixtures/native_optimization_project"
    source = fixture / "benchmarks/run_scalar_hard.py"
    source.parent.mkdir(parents=True)
    source.write_text("def workload():\n    return 1\n", encoding="utf-8")
    kernels = fixture / "src/package/kernels.py"
    kernels.parent.mkdir(parents=True)
    kernels.write_text("def kernel():\n    return 1\n", encoding="utf-8")
    (fixture / "pyproject.toml").write_text("[project]\nname = 'fixture'\n", encoding="utf-8")
    paths = (
        PurePosixPath("scripts/run_native_optimizer_benchmark.py"),
        PurePosixPath("tests/fixtures/native_optimization_project"),
    )
    return source, calibration_execution_digest(tmp_path, paths)


def _git(root: Path, *arguments: str) -> str:
    git = shutil.which("git")
    assert git is not None
    completed = subprocess.run(
        (git, "-C", str(root), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()
