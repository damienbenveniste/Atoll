"""Tests for disposable policy evidence and canonical case reports."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path, PurePosixPath

import pytest
from scripts.benchmark_corpus.models import (
    CaseResult,
    CaseStatus,
    CompilePolicy,
    CorpusTier,
    PhaseEvidence,
    RatioEvidence,
)
from scripts.benchmark_corpus.policy import append_compile_policy, render_compile_policy
from scripts.benchmark_corpus.results import (
    classify_compile_report,
    render_case_markdown,
    write_case_result,
)

_EXPECTED_FINAL_SPEEDUP = 1.25


def test_render_compile_policy_preserves_argv_and_backend_order() -> None:
    policy = CompilePolicy(
        backends=("cython", "mypyc"),
        test_command=("python", "oracle.py", "--value=a b"),
        benchmark_command=("python", "workload.py"),
        benchmark_warmups=2,
        benchmark_samples=9,
        minimum_speedup=1.25,
    )

    rendered = render_compile_policy(policy)
    parsed = tomllib.loads(rendered)["tool"]["atoll"]["compile"]

    assert parsed == {
        "backends": ["cython", "mypyc"],
        "test_command": ["python", "oracle.py", "--value=a b"],
        "benchmark_command": ["python", "workload.py"],
        "benchmark_warmups": 2,
        "benchmark_samples": 9,
        "minimum_speedup": 1.25,
    }


def test_render_compile_policy_rejects_inconsistent_benchmark() -> None:
    with pytest.raises(ValueError, match="semantic test"):
        render_compile_policy(
            CompilePolicy(
                backends=("mypyc", "cython"),
                benchmark_command=("python", "workload.py"),
            )
        )


def test_append_compile_policy_records_exact_patch_without_touching_other_files(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    pyproject = checkout / "pyproject.toml"
    pyproject.write_text('[project]\nname = "fixture"\nversion = "1"\n', encoding="utf-8")
    source = checkout / "fixture.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")

    evidence = append_compile_policy(
        pyproject,
        CompilePolicy(backends=("mypyc", "cython")),
        tmp_path / "evidence",
        checkout,
    )

    patch = tmp_path / "evidence" / evidence.patch_path
    assert evidence.source_path.as_posix() == "pyproject.toml"
    assert "[tool.atoll.compile]" in pyproject.read_text(encoding="utf-8")
    assert "+++ b/pyproject.toml" in patch.read_text(encoding="utf-8")
    assert source.read_text(encoding="utf-8") == "VALUE = 1\n"


def test_append_compile_policy_rejects_upstream_compile_table(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    pyproject = checkout / "pyproject.toml"
    pyproject.write_text("[tool.atoll.compile]\nbackends = ['mypyc']\n", encoding="utf-8")

    with pytest.raises(ValueError, match="already defines"):
        append_compile_policy(
            pyproject,
            CompilePolicy(backends=("mypyc", "cython")),
            tmp_path / "evidence",
            checkout,
        )


def test_case_result_json_and_markdown_are_derived_from_same_evidence(tmp_path: Path) -> None:
    result = _result()

    json_path, markdown_path = write_case_result(result, tmp_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert payload["status"] == "compiled-unbenchmarked"
    assert payload["ratios"]["final_wheel_vs_original"] == _EXPECTED_FINAL_SPEEDUP
    assert "Final wheel versus original: 1.250x" in markdown
    assert "Native layer versus source-only wheel: not measured" in markdown
    assert markdown == render_case_markdown(result)


@pytest.mark.parametrize(
    ("tier", "performance_status", "composition", "expected"),
    [
        ("performance", "accepted", {}, "accelerated"),
        ("performance", "not-profitable", {}, "not-profitable"),
        ("performance", "unstable", {}, "unstable"),
        ("compatibility", None, {}, "supported-no-op"),
        (
            "compatibility",
            None,
            {"native_variant_ids": ["variant-1"]},
            "compiled-unbenchmarked",
        ),
    ],
)
def test_classify_compile_report_preserves_every_success_outcome(
    tier: CorpusTier,
    performance_status: str | None,
    composition: dict[str, object],
    expected: CaseStatus,
) -> None:
    """Compatible no-op and rejected acceleration remain visible outcomes."""
    report: dict[str, object] = {"final_composition": composition}
    if performance_status is not None:
        report["performance"] = {"status": performance_status}

    assert classify_compile_report(report, tier) == expected


def _result() -> CaseResult:
    return CaseResult(
        schema_version=1,
        case_id="fixture",
        tier="compatibility",
        platform="ubuntu-24.04",
        status="compiled-unbenchmarked",
        repository="https://example.invalid/fixture.git",
        revision="a" * 40,
        manifest_digest="b" * 64,
        case_digest="c" * 64,
        comparison_key=None,
        diagnostics=(),
        source_digest_before="d" * 64,
        source_digest_after="d" * 64,
        source_unchanged=True,
        policy=None,
        environment=None,
        phases=(
            PhaseEvidence(
                name="compile-cold",
                argv=("python", "-m", "atoll", "compile"),
                exit_code=0,
                timed_out=False,
                duration_seconds=1.2345,
                log_path=PurePosixPath("cold.log"),
                log_truncated=False,
            ),
        ),
        baseline_wheel_digest="e" * 64,
        compiled_wheel_digest="f" * 64,
        cold_report_path=PurePosixPath("cold.compile-report.json"),
        warm_report_path=PurePosixPath("warm.compile-report.json"),
        baseline_oracle_digest="1" * 64,
        compiled_oracle_digest="1" * 64,
        cold_compiler_invocations=1,
        warm_compiler_invocations=0,
        ratios=RatioEvidence(final_wheel_vs_original=_EXPECTED_FINAL_SPEEDUP),
    )
