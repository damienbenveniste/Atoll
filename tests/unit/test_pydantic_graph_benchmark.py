"""Deterministic checks for the manual Pydantic Graph hard-benchmark harness."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from scripts.run_pydantic_graph_benchmark import (
    BENCHMARK_SAMPLES,
    COLD_MYPYC_BASELINE_SECONDS,
    COLD_MYPYC_TARGET_SECONDS,
    MINIMUM_FINAL_SPEEDUP,
    MINIMUM_MARGINAL_SPEEDUP,
    BenchmarkEvidenceInputs,
    append_compile_policy,
    evaluate_reports,
    source_manifest,
)

TEMPLATE_ROOT = Path("benchmarks/pydantic_graph")
COLD_TEST_SECONDS = 90.0
FINAL_TEST_SPEEDUP = 1.12
PROBE_TEST_COUNT = 2
PLAN_ID = "exec-plan:fixture"


def test_evaluate_reports_accepts_profiled_cold_and_cache_only_warm_builds() -> None:
    cold = _report(cache_status="miss", mypyc_seconds=COLD_TEST_SECONDS)
    warm = _report(cache_status="hit", mypyc_seconds=0.0)

    evaluation = evaluate_reports(
        BenchmarkEvidenceInputs(
            cold_report=cold,
            warm_report=warm,
            sources_unchanged=True,
            cold_exit_code=0,
            warm_exit_code=0,
            wheel_present=True,
            cold_compiler_probe_count=PROBE_TEST_COUNT,
            warm_compiler_probe_count=0,
        )
    )

    assert evaluation.succeeded is True
    assert evaluation.errors == ()
    assert evaluation.cold_mypyc_seconds == COLD_TEST_SECONDS
    assert evaluation.cold_native_phase_count == 1
    assert evaluation.cold_compiler_probe_count == PROBE_TEST_COUNT
    assert evaluation.warm_native_phase_count == 0
    assert evaluation.warm_compiler_probe_count == 0
    assert evaluation.final_speedup == FINAL_TEST_SPEEDUP
    assert evaluation.accepted_candidates == 1
    assert evaluation.execution_plan_count == 1
    assert evaluation.applied_execution_plan_count == 1
    assert evaluation.execution_plan_trial_count == 1
    assert evaluation.accepted_execution_plan_trials == 1
    assert evaluation.fusion_plan_count == 1
    assert evaluation.eligible_fusion_plan_count == 0
    assert evaluation.fusion_trial_count == 0


def test_evaluate_reports_accepts_plan_only_success_without_native_invocations() -> None:
    cold = _report(
        cache_status="miss",
        mypyc_seconds=0.0,
        native_evidence=False,
        options=_ReportOptions(execution_plan_cache_status="miss"),
    )
    warm = _report(
        cache_status="hit",
        mypyc_seconds=0.0,
        native_evidence=False,
        options=_ReportOptions(execution_plan_cache_status="hit"),
    )

    evaluation = evaluate_reports(
        BenchmarkEvidenceInputs(
            cold_report=cold,
            warm_report=warm,
            sources_unchanged=True,
            cold_exit_code=0,
            warm_exit_code=0,
            wheel_present=True,
            cold_compiler_probe_count=0,
            warm_compiler_probe_count=0,
        )
    )

    assert evaluation.succeeded is True
    assert evaluation.errors == ()
    assert evaluation.accepted_candidates == 0
    assert evaluation.cold_native_phase_count == 0
    assert evaluation.cold_compiler_probe_count == 0
    assert evaluation.accepted_execution_plan_trials == 1


def test_evaluate_reports_explains_every_hard_gate_failure() -> None:
    cold = _report(cache_status="partial", mypyc_seconds=COLD_MYPYC_TARGET_SECONDS + 1)
    warm = _report(
        cache_status="partial",
        mypyc_seconds=1.0,
        final_speedup=MINIMUM_FINAL_SPEEDUP - 0.01,
        options=_ReportOptions(
            marginal_speedup=MINIMUM_MARGINAL_SPEEDUP - 0.01,
            region_cache_status="miss",
            execution_plan_cache_status="miss",
            execution_plan_marginal_speedup=MINIMUM_MARGINAL_SPEEDUP - 0.01,
            execution_plan_overall_speedup=MINIMUM_FINAL_SPEEDUP - 0.01,
        ),
    )
    warm["version"] = 2
    cast(dict[str, object], warm["profile"])["status"] = "static-fallback"
    cast(dict[str, object], cast(list[object], warm["execution_plans"])[0])["source_hash"] = (
        "changed-plan-source-hash"
    )
    warm["applied_execution_plans"] = [PLAN_ID, "exec-plan:unselected"]

    evaluation = evaluate_reports(
        BenchmarkEvidenceInputs(
            cold_report=cold,
            warm_report=warm,
            sources_unchanged=False,
            cold_exit_code=1,
            warm_exit_code=1,
            wheel_present=False,
            cold_compiler_probe_count=0,
            warm_compiler_probe_count=PROBE_TEST_COUNT,
        )
    )

    assert evaluation.succeeded is False
    message = "\n".join(evaluation.errors)
    assert "cold compile did not succeed" in message
    assert "warm report is not schema version 5" in message
    assert "warm profile status is static-fallback" in message
    assert "cold cache status is partial" in message
    assert "cold mypyc took" in message
    assert "compiler probe observed no cold native invocation" in message
    assert "warm cache status is partial" in message
    assert "warm compiled regions were not cache hits" in message
    assert "warm report contains" in message
    assert "compiler probe observed 2 warm invocation" in message
    assert "source hashes changed" in message
    assert "execution-plan identities or source hashes changed" in message
    assert "was not discovered as selected" in message
    assert "has no accepted trial" in message
    assert "below 1.05x marginal" in message
    assert "below 1.10x overall" not in message
    assert "warm execution-plan trial cache status is miss" in message
    assert "did not leave a promoted wheel" in message
    assert "missing the required 1.05x" in message
    assert "final speedup is below 1.10x" in message


def test_append_compile_policy_uses_external_profiled_workload(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    workload = tmp_path / "evidence" / "workload.py"
    pyproject.write_text(
        '[project]\nname = "fixture"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )

    append_compile_policy(pyproject, workload)

    parsed = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    compile_config = cast(
        dict[str, object],
        cast(dict[str, object], cast(dict[str, object], parsed["tool"])["atoll"])["compile"],
    )
    assert compile_config["backends"] == ["mypyc", "cython"]
    assert compile_config["test_command"] == ["python", str(workload.resolve()), "--verify"]
    assert compile_config["benchmark_command"] == ["python", str(workload.resolve())]
    assert compile_config["benchmark_warmups"] == 1
    assert compile_config["benchmark_samples"] == BENCHMARK_SAMPLES
    assert compile_config["minimum_speedup"] == MINIMUM_FINAL_SPEEDUP


def test_evaluate_reports_requires_discovered_execution_plans() -> None:
    cold = _report(cache_status="miss", mypyc_seconds=COLD_TEST_SECONDS)
    warm = _report(cache_status="hit", mypyc_seconds=0.0)
    cold["execution_plans"] = []
    warm["execution_plans"] = []

    evaluation = evaluate_reports(
        BenchmarkEvidenceInputs(
            cold_report=cold,
            warm_report=warm,
            sources_unchanged=True,
            cold_exit_code=0,
            warm_exit_code=0,
            wheel_present=True,
            cold_compiler_probe_count=PROBE_TEST_COUNT,
            warm_compiler_probe_count=0,
        )
    )

    assert "warm report contains no discovered execution plan" in "\n".join(evaluation.errors)


def test_evaluate_reports_requires_applied_execution_plans() -> None:
    cold = _report(cache_status="miss", mypyc_seconds=COLD_TEST_SECONDS)
    warm = _report(cache_status="hit", mypyc_seconds=0.0)
    warm["applied_execution_plans"] = []

    evaluation = evaluate_reports(
        BenchmarkEvidenceInputs(
            cold_report=cold,
            warm_report=warm,
            sources_unchanged=True,
            cold_exit_code=0,
            warm_exit_code=0,
            wheel_present=True,
            cold_compiler_probe_count=PROBE_TEST_COUNT,
            warm_compiler_probe_count=0,
        )
    )

    assert "warm report contains no applied execution plan" in "\n".join(evaluation.errors)


def test_evaluate_reports_rejects_missing_execution_plan_source_hashes() -> None:
    cold = _report(cache_status="miss", mypyc_seconds=0.0, native_evidence=False)
    warm = _report(cache_status="hit", mypyc_seconds=0.0, native_evidence=False)
    cold_plan = cast(dict[str, object], cast(list[object], cold["execution_plans"])[0])
    warm_plan = cast(dict[str, object], cast(list[object], warm["execution_plans"])[0])
    cold_plan["source_hash"] = None
    warm_plan["source_hash"] = None
    cold_plan["source_hashes"] = {}
    warm_plan["source_hashes"] = {}

    evaluation = evaluate_reports(
        BenchmarkEvidenceInputs(
            cold_report=cold,
            warm_report=warm,
            sources_unchanged=True,
            cold_exit_code=0,
            warm_exit_code=0,
            wheel_present=True,
            cold_compiler_probe_count=0,
            warm_compiler_probe_count=0,
        )
    )

    assert evaluation.succeeded is False
    message = "\n".join(evaluation.errors)
    assert "cold selected execution plan" in message
    assert "has no source hash" in message
    assert "has no per-module source hashes" in message


def test_evaluate_reports_rejects_subthreshold_execution_plan_margin() -> None:
    cold = _report(cache_status="miss", mypyc_seconds=COLD_TEST_SECONDS)
    warm = _report(
        cache_status="hit",
        mypyc_seconds=0.0,
        options=_ReportOptions(
            execution_plan_marginal_speedup=1.04,
            execution_plan_overall_speedup=1.09,
        ),
    )

    evaluation = evaluate_reports(
        BenchmarkEvidenceInputs(
            cold_report=cold,
            warm_report=warm,
            sources_unchanged=True,
            cold_exit_code=0,
            warm_exit_code=0,
            wheel_present=True,
            cold_compiler_probe_count=PROBE_TEST_COUNT,
            warm_compiler_probe_count=0,
        )
    )

    message = "\n".join(evaluation.errors)
    assert "below 1.05x marginal" in message
    assert "below 1.10x overall" not in message


def test_source_manifest_detects_only_python_source_changes(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    source = package / "worker.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    (package / "ignored.txt").write_text("first\n", encoding="utf-8")
    before = source_manifest(package)

    (package / "ignored.txt").write_text("second\n", encoding="utf-8")
    assert source_manifest(package) == before
    source.write_text("VALUE = 2\n", encoding="utf-8")
    assert source_manifest(package) != before


def test_benchmark_templates_and_baseline_are_well_formed() -> None:
    workload = TEMPLATE_ROOT / "workload.py.in"
    compile(workload.read_text(encoding="utf-8"), str(workload), "exec")
    semantic_probe = TEMPLATE_ROOT / "ceiling_semantics.py.in"
    compile(semantic_probe.read_text(encoding="utf-8"), str(semantic_probe), "exec")
    probe = (TEMPLATE_ROOT / "compiler_probe.sh.in").read_text(encoding="utf-8")
    assert probe.startswith("#!/bin/sh\n")
    assert 'exec "$compiler" "$@"' in probe
    baseline = cast(
        dict[str, object],
        json.loads((TEMPLATE_ROOT / "baseline.json").read_text(encoding="utf-8")),
    )
    assert baseline["cold_mypycify_seconds"] == COLD_MYPYC_BASELINE_SECONDS
    assert baseline["target_mypycify_seconds"] == COLD_MYPYC_TARGET_SECONDS


@dataclass(frozen=True, slots=True)
class _ReportOptions:
    marginal_speedup: float = 1.06
    region_cache_status: str = "hit"
    execution_plan_cache_status: str | None = None
    execution_plan_marginal_speedup: float = 1.25
    execution_plan_overall_speedup: float = 1.50


DEFAULT_REPORT_OPTIONS = _ReportOptions()


def _report(
    *,
    cache_status: str,
    mypyc_seconds: float,
    final_speedup: float = FINAL_TEST_SPEEDUP,
    native_evidence: bool = True,
    options: _ReportOptions = DEFAULT_REPORT_OPTIONS,
) -> dict[str, object]:
    timings: list[dict[str, object]] = []
    if mypyc_seconds:
        timings.append({"name": "mypycify", "duration_seconds": mypyc_seconds})
    candidate_trials: list[dict[str, object]] = []
    compiled_regions: list[dict[str, object]] = []
    typed_regions: list[dict[str, object]] = []
    if native_evidence:
        candidate_trials.append(
            {"status": "accepted", "marginal_speedup": options.marginal_speedup}
        )
        compiled_regions.append({"cache_status": options.region_cache_status})
        typed_regions.append({"id": "fixture::hot", "source_hash": "source-hash"})
    plan_cache_status = options.execution_plan_cache_status or (
        "miss" if cache_status == "miss" else "hit"
    )
    return {
        "version": 5,
        "success": True,
        "build": {"cache_status": cache_status, "phase_timings": timings},
        "profile": {"status": "profiled"},
        "performance": {"speedup": final_speedup},
        "candidate_trials": candidate_trials,
        "compiled_regions": compiled_regions,
        "typed_regions": typed_regions,
        "execution_plans": [
            {
                "id": PLAN_ID,
                "status": "selected",
                "source_hash": "plan-source-hash",
                "source_hashes": {"fixture.scheduler": "plan-module-source-hash"},
            }
        ],
        "applied_execution_plans": [PLAN_ID],
        "execution_plan_trials": [
            {
                "plan_id": PLAN_ID,
                "status": "accepted",
                "marginal_speedup": options.execution_plan_marginal_speedup,
                "overall_speedup": options.execution_plan_overall_speedup,
                "cache_status": plan_cache_status,
            }
        ],
        "fusion_plans": [
            {
                "id": "task-fusion:fixture",
                "source_hash": "fusion-source-hash",
                "eligible": False,
            }
        ],
        "fusion_trials": [],
    }
