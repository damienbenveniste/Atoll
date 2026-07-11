"""Deterministic checks for the manual Pydantic Graph hard-benchmark harness."""

from __future__ import annotations

import json
import tomllib
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
    assert evaluation.fusion_plan_count == 1
    assert evaluation.eligible_fusion_plan_count == 0
    assert evaluation.fusion_trial_count == 0


def test_evaluate_reports_explains_every_hard_gate_failure() -> None:
    cold = _report(cache_status="partial", mypyc_seconds=COLD_MYPYC_TARGET_SECONDS + 1)
    warm = _report(
        cache_status="partial",
        mypyc_seconds=1.0,
        final_speedup=MINIMUM_FINAL_SPEEDUP - 0.01,
        marginal_speedup=MINIMUM_MARGINAL_SPEEDUP - 0.01,
        region_cache_status="miss",
    )
    warm["version"] = 2
    cast(dict[str, object], warm["profile"])["status"] = "static-fallback"
    warm["fusion_plans"] = []
    warm["fusion_trials"] = [
        {
            "plan_id": "task-fusion:unrelated",
            "status": "passed",
            "unfused_over_fused": 1.0,
            "baseline_over_fused": 1.0,
        }
    ]

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
    assert "warm report is not schema version 3" in message
    assert "warm profile status is static-fallback" in message
    assert "cold cache status is partial" in message
    assert "cold mypyc took" in message
    assert "compiler probe observed no cold native invocation" in message
    assert "warm cache status is partial" in message
    assert "warm compiled regions were not cache hits" in message
    assert "warm report contains" in message
    assert "compiler probe observed 2 warm invocation" in message
    assert "source hashes changed" in message
    assert "task-fusion plan identities or source hashes changed" in message
    assert "contains no task-fusion safety plan" in message
    assert "does not match an eligible" in message
    assert "did not leave a promoted wheel" in message
    assert "missing the required 1.01x" in message
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


def test_evaluate_reports_requires_one_passing_trial_for_each_eligible_plan() -> None:
    cold = _report(cache_status="miss", mypyc_seconds=COLD_TEST_SECONDS)
    warm = _report(
        cache_status="hit",
        mypyc_seconds=0.0,
        final_speedup=MINIMUM_FINAL_SPEEDUP - 0.01,
    )
    plan = cast(dict[str, object], cast(list[object], warm["fusion_plans"])[0])
    plan["eligible"] = True
    warm["fusion_trials"] = [
        {
            "plan_id": plan["id"],
            "status": "not-profitable",
            "unfused_over_fused": 1.20,
            "baseline_over_fused": 1.20,
        }
    ]

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

    assert "has no passed three-arm trial" in "\n".join(evaluation.errors)


def test_evaluate_reports_rejects_passed_trial_for_unrelated_plan() -> None:
    cold = _report(cache_status="miss", mypyc_seconds=COLD_TEST_SECONDS)
    warm = _report(cache_status="hit", mypyc_seconds=0.0)
    warm["fusion_trials"] = [
        {
            "plan_id": "task-fusion:unrelated",
            "status": "passed",
            "unfused_over_fused": 1.20,
            "baseline_over_fused": 1.20,
        }
    ]

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

    assert "does not match an eligible" in "\n".join(evaluation.errors)


def test_evaluate_reports_rejects_subthreshold_matching_fusion_trial() -> None:
    cold = _report(cache_status="miss", mypyc_seconds=COLD_TEST_SECONDS)
    warm = _report(
        cache_status="hit",
        mypyc_seconds=0.0,
        final_speedup=MINIMUM_FINAL_SPEEDUP - 0.01,
    )
    plan = cast(dict[str, object], cast(list[object], warm["fusion_plans"])[0])
    plan["eligible"] = True
    warm["fusion_trials"] = [
        {
            "plan_id": plan["id"],
            "status": "passed",
            "unfused_over_fused": 1.0,
            "baseline_over_fused": 1.0,
        }
    ]

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

    assert "passed task-fusion trial is below" in "\n".join(evaluation.errors)


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


def _report(
    *,
    cache_status: str,
    mypyc_seconds: float,
    final_speedup: float = FINAL_TEST_SPEEDUP,
    marginal_speedup: float = 1.02,
    region_cache_status: str = "hit",
) -> dict[str, object]:
    timings: list[dict[str, object]] = []
    if mypyc_seconds:
        timings.append({"name": "mypycify", "duration_seconds": mypyc_seconds})
    return {
        "version": 3,
        "success": True,
        "build": {"cache_status": cache_status, "phase_timings": timings},
        "profile": {"status": "profiled"},
        "performance": {"speedup": final_speedup},
        "candidate_trials": [{"status": "accepted", "marginal_speedup": marginal_speedup}],
        "compiled_regions": [{"cache_status": region_cache_status}],
        "typed_regions": [{"id": "fixture::hot", "source_hash": "source-hash"}],
        "fusion_plans": [
            {
                "id": "task-fusion:fixture",
                "source_hash": "fusion-source-hash",
                "eligible": False,
            }
        ],
        "fusion_trials": [],
    }
