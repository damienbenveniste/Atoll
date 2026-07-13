"""Deterministic checks for the schema-v6 Pydantic Graph hard-benchmark gate."""

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
    COMPILE_REPORT_VERSION,
    MINIMUM_COMPOSED_MARGINAL_SPEEDUP,
    MINIMUM_FINAL_SPEEDUP,
    MINIMUM_SOURCE_SPEEDUP,
    BenchmarkEvidenceInputs,
    append_compile_policy,
    evaluate_reports,
    source_manifest,
)

TEMPLATE_ROOT = Path("benchmarks/pydantic_graph")
PLAN_ID = "source-opt-fixture"
CANDIDATE_ID = "source-candidate-fixture"
PATCH_PATH = f".atoll/patches/{CANDIDATE_ID}.patch"
SOURCE_SPEEDUP = 3.6
WHEEL_SPEEDUP = 3.5
COMPOSED_MARGINAL_SPEEDUP = 1.25
FINAL_SPEEDUP = WHEEL_SPEEDUP * COMPOSED_MARGINAL_SPEEDUP
TRANSFORMATION_IDS = [
    "private-transport-batch-drain:batch-drain-v1:fixture.Owner.consume",
    "quiescent-callable-execution:quiescent-callable-v1:fixture.Owner.worker",
    "local-state-machine-fusion:state-machine-v1:fixture.Owner.dispatch",
    "private-protocol-auto-forwarding:protocol-forward-v1:fixture.run",
]


def test_evaluate_reports_accepts_reproducible_3x_source_patch() -> None:
    evaluation = evaluate_reports(
        _inputs(
            cold=_report(cache_status="miss"),
            warm=_report(cache_status="hit"),
        )
    )

    assert evaluation.succeeded is True
    assert evaluation.errors == ()
    assert evaluation.cold_patch_cache_status == "miss"
    assert evaluation.warm_patch_cache_status == "hit"
    assert evaluation.final_speedup == FINAL_SPEEDUP
    assert evaluation.composed_marginal_speedup == COMPOSED_MARGINAL_SPEEDUP
    assert evaluation.source_speedup == SOURCE_SPEEDUP
    assert evaluation.wheel_speedup == WHEEL_SPEEDUP
    assert evaluation.source_plan_count == 1
    assert evaluation.trial_ready_source_plan_count == 1
    assert evaluation.source_trial_count == 1
    assert evaluation.accepted_source_trials == 1
    assert evaluation.patch_path == PATCH_PATH
    assert evaluation.application_status == "not-applied"
    assert evaluation.cold_native_phase_count == 0
    assert evaluation.warm_native_phase_count == 0


def test_evaluate_reports_rejects_every_source_promotion_boundary() -> None:
    cold = _report(cache_status="hit")
    warm = _report(
        cache_status="miss",
        source_speedup=MINIMUM_SOURCE_SPEEDUP - 0.01,
        wheel_speedup=MINIMUM_FINAL_SPEEDUP - 0.01,
        composed_marginal_speedup=0.9,
    )
    warm["version"] = 4
    cast(dict[str, object], warm["profile"])["status"] = "static-fallback"
    cast(dict[str, object], warm["performance"])["status"] = "not-profitable"
    cast(dict[str, object], warm["performance"])["samples"] = []
    source = cast(dict[str, object], warm["source_optimization"])
    source["status"] = "not-profitable"
    source["minimum_speedup"] = 2.9
    source["application_status"] = "applied"
    source["patch_path"] = "outside.patch"
    plan = cast(dict[str, object], cast(list[object], source["plans"])[0])
    identity = cast(dict[str, object], plan["identity"])
    identity["source_hashes"] = {}
    assessment = cast(dict[str, object], cast(list[object], source["assessments"])[0])
    assessment["observed_work_items"] = 999
    assessment["attributed_hot_share"] = 0.2
    trial = cast(dict[str, object], cast(list[object], source["trials"])[0])
    trial["candidate_id"] = "changed-candidate"
    trial["patch_path"] = "outside.patch"
    trial["semantic_exit_code"] = 1
    trial["source_edits"] = []
    trial["transformation_ids"] = TRANSFORMATION_IDS[:-1]

    evaluation = evaluate_reports(
        _inputs(
            cold=cold,
            warm=warm,
            options=_InputOptions(
                sources_unchanged=False,
                cold_exit_code=1,
                warm_exit_code=1,
                wheel_present=False,
                patch_present=False,
                warm_compiler_probe_count=2,
            ),
        )
    )

    assert evaluation.succeeded is False
    message = "\n".join(evaluation.errors)
    assert "cold compile did not succeed" in message
    assert f"warm report is not schema version {COMPILE_REPORT_VERSION}" in message
    assert "warm profile status is static-fallback" in message
    assert "warm performance status is not-profitable" in message
    assert "warm source-optimization status is not-profitable" in message
    assert "below 3.000x" in message
    assert "cold source-patch cache status is hit" in message
    assert "warm source-patch cache status is miss" in message
    assert "compiler probe observed 2 warm invocation" in message
    assert "source hashes changed" in message
    assert "source-plan identities or source hashes changed" in message
    assert "accepted source candidate or patch identity changed" in message
    assert "outside .atoll/patches" in message
    assert "application status is applied" in message
    assert "no valid per-file source hashes" in message
    assert "no safe .atoll patch path" in message
    assert "contains no source edits" in message
    assert "did not pass semantic tests" in message
    assert "private-protocol-auto-forwarding" in message
    assert "fewer than 10,000 work items" in message
    assert "less than 70%" in message
    assert "did not rerun 7 baseline/compiled timing pairs" in message
    assert "did not leave a promoted wheel" in message
    assert "did not leave the accepted source patch" in message
    assert "transformed source speedup is below 3.00x" in message
    assert "normal wheel speedup is below 3.00x" in message
    assert "final speedup is below 3.00x" in message
    assert "accepted source-only arm by less than 1.05x" in message


def test_evaluate_reports_requires_source_plan_and_accepted_trial() -> None:
    cold = _report(cache_status="miss")
    warm = _report(cache_status="hit")
    for report in (cold, warm):
        source = cast(dict[str, object], report["source_optimization"])
        source["plans"] = []
        source["assessments"] = []
        source["trials"] = []
        source["patch_path"] = None

    evaluation = evaluate_reports(
        _inputs(cold=cold, warm=warm, options=_InputOptions(patch_present=False))
    )

    message = "\n".join(evaluation.errors)
    assert "contains no source-optimization plan" in message
    assert "contains no trial-ready source plan" in message
    assert "contains 0 accepted source trials" in message


def test_evaluate_reports_rejects_stale_source_identity() -> None:
    cold = _report(cache_status="miss")
    warm = _report(cache_status="hit")
    source = cast(dict[str, object], warm["source_optimization"])
    plan = cast(dict[str, object], cast(list[object], source["plans"])[0])
    identity = cast(dict[str, object], plan["identity"])
    identity["source_hashes"] = {"fixture/workflow.py": "changed"}

    evaluation = evaluate_reports(_inputs(cold=cold, warm=warm))

    assert "source-plan identities or source hashes changed" in "\n".join(evaluation.errors)


def test_append_compile_policy_profiles_only_the_async_pipeline(tmp_path: Path) -> None:
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
    assert compile_config["test_command"] == [
        "python",
        str(workload.resolve()),
        "--verify",
    ]
    assert compile_config["benchmark_command"] == [
        "python",
        str(workload.resolve()),
        "--build-repetitions",
        "0",
    ]
    assert compile_config["benchmark_warmups"] == 1
    assert compile_config["benchmark_samples"] == BENCHMARK_SAMPLES
    assert compile_config["minimum_speedup"] == MINIMUM_FINAL_SPEEDUP


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


def test_benchmark_templates_and_legacy_baseline_are_well_formed() -> None:
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
class _InputOptions:
    sources_unchanged: bool = True
    cold_exit_code: int = 0
    warm_exit_code: int = 0
    wheel_present: bool = True
    patch_present: bool = True
    warm_compiler_probe_count: int = 0


DEFAULT_INPUT_OPTIONS = _InputOptions()


def _inputs(
    *,
    cold: dict[str, object],
    warm: dict[str, object],
    options: _InputOptions = DEFAULT_INPUT_OPTIONS,
) -> BenchmarkEvidenceInputs:
    return BenchmarkEvidenceInputs(
        cold_report=cold,
        warm_report=warm,
        sources_unchanged=options.sources_unchanged,
        cold_exit_code=options.cold_exit_code,
        warm_exit_code=options.warm_exit_code,
        wheel_present=options.wheel_present,
        patch_present=options.patch_present,
        cold_compiler_probe_count=0,
        warm_compiler_probe_count=options.warm_compiler_probe_count,
    )


def _report(
    *,
    cache_status: str,
    source_speedup: float = SOURCE_SPEEDUP,
    wheel_speedup: float = WHEEL_SPEEDUP,
    composed_marginal_speedup: float = COMPOSED_MARGINAL_SPEEDUP,
) -> dict[str, object]:
    baseline_median = wheel_speedup
    wheel_median = 1.0
    composed_median = wheel_median / composed_marginal_speedup
    samples = [
        {"mode": mode, "duration_seconds": 1.0}
        for _ in range(BENCHMARK_SAMPLES)
        for mode in ("baseline", "compiled")
    ]
    return {
        "version": COMPILE_REPORT_VERSION,
        "success": True,
        "build": {"cache_status": "disabled", "phase_timings": []},
        "profile": {"status": "profiled"},
        "performance": {
            "status": "passed",
            "baseline_median_seconds": wheel_median,
            "compiled_median_seconds": composed_median,
            "speedup": composed_marginal_speedup,
            "samples": samples,
        },
        "optimization_policy": {
            "version": 1,
            "stability_floor_seconds": 0.25,
            "profile_guided_minimum_marginal_speedup": 1.01,
            "specialized_minimum_marginal_speedup": MINIMUM_COMPOSED_MARGINAL_SPEEDUP,
            "final_minimum_speedup": MINIMUM_FINAL_SPEEDUP,
            "hard_benchmark_minimum_speedup": MINIMUM_FINAL_SPEEDUP,
        },
        "stage_medians": [
            {
                "stage": "source-final",
                "status": "accepted",
                "baseline_median_seconds": baseline_median,
                "candidate_median_seconds": wheel_median,
                "speedup": wheel_speedup,
                "minimum_speedup": MINIMUM_SOURCE_SPEEDUP,
            },
            {
                "stage": "final-payload",
                "status": "passed",
                "baseline_median_seconds": wheel_median,
                "candidate_median_seconds": composed_median,
                "speedup": composed_marginal_speedup,
                "minimum_speedup": MINIMUM_COMPOSED_MARGINAL_SPEEDUP,
            },
        ],
        "final_composition": {
            "source_plan_ids": [PLAN_ID],
            "transformation_ids": TRANSFORMATION_IDS,
            "native_variant_ids": ["fixture::native@cython"],
            "execution_plan_ids": [],
            "artifacts": [".atoll/artifacts/fixture.so"],
            "wheel_path": ".atoll/dist/fixture.whl",
            "retained_previous_arm": False,
        },
        "source_optimization": {
            "status": "accepted",
            "minimum_speedup": 3.0,
            "patch_path": PATCH_PATH,
            "application_status": "not-applied",
            "plans": [
                {
                    "id": PLAN_ID,
                    "identity": {
                        "dialect": "anyio-on-asyncio",
                        "python_abi": "cpython-312",
                        "source_hashes": {
                            "fixture/workflow.py": "source-hash",
                        },
                    },
                }
            ],
            "assessments": [
                {
                    "plan_id": PLAN_ID,
                    "status": "trial-ready",
                    "observed_work_items": 20_000,
                    "attributed_hot_share": 0.8,
                }
            ],
            "trials": [
                {
                    "plan_id": PLAN_ID,
                    "status": "accepted",
                    "candidate_id": CANDIDATE_ID,
                    "source_speedup": source_speedup,
                    "wheel_speedup": wheel_speedup,
                    "baseline_median_seconds": baseline_median,
                    "source_median_seconds": baseline_median / source_speedup,
                    "wheel_median_seconds": wheel_median,
                    "patch_path": PATCH_PATH,
                    "source_edits": [
                        {
                            "path": "fixture/workflow.py",
                            "before_hash": "source-hash",
                            "after_hash": "optimized-hash",
                        }
                    ],
                    "application_status": "not-applied",
                    "diagnostics": [f"cache {cache_status}"],
                    "transformation_ids": TRANSFORMATION_IDS,
                    "semantic_exit_code": 0,
                }
            ],
        },
    }
