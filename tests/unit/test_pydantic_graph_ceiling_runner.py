"""Runner-level tests for the Pydantic Graph ceiling experiment."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from scripts import pydantic_graph_ceiling
from scripts.pydantic_graph_ceiling import (
    ARMS,
    CeilingExperimentOptions,
    CeilingExperimentResult,
    CommandEvidence,
    ExperimentArm,
)

BASELINE_SAMPLE_SECONDS = 4.0
REFLECTION_SAMPLE_SECONDS = 2.0
BUFFERED_SAMPLE_SECONDS = 1.5
UNSAFE_CEILING_SAMPLE_SECONDS = 1.0
REPORT_HEADROOM = 3.0
EXPECTED_CEILING_HEADROOM = BASELINE_SAMPLE_SECONDS / UNSAFE_CEILING_SAMPLE_SECONDS
EXPECTED_MANIFEST_CALLS = 2


def test_summary_reports_markdown_and_sorted_json(tmp_path: Path) -> None:
    result = _summarize_result(tmp_path, source_unchanged=True)
    _write_reports(result)

    payload = json.loads(result.report_json.read_text(encoding="utf-8"))
    markdown = result.report_markdown.read_text(encoding="utf-8")

    assert payload["observed_headroom"] == EXPECTED_CEILING_HEADROOM
    assert payload["promising_research_direction"] is True
    assert payload["source_unchanged"] is True
    assert [summary["arm"] for summary in payload["summaries"]] == list(ARMS)
    assert len(payload["commands"]) == len(ARMS) + 1
    assert payload["commands"][-1] == {
        "arm": "baseline",
        "duration_seconds": BASELINE_SAMPLE_SECONDS,
        "phase": "sample",
        "returncode": 0,
        "stderr": "",
        "stdout": "ok",
    }
    assert markdown.startswith("# Pydantic Graph Optimization Ceiling\n\n")
    assert "- Checkout sources unchanged: yes" in markdown
    assert "- Scheduler semantics: `not established`" in markdown
    assert "- Recommendation: `investigate-guarded-design`" in markdown
    assert "| baseline | 4.000000s | 1.000x | passed | yes | reference |" in markdown
    assert "| unsafe_ceiling | 1.000000s | 4.000x | passed | no | not-established |" in markdown


def test_runner_marks_fast_result_unpromising_when_staged_sources_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_calls = 0

    def fake_validate_inputs(_options: CeilingExperimentOptions, _package_root: Path) -> None:
        return None

    def fake_source_manifest(_package_root: Path) -> dict[str, str]:
        nonlocal manifest_calls
        manifest_calls += 1
        return {"join.py": f"digest-{manifest_calls}"}

    def fake_stage_payloads(_package_root: Path, payload_root: Path) -> dict[ExperimentArm, Path]:
        return {arm: payload_root / arm for arm in ARMS}

    def fake_semantic_probes(
        _options: CeilingExperimentOptions, _payloads: dict[ExperimentArm, Path]
    ) -> tuple[list[CommandEvidence], dict[ExperimentArm, bool]]:
        return [], dict.fromkeys(ARMS, True)

    def fake_measurements(
        _options: CeilingExperimentOptions,
        _payloads: dict[ExperimentArm, Path],
        _commands: list[CommandEvidence],
    ) -> dict[ExperimentArm, list[float]]:
        return {
            "baseline": [BASELINE_SAMPLE_SECONDS],
            "reflection": [REFLECTION_SAMPLE_SECONDS],
            "buffered": [BUFFERED_SAMPLE_SECONDS],
            "unsafe_ceiling": [UNSAFE_CEILING_SAMPLE_SECONDS],
        }

    monkeypatch.setattr(pydantic_graph_ceiling, "_validate_inputs", fake_validate_inputs)
    monkeypatch.setattr(pydantic_graph_ceiling, "source_manifest", fake_source_manifest)
    monkeypatch.setattr(pydantic_graph_ceiling, "_stage_payloads", fake_stage_payloads)
    monkeypatch.setattr(pydantic_graph_ceiling, "_run_semantic_probes", fake_semantic_probes)
    monkeypatch.setattr(pydantic_graph_ceiling, "_run_measurements", fake_measurements)

    result = pydantic_graph_ceiling.run_ceiling_experiment(
        CeilingExperimentOptions(
            checkout=tmp_path / "checkout",
            evidence_root=tmp_path / "evidence",
            workload=tmp_path / "workload.py",
            semantic_probe=tmp_path / "semantic.py",
            python=tmp_path / "python",
            minimum_headroom=REPORT_HEADROOM,
        )
    )

    payload = json.loads(result.report_json.read_text(encoding="utf-8"))
    markdown = result.report_markdown.read_text(encoding="utf-8")

    assert manifest_calls == EXPECTED_MANIFEST_CALLS
    assert result.source_unchanged is False
    assert result.observed_headroom == EXPECTED_CEILING_HEADROOM
    assert result.promising_research_direction is False
    assert payload["source_unchanged"] is False
    assert payload["promising_research_direction"] is False
    assert "- Checkout sources unchanged: no" in markdown
    assert "- Recommendation: `stop`" in markdown


def _summarize_result(tmp_path: Path, *, source_unchanged: bool) -> CeilingExperimentResult:
    return _summarize(
        CeilingExperimentOptions(
            checkout=tmp_path / "checkout",
            evidence_root=tmp_path,
            workload=tmp_path / "workload.py",
            semantic_probe=tmp_path / "semantic.py",
            python=tmp_path / "python",
            minimum_headroom=REPORT_HEADROOM,
        ),
        source_unchanged,
        dict.fromkeys(ARMS, True),
        {
            "baseline": [BASELINE_SAMPLE_SECONDS],
            "reflection": [REFLECTION_SAMPLE_SECONDS],
            "buffered": [BUFFERED_SAMPLE_SECONDS],
            "unsafe_ceiling": [UNSAFE_CEILING_SAMPLE_SECONDS],
        },
        [
            *(
                CommandEvidence(
                    arm=arm,
                    phase="probe",
                    returncode=0,
                    duration_seconds=0.1,
                    stdout=json.dumps(
                        {
                            "arm": arm,
                            "context_isolated": arm != "unsafe_ceiling",
                            "signature_guarded": True,
                        }
                    ),
                    stderr="",
                )
                for arm in ARMS
            ),
            CommandEvidence(
                arm="baseline",
                phase="sample",
                returncode=0,
                duration_seconds=BASELINE_SAMPLE_SECONDS,
                stdout="ok",
                stderr="",
            ),
        ],
    )


_summarize = cast(
    Callable[
        [
            CeilingExperimentOptions,
            bool,
            dict[ExperimentArm, bool],
            dict[ExperimentArm, list[float]],
            list[CommandEvidence],
        ],
        CeilingExperimentResult,
    ],
    vars(pydantic_graph_ceiling)["_summarize"],
)
_write_reports = cast(
    Callable[[CeilingExperimentResult], None],
    vars(pydantic_graph_ceiling)["_write_reports"],
)
