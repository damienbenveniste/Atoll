"""Static contracts for trusted corpus workflow orchestration."""

from __future__ import annotations

import re
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast


class _YamlModule(Protocol):
    """Typed boundary for the test environment's MkDocs YAML dependency."""

    def safe_load(self, stream: str) -> object:
        """Parse one YAML document without object construction."""


yaml = cast(_YamlModule, import_module("yaml"))

COMPATIBILITY = Path(".github/workflows/corpus-compatibility.yml")
PERFORMANCE = Path(".github/workflows/corpus-performance.yml")
NATIVE_CALIBRATION = Path(".github/workflows/native-optimizer-benchmark.yml")
_PINNED_ACTION = re.compile(r"^\s*uses:\s+[^\s@]+@[0-9a-f]{40}(?:\s+#.*)?$", re.MULTILINE)
_USES_LINE = re.compile(r"^\s*uses:\s+.+$", re.MULTILINE)
_PERFORMANCE_CASES = {
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
_MAX_PARALLEL = 4


def test_compatibility_workflow_is_weekly_trusted_and_complete() -> None:
    payload = _workflow(COMPATIBILITY)
    triggers = _mapping(payload["on"])
    schedule = cast(list[dict[str, str]], triggers["schedule"])
    jobs = _mapping(payload["jobs"])
    case_job = _mapping(jobs["case"])
    strategy = _mapping(case_job["strategy"])
    text = COMPATIBILITY.read_text(encoding="utf-8")

    assert schedule == [{"cron": "0 6 * * 0"}]
    assert "pull_request" not in triggers
    assert _mapping(jobs["matrix"])["if"] == "github.ref == 'refs/heads/main'"
    assert strategy["max-parallel"] == _MAX_PARALLEL
    assert "Resolve all 25 pinned cases" in text
    assert "--tier compatibility --platform ubuntu-24.04" in text
    assert "retention-days: 30" in text
    assert "GITHUB_STEP_SUMMARY" in text
    assert "--allow-unsandboxed" in text
    assert "Install Bubblewrap" not in text
    assert "promote" not in text


def test_performance_workflow_has_reviewed_manual_inputs_and_both_platforms() -> None:
    payload = _workflow(PERFORMANCE)
    triggers = _mapping(payload["on"])
    dispatch = _mapping(triggers["workflow_dispatch"])
    inputs = _mapping(dispatch["inputs"])
    case_input = _mapping(inputs["case"])
    platform_input = _mapping(inputs["platform"])
    jobs = _mapping(payload["jobs"])
    text = PERFORMANCE.read_text(encoding="utf-8")

    assert set(triggers) == {"workflow_dispatch"}
    assert set(inputs) == {"case", "platform", "experiment-label"}
    assert set(cast(list[str], case_input["options"])) == _PERFORMANCE_CASES | {"all"}
    assert cast(list[str], platform_input["options"]) == ["ubuntu-24.04", "macos-14"]
    assert _mapping(jobs["matrix"])["if"] == "github.ref == 'refs/heads/main'"
    assert "--tier performance" in text
    assert "Aggregate all 12 expected performance cases" in text
    assert "Validate experiment label" in text
    assert "^[a-z0-9][a-z0-9._-]{0,63}$" in text
    assert "experiment.json" in text
    assert "github.run_id" in text
    assert "github.run_attempt" in text
    assert "github.workflow_ref" in text
    assert "github.sha" in text
    assert "retention-days: 30" in text
    assert "GITHUB_STEP_SUMMARY" in text
    assert "--allow-unsandboxed" in text
    assert "Install Bubblewrap" not in text
    assert "promote" not in text


def test_benchmark_workflows_pin_actions_and_disable_checkout_credentials() -> None:
    for path in (COMPATIBILITY, PERFORMANCE, NATIVE_CALIBRATION):
        text = path.read_text(encoding="utf-8")
        uses = _USES_LINE.findall(text)
        pinned = _PINNED_ACTION.findall(text)

        assert uses
        assert len(pinned) == len(uses)
        assert text.count("persist-credentials: false") == text.count("actions/checkout@")
        assert "permissions:\n  contents: read" in text


def test_native_calibration_workflow_validates_bundle_before_execution() -> None:
    text = NATIVE_CALIBRATION.read_text(encoding="utf-8")
    validation = "python -m scripts.benchmark_corpus validate"
    runner = "python scripts/run_native_optimizer_benchmark.py"

    assert validation in text
    assert runner in text
    assert text.index(validation) < text.index(runner)
    assert '--workspace "$RUNNER_TEMP/native-optimizer-${{ runner.os }}"' in text
    assert '--evidence-root "$RUNNER_TEMP/native-optimizer-evidence-${{ runner.os }}"' in text


def _workflow(path: Path) -> dict[str, object]:
    value: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    raw = cast(dict[object, object], value)
    if True in raw and "on" not in raw:
        raw["on"] = raw.pop(True)
    return _mapping(raw)


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    raw = cast(dict[object, object], value)
    assert all(isinstance(key, str) for key in raw)
    return cast(dict[str, object], value)
