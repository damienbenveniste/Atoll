"""Canonical JSON and Markdown rendering for corpus case evidence.

JSON is authoritative.  Markdown is generated exclusively from the immutable
result object so human summaries cannot silently diverge from aggregation data.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import cast

from scripts.benchmark_corpus.models import CaseResult, CaseStatus, CorpusTier


def classify_compile_report(report: dict[str, object], tier: CorpusTier) -> CaseStatus:
    """Classify a successful compile without erasing compatible no-op cases.

    Args:
        report: Valid schema-v6 Atoll compile report.
        tier: Corpus role controlling whether profitability evidence is required.

    Returns:
        CaseStatus: Acceleration, profitability, stability, compiled, or no-op status.
    """
    if tier == "performance":
        performance = _mapping_field(report, "performance")
        status = performance.get("status")
        if status == "passed":
            return "accelerated"
        if status == "not-profitable":
            return "not-profitable"
        if status == "invalid" and "too noisy" in str(performance.get("reason", "")):
            return "unstable"
    composition = _mapping_field(report, "final_composition")
    active = any(
        isinstance(composition.get(key), list) and bool(composition[key])
        for key in ("source_plan_ids", "native_variant_ids", "execution_plan_ids", "artifacts")
    )
    return "compiled-unbenchmarked" if active else "supported-no-op"


def write_case_result(result: CaseResult, evidence_root: Path) -> tuple[Path, Path]:
    """Write schema-v1 JSON and deterministic Markdown for one case.

    Args:
        result: Complete success or failure evidence envelope.
        evidence_root: Case-specific directory receiving both reports.

    Returns:
        tuple[Path, Path]: JSON path followed by Markdown path.
    """
    evidence_root.mkdir(parents=True, exist_ok=True)
    json_path = evidence_root / "case-result.json"
    markdown_path = evidence_root / "case-result.md"
    json_path.write_text(
        f"{json.dumps(asdict(result), default=_json_default, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_case_markdown(result), encoding="utf-8")
    return json_path, markdown_path


def render_case_markdown(result: CaseResult) -> str:
    """Render one result with ratios named by the arms they compare.

    Args:
        result: Immutable corpus case evidence.

    Returns:
        str: Deterministic Markdown ending in one newline.
    """
    lines = [
        f"# {result.case_id}",
        "",
        f"- Status: `{result.status}`",
        f"- Tier: `{result.tier}`",
        f"- Platform: `{result.platform}`",
        f"- Upstream revision: `{result.revision}`",
        f"- Source unchanged after policy injection: `{_optional_bool(result.source_unchanged)}`",
        f"- Python rewrite versus original: {_ratio(result.ratios.python_rewrite_vs_original)}",
        f"- Final wheel versus original: {_ratio(result.ratios.final_wheel_vs_original)}",
        f"- Native layer versus source-only wheel: {_ratio(result.ratios.native_vs_source_only)}",
        "",
        "## Diagnostics",
        "",
    ]
    lines.extend(f"- {diagnostic}" for diagnostic in result.diagnostics)
    if not result.diagnostics:
        lines.append("- None")
    lines.extend(("", "## Phases", ""))
    lines.extend(
        (
            f"- `{phase.name}`: exit `{phase.exit_code}`, "
            f"{phase.duration_seconds:.3f}s, log `{phase.log_path}`"
        )
        for phase in result.phases
    )
    if not result.phases:
        lines.append("- None")
    return f"{'\n'.join(lines)}\n"


def _ratio(value: float | None) -> str:
    return "not measured" if value is None else f"{value:.3f}x"


def _optional_bool(value: bool | None) -> str:
    if value is None:
        return "unknown"
    return "yes" if value else "no"


def _json_default(value: object) -> str:
    if isinstance(value, PurePosixPath):
        return value.as_posix()
    raise TypeError(f"unsupported corpus result value: {type(value).__name__}")


def _mapping_field(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    return cast(dict[str, object], value) if isinstance(value, dict) else {}
