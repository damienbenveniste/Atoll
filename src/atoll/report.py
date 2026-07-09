"""Report rendering for Atoll scan results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

from atoll.models import (
    Blocker,
    BlockerSeverity,
    CompileAttempt,
    Confidence,
    ConstantKind,
    DependencyKind,
    DiagnosticSeverity,
    EnabledIslandConfig,
    IslandRisk,
    ModuleScan,
    MypyDiagnostic,
    ScanResult,
    SymbolId,
    SymbolKind,
    VerifyResult,
    Visibility,
)

_STRONG_SCORE = 90
_GOOD_SCORE = 80
_POSSIBLE_SCORE = 70
_ATOLL_PART_INDEX = 0
_ATOLL_GENERATED_INPUT_DIR_INDEX = 1


class BlockerReport(TypedDict):
    severity: BlockerSeverity
    code: str
    message: str
    lineno: int | None
    symbol: str | None


class ImportReport(TypedDict):
    source_text: str
    imported_names: list[str]
    module: str | None
    level: int
    lineno: int
    end_lineno: int


class ConstantReport(TypedDict):
    name: str
    kind: ConstantKind
    source_text: str
    lineno: int
    end_lineno: int


class SymbolReport(TypedDict):
    id: str
    qualname: str
    kind: SymbolKind
    visibility: Visibility
    lineno: int
    end_lineno: int
    decorators: list[str]
    arg_count: int
    annotated_arg_count: int
    has_return_annotation: bool
    has_any_annotation: bool
    called_names: list[str]
    uses_globals: list[str]
    local_names: list[str]
    referenced_names: list[str]
    blockers: list[BlockerReport]
    mypy_diagnostics: list[MypyDiagnosticReport]


class MypyDiagnosticReport(TypedDict):
    path: str
    line: int
    column: int | None
    severity: DiagnosticSeverity
    code: str | None
    message: str
    symbol: str | None


class DependencyEdgeReport(TypedDict):
    src: str
    dst: str
    kind: DependencyKind
    confidence: Confidence
    lineno: int | None


class IslandCandidateReport(TypedDict):
    symbols: list[str]
    required_imports: list[str]
    required_constants: list[str]
    required_local_symbols: list[str]
    rejected_symbols: list[str]
    score: int
    score_label: str
    score_summary: str
    risk: IslandRisk
    risk_summary: str
    reasons: list[str]


class PoisonRadiusReport(TypedDict):
    poison: str
    impacted: list[str]
    reason: str


class ModuleReport(TypedDict):
    module: str
    path: str
    imports: list[ImportReport]
    constants: list[ConstantReport]
    symbols: list[SymbolReport]
    blockers: list[BlockerReport]
    top_level_statement_lines: list[int]
    mypy_diagnostics: list[MypyDiagnosticReport]
    dependency_edges: list[DependencyEdgeReport]
    island_candidates: list[IslandCandidateReport]
    poison_radii: list[PoisonRadiusReport]


class SummaryReport(TypedDict):
    modules_scanned: int
    symbols_scanned: int
    island_candidates: int
    hard_blockers: int
    soft_blockers: int


class ScanReport(TypedDict):
    version: int
    tool: str
    project_root: str
    source_roots: list[str]
    summary: SummaryReport
    modules: list[ModuleReport]


CompilationOperation = Literal["build", "compile"]


class CompilationSummaryReport(TypedDict):
    islands: int
    symbols: int
    artifacts: int
    support_artifacts: int
    verified: int
    verify_failures: int
    duration_seconds: float


class CompilationBuildReport(TypedDict):
    success: bool
    command: list[str]
    duration_seconds: float
    stdout: str
    stderr: str
    artifacts: list[str]
    support_artifacts: list[str]


class CompilationCleanupReport(TypedDict):
    removed: list[str]


class CompilationVerifySymbolReport(TypedDict):
    symbol: str
    rebound: bool


class CompilationVerifyReport(TypedDict):
    active: bool
    compiled: bool
    origin: str | None
    symbols: list[CompilationVerifySymbolReport]
    error: str | None


class CompilationIslandReport(TypedDict):
    source_module: str
    source_path: str
    generated_module: str
    symbols: list[str]
    artifacts: list[str]
    verification: CompilationVerifyReport | None


class CompilationReport(TypedDict):
    version: int
    tool: str
    operation: CompilationOperation
    project_root: str
    module_filter: str | None
    success: bool
    summary: CompilationSummaryReport
    build: CompilationBuildReport
    cleanup: CompilationCleanupReport
    islands: list[CompilationIslandReport]


@dataclass(frozen=True, slots=True)
class CompilationReportInput:
    """Inputs needed to render one compilation report."""

    root: Path
    operation: CompilationOperation
    module_filter: str | None
    islands: tuple[EnabledIslandConfig, ...]
    build: CompileAttempt
    verification: tuple[VerifyResult, ...] = ()
    cleanup_removed: tuple[Path, ...] = ()


def build_scan_report(result: ScanResult) -> ScanReport:
    """Convert scan dataclasses into a stable JSON report shape."""
    module_reports = [_module_report(module) for module in result.modules]
    all_blockers = [
        blocker
        for module in result.modules
        for blocker in (
            *module.blockers,
            *(blocker for symbol in module.symbols for blocker in symbol.blockers),
        )
    ]
    hard_count = sum(blocker.severity == "hard" for blocker in all_blockers)
    soft_count = sum(blocker.severity == "soft" for blocker in all_blockers)
    return {
        "version": 1,
        "tool": "atoll",
        "project_root": str(result.config.root),
        "source_roots": [str(path) for path in result.config.source_roots],
        "summary": {
            "modules_scanned": len(result.modules),
            "symbols_scanned": sum(len(module.symbols) for module in result.modules),
            "island_candidates": sum(len(module.island_candidates) for module in result.modules),
            "hard_blockers": hard_count,
            "soft_blockers": soft_count,
        },
        "modules": module_reports,
    }


def write_json_report(path: Path, report: ScanReport) -> None:
    """Write `report` as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8")


def write_markdown_report(path: Path, report: ScanReport) -> None:
    """Write a human-readable scan report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown_report(report), encoding="utf-8")


def render_markdown_report(report: ScanReport) -> str:
    """Render a concise Markdown scan report."""
    lines = [
        "# Atoll Scan Report",
        "",
        "## Summary",
        "",
        f"- Modules scanned: {report['summary']['modules_scanned']}",
        f"- Symbols scanned: {report['summary']['symbols_scanned']}",
        f"- Island candidates: {report['summary']['island_candidates']}",
        f"- Hard blockers: {report['summary']['hard_blockers']}",
        f"- Soft blockers: {report['summary']['soft_blockers']}",
        "",
        "## How To Read Candidates",
        "",
        "- Score is a 0-100 heuristic for how promising the island looks before compilation.",
        "- Risk is extraction risk: `low` means Atoll saw only high-confidence dependencies.",
        "- Candidates are predictions; `atoll build` and `atoll verify` prove them.",
        "",
    ]
    for module in report["modules"]:
        lines.extend(_markdown_module(module))
    return "\n".join(lines).rstrip() + "\n"


def build_compilation_report(report_input: CompilationReportInput) -> CompilationReport:
    """Convert a build or compile attempt into a stable compilation report."""
    verify_by_module = {result.source_module: result for result in report_input.verification}
    artifact_paths = tuple(report_input.build.artifact_paths)
    island_artifacts = {
        island.source_module: _island_artifacts(island, artifact_paths)
        for island in report_input.islands
    }
    mapped_artifacts = {
        artifact for artifacts in island_artifacts.values() for artifact in artifacts
    }
    support_artifacts = tuple(path for path in artifact_paths if path not in mapped_artifacts)
    verify_failures = sum(result.error is not None for result in report_input.verification)
    success = report_input.build.success and verify_failures == 0
    return {
        "version": 1,
        "tool": "atoll",
        "operation": report_input.operation,
        "project_root": str(report_input.root.resolve()),
        "module_filter": report_input.module_filter,
        "success": success,
        "summary": {
            "islands": len(report_input.islands),
            "symbols": sum(len(island.symbols) for island in report_input.islands),
            "artifacts": len(artifact_paths),
            "support_artifacts": len(support_artifacts),
            "verified": len(report_input.verification),
            "verify_failures": verify_failures,
            "duration_seconds": report_input.build.duration_seconds,
        },
        "build": {
            "success": report_input.build.success,
            "command": _build_command_report(report_input.root, report_input.build.command),
            "duration_seconds": report_input.build.duration_seconds,
            "stdout": report_input.build.stdout,
            "stderr": report_input.build.stderr,
            "artifacts": [_path_text(report_input.root, path) for path in artifact_paths],
            "support_artifacts": [
                _path_text(report_input.root, path) for path in support_artifacts
            ],
        },
        "cleanup": {
            "removed": [
                *_generated_input_cleanup_reports(report_input.root, report_input.cleanup_removed),
                *[
                    _path_text(report_input.root, path)
                    for path in report_input.cleanup_removed
                    if not _is_generated_input_path(report_input.root, path)
                ],
            ],
        },
        "islands": [
            {
                "source_module": island.source_module,
                "source_path": _path_text(report_input.root, island.source_path),
                "generated_module": island.sidecar_module,
                "symbols": list(island.symbols),
                "artifacts": [
                    _path_text(report_input.root, path)
                    for path in island_artifacts[island.source_module]
                ],
                "verification": _compilation_verify_report(
                    verify_by_module.get(island.source_module)
                ),
            }
            for island in report_input.islands
        ],
    }


def write_compilation_json_report(path: Path, report: CompilationReport) -> None:
    """Write a machine-readable compilation report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8")


def write_compilation_markdown_report(path: Path, report: CompilationReport) -> None:
    """Write a human-readable compilation report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_compilation_markdown_report(report), encoding="utf-8")


def render_compilation_markdown_report(report: CompilationReport) -> str:
    """Render a concise Markdown report for a build or compile attempt."""
    status = "success" if report["success"] else "failed"
    module_filter = report["module_filter"] or "all enabled modules"
    lines = [
        "# Atoll Compilation Report",
        "",
        "## Summary",
        "",
        f"- Operation: {report['operation']}",
        f"- Status: {status}",
        f"- Module filter: {module_filter}",
        f"- Islands: {report['summary']['islands']}",
        f"- Symbols: {report['summary']['symbols']}",
        f"- Artifacts: {report['summary']['artifacts']}",
        f"- Support artifacts: {report['summary']['support_artifacts']}",
        f"- Verified islands: {report['summary']['verified']}",
        f"- Verification failures: {report['summary']['verify_failures']}",
        f"- Build duration: {report['summary']['duration_seconds']:.3f}s",
        "",
        "## Build",
        "",
        f"- Success: {_yes_no(report['build']['success'])}",
        f"- Command: `{' '.join(report['build']['command'])}`",
    ]
    if report["build"]["stderr"]:
        lines.append(f"- Error: `{_first_line(report['build']['stderr'])}`")
    if report["build"]["artifacts"]:
        lines.extend(["", "### Artifacts", ""])
        lines.extend(f"- `{artifact}`" for artifact in report["build"]["artifacts"])
    if report["build"]["support_artifacts"]:
        lines.extend(["", "### Support Artifacts", ""])
        lines.extend(f"- `{artifact}`" for artifact in report["build"]["support_artifacts"])
    lines.extend(["", "## Cleanup", ""])
    if report["cleanup"]["removed"]:
        lines.extend(f"- Removed `{path}`" for path in report["cleanup"]["removed"])
    else:
        lines.append("- Removed: none")
    lines.extend(["", "## Islands", ""])
    if not report["islands"]:
        lines.append("- None")
    for island in report["islands"]:
        lines.extend(_compilation_markdown_island(island))
    return "\n".join(lines).rstrip() + "\n"


def _module_report(module: ModuleScan) -> ModuleReport:
    return {
        "module": module.module.name,
        "path": str(module.module.path),
        "imports": [
            {
                "source_text": record.source_text,
                "imported_names": list(record.imported_names),
                "module": record.module,
                "level": record.level,
                "lineno": record.lineno,
                "end_lineno": record.end_lineno,
            }
            for record in module.imports
        ],
        "constants": [
            {
                "name": record.name,
                "kind": record.kind,
                "source_text": record.source_text,
                "lineno": record.lineno,
                "end_lineno": record.end_lineno,
            }
            for record in module.constants
        ],
        "symbols": [
            {
                "id": symbol.id.stable_id,
                "qualname": symbol.id.qualname,
                "kind": symbol.kind,
                "visibility": symbol.visibility,
                "lineno": symbol.lineno,
                "end_lineno": symbol.end_lineno,
                "decorators": list(symbol.decorators),
                "arg_count": symbol.arg_count,
                "annotated_arg_count": symbol.annotated_arg_count,
                "has_return_annotation": symbol.has_return_annotation,
                "has_any_annotation": symbol.has_any_annotation,
                "called_names": list(symbol.called_names),
                "uses_globals": list(symbol.uses_globals),
                "local_names": list(symbol.local_names),
                "referenced_names": list(symbol.referenced_names),
                "blockers": [_blocker_report(blocker) for blocker in symbol.blockers],
                "mypy_diagnostics": [
                    _mypy_diagnostic_report(diagnostic) for diagnostic in symbol.mypy_diagnostics
                ],
            }
            for symbol in module.symbols
        ],
        "blockers": [_blocker_report(blocker) for blocker in module.blockers],
        "top_level_statement_lines": list(module.top_level_statement_lines),
        "mypy_diagnostics": [
            _mypy_diagnostic_report(diagnostic) for diagnostic in module.mypy_diagnostics
        ],
        "dependency_edges": [
            {
                "src": edge.src.stable_id,
                "dst": _edge_dst_text(edge.dst),
                "kind": edge.kind,
                "confidence": edge.confidence,
                "lineno": edge.lineno,
            }
            for edge in module.dependency_edges
        ],
        "island_candidates": [
            {
                "symbols": [symbol.stable_id for symbol in candidate.symbols],
                "required_imports": list(candidate.required_imports),
                "required_constants": list(candidate.required_constants),
                "required_local_symbols": [
                    symbol.stable_id for symbol in candidate.required_local_symbols
                ],
                "rejected_symbols": [symbol.stable_id for symbol in candidate.rejected_symbols],
                "score": candidate.score,
                "score_label": score_label(candidate.score),
                "score_summary": score_summary(candidate.score),
                "risk": candidate.risk,
                "risk_summary": risk_summary(candidate.risk),
                "reasons": list(candidate.reasons),
            }
            for candidate in module.island_candidates
        ],
        "poison_radii": [
            {
                "poison": radius.poison.stable_id,
                "impacted": [symbol.stable_id for symbol in radius.impacted],
                "reason": radius.reason,
            }
            for radius in module.poison_radii
        ],
    }


def _blocker_report(blocker: Blocker) -> BlockerReport:
    return {
        "severity": blocker.severity,
        "code": blocker.code,
        "message": blocker.message,
        "lineno": blocker.lineno,
        "symbol": blocker.symbol.stable_id if blocker.symbol is not None else None,
    }


def _mypy_diagnostic_report(diagnostic: MypyDiagnostic) -> MypyDiagnosticReport:
    return {
        "path": str(diagnostic.path),
        "line": diagnostic.line,
        "column": diagnostic.column,
        "severity": diagnostic.severity,
        "code": diagnostic.code,
        "message": diagnostic.message,
        "symbol": diagnostic.symbol.stable_id if diagnostic.symbol is not None else None,
    }


def _edge_dst_text(dst: SymbolId | str) -> str:
    if isinstance(dst, SymbolId):
        return dst.stable_id
    return dst


def score_label(score: int) -> str:
    """Return a short label for a candidate score."""
    if score >= _STRONG_SCORE:
        return "strong"
    if score >= _GOOD_SCORE:
        return "good"
    if score >= _POSSIBLE_SCORE:
        return "possible"
    return "weak"


def score_summary(score: int) -> str:
    """Explain a candidate score in user-facing language."""
    label = score_label(score)
    if label == "strong":
        detail = "very promising scan-only candidate"
    elif label == "good":
        detail = "promising scan-only candidate"
    elif label == "possible":
        detail = "worth trying, but less compelling"
    else:
        detail = "below Atoll's normal recommendation threshold"
    return f"{score}/100, {detail}"


def risk_summary(risk: IslandRisk) -> str:
    """Explain candidate extraction risk in user-facing language."""
    if risk == "low":
        return "low extraction risk; only high-confidence internal dependencies were seen"
    if risk == "medium":
        return "medium extraction risk; a low-confidence dependency needs trial validation"
    return "high extraction risk; expect manual review before enabling"


def _markdown_module(module: ModuleReport) -> list[str]:
    lines = [f"## {module['module']}", ""]
    if not module["symbols"]:
        lines.extend(["No top-level symbols found.", ""])
        return lines
    for symbol in module["symbols"]:
        blocker_codes = ", ".join(blocker["code"] for blocker in symbol["blockers"])
        status = blocker_codes or "no blockers"
        lines.append(f"- `{symbol['qualname']}` ({symbol['kind']}): {status}")
    if module["island_candidates"]:
        lines.extend(["", "Candidates:"])
        for candidate in module["island_candidates"]:
            symbols = ", ".join(f"`{symbol}`" for symbol in candidate["symbols"])
            lines.append(f"- {candidate['score_summary']}; {candidate['risk_summary']}: {symbols}")
    if module["poison_radii"]:
        lines.extend(["", "Poison residue:"])
        for radius in module["poison_radii"]:
            impacted = ", ".join(radius["impacted"]) or "none"
            lines.append(f"- `{radius['poison']}` impacts: {impacted}")
    lines.append("")
    return lines


def _island_artifacts(
    island: EnabledIslandConfig,
    artifact_paths: tuple[Path, ...],
) -> tuple[Path, ...]:
    return tuple(
        path for path in artifact_paths if path.name.startswith(f"{island.sidecar_path.stem}.")
    )


def _compilation_verify_report(result: VerifyResult | None) -> CompilationVerifyReport | None:
    if result is None:
        return None
    return {
        "active": result.active,
        "compiled": result.compiled,
        "origin": result.origin,
        "symbols": [{"symbol": symbol, "rebound": rebound} for symbol, rebound in result.symbols],
        "error": result.error,
    }


def _compilation_markdown_island(island: CompilationIslandReport) -> list[str]:
    verification = island["verification"]
    if verification is None:
        verify_status = "not run"
    elif verification["error"] is None:
        verify_status = "ok"
    else:
        verify_status = f"failed: {verification['error']}"
    lines = [
        f"### {island['source_module']}",
        "",
        f"- Source: `{island['source_path']}`",
        f"- Generated module: `{island['generated_module']}`",
        f"- Symbols: {', '.join(island['symbols'])}",
        f"- Verification: {verify_status}",
    ]
    if island["artifacts"]:
        lines.append("- Artifacts:")
        lines.extend(f"  - `{artifact}`" for artifact in island["artifacts"])
    else:
        lines.append("- Artifacts: none")
    lines.append("")
    return lines


def _path_text(root: Path, path: Path) -> str:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        return str(resolved_path)


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _first_line(value: str) -> str:
    return next((line for line in value.splitlines() if line.strip()), "")


def _build_command_report(root: Path, command: tuple[str, ...]) -> list[str]:
    reported: list[str] = []
    generated_inputs = 0
    for item in command:
        if _is_generated_input_text(root, item):
            generated_inputs += 1
            continue
        if generated_inputs:
            reported.append(_generated_inputs_label(generated_inputs))
            generated_inputs = 0
        reported.append(item)
    if generated_inputs:
        reported.append(_generated_inputs_label(generated_inputs))
    return reported


def _generated_input_cleanup_reports(
    root: Path,
    paths: tuple[Path, ...],
) -> list[str]:
    count = sum(_is_generated_input_path(root, path) for path in paths)
    return [_generated_inputs_label(count)] if count else []


def _generated_inputs_label(count: int) -> str:
    suffix = "" if count == 1 else "s"
    return f"<{count} generated Python build input{suffix}>"


def _is_generated_input_text(root: Path, value: str) -> bool:
    if ".atoll/sidecars/" in value or value == ".atoll/sidecars":
        return True
    try:
        return _is_generated_input_path(root, Path(value))
    except OSError:
        return False


def _is_generated_input_path(root: Path, path: Path) -> bool:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    parts = relative.parts
    return (
        len(parts) > _ATOLL_GENERATED_INPUT_DIR_INDEX
        and parts[_ATOLL_PART_INDEX] == ".atoll"
        and parts[_ATOLL_GENERATED_INPUT_DIR_INDEX] == "sidecars"
    )
