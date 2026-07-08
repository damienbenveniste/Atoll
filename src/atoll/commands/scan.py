"""Implementation of the `atoll scan` command."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from atoll.analysis.clustering import enrich_island_analysis
from atoll.analysis.type_readiness import attach_mypy_diagnostics
from atoll.backends.mypy import run_mypy
from atoll.cache import CacheStats, scan_modules_with_cache
from atoll.models import ScanResult
from atoll.project import discover_project
from atoll.report import ScanReport, build_scan_report, write_json_report, write_markdown_report


@dataclass(frozen=True, slots=True)
class ScanOptions:
    """User-facing options for a scan run."""

    root: Path
    source_roots: tuple[Path, ...] = ()
    json_path: Path | None = None
    markdown_path: Path | None = None
    max_files: int | None = None
    mypy_enabled: bool = True


@dataclass(frozen=True, slots=True)
class ScanCommandResult:
    """Artifacts produced by `atoll scan`."""

    result: ScanResult
    report: ScanReport
    json_path: Path
    markdown_path: Path
    cache: CacheStats


def execute_scan(options: ScanOptions) -> ScanCommandResult:
    """Run project discovery, AST scanning, and report generation."""
    discovered = discover_project(
        options.root,
        source_roots=options.source_roots,
        max_files=options.max_files,
    )
    module_scans, cache_stats = scan_modules_with_cache(discovered.config, discovered.modules)
    if options.mypy_enabled:
        mypy_run = run_mypy(discovered.config)
        module_scans = attach_mypy_diagnostics(module_scans, mypy_run.diagnostics)
    module_scans = tuple(enrich_island_analysis(module) for module in module_scans)
    result = ScanResult(config=discovered.config, modules=module_scans)
    report = build_scan_report(result)
    json_path = options.json_path or discovered.config.report_dir / "report.json"
    markdown_path = options.markdown_path or discovered.config.report_dir / "report.md"
    write_json_report(json_path, report)
    write_markdown_report(markdown_path, report)
    return ScanCommandResult(
        result=result,
        report=report,
        json_path=json_path,
        markdown_path=markdown_path,
        cache=cache_stats,
    )
