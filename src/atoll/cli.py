"""Command-line interface for Atoll.

The CLI translates parsed arguments into typed command option objects and keeps
user-facing printing at the boundary. Command modules own behavior and return
structured results; this module owns parser wiring, exit codes, summaries, and
report file emission after build/package workflows.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from atoll.commands.build import BuildOptions, execute_build
from atoll.commands.clean import CleanOptions, execute_clean
from atoll.commands.disable import DisableOptions, execute_disable
from atoll.commands.enable import (
    EnableAllOptions,
    EnableCommandResult,
    EnableOptions,
    execute_enable,
    execute_enable_all,
)
from atoll.commands.explain import ExplainOptions, execute_explain
from atoll.commands.generate import GenerateOptions, execute_generate
from atoll.commands.package import (
    PackageBuildFailure,
    PackageCommandResult,
    PackageOptions,
    PackagePreflightFailure,
    execute_package,
)
from atoll.commands.scan import ScanOptions, execute_scan
from atoll.commands.trial import TrialOptions, execute_trial
from atoll.commands.verify import VerifyOptions, execute_verify
from atoll.config import CompileConfigError
from atoll.models import CompileAttempt, EnabledIslandConfig, PytestRunResult, VerifyResult
from atoll.project import discover_project
from atoll.report import (
    CompilationPreflightBlockerInput,
    CompilationReportInput,
    CompilationSkippedModuleInput,
    build_compilation_report,
    write_compilation_json_report,
    write_compilation_markdown_report,
)
from atoll.runtime.test_runner import parse_pytest_command, run_pytest_command


class _Subparsers(Protocol):
    """Small parser protocol used by helper functions that register subcommands."""

    def add_parser(
        self,
        name: str,
        *,
        help: str | None = None,
    ) -> argparse.ArgumentParser:
        """Create and return an argparse parser for one subcommand.

        Args:
            name: Source or parser name being resolved.
            help: Help text shown by the argument parser.

        Returns:
            argparse.ArgumentParser: Configured child argument parser.
        """
        ...


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Atoll command-line interface and return a process exit code.

    `argv` is accepted for tests and programmatic callers. When no subcommand is
    provided, help is printed and the command fails with exit code 1.

    Args:
        argv: Command-line arguments excluding the executable name; `None` reads `sys.argv`.

    Returns:
        int: Process exit code selected from command success or failure.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = _COMMAND_HANDLERS.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atoll",
        description="Discover compileable Python islands.",
    )
    subparsers = parser.add_subparsers(dest="command")
    scan = subparsers.add_parser("scan", help="scan a Python project")
    scan.add_argument("root", nargs="?", default=".", help="project root to scan")
    scan.add_argument(
        "--source-root",
        action="append",
        default=(),
        help="source root relative to root",
    )
    scan.add_argument("--json", type=Path, default=None, help="path for JSON report")
    scan.add_argument("--markdown", type=Path, default=None, help="path for Markdown report")
    scan.add_argument("--max-files", type=int, default=None, help="maximum Python files to scan")
    scan.add_argument(
        "--no-mypy",
        action="store_true",
        help="skip mypy diagnostic mapping",
    )
    generate = subparsers.add_parser("generate", help="generate enabled Atoll sidecars")
    generate.add_argument("--root", type=Path, default=Path(), help="project root")
    generate.add_argument("--module", default=None, help="limit generation to one source module")
    generate.add_argument("--check", action="store_true", help="fail if sidecars are stale")
    enable = subparsers.add_parser("enable", help="enable an Atoll island")
    enable.add_argument("module", nargs="?", help="source module to enable")
    enable.add_argument("--root", type=Path, default=Path(), help="project root")
    enable.add_argument("--symbols", default=None, help="comma-separated exported symbols")
    enable.add_argument(
        "--all-candidates",
        action="store_true",
        help="enable every scan candidate, optionally limited to MODULE",
    )
    enable.add_argument("--sidecar", default=None, help="override sidecar module name")
    enable.add_argument("--build", action="store_true", help="compile the enabled sidecar")
    enable.add_argument("--dry-run", action="store_true", help="show changes without writing files")
    enable.add_argument("--yes", action="store_true", help="suppress managed shim diff output")
    _add_compile_parser(subparsers)
    disable = subparsers.add_parser("disable", help="disable an Atoll island")
    disable.add_argument("module", help="source module to disable")
    disable.add_argument("--root", type=Path, default=Path(), help="project root")
    disable.add_argument(
        "--delete-sidecar", action="store_true", help="delete generated sidecar source"
    )
    disable.add_argument(
        "--dry-run", action="store_true", help="show changes without writing files"
    )
    build = subparsers.add_parser("build", help="compile enabled Atoll sidecars")
    build.add_argument("--root", type=Path, default=Path(), help="project root")
    build.add_argument("--module", default=None, help="limit build to one source module")
    build.add_argument("--clean-first", action="store_true", help="remove Atoll build cache first")
    build.add_argument("--inplace", action="store_true", default=True, help=argparse.SUPPRESS)
    _add_package_parser(subparsers)
    verify = subparsers.add_parser("verify", help="verify Atoll managed routing")
    verify.add_argument("--root", type=Path, default=Path(), help="project root")
    verify.add_argument("--module", default=None, help="limit verification to one source module")
    verify.add_argument(
        "--require-compiled",
        action="store_true",
        help="require sidecar modules to be compiled extensions",
    )
    explain = subparsers.add_parser("explain", help="explain a module or symbol")
    explain.add_argument("target", help="module or module::symbol target")
    explain.add_argument("--root", type=Path, default=Path(), help="project root")
    explain.add_argument("--no-mypy", action="store_true", help="skip mypy diagnostics")
    clean = subparsers.add_parser("clean", help="remove Atoll-generated outputs")
    clean.add_argument("--root", type=Path, default=Path(), help="project root")
    clean.add_argument("--cache", action="store_true", help="remove Atoll cache and build dirs")
    clean.add_argument("--artifacts", action="store_true", help="remove compiled sidecar artifacts")
    clean.add_argument("--all", action="store_true", help="remove all Atoll-generated outputs")
    trial = subparsers.add_parser(
        "trial",
        help="compile and test selected typed regions in a temporary wheel",
    )
    trial.add_argument("--root", type=Path, default=Path(), help="project root")
    trial.add_argument("--candidate", default=None, help="candidate like app.module::symbol,helper")
    trial.add_argument("--top", type=int, default=None, help="try the top N scan candidates")
    trial.add_argument(
        "--test",
        default=None,
        help='pytest command run against the compiled payload, for example "pytest tests"',
    )
    trial.add_argument(
        "--benchmark",
        default=None,
        help='pytest workload run after tests; checks exit status, for example "pytest benchmarks"',
    )
    trial.add_argument(
        "--keep-temp",
        action="store_true",
        help="keep the temporary wheel and install payload",
    )
    compiled = trial.add_mutually_exclusive_group()
    compiled.add_argument(
        "--require-compiled",
        action="store_true",
        default=True,
        help="require selected bindings to use compiled extensions during tests",
    )
    compiled.add_argument(
        "--allow-interpreted",
        "--allow-python-sidecar",
        action="store_false",
        dest="require_compiled",
        help="allow interpreted fallback during trial commands",
    )
    return parser


def _add_compile_parser(subparsers: _Subparsers) -> None:
    compile_cmd = subparsers.add_parser(
        "compile",
        help="build source-clean compiled wheel artifacts",
    )
    compile_cmd.add_argument(
        "module",
        nargs="?",
        default=None,
        help="optional source module to compile",
    )
    compile_cmd.add_argument("--root", type=Path, default=Path(), help="project root")
    compile_cmd.add_argument(
        "--in-place",
        action="store_true",
        help="modify source files with Atoll shims before compiling",
    )
    compile_cmd.add_argument(
        "--test",
        default=None,
        help='pytest command to run after --in-place compiled routing, for example "pytest tests"',
    )
    compile_cmd.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output directory for source-clean wheel artifacts",
    )
    compile_cmd.add_argument(
        "--keep-install-tree",
        action="store_true",
        help="keep the temporary source-clean install tree for debugging",
    )
    compile_cmd.add_argument(
        "--apply-source",
        action="store_true",
        help="apply an accepted 3x source patch after semantic and benchmark gates",
    )


def _add_package_parser(subparsers: _Subparsers) -> None:
    package = subparsers.add_parser(
        "package",
        help=argparse.SUPPRESS,
    )
    package.add_argument(
        "module",
        nargs="?",
        default=None,
        help="optional source module to package",
    )
    package.add_argument("--root", type=Path, default=Path(), help="project root")
    package.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output directory for source-clean wheel artifacts",
    )
    package.add_argument(
        "--keep-install-tree",
        action="store_true",
        help="keep the temporary source-clean install tree for debugging",
    )
    package.add_argument(
        "--apply-source",
        action="store_true",
        help="apply an accepted 3x source patch after semantic and benchmark gates",
    )


def _run_scan(args: argparse.Namespace) -> int:
    root = Path(args.root)
    source_roots = tuple(Path(path) for path in args.source_root)
    result = execute_scan(
        ScanOptions(
            root=root,
            source_roots=source_roots,
            json_path=args.json,
            markdown_path=args.markdown,
            max_files=args.max_files,
            mypy_enabled=not args.no_mypy,
        )
    )
    summary = result.report["summary"]
    print(
        "Atoll scanned "
        f"{summary['modules_scanned']} modules and {summary['symbols_scanned']} symbols. "
        f"Cache: {result.cache['hits']} hit(s), {result.cache['misses']} miss(es). "
        f"Reports: {result.json_path}, {result.markdown_path}"
    )
    return 0


def _run_generate(args: argparse.Namespace) -> int:
    result = execute_generate(
        GenerateOptions(root=args.root, module_name=args.module, check=args.check)
    )
    if result.stale_paths:
        for path in result.stale_paths:
            print(f"stale Atoll sidecar: {path}")
        return 1
    print(f"Atoll generated {len(result.generated)} sidecar(s).")
    return 0


def _run_enable(args: argparse.Namespace) -> int:
    if args.all_candidates:
        return _run_enable_all(args)
    if args.module is None or args.symbols is None:
        print("enable requires MODULE and --symbols unless --all-candidates is used")
        return 2
    result = execute_enable(
        EnableOptions(
            root=args.root,
            module_name=args.module,
            symbols=_symbols(args.symbols),
            sidecar_module=args.sidecar,
            dry_run=args.dry_run,
            yes=args.yes,
        )
    )
    _print_enable_result(result, yes=args.yes)
    verb = "would enable" if args.dry_run else "enabled"
    print(f"Atoll {verb} {result.island.source_module}: {', '.join(result.island.symbols)}")
    if args.build and not args.dry_run:
        build_result = execute_build(BuildOptions(root=args.root, module_name=args.module))
        project_root = discover_project(args.root).config.root
        cleanup_removed = (
            _cleanup_successful_build_scratch(project_root) if build_result.success else ()
        )
        report_paths = _write_compilation_report(
            CompilationReportInput(
                root=project_root,
                operation="build",
                module_filter=args.module,
                islands=(result.island,),
                build=build_result,
                cleanup_removed=cleanup_removed,
            )
        )
        _print_compilation_report_paths(report_paths)
        return _build_exit_code(build_result)
    return 0


def _run_enable_all(args: argparse.Namespace) -> int:
    if args.symbols is not None:
        print("--symbols cannot be used with --all-candidates")
        return 2
    if args.sidecar is not None:
        print("--sidecar can only be used with manual single-module enable")
        return 2
    try:
        result = execute_enable_all(
            EnableAllOptions(
                root=args.root,
                module_name=args.module,
                dry_run=args.dry_run,
                yes=args.yes,
            )
        )
    except ValueError as error:
        print(error)
        return 1
    for enabled in result.enabled:
        _print_enable_result(enabled, yes=args.yes)
    verb = "would enable" if args.dry_run else "enabled"
    print(
        f"Atoll {verb} {len(result.enabled)} module(s) "
        f"and {result.symbol_count} candidate symbol(s)."
    )
    for enabled in result.enabled:
        print(f"- {enabled.island.source_module}: {', '.join(enabled.island.symbols)}")
    if args.build and not args.dry_run:
        build_result = execute_build(BuildOptions(root=args.root, module_name=args.module))
        project_root = discover_project(args.root).config.root
        cleanup_removed = (
            _cleanup_successful_build_scratch(project_root) if build_result.success else ()
        )
        report_paths = _write_compilation_report(
            CompilationReportInput(
                root=project_root,
                operation="build",
                module_filter=args.module,
                islands=tuple(enabled.island for enabled in result.enabled),
                build=build_result,
                cleanup_removed=cleanup_removed,
            )
        )
        _print_compilation_report_paths(report_paths)
        return _build_exit_code(build_result)
    return 0


def _print_enable_result(result: EnableCommandResult, *, yes: bool) -> None:
    if not yes and result.shim_edit.diff:
        print(result.shim_edit.diff)


def _run_compile(args: argparse.Namespace) -> int:
    option_error = _compile_option_error(args)
    if option_error is not None:
        print(option_error)
        return 2
    if not args.in_place:
        return _run_source_clean_artifact_build(
            root=args.root,
            module_name=args.module,
            output_dir=args.output,
            keep_install_tree=args.keep_install_tree,
            apply_source=args.apply_source,
        )
    return _run_inplace_compile(args)


def _run_inplace_compile(args: argparse.Namespace) -> int:
    try:
        enable_result = execute_enable_all(
            EnableAllOptions(root=args.root, module_name=args.module, yes=True)
        )
    except ValueError as error:
        print(error)
        return 1

    print(
        f"Atoll enabled {len(enable_result.enabled)} module(s) "
        f"and {enable_result.symbol_count} candidate symbol(s)."
    )
    for enabled in enable_result.enabled:
        print(f"- {enabled.island.source_module}: {', '.join(enabled.island.symbols)}")

    build_result = execute_build(
        BuildOptions(root=args.root, module_name=args.module, clean_first=True)
    )
    if not build_result.success:
        report_paths = _write_compilation_report(
            CompilationReportInput(
                root=discover_project(args.root).config.root,
                operation="compile",
                module_filter=args.module,
                islands=tuple(enabled.island for enabled in enable_result.enabled),
                build=build_result,
            )
        )
        _print_compilation_report_paths(report_paths)
        print(build_result.stderr)
        return 1
    print(f"Atoll build succeeded: {len(build_result.artifact_paths)} artifact(s)")

    project = discover_project(args.root)
    verify_results = execute_verify(
        VerifyOptions(root=args.root, module_name=args.module, require_compiled=True)
    )
    verify_failures = _print_verify_results(verify_results)
    project_root = project.config.root
    islands = tuple(enabled.island for enabled in enable_result.enabled)
    test_result = (
        None
        if verify_failures
        else _run_compile_test_gate(args.test, project_root, project.config.source_roots)
    )
    test_failed = test_result is not None and not test_result.success
    cleanup_removed = (
        _cleanup_successful_compile_outputs(project_root, islands)
        if not verify_failures and not test_failed
        else ()
    )
    report_paths = _write_compilation_report(
        CompilationReportInput(
            root=project_root,
            operation="compile",
            module_filter=args.module,
            islands=islands,
            build=build_result,
            verification=verify_results,
            tests=test_result,
            cleanup_removed=cleanup_removed,
        )
    )
    _print_compilation_report_paths(report_paths)
    if verify_failures:
        return 1
    if test_failed:
        return 1
    _print_compile_success(test_result)
    return 0


def _compile_option_error(args: argparse.Namespace) -> str | None:
    if not args.in_place and args.test is not None:
        return "--test requires --in-place"
    incompatible = (
        (args.output is not None, "--output cannot be used with --in-place"),
        (args.keep_install_tree, "--keep-install-tree cannot be used with --in-place"),
        (args.apply_source, "--apply-source cannot be used with --in-place"),
    )
    if args.in_place:
        error = next((message for blocked, message in incompatible if blocked), None)
        if error is not None:
            return error
    if args.test is None:
        return None
    try:
        parse_pytest_command(args.test)
    except ValueError as error:
        return str(error)
    return None


def _run_compile_test_gate(
    command: str | None,
    project_root: Path,
    source_roots: tuple[Path, ...],
) -> PytestRunResult | None:
    if command is None:
        return None
    result = run_pytest_command(
        command,
        root=project_root,
        source_roots=source_roots,
        require_compiled=True,
    )
    if result.success:
        print(f"Atoll semantic test gate passed: {' '.join(result.command)}")
    else:
        print(
            f"Atoll semantic test gate failed: {' '.join(result.command)} exited {result.exit_code}"
        )
    return result


def _print_compile_success(test_result: PytestRunResult | None) -> None:
    if test_result is None:
        print("Atoll compile succeeded: compiled routing verified; semantic tests not run.")
        return
    print("Atoll compile succeeded: compiled routing and semantic test gate passed.")


def _run_disable(args: argparse.Namespace) -> int:
    result = execute_disable(
        DisableOptions(
            root=args.root,
            module_name=args.module,
            delete_sidecar=args.delete_sidecar,
            dry_run=args.dry_run,
        )
    )
    if result.shim_edit.diff:
        print(result.shim_edit.diff)
    verb = "would disable" if args.dry_run else "disabled"
    print(f"Atoll {verb} {result.island.source_module}")
    return 0


def _run_build(args: argparse.Namespace) -> int:
    result = execute_build(
        BuildOptions(root=args.root, module_name=args.module, clean_first=args.clean_first)
    )
    project_root = discover_project(args.root).config.root
    cleanup_removed = _cleanup_successful_build_scratch(project_root) if result.success else ()
    report_paths = _write_compilation_report(
        CompilationReportInput(
            root=project_root,
            operation="build",
            module_filter=args.module,
            islands=_enabled_islands(args.root, args.module),
            build=result,
            cleanup_removed=cleanup_removed,
        )
    )
    _print_compilation_report_paths(report_paths)
    return _build_exit_code(result)


def _run_package(args: argparse.Namespace) -> int:
    return _run_source_clean_artifact_build(
        root=args.root,
        module_name=args.module,
        output_dir=args.output,
        keep_install_tree=args.keep_install_tree,
        apply_source=args.apply_source,
    )


def _run_source_clean_artifact_build(
    *,
    root: Path,
    module_name: str | None,
    output_dir: Path | None,
    keep_install_tree: bool,
    apply_source: bool,
) -> int:
    progress = _source_clean_progress_reporter()
    try:
        result = execute_package(
            PackageOptions(
                root=root,
                module_name=module_name,
                output_dir=output_dir,
                keep_install_tree=keep_install_tree,
                apply_source=apply_source,
                progress=progress,
            )
        )
    except CompileConfigError as error:
        print(f"Atoll compile configuration error: {error}", file=sys.stderr)
        return 2
    report_paths = _write_source_clean_compile_report(result, module_name=module_name)
    if not result.success:
        _print_compile_report_paths(report_paths)
        print(result.error or result.build.stderr)
        return 1
    _print_source_clean_success(
        result,
        label="source-clean compile",
        report_paths=report_paths,
    )
    return 0


def _print_source_clean_success(
    result: PackageCommandResult,
    *,
    label: str,
    report_paths: tuple[Path, Path],
) -> None:
    """Print the compiled scope, fallbacks, performance status, and artifact paths.

    Args:
        result: Operation result being normalized or rendered.
        label: Human-readable progress label for the operation.
        report_paths: Report artifact paths to retain in the final result.
    """
    compiled_modules = {
        *(island.source_module for island in result.islands),
        *(binding.source.module for binding in result.compiled_bindings),
    }
    compiled_symbols = sum(len(island.symbols) for island in result.islands) + len(
        result.compiled_bindings
    )
    accepted_source_count = sum(
        trial.status == "accepted" for trial in result.source_optimization_trials
    )
    print(
        _source_clean_success_message(
            label=label,
            compiled_module_count=len(compiled_modules),
            compiled_symbol_count=compiled_symbols,
            applied_plan_count=len(result.applied_execution_plans),
            accepted_source_count=accepted_source_count,
        )
    )
    if result.skipped:
        print(f"Skipped {len(result.skipped)} module(s) that mypyc could not build.")
        for failure in result.skipped[:10]:
            first_line = failure.build.stderr.splitlines()[0] if failure.build.stderr else "failed"
            print(f"- {failure.island.source_module}: {first_line}")
    if result.preflight_skipped:
        print(
            f"Skipped {len(result.preflight_skipped)} module(s) with known mypyc typing blockers."
        )
        for preflight_failure in result.preflight_skipped[:10]:
            first = preflight_failure.blockers[0]
            location = f"line {first.lineno}" if first.lineno is not None else "module"
            print(f"- {preflight_failure.scan.module.name}: {location}: {first.message}")
    if result.region_skipped:
        print(f"Kept {len(result.region_skipped)} typed region(s) as interpreted fallback.")
        for region_failure in result.region_skipped[:10]:
            first_line = (
                region_failure.build.stderr.splitlines()[0]
                if region_failure.build.stderr
                else "failed"
            )
            print(f"- {region_failure.variant_id} [{region_failure.backend}]: {first_line}")
    if result.install_tree_kept:
        print(f"Install tree: {result.install_root}")
    _print_candidate_trial_summary(result)
    _print_execution_plan_trial_summary(result)
    _print_source_optimization_trial_summary(result)
    if result.performance is not None:
        if result.performance.status == "passed" and result.performance.speedup is not None:
            print(f"Performance: {result.performance.speedup:.3f}x median speedup (passed).")
        else:
            print(f"Performance: {result.performance.status}.")
    print(f"Wheel: {result.wheel_path}")
    _print_compile_report_paths(report_paths)


def _print_source_optimization_trial_summary(result: PackageCommandResult) -> None:
    """Print the accepted source patch and application state when present.

    Args:
        result: Source-clean result containing source candidate trial evidence.
    """
    accepted = next(
        (trial for trial in result.source_optimization_trials if trial.status == "accepted"),
        None,
    )
    if accepted is None or accepted.patch_path is None:
        return
    print(f"Source optimization: accepted {accepted.candidate_id}.")
    print(f"Patch: {accepted.patch_path}")
    if accepted.application_status == "applied":
        print("Source application: applied and revalidated.")


def _source_clean_success_message(
    *,
    label: str,
    compiled_module_count: int,
    compiled_symbol_count: int,
    applied_plan_count: int,
    accepted_source_count: int,
) -> str:
    """Describe native, scheduler-plan, or interpreted-only wheel work.

    Args:
        label: Human-readable operation label.
        compiled_module_count: Number of modules with retained native bindings.
        compiled_symbol_count: Number of retained native bindings.
        applied_plan_count: Number of profitable scheduler plans in the payload.
        accepted_source_count: Number of source candidates that passed both 3x gates.

    Returns:
        str: Single-line success summary that does not describe plan-only work as zero work.
    """
    if compiled_module_count or compiled_symbol_count:
        message = (
            f"Atoll {label} built {compiled_module_count} module(s) and "
            f"{compiled_symbol_count} symbol(s)"
        )
        if applied_plan_count:
            message += f", and applied {applied_plan_count} async execution plan(s)"
        return f"{message}."
    if accepted_source_count:
        return (
            f"Atoll {label} built a verified project wheel from "
            f"{accepted_source_count} accepted source optimization(s)."
        )
    if applied_plan_count:
        return (
            f"Atoll {label} applied {applied_plan_count} async execution plan(s); "
            "no native regions were retained."
        )
    return f"Atoll {label} produced a verified wheel with interpreted fallbacks only."


def _print_candidate_trial_summary(result: PackageCommandResult) -> None:
    """Print accepted candidate count and retained hot-path coverage.

    Args:
        result: Source-clean compile result containing optional trial evidence.
    """
    if not result.candidate_trials:
        return
    accepted_trials = sum(trial.status == "accepted" for trial in result.candidate_trials)
    accepted_coverage = result.candidate_trials[-1].accepted_hot_coverage
    print(
        f"Candidate trials: {accepted_trials}/{len(result.candidate_trials)} accepted; "
        f"{accepted_coverage:.1%} mapped hot-path coverage."
    )


def _print_execution_plan_trial_summary(result: PackageCommandResult) -> None:
    """Print applied scheduler-plan and helper-cache evidence.

    Args:
        result: Source-clean result containing scheduler-plan trials.
    """
    if not result.execution_plan_trials:
        return
    accepted = sum(trial.status == "accepted" for trial in result.execution_plan_trials)
    cache_hits = sum(trial.cache_status == "hit" for trial in result.execution_plan_trials)
    print(
        f"Execution-plan trials: {accepted}/{len(result.execution_plan_trials)} accepted; "
        f"{cache_hits} staging cache hit(s)."
    )


def _source_clean_progress_reporter(*, operation: str = "compile") -> Callable[[str], None]:
    started = time.perf_counter()

    def progress(message: str) -> None:
        """Print elapsed source-clean compile progress messages to stderr.

        Args:
            message: Progress message emitted to the optional callback.
        """
        elapsed = time.perf_counter() - started
        print(f"Atoll {operation} [{elapsed:6.2f}s] {message}", file=sys.stderr)

    return progress


def _build_exit_code(result: CompileAttempt) -> int:
    if result.success:
        print(f"Atoll build succeeded: {len(result.artifact_paths)} artifact(s)")
        return 0
    print(result.stderr)
    return 1


def _run_verify(args: argparse.Namespace) -> int:
    results = execute_verify(
        VerifyOptions(
            root=args.root,
            module_name=args.module,
            require_compiled=args.require_compiled,
        )
    )
    return 1 if _print_verify_results(results) else 0


def _print_verify_results(results: tuple[VerifyResult, ...]) -> bool:
    failures = False
    for result in results:
        failures = failures or result.error is not None
        state = "ok" if result.error is None else f"failed: {result.error}"
        print(f"{result.source_module} -> {result.sidecar_module}: {state}")
    return failures


def _run_explain(args: argparse.Namespace) -> int:
    text = execute_explain(
        ExplainOptions(root=args.root, target=args.target, mypy_enabled=not args.no_mypy)
    )
    print(text, end="")
    return 0


def _run_clean(args: argparse.Namespace) -> int:
    result = execute_clean(
        CleanOptions(
            root=args.root,
            cache=args.cache,
            artifacts=args.artifacts,
            all_outputs=args.all,
        )
    )
    print(f"Atoll removed {len(result.removed)} path(s).")
    return 0


def _run_trial(args: argparse.Namespace) -> int:
    result = execute_trial(
        TrialOptions(
            root=args.root,
            candidate=args.candidate,
            top=args.top,
            test_command=args.test,
            benchmark_command=args.benchmark,
            keep_temp=args.keep_temp,
            require_compiled=args.require_compiled,
            progress=_source_clean_progress_reporter(operation="trial"),
        )
    )
    if result.error is not None:
        print(result.error)
    print(
        f"Atoll trial selected {len(result.selections)} candidate(s), "
        f"{len(result.selected)} member(s), and compiled "
        f"{len(result.compiled_bindings)} public binding(s)."
    )
    if result.artifact_root.exists():
        print(f"Atoll trial artifacts: {result.artifact_root}")
        if result.wheel_path is not None:
            print(f"Atoll trial wheel: {result.wheel_path}")
    else:
        print("Atoll trial temporary artifacts cleaned.")
    if result.test_exit_code is not None:
        print(f"Atoll trial test exit code: {result.test_exit_code}")
    if result.benchmark_exit_code is not None:
        print(f"Atoll trial benchmark exit code: {result.benchmark_exit_code}")
    return 0 if result.success else 1


def _symbols(value: str) -> tuple[str, ...]:
    symbols = tuple(symbol.strip() for symbol in value.split(",") if symbol.strip())
    if not symbols:
        raise ValueError("--symbols must include at least one symbol")
    return symbols


def _write_compilation_report(report_input: CompilationReportInput) -> tuple[Path, Path]:
    project_root = report_input.root
    report = build_compilation_report(report_input)
    json_path = project_root / ".atoll" / "compilation-report.json"
    markdown_path = project_root / ".atoll" / "compilation-report.md"
    write_compilation_json_report(json_path, report)
    write_compilation_markdown_report(markdown_path, report)
    return json_path, markdown_path


def _write_source_clean_compile_report(
    result: PackageCommandResult,
    *,
    module_name: str | None,
) -> tuple[Path, Path]:
    report = build_compilation_report(
        CompilationReportInput(
            root=result.project_root,
            operation="compile",
            mode="source-clean",
            module_filter=module_name,
            islands=result.islands,
            build=_source_clean_report_build(result),
            wheel_path=result.wheel_path,
            cleanup_removed=result.cleanup_removed,
            cleanup_kept=result.cleanup_kept,
            skipped_modules=(
                *_package_skipped_module_inputs(result.skipped),
                *_package_region_skipped_inputs(result),
            ),
            preflight_blockers=_package_preflight_blocker_inputs(result.preflight_skipped),
            native_readiness=result.native_readiness,
            typed_regions=result.typed_regions,
            compiled_regions=result.compiled_regions,
            compiled_bindings=result.compiled_bindings,
            compiled_variants=result.compiled_variants,
            backend_assessments=result.backend_assessments,
            artifact_records=result.artifact_records,
            verification_steps=result.verification_steps,
            test_results=result.test_results,
            performance=result.performance,
            profile=result.profile,
            candidate_trials=result.candidate_trials,
            execution_plans=result.execution_plans,
            applied_execution_plans=result.applied_execution_plans,
            execution_plan_trials=result.execution_plan_trials,
            fusion_plans=result.fusion_plans,
            fusion_trials=result.fusion_trials,
            source_optimization_plans=result.source_optimization_plans,
            source_optimization_assessments=result.source_optimization_assessments,
            source_optimization_trials=result.source_optimization_trials,
        )
    )
    json_path = result.project_root / ".atoll" / "compile-report.json"
    markdown_path = result.project_root / ".atoll" / "compile-report.md"
    write_compilation_json_report(json_path, report)
    write_compilation_markdown_report(markdown_path, report)
    return json_path, markdown_path


def _source_clean_report_build(result: PackageCommandResult) -> CompileAttempt:
    return CompileAttempt(
        success=result.build.success,
        command=result.build.command,
        stdout=result.build.stdout,
        stderr=result.build.stderr,
        artifact_paths=result.report_artifact_paths,
        duration_seconds=result.build.duration_seconds,
        phase_timings=result.build.phase_timings,
        cache_status=result.build.cache_status,
    )


def _package_skipped_module_inputs(
    skipped: tuple[PackageBuildFailure, ...],
) -> tuple[CompilationSkippedModuleInput, ...]:
    return tuple(
        CompilationSkippedModuleInput(
            module=failure.island.source_module,
            reason=failure.build.stderr or "mypy rejected generated island",
        )
        for failure in skipped
    )


def _package_region_skipped_inputs(
    result: PackageCommandResult,
) -> tuple[CompilationSkippedModuleInput, ...]:
    return tuple(
        CompilationSkippedModuleInput(
            module=failure.region.source_module.name,
            reason=failure.build.stderr or "backend rejected typed region",
        )
        for failure in result.region_skipped
    )


def _package_preflight_blocker_inputs(
    skipped: tuple[PackagePreflightFailure, ...],
) -> tuple[CompilationPreflightBlockerInput, ...]:
    return tuple(
        CompilationPreflightBlockerInput(
            module=failure.scan.module.name,
            path=failure.scan.module.path,
            line=blocker.lineno,
            code=blocker.code,
            message=blocker.message,
        )
        for failure in skipped
        for blocker in failure.blockers
    )


def _print_compilation_report_paths(paths: tuple[Path, Path]) -> None:
    json_path, markdown_path = paths
    print(f"Compilation reports: {json_path}, {markdown_path}")


def _print_compile_report_paths(paths: tuple[Path, Path]) -> None:
    json_path, markdown_path = paths
    print(f"Compile reports: {json_path}, {markdown_path}")


def _cleanup_successful_build_scratch(root: Path) -> tuple[Path, ...]:
    return _remove_path(root / ".atoll" / "build")


def _cleanup_successful_compile_outputs(
    root: Path,
    islands: tuple[EnabledIslandConfig, ...],
) -> tuple[Path, ...]:
    removed: list[Path] = []
    for island in islands:
        removed.extend(_remove_path(island.sidecar_path))
    sidecar_dirs = tuple(
        sorted({island.sidecar_path.parent for island in islands}, key=lambda path: len(path.parts))
    )
    for sidecar_dir in reversed(sidecar_dirs):
        if sidecar_dir.exists() and not any(sidecar_dir.iterdir()):
            sidecar_dir.rmdir()
            removed.append(sidecar_dir)
    removed.extend(_cleanup_successful_build_scratch(root))
    return tuple(removed)


def _remove_path(path: Path) -> tuple[Path, ...]:
    if not path.exists():
        return ()
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return (path,)


def _enabled_islands(root: Path, module_name: str | None) -> tuple[EnabledIslandConfig, ...]:
    project = discover_project(root)
    return tuple(
        island
        for island in project.config.islands
        if island.enabled and (module_name is None or island.source_module == module_name)
    )


_COMMAND_HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "scan": _run_scan,
    "generate": _run_generate,
    "enable": _run_enable,
    "compile": _run_compile,
    "disable": _run_disable,
    "build": _run_build,
    "package": _run_package,
    "verify": _run_verify,
    "explain": _run_explain,
    "clean": _run_clean,
    "trial": _run_trial,
}
