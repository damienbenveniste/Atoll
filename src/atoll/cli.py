"""Command-line interface for Atoll."""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path

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
from atoll.commands.scan import ScanOptions, execute_scan
from atoll.commands.trial import TrialOptions, execute_trial
from atoll.commands.verify import VerifyOptions, execute_verify
from atoll.models import CompileAttempt, EnabledIslandConfig, VerifyResult
from atoll.project import discover_project
from atoll.report import (
    CompilationReportInput,
    build_compilation_report,
    write_compilation_json_report,
    write_compilation_markdown_report,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Atoll command-line interface."""
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
    compile_cmd = subparsers.add_parser(
        "compile",
        help="enable, build, and verify discovered candidates",
    )
    compile_cmd.add_argument(
        "module",
        nargs="?",
        default=None,
        help="optional source module to compile",
    )
    compile_cmd.add_argument("--root", type=Path, default=Path(), help="project root")
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
    trial = subparsers.add_parser("trial", help="compile and test candidates in a temp overlay")
    trial.add_argument("--root", type=Path, default=Path(), help="project root")
    trial.add_argument("--candidate", default=None, help="candidate like app.module::symbol,helper")
    trial.add_argument("--top", type=int, default=None, help="try the top N scan candidates")
    trial.add_argument("--test", default=None, help='pytest command, for example "pytest tests"')
    trial.add_argument(
        "--benchmark",
        default=None,
        help='pytest benchmark command, for example "pytest benchmarks"',
    )
    trial.add_argument("--keep-temp", action="store_true", help="keep the temporary overlay")
    compiled = trial.add_mutually_exclusive_group()
    compiled.add_argument(
        "--require-compiled",
        action="store_true",
        default=True,
        help="require compiled extension routing after build",
    )
    compiled.add_argument(
        "--allow-python-sidecar",
        action="store_false",
        dest="require_compiled",
        help="do not require compiled extension routing after build",
    )
    return parser


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

    verify_results = execute_verify(
        VerifyOptions(root=args.root, module_name=args.module, require_compiled=True)
    )
    verify_failures = _print_verify_results(verify_results)
    project_root = discover_project(args.root).config.root
    islands = tuple(enabled.island for enabled in enable_result.enabled)
    cleanup_removed = (
        _cleanup_successful_compile_outputs(project_root, islands) if not verify_failures else ()
    )
    report_paths = _write_compilation_report(
        CompilationReportInput(
            root=project_root,
            operation="compile",
            module_filter=args.module,
            islands=islands,
            build=build_result,
            verification=verify_results,
            cleanup_removed=cleanup_removed,
        )
    )
    _print_compilation_report_paths(report_paths)
    if verify_failures:
        return 1
    print("Atoll compile succeeded.")
    return 0


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
        )
    )
    if result.error is not None:
        print(result.error)
    print(f"Atoll trial overlay: {result.overlay_root}")
    print(f"Atoll trial enabled {len(result.enabled)} island(s).")
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


def _print_compilation_report_paths(paths: tuple[Path, Path]) -> None:
    json_path, markdown_path = paths
    print(f"Compilation reports: {json_path}, {markdown_path}")


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
    "verify": _run_verify,
    "explain": _run_explain,
    "clean": _run_clean,
    "trial": _run_trial,
}
