"""End-to-end acceptance for suspension-sensitive typed-region wheels."""

from __future__ import annotations

import json
import shutil
import sys
import venv
from pathlib import Path
from typing import TypedDict, cast

from atoll.cli import main
from atoll.runtime.performance import run_performance_command

FIXTURE_ROOT = Path("tests/fixtures/typed_region_project")
MODULE_NAME = "typed_region_project.async_runner"
COMPILED_BINDINGS = frozenset(
    {
        f"{MODULE_NAME}::ProtocolRunner.async_exchange",
        f"{MODULE_NAME}::ProtocolRunner.cold_decoy",
        f"{MODULE_NAME}::ProtocolRunner.compute",
        f"{MODULE_NAME}::ProtocolRunner.exchange",
        f"{MODULE_NAME}::ProtocolRunner.fail_after_suspend",
        f"{MODULE_NAME}::ProtocolRunner.parse",
        f"{MODULE_NAME}::ProtocolRunner.wait_until_cancelled",
        f"{MODULE_NAME}::ProtocolRunner.with_bias",
    }
)
INTERPRETED_BINDINGS = frozenset(
    {
        f"{MODULE_NAME}::GenericRunner.identity",
        f"{MODULE_NAME}::DynamicRunner.calculate",
    }
)
NATIVE_COMPILER_PHASES = frozenset({"mypycify", "cythonize", "build_ext"})


class _PhaseTiming(TypedDict):
    name: str


class _BuildReport(TypedDict):
    cache_status: str
    phase_timings: list[_PhaseTiming]


class _BindingReport(TypedDict):
    source: str


class _CompiledRegionReport(TypedDict):
    backend: str
    cache_status: str
    bindings: list[_BindingReport]


class _TypedRegionReport(TypedDict):
    id: str
    source_hash: str


class _CompileReport(TypedDict):
    build: _BuildReport
    compiled_regions: list[_CompiledRegionReport]
    typed_regions: list[_TypedRegionReport]


def test_compile_preserves_async_protocols_and_uses_warm_region_cache(tmp_path: Path) -> None:
    """A real wheel preserves suspension semantics under strict and fallback routing."""
    project_root = tmp_path / "typed_region_project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    original_sources = _source_contents(project_root)

    first_exit = main(
        [
            "compile",
            MODULE_NAME,
            "--root",
            str(project_root),
            "--keep-install-tree",
        ]
    )
    first_report = _read_report(project_root)
    second_exit = main(
        [
            "compile",
            MODULE_NAME,
            "--root",
            str(project_root),
            "--keep-install-tree",
        ]
    )
    report = _read_report(project_root)

    output_root = project_root / ".atoll" / "dist"
    install_root = output_root / "install"
    wheel_path = next(output_root.glob("*.whl"))
    assert first_exit == 0
    assert second_exit == 0
    assert first_report["build"]["cache_status"] == "miss"
    assert report["build"]["cache_status"] == "hit"
    assert all(region["cache_status"] == "hit" for region in report["compiled_regions"])
    assert NATIVE_COMPILER_PHASES.isdisjoint(
        timing["name"] for timing in report["build"]["phase_timings"]
    )
    assert _source_contents(project_root) == original_sources
    assert _region_hashes(report) == _region_hashes(first_report)
    assert _compiled_bindings(report) == COMPILED_BINDINGS
    assert _compiled_bindings(report).isdisjoint(INTERPRETED_BINDINGS)
    assert {region["backend"] for region in report["compiled_regions"]} == {"cython"}
    assert wheel_path.is_file()
    assert not (output_root / "build").exists()
    assert not tuple(install_root.rglob("_atoll_region_*.py"))
    assert tuple((install_root / ".atoll" / "artifacts").rglob("*.so"))

    fixture_test = project_root / "tests" / "test_async_runner.py"
    compiled = run_performance_command(
        (sys.executable, "-m", "pytest", "-q", str(fixture_test)),
        project_root=project_root,
        payload_root=install_root,
        mode="compiled",
    )
    fallback = run_performance_command(
        (sys.executable, "-m", "pytest", "-q", str(fixture_test)),
        project_root=project_root,
        payload_root=install_root,
        mode="baseline",
    )
    assert compiled.succeeded, compiled.stderr
    assert fallback.succeeded, fallback.stderr
    _assert_installed_wheel_routes(wheel_path, tmp_path)


def _read_report(project_root: Path) -> _CompileReport:
    return cast(
        _CompileReport,
        json.loads((project_root / ".atoll" / "compile-report.json").read_text()),
    )


def _source_contents(root: Path) -> dict[Path, str]:
    return {
        path.relative_to(root): path.read_text(encoding="utf-8")
        for path in sorted((root / "src").rglob("*.py"))
    }


def _region_hashes(report: _CompileReport) -> dict[str, str]:
    return {region["id"]: region["source_hash"] for region in report["typed_regions"]}


def _compiled_bindings(report: _CompileReport) -> frozenset[str]:
    return frozenset(
        binding["source"] for region in report["compiled_regions"] for binding in region["bindings"]
    )


def _assert_installed_wheel_routes(wheel_path: Path, tmp_path: Path) -> None:
    environment_root = tmp_path / "async-wheel-environment"
    venv.EnvBuilder(with_pip=True, clear=True).create(environment_root)
    python = (
        environment_root / "Scripts" / "python.exe"
        if sys.platform == "win32"
        else environment_root / "bin" / "python"
    )
    empty_pythonpath = tmp_path / "empty-async-pythonpath"
    empty_pythonpath.mkdir()
    install = run_performance_command(
        (str(python), "-I", "-m", "pip", "install", "--no-deps", str(wheel_path)),
        project_root=tmp_path,
        payload_root=empty_pythonpath,
        mode="baseline",
    )
    assert install.succeeded, install.stderr
    smoke = run_performance_command(
        (
            str(python),
            "-I",
            "-c",
            "\n".join(
                (
                    "import asyncio",
                    "import inspect",
                    "from typed_region_project.async_runner import ProtocolRunner",
                    "runner = ProtocolRunner()",
                    "stream = runner.exchange(1)",
                    "assert next(stream) == 3",
                    "assert stream.send(5) == 7",
                    "stream.close()",
                    "assert runner.sync_finalized == 1",
                    "assert inspect.isasyncgenfunction(ProtocolRunner.async_exchange)",
                    "assert hasattr(ProtocolRunner.exchange, '__atoll_compiled_target__')",
                    "assert hasattr(ProtocolRunner.async_exchange, '__atoll_compiled_target__')",
                    "async def exercise():",
                    "    async_stream = runner.async_exchange(1)",
                    "    assert await anext(async_stream) == 3",
                    "    assert await async_stream.asend(5) == 7",
                    "    await async_stream.aclose()",
                    "asyncio.run(exercise())",
                )
            ),
        ),
        project_root=tmp_path,
        payload_root=empty_pythonpath,
        mode="compiled",
    )
    assert smoke.succeeded, smoke.stderr
