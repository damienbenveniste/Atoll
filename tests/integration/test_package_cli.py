"""Integration tests for source-clean compile artifact builds."""

from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import cast

import pytest

from atoll.cli import main
from atoll.commands import package as package_command
from atoll.commands.package import PackageCommandResult, PackageOptions, PackagePreflightFailure
from atoll.models import Blocker, CompileAttempt, ModuleId, ModuleScan
from atoll.report import COMPILE_REPORT_SCHEMA_VERSION
from atoll.runtime.package_verify import PackageVerificationResult
from atoll.runtime.performance import (
    BenchmarkGateConfig,
    BenchmarkGateResult,
    CommandRunEvidence,
    run_performance_command,
)

FIXTURE_ROOT = Path("tests/fixtures/simple_project")
EXIT_USAGE = 2
RANKING_SYMBOL_COUNT = 3
GATE_FAILURE_CODE = 7
PROFILE_BENCHMARK_ITERATIONS = 100_000
PROFILE_REPORT_SCHEMA_VERSION = COMPILE_REPORT_SCHEMA_VERSION
MINIMUM_PROFILE_SAMPLES = 100
CLASS_DEPENDENCY_COMPILED_SYMBOLS = 2


def test_compile_builds_wheel_without_source_edits_or_kept_install_tree(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`atoll compile` keeps the checkout and output directory clean by default."""
    project_root = tmp_path / "simple_project"
    output_dir = project_root / ".atoll" / "dist"
    shutil.copytree(FIXTURE_ROOT, project_root)
    source_path = project_root / "src" / "app" / "ranking.py"
    original_source = source_path.read_text(encoding="utf-8")

    exit_code = main(["compile", "app.ranking", "--root", str(project_root)])

    captured = capsys.readouterr()
    wheel_path = next(output_dir.glob("*.whl"))
    report_json_path = project_root / ".atoll" / "compile-report.json"
    report_markdown_path = project_root / ".atoll" / "compile-report.md"
    report = json.loads(report_json_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert source_path.read_text(encoding="utf-8") == original_source
    assert "# BEGIN ATOLL MANAGED" not in original_source
    assert not (output_dir / "build").exists()
    assert not (output_dir / "install").exists()
    assert report_markdown_path.exists()
    assert not (project_root / ".atoll" / "compilation-report.json").exists()
    assert report["mode"] == "source-clean"
    assert report["wheel_path"] == f".atoll/dist/{wheel_path.name}"
    assert report["summary"]["islands"] == 0
    assert report["summary"]["compiled_regions"] == 1
    assert report["summary"]["symbols"] == RANKING_SYMBOL_COUNT
    assert report["summary"]["native_ready_symbols"] == 0
    assert report["summary"]["native_rejected_symbols"] == 0
    assert report["native_readiness"] == []
    assert {binding["source"] for binding in report["compiled_regions"][0]["bindings"]} == {
        "app.ranking::normalize_features",
        "app.ranking::score_user",
        "app.ranking::rank_candidates",
    }
    assert report["build"]["cache_status"] == "miss"
    assert report["performance"]["status"] == "unbenchmarked"
    assert report["performance"]["speedup"] is None
    assert report["execution_plans"] == []
    assert report["applied_execution_plans"] == []
    assert report["execution_plan_trials"] == []
    assert any(timing["name"] == "mypycify" for timing in report["build"]["phase_timings"])
    assert any(timing["name"] == "build_ext" for timing in report["build"]["phase_timings"])
    assert report["cleanup"]["removed"] == [".atoll/dist/build", ".atoll/dist/install"]
    assert report["cleanup"]["kept"] == []
    markdown_report = report_markdown_path.read_text(encoding="utf-8")
    assert "normal PEP 517 wheel" in markdown_report
    assert "unbenchmarked" in markdown_report
    assert "Install tree:" not in captured.out
    assert "Compile reports:" in captured.out
    assert "Atoll compile [" in captured.err
    assert "discovering project" in captured.err
    assert "compiling typed region variant" in captured.err
    assert "cleaned temporary outputs" in captured.err
    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())
        metadata_path = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = wheel.read(metadata_path).decode("utf-8")
    assert "app/ranking.py" in names
    assert "app/data/schema.json" in names
    assert "app/py.typed" in names
    assert any(name.endswith(".dist-info/entry_points.txt") for name in names)
    assert "Summary: Fixture metadata preserved by Atoll wheel overlays." in metadata
    assert any(name.startswith(".atoll/artifacts/") and name.endswith(".so") for name in names)
    artifact_names = sorted(
        name for name in names if name.startswith(".atoll/artifacts/") and name.endswith(".so")
    )
    assert report["summary"]["artifacts"] == len(artifact_names)
    assert sorted(report["build"]["artifacts"]) == artifact_names


def test_compile_uses_cache_on_unchanged_second_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A repeated source-clean compile restores unchanged artifacts from cache."""
    project_root = tmp_path / "simple_project"
    output_dir = project_root / ".atoll" / "dist"
    shutil.copytree(FIXTURE_ROOT, project_root)

    first_exit = main(["compile", "app.ranking", "--root", str(project_root)])
    first_report = json.loads((project_root / ".atoll" / "compile-report.json").read_text())
    first_capture = capsys.readouterr()
    second_exit = main(["compile", "app.ranking", "--root", str(project_root)])
    second_capture = capsys.readouterr()

    wheel_path = next(output_dir.glob("*.whl"))
    second_report = json.loads((project_root / ".atoll" / "compile-report.json").read_text())
    assert first_exit == 0
    assert second_exit == 0
    assert first_report["build"]["cache_status"] == "miss"
    assert second_report["build"]["cache_status"] == "hit"
    assert "compile cache miss" in first_capture.err
    assert "compile cache hit" in second_capture.err
    assert [
        timing["name"]
        for timing in second_report["build"]["phase_timings"]
        if timing["name"].startswith("cache_")
    ] == [
        "cache_lookup",
        "cache_restore",
    ]
    assert not (output_dir / "build").exists()
    assert not (output_dir / "install").exists()
    with zipfile.ZipFile(wheel_path) as wheel:
        assert any(name.startswith(".atoll/artifacts/") for name in wheel.namelist())


def test_compile_reports_static_async_execution_plan_candidates(tmp_path: Path) -> None:
    """Unbenchmarked scheduler sites are reported without activating a plan."""
    project_root = tmp_path / "execution_plan_project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    module_path = project_root / "src" / "app" / "scheduler.py"
    module_path.write_text(
        """import asyncio

async def _produce(queue: asyncio.Queue[int]) -> None:
    await queue.put(1)

async def _consume(queue: asyncio.Queue[int]) -> int:
    return await queue.get()

async def run() -> int:
    queue: asyncio.Queue[int] = asyncio.Queue(maxsize=1)
    async with asyncio.TaskGroup() as group:
        group.create_task(_produce(queue))
        consumer = group.create_task(_consume(queue))
    return consumer.result()
""",
        encoding="utf-8",
    )
    original = module_path.read_text(encoding="utf-8")

    main(["compile", "app.scheduler", "--root", str(project_root)])

    report = json.loads((project_root / ".atoll" / "compile-report.json").read_text())
    assert report["version"] == PROFILE_REPORT_SCHEMA_VERSION
    assert module_path.read_text(encoding="utf-8") == original
    assert report["summary"]["execution_plans"] == 1
    assert report["summary"]["execution_selected_plans"] == 0
    assert report["summary"]["execution_applied_plans"] == 0
    assert report["execution_plans"][0]["owner"] == "app.scheduler::run"
    assert report["execution_plans"][0]["status"] == "rejected"
    assert report["execution_plans"][0]["rejections"][0]["code"] == "low-hotness"
    assert report["applied_execution_plans"] == []
    assert report["execution_plan_trials"] == []


def test_compile_runs_semantic_gate_without_flat_checkout_shadowing(
    tmp_path: Path,
) -> None:
    """Configured commands import the staged payload from a source-stripped project mirror."""
    project_root = tmp_path / "flat_project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    shutil.move(project_root / "src" / "app", project_root / "app")
    (project_root / "src").rmdir()
    pyproject = project_root / "pyproject.toml"
    config = pyproject.read_text(encoding="utf-8").replace('where = ["src"]', 'where = ["."]')
    probe = (
        "import app.ranking as module; "
        "assert '/.atoll/dist/install/' in module.__file__.replace('\\\\', '/')"
    )
    pyproject.write_text(
        config
        + "\n".join(
            (
                "",
                "[tool.atoll.compile]",
                f'test_command = [{json.dumps(sys.executable)}, "-c", {json.dumps(probe)}]',
                "",
            )
        ),
        encoding="utf-8",
    )

    exit_code = main(["compile", "app.ranking", "--root", str(project_root)])

    report = json.loads((project_root / ".atoll" / "compile-report.json").read_text())
    assert exit_code == 0
    assert report["test_results"][0]["mode"] == "compiled"
    assert report["test_results"][0]["returncode"] == 0
    assert report["performance"]["status"] == "unbenchmarked"


def test_compile_profiles_configured_benchmark_before_hot_region_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real workload profiles hot symbols and packages only profitable trial results."""
    project_root = tmp_path / "profiled_project"
    monkeypatch.setattr(package_command, "_CANDIDATE_MINIMUM_SPEEDUP", 1.01)
    shutil.copytree(FIXTURE_ROOT, project_root)
    benchmark_path = project_root / "benchmark.py"
    benchmark_path.write_text(
        """from app.ranking import rank_candidates
from app.types import Event, User

users = [User(float(index)) for index in range(8)]
events = [Event(str(index)) for index in range(4)]
for _ in range(ITERATIONS):
    rank_candidates(users, events)
""".replace("ITERATIONS", str(PROFILE_BENCHMARK_ITERATIONS)),
        encoding="utf-8",
    )
    pyproject = project_root / "pyproject.toml"
    semantic_probe = "import app.ranking; assert callable(app.ranking.rank_candidates)"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8")
        + "\n".join(
            (
                "",
                "[tool.atoll.compile]",
                (
                    f'test_command = [{json.dumps(sys.executable)}, "-c", '
                    f"{json.dumps(semantic_probe)}]"
                ),
                (f'benchmark_command = [{json.dumps(sys.executable)}, "benchmark.py"]'),
                "benchmark_warmups = 0",
                "benchmark_samples = 1",
                "minimum_speedup = 1.01",
                "",
            )
        ),
        encoding="utf-8",
    )

    def accept_measured_gate(
        config: BenchmarkGateConfig,
        **kwargs: object,
    ) -> BenchmarkGateResult:
        project = cast(Path, kwargs["project_root"])
        baseline = cast(Path, kwargs["baseline_payload_root"])
        compiled = cast(Path, kwargs["compiled_payload_root"])
        return BenchmarkGateResult(
            status="passed",
            reason="stable fixture benchmark passed",
            minimum_speedup=config.minimum_speedup,
            baseline_median_seconds=0.5,
            compiled_median_seconds=0.4,
            speedup=1.25,
            warmups=(),
            samples=(
                CommandRunEvidence(
                    config.command or (),
                    project,
                    baseline,
                    "baseline",
                    0,
                    "",
                    "",
                    0.5,
                ),
                CommandRunEvidence(
                    config.command or (),
                    project,
                    compiled,
                    "compiled",
                    0,
                    "",
                    "",
                    0.4,
                ),
            ),
        )

    monkeypatch.setattr(package_command, "run_benchmark_gate", accept_measured_gate)

    exit_code = main(["compile", "app.ranking", "--root", str(project_root)])

    report = json.loads((project_root / ".atoll" / "compile-report.json").read_text())
    profile = report["profile"]
    selected = set(profile["selected_symbols"])
    compiled = {
        binding["source"] for region in report["compiled_regions"] for binding in region["bindings"]
    }
    trials = report["candidate_trials"]
    trial_symbols = {symbol for trial in trials for symbol in trial["symbols"]}
    accepted_symbols = {
        symbol for trial in trials if trial["status"] == "accepted" for symbol in trial["symbols"]
    }
    assert exit_code == 0
    assert report["version"] == PROFILE_REPORT_SCHEMA_VERSION
    assert report["success"] is True
    assert profile["status"] == "profiled"
    assert profile["launch_kind"] == "script"
    assert profile["total_samples"] >= MINIMUM_PROFILE_SAMPLES
    assert profile["mapped_project_samples"] > 0
    assert profile["selected_hot_coverage"] > 0
    assert [run["pass_kind"] for run in profile["child_passes"]] == ["sampling", "types"]
    assert selected
    assert trial_symbols == selected
    assert compiled == accepted_symbols
    assert report["summary"]["symbols"] == len(accepted_symbols)
    assert report["summary"]["profile_accepted_hot_coverage"] == trials[-1]["accepted_hot_coverage"]
    assert report["performance"]["status"] == "passed"
    assert {timing["name"] for timing in report["build"]["phase_timings"]}.issuperset(
        {"profile_sampling", "profile_types", "candidate_semantic_test", "candidate_benchmark"}
    )
    assert not (project_root / ".atoll" / "dist" / "build").exists()
    assert not (project_root / ".atoll" / "dist" / "install").exists()


def test_compile_keeps_same_module_class_dependency_as_runtime_boundary(
    tmp_path: Path,
) -> None:
    """A compiled caller resolves its interpreted class dependency at runtime."""
    project_root = tmp_path / "class_dependency_project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    module_path = project_root / "src" / "app" / "class_dependency.py"
    module_path.write_text(
        """class Payload:
    def __init__(self, value: int) -> None:
        self.value = value

def make_payload(value: int) -> Payload:
    return Payload(value)

def add_one(value: int) -> int:
    return value + 1
""",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "compile",
            "app.class_dependency",
            "--root",
            str(project_root),
            "--keep-install-tree",
        ]
    )

    install_root = project_root / ".atoll" / "dist" / "install"
    report = json.loads((project_root / ".atoll" / "compile-report.json").read_text())
    probe = run_performance_command(
        (
            sys.executable,
            "-c",
            (
                "import app.class_dependency as module; "
                "assert module.add_one(3) == 4; "
                "assert hasattr(module.add_one, '__atoll_compiled_target__'); "
                "assert module.make_payload(5).value == 5; "
                "assert hasattr(module.make_payload, '__atoll_compiled_target__')"
            ),
        ),
        project_root=project_root,
        payload_root=install_root,
        mode="compiled",
    )

    assert exit_code == 0
    assert probe.succeeded is True
    assert report["summary"]["symbols"] == CLASS_DEPENDENCY_COMPILED_SYMBOLS
    compiled = {
        binding["source"] for region in report["compiled_regions"] for binding in region["bindings"]
    }
    assert compiled == {
        "app.class_dependency::add_one",
        "app.class_dependency::make_payload",
    }


def test_compile_rejects_wheel_after_real_semantic_gate_failure(tmp_path: Path) -> None:
    """A nonzero configured command reports failure and removes candidate scratch."""
    project_root = tmp_path / "failed_gate_project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    pyproject = project_root / "pyproject.toml"
    command = "raise SystemExit(7)"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8")
        + "\n".join(
            (
                "",
                "[tool.atoll.compile]",
                f'test_command = [{json.dumps(sys.executable)}, "-c", {json.dumps(command)}]',
                "",
            )
        ),
        encoding="utf-8",
    )

    exit_code = main(["compile", "app.ranking", "--root", str(project_root)])

    output_dir = project_root / ".atoll" / "dist"
    report = json.loads((project_root / ".atoll" / "compile-report.json").read_text())
    assert exit_code == 1
    assert report["success"] is False
    assert report["test_results"][0]["returncode"] == GATE_FAILURE_CODE
    assert report["performance"]["status"] == "invalid"
    assert report["cleanup"]["removed"] == [".atoll/dist/build", ".atoll/dist/install"]
    assert report["cleanup"]["kept"] == []
    assert not tuple(output_dir.glob("*.whl"))
    assert not (output_dir / "build").exists()
    assert not (output_dir / "install").exists()


def test_compile_reports_backend_wheel_omission_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A backend wheel that omits the selected module becomes a diagnostic failure."""
    project_root = tmp_path / "omitted_project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    pyproject = project_root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(
            'where = ["src"]',
            'where = ["wheel_src"]',
        ),
        encoding="utf-8",
    )
    (project_root / "wheel_src").mkdir()

    exit_code = main(["compile", "app.ranking", "--root", str(project_root)])

    captured = capsys.readouterr()
    report = json.loads((project_root / ".atoll" / "compile-report.json").read_text())
    assert exit_code == 1
    assert report["success"] is False
    assert "target PEP 517 wheel omitted" in report["build"]["stderr"]
    assert "Traceback" not in captured.err
    assert not tuple((project_root / ".atoll" / "dist").glob("*.whl"))
    assert not (project_root / ".atoll" / "dist" / "build").exists()
    assert not (project_root / ".atoll" / "dist" / "install").exists()


def test_compile_skips_automatic_modules_omitted_by_backend_wheel(tmp_path: Path) -> None:
    """Whole-project compile follows the PEP 517 wheel's distributable module boundary."""
    project_root = tmp_path / "mixed_flat_project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    shutil.move(project_root / "src" / "app", project_root / "app")
    (project_root / "src").rmdir()
    tools = project_root / "tools"
    tools.mkdir()
    (tools / "__init__.py").write_text("", encoding="utf-8")
    (tools / "helper.py").write_text(
        "def increment(value: int) -> int:\n    return value + 1\n",
        encoding="utf-8",
    )
    pyproject = project_root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(
            'where = ["src"]',
            'where = ["."]\ninclude = ["app*"]',
        ),
        encoding="utf-8",
    )

    exit_code = main(["compile", "--root", str(project_root)])

    wheels = tuple((project_root / ".atoll" / "dist").glob("*.whl"))
    assert exit_code == 0
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        names = set(archive.namelist())
    assert "app/ranking.py" in names
    assert "tools/helper.py" not in names


def test_compile_rejects_invalid_compile_config_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Configuration validation errors use a stable CLI exit instead of escaping."""
    project_root = tmp_path / "invalid_config_project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    pyproject = project_root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8")
        + '\n[tool.atoll.compile]\nbenchmark_command = ["python", "bench.py"]\n',
        encoding="utf-8",
    )

    exit_code = main(["compile", "--root", str(project_root)])

    captured = capsys.readouterr()
    assert exit_code == EXIT_USAGE
    assert "Atoll compile configuration error" in captured.err
    assert "requires test_command" in captured.err
    assert "Traceback" not in captured.err


def test_compile_can_keep_install_tree_for_debugging(
    tmp_path: Path,
) -> None:
    """`--keep-install-tree` preserves the generated tree for direct routing checks."""
    project_root = tmp_path / "simple_project"
    output_dir = project_root / ".atoll" / "dist"
    shutil.copytree(FIXTURE_ROOT, project_root)
    source_path = project_root / "src" / "app" / "ranking.py"
    original_source = source_path.read_text(encoding="utf-8")

    exit_code = main(
        [
            "compile",
            "app.ranking",
            "--root",
            str(project_root),
            "--keep-install-tree",
        ]
    )

    install_root = output_dir / "install"
    install_source = install_root / "app" / "ranking.py"
    wheel_path = next(output_dir.glob("*.whl"))
    report = json.loads((project_root / ".atoll" / "compile-report.json").read_text())
    assert exit_code == 0
    assert source_path.read_text(encoding="utf-8") == original_source
    assert "# BEGIN ATOLL MANAGED" not in original_source
    assert "# BEGIN ATOLL TYPED REGIONS: app.ranking" in install_source.read_text(encoding="utf-8")
    assert not (output_dir / "build").exists()
    assert not (install_root / "app" / "_atoll_app_ranking.py").exists()
    assert tuple((install_root / ".atoll" / "artifacts").rglob("*.so"))
    assert report["cleanup"]["removed"] == [".atoll/dist/build"]
    assert report["cleanup"]["kept"] == [".atoll/dist/install"]

    probe = run_performance_command(
        (
            sys.executable,
            "-c",
            (
                "import app.ranking as ranking; "
                "assert ranking.normalize_features([1.0, 3.0]) == [0.25, 0.75]; "
                "assert hasattr(ranking.normalize_features, '__atoll_compiled_target__')"
            ),
        ),
        project_root=project_root,
        payload_root=install_root,
        mode="compiled",
    )
    assert probe.succeeded is True
    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())
    assert "app/ranking.py" in names
    assert "app/types.py" in names
    assert any(name.startswith(".atoll/artifacts/") and name.endswith(".so") for name in names)
    assert any(name.endswith(".dist-info/METADATA") for name in names)
    assert any(name.endswith(".dist-info/WHEEL") for name in names)
    assert any(name.endswith(".dist-info/RECORD") for name in names)


def test_package_command_reports_no_candidates(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Package mode reports when no modules contain supported typed regions."""
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "__init__.py").write_text("", encoding="utf-8")

    exit_code = main(["package", "--root", str(tmp_path), "--output", str(tmp_path / "out")])

    captured = capsys.readouterr()
    report = json.loads((tmp_path / ".atoll" / "compile-report.json").read_text())
    assert exit_code == 1
    assert "scan found no backend-supported typed regions" in captured.out
    assert report["success"] is False
    assert report["build"]["stderr"] == "scan found no backend-supported typed regions"


def test_compile_reports_no_candidates_from_source_clean_default(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Source-clean compile reports when no modules contain supported typed regions."""
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "__init__.py").write_text("", encoding="utf-8")

    exit_code = main(["compile", "--root", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "scan found no backend-supported typed regions" in captured.out


def test_compile_reports_preflight_skipped_modules(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Source-clean compile reports modules skipped before mypyc runs."""
    module_path = tmp_path / "src" / "pkg" / "blocked.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("", encoding="utf-8")
    blocker = Blocker(
        severity="hard",
        code="MYPYC_UNSUPPORTED_TYPEVAR",
        message="TypeVar keyword(s) default are rejected by mypyc",
        lineno=4,
        symbol=None,
    )
    scan = ModuleScan(
        module=ModuleId(name="pkg.blocked", path=module_path),
        imports=(),
        constants=(),
        symbols=(),
        blockers=(blocker,),
        top_level_statement_lines=(),
    )

    def fake_execute_package(*_args: object, **_kwargs: object) -> PackageCommandResult:
        return PackageCommandResult(
            success=True,
            project_root=tmp_path,
            output_dir=tmp_path / "out",
            install_root=tmp_path / "out" / "install",
            wheel_path=tmp_path / "out" / "pkg-0-atoll.whl",
            islands=(),
            build=CompileAttempt(
                success=True,
                command=("mypyc",),
                stdout="",
                stderr="",
                artifact_paths=(),
                duration_seconds=0.0,
            ),
            preflight_skipped=(PackagePreflightFailure(scan=scan, blockers=(blocker,)),),
        )

    monkeypatch.setattr("atoll.cli.execute_package", fake_execute_package)

    exit_code = main(["compile", "--root", str(tmp_path)])

    captured = capsys.readouterr()
    report = json.loads((tmp_path / ".atoll" / "compile-report.json").read_text())
    assert exit_code == 0
    assert "Skipped 1 module(s) with known mypyc typing blockers." in captured.out
    assert "- pkg.blocked: line 4: TypeVar keyword(s) default are rejected by mypyc" in captured.out
    assert report["summary"]["preflight_blockers"] == 1
    assert report["preflight_blockers"][0]["module"] == "pkg.blocked"


def test_compile_report_uses_successful_final_package_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed safety probes stay diagnostic after final package promotion succeeds."""
    output_dir = tmp_path / ".atoll" / "dist"
    wheel_path = output_dir / "pkg-0+atoll.whl"
    wheel_path.parent.mkdir(parents=True)
    wheel_path.write_bytes(b"wheel")
    failed_probe = PackageVerificationResult(
        stage="payload",
        target=output_dir / "install",
        command=("python", "verify"),
        success=False,
        exit_code=1,
        stdout="",
        stderr="candidate rejected",
        duration_seconds=0.1,
    )
    final_verification = PackageVerificationResult(
        stage="wheel",
        target=wheel_path,
        command=("python", "verify"),
        success=True,
        exit_code=0,
        stdout="",
        stderr="",
        duration_seconds=0.1,
    )

    def fake_execute_package(*_args: object, **_kwargs: object) -> PackageCommandResult:
        return PackageCommandResult(
            success=True,
            project_root=tmp_path,
            output_dir=output_dir,
            install_root=output_dir / "install",
            wheel_path=wheel_path,
            islands=(),
            build=CompileAttempt(
                success=True,
                command=("mypyc",),
                stdout="",
                stderr="",
                artifact_paths=(),
                duration_seconds=0.2,
            ),
            verification_steps=(failed_probe, final_verification),
        )

    monkeypatch.setattr("atoll.cli.execute_package", fake_execute_package)

    exit_code = main(["compile", "--root", str(tmp_path)])
    report = json.loads((tmp_path / ".atoll" / "compile-report.json").read_text())

    assert exit_code == 0
    assert report["success"] is True
    assert report["summary"]["subprocess_verification_failures"] == 1


def test_compile_keeps_unresolved_generic_functions_interpreted(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Unresolved public generics remain fallback without generated type erasure."""
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        "\n".join(["[project]", 'name = "pkg"', 'version = "0.1.0"', ""]),
        encoding="utf-8",
    )
    output_dir = tmp_path / ".atoll" / "dist"
    output_dir.mkdir(parents=True)
    (output_dir / "pkg-0.1.0-cp311-cp311-macosx_11_0_arm64.whl").write_bytes(b"stale")
    (tmp_path / "src" / "pkg" / "mod.py").write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "from typing_extensions import TypeVar",
                "",
                "T = TypeVar('T', default=str)",
                "",
                "def helper(value: T) -> T:",
                "    return value",
                "",
                "def candidate(value: T) -> T:",
                "    adjusted = helper(value)",
                "    return adjusted",
                "",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["compile", "--root", str(tmp_path)])

    captured = capsys.readouterr()
    report = json.loads((tmp_path / ".atoll" / "compile-report.json").read_text())
    assert exit_code == 1
    assert "scan found no backend-supported typed regions" in captured.out
    assert report["success"] is False
    assert report["summary"]["preflight_blockers"] == 0
    assert report["summary"]["symbols"] == 0
    assert report["summary"]["typed_regions"] == 1
    assert report["typed_regions"][0]["decisions"][0]["action"] == "fallback"
    assert report["summary"]["native_rejected_symbols"] == 0
    assert report["native_readiness"] == []
    assert report["build"]["command"] == []
    assert not tuple(output_dir.glob("*.whl"))


def test_compile_in_place_rejects_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`compile --output` only applies to the source-clean artifact mode."""
    exit_code = main(
        ["compile", "--root", str(tmp_path), "--in-place", "--output", str(tmp_path / "out")]
    )

    captured = capsys.readouterr()
    assert exit_code == EXIT_USAGE
    assert "--output cannot be used with --in-place" in captured.out


def test_compile_in_place_rejects_keep_install_tree(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--keep-install-tree` only applies to source-clean artifact mode."""
    exit_code = main(["compile", "--root", str(tmp_path), "--in-place", "--keep-install-tree"])

    captured = capsys.readouterr()
    assert exit_code == EXIT_USAGE
    assert "--keep-install-tree cannot be used with --in-place" in captured.out


def test_compile_in_place_rejects_apply_source(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Source patch application is incompatible with the legacy in-place workflow."""
    exit_code = main(["compile", "--root", str(tmp_path), "--in-place", "--apply-source"])

    captured = capsys.readouterr()
    assert exit_code == EXIT_USAGE
    assert "--apply-source cannot be used with --in-place" in captured.out


def test_compile_forwards_apply_source_to_source_clean_packaging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public compile flag reaches source-clean package orchestration unchanged."""
    observed: list[bool] = []

    def fake_execute(options: PackageOptions) -> PackageCommandResult:
        observed.append(options.apply_source)
        return PackageCommandResult(
            success=False,
            project_root=tmp_path,
            output_dir=tmp_path / ".atoll" / "dist",
            install_root=tmp_path / ".atoll" / "dist" / "install",
            wheel_path=None,
            islands=(),
            build=CompileAttempt(
                success=False,
                command=(),
                stdout="",
                stderr="fixture stopped after option forwarding",
                artifact_paths=(),
                duration_seconds=0.0,
            ),
            error="fixture stopped after option forwarding",
        )

    monkeypatch.setattr("atoll.cli.execute_package", fake_execute)

    exit_code = main(["compile", "--root", str(tmp_path), "--apply-source"])

    assert exit_code == 1
    assert observed == [True]


def test_compile_source_clean_default_rejects_test_gate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The semantic test gate remains scoped to in-place compile mode."""
    exit_code = main(["compile", "--root", str(tmp_path), "--test", "pytest"])

    captured = capsys.readouterr()
    assert exit_code == EXIT_USAGE
    assert "--test requires --in-place" in captured.out
