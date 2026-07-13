"""Network-free acceptance coverage for the external-project case lifecycle."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from scripts.benchmark_corpus.lifecycle import (
    LifecycleOptions,
    dependency_bootstrap_commands,
    run_case,
    tools_venv_command,
    venv_python,
)
from scripts.benchmark_corpus.manifest import load_manifest
from scripts.benchmark_corpus.process import detect_sandbox

ATOLL_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = ATOLL_ROOT / "tests" / "fixtures" / "simple_project"
ADAPTER_ROOT = ATOLL_ROOT / "tests" / "fixtures"
BACKEND_PATH = ATOLL_ROOT / "tests" / "fixtures" / "corpus_backend.py"


def test_venv_python_preserves_environment_symlink_path(tmp_path: Path) -> None:
    """The case interpreter path must not resolve back to its host Python."""
    environment = tmp_path / "environment"
    binary = environment / "bin"
    binary.mkdir(parents=True)
    python = binary / "python"
    try:
        python.symlink_to(sys.executable)
    except OSError:
        pytest.skip("symbolic links are unavailable")

    observed = venv_python(environment)

    assert observed == python.absolute()
    assert os.fspath(observed) != os.fspath(python.resolve())


def test_isolated_bootstrap_uses_bundled_pip_and_hash_verified_locks(
    tmp_path: Path,
) -> None:
    """Network bootstrap never executes an unpinned downloaded pip package."""
    environment = tmp_path / "tools"
    tools_python = environment / "bin" / "python"
    lock = tmp_path / "lock.txt"
    wheelhouse = tmp_path / "wheelhouse"

    create = tools_venv_command("/uv", "3.12", environment)
    bootstrap = dependency_bootstrap_commands(
        "/uv",
        tools_python,
        lock,
        wheelhouse,
    )

    assert create == ("/uv", "venv", "--python", "3.12", str(environment))
    assert "--seed" not in create
    assert bootstrap.ensure_pip[-3:] == ("-m", "ensurepip", "--upgrade")
    assert bootstrap.download[:4] == (str(tools_python), "-m", "pip", "download")
    assert "--require-hashes" in bootstrap.download
    assert bootstrap.download[-1] == str(wheelhouse)
    assert bootstrap.sync[:3] == ("/uv", "pip", "sync")
    assert "--require-hashes" in bootstrap.sync
    assert "--offline" in bootstrap.sync


def test_local_pinned_lifecycle_is_source_clean_and_warm_cached(tmp_path: Path) -> None:
    """A moving local branch cannot affect the pinned, source-clean wheel trial."""
    remote, revision = _local_remote(tmp_path)
    manifest = load_manifest(_write_manifest(tmp_path, revision))

    summary = run_case(
        manifest,
        "simple-project",
        LifecycleOptions(
            atoll_root=ATOLL_ROOT,
            workspace_root=tmp_path / "workspaces",
            evidence_root=tmp_path / "evidence",
            tier="compatibility",
            platform="ubuntu-24.04",
            allow_unsandboxed=True,
            repository_mirror=remote,
            adapter_root=ADAPTER_ROOT,
            sandbox_override="unsandboxed",
            environment_mode="current-test",
        ),
    )

    assert summary.result.status in {"compiled-unbenchmarked", "supported-no-op"}
    assert summary.result.revision == revision
    assert summary.result.source_unchanged is True
    assert summary.result.baseline_oracle_digest == summary.result.compiled_oracle_digest
    assert summary.result.warm_compiler_invocations == 0
    assert not (tmp_path / "workspaces" / "simple-project-compatibility-ubuntu-24.04").exists()
    assert not tuple((tmp_path / "evidence").glob("*.whl"))
    payload = json.loads(summary.json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["source_unchanged"] is True
    assert (tmp_path / "evidence" / "cold.compile-report.json").is_file()
    assert (tmp_path / "evidence" / "warm.compile-report.json").is_file()
    assert (tmp_path / "evidence" / "compile-policy.patch").is_file()
    assert (tmp_path / "evidence" / "compiler-probe.log").is_file()


def test_baseline_build_failure_is_attributed_upstream(tmp_path: Path) -> None:
    """A project that cannot build normally is not reported as an Atoll failure."""
    _remote, revision = _local_remote(tmp_path, broken_backend=True)
    manifest = load_manifest(_write_manifest(tmp_path, revision))

    summary = run_case(manifest, "simple-project", _options(tmp_path))

    assert summary.result.status == "upstream-broken"
    assert "normal PEP 517 baseline wheel build" in summary.result.diagnostics[0]
    assert summary.result.cold_report_path is None


def test_compiled_wheel_oracle_corruption_is_a_regression(tmp_path: Path) -> None:
    """A final wheel with changed canonical behavior fails compatibility."""
    _remote, revision = _local_remote(tmp_path)
    manifest = load_manifest(
        _write_manifest(tmp_path, revision, oracle_arguments=("--corrupt-compiled",))
    )

    summary = run_case(manifest, "simple-project", _options(tmp_path))

    assert summary.result.status == "compatibility-regression"
    assert summary.result.baseline_oracle_digest != summary.result.compiled_oracle_digest


def test_no_native_candidates_remain_a_compatible_no_op(tmp_path: Path) -> None:
    """A source-clean wheel with no accepted native binding remains supported."""
    _remote, revision = _local_remote(tmp_path, no_op=True)
    manifest = load_manifest(_write_manifest(tmp_path, revision))

    summary = run_case(manifest, "simple-project", _options(tmp_path))

    assert summary.result.status == "supported-no-op"
    assert summary.result.source_unchanged is True


def test_upstream_tracked_mutation_is_attributed_before_compile(tmp_path: Path) -> None:
    """A mutating build backend cannot redefine the source Atoll receives."""
    _remote, revision = _local_remote(tmp_path, mutate_tracked=True)
    manifest = load_manifest(_write_manifest(tmp_path, revision))

    summary = run_case(manifest, "simple-project", _options(tmp_path))

    assert summary.result.status == "upstream-broken"
    assert "changed tracked project files" in summary.result.diagnostics[0]
    assert summary.result.cold_report_path is None


def test_upstream_generated_files_are_removed_before_compile(tmp_path: Path) -> None:
    """Untracked build output cannot become an accidental compiler input."""
    _remote, revision = _local_remote(tmp_path, generate_untracked=True)
    manifest = load_manifest(_write_manifest(tmp_path, revision))
    options = replace(_options(tmp_path), keep_workspace=True)

    summary = run_case(manifest, "simple-project", options)

    workspace = tmp_path / "workspaces" / "simple-project-compatibility-ubuntu-24.04"
    generated = workspace / "checkout" / "src" / "app" / "generated_by_build.py"
    assert summary.result.status == "compiled-unbenchmarked"
    assert generated.exists() is False


def test_lifecycle_runs_inside_available_platform_sandbox(tmp_path: Path) -> None:
    """The complete local lifecycle honors the advertised sandbox boundary."""
    try:
        sandbox = detect_sandbox(allow_unsandboxed=False)
    except RuntimeError:
        pytest.skip("supported platform sandbox is not installed")
    _remote, revision = _local_remote(tmp_path)
    manifest = load_manifest(_write_manifest(tmp_path, revision))
    options = replace(
        _options(tmp_path),
        allow_unsandboxed=False,
        sandbox_override=sandbox,
    )

    summary = run_case(manifest, "simple-project", options)

    assert summary.result.status == "compiled-unbenchmarked"
    assert summary.result.source_unchanged is True


def _options(tmp_path: Path) -> LifecycleOptions:
    return LifecycleOptions(
        atoll_root=ATOLL_ROOT,
        workspace_root=tmp_path / "workspaces",
        evidence_root=tmp_path / "evidence",
        tier="compatibility",
        platform="ubuntu-24.04",
        allow_unsandboxed=True,
        repository_mirror=tmp_path / "upstream",
        adapter_root=ADAPTER_ROOT,
        sandbox_override="unsandboxed",
        environment_mode="current-test",
    )


def _local_remote(
    tmp_path: Path,
    *,
    broken_backend: bool = False,
    no_op: bool = False,
    mutate_tracked: bool = False,
    generate_untracked: bool = False,
) -> tuple[Path, str]:
    checkout = tmp_path / "upstream"
    shutil.copytree(
        FIXTURE_ROOT,
        checkout,
        ignore=shutil.ignore_patterns(".atoll", "__pycache__", ".pytest_cache", "*.egg-info"),
    )
    shutil.copy2(BACKEND_PATH, checkout / "corpus_backend.py")
    if broken_backend:
        (checkout / "corpus_backend.py").write_text(
            "def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):\n"
            "    raise RuntimeError('fixture baseline failure')\n",
            encoding="utf-8",
        )
    if mutate_tracked or generate_untracked:
        backend = checkout / "corpus_backend.py"
        source = backend.read_text(encoding="utf-8")
        injected: list[str] = []
        if mutate_tracked:
            injected.append(
                "    (root / 'src' / 'app' / 'ranking.py').write_text("
                "(root / 'src' / 'app' / 'ranking.py').read_text() + '\\n')"
            )
        if generate_untracked:
            injected.append(
                "    (root / 'src' / 'app' / 'generated_by_build.py').write_text('VALUE = 1\\n')"
            )
        source = source.replace(
            "    root = Path.cwd()", "\n".join(("    root = Path.cwd()", *injected))
        )
        backend.write_text(source, encoding="utf-8")
    if no_op:
        ranking = checkout / "src" / "app" / "ranking.py"
        source = ranking.read_text(encoding="utf-8")
        source = source.replace(
            "from __future__ import annotations\n",
            "from __future__ import annotations\n\nimport inspect\n",
        )
        for statement in (
            "    total = sum(xs)",
            "    features = normalize_features([float(len(events)), user.activity])",
            "    return [score_user(user, events) for user in users]",
        ):
            source = source.replace(statement, f"    inspect.currentframe()\n{statement}")
        ranking.write_text(source, encoding="utf-8")
    (checkout / "pyproject.toml").write_text(
        "\n".join(
            (
                "[project]",
                'name = "simple-project"',
                'version = "0.1.0"',
                'requires-python = ">=3.12"',
                "",
                "[build-system]",
                "requires = []",
                'build-backend = "corpus_backend"',
                'backend-path = ["."]',
                "",
            )
        ),
        encoding="utf-8",
    )
    _git(checkout, "init")
    _git(checkout, "config", "user.name", "Corpus Fixture")
    _git(checkout, "config", "user.email", "corpus@example.invalid")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-m", "pinned fixture")
    revision = _git(checkout, "rev-parse", "HEAD").stdout.strip()
    (checkout / "moving-branch.txt").write_text("not in pinned commit\n", encoding="utf-8")
    _git(checkout, "add", "moving-branch.txt")
    _git(checkout, "commit", "-m", "move branch")
    return checkout, revision


def _write_manifest(
    tmp_path: Path,
    revision: str,
    *,
    oracle_arguments: tuple[str, ...] = (),
) -> Path:
    path = tmp_path / "manifest.toml"
    path.write_text(
        "\n".join(
            (
                "schema_version = 1",
                'python_version = "3.12"',
                'backends = ["mypyc", "cython"]',
                "test_timeout_seconds = 300",
                "compile_timeout_seconds = 900",
                "performance_timeout_seconds = 900",
                "max_log_bytes = 10485760",
                "",
                "[[case]]",
                'id = "simple-project"',
                'name = "Simple Project"',
                'repository = "https://example.invalid/simple-project.git"',
                f'revision = "{revision}"',
                'project_subroot = "."',
                'dependency_lock = "uv.lock"',
                'focused_test_command = ["python", "-m", "pytest", "tests/test_ranking.py", "-q"]',
                'oracle_adapter = "corpus_oracle"',
                f"oracle_arguments = {json.dumps(oracle_arguments)}",
                'tiers = ["compatibility"]',
                'platforms = ["ubuntu-24.04"]',
                "",
            )
        ),
        encoding="utf-8",
    )
    return path


def _git(checkout: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required for the local corpus fixture")
    return subprocess.run(
        (git, "-C", str(checkout), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
