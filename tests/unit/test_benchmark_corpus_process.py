"""Tests for bounded and isolated benchmark-corpus process execution."""

from __future__ import annotations

import os
import platform
import shutil
import sys
from pathlib import Path

import pytest
from scripts.benchmark_corpus import process as process_module
from scripts.benchmark_corpus.process import (
    ProcessLimits,
    ProcessRequest,
    Sandbox,
    detect_sandbox,
    run_process,
    sanitized_environment,
)

_MAX_EXPECTED_TIMEOUT_DURATION_SECONDS = 2.0


def _missing_executable(_name: str) -> None:
    """Model a platform where no sandbox executable is installed."""


def test_sanitized_environment_strips_credentials_and_forces_offline_flags(
    tmp_path: Path,
) -> None:
    """Only benign host settings cross the boundary and offline mode is explicit."""
    home = tmp_path / "home"
    temporary = tmp_path / "tmp"
    environment = sanitized_environment(
        home,
        temporary,
        offline=True,
        base={
            "PATH": "/usr/bin",
            "LANG": "en_US.UTF-8",
            "AWS_SECRET_ACCESS_KEY": "secret",
            "GITHUB_TOKEN": "token",
            "PIP_INDEX_URL": "https://user:password@example.invalid/simple",
            "UV_INDEX_URL": "https://token@example.invalid/simple",
            "SSH_AUTH_SOCK": str(tmp_path / "agent.sock"),
        },
    )

    assert environment["PATH"] == "/usr/bin"
    assert environment["LANG"] == "en_US.UTF-8"
    assert environment["HOME"] == str(home)
    assert environment["TMPDIR"] == str(temporary)
    assert environment["PYTHONHASHSEED"] == "0"
    assert environment["UV_OFFLINE"] == "1"
    assert environment["PIP_NO_INDEX"] == "1"
    assert environment["PIP_CONFIG_FILE"] == os.devnull
    assert "AWS_SECRET_ACCESS_KEY" not in environment
    assert "GITHUB_TOKEN" not in environment
    assert "PIP_INDEX_URL" not in environment
    assert "UV_INDEX_URL" not in environment
    assert "SSH_AUTH_SOCK" not in environment


def test_github_actions_marker_cannot_bypass_missing_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A spoofable environment marker is not accepted as isolation evidence."""
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setattr("scripts.benchmark_corpus.process.platform.system", lambda: "FreeBSD")
    monkeypatch.setattr("scripts.benchmark_corpus.process.shutil.which", _missing_executable)

    with pytest.raises(RuntimeError, match="external execution refused"):
        detect_sandbox(allow_unsandboxed=False)


def test_detect_sandbox_refuses_unsupported_local_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """External code cannot run locally without isolation or explicit consent."""
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setattr("scripts.benchmark_corpus.process.platform.system", lambda: "FreeBSD")
    monkeypatch.setattr("scripts.benchmark_corpus.process.shutil.which", _missing_executable)

    with pytest.raises(RuntimeError, match="external execution refused"):
        detect_sandbox(allow_unsandboxed=False)

    assert detect_sandbox(allow_unsandboxed=True) == "unsandboxed"


def test_run_process_caps_log_while_draining_excess_output(tmp_path: Path) -> None:
    """Output beyond the retention boundary is discarded without deadlock."""
    log_path = tmp_path / "bounded.log"
    result = run_process(
        ProcessRequest(
            argv=(sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'x' * 1000000)"),
            cwd=tmp_path,
            environment=sanitized_environment(tmp_path / "home", tmp_path, offline=True),
            log_path=log_path,
        ),
        limits=ProcessLimits(timeout_seconds=5, max_log_bytes=4096),
        sandbox="unsandboxed",
    )

    assert result.exit_code == 0
    assert result.timed_out is False
    assert result.log_truncated is True
    assert log_path.read_bytes() == b"x" * 4096
    assert result.argv[0] == sys.executable


def test_run_process_times_out_and_terminates_child_process_group(tmp_path: Path) -> None:
    """Timeout cleanup reaches descendants that share the new process group."""
    child_code = "import os,time; print(os.getpid(), flush=True); time.sleep(30)"
    parent_code = (
        "import subprocess, sys, time; "
        f"child=subprocess.Popen([sys.executable, '-c', {child_code!r}], "
        "stdout=subprocess.PIPE, text=True); "
        "print(child.stdout.readline().strip(), flush=True); "
        "time.sleep(30)"
    )
    result = run_process(
        ProcessRequest(
            argv=(sys.executable, "-c", parent_code),
            cwd=tmp_path,
            environment=sanitized_environment(tmp_path / "home", tmp_path, offline=True),
            log_path=tmp_path / "timeout.log",
        ),
        limits=ProcessLimits(
            timeout_seconds=0.2,
            max_log_bytes=4096,
            terminate_grace_seconds=0.2,
        ),
        sandbox="unsandboxed",
    )

    assert result.timed_out is True
    assert result.duration_seconds < _MAX_EXPECTED_TIMEOUT_DURATION_SECONDS
    child_pid = int((tmp_path / "timeout.log").read_text(encoding="utf-8").strip())
    assert _pid_exists(child_pid) is False


def test_run_process_terminates_descendant_after_parent_success(tmp_path: Path) -> None:
    """A successful parent cannot leave an unsupervised background process."""
    child_code = "import os,time; print(os.getpid(), flush=True); time.sleep(30)"
    parent_code = (
        "import subprocess, sys; "
        f"child=subprocess.Popen([sys.executable, '-c', {child_code!r}], "
        "stdout=subprocess.PIPE, text=True); "
        "print(child.stdout.readline().strip(), flush=True)"
    )

    result = run_process(
        ProcessRequest(
            argv=(sys.executable, "-c", parent_code),
            cwd=tmp_path,
            environment=sanitized_environment(tmp_path / "home", tmp_path, offline=True),
            log_path=tmp_path / "success.log",
        ),
        limits=ProcessLimits(
            timeout_seconds=5,
            max_log_bytes=4096,
            terminate_grace_seconds=0.2,
        ),
        sandbox="unsandboxed",
    )

    assert result.exit_code == 0
    child_pid = int((tmp_path / "success.log").read_text(encoding="utf-8").strip())
    assert _pid_exists(child_pid) is False


def test_bubblewrap_uses_explicit_roots_and_process_namespaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux sandbox construction never exposes the host root filesystem."""
    workspace = tmp_path / "workspace"
    home = workspace / "home"
    temporary = workspace / "tmp"
    for path in (workspace, home, temporary):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("scripts.benchmark_corpus.process.platform.system", lambda: "Linux")
    environment = sanitized_environment(home, temporary, offline=True)

    request = ProcessRequest(
        argv=(sys.executable, "-c", "pass"),
        cwd=workspace,
        environment=environment,
        log_path=tmp_path / "bwrap.log",
        readable_paths=(Path(sys.executable).parent,),
        writable_paths=(workspace,),
    )
    command = process_module.sandbox_command(
        request,
        workspace,
        "bwrap",
    )

    assert tuple(command[command.index("--ro-bind") :][:3]) != ("--ro-bind", "/", "/")
    assert "--unshare-pid" in command
    assert "--unshare-ipc" in command
    assert "--unshare-net" in command
    assert "--bind" in command


def test_sandbox_exec_profile_has_no_unrestricted_host_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """macOS policy permits reads only beneath explicit runtime roots."""
    workspace = tmp_path / "workspace"
    home = workspace / "home"
    temporary = workspace / "tmp"
    for path in (workspace, home, temporary):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("scripts.benchmark_corpus.process.platform.system", lambda: "Darwin")
    environment = sanitized_environment(home, temporary, offline=True)

    request = ProcessRequest(
        argv=(sys.executable, "-c", "pass"),
        cwd=workspace,
        environment=environment,
        log_path=tmp_path / "sandbox-exec.log",
        readable_paths=(Path(sys.executable).parent,),
        writable_paths=(workspace,),
    )
    command = process_module.sandbox_command(
        request,
        workspace,
        "sandbox-exec",
    )

    profile = command[2]
    assert "(allow file-read*)" not in profile
    assert "(allow network*)" not in profile
    assert str(workspace) in profile


def test_available_platform_sandbox_blocks_unreviewed_host_file(tmp_path: Path) -> None:
    """A real local sandbox cannot read a sibling host path outside its roots."""
    system = platform.system()
    executable = "sandbox-exec" if system == "Darwin" else "bwrap"
    if system not in {"Darwin", "Linux"} or shutil.which(executable) is None:
        pytest.skip("supported platform sandbox is not installed")
    sandbox: Sandbox = "sandbox-exec" if system == "Darwin" else "bwrap"
    workspace = tmp_path / "workspace"
    home = workspace / "home"
    temporary = workspace / "tmp"
    for path in (workspace, home, temporary):
        path.mkdir(parents=True, exist_ok=True)
    secret = tmp_path / "host-secret"
    secret.write_text("not exposed", encoding="utf-8")
    environment = sanitized_environment(home, temporary, offline=True)

    result = run_process(
        ProcessRequest(
            argv=(
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(secret)!r}).read_text()",
            ),
            cwd=workspace,
            environment=environment,
            log_path=tmp_path / "denied.log",
            readable_paths=(Path(sys.executable).parent,),
            writable_paths=(workspace,),
        ),
        ProcessLimits(timeout_seconds=5, max_log_bytes=4096),
        sandbox,
    )

    assert result.exit_code != 0


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True
