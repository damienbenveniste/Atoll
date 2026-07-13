"""Bounded, credential-free process execution for external corpus projects.

This module owns subprocess isolation, environment construction, and log
retention. It does not download projects, resolve dependencies, or interpret
the command's output.
"""

from __future__ import annotations

import os
import platform
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal

Sandbox = Literal["sandbox-exec", "bwrap", "unsandboxed"]

_DARWIN_READ_ROOTS = (
    "/Applications/Xcode.app",
    "/Library",
    "/System",
    "/bin",
    "/dev",
    "/opt",
    "/private/etc",
    "/private/var/db",
    "/sbin",
    "/usr",
)
_LINUX_READ_ROOTS = ("/etc", "/nix/store", "/opt", "/run", "/sys", "/usr")
_LINUX_COMPATIBILITY_SYMLINKS = ("/bin", "/lib", "/lib64", "/sbin")

_ENVIRONMENT_ALLOWLIST = frozenset(
    {
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "PATH",
        "SYSTEMROOT",
        "TERM",
        "TZ",
    }
)


@dataclass(frozen=True, slots=True)
class ProcessLimits:
    """Resource boundaries for one external command.

    Attributes:
        timeout_seconds: Wall-clock time allowed before group termination.
        max_log_bytes: Maximum combined stdout and stderr bytes retained.
        terminate_grace_seconds: Time allowed after SIGTERM before SIGKILL.
    """

    timeout_seconds: float
    max_log_bytes: int
    terminate_grace_seconds: float = 2.0

    def __post_init__(self) -> None:
        """Reject limits that cannot provide a meaningful execution bound.

        Raises:
            ValueError: If any duration or byte boundary is not positive.
        """
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_log_bytes <= 0:
            raise ValueError("max_log_bytes must be positive")
        if self.terminate_grace_seconds <= 0:
            raise ValueError("terminate_grace_seconds must be positive")


@dataclass(frozen=True, slots=True)
class ProcessResult:
    """Stable outcome of one bounded process execution.

    Attributes:
        exit_code: Final process return code, including signal-derived values.
        timed_out: Whether the timeout initiated process-group termination.
        duration_seconds: Monotonic elapsed wall-clock duration.
        log_truncated: Whether bytes beyond the retained log cap were discarded.
        argv: Original command arguments, excluding sandbox wrapper arguments.
    """

    exit_code: int
    timed_out: bool
    duration_seconds: float
    log_truncated: bool
    argv: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProcessRequest:
    """Complete command and filesystem boundary for one subprocess.

    Attributes:
        argv: Direct executable and arguments, never interpreted by a shell.
        cwd: Existing command working directory.
        environment: Complete sanitized child environment.
        log_path: Destination for retained combined output.
        readable_paths: Additional trusted inputs exposed read-only in a sandbox.
        writable_paths: Case-owned roots exposed read-write in a sandbox.
        network_allowed: Whether a tool-backed sandbox exposes networking.
    """

    argv: tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str]
    log_path: Path
    readable_paths: tuple[Path, ...] = ()
    writable_paths: tuple[Path, ...] = ()
    network_allowed: bool = False


def sanitized_environment(
    home: Path,
    tmp: Path,
    offline: bool,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build a minimal subprocess environment without inherited credentials.

    Args:
        home: Isolated home directory exposed to the command.
        tmp: Isolated temporary and cache directory exposed to the command.
        offline: Whether package installers must be forced into offline mode.
        base: Source environment. The current process environment is used when
            omitted, but only explicitly allowlisted keys can cross the boundary.

    Returns:
        dict[str, str]: A new environment with deterministic hashing and
        isolated home, temporary, cache, and package-tool configuration.
    """
    source = os.environ if base is None else base
    environment = {key: value for key, value in source.items() if key in _ENVIRONMENT_ALLOWLIST}
    environment.update(
        {
            "HOME": str(home),
            "TMPDIR": str(tmp),
            "TMP": str(tmp),
            "TEMP": str(tmp),
            "XDG_CACHE_HOME": str(tmp / "cache"),
            "XDG_CONFIG_HOME": str(home / ".config"),
            "PYTHONHASHSEED": "0",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_CONFIG_FILE": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "UV_CACHE_DIR": str(tmp / "uv-cache"),
        }
    )
    if offline:
        environment.update({"UV_OFFLINE": "1", "PIP_NO_INDEX": "1"})
    return environment


def detect_sandbox(allow_unsandboxed: bool) -> Sandbox:
    """Select a supported boundary for running untrusted external code.

    macOS uses ``sandbox-exec`` and Linux uses Bubblewrap, including on CI.
    Environment markers are not trusted as isolation evidence. A machine
    without the platform tool is refused unless the caller explicitly opts
    into unsandboxed execution.

    Args:
        allow_unsandboxed: Permit local execution without an isolation tool.

    Returns:
        Sandbox: Execution mode to pass to :func:`run_process`.

    Raises:
        RuntimeError: If no supported boundary exists and unsandboxed execution
            was not explicitly allowed.
    """
    system = platform.system()
    if system == "Darwin" and shutil.which("sandbox-exec") is not None:
        return "sandbox-exec"
    if system == "Linux" and shutil.which("bwrap") is not None:
        return "bwrap"
    if allow_unsandboxed:
        return "unsandboxed"
    raise RuntimeError(
        "external execution refused: no supported sandbox is available; "
        "use an ephemeral GitHub Actions runner, install sandbox-exec/bwrap, "
        "or explicitly allow unsandboxed execution"
    )


def run_process(
    request: ProcessRequest,
    limits: ProcessLimits,
    sandbox: Sandbox,
) -> ProcessResult:
    """Run one argv directly while bounding time and retained output.

    Stdout and stderr are merged and continuously drained. Once the log limit
    is reached, additional bytes are discarded without blocking the child. On
    timeout, the complete process group receives SIGTERM and then SIGKILL if it
    does not exit during the configured grace period.

    Args:
        request: Direct argv, filesystem, environment, log, and network boundary.
        limits: Timeout, retained-output, and termination-grace boundaries.
        sandbox: Isolation mode selected by :func:`detect_sandbox`.
        network_allowed: Allow network access inside tool-backed sandboxes.

    Returns:
        ProcessResult: Immutable exit, timing, truncation, and argv record.

    Raises:
        ValueError: If argv is empty or contains an empty argument.
        OSError: If directories, the log, sandbox, or executable cannot be used.
    """
    original_argv = tuple(request.argv)
    if not original_argv or any(not argument for argument in original_argv):
        raise ValueError("argv must contain a non-empty executable and arguments")

    resolved_cwd = request.cwd.resolve(strict=True)
    request.log_path.parent.mkdir(parents=True, exist_ok=True)
    command = sandbox_command(
        request,
        resolved_cwd,
        sandbox,
    )
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=resolved_cwd,
        env=dict(request.environment),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=False,
        start_new_session=True,
    )
    if process.stdout is None:  # pragma: no cover - guaranteed by stdout=PIPE
        raise RuntimeError("subprocess stdout pipe was not created")

    drain_state = _DrainState()
    drain_thread = threading.Thread(
        target=_drain_output,
        args=(process.stdout, request.log_path, limits.max_log_bytes, drain_state),
        name="benchmark-corpus-log-drain",
        daemon=True,
    )
    drain_thread.start()
    timed_out = False
    try:
        process.wait(timeout=limits.timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_process_group(process, limits.terminate_grace_seconds)
    finally:
        if not timed_out:
            _terminate_process_group(process, limits.terminate_grace_seconds)
        drain_thread.join(timeout=limits.terminate_grace_seconds)
        process.stdout.close()
        drain_thread.join(timeout=limits.terminate_grace_seconds)

    duration = time.monotonic() - started
    if process.returncode is None:  # pragma: no cover - wait or termination always sets it
        raise RuntimeError("subprocess ended without a return code")
    return ProcessResult(
        exit_code=process.returncode,
        timed_out=timed_out,
        duration_seconds=duration,
        log_truncated=drain_state.truncated,
        argv=original_argv,
    )


@dataclass(slots=True)
class _DrainState:
    """Mutable handoff populated by the single output-draining thread."""

    truncated: bool = False


def _drain_output(
    stream: BinaryIO,
    log_path: Path,
    max_log_bytes: int,
    state: _DrainState,
) -> None:
    """Retain a prefix of a byte stream while consuming it through EOF."""
    retained = 0
    with log_path.open("wb") as log:
        while chunk := stream.read(64 * 1024):
            remaining = max_log_bytes - retained
            if remaining > 0:
                kept = chunk[:remaining]
                log.write(kept)
                retained += len(kept)
            if len(chunk) > remaining:
                state.truncated = True


def _terminate_process_group(process: subprocess.Popen[bytes], grace_seconds: float) -> None:
    """Terminate the complete POSIX process group, even after its leader exits."""
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while _process_group_exists(process.pid) and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.01)
    if _process_group_exists(process.pid):
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    if process.returncode is None:
        process.wait()


def _process_group_exists(group_id: int) -> bool:
    try:
        os.killpg(group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def sandbox_command(
    request: ProcessRequest,
    cwd: Path,
    sandbox: Sandbox,
) -> tuple[str, ...]:
    """Wrap a process request in an explicit platform sandbox command.

    Args:
        request: Original command, environment, and declared filesystem roots.
        cwd: Resolved working directory.
        sandbox: Selected isolation implementation.

    Returns:
        tuple[str, ...]: Direct argv for the sandbox launcher or original command.
    """
    argv = request.argv
    if sandbox == "unsandboxed":
        return argv
    readable, writable = _sandbox_paths(
        request,
        cwd,
        platform.system(),
    )
    if sandbox == "sandbox-exec":
        read_rules = " ".join(f'(subpath "{_sandbox_profile_path(path)}")' for path in readable)
        write_rules = " ".join(f'(subpath "{_sandbox_profile_path(path)}")' for path in writable)
        ancestor_rules = " ".join(
            f'(literal "{_sandbox_profile_path(path)}")'
            for path in _path_ancestors((*readable, *writable))
        )
        network_rule = "(allow network*)" if request.network_allowed else ""
        profile = (
            '(version 1) (deny default) (import "system.sb") (allow process*) '
            f"(allow file-read-metadata file-test-existence {ancestor_rules}) "
            f"(allow file-read* {read_rules}) (allow file-write* {write_rules}) "
            f"{network_rule}"
        )
        return ("sandbox-exec", "-p", profile, "--", *argv)
    if sandbox == "bwrap":
        network_arguments = () if request.network_allowed else ("--unshare-net",)
        directory_arguments = _bubblewrap_directory_arguments((*readable, *writable))
        symlink_arguments = _bubblewrap_symlink_arguments()
        read_arguments = tuple(
            argument for path in readable for argument in ("--ro-bind", str(path), str(path))
        )
        write_arguments = tuple(
            argument for path in writable for argument in ("--bind", str(path), str(path))
        )
        return (
            "bwrap",
            "--die-with-parent",
            "--new-session",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            *network_arguments,
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--tmpfs",
            "/tmp",
            *directory_arguments,
            *symlink_arguments,
            *read_arguments,
            *write_arguments,
            "--chdir",
            str(cwd),
            "--",
            *argv,
        )
    raise ValueError(f"unsupported sandbox mode: {sandbox}")


def _sandbox_profile_path(path: Path) -> str:
    """Escape one filesystem path for a sandbox-exec string literal."""
    return str(path.resolve()).replace("\\", "\\\\").replace('"', '\\"')


def _sandbox_paths(
    request: ProcessRequest,
    cwd: Path,
    system: str,
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Return compact explicit read and write roots for a sandbox invocation."""
    environment = request.environment
    writable = _compact_paths(
        (
            cwd,
            Path(environment["HOME"]),
            Path(environment["TMPDIR"]),
            *request.writable_paths,
        )
    )
    system_roots = _DARWIN_READ_ROOTS if system == "Darwin" else _LINUX_READ_ROOTS
    path_directories = tuple(
        Path(item) for item in environment.get("PATH", "").split(os.pathsep) if item
    )
    path_roots = tuple(
        root for directory in path_directories for root in _tool_runtime_roots(directory)
    )
    executable = _executable_path(request.argv[0], environment)
    candidates = (
        *(Path(item) for item in system_roots),
        *path_roots,
        *_tool_runtime_roots(executable.parent),
        *request.readable_paths,
        *writable,
    )
    readable = _compact_paths(tuple(path for path in candidates if path.exists()))
    return readable, writable


def _executable_path(value: str, environment: Mapping[str, str]) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve(strict=True)
    executable = shutil.which(value, path=environment.get("PATH"))
    if executable is None:
        raise OSError(f"sandbox executable input cannot be resolved: {value}")
    return Path(executable).resolve(strict=True)


def _tool_runtime_roots(directory: Path) -> tuple[Path, ...]:
    """Include a selected bin directory and its wrapper-owned installation prefix."""
    roots = [directory]
    if directory.name in {"bin", "sbin"} and directory.parent != Path("/"):
        roots.append(directory.parent)
    return tuple(roots)


def _compact_paths(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    resolved = tuple(
        sorted({path.resolve(strict=True) for path in paths}, key=lambda p: len(p.parts))
    )
    selected: list[Path] = []
    for path in resolved:
        if not any(path == parent or path.is_relative_to(parent) for parent in selected):
            selected.append(path)
    return tuple(sorted(selected))


def _path_ancestors(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    ancestors = {
        parent for path in paths for parent in path.parents if parent not in {Path("/"), path}
    }
    return tuple(sorted(ancestors))


def _bubblewrap_directory_arguments(paths: tuple[Path, ...]) -> tuple[str, ...]:
    directories: set[Path] = set()
    for path in paths:
        directories.update(parent for parent in path.parents if parent != Path("/"))
        directories.add(path)
    directories.discard(Path("/tmp"))
    return tuple(
        argument
        for path in sorted(directories, key=lambda item: (len(item.parts), str(item)))
        for argument in ("--dir", str(path))
    )


def _bubblewrap_symlink_arguments() -> tuple[str, ...]:
    arguments: list[str] = []
    for raw_path in _LINUX_COMPATIBILITY_SYMLINKS:
        path = Path(raw_path)
        if not path.is_symlink():
            continue
        arguments.extend(("--symlink", str(path.readlink()), raw_path))
    return tuple(arguments)
