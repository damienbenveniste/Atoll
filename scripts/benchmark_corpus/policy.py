"""Render and record compile policy for disposable benchmark checkouts.

The checkout's original configuration is never edited in place outside the
disposable clone.  The exact append operation is retained as a unified patch so
reviewers can reproduce the source identity used for compilation.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import tomllib
from pathlib import Path, PurePosixPath
from typing import cast

from scripts.benchmark_corpus.manifest import ManifestError
from scripts.benchmark_corpus.models import CompilePolicy, PolicyEvidence

_REQUIRED_BACKEND_COUNT = 2
_LEGACY_BUILD_SYSTEM = """[build-system]
requires = ["setuptools>=77", "wheel"]
build-backend = "setuptools.build_meta"
"""


def append_compile_policy(
    pyproject: Path,
    policy: CompilePolicy,
    evidence_root: Path,
    checkout_root: Path,
) -> PolicyEvidence:
    """Append exact Atoll policy and retain its reviewed unified patch.

    Args:
        pyproject: Disposable target project's ``pyproject.toml``.
        policy: Exact compile configuration to append.
        evidence_root: Directory receiving the policy patch.
        checkout_root: Checkout root used for safe relative paths.

    Returns:
        PolicyEvidence: Digest and repository-relative evidence paths.

    Raises:
        ManifestError: If an upstream compile table exists or paths escape their roots.
    """
    resolved_checkout = checkout_root.resolve()
    resolved_pyproject = pyproject.resolve()
    if not resolved_pyproject.is_relative_to(resolved_checkout):
        raise ManifestError("target pyproject resolves outside the disposable checkout")
    existed = pyproject.exists()
    original = pyproject.read_text(encoding="utf-8") if existed else ""
    if existed:
        parsed = cast(dict[str, object], tomllib.loads(original))
        if _has_compile_policy(parsed):
            raise ManifestError("upstream project already defines [tool.atoll.compile]")
    rendered = render_compile_policy(policy)
    prefix = original.rstrip() if existed else _LEGACY_BUILD_SYSTEM.rstrip()
    updated = f"{prefix}\n\n{rendered}"
    pyproject.write_text(updated, encoding="utf-8")
    relative_source = resolved_pyproject.relative_to(resolved_checkout).as_posix()
    patch_text = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=f"a/{relative_source}" if existed else "/dev/null",
            tofile=f"b/{relative_source}",
        )
    )
    evidence_root.mkdir(parents=True, exist_ok=True)
    patch_path = evidence_root / "compile-policy.patch"
    patch_path.write_text(patch_text, encoding="utf-8")
    return PolicyEvidence(
        digest=hashlib.sha256(patch_text.encode()).hexdigest(),
        patch_path=PurePosixPath(patch_path.relative_to(evidence_root).as_posix()),
        source_path=PurePosixPath(relative_source),
    )


def render_compile_policy(policy: CompilePolicy) -> str:
    """Render deterministic TOML accepted by Atoll's source-clean compiler.

    Args:
        policy: Validated benchmark policy.

    Returns:
        str: Complete ``[tool.atoll.compile]`` table ending in a newline.

    Raises:
        ManifestError: If commands or performance thresholds are inconsistent.
    """
    if (
        set(policy.backends) != {"mypyc", "cython"}
        or len(policy.backends) != _REQUIRED_BACKEND_COUNT
    ):
        raise ManifestError("benchmark policy must contain mypyc and cython exactly once")
    if policy.benchmark_command is not None and policy.test_command is None:
        raise ManifestError("benchmark policy requires a semantic test command")
    if policy.benchmark_warmups < 0 or policy.benchmark_samples <= 0:
        raise ManifestError("benchmark warmups and samples must be nonnegative and positive")
    if policy.minimum_speedup <= 1.0:
        raise ManifestError("benchmark minimum_speedup must be greater than 1.0")
    lines = [
        "[tool.atoll.compile]",
        f"backends = {_toml_array(policy.backends)}",
    ]
    if policy.test_command is not None:
        lines.append(f"test_command = {_toml_array(policy.test_command)}")
    if policy.benchmark_command is not None:
        lines.extend(
            (
                f"benchmark_command = {_toml_array(policy.benchmark_command)}",
                f"benchmark_warmups = {policy.benchmark_warmups}",
                f"benchmark_samples = {policy.benchmark_samples}",
                f"minimum_speedup = {policy.minimum_speedup:.6g}",
            )
        )
    return f"{'\n'.join(lines)}\n"


def _toml_array(values: tuple[str, ...]) -> str:
    return f"[{', '.join(json.dumps(value) for value in values)}]"


def _has_compile_policy(payload: dict[str, object]) -> bool:
    tool = payload.get("tool")
    if not isinstance(tool, dict):
        return False
    atoll = cast(dict[object, object], tool).get("atoll")
    return isinstance(atoll, dict) and "compile" in cast(dict[object, object], atoll)
