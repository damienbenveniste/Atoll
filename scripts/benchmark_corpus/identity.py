"""Canonical case and historical-comparison identities for corpus evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import PurePosixPath

from scripts.benchmark_corpus.models import (
    CorpusCase,
    EnvironmentEvidence,
    PolicyEvidence,
)


def case_digest(case: CorpusCase) -> str:
    """Hash every normalized manifest field for one corpus case.

    Args:
        case: Parsed immutable manifest case.

    Returns:
        str: SHA-256 digest used to reject stale case results.
    """
    encoded = json.dumps(
        asdict(case),
        default=_json_default,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def comparison_key(
    case: CorpusCase,
    environment: EnvironmentEvidence,
    policy: PolicyEvidence,
) -> str:
    """Hash every comparison-critical field except the Atoll revision.

    Args:
        case: Pinned project and workload identity.
        environment: Python, dependency, compiler, platform, and hardware evidence.
        policy: Exact disposable benchmark policy patch identity.

    Returns:
        str: Like-for-like history key. Different Atoll revisions intentionally
            share the key so compiler changes can be compared.
    """
    workload = None if case.workload is None else asdict(case.workload)
    payload = {
        "architecture": environment.architecture,
        "compiler": environment.compiler,
        "cython": environment.cython,
        "dependency_lock": environment.dependency_lock_digest,
        "hardware_class": environment.hardware_class,
        "mypy": environment.mypy,
        "operating_system": environment.operating_system,
        "platform": environment.runner_image,
        "policy": policy.digest,
        "python": environment.python,
        "revision": case.revision,
        "workload": workload,
    }
    encoded = json.dumps(payload, default=_json_default, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _json_default(value: object) -> str:
    if isinstance(value, PurePosixPath):
        return value.as_posix()
    raise TypeError(f"unsupported corpus identity value: {type(value).__name__}")
