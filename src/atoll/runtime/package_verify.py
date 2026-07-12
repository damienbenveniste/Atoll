"""Fresh-interpreter verification for staged payloads and final wheels.

Verification runs with isolated Python path handling, imports every module that
Atoll promised to route, and checks artifact digests plus per-region runtime
status. It does not install dependencies or mutate the target checkout.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from atoll.models import BindingKind, ExecutionKind

VerificationStage = Literal["payload", "wheel"]

_VERIFY_SCRIPT = r"""
import hashlib
import importlib
import inspect
import json
import os
import pathlib
import sys
import tempfile
import zipfile

stage = sys.argv[1]
target = pathlib.Path(sys.argv[2]).resolve()
plan = json.loads(sys.argv[3])
temporary = None
if stage == "wheel":
    temporary = tempfile.TemporaryDirectory(prefix="atoll-wheel-verify-")
    root = pathlib.Path(temporary.name)
    with zipfile.ZipFile(target) as archive:
        archive.extractall(root)
else:
    root = target
sys.path.insert(0, str(root))
os.environ.pop("ATOLL_DISABLE", None)
os.environ["ATOLL_REQUIRE_COMPILED"] = "1"
for artifact in plan["artifacts"]:
    path = root / artifact["path"]
    if not path.is_file():
        raise RuntimeError(f"missing Atoll artifact: {artifact['path']}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != artifact["digest"]:
        raise RuntimeError(f"Atoll artifact digest mismatch: {artifact['path']}")
for module_name in plan["modules"]:
    module = importlib.import_module(module_name)
    expected_regions = plan["regions"].get(module_name, [])
    if not expected_regions:
        status = getattr(module, "__atoll_status__", None)
        if not isinstance(status, dict) or not status.get("compiled"):
            raise RuntimeError(f"Atoll module did not report compiled routing: {module_name}")
        continue
    statuses = getattr(module, "__atoll_region_status__", None)
    if not isinstance(statuses, dict):
        raise RuntimeError(f"Atoll module has no per-region status: {module_name}")
    for region_id in expected_regions:
        region_status = statuses.get(region_id)
        if not isinstance(region_status, dict) or not region_status.get("compiled"):
            raise RuntimeError(f"Atoll region did not compile: {region_id}")
for binding in plan["bindings"]:
    module = importlib.import_module(binding["module"])
    parts = binding["qualname"].split(".")
    if binding["kind"] in {"module", "class"}:
        value = module
        for part in parts:
            value = getattr(value, part)
        descriptor = value
    else:
        owner = module
        for part in parts[:-1]:
            owner = getattr(owner, part)
        descriptor = vars(owner).get(parts[-1])
        if descriptor is None:
            raise RuntimeError(f"Atoll binding target is missing: {binding['qualname']}")
        if binding["kind"] == "staticmethod":
            if not isinstance(descriptor, staticmethod):
                raise RuntimeError(
                    f"Atoll binding changed staticmethod descriptor: {binding['qualname']}"
                )
            value = descriptor.__func__
        elif binding["kind"] == "classmethod":
            if not isinstance(descriptor, classmethod):
                raise RuntimeError(
                    f"Atoll binding changed classmethod descriptor: {binding['qualname']}"
                )
            value = descriptor.__func__
        else:
            if isinstance(descriptor, (staticmethod, classmethod)):
                raise RuntimeError(
                    f"Atoll binding changed instance method descriptor: {binding['qualname']}"
                )
            value = descriptor
    expected_kind = binding["execution_kind"]
    actual_kind = (
        "class"
        if isinstance(value, type)
        else "async_generator"
        if inspect.isasyncgenfunction(value)
        else "coroutine"
        if inspect.iscoroutinefunction(value)
        else "generator"
        if inspect.isgeneratorfunction(value)
        else "sync"
        if callable(value)
        else "invalid"
    )
    if actual_kind != expected_kind:
        raise RuntimeError(
            f"Atoll binding changed execution kind for {binding['qualname']}: "
            f"expected {expected_kind}, got {actual_kind}"
        )
    compiled_target = getattr(value, "__atoll_compiled_target__", None)
    fallback = getattr(value, "__atoll_python_fallback__", None)
    if compiled_target is None or fallback is None:
        raise RuntimeError(f"Atoll binding lacks routing metadata: {binding['qualname']}")
    expected_variants = binding.get("variant_ids", [])
    if expected_variants and expected_kind != "class":
        candidates = getattr(value, "__atoll_binding_variants__", None)
        if not isinstance(candidates, tuple):
            raise RuntimeError(f"Atoll binding lacks variant metadata: {binding['qualname']}")
        actual_variants = sorted(
            candidate.get("variant_id")
            for candidate in candidates
            if isinstance(candidate, dict)
        )
        if actual_variants != sorted(expected_variants):
            raise RuntimeError(f"Atoll binding variant mismatch: {binding['qualname']}")
    try:
        if inspect.signature(value) != inspect.signature(fallback):
            raise RuntimeError(f"Atoll binding changed signature: {binding['qualname']}")
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"Atoll binding signature cannot be verified: {binding['qualname']}"
        ) from error
    if expected_kind != "class" and (
        getattr(value, "__defaults__", None) is not getattr(fallback, "__defaults__", None)
        or getattr(value, "__kwdefaults__", None)
        is not getattr(fallback, "__kwdefaults__", None)
    ):
        raise RuntimeError(f"Atoll binding changed default objects: {binding['qualname']}")
if temporary is not None:
    temporary.cleanup()
"""


@dataclass(frozen=True, slots=True)
class VerificationArtifact:
    """One install-relative native artifact and its expected SHA-256 digest.

    Attributes:
        path: POSIX artifact path relative to the staged package payload.
        digest: Lowercase SHA-256 digest of artifact content.
    """

    path: str
    digest: str


@dataclass(frozen=True, slots=True)
class VerificationBinding:
    """One required public binding checked after its compiled wheel is imported.

    Attributes:
        module: Importable module containing the staged binding.
        qualname: Public runtime-qualified path within the module.
        kind: Module, class, or descriptor-aware binding category.
        execution_kind: Callable shape promised by the typed-region frontend.
        variant_ids: Compiled dispatcher variants promised for this binding.
    """

    module: str
    qualname: str
    kind: BindingKind
    execution_kind: ExecutionKind
    variant_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PackageVerificationPlan:
    """Modules, region promises, and artifacts checked in a child interpreter.

    Attributes:
        modules: Importable source modules that must report compiled routing.
        regions: Expected compiled region IDs grouped by source module.
        artifacts: Install-relative native files and digests that must be present.
        bindings: Required public bindings whose descriptor and execution shape must survive.
    """

    modules: tuple[str, ...]
    regions: tuple[tuple[str, tuple[str, ...]], ...]
    artifacts: tuple[VerificationArtifact, ...]
    bindings: tuple[VerificationBinding, ...] = ()


@dataclass(frozen=True, slots=True)
class PackageVerificationResult:
    """Captured subprocess evidence for one payload or wheel verification stage.

    Attributes:
        stage: Package verification stage represented by the result.
        target: Unpacked payload directory or final wheel archive under test.
        command: Normalized command argument vector.
        success: Whether the represented operation completed successfully.
        exit_code: Child process exit code.
        stdout: Captured child process standard output.
        stderr: Captured child process standard error.
        duration_seconds: Elapsed wall-clock duration in seconds.
    """

    stage: VerificationStage
    target: Path
    command: tuple[str, ...]
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float


def verify_package_subprocess(
    *,
    stage: VerificationStage,
    target: Path,
    plan: PackageVerificationPlan,
    project_root: Path,
) -> PackageVerificationResult:
    """Verify compiled routing from an unpacked payload or extracted final wheel.

    Args:
        stage: Verification stage that determines the target and failure context.
        target: Wheel or payload path verified in an isolated child process.
        plan: Expected modules, regions, and artifacts for package verification.
        project_root: Root directory of the target Python project.

    Returns:
        PackageVerificationResult: Captured isolated verification evidence for the requested stage.
    """
    payload = json.dumps(
        {
            "modules": list(plan.modules),
            "regions": {module: list(regions) for module, regions in plan.regions},
            "artifacts": [
                {"path": artifact.path, "digest": artifact.digest} for artifact in plan.artifacts
            ],
            "bindings": [
                {
                    "module": binding.module,
                    "qualname": binding.qualname,
                    "kind": binding.kind,
                    "execution_kind": binding.execution_kind,
                    "variant_ids": list(binding.variant_ids),
                }
                for binding in plan.bindings
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    command = (sys.executable, "-I", "-c", _VERIFY_SCRIPT, stage, str(target.resolve()), payload)
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=project_root.resolve(),
        check=False,
        shell=False,
        capture_output=True,
        text=True,
    )
    return PackageVerificationResult(
        stage=stage,
        target=target.resolve(),
        command=command,
        success=completed.returncode == 0,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        duration_seconds=time.perf_counter() - started,
    )
