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

VerificationStage = Literal["payload", "wheel"]

_VERIFY_SCRIPT = r"""
import hashlib
import importlib
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
if temporary is not None:
    temporary.cleanup()
"""


@dataclass(frozen=True, slots=True)
class VerificationArtifact:
    """One install-relative native artifact and its expected SHA-256 digest."""

    path: str
    digest: str


@dataclass(frozen=True, slots=True)
class PackageVerificationPlan:
    """Modules, region promises, and artifacts checked in a child interpreter."""

    modules: tuple[str, ...]
    regions: tuple[tuple[str, tuple[str, ...]], ...]
    artifacts: tuple[VerificationArtifact, ...]


@dataclass(frozen=True, slots=True)
class PackageVerificationResult:
    """Captured subprocess evidence for one payload or wheel verification stage."""

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
    """Verify compiled routing from an unpacked payload or extracted final wheel."""
    payload = json.dumps(
        {
            "modules": list(plan.modules),
            "regions": {module: list(regions) for module, regions in plan.regions},
            "artifacts": [
                {"path": artifact.path, "digest": artifact.digest} for artifact in plan.artifacts
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
