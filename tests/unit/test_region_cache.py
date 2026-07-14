"""Tests for typed-region artifacts and deterministic backend decisions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

from atoll.backends.base import CompilerBackend
from atoll.models import (
    ArtifactRecord,
    Backend,
    BackendCompileContext,
    BackendCompileResult,
    CompilationUnit,
    CompileAttempt,
    SymbolId,
)
from atoll.region_cache import (
    compile_many_with_region_cache,
    compile_with_region_cache,
    probe_region_cache,
)

EXPECTED_DOUBLE_COMPILE_COUNT = 2
EXPECTED_BATCH_ARTIFACT_COUNT = 2


class _FakeBackend:
    """Minimal compiler double whose artifact writes expose cache behavior."""

    def __init__(self, *, name: Backend = "mypyc", key: str = "fingerprint") -> None:
        self.name: Backend = name
        self.key = key
        self.compile_count = 0
        self.fail = False
        self.failure_stderr = "transient failure"
        self.write_outside_output_root = False

    def fingerprint(self, unit: CompilationUnit, context: BackendCompileContext) -> str:
        _ = (unit, context)
        return self.key

    def compile(
        self,
        units: tuple[CompilationUnit, ...],
        context: BackendCompileContext,
    ) -> BackendCompileResult:
        self.compile_count += 1
        if self.fail:
            return BackendCompileResult(
                attempt=CompileAttempt(
                    success=False,
                    command=("fake",),
                    stdout="",
                    stderr=self.failure_stderr,
                    artifact_paths=(),
                    duration_seconds=0.1,
                ),
                artifacts=(),
            )
        artifact = (
            context.project_root / "outside.so"
            if self.write_outside_output_root
            else context.build_dir.parent / "artifacts" / "native.so"
        )
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(f"build-{self.compile_count}".encode())
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        return BackendCompileResult(
            attempt=CompileAttempt(
                success=True,
                command=("fake",),
                stdout="",
                stderr="",
                artifact_paths=(artifact,),
                duration_seconds=0.1,
            ),
            artifacts=(
                ArtifactRecord(
                    region_id=units[0].region_id,
                    backend=self.name,
                    logical_module=units[0].logical_module,
                    role="primary",
                    install_relative_path=f"{units[0].install_relative_dir}/native.so",
                    digest=digest,
                    abi="cp312",
                    platform_tag="test-platform",
                ),
            ),
        )


class _BatchBackend:
    """Cython-shaped compiler double with per-unit artifacts and failure controls."""

    name: Backend = "cython"

    def __init__(self) -> None:
        self.compile_calls: list[tuple[str, ...]] = []
        self.deterministic_bad_region: str | None = None
        self.transient_failure = False
        self.raise_on_compile = False
        self.emit_shared_artifact = False

    def fingerprint(self, unit: CompilationUnit, context: BackendCompileContext) -> str:
        _ = context
        return hashlib.sha256(f"batch:{unit.region_id}".encode()).hexdigest()

    def compile(
        self,
        units: tuple[CompilationUnit, ...],
        context: BackendCompileContext,
    ) -> BackendCompileResult:
        if self.raise_on_compile:
            raise AssertionError("warm cache reached native compiler")
        self.compile_calls.append(tuple(unit.region_id for unit in units))
        if self.transient_failure or any(
            unit.region_id == self.deterministic_bad_region for unit in units
        ):
            stderr = (
                "transient compiler outage"
                if self.transient_failure
                else "CYTHON_COMPILE_ERROR: rejected generated unit"
            )
            return BackendCompileResult(
                attempt=CompileAttempt(
                    success=False,
                    command=("fake-cython", *self.compile_calls[-1]),
                    stdout="",
                    stderr=stderr,
                    artifact_paths=(),
                    duration_seconds=0.1,
                ),
                artifacts=(),
            )
        artifact_root = context.build_dir.parent / "artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        records: list[ArtifactRecord] = []
        for unit in units:
            artifact = artifact_root / f"{unit.logical_module}.so"
            artifact.write_bytes(unit.region_id.encode())
            paths.append(artifact)
            records.append(_artifact_record(unit, artifact))
        if self.emit_shared_artifact:
            shared = artifact_root / "shared-support.so"
            shared.write_bytes(b"shared")
            paths.append(shared)
            records.extend(
                ArtifactRecord(
                    region_id="__shared__",
                    backend="cython",
                    logical_module="shared-support",
                    role="support",
                    install_relative_path=(f"{unit.install_relative_dir}/{shared.name}"),
                    digest=hashlib.sha256(shared.read_bytes()).hexdigest(),
                    abi="cp312",
                    platform_tag="test-platform",
                )
                for unit in units
            )
        return BackendCompileResult(
            attempt=CompileAttempt(
                success=True,
                command=("fake-cython", *self.compile_calls[-1]),
                stdout="",
                stderr="",
                artifact_paths=tuple(paths),
                duration_seconds=0.1,
            ),
            artifacts=tuple(records),
        )


def _artifact_record(unit: CompilationUnit, artifact: Path) -> ArtifactRecord:
    return ArtifactRecord(
        region_id=unit.region_id,
        backend="cython",
        logical_module=unit.logical_module,
        role="primary",
        install_relative_path=f"{unit.install_relative_dir}/{artifact.name}",
        digest=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        abi="cp312",
        platform_tag="test-platform",
    )


def test_region_cache_restores_successful_artifact_and_record(tmp_path: Path) -> None:
    """An unchanged backend variant restores bytes without invoking the compiler."""
    backend = _FakeBackend()
    context = _context(tmp_path)
    unit = _unit(tmp_path)
    stale_temp = tmp_path / "cache" / "mypyc" / f"{backend.key}.tmp"
    stale_temp.mkdir(parents=True)

    first = compile_with_region_cache(
        cast(CompilerBackend, backend),
        unit,
        context,
        cache_root=tmp_path / "cache",
    )
    first.attempt.artifact_paths[0].unlink()
    second = compile_with_region_cache(
        cast(CompilerBackend, backend),
        unit,
        context,
        cache_root=tmp_path / "cache",
    )

    assert backend.compile_count == 1
    assert first.attempt.cache_status == "miss"
    assert second.attempt.cache_status == "hit"
    assert second.attempt.artifact_paths[0].read_bytes() == b"build-1"
    assert second.artifacts == first.artifacts
    assert [timing.name for timing in second.attempt.phase_timings] == [
        "cache_lookup",
        "cache_restore",
    ]


def test_region_cache_probe_restores_success_without_compiling(tmp_path: Path) -> None:
    """A cache-only probe honors an existing success and leaves misses untouched."""
    backend = _FakeBackend()
    context = _context(tmp_path)
    unit = _unit(tmp_path)

    assert (
        probe_region_cache(
            cast(CompilerBackend, backend),
            unit,
            context,
            cache_root=tmp_path / "cache",
        )
        is None
    )
    first = compile_with_region_cache(
        cast(CompilerBackend, backend),
        unit,
        context,
        cache_root=tmp_path / "cache",
    )
    first.attempt.artifact_paths[0].unlink()
    restored = probe_region_cache(
        cast(CompilerBackend, backend),
        unit,
        context,
        cache_root=tmp_path / "cache",
    )

    assert backend.compile_count == 1
    assert restored is not None
    assert restored.attempt.success is True
    assert restored.attempt.cache_status == "hit"
    assert restored.attempt.artifact_paths[0].read_bytes() == b"build-1"


def test_region_cache_treats_corruption_as_miss(tmp_path: Path) -> None:
    """Digest mismatch triggers a fresh build and atomically replaces the entry."""
    backend = _FakeBackend()
    context = _context(tmp_path)
    unit = _unit(tmp_path)
    compile_with_region_cache(
        cast(CompilerBackend, backend),
        unit,
        context,
        cache_root=tmp_path / "cache",
    )
    cached = next((tmp_path / "cache" / "mypyc" / backend.key / "artifacts").rglob("*.so"))
    cached.write_bytes(b"corrupt")

    result = compile_with_region_cache(
        cast(CompilerBackend, backend),
        unit,
        context,
        cache_root=tmp_path / "cache",
    )

    assert backend.compile_count == EXPECTED_DOUBLE_COMPILE_COUNT
    assert result.attempt.cache_status == "miss"
    assert result.attempt.artifact_paths[0].read_bytes() == b"build-2"


def test_region_cache_never_stores_failures(tmp_path: Path) -> None:
    """Repeated transient failures always reach the backend again."""
    backend = _FakeBackend()
    backend.fail = True
    context = _context(tmp_path)
    unit = _unit(tmp_path)

    first = compile_with_region_cache(
        cast(CompilerBackend, backend),
        unit,
        context,
        cache_root=tmp_path / "cache",
    )
    second = compile_with_region_cache(
        cast(CompilerBackend, backend),
        unit,
        context,
        cache_root=tmp_path / "cache",
    )

    assert backend.compile_count == EXPECTED_DOUBLE_COMPILE_COUNT
    assert first.attempt.cache_status == "miss"
    assert second.attempt.cache_status == "miss"
    assert not (tmp_path / "cache" / "mypyc" / backend.key).exists()
    assert not (tmp_path / "cache" / "decisions" / "mypyc" / f"{backend.key}.json").exists()


def test_batched_region_cache_handles_empty_and_mismatched_inputs(tmp_path: Path) -> None:
    """Batch orchestration is empty-safe and rejects units for another backend."""
    backend = _BatchBackend()
    context = _context(tmp_path)

    assert (
        compile_many_with_region_cache(
            cast(CompilerBackend, backend),
            (),
            context,
            cache_root=tmp_path / "cache",
        )
        == ()
    )
    mismatched = _unit(tmp_path)
    with pytest.raises(ValueError, match="selected backend"):
        compile_many_with_region_cache(
            cast(CompilerBackend, backend),
            (mismatched,),
            context,
            cache_root=tmp_path / "cache",
        )


def test_batched_region_cache_delegates_non_cython_units(tmp_path: Path) -> None:
    """A non-Cython batch retains the established independent compile path."""
    backend = _FakeBackend()
    context = _context(tmp_path)

    results = compile_many_with_region_cache(
        cast(CompilerBackend, backend),
        (_unit(tmp_path),),
        context,
        cache_root=tmp_path / "cache",
    )

    assert backend.compile_count == 1
    assert len(results) == 1
    assert results[0].attempt.cache_status == "miss"


def test_region_cache_batches_only_misses_and_warm_run_never_compiles(tmp_path: Path) -> None:
    """One warm entry is restored while cold peers share one physical invocation."""
    backend = _BatchBackend()
    context = _context(tmp_path)
    units = _units(tmp_path, 3)
    compile_with_region_cache(
        cast(CompilerBackend, backend),
        units[0],
        context,
        cache_root=tmp_path / "cache",
    )
    backend.compile_calls.clear()

    mixed = compile_many_with_region_cache(
        cast(CompilerBackend, backend),
        units,
        context,
        cache_root=tmp_path / "cache",
    )

    assert backend.compile_calls == [(units[1].region_id, units[2].region_id)]
    assert [result.attempt.cache_status for result in mixed] == ["hit", "miss", "miss"]
    backend.raise_on_compile = True
    warm = compile_many_with_region_cache(
        cast(CompilerBackend, backend),
        units,
        context,
        cache_root=tmp_path / "cache",
    )
    assert all(result.attempt.cache_status == "hit" for result in warm)


def test_region_cache_bisects_deterministic_cython_failure(tmp_path: Path) -> None:
    """One bad unit cannot discard good peers or poison an aggregate cache entry."""
    backend = _BatchBackend()
    context = _context(tmp_path)
    units = _units(tmp_path, 4)
    backend.deterministic_bad_region = units[1].region_id

    results = compile_many_with_region_cache(
        cast(CompilerBackend, backend),
        units,
        context,
        cache_root=tmp_path / "cache",
    )

    assert backend.compile_calls == [
        tuple(unit.region_id for unit in units),
        (units[0].region_id, units[1].region_id),
        (units[0].region_id,),
        (units[1].region_id,),
        (units[2].region_id, units[3].region_id),
    ]
    assert sum(
        timing.name in {"cython_batch", "cython_batch_retry"}
        for result in results
        for timing in result.attempt.phase_timings
    ) == len(backend.compile_calls)
    assert [result.attempt.success for result in results] == [True, False, True, True]
    assert results[1].attempt.phase_timings[-1].name == "backend_decision_store"
    backend.compile_calls.clear()
    backend.raise_on_compile = True
    warm = compile_many_with_region_cache(
        cast(CompilerBackend, backend),
        units,
        context,
        cache_root=tmp_path / "cache",
    )
    assert [result.attempt.success for result in warm] == [True, False, True, True]
    assert all(result.attempt.cache_status == "hit" for result in warm)


def test_region_cache_does_not_bisect_transient_batch_failure(tmp_path: Path) -> None:
    """Environment failures remain one uncached physical attempt for every miss."""
    backend = _BatchBackend()
    backend.transient_failure = True
    context = _context(tmp_path)
    units = _units(tmp_path, 3)

    first = compile_many_with_region_cache(
        cast(CompilerBackend, backend),
        units,
        context,
        cache_root=tmp_path / "cache",
    )
    second = compile_many_with_region_cache(
        cast(CompilerBackend, backend),
        units,
        context,
        cache_root=tmp_path / "cache",
    )

    assert backend.compile_calls == [
        tuple(unit.region_id for unit in units),
        tuple(unit.region_id for unit in units),
    ]
    assert all(not result.attempt.success for result in (*first, *second))
    assert not (tmp_path / "cache" / "decisions").exists()


def test_region_cache_partitions_shared_artifacts_per_unit(tmp_path: Path) -> None:
    """Primary and colocated support records survive independent warm restores."""
    backend = _BatchBackend()
    backend.emit_shared_artifact = True
    context = _context(tmp_path)
    units = _units(tmp_path, 2)

    cold = compile_many_with_region_cache(
        cast(CompilerBackend, backend),
        units,
        context,
        cache_root=tmp_path / "cache",
    )
    backend.raise_on_compile = True
    warm = compile_many_with_region_cache(
        cast(CompilerBackend, backend),
        units,
        context,
        cache_root=tmp_path / "cache",
    )

    assert all(
        tuple(record.role for record in result.artifacts) == ("primary", "support")
        for result in cold
    )
    assert tuple(result.artifacts for result in warm) == tuple(result.artifacts for result in cold)
    assert all(
        len(result.attempt.artifact_paths) == EXPECTED_BATCH_ARTIFACT_COUNT for result in warm
    )


def test_region_cache_restores_deterministic_backend_rejection(tmp_path: Path) -> None:
    """A stable type rejection skips the unchanged backend on the next run."""
    backend = _FakeBackend()
    backend.fail = True
    backend.failure_stderr = "MYPYC_TYPE_ERROR: generated unit is incompatible"
    context = _context(tmp_path)
    unit = _unit(tmp_path)

    first = compile_with_region_cache(
        cast(CompilerBackend, backend),
        unit,
        context,
        cache_root=tmp_path / "cache",
    )
    second = compile_with_region_cache(
        cast(CompilerBackend, backend),
        unit,
        context,
        cache_root=tmp_path / "cache",
    )

    decision = tmp_path / "cache" / "decisions" / "mypyc" / f"{backend.key}.json"
    assert backend.compile_count == 1
    assert first.attempt.cache_status == "miss"
    assert first.attempt.phase_timings[-1].name == "backend_decision_store"
    assert second.attempt.cache_status == "hit"
    assert second.attempt.command[:3] == ("atoll", "cache", "reject")
    assert second.attempt.stderr.startswith("MYPYC_TYPE_ERROR:")
    assert first.diagnostic_scope == "unit"
    assert second.diagnostic_scope == "unit"
    assert second.attempt.phase_timings[0].name == "backend_decision_cache"
    assert decision.is_file()


def test_region_cache_restores_project_scoped_mypyc_rejection(tmp_path: Path) -> None:
    """Imported project diagnostics retain scope across the decision cache."""
    backend = _FakeBackend()
    backend.fail = True
    dependency = tmp_path / "src" / "pkg" / "dependency.py"
    dependency.parent.mkdir(parents=True)
    dependency.write_text("value: int = 'invalid'\n", encoding="utf-8")
    backend.failure_stderr = (
        "MYPYC_TYPE_ERROR: SystemExit(1)\n"
        "src/pkg/dependency.py:1: error: Incompatible types in assignment  [assignment]"
    )
    context = _context(tmp_path, project_source_digest="tree-a")
    unit = _unit(tmp_path)

    first = compile_with_region_cache(
        cast(CompilerBackend, backend),
        unit,
        context,
        cache_root=tmp_path / "cache",
    )
    second = compile_with_region_cache(
        cast(CompilerBackend, backend),
        unit,
        context,
        cache_root=tmp_path / "cache",
    )

    manifest = json.loads(
        (tmp_path / "cache" / "decisions" / "mypyc" / f"{backend.key}.json").read_text(
            encoding="utf-8"
        )
    )
    assert backend.compile_count == 1
    assert first.diagnostic_scope == "project"
    assert second.diagnostic_scope == "project"
    assert manifest["diagnostic_scope"] == "project"
    assert manifest["project_source_digest"] == "tree-a"


def test_region_cache_invalidates_project_rejection_when_source_digest_changes(
    tmp_path: Path,
) -> None:
    """A source snapshot change retries a formerly project-blocked backend."""
    backend = _FakeBackend()
    backend.fail = True
    dependency = tmp_path / "src" / "pkg" / "dependency.py"
    dependency.parent.mkdir(parents=True)
    dependency.write_text("value: int = 'invalid'\n", encoding="utf-8")
    backend.failure_stderr = (
        "MYPYC_TYPE_ERROR: SystemExit(1)\n"
        "src/pkg/dependency.py:1: error: Incompatible types in assignment  [assignment]"
    )
    unit = _unit(tmp_path)

    first = compile_with_region_cache(
        cast(CompilerBackend, backend),
        unit,
        _context(tmp_path, project_source_digest="tree-a"),
        cache_root=tmp_path / "cache",
    )
    second = compile_with_region_cache(
        cast(CompilerBackend, backend),
        unit,
        _context(tmp_path, project_source_digest="tree-b"),
        cache_root=tmp_path / "cache",
    )

    assert backend.compile_count == EXPECTED_DOUBLE_COMPILE_COUNT
    assert first.diagnostic_scope == "project"
    assert second.diagnostic_scope == "project"
    assert second.attempt.cache_status == "miss"


def test_region_cache_keeps_generated_mypyc_diagnostic_unit_scoped(tmp_path: Path) -> None:
    """An error in the submitted generated source cannot open a project circuit."""
    backend = _FakeBackend()
    backend.fail = True
    backend.failure_stderr = (
        "MYPYC_TYPE_ERROR: SystemExit(1)\n"
        f"{(tmp_path / 'generated.py').as_posix()}:1: error: Invalid generated code  [misc]"
    )
    unit = _unit(tmp_path)
    context = BackendCompileContext(
        project_root=tmp_path,
        build_dir=tmp_path / "native" / "build",
        source_roots=(tmp_path,),
    )

    result = compile_with_region_cache(
        cast(CompilerBackend, backend),
        unit,
        context,
        cache_root=tmp_path / "cache",
    )

    assert result.diagnostic_scope == "unit"


@pytest.mark.parametrize("generated_first", [True, False])
def test_region_cache_keeps_mixed_mypyc_diagnostics_unit_scoped(
    tmp_path: Path,
    *,
    generated_first: bool,
) -> None:
    """Any generated-unit error prevents propagation to the project circuit."""
    backend = _FakeBackend()
    backend.fail = True
    unit = _unit(tmp_path)
    dependency = tmp_path / "pkg" / "dependency.py"
    dependency.parent.mkdir()
    dependency.write_text("value: int = 'invalid'\n", encoding="utf-8")
    generated_error = f"{unit.source_paths[0]}:1: error: Invalid generated code  [misc]"
    project_error = f"{dependency}:1: error: Invalid imported code  [misc]"
    errors = (
        (generated_error, project_error) if generated_first else (project_error, generated_error)
    )
    backend.failure_stderr = "MYPYC_TYPE_ERROR: SystemExit(1)\n" + "\n".join(errors)
    context = BackendCompileContext(
        project_root=tmp_path,
        build_dir=tmp_path / "native" / "build",
        source_roots=(tmp_path,),
    )

    result = compile_with_region_cache(
        cast(CompilerBackend, backend),
        unit,
        context,
        cache_root=tmp_path / "cache",
    )

    assert result.diagnostic_scope == "unit"


def test_region_cache_invalidates_rejection_when_backend_fingerprint_changes(
    tmp_path: Path,
) -> None:
    """Generated source or toolchain fingerprint changes force a fresh attempt."""
    backend = _FakeBackend(key="first")
    backend.fail = True
    backend.failure_stderr = "MYPYC_TYPE_ERROR: deterministic"
    context = _context(tmp_path)
    unit = _unit(tmp_path)
    compile_with_region_cache(
        cast(CompilerBackend, backend),
        unit,
        context,
        cache_root=tmp_path / "cache",
    )
    backend.key = "second"

    result = compile_with_region_cache(
        cast(CompilerBackend, backend),
        unit,
        context,
        cache_root=tmp_path / "cache",
    )

    assert backend.compile_count == EXPECTED_DOUBLE_COMPILE_COUNT
    assert result.attempt.cache_status == "miss"


def test_region_cache_ignores_malformed_rejection_manifest(tmp_path: Path) -> None:
    """Malformed decision evidence cannot suppress a backend invocation."""
    backend = _FakeBackend()
    backend.fail = True
    backend.failure_stderr = "CYTHON_COMPILE_ERROR: deterministic"
    context = _context(tmp_path)
    unit = _unit(tmp_path)
    compile_with_region_cache(
        cast(CompilerBackend, backend),
        unit,
        context,
        cache_root=tmp_path / "cache",
    )
    decision = tmp_path / "cache" / "decisions" / "mypyc" / f"{backend.key}.json"
    decision.write_text('{"version": -1}', encoding="utf-8")

    result = compile_with_region_cache(
        cast(CompilerBackend, backend),
        unit,
        context,
        cache_root=tmp_path / "cache",
    )

    assert backend.compile_count == EXPECTED_DOUBLE_COMPILE_COUNT
    assert result.attempt.cache_status == "miss"


def test_region_cache_ignores_unknown_rejection_code(tmp_path: Path) -> None:
    """Only normalized deterministic backend codes may suppress compilation."""
    backend = _FakeBackend()
    backend.fail = True
    backend.failure_stderr = "MYPYC_TYPE_ERROR: deterministic"
    context = _context(tmp_path)
    unit = _unit(tmp_path)
    compile_with_region_cache(
        cast(CompilerBackend, backend),
        unit,
        context,
        cache_root=tmp_path / "cache",
    )
    decision = tmp_path / "cache" / "decisions" / "mypyc" / f"{backend.key}.json"
    manifest = cast(dict[str, object], json.loads(decision.read_text(encoding="utf-8")))
    manifest["diagnostic_code"] = "UNKNOWN_BUILD_ERROR"
    decision.write_text(json.dumps(manifest), encoding="utf-8")

    result = compile_with_region_cache(
        cast(CompilerBackend, backend),
        unit,
        context,
        cache_root=tmp_path / "cache",
    )

    assert backend.compile_count == EXPECTED_DOUBLE_COMPILE_COUNT
    assert result.attempt.cache_status == "miss"


def test_region_cache_reports_rejection_store_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cache write error never replaces the actual backend rejection."""
    backend = _FakeBackend()
    backend.fail = True
    backend.failure_stderr = "MYPYC_TYPE_ERROR: deterministic"
    context = _context(tmp_path)
    unit = _unit(tmp_path)
    original_replace = Path.replace

    def failing_replace(path: Path, target: Path) -> Path:
        if path.suffix == ".tmp":
            raise OSError("read-only cache")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", failing_replace)

    result = compile_with_region_cache(
        cast(CompilerBackend, backend),
        unit,
        context,
        cache_root=tmp_path / "cache",
    )

    detail = result.attempt.phase_timings[-1].detail
    assert result.attempt.success is False
    assert detail is not None
    assert "decision store failed" in detail


@pytest.mark.parametrize(
    "variant",
    [
        "non_object",
        "version",
        "files_type",
        "file_entry_type",
        "file_field_type",
        "unsafe_file_path",
        "artifacts_type",
        "record_field_type",
        "record_backend",
        "record_role",
    ],
)
def test_region_cache_treats_malformed_artifact_evidence_as_miss(
    tmp_path: Path,
    variant: str,
) -> None:
    """Invalid manifest field types cannot escape the cache boundary."""
    backend = _FakeBackend()
    context = _context(tmp_path)
    unit = _unit(tmp_path)
    compile_with_region_cache(
        cast(CompilerBackend, backend),
        unit,
        context,
        cache_root=tmp_path / "cache",
    )
    manifest_path = tmp_path / "cache" / "mypyc" / backend.key / "manifest.json"
    manifest = cast(dict[str, object], json.loads(manifest_path.read_text(encoding="utf-8")))
    manifest_path.write_text(
        json.dumps(_malformed_manifest(manifest, variant)),
        encoding="utf-8",
    )

    result = compile_with_region_cache(
        cast(CompilerBackend, backend),
        unit,
        context,
        cache_root=tmp_path / "cache",
    )

    assert result.attempt.cache_status == "miss"
    assert backend.compile_count == EXPECTED_DOUBLE_COMPILE_COUNT


def test_region_cache_separates_backend_fingerprints(tmp_path: Path) -> None:
    """Backend namespaces cannot restore another adapter's native artifact."""
    context = _context(tmp_path)
    unit = _unit(tmp_path)
    mypyc = _FakeBackend(name="mypyc", key="shared-key")
    cython = _FakeBackend(name="cython", key="shared-key")

    compile_with_region_cache(
        cast(CompilerBackend, mypyc),
        unit,
        context,
        cache_root=tmp_path / "cache",
    )
    result = compile_with_region_cache(
        cast(CompilerBackend, cython),
        unit,
        context,
        cache_root=tmp_path / "cache",
    )

    assert mypyc.compile_count == 1
    assert cython.compile_count == 1
    assert result.attempt.cache_status == "miss"


def test_region_cache_restores_support_artifact_role(tmp_path: Path) -> None:
    """Support artifact metadata survives a strict cache round trip."""
    backend = _FakeBackend()
    context = _context(tmp_path)
    unit = _unit(tmp_path)
    first = compile_with_region_cache(
        cast(CompilerBackend, backend),
        unit,
        context,
        cache_root=tmp_path / "cache",
    )
    manifest_path = tmp_path / "cache" / "mypyc" / backend.key / "manifest.json"
    manifest = cast(dict[str, object], json.loads(manifest_path.read_text(encoding="utf-8")))
    artifacts = cast(list[dict[str, object]], manifest["artifacts"])
    artifacts[0]["role"] = "support"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    first.attempt.artifact_paths[0].unlink()

    restored = compile_with_region_cache(
        cast(CompilerBackend, backend),
        unit,
        context,
        cache_root=tmp_path / "cache",
    )

    assert restored.attempt.cache_status == "hit"
    assert restored.artifacts[0].role == "support"


def test_region_cache_does_not_store_artifacts_outside_backend_output(tmp_path: Path) -> None:
    """A backend contract violation cannot copy arbitrary files into cache state."""
    backend = _FakeBackend()
    backend.write_outside_output_root = True
    context = _context(tmp_path)
    unit = _unit(tmp_path)

    first = compile_with_region_cache(
        cast(CompilerBackend, backend),
        unit,
        context,
        cache_root=tmp_path / "cache",
    )
    second = compile_with_region_cache(
        cast(CompilerBackend, backend),
        unit,
        context,
        cache_root=tmp_path / "cache",
    )

    assert backend.compile_count == EXPECTED_DOUBLE_COMPILE_COUNT
    store_detail = first.attempt.phase_timings[-1].detail
    assert store_detail is not None
    assert "outside the backend output root" in store_detail
    assert second.attempt.cache_status == "miss"


def test_region_cache_storage_failure_does_not_fail_compilation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache I/O failure remains diagnostic evidence, not a compiler failure."""
    backend = _FakeBackend()
    context = _context(tmp_path)
    unit = _unit(tmp_path)
    original_write_text = Path.write_text

    def failing_manifest_write(path: Path, data: str, *, encoding: str | None = None) -> int:
        if path.name == "manifest.json":
            raise OSError("cache unavailable")
        return original_write_text(path, data, encoding=encoding)

    monkeypatch.setattr(Path, "write_text", failing_manifest_write)

    result = compile_with_region_cache(
        cast(CompilerBackend, backend),
        unit,
        context,
        cache_root=tmp_path / "cache",
    )

    assert result.attempt.success is True
    store_detail = result.attempt.phase_timings[-1].detail
    assert store_detail is not None
    assert "store failed: cache unavailable" in store_detail
    assert not (tmp_path / "cache" / "mypyc" / backend.key).exists()


def _malformed_manifest(manifest: dict[str, object], variant: str) -> object:
    if variant == "non_object":
        return []
    if variant == "version":
        manifest["version"] = -1
    elif variant == "files_type":
        manifest["files"] = "invalid"
    elif variant == "file_entry_type":
        manifest["files"] = [7]
    elif variant in {"file_field_type", "unsafe_file_path"}:
        files = cast(list[dict[str, object]], manifest["files"])
        files[0]["digest" if variant == "file_field_type" else "path"] = (
            7 if variant == "file_field_type" else "../escape.so"
        )
    elif variant == "artifacts_type":
        manifest["artifacts"] = "invalid"
    else:
        records = cast(list[dict[str, object]], manifest["artifacts"])
        field = {
            "record_field_type": "region_id",
            "record_backend": "backend",
            "record_role": "role",
        }[variant]
        records[0][field] = 7 if variant == "record_field_type" else "invalid"
    return manifest


def _context(
    root: Path,
    *,
    project_source_digest: str | None = None,
) -> BackendCompileContext:
    return BackendCompileContext(
        project_root=root,
        build_dir=root / "native" / "build",
        source_roots=(root / "src",),
        project_source_digest=project_source_digest,
    )


def _unit(root: Path) -> CompilationUnit:
    source = root / "generated.py"
    source.write_text("def value() -> int:\n    return 1\n", encoding="utf-8")
    return CompilationUnit(
        region_id="pkg.module::value@backend",
        backend="mypyc",
        logical_module="_atoll_value",
        source_paths=(source,),
        source_hash=hashlib.sha256(source.read_bytes()).hexdigest(),
        members=(SymbolId(module="pkg.module", qualname="value"),),
        install_relative_dir=".atoll/artifacts/region",
    )


def _units(root: Path, count: int) -> tuple[CompilationUnit, ...]:
    units: list[CompilationUnit] = []
    for index in range(count):
        source = root / f"generated_{index}.py"
        source.write_text(f"def value_{index}() -> int:\n    return {index}\n", encoding="utf-8")
        units.append(
            CompilationUnit(
                region_id=f"pkg.module::value_{index}@cython",
                backend="cython",
                logical_module=f"_atoll_value_{index}",
                source_paths=(source,),
                source_hash=hashlib.sha256(source.read_bytes()).hexdigest(),
                members=(SymbolId(module="pkg.module", qualname=f"value_{index}"),),
                install_relative_dir=f".atoll/artifacts/region-{index}",
            )
        )
    return tuple(units)
