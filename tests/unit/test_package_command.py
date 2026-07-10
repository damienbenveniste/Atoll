"""Unit tests for source-clean package artifact helpers."""

from __future__ import annotations

import hashlib
import importlib.machinery
import shutil
import zipfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Protocol, cast

import pytest

from atoll import cli as cli_module
from atoll.commands import package as package_command
from atoll.generation.region_shim import RegionShimConfig
from atoll.models import (
    ArtifactRecord,
    ArtifactRole,
    Backend,
    BackendAssessment,
    BackendCompileContext,
    BackendCompileResult,
    BindingTarget,
    Blocker,
    CompilationUnit,
    CompileAttempt,
    CompiledRegionVariant,
    CompilePhaseTiming,
    EnabledIslandConfig,
    LoweringDecision,
    LoweringMode,
    ModuleId,
    ModuleScan,
    RegionSpecialization,
    SymbolId,
    TypedRegion,
)
from atoll.project import DiscoveredProject, discover_project
from atoll.report import CompilationReportInput, build_compilation_report
from atoll.runtime.package_verify import (
    PackageVerificationPlan,
    PackageVerificationResult,
    VerificationStage,
)
from atoll.runtime.performance import (
    BenchmarkGateConfig,
    BenchmarkGateResult,
    BenchmarkProgress,
    BenchmarkStatus,
    CommandRunEvidence,
    RuntimeMode,
)
from atoll.runtime.profiling import (
    LifecycleCounts,
    ProfiledMember,
    ProfileResult,
    unconfigured_profile,
)

FIXTURE_ROOT = Path("tests/fixtures/simple_project")
TYPED_FIXTURE_ROOT = Path("tests/fixtures/typed_region_project")
EXPECTED_ATOMIC_SELECTION_COUNT = 2
TEST_FAILURE_RETURN_CODE = 9
RANKING_BINDING_COUNT = 3
OUTLINED_COMPILE_CALL_COUNT = 2
_CANDIDATE_SPEEDUP = 1.01
EXPECTED_FINAL_TEST_RESULTS = 2


@pytest.fixture(autouse=True)
def stub_native_subprocess_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    """Package orchestration tests stop at the separately tested verification boundary."""

    def verify(**kwargs: object) -> PackageVerificationResult:
        stage = cast(VerificationStage, kwargs["stage"])
        target = cast(Path, kwargs["target"])
        return PackageVerificationResult(
            stage=stage,
            target=target,
            command=("python", "verify"),
            success=True,
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=0.0,
        )

    monkeypatch.setattr(package_command, "verify_package_subprocess", verify)


class _Metadata(Protocol):
    name: str
    version: str
    requires_python: str | None
    dependencies: tuple[str, ...]


class _BaselinePayloadFactory(Protocol):
    def __call__(
        self,
        *,
        wheel_path: Path | None,
        build: CompileAttempt,
        baseline_install_root: Path | None = None,
        quality_project_root: Path | None = None,
        semantic_test_result: CommandRunEvidence | None = None,
    ) -> object: ...


class _QualityGateOutcomeView(Protocol):
    success: bool
    tests: tuple[CommandRunEvidence, ...]
    performance: BenchmarkGateResult
    error: str | None


class _PromotionResultView(Protocol):
    success: bool
    wheel_path: Path | None
    error: str | None


class _SelectedScans(Protocol):
    def __call__(
        self,
        project: DiscoveredProject,
        module_name: str | None,
        selected_members: tuple[SymbolId, ...] = (),
    ) -> tuple[ModuleScan, ...]: ...


class _TypedSelection(Protocol):
    scan: ModuleScan
    backend: Backend
    variant_id: str
    region: TypedRegion
    assessment: BackendAssessment
    members: tuple[SymbolId, ...]
    bound_members: tuple[SymbolId, ...] | None
    specialization: RegionSpecialization | None
    conditional_on_failure_of: str | None
    source_region_id: str | None
    slice_root: SymbolId | None


class _TypedGeneration(Protocol):
    backend: Backend
    region: TypedRegion
    bindings: tuple[BindingTarget, ...]


class _PreparedTypedRegion(Protocol):
    generation: _TypedGeneration
    assessment: BackendAssessment
    unit: CompilationUnit
    fallback: _PreparedTypedRegion | None
    conditional_on_failure_of: str | None
    lowering_mode: LoweringMode
    native_helpers: tuple[str, ...]
    fallback_reason: str | None
    shim: RegionShimConfig


class _TypedRegionOutcome(Protocol):
    successful: tuple[_PreparedTypedRegion, ...]
    build: CompileAttempt
    artifacts: tuple[ArtifactRecord, ...]
    skipped: tuple[_TypedRegionFailure, ...]


class _TypedRegionFailure(Protocol):
    variant_id: str


class _FakeCompileBackend:
    """Backend stub used to force deterministic retry orchestration."""

    def __init__(self, result: BackendCompileResult) -> None:
        self.result = result
        self.calls: list[tuple[CompilationUnit, ...]] = []
        self.name = cast(Backend, result.attempt.command[0])

    def fingerprint(
        self,
        unit: CompilationUnit,
        context: BackendCompileContext,
    ) -> str:
        """Return a stable per-variant key for cache orchestration tests."""
        _ = context
        return hashlib.sha256(
            f"{self.name}:{unit.region_id}:{unit.source_hash}".encode()
        ).hexdigest()

    def compile(
        self,
        units: tuple[CompilationUnit, ...],
        context: BackendCompileContext,
    ) -> BackendCompileResult:
        """Record one invocation and return configured compiler evidence."""
        _ = context
        self.calls.append(units)
        return self.result


class _SequencedCompileBackend:
    """Backend stub returning one configured result per distinct fallback attempt."""

    def __init__(
        self,
        name: Backend,
        results: tuple[BackendCompileResult, ...],
    ) -> None:
        self.name = name
        self.results = results
        self.calls: list[tuple[CompilationUnit, ...]] = []

    def fingerprint(
        self,
        unit: CompilationUnit,
        context: BackendCompileContext,
    ) -> str:
        """Return a stable key that distinguishes whole and outlined units."""
        _ = context
        return hashlib.sha256(
            f"{self.name}:{unit.region_id}:{unit.source_hash}".encode()
        ).hexdigest()

    def compile(
        self,
        units: tuple[CompilationUnit, ...],
        context: BackendCompileContext,
    ) -> BackendCompileResult:
        """Return the next result and reject unexpected additional invocations."""
        _ = context
        index = len(self.calls)
        if index >= len(self.results):
            raise AssertionError("native backend received an unexpected compile invocation")
        self.calls.append(units)
        return self.results[index]


def _package_attr(name: str) -> object:
    return vars(package_command)[name]


_copy_atoll_artifacts = cast(
    Callable[[tuple[Path, ...], Path], None],
    _package_attr("_copy_atoll_artifacts"),
)
_copy_if_different = cast(
    Callable[[Path, Path], None],
    _package_attr("_copy_if_different"),
)
_copy_source_roots = cast(
    Callable[[DiscoveredProject, Path], tuple[Path, ...]],
    _package_attr("_copy_source_roots"),
)
_find_module = cast(
    Callable[[tuple[ModuleId, ...], str], ModuleId],
    _package_attr("_find_module"),
)
_mapping = cast(
    Callable[[object], dict[str, object]],
    _package_attr("_mapping"),
)
_project_metadata = cast(
    Callable[[Path], _Metadata],
    _package_attr("_project_metadata"),
)
_relative_source_root = cast(
    Callable[[Path, Path], Path],
    _package_attr("_relative_source_root"),
)
_reset_dir = cast(Callable[[Path], None], _package_attr("_reset_dir"))
_sequence = cast(
    Callable[[object], tuple[object, ...]],
    _package_attr("_sequence"),
)
_staged_module = cast(
    Callable[[ModuleId, DiscoveredProject, tuple[Path, ...]], ModuleId],
    _package_attr("_staged_module"),
)
_string = cast(Callable[[object], str | None], _package_attr("_string"))
_selected_scans = cast(
    _SelectedScans,
    _package_attr("_selected_scans"),
)
_selected_typed_regions = cast(
    Callable[..., tuple[_TypedSelection, ...]],
    _package_attr("_selected_typed_regions"),
)
_prepare_typed_region = cast(
    Callable[..., _PreparedTypedRegion], _package_attr("_prepare_typed_region")
)
_artifact_records_for_prepared = cast(
    Callable[
        [tuple[_PreparedTypedRegion, ...], tuple[ArtifactRecord, ...]],
        tuple[ArtifactRecord, ...],
    ],
    _package_attr("_artifact_records_for_prepared"),
)
_materialize_profitable_payload = cast(
    Callable[..., str | None],
    _package_attr("_materialize_profitable_payload"),
)
_SelectedTypedRegion = cast(Callable[..., _TypedSelection], _package_attr("_SelectedTypedRegion"))
_RequestedCallableVariant = cast(Callable[..., object], _package_attr("_RequestedCallableVariant"))
_staged_typed_selection = cast(
    Callable[[ModuleScan, _TypedSelection], _TypedSelection],
    _package_attr("_staged_typed_selection"),
)
_profile_candidate_members = cast(
    Callable[[tuple[ModuleScan, ...], tuple[Backend, ...]], tuple[SymbolId, ...]],
    _package_attr("_profile_candidate_members"),
)
_runtime_member_closure = cast(
    Callable[[TypedRegion, tuple[SymbolId, ...], frozenset[SymbolId]], tuple[SymbolId, ...]],
    _package_attr("_runtime_member_closure"),
)
_selected_requested_callable_variant = cast(
    Callable[[object], tuple[_TypedSelection, ...]],
    _package_attr("_selected_requested_callable_variant"),
)
_build_typed_regions = cast(
    Callable[..., _TypedRegionOutcome], _package_attr("_build_typed_regions")
)
_TypedRegionBuildContext = cast(Callable[..., object], _package_attr("_TypedRegionBuildContext"))
_TypedRegionBuildOutcome = cast(Callable[..., object], _package_attr("_TypedRegionBuildOutcome"))
_compiler_backends = cast(dict[Backend, object], _package_attr("_COMPILER_BACKENDS"))
_member_requires_source_class = cast(
    Callable[[str], bool],
    _package_attr("_member_requires_source_class"),
)
_owner_disallows_method_binding = cast(
    Callable[[str | None, str, dict[str, LoweringDecision]], bool],
    _package_attr("_owner_disallows_method_binding"),
)
_BaselineWheelPayload = cast(
    _BaselinePayloadFactory,
    _package_attr("_BaselineWheelPayload"),
)
_run_configured_quality_gate = cast(
    Callable[..., _QualityGateOutcomeView],
    _package_attr("_run_configured_quality_gate"),
)
_QualityGateOutcome = cast(
    Callable[..., _QualityGateOutcomeView],
    _package_attr("_QualityGateOutcome"),
)
_SourceCleanPromotionContext = cast(
    Callable[..., object],
    _package_attr("_SourceCleanPromotionContext"),
)
_promote_source_clean_payload = cast(
    Callable[[object], _PromotionResultView],
    _package_attr("_promote_source_clean_payload"),
)
_print_source_clean_success = cast(
    Callable[..., None],
    vars(cli_module)["_print_source_clean_success"],
)


def test_typed_region_selection_prefers_mypyc_for_safe_specializations(
    tmp_path: Path,
) -> None:
    """Subclass and closed-call specializations enter the normal automatic routing path."""
    project_root = tmp_path / "typed_region_project"
    shutil.copytree(TYPED_FIXTURE_ROOT, project_root)
    project = discover_project(project_root)

    worker_selections = _selected_typed_regions(
        _selected_scans(project, "typed_region_project.worker")
    )
    worker_specializations = tuple(
        selection.specialization
        for selection in worker_selections
        if selection.specialization is not None
    )
    function_selections = _selected_typed_regions(
        _selected_scans(project, "typed_region_project.generic_functions")
    )

    assert {
        (specialization.origin, specialization.target_owner_class)
        for specialization in worker_specializations
    } == {
        ("concrete_subclass", "IntPairer"),
        ("concrete_subclass", "PayloadPairer"),
    }
    assert all(
        selection.backend == "mypyc"
        for selection in worker_selections
        if selection.specialization is not None
    )
    assert len(function_selections) == EXPECTED_ATOMIC_SELECTION_COUNT
    ordinary = next(
        selection for selection in function_selections if selection.specialization is None
    )
    assert ordinary.backend == "mypyc"
    assert tuple(member.qualname for member in ordinary.members) == ("pair_int",)
    function_specialization = next(
        selection for selection in function_selections if selection.specialization is not None
    )
    assert function_specialization.backend == "mypyc"
    assert function_specialization.specialization is not None
    assert function_specialization.specialization.origin == "closed_call"


def test_explicit_function_selection_creates_one_directed_slice_per_binding(
    tmp_path: Path,
) -> None:
    """Independent requested roots do not inherit an oversized connected component."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    requested = (
        SymbolId(module="app.ranking", qualname="score_user"),
        SymbolId(module="app.ranking", qualname="rank_candidates"),
    )

    selections = _selected_typed_regions(
        _selected_scans(project, "app.ranking"),
        ("mypyc", "cython"),
        requested,
    )

    assert len(selections) == EXPECTED_ATOMIC_SELECTION_COUNT
    assert all(selection.backend == "mypyc" for selection in selections)
    assert {tuple(member.qualname for member in selection.members) for selection in selections} == {
        ("score_user",),
        ("rank_candidates",),
    }
    assert {
        member for selection in selections for member in (selection.bound_members or ())
    } == set(requested)
    assert all(selection.slice_root is not None for selection in selections)
    assert all(selection.source_region_id is not None for selection in selections)


def test_directed_selection_rejects_staged_drift_and_missing_root(tmp_path: Path) -> None:
    """Staged rescanning must reproduce the exact checkout-derived slice."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    scans = _selected_scans(project, "app.ranking")
    requested = (SymbolId("app.ranking", "score_user"),)
    selection = _selected_typed_regions(scans, requested_members=requested)[0]
    assert selection.source_region_id is not None
    assert selection.slice_root is not None

    missing_root = _SelectedTypedRegion(
        scan=selection.scan,
        region=selection.region,
        variant_id=selection.variant_id,
        backend=selection.backend,
        assessment=selection.assessment,
        members=selection.members,
        bound_members=selection.bound_members,
        specialization=selection.specialization,
        conditional_on_failure_of=selection.conditional_on_failure_of,
        source_region_id=selection.source_region_id,
        slice_root=None,
    )
    with pytest.raises(ValueError, match="requires a slice root"):
        _staged_typed_selection(scans[0], missing_root)

    drifted = _SelectedTypedRegion(
        scan=selection.scan,
        region=replace(selection.region, id=f"{selection.region.id}:drifted"),
        variant_id=selection.variant_id,
        backend=selection.backend,
        assessment=selection.assessment,
        members=selection.members,
        bound_members=selection.bound_members,
        specialization=selection.specialization,
        conditional_on_failure_of=selection.conditional_on_failure_of,
        source_region_id=selection.source_region_id,
        slice_root=selection.slice_root,
    )
    with pytest.raises(ValueError, match="staged directed slice differs"):
        _staged_typed_selection(scans[0], drifted)


def test_profile_and_directed_closure_helpers_cover_backend_boundaries(tmp_path: Path) -> None:
    """Profile preflight and required-edge closure remain conservative."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    scans = _selected_scans(project, "app.ranking")
    assert _profile_candidate_members(scans, ()) == ()

    region = next(
        region
        for region in scans[0].typed_regions
        if any(
            isinstance(dependency.dst, SymbolId)
            and dependency.src
            in {member.id for member in region.members if member.kind == "function"}
            and dependency.dst
            in {member.id for member in region.members if member.kind == "function"}
            for dependency in region.dependencies
        )
    )
    member_ids = tuple(member.id for member in region.members if member.kind == "function")
    dependency = next(
        dependency
        for dependency in region.dependencies
        if dependency.src in member_ids
        and isinstance(dependency.dst, SymbolId)
        and dependency.dst in member_ids
    )
    assert isinstance(dependency.dst, SymbolId)
    required = replace(
        region,
        dependencies=tuple(
            replace(item, requires_same_unit=True) if item is dependency else item
            for item in region.dependencies
        ),
    )
    closure = _runtime_member_closure(required, member_ids, frozenset({dependency.src}))
    assert {dependency.src, dependency.dst} <= set(closure)

    blocked_closure = _runtime_member_closure(
        required,
        tuple(member for member in member_ids if member != dependency.dst),
        frozenset({dependency.src}),
    )
    assert blocked_closure == ()

    external = replace(
        required,
        dependencies=tuple(
            replace(item, dst="external.boundary")
            if item.src == dependency.src and item.dst == dependency.dst and item.requires_same_unit
            else item
            for item in required.dependencies
        ),
    )
    assert _runtime_member_closure(
        external,
        member_ids,
        frozenset({dependency.src}),
    ) == (dependency.src,)

    empty_inputs = _RequestedCallableVariant(
        scan=scans[0],
        region=region,
        closure=(),
        requested=frozenset({dependency.src}),
        backends=("mypyc", "cython"),
        source_region_id=region.id,
        slice_root=dependency.src,
    )
    unsupported_inputs = _RequestedCallableVariant(
        scan=scans[0],
        region=region,
        closure=(dependency.src,),
        requested=frozenset({dependency.src}),
        backends=(),
        source_region_id=region.id,
        slice_root=dependency.src,
    )
    assert _selected_requested_callable_variant(empty_inputs) == ()
    assert _selected_requested_callable_variant(unsupported_inputs) == ()


def test_selected_scans_reject_cross_module_member_scope(tmp_path: Path) -> None:
    """A module-filtered compile cannot smuggle in a member from another module."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    project = discover_project(project_root)

    with pytest.raises(ValueError, match="must belong to the requested module scope"):
        _selected_scans(
            project,
            "app.ranking",
            (SymbolId("app.models", "User"),),
        )


def test_explicit_package_fails_when_one_requested_region_does_not_compile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial region build cannot promote a wheel that promised explicit members."""
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    ranking_source = project_root / "src" / "app" / "ranking.py"
    (project_root / "src" / "app" / "extra.py").write_text(
        ranking_source.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    requested = (
        SymbolId(module="app.ranking", qualname="normalize_features"),
        SymbolId(module="app.extra", qualname="normalize_features"),
    )

    def partial_build(**kwargs: object) -> object:
        prepared = cast(tuple[_PreparedTypedRegion, ...], kwargs["prepared"])
        assert len(prepared) == EXPECTED_ATOMIC_SELECTION_COUNT
        return _TypedRegionBuildOutcome(
            successful=(prepared[0],),
            build=_successful_attempt(),
            artifacts=(),
            skipped=(),
        )

    monkeypatch.setattr(package_command, "_build_typed_regions", partial_build)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            output_dir=output_dir,
            selected_members=requested,
        )
    )

    assert result.success is False
    assert result.wheel_path is None
    assert result.error == (
        "requested member(s) did not compile successfully: app.extra::normalize_features"
    )
    assert result.cleanup_removed == (output_dir / "install",)
    assert result.cleanup_kept == (output_dir / "build",)


def test_function_with_same_region_class_dependency_uses_runtime_boundary(
    tmp_path: Path,
) -> None:
    """A local class can remain interpreted while its caller compiles."""
    project_root = tmp_path / "simple_project"
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
    project = discover_project(project_root)

    selections = _selected_typed_regions(_selected_scans(project, "app.class_dependency"))

    bound = {
        member
        for selection in selections
        for member in (selection.bound_members or selection.members)
    }
    assert SymbolId(module="app.class_dependency", qualname="add_one") in bound
    assert SymbolId(module="app.class_dependency", qualname="make_payload") in bound


def test_atomic_class_selection_is_exclusive_and_partial_classes_split(
    tmp_path: Path,
) -> None:
    """A closed class has one class variant while mixed shapes remain per-member."""
    project_root = tmp_path / "typed_region_project"
    shutil.copytree(TYPED_FIXTURE_ROOT, project_root)
    project = discover_project(project_root)

    selections = _selected_typed_regions(_selected_scans(project, "typed_region_project.worker"))
    scale_selections = tuple(
        selection
        for selection in selections
        if any(member.qualname.startswith("ScaleModel") for member in selection.members)
    )
    worker_selections = tuple(
        selection
        for selection in selections
        if any(member.qualname.startswith("Worker") for member in selection.members)
    )

    class_selection = next(
        selection
        for selection in scale_selections
        if selection.variant_id.endswith("@cython-class")
    )
    method_fallback = next(
        selection for selection in scale_selections if selection is not class_selection
    )
    assert len(scale_selections) == EXPECTED_ATOMIC_SELECTION_COUNT
    assert class_selection.backend == "cython"
    assert tuple(member.qualname for member in class_selection.members) == ("ScaleModel",)
    assert method_fallback.backend == "mypyc"
    assert {member.qualname for member in method_fallback.members} == {
        "ScaleModel.apply",
        "ScaleModel.describe",
    }
    assert method_fallback.conditional_on_failure_of == class_selection.variant_id
    assert {member.qualname for selection in worker_selections for member in selection.members} == {
        "Worker.adjust",
        "Worker.exchange",
        "Worker.parse",
        "Worker.scale",
        "Worker.score",
        "Worker.values",
    }
    assert {
        selection.backend
        for selection in worker_selections
        if any(member.qualname == "Worker.exchange" for member in selection.members)
    } == {"cython"}
    assert all(
        member.qualname not in {"Worker", "Worker.__init__"}
        for selection in worker_selections
        for member in selection.members
    )


def test_method_selection_rejects_class_cell_and_private_name_semantics() -> None:
    """Top-level extraction never guesses class-cell or name-mangling behavior."""
    assert _member_requires_source_class("def value(self) -> int:\n    return super().value()\n")
    assert _member_requires_source_class("def owner(self) -> type[object]:\n    return __class__\n")
    assert _member_requires_source_class("def secret(self) -> int:\n    return self.__secret\n")
    assert not _member_requires_source_class(
        "def regular(self) -> int:\n    return len(self.__dict__)\n"
    )


def test_method_selection_preserves_registered_and_dynamic_owner_classes() -> None:
    """Method mutation is rejected when an owner may be replaced or intercept writes."""
    registered = LoweringDecision(
        target="module::Registered",
        action="fallback",
        reason="class remains interpreted because decorators may register or replace it",
    )
    eager = LoweringDecision(
        target="module::Eager",
        action="fallback",
        reason="class remains interpreted because module-time code retains its original identity",
    )

    assert _owner_disallows_method_binding(
        "Registered",
        "module",
        {registered.target: registered},
    )
    assert not _owner_disallows_method_binding(
        "Eager",
        "module",
        {eager.target: eager},
    )


def test_profile_selected_dataclass_methods_use_boxed_cython_slices(
    tmp_path: Path,
) -> None:
    """Hot boxed methods rebind safely on the original recognized dataclass."""
    project_root = tmp_path / "simple_project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    module_path = project_root / "src" / "app" / "boxed_runner.py"
    module_path.write_text(
        """from __future__ import annotations

import typing
from dataclasses import dataclass


@dataclass
class Runner:
    bias: int

    def dynamic(self, value: typing.Any) -> typing.Any:
        return value

    def incomplete(self, value):
        return value + self.bias

    def identity[T](self, value: T) -> T:
        return value
""",
        encoding="utf-8",
    )
    project = discover_project(project_root)
    scans = _selected_scans(project, "app.boxed_runner")
    hot = tuple(
        SymbolId("app.boxed_runner", qualname)
        for qualname in ("Runner.dynamic", "Runner.incomplete", "Runner.identity")
    )

    static = _selected_typed_regions(scans)
    selected = _selected_typed_regions(
        scans,
        ("mypyc", "cython"),
        hot,
        hot_members=hot,
    )

    assert static == ()
    assert len(selected) == len(hot)
    assert all(selection.backend == "cython" for selection in selected)
    assert {selection.slice_root for selection in selected} == set(hot)
    assert all(len(selection.members) == 1 for selection in selected)
    assert all(selection.region.atomic_class is False for selection in selected)


def test_package_default_does_not_call_legacy_sidecar_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Source-clean package compilation bypasses the legacy sidecar facade."""
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    source_path = project_root / "src" / "app" / "ranking.py"
    original_source = source_path.read_text(encoding="utf-8")

    def failing_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        assert args
        assert kwargs
        raise AssertionError("legacy sidecar backend was invoked")

    monkeypatch.setattr(package_command, "build_sidecars", failing_build_sidecars, raising=False)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
        )
    )

    assert result.success is True
    assert result.error is None
    assert result.wheel_path is not None
    assert source_path.read_text(encoding="utf-8") == original_source
    assert not (output_dir / "build").exists()
    assert not (output_dir / "install").exists()
    assert result.islands == ()
    assert len(result.compiled_bindings) == RANKING_BINDING_COUNT


@pytest.mark.parametrize("failed_stage", ["payload", "wheel"])
def test_package_preserves_payload_for_subprocess_verification_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_stage: VerificationStage,
) -> None:
    """Routing failures keep the exact payload and rejected wheel under diagnostics."""
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]

    def successful_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        assert kwargs
        path = cast(tuple[Path, ...], args[0])[0]
        artifact = tmp_path / f"{path.stem}{suffix}"
        artifact.write_text("binary", encoding="utf-8")
        return CompileAttempt(
            success=True,
            command=("mypyc",),
            stdout="",
            stderr="",
            artifact_paths=(artifact,),
            duration_seconds=0.1,
        )

    def verify(**kwargs: object) -> PackageVerificationResult:
        stage = cast(VerificationStage, kwargs["stage"])
        target = cast(Path, kwargs["target"])
        success = stage != failed_stage
        return PackageVerificationResult(
            stage=stage,
            target=target,
            command=("python", "verify"),
            success=success,
            exit_code=0 if success else 1,
            stdout="",
            stderr="routing failed" if not success else "",
            duration_seconds=0.1,
        )

    monkeypatch.setattr(package_command, "build_sidecars", successful_build_sidecars, raising=False)
    monkeypatch.setattr(package_command, "verify_package_subprocess", verify)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
        )
    )

    assert result.success is False
    assert result.error == "routing failed"
    assert result.cleanup_removed == ()
    assert result.cleanup_kept == (output_dir / "build", output_dir / "install")
    assert (output_dir / "install").exists()
    assert not tuple(output_dir.glob("*.whl"))
    assert result.verification_steps[-1].stage == failed_stage
    if failed_stage == "wheel":
        assert result.verification_steps[-1].target.parent == output_dir / "build" / "diagnostics"


def _prepare_outlined_coroutine_fixture(
    tmp_path: Path,
) -> tuple[_PreparedTypedRegion, Path, tuple[Path, ...]]:
    project_root = tmp_path / "typed_region_project"
    build_root = tmp_path / "build"
    shutil.copytree(TYPED_FIXTURE_ROOT, project_root)
    module_path = project_root / "src" / "typed_region_project" / "outline_worker.py"
    module_path.write_text(
        """async def checkpoint() -> None:
    return None


async def hot(values: list[int]) -> int:
    start = len(values) + 1
    doubled = start * 2
    total = doubled + 3
    await checkpoint()
    return total
""",
        encoding="utf-8",
    )
    project = discover_project(project_root)
    scans = _selected_scans(project, "typed_region_project.outline_worker")
    hot = SymbolId("typed_region_project.outline_worker", "hot")
    selections = _selected_typed_regions(
        scans,
        ("mypyc", "cython"),
        (hot,),
        hot_members=(hot,),
    )
    assert len(selections) == 1
    staged_source_roots = _copy_source_roots(project, build_root)
    prepared = _prepare_typed_region(
        project=project,
        build_root=build_root,
        staged_source_roots=staged_source_roots,
        selection=selections[0],
    )
    return prepared, build_root, staged_source_roots


def test_prepare_typed_region_appends_outlined_cython_fallback(tmp_path: Path) -> None:
    """A precise async root prepares whole-callable and outlined backend variants."""
    prepared, _build_root, _staged_source_roots = _prepare_outlined_coroutine_fixture(tmp_path)

    assert prepared.generation.backend == "mypyc"
    assert prepared.fallback is not None
    assert prepared.fallback.lowering_mode == "whole-callable"
    assert prepared.fallback.fallback is not None
    outlined = prepared.fallback.fallback
    assert outlined.generation.backend == "cython"
    assert outlined.lowering_mode == "outlined-block"
    assert outlined.native_helpers
    assert outlined.unit.region_id.endswith("@cython-outline")


def test_artifact_filter_keeps_only_accepted_region_support_files(tmp_path: Path) -> None:
    """Shared support records follow their collision-resistant variant directory."""
    accepted, _build_root, _staged_source_roots = _prepare_outlined_coroutine_fixture(tmp_path)
    assert accepted.fallback is not None
    rejected = accepted.fallback
    digest = "0" * 64

    def record(
        prepared: _PreparedTypedRegion,
        *,
        region_id: str,
        role: ArtifactRole,
        filename: str,
    ) -> ArtifactRecord:
        return ArtifactRecord(
            region_id=region_id,
            backend=prepared.generation.backend,
            logical_module=prepared.unit.logical_module,
            role=role,
            install_relative_path=f"{prepared.unit.install_relative_dir}/{filename}",
            digest=digest,
            abi="cp312",
            platform_tag="test-platform",
        )

    accepted_primary = record(
        accepted,
        region_id=accepted.unit.region_id,
        role="primary",
        filename="accepted.so",
    )
    accepted_support = record(
        accepted,
        region_id="__shared__",
        role="support",
        filename="accepted-support.so",
    )
    rejected_primary = record(
        rejected,
        region_id=rejected.unit.region_id,
        role="primary",
        filename="rejected.so",
    )
    rejected_support = record(
        rejected,
        region_id="__shared__",
        role="support",
        filename="rejected-support.so",
    )

    filtered = _artifact_records_for_prepared(
        (accepted,),
        (accepted_primary, accepted_support, rejected_primary, rejected_support),
    )

    assert filtered == (accepted_primary, accepted_support)


def test_rejected_module_keeps_baseline_wheel_source_bytes(tmp_path: Path) -> None:
    """A module with no accepted candidate is not overlaid from the copied checkout."""
    rejected, build_root, staged_source_roots = _prepare_outlined_coroutine_fixture(tmp_path)
    rejected.shim.source_path.write_text("staged checkout bytes\n", encoding="utf-8")
    baseline_root = build_root / "baseline-install"
    baseline_module = baseline_root / "typed_region_project" / "outline_worker.py"
    baseline_module.parent.mkdir(parents=True)
    baseline_module.write_text("baseline wheel bytes\n", encoding="utf-8")
    install_root = tmp_path / "install"

    error = _materialize_profitable_payload(
        baseline=_BaselineWheelPayload(
            wheel_path=build_root / "baseline.whl",
            build=_successful_attempt(),
            baseline_install_root=baseline_root,
        ),
        staged_source_roots=staged_source_roots,
        install_root=install_root,
        superset=(rejected,),
        accepted=(),
    )

    assert error is None
    assert (install_root / "typed_region_project" / "outline_worker.py").read_text(
        encoding="utf-8"
    ) == "baseline wheel bytes\n"


def test_typed_region_build_retries_whole_callable_failure_with_outline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deterministic backend failures continue through the outlined Cython variant."""
    prepared, build_root, staged_source_roots = _prepare_outlined_coroutine_fixture(tmp_path)
    mypyc = _FakeCompileBackend(
        BackendCompileResult(
            attempt=CompileAttempt(
                success=False,
                command=("mypyc",),
                stdout="",
                stderr="MYPYC_TYPE_ERROR: fixture",
                artifact_paths=(),
                duration_seconds=0.1,
            ),
            artifacts=(),
        )
    )
    cython = _SequencedCompileBackend(
        "cython",
        (
            BackendCompileResult(
                attempt=CompileAttempt(
                    success=False,
                    command=("cython",),
                    stdout="",
                    stderr="CYTHON_COMPILE_ERROR: whole callable fixture",
                    artifact_paths=(),
                    duration_seconds=0.2,
                ),
                artifacts=(),
            ),
            BackendCompileResult(
                attempt=CompileAttempt(
                    success=True,
                    command=("cython",),
                    stdout="",
                    stderr="",
                    artifact_paths=(),
                    duration_seconds=0.2,
                ),
                artifacts=(),
            ),
        ),
    )
    monkeypatch.setitem(_compiler_backends, "mypyc", mypyc)
    monkeypatch.setitem(_compiler_backends, "cython", cython)

    outcome = _build_typed_regions(
        prepared=(prepared,),
        context=_TypedRegionBuildContext(
            build_root=build_root,
            staged_source_roots=staged_source_roots,
            mypy_cache_dir=tmp_path / "mypy-cache",
            compile_cache_dir=tmp_path / "compile-cache",
            progress=None,
        ),
        initial_failures=(),
    )

    assert outcome.build.success is True
    assert outcome.skipped == ()
    assert len(mypyc.calls) == 1
    assert len(cython.calls) == OUTLINED_COMPILE_CALL_COUNT
    assert len(outcome.successful) == 1
    assert outcome.successful[0].lowering_mode == "outlined-block"
    assert outcome.successful[0].native_helpers
    assert outcome.successful[0].fallback_reason == (
        "mypyc whole-callable: MYPYC_TYPE_ERROR: fixture; "
        "cython whole-callable: CYTHON_COMPILE_ERROR: whole callable fixture"
    )
    assert "outlined Cython" in outcome.build.stdout


def test_outlined_fallback_chain_restores_all_warm_cache_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A warm outline build invokes neither backend and restores its native artifact."""
    prepared, build_root, staged_source_roots = _prepare_outlined_coroutine_fixture(tmp_path)
    assert prepared.fallback is not None
    assert prepared.fallback.fallback is not None
    outlined = prepared.fallback.fallback
    artifact = (
        build_root / ".atoll" / "artifacts" / outlined.unit.install_relative_dir / "native.so"
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"outlined-native-artifact")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    mypyc = _FakeCompileBackend(
        BackendCompileResult(
            attempt=CompileAttempt(
                success=False,
                command=("mypyc",),
                stdout="",
                stderr="MYPYC_TYPE_ERROR: deterministic fixture rejection",
                artifact_paths=(),
                duration_seconds=0.1,
            ),
            artifacts=(),
        )
    )
    cython = _SequencedCompileBackend(
        "cython",
        (
            BackendCompileResult(
                attempt=CompileAttempt(
                    success=False,
                    command=("cython",),
                    stdout="",
                    stderr="CYTHON_COMPILE_ERROR: deterministic whole-callable rejection",
                    artifact_paths=(),
                    duration_seconds=0.2,
                ),
                artifacts=(),
            ),
            BackendCompileResult(
                attempt=CompileAttempt(
                    success=True,
                    command=("cython",),
                    stdout="",
                    stderr="",
                    artifact_paths=(artifact,),
                    duration_seconds=0.2,
                ),
                artifacts=(
                    ArtifactRecord(
                        region_id=outlined.unit.region_id,
                        backend="cython",
                        logical_module=outlined.unit.logical_module,
                        role="primary",
                        install_relative_path=(f"{outlined.unit.install_relative_dir}/native.so"),
                        digest=digest,
                        abi="cp312",
                        platform_tag="test-platform",
                    ),
                ),
            ),
        ),
    )
    monkeypatch.setitem(_compiler_backends, "mypyc", mypyc)
    monkeypatch.setitem(_compiler_backends, "cython", cython)
    context = _TypedRegionBuildContext(
        build_root=build_root,
        staged_source_roots=staged_source_roots,
        mypy_cache_dir=tmp_path / "mypy-cache",
        compile_cache_dir=tmp_path / "compile-cache",
        progress=None,
    )

    first = _build_typed_regions(
        prepared=(prepared,),
        context=context,
        initial_failures=(),
    )
    artifact.unlink()
    second = _build_typed_regions(
        prepared=(prepared,),
        context=context,
        initial_failures=(),
    )

    assert first.build.success is True
    assert second.build.success is True
    assert len(mypyc.calls) == 1
    assert len(cython.calls) == OUTLINED_COMPILE_CALL_COUNT
    assert second.successful[0].lowering_mode == "outlined-block"
    assert second.build.artifact_paths[0].read_bytes() == b"outlined-native-artifact"
    timing_names = {timing.name for timing in second.build.phase_timings}
    assert "backend_decision_cache" in timing_names
    assert "cache_restore" in timing_names


def test_typed_region_build_retries_deterministic_mypyc_failure_with_cython(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mypyc type rejection uses the prepared Cython variant before fallback."""
    project_root = tmp_path / "typed_region_project"
    build_root = tmp_path / "build"
    shutil.copytree(TYPED_FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    scans = _selected_scans(project, "typed_region_project.worker")
    selections = _selected_typed_regions(scans)
    mypyc_selection = next(selection for selection in selections if selection.backend == "mypyc")
    staged_source_roots = _copy_source_roots(project, build_root)
    prepared = _prepare_typed_region(
        project=project,
        build_root=build_root,
        staged_source_roots=staged_source_roots,
        selection=mypyc_selection,
    )
    assert prepared.fallback is not None
    mypyc = _FakeCompileBackend(
        BackendCompileResult(
            attempt=CompileAttempt(
                success=False,
                command=("mypyc",),
                stdout="",
                stderr="MYPYC_TYPE_ERROR: fixture",
                artifact_paths=(),
                duration_seconds=0.1,
            ),
            artifacts=(),
        )
    )
    cython = _FakeCompileBackend(
        BackendCompileResult(
            attempt=CompileAttempt(
                success=True,
                command=("cython",),
                stdout="",
                stderr="",
                artifact_paths=(),
                duration_seconds=0.2,
            ),
            artifacts=(),
        )
    )
    monkeypatch.setitem(_compiler_backends, "mypyc", mypyc)
    monkeypatch.setitem(_compiler_backends, "cython", cython)

    outcome = _build_typed_regions(
        prepared=(prepared,),
        context=_TypedRegionBuildContext(
            build_root=build_root,
            staged_source_roots=staged_source_roots,
            mypy_cache_dir=tmp_path / "mypy-cache",
            compile_cache_dir=tmp_path / "compile-cache",
            progress=None,
        ),
        initial_failures=(),
    )

    assert outcome.build.success is True
    assert outcome.build.stderr == ""
    assert "compiled" in outcome.build.stdout
    assert [item.generation.backend for item in outcome.successful] == ["cython"]
    assert outcome.successful[0].fallback_reason == (
        "mypyc whole-callable: MYPYC_TYPE_ERROR: fixture"
    )
    assert outcome.skipped == ()
    assert len(mypyc.calls) == 1
    assert len(cython.calls) == 1


def test_typed_region_build_restores_rejection_and_cython_artifact_on_second_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unchanged retry path invokes neither native compiler a second time."""
    project_root = tmp_path / "typed_region_project"
    build_root = tmp_path / "build"
    shutil.copytree(TYPED_FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    scans = _selected_scans(project, "typed_region_project.worker")
    selections = _selected_typed_regions(scans)
    mypyc_selection = next(selection for selection in selections if selection.backend == "mypyc")
    staged_source_roots = _copy_source_roots(project, build_root)
    prepared = _prepare_typed_region(
        project=project,
        build_root=build_root,
        staged_source_roots=staged_source_roots,
        selection=mypyc_selection,
    )
    assert prepared.fallback is not None
    fallback = prepared.fallback
    artifact = (
        build_root / ".atoll" / "artifacts" / fallback.unit.install_relative_dir / "native.so"
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"native-cython-artifact")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    mypyc = _FakeCompileBackend(
        BackendCompileResult(
            attempt=CompileAttempt(
                success=False,
                command=("mypyc",),
                stdout="",
                stderr="MYPYC_TYPE_ERROR: deterministic fixture rejection",
                artifact_paths=(),
                duration_seconds=0.1,
            ),
            artifacts=(),
        )
    )
    cython = _FakeCompileBackend(
        BackendCompileResult(
            attempt=CompileAttempt(
                success=True,
                command=("cython",),
                stdout="",
                stderr="",
                artifact_paths=(artifact,),
                duration_seconds=0.2,
            ),
            artifacts=(
                ArtifactRecord(
                    region_id=fallback.unit.region_id,
                    backend="cython",
                    logical_module=fallback.unit.logical_module,
                    role="primary",
                    install_relative_path=(f"{fallback.unit.install_relative_dir}/native.so"),
                    digest=digest,
                    abi="cp312",
                    platform_tag="test-platform",
                ),
            ),
        )
    )
    monkeypatch.setitem(_compiler_backends, "mypyc", mypyc)
    monkeypatch.setitem(_compiler_backends, "cython", cython)
    context = _TypedRegionBuildContext(
        build_root=build_root,
        staged_source_roots=staged_source_roots,
        mypy_cache_dir=tmp_path / "mypy-cache",
        compile_cache_dir=tmp_path / "compile-cache",
        progress=None,
    )

    first = _build_typed_regions(
        prepared=(prepared,),
        context=context,
        initial_failures=(),
    )
    artifact.unlink()
    second = _build_typed_regions(
        prepared=(prepared,),
        context=context,
        initial_failures=(),
    )

    assert first.build.success is True
    assert second.build.success is True
    assert len(mypyc.calls) == 1
    assert len(cython.calls) == 1
    assert second.successful[0].generation.backend == "cython"
    assert second.build.artifact_paths[0].read_bytes() == b"native-cython-artifact"
    assert "backend_decision_cache" in {timing.name for timing in second.build.phase_timings}
    assert "cache_restore" in {timing.name for timing in second.build.phase_timings}


def test_typed_region_build_does_not_compile_speculative_cython_after_mypyc_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prepared Cython fallback remains dormant when the preferred backend succeeds."""
    project_root = tmp_path / "typed_region_project"
    build_root = tmp_path / "build"
    shutil.copytree(TYPED_FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    scans = _selected_scans(project, "typed_region_project.worker")
    selections = _selected_typed_regions(scans)
    mypyc_selection = next(selection for selection in selections if selection.backend == "mypyc")
    staged_source_roots = _copy_source_roots(project, build_root)
    prepared = _prepare_typed_region(
        project=project,
        build_root=build_root,
        staged_source_roots=staged_source_roots,
        selection=mypyc_selection,
    )
    assert prepared.fallback is not None
    mypyc = _FakeCompileBackend(
        BackendCompileResult(
            attempt=CompileAttempt(
                success=True,
                command=("mypyc",),
                stdout="",
                stderr="",
                artifact_paths=(),
                duration_seconds=0.1,
            ),
            artifacts=(),
        )
    )
    cython = _FakeCompileBackend(
        BackendCompileResult(
            attempt=CompileAttempt(
                success=True,
                command=("cython",),
                stdout="",
                stderr="",
                artifact_paths=(),
                duration_seconds=0.2,
            ),
            artifacts=(),
        )
    )
    monkeypatch.setitem(_compiler_backends, "mypyc", mypyc)
    monkeypatch.setitem(_compiler_backends, "cython", cython)

    outcome = _build_typed_regions(
        prepared=(prepared,),
        context=_TypedRegionBuildContext(
            build_root=build_root,
            staged_source_roots=staged_source_roots,
            mypy_cache_dir=tmp_path / "mypy-cache",
            compile_cache_dir=tmp_path / "compile-cache",
            progress=None,
        ),
        initial_failures=(),
    )

    assert outcome.build.success is True
    assert [item.generation.backend for item in outcome.successful] == ["mypyc"]
    assert outcome.skipped == ()
    assert len(mypyc.calls) == 1
    assert cython.calls == []


@pytest.mark.parametrize("class_succeeds", [True, False])
def test_atomic_class_build_conditionally_uses_method_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    class_succeeds: bool,
) -> None:
    """Method variants stay dormant unless the selected atomic class fails."""
    project_root = tmp_path / "typed_region_project"
    build_root = tmp_path / "build"
    shutil.copytree(TYPED_FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    selections = _selected_typed_regions(_selected_scans(project, "typed_region_project.worker"))
    class_selection = next(
        selection for selection in selections if selection.variant_id.endswith("@cython-class")
    )
    method_selection = next(
        selection
        for selection in selections
        if selection.conditional_on_failure_of == class_selection.variant_id
    )
    staged_source_roots = _copy_source_roots(project, build_root)
    prepared = tuple(
        _prepare_typed_region(
            project=project,
            build_root=build_root,
            staged_source_roots=staged_source_roots,
            selection=selection,
        )
        for selection in (class_selection, method_selection)
    )
    cython = _FakeCompileBackend(
        BackendCompileResult(
            attempt=CompileAttempt(
                success=class_succeeds,
                command=("cython",),
                stdout="",
                stderr="" if class_succeeds else "CYTHON_COMPILE_ERROR: fixture",
                artifact_paths=(),
                duration_seconds=0.1,
            ),
            artifacts=(),
        )
    )
    mypyc = _FakeCompileBackend(
        BackendCompileResult(
            attempt=CompileAttempt(
                success=True,
                command=("mypyc",),
                stdout="",
                stderr="",
                artifact_paths=(),
                duration_seconds=0.1,
            ),
            artifacts=(),
        )
    )
    monkeypatch.setitem(_compiler_backends, "cython", cython)
    monkeypatch.setitem(_compiler_backends, "mypyc", mypyc)

    outcome = _build_typed_regions(
        prepared=prepared,
        context=_TypedRegionBuildContext(
            build_root=build_root,
            staged_source_roots=staged_source_roots,
            mypy_cache_dir=tmp_path / "mypy-cache",
            compile_cache_dir=tmp_path / "compile-cache",
            progress=None,
        ),
        initial_failures=(),
    )

    assert outcome.build.success is True
    if class_succeeds:
        assert [item.unit.region_id for item in outcome.successful] == [class_selection.variant_id]
        assert outcome.skipped == ()
        assert mypyc.calls == []
    else:
        assert [item.unit.region_id for item in outcome.successful] == [method_selection.variant_id]
        assert [failure.variant_id for failure in outcome.skipped] == [class_selection.variant_id]
        assert len(mypyc.calls) == 1
    assert len(cython.calls) == 1


def test_typed_region_build_records_real_cython_artifacts_after_mypyc_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deterministic retry compiles a real Cython artifact owned by its variant."""
    project_root = tmp_path / "typed_region_project"
    build_root = tmp_path / "build"
    shutil.copytree(TYPED_FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    scans = _selected_scans(project, "typed_region_project.worker")
    selections = _selected_typed_regions(scans)
    mypyc_selection = next(selection for selection in selections if selection.backend == "mypyc")
    staged_source_roots = _copy_source_roots(project, build_root)
    prepared = _prepare_typed_region(
        project=project,
        build_root=build_root,
        staged_source_roots=staged_source_roots,
        selection=mypyc_selection,
    )
    assert prepared.fallback is not None
    mypyc = _FakeCompileBackend(
        BackendCompileResult(
            attempt=CompileAttempt(
                success=False,
                command=("mypyc",),
                stdout="",
                stderr="MYPYC_TYPE_ERROR: fixture",
                artifact_paths=(),
                duration_seconds=0.1,
            ),
            artifacts=(),
        )
    )
    monkeypatch.setitem(_compiler_backends, "mypyc", mypyc)

    outcome = _build_typed_regions(
        prepared=(prepared,),
        context=_TypedRegionBuildContext(
            build_root=build_root,
            staged_source_roots=staged_source_roots,
            mypy_cache_dir=tmp_path / "mypy-cache",
            compile_cache_dir=tmp_path / "compile-cache",
            progress=None,
        ),
        initial_failures=(),
    )

    successful = outcome.successful[0]
    variant = CompiledRegionVariant(
        id=successful.unit.region_id,
        region=successful.generation.region,
        backend=successful.generation.backend,
        bindings=successful.generation.bindings,
    )
    report = build_compilation_report(
        CompilationReportInput(
            root=project_root,
            operation="compile",
            module_filter="typed_region_project.worker",
            islands=(),
            build=outcome.build,
            typed_regions=(variant.region,),
            compiled_regions=(variant.region,),
            compiled_bindings=variant.bindings,
            compiled_variants=(variant,),
            backend_assessments=(successful.assessment,),
            artifact_records=outcome.artifacts,
        )
    )

    assert outcome.build.success is True
    assert successful.generation.backend == "cython"
    assert successful.unit.region_id.endswith("@cython-mypyc-fallback")
    assert outcome.artifacts
    assert all(path.is_file() for path in outcome.build.artifact_paths)
    assert {artifact.region_id for artifact in outcome.artifacts} == {variant.id}
    assert report["compiled_regions"][0]["backend"] == "cython"
    assert report["compiled_regions"][0]["variant_id"] == variant.id
    assert report["compiled_regions"][0]["artifacts"]


def test_package_whole_project_compiles_regions_without_legacy_batch_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whole-project package mode compiles each typed region without sidecar batching."""
    project_root = tmp_path / "project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    package_dir = project_root / "src" / "app"
    ranking_source = package_dir / "ranking.py"
    (package_dir / "good.py").write_text(ranking_source.read_text(encoding="utf-8"))
    (package_dir / "bad.py").write_text(ranking_source.read_text(encoding="utf-8"))
    ranking_source.unlink()
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]

    def mixed_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        assert kwargs
        assert args
        paths = cast(tuple[Path, ...], args[0])
        assert paths
        if len(paths) > 1:
            return CompileAttempt(
                success=False,
                command=("mypyc", "batch"),
                stdout="",
                stderr="MYPYC_TYPE_ERROR: batch failed",
                artifact_paths=(),
                duration_seconds=0.1,
            )
        path = next(iter(paths))
        if path.stem.endswith("_good"):
            artifact = tmp_path / f"{path.stem}{suffix}"
            artifact.write_text("binary", encoding="utf-8")
            return CompileAttempt(
                success=True,
                command=("mypyc", str(path)),
                stdout="",
                stderr="",
                artifact_paths=(artifact,),
                duration_seconds=0.1,
            )
        return CompileAttempt(
            success=False,
            command=("mypyc", str(path)),
            stdout="",
            stderr="MYPYC_TYPE_ERROR: bad failed",
            artifact_paths=(),
            duration_seconds=0.1,
        )

    monkeypatch.setattr(package_command, "build_sidecars", mixed_build_sidecars, raising=False)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            output_dir=output_dir,
            keep_install_tree=True,
        )
    )

    good_text = (output_dir / "install" / "app" / "good.py").read_text(encoding="utf-8")
    bad_text = (output_dir / "install" / "app" / "bad.py").read_text(encoding="utf-8")
    assert result.success is True
    assert result.install_tree_kept is True
    assert result.cleanup_removed == (output_dir / "build",)
    assert result.cleanup_kept == (output_dir / "install",)
    assert result.islands == ()
    assert result.skipped == ()
    assert {binding.source.module for binding in result.compiled_bindings} == {
        "app.good",
        "app.bad",
    }
    assert "Initial batch build failed" not in result.build.stdout
    assert "# BEGIN ATOLL TYPED REGIONS: app.good" in good_text
    assert "# BEGIN ATOLL TYPED REGIONS: app.bad" in bad_text
    assert tuple((output_dir / "install" / ".atoll" / "artifacts").rglob(f"*{suffix}"))
    assert result.wheel_path is not None


def test_package_reports_progress_for_expensive_phases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Source-clean package builds expose phase progress to the CLI."""
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    messages: list[str] = []

    def successful_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        assert kwargs
        paths = cast(tuple[Path, ...], args[0])
        artifacts: list[Path] = []
        for path in paths:
            artifact = tmp_path / f"{path.stem}{suffix}"
            artifact.write_text("binary", encoding="utf-8")
            artifacts.append(artifact)
        return CompileAttempt(
            success=True,
            command=("mypyc",),
            stdout="",
            stderr="",
            artifact_paths=tuple(artifacts),
            duration_seconds=0.1,
        )

    monkeypatch.setattr(package_command, "build_sidecars", successful_build_sidecars, raising=False)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
            progress=messages.append,
        )
    )

    assert result.success is True
    assert any(message.startswith("discovered ") for message in messages)
    assert any(message.startswith("scanned ") for message in messages)
    assert any(message.startswith("compiling typed region variant") for message in messages)
    assert any(message.startswith("compile cache miss") for message in messages)
    assert any(message.startswith("writing wheel") for message in messages)


def test_quality_gate_rejects_missing_source_stripped_project(tmp_path: Path) -> None:
    """Configured commands cannot silently fall back to the target checkout."""
    project = _quality_gate_project(
        tmp_path,
        ('test_command = ["python", "-c", "pass"]',),
    )
    baseline = _BaselineWheelPayload(
        wheel_path=tmp_path / "baseline.whl",
        build=_successful_attempt(),
    )

    outcome = _run_configured_quality_gate(
        project=project,
        baseline=baseline,
        compiled_payload_root=tmp_path / "compiled",
        progress=None,
    )

    assert outcome.success is False
    assert outcome.error == "quality-gate project is missing"
    assert outcome.performance.status == "invalid"


def test_quality_gate_rejects_missing_benchmark_baseline(tmp_path: Path) -> None:
    """Benchmarking requires a distinct unpacked baseline payload."""
    project = _quality_gate_project(
        tmp_path,
        (
            'test_command = ["python", "-c", "pass"]',
            'benchmark_command = ["python", "bench.py"]',
        ),
    )
    baseline = _BaselineWheelPayload(
        wheel_path=tmp_path / "baseline.whl",
        build=_successful_attempt(),
        quality_project_root=tmp_path / "quality-project",
    )

    outcome = _run_configured_quality_gate(
        project=project,
        baseline=baseline,
        compiled_payload_root=tmp_path / "compiled",
        progress=None,
    )

    assert outcome.success is False
    assert outcome.error == "baseline payload is missing"


def test_quality_gate_reuses_early_baseline_semantic_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Promotion runs only the compiled test after the baseline passed before profiling."""
    project = _quality_gate_project(
        tmp_path,
        (
            'test_command = ["python", "-c", "pass"]',
            'benchmark_command = ["python", "bench.py"]',
        ),
    )
    quality_root = tmp_path / "quality-project"
    baseline_root = tmp_path / "baseline"
    compiled_root = tmp_path / "compiled"
    quality_root.mkdir()
    baseline_root.mkdir()
    compiled_root.mkdir()
    baseline_result = CommandRunEvidence(
        command=("python", "-c", "pass"),
        project_root=quality_root,
        payload_root=baseline_root,
        mode="baseline",
        returncode=0,
        stdout="",
        stderr="",
        duration_seconds=0.2,
    )
    baseline = _BaselineWheelPayload(
        wheel_path=tmp_path / "baseline.whl",
        build=_successful_attempt(),
        baseline_install_root=baseline_root,
        quality_project_root=quality_root,
        semantic_test_result=baseline_result,
    )
    executed_modes: list[RuntimeMode] = []

    def run_test(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        mode: RuntimeMode,
    ) -> CommandRunEvidence:
        executed_modes.append(mode)
        return CommandRunEvidence(
            command=command,
            project_root=project_root,
            payload_root=payload_root,
            mode=mode,
            returncode=0,
            stdout="",
            stderr="",
            duration_seconds=0.2,
        )

    def pass_benchmark(*args: object, **kwargs: object) -> BenchmarkGateResult:
        assert len(args) == 1
        assert kwargs
        return BenchmarkGateResult(
            status="passed",
            reason="fixture passed",
            minimum_speedup=1.1,
            baseline_median_seconds=1.1,
            compiled_median_seconds=1.0,
            speedup=1.1,
            warmups=(),
            samples=(),
        )

    monkeypatch.setattr(package_command, "run_performance_command", run_test)
    monkeypatch.setattr(package_command, "run_benchmark_gate", pass_benchmark)

    outcome = _run_configured_quality_gate(
        project=project,
        baseline=baseline,
        compiled_payload_root=compiled_root,
        progress=None,
    )

    assert outcome.success is True
    assert outcome.tests[0] is baseline_result
    assert [result.mode for result in outcome.tests] == ["baseline", "compiled"]
    assert executed_modes == ["compiled"]


def test_quality_gate_reports_semantic_test_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed configured test stops before benchmarking and preserves exit evidence."""
    project = _quality_gate_project(
        tmp_path,
        ('test_command = ["python", "-c", "raise SystemExit(9)"]',),
    )
    quality_root = tmp_path / "quality-project"
    quality_root.mkdir()
    baseline = _BaselineWheelPayload(
        wheel_path=tmp_path / "baseline.whl",
        build=_successful_attempt(),
        quality_project_root=quality_root,
    )
    messages: list[str] = []

    def failing_test(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        mode: RuntimeMode,
    ) -> CommandRunEvidence:
        return CommandRunEvidence(
            command=command,
            project_root=project_root,
            payload_root=payload_root,
            mode=mode,
            returncode=9,
            stdout="",
            stderr="",
            duration_seconds=0.5,
        )

    monkeypatch.setattr(package_command, "run_performance_command", failing_test)

    outcome = _run_configured_quality_gate(
        project=project,
        baseline=baseline,
        compiled_payload_root=tmp_path / "compiled",
        progress=messages.append,
    )

    assert outcome.success is False
    assert outcome.error == "compiled semantic test command exited 9"
    assert outcome.tests[0].returncode == TEST_FAILURE_RETURN_CODE
    assert outcome.performance.reason == "compiled semantic test command failed"
    assert messages == ["compiled semantic tests failed with exit 9 in 0.50s"]


def test_source_clean_success_summary_lists_every_fallback_kind(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI success output distinguishes build, preflight, and typed-region fallbacks."""
    project_root = tmp_path / "typed_region_project"
    shutil.copytree(TYPED_FIXTURE_ROOT, project_root)
    project = discover_project(project_root)
    scan = _selected_scans(project, "typed_region_project.worker")[0]
    selection = _selected_typed_regions((scan,))[0]
    island = EnabledIslandConfig(
        source_module=scan.module.name,
        source_path=scan.module.path,
        sidecar_module="_atoll_fixture",
        sidecar_path=tmp_path / "_atoll_fixture.py",
        symbols=("passthrough",),
    )
    failed_attempt = CompileAttempt(
        success=False,
        command=("compiler",),
        stdout="",
        stderr="",
        artifact_paths=(),
        duration_seconds=0.1,
    )
    result = package_command.PackageCommandResult(
        success=True,
        project_root=project_root,
        output_dir=tmp_path / "dist",
        install_root=tmp_path / "dist" / "install",
        wheel_path=tmp_path / "dist" / "fixture.whl",
        islands=(island,),
        build=_successful_attempt(),
        install_tree_kept=True,
        skipped=(package_command.PackageBuildFailure(island=island, build=failed_attempt),),
        preflight_skipped=(
            package_command.PackagePreflightFailure(
                scan=scan,
                blockers=(Blocker(severity="hard", code="module", message="module blocker"),),
            ),
            package_command.PackagePreflightFailure(
                scan=scan,
                blockers=(
                    Blocker(
                        severity="hard",
                        code="line",
                        message="line blocker",
                        lineno=7,
                    ),
                ),
            ),
        ),
        region_skipped=(
            package_command.PackageRegionBuildFailure(
                region=selection.region,
                variant_id=selection.variant_id,
                backend=selection.backend,
                assessment=selection.assessment,
                build=failed_attempt,
            ),
        ),
        performance=BenchmarkGateResult(
            status="passed",
            reason="fixture",
            minimum_speedup=1.1,
            baseline_median_seconds=1.2,
            compiled_median_seconds=1.0,
            speedup=1.2,
            warmups=(),
            samples=(),
        ),
    )

    _print_source_clean_success(
        result,
        label="source-clean compile",
        report_paths=(tmp_path / "report.json", tmp_path / "report.md"),
    )

    output = capsys.readouterr().out
    assert "Skipped 1 module(s) that mypyc could not build." in output
    assert f"- {scan.module.name}: failed" in output
    assert f"- {scan.module.name}: module: module blocker" in output
    assert f"- {scan.module.name}: line 7: line blocker" in output
    assert "Kept 1 typed region(s) as interpreted fallback." in output
    assert f"- {selection.variant_id} [{selection.backend}]: failed" in output
    assert "Install tree:" in output
    assert "Performance: 1.200x median speedup (passed)." in output


def test_package_rejects_not_profitable_wheel_after_semantic_tests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured benchmark below threshold removes the candidate wheel."""
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    pyproject = project_root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8")
        + "\n".join(
            (
                "",
                "[tool.atoll.compile]",
                'test_command = ["python", "-m", "pytest", "-q"]',
                'benchmark_command = ["python", "bench.py"]',
                "benchmark_warmups = 0",
                "benchmark_samples = 1",
                "minimum_speedup = 1.10",
                "",
            )
        ),
        encoding="utf-8",
    )
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    test_modes: list[RuntimeMode] = []
    target_project_root = project_root

    def successful_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        assert kwargs
        path = cast(tuple[Path, ...], args[0])[0]
        artifact = tmp_path / f"{path.stem}{suffix}"
        artifact.write_text("binary", encoding="utf-8")
        return CompileAttempt(
            success=True,
            command=("mypyc",),
            stdout="",
            stderr="",
            artifact_paths=(artifact,),
            duration_seconds=0.1,
        )

    def passing_test_command(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        mode: RuntimeMode,
    ) -> CommandRunEvidence:
        test_modes.append(mode)
        assert project_root != target_project_root
        assert not tuple((project_root / "src").rglob("*.py"))
        return CommandRunEvidence(
            command=command,
            project_root=project_root,
            payload_root=payload_root,
            mode=mode,
            returncode=0,
            stdout="passed",
            stderr="",
            duration_seconds=0.5,
        )

    def rejecting_benchmark(
        config: BenchmarkGateConfig,
        *,
        project_root: Path,
        baseline_payload_root: Path,
        compiled_payload_root: Path,
        progress: Callable[[BenchmarkProgress], None] | None = None,
    ) -> BenchmarkGateResult:
        assert config.command == ("python", "bench.py")
        assert project_root == project_root.resolve()
        assert baseline_payload_root != compiled_payload_root
        assert progress is not None
        return BenchmarkGateResult(
            status="not-profitable",
            reason="compiled median speedup 1.020 is below threshold 1.100",
            minimum_speedup=1.1,
            baseline_median_seconds=1.02,
            compiled_median_seconds=1.0,
            speedup=1.02,
            warmups=(),
            samples=(),
        )

    def insufficient_profile(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        module_paths: tuple[tuple[str, str], ...],
        scratch_dir: Path,
    ) -> ProfileResult:
        assert command == ("python", "bench.py")
        assert project_root != target_project_root
        assert payload_root.is_dir()
        assert module_paths
        assert scratch_dir.name == "profile"
        return replace(
            unconfigured_profile(),
            status="static-fallback",
            reason="insufficient baseline profile samples: observed 90, required 100",
            launch_kind="script",
            total_samples=90,
        )

    monkeypatch.setattr(package_command, "build_sidecars", successful_build_sidecars, raising=False)
    monkeypatch.setattr(package_command, "run_performance_command", passing_test_command)
    monkeypatch.setattr(package_command, "run_baseline_profile", insufficient_profile)
    monkeypatch.setattr(package_command, "run_benchmark_gate", rejecting_benchmark)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
        )
    )

    assert result.success is False
    assert result.wheel_path is None
    assert result.performance is not None
    assert result.performance.status == "not-profitable"
    assert result.error == result.performance.reason
    assert test_modes == ["baseline", "compiled"]
    assert not tuple(output_dir.glob("*.whl"))
    assert (output_dir / "install").exists()
    assert (output_dir / "build").exists()
    diagnostic_wheels = tuple((output_dir / "build" / "diagnostics").glob("*.whl"))
    assert diagnostic_wheels
    assert result.verification_steps[-1].target == diagnostic_wheels[0]


def test_profiled_promotion_rejects_a_wheel_without_profitable_regions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full gate still runs, but an all-rejected profile cannot publish a no-op wheel."""
    project = _quality_gate_project(tmp_path, ())
    output_dir = tmp_path / "out"
    build_root = output_dir / "build"
    install_root = output_dir / "install"
    build_root.mkdir(parents=True)
    install_root.mkdir(parents=True)
    baseline_wheel = build_root / "baseline.whl"
    baseline_wheel.write_bytes(b"baseline")
    gate_calls = 0

    def repack(**kwargs: object) -> Path:
        candidate_output = cast(Path, kwargs["output_dir"])
        candidate_output.mkdir(parents=True, exist_ok=True)
        candidate = candidate_output / "candidate.whl"
        candidate.write_bytes(b"candidate")
        return candidate

    def pass_full_gate(**kwargs: object) -> object:
        nonlocal gate_calls
        assert kwargs
        gate_calls += 1
        return _QualityGateOutcome(
            success=True,
            tests=(),
            performance=BenchmarkGateResult(
                status="passed",
                reason="fixture full gate passed",
                minimum_speedup=0.5,
                baseline_median_seconds=1.0,
                compiled_median_seconds=1.0,
                speedup=1.0,
                warmups=(),
                samples=(),
            ),
        )

    monkeypatch.setattr(package_command, "repack_overlaid_wheel", repack)
    monkeypatch.setattr(package_command, "_run_configured_quality_gate", pass_full_gate)

    result = _promote_source_clean_payload(
        _SourceCleanPromotionContext(
            options=package_command.PackageOptions(root=project.config.root),
            project=project,
            output_dir=output_dir,
            build_root=build_root,
            install_root=install_root,
            baseline=_BaselineWheelPayload(
                wheel_path=baseline_wheel,
                build=_successful_attempt(),
            ),
            verification_plan=PackageVerificationPlan(
                modules=(),
                regions=(),
                artifacts=(),
            ),
            build=_successful_attempt(),
            requires_native_artifact=True,
        )
    )

    assert gate_calls == 1
    assert result.success is False
    assert result.wheel_path is None
    assert result.error == "no profile-guided candidate met the 1.01x marginal speedup threshold"
    assert not tuple(output_dir.glob("*.whl"))
    assert tuple((build_root / "diagnostics").glob("*.whl"))


def test_package_profiles_before_backend_selection_and_scopes_hot_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured benchmark selects its hot member before backend assessment."""
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    pyproject = project_root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8")
        + """

[tool.atoll.compile]
test_command = ["python", "-c", "pass"]
benchmark_command = ["python", "bench.py"]
benchmark_warmups = 0
benchmark_samples = 1
minimum_speedup = 1.10
""",
        encoding="utf-8",
    )
    target_project_root = project_root
    events: list[str] = []
    progress_messages: list[str] = []
    original_selection = _selected_typed_regions

    def run_test(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        mode: RuntimeMode,
        region_allowlist: frozenset[str] | None = None,
    ) -> CommandRunEvidence:
        del region_allowlist
        events.append(f"test:{mode}")
        return CommandRunEvidence(
            command=command,
            project_root=project_root,
            payload_root=payload_root,
            mode=mode,
            returncode=0,
            stdout="",
            stderr="",
            duration_seconds=0.2,
        )

    def profile_baseline(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        module_paths: tuple[tuple[str, str], ...],
        scratch_dir: Path,
    ) -> ProfileResult:
        events.append("profile")
        assert command == ("python", "bench.py")
        assert project_root != target_project_root
        assert payload_root.is_dir()
        assert ("app.ranking", "app/ranking.py") in module_paths
        assert scratch_dir.name == "profile"
        return replace(
            unconfigured_profile(),
            status="profiled",
            reason="fixture profile",
            launch_kind="script",
            total_samples=200,
            mapped_project_samples=180,
            mapped_coverage=0.9,
            lifecycle=LifecycleCounts(
                start=10,
                return_=10,
                yield_=0,
                resume=0,
                unwind=0,
                throw=0,
            ),
            members=(
                ProfiledMember(
                    module="app.ranking",
                    qualname="rank_candidates",
                    samples=180,
                    coverage=0.9,
                    call_count=10,
                    lifecycle=LifecycleCounts(
                        start=10,
                        return_=10,
                        yield_=0,
                        resume=0,
                        unwind=0,
                        throw=0,
                    ),
                    signatures=(),
                    polymorphic_overflow=False,
                ),
            ),
        )

    def record_selection(*args: object, **kwargs: object) -> tuple[_TypedSelection, ...]:
        events.append("selection")
        return original_selection(*args, **kwargs)

    def pass_benchmark(*args: object, **kwargs: object) -> BenchmarkGateResult:
        assert len(args) == 1
        assert kwargs
        progress = cast(Callable[[BenchmarkProgress], None], kwargs["progress"])
        progress(
            BenchmarkProgress(
                phase="sample",
                pair_index=1,
                sample_index=1,
                mode="baseline",
                duration_seconds=0.125,
            )
        )
        return BenchmarkGateResult(
            status="passed",
            reason="fixture passed",
            minimum_speedup=1.1,
            baseline_median_seconds=1.1,
            compiled_median_seconds=1.0,
            speedup=1.1,
            warmups=(),
            samples=(),
        )

    monkeypatch.setattr(package_command, "run_performance_command", run_test)
    monkeypatch.setattr(package_command, "run_baseline_profile", profile_baseline)
    monkeypatch.setattr(package_command, "_selected_typed_regions", record_selection)
    monkeypatch.setattr(package_command, "run_benchmark_gate", pass_benchmark)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
            progress=progress_messages.append,
        )
    )

    assert result.success is True
    assert result.profile is not None
    assert result.profile.selected_symbols == (SymbolId("app.ranking", "rank_candidates"),)
    assert result.profile.selected_hot_coverage == 1.0
    assert events == [
        "selection",
        "test:baseline",
        "profile",
        "selection",
        "test:compiled",
        "test:compiled",
    ]
    assert "benchmark sample pair 1 baseline completed in 0.12s" in progress_messages
    assert {binding.source.qualname for binding in result.compiled_bindings} == {"rank_candidates"}


def test_package_greedily_keeps_only_profitable_profile_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profile order drives marginal trials and rejected artifacts never reach the wheel."""
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    pyproject = project_root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8")
        + """

[tool.atoll.compile]
test_command = ["python", "-c", "pass"]
benchmark_command = ["python", "bench.py"]
benchmark_warmups = 1
benchmark_samples = 7
minimum_speedup = 1.10
""",
        encoding="utf-8",
    )
    zero_lifecycle = LifecycleCounts(
        start=0,
        return_=0,
        yield_=0,
        resume=0,
        unwind=0,
        throw=0,
    )
    observed_allowlists: list[frozenset[str] | None] = []
    benchmark_thresholds: list[float] = []

    def run_test(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        mode: RuntimeMode,
        region_allowlist: frozenset[str] | None = None,
    ) -> CommandRunEvidence:
        observed_allowlists.append(region_allowlist)
        return CommandRunEvidence(
            command=command,
            project_root=project_root,
            payload_root=payload_root,
            mode=mode,
            returncode=0,
            stdout="",
            stderr="",
            duration_seconds=0.5,
        )

    def profile_baseline(*args: object, **kwargs: object) -> ProfileResult:
        assert args
        assert kwargs
        return replace(
            unconfigured_profile(),
            status="profiled",
            reason="two hot fixture members",
            launch_kind="script",
            total_samples=200,
            mapped_project_samples=180,
            mapped_coverage=0.9,
            lifecycle=zero_lifecycle,
            members=(
                ProfiledMember(
                    module="app.ranking",
                    qualname="rank_candidates",
                    samples=120,
                    coverage=0.6,
                    call_count=10,
                    lifecycle=zero_lifecycle,
                    signatures=(),
                    polymorphic_overflow=False,
                ),
                ProfiledMember(
                    module="app.ranking",
                    qualname="normalize_features",
                    samples=60,
                    coverage=0.3,
                    call_count=10,
                    lifecycle=zero_lifecycle,
                    signatures=(),
                    polymorphic_overflow=False,
                ),
            ),
        )

    def benchmark(
        config: BenchmarkGateConfig,
        **kwargs: object,
    ) -> BenchmarkGateResult:
        benchmark_thresholds.append(config.minimum_speedup)
        if config.minimum_speedup == _CANDIDATE_SPEEDUP:
            candidate_index = benchmark_thresholds.count(_CANDIDATE_SPEEDUP)
            speedup = 1.02 if candidate_index == 1 else 1.005
            status: BenchmarkStatus = "passed" if candidate_index == 1 else "not-profitable"
            assert kwargs["baseline_payload_root"] == kwargs["compiled_payload_root"]
            assert "baseline_region_allowlist" in kwargs
            assert "compiled_region_allowlist" in kwargs
        else:
            speedup = 1.12
            status = "passed"
            assert kwargs["baseline_payload_root"] != kwargs["compiled_payload_root"]
        return BenchmarkGateResult(
            status=status,
            reason=f"fixture speedup {speedup:.3f}",
            minimum_speedup=config.minimum_speedup,
            baseline_median_seconds=speedup,
            compiled_median_seconds=1.0,
            speedup=speedup,
            warmups=(),
            samples=(),
        )

    monkeypatch.setattr(package_command, "run_performance_command", run_test)
    monkeypatch.setattr(package_command, "run_baseline_profile", profile_baseline)
    monkeypatch.setattr(package_command, "run_benchmark_gate", benchmark)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
        )
    )

    assert result.success is True
    assert result.wheel_path is not None
    assert [trial.status for trial in result.candidate_trials] == ["accepted", "rejected"]
    assert result.candidate_trials[0].symbols == ("app.ranking::rank_candidates",)
    assert result.candidate_trials[1].symbols == ("app.ranking::normalize_features",)
    assert result.candidate_trials[0].accepted_hot_coverage == pytest.approx(2 / 3)
    assert result.candidate_trials[1].accepted_hot_coverage == pytest.approx(2 / 3)
    assert result.candidate_trials[0].marginal_speedup == pytest.approx(1.02)
    assert result.performance is not None
    assert result.performance.speedup == pytest.approx(1.12)
    assert {binding.source.qualname for binding in result.compiled_bindings} == {"rank_candidates"}
    assert benchmark_thresholds == [_CANDIDATE_SPEEDUP, _CANDIDATE_SPEEDUP, 1.1]
    assert observed_allowlists[0] is None
    assert observed_allowlists[1] == frozenset({result.candidate_trials[0].variant_id})
    assert observed_allowlists[2] == frozenset(
        trial.variant_id for trial in result.candidate_trials
    )
    assert observed_allowlists[3] is None
    assert len(result.test_results) == EXPECTED_FINAL_TEST_RESULTS
    with zipfile.ZipFile(result.wheel_path) as wheel:
        native_entries = {
            name
            for name in wheel.namelist()
            if any(name.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES)
        }
    assert native_entries == {record.install_relative_path for record in result.artifact_records}


def test_package_stops_before_profiling_when_baseline_semantics_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed interpreted baseline test prevents profile and native work."""
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    pyproject = project_root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8")
        + """

[tool.atoll.compile]
test_command = ["python", "-c", "raise SystemExit(9)"]
benchmark_command = ["python", "bench.py"]
""",
        encoding="utf-8",
    )

    def fail_baseline(
        command: tuple[str, ...],
        *,
        project_root: Path,
        payload_root: Path,
        mode: RuntimeMode,
    ) -> CommandRunEvidence:
        assert mode == "baseline"
        return CommandRunEvidence(
            command=command,
            project_root=project_root,
            payload_root=payload_root,
            mode=mode,
            returncode=9,
            stdout="",
            stderr="baseline fixture failed",
            duration_seconds=0.2,
        )

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("work continued after the baseline semantic failure")

    monkeypatch.setattr(package_command, "run_performance_command", fail_baseline)
    monkeypatch.setattr(package_command, "run_baseline_profile", forbidden)
    monkeypatch.setattr(package_command, "_build_typed_regions", forbidden)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
        )
    )

    assert result.success is False
    assert result.error == "baseline fixture failed"
    assert result.profile is None
    assert [run.mode for run in result.test_results] == ["baseline"]
    assert result.performance is not None
    assert result.performance.status == "invalid"
    assert not tuple(output_dir.glob("*.whl"))


def test_package_rejects_invalid_member_before_baseline_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Static request preflight prevents side effects for an invalid member."""
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    pyproject = project_root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8")
        + """

[tool.atoll.compile]
test_command = ["python", "-c", "pass"]
benchmark_command = ["python", "bench.py"]
""",
        encoding="utf-8",
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("configured target command ran before request validation")

    monkeypatch.setattr(package_command, "_prepare_baseline_wheel_payload", forbidden)
    monkeypatch.setattr(package_command, "run_performance_command", forbidden)
    monkeypatch.setattr(package_command, "run_baseline_profile", forbidden)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
            selected_members=(SymbolId("app.ranking", "missing_member"),),
        )
    )

    assert result.success is False
    assert result.error == (
        "requested member(s) are not backend-supported typed regions: app.ranking::missing_member"
    )
    assert result.profile is None
    assert result.test_results == ()
    assert not output_dir.exists()


def test_package_reuses_region_cache_for_unchanged_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unchanged second source-clean package build restores region artifacts."""
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    calls = 0

    def successful_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        nonlocal calls
        assert kwargs
        calls += 1
        if calls > 1:
            raise AssertionError("compile cache did not skip mypyc")
        path = cast(tuple[Path, ...], args[0])[0]
        artifact = tmp_path / f"{path.stem}{suffix}"
        artifact.write_text("binary", encoding="utf-8")
        return CompileAttempt(
            success=True,
            command=("mypyc",),
            stdout="",
            stderr="",
            artifact_paths=(artifact,),
            duration_seconds=0.1,
            phase_timings=(
                CompilePhaseTiming(name="mypycify", duration_seconds=0.08),
                CompilePhaseTiming(name="build_ext", duration_seconds=0.02),
            ),
        )

    monkeypatch.setattr(package_command, "build_sidecars", successful_build_sidecars, raising=False)

    first = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
        )
    )
    second = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
        )
    )

    assert first.success is True
    assert first.build.cache_status == "miss"
    assert second.success is True
    assert second.build.cache_status == "hit"
    assert calls == 0
    cache_timings = tuple(
        timing.name for timing in second.build.phase_timings if timing.name.startswith("cache_")
    )
    assert cache_timings == ("cache_lookup", "cache_restore")
    assert second.wheel_path is not None
    with zipfile.ZipFile(second.wheel_path) as wheel:
        assert any(name.startswith(".atoll/artifacts/") for name in wheel.namelist())


def test_package_caches_multiple_regions_independently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whole-project source-clean builds restore every unchanged region independently."""
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    ranking_source = project_root / "src" / "app" / "ranking.py"
    extra_source = project_root / "src" / "app" / "extra.py"
    extra_source.write_text(ranking_source.read_text(encoding="utf-8"), encoding="utf-8")
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    calls = 0

    def partial_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        nonlocal calls
        assert kwargs
        calls += 1
        paths = cast(tuple[Path, ...], args[0])
        if len(paths) > 1:
            return CompileAttempt(
                success=False,
                command=("mypyc",),
                stdout="",
                stderr="batch failed",
                artifact_paths=(),
                duration_seconds=0.1,
            )
        assert paths
        path = paths[0]
        if "extra" in path.stem:
            return CompileAttempt(
                success=False,
                command=("mypyc",),
                stdout="",
                stderr="extra failed",
                artifact_paths=(),
                duration_seconds=0.1,
            )
        artifact = tmp_path / f"{path.stem}{suffix}"
        artifact.write_text("binary", encoding="utf-8")
        return CompileAttempt(
            success=True,
            command=("mypyc",),
            stdout="",
            stderr="",
            artifact_paths=(artifact,),
            duration_seconds=0.1,
        )

    monkeypatch.setattr(package_command, "build_sidecars", partial_build_sidecars, raising=False)

    first = package_command.execute_package(
        package_command.PackageOptions(root=project_root, output_dir=output_dir)
    )
    second = package_command.execute_package(
        package_command.PackageOptions(root=project_root, output_dir=output_dir)
    )

    assert first.success is True
    assert first.build.cache_status == "miss"
    assert first.skipped == ()
    assert second.success is True
    assert second.build.cache_status == "hit"
    assert second.skipped == ()
    assert calls == 0
    cache_timings = tuple(
        timing.name for timing in second.build.phase_timings if timing.name.startswith("cache_")
    )
    assert cache_timings.count("cache_lookup") == EXPECTED_ATOMIC_SELECTION_COUNT
    assert cache_timings.count("cache_restore") == EXPECTED_ATOMIC_SELECTION_COUNT
    assert second.wheel_path is not None
    with zipfile.ZipFile(second.wheel_path) as wheel:
        names = set(wheel.namelist())
    assert any(name.startswith(".atoll/artifacts/") for name in names)
    assert "app/extra.py" in names
    with zipfile.ZipFile(second.wheel_path) as wheel:
        extra_text = wheel.read("app/extra.py").decode()
    assert "BEGIN ATOLL TYPED REGIONS: app.extra" in extra_text


def test_package_cache_invalidates_when_source_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Region cache keys change when retained function source changes."""
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    calls = 0

    def successful_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        nonlocal calls
        assert kwargs
        calls += 1
        path = cast(tuple[Path, ...], args[0])[0]
        artifact = tmp_path / f"{path.stem}-{calls}{suffix}"
        artifact.write_text("binary", encoding="utf-8")
        return CompileAttempt(
            success=True,
            command=("mypyc",),
            stdout="",
            stderr="",
            artifact_paths=(artifact,),
            duration_seconds=0.1,
        )

    monkeypatch.setattr(package_command, "build_sidecars", successful_build_sidecars, raising=False)

    first = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
        )
    )
    ranking_source = project_root / "src" / "app" / "ranking.py"
    ranking_source.write_text(
        ranking_source.read_text(encoding="utf-8").replace(
            "DEFAULT_WEIGHT = 1.5",
            "DEFAULT_WEIGHT = 1.75",
        ),
        encoding="utf-8",
    )
    second = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
        )
    )

    assert first.build.cache_status == "miss"
    assert second.build.cache_status == "miss"
    assert calls == 0


def test_package_cache_invalidates_when_generator_version_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Region cache keys include the typed-region generator version."""
    project_root = tmp_path / "simple_project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    calls = 0

    def successful_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        nonlocal calls
        assert kwargs
        calls += 1
        path = cast(tuple[Path, ...], args[0])[0]
        artifact = tmp_path / f"{path.stem}-{calls}{suffix}"
        artifact.write_text("binary", encoding="utf-8")
        return CompileAttempt(
            success=True,
            command=("mypyc",),
            stdout="",
            stderr="",
            artifact_paths=(artifact,),
            duration_seconds=0.1,
        )

    monkeypatch.setattr(package_command, "build_sidecars", successful_build_sidecars, raising=False)

    first = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
        )
    )
    monkeypatch.setattr(package_command, "TYPED_METHOD_GENERATOR_VERSION", "changed")
    second = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.ranking",
            output_dir=output_dir,
        )
    )

    assert first.build.cache_status == "miss"
    assert second.build.cache_status == "miss"
    assert calls == 0


def test_package_whole_project_never_enters_legacy_retry_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Typed-region whole-project compilation never calls the sidecar retry loop."""
    project_root = tmp_path / "project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    package_dir = project_root / "src" / "app"
    ranking_source = package_dir / "ranking.py"
    (package_dir / "first.py").write_text(ranking_source.read_text(encoding="utf-8"))
    (package_dir / "second.py").write_text(ranking_source.read_text(encoding="utf-8"))
    ranking_source.unlink()

    def failing_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        assert kwargs
        assert args
        raise AssertionError("legacy sidecar retry loop was invoked")

    monkeypatch.setattr(package_command, "build_sidecars", failing_build_sidecars, raising=False)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            output_dir=output_dir,
            keep_install_tree=True,
        )
    )

    assert result.success is True
    assert result.error is None
    assert result.islands == ()
    assert result.skipped == ()
    assert {binding.source.module for binding in result.compiled_bindings} == {
        "app.first",
        "app.second",
    }


def test_package_compiles_typed_functions_despite_unrelated_typevar_syntax(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Typed functions are assessed independently from unrelated module TypeVars."""
    project_root = tmp_path / "project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    bad_module = project_root / "src" / "app" / "typing_features.py"
    bad_module.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "from typing_extensions import TypeVar",
                "",
                "T = TypeVar('T', infer_variance=True)",
                "",
                "def helper(value: int) -> int:",
                "    return value + 1",
                "",
                "def candidate(value: int) -> int:",
                "    adjusted = helper(value)",
                "    return adjusted",
                "",
            ]
        ),
        encoding="utf-8",
    )

    build_calls: list[tuple[Path, ...]] = []

    def failing_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        assert kwargs
        paths = cast(tuple[Path, ...], args[0])
        build_calls.append(paths)
        return CompileAttempt(
            success=False,
            command=("mypyc", *(str(path) for path in paths)),
            stdout="",
            stderr="MYPYC_TYPE_ERROR: generated sidecar failed",
            artifact_paths=(),
            duration_seconds=0.1,
        )

    monkeypatch.setattr(package_command, "build_sidecars", failing_build_sidecars, raising=False)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            module_name="app.typing_features",
            output_dir=output_dir,
        )
    )

    assert result.success is True
    assert build_calls == []
    assert result.error is None
    assert result.preflight_skipped == ()
    assert not (output_dir / "build").exists()
    assert result.cleanup_removed == (output_dir / "build", output_dir / "install")
    assert result.cleanup_kept == ()
    assert result.native_readiness == ()
    assert {binding.source.qualname for binding in result.compiled_bindings} == {
        "helper",
        "candidate",
    }


def test_package_whole_project_uses_region_assessments_in_typing_heavy_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whole-project mode compiles safe regions without module-level readiness gates."""
    project_root = tmp_path / "project"
    output_dir = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, project_root)
    package_dir = project_root / "src" / "app"
    clean_source = package_dir / "ranking.py"
    original_clean_source = clean_source.read_text(encoding="utf-8")
    (package_dir / "blocked.py").write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "from typing_extensions import TypeVar",
                "",
                "T = TypeVar('T', default=str)",
                "",
                "def helper(value: int) -> int:",
                "    return value + 1",
                "",
                "def candidate(value: int) -> int:",
                "    adjusted = helper(value)",
                "    return adjusted",
                "",
            ]
        ),
        encoding="utf-8",
    )
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]

    def successful_build_sidecars(*args: object, **kwargs: object) -> CompileAttempt:
        assert kwargs
        paths = cast(tuple[Path, ...], args[0])
        artifacts: list[Path] = []
        for path in paths:
            artifact = tmp_path / f"{path.stem}{suffix}"
            artifact.write_text("binary", encoding="utf-8")
            artifacts.append(artifact)
        return CompileAttempt(
            success=True,
            command=("mypyc",),
            stdout="",
            stderr="",
            artifact_paths=tuple(artifacts),
            duration_seconds=0.1,
        )

    monkeypatch.setattr(package_command, "build_sidecars", successful_build_sidecars, raising=False)

    result = package_command.execute_package(
        package_command.PackageOptions(
            root=project_root,
            output_dir=output_dir,
            keep_install_tree=True,
        )
    )

    assert result.success is True
    assert result.install_tree_kept is True
    assert result.cleanup_removed == (output_dir / "build",)
    assert result.cleanup_kept == (output_dir / "install",)
    assert result.islands == ()
    assert {binding.source.module for binding in result.compiled_bindings} == {
        "app.ranking",
        "app.blocked",
    }
    assert result.preflight_skipped == ()
    assert "# BEGIN ATOLL TYPED REGIONS: app.ranking" in (
        output_dir / "install" / "app" / "ranking.py"
    ).read_text(encoding="utf-8")
    assert "# BEGIN ATOLL TYPED REGIONS: app.blocked" in (
        output_dir / "install" / "app" / "blocked.py"
    ).read_text(encoding="utf-8")
    assert result.native_readiness == ()
    assert clean_source.read_text(encoding="utf-8") == original_clean_source


def test_package_helpers_handle_flat_source_roots(tmp_path: Path) -> None:
    """Flat source roots copy their contents into the build root."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    project = discover_project(tmp_path)
    build_root = tmp_path / "build"
    build_root.mkdir()

    staged_roots = _copy_source_roots(project, build_root)

    assert staged_roots == (build_root,)
    assert (build_root / "pkg" / "mod.py").exists()


def test_atoll_artifact_helpers_copy_artifacts_and_skip_same_file(tmp_path: Path) -> None:
    """Source-clean artifact copies tolerate missing roots and an identical target."""
    source_root = tmp_path / "source"
    artifact_dir = source_root / ".atoll" / "artifacts"
    install_root = tmp_path / "install"
    artifact_dir.mkdir(parents=True)
    native = artifact_dir / f"_sidecar{importlib.machinery.EXTENSION_SUFFIXES[0]}"
    native.write_text("binary", encoding="utf-8")

    _copy_atoll_artifacts((tmp_path / "missing", source_root), install_root)
    copied = install_root / ".atoll" / "artifacts" / native.name
    _copy_if_different(copied, copied)

    assert copied.read_text(encoding="utf-8") == "binary"


def test_project_metadata_falls_back_for_missing_or_dynamic_version(tmp_path: Path) -> None:
    """Project metadata falls back to stable Atoll values when version is dynamic."""
    fallback = _project_metadata(tmp_path)
    assert fallback.name == tmp_path.name
    assert fallback.version == "0+atoll"

    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "dynamic-project"',
                'dynamic = ["version"]',
                'requires-python = ">=3.12"',
                'dependencies = ["pydantic>=2"]',
            ]
        ),
        encoding="utf-8",
    )

    metadata = _project_metadata(project_root)
    assert metadata.name == "dynamic-project"
    assert metadata.version == "0+atoll"
    assert metadata.requires_python == ">=3.12"
    assert metadata.dependencies == ("pydantic>=2",)


def test_package_small_helpers_cover_fallbacks(tmp_path: Path) -> None:
    """Small helper fallbacks stay deterministic."""
    path = tmp_path / "existing"
    path.mkdir()
    (path / "old.txt").write_text("old", encoding="utf-8")
    _reset_dir(path)
    assert path.exists()
    assert not (path / "old.txt").exists()

    assert _relative_source_root(tmp_path, tmp_path / "src") == Path("src")
    outside_root = tmp_path.parent / "not-under-root"
    assert _relative_source_root(tmp_path, outside_root) != outside_root
    assert _mapping(None) == {}
    assert _sequence(None) == ()
    assert _string(1) is None


def test_package_helpers_report_missing_modules(tmp_path: Path) -> None:
    """Module lookup helpers fail clearly for impossible paths."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    project = DiscoveredProject(
        config=discover_project(project_root).config,
        modules=(),
    )

    with pytest.raises(ValueError, match="module not found"):
        _find_module((), "missing")
    with pytest.raises(ValueError, match="outside configured source roots"):
        _staged_module(
            ModuleId(name="missing", path=tmp_path / "outside.py"),
            project,
            (tmp_path / "stage",),
        )


def _quality_gate_project(
    tmp_path: Path,
    compile_lines: tuple[str, ...],
) -> DiscoveredProject:
    project_root = tmp_path / "quality-gate-project"
    shutil.copytree(FIXTURE_ROOT, project_root)
    pyproject = project_root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8")
        + "\n".join(("", "[tool.atoll.compile]", *compile_lines, "")),
        encoding="utf-8",
    )
    return discover_project(project_root)


def _successful_attempt() -> CompileAttempt:
    return CompileAttempt(
        success=True,
        command=("fixture",),
        stdout="",
        stderr="",
        artifact_paths=(),
        duration_seconds=0.0,
    )
