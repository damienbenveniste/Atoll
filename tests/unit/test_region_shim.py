"""Tests for staged-wheel typed-region shim generation."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.machinery
import importlib.util
import inspect
import sys
from collections.abc import AsyncGenerator, Callable, Coroutine, Generator
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import pytest

from atoll.generation.region_shim import (
    OutlinedShellConfig,
    RegionShimConfig,
    insert_or_replace_region_shim,
    remove_region_shim,
    render_region_shim,
)
from atoll.models import (
    BindingKind,
    BindingTarget,
    ExecutionKind,
    RuntimeTypeGuard,
    SymbolId,
)
from atoll.native_optimization.models import (
    BufferLayoutGuardPayload,
    CallableCodeIdentityGuardPayload,
    DirectFieldGuardPayload,
    ExactTypeGuardPayload,
    GuardExpression,
)

FALLBACK_RESULT = 4


class _WorkerProtocol(Protocol):
    """Runtime shape used to type the class created by the shim test."""

    def scale(self, value: int) -> int:
        """Return the fixture's source fallback result."""
        ...


class _RuntimeWorkerProtocol(Protocol):
    """Runtime shape loaded from a generated region-shim fixture."""

    def scale(self, value: int = 1, token: object = ...) -> tuple[str, object]:
        """Return the selected scale implementation and token."""
        ...

    def values(
        self,
        token: object = ...,
    ) -> Generator[tuple[str, object] | tuple[str, str], str, None]:
        """Yield through the selected generator implementation."""
        ...

    def score(self, token: object = ...) -> Coroutine[object, object, object]:
        """Return the selected coroutine implementation."""
        ...

    def stream(
        self,
        token: object = ...,
    ) -> AsyncGenerator[tuple[str, object] | tuple[str, str] | str, str]:
        """Yield through the selected async-generator implementation."""
        ...


class _TextScaleWorkerProtocol(Protocol):
    """Runtime shape for direct-call fast-path tests."""

    def scale(self, value: int) -> str:
        """Return the selected scale implementation."""
        ...


class _FieldWorkerProtocol(Protocol):
    """Exact-field instance surface used by structured guard tests."""

    def scale(self, value: int) -> tuple[str, int]:
        """Return the selected scale route and value."""
        ...


class _FieldWorkerTypeProtocol(Protocol):
    """Constructor surface used by structured direct-field guard tests."""

    def __call__(self, factor: object) -> _FieldWorkerProtocol:
        """Create one worker with the requested field value."""
        ...


class _CallableGuardModule(Protocol):
    """Dynamic module surface used by callable-identity dispatcher tests."""

    helper: Callable[[int], int]

    def root(self, value: int) -> tuple[str, int]:
        """Return the selected call-chain route and value."""
        ...


class _BufferGuardModule(Protocol):
    """Dynamic module surface used by zero-copy buffer dispatcher tests."""

    def checksum(self, data: object) -> tuple[str, int]:
        """Return the selected buffer route and checksum."""
        ...


class _RuntimeWorkerTypeProtocol(Protocol):
    """Descriptor surface loaded from a generated region-shim fixture."""

    def __call__(self) -> _RuntimeWorkerProtocol:
        """Create a runtime worker instance."""
        ...

    @staticmethod
    def parse(text: str = "source") -> str:
        """Return the selected staticmethod implementation."""
        ...

    @classmethod
    def create(cls, token: object = ...) -> object:
        """Return the selected classmethod implementation."""
        ...


def _binding(
    name: str,
    kind: BindingKind,
    execution_kind: ExecutionKind = "sync",
) -> BindingTarget:
    return BindingTarget(
        source=SymbolId(module="pkg.worker", qualname=f"Worker.{name}"),
        compiled_name=f"Worker__{name}",
        kind=kind,
        owner_class="Worker",
        execution_kind=execution_kind,
    )


def _config(tmp_path: Path) -> RegionShimConfig:
    source_path = tmp_path / "pkg" / "worker.py"
    source_path.parent.mkdir(parents=True)
    return RegionShimConfig(
        source_module="pkg.worker",
        source_path=source_path,
        region_id="pkg.worker::Worker:abc",
        backend="mypyc",
        compiled_module="_atoll_pkg_worker_abc",
        artifact_dir=tmp_path / ".atoll" / "artifacts",
        bindings=(
            _binding("scale", "instance_method"),
            _binding("parse", "staticmethod"),
            _binding("create", "classmethod"),
            _binding("values", "instance_method", "generator"),
            _binding("score", "instance_method", "coroutine"),
        ),
    )


def _load_region_module(
    config: RegionShimConfig,
    source: str,
    compiled: str,
    monkeypatch: pytest.MonkeyPatch,
) -> ModuleType:
    module_name = config.region_id.replace(":", "_").replace("@", "_").replace(".", "_")
    config.source_path.parent.mkdir(parents=True, exist_ok=True)
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    config.source_path.write_text(
        insert_or_replace_region_shim(source, (config,)).new_text,
        encoding="utf-8",
    )
    compiled_path = config.artifact_dir / f"{config.compiled_module}.py"
    compiled_path.write_text(compiled, encoding="utf-8")
    monkeypatch.setattr(importlib.machinery, "EXTENSION_SUFFIXES", (".py",))
    spec = importlib.util.spec_from_file_location(
        f"shim_runtime_{module_name}",
        config.source_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_region_shim_falls_back_when_direct_helper_identity_or_code_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Call-chain guards reject monkey-patched helpers before native execution."""
    source_path = tmp_path / "pkg" / "worker.py"
    artifact_dir = tmp_path / "pkg" / "artifacts"
    binding = BindingTarget(
        source=SymbolId(module="pkg.worker", qualname="root"),
        compiled_name="compiled_root",
        kind="module",
        owner_class=None,
        execution_kind="sync",
    )
    guard = GuardExpression(
        kind="callable-code-identity",
        payload=CallableCodeIdentityGuardPayload(
            subject="helper",
            callable_module="pkg.worker",
            callable_qualname="helper",
            code_fingerprint=hashlib.sha256(
                b"def helper(value: int) -> int:\n    return value + 1"
            ).hexdigest(),
            code_firstlineno=1,
        ),
        message="helper identity must match",
    )
    config = RegionShimConfig(
        source_module="pkg.worker",
        source_path=source_path,
        region_id="chain-region",
        backend="cython",
        compiled_module="_atoll_chain",
        artifact_dir=artifact_dir,
        bindings=(binding,),
        variant_id="chain-i32",
        dispatch_rank=10,
        variant_guards=(guard,),
    )
    source = """def helper(value: int) -> int:
    return value + 1

def root(value: int) -> tuple[str, int]:
    return ("python", helper(value))
"""
    compiled = """def compiled_root(value):
    return ("compiled", value + 1)
"""
    module = cast(_CallableGuardModule, _load_region_module(config, source, compiled, monkeypatch))
    original_helper = module.helper

    assert module.root(3) == ("compiled", 4)

    def replacement(value: int) -> int:
        return value + 10

    module.helper = replacement
    assert module.root(3) == ("python", 13)

    module.helper = original_helper
    original_helper.__code__ = replacement.__code__
    assert module.root(3) == ("python", 13)


def test_region_shim_guards_exact_owner_and_direct_integer_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Instance native entry requires exact class, field type, and closed field domain."""
    config = RegionShimConfig(
        source_module="pkg.worker",
        source_path=tmp_path / "pkg" / "worker.py",
        region_id="instance-chain",
        backend="cython",
        compiled_module="_atoll_instance_chain",
        artifact_dir=tmp_path / "pkg" / "artifacts",
        bindings=(_binding("scale", "instance_method"),),
        variant_id="instance-chain-i32",
        dispatch_rank=10,
        variant_guards=(
            GuardExpression(
                kind="direct-field",
                payload=DirectFieldGuardPayload(
                    owner_subject="self",
                    owner_type_module="pkg.worker",
                    owner_type_qualname="Worker",
                    field_name="factor",
                    field_type="int",
                    minimum=0,
                    maximum=100,
                ),
                message="worker.factor must fit the proven domain",
            ),
        ),
    )
    source = """class Worker:
    def __init__(self, factor):
        self.factor = factor

    def scale(self, value: int) -> tuple[str, int]:
        return ("python", value * self.factor)
"""
    compiled = """def Worker__scale(self, value):
    return ("compiled", value * self.factor)
"""
    module = _load_region_module(config, source, compiled, monkeypatch)
    worker_type = cast(_FieldWorkerTypeProtocol, module.Worker)
    runtime_worker_class = cast(type[object], module.Worker)
    child_type = cast(_FieldWorkerTypeProtocol, type("Child", (runtime_worker_class,), {}))

    assert worker_type(3).scale(4) == ("compiled", 12)
    assert worker_type(True).scale(4) == ("python", 4)
    assert worker_type(101).scale(4) == ("python", 404)
    assert child_type(3).scale(4) == ("python", 12)


def test_region_shim_guards_exact_buffer_layout_and_safe_length(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Buffer variants enter native code only after constant-time layout proof."""
    binding = BindingTarget(
        source=SymbolId(module="pkg.worker", qualname="checksum"),
        compiled_name="compiled_checksum",
        kind="module",
        owner_class=None,
        execution_kind="sync",
    )
    config = RegionShimConfig(
        source_module="pkg.worker",
        source_path=tmp_path / "pkg" / "worker.py",
        region_id="buffer-kernel",
        backend="cython",
        compiled_module="_atoll_buffer_kernel",
        artifact_dir=tmp_path / "pkg" / "artifacts",
        bindings=(binding,),
        variant_id="buffer-bytes-B",
        dispatch_rank=30,
        variant_guards=(
            GuardExpression(
                kind="exact-type",
                payload=ExactTypeGuardPayload(
                    subject="data",
                    type_module="builtins",
                    type_qualname="bytes",
                ),
                message="data must be exact bytes",
            ),
            GuardExpression(
                kind="buffer-layout",
                payload=BufferLayoutGuardPayload(
                    subject="data",
                    format="B",
                    itemsize=1,
                    ndim=1,
                    c_contiguous=True,
                    f_contiguous=True,
                    readonly=True,
                    minimum_length=0,
                    maximum_length=4,
                ),
                message="data must match the proven byte layout and length",
            ),
        ),
    )
    source = """def checksum(data):
    return ("python", sum(data))
"""
    compiled = """def compiled_checksum(data):
    return ("compiled", sum(data))
"""
    module = cast(
        _BufferGuardModule,
        _load_region_module(config, source, compiled, monkeypatch),
    )

    assert module.checksum(b"abc") == ("compiled", 294)
    assert module.checksum(b"abcdef") == ("python", 597)
    assert module.checksum(bytearray(b"abc")) == ("python", 294)


def test_region_shim_rejects_helper_rebound_during_module_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Source fingerprint validation rejects a stale native helper before installation."""
    original_source = "def helper(value: int) -> int:\n    return value + 1"
    guard = GuardExpression(
        kind="callable-code-identity",
        payload=CallableCodeIdentityGuardPayload(
            subject="helper",
            callable_module="pkg.worker",
            callable_qualname="helper",
            code_fingerprint=hashlib.sha256(original_source.encode()).hexdigest(),
            code_firstlineno=1,
        ),
        message="helper source must match",
    )
    config = RegionShimConfig(
        source_module="pkg.worker",
        source_path=tmp_path / "pkg" / "worker.py",
        region_id="module-rebind-chain",
        backend="cython",
        compiled_module="_atoll_rebound_chain",
        artifact_dir=tmp_path / "pkg" / "artifacts",
        bindings=(
            BindingTarget(
                source=SymbolId("pkg.worker", "root"),
                compiled_name="compiled_root",
                kind="module",
                owner_class=None,
                execution_kind="sync",
            ),
        ),
        variant_guards=(guard,),
    )
    source = f"""{original_source}

def replacement(value: int) -> int:
    return value + 10

helper = replacement

def root(value: int) -> tuple[str, int]:
    return ("python", helper(value))
"""
    module = cast(
        _CallableGuardModule,
        _load_region_module(
            config,
            source,
            'def compiled_root(value):\n    return ("compiled", value + 1)\n',
            monkeypatch,
        ),
    )

    assert module.root(3) == ("python", 13)


def test_region_shim_renders_descriptor_and_status_contract(tmp_path: Path) -> None:
    """The staged block records and binds every promised method independently."""
    rendered = render_region_shim((_config(tmp_path),))

    compile(rendered, "worker.py", "exec")
    assert "# BEGIN ATOLL TYPED REGIONS: pkg.worker" in rendered
    assert "__atoll_region_status__" in rendered
    assert "staticmethod(_atoll_value)" in rendered
    assert "classmethod(_atoll_value)" in rendered
    assert "isgeneratorfunction" in rendered
    assert "iscoroutinefunction" in rendered
    assert "_atoll_verify_execution_kind" in rendered
    assert "_atoll_rollback" in rendered
    assert "ATOLL_DISABLE" in rendered
    assert "ATOLL_REQUIRE_COMPILED" in rendered
    assert "ATOLL_REGION_ALLOWLIST" in rendered
    assert "ATOLL_VARIANT_ALLOWLIST" in rendered
    assert "__atoll_compiled_target__" in rendered
    assert "__atoll_python_fallback__" in rendered
    assert "_atoll_guards_pass" in rendered
    assert "_atoll_name_to_remove.startswith" in rendered


def test_region_shim_builds_one_signature_shaped_dispatcher_for_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multiple native variants merge into one ordered source-shaped dispatcher."""
    source_path = tmp_path / "pkg" / "worker.py"
    source_path.parent.mkdir(parents=True)
    binding = BindingTarget(
        source=SymbolId(module="pkg.worker", qualname="choose"),
        compiled_name="choose",
        kind="module",
        owner_class=None,
        execution_kind="sync",
    )
    safe_binding = replace(
        binding,
        guards=(
            RuntimeTypeGuard(
                parameter_name="value",
                positional_index=0,
                annotation="int",
                nominal_type_paths=("int",),
                allow_none=False,
            ),
        ),
    )
    generic = RegionShimConfig(
        source_module="pkg.worker",
        source_path=source_path,
        region_id="semantic-region",
        variant_id="generic-cython",
        dispatch_rank=210,
        backend="cython",
        compiled_module="_atoll_generic",
        artifact_dir=source_path.parent / "generic-artifacts",
        bindings=(binding,),
    )
    safe = replace(
        generic,
        variant_id="safe-int32",
        dispatch_rank=0,
        compiled_module="_atoll_safe",
        artifact_dir=source_path.parent / "safe-artifacts",
        bindings=(safe_binding,),
    )
    source = """DEFAULT = object()

def choose(value, /, scale=2, *, token=DEFAULT):
    return ("source", value, scale, token)
"""
    for config, label in ((generic, "generic"), (safe, "safe")):
        config.artifact_dir.mkdir(parents=True)
        (config.artifact_dir / f"{config.compiled_module}.py").write_text(
            "def choose(value, scale=2, *, token=None):\n"
            f"    return ({label!r}, value, scale, token)\n",
            encoding="utf-8",
        )
    source_path.write_text(
        insert_or_replace_region_shim(source, (generic, safe)).new_text,
        encoding="utf-8",
    )
    monkeypatch.setattr(importlib.machinery, "EXTENSION_SUFFIXES", (".py",))

    spec = importlib.util.spec_from_file_location("shim_variants", source_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    signature = inspect.signature(module.choose, follow_wrapped=False)
    assert module.choose(3)[0] == "safe"
    assert module.choose("text")[0] == "generic"
    assert signature.parameters["value"].kind is inspect.Parameter.POSITIONAL_ONLY
    assert signature.parameters["token"].default is module.DEFAULT
    assert module.choose.__defaults__ is module.choose.__atoll_python_fallback__.__defaults__
    assert module.choose.__kwdefaults__ is module.choose.__atoll_python_fallback__.__kwdefaults__
    assert module.choose(3)[3] is module.DEFAULT
    with pytest.raises(TypeError):
        module.choose(value=3)
    variants = module.choose.__atoll_binding_variants__
    assert tuple(item["variant_id"] for item in variants) == (
        "safe-int32",
        "generic-cython",
    )
    assert not hasattr(module.choose.__atoll_python_fallback__, "__atoll_binding_variants__")
    assert set(module.__atoll_status__["regions"]) == {"safe-int32", "generic-cython"}

    monkeypatch.setenv("ATOLL_VARIANT_ALLOWLIST", "generic-cython")
    filtered_spec = importlib.util.spec_from_file_location("shim_variant_filtered", source_path)
    assert filtered_spec is not None
    assert filtered_spec.loader is not None
    filtered = importlib.util.module_from_spec(filtered_spec)
    sys.modules[filtered_spec.name] = filtered
    filtered_spec.loader.exec_module(filtered)

    assert filtered.choose(3)[0] == "generic"
    filtered_status = filtered.__atoll_status__["regions"]
    assert filtered_status["safe-int32"]["selected"] is False
    assert filtered_status["generic-cython"]["selected"] is True


def test_region_shim_activates_only_allowlisted_regions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidate trials can route one staged region without loading its peers."""
    base = _config(tmp_path)
    source_path = base.source_path
    scale = replace(
        base,
        source_path=source_path,
        region_id="region-scale",
        compiled_module="_atoll_scale",
        artifact_dir=source_path.parent / "artifacts-scale",
        bindings=(_binding("scale", "instance_method"),),
    )
    score = replace(
        scale,
        region_id="region-score",
        compiled_module="_atoll_score",
        artifact_dir=source_path.parent / "artifacts-score",
        bindings=(_binding("score", "instance_method"),),
    )
    source = """class Worker:
    def scale(self, value: int) -> str:
        return "source-scale"

    def score(self, value: int) -> str:
        return "source-score"
"""
    score.artifact_dir.mkdir(parents=True)
    (score.artifact_dir / f"{score.compiled_module}.py").write_text(
        'def Worker__score(self, value):\n    return "compiled-score"\n',
        encoding="utf-8",
    )
    source_path.write_text(
        insert_or_replace_region_shim(source, (scale, score)).new_text,
        encoding="utf-8",
    )
    monkeypatch.setenv("ATOLL_REGION_ALLOWLIST", score.region_id)
    monkeypatch.setattr(importlib.machinery, "EXTENSION_SUFFIXES", (".py",))

    spec = importlib.util.spec_from_file_location("shim_allowlist", source_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert module.Worker().scale(1) == "source-scale"
    assert module.Worker().score(1) == "compiled-score"
    statuses = module.__atoll_status__["regions"]
    assert statuses[scale.region_id]["selected"] is False
    assert statuses[scale.region_id]["active"] is False
    assert statuses[score.region_id]["selected"] is True
    assert statuses[score.region_id]["active"] is True
    assert module.__atoll_status__["compiled"] is True


@pytest.mark.parametrize("allowlist", ["", "other-module-region"])
def test_region_shim_treats_unmatched_allowlist_as_intentional_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    allowlist: str,
) -> None:
    """Modules outside the candidate set retain source behavior without strict errors."""
    config = replace(
        _config(tmp_path),
        source_path=tmp_path / "pkg" / "worker.py",
        artifact_dir=tmp_path / "pkg" / "missing-artifacts",
        bindings=(_binding("scale", "instance_method"),),
    )
    source = """class Worker:
    def scale(self, value: int) -> str:
        return "source-scale"
"""
    monkeypatch.setenv("ATOLL_REGION_ALLOWLIST", allowlist)
    monkeypatch.setenv("ATOLL_REQUIRE_COMPILED", "1")

    config.source_path.parent.mkdir(parents=True, exist_ok=True)
    config.source_path.write_text(
        insert_or_replace_region_shim(source, (config,)).new_text,
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("shim_empty_allowlist", config.source_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert module.Worker().scale(1) == "source-scale"
    region_status = module.__atoll_status__["regions"][config.region_id]
    assert region_status["selected"] is False
    assert region_status["active"] is False
    assert module.__atoll_status__["compiled"] is True


def test_region_shim_renders_atomic_class_identity_checks(tmp_path: Path) -> None:
    """A class promise validates public identity before replacing the module binding."""
    source_path = tmp_path / "pkg" / "worker.py"
    binding = BindingTarget(
        source=SymbolId(module="pkg.worker", qualname="Worker"),
        compiled_name="Worker",
        kind="class",
        owner_class=None,
        execution_kind="class",
    )
    config = RegionShimConfig(
        source_module="pkg.worker",
        source_path=source_path,
        region_id="pkg.worker::Worker:atomic",
        backend="mypyc",
        compiled_module="_atoll_pkg_worker_atomic",
        artifact_dir=tmp_path / "artifacts",
        bindings=(binding,),
    )

    rendered = render_region_shim((config,))

    compile(rendered, "worker.py", "exec")
    assert "def _atoll_prepare_class" in rendered
    assert "compiled class changed source inheritance" in rendered
    assert "compiled class changed its constructor signature" in rendered
    assert "globals()[_atoll_name] = _atoll_value" in rendered
    assert "'kind': 'class'" in rendered


def test_region_shim_renders_subclass_target_and_constant_time_guard(tmp_path: Path) -> None:
    """Specialized methods read the base descriptor but bind only the target subclass."""
    source_path = tmp_path / "worker.py"
    binding = BindingTarget(
        source=SymbolId(module="pkg.worker", qualname="Pairer.pair"),
        compiled_name="IntPairer__pair__specialized",
        kind="instance_method",
        owner_class="Pairer",
        target_owner_class="IntPairer",
        execution_kind="sync",
        guards=(
            RuntimeTypeGuard(
                parameter_name="value",
                positional_index=1,
                annotation="int | None",
                nominal_type_paths=("int",),
                allow_none=True,
            ),
        ),
    )
    config = RegionShimConfig(
        source_module="pkg.worker",
        source_path=source_path,
        region_id="pairer@int",
        backend="mypyc",
        compiled_module="compiled_pairer",
        artifact_dir=tmp_path / "artifacts",
        bindings=(binding,),
    )

    rendered = render_region_shim((config,))

    compile(rendered, "worker.py", "exec")
    assert "'source_owner_class': 'Pairer'" in rendered
    assert "'target_owner_class': 'IntPairer'" in rendered
    assert "'qualname': 'IntPairer.pair'" in rendered
    assert "'nominal_type_paths': ('int',)" in rendered
    assert "_atoll_guard_check(_atoll_candidate['guards'], _atoll_values)" in rendered
    assert "except Exception:" in rendered


def test_region_shim_insertion_is_idempotent_and_removable(tmp_path: Path) -> None:
    """Balanced markers are replaced in place and can be removed exactly."""
    config = _config(tmp_path)
    source = "class Worker:\n    pass\n"

    first = insert_or_replace_region_shim(source, (config,))
    second = insert_or_replace_region_shim(first.new_text, (config,))
    removed = remove_region_shim(
        second.new_text,
        source_module=config.source_module,
        filename=config.source_path.name,
    )

    assert first.new_text == second.new_text
    assert first.new_text.count("# BEGIN ATOLL TYPED REGIONS") == 1
    assert removed.new_text == source


def test_region_shim_renders_async_generator_protocol_forwarding(tmp_path: Path) -> None:
    """Async generator wrappers preserve send, throw, and close operations."""
    source_path = tmp_path / "worker.py"
    config = RegionShimConfig(
        source_module="pkg.worker",
        source_path=source_path,
        region_id="region@cython",
        backend="cython",
        compiled_module="compiled",
        artifact_dir=tmp_path / "artifacts",
        bindings=(_binding("stream", "instance_method", "async_generator"),),
    )

    rendered = render_region_shim((config,))

    assert "isasyncgenfunction" in rendered
    assert "_atoll_generator.asend" in rendered
    assert "_atoll_generator.athrow" in rendered
    assert "_atoll_generator.aclose" in rendered


def test_region_shim_binds_execution_kinds_descriptors_and_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime wrappers preserve source metadata, defaults, and descriptor kinds."""
    source_path = tmp_path / "pkg" / "worker.py"
    config = replace(
        _config(tmp_path),
        source_path=source_path,
        artifact_dir=source_path.parent / "artifacts",
        bindings=(
            _binding("scale", "instance_method"),
            _binding("parse", "staticmethod"),
            _binding("create", "classmethod"),
            _binding("values", "instance_method", "generator"),
            _binding("score", "instance_method", "coroutine"),
            _binding("stream", "instance_method", "async_generator"),
        ),
    )
    source = '''DEFAULT = object()

class Worker:
    def scale(self, value: int = 1, token: object = DEFAULT) -> object:
        """source scale doc"""
        return ("source-scale", token)

    @staticmethod
    def parse(text: str = "source") -> str:
        """source parse doc"""
        return text

    @classmethod
    def create(cls, token: object = DEFAULT) -> object:
        return token

    def values(self, token: object = DEFAULT):
        yield ("source-values", token)

    async def score(self, token: object = DEFAULT) -> object:
        return token

    async def stream(self, token: object = DEFAULT):
        yield ("source-stream", token)
'''
    compiled = """EVENTS = []

def Worker__scale(self, value=99, token=object()):
    return ("compiled-scale", token)

def Worker__parse(text="compiled"):
    return f"compiled-parse:{text}"

def Worker__create(cls, token=object()):
    return token

def Worker__values(self, token=object()):
    sent = yield ("compiled-values", token)
    yield ("compiled-sent", sent)

async def Worker__score(self, token=object()):
    return token

async def Worker__stream(self, token=object()):
    try:
        sent = yield ("compiled-stream", token)
        EVENTS.append(("asend", sent))
        try:
            yield ("compiled-stream-sent", sent)
        except ValueError as exc:
            EVENTS.append(("athrow", str(exc)))
            yield "handled"
    finally:
        EVENTS.append(("aclose", None))
"""

    module = _load_region_module(config, source, compiled, monkeypatch)
    worker_type = cast(_RuntimeWorkerTypeProtocol, module.Worker)
    worker = worker_type()

    assert vars(worker_type)["parse"].__func__.__name__ == "parse"
    assert isinstance(vars(worker_type)["parse"], staticmethod)
    assert isinstance(vars(worker_type)["create"], classmethod)
    assert worker.scale.__name__ == "scale"
    assert worker.scale.__doc__ == "source scale doc"
    assert inspect.signature(worker.scale).parameters["token"].default is module.DEFAULT
    assert worker.scale()[1] is module.DEFAULT
    assert worker_type.parse() == "compiled-parse:source"
    assert worker_type.create() is module.DEFAULT

    generator = worker.values()
    assert next(generator)[1] is module.DEFAULT
    assert generator.send("sent-value") == ("compiled-sent", "sent-value")
    with pytest.raises(StopIteration):
        next(generator)
    assert asyncio.run(worker.score()) is module.DEFAULT

    async def exercise_async_generator() -> tuple[object, object, object, list[object]]:
        stream = worker.stream()
        first = await anext(stream)
        second = await stream.asend("async-sent")
        handled = await stream.athrow(ValueError("boom"))
        await stream.aclose()
        compiled_module = sys.modules[config.compiled_module]
        return first, second, handled, compiled_module.EVENTS

    first, second, handled, events = asyncio.run(exercise_async_generator())
    first_pair = cast(tuple[str, object], first)
    assert first_pair[1] is module.DEFAULT
    assert second == ("compiled-stream-sent", "async-sent")
    assert handled == "handled"
    assert events == [("asend", "async-sent"), ("athrow", "boom"), ("aclose", None)]


def test_region_shim_binds_outlined_coroutine_shell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An outlined coroutine keeps its Python shell and exact source defaults."""
    source_path = tmp_path / "pkg" / "worker.py"
    config = RegionShimConfig(
        source_module="pkg.worker",
        source_path=source_path,
        region_id="pkg.worker::Worker.score:outline",
        backend="cython",
        compiled_module="_atoll_pkg_worker_score_outline",
        artifact_dir=source_path.parent / "artifacts",
        bindings=(_binding("score", "instance_method", "coroutine"),),
        outlined_shell=OutlinedShellConfig(
            factory_name="_atoll_make_shell",
            factory_source="""def _atoll_make_shell(_atoll_native):
    async def Worker__score(self, token=None):
        return _atoll_native.Worker__score__block_0(self, token)
    return Worker__score
""",
            helper_names=("Worker__score__block_0",),
        ),
    )
    source = """DEFAULT = object()

class Worker:
    async def score(self, token: object = DEFAULT) -> object:
        return ("source", token)
"""
    compiled = """def Worker__score__block_0(self, token):
    return ("compiled", token)
"""

    module = _load_region_module(config, source, compiled, monkeypatch)
    worker = module.Worker()

    assert inspect.iscoroutinefunction(worker.score)
    assert inspect.signature(worker.score).parameters["token"].default is module.DEFAULT
    assert asyncio.run(worker.score()) == ("compiled", module.DEFAULT)
    compiled_target = worker.score.__atoll_compiled_target__
    assert inspect.iscoroutinefunction(compiled_target)
    assert len(compiled_target.__atoll_native_helpers__) == 1
    region_status = module.__atoll_status__["regions"][config.region_id]
    assert region_status["active"] is True
    assert region_status["compiled"] is True


def test_region_shim_rejects_invalid_outlined_helper_transactionally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-synchronous native helper leaves the source binding untouched."""
    source_path = tmp_path / "pkg" / "worker.py"
    config = RegionShimConfig(
        source_module="pkg.worker",
        source_path=source_path,
        region_id="pkg.worker::Worker.score:outline",
        backend="cython",
        compiled_module="_atoll_pkg_worker_score_outline",
        artifact_dir=source_path.parent / "artifacts",
        bindings=(_binding("score", "instance_method", "coroutine"),),
        outlined_shell=OutlinedShellConfig(
            factory_name="_atoll_make_shell",
            factory_source="""def _atoll_make_shell(_atoll_native):
    async def Worker__score(self):
        return _atoll_native.Worker__score__block_0(self)
    return Worker__score
""",
            helper_names=("Worker__score__block_0",),
        ),
    )
    source = """class Worker:
    async def score(self) -> str:
        return "source"
"""
    compiled = """async def Worker__score__block_0(self):
    return "invalid"
"""

    module = _load_region_module(config, source, compiled, monkeypatch)

    assert asyncio.run(module.Worker().score()) == "source"
    region_status = module.__atoll_status__["regions"][config.region_id]
    assert region_status["active"] is False
    assert region_status["compiled"] is False
    error = region_status["bindings"]["Worker.score"]["error"]
    assert "expected sync execution" in str(error)


def test_region_shim_reapplies_outlined_staticmethod_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An outlined private shell is installed with its original static descriptor."""
    source_path = tmp_path / "pkg" / "worker.py"
    binding = _binding("score", "staticmethod", "coroutine")
    config = RegionShimConfig(
        source_module="pkg.worker",
        source_path=source_path,
        region_id="pkg.worker::Worker.score:outline-static",
        backend="cython",
        compiled_module="_atoll_pkg_worker_score_outline_static",
        artifact_dir=source_path.parent / "artifacts",
        bindings=(binding,),
        outlined_shell=OutlinedShellConfig(
            factory_name="_atoll_make_shell",
            factory_source="""def _atoll_make_shell(_atoll_native):
    class Worker:
        async def score(values):
            return _atoll_native.Worker__score__block_0(values)
    return Worker.__dict__["score"]
""",
            helper_names=("Worker__score__block_0",),
        ),
    )
    source = """class Worker:
    @staticmethod
    async def score(values: list[int]) -> int:
        return -1
"""
    compiled = """def Worker__score__block_0(values):
    return sum(values)
"""

    module = _load_region_module(config, source, compiled, monkeypatch)

    assert isinstance(vars(module.Worker)["score"], staticmethod)
    expected_result = 3
    assert asyncio.run(module.Worker.score([1, 2])) == expected_result


def test_region_shim_reapplies_outlined_classmethod_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An outlined classmethod receives the original source owner class."""
    source_path = tmp_path / "pkg" / "worker.py"
    binding = _binding("score", "classmethod", "coroutine")
    config = RegionShimConfig(
        source_module="pkg.worker",
        source_path=source_path,
        region_id="pkg.worker::Worker.score:outline-class",
        backend="cython",
        compiled_module="_atoll_pkg_worker_score_outline_class",
        artifact_dir=source_path.parent / "artifacts",
        bindings=(binding,),
        outlined_shell=OutlinedShellConfig(
            factory_name="_atoll_make_shell",
            factory_source="""def _atoll_make_shell(_atoll_native):
    class Worker:
        async def score(cls, values):
            return _atoll_native.Worker__score__block_0(cls, values)
    return Worker.__dict__["score"]
""",
            helper_names=("Worker__score__block_0",),
        ),
    )
    source = """class Worker:
    @classmethod
    async def score(cls, values: list[int]) -> tuple[str, int]:
        return "source", -1
"""
    compiled = """def Worker__score__block_0(cls, values):
    return cls.__name__, sum(values)
"""

    module = _load_region_module(config, source, compiled, monkeypatch)

    assert isinstance(vars(module.Worker)["score"], classmethod)
    expected_result = ("Worker", 3)
    assert asyncio.run(module.Worker.score([1, 2])) == expected_result


def test_region_shim_preflight_rejects_execution_kind_without_partial_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A required kind mismatch aborts the region before any binding is installed."""
    config = replace(
        _config(tmp_path),
        source_path=tmp_path / "pkg" / "worker.py",
        artifact_dir=tmp_path / "pkg" / "artifacts",
        bindings=(
            _binding("scale", "instance_method"),
            _binding("score", "instance_method", "coroutine"),
        ),
    )
    source = """class Worker:
    def scale(self, value: int) -> str:
        return "source-scale"

    async def score(self) -> str:
        return "source-score"
"""
    compiled = """def Worker__scale(self, value):
    return "compiled-scale"

def Worker__score(self):
    return "wrong-kind"
"""

    module = _load_region_module(config, source, compiled, monkeypatch)

    worker = module.Worker()
    assert worker.scale(1) == "source-scale"
    region_status = module.__atoll_status__["regions"][config.region_id]
    assert region_status["active"] is False
    assert region_status["compiled"] is False
    assert "expected coroutine execution" in str(region_status["bindings"]["Worker.score"]["error"])


def test_region_shim_uses_direct_call_path_without_source_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No-default source callables bypass signature binding at wrapper call time."""
    config = replace(
        _config(tmp_path),
        source_path=tmp_path / "pkg" / "worker.py",
        artifact_dir=tmp_path / "pkg" / "artifacts",
        bindings=(_binding("scale", "instance_method"),),
    )
    source = """class Worker:
    def scale(self, value: int) -> str:
        return "source-scale"
"""
    compiled = """def Worker__scale(self, value):
    return f"compiled:{value}"
"""

    module = _load_region_module(config, source, compiled, monkeypatch)
    worker_type = cast(type[_TextScaleWorkerProtocol], module.Worker)
    worker = worker_type()

    def fail_bind(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Signature.bind should not run without source defaults")

    monkeypatch.setattr(inspect.Signature, "bind", fail_bind)

    assert worker.scale(7) == "compiled:7"


def test_region_shim_rolls_back_when_apply_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later setattr failure restores earlier replacements from the same transaction."""
    config = replace(
        _config(tmp_path),
        source_path=tmp_path / "pkg" / "worker.py",
        artifact_dir=tmp_path / "pkg" / "artifacts",
        bindings=(
            _binding("scale", "instance_method"),
            _binding("parse", "staticmethod"),
        ),
    )
    source = """class BlockParse(type):
    def __setattr__(cls, name, value):
        if name == "parse":
            raise RuntimeError("blocked setattr")
        super().__setattr__(name, value)

class Worker(metaclass=BlockParse):
    def scale(self, value: int) -> str:
        return "source-scale"

    @staticmethod
    def parse(text: str) -> str:
        return "source-parse"
"""
    compiled = """def Worker__scale(self, value):
    return "compiled-scale"

def Worker__parse(text):
    return "compiled-parse"
"""

    module = _load_region_module(config, source, compiled, monkeypatch)

    assert module.Worker().scale(1) == "source-scale"
    assert module.Worker.parse("value") == "source-parse"
    region_status = module.__atoll_status__["regions"][config.region_id]
    assert region_status["active"] is False
    assert region_status["compiled"] is False
    assert "RuntimeError('blocked setattr')" in str(region_status["rollback_errors"])
    assert "RuntimeError" in str(module.__atoll_status__["error"])


def test_region_shim_falls_back_without_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unavailable native artifacts retain source methods and clean helpers."""
    config = _config(tmp_path)
    source = """class Worker:
    def scale(self, value: int) -> int:
        return value + 1
"""
    monkeypatch.delenv("ATOLL_REQUIRE_COMPILED", raising=False)
    monkeypatch.delenv("ATOLL_STRICT", raising=False)

    config.source_path.write_text(
        insert_or_replace_region_shim(source, (config,)).new_text,
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("shim_fallback", config.source_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    namespace = vars(module)

    worker_type = cast(type[_WorkerProtocol], namespace["Worker"])
    assert worker_type().scale(3) == FALLBACK_RESULT
    status = cast(dict[str, object], namespace["__atoll_status__"])
    assert status["compiled"] is False
    assert "ImportError" in str(status["error"])
    assert not any(name.startswith("_atoll_") for name in namespace)


def test_region_shim_config_rejects_invalid_promises(tmp_path: Path) -> None:
    """Malformed identifiers and unsupported binding targets fail immediately."""
    config = _config(tmp_path)
    binding = config.bindings[0]

    with pytest.raises(ValueError, match="identifiers"):
        replace(config, source_module="")
    with pytest.raises(ValueError, match="at least one"):
        replace(config, bindings=())
    with pytest.raises(ValueError, match="factory and native helpers"):
        OutlinedShellConfig(factory_name="", factory_source="", helper_names=())
    with pytest.raises(ValueError, match="must be unique"):
        OutlinedShellConfig(
            factory_name="factory",
            factory_source="def factory(native):\n    return native\n",
            helper_names=("helper", "helper"),
        )
    shell = OutlinedShellConfig(
        factory_name="factory",
        factory_source="def factory(native):\n    return native\n",
        helper_names=("helper",),
    )
    with pytest.raises(ValueError, match="exactly one public binding"):
        replace(config, outlined_shell=shell)
    with pytest.raises(ValueError, match="another source module"):
        replace(
            config,
            bindings=(replace(binding, source=SymbolId("other", "Worker.scale")),),
        )
    with pytest.raises(ValueError, match="module or class region shim binding"):
        replace(config, bindings=(replace(binding, kind="class"),))
    with pytest.raises(ValueError, match="owner class"):
        replace(config, bindings=(replace(binding, owner_class=None),))
    with pytest.raises(ValueError, match="module or class region shim binding"):
        replace(
            config,
            bindings=(
                BindingTarget(
                    source=SymbolId("pkg.worker", "score"),
                    compiled_name="score",
                    kind="module",
                    owner_class="Worker",
                    execution_kind="sync",
                ),
            ),
        )


def test_region_shim_rejects_ambiguous_config_sets_and_markers(tmp_path: Path) -> None:
    """One staged module cannot contain duplicate variants or unbalanced blocks."""
    config = _config(tmp_path)
    other_path = replace(config, source_path=tmp_path / "other.py", region_id="other")

    with pytest.raises(ValueError, match="at least one region config"):
        render_region_shim(())
    with pytest.raises(ValueError, match="one source module"):
        render_region_shim((config, other_path))
    with pytest.raises(ValueError, match="unique variant IDs"):
        render_region_shim((config, config))
    incompatible = replace(
        config,
        variant_id="other-variant",
        bindings=(replace(config.bindings[0], execution_kind="coroutine"),),
    )
    with pytest.raises(ValueError, match="disagree about binding"):
        render_region_shim((config, incompatible))
    with pytest.raises(ValueError, match="unbalanced"):
        remove_region_shim(
            "# BEGIN ATOLL TYPED REGIONS: pkg.worker\n",
            source_module="pkg.worker",
            filename="worker.py",
        )
    duplicate = "\n".join(
        (
            "# BEGIN ATOLL TYPED REGIONS: pkg.worker",
            "# END ATOLL TYPED REGIONS: pkg.worker",
            "# BEGIN ATOLL TYPED REGIONS: pkg.worker",
            "# END ATOLL TYPED REGIONS: pkg.worker",
        )
    )
    with pytest.raises(ValueError, match="multiple"):
        remove_region_shim(
            duplicate,
            source_module="pkg.worker",
            filename="worker.py",
        )

    untouched = remove_region_shim(
        "value = 1\n",
        source_module="pkg.worker",
        filename="worker.py",
    )
    assert untouched.new_text == "value = 1\n"
