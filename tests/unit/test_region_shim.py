"""Tests for staged-wheel typed-region shim generation."""

from __future__ import annotations

import asyncio
import importlib.machinery
import importlib.util
import inspect
import sys
from collections.abc import AsyncGenerator, Coroutine, Generator
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
    assert "__atoll_compiled_target__" in rendered
    assert "__atoll_python_fallback__" in rendered
    assert "_atoll_guards_pass" in rendered
    assert "_atoll_name_to_remove.startswith" in rendered


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
    assert "if _atoll_guard_check(_atoll_guards, args, kwargs)" in rendered
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
    """One staged module cannot contain duplicate regions or unbalanced blocks."""
    config = _config(tmp_path)
    other_path = replace(config, source_path=tmp_path / "other.py", region_id="other")

    with pytest.raises(ValueError, match="at least one region config"):
        render_region_shim(())
    with pytest.raises(ValueError, match="one source module"):
        render_region_shim((config, other_path))
    with pytest.raises(ValueError, match="unique region IDs"):
        render_region_shim((config, config))
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
