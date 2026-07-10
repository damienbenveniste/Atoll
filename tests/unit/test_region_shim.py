"""Tests for staged-wheel typed-region shim generation."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from typing import Protocol, cast

import pytest

from atoll.generation.region_shim import (
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


def test_region_shim_renders_descriptor_and_status_contract(tmp_path: Path) -> None:
    """The staged block records and binds every promised method independently."""
    rendered = render_region_shim((_config(tmp_path),))

    compile(rendered, "worker.py", "exec")
    assert "# BEGIN ATOLL TYPED REGIONS: pkg.worker" in rendered
    assert "__atoll_region_status__" in rendered
    assert "staticmethod(_atoll_wrapped)" in rendered
    assert "classmethod(_atoll_wrapped)" in rendered
    assert "isgeneratorfunction" in rendered
    assert "iscoroutinefunction" in rendered
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
    assert "globals()[_atoll_name] = _atoll_target" in rendered
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
