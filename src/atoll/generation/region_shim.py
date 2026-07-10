"""Generate staged-wheel shims for class and descriptor typed-region bindings.

Unlike the legacy in-place shim, this block is inserted only into a copied
package payload. It loads region artifacts by file location, installs verified
atomic classes or binds compiled callables onto original source classes, and
records status per promised binding so interpreted members do not become strict
failures.
"""

from __future__ import annotations

import difflib
import os
from dataclasses import dataclass
from pathlib import Path

from atoll.models import Backend, BindingTarget

_MARKER_LABEL = "ATOLL TYPED REGIONS"


@dataclass(frozen=True, slots=True)
class RegionShimConfig:
    """Runtime loading and binding contract for one compiled typed region."""

    source_module: str
    source_path: Path
    region_id: str
    backend: Backend
    compiled_module: str
    artifact_dir: Path
    bindings: tuple[BindingTarget, ...]

    def __post_init__(self) -> None:
        """Reject configs the current runtime binder cannot preserve."""
        if not self.source_module or not self.region_id or not self.compiled_module:
            raise ValueError("region shim identifiers must be non-empty")
        if not self.bindings:
            raise ValueError("region shim requires at least one promised binding")
        for binding in self.bindings:
            if binding.source.module != self.source_module:
                raise ValueError("region shim binding belongs to another source module")
            if binding.kind not in {
                "module",
                "class",
                "instance_method",
                "staticmethod",
                "classmethod",
            }:
                raise ValueError(f"unsupported region shim binding: {binding.kind}")
            if binding.kind in {"instance_method", "staticmethod", "classmethod"} and (
                binding.owner_class is None
            ):
                raise ValueError("method region shim binding requires an owner class")
            if binding.kind in {"module", "class"} and (
                binding.owner_class is not None or binding.target_owner_class is not None
            ):
                raise ValueError("module or class region shim binding cannot name an owner class")


@dataclass(frozen=True, slots=True)
class RegionShimEdit:
    """Original and updated staged source plus a reviewable unified diff."""

    old_text: str
    new_text: str
    diff: str


def insert_or_replace_region_shim(
    source_text: str,
    configs: tuple[RegionShimConfig, ...],
) -> RegionShimEdit:
    """Append or replace one module-level typed-region runtime block."""
    source_module = _validate_configs(configs)
    new_text = _replace_block(source_text, source_module, render_region_shim(configs))
    return _edit(source_text, new_text, configs[0].source_path.name)


def remove_region_shim(
    source_text: str,
    *,
    source_module: str,
    filename: str,
) -> RegionShimEdit:
    """Remove a typed-region block while rejecting ambiguous markers."""
    new_text = _replace_block(source_text, source_module, "")
    return _edit(source_text, new_text, filename)


def render_region_shim(configs: tuple[RegionShimConfig, ...]) -> str:
    """Render a staged-wheel loader for guarded functions and descriptors."""
    source_module = _validate_configs(configs)
    regions = tuple(_runtime_region(config) for config in configs)
    promised_symbols = tuple(
        _binding_runtime_qualname(binding) for config in configs for binding in config.bindings
    )
    return "\n".join(
        [
            _begin_marker(source_module),
            "# This staged-wheel block is managed by Atoll. Do not edit manually.",
            "try:",
            "    _atoll_preexisting_names = frozenset(globals())",
            "    import builtins as _atoll_builtins",
            "    import functools as _atoll_functools",
            "    import importlib.machinery as _atoll_machinery",
            "    import importlib.util as _atoll_util",
            "    import inspect as _atoll_inspect",
            "    import os as _atoll_os",
            "    import pathlib as _atoll_pathlib",
            "    import sys as _atoll_sys",
            "",
            "    def _atoll_resolve_type(_atoll_path):",
            "        _atoll_parts = _atoll_path.split('.')",
            "        _atoll_value = globals().get(_atoll_parts[0])",
            "        if _atoll_value is None:",
            "            _atoll_value = getattr(_atoll_builtins, _atoll_parts[0])",
            "        for _atoll_part in _atoll_parts[1:]:",
            "            _atoll_value = getattr(_atoll_value, _atoll_part)",
            "        if not isinstance(_atoll_value, type):",
            "            raise TypeError(f'Atoll guard is not a nominal type: {_atoll_path}')",
            "        return _atoll_value",
            "",
            "    def _atoll_resolve_guards(_atoll_guards):",
            "        return tuple(",
            "            {",
            "                **_atoll_guard,",
            "                'types': tuple(",
            "                    _atoll_resolve_type(_atoll_path)",
            "                    for _atoll_path in _atoll_guard['nominal_type_paths']",
            "                ),",
            "            }",
            "            for _atoll_guard in _atoll_guards",
            "        )",
            "",
            "    def _atoll_guards_pass(_atoll_guards, _atoll_args, _atoll_kwargs):",
            "        for _atoll_guard in _atoll_guards:",
            "            _atoll_index = _atoll_guard['positional_index']",
            "            _atoll_parameter = _atoll_guard['parameter_name']",
            "            if _atoll_index is not None and _atoll_index < len(_atoll_args):",
            "                _atoll_value = _atoll_args[_atoll_index]",
            "            elif _atoll_parameter in _atoll_kwargs:",
            "                _atoll_value = _atoll_kwargs[_atoll_parameter]",
            "            else:",
            "                return False",
            "            if _atoll_value is None and _atoll_guard['allow_none']:",
            "                continue",
            "            if _atoll_guard['types']:",
            "                try:",
            "                    _atoll_matches = isinstance(",
            "                        _atoll_value, _atoll_guard['types']",
            "                    )",
            "                except Exception:",
            "                    return False",
            "                if _atoll_matches:",
            "                    continue",
            "            return False",
            "        return True",
            "",
            "    def _atoll_bind(_atoll_source, _atoll_target, _atoll_guards):",
            "        _atoll_guard_check = _atoll_guards_pass",
            "        if _atoll_inspect.isasyncgenfunction(_atoll_source):",
            "            @_atoll_functools.wraps(_atoll_source)",
            "            async def _atoll_wrapped(*args, **kwargs):",
            "                _atoll_callable = (",
            "                    _atoll_target",
            "                    if _atoll_guard_check(_atoll_guards, args, kwargs)",
            "                    else _atoll_source",
            "                )",
            "                _atoll_generator = _atoll_callable(*args, **kwargs)",
            "                try:",
            "                    _atoll_value = await _atoll_generator.__anext__()",
            "                    while True:",
            "                        try:",
            "                            _atoll_sent = yield _atoll_value",
            "                        except GeneratorExit:",
            "                            await _atoll_generator.aclose()",
            "                            raise",
            "                        except BaseException as _atoll_thrown:",
            "                            _atoll_value = await _atoll_generator.athrow(",
            "                                _atoll_thrown",
            "                            )",
            "                        else:",
            "                            if _atoll_sent is None:",
            "                                _atoll_value = await _atoll_generator.__anext__()",
            "                            else:",
            "                                _atoll_value = await _atoll_generator.asend(",
            "                                    _atoll_sent",
            "                                )",
            "                except StopAsyncIteration:",
            "                    return",
            "        elif _atoll_inspect.iscoroutinefunction(_atoll_source):",
            "            @_atoll_functools.wraps(_atoll_source)",
            "            async def _atoll_wrapped(*args, **kwargs):",
            "                _atoll_callable = (",
            "                    _atoll_target",
            "                    if _atoll_guard_check(_atoll_guards, args, kwargs)",
            "                    else _atoll_source",
            "                )",
            "                return await _atoll_callable(*args, **kwargs)",
            "        elif _atoll_inspect.isgeneratorfunction(_atoll_source):",
            "            @_atoll_functools.wraps(_atoll_source)",
            "            def _atoll_wrapped(*args, **kwargs):",
            "                _atoll_callable = (",
            "                    _atoll_target",
            "                    if _atoll_guard_check(_atoll_guards, args, kwargs)",
            "                    else _atoll_source",
            "                )",
            "                return (yield from _atoll_callable(*args, **kwargs))",
            "        else:",
            "            @_atoll_functools.wraps(_atoll_source)",
            "            def _atoll_wrapped(*args, **kwargs):",
            "                _atoll_callable = (",
            "                    _atoll_target",
            "                    if _atoll_guard_check(_atoll_guards, args, kwargs)",
            "                    else _atoll_source",
            "                )",
            "                return _atoll_callable(*args, **kwargs)",
            "        try:",
            "            _atoll_wrapped.__signature__ = _atoll_inspect.signature(_atoll_source)",
            "        except (TypeError, ValueError):",
            "            pass",
            "        _atoll_wrapped.__atoll_compiled_target__ = _atoll_target",
            "        _atoll_wrapped.__atoll_python_fallback__ = _atoll_source",
            "        _atoll_wrapped.__atoll_runtime_guards__ = _atoll_guards",
            "        return _atoll_wrapped",
            "",
            "    def _atoll_callable_descriptor(_atoll_descriptor):",
            "        if isinstance(_atoll_descriptor, (staticmethod, classmethod)):",
            "            return _atoll_descriptor.__func__",
            '        if callable(_atoll_descriptor) and hasattr(_atoll_descriptor, "__name__"):',
            "            return _atoll_descriptor",
            "        return None",
            "",
            "    def _atoll_prepare_class(_atoll_source, _atoll_target):",
            (
                "        if not isinstance(_atoll_source, type) or not "
                "isinstance(_atoll_target, type):"
            ),
            '            raise TypeError("Atoll class binding requires two class objects")',
            "        if type(_atoll_source) is not type(_atoll_target):",
            '            raise TypeError("compiled class changed the source metaclass")',
            "        if _atoll_source.__bases__ != _atoll_target.__bases__:",
            '            raise TypeError("compiled class changed source inheritance")',
            "        _atoll_signature = _atoll_inspect.signature(_atoll_source)",
            '        for _atoll_attribute in ("__module__", "__qualname__", "__doc__"):',
            "            _atoll_expected = getattr(_atoll_source, _atoll_attribute)",
            "            if getattr(_atoll_target, _atoll_attribute) != _atoll_expected:",
            "                setattr(_atoll_target, _atoll_attribute, _atoll_expected)",
            "        _atoll_source_annotations = dict(",
            '            getattr(_atoll_source, "__annotations__", {})',
            "        )",
            "        _atoll_target_annotations = getattr(",
            '            _atoll_target, "__annotations__", None',
            "        )",
            "        if isinstance(_atoll_target_annotations, dict):",
            "            _atoll_target_annotations.clear()",
            "            _atoll_target_annotations.update(_atoll_source_annotations)",
            "        elif _atoll_source_annotations:",
            '            setattr(_atoll_target, "__annotations__", _atoll_source_annotations)',
            "        _atoll_target_namespace = vars(_atoll_target)",
            "        for _atoll_method_name, _atoll_source_descriptor in vars(",
            "            _atoll_source",
            "        ).items():",
            "            _atoll_source_callable = _atoll_callable_descriptor(",
            "                _atoll_source_descriptor",
            "            )",
            "            if _atoll_source_callable is None:",
            "                continue",
            "            _atoll_target_callable = _atoll_callable_descriptor(",
            "                _atoll_target_namespace.get(_atoll_method_name)",
            "            )",
            "            if _atoll_target_callable is None:",
            "                raise TypeError(",
            '                    f"compiled class lost method {_atoll_method_name}"',
            "                )",
            "            try:",
            "                _atoll_functools.update_wrapper(",
            "                    _atoll_target_callable, _atoll_source_callable",
            "                )",
            "                _atoll_target_callable.__signature__ = (",
            "                    _atoll_inspect.signature(_atoll_source_callable)",
            "                )",
            "                _atoll_target_callable.__atoll_compiled_target__ = (",
            "                    _atoll_target_callable",
            "                )",
            "                _atoll_target_callable.__atoll_python_fallback__ = (",
            "                    _atoll_source_callable",
            "                )",
            "            except (AttributeError, TypeError, ValueError) as _atoll_metadata_error:",
            "                raise TypeError(",
            '                    f"compiled class cannot preserve {_atoll_method_name} metadata"',
            "                ) from _atoll_metadata_error",
            "            if _atoll_inspect.signature(_atoll_target_callable) != (",
            "                _atoll_inspect.signature(_atoll_source_callable)",
            "            ):",
            "                raise TypeError(",
            '                    f"compiled class changed {_atoll_method_name} signature"',
            "                )",
            "        try:",
            "            _atoll_target.__signature__ = _atoll_signature",
            "        except (AttributeError, TypeError):",
            "            if _atoll_inspect.signature(_atoll_target) != _atoll_signature:",
            '                raise TypeError("compiled class changed its constructor signature")',
            "        if _atoll_target.__module__ != _atoll_source.__module__:",
            '            raise TypeError("compiled class changed its public module")',
            "        if _atoll_target.__qualname__ != _atoll_source.__qualname__:",
            '            raise TypeError("compiled class changed its public qualname")',
            '        if dict(getattr(_atoll_target, "__annotations__", {})) != (',
            "            _atoll_source_annotations",
            "        ):",
            '            raise TypeError("compiled class changed its annotations")',
            "        _atoll_target.__atoll_compiled_target__ = _atoll_target",
            "        _atoll_target.__atoll_python_fallback__ = _atoll_source",
            "        return _atoll_target",
            "",
            f"    _atoll_regions = {regions!r}",
            "    __atoll_region_status__ = {}",
            "    __atoll_status__ = {",
            f'        "source_module": {source_module!r},',
            '        "sidecar_module": None,',
            '        "active": False,',
            '        "compiled": False,',
            f'        "symbols": {promised_symbols!r},',
            '        "origin": None,',
            '        "error": None,',
            '        "regions": __atoll_region_status__,',
            "    }",
            "    _atoll_origins = []",
            "    _atoll_errors = []",
            "",
            '    if _atoll_os.getenv("ATOLL_DISABLE") != "1":',
            "        for _atoll_region in _atoll_regions:",
            "            _atoll_region_status = {",
            '                "backend": _atoll_region["backend"],',
            '                "compiled_module": _atoll_region["compiled_module"],',
            '                "active": False,',
            '                "compiled": False,',
            '                "origin": None,',
            '                "error": None,',
            '                "bindings": {},',
            "            }",
            '            __atoll_region_status__[_atoll_region["region_id"]] = (',
            "                _atoll_region_status",
            "            )",
            "            _atoll_added_path = False",
            "            _atoll_artifact_dir_text = None",
            "            try:",
            "                _atoll_source_dir = _atoll_pathlib.Path(__file__).resolve().parent",
            "                _atoll_artifact_dir = (",
            '                    _atoll_source_dir / _atoll_region["artifact_relative"]',
            "                ).resolve()",
            '                _atoll_stem = _atoll_region["compiled_module"].rsplit(".", 1)[-1]',
            "                _atoll_paths = tuple(",
            "                    sorted(",
            "                        _atoll_candidate",
            "                        for _atoll_suffix in _atoll_machinery.EXTENSION_SUFFIXES",
            "                        for _atoll_candidate in _atoll_artifact_dir.rglob(",
            '                            f"{_atoll_stem}*{_atoll_suffix}"',
            "                        )",
            "                    )",
            "                )",
            "                if not _atoll_paths:",
            "                    raise ImportError(",
            "                        f\"Atoll region {_atoll_region['region_id']} has no \"",
            '                        "compiled extension"',
            "                    )",
            "                _atoll_origin_path = _atoll_paths[0]",
            "                _atoll_artifact_dir_text = str(_atoll_artifact_dir)",
            "                if _atoll_artifact_dir_text not in _atoll_sys.path:",
            "                    _atoll_sys.path.insert(0, _atoll_artifact_dir_text)",
            "                    _atoll_added_path = True",
            "                _atoll_spec = _atoll_util.spec_from_file_location(",
            '                    _atoll_region["compiled_module"], _atoll_origin_path',
            "                )",
            "                if _atoll_spec is None or _atoll_spec.loader is None:",
            '                    raise ImportError("Atoll region extension cannot be loaded")',
            "                _atoll_mod = _atoll_util.module_from_spec(_atoll_spec)",
            '                _atoll_sys.modules[_atoll_region["compiled_module"]] = _atoll_mod',
            "                _atoll_spec.loader.exec_module(_atoll_mod)",
            "                if _atoll_added_path and _atoll_artifact_dir_text is not None:",
            "                    _atoll_sys.path.remove(_atoll_artifact_dir_text)",
            "                    _atoll_added_path = False",
            "",
            '                for _atoll_binding in _atoll_region["bindings"]:',
            "                    _atoll_binding_status = {",
            '                        "required": _atoll_binding["required"],',
            '                        "active": False,',
            '                        "compiled": False,',
            '                        "error": None,',
            "                    }",
            '                    _atoll_region_status["bindings"][_atoll_binding["qualname"]] = (',
            "                        _atoll_binding_status",
            "                    )",
            "                    try:",
            "                        _atoll_source_qualname = (",
            '                            _atoll_binding["source_qualname"]',
            "                        )",
            '                        _atoll_name = _atoll_source_qualname.rsplit(".", 1)[-1]',
            '                        if _atoll_binding["kind"] in {"module", "class"}:',
            "                            _atoll_source = globals()[_atoll_name]",
            "                            _atoll_target_owner = None",
            "                        else:",
            "                            _atoll_source_owner = globals()[",
            '                                _atoll_binding["source_owner_class"]',
            "                            ]",
            "                            _atoll_target_owner = globals()[",
            '                                _atoll_binding["target_owner_class"]',
            "                            ]",
            "                            _atoll_descriptor = vars(_atoll_source_owner)[",
            "                                _atoll_name",
            "                            ]",
            '                            if _atoll_binding["kind"] in {',
            '                                "staticmethod", "classmethod"',
            "                            }:",
            "                                _atoll_source = _atoll_descriptor.__func__",
            "                            else:",
            "                                _atoll_source = _atoll_descriptor",
            "                        _atoll_target = getattr(",
            '                            _atoll_mod, _atoll_binding["compiled_name"]',
            "                        )",
            '                        if _atoll_binding["kind"] == "class":',
            "                            _atoll_target = _atoll_prepare_class(",
            "                                _atoll_source, _atoll_target",
            "                            )",
            "                            globals()[_atoll_name] = _atoll_target",
            "                        else:",
            "                            _atoll_guards = _atoll_resolve_guards(",
            '                                _atoll_binding["guards"]',
            "                            )",
            "                            _atoll_wrapped = _atoll_bind(",
            "                                _atoll_source, _atoll_target, _atoll_guards",
            "                            )",
            '                            if _atoll_binding["kind"] == "module":',
            "                                globals()[_atoll_name] = _atoll_wrapped",
            '                            elif _atoll_binding["kind"] == "staticmethod":',
            "                                setattr(",
            "                                    _atoll_target_owner,",
            "                                    _atoll_name,",
            "                                    staticmethod(_atoll_wrapped),",
            "                                )",
            '                            elif _atoll_binding["kind"] == "classmethod":',
            "                                setattr(",
            "                                    _atoll_target_owner,",
            "                                    _atoll_name,",
            "                                    classmethod(_atoll_wrapped),",
            "                                )",
            "                            else:",
            "                                setattr(",
            "                                    _atoll_target_owner, _atoll_name, _atoll_wrapped",
            "                                )",
            '                        _atoll_binding_status["active"] = True',
            '                        _atoll_binding_status["compiled"] = True',
            "                    except Exception as _atoll_binding_error:",
            '                        _atoll_binding_status["error"] = repr(_atoll_binding_error)',
            '                        if _atoll_binding["required"]:',
            "                            _atoll_errors.append(_atoll_binding_error)",
            '                _atoll_region_status["active"] = any(',
            '                    item["active"]',
            '                    for item in _atoll_region_status["bindings"].values()',
            "                )",
            '                _atoll_region_status["compiled"] = all(',
            '                    item["compiled"]',
            '                    for item in _atoll_region_status["bindings"].values()',
            '                    if item["required"]',
            "                )",
            '                _atoll_region_status["origin"] = str(_atoll_origin_path)',
            "                _atoll_origins.append(str(_atoll_origin_path))",
            "            except Exception as _atoll_region_error:",
            '                _atoll_region_status["error"] = repr(_atoll_region_error)',
            "                _atoll_errors.append(_atoll_region_error)",
            "            finally:",
            "                if _atoll_added_path and _atoll_artifact_dir_text is not None:",
            "                    try:",
            "                        _atoll_sys.path.remove(_atoll_artifact_dir_text)",
            "                    except ValueError:",
            "                        pass",
            "",
            '        __atoll_status__["active"] = any(',
            '            region["active"] for region in __atoll_region_status__.values()',
            "        )",
            '        __atoll_status__["compiled"] = bool(__atoll_region_status__) and all(',
            '            region["compiled"] for region in __atoll_region_status__.values()',
            "        )",
            '        __atoll_status__["origin"] = tuple(_atoll_origins)',
            "        if _atoll_errors:",
            '            __atoll_status__["error"] = repr(_atoll_errors[0])',
            "        if _atoll_errors and (",
            '            _atoll_os.getenv("ATOLL_STRICT") == "1"',
            '            or _atoll_os.getenv("ATOLL_REQUIRE_COMPILED") == "1"',
            "        ):",
            "            raise _atoll_errors[0]",
            "finally:",
            "    for _atoll_name_to_remove in tuple(globals()):",
            "        if (",
            '            _atoll_name_to_remove.startswith("_atoll_")',
            "            and _atoll_name_to_remove not in _atoll_preexisting_names",
            '            and _atoll_name_to_remove != "_atoll_preexisting_names"',
            "        ):",
            "            globals().pop(_atoll_name_to_remove, None)",
            '    globals().pop("_atoll_name_to_remove", None)',
            '    globals().pop("_atoll_preexisting_names", None)',
            _end_marker(source_module),
            "",
        ]
    )


def _runtime_region(config: RegionShimConfig) -> dict[str, object]:
    return {
        "region_id": config.region_id,
        "backend": config.backend,
        "compiled_module": config.compiled_module,
        "artifact_relative": _relative_path_text(config.source_path.parent, config.artifact_dir),
        "bindings": tuple(
            {
                "qualname": _binding_runtime_qualname(binding),
                "source_qualname": binding.source.qualname,
                "compiled_name": binding.compiled_name,
                "kind": binding.kind,
                "source_owner_class": binding.owner_class,
                "target_owner_class": binding.target_owner_class or binding.owner_class,
                "execution_kind": binding.execution_kind,
                "required": binding.required,
                "guards": tuple(
                    {
                        "parameter_name": guard.parameter_name,
                        "positional_index": guard.positional_index,
                        "annotation": guard.annotation,
                        "nominal_type_paths": guard.nominal_type_paths,
                        "allow_none": guard.allow_none,
                    }
                    for guard in binding.guards
                ),
            }
            for binding in config.bindings
        ),
    }


def _binding_runtime_qualname(binding: BindingTarget) -> str:
    """Return the binding key users see, including concrete subclass targets."""
    member_name = binding.source.qualname.rsplit(".", maxsplit=1)[-1]
    if binding.target_owner_class is not None:
        return f"{binding.target_owner_class}.{member_name}"
    return binding.source.qualname


def _validate_configs(configs: tuple[RegionShimConfig, ...]) -> str:
    if not configs:
        raise ValueError("typed-region shim requires at least one region config")
    source_modules = {config.source_module for config in configs}
    source_paths = {config.source_path.resolve() for config in configs}
    region_ids = [config.region_id for config in configs]
    if len(source_modules) != 1 or len(source_paths) != 1:
        raise ValueError("typed-region shim configs must target one source module")
    if len(region_ids) != len(set(region_ids)):
        raise ValueError("typed-region shim configs must use unique region IDs")
    return configs[0].source_module


def _replace_block(source_text: str, source_module: str, block: str) -> str:
    begin = _begin_marker(source_module)
    end = _end_marker(source_module)
    begin_count = source_text.count(begin)
    end_count = source_text.count(end)
    if begin_count != end_count:
        raise ValueError(f"unbalanced typed-region markers for {source_module}")
    if begin_count > 1:
        raise ValueError(f"multiple typed-region blocks for {source_module}")
    if begin_count == 0:
        if not block:
            return source_text
        return f"{source_text.rstrip()}\n\n{block}"
    start = source_text.index(begin)
    stop = source_text.index(end, start) + len(end)
    replacement = block.rstrip()
    prefix = source_text[:start].rstrip()
    suffix = source_text[stop:].lstrip("\n")
    if replacement and suffix:
        return f"{prefix}\n\n{replacement}\n\n{suffix}"
    if replacement:
        return f"{prefix}\n\n{replacement}\n"
    if suffix:
        return f"{prefix}\n\n{suffix}"
    return f"{prefix}\n"


def _edit(old_text: str, new_text: str, filename: str) -> RegionShimEdit:
    return RegionShimEdit(
        old_text=old_text,
        new_text=new_text,
        diff="".join(
            difflib.unified_diff(
                old_text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=f"{filename}:before",
                tofile=f"{filename}:after",
            )
        ),
    )


def _begin_marker(source_module: str) -> str:
    return f"# BEGIN {_MARKER_LABEL}: {source_module}"


def _end_marker(source_module: str) -> str:
    return f"# END {_MARKER_LABEL}: {source_module}"


def _relative_path_text(start: Path, path: Path) -> str:
    return os.path.relpath(os.fspath(path.resolve()), start=os.fspath(start.resolve()))
