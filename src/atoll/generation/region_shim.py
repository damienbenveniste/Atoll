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
from atoll.native_optimization.models import (
    BufferLayoutGuardPayload,
    CallableCodeIdentityGuardPayload,
    DirectFieldGuardPayload,
    ExactTypeGuardPayload,
    GuardExpression,
    IntegerDomainGuardPayload,
)

_MARKER_LABEL = "ATOLL TYPED REGIONS"


@dataclass(frozen=True, slots=True)
class OutlinedShellConfig:
    """Private Python shell factory backed by synchronous native helpers.

    The factory source is inserted only into the staged wheel shim. Calling the
    factory with the loaded native module returns a coroutine, generator, or
    async-generator shell whose suspension protocol remains Python-owned.

    Attributes:
        factory_name: Private factory function defined by `factory_source`.
        factory_source: Deterministic source defining the private shell factory.
        helper_names: Synchronous native helper attributes required by the shell.
    """

    factory_name: str
    factory_source: str
    helper_names: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject incomplete shell contracts before rendering executable code.

        Raises:
            ValueError: If the factory identity, source, or helper set is empty.
        """
        if not self.factory_name or not self.factory_source.strip() or not self.helper_names:
            raise ValueError("outlined shell requires a factory and native helpers")
        if len(set(self.helper_names)) != len(self.helper_names):
            raise ValueError("outlined shell helper names must be unique")


@dataclass(frozen=True, slots=True)
class RegionShimConfig:
    """Runtime loading and binding contract for one compiled typed region.

    Attributes:
        source_module: Importable source module name.
        source_path: Filesystem path of the source module or prepared source.
        region_id: Stable typed-region identifier owning the artifact or unit.
        backend: Native compiler backend selected for this record.
        compiled_module: Importable native module loaded by the managed region shim.
        artifact_dir: Directory from which the runtime shim loads native artifacts.
        bindings: Source bindings promised by the compiled region or variant.
        outlined_shell: Optional Python suspension shell backed by this region's helpers.
        variant_id: Stable dispatcher-variant identity. The region ID is used when omitted.
        dispatch_rank: Explicit dispatcher priority. Generic mypyc/Cython defaults are used when
            omitted; specialized lowerings provide a smaller rank.
        variant_guards: Structured checks applied before selecting this compiled variant.
    """

    source_module: str
    source_path: Path
    region_id: str
    backend: Backend
    compiled_module: str
    artifact_dir: Path
    bindings: tuple[BindingTarget, ...]
    outlined_shell: OutlinedShellConfig | None = None
    variant_id: str | None = None
    dispatch_rank: int | None = None
    variant_guards: tuple[GuardExpression, ...] = ()

    def __post_init__(self) -> None:
        """Reject configs the current runtime binder cannot preserve.

        Raises:
            ValueError: If identifiers are empty or bindings are absent, cross-module, or
                unsupported by the runtime binder.
        """
        if not self.source_module or not self.region_id or not self.compiled_module:
            raise ValueError("region shim identifiers must be non-empty")
        self._validate_dispatch_identity()
        if not self.bindings:
            raise ValueError("region shim requires at least one promised binding")
        if self.outlined_shell is not None and len(self.bindings) != 1:
            raise ValueError("outlined region shim requires exactly one public binding")
        self._validate_variant_guards()
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

    def _validate_variant_guards(self) -> None:
        """Reject unsupported or cross-module structured dispatcher guards.

        Raises:
            ValueError: If a guard kind is unsupported or pins a callable in another module.
        """
        supported = {
            "exact-type",
            "integer-domain",
            "direct-field",
            "callable-code-identity",
            "buffer-layout",
        }
        if any(guard.kind not in supported for guard in self.variant_guards):
            raise ValueError(
                "region shim supports only exact-type, integer-domain, direct-field, "
                "callable identity, and buffer-layout variant guards"
            )
        if any(
            isinstance(guard.payload, CallableCodeIdentityGuardPayload)
            and guard.payload.callable_module != self.source_module
            for guard in self.variant_guards
        ):
            raise ValueError("callable identity guard belongs to another source module")

    def _validate_dispatch_identity(self) -> None:
        """Validate optional variant identity and deterministic dispatch priority.

        Raises:
            ValueError: If a provided variant ID is empty or dispatch rank is negative.
        """
        if self.variant_id is not None and not self.variant_id.strip():
            raise ValueError("region shim variant ID must be non-empty when provided")
        if self.dispatch_rank is not None and self.dispatch_rank < 0:
            raise ValueError("region shim dispatch rank must be non-negative")


@dataclass(frozen=True, slots=True)
class RegionShimEdit:
    """Original and updated staged source plus a reviewable unified diff.

    Attributes:
        old_text: Source text before applying the managed edit.
        new_text: Source text after applying the managed edit.
        diff: Unified diff between original and transformed source.
    """

    old_text: str
    new_text: str
    diff: str


def insert_or_replace_region_shim(
    source_text: str,
    configs: tuple[RegionShimConfig, ...],
) -> RegionShimEdit:
    """Append or replace one module-level typed-region runtime block.

    Args:
        source_text: Original Python source text to transform.
        configs: Region shim configurations to render in deterministic order.

    Returns:
        RegionShimEdit: Original text, transformed text, and unified diff for the region shim edit.
    """
    source_module = _validate_configs(configs)
    new_text = _replace_block(source_text, source_module, render_region_shim(configs))
    return _edit(source_text, new_text, configs[0].source_path.name)


def remove_region_shim(
    source_text: str,
    *,
    source_module: str,
    filename: str,
) -> RegionShimEdit:
    """Remove a typed-region block while rejecting ambiguous markers.

    Args:
        source_text: Original Python source text to transform.
        source_module: Importable source module name.
        filename: Filename used in unified diff headers.

    Returns:
        RegionShimEdit: Original text, transformed text, and unified diff after shim removal.
    """
    new_text = _replace_block(source_text, source_module, "")
    return _edit(source_text, new_text, filename)


def render_region_shim(configs: tuple[RegionShimConfig, ...]) -> str:
    """Render a staged-wheel loader for guarded functions and descriptors.

    Args:
        configs: Region shim configurations to render in deterministic order.

    Returns:
        str: Deterministic managed region-shim block.
    """
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
            "    if __builtins__.__class__.__name__ == 'dict':",
            "        _atoll_preexisting_names = __builtins__['frozenset'](",
            "            __builtins__['globals']()",
            "        )",
            "    else:",
            "        _atoll_preexisting_names = __builtins__.frozenset(",
            "            __builtins__.globals()",
            "        )",
            "    import ast as _atoll_ast",
            "    import builtins as _atoll_builtins",
            "    import functools as _atoll_functools",
            "    import hashlib as _atoll_hashlib",
            "    import importlib.machinery as _atoll_machinery",
            "    import importlib.util as _atoll_util",
            "    import inspect as _atoll_inspect",
            "    import os as _atoll_os",
            "    import pathlib as _atoll_pathlib",
            "    import sys as _atoll_sys",
            "",
            "    def _atoll_resolve_type(_atoll_path):",
            "        _atoll_parts = _atoll_path.split('.')",
            "        _atoll_value = _atoll_builtins.globals().get(_atoll_parts[0])",
            "        if _atoll_value is None:",
            "            _atoll_value = getattr(_atoll_builtins, _atoll_parts[0])",
            "        for _atoll_part in _atoll_parts[1:]:",
            "            _atoll_value = getattr(_atoll_value, _atoll_part)",
            "        if not isinstance(_atoll_value, type):",
            "            raise TypeError(f'Atoll guard is not a nominal type: {_atoll_path}')",
            "        return _atoll_value",
            "",
            "    def _atoll_source_fingerprint(_atoll_qualname):",
            "        _atoll_source = _atoll_pathlib.Path(__file__).read_text(encoding='utf-8')",
            "        _atoll_tree = _atoll_ast.parse(_atoll_source)",
            "        _atoll_body = _atoll_tree.body",
            "        _atoll_node = None",
            "        _atoll_parts = _atoll_qualname.split('.')",
            "        for _atoll_index, _atoll_part in enumerate(_atoll_parts):",
            "            _atoll_node = next(",
            (
                "                (item for item in _atoll_body "
                "if getattr(item, 'name', None) == _atoll_part), None"
            ),
            "            )",
            "            if _atoll_node is None:",
            "                raise AttributeError(_atoll_qualname)",
            "            if _atoll_index < len(_atoll_parts) - 1:",
            "                if not isinstance(_atoll_node, _atoll_ast.ClassDef):",
            "                    raise TypeError('Atoll callable owner is not a class')",
            "                _atoll_body = _atoll_node.body",
            "        if not isinstance(",
            "            _atoll_node, (_atoll_ast.FunctionDef, _atoll_ast.AsyncFunctionDef)",
            "        ):",
            "            raise TypeError('Atoll callable source declaration is not a function')",
            "        _atoll_start = min(",
            "            (_atoll_item.lineno for _atoll_item in _atoll_node.decorator_list),",
            "            default=_atoll_node.lineno,",
            "        )",
            "        _atoll_lines = _atoll_source.splitlines(keepends=True)",
            "        _atoll_declaration = ''.join(",
            "            _atoll_lines[_atoll_start - 1 : _atoll_node.end_lineno]",
            "        ).removesuffix('\\n')",
            "        return _atoll_hashlib.sha256(_atoll_declaration.encode('utf-8')).hexdigest()",
            "",
            "    def _atoll_callable_slot(_atoll_qualname):",
            "        _atoll_parts = _atoll_qualname.split('.')",
            "        if len(_atoll_parts) == 1:",
            "            return _atoll_builtins.globals(), _atoll_parts[0]",
            "        _atoll_owner = _atoll_builtins.globals().get(_atoll_parts[0])",
            "        if _atoll_owner is None:",
            "            raise AttributeError(_atoll_qualname)",
            "        for _atoll_part in _atoll_parts[1:-1]:",
            "            if isinstance(_atoll_owner, type):",
            "                _atoll_owner = vars(_atoll_owner).get(_atoll_part)",
            "            else:",
            "                _atoll_owner = getattr(_atoll_owner, _atoll_part)",
            "        return _atoll_owner, _atoll_parts[-1]",
            "",
            "    def _atoll_read_callable_slot(_atoll_owner, _atoll_name):",
            "        if isinstance(_atoll_owner, dict):",
            "            _atoll_value = _atoll_owner.get(_atoll_name)",
            "        elif isinstance(_atoll_owner, type):",
            "            _atoll_value = vars(_atoll_owner).get(_atoll_name)",
            "        else:",
            "            _atoll_value = getattr(_atoll_owner, _atoll_name)",
            "        if isinstance(_atoll_value, (staticmethod, classmethod)):",
            "            _atoll_value = _atoll_value.__func__",
            "        return _atoll_value",
            "",
            "    def _atoll_identity_callable(_atoll_value):",
            "        _atoll_fallback = getattr(",
            "            _atoll_value, '__atoll_python_fallback__', None",
            "        )",
            "        _atoll_variants = getattr(",
            "            _atoll_value, '__atoll_binding_variants__', ()",
            "        )",
            "        if callable(_atoll_fallback) and _atoll_variants:",
            "            return _atoll_fallback",
            "        return _atoll_value",
            "",
            "    def _atoll_resolve_guards(_atoll_guards):",
            "        _atoll_resolved = []",
            "        for _atoll_guard in _atoll_guards:",
            "            _atoll_item = {",
            "                **_atoll_guard,",
            "                'types': tuple(",
            "                    _atoll_resolve_type(_atoll_path)",
            "                    for _atoll_path in _atoll_guard.get('nominal_type_paths', ())",
            "                ),",
            "            }",
            "            if _atoll_guard.get('kind') == 'callable-code-identity':",
            "                _atoll_owner, _atoll_name = _atoll_callable_slot(",
            "                    _atoll_guard['callable_qualname']",
            "                )",
            "                _atoll_expected = _atoll_read_callable_slot(",
            "                    _atoll_owner, _atoll_name",
            "                )",
            "                if not callable(_atoll_expected):",
            "                    raise TypeError('Atoll identity guard target is not callable')",
            "                _atoll_expected = _atoll_identity_callable(_atoll_expected)",
            "                _atoll_source_digest = _atoll_source_fingerprint(",
            "                    _atoll_guard['callable_qualname']",
            "                )",
            "                if _atoll_source_digest != _atoll_guard['code_fingerprint']:",
            "                    raise TypeError('Atoll callable source fingerprint changed')",
            "                _atoll_expected_line = _atoll_guard.get('code_firstlineno')",
            "                if _atoll_expected_line is not None and getattr(",
            "                    _atoll_expected, '__code__', None",
            "                ).co_firstlineno != _atoll_expected_line:",
            "                    raise TypeError('Atoll callable code location changed')",
            "                _atoll_item['slot_owner'] = _atoll_owner",
            "                _atoll_item['slot_name'] = _atoll_name",
            "                _atoll_item['expected_callable'] = _atoll_expected",
            (
                "                _atoll_item['expected_code'] = "
                "getattr(_atoll_expected, '__code__', None)"
            ),
            "                if _atoll_item['expected_code'] is None:",
            (
                "                    raise TypeError("
                "'Atoll callable identity guard requires Python code')"
            ),
            "            _atoll_resolved.append(_atoll_item)",
            "        return tuple(_atoll_resolved)",
            "",
            "    def _atoll_guards_pass(",
            "        _atoll_guards,",
            "        _atoll_values,",
            "        _atoll_read_callable_value=_atoll_read_callable_slot,",
            "        _atoll_canonical_callable=_atoll_identity_callable,",
            "    ):",
            "        for _atoll_guard in _atoll_guards:",
            "            _atoll_kind = _atoll_guard.get('kind', 'runtime-type')",
            "            if _atoll_kind == 'callable-code-identity':",
            "                try:",
            "                    _atoll_live = _atoll_read_callable_value(",
            "                        _atoll_guard['slot_owner'], _atoll_guard['slot_name']",
            "                    )",
            "                except Exception:",
            "                    return False",
            "                _atoll_live = _atoll_canonical_callable(_atoll_live)",
            "                _atoll_receiver_name = _atoll_guard.get('receiver_subject')",
            "                if _atoll_receiver_name is not None:",
            "                    if _atoll_receiver_name not in _atoll_values:",
            "                        return False",
            "                    _atoll_instance = _atoll_values[_atoll_receiver_name]",
            "                    try:",
            "                        _atoll_instance_values = vars(_atoll_instance)",
            "                    except TypeError:",
            "                        _atoll_instance_values = {}",
            "                    if _atoll_guard['slot_name'] in _atoll_instance_values:",
            "                        return False",
            "                if (",
            "                    _atoll_live is _atoll_guard['expected_callable']",
            "                    and getattr(_atoll_live, '__code__', None)",
            "                    is _atoll_guard['expected_code']",
            "                ):",
            "                    continue",
            "                return False",
            "            _atoll_parameter = _atoll_guard['parameter_name']",
            "            if _atoll_parameter not in _atoll_values:",
            "                return False",
            "            _atoll_value = _atoll_values[_atoll_parameter]",
            "            if _atoll_kind == 'exact-type':",
            "                if len(_atoll_guard['types']) == 1 and type(_atoll_value) is (",
            "                    _atoll_guard['types'][0]",
            "                ):",
            "                    continue",
            "                return False",
            "            if _atoll_kind == 'integer-domain':",
            "                if type(_atoll_value) is int and (",
            "                    _atoll_guard['minimum'] <= _atoll_value",
            "                    <= _atoll_guard['maximum']",
            "                ):",
            "                    continue",
            "                return False",
            "            if _atoll_kind == 'direct-field':",
            "                if len(_atoll_guard['types']) != 1 or type(_atoll_value) is not (",
            "                    _atoll_guard['types'][0]",
            "                ):",
            "                    return False",
            "                try:",
            "                    _atoll_field = getattr(_atoll_value, _atoll_guard['field_name'])",
            "                except Exception:",
            "                    return False",
            "                if type(_atoll_field) is not int:",
            "                    return False",
            "                if not (",
            "                    _atoll_guard['minimum'] <= _atoll_field",
            "                    <= _atoll_guard['maximum']",
            "                ):",
            "                    return False",
            "                continue",
            "            if _atoll_kind == 'buffer-layout':",
            "                try:",
            "                    _atoll_view = memoryview(_atoll_value)",
            "                except (TypeError, ValueError):",
            "                    return False",
            "                try:",
            "                    _atoll_layout_matches = (",
            "                        _atoll_view.format == _atoll_guard['format']",
            "                        and _atoll_view.itemsize == _atoll_guard['itemsize']",
            "                        and _atoll_view.ndim == _atoll_guard['ndim']",
            "                        and _atoll_view.c_contiguous",
            "                        == _atoll_guard['c_contiguous']",
            "                        and _atoll_view.f_contiguous",
            "                        == _atoll_guard['f_contiguous']",
            "                        and (",
            "                            _atoll_guard['readonly'] is None",
            "                            or _atoll_view.readonly == _atoll_guard['readonly']",
            "                        )",
            "                        and (",
            "                            _atoll_guard['minimum_length'] is None",
            "                            or _atoll_guard['minimum_length'] <= len(_atoll_view)",
            "                            <= _atoll_guard['maximum_length']",
            "                        )",
            "                    )",
            "                finally:",
            "                    _atoll_view.release()",
            "                if _atoll_layout_matches:",
            "                    continue",
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
            "    def _atoll_select_variant(",
            "        _atoll_candidates,",
            "        _atoll_values,",
            "        _atoll_fallback,",
            "        _atoll_guard_check=_atoll_guards_pass,",
            "    ):",
            "        for _atoll_candidate in _atoll_candidates:",
            "            if _atoll_guard_check(_atoll_candidate['guards'], _atoll_values):",
            "                return _atoll_candidate['target']",
            "        return _atoll_fallback",
            "",
            "    def _atoll_execution_kind(_atoll_callable):",
            "        if isinstance(_atoll_callable, type):",
            "            return 'class'",
            "        if _atoll_inspect.isasyncgenfunction(_atoll_callable):",
            "            return 'async_generator'",
            "        if _atoll_inspect.iscoroutinefunction(_atoll_callable):",
            "            return 'coroutine'",
            "        if _atoll_inspect.isgeneratorfunction(_atoll_callable):",
            "            return 'generator'",
            "        if callable(_atoll_callable):",
            "            return 'sync'",
            "        return 'unknown'",
            "",
            "    def _atoll_verify_execution_kind(_atoll_callable, _atoll_expected):",
            "        _atoll_actual = _atoll_execution_kind(_atoll_callable)",
            "        if _atoll_actual != _atoll_expected:",
            "            raise TypeError(",
            "                f'Atoll binding expected {_atoll_expected} execution, '",
            "                f'got {_atoll_actual}'",
            "            )",
            "",
            "    def _atoll_verify_compiled_execution_kind(",
            "        _atoll_module, _atoll_name, _atoll_callable, _atoll_expected",
            "    ):",
            "        _atoll_actual = _atoll_execution_kind(_atoll_callable)",
            "        if _atoll_actual == _atoll_expected:",
            "            return",
            "        _atoll_declared = getattr(",
            "            _atoll_module, '__atoll_execution_kinds__', None",
            "        )",
            "        if isinstance(_atoll_declared, dict) and (",
            "            _atoll_declared.get(_atoll_name) == _atoll_expected",
            "        ):",
            "            return",
            "        raise TypeError(",
            "            f'Atoll binding expected {_atoll_expected} execution, '",
            "            f'got {_atoll_actual}'",
            "        )",
            "",
            "    def _atoll_dispatch_shape(_atoll_signature):",
            "        _atoll_positional_only = []",
            "        _atoll_positional_or_keyword = []",
            "        _atoll_var_positional = None",
            "        _atoll_keyword_only = []",
            "        _atoll_var_keyword = None",
            "        _atoll_call_arguments = []",
            "        _atoll_value_names = []",
            "        _atoll_defaults = {}",
            "        for _atoll_parameter in _atoll_signature.parameters.values():",
            "            _atoll_name = _atoll_parameter.name",
            "            _atoll_declaration = _atoll_name",
            "            if _atoll_parameter.default is not _atoll_parameter.empty:",
            "                _atoll_default_name = f'_atoll_default_{len(_atoll_defaults)}'",
            "                _atoll_defaults[_atoll_default_name] = _atoll_parameter.default",
            "                _atoll_declaration += f'={_atoll_default_name}'",
            "            _atoll_value_names.append(_atoll_name)",
            "            if _atoll_parameter.kind is _atoll_parameter.POSITIONAL_ONLY:",
            "                _atoll_positional_only.append(_atoll_declaration)",
            "                _atoll_call_arguments.append(_atoll_name)",
            "            elif _atoll_parameter.kind is _atoll_parameter.POSITIONAL_OR_KEYWORD:",
            "                _atoll_positional_or_keyword.append(_atoll_declaration)",
            "                _atoll_call_arguments.append(_atoll_name)",
            "            elif _atoll_parameter.kind is _atoll_parameter.VAR_POSITIONAL:",
            "                _atoll_var_positional = f'*{_atoll_name}'",
            "                _atoll_call_arguments.append(f'*{_atoll_name}')",
            "            elif _atoll_parameter.kind is _atoll_parameter.KEYWORD_ONLY:",
            "                _atoll_keyword_only.append(_atoll_declaration)",
            "                _atoll_call_arguments.append(f'{_atoll_name}={_atoll_name}')",
            "            elif _atoll_parameter.kind is _atoll_parameter.VAR_KEYWORD:",
            "                _atoll_var_keyword = f'**{_atoll_name}'",
            "                _atoll_call_arguments.append(f'**{_atoll_name}')",
            "        _atoll_declarations = list(_atoll_positional_only)",
            "        if _atoll_positional_only:",
            "            _atoll_declarations.append('/')",
            "        _atoll_declarations.extend(_atoll_positional_or_keyword)",
            "        if _atoll_var_positional is not None:",
            "            _atoll_declarations.append(_atoll_var_positional)",
            "        elif _atoll_keyword_only:",
            "            _atoll_declarations.append('*')",
            "        _atoll_declarations.extend(_atoll_keyword_only)",
            "        if _atoll_var_keyword is not None:",
            "            _atoll_declarations.append(_atoll_var_keyword)",
            "        return (",
            "            ', '.join(_atoll_declarations),",
            "            ', '.join(_atoll_call_arguments),",
            "            tuple(_atoll_value_names),",
            "            _atoll_defaults,",
            "        )",
            "",
            "    def _atoll_dispatch_source(",
            "        _atoll_kind, _atoll_declaration, _atoll_call_arguments, _atoll_value_names",
            "    ):",
            "        _atoll_values = ', '.join(",
            "            f'{_atoll_name!r}: {_atoll_name}' for _atoll_name in _atoll_value_names",
            "        )",
            "        _atoll_value_line = f'    _atoll_values = {{{_atoll_values}}}'",
            "        _atoll_select_line = (",
            "            '    _atoll_callable = _atoll_select_variant('",
            "            '_atoll_candidates, _atoll_values, _atoll_fallback)'",
            "        )",
            "        _atoll_call = f'_atoll_callable({_atoll_call_arguments})'",
            "        if _atoll_kind == 'async_generator':",
            "            _atoll_body = [",
            "                _atoll_value_line,",
            "                _atoll_select_line,",
            "                f'    _atoll_generator = {_atoll_call}',",
            "                '    try:',",
            "                '        _atoll_value = await _atoll_generator.__anext__()',",
            "                '        while True:',",
            "                '            try:',",
            "                '                _atoll_sent = yield _atoll_value',",
            "                '            except GeneratorExit:',",
            "                '                await _atoll_generator.aclose()',",
            "                '                raise',",
            "                '            except BaseException as _atoll_thrown:',",
            "                '                _atoll_value = await _atoll_generator.athrow('",
            "                '                    _atoll_thrown',",
            "                '                )',",
            "                '            else:',",
            "                '                if _atoll_sent is None:',",
            (
                "                '                    _atoll_value = await "
                "_atoll_generator.__anext__()',"
            ),
            "                '                else:',",
            "                '                    _atoll_value = await _atoll_generator.asend('",
            "                '                        _atoll_sent',",
            "                '                    )',",
            "                '    except StopAsyncIteration:',",
            "                '        return',",
            "            ]",
            "            _atoll_prefix = 'async def'",
            "        elif _atoll_kind == 'coroutine':",
            "            _atoll_body = [",
            "                _atoll_value_line,",
            "                _atoll_select_line,",
            "                f'    return await {_atoll_call}',",
            "            ]",
            "            _atoll_prefix = 'async def'",
            "        elif _atoll_kind == 'generator':",
            "            _atoll_body = [",
            "                _atoll_value_line,",
            "                _atoll_select_line,",
            "                f'    return (yield from {_atoll_call})',",
            "            ]",
            "            _atoll_prefix = 'def'",
            "        else:",
            "            _atoll_body = [",
            "                _atoll_value_line,",
            "                _atoll_select_line,",
            "                f'    return {_atoll_call}',",
            "            ]",
            "            _atoll_prefix = 'def'",
            "        return '\\n'.join(",
            (
                "            [f'{_atoll_prefix} "
                "_atoll_generated({_atoll_declaration}):', *_atoll_body]"
            ),
            "        )",
            "",
            "    def _atoll_bind(",
            "        _atoll_source,",
            "        _atoll_target,",
            "        _atoll_guards,",
            "        _atoll_kind,",
            "        _atoll_variant_id,",
            "        _atoll_dispatch_rank,",
            "    ):",
            "        _atoll_fallback = getattr(",
            "            _atoll_source, '__atoll_python_fallback__', _atoll_source",
            "        )",
            "        _atoll_verify_execution_kind(_atoll_fallback, _atoll_kind)",
            "        _atoll_candidates_by_id = {",
            "            _atoll_candidate['variant_id']: _atoll_candidate",
            "            for _atoll_candidate in getattr(",
            "                _atoll_source, '__atoll_binding_variants__', ()",
            "            )",
            "        }",
            "        _atoll_candidates_by_id[_atoll_variant_id] = {",
            "            'variant_id': _atoll_variant_id,",
            "            'dispatch_rank': _atoll_dispatch_rank,",
            "            'target': _atoll_target,",
            "            'guards': _atoll_guards,",
            "        }",
            "        _atoll_candidates = tuple(",
            "            sorted(",
            "                _atoll_candidates_by_id.values(),",
            "                key=lambda item: (item['dispatch_rank'], item['variant_id']),",
            "            )",
            "        )",
            "        _atoll_signature = _atoll_inspect.signature(_atoll_fallback)",
            "        (",
            "            _atoll_declaration,",
            "            _atoll_call_arguments,",
            "            _atoll_value_names,",
            "            _atoll_defaults,",
            "        ) = _atoll_dispatch_shape(_atoll_signature)",
            "        _atoll_namespace = {",
            "            '_atoll_candidates': _atoll_candidates,",
            "            '_atoll_fallback': _atoll_fallback,",
            "            '_atoll_select_variant': _atoll_select_variant,",
            "            **_atoll_defaults,",
            "        }",
            "        exec(",
            "            _atoll_dispatch_source(",
            "                _atoll_kind,",
            "                _atoll_declaration,",
            "                _atoll_call_arguments,",
            "                _atoll_value_names,",
            "            ),",
            "            _atoll_namespace,",
            "        )",
            "        _atoll_wrapped = _atoll_namespace['_atoll_generated']",
            "        _atoll_functools.update_wrapper(_atoll_wrapped, _atoll_fallback)",
            "        _atoll_wrapped.__signature__ = _atoll_signature",
            "        try:",
            "            _atoll_wrapped.__defaults__ = _atoll_fallback.__defaults__",
            "            _atoll_wrapped.__kwdefaults__ = _atoll_fallback.__kwdefaults__",
            "        except AttributeError:",
            "            pass",
            "        _atoll_wrapped.__atoll_compiled_target__ = _atoll_candidates[0]['target']",
            "        _atoll_wrapped.__atoll_compiled_targets__ = tuple(",
            "            item['target'] for item in _atoll_candidates",
            "        )",
            "        _atoll_wrapped.__atoll_python_fallback__ = _atoll_fallback",
            "        _atoll_wrapped.__atoll_runtime_guards__ = _atoll_candidates[0]['guards']",
            "        _atoll_wrapped.__atoll_variant_guards__ = tuple(",
            "            item['guards'] for item in _atoll_candidates",
            "        )",
            "        _atoll_wrapped.__atoll_binding_variants__ = _atoll_candidates",
            "        return _atoll_wrapped",
            "",
            "    def _atoll_direct_shape(_atoll_callable):",
            "        _atoll_signature = _atoll_inspect.signature(_atoll_callable)",
            "        return tuple(",
            "            (_atoll_parameter.name, _atoll_parameter.kind)",
            "            for _atoll_parameter in _atoll_signature.parameters.values()",
            "        )",
            "",
            "    def _atoll_restore_direct_metadata(_atoll_snapshot):",
            "        _atoll_target, _atoll_namespace, _atoll_attributes = _atoll_snapshot",
            "        _atoll_target_namespace = getattr(_atoll_target, '__dict__', None)",
            "        if not isinstance(_atoll_target_namespace, dict):",
            "            raise TypeError('Atoll direct target metadata became immutable')",
            "        _atoll_target_namespace.clear()",
            "        _atoll_target_namespace.update(_atoll_namespace)",
            "        for _atoll_name, _atoll_exists, _atoll_value in _atoll_attributes:",
            "            if _atoll_exists:",
            "                setattr(_atoll_target, _atoll_name, _atoll_value)",
            "            else:",
            "                try:",
            "                    delattr(_atoll_target, _atoll_name)",
            "                except AttributeError:",
            "                    pass",
            "",
            "    def _atoll_prepare_direct_binding(",
            "        _atoll_source,",
            "        _atoll_target,",
            "        _atoll_guards,",
            "        _atoll_kind,",
            "        _atoll_binding_kind,",
            "        _atoll_owner,",
            "        _atoll_variant_id,",
            "        _atoll_dispatch_rank,",
            "    ):",
            '        """Return a metadata-safe direct target and its rollback snapshot."""',
            "        if _atoll_guards or getattr(",
            "            _atoll_source, '__atoll_binding_variants__', ()",
            "        ):",
            "            return None",
            "        if _atoll_execution_kind(_atoll_target) != _atoll_kind:",
            "            return None",
            "        try:",
            "            if _atoll_binding_kind == 'instance_method':",
            "                _atoll_source_get = getattr(type(_atoll_source), '__get__', None)",
            "                _atoll_target_get = getattr(type(_atoll_target), '__get__', None)",
            "                if (",
            "                    _atoll_owner is None",
            "                    or _atoll_source_get is None",
            "                    or _atoll_target_get is None",
            "                ):",
            "                    return None",
            "                _atoll_binding_probe = object()",
            "                _atoll_source_bound = _atoll_source_get(",
            "                    _atoll_source, _atoll_binding_probe, _atoll_owner",
            "                )",
            "                _atoll_target_bound = _atoll_target_get(",
            "                    _atoll_target, _atoll_binding_probe, _atoll_owner",
            "                )",
            "                if _atoll_direct_shape(_atoll_target_bound) != (",
            "                    _atoll_direct_shape(_atoll_source_bound)",
            "                ):",
            "                    return None",
            "            _atoll_signature = _atoll_inspect.signature(_atoll_source)",
            "            _atoll_target_signature = _atoll_inspect.signature(_atoll_target)",
            "            if any(",
            "                _atoll_parameter.default is not _atoll_inspect.Parameter.empty",
            "                for _atoll_parameter in (",
            "                    *_atoll_signature.parameters.values(),",
            "                    *_atoll_target_signature.parameters.values(),",
            "                )",
            "            ):",
            "                return None",
            "            if _atoll_direct_shape(_atoll_target) != _atoll_direct_shape(",
            "                _atoll_source",
            "            ):",
            "                return None",
            "            _atoll_target_namespace = getattr(_atoll_target, '__dict__', None)",
            "            if not isinstance(_atoll_target_namespace, dict):",
            "                return None",
            "            _atoll_metadata_names = (",
            "                '__module__',",
            "                '__name__',",
            "                '__qualname__',",
            "                '__doc__',",
            "                '__annotations__',",
            "                '__type_params__',",
            "                '__signature__',",
            "            )",
            "            _atoll_snapshot = (",
            "                _atoll_target,",
            "                dict(_atoll_target_namespace),",
            "                tuple(",
            "                    (",
            "                        _atoll_name,",
            "                        hasattr(_atoll_target, _atoll_name),",
            "                        getattr(_atoll_target, _atoll_name, None),",
            "                    )",
            "                    for _atoll_name in _atoll_metadata_names",
            "                ),",
            "            )",
            "            _atoll_candidate = {",
            "                'variant_id': _atoll_variant_id,",
            "                'dispatch_rank': _atoll_dispatch_rank,",
            "                'target': _atoll_target,",
            "                'guards': (),",
            "            }",
            "            try:",
            "                _atoll_functools.update_wrapper(_atoll_target, _atoll_source)",
            "                _atoll_target.__signature__ = _atoll_signature",
            "                _atoll_target.__atoll_compiled_target__ = _atoll_target",
            "                _atoll_target.__atoll_compiled_targets__ = (_atoll_target,)",
            "                _atoll_target.__atoll_python_fallback__ = _atoll_source",
            "                _atoll_target.__atoll_runtime_guards__ = ()",
            "                _atoll_target.__atoll_variant_guards__ = ((),)",
            "                _atoll_target.__atoll_binding_variants__ = (_atoll_candidate,)",
            "                if _atoll_inspect.signature(_atoll_target) != _atoll_signature:",
            "                    raise TypeError('Atoll direct target changed source signature')",
            "                if _atoll_target.__annotations__ is not (",
            "                    _atoll_source.__annotations__",
            "                ):",
            "                    raise TypeError('Atoll direct target changed source annotations')",
            "            except Exception:",
            "                _atoll_restore_direct_metadata(_atoll_snapshot)",
            "                return None",
            "            return _atoll_target, _atoll_snapshot",
            "        except (AttributeError, TypeError, ValueError):",
            "            return None",
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
            "    def _atoll_install_binding(_atoll_plan_item, _atoll_applied):",
            "        _atoll_binding = _atoll_plan_item['binding']",
            "        _atoll_name = _atoll_plan_item['name']",
            "        _atoll_value = _atoll_plan_item['value']",
            "        _atoll_owner = _atoll_plan_item['owner']",
            '        if _atoll_binding["kind"] in {"module", "class"}:',
            "            _atoll_exists = _atoll_name in _atoll_builtins.globals()",
            "            _atoll_previous = _atoll_builtins.globals().get(_atoll_name)",
            "            _atoll_applied.append(",
            "                (None, _atoll_name, _atoll_exists, _atoll_previous)",
            "            )",
            "            _atoll_builtins.globals()[_atoll_name] = _atoll_value",
            "            return",
            "        _atoll_namespace = vars(_atoll_owner)",
            "        _atoll_exists = _atoll_name in _atoll_namespace",
            "        _atoll_previous = _atoll_namespace.get(_atoll_name)",
            "        _atoll_applied.append(",
            "            (_atoll_owner, _atoll_name, _atoll_exists, _atoll_previous)",
            "        )",
            '        if _atoll_binding["kind"] == "staticmethod":',
            "            setattr(_atoll_owner, _atoll_name, staticmethod(_atoll_value))",
            '        elif _atoll_binding["kind"] == "classmethod":',
            "            setattr(_atoll_owner, _atoll_name, classmethod(_atoll_value))",
            "        else:",
            "            setattr(_atoll_owner, _atoll_name, _atoll_value)",
            "",
            "    def _atoll_rollback(_atoll_applied):",
            "        _atoll_rollback_errors = []",
            "        for _atoll_owner, _atoll_name, _atoll_exists, _atoll_previous in reversed(",
            "            _atoll_applied",
            "        ):",
            "            try:",
            "                if _atoll_owner is None:",
            "                    if _atoll_exists:",
            "                        _atoll_builtins.globals()[_atoll_name] = _atoll_previous",
            "                    else:",
            "                        _atoll_builtins.globals().pop(_atoll_name, None)",
            "                elif _atoll_exists:",
            "                    setattr(_atoll_owner, _atoll_name, _atoll_previous)",
            "                else:",
            "                    try:",
            "                        delattr(_atoll_owner, _atoll_name)",
            "                    except AttributeError:",
            "                        pass",
            "            except Exception as _atoll_rollback_error:",
            "                _atoll_rollback_errors.append(_atoll_rollback_error)",
            "        return tuple(_atoll_rollback_errors)",
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
            "    _atoll_pending_plan = []",
            '    _atoll_allowlist_text = _atoll_os.getenv("ATOLL_REGION_ALLOWLIST")',
            "    _atoll_region_allowlist = (",
            "        None",
            "        if _atoll_allowlist_text is None",
            "        else frozenset(_atoll_allowlist_text.splitlines())",
            "    )",
            '    _atoll_variant_allowlist_text = _atoll_os.getenv("ATOLL_VARIANT_ALLOWLIST")',
            "    _atoll_variant_allowlist = (",
            "        None",
            "        if _atoll_variant_allowlist_text is None",
            "        else frozenset(_atoll_variant_allowlist_text.splitlines())",
            "    )",
            "",
            '    if _atoll_os.getenv("ATOLL_DISABLE") != "1":',
            "        for _atoll_region in _atoll_regions:",
            "            _atoll_region_allowlisted = (",
            "                _atoll_region_allowlist is None",
            '                or _atoll_region["region_id"] in _atoll_region_allowlist',
            "            )",
            "            _atoll_variant_selected = (",
            "                _atoll_variant_allowlist is None",
            '                or _atoll_region["variant_id"] in _atoll_variant_allowlist',
            "            )",
            "            _atoll_region_selected = (",
            "                _atoll_region_allowlisted and _atoll_variant_selected",
            "            )",
            "            _atoll_region_status = {",
            '                "backend": _atoll_region["backend"],',
            '                "compiled_module": _atoll_region["compiled_module"],',
            '                "variant_id": _atoll_region["variant_id"],',
            '                "region_allowlisted": _atoll_region_allowlisted,',
            '                "variant_selected": _atoll_variant_selected,',
            '                "selected": _atoll_region_selected,',
            '                "active": False,',
            '                "compiled": False,',
            '                "origin": None,',
            '                "error": None,',
            '                "bindings": {},',
            "            }",
            '            __atoll_region_status__[_atoll_region["variant_id"]] = (',
            "                _atoll_region_status",
            "            )",
            "            if not _atoll_region_selected:",
            "                continue",
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
            "                _atoll_plan = []",
            "                _atoll_required_binding_errors = []",
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
            "                            _atoll_source = _atoll_builtins.globals()[_atoll_name]",
            "                            _atoll_target_owner = None",
            "                        else:",
            "                            _atoll_source_owner = _atoll_builtins.globals()[",
            '                                _atoll_binding["source_owner_class"]',
            "                            ]",
            "                            _atoll_target_owner = _atoll_builtins.globals()[",
            '                                _atoll_binding["target_owner_class"]',
            "                            ]",
            "                            _atoll_descriptor = vars(_atoll_source_owner)[",
            "                                _atoll_name",
            "                            ]",
            '                            if _atoll_binding["kind"] == "staticmethod" and not (',
            "                                isinstance(_atoll_descriptor, staticmethod)",
            "                            ):",
            "                                raise TypeError(",
            "                                    'Atoll staticmethod binding requires a '",
            "                                    'staticmethod source descriptor'",
            "                                )",
            '                            if _atoll_binding["kind"] == "classmethod" and not (',
            "                                isinstance(_atoll_descriptor, classmethod)",
            "                            ):",
            "                                raise TypeError(",
            "                                    'Atoll classmethod binding requires a '",
            "                                    'classmethod source descriptor'",
            "                                )",
            '                            if _atoll_binding["kind"] == "instance_method" and (',
            "                                isinstance(",
            "                                    _atoll_descriptor, (staticmethod, classmethod)",
            "                                )",
            "                            ):",
            "                                raise TypeError(",
            "                                    'Atoll instance-method binding requires a '",
            "                                    'plain source descriptor'",
            "                                )",
            '                            if _atoll_binding["kind"] in {',
            '                                "staticmethod", "classmethod"',
            "                            }:",
            "                                _atoll_source = _atoll_descriptor.__func__",
            "                            else:",
            "                                _atoll_source = _atoll_descriptor",
            "                        _atoll_verify_execution_kind(",
            '                            _atoll_source, _atoll_binding["execution_kind"]',
            "                        )",
            '                        _atoll_shell = _atoll_region["outlined_shell"]',
            "                        if _atoll_shell is None:",
            "                            _atoll_target = getattr(",
            '                                _atoll_mod, _atoll_binding["compiled_name"]',
            "                            )",
            "                            _atoll_verify_compiled_execution_kind(",
            "                                _atoll_mod,",
            '                                _atoll_binding["compiled_name"],',
            "                                _atoll_target,",
            '                                _atoll_binding["execution_kind"],',
            "                            )",
            "                        else:",
            "                            _atoll_helpers = tuple(",
            "                                getattr(_atoll_mod, _atoll_helper_name)",
            (
                "                                for _atoll_helper_name in "
                '_atoll_shell["helper_names"]'
            ),
            "                            )",
            "                            for _atoll_helper_name, _atoll_helper in zip(",
            '                                _atoll_shell["helper_names"],',
            "                                _atoll_helpers,",
            "                                strict=True,",
            "                            ):",
            "                                _atoll_verify_compiled_execution_kind(",
            "                                    _atoll_mod,",
            "                                    _atoll_helper_name,",
            "                                    _atoll_helper,",
            "                                    'sync',",
            "                                )",
            "                            _atoll_shell_namespace = {}",
            "                            exec(",
            '                                _atoll_shell["factory_source"],',
            "                                _atoll_builtins.globals(),",
            "                                _atoll_shell_namespace,",
            "                            )",
            "                            _atoll_factory = _atoll_shell_namespace.get(",
            '                                _atoll_shell["factory_name"]',
            "                            )",
            "                            if not callable(_atoll_factory):",
            "                                raise TypeError(",
            "                                    'Atoll outlined shell factory is not callable'",
            "                                )",
            "                            _atoll_target = _atoll_factory(_atoll_mod)",
            "                            _atoll_verify_execution_kind(",
            "                                _atoll_target,",
            '                                _atoll_binding["execution_kind"],',
            "                            )",
            "                            _atoll_target.__atoll_native_helpers__ = (",
            "                                _atoll_helpers",
            "                            )",
            '                        if _atoll_binding["kind"] == "class":',
            "                            _atoll_guards = ()",
            "                        else:",
            "                            _atoll_guards = _atoll_resolve_guards(",
            '                                _atoll_binding["guards"]',
            "                            )",
            "                        _atoll_plan.append(",
            "                            {",
            "                                'binding': _atoll_binding,",
            "                                'status': _atoll_binding_status,",
            "                                'region_status': _atoll_region_status,",
            "                                'name': _atoll_name,",
            "                                'owner': _atoll_target_owner,",
            "                                'source': _atoll_source,",
            "                                'target': _atoll_target,",
            "                                'guards': _atoll_guards,",
            "                                'variant_id': _atoll_region['variant_id'],",
            "                                'dispatch_rank': _atoll_region['dispatch_rank'],",
            "                            }",
            "                        )",
            "                    except Exception as _atoll_binding_error:",
            '                        _atoll_binding_status["error"] = repr(_atoll_binding_error)',
            '                        if _atoll_binding["required"]:',
            "                            _atoll_required_binding_errors.append(",
            "                                _atoll_binding_error",
            "                            )",
            "                            _atoll_errors.append(_atoll_binding_error)",
            "                if not _atoll_required_binding_errors:",
            "                    _atoll_pending_plan.extend(_atoll_plan)",
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
            "        _atoll_dispatch_groups = {}",
            "        for _atoll_plan_item in _atoll_pending_plan:",
            "            _atoll_binding = _atoll_plan_item['binding']",
            "            _atoll_group_key = (",
            "                _atoll_binding['qualname'],",
            "                _atoll_binding['kind'],",
            "                _atoll_binding['execution_kind'],",
            "            )",
            "            _atoll_dispatch_groups.setdefault(_atoll_group_key, []).append(",
            "                _atoll_plan_item",
            "            )",
            "",
            "        _atoll_install_plan = []",
            "        _atoll_dispatch_errors = []",
            "        _atoll_direct_snapshots = []",
            "        for _atoll_group_key in sorted(_atoll_dispatch_groups):",
            "            _atoll_group = _atoll_dispatch_groups[_atoll_group_key]",
            "            _atoll_first = _atoll_group[0]",
            "            try:",
            "                if any(",
            "                    _atoll_item['source'] is not _atoll_first['source']",
            "                    or _atoll_item['owner'] is not _atoll_first['owner']",
            "                    for _atoll_item in _atoll_group[1:]",
            "                ):",
            "                    raise TypeError(",
            "                        'Atoll variants disagree on source binding identity'",
            "                    )",
            "                if _atoll_first['binding']['kind'] == 'class':",
            "                    if len(_atoll_group) != 1:",
            "                        raise TypeError(",
            "                            'Atoll class bindings do not support multiple variants'",
            "                        )",
            "                    _atoll_value = _atoll_prepare_class(",
            "                        _atoll_first['source'], _atoll_first['target']",
            "                    )",
            "                else:",
            "                    _atoll_direct = None",
            "                    if len(_atoll_group) == 1:",
            "                        _atoll_direct = _atoll_prepare_direct_binding(",
            "                            _atoll_first['source'],",
            "                            _atoll_first['target'],",
            "                            _atoll_first['guards'],",
            "                            _atoll_first['binding']['execution_kind'],",
            "                            _atoll_first['binding']['kind'],",
            "                            _atoll_first['owner'],",
            "                            _atoll_first['variant_id'],",
            "                            _atoll_first['dispatch_rank'],",
            "                        )",
            "                    if _atoll_direct is not None:",
            "                        _atoll_value, _atoll_direct_snapshot = _atoll_direct",
            "                        _atoll_direct_snapshots.append(_atoll_direct_snapshot)",
            "                    else:",
            "                        _atoll_value = _atoll_first['source']",
            "                        for _atoll_item in sorted(",
            "                            _atoll_group,",
            "                            key=lambda item: (",
            "                                item['dispatch_rank'], item['variant_id']",
            "                            ),",
            "                        ):",
            "                            _atoll_value = _atoll_bind(",
            "                                _atoll_value,",
            "                                _atoll_item['target'],",
            "                                _atoll_item['guards'],",
            "                                _atoll_item['binding']['execution_kind'],",
            "                                _atoll_item['variant_id'],",
            "                                _atoll_item['dispatch_rank'],",
            "                            )",
            "                _atoll_install_item = dict(_atoll_first)",
            "                _atoll_install_item['value'] = _atoll_value",
            "                _atoll_install_item['members'] = tuple(_atoll_group)",
            "                _atoll_install_plan.append(_atoll_install_item)",
            "            except Exception as _atoll_dispatch_error:",
            "                _atoll_dispatch_errors.append(_atoll_dispatch_error)",
            "                _atoll_errors.append(_atoll_dispatch_error)",
            "                for _atoll_item in _atoll_group:",
            "                    _atoll_item['status']['error'] = repr(_atoll_dispatch_error)",
            "",
            "        if _atoll_dispatch_errors:",
            "            for _atoll_direct_snapshot in reversed(_atoll_direct_snapshots):",
            "                try:",
            "                    _atoll_restore_direct_metadata(_atoll_direct_snapshot)",
            "                except Exception as _atoll_restore_error:",
            "                    _atoll_errors.append(_atoll_restore_error)",
            "            _atoll_dispatch_error = _atoll_dispatch_errors[0]",
            "            for _atoll_plan_item in _atoll_pending_plan:",
            "                if _atoll_plan_item['status']['error'] is None:",
            "                    _atoll_plan_item['status']['error'] = repr(",
            "                        _atoll_dispatch_error",
            "                    )",
            "        elif _atoll_install_plan:",
            "            _atoll_applied = []",
            "            try:",
            "                for _atoll_install_item in _atoll_install_plan:",
            "                    _atoll_install_binding(_atoll_install_item, _atoll_applied)",
            "                for _atoll_install_item in _atoll_install_plan:",
            "                    for _atoll_member in _atoll_install_item['members']:",
            "                        _atoll_member['status']['active'] = True",
            "                        _atoll_member['status']['compiled'] = True",
            "            except Exception as _atoll_apply_error:",
            "                _atoll_rollback_errors = _atoll_rollback(_atoll_applied)",
            "                for _atoll_direct_snapshot in reversed(_atoll_direct_snapshots):",
            "                    try:",
            "                        _atoll_restore_direct_metadata(_atoll_direct_snapshot)",
            "                    except Exception as _atoll_restore_error:",
            "                        _atoll_rollback_errors += (_atoll_restore_error,)",
            "                for _atoll_plan_item in _atoll_pending_plan:",
            "                    _atoll_plan_item['status']['active'] = False",
            "                    _atoll_plan_item['status']['compiled'] = False",
            "                    if _atoll_plan_item['status']['error'] is None:",
            "                        _atoll_plan_item['status']['error'] = repr(",
            "                            _atoll_apply_error",
            "                        )",
            "                if _atoll_rollback_errors:",
            "                    _atoll_rollback_error_text = tuple(",
            "                        repr(_atoll_error) for _atoll_error in _atoll_rollback_errors",
            "                    )",
            "                    for _atoll_region_status in __atoll_region_status__.values():",
            "                        if _atoll_region_status['selected']:",
            "                            _atoll_region_status['rollback_errors'] = (",
            "                                _atoll_rollback_error_text",
            "                            )",
            "                _atoll_errors.append(_atoll_apply_error)",
            "",
            "        for _atoll_region_status in __atoll_region_status__.values():",
            "            _atoll_binding_statuses = tuple(",
            "                _atoll_region_status['bindings'].values()",
            "            )",
            "            _atoll_region_status['active'] = any(",
            "                item['active'] for item in _atoll_binding_statuses",
            "            )",
            "            if _atoll_region_status['selected']:",
            "                _atoll_region_status['compiled'] = (",
            "                    _atoll_region_status['error'] is None",
            "                    and all(",
            "                        item['compiled']",
            "                        for item in _atoll_binding_statuses",
            "                        if item['required']",
            "                    )",
            "                )",
            "",
            '        __atoll_status__["active"] = any(',
            '            region["active"] for region in __atoll_region_status__.values()',
            "        )",
            '        __atoll_status__["compiled"] = bool(__atoll_region_status__) and all(',
            '            region["compiled"] for region in __atoll_region_status__.values()',
            '            if region["selected"]',
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
            "    for _atoll_name_to_remove in tuple(_atoll_builtins.globals()):",
            "        if (",
            '            _atoll_name_to_remove.startswith("_atoll_")',
            "            and _atoll_name_to_remove not in _atoll_preexisting_names",
            "            and _atoll_name_to_remove not in {",
            '                "_atoll_builtins", "_atoll_preexisting_names"',
            "            }",
            "        ):",
            "            _atoll_builtins.globals().pop(_atoll_name_to_remove, None)",
            '    _atoll_builtins.globals().pop("_atoll_name_to_remove", None)',
            '    _atoll_builtins.globals().pop("_atoll_preexisting_names", None)',
            '    _atoll_builtins.globals().pop("_atoll_builtins", None)',
            _end_marker(source_module),
            "",
        ]
    )


def _runtime_region(config: RegionShimConfig) -> dict[str, object]:
    dispatch_rank = config.dispatch_rank
    if dispatch_rank is None:
        dispatch_rank = 200 if config.backend == "mypyc" else 210
    return {
        "region_id": config.region_id,
        "variant_id": config.variant_id or config.region_id,
        "dispatch_rank": dispatch_rank,
        "backend": config.backend,
        "compiled_module": config.compiled_module,
        "artifact_relative": _relative_path_text(config.source_path.parent, config.artifact_dir),
        "outlined_shell": (
            {
                "factory_name": config.outlined_shell.factory_name,
                "factory_source": config.outlined_shell.factory_source,
                "helper_names": config.outlined_shell.helper_names,
            }
            if config.outlined_shell is not None
            else None
        ),
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
                        "kind": "runtime-type",
                        "parameter_name": guard.parameter_name,
                        "positional_index": guard.positional_index,
                        "annotation": guard.annotation,
                        "nominal_type_paths": guard.nominal_type_paths,
                        "allow_none": guard.allow_none,
                    }
                    for guard in binding.guards
                )
                + tuple(
                    _structured_guard(guard, source_module=config.source_module)
                    for guard in config.variant_guards
                ),
            }
            for binding in config.bindings
        ),
    }


def _structured_guard(
    guard: GuardExpression,
    *,
    source_module: str,
) -> dict[str, object]:
    """Serialize one safe scalar guard for staged runtime dispatch.

    Args:
        guard: Structured guard produced by scalar proof analysis.
        source_module: Module whose globals own same-module nominal guard types.

    Returns:
        dict[str, object]: Literal-only runtime mapping embedded in the staged shim.

    Raises:
        ValueError: If a future guard kind reaches this milestone's renderer.
    """
    if guard.kind == "exact-type" and isinstance(guard.payload, ExactTypeGuardPayload):
        path = _runtime_type_path(
            guard.payload.type_module,
            guard.payload.type_qualname,
            source_module,
        )
        return {
            "kind": "exact-type",
            "parameter_name": guard.payload.subject,
            "nominal_type_paths": (path,),
        }
    if guard.kind == "integer-domain" and isinstance(guard.payload, IntegerDomainGuardPayload):
        return {
            "kind": "integer-domain",
            "parameter_name": guard.payload.subject,
            "minimum": guard.payload.minimum,
            "maximum": guard.payload.maximum,
        }
    if guard.kind == "direct-field" and isinstance(guard.payload, DirectFieldGuardPayload):
        if guard.payload.minimum is None or guard.payload.maximum is None:
            raise ValueError("runtime direct-field integer guards require closed bounds")
        owner_path = _runtime_type_path(
            guard.payload.owner_type_module,
            guard.payload.owner_type_qualname,
            source_module,
        )
        return {
            "kind": "direct-field",
            "parameter_name": guard.payload.owner_subject,
            "nominal_type_paths": (owner_path,),
            "field_name": guard.payload.field_name,
            "minimum": guard.payload.minimum,
            "maximum": guard.payload.maximum,
        }
    if guard.kind == "callable-code-identity" and isinstance(
        guard.payload, CallableCodeIdentityGuardPayload
    ):
        return {
            "kind": "callable-code-identity",
            "callable_module": guard.payload.callable_module,
            "callable_qualname": guard.payload.callable_qualname,
            "code_fingerprint": guard.payload.code_fingerprint,
            "receiver_subject": guard.payload.receiver_subject,
            "code_firstlineno": guard.payload.code_firstlineno,
        }
    if guard.kind == "buffer-layout" and isinstance(guard.payload, BufferLayoutGuardPayload):
        return {
            "kind": "buffer-layout",
            "parameter_name": guard.payload.subject,
            "format": guard.payload.format,
            "itemsize": guard.payload.itemsize,
            "ndim": guard.payload.ndim,
            "c_contiguous": guard.payload.c_contiguous,
            "f_contiguous": guard.payload.f_contiguous,
            "readonly": guard.payload.readonly,
            "minimum_length": guard.payload.minimum_length,
            "maximum_length": guard.payload.maximum_length,
        }
    raise ValueError(f"unsupported structured region guard: {guard.kind}")


def _runtime_type_path(type_module: str, type_qualname: str, source_module: str) -> str:
    if type_module in {"builtins", source_module}:
        return type_qualname
    return f"{type_module}.{type_qualname}"


def _binding_runtime_qualname(binding: BindingTarget) -> str:
    """Return the binding key users see, including concrete subclass targets.

    Args:
        binding: Source binding being rendered or verified.

    Returns:
        str: Runtime qualified name used to resolve the binding.
    """
    member_name = binding.source.qualname.rsplit(".", maxsplit=1)[-1]
    if binding.target_owner_class is not None:
        return f"{binding.target_owner_class}.{member_name}"
    return binding.source.qualname


def _validate_configs(configs: tuple[RegionShimConfig, ...]) -> str:
    if not configs:
        raise ValueError("typed-region shim requires at least one region config")
    source_modules = {config.source_module for config in configs}
    source_paths = {config.source_path.resolve() for config in configs}
    variant_ids = [config.variant_id or config.region_id for config in configs]
    if len(source_modules) != 1 or len(source_paths) != 1:
        raise ValueError("typed-region shim configs must target one source module")
    if len(variant_ids) != len(set(variant_ids)):
        raise ValueError("typed-region shim configs must use unique variant IDs")
    _validate_binding_dispatches(configs)
    return configs[0].source_module


def _validate_binding_dispatches(configs: tuple[RegionShimConfig, ...]) -> None:
    """Reject variants that disagree about one runtime installation destination.

    Args:
        configs: Configurations already proven to target one source module and path.

    Raises:
        ValueError: If variants for one runtime binding disagree about descriptor or execution
            identity, or if multiple class variants target the same binding.
    """
    identities: dict[str, tuple[object, ...]] = {}
    class_counts: dict[str, int] = {}
    for config in configs:
        for binding in config.bindings:
            runtime_qualname = _binding_runtime_qualname(binding)
            identity = (
                binding.kind,
                binding.owner_class,
                binding.target_owner_class,
                binding.execution_kind,
            )
            previous = identities.setdefault(runtime_qualname, identity)
            if previous != identity:
                raise ValueError(f"typed-region variants disagree about binding {runtime_qualname}")
            if binding.kind == "class":
                class_counts[runtime_qualname] = class_counts.get(runtime_qualname, 0) + 1
    duplicate_class = next((name for name, count in class_counts.items() if count > 1), None)
    if duplicate_class is not None:
        raise ValueError(f"typed-region class binding has multiple variants: {duplicate_class}")


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
