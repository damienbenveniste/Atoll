"""Private child-process bootstrap for baseline profiling.

This module is executed with `python -m atoll.runtime._profile_bootstrap` by the
profiling parent. It reads a JSON configuration, runs the requested script or
module in-process through `runpy`, writes structured evidence, and lets target
stdout, stderr, and exit status flow through the child process unchanged.
"""

from __future__ import annotations

import inspect
import json
import os
import runpy
import signal
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from dis import Positions, get_instructions
from pathlib import Path
from types import CodeType, FrameType
from typing import Literal, cast

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
type JsonObject = dict[str, JsonValue]
type ProfilePass = Literal["sampling", "types"]
type LaunchKind = Literal["script", "module"]
type _SignalHandler = signal.Handlers | int | Callable[[int, FrameType | None], object] | None

_SAMPLE_INTERVAL_SECONDS = 0.002
_MAX_SIGNATURES_PER_MEMBER = 8
_MAX_TYPE_OBSERVATIONS_PER_MEMBER = 256
_MONITORING_TOOL_ID = 5
_MODULE_PATH_PARTS = 2
_LIFECYCLE_EVENT_NAMES = ("start", "return", "yield", "resume", "unwind", "throw")


@dataclass(frozen=True, slots=True)
class _Config:
    profile_stage: ProfilePass
    launch_kind: LaunchKind
    target: str
    args: tuple[str, ...]
    project_root: Path
    payload_root: Path
    module_paths: tuple[tuple[str, str], ...]
    result_path: Path
    targets: frozenset[str]
    spawn_targets: tuple[_SpawnTarget, ...]


@dataclass(frozen=True, slots=True)
class _ModulePath:
    module: str
    suffix: str
    project_path: Path
    payload_path: Path


@dataclass(frozen=True, slots=True)
class _SpawnTarget:
    id: str
    owner: str
    lineno: int
    col_offset: int
    scheduler_method: str
    end_lineno: int | None
    end_col_offset: int | None


def main(argv: tuple[str, ...] | None = None) -> int:
    """Run one configured profiling pass and write JSON evidence.

    Args:
        argv: Optional command-line override used by tests.

    Returns:
        int: Target command exit status.

    Raises:
        BaseException: Re-raises target failures after persisting partial profile evidence.
    """
    args = tuple(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        return 2
    config = _read_config(Path(args[0]))
    os.chdir(config.project_root)
    _configure_import_path(config.payload_root)
    profiler: _SamplingProfiler | _TypeProfiler
    if config.profile_stage == "sampling":
        profiler = _SamplingProfiler(config)
    else:
        profiler = _TypeProfiler(config)
    profiler.start()
    exit_code = 0
    try:
        _run_target(config)
    except SystemExit as exc:
        exit_code = _system_exit_code(exc)
    except BaseException:
        profiler.stop()
        _write_json(config.result_path, profiler.payload())
        raise
    profiler.stop()
    _write_json(config.result_path, profiler.payload())
    return exit_code


class _SamplingProfiler:
    def __init__(self, config: _Config) -> None:
        self._mapper = _FrameMapper(config)
        self._sample_counts: Counter[str] = Counter()
        self._total_samples = 0
        self._previous_handler: _SignalHandler = None

    def start(self) -> None:
        """Enable statistical leaf-frame sampling without tracing distortion."""
        self._previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self._sample)
        signal.setitimer(signal.ITIMER_REAL, _SAMPLE_INTERVAL_SECONDS, _SAMPLE_INTERVAL_SECONDS)

    def stop(self) -> None:
        """Disable sampling and monitoring callbacks installed by this pass."""
        signal.setitimer(signal.ITIMER_REAL, 0.0, 0.0)
        if self._previous_handler is not None:
            signal.signal(signal.SIGALRM, signal.SIG_IGN)
            signal.signal(signal.SIGALRM, self._previous_handler)

    def payload(self) -> JsonObject:
        """Return JSON evidence from the sampling pass.

        Returns:
            JsonObject: Leaf samples and empty targeted-observation sections.
        """
        return {
            "total_samples": self._total_samples,
            "sample_counts": dict(sorted(self._sample_counts.items())),
            "lifecycle": _empty_lifecycle_payload(),
            "member_lifecycle": {},
            "signatures": {},
            "spawn_sites": [],
        }

    def _sample(self, _signum: int, frame: FrameType | None) -> None:
        self._total_samples += 1
        key = self._mapper.member_key(frame) if frame is not None else None
        if key is not None:
            self._sample_counts[key] += 1


class _TypeProfiler:
    def __init__(self, config: _Config) -> None:
        self._mapper = _FrameMapper(config)
        self._targets = config.targets
        self._spawn_targets = config.spawn_targets
        self._spawn_targets_by_owner = _spawn_targets_by_owner(config.spawn_targets)
        self._signatures: dict[str, Counter[tuple[tuple[str, str], ...]]] = {}
        self._call_counts: Counter[str] = Counter()
        self._invocation_counts: Counter[str] = Counter()
        self._completed_calls: Counter[str] = Counter()
        self._active_calls: Counter[str] = Counter()
        self._max_active_calls: Counter[str] = Counter()
        self._pre_completion_suspensions: Counter[str] = Counter()
        self._overflow: set[str] = set()
        self._observation_capped: set[str] = set()
        self._lifecycle_counts: Counter[str] = Counter(dict.fromkeys(_LIFECYCLE_EVENT_NAMES, 0))
        self._member_lifecycle_counts: dict[str, Counter[str]] = {}
        self._target_codes: set[CodeType] = set()
        self._instruction_positions: dict[
            CodeType,
            dict[int, tuple[int | None, int | None, int | None, int | None]],
        ] = {}
        self._spawn_counts: Counter[str] = Counter()
        self._spawn_callable_counts: dict[str, Counter[str]] = {}
        self._enabled = False

    def start(self) -> None:
        """Enable complete lifecycle counts and bounded hot-member argument types.

        Raises:
            RuntimeError: Python's monitoring tool slot cannot be reserved.
        """
        monitoring = sys.monitoring
        events = monitoring.events
        try:
            monitoring.use_tool_id(_MONITORING_TOOL_ID, "atoll-baseline-profile")
            monitoring.register_callback(
                _MONITORING_TOOL_ID,
                events.PY_START,
                self._on_start,
            )
            monitoring.register_callback(
                _MONITORING_TOOL_ID,
                events.CALL,
                self._on_call,
            )
            for event, name in (
                (events.PY_RETURN, "return"),
                (events.PY_YIELD, "yield"),
                (events.PY_RESUME, "resume"),
                (events.PY_UNWIND, "unwind"),
                (events.PY_THROW, "throw"),
            ):
                monitoring.register_callback(
                    _MONITORING_TOOL_ID,
                    event,
                    self._lifecycle_callback(name),
                )
            monitoring.set_events(
                _MONITORING_TOOL_ID,
                events.PY_START | events.PY_UNWIND | events.PY_THROW,
            )
            self._enabled = True
        except ValueError as exc:
            raise RuntimeError("unable to reserve a Python monitoring tool for profiling") from exc

    def stop(self) -> None:
        """Remove global and code-local monitoring installed by this pass."""
        if not self._enabled:
            return
        monitoring = sys.monitoring
        try:
            monitoring.set_events(_MONITORING_TOOL_ID, monitoring.events.NO_EVENTS)
            for code in self._target_codes:
                monitoring.set_local_events(
                    _MONITORING_TOOL_ID,
                    code,
                    monitoring.events.NO_EVENTS,
                )
            monitoring.free_tool_id(_MONITORING_TOOL_ID)
        finally:
            self._enabled = False

    def payload(self) -> JsonObject:
        """Return JSON evidence from the type-observation pass.

        Returns:
            JsonObject: Complete invocation/lifecycle and bounded canonical type observations.
        """
        signatures: JsonObject = {}
        for key in sorted(self._signatures):
            counter = self._signatures[key]
            signatures[key] = {
                "call_count": self._call_counts[key],
                "invocation_count": self._invocation_counts[key],
                "polymorphic_overflow": key in self._overflow,
                "observation_capped": key in self._observation_capped,
                "signatures": [
                    {
                        "count": count,
                        "parameters": [
                            {
                                "parameter_name": name,
                                "type_path": type_path,
                                "count": count,
                            }
                            for name, type_path in signature
                        ],
                    }
                    for signature, count in sorted(
                        counter.items(),
                        key=lambda item: (-item[1], item[0]),
                    )
                ],
            }
        for key in sorted(self._targets | frozenset(self._call_counts)):
            signatures.setdefault(
                key,
                {
                    "call_count": self._call_counts[key],
                    "invocation_count": self._invocation_counts[key],
                    "completed_calls": self._completed_calls[key],
                    "max_active_calls": self._max_active_calls[key],
                    "pre_completion_suspensions": self._pre_completion_suspensions[key],
                    "polymorphic_overflow": key in self._overflow,
                    "observation_capped": key in self._observation_capped,
                    "signatures": [],
                },
            )
            member_payload = _object_value(signatures[key])
            member_payload["completed_calls"] = self._completed_calls[key]
            member_payload["max_active_calls"] = self._max_active_calls[key]
            member_payload["pre_completion_suspensions"] = self._pre_completion_suspensions[key]
        return {
            "total_samples": 0,
            "sample_counts": {},
            "lifecycle": dict(self._lifecycle_counts),
            "member_lifecycle": {
                key: dict(counts) for key, counts in sorted(self._member_lifecycle_counts.items())
            },
            "signatures": signatures,
            "spawn_sites": [
                {
                    "id": target.id,
                    "owner": target.owner,
                    "lineno": target.lineno,
                    "col_offset": target.col_offset,
                    "scheduler_method": target.scheduler_method,
                    "end_lineno": target.end_lineno,
                    "end_col_offset": target.end_col_offset,
                    "invocation_count": self._spawn_counts[target.id],
                    "callable_identities": dict(
                        sorted(self._spawn_callable_counts.get(target.id, Counter()).items())
                    ),
                }
                for target in self._spawn_targets
            ],
        }

    def _on_start(self, code: CodeType, _offset: int) -> object:
        key = self._mapper.code_key(code)
        if key is None or key not in self._targets:
            return sys.monitoring.DISABLE
        self._enable_target_lifecycle(code, key)
        self._count_lifecycle(key, "start")
        self._start_invocation(key)
        self._invocation_counts[key] += 1
        if key in self._observation_capped:
            return None
        callback_frame = inspect.currentframe()
        frame = callback_frame.f_back if callback_frame is not None else None
        if frame is None or frame.f_code is not code:
            return None
        self._call_counts[key] += 1
        signature = _frame_signature(frame)
        counter = self._signatures.setdefault(key, Counter())
        if signature not in counter and len(counter) >= _MAX_SIGNATURES_PER_MEMBER:
            self._overflow.add(key)
        else:
            counter[signature] += 1
        if self._call_counts[key] < _MAX_TYPE_OBSERVATIONS_PER_MEMBER:
            return None
        self._observation_capped.add(key)
        return None

    def _enable_target_lifecycle(self, code: CodeType, key: str) -> None:
        if code in self._target_codes:
            return
        events = sys.monitoring.events
        local_events = events.PY_RETURN | events.PY_YIELD | events.PY_RESUME
        if key in self._spawn_targets_by_owner:
            local_events |= events.CALL
        sys.monitoring.set_local_events(
            _MONITORING_TOOL_ID,
            code,
            local_events,
        )
        self._target_codes.add(code)

    def _on_call(
        self,
        code: CodeType,
        instruction_offset: int,
        callable_object: object,
        _arg0: object,
    ) -> object:
        key = self._mapper.code_key(code)
        targets = self._spawn_targets_by_owner.get(key or "", ())
        if not targets:
            return None
        position = self._instruction_position(code, instruction_offset)
        callable_name = getattr(callable_object, "__name__", None)
        for target in targets:
            if target.scheduler_method != callable_name or not _position_within_spawn_target(
                position, target
            ):
                continue
            self._spawn_counts[target.id] += 1
            identity = _canonical_callable_identity(callable_object)
            self._spawn_callable_counts.setdefault(target.id, Counter())[identity] += 1
        return None

    def _instruction_position(
        self,
        code: CodeType,
        offset: int,
    ) -> tuple[int | None, int | None, int | None, int | None]:
        positions = self._instruction_positions.get(code)
        if positions is None:
            positions = {
                instruction.offset: _instruction_source_position(instruction.positions)
                for instruction in get_instructions(code)
            }
            self._instruction_positions[code] = positions
        return positions.get(offset, (None, None, None, None))

    def _lifecycle_callback(self, name: str) -> Callable[..., object]:
        def callback(code: CodeType, *_args: object) -> object:
            key = self._mapper.code_key(code)
            if key is not None and key in self._targets:
                self._count_active_lifecycle(key, name)
            return None

        return callback

    def _start_invocation(self, key: str) -> None:
        self._active_calls[key] += 1
        self._max_active_calls[key] = max(self._max_active_calls[key], self._active_calls[key])

    def _count_active_lifecycle(self, key: str, name: str) -> None:
        if self._active_calls[key] <= 0:
            return
        self._count_lifecycle(key, name)
        if name == "yield":
            self._pre_completion_suspensions[key] += 1
        if name in {"return", "unwind"}:
            self._completed_calls[key] += 1
            self._active_calls[key] -= 1

    def _count_lifecycle(self, key: str, name: str) -> None:
        self._lifecycle_counts[name] += 1
        self._member_lifecycle_counts.setdefault(
            key,
            Counter(dict.fromkeys(_LIFECYCLE_EVENT_NAMES, 0)),
        )[name] += 1


class _FrameMapper:
    def __init__(self, config: _Config) -> None:
        self._paths = tuple(
            _ModulePath(
                module=module,
                suffix=suffix,
                project_path=(config.project_root / suffix).resolve(),
                payload_path=(config.payload_root / suffix).resolve(),
            )
            for module, suffix in sorted(config.module_paths, key=lambda item: (item[1], item[0]))
        )
        self._code_keys: dict[CodeType, str | None] = {}

    def member_key(self, frame: FrameType | None) -> str | None:
        """Map a Python frame to `module::qualname` using configured module paths.

        Args:
            frame: Runtime frame to map, or `None` when sampling has no active frame.

        Returns:
            str | None: Canonical member key when the frame belongs to project code.
        """
        if frame is None:
            return None
        return self.code_key(frame.f_code)

    def code_key(self, code: CodeType) -> str | None:
        """Map a code object to `module::qualname` using configured module paths.

        Args:
            code: Runtime code object to map and cache.

        Returns:
            str | None: Canonical member key for project code, otherwise `None`.
        """
        if code in self._code_keys:
            return self._code_keys[code]
        filename = code.co_filename
        qualname = code.co_qualname
        path = Path(filename).resolve()
        for module_path in self._paths:
            if path in {module_path.project_path, module_path.payload_path}:
                if "<locals>" in qualname or qualname == "<module>":
                    self._code_keys[code] = None
                    return None
                key = f"{module_path.module}::{qualname}"
                self._code_keys[code] = key
                return key
        self._code_keys[code] = None
        return None


def _run_target(config: _Config) -> None:
    original_argv = sys.argv[:]
    try:
        if config.launch_kind == "module":
            sys.argv = [config.target, *config.args]
            runpy.run_module(config.target, run_name="__main__", alter_sys=True)
        else:
            script_path = Path(config.target)
            if not script_path.is_absolute():
                script_path = config.project_root / script_path
            sys.argv = [str(script_path), *config.args]
            runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = original_argv


def _frame_signature(frame: FrameType) -> tuple[tuple[str, str], ...]:
    code = frame.f_code
    arg_count = code.co_argcount + code.co_kwonlyargcount
    names = list(code.co_varnames[:arg_count])
    next_index = arg_count
    if code.co_flags & inspect.CO_VARARGS:
        names.append(code.co_varnames[next_index])
        next_index += 1
    if code.co_flags & inspect.CO_VARKEYWORDS:
        names.append(code.co_varnames[next_index])
    return tuple(
        (name, _canonical_type(frame.f_locals[name])) for name in names if name in frame.f_locals
    )


def _canonical_type(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _canonical_callable_identity(value: object) -> str:
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if isinstance(module, str) and isinstance(qualname, str):
        return f"{module}.{qualname}"
    return _canonical_type(value)


def _instruction_source_position(
    position: Positions | None,
) -> tuple[int | None, int | None, int | None, int | None]:
    if position is None:
        return None, None, None, None
    return position.lineno, position.col_offset, position.end_lineno, position.end_col_offset


def _position_within_spawn_target(
    position: tuple[int | None, int | None, int | None, int | None],
    target: _SpawnTarget,
) -> bool:
    lineno, col_offset, end_lineno, end_col_offset = position
    if lineno is None or col_offset is None:
        return False
    if target.end_lineno is None or target.end_col_offset is None:
        return lineno == target.lineno and col_offset >= target.col_offset
    if end_lineno is None or end_col_offset is None:
        return False
    return (lineno, col_offset) >= (target.lineno, target.col_offset) and (
        end_lineno,
        end_col_offset,
    ) <= (target.end_lineno, target.end_col_offset)


def _spawn_targets_by_owner(
    targets: tuple[_SpawnTarget, ...],
) -> dict[str, tuple[_SpawnTarget, ...]]:
    grouped: dict[str, list[_SpawnTarget]] = {}
    for target in targets:
        grouped.setdefault(target.owner, []).append(target)
    return {
        owner: tuple(sorted(items, key=lambda item: (item.lineno, item.col_offset, item.id)))
        for owner, items in grouped.items()
    }


def _empty_lifecycle_payload() -> JsonObject:
    return {"start": 0, "return": 0, "yield": 0, "resume": 0, "unwind": 0, "throw": 0}


def _configure_import_path(payload_root: Path) -> None:
    payload_text = str(payload_root)
    if sys.path[:1] != [payload_text]:
        sys.path.insert(0, payload_text)
    existing = tuple(path for path in os.environ.get("PYTHONPATH", "").split(os.pathsep) if path)
    os.environ["PYTHONPATH"] = os.pathsep.join((payload_text, *existing))
    os.environ["ATOLL_DISABLE"] = "1"
    os.environ.pop("ATOLL_REQUIRE_COMPILED", None)


def _read_config(path: Path) -> _Config:
    payload = _object_value(cast(JsonValue, json.loads(path.read_text(encoding="utf-8"))))
    return _Config(
        profile_stage=_profile_pass(_string_field(payload, "profile_stage")),
        launch_kind=_launch_kind(_string_field(payload, "launch_kind")),
        target=_string_field(payload, "target"),
        args=tuple(_string_items(payload.get("args", []))),
        project_root=Path(_string_field(payload, "project_root")).resolve(),
        payload_root=Path(_string_field(payload, "payload_root")).resolve(),
        module_paths=tuple(
            (parts[0], parts[1])
            for parts in (
                tuple(_string_items(item)) for item in _list_value(payload.get("module_paths", []))
            )
            if len(parts) == _MODULE_PATH_PARTS
        ),
        result_path=Path(_string_field(payload, "result_path")).resolve(),
        targets=frozenset(_string_items(payload.get("targets", []))),
        spawn_targets=tuple(
            _spawn_target(_object_value(item))
            for item in _list_value(payload.get("spawn_targets", []))
        ),
    )


def _spawn_target(payload: JsonObject) -> _SpawnTarget:
    return _SpawnTarget(
        id=_string_field(payload, "id"),
        owner=_string_field(payload, "owner"),
        lineno=_int_field(payload, "lineno"),
        col_offset=_int_field(payload, "col_offset"),
        scheduler_method=_string_field(payload, "scheduler_method"),
        end_lineno=_optional_int_field(payload, "end_lineno"),
        end_col_offset=_optional_int_field(payload, "end_col_offset"),
    )


def _write_json(path: Path, payload: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _system_exit_code(exc: SystemExit) -> int:
    code = exc.code
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    return 1


def _profile_pass(value: str) -> ProfilePass:
    return "types" if value == "types" else "sampling"


def _launch_kind(value: str) -> LaunchKind:
    return "module" if value == "module" else "script"


def _object_value(value: JsonValue) -> JsonObject:
    if isinstance(value, dict):
        return value
    return {}


def _list_value(value: JsonValue) -> list[JsonValue]:
    if isinstance(value, list):
        return value
    return []


def _string_items(value: JsonValue) -> tuple[str, ...]:
    return tuple(item for item in _list_value(value) if isinstance(item, str))


def _string_field(payload: JsonObject, key: str) -> str:
    value = payload.get(key, "")
    return value if isinstance(value, str) else ""


def _int_field(payload: JsonObject, key: str) -> int:
    value = payload.get(key, 0)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _optional_int_field(payload: JsonObject, key: str) -> int | None:
    value = payload.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


if __name__ == "__main__":
    raise SystemExit(main())
