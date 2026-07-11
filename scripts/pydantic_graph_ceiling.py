"""Support a disposable Pydantic Graph optimization-ceiling experiment.

This module owns benchmark-only source transformations and measurement. It
never edits a target checkout: each transformation is applied to a copied
payload under the requested evidence directory. The experiment estimates
whether reflection hoisting and immediate-completion batching have enough
headroom to justify a semantics-preserving Atoll implementation.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Literal, TypedDict, cast

from scripts.run_pydantic_graph_benchmark import PYDANTIC_AI_REVISION, source_manifest

ExperimentArm = Literal["baseline", "reflection", "buffered", "unsafe_ceiling"]
ARMS: tuple[ExperimentArm, ...] = (
    "baseline",
    "reflection",
    "buffered",
    "unsafe_ceiling",
)
MINIMUM_HEADROOM = 1.15


class CeilingExperimentError(RuntimeError):
    """Raised when the experiment cannot produce trustworthy evidence."""


@dataclass(frozen=True, slots=True)
class CeilingExperimentOptions:
    """Inputs and measurement policy for one disposable ceiling experiment.

    Attributes:
        checkout: Existing pinned Pydantic AI checkout, treated as read-only.
        evidence_root: Destination for copied payloads, logs, and reports.
        workload: Stable benchmark program executed for each arm.
        semantic_probe: Short correctness-smoke program executed before timing.
        python: Interpreter from an environment containing target dependencies.
        warmups: Number of unmeasured rotating arm groups.
        samples: Number of measured rotating arm groups.
        minimum_headroom: Ratio required to recommend implementation work.
    """

    checkout: Path
    evidence_root: Path
    workload: Path
    semantic_probe: Path
    python: Path
    warmups: int = 1
    samples: int = 7
    minimum_headroom: float = MINIMUM_HEADROOM

    def __post_init__(self) -> None:
        """Reject invalid timing policy before creating evidence state."""
        if self.warmups < 0:
            raise ValueError("warmups must be non-negative")
        if self.samples < 1:
            raise ValueError("samples must be positive")
        if self.minimum_headroom <= 1:
            raise ValueError("minimum_headroom must be greater than 1")


@dataclass(frozen=True, slots=True)
class CommandEvidence:
    """One subprocess invocation retained as experiment evidence."""

    arm: ExperimentArm
    phase: str
    returncode: int
    duration_seconds: float
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class ArmSummary:
    """Correctness-smoke status and measured median for one experiment arm."""

    arm: ExperimentArm
    probe_passed: bool
    semantic_status: str
    context_isolated: bool | None
    sample_seconds: tuple[float, ...]
    median_seconds: float
    speedup_over_baseline: float


@dataclass(frozen=True, slots=True)
class CeilingExperimentResult:
    """Retained result of an optimization-ceiling experiment."""

    revision: str
    source_unchanged: bool
    summaries: tuple[ArmSummary, ...]
    commands: tuple[CommandEvidence, ...]
    minimum_headroom: float
    observed_headroom: float
    promising_research_direction: bool
    report_json: Path
    report_markdown: Path


class _ArmJson(TypedDict):
    arm: str
    median_seconds: float
    sample_seconds: list[float]
    probe_passed: bool
    semantic_status: str
    context_isolated: bool | None
    speedup_over_baseline: float


class _CommandJson(TypedDict):
    arm: str
    duration_seconds: float
    phase: str
    returncode: int
    stderr: str
    stdout: str


def run_ceiling_experiment(options: CeilingExperimentOptions) -> CeilingExperimentResult:
    """Stage source copies, validate every arm, and measure alternating runs.

    Args:
        options: Checkout, evidence, interpreter, and sampling policy.

    Returns:
        CeilingExperimentResult: Reports and normalized speedup evidence.

    Raises:
        CeilingExperimentError: If source anchors drift, an arm fails, or the
            checkout is not the pinned revision.
    """
    package_root = options.checkout.resolve() / "pydantic_graph" / "pydantic_graph"
    _validate_inputs(options, package_root)
    evidence_root = options.evidence_root.resolve()
    if evidence_root.exists():
        raise CeilingExperimentError(f"evidence root already exists: {evidence_root}")
    evidence_root.mkdir(parents=True)
    before = source_manifest(package_root)
    payloads = _stage_payloads(package_root, evidence_root / "payloads")
    commands, semantic_status = _run_semantic_probes(options, payloads)
    samples = _run_measurements(options, payloads, commands)
    result = _summarize(
        options,
        before == source_manifest(package_root),
        semantic_status,
        samples,
        commands,
    )
    _write_reports(result)
    return result


def apply_reflection_hoist(source: str) -> str:
    """Cache reducer arity while guarding private reducer replacement.

    Args:
        source: Original ``pydantic_graph.join`` source text.

    Returns:
        str: Transformed source for a disposable payload.

    Raises:
        CeilingExperimentError: If pinned source anchors changed.
    """
    updated = _replace_once(
        source,
        "        self._reducer = reducer\n        self._initial_factory = initial_factory\n",
        "        self._reducer = reducer\n"
        "        self._atoll_reducer_identity = None\n"
        "        self._atoll_reducer_code = None\n"
        "        self._atoll_reducer_parameter_count = None\n"
        "        self._initial_factory = initial_factory\n",
        "Join initializer",
    )
    return _replace_once(
        updated,
        "        n_parameters = len(inspect.signature(self.reducer).parameters)\n"
        "        if n_parameters == 2:\n"
        "            return cast(PlainReducerFunction[InputT, OutputT], "
        "self.reducer)(current, inputs)\n"
        "        else:\n"
        "            return cast(ContextReducerFunction[StateT, DepsT, InputT, OutputT], "
        "self.reducer)(ctx, current, inputs)\n",
        "        reducer = self.reducer\n"
        "        code = getattr(reducer, '__code__', None)\n"
        "        cacheable = (\n"
        "            inspect.isfunction(reducer)\n"
        "            and not hasattr(reducer, '__signature__')\n"
        "            and not hasattr(reducer, '__wrapped__')\n"
        "        )\n"
        "        if (\n"
        "            cacheable\n"
        "            and reducer is self._atoll_reducer_identity\n"
        "            and code is self._atoll_reducer_code\n"
        "            and self._atoll_reducer_parameter_count is not None\n"
        "        ):\n"
        "            n_parameters = self._atoll_reducer_parameter_count\n"
        "        else:\n"
        "            n_parameters = len(inspect.signature(reducer).parameters)\n"
        "            self._atoll_reducer_identity = reducer if cacheable else None\n"
        "            self._atoll_reducer_code = code if cacheable else None\n"
        "            self._atoll_reducer_parameter_count = "
        "n_parameters if cacheable else None\n"
        "        if n_parameters == 2:\n"
        "            return cast(PlainReducerFunction[InputT, OutputT], reducer)(current, inputs)\n"
        "        return cast(ContextReducerFunction[StateT, DepsT, InputT, OutputT], "
        "reducer)(ctx, current, inputs)\n",
        "Join.reduce",
    )


def apply_result_buffering(source: str) -> str:
    """Remove result-stream backpressure in an explicitly unsafe copied payload.

    Args:
        source: Original ``pydantic_graph.graph_builder`` source text.

    Returns:
        str: Transformed source for the buffering-only ceiling arm.

    Raises:
        CeilingExperimentError: If pinned source anchors changed.
    """
    return _replace_once(
        source,
        "        self.iter_stream_sender, self.iter_stream_receiver = "
        "create_memory_object_stream[_GraphTaskResult]()\n"
        "        self._next_node_run_id = 1\n",
        "        self.iter_stream_sender, self.iter_stream_receiver = "
        "create_memory_object_stream[_GraphTaskResult](\n"
        "            sys.maxsize\n"
        "        )\n"
        "        self._next_node_run_id = 1\n",
        "GraphIterator result buffering",
    )


def apply_immediate_batching(source: str) -> str:
    """Add immediate execution to an already-buffered unsafe ceiling payload.

    Awaiting steps, forks, unknown callables, dynamic scheduling, directly
    visible context mutation, and generator bytecode retain the original task
    path. Indirect task/context effects are intentionally not claimed safe.
    Unexpected suspension is surfaced as an error and is never retried.

    Args:
        source: Buffered ``pydantic_graph.graph_builder`` source text.

    Returns:
        str: Transformed source for a disposable payload.

    Raises:
        CeilingExperimentError: If pinned source anchors changed.
    """
    updated = _replace_once(
        source,
        "import inspect\nimport sys\n",
        "import dis\nimport inspect\nimport sys\n",
        "graph_builder imports",
    )
    updated = _replace_once(
        updated,
        "        self.iter_stream_sender, self.iter_stream_receiver = "
        "create_memory_object_stream[_GraphTaskResult](\n"
        "            sys.maxsize\n"
        "        )\n"
        "        self._next_node_run_id = 1\n",
        "        self.iter_stream_sender, self.iter_stream_receiver = "
        "create_memory_object_stream[_GraphTaskResult](\n"
        "            sys.maxsize\n"
        "        )\n"
        "        self._atoll_immediate_nodes = {}\n"
        "        self._next_node_run_id = 1\n",
        "GraphIterator immediate-node cache",
    )
    return _replace_once(
        updated,
        _ORIGINAL_EXECUTION_ROUTING,
        _BATCHED_EXECUTION_ROUTING,
        "GraphIterator execution routing",
    )


_ORIGINAL_EXECUTION_ROUTING = (
    "    def _handle_execution_request(self, request: Sequence[GraphTask]) -> None:\n"
    "        for new_task in request:\n"
    "            self.active_tasks[new_task.task_id] = new_task\n"
    "        for new_task in request:\n"
    "            self.task_group.start_soon(self._run_tracked_task, new_task)\n"
    "\n"
)

_BATCHED_EXECUTION_ROUTING = (
    "    def _handle_execution_request(self, request: Sequence[GraphTask]) -> None:\n"
    "        for new_task in request:\n"
    "            self.active_tasks[new_task.task_id] = new_task\n"
    "        for new_task in request:\n"
    "            if self._atoll_can_run_immediately(new_task):\n"
    "                self._atoll_run_immediately(new_task)\n"
    "            else:\n"
    "                self.task_group.start_soon(self._run_tracked_task, new_task)\n"
    "\n"
    "    def _atoll_can_run_immediately(self, task: GraphTask) -> bool:\n"
    "        cached = self._atoll_immediate_nodes.get(task.node_id)\n"
    "        if cached is not None:\n"
    "            return cached\n"
    "        node = self.graph.nodes[task.node_id]\n"
    "        if isinstance(node, Fork):\n"
    "            eligible = False\n"
    "        elif not isinstance(node, Step):\n"
    "            eligible = True\n"
    "        else:\n"
    "            call = node.call\n"
    "            code = getattr(call, '__code__', None)\n"
    "            blocked_names = {\n"
    "                'ContextVar', 'cancel', 'cancelled', 'checkpoint', 'copy_context',\n"
    "                'create_task', 'current_task', 'ensure_future', 'get_event_loop',\n"
    "                'get_running_loop', 'set', 'sleep', 'start_soon',\n"
    "            }\n"
    "            blocked_opcodes = {\n"
    "                'DELETE_DEREF', 'DELETE_GLOBAL', 'SEND', 'STORE_DEREF',\n"
    "                'STORE_GLOBAL', 'YIELD_FROM', 'YIELD_VALUE',\n"
    "            }\n"
    "            eligible = (\n"
    "                code is not None\n"
    "                and inspect.iscoroutinefunction(call)\n"
    "                and not blocked_names.intersection(code.co_names)\n"
    "                and not any(\n"
    "                    instruction.opname in blocked_opcodes\n"
    "                    for instruction in dis.get_instructions(code)\n"
    "                )\n"
    "            )\n"
    "        self._atoll_immediate_nodes[task.node_id] = eligible\n"
    "        return eligible\n"
    "\n"
    "    def _atoll_run_immediately(self, task: GraphTask) -> None:\n"
    "        coroutine = self._run_task(task)\n"
    "        try:\n"
    "            coroutine.send(None)\n"
    "        except StopIteration as completed:\n"
    "            result = _GraphTaskResult(task, completed.value)\n"
    "        except BaseException as exc:\n"
    "            result = _GraphTaskResult(task, [], error=exc)\n"
    "        else:\n"
    "            coroutine.close()\n"
    "            result = _GraphTaskResult(\n"
    "                task, [], error=RuntimeError('immediate graph task suspended unexpectedly')\n"
    "            )\n"
    "        try:\n"
    "            self.iter_stream_sender.send_nowait(result)\n"
    "        except (BrokenResourceError, ClosedResourceError):\n"
    "            pass\n"
    "\n"
)


def _validate_inputs(options: CeilingExperimentOptions, package_root: Path) -> None:
    if not package_root.is_dir():
        raise CeilingExperimentError(f"Pydantic Graph package is unavailable: {package_root}")
    git = shutil.which("git")
    if git is None:
        raise CeilingExperimentError("git is required for revision verification")
    revision = subprocess.run(
        (git, "-C", str(options.checkout), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != PYDANTIC_AI_REVISION:
        raise CeilingExperimentError(
            f"checkout revision is {revision}; expected {PYDANTIC_AI_REVISION}"
        )
    for label, path in (
        ("workload", options.workload),
        ("correctness probe", options.semantic_probe),
        ("python", options.python),
    ):
        if not path.is_file():
            raise CeilingExperimentError(f"{label} is unavailable: {path}")


def _stage_payloads(package_root: Path, payload_root: Path) -> dict[ExperimentArm, Path]:
    payloads: dict[ExperimentArm, Path] = {}
    for arm in ARMS:
        destination = payload_root / arm
        shutil.copytree(package_root, destination / "pydantic_graph")
        payloads[arm] = destination
    for arm in ("reflection", "buffered", "unsafe_ceiling"):
        join_path = payloads[arm] / "pydantic_graph" / "join.py"
        join_path.write_text(
            apply_reflection_hoist(join_path.read_text(encoding="utf-8")),
            encoding="utf-8",
        )
    buffered_builder = payloads["buffered"] / "pydantic_graph" / "graph_builder.py"
    buffered_builder.write_text(
        apply_result_buffering(buffered_builder.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    ceiling_builder = payloads["unsafe_ceiling"] / "pydantic_graph" / "graph_builder.py"
    ceiling_builder.write_text(
        apply_immediate_batching(
            apply_result_buffering(ceiling_builder.read_text(encoding="utf-8"))
        ),
        encoding="utf-8",
    )
    return payloads


def _run_semantic_probes(
    options: CeilingExperimentOptions,
    payloads: dict[ExperimentArm, Path],
) -> tuple[list[CommandEvidence], dict[ExperimentArm, bool]]:
    commands: list[CommandEvidence] = []
    statuses: dict[ExperimentArm, bool] = {}
    for arm in ARMS:
        print(f"Ceiling experiment: correctness probe [{arm}]")
        evidence = _run_arm_command(
            arm,
            "probe",
            options.python,
            options.semantic_probe,
            payloads[arm],
        )
        commands.append(evidence)
        statuses[arm] = evidence.returncode == 0
        _require_success(evidence)
        _validate_probe_evidence(evidence)
        print(f"Ceiling experiment: correctness probe [{arm}] passed")
    return commands, statuses


def _run_measurements(
    options: CeilingExperimentOptions,
    payloads: dict[ExperimentArm, Path],
    commands: list[CommandEvidence],
) -> dict[ExperimentArm, list[float]]:
    for group_index in range(options.warmups):
        for arm in _rotated_arms(group_index):
            print(f"Ceiling experiment: warmup {group_index + 1}/{options.warmups} [{arm}]")
            evidence = _run_arm_command(
                arm, "warmup", options.python, options.workload, payloads[arm]
            )
            commands.append(evidence)
            _require_success(evidence)
            print(f"Ceiling experiment: warmup [{arm}] {evidence.duration_seconds:.3f}s")
    samples: dict[ExperimentArm, list[float]] = {arm: [] for arm in ARMS}
    for group_index in range(options.samples):
        for arm in _rotated_arms(group_index + options.warmups):
            print(f"Ceiling experiment: sample {group_index + 1}/{options.samples} [{arm}]")
            evidence = _run_arm_command(
                arm, "sample", options.python, options.workload, payloads[arm]
            )
            commands.append(evidence)
            _require_success(evidence)
            samples[arm].append(evidence.duration_seconds)
            print(f"Ceiling experiment: sample [{arm}] {evidence.duration_seconds:.3f}s")
    return samples


def _run_arm_command(
    arm: ExperimentArm,
    phase: str,
    python: Path,
    command: Path,
    payload: Path,
) -> CommandEvidence:
    environment = {
        **os.environ,
        "ATOLL_DISABLE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": str(payload),
        "ATOLL_CEILING_ARM": arm,
        "ATOLL_CEILING_PAYLOAD": str(payload),
    }
    argv: tuple[str, ...] = (str(python), str(command))
    if phase == "probe":
        argv += ("--verify",)
    started = time.perf_counter()
    completed = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    return CommandEvidence(
        arm=arm,
        phase=phase,
        returncode=completed.returncode,
        duration_seconds=time.perf_counter() - started,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _summarize(
    options: CeilingExperimentOptions,
    source_unchanged: bool,
    semantic_status: dict[ExperimentArm, bool],
    samples: dict[ExperimentArm, list[float]],
    commands: list[CommandEvidence],
) -> CeilingExperimentResult:
    baseline_median = median(samples["baseline"])
    summaries = tuple(
        ArmSummary(
            arm=arm,
            probe_passed=semantic_status[arm],
            semantic_status=_semantic_status(arm),
            context_isolated=_context_isolated(commands, arm),
            sample_seconds=tuple(samples[arm]),
            median_seconds=median(samples[arm]),
            speedup_over_baseline=baseline_median / median(samples[arm]),
        )
        for arm in ARMS
    )
    ceiling = next(summary for summary in summaries if summary.arm == "unsafe_ceiling")
    report_root = options.evidence_root.resolve()
    return CeilingExperimentResult(
        revision=PYDANTIC_AI_REVISION,
        source_unchanged=source_unchanged,
        summaries=summaries,
        commands=tuple(commands),
        minimum_headroom=options.minimum_headroom,
        observed_headroom=ceiling.speedup_over_baseline,
        promising_research_direction=source_unchanged
        and ceiling.speedup_over_baseline >= options.minimum_headroom,
        report_json=report_root / "ceiling-report.json",
        report_markdown=report_root / "ceiling-report.md",
    )


def _rotated_arms(index: int) -> tuple[ExperimentArm, ...]:
    offset = index % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def _semantic_status(arm: ExperimentArm) -> str:
    if arm == "baseline":
        return "reference"
    if arm == "reflection":
        return "guarded"
    return "not-established"


def _validate_probe_evidence(evidence: CommandEvidence) -> None:
    payload = _probe_payload(evidence)
    if payload.get("arm") != evidence.arm:
        raise CeilingExperimentError(f"{evidence.arm} probe reported a different arm")
    if not isinstance(payload.get("context_isolated"), bool):
        raise CeilingExperimentError(f"{evidence.arm} probe omitted context isolation evidence")
    if payload.get("signature_guarded") is not True:
        raise CeilingExperimentError(f"{evidence.arm} probe did not verify signature guards")


def _context_isolated(commands: list[CommandEvidence], arm: ExperimentArm) -> bool | None:
    evidence = next(
        (command for command in commands if command.arm == arm and command.phase == "probe"),
        None,
    )
    if evidence is None:
        return None
    value = _probe_payload(evidence).get("context_isolated")
    return value if isinstance(value, bool) else None


def _probe_payload(evidence: CommandEvidence) -> dict[str, object]:
    try:
        payload: object = json.loads(evidence.stdout)
    except json.JSONDecodeError as error:
        raise CeilingExperimentError(
            f"{evidence.arm} probe did not emit valid JSON: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise CeilingExperimentError(f"{evidence.arm} probe emitted a non-object result")
    mapping = cast(dict[object, object], payload)
    return {str(key): value for key, value in mapping.items()}


def _require_success(evidence: CommandEvidence) -> None:
    if evidence.returncode != 0:
        raise CeilingExperimentError(
            f"{evidence.arm} {evidence.phase} failed with exit code "
            f"{evidence.returncode}: {evidence.stderr.strip()}"
        )


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    occurrences = source.count(old)
    if occurrences != 1:
        raise CeilingExperimentError(
            f"{label} anchor occurred {occurrences} times; expected pinned source exactly once"
        )
    return source.replace(old, new, 1)


def _write_reports(result: CeilingExperimentResult) -> None:
    payload = {
        "commands": [_command_json(command) for command in result.commands],
        "minimum_headroom": result.minimum_headroom,
        "observed_headroom": result.observed_headroom,
        "promising_research_direction": result.promising_research_direction,
        "revision": result.revision,
        "source_unchanged": result.source_unchanged,
        "summaries": [_arm_json(summary) for summary in result.summaries],
    }
    result.report_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result.report_markdown.write_text(_markdown_report(result), encoding="utf-8")


def _markdown_report(result: CeilingExperimentResult) -> str:
    lines = [
        "# Pydantic Graph Optimization Ceiling",
        "",
        f"- Revision: `{result.revision}`",
        f"- Checkout sources unchanged: {'yes' if result.source_unchanged else 'no'}",
        f"- Required implementation headroom: `{result.minimum_headroom:.3f}x`",
        f"- Observed unsafe ceiling: `{result.observed_headroom:.3f}x`",
        "- Scheduler semantics: `not established`",
        "- Recommendation: "
        f"`{'investigate-guarded-design' if result.promising_research_direction else 'stop'}`",
        "",
        "| Arm | Median | Speedup | Correctness smoke | Task context isolated | Semantic status |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    lines.extend(
        f"| {summary.arm} | {summary.median_seconds:.6f}s | "
        f"{summary.speedup_over_baseline:.3f}x | "
        f"{'passed' if summary.probe_passed else 'failed'} | "
        f"{_context_label(summary.context_isolated)} | "
        f"{summary.semantic_status} |"
        for summary in result.summaries
    )
    lines.extend(
        (
            "",
            "The `reflection` arm uses a lazy guarded reducer-arity cache. The `buffered` arm "
            "also removes result-stream backpressure. The `unsafe_ceiling` arm additionally "
            "drives statically non-suspending graph tasks immediately. Buffering and immediate "
            "execution change observable async behavior; their speedup is only an upper bound, "
            "not a promotable Atoll optimization.",
            "",
        )
    )
    return "\n".join(lines)


def _command_json(command: CommandEvidence) -> _CommandJson:
    return {
        "arm": command.arm,
        "duration_seconds": command.duration_seconds,
        "phase": command.phase,
        "returncode": command.returncode,
        "stderr": command.stderr,
        "stdout": command.stdout,
    }


def _context_label(value: bool | None) -> str:
    if value is None:
        return "not-recorded"
    return "yes" if value else "no"


def _arm_json(summary: ArmSummary) -> _ArmJson:
    return {
        "arm": summary.arm,
        "median_seconds": summary.median_seconds,
        "sample_seconds": list(summary.sample_seconds),
        "probe_passed": summary.probe_passed,
        "semantic_status": summary.semantic_status,
        "context_isolated": summary.context_isolated,
        "speedup_over_baseline": summary.speedup_over_baseline,
    }
