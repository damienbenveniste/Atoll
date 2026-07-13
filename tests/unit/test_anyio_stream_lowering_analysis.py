"""Focused structural coverage for the generic AnyIO source lowerer."""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Protocol, cast

import libcst as cst
import pytest

from atoll.models import SymbolId
from atoll.source_optimization import anyio_stream_lowering as implementation
from atoll.source_optimization.models import (
    SourceOptimizationIdentity,
    SourceOptimizationPlan,
)


class _ProtocolView(Protocol):
    entry_class: str
    entry_method: str
    runner_class: str
    runner_name: str
    next_method: str
    owner_attribute: str
    terminal_type: str


class _ReducerView(Protocol):
    class_expression: str
    owner_expression: str
    method_name: str
    callable_property: str


class _Subject(Protocol):
    await_completion_lowering: Callable[[str], ast.NodeTransformer]

    def source_path(self, root: Path, relative: PurePosixPath) -> Path: ...
    def expected_source_hash(self, plan: SourceOptimizationPlan) -> str: ...
    def sha256(self, source: str) -> str: ...
    def analyze_shape(
        self,
        root: Path,
        tree: ast.Module,
        plan: SourceOptimizationPlan,
    ) -> object: ...
    def shared_owner_class(self, *symbols: SymbolId) -> str: ...
    def class_node(self, tree: ast.Module, name: str) -> ast.ClassDef: ...
    def validate_class(self, node: ast.ClassDef) -> None: ...
    def named_method(
        self,
        node: ast.ClassDef,
        name: str,
        expected: type[ast.FunctionDef] | type[ast.AsyncFunctionDef],
        *,
        role: str,
    ) -> ast.FunctionDef | ast.AsyncFunctionDef: ...
    def transport_pair(self, transport: str) -> tuple[str, str]: ...
    def stream_capacity(
        self,
        initializer: ast.FunctionDef,
        sender: str,
        receiver: str,
    ) -> int: ...
    def spawn_shape(
        self,
        owner: ast.FunctionDef,
        worker: SymbolId,
    ) -> tuple[str, str, str, str, str]: ...
    def worker_delegate(
        self,
        worker: ast.AsyncFunctionDef,
        *,
        sender_expression: str,
    ) -> tuple[str, str, str]: ...
    def run_shape(
        self,
        owner: ast.ClassDef,
        run: ast.AsyncFunctionDef,
        task_name: str,
        *,
        async_result_type: str | None,
    ) -> tuple[str, str, str, str, str, tuple[str, ...]]: ...
    def result_shape(
        self,
        worker: ast.AsyncFunctionDef,
        *,
        sender_expression: str,
        task_name: str,
        result_name: str,
    ) -> tuple[str, str, str, str | None]: ...
    def validate_worker_wrapper(
        self,
        worker: ast.AsyncFunctionDef,
        *,
        run_method_name: str,
        sender_expression: str,
        result_constructor: str,
        async_result_type: str | None,
    ) -> None: ...
    def validate_consumer(self, consumer: ast.AsyncFunctionDef, receiver: str) -> None: ...
    def protocol_shape(
        self,
        tree: ast.Module,
        owner_class: str,
        consumer_name: str,
    ) -> _ProtocolView | None: ...
    def echo_protocol_candidate(
        self,
        method: ast.AsyncFunctionDef,
    ) -> tuple[str, str, str] | None: ...
    def names(self, plan_id: str) -> object: ...
    def protocol_support(self, protocol: object | None, names: object) -> str: ...
    def reducer_shape(
        self,
        root: Path,
        tree: ast.Module,
        plan: SourceOptimizationPlan,
        consumer: ast.AsyncFunctionDef,
    ) -> _ReducerView | None: ...
    def reducer_support(self, reducer: object | None, names: object) -> str: ...
    def reflected_callable_property(self, method: ast.FunctionDef) -> str: ...
    def validate_callable_property(self, node: ast.ClassDef, name: str) -> None: ...
    def consumer_reducer_call(
        self,
        tree: ast.Module,
        consumer: ast.AsyncFunctionDef,
        *,
        method_name: str,
        class_name: str,
    ) -> tuple[str, str]: ...
    def cst_method(self, module: cst.Module, qualname: str) -> cst.FunctionDef: ...
    def lazy_scan_match(
        self,
        statements: list[cst.BaseStatement],
        index: int,
    ) -> tuple[cst.SimpleStatementLine, cst.SimpleStatementLine, cst.For, str] | None: ...
    def body_source(self, module: cst.Module, body: cst.IndentedBlock) -> str: ...
    def after_docstring(self, statements: Sequence[cst.BaseStatement]) -> int: ...
    def cst_path(self, node: cst.CSTNode) -> str | None: ...
    def expression_path(self, node: ast.expr) -> str | None: ...
    def excluded_node_guard(self, excluded: tuple[str, ...]) -> str: ...
    def protocol_body(self, module: cst.Module, protocol: object, names: object) -> str: ...
    def identity_captures(self, shape: object, names: object) -> tuple[str, ...]: ...
    def cst_spawn_loop(self, node: cst.For, shape: object) -> bool: ...

    consumer_body_transformer: Callable[
        [str, object | None, object | None, object], cst.CSTTransformer
    ]


_MEMBERS = vars(implementation)
_active_mapping_use_rejection = cast(
    Callable[[ast.Attribute, ast.AST | None, dict[ast.AST, ast.AST]], str | None],
    _MEMBERS["_active_mapping_use_rejection"],
)
subject = cast(
    _Subject,
    SimpleNamespace(
        await_completion_lowering=_MEMBERS["_AwaitCompletionLowering"],
        source_path=_MEMBERS["_source_path"],
        expected_source_hash=_MEMBERS["_expected_source_hash"],
        sha256=_MEMBERS["_sha256"],
        analyze_shape=_MEMBERS["_analyze_shape"],
        shared_owner_class=_MEMBERS["_shared_owner_class"],
        class_node=_MEMBERS["_class_node"],
        validate_class=_MEMBERS["_validate_class"],
        named_method=_MEMBERS["_named_method"],
        transport_pair=_MEMBERS["_transport_pair"],
        stream_capacity=_MEMBERS["_stream_capacity"],
        spawn_shape=_MEMBERS["_spawn_shape"],
        worker_delegate=_MEMBERS["_worker_delegate"],
        run_shape=_MEMBERS["_run_shape"],
        result_shape=_MEMBERS["_result_shape"],
        validate_worker_wrapper=_MEMBERS["_validate_worker_wrapper"],
        validate_consumer=_MEMBERS["_validate_consumer"],
        protocol_shape=_MEMBERS["_protocol_shape"],
        echo_protocol_candidate=_MEMBERS["_echo_protocol_candidate"],
        names=_MEMBERS["_names"],
        protocol_support=_MEMBERS["_protocol_support"],
        reducer_shape=_MEMBERS["_reducer_shape"],
        reducer_support=_MEMBERS["_reducer_support"],
        reflected_callable_property=_MEMBERS["_reflected_callable_property"],
        validate_callable_property=_MEMBERS["_validate_callable_property"],
        consumer_reducer_call=_MEMBERS["_consumer_reducer_call"],
        cst_method=_MEMBERS["_cst_method"],
        lazy_scan_match=_MEMBERS["_lazy_scan_match"],
        body_source=_MEMBERS["_body_source"],
        after_docstring=_MEMBERS["_after_docstring"],
        cst_path=_MEMBERS["_cst_path"],
        expression_path=_MEMBERS["_expression_path"],
        excluded_node_guard=_MEMBERS["_excluded_node_guard"],
        protocol_body=_MEMBERS["_protocol_body"],
        identity_captures=_MEMBERS["_identity_captures"],
        cst_spawn_loop=_MEMBERS["_cst_spawn_loop"],
        consumer_body_transformer=_MEMBERS["_ConsumerBodyTransformer"],
    ),
)

_PROTOCOL_SOURCE = """
class Terminal:
    pass

class Owner:
    async def events(self):
        yield Terminal()

class Runner:
    def __init__(self):
        self.owner = Owner()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def next(self, event):
        raise StopAsyncIteration

class Entry:
    def iterate(self):
        return Runner()

    async def run(self):
        event = None
        async with self.iterate() as runner:
            try:
                while not isinstance(event, Terminal):
                    event = await runner.next(event)
            except StopAsyncIteration:
                return event
        return event
"""

_REDUCER_SOURCE = """
import inspect

class Reducer:
    def __init__(self, call):
        self._call = call

    @property
    def call(self):
        return self._call

    def reduce(self, context, current, inputs):
        count = len(inspect.signature(self.call).parameters)
        if count == 2:
            return self.call(current, inputs)
        return self.call(context, current, inputs)
"""

_CONSUMER_SOURCE = """
from fixture.reducer import Reducer

class Owner:
    async def events(self):
        if isinstance(self.reducer, Reducer):
            value = self.reducer.reduce(context, current, inputs)
        return value
"""

_RUN_SHAPE_SOURCE = """
class Owner:
    def make_async(self, node) -> AsyncResult:
        return node

    def make_sync(self, node) -> int:
        return 0

    def wrap(self, node) -> AsyncResult:
        if isinstance(node, ForkNode):
            return self.make_async(node)
        return self.make_sync(node)

    async def run(self, task: WorkItem):
        mapping = self.nodes
        node = mapping[task.node_id]
        if isinstance(node, AsyncNode):
            return self.wrap(node)
        if isinstance(node, StepNode):
            return await node.call(task.value)
        return self.make_sync(node)
"""

_WORKER_SOURCE = """
async def worker(self, item):
    try:
        result = await self.run(item)
    except BaseException as exc:
        await self.sender.send(Result(item, (), error=exc))
        return
    if isinstance(result, AsyncResult):
        async for value in result.iterable:
            await self.sender.send(Result(item, value))
        await self.sender.send(Result(item, ()))
    else:
        await self.sender.send(Result(item, result))
"""


def test_source_paths_and_hash_identity_reject_unsafe_or_stale_inputs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    source = root / "owner.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")

    assert subject.source_path(root, PurePosixPath("owner.py")) == source
    absolute_path = PurePosixPath(f"{chr(47)}owner.py")
    for path in (absolute_path, PurePosixPath("../owner.py")):
        with pytest.raises(ValueError, match="unsafe source path"):
            subject.source_path(root, path)
    with pytest.raises(ValueError, match="does not exist"):
        subject.source_path(root, PurePosixPath("missing.py"))

    outside = tmp_path / "outside.py"
    outside.write_text("VALUE = 2\n", encoding="utf-8")
    (root / "escape.py").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes project root"):
        subject.source_path(root, PurePosixPath("escape.py"))

    digest = subject.sha256(source.read_text(encoding="utf-8"))
    assert subject.expected_source_hash(_plan(((PurePosixPath("owner.py"), digest),))) == digest
    for hashes in ((), ((PurePosixPath("other.py"), digest),)):
        with pytest.raises(ValueError, match="one exact hash"):
            subject.expected_source_hash(_plan(hashes))


def test_shape_entry_rejects_missing_consumer_and_unshared_owner() -> None:
    plan = replace(_plan(()), consumer=None)
    with pytest.raises(ValueError, match="distinct consumer"):
        subject.analyze_shape(Path.cwd(), ast.parse(""), plan)

    with pytest.raises(ValueError, match="same-class methods"):
        subject.shared_owner_class(
            SymbolId("owner", "Owner.submit"),
            SymbolId("owner", "Other.worker"),
        )
    with pytest.raises(ValueError, match="nested owner"):
        subject.shared_owner_class(
            SymbolId("owner", "Outer.Owner.submit"),
            SymbolId("owner", "Outer.Owner.worker"),
        )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("self.active = {}", "has unsupported mutation"),
        ("value = self.active[0]", "subscript escapes ownership"),
        ("value = self.active.copy", "attribute escapes ownership"),
        ("consume(self.active)", "escapes through a call"),
        ("value = self.active or {}", "has unsupported observation"),
    ],
)
def test_completion_accounting_classifies_every_unowned_active_map_use(
    source: str,
    expected: str,
) -> None:
    """Every mutation or escape shape has deterministic rejection evidence."""
    tree = ast.parse(source)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    active = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and subject.expression_path(node) == "self.active"
    )

    assert _active_mapping_use_rejection(active, parents.get(active), parents) == expected


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("class Owner(metaclass=Meta):\n    pass\n", "metaclasses"),
        ("class Owner:\n    __slots__ = ()\n", "slotted owner"),
        ("@dataclass(slots=True)\nclass Owner:\n    pass\n", "slotted dataclasses"),
    ],
)
def test_class_validation_rejects_dynamic_storage(source: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        subject.validate_class(_class(source, "Owner"))


def test_class_validation_ignores_non_dataclass_decorators() -> None:
    subject.validate_class(_class("@registered\nclass Owner:\n    pass\n", "Owner"))


def test_class_and_method_lookup_reject_ambiguity_and_descriptors() -> None:
    tree = ast.parse("class Other:\n    pass\n")
    with pytest.raises(ValueError, match="one owner class"):
        subject.class_node(tree, "Owner")
    duplicate = ast.parse("class Owner:\n    pass\nclass Owner:\n    pass\n")
    with pytest.raises(ValueError, match="one owner class"):
        subject.class_node(duplicate, "Owner")

    owner = _class(
        "class Owner:\n    @staticmethod\n    def work(value):\n        return value\n",
        "Owner",
    )
    with pytest.raises(ValueError, match="decorators"):
        subject.named_method(owner, "work", ast.FunctionDef, role="owner")
    with pytest.raises(ValueError, match="one owner method"):
        subject.named_method(owner, "missing", ast.FunctionDef, role="owner")

    no_self = _class("class Owner:\n    def work(value):\n        return value\n", "Owner")
    with pytest.raises(ValueError, match="instance method"):
        subject.named_method(no_self, "work", ast.FunctionDef, role="owner")


@pytest.mark.parametrize(
    "transport",
    ["self.sender", "self.sender|self.sender", "sender|self.receiver", "a|b|c"],
)
def test_transport_pair_requires_distinct_instance_fields(transport: str) -> None:
    with pytest.raises(ValueError, match=r"sender|distinct"):
        subject.transport_pair(transport)


@pytest.mark.parametrize(
    ("expression", "expected", "error"),
    [
        ("create_stream()", 0, None),
        ("create_stream(3)", 3, None),
        ("create_stream(size=1)", None, ValueError),
        ("create_stream(1, 2)", None, ValueError),
        ("create_stream('large')", None, TypeError),
        ("create_stream(-1)", None, ValueError),
    ],
)
def test_stream_capacity_accepts_only_one_nonnegative_literal(
    expression: str,
    expected: int | None,
    error: type[Exception] | None,
) -> None:
    initializer = _function(
        f"def __post_init__(self):\n    self.sender, self.receiver = {expression}\n"
    )
    if error is None:
        assert subject.stream_capacity(initializer, "self.sender", "self.receiver") == expected
        return
    with pytest.raises(error):
        subject.stream_capacity(initializer, "self.sender", "self.receiver")


def test_stream_capacity_rejects_missing_endpoint_pair() -> None:
    initializer = _function("def __post_init__(self):\n    self.other = create_stream()\n")
    with pytest.raises(ValueError, match="planned sender and receiver"):
        subject.stream_capacity(initializer, "self.sender", "self.receiver")


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            "def submit(self, request, extra):\n"
            "    for item in request:\n"
            "        self.group.start_soon(self.worker, item)\n",
            "only self and one request",
        ),
        (
            "def submit(self, request):\n"
            "    for left, right in request:\n"
            "        self.group.start_soon(self.worker, left)\n",
            "one named item",
        ),
        (
            "def submit(self, request):\n"
            "    for item in other:\n"
            "        self.group.start_soon(self.worker, item)\n",
            "exactly one worker",
        ),
        (
            "def submit(self, request):\n"
            "    for item in request:\n"
            "        self.group.start_soon(self.other, item)\n",
            "planned same-class worker",
        ),
        (
            "def submit(self, request):\n"
            "    for item in request:\n"
            "        self.group.start_soon(self.worker, request)\n",
            "loop target",
        ),
        ("def submit(self, request):\n    return None\n", "one simple TaskGroup"),
    ],
)
def test_spawn_shape_rejects_ambiguous_scheduling(source: str, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        subject.spawn_shape(
            _function(source),
            SymbolId("fixture", "Owner.worker"),
        )


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            "async def worker(self, item, extra):\n"
            "    result = await self.run(item)\n"
            "    await self.sender.send(Result(item, result))\n",
            "only self and one work item",
        ),
        (
            "async def worker(self, item):\n"
            "    result = item\n"
            "    await self.sender.send(Result(item, result))\n",
            "delegate once",
        ),
        (
            "async def worker(self, item):\n    result = await self.run(item)\n    return result\n",
            "does not send records",
        ),
    ],
)
def test_worker_delegate_rejects_unsupported_wrappers(source: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        subject.worker_delegate(_async_function(source), sender_expression="self.sender")


def test_run_shape_preserves_aliases_and_excludes_async_protocol_nodes() -> None:
    tree = ast.parse(_RUN_SHAPE_SOURCE)
    owner = subject.class_node(tree, "Owner")
    run = cast(
        ast.AsyncFunctionDef,
        subject.named_method(owner, "run", ast.AsyncFunctionDef, role="run"),
    )

    shape = subject.run_shape(owner, run, "task", async_result_type="AsyncResult")

    assert shape == (
        "WorkItem",
        "self.nodes",
        "task.node_id",
        "StepNode",
        "call",
        ("ForkNode",),
    )


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            _RUN_SHAPE_SOURCE.replace(
                "mapping = self.nodes",
                "global STATE\n        mapping = self.nodes",
            ),
            "global, closure, or generator",
        ),
        (
            _RUN_SHAPE_SOURCE.replace("await node.call", "node.call"),
            "exactly one awaited callable",
        ),
        (
            _RUN_SHAPE_SOURCE.replace("await node.call", "await invoke"),
            "one loaded node attribute",
        ),
        (
            _RUN_SHAPE_SOURCE.replace(
                "node = mapping[task.node_id]",
                "node = self.default",
            ),
            "load its node",
        ),
        (
            _RUN_SHAPE_SOURCE.replace("mapping[task.node_id]", "mapping[0]"),
            "derive from the work item",
        ),
        (
            _RUN_SHAPE_SOURCE.replace("task: WorkItem", "task"),
            "explicit nominal annotation",
        ),
        (
            _RUN_SHAPE_SOURCE.replace("if isinstance(node, StepNode):", "if node:"),
            "explicit isinstance guard",
        ),
    ],
)
def test_run_shape_rejects_unsafe_execution_forms(source: str, message: str) -> None:
    tree = ast.parse(source)
    owner = subject.class_node(tree, "Owner")
    run = cast(
        ast.AsyncFunctionDef,
        subject.named_method(owner, "run", ast.AsyncFunctionDef, role="run"),
    )
    with pytest.raises((TypeError, ValueError), match=message):
        subject.run_shape(owner, run, "task", async_result_type="AsyncResult")


def test_result_shape_and_wrapper_validation_accept_private_protocol_fallback() -> None:
    worker = _async_function(_WORKER_SOURCE)

    result = subject.result_shape(
        worker,
        sender_expression="self.sender",
        task_name="item",
        result_name="result",
    )

    assert result == ("Result", "()", "error", "AsyncResult")
    subject.validate_worker_wrapper(
        worker,
        run_method_name="run",
        sender_expression="self.sender",
        result_constructor="Result",
        async_result_type="AsyncResult",
    )


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            "async def worker(self, item):\n    result = await self.run(item)\n",
            "constructed result records",
        ),
        (
            _WORKER_SOURCE.replace(
                "Result(item, result)",
                "OtherResult(item, result)",
            ),
            "constructor must be statically stable",
        ),
        (
            _WORKER_SOURCE.replace(", error=exc", ""),
            "ordinary success and exception",
        ),
        (
            _WORKER_SOURCE.replace("error=exc", "**error_fields"),
            "keyword expansion",
        ),
    ],
)
def test_result_shape_rejects_ambiguous_records(source: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        subject.result_shape(
            _async_function(source),
            sender_expression="self.sender",
            task_name="item",
            result_name="result",
        )


@pytest.mark.parametrize(
    ("source", "async_result_type", "message"),
    [
        (
            _WORKER_SOURCE.replace(
                "try:",
                "def nested():\n        return None\n    try:",
            ),
            "AsyncResult",
            "nested or nonlocal",
        ),
        (
            _WORKER_SOURCE.replace(
                "result = await self.run(item)",
                "result = await self.run(item)\n        observe(result)",
            ),
            "AsyncResult",
            "behavior outside run",
        ),
        (
            _WORKER_SOURCE.replace(
                "if isinstance(result, AsyncResult):\n"
                "        async for value in result.iterable:\n",
                "if isinstance(result, AsyncResult):\n        for value in result.iterable:\n",
            ),
            "AsyncResult",
            "no matching async-for",
        ),
    ],
)
def test_worker_wrapper_rejects_unproven_behavior(
    source: str,
    async_result_type: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        subject.validate_worker_wrapper(
            _async_function(source),
            run_method_name="run",
            sender_expression="self.sender",
            result_constructor="Result",
            async_result_type=async_result_type,
        )


def test_consumer_validation_requires_one_named_private_receiver_loop() -> None:
    missing = _async_function("async def events(self):\n    return None\n")
    with pytest.raises(ValueError, match="one private receiver"):
        subject.validate_consumer(missing, "self.receiver")
    tuple_target = _async_function(
        "async def events(self):\n    async for left, right in self.receiver:\n        pass\n"
    )
    with pytest.raises(TypeError, match="one named record"):
        subject.validate_consumer(tuple_target, "self.receiver")


def test_protocol_shape_recognizes_one_private_echo_loop() -> None:
    tree = ast.parse(_PROTOCOL_SOURCE)

    shape = subject.protocol_shape(tree, "Owner", "events")

    assert shape is not None
    assert (
        shape.entry_class,
        shape.entry_method,
        shape.runner_class,
        shape.runner_name,
        shape.next_method,
        shape.owner_attribute,
        shape.terminal_type,
    ) == ("Entry", "run", "Runner", "runner", "next", "owner", "Terminal")
    names = subject.names("source-opt-protocol")
    assert "ContextVar" in subject.protocol_support(shape, names)
    assert subject.protocol_support(None, names) == ""


def test_protocol_shape_rejects_ambiguous_entrypoints_and_owner_construction() -> None:
    assert (
        subject.protocol_shape(
            ast.parse(f"{_PROTOCOL_SOURCE}\n{_PROTOCOL_SOURCE}"),
            "Owner",
            "events",
        )
        is None
    )
    missing_owner = _PROTOCOL_SOURCE.replace("self.owner = Owner()", "self.other = object()")
    assert subject.protocol_shape(ast.parse(missing_owner), "Owner", "events") is None


@pytest.mark.parametrize(
    "source",
    [
        "async def run(self):\n    yield 1\n",
        "async def run(self):\n    event = None\n    event = await runner.next(event)\n",
        (
            "async def run(self):\n"
            "    async with self.iterate() as runner:\n"
            "        event = await runner.next(event)\n"
        ),
        (
            "async def run(self):\n"
            "    event = None\n"
            "    async with self.iterate() as runner:\n"
            "        event = await runner.next(event)\n"
        ),
        (
            "async def run(self):\n"
            "    event = None\n"
            "    async with self.iterate() as runner:\n"
            "        event = await runner.next(event)\n"
            "        event = await runner.next(event)\n"
        ),
    ],
)
def test_echo_protocol_rejects_incomplete_shapes(source: str) -> None:
    assert subject.echo_protocol_candidate(_async_function(source)) is None


def test_reflected_reducer_shape_is_proven_from_source_and_consumer(
    tmp_path: Path,
) -> None:
    reducer_path = tmp_path / "fixture" / "reducer.py"
    reducer_path.parent.mkdir()
    reducer_path.write_text(_REDUCER_SOURCE, encoding="utf-8")
    digest = hashlib.sha256(reducer_path.read_bytes()).hexdigest()
    plan = _plan(
        (
            (PurePosixPath("owner.py"), "owner-hash"),
            (PurePosixPath("fixture/reducer.py"), digest),
        ),
        reducer=SymbolId("fixture.reducer", "Reducer.reduce"),
    )
    owner_tree = ast.parse(_CONSUMER_SOURCE)
    consumer = cast(
        ast.AsyncFunctionDef,
        _class_method(owner_tree, "Owner", "events", ast.AsyncFunctionDef),
    )

    shape = subject.reducer_shape(tmp_path, owner_tree, plan, consumer)

    assert shape is not None
    assert (
        shape.class_expression,
        shape.owner_expression,
        shape.method_name,
        shape.callable_property,
    ) == ("Reducer", "self.reducer", "reduce", "call")
    names = subject.names("source-opt-reducer")
    assert "inspect.signature" in subject.reducer_support(shape, names)
    assert subject.reducer_support(None, names) == ""


def test_reducer_shape_rejects_nonmethod_nested_and_stale_sources(tmp_path: Path) -> None:
    consumer_tree = ast.parse(_CONSUMER_SOURCE)
    consumer = cast(
        ast.AsyncFunctionDef,
        _class_method(consumer_tree, "Owner", "events", ast.AsyncFunctionDef),
    )
    invalid_symbols = (
        (SymbolId("fixture.reducer", "reduce"), "direct class method"),
        (SymbolId("fixture.reducer", "Outer.Reducer.reduce"), "nested reducer"),
    )
    for reducer, message in invalid_symbols:
        with pytest.raises(ValueError, match=message):
            subject.reducer_shape(
                tmp_path,
                consumer_tree,
                _plan((), reducer=reducer),
                consumer,
            )

    reducer_path = tmp_path / "fixture" / "reducer.py"
    reducer_path.parent.mkdir()
    reducer_path.write_text(_REDUCER_SOURCE, encoding="utf-8")
    stale_plan = _plan(
        ((PurePosixPath("fixture/reducer.py"), "stale"),),
        reducer=SymbolId("fixture.reducer", "Reducer.reduce"),
    )
    with pytest.raises(ValueError, match="stale reducer source"):
        subject.reducer_shape(tmp_path, consumer_tree, stale_plan, consumer)

    missing_hash_plan = _plan(
        ((PurePosixPath("other.py"), "digest"),),
        reducer=SymbolId("fixture.reducer", "Reducer.reduce"),
    )
    with pytest.raises(ValueError, match="no exact hash"):
        subject.reducer_shape(tmp_path, consumer_tree, missing_hash_plan, consumer)


def test_consumer_reducer_call_requires_one_imported_nominal_owner() -> None:
    no_call_tree = ast.parse(
        "from fixture.reducer import Reducer\n"
        "class Owner:\n"
        "    async def events(self):\n"
        "        return None\n"
    )
    no_call = cast(
        ast.AsyncFunctionDef,
        _class_method(no_call_tree, "Owner", "events", ast.AsyncFunctionDef),
    )
    with pytest.raises(ValueError, match="exactly once"):
        subject.consumer_reducer_call(
            no_call_tree,
            no_call,
            method_name="reduce",
            class_name="Reducer",
        )

    no_proof_tree = ast.parse(
        _CONSUMER_SOURCE.replace("if isinstance(self.reducer, Reducer):", "if True:")
    )
    no_proof = cast(
        ast.AsyncFunctionDef,
        _class_method(no_proof_tree, "Owner", "events", ast.AsyncFunctionDef),
    )
    with pytest.raises(ValueError, match="isinstance proof"):
        subject.consumer_reducer_call(
            no_proof_tree,
            no_proof,
            method_name="reduce",
            class_name="Reducer",
        )

    no_import_tree = ast.parse(
        _CONSUMER_SOURCE.replace("from fixture.reducer import Reducer\n", "")
    )
    no_import = cast(
        ast.AsyncFunctionDef,
        _class_method(no_import_tree, "Owner", "events", ast.AsyncFunctionDef),
    )
    with pytest.raises(ValueError, match="import the reducer class"):
        subject.consumer_reducer_call(
            no_import_tree,
            no_import,
            method_name="reduce",
            class_name="Reducer",
        )


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            _REDUCER_SOURCE.replace(
                "def reduce(self, context, current, inputs):",
                "def reduce(self, current, inputs):",
            ),
            "self, context, current, and input",
        ),
        (
            _REDUCER_SOURCE.replace(
                "count = len(inspect.signature(self.call).parameters)",
                "count = 2",
            ),
            "reflect one direct callable property",
        ),
        (
            _REDUCER_SOURCE.replace("if count == 2:", "if count == 3:"),
            "two-argument",
        ),
        (
            _REDUCER_SOURCE.replace(
                "return self.call(context, current, inputs)",
                "return current",
            ).replace("return self.call(current, inputs)", "return current"),
            "both arity branches",
        ),
    ],
)
def test_reflected_reducer_rejects_unproven_shapes(source: str, message: str) -> None:
    reducer = cast(
        ast.FunctionDef,
        _class_method(ast.parse(source), "Reducer", "reduce", ast.FunctionDef),
    )
    with pytest.raises(ValueError, match=message):
        subject.reflected_callable_property(reducer)


@pytest.mark.parametrize(
    "source",
    [
        "class Reducer:\n    pass\n",
        (
            "class Reducer:\n"
            "    @property\n"
            "    def call(self):\n"
            "        value = self._call\n"
            "        return value\n"
        ),
        ("class Reducer:\n    @property\n    def call(self):\n        return make_call()\n"),
    ],
)
def test_callable_property_requires_one_direct_readonly_field(source: str) -> None:
    with pytest.raises(ValueError, match=r"property|direct instance field"):
        subject.validate_callable_property(_class(source, "Reducer"), "call")


def test_cst_helpers_match_lazy_reducer_scan_and_method_bodies() -> None:
    module = cst.parse_module(
        "class Owner:\n"
        "    def work(self):\n"
        "        snapshot = list(self.active.values())\n"
        "        completed = []\n"
        "        for item in self.scan(source, snapshot):\n"
        "            self.reducers.pop(item)\n"
    )
    method = subject.cst_method(module, "Owner.work")
    body = cast(cst.IndentedBlock, method.body)

    match = subject.lazy_scan_match(list(body.body), 0)

    assert match is not None
    assert match[-1] == "reducers"
    assert "snapshot" in subject.body_source(module, body)
    assert subject.after_docstring(body.body) == 0
    assert (
        subject.after_docstring(
            cast(cst.IndentedBlock, cst.parse_statement("def f():\n    'doc'\n").body).body
        )
        == 1
    )
    assert subject.cst_path(cst.Integer("1")) is None
    assert subject.expression_path(ast.Constant(value=1)) is None
    assert subject.excluded_node_guard(()) == ""
    assert "Solo," in subject.excluded_node_guard(("Solo",))
    assert "Left, Right" in subject.excluded_node_guard(("Left", "Right"))


def test_protocol_and_consumer_cst_lowering_cover_private_fast_paths() -> None:
    protocol = subject.protocol_shape(ast.parse(_PROTOCOL_SOURCE), "Owner", "events")
    assert protocol is not None
    names = subject.names("source-opt-cst")

    protocol_body = subject.protocol_body(
        cst.parse_module(_PROTOCOL_SOURCE),
        protocol,
        names,
    )

    assert "protocol_next" in protocol_body
    consumer_module = cst.parse_module(
        "class Owner:\n"
        "    async def events(self):\n"
        "        async for record in self.receiver:\n"
        "            event = yield record\n"
        "        snapshot = list(self.active.values())\n"
        "        completed = []\n"
        "        for item in self.scan(source, snapshot):\n"
        "            self.reducers.pop(item)\n"
    )
    protocol_transformer = subject.consumer_body_transformer(
        "self.receiver",
        None,
        protocol,
        names,
    )
    protocol_code = consumer_module.visit(protocol_transformer).code
    assert "receive_nowait" in protocol_code
    assert "protocol_forward" in protocol_code
    assert "if self.reducers:" in protocol_code

    reducer = SimpleNamespace(owner_expression="self.reducer", method_name="reduce")
    reducer_module = cst.parse_module("result = self.reducer.reduce(context, current, inputs)\n")
    reducer_transformer = subject.consumer_body_transformer(
        "self.receiver",
        reducer,
        None,
        names,
    )
    reducer_code = reducer_module.visit(reducer_transformer).code
    assert "_reduce(self.reducer, context, current, inputs)" in reducer_code
    nonmatching = cst.parse_module("result = self.other.reduce(context, current, inputs)\n")
    assert nonmatching.visit(reducer_transformer).code == nonmatching.code

    tuple_receiver = cst.parse_module(
        "async def events(self):\n    async for left, right in self.receiver:\n        pass\n"
    )
    assert "async for left, right" in tuple_receiver.visit(protocol_transformer).code

    extra_awaits = _PROTOCOL_SOURCE.replace(
        "                    event = await runner.next(event)",
        "                    await pending\n"
        "                    await self.other()\n"
        "                    event = await runner.next(event)",
    )
    assert "protocol_next" in subject.protocol_body(
        cst.parse_module(extra_awaits),
        protocol,
        names,
    )


def test_lazy_scan_match_rejects_async_and_nonreducer_scans() -> None:
    async_module = cst.parse_module(
        "snapshot = list(self.active.values())\n"
        "completed = []\n"
        "async for item in self.scan(snapshot):\n"
        "    self.reducers.pop(item)\n"
    )
    assert subject.lazy_scan_match(list(async_module.body), 0) is None

    no_reducer = cst.parse_module(
        "snapshot = list(self.active.values())\n"
        "completed = []\n"
        "for item in self.scan(snapshot):\n"
        "    completed.append(item)\n"
    )
    assert subject.lazy_scan_match(list(no_reducer.body), 0) is None

    malformed_snapshot = cst.parse_module(
        "snapshot = self.active\n"
        "completed = []\n"
        "for item in self.scan(snapshot):\n"
        "    self.reducers.pop(item)\n"
    )
    assert subject.lazy_scan_match(list(malformed_snapshot.body), 0) is None


def test_advanced_identity_captures_include_reducer_and_protocol_guards() -> None:
    protocol = subject.protocol_shape(ast.parse(_PROTOCOL_SOURCE), "Owner", "events")
    assert protocol is not None
    reducer = SimpleNamespace(
        class_expression="Reducer",
        owner_expression="self.reducer",
        method_name="reduce",
        callable_property="call",
    )
    shape = SimpleNamespace(
        class_name="Owner",
        worker_method="worker",
        run_method_name="run_item",
        consumer=SimpleNamespace(name="events"),
        result_constructor="Result",
        reducer=reducer,
        protocol=protocol,
    )

    captures = subject.identity_captures(shape, subject.names("source-opt-captures"))

    assert any("expected_reducer_class = Reducer" in value for value in captures)
    assert any("expected_entry_class = Entry" in value for value in captures)
    assert any("expected_runner_class = Runner" in value for value in captures)


def test_cst_spawn_and_protocol_helpers_reject_stale_coordinates() -> None:
    shape = SimpleNamespace(
        spawn_item_name="item",
        request_name="request",
        task_group_expression="self.group",
        task_group_method="start_soon",
    )
    wrong_target = cast(
        cst.For,
        cst.parse_statement(
            "for other in request:\n    self.group.start_soon(self.worker, other)\n"
        ),
    )
    wrong_request = cast(
        cst.For,
        cst.parse_statement("for item in other:\n    self.group.start_soon(self.worker, item)\n"),
    )
    assert subject.cst_spawn_loop(wrong_target, shape) is False
    assert subject.cst_spawn_loop(wrong_request, shape) is False

    protocol = subject.protocol_shape(ast.parse(_PROTOCOL_SOURCE), "Owner", "events")
    assert protocol is not None
    stale_module = cst.parse_module("class Entry:\n    async def run(self):\n        return None\n")
    with pytest.raises(ValueError, match="replace one private protocol next call"):
        subject.protocol_body(stale_module, protocol, subject.names("source-opt-stale"))


def test_cst_method_and_await_lowering_reject_ambiguous_syntax() -> None:
    module = cst.parse_module("class Owner:\n    pass\n")
    with pytest.raises(ValueError, match="one method"):
        subject.cst_method(module, "Owner.missing")
    transformer = subject.await_completion_lowering("complete")
    with pytest.raises(TypeError, match="direct call"):
        transformer.visit(ast.Await(value=ast.Name(id="value")))


def _plan(
    source_hashes: tuple[tuple[PurePosixPath, str], ...],
    *,
    reducer: SymbolId | None = None,
) -> SourceOptimizationPlan:
    identity = SourceOptimizationIdentity(
        execution_plan_id="exec-plan-fixture",
        source_hashes=source_hashes,
        topology_fingerprint="topology",
        dialect="anyio-on-asyncio",
        lowering_version="lowering-v1",
        python_abi="cpython-312",
        transformation_versions=(),
    )
    return SourceOptimizationPlan(
        id="source-opt-fixture",
        identity=identity,
        source=PurePosixPath("owner.py"),
        owner=SymbolId("owner", "Owner.submit"),
        worker=SymbolId("owner", "Owner.worker"),
        consumer=SymbolId("owner", "Owner.events"),
        reducer=reducer,
        transport="self.sender|self.receiver",
        access_sites=(),
        entrypoint=SymbolId("owner", "Entry.run"),
        steps=(),
        semantic_boundaries=(),
    )


def _class(source: str, name: str) -> ast.ClassDef:
    return subject.class_node(ast.parse(source), name)


def _function(source: str) -> ast.FunctionDef:
    return cast(ast.FunctionDef, ast.parse(source).body[0])


def _async_function(source: str) -> ast.AsyncFunctionDef:
    return cast(ast.AsyncFunctionDef, ast.parse(source).body[0])


def _class_method(
    tree: ast.Module,
    class_name: str,
    method_name: str,
    expected: type[ast.FunctionDef] | type[ast.AsyncFunctionDef],
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    return subject.named_method(
        subject.class_node(tree, class_name),
        method_name,
        expected,
        role="fixture",
    )
