"""Deterministic public-API compatibility probes for installed corpus wheels."""

from __future__ import annotations

import argparse
import importlib
import io
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Protocol

type Canonical = dict[str, object]
type ProbeResult = tuple[Canonical, tuple[ModuleType, ...]]
type Probe = Callable[[], ProbeResult]


class _IntegerInputs(Protocol):
    @property
    def inputs(self) -> int: ...


def _module_path(module: ModuleType) -> str:
    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError(f"imported module {module.__name__!r} has no package file")
    return str(Path(raw_path).resolve(strict=True))


def _module_paths(modules: tuple[ModuleType, ...]) -> list[str]:
    return [_module_path(module) for module in modules]


def _run_probe(probe: Probe) -> ProbeResult:
    """Run a project probe without allowing sibling adapters to shadow it."""
    adapter_root = Path(__file__).resolve().parent
    original_path = sys.path[:]
    try:
        sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != adapter_root]
        return probe()
    finally:
        sys.path[:] = original_path


def _probe_anyio() -> ProbeResult:
    anyio = importlib.import_module("anyio")

    async def exchange() -> str:
        send, receive = anyio.create_memory_object_stream[str](1)
        async with send, receive:
            await send.send("atoll")
            return await receive.receive()

    return {"received": anyio.run(exchange)}, (anyio,)


def _probe_attrs() -> ProbeResult:
    attrs = importlib.import_module("attrs")

    record_type = attrs.make_class(
        "Record",
        {"count": attrs.field(), "label": attrs.field(default="atoll")},
    )
    record = record_type(3)
    return {"record": attrs.asdict(record)}, (attrs,)


def _probe_cattrs() -> ProbeResult:
    attrs = importlib.import_module("attrs")
    cattrs = importlib.import_module("cattrs")

    @attrs.define
    class Record:
        count: int
        tags: list[str]

    converter = cattrs.Converter()
    record = converter.structure({"count": 3, "tags": ["a", "b"]}, Record)
    return {"record": converter.unstructure(record)}, (cattrs, attrs)


def _probe_click() -> ProbeResult:
    click = importlib.import_module("click")
    click_testing = importlib.import_module("click.testing")

    @click.command()
    @click.option("--count", type=int, required=True)
    def repeat(count: int) -> None:
        click.echo("atoll" * count)

    result = click_testing.CliRunner().invoke(repeat, ["--count", "2"])
    return {"exit_code": result.exit_code, "output": result.output}, (click, click_testing)


def _probe_dulwich() -> ProbeResult:
    dulwich = importlib.import_module("dulwich")
    dulwich_objects = importlib.import_module("dulwich.objects")

    blob = dulwich_objects.Blob.from_string(b"atoll\n")
    return {"data": blob.data.decode(), "object_id": blob.id.decode()}, (dulwich, dulwich_objects)


def _probe_html5lib() -> ProbeResult:
    html5lib = importlib.import_module("html5lib")

    fragment = html5lib.parseFragment("<p>A&amp;B</p>")
    rendered = html5lib.serialize(fragment, omit_optional_tags=False, quote_attr_values="always")
    return {"html": rendered}, (html5lib,)


def _probe_httpx() -> ProbeResult:
    httpx = importlib.import_module("httpx")

    url = httpx.URL("https://example.test/a path?tag=b&tag=a").copy_add_param("page", 2)
    return {"query": list(url.params.multi_items()), "url": str(url)}, (httpx,)


def _probe_jsonschema() -> ProbeResult:
    jsonschema = importlib.import_module("jsonschema")

    validator = jsonschema.Draft202012Validator(
        {
            "type": "object",
            "properties": {"count": {"type": "integer", "minimum": 1}},
            "required": ["count"],
            "additionalProperties": False,
        }
    )
    errors = sorted(error.message for error in validator.iter_errors({"count": 0}))
    return {"errors": errors, "valid": validator.is_valid({"count": 2})}, (jsonschema,)


def _probe_mako() -> ProbeResult:
    mako = importlib.import_module("mako")
    mako_template = importlib.import_module("mako.template")

    template_type = vars(mako_template)["Template"]
    template = template_type("${name}: ${sum(values)}")
    return {"rendered": template.render(name="atoll", values=[1, 2, 3])}, (mako, mako_template)


def _probe_markdown() -> ProbeResult:
    markdown = importlib.import_module("markdown")

    rendered = markdown.markdown("# Atoll\n\n* one\n* two", output_format="html")
    return {"html": rendered}, (markdown,)


def _probe_marshmallow() -> ProbeResult:
    marshmallow = importlib.import_module("marshmallow")

    schema_type = marshmallow.Schema.from_dict({"count": marshmallow.fields.Int(required=True)})
    schema = schema_type()
    loaded = schema.load({"count": "3"})
    return {"dumped": schema.dump(loaded), "loaded": loaded}, (marshmallow,)


def _probe_mypy() -> ProbeResult:
    mypy = importlib.import_module("mypy")
    mypy_api = importlib.import_module("mypy.api")

    stdout, stderr, status = mypy_api.run(["-c", "value: int = 3", "--no-error-summary"])
    return {"status": status, "stderr": stderr, "stdout": stdout}, (mypy, mypy_api)


def _probe_networkx() -> ProbeResult:
    networkx = importlib.import_module("networkx")

    graph = networkx.Graph([(1, 2), (2, 3), (1, 3), (3, 4)])
    return {
        "degrees": [[node, degree] for node, degree in sorted(graph.degree())],
        "shortest_path": networkx.shortest_path(graph, 1, 4),
    }, (networkx,)


def _probe_pluggy() -> ProbeResult:
    pluggy = importlib.import_module("pluggy")

    hookspec = pluggy.HookspecMarker("atoll")
    hookimpl = pluggy.HookimplMarker("atoll")

    class Specs:
        @hookspec
        def transform(self, value: str) -> str:
            raise NotImplementedError

    class Plugin:
        @hookimpl
        def transform(self, value: str) -> str:
            return value.upper()

    manager = pluggy.PluginManager("atoll")
    manager.add_hookspecs(Specs)
    manager.register(Plugin())
    return {"results": manager.hook.transform(value="atoll")}, (pluggy,)


def _probe_pydantic() -> ProbeResult:
    pydantic = importlib.import_module("pydantic")

    model_type = pydantic.create_model("Record", count=(int, ...), label=(str, "atoll"))
    record = model_type.model_validate({"count": "3"})
    return {"record": record.model_dump(mode="json")}, (pydantic,)


def _probe_pydantic_graph() -> ProbeResult:
    anyio = importlib.import_module("anyio")
    pydantic_graph = importlib.import_module("pydantic_graph")

    async def run_graph() -> int:
        graph_builder = pydantic_graph.GraphBuilder(output_type=int)

        async def fan_out_call(_ctx: object) -> list[int]:
            return [1, 2, 3]

        async def double_call(ctx: _IntegerInputs) -> int:
            return ctx.inputs * 2

        fan_out_call.__annotations__ = {
            "_ctx": pydantic_graph.StepContext[None, None, None],
            "return": list[int],
        }
        double_call.__annotations__ = {
            "ctx": pydantic_graph.StepContext[None, None, int],
            "return": int,
        }
        fan_out = graph_builder.step(fan_out_call, node_id="fan_out")
        double = graph_builder.step(double_call, node_id="double")
        total = graph_builder.join(pydantic_graph.reduce_sum, initial_factory=lambda: 0)
        graph_builder.add_mapping_edge(fan_out, double)
        graph_builder.add(
            graph_builder.edge_from(graph_builder.start_node).to(fan_out),
            graph_builder.edge_from(double).to(total),
            graph_builder.edge_from(total).to(graph_builder.end_node),
        )
        return await graph_builder.build().run(state=None)

    return {"sum": anyio.run(run_graph)}, (pydantic_graph,)


def _probe_rich() -> ProbeResult:
    rich = importlib.import_module("rich")
    rich_console = importlib.import_module("rich.console")

    stream = io.StringIO()
    console = rich_console.Console(file=stream, width=20, color_system=None)
    console.print("[bold]Atoll[/bold]", {"count": 3})
    return {"text": stream.getvalue()}, (rich, rich_console)


def _probe_sortedcontainers() -> ProbeResult:
    sortedcontainers = importlib.import_module("sortedcontainers")

    values = sortedcontainers.SortedList([3, 1, 2, 2])
    values.add(0)
    return {"index_of_two": values.bisect_left(2), "values": list(values)}, (sortedcontainers,)


def _probe_sqlalchemy() -> ProbeResult:
    sqlalchemy = importlib.import_module("sqlalchemy")

    metadata = sqlalchemy.MetaData()
    records = sqlalchemy.Table(
        "records",
        metadata,
        sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
        sqlalchemy.Column("name", sqlalchemy.String, nullable=False),
    )
    engine = sqlalchemy.create_engine("sqlite://")
    metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(records.insert(), [{"id": 2, "name": "b"}, {"id": 1, "name": "a"}])
        rows = connection.execute(
            sqlalchemy.select(records.c.id, records.c.name).order_by(records.c.id)
        ).all()
    return {"rows": [list(row) for row in rows]}, (sqlalchemy,)


def _probe_sqlglot() -> ProbeResult:
    sqlglot = importlib.import_module("sqlglot")

    converted = sqlglot.transpile(
        "SELECT IF(score > 1, 'yes', 'no') AS answer",
        read="mysql",
        write="postgres",
    )
    return {"sql": converted}, (sqlglot,)


def _probe_sympy() -> ProbeResult:
    sympy = importlib.import_module("sympy")

    x = sympy.Symbol("x")
    expression = sympy.expand((x + 1) ** 3)
    return {"derivative": str(sympy.diff(expression, x)), "expression": str(expression)}, (sympy,)


def _probe_tomli() -> ProbeResult:
    tomli = importlib.import_module("tomli")

    document = tomli.loads('title = "Atoll"\nvalues = [3, 1, 2]\n[meta]\nenabled = true\n')
    return {"document": document}, (tomli,)


def _probe_tornado() -> ProbeResult:
    tornado = importlib.import_module("tornado")
    tornado_escape = importlib.import_module("tornado.escape")

    encoded = tornado_escape.json_encode({"name": "Atoll", "values": [1, 2]})
    return {
        "json": tornado_escape.json_decode(encoded),
        "query": tornado_escape.url_escape("a b/c", plus=False),
    }, (tornado, tornado_escape)


def _probe_trio() -> ProbeResult:
    trio = importlib.import_module("trio")

    async def exchange() -> list[int]:
        send, receive = trio.open_memory_channel[int](2)
        async with send, receive:
            await send.send(2)
            await send.send(3)
            return [await receive.receive(), await receive.receive()]

    return {"received": trio.run(exchange)}, (trio,)


def _probe_websockets() -> ProbeResult:
    websockets = importlib.import_module("websockets")
    websockets_datastructures = importlib.import_module("websockets.datastructures")

    headers = websockets_datastructures.Headers()
    headers["X-Atoll"] = "one"
    headers["X-Atoll"] = "two"
    headers["Content-Type"] = "text/plain"
    return {
        "items": list(headers.raw_items()),
        "values": headers.get_all("X-Atoll"),
    }, (websockets, websockets_datastructures)


PROBES: dict[str, Probe] = {
    "anyio": _probe_anyio,
    "attrs": _probe_attrs,
    "cattrs": _probe_cattrs,
    "click": _probe_click,
    "dulwich": _probe_dulwich,
    "html5lib": _probe_html5lib,
    "httpx": _probe_httpx,
    "jsonschema": _probe_jsonschema,
    "mako": _probe_mako,
    "markdown": _probe_markdown,
    "marshmallow": _probe_marshmallow,
    "mypy": _probe_mypy,
    "networkx": _probe_networkx,
    "pluggy": _probe_pluggy,
    "pydantic": _probe_pydantic,
    "pydantic-graph": _probe_pydantic_graph,
    "rich": _probe_rich,
    "sortedcontainers": _probe_sortedcontainers,
    "sqlalchemy": _probe_sqlalchemy,
    "sqlglot": _probe_sqlglot,
    "sympy": _probe_sympy,
    "tomli": _probe_tomli,
    "tornado": _probe_tornado,
    "trio": _probe_trio,
    "websockets": _probe_websockets,
}


def main(argv: tuple[str, ...] | None = None) -> int:
    """Run one installed-wheel probe and print its canonical JSON payload."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--case", choices=tuple(PROBES), required=True)
    arguments = parser.parse_args(argv)
    if not arguments.project_root.resolve(strict=True).is_dir():
        parser.error("--project-root must identify a directory")
    canonical, modules = _run_probe(PROBES[arguments.case])
    payload = {"canonical": canonical, "imports": _module_paths(modules)}
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
