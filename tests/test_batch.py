"""Constructors, model-typed arguments and dataclass_batch."""

import asyncio
import gc
import weakref

import pytest
from trame_dataclass.v2 import StateDataModel, Sync

from trame_mcp import MCPStateDataModel, constructor, mcp_field, tool
from trame_mcp.dataclass_tools import BatchError
from trame_mcp.schema import ValidationError

# -----------------------------------------------------------------------------
# A small graph: nodes with ports, links between ports
# -----------------------------------------------------------------------------


class Port(MCPStateDataModel):
    name = mcp_field(Sync(str, ""))


class Node(MCPStateDataModel):
    """A node of the graph."""

    label = mcp_field(Sync(str, ""))
    ports = mcp_field(Sync(list[Port], list, has_dataclass=True))

    @constructor(description="A node with one port, in no graph yet")
    @classmethod
    def new(cls, server, label: str, port: str = "out") -> "Node":
        node = cls(server, label=label)
        node.ports = [Port(server, name=port)]
        return node

    @tool
    def rename(self, label: str) -> str:
        """Rename the node."""
        self.label = label
        return label


class Special(Node):  # inherits the constructor, called with cls=Special
    level = mcp_field(Sync(int, 0))


class Custom(Node):
    @classmethod
    @constructor  # bare, below @classmethod
    def make(cls, server, label: str) -> "Custom":
        """A custom node."""
        return cls(server, label=label.upper())


class Broken(Node):
    @constructor
    @classmethod
    def new(cls, server, label: str) -> "Node":
        return Node(server, label=label)  # not a Broken


class Link(MCPStateDataModel):
    source = mcp_field(Sync(Port, None, has_dataclass=True))
    target = mcp_field(Sync(Port, None, has_dataclass=True))


class Graph(MCPStateDataModel):
    nodes = mcp_field(Sync(list[Node], list, has_dataclass=True))
    links = mcp_field(Sync(list[Link], list, has_dataclass=True))

    @tool
    def add_node(self, node: Node) -> Node:
        """Add a node."""
        if node in self.nodes:
            msg = f"{node.label} is already in the graph"
            raise ValueError(msg)
        self.nodes = [*self.nodes, node]
        return node

    @tool
    def remove_node(self, node: Node) -> None:
        """Remove a node."""
        self.nodes = [n for n in self.nodes if n is not node]

    @tool
    def connect(self, source: Port, target: Port) -> Link:
        """Link two ports."""
        link = Link(self.server, source=source, target=target)
        self.links = [*self.links, link]
        return link

    @tool(read_only=True)
    def count(self, nodes: list[Node] | None = None) -> int:
        """Count the given nodes, or all of them."""
        return len(self.nodes if nodes is None else nodes)

    @tool
    async def wait_for(self, event: str) -> str:
        """Wait until the test releases ``event``."""
        await EVENTS[event].wait()
        return event


EVENTS = {}


@pytest.fixture
def graph(mcp):
    graph = Graph()
    graph.nodes = [Node.new(None, "source")]
    mcp.dataclass.expose_instance(graph, name="graph")
    return graph


def fn(mcp, name):
    return mcp.get_tool(name).fn


def batch(mcp, steps):
    return fn(mcp, "dataclass_batch")(steps)


# -----------------------------------------------------------------------------
# Constructors
# -----------------------------------------------------------------------------


def test_constructors_are_described_without_their_server(mcp, graph):
    models = fn(mcp, "dataclass_schemas")()["models"]
    assert models["Node"]["constructor"] == {
        "name": "new",
        "description": "A node with one port, in no graph yet",
        "parameters": {
            "type": "object",
            "properties": {"label": {"type": "string"}, "port": {"type": "string", "default": "out"}},
            "required": ["label"],
            "additionalProperties": False,
        },
    }
    # inherited: the delta does not repeat it; a subclass's own one is listed
    assert "constructor" not in models["Special"]
    assert models["Custom"]["constructor"]["name"] == "make"
    assert models["Custom"]["constructor"]["description"] == "A custom node."
    assert "constructor" not in models["Graph"]


def test_constructor_marker_validation(mcp):
    with pytest.raises(TypeError, match="not a staticmethod"):

        class Static(MCPStateDataModel):
            @constructor
            @staticmethod
            def new(server):
                pass

    class Twice(MCPStateDataModel):
        label = mcp_field(Sync(str, ""))

        @constructor
        @classmethod
        def a(cls, server):
            pass

        @constructor
        @classmethod
        def b(cls, server):
            pass

    with pytest.raises(ValueError, match="several constructors"):
        mcp.dataclass._policy(Twice)

    class Plain(MCPStateDataModel):
        label = mcp_field(Sync(str, ""))

        @constructor
        def new(self, server):
            pass

    with pytest.raises(ValueError, match="not a classmethod"):
        mcp.dataclass._policy(Plain)

    class NoServer(MCPStateDataModel):
        label = mcp_field(Sync(str, ""))

        @constructor
        @classmethod
        def new(cls):
            pass

    with pytest.raises(ValueError, match="trame server as its first parameter"):
        mcp.dataclass._policy(NoServer)


def test_expose_model_names_or_hides_a_constructor(mcp):
    class Widget(StateDataModel):
        label = Sync(str, "")

        @classmethod
        def create(cls, server, label: str):
            """Make a widget."""
            return cls(server, label=label)

    mcp.dataclass.expose_model(Widget, all_fields=True, constructor="create")
    ctor = mcp.dataclass._policy(Widget).constructor
    assert (ctor.attr, ctor.description) == ("create", "Make a widget.")

    class Hidden(Node):
        pass

    mcp.dataclass.expose_model(Hidden, constructor=False)
    assert mcp.dataclass._policy(Hidden).constructor is None

    class Described(Node):
        pass

    mcp.dataclass.expose_model(Described, constructor={"name": "new", "description": "Override"})
    assert mcp.dataclass._policy(Described).constructor.description == "Override"

    with pytest.raises(ValueError, match="not a classmethod"):
        mcp.dataclass.expose_model(Port, constructor="name")
    with pytest.raises(ValueError, match="unknown options"):
        mcp.dataclass.expose_model(Link, constructor={"name": "x", "bogus": 1})


# -----------------------------------------------------------------------------
# Model-typed arguments
# -----------------------------------------------------------------------------


def test_model_parameters_take_ids(mcp, graph):
    methods = fn(mcp, "dataclass_schemas")()["models"]["Graph"]["methods"]
    node_schema = {"type": "string", "model": "Node", "description": "Id of a Node instance"}
    assert methods["add_node"]["parameters"]["properties"]["node"] == node_schema
    assert methods["count"]["parameters"]["properties"]["nodes"]["anyOf"] == [
        {"type": "array", "items": node_schema},
        {"type": "null"},
    ]


async def test_call_resolves_ids_to_instances(mcp, graph):
    call = fn(mcp, "dataclass_call")
    source = graph.nodes[0]
    assert await call("graph", "count", {"nodes": [source._id, source._id]}) == 2

    with pytest.raises(ValidationError, match=r"args\.node: '\w+' is a Port, not a Node"):
        await call("graph", "remove_node", {"node": source.ports[0]._id})
    with pytest.raises(ValidationError, match="is a Graph, not a Node"):
        await call("graph", "remove_node", {"node": "graph"})  # an entry name
    with pytest.raises(ValidationError, match="not the id of a reachable instance"):
        await call("graph", "remove_node", {"node": "nope"})
    stray = Node.new(None, "stray")  # exists, but nothing exposed reaches it
    with pytest.raises(ValidationError, match="not the id of a reachable instance"):
        await call("graph", "remove_node", {"node": stray._id})
    assert graph.nodes == [source]

    await call("graph", "remove_node", {"node": source._id})
    assert graph.nodes == []


async def test_read_only_instances_go_to_read_only_methods_only(mcp, graph):
    archive = Graph()
    archive.nodes = [Node.new(None, "archived")]
    mcp.dataclass.expose_instance(archive, name="archive", read_only=True)
    archived = archive.nodes[0]._id
    call = fn(mcp, "dataclass_call")

    assert await call("graph", "count", {"nodes": [archived]}) == 1
    with pytest.raises(PermissionError, match="exposed read-only"):
        await call("graph", "add_node", {"node": archived})


async def test_plain_tools_resolve_model_parameters(mcp, graph):
    @mcp.tool
    def describe(node: Node) -> str:
        return node.label

    source = graph.nodes[0]
    assert mcp.get_tool("describe").input_schema["properties"]["node"]["model"] == "Node"
    response = await mcp.protocol.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "describe", "arguments": {"node": source._id}},
        }
    )
    assert response["result"]["content"][0]["text"] == "source"
    response = await mcp.protocol.handle(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "describe", "arguments": {"node": "nope"}},
        }
    )
    assert response["result"]["isError"]
    assert "not the id of a reachable instance" in response["result"]["content"][0]["text"]


async def test_returned_instances_are_encoded(mcp, graph):
    source, = graph.nodes
    target = Node.new(None, "target")
    graph.nodes = [source, target]
    link = await fn(mcp, "dataclass_call")(
        "graph", "connect", {"source": source.ports[0]._id, "target": target.ports[0]._id}
    )
    assert link["type"] == "Link"
    assert link["values"]["source"] == {"id": source.ports[0]._id, "type": "Port"}


# -----------------------------------------------------------------------------
# dataclass_batch
# -----------------------------------------------------------------------------


async def test_batch_builds_attaches_and_links(mcp, graph):
    result = await batch(
        mcp,
        [
            {"new": "Node", "args": {"label": "blur"}, "as": "blur"},
            {"call": "graph", "method": "add_node", "args": {"node": "@blur"}},
            {"new": "Special", "args": {"label": "view", "port": "in"}, "as": "view"},
            {"update": "@view", "values": {"level": 3}},  # built, not attached yet: writable
            {"call": "graph", "method": "add_node", "args": {"node": "@view"}},
            {
                "call": "graph",
                "method": "connect",
                "args": {"source": "@blur.ports.0", "target": "@view.ports.0"},
                "as": "link",
            },
            {"call": "@blur", "method": "rename", "args": {"label": "@@blur"}},
        ],
    )
    _, blur, view = graph.nodes
    assert (blur.label, view.label, type(view), view.level) == ("@blur", "view", Special, 3)
    (link,) = graph.links
    assert (link.source, link.target) == (blur.ports[0], view.ports[0])

    results = result["results"]
    assert results[0]["id"] == blur._id
    assert results[0]["values"]["ports"] == [{"id": blur.ports[0]._id, "type": "Port"}]
    assert results[3] == {"id": view._id, "updated": ["level"]}
    assert results[5]["type"] == "Link"
    assert results[6] == "@blur"
    assert result["unattached"] == []


async def test_batch_results_show_links_once(mcp, graph):
    steps = [
        {"new": "Node", "args": {"label": "blur"}, "as": "blur"},
        {"call": "graph", "method": "add_node", "args": {"node": "@blur"}},
    ]
    built, added = (await batch(mcp, steps))["results"]
    _, blur = graph.nodes
    # Its references to other instances, not its other fields
    assert built == {
        "id": blur._id,
        "type": "Node",
        "values": {"ports": [{"id": blur.ports[0]._id, "type": "Port"}]},
    }
    assert added == {"id": blur._id, "type": "Node"}  # shown already

    steps[0]["args"]["label"] = "sharpen"
    built, added = (
        await fn(mcp, "dataclass_batch")(steps, results="full")
    )["results"]
    assert built["values"]["label"] == "sharpen"
    assert added["values"]["label"] == "sharpen"


async def test_batch_follows_keys_of_returned_objects(mcp, graph):
    @mcp.tool
    def unused():
        pass

    result = await batch(
        mcp,
        [
            {"new": "Custom", "args": {"label": "c"}, "as": "c"},
            {"call": "graph", "method": "add_node", "args": {"node": "@c"}, "as": "added"},
            {"call": "@added", "method": "rename", "args": {"label": "@c.label"}},
        ],
    )
    assert graph.nodes[1].label == "C"
    assert result["results"][2] == "C"


async def test_batch_drops_what_it_never_attaches(mcp, graph, monkeypatch):
    from trame_mcp import dataclass_tools

    scopes = []

    class Recorded(dataclass_tools._Scope):
        def __init__(self):
            super().__init__()
            scopes.append(weakref.ref(self))

    monkeypatch.setattr(dataclass_tools, "_Scope", Recorded)
    result = await batch(mcp, [{"new": "Node", "args": {"label": "lost"}, "as": "lost"}])
    (lost,) = result["unattached"]
    assert lost["type"] == "Node"
    assert len(graph.nodes) == 1
    # What held the built instance is gone: trame-mcp keeps no reference
    gc.collect()
    assert len(scopes) == 1
    assert scopes[0]() is None


@pytest.mark.parametrize(
    ("steps", "error"),
    [
        ("nope", "must be an array"),
        (["nope"], r"steps\[2\]: expected an object"),
        ([{"new": "Node", "call": "graph"}], "exactly one of"),
        ([{"call": "graph", "method": "count", "bogus": 1}], r"unexpected \['bogus'\]"),
        ([{"call": "graph"}], "needs 'method'"),
        ([{"update": "graph"}], "needs 'values'"),
        ([{"call": "graph", "method": "count", "args": []}], r"steps\[2\]\.args: expected an object"),
        ([{"new": "Graph"}], "no model 'Graph' has a constructor"),
        ([{"new": "Nope"}], "no model 'Nope'"),
        ([{"call": "@x", "method": "count"}], "'@x' names no earlier step"),
        (
            [
                {"new": "Node", "args": {"label": "a"}, "as": "a"},
                {"new": "Node", "args": {"label": "b"}, "as": "a"},
            ],
            "'a' already names an earlier step",
        ),
        ([{"new": "Node", "args": {"label": "a"}, "as": "not a name"}], "letters, digits"),
        (
            [{"new": "Node", "args": {"label": "@a"}, "as": "a"}],  # not its own result
            "'@a' names no earlier step",
        ),
    ],
)
async def test_batch_steps_are_checked_before_any_runs(mcp, graph, steps, error):
    if isinstance(steps, list):  # a valid step first: it must not run either
        steps = [
            {"new": "Node", "args": {"label": "ok"}, "as": "first"},
            {"call": "graph", "method": "add_node", "args": {"node": "@first"}},
            *steps,
        ]
    with pytest.raises(ValidationError, match=error) as info:
        await batch(mcp, steps)
    assert str(info.value).startswith("Nothing was applied")
    assert len(graph.nodes) == 1


async def test_batch_stops_at_the_first_failing_step(mcp, graph):
    with pytest.raises(BatchError) as info:
        await batch(
            mcp,
            [
                {"new": "Node", "args": {"label": "a"}, "as": "a"},
                {"call": "graph", "method": "add_node", "args": {"node": "@a"}},
                {"call": "graph", "method": "add_node", "args": {"node": "@a"}},
                {"new": "Node", "args": {"label": "never"}, "as": "never"},
            ],
        )
    message = str(info.value)
    assert message.startswith(
        "steps[2] (call add_node on graph) failed: ValueError: a is already in the graph."
    )
    assert "Steps 0-1 stay applied (no rollback)" in message
    assert [n.label for n in graph.nodes] == ["source", "a"]


@pytest.mark.parametrize(
    ("step", "error"),
    [
        ({"call": "@a.missing", "method": "rename"}, "@a is a Node, which has no field 'missing'"),
        ({"call": "@a.ports.5", "method": "rename"}, "@a.ports has 1 items, no index '5'"),
        ({"call": "@a.label.x", "method": "rename"}, "@a.label is str, which has no 'x'"),
        ({"call": "nope", "method": "rename"}, "Unknown target 'nope'"),
        ({"call": "@a", "method": "nope"}, "Unknown method 'nope' on Node"),
        ({"call": "@a", "method": "rename", "args": {"label": 3}}, r"label: expected string"),
        ({"update": "@a", "values": {"bogus": 1}}, "bogus \\(unknown field\\)"),
        ({"new": "Node", "args": {}}, "missing required 'label'"),
        ({"new": "Broken", "args": {"label": "b"}}, "returned Node, not a Broken"),
    ],
)
async def test_batch_reports_the_failing_step(mcp, graph, step, error):
    with pytest.raises(BatchError, match=error) as info:
        await batch(mcp, [{"new": "Node", "args": {"label": "a"}, "as": "a"}, step])
    assert str(info.value).startswith("steps[1] (")


async def test_batch_failure_on_the_first_step_applied_nothing(mcp, graph):
    with pytest.raises(BatchError, match=r"failed: .*\. Nothing was applied\.$"):
        await batch(mcp, [{"call": "graph", "method": "remove_node", "args": {"node": "nope"}}])


def rpc(rpc_id, name, arguments):
    return {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }


async def test_batch_holds_the_lock_once_and_flushes_once(mcp, graph):
    state = mcp.server.state
    state.labels = []
    state.ready()
    seen = []

    @state.change("labels")
    def on_labels(labels, **_):
        seen.append(list(labels))

    class Tracked(Node):
        @tool
        def publish(self) -> None:
            """Append the label to the shared state."""
            state.labels = [*state.labels, self.label]

    mcp.dataclass.expose_model(Tracked)
    EVENTS["batch"] = asyncio.Event()
    running = asyncio.ensure_future(
        mcp.protocol.handle(
            rpc(
                1,
                "dataclass_batch",
                {
                    "steps": [
                        {"new": "Tracked", "args": {"label": "a"}, "as": "a"},
                        {"call": "graph", "method": "add_node", "args": {"node": "@a"}},
                        {"call": "@a", "method": "publish"},
                        {"call": "graph", "method": "wait_for", "args": {"event": "batch"}},
                        {"call": "@a", "method": "rename", "args": {"label": "b"}},
                        {"call": "@a", "method": "publish"},
                    ]
                },
            )
        )
    )
    await asyncio.sleep(0.01)
    # A serialized call waits for the whole batch
    other = asyncio.ensure_future(
        mcp.protocol.handle(
            rpc(2, "dataclass_call", {"target": "graph", "method": "remove_node", "args": {"node": graph.nodes[0]._id}})
        )
    )
    await asyncio.sleep(0.01)
    assert not other.done()
    assert len(graph.nodes) == 2

    EVENTS["batch"].set()
    response = await running
    assert not response["result"].get("isError"), response
    await other
    assert seen == [["a", "b"]]  # one flush, at the end
    assert [n.label for n in graph.nodes] == ["b"]


async def test_a_cancel_stops_a_batch_between_steps(mcp, graph):
    from trame_mcp.protocol import REQUEST_CANCELLED

    EVENTS["cancel"] = asyncio.Event()
    running = asyncio.ensure_future(
        mcp.protocol.handle(
            rpc(
                7,
                "dataclass_batch",
                {
                    "steps": [
                        {"call": "graph", "method": "wait_for", "args": {"event": "cancel"}},
                        {"new": "Node", "args": {"label": "late"}, "as": "late"},
                        {"call": "graph", "method": "add_node", "args": {"node": "@late"}},
                    ]
                },
            )
        )
    )
    await asyncio.sleep(0.01)
    assert mcp.protocol.cancel(7, reason="changed my mind")
    response = await running
    assert response["error"]["code"] == REQUEST_CANCELLED
    assert [n.label for n in graph.nodes] == ["source"]


async def test_batch_is_reported_through_the_protocol(mcp, graph):
    response = await mcp.protocol.handle(
        rpc(
            3,
            "dataclass_batch",
            {
                "steps": [
                    {"new": "Node", "args": {"label": "a"}, "as": "a"},
                    {"call": "graph", "method": "add_node", "args": {"node": "@a"}},
                    {"call": "graph", "method": "add_node", "args": {"node": "@a"}},
                ]
            },
        )
    )
    result = response["result"]
    assert result["isError"]
    text = result["content"][0]["text"]
    assert text.startswith("steps[2] (call add_node on graph) failed: ValueError: a is already")
    assert "BatchError" not in text
