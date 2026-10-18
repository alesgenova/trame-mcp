import asyncio
from typing import Annotated

import pytest


def rpc(method, params=None, rpc_id=1):
    msg = {"jsonrpc": "2.0", "id": rpc_id, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


@pytest.fixture
def protocol(mcp):
    @mcp.tool
    def add(a: int, b: int = 1) -> int:
        """Add two numbers."""
        return a + b

    @mcp.tool
    async def async_echo(text: str) -> str:
        return text

    @mcp.tool
    def failing():
        msg = "boom"
        raise RuntimeError(msg)

    @mcp.resource("trame://info", mime_type="application/json")
    def info():
        return {"name": "test"}

    @mcp.prompt
    def walkthrough(topic: Annotated[str, "Part of the app to explain"] = "app") -> str:
        return f"Explain the {topic}"

    return mcp.protocol


async def test_initialize(protocol):
    response = await protocol.handle(
        rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}})
    )
    result = response["result"]
    assert result["protocolVersion"] == "2025-06-18"
    assert "tools" in result["capabilities"]
    assert result["serverInfo"]["name"]


async def test_initialize_sends_app_then_generated_instructions(mcp):
    from trame_dataclass.v2 import Sync

    from trame_mcp import MCPStateDataModel, mcp_field

    class Book(MCPStateDataModel):
        title = mcp_field(Sync(str, ""))

    async def instructions():
        response = await mcp.protocol.handle(rpc("initialize"))
        return response["result"]["instructions"]

    # nothing exposed: only the live-app notice
    text = await instructions()
    assert text.startswith("This MCP server belongs to a running trame application")
    assert "\n-" not in text

    mcp.instructions = "A library catalog.\n"
    mcp.dataclass.expose_instance(Book(), name="book")
    mcp.dataclass.expose_instance(lambda: [Book()], name="shelf", read_only=True)
    mcp.state.expose("query", read_only=True)
    lines = (await instructions()).split("\n")
    assert lines[0] == "A library catalog."
    assert lines[2].startswith("This MCP server belongs to a running trame application")
    assert lines[3] == (
        "- Application data: call dataclass_schemas first, then dataclass_get on an "
        "entry point: `book` (Book), `shelf` (collection, read-only)."
    )
    assert lines[4] == "- Shared state: read it with state_get."


async def test_initialize_unknown_version_falls_back(protocol):
    response = await protocol.handle(
        rpc("initialize", {"protocolVersion": "1999-01-01"})
    )
    assert response["result"]["protocolVersion"] == "2025-06-18"


async def test_notifications_get_no_response(protocol):
    msg = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    assert await protocol.handle(msg) is None


async def test_tools_list(protocol, mcp):
    @mcp.tool(read_only=True)
    def peek() -> str:
        return "ok"

    response = await protocol.handle(rpc("tools/list"))
    tools = {t["name"]: t for t in response["result"]["tools"]}
    assert tools["add"]["description"] == "Add two numbers."
    assert tools["add"]["inputSchema"]["required"] == ["a"]
    assert tools["add"]["inputSchema"]["additionalProperties"] is False
    assert "annotations" not in tools["add"]
    assert tools["peek"]["annotations"] == {"readOnlyHint": True}


async def test_tools_call_sync_and_async(protocol):
    response = await protocol.handle(
        rpc("tools/call", {"name": "add", "arguments": {"a": 2, "b": 3}})
    )
    assert response["result"]["content"] == [{"type": "text", "text": "5"}]

    response = await protocol.handle(
        rpc("tools/call", {"name": "async_echo", "arguments": {"text": "hi"}})
    )
    assert response["result"]["content"] == [{"type": "text", "text": "hi"}]


async def test_tools_call_error_goes_in_result(protocol):
    response = await protocol.handle(rpc("tools/call", {"name": "failing"}))
    result = response["result"]
    assert result["isError"] is True
    assert "boom" in result["content"][0]["text"]


async def test_tools_call_bad_arguments(protocol):
    async def call(arguments):
        response = await protocol.handle(
            rpc("tools/call", {"name": "add", "arguments": arguments})
        )
        result = response["result"]
        assert result["isError"] is True
        return result["content"][0]["text"]

    assert "unexpected 'wrong'" in await call({"a": 1, "wrong": 1})
    assert "missing required 'a'" in await call({})
    text = await call({"a": "1"})
    assert text == "Invalid arguments for tool add: arguments.a: expected integer, got string"


async def test_tools_call_type_error_in_body_is_not_blamed_on_arguments(mcp):
    @mcp.tool
    def measure(x: int) -> int:
        return len(x)

    response = await mcp.protocol.handle(
        rpc("tools/call", {"name": "measure", "arguments": {"x": 3}})
    )
    text = response["result"]["content"][0]["text"]
    assert text.startswith("TypeError:")
    assert "Invalid arguments" not in text


async def test_tools_call_coerces_lossless_numbers(mcp):
    @mcp.tool
    def kinds(i: int, f: float) -> dict:
        return {"i": type(i).__name__, "f": type(f).__name__}

    response = await mcp.protocol.handle(
        rpc("tools/call", {"name": "kinds", "arguments": {"i": 2.0, "f": 2}})
    )
    assert response["result"]["structuredContent"] == {"i": "int", "f": "float"}


async def test_tools_call_unknown_tool(protocol):
    response = await protocol.handle(rpc("tools/call", {"name": "nope"}))
    assert response["error"]["code"] == -32602


async def test_resources(protocol):
    response = await protocol.handle(rpc("resources/list"))
    resources = response["result"]["resources"]
    assert resources[0]["uri"] == "trame://info"

    response = await protocol.handle(rpc("resources/read", {"uri": "trame://info"}))
    content = response["result"]["contents"][0]
    assert content["mimeType"] == "application/json"
    assert '"name"' in content["text"]


async def test_prompts(protocol):
    response = await protocol.handle(rpc("prompts/list"))
    prompts = response["result"]["prompts"]
    assert prompts[0]["name"] == "walkthrough"
    assert prompts[0]["arguments"] == [
        {"name": "topic", "required": False, "description": "Part of the app to explain"}
    ]

    response = await protocol.handle(
        rpc("prompts/get", {"name": "walkthrough", "arguments": {"topic": "cone"}})
    )
    message = response["result"]["messages"][0]
    assert message["role"] == "user"
    assert message["content"]["text"] == "Explain the cone"


async def test_disabled_blocks_tools_resources_and_prompts(protocol, mcp):
    from trame_mcp import MCPDisabledError

    mcp.enabled = False

    response = await protocol.handle(rpc("tools/call", {"name": "add", "arguments": {"a": 1}}))
    assert response["result"]["isError"] is True
    assert "disabled" in response["result"]["content"][0]["text"]
    with pytest.raises(MCPDisabledError):
        await protocol.execute(mcp.get_tool("add"), {"a": 1})

    response = await protocol.handle(rpc("resources/read", {"uri": "trame://info"}))
    assert "disabled" in response["error"]["message"]
    response = await protocol.handle(rpc("prompts/get", {"name": "walkthrough"}))
    assert "disabled" in response["error"]["message"]

    mcp.enabled = True
    assert await protocol.execute(mcp.get_tool("add"), {"a": 1}) == 2


async def test_malformed_requests(protocol):
    response = await protocol.handle(rpc("tools/list", params=["not", "a", "dict"]))
    assert response["error"]["code"] == -32602
    response = await protocol.handle({"jsonrpc": "2.0", "id": 1, "method": 42})
    assert response["error"]["code"] == -32600
    response = await protocol.handle(
        rpc("prompts/get", {"name": "walkthrough", "arguments": {"bogus": "x"}})
    )
    assert response["error"]["code"] == -32602


async def test_resource_templates_list(protocol):
    response = await protocol.handle(rpc("resources/templates/list"))
    assert response["result"] == {"resourceTemplates": []}


async def test_unknown_method(protocol):
    response = await protocol.handle(rpc("bogus/method"))
    assert response["error"]["code"] == -32601


async def test_ping(protocol):
    response = await protocol.handle(rpc("ping"))
    assert response["result"] == {}


async def test_parallel_async_tool_calls_are_serialized(mcp):
    import asyncio

    log = []

    @mcp.tool
    async def slow(tag: str) -> str:
        log.append(f"start:{tag}")
        await asyncio.sleep(0.01)
        log.append(f"end:{tag}")
        return tag

    await asyncio.gather(
        mcp.protocol.handle(
            rpc("tools/call", {"name": "slow", "arguments": {"tag": "a"}})
        ),
        mcp.protocol.handle(
            rpc("tools/call", {"name": "slow", "arguments": {"tag": "b"}})
        ),
    )

    # With serialization, each start is immediately followed by its own end
    assert len(log) == 4
    for i in (0, 2):
        tag = log[i].split(":")[1]
        assert log[i] == f"start:{tag}"
        assert log[i + 1] == f"end:{tag}"


async def test_tool_state_changes_are_flushed(mcp):
    state = mcp.server.state
    state.resolution = 6
    state.ready()  # what starting the server does: flushes run from now on
    seen = []

    @state.change("resolution")
    def on_resolution(resolution, **_):
        seen.append(resolution)

    @mcp.tool
    def set_resolution(resolution: int) -> int:
        state.resolution = resolution  # no `with state:`
        return resolution

    @mcp.tool
    async def set_later(resolution: int) -> int:
        await asyncio.sleep(0)
        state.resolution = resolution
        return resolution

    await mcp.protocol.handle(
        rpc("tools/call", {"name": "set_resolution", "arguments": {"resolution": 12}})
    )
    assert seen == [12]

    await mcp.protocol.handle(
        rpc("tools/call", {"name": "set_later", "arguments": {"resolution": 18}})
    )
    assert seen == [12, 18]


def call_tool(mcp, name, **arguments):
    return mcp.protocol.handle(rpc("tools/call", {"name": name, "arguments": arguments}))


async def test_unserialized_tools_run_while_a_call_holds_the_lock(mcp):
    release = asyncio.Event()
    log = []

    @mcp.tool
    async def hold() -> str:
        log.append("hold:start")
        await release.wait()
        log.append("hold:end")
        return "held"

    @mcp.tool(read_only=True, serialize=False)
    async def peek() -> str:
        log.append("peek")
        return "peeked"

    @mcp.tool
    def poke() -> str:
        log.append("poke")
        return "poked"

    holding = asyncio.ensure_future(call_tool(mcp, "hold"))
    await asyncio.sleep(0)  # hold takes the lock

    # The unserialized tool is not held back...
    response = await asyncio.wait_for(call_tool(mcp, "peek"), 1)
    assert response["result"]["content"][0]["text"] == "peeked"
    # ... a serialized one waits for the lock
    poking = asyncio.ensure_future(call_tool(mcp, "poke"))
    await asyncio.sleep(0.01)
    assert not poking.done()

    release.set()
    await asyncio.gather(holding, poking)
    assert log == ["hold:start", "peek", "hold:end", "poke"]


async def test_marked_methods_carry_serialize(mcp):
    from trame_mcp import tool

    class Jobs:
        @tool(read_only=True, serialize=False)
        async def wait_for_jobs(self) -> str:
            return "idle"

        @tool
        def start_job(self) -> str:
            return "started"

    mcp.expose(Jobs())
    assert mcp.get_tool("wait_for_jobs").serialize is False
    assert mcp.get_tool("start_job").serialize is True


def cancel_notification(request_id, reason=None):
    params = {"requestId": request_id}
    if reason is not None:
        params["reason"] = reason
    return {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": params}


async def test_cancelled_tool_call_stops_releases_the_lock_and_flushes(mcp):
    from trame_mcp.protocol import REQUEST_CANCELLED

    state = mcp.server.state
    state.progress = 0
    state.ready()
    seen, log = [], []

    @state.change("progress")
    def on_progress(progress, **_):
        seen.append(progress)

    @mcp.tool
    async def long_job() -> str:
        state.progress = 50
        try:
            await asyncio.Event().wait()  # never set
        finally:
            log.append("cleaned up")
        return "done"

    @mcp.tool
    def quick() -> str:
        return "quick"

    job = asyncio.ensure_future(
        mcp.protocol.handle(rpc("tools/call", {"name": "long_job"}, rpc_id=7))
    )
    await asyncio.sleep(0.01)
    assert not job.done()

    assert await mcp.protocol.handle(cancel_notification(7, "user stopped")) is None
    response = await asyncio.wait_for(job, 1)
    assert response["id"] == 7
    assert response["error"]["code"] == REQUEST_CANCELLED
    assert "user stopped" in response["error"]["message"]
    assert log == ["cleaned up"]
    assert seen == [50]  # what it changed before the cancel was flushed

    # The lock is free again
    response = await asyncio.wait_for(call_tool(mcp, "quick"), 1)
    assert response["result"]["content"][0]["text"] == "quick"
    assert not mcp.protocol._in_flight


async def test_cancel_reaches_only_the_requests_of_its_session(mcp):
    release = asyncio.Event()

    @mcp.tool(serialize=False)
    async def wait_for_release() -> str:
        await release.wait()
        return "released"

    message = rpc("tools/call", {"name": "wait_for_release"}, rpc_id=1)
    first = asyncio.ensure_future(mcp.protocol.handle(message, session="a"))
    second = asyncio.ensure_future(mcp.protocol.handle(message, session="b"))
    await asyncio.sleep(0.01)

    await mcp.protocol.handle(cancel_notification(1), session="a")
    assert "error" in await asyncio.wait_for(first, 1)
    assert not second.done()

    # Unknown, finished and malformed requests are ignored
    await mcp.protocol.handle(cancel_notification(1), session="a")
    await mcp.protocol.handle(cancel_notification(99), session="b")
    await mcp.protocol.handle(cancel_notification([1]), session="b")
    await mcp.protocol.handle(
        {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": "x"}
    )
    assert not second.done()

    release.set()
    response = await asyncio.wait_for(second, 1)
    assert response["result"]["content"][0]["text"] == "released"


async def test_a_call_cancelled_while_waiting_for_the_lock_never_runs(mcp):
    release = asyncio.Event()
    log = []

    @mcp.tool
    async def hold() -> str:
        await release.wait()
        return "held"

    @mcp.tool
    def poke() -> str:
        log.append("poke")
        return "poked"

    holding = asyncio.ensure_future(
        mcp.protocol.handle(rpc("tools/call", {"name": "hold"}, rpc_id=1))
    )
    poking = asyncio.ensure_future(
        mcp.protocol.handle(rpc("tools/call", {"name": "poke"}, rpc_id=2))
    )
    await asyncio.sleep(0.01)
    await mcp.protocol.handle(cancel_notification(2))
    assert "error" in await asyncio.wait_for(poking, 1)

    release.set()
    assert "result" in await asyncio.wait_for(holding, 1)
    assert log == []
    response = await asyncio.wait_for(call_tool(mcp, "poke"), 1)
    assert response["result"]["content"][0]["text"] == "poked"
