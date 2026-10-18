"""End-to-end: real trame server, MCP over HTTP."""

import itertools

import aiohttp
import pytest

from trame_mcp import get_mcp
from trame_mcp.endpoint import is_loopback

_counter = itertools.count()


def rpc(method, params=None, rpc_id=1):
    msg = {"jsonrpc": "2.0", "id": rpc_id, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


async def start(server, **kwargs):
    server.start(
        exec_mode="task",
        port=0,
        open_browser=False,
        show_connection_info=False,
        disable_logging=True,
        **kwargs,
    )
    await server.ready


@pytest.fixture
async def running_server():
    from trame.app import get_server

    server = get_server(f"trame_mcp_test_http_{next(_counter)}")
    mcp = get_mcp(server, enabled=True)

    server.state.resolution = 6
    mcp.state.expose("resolution", type=int)

    @mcp.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    await start(server)
    yield server, mcp
    await server.stop()


async def post(session, server, message=None, **kwargs):
    url = f"http://localhost:{server.port}/mcp"
    kwargs.setdefault("json", message or rpc("ping"))
    async with session.post(url, **kwargs) as resp:
        body = await resp.json(content_type=None) if resp.status != 202 else None
        return resp.status, body


def test_is_loopback():
    for host in ("localhost", "LOCALHOST", "app.localhost", "127.0.0.1", "127.8.0.1", "::1", "[::1]"):
        assert is_loopback(host), host
    for host in (None, "", "0.0.0.0", "::", "192.168.1.5", "example.com", "localhost.evil.com"):
        assert not is_loopback(host), host


async def test_http_roundtrip(running_server):
    server, mcp = running_server
    url = f"http://localhost:{server.port}/mcp"
    assert mcp.url == url

    async with aiohttp.ClientSession() as session:
        # initialize
        status, body = await post(
            session, server, rpc("initialize", {"protocolVersion": "2025-06-18"})
        )
        assert status == 200
        assert body["result"]["protocolVersion"] == "2025-06-18"

        # initialized notification -> 202, empty
        status, _ = await post(
            session, server, {"jsonrpc": "2.0", "method": "notifications/initialized"}
        )
        assert status == 202

        # tools/list
        _, body = await post(session, server, rpc("tools/list"))
        assert {"add", "state_get", "state_update"} <= {t["name"] for t in body["result"]["tools"]}

        # tools/call -> touches real trame state
        _, body = await post(
            session,
            server,
            rpc("tools/call", {"name": "state_update", "arguments": {"updates": {"resolution": 24}}}),
        )
        assert body["result"].get("isError") is None
        assert server.state.resolution == 24

        # schema-checked: a string never reaches the state
        _, body = await post(
            session,
            server,
            rpc("tools/call", {"name": "state_update", "arguments": {"updates": {"resolution": "x"}}}),
        )
        assert body["result"]["isError"] is True
        assert server.state.resolution == 24

        # GET is not supported (stateless server)
        async with session.get(url) as resp:
            assert resp.status == 405

        # batch rejected
        status, _ = await post(session, server, [rpc("ping")])
        assert status == 400

        # malformed json
        status, _ = await post(
            session, server, json=None, data=b"{not json", headers={"Content-Type": "application/json"}
        )
        assert status == 400


async def test_cross_site_requests_are_rejected(running_server):
    server, mcp = running_server
    attack = rpc(
        "tools/call", {"name": "state_update", "arguments": {"updates": {"resolution": 99}}}
    )

    async with aiohttp.ClientSession() as session:
        # A page on another site: browsers attach its Origin
        status, _ = await post(session, server, attack, headers={"Origin": "https://evil.example"})
        assert status == 403
        status, _ = await post(session, server, attack, headers={"Origin": "null"})
        assert status == 403

        # "Simple" request a browser sends without CORS preflight
        status, _ = await post(
            session, server, json=None, data=b"{}", headers={"Content-Type": "text/plain"}
        )
        assert status == 415

        # DNS rebinding: attacker domain resolved to 127.0.0.1, same-origin to the browser
        rebound = f"evil.example:{server.port}"
        status, body = await post(
            session, server, attack, headers={"Host": rebound, "Origin": f"http://{rebound}"}
        )
        assert status == 403
        assert "set a token" in body["error"]["message"]

        assert server.state.resolution == 6

        # Local browser tools (e.g. MCP Inspector) and listed origins are fine
        status, _ = await post(session, server, headers={"Origin": "http://localhost:6274"})
        assert status == 200
        mcp.allowed_origins = ["https://trusted.example/"]
        status, _ = await post(session, server, headers={"Origin": "https://trusted.example"})
        assert status == 200


async def test_token_auth(running_server):
    server, mcp = running_server
    mcp.token = "s3cret"

    async with aiohttp.ClientSession() as session:
        status, _ = await post(session, server)
        assert status == 401
        status, _ = await post(session, server, headers={"Authorization": "Bearer nope"})
        assert status == 401
        status, _ = await post(session, server, headers={"Authorization": "bearer s3cret"})
        assert status == 200

        # with a token, the host name is no longer restricted to loopback
        status, _ = await post(
            session,
            server,
            headers={"Authorization": "Bearer s3cret", "Host": f"myhost.lan:{server.port}"},
        )
        assert status == 200

        mcp.token = None
        status, _ = await post(session, server)
        assert status == 200


async def test_runtime_enable_switch(running_server):
    server, mcp = running_server

    async with aiohttp.ClientSession() as session:
        mcp.enabled = False
        status, body = await post(session, server)
        assert status == 503
        assert "disabled" in body["error"]["message"]

        mcp.enabled = True
        status, _ = await post(session, server)
        assert status == 200


async def test_off_until_enabled_then_announced(capsys):
    """Declared but off: 503 and no banner; once running, the switch applies
    and the banner prints when access goes live."""
    from trame.app import get_server

    server = get_server("trame_mcp_test_http_late_enable")
    mcp = get_mcp(server)

    @mcp.tool
    def hello() -> str:
        """Say hi."""
        return "hi"

    await start(server)
    try:
        assert mcp.enabled is False
        assert "MCP server available" not in capsys.readouterr().out
        async with aiohttp.ClientSession() as session:
            status, _ = await post(session, server)
            assert status == 503

            mcp.enabled = True
            assert f"MCP server available at {mcp.url}" in capsys.readouterr().out
            status, _ = await post(session, server)
            assert status == 200

            mcp.enabled = False
            mcp.enabled = True
            assert "MCP server available" not in capsys.readouterr().out  # once
    finally:
        await server.stop()


async def test_no_mcp_keeps_the_endpoint_off_for_the_run(monkeypatch):
    from trame.app import get_server

    monkeypatch.setenv("TRAME_ARGS", "--no-mcp")
    server = get_server("trame_mcp_test_http_no_mcp")
    mcp = get_mcp(server, enabled=True)
    assert mcp.enabled_locked is True

    await start(server)
    try:
        mcp.enabled = True  # a UI switch, once running: still off
        assert mcp.enabled is False
        async with aiohttp.ClientSession() as session:
            status, _ = await post(session, server)
            assert status == 503
    finally:
        await server.stop()


async def test_library_component_does_not_turn_mcp_on(capsys):
    """A plain TrameApp using an MCPComponent serves nothing by default."""
    from trame.app import TrameApp

    from trame_mcp import MCPComponent, tool

    class Viewer(MCPComponent):
        @tool
        def reset_camera(self) -> str:
            """Reset the camera."""
            return "reset"

    class App(TrameApp):
        def __init__(self):
            super().__init__("trame_mcp_test_http_library_component")
            self.viewer = Viewer(self.server)

    server = App().server
    await start(server)
    try:
        assert "MCP server available" not in capsys.readouterr().out
        async with aiohttp.ClientSession() as session:
            status, _ = await post(session, server, rpc("tools/list"))
            assert status == 503
    finally:
        await server.stop()


async def test_path_and_http_fixed_once_mounted(running_server):
    _, mcp = running_server
    with pytest.raises(RuntimeError, match="started"):
        mcp.path = "/other"
    with pytest.raises(RuntimeError, match="started"):
        mcp.http = False


async def test_token_generated_when_listening_beyond_localhost(capsys):
    from trame.app import get_server

    server = get_server("trame_mcp_test_http_public")
    mcp = get_mcp(server, enabled=True)
    assert mcp.token is None

    await start(server, host="0.0.0.0")
    try:
        assert mcp.token
        assert mcp.url == f"http://localhost:{server.port}/mcp"
        assert f"Bearer {mcp.token}" in capsys.readouterr().out

        async with aiohttp.ClientSession() as session:
            status, _ = await post(session, server)
            assert status == 401
            status, _ = await post(
                session, server, headers={"Authorization": f"Bearer {mcp.token}"}
            )
            assert status == 200
    finally:
        await server.stop()


async def test_http_disabled():
    """http=False: no /mcp route is mounted, registry still works in-process."""
    from trame.app import get_server

    server = get_server("trame_mcp_test_http_off")
    mcp = get_mcp(server, http=False)

    @mcp.tool
    def hello() -> str:
        """Say hi."""
        return "hi"

    await start(server)
    try:
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{server.port}/mcp", json=rpc("ping")
        ) as resp:
            # 405 comes from trame's GET-only static route: our POST
            # handler (which would answer 200, Allow: POST) is not there
            assert resp.status in (404, 405)
            assert "POST" not in resp.headers.get("Allow", "")

        assert mcp.get_tool("hello").fn() == "hi"
    finally:
        await server.stop()


async def test_initialize_gives_a_session_that_scopes_cancellation(running_server):
    import asyncio

    from trame_mcp.protocol import REQUEST_CANCELLED

    server, mcp = running_server
    url = f"http://localhost:{server.port}/mcp"
    started, cleaned_up = asyncio.Event(), asyncio.Event()

    @mcp.tool
    async def long_job() -> str:
        started.set()
        try:
            await asyncio.Event().wait()  # until cancelled
        finally:
            cleaned_up.set()
        return "done"

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=rpc("initialize", {})) as resp:
            session_id = resp.headers["Mcp-Session-Id"]
        async with session.post(url, json=rpc("initialize", {})) as resp:
            assert resp.headers["Mcp-Session-Id"] != session_id  # one per client
        # other responses carry none
        async with session.post(url, json=rpc("ping")) as resp:
            assert "Mcp-Session-Id" not in resp.headers

        headers = {"Mcp-Session-Id": session_id}
        job = asyncio.ensure_future(
            post(session, server, rpc("tools/call", {"name": "long_job"}, rpc_id=3), headers=headers)
        )
        await asyncio.wait_for(started.wait(), 2)

        cancel = {
            "jsonrpc": "2.0",
            "method": "notifications/cancelled",
            "params": {"requestId": 3, "reason": "timed out"},
        }
        # Another session's cancel leaves it running
        status, _ = await post(session, server, cancel, headers={"Mcp-Session-Id": "other"})
        assert status == 202
        await asyncio.sleep(0.05)
        assert not job.done()

        status, _ = await post(session, server, cancel, headers=headers)
        assert status == 202
        status, body = await asyncio.wait_for(job, 2)
        assert status == 200
        assert body["id"] == 3
        assert body["error"]["code"] == REQUEST_CANCELLED
        assert cleaned_up.is_set()


async def test_a_client_that_disconnects_cancels_its_call(running_server, monkeypatch):
    import asyncio

    from trame_mcp import endpoint

    monkeypatch.setattr(endpoint, "DISCONNECT_POLL", 0.02)
    server, mcp = running_server
    url = f"http://localhost:{server.port}/mcp"
    started, cleaned_up = asyncio.Event(), asyncio.Event()

    @mcp.tool
    async def wait_forever() -> str:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned_up.set()
        return "done"

    async with aiohttp.ClientSession() as session:
        # A client giving up: its request times out and the connection closes
        with pytest.raises(asyncio.TimeoutError):
            await session.post(
                url,
                json=rpc("tools/call", {"name": "wait_forever"}),
                timeout=aiohttp.ClientTimeout(total=0.3),
            )
        assert started.is_set()
        await asyncio.wait_for(cleaned_up.wait(), 2)

        # The tool lock was released: the next call runs
        status, body = await post(
            session, server, rpc("tools/call", {"name": "add", "arguments": {"a": 1, "b": 2}})
        )
        assert status == 200
        assert body["result"]["content"][0]["text"] == "3"
