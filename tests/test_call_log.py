"""The call log (``mcp.log``, ``--mcp-log``): what agents call, logged once
whichever way the call came in."""

import asyncio
import logging
import re

import pytest

from trame_mcp import get_mcp
from trame_mcp.call_log import LOGGER, shorten
from trame_mcp.dataclass_tools import summarize_batch, summarize_call, summarize_update

SESSION = "abcdef0123456789"


def rpc(method, params=None, rpc_id=1):
    msg = {"jsonrpc": "2.0", "id": rpc_id, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


class _Lines(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@pytest.fixture
def pristine_logger():
    """The call logger as an application that never configured it has it,
    restored afterwards (it is process-wide)."""
    saved = LOGGER.handlers[:], LOGGER.level, LOGGER.propagate
    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.NOTSET)
    LOGGER.propagate = True
    yield LOGGER
    LOGGER.handlers[:], LOGGER.level, LOGGER.propagate = saved


@pytest.fixture
def lines(pristine_logger):
    """The logged lines, through a handler of the application's own."""
    handler = _Lines()
    pristine_logger.addHandler(handler)
    return handler.lines


@pytest.fixture
def tools(mcp):
    @mcp.tool
    def add(a: int, b: int = 1) -> int:
        return a + b

    @mcp.tool
    def failing():
        msg = "boom"
        raise RuntimeError(msg)

    @mcp.tool
    def echo(text: str) -> str:
        return text

    @mcp.resource("trame://info", mime_type="application/json")
    def info():
        return {"name": "test"}

    @mcp.prompt
    def walkthrough(topic: str = "app") -> str:
        return f"Explain the {topic}"

    return mcp


async def test_off_by_default(tools, lines):
    assert tools.log == "off"
    await tools.protocol.handle(rpc("tools/call", {"name": "add", "arguments": {"a": 2}}))
    assert lines == []


async def test_a_call_is_logged_once_when_it_ends(tools, lines):
    tools.log = "calls"
    response = await tools.protocol.handle(
        rpc("tools/call", {"name": "add", "arguments": {"a": 2}}), session=SESSION
    )
    assert response["result"]["content"][0]["text"] == "3"
    (line,) = lines
    assert re.fullmatch(r"mcp\[abcdef\] add \{a: 2\} -> ok \d+\.\d\ds \(\d+ B\)", line), line


async def test_in_process_calls_are_logged_too(tools, lines):
    tools.log = "calls"
    assert await tools.protocol.execute(tools.get_tool("add"), {"a": 1}) == 2
    assert await tools.protocol.read_resource(tools.get_resource("trame://info")) == {
        "name": "test"
    }
    assert lines[0].startswith("mcp[in-process] add {a: 1} -> ok")
    assert lines[1].startswith("mcp[in-process] resource trame://info -> ok")


async def test_failures_say_what_the_agent_saw(tools, lines):
    tools.log = "calls"
    await tools.protocol.handle(rpc("tools/call", {"name": "failing"}))
    await tools.protocol.handle(rpc("tools/call", {"name": "add", "arguments": {"a": "x"}}))
    await tools.protocol.handle(rpc("tools/call", {"name": "missing", "arguments": {"q": 1}}))
    failing, invalid, unknown = lines
    assert re.match(r"mcp\[-\] failing -> error \d+\.\d\ds: RuntimeError: boom$", failing)
    assert "add {a: \"x\"} -> error" in invalid
    assert "Invalid arguments for tool add" in invalid
    assert unknown.startswith("mcp[-] missing {q: 1} -> error")
    assert unknown.endswith("Unknown tool: missing")


async def test_a_refused_permission_is_logged_as_denied(tools, lines):
    tools.log = "calls"
    tools.permissions.declare("files", "Read files", allowed=False)

    @tools.tool(requires="files")
    def read(path: str) -> str:
        return path

    await tools.protocol.handle(rpc("tools/call", {"name": "read", "arguments": {"path": "/"}}))
    (line,) = lines
    assert re.match(r"mcp\[-\] read \{path: \"/\"\} -> denied \d+\.\d\ds: read needs the 'files'", line)


async def test_disabled_access_is_logged(tools, lines):
    tools.log = "calls"
    tools.enabled = False
    await tools.protocol.handle(rpc("tools/call", {"name": "add", "arguments": {"a": 1}}))
    (line,) = lines
    assert "-> error" in line
    assert line.endswith("MCP access is currently disabled by the application")


async def test_a_cancelled_call(mcp, lines):
    started = asyncio.Event()

    @mcp.tool
    async def wait_forever():
        started.set()
        await asyncio.Event().wait()

    mcp.log = "calls"
    handling = asyncio.ensure_future(
        mcp.protocol.handle(rpc("tools/call", {"name": "wait_forever"}, rpc_id=7))
    )
    await started.wait()
    await mcp.protocol.handle(
        {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 7}}
    )
    await handling
    (line,) = lines
    assert re.fullmatch(r"mcp\[-\] wait_forever -> cancelled \d+\.\d\ds", line), line


async def test_a_call_waiting_for_the_tool_lock_says_so(mcp, lines):
    release = asyncio.Event()

    @mcp.tool
    async def hold():
        await release.wait()

    @mcp.tool
    def quick() -> int:
        return 1

    mcp.log = "calls"
    holding = asyncio.ensure_future(mcp.protocol.handle(rpc("tools/call", {"name": "hold"})))
    await asyncio.sleep(0.01)
    queued = asyncio.ensure_future(mcp.protocol.handle(rpc("tools/call", {"name": "quick"})))
    await asyncio.sleep(0.2)
    release.set()
    await asyncio.gather(holding, queued)
    hold_line, quick_line = lines
    assert "queued" not in hold_line
    assert re.search(r"-> ok \d+\.\d\ds \(\d+ B, queued 0\.\ds\)$", quick_line), quick_line


async def test_long_arguments_are_shortened(tools, lines):
    tools.log = "calls"
    await tools.protocol.handle(
        rpc("tools/call", {"name": "echo", "arguments": {"text": "x" * 500}})
    )
    assert lines[0].startswith("mcp[-] echo {text: <str 500 chars>} -> ok")


def test_shorten():
    assert shorten({"a": 1, "b": [True, None], "c": "hi"}) == '{a: 1, b: [true, null], c: "hi"}'
    assert shorten(list(range(10))) == "[0, 1, 2, 3, 4, 5, 6, 7, +2]"
    assert shorten({"a": {"b": {"c": {"d": 1}}}}) == "{a: {b: {c: {...}}}}"
    assert shorten(object()) == "<object>"


def test_the_generic_tools_summarize_themselves():
    steps = [
        {"call": "pipeline", "method": "set_paused", "args": {"paused": True}},
        {"new": "TransformNodeModel", "args": {"entry_name": "Blur"}, "as": "t"},
        {"update": "@t", "values": {"label": "x", "opacity": 1}},
    ]
    assert summarize_batch({"steps": steps}) == (
        "call pipeline.set_paused; new TransformNodeModel as t; update @t {label, opacity}"
    )
    assert summarize_call({"target": "6x", "method": "edit", "args": {"script": "s" * 99}}) == (
        "6x.edit {script: <str 99 chars>}"
    )
    assert summarize_call({"target": "6x", "method": "reset"}) == "6x.reset"
    assert summarize_update({"target": "5a", "updates": {"Slice": 3}}) == "5a {Slice: 3}"


async def test_a_malformed_batch_is_still_logged(mcp, lines):
    mcp.dataclass._register()
    mcp.log = "calls"
    await mcp.protocol.handle(rpc("tools/call", {"name": "dataclass_batch", "arguments": {"steps": [3]}}))
    (line,) = lines
    assert line.startswith("mcp[-] dataclass_batch {steps: [3]} -> error")


async def test_all_adds_the_protocol_traffic_and_full_arguments(tools, lines):
    tools.log = "all"
    await tools.protocol.handle(
        rpc("initialize", {"protocolVersion": "2025-06-18", "clientInfo": {"name": "cli"}})
    )
    await tools.protocol.handle(rpc("tools/list", rpc_id=2), session=SESSION)
    await tools.protocol.handle(
        rpc(
            "tools/call",
            {"name": "echo", "arguments": {"text": "y" * 100}, "_meta": {"progressToken": 4}},
            rpc_id=3,
        ),
        session=SESSION,
    )
    await tools.protocol.handle(rpc("prompts/get", {"name": "walkthrough"}, rpc_id=4))
    initialize, listing, start, end, prompt_start, prompt_end = lines
    assert initialize.startswith('mcp[- #1] initialize {"protocolVersion": "2025-06-18"')
    assert listing == "mcp[abcdef #2] tools/list"
    assert start == (
        f'mcp[abcdef #3] echo {{"text": "{"y" * 100}"}} _meta={{"progressToken": 4}} ...'
    )
    assert re.fullmatch(r"mcp\[abcdef #3\] echo -> ok \d+\.\d\ds \(\d+ B\)", end), end
    assert prompt_start == "mcp[- #4] prompt walkthrough ..."
    assert prompt_end.startswith("mcp[- #4] prompt walkthrough -> ok")


async def test_switched_at_runtime(tools, lines):
    tools.log = True
    assert tools.log == "calls"
    await tools.protocol.handle(rpc("tools/call", {"name": "add", "arguments": {"a": 1}}))
    tools.configure(log=False)
    assert tools.log == "off"
    await tools.protocol.handle(rpc("tools/call", {"name": "add", "arguments": {"a": 1}}))
    assert len(lines) == 1
    with pytest.raises(ValueError, match="log must be one of off, calls, all"):
        tools.log = "loud"


async def test_without_a_handler_lines_go_to_stderr(tools, pristine_logger, capsys):
    tools.log = "calls"
    (handler,) = pristine_logger.handlers
    assert isinstance(handler, logging.StreamHandler)
    assert pristine_logger.propagate is False
    await tools.protocol.handle(rpc("tools/call", {"name": "add", "arguments": {"a": 1}}))
    assert re.search(r"\d\d:\d\d:\d\d mcp\[-\] add \{a: 1\} -> ok", capsys.readouterr().err)


def test_an_application_handler_is_kept(tools, lines, pristine_logger):
    tools.log = "calls"
    assert len(pristine_logger.handlers) == 1  # the application's, no stderr one
    assert pristine_logger.propagate is True


@pytest.mark.parametrize(
    ("argv", "level"),
    [("--mcp-log", "calls"), ("--mcp-log all", "all"), ("--mcp-log off", "off")],
)
def test_the_command_line_wins_until_start(monkeypatch, pristine_logger, argv, level):
    from trame.app import get_server

    # trame reads TRAME_ARGS instead of sys.argv when running under pytest
    monkeypatch.setenv("TRAME_ARGS", argv)
    server = get_server(f"trame_mcp_test_log_{level}")
    mcp = get_mcp(server, log="calls" if level == "off" else "off")
    assert mcp.log == level
    mcp.log = "calls" if level == "off" else "off"
    assert mcp.log == level


def test_the_environment_sets_it(monkeypatch, pristine_logger):
    from trame.app import get_server

    monkeypatch.setenv("TRAME_MCP_LOG", "all")
    assert get_mcp(get_server("trame_mcp_test_log_env")).log == "all"
    monkeypatch.setenv("TRAME_MCP_LOG", "verbose")
    assert get_mcp(get_server("trame_mcp_test_log_env_bad")).log == "off"
