"""LLMAdapter: the trame-mcp registry consumed through trame-llm's contract.

These tests drive the FastMCP instance that ``LLMClient.register_mcp()``
builds from the adapter — the exact object trame-llm's chat loop talks to —
through an in-memory ``fastmcp.Client``, mirroring ``_discover_mcp_tools`` /
``call_mcp_tool``.
"""

import asyncio

import pytest

pytest.importorskip("fastmcp")
pytest.importorskip("trame_llm")

from fastmcp import Client

from trame_mcp import Image, LLMAdapter

GENERIC_SCHEMA = {
    "type": "object",
    "properties": {
        "payload": {"type": "object", "description": "Any JSON object"},
    },
    "required": ["payload"],
}


@pytest.fixture
def populated_mcp(mcp):
    @mcp.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @mcp.tool(name="renamed")
    def internal_name() -> str:
        """Tool whose public name differs from the function name."""
        return "ok"

    @mcp.tool
    def boom() -> str:
        """Always fails."""
        msg = "nope"
        raise RuntimeError(msg)

    @mcp.tool
    def screenshot() -> Image:
        """Return an image."""
        return Image(b"not-really-png", mime_type="image/png")

    # A hand-authored schema, like the dataclass_*/state_* generic tools:
    # unrecoverable from the function signature alone.
    mcp._add_tool(
        lambda payload: payload,
        name="generic_echo",
        description="Echo the payload.",
        input_schema=GENERIC_SCHEMA,
        annotations={"readOnlyHint": True},
    )

    @mcp.resource("resource://greeting", mime_type="text/plain")
    def greeting() -> str:
        """A greeting."""
        return "hello"

    @mcp.prompt
    def demo() -> str:
        """A prompt."""
        return "do things"

    mcp.instructions = "Test instructions."
    return mcp


def register(mcp):
    """Register through trame-llm and return what its chat loop would drive."""
    from trame_llm.core import LLMClient

    llm_client = LLMClient()
    llm_client.register_mcp(LLMAdapter(mcp))
    return llm_client._mcps[0]


async def test_registry_is_mirrored_exactly(populated_mcp):
    fastmcp_server = register(populated_mcp)

    async with Client(fastmcp_server) as client:
        tools = {t.name: t for t in await client.list_tools()}
        expected = {e.name: e for e in populated_mcp.list_tools()}
        assert set(tools) == set(expected)
        assert set(expected) == {"add", "renamed", "boom", "screenshot", "generic_echo"}

        for name, entry in expected.items():
            assert tools[name].description == entry.description
            assert tools[name].inputSchema == entry.input_schema
        assert tools["generic_echo"].annotations.readOnlyHint is True
        assert tools["add"].annotations is None

        resources = await client.list_resources()
        assert [str(r.uri) for r in resources] == ["resource://greeting"]
        content = await client.read_resource("resource://greeting")
        assert content[0].text == "hello"

        # trame-llm's contract has no prompt support: not mirrored
        assert await client.list_prompts() == []


async def test_tool_calls_and_errors(populated_mcp):
    async with Client(register(populated_mcp)) as client:
        result = await client.call_tool("add", {"a": 2, "b": 3})
        assert result.content[0].text == "5"

        result = await client.call_tool("renamed", {})
        assert result.content[0].text == "ok"

        payload = {"nested": [1, 2, {"deep": True}]}
        result = await client.call_tool("generic_echo", {"payload": payload})
        assert result.structured_content == payload

        result = await client.call_tool("boom", {}, raise_on_error=False)
        assert result.is_error
        assert "nope" in result.content[0].text

        result = await client.call_tool("screenshot", {})
        assert result.content[0].type == "image"
        assert result.content[0].mimeType == "image/png"


async def test_arguments_validated_and_enabled_switch_applies(populated_mcp):
    async with Client(register(populated_mcp)) as client:
        result = await client.call_tool("add", {"a": 2, "b": "3"}, raise_on_error=False)
        assert result.is_error

        populated_mcp.enabled = False
        result = await client.call_tool("add", {"a": 2, "b": 3}, raise_on_error=False)
        assert result.is_error
        assert "disabled" in result.content[0].text
        with pytest.raises(Exception, match="disabled"):
            await client.read_resource("resource://greeting")

        populated_mcp.enabled = True
        result = await client.call_tool("add", {"a": 2, "b": 3})
        assert result.content[0].text == "5"


async def test_serialize_tools_lock_is_shared(mcp):
    order = []

    @mcp.tool
    async def slow(tag: str) -> str:
        """Interleaving canary."""
        order.append(f"{tag}:start")
        await asyncio.sleep(0.01)
        order.append(f"{tag}:end")
        return tag

    async with Client(register(mcp)) as client:
        await asyncio.gather(
            client.call_tool("slow", {"tag": "a"}),
            client.call_tool("slow", {"tag": "b"}),
        )

    assert len(order) == 4
    # serialize_tools (default True) must prevent interleaving across awaits
    assert order[0].split(":")[0] == order[1].split(":")[0]
    assert order[2].split(":")[0] == order[3].split(":")[0]
