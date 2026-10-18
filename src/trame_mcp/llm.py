"""Bridge a trame-mcp ``MCP`` registry into trame-llm's ``LLMClient``.

Requires ``trame-llm`` (which brings ``fastmcp``); trame-mcp itself does not
depend on either — everything fastmcp-specific is imported lazily.
"""

from __future__ import annotations

from trame_mcp.content import Image

__all__ = ["LLMAdapter"]


class LLMAdapter:
    """Adapt a :class:`trame_mcp.MCP` to trame-llm's ``register_mcp()`` contract.

    ``LLMClient.register_mcp(controller)`` reads ``controller.mcp`` — a config
    dict it feeds to ``FastMCP(**config)``. This adapter mirrors the trame-mcp
    registry through that contract, so a single set of tool definitions serves
    both external agents (the HTTP endpoint) and the embedded chat in-process:

    .. code-block:: python

        from trame_llm.core import LLMClient
        from trame_mcp import LLMAdapter, get_mcp

        mcp = get_mcp(server, enabled=True)  # tools exposed as usual
        llm_client = LLMClient()
        llm_client.register_mcp(LLMAdapter(mcp))

    Tool names, descriptions, input schemas and annotations are mirrored
    exactly (including the hand-authored schemas of the generic
    ``dataclass_*`` / ``state_*`` tools, which cannot be derived from their
    function signatures). Calls and resource reads go through the same path
    as the HTTP endpoint: ``mcp.enabled``, argument validation and the
    ``serialize_tools`` lock apply to external and chat calls alike.

    Notes:
        - Agent access is off until ``mcp.enabled`` is set; while off, the
          chat still lists the tools but every call fails with
          :class:`trame_mcp.MCPDisabledError`.
        - trame-llm snapshots the registry when ``register_mcp()`` runs;
          register the adapter after everything has been exposed.
        - MCP prompts are not mirrored: trame-llm's contract has no prompt
          support.
        - The HTTP endpoint is independent of the adapter: set
          ``mcp.http = False`` (or ``get_mcp(server, http=False)``) for a
          chat-only app with no externally reachable MCP route.
    """

    def __init__(self, mcp):
        self._mcp = mcp

    @property
    def mcp(self):
        config = {
            "name": self._mcp.name,
            "tools": [self._mirror_tool(entry) for entry in self._mcp.list_tools()],
        }
        config["instructions"] = self._mcp._agent_instructions()
        resources = {
            entry.uri: self._mirror_resource(entry) for entry in self._mcp.list_resources()
        }
        if resources:
            config["resources"] = resources
        return config

    def _mirror_tool(self, entry):
        from fastmcp.tools.tool import FunctionTool
        from mcp.types import ToolAnnotations

        protocol = self._mcp.protocol

        async def run(**arguments):
            return _adapt_result(await protocol.execute(entry, arguments))

        run.__name__ = entry.name
        return FunctionTool(
            fn=run,
            name=entry.name,
            description=entry.description,
            parameters=entry.input_schema,
            annotations=ToolAnnotations(**entry.annotations) if entry.annotations else None,
        )

    def _mirror_resource(self, entry):
        protocol = self._mcp.protocol

        async def read():
            return await protocol.read_resource(entry)

        read.__name__ = entry.name
        read.__doc__ = entry.description
        return read


def _adapt_result(value):
    """Translate trame-mcp content types into their fastmcp equivalents."""
    if isinstance(value, Image):
        from fastmcp.utilities.types import Image as FastMCPImage

        return FastMCPImage(
            data=value.data, format=value.mime_type.partition("/")[2] or None
        )
    if isinstance(value, (list, tuple)):
        return [_adapt_result(v) for v in value]
    return value
