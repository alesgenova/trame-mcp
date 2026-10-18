"""trame-mcp: add an MCP server to any trame application."""

from trame_mcp.content import Image
from trame_mcp.core import MCP, MCPDisabledError, get_mcp
from trame_mcp.decorators import constructor, mcp_field, prompt, resource, tool
from trame_mcp.llm import LLMAdapter
from trame_mcp.permissions import PermissionDeniedError

__version__ = "0.1.0"

__all__ = [
    "MCP",
    "Image",
    "LLMAdapter",
    "MCPApp",
    "MCPComponent",
    "MCPDisabledError",
    "MCPMixin",
    "MCPStateDataModel",
    "PermissionDeniedError",
    "constructor",
    "get_mcp",
    "mcp_field",
    "prompt",
    "resource",
    "tool",
]

_LAZY = {
    "MCPApp": "trame_mcp.app",
    "MCPComponent": "trame_mcp.app",
    "MCPMixin": "trame_mcp.app",
    "MCPStateDataModel": "trame_mcp.models",
}


def __getattr__(name):
    # The app classes require `trame` and MCPStateDataModel requires
    # `trame-dataclass`; import lazily so the core library works with
    # trame-server alone.
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
