"""MCP-enabled variants of the modern trame application base classes.

When built, they expose their ``@tool`` / ``@resource`` / ``@prompt``
methods on the server's MCP (``self.mcp.expose(self)``), which stays off
until the application sets ``self.mcp.enabled`` (or the end user launches
with ``--mcp``). A library can therefore build on ``MCPComponent`` without
turning anything on in the applications that use it.
"""

from __future__ import annotations

from trame.app import TrameApp, TrameComponent

from trame_mcp.core import get_mcp

__all__ = ["MCPApp", "MCPComponent", "MCPMixin"]


def _reject_mcp_options(cls, kwargs):
    # TrameApp / TrameComponent swallow unknown keyword arguments, which would
    # silently drop e.g. mcp_enabled=True.
    stray = sorted(k for k in kwargs if k.startswith("mcp_"))
    if stray:
        msg = (
            f"{cls.__name__}() got unexpected keyword arguments {stray}; "
            f"configure MCP with self.mcp.configure(...) after super().__init__()"
        )
        raise TypeError(msg)


class MCPMixin:
    """Exposes the marked methods of any ``TrameComponent`` subclass.

    Rides on ``_bind_annotated_methods()`` (called during construction, like
    trame's own ``@change`` / ``@trigger``), so it composes with existing base
    classes without cooperative ``__init__``:

    .. code-block:: python

        class MyApp(MCPMixin, TheirTrameAppSubclass): ...
    """

    _mcp_prefix = ""

    def _bind_annotated_methods(self):
        super()._bind_annotated_methods()
        get_mcp(self.server)._expose_object(self, prefix=self._mcp_prefix)

    @property
    def mcp(self):
        return get_mcp(self.server)


class MCPApp(MCPMixin, TrameApp):
    """A ``trame.app.TrameApp`` whose ``@tool`` / ``@resource`` / ``@prompt``
    methods are exposed on the server's MCP.

    Configure through ``self.mcp`` once ``super().__init__()`` has run:

    .. code-block:: python

        class ConeApp(MCPApp):
            def __init__(self, server=None):
                super().__init__(server)
                self.mcp.configure(name="cone-app", enabled=True)
    """

    def __init__(self, server=None, client_type="vue3", ctx_name=None, **kwargs):
        _reject_mcp_options(type(self), kwargs)
        super().__init__(server=server, client_type=client_type, ctx_name=ctx_name, **kwargs)


class MCPComponent(MCPMixin, TrameComponent):
    """A ``trame.app.TrameComponent`` contributing tools to the app's MCP,
    optionally namespaced with ``mcp_prefix``."""

    def __init__(self, server, ctx_name=None, mcp_prefix="", **kwargs):
        _reject_mcp_options(type(self), kwargs)
        self._mcp_prefix = mcp_prefix
        super().__init__(server, ctx_name=ctx_name, **kwargs)
