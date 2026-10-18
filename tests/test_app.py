import pytest

from trame_mcp import MCPApp, MCPComponent, tool


class Panel(MCPComponent):
    @tool
    def reset(self) -> str:
        """Reset the panel."""
        return "reset"


class MyApp(MCPApp):
    def __init__(self, server=None, **kwargs):
        super().__init__(server=server, **kwargs)
        self.panel = Panel(self.server, mcp_prefix="panel_")

    @tool
    def do_thing(self, value: int) -> int:
        """Do the thing."""
        return value * 2


def test_mcp_app_and_component_binding():
    app = MyApp(server="trame_mcp_test_app")

    assert app.mcp.get_tool("do_thing") is not None
    assert app.mcp.get_tool("panel_reset") is not None
    assert app.mcp.get_tool("reset") is None

    assert app.mcp.get_tool("do_thing").fn(3) == 6
    # declaring tools turns nothing on
    assert app.mcp.enabled is False


def test_mcp_app_configured_through_mcp():
    class Configured(MCPApp):
        def __init__(self, server=None):
            super().__init__(server)
            self.mcp.configure(path="custom/mcp", name="my-app", enabled=True)

    app = Configured("trame_mcp_test_app_cfg")
    assert app.mcp.path == "/custom/mcp"
    assert app.mcp.name == "my-app"
    assert app.mcp.enabled is True


def test_mcp_options_rejected_as_constructor_arguments():
    # TrameApp would silently swallow them
    with pytest.raises(TypeError, match="mcp_enabled"):
        MyApp(server="trame_mcp_test_app_kwargs", mcp_enabled=True)
    with pytest.raises(TypeError, match="mcp_prefx"):
        Panel(MyApp(server="trame_mcp_test_app_kwargs_2").server, mcp_prefx="p_")


def test_mcp_app_keeps_prior_configuration():
    from trame_mcp import get_mcp

    mcp = get_mcp("trame_mcp_test_app_prior", path="/agent", serialize_tools=False)
    app = MyApp(server="trame_mcp_test_app_prior")
    assert app.mcp is mcp
    assert mcp.path == "/agent"
    assert mcp.serialize_tools is False
