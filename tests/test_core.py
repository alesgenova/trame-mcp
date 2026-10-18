from typing import Annotated

import pytest

from trame_mcp import get_mcp, prompt, resource, tool


def test_get_mcp_is_per_server_singleton(trame_server):
    assert get_mcp(trame_server) is get_mcp(trame_server)


def test_tool_decorator_bare_and_options(mcp):
    @mcp.tool
    def alpha(x: int) -> int:
        """Add one."""
        return x + 1

    @mcp.tool(name="beta_renamed", description="custom")
    def beta() -> str:
        return "ok"

    names = {t.name for t in mcp.list_tools()}
    assert {"alpha", "beta_renamed"} <= names
    assert mcp.get_tool("alpha").description == "Add one."
    assert mcp.get_tool("beta_renamed").description == "custom"
    assert mcp.get_tool("alpha").input_schema["required"] == ["x"]


def test_expose_marked_functions(mcp):
    @tool(name="renamed", read_only=True)
    def alpha() -> str:
        return "ok"

    @resource("app://info")
    def info() -> dict:
        return {}

    # markers are inert until exposed
    assert mcp.get_tool("renamed") is None
    mcp.expose(alpha, info)
    assert mcp.get_tool("renamed").annotations == {"readOnlyHint": True}
    assert mcp.get_resource("app://info") is not None


def test_tool_name_collision_raises(mcp):
    @mcp.tool
    def gamma():
        pass

    with pytest.raises(ValueError, match="already registered"):

        @mcp.tool(name="gamma")
        def other():
            pass


def test_expose_rejects_what_it_cannot_expose(mcp):
    from trame_dataclass.v2 import StateDataModel, Sync

    def unmarked():
        pass

    class Plain:
        def method(self):
            pass

    class Model(StateDataModel):
        value = Sync(int, 0)

    with pytest.raises(TypeError, match="not marked"):
        mcp.expose(unmarked)
    with pytest.raises(ValueError, match="no methods marked"):
        mcp.expose(Plain())
    with pytest.raises(TypeError, match="functions or objects"):
        mcp.expose(Plain)
    with pytest.raises(TypeError, match="expose_instance"):
        mcp.expose(Model(mcp.server))
    assert mcp.list_tools() == []


def test_expose_object_with_prefix(mcp):
    class Component:
        @tool
        def do_thing(self, value: int) -> int:
            """Do the thing."""
            return value

        @resource("trame://component/info")
        def info(self):
            return {"ok": True}

        @prompt
        def guide(self) -> str:
            return "hello"

        def unmarked(self):
            pass

    mcp.expose(Component(), prefix="comp_")

    assert mcp.get_tool("comp_do_thing") is not None
    assert mcp.get_tool("comp_unmarked") is None
    assert mcp.get_resource("trame://component/info") is not None
    assert mcp.get_prompt("comp_guide") is not None
    # bound method schema does not include self
    assert list(mcp.get_tool("comp_do_thing").input_schema["properties"]) == ["value"]


def test_prompt_keeps_its_description_apart_from_its_arguments(mcp):
    @mcp.prompt(description="Explore a file.")
    def explore(path: Annotated[str, "File to explore"], depth: int = 1) -> str:
        return path

    @mcp.prompt
    def documented(path: Annotated[str, "File to explore"]) -> str:
        """Explore with the docstring."""
        return path

    explore_entry = mcp.get_prompt("explore")
    assert explore_entry.description == "Explore a file."
    assert explore_entry.arguments == [
        {"name": "path", "required": True, "description": "File to explore"},
        {"name": "depth", "required": False},
    ]
    assert mcp.get_prompt("documented").description == "Explore with the docstring."


def test_expose_state_tools(mcp):
    server = mcp.server
    server.state.resolution = 6
    server.state.color = "red"
    server.state.secret = "hidden"

    mcp.state.expose("color", read_only=True)
    # nothing writable yet: no state_update to advertise
    assert mcp.get_tool("state_get") is not None
    assert mcp.get_tool("state_update") is None

    mcp.state.expose("resolution", type=int, schema={"minimum": 3})

    state_get = mcp.get_tool("state_get")
    state_update = mcp.get_tool("state_update")
    assert state_get.annotations == {"readOnlyHint": True}

    assert state_get.fn() == {"color": "red", "resolution": 6}
    assert state_get.fn(names=["resolution"]) == {"resolution": 6}
    with pytest.raises(ValueError, match="not exposed"):
        state_get.fn(names=["secret"])

    assert state_update.fn(updates={"resolution": 12}) == {"updated": ["resolution"]}
    assert server.state.resolution == 12

    with pytest.raises(ValueError, match="not writable"):
        state_update.fn(updates={"color": "blue"})
    with pytest.raises(ValueError, match="not writable"):
        state_update.fn(updates={"secret": "x"})

    # type and constraints are enforced, all-or-nothing
    with pytest.raises(ValueError, match="expected integer"):
        state_update.fn(updates={"resolution": "many"})
    with pytest.raises(ValueError, match="at least 3"):
        state_update.fn(updates={"resolution": 1})
    assert server.state.resolution == 12

    # the writable variables' schemas are part of the advertised input schema
    updates = state_update.input_schema["properties"]["updates"]
    assert updates["properties"] == {"resolution": {"type": "integer", "minimum": 3}}
    assert "`resolution` (integer, writable)" in state_update.description


def test_expose_state_rejects_non_bool_read_only(mcp):
    with pytest.raises(TypeError, match="True or False"):
        mcp.state.expose("a", "b", read_only=["a"])


def test_unexpose_state(mcp):
    mcp.server.state.a = 1
    mcp.state.expose("a", "b")
    mcp.state.unexpose("b")

    assert mcp.get_tool("state_get").fn() == {"a": 1}
    assert "`b`" not in mcp.get_tool("state_update").description
    with pytest.raises(ValueError, match="not writable"):
        mcp.get_tool("state_update").fn(updates={"b": 2})


def test_tool_annotations(mcp):
    @mcp.tool(read_only=True)
    def look() -> str:
        return "ok"

    @mcp.tool(annotations={"destructiveHint": True})
    def wipe() -> str:
        return "ok"

    @mcp.tool
    def plain() -> str:
        return "ok"

    assert mcp.get_tool("look").annotations == {"readOnlyHint": True}
    assert mcp.get_tool("wipe").annotations == {"destructiveHint": True}
    assert mcp.get_tool("plain").annotations is None


def test_get_mcp_applies_config_on_every_call(trame_server):
    mcp = get_mcp(trame_server)
    assert mcp.path == "/mcp"
    assert get_mcp(trame_server, name="renamed", path="api/mcp") is mcp
    assert mcp.name == "renamed"
    assert mcp.path == "/api/mcp"
    # None leaves options unchanged
    get_mcp(trame_server, name=None)
    assert mcp.name == "renamed"


def test_configure_rejects_unknown_options(mcp):
    with pytest.raises(TypeError):
        mcp.configure(bogus=1)


def test_off_by_default(trame_server):
    mcp = get_mcp(trame_server)

    @mcp.tool
    def ping() -> str:
        """Ping."""
        return "pong"

    mcp.state.expose("x")
    assert mcp.enabled is False


def test_launch_options_win_until_start(monkeypatch):
    from trame.app import get_server

    # trame reads TRAME_ARGS instead of sys.argv when running under pytest
    monkeypatch.setenv("TRAME_ARGS", "--no-mcp --mcp-token from-cli")
    server = get_server("trame_mcp_test_launch_options")
    mcp = get_mcp(server, enabled=True, token="from-code")
    assert mcp.enabled is False
    assert mcp.token == "from-cli"

    # the application's values are only defaults before start, however set
    mcp.enabled = True
    mcp.configure(token="from-configure")
    assert mcp.enabled is False
    assert mcp.token == "from-cli"


def test_only_disabling_launch_options_hold_after_start(monkeypatch):
    from trame.app import get_server

    monkeypatch.setenv("TRAME_ARGS", "--mcp")
    mcp = get_mcp(get_server("trame_mcp_test_launch_mcp"))
    assert mcp.enabled is True
    assert mcp.enabled_locked is False
    mcp._start()
    mcp.enabled = False  # turning on only set the starting value
    assert mcp.enabled is False

    monkeypatch.setenv("TRAME_ARGS", "--no-mcp")
    mcp = get_mcp(get_server("trame_mcp_test_launch_no_mcp"))
    assert mcp.enabled_locked is True
    mcp._start()
    mcp.enabled = True  # turning off holds for the run
    assert mcp.enabled is False


def test_token_from_environment(monkeypatch):
    from trame.app import get_server

    monkeypatch.setenv("TRAME_MCP_TOKEN", "from-env")
    mcp = get_mcp(get_server("trame_mcp_test_token_env"))
    assert mcp.token == "from-env"
