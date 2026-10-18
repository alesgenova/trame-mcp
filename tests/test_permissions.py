"""Permissions (``mcp.permissions``, ``requires=``, ``--mcp-deny``): powers
the people running an application withhold from agents."""

import logging

import pytest
from trame_dataclass.v2 import StateDataModel, Sync

from trame_mcp import (
    MCPStateDataModel,
    PermissionDeniedError,
    constructor,
    get_mcp,
    mcp_field,
    tool,
)
from trame_mcp.dataclass_tools import BatchError

REFUSED = "which is turned off. Only the people running the application can turn it on."


def rpc(method, params=None):
    return {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}


class Script(MCPStateDataModel):
    """Code an agent wrote."""

    code = mcp_field(Sync(str, ""))

    @constructor(requires="scripts")
    @classmethod
    def new(cls, server, code: str) -> "Script":
        return cls(server, code=code)

    @tool(requires="scripts")
    def edit(self, code: str) -> str:
        self.code = code
        return code

    @tool(read_only=True)
    def read(self) -> str:
        return self.code


class Note(MCPStateDataModel):
    path = mcp_field(Sync(str, ""))

    @constructor
    @classmethod
    def new(cls, server) -> "Note":
        return cls(server)

    @tool(requires="files")
    def attach(self, path: str) -> str:
        self.path = path
        return path


class Workspace(MCPStateDataModel):
    name = mcp_field(Sync(str, "a"))
    scripts = mcp_field(Sync(list[Script], list, has_dataclass=True))
    notes = mcp_field(Sync(list[Note], list, has_dataclass=True))

    @tool
    def rename(self, name: str) -> "Workspace":
        self.name = name
        return self


@pytest.fixture
def permissions(mcp):
    mcp.permissions.declare("files", "Read files and folders")
    mcp.permissions.declare("scripts", "Run code the agent writes", implies="files")
    return mcp.permissions


@pytest.fixture
def workspace(mcp):
    workspace = Workspace()
    workspace.scripts = [Script.new(None, "print(1)")]
    mcp.dataclass.expose_instance(workspace, name="workspace")
    return workspace


def fn(mcp, name):
    return mcp.get_tool(name).fn


# -----------------------------------------------------------------------------
# Declaring and setting
# -----------------------------------------------------------------------------


def test_implied_permissions_follow_each_other(permissions):
    assert dict(permissions) == {"files": True, "scripts": True}
    permissions["files"] = False  # a script could read files anyway
    assert permissions.denied == ("files", "scripts")
    permissions["scripts"] = True  # and needs them
    assert permissions.denied == ()
    permissions["scripts"] = False
    assert permissions.denied == ("scripts",)

    with pytest.raises(ValueError, match="Cannot allow and deny"):
        permissions.update(scripts=True, files=False)
    assert permissions.denied == ("scripts",)
    with pytest.raises(KeyError, match="No permission 'script' is declared"):
        permissions["script"] = False


def test_declarations_are_checked(mcp):
    permissions = mcp.permissions
    with pytest.raises(ValueError, match="not declared"):
        permissions.declare("scripts", "Run code", implies="files")
    permissions.declare("files", "Read files", allowed=False)
    permissions.declare("scripts", "Run code", implies=["files"])
    assert permissions["scripts"] is False  # never allowed without what it implies
    with pytest.raises(ValueError, match="already declared"):
        permissions.declare("files", "Read files")
    with pytest.raises(TypeError, match="permission name"):
        tool(requires=3)
    with pytest.raises(TypeError, match="permission name"):
        constructor(requires=[""])


# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------


async def test_a_denied_tool_is_not_offered_and_refused(mcp, permissions):
    @mcp.tool(requires="files")
    def list_files(path: str) -> list:
        return [path]

    @mcp.tool
    def ping() -> str:
        return "pong"

    permissions["files"] = False
    listed = await mcp.protocol.handle(rpc("tools/list"))
    assert [t["name"] for t in listed["result"]["tools"]] == ["ping"]
    assert mcp.list_tools() == [mcp.get_tool("ping")]  # what LLMAdapter mirrors

    # A client that listed it earlier still calls it
    reply = await mcp.protocol.handle(
        rpc("tools/call", {"name": "list_files", "arguments": {"path": "/"}})
    )
    assert reply["result"]["isError"]
    assert reply["result"]["content"][0]["text"] == (
        f"list_files needs the 'files' permission (Read files and folders), {REFUSED}"
    )

    permissions["files"] = True
    assert await mcp.protocol.execute(mcp.get_tool("list_files"), {"path": "/"}) == ["/"]


async def test_a_call_checks_what_its_arguments_need(mcp, permissions):
    @mcp.tool
    def run(external: bool = False) -> str:
        if external:
            mcp.permissions.require("files", what="run(external)")
        return "ran"

    permissions["files"] = False
    assert await mcp.protocol.execute(mcp.get_tool("run"), {}) == "ran"
    with pytest.raises(PermissionDeniedError, match=r"^run\(external\) needs the 'files'"):
        await mcp.protocol.execute(mcp.get_tool("run"), {"external": True})


async def test_the_instructions_say_what_is_turned_off(mcp, permissions):
    permissions["scripts"] = False
    reply = await mcp.protocol.handle(rpc("initialize", {"protocolVersion": "2025-06-18"}))
    assert (
        "Turned off in this application: scripts (Run code the agent writes). What needs "
        "them is not offered, and calls to it are refused."
    ) in reply["result"]["instructions"]


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------


def test_denied_methods_and_constructors_leave_the_schemas(mcp, permissions, workspace):
    permissions["scripts"] = False
    script = fn(mcp, "dataclass_schemas")()["models"]["Script"]
    assert "constructor" not in script
    assert list(script["methods"]) == ["read"]

    permissions["scripts"] = True
    script = fn(mcp, "dataclass_schemas")()["models"]["Script"]
    assert script["constructor"]["name"] == "new"
    assert sorted(script["methods"]) == ["edit", "read"]


def test_expose_model_names_requirements(mcp, permissions):
    class Library(StateDataModel):
        title = Sync(str, "")

        def burn(self):
            pass

    mcp.dataclass.expose_model(Library, all_fields=True, tools={"burn": {"requires": "scripts"}})
    assert mcp.dataclass._policy(Library).tools["burn"].requires == ("scripts",)


async def test_a_denied_method_is_refused(mcp, permissions, workspace):
    permissions["scripts"] = False
    script = workspace.scripts[0]
    call = fn(mcp, "dataclass_call")
    with pytest.raises(PermissionDeniedError, match=r"^Script.edit needs the 'scripts' permission"):
        await call(script._id, "edit", {"code": "rm()"})
    assert script.code == "print(1)"
    assert await call(script._id, "read") == "print(1)"


async def test_a_batch_with_a_denied_step_applies_nothing(mcp, permissions, workspace):
    batch = fn(mcp, "dataclass_batch")
    rename = {"call": "workspace", "method": "rename", "args": {"name": "b"}}
    script = workspace.scripts[0]._id

    permissions["scripts"] = False
    for step, what in [
        ({"new": "Script", "args": {"code": "x"}}, "new Script"),
        ({"call": script, "method": "edit", "args": {"code": "x"}}, f"call edit on {script}"),
    ]:
        with pytest.raises(PermissionDeniedError) as refused:
            await batch([rename, step])
        assert str(refused.value).startswith(
            f"Nothing was applied: steps[1] ({what}) needs the 'scripts' permission"
        )
    # a method of an instance the batch builds
    permissions["files"] = False
    with pytest.raises(PermissionDeniedError, match=r"steps\[2\] \(call attach on @n\) needs the 'files'"):
        await batch([rename, {"new": "Note", "as": "n"}, {"call": "@n", "method": "attach", "args": {"path": "/"}}])
    assert workspace.name == "a"


async def test_a_step_reached_while_running_is_refused_there(mcp, permissions, workspace):
    """A target only known once earlier steps ran is checked by its step."""
    permissions["scripts"] = False
    with pytest.raises(BatchError) as failed:
        await fn(mcp, "dataclass_batch")(
            [
                {"call": "workspace", "method": "rename", "args": {"name": "b"}, "as": "w"},
                {"call": "@w.scripts.0", "method": "edit", "args": {"code": "x"}},
            ]
        )
    assert str(failed.value).startswith(
        "steps[1] (call edit on @w.scripts.0) failed: Script.edit needs the 'scripts' permission"
    )
    assert workspace.name == "b"  # the steps before stay applied
    assert workspace.scripts[0].code == "print(1)"


# -----------------------------------------------------------------------------
# Launch
# -----------------------------------------------------------------------------


def _launched(monkeypatch, name, argv, env=None):
    from trame.app import get_server

    # trame reads TRAME_ARGS instead of sys.argv when running under pytest
    monkeypatch.setenv("TRAME_ARGS", argv)
    if env is not None:
        monkeypatch.setenv("TRAME_MCP_DENY", env)
    mcp = get_mcp(get_server(f"trame_mcp_test_deny_{name}"), enabled=True)
    mcp.permissions.declare("files", "Read files")
    mcp.permissions.declare("scripts", "Run code", implies="files")
    return mcp


def test_the_command_line_denies_for_the_whole_run(monkeypatch):
    mcp = _launched(monkeypatch, "cli", "--mcp-deny files")
    permissions = mcp.permissions
    assert permissions.denied == permissions.locked == ("files", "scripts")
    permissions["scripts"] = True
    assert permissions.denied == ("files", "scripts")
    mcp._start()
    permissions.update(files=True, scripts=True)  # a UI switch, once running
    assert permissions.denied == ("files", "scripts")

    # Denying scripts leaves files to the application
    mcp = _launched(monkeypatch, "cli_scripts", "--mcp-deny scripts")
    permissions = mcp.permissions
    assert permissions.locked == ("scripts",)
    mcp._start()
    permissions["files"] = False
    permissions["scripts"] = True  # it would bring files back with it
    assert permissions.denied == ("files", "scripts")
    permissions["files"] = True
    assert permissions.denied == ("scripts",)


def test_the_environment_denies_too(monkeypatch):
    mcp = _launched(monkeypatch, "env", "--mcp-deny files --mcp-deny net", env="scripts, ")
    assert mcp.permissions._launch_denied == {"files", "net", "scripts"}


async def test_an_unknown_name_at_launch_refuses_every_call(monkeypatch, caplog):
    mcp = _launched(monkeypatch, "typo", "--mcp-deny script")

    @mcp.tool
    def ping() -> str:
        return "pong"

    with caplog.at_level(logging.ERROR, logger="trame_mcp.permissions"):
        mcp._start()
    assert "--mcp-deny names script, which this application does not declare" in caplog.text
    with pytest.raises(PermissionDeniedError, match="launched denying script"):
        await mcp.protocol.execute(mcp.get_tool("ping"), {})
