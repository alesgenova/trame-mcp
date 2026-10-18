# trame-mcp

Add an MCP (Model Context Protocol) server to any trame application, so LLM
clients (Claude Code, Claude Desktop, ...) can inspect and drive the app while
human users watch it update live in the browser.

The MCP endpoint is served on the same port as the application itself — no
second server, no second process.

## Quick start

```python
from trame.app import get_server

from trame_mcp import get_mcp

server = get_server()
mcp = get_mcp(server)  # the MCP server attached to this trame server

server.state.resolution = 6


@mcp.tool
def set_resolution(resolution: int) -> str:
    """Set the number of facets of the cone."""
    with server.state as state:
        state.resolution = resolution
    return f"Resolution set to {resolution}"


mcp.enabled = True  # nothing is served until agent access is on
server.start()
```

At startup the console shows how to connect:

```
MCP server available at http://localhost:8080/mcp
  claude mcp add --transport http trame http://localhost:8080/mcp
```

## Declaring what the agent can use

The decorators follow trame's own split between plain functions and methods:

|                                                        | trame                                     | trame-mcp                                        |
| ------------------------------------------------------ | ----------------------------------------- | ------------------------------------------------ |
| Plain functions: exposed on that server right away     | `@state.change("a")`, `@ctrl.set("name")`  | `@mcp.tool`, `@mcp.resource(uri)`, `@mcp.prompt` |
| Methods: inert markers, bound when the object is built | `@change("a")`, `@controller.set("name")` | `@tool`, `@resource(uri)`, `@prompt`             |

`mcp` is always the object `get_mcp(server)` returns: the MCP server of that
trame server, one per server like `get_server()`. The markers (`tool`,
`resource`, `prompt`, plus `mcp_field` for trame-dataclass fields) are
imported from `trame_mcp`; like `trame.decorators`, they need no server, so
libraries can use them freely. So do the base classes, which pair with
trame's: `MCPApp` / `TrameApp`, `MCPComponent` / `TrameComponent` and
`MCPStateDataModel` / `StateDataModel`.

| What                      | Declared with                                                         | The agent gets                  |
| ------------------------- | --------------------------------------------------------------------- | ------------------------------- |
| Functions                 | `@mcp.tool`, `@mcp.resource(uri)`, `@mcp.prompt`                      | a tool, resource or prompt each |
| App and component methods | `@tool`, `@resource(uri)`, `@prompt` in an `MCPApp` / `MCPComponent`  | the same, optionally prefixed   |
| Plain trame state         | `mcp.state.expose(name, ...)`                                         | `state_get` / `state_update`    |
| trame-dataclass models    | `MCPStateDataModel` with `mcp_field`, `@tool` and `@constructor` for models you own; `mcp.dataclass.expose_model(cls)` for others; `mcp.dataclass.expose_instance(obj, name)` for entry points | `dataclass_schemas` / `dataclass_get` / `dataclass_update` / `dataclass_call` / `dataclass_batch` |

Declaring serves nothing: agent access stays off until `mcp.enabled = True`
([Declare, then go live](#declare-then-go-live)). Every value an agent sends
is validated against the schema it was shown before any app code runs.

## Functions

```python
from typing import Annotated

from trame.app import get_server

from trame_mcp import get_mcp

server = get_server()
mcp = get_mcp(server)

server.state.resolution = 6
server.state.opacity = 1.0


@mcp.tool
def set_opacity(
    opacity: Annotated[float, "0 = transparent, 1 = opaque", {"minimum": 0, "maximum": 1}],
) -> str:
    """Set the opacity of the cone."""
    with server.state as state:
        state.opacity = opacity
    return f"Opacity set to {opacity}"


@mcp.tool(read_only=True)  # advertised as safe: clients may skip confirmation
def get_settings() -> dict:
    """Current resolution and opacity."""
    return {"resolution": server.state.resolution, "opacity": server.state.opacity}


@mcp.resource("app://help", mime_type="text/markdown")
def help_page() -> str:
    """How to use this application."""
    return "# Cone viewer\nChange the resolution and the opacity of the cone."


@mcp.prompt
def smooth_cone(level: str = "very") -> str:
    """Ask the agent to smooth the cone."""
    return f"Make the cone {level} smooth without making it transparent."
```

- The name comes from the function and the description from its docstring
  (override with `name=` / `description=`).
- The input schema comes from the type hints; `Annotated` adds a description
  (a string) and JSON Schema constraints (a dict).
- `str`, `dict` and `list` results become text (dicts also as structured
  content); return `trame_mcp.Image(png_bytes)` to send an image.
- Functions may be sync or `async`; all run on trame's event loop, inside
  `with state:`, so what they assign reaches browsers (and `@state.change`
  reactions run) when they return.
- Calls run one at a time (`serialize_tools`). A tool that only reads or
  waits (a status poll, a wait for a job) should say
  `@mcp.tool(serialize=False)`: it then runs alongside the other calls
  instead of holding them back, a cancel included, for as long as it waits.
- A module that has no server at hand can mark its functions with `@tool`
  instead; the app then exposes them with `mcp.expose(fn, ...)`.

## Methods of app classes and components

```python
from typing import Literal

from trame_mcp import MCPApp, MCPComponent, resource, tool


class Viewer(MCPComponent):  # may ship in a library: it never turns MCP on
    @tool
    def set_background(self, color: Literal["black", "white", "gray"]) -> str:
        """Set the background color of the 3D view."""
        with self.state:
            self.state.background = color
        return f"Background set to {color}"


class ConeApp(MCPApp):
    def __init__(self, server=None):
        super().__init__(server)
        self.state.resolution = 6
        self.viewer = Viewer(self.server, mcp_prefix="viewer_")
        self.mcp.configure(name="cone-app", enabled=True)

    @tool
    def set_resolution(self, resolution: int) -> str:
        """Set the number of facets of the cone."""
        with self.state:
            self.state.resolution = resolution
        return f"Resolution set to {resolution}"

    @resource("app://settings", mime_type="application/json")
    def settings(self) -> dict:
        """Current view settings."""
        return {"resolution": self.state.resolution}


if __name__ == "__main__":
    ConeApp().server.start()
```

`MCPApp` and `MCPComponent` are trame's `TrameApp` and `TrameComponent`,
plus one thing: when built, they expose their own marked methods, the way
trame binds `@change` and `@trigger`. The agent sees `set_resolution`,
`viewer_set_background` and the `app://settings` resource. `self.mcp` is the
object `get_mcp(self.server)` returns; use it once `super().__init__()` has
run.

- `MCPMixin` does the same for an app that already derives from another
  `TrameApp` subclass: `class App(MCPMixin, TheirApp)`.
- Any other object with marked methods is exposed explicitly:
  `mcp.expose(obj, prefix="obj_")`.

## Plain trame state

```python
from trame.app import get_server

from trame_mcp import get_mcp

server = get_server()
server.state.resolution = 6
server.state.color = "red"

mcp = get_mcp(server)
mcp.state.expose(
    "resolution",
    description="Number of facets of the cone",
    type=int,
    schema={"minimum": 3, "maximum": 60},
)
mcp.state.expose("color", read_only=True, description="Cone color, a CSS color name")
```

The agent reads exposed variables with `state_get` and assigns the writable
ones with `state_update`. Updates are all-or-nothing, checked against `type`
and `schema`, and applied inside `with state:`, so `@state.change` reactions
fire and browsers update. `mcp.state.unexpose("color")` withdraws a variable
at any time.

## trame-dataclass models

Two declarations work together: **models** say which classes the agent may
see, and which of their fields and methods; **instances** say where it
starts. Everything reachable from an exposed instance through exposed fields
is then addressable, including instances created later.

Models you own derive from `MCPStateDataModel`, the agent-visible
`StateDataModel`, and mark their fields and methods. This needs no server:

```python
from trame_dataclass.v2 import ServerOnly, Sync

from trame_mcp import MCPStateDataModel, mcp_field, tool


class Person(MCPStateDataModel):
    name = mcp_field(Sync(str, ""), "Full name", schema={"minLength": 1})
    age = mcp_field(Sync(int, 0), "Age in years", schema={"minimum": 0})
    notes = ServerOnly(str)  # not marked: invisible to the agent

    @tool
    def celebrate_birthday(self) -> str:
        """Increment this person's age by one year."""
        self.age += 1
        return f"{self.name} is now {self.age}"


class AddressBook(
    MCPStateDataModel, all_fields=True, all_tools=True, tools={"export_csv": {"hidden": True}}
):
    title = Sync(str, "My contacts")
    contacts = Sync(list[Person], list, has_dataclass=True)

    def add_contact(self, name: str, age: int = 0) -> dict:
        """Create a contact and add it to the book."""
        person = Person(self.server, name=name, age=age)
        self.contacts = [*self.contacts, person]
        return {"id": person._id}

    def export_csv(self, path: str) -> None:  # public, but hidden from the agent
        ...
```

- `mcp_field(descriptor, description, read_only=, hidden=, schema=)` opts a
  field in. Whether the agent may write it follows the field's mode (`Sync`
  writable, `ClientOnly` read-only; a `ServerOnly` field needs an explicit
  `read_only=`), and `schema` constraints are enforced on every write.
- `@tool` opts a method in; `@tool(read_only=True)` keeps it callable on
  read-only instances, and `@tool(serialize=False)` lets `dataclass_call` run
  it while other calls are in progress.
- A parameter annotated with a model class (`person: Person`, also
  `Person | None` and `list[Person]`) takes the instance's id; trame-mcp
  passes the instance, after checking it is reachable and of that class. The
  same holds for `@mcp.tool` functions. A returned instance reaches the agent
  as `dataclass_get` shows it.
- `@constructor` marks the classmethod agents create instances with (see
  [Several steps in one call](#several-steps-in-one-call)).
- Class keywords set the options: `all_fields=True` / `all_tools=True` opt
  in every field by its mode and every public method. `fields=` / `tools=`
  list names, or map them to the same options as `mcp_field` / `@tool`;
  `{"hidden": True}` excludes one.
- Subclasses inherit the options. Keywords on a subclass replace
  `all_fields` / `all_tools` and add `fields` / `tools` entries.

The application then exposes, on its server, the models it doesn't own and
the instances the agent starts from:

```python
from trame.app import get_server
from trame_dataclass.v2 import StateDataModel, Sync

from trame_mcp import get_mcp


class Camera(StateDataModel):  # stands in for a model from a library
    position = Sync(list[float], lambda: [0.0, 0.0, 1.0])
    focal_point = Sync(list[float], lambda: [0.0, 0.0, 0.0])
    cache_key = Sync(str, "")

    def reset(self) -> None:
        """Look at the origin from the default position."""
        self.position = [0.0, 0.0, 1.0]
        self.focal_point = [0.0, 0.0, 0.0]


server = get_server()
mcp = get_mcp(server)

mcp.dataclass.expose_model(
    Camera,
    fields={
        "position": {"description": "[x, y, z]", "schema": {"minItems": 3, "maxItems": 3}},
        "focal_point": {"read_only": True},
    },
    tools=["reset"],
)

# Person and AddressBook as defined above
book = AddressBook(server, contacts=[Person(server, name="Ada", age=36)])
camera = Camera(server)
presets = {"front": Camera(server), "top": Camera(server, position=[0.0, 1.0, 0.0])}

mcp.dataclass.expose_instance(book, name="address_book")
mcp.dataclass.expose_instance(lambda: camera, name="camera")  # looked up on every call
mcp.dataclass.expose_instance(presets, name="presets", read_only=True)
```

- `expose_model()` takes the same options as the `MCPStateDataModel` class
  keywords, once per class; a plain list works too
  (`fields=["position", "focal_point"]`). It also overrides the options of an
  `MCPStateDataModel` subclass, so the app has the final word.
- An instance source is an instance, a list or dict of instances, or a
  function evaluated on every call, which keeps following a variable or
  attribute that gets reassigned. `read_only=True` allows reading only, plus
  `@tool(read_only=True)` methods. `mcp.dataclass.unexpose_instance(name)`
  withdraws an entry point.

The agent then works with five generic tools:

```
dataclass_schemas()                                      # models, fields, methods, entry points
dataclass_get("address_book", depth=2)                   # values, nested instances inlined
dataclass_call("address_book", "add_contact", {"name": "Grace", "age": 36})
dataclass_update("<person id>", {"age": 37})             # validated, all-or-nothing
dataclass_batch([...])                                   # several steps in one call
```

`dataclass_update` and `dataclass_call` return once the instance's
`@watch`ers have run, so what they derive is in the agent's next read.

### Several steps in one call

`dataclass_batch(steps)` runs a list of steps in order, in one call: one turn
of the tool lock and one state flush for the whole list. A step creates an
instance (`new`), calls a method (`call`) or updates fields (`update`), and a
later step refers to an earlier one's result by the name it gave with `as`:

```python
class Person(MCPStateDataModel):
    name = mcp_field(Sync(str, ""))

    @constructor(description="A new person, in no address book yet")
    @classmethod
    def new(cls, server, name: str) -> "Person":
        return cls(server, name=name)


class AddressBook(MCPStateDataModel):
    contacts = mcp_field(Sync(list[Person], list, has_dataclass=True))

    @tool
    def add(self, person: Person) -> None:  # takes the person's id
        """Add a person to the book."""
        self.contacts = [*self.contacts, person]
```

```
dataclass_batch([
  {"new": "Person", "args": {"name": "Grace"}, "as": "grace"},
  {"update": "@grace", "values": {"name": "Grace Hopper"}},
  {"call": "address_book", "method": "add", "args": {"person": "@grace"}},
])
→ {"results": [{"id": "31", "type": "Person", "values": {}}, {"id": "31", "updated": ["name"]}, null],
   "unattached": []}
```

- **Compact results.** An instance in the results shows its id and its
  references to other instances (as ids, what later calls link with), not
  its other fields, and only its id once an earlier step showed it: a batch
  that builds and adds a dozen models stays small. `results: "full"` shows
  each instance as `dataclass_get` does.

- **Constructors.** `@constructor` marks one classmethod per class (above or
  below `@classmethod`); trame-mcp calls it with the server first, which is
  not part of its schema, and subclasses inherit it. `expose_model(cls,
  constructor="create")` names one for a class you don't own, `False` hides
  one. `dataclass_schemas` lists each model's constructor.
- **Built instances live for the batch.** What a `new` step builds is
  reachable by the following steps only, until one attaches it somewhere an
  entry point reaches. One never attached is dropped and listed in
  `unattached`: nothing is added implicitly, and trame-mcp keeps no reference.
- **References.** `"@grace"` is the result named `grace`; `"@grace.ports.0"`
  follows a field of an instance, a key of an object or an index of a list.
  Instances in `args` travel by id. `"@@"` writes a literal leading `@`.
- **Checked first, no rollback.** Step shapes, names and references are
  checked before any step runs. The batch stops at the first failing step,
  and the error names it, gives the reason and lists the results of the steps
  before it, which stay applied.

## What the agent reads

Everything the agent knows about the app comes from what you declare:

- **On connection**, the server sends instructions: your `mcp.instructions`
  text, if any, then a generated overview. It says the app is live and shared
  with the people using it, and where to start (data entry points, state).
- **Tools, resources and prompts** are described by their docstrings.
  Parameters, prompt arguments included, take descriptions and constraints
  from `Annotated[T, "description", {...}]`.
- **trame-dataclass models** are described by `dataclass_schemas()`: class
  docstrings, `mcp_field` descriptions and schemas, method docstrings and
  parameters. A subclass lists only what it adds or changes, under
  `"extends": "BaseModel"`.

Docstrings reach the agent verbatim. When one is written for developers, give
the agent its own text with `description=`: on `@mcp.tool` / `@tool`, as an
`MCPStateDataModel` class keyword or an `expose_model()` option, and in
`fields=` / `tools=` entries. A model's description, like its docstring, is
not inherited by its subclasses.

## Declare, then go live

Everything above only *declares* what an agent could use. None of it is
reachable until agent access is turned on, so an app, or a library it uses,
can carry its whole MCP surface and still ship with it off.

- **One switch goes live**: `mcp.enabled` (default `False`) covers every
  agent channel, the HTTP endpoint and the in-process chat (`LLMAdapter`)
  alike. Set it in code, flip it at runtime, or let the end user decide with
  `--mcp` / `--no-mcp`.
- **The command line has the last word at launch**: before the server
  starts, values set in code are defaults that `--mcp` (and `--mcp-token`,
  `--mcp-log`) override. Once it is running, every assignment applies.
- **What the command line withholds stays withheld**: `--no-mcp` and
  `--mcp-deny` hold for the whole run, whatever the code or a UI switch
  assigns later. `mcp.enabled_locked` and `mcp.permissions.locked` tell a UI
  which switches to disable.

```python
from trame.app import get_server

from trame_mcp import get_mcp

server = get_server()
mcp = get_mcp(server)
mcp.enabled = True  # a default: --no-mcp keeps it off for the whole run


@server.state.change("agent_access")  # e.g. bound to a checkbox in the UI
def on_agent_access(agent_access, **_):
    mcp.enabled = agent_access
```

|                         | `enabled=False` (default)       | `enabled=True` |
| ----------------------- | ------------------------------- | -------------- |
| HTTP endpoint           | answers 503                     | served         |
| HTTP with `http=False`  | no route                        | no route       |
| In-process `LLMAdapter` | calls raise `MCPDisabledError`  | served         |

`http=False` is the structural choice, fixed once the server starts: no route
is ever mounted, for an app whose embedded chat is the only agent. Otherwise
the route is mounted even while access is off, so it can be switched on later;
the connection banner prints when access first goes live.

## Letting users withhold powers

Some tools hand an agent more than the app itself: a path anywhere on disk,
code to run. Declare each such power as a permission and name it where it is
needed; the people running the app can then withhold it, at launch or from
your UI.

```python
mcp.permissions.declare("files", "Read files and folders on this computer")
mcp.permissions.declare("scripts", "Run Python code the agent writes", implies="files")


class Viewer(MCPApp):
    @tool(requires="files", read_only=True)
    def list_files(self, path: str) -> list[str]: ...


class Node(MCPStateDataModel):
    @constructor(requires="scripts")
    @classmethod
    def from_script(cls, server, script: str) -> "Node": ...

    @tool
    def set_executor(self, env_path: str = ""):
        if env_path:  # a need that depends on the arguments
            get_mcp(self.server).permissions.require("files", what="set_executor(env_path)")
```

- `requires=` takes a name or a list, on `@tool`, `@mcp.tool` and
  `@constructor`, and in `expose_model()`'s `tools=` and `constructor=`
  options.
- While a permission is denied, what needs it is left out of `tools/list`
  and `dataclass_schemas`, and the instructions sent on connection say it is
  turned off. A client that listed the tools earlier keeps its list (the
  server cannot tell it the list changed), so calls are refused too, with
  `PermissionDeniedError`: the agent reads which permission is off and that
  only the people running the app can turn it on.
- A `dataclass_batch` checks its steps before running any, so a refused step
  leaves nothing half-done. Only a step whose target is known once earlier
  steps ran (`"@w.nodes.0"`) is checked when it runs.
- `implies` says a permission grants another anyway (a script can read any
  file): allowing it allows the ones it implies, and denying one denies the
  ones implying it. A UI reads the permissions back after a change.
- `mcp.permissions` maps each name to whether it is allowed:
  `permissions["scripts"] = False`, `permissions.update(files=True)`,
  `permissions.denied`.
- `--mcp-deny files,scripts` (repeatable) or `TRAME_MCP_DENY` deny
  permissions for the whole run, with those implying them:
  `permissions.locked` lists them, and allowing one is ignored. A name the
  app does not declare refuses every agent call once the server runs: a typo
  must not leave a power allowed.

The call log shows a refused call as `denied`.

## Options

`get_mcp(server, **options)` and `mcp.configure(**options)` set several
options at once; each is also an attribute (`mcp.name = "cone-app"`).

| Option            | Default               | Meaning                                                              |
| ----------------- | --------------------- | -------------------------------------------------------------------- |
| `enabled`         | `False`               | agent access switch; `--mcp` / `--no-mcp` at launch                  |
| `name`            | the trame server name | server name reported to MCP clients                                  |
| `instructions`    | `None`                | your guidance for MCP clients, sent before the generated overview    |
| `token`           | `None`                | bearer token required on the endpoint; `--mcp-token`, `TRAME_MCP_TOKEN` |
| `allowed_origins` | `()`                  | extra browser origins allowed to call the endpoint                   |
| `http`            | `True`                | mount the HTTP endpoint; `False` for in-process use only             |
| `path`            | `"/mcp"`              | endpoint path on the trame web server                                |
| `serialize_tools` | `True`                | run tool calls one at a time (but `@tool(serialize=False)` ones)     |
| `log`             | `"off"`               | log what agents call: `"calls"` or `"all"`; `--mcp-log`, `TRAME_MCP_LOG` |

`http` and `path` are fixed once the server starts; the others can change at
any time.

## Long calls and cancellation

A client that gives up on a call (its user interrupted it, or its own
timeout ran out) stops it: on `notifications/cancelled`, or when its HTTP
connection closes, the tool is cancelled at its next `await` and the tool
lock is released. Long waits are best kept short anyway, returning a status
the agent can poll, and marked `serialize=False`: clients cut off a call
that sends nothing back for a few minutes (Claude Code's default idle
timeout for HTTP servers is 5 minutes), and the server cannot send progress
over its plain JSON responses. A cancelled `dataclass_batch` stops between
steps; the ones that ran stay applied.

## Seeing what agents call

`--mcp-log` (or `mcp.log = "calls"`, any time) logs every tool call,
resource read and prompt when it ends, whether it came over HTTP or
in-process through `LLMAdapter`:

```
14:02:11 mcp[3f2aQx] dataclass_batch new TransformNodeModel as t; call pipeline.add_node -> ok 0.21s (2.1 KB)
14:02:41 mcp[3f2aQx] pipeline_status {wait: true, since: 12} -> ok 30.02s (1.4 KB, queued 0.3s)
14:02:42 mcp[3f2aQx] dataclass_call 6x.set_parameters {values: {n: 256}} -> error 0.01s: 'n' must be at most 128, got 256
14:02:43 mcp[in-process] screenshot -> ok 0.40s (image, 84.2 KB)
```

- The caller is the start of the client's `Mcp-Session-Id` (`-` without
  one), or `in-process`.
- Arguments are shortened (long strings become `<str 2134 chars>`); the
  generic `dataclass_*` tools summarize themselves (a batch lists its
  steps, a call reads `target.method`).
- `queued` is the time spent waiting for the tool lock behind other calls.
- `--mcp-log all` (`mcp.log = "all"`) also logs each call as it starts, with
  its full arguments and the request's `_meta`, and every other protocol
  message (`initialize`, the lists, notifications).

The lines go to the `trame_mcp.calls` logger. Turning the log on gives it a
stderr handler unless the application has given it one of its own, so it
prints without any logging setup; add a handler to send the lines elsewhere.

## Endpoint security

- Browser pages from other sites cannot reach the endpoint (Origin, Host and
  Content-Type checks), so the default tokenless setup is safe on localhost.
- `token=` / `--mcp-token` / `TRAME_MCP_TOKEN` require a bearer token; when the
  server listens beyond localhost without one, a token is generated and
  printed with the connection banner.

## Using with trame-llm

The same `mcp` can feed [trame-llm](https://github.com/Kitware/trame-llm)'s
embedded chat widget, in-process, through `LLMAdapter`:

```python
from trame.app import get_server
from trame_llm.core import LLMClient

from trame_mcp import LLMAdapter, get_mcp

mcp = get_mcp(get_server())
llm_client = LLMClient()
llm_client.register_mcp(LLMAdapter(mcp))  # after everything is declared
```

`mcp.enabled` applies to the chat too. While it is off, the chat still lists
the tools (trame-llm reads them once at startup), but every call fails with
"MCP access is currently disabled by the application". For a chat-only
application that should not accept outside MCP connections, pass
`http=False` to `get_mcp()` (or `self.mcp.configure(http=False)`).

## Examples

- `examples/00_quickstart.py` — a function tool and plain state.
- `examples/01_class_based.py` — `MCPApp`; starts with agent access off and
  toggles it from the UI.
- `examples/02_address_book.py` — trame-dataclass models and instances.
- `examples/03_wasm_quadrants.py` — nested models, inheritance,
  `all_fields` / `all_tools`, VTK scenes.

See `DESIGN.md` for the full design document.

## License

This library is OpenSource and follows the Apache Software License.
