# trame-mcp — Design

Add an MCP (Model Context Protocol) server to any trame application, so LLM
clients (Claude Code, Claude Desktop, etc.) can inspect and drive the app while
human users watch it update live in the browser.

## Goals

- One line to attach an MCP server to a trame app, served on the **same port**
  as the app itself (no second server, no second process).
- Declarations that read like the rest of trame: server-bound decorators for
  plain functions (`@mcp.tool`, like `@state.change`), agent-aware variants
  of trame's base classes (`MCPApp`, `MCPStateDataModel`, like `TrameApp`,
  `StateDataModel`), inert markers for methods and fields (`@tool`, like
  `@change`), explicit calls for what you don't own, then one switch to go
  live. It uses trame-dataclass's own vocabulary.
- Safe-by-default interplay with trame state: tool calls run on the server's
  event loop and state mutations are flushed to connected clients immediately.
- First-class support for `trame-dataclass` models: typed schemas, field-mode
  aware permissions, and dynamic instance addressing come from the descriptors.
- No magic: every tool, state variable, and model is opted in explicitly by
  the author.
- Declaring is not serving: an app, or a library it uses, can carry its whole
  MCP surface while agent access stays off until the app turns it on.

## Non-goals (v1)

- Multi-user session routing (trame launcher deployments) — one MCP server per
  app process, which is trame's natural unit anyway.
- Server→client MCP features (sampling, elicitation) — possible later.

## Public API surface

```python
from trame_mcp import (
    get_mcp,                                    # the server's MCP
    tool, resource, prompt, mcp_field,          # markers
    constructor,                                # a model's agent constructor
    MCPApp, MCPComponent, MCPMixin,             # agent-aware trame classes
    MCPStateDataModel,
    Image, LLMAdapter, MCPDisabledError, PermissionDeniedError,
)
```

The decorators follow trame's own split between plain functions and
methods:

| trame | trame-mcp | Used on |
| --- | --- | --- |
| `@state.change(...)`, `@ctrl.set(...)` | `@mcp.tool`, `@mcp.resource(uri)`, `@mcp.prompt` | plain functions: exposed on that server immediately |
| `@change(...)`, `@controller.set(...)` | `@tool`, `@resource(uri)`, `@prompt` | methods: inert markers, bound when the object is built |

Markers (`tool`, `resource`, `prompt`, `mcp_field`, `constructor`) only attach metadata,
need no server, and can live in a library; so can the base classes, which
pair with trame's (`MCPApp` / `TrameApp`, `MCPComponent` / `TrameComponent`,
`MCPStateDataModel` / `StateDataModel`). Everything else
goes through `mcp`, `get_mcp(server=None, **config)`, the one MCP bound to a
trame server (mirrors `get_server()`: one per trame server):

| Call | Declares |
| --- | --- |
| `@mcp.tool` / `@mcp.resource(uri)` / `@mcp.prompt` | a plain function |
| `mcp.expose(*objs_or_fns, prefix="")` | marked methods of other objects (or marked functions) |
| `mcp.state.expose(*names, ...)` / `.unexpose(*names)` | plain trame state variables |
| `mcp.dataclass.expose_model(cls, ...)` | a trame-dataclass class you don't own |
| `mcp.dataclass.expose_instance(src, name)` / `.unexpose_instance(name)` | trame-dataclass entry points |
| `mcp.permissions.declare(name, description, implies=)` | a power users can withhold (`requires=` on markers) |
| `mcp.enabled = True` | goes live (nothing is served before) |

Layers with their own generic tools are grouped by the trame module they
cover (`mcp.state` ↔ `state_*` tools, `mcp.dataclass` ↔ `dataclass_*`
tools); `mcp.state` is the MCP's view of state, not trame's `server.state`.

---

## Quick start (functional style)

```python
from trame.app import get_server
from trame_mcp import get_mcp

server = get_server()
state, ctrl = server.state, server.controller
mcp = get_mcp(server)   # mounts /mcp via on_server_bind; access is off

state.resolution = 6

@mcp.tool               # exposed on this server, like @state.change
def set_resolution(resolution: int) -> str:
    """Set the number of facets of the cone (3-60)."""
    with state:
        state.resolution = resolution
    return f"Resolution set to {resolution}"

@mcp.tool
def reset_camera() -> str:
    """Reset the 3D camera to fit the scene."""
    ctrl.reset_camera()
    return "Camera reset"

mcp.enabled = True      # go live (--no-mcp at launch still overrides)
server.start()
```

Once access is live, the log tells the developer where to point their client:

```
MCP server available at http://localhost:8080/mcp
  claude mcp add --transport http my-app http://localhost:8080/mcp
```

### Decorator contract

- `@mcp.tool` (plain functions) and `@tool` (methods) take the same options,
  bare or `(name="set_resolution", description="...", read_only=True)`.
- `mcp.expose(...)` takes objects whose class has marked methods (`prefix=`
  namespaces tool and prompt names), or functions marked with `@tool` in a
  module that has no server at hand. An unmarked function, an object without
  marked methods, a class, or a trame-dataclass instance
  (→ `mcp.dataclass.expose_instance()`) raises.
- Tool name defaults to the function name, description to the docstring.
- `read_only=True` is advertised as the MCP `readOnlyHint` annotation (clients
  may skip confirmation for it); `annotations={...}` passes any other MCP hint
  (`destructiveHint`, `idempotentHint`, `openWorldHint`, `title`).
- `serialize=False` takes the tool out of the one-at-a-time rule
  (`serialize_tools`): its calls run while other calls are in progress,
  interleaving with them at `await` points. For read-only and waiting tools
  (a status poll, a wait for a job to finish), which would otherwise hold
  back every other call, a cancel included. On a model method it applies to
  `dataclass_call`, which takes the lock per method.
- The JSON input schema is derived from type hints (plain types, `Literal`,
  fixed-length tuples, defaults → optional params, a `None` default → nullable).
  `Annotated` metadata adds a description (`str`) or constraints (`dict`):
  `resolution: Annotated[int, "Number of facets", {"minimum": 3, "maximum": 60}]`.
- **Arguments are validated** against that schema before the call (types,
  ranges, enums, unknown/missing arguments), with lossless numeric coercion
  (`3.0` → `3` for an `int`, `3` → `3.0` for a `float`). The model gets an
  error naming the offending argument; the function body never sees
  ill-typed input, and gets the Python shape its hints declare (JSON has no
  tuples: an array comes back as a tuple where the hint says `tuple[...]`;
  model field writes get the same treatment from the field's type). Return values are coerced to MCP content (str/dict/list →
  text or structured content, `trame_mcp.Image` → image content).
- Sync and `async def` functions both work. Everything runs on trame's asyncio
  loop, inside `with state:`: what a tool assigns to the state is pushed to
  browsers, and its `@state.change` reactions run, when it returns (an HTTP
  request never flushes trame's state by itself). Blocking work should use
  async + executor as usual in trame.
- Same pattern for `@mcp.resource(uri)` / `@resource(uri)` and
  `@mcp.prompt` / `@prompt`.

## Class-based apps (modern `trame.app.TrameApp` pattern)

`MCPApp` subclasses `trame.app.TrameApp` and extends its
`_bind_annotated_methods()` step, so it exposes its own marked methods at
construction (`self.mcp.expose(self)`), exactly like `@change` / `@trigger` /
`@controller.add` are bound:

```python
from trame.decorators import change
from trame_mcp import MCPApp, tool

class ConeApp(MCPApp):
    def __init__(self, server=None):
        super().__init__(server)          # tools exposed, self.mcp available
        self.mcp.configure(name="cone-app", enabled=True)
        self._build_ui()

    @tool
    def set_resolution(self, resolution: int) -> str:
        """Set the number of facets of the cone (3-60)."""
        with self.state as state:
            state.resolution = resolution
        return f"Resolution set to {resolution}"

    @change("resolution")
    def _on_resolution(self, resolution, **_):
        ...
```

Structure of the provided classes:

```python
class MCPMixin:
    """Exposes the marked methods of any TrameComponent subclass."""
    def _bind_annotated_methods(self):
        super()._bind_annotated_methods()
        # expose(self), tolerating classes without marked methods
        get_mcp(self.server)._expose_object(self, prefix=self._mcp_prefix)

class MCPApp(MCPMixin, TrameApp): ...
class MCPComponent(MCPMixin, TrameComponent): ...
```

- **`MCPApp`** — the common case: your app class, MCP included.
- **`MCPComponent`** — sub-components contributing their own tools, with an
  optional `mcp_prefix` constructor argument to namespace them. Like
  everything else it only declares, so a library can build on it: an app
  using such a component serves nothing until it sets `mcp.enabled`.
- **`MCPMixin`** — for apps that already derive from another `TrameApp`
  subclass: `class MyApp(MCPMixin, TheirAppBase)`. Works without cooperative
  `__init__` because binding rides on `_bind_annotated_methods()`.
- Any other object with marked methods: `mcp.expose(obj, prefix=...)`.

MCP is configured the same way in every style: through `self.mcp`
(`configure()` or attributes) once `super().__init__()` has run. The
constructors take no MCP options; stray `mcp_*` keyword arguments raise
instead of being swallowed by `TrameApp`'s `**kwargs`.

## Exposing plain state

Most trame apps are state-driven, so the common case should not require
writing a tool per variable:

```python
mcp.state.expose("resolution")
mcp.state.expose("color", read_only=True)
```

This registers two generic tools backed by an allowlist:

- `state_get(names: list[str] | None)` → current values of exposed variables
- `state_update(updates: dict)` → assign writable variables (inside
  `with state:` so `@state.change` reactions fire and browsers update);
  registered with the first writable variable, so a read-only exposure does
  not advertise a tool that can never succeed

Rationale: a get/update pair keeps the tool list small (LLM context economy)
regardless of how many variables are exposed; the allowlist keeps internal
state (`trame__*`, private vars) unreachable. Descriptions, value types and
constraints help the model and are enforced on writes (`state_update` is
all-or-nothing):

```python
mcp.state.expose(
    "resolution",
    description="Number of facets of the cone",
    type=int,
    schema={"minimum": 3, "maximum": 60},
)
```

Exposed variables are writable, as trame state is from the browser, unless
`read_only=True` (a plain bool: call `expose()` once per access level). The
writable variables' schemas are part of `state_update`'s input schema, so
the model sees exact types up front. Without `type`/`schema` any JSON value
is accepted. `mcp.state.unexpose(*names)` withdraws variables at runtime.

## First-class `trame-dataclass` support

`trame-dataclass` is the preferred way to manage dynamic, typed pieces of
state, and its descriptors already carry most of what MCP needs:

- **Typed schemas for free** — field types and defaults come from the
  descriptors (`Sync(int, 1)`); a `None` default makes the field nullable.
- **Access from the descriptor mode** — `Sync` is read-write, `ClientOnly`
  read-only (the server seeds it, doesn't read it back); `ServerOnly` has no
  shareable default (private or heavy server data, often not JSON).
- **Validated writes** — every MCP write is checked against the field schema
  before anything is applied. trame-dataclass itself only *warns* on a type
  mismatch by default, so this check is what keeps a string out of an `int`
  field.
- **Full reactivity** — MCP writes go through the descriptors, so `@watch`ers
  and client synchronization fire exactly as if the server had mutated the
  field. trame-dataclass runs watchers in a task, after the write; so that
  the result and the agent's next read reflect what they derive,
  `dataclass_update` and `dataclass_call` return once the target instance's
  watchers have run (`await instance.completion()`, trame-dataclass >= 2.2).
- **Dynamic instances** — every `StateDataModel` has an `_id` in the instance
  registry; nested instances (`Sync(list[Person], has_dataclass=True)`) are
  reported as `{"id": ..., "type": "Person"}` references the agent can follow
  — the same encoding the browser client uses.

Exposure takes two declarations, both under `mcp.dataclass`, using
trame-dataclass's own vocabulary (a *model* is the class, an *instance* the
object):

- **Models** say which classes the agent may see, and which of their fields
  and methods: `MCPStateDataModel` subclasses you own, `expose_model()` for
  any other class.
- **Instances** say where the agent starts: `expose_instance(source, name)`.

Neither works alone: exposed models without an exposed instance give the
agent nothing to start from, and an instance of an unexposed class is
invisible.

### Exposing models you own

`MCPStateDataModel` is to `StateDataModel` what `MCPApp` is to `TrameApp`:
deriving from it makes a class visible to agents, and class keywords carry
the options.

```python
from trame_mcp import MCPStateDataModel, mcp_field, tool

class Person(MCPStateDataModel):
    """A contact in the address book."""

    name = mcp_field(Sync(str, ""), "Full name", schema={"minLength": 1})
    age = mcp_field(Sync(int, 0), schema={"minimum": 0})
    nickname = Sync(str, "")                      # not marked: not exposed
    score = mcp_field(ServerOnly(int, 0), read_only=True)

    @tool
    def celebrate_birthday(self) -> str:
        """Increment this person's age by one year."""
        ...
```

- **Fields are opt-in.** A field is visible when it is marked with
  `mcp_field()` or listed in `fields=`. The `all_fields=True` class keyword
  also exposes every unmarked field by its mode (`ServerOnly` stays hidden).
- **Methods are opt-in.** A method is callable when it is marked with
  `@tool` or listed in `tools=`. `all_tools=True` also exposes every public
  method: no leading underscore, not part of `StateDataModel`'s
  own API (`update`, `flush`, `update_from_client_state`, ...), not a
  classmethod/staticmethod/property, not a `@watch` callback.
- **Marking opts in; access follows the mode** unless `read_only=` says
  otherwise. A `ServerOnly` field has no mode default, so marking or listing
  one requires `read_only=True` or `False`.
- `mcp_field(descriptor, description, read_only=, hidden=, schema=)` returns
  the descriptor unchanged. `schema` holds JSON Schema keywords merged into
  the type-derived schema, advertised in `dataclass_schemas` and enforced on
  every MCP write. `hidden=True` keeps a field out even under `all_fields`.
- The class keywords are the same `all_fields=` / `all_tools=` / `fields=` /
  `tools=` as `expose_model()`, checked when the class is created; a
  `tools={"name": {"hidden": True}}` entry excludes a method `all_tools` would
  expose (example 03 hides its scenes' lifecycle hooks this way).
- **Inheritance.** Subclasses inherit the options; keywords on a subclass
  replace `all_fields` / `all_tools` and add `fields` / `tools` entries. A
  subclass that redeclares a field (to change its default) keeps the
  parent's marker unless it marks the field again.
- At runtime an `MCPStateDataModel` is a plain `StateDataModel`: nothing
  happens at construction, since reachability from exposed instances decides
  what an agent sees.
- **Reference fields need opting in like any other field**, otherwise the
  agent cannot walk through them. A startup audit logs a warning when an
  unexposed reference field holds exposed models, and when an exposed one
  can only hold unexposed models.
- Hidden fields are indistinguishable from nonexistent ones over MCP, and a
  hidden reference field is not followed for reachability.
- Constraints guard MCP writes only; app code (constructors, methods) sets
  fields freely. Model methods validate their own arguments through their
  signatures (`Annotated` constraints).

### Exposing models you don't own

`expose_model()` exposes any class, typically one from a library, with the
same options as the `MCPStateDataModel` class keywords. `fields=` and `tools=` take a list of names,
or a dict whose values carry the `mcp_field` / `@tool` options:

```python
mcp.dataclass.expose_model(lib.Camera, fields=["position", "focal_point"])
mcp.dataclass.expose_model(
    lib.Light,
    fields={
        "color": {"description": "RGB, 0-1 each", "schema": {"maxItems": 3}},
        "intensity": {"read_only": True},
    },
    tools={"switch_off": {"description": "Turn the light off"}},
)
mcp.dataclass.expose_model(lib.Grid, all_fields=True, fields={"cache": {"hidden": True}})
```

- One call per class: exposing it twice raises, so its whole policy reads in
  one place.
- It also overrides how an `MCPStateDataModel` subclass is exposed in this
  application (the app has the final word). Markers on the class still opt fields in;
  the dict entries override them key by key, and `{"hidden": True}` removes
  a field or a method.
- Precedence: the nearest class passed to `expose_model()` along the MRO,
  otherwise the class's own `MCPStateDataModel` options.
- `fields=` and `tools=` names are checked against the class when declared.

### Exposing instances

```python
class ContactsApp(MCPApp):
    def __init__(self, server=None):
        super().__init__(server)
        self.book = AddressBook(self.server)
        self.mcp.dataclass.expose_instance(self.book, name="address_book")
        self.mcp.dataclass.expose_instance(lambda: self.selection, name="selection")
        self.mcp.dataclass.expose_instance(self.settings, name="settings", read_only=True)
```

`source` is a model instance, a list/dict of instances, or a zero-argument
callable returning either, evaluated on every call: `lambda: self.workspace`
keeps working when the attribute is replaced, and models kept in a plain
Python list need no wrapper model. A direct instance stays exposed until
`unexpose_instance(name)`.

Dynamically created instances are never exposed one by one — anything
reachable from an exposed instance through visible reference fields, and of
an exposed class, is addressable by its `_id`. One
`expose_instance(self.book, name="address_book")` therefore covers every
`Person` that will ever live inside it. Anything else stays invisible
regardless of the global instance registry — reachability is the
access-control boundary, checked on every call. Unexposed instances are
filtered out of lists and dicts, and a single reference to one reads `null`.

Why not "every instance of an exposed class": trame-dataclass only keeps a
process-wide weak registry, so that view would include detached objects
still referenced somewhere (undo stacks, reference cycles awaiting
collection), instances of other servers, and would lose the structure the
agent needs to tell which scene a cone belongs to. Reachability answers
"what is attached right now", like the browser, which also only sees
instances bound through synced fields.

`read_only=True` exposes everything reached only through that entry for
reading: `dataclass_update` is refused and only `read_only` methods are
callable. An instance also reachable from a writable entry is writable.

Exposing an instance registers five generic data tools (constant tool count
no matter how many instances or methods exist at runtime):

- `dataclass_schemas()` — the named instances, and every exposed model the
  agent can encounter: those passed to `expose_model()`, those of
  reachable instances, and those declared by visible reference fields, each
  with its exposed subclasses — so a model is described before any instance
  of it exists. Field names, types, constraints, read-only flags,
  callable methods with their signatures, and the constructor, if any. A
  subclass is listed as
  `"extends": "Base"` plus only what it adds or changes (see "What the agent
  reads"). Agent-visible classes the application never reaches are not
  listed.
- `dataclass_get(target, depth=1)` — `target` is an exposed instance name
  (`"address_book"`) or a reachable instance `_id`; returns
  `{"id", "type", "values": {...}}`, with nested instances inside `values`
  as `{"id", "type"}` references. A larger `depth` inlines nested instances
  (each still carrying its `_id`) to fetch a subtree in one call; payloads
  are never unbounded by default. A name exposing a collection returns
  `{"name", "items"}`.
- `dataclass_update(target, updates)` — a targeted patch: `updates` holds only
  the fields being changed (the shape of `values` from `dataclass_get`), all
  owned by the target instance. Every value is validated against its field
  schema; unknown/hidden/read-only fields or invalid values fail the whole
  call before anything is applied, with an error listing the writable fields.
- `dataclass_call(target, method, args)` — invokes an exposed method on the
  target instance (see "Model methods" below).
- `dataclass_batch(steps)` — runs `new` / `call` / `update` steps in order in
  one call, later steps referring to earlier results (see "Constructors and
  batches" below).

### Worked example: updating a deeply nested field

```python
class Address(MCPStateDataModel, all_fields=True):
    street = Sync(str, "")
    city = Sync(str, "")

class Person(MCPStateDataModel, all_fields=True):
    name = Sync(str, "")
    tags = Sync(list[str], list)
    home = Sync(Address, None, has_dataclass=True)

class AddressBook(MCPStateDataModel, all_fields=True):
    title = Sync(str, "My contacts")
    contacts = Sync(list[Person], list, has_dataclass=True)

mcp.dataclass.expose_instance(book, name="address_book")
```

The client fixes the street of Ada's home address:

```
dataclass_get("address_book")
→ {"id": "17", "type": "AddressBook",
   "values": {"title": "My contacts",
              "contacts": [{"id": "18", "type": "Person"},
                           {"id": "21", "type": "Person"}]}}

dataclass_get("18")
→ {"id": "18", "type": "Person",
   "values": {"name": "Ada", "tags": ["friend"],
              "home": {"id": "19", "type": "Address"}}}

dataclass_update("19", {"street": "5 Main St"})
→ {"id": "19", "updated": ["street"]}
```

The refs returned by `dataclass_get` are the update handles: PATCH-by-id, never
PUT-the-document.

### Update semantics

- **Targeted patches, never document writes.** The exposed data never
  round-trips. A whole-tree write would silently clobber concurrent edits the
  browser user made between the LLM's read and write; a patch touches exactly
  the named fields and everything else keeps reacting to the human.
- **Addressing is by id, not by path.** No `"contacts[0].home.street"` syntax:
  indices shift as lists shuffle (including from browser-side edits), and ids
  are the same stable handles the web client already uses.
- **One instance per call.** Fields of different instances are separate
  `dataclass_update` calls; each assignment goes through its descriptor, so
  watchers fire with normal granularity (no diff-and-apply machinery).
- **Whole-value replacement.** `{"tags": ["friend", "vip"]}` replaces the
  list; there is no JSON-patch sub-syntax inside values.
- **Validate, then mutate.** The call is rejected before any field is applied
  if it names unknown, hidden, or read-only fields, or if any value violates
  its field schema — the error lists the writable fields / offending values
  so the client can self-correct.
- **Structural changes are app logic, not `dataclass_update`.** Creating a
  `Person`, inserting into `contacts`, or deleting one involves constructor
  arguments and placement decisions that belong to the app — expose them as
  model methods invoked via `dataclass_call` (e.g. `add_contact` on
  `AddressBook`; see "Model methods" below). `dataclass_update`
  writes plain-data fields only; reference fields are read-only refs (a
  later revision may allow re-pointing to an existing instance by `_id`, but
  never creating through it).

### Model methods

Model methods put behavior next to the data it operates on: an exposed method
on a `StateDataModel` subclass becomes callable over MCP — deliberately
**not** as a top-level named tool, but through the generic `dataclass_call`
data tool.

```python
class AddressBook(MCPStateDataModel, all_fields=True):
    title = Sync(str, "My contacts")
    contacts = Sync(list[Person], list, has_dataclass=True)

    @tool
    def add_contact(self, name: str, age: int = 0) -> dict:
        """Create a new contact and add it to the book."""
        person = Person(self.server, name=name, age=age)
        self.contacts = [*self.contacts, person]
        return {"id": person._id}

    @tool
    def remove_contact(self, contact_id: str) -> str:
        """Remove a contact from the book."""
        self.contacts = [p for p in self.contacts if p._id != contact_id]
        return "removed"
```

With `expose_instance(book, name="address_book")`, a session might look
like:

```
dataclass_schemas()
→ ... AddressBook: fields {title: str},
                   methods {add_contact(name: str, age: int = 0),
                            remove_contact(contact_id: str)}
      Person:      fields {name: str, age: int},
                   methods {celebrate_birthday()}

dataclass_call("address_book", "add_contact", {"name": "Grace", "age": 36})
→ {"id": "23"}

dataclass_call("23", "celebrate_birthday")
→ "Grace is now 37"
```

Why not one named tool per method: the top-level tool list is a flat, global
namespace. Flattening model methods into it forces a naming scheme (derived
class prefixes, or bare names with collision policing) and — with singleton
shortcuts — tool schemas that change shape mid-session when a second
instance becomes reachable. Behind `dataclass_call`, methods stay namespaced by
their class, exactly like fields: constant generic tools, schemas in
`dataclass_schemas`. App-side the option is still called `tools=`, mirroring
`@tool`; agent-side they are *methods*, since they never appear in the MCP
tool list.

Contract:

- **Exposure** — `@tool` (bare or `@tool(name=..., description=...,
  read_only=...)`), a `tools=` entry, or `all_tools=True`; nothing else is
  ever callable. Docstring → description, type hints → argument validation,
  return value coerced like any other tool result (an instance comes back as
  `dataclass_get` shows it, nested ones as references); sync and `async def`
  both run on the server loop. `read_only` methods stay callable on read-only
  exposures; methods exposed by `all_tools` alone are not `read_only`.
- **Instances as arguments** — a parameter annotated with a model class
  (`X`, `X | None`, `list[X]`) is advertised as an id (`{"type": "string",
  "model": "X"}`) and receives the instance: it must be reachable, of class
  `X`, and, when reachable only through read-only entries, go to a
  `read_only` method. Named `@mcp.tool` functions get the same treatment.
- **Markers, read late** — model methods are marked like trame-dataclass's
  own `@watch`, but nothing binds them at construction: the `dataclass_*`
  tools read `@tool` markers when the agent reaches an instance.
- **One meaning per layer** — on app/component classes `@tool` yields a
  top-level named tool (a namespace curated by the app author); on models it
  yields a `dataclass_call`-callable. Model methods never enter the top-level
  namespace, so there are no prefixes to derive and no collisions to police.
- **Target resolution** — `target` accepts an exposed instance name or a
  reachable `_id`, with the same reachability check as `dataclass_get` /
  `dataclass_update`, and is always explicit: no singleton special case, no
  schemas that change shape at runtime. An unknown method or stale target
  errors with the list of available methods / a hint to `dataclass_get` again.
- **Static tool list** — exposing more models or creating instances never
  adds top-level tools, so the MCP tool list is constant after setup;
  `dataclass_schemas()` is the live per-class catalog of fields and methods.
- **Inside the method** the instance is ordinary Python: mutate fields,
  create and attach other instances, call other methods — watchers and
  client sync fire as usual. Whatever gets attached to visible reference
  fields becomes reachable immediately; returning the new `_id` (as
  `add_contact` does) hands the client a handle in the same round trip.
- **Promotion is explicit** — when a model action deserves top-level
  billing, the app author delegates with a named tool:

  ```python
  @tool
  def add_contact(self, name: str, age: int = 0) -> dict:
      """Add a contact to the shared address book."""
      return self.book.add_contact(name, age)
  ```

Model methods are the blessed home for structural changes, which
`dataclass_update` deliberately refuses: creation, insertion, and deletion carry
app decisions — constructor arguments, placement, side effects — that only
the model author can make.

### Constructors and batches

Some models are best edited the way their Python API reads: build an object,
then attach it (`graph.add_node(Node(...))`). Over MCP that takes two calls,
and between them the new instance is reachable from nothing, so the agent
could not name it in the second. `dataclass_batch` runs both in one call,
holding what the first built for the second:

```python
class Node(MCPStateDataModel):
    label = mcp_field(Sync(str, ""))
    ports = mcp_field(Sync(list[Port], list, has_dataclass=True))

    @constructor(description="A node with one port, in no graph yet")
    @classmethod
    def new(cls, server, label: str) -> Node:
        return cls(server, label=label, ports=[Port(server)])


class Graph(MCPStateDataModel):
    nodes = mcp_field(Sync(list[Node], list, has_dataclass=True))

    @tool
    def add_node(self, node: Node) -> None: ...

    @tool
    def connect(self, source: Port, target: Port) -> Link: ...
```

```
dataclass_batch([
  {"new": "Node", "args": {"label": "blur"}, "as": "blur"},
  {"call": "graph", "method": "add_node", "args": {"node": "@blur"}},
  {"call": "graph", "method": "connect",
   "args": {"source": "<a port id>", "target": "@blur.ports.0"}},
])
→ {"results": [{"id": "40", "type": "Node", "values": {...}}, null, {"id": "44", "type": "Link", ...}],
   "unattached": []}
```

Contract:

- **Constructors** — `@constructor` marks one classmethod per class, called
  `cls.<name>(server, **args)`; the server parameter is left out of the
  schema, the result must be an instance of `cls`, and subclasses inherit
  the constructor (with their own `cls`) unless they mark one.
  `expose_model(constructor=...)` names one (a name, or `{"name",
  "description"}`) or hides one (`False`). A constructor is only reachable
  as a batch step: there is no standalone `dataclass_new`.
- **Steps** — `{"new": Model, "args", "as"}`, `{"call": target, "method",
  "args", "as"}`, `{"update": target, "values"}`, run in order. Shapes,
  `as` names, references and `new` models are checked before any step runs
  (nothing applied on a mistake); arguments are validated when their step
  runs.
- **References** — `"@name"` is the result of the step named `name`;
  `"@name.seg..."` follows visible fields of instances (lists of references
  as `dataclass_get` shows them), keys of objects and indexes of lists.
  Allowed in targets and anywhere in `args` / `values`; instances in
  arguments become ids. `"@@"` escapes a literal `@`.
- **Access** — targets and arguments are reachable instances, or instances
  this batch built, which are writable. A built instance that is still
  unattached when the batch ends is dropped and listed in `unattached`.
- **One unit** — one turn of the tool lock, one `with state:` flush;
  watchers of each step's instance are awaited before the next step.
  The batch stops at the first failing step: its error names the step, the
  reason, and the results of the steps already applied, which stay (no
  rollback). A cancel stops it between steps.

## Configuration

```python
get_mcp(
    server=None,          # trame server or name, resolved like get_server()
    name=None,            # MCP server name; defaults to server.name
    path="/mcp",          # endpoint path on the trame web server
    http=True,            # mount the HTTP endpoint; False = in-process only
    enabled=False,        # agent access switch (see below)
    token=None,           # bearer token required on the endpoint
    allowed_origins=(),   # extra browser origins allowed to call the endpoint
    serialize_tools=True,
    instructions=None,    # the app's text, sent before the generated overview
)
```

Every option is also a settable attribute (`mcp.enabled = True`);
`configure()` sets several at once and `get_mcp()` applies its keyword
arguments on every call, so the order in which code obtains the instance does
not matter. `path` and `http` are fixed once the server has started (the
route is mounted then); everything else can change at runtime.

**Declare, then go live.** Exposing functions, objects, state, models and
instances, constructing `MCPApp` / `MCPComponent` — all of it only declares.
`mcp.enabled` (default `False`) is the one switch for all agent access,
settable at any time — before start, in a lifecycle hook, from a UI toggle
bound to state:

- HTTP: the route is mounted regardless (unless `http=False`) and answers 503
  while access is off — aiohttp freezes its routes at startup, so a route
  mounted only when live could never be switched on later. The connection
  banner prints when access first goes live, at startup or later.
- In-process: `LLMAdapter` tool calls and resource reads raise
  `MCPDisabledError`, so the embedded chat is switched off with the same flag
  (it still lists the tools: trame-llm discovers them once, at startup).

Launch options let the end user decide without touching code:

- `--mcp` / `--no-mcp` — `enabled`.
- `--mcp-token TOKEN` or `TRAME_MCP_TOKEN=...` — bearer token (prefer the
  environment variable: command lines are visible to other local users).

They have the last word at launch: until the server starts, values the
application sets (attributes, `configure()`, `get_mcp()` alike) are defaults
that a launch option overrides, and the attribute always reads the effective
value. Once the server is running, every assignment applies, except against
the options that withhold something: `--no-mcp` and `--mcp-deny` hold for
the whole run (`enabled_locked`, `permissions.locked`).

`http=False` keeps everything exposed usable by in-process consumers —
e.g. trame-llm's chat widget via `trame_mcp.LLMAdapter` — without exposing an
MCP route that third parties could connect to.

`mcp.url` gives the full endpoint URL once the server is running.

## What the agent reads

The agent learns about the application only from what the server sends, so
every declaration doubles as documentation. Three sources, each derived from
code by default and overridable:

- **Server instructions** (`initialize`): the application's `instructions`,
  then a generated overview — a notice that the app is live and shared with
  the people using it (changes appear in their browsers, and they may be
  editing the same data), and where to start: the data entry points with
  their types (`` `workspace` (Workspace) ``) and the state tools, when
  exposed. It is computed at connection time, so it reflects the exposures
  made by then. `LLMAdapter` passes the same text to trame-llm.
- **Tool, resource and prompt descriptions**: the docstring, or
  `description=`. Parameters get descriptions and constraints from
  `Annotated[T, "description", {...}]` — tool and model method parameters
  in their input schemas, prompt arguments as their MCP `description`. The
  generic tools carry fixed workflow descriptions, and `state_get` /
  `state_update` list the exposed variables.
- **The model catalog** (`dataclass_schemas()`): a model's description is its
  own `description` option (`MCPStateDataModel` class keyword or
  `expose_model()` argument), else its own docstring; neither is inherited,
  since text about a base class rarely fits a subclass. Fields carry their
  `mcp_field` description, methods their docstring or `tools=`
  description.

Docstrings are written for developers first and reach the agent verbatim, so
every level has an agent-facing override: `@mcp.tool(description=...)` /
`@tool(description=...)`, the model `description` option, and `description`
in `fields=` / `tools=` entries (example 03 overrides `Scene.clear_objects`,
whose docstring explains renderer internals).

**Inheritance keeps the catalog small.** A model whose nearest agent-visible
base declares fields is listed as a delta: `"extends": "SceneObject"` plus
only the fields and methods it adds or changes. The base is listed even when
the agent never meets an instance of it, so the chain always resolves. A
subclass that hides something its base exposes is listed in full instead,
since a delta cannot express removals. In example 03 this cuts
`dataclass_schemas()` from about 18k to 5k characters: the six object types
no longer repeat their eight shared fields and two methods.

## Transport & endpoint (summary)

- MCP **Streamable HTTP** transport, stateless, at `path`: POST carries
  JSON-RPC; GET (SSE stream) and DELETE (session teardown) answer 405. The
  `initialize` response carries an `Mcp-Session-Id`, which only scopes
  request ids for cancellation (decision 25). A request is cancelled by
  `notifications/cancelled` or when its client disconnects, and then
  answered with a `-32800` "Request cancelled" error. Mounted
  on the wslink aiohttp application via the `on_server_bind` life-cycle hook —
  same mechanism as `trame/examples/09_advanced/custom_endpoint`.
- The tool list is static after setup (`listChanged: false`); generic tool
  descriptions/schemas reflect exposures made before the client lists tools.

## Security

The endpoint is an automation API mounted on the app's own port, so the
threat that matters most is not a rogue MCP client but **a web page in the
user's browser**: any site can POST to `http://localhost:8080/mcp`, and a
`Content-Type: text/plain` request goes out without a CORS preflight. Every
request therefore passes, in order:

1. **Origin check** — browsers always send `Origin` on cross-site requests;
   only loopback origins, the endpoint's own origin and `allowed_origins`
   pass (403 otherwise). Requests without `Origin` (non-browser clients) are
   unaffected. This is the MCP spec's DNS-rebinding/CSRF requirement.
2. **Authentication** — with a token, `Authorization: Bearer <token>` is
   required (constant-time comparison, 401 otherwise). Without a token the
   `Host` header must name a loopback address (403 otherwise), which defeats
   DNS rebinding (an attacker domain re-pointed at 127.0.0.1 is still sent
   with the attacker's `Host`).
3. **Access switch** — 503 while `mcp.enabled` is False (the default).
4. **Content type** — `application/json` only (415), so no browser request
   can skip the preflight, which this endpoint never grants (no OPTIONS
   route, no CORS headers).

**Defaults by bind address.** On a loopback interface (trame's default) no
token is needed: only local processes can connect, and checks 1-2 keep remote
pages out. When the server listens beyond loopback (`--host 0.0.0.0`) and no
token is configured, a random token is generated at startup and printed with
the `claude mcp add ... --header "Authorization: Bearer ..."` command, Jupyter
style — an unauthenticated automation API is never served to the network.

**What an agent can reach** is exactly what the app opted in, with per-item
access:

- powers beyond the app itself (paths on disk, code to run) behind
  permissions the people running it can withhold, at launch (`--mcp-deny`)
  or from the app's UI (`requires=`, `mcp.permissions`);

- functions and methods the author exposed; `read_only` ones are flagged so
  clients can auto-approve them and confirm the rest;
- allowlisted state variables, writable unless `read_only=True`;
- instances of explicitly exposed model classes (`MCPStateDataModel` or
  `expose_model()`), reachable from exposed instances through *visible*
  reference fields — a library model attached somewhere in the tree stays
  invisible until the app exposes its class;
- within those classes, only fields and methods that are marked, listed, or
  covered by an explicit `all_fields` / `all_tools`; `ServerOnly` fields are
  never exposed without an explicit `read_only=`, and whole subtrees can be
  `read_only`.

All MCP-supplied values are validated against the advertised schemas before
any app code runs.

**Residual risks** (inherent, documented rather than solved here):

- *Prompt injection.* Agents read app data that browser users (or files) put
  there; a malicious string can steer the agent into calling the tools it was
  given. Don't expose destructive tools, or mark them
  (`annotations={"destructiveHint": True}`) so clients ask first, and prefer
  `read_only` exposures for data the agent only needs to see.
- *The browser channel.* trame-dataclass's websocket protocol lets any
  connected client write any field (including `ServerOnly`) of any live
  instance by id, and ids are sequential. MCP's reachability and field policy
  are strictly narrower than that channel; securing the websocket itself
  (trame `--authKey`, network exposure) is the app's responsibility.
- *Tool errors* are returned to the agent verbatim (exception type and
  message), which helps self-correction but can reveal paths or internals —
  raise `ValueError` with agent-facing messages in tools.

**What permissions do not stop** (`mcp.permissions`; reviewed by asking how
an agent would get around them, nothing changed yet). They bound an agent
whose only access is MCP and that was never allowed a power that leaves
something behind:

- *Another channel.* They gate MCP calls only. An agent that can also run
  commands, or open the app's websocket like a browser, reaches what the UI
  can: any state key (a switch, unless `--mcp-deny` locked it) and any model
  field, and whatever the UI does with them (an editor that applies a
  script). Give the agent MCP alone, and keep other local programs off the
  websocket (`--authKey`).
- *What an allowed power left behind.* Denying a permission stops new calls,
  not what earlier ones set up: code an agent wrote while allowed can stay in
  the session (and in saved files) and keep being run, with arguments the
  agent still controls, or can have changed files on disk, the app's own
  settings and plugin folders included. A permission to run its own code is
  trust for the user's whole account, not for the session.
- *Unmarked needs.* An entry needing a permission but not marked
  `requires=` is open; the startup audit only catches names nobody
  declared. Needs that depend on the arguments (a path given as a
  parameter) are the app's own `require()` calls, as complete as the app's
  knowledge of which arguments are paths.
- *Asking.* A refusal names the permission and who can grant it; an agent
  steered by injected text will ask the user to grant it.

Possible follow-ups: mark what an agent wrote (provenance an app can check
and keep refusing while the permission is off), and narrower permissions
(allowed folders, reading apart from writing).

## Decision log

1. **Generic data tools** (`dataclass_schemas` / `dataclass_get` / `dataclass_update` /
   `dataclass_call`) over per-exposure or per-method tools: constant tool
   count; the lost schema precision is recovered by `dataclass_schemas()`, which
   advertises every model's structure up front.
2. **Model methods ride the generic layer** via `dataclass_call(target, method,
   args)` instead of becoming top-level tools (supersedes an earlier
   singleton-target rule): `target` is always explicit, and no tool schema
   ever changes shape at runtime.
3. **No magic**: a bare `get_mcp(server)` exposes nothing; all exposure is
   explicit, and nothing is served until access is turned on (decision 15).
4. **All three primitives** (tools, resources, prompts), with the same
   decorator pairs (`@mcp.tool` / `@tool`, ...).
5. **Exposure is grouped by the trame module it covers** (explicit beats
   polymorphic dispatch): `@mcp.tool` & co. and `mcp.expose()` for
   functions and objects,
   `mcp.state` for plain state, `mcp.dataclass` for trame-dataclass, each
   group mirroring its generic tool family (`state_*`, `dataclass_*`).
6. **Top-level tool names are never derived.** Every name in the tool list
   was written by the app author (`@tool`, explicit `mcp_prefix` /
   `prefix=`); model methods never enter the top-level namespace at all,
   and promoting one to a named tool is an explicit one-line delegation.
7. **Generic tools are family-prefixed** — `state_get` / `state_update` for
   plain state; `dataclass_schemas` / `dataclass_get` / `dataclass_update` /
   `dataclass_call` for models. The prefix marks the layer, and the verbs
   align across the two families.
8. **Implementation: minimal in-house protocol layer** (stateless streamable
   HTTP on aiohttp) rather than bridging the official `mcp` SDK. The static
   tool list removes the need for SSE/`list_changed`, v1 has no
   server-to-client features, and the SDK transport is Starlette-coupled
   while trame is aiohttp (still stateless: the session id of decision 25
   holds no state). GET/DELETE answer 405; batching is rejected
   (removed in protocol 2025-06-18).
9. **Tool calls are serialized by default** (`serialize_tools=True`). LLM
   clients routinely fire tool calls in parallel; synchronous tool bodies are
   already atomic on trame's event loop, and the per-server lock extends that
   guarantee across `await` points in async tools, so parallel calls cannot
   interleave read-modify-write logic. Opt out per tool
   (`@tool(serialize=False)`) for read-only and waiting tools, or per server
   (`serialize_tools=False`); see decision 24. Note the remaining, inherent concurrency: browser
   websocket messages still interleave with tools at await points — that is
   trame's shared-loop model, not something MCP can (or should) lock away.
10. **Payloads envelope model data** — `dataclass_get` returns
    `{"id", "type", "values": {...}}`: protocol metadata and model fields are
    structurally separated, so no field name can ever collide with protocol
    keys, and the metadata keys stay bare like trame-dataclass's own wire
    format (`{"id", "state"}`). `updates` in `dataclass_update` takes exactly
    the shape of `values`.
11. **Agent-aware classes are base-class variants, as in modern trame.**
    `MCPApp` / `MCPComponent` / `MCPStateDataModel` pair with `TrameApp` /
    `TrameComponent` / `StateDataModel` (trame itself moved from the
    `@TrameApp()` class decorator to base classes). `MCPStateDataModel`
    carries its options as class keywords, inherited by subclasses; at
    runtime it is a plain `StateDataModel`. The app classes take no MCP
    options: configuration goes through `self.mcp`, the same as in
    functional apps. Fields stay `mcp_field()` wrappers rather than MCP
    variants of `Sync` / `ClientOnly` / `ServerOnly`: the agent options are
    independent of the sync mode, so they compose with any descriptor
    instead of multiplying the descriptor types.
12. **Exposure is opted into per class, then per field and method.** A
    class is visible only through `MCPStateDataModel` (classes you own) or
    `expose_model()` (any class, e.g. from a library), so nothing is swept in
    just by being reachable. Within it, fields and methods are opted in by
    markers (`mcp_field`, `@tool`), by `fields=` / `tools=` lists, or by an
    explicit `all_fields` / `all_tools` flag — never implicitly, so whether a
    field is visible never depends on whether some *other* field is marked.
    Field policy lives on the field where you own the class, and in a dict
    with the same options where you don't.
13. **Validate against the advertised schema.** Tool arguments, model method
    arguments, field writes and state writes are all checked against the
    same JSON Schema the client sees (a small in-house validator for the
    subset we generate), because neither Python type hints nor
    trame-dataclass's default `WARNING` validation stop ill-typed input.
14. **Read-only exposure is per instance entry, writability wins on
    overlap.** An entry can be exposed `read_only`; an instance reachable
    from both kinds of entry is writable, since the app already granted that
    through the other path.
15. **Declared is not live; one switch for agent access.** `mcp.enabled`
    defaults to off and gates the HTTP endpoint and in-process consumers
    alike, settable at any time; `http=False` stays the construction-time
    "never mount a route" choice. Off by default is what lets libraries
    build on `MCPComponent` and `MCPStateDataModel` without turning anything
    on in the apps that use them, with no per-component opt-in to write.
    Launch options (`--mcp`/`--no-mcp`, `--mcp-token`, `TRAME_MCP_TOKEN`)
    beat whatever the application sets before the server starts, however it
    sets it, so `self.mcp.enabled = True` in an app never takes the choice
    away from the end user; after start every assignment applies, except
    against `--no-mcp` (decision 31).
16. **Safe without configuration.** Origin + loopback-`Host` checks make the
    tokenless default safe on localhost; a non-loopback bind gets a generated
    token rather than an open endpoint.
17. **Entry points are live and held.** `expose_instance()` takes an
    instance, a collection, or a getter evaluated on every call, so replaced
    attributes and plain-Python containers need no re-exposing or wrapper
    model. Direct instances are held strongly: an exposure never disappears
    on garbage-collection timing; `unexpose_instance()` is explicit.
18. **Reachability stays the instance boundary.** Exposing every live
    instance of a class was rejected: trame-dataclass's process-wide weak
    registry includes detached and not-yet-collected objects and instances
    of other servers, and a flat list loses the structure the agent needs.
19. **The app has the final word.** `expose_model()` overrides the
    `MCPStateDataModel` options of a class and its subclasses (nearest
    `expose_model()` class along the MRO, app first), so an application can
    narrow models a library ships agent-ready.
20. **trame's decorator split.** A plain function has no object whose
    construction could expose it, so it takes the server-bound decorator
    (`@mcp.tool`, as `@state.change`); methods and models take the inert
    marker of the same name (`@tool`, as `@change`), bound when the object
    is built or read when the agent reaches a model. Same names, same
    options, one rule. Rejected: markers only plus `mcp.expose(fn)` for
    functions (a redundant second step, kept only for functions marked
    where no server is in scope), and a process-wide registry of marked
    functions (every server, and every imported library, would contribute
    tools that close over the wrong `server`).
21. **One access vocabulary.** `read_only=True` means the agent cannot
    change anything through it — a tool hint, a field, a state variable, an
    instance entry — and `hidden=True` excludes something an `all_*` flag
    would include. Exposed values take their natural access (`Sync` fields
    and state variables writable, `ClientOnly` fields read-only), since
    exposing is already an explicit, per-name opt-in.
22. **Agent documentation is derived, overridable and compact.** Everything
    the agent reads comes from declarations (docstrings, type hints,
    `Annotated`, `mcp_field`), with a `description` override at every level,
    because the same docstring serves developers and the agent. Model
    descriptions are not inherited, like docstrings. The catalog states
    inheritance instead of repeating it, and the server instructions add a
    generated overview after the application's own text — the live-app
    notice and the starting points, which no tool description can carry.
23. **A call returns once its effects are visible.** Tool calls run inside
    `with state:` and `dataclass_update` / `dataclass_call` await the target
    instance's watchers, so a follow-up read never sees the state half
    updated: an agent cannot wait for a flush or a watcher task the way a
    browser's next frame does. Reactions on *other* instances (a watcher
    chaining into another model) remain asynchronous.
24. **The lock is a default, not a law: tools opt out one by one.** A tool
    that waits (for a job, for the pipeline to go idle) holds the lock for as
    long as it waits, so every other call queues behind it, including the
    one that would cancel what it waits for. `serialize=False` on the marker
    or `@mcp.tool` exempts such a tool, and only it: the rest keep their
    read-modify-write guarantee. `dataclass_call` is registered unlocked and
    takes the lock around the method it calls when that method is
    serialized, resolving the target again under the lock, so a model method
    gets the same choice as a named tool. Rejected: a per-call argument
    (the agent cannot know which tools are safe to overlap) and a separate
    lock per tool (a waiting tool would still block its own kind).
25. **A client can stop what it started.** `notifications/cancelled`
    cancels the request's task, so its tool stops at its next `await`,
    leaves the tool lock and flushes the state it changed; a synchronous
    tool has finished before any cancel can arrive. Request ids are only
    unique per client, so `initialize` hands out an `Mcp-Session-Id` and a
    cancel reaches only the requests sent with the same id (clients without
    it share one scope). The server keeps nothing per session and never
    requires the header: a restarted server accepts ids it never issued.
    The spec asks for no response to a cancelled request, but an HTTP POST
    must end with one: it gets a `-32800` "Request cancelled" error (LSP's
    code), which the client ignores, as the spec requires of the side that
    cancelled.
26. **A client that disconnects cancels its call.** aiohttp keeps a handler
    running after its client is gone (`handler_cancellation` is off by
    default, and it is the whole server's setting, wslink's included), so a
    call whose client timed out would keep running and keep the lock. The
    endpoint polls the request's transport while the call runs (every
    `DISCONNECT_POLL`, 0.5 s) and cancels it once the connection closes.
    Polling, because aiohttp exposes no disconnect callback for a request
    whose body has been read. Rejected: progress notifications to keep
    clients waiting, which need SSE responses (decision 8) and would not
    lift a client's wall-clock limit anyway.
27. **Constructors are marked, and run only inside a batch.** A model API
    that builds an object and then attaches it needs the object to exist
    between two steps. Rejected: constructors that attach what they build
    (an implicit destination, and a side effect the Python API does not
    have); a `dataclass_new` tool whose result stays reachable for the
    session (state held per client, with no natural end when the agent never
    attaches it). A batch bounds that life to one call and adds nothing
    implicitly.
28. **A model parameter takes an id and receives the instance.** Methods
    keep Python signatures (`add_node(self, node: Node)`) instead of taking
    strings and looking ids up themselves, and the lookup applies the same
    reachability and read-only rules as a target, centrally. Ids rather than
    reference objects (`{"id", "type"}`), since an id is what every other
    tool takes.
29. **A batch may touch what it built, and nothing else beyond the
    boundary.** Reachability (decision 18) stays the rule; the one
    extension is the batch's own new instances, which no other call can
    name and which are dropped when the batch ends unattached. The batch is
    one unit for the app (one lock turn, one flush) but not a transaction:
    graph edits cannot be undone generically, so a failure reports what was
    applied instead of rolling it back.
30. **Permissions are declared by the app and enforced where calls run.**
    A client can allow or deny whole tools, but a model method or
    constructor is reached through `dataclass_call` and `dataclass_batch`,
    the tools every harmless edit uses too, so only the server can tell
    them apart. Denied entries leave the listings and are refused when
    called, since the stateless endpoint cannot tell a client its tool list
    changed. `implies` keeps related permissions consistent in both
    directions, so CLI, code and UI never hold a power without one it
    grants anyway. Rejected: hiding only (a client that listed earlier would
    still call), refusing only (agents would keep trying what they cannot
    use), and an unknown `--mcp-deny` name ignored with a warning (a typo
    would leave the power allowed).
31. **A launch option that withholds something holds for the whole run.**
    Whoever launches with `--no-mcp` or `--mcp-deny` (a user, an admin, a
    launch script) decides for the run: neither the app's code nor its UI
    can undo it, and `enabled_locked` / `permissions.locked` let the UI show
    the switch as fixed. Options that turn something on or set a value
    remain defaults until start, so a UI can still turn access off at
    runtime. Supersedes the part of decision 15 that let a runtime
    assignment override `--no-mcp`.
