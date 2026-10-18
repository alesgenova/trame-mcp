"""Markers for trame-mcp, also importable from ``trame_mcp``.

Like ``trame.decorators`` they are inert: they only attach metadata to a
method or a field descriptor, and need no server. ``MCPApp`` /
``MCPComponent`` bind their marked methods when built (``mcp.expose(obj)``
for any other object), and the ``dataclass_*`` tools read the markers of
``MCPStateDataModel`` classes. Plain functions use the server-bound
counterparts instead — ``@mcp.tool`` is to ``@tool`` what trame's
``@state.change`` is to ``@change``.

Libraries can apply them freely; whether anything is exposed, and whether it
is served, stays the application's decision.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

from trame_mcp.permissions import requirements

MARKER = "_trame_mcp"
FIELD_MARKER = "_trame_mcp_field"
MODEL_MARKER = "_trame_mcp_model"

FIELD_OPTIONS = ("description", "read_only", "hidden", "schema")
TOOL_OPTIONS = ("description", "read_only", "hidden", "serialize", "requires")
CONSTRUCTOR_OPTIONS = ("name", "description", "requires")

__all__ = [
    "constructor",
    "mcp_field",
    "prompt",
    "resource",
    "tool",
]


def _mark(kind, f, **options):
    markers = f.__dict__.setdefault(MARKER, {})
    markers[kind] = options
    return f


def tool(
    fn=None,
    *,
    name=None,
    description=None,
    read_only=False,
    annotations=None,
    serialize=True,
    requires=None,
):
    """Mark a method as an MCP tool.

    A method of an exposed object (``MCPApp``, ``MCPComponent``,
    ``mcp.expose(obj)``) becomes a named tool; a method of a
    ``StateDataModel`` becomes callable through ``dataclass_call``. For a
    plain function use ``@mcp.tool``, which takes the same options.

    Usable bare (``@tool``) or with options. The name defaults to the
    function's and the description to its docstring.

    :param read_only: calling it does not modify anything. Advertised to
        clients as ``readOnlyHint`` (they may skip confirmation prompts), and
        required for a model method to be callable on a ``read_only`` instance.
    :param annotations: extra MCP tool annotations (``destructiveHint``,
        ``idempotentHint``, ``openWorldHint``, ``title``).
    :param serialize: with ``False``, calls skip the server's tool lock
        (``serialize_tools``) and may run while other tool calls are in
        progress, at their ``await`` points and they at its. Meant for
        read-only or waiting tools (a status poll, a wait for a job), which
        would otherwise hold back every other call, a cancel included.
    :param requires: the permission (or list of them) a call needs, declared
        with ``mcp.permissions.declare()``. While one is denied the tool is
        not offered, and calls to it are refused.
    """
    requires = requirements(requires)

    def decorate(f):
        return _mark(
            "tool",
            f,
            name=name,
            description=description,
            read_only=read_only,
            annotations=annotations,
            serialize=serialize,
            requires=requires,
        )

    if fn is None:
        return decorate
    return decorate(fn)


def _function_of(member):
    """The function behind a class member: a classmethod's or
    staticmethod's ``__func__``, a plain function itself, else None."""
    if isinstance(member, (classmethod, staticmethod)):
        return member.__func__
    return member if inspect.isfunction(member) else None


def constructor(fn=None, *, description=None, requires=None):
    """Mark a classmethod of a model as the constructor agents use.

    .. code-block:: python

        class Person(MCPStateDataModel):
            @constructor(description="A new person, not in any book yet")
            @classmethod
            def new(cls, server, name: str) -> Person:
                return cls(server, name=name)

    trame-mcp calls it as ``cls.new(server, **args)``: the first parameter
    after ``cls`` receives the trame server and is not part of the
    advertised schema; the others are validated like a method's. It must
    return an instance of ``cls``. Agents use it in a ``dataclass_batch``
    ``new`` step, whose later steps attach the instance somewhere reachable
    (the batch drops it otherwise).

    One constructor per class; subclasses inherit it (called with their own
    ``cls``) unless they mark one. The marker goes above or below
    ``@classmethod``; usable bare or with options. The description defaults
    to the docstring. ``requires`` names the permissions it needs, as for
    ``@tool``.
    """
    requires = requirements(requires)

    def decorate(f):
        if isinstance(f, staticmethod):
            msg = "@constructor marks a classmethod, not a staticmethod"
            raise TypeError(msg)
        target = _function_of(f)
        if target is None:
            msg = f"@constructor marks a classmethod, got {type(f).__name__}"
            raise TypeError(msg)
        _mark("constructor", target, description=description, requires=requires)
        return f

    if fn is None:
        return decorate
    return decorate(fn)


def find_constructor(cls):
    """``(attribute name, marker options)`` of the nearest constructor
    marked along ``cls``'s MRO, or None. Raises when a class marks two."""
    for klass in cls.__mro__:
        marked = [
            (attr, get_markers(_function_of(member))["constructor"])
            for attr, member in vars(klass).items()
            if "constructor" in get_markers(_function_of(member))
        ]
        if len(marked) > 1:
            msg = f"{klass.__name__} marks several constructors: {sorted(a for a, _ in marked)}"
            raise ValueError(msg)
        if marked:
            return marked[0]
    return None


def resource(uri, *, name=None, description=None, mime_type=None):
    """Mark a method as an MCP resource served at ``uri`` (``@mcp.resource``
    for a plain function)."""

    def decorate(f):
        return _mark(
            "resource", f, uri=uri, name=name, description=description, mime_type=mime_type
        )

    return decorate


def prompt(fn=None, *, name=None, description=None):
    """Mark a method as an MCP prompt template (``@mcp.prompt`` for a plain
    function)."""

    def decorate(f):
        return _mark("prompt", f, name=name, description=description)

    if fn is None:
        return decorate
    return decorate(fn)


def get_markers(fn):
    """Return the marker dict attached to a function, or an empty dict."""
    return getattr(fn, "__dict__", {}).get(MARKER, {})


# -----------------------------------------------------------------------------
# Model fields
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldSpec:
    description: str | None = None
    access: str | None = None  # "read-write" | "read-only" | "hidden" | None (by mode)
    schema: dict | None = None


def field_access(read_only=None, hidden=False, label="mcp_field()"):
    """Fold the public ``read_only`` / ``hidden`` options into one access
    mode, or None to keep the field's own mode."""
    if read_only is not None and not isinstance(read_only, bool):
        msg = f"{label}: read_only must be True, False or None, got {read_only!r}"
        raise TypeError(msg)
    if not isinstance(hidden, bool):
        msg = f"{label}: hidden must be True or False, got {hidden!r}"
        raise TypeError(msg)
    if hidden:
        if read_only is not None:
            msg = f"{label}: a hidden field takes no read_only"
            raise ValueError(msg)
        return "hidden"
    if read_only is None:
        return None
    return "read-only" if read_only else "read-write"


def mcp_field(descriptor, description=None, *, read_only=None, hidden=False, schema=None):
    """Annotate a ``StateDataModel`` field descriptor for MCP.

    .. code-block:: python

        class SceneObject(MCPStateDataModel):
            opacity = mcp_field(
                Sync(float, 1.0),
                "0 = transparent, 1 = opaque",
                schema={"minimum": 0, "maximum": 1},
            )
            score = mcp_field(ServerOnly(int, 0), read_only=True)

    Marking a field opts it in: on an exposed model it is visible even when
    the model does not use ``all_fields``.

    :param descriptor: the ``Sync`` / ``ClientOnly`` / ``ServerOnly`` field,
        returned unchanged (the annotation is stored on it).
    :param description: shown to the client in ``dataclass_schemas``.
    :param read_only: whether the agent may only read the field. Defaults to
        the descriptor mode: ``Sync`` writable, ``ClientOnly`` read-only.
        ``ServerOnly`` fields have no shareable default, so marking one
        requires ``read_only=True`` or ``False``. Nested-model reference
        fields are never writable.
    :param hidden: keep the field invisible, e.g. one that ``all_fields``
        would otherwise expose.
    :param schema: extra JSON Schema keywords merged into the schema derived
        from the field type (``minimum``, ``enum``, ``maxItems``, ...).
        Advertised to the client and enforced on every MCP write.

    Subclasses that redeclare a field (e.g. to change its default) inherit
    the annotation unless they annotate it again.
    """
    access = field_access(read_only, hidden)
    setattr(
        descriptor,
        FIELD_MARKER,
        FieldSpec(description=description, access=access, schema=dict(schema or {})),
    )
    return descriptor


def get_field_spec(cls, name):
    """Return the nearest :class:`FieldSpec` for ``cls.<name>`` along the MRO."""
    for klass in cls.__mro__:
        descriptor = klass.__dict__.get(name)
        spec = getattr(descriptor, FIELD_MARKER, None)
        if spec is not None:
            return spec
    return None


def get_field_descriptor(cls, name):
    for klass in cls.__mro__:
        if name in klass.__dict__:
            return klass.__dict__[name]
    return None


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """How a model class is exposed; built by :func:`model_spec`."""

    all_fields: bool
    all_tools: bool
    fields: dict  # field name -> {"description", "access", "schema"}
    tools: dict  # method name -> options (TOOL_OPTIONS)
    description: str | None = None  # for this class only, not its subclasses
    # None: the marked one (if any); False: none; else {"name", "description"}
    constructor: dict | bool | None = None


def _normalize(entries, allowed, label):
    if entries is None:
        return {}
    if isinstance(entries, str):
        msg = f"{label} must be a list or a dict of names, got a string"
        raise TypeError(msg)
    if not isinstance(entries, dict):
        entries = dict.fromkeys(entries)
    result = {}
    for name, given in entries.items():
        options = dict(given or {})
        unknown = set(options) - set(allowed)
        if unknown:
            msg = f"{label}[{name!r}]: unknown options {sorted(unknown)}, allowed: {list(allowed)}"
            raise ValueError(msg)
        if label == "fields":
            access = field_access(
                options.pop("read_only", None),
                options.pop("hidden", False),
                f"fields[{name!r}]",
            )
            if access is not None:
                options["access"] = access
        if "requires" in options:
            options["requires"] = requirements(options["requires"], f"{label}[{name!r}].requires")
        result[name] = options
    return result


def _normalize_constructor(option):
    """``constructor=`` as None, False or ``{"name", "description"}``."""
    if option is None or option is False:
        return option
    if isinstance(option, str):
        option = {"name": option}
    if not isinstance(option, dict):
        msg = (
            "constructor must be a classmethod name, a dict of "
            f"{list(CONSTRUCTOR_OPTIONS)}, False or None, got {option!r}"
        )
        raise TypeError(msg)
    unknown = set(option) - set(CONSTRUCTOR_OPTIONS)
    if unknown:
        msg = f"constructor: unknown options {sorted(unknown)}, allowed: {list(CONSTRUCTOR_OPTIONS)}"
        raise ValueError(msg)
    if not isinstance(option.get("name"), str) or not option["name"]:
        msg = "constructor: 'name' (the classmethod) is required"
        raise ValueError(msg)
    option = dict(option)
    if "requires" in option:
        option["requires"] = requirements(option["requires"], "constructor requires")
    return option


def model_spec(
    description=None,
    all_fields=None,
    all_tools=None,
    fields=None,
    tools=None,
    parent=None,
    constructor=None,
):
    """Validate and normalize the options shared by ``MCPStateDataModel``
    subclasses and ``mcp.dataclass.expose_model()``.

    With a ``parent`` spec, options left as None are inherited from it and
    ``fields`` / ``tools`` entries are added to its own, name by name. The
    description is never inherited, like a docstring.
    """
    if description is not None and not isinstance(description, str):
        msg = f"description must be a string, got {description!r}"
        raise TypeError(msg)
    parent = parent or ModelSpec(all_fields=False, all_tools=False, fields={}, tools={})
    constructor = _normalize_constructor(constructor)
    return ModelSpec(
        all_fields=parent.all_fields if all_fields is None else bool(all_fields),
        all_tools=parent.all_tools if all_tools is None else bool(all_tools),
        fields={**parent.fields, **_normalize(fields, FIELD_OPTIONS, "fields")},
        tools={**parent.tools, **_normalize(tools, TOOL_OPTIONS, "tools")},
        description=description,
        constructor=parent.constructor if constructor is None else constructor,
    )


def check_model_spec(cls, spec):
    """Catch typos in ``fields=`` / ``tools=`` names against the class."""
    field_names = getattr(cls, "FIELD_NAMES", set())
    unknown = sorted(set(spec.fields) - set(field_names))
    if unknown:
        msg = f"{cls.__name__} has no fields {unknown}; fields: {sorted(field_names)}"
        raise ValueError(msg)
    for name in spec.tools:
        if not inspect.isfunction(inspect.getattr_static(cls, name, None)):
            msg = f"{cls.__name__}.{name} is not a method"
            raise ValueError(msg)
    if isinstance(spec.constructor, dict):
        name = spec.constructor["name"]
        if not isinstance(inspect.getattr_static(cls, name, None), classmethod):
            msg = f"{cls.__name__}.{name} is not a classmethod (constructor option)"
            raise ValueError(msg)


def get_model_spec(cls):
    """Return the nearest ``MCPStateDataModel`` spec along the MRO, or None."""
    for klass in cls.__mro__:
        spec = klass.__dict__.get(MODEL_MARKER)
        if spec is not None:
            return spec
    return None
