"""Generic tools over trame-dataclass models, reached through ``mcp.dataclass``.

Two declarations, see DESIGN.md:
- *Models* say which classes the agent may see and which of their fields and
  methods: ``MCPStateDataModel`` subclasses you own, ``expose_model()`` for
  any other class.
- *Instances* say where the agent starts: ``expose_instance(source, name)``.
  Everything reachable from them through visible reference fields, and of an
  exposed class, is addressable by id; anything else is invisible.

Five generic tools serve them: dataclass_schemas / dataclass_get /
dataclass_update / dataclass_call, and dataclass_batch, which runs several
of those steps, and constructors (``@constructor``), in one call.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import typing
from dataclasses import dataclass

from trame_mcp.call_log import shorten
from trame_mcp.decorators import (
    MODEL_MARKER,
    check_model_spec,
    find_constructor,
    get_field_descriptor,
    get_field_spec,
    get_markers,
    get_model_spec,
    model_spec,
)
from trame_mcp.permissions import PermissionDeniedError, requirements
from trame_mcp.schema import (
    ToolError,
    ValidationError,
    bind_arguments,
    from_json,
    function_input_schema,
    nullable,
    resolve_model_arguments,
    type_to_schema,
    validate,
)

logger = logging.getLogger(__name__)

__all__ = ["BatchError", "DataclassLayer"]

MAX_DEPTH = 10
# Steps of a batch the call log names
MAX_LOGGED_STEPS = 12

SCHEMAS_DESCRIPTION = """\
Describe the application's data models: the named entry instances, and every
model class the agent can encounter, with field names, types, constraints
(read-only fields are flagged), the methods callable through
`dataclass_call`, and the "constructor" that creates an instance in a
`dataclass_batch` "new" step. A parameter with "model" takes the id of an
instance of that model. A model with "extends" also has every field, method
and constructor of the model it names, plus the ones it lists (which take
precedence). Call this first to learn the exact structure before reading or
writing."""

GET_DESCRIPTION = """\
Read a model instance as {"id", "type", "values": {...}}. `target` is an
exposed instance name or an instance id previously returned by this tool.
Nested instances appear inside `values` as {"id", "type"} references;
increase `depth` to inline them. Instances exposed for reading only carry
"read_only": true. A name exposing a collection returns {"name", "items"}."""

UPDATE_DESCRIPTION = """\
Update writable fields of one model instance, as a targeted patch: `updates`
maps field names (as listed under `values` by dataclass_get) to new values,
which must match the field schemas from dataclass_schemas. Nested instance
references are read-only; use `dataclass_call` for structural changes.
Nothing is applied if any field is unknown, read-only, or invalid."""

CALL_DESCRIPTION = """\
Invoke a method on a model instance. `target` is an exposed instance name or
an instance id, `method` is one of the methods listed by `dataclass_schemas`
for its type, `args` maps parameter names to values (an instance by its id).
A returned instance comes back as dataclass_get shows it."""

BATCH_DESCRIPTION = """\
Run several steps in one call, in order: one update of the application, one
turn of the tool lock. `steps` is a list of
- {"new": "<Model>", "args": {...}, "as": "<name>"}: create an instance with
  the model's constructor (see dataclass_schemas). It exists for this batch
  only, until a later step attaches it somewhere reachable; one left
  unattached is dropped and listed in `unattached`.
- {"call": "<target>", "method": "<method>", "args": {...}, "as": "<name>"}:
  as dataclass_call.
- {"update": "<target>", "values": {...}}: as dataclass_update's `updates`.
A string "@name" in a target or anywhere in args/values stands for the result
of the earlier step named so with "as"; "@name.field.0" follows a field of an
instance, a key of an object or an index of a list ("@@" writes a literal "@").
Instances in args are sent by id. Steps are checked before any runs. Returns
{"results": [one per step], "unattached": [...]}. An instance in the results
shows its id and its references to other instances (as ids), not its other
fields, and only its id once already shown; `results: "full"` shows each as
dataclass_get does. The batch stops at the first failing step: the steps
before it stay applied (no rollback), and the error lists their results."""

_STEP_KEYS = {
    "new": ("new", "args", "as"),
    "call": ("call", "method", "args", "as"),
    "update": ("update", "values"),
}
_STEP_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_MISSING = object()
_HIDDEN = object()


def _state_data_model():
    from trame_dataclass.v2 import StateDataModel

    return StateDataModel


def _model_types(annotation):
    """StateDataModel subclasses named anywhere in a type annotation."""
    found = set()
    stack = [annotation]
    while stack:
        current = stack.pop()
        if typing.get_origin(current) is None:
            if isinstance(current, type) and issubclass(current, _state_data_model()):
                found.add(current)
        else:
            stack.extend(typing.get_args(current))
    return found


def _is_concrete(cls):
    """Whether ``cls`` declares fields: the abstract bases (StateDataModel,
    MCPStateDataModel) do not, and trame-dataclass builds no instance of a
    class without any."""
    return hasattr(cls, "FIELD_NAMES")


def _subclasses(cls):
    result, stack = [], [cls]
    while stack:
        klass = stack.pop()
        result.append(klass)
        stack.extend(klass.__subclasses__())
    return result


def _extending(model, base_name, base):
    """``model`` as what it adds to or changes in ``base``, or None when it
    drops something ``base`` has (a delta cannot express removals)."""
    if not (
        base["fields"].keys() <= model["fields"].keys()
        and base["methods"].keys() <= model["methods"].keys()
        and ("constructor" in model or "constructor" not in base)
    ):
        return None
    delta = {"extends": base_name}
    if "description" in model:
        delta["description"] = model["description"]
    for key in ("fields", "methods"):
        delta[key] = {k: v for k, v in model[key].items() if base[key].get(k) != v}
    if "constructor" in model and model["constructor"] != base.get("constructor"):
        delta["constructor"] = model["constructor"]
    return delta


async def _settle(instance):
    """Wait for the watchers a change triggered: trame-dataclass runs them
    in a task, so without this the result, and the agent's next read, would
    not reflect what they derive."""
    completion = getattr(instance, "completion", None)  # trame-dataclass >= 2.2
    if completion is None:
        return
    try:
        await completion()
    except Exception:
        # The change itself is applied: a failing watcher is the app's bug,
        # not the agent's, and must not read as "nothing was applied".
        logger.exception(
            "trame-mcp: a watcher of %s %s raised", type(instance).__name__, instance._id
        )


def _models_in(value):
    model_cls = _state_data_model()
    if isinstance(value, model_cls):
        yield value
    elif isinstance(value, (list, tuple, set, frozenset)):
        yield from (v for v in value if isinstance(v, model_cls))
    elif isinstance(value, dict):
        yield from (v for v in value.values() if isinstance(v, model_cls))


@dataclass(frozen=True)
class _Field:
    name: str
    access: str  # "read-write" | "read-only"
    reference: bool  # holds nested model instance(s)
    schema: dict
    description: str | None
    type_def: object = None  # the declared type, to restore tuples


@dataclass(frozen=True)
class _Tool:
    name: str  # public name
    attr: str
    description: str
    read_only: bool
    serialize: bool = True  # dataclass_call takes the tool lock for it
    requires: tuple = ()  # permission names (mcp.permissions)


@dataclass(frozen=True)
class _Constructor:
    attr: str  # the classmethod
    description: str
    server: str  # its first parameter, which receives the trame server
    requires: tuple = ()  # permission names (mcp.permissions)


@dataclass(frozen=True)
class _Policy:
    fields: dict  # visible fields, name -> _Field
    tools: dict  # public name -> _Tool
    # Reference fields that are not exposed, name -> declared model types
    # (startup audit: an exposed model behind one is probably a mistake)
    hidden_references: dict
    constructor: _Constructor | None = None


class BatchError(ToolError):
    """A ``dataclass_batch`` step failed; the message says which, why, and
    what the steps before it returned."""


class _Scope:
    """What one ``dataclass_batch`` holds: the results named with ``as``,
    and the instances its ``new`` steps built (strongly, until it ends)."""

    def __init__(self):
        self.names = {}
        self.built = {}  # _id -> instance


class _Entry:
    __slots__ = ("read_only", "source")

    def __init__(self, source, read_only):
        if isinstance(source, (_state_data_model(), list, tuple, set, frozenset, dict)) or (
            callable(source) and not isinstance(source, type)
        ):
            self.source = source
        else:
            msg = (
                "expose_instance() expects a StateDataModel instance, a list/dict "
                f"of them, or a callable returning one; got {type(source).__name__}"
            )
            raise TypeError(msg)
        self.read_only = bool(read_only)

    def resolve(self):
        if callable(self.source) and not isinstance(self.source, _state_data_model()):
            return self.source()
        return self.source


class DataclassLayer:
    """``mcp.dataclass``: exposure of trame-dataclass models and instances."""

    def __init__(self, mcp):
        self._mcp = mcp
        self._models = {}  # class -> ModelSpec (expose_model)
        self._entries = {}  # name -> _Entry
        self._policies = {}  # class -> _Policy | None
        self._warned = set()
        self._registered = False

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def expose_model(
        self,
        cls,
        *,
        description=None,
        all_fields=False,
        all_tools=False,
        fields=None,
        tools=None,
        constructor=None,
    ):
        """Expose a ``StateDataModel`` class (and its subclasses) to the agent.

        For classes that do not derive from ``MCPStateDataModel`` — library
        models — or to override how one that does is exposed in this
        application. Options as in :class:`trame_mcp.MCPStateDataModel`. One
        call per class.

        :param constructor: the classmethod agents create instances with, as
            :func:`trame_mcp.constructor` marks one: its name, or
            ``{"name", "description"}``; ``False`` for none, None (default)
            for the marked one, if any.
        """
        if not (isinstance(cls, type) and issubclass(cls, _state_data_model())):
            msg = f"expose_model() expects a StateDataModel subclass, got {cls!r}"
            raise TypeError(msg)
        if cls in self._models:
            msg = f"{cls.__name__} is already exposed; use one expose_model() call per model"
            raise ValueError(msg)

        spec = model_spec(description, all_fields, all_tools, fields, tools, constructor=constructor)
        check_model_spec(cls, spec)
        self._models[cls] = spec
        self._policies.clear()
        try:
            for klass in _subclasses(cls):
                self._policy(klass)
        except Exception:
            del self._models[cls]
            self._policies.clear()
            raise

    def expose_instance(self, source, name, *, read_only=False):
        """Expose a named entry point for the agent.

        :param source: a model instance, a list/dict of instances, or a
            zero-argument callable returning either, evaluated on every call —
            ``lambda: self.workspace`` keeps working when the attribute is
            replaced, while a direct instance stays exposed until
            :meth:`unexpose_instance`.
        :param name: the name the agent uses as ``target``.
        :param read_only: no field updates on anything reached only through
            this entry, and only ``read_only`` methods are callable. An
            instance also reachable from a writable entry stays writable.

        Only instances of exposed classes are visible; reachability continues
        through their visible reference fields.
        """
        if not name or not isinstance(name, str):
            msg = "expose_instance() requires a non-empty string name"
            raise ValueError(msg)
        self._entries[name] = _Entry(source, read_only)
        self._register()

    def unexpose_instance(self, name):
        """Withdraw an entry point added with :meth:`expose_instance`."""
        if self._entries.pop(name, None) is None:
            msg = f"No instance is exposed as {name!r}"
            raise KeyError(msg)

    # -------------------------------------------------------------------------
    # Policy
    # -------------------------------------------------------------------------

    def _spec(self, cls):
        """Nearest class passed to ``expose_model()`` along the MRO, else
        the class's own ``MCPStateDataModel`` options."""
        for klass in cls.__mro__:
            if klass in self._models:
                return self._models[klass]
        return get_model_spec(cls)

    def _exposed(self, cls):
        return self._policy(cls) is not None

    def _policy(self, cls):
        if cls not in self._policies:
            # The abstract bases (StateDataModel, MCPStateDataModel) have no
            # fields: a reference typed with one still reaches the exposed
            # subclasses, never the base itself.
            spec = self._spec(cls) if _is_concrete(cls) else None
            self._policies[cls] = None if spec is None else self._build_policy(cls, spec)
        return self._policies[cls]

    def _model_ref_schema(self, annotation):
        """``type_to_schema`` hook: nested exposed models by class name."""
        if (
            typing.get_origin(annotation) is None
            and isinstance(annotation, type)
            and issubclass(annotation, _state_data_model())
        ):
            exposed = _is_concrete(annotation) and self._spec(annotation)
            return {"model": annotation.__name__} if exposed else {}
        return None

    def _build_policy(self, cls, spec):
        fields, hidden_references = {}, {}
        for name in sorted(cls.FIELD_NAMES):
            marker = get_field_spec(cls, name)
            listed = spec.fields.get(name)
            options = {}
            if marker is not None:
                options = {
                    "description": marker.description,
                    "access": marker.access,
                    "schema": marker.schema,
                }
            if listed is not None:
                options.update(listed)

            if name in cls.CLIENT_ONLY_NAMES:
                mode = "read-only"
            elif name in cls.CLIENT_NAMES:
                mode = "read-write"
            else:
                mode = "hidden"
            explicit = marker is not None or listed is not None
            if explicit and mode == "hidden" and not options.get("access"):
                msg = (
                    f"{cls.__name__}.{name} is a ServerOnly field: marking or listing it "
                    f"requires read_only=True or False"
                )
                raise ValueError(msg)
            access = options.get("access") or mode

            descriptor = get_field_descriptor(cls, name)
            reference = name in cls.DATACLASS_NAMES or bool(
                getattr(descriptor, "_has_dataclass", False)
            )
            checker = cls.TYPE_CHECKING.get(name)
            type_def = checker.type_def if checker else None

            opted_in = explicit or (spec.all_fields and mode != "hidden")
            if not opted_in or access == "hidden":
                if reference and not (explicit and access == "hidden"):
                    hidden_references[name] = _model_types(type_def)
                continue

            schema = type_to_schema(type_def, self._model_ref_schema)
            if options.get("schema"):
                schema.update(options["schema"])
            if getattr(descriptor, "_default", _MISSING) is None:
                schema = nullable(schema)
            description = options.get("description")
            if reference and description is None:
                description = "Reference(s) to nested model instances, as {id, type}"

            fields[name] = _Field(
                name=name,
                access="read-only" if reference else access,
                reference=reference,
                schema=schema,
                description=description,
                type_def=type_def,
            )

        return _Policy(
            fields=fields,
            tools=self._build_tools(cls, spec),
            hidden_references=hidden_references,
            constructor=self._build_constructor(cls, spec),
        )

    @staticmethod
    def _build_constructor(model_cls, spec):
        option = spec.constructor
        if option is False:
            return None
        if option is None:
            found = find_constructor(model_cls)
            if found is None:
                return None
            attr, options = found
        else:
            attr, options = option["name"], option
        member = inspect.getattr_static(model_cls, attr, None)
        if not isinstance(member, classmethod):
            msg = f"{model_cls.__name__}.{attr} is the constructor but is not a classmethod"
            raise ValueError(msg)
        marker = get_markers(member.__func__).get("constructor") or {}
        parameters = list(inspect.signature(getattr(model_cls, attr)).parameters)
        if not parameters:
            msg = (
                f"{model_cls.__name__}.{attr}: a constructor takes the trame server as its "
                f"first parameter after cls"
            )
            raise ValueError(msg)
        return _Constructor(
            attr=attr,
            description=options.get("description")
            or marker.get("description")
            or inspect.getdoc(member.__func__)
            or "",
            server=parameters[0],
            requires=requirements(options.get("requires", marker.get("requires"))),
        )

    @staticmethod
    def _build_tools(model_cls, spec):
        base_api = set(dir(_state_data_model()))
        tools = {}
        for attr in sorted(dir(model_cls)):
            fn = inspect.getattr_static(model_cls, attr)
            if not inspect.isfunction(fn):
                continue  # classmethods, staticmethods, properties, data
            marker = get_markers(fn).get("tool")
            listed = spec.tools.get(attr)
            automatic = (
                spec.all_tools
                and not attr.startswith("_")
                and attr not in base_api
                and "_watch" not in fn.__dict__
            )
            if marker is None and listed is None and not automatic:
                continue
            if attr in base_api:
                msg = f"{model_cls.__name__}.{attr} is part of StateDataModel's API and cannot be exposed"
                raise ValueError(msg)

            options = {**(marker or {}), **(listed or {})}
            if options.get("hidden"):
                continue
            public = (marker or {}).get("name") or attr
            if public in tools:
                msg = f"{model_cls.__name__}: two methods are exposed as {public!r}"
                raise ValueError(msg)
            tools[public] = _Tool(
                name=public,
                attr=attr,
                description=options.get("description") or inspect.getdoc(fn) or "",
                read_only=bool(options.get("read_only")),
                serialize=bool(options.get("serialize", True)),
                requires=requirements(options.get("requires")),
            )
        return tools

    # -------------------------------------------------------------------------
    # Reachability
    # -------------------------------------------------------------------------

    def _warn_once(self, key, msg, *args):
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(msg, *args)

    def _visible_models(self, value):
        """Instances of exposed classes found directly in ``value``."""
        for instance in _models_in(value):
            if self._exposed(type(instance)):
                yield instance
            else:
                cls = type(instance)
                self._warn_once(
                    ("unexposed", cls),
                    "trame-mcp: %s instances are reachable but not exposed; derive the "
                    "class from MCPStateDataModel or call mcp.dataclass.expose_model(%s)",
                    cls.__name__,
                    cls.__name__,
                )

    def _instructions(self):
        """The line this layer adds to the server instructions, if any."""
        if not self._entries:
            return None
        names = []
        for name, entry in self._entries.items():
            try:
                value = entry.resolve()
            except Exception:
                value = None
            if isinstance(value, _state_data_model()):
                details = [type(value).__name__]
            else:
                details = ["collection"] if isinstance(value, (list, tuple, dict)) else []
            if entry.read_only:
                details.append("read-only")
            names.append(f"`{name}` ({', '.join(details)})" if details else f"`{name}`")
        return (
            "Application data: call dataclass_schemas first, then dataclass_get on an "
            f"entry point: {', '.join(names)}."
        )

    def _entry_values(self):
        """``{name: (value, read_only)}`` for the current entry points."""
        return {name: (entry.resolve(), entry.read_only) for name, entry in self._entries.items()}

    def _children(self, instance):
        for f in self._policy(type(instance)).fields.values():
            if f.reference:
                yield from self._visible_models(getattr(instance, f.name))

    def _reachable(self, entries):
        """``{_id: (instance, writable)}`` for everything reachable from the
        entries. An instance is writable when any writable entry reaches it."""
        result = {}
        for writable in (True, False):
            stack = [
                instance
                for value, read_only in entries.values()
                if read_only != writable
                for instance in self._visible_models(value)
            ]
            while stack:
                instance = stack.pop()
                if instance._id in result:
                    continue
                result[instance._id] = (instance, writable)
                stack.extend(self._children(instance))
        return result

    def _lookup(self, target):
        """Return ``(value, reachable)``: an instance, or the collection an
        entry name exposes."""
        entries = self._entry_values()
        reachable = self._reachable(entries)
        if target in entries:
            value = entries[target][0]
            if value is None:
                msg = f"The instance exposed as {target!r} is currently empty"
                raise LookupError(msg)
            if not isinstance(value, _state_data_model()):
                return value, reachable
            if value._id in reachable:
                return value, reachable
        elif target in reachable:
            return reachable[target][0], reachable

        msg = (
            f"Unknown target {target!r}: not an exposed instance name or a reachable "
            f"instance id. Exposed instances: {sorted(entries)}. Use dataclass_get on "
            f"one to discover ids, and dataclass_schemas for the data structure."
        )
        raise LookupError(msg)

    def _instance(self, target):
        """Return ``(instance, writable)``; collections are not valid here."""
        value, reachable = self._lookup(target)
        if not isinstance(value, _state_data_model()):
            msg = f"{target!r} is a collection; target one of its items by id"
            raise LookupError(msg)
        return value, reachable[value._id][1]

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def _encode(self, value, reachable, depth, level):
        model_cls = _state_data_model()
        if isinstance(value, model_cls):
            if not self._exposed(type(value)):
                return _HIDDEN
            if depth - level > 1:
                return self._serialize(value, reachable, depth, level + 1)
            return {"id": value._id, "type": type(value).__name__}
        if isinstance(value, (list, tuple, set, frozenset)):
            items = [self._encode(v, reachable, depth, level) for v in value]
            items = [v for v in items if v is not _HIDDEN]
            return items if isinstance(value, (list, tuple)) else sorted(items, key=str)
        if isinstance(value, dict):
            items = {k: self._encode(v, reachable, depth, level) for k, v in value.items()}
            return {k: v for k, v in items.items() if v is not _HIDDEN}
        return value

    def _serialize(self, instance, reachable, depth=1, level=0):
        values = {}
        for name in self._policy(type(instance)).fields:
            encoded = self._encode(getattr(instance, name), reachable, depth, level)
            values[name] = None if encoded is _HIDDEN else encoded
        # Metadata and model fields are structurally separated so a field can
        # never collide with protocol keys, whatever it is named
        result = {"id": instance._id, "type": type(instance).__name__, "values": values}
        found = reachable.get(instance._id)
        if found is not None and not found[1]:
            result["read_only"] = True
        return result

    def _encode_result(self, value, reachable):
        """A method's or constructor's return value for the agent: an
        instance as dataclass_get shows it, nested ones as references."""
        if isinstance(value, _state_data_model()):
            if not self._exposed(type(value)):
                return None
            return self._serialize(value, reachable)
        encoded = self._encode(value, reachable, 1, 0)
        return None if encoded is _HIDDEN else encoded

    # -------------------------------------------------------------------------
    # Class discovery and startup audit
    # -------------------------------------------------------------------------

    def _known_classes(self, reachable):
        """Exposed classes the agent can encounter: those passed to
        ``expose_model()``, those of reachable instances, and those declared
        by visible reference fields, each with its exposed subclasses."""
        pending = [*self._models, *(type(i) for i, _ in reachable.values())]
        known = set()
        while pending:
            for klass in _subclasses(pending.pop()):
                if klass in known or not self._exposed(klass):
                    continue
                known.add(klass)
                for f in self._policy(klass).fields.values():
                    if f.reference:
                        checker = klass.TYPE_CHECKING.get(f.name)
                        pending.extend(_model_types(checker.type_def if checker else None))
        return known

    def _audit(self, **_):
        """Log likely exposure mistakes once the application is set up."""
        if not self._registered:
            return
        try:
            entries = self._entry_values()
            for name, (value, _) in entries.items():
                if value is not None and not isinstance(
                    value, (_state_data_model(), list, tuple, set, frozenset, dict)
                ):
                    logger.warning(
                        "trame-mcp: exposed instance %r resolves to %s, not a model "
                        "instance or a collection of them",
                        name,
                        type(value).__name__,
                    )
            classes = self._known_classes(self._reachable(entries))
        except Exception:
            logger.exception("trame-mcp: invalid model exposure")
            return

        for cls in sorted(classes, key=lambda c: c.__name__):
            policy = self._policy(cls)
            needs = {f"{cls.__name__}.{t.attr}": t.requires for t in policy.tools.values()}
            if policy.constructor is not None:
                needs[f"{cls.__name__}.{policy.constructor.attr}"] = policy.constructor.requires
            for what, requires in needs.items():
                unknown = [n for n in requires if n not in self._mcp.permissions]
                if unknown:
                    logger.warning(
                        "trame-mcp: %s requires undeclared permissions %s; it is never offered",
                        what,
                        unknown,
                    )
            for name, types in policy.hidden_references.items():
                exposed = sorted(
                    k.__name__ for t in types for k in _subclasses(t) if self._exposed(k)
                )
                if exposed:
                    logger.warning(
                        "trame-mcp: %s.%s holds exposed models (%s) but is not exposed, so "
                        "they are unreachable through it; mark it with mcp_field() or "
                        "hide it explicitly (hidden=True)",
                        cls.__name__,
                        name,
                        ", ".join(exposed),
                    )
            for f in policy.fields.values():
                if not f.reference:
                    continue
                checker = cls.TYPE_CHECKING.get(f.name)
                types = _model_types(checker.type_def if checker else None)
                if types and not any(self._exposed(k) for t in types for k in _subclasses(t)):
                    logger.warning(
                        "trame-mcp: %s.%s references models that are not exposed (%s); "
                        "their instances stay invisible",
                        cls.__name__,
                        f.name,
                        ", ".join(sorted(t.__name__ for t in types)),
                    )

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    def _register(self):
        if self._registered:
            return
        self._registered = True

        target_prop = {
            "type": "string",
            "description": "Exposed instance name or instance id",
        }
        self._mcp._add_tool(
            self._schemas,
            name="dataclass_schemas",
            description=SCHEMAS_DESCRIPTION,
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            annotations={"readOnlyHint": True},
        )
        self._mcp._add_tool(
            self._get,
            name="dataclass_get",
            description=GET_DESCRIPTION,
            input_schema={
                "type": "object",
                "properties": {
                    "target": target_prop,
                    "depth": {
                        "type": "integer",
                        "default": 1,
                        "minimum": 1,
                        "maximum": MAX_DEPTH,
                        "description": "How many nested instance levels to inline",
                    },
                },
                "required": ["target"],
                "additionalProperties": False,
            },
            annotations={"readOnlyHint": True},
            log_summary=summarize_get,
        )
        self._mcp._add_tool(
            self._update,
            name="dataclass_update",
            description=UPDATE_DESCRIPTION,
            input_schema={
                "type": "object",
                "properties": {
                    "target": target_prop,
                    "updates": {
                        "type": "object",
                        "description": "Map of writable field names to new values",
                    },
                },
                "required": ["target", "updates"],
                "additionalProperties": False,
            },
            annotations={"idempotentHint": True},
            log_summary=summarize_update,
        )
        self._mcp._add_tool(
            self._call,
            name="dataclass_call",
            description=CALL_DESCRIPTION,
            input_schema={
                "type": "object",
                "properties": {
                    "target": target_prop,
                    "method": {"type": "string"},
                    "args": {
                        "type": ["object", "null"],
                        "description": "Map of parameter names to values",
                    },
                },
                "required": ["target", "method"],
                "additionalProperties": False,
            },
            # Locked per method, by _call: the method decides
            serialize=False,
            log_summary=summarize_call,
        )
        self._mcp._add_tool(
            self._batch,
            name="dataclass_batch",
            description=BATCH_DESCRIPTION,
            input_schema={
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "The steps, run in order (see the tool description)",
                    },
                    "results": {
                        "enum": ["links", "full"],
                        "default": "links",
                        "description": "How results show instances: their references to "
                        "other instances (links), or every field (full)",
                    },
                },
                "required": ["steps"],
                "additionalProperties": False,
            },
            # The whole batch holds the tool lock, once
            log_summary=summarize_batch,
        )

    # -------------------------------------------------------------------------
    # Tool implementations
    # -------------------------------------------------------------------------

    def _describe_model(self, cls):
        policy = self._policy(cls)
        fields = {}
        for f in policy.fields.values():
            entry = dict(f.schema)
            if f.description:
                entry["description"] = f.description
            if f.access == "read-only":
                entry["read_only"] = True
            fields[f.name] = entry

        permissions = self._mcp.permissions
        methods = {}
        for t in policy.tools.values():
            if not permissions.allows(t.requires):
                continue
            methods[t.name] = {
                "description": t.description,
                "parameters": function_input_schema(getattr(cls, t.attr)),
            }
            if t.read_only:
                methods[t.name]["read_only"] = True

        model = {"fields": fields, "methods": methods}
        if policy.constructor is not None and permissions.allows(policy.constructor.requires):
            model["constructor"] = {
                "name": policy.constructor.attr,
                "description": policy.constructor.description,
                "parameters": self._constructor_schema(cls),
            }
        description = self._model_description(cls)
        if description:
            model = {"description": description, **model}
        return model

    def _model_description(self, cls):
        """The class's own ``description`` option, else its own docstring;
        neither is inherited."""
        own = self._models.get(cls) or cls.__dict__.get(MODEL_MARKER)
        if own is not None and own.description:
            return own.description
        doc = cls.__dict__.get("__doc__")
        return inspect.cleandoc(doc) if doc else None

    def _catalog_base(self, cls):
        """Nearest exposed ancestor with fields, which ``cls`` can extend."""
        model_cls = _state_data_model()
        for base in cls.__mro__[1:]:
            if base is model_cls or not issubclass(base, model_cls):
                continue
            if hasattr(base, "FIELD_NAMES") and self._exposed(base):
                return base
        return None

    def _schemas(self):
        entries = self._entry_values()
        reachable = self._reachable(entries)

        instances = {}
        for name, (value, read_only) in entries.items():
            if value is None:
                entry = {"empty": True}
            elif isinstance(value, _state_data_model()):
                if value._id not in reachable:
                    continue  # class not exposed
                entry = {"id": value._id, "type": type(value).__name__}
            else:
                kind = "dict" if isinstance(value, dict) else "list"
                entry = {"collection": kind, "size": len(list(self._visible_models(value)))}
            if read_only:
                entry["read_only"] = True
            instances[name] = entry

        # Subclasses are described as deltas on their nearest exposed base,
        # which is listed even when the agent cannot meet it directly.
        classes = self._known_classes(reachable)
        for cls in list(classes):
            base = self._catalog_base(cls)
            while base is not None and base not in classes:
                classes.add(base)
                base = self._catalog_base(base)
        full = {cls: self._describe_model(cls) for cls in classes}
        models = {}
        for cls in sorted(classes, key=lambda c: c.__name__):
            base = self._catalog_base(cls)
            delta = _extending(full[cls], base.__name__, full[base]) if base else None
            models[cls.__name__] = full[cls] if delta is None else delta
        return {"instances": instances, "models": models}

    def _get(self, target, depth=1):
        value, reachable = self._lookup(target)
        depth = max(1, min(int(depth), MAX_DEPTH))
        if isinstance(value, _state_data_model()):
            return self._serialize(value, reachable, depth=depth)
        return {"name": target, "items": self._encode(value, reachable, depth, 0)}

    def _constructor_schema(self, cls):
        """The parameters of ``cls``'s constructor, its server left out."""
        ctor = self._policy(cls).constructor
        return function_input_schema(getattr(cls, ctor.attr), skip=(ctor.server,))

    async def _update(self, target, updates):
        if not isinstance(updates, dict):
            msg = "`updates` must be an object mapping field names to values"
            raise ValidationError(msg)
        instance, writable = self._instance(target)
        return await self._apply_update(instance, writable, updates)

    async def _apply_update(self, instance, writable, updates):
        if not isinstance(updates, dict):
            msg = "`updates` must be an object mapping field names to values"
            raise ValidationError(msg)
        cls = type(instance)
        if not writable:
            msg = f"Nothing was applied: {cls.__name__} {instance._id} is exposed read-only."
            raise PermissionError(msg)

        fields = self._policy(cls).fields
        writable_fields = sorted(n for n, f in fields.items() if f.access == "read-write")
        rejected = []
        for name in updates:
            f = fields.get(name)
            if f is None:
                rejected.append(f"{name} (unknown field)")
            elif f.reference:
                rejected.append(f"{name} (nested model reference, use dataclass_call)")
            elif f.access != "read-write":
                rejected.append(f"{name} (read-only)")
        if rejected:
            msg = (
                f"Nothing was applied. Rejected fields on {cls.__name__}: "
                f"{', '.join(rejected)}. Writable fields: {writable_fields}"
            )
            raise ValidationError(msg)

        values, errors = {}, []
        for name, value in updates.items():
            try:
                checked = validate(value, fields[name].schema, name)
                values[name] = from_json(checked, fields[name].type_def)
            except ValidationError as e:
                errors.append(str(e))
        if errors:
            msg = f"Nothing was applied: {'; '.join(errors)}"
            raise ValidationError(msg)

        for name, value in values.items():
            setattr(instance, name, value)
        await _settle(instance)

        return {"id": instance._id, "updated": sorted(values)}

    def _method(self, target, method):
        """The instance ``target`` names and its exposed ``method``."""
        instance, writable = self._instance(target)
        return self._method_of(instance, writable, method)

    def _method_of(self, instance, writable, method):
        cls = type(instance)
        tools = self._policy(cls).tools

        if method not in tools:
            msg = f"Unknown method {method!r} on {cls.__name__}. Available methods: {sorted(tools)}"
            raise LookupError(msg)

        exposed = tools[method]
        if not writable and not exposed.read_only:
            allowed = sorted(name for name, t in tools.items() if t.read_only)
            msg = (
                f"{cls.__name__} {instance._id} is exposed read-only; "
                f"only read-only methods can be called: {allowed}"
            )
            raise PermissionError(msg)
        self._mcp.permissions.require(*exposed.requires, what=f"{cls.__name__}.{method}")
        return instance, exposed

    async def _call(self, target, method, args=None):
        # dataclass_call skips the protocol's lock: the method's own
        # serialize option decides, so it is looked up first, then again
        # under the lock, where the graph is as earlier calls left it.
        _, exposed = self._method(target, method)
        async with self._mcp.protocol.serialized(exposed.serialize):
            instance, exposed = self._method(target, method)
            _, encoded = await self._invoke(instance, exposed, args, "args")
        return encoded

    async def _invoke(self, instance, exposed, args, label, scope=None):
        """Call an exposed method: bind, resolve instance ids, run, settle,
        and encode what it returns."""
        fn = getattr(instance, exposed.attr)
        args = bind_arguments(fn, args, function_input_schema(fn), label=label)
        args = self.resolve_arguments(fn, args, read_only=exposed.read_only, scope=scope, label=label)
        result = fn(**args)
        if inspect.isawaitable(result):
            result = await result
        await _settle(instance)
        return result, self._encode_result(result, self._reachable_in(scope))

    # -------------------------------------------------------------------------
    # Instances as arguments
    # -------------------------------------------------------------------------

    def _reachable_in(self, scope):
        """Reachable instances, plus those a batch built (writable)."""
        reachable = self._reachable(self._entry_values())
        if scope is not None:
            for _id, instance in scope.built.items():
                reachable.setdefault(_id, (instance, True))
        return reachable

    def resolve_arguments(self, fn, arguments, *, read_only=False, scope=None, label="arguments"):
        """``arguments`` of ``fn`` with the ids given for its model-typed
        parameters replaced by the instances. An instance must be reachable
        (or built by the batch ``scope``) and of the annotated class; one
        reachable only through read-only entries goes to ``read_only``
        callables only."""
        entries = None
        reachable = None

        def resolve(value, cls, path):
            nonlocal entries, reachable
            if reachable is None:
                entries = self._entry_values()
                reachable = self._reachable_in(scope)
            found = reachable.get(value)
            if found is None and value in entries:
                instance = entries[value][0]
                if isinstance(instance, _state_data_model()):
                    found = reachable.get(instance._id)
            if found is None:
                msg = f"{path}: {value!r} is not the id of a reachable instance (a {cls.__name__})"
                raise ValidationError(msg)
            instance, writable = found
            if not isinstance(instance, cls):
                msg = f"{path}: {value!r} is a {type(instance).__name__}, not a {cls.__name__}"
                raise ValidationError(msg)
            if not writable and not read_only:
                msg = (
                    f"{path}: {type(instance).__name__} {instance._id} is exposed read-only; "
                    f"only read-only methods can take it"
                )
                raise PermissionError(msg)
            return instance

        return resolve_model_arguments(fn, arguments, resolve, label)

    # -------------------------------------------------------------------------
    # dataclass_batch
    # -------------------------------------------------------------------------

    def _constructible(self):
        """Exposed classes with a constructor, by name (as dataclass_schemas
        lists them); a name shared by two classes maps to None."""
        classes = self._known_classes(self._reachable(self._entry_values()))
        for cls in list(classes):
            base = self._catalog_base(cls)
            while base is not None and base not in classes:
                classes.add(base)
                base = self._catalog_base(base)
        found = {}
        for cls in classes:
            if self._policy(cls).constructor is None:
                continue
            found[cls.__name__] = None if cls.__name__ in found else cls
        return found

    @staticmethod
    def _references(value):
        """Every "@..." reference string inside a step value."""
        if isinstance(value, str):
            if value.startswith("@") and not value.startswith("@@"):
                yield value
        elif isinstance(value, list):
            for item in value:
                yield from DataclassLayer._references(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from DataclassLayer._references(item)

    def _check_steps(self, steps):
        """Validate the shape of every step before any runs."""

        def fail(message):
            msg = f"Nothing was applied: {message}"
            raise ValidationError(msg)

        if not isinstance(steps, list):
            fail("`steps` must be an array of step objects")
        constructible = None
        defined = set()
        built = {}  # "as" name of a new step -> its class
        for index, step in enumerate(steps):
            path = f"steps[{index}]"
            if not isinstance(step, dict):
                fail(f"{path}: expected an object, got {type(step).__name__}")
            kinds = [kind for kind in _STEP_KEYS if kind in step]
            if len(kinds) != 1:
                fail(f"{path}: a step has exactly one of 'new', 'call' or 'update'")
            kind = kinds[0]
            unexpected = sorted(set(step) - set(_STEP_KEYS[kind]))
            if unexpected:
                fail(f"{path}: unexpected {unexpected} in a {kind} step (allowed: {list(_STEP_KEYS[kind])})")
            if not isinstance(step[kind], str) or not step[kind]:
                fail(f"{path}.{kind}: expected a non-empty string")
            if kind == "call" and (not isinstance(step.get("method"), str) or not step["method"]):
                fail(f"{path}: a call step needs 'method'")
            if kind == "update" and not isinstance(step.get("values"), dict):
                fail(f"{path}: an update step needs 'values', an object of field values")
            if step.get("args") is not None and not isinstance(step["args"], dict):
                fail(f"{path}.args: expected an object")
            if kind == "new":
                if constructible is None:
                    constructible = self._constructible()
                name = step["new"]
                if name not in constructible:
                    fail(f"{path}: no model {name!r} has a constructor; models with one: {sorted(constructible)}")
                if constructible[name] is None:
                    fail(f"{path}: several models are named {name!r}")
                cls = constructible[name]
                self._check_permissions(self._policy(cls).constructor.requires, f"{path} (new {name})")
                if "as" in step:
                    built[step["as"]] = cls
                checked = [step.get("args")]
            else:
                checked = [step[kind], step.get("args"), step.get("values")]
            if kind == "call":
                cls = self._class_before_run(step["call"], built)
                exposed = self._policy(cls).tools.get(step["method"]) if cls else None
                if exposed is not None:
                    what = f"{path} (call {step['method']} on {step['call']})"
                    self._check_permissions(exposed.requires, what)
            for reference in self._references(checked):
                if reference[1:].split(".")[0] not in defined:
                    fail(f"{path}: {reference!r} names no earlier step (give that step \"as\")")
            if "as" in step:
                name = step["as"]
                if not isinstance(name, str) or not _STEP_NAME.match(name):
                    fail(f"{path}.as: expected a name of letters, digits and underscores")
                if name in defined:
                    fail(f"{path}.as: {name!r} already names an earlier step")
                defined.add(name)
        return constructible or {}

    def _check_permissions(self, requires, what):
        """A step's permissions, checked before any step runs."""
        try:
            self._mcp.permissions.require(*requires, what=what)
        except PermissionDeniedError as e:
            msg = f"Nothing was applied: {e}"
            raise PermissionDeniedError(msg) from None

    def _class_before_run(self, target, built):
        """The class of a step's target when it is known before the batch
        runs: an instance reachable now, or "@name" of a new step. None for
        the rest, which the step itself checks."""
        if target.startswith("@") and not target.startswith("@@"):
            return built.get(target[1:])
        try:
            instance, _ = self._instance(target.removeprefix("@"))
        except LookupError:
            return None
        return type(instance)

    def _follow(self, reference, scope, path):
        """The value an "@name.seg..." reference stands for."""
        name, *segments = reference[1:].split(".")
        value = scope.names[name]
        walked = f"@{name}"
        model_cls = _state_data_model()
        for segment in segments:
            if isinstance(value, model_cls):
                policy = self._policy(type(value))
                f = policy.fields.get(segment) if policy else None
                if f is None:
                    msg = f"{path}: {walked} is a {type(value).__name__}, which has no field {segment!r}"
                    raise LookupError(msg)
                value = getattr(value, segment)
                if f.reference:  # as dataclass_get shows it: exposed instances only
                    if isinstance(value, (list, tuple)):
                        value = [
                            v for v in value if not isinstance(v, model_cls) or self._exposed(type(v))
                        ]
                    elif isinstance(value, model_cls) and not self._exposed(type(value)):
                        value = None
            elif isinstance(value, dict):
                if segment not in value:
                    msg = f"{path}: {walked} has no key {segment!r}"
                    raise LookupError(msg)
                value = value[segment]
            elif isinstance(value, (list, tuple)):
                try:
                    value = value[int(segment)]
                except (ValueError, IndexError):
                    msg = f"{path}: {walked} has {len(value)} items, no index {segment!r}"
                    raise LookupError(msg) from None
            else:
                kind = "null" if value is None else type(value).__name__
                msg = f"{path}: {walked} is {kind}, which has no {segment!r}"
                raise LookupError(msg)
            walked = f"{walked}.{segment}"
        return value

    def _substitute(self, value, scope, path, ids=True):
        """``value`` with its references resolved; with ``ids``, instances
        become their ids (arguments travel as JSON)."""
        if isinstance(value, str):
            if value.startswith("@@"):
                return value[1:]
            if value.startswith("@"):
                value = self._follow(value, scope, path)
                return self._as_ids(value) if ids else value
            return value
        if isinstance(value, list):
            return [self._substitute(v, scope, f"{path}[{i}]", ids) for i, v in enumerate(value)]
        if isinstance(value, dict):
            return {k: self._substitute(v, scope, f"{path}.{k}", ids) for k, v in value.items()}
        return value

    def _as_ids(self, value):
        if isinstance(value, _state_data_model()):
            return value._id
        if isinstance(value, (list, tuple)):
            return [self._as_ids(v) for v in value]
        if isinstance(value, dict):
            return {k: self._as_ids(v) for k, v in value.items()}
        return value

    def _target(self, target, scope, path):
        """``(instance, writable)`` for a step's target: a name, an id, or a
        reference to an instance."""
        target = self._substitute(target, scope, path, ids=False)
        if isinstance(target, _state_data_model()):
            if target._id in scope.built:
                return target, True
            found = self._reachable(self._entry_values()).get(target._id)
            if found is None or found[0] is not target:
                msg = f"{path}: {type(target).__name__} {target._id} is not reachable"
                raise LookupError(msg)
            return found
        if not isinstance(target, str):
            msg = f"{path}: the target is {type(target).__name__}, not an instance"
            raise LookupError(msg)
        if target in scope.built:
            return scope.built[target], True
        return self._instance(target)

    async def _run_step(self, kind, step, scope, path, constructible):
        """Run one step; returns (encoded result, raw result for "as")."""
        if kind == "new":
            cls = constructible[step["new"]]
            ctor = self._policy(cls).constructor
            self._mcp.permissions.require(*ctor.requires, what=f"{cls.__name__}.{ctor.attr}")
            fn = getattr(cls, ctor.attr)
            label = f"{path}.args"
            args = self._substitute(step.get("args"), scope, label)
            args = bind_arguments(
                fn, args, self._constructor_schema(cls), label=label, positional=(None,)
            )
            args = self.resolve_arguments(fn, args, scope=scope, label=label)
            instance = fn(self._mcp.server, **args)
            if inspect.isawaitable(instance):
                instance = await instance
            if not isinstance(instance, cls):
                msg = (
                    f"{cls.__name__}.{ctor.attr} returned {type(instance).__name__}, "
                    f"not a {cls.__name__}"
                )
                raise TypeError(msg)
            scope.built[instance._id] = instance
            await _settle(instance)
            return self._serialize(instance, self._reachable_in(scope)), instance

        instance, writable = self._target(step[kind], scope, f"{path}.{kind}")
        if kind == "update":
            values = self._substitute(step["values"], scope, f"{path}.values")
            result = await self._apply_update(instance, writable, values)
            return result, result
        _, exposed = self._method_of(instance, writable, step["method"])
        args = self._substitute(step.get("args"), scope, f"{path}.args")
        raw, encoded = await self._invoke(instance, exposed, args, f"{path}.args", scope)
        return encoded, raw

    async def _batch(self, steps, results="links"):
        shown = _links_only if results == "links" else (lambda encoded: encoded)
        constructible = self._check_steps(steps)
        scope = _Scope()
        encoded_results = []
        for index, step in enumerate(steps):
            if index:
                await asyncio.sleep(0)  # a cancel lands between steps
            kind = next(k for k in _STEP_KEYS if k in step)
            path = f"steps[{index}]"
            try:
                encoded, raw = await self._run_step(kind, step, scope, path, constructible)
            except Exception as e:
                failure = self._batch_failure(index, kind, step, e, shown(encoded_results))
                raise BatchError(failure) from e
            encoded_results.append(encoded)
            if "as" in step:
                scope.names[step["as"]] = raw
        reachable = self._reachable(self._entry_values())
        unattached = [
            {"id": instance._id, "type": type(instance).__name__}
            for instance in scope.built.values()
            if instance._id not in reachable
        ]
        return {"results": shown(encoded_results), "unattached": unattached}

    @staticmethod
    def _batch_failure(index, kind, step, error, results):
        if isinstance(error, (ToolError, PermissionError)):
            reason = str(error)
        else:
            if not isinstance(error, (ValueError, LookupError)):
                logger.error("dataclass_batch step %d raised", index, exc_info=error)
            reason = f"{type(error).__name__}: {error}"
        if kind == "new":
            what = f"new {step['new']}"
        elif kind == "call":
            what = f"call {step['method']} on {step['call']}"
        else:
            what = f"update {step['update']}"
        message = f"steps[{index}] ({what}) failed: {reason}"
        if not results:
            return f"{message}. Nothing was applied."
        applied = json.dumps(results, default=repr)
        return (
            f"{message}. Steps 0-{index - 1} stay applied (no rollback); "
            f"their results: {applied}"
        )


# ---------------------------------------------------------------------------
# The call log's view of the generic tools (see trame_mcp.call_log)
# ---------------------------------------------------------------------------


def summarize_get(arguments) -> str:
    depth = arguments.get("depth")
    return f"{arguments.get('target')}" + (f" depth={depth}" if depth is not None else "")


def summarize_update(arguments) -> str:
    return f"{arguments.get('target')} {shorten(arguments.get('updates') or {})}"


def summarize_call(arguments) -> str:
    call = f"{arguments.get('target')}.{arguments.get('method')}"
    args = arguments.get("args")
    return f"{call} {shorten(args)}" if args else call


def summarize_batch(arguments) -> str:
    """The steps, one phrase each: ``new Model as name``, ``call
    target.method``, ``update target {fields}``."""
    steps = arguments.get("steps") or []
    phrases = []
    for step in steps[:MAX_LOGGED_STEPS]:
        if "new" in step:
            phrase = f"new {step['new']}"
        elif "call" in step:
            phrase = f"call {step['call']}.{step.get('method')}"
        elif "update" in step:
            fields = ", ".join(step.get("values") or {})
            phrase = f"update {step['update']} {{{fields}}}"
        else:
            phrase = "?"
        if step.get("as"):
            phrase += f" as {step['as']}"
        phrases.append(phrase)
    if len(steps) > MAX_LOGGED_STEPS:
        phrases.append(f"+{len(steps) - MAX_LOGGED_STEPS} steps")
    return "; ".join(phrases)


# ---------------------------------------------------------------------------
# dataclass_batch's compact results
# ---------------------------------------------------------------------------


def _is_reference(value) -> bool:
    return isinstance(value, dict) and value.keys() == {"id", "type"}


def _is_instance(value) -> bool:
    return (
        isinstance(value, dict)
        and {"id", "type", "values"} <= value.keys()
        and isinstance(value["values"], dict)
    )


def _links_only(results, shown=None):
    """Encoded batch results as ``results: "links"`` shows them: an
    instance keeps its references to other instances (the ids an agent
    links with) and drops its other fields; one already shown earlier in
    the batch keeps only its id."""
    shown = set() if shown is None else shown
    if _is_instance(results):
        if results["id"] in shown:
            return {"id": results["id"], "type": results["type"]}
        shown.add(results["id"])
        links = {
            name: value
            for name, value in results["values"].items()
            if _is_reference(value)
            or (isinstance(value, list) and value and all(_is_reference(v) for v in value))
        }
        view = {"id": results["id"], "type": results["type"], "values": links}
        if results.get("read_only"):
            view["read_only"] = True
        return view
    if isinstance(results, dict):
        return {key: _links_only(value, shown) for key, value in results.items()}
    if isinstance(results, list):
        return [_links_only(value, shown) for value in results]
    return results
