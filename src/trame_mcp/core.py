"""The MCP class and the per-server ``get_mcp`` factory."""

from __future__ import annotations

import argparse
import contextlib
import inspect
import logging
import os
import typing
import weakref
from dataclasses import dataclass, field

from trame_mcp.call_log import LEVELS, CallLog
from trame_mcp.dataclass_tools import DataclassLayer
from trame_mcp.decorators import get_markers
from trame_mcp.endpoint import WILDCARD_HOSTS, MCPEndpoint
from trame_mcp.permissions import DENY_ENV, Permissions, launch_names, requirements
from trame_mcp.protocol import MCPProtocol
from trame_mcp.schema import function_input_schema, type_to_schema
from trame_mcp.state_tools import StateLayer

logger = logging.getLogger(__name__)

__all__ = ["MCP", "MCPDisabledError", "get_mcp"]

TOKEN_ENV = "TRAME_MCP_TOKEN"
LOG_ENV = "TRAME_MCP_LOG"

LIVE_APP_INSTRUCTIONS = (
    "This MCP server belongs to a running trame application: every change made "
    "through it appears immediately in the browsers of the people using the app, "
    "who may be changing the same data."
)

_INSTANCES = weakref.WeakKeyDictionary()  # trame server -> MCP


class MCPDisabledError(PermissionError):
    """Raised when a tool, resource or prompt is used while ``mcp.enabled`` is False."""


@dataclass
class ToolEntry:
    name: str
    description: str
    fn: object
    input_schema: dict = field(default_factory=dict)
    annotations: dict | None = None
    serialize: bool = True  # take the tool lock (with serialize_tools)
    log_summary: object = None  # arguments -> text, for the call log
    requires: tuple = ()  # permission names (mcp.permissions)


@dataclass
class ResourceEntry:
    uri: str
    name: str
    description: str | None
    mime_type: str | None
    fn: object


@dataclass
class PromptEntry:
    name: str
    description: str | None
    arguments: list
    fn: object


def _tool_annotations(read_only, annotations):
    result = dict(annotations or {})
    if read_only:
        result["readOnlyHint"] = True
    return result or None


class MCP:
    """MCP server bound to a trame server. Use :func:`get_mcp` to obtain one.

    Exposing functions (``@mcp.tool``, ``@mcp.resource``, ``@mcp.prompt``),
    objects (:meth:`expose`), state (``mcp.state``) and trame-dataclass models
    (``mcp.dataclass``) only declares what an agent could use: nothing is
    served until ``enabled`` is set (in code, or with ``--mcp`` at launch).

    Every option is a settable attribute, so any code holding the instance can
    configure it; ``path`` and ``http`` are fixed once the trame server has
    started (the route is mounted then), everything else can change at
    runtime.
    """

    def __init__(self, server, **config):
        self._server = server.root_server
        self._started = self._server.running
        self._name = None
        self._path = "/mcp"
        self._http = True
        self._enabled = False
        self._token = None
        self._allowed_origins = ()
        self._serialize_tools = True
        self._call_log = CallLog()
        self._permissions = Permissions()
        self.instructions = None

        self._tools: dict[str, ToolEntry] = {}
        self._resources: dict[str, ResourceEntry] = {}
        self._prompts: dict[str, PromptEntry] = {}

        self._state_layer = StateLayer(self)
        self._dataclass_layer = DataclassLayer(self)
        self._protocol = MCPProtocol(self)
        self._endpoint = MCPEndpoint(self)

        self._apply_launch_options()
        self.configure(**config)

        ctrl = self._server.controller
        ctrl.on_server_bind.add(self._on_server_bind)
        ctrl.on_server_ready.add(self._endpoint.on_server_ready)
        ctrl.on_server_ready.add(self._dataclass_layer._audit)

    def _on_server_bind(self, wslink_server):
        self._start()
        self._endpoint.on_server_bind(wslink_server)

    def _start(self):
        """The server starts: from here on, the launch options that turn
        something on or set a value no longer override assignments (those
        withholding something, ``--no-mcp`` and ``--mcp-deny``, hold)."""
        self._started = True
        self._permissions._start()
        for entry in self._tools.values():
            unknown = [n for n in entry.requires if n not in self._permissions]
            if unknown:
                logger.warning(
                    "trame-mcp: tool %s requires undeclared permissions %s; it is never offered",
                    entry.name,
                    unknown,
                )

    def configure(
        self,
        *,
        name=None,
        path=None,
        http=None,
        enabled=None,
        token=None,
        allowed_origins=None,
        serialize_tools=None,
        instructions=None,
        log=None,
    ):
        """Set any subset of options; ``None`` leaves an option unchanged.

        Same as assigning the attributes one by one.
        """
        options = {
            "name": name,
            "path": path,
            "http": http,
            "enabled": enabled,
            "token": token,
            "allowed_origins": allowed_origins,
            "serialize_tools": serialize_tools,
            "instructions": instructions,
            "log": log,
        }
        for key, value in options.items():
            if value is not None:
                setattr(self, key, value)

    def _apply_launch_options(self):
        """``enabled`` / ``token`` / ``log`` / denied permissions from the
        command line and environment.

        They are the end user's choice, so until the server starts they win
        over whatever the application assigns (its values are defaults). The
        ones withholding something, ``--no-mcp`` and ``--mcp-deny``, hold
        for the whole run.
        """
        cli = self._server.cli
        with contextlib.suppress(argparse.ArgumentError):
            cli.add_argument(
                "--mcp",
                dest="mcp_enabled",
                action=argparse.BooleanOptionalAction,
                default=None,
                help="Enable or disable agent access through MCP",
            )
        with contextlib.suppress(argparse.ArgumentError):
            cli.add_argument(
                "--mcp-token",
                dest="mcp_token",
                help=f"Bearer token required on the MCP endpoint (or set {TOKEN_ENV})",
            )
        with contextlib.suppress(argparse.ArgumentError):
            cli.add_argument(
                "--mcp-log",
                dest="mcp_log",
                nargs="?",
                const="calls",
                choices=LEVELS,
                default=None,
                help=f"Log what agents call: calls (the default when given), all or off "
                f"(or set {LOG_ENV})",
            )
        with contextlib.suppress(argparse.ArgumentError):
            cli.add_argument(
                "--mcp-deny",
                dest="mcp_deny",
                action="append",
                metavar="PERMISSION",
                default=None,
                help="Deny agents permissions the application declares: comma-separated, "
                f"repeatable (or set {DENY_ENV})",
            )
        args, _ = cli.parse_known_args()
        self._launch_options = {
            "enabled": getattr(args, "mcp_enabled", None),
            "token": getattr(args, "mcp_token", None) or os.environ.get(TOKEN_ENV) or None,
            "log": getattr(args, "mcp_log", None) or self._log_from_environment(),
        }
        if self._launch_options["enabled"] is not None:
            self._enabled = self._launch_options["enabled"]
        if self._launch_options["token"] is not None:
            self._token = str(self._launch_options["token"])
        if self._launch_options["log"] is not None:
            self._call_log.set_level(self._launch_options["log"])
        # Both: each one is meant to withhold something
        self._permissions._deny_at_launch(
            launch_names([*(getattr(args, "mcp_deny", None) or ()), os.environ.get(DENY_ENV)])
        )

    @staticmethod
    def _log_from_environment():
        value = os.environ.get(LOG_ENV) or None
        if value is not None and value not in LEVELS:
            logger.warning("trame-mcp: ignoring %s=%r (one of %s)", LOG_ENV, value, ", ".join(LEVELS))
            return None
        return value

    def _launch_overrides(self, option):
        """True when a launch option decides ``option`` and the server has not
        started yet: the application's assignment is then only a default."""
        if self._started or self._launch_options.get(option) is None:
            return False
        logger.info("trame-mcp: %s is set at launch; the application's value is ignored", option)
        return True

    # -------------------------------------------------------------------------
    # Options
    # -------------------------------------------------------------------------

    @property
    def server(self):
        return self._server

    @property
    def protocol(self):
        return self._protocol

    @property
    def state(self):
        """Plain trame state exposure: ``expose()``, ``unexpose()`` (see
        :class:`trame_mcp.state_tools.StateLayer`)."""
        return self._state_layer

    @property
    def dataclass(self):
        """trame-dataclass exposure: ``expose_model()``, ``expose_instance()``,
        ``unexpose_instance()`` (see :class:`trame_mcp.dataclass_tools.DataclassLayer`)."""
        return self._dataclass_layer

    @property
    def name(self):
        """MCP server name reported to clients; defaults to the trame server name."""
        return self._name or self._server.name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def path(self):
        """Endpoint path on the trame web server."""
        return self._path

    @path.setter
    def path(self, value):
        self._ensure_not_started("path")
        self._path = value if value.startswith("/") else f"/{value}"

    @property
    def http(self):
        """Serve the MCP endpoint on the trame web server (default True).

        With ``http=False`` no route is ever mounted, so nothing outside the
        process can connect; everything exposed stays usable in-process (e.g.
        trame-llm's chat via :class:`trame_mcp.LLMAdapter`).
        """
        return self._http

    @http.setter
    def http(self, value):
        self._ensure_not_started("http")
        self._http = bool(value)

    @property
    def enabled(self):
        """Whether agents can use what has been declared (default False).

        The one switch for all agent access, settable at any time (before
        start, in a lifecycle hook, from a UI toggle). While False, the HTTP
        endpoint answers 503 and in-process callers
        (:class:`trame_mcp.LLMAdapter`) get :class:`MCPDisabledError`.

        ``--mcp`` at launch takes precedence over values the application
        sets before the server starts; once it is running, every assignment
        applies. ``--no-mcp`` keeps access off for the whole run
        (:attr:`enabled_locked`).
        """
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        if value and self.enabled_locked:
            logger.info("trame-mcp: --no-mcp keeps agent access off for this run")
            return
        if self._launch_overrides("enabled"):
            return
        self._enabled = bool(value)
        if self._enabled:
            self._endpoint.announce()

    @property
    def enabled_locked(self):
        """True when ``--no-mcp`` keeps agent access off for the whole run:
        turning it on is ignored, so a UI disables its switch."""
        return self._launch_options["enabled"] is False

    @property
    def token(self):
        """Bearer token required on the HTTP endpoint, or None.

        ``--mcp-token`` or the ``TRAME_MCP_TOKEN`` environment variable take
        precedence over values the application sets before the server
        starts. When the server listens on a non-loopback interface and no
        token is set, a random one is generated at startup and printed with
        the connection instructions.
        """
        return self._token

    @token.setter
    def token(self, value):
        if self._launch_overrides("token"):
            return
        self._token = str(value) if value else None

    @property
    def permissions(self):
        """The powers the application lets the user withhold from agents:
        ``declare()`` them, then name the one a tool, method or constructor
        needs with ``requires=`` (see :class:`trame_mcp.permissions.Permissions`).

        ``--mcp-deny`` or ``TRAME_MCP_DENY`` deny permissions for the whole
        run (``permissions.locked``).
        """
        return self._permissions

    @property
    def log(self):
        """What agents' calls are logged (default ``"off"``).

        ``"calls"`` logs one line per tool call, resource read and prompt
        when it ends: who called (the HTTP session, or ``in-process``), the
        arguments shortened, the outcome and the time taken. ``"all"`` also
        logs each call as it starts with its full arguments and ``_meta``,
        and every other protocol message. A bool sets ``"calls"`` or
        ``"off"``. The lines go to the ``trame_mcp.calls`` logger, on stderr
        unless the application gives it a handler.

        ``--mcp-log [calls|all|off]`` or ``TRAME_MCP_LOG`` at launch take
        precedence over values the application sets before the server
        starts; once it is running, every assignment applies.
        """
        return self._call_log.level

    @log.setter
    def log(self, value):
        if self._launch_overrides("log"):
            return
        self._call_log.set_level(value)

    @property
    def allowed_origins(self):
        """Extra browser origins (``"https://host:port"``) allowed to call the
        endpoint. Loopback origins and the endpoint's own origin are always
        allowed; requests without an ``Origin`` header (non-browser clients)
        are unaffected."""
        return self._allowed_origins

    @allowed_origins.setter
    def allowed_origins(self, value):
        if isinstance(value, str):
            value = (value,)
        self._allowed_origins = tuple(o.rstrip("/") for o in value)

    @property
    def serialize_tools(self):
        """When True (default), tool calls execute one at a time.

        Synchronous tools are always atomic on trame's event loop; the lock
        extends that guarantee across ``await`` points in async tools, so
        parallel calls from an LLM client cannot interleave read-modify-write
        logic. A tool declared with ``serialize=False`` skips the lock (a
        status poll or a wait that must not hold back the other calls);
        setting this to False opts every tool out.
        """
        return self._serialize_tools

    @serialize_tools.setter
    def serialize_tools(self, value):
        self._serialize_tools = bool(value)

    @property
    def url(self):
        host = self._endpoint.bind_host
        if not host or host in WILDCARD_HOSTS:
            host = "localhost"
        elif ":" in host:
            host = f"[{host}]"
        return f"http://{host}:{self._server.port}{self._path}"

    def _ensure_not_started(self, option):
        if self._started:
            msg = f"MCP {option!r} cannot change once the server has started"
            raise RuntimeError(msg)

    def _agent_instructions(self):
        """What clients receive as ``instructions``: the application's own
        ``instructions``, then a generated overview of this server."""
        usage = [
            f"- {line}"
            for line in (self._dataclass_layer._instructions(), self._state_layer._instructions())
            if line
        ]
        denied = self._permissions.denied
        if denied:
            names = ", ".join(f"{n} ({self._permissions.description(n)})" for n in denied)
            usage.append(
                f"- Turned off in this application: {names}. What needs them is not "
                f"offered, and calls to it are refused."
            )
        overview = "\n".join([LIVE_APP_INSTRUCTIONS, *usage])
        if self.instructions:
            return f"{self.instructions.strip()}\n\n{overview}"
        return overview

    def _check_enabled(self):
        if not self._enabled:
            msg = "MCP access is currently disabled by the application"
            raise MCPDisabledError(msg)
        self._permissions._check_launch()

    # -------------------------------------------------------------------------
    # Function decorators
    # -------------------------------------------------------------------------

    def tool(
        self,
        fn=None,
        *,
        name=None,
        description=None,
        read_only=False,
        annotations=None,
        serialize=True,
        requires=None,
    ):
        """Expose a plain function as a tool on this server (decorator).

        The server-bound counterpart of the :func:`trame_mcp.tool` marker,
        as ``@state.change`` is of trame's ``@change``; same options.
        """

        def decorate(f):
            self._add_tool(
                f,
                name=name,
                description=description,
                annotations=_tool_annotations(read_only, annotations),
                serialize=serialize,
                requires=requires,
            )
            return f

        if fn is None:
            return decorate
        return decorate(fn)

    def resource(self, uri, *, name=None, description=None, mime_type=None):
        """Expose a plain function as a resource on this server (decorator).

        The server-bound counterpart of :func:`trame_mcp.resource`.
        """

        def decorate(f):
            self._add_resource(f, uri=uri, name=name, description=description, mime_type=mime_type)
            return f

        return decorate

    def prompt(self, fn=None, *, name=None, description=None):
        """Expose a plain function as a prompt on this server (decorator).

        The server-bound counterpart of :func:`trame_mcp.prompt`.
        """

        def decorate(f):
            self._add_prompt(f, name=name, description=description)
            return f

        if fn is None:
            return decorate
        return decorate(fn)

    # -------------------------------------------------------------------------
    # Registry
    # -------------------------------------------------------------------------

    def _add_tool(
        self,
        fn,
        name=None,
        description=None,
        input_schema=None,
        annotations=None,
        prefix="",
        serialize=True,
        log_summary=None,
        requires=None,
    ):
        tool_name = f"{prefix}{name or fn.__name__}"
        if tool_name in self._tools:
            msg = (
                f"A tool named {tool_name!r} is already registered. "
                f"Disambiguate with @tool(name=...) or an expose() prefix."
            )
            raise ValueError(msg)
        self._tools[tool_name] = ToolEntry(
            name=tool_name,
            description=description or inspect.getdoc(fn) or "",
            fn=fn,
            input_schema=input_schema
            if input_schema is not None
            else function_input_schema(fn),
            annotations=annotations,
            serialize=bool(serialize),
            log_summary=log_summary,
            requires=requirements(requires),
        )

    def _add_resource(self, fn, uri, name=None, description=None, mime_type=None):
        if uri in self._resources:
            msg = f"A resource with uri {uri!r} is already registered."
            raise ValueError(msg)
        self._resources[uri] = ResourceEntry(
            uri=uri,
            name=name or fn.__name__,
            description=description or inspect.getdoc(fn),
            mime_type=mime_type,
            fn=fn,
        )

    def _add_prompt(self, fn, name=None, description=None, prefix=""):
        prompt_name = f"{prefix}{name or fn.__name__}"
        if prompt_name in self._prompts:
            msg = f"A prompt named {prompt_name!r} is already registered."
            raise ValueError(msg)

        try:
            hints = typing.get_type_hints(fn, include_extras=True)
        except Exception:
            hints = {}
        arguments = []
        for pname, param in inspect.signature(fn).parameters.items():
            if pname == "self" or param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            argument = {"name": pname, "required": param.default is inspect.Parameter.empty}
            # Annotated[str, "description"], as for tool parameters
            argument_description = type_to_schema(hints.get(pname)).get("description")
            if argument_description:
                argument["description"] = argument_description
            arguments.append(argument)
        self._prompts[prompt_name] = PromptEntry(
            name=prompt_name,
            description=description or inspect.getdoc(fn),
            arguments=arguments,
            fn=fn,
        )

    def list_tools(self):
        """The tools agents are offered: those whose permissions are all
        allowed. ``get_tool`` finds the others too, so a call to one is
        refused with the reason."""
        return [t for t in self._tools.values() if self._permissions.allows(t.requires)]

    def get_tool(self, name):
        return self._tools.get(name)

    def list_resources(self):
        return list(self._resources.values())

    def get_resource(self, uri):
        return self._resources.get(uri)

    def list_prompts(self):
        return list(self._prompts.values())

    def get_prompt(self, name):
        return self._prompts.get(name)

    # -------------------------------------------------------------------------
    # Exposure
    # -------------------------------------------------------------------------

    def expose(self, *targets, prefix=""):
        """Expose the ``@tool`` / ``@resource`` / ``@prompt`` methods of
        objects, or functions marked with them.

        .. code-block:: python

            mcp.expose(viewer, prefix="viewer_")
            mcp.expose(shared_tools.export_scene)  # marked in a module without a server

        ``MCPApp`` and ``MCPComponent`` expose themselves when built, and plain
        functions are usually exposed with ``@mcp.tool`` directly. Exposing
        declares only: nothing is served until ``enabled`` is set.

        :param targets: objects whose class has marked methods, or marked
            functions. trame-dataclass instances go through
            ``mcp.dataclass.expose_instance()`` instead.
        :param prefix: prepended to tool and prompt names, to namespace
            several objects of the same kind.
        """
        for target in targets:
            if inspect.isfunction(target) or inspect.ismethod(target):
                if not get_markers(target):
                    msg = (
                        f"{target.__qualname__} is not marked; decorate it with "
                        f"@tool, @resource or @prompt (or expose it with @mcp.tool)"
                    )
                    raise TypeError(msg)
                self._register(target, prefix)
            elif isinstance(target, type) or inspect.ismodule(target):
                msg = f"expose() takes functions or objects, got {target!r}"
                raise TypeError(msg)
            elif _is_state_data_model(target):
                msg = (
                    f"{type(target).__name__} is a trame-dataclass model; expose it with "
                    f"mcp.dataclass.expose_instance()"
                )
                raise TypeError(msg)
            elif not self._expose_object(target, prefix):
                msg = f"{type(target).__name__} has no methods marked with @tool, @resource or @prompt"
                raise ValueError(msg)

    def _expose_object(self, obj, prefix=""):
        """Expose the marked methods of ``obj``; return how many."""
        count = 0
        for method_name, _fn in inspect.getmembers(
            type(obj), lambda x: inspect.isfunction(x) or inspect.ismethod(x)
        ):
            bound = getattr(obj, method_name)
            if get_markers(bound):
                self._register(bound, prefix)
                count += 1
        return count

    def _register(self, fn, prefix):
        markers = get_markers(fn)
        if "tool" in markers:
            options = markers["tool"]
            self._add_tool(
                fn,
                name=options.get("name"),
                description=options.get("description"),
                annotations=_tool_annotations(
                    options.get("read_only"), options.get("annotations")
                ),
                prefix=prefix,
                serialize=options.get("serialize", True),
                requires=options.get("requires"),
            )
        if "resource" in markers:
            options = markers["resource"]
            self._add_resource(
                fn,
                uri=options["uri"],
                name=options.get("name"),
                description=options.get("description"),
                mime_type=options.get("mime_type"),
            )
        if "prompt" in markers:
            options = markers["prompt"]
            self._add_prompt(
                fn,
                name=options.get("name"),
                description=options.get("description"),
                prefix=prefix,
            )


def _is_state_data_model(obj):
    try:
        from trame_dataclass.v2 import StateDataModel
    except ImportError:
        return False
    return isinstance(obj, StateDataModel)


def get_mcp(server=None, **config) -> MCP:
    """Create-or-get the MCP instance bound to a trame server.

    Mirrors ``get_server()`` semantics: one MCP per (root) trame server.
    Keyword arguments are applied with :meth:`MCP.configure` on every call,
    so the order in which callers obtain the instance does not matter.

    Creating the instance serves nothing by itself: agent access stays off
    until ``enabled`` is set, in code or with ``--mcp`` at launch.
    """
    if server is None or isinstance(server, str):
        from trame.app import get_server

        server = get_server(server)

    server = server.root_server
    mcp = _INSTANCES.get(server)
    if mcp is None:
        mcp = MCP(server, **config)
        _INSTANCES[server] = mcp
    elif config:
        mcp.configure(**config)
    return mcp
