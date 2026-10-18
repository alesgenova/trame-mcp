"""The log of what agents call (``mcp.log``, ``--mcp-log``).

Every tool call, resource read and prompt is logged once, when it ends,
whichever way it came in: over HTTP (``MCPProtocol.handle``) or in-process
(:class:`trame_mcp.LLMAdapter`, which calls ``MCPProtocol.execute`` and
``read_resource`` directly). ``handle`` records who is asking in a context
variable, which follows the request into the task running it; a call
without one came from in-process.

Levels:

- ``"off"`` (default): nothing.
- ``"calls"``: one line per call when it ends, with its arguments
  shortened (or a tool's own summary), how it ended and how long it took.
- ``"all"``: also a line when a call starts, with its full arguments and
  the request's ``_meta``, and one per other protocol message
  (``initialize``, the lists, ``ping``, notifications).

The lines go to the ``trame_mcp.calls`` logger. Turning the log on gives
that logger a stderr handler unless the application gave it one of its own.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import logging
import sys
import time
from dataclasses import dataclass

from trame_mcp.permissions import PermissionDeniedError

__all__ = ["LEVELS", "LOGGER", "CallLog", "normalize_level", "shorten"]

LEVELS = ("off", "calls", "all")
LOGGER = logging.getLogger("trame_mcp.calls")

# How much of an argument "calls" shows
MAX_STRING = 60
MAX_ITEMS = 8
MAX_DEPTH = 3
MAX_WHAT = 300
MAX_ERROR = 200
# A wait for the tool lock worth mentioning, in seconds
QUEUED = 0.1
# The methods logged as calls; "all" logs the others as they arrive
CALL_METHODS = ("tools/call", "resources/read", "prompts/get")


@dataclass
class Request:
    """Who sent the request being handled (HTTP side)."""

    session: str | None
    rpc_id: object
    meta: dict | None


_request = contextvars.ContextVar("trame_mcp_request", default=None)
_call = contextvars.ContextVar("trame_mcp_call", default=None)


def normalize_level(value) -> str:
    """A level from ``value``: one of ``LEVELS``, or a bool (``True`` is
    ``"calls"``) or None (``"off"``)."""
    if value is None or value is False:
        return "off"
    if value is True:
        return "calls"
    if value in LEVELS:
        return value
    msg = f"log must be one of {', '.join(LEVELS)} (or a bool), got {value!r}"
    raise ValueError(msg)


def shorten(value, depth=0) -> str:
    """``value`` as compact JSON-like text: bare keys, long strings replaced
    by their length, long lists and dicts cut, deep nesting elided."""
    if isinstance(value, str):
        if len(value) > MAX_STRING:
            return f"<str {len(value)} chars>"
        return json.dumps(value)
    if isinstance(value, dict):
        if depth >= MAX_DEPTH:
            return "{...}"
        items = [f"{k}: {shorten(v, depth + 1)}" for k, v in list(value.items())[:MAX_ITEMS]]
        if len(value) > MAX_ITEMS:
            items.append(f"+{len(value) - MAX_ITEMS}")
        return "{" + ", ".join(items) + "}"
    if isinstance(value, (list, tuple)):
        if depth >= MAX_DEPTH:
            return "[...]"
        items = [shorten(v, depth + 1) for v in value[:MAX_ITEMS]]
        if len(value) > MAX_ITEMS:
            items.append(f"+{len(value) - MAX_ITEMS}")
        return "[" + ", ".join(items) + "]"
    if value is None or isinstance(value, (bool, int, float)):
        return json.dumps(value)
    return f"<{type(value).__name__}>"


def _full(value) -> str:
    return json.dumps(value, default=repr)


def _size(count: int) -> str:
    if count < 1024:
        return f"{count} B"
    if count < 1024 * 1024:
        return f"{count / 1024:.1f} KB"
    return f"{count / (1024 * 1024):.1f} MB"


def _describe_result(value) -> str:
    """What a call returned, in a few words: its size, and what kind of
    content when it is not plain data."""
    from trame_mcp.content import Image

    if value is None:
        return "empty"
    if isinstance(value, Image):
        return f"image, {_size(len(value.data))}"
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, (Image, str)) for v in value):
        images = sum(isinstance(v, Image) for v in value)
        texts = len(value) - images
        kinds = [
            f"{n} {kind}{'s' if n > 1 else ''}"
            for n, kind in ((images, "image"), (texts, "text"))
            if n
        ]
        count = sum(len(v.data) if isinstance(v, Image) else len(v.encode()) for v in value)
        return f"{' + '.join(kinds)}, {_size(count)}"
    text = value if isinstance(value, str) else json.dumps(value, default=repr)
    return _size(len(text.encode()))


def _describe_error(error) -> str:
    text = str(error).strip()
    first = text.splitlines()[0] if text else ""
    if not isinstance(error, (ValueError, LookupError, PermissionError)):
        first = f"{type(error).__name__}: {first}" if first else type(error).__name__
    if len(first) > MAX_ERROR:
        first = first[: MAX_ERROR - 1] + "…"
    return first


def _caller(request: Request | None, level: str) -> str:
    if request is None:
        return "in-process"
    session = request.session[:6] if request.session else "-"
    return session if level == "calls" else f"{session} #{request.rpc_id}"


def _emit(caller: str, text: str):
    LOGGER.info("mcp[%s] %s", caller, text)


class Call:
    """A call being logged, from its start to the line written when it ends.
    The caller sets ``result`` before a normal exit."""

    def __init__(self, level, kind, name, arguments, summarize):
        self.level = level
        self.request = _request.get()
        self.what = name if kind == "tool" else f"{kind} {name}"
        self.arguments = arguments or {}
        self.summarize = summarize
        self.result = None
        self.queued = 0.0
        self.started = time.monotonic()
        if level == "all":
            line = self.what
            if self.arguments:
                line += f" {_full(self.arguments)}"
            meta = self.request.meta if self.request is not None else None
            if meta:
                line += f" _meta={_full(meta)}"
            _emit(self.caller, f"{line} ...")

    @property
    def caller(self) -> str:
        return _caller(self.request, self.level)

    def _arguments(self) -> str:
        if self.summarize is not None:
            with contextlib.suppress(Exception):  # a summary never breaks a call
                return self.summarize(self.arguments)
        return shorten(self.arguments) if self.arguments else ""

    def end(self, outcome: str, detail: str | None = None, error: str | None = None):
        seconds = time.monotonic() - self.started
        what = self.what
        if self.level == "calls":
            arguments = self._arguments()
            if arguments:
                what = f"{what} {arguments}"
            if len(what) > MAX_WHAT:
                what = what[: MAX_WHAT - 1] + "…"
        extras = [detail] if detail else []
        if self.queued >= QUEUED:
            extras.append(f"queued {self.queued:.1f}s")
        text = f"{what} -> {outcome} {seconds:.2f}s"
        if extras:
            text += f" ({', '.join(extras)})"
        if error:
            text += f": {error}"
        _emit(self.caller, text)


class _Untracked:
    """Stands for a ``Call`` while the log is off."""

    result = None


class CallLog:
    """The level, and the logging of calls and protocol messages at it."""

    def __init__(self):
        self.level = "off"

    def set_level(self, value):
        self.level = normalize_level(value)
        if self.level != "off":
            self._ensure_output()

    @staticmethod
    def _ensure_output():
        """Make the lines visible: INFO passes the logger, and without a
        handler of the application's they go to stderr (not to the root
        logger too, which may print them a second time)."""
        if LOGGER.level == logging.NOTSET or LOGGER.level > logging.INFO:
            LOGGER.setLevel(logging.INFO)
        if LOGGER.handlers:
            return
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        LOGGER.addHandler(handler)
        LOGGER.propagate = False

    @contextlib.contextmanager
    def call(self, kind: str, name: str, arguments=None, summarize=None):
        """Log the call run in the block: ``kind`` is "tool", "resource" or
        "prompt"; ``summarize(arguments)`` shortens a tool's arguments its
        own way. Yields the ``Call``, whose ``result`` the block sets."""
        if self.level == "off":
            yield _Untracked()
            return
        call = Call(self.level, kind, name, arguments, summarize)
        token = _call.set(call)
        try:
            yield call
        except asyncio.CancelledError:
            call.end("cancelled")
            raise
        except PermissionDeniedError as error:
            call.end("denied", error=_describe_error(error))
            raise
        except BaseException as error:
            call.end("error", error=_describe_error(error))
            raise
        else:
            call.end("ok", _describe_result(call.result))
        finally:
            _call.reset(token)

    @contextlib.contextmanager
    def request(self, session, rpc_id, params):
        """Mark the block as handling an HTTP-side request, so the calls in
        it name their caller; with "all", log a message that is not a call."""
        meta = params.get("_meta") if isinstance(params, dict) else None
        request = Request(session, rpc_id, meta if isinstance(meta, dict) else None)
        token = _request.set(request)
        try:
            yield
        finally:
            _request.reset(token)

    def message(self, method: str, params):
        """With "all", log a protocol message that is not a call, as it
        arrives."""
        if self.level != "all" or method in CALL_METHODS:
            return
        line = method
        if params:
            line += f" {_full(params)}"
        _emit(_caller(_request.get(), self.level), line)

    @staticmethod
    def queued(seconds: float):
        """Add ``seconds`` waited for the tool lock to the current call."""
        call = _call.get()
        if call is not None:
            call.queued += seconds
