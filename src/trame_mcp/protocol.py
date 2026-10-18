"""MCP JSON-RPC method dispatch (transport-independent)."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import inspect
import json
import logging
import time

from trame_mcp.content import to_error_result, to_tool_result
from trame_mcp.schema import ToolError, ValidationError, bind_arguments

logger = logging.getLogger(__name__)

__all__ = ["MCPProtocol"]

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")

# JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
# The reply to a request the client cancelled (LSP's code; MCP defines none).
# The client ignores it, but an HTTP POST has to end with something.
REQUEST_CANCELLED = -32800


def _error(rpc_id, code, message):
    return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": message}}


def _result(rpc_id, payload):
    return {"jsonrpc": "2.0", "id": rpc_id, "result": payload}


async def _call(fn, arguments):
    value = fn(**arguments)
    if inspect.isawaitable(value):
        value = await value
    return value


def _describe_failure(name, error):
    """Turn a tool exception into the text the model sees; log accordingly."""
    if isinstance(error, (ValueError, LookupError, PermissionError)):
        # Expected, actionable failures (bad input, unknown target, ...)
        logger.info("Tool %s failed: %s", name, error)
        if isinstance(error, (ToolError, PermissionError)):
            return str(error)
    else:
        logger.error("Tool %s raised", name, exc_info=error)
    return f"{type(error).__name__}: {error}"


class MCPProtocol:
    """Dispatches MCP JSON-RPC messages against an ``MCP`` instance's registries."""

    def __init__(self, mcp):
        self._mcp = mcp
        self._tools_lock = asyncio.Lock()
        # (session, request id) -> task of a request still running, and the
        # tasks cancelled by the client -> its reason
        self._in_flight = {}
        self._cancelled = {}
        self._handlers = {
            "initialize": self._on_initialize,
            "ping": self._on_ping,
            "tools/list": self._on_tools_list,
            "tools/call": self._on_tools_call,
            "resources/list": self._on_resources_list,
            "resources/templates/list": self._on_resource_templates_list,
            "resources/read": self._on_resources_read,
            "prompts/list": self._on_prompts_list,
            "prompts/get": self._on_prompts_get,
        }

    async def handle(self, message: dict, *, session=None) -> dict | None:
        """Handle one JSON-RPC message. Returns a response dict, or None for
        notifications/responses (which expect no reply).

        ``session`` scopes request ids, so ``notifications/cancelled`` only
        reaches the requests of the client that sent it (the HTTP endpoint
        passes the ``Mcp-Session-Id`` header). A request the client cancels
        is answered with a ``REQUEST_CANCELLED`` error, which it ignores.
        """
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return _error(None, INVALID_REQUEST, "Invalid JSON-RPC 2.0 message")

        method = message.get("method")
        rpc_id = message.get("id")

        if method is None:
            # A response from the client (to a server request) - nothing to do.
            return None
        if not isinstance(method, str):
            return _error(rpc_id, INVALID_REQUEST, "Invalid method")

        # The calls made while handling it know who asked (the call log)
        call_log = self._mcp._call_log
        with call_log.request(session, rpc_id, message.get("params")):
            call_log.message(method, message.get("params"))
            return await self._dispatch(message, method, rpc_id, session)

    async def _dispatch(self, message, method, rpc_id, session):
        if method.startswith("notifications/"):
            if method == "notifications/cancelled":
                params = message.get("params")
                if isinstance(params, dict):
                    self.cancel(
                        params.get("requestId"), session=session, reason=params.get("reason")
                    )
            return None

        handler = self._handlers.get(method)
        if handler is None:
            return _error(rpc_id, METHOD_NOT_FOUND, f"Method not found: {method}")

        params = message.get("params") or {}
        if not isinstance(params, dict):
            return _error(rpc_id, INVALID_PARAMS, "params must be an object")

        try:
            payload = handler(params)
            if inspect.isawaitable(payload):
                payload = await self._run_cancellable(payload, (session, rpc_id))
        except _RequestCancelled as e:
            return _error(rpc_id, REQUEST_CANCELLED, str(e))
        except _InvalidParams as e:
            return _error(rpc_id, INVALID_PARAMS, str(e))
        except PermissionError as e:
            return _error(rpc_id, INVALID_REQUEST, str(e))
        except Exception as e:
            logger.exception("Internal error handling %s", method)
            return _error(rpc_id, INTERNAL_ERROR, f"{type(e).__name__}: {e}")

        return _result(rpc_id, payload)

    # -------------------------------------------------------------------------
    # Cancellation
    # -------------------------------------------------------------------------

    def cancel(self, request_id, *, session=None, reason=None) -> bool:
        """Cancel the running request ``request_id`` of ``session``, as
        ``notifications/cancelled`` does. Returns whether one was running.

        The request's task is cancelled: its tool stops at its next
        ``await`` (a synchronous tool has finished before any cancel can
        arrive), releasing the tool lock and flushing the state it changed.
        """
        try:
            task = self._in_flight.get((session, request_id))
        except TypeError:  # an unhashable id: no request has it
            return False
        if task is None or task.done():
            return False
        logger.info("Request %r cancelled by the client: %s", request_id, reason or "no reason")
        self._cancelled[task] = reason
        task.cancel()
        return True

    async def _run_cancellable(self, awaitable, key):
        """Await ``awaitable`` as a task that ``cancel(key)`` can stop."""
        task = asyncio.ensure_future(awaitable)
        try:
            registered = self._in_flight.setdefault(key, task) is task
        except TypeError:
            registered = False
        try:
            return await task
        except asyncio.CancelledError:
            current = asyncio.current_task()
            # Task.cancelling() is Python 3.11+; without it, assume not
            outer_cancelled = current is not None and getattr(current, "cancelling", int)()
            if task not in self._cancelled or outer_cancelled:
                raise  # cancelled from the outside, e.g. the client disconnected
            reason = self._cancelled[task]
            msg = f"Request cancelled: {reason}" if reason else "Request cancelled"
            raise _RequestCancelled(msg) from None
        finally:
            self._cancelled.pop(task, None)
            if registered and self._in_flight.get(key) is task:
                del self._in_flight[key]

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def _on_initialize(self, params):
        from trame_mcp import __version__

        requested = params.get("protocolVersion")
        version = requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        payload = {
            "protocolVersion": version,
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"listChanged": False},
                "prompts": {"listChanged": False},
            },
            "serverInfo": {
                "name": self._mcp.name,
                "version": __version__,
            },
        }
        payload["instructions"] = self._mcp._agent_instructions()
        return payload

    def _on_ping(self, _params):
        return {}

    # -------------------------------------------------------------------------
    # Tools
    # -------------------------------------------------------------------------

    def _on_tools_list(self, _params):
        tools = []
        for entry in self._mcp.list_tools():
            descriptor = {
                "name": entry.name,
                "description": entry.description,
                "inputSchema": entry.input_schema,
            }
            if entry.annotations:
                descriptor["annotations"] = entry.annotations
            tools.append(descriptor)
        return {"tools": tools}

    async def _on_tools_call(self, params):
        name = params.get("name")
        entry = self._mcp.get_tool(name)
        if entry is None:
            msg = f"Unknown tool: {name}"
            with self._mcp._call_log.call("tool", str(name), params.get("arguments")):
                raise _InvalidParams(msg)

        try:
            value = await self.execute(entry, params.get("arguments"))
        except Exception as e:
            return to_error_result(_describe_failure(name, e))

        return to_tool_result(value)

    async def execute(self, entry, arguments):
        """Run a tool entry: access and permission checks, argument
        validation, then the call under the server's serialization policy.

        Shared by the HTTP path (``tools/call``) and in-process consumers
        (:class:`trame_mcp.llm.LLMAdapter`). Returns the tool's raw value;
        raises on failure.
        """
        with self._mcp._call_log.call(
            "tool", entry.name, arguments, entry.log_summary
        ) as call:
            self._mcp._check_enabled()
            self._mcp.permissions.require(*entry.requires, what=entry.name)
            try:
                arguments = bind_arguments(entry.fn, arguments, entry.input_schema)
            except ValidationError as e:
                msg = f"Invalid arguments for tool {entry.name}: {e}"
                raise ValidationError(msg) from None

            async with self.serialized(entry.serialize):
                # Ids given for model-typed parameters become instances here,
                # under the lock, where the graph is as earlier calls left it.
                arguments = self._mcp.dataclass.resolve_arguments(
                    entry.fn,
                    arguments,
                    read_only=bool((entry.annotations or {}).get("readOnlyHint")),
                )
                call.result = await self._call_flushing(entry.fn, arguments)
            return call.result

    @contextlib.asynccontextmanager
    async def serialized(self, serialize=True):
        """Hold the tool lock for the block when tools are serialized
        (``serialize_tools``) and ``serialize`` is True; a no-op otherwise.

        ``dataclass_call`` runs unlocked and takes the lock here itself, so
        the called method's own ``serialize`` option decides.
        """
        if serialize and self._mcp.serialize_tools:
            waiting = time.monotonic()
            async with self._tools_lock:
                self._mcp._call_log.queued(time.monotonic() - waiting)
                yield
        else:
            yield

    async def _call_flushing(self, fn, arguments):
        # trame pushes state changes and runs @state.change reactions on a
        # flush, which an HTTP request never triggers by itself: without one,
        # what a tool assigns would wait for some unrelated flush.
        with self._mcp.server.state:
            return await _call(fn, arguments)

    # -------------------------------------------------------------------------
    # Resources
    # -------------------------------------------------------------------------

    def _on_resources_list(self, _params):
        return {
            "resources": [
                {
                    "uri": entry.uri,
                    "name": entry.name,
                    **({"description": entry.description} if entry.description else {}),
                    **({"mimeType": entry.mime_type} if entry.mime_type else {}),
                }
                for entry in self._mcp.list_resources()
            ]
        }

    def _on_resource_templates_list(self, _params):
        return {"resourceTemplates": []}

    async def read_resource(self, entry):
        """Produce a resource's raw value (shared with in-process consumers)."""
        with self._mcp._call_log.call("resource", entry.uri) as call:
            self._mcp._check_enabled()
            call.result = await _call(entry.fn, {})
            return call.result

    async def _on_resources_read(self, params):
        uri = params.get("uri")
        entry = self._mcp.get_resource(uri)
        if entry is None:
            msg = f"Unknown resource: {uri}"
            with self._mcp._call_log.call("resource", str(uri)):
                raise _InvalidParams(msg)

        value = await self.read_resource(entry)
        if isinstance(value, bytes):
            content = {
                "uri": uri,
                "mimeType": entry.mime_type or "application/octet-stream",
                "blob": base64.b64encode(value).decode("ascii"),
            }
        else:
            if not isinstance(value, str):
                value = json.dumps(value)
            content = {
                "uri": uri,
                "mimeType": entry.mime_type or "text/plain",
                "text": value,
            }
        return {"contents": [content]}

    # -------------------------------------------------------------------------
    # Prompts
    # -------------------------------------------------------------------------

    def _on_prompts_list(self, _params):
        return {
            "prompts": [
                {
                    "name": entry.name,
                    **({"description": entry.description} if entry.description else {}),
                    "arguments": entry.arguments,
                }
                for entry in self._mcp.list_prompts()
            ]
        }

    async def _on_prompts_get(self, params):
        name = params.get("name")
        entry = self._mcp.get_prompt(name)
        if entry is None:
            msg = f"Unknown prompt: {name}"
            with self._mcp._call_log.call("prompt", str(name), params.get("arguments")):
                raise _InvalidParams(msg)

        arguments = params.get("arguments") or {}
        with self._mcp._call_log.call("prompt", name, arguments) as call:
            self._mcp._check_enabled()
            try:
                inspect.signature(entry.fn).bind(**arguments)
            except TypeError as e:
                msg = f"Invalid arguments for prompt {name}: {e}"
                raise _InvalidParams(msg) from None

            call.result = value = await _call(entry.fn, arguments)
        if isinstance(value, str):
            messages = [
                {"role": "user", "content": {"type": "text", "text": value}}
            ]
        else:
            messages = value
        return {
            **({"description": entry.description} if entry.description else {}),
            "messages": messages,
        }


class _InvalidParams(ValueError):
    pass


class _RequestCancelled(Exception):
    pass
