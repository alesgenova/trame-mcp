"""Mounts the MCP endpoint on the trame (wslink/aiohttp) web server."""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import logging
import secrets

from aiohttp import web
from yarl import URL

logger = logging.getLogger(__name__)

__all__ = ["MCPEndpoint", "is_loopback"]

WILDCARD_HOSTS = ("", "0.0.0.0", "::")
SERVER_ERROR = -32000
SESSION_HEADER = "Mcp-Session-Id"
# Seconds between checks that the client of a running request is still there
DISCONNECT_POLL = 0.5


def is_loopback(host) -> bool:
    """True for ``localhost`` names and loopback IP literals."""
    if not host:
        return False
    host = host.strip("[]").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _error_response(status, message, headers=None):
    return web.json_response(
        {"jsonrpc": "2.0", "id": None, "error": {"code": SERVER_ERROR, "message": message}},
        status=status,
        headers=headers,
    )


async def _cancel_on_disconnect(request, task):
    """Cancel ``task`` if the client of ``request`` disconnects before it
    ends; returns whether it did. aiohttp keeps a handler running after its
    client is gone (``handler_cancellation`` is off by default, and it is the
    server's setting, not this route's)."""
    while not task.done():
        await asyncio.sleep(DISCONNECT_POLL)
        transport = request.transport
        if transport is None or transport.is_closing():
            task.cancel()
            return True
    return False


class MCPEndpoint:
    """Serves MCP over streamable HTTP (stateless) at ``path`` on the trame app.

    Every request goes through, in order:

    1. ``Origin`` check — browsers always send it on cross-site requests, so
       a page from another site can never drive the tools (CSRF), whatever
       the content type or authentication.
    2. Bearer token, when one is set. Without a token the ``Host`` header
       must name a loopback address, which defeats DNS rebinding (a remote
       page whose domain was re-pointed at 127.0.0.1).
    3. ``mcp.enabled`` — 503 while agent access is off. The route is mounted
       regardless, because aiohttp freezes its routes at startup and access
       may be switched on later.
    4. ``Content-Type: application/json``, so browsers cannot send a
       request without a CORS preflight, which this endpoint never grants.

    The ``initialize`` response carries an ``Mcp-Session-Id``: the server
    keeps no state for it and never requires it, it only scopes request ids
    so that a client's ``notifications/cancelled`` reaches its own requests.
    A request whose client disconnects is cancelled too.
    """

    def __init__(self, mcp):
        self._mcp = mcp
        self.mounted = False
        self.ready = False
        self.announced = False
        self.bind_host = None
        self.token_generated = False

    # -------------------------------------------------------------------------
    # trame life cycle
    # -------------------------------------------------------------------------

    def on_server_bind(self, wslink_server):
        if not self._mcp.http:
            logger.info("trame-mcp: HTTP endpoint disabled (http=False)")
            return

        self.bind_host = self._bind_host(wslink_server)
        if self._mcp.token is None and not is_loopback(self.bind_host):
            # Listening beyond this machine: never serve an unauthenticated
            # automation API to the network.
            self._mcp.token = secrets.token_urlsafe(32)
            self.token_generated = True

        wslink_server.app.add_routes(
            [
                web.post(self._mcp.path, self._handle_post),
                web.get(self._mcp.path, self._handle_unsupported),
                web.delete(self._mcp.path, self._handle_unsupported),
            ]
        )
        self.mounted = True

    def on_server_ready(self, **_):
        self.ready = True
        if self._mcp.enabled:
            self.announce()
        elif self.mounted:
            logger.info(
                "trame-mcp: %s mounted, agent access off "
                "(set mcp.enabled = True or launch with --mcp)",
                self._mcp.url,
            )

    def announce(self):
        """Print connection instructions the first time access is live."""
        if self.announced or not (self.ready and self.mounted):
            return
        self.announced = True

        url = self._mcp.url
        command = f"claude mcp add --transport http {self._mcp.name} {url}"
        if self.token_generated:
            command += f' --header "Authorization: Bearer {self._mcp.token}"'
        elif self._mcp.token:
            command += ' --header "Authorization: Bearer <token>"'

        # Match trame's connection-info style: plain print.
        lines = [f"\nMCP server available at {url}", f"  {command}"]
        if self.token_generated:
            lines.append(
                "  (token generated because the server listens beyond localhost; "
                "set it with --mcp-token or TRAME_MCP_TOKEN)"
            )
        # flush so the banner shows up promptly when stdout is redirected.
        print("\n".join(lines) + "\n", flush=True)

    def _bind_host(self, wslink_server):
        host = getattr(wslink_server, "host", None)
        if host is None:
            options = getattr(self._mcp.server, "_server_options", None)
            host = getattr(options, "host", None)
        return host

    # -------------------------------------------------------------------------
    # Request checks
    # -------------------------------------------------------------------------

    def _origin_allowed(self, origin, request):
        if origin in self._mcp.allowed_origins:
            return True
        try:
            parsed = URL(origin)
        except ValueError:
            return False
        if not parsed.host:  # e.g. "null" from sandboxed frames / file://
            return False
        return is_loopback(parsed.host) or parsed.raw_authority == request.host

    def _check(self, request):
        origin = request.headers.get("Origin")
        if origin is not None and not self._origin_allowed(origin, request):
            logger.warning("trame-mcp: rejected request from origin %r", origin)
            return _error_response(403, "Origin not allowed")

        token = self._mcp.token
        if token:
            scheme, _, credentials = request.headers.get("Authorization", "").partition(" ")
            if scheme.lower() != "bearer" or not hmac.compare_digest(
                credentials.strip().encode(), token.encode()
            ):
                return _error_response(
                    401, "Unauthorized", headers={"WWW-Authenticate": "Bearer"}
                )
        elif not is_loopback(request.url.host):
            return _error_response(
                403,
                "Host not allowed: set a token (token=, --mcp-token or "
                "TRAME_MCP_TOKEN) to serve MCP under a non-loopback host name",
            )

        if not self._mcp.enabled:
            return _error_response(503, "MCP access is currently disabled by the application")

        if request.content_type != "application/json":
            return _error_response(415, "Content-Type must be application/json")

        return None

    # -------------------------------------------------------------------------
    # HTTP handlers
    # -------------------------------------------------------------------------

    async def _handle_post(self, request):
        rejection = self._check(request)
        if rejection is not None:
            return rejection

        try:
            message = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return web.json_response(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"},
                },
                status=400,
            )

        if isinstance(message, list):
            return web.json_response(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32600,
                        "message": "JSON-RPC batching is not supported",
                    },
                },
                status=400,
            )

        session = request.headers.get(SESSION_HEADER)
        is_request = isinstance(message, dict) and "method" in message and "id" in message
        if is_request:
            handling = asyncio.ensure_future(self._mcp.protocol.handle(message, session=session))
            watching = asyncio.ensure_future(_cancel_on_disconnect(request, handling))
            try:
                response = await handling
            except asyncio.CancelledError:
                if not (watching.done() and not watching.cancelled() and watching.result()):
                    raise
                logger.info(
                    "trame-mcp: client disconnected, cancelled %s request %r",
                    message.get("method"),
                    message.get("id"),
                )
                # Never delivered; for the access log, as nginx does
                return web.Response(status=499, reason="Client Closed Request")
            finally:
                watching.cancel()
        else:
            response = await self._mcp.protocol.handle(message, session=session)
        if response is None:
            # Notification or client response: acknowledge without a body
            return web.Response(status=202)

        headers = None
        if is_request and message["method"] == "initialize" and "result" in response:
            headers = {SESSION_HEADER: secrets.token_urlsafe(24)}
        return web.json_response(response, headers=headers)

    async def _handle_unsupported(self, _request):
        # Stateless server: no SSE stream to offer (GET), no session to end (DELETE)
        return web.Response(status=405, headers={"Allow": "POST"})
