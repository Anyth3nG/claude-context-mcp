"""
MCP server entrypoint. Three ways in, one server definition:

  - stdio (default)          — local Claude Code use, launched per-session as
                               a subprocess. MCP_TRANSPORT=stdio.
  - streamable-http          — long-running process, for running the HTTP
                               transport locally or on a box you own.
                               MCP_TRANSPORT=streamable-http.
  - handler()                — AWS Lambda entry point (the deployed path).
                               Configure the function handler as
                               `mcp_server.server.handler`.

The Lambda path is the one with non-obvious constraints; see handler() below.
"""
import hmac
import os
import sys
from pathlib import Path

# claude mcp add invokes this file directly (`python .../mcp_server/server.py`),
# not as `-m mcp_server.server` — so the repo root isn't on sys.path by default
# and the `mcp_server.*` / `shared.*` absolute imports below would fail. Put it
# there explicitly rather than depending on how this script gets launched.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

from mcp_server.map_routes import register_map_routes
from mcp_server.tools.search_context import DESCRIPTION as SEARCH_CONTEXT_DESCRIPTION, search_context
from mcp_server.tools.save_update import DESCRIPTION as SAVE_UPDATE_DESCRIPTION, save_update

# python-dotenv isn't in the Lambda bundle — there's no .env there, config
# comes from the function's environment variables. Import it optionally so the
# same module works in both places.
try:
    from dotenv import load_dotenv

    # load_dotenv() with no args searches upward from cwd — fine when launched
    # from the repo, but this server is meant to be reachable from any project
    # directory, where there's no .env to find. Anchor it to the repo explicitly.
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
except ModuleNotFoundError:
    pass

mcp = MCPServer("context-mcp")
mcp.add_tool(search_context, description=SEARCH_CONTEXT_DESCRIPTION)
mcp.add_tool(save_update, description=SAVE_UPDATE_DESCRIPTION)
register_map_routes(mcp)  # /map + /map/data — HTTP transport only


def _transport_security() -> TransportSecuritySettings:
    """
    DNS-rebinding protection for the HTTP transports.

    MCP_ALLOWED_HOST must be the EXACT hostname the server is reached on (for
    Lambda, the generated *.lambda-url.<region>.on.aws domain). "*" is not a
    wildcard here — the validator compares it literally, so a "*" entry matches
    nothing and every request comes back HTTP 421 "Invalid Host header". A
    missing Host header is rejected outright too.
    """
    allowed = os.environ.get("MCP_ALLOWED_HOST")
    if not allowed:
        raise RuntimeError(
            "MCP_ALLOWED_HOST is not set. It must be the exact hostname this "
            "server is reached on — the Lambda Function URL domain in "
            "production, or 127.0.0.1:8000 for local HTTP. Note '*' does not "
            "work as a wildcard."
        )
    return TransportSecuritySettings(allowed_hosts=[h.strip() for h in allowed.split(",")])


class BearerAuthMiddleware:
    """
    Bearer-token gate in front of the whole ASGI app.

    Deliberately wraps everything rather than sitting at the MCP layer, because
    the /map and /map/data custom routes bypass MCP-level auth by design — and
    /map/data serves the entire store. Anything reachable over the network has
    to be behind this.

    No-op when AUTH_TOKEN is unset, which keeps local stdio/dev usable; the
    deploy is responsible for always setting it.
    """

    def __init__(self, app, token: str | None):
        self.app = app
        self.token = token

    async def __call__(self, scope, receive, send):
        if self.token and scope["type"] == "http":
            headers = dict(scope.get("headers") or [])
            presented = headers.get(b"authorization", b"").decode()
            # compare_digest, not ==: a plain comparison short-circuits on the
            # first differing byte, which leaks the token prefix through
            # response timing.
            if not hmac.compare_digest(presented, f"Bearer {self.token}"):
                await send({
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [(b"content-type", b"text/plain")],
                })
                await send({"type": "http.response.body", "body": b"Unauthorized"})
                return
        await self.app(scope, receive, send)


def build_asgi_app():
    """Stateless streamable-HTTP app, auth-wrapped. Fresh instance per call."""
    app = mcp.streamable_http_app(
        stateless_http=True,
        json_response=True,
        transport_security=_transport_security(),
    )
    return BearerAuthMiddleware(app, os.environ.get("AUTH_TOKEN"))


def handler(event, context):
    """
    AWS Lambda entry point.

    The app is rebuilt on EVERY invocation, which looks wasteful and isn't.
    Two constraints collide:

      1. The ASGI lifespan must run — StreamableHTTPSessionManager creates its
         anyio task group during lifespan startup, so Mangum(lifespan="off")
         makes every request fail with "Task group is not initialized".
      2. Mangum runs the lifespan on every invocation (no event loop survives
         between them), but StreamableHTTPSessionManager.run() raises "can only
         be called once per instance" the second time it's started.

    So a module-level app succeeds exactly once and then fails on every warm
    reuse — which a single-request smoke test will not catch. Building a fresh
    app per request gives each session manager its own single run(). Measured
    cost is ~0.1ms; the expensive state (Chroma client, HTTP connection pool,
    tool registry) stays at module scope where it survives warm invocations.
    """
    from mangum import Mangum  # not installed outside the Lambda bundle

    return Mangum(build_asgi_app(), lifespan="auto")(event, context)


if __name__ == "__main__":
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport == "streamable-http":
        mcp.run(
            transport="streamable-http",
            host=os.environ.get("MCP_HOST", "127.0.0.1"),
            port=int(os.environ.get("MCP_PORT", "8000")),
        )
    else:
        mcp.run(transport="stdio")
