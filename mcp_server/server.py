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

from shared.config import load_secrets

from mcp_server.auth import REASON_NO_TOKEN, log_auth_rejection, verifier_from_env
from mcp_server.map_routes import register_map_routes
from mcp_server.tools.search_context import DESCRIPTION as SEARCH_CONTEXT_DESCRIPTION, search_context
from mcp_server.tools.get_index import DESCRIPTION as GET_INDEX_DESCRIPTION, get_index
from mcp_server.tools.get_context import DESCRIPTION as GET_CONTEXT_DESCRIPTION, get_context
from mcp_server.tools.get_history import DESCRIPTION as GET_HISTORY_DESCRIPTION, get_history
from mcp_server.tools.add_update import DESCRIPTION as ADD_UPDATE_DESCRIPTION, add_update
from mcp_server.tools.patch_context import DESCRIPTION as PATCH_CONTEXT_DESCRIPTION, patch_context
from mcp_server.tools.archive import DESCRIPTION as ARCHIVE_DESCRIPTION, archive

# python-dotenv isn't in the Lambda bundle — there's no .env there. Import it
# optionally so the same module works in both places.
try:
    from dotenv import load_dotenv

    # load_dotenv() with no args searches upward from cwd — fine when launched
    # from the repo, but this server is meant to be reachable from any project
    # directory, where there's no .env to find. Anchor it to the repo explicitly.
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
except ModuleNotFoundError:
    pass

# Deployed, credentials come from Secrets Manager rather than the function's
# own configuration — see shared/config.py for why. No-op locally. This runs at
# import so it lands in Lambda's init phase, where it's paid once per cold
# start rather than once per request.
load_secrets()

def _auth_settings(verifier) -> "AuthSettings | None":
    """
    OAuth configuration for the MCP endpoint, or None when Cognito isn't set up
    (local stdio development).

    issuer_url points at Cognito — the authorization server clients get
    redirected to. resource_server_url identifies THIS server, and is what the
    SDK advertises in protected-resource metadata so a client that arrives with
    no token knows where to go and what it's authenticating against.
    """
    if verifier is None:
        return None
    from mcp.server.auth.settings import AuthSettings

    host = os.environ.get("MCP_ALLOWED_HOST", "").split(",")[0].strip()
    if not host:
        raise RuntimeError("MCP_ALLOWED_HOST must be set when Cognito auth is enabled")
    return AuthSettings(
        issuer_url=verifier.issuer,
        resource_server_url=f"https://{host}/mcp",
        required_scopes=verifier.required_scopes or None,
    )


_token_verifier = verifier_from_env()
OAUTH_ENABLED = _token_verifier is not None

# Delivered to the client with the server's capabilities, BEFORE any tool call.
# It is the only thing here that reaches a session without being asked for, which
# makes it the whole answer to "nothing makes a new session look".
#
# Deliberately STATIC — no project list, no counts. Those would have to be built
# at import, and SnapStart snapshots init at publish time, so a project added
# next week would not appear until a redeploy and an alias move (the same trap
# documented in config/rotation). The live numbers belong in get_index, which is
# one cheap call away; this text only has to make that call happen.
#
# Every session pays for this, so it stays short.
INSTRUCTIONS = """Durable memory for this user, shared across every machine and both clients (Claude Code and claude.ai). Context saved in another session is available in this one — do NOT assume it is empty, and do not re-derive what may already be recorded.

Open with get_index(detail="projects"): a table of contents for tens of tokens. Then go only as deep as the task needs — get_context(project), narrowed by category and then key, get_history for how a slot changed, search_context when you don't know where to look.

Write sparingly: patch_context revises a fact that already has a value, add_update appends a new one. Decisions worth having later, not conversation."""

mcp = MCPServer(
    "context-mcp",
    instructions=INSTRUCTIONS,
    token_verifier=_token_verifier,
    auth=_auth_settings(_token_verifier),
)
# Read side, cheapest first — which is also the order they should be reached
# for. get_index is a map with no contents and is what a session should open
# with; get_context is the deterministic lookup returning whole documents, at
# whatever depth the address it is given implies; search_context ranks and
# truncates, so it's for history only.
mcp.add_tool(get_index, description=GET_INDEX_DESCRIPTION)
mcp.add_tool(get_context, description=GET_CONTEXT_DESCRIPTION)
# Between get_context and search_context by design: it answers "how did this
# known slot change" deterministically, which is the question people reach for
# search_context to answer and get ranking and truncation instead. Search stays
# the tool for when you do NOT know where to look.
mcp.add_tool(get_history, description=GET_HISTORY_DESCRIPTION)
mcp.add_tool(search_context, description=SEARCH_CONTEXT_DESCRIPTION)
# Write side, in the order they should be reached for. add_update appends to
# history; patch_context writes current state, either as a diff or — since
# change_update was folded into it — as a wholesale replacement, chosen by
# whether old_str is present.
mcp.add_tool(add_update, description=ADD_UPDATE_DESCRIPTION)
mcp.add_tool(patch_context, description=PATCH_CONTEXT_DESCRIPTION)
# Registered last because it is the rarest write and the only one that takes
# something OUT of the default read. It carries both senses of that, dispatched
# by what it is pointed at: an id marks a chunk WRONG, a category retires a
# FINISHED slot. Merged from retire_chunk and archive_slot, which is safe only
# because the distinction is preserved in the tool's own responses — superseded
# stays searchable as history, retired is held back as known-false.
mcp.add_tool(archive, description=ARCHIVE_DESCRIPTION)

# /map is registered again as of 2026-08-10, with the browser-compatible auth
# path it always needed: a Cognito hosted-UI login terminating in a session
# cookie, verified by the same CognitoTokenVerifier that guards /mcp. See
# map_routes.py for why the guard lives inside the handlers — custom routes
# bypass MCP-level auth by design, so nothing above them protects the store.
#
# /map/data is deliberately NOT a route. The page inlines its data, so there is
# no second endpoint serving the store and no fetch to authorize separately.
register_map_routes(mcp)


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


class RejectStreamGet:
    """
    Answers GET /mcp with 405 instead of letting it hang to the Lambda timeout.

    Streamable HTTP has an OPTIONAL server-to-client SSE stream, opened with a
    GET to the same endpoint. Clients try it on connect. This server runs
    stateless behind API Gateway, where a long-lived stream cannot work — so
    nothing ever answers, and the request sits there until Lambda kills it. One
    per client connection: cheap, but it burns the full timeout every time and
    makes real hangs impossible to spot in the logs.

    405 is the spec's own answer for a server that does not offer the stream, so
    clients treat it as "fine, POST only" and move on immediately.

    Runs BEFORE auth on purpose. The method is wrong regardless of who is
    asking, and a 405 discloses nothing — it says only that this endpoint speaks
    POST, which its own protocol documentation already says. Answering 401 first
    would mean an authenticated client still had to discover the hang.

    SCOPED TO ONE PATH, and that is not a detail. This originally matched on
    method alone, which was invisible while /mcp was the only route anyone
    reached by GET — but it meant every other GET in the app got the same 405,
    including the OAuth protected-resource metadata the SDK serves and, once it
    was registered, the whole of /map. A GET elsewhere is not a client trying to
    open a stream, and answering it here says nothing true about it.
    """

    def __init__(self, app, path: str = "/mcp"):
        self.app = app
        self.path = path

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] == "http"
            and scope.get("method") == "GET"
            and scope.get("path") == self.path
        ):
            await send(
                {
                    "type": "http.response.start",
                    "status": 405,
                    "headers": [
                        (b"content-type", b"application/json"),
                        # Spec-correct on a 405, and tells a client what to do
                        # next without it having to guess.
                        (b"allow", b"POST"),
                    ],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b'{"error":"method_not_allowed",'
                    b'"detail":"This endpoint is POST-only. The optional GET SSE '
                    b'stream is not supported: the server runs stateless behind '
                    b'API Gateway, where a long-lived stream cannot be held."}',
                }
            )
            return
        await self.app(scope, receive, send)


class UnauthenticatedRejectionLogger:
    """
    Logs the one rejection class the token verifier never sees.

    CognitoTokenVerifier.verify_token only runs once the SDK has extracted a
    bearer token, so a request carrying no Authorization header — or one that
    is not a Bearer credential — is refused upstream of it and would otherwise
    be counted nowhere. That is exactly the shape of an unauthenticated probe
    against a public endpoint, so it is the case the alarm most needs.

    Scoped narrowly on purpose: it logs only when the response is a 401 AND no
    bearer credential was present. Anything with a bearer token has already
    been logged, with a more specific reason, by the verifier.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        auth = headers.get(b"authorization", b"").decode(errors="replace")
        had_bearer = auth.lower().startswith("bearer ") and len(auth) > len("bearer ")

        async def send_wrapper(message):
            if (
                not had_bearer
                and message["type"] == "http.response.start"
                and message["status"] == 401
            ):
                client = scope.get("client") or ()
                peer = client[0] if client else headers.get(b"x-forwarded-for", b"").decode(
                    errors="replace"
                ).split(",")[0].strip()
                log_auth_rejection(
                    REASON_NO_TOKEN,
                    # Distinguishes "sent nothing" from "sent something that
                    # was not a Bearer credential" without recording what.
                    "absent" if not auth else "not_bearer",
                    path=scope.get("path", ""),
                    peer=peer,
                )
            await send(message)

        await self.app(scope, receive, send_wrapper)


def build_asgi_app():
    """
    Stateless streamable-HTTP app. Fresh instance per call.

    No auth wrapper here: /mcp is guarded by the MCP layer itself against
    Cognito, and /map carries its own guard inside its handlers because custom
    routes bypass that layer by design (see map_routes.py). The one wrapper
    that remains logs rejections; it never decides them.

    That leaves one gap worth being loud about — if Cognito is not configured,
    the MCP layer has nothing to verify against and this app serves the store
    to anyone who can reach it. That is fine for loopback development and is
    never fine anywhere else, so say so rather than failing silently.
    """
    if not OAUTH_ENABLED:
        print(
            "WARNING: serving MCP over HTTP with NO authentication — Cognito is "
            "not configured (COGNITO_REGION / COGNITO_USER_POOL_ID / "
            "COGNITO_ALLOWED_CLIENT_IDS). Acceptable on loopback only.",
            file=sys.stderr,
        )
    app = mcp.streamable_http_app(
        stateless_http=True,
        json_response=True,
        transport_security=_transport_security(),
    )
    # Order matters. RejectStreamGet is outermost so a GET is answered before
    # anything else runs: it never reaches auth, so it never produces a 401 for
    # the rejection logger to count. That is correct — a wrong method is not a
    # failed authentication, and counting it as one would pollute the alarm that
    # exists to catch probes.
    return RejectStreamGet(UnauthenticatedRejectionLogger(app))


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
