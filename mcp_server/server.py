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
# map_routes is intentionally not imported — see the note where the tools are
# registered below.
from mcp_server.tools.search_context import DESCRIPTION as SEARCH_CONTEXT_DESCRIPTION, search_context
from mcp_server.tools.get_index import DESCRIPTION as GET_INDEX_DESCRIPTION, get_index
from mcp_server.tools.get_brief import DESCRIPTION as GET_BRIEF_DESCRIPTION, get_brief
from mcp_server.tools.get_value import DESCRIPTION as GET_VALUE_DESCRIPTION, get_value
from mcp_server.tools.get_history import DESCRIPTION as GET_HISTORY_DESCRIPTION, get_history
from mcp_server.tools.add_update import DESCRIPTION as ADD_UPDATE_DESCRIPTION, add_update
from mcp_server.tools.change_update import DESCRIPTION as CHANGE_UPDATE_DESCRIPTION, change_update
from mcp_server.tools.patch_context import DESCRIPTION as PATCH_CONTEXT_DESCRIPTION, patch_context
from mcp_server.tools.retire_chunk import DESCRIPTION as RETIRE_CHUNK_DESCRIPTION, retire_chunk
from mcp_server.tools.archive_slot import DESCRIPTION as ARCHIVE_SLOT_DESCRIPTION, archive_slot

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

mcp = MCPServer(
    "context-mcp",
    token_verifier=_token_verifier,
    auth=_auth_settings(_token_verifier),
)
# Read side, cheapest first — which is also the order they should be reached
# for. get_index is a map with no contents and is what a session should open
# with; get_brief/get_value are deterministic lookups returning whole documents;
# search_context ranks and truncates, so it's for history only.
mcp.add_tool(get_index, description=GET_INDEX_DESCRIPTION)
mcp.add_tool(get_brief, description=GET_BRIEF_DESCRIPTION)
mcp.add_tool(get_value, description=GET_VALUE_DESCRIPTION)
# Between get_value and search_context by design: it answers "how did this
# known slot change" deterministically, which is the question people reach for
# search_context to answer and get ranking and truncation instead. Search stays
# the tool for when you do NOT know where to look.
mcp.add_tool(get_history, description=GET_HISTORY_DESCRIPTION)
mcp.add_tool(search_context, description=SEARCH_CONTEXT_DESCRIPTION)
# Write side, in the order they should be reached for: add_update appends
# history, patch_context edits current state in place, change_update rewrites a
# slot wholesale. patch_context is registered before change_update deliberately
# — it is the default write, and change_update is now the exception, kept for
# creating a slot or replacing one outright.
mcp.add_tool(add_update, description=ADD_UPDATE_DESCRIPTION)
mcp.add_tool(patch_context, description=PATCH_CONTEXT_DESCRIPTION)
mcp.add_tool(change_update, description=CHANGE_UPDATE_DESCRIPTION)
# Registered last because it is the rarest write and the only destructive-ish
# one. The other three change what the store says; this one says an existing
# entry was wrong. Chunks are append-only, so without it a disproved fact keeps
# ranking against the entry that corrected it, with nothing marking which is
# which — the failure the 2026-08-05 retrieval check surfaced.
mcp.add_tool(retire_chunk, description=RETIRE_CHUNK_DESCRIPTION)
# The other half of the same idea: retire_chunk says a fact was WRONG,
# archive_slot says a slot's work is FINISHED. Both take something out of the
# default read; only the first is a correction. Without this, `goal` had no way
# to distinguish what needs doing from what was done, and completed phases kept
# loading with every brief.
mcp.add_tool(archive_slot, description=ARCHIVE_SLOT_DESCRIPTION)

# /map and /map/data are NOT registered. Custom routes bypass MCP-level auth by
# design, and /map/data serves the entire store — so they needed a guard of
# their own, which was the static bearer token. That token is gone (see
# auth.py), and these routes have no OAuth flow to fall back on: a browser
# navigation cannot send an Authorization header, which is why the map was
# already unreachable from a browser. Registering them now would publish the
# whole store unauthenticated.
#
# map_routes.py and static/map.html are kept deliberately — they are the
# starting point for the rebuild, where auth is designed in rather than bolted
# on. Re-enable only together with a real browser-compatible auth path.
# register_map_routes(mcp)


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

    No auth wrapper here any more: /mcp is guarded by the MCP layer itself
    against Cognito, and the only routes that ever needed a separate gate were
    /map*, which are no longer registered. The one wrapper that remains logs
    rejections; it never decides them.

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
    return UnauthenticatedRejectionLogger(app)


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
