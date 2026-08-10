"""
The map — the context store as a page, served by the same process.

  GET /map           the atlas, with the store inlined fresh on every request
  GET /map/callback  the Cognito redirect target that establishes the session

WHY THESE ROUTES CARRY THEIR OWN AUTH. The MCP SDK's custom_route explicitly
does not require authorization: enforcement for /mcp is RequireAuthMiddleware
wrapped around that one endpoint, not applied app-wide, and the app-wide
AuthenticationMiddleware is a no-op for a request with no Authorization header —
which is every browser navigation. Nothing above these handlers protects them.
The guard below is the only thing standing between a page load and publishing
the entire store, which is why an unconfigured deployment fails closed here
rather than rendering.

WHY A COOKIE. A browser navigation cannot send an Authorization header, so the
bearer flow that guards /mcp cannot reach a page at all. The session cookie
carries a Cognito access token and is verified by the SAME CognitoTokenVerifier
that guards /mcp — same pool, same signature and token_use checks, same
rejection logging. No second notion of identity exists.

WHY A SEPARATE APP CLIENT. The verifier here allows only MAP_CLIENT_ID, and
that id is deliberately NOT in COGNITO_ALLOWED_CLIENT_IDS. So a stolen map
cookie is rejected by /mcp, and an MCP access token is rejected here. Adding
the map client to the shared allowlist would have made the cookie a full
read/write credential for the whole tool surface.

SameSite=Lax IS LOAD-BEARING, not a default. The state cookie is set before
redirecting to Cognito and read on the way back, and that return trip is a
top-level navigation from Cognito's domain — cross-site. Under Strict the
browser withholds the cookie on exactly that request and every login fails the
state check, which looks like a bug in the check rather than in the policy.
"""
from __future__ import annotations

import os
import secrets
import time
import urllib.parse

import httpx
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, Response

# Imported at module scope rather than inside the registration function so a
# test can substitute them. The guard below is the only thing protecting the
# store, and a test that cannot reach the render path cannot prove the guard
# lets the right request through.
from mcp_server.auth import REASON_INVALID_TOKEN, REASON_NO_TOKEN, log_auth_rejection
from mcp_server.context import get_store
from shared.atlas import build_atlas_data, render_page

SESSION_COOKIE = "cmcp_session"
STATE_COOKIE = "cmcp_state"

# Both routes live under /map, and cookie path matching is prefix-with-boundary:
# "/map" matches /map and /map/callback but not /mapsomething. This keeps the
# session cookie off /mcp entirely.
COOKIE_PATH = "/map"

STATE_TTL = 600           # a login that takes over ten minutes can start again
SESSION_FALLBACK_TTL = 3600


def _origin() -> str:
    """
    The public origin, derived from MCP_ALLOWED_HOST rather than configured
    twice. That variable is already required to be the EXACT hostname this
    server is reached on, which is precisely what the redirect URI must match —
    Cognito compares redirect URIs literally, so deriving it from the one value
    that is already exact removes a whole class of mismatch.
    """
    host = os.environ.get("MCP_ALLOWED_HOST", "").split(",")[0].strip()
    if not host:
        raise RuntimeError("MCP_ALLOWED_HOST is not set; the map cannot build its redirect URI.")
    scheme = "http" if host.startswith(("127.0.0.1", "localhost", "[::1]")) else "https"
    return f"{scheme}://{host}"


def _redirect_uri() -> str:
    return f"{_origin()}/map/callback"


def _verifier():
    """
    A verifier bound to the map's own app client, or None if the map is not
    configured. Returning None is what makes the handlers fail closed.
    """
    from mcp_server.auth import CognitoTokenVerifier

    region = os.environ.get("COGNITO_REGION")
    pool_id = os.environ.get("COGNITO_USER_POOL_ID")
    client_id = os.environ.get("MAP_CLIENT_ID")
    if not (region and pool_id and client_id):
        return None
    scopes = [s for s in os.environ.get("MCP_REQUIRED_SCOPES", "").split() if s]
    return CognitoTokenVerifier(
        region=region,
        user_pool_id=pool_id,
        allowed_client_ids={client_id},
        required_scopes=scopes,
    )


def _set_cookie(resp: Response, name: str, value: str, max_age: int) -> None:
    resp.set_cookie(
        name, value,
        max_age=max_age,
        path=COOKIE_PATH,
        httponly=True,          # the page never needs to read it; XSS cannot lift it
        secure=True,
        samesite="lax",         # see the module docstring — Strict breaks the callback
    )


def register_map_routes(mcp) -> None:
    def _unconfigured(path: str) -> Response:
        # Fail closed. A deployment missing MAP_CLIENT_ID has no way to
        # authenticate anyone, and rendering anyway would serve the whole store
        # to whoever asked.
        log_auth_rejection("map_unconfigured", "MAP_CLIENT_ID or Cognito config absent", path=path)
        return Response("The map is not configured on this deployment.", status_code=503)

    @mcp.custom_route("/map", methods=["GET"])
    async def map_page(request: Request):
        verifier = _verifier()
        if verifier is None:
            return _unconfigured("/map")

        token = request.cookies.get(SESSION_COOKIE)
        if token:
            access = await verifier.verify_token(token)
            if access:
                data = build_atlas_data(get_store())
                resp = HTMLResponse(render_page(data))
                # The body is the entire store. Nothing may hold a copy.
                resp.headers["Cache-Control"] = "no-store, private"
                resp.headers["Referrer-Policy"] = "no-referrer"
                return resp
            # Expired or otherwise unacceptable: fall through and re-authenticate
            # rather than showing an error. Cognito usually makes this invisible.
            log_auth_rejection(REASON_INVALID_TOKEN, "map session cookie", path="/map")
        else:
            log_auth_rejection(REASON_NO_TOKEN, "map page", path="/map")

        state = secrets.token_urlsafe(32)
        params = urllib.parse.urlencode({
            "response_type": "code",
            "client_id": os.environ["MAP_CLIENT_ID"],
            "redirect_uri": _redirect_uri(),
            "scope": " ".join(["openid"] + [s for s in os.environ.get("MCP_REQUIRED_SCOPES", "").split() if s]),
            "state": state,
        })
        resp = RedirectResponse(f"{os.environ['MAP_COGNITO_DOMAIN']}/oauth2/authorize?{params}", status_code=302)
        _set_cookie(resp, STATE_COOKIE, state, STATE_TTL)
        return resp

    @mcp.custom_route("/map/callback", methods=["GET"])
    async def map_callback(request: Request):
        verifier = _verifier()
        if verifier is None:
            return _unconfigured("/map/callback")

        expected = request.cookies.get(STATE_COOKIE)
        presented = request.query_params.get("state")
        # Compared in constant time and required to be present on BOTH sides: a
        # missing cookie must not compare equal to a missing parameter, or the
        # CSRF check passes for a request that never started here.
        if not expected or not presented or not secrets.compare_digest(expected, presented):
            log_auth_rejection("map_state_mismatch", "callback state", path="/map/callback")
            return Response("Login could not be verified. Start again from /map.", status_code=400)

        code = request.query_params.get("code")
        if not code:
            log_auth_rejection("map_no_code", str(request.query_params.get("error", "")), path="/map/callback")
            return Response("Login did not complete.", status_code=400)

        async with httpx.AsyncClient(timeout=10) as client:
            token_resp = await client.post(
                f"{os.environ['MAP_COGNITO_DOMAIN']}/oauth2/token",
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": _redirect_uri(),
                    "client_id": os.environ["MAP_CLIENT_ID"],
                },
                # The secret goes in the Basic header, never the body or a log.
                auth=(os.environ["MAP_CLIENT_ID"], os.environ["MAP_CLIENT_SECRET"]),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        if token_resp.status_code != 200:
            # Status only. The body of a failed token response can echo request
            # parameters, and this line is readable by anything with CloudWatch.
            log_auth_rejection("map_token_exchange_failed", f"http_{token_resp.status_code}", path="/map/callback")
            return Response("Login could not be completed.", status_code=400)

        access_token = token_resp.json().get("access_token", "")
        access = await verifier.verify_token(access_token)
        if not access:
            # Cognito issued it and we still refuse it — wrong client, wrong
            # scope, wrong token_use. Verifying rather than trusting the
            # exchange is what keeps this the same standard as /mcp.
            log_auth_rejection(REASON_INVALID_TOKEN, "exchanged map token", path="/map/callback")
            return Response("Login could not be completed.", status_code=400)

        resp = RedirectResponse("/map", status_code=302)
        remaining = (access.expires_at - int(time.time())) if access.expires_at else SESSION_FALLBACK_TTL
        _set_cookie(resp, SESSION_COOKIE, access_token, max(remaining, 0))
        resp.delete_cookie(STATE_COOKIE, path=COOKIE_PATH)
        return resp
