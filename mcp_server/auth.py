"""
OAuth token verification against an Amazon Cognito user pool.

Why Cognito rather than a bearer token: claude.ai's custom-connector UI does
not expose a request-header field on all accounts (that feature is in beta), so
it authenticates by running an OAuth 2.1 flow against the server. This module
is the resource-server half of that — the MCP SDK serves the protected-resource
metadata and issues the 401 challenge; all that's left is deciding whether a
presented access token is real.

Cognito is the authorization server. It is not written here on purpose: an
authorization server is security-critical code that should never be hand-rolled
for a project like this, and Cognito's free tier covers far more than one user.

TOKEN SHAPE NOTE — Cognito access tokens are not ID tokens:
  - There is no `aud` claim. Audience binding is done by checking `client_id`
    against the app clients we expect, so signature verification must have
    audience checking switched off or every token is rejected.
  - `token_use` distinguishes access from id tokens. An ID token has a valid
    signature from the same pool, so without this check an ID token would be
    accepted as an access token.
"""
from __future__ import annotations

import hmac
import os
import time
from typing import Optional

from mcp.server.auth.provider import AccessToken, TokenVerifier


class CognitoTokenVerifier(TokenVerifier):
    """
    Verifies Cognito access tokens. Returning None (rather than raising) is the
    SDK's contract for "reject this" — it turns into a 401 with the
    WWW-Authenticate challenge that points clients at the auth server.
    """

    def __init__(
        self,
        region: str,
        user_pool_id: str,
        allowed_client_ids: set[str],
        required_scopes: Optional[list[str]] = None,
        jwks_cache_seconds: int = 3600,
    ):
        self.region = region
        self.user_pool_id = user_pool_id
        self.allowed_client_ids = allowed_client_ids
        self.required_scopes = required_scopes or []
        self.issuer = f"https://cognito-idp.{region}.amazonaws.com/{user_pool_id}"
        self._jwks_client = None
        self._jwks_cache_seconds = jwks_cache_seconds

    def _jwks(self):
        # Built lazily and kept at instance scope, which on Lambda means one
        # JWKS fetch per container rather than one per request. PyJWKClient
        # does its own caching on top; the lifespan bounds how long a rotated
        # signing key stays stale.
        if self._jwks_client is None:
            from jwt import PyJWKClient

            self._jwks_client = PyJWKClient(
                f"{self.issuer}/.well-known/jwks.json",
                cache_keys=True,
                lifespan=self._jwks_cache_seconds,
            )
        return self._jwks_client

    def _reject(self, reason: str) -> None:
        # Logged, never returned to the caller: the client gets an opaque 401,
        # because telling an unauthenticated caller *why* their token failed
        # helps an attacker more than it helps a legitimate user.
        print(f"[auth] rejected token: {reason}")
        return None

    async def verify_token(self, token: str) -> AccessToken | None:
        import jwt

        try:
            signing_key = self._jwks().get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                issuer=self.issuer,
                # Cognito access tokens carry no `aud` claim — leaving this on
                # would reject every valid token. client_id is checked below
                # instead, which is the equivalent binding for this token type.
                options={"verify_aud": False, "require": ["exp", "iss", "sub"]},
            )
        except Exception as exc:  # signature, expiry, malformed, issuer mismatch
            return self._reject(f"{type(exc).__name__}: {exc}")

        if claims.get("token_use") != "access":
            # An ID token from the same pool would pass signature and issuer
            # checks, so this is load-bearing, not defensive noise.
            return self._reject(f"token_use={claims.get('token_use')!r}, expected 'access'")

        client_id = claims.get("client_id")
        if client_id not in self.allowed_client_ids:
            return self._reject(f"client_id {client_id!r} not in allowed set")

        scopes = claims.get("scope", "").split()
        missing = [s for s in self.required_scopes if s not in scopes]
        if missing:
            return self._reject(f"missing required scopes: {missing}")

        return AccessToken(
            token=token,
            client_id=client_id,
            scopes=scopes,
            expires_at=int(claims["exp"]),
            subject=claims.get("sub"),
            claims=claims,
        )


class StaticOrOAuthVerifier(TokenVerifier):
    """
    Accepts EITHER a long-lived static bearer token OR a real OAuth token.

    This exists for one specific reason: Cognito has no dynamic client
    registration and demands exact redirect-URI matches including the port,
    while Claude Code registers dynamically and uses loopback callbacks on
    ephemeral ports. claude.ai can use OAuth (it takes a manually-issued client
    id/secret); Claude Code realistically cannot, without pinning a callback
    port and having somewhere to put a pre-registered client id.

    The trade being made deliberately: the static token is a shared password
    that does not expire and cannot be revoked per-device, so overall security
    is only as strong as that token. It is kept because the alternative is
    Claude Code losing access to the shared store entirely. If Anthropic ships
    request-header auth or Cognito-compatible registration, delete this class
    and the static path with it.
    """

    def __init__(self, static_token: str, delegate: TokenVerifier, scopes: Optional[list[str]] = None):
        self.static_token = static_token
        self.delegate = delegate
        self.scopes = scopes or []

    async def verify_token(self, token: str) -> AccessToken | None:
        # compare_digest, not ==: a plain comparison short-circuits on the first
        # differing byte, leaking the token prefix through response timing.
        if self.static_token and hmac.compare_digest(token, self.static_token):
            return AccessToken(
                token=token,
                # A distinct client_id so log lines make it obvious which path
                # authenticated a request — static tokens and OAuth sessions
                # should never be indistinguishable after the fact.
                client_id="static-bearer",
                scopes=list(self.scopes),
                # Nominal, not enforcement: every request is verified from
                # scratch (the server is stateless), and this token genuinely
                # has no expiry. Rotating it is a manual act.
                expires_at=int(time.time()) + 3600,
                subject="static-bearer",
            )
        return await self.delegate.verify_token(token)


def verifier_from_env() -> Optional[TokenVerifier]:
    """
    Build a verifier from environment, or None if Cognito isn't configured —
    which is how local stdio development keeps working with no auth at all.

    Deliberately all-or-nothing: a half-set configuration raises instead of
    silently leaving the endpoint unauthenticated, which is the failure mode
    that actually matters here.
    """
    region = os.environ.get("COGNITO_REGION")
    pool_id = os.environ.get("COGNITO_USER_POOL_ID")
    client_ids = os.environ.get("COGNITO_ALLOWED_CLIENT_IDS", "")

    provided = [n for n, v in (("COGNITO_REGION", region),
                               ("COGNITO_USER_POOL_ID", pool_id),
                               ("COGNITO_ALLOWED_CLIENT_IDS", client_ids)) if v]
    if not provided:
        return None
    if len(provided) < 3:
        missing = {"COGNITO_REGION", "COGNITO_USER_POOL_ID", "COGNITO_ALLOWED_CLIENT_IDS"} - set(provided)
        raise RuntimeError(
            f"Partial Cognito config: {', '.join(sorted(provided))} set but "
            f"{', '.join(sorted(missing))} missing. Set all three, or none for "
            "unauthenticated local development."
        )

    scopes = [s for s in os.environ.get("MCP_REQUIRED_SCOPES", "").split() if s]
    cognito = CognitoTokenVerifier(
        region=region,
        user_pool_id=pool_id,
        allowed_client_ids={c.strip() for c in client_ids.split(",") if c.strip()},
        required_scopes=scopes,
    )

    # The static token is accepted on the MCP endpoint only while
    # MCP_ALLOW_STATIC_TOKEN is on. Claude Code 2.1.220+ supports --client-id
    # and --callback-port, so it CAN do real OAuth against its own Cognito app
    # client; once that's confirmed working, set MCP_ALLOW_STATIC_TOKEN=0 and
    # the shared password stops granting access to the store at all (it stays
    # in use only for the /map routes, which have no OAuth flow).
    allow_static = os.environ.get("MCP_ALLOW_STATIC_TOKEN", "1").lower() not in ("0", "false", "no")
    static = os.environ.get("AUTH_TOKEN")
    if static and allow_static:
        wrapped = StaticOrOAuthVerifier(static, cognito, scopes)
        # issuer/required_scopes are read off the verifier by server.py when it
        # builds AuthSettings, so the wrapper has to keep exposing them.
        wrapped.issuer = cognito.issuer
        wrapped.required_scopes = cognito.required_scopes
        return wrapped
    return cognito
