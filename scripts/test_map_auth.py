#!/usr/bin/env python3
"""
The guard on /map, asserted.

Registering /map removed the 404 that used to be the only thing keeping the
store off the public internet. What replaces it is a check inside the handler —
and nothing above that handler would catch it if the check regressed, because
custom routes bypass MCP-level auth by design. So this file exists to fail
loudly if an unauthenticated request ever renders the page.

    .venv/bin/python scripts/test_map_auth.py

No pytest: this repo has no test suite and this is not the change that should
introduce a dependency. Exits non-zero on the first failure.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Set BEFORE importing the server: it reads config at import time, and a real
# Chroma connection or a real secret fetch would make this a network test.
os.environ["CONTEXT_MCP_SECRET_ID"] = ""
os.environ["CONTEXT_MCP_FORCE_LOCAL"] = "1"
os.environ["ALLOW_LOCAL_FALLBACK"] = "1"
os.environ["MCP_ALLOWED_HOST"] = "testserver"
os.environ.setdefault("COGNITO_REGION", "eu-west-1")
os.environ.setdefault("COGNITO_USER_POOL_ID", "eu-west-1_test")
os.environ.setdefault("COGNITO_ALLOWED_CLIENT_IDS", "mcp-client")
os.environ["MAP_CLIENT_ID"] = "map-client"
os.environ["MAP_CLIENT_SECRET"] = "not-a-real-secret"
os.environ["MAP_COGNITO_DOMAIN"] = "https://example.auth.eu-west-1.amazoncognito.com"

from starlette.testclient import TestClient          # noqa: E402
from mcp_server import map_routes                    # noqa: E402
from mcp_server.server import build_asgi_app         # noqa: E402

# A marker that appears in the rendered atlas and nowhere else, so "did this
# leak the store" is a fact rather than a guess about status codes.
STORE_MARKER = "constellations"

failures = []


def check(name, condition, detail=""):
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not condition else ""))
    if not condition:
        failures.append(name)


class _Stub:
    """Stands in for the Cognito verifier so nothing here touches the network."""

    def __init__(self, result):
        self.result = result

    async def verify_token(self, token):
        return self.result


def with_verifier(result):
    map_routes._verifier = lambda: result


_real_verifier = map_routes._verifier
client = TestClient(build_asgi_app())

print("\nunauthenticated access")
with_verifier(_Stub(None))                     # a verifier that accepts nothing
r = client.get("/map", follow_redirects=False)
check("no cookie does not render the store", STORE_MARKER not in r.text, f"status {r.status_code}")
check("no cookie redirects to login", r.status_code == 302, f"status {r.status_code}")
check("redirect goes to Cognito", "amazoncognito.com" in r.headers.get("location", ""))
check("state cookie is set", "cmcp_state" in r.headers.get("set-cookie", ""))

cookie_header = r.headers.get("set-cookie", "")
print("\ncookie policy")
check("state cookie is HttpOnly", "HttpOnly" in cookie_header)
check("state cookie is Secure", "Secure" in cookie_header)
check("state cookie is SameSite=Lax, not Strict", "samesite=lax" in cookie_header.lower())
check("cookie is scoped to /map", "Path=/map" in cookie_header)

print("\nrejected session cookie")
r = client.get("/map", cookies={"cmcp_session": "garbage"}, follow_redirects=False)
check("invalid cookie does not render the store", STORE_MARKER not in r.text, f"status {r.status_code}")
check("invalid cookie redirects to login", r.status_code == 302, f"status {r.status_code}")

print("\ncallback CSRF state")
r = client.get("/map/callback?code=abc&state=attacker", follow_redirects=False)
check("missing state cookie is rejected", r.status_code == 400, f"status {r.status_code}")
r = client.get("/map/callback?code=abc", cookies={"cmcp_state": "expected"}, follow_redirects=False)
check("absent state parameter is rejected", r.status_code == 400, f"status {r.status_code}")
r = client.get("/map/callback?state=expected", cookies={"cmcp_state": "expected"}, follow_redirects=False)
check("matching state without a code is rejected", r.status_code == 400, f"status {r.status_code}")

print("\nunconfigured deployment fails closed")
map_routes._verifier = _real_verifier
saved = os.environ.pop("MAP_CLIENT_ID")
r = client.get("/map", follow_redirects=False)
check("no MAP_CLIENT_ID does not render the store", STORE_MARKER not in r.text, f"status {r.status_code}")
check("no MAP_CLIENT_ID returns 503", r.status_code == 503, f"status {r.status_code}")
os.environ["MAP_CLIENT_ID"] = saved

print("\nauthorised access still works")


class _Token:
    expires_at = None


# The payload is a fixture, not the real store: this asserts that an accepted
# cookie reaches the render path, which is a different question from whether the
# transform is correct, and it keeps the test off Chroma entirely.
FIXTURE = {"projects": [{"name": "fixture-project", "tier": "personal", "slots": [],
                         "chars": 0, "chunks": 0, "archived": 0, "history": []}],
           "totals": {"projects": 1, "slots": 0, "chunks": 0, "chars": 0}}
map_routes.build_atlas_data = lambda store: FIXTURE
map_routes.get_store = lambda: None

with_verifier(_Stub(_Token()))
r = client.get("/map", cookies={"cmcp_session": "accepted-by-stub"}, follow_redirects=False)
check("valid cookie renders the page", r.status_code == 200, f"status {r.status_code}")
check("rendered page is the atlas", STORE_MARKER in r.text)
check("rendered page carries the payload", "fixture-project" in r.text)
check("page is not cacheable", "no-store" in r.headers.get("cache-control", ""))

print("\n/mcp is unaffected")
# RejectStreamGet was scoped to a path so it would stop swallowing /map. The
# reason it exists — answering the stream GET instead of hanging to the Lambda
# timeout — has to survive that change.
r = client.get("/mcp", follow_redirects=False)
check("GET /mcp still returns 405", r.status_code == 405, f"status {r.status_code}")
check("405 still advertises POST", r.headers.get("allow") == "POST")

print()
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all checks passed")
