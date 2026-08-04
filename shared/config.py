"""
Secret loading for the deployed path.

The deployed function holds no credentials in its own configuration. It holds
only the *name* of a Secrets Manager secret; the values are fetched at startup
using the function's execution role. The reason is narrow but real: reading a
Lambda's configuration returns its environment variables in plaintext, and the
CI deploy role needs configuration-read access to poll SnapStart status. Keep
the credentials in the environment and CI can read them. Keep them in Secrets
Manager and it can't.

Everything downstream still reads os.environ, so this is the only place that
knows secrets are stored remotely. Locally, where CONTEXT_MCP_SECRET_ID isn't
set, this is a no-op and .env continues to work unchanged.
"""
from __future__ import annotations

import json
import os

SECRET_ID_VAR = "CONTEXT_MCP_SECRET_ID"

# One secret holding a JSON object, not one secret per value: Secrets Manager
# bills per secret per month, so four separate secrets would cost four times as
# much for no benefit.
#
# AUTH_TOKEN used to be here. It guarded the /map routes and was accepted as a
# fallback on /mcp; both uses are gone and the endpoint is OAuth-only, so the
# key is no longer required. Any value still sitting in the secret is ignored —
# extra keys are loaded into the environment but nothing reads them.
_EXPECTED_KEYS = (
    "VOYAGE_API_KEY",
    "CHROMA_TENANT",
    "CHROMA_DATABASE",
    "CHROMA_API_KEY",
)

_loaded = False


def load_secrets(force: bool = False) -> bool:
    """
    Populate os.environ from the configured secret. Returns True if a secret
    was read, False if none is configured (the local case).

    Runs once per process — on Lambda that means once per cold start, with the
    values surviving in os.environ across warm invocations, so this costs one
    API call per container rather than one per request.
    """
    global _loaded
    if _loaded and not force:
        return True

    secret_id = os.environ.get(SECRET_ID_VAR)
    if not secret_id:
        return False

    # Imported lazily, and not listed in requirements-lambda.txt, because the
    # Lambda Python runtime already ships boto3 — bundling it again would add
    # weight to a bundle that has a hard size limit.
    import boto3

    resp = boto3.client("secretsmanager").get_secret_value(SecretId=secret_id)
    payload = json.loads(resp["SecretString"])

    # setdefault, not assignment: an explicitly-set environment variable wins.
    # That keeps it possible to override a single value for debugging without
    # editing the shared secret.
    for key, value in payload.items():
        if value is not None:
            os.environ.setdefault(key, str(value))

    missing = [k for k in _EXPECTED_KEYS if not os.environ.get(k)]
    if missing:
        # Fail here, naming what's absent, rather than letting ContextStore
        # raise a confusing "no VOYAGE_API_KEY" three frames later.
        raise RuntimeError(
            f"Secret '{secret_id}' is missing required keys: {', '.join(missing)}. "
            f"It must be a JSON object containing: {', '.join(_EXPECTED_KEYS)}."
        )

    _loaded = True
    return True
