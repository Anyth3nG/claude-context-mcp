"""
Shared ContextStore instance for the MCP server's tools — one process, one
backend connection, so the tools aren't each opening their own client.
"""
from __future__ import annotations
import os
from pathlib import Path

from shared.store import ContextStore

# The whole point of this server is to be ONE consistent store regardless of
# which project directory it happens to get launched from (it's meant to be
# registered across machines/clients, not just this one repo) — so the
# default persist path must be anchored to this file's location, not to
# whatever the subprocess's cwd happens to be. A bare relative "./chroma_data"
# would silently fragment into a different store per launch directory.
DEFAULT_CHROMA_PATH = str(Path(__file__).resolve().parent.parent / "chroma_data")

_store: ContextStore | None = None


def get_store() -> ContextStore:
    global _store
    if _store is None:
        # ALLOW_LOCAL_FALLBACK is for local/offline dev of the MCP server itself
        # (e.g. proving the tool wiring works before a Voyage key is configured).
        # Defaults to False so production behavior still matches docs/schema.md:
        # Voyage is required unless someone deliberately opts out.
        #
        # CONTEXT_MCP_FORCE_LOCAL is the tool-layer equivalent of the use_cloud=False
        # that ContextStore's own smoke tests pass, and it exists because clearing
        # the CHROMA_* variables is NOT enough to stay off the real store: server.py
        # calls load_dotenv() at import, so anything unset in the environment is
        # immediately repopulated from .env. Exercising the tools against a
        # developer's populated .env therefore writes fixture entries straight into
        # the shared Cloud store — which is exactly what happened once. Set this to
        # 1 in any test that drives the tools.
        table = os.environ.get("DYNAMODB_TABLE")
        if table:
            # The backend is still chosen by configuration rather than by a code
            # edit, but this is no longer a rollback switch. Chroma has been
            # retired: its credentials are out of the secret and chromadb is out
            # of the deployed bundle, so unsetting DYNAMODB_TABLE in Lambda now
            # fails at startup (shared/config.py says so in as many words)
            # instead of quietly serving a stale store. The branch below survives
            # for local development, where chromadb is still installed and the
            # contract in shared/conformance.py runs against both backends.
            from shared.dynamo_driver import DynamoDriver
            from shared.store import DEFAULT_VOYAGE_MODEL, VoyageRestEmbedding

            api_key = os.environ.get("VOYAGE_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "DYNAMODB_TABLE is set but VOYAGE_API_KEY is not. DynamoDB does "
                    "not generate embeddings, so the store cannot write or search "
                    "without one."
                )
            driver = DynamoDriver(table, VoyageRestEmbedding(api_key),
                                  region=os.environ.get("AWS_REGION"))
            # Refuses to open a table written with a different embedding model.
            # Mixing embedding spaces corrupts similarity search silently — every
            # write succeeds and every search returns plausible nonsense — so this
            # is checked once, at construction, rather than trusted.
            driver.assert_embedding_space(DEFAULT_VOYAGE_MODEL)
            _store = ContextStore(driver=driver)
        else:
            _store = ContextStore(
                persist_path=os.environ.get("CHROMA_PATH", DEFAULT_CHROMA_PATH),
                allow_local_fallback=os.environ.get("ALLOW_LOCAL_FALLBACK") == "1",
                use_cloud=False if os.environ.get("CONTEXT_MCP_FORCE_LOCAL") == "1" else None,
            )
    return _store
