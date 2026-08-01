"""
Shared ContextStore instance for the MCP server's tools — one process, one
ChromaDB connection, so search_context and save_update aren't each opening
their own client.
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
        _store = ContextStore(
            persist_path=os.environ.get("CHROMA_PATH", DEFAULT_CHROMA_PATH),
            allow_local_fallback=os.environ.get("ALLOW_LOCAL_FALLBACK") == "1",
        )
    return _store
