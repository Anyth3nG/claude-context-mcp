"""
Constellation map — a visual layer over the context store, served by the same
process (HTTP transport only; custom routes don't exist over stdio).

  GET /map       the page (self-contained HTML, no external assets)
  GET /map/data  every entry in the store, shaped for the visualization

NOTE for Phase 3: custom routes bypass MCP-level auth by design (see
MCPServer.custom_route docstring) — when this goes public, nginx/auth.py must
cover /map* too, since the data endpoint exposes store contents.
"""
from __future__ import annotations
from pathlib import Path

from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse

MAP_HTML = Path(__file__).resolve().parent / "static" / "map.html"

# Tooltip preview only — full documents stay in the store, the map never
# needs more than this per node.
PREVIEW_CHARS = 280


def register_map_routes(mcp) -> None:
    from mcp_server.context import get_store

    @mcp.custom_route("/map", methods=["GET"])
    async def map_page(request: Request):
        return FileResponse(MAP_HTML, media_type="text/html")

    @mcp.custom_route("/map/data", methods=["GET"])
    async def map_data(request: Request):
        store = get_store()
        data = store.collection.get(include=["documents", "metadatas"])
        entries = []
        for id_, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
            preview = doc if len(doc) <= PREVIEW_CHARS else doc[:PREVIEW_CHARS] + " …"
            entries.append(
                {
                    "id": id_,
                    "preview": preview,
                    "project": meta.get("project", "general"),
                    "tier": meta.get("tier"),
                    "category": meta.get("category", "note"),
                    "type": meta.get("type", "chunk"),
                    "source": meta.get("source", "live"),
                    "timestamp": meta.get("timestamp"),
                }
            )
        # Stable order: oldest first, so first-seen project → color-slot
        # assignment never repaints when new entries arrive.
        entries.sort(key=lambda e: e["timestamp"] or "")
        return JSONResponse({"entries": entries})
