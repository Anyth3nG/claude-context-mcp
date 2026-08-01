"""
MCP server entrypoint. Transport is selected by MCP_TRANSPORT:
  - "stdio" (default) — local Claude Code use, launched per-session as a subprocess.
  - "streamable-http" — the remote mode (Phase 3+): one long-running process,
    reached over the network by claude.ai and Claude Code alike. MCP_HOST /
    MCP_PORT control the bind (default 127.0.0.1:8000 — nginx terminates TLS
    in front of it on the instance; never bind it to a public interface bare).
"""
import os
import sys
from pathlib import Path

# claude mcp add invokes this file directly (`python .../mcp_server/server.py`),
# not as `-m mcp_server.server` — so the repo root isn't on sys.path by default
# and the `mcp_server.*` / `shared.*` absolute imports below would fail. Put it
# there explicitly rather than depending on how this script gets launched.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer

from mcp_server.map_routes import register_map_routes
from mcp_server.tools.search_context import DESCRIPTION as SEARCH_CONTEXT_DESCRIPTION, search_context
from mcp_server.tools.save_update import DESCRIPTION as SAVE_UPDATE_DESCRIPTION, save_update

# load_dotenv() with no args searches upward from cwd — fine when launched
# from the repo, but this server is meant to be reachable from any project
# directory, where there's no .env to find. Anchor it to the repo explicitly.
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

mcp = MCPServer("context-mcp")
mcp.add_tool(search_context, description=SEARCH_CONTEXT_DESCRIPTION)
mcp.add_tool(save_update, description=SAVE_UPDATE_DESCRIPTION)
register_map_routes(mcp)  # /map + /map/data — HTTP transport only

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
