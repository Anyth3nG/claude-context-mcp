# Memory policy for this project

This repo builds `context-mcp`, its own canonical memory store — cross-machine, cross-client (claude.ai and Claude Code both read/write it), reachable via the `search_context` and `save_update` MCP tools once the `context-mcp` server is connected.

Do NOT use the local, machine-only auto-memory system (`~/.claude/projects/.../memory/`) for anything project-relevant here — decisions, architecture, config, infra choices, or facts about this project. That system is local to this one machine and invisible to claude.ai and to Claude Code running anywhere else — using it here silently defeats the entire point of this project (consistent context across every machine and client).

Instead:
- Use `save_update` for anything decision-worthy about this project (see that tool's own description for what qualifies).
- Use `search_context` to check what's already known before assuming it isn't.

This applies specifically to project-scoped context. General working-style feedback about how you collaborate with the user is a separate concern and can still go through normal channels.
