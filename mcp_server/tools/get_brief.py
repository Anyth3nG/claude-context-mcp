"""
Read tool: get_brief. Every living summary for a project, returned whole.

The counterpart to search_context, and the right instrument for "what is the
current state of X". Search ranks by similarity and truncates each hit to keep
one call from flooding the conversation; that's correct for exploring history,
but it means a long entry is only ever visible from the top. A brief is a
deterministic lookup of the summary slots, so nothing is ranked away or cut.

No longer the way to OPEN a session, though. Returning everything whole is the
most expensive read here, and paying it before knowing whether the store holds
anything relevant is the wrong default — get_index answers that for a fraction
of the cost, and usually redirects to a single get_value.
"""
from __future__ import annotations
from typing import Annotated, Optional

from pydantic import Field

from mcp_server.context import get_store

DESCRIPTION = """Get the complete current-state brief for a project — its tech stack, architecture, configuration and standing decisions — as whole documents, not search results.

This returns EVERYTHING for a project untruncated, so it is the most expensive read here (thousands of tokens for a well-populated project). Use it when you genuinely want the whole picture: the user asks what a project is, how it works, or where it stands, or you are about to do substantial work across it.

Check get_index FIRST if you don't already know what's stored. It costs a fraction of this, shows which categories exist and how large each one is, and will often point you at get_value for the single category you actually needed.

Prefer this over search_context whenever the question is about CURRENT STATE. search_context ranks by similarity and shortens each result, so a long entry is only ever partly visible; it is the right tool for history — "what did we decide about X", "why did we do Y", "when did Z change" — and the wrong one for "what is the stack".

Returns one entry per category that has a summary, each with its full text and metadata. An empty result means the project has no summaries yet, not that the project is unknown — try search_context in that case."""


def get_brief(
    project: Annotated[
        Optional[str],
        Field(
            description='Name of the project (e.g. "context-mcp"). Omit to get the '
            "general, non-project-specific brief instead."
        ),
    ] = None,
) -> dict:
    store = get_store()
    entries = store.get_brief(project=project)
    return {
        "project": project or "general",
        "categories": [e["category"] for e in entries],
        "entries": entries,
        "count": len(entries),
        **(
            {}
            if entries
            else {
                "note": "No summaries recorded for this project yet. "
                "Try search_context for history, or record current state with change_update."
            }
        ),
    }
