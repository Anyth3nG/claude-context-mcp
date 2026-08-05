"""
Read tool: get_index. What the store holds, without any of the contents.

The intended first call of a session, and the reason get_brief should no longer
be one. A brief returns every summary whole — measured at ~4,280 tokens for a
single project against this store — which is the right price for "tell me
everything" and a bad one for "is there anything here". The index answers the
second question in roughly fifty tokens, and what it returns is exactly what's
needed to decide whether to spend more.

No embedding call: both underlying reads are metadata lookups, not similarity
queries.
"""
from __future__ import annotations
from typing import Annotated, Optional

from pydantic import Field

from mcp_server.context import get_store

DESCRIPTION = """List what this store knows about — every project, which categories have a summary, how big each is, and when it last changed. No content, just the map.

CALL THIS FIRST when you need stored context and don't know what's there. It is far cheaper than get_brief (tens of tokens rather than thousands), needs no query, and tells you exactly which follow-up is worth making: get_value for one category, get_brief for a whole project, search_context for history.

OPEN A SESSION WITH detail="projects". That returns the table of contents — one line per project with its size and shape, no per-slot detail — for roughly a tenth of the full form. It is the cheapest possible way to learn whether this store holds anything relevant to what you are about to do, and the answer is usually "yes, in one specific project", which is then a single get_brief away.

Use the default detail="slots" once you know which project you care about, or when you need to see which individual categories exist and which have prior_versions worth pulling with get_history.

Omit `project` to see everything, which is the normal way to get oriented at the start of a session. Pass one to scope it.

`chars` is the size of the stored summary, so you can tell what a get_brief or get_value would actually cost before paying it. `updated` is the last write to that category — useful for spotting context that has gone stale. `history_chunks` counts accumulated point-in-time entries, which are reachable through search_context rather than get_brief.

An empty result means the store holds nothing yet, not that a lookup failed."""


def get_index(
    project: Annotated[
        Optional[str],
        Field(
            description="Scope the index to one project. Omit to list every project, "
            "including general (non-project-specific) entries."
        ),
    ] = None,
    detail: Annotated[
        str,
        Field(
            description='"projects" for the table of contents — one line per project, no '
            'per-slot detail, and the right way to open a session. "slots" (the default) '
            "for the full map including every category, its size, and its version count."
        ),
    ] = "slots",
) -> dict:
    store = get_store()
    try:
        result = store.index(project=project, detail=detail)
    except ValueError as exc:
        return {"error": str(exc), "projects": {}}

    if not result["projects"]:
        return {
            **result,
            "note": (
                f"Nothing stored for '{project}' yet."
                if project
                else "The store is empty — nothing has been saved yet."
            ),
        }
    return result
