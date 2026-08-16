"""
Write tool: archive_slot. Retires a summary slot whose work is finished.

The gap this closes: a summary slot can hold "what we are doing" or "what we
did", and only the first is current state. Nothing could express the second —
change_update archives the old text but insists on a new live value, and there
is no value meaning "this no longer applies". So finished work kept loading with
every brief, charging tokens for a question nobody was asking.

Archived as SUPERSEDED, never retired. Retired means wrong; superseded means it
was true when written and still is as history — which is what finished work is.
"""
from __future__ import annotations
from typing import Annotated, Optional

from pydantic import Field

from shared.store import PatchSlotMissing

from mcp_server.context import get_store
from mcp_server.review_log import log_write

DESCRIPTION = """Retire a summary slot whose work is FINISHED, so it stops loading with every brief while staying retrievable.

Use this when a slot has stopped describing current state and started describing history — most often a finished piece of work. `tasks` is meant to answer "what needs doing next"; once something is done, its detail is a record of what happened, and leaving it live means every future session pays for it on every get_context.

The text is not deleted. It is archived as a SUPERSEDED chunk, keeping its existing embedding, and stays reachable through ordinary search_context results and through get_history on the slot — which is how you compare what exists now against what was worked on before.

DO use this for: a finished piece of work, a task that has been done, a config surface that no longer exists, any slot that has become purely historical.

Do NOT use this for a slot that is merely long, out of date, or partly wrong. Long belongs in patch_context, out of date belongs in patch_context or change_update, and wrong belongs in retire_chunk. Archiving something still in use makes it invisible to get_context, and a future session will not know to ask for it.

Leave a `reason` — it is stored on the archived copy and is what tells a later reader why this stopped being current rather than having been abandoned."""


def archive_slot(
    category: Annotated[
        str,
        Field(
            description="Which category the slot lives in: tech_stack, architecture, config, "
            "decisions for project-scoped entries; preference, fact, tasks, note for general ones."
        ),
    ],
    project: Annotated[
        Optional[str],
        Field(description="Name of the project. Omit for general (non-project-specific) entries."),
    ] = None,
    key: Annotated[
        Optional[str],
        Field(
            description="Sub-topic within the category, e.g. 'browser-login' under tasks. "
            "REQUIRED — every summary lives under a key; there is no keyless main slot. get_index lists the ones in use."
        ),
    ] = None,
    reason: Annotated[
        str,
        Field(
            description="Why this stopped being current — usually what completed it, and when. "
            "Stored on the archived copy, so write it for someone who was not here."
        ),
    ] = "",
) -> dict:
    store = get_store()
    try:
        result = store.archive_slot(project=project, category=category, key=key, reason=reason)
    except PatchSlotMissing as exc:
        return {"archived": False, "error": str(exc), "category": category, "key": key}

    log_write(
        {
            "tool": "archive_slot",
            "id": result["id"],
            "archived_id": result["archived_id"],
            "project": result["project"],
            "category": category,
            "key": key,
            "chars_freed": result["chars_freed"],
            "reason": result["reason"],
        }
    )
    return result
