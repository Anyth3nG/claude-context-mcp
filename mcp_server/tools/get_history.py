"""
Read tool: get_history. Every archived version of one slot, newest first.

The gap this closes: history was only reachable through search_context, which
means guessing at words. A caller that knows exactly which slot it cares about
still had to phrase a query and hope ranking cooperated — and ranking is the
wrong instrument when the target is already known by name.

The chain existed all along. Every archive records superseded_from and
superseded_at; nothing could read them. This is a metadata lookup, so there is
no embedding call, no ranking, and no top_k to fall off the end of.
"""
from __future__ import annotations
from typing import Annotated, Optional

from pydantic import Field

from mcp_server.context import get_store

DESCRIPTION = """Get how ONE slot changed over time — every archived version, newest first, with the current value alongside for comparison.

Use this when you know which entry you care about and want its past rather than its present: "what did config/rotation say before", "when did this change", "what was the phase B plan we finished". get_index shows a `prior_versions` count on any slot that has one, which is how you know there is something here to ask for.

Prefer this over search_context for a known slot. Search ranks by similarity and truncates, so asking it for history means guessing at wording and hoping the right version places in the top few. This is a direct lookup: complete text, every version, in order, no embedding call.

Returns the live value (or null if the slot was archived outright), then each archived version with its text, when it was written, when it stopped being current, and any reason recorded at the time.

For a slot archived by archive_slot — a finished phase, say — `current` is null and `archived_slot` is true. The work is done and the record is here.

This is slot-scoped on purpose. "How did this entry evolve" is a different question from "what happened last Tuesday"; if you want the second, search_context with a date in the query is the closest thing today."""


def get_history(
    category: Annotated[
        str,
        Field(
            description="Which category the slot lives in: tech_stack, architecture, config, "
            "decisions for project-scoped entries; preference, fact, goal, note for general ones."
        ),
    ],
    project: Annotated[
        Optional[str],
        Field(description="Name of the project. Omit for general (non-project-specific) entries."),
    ] = None,
    key: Annotated[
        Optional[str],
        Field(
            description="Sub-topic within the category, e.g. 'rotation' under config. "
            "Omit for the category's main slot."
        ),
    ] = None,
) -> dict:
    store = get_store()
    result = store.slot_history(project=project, category=category, key=key)

    if not result["versions"]:
        result["note"] = (
            f"No archived versions of {result['slot']} — it has either never been "
            "rewritten, or was only ever patched in small pieces that did not "
            "trigger an archive. The current value is the whole story."
        ) if result["current"] else (
            f"Nothing at {result['slot']}: no live summary and no archived versions. "
            "Check get_index for the slots this project actually has."
        )
    return result
