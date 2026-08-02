"""
Write tool: add_update. Appends a chunk — the "diary" half of the store.

Replaces the older save_update, which did exactly this under a name that didn't
say whether it added or changed. Pairs with change_update, which replaces a
living summary slot instead of appending.

Always type="chunk": insert-only, unique id, never overwritten. A single live
call sees one fact in isolation, so it must never touch a summary slot — doing
so would erase everything captured there before it.
"""
from __future__ import annotations
from typing import Annotated, Optional

from pydantic import Field

from mcp_server.context import get_store
from mcp_server.review_log import log_write

DESCRIPTION = """Append a durable, decision-worthy fact from this conversation so it's retrievable in future sessions — on ANY machine or client (claude.ai or Claude Code), not just this one. This should be the exception, not routine; most of a conversation is not worth saving.

This is the canonical cross-machine, cross-client store — NOT the same thing as any local, single-machine memory. Local memory doesn't follow the user to another device or to claude.ai, and can't see what was captured there. So still call this even if the content feels "already saved" locally; skipping it because of a local save is exactly the failure mode this tool prevents.

Use add_update for point-in-time facts that should ACCUMULATE: a decision that was made, an event, a discovery, something the user stated. Nothing is ever overwritten, so a history builds up over time.

Use change_update INSTEAD when you're revising the CURRENT STATE of something that already has a value — the tech stack, the architecture, the config. Appending a corrected version with this tool leaves the outdated one in place, and future sessions then have to read both and guess which one still holds.

Save when: a real decision gets made, the user states a lasting preference or fact about themselves, or the user sets or updates a goal.

Do NOT save: hypotheticals or options being weighed but not chosen, restatements of something already in THIS store (check with search_context, not against local memory), transient debugging detail, or anything you're not confident is worth surfacing again. When in doubt, don't — a missed save is cheap to redo; a bad save pollutes retrieval permanently."""


def add_update(
    content: Annotated[
        str,
        Field(description="The durable fact or decision itself, written so it stands alone without the surrounding conversation."),
    ],
    category: Annotated[
        str,
        Field(
            description="tech_stack, architecture, config, or decisions for project-scoped entries; "
            "preference, fact, goal, or note for general ones. Close typos are auto-corrected."
        ),
    ],
    project: Annotated[
        Optional[str],
        Field(description="Name of the project this belongs to. Omit for general (not project-specific) entries."),
    ] = None,
    tier: Annotated[
        Optional[str],
        Field(
            description='"client" or "personal" — signals how much retrieval depth is warranted. '
            "Required if project is set, omitted otherwise."
        ),
    ] = None,
) -> dict:
    store = get_store()
    result = store.save(
        document=content,
        category=category,
        type="chunk",
        project=project,
        tier=tier,
        source="live",
    )

    log_write(
        {
            "id": result["id"],
            "project": result["project"],
            "category": result["category"],
            "corrected_from": result.get("corrected_from"),
            "tier": result.get("tier"),
            "content": content,
        }
    )

    response: dict = {"added": True, "id": result["id"], "category": result["category"]}
    if result.get("corrected_from"):
        response["note"] = f"'{result['corrected_from']}' isn't a valid category — saved under '{result['category']}'."
    return response
