"""
Write tool: save_update. Wraps ContextStore.save().

Design call: live save_update writes always use type="chunk" (always-insert,
never overwritten), not type="summary" (deterministic-id, overwrite-in-place).
Summaries are meant to be a single, periodically regenerated distillation per
project+category — produced by the Phase 6 backfill/summarizer, which can see
a whole conversation (or project) at once. A single live save_update call only
ever sees one decision/fact in isolation; if it upserted onto the summary slot,
each new save would silently erase everything captured there before it. Chunks
accumulate instead, which is the correct behavior for point-in-time captures.
"""
from __future__ import annotations
from typing import Annotated, Optional

from pydantic import Field

from mcp_server.context import get_store
from mcp_server.review_log import log_write

DESCRIPTION = """Record something durable and decision-worthy from this conversation, so it's retrievable in future sessions — on ANY machine or client (claude.ai or Claude Code), not just this one. This should be the exception, not routine — most of a conversation is not worth saving.

This is the canonical cross-machine, cross-client store — it is NOT the same thing as any local, single-machine memory you might also maintain. Local memory doesn't follow the user to another device or to claude.ai, and can't see what was captured there either. So: still call this even if the content feels like it's "already saved" locally — a local save does not satisfy this tool's purpose, and skipping this because of one is exactly the failure mode this tool exists to prevent.

Save when: a real decision gets made (an architecture, tech stack, config, or other concrete choice for a project), the user states a lasting preference or fact about themselves, or the user sets or updates a goal they're working toward.

Do NOT save: hypotheticals or options being weighed but not yet chosen, restatements of something already saved IN THIS STORE (check with search_context if unsure, not against local memory), transient debugging detail, or anything you're not confident is worth surfacing again later. When in doubt, don't call this — a missed save is cheap to redo manually; a bad save pollutes future retrieval permanently.

Each call is stored as its own standalone entry — it never silently overwrites a previous save, so a history of decisions accumulates rather than getting erased by the next call."""


def save_update(
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

    response: dict = {"saved": True, "id": result["id"], "category": result["category"]}
    if result.get("corrected_from"):
        response["note"] = f"'{result['corrected_from']}' isn't a valid category — saved under '{result['category']}'."
    return response
