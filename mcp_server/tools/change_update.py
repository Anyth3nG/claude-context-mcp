"""
Write tool: change_update. Replaces one summary slot — the "whiteboard" half.

One living document per project+category, at a deterministic id. Updating the
tech stack physically cannot touch the architecture slot: different id,
different row. That granularity is the whole point — a change to one thing
never requires resending the rest of a project's brief.

Replacing is destructive in a way appending isn't, so three safeguards apply
(see ContextStore.update_summary): the previous content comes back in the
response, an implausibly short replacement is refused outright, and whatever
was there is archived as a chunk before being overwritten.
"""
from __future__ import annotations
from typing import Annotated, Optional

from pydantic import Field

from shared.store import SummaryShrinkRefused

from mcp_server.context import get_store
from mcp_server.review_log import log_write

DESCRIPTION = """Update the CURRENT STATE of one thing about a project, replacing whatever was recorded before. Use this when a fact that already has a value has changed — the tech stack, the architecture, the config, a standing preference.

Each project+category is ONE living document. Writing to tech_stack replaces only tech_stack; architecture, config and everything else are untouched. So when something in a brief changes, update just that category — never resend the whole brief.

CRITICAL: this REPLACES the entire contents of that category, it does not merge. `content` must be the COMPLETE new state, not just the part that changed. Read the current value first (search_context, or the brief you were given) and send it back in full with your change folded in. Sending only the changed fragment destroys everything else in that category — a replacement much shorter than what's stored is refused for exactly this reason.

Use add_update INSTEAD for point-in-time facts that should accumulate — a decision that was made, an event, something discovered. Those belong in the history, not in a slot that gets overwritten.

The previous content is returned so you can confirm nothing was lost, and it's archived automatically, so a bad overwrite is recoverable rather than fatal."""


def change_update(
    content: Annotated[
        str,
        Field(description="The COMPLETE new state of this project+category, not a delta. Whatever was stored before is replaced entirely."),
    ],
    category: Annotated[
        str,
        Field(
            description="Which slot to replace: tech_stack, architecture, config, or decisions for "
            "project-scoped entries; preference, fact, goal, or note for general ones. "
            "Close typos are auto-corrected."
        ),
    ],
    project: Annotated[
        Optional[str],
        Field(description="Name of the project this belongs to. Omit for general (not project-specific) entries."),
    ] = None,
    tier: Annotated[
        Optional[str],
        Field(
            description='"client" or "personal". Required if project is set, omitted otherwise.'
        ),
    ] = None,
    allow_shrink: Annotated[
        bool,
        Field(
            description="Set true ONLY when deliberately condensing a bloated summary. Without it, a "
            "replacement under half the length of the current value is refused, on the assumption "
            "that a fragment was sent where the full new state was meant."
        ),
    ] = False,
) -> dict:
    store = get_store()
    try:
        result = store.update_summary(
            document=content,
            category=category,
            project=project,
            tier=tier,
            source="live",
            allow_shrink=allow_shrink,
        )
    except SummaryShrinkRefused as refusal:
        # Returned rather than raised: the model needs the current text in order
        # to merge properly, and an exception string alone doesn't give it that.
        return {
            "changed": False,
            "refused": "content_too_short",
            "reason": str(refusal),
            "current_content": refusal.previous,
            "hint": "Merge your change into the text above and resend it in full, "
                    "or pass allow_shrink=true if the summary is genuinely being condensed.",
        }

    log_write(
        {
            "id": result["id"],
            "project": result["project"],
            "category": result["category"],
            "corrected_from": result.get("corrected_from"),
            "tier": result.get("tier"),
            "content": content,
            "replaced": result.get("previous"),
        }
    )

    response: dict = {
        "changed": True,
        "id": result["id"],
        "category": result["category"],
        "had_previous_value": result.get("previous") is not None,
    }
    if result.get("previous") is not None:
        response["replaced_content"] = result["previous"]
        response["archived_id"] = result["archived_id"]
        response["note"] = (
            "The previous value was archived and is excluded from normal search. "
            "Check replaced_content — if it held anything your new content dropped, "
            "call change_update again with the full merged text."
        )
    if result.get("corrected_from"):
        response["category_note"] = (
            f"'{result['corrected_from']}' isn't a valid category — used '{result['category']}'."
        )
    return response
