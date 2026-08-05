"""
Write tool: change_update. Replaces one summary slot — the "whiteboard" half.

NO LONGER THE DEFAULT WRITE. patch_context edits a passage in place and is the
right tool for almost every change; this one exists for the wholesale rewrite,
where the new document genuinely bears no resemblance to the old. Preferring it
by habit means regenerating a thousand-token summary to move one line.

One living document per project+category, at a deterministic id. Updating the
tech stack physically cannot touch the architecture slot: different id,
different row.

Replacing is destructive in a way appending or patching isn't, so three
safeguards apply (see ContextStore.update_summary): the previous content comes
back in the response, an implausibly short replacement is refused outright, and
whatever was there is archived as a chunk before being overwritten.
"""
from __future__ import annotations
from typing import Annotated, Optional

from pydantic import Field

from shared.store import SummaryShrinkRefused, UnknownSummaryKey

from mcp_server.context import get_store
from mcp_server.review_log import log_write

DESCRIPTION = """Replace a category's summary WHOLESALE — creating it for the first time, or rewriting it from scratch when the new version bears no resemblance to the old.

Use patch_context INSTEAD for ordinary changes to something that already has a value. This tool REPLACES the entire category and does not merge, so `content` must be the COMPLETE new state; sending only the changed part destroys the rest, and regenerating a long summary to alter a line or two is exactly the waste patch_context exists to avoid. A replacement much shorter than what's stored is refused.

Use add_update INSTEAD for point-in-time facts that should accumulate — a decision made, an event, something discovered. Those belong in history, not in a slot that gets overwritten.

The previous content is returned and archived automatically, so a bad overwrite is recoverable."""


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
    key: Annotated[
        Optional[str],
        Field(
            description="Sub-topic within the category, e.g. 'cognito' or 'deploy' under config. "
            "Omit for the category's main slot. Use get_index to see which keys already exist — "
            "reuse an existing one rather than coining a synonym, or the category fragments."
        ),
    ] = None,
    create_key: Annotated[
        bool,
        Field(
            description="Required to open a NEW key in a category that already has slots. Without "
            "it the write is refused and the existing keys come back, closest first — because a "
            "near-duplicate key ('compute' beside 'lambda') splits a topic in two and nothing "
            "later notices. Set true only after checking that none of them already means this."
        ),
    ] = False,
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
            key=key,
            create_key=create_key,
        )
    except UnknownSummaryKey as refusal:
        # Returned, not raised, for the same reason as the shrink refusal: the
        # caller can only avoid fragmenting the category if it can see what is
        # already in it. Ranked closest-first by content, as a suggestion — the
        # choice is the caller's, because no similarity cutoff separates
        # synonyms from unrelated keys reliably enough to make it automatically.
        return {
            "changed": False,
            "refused": "unknown_key",
            "reason": str(refusal),
            "existing_keys": [k if k else "(unkeyed)" for k in refusal.existing],
            "hint": "Reuse whichever of these already means the same thing, or resend "
                    "with create_key=true if this really is a new topic.",
        }
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
