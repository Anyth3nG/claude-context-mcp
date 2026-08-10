"""
Write tool: patch_context. Edits one passage of a summary, in place.

The default way to change a summary, and the reason change_update now exists
only for wholesale rewrites. change_update makes the caller regenerate the
whole document to move one line — a thousand output tokens to change a
handful of characters. This sends the diff.

Every refusal returns the stored document rather than only an error, because a
patch fails exactly when the caller's copy of the text has drifted from what is
stored — and an error string alone would send it back to guess at the same
stale text again.
"""
from __future__ import annotations
from typing import Annotated, Optional

from pydantic import Field

from shared.store import PatchAmbiguous, PatchFailed, PatchNoMatch, PatchNoOp, PatchSlotMissing

from mcp_server.context import get_store
from mcp_server.review_log import log_write

DESCRIPTION = """Change one passage inside a category's summary, leaving everything else untouched. THIS IS THE DEFAULT WAY TO UPDATE STORED CONTEXT — reach for it whenever a fact that already has a value has changed.

Send only what changed: `old_str` is text copied exactly from the stored summary, `new_str` is what replaces it. Nothing else in the document moves. Use change_update only when rewriting a whole category from scratch.

`old_str` must appear EXACTLY ONCE. If it appears several times, the call is refused rather than guessing — extend it with surrounding text until it is unique. If it doesn't appear at all, or the category has no summary yet, that is refused too. Every refusal returns the current stored text, so read it and retry against what is actually there.

To confirm the write landed cleanly, check the returned delta: it equals len(new_str) - len(old_str) exactly when nothing outside the match was disturbed."""


def patch_context(
    old_str: Annotated[
        str,
        Field(
            description="Text to replace, copied exactly from the stored summary. Must match "
            "one place and one place only — include surrounding context to disambiguate."
        ),
    ],
    new_str: Annotated[
        str,
        Field(description="What that text becomes. Empty string deletes the matched passage."),
    ],
    category: Annotated[
        str,
        Field(
            description="Which slot to edit: tech_stack, architecture, config, or decisions for "
            "project-scoped entries; preference, fact, tasks, or note for general ones. "
            "Close typos are auto-corrected."
        ),
    ],
    project: Annotated[
        Optional[str],
        Field(description="Name of the project. Omit for general (non-project-specific) entries."),
    ] = None,
    key: Annotated[
        Optional[str],
        Field(
            description="Sub-topic within the category, e.g. 'cognito' or 'deploy' under config. "
            "Omit for the category's main slot. Use get_index to see which keys already exist — "
            "reuse an existing one rather than coining a synonym, or the category fragments."
        ),
    ] = None,
) -> dict:
    store = get_store()
    try:
        result = store.patch_summary(
            old_str=old_str,
            new_str=new_str,
            category=category,
            project=project,
            key=key,
            source="live",
        )
    except PatchFailed as refusal:
        # Returned, not raised — the caller needs the stored text to build a
        # correct patch, and an exception string alone doesn't carry it. Same
        # contract as change_update's shrink refusal.
        response = {
            "patched": False,
            "refused": {
                PatchSlotMissing: "no_summary_yet",
                PatchNoMatch: "no_match",
                PatchAmbiguous: "not_unique",
                PatchNoOp: "no_change",
            }[type(refusal)],
            "reason": str(refusal),
        }
        if refusal.current is not None:
            response["current_content"] = refusal.current
        if isinstance(refusal, PatchAmbiguous):
            response["occurrences"] = refusal.count
        if isinstance(refusal, PatchSlotMissing):
            response["hint"] = "Nothing to patch here yet — use change_update to create this category."
        return response

    log_write(
        {
            "id": result["id"],
            "project": result["project"],
            "category": result["category"],
            "corrected_from": result.get("corrected_from"),
            "tier": result.get("tier"),
            "patch": {"old": old_str, "new": new_str},
            "chars_before": result["chars_before"],
            "chars_after": result["chars_after"],
        }
    )

    response: dict = {
        "patched": True,
        "id": result["id"],
        "category": result["category"],
        "chars_before": result["chars_before"],
        "chars_after": result["chars_after"],
        "delta": result["delta"],
    }
    if result.get("archived_id"):
        response["archived_id"] = result["archived_id"]
        response["note"] = (
            "A full copy of the pre-patch document was archived — this edit was "
            "large, or enough small ones have accumulated to warrant a checkpoint."
        )
    if result.get("corrected_from"):
        response["category_note"] = (
            f"'{result['corrected_from']}' isn't a valid category — used '{result['category']}'."
        )
    return response
