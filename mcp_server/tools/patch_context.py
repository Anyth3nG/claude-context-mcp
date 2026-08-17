"""
Write tool: patch_context. The single way to write a summary slot, in either of
the two shapes a summary write can take.

PATCH (old_str given) edits one passage in place. WHOLESALE (content given
instead) replaces or creates the slot outright — the job change_update used to
do as a separate tool until it was folded in here.

They are one tool because they are one intent: "make this slot say something
different." They stayed separate for a long time because the surrounding tools
had to teach when to pick which, and a caller that guessed wrong either
regenerated a thousand tokens to move a line, or destroyed the rest of a
document it forgot to mention. Dispatching on whether `old_str` is present
removes the choice: send a diff if you have one, send the document if you don't.

Every refusal returns the stored document rather than only an error, because a
write fails exactly when the caller's copy of the text has drifted from what is
stored — and an error string alone would send it back to guess at the same
stale text again.
"""
from __future__ import annotations
from typing import Annotated, Optional

from pydantic import Field

from shared.store import (
    MissingSummaryKey,
    PatchAmbiguous,
    PatchFailed,
    PatchNoMatch,
    PatchNoOp,
    PatchSlotMissing,
    SummaryShrinkRefused,
    UnknownSummaryKey,
)

from mcp_server.context import get_store
from mcp_server.review_log import log_write

DESCRIPTION = """Write a summary slot — the current state of one topic. THIS IS THE DEFAULT WAY TO UPDATE STORED CONTEXT.

Two shapes, chosen by whether you pass `old_str`:

PATCH — pass `old_str` and `new_str`. Changes one passage and leaves everything else untouched. PREFER THIS whenever the slot already has a value: it sends only the diff instead of making you regenerate a whole document to move one line. `old_str` must appear EXACTLY ONCE; if it appears several times the call is refused rather than guessing, so extend it with surrounding text until it is unique.

WHOLESALE — pass `content` and omit `old_str`. Replaces the slot entirely, or creates it if it does not exist yet. `content` must be the COMPLETE new state, not a fragment — whatever was stored is replaced, so sending only the changed part destroys the rest. A replacement under half the stored length is refused unless you pass `allow_shrink=true`, and opening a NEW key in a category that already has slots needs `create_key=true`.

Use add_update INSTEAD for point-in-time facts that should accumulate — a decision made, an event, something discovered. Those belong in history, not in a slot that gets overwritten.

Every refusal returns the current stored text, so read it and retry against what is actually there. The previous value is archived on any write that replaces enough of it, so a bad overwrite is recoverable through get_history."""


def patch_context(
    category: Annotated[
        str,
        Field(
            description="Which slot to write: tech_stack, architecture, config, or decisions for "
            "project-scoped entries; preference, fact, tasks, or note for general ones. "
            "Close typos are auto-corrected."
        ),
    ],
    old_str: Annotated[
        Optional[str],
        Field(
            description="PATCH MODE. Text to replace, copied exactly from the stored summary. "
            "Must match one place and one place only — include surrounding context to "
            "disambiguate. Omit entirely to replace the whole slot with `content` instead."
        ),
    ] = None,
    new_str: Annotated[
        Optional[str],
        Field(
            description="PATCH MODE. What `old_str` becomes. Empty string deletes the matched "
            "passage. Required when `old_str` is given, ignored otherwise."
        ),
    ] = None,
    content: Annotated[
        Optional[str],
        Field(
            description="WHOLESALE MODE. The COMPLETE new state of this slot, replacing whatever "
            "is stored. Not a fragment — sending only what changed destroys the rest. Use this "
            "to create a slot, or to rewrite one whose new value bears no resemblance to the old."
        ),
    ] = None,
    project: Annotated[
        Optional[str],
        Field(description="Name of the project. Omit for general (non-project-specific) entries."),
    ] = None,
    key: Annotated[
        Optional[str],
        Field(
            description="Sub-topic within the category, e.g. 'cognito' or 'deploy' under config. "
            "REQUIRED — every summary lives under a key; there is no keyless main slot. Use get_index to see which keys already exist — "
            "reuse an existing one rather than coining a synonym, or the category fragments."
        ),
    ] = None,
    tier: Annotated[
        Optional[str],
        Field(
            description='WHOLESALE MODE. "client" or "personal". Required when creating a slot '
            "under a project; a patch inherits it from the slot instead."
        ),
    ] = None,
    allow_shrink: Annotated[
        bool,
        Field(
            description="WHOLESALE MODE. Set true ONLY when deliberately condensing a bloated "
            "summary. Without it a replacement under half the stored length is refused, on the "
            "assumption a fragment was sent where the full new state was meant."
        ),
    ] = False,
    create_key: Annotated[
        bool,
        Field(
            description="WHOLESALE MODE. Required to open a NEW key in a category that already "
            "has slots. Without it the write is refused and the existing keys come back, closest "
            "first — a near-duplicate key splits a topic in two and nothing later notices."
        ),
    ] = False,
) -> dict:
    store = get_store()

    # Dispatch on old_str alone. Not on a combination of fields, and not on a
    # mode argument: one obvious signal, so a caller cannot land in the wrong
    # branch by filling in the wrong subset. Contradictory input is refused
    # rather than resolved by precedence, which would silently discard half of
    # what was sent.
    if old_str is not None and content is not None:
        return {
            "written": False,
            "refused": "ambiguous_mode",
            "reason": "Pass old_str + new_str to patch one passage, OR content to replace the "
            "whole slot — not both. Refused rather than guessing which you meant.",
        }
    if old_str is None and content is None:
        return {
            "written": False,
            "refused": "nothing_to_write",
            "reason": "Give either old_str + new_str (patch one passage) or content (replace the "
            "whole slot).",
        }
    if old_str is not None and new_str is None:
        return {
            "written": False,
            "refused": "missing_new_str",
            "reason": "old_str was given without new_str. To delete the matched passage, pass "
            'new_str as an empty string explicitly.',
        }

    if content is not None:
        return _wholesale(store, content, category, project, key, tier, allow_shrink, create_key)

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
        # contract as the wholesale shrink refusal below.
        response = {
            "written": False,
            "mode": "patch",
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
            response["hint"] = (
                "Nothing to patch here yet. Resend with `content` instead of old_str/new_str "
                "to create this slot."
            )
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

    stored_at = f"{result['project']}/{result['category']}"
    if result.get("key"):
        stored_at += f"/{result['key']}"
    response: dict = {
        "written": True,
        "mode": "patch",
        "stored_at": stored_at,
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
    if result.get("archived_split_into"):
        response["archived_split_into"] = result["archived_split_into"]
    if result.get("corrected_from"):
        response["category_note"] = (
            f"'{result['corrected_from']}' isn't a valid category — used '{result['category']}'."
        )
    return response


def _wholesale(store, content, category, project, key, tier, allow_shrink, create_key) -> dict:
    """
    Replace or create a slot outright — what change_update did before it was
    folded in here.

    Kept as a separate function rather than inlined: the two modes share an
    address and nothing else. Patching refuses on how the text MATCHES, replacing
    refuses on what the write would DESTROY, and interleaving those two sets of
    guards in one body is how one of them ends up quietly skipped.
    """
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
        # Returned, not raised: the caller can only avoid fragmenting the
        # category if it can see what is already in it. Ranked closest-first by
        # content, as a suggestion — the choice stays the caller's, because no
        # similarity cutoff separates synonyms from unrelated keys reliably.
        return {
            "written": False,
            "refused": "unknown_key",
            "reason": str(refusal),
            "existing_keys": [k if k else "(unkeyed)" for k in refusal.existing],
            "hint": "Reuse whichever of these already means the same thing, or resend "
                    "with create_key=true if this really is a new topic.",
        }
    except MissingSummaryKey as refusal:
        return {
            "written": False,
            "refused": "missing_key",
            "reason": str(refusal),
            "existing_keys": [k for k in refusal.existing if k],
            "hint": "Resend with a key. Reuse an existing one if it means the same thing, "
                    "or add create_key=true alongside a new key for a genuinely new topic.",
        }
    except SummaryShrinkRefused as refusal:
        # The model needs the current text to merge properly, and an exception
        # string alone does not carry it.
        return {
            "written": False,
            "refused": "content_too_short",
            "reason": str(refusal),
            "current_content": refusal.previous,
            "hint": "Merge your change into the text above and resend it in full, "
                    "or pass allow_shrink=true if the summary is genuinely being condensed.",
        }
    except ValueError as refusal:
        # Bad tier, or a tier missing on a project-scoped write.
        return {"written": False, "refused": "invalid_argument", "reason": str(refusal)}

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

    stored_at = f"{result['project']}/{result['category']}"
    if result.get("key"):
        stored_at += f"/{result['key']}"
    response: dict = {
        "written": True,
        "mode": "wholesale",
        "stored_at": stored_at,
        "id": result["id"],
        "category": result["category"],
        "had_previous_value": result.get("previous") is not None,
    }
    if result.get("previous") is not None:
        response["replaced_content"] = result["previous"]
        response["archived_id"] = result["archived_id"]
        response["note"] = (
            "The previous value was archived and stays reachable through get_history. "
            "Check replaced_content — if it held anything your new content dropped, "
            "resend with the full merged text."
        )
    if result.get("archived_split_into"):
        response["archived_split_into"] = result["archived_split_into"]
    if result.get("corrected_from"):
        response["category_note"] = (
            f"'{result['corrected_from']}' isn't a valid category — used '{result['category']}'."
        )
    return response
