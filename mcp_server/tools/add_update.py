"""
Write tool: add_update. Appends chunks — the "diary" half of the store.

Pairs with patch_context, which writes a living summary slot
instead of appending.

Always type="chunk": insert-only, content-addressed id, never overwritten. A
single live call sees one fact in isolation, so it must never touch a summary
slot — doing so would erase everything captured there before it.

Takes a list as readily as a single string, and that is load-bearing rather
than a convenience. Long entries are only ever returned truncated by
search_context, so a sprawling fact is a partly-invisible fact; splitting is
the fix, and batching through one upsert (one embedding call, whatever the
count) is what makes splitting cost nothing.
"""
from __future__ import annotations
from typing import Annotated, Optional, Union

from pydantic import Field

from shared.store import MAX_DOC_CHARS, UnknownCategory

from mcp_server.context import get_store
from mcp_server.review_log import log_write

DESCRIPTION = """Append a durable, decision-worthy fact from this conversation so it's retrievable in future sessions — on ANY machine or client (claude.ai or Claude Code), not just this one. This should be the exception, not routine; most of a conversation is not worth saving.

This is the canonical cross-machine, cross-client store — NOT any local, single-machine memory. Local memory doesn't follow the user to another device or to claude.ai. So still call this even if the content feels "already saved" locally; skipping it for that reason is exactly the failure this tool prevents.

SPLIT AS YOU WRITE. `content` accepts a LIST, and several focused entries cost the same as one long one — the whole list is embedded in a single call. Retrieval truncates long entries to ~1000 characters, so anything longer is only ever partly visible no matter how it's searched for. One self-contained fact per list item; never pad a short fact to fill one.

add_update WRITES HISTORY. Everything it stores is filed as history (source=appended), never as current state: use it for what HAPPENED, not for what is TRUE — an event, a measurement, an observation that is not a revision of one statement, finished work whose slot no longer exists. Nothing is overwritten.

Use patch_context INSTEAD for anything that describes current state: a decision, a preference, a fact about the user, a task. Those live in a slot, where get_context finds them; appended here they are history from the moment they are written, invisible to get_context, and a future session reading the slot never learns they exist. Record a decision here only as the event — when and why — beside the slot that holds it.

Pass `key` when the fact clearly belongs to one sub-topic — it files the entry alongside that slot's history instead of loose in the category. Unlike a summary key it is ungated: the key need not already have a summary, and coining a new one is fine. Omit it rather than guessing; a wrong key is worse than none.

Save when: something happened that a future session would want to know happened — a deploy, an incident, a measurement, a step completed.

Do NOT save: hypotheticals or options weighed but not chosen, restatements of something already in THIS store (check with search_context), transient debugging detail, or anything you're unsure is worth surfacing again. When in doubt, don't — a missed save is cheap to redo; a bad save pollutes retrieval permanently."""


def add_update(
    content: Annotated[
        Union[str, list[str]],
        Field(
            description="The durable fact, written to stand alone without the surrounding "
            "conversation. Pass a LIST to record several facts at once — preferred whenever "
            "the material covers more than one thing, or would otherwise run past ~1000 "
            "characters and be truncated on retrieval."
        ),
    ],
    category: Annotated[
        str,
        Field(
            description="What kind of entry this is. Built in: tech_stack, architecture, config, "
            "decisions (usually project-scoped); preference, fact, tasks, note (usually general). "
            "Categories created later are listed by get_index. Close typos are auto-corrected; "
            "a new name needs create_category=true."
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
    key: Annotated[
        Optional[str],
        Field(
            description="Optional sub-topic this belongs under, e.g. 'rotation' under config. "
            "File a fact under the key it is about when you know it — it groups the entry with "
            "that slot's history and keeps identical text under different keys from collapsing "
            "into one entry. Unlike a summary key this is NOT gated: the key does not need to "
            "have a summary yet, and coining a new one is fine. Omit it if the material does not "
            "clearly belong to one topic; a wrong key is worse than none."
        ),
    ] = None,
    create_category: Annotated[
        bool,
        Field(
            description="Set true ONLY to create a category that does not exist yet. It becomes "
            "available to every project, and search can filter on it. Without it an unknown "
            "category is refused and the existing ones come back. Most new topics are a key, not "
            "a category — create one only for a new KIND of entry you would want to search by."
        ),
    ] = False,
) -> dict:
    store = get_store()
    documents = [content] if isinstance(content, str) else list(content)

    try:
        result = store.save_chunks(
            documents=documents,
            category=category,
            project=project,
            tier=tier,
            key=key,
            create_category=create_category,
        )
    except UnknownCategory as refusal:
        # Returned, not raised, for the same reason as an unknown key: the caller
        # can only pick an existing category if it can see the list.
        return {
            "added": False,
            "refused": "unknown_category",
            "reason": str(refusal),
            "existing_categories": refusal.known,
            "hint": "Reuse whichever of these fits, or resend with create_category=true "
                    "if this really is a new kind of entry.",
        }

    for chunk_id, document in zip(result["ids"], documents):
        log_write(
            {
                "id": chunk_id,
                "project": result["project"],
                "category": result["category"],
                "key": result.get("key"),
                "corrected_from": result.get("corrected_from"),
                "tier": result.get("tier"),
                "content": document,
            }
        )

    # Name the destination in full. This used to return only {added, ids, count,
    # category}, so a caller could not tell from the response WHERE its write
    # landed — which is how a write aimed at one slot went somewhere else and
    # nobody noticed until the entry could not be found again.
    stored_at = f"{result['project']}/{result['category']}"
    if result.get("key"):
        stored_at += f"/{result['key']}"
    response: dict = {
        "added": True,
        "stored_at": stored_at,
        "ids": result["ids"],
        "count": result["count"],
        "category": result["category"],
    }
    if result.get("key"):
        response["key"] = result["key"]
    if result.get("category_created"):
        response["category_created"] = result["category"]
    if result["duplicates_collapsed"]:
        response["duplicates_collapsed"] = result["duplicates_collapsed"]
        response["duplicate_note"] = (
            "Identical text already stored under this project and category — "
            "content-addressed ids mean a repeat updates the existing entry "
            "rather than creating a second copy."
        )
    if result["oversized"]:
        # Said at write time, where splitting is still possible and free —
        # rather than discovered later as a silently truncated search result.
        response["oversized"] = result["oversized"]
        response["oversized_note"] = (
            f"{len(result['oversized'])} entry(ies) exceed {MAX_DOC_CHARS} characters and "
            "will come back truncated from search_context. Consider re-saving that "
            "material as several smaller, self-contained facts."
        )
    if result.get("corrected_from"):
        response["note"] = (
            f"'{result['corrected_from']}' isn't a valid category — saved under "
            f"'{result['category']}'."
        )
    return response
