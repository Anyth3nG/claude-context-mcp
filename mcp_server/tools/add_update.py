"""
Write tool: add_update. Appends chunks — the "diary" half of the store.

Pairs with patch_context / change_update, which edit a living summary slot
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

from shared.store import MAX_DOC_CHARS

from mcp_server.context import get_store
from mcp_server.review_log import log_write

DESCRIPTION = """Append a durable, decision-worthy fact from this conversation so it's retrievable in future sessions — on ANY machine or client (claude.ai or Claude Code), not just this one. This should be the exception, not routine; most of a conversation is not worth saving.

This is the canonical cross-machine, cross-client store — NOT any local, single-machine memory. Local memory doesn't follow the user to another device or to claude.ai. So still call this even if the content feels "already saved" locally; skipping it for that reason is exactly the failure this tool prevents.

SPLIT AS YOU WRITE. `content` accepts a LIST, and several focused entries cost the same as one long one — the whole list is embedded in a single call. Retrieval truncates long entries to ~800 characters, so anything longer is only ever partly visible no matter how it's searched for. One self-contained fact per list item; never pad a short fact to fill one.

Use add_update for point-in-time facts that ACCUMULATE: a decision made, an event, a discovery, something the user stated. Nothing is overwritten, so history builds up.

Use patch_context INSTEAD when revising the CURRENT STATE of something that already has a value. Appending a corrected version here leaves the outdated one in place, and future sessions must then read both and guess which still holds.

Save when: a real decision gets made, the user states a lasting preference or fact about themselves, or sets or updates a task.

Do NOT save: hypotheticals or options weighed but not chosen, restatements of something already in THIS store (check with search_context), transient debugging detail, or anything you're unsure is worth surfacing again. When in doubt, don't — a missed save is cheap to redo; a bad save pollutes retrieval permanently."""


def add_update(
    content: Annotated[
        Union[str, list[str]],
        Field(
            description="The durable fact, written to stand alone without the surrounding "
            "conversation. Pass a LIST to record several facts at once — preferred whenever "
            "the material covers more than one thing, or would otherwise run past ~800 "
            "characters and be truncated on retrieval."
        ),
    ],
    category: Annotated[
        str,
        Field(
            description="tech_stack, architecture, config, or decisions for project-scoped entries; "
            "preference, fact, tasks, or note for general ones. Close typos are auto-corrected."
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
    documents = [content] if isinstance(content, str) else list(content)

    result = store.save_chunks(
        documents=documents,
        category=category,
        project=project,
        tier=tier,
        source="live",
    )

    for chunk_id, document in zip(result["ids"], documents):
        log_write(
            {
                "id": chunk_id,
                "project": result["project"],
                "category": result["category"],
                "corrected_from": result.get("corrected_from"),
                "tier": result.get("tier"),
                "content": document,
            }
        )

    response: dict = {
        "added": True,
        "ids": result["ids"],
        "count": result["count"],
        "category": result["category"],
    }
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
