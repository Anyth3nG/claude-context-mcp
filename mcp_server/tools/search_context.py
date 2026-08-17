"""
Read tool: search_context. Wraps ContextStore.search(), shaping the return
into documents + metadata rather than exposing raw ChromaDB query internals
(ids, distances, nested per-query result lists).
"""
from __future__ import annotations
from typing import Annotated, Optional

from pydantic import Field

from mcp_server.context import get_store

DESCRIPTION = """Search the HISTORY of prior context — past decisions, architecture, tech stack, preferences, facts, or tasks — that isn't already visible in this conversation. This store is cross-machine and cross-client (claude.ai and Claude Code both write to it), so it can hold things captured on a different device or the other client that no local, single-machine memory would know about.

Use this when the user references something from before ("last time we...", "what did we decide about...", "remind me how X works", "what's my usual approach to..."), when you start working on a project you don't already have context for, or when project-specific facts (tech stack, architecture, config, past decisions) or general facts/preferences about the user would materially improve your answer. Prefer checking here over assuming local memory has the full picture — it may not, especially for anything set up on another machine or in claude.ai.

For CURRENT STATE, use get_context instead — "what is the stack", "how is this set up", "what does this project do". This tool ranks by similarity and truncates each hit to ~1000 characters, so a long entry is only ever partly visible; get_context is a direct lookup and returns whole documents. Use this tool for questions about history and reasoning: "what did we decide about X", "why did we choose Y", "when did Z change".

Do NOT use this for general knowledge questions, for things already stated earlier in this same conversation, or as a substitute for reasoning about what's already in front of you — only to retrieve durable context saved in a previous session.

Returns up to `top_k` matching entries, each with its `id`, its text (truncated to ~1000 characters) and metadata (project, category, type, source, timestamp). Keep the id if you may need to act on that specific entry — archive takes one.

A long archived version arrives as several pieces rather than one clipped hit, each carrying a `split_note` saying which part it is. A piece is complete in itself but partial as a version — use get_history on that slot to read the whole thing.

Results include earlier, archived versions of summaries alongside ordinary history chunks — both are history, and an archived version is often exactly what a "what did this used to say" question wants. Each carries `superseded_from` naming the slot it came from, so you can tell a past value from a standalone note.

Excluded by default: only chunks retired as INCORRECT. That is a narrower exclusion than it sounds — superseded means "was true when written", retired means "wrong", and just those are held back.

If a result contradicts a current summary, it is a candidate for archive: chunks are append-only, so a fact that was later disproved keeps ranking on the same queries as the entry that corrected it."""


def search_context(
    query: Annotated[str, Field(description="What to search for, in natural language.")],
    project: Annotated[
        Optional[str],
        Field(
            description='Name of the project to scope the search to (e.g. "ticketing-saas"). '
            "Omit to search everything, including general (non-project) entries."
        ),
    ] = None,
    category: Annotated[
        Optional[str],
        Field(
            description="One of tech_stack, architecture, config, decisions (project-scoped) "
            "or preference, fact, tasks, note (general). Omit to search all categories. "
            "Close typos are auto-corrected."
        ),
    ] = None,
    top_k: Annotated[
        int,
        Field(description="How many entries to return (1-10, default 5).", ge=1, le=10),
    ] = 5,
    include_retired: Annotated[
        bool,
        Field(
            description="Include chunks retired as INCORRECT. Off by default and rarely worth "
            "turning on: these are facts the store has been told are wrong, and each carries an "
            "archived_reason. Use only when auditing what was once believed, never to answer a "
            "question about how something works."
        ),
    ] = False,
) -> dict:
    store = get_store()
    raw = store.search(
        query=query,
        project=project,
        category=category,
        top_k=top_k,
        include_retired=include_retired,
    )

    docs = raw["documents"][0]
    metas = raw["metadatas"][0]
    # The id is carried through deliberately. It is the only handle a caller has
    # on a specific entry, and archive needs one — without it a reader can
    # see that a chunk is wrong and have no way to say which chunk it meant.
    ids = raw.get("ids", [[]])[0]
    results = []
    for cid, doc, meta in zip(ids, docs, metas):
        entry = {"id": cid, "content": doc, **meta}
        # A split piece is sized to fit UNDER the truncation cap, so it never
        # gets the " …[truncated]" marker that signals an incomplete document.
        # It reads as a whole, self-contained entry when it is actually one
        # section of a longer version — the one case where a result looks
        # complete and isn't. Say so explicitly, or nothing will.
        if meta.get("split_count"):
            slot = meta.get("category")
            if meta.get("key"):
                slot = f"{slot}/{meta['key']}"
            entry["split_note"] = (
                f"Part {(meta.get('split_index') or 0) + 1} of {meta['split_count']} of one "
                f"archived version of {slot}. This text is complete in itself but partial as a "
                f"version — get_history on that slot returns every piece stitched back together."
            )
        results.append(entry)

    response: dict = {"results": results, "count": len(results)}
    corrected_from = raw.get("category_corrected_from")
    if corrected_from:
        resolved = results[0]["category"] if results else None
        if resolved:
            response["note"] = f"'{corrected_from}' isn't a valid category — interpreted as '{resolved}'."
        else:
            response["note"] = f"'{corrected_from}' isn't a valid category and no close match was found."
    return response
