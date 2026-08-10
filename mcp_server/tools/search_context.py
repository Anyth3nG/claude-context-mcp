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

For CURRENT STATE, use get_brief or get_value instead — "what is the stack", "how is this set up", "what does this project do". This tool ranks by similarity and truncates each hit to ~800 characters, so a long entry is only ever partly visible; get_brief and get_value are direct lookups and return whole documents. Use this tool for questions about history and reasoning: "what did we decide about X", "why did we choose Y", "when did Z change".

Do NOT use this for general knowledge questions, for things already stated earlier in this same conversation, or as a substitute for reasoning about what's already in front of you — only to retrieve durable context saved in a previous session.

Returns up to `top_k` matching entries, each with its `id`, its text (truncated to ~800 characters) and metadata (project, category, type, source, timestamp). Keep the id if you may need to act on that specific entry — retire_chunk takes one.

Excluded by default: superseded versions of summaries, and chunks retired as incorrect. Those are separate flags because they mean different things — a superseded summary was true when written, a retired chunk was wrong.

If a result contradicts a current summary, it is a candidate for retire_chunk: chunks are append-only, so a fact that was later disproved keeps ranking on the same queries as the entry that corrected it."""


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
    include_superseded: Annotated[
        bool,
        Field(
            description="Include archived earlier versions of summaries. Off by default, because "
            "stale values otherwise resurface alongside current ones. Turn on only when asking "
            "what something USED to be. This does NOT bring back chunks retired as wrong — "
            "see include_retired."
        ),
    ] = False,
    include_retired: Annotated[
        bool,
        Field(
            description="Include chunks retired as INCORRECT. Off by default and rarely worth "
            "turning on: these are facts the store has been told are wrong, and each carries a "
            "retired_reason. Use only when auditing what was once believed, never to answer a "
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
        include_superseded=include_superseded,
        include_retired=include_retired,
    )

    docs = raw["documents"][0]
    metas = raw["metadatas"][0]
    # The id is carried through deliberately. It is the only handle a caller has
    # on a specific entry, and retire_chunk needs one — without it a reader can
    # see that a chunk is wrong and have no way to say which chunk it meant.
    ids = raw.get("ids", [[]])[0]
    results = [
        {"id": cid, "content": doc, **meta}
        for cid, doc, meta in zip(ids, docs, metas)
    ]

    response: dict = {"results": results, "count": len(results)}
    corrected_from = raw.get("category_corrected_from")
    if corrected_from:
        resolved = results[0]["category"] if results else None
        if resolved:
            response["note"] = f"'{corrected_from}' isn't a valid category — interpreted as '{resolved}'."
        else:
            response["note"] = f"'{corrected_from}' isn't a valid category and no close match was found."
    return response
