"""
Write tool: retire_chunk. Marks a history chunk as wrong.

The gap this closes: chunks are append-only, so a fact recorded in good faith
and later disproved stays semantically searchable forever. It ranks on the same
queries as the entry that corrected it, and nothing in the document says it was
contradicted — leaving a reader two plausible answers and no way to choose.

Deliberately narrow. It cannot touch summaries, whose current value is replaced
through patch_context or change_update, and it does not delete: the text stays,
carrying the reason it stopped being true.
"""
from __future__ import annotations
from typing import Annotated, Optional

from pydantic import Field

from shared.store import ChunkNotFound

from mcp_server.context import get_store
from mcp_server.review_log import log_write

DESCRIPTION = """Mark a history chunk as INCORRECT so it stops competing with entries that are still true.

Use this when search_context returns a chunk that contradicts a current summary — the chunk is append-only history, so it keeps ranking on the same queries as whatever corrected it, and a future reader has no way to tell which of the two still holds.

Pass the `id` exactly as search_context returned it. The chunk is not deleted: its text stays, tagged with your `reason` and hidden from search by default. That preserves the record that something was believed and when it stopped being true, which is usually worth more than the space it takes.

Set `superseded_by` to the slot that now holds the truth (e.g. "config/rotation") whenever there is one. It turns an isolated "this was wrong" into a pointer at the right answer.

DO use this for: a chunk stating a fact later disproved, a measurement invalidated by a change, guidance that no longer applies.

Do NOT use this for: material that is merely old but still accurate — age is not error, and retiring correct history quietly destroys the reasoning trail. Do not use it to edit a summary either; that is patch_context or change_update, which archive the previous text properly. Refused on anything that is not a chunk.

Irreversible through this tool — there is no un-retire. When unsure whether a chunk is wrong or just superseded in emphasis, leave it."""


def retire_chunk(
    id: Annotated[
        str,
        Field(
            description="The chunk's id, copied from a search_context result "
            '(e.g. "chunk-4c504f2f75588c82"). Do not construct one by hand.'
        ),
    ],
    reason: Annotated[
        str,
        Field(
            description="Why this is wrong, in one sentence, written so it makes sense to "
            "someone who never saw the conversation that retired it. Required — a retired "
            "chunk with no stated reason is worse than one left alone."
        ),
    ],
    superseded_by: Annotated[
        Optional[str],
        Field(
            description='Where the correct version now lives, as "category/key" or a short '
            'phrase (e.g. "config/rotation"). Omit only when nothing replaced it.'
        ),
    ] = None,
) -> dict:
    store = get_store()
    try:
        result = store.retire_chunk(chunk_id=id, reason=reason, superseded_by=superseded_by)
    except ChunkNotFound as exc:
        return {"retired": False, "error": str(exc), "id": id}
    except ValueError as exc:
        return {"retired": False, "error": str(exc), "id": id}

    if result.get("retired"):
        log_write(
            {
                "tool": "retire_chunk",
                "id": id,
                "project": result.get("project"),
                "category": result.get("category"),
                "reason": result.get("reason"),
                "superseded_by": superseded_by,
            }
        )
    return result
