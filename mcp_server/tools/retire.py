"""
Write tool: retire. Takes something out of the default read, in the two ways
that can be meant — a slot whose work is FINISHED, or a chunk that is WRONG.

These were archive_slot and retire_chunk. Merging them is only safe because the
distinction survives the merge, and keeping it visible is most of this file.
Archiving says "this was true and is now history"; retiring says "this was never
true". Collapsing them into one generic "remove" would lose the difference the
store is built on — superseded stays searchable as history, retired is held back
because handing it to a reader answers a question with something known to be
false.

Dispatch is on the shape of the target, not on a mode argument: an `id` names a
chunk, a `category` (+key) names a slot. One signal, no combination to get wrong
— the same reason get_context dispatches on how much of an address it is given.
"""
from __future__ import annotations
from typing import Annotated, Optional

from pydantic import Field

from shared.store import ChunkNotFound, PatchSlotMissing

from mcp_server.context import get_store
from mcp_server.review_log import log_write

DESCRIPTION = """Take an entry out of the default read, in one of two senses. Which one you get depends on what you point at, and they are NOT interchangeable.

FINISHED WORK — pass `category` (+`key`, +`project`) to retire a whole summary slot. Use when a slot has stopped describing current state and become a record of what happened: a completed task, a config surface that no longer exists. The text is archived, stays searchable as history, and remains reachable through get_history. `tasks` is meant to answer "what needs doing next", so leaving finished work live makes every future session pay for it on every read.

WRONG FACT — pass `id` (a chunk id from search_context) to mark a chunk INCORRECT. Use when a chunk contradicts something that is still true: chunks are append-only, so a fact later disproved keeps ranking on the same queries as whatever corrected it, and a reader has no way to tell which one holds. Retired chunks are excluded from search by default, because handing one back answers a question with material known to be false.

DO NOT use the id form on something merely old but still accurate — age is not error, and retiring correct history destroys the reasoning trail. DO NOT use the slot form on a slot that is still current; archiving something in use makes it invisible to get_context and a future session will not know to ask for it. A slot that is merely long or out of date wants patch_context, not this.

`reason` is required either way, and is stored on the archived copy. Write it for someone who was not here: what completed the work, or what proved the fact wrong."""


def retire(
    reason: Annotated[
        str,
        Field(
            description="Why this is being retired, in one sentence, written for someone who "
            "never saw the conversation. For finished work: what completed it. For a wrong "
            "fact: what disproved it. Stored on the archived copy — an entry retired with no "
            "stated reason is worse than one left alone."
        ),
    ],
    id: Annotated[
        Optional[str],
        Field(
            description="WRONG-FACT MODE. A chunk id copied from a search_context result "
            '(e.g. "chunk-4c504f2f75588c82"). Do not construct one by hand. Marks that chunk '
            "INCORRECT. Omit this to retire a finished slot instead."
        ),
    ] = None,
    category: Annotated[
        Optional[str],
        Field(
            description="FINISHED-WORK MODE. Which category the slot lives in: tech_stack, "
            "architecture, config, decisions for project-scoped entries; preference, fact, "
            "tasks, note for general ones."
        ),
    ] = None,
    key: Annotated[
        Optional[str],
        Field(
            description="FINISHED-WORK MODE. Sub-topic within the category, e.g. 'phase-x' "
            "under tasks. Every summary lives under a key; get_index lists those in use."
        ),
    ] = None,
    project: Annotated[
        Optional[str],
        Field(description="Name of the project. Omit for general (non-project-specific) entries."),
    ] = None,
    superseded_by: Annotated[
        Optional[str],
        Field(
            description='WRONG-FACT MODE, optional. Where the correct version now lives, as '
            '"category/key" (e.g. "config/rotation"). Turns an isolated "this was wrong" into '
            "a pointer at the right answer."
        ),
    ] = None,
) -> dict:
    store = get_store()

    if not reason or not reason.strip():
        return {
            "retired": False,
            "refused": "missing_reason",
            "reason": "A reason is required. An entry retired with no stated reason leaves a "
            "later reader unable to tell whether it was finished or disproved.",
        }
    if id is not None and category is not None:
        return {
            "retired": False,
            "refused": "ambiguous_target",
            "reason": "Pass `id` to mark a chunk WRONG, or `category` (+key) to retire a "
            "FINISHED slot — not both. These mean different things and are refused rather "
            "than guessed at.",
        }
    if id is None and category is None:
        return {
            "retired": False,
            "refused": "no_target",
            "reason": "Give either `id` (a chunk id from search_context, marks it incorrect) "
            "or `category` (+key, retires a finished slot).",
        }

    if id is not None:
        return _retire_chunk(store, id, reason, superseded_by)
    return _archive_slot(store, project, category, key, reason)


def _retire_chunk(store, chunk_id: str, reason: str, superseded_by: Optional[str]) -> dict:
    """Mark a chunk WRONG. Metadata-only — the text and its vector are untouched."""
    try:
        result = store.retire_chunk(chunk_id=chunk_id, reason=reason, superseded_by=superseded_by)
    except ChunkNotFound as exc:
        return {"retired": False, "refused": "not_found", "reason": str(exc), "id": chunk_id}
    except ValueError as exc:
        # Most often: the id names a summary, not a chunk. A summary's value is
        # replaced through patch_context, which archives the old text properly;
        # flagging a live slot as wrong would leave the category with no value
        # at all and nothing pointing at what replaced it.
        return {"retired": False, "refused": "not_a_chunk", "reason": str(exc), "id": chunk_id}

    if result.get("retired"):
        log_write(
            {
                "tool": "retire",
                "action": "retired_incorrect",
                "id": chunk_id,
                "project": result.get("project"),
                "category": result.get("category"),
                "reason": result.get("reason"),
                "superseded_by": superseded_by,
            }
        )
    # `action` is spelled out rather than left implicit in which fields came
    # back: the whole risk of merging these two is a caller believing it marked
    # something wrong when it archived finished work, or the reverse.
    result["action"] = "retired_incorrect"
    result["meaning"] = "This chunk is marked WRONG and is excluded from search by default."
    return result


def _archive_slot(store, project, category, key, reason: str) -> dict:
    """Retire a FINISHED slot. Text is archived and stays visible as history."""
    try:
        result = store.archive_slot(
            project=project, category=category, key=key, reason=reason
        )
    except PatchSlotMissing as exc:
        return {
            "retired": False,
            "refused": "no_such_slot",
            "reason": str(exc),
            "hint": "Check get_index for the slots this project actually has.",
        }
    except ValueError as exc:
        return {"retired": False, "refused": "invalid_argument", "reason": str(exc)}

    log_write(
        {
            "tool": "retire",
            "action": "archived_finished",
            "id": result["id"],
            "project": result.get("project"),
            "category": result.get("category"),
            "key": result.get("key"),
            "reason": result.get("reason"),
        }
    )
    result["action"] = "archived_finished"
    result["meaning"] = (
        "This slot's work is FINISHED. The text is archived, stays searchable as history, "
        "and is reachable through get_history."
    )
    return result
