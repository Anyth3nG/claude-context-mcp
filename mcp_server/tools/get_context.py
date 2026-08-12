"""
Read tool: get_context. The whole project, one category, or one slot.

Replaces get_brief and get_value, which differed only in how much they returned:
both were deterministic lookups handing back whole documents, so two tools with
overlapping arguments taught the model a distinction that was never real. Depth
now comes from how much of the address is supplied.

get_index is deliberately NOT folded in as a further step up. It returns
metadata ABOUT documents while this returns the documents, and letting argument
count select between those two shapes would give one tool two incompatible
response types — see decisions/get-index-stays-separate.

This merge only became unambiguous once keyless slots were removed. While a
category could hold both a main slot and keyed slots, (project, category) meant
either "the category's own summary" or "everything filed under it", and there
was no way to say which — see decisions/no-keyless-slots.
"""
from __future__ import annotations

from typing import Annotated, Optional

from pydantic import Field

from mcp_server.context import get_store

DESCRIPTION = """Read stored context as whole documents — a project, one of its categories, or a single slot. Depth follows the address you give: each argument you add narrows the scope.

    get_context(project)                  every live summary for that project
    get_context(project, category)        just that category's slots
    get_context(project, category, key)   one slot

Prefer this over search_context whenever the question is about CURRENT STATE — "what is the stack", "how is this set up", "where does this project stand". search_context ranks by similarity and cuts each hit to ~800 characters, so a long entry is only ever partly visible; this is a direct lookup and returns text whole. Search is for history and reasoning: "what did we decide about X", "why did we choose Y".

Check get_index FIRST if you don't know what's stored. It costs a fraction of this and shows every slot with its size, so you can tell what a call here would cost before paying it — a whole project can run to thousands of tokens.

NARROW WHENEVER YOU CAN. Passing a category is the difference between a few hundred tokens and several thousand on a project with real depth: a question about auth wants architecture and decisions, not the roadmap.

ALSO use this before writing. patch_context needs text copied exactly from the stored summary, and change_update replaces a slot outright, so either way you need the current text in front of you first.

Every response says what else is there, so the next call is obvious without a separate lookup: a project-level call lists `categories` with slot counts, a category-level call lists the other `siblings` categories, and a slot-level call lists the other keys beside it. An empty result means nothing is stored at that address, not that the project is unknown."""


def get_context(
    project: Annotated[
        Optional[str],
        Field(
            description='Name of the project (e.g. "context-mcp"). Omit for general, '
            "non-project-specific entries."
        ),
    ] = None,
    category: Annotated[
        Optional[str],
        Field(
            description="Narrow to one category: tech_stack, architecture, config or decisions "
            "for project-scoped entries; preference, fact, tasks or note for general ones. Omit "
            "for the whole project. Close typos are auto-corrected."
        ),
    ] = None,
    key: Annotated[
        Optional[str],
        Field(
            description="Narrow to ONE slot within the category, e.g. 'cognito' or 'deploy' "
            "under config. Requires `category`. Omit to get every slot in the category; "
            "get_index lists the keys in use."
        ),
    ] = None,
) -> dict:
    store = get_store()
    project_label = project or "general"

    if key and not category:
        # Refused rather than guessed: a key is only unique within its category,
        # and scanning every category for one would silently pick a winner.
        return {
            "found": False,
            "error": "key requires category — a key is only unique within one category.",
            "project": project_label,
            "key": key,
            "hint": "Resend with the category, or call get_index to see where that key lives.",
        }

    if key:
        return _one_slot(store, project, project_label, category, key)
    return _many(store, project, project_label, category)


def _category_counts(store, project: Optional[str]) -> dict[str, int]:
    """How many slots each category holds, from metadata only — no documents."""
    entry = store.index(project=project)["projects"].get(project or "general", {})
    counts: dict[str, int] = {}
    for label in entry.get("summaries", {}):
        cat = label.split("/", 1)[0]
        counts[cat] = counts.get(cat, 0) + 1
    return counts


def _one_slot(store, project, project_label, category, key) -> dict:
    found = store.get_summary(project=project, category=category, key=key)
    siblings = [k for k in store.summary_keys(project, category) if k and k != key]

    if found is None:
        return {
            "found": False,
            "project": project_label,
            "category": category,
            "key": key,
            "siblings": siblings,
            "note": (
                f"Nothing stored at {category}/{key}. "
                + (
                    f"Other keys in this category: {', '.join(siblings)}."
                    if siblings
                    else "This category has no slots yet."
                )
                + " change_update will create it; `key` is required when it does."
            ),
        }

    document, metadata, _ = found
    return {
        "found": True,
        "content": document,
        "project": metadata.get("project"),
        "category": metadata.get("category"),
        "key": metadata.get("key"),
        "tier": metadata.get("tier"),
        "source": metadata.get("source"),
        "timestamp": metadata.get("timestamp"),
        "siblings": siblings,
        **(
            {"next": f"get_context('{project_label}', '{category}') for all {len(siblings) + 1} slots here"}
            if siblings
            else {}
        ),
    }


def _many(store, project, project_label, category) -> dict:
    entries = store.get_brief(project=project, category=category)
    corrected = entries[0].get("category_corrected_from") if entries else None
    resolved = entries[0]["category"] if entries else category

    response: dict = {
        "project": project_label,
        **({"category": resolved} if category else {}),
        "entries": entries,
        "count": len(entries),
    }

    counts = _category_counts(store, project)
    if category:
        # At category level, siblings are the OTHER categories — the useful next
        # step. At project level everything is already in hand, so a sibling map
        # would only restate the response.
        response["siblings"] = {c: n for c, n in sorted(counts.items()) if c != resolved}
    else:
        response["categories"] = dict(sorted(counts.items()))

    if corrected:
        response["note"] = (
            f"'{corrected}' isn't a valid category — interpreted as '{resolved}'."
        )
    elif not entries:
        # A category filter matching nothing means something quite different from
        # an empty project, and saying the wrong one sends the caller off to
        # recreate context that already exists.
        response["note"] = (
            (
                f"No '{category}' summaries for this project. It does have: "
                f"{', '.join(sorted(counts)) or 'nothing yet'}."
            )
            if category
            else "No summaries recorded for this project yet. Try search_context for "
            "history, or record current state with change_update."
        )
    return response
