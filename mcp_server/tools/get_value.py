"""
Read tool: get_value. One summary slot, fetched by id, returned whole.

Same idea as get_brief but narrower — use it when you already know which
category you want, or when you are about to overwrite a slot with
change_update and need the current text in order to merge rather than clobber.
"""
from __future__ import annotations
from typing import Annotated, Optional

from pydantic import Field

from mcp_server.context import get_store

DESCRIPTION = """Get the current value of ONE category for a project — the tech stack, the architecture, the config — as a whole document.

Use this when you know exactly which category you need, rather than pulling the whole brief. It is a direct lookup, so the content comes back complete and no query has to be guessed at.

ALSO use this before writing. patch_context needs text copied exactly from the stored summary in order to match, and change_update replaces a category's entire contents, so either way you need the current text in front of you first.

Returns the full text plus metadata, or a not-found result if that category has no summary yet — in which case change_update will create it."""


def get_value(
    category: Annotated[
        str,
        Field(
            description="Which category to fetch: tech_stack, architecture, config or decisions "
            "for project-scoped entries; preference, fact, goal or note for general ones. "
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
    found = store.get_summary(project=project, category=category, key=key)
    if found is None:
        return {
            "found": False,
            "project": project or "general",
            "category": category,
            "key": key,
            "note": "No summary recorded for this category yet. change_update will create it.",
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
    }
