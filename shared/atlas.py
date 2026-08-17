"""
The atlas data shape: the whole store arranged as projects -> categories ->
slots, with history kept separate.

This lives here rather than in scripts/build_atlas.py because two callers need
the SAME shape from the SAME code: the offline generator, which bakes it into a
static page, and the served /map route, which reads it fresh on every request.
When the transform lived only in the generator, the honesty rules below held
offline and nowhere else — see decisions/map-honesty in the store.

What this deliberately does NOT do is as load-bearing as what it does:

  - No links are computed. Only what the store explicitly asserts is drawn.
    A whole-store map with hand-written cross-project links stated a
    relationship that did not exist, and deriving links by name-matching
    produced one out of a sentence saying two projects should NOT be
    conflated. Similarity is never an edge.
  - Superseded and retired chunks are dropped. They belong to the slot that
    replaced them, not to the history band.
  - Nothing invents a project overview. A project is its slots.
"""
from __future__ import annotations

import json
from pathlib import Path

# The page template, with a __DATA__ placeholder where the payload goes. It is a
# file rather than a string literal so the served route and the offline
# generator render the SAME page — a template that lived inside the generator
# could only ever be reached by importing that script, which would run it.
PAGE_TEMPLATE = Path(__file__).resolve().parent / "atlas_page.html"


def render_page(data: dict) -> str:
    """
    Inline the payload into the page and return the whole self-contained
    document. No external assets, no fetch: the data is present before any
    script runs, which is what keeps the served page and the generated file
    the same artifact.
    """
    payload = json.dumps(data, ensure_ascii=False)
    # A stored document containing "</script>" would otherwise close the script
    # element early and the rest of the payload would land in the DOM as markup.
    # "\/" is a legal JSON string escape, so this only ever alters string
    # contents and the parsed data is unchanged. No stored entry contains the
    # sequence today; this is here so that stops being load-bearing.
    payload = payload.replace("</", r"<\/")
    return PAGE_TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", payload)


def build_atlas_data(store) -> dict:
    """
    Build the atlas payload from a ContextStore.

    Returns {"projects": [...], "totals": {...}} — JSON-serializable, with
    every stored document included in full. Callers decide what to do about
    that: the generator writes it outside this repo because the repo is
    public, and the route serves it only behind auth.
    """
    idx = store.index()["projects"]

    # The live history: point-in-time facts that no summary represents. They
    # carry a project and a category, which is enough to place them.
    history: dict[str, list] = {}
    for record in store.records(type="chunk", source="live"):
        # superseded and retired stay out; they belong to a slot
        doc = record.get("document") or ""
        history.setdefault(record.get("project") or "general", []).append({
            "id": record["id"], "cat": record.get("category") or "note", "text": doc,
            "at": (record.get("timestamp") or "")[:10], "chars": len(doc),
        })
    for v in history.values():
        v.sort(key=lambda c: (c["cat"], c["at"]))

    projects = []
    for name, meta in idx.items():
        entries = store.get_brief(name if name != "general" else None)
        slots = []
        for e in entries:
            label = e["category"] + (f"/{e['key']}" if e["key"] else "")
            prior = meta["summaries"].get(label, {}).get("prior_versions", 0)
            versions = []
            if prior:
                h = store.slot_history(name if name != "general" else None, e["category"], e["key"])
                versions = [{"at": (v["superseded_at"] or "")[:16].replace("T", " "),
                             "chars": v["chars"], "text": v["content"], "why": v["reason"]}
                            for v in h["versions"]]
            slots.append({"cat": e["category"], "key": e["key"], "label": label,
                          "chars": len(e["content"]), "text": e["content"],
                          "updated": (e["timestamp"] or "")[:10],
                          "prior": prior, "versions": versions})
        slots.sort(key=lambda x: (x["cat"], x["key"] or ""))
        projects.append({"name": name, "tier": meta.get("tier", "general"), "slots": slots,
                         "chars": meta["brief_chars"], "chunks": meta["history_chunks"],
                         "archived": meta.get("archived_slots", 0),
                         "history": history.get(name, [])})
    projects.sort(key=lambda p: -p["chars"])

    return {"projects": projects, "totals": {
        "projects": len(projects), "slots": sum(len(p["slots"]) for p in projects),
        "chunks": sum(p["chunks"] for p in projects),
        "chars": sum(p["chars"] for p in projects)}}
