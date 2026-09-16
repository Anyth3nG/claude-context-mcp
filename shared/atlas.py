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
  - Retired chunks are dropped: the store has been told they are wrong.
    Everything else that is not current — appended entries and archived
    versions alike — is ONE history, grouped under the key it belongs to.
    Until 2026-09-16 archived versions were left out of the history band and
    slots archived outright appeared nowhere but a count.
  - Nothing invents a project overview. A project is its slots.
"""
from __future__ import annotations

import json
from pathlib import Path

from shared.store import APPENDED_SOURCE, SUPERSEDED_SOURCE

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

    # Every history record, read in two scans and grouped by the key it belongs
    # to. Two scans rather than one slot_history call per key: the atlas wants
    # every key at once, and per-key queries are what once pushed /map past the
    # Lambda timeout.
    groups: dict[tuple, dict] = {}

    def group(project: str, cat: str, key):
        return groups.setdefault((project, cat, key), {
            "cat": cat, "key": key, "versions": [], "appended": [], "_events": {},
        })

    for r in store.records(type="chunk", source=APPENDED_SOURCE):
        doc = r.get("document") or ""
        g = group(r.get("project") or "general", r.get("category") or "note", r.get("key") or None)
        g["appended"].append({"id": r["id"], "at": (r.get("timestamp") or "")[:10],
                              "chars": len(doc), "text": doc})
    for r in store.records(source=SUPERSEDED_SOURCE):
        g = group(r.get("project") or "general", r.get("category") or "note", r.get("key") or None)
        # One archival event may be several pieces (see _split_for_archive);
        # they share a superseded_at stamp and stitch back into one version.
        g["_events"].setdefault(r.get("superseded_at") or "", []).append(r)

    for g in groups.values():
        for stamp, pieces in g["_events"].items():
            pieces.sort(key=lambda r: r.get("split_index") or 0)
            text = "\n\n".join(r.get("document") or "" for r in pieces)
            g["versions"].append({"at": stamp[:16].replace("T", " "), "chars": len(text),
                                  "text": text, "why": pieces[0].get("archived_reason")})
        del g["_events"]
        g["versions"].sort(key=lambda v: v["at"], reverse=True)
        g["appended"].sort(key=lambda a: a["at"], reverse=True)
        g["last"] = max([v["at"][:10] for v in g["versions"]] +
                        [a["at"] for a in g["appended"]] + [""])

    projects = []
    for name, meta in idx.items():
        entries = store.get_brief(name if name != "general" else None)
        live_keys = {(e["category"], e["key"]) for e in entries}
        slots = []
        for e in entries:
            label = e["category"] + (f"/{e['key']}" if e["key"] else "")
            g = groups.get((name, e["category"], e["key"]))
            versions = g["versions"] if g else []
            slots.append({"cat": e["category"], "key": e["key"], "label": label,
                          "chars": len(e["content"]), "text": e["content"],
                          "updated": (e["timestamp"] or "")[:10],
                          "prior": len(versions), "versions": versions})
        slots.sort(key=lambda x: (x["cat"], x["key"] or ""))

        history = []
        for (proj, cat, key), g in groups.items():
            if proj != name:
                continue
            # What the key IS today decides how its history reads: versions of
            # a slot still live, the whole record of a slot that was archived,
            # entries filed under a key that never had a slot, or entries with
            # no key at all — the ones still waiting to be sorted.
            kind = ("unsorted" if key is None else
                    "live" if (cat, key) in live_keys else
                    "archived" if g["versions"] else "appended")
            history.append({**g, "kind": kind})
        # Newest activity first within a kind; kinds in the order a reader
        # wants them, current slots' pasts before orphans; unsorted last.
        order = {"live": 0, "archived": 1, "appended": 2, "unsorted": 3}
        history.sort(key=lambda g: g["last"], reverse=True)
        history.sort(key=lambda g: (g["cat"], order[g["kind"]]))

        # Counted from what the page lists, not from the index: a version
        # archived in pieces is several records there and one version here.
        projects.append({"name": name, "tier": meta.get("tier", "general"), "slots": slots,
                         "chars": meta["brief_chars"],
                         "chunks": sum(len(g["versions"]) + len(g["appended"]) for g in history),
                         "archived": meta.get("archived_slots", 0),
                         "history": history})
    projects.sort(key=lambda p: -p["chars"])

    return {"projects": projects, "totals": {
        "projects": len(projects), "slots": sum(len(p["slots"]) for p in projects),
        "chunks": sum(p["chunks"] for p in projects),
        "chars": sum(p["chars"] for p in projects)}}
