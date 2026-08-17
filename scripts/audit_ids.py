#!/usr/bin/env python3
"""
Audit every record for the id/metadata disagreement behind tasks/id-mismatch-audit.

THE BUG, 2026-08-07. An entry existed whose metadata said type=summary while its
id was not the id summary_id() computes for its own project/category/key. index()
reads metadata, so it listed the slot; get_summary() does a direct lookup by
computed id, so it returned not_found for the same slot. The index and a direct
read disagreed about something that was visibly there. One instance was repaired;
the write path that produced it was never found, and nothing has checked whether
other records are affected the same way.

READ ONLY. Writes nothing, so it is safe to run against the live store at any time.

Checks, each independently reportable:
  1. SUMMARY ID MISMATCH — type=summary whose id != summary_id(project, category, key).
     This is the bug itself.
  2. UNREACHABLE SLOT — the observable symptom: a summary the index lists that
     get_summary() cannot retrieve. Catches case 1 from the other direction, plus
     anything reachable-by-metadata but not by lookup for a reason not yet imagined.
  3. SPLIT-BRAIN SLOT — two summaries claiming the same project/category/key under
     different ids. The state a repeated bad write would produce.
  4. KEYLESS SUMMARY — no key at all. decisions/no-keyless-slots made these
     unrepresentable; any survivor predates the migration.
  5. STRANGE CHUNK ID — type=chunk whose id starts with neither chunk- nor
     superseded-. Note a chunk carrying a `key` is NOT a fault: chunk keys became
     legitimate with tasks/chunk-schema.

    python3 scripts/audit_ids.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from shared.store import ContextStore

VALID_CHUNK_PREFIXES = ("chunk-", "superseded-")

# Chroma Cloud refuses a Get asking for more than this and silently returns at
# most this many when asked for everything. The store passed 300 records some
# time ago, so an unpaginated get() now truncates without raising — which is
# exactly how an audit reports "all clear" on two thirds of the data.
GET_PAGE = 300


def fetch_all(collection, **kwargs) -> dict:
    """Every record, page by page, because one get() cannot return them all."""
    ids, metas = [], []
    offset = 0
    while True:
        page = collection.get(limit=GET_PAGE, offset=offset, **kwargs)
        if not page["ids"]:
            break
        ids.extend(page["ids"])
        metas.extend(page.get("metadatas") or [])
        if len(page["ids"]) < GET_PAGE:
            break
        offset += GET_PAGE
    return {"ids": ids, "metadatas": metas}


def main() -> None:
    store = ContextStore()
    expected_total = store.collection.count()
    everything = fetch_all(store.collection, include=["metadatas"])
    ids, metas = everything["ids"], everything["metadatas"]
    print(f"scanning {len(ids)} records (collection.count() reports {expected_total})")
    if len(ids) != expected_total:
        print(f"  WARNING: fetched {len(ids)} but count() says {expected_total} — "
              "pagination may still be incomplete")
    print()

    summary_id_mismatch = []
    unreachable = []
    keyless = []
    strange_chunk_id = []
    by_slot = defaultdict(list)

    summaries = 0
    for rid, meta in zip(ids, metas):
        rtype = meta.get("type")
        project, category, key = meta.get("project"), meta.get("category"), meta.get("key")

        if rtype == "summary":
            summaries += 1
            expected = store.summary_id(project, category, key)
            if rid != expected:
                summary_id_mismatch.append((rid, expected, project, category, key))
            if not key:
                keyless.append((rid, project, category))
            by_slot[(project, category, key)].append(rid)
        elif rtype == "chunk":
            if not rid.startswith(VALID_CHUNK_PREFIXES):
                strange_chunk_id.append((rid, project, category, key))
        else:
            strange_chunk_id.append((rid, f"type={rtype!r}", category, key))

    # The symptom, checked independently of check 1 rather than inferred from it:
    # walk what the index advertises and try to actually fetch each one.
    idx = store.index()
    listed = 0
    for pname, pdata in idx["projects"].items():
        for label in pdata["summaries"]:
            listed += 1
            category, _, key = label.partition("/")
            project = None if pname == "general" else pname
            if store.get_summary(project, category, key=key or None) is None:
                unreachable.append(f"{pname}/{label}")

    split_brain = {slot: rids for slot, rids in by_slot.items() if len(rids) > 1}

    def report(title: str, rows: list, fmt=str) -> bool:
        if not rows:
            print(f"  PASS  {title}")
            return True
        print(f"  FAIL  {title} — {len(rows)}")
        for row in rows:
            print(f"          {fmt(row)}")
        return False

    print(f"summaries by metadata: {summaries}   listed by index: {listed}\n")

    clean = all([
        report("1. summary ids match summary_id()", summary_id_mismatch,
               lambda r: f"{r[0]!r} should be {r[1]!r}  ({r[2]}/{r[3]}/{r[4]})"),
        report("2. every indexed slot is retrievable", unreachable),
        report("3. no slot claimed by two ids", list(split_brain.items()),
               lambda r: f"{r[0]} -> {r[1]}"),
        report("4. no keyless summaries", keyless,
               lambda r: f"{r[0]!r}  ({r[1]}/{r[2]})"),
        report("5. chunk ids use a known prefix", strange_chunk_id,
               lambda r: f"{r[0]!r}  ({r[1]}/{r[2]}, key={r[3]})"),
    ])

    if summaries != listed:
        clean = False
        print(f"\n  FAIL  metadata counts {summaries} summaries but the index lists "
              f"{listed} — these must agree")

    print("\nCLEAN — no id/metadata disagreement found." if clean
          else "\nISSUES FOUND — see above.")
    sys.exit(0 if clean else 1)


if __name__ == "__main__":
    main()
