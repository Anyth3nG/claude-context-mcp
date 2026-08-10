#!/usr/bin/env python3
"""
Rename the `goal` category to `tasks` across the whole store.

Two different moves, because ids are built differently:

  - SUMMARIES carry the category in their id ('{project}-goal-summary'), which
    Chroma cannot rename in place. Each is re-added under its new id and the old
    one deleted. The EXISTING EMBEDDING is carried over rather than recomputed:
    the document text is untouched, so the old vector is still exactly right,
    and re-embedding would be a pointless Voyage bill.
  - EVERYTHING ELSE (append-only chunks, superseded snapshots) has a hash id
    that encodes nothing, so only the metadata changes.

    python3 scripts/migrate_goal_to_tasks.py          # dry run
    python3 scripts/migrate_goal_to_tasks.py --apply

Run this AFTER shared/store.py accepts 'tasks', and expect a window where the
deployed server disagrees with the data until the new code is live.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
from shared.store import ContextStore

APPLY = "--apply" in sys.argv

s = ContextStore()
raw = s.collection.get(where={"category": "goal"},
                       include=["documents", "metadatas", "embeddings"])
if not raw["ids"]:
    print("nothing with category=goal — already migrated, or nothing to do.")
    sys.exit(0)

renames, retags = [], []
for id_, doc, meta, emb in zip(raw["ids"], raw["documents"], raw["metadatas"], raw["embeddings"]):
    new_meta = dict(meta)
    new_meta["category"] = "tasks"
    if meta.get("type") == "summary" and "-goal-" in id_:
        renames.append((id_, id_.replace("-goal-", "-tasks-"), doc, new_meta, emb))
    else:
        retags.append((id_, new_meta))

print(f"summaries to re-id : {len(renames)}")
for old, new, *_ in renames:
    print(f"    {old}\n      -> {new}")
print(f"entries to re-tag  : {len(retags)}  (metadata only, ids unchanged)")

if not APPLY:
    print("\nDRY RUN. Re-run with --apply to perform the migration.")
    sys.exit(0)

for old, new, doc, meta, emb in renames:
    s.collection.add(ids=[new], documents=[doc], metadatas=[meta], embeddings=[emb])
    s.collection.delete(ids=[old])
    print(f"moved {old} -> {new}")

if retags:
    s.collection.update(ids=[i for i, _ in retags], metadatas=[m for _, m in retags])
    print(f"re-tagged {len(retags)} entries")

left = s.collection.get(where={"category": "goal"}, include=["metadatas"])
now = s.collection.get(where={"category": "tasks"}, include=["metadatas"])
print(f"\nremaining category=goal : {len(left['ids'])}  (expected 0)")
print(f"now category=tasks      : {len(now['ids'])}")
