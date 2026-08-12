#!/usr/bin/env python3
"""
Delete one orphaned record by id.

An orphan here is a record whose id prefix disagrees with its `type` metadata —
a `chunk-` id carrying type="summary". It is unreachable by every MCP write tool:
retire_chunk refuses it because the metadata says summary, while archive_slot and
change_update address the slot by its deterministic id and so never touch this one.
It still carries source="live", so get_brief keeps returning it. See
note/store-id-mismatch in the store.

This deletes outright rather than flipping source to "superseded". That loses the
record that the text was once believed, which is the usual reason to prefer the
flip — chosen deliberately here.

    python3 scripts/delete_orphan_summary.py <id>            # dry run
    python3 scripts/delete_orphan_summary.py <id> --apply
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
from shared.store import ContextStore

args = [a for a in sys.argv[1:] if not a.startswith("--")]
if len(args) != 1:
    print(__doc__)
    sys.exit(2)

TARGET = args[0]
APPLY = "--apply" in sys.argv

s = ContextStore()
got = s.collection.get(ids=[TARGET], include=["documents", "metadatas"])
if not got["ids"]:
    print(f"no record with id {TARGET} — already gone, or the id is wrong.")
    sys.exit(0)

doc, meta = got["documents"][0], got["metadatas"][0]

# Written before the delete, not after: the point of a backup is to exist when
# the thing it backs up does not.
backup = ROOT / f"deleted-{TARGET}.json"
backup.write_text(json.dumps(
    {"id": TARGET, "document": doc, "metadata": meta,
     "deleted_at": datetime.now(timezone.utc).isoformat()},
    indent=2, ensure_ascii=False,
))

print(f"id       : {TARGET}")
print(f"type     : {meta.get('type')}   source: {meta.get('source')}")
print(f"slot     : {meta.get('project')}/{meta.get('category')}/{meta.get('key')}")
print(f"backup   : {backup}")
print(f"document :\n{doc}\n")

if not APPLY:
    print("DRY RUN — nothing deleted. Re-run with --apply.")
    sys.exit(0)

s.collection.delete(ids=[TARGET])

still = s.collection.get(ids=[TARGET], include=["metadatas"])
print("deleted." if not still["ids"] else "DELETE DID NOT TAKE — record still present.")
