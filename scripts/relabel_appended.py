#!/usr/bin/env python3
"""
Relabel every history chunk stored as source="live" to source="appended".

Until 2026-09-16 add_update wrote its chunks with the same source a current
summary carries, so "live" named two different things and the only way to tell
history from current state was the record type. Now "live" means a summary and
nothing else; every chunk is history, and its source says which kind. The code
already writes "appended" — this moves the rows written before it did.

Metadata only. Ids are content hashes that encode nothing about source, the
documents are untouched, and the vectors stay where they are, so this costs no
embedding call. Retired chunks keep the source they were retired from in
retired_from_source; that is a record of what WAS, and is left alone.

    python3 scripts/relabel_appended.py            # dry run
    python3 scripts/relabel_appended.py --apply

Run AFTER deploying the code that writes "appended": until then a new write
would land as "live" again and need a second pass.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from shared.dynamo_driver import DynamoDriver  # noqa: E402
from shared.store import APPENDED_SOURCE, LIVE_SOURCE, ContextStore, VoyageRestEmbedding  # noqa: E402

APPLY = "--apply" in sys.argv
TABLE = os.environ.get("DYNAMODB_TABLE", "context-mcp-store")
REGION = os.environ.get("AWS_REGION", "eu-west-1")

# Nothing here embeds, so the embedder is never called; the key is only
# required by the constructor.
store = ContextStore(driver=DynamoDriver(TABLE, VoyageRestEmbedding("unused"), region=REGION))

# The driver's own scan, not store.records(): update_metadata wants the
# metadata dict as stored, and records() folds id and document into it.
rows = store.driver.scan({"type": "chunk", "source": LIVE_SOURCE}, with_documents=False)
print(f"table {TABLE}: {len(rows)} chunks carry source={LIVE_SOURCE!r}")
if not rows:
    print("nothing to do.")
    sys.exit(0)

by_project: dict[str, int] = {}
for r in rows:
    p = r["metadata"].get("project") or "general"
    by_project[p] = by_project.get(p, 0) + 1
for p, n in sorted(by_project.items()):
    print(f"  {p:28} {n}")

if not APPLY:
    print(f"\nDRY RUN. Re-run with --apply to set source={APPENDED_SOURCE!r} on all {len(rows)}.")
    sys.exit(0)

done = 0
for r in rows:
    meta = dict(r["metadata"])
    meta["source"] = APPENDED_SOURCE
    store.driver.update_metadata(r["id"], meta)
    done += 1
    if done % 50 == 0:
        print(f"  {done}/{len(rows)}")
print(f"relabelled {done}.")

left = store.driver.scan({"type": "chunk", "source": LIVE_SOURCE}, with_documents=False)
print(f"verify: {len(left)} chunks still carry source={LIVE_SOURCE!r}" + ("" if not left else " — INVESTIGATE"))
