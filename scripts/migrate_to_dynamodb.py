#!/usr/bin/env python3
"""
Copy the live store from Chroma Cloud into DynamoDB.

THIS IS NOT THE CUTOVER. Chroma stays authoritative and untouched; this only
populates DynamoDB so the two can be compared. Nothing reads the new table until
the deploy that switches drivers.

RE-RUNNABLE, and that matters: writes landing in Chroma after this runs will not
be in DynamoDB, so the two diverge from the moment it finishes. Items are written
with PutItem, which overwrites, so running it again immediately before cutover is
the way to close that gap.

VECTORS ARE CARRIED, NOT RECOMPUTED. All 448 records already hold voyage-3.5
embeddings, so re-embedding would cost 448 API calls to produce vectors that must
be identical anyway. The embedder passed to the driver is a tripwire that raises
if called — if this script completes, nothing was re-embedded, and that is proven
rather than assumed.

    python3 scripts/migrate_to_dynamodb.py            # dry run
    python3 scripts/migrate_to_dynamodb.py --apply
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import boto3
from botocore.exceptions import ClientError

from shared.dynamo_driver import DynamoDriver
from shared.smoke_dynamodb import create_table
from shared.store import ContextStore

REGION = "eu-west-1"
TABLE = "context-mcp-store"
DIMS = 1024  # voyage-3.5
APPLY = "--apply" in sys.argv

# Physical bookkeeping the driver adds on write and strips on read. Excluded from
# the metadata comparison because the source has no equivalent.
DRIVER_FIELDS = {"sk", "type_sk", "searchable", "chars"}


def refuse_to_embed(texts):
    raise AssertionError(
        f"the embedder was called for {len(texts)} text(s) — every record should "
        "have carried its existing vector. Something arrived without one; "
        "investigate rather than paying to regenerate it."
    )


def main() -> None:
    ddb = boto3.client("dynamodb", region_name=REGION)

    print("reading Chroma…")
    chroma = ContextStore()
    source = chroma.driver.scan({}, with_documents=True, with_embeddings=True)
    print(f"  {len(source)} records, Chroma count() reports {chroma.count()}")
    if len(source) != chroma.count():
        sys.exit(f"ABORT: read {len(source)} but count() says {chroma.count()}")

    bad = [r["id"] for r in source
           if r.get("embedding") is None or len(r["embedding"]) != DIMS]
    if bad:
        sys.exit(f"ABORT: {len(bad)} records lack a {DIMS}-dim vector: {bad[:5]}")
    print(f"  all carry a {DIMS}-dim vector")

    by_type: dict[str, int] = {}
    for r in source:
        t = r["metadata"].get("type", "?")
        by_type[t] = by_type.get(t, 0) + 1
    print(f"  by type: {by_type}")

    if not APPLY:
        print("\nDRY RUN — nothing written. Re-run with --apply.")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = ROOT / "rechunk_backups" / f"dynamodb-migration-{stamp}.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(
        {"table": TABLE, "count": len(source),
         "ids": sorted(r["id"] for r in source)}, indent=2), encoding="utf-8")
    print(f"\nmanifest: {manifest}")

    try:
        ddb.describe_table(TableName=TABLE)
        print(f"{TABLE} already exists — writing into it (PutItem overwrites)")
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise
        create_table(ddb, TABLE, DIMS)

    driver = DynamoDriver(TABLE, refuse_to_embed, region=REGION)
    print(f"\nwriting {len(source)} records…")
    BATCH = 25
    for start in range(0, len(source), BATCH):
        driver.put(source[start:start + BATCH])
        print(f"  {min(start + BATCH, len(source))}/{len(source)}", end="\r")
    print(f"  {len(source)}/{len(source)} written")

    # ---- verification --------------------------------------------------
    #
    # Global secondary indexes are eventually consistent, and scan() with no
    # project reaches records through by_type — so a read taken the instant a
    # bulk write finishes can disagree with itself. Observed once: 452 written,
    # 455 read back, while every id, document and metadata field still matched.
    # The data was right and the count was early. So converge before judging.
    print("\nverifying (waiting for the index to settle)…")
    landed = []
    for attempt in range(12):
        landed = driver.scan({}, with_documents=True)
        if len(landed) == len(source):
            break
        print(f"  index still settling: {len(landed)} vs {len(source)} expected, retrying…")
        time.sleep(5)
    src_by_id = {r["id"]: r for r in source}
    dst_by_id = {r["id"]: r for r in landed}

    problems: list[str] = []
    if len(landed) != len(source):
        problems.append(f"count: wrote {len(source)}, read back {len(landed)}")
    missing = sorted(set(src_by_id) - set(dst_by_id))
    extra = sorted(set(dst_by_id) - set(src_by_id))
    if missing:
        problems.append(f"{len(missing)} ids missing: {missing[:5]}")
    if extra:
        problems.append(f"{len(extra)} unexpected ids: {extra[:5]}")

    doc_diffs, meta_diffs = [], []
    for rid, src in src_by_id.items():
        dst = dst_by_id.get(rid)
        if dst is None:
            continue
        if src["document"] != dst["document"]:
            doc_diffs.append(rid)
        clean = {k: v for k, v in dst["metadata"].items() if k not in DRIVER_FIELDS}
        if src["metadata"] != clean:
            only_src = {k: v for k, v in src["metadata"].items() if clean.get(k) != v}
            only_dst = {k: v for k, v in clean.items() if src["metadata"].get(k) != v}
            meta_diffs.append((rid, only_src, only_dst))
    if doc_diffs:
        problems.append(f"{len(doc_diffs)} documents differ: {doc_diffs[:3]}")
    if meta_diffs:
        problems.append(f"{len(meta_diffs)} metadata differ, first: {meta_diffs[0]}")

    print(f"  ids            {len(dst_by_id)}/{len(src_by_id)} present")
    print(f"  documents      {len(src_by_id) - len(doc_diffs)}/{len(src_by_id)} identical")
    print(f"  metadata       {len(src_by_id) - len(meta_diffs)}/{len(src_by_id)} identical")

    if problems:
        print("\nVERIFICATION FAILED:")
        for p in problems:
            print(f"  - {p}")
        print(f"\nChroma is untouched and still authoritative. Delete the table and retry:\n"
              f"  aws dynamodb delete-table --table-name {TABLE} --region {REGION}")
        sys.exit(1)

    print("\nVERIFIED — every record present, byte-identical, metadata intact.")
    print("Chroma remains authoritative. Re-run this immediately before cutover to "
          "pick up anything written in between.")


if __name__ == "__main__":
    main()
