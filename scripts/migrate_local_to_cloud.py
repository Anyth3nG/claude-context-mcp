"""
One-off migration: local on-disk ChromaDB -> Chroma Cloud.

Copies existing embeddings across rather than re-embedding. Both sides use
voyage-3.5, so the stored vectors are already correct — re-embedding would
spend API calls to produce the same numbers, and would risk drift if the model
were ever silently updated between the two runs.

Idempotent: entries keep their original ids and are upserted, so re-running
after a partial failure repairs rather than duplicates.

Usage (from the repo root, with the CHROMA_* trio and VOYAGE_API_KEY set):
    python scripts/migrate_local_to_cloud.py            # dry run, shows plan
    python scripts/migrate_local_to_cloud.py --apply    # actually writes
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb
from dotenv import load_dotenv

from shared.store import COLLECTION_NAME, DEFAULT_VOYAGE_MODEL, VoyageRestEmbedding

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

DEFAULT_LOCAL_PATH = str(Path(__file__).resolve().parent.parent / "chroma_data")
BATCH = 100


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-path", default=DEFAULT_LOCAL_PATH)
    ap.add_argument("--apply", action="store_true", help="write; otherwise dry run")
    args = ap.parse_args()

    for var in ("VOYAGE_API_KEY", "CHROMA_TENANT", "CHROMA_DATABASE", "CHROMA_API_KEY"):
        if not os.environ.get(var):
            sys.exit(f"Missing {var}")

    # Read side: raw PersistentClient, no embedding function. We're copying
    # stored vectors verbatim, so nothing here needs to embed — and asking for
    # an embedding function would force the mismatch guard into a comparison
    # that isn't relevant to a read-only export.
    local = chromadb.PersistentClient(path=args.local_path)
    names = [c.name for c in local.list_collections()]
    if COLLECTION_NAME not in names:
        sys.exit(f"No '{COLLECTION_NAME}' collection at {args.local_path} (found: {names})")
    src = local.get_collection(COLLECTION_NAME)
    src_tag = (src.metadata or {}).get("embedding_function_name")
    if src_tag != DEFAULT_VOYAGE_MODEL:
        sys.exit(
            f"Local collection was built with '{src_tag}', not '{DEFAULT_VOYAGE_MODEL}'. "
            "Copying its vectors into a Cloud collection tagged voyage-3.5 would mix "
            "embedding spaces and silently corrupt similarity search. Re-embed instead."
        )

    data = src.get(include=["documents", "metadatas", "embeddings"])
    ids = data["ids"]
    print(f"source: {len(ids)} entries in {args.local_path}")
    for i, m in zip(ids, data["metadatas"]):
        print(f"  {i}  project={m.get('project')} category={m.get('category')} type={m.get('type')}")

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply to migrate.")
        return

    dst = chromadb.CloudClient(
        tenant=os.environ["CHROMA_TENANT"],
        database=os.environ["CHROMA_DATABASE"],
        api_key=os.environ["CHROMA_API_KEY"],
    ).get_or_create_collection(
        COLLECTION_NAME,
        embedding_function=VoyageRestEmbedding(os.environ["VOYAGE_API_KEY"]),
        metadata={"embedding_function_name": DEFAULT_VOYAGE_MODEL},
    )

    before = dst.count()
    for start in range(0, len(ids), BATCH):
        sl = slice(start, start + BATCH)
        dst.upsert(
            ids=ids[sl],
            documents=data["documents"][sl],
            metadatas=data["metadatas"][sl],
            embeddings=data["embeddings"][sl],
        )
        print(f"  upserted {min(start + BATCH, len(ids))}/{len(ids)}")

    after = dst.count()
    print(f"\ncloud collection '{COLLECTION_NAME}': {before} -> {after} entries")

    missing = [i for i in ids if not dst.get(ids=[i])["ids"]]
    if missing:
        sys.exit(f"FAILED: {len(missing)} ids missing after migration: {missing[:5]}")
    print(f"verified: all {len(ids)} source ids present in Cloud")


if __name__ == "__main__":
    main()
