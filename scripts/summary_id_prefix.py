#!/usr/bin/env python3
"""
Rename every summary id from the SUFFIX form to the PREFIX form.

    {project}-{category}-summary        ->  summary-{project}-{category}
    {project}-{category}-{key}-summary  ->  summary-{project}-{category}-{key}

WHY. Chunk ids already announce their type from the first token — `chunk-<hash>`
and `superseded-<hash>` — so a reader or a dispatcher can tell type by prefix
alone. Summary ids were the one exception, forcing an "ends with -summary" check
instead of the "starts with X-" rule used everywhere else. That asymmetry
surfaced while designing the merged archive()/retire tool (tasks/tool-surface),
whose target-type dispatch had to special-case it.

WHY THIS NEEDS A MIGRATION AT ALL, unlike sub-keys. Adding `key` was additive:
omitting it reproduced the old id byte for byte, so old slots kept working and
could be converted lazily. This changes the shape of EVERY summary id, keyed or
not. An unmigrated slot would silently stop matching summary_id()'s output, and
the next update_summary/patch_summary against it would create a SECOND live
document rather than updating the original — a split-brain slot, exactly the
failure tasks/id-mismatch-audit exists to chase down. So this is one pass, all
of them, or none.

SECOND HALF, easy to miss: `superseded_from` on archived chunks stores the id of
the slot it came from, and slot_history() looks history up by querying
`where={"superseded_from": <current-format id>}`. Renaming the live summaries
alone would leave every archived version pointing at an id that no longer
exists — get_history would return "no versions" for slots that demonstrably have
them, and index()'s archived_slots count would silently drop to zero. Those
pointers are rewritten here in the same pass, as metadata-only updates.

Nothing is re-embedded. Documents are unchanged, so the existing vectors stay
exactly right — same trick scripts/key_the_keyless.py used for the same reason.

    python3 scripts/summary_id_prefix.py           # dry run
    python3 scripts/summary_id_prefix.py --apply
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from shared.store import ContextStore

APPLY = "--apply" in sys.argv


def old_summary_id(project: str | None, category: str, key: str | None = None) -> str:
    """The pre-migration SUFFIX form, kept here so the rename has both sides."""
    base = f"{project or 'general'}-{category}"
    return f"{base}-{key}-summary" if key else f"{base}-summary"


def main() -> None:
    store = ContextStore()

    # ---- Half 1: the live summaries themselves -----------------------------
    summaries = store.collection.get(
        where={"type": "summary"}, include=["documents", "metadatas", "embeddings"]
    )
    embeddings = summaries.get("embeddings")

    renames = []
    already = 0
    for i, (sid, doc, meta) in enumerate(
        zip(summaries["ids"], summaries["documents"], summaries["metadatas"])
    ):
        project, category, key = meta.get("project"), meta.get("category"), meta.get("key")
        new_id = store.summary_id(project, category, key)
        if sid == new_id:
            already += 1
            continue
        expected_old = old_summary_id(project, category, key)
        if sid != expected_old:
            # Neither the old format nor the new one. Renaming on a guess could
            # land two slots on one id, so this stops rather than improvising.
            sys.exit(
                f"ABORT: {sid!r} matches neither the old format ({expected_old!r}) "
                f"nor the new one ({new_id!r}). Resolve by hand — see "
                "tasks/id-mismatch-audit."
            )
        emb = embeddings[i] if embeddings is not None and len(embeddings) else None
        renames.append((sid, new_id, doc, meta, emb))

    collisions = [n for _, n, _, _, _ in renames if n in set(summaries["ids"])]
    if collisions:
        sys.exit(f"ABORT: {len(collisions)} new ids already exist: {collisions}")

    # ---- Half 2: superseded_from pointers on archived chunks ---------------
    archived = store.collection.get(
        where={"type": "chunk"}, include=["metadatas"]
    )
    pointer_updates = []
    pointers_already = 0
    for cid, meta in zip(archived["ids"], archived["metadatas"]):
        origin = meta.get("superseded_from")
        if not origin:
            continue
        # An archived copy carries the slot's own metadata (see _archive), so the
        # new pointer is derivable without parsing the old id string.
        new_origin = store.summary_id(meta.get("project"), meta.get("category"), meta.get("key"))
        if origin == new_origin:
            pointers_already += 1
            continue
        expected_old = old_summary_id(meta.get("project"), meta.get("category"), meta.get("key"))
        if origin != expected_old:
            print(
                f"  WARN {cid}: superseded_from={origin!r} does not match its own "
                f"metadata ({expected_old!r}). Left untouched."
            )
            continue
        pointer_updates.append((cid, dict(meta), origin, new_origin))

    # ---- Report ------------------------------------------------------------
    for old_id, new_id, doc, _, _ in renames:
        print(f"  {old_id:52} -> {new_id:52} {len(doc):5d}ch")
    print(
        f"\n{len(renames)} summaries to rename "
        f"({already} already in the new format)"
    )
    print(
        f"{len(pointer_updates)} superseded_from pointers to repoint "
        f"({pointers_already} already correct)"
    )

    if not renames and not pointer_updates:
        print("\nNothing to do.")
        return

    if not APPLY:
        print("\nDRY RUN — nothing written. Re-run with --apply.")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = ROOT / "rechunk_backups" / f"summary-id-prefix-{stamp}.json"
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup.write_text(
        json.dumps(
            {
                "summaries": [
                    {"old_id": o, "new_id": n, "document": d, "metadata": m}
                    for o, n, d, m, _ in renames
                ],
                "pointers": [
                    {"chunk_id": c, "old_origin": o, "new_origin": n}
                    for c, _, o, n in pointer_updates
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nbackup written: {backup}")

    # Write every new id BEFORE deleting any old one, so a failure midway leaves
    # the originals intact and the whole thing simply re-runnable.
    for _, new_id, doc, meta, emb in renames:
        kwargs = {"ids": [new_id], "documents": [doc], "metadatas": [dict(meta)]}
        if emb is not None:
            kwargs["embeddings"] = [emb]
        store.collection.upsert(**kwargs)
    print(f"wrote {len(renames)} slots under their new ids")

    missing = [n for _, n, _, _, _ in renames
               if not store.collection.get(ids=[n], include=[])["ids"]]
    if missing:
        sys.exit(f"ABORT: {len(missing)} new ids missing; originals left intact: {missing}")
    print(f"verified: all {len(renames)} new ids present")

    # Repoint history before dropping the old slots — if this half fails, the old
    # ids are still there and the pointers still resolve.
    for cid, meta, _, new_origin in pointer_updates:
        meta["superseded_from"] = new_origin
        store.collection.update(ids=[cid], metadatas=[meta])
    print(f"repointed {len(pointer_updates)} superseded_from pointers")

    store.collection.delete(ids=[o for o, _, _, _, _ in renames])
    print(f"deleted {len(renames)} old ids")

    print(f"\nDone. Restore from {backup} if anything looks wrong.")


if __name__ == "__main__":
    main()
