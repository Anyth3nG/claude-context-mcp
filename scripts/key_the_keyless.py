#!/usr/bin/env python3
"""
Give every keyless summary a key. Step 1 of decisions/keyless-migration-order.

WHY. A category that can hold BOTH a keyless main slot and keyed slots makes
get_context(project, category) ambiguous — "the category's own summary" or
"everything filed under it" — which is what blocks folding get_brief and
get_value together. Removing the keyless slot makes the ambiguity
unrepresentable rather than resolved by convention.

WHAT THIS IS NOT. Not a split. All 31 keyless slots measure 490-792 chars,
under MAX_DOC_CHARS (800), so none is truncated in search and cutting them into
smaller slots would buy nothing. split_summaries.py does not fit this job in any
case: it maps paragraph numbers to keys, and 28 of the 31 are a single
paragraph. Text here moves VERBATIM — one keyed slot per keyless slot, same
bytes, no reseaming and no editing.

The embedding is carried over rather than recomputed. The document is unchanged,
so the old vector is still exactly right and re-embedding would be 31 pointless
Voyage calls.

    python3 scripts/key_the_keyless.py           # dry run
    python3 scripts/key_the_keyless.py --apply
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

# (project, category) -> the key its text should live under. Named for what the
# slot actually says, not a uniform "overview" — a key that describes nothing is
# the same navigation problem as no key at all.
KEYS: dict[tuple[str, str], str] = {
    ("ai-systems-study", "architecture"):           "coverage",
    ("ai-systems-study", "decisions"):              "teaching-method",
    ("ai-systems-study", "tasks"):                  "next-step",
    ("aws-saa", "decisions"):                       "study-method",
    ("aws-saa", "tasks"):                           "exam-plan",
    ("context-mcp", "tech_stack"):                  "overview",
    ("devops-cloud-skillbuilding", "architecture"): "portfolio-arc",
    ("devops-cloud-skillbuilding", "decisions"):    "working-method",
    ("devops-cloud-skillbuilding", "tasks"):        "remaining",
    ("devops-cloud-skillbuilding", "tech_stack"):   "overview",
    ("household-app", "architecture"):              "overview",
    ("household-app", "decisions"):                 "tradeoffs",
    ("household-app", "tasks"):                     "planned-next",
    ("household-app", "tech_stack"):                "overview",
    ("job-search", "decisions"):                    "strategy",
    ("job-search", "tasks"):                        "status",
    ("launchpad", "architecture"):                  "overview",
    ("launchpad", "decisions"):                     "working-rules",
    ("launchpad", "tasks"):                         "status",
    ("launchpad", "tech_stack"):                    "overview",
    ("networking-study", "architecture"):           "lab-progression",
    ("networking-study", "config"):                 "lab-patterns",
    ("networking-study", "decisions"):              "lab-method",
    ("networking-study", "tasks"):                  "status",
    ("sharepoint-migration", "architecture"):       "overview",
    ("sharepoint-migration", "config"):             "remediation-workflow",
    ("sharepoint-migration", "decisions"):          "field-rules",
    ("ticketing-saas", "architecture"):             "overview",
    ("ticketing-saas", "decisions"):                "infra-choices",
    ("ticketing-saas", "tasks"):                    "status",
    ("ticketing-saas", "tech_stack"):               "overview",
}


def main() -> None:
    store = ContextStore()

    plan = []
    for (project, category), key in sorted(KEYS.items()):
        found = store.get_summary(project, category, with_embedding=True)
        if found is None:
            print(f"skip {project}/{category}: no keyless summary (already done?)")
            continue
        doc, meta, emb = found

        old_id = store.summary_id(project, category)
        new_id = store.summary_id(project, category, key)

        # Refuse to land on top of an existing keyed slot. Nothing in the current
        # store collides, but a rerun after a partial apply could.
        if store.get_summary(project, category, key=key) is not None:
            sys.exit(f"ABORT: {project}/{category}/{key} already exists — resolve by hand.")

        plan.append((project, category, key, old_id, new_id, doc, meta, emb))

    if not plan:
        print("Nothing to do — no keyless summaries found.")
        return

    for project, category, key, old_id, _, doc, _, _ in plan:
        print(f"  {project:28} {category:13} -> {key:22} {len(doc):4d}ch")
    print(f"\n{len(plan)} keyless slots -> {len(plan)} keyed slots, text unchanged")

    if not APPLY:
        print("\nDRY RUN — nothing written. Re-run with --apply.")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = ROOT / "rechunk_backups" / f"key-the-keyless-{stamp}.json"
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup.write_text(
        json.dumps(
            [{"project": p, "category": c, "new_key": k, "old_id": o,
              "document": d, "metadata": m}
             for p, c, k, o, _, d, m, _ in plan],
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nbackup written: {backup}")

    # Write every keyed slot before deleting anything, so a failure midway leaves
    # the originals intact and the migration simply re-runnable.
    for project, category, key, _, new_id, doc, meta, emb in plan:
        new_meta = dict(meta)
        new_meta["key"] = key
        store.collection.upsert(
            ids=[new_id], documents=[doc], metadatas=[new_meta], embeddings=[emb]
        )
    print(f"wrote {len(plan)} keyed slots")

    missing = [f"{p}/{c}/{k}" for p, c, k, *_ in plan
               if store.get_summary(p, c, key=k) is None]
    if missing:
        sys.exit(f"ABORT: {len(missing)} keyed slots missing; originals left intact: {missing}")
    print(f"verified: all {len(plan)} keyed slots present")

    # Only now retire the originals. Archived rather than dropped: the text lives
    # on under the new key, but the archive records that the slot was once
    # keyless, which is the thing a later reader would otherwise have to guess.
    for project, category, _, old_id, _, doc, meta, emb in plan:
        archived = store._archive(old_id, doc, meta, emb)
        store.collection.delete(ids=[old_id])
        print(f"  retired {old_id} (archived as {archived})")

    print(f"\nDone. Restore from {backup} if anything looks wrong.")


if __name__ == "__main__":
    main()
