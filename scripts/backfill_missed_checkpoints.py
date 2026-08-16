#!/usr/bin/env python3
"""
Backfill the checkpoints that patch_summary's archive trigger failed to take.

THE BUG. PATCH_ARCHIVE_RATIO was evaluated as len(old_str) / len(previous_doc) —
how much text a patch DISPLACED. That is blind to the append: a short old_str
swapped for a much longer new_str displaces almost nothing while growing the
document by half. On 2026-08-16 five slots were edited that way and archived
nothing between them; three of them grew 45-145%. get_history then reported they
had "only ever been patched in small pieces", which was false. Fixed in
shared/store.py by measuring max(len(old_str), len(new_str)) instead.

WHAT THIS SCRIPT RECOVERS. The pre-edit text of those five slots, reconstructed
by undoing that session's edits, written back as the superseded chunks the patch
should have produced at the time.

WHY RECONSTRUCTION IS SAFE HERE. Each edit is undone with short unique anchors
rather than by re-typing whole passages, and every result is checked against the
`chars_before` the tool reported when the patch ran. That number is an exact
checksum: if a reconstruction is off by even one character it will not match, and
the script refuses to write anything. It is all-or-nothing across all five.

    python3 scripts/backfill_missed_checkpoints.py           # dry run
    python3 scripts/backfill_missed_checkpoints.py --apply
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

from shared.store import ContextStore, SUPERSEDED_SOURCE

APPLY = "--apply" in sys.argv

# When the recovered text was actually current. All five slots last had a real
# value on 2026-08-13; the edits that should have archived them ran on 08-16.
WRITTEN_AT = "2026-08-13T14:02:00+00:00"

REASON = (
    "Backfilled checkpoint. patch_summary's archive trigger measured only the text a "
    "patch displaced, not the text it added, so the append-shaped edits of 2026-08-16 "
    "grew this slot substantially without taking the copy they should have. This is the "
    "pre-edit value, reconstructed and verified against the recorded chars_before."
)


def cut(text: str, start: str, end: str | None) -> str:
    """Remove start..end (end exclusive; to the end of the document if None)."""
    i = text.index(start)
    j = text.index(end, i) if end else len(text)
    return text[:i] + text[j:]


def swap(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"anchor not unique ({text.count(old)}x): {old[:60]!r}")
    return text.replace(old, new, 1)


# (project, category, key, expected_chars, reversal)
# Each reversal undoes that session's edits, newest first.
PLAN = [
    (
        "context-mcp", "tasks", "tool-surface", 1452,
        lambda t: cut(
            t,
            " CONSTRAINT, folded in from note/retire-semantics",
            " CONSTRAINT, and it is the reason this has not been done casually:",
        ),
    ),
    (
        "context-mcp", "tasks", "chunk-schema", 2774,
        lambda t: swap(
            cut(t, "WHAT SHIPPED 2026-08-16:", "RELATED: note/write-tool-consolidation")
            .replace(
                "RELATED: note/write-tool-consolidation (tool-layer half — giving add_update a "
                "key parameter — still open, tracked there), tasks/chunk-visibility (default "
                "search visibility of history, now unblocked), tasks/chunk-splitting "
                "(long-document reachability, builds on chunk-visibility). NOT SCHEDULED: the "
                "tool-layer half above.",
                "RELATED: note/write-tool-consolidation (tool-layer half — giving add_update a "
                "key parameter — tracked there), tasks/chunk-visibility (default search "
                "visibility of history), tasks/chunk-splitting (long-document reachability). "
                "NOT SCHEDULED.",
            ),
            "STORE-LAYER DONE, shipped 2026-08-16 (shared/store.py, docs/schema.md). The "
            "tool-layer half (add_update's key parameter) is deliberately NOT part of this — "
            "see the RELATED note below, still tracked in note/write-tool-consolidation. "
            "Decided 2026-08-13 alongside that discussion. Split out of a single larger task "
            "the same day — see tasks/chunk-visibility and tasks/chunk-splitting, both now "
            "unblocked since chunks have a real key.",
            "OPEN, not started. Core chunk schema changes, decided 2026-08-13 alongside the "
            "write-tool-consolidation discussion (note/write-tool-consolidation). Split out of "
            "a single larger task the same day — see tasks/chunk-visibility and "
            "tasks/chunk-splitting for the related but separable work. Suggested order: this "
            "one first, since both of the others build more cleanly once chunks have a real key.",
        ),
    ),
    (
        "context-mcp", "tasks", "chunk-visibility", 2159,
        lambda t: swap(
            cut(t, "WHAT SHIPPED: HIDDEN_SOURCES", "DECISION, reached 2026-08-13:"),
            "DONE AND DEPLOYED 2026-08-16 (shared/store.py, "
            "mcp_server/tools/search_context.py, mcp_server/tools/archive_slot.py, "
            "docs/schema.md). The two-constant split was confirmed correct in production: "
            "post-deploy get_index still reports history_chunks 119 for context-mcp and 158 "
            "total, unchanged from before the change. Had SEARCH_HIDDEN_SOURCES and "
            "INDEX_EXCLUDED_SOURCES been left as one narrowed constant, those counts would "
            "have jumped by roughly the 87 superseded copies — so the index arithmetic held "
            "while search visibility changed underneath it, which is exactly the intended "
            "split. Split out of tasks/chunk-schema on 2026-08-13 (that task was getting too "
            "large); built on chunk-schema, which shipped the same day.",
            "OPEN, not started. Split out of tasks/chunk-schema on 2026-08-13 (that task was "
            "getting too large). Builds on tasks/chunk-schema shipping first.",
        ),
    ),
    (
        "context-mcp", "tasks", "chunk-splitting", 3556,
        lambda t: swap(
            cut(t, "WHAT SHIPPED: MAX_DOC_CHARS", "1. THRESHOLD:"),
            "DONE 2026-08-16, all five parts, NOT YET DEPLOYED (branch "
            "summary-id-prefix-and-chunk-visibility).",
            "OPEN, not started.",
        ),
    ),
    (
        "context-mcp", "tasks", "summary-id-prefix", 1815,
        lambda t: swap(
            cut(t, "TWO THINGS THE ORIGINAL SCOPE MISSED", "NOT SCHEDULED. Raised 2026-08-13"),
            "DONE AND DEPLOYED 2026-08-16. Code merged and deployed, then "
            "scripts/summary_id_prefix.py --apply run against the live store: 102 summaries "
            "renamed, 87 superseded_from pointers repointed, 0 failures. Backup at "
            "rechunk_backups/summary-id-prefix-20260816T145011Z.json. Verified post-migration "
            "against the DEPLOYED server (not just locally): get_index returns all 11 projects "
            "with slot counts and archived_slots intact, and get_history on tasks/chunk-schema "
            "returns summary_id \"summary-context-mcp-tasks-chunk-schema\" with its archived "
            "version still reachable — which is the specific proof the pointer repointing "
            "worked, since that lookup queries superseded_from. The DEPLOY ORDER warning below "
            "is retained as the record of why this needed care.",
            "OPEN, not started.",
        ),
    ),
]


def main() -> None:
    store = ContextStore()
    prepared, failed = [], []

    for project, category, key, expected, reverse in PLAN:
        slot = f"{category}/{key}"
        found = store.get_summary(project, category, key=key)
        if found is None:
            failed.append(f"{slot}: no live summary")
            continue
        current_doc, meta, _ = found
        try:
            recovered = reverse(current_doc)
        except (ValueError, IndexError) as exc:
            failed.append(f"{slot}: reversal did not apply — {exc}")
            continue

        status = "OK " if len(recovered) == expected else "MISMATCH"
        print(f"  {status} {slot:22} {len(current_doc):5d}ch now -> "
              f"{len(recovered):5d}ch recovered (expected {expected})")
        if len(recovered) != expected:
            failed.append(
                f"{slot}: recovered {len(recovered)} chars, expected {expected}"
            )
            continue
        prepared.append((project, category, key, recovered, meta))

    if failed:
        print("\nREFUSING TO WRITE — reconstruction is not exact for:")
        for f in failed:
            print(f"  - {f}")
        print("\nNothing was written. The checksum exists precisely to stop a guess "
              "being archived as though it were the real previous value.")
        sys.exit(1)

    print(f"\nall {len(prepared)} reconstructions match their recorded chars_before")

    if not APPLY:
        print("\nDRY RUN — nothing written. Re-run with --apply.")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = ROOT / "rechunk_backups" / f"backfill-checkpoints-{stamp}.json"
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup.write_text(
        json.dumps(
            [{"project": p, "category": c, "key": k, "recovered": d}
             for p, c, k, d, _ in prepared],
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"backup written: {backup}")

    now = datetime.now(timezone.utc).isoformat()
    for project, category, key, recovered, meta in prepared:
        sid = store.summary_id(project, category, key)
        archive_meta = {
            **meta,
            "type": "chunk",
            "source": SUPERSEDED_SOURCE,
            "superseded_from": sid,
            "superseded_at": now,
            # The recovered text was current on 08-13, not today. Carrying today's
            # timestamp would date the recovered value to the session that broke it.
            "timestamp": WRITTEN_AT,
            "archived_reason": REASON,
            "backfilled": True,
        }
        archive_meta.pop("unarchived_patches", None)
        # No embedding passed: this text is not the live document, so nothing
        # cached applies to it and it needs its own vector.
        store.collection.upsert(
            ids=[f"superseded-backfill-{key}"],
            documents=[recovered],
            metadatas=[archive_meta],
        )
        print(f"  checkpointed {category}/{key} ({len(recovered)}ch)")

    print(f"\nDone. {len(prepared)} checkpoints written.")


if __name__ == "__main__":
    main()
