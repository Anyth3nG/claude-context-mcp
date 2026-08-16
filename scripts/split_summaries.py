"""
Split oversized summaries into sub-keyed slots, preserving their text verbatim.

WHY. search_context truncates every hit at MAX_DOC_CHARS (800), so a long
summary is only ever visible from the top: measured before this ran, the five
context-mcp summaries were 14-30% reachable, and a query for a fact genuinely
stored in `config` ranked that summary correctly and still returned an opening
that did not contain the answer. scripts/rechunk.py fixed the same problem for
history chunks; summaries need a different instrument, because shattering one
into chunks would push current state into the history layer.

WHAT IT DOES NOT DO: change any wording. Paragraphs move into keyed slots
exactly as written. Splitting and editing are separate operations, and doing
them together makes it impossible to tell which one lost something. Content
that has gone stale is a job for patch_context afterwards — cheaply, now that
the slots are small.

The mapping below is a judgment call about topic boundaries, not something
derived. It is written out explicitly so it can be argued with, and so a dry
run shows exactly what would land.

Usage (from the repo root, with the CHROMA_* trio and VOYAGE_API_KEY set):
    python scripts/split_summaries.py            # dry run
    python scripts/split_summaries.py --apply    # write keyed slots, retire the original
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from shared.store import ContextStore

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

PROJECT = "context-mcp"

# category -> [(key, [1-based paragraph numbers])]
# Paragraphs are grouped only where merging keeps the slot under the 800-char
# display cap; where it would not, the halves get their own keys instead
# (alerting / alerting-todo, phase-f-alerting / phase-f-todo). architecture's
# "auth" is the one deliberate non-adjacent merge: paragraph 8 is a consequence
# of paragraph 4 and reads as a fragment on its own.
SPLITS: dict[str, list[tuple[str, list[int]]]] = {
    "config": [
        ("api-gateway", [1]),
        ("lambda", [2]),
        ("cognito", [3]),
        ("env-vars", [4]),
        ("secrets", [5]),
        ("alerting", [6]),
        ("alerting-todo", [7]),
        ("rotation", [8]),
        ("repo-ci", [9]),
    ],
    "architecture": [
        ("storage-model", [1]),
        ("tool-design", [2]),
        ("lambda-runtime", [3]),
        ("auth", [4, 8]),
        ("map", [5]),
        ("rejection-logging", [6]),
        ("logging-redaction", [7]),
    ],
    "decisions": [
        ("serverless-over-ec2", [1]),
        ("voyage-embeddings", [2]),
        ("cognito-only", [3]),
        ("map-unregistered", [4]),
        ("no-ip-allowlist", [5]),
    ],
    "goal": [
        ("status", [1, 2]),
        ("phase-a-ci", [3]),
        ("phase-b-auth", [4]),
        ("phase-f-alerting", [5]),
        ("phase-f-todo", [6]),
        ("phase-c-backfill", [7]),
        ("phase-d-retrieval", [8]),
        ("phase-e-loose-ends", [9]),
        ("phase-g-map", [10]),
        ("phase-g-scope", [11]),
    ],
    # tech_stack is left whole on purpose: one topic, 1,047 chars, already 76%
    # reachable. Splitting it would add keys without buying visibility.
}


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--backup-dir", default="./rechunk_backups")
    args = ap.parse_args()

    store = ContextStore()
    plan = []
    for category, mapping in SPLITS.items():
        found = store.get_summary(args.project, category)
        if found is None:
            print(f"skip {category}: no summary")
            continue
        doc, meta, _ = found
        paras = paragraphs(doc)

        used = sorted(n for _, nums in mapping for n in nums)
        expected = list(range(1, len(paras) + 1))
        if used != expected:
            sys.exit(
                f"ABORT {category}: mapping covers paragraphs {used} but the stored "
                f"summary has {len(paras)}. It has changed since the mapping was "
                "written — re-read it and update SPLITS."
            )
        slots = [(key, "\n\n".join(paras[n - 1] for n in nums)) for key, nums in mapping]
        plan.append((category, doc, meta, slots))

    if not plan:
        print("Nothing to split.")
        return

    for category, doc, meta, slots in plan:
        print(f"\n=== {args.project}/{category}  {len(doc)}ch -> {len(slots)} keyed slots ===")
        for key, text in slots:
            over = "!" if len(text) > 800 else " "
            print(f"  {over}{key:22} {len(text):4d}ch  {' '.join(text.split())[:78]}")

    total_before = sum(len(d) for _, d, _, _ in plan)
    all_slots = [s for *_, slots in plan for s in slots]
    over = [k for k, t in all_slots if len(t) > 800]
    print(f"\n{len(plan)} summaries ({total_before}ch) -> {len(all_slots)} keyed slots")
    print(f"slots over 800 chars: {len(over)}{' ' + str(over) if over else ''}")

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply.")
        return

    backup_dir = Path(args.backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = backup_dir / f"split-summaries-{stamp}.json"
    backup.write_text(
        json.dumps(
            [{"category": c, "document": d, "metadata": m} for c, d, m, _ in plan],
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nbackup written: {backup}")

    for category, doc, meta, slots in plan:
        for key, text in slots:
            store.update_summary(
                document=text,
                category=category,
                project=args.project,
                tier=meta.get("tier"),
                source=meta.get("source", "live"),
                key=key,
                # The whole point of this run is to open these keys; the
                # interactive guard would refuse every one of them.
                create_key=True,
            )
        print(f"  wrote {len(slots)} slots for {category}")

    # Verify before retiring anything.
    missing = [
        f"{c}/{k}" for c, _, _, slots in plan for k, _ in slots
        if store.get_summary(args.project, c, key=k) is None
    ]
    if missing:
        sys.exit(f"ABORT: {len(missing)} keyed slots missing; originals left intact: {missing}")
    print(f"verified: all {len(all_slots)} keyed slots present")

    # Retire the now-redundant unkeyed slot, but archive it first — the text
    # lives on in the keys, yet the original grouping is the only record of how
    # it was once organised, and _archive costs no embedding call.
    for category, doc, meta, _ in plan:
        sid = store.summary_id(args.project, category)
        found = store.get_summary(args.project, category, with_embedding=True)
        if found is None:
            continue
        prev_doc, prev_meta, prev_emb = found
        archived = store._archive(sid, prev_doc, prev_meta, prev_emb)
        store.collection.delete(ids=[sid])
        print(f"  retired {sid} (archived as {', '.join(archived)})")

    print(f"\nDone. Restore from {backup} if anything looks wrong.")


if __name__ == "__main__":
    main()
