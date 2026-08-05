"""
Re-chunk oversized entries already in the store, so retrieval can actually see them.

THE PROBLEM. search_context truncates every hit to MAX_DOC_CHARS (800) at
DISPLAY time, so an entry longer than that is a partly invisible entry — no
phrasing of any query reaches past the opening. Measured before this ran:
11 of 12 live chunks were over the cap, and the five summaries were 14-30%
reachable. A query for a fact genuinely stored in the config summary ranked that
summary correctly and still returned an 800-character opening that did not
contain the answer.

Splitting fixes two things at once, which is why it beats simply raising the
cap. Visibility is the obvious one. The other is embedding quality: a
4,500-character entry covering eight topics is ONE vector averaging all of them,
so it is mediocre at matching any single one. Small entries get sharp vectors.

WHAT IT DOES NOT TOUCH:
  - Summaries. A summary is current state, reached deterministically through
    get_value/get_brief. Shattering one into chunks would push current state
    into the history layer, which is precisely the confusion add_update's
    description warns against. Their reachability problem is real but needs a
    different fix (a search-side pointer, or sub-keyed summary slots).
  - Superseded archives. They are excluded from search already, so their size
    costs nothing, and rewriting history is not this script's business.

SAFETY. Dry run by default. Writes a full JSON backup of every affected entry
before touching anything. New pieces are written and VERIFIED present before a
single original is deleted, so an interruption leaves duplicates (harmless,
content-addressed, re-runnable) rather than a hole.

Usage (from the repo root, with the CHROMA_* trio and VOYAGE_API_KEY set):
    python scripts/rechunk.py                 # dry run — shows every proposed split
    python scripts/rechunk.py --apply         # write, verify, then delete originals
    python scripts/rechunk.py --project X     # limit to one project
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from shared.store import MAX_DOC_CHARS, SUPERSEDED_SOURCE, ContextStore

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# Target well under the display cap rather than right at it. A piece that only
# just fits leaves no room for the context prefix, and packing to the limit
# recreates the "several topics, one vector" problem at a smaller scale.
TARGET_CHARS = 600
# A piece may exceed TARGET_CHARS to avoid breaking mid-thought, but never this.
HARD_MAX = MAX_DOC_CHARS

# How much of the parent's opening is carried onto later pieces.
CONTEXT_PREFIX_CHARS = 110

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _first_sentence(text: str) -> str:
    """The parent's opening, as a short label for the pieces that follow it."""
    opening = _SENTENCE_END.split(text.strip(), maxsplit=1)[0].strip()
    opening = " ".join(opening.split())
    if len(opening) > CONTEXT_PREFIX_CHARS:
        opening = opening[:CONTEXT_PREFIX_CHARS].rsplit(" ", 1)[0] + "…"
    return opening


def _split_paragraph(para: str, target: int) -> list[str]:
    """Break one over-long paragraph at sentence boundaries."""
    pieces, current = [], ""
    for sentence in _SENTENCE_END.split(para):
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > target:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def split_document(text: str) -> list[str]:
    """
    Group a document into retrieval-sized pieces along its own structure.

    Paragraph boundaries first, because the author already put the seams there —
    every oversized entry in this store was written as 4-11 paragraphs. Only a
    paragraph that is itself too long gets cut at sentence boundaries, and a
    single sentence is never cut, since a fragment of one retrieves badly and
    reads worse. (That is the one case where a piece can still exceed the cap;
    the caller reports it rather than mangling the sentence.)

    Pieces after the first carry a short prefix naming the parent's opening.
    Without it a piece beginning "It is still PendingConfirmation" has lost what
    "it" refers to — the split would improve visibility while degrading the very
    thing being made visible.

    The prefix is budgeted for BEFORE grouping, not bolted on after. Sizing
    pieces to the cap and then prepending context is how a splitter produces
    pieces that are still over the cap — which is exactly what it exists to
    prevent.
    """
    text = text.strip()
    if len(text) <= HARD_MAX:
        return [text]

    context = _first_sentence(text)
    reserved = len(context) + 3  # "(" + ") " around the prefix
    target = TARGET_CHARS - reserved
    hard = HARD_MAX - reserved

    units: list[str] = []
    for para in (p.strip() for p in text.split("\n\n")):
        if not para:
            continue
        units.extend(_split_paragraph(para, target) if len(para) > hard else [para])

    grouped: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}".strip()
        if current and len(candidate) > target:
            grouped.append(current)
            current = unit
        else:
            current = candidate
    if current:
        grouped.append(current)

    # The first piece opens with the context naturally; the rest need it back.
    return [p if n == 0 else f"({context}) {p}" for n, p in enumerate(grouped)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write; otherwise dry run")
    ap.add_argument("--project", default=None, help="limit to one project")
    ap.add_argument("--backup-dir", default="./rechunk_backups")
    args = ap.parse_args()

    store = ContextStore()
    got = store.collection.get(include=["documents", "metadatas"])

    candidates = []
    for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"]):
        if meta.get("type") != "chunk" or meta.get("source") == SUPERSEDED_SOURCE:
            continue
        if args.project and meta.get("project") != args.project:
            continue
        if len(doc) <= MAX_DOC_CHARS:
            continue
        candidates.append((cid, doc, meta))

    if not candidates:
        print("Nothing over the cap. Store is already retrieval-sized.")
        return

    total_before = sum(len(d) for _, d, _ in candidates)
    plan = []
    for cid, doc, meta in candidates:
        pieces = split_document(doc)
        plan.append((cid, doc, meta, pieces))

    print(f"{len(candidates)} oversized live chunk(s), {total_before} chars total\n")
    for cid, doc, meta, pieces in plan:
        print(f"=== {cid}  {meta.get('project')}/{meta.get('category')}  "
              f"{len(doc)}ch -> {len(pieces)} pieces ===")
        for n, piece in enumerate(pieces, 1):
            marker = " " if len(piece) <= MAX_DOC_CHARS else "!"
            print(f"  {marker}[{n}] {len(piece):4d}ch  {piece[:96]!r}")
        print()

    produced = sum(len(p) for _, _, _, ps in plan for p in ps)
    still_over = [p for _, _, _, ps in plan for p in ps if len(p) > MAX_DOC_CHARS]
    print(f"{len(candidates)} entries -> {sum(len(ps) for *_, ps in plan)} pieces")
    print(f"{total_before} chars -> {produced} chars (context prefixes add "
          f"{produced - total_before})")
    print(f"pieces still over {MAX_DOC_CHARS}: {len(still_over)}")

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply.")
        return

    backup_dir = Path(args.backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = backup_dir / f"rechunk-{stamp}.json"
    backup.write_text(
        json.dumps(
            [{"id": c, "document": d, "metadata": m} for c, d, m, _ in plan],
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nbackup written: {backup}")

    # Write everything first, verify, and only then delete. The reverse order
    # would put the store in a state where an interruption loses content
    # outright; this way an interruption leaves the originals alongside their
    # replacements, which a re-run reconciles.
    written: list[str] = []
    for cid, doc, meta, pieces in plan:
        result = store.save_chunks(
            documents=pieces,
            category=meta["category"],
            project=None if meta.get("project") == "general" else meta.get("project"),
            tier=meta.get("tier"),
            source=meta.get("source", "live"),
            chat_title=meta.get("chat_title"),
            # Carry the original's date. These pieces are the same knowledge
            # re-shaped, not something learned today, and get_index surfaces
            # `updated` as a staleness signal that would otherwise be reset
            # across the whole store by a single re-chunking run.
            timestamp=meta.get("timestamp"),
        )
        written.extend(result["ids"])
        print(f"  wrote {result['count']} piece(s) for {cid}")

    missing = [i for i in written if not store.collection.get(ids=[i])["ids"]]
    if missing:
        sys.exit(f"ABORT: {len(missing)} new pieces missing; originals left intact: {missing[:5]}")
    print(f"verified: all {len(written)} new pieces present")

    originals = [cid for cid, *_ in plan]
    # A piece can hash to its original's id when a document needed no context
    # prefix and split into one part — deleting it would undo the write.
    to_delete = [cid for cid in originals if cid not in set(written)]
    store.collection.delete(ids=to_delete)
    left = [cid for cid in to_delete if store.collection.get(ids=[cid])["ids"]]
    print(f"deleted {len(to_delete)} original(s); {len(left)} failed to delete")

    print(f"\nDone. Restore from {backup} if anything looks wrong.")


if __name__ == "__main__":
    main()
