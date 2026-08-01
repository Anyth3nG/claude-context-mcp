"""
Append-only log of every live save_update write, for the Phase 7 judgment
loop: skim what the model chose to save, spot over/under-triggering, tighten
the tool description or CLAUDE.md instructions accordingly.

Lives at the tool layer (save_update), NOT in ContextStore.save() — backfill
writes are already curated through the summarizer iteration process; logging
them here would only bury the signal this log exists to surface.

Run directly to skim recent entries:  python mcp_server/review_log.py [-n 20]
"""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOG_PATH = Path(__file__).resolve().parent.parent / "review_log.jsonl"


def _log_path() -> Path:
    return Path(os.environ.get("REVIEW_LOG_PATH", DEFAULT_LOG_PATH))


def log_write(entry: dict) -> None:
    """
    Append one save_update write to the review log. Must never block or fail
    a save — the log is an observability layer, not a dependency, so any
    logging error is swallowed after best effort.
    """
    record = {"logged_at": datetime.now(timezone.utc).isoformat(), **entry}
    try:
        with open(_log_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _skim(n: int) -> None:
    path = _log_path()
    if not path.exists():
        print(f"No review log yet at {path}")
        return
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    print(f"{len(lines)} total write(s) logged — showing last {min(n, len(lines))}:\n")
    for line in lines[-n:]:
        e = json.loads(line)
        header = f"[{e.get('logged_at', '?')}] {e.get('project', 'general')}/{e.get('category', '?')}"
        if e.get("corrected_from"):
            header += f" (corrected from '{e['corrected_from']}')"
        print(header)
        print(f"  {e.get('content', '')}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Skim recent save_update writes.")
    parser.add_argument("-n", type=int, default=20, help="how many recent entries to show")
    args = parser.parse_args()
    _skim(args.n)
