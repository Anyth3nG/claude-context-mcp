"""
Append-only log of every live write, for the judgment loop: skim what the model
chose to save, spot over/under-triggering, tighten the tool description or
CLAUDE.md instructions accordingly.

Lives at the tool layer, NOT in ContextStore.save() — backfill writes are
already curated through the summarizer iteration process; logging them here
would only bury the signal this log exists to surface.

TWO SINKS, because a file is the wrong one in production. This wrote only to a
path under the repo root, which is read-only on Lambda, and swallowed the
resulting OSError — so every deployed write was logged nowhere and the failure
was invisible. On Lambda it now prints instead, which CloudWatch captures, in
the same shape as auth.py's rejection lines so the two are greppable together.

Run directly to skim recent local entries:  python mcp_server/review_log.py [-n 20]
"""
from __future__ import annotations
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DEFAULT_LOG_PATH = Path(__file__).resolve().parent.parent / "review_log.jsonl"

# One fixed, greppable marker, matching auth.py's REJECT_MARKER convention so a
# CloudWatch Logs Insights query or a metric filter can pick these out.
WRITE_MARKER = "CONTEXT_MCP_WRITE"


def _log_path() -> Optional[Path]:
    """
    Where to append, or None when there is no writable place to append to.

    An explicit REVIEW_LOG_PATH always wins — that is what tests set. Otherwise
    a filesystem log only makes sense off Lambda: /var/task is read-only, and
    /tmp would survive only until the next cold start, which is worse than
    useless for a log whose whole purpose is to be read days later.
    """
    override = os.environ.get("REVIEW_LOG_PATH")
    if override:
        return Path(override)
    if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        return None
    return DEFAULT_LOG_PATH


def log_write(entry: dict) -> None:
    """
    Record one live write. Must never block or fail a save — this is an
    observability layer, not a dependency, so any logging error is swallowed
    after best effort.

    Note that on Lambda the record goes to CloudWatch, which is a wider audience
    than a file on one machine. What lands here is content the user deliberately
    chose to store, never a credential — keep it that way.
    """
    record = {"logged_at": datetime.now(timezone.utc).isoformat(), **entry}
    line = json.dumps(record, ensure_ascii=False)

    path = _log_path()
    if path is None:
        print(f"{WRITE_MARKER} {line}", file=sys.stdout, flush=True)
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        # Last resort: a log line on stdout beats losing the record silently,
        # which is the failure this module just spent a paragraph explaining.
        print(f"{WRITE_MARKER} {line}", file=sys.stdout, flush=True)


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
