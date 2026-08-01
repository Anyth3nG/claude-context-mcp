# Context store schema

Every entry written to ChromaDB (whether from backfill or a live `save_update`
call) carries this metadata. This is the contract every part of the system
relies on — the backfill script, the MCP server's read/write tools, and any
future visualization all read/write against this shape. Change it deliberately.

## Fields

| Field | Type | Required | Notes |
|---|---|---|---|
| `project` | string \| null | No | Name of the project this belongs to. `null` (stored as the literal string `"general"`) for things that aren't project-specific — preferences, facts about you, things you're mulling over. |
| `tier` | string | Yes if `project` set | `"client"` or `"personal"`. Signals how much retrieval depth/effort is warranted. Omitted for general entries — `ContextStore.save()` strips it if `project` isn't set, even if a caller passes one. |
| `category` | string | Yes | What kind of thing this is. See categories below. Applies to every entry, project or not. |
| `type` | string | Yes | `"summary"` (living, distilled doc for a project+category) or `"chunk"` (raw conversation fragment, fallback layer). |
| `source` | string | Yes | `"backfill"` or `"live"` — where this entry came from. Useful for debugging and for knowing what's safe to bulk-regenerate. |
| `timestamp` | ISO 8601 string | Yes | When this was written (not necessarily when the underlying conversation happened — track both if they diverge). |
| `chat_title` | string | No | Original conversation title, if this came from a claude.ai export. Helps trace an entry back to its source chat. |
| `category_corrected_from` | string | No | Present only if the `category` passed in was a typo that got auto-corrected (see Category enforcement below). Holds the original, uncorrected value. |

## Categories

Project-scoped:
- `tech_stack`
- `architecture`
- `config`
- `decisions`

General (`project = null`):
- `preference` — how you like things done, working style
- `fact` — durable facts about you
- `goal` — things you're working toward or considering
- `note` — anything that doesn't fit the above but is worth keeping

This list is expected to grow. Adding a category is cheap — a one-line change
to `VALID_CATEGORIES` in `shared/store.py` — but it must be deliberate, not a
typo that silently creates a bucket filtered queries can never find. Removing
or renaming a category means a migration pass over existing data.

## Collection structure in ChromaDB

Single collection, `context_store`, for everything — summaries, chunks,
project-scoped, and general. Filtering happens via the `where` clause on
metadata (`project`, `category`, `type`), not via separate collections.
Keeping it one collection avoids needing to fan a query out across many
collections later, and cross-project / project-agnostic queries stay simple.

## Embedding model

**Decision: Voyage** (`voyage-3.5`), not Chroma's local default
(sentence-transformers/MiniLM). Free tier (200M tokens) comfortably covers
this project's full backfill + ongoing use — cost is effectively $0. Chosen
over the local default because retrieval quality on nuanced "what did we
decide about X" queries is the actual bottleneck this system exists to
solve, and Voyage is materially stronger there. Trade-off accepted: document
text is sent to Voyage's API to be embedded (not used for their model
training), rather than staying fully on-box.

This is enforced, not just documented: `ContextStore.__init__` raises
`RuntimeError` if `VOYAGE_API_KEY` isn't set and no embedding function is
explicitly provided — it will not silently fall back to the weaker local
default. Opting out requires an explicit `allow_local_fallback=True`. The
collection also remembers which embedding function created it (stored in
collection metadata as `embedding_function_name`) and refuses to reopen with
a different one, since mixing embedding spaces silently corrupts similarity
search with no error otherwise. See `shared/store.py::voyage_embedding_function`.

## Write semantics

- **Summaries upsert** on a deterministic id: `f"{project}-{category}-summary"`
  (or `f"general-{category}-summary"` when there's no project). Writing a new
  summary for the same project+category REPLACES the old one — there is only
  ever one living summary per slot. Enforced in `ContextStore.save()`, not
  left to callers to get right.
- **Chunks always insert** with a unique generated id — never overwritten,
  accumulate over time as a raw fallback layer.

## Category enforcement

`category` is checked against a fixed set (`VALID_CATEGORIES` in
`shared/store.py`). Close typos (e.g. `"desicions"` → `"decisions"`, difflib
ratio ≥ 0.75) are auto-corrected, not rejected — but the correction is always
visible: `save()` returns `corrected_from`, `search()` returns
`category_corrected_from`, and a corrected write additionally stores
`category_corrected_from` in its own metadata. Only genuinely unmatched input
(no category within the 0.75 threshold) raises an error. This balances two
things that were in tension: typos are the expected common case (not
malicious or ambiguous), but a silent correction would just relocate the
original problem — an entry filed under a category the caller didn't
realize was substituted.

## Retrieval budget

`search_context` defaults to `top_k=5`, hard-capped at 10, and truncates each
returned document to ~800 characters with a visible `…[truncated]` marker.
Prevents one call from dumping unbounded tokens into the conversation. If a
query genuinely needs full content beyond that, the summary itself should be
the thing that's short — this is a signal to fix the summary, not a reason to
raise the cap by default.

## Example entries

```json
{
  "id": "proj-ticketing-arch-001",
  "document": "The ticketing SaaS uses a Postgres primary store with...",
  "metadata": {
    "project": "ticketing-saas",
    "tier": "client",
    "category": "architecture",
    "type": "summary",
    "source": "backfill",
    "timestamp": "2026-08-01T10:00:00Z"
  }
}
```

```json
{
  "id": "general-pref-003",
  "document": "Prefers intuition-first explanations with no math, wants to control pace of learning.",
  "metadata": {
    "project": "general",
    "category": "preference",
    "type": "summary",
    "source": "live",
    "timestamp": "2026-08-01T10:05:00Z"
  }
}
```
