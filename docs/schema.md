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
- `tasks` — things that need doing, each under a titled key
- `note` — anything that doesn't fit the above but is worth keeping

This list is expected to grow. Adding a category is cheap — a one-line change
to `VALID_CATEGORIES` in `shared/store.py` — but it must be deliberate, not a
typo that silently creates a bucket filtered queries can never find. Removing
or renaming a category means a migration pass over existing data — see
`scripts/migrate_goal_to_tasks.py`, which renamed `goal` to `tasks` on
2026-08-10 and is the reference for how to do it: summaries carry the category
in their id and must be re-added under a new one, everything else needs only a
metadata update, and the existing embedding is reused because the text is
unchanged.

The project/general grouping above is **guidance, not enforcement**:
`VALID_CATEGORIES` is one flat set, and any category may be used with or
without a project. It reflects where each category usually belongs, not a
constraint the code applies. In practice `tasks` is used project-scoped to hold
a project's outstanding work — which genuinely belongs to a project — and
`get_context` returns every summary for a project regardless of which group its
category nominally sits in.

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
search with no error otherwise. See `shared/store.py::VoyageRestEmbedding`.

Voyage is called over its REST endpoint directly rather than through the
`voyageai` SDK. Identical endpoint and vectors, but the SDK pulls in pillow,
tokenizers, hf_xet and aiohttp for multimodal and local-tokenizer features
this project never uses — ~90MB that the Lambda bundle can't afford (250MB
unzipped limit). Embedding functions here must return **numpy arrays**, not
plain lists: Chroma's HTTP/Cloud client calls `.tolist()` on query embeddings,
so lists fail on reads against Cloud while working fine locally.

## Write semantics

- **Summaries upsert** on a deterministic id: `f"{project}-{category}-summary"`
  (or `f"general-{category}-summary"` when there's no project). Writing a new
  summary for the same project+category REPLACES the old one — there is only
  ever one living summary per slot. Enforced in `ContextStore.save()`, not
  left to callers to get right.
- **Chunks insert** under a **content-addressed** id:
  `sha1(f"{project}-{category}-{document}")`. Same text in the same
  project+category is the same entry, always. This id previously mixed in
  `datetime.now().timestamp()`, which made every write unique and therefore made
  a re-run after a partial failure DUPLICATE everything already written instead
  of resuming — fatal for a bulk backfill, where partial failure is the expected
  case. The trade-off is accepted deliberately: byte-identical text written twice
  now collapses to one entry. A chunk carries no position or ordering, so a
  duplicate holds no information the first copy didn't, and its only effect on
  retrieval is to occupy a second slot in `top_k`.

### Sub-keys: one slot per topic, not per category

A summary slot may carry an optional `key`, making its id
`f"{project}-{category}-{key}-summary"`. **Omitting the key reproduces the
original id byte for byte**, which is what makes this additive: slots written
before keys existed keep working untouched, and a bloated category is split when
someone next has reason to touch it rather than in a migration.

The motivation is retrieval, not write cost — `patch_summary` already solved the
write side. Search truncates at 800 characters, so before splitting, the five
context-mcp summaries were 14-30% reachable; a query for a fact genuinely stored
in `config` ranked that summary first and still returned an opening that did not
contain the answer. After splitting into 31 keyed slots, mean reachability is
98% and the same query returns three untruncated hits that all contain it.

**Key collisions are the caller's decision, never the server's.** Creating a new
key in a category that already has slots requires `create_key=True`; without it
the write is refused and the category's existing keys come back, ranked closest
first. That design is forced by measurement:

| Approach | Result |
|---|---|
| difflib on key names | `lambda`/`compute` 0.15, `cognito`/`auth` 0.18 — blind to synonyms. Worse, `config-lambda`/`config-lambda-settings` scores 0.74 against a 0.75 cutoff: the real fragmentation case, missed by 0.01. |
| Voyage on key names | Classes **overlap**. Lowest same-concept pair 0.538 (`chroma`/`storage`); highest unrelated pair 0.635 (`networking`/`credentials`). No cutoff separates them. |
| Voyage on slot **content** | Right slot ranked top in most cases, near the top in all — good enough to *suggest*, not to *decide*. |

So embeddings rank the candidates (`summary_keys_ranked`, one query on the
refusal path only) and the caller chooses. A duplicate key can be merged later;
content silently merged into the wrong key cannot be unmerged.

`summary_keys()` lists a category's keys with the unkeyed slot included as
`None` — during a split it is the thing a new key is most likely to duplicate,
and leaving it out is how a category ends up with both `config` and
`config-lambda` disagreeing about the same subject.

Keys are slugified (lowercase, hyphenated, alphanumeric, ≤40 chars); `summary`
is reserved because it would collide with the unkeyed id. Slugification handles
only meaningless differences — case, spacing, punctuation. Whether `lambda` and
`compute` are the same topic is not a string problem and is not treated as one.

**Cost note:** the index grows with splitting — 114 tokens at 5 slots, 514 at
32. Still far below a brief (~4,270), but it scales linearly, so an unscoped
index across many split projects will eventually need a rollup rather than a
line per slot.

### Changing a summary: patch, don't rewrite

`patch_summary()` is the DEFAULT way to change a stored summary;
`update_summary()` is for creating a slot or rewriting one wholesale.

The reason is write cost, and it is asymmetric in a way that isn't obvious.
Replacement requires the caller to reproduce the entire new document — so
altering one line of a 1,000-token summary means *generating* 1,000 tokens, the
expensive and slow kind, to move a few characters. A patch sends only the diff.
Measured against this store's own summaries at the time of the change:
`context-mcp/config` was 966 tokens and `context-mcp/goal` (now `tasks`) 1,363, so every
correction to either paid four figures of output tokens regardless of size.

Patching is also the safer operation, which is why its guardrails are lighter:

| | `update_summary` | `patch_summary` |
|---|---|---|
| Blast radius | the whole slot | bounded by `len(old_str)` |
| Shrink guard | yes (`SHRINK_GUARD_RATIO`) | not needed — it can only rewrite what it matched |
| Archives previous | always | conditionally (below) |

Three refusals, each returning the stored document so the caller can retry
against reality rather than its own stale copy — the same contract as
`SummaryShrinkRefused`: **no match**, **more than one match** (extend `old_str`
until it is unique; patching the wrong occurrence is silent and near-impossible
to spot later), and **a patch that would change nothing**. `tier` is inherited
from the slot rather than accepted as an argument, since a patch edits something
that already exists and re-declaring its tier could only introduce disagreement.

Archiving is conditional on two independent triggers, tracked by an
`unarchived_patches` counter in the summary's own metadata:

- `PATCH_ARCHIVE_RATIO` (0.2) — a patch touching this fraction of the document
  is approaching a rewrite, so it gets archived like one.
- `PATCH_ARCHIVE_EVERY` (5) — the per-patch reasoning holds individually but not
  in aggregate: twenty surgical edits, each far under the ratio, can still
  rewrite a document between checkpoints. A forced copy every N patches bounds
  that drift.

Both are cheap because `_archive()` reuses the embedding Chroma already holds —
the text is unchanged, so its vector is still exactly right and a checkpoint
costs no Voyage call.

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

## Reading: index first, then narrow

Three read instruments, cheapest first. The order they're listed in is the order
they should be reached for:

| Tool | Cost against this store | Answers |
|---|---|---|
| `get_index` | ~114 tokens | what exists, how big, how stale |
| `get_context(p, cat, key)` | one slot | what does this one slot say |
| `get_context(p, cat)` | one category | what does X currently say |
| `get_context(p)` | ~4,280 tokens (one project) | everything, whole |
| `search_context` | ranked, truncated | history and reasoning |

`get_context` was two tools until 2026-08-12, `get_brief` and `get_value`. They
differed only in how much they returned — both deterministic lookups handing
back whole documents — so depth now comes from how much of the address is
supplied. The merge needed the keyless main slot gone first: while a category
could hold both a keyless slot and keyed ones, `(project, category)` meant
either "the category's own summary" or "everything filed under it".

`ContextStore.index()` returns the map without any of the contents: one line per
summary slot with its size and last-write date, plus a count of history chunks.
It makes **no Voyage call** — both underlying reads are metadata lookups rather
than similarity queries — and it stays small as the store grows, because there
is only ever one living summary per project+category no matter how much history
accumulates beneath it.

The size figures are the point of it: they let a caller see what a `get_context`
would cost *before* paying it, so "is there anything relevant
here" stops being a question that costs thousands of tokens to ask. Superseded
archives are excluded from the counts — they are recoverable history, not part
of what the store currently knows, and including them would make every edited
slot look like it had grown.

Documents are fetched (not just metadata) to measure their length, but the text
never leaves `index()`. That is affordable precisely because summaries are one
per slot; it would not be if the index covered chunks, which is one reason it
counts them rather than measuring them.

**This does not make a session look.** Nothing here prompts a lookup — the index
only makes looking cheap once something decides to. Getting a session to check
unprompted is a client-instruction problem, not a tool one.

## Retrieval budget

`search_context` defaults to `top_k=5`, hard-capped at 10, and truncates each
returned document to ~800 characters with a visible `…[truncated]` marker.
Prevents one call from dumping unbounded tokens into the conversation. If a
query genuinely needs full content beyond that, the summary itself should be
the thing that's short — this is a signal to fix the summary, not a reason to
raise the cap by default.

### Split at write time, not read time

The cap is a display-time cut, so an entry longer than it is a *partly
invisible* entry: no phrasing of any query reaches past the first 800
characters. When this was measured, **21 of 23 stored chunks exceeded the cap**
(median 2,584 characters), meaning `search_context` was returning a partial
document roughly 91% of the time. The fix belongs at write time, where the
material can still be divided along its own seams — only the writer knows where
those are.

Two things make that practical rather than merely advisable:

- `save_chunks()` writes a whole list in ONE upsert, and therefore one embedding
  call. Voyage bills and rate-limits per *request*, not per document, and Chroma
  issues one embed call per upsert regardless of how many documents it carries —
  so five focused chunks cost the same as one sprawling one. Without this,
  splitting would trade a retrieval problem for five times the write latency and
  nobody would do it.
- Writes that still exceed the cap come back flagged (`oversized`), at the
  moment splitting is free, rather than being discovered later as a silently
  truncated search result.

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
