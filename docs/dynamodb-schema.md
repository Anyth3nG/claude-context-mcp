# DynamoDB table design

Step 3 of `tasks/dynamodb-migration`. Written before any backend code, because
two things about DynamoDB are fixed at table-creation time and expensive to
change afterwards: which attributes a vector search may filter on, and the sort
key's shape.

The design has to satisfy the 18-method surface named by `shared/conformance.py`.
Every access pattern is mapped to a concrete query at the end; if a pattern isn't
in that table, the design doesn't support it.

> **One distinction to hold onto.** The equality-only restriction discovered by
> spike applies to `SearchVectors`' `SearchConditionExpression` **only**. An
> ordinary `Query` uses `KeyConditionExpression`, which supports `begins_with`
> on the sort key as normal. The design leans on `begins_with` heavily and that
> is not in tension with the spike's finding.

## Addressing

One table, `context_store`. Partition by project, because every read except the
whole-store index is already project-scoped.

| record | PK | SK |
|---|---|---|
| summary | `{project}` | `summary#{category}#{key}` |
| appended chunk | `{project}` | `chunk#{category}#{hash}` |
| superseded copy | `{project}` | `superseded#{category}#{key}#{superseded_at}#{split_index}` |

`{project}` is the literal string `general` when there is no project, matching
today's convention.

**The superseded sort key is doing real work.** Encoding the origin slot *and*
the archival timestamp means `slot_history` becomes a single partition query with
`begins_with(sk, "superseded#{category}#{key}#")`, returned in timestamp order,
with the split pieces of one archival event adjacent because they share a
timestamp and differ only in `split_index`. Reverse with
`ScanIndexForward=False` for newest-first, and group by the timestamp component
to reassemble a version.

**This removes an index the original plan called for.** `note/dynamodb-vs-chroma-eval`
specified a GSI on `superseded_from` for `slot_history`. It isn't needed: the same
query is answerable from the base table, which is cheaper, strongly consistent,
and one less thing to keep in sync.

## Global secondary indexes

Two, both earning their place.

### GSI1 — `by_type`, for the whole-store index

```
PK: type            ("summary" | "chunk")
SK: type_sk         ("{project}#{sk}")
Projection: INCLUDE [project, category, key, source, timestamp, chars,
                     superseded_from, tier]
```

`get_index()` with no project is the only access pattern that doesn't decompose
per-project, and it's also the recommended way to open a session, so it has to be
cheap. `Query` on `PK="summary"` returns every slot in the store; `PK="chunk"`
returns all history.

`type_sk` carries the project because `summary#config#lambda` is not unique
across projects — two projects can hold the same slot address.

**The projection is the point.** With `chars` projected, `get_index` never reads
a document. Today it fetches all ~96 summaries in full — roughly 40KB of text —
purely to call `len()` on each and report integers. Storing the length at write
time turns the cheapest read instrument into a genuinely cheap one.

Partition cardinality is two, which is normally a warning sign. At 441 items it
is irrelevant, and the alternative — maintaining counter items — trades a fresh
read for cache drift, which contradicts `decisions/map-honesty`.

### GSI2 — `by_id`, for id-addressed operations

```
PK: id
Projection: ALL
```

`search_context` hands back ids, and `archive(id)` and `record(id)` consume them.
Those ids cannot be parsed back into an address: `summary-context-mcp-config-api-gateway`
has hyphens in the project *and* the key, so recovering the parts requires
scanning for a token that happens to be in `VALID_CATEGORIES`. That works today
and would break the first time a project is named after a category.

So ids stay opaque handles and get their own index. Keeping the existing id
format also means every id already recorded in the store, in `docs/`, and in
tool responses stays valid across the migration.

## The vector index

```
IndexName:        semantic
VectorAttribute:  embedding
Dimensions:       1024               (voyage-3.5)
DistanceFunction: COSINE
Projection:       ALL
SearchSchema:     project    INLINE_FILTER
                  category   INLINE_FILTER
                  type       INLINE_FILTER
                  searchable INLINE_FILTER
```

No `HASH` element, deliberately: the spike confirmed an unscoped search then
spans all partitions, which `search_context(project=None)` requires.

The four filterable attributes are exactly what the two vector call sites need —
`search` filters on project, category and visibility; `summary_keys_ranked`
additionally needs `type = "summary"`. **Every one of these must also appear in
`AttributeDefinitions`, and the set is fixed when the table is created**, so
adding a fifth filter dimension later is a table-level change. Choose now or pay
later.

### `searchable` replaces a filter that cannot be expressed

`search()` currently excludes retired chunks with `{"source": {"$ne": "retired"}}`.
`SearchConditionExpression` rejects `<>`, `NOT`, `IN` and `OR` — so the exclusion
inverts into a stored positive flag:

- `searchable = "y"` on every record **except** retired ones
- default search filters `searchable = :y`
- `include_retired=True` omits the clause entirely, which returns everything —
  matching today's behaviour exactly

Retired records keep their vector. They must remain *findable* when explicitly
asked for, so dropping them out of the index isn't an option.

## Attributes

| attribute | type | on | notes |
|---|---|---|---|
| `project` | S | all | PK. **Reserved word** — needs `ExpressionAttributeNames` aliasing in every expression that names it. |
| `sk` | S | all | SK |
| `id` | S | all | the legacy id string; GSI2 PK |
| `type` | S | all | `summary` \| `chunk`; GSI1 PK |
| `type_sk` | S | all | `{project}#{sk}`; GSI1 SK |
| `category` | S | all | filterable |
| `key` | S | most | absent on keyless legacy chunks |
| `source` | S | all | `live` \| `backfill` \| `superseded` \| `retired` |
| `searchable` | S | all | `"y"`, or `"n"` when retired |
| `timestamp` | S | all | when written |
| `chars` | N | all | **new** — length of `document`, so the index needn't read it |
| `document` | S | all | max observed 5,846 chars |
| `embedding` | L of N | all | 1024 floats, ~12KB per item |
| `tier`, `chat_title`, `category_corrected_from` | S | optional | unchanged |
| `unarchived_patches` | N | summaries | patch checkpoint counter |
| `superseded_from`, `superseded_at` | S | superseded | provenance; redundant with SK but kept queryable |
| `split_index`, `split_count` | N | split pieces | reassembly |
| `archived_reason`, `retired_at`, `retired_from_source`, `superseded_by` | S | optional | unchanged |

Item size is dominated by the embedding at ~12KB, comfortably under the 400KB
limit. Billing is `PAY_PER_REQUEST` — vector indexes support on-demand only.

## Access patterns → queries

| contract method | operation |
|---|---|
| `get_summary` | `GetItem` PK=project, SK=`summary#{cat}#{key}`, strongly consistent |
| `get_brief(p)` | `Query` PK=p, `begins_with(sk, "summary#")` |
| `get_brief(p, cat)` | `Query` PK=p, `begins_with(sk, "summary#{cat}#")` |
| `summary_keys` | same as above, project the key |
| `summary_keys_ranked` | `SearchVectors` filtered `project` + `category` + `type="summary"` |
| `slot_history` | `Query` PK=p, `begins_with(sk, "superseded#{cat}#{key}#")`, reversed |
| `index(p)` | 3× `Query` on PK=p with `begins_with` per prefix |
| `index()` | 2× `Query` on GSI1 (`type="summary"`, `type="chunk"`) |
| `search` | `SearchVectors` filtered `searchable` (+ project/category when given) |
| `record(id)` | `Query` GSI2 PK=id |
| `records`/`count` | `Query` on base table or GSI1, filters as equality |
| `save`, `update_summary` | `PutItem` |
| `save_chunks` | `BatchWriteItem`, 25 per request |
| `patch_summary` | `PutItem` (+ archive writes) |
| `archive_slot` | write superseded item(s), then `DeleteItem` on the summary |
| `retire_chunk` | `UpdateItem` — sets `source`, `searchable="n"`, reason |

Every read is a `GetItem` or a `Query`. **No `Scan` anywhere**, which was the
substance of the argument for moving.

## Three changes this forces in the store's own contract

1. **`chars` is written on every item.** New attribute; the write path must set
   it and the migration must backfill it.
2. **`searchable` is written on every item.** Derived from `source`, but stored,
   because the filter grammar cannot compute it.
3. **`records(superseded_from=sid)` should take components instead.**
   `shared/conformance.py` calls it with a raw sid, which would force id parsing.
   Change it to `records(project=…, category=…, key=…, source="superseded")` —
   one call site, and it is more honest about what it is asking for.

## Pagination, for free

DynamoDB `Query` paginates with `LastEvaluatedKey`, and it is explicit: you know
when there is more. Chroma's 300-record `get()` cap returns a short answer that
looks complete, which is `tasks/get-pagination`. That defect does not survive the
migration — which is the argument for not fixing it on the Chroma side first.

## Open, needing a decision before the table is created

- **Is the filterable set final?** `project`, `category`, `type`, `searchable`.
  Anything you might later want to filter a *semantic search* by — `tier`, say,
  or a date range — has to be declared now.
- **`COSINE` vs `DOT_PRODUCT`.** Voyage returns normalised vectors, for which the
  two rank identically; cosine is the safer default if that ever changes.
- **One table or two?** A single table with a `type` partition on GSI1 is
  proposed. Separate summary and chunk tables would give cleaner index
  cardinality at the cost of losing single-partition reads across both.
