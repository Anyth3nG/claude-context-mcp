# claude-context-mcp

A durable memory store for Claude, shared across every machine and both clients.
Context saved from Claude Code on one laptop is available to claude.ai in a
browser on another, because both talk to the same MCP server over the network
rather than to anything on disk.

It exists because local memory doesn't follow you. A decision recorded on one
machine is invisible from a second machine, and invisible from claude.ai
entirely — so the same ground gets re-covered, or worse, re-decided differently.

## What it stores

Two tiers, and the distinction runs through everything:

- **Summaries** are current state. One living document per
  `project / category / key` — "what is true now".
- **Chunks** are history. Appended facts and archived earlier versions of
  summaries, kept forever and searchable, never treated as current.

A **key** names a sub-topic (`config/cognito`, `tasks/status`), which is what
keeps entries short enough to be retrievable — search truncates, so a sprawling
slot is a partly-invisible slot.

Full data model, write semantics, and the reasoning behind each: **[docs/schema.md](docs/schema.md)**.

## The seven tools

| Tool | | |
|---|---|---|
| `get_index` | read | the map — what exists, how big, how stale. No contents. |
| `get_context` | read | whole documents, at whatever depth the address implies |
| `get_history` | read | how one slot changed, newest first |
| `search_context` | read | ranked and truncated; for when you don't know where to look |
| `add_update` | write | append a fact to history |
| `patch_context` | write | write current state — a diff, or a wholesale replacement |
| `archive` | write | take something out of the default read |

Open with `get_index(detail="projects")` — a table of contents for tens of
tokens — then go only as deep as the task needs.

## Stack

Python 3.13. MCP server on the `mcp` SDK (streamable-HTTP, stateless), deployed
to AWS Lambda (arm64) behind API Gateway, with Mangum adapting ASGI to Lambda
events. Vectors in DynamoDB (its native vector search — `SearchVectors`),
embeddings from Voyage `voyage-3.5` over REST. Auth is Cognito OAuth 2.1, tokens
verified locally against Cognito's JWKS. Credentials in Secrets Manager. CI is
GitHub Actions over OIDC.

The store is selected at runtime: setting `DYNAMODB_TABLE` picks the DynamoDB
driver, unsetting it falls back to Chroma (Cloud with the `CHROMA_*` trio
configured, a local persist directory otherwise). Chroma remains installed and
configured as the rollback path — reverting the backend is an environment
change, not a redeploy. Table design: **[docs/dynamodb-schema.md](docs/dynamodb-schema.md)**.

## Running your own

There is no hosted instance — this is bring-your-own-everything. You need an AWS
account and a Voyage API key. (A Chroma Cloud account only if you run the Chroma
backend instead of DynamoDB.)

### 1. Secrets

Create a Secrets Manager secret (default id `context-mcp/credentials`).
`shared/config.py` raises at cold start naming whatever is missing:

```
VOYAGE_API_KEY               # always required — DynamoDB stores vectors,
                             # Voyage produces them
CHROMA_TENANT                # required only when DYNAMODB_TABLE is unset
CHROMA_DATABASE              # (the Chroma backend / rollback path)
CHROMA_API_KEY
MAP_CLIENT_SECRET            # optional — /map browser login; /map fails
                             # closed without it
```

Every key in the secret is loaded into the environment at cold start, so
anything that must not sit in the function's configuration (which CI can read)
belongs here.

### 2. Non-secret environment

Seven variables on the Lambda:

```
CONTEXT_MCP_SECRET_ID        # the secret id above
DYNAMODB_TABLE               # selects the DynamoDB store; unset = Chroma
MCP_ALLOWED_HOST             # exact API Gateway hostname — "*" is compared
                             # literally and yields HTTP 421
COGNITO_REGION
COGNITO_USER_POOL_ID
COGNITO_ALLOWED_CLIENT_IDS
MCP_REQUIRED_SCOPES
```

### 3. Cognito, by hand

Manual console work, not covered by `infra/setup.sh`. Create a user pool with a
hosted UI domain and a resource server exposing an access scope, then two app
clients:

- **Confidential client** (has a secret) for the claude.ai connector.
- **Public client** with PKCE for Claude Code, callback
  `http://localhost:8765/callback`.

Cognito matches redirect URIs **exactly** — the port is pinned, and a mismatch
fails at login with nothing pointing at why.

### 4. Deploy

```bash
./infra/setup.sh          # one-time: Lambda, API Gateway, IAM, alarm
```

After that, CI deploys on every push to `main`: build bundle → update function →
publish version → wait for SnapStart optimization → move the `live` alias →
prune superseded versions.

### 5. Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # full chromadb, for a local store
python -m shared.smoke                   # full suite, offline, no API keys
```

The suite is split in two on purpose. `shared/conformance.py` holds the
**behavioural contract** — everything true of this store regardless of what
stores it — and takes a factory, so it can be pointed at a second backend and a
passing run is that backend's proof. `shared/smoke.py` is the Chroma harness: it
runs the contract, then Chroma's own engine tests (the embedding-function
mismatch guard, local-fallback reopen), which assert things about the engine
rather than about this store's semantics. `shared/smoke_dynamodb.py` is the
DynamoDB harness: the same contract against a throwaway table it creates and
deletes (needs AWS credentials, no API keys).

The entry point deliberately does **not** live in `shared/store.py`. Running that
as `__main__` gives Python two copies of the module and two sets of exception
classes, so `except PatchNoMatch` stops catching the `PatchNoMatch` that was
raised. `python -m shared.store` still works and just delegates.

`requirements.txt` pulls full `chromadb`; the Lambda bundle uses
`requirements-lambda.txt` with the thin `chromadb-client`, which exposes
`PersistentClient` but raises "http-only client mode" if constructed. The bundle
also carries its own `boto3` (~35MB): the runtime's copy predates DynamoDB's
`SearchVectors` API, and that gap surfaced only in production — see the note in
`requirements-lambda.txt`. The bundle has a hard 250MB unzipped limit and
currently sits at ~128MB unzipped / ~50MB zipped. The zipped figure is the one
to watch: it is within a few percent of the ~52MB direct-upload cap, and
crossing it breaks the deploy, not the runtime. Dropping `chromadb-client`
once the Chroma rollback path retires buys the headroom back.

## Known gap: `/map` is not reproducible from infra

The `/map` visualization reads three further variables — `MAP_CLIENT_ID`,
`MAP_COGNITO_DOMAIN`, `MAP_CLIENT_SECRET` — and **nothing in `infra/setup.sh` or
`.github/` sets any of them**. In the live deployment all three sit in the
Secrets Manager secret (every key there is loaded into the environment at cold
start, so the client secret no longer lives in the function's configuration),
but the Cognito app client they describe is created by hand. A clean deploy
therefore brings up the MCP server correctly and leaves `/map` non-functional —
it fails closed, with the reason visible only in the auth-rejection log.

## Scripts

`scripts/` holds one-shot migrations and audits, each with a dry-run mode and a
written rationale at the top. `audit_ids.py` is the one worth knowing about: it
checks that every record's id agrees with its own metadata, and is safe to run
against a live store at any time.

Note that Chroma Cloud caps a single `get()` at 300 records and returns at most
that many **silently** — any script walking the whole store must paginate, or it
will confidently report on a subset. See `fetch_all()` in `audit_ids.py`.
(DynamoDB has the same trap in a different shape: `Query` and `Scan` stop at
1MB per page and signal it only via `LastEvaluatedKey`.)
