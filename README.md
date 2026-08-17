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
events. Vectors in Chroma Cloud, embeddings from Voyage `voyage-3.5` over REST.
Auth is Cognito OAuth 2.1, tokens verified locally against Cognito's JWKS.
Credentials in Secrets Manager. CI is GitHub Actions over OIDC.

## Running your own

There is no hosted instance — this is bring-your-own-everything. You need an AWS
account, a Chroma Cloud account, and a Voyage API key.

### 1. Secrets

Create a Secrets Manager secret (default id `context-mcp/credentials`) with
**exactly** these four keys. `shared/config.py` raises at import if any is
missing:

```
VOYAGE_API_KEY
CHROMA_TENANT
CHROMA_DATABASE
CHROMA_API_KEY
```

### 2. Non-secret environment

Six variables on the Lambda:

```
CONTEXT_MCP_SECRET_ID        # the secret id above
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
python -m shared.store                   # smoke suite, offline, no API keys
```

`requirements.txt` pulls full `chromadb`; the Lambda bundle uses
`requirements-lambda.txt` with the thin `chromadb-client`, which exposes
`PersistentClient` but raises "http-only client mode" if constructed. The bundle
has a hard 250MB unzipped limit and currently sits at ~108MB.

## Known gap: `/map` is not reproducible from infra

The `/map` visualization reads three further variables — `MAP_CLIENT_ID`,
`MAP_COGNITO_DOMAIN`, `MAP_CLIENT_SECRET` — and **nothing in `infra/setup.sh` or
`.github/` sets any of them**. They are configured out of band, so a clean deploy
brings up the MCP server correctly but leaves `/map` non-functional with no error
explaining why.

`MAP_CLIENT_SECRET` also breaks the project's own rule that only non-secret
config lives in the environment: it is a client secret sitting outside Secrets
Manager.

Both are known and unfixed. Flagged here rather than omitted, because the failure
is silent.

## Scripts

`scripts/` holds one-shot migrations and audits, each with a dry-run mode and a
written rationale at the top. `audit_ids.py` is the one worth knowing about: it
checks that every record's id agrees with its own metadata, and is safe to run
against a live store at any time.

Note that Chroma Cloud caps a single `get()` at 300 records and returns at most
that many **silently** — any script walking the whole store must paginate, or it
will confidently report on a subset. See `fetch_all()` in `audit_ids.py`.
