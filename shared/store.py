"""
Thin wrapper around ChromaDB implementing the schema in docs/schema.md.
Single collection, metadata-filtered queries. This is the module both
the MCP server and the backfill script will import — logic lives here once.
"""
from __future__ import annotations
import chromadb
import hashlib
import httpx
import numpy as np
import os
import time
from datetime import datetime, timezone
from typing import Optional

COLLECTION_NAME = "context_store"

# Locked taxonomy from docs/schema.md. Extending this is a one-line change
# here (not a data migration) — but it must be deliberate, not a typo that
# silently creates an invisible bucket search_context can never filter to.
VALID_CATEGORIES = {
    "tech_stack",
    "architecture",
    "config",
    "decisions",
    "preference",
    "fact",
    "goal",
    "note",
}
VALID_TYPES = {"summary", "chunk"}
VALID_TIERS = {"client", "personal"}

# Retrieval budget. Keeps a single search_context call from dumping
# unbounded tokens into the conversation.
DEFAULT_TOP_K = 5
MAX_TOP_K = 10
MAX_DOC_CHARS = 800

# How close a typo must be to auto-correct (0-1, difflib ratio).
CATEGORY_MATCH_CUTOFF = 0.75

# Marks a summary that has been replaced. Archived copies keep the full text but
# are excluded from search by default — otherwise every superseded fact stays
# semantically searchable forever and "what do we use for vectors?" starts
# returning last month's answer alongside this month's.
SUPERSEDED_SOURCE = "superseded"

# A replacement shorter than this fraction of the current summary is refused.
# Catches the characteristic failure of overwrite-in-place: a caller that read a
# four-line summary, noticed one fact changed, and sends back only that fact —
# silently destroying the rest. Overridable, because deliberately condensing a
# bloated summary is legitimate.
SHRINK_GUARD_RATIO = 0.5


class SummaryShrinkRefused(Exception):
    """
    Raised instead of performing a suspiciously destructive summary overwrite.
    Carries the current content so the caller can merge rather than guess.
    """

    def __init__(self, summary_id: str, previous: str, proposed: str):
        self.summary_id = summary_id
        self.previous = previous
        self.proposed = proposed
        super().__init__(
            f"Refusing to replace '{summary_id}': the new content is "
            f"{len(proposed)} chars against {len(previous)} currently stored "
            f"(under {int(SHRINK_GUARD_RATIO * 100)}%). A summary write REPLACES "
            "the whole slot — send the complete new state, not just what changed. "
            "Pass allow_shrink=True if the summary is genuinely being condensed."
        )

DEFAULT_VOYAGE_MODEL = "voyage-3.5"


def _normalize_category(category: str) -> tuple[str, Optional[str]]:
    """
    Returns (normalized_category, corrected_from). corrected_from is None if
    the input was already valid, otherwise it's the original (wrong) input —
    so callers can surface that a correction happened rather than silently
    swallowing it. Raises only when nothing is a close enough match, since
    guessing at that distance risks filing something under the wrong
    category with no way to tell.
    """
    if category in VALID_CATEGORIES:
        return category, None
    import difflib

    matches = difflib.get_close_matches(
        category, VALID_CATEGORIES, n=1, cutoff=CATEGORY_MATCH_CUTOFF
    )
    if matches:
        return matches[0], category
    raise ValueError(
        f"Unknown category '{category}' — no close match found. "
        f"Valid: {sorted(VALID_CATEGORIES)}"
    )


VOYAGE_EMBEDDINGS_URL = "https://api.voyageai.com/v1/embeddings"


class VoyageRestEmbedding:
    """
    Real embedding function for actual use (decision: Voyage, not local).

    Deliberately calls Voyage's REST endpoint directly instead of using the
    `voyageai` SDK. Same endpoint, same request, same vectors — but the SDK
    declares pillow, tokenizers, hf_xet and aiohttp for multimodal and local
    tokenizer features this project never touches, ~90MB of dependencies. That
    matters because the deploy target is Lambda, where the whole bundle has to
    fit in 250MB unzipped; dropping the SDK took it from 200MB to 110MB.

    Chroma's embedding-function protocol is duck-typed — a callable taking a
    list of texts and returning a list of vectors, plus name() — so this drops
    straight into the (embedding_function, embedding_function_name) seam.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_VOYAGE_MODEL,
        timeout: float = 30.0,
        max_retries: int = 4,
    ):
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        # One client, reused: on Lambda this lives at module scope across warm
        # invocations, so a per-call TLS handshake would be pure waste. It also
        # opens no socket at construction, which is what lets SnapStart
        # snapshot this object without freezing a dead connection into the
        # image (no after_restore hook needed).
        self._client = httpx.Client(timeout=timeout)

    def name(self) -> str:
        return self.model

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        # A 429 raises straight out of collection.upsert()/query(), so without
        # this a rate limit surfaces as a failed write rather than a slow one.
        # Backoff starts at the length of a rate-limit window, not 1-2s —
        # short retries just re-hit the same closed window. Note Voyage
        # throttles an account with no payment method to 3 RPM regardless of
        # tier; with one, voyage-3.5 allows 2000 RPM.
        delay = 25.0
        for attempt in range(self.max_retries):
            resp = self._client.post(
                VOYAGE_EMBEDDINGS_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": list(input)},
            )
            if resp.status_code != 429 or attempt == self.max_retries - 1:
                break
            time.sleep(float(resp.headers.get("retry-after", delay)))
            delay = min(delay * 2, 60.0)
        resp.raise_for_status()
        data = resp.json()["data"]
        # Sort defensively rather than trusting response order — a silent
        # misalignment here attaches the wrong vector to the wrong document
        # and is near-impossible to debug afterwards.
        ordered = [item["embedding"] for item in sorted(data, key=lambda d: d["index"])]
        # MUST be numpy, not plain lists. Chroma's HTTP/Cloud client calls
        # .tolist() on query embeddings (api/fastapi.py::_query ->
        # convert_np_embeddings_to_list); plain lists raise AttributeError
        # there. Writes tolerate lists and the local PersistentClient never
        # takes that path, so this breaks ONLY on reads against Cloud.
        return [np.array(vec, dtype=np.float32) for vec in ordered]

    def embed_documents(self, input):
        return self(input)

    def embed_query(self, input):
        return self(input)


def _make_client(persist_path: str, use_cloud: Optional[bool] = None):
    """
    Chroma Cloud when the CHROMA_* trio is configured (the deployed path —
    Lambda has no persistent disk to put a database on), local PersistentClient
    otherwise (stdio dev, offline work).

    `use_cloud` overrides the env sniffing: False forces a local store even
    when Cloud credentials are present. Tests MUST pass False — otherwise a
    developer with a populated .env runs the smoke suite straight into the
    real shared store and writes fixture data into it.

    Partial configuration raises rather than quietly falling back to a local
    store: an incomplete Cloud config almost always means a deploy is
    misconfigured, and silently writing to a local disk instead would look
    like it worked while the entries went nowhere anyone else can read.

    NOTE: PersistentClient requires the full `chromadb` package. The Lambda
    bundle installs `chromadb-client`, which exposes the name but raises
    "http-only client mode" on construction — so the deployed path must always
    resolve to Cloud. See requirements.txt vs requirements-lambda.txt.
    """
    tenant = os.environ.get("CHROMA_TENANT")
    database = os.environ.get("CHROMA_DATABASE")
    api_key = os.environ.get("CHROMA_API_KEY")
    provided = [n for n, v in
                (("CHROMA_TENANT", tenant), ("CHROMA_DATABASE", database),
                 ("CHROMA_API_KEY", api_key)) if v]

    if use_cloud is False:
        return chromadb.PersistentClient(path=persist_path)
    if provided and len(provided) < 3:
        missing = {"CHROMA_TENANT", "CHROMA_DATABASE", "CHROMA_API_KEY"} - set(provided)
        raise RuntimeError(
            f"Partial Chroma Cloud config: {', '.join(sorted(provided))} set but "
            f"{', '.join(sorted(missing))} missing. Set all three to use Cloud, "
            "or none to use a local store."
        )
    if use_cloud and not provided:
        raise RuntimeError(
            "use_cloud=True but no CHROMA_TENANT / CHROMA_DATABASE / CHROMA_API_KEY "
            "are set."
        )
    if provided:
        return chromadb.CloudClient(tenant=tenant, database=database, api_key=api_key)
    return chromadb.PersistentClient(path=persist_path)


class ContextStore:
    def __init__(
        self,
        persist_path: str = "./chroma_data",
        embedding_function=None,
        embedding_function_name: Optional[str] = None,
        allow_local_fallback: bool = False,
        use_cloud: Optional[bool] = None,
    ):
        """
        Voyage is REQUIRED by default (see docs/schema.md decision) — this
        raises loudly rather than silently falling back to Chroma's weaker
        local default, so the "Voyage because retrieval quality matters"
        decision can't be undone by someone forgetting a kwarg.

        - Normal use: set VOYAGE_API_KEY in the environment, call with no args.
        - Explicit override (e.g. offline dev/testing): pass both
          embedding_function AND embedding_function_name.
        - Explicit, deliberate opt-out of Voyage: allow_local_fallback=True.
        """
        if embedding_function is None:
            api_key = os.environ.get("VOYAGE_API_KEY")
            if api_key:
                embedding_function = VoyageRestEmbedding(api_key)
                embedding_function_name = DEFAULT_VOYAGE_MODEL
            elif allow_local_fallback:
                embedding_function = None  # Chroma's built-in local default
                embedding_function_name = embedding_function_name or "local-default"
            else:
                raise RuntimeError(
                    "No VOYAGE_API_KEY set and no embedding_function provided. "
                    "This project deliberately chose Voyage for retrieval quality "
                    "(see docs/schema.md) — silently falling back to the weaker "
                    "local model would undo that decision. Set VOYAGE_API_KEY, "
                    "pass an explicit embedding_function + embedding_function_name, "
                    "or pass allow_local_fallback=True if you're intentionally "
                    "opting out."
                )
        elif embedding_function_name is None:
            raise ValueError(
                "embedding_function_name is required when passing a custom "
                "embedding_function, so the collection can record what created "
                "it and detect a mismatch if reopened differently later."
            )

        self.client = _make_client(persist_path, use_cloud)
        self.collection = self._open_or_create_collection(
            embedding_function, embedding_function_name
        )

    def _open_or_create_collection(self, embedding_function, embedding_function_name):
        existing_names = [c.name for c in self.client.list_collections()]

        if COLLECTION_NAME in existing_names:
            get_kwargs = {}
            if embedding_function is not None:
                get_kwargs["embedding_function"] = embedding_function
            collection = self.client.get_collection(COLLECTION_NAME, **get_kwargs)
            existing_tag = (collection.metadata or {}).get("embedding_function_name")
            if existing_tag and existing_tag != embedding_function_name:
                raise RuntimeError(
                    f"This collection was created with embedding function "
                    f"'{existing_tag}', but you're opening it with "
                    f"'{embedding_function_name}'. Mixing embedding spaces "
                    "silently corrupts similarity search — either match the "
                    "original embedding function or use a fresh persist_path."
                )
            return collection
        else:
            create_kwargs = {"metadata": {"embedding_function_name": embedding_function_name}}
            if embedding_function is not None:
                create_kwargs["embedding_function"] = embedding_function
            return self.client.get_or_create_collection(COLLECTION_NAME, **create_kwargs)

    def save(
        self,
        document: str,
        category: str,
        type: str,  # "summary" | "chunk"
        project: Optional[str] = None,  # None -> "general"
        tier: Optional[str] = None,  # required if project set
        source: str = "live",
        chat_title: Optional[str] = None,
    ):
        category, corrected_from = _normalize_category(category)
        if type not in VALID_TYPES:
            raise ValueError(f"type must be one of {VALID_TYPES}, got '{type}'")

        project_key = project or "general"
        if not project:
            tier = None  # tier is meaningless without a project — don't let it leak in
        elif tier and tier not in VALID_TIERS:
            raise ValueError(f"tier must be one of {VALID_TIERS}, got '{tier}'")
        elif not tier:
            raise ValueError("tier is required when project is set")

        # Upsert semantics: summaries are ONE living document per
        # project+category — deterministic id, always overwritten. Chunks
        # are raw fragments — always a new id, always inserted, never
        # overwritten.
        if type == "summary":
            id = f"{project_key}-{category}-summary"
        else:
            raw = f"{project_key}-{category}-{document}-{datetime.now().timestamp()}"
            id = f"chunk-{hashlib.sha1(raw.encode()).hexdigest()[:16]}"

        metadata = {
            "project": project_key,
            "category": category,
            "type": type,
            "source": source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if tier:
            metadata["tier"] = tier
        if chat_title:
            metadata["chat_title"] = chat_title
        if corrected_from:
            metadata["category_corrected_from"] = corrected_from

        self.collection.upsert(ids=[id], documents=[document], metadatas=[metadata])
        return {"id": id, "corrected_from": corrected_from, **metadata}

    def summary_id(self, project: Optional[str], category: str) -> str:
        """The deterministic slot id — one living document per project+category."""
        return f"{project or 'general'}-{category}-summary"

    def get_summary(self, project: Optional[str], category: str, with_embedding: bool = False):
        """
        Current content of one summary slot, or None if empty.
        Returns (document, metadata, embedding) so callers that are about to
        overwrite can archive the old version without re-embedding it.
        """
        category, _ = _normalize_category(category)
        include = ["documents", "metadatas"] + (["embeddings"] if with_embedding else [])
        got = self.collection.get(ids=[self.summary_id(project, category)], include=include)
        if not got["ids"]:
            return None
        emb = got.get("embeddings")
        return (
            got["documents"][0],
            got["metadatas"][0],
            emb[0] if with_embedding and emb is not None and len(emb) else None,
        )

    def get_brief(self, project: Optional[str] = None) -> list[dict]:
        """
        Every living summary for a project, returned WHOLE — no similarity
        ranking, no top_k, no truncation.

        This exists because search() is the wrong instrument for "what is the
        current state of X". Ranked search returns the first ~800 chars of the
        five best-matching documents, which for long entries means most of the
        content is unreachable no matter how the query is phrased. A brief is a
        deterministic id lookup, so it can hand back the complete text.
        """
        project_key = project or "general"
        got = self.collection.get(
            where={"$and": [{"project": project_key}, {"type": "summary"}]},
            include=["documents", "metadatas"],
        )
        entries = [
            {
                "category": meta.get("category"),
                "content": doc,
                "tier": meta.get("tier"),
                "source": meta.get("source"),
                "timestamp": meta.get("timestamp"),
            }
            for doc, meta in zip(got["documents"], got["metadatas"])
        ]
        # Stable, readable order rather than whatever the store returns.
        entries.sort(key=lambda e: e["category"] or "")
        return entries

    def update_summary(
        self,
        document: str,
        category: str,
        project: Optional[str] = None,
        tier: Optional[str] = None,
        source: str = "live",
        chat_title: Optional[str] = None,
        allow_shrink: bool = False,
    ):
        """
        Replace one summary slot in place, archiving whatever was there.

        This is the "whiteboard" write, as opposed to save(type="chunk") which
        is the "diary". Three safeguards, because replacing is destructive in a
        way appending never is:
          1. The previous content is returned, so a truncating write is visible
             immediately rather than discovered later.
          2. A replacement far shorter than what it replaces is refused
             (SHRINK_GUARD_RATIO).
          3. The old version is archived as a chunk before being overwritten, so
             nothing is ever actually lost — worst case the summary reads wrong
             and the previous version is one query away.
        """
        category, corrected_from = _normalize_category(category)
        project_key = project or "general"
        if not project:
            tier = None  # tier is meaningless without a project
        elif tier and tier not in VALID_TIERS:
            raise ValueError(f"tier must be one of {VALID_TIERS}, got '{tier}'")
        elif not tier:
            raise ValueError("tier is required when project is set")

        sid = self.summary_id(project, category)
        current = self.get_summary(project, category, with_embedding=True)

        archived_id = None
        previous_doc = None
        if current is not None:
            previous_doc, previous_meta, previous_emb = current

            if not allow_shrink and len(document) < len(previous_doc) * SHRINK_GUARD_RATIO:
                raise SummaryShrinkRefused(sid, previous_doc, document)

            # Deterministic id from the content being archived, so re-running the
            # same replacement doesn't pile up duplicate copies of one old value.
            digest = hashlib.sha1(f"{sid}-{previous_doc}".encode()).hexdigest()[:16]
            archived_id = f"superseded-{digest}"
            archive_meta = {
                **previous_meta,
                "type": "chunk",
                "source": SUPERSEDED_SOURCE,
                "superseded_from": sid,
                "superseded_at": datetime.now(timezone.utc).isoformat(),
            }
            archive_kwargs = {
                "ids": [archived_id],
                "documents": [previous_doc],
                "metadatas": [archive_meta],
            }
            # Reuse the vector Chroma already holds. Archiving therefore costs no
            # embedding call at all — the text is unchanged, so its embedding is
            # still exactly right.
            if previous_emb is not None:
                archive_kwargs["embeddings"] = [previous_emb]
            self.collection.upsert(**archive_kwargs)

        metadata = {
            "project": project_key,
            "category": category,
            "type": "summary",
            "source": source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if tier:
            metadata["tier"] = tier
        if chat_title:
            metadata["chat_title"] = chat_title
        if corrected_from:
            metadata["category_corrected_from"] = corrected_from

        self.collection.upsert(ids=[sid], documents=[document], metadatas=[metadata])
        return {
            "id": sid,
            "previous": previous_doc,
            "archived_id": archived_id,
            "corrected_from": corrected_from,
            **metadata,
        }

    def search(
        self,
        query: str,
        project: Optional[str] = None,  # None = search across all (incl. general)
        category: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
        include_superseded: bool = False,
    ):
        corrected_from = None
        if category is not None:
            category, corrected_from = _normalize_category(category)
        top_k = min(top_k, MAX_TOP_K)

        where_clauses = []
        if project:
            where_clauses.append({"project": project})
        if category:
            where_clauses.append({"category": category})
        if not include_superseded:
            # Without this, archived summaries compete with live ones and stale
            # facts resurface as if current. History stays retrievable, but only
            # when a caller explicitly asks for it.
            where_clauses.append({"source": {"$ne": SUPERSEDED_SOURCE}})

        where = None
        if len(where_clauses) == 1:
            where = where_clauses[0]
        elif len(where_clauses) > 1:
            where = {"$and": where_clauses}

        results = self.collection.query(
            query_texts=[query], n_results=top_k, where=where
        )

        # Truncate long documents so one call can't dump unbounded tokens
        # into the conversation. Full content should live in a proper
        # summary anyway if it's meant to be retrieved whole.
        docs = results.get("documents", [[]])[0]
        truncated_docs = []
        for d in docs:
            if len(d) > MAX_DOC_CHARS:
                truncated_docs.append(d[:MAX_DOC_CHARS] + " …[truncated]")
            else:
                truncated_docs.append(d)
        results["documents"] = [truncated_docs]
        # Visible signal, not silent — caller/tool-response can surface this
        # ("interpreted category 'desicions' as 'decisions'") instead of the
        # correction happening invisibly.
        results["category_corrected_from"] = corrected_from
        return results


class _OfflineHashEmbedding:
    """
    Deterministic, dependency-free embedding function used ONLY because this
    sandbox can't reach the CDN that hosts ChromaDB's default onnx model, and
    has no Voyage API key configured. It's a crude bag-of-words hash into a
    fixed-size vector — good enough to prove filtering + retrieval logic
    works, NOT something to actually use. On your machine, use
    VoyageRestEmbedding with a real API key instead.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def name(self) -> str:
        return "offline-hash-stub"

    def __call__(self, input):
        vectors = []
        for text in input:
            vec = np.zeros(self.dim, dtype=np.float32)
            for word in text.lower().split():
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                vec[h % self.dim] += 1.0
            # numpy, not a list, for the same reason as VoyageRestEmbedding —
            # otherwise this stub works locally and breaks the moment it's
            # pointed at a Cloud collection.
            vectors.append(vec)
        return vectors

    def embed_documents(self, input):
        return self(input)

    def embed_query(self, input):
        return self(input)


if __name__ == "__main__":
    import shutil

    TEST_PATH = "./chroma_data_smoketest"
    shutil.rmtree(TEST_PATH, ignore_errors=True)

    # Smoke test: write tagged entries, prove upsert semantics (writing the
    # same summary twice should REPLACE it, not duplicate it), prove
    # category validation auto-corrects typos visibly, prove project-filtered
    # retrieval still respects boundaries, prove result truncation kicks in,
    # and prove the two embedding-function safety paths actually work
    # (mismatch guard raises; local-fallback reopen doesn't crash).
    # use_cloud=False on every construction below: without it, a developer with
    # Chroma Cloud credentials in .env would run this suite against the real
    # shared store and write fixture data into it.
    store = ContextStore(
        persist_path=TEST_PATH,
        embedding_function=_OfflineHashEmbedding(),
        embedding_function_name="offline-hash-stub",
        use_cloud=False,
    )

    store.save(
        document="The ticketing SaaS uses Postgres with a Redis cache. Backend is FastAPI, frontend is React.",
        category="architecture",
        type="summary",
        project="ticketing-saas",
        tier="client",
        source="backfill",
    )
    # Save the SAME summary again with different content — should overwrite,
    # not create a second "living" summary.
    result = store.save(
        document="UPDATED: The ticketing SaaS uses Postgres with a Redis cache and now also runs a Kafka queue for async jobs.",
        category="architecture",
        type="summary",
        project="ticketing-saas",
        tier="client",
        source="live",
    )
    print("=== Upsert test: same id both times? ===")
    print(result["id"])

    count = store.collection.count()
    print(f"Total docs in collection after two summary writes to the same slot: {count}")
    assert count == 1, f"expected 1 doc, got {count}"

    print("\n=== Typo correction test: 'desicions' should auto-correct to 'decisions', visibly ===")
    result = store.save(
        document="Decided to use Kafka for async jobs.",
        category="desicions",  # typo, on purpose
        type="summary",
        project="ticketing-saas",
        tier="client",
    )
    print(f"Saved under corrected category: '{result['category']}', corrected_from='{result['corrected_from']}'")
    assert result["category"] == "decisions"
    assert result["corrected_from"] == "desicions"

    r = store.search("kafka decision", project="ticketing-saas", category="desicions")
    print(f"Searching with the SAME typo still finds it: {r['documents']}")
    print(f"Search also surfaces the correction: category_corrected_from={r['category_corrected_from']}")
    assert r["category_corrected_from"] == "desicions"

    print("\n=== Nonsense category test: should still fail, no reasonable match ===")
    try:
        store.save(
            document="test",
            category="xyzabc123",
            type="summary",
            project="ticketing-saas",
            tier="client",
        )
        print("FAIL: nonsense category was accepted")
    except ValueError as e:
        print(f"OK, rejected: {e}")

    print("\n=== Retrieval reflects the UPDATED summary, not the original ===")
    r = store.search("what does the architecture look like", project="ticketing-saas")
    print(r["documents"])

    print("\n=== Truncation test ===")
    long_text = "This is a very long chunk. " * 60  # > 800 chars
    store.save(document=long_text, category="note", type="chunk", project=None, source="live")
    r = store.search("very long chunk", category="note")
    doc = r["documents"][0][0]
    print(f"Returned doc length: {len(doc)} (should be <= ~815 incl. truncation marker)")
    print(doc[-40:])
    assert len(doc) <= 820

    print("\n=== tier-stripped-for-general test ===")
    result = store.save(
        document="I prefer terse responses.",
        category="preference",
        type="summary",
        project=None,
        tier="personal",  # should be silently stripped, not stored
    )
    assert "tier" not in result, f"tier leaked into a general entry: {result}"
    print("OK, tier not present on general entry")

    print("\n=== Embedding-function mismatch guard test ===")
    try:
        ContextStore(
            persist_path=TEST_PATH,
            embedding_function=_OfflineHashEmbedding(),
            embedding_function_name="some-other-function",
            use_cloud=False,
        )
        print("FAIL: reopening with a different embedding_function_name did not raise")
    except RuntimeError as e:
        print(f"OK, rejected: {e}")

    print("\n=== Local-fallback reopen test (allow_local_fallback=True, no explicit function) ===")
    FALLBACK_PATH = "./chroma_data_smoketest_fallback"
    shutil.rmtree(FALLBACK_PATH, ignore_errors=True)
    fallback_store = ContextStore(
        persist_path=FALLBACK_PATH, allow_local_fallback=True, use_cloud=False
    )
    fallback_store.save(
        document="Testing the local fallback embedding path.",
        category="note",
        type="chunk",
        project=None,
        source="live",
    )
    # Reopen the SAME path with allow_local_fallback again — exercises the
    # get_collection(embedding_function=None) branch, not just creation.
    fallback_reopen = ContextStore(
        persist_path=FALLBACK_PATH, allow_local_fallback=True, use_cloud=False
    )
    r = fallback_reopen.search("local fallback embedding")
    print(f"Local-fallback reopen query returned: {r['documents']}")
    assert len(r["documents"][0]) >= 1, "local-fallback reopen returned no results"
    print("OK, local-fallback reopen works")

    shutil.rmtree(TEST_PATH, ignore_errors=True)
    shutil.rmtree(FALLBACK_PATH, ignore_errors=True)
    print("\nAll smoke tests passed.")
